import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from coordinator_state import CoordinatorState
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess import tess_blind_transit_search as blind
from workflows.tess import tess_exhausted_residual_candidates as distributed
from workflows.tess import tess_investigation
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION


class ExhaustedResidualCandidateTests(unittest.TestCase):
    def _complete(self, store, investigation, stage_id, handler_id, result):
        running = InvestigationStage(
            stage_id, handler_id, "RUNNING", None, {}
        )
        investigation = store.append_running_stage(investigation, running)
        terminal = store.build_terminal_stage(
            stage_id=stage_id,
            handler_id=handler_id,
            status="COMPLETE",
            triggered_by_stage_id=None,
            parameters={},
            result=result,
            error=None,
            software_id="test",
            software_version="1",
            started_at=running.started_at,
        )
        return store.complete_current_stage(investigation, terminal)

    def _blind_result(self, *, family_rank=29):
        family = {
            "objectiveRank": family_rank,
            "objectiveScore": 5.0,
            "coarseFrequencyPerDay": 0.3,
            "coarsePeriodDays": 1.0 / 0.3,
        }
        return {
            "classification": "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
            "recommendedNextTest": (
                "GENERIC_DISTRIBUTED_RESIDUAL_TRANSIT_CANDIDATE_GENERATION"
            ),
            "independentSectorAvailability": {
                "allCandidateSectorsPrepared": True,
            },
            "iterativeSearch": {
                "terminationReason": "NEXT_RESIDUAL_SIGNAL_UNRESOLVED",
            },
            "candidateSignals": [{
                "candidateIndex": 1,
                "candidatePeriodDays": 2.0,
                "linearEphemeris": {
                    "refinedPeriodDays": 2.0,
                    "referenceEpoch": 1000.5,
                },
                "jointTransitSearch": {
                    "frequencyPerDay": 0.5,
                    "durationDays": 0.08,
                },
                "sectorResults": [],
            }],
            "exhaustedSectorResidualFamilyCensus": {
                "methods": [
                    {
                        "residualSearchMethod": method,
                        "candidateGenerationAudit": {
                            "recordedFamilyCount": 64,
                            "families": [dict(family)],
                        },
                    }
                    for method in distributed.RESIDUAL_METHODS
                ],
            },
        }

    def _dataset(self, root, sector, origin):
        times = [index * 0.01 for index in range(1201)]
        flux = []
        for relative in times:
            absolute = origin + relative
            value = 0.01 * math.sin(2.0 * math.pi * absolute / 3.3)
            distance = abs(
                (absolute - 1000.5 + 1.0) % 2.0 - 1.0
            )
            if distance <= 0.04:
                value -= 0.05
            flux.append(value)
        path = root / f"sector-{sector}.json"
        path.write_text(json.dumps({
            "id": f"sector-{sector}",
            "source": {
                "sector": sector,
                "originalTimeOriginDays": origin,
                "baselineDays": times[-1],
            },
            "frequencySearch": {
                "minimumFrequency": 0.1,
                "maximumFrequency": 5.0,
            },
            "times": times,
            "flux": flux,
        }), encoding="utf-8")
        return path

    def test_warranted_boundary_is_exact(self):
        result = self._blind_result()
        self.assertTrue(
            distributed.distributed_candidate_generation_warranted(result)
        )
        for mutation in (
            {"classification": "BLIND_TRANSIT_PERIOD_UNRESOLVED"},
            {"recommendedNextTest": "HUMAN_SCIENTIFIC_REVIEW"},
            {"candidateSignals": []},
        ):
            changed = {**result, **mutation}
            self.assertFalse(
                distributed.distributed_candidate_generation_warranted(changed)
            )
        changed = json.loads(json.dumps(result))
        changed["independentSectorAvailability"][
            "allCandidateSectorsPrepared"
        ] = False
        self.assertFalse(
            distributed.distributed_candidate_generation_warranted(changed)
        )

    def test_builder_emits_only_generic_lomb_scargle_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_project = root / "source-project.json"
            source_project.write_text(json.dumps({
                "id": "source",
                "workloadID": "openstar.lomb-scargle.v1",
            }), encoding="utf-8")
            primary = self._dataset(root, 3, 1000.0)
            independent_paths = [
                self._dataset(root, sector, origin)
                for sector, origin in ((4, 1100.0), (5, 1200.0))
            ]
            preparation = distributed.build_exhausted_residual_candidate_project(
                source_project_path=source_project,
                primary_dataset_path=primary,
                independent_spec={
                    "preparedSectors": [
                        {"sector": sector, "datasetPath": str(path)}
                        for sector, path in zip((4, 5), independent_paths)
                    ],
                },
                blind_transit_result=self._blind_result(),
                output_dir=root / "artifacts",
                investigation_id="test-investigation",
            )
            manifest = json.loads(
                Path(preparation["projectPath"]).read_text(encoding="utf-8")
            )
            datasets = [
                json.loads(
                    Path(item["datasetPath"]).read_text(encoding="utf-8")
                )
                for item in preparation["preparedDatasets"]
            ]
            coordinator_status = CoordinatorState(
                preparation["projectPath"]
            ).project_status()

        self.assertEqual("openstar.lomb-scargle.v1", manifest["workloadID"])
        self.assertEqual(2, len(manifest["datasets"]))
        self.assertEqual(128, preparation["totalWorkUnits"])
        self.assertEqual(128, coordinator_status["projectTotalWorkUnits"])
        self.assertEqual(2, len(coordinator_status["datasets"]))
        self.assertFalse(preparation["specializedTessWorkerLogic"])
        self.assertFalse(preparation["normalTopTwelveSelectionPathChanged"])
        self.assertFalse(preparation["scienceThresholdsChanged"])
        self.assertEqual(
            set(distributed.RESIDUAL_METHODS),
            {item["source"]["residualSearchMethod"] for item in datasets},
        )
        self.assertTrue(all(item["times"] for item in datasets))
        self.assertTrue(all(item["flux"] for item in datasets))

    def test_interpreter_is_fail_closed_and_returns_numerical_candidates_only(self):
        preparation = {
            "version": distributed.PREPARATION_VERSION,
            "projectID": "project",
            "workloadID": distributed.WORKLOAD_ID,
            "preparedDatasets": [
                {"datasetID": f"dataset-{index}",
                 "residualSearchMethod": method}
                for index, method in enumerate(distributed.RESIDUAL_METHODS)
            ],
        }
        status = {
            "projectID": "project",
            "workloadID": distributed.WORKLOAD_ID,
            "datasets": [
                {
                    "id": f"dataset-{index}",
                    "coverageComplete": True,
                    "failedWorkUnits": 0,
                    "periodStatus": "RELIABLE",
                    "independentCandidates": [
                        {"frequency": 0.3 + index * 0.01, "power": 0.2},
                    ],
                }
                for index in range(2)
            ],
        }
        result = distributed.interpret_exhausted_residual_candidate_project(
            preparation=preparation, project_status=status
        )
        self.assertFalse(result["candidateSelectionPerformedByWorkers"])
        self.assertFalse(result["claimDecisionPerformedByWorkers"])
        self.assertEqual(
            [0.3],
            [item["frequency"] for item in result["candidateMap"][
                distributed.RESIDUAL_METHODS[0]
            ]],
        )

        incomplete = json.loads(json.dumps(status))
        incomplete["datasets"][0]["coverageComplete"] = False
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            distributed.interpret_exhausted_residual_candidate_project(
                preparation=preparation, project_status=incomplete
            )
        wrong_workload = {**status, "workloadID": "specialized.tess.work"}
        with self.assertRaisesRegex(ValueError, "unexpected workload"):
            distributed.interpret_exhausted_residual_candidate_project(
                preparation=preparation, project_status=wrong_workload
            )

    def test_server_validates_only_dual_method_families_below_rank_twelve(self):
        import numpy as np

        primary = {"id": "primary", "frequencySearch": {
            "minimumFrequency": 0.1, "maximumFrequency": 5.0,
        }}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "primary.json"
            path.write_text(json.dumps(primary), encoding="utf-8")
            sectors = [{
                "times": np.asarray([0.0, 10.0]),
                "residual": np.asarray([0.0, 0.0]),
                "sigma": 1.0,
            }]
            evaluated = {
                "rawPeriodDays": 1.0 / 0.3,
                "frequencyPerDay": 0.3,
                "periodDays": 1.0 / 0.3,
                "objectiveScore": 9.0,
                "jointTransitSearch": {},
                "alternatingCycleAliasResolution": {
                    "decision": "RETAIN_BASE_PERIOD",
                },
                "sectorResults": [],
                "primarySectorSupported": True,
                "independentSupporters": [
                    {"sector": sector} for sector in (4, 5, 30)
                ],
                "linearEphemeris": {
                    "coherent": True,
                    "refinedPeriodDays": 1.0 / 0.3,
                },
                "supported": True,
                "recurrenceSupportGate": {"mode": "UNCHANGED_TEST_GATE"},
            }
            candidates = {
                method: [{"frequency": 0.3, "power": 0.2}]
                for method in distributed.RESIDUAL_METHODS
            }
            with mock.patch.object(
                blind, "prepare_exhausted_residual_sectors",
                return_value=sectors,
            ), mock.patch.object(
                blind, "_explicit_family_hypothesis",
                return_value=((0.3, 9.0, []), {
                    "fineFrequencyStepPerDay": 0.0001,
                }),
            ), mock.patch.object(
                blind, "_evaluate_frequency_hypothesis",
                return_value=evaluated,
            ), mock.patch.object(
                blind, "_distinct_frequency_family",
                return_value=([], {"distinct": True}),
            ):
                accepted = blind.analyze_exhausted_distributed_residual_candidates(
                    primary_dataset_path=path,
                    independent_spec={
                        "preparedSectors": [{"sector": index} for index in range(6)]
                    },
                    blind_transit_result=self._blind_result(family_rank=29),
                    distributed_candidates=candidates,
                )
                one_method = dict(candidates)
                one_method[distributed.RESIDUAL_METHODS[1]] = []
                rejected = blind.analyze_exhausted_distributed_residual_candidates(
                    primary_dataset_path=path,
                    independent_spec={
                        "preparedSectors": [{"sector": index} for index in range(6)]
                    },
                    blind_transit_result=self._blind_result(family_rank=29),
                    distributed_candidates=one_method,
                )
                top_twelve = blind.analyze_exhausted_distributed_residual_candidates(
                    primary_dataset_path=path,
                    independent_spec={
                        "preparedSectors": [{"sector": index} for index in range(6)]
                    },
                    blind_transit_result=self._blind_result(family_rank=12),
                    distributed_candidates=candidates,
                )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(2, accepted["candidateSignal"]["candidateIndex"])
        self.assertFalse(accepted["scienceThresholdsChanged"])
        self.assertFalse(accepted["normalTopTwelveSelectionPathChanged"])
        self.assertFalse(rejected["accepted"])
        self.assertEqual(0, rejected["corroboratedFamilyGroupCount"])
        self.assertFalse(top_twelve["accepted"])
        self.assertEqual(0, top_twelve["corroboratedFamilyGroupCount"])

    def test_fixed_fourier_family_can_map_a_narrow_transit_harmonic(self):
        match = blind._distributed_family_match(
            {"coarseFrequencyPerDay": 0.3},
            [{"frequency": 1.5, "power": 0.25}],
            coarse_step=0.001,
            minimum=0.1,
            maximum=5.0,
        )
        self.assertIsNotNone(match)
        self.assertAlmostEqual(0.2, match["harmonicMultiplier"])
        self.assertAlmostEqual(0.3, match["projectedBoxFrequencyPerDay"])

    def test_lifecycle_runs_generic_project_then_returns_to_server_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InvestigationStore(root / "investigations")
            investigation = store.create(
                "distributed-residual-lifecycle", WORKFLOW_ID, WORKFLOW_VERSION
            )
            preparation = {
                "projectPath": str(root / "project.json"),
                "primaryDatasetPath": str(root / "primary.json"),
                "independentSpec": {"preparedSectors": []},
            }
            blind_result = {
                **self._blind_result(),
                "claimDecision": {
                    "claim": "CANDIDATE_PERIOD",
                    "rationale": ["first clock retained"],
                },
                "distributedResidualCandidateGenerationPreparation": preparation,
            }
            investigation = self._complete(
                store,
                investigation,
                "010-blind",
                blind.HANDLER_ID,
                blind_result,
            )
            status = {
                "projectID": "generic-project",
                "workloadID": distributed.WORKLOAD_ID,
                "datasets": [],
            }
            coordinator = types.SimpleNamespace(
                run_project=mock.Mock(return_value=types.SimpleNamespace(
                    status=status,
                    node_contributions={"generic-node": 128},
                    project_id="generic-project",
                ))
            )
            engine = tess_investigation.build_engine(
                store, coordinator, poll_interval=0, timeout=1
            )
            engine.chain_stages = False
            completed, next_request = engine.run_stage(
                investigation,
                StageRequest(
                    "011-run",
                    tess_investigation.EXHAUSTED_RESIDUAL_RUN_HANDLER_ID,
                    {"projectPath": preparation["projectPath"]},
                    "010-blind",
                ),
                software_id="test",
                software_version="1",
            )
            self.assertEqual(
                tess_investigation.EXHAUSTED_RESIDUAL_INTERPRET_HANDLER_ID,
                next_request.handler_id,
            )
            candidate = {
                "candidateIndex": 2,
                "candidatePeriodDays": 3.36,
                "supportingIndependentSectors": [4, 5, 30],
                "linearEphemeris": {"coherent": True},
            }
            generic = {
                "workerSemantics": "GENERIC_LOMB_SCARGLE",
                "candidateMap": {method: [] for method in distributed.RESIDUAL_METHODS},
            }
            validation = {
                "accepted": True,
                "candidateSignal": candidate,
                "corroboratedFamilyGroupCount": 1,
            }
            with mock.patch.object(
                tess_investigation,
                "interpret_exhausted_residual_candidate_project",
                return_value=generic,
            ), mock.patch.object(
                tess_investigation,
                "analyze_exhausted_distributed_residual_candidates",
                return_value=validation,
            ):
                interpreted, finalize_request = engine.run_stage(
                    completed,
                    next_request,
                    software_id="test",
                    software_version="1",
                )

            latest = tess_investigation._latest_blind_transit_result(
                interpreted
            )

        coordinator.run_project.assert_called_once_with(
            preparation["projectPath"], poll_interval=0, timeout=1
        )
        self.assertEqual("openstar.tess.finalize", finalize_request.handler_id)
        self.assertEqual(2, len(latest["candidateSignals"]))
        self.assertEqual(3.36, latest["candidateSignals"][1]["candidatePeriodDays"])
        self.assertEqual("HUMAN_SCIENTIFIC_REVIEW", latest["recommendedNextTest"])
        self.assertFalse(
            latest["distributedResidualCandidateGeneration"][
                "scienceThresholdsChanged"
            ]
        )


if __name__ == "__main__":
    unittest.main()
