import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTarget
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION, plan_tess_branches

from workflows.tess.tess_hypotheses import (
    broad_independent_next_handler,
    interpret_broad_independent_sectors,
)

# The integration exercises handlers whose selected path is pure Python.  The
# full adapter also registers optional NumPy-backed archival handlers; keep
# those unavailable branches importable in the dependency-minimal server test
# environment without pretending to execute them.
if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.ModuleType("numpy")

from workflows.tess.tess_investigation import build_engine


class BroadIndependentCharacterizationTests(unittest.TestCase):
    def _complete(self, store, investigation, stage_id, handler_id, result):
        running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
        investigation = store.append_running_stage(investigation, running)
        terminal = store.build_terminal_stage(
            stage_id=stage_id,
            handler_id=handler_id,
            status="COMPLETE",
            triggered_by_stage_id=None,
            parameters={},
            result=result,
            error=None,
            software_id="integration-replay",
            software_version="20.28",
            started_at=running.started_at,
        )
        return store.complete_current_stage(investigation, terminal)

    def _write_light_curve(self, root, name, sector, phase_offset):
        path = root / f"{name}.json"
        times = [index * 0.01 for index in range(3001)]
        flux = [
            1.0
            + 0.02 * math.sin(2.0 * math.pi * time / 6.86 + phase_offset)
            + 0.008 * math.sin(2.0 * math.pi * time / 13.72)
            for time in times
        ]
        path.write_text(
            json.dumps(
                {
                    "id": name,
                    "targetName": "Synthetic recurrent source",
                    "source": {"sector": sector, "baselineDays": 30.0},
                    "times": times,
                    "flux": flux,
                }
            ),
            encoding="utf-8",
        )
        return path

    def _interpret(self, periods, prominences, reliable=None):
        reliable = reliable or [True] * len(periods)
        prepared = []
        datasets = []
        for index, (period, prominence, is_reliable) in enumerate(
            zip(periods, prominences, reliable), start=1
        ):
            dataset_id = f"sector-{index}"
            prepared.append(
                {"datasetID": dataset_id, "sector": index, "baselineDays": 27.0}
            )
            datasets.append(
                {
                    "datasetID": dataset_id,
                    "periodStatus": "RELIABLE" if is_reliable else "UNRELIABLE",
                    "periodConfidence": "high" if is_reliable else "low",
                    "candidatePeriodDays": period,
                    "candidateFrequency": 1.0 / period,
                    "candidatePeakProminenceRatio": prominence,
                }
            )
        return interpret_broad_independent_sectors(
            project_status={"datasets": datasets},
            broad_spec={
                "frequencySearch": {
                    "minimumFrequency": 0.01,
                    "maximumFrequency": 0.5,
                    "frequencyStep": 0.0001,
                },
                "preparedSectors": prepared,
            },
            primary_raw_period_days=7.0,
            primary_preferred_period_days=14.0,
            same_sector_candidate_days=13.8,
        )

    def test_stable_promoted_recurrence_finalizes_normally(self):
        evidence = self._interpret([6.98, 7.0, 7.02], [2.0, 2.1, 1.9])

        self.assertTrue(evidence["promotionEligible"])
        self.assertFalse(evidence["variabilityCharacterization"]["warranted"])
        self.assertEqual(
            "openstar.tess.finalize", broad_independent_next_handler(evidence)
        )

    def test_no_meaningful_recurrent_family_finalizes_conservatively(self):
        evidence = self._interpret(
            [4.0, 7.0, 11.0], [2.0, 2.0, 2.0], [True, False, False]
        )

        self.assertEqual("HUMAN_REVIEW_REQUIRED", evidence["claimDecision"]["claim"])
        self.assertIsNone(evidence["harmonicFamily"])
        self.assertEqual(
            "openstar.tess.finalize", broad_independent_next_handler(evidence)
        )

    def test_recurrent_inconsistent_family_routes_to_morphology(self):
        evidence = self._interpret([6.52, 7.20], [1.2, 1.3])

        characterization = evidence["variabilityCharacterization"]
        self.assertEqual(
            "RECURRENT_BUT_UNRESOLVED_CROSS_SECTOR_VARIABILITY",
            characterization["state"],
        )
        self.assertTrue(characterization["warranted"])
        self.assertIn("cluster-spread-too-wide", characterization["reasons"])
        self.assertEqual(
            "openstar.tess.morphology.analyze",
            broad_independent_next_handler(evidence),
        )

    def test_restart_selection_is_deterministic_from_persisted_evidence(self):
        evidence = self._interpret([6.52, 7.20], [1.2, 1.3])

        first = broad_independent_next_handler(evidence)
        second = broad_independent_next_handler(evidence)
        self.assertEqual(first, second)

    def test_broad_continuation_executes_existing_morphology_handler(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = InvestigationStore(root / "investigations")
        target = InvestigationTarget(
            id="synthetic:unresolved-family",
            investigation_id="synthetic-unresolved-family",
            workflow_id=WORKFLOW_ID,
            workflow_version=WORKFLOW_VERSION,
            priority=0,
            eligible=True,
            metadata={},
        )
        investigation = store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        primary_path = self._write_light_curve(root, "primary", 62, 0.0)
        sector_66_path = self._write_light_curve(root, "sector-66", 66, 0.2)
        sector_67_path = self._write_light_curve(root, "sector-67", 67, 0.5)
        independent = {
            "preparedSectors": [
                {
                    "datasetID": "sector-66",
                    "sector": 66,
                    "baselineDays": 30.0,
                    "datasetPath": str(sector_66_path),
                },
                {
                    "datasetID": "sector-67",
                    "sector": 67,
                    "baselineDays": 30.0,
                    "datasetPath": str(sector_67_path),
                },
            ]
        }
        broad_spec = {
            **independent,
            "frequencySearch": {
                "minimumFrequency": 0.01,
                "maximumFrequency": 0.5,
                "frequencyStep": 0.0001,
            },
        }
        broad_run = {
            "datasets": [
                {
                    "datasetID": "sector-66",
                    "periodStatus": "RELIABLE",
                    "periodConfidence": "high",
                    "candidatePeriodDays": 6.52,
                    "candidateFrequency": 1.0 / 6.52,
                    "candidatePeakProminenceRatio": 1.2,
                },
                {
                    "datasetID": "sector-67",
                    "periodStatus": "RELIABLE",
                    "periodConfidence": "high",
                    "candidatePeriodDays": 7.20,
                    "candidateFrequency": 1.0 / 7.20,
                    "candidatePeakProminenceRatio": 1.3,
                },
            ]
        }
        replay = (
            (
                "001-prepare-target",
                "openstar.tess.prepare-target",
                {"datasetPath": str(primary_path)},
            ),
            (
                "002-primary-distributed-search",
                "openstar.tess.primary-project.run",
                {"datasets": []},
            ),
            (
                "004-hypotheses",
                "openstar.tess.hypotheses",
                {"rawCandidatePeriodDays": 7.0, "observedPeriodDays": 14.0},
            ),
            (
                "006-prepare-independent",
                "openstar.tess.independent.prepare",
                independent,
            ),
            (
                "007-run-independent",
                "openstar.tess.independent.run",
                {"status": "COMPLETE"},
            ),
            (
                "008-prepare-broad",
                "openstar.tess.independent.broad.prepare",
                broad_spec,
            ),
            ("009-run-broad", "openstar.tess.independent.broad.run", broad_run),
        )
        for stage_id, handler_id, result in replay:
            investigation = self._complete(
                store, investigation, stage_id, handler_id, result
            )

        engine = build_engine(
            store, coordinator=object(), poll_interval=0.0, timeout=None
        )
        engine.chain_stages = False
        broad_request = StageRequest(
            "010-interpret-broad",
            "openstar.tess.independent.broad.interpret",
            {},
            "009-run-broad",
        )
        investigation, morphology_request = engine.run_stage(
            investigation,
            broad_request,
            software_id="integration",
            software_version="20.28",
        )

        self.assertEqual(
            "openstar.tess.morphology.analyze", morphology_request.handler_id
        )
        persisted_broad = investigation.stages[-1]
        self.assertEqual(
            "RECURRENT_BUT_UNRESOLVED_CROSS_SECTOR_VARIABILITY",
            persisted_broad.result["variabilityCharacterization"]["state"],
        )

        # A process restart reconstructs exactly the persisted continuation and
        # does not replay any of the seven completed prerequisite stages.
        restarted = store.load(investigation.id)
        restarted_request = plan_tess_branches(restarted, target)[0].experiment
        self.assertEqual(morphology_request, restarted_request)
        self.assertEqual(8, len(restarted.stages))

        completed, next_request = engine.run_stage(
            restarted,
            restarted_request,
            software_id="integration",
            software_version="20.28",
        )

        morphology_stage = completed.stages[-1]
        self.assertEqual("COMPLETE", morphology_stage.status)
        self.assertEqual(
            "openstar.tess.morphology.analyze", morphology_stage.handler_id
        )
        self.assertEqual(3, len(morphology_stage.result["sectorResults"]))
        self.assertEqual(
            {
                "periodFamily",
                "primaryDataset",
                "independentSector66",
                "independentSector67",
            },
            set(morphology_stage.provenance.input_hashes),
        )
        self.assertEqual(1, len(morphology_stage.artifacts))
        self.assertEqual("openstar.tess.finalize", next_request.handler_id)
        self.assertEqual(9, len(completed.stages))


if __name__ == "__main__":
    unittest.main()
