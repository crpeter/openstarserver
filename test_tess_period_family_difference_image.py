import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_period_family_difference_image import (
    PREPARE_HANDLER,
    admit_period_family_difference_imaging,
    freeze_period_family_boundary,
    interpret_period_family_difference_imaging,
    run_period_family_difference_imaging,
    verified_period_family_boundary,
)


SECTORS = (98, 96, 95, 93)
FREQUENCIES = (0.2195011427899658, 0.22126030428069043,
               0.22155137519848722, 0.22041593035095275)


class PeriodFamilyDifferenceImageTests(unittest.TestCase):
    def _boundary(self, root: Path):
        store = InvestigationStore(root / "investigations")
        investigation = store.create(
            "manual-period-family", "openstar.workflow.tess-investigation.v1", "20.2",
            {"ticID": 238919539},
        )
        datasets = [
            {
                "sector": sector,
                "candidateFrequency": frequency,
                "candidatePeriodDays": 1.0 / frequency,
                "candidatePower": 0.9,
                "candidatePeakProminenceRatio": 5.0,
                "candidateFoldCoherence": 0.95,
                "periodStatus": "RELIABLE",
                "periodConfidence": "high",
            }
            for sector, frequency in zip(SECTORS, FREQUENCIES)
        ]
        independent_results = [
            {
                "sector": sector,
                "datasetID": f"sector-{sector}",
                "candidateFrequency": frequency,
                "candidatePeriodDays": 1.0 / frequency,
                "recurrenceClassification": "RESOLUTION_LIMITED",
                "resolutionLimited": True,
                "supportsTarget": False,
                "eligibleForRecurrence": True,
                "boundaryHit": False,
            }
            for sector, frequency in zip(SECTORS, FREQUENCIES)
        ]
        for result, dataset in zip(independent_results, datasets):
            dataset["datasetID"] = result["datasetID"]
        broad_results = [
            {"sector": sector, "boundaryHit": index > 0}
            for index, sector in enumerate(SECTORS)
        ]
        prepared_sectors = [{"sector": sector} for sector in SECTORS]
        stage_results = {
            "001-prepare-target": {"ticID": 238919539, "sector": 1},
            "002-primary-distributed-search": {
                "candidateFrequency": 0.21976441741253439,
                "candidatePeriodDays": 4.550327172040929,
                "candidatePower": 0.9718,
                "periodStatus": "RELIABLE",
                "periodConfidence": "high",
            },
            "003-catalog-identity": {
                "ticID": 238919539,
                "tic": {"metadata": {"raDeg": 94.229274, "decDeg": -52.873880}},
                "tess": {"officialSectors": [1, *SECTORS]},
            },
            "004-hypotheses": {"observedPeriodDays": 4.550327172040929},
            "005-planner": {"action": "INDEPENDENT_SECTOR_VERIFICATION"},
            "006-prepare-independent-sectors": {
                "targetPeriodDays": 4.550327172040929,
                "preparedSectors": prepared_sectors,
            },
            "007-run-independent-sectors": {"datasets": datasets},
            "008-interpret-independent-sectors": {
                "eligibleSectorCount": 4,
                "supportingSectorCount": 0,
                "resolutionLimitedSectorCount": 4,
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "contradictionPlan": {
                    "action": "BROAD_INDEPENDENT_SEARCH",
                    "reason": "targeted-candidate-not-recurrent-independent-sectors-contain-alternate-reliable-structure",
                    "reliableSectorCount": 4,
                },
                "sectorResults": independent_results,
            },
            "009-prepare-broad-independent-search": {"preparedSectors": prepared_sectors},
            "010-run-broad-independent-search": {"datasets": []},
            "011-interpret-broad-independent-search": {
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "sectorResults": broad_results,
                "eligibleSectorCount": 1,
                "bestCluster": {"count": 1},
                "promotionEligible": False,
                "promotionBlockers": ["insufficient-independent-sector-support"],
                "selectedPeriodDays": None,
            },
            "012-finalize": {
                "claim": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "automaticDiscoveryClaim": False,
                "selectedPeriodDays": None,
                "recommendedNextTest": None,
            },
        }
        handlers = {
            "001-prepare-target": "openstar.tess.prepare-target",
            "002-primary-distributed-search": "openstar.tess.primary-project.run",
            "003-catalog-identity": "openstar.tess.catalog-identity",
            "004-hypotheses": "openstar.tess.hypotheses",
            "005-planner": "openstar.tess.planner",
            "006-prepare-independent-sectors": "openstar.tess.independent.prepare",
            "007-run-independent-sectors": "openstar.tess.independent.run",
            "008-interpret-independent-sectors": "openstar.tess.independent.interpret",
            "009-prepare-broad-independent-search": "openstar.tess.independent.broad.prepare",
            "010-run-broad-independent-search": "openstar.tess.independent.broad.run",
            "011-interpret-broad-independent-search": "openstar.tess.independent.broad.interpret",
            "012-finalize": "openstar.tess.finalize",
        }
        previous = None
        for stage_id, handler in handlers.items():
            running = InvestigationStage(stage_id, handler, "RUNNING", previous, {})
            investigation = store.append_running_stage(investigation, running)
            terminal = store.build_terminal_stage(
                stage_id=stage_id,
                handler_id=handler,
                status="COMPLETE",
                triggered_by_stage_id=previous,
                parameters={},
                result=stage_results[stage_id],
                error=None,
                software_id="test",
                software_version="1",
                started_at=running.started_at,
                stop=stage_id == "012-finalize",
            )
            investigation = store.complete_current_stage(investigation, terminal)
            previous = stage_id
        investigation = store.set_control_state(
            investigation,
            status="COMPLETE",
            control_state={
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            },
        )
        return store, investigation

    @staticmethod
    def _preparation(root: Path):
        return {
            "artifactRoot": str(root),
            "ticID": 238919539,
            "targetSky": {"raDeg": 94.2, "decDeg": -52.8},
            "sectorDetections": [
                {"sector": sector, "frequencyCyclesPerDay": frequency,
                 "periodDays": 1.0 / frequency}
                for sector, frequency in zip(SECTORS, FREQUENCIES)
            ],
        }

    @staticmethod
    def _target_input(sector: int, frequency: float):
        rng = np.random.default_rng(sector)
        times = np.linspace(0.0, 27.0, 500)
        cube = rng.normal(0.0, 0.04, (len(times), 5, 5))
        cube[:, 2, 2] += 2.0 * np.sin(2.0 * math.pi * frequency * times)
        return {
            "sector": sector,
            "times": times,
            "fluxCube": cube,
            "targetPixel": {"x": 2.0, "y": 2.0},
            "pixelScaleArcsec": 21.0,
            "acquisitionProvenance": {"sourceType": "SYNTHETIC_TEST"},
        }

    def test_uploaded_boundary_shape_is_generic_and_ledger_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._boundary(Path(temporary))
            frozen, hashes = verified_period_family_boundary(store, investigation)
            self.assertEqual(238919539, frozen["ticID"])
            self.assertEqual(list(SECTORS),
                             [item["sector"] for item in frozen["independentSectorDetections"]])
            self.assertEqual(12, len(hashes))
            self.assertFalse(frozen["periodDetectionRecomputed"])

    def test_admission_changes_only_control_state_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._boundary(Path(temporary))
            history = copy.deepcopy(investigation.stages)
            ledger_bytes = {
                stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                for stage in investigation.stages
            }
            admitted = admit_period_family_difference_imaging(store, investigation)
            self.assertEqual(history, admitted.stages)
            self.assertEqual("RUNNING", admitted.status)
            selected = admitted.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("013-prepare-period-family-difference-imaging", selected["id"])
            self.assertEqual(PREPARE_HANDLER, selected["handler_id"])
            self.assertEqual(admitted, admit_period_family_difference_imaging(store, admitted))
            self.assertEqual(ledger_bytes, {
                stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                for stage in investigation.stages
            })

    def test_snapshot_or_ledger_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._boundary(Path(temporary))
            path = store.stage_path_for(investigation.id, "008-interpret-independent-sectors")
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ledger verification failed"):
                verified_period_family_boundary(store, investigation)

    def test_real_phase_difference_images_support_target_without_period_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation = self._preparation(Path(temporary))
            inputs = [self._target_input(sector, frequency)
                      for sector, frequency in zip(SECTORS, FREQUENCIES)]
            run = run_period_family_difference_imaging(preparation, sector_inputs=inputs)
            result = interpret_period_family_difference_imaging(preparation, run)
            self.assertEqual("TARGET_PERIOD_FAMILY_SUPPORTED", result["classification"])
            self.assertEqual(sorted(SECTORS), result["targetSupportingSectors"])
            self.assertFalse(run["periodDetectionRecomputed"])
            self.assertFalse(result["periodFamilyResolved"])
            self.assertFalse(result["physicalMechanismResolved"])
            self.assertEqual("HUMAN_REVIEW_REQUIRED", result["claimDecision"]["claim"])
            self.assertEqual("UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION",
                             result["recommendedNextTest"])

    def test_cross_sector_source_switching_and_off_target_are_distinct(self):
        preparation = {"version": "test"}
        mixed = interpret_period_family_difference_imaging(preparation, {
            "sectorResults": [
                {"sector": 1, "classification": "TARGET_CONSISTENT"},
                {"sector": 2, "classification": "TARGET_CONSISTENT"},
                {"sector": 3, "classification": "OFF_TARGET",
                 "skyOffsetEastArcsec": 30.0, "skyOffsetNorthArcsec": 0.0},
                {"sector": 4, "classification": "OFF_TARGET",
                 "skyOffsetEastArcsec": 31.0, "skyOffsetNorthArcsec": 0.0},
            ]
        })
        self.assertEqual("SOURCE_SWITCHING_BY_SECTOR", mixed["classification"])
        off_target = interpret_period_family_difference_imaging(preparation, {
            "sectorResults": [
                {"sector": sector, "classification": "OFF_TARGET",
                 "skyOffsetEastArcsec": 30.0 + index, "skyOffsetNorthArcsec": 4.0}
                for index, sector in enumerate(SECTORS)
            ]
        })
        self.assertEqual("OFF_TARGET_PERIOD_FAMILY_SUPPORTED", off_target["classification"])
        self.assertEqual("OFFSET_SOURCE_CATALOG_IDENTIFICATION",
                         off_target["recommendedNextTest"])

    def test_full_mocked_lifecycle_appends_013_to_015_without_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, investigation = self._boundary(root)
            history = copy.deepcopy(investigation.stages)
            ledger_bytes = {
                stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                for stage in history
            }
            admitted = admit_period_family_difference_imaging(store, investigation)
            request = StageRequest(**admitted.metadata["controlState"]["selectedExperiment"])
            engine = build_engine(store, object(), poll_interval=0, timeout=None)
            with mock.patch(
                "workflows.tess.tess_period_family_difference_image._production_sector_input",
                side_effect=lambda preparation, detection: self._target_input(
                    detection["sector"], detection["frequencyCyclesPerDay"]
                ),
            ):
                completed = engine.run(
                    admitted, request, software_id="test", software_version="1", max_stages=3
                )
            self.assertEqual(
                ["013-prepare-period-family-difference-imaging",
                 "014-run-period-family-difference-imaging",
                 "015-interpret-period-family-difference-imaging"],
                [stage.id for stage in completed.stages[len(history):]],
            )
            self.assertEqual(history, completed.stages[:len(history)])
            self.assertEqual("QUIESCENT_AWAITING_DATA", completed.status)
            self.assertEqual("TARGET_PERIOD_FAMILY_SUPPORTED",
                             completed.stages[-1].result["classification"])
            self.assertEqual(ledger_bytes, {
                stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                for stage in history
            })
            self.assertTrue(all(
                Path(reference.path).resolve().is_relative_to(store.directory_for(investigation.id).resolve())
                for stage in completed.stages[len(history):]
                for reference in stage.artifacts
            ))


if __name__ == "__main__":
    unittest.main()
