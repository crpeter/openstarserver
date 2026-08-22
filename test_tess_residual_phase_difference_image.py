import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import build_engine

from workflows.tess.tess_residual_phase_difference_image import (
    interpret_residual_phase_difference_imaging,
    prepare_residual_phase_difference_imaging,
    run_residual_phase_difference_imaging,
    _production_difference_image_inputs,
)


class ResidualPhaseDifferenceImageTests(unittest.TestCase):
    def _preparation(self, root):
        bridge = {
            "version": "bridge", "ticID": 277940827,
            "targetSky": {"raDeg": 1.0, "decDeg": 2.0},
            "catalogCandidates": [
                {"raDeg": 1.001, "decDeg": 2.0, "catalogIDs": {"ticID": 1}},
                {"raDeg": 0.999, "decDeg": 2.0, "catalogIDs": {"ticID": 2}},
            ],
            "sectors": [94, 95, 102, 103],
            "referenceFamilyPeriodDays": 10.30084080080649,
            "residualReferenceFrequency": 1 / 2.207,
            "residualTimeReferenceDays": 2500.0,
            "fractionalFrequencyDriftPerDay": 0.0,
            "subtractedHarmonicOrders": [1, 2, 3, 4],
            "physicalCycleResolved": False,
        }
        summary = {"version": "stage-047", "classification": "UNRESOLVED",
                   "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
                   "sourceAttributionResolved": False,
                   "catalogCandidates": bridge["catalogCandidates"]}
        return prepare_residual_phase_difference_imaging(
            localization_summary=summary, localization_preparation=bridge,
            output_dir=root, investigation_id="tic-277940827")

    def test_real_numpy_difference_images_support_target_without_prf_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation = self._preparation(Path(temporary))
            inputs = []
            rng = np.random.default_rng(277940827)
            for sector in preparation["sectors"]:
                times = np.linspace(2500, 2527, 400)
                cube = rng.normal(0, .08, (400, 5, 5))
                phase = 2 * np.pi * preparation["residualReferenceFrequency"] * (times - 2500)
                cube[:, 2, 2] += 2.0 * np.sin(phase)
                inputs.append({"sector": sector, "times": times, "prewhitened": cube,
                               "valid": np.ones((5, 5), bool),
                               "componentPixelCenters": [
                                   {"componentID": "target", "x": 2.0, "y": 2.0},
                                   {"componentID": "candidate-1", "x": 3.5, "y": 2.0},
                                   {"componentID": "candidate-2", "x": .5, "y": 2.0}]})
            run = run_residual_phase_difference_imaging(preparation, sector_inputs=inputs)
            result = interpret_residual_phase_difference_imaging(preparation, run)
            self.assertEqual("TARGET_SUPPORTED", result["classification"])
            self.assertTrue(result["sourceAttributionResolved"])
            self.assertEqual([1, 2, 3, 4], result["subtractedHarmonicOrders"])
            self.assertFalse(result["physicalCycleResolved"])
            self.assertEqual("TARGET_INTRINSIC_RESIDUAL_MODELING",
                             result["recommendedNextTest"])
            self.assertFalse(result["physicalMechanismResolved"])

    def test_cross_sector_disagreement_is_source_switching(self):
        preparation = {"referenceFamilyPeriodDays": 10.30084080080649,
                       "subtractedHarmonicOrders": [1, 2, 3, 4],
                       "residualReferenceFrequency": 1 / 2.207}
        result = interpret_residual_phase_difference_imaging(
            preparation, {"sectorResults": [
                {"sector": 94, "classification": "TARGET_SUPPORTED"},
                {"sector": 95, "classification": "CANDIDATE_1_SUPPORTED"}]})
        self.assertEqual("SOURCE_SWITCHING_BY_SECTOR", result["classification"])
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertEqual("SOURCE_SWITCHING_TEMPORAL_MODEL", result["recommendedNextTest"])

    def test_production_input_builder_has_no_prf_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation = self._preparation(Path(temporary))
            times = np.linspace(2500, 2527, 200)
            cube = np.random.default_rng(1).normal(size=(200, 4, 4))
            class Value:
                def __init__(self, value): self.value = value
            class WCS:
                def world_to_pixel(self, coordinate): return coordinate
            tpf = type("TPF", (), {"time": Value(times), "flux": Value(cube), "wcs": WCS()})()
            # Any accidental regression to the catalog-guided production builder,
            # PRF download, interpolation, renderer, or calibration fails loudly.
            forbidden = [
                "_list_official_prf_grid", "_official_prf_at_detector_position",
                "_render_prf_template", "_fit_static_image",
            ]
            patches = [mock.patch(
                f"workflows.tess.tess_catalog_guided_localization.{name}",
                side_effect=AssertionError(f"PRF helper called: {name}")) for name in forbidden]
            with mock.patch(
                "workflows.tess.tess_residual_phase_difference_image._download_tpf",
                return_value=(tpf, {"sourceType": "SPOC TPF"})), mock.patch(
                "workflows.tess.tess_residual_phase_difference_image._skycoord",
                side_effect=lambda ra, dec: (ra, dec)):
                for patch in patches: patch.start()
                try:
                    inputs = _production_difference_image_inputs(preparation)
                finally:
                    for patch in patches: patch.stop()
            self.assertEqual([94, 95, 102, 103], [item["sector"] for item in inputs])
            self.assertEqual(list(("target", "candidate-1", "candidate-2")),
                             [item["componentID"] for item in inputs[0]["componentPixelCenters"]])
            self.assertNotIn("renderTemplates", inputs[0])

    def _workflow_interpret(self, root, classification):
        store = InvestigationStore(root / "store")
        investigation = store.create("tic-277940827", "workflow", "1")
        preparation = self._preparation(root)
        candidates = preparation["catalogCandidates"]
        labels = (classification if isinstance(classification, list)
                  else [classification] * len(preparation["sectors"]))
        run = {"sectorResults": [{"sector": sector, "classification": label}
                                 for sector, label in zip(preparation["sectors"], labels)]}
        for stage_id, handler, result in (
            ("048-prepare-residual-phase-difference-imaging",
             "openstar.tess.residual-phase-difference-imaging.prepare", preparation),
            ("049-run-residual-phase-difference-imaging",
             "openstar.tess.residual-phase-difference-imaging.run", run),
        ):
            running = InvestigationStage(stage_id, handler, "RUNNING", None, {})
            investigation = store.append_running_stage(investigation, running)
            terminal = store.build_terminal_stage(
                stage_id=stage_id, handler_id=handler, status="COMPLETE",
                triggered_by_stage_id=None, parameters={}, result=result, error=None,
                software_id="test", software_version="1", started_at=running.started_at)
            investigation = store.complete_current_stage(investigation, terminal)
        engine = build_engine(store, object(), poll_interval=0, timeout=0)
        request = StageRequest("050-interpret-residual-phase-difference-imaging",
                               "openstar.tess.residual-phase-difference-imaging.interpret", {},
                               "049-run-residual-phase-difference-imaging")
        completed, next_stage = engine.run_stage(
            investigation, request, software_id="test", software_version="1")
        return completed, next_stage, candidates

    def test_unresolved_stage_050_completes_and_quiesces(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, next_stage, _ = self._workflow_interpret(Path(temporary), "UNRESOLVED")
            self.assertEqual("COMPLETE", completed.stages[-1].status)
            self.assertEqual("QUIESCENT_AWAITING_DATA", completed.status)
            self.assertIsNone(next_stage)

    def test_candidate_stage_050_schedules_direct_variability(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, next_stage, candidates = self._workflow_interpret(
                Path(temporary), "CANDIDATE_2_SUPPORTED")
            result = completed.stages[-1].result
            self.assertEqual("COMPLETE", completed.stages[-1].status)
            self.assertEqual(candidates[1], result["preferredCandidate"])
            self.assertEqual(candidates, result["catalogCandidates"])
            self.assertEqual("openstar.tess.offset-source-variability.prepare",
                             next_stage.handler_id)

    def test_target_stage_050_remains_physically_unresolved_and_quiescent(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, next_stage, _ = self._workflow_interpret(
                Path(temporary), "TARGET_SUPPORTED")
            result = completed.stages[-1].result
            self.assertFalse(result["physicalMechanismResolved"])
            self.assertEqual("TARGET_INTRINSIC_RESIDUAL_MODELING",
                             result["recommendedNextTest"])
            self.assertEqual("QUIESCENT_AWAITING_DATA", completed.status)
            self.assertIsNone(next_stage)

    def test_real_stage_050_source_switching_schedules_stage_051_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, next_stage, _ = self._workflow_interpret(
                Path(temporary), ["UNRESOLVED", "UNRESOLVED", "TARGET_SUPPORTED",
                                  "CANDIDATE_1_SUPPORTED"])
            self.assertEqual("SOURCE_SWITCHING_BY_SECTOR",
                             completed.stages[-1].result["classification"])
            self.assertEqual("051-prepare-source-switching-temporal-model",
                             next_stage.id)
            self.assertEqual("openstar.tess.source-switching-temporal-model.prepare",
                             next_stage.handler_id)
