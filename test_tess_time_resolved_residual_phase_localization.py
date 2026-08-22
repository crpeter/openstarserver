import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_time_resolved_residual_phase_localization import (
    interpret_time_resolved_residual_phase_localization,
    prepare_time_resolved_residual_phase_localization,
    run_time_resolved_residual_phase_localization,
)


class TimeResolvedResidualPhaseLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.preparation = {
            "ticID": 277940827, "sectors": [94, 95, 102, 103],
            "targetSky": {"raDeg": 10., "decDeg": 20.},
            "catalogCandidates": [{"raDeg": 10.1, "decDeg": 20.1,
                                   "catalogIDs": {"ticID": 111}},
                                  {"raDeg": 10.2, "decDeg": 20.2,
                                   "catalogIDs": {"ticID": 222}}],
            "referenceFamilyPeriodDays": 10.30084080080649,
            "subtractedHarmonicOrders": [4, 2], "physicalCycleResolved": False,
            "residualReferenceFrequency": .45, "residualTimeReferenceDays": 2500.,
            "fractionalFrequencyDriftPerDay": 0.,
        }
        self.centers = [{"componentID": "target", "x": 1., "y": 1.},
                        {"componentID": "candidate-1", "x": 3., "y": 1.},
                        {"componentID": "candidate-2", "x": 2., "y": 3.}]

    def _inputs(self, sources_by_sector, noise=.06, centers=None):
        rng = np.random.default_rng(277940827)
        result = []
        yy, xx = np.mgrid[:5, :5]
        centers = centers or self.centers
        templates = {c["componentID"]: np.exp(-((xx-c["x"])**2+(yy-c["y"])**2)/.35)
                     for c in centers}
        for sector, sources in zip(self.preparation["sectors"], sources_by_sector):
            times = np.linspace(2500., 2520., 640)
            cube = rng.normal(0., noise, (len(times), 5, 5))
            halves = np.array_split(np.arange(len(times)), 2)
            for indices, source in zip(halves, sources):
                phase = 2*np.pi*self.preparation["residualReferenceFrequency"]*(times[indices]-2500.)
                cube[indices] += 2.0*np.sin(phase)[:, None, None]*templates[source]
            result.append({"sector": sector, "times": times, "prewhitened": cube,
                           "valid": np.ones((5, 5), bool),
                           "componentPixelCenters": centers})
        return result

    def _classify(self, sources, **kwargs):
        run = run_time_resolved_residual_phase_localization(
            self.preparation, sector_inputs=self._inputs(sources, **kwargs))
        return interpret_time_resolved_residual_phase_localization(self.preparation, run), run

    def test_target_in_all_windows_is_stable_target(self):
        result, run = self._classify([["target", "target"]] * 4)
        self.assertEqual("STABLE_TARGET_LOCALIZATION", result["classification"])
        self.assertIsNone(result["preferredCandidate"])
        self.assertEqual("TARGET_INTRINSIC_RESIDUAL_MODELING",
                         result["recommendedNextTest"])
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertTrue(all(len(s["windowResults"]) == 2 for s in run["sectorResults"]))

    def test_candidate_one_in_all_windows_preserves_candidates_and_validation(self):
        result, _ = self._classify([["candidate-1", "candidate-1"]] * 4)
        self.assertEqual("STABLE_CANDIDATE_1_LOCALIZATION", result["classification"])
        self.assertIs(self.preparation["catalogCandidates"][0], result["preferredCandidate"])
        self.assertIs(self.preparation["catalogCandidates"], result["catalogCandidates"])
        self.assertTrue(result["sourceAttributionResolved"])
        self.assertEqual("INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
                         result["recommendedNextTest"])

    def test_candidate_two_in_all_windows_preserves_candidates_and_validation(self):
        result, _ = self._classify([["candidate-2", "candidate-2"]] * 4)
        self.assertEqual("STABLE_CANDIDATE_2_LOCALIZATION", result["classification"])
        self.assertIs(self.preparation["catalogCandidates"][1], result["preferredCandidate"])
        self.assertIs(self.preparation["catalogCandidates"], result["catalogCandidates"])
        self.assertTrue(result["sourceAttributionResolved"])
        self.assertEqual("INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
                         result["recommendedNextTest"])

    def test_stage_056_routes_candidates_and_quiesces_target(self):
        for source, expected_next in (("candidate-1", True), ("candidate-2", True),
                                      ("target", False)):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as root:
                root = Path(root); store = InvestigationStore(root / "store")
                investigation = store.create("tic-277940827", "workflow", "1")
                preparation = {**self.preparation, "artifactRoot": str(root / "artifacts")}
                (root / "artifacts").mkdir()
                run = run_time_resolved_residual_phase_localization(
                    preparation, sector_inputs=self._inputs([[source, source]] * 4))
                for stage_id, handler, value in (
                    ("054-prepare-time-resolved-residual-phase-localization",
                     "openstar.tess.time-resolved-residual-phase-localization.prepare", preparation),
                    ("055-run-time-resolved-residual-phase-localization",
                     "openstar.tess.time-resolved-residual-phase-localization.run", run)):
                    running = InvestigationStage(stage_id, handler, "RUNNING", None, {})
                    investigation = store.append_running_stage(investigation, running)
                    terminal = store.build_terminal_stage(
                        stage_id=stage_id, handler_id=handler, status="COMPLETE",
                        triggered_by_stage_id=None, parameters={}, result=value, error=None,
                        software_id="test", software_version="1", started_at=running.started_at)
                    investigation = store.complete_current_stage(investigation, terminal)
                completed, next_stage = build_engine(
                    store, object(), poll_interval=0, timeout=0).run_stage(
                        investigation, StageRequest(
                            "056-interpret-time-resolved-residual-phase-localization",
                            "openstar.tess.time-resolved-residual-phase-localization.interpret", {},
                            "055-run-time-resolved-residual-phase-localization"),
                        software_id="test", software_version="1")
                if expected_next:
                    self.assertEqual("057-prepare-offset-source-variability", next_stage.id)
                    self.assertEqual("openstar.tess.offset-source-variability.prepare",
                                     next_stage.handler_id)
                    self.assertFalse(completed.stages[-1].stop)
                else:
                    self.assertIsNone(next_stage)
                    self.assertTrue(completed.stages[-1].stop)
                    self.assertEqual("QUIESCENT_AWAITING_DATA", completed.status)

    def test_target_then_candidate_one_within_sector_switches(self):
        result, _ = self._classify([["target", "candidate-1"]] * 4)
        self.assertEqual("WITHIN_SECTOR_SOURCE_SWITCHING", result["classification"])

    def test_stable_sectors_with_different_sources_switch_cross_sector(self):
        result, _ = self._classify([["target", "target"], ["target", "target"],
                                    ["candidate-1", "candidate-1"],
                                    ["candidate-1", "candidate-1"]])
        self.assertEqual("CROSS_SECTOR_SOURCE_SWITCHING", result["classification"])

    def test_overlapping_candidates_never_force_switching(self):
        centers = [self.centers[0], {"componentID":"candidate-1","x":3.,"y":1.},
                   {"componentID":"candidate-2","x":3.05,"y":1.}]
        result, run = self._classify([["candidate-1", "candidate-2"]] * 4, centers=centers)
        labels = {w["classification"] for s in run["sectorResults"] for w in s["windowResults"]}
        self.assertTrue(labels <= {"MULTIPLE_OR_BLENDED", "UNRESOLVED",
                                   "NO_QUALITY_LOCALIZATION"})
        self.assertNotIn("SOURCE_SWITCHING", result["classification"])

    def test_weak_windows_have_no_quality_and_are_unresolved(self):
        result, run = self._classify([["target", "target"]] * 4, noise=50.)
        self.assertEqual("UNRESOLVED", result["classification"])
        self.assertTrue(any(w["classification"] == "NO_QUALITY_LOCALIZATION"
                            for s in run["sectorResults"] for w in s["windowResults"]))

    def test_prepare_and_real_prewhitening_preserve_nondefault_orders_verbatim(self):
        interpretation = {"classification": "SECTOR_VARIABLE_MULTI_SOURCE",
            "sourceIdentifiable": True, "sourceAttributionResolved": False,
            "physicalMechanismResolved": False,
            "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA"}
        with tempfile.TemporaryDirectory() as root:
            prepared = prepare_time_resolved_residual_phase_localization(
                temporal_interpretation=interpretation, temporal_preparation=self.preparation,
                output_dir=Path(root), investigation_id="tic-277940827")
        self.assertEqual([4, 2], prepared["subtractedHarmonicOrders"])
        class Value:
            def __init__(self, value): self.value = value
        class WCS:
            def world_to_pixel(self, coordinate): return (1., 1.)
        class TPF:
            time = Value(np.linspace(2500., 2520., 640))
            flux = Value(np.ones((640, 5, 5)))
            wcs = WCS()
        with mock.patch(
            "workflows.tess.tess_residual_phase_difference_image._prewhiten_cube_raw",
            return_value=(np.ones((640, 5, 5)), np.ones((5, 5), bool))
        ) as prewhiten, mock.patch(
            "workflows.tess.tess_residual_phase_difference_image._download_tpf",
            return_value=(TPF(), {"archive": "test"})), mock.patch(
            "workflows.tess.tess_residual_phase_difference_image._skycoord",
            side_effect=lambda ra, dec: (ra, dec)):
            # A flat injected cube may fail later quality localization; the real
            # acquisition/prewhitening boundary must nevertheless be reached.
            run_time_resolved_residual_phase_localization(prepared)
        self.assertEqual(4, prewhiten.call_count)
        self.assertTrue(all(call.kwargs["harmonic_orders"] == (4, 2)
                            for call in prewhiten.call_args_list))
