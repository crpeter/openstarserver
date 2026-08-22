import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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
            "catalogCandidates": [{"raDeg": 10.1, "decDeg": 20.1},
                                  {"raDeg": 10.2, "decDeg": 20.2}],
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
        self.assertTrue(all(len(s["windowResults"]) == 2 for s in run["sectorResults"]))

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
