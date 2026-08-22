import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.tess.tess_residual_phase_difference_image import (
    interpret_residual_phase_difference_imaging,
    prepare_residual_phase_difference_imaging,
    run_residual_phase_difference_imaging,
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

