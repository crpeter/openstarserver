import json
import math
import tempfile
import unittest
from pathlib import Path


from workflows.tess.tess_mode_identification import (
    GENERIC_REFINEMENT_WORKLOAD_ID,
    identify_residual_mode,
)


class TessModeIdentificationTests(unittest.TestCase):
    def _datasets(self, root, *, family_period=10.0, residual_period=None,
                  residual_amplitude=0.0, sectors=4):
        paths = []
        for sector in range(sectors):
            times = [sector * 100.0 + 25.0 * index / 499 for index in range(500)]
            flux = [(0.8 * math.sin(2 * math.pi * time / family_period)
                     + 0.35 * math.cos(4 * math.pi * time / family_period)
                     + (residual_amplitude * math.sin(2 * math.pi * time / residual_period + 0.2)
                        if residual_period else 0.0)) for time in times]
            path = Path(root) / f"sector-{sector}.json"
            path.write_text(json.dumps({"times": times, "flux": flux}))
            paths.append(path)
        return paths

    def test_higher_harmonic_is_not_an_independent_mode(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, residual_period=2.5, residual_amplitude=0.5)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=2.5, independent_sectors=[1, 2, 3, 4])
        self.assertEqual("HIGHER_ORDER_HARMONIC_STRUCTURE", result["classification"])
        self.assertFalse(result["independentModeEvidenceSurvived"])
        self.assertTrue(result["harmonicRelation"]["commensurateWithinResolution"])

    def test_supported_off_harmonic_mode_survives(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, residual_period=3.1, residual_amplitude=0.6)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=3.1, independent_sectors=[94, 95, 102, 103])
        self.assertEqual("INDEPENDENT_STABLE_MODE", result["classification"])
        self.assertTrue(result["independentModeEvidenceSurvived"])
        self.assertEqual("RESIDUAL_MODE_PIXEL_LOCALIZATION", result["recommendedNextTest"])

    def test_weak_improvement_is_not_compelling(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=3.1, independent_sectors=[1, 2, 3])
        self.assertEqual("NO_COMPELLING_RESIDUAL_MODE", result["classification"])

    def test_insufficient_support_prevents_strong_independent_claim(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, residual_period=3.1, residual_amplitude=0.6)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=3.1, independent_sectors=[94, 95])
        self.assertFalse(result["independentModeEvidenceSurvived"])
        self.assertNotEqual("INDEPENDENT_STABLE_MODE", result["classification"])

    def test_tic_277940827_relation_runs_nested_comparison(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, family_period=10.3008408008,
                                  residual_period=2.5751446508, residual_amplitude=0.5)
            result = identify_residual_mode(dataset_paths=paths,
                established_period_days=10.3008408008, residual_period_days=2.5751446508,
                independent_sectors=[94, 95, 102, 103])
        self.assertEqual(4, result["harmonicRelation"]["testedOrder"])
        self.assertEqual(3, len(result["modelComparison"]["models"]))
        self.assertEqual("openstar.lomb-scargle.v1", GENERIC_REFINEMENT_WORKLOAD_ID)




if __name__ == "__main__":
    unittest.main()
