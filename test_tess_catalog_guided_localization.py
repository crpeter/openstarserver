import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.tess.tess_catalog_guided_localization import (
    COMPONENT_IDS,
    _fit_shared_astrometric_shift,
    _temporal_predictive_validation,
    prepare_catalog_guided_localization,
)


class CatalogGuidedLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0.0, 24.0, 240, endpoint=False)
        phase = 2 * np.pi * 0.37 * self.times
        self.basis = np.column_stack((np.sin(phase), np.cos(phase)))
        self.templates = np.asarray([
            [1.0, 0.18, 0.04], [0.6, 0.65, 0.10], [0.12, 0.72, 0.38],
            [0.04, 0.22, 0.90], [0.24, 0.08, 0.55],
        ])

    def _cube(self, vectors):
        cube = np.zeros((len(self.times), len(self.templates)))
        blocks = np.array_split(np.arange(len(self.times)), 4)
        for block, source_vectors in zip(blocks, vectors):
            for source, vector in enumerate(source_vectors):
                cube[block] += ((self.basis[block] @ np.asarray(vector))[:, None]
                                * self.templates[:, source][None, :])
        # deterministic nonzero residual variance makes coherent covariances proper.
        cube += 0.015 * np.sin(np.arange(len(cube))[:, None] * 1.731
                              + np.arange(cube.shape[1])[None, :] * 0.43)
        return cube[:, :, None]

    def _validate(self, vectors):
        return _temporal_predictive_validation(
            times=self.times, prewhitened=self._cube(vectors),
            valid=np.ones((len(self.templates), 1), dtype=bool), templates=self.templates,
            residual_frequency=0.37, time_reference=0.0, drift=0.0,
            coherent_basis=self.basis)

    def test_refit_trap_is_rejected_by_frozen_aggregate_prediction(self):
        # Candidate 1 is strong in each separately refitted block, but its vector flips.
        vectors = [[(1.0, 0.35), (0.9 * (-1) ** block, 0.6 * (-1) ** block), (0, 0)]
                   for block in range(4)]
        result = self._validate(vectors)
        self.assertEqual("TARGET_ONLY", result["predictiveModel"])
        self.assertEqual(set(result["totalHeldOutLogLikelihoodByModel"]), {
            "TARGET_ONLY", "CANDIDATE_1_ONLY", "CANDIDATE_2_ONLY",
            "TARGET_PLUS_CANDIDATE_1", "TARGET_PLUS_CANDIDATE_2",
            "CANDIDATE_1_PLUS_CANDIDATE_2", "TARGET_PLUS_BOTH"})
        for fold in result["folds"]:
            for model in fold["models"].values():
                self.assertIn("heldOutChiSquare", model)
                self.assertIn("heldOutLogLikelihood", model)

    def test_stable_candidate_survives_aggregate_prediction(self):
        vectors = [[(0, 0), (1.2, -0.45), (0, 0)] for _ in range(4)]
        result = self._validate(vectors)
        self.assertEqual("CANDIDATE_1_ONLY", result["predictiveModel"])

    def test_one_shared_astrometric_shift_applies_to_all_sources(self):
        expected = (0.2, -0.2)
        base = self.templates
        def render(dx, dy):
            penalty = (dx - expected[0]) ** 2 + (dy - expected[1]) ** 2
            return base + penalty * np.asarray([1, -1, 1, -1, 1])[:, None]
        truth = render(*expected)
        observations = truth * np.asarray([1.0, 0.5, -0.3])[None, :]
        observations = np.column_stack((observations.sum(axis=1),
                                        0.3 * observations.sum(axis=1)))
        result = _fit_shared_astrometric_shift(
            observations=observations,
            covariances=np.repeat(np.eye(2)[None, :, :] * 1e-3, len(base), axis=0),
            render_templates=render)
        calibration = result["sharedAstrometricCalibration"]
        self.assertEqual(expected, (calibration["dxPixels"], calibration["dyPixels"]))
        self.assertEqual(list(COMPONENT_IDS), calibration["appliedToComponents"])
        self.assertFalse(calibration["independentSourceMotion"])

    def test_tic_path_preserves_ambiguous_preparation_contract(self):
        catalog = {"recommendedNextTest": "CATALOG_GUIDED_SOURCE_LOCALIZATION",
                   "physicalMechanismResolved": False,
                   "plausibleCatalogCandidates": [{"id": 1}, {"id": 2}]}
        prf = {"target": {"componentID": "target"}, "residualReferenceFrequency": 0.3,
               "residualTimeReferenceDays": 1.0, "fractionalFrequencyDriftPerDay": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_catalog_guided_localization(
                catalog_summary=catalog, prf_preparation=prf,
                output_dir=Path(directory), investigation_id="TIC-277940827")
        self.assertEqual([{"id": 1}, {"id": 2}], result["catalogCandidates"])
        self.assertIsNone(result["preferredCandidate"])
        self.assertEqual([1, 2, 3, 4], result["subtractedHarmonicOrders"])
        self.assertFalse(result["physicalCycleResolved"])


if __name__ == "__main__":
    unittest.main()
