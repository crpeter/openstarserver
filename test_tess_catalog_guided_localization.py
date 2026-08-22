import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")
    HAS_NUMPY = False
    _installed_numpy_stub = True
else:
    _installed_numpy_stub = False

from workflows.tess.tess_catalog_guided_localization import (
    HYPOTHESES, _calibrate_shared_astrometric_offset,
    _temporal_predictive_validation, interpret_catalog_guided_localization,
    prepare_catalog_guided_localization,
)
from workflows.tess.tess_investigation import catalog_counterpart_variability_continuation

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


class CatalogGuidedLocalizationTest(unittest.TestCase):
    def _catalog(self):
        return {"classification": "AMBIGUOUS_MULTIPLE_CATALOG_COUNTERPARTS",
                "counterpartIdentified": False, "preferredCandidate": None,
                "physicalMechanismResolved": False,
                "recommendedNextTest": "CATALOG_GUIDED_SOURCE_LOCALIZATION",
                "plausibleCatalogCandidates": [
                    {"raDeg": 1.01, "decDeg": 2.01,
                     "catalogIDs": {"ticID": 277940823, "gaiaDR3SourceID": 6380347301744012800}},
                    {"raDeg": 1.02, "decDeg": 2.02,
                     "catalogIDs": {"ticID": 2054721323, "gaiaDR3SourceID": 6380341421932894720}},
                ]}

    def _evidence(self):
        return {"targetPreparation": {"ticID": 277940827},
                "catalogCounterpartIdentification": self._catalog(),
                "prfPreparation": {"targetSky": {"raDeg": 1.0, "decDeg": 2.0}, "sectors": [1, 28]},
                "prfInterpretation": {"classification": "PRF_SOURCE_SWITCHING"},
                "decompositionPreparation": {
                    "referenceFamilyPeriodDays": 10.30084080080649,
                    "subtractedHarmonicOrders": [1, 2, 3, 4], "physicalCycleResolved": False,
                    "residualModelProvenance": {"referenceFrequency": 0.25,
                        "timeReferenceDays": 1400.0, "fractionalFrequencyDriftPerDay": 0.001}}}

    def test_real_ambiguous_contract_schedules_append_only_localization(self):
        catalog = self._catalog()
        before = repr(catalog)
        request = catalog_counterpart_variability_continuation(catalog, request_id="044")
        self.assertEqual("045-prepare-catalog-guided-source-localization", request.id)
        self.assertEqual("openstar.tess.catalog-guided-source-localization.prepare", request.handler_id)
        self.assertEqual(before, repr(catalog))
        self.assertIsNone(catalog["preferredCandidate"])

    def test_preparation_carries_both_candidates_and_frozen_family(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_catalog_guided_localization(
                evidence=self._evidence(), output_dir=Path(directory), investigation_id="real")
        self.assertEqual([277940823, 2054721323], [x["catalogIDs"]["ticID"]
                                                  for x in result["plausibleCatalogCandidates"]])
        self.assertIsNone(result["preferredCandidate"])
        self.assertEqual([1, 2, 3, 4], result["subtractedHarmonicOrders"])
        self.assertFalse(result["physicalCycleResolved"])
        self.assertEqual(set(HYPOTHESES), set(result["sourceHypotheses"]))

    def test_only_consistent_identifiable_candidate_is_promoted(self):
        preparation = {**self._evidence()["decompositionPreparation"],
            "plausibleCatalogCandidates": self._catalog()["plausibleCatalogCandidates"]}
        sectors = [{"sector": x, "decisive": True, "bestModel": "CANDIDATE_2_ONLY"}
                   for x in (1, 28)]
        result = interpret_catalog_guided_localization(preparation, {"sectorResults": sectors})
        self.assertEqual(2054721323, result["preferredCandidate"]["catalogIDs"]["ticID"])
        self.assertEqual("INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
                         result["recommendedNextTest"])
        unresolved = interpret_catalog_guided_localization(preparation, {"sectorResults": sectors[:1]})
        self.assertIsNone(unresolved["preferredCandidate"])
        self.assertEqual("HIGHER_RESOLUTION_SPATIAL_FOLLOWUP", unresolved["recommendedNextTest"])

    @unittest.skipUnless(HAS_NUMPY, "NumPy is required for predictive validation")
    def test_refit_complexity_does_not_substitute_for_frozen_prediction(self):
        times = np.arange(80, dtype=float)
        phase = 2.0 * np.pi * times / 10.0
        basis = np.column_stack((np.sin(phase), np.cos(phase)))
        templates = np.eye(3)
        pixels = np.zeros((80, 3))
        pixels[:, 0] = basis @ np.array([1.0, 0.3])
        pixels[:40, 1] = basis[:40] @ np.array([0.9, 0.2])
        pixels[40:, 1] = basis[40:] @ np.array([-0.9, -0.2])
        pixels += np.random.default_rng(4).normal(0.0, 0.01, pixels.shape)
        result = _temporal_predictive_validation(
            times=times, pixels=pixels, coherent_basis=basis,
            templates=templates, component_ids=["target", "candidate-1", "candidate-2"],
            block_count=2,
        )
        self.assertTrue(all(fold["independentHeldOutDiagnostic"]["bestModel"]
                            == "TARGET_PLUS_CANDIDATE_1" for fold in result["folds"]))
        self.assertNotEqual("TARGET_PLUS_CANDIDATE_1", result["predictiveWinner"])

    @unittest.skipUnless(HAS_NUMPY, "NumPy is required for predictive validation")
    def test_stable_candidate_survives_frozen_train_to_held_out_prediction(self):
        times = np.arange(120, dtype=float)
        phase = 2.0 * np.pi * times / 11.0
        basis = np.column_stack((np.sin(phase), np.cos(phase)))
        templates = np.eye(3)
        pixels = np.zeros((120, 3))
        pixels[:, 1] = basis @ np.array([1.2, -0.4])
        pixels += np.random.default_rng(7).normal(0.0, 0.01, pixels.shape)
        result = _temporal_predictive_validation(
            times=times, pixels=pixels, coherent_basis=basis,
            templates=templates, component_ids=["target", "candidate-1", "candidate-2"],
        )
        self.assertTrue(result["consistent"])
        self.assertEqual("CANDIDATE_1_ONLY", result["predictiveWinner"])
        for fold in result["folds"]:
            self.assertIn("trainingParameterEstimates",
                          fold["models"]["CANDIDATE_1_ONLY"])
            self.assertIn("heldOutLogLikelihood", fold["models"]["CANDIDATE_1_ONLY"])

    @unittest.skipUnless(HAS_NUMPY, "NumPy is required for astrometric calibration")
    def test_astrometric_calibration_applies_one_real_shared_shift(self):
        sources = [{"componentID": name, "x": float(index), "y": float(index),
                    "image": np.ones((2, 2)), "header": {}}
                   for index, name in enumerate(("target", "candidate-1", "candidate-2"))]

        def render(*, source_x, source_y, **_kwargs):
            return np.array([source_x, source_y, 1.0, source_x + source_y])

        def fit(_design, _image, _count):
            # The deterministic objective selects the one common (+0.2, -0.2) trial.
            dx = _design[0, 0] - sources[0]["x"]
            dy = _design[1, 0] - sources[0]["y"]
            return ((dx - 0.2) ** 2 + (dy + 0.2) ** 2,
                    np.ones(_design.shape[1]), 0.9)

        with mock.patch("workflows.tess.tess_catalog_guided_localization._render_prf_template",
                        side_effect=render), mock.patch(
            "workflows.tess.tess_catalog_guided_localization._fit_static_image", side_effect=fit):
            result = _calibrate_shared_astrometric_offset(
                corrected_cube=np.ones((20, 2, 2)), valid_pixels=np.ones((2, 2), dtype=bool),
                source_models=sources)
        self.assertEqual(0.2, result["dxPixels"])
        self.assertEqual(-0.2, result["dyPixels"])
        self.assertEqual(["target", "candidate-1", "candidate-2"],
                         result["sharedAcrossComponentIDs"])


if __name__ == "__main__":
    unittest.main()
