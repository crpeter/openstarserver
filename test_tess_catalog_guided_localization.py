import tempfile
import unittest
import sys
import types
from pathlib import Path

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")

from workflows.tess.tess_catalog_guided_localization import (
    HYPOTHESES, interpret_catalog_guided_localization,
    prepare_catalog_guided_localization,
)
from workflows.tess.tess_investigation import catalog_counterpart_variability_continuation


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


if __name__ == "__main__":
    unittest.main()
