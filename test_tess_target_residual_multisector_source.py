import unittest

from workflows.tess.tess_catalog_guided_localization import generate_source_hypotheses
from workflows.tess.tess_target_residual_multisector_source import (
    MAX_COMPETING_SOURCES, V2018_SECTOR_IDS, derive_additional_sectors,
    derive_competing_sources, interpret_multisector,
)


def evidence(sector):
    return {"sector": sector, "candidateFrequency": 0.19 + sector / 10000,
            "originalTimeOriginDays": 1000.0 + sector,
            "supportsHistoricalResidualFamily": True,
            "recurrenceClassification": "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY"}


def sector(sector_id, sources=(), *, available=True, identifiable=True):
    model = "model-" + "-".join(sources)
    return {"sector": sector_id, "availability": "AVAILABLE" if available else "UNAVAILABLE",
        "fullDataComparison": {"bestModel": model, "bestModelSourceIDs": list(sources),
            "bestModelIdentifiable": identifiable, "completeModelFullRank": identifiable,
            "conditionallyIdentifiableSources": list(sources) if identifiable else []},
        "temporalPredictiveValidation": {"predictiveModel": model, "predictiveSupport": True}}


class MultisectorSourceTests(unittest.TestCase):
    def test_hypotheses_are_all_nonempty_subsets_in_deterministic_order(self):
        self.assertEqual(255, len(generate_source_hypotheses([f"s{i}" for i in range(8)])))
        self.assertEqual(list(generate_source_hypotheses(("a", "b", "c"))), [
            "SOURCE_SUBSET_a", "SOURCE_SUBSET_b", "SOURCE_SUBSET_c",
            "SOURCE_SUBSET_a__b", "SOURCE_SUBSET_a__c", "SOURCE_SUBSET_b__c",
            "SOURCE_SUBSET_a__b__c"])

    def test_exact_old_sectors_excluded_and_remaining_support_used(self):
        all_rows = [evidence(value) for value in (*V2018_SECTOR_IDS, 4, 5, 6, 7)]
        selected = derive_additional_sectors(all_rows)
        self.assertEqual([4, 5, 6, 7], [row["sector"] for row in selected])
        self.assertFalse(set(V2018_SECTOR_IDS) & {row["sector"] for row in selected})

    def test_real_shape_selects_eleven_without_truncation(self):
        rows = [evidence(value) for value in (*V2018_SECTOR_IDS, *range(100, 111))]
        self.assertEqual(11, len(derive_additional_sectors(rows)))

    def test_competitors_preserve_frozen_order_and_bound(self):
        frozen = [{"sourceID": f"s{i}", "raDeg": i, "decDeg": -i} for i in range(8)]
        selected = derive_competing_sources(frozen, [{"classification": "AMBIGUOUS_OR_BLENDED",
            "distancesPixels": {row["sourceID"]: 0.5 for row in frozen}}])
        self.assertEqual(frozen, selected)
        too_many = frozen + [{"sourceID": "s8", "raDeg": 8, "decDeg": -8}]
        with self.assertRaises(RuntimeError):
            derive_competing_sources(too_many, [{"classification": "AMBIGUOUS_OR_BLENDED",
                "distancesPixels": {row["sourceID"]: 0.5 for row in too_many}}])
        self.assertEqual(8, MAX_COMPETING_SOURCES)

    def test_target_catalog_blend_switching_unavailable_and_unresolved(self):
        target = interpret_multisector([sector(i, ("target",)) for i in range(3)], "target")
        self.assertEqual("TARGET_SUPPORTED", target["classification"])
        catalog = interpret_multisector([sector(i, ("other",)) for i in range(3)], "target")
        self.assertEqual("CATALOG_SOURCE_SUPPORTED", catalog["classification"])
        blend = interpret_multisector([sector(i, ("target", "other")) for i in range(2)], "target")
        self.assertEqual("SOURCE_SWITCHING_OR_BLEND", blend["classification"])
        switching = interpret_multisector([sector(1, ("target",)), sector(2, ("target",)),
            sector(3, ("other",)), sector(4, ("other",))], "target")
        self.assertEqual("SOURCE_SWITCHING_OR_BLEND", switching["classification"])
        unresolved = interpret_multisector([sector(1, ("target",)), sector(2, available=False),
            sector(3, ("target",), identifiable=False)], "target")
        self.assertEqual("UNRESOLVED", unresolved["classification"])
        self.assertEqual([2], unresolved["unavailableSectors"])
        self.assertEqual("TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY",
                         unresolved["recommendedNextTest"])


if __name__ == "__main__":
    unittest.main()
