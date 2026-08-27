import json
import unittest
import urllib.error

from workflows.tess.tess_external_companion_evidence import (
    FIELDS, ExternalEvidenceTransientError, acquire_external_evidence,
    interpret_external_evidence, localization_gate, review_source_attribution,
)


class Response:
    status = 200
    headers = {"Content-Type": "application/json"}
    def __init__(self, value): self.value = value
    def read(self): return self.value
    def __enter__(self): return self
    def __exit__(self, *args): pass


class ExternalCompanionEvidenceTests(unittest.TestCase):
    def setUp(self):
        sectors = [{"sector": 1, "role": "PRIMARY", "usable": True,
                    "classification": "TARGET_CONSISTENT", "matchedCatalogHypothesis": "TIC-42"}]
        sectors += [{"sector": number, "role": "INDEPENDENT", "usable": True,
                     "classification": "TARGET_CONSISTENT", "matchedCatalogHypothesis": "TIC-42"}
                    for number in (2, 3, 4)]
        self.localization = {
            "resultVersion": "openstar.tess-eclipse-event-source-localization.v1",
            "sourceAttributionResolved": True, "classification": "TARGET_CONSISTENT_ECLIPSE_SOURCE",
            "recommendedNextTest": "SOURCE_ATTRIBUTION_REVIEW",
            "pixelDataChangedFrozenEventDefinition": False, "catalogAnswerKeyUsed": False,
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "attributedCatalogHypothesis": "TIC-42", "sectorResults": sectors,
            "frozenCatalog": {"catalogHypotheses": [{"sourceID": "TIC-42", "isTarget": True,
                                                       "ticID": 42, "gaiaDR3SourceID": "Gaia DR3 7"}]},
            "frozenEphemeris": {"refinedPeriodDays": 2.0}, "binaryConfirmationSHA256": "b" * 64,
        }
        self.review = review_source_attribution(self.localization)

    def row(self, mass=10.0, up=1.0, down=-1.0, period=2.0, **changes):
        row = {field: None for field in FIELDS}
        row.update({"pl_name": "published object", "hostname": "published host", "tic_id": "TIC 42",
                    "gaia_dr3_id": "Gaia DR3 7", "default_flag": 1, "soltype": "Published Confirmed",
                    "pl_controv_flag": 0, "rv_flag": 1, "tran_flag": 1, "obm_flag": 0,
                    "pl_orbper": period, "pl_orbpererr1": 0.001, "pl_orbpererr2": -0.001,
                    "pl_bmassj": mass, "pl_bmassjerr1": up, "pl_bmassjerr2": down,
                    "pl_bmassjlim": 0, "pl_bmassprov": "Mass"})
        row.update(changes)
        return row

    def freeze(self, rows):
        raw = json.dumps(rows, separators=(",", ":")).encode()
        return acquire_external_evidence(self.review, opener=lambda *a, **k: Response(raw),
                                         retrieved_at="2026-01-01T00:00:00+00:00")

    def test_every_gate_field_is_exact(self):
        self.assertTrue(localization_gate(self.localization))
        changes = {"resultVersion": "old", "sourceAttributionResolved": False,
                   "classification": "OTHER", "recommendedNextTest": "OTHER",
                   "pixelDataChangedFrozenEventDefinition": True, "catalogAnswerKeyUsed": True,
                   "physicalMechanismResolved": True, "companionNatureResolved": True}
        for key, bad in changes.items():
            value = dict(self.localization); value[key] = bad
            self.assertFalse(localization_gate(value), key)

    def test_review_recomputes_independent_support_and_ignores_primary_and_ambiguous(self):
        self.assertEqual(3, self.review["supportingIndependentSectorCount"])
        value = dict(self.localization); value["sectorResults"] = list(self.localization["sectorResults"])
        value["sectorResults"][3] = {"sector": 4, "role": "INDEPENDENT", "usable": False,
                                     "classification": "AMBIGUOUS"}
        result = review_source_attribution(value)
        self.assertEqual("SOURCE_ATTRIBUTION_REVIEW_FAILED", result["classification"])
        self.assertEqual([4], result["ambiguousIndependentSectors"])

    def test_conflicting_source_fails_closed(self):
        value = json.loads(json.dumps(self.localization))
        value["sectorResults"].append({"sector": 5, "role": "INDEPENDENT", "usable": True,
                                       "classification": "CATALOG_CANDIDATE_CONSISTENT",
                                       "matchedCatalogHypothesis": "OTHER"})
        self.assertEqual("SOURCE_ATTRIBUTION_REVIEW_FAILED", review_source_attribution(value)["classification"])

    def test_off_target_catalog_and_off_catalog_are_distinct(self):
        value = json.loads(json.dumps(self.localization)); value["classification"] = "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE"
        self.assertEqual("OFF_TARGET_CATALOG_ATTRIBUTION_REVIEW_PASSED", review_source_attribution(value)["classification"])
        value["classification"] = "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE"
        value["attributedCatalogHypothesis"] = "OFF_CATALOG_SKY_CLUSTER"
        for sector in value["sectorResults"]: sector.update({"classification": "OFF_CATALOG", "matchedCatalogHypothesis": None})
        value["frozenCatalog"]["catalogHypotheses"] = []
        result = review_source_attribution(value)
        self.assertEqual("UNCATALOGUED_SOURCE_REQUIRES_FOLLOWUP", result["classification"])
        self.assertIsNone(result["attributedSource"])

    def test_freeze_exact_identity_hash_and_empty_success(self):
        frozen = self.freeze([])
        self.assertEqual([], frozen["returnedRows"])
        self.assertEqual("TIC 42", frozen["attributedSourceIdentifiers"]["ticID"])
        self.assertEqual(64, len(frozen["rawResponseSHA256"]))
        self.assertIn("default_flag+%3D+1", frozen["exactEncodedQuery"])
        self.assertEqual("NO_MATCHING_EXTERNAL_COMPANION_EVIDENCE",
                         interpret_external_evidence(frozen)["classification"])

    def test_malformed_and_schema_responses_fail_closed(self):
        for raw in (b"not json", b"{}", b"[{}]"):
            with self.assertRaises(ValueError):
                acquire_external_evidence(self.review, opener=lambda *a, raw=raw, **k: Response(raw))

    def test_identity_conflict_and_multiple_period_matches(self):
        conflict = self.row(tic_id="TIC 99")
        self.assertEqual("CONFLICTING_EXTERNAL_COMPANION_EVIDENCE",
                         interpret_external_evidence(self.freeze([conflict]))["classification"])
        self.assertEqual("MULTIPLE_MATCHING_EXTERNAL_COMPANIONS",
                         interpret_external_evidence(self.freeze([self.row(), self.row(mass=11)]))["classification"])

    def test_period_tolerance_boundary_and_mismatch(self):
        self.assertEqual("PERIOD_MATCHED_PLANETARY_MASS_COMPANION_SUPPORTED",
                         interpret_external_evidence(self.freeze([self.row(period=2.011)]))["classification"])
        self.assertEqual("NO_MATCHING_EXTERNAL_COMPANION_EVIDENCE",
                         interpret_external_evidence(self.freeze([self.row(period=2.01101)]))["classification"])

    def test_required_published_mass_evidence(self):
        for changes in ({"rv_flag": 0}, {"pl_controv_flag": 1}, {"pl_bmassj": None},
                        {"pl_bmassjerr1": None}, {"pl_bmassjerr2": None}, {"pl_bmassjlim": 1}):
            result = interpret_external_evidence(self.freeze([self.row(**changes)]))
            self.assertEqual("PERIOD_MATCHED_COMPANION_MASS_UNRESOLVED", result["classification"])

    def test_mass_regimes_and_boundaries(self):
        cases = [(10, 2, -1, "PLANETARY"), (13, 1, -1, None), (30, 2, -2, "BROWN_DWARF"),
                 (80, 2, -1, None), (82, 2, -2, "STELLAR")]
        for mass, up, down, expected in cases:
            result = interpret_external_evidence(self.freeze([self.row(mass, up, down)]))
            self.assertEqual(expected, result["supportedCompanionMassRegime"])
            self.assertFalse(result["physicalMechanismResolved"])
            self.assertFalse(result["companionNatureResolved"])
            self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_network_failure_is_retryable(self):
        def fail(*args, **kwargs): raise urllib.error.URLError("offline")
        with self.assertRaises(ExternalEvidenceTransientError):
            acquire_external_evidence(self.review, opener=fail)


if __name__ == "__main__": unittest.main()
