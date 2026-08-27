import math
import unittest

from openstar_investigation import sha256_json
from workflows.tess.tess_companion_evidence_synthesis import (
    HANDLER_ID, RESULT_VERSION, synthesize_companion_evidence,
)
from workflows.tess.tess_external_companion_evidence import (
    FIELDS, FREEZE_VERSION, interpret_external_evidence, review_source_attribution,
)


class CompanionEvidenceSynthesisTests(unittest.TestCase):
    def evidence(self, regime="PLANETARY", off_target=False):
        binary = {"resultVersion": "2.0", "catalogAnswerKeyUsed": False,
                  "independentEvidence": {"classification": "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED"}}
        source_id = "NEIGHBOR" if off_target else "TARGET"
        localization = {
            "resultVersion": "openstar.tess-eclipse-event-source-localization.v1",
            "classification": ("OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE" if off_target
                               else "TARGET_CONSISTENT_ECLIPSE_SOURCE"),
            "sourceAttributionResolved": True, "pixelDataChangedFrozenEventDefinition": False,
            "recommendedNextTest": "SOURCE_ATTRIBUTION_REVIEW",
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "attributedCatalogHypothesis": source_id, "catalogAnswerKeyUsed": False,
            "frozenEphemeris": {"refinedPeriodDays": 5.0},
            "binaryConfirmationSHA256": sha256_json(binary),
        }
        source = {"sourceID": source_id, "isTarget": not off_target,
                  "ticID": 12345, "gaiaDR3SourceID": "Gaia DR3 67890"}
        hypotheses = [source]
        if off_target:
            hypotheses.append({"sourceID": "TARGET", "isTarget": True,
                               "ticID": 99999, "gaiaDR3SourceID": "Gaia DR3 99998"})
        localization["frozenCatalog"] = {"catalogHypotheses": hypotheses}
        sector_class = "CATALOG_CANDIDATE_CONSISTENT" if off_target else "TARGET_CONSISTENT"
        localization["sectorResults"] = [
            {"sector": sector, "role": "INDEPENDENT", "usable": True,
             "classification": sector_class, "matchedCatalogHypothesis": source_id}
            for sector in (2, 3, 4)]
        review = review_source_attribution(localization)
        identifiers = {"ticID": "TIC 12345", "gaiaDR3SourceID": "Gaia DR3 67890"}
        masses = {"PLANETARY": (8.0, [7.0, 9.0]), "BROWN_DWARF": (40.0, [35.0, 45.0]),
                  "STELLAR": (100.0, [90.0, 110.0])}
        mass, interval = masses[regime]
        row = {field: None for field in FIELDS}
        row.update({"tic_id": "TIC 12345", "gaia_dr3_id": "Gaia DR3 67890",
                    "pl_orbper": 5.001, "pl_orbpererr1": 0.0, "pl_orbpererr2": 0.0,
                    "pl_bmassj": mass, "pl_bmassjerr2": interval[0] - mass,
                    "pl_bmassjerr1": interval[1] - mass, "default_flag": 1,
                    "pl_controv_flag": 0, "rv_flag": 1, "pl_bmassjlim": 0,
                    "pl_bmassprov": "Mass"})
        frozen = {"resultVersion": FREEZE_VERSION, "catalogAnswerKeyUsed": False,
                  "attributedSourceIdentifiers": identifiers, "returnedRows": [row],
                  "frozenEphemeris": localization["frozenEphemeris"],
                  "sourceAttributionReviewSHA256": sha256_json(review)}
        external = interpret_external_evidence(frozen)
        return [binary, localization, review, frozen, external]

    def refresh_after_localization(self, values):
        values[2] = review_source_attribution(values[1])
        values[3]["sourceAttributionReviewSHA256"] = sha256_json(values[2])
        values[4] = interpret_external_evidence(values[3])
        return values

    def test_target_planetary_happy_path(self):
        result = synthesize_companion_evidence(*self.evidence())
        self.assertEqual(RESULT_VERSION, result["resultVersion"])
        self.assertEqual("TARGET_ASSOCIATED_KNOWN_PLANETARY_COMPANION_SUPPORTED", result["classification"])
        self.assertTrue(result["companionNatureResolved"])
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertFalse(result["automaticDiscoveryClaim"])
        self.assertEqual("HUMAN_SCIENTIFIC_REVIEW", result["recommendedNextTest"])
        self.assertEqual(64, len(result["externalCompanionEvidenceSHA256"]))

    def test_all_mass_and_source_relationship_mappings(self):
        for regime in ("PLANETARY", "BROWN_DWARF", "STELLAR"):
            target = synthesize_companion_evidence(*self.evidence(regime))
            off = synthesize_companion_evidence(*self.evidence(regime, True))
            self.assertEqual(f"TARGET_ASSOCIATED_KNOWN_{regime}_COMPANION_SUPPORTED", target["classification"])
            self.assertEqual(f"OFF_TARGET_KNOWN_{regime}_COMPANION_IDENTIFIED", off["classification"])
            self.assertNotIn("TARGET_ASSOCIATED", off["classification"])

    def test_mutation_of_each_hash_link_is_rejected(self):
        for index, field in ((0, "extra"), (1, "extra"), (2, "extra"), (3, "extra")):
            values = self.evidence(); values[index][field] = True
            with self.assertRaises(ValueError, msg=str(index)):
                synthesize_companion_evidence(*values)
        values = self.evidence(); values[4]["externalMassJupiter"] = 8.5
        with self.assertRaises(ValueError): synthesize_companion_evidence(*values)

    def test_empty_freeze_and_row_absent_are_rejected(self):
        values = self.evidence(); values[3]["returnedRows"] = []
        values[4]["externalEvidenceFreezeSHA256"] = sha256_json(values[3])
        with self.assertRaises(ValueError): synthesize_companion_evidence(*values)
        values = self.evidence(); values[3]["returnedRows"] = [dict(values[3]["returnedRows"][0], tic_id="TIC 54321")]
        values[4]["externalEvidenceFreezeSHA256"] = sha256_json(values[3])
        with self.assertRaises(ValueError): synthesize_companion_evidence(*values)

    def test_coherent_external_mutation_and_impossible_regime_are_rejected(self):
        values = self.evidence()
        values[4]["externalMassJupiter"] = 100.0
        values[4]["externalMassIntervalJupiter"] = [90.0, 110.0]
        values[4]["selectedExternalRow"]["pl_bmassj"] = 100.0
        values[4]["selectedExternalRow"]["pl_bmassjerr2"] = -10.0
        values[4]["selectedExternalRow"]["pl_bmassjerr1"] = 10.0
        with self.assertRaises(ValueError): synthesize_companion_evidence(*values)

    def test_valid_tic_only_source_and_gaia_inconsistencies(self):
        values = self.evidence()
        source = values[1]["frozenCatalog"]["catalogHypotheses"][0]
        source["gaiaDR3SourceID"] = None
        values[3]["attributedSourceIdentifiers"]["gaiaDR3SourceID"] = None
        values[3]["returnedRows"][0]["gaia_dr3_id"] = None
        values = self.refresh_after_localization(values)
        result = synthesize_companion_evidence(*values)
        self.assertIsNone(result["attributedGaiaDR3SourceID"])
        for location, bad in (("freeze", "Gaia DR3 1"), ("source", "Gaia DR3 1")):
            broken = self.evidence()
            broken[1]["frozenCatalog"]["catalogHypotheses"][0]["gaiaDR3SourceID"] = None
            broken[3]["attributedSourceIdentifiers"]["gaiaDR3SourceID"] = None
            broken[3]["returnedRows"][0]["gaia_dr3_id"] = None
            broken = self.refresh_after_localization(broken)
            if location == "freeze":
                broken[3]["attributedSourceIdentifiers"]["gaiaDR3SourceID"] = bad
                broken[4]["externalEvidenceFreezeSHA256"] = sha256_json(broken[3])
            else:
                broken[2]["attributedSource"]["gaiaDR3SourceID"] = bad
                broken[3]["sourceAttributionReviewSHA256"] = sha256_json(broken[2])
                broken[4]["externalEvidenceFreezeSHA256"] = sha256_json(broken[3])
            with self.assertRaises(ValueError): synthesize_companion_evidence(*broken)

    def test_source_role_and_source_id_conflicts_are_rejected(self):
        for field, bad in (("isTarget", False), ("sourceID", "OTHER")):
            values = self.evidence(); values[2]["attributedSource"][field] = bad
            values[3]["sourceAttributionReviewSHA256"] = sha256_json(values[2])
            values[4]["externalEvidenceFreezeSHA256"] = sha256_json(values[3])
            with self.assertRaises(ValueError): synthesize_companion_evidence(*values)

    def test_answer_key_and_blind_flags_are_exact(self):
        for index in range(5):
            for bad in (True, None, 0):
                values = self.evidence(); values[index]["catalogAnswerKeyUsed"] = bad
                with self.assertRaises(ValueError): synthesize_companion_evidence(*values)
        values = self.evidence(); values[4]["softwareBlindPhotometricEvidencePreserved"] = 1
        with self.assertRaises(ValueError): synthesize_companion_evidence(*values)

    def test_unresolved_conflicting_multiple_and_mass_unresolved_rejected(self):
        for classification in ("NO_MATCHING_EXTERNAL_COMPANION_EVIDENCE",
                               "CONFLICTING_EXTERNAL_COMPANION_EVIDENCE",
                               "MULTIPLE_MATCHING_EXTERNAL_COMPANIONS",
                               "PERIOD_MATCHED_COMPANION_MASS_UNRESOLVED"):
            values = self.evidence(); values[4]["classification"] = classification
            values[4]["externalCompanionEvidenceResolved"] = False
            values[4]["externalEvidenceFreezeSHA256"] = sha256_json(values[3])
            with self.assertRaises(ValueError): synthesize_companion_evidence(*values)

    def test_classification_regime_identity_and_numeric_fail_closed(self):
        mutations = [
            (4, "classification", "PERIOD_MATCHED_STELLAR_MASS_COMPANION_SUPPORTED"),
            (4, "externalOrbitalPeriodDays", 0), (4, "externalOrbitalPeriodDays", math.inf),
            (4, "externalOrbitalPeriodDifferenceDays", -1),
            (4, "externalMassJupiter", math.nan), (4, "externalMassIntervalJupiter", [9, 7]),
        ]
        for index, key, bad in mutations:
            values = self.evidence(); values[index][key] = bad
            with self.assertRaises(ValueError, msg=key): synthesize_companion_evidence(*values)
        for where, key in ((2, "ticID"), (3, "ticID"), (4, "tic_id")):
            values = self.evidence()
            if where == 2: values[2]["attributedSource"][key] = "bad"
            elif where == 3: values[3]["attributedSourceIdentifiers"][key] = "bad"
            else: values[4]["selectedExternalRow"][key] = "bad"
            with self.assertRaises(ValueError): synthesize_companion_evidence(*values)

    def test_constants_are_stable(self):
        self.assertEqual("openstar.tess.final-companion-evidence-synthesis", HANDLER_ID)


if __name__ == "__main__":
    unittest.main()
