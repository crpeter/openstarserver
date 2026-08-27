import math
import unittest

from openstar_investigation import sha256_json
from workflows.tess.tess_companion_evidence_synthesis import (
    HANDLER_ID, RESULT_VERSION, synthesize_companion_evidence,
)
from workflows.tess.tess_external_companion_evidence import (
    FREEZE_VERSION, RESULT_VERSION as EXTERNAL_VERSION, REVIEW_VERSION,
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
            "attributedCatalogHypothesis": source_id, "catalogAnswerKeyUsed": False,
            "frozenEphemeris": {"refinedPeriodDays": 5.0},
            "binaryConfirmationSHA256": sha256_json(binary),
        }
        source = {"sourceID": source_id, "isTarget": not off_target,
                  "ticID": 12345, "gaiaDR3SourceID": "Gaia DR3 67890"}
        review = {
            "resultVersion": REVIEW_VERSION,
            "classification": ("OFF_TARGET_CATALOG_ATTRIBUTION_REVIEW_PASSED" if off_target
                               else "TARGET_SOURCE_ATTRIBUTION_REVIEW_PASSED"),
            "sourceAttributionReviewPassed": True, "supportingIndependentSectors": [2, 3, 4],
            "supportingIndependentSectorCount": 3, "primarySectorCanSatisfyReplication": False,
            "conflictingIndependentSectors": [], "duplicateIndependentSectors": [],
            "attributedSource": source, "attributedCatalogHypothesis": source_id,
            "sourceLocalizationSHA256": sha256_json(localization),
            "binaryConfirmationSHA256": sha256_json(binary),
            "frozenEphemeris": localization["frozenEphemeris"], "catalogAnswerKeyUsed": False,
        }
        identifiers = {"ticID": "TIC 12345", "gaiaDR3SourceID": "Gaia DR3 67890"}
        frozen = {"resultVersion": FREEZE_VERSION, "catalogAnswerKeyUsed": False,
                  "attributedSourceIdentifiers": identifiers,
                  "frozenEphemeris": localization["frozenEphemeris"],
                  "sourceAttributionReviewSHA256": sha256_json(review)}
        masses = {"PLANETARY": (8.0, [7.0, 9.0]), "BROWN_DWARF": (40.0, [35.0, 45.0]),
                  "STELLAR": (100.0, [90.0, 110.0])}
        mass, interval = masses[regime]
        row = {"tic_id": "TIC 12345", "gaia_dr3_id": "Gaia DR3 67890",
               "pl_orbper": 5.001, "pl_bmassj": mass,
               "pl_bmassjerr2": interval[0] - mass, "pl_bmassjerr1": interval[1] - mass}
        external = {
            "resultVersion": EXTERNAL_VERSION,
            "classification": f"PERIOD_MATCHED_{regime}_MASS_COMPANION_SUPPORTED",
            "externalCompanionEvidenceResolved": True, "supportedCompanionMassRegime": regime,
            "externalOrbitalPeriodDays": 5.001, "externalOrbitalPeriodDifferenceDays": .001,
            "externalMassJupiter": mass, "externalMassIntervalJupiter": interval,
            "selectedExternalRow": row, "externalKnownObjectCatalogUsed": True,
            "softwareBlindPhotometricEvidencePreserved": True, "catalogAnswerKeyUsed": False,
            "recommendedNextTest": "FINAL_COMPANION_EVIDENCE_SYNTHESIS",
            "externalEvidenceFreezeSHA256": sha256_json(frozen),
        }
        return [binary, localization, review, frozen, external]

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
