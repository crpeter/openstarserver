import math
import unittest
from pathlib import Path

from workflows.tess.tess_event_depth_accuracy import (
    AUDIT_HANDLER_ID, FREEZE_HANDLER_ID, _measure, _protected_harmonic,
    audit_depth_attenuation, freeze_photometry, unresolved_freeze,
)


class TessEventDepthAccuracyTests(unittest.TestCase):
    def product(self, sector, *, cadence=0.01, noise=0.0001, depth=0.01,
                duration=0.12, phase_amplitude=0.003, offset=0.0):
        times = [sector * 20 + i * cadence for i in range(int(12 / cadence))]
        flux = []
        for i, time in enumerate(times):
            primary = abs((time + 1) % 2 - 1) <= duration / 2
            opposite = abs((time - 1 + 1) % 2 - 1) <= duration / 2
            value = 1000 * (1 + offset + phase_amplitude * math.sin(math.pi * time)
                            - depth * primary - depth * 0.25 * opposite
                            + noise * math.sin(i * 1.618))
            flux.append(value)
        return {"sector": sector, "time": times, "flux": flux, "cadenceSeconds": cadence * 86400,
                "author": "SYNTHETIC_OFFICIAL_POLICY", "productIdentity": f"generic-{sector}",
                "sourceProductProvenance": {"local": True}, "fluxColumn": "SAP_FLUX",
                "fluxUnits": "electron / s", "qualityMaskPolicy": "default",
                "normalization": "DIVIDE_BY_MEDIAN"}

    def binary(self, sectors=(1, 2, 3)):
        return {"linearEphemeris": {"coherent": True, "referenceEpoch": 20.0,
                                    "refinedPeriodDays": 2.0, "timingSectors": list(sectors)},
                "sectorResults": [{"usable": True, "dutyCycle": 0.06} for _ in sectors]}

    def test_freeze_preserves_float64_samples_provenance_and_hashes(self):
        products = [self.product(i) for i in (1, 2, 3)]
        result = freeze_photometry(products, [1, 2, 3], before_external_known_object_query=True)
        self.assertEqual("FROZEN", result["status"])
        self.assertTrue(result["fullFiniteCadencePreserved"])
        self.assertTrue(result["frozenBeforeExternalKnownObjectQuery"])
        self.assertFalse(result["externalCatalogInformationUsed"])
        self.assertFalse(result["catalogAnswerKeyUsed"])
        self.assertEqual(len(products[0]["time"]), result["sectors"][0]["sampleCount"])
        original_hash = result["sectors"][0]["frozenInputSHA256"]
        products[0]["flux"][0] += 1
        repeated = freeze_photometry(products, [1, 2, 3], before_external_known_object_query=True)
        self.assertNotEqual(original_hash, repeated["sectors"][0]["frozenInputSHA256"])

    def test_freeze_fails_closed_on_missing_provenance_and_wrong_chronology(self):
        bad = self.product(1); bad.pop("productIdentity")
        with self.assertRaises(ValueError): freeze_photometry([bad], [1], before_external_known_object_query=True)
        with self.assertRaises(ValueError): freeze_photometry([self.product(1)], [1], before_external_known_object_query=False)

    def test_generic_injection_recovers_full_depth_and_separates_transformations(self):
        frozen = freeze_photometry([self.product(i) for i in (1, 2, 3)], [1, 2, 3],
                                   before_external_known_object_query=True)
        result = audit_depth_attenuation(frozen, self.binary(), downsampling_cap=10000)
        self.assertEqual("COMPLETE", result["status"])
        self.assertTrue(result["suitableForLaterPrecisionModeling"])
        for sector in result["sectorResults"]:
            self.assertAlmostEqual(.01, sector["fullPrecisionLocalBaseline"]["depthFractionalFlux"], delta=.002)
            self.assertAlmostEqual(0, sector["attenuationFractions"]["downsampling"], delta=.01)
            self.assertAlmostEqual(0, sector["attenuationFractions"]["standardizationFloat32"], delta=.001)
            self.assertTrue(sector["primaryAndOppositeConjunctionProtected"])

    def test_downsampling_duration_and_combined_attenuation_are_visible(self):
        frozen = freeze_photometry([self.product(i, cadence=.002) for i in (1, 2, 3)], [1, 2, 3],
                                   before_external_known_object_query=True)
        result = audit_depth_attenuation(frozen, self.binary(), downsampling_cap=35)
        self.assertTrue(any(abs(x["attenuationFractions"]["downsampling"] or 0) > .02
                            for x in result["sectorResults"]))
        self.assertTrue(any(abs(x["attenuationFractions"]["discreteBoxDuration"] or 0) > .02
                            for x in result["sectorResults"]))

    def test_masked_and_unmasked_harmonic_fit_demonstrate_suppression(self):
        item = self.product(1, phase_amplitude=.01)
        times = item["time"]; scale = sorted(item["flux"])[len(item["flux"]) // 2]
        flux = [x / scale for x in item["flux"]]
        masks = [abs((x + 1) % 2 - 1) <= .12 for x in times]
        protected = _protected_harmonic(times, flux, 2.0, masks)
        unprotected = _protected_harmonic(times, flux, 2.0, [False] * len(times))
        raw_depth = _measure(times, flux, 20, 2, .12)["depthFractionalFlux"]
        protected_depth = _measure(times, protected, 20, 2, .12)["depthFractionalFlux"]
        unprotected_depth = _measure(times, unprotected, 20, 2, .12)["depthFractionalFlux"]
        self.assertAlmostEqual(.01, protected_depth, delta=.001)
        self.assertGreater(raw_depth, protected_depth)  # smooth orbital variation contaminated raw baseline
        self.assertLess(unprotected_depth, protected_depth)

    def test_cadence_integration_sector_baselines_noise_and_opposite_event(self):
        products = [self.product(1, cadence=.02, noise=.0002, offset=.02),
                    self.product(2, cadence=.01, noise=.0005, offset=-.01),
                    self.product(3, cadence=.005, noise=.001)]
        frozen = freeze_photometry(products, [1, 2, 3], before_external_known_object_query=True)
        result = audit_depth_attenuation(frozen, self.binary())
        self.assertEqual(3, len(result["sectorResults"]))
        self.assertEqual(3, result["crossSectorRobustSummary"]["downsampling"]["sectorCount"])
        self.assertEqual(3, len({x["sampleCounts"]["full"] for x in result["sectorResults"]}))

    def test_incoherent_insufficient_and_malformed_inputs_are_unresolved(self):
        unresolved = audit_depth_attenuation(unresolved_freeze(["missing"]), self.binary())
        self.assertEqual("UNRESOLVED", unresolved["status"])
        frozen = freeze_photometry([self.product(1)], [1], before_external_known_object_query=True)
        binary = self.binary((1,)); binary["linearEphemeris"]["coherent"] = False
        self.assertEqual("UNRESOLVED", audit_depth_attenuation(frozen, binary)["status"])
        with self.assertRaises(ValueError): freeze_photometry([], [1], before_external_known_object_query=True)

    def test_registered_identifiers_and_diagnostic_contract(self):
        self.assertEqual("openstar.tess.event-depth-photometry.freeze", FREEZE_HANDLER_ID)
        self.assertEqual("openstar.tess.event-depth-attenuation.audit", AUDIT_HANDLER_ID)

    def test_lifecycle_order_is_before_external_evidence_and_module_has_no_answer_key(self):
        source = Path("workflows/tess/tess_investigation.py").read_text(encoding="utf-8")
        block = source[source.index("def source_attribution_review_stage"):
                       source.index("def external_evidence_interpret_stage")]
        self.assertLess(block.index("EVENT_DEPTH_FREEZE_HANDLER_ID"),
                        block.index("def event_depth_attenuation_audit_stage"))
        self.assertLess(block.index("def event_depth_attenuation_audit_stage"),
                        block.index("def external_evidence_freeze_stage"))
        module = Path("workflows/tess/tess_event_depth_accuracy.py").read_text(encoding="utf-8")
        self.assertNotIn("exoplanetarchive", module.lower())
        self.assertNotIn("wasp", module.lower())


if __name__ == "__main__": unittest.main()
