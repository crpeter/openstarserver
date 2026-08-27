import unittest

import numpy as np

from workflows.tess.tess_eclipse_event_localization import (
    authoritative_binary_gate,
    localize_eclipse_events,
)


class EclipseEventLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.binary = {
            "resultVersion": "2.0", "recommendedNextTest": "ECLIPSE_EVENT_SOURCE_LOCALIZATION",
            "catalogAnswerKeyUsed": False,
            "independentEvidence": {"classification": "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
                                    "supportingIndependentSectorCount": 3},
            "linearEphemeris": {"coherent": True, "primaryTimingConsistent": True,
                                "refinedPeriodDays": 2.0, "referenceEpoch": 0.5,
                                "cycleAssignments": []},
            "sectorResults": [
                {"sector": sector, "role": "PRIMARY" if sector == 1 else "INDEPENDENT",
                 "usable": True, "durationDays": 0.2}
                for sector in (1, 2, 3, 4)
            ],
        }
        self.identity = {"tic": {"metadata": {"raDeg": 1.0, "decDeg": 2.0}}}

    def sector(self, sector, source=(3.0, 3.0), weak=False, candidates=None):
        rng = np.random.default_rng(sector)
        times = np.linspace(0, 12, 1200)
        yy, xx = np.mgrid[:7, :7]
        psf = np.exp(-((xx-source[0])**2 + (yy-source[1])**2) / 1.2)
        phase = abs((times - 0.5 + 1.0) % 2.0 - 1.0)
        depth = 0.03 if weak else 4.0
        cube = rng.normal(0, 0.05, (len(times), 7, 7)) + 20 * psf
        cube[phase <= 0.1] -= depth * psf
        return {"sector": sector, "times": times, "fluxCube": cube,
                "targetPixel": {"x": 3.0, "y": 3.0},
                "catalogHypotheses": candidates or [
                    {"id": "TARGET", "isTarget": True, "pixel": {"x": 3.0, "y": 3.0}}]}

    def localize(self, inputs):
        return localize_eclipse_events(binary_confirmation=self.binary, identity=self.identity,
                                       tic_id=42, sector_inputs=inputs)

    def test_target_centered_eclipses_resolve_to_target(self):
        result = self.localize([self.sector(i) for i in (1, 2, 3, 4)])
        self.assertEqual("TARGET_CONSISTENT_ECLIPSE_SOURCE", result["classification"])

    def test_off_target_eclipses_resolve_to_frozen_candidate(self):
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 5, "y": 3}}]
        result = self.localize([self.sector(i, (5, 3), candidates=candidates) for i in (1, 2, 3, 4)])
        self.assertEqual("OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE", result["classification"])
        self.assertEqual("NEIGHBOR", result["attributedCatalogHypothesis"])

    def test_conflicting_independent_sectors_are_unresolved(self):
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 5, "y": 3}}]
        inputs = [self.sector(1, candidates=candidates), self.sector(2, candidates=candidates),
                  self.sector(3, candidates=candidates), self.sector(4, (5, 3), candidates=candidates)]
        self.assertEqual("CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND", self.localize(inputs)["classification"])

    def test_primary_cannot_rescue_two_independent_sectors(self):
        binary = dict(self.binary)
        binary["sectorResults"] = self.binary["sectorResults"][:3]
        result = localize_eclipse_events(binary_confirmation=binary, identity=self.identity, tic_id=42,
                                         sector_inputs=[self.sector(i) for i in (1, 2, 3)])
        self.assertFalse(result["sourceAttributionResolved"])

    def test_weak_images_remain_unresolved(self):
        result = self.localize([self.sector(i, weak=True) for i in (1, 2, 3, 4)])
        self.assertFalse(result["sourceAttributionResolved"])

    def test_overlapping_catalog_positions_fail_closed(self):
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "BLEND", "isTarget": False, "pixel": {"x": 3.1, "y": 3.1}}]
        result = self.localize([self.sector(i, candidates=candidates) for i in (1, 2, 3, 4)])
        self.assertFalse(result["sourceAttributionResolved"])

    def test_pixel_data_cannot_change_frozen_clock_or_duration(self):
        result = self.localize([self.sector(i) for i in (1, 2, 3, 4)])
        self.assertEqual(2.0, result["frozenEphemeris"]["refinedPeriodDays"])
        self.assertTrue(all(item["frozenMask"]["durationDays"] == 0.2
                            and not item["frozenMask"]["phaseOrDurationSearched"]
                            for item in result["sectorResults"]))

    def test_exact_gate_rejects_answer_key_or_wrong_recommendation(self):
        self.assertTrue(authoritative_binary_gate(self.binary))
        for key, value in (("catalogAnswerKeyUsed", True), ("recommendedNextTest", "OTHER")):
            changed = dict(self.binary); changed[key] = value
            self.assertFalse(authoritative_binary_gate(changed))


if __name__ == "__main__":
    unittest.main()
