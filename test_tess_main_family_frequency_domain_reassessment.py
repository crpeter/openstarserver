import math
import unittest

import numpy as np

from workflows.tess.tess_main_family_frequency_domain_reassessment import (
    FAMILY_SURVIVES, METHOD, ROTATION_MULTICYCLE, UNRESOLVED, WINDOW_ALIAS,
    analyze_frequency_domain_reassessment, analyze_sector, combine_sector_results,
)


ROTATION = 3.600708338567666
FAMILY = 7.546257528330875
DOUBLE = 15.09251505666175


class MainFamilyFrequencyReassessmentTests(unittest.TestCase):
    def series(self, period, sector, irregular=False):
        time = np.linspace(0, 27, 1200)
        if irregular:
            time = time[(np.arange(len(time)) % 5) != 0]
        flux = np.sin(2 * np.pi * time / period) + 0.15 * np.cos(4 * np.pi * time / period)
        return {"sectorID": sector, "time": time, "flux": flux}

    def alias_series(self, sector, time_offset=0.0):
        # Preregistered synthetic alias: sampling frequency = 0.07 cycles/day,
        # source = 0.23 c/d, frozen family = 0.37 c/d. The 0.14 c/d
        # displacement and its 0.07 c/d half-frequency counterpart are exact
        # spectral-window peaks and are many Rayleigh elements apart.
        sampling_frequency = 0.07
        source_frequency = 0.23
        time = time_offset + np.arange(20) / sampling_frequency
        flux = np.sin(2 * np.pi * source_frequency * time + 0.37)
        return {"sectorID": sector, "time": time, "flux": flux}

    def prior(self):
        return {"classification": "FREQUENCY_FAMILY_NOT_TIME_DOMAIN_REPLICATED",
            "rawFamilyRecurrenceSectorIDs": [], "possibleDoubleRecurrenceSectorIDs": [],
            "rawFamilyCoverageSectorIDs": [95, 94, 68, 67],
            "possibleDoubleCoverageSectorIDs": [95, 94, 68, 67]}

    def analyze(self, sectors):
        return analyze_frequency_domain_reassessment(sectors,
            rotation_period_days=ROTATION, family_period_days=FAMILY,
            possible_double_days=DOUBLE, prior_time_domain=self.prior())

    def test_rotation_multicycle_structure_replicates(self):
        result = self.analyze([self.series(2 * ROTATION, 95), self.series(2 * ROTATION, 94)])
        self.assertEqual(ROTATION_MULTICYCLE, result["classification"])
        self.assertFalse(result["physicalCycleResolved"])
        self.assertFalse(result["sectorResults"][0]["samplingWindowAliasSupported"])
        self.assertFalse(result["sectorResults"][0]["samplingWindowDiagnostics"]
            ["frozenFamily"]["separationPassesResolutionGate"])

    def test_distinct_frozen_family_survives_frequency_only(self):
        result = self.analyze([self.series(FAMILY, 95), self.series(FAMILY, 94)])
        self.assertEqual(FAMILY_SURVIVES, result["classification"])
        self.assertFalse(result["exactPhysicalCycleResolved"])

    def test_prior_negative_evidence_is_carried_verbatim(self):
        result = self.analyze([self.series(FAMILY, 95), self.series(FAMILY, 94)])
        self.assertEqual(self.prior()["classification"], result["priorTimeDomainClassification"])
        self.assertEqual([], result["priorTimeDomainRawFamilyRecurrenceSectorIDs"])
        self.assertEqual([95, 94, 68, 67], result["priorTimeDomainPossibleDoubleCoverageSectorIDs"])

    def test_resolution_and_all_fixed_hypotheses_are_persisted(self):
        data = self.series(FAMILY, 95)
        result = analyze_sector(data["time"], data["flux"], sector_id=95, rotation_period_days=ROTATION,
            family_period_days=FAMILY, possible_double_days=DOUBLE)
        self.assertAlmostEqual(1 / result["baselineDays"], result["frequencyResolutionCyclesPerDay"])
        self.assertEqual(5, len(result["fixedHypothesisFits"]))
        self.assertEqual(1200, result["sampleCount"])

    def test_mixed_sector_preferences_fail_closed(self):
        result = self.analyze([self.series(FAMILY, 95), self.series(2 * ROTATION, 94)])
        self.assertEqual(UNRESOLVED, result["classification"])

    def test_insufficient_replication_fails_closed(self):
        result = self.analyze([self.series(FAMILY, 95)])
        self.assertEqual(UNRESOLVED, result["classification"])

    def test_direct_window_alias_requires_replication(self):
        base = {"sectorPreference": "ROTATION_MULTICYCLE", "samplingWindowAliasSupported": True}
        result = combine_sector_results([{**base, "sectorID": 95}, {**base, "sectorID": 94}])
        self.assertEqual(WINDOW_ALIAS, result["classification"])
        result = combine_sector_results([{**base, "sectorID": 95}])
        self.assertEqual(UNRESOLVED, result["classification"])

    def test_resolved_sampling_window_alias_is_detected_end_to_end(self):
        source_frequency = 0.23
        family_frequency = 0.37
        rotation_period = 1.0 / (2.0 * source_frequency)
        family_period = 1.0 / family_frequency
        possible_double = 2.0 * family_period
        sectors = [self.alias_series(95), self.alias_series(94, time_offset=1.9)]

        result = analyze_frequency_domain_reassessment(sectors,
            rotation_period_days=rotation_period, family_period_days=family_period,
            possible_double_days=possible_double, prior_time_domain=self.prior())
        self.assertEqual(WINDOW_ALIAS, result["classification"])
        for sector in result["sectorResults"]:
            self.assertTrue(sector["samplingWindowAliasSupported"])
            for diagnostic in sector["samplingWindowDiagnostics"].values():
                self.assertTrue(diagnostic["separationPassesResolutionGate"])
                self.assertGreaterEqual(diagnostic["separationInResolutionUnits"],
                    METHOD["supportIntervalHalfWidthResolutionMultiples"])
                self.assertGreaterEqual(diagnostic["samplingWindowResponse"],
                    METHOD["minimumDirectWindowResponse"])
                self.assertTrue(diagnostic["directWindowSupport"])

        single = analyze_frequency_domain_reassessment(sectors[:1],
            rotation_period_days=rotation_period, family_period_days=family_period,
            possible_double_days=possible_double, prior_time_domain=self.prior())
        self.assertEqual(UNRESOLVED, single["classification"])

    def test_window_diagnostic_uses_actual_sampling(self):
        first, second = self.series(FAMILY, 95), self.series(FAMILY, 94, True)
        regular = analyze_sector(first["time"], first["flux"], sector_id=95, rotation_period_days=ROTATION,
            family_period_days=FAMILY, possible_double_days=DOUBLE)
        irregular = analyze_sector(second["time"], second["flux"], sector_id=94, rotation_period_days=ROTATION,
            family_period_days=FAMILY, possible_double_days=DOUBLE)
        a = regular["samplingWindowDiagnostics"]["frozenFamily"]["samplingWindowResponse"]
        b = irregular["samplingWindowDiagnostics"]["frozenFamily"]["samplingWindowResponse"]
        self.assertNotEqual(a, b)

    def test_method_is_preregistered_and_no_search_grid_exists(self):
        self.assertEqual(2, METHOD["minimumReplicatedSectorCount"])
        self.assertNotIn("frequencyGrid", METHOD)


if __name__ == "__main__":
    unittest.main()
