import unittest

import numpy as np

from workflows.tess.tess_main_family_frequency_domain_reassessment import (
    FAMILY_SURVIVES, METHOD, ROTATION_MULTICYCLE, SAMPLING_ALIAS, UNRESOLVED,
    analyze_frequency_domain_reassessment,
)


class MainFamilyFrequencyReassessmentTests(unittest.TestCase):
    rotation = 3.600708338567666
    family = 7.546257528330875

    @staticmethod
    def sector(sector_id, frequency, time):
        return {"sectorID": sector_id, "time": time,
                "flux": np.sin(2 * np.pi * frequency * np.asarray(time))}

    def test_rotation_multicycle_structure_replicates(self):
        time = np.arange(0, 27, 0.02)
        frequency = 1 / (2 * self.rotation)
        result = analyze_frequency_domain_reassessment(
            [self.sector(i, frequency, time) for i in (94, 95)],
            rotation_period_days=self.rotation, family_period_days=self.family)
        self.assertEqual(ROTATION_MULTICYCLE, result["classification"])
        self.assertTrue(all(not row["separationEmpiricallyResolvable"]
                            for row in result["sectorResults"]))
        self.assertTrue(all(not row["samplingWindowAliasSupported"]
                            for row in result["sectorResults"]))

    def test_resolved_sampling_window_alias_is_measured_end_to_end(self):
        # Five-day visits make the +0.2 cycle/day window sidelobe deterministic.
        time = np.concatenate([visit + np.arange(0, .20, .01)
                               for visit in np.arange(0, 30, 5.0)])
        source_frequency = 0.14
        aliased_family_frequency = source_frequency + 0.20
        family_period = 1 / aliased_family_frequency
        sectors = [self.sector(i, source_frequency, time + offset)
                   for i, offset in ((94, 0.0), (95, .37))]
        result = analyze_frequency_domain_reassessment(
            sectors, rotation_period_days=1 / (2 * source_frequency),
            family_period_days=family_period)
        self.assertEqual(SAMPLING_ALIAS, result["classification"])
        for row in result["sectorResults"]:
            self.assertGreaterEqual(
                row["familyRotationSeparationResolutionUnits"],
                METHOD["supportIntervalHalfWidthResolutionMultiples"])
            self.assertGreaterEqual(row["samplingWindowResponse"],
                                    METHOD["minimumDirectWindowResponse"])
            self.assertTrue(row["samplingWindowAliasSupported"])
        one = analyze_frequency_domain_reassessment(
            sectors[:1], rotation_period_days=1 / (2 * source_frequency),
            family_period_days=family_period)
        self.assertEqual(UNRESOLVED, one["classification"])

    def test_frozen_family_injection_survives(self):
        time = np.arange(0, 27, .02)
        result = analyze_frequency_domain_reassessment(
            [self.sector(i, 1 / self.family, time) for i in (94, 95)],
            rotation_period_days=self.rotation, family_period_days=self.family)
        self.assertEqual(FAMILY_SURVIVES, result["classification"])

    def test_mixed_family_rotation_sectors_fail_closed(self):
        time = np.arange(0, 27, .02)
        result = analyze_frequency_domain_reassessment([
            self.sector(94, 1 / self.family, time),
            self.sector(95, 1 / (2 * self.rotation), time)],
            rotation_period_days=self.rotation, family_period_days=self.family)
        self.assertEqual(UNRESOLVED, result["classification"])
        self.assertFalse(result["noStrongerContradiction"])


if __name__ == "__main__":
    unittest.main()
