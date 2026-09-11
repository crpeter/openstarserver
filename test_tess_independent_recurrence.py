import unittest

from workflows.tess.tess_hypotheses import (
    interpret_independent_sectors,
    plan_independent_contradiction_resolution,
)


PRIMARY = 9.259243300072583


def interpretation(period, *, baseline=22.872, confidence_interval=None):
    dataset_id = "tess-blind-a-sector-28-independent-v1"
    return interpret_independent_sectors(
        target_period_days=PRIMARY,
        project_status={
            "datasets": [{
                "datasetID": dataset_id,
                "periodStatus": "RELIABLE",
                "periodConfidence": "high",
                "candidatePeriodDays": period,
                "candidateFrequency": 1.0 / period,
                "candidateFrequencyConfidenceInterval": confidence_interval,
            }]
        },
        independent_spec={
            "preparedSectors": [{
                "datasetID": dataset_id,
                "sector": 28,
                "baselineDays": baseline,
            }],
            "frequencySearch": {
                "minimumFrequency": (1.0 / PRIMARY) * 0.65,
                "maximumFrequency": (1.0 / PRIMARY) * 1.45,
                "frequencyStep": 0.00001,
            },
        },
    )


class IndependentRecurrenceTests(unittest.TestCase):
    def _mixed_interpretation(self, kinds):
        target_frequency = 1.0 / PRIMARY
        datasets = []
        prepared = []
        for sector, kind in enumerate(kinds, start=2):
            frequency = target_frequency if kind != "alternate" else 0.15
            dataset = {
                "datasetID": str(sector), "periodStatus": "RELIABLE",
                "periodConfidence": "high", "candidateFrequency": frequency,
                "candidatePeriodDays": 1.0 / frequency,
            }
            if kind == "uncertain":
                dataset["candidateFrequencyUncertaintyDiagnostics"] = {
                    "trustworthy": False, "unavailableReason": "competitive straddler",
                }
            else:
                dataset["candidateFrequencyConfidenceInterval"] = {
                    "lower": frequency - 0.0001, "upper": frequency + 0.0001,
                }
            datasets.append(dataset)
            prepared.append({"datasetID": str(sector), "sector": sector,
                             "baselineDays": 100.0})
        return interpret_independent_sectors(
            target_period_days=PRIMARY, project_status={"datasets": datasets},
            independent_spec={"preparedSectors": prepared, "frequencySearch": {
                "minimumFrequency": 0.04, "maximumFrequency": 0.2,
                "frequencyStep": 0.00001,
            }},
        )

    def test_three_uncertain_sectors_request_uncertainty_followup(self):
        result = self._mixed_interpretation(["support", "uncertain", "uncertain", "uncertain"])
        plan = plan_independent_contradiction_resolution(result)
        self.assertEqual("FREQUENCY_UNCERTAINTY_FOLLOWUP", plan["action"])
        self.assertEqual(plan["action"], plan["recommendedNextTest"])
        self.assertEqual(3, plan["uncertaintyLimitedSectorCount"])
        self.assertEqual(0, plan["alternateReliableSectorCount"])
        self.assertEqual(1, result["supportingSectorCount"])
        self.assertEqual(3, result["requiredSupportingSectorCount"])
        self.assertEqual("CANDIDATE_PERIOD", result["claimDecision"]["claim"])
        self.assertEqual("competitive straddler", result["sectorResults"][1][
            "candidateFrequencyUncertaintyDiagnostics"]["unavailableReason"])

    def test_two_real_alternates_still_request_broad_search(self):
        result = self._mixed_interpretation(["support", "alternate", "alternate", "uncertain"])
        plan = plan_independent_contradiction_resolution(result)
        self.assertEqual("BROAD_INDEPENDENT_SEARCH", plan["action"])
        self.assertEqual(2, plan["alternateReliableSectorCount"])
        self.assertEqual(1, plan["uncertaintyLimitedSectorCount"])

    def test_one_alternate_does_not_turn_uncertainty_into_broad_evidence(self):
        result = self._mixed_interpretation(["support", "alternate", "uncertain", "uncertain"])
        self.assertEqual("FREQUENCY_UNCERTAINTY_FOLLOWUP",
                         plan_independent_contradiction_resolution(result)["action"])

    def test_confirmed_recurrence_preserves_stop_route(self):
        result = self._mixed_interpretation(["support", "support", "support", "uncertain"])
        plan = plan_independent_contradiction_resolution(result)
        self.assertEqual("STOP", plan["action"])
        self.assertEqual("targeted-independent-recurrence-confirmed", plan["reason"])
        self.assertNotIn("recommendedNextTest", plan)

    def test_blind_a_periods_do_not_count_as_recurrence(self):
        result = interpretation(9.97870593974828)
        sector = result["sectorResults"][0]

        self.assertAlmostEqual(
            0.0720997937026939,
            sector["targetFrequencyRelativeError"],
            places=12,
        )
        self.assertTrue(sector["resolutionLimited"])
        self.assertFalse(sector["supportsTarget"])
        self.assertEqual("RESOLUTION_LIMITED", sector["recurrenceClassification"])
        self.assertEqual(1, result["resolutionLimitedSectorCount"])
        self.assertEqual("HUMAN_REVIEW_REQUIRED", result["claimDecision"]["claim"])

    def test_blind_a_narrow_local_interval_does_not_hide_resolution_limit(self):
        recovered_period = 9.97870593974828
        recovered_frequency = 1.0 / recovered_period
        result = interpretation(
            recovered_period,
            confidence_interval={
                "lower": recovered_frequency - 0.0001,
                "upper": recovered_frequency + 0.0001,
                "method": "sinusoid-profile-likelihood",
                "confidenceLevel": 0.95,
            },
        )
        sector = result["sectorResults"][0]
        self.assertEqual("RESOLUTION_LIMITED", sector["recurrenceClassification"])
        self.assertFalse(sector["supportsTarget"])

    def test_well_resolved_close_recurrence_passes(self):
        target_frequency = 1.0 / PRIMARY
        result = interpretation(
            PRIMARY * 1.001,
            baseline=1000.0,
            confidence_interval={
                "lower": target_frequency - 0.0002,
                "upper": target_frequency + 0.0002,
            },
        )
        self.assertTrue(result["sectorResults"][0]["supportsTarget"])
        self.assertEqual("INDEPENDENT_PERIOD_ESTIMATE", result["claimDecision"]["claim"])

    def test_well_resolved_different_period_is_nonsupporting(self):
        frequency = 1.0 / 8.0
        result = interpretation(
            8.0,
            baseline=1000.0,
            confidence_interval={
                "lower": frequency - 0.0002,
                "upper": frequency + 0.0002,
            },
        )
        sector = result["sectorResults"][0]
        self.assertEqual("NONSUPPORTING", sector["recurrenceClassification"])
        self.assertFalse(sector["supportsTarget"])

    def test_harmonic_is_not_silently_accepted(self):
        result = interpretation(PRIMARY * 2, baseline=40.0)
        sector = result["sectorResults"][0]
        self.assertFalse(sector["harmonicOrAliasAccepted"])
        self.assertFalse(sector["supportsTarget"])

    def test_short_baseline_nearby_peak_is_not_affirmative_evidence(self):
        result = interpretation(PRIMARY * 1.001, baseline=22.872)
        sector = result["sectorResults"][0]
        self.assertEqual("RESOLUTION_LIMITED", sector["recurrenceClassification"])
        self.assertFalse(sector["supportsTarget"])

    def test_unrelated_period_fails(self):
        result = interpretation(6.5)
        self.assertFalse(result["sectorResults"][0]["supportsTarget"])

    def test_interpretation_is_idempotent(self):
        first = interpretation(PRIMARY * 1.02)
        restarted = interpretation(PRIMARY * 1.02)
        self.assertEqual(first, restarted)


if __name__ == "__main__":
    unittest.main()
