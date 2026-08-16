import math
import random
import unittest
from unittest import mock

import frequency_uncertainty
from frequency_uncertainty import estimate_frequency_interval
from workflows.tess.tess_hypotheses import interpret_independent_sectors


def signal_dataset(*, frequency, baseline, count=240, noise=0.08, seed=7):
    randomizer = random.Random(seed)
    times = [baseline * index / (count - 1) for index in range(count)]
    values = [
        math.sin(2.0 * math.pi * frequency * time + 0.3)
        + randomizer.gauss(0.0, noise)
        for time in times
    ]
    return {
        "times": times,
        "flux": values,
        "frequencySearch": {
            "minimumFrequency": 0.04,
            "maximumFrequency": 0.2,
            "frequencyStep": 0.0002,
            "totalFrequencies": 801,
        },
    }


class FrequencyUncertaintyTests(unittest.TestCase):
    def test_long_clean_signal_has_narrow_interval_containing_truth(self):
        truth = 0.108
        interval, diagnostics = estimate_frequency_interval(
            signal_dataset(frequency=truth, baseline=300.0), truth
        )
        self.assertTrue(diagnostics["trustworthy"])
        self.assertEqual(
            "profiled-global-residual-scale",
            diagnostics["noiseScaleTreatment"],
        )
        self.assertLess(interval["lower"], truth)
        self.assertGreater(interval["upper"], truth)
        self.assertLess(interval["upper"] - interval["lower"], 1.0 / 300.0)

    def test_short_baseline_interval_is_not_resolved(self):
        truth = 0.108
        interval, _ = estimate_frequency_interval(
            signal_dataset(frequency=truth, baseline=12.0, noise=0.25), truth
        )
        self.assertIsNotNone(interval)
        self.assertGreater(interval["upper"] - interval["lower"], 0.001)

    def test_noise_widens_interval(self):
        truth = 0.108
        clean, _ = estimate_frequency_interval(
            signal_dataset(frequency=truth, baseline=100.0, noise=0.03), truth
        )
        noisy, _ = estimate_frequency_interval(
            signal_dataset(frequency=truth, baseline=100.0, noise=0.5), truth
        )
        self.assertGreater(
            noisy["upper"] - noisy["lower"], clean["upper"] - clean["lower"]
        )

    def test_known_heteroscedastic_sigmas_use_absolute_noise_scale(self):
        truth = 0.108
        dataset = signal_dataset(frequency=truth, baseline=180.0, noise=0.0)
        sigmas = [0.04 if index % 3 else 0.12 for index in range(len(dataset["times"]))]
        randomizer = random.Random(31)
        dataset["flux"] = [
            math.sin(2.0 * math.pi * truth * time + 0.3)
            + randomizer.gauss(0.0, sigma)
            for time, sigma in zip(dataset["times"], sigmas)
        ]
        dataset["measurementUncertainties"] = sigmas

        interval, diagnostics = estimate_frequency_interval(dataset, truth)

        self.assertTrue(diagnostics["trustworthy"])
        self.assertEqual(
            "known-per-observation-standard-deviations",
            diagnostics["noiseScaleTreatment"],
        )
        self.assertLess(interval["lower"], truth)
        self.assertGreater(interval["upper"], truth)

    def test_known_sigma_magnitude_changes_interval(self):
        truth = 0.108
        dataset = signal_dataset(frequency=truth, baseline=180.0, noise=0.03)
        dataset["measurementUncertainties"] = [0.03] * len(dataset["times"])
        precise, _ = estimate_frequency_interval(dataset, truth)
        dataset["measurementUncertainties"] = [0.15] * len(dataset["times"])
        imprecise, _ = estimate_frequency_interval(dataset, truth)
        self.assertGreater(
            imprecise["upper"] - imprecise["lower"],
            precise["upper"] - precise["lower"],
        )

    def test_invalid_sigmas_refuse_interval(self):
        dataset = signal_dataset(frequency=0.108, baseline=180.0)
        dataset["measurementUncertainties"] = [0.08] * len(dataset["times"])
        dataset["measurementUncertainties"][4] = 0.0
        interval, diagnostics = estimate_frequency_interval(dataset, 0.108)
        self.assertIsNone(interval)
        self.assertEqual(
            "measurement uncertainties are invalid",
            diagnostics["unavailableReason"],
        )

    def test_competing_likelihood_mode_suppresses_interval(self):
        dataset = signal_dataset(frequency=0.108, baseline=200.0)
        # An identical-frequency duplicate is deliberately separated in the
        # timestamps by integer sampling, producing an exact sampling alias.
        dataset["times"] = [float(index) for index in range(240)]
        randomizer = random.Random(19)
        dataset["flux"] = [
            math.sin(2.0 * math.pi * 0.108 * time + 0.3)
            + randomizer.gauss(0.0, 0.03)
            for time in dataset["times"]
        ]
        dataset["frequencySearch"]["maximumFrequency"] = 1.2
        dataset["frequencySearch"]["frequencyStep"] = 0.002
        dataset["frequencySearch"]["totalFrequencies"] = 581
        coverage = {
            "complete": True,
            "objectiveMatches": True,
            "selectedPower": 0.99,
            "chunks": [
                {"frequency": 0.108, "power": 0.99, "startFrequency": 0.107, "endFrequency": 0.109},
                {"frequency": 1.108, "power": 0.99, "startFrequency": 1.107, "endFrequency": 1.109},
            ],
        }
        interval, diagnostics = estimate_frequency_interval(
            dataset, 0.108, competing_mode_coverage=coverage
        )
        self.assertIsNone(interval)
        self.assertFalse(diagnostics["trustworthy"])
        self.assertTrue(any(
            math.isclose(item, 1.108, abs_tol=0.0001)
            for item in diagnostics["plausibleCompetingFrequencies"]
        ))
        self.assertEqual(
            "all-distributed-chunk-maxima",
            diagnostics["competingModeSearch"],
        )

    def test_realistic_grid_uses_only_distributed_chunk_winners(self):
        total = 262144
        per_work = 4096
        step = 0.0000005
        minimum = 0.04
        dataset = signal_dataset(frequency=0.108, baseline=300.0)
        dataset["frequencySearch"] = {
            "minimumFrequency": minimum,
            "frequencyStep": step,
            "totalFrequencies": total,
            "frequenciesPerWorkUnit": per_work,
        }
        chunks = []
        for start_index in range(0, total, per_work):
            count = min(per_work, total - start_index)
            start = minimum + start_index * step
            end = start + (count - 1) * step
            if start <= 0.108 <= end:
                winner, power = 0.108, 0.99
            else:
                winner, power = (start + end) / 2.0, 0.1
                rayleigh = 1.0 / 300.0
                if start < 0.108 - rayleigh < end:
                    winner = start
                elif start < 0.108 + rayleigh < end:
                    winner = end
            chunks.append({
                "frequency": winner,
                "power": power,
                "startFrequency": start,
                "endFrequency": end,
            })

        original = frequency_uncertainty._sinusoid_rss
        with mock.patch(
            "frequency_uncertainty._sinusoid_rss", wraps=original
        ) as evaluated:
            interval, diagnostics = estimate_frequency_interval(
                dataset,
                0.108,
                competing_mode_coverage={
                    "complete": True,
                    "objectiveMatches": True,
                    "selectedPower": 0.99,
                    "chunks": chunks,
                },
            )

        self.assertIsNotNone(interval)
        self.assertEqual(64, diagnostics["competingModeChunkCount"])
        self.assertTrue(diagnostics["competingModeCoverageSufficient"])
        self.assertEqual(64, diagnostics["rawChunkCandidatesInspected"])
        self.assertEqual(0, diagnostics["competingChunksRefined"])
        self.assertLess(evaluated.call_count, 300)
        self.assertEqual(
            evaluated.call_count,
            diagnostics["profileModelEvaluations"],
        )

    def test_coarse_chunk_coverage_refuses_interval(self):
        dataset = signal_dataset(frequency=0.108, baseline=300.0)
        interval, diagnostics = estimate_frequency_interval(
            dataset,
            0.108,
            competing_mode_coverage={
                "complete": True,
                "objectiveMatches": True,
                "selectedPower": 0.99,
                "chunks": [{
                    "frequency": 0.108,
                    "power": 0.99,
                    "startFrequency": 0.04,
                    "endFrequency": 0.2,
                }],
            },
        )
        self.assertIsNone(interval)
        self.assertFalse(diagnostics["competingModeCoverageSufficient"])

    def test_boundary_straddling_chunk_refuses_interval(self):
        dataset = signal_dataset(frequency=0.108, baseline=300.0)
        rayleigh = 1.0 / 300.0
        interval, diagnostics = estimate_frequency_interval(
            dataset,
            0.108,
            competing_mode_coverage={
                "complete": True,
                "objectiveMatches": True,
                "selectedPower": 0.99,
                "chunks": [{
                    "frequency": 0.108 - rayleigh * 0.9,
                    "power": 0.98,
                    "startFrequency": 0.108 - rayleigh * 1.05,
                    "endFrequency": 0.108 - rayleigh * 0.1,
                }],
            },
        )
        self.assertIsNone(interval)
        self.assertEqual(1, diagnostics["boundaryStraddlingChunkCount"])
        self.assertIn("outside the Rayleigh", diagnostics["unavailableReason"])

    def test_independent_realization_can_support_and_different_one_does_not(self):
        primary_period = 1.0 / 0.108
        interval, _ = estimate_frequency_interval(
            signal_dataset(frequency=0.108, baseline=300.0, seed=22), 0.108
        )
        base = {
            "preparedSectors": [{"datasetID": "independent", "baselineDays": 300.0}],
            "frequencySearch": {
                "minimumFrequency": 0.04,
                "maximumFrequency": 0.2,
                "frequencyStep": 0.00001,
            },
        }
        supporting = interpret_independent_sectors(
            target_period_days=primary_period,
            project_status={"datasets": [{
                "datasetID": "independent", "periodStatus": "RELIABLE",
                "periodConfidence": "high", "candidatePeriodDays": primary_period,
                "candidateFrequency": 0.108,
                "candidateFrequencyConfidenceInterval": interval,
            }]},
            independent_spec=base,
        )
        self.assertEqual("SUPPORTING", supporting["sectorResults"][0]["recurrenceClassification"])

        different_frequency = 0.14
        different_interval, _ = estimate_frequency_interval(
            signal_dataset(frequency=different_frequency, baseline=300.0),
            different_frequency,
        )
        nonsupporting = interpret_independent_sectors(
            target_period_days=primary_period,
            project_status={"datasets": [{
                "datasetID": "independent", "periodStatus": "RELIABLE",
                "periodConfidence": "high",
                "candidatePeriodDays": 1.0 / different_frequency,
                "candidateFrequency": different_frequency,
                "candidateFrequencyConfidenceInterval": different_interval,
            }]},
            independent_spec=base,
        )
        self.assertEqual("NONSUPPORTING", nonsupporting["sectorResults"][0]["recurrenceClassification"])


if __name__ == "__main__":
    unittest.main()
