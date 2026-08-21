import math
import random
import time
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


def benchmark_realistic_frequency_interval(sample_count=18000):
    """Run the optimized profile on a TESS-sized series and return wall time."""
    dataset = signal_dataset(
        frequency=0.108,
        baseline=27.0,
        count=sample_count,
        noise=0.08,
    )
    started = time.perf_counter()
    result = estimate_frequency_interval(dataset, 0.108)
    return time.perf_counter() - started, result


def assert_nested_equivalent(test_case, optimized, reference, path="result"):
    test_case.assertEqual(type(reference), type(optimized), path)
    if isinstance(reference, dict):
        test_case.assertEqual(reference.keys(), optimized.keys(), path)
        for key in reference:
            assert_nested_equivalent(
                test_case, optimized[key], reference[key], f"{path}.{key}"
            )
    elif isinstance(reference, list):
        test_case.assertEqual(len(reference), len(optimized), path)
        for index, value in enumerate(reference):
            assert_nested_equivalent(
                test_case, optimized[index], value, f"{path}[{index}]"
            )
    elif isinstance(reference, float):
        test_case.assertTrue(
            math.isclose(optimized, reference, rel_tol=2e-11, abs_tol=2e-12),
            f"{path}: optimized={optimized!r}, reference={reference!r}",
        )
    else:
        test_case.assertEqual(reference, optimized, path)


@unittest.skipIf(frequency_uncertainty.np is None, "NumPy is not installed")
class VectorizedFrequencyUncertaintyEquivalenceTests(unittest.TestCase):
    def assert_profile_equivalent(self, dataset, selected, **kwargs):
        optimized = estimate_frequency_interval(dataset, selected, **kwargs)
        with mock.patch("frequency_uncertainty.np", None):
            reference = estimate_frequency_interval(dataset, selected, **kwargs)
        assert_nested_equivalent(self, optimized, reference)
        return optimized

    def test_unweighted_profile_matches_scalar_reference(self):
        self.assert_profile_equivalent(
            signal_dataset(frequency=0.108, baseline=300.0), 0.108
        )

    def test_weighted_profile_matches_scalar_reference(self):
        dataset = signal_dataset(frequency=0.108, baseline=180.0, noise=0.04)
        dataset["measurementUncertainties"] = [
            0.03 if index % 3 else 0.11
            for index in range(len(dataset["times"]))
        ]
        _, diagnostics = self.assert_profile_equivalent(dataset, 0.108)
        self.assertEqual(
            "known-per-observation-standard-deviations",
            diagnostics["noiseScaleTreatment"],
        )

    def test_relative_uncertainties_match_scalar_reference(self):
        dataset = signal_dataset(frequency=0.108, baseline=180.0, noise=0.04)
        dataset["measurementUncertainties"] = [
            0.7 + (index % 5) * 0.1 for index in range(len(dataset["times"]))
        ]
        dataset["measurementUncertaintiesAreRelative"] = True
        _, diagnostics = self.assert_profile_equivalent(dataset, 0.108)
        self.assertEqual(
            "profiled-global-residual-scale",
            diagnostics["noiseScaleTreatment"],
        )

    def test_competing_mode_decision_matches_scalar_reference(self):
        dataset = signal_dataset(frequency=0.108, baseline=239.0, noise=0.03)
        dataset["times"] = [float(index) for index in range(240)]
        randomizer = random.Random(19)
        dataset["flux"] = [
            math.sin(2.0 * math.pi * 0.108 * sample_time + 0.3)
            + randomizer.gauss(0.0, 0.03)
            for sample_time in dataset["times"]
        ]
        dataset["frequencySearch"]["maximumFrequency"] = 1.2
        interval, diagnostics = self.assert_profile_equivalent(
            dataset, 0.108, competing_frequencies=(1.108,)
        )
        self.assertIsNone(interval)
        self.assertIn("competing peak", diagnostics["unavailableReason"])

    def test_boundary_truncation_matches_scalar_reference(self):
        dataset = signal_dataset(frequency=0.041, baseline=12.0, noise=0.3)
        interval, diagnostics = self.assert_profile_equivalent(dataset, 0.041)
        self.assertIsNone(interval)
        self.assertIn("search boundary", diagnostics["unavailableReason"])

    def test_singular_and_invalid_cases_match_scalar_reference(self):
        singular = signal_dataset(frequency=0.108, baseline=0.0, count=12)
        invalid = signal_dataset(frequency=0.108, baseline=20.0, count=12)
        invalid["measurementUncertainties"] = [0.1] * 11 + [0.0]
        singular_interval, singular_diagnostics = self.assert_profile_equivalent(
            singular, 0.108
        )
        invalid_interval, invalid_diagnostics = self.assert_profile_equivalent(
            invalid, 0.108
        )
        self.assertIsNone(singular_interval)
        self.assertEqual(
            "positive time baseline is required",
            singular_diagnostics["unavailableReason"],
        )
        self.assertIsNone(invalid_interval)
        self.assertEqual(
            "measurement uncertainties are invalid",
            invalid_diagnostics["unavailableReason"],
        )

    def test_realistic_tess_benchmark_helper(self):
        elapsed, (interval, diagnostics) = benchmark_realistic_frequency_interval()
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertIsNotNone(interval)
        self.assertTrue(diagnostics["trustworthy"])


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

        method = (
            frequency_uncertainty._SinusoidProfile.rss
            if frequency_uncertainty.np is not None
            else frequency_uncertainty._sinusoid_rss_scalar
        )
        target = (
            "frequency_uncertainty._SinusoidProfile.rss"
            if frequency_uncertainty.np is not None
            else "frequency_uncertainty._sinusoid_rss_scalar"
        )
        with mock.patch(target, autospec=True, wraps=method) as evaluated:
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
