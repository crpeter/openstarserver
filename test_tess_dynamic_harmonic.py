import json
import math
import tempfile
import unittest
from pathlib import Path

from workflows.tess.tess_dynamic_harmonic import (
    compare_unresolved_family_dynamic_harmonics,
    model_dynamic_harmonics,
    read_frozen_light_curve,
    refine_harmonic_family_frequency,
)


class TessDynamicHarmonicTests(unittest.TestCase):
    def _data(self, root, *, period=10.0, amplitudes=None, phases=None,
              fourth=0.2, residual=0.0):
        paths = []
        amplitudes = amplitudes or [[0.8, 0.8, 0.8, 0.8]] * 4
        phases = phases or [[0.1, 0.1, 0.1, 0.1]] * 4
        for sector in range(4):
            times = [sector * 70 + 24 * i / 399 for i in range(400)]
            flux = []
            for i, time in enumerate(times):
                value = 0.001 * math.sin(i * 1.731 + sector)
                harmonic_amplitudes = [amplitudes[sector][0], 0.35, 0.12, fourth]
                for order, amplitude in enumerate(harmonic_amplitudes, 1):
                    value += amplitude * math.sin(2 * math.pi * order * time / period + phases[sector][order-1])
                value += residual * math.sin(2 * math.pi * time / 3.13 + 0.4)
                flux.append(value)
            path = Path(root) / f"sector-{sector}.json"
            path.write_text(json.dumps({"times": times, "flux": flux,
                                        "source": {"sector": sector}}))
            paths.append(path)
        return paths

    def _run(self, **kwargs):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        return model_dynamic_harmonics(dataset_paths=self._data(root.name, **kwargs),
                                       reference_period_days=kwargs.get("period", 10.0))

    def _alias_data(self, root, components):
        paths = []
        for sector in range(4):
            sector_components = (
                components(sector) if callable(components) else components
            )
            times = [sector * 47 + 18 * index / 599 for index in range(600)]
            scale = (0.7, 1.0, 0.8, 1.2)[sector]
            flux = [
                sum(
                    scale * amplitude * math.sin(
                        2 * math.pi * frequency * time + phase
                    )
                    for frequency, amplitude, phase in sector_components
                ) + 0.003 * math.sin(1.731 * index + sector)
                for index, time in enumerate(times)
            ]
            path = Path(root) / f"alias-sector-{sector}.json"
            path.write_text(json.dumps({
                "times": times,
                "flux": flux,
                "source": {"sector": sector},
            }))
            paths.append(path)
        return paths

    def _compare_alias(self, components):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        return compare_unresolved_family_dynamic_harmonics(
            dataset_paths=self._alias_data(root.name, components),
            raw_period_days=5.0,
            double_cycle_period_days=10.0,
            primary_sector=0,
        )

    def test_coherent_static_family(self):
        result = self._run()
        self.assertEqual("COHERENT_STATIC_HARMONIC_FAMILY", result["classification"])
        self.assertFalse(result["dataReuse"]["downloadPerformed"])

    def test_sector_scan_metadata_preserves_absolute_time_and_sector(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = Path(root.name) / "scan-dataset.json"
        path.write_text(json.dumps({
            "times": [0.1 * index for index in range(20)],
            "flux": [float(index) for index in range(20)],
            "metadata": {
                "sector": 1,
                "originalTimeOriginDays": 1325.25,
            },
        }))
        result = read_frozen_light_curve(path, position=99)
        self.assertEqual(1, result["sector"])
        self.assertEqual(1325.25, result["appliedTimeOriginDays"])
        self.assertAlmostEqual(1325.25, result["times"][0])

    def test_amplitude_only_evolution(self):
        amplitudes = [[value] * 4 for value in (0.5, 0.8, 1.1, 0.7)]
        result = self._run(amplitudes=amplitudes)
        self.assertEqual("COHERENT_HARMONIC_FAMILY_WITH_AMPLITUDE_EVOLUTION", result["classification"])

    def test_phase_only_evolution(self):
        phases = [[value] * 4 for value in (0.0, 0.25, -0.18, 0.12)]
        result = self._run(phases=phases)
        self.assertEqual("COHERENT_HARMONIC_FAMILY_WITH_PHASE_EVOLUTION", result["classification"])

    def test_amplitude_and_phase_evolution(self):
        amplitudes = [[value] * 4 for value in (0.5, 0.8, 1.1, 0.7)]
        phases = [[value] * 4 for value in (0.0, 0.25, -0.18, 0.12)]
        result = self._run(amplitudes=amplitudes, phases=phases)
        self.assertEqual("COHERENT_HARMONIC_FAMILY_WITH_AMPLITUDE_AND_PHASE_EVOLUTION", result["classification"])

    def test_wrong_reference_frequency_is_refinement_not_period_change(self):
        root = tempfile.TemporaryDirectory(); self.addCleanup(root.cleanup)
        paths = self._data(root.name, period=10.0)
        result = model_dynamic_harmonics(dataset_paths=paths, reference_period_days=10.02)
        self.assertEqual("HARMONIC_FAMILY_REQUIRES_FREQUENCY_REFINEMENT", result["classification"])
        self.assertEqual("openstar.lomb-scargle.v1",
                         result["frequencyRefinement"]["genericDistributedWorkloadIfNeeded"])
        refined = refine_harmonic_family_frequency(result)
        self.assertAlmostEqual(10.0, refined["refinedPeriodDays"], places=2)
        self.assertFalse(refined["physicalPeriodChangeClaimed"])
        self.assertEqual("openstar.lomb-scargle.v1",
                         refined["distributedRefinement"]["workloadID"])

    def test_additional_mode_remains(self):
        result = self._run(residual=0.8)
        self.assertEqual("ADDITIONAL_VARIABILITY_REMAINS", result["classification"])
        self.assertEqual("RESIDUAL_MULTIMODE_LOCALIZATION", result["recommendedNextTest"])

    def test_unsupported_extra_harmonic_fails_bic_threshold(self):
        result = self._run(fourth=0.0)
        self.assertFalse(result["modelComparison"]["highestTestedHarmonicSupported"])

    def test_tic_like_fourth_harmonic_is_supported(self):
        result = self._run(period=10.3008408008, fourth=0.5)
        self.assertEqual([1, 2, 3, 4], result["harmonicOrdersTested"])
        self.assertTrue(result["modelComparison"]["highestTestedHarmonicSupported"])
        self.assertFalse(result["physicalMechanismResolved"])

    def test_equal_half_waveform_cannot_resolve_shorter_physical_cycle(self):
        result = self._compare_alias((
            (1 / 5.0, 0.9, 0.2),
            (3 / 5.0, 0.45, -0.3),
        ))
        self.assertEqual("UNRESOLVED_FAMILY_DYNAMIC_HARMONIC_ALIAS_AMBIGUOUS",
                         result["classification"])
        self.assertFalse(result["physicalCycleResolved"])
        self.assertIsNone(result["resolvedPhysicalPeriodDays"])
        resolution = result["periodAliasResolution"]
        self.assertEqual([],
                         resolution["oddHarmonicSupportingHeldOutSectors"])
        self.assertIn("NON_RESOLUTION_ONLY",
                      resolution["equalHalfOutcomeInterpretation"])

    def test_odd_harmonics_predictively_select_double_cycle(self):
        result = self._compare_alias((
            (1 / 10.0, 0.9, 0.2),
            (3 / 10.0, 0.45, -0.3),
        ))
        self.assertEqual("DOUBLE_CYCLE_ODD_HARMONICS_PREDICTIVELY_SUPPORTED",
                         result["classification"])
        self.assertTrue(result["physicalCycleResolved"])
        self.assertEqual(10.0, result["resolvedPhysicalPeriodDays"])
        resolution = result["periodAliasResolution"]
        self.assertEqual([0, 1, 2, 3],
                         resolution["oddHarmonicSupportingHeldOutSectors"])
        self.assertEqual([1, 2, 3],
                         resolution[
                             "oddHarmonicSupportingIndependentHeldOutSectors"])
        self.assertEqual([2, 4, 6, 8], result["periodHypothesisModels"]
                         ["equalHalfEvenOnly"]["harmonicOrdersTested"])
        self.assertEqual(list(range(1, 9)), result["periodHypothesisModels"]
                         ["fullDoubleCycle"]["harmonicOrdersTested"])
        comparison = resolution["comparisons"][0]
        self.assertEqual(
            list(range(1, 9)),
            comparison["equalHalfEvenOnlyHypothesis"]
            ["phaseLearningHarmonicOrders"],
        )
        self.assertEqual(
            comparison["equalHalfEvenOnlyHypothesis"]
            ["phaseLearningHarmonicOrders"],
            comparison["fullDoubleCycleHypothesis"]
            ["phaseLearningHarmonicOrders"],
        )

    def test_primary_sector_does_not_veto_independent_odd_harmonic_support(self):
        result = self._compare_alias(lambda sector: (
            (1 / 5.0, 0.9, 0.2),
            ((3 / 5.0, 0.45, -0.3)
             if sector == 0 else
             (1 / 10.0, 0.45, -0.3)),
        ))
        self.assertEqual(
            "DOUBLE_CYCLE_ODD_HARMONICS_PREDICTIVELY_SUPPORTED",
            result["classification"],
        )
        self.assertEqual(
            [1, 2, 3],
            result["periodAliasResolution"]
            ["oddHarmonicSupportingIndependentHeldOutSectors"],
        )

    def test_two_independent_odd_harmonic_sectors_remain_ambiguous(self):
        result = self._compare_alias(lambda sector: (
            (1 / 5.0, 0.9, 0.2),
            ((1 / 10.0, 0.45, -0.3)
             if sector in (1, 2) else
             (3 / 5.0, 0.45, -0.3)),
        ))
        self.assertEqual(
            "UNRESOLVED_FAMILY_DYNAMIC_HARMONIC_ALIAS_AMBIGUOUS",
            result["classification"],
        )
        self.assertFalse(result["physicalCycleResolved"])
        self.assertIsNone(result["resolvedPhysicalPeriodDays"])
        self.assertEqual(
            "ADDITIONAL_INDEPENDENT_SECTOR_CYCLE_ALIAS_CONFIRMATION",
            result["recommendedNextTest"],
        )
        self.assertEqual(
            "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE",
            result["referencePeriodRole"],
        )

    def test_unresolved_alias_comparison_rejects_invalid_lineage(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        paths = self._alias_data(root.name, ((1 / 5.0, 0.9, 0.2),))
        with self.assertRaisesRegex(ValueError, "exact raw/2x"):
            compare_unresolved_family_dynamic_harmonics(
                dataset_paths=paths,
                raw_period_days=5.0,
                double_cycle_period_days=9.9,
                primary_sector=0,
            )
        with self.assertRaisesRegex(RuntimeError, "primary and at least three"):
            compare_unresolved_family_dynamic_harmonics(
                dataset_paths=paths[:2],
                raw_period_days=5.0,
                double_cycle_period_days=10.0,
                primary_sector=0,
            )


if __name__ == "__main__":
    unittest.main()
