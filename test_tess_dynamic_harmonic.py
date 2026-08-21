import json
import math
import tempfile
import unittest
from pathlib import Path

from workflows.tess.tess_dynamic_harmonic import model_dynamic_harmonics


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

    def test_coherent_static_family(self):
        result = self._run()
        self.assertEqual("COHERENT_STATIC_HARMONIC_FAMILY", result["classification"])
        self.assertFalse(result["dataReuse"]["downloadPerformed"])

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


if __name__ == "__main__":
    unittest.main()
