import copy
import json
import math
import tempfile
import unittest
from pathlib import Path


from workflows.tess.tess_mode_identification import (
    GENERIC_REFINEMENT_WORKLOAD_ID,
    MULTIMODE_MODE_EVIDENCE_LINEAGE,
    identify_residual_mode,
    validated_multimode_mode_evidence,
)


class TessModeIdentificationTests(unittest.TestCase):
    @staticmethod
    def _multimode_summary():
        frequency = 0.3
        points = [{
            "iteration": 1,
            "sector": sector,
            "role": "independent-residual-multimode",
            "candidateFrequency": frequency,
            "candidatePeriodDays": 1.0 / frequency,
            "candidatePeakProminenceRatio": 4.0,
            "acceptedDistinctMode": True,
        } for sector in (2, 4, 97, 98)]
        members = [{
            "iteration": item["iteration"],
            "sector": item["sector"],
            "role": item["role"],
            "frequency": item["candidateFrequency"],
            "periodDays": item["candidatePeriodDays"],
            "prominence": item["candidatePeakProminenceRatio"],
        } for item in points]
        recurrent = {
            "medianFrequency": frequency,
            "medianPeriodDays": 1.0 / frequency,
            "independentSectors": [2, 4, 97, 98],
            "independentSectorCount": 4,
            "combinedSupport": False,
            "members": members,
        }
        return {
            "classification": "MULTI_MODE_RECURRENT",
            "physicalPeriodDays": 13.0,
            "physicalFrequency": 1.0 / 13.0,
            "firstHarmonicFrequency": 2.0 / 13.0,
            "iterationsCompleted": 2,
            "acceptedResidualModes": points,
            "frequencyClusters": [recurrent],
            "bestRecurrentSecondaryMode": recurrent,
            "independentSectorsWithAcceptedResidualModes": [2, 4, 97, 98],
            "minimumRecurrentIndependentSectorCount": 3,
            "clusterRelativeTolerance": 0.05,
            "physicalMechanismResolved": False,
            "claimLevelChanged": False,
            "recommendedNextTest":
            "MODE_IDENTIFICATION_OR_PULSATION_MODELING",
        }

    def test_validates_recurrent_multimode_input_contract(self):
        evidence = validated_multimode_mode_evidence(
            self._multimode_summary(),
            physical_period_days=13.0,
            target_supporting_sectors=[2, 4, 97, 98],
            iteration_count=2,
        )
        self.assertEqual(MULTIMODE_MODE_EVIDENCE_LINEAGE,
                         evidence["evidenceLineage"])
        self.assertEqual(13.0, evidence["establishedPeriodDays"])
        self.assertEqual([2, 4, 97, 98], evidence["independentSectors"])

    def test_recurrent_multimode_input_contract_fails_closed(self):
        mutations = (
            lambda value: value.update(physicalPeriodDays=12.0),
            lambda value: value.update(recommendedNextTest="OTHER"),
            lambda value: value["bestRecurrentSecondaryMode"].update(
                medianPeriodDays=4.0),
            lambda value: value["bestRecurrentSecondaryMode"].update(
                independentSectors=[2, 4, 96, 97]),
            lambda value: value.update(frequencyClusters=[]),
            lambda value: value["acceptedResidualModes"][0].update(
                acceptedDistinctMode=False),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                summary = copy.deepcopy(self._multimode_summary())
                mutate(summary)
                self.assertIsNone(validated_multimode_mode_evidence(
                    summary,
                    physical_period_days=13.0,
                    target_supporting_sectors=[2, 4, 97, 98],
                    iteration_count=2,
                ))

    def _datasets(self, root, *, family_period=10.0, residual_period=None,
                  residual_amplitude=0.0, sectors=4):
        paths = []
        for sector in range(sectors):
            times = [sector * 100.0 + 25.0 * index / 499 for index in range(500)]
            flux = [(0.8 * math.sin(2 * math.pi * time / family_period)
                     + 0.35 * math.cos(4 * math.pi * time / family_period)
                     + (residual_amplitude * math.sin(2 * math.pi * time / residual_period + 0.2)
                        if residual_period else 0.0)) for time in times]
            path = Path(root) / f"sector-{sector}.json"
            path.write_text(json.dumps({"times": times, "flux": flux}))
            paths.append(path)
        return paths

    def test_higher_harmonic_is_not_an_independent_mode(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, residual_period=2.5, residual_amplitude=0.5)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=2.5, independent_sectors=[1, 2, 3, 4])
        self.assertEqual("HIGHER_ORDER_HARMONIC_STRUCTURE", result["classification"])
        self.assertFalse(result["independentModeEvidenceSurvived"])
        self.assertTrue(result["harmonicRelation"]["commensurateWithinResolution"])

    def test_supported_off_harmonic_mode_survives(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, residual_period=3.1, residual_amplitude=0.6)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=3.1, independent_sectors=[94, 95, 102, 103])
        self.assertEqual("INDEPENDENT_STABLE_MODE", result["classification"])
        self.assertTrue(result["independentModeEvidenceSurvived"])
        self.assertEqual("RESIDUAL_MODE_PIXEL_LOCALIZATION", result["recommendedNextTest"])

    def test_weak_improvement_is_not_compelling(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=3.1, independent_sectors=[1, 2, 3])
        self.assertEqual("NO_COMPELLING_RESIDUAL_MODE", result["classification"])

    def test_insufficient_support_prevents_strong_independent_claim(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, residual_period=3.1, residual_amplitude=0.6)
            result = identify_residual_mode(dataset_paths=paths, established_period_days=10,
                                            residual_period_days=3.1, independent_sectors=[94, 95])
        self.assertFalse(result["independentModeEvidenceSurvived"])
        self.assertNotEqual("INDEPENDENT_STABLE_MODE", result["classification"])

    def test_tic_277940827_relation_runs_nested_comparison(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._datasets(root, family_period=10.3008408008,
                                  residual_period=2.5751446508, residual_amplitude=0.5)
            result = identify_residual_mode(dataset_paths=paths,
                established_period_days=10.3008408008, residual_period_days=2.5751446508,
                independent_sectors=[94, 95, 102, 103])
        self.assertEqual(4, result["harmonicRelation"]["testedOrder"])
        self.assertEqual(3, len(result["modelComparison"]["models"]))
        self.assertEqual("openstar.lomb-scargle.v1", GENERIC_REFINEMENT_WORKLOAD_ID)




if __name__ == "__main__":
    unittest.main()
