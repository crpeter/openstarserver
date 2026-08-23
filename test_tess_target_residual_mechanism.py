from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from openstar_investigation import ArtifactReference, sha256_file
from workflows.tess.tess_target_residual_mechanism import analyze_target_residual_mechanism


class TessTargetResidualMechanismTests(unittest.TestCase):
    def inputs(self, root: Path, generator, classification, *, sectors=(1, 2, 3), shifts=None):
        shifts = shifts or (0.0,) * len(sectors)
        prepared, artifacts, temporal = [], [], []
        for sector, shift in zip(sectors, shifts):
            times = [index / 20 for index in range(400)]
            values = [generator(time + shift) for time in times]
            coefficient = root / f"target-{sector}-coefficient.json"
            dataset = root / f"target-{sector}.json"
            coefficient.write_text(json.dumps({"componentID": "target", "times": times,
                                               "coefficients": values}))
            dataset.write_text(json.dumps({"science": {"componentID": "target"}}))
            dataset_id = f"dataset-{sector}"
            prepared.append({"datasetID": dataset_id, "sector": sector, "componentID": "target",
                             "componentType": "TARGET", "combined": False,
                             "coefficientSeriesPath": str(coefficient), "datasetPath": str(dataset)})
            temporal.append({"datasetID": dataset_id, "sector": sector, "frequency": 1.0})
            for path in (coefficient, dataset):
                artifacts.append(ArtifactReference(str(path.resolve()), sha256_file(path),
                                                   "application/json"))
        preparation = {"preparedSeries": prepared}
        decomposition = {"targetComponentID": "target"}
        v13 = {"classification": classification, "physicalMechanismResolved": False,
               "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP",
               "temporalModelEvidence": temporal}
        return preparation, decomposition, v13, artifacts

    def analyze(self, inputs, **kwargs):
        preparation, decomposition, v13, artifacts = inputs
        return analyze_target_residual_mechanism(preparation=preparation,
            decomposition=decomposition, v2013_result=v13,
            authoritative_artifacts=artifacts,
            v2013_lineage_verified=kwargs.get("lineage", True))

    def test_true_close_frequencies_prefer_beating(self):
        signal = lambda t: math.sin(2 * math.pi * .96 * t) + math.sin(2 * math.pi * 1.04 * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"))
        self.assertEqual("COHERENT_TWO_MODE_BEATING_SUPPORTED", result["classification"])

    def test_smooth_evolution_is_not_false_beating(self):
        signal = lambda t: (1 + .7 * ((t - 10) / 10) ** 2) * math.sin(2 * math.pi * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"))
        self.assertEqual("SMOOTH_SINGLE_MODE_AMPLITUDE_EVOLUTION", result["classification"])

    def test_intermittent_suppression_and_reappearance(self):
        def signal(t):
            amplitude = .05 if 8 <= t < 12 else 1.0
            return amplitude * math.sin(2 * math.pi * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"))
        self.assertEqual("EPISODIC_TARGET_MODE_ACTIVATION", result["classification"])

    def test_sector_clock_resets_and_ids_do_not_change_result(self):
        signal = lambda t: math.sin(2 * math.pi * .96 * t) + math.sin(2 * math.pi * 1.04 * t)
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first = self.analyze(self.inputs(Path(left), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL", sectors=(1, 2, 3)))
            second = self.analyze(self.inputs(Path(right), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL", sectors=(91, 7, 44), shifts=(3, 17, 41)))
        self.assertEqual(first["classification"], second["classification"])
        self.assertFalse(second["crossSectorPhaseUsed"])
        self.assertTrue(all(item["timingCoordinate"] == "SECTOR_LOCAL_WARPED_TIME"
                            for item in second["sectorModelEvidence"]))

    def test_modified_artifact_fails_closed(self):
        signal = lambda t: math.sin(2 * math.pi * t)
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), signal, "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            Path(inputs[0]["preparedSeries"][0]["coefficientSeriesPath"]).write_text("{}")
            result = self.analyze(inputs)
        self.assertTrue(any("SHA" in reason for reason in result["failClosedReasons"]))
        self.assertEqual("AMPLITUDE_EVOLUTION_MECHANISM_UNRESOLVED", result["classification"])

    def test_mismatched_v2013_lineage_fails_closed(self):
        signal = lambda t: math.sin(2 * math.pi * .96 * t) + math.sin(2 * math.pi * 1.04 * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"), lineage=False)
        self.assertEqual("AMPLITUDE_EVOLUTION_MECHANISM_UNRESOLVED", result["classification"])
        self.assertTrue(result["failClosedReasons"])

    def test_offset_and_blended_series_cannot_enter_observable(self):
        signal = lambda t: math.sin(2 * math.pi * t)
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), signal, "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            inputs[0]["preparedSeries"].extend([
                {"componentID": "offset-1", "componentType": "OFFSET", "sector": 1},
                {"componentID": "target", "componentType": "TARGET", "combined": True, "sector": 1},
            ])
            result = self.analyze(inputs)
        self.assertEqual(3, len(result["sectorModelEvidence"]))
        self.assertIn("target coefficient", result["observable"])

if __name__ == "__main__":
    unittest.main()
