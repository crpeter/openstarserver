from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_investigation import ArtifactReference, sha256_file
from workflows.tess.tess_target_residual_mechanism import (
    BEAT_MODEL_PARAMETERS,
    DECISIVE_DELTA_BIC,
    _bic,
    _model_sector,
    adjudicate_sector_model_evidence,
    analyze_target_residual_mechanism,
)


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
        v13 = {"classification": classification, "physicalMechanismResolved": False,
               "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP",
               "temporalModelEvidence": temporal}
        v13_path = root / "intrinsic-nonstationary-v20.13.json"
        v13_path.write_text(json.dumps(v13, sort_keys=True))
        v13_artifacts = (ArtifactReference(str(v13_path.resolve()), sha256_file(v13_path),
                                          "application/json"),)
        return {"preparedSeries": prepared}, {"targetComponentID": "target"}, v13, artifacts, v13_artifacts

    def analyze(self, inputs, **kwargs):
        preparation, decomposition, v13, artifacts, v13_artifacts = inputs
        return analyze_target_residual_mechanism(preparation=preparation,
            decomposition=decomposition, v2013_result=v13,
            authoritative_artifacts=artifacts,
            v2013_lineage_verified=kwargs.get("lineage", True),
            authoritative_v2013_artifacts=v13_artifacts)

    def test_true_close_frequencies_prefer_beating(self):
        signal = lambda t: math.sin(2 * math.pi * .96 * t) + math.sin(2 * math.pi * 1.04 * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"))
        self.assertEqual("COHERENT_TWO_MODE_BEATING_SUPPORTED", result["classification"])
        self.assertEqual(6, result["preRegisteredRules"]["modelParameterCounts"]
                         ["twoFrequencyIncludingGridOptimizedSeparation"])

    def test_grid_separation_penalty_changes_boundary_decision(self):
        count = 400
        # This RSS produces an 8-point win after charging six parameters. The
        # old five-parameter score adds ln(400), crossing the decisive +10 rule.
        beat_rss = math.exp(-(8 + 3 * math.log(count)) / count)
        def fitted(rows, values):
            width = len(rows[0])
            return (beat_rss if width == 5 else 1.0), [0.0] * width
        with patch("workflows.tess.tess_target_residual_mechanism._linear_fit", side_effect=fitted):
            model = _model_sector([index / 20 for index in range(count)], [0.0] * count, 1.0)
        current_margin = model["constantAmplitudeBIC"] - model["twoFrequencyBIC"]
        old_margin = model["constantAmplitudeBIC"] - _bic(beat_rss, count, 5)
        self.assertLess(current_margin, DECISIVE_DELTA_BIC)
        self.assertGreaterEqual(old_margin, DECISIVE_DELTA_BIC)
        self.assertEqual(6, BEAT_MODEL_PARAMETERS)

    def test_smooth_evolution_is_not_false_beating(self):
        signal = lambda t: (1 + .7 * ((t - 10) / 10) ** 2) * math.sin(2 * math.pi * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"))
        self.assertEqual("SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION", result["classification"])

    def test_smooth_phase_evolution_is_not_called_amplitude_evolution(self):
        signal = lambda t: math.sin(2 * math.pi * t + .9 * ((t - 10) / 10) ** 2)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"))
        self.assertNotEqual("SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION", result["classification"])

    def test_intermittent_suppression_and_reappearance(self):
        signal = lambda t: (.05 if 8 <= t < 12 else 1.0) * math.sin(2 * math.pi * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"))
        self.assertEqual("EPISODIC_TARGET_MODE_ACTIVATION", result["classification"])

    def test_phase_jumps_are_not_called_episodic_activation(self):
        def signal(t):
            phase = 0.0 if t < 8 else (math.pi / 2 if t < 12 else math.pi / 4)
            return math.sin(2 * math.pi * t + phase)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"))
        self.assertNotEqual("EPISODIC_TARGET_MODE_ACTIVATION", result["classification"])

    def test_transient_boundary_can_select_smooth_modulation(self):
        signal = lambda t: (1 + .8 * ((t - 10) / 10) ** 2) * math.sin(2 * math.pi * t)
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), signal,
                "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"))
        self.assertEqual("SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION", result["classification"])

    def test_sector_clock_reset_invariance_and_no_cross_sector_phase(self):
        signal = lambda t: math.sin(2 * math.pi * .96 * t) + math.sin(2 * math.pi * 1.04 * t)
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first = self.analyze(self.inputs(Path(left), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"))
            reset = self.analyze(self.inputs(Path(right), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL", shifts=(3, 17, 41)))
        self.assertEqual(first["classification"], reset["classification"])
        self.assertFalse(reset["crossSectorPhaseUsed"])
        self.assertTrue(all(item["timingCoordinate"] == "SECTOR_LOCAL_WARPED_TIME"
                            for item in reset["sectorModelEvidence"]))

    def test_sector_id_permutation_invariance(self):
        signal = lambda t: math.sin(2 * math.pi * .96 * t) + math.sin(2 * math.pi * 1.04 * t)
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first = self.analyze(self.inputs(Path(left), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL", sectors=(1, 2, 3)))
            permuted = self.analyze(self.inputs(Path(right), signal,
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL", sectors=(91, 7, 44)))
        self.assertEqual(first["classification"], permuted["classification"])

    def test_modified_v2012_coefficient_fails_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                                 "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            Path(inputs[0]["preparedSeries"][0]["coefficientSeriesPath"]).write_text("{}")
            result = self.analyze(inputs)
        self.assertTrue(any("v20.12 SHA" in reason for reason in result["failClosedReasons"]))

    def test_modified_v2012_dataset_fails_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                                 "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            Path(inputs[0]["preparedSeries"][0]["datasetPath"]).write_text("{}")
            result = self.analyze(inputs)
        self.assertTrue(any("v20.12 SHA" in reason for reason in result["failClosedReasons"]))

    def test_modified_v2013_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                                 "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            Path(inputs[4][0].path).write_text("{}")
            result = self.analyze(inputs)
        self.assertTrue(any("v20.13 artifact" in reason for reason in result["failClosedReasons"]))

    def test_falsified_v2013_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                                 "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            inputs[2]["temporalModelEvidence"][0]["frequency"] = 1.01
            result = self.analyze(inputs)
        self.assertTrue(any("v20.13 artifact" in reason for reason in result["failClosedReasons"]))

    def test_mismatched_v2013_lineage_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.analyze(self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"), lineage=False)
        self.assertTrue(any("lineage" in reason for reason in result["failClosedReasons"]))

    def test_offset_and_blended_series_cannot_enter_observable(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                                 "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            inputs[0]["preparedSeries"].extend([
                {"componentID": "offset-1", "componentType": "OFFSET", "sector": 1},
                {"componentID": "target", "componentType": "TARGET", "combined": True, "sector": 1},
            ])
            result = self.analyze(inputs)
        self.assertEqual(3, len(result["sectorModelEvidence"]))
        self.assertIn("target coefficient", result["observable"])

    def test_mixed_replicated_mechanisms_do_not_advance(self):
        beating = {"constantAmplitudeBIC": 30., "smoothEnvelopeBIC": 20., "twoFrequencyBIC": 0.,
            "intermittentEnvelopeBIC": 40., "beatFrequencySeparation": .08,
            "intermittentSegmentAmplitudes": [1.] * 5, "episodicSuppressionAndReappearance": False}
        smooth = dict(beating, constantAmplitudeBIC=30., smoothEnvelopeBIC=0., twoFrequencyBIC=20.)
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL", sectors=(1, 2, 3, 4))
            with patch("workflows.tess.tess_target_residual_mechanism._model_sector",
                       side_effect=[copy.deepcopy(beating), copy.deepcopy(beating),
                                    copy.deepcopy(smooth), copy.deepcopy(smooth)]):
                result = self.analyze(inputs)
        self.assertEqual("TARGET_RESIDUAL_MECHANISM_UNRESOLVED", result["classification"])
        self.assertEqual("TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP",
                         result["recommendedNextTest"])

    def test_consumed_science_results_are_not_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(Path(directory), lambda t: math.sin(2 * math.pi * t),
                                 "AMPLITUDE_EVOLVING_TARGET_RESIDUAL")
            frozen_results = copy.deepcopy(inputs[:3])
            self.analyze(inputs)
        self.assertEqual(frozen_results, inputs[:3])

    @staticmethod
    def evidence(best, *, episodic=True, sector=1, gap=20.0):
        values = {"constantAmplitudeBIC": 100.0, "smoothEnvelopeBIC": 100.0,
                  "twoFrequencyBIC": 100.0, "intermittentEnvelopeBIC": 100.0}
        fields = {"constant": "constantAmplitudeBIC", "smooth": "smoothEnvelopeBIC",
                  "beating": "twoFrequencyBIC", "episodic": "intermittentEnvelopeBIC"}
        values[fields[best]] = 100.0 - gap
        return {"sector": sector, **values,
                "episodicSuppressionAndReappearance": episodic}

    def test_route_independent_sector_winners(self):
        for admission in ("AMPLITUDE_EVOLVING_TARGET_RESIDUAL",
                          "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"):
            for model, expected in (
                ("beating", "COHERENT_TWO_MODE_BEATING_SUPPORTED"),
                ("smooth", "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"),
                ("episodic", "EPISODIC_TARGET_MODE_ACTIVATION"),
            ):
                with self.subTest(admission=admission, model=model):
                    # Admission is intentionally absent from the shared API.
                    result = adjudicate_sector_model_evidence(
                        [self.evidence(model), self.evidence(model, sector=2)])
                    self.assertEqual(expected, result["classification"])

    def test_intermitttent_best_without_shape_has_no_smooth_fallback(self):
        real = self.evidence("episodic", episodic=False)
        real.update(smoothEnvelopeBIC=31402.58867522472,
                    intermittentEnvelopeBIC=30943.414321673405,
                    constantAmplitudeBIC=31600.0, twoFrequencyBIC=31700.0)
        result = adjudicate_sector_model_evidence([real])
        sector = result["sectorModelEvidence"][0]
        self.assertEqual("EPISODIC_ACTIVATION", sector["bestModel"])
        self.assertTrue(sector["extraMorphologyGateBlockedPromotion"])
        self.assertEqual("TARGET_RESIDUAL_MECHANISM_UNRESOLVED",
                         sector["sectorClassification"])

    def test_real_sector_69_beating_competes_but_one_sector_does_not_promote(self):
        real = self.evidence("beating")
        real.update(smoothEnvelopeBIC=24868.075748741066,
                    twoFrequencyBIC=24839.166068701925,
                    constantAmplitudeBIC=25000.0, intermittentEnvelopeBIC=25100.0)
        result = adjudicate_sector_model_evidence([real])
        self.assertEqual("COHERENT_TWO_MODE_BEATING_SUPPORTED",
                         result["sectorModelEvidence"][0]["sectorClassification"])
        self.assertEqual("TARGET_RESIDUAL_MECHANISM_UNRESOLVED", result["classification"])

    def test_nondecisive_and_constant_best_are_unresolved(self):
        nondecisive = self.evidence("smooth", gap=9.99)
        constant = self.evidence("constant")
        for item in (nondecisive, constant):
            with self.subTest(best=min((key for key in item if key.endswith("BIC")), key=item.get)):
                result = adjudicate_sector_model_evidence([item, dict(item, sector=2)])
                self.assertEqual("TARGET_RESIDUAL_MECHANISM_UNRESOLVED", result["classification"])

    def test_one_replicating_mechanism_promotes_but_two_do_not(self):
        beating = [self.evidence("beating", sector=1), self.evidence("beating", sector=2)]
        promoted = adjudicate_sector_model_evidence(beating + [self.evidence("smooth", sector=3)])
        self.assertEqual("COHERENT_TWO_MODE_BEATING_SUPPORTED", promoted["classification"])
        conflict = adjudicate_sector_model_evidence(beating +
            [self.evidence("smooth", sector=3), self.evidence("smooth", sector=4)])
        self.assertEqual("TARGET_RESIDUAL_MECHANISM_UNRESOLVED", conflict["classification"])

    def test_duplicate_same_sector_rows_never_establish_independent_replication(self):
        duplicated = [self.evidence("beating", sector=69),
                      self.evidence("beating", sector=69)]
        result = adjudicate_sector_model_evidence(duplicated)
        self.assertEqual("TARGET_RESIDUAL_MECHANISM_UNRESOLVED", result["classification"])
        self.assertEqual([], result["replicatedMechanisms"])
        self.assertTrue(any("duplicate persisted sector" in reason
                            for reason in result["failClosedReasons"]))

    def test_distinct_sector_replication_records_auditable_support(self):
        result = adjudicate_sector_model_evidence([
            self.evidence("beating", sector=69), self.evidence("beating", sector=95)])
        self.assertEqual("COHERENT_TWO_MODE_BEATING_SUPPORTED", result["classification"])
        self.assertEqual(
            {"COHERENT_TWO_MODE_BEATING_SUPPORTED": [69, 95]},
            result["replicatedMechanismSupportingSectorIDs"],
        )

    def test_conflicting_duplicate_sector_evidence_fails_closed(self):
        result = adjudicate_sector_model_evidence([
            self.evidence("beating", sector=69), self.evidence("smooth", sector=69),
            self.evidence("beating", sector=70), self.evidence("smooth", sector=71)])
        self.assertEqual("TARGET_RESIDUAL_MECHANISM_UNRESOLVED", result["classification"])
        self.assertTrue(any("duplicate persisted sector evidence IDs: 69" == reason
                            for reason in result["failClosedReasons"]))


if __name__ == "__main__":
    unittest.main()
