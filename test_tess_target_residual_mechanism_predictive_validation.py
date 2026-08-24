from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from workflows.tess.tess_target_residual_mechanism_predictive_validation import (
    PREDICTIVE_FOLDS, UNRESOLVED, adjudicate_predictive_sectors,
    analyze_predictive_validation,
    construct_folds, fit_training_model,
    freeze_model_domain, validate_sector, v2013_lineage_matches,
)
from openstar_investigation import (
    ArtifactReference, InvestigationStage, InvestigationStore, sha256_file,
    sha256_json,
)
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION

try:
    import numpy as _real_numpy
except ModuleNotFoundError:
    _real_numpy = None
_installed_numpy_stub = _real_numpy is None
if _installed_numpy_stub:
    sys.modules["numpy"] = types.ModuleType("numpy")
from workflows.tess.tess_investigation import build_engine
if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


class TessTargetResidualMechanismPredictiveValidationTests(unittest.TestCase):
    def setUp(self):
        self.times = [index / 20 for index in range(400)]
        self.domain = freeze_model_domain(self.times, 1.0)

    def test_folds_are_deterministic_blocked_and_retain_every_segment(self):
        first = construct_folds(self.times, self.domain)
        self.assertEqual(first, construct_folds(self.times, self.domain))
        self.assertEqual(PREDICTIVE_FOLDS, len(first))
        self.assertTrue(all(len(fold["heldOutBlocks"]) == 5 for fold in first))
        self.assertTrue(all(block["count"] for fold in first for block in fold["heldOutBlocks"]))

    def test_domain_and_beat_grid_are_frozen_before_edge_holdout(self):
        folds = construct_folds(self.times, self.domain)
        self.assertIn(0, folds[0]["heldOutIndices"])
        for fold in folds:
            self.assertEqual(self.times[0], self.domain["fullDomainStart"])
            self.assertEqual(self.times[-1], self.domain["fullDomainEnd"])
            self.assertEqual(81, len(self.domain["beatDeltaGrid"]))

    def test_held_out_values_cannot_change_any_training_fit(self):
        values = [(1 + .5 * ((time - 10) / 10) ** 2) * math.sin(2*math.pi*time)
                  for time in self.times]
        fold = construct_folds(self.times, self.domain)[0]
        train = fold["trainingIndices"]
        altered = copy.copy(values)
        for index in fold["heldOutIndices"]: altered[index] += 1e9
        for model in ("CONSTANT_AMPLITUDE", "SMOOTH_AMPLITUDE_MODULATION",
                      "COHERENT_TWO_MODE_BEATING", "EPISODIC_ACTIVATION"):
            left = fit_training_model(model, [self.times[i] for i in train],
                                      [values[i] for i in train], 1.0, self.domain)
            right = fit_training_model(model, [self.times[i] for i in train],
                                       [altered[i] for i in train], 1.0, self.domain)
            self.assertEqual(left, right)

    def test_constant_predicts_best_but_does_not_claim_mechanism(self):
        result = validate_sector(self.times, [math.sin(2*math.pi*t) for t in self.times], 1.0,
            sector=1, dataset_id="one", timing_coordinate="SECTOR_LOCAL_WARPED_TIME",
            episodic_morphology=False)
        self.assertEqual("CONSTANT_AMPLITUDE", result["bestPredictiveModel"])
        self.assertEqual(UNRESOLVED, result["sectorClassification"])

    def test_smooth_beating_and_episodic_synthetic_signals(self):
        cases = (
            (lambda t: (1+.8*((t-10)/10)**2)*math.sin(2*math.pi*t), False,
             "SMOOTH_AMPLITUDE_MODULATION"),
            (lambda t: math.sin(2*math.pi*.96*t)+math.sin(2*math.pi*1.04*t), False,
             "COHERENT_TWO_MODE_BEATING"),
            (lambda t: (.05 if 8 <= t < 12 else 1)*math.sin(2*math.pi*t), True,
             "EPISODIC_ACTIVATION"),
        )
        for signal, morphology, expected in cases:
            with self.subTest(model=expected):
                result = validate_sector(self.times, [signal(t) for t in self.times], 1.0,
                    sector=1, dataset_id="one", timing_coordinate="SECTOR_LOCAL_WARPED_TIME",
                    episodic_morphology=morphology)
                self.assertEqual(expected, result["bestPredictiveModel"])
                if expected == "COHERENT_TWO_MODE_BEATING":
                    # The other constrained families can legitimately fail on
                    # a phase-reversing beat. Fair-comparison admission then
                    # overrides even an overwhelming beating score.
                    self.assertEqual(UNRESOLVED, result["sectorClassification"])
                    self.assertFalse(result["fairAllModelComparisonCompleted"])
                else:
                    self.assertNotEqual(UNRESOLVED, result["sectorClassification"])

    def test_episodic_predictive_win_obeys_frozen_morphology_veto(self):
        values = [(.05 if 8 <= t < 12 else 1)*math.sin(2*math.pi*t) for t in self.times]
        result = validate_sector(self.times, values, 1.0, sector=1, dataset_id="one",
            timing_coordinate="SECTOR_LOCAL_WARPED_TIME", episodic_morphology=False)
        self.assertEqual("EPISODIC_ACTIVATION", result["bestPredictiveModel"])
        self.assertTrue(result["morphologyGateBlockedPromotion"])
        self.assertEqual(UNRESOLVED, result["sectorClassification"])

    def test_singular_smooth_design_fails_without_zero_envelope_fallback(self):
        with patch("workflows.tess.tess_target_residual_mechanism_predictive_validation._fit",
                   side_effect=ValueError("singular training design")):
            with self.assertRaisesRegex(ValueError, "no identifiable valid nonnegative"):
                fit_training_model("SMOOTH_AMPLITUDE_MODULATION", self.times,
                    [math.sin(2*math.pi*t) for t in self.times], 1.0, self.domain)

    def test_no_valid_nonnegative_smooth_fit_fails_closed(self):
        def negative_smooth(rows, values):
            return 1.0, [0.0, -1.0, 0.0, 0.0]
        with patch("workflows.tess.tess_target_residual_mechanism_predictive_validation._fit",
                   side_effect=negative_smooth):
            with self.assertRaises(ValueError):
                fit_training_model("SMOOTH_AMPLITUDE_MODULATION", self.times,
                                   [0.0] * len(self.times), 1.0, self.domain)

    def test_no_valid_nonnegative_intermittent_fit_fails_closed(self):
        def negative_segment(rows, values):
            return 1.0, [0.0] + [-1.0] * 5
        with patch("workflows.tess.tess_target_residual_mechanism_predictive_validation._fit",
                   side_effect=negative_segment):
            with self.assertRaises(ValueError):
                fit_training_model("EPISODIC_ACTIVATION", self.times,
                                   [0.0] * len(self.times), 1.0, self.domain)

    def test_one_model_fold_failure_forces_unresolved_even_when_beating_wins(self):
        original = fit_training_model
        def fail_intermittent(model, *args, **kwargs):
            if model == "EPISODIC_ACTIVATION":
                raise ValueError("forced constrained failure")
            return original(model, *args, **kwargs)
        values = [math.sin(2*math.pi*.96*t)+math.sin(2*math.pi*1.04*t)
                  for t in self.times]
        with patch("workflows.tess.tess_target_residual_mechanism_predictive_validation.fit_training_model",
                   side_effect=fail_intermittent):
            result = validate_sector(self.times, values, 1.0, sector=1, dataset_id="one",
                timing_coordinate="SECTOR_LOCAL_WARPED_TIME", episodic_morphology=False)
        self.assertEqual("COHERENT_TWO_MODE_BEATING", result["bestPredictiveModel"])
        self.assertFalse(result["fairAllModelComparisonCompleted"])
        self.assertFalse(result["decisivePredictiveWinner"])
        self.assertEqual(UNRESOLVED, result["sectorClassification"])

    def _direct_inputs(self, root, version="route-independent-all-models-v1"):
        v13 = {"temporalModelEvidence": []}
        v14 = {"adjudicationVersion": version,
            "classification": "TARGET_RESIDUAL_MECHANISM_UNRESOLVED",
            "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP",
            "physicalMechanismResolved": False, "failClosedReasons": [],
            "sectorModelEvidence": []}
        refs = []
        for name, value in (("intrinsic-nonstationary-v20.13.json", v13),
                            ("target-residual-mechanism-v20.14.json", v14)):
            path = root / name
            path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            refs.append(ArtifactReference(str(path), sha256_file(path), "application/json"))
        return v13, v14, refs

    def test_direct_corrected_v2014_is_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            v13, v14, refs = self._direct_inputs(Path(directory))
            result = analyze_predictive_validation(preparation={"preparedSeries": []},
                v2013_result=v13, v2014_result=v14, adjudication_result=v14,
                adjudication_stage_id="028-target-residual-mechanism",
                adjudication_handler_id="openstar.tess.target-residual-mechanism.analyze",
                preparation_artifacts=(), v2013_artifacts=(refs[0],),
                v2014_artifacts=(refs[1],), adjudication_artifacts=(refs[1],),
                lineage_verified=True)
        self.assertEqual(UNRESOLVED, result["classification"])

    def test_direct_old_or_wrong_v2014_semantics_are_rejected(self):
        for version in (None, "wrong-version"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                v13, v14, refs = self._direct_inputs(Path(directory), version)
                if version is None:
                    v14.pop("adjudicationVersion")
                    path = Path(refs[1].path)
                    path.write_text(json.dumps(v14, sort_keys=True), encoding="utf-8")
                    refs[1] = ArtifactReference(str(path), sha256_file(path), "application/json")
                with self.assertRaisesRegex(RuntimeError, "route-independent"):
                    analyze_predictive_validation(preparation={"preparedSeries": []},
                        v2013_result=v13, v2014_result=v14, adjudication_result=v14,
                        adjudication_stage_id="028-target-residual-mechanism",
                        adjudication_handler_id="openstar.tess.target-residual-mechanism.analyze",
                        preparation_artifacts=(), v2013_artifacts=(refs[0],),
                        v2014_artifacts=(refs[1],), adjudication_artifacts=(refs[1],),
                        lineage_verified=True)

    def test_v2013_lineage_requires_both_exact_v2012_snapshots(self):
        preparation, interpretation = {"prepared": 1}, {"interpreted": 2}
        hashes = {"v20.12Preparation": sha256_json(preparation),
                  "v20.12Interpretation": sha256_json(interpretation)}
        provenance = {"v20.12PreparationResultHash": sha256_json(preparation),
                      "v20.12InterpretationResultHash": sha256_json(interpretation)}
        self.assertTrue(v2013_lineage_matches(stage_input_hashes=hashes,
            result_input_provenance=provenance, preparation=preparation,
            interpretation=interpretation))
        for field in ("v20.12Preparation", "v20.12Interpretation"):
            with self.subTest(stage_field=field):
                changed = {**hashes, field: "0" * 64}
                self.assertFalse(v2013_lineage_matches(stage_input_hashes=changed,
                    result_input_provenance=provenance, preparation=preparation,
                    interpretation=interpretation))
        for field in ("v20.12PreparationResultHash", "v20.12InterpretationResultHash"):
            with self.subTest(result_field=field):
                changed = {**provenance, field: "0" * 64}
                self.assertFalse(v2013_lineage_matches(stage_input_hashes=hashes,
                    result_input_provenance=changed, preparation=preparation,
                    interpretation=interpretation))

    @staticmethod
    def _supported_sector(sector, label):
        return {"sector": sector, "sectorClassification": label,
                "fairAllModelComparisonCompleted": True, "failClosedReasons": []}

    def test_predictive_replication_requires_two_distinct_sectors(self):
        for label in ("SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION_PREDICTIVELY_VALIDATED",
                      "COHERENT_TWO_MODE_BEATING_PREDICTIVELY_VALIDATED",
                      "EPISODIC_TARGET_MODE_ACTIVATION_PREDICTIVELY_VALIDATED"):
            with self.subTest(label=label):
                one = adjudicate_predictive_sectors([self._supported_sector(1, label)])
                self.assertEqual(UNRESOLVED, one["classification"])
                two = adjudicate_predictive_sectors([
                    self._supported_sector(1, label), self._supported_sector(2, label)])
                self.assertEqual(label, two["classification"])

    def test_duplicate_and_invalid_sector_ids_fail_closed(self):
        label = "COHERENT_TWO_MODE_BEATING_PREDICTIVELY_VALIDATED"
        duplicate = adjudicate_predictive_sectors([
            self._supported_sector(3, label), self._supported_sector(3, label)])
        self.assertEqual(UNRESOLVED, duplicate["classification"])
        self.assertTrue(any("duplicate" in reason for reason in duplicate["failClosedReasons"]))
        invalid = adjudicate_predictive_sectors([
            self._supported_sector(1, label), self._supported_sector(None, label)])
        self.assertEqual(UNRESOLVED, invalid["classification"])
        self.assertTrue(any("valid persisted" in reason for reason in invalid["failClosedReasons"]))

    def test_two_distinct_replicated_mechanisms_remain_unresolved(self):
        smooth = "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION_PREDICTIVELY_VALIDATED"
        beating = "COHERENT_TWO_MODE_BEATING_PREDICTIVELY_VALIDATED"
        result = adjudicate_predictive_sectors([
            self._supported_sector(1, smooth), self._supported_sector(2, smooth),
            self._supported_sector(3, beating), self._supported_sector(4, beating)])
        self.assertEqual(UNRESOLVED, result["classification"])
        self.assertEqual(2, len(result["replicatedPredictiveMechanisms"]))

    def test_unfair_sector_never_contributes_replication_support(self):
        label = "COHERENT_TWO_MODE_BEATING_PREDICTIVELY_VALIDATED"
        unfair = self._supported_sector(2, label)
        unfair["fairAllModelComparisonCompleted"] = False
        unfair["failClosedReasons"] = ["intermittent fit failed"]
        result = adjudicate_predictive_sectors([self._supported_sector(1, label), unfair])
        self.assertEqual(UNRESOLVED, result["classification"])
        self.assertEqual([], result["replicatedPredictiveMechanisms"])

    def _complete(self, store, investigation, stage_id, handler_id, result):
        running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
        investigation = store.append_running_stage(investigation, running)
        terminal = store.build_terminal_stage(
            stage_id=stage_id, handler_id=handler_id, status="COMPLETE",
            triggered_by_stage_id=None, parameters={}, result=result, error=None,
            software_id="predictive-finalization-regression",
            software_version="20.16", started_at=running.started_at,
        )
        return store.complete_current_stage(investigation, terminal)

    def _finalization_history(
        self, store, include_predictive=True, *, predictive_overrides=None,
        later_unrelated_recommendation=None,
    ):
        investigation = store.create("predictive-finalization", WORKFLOW_ID, WORKFLOW_VERSION)
        stages = [
            ("001-prepare-target", "openstar.tess.prepare-target", {
                "datasetID": "tic-350519062", "ticID": 350519062,
                "targetName": "TIC 350519062", "sector": 10,
            }),
            ("002-hypotheses", "openstar.tess.hypotheses", {
                "rawCandidatePeriodDays": 1.25, "observedPeriodDays": 2.5,
            }),
            ("003-planner", "openstar.tess.planner", {
                "claimDecision": {"claim": "CANDIDATE_PERIOD", "rationale": [
                    "Established main periodic-family evidence is preserved."
                ]},
            }),
            ("020-multisource", "openstar.tess.multi-source-residual.interpret", {
                "classification": "TARGET_RESIDUAL_COMPONENT_DOMINANT",
                "recommendedNextTest": "STALE_TEMPORAL_MECHANISM_TEST",
            }),
        ]
        predictive = {
            "classification": "TARGET_RESIDUAL_MECHANISM_PREDICTIVE_VALIDATION_UNRESOLVED",
            "recommendedNextTest": "ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION",
            "replicatedPredictiveMechanisms": [],
            "replicatedPredictiveMechanismSupportingSectorIDs": {},
            "failClosedReasons": [
                "sector 69 smooth-amplitude training fit failed conservatively"
            ],
            "sectorPredictiveEvidence": [{
                "sector": 69,
                "sectorClassification": "TARGET_RESIDUAL_MECHANISM_PREDICTIVE_VALIDATION_UNRESOLVED",
                "bestPredictiveModel": "CONSTANT_AMPLITUDE",
                "secondBestPredictiveModel": "COHERENT_TWO_MODE_BEATING",
                "predictiveDeltaLogLikelihood": 1.75,
                "foldWinsByModel": {"CONSTANT_AMPLITUDE": 4},
                "fairAllModelComparisonCompleted": False,
                "morphologyGateBlockedPromotion": False,
                # Real results contain detailed fit payloads; the report must not dump them.
                "modelEvidence": [{"coefficients": [1.0, 2.0, 3.0]}],
            }],
        }
        predictive.update(predictive_overrides or {})
        if include_predictive:
            stages.append((
                "030-target-residual-mechanism-predictive-validation",
                "openstar.tess.target-residual-mechanism-predictive-validation.analyze",
                predictive,
            ))
        if later_unrelated_recommendation is not None:
            stages.append((
                "032-offset-source-identification",
                "openstar.tess.offset-source-identification.analyze",
                {
                    "classification": "CATALOG_COUNTERPART_IDENTIFIED",
                    "recommendedNextTest": later_unrelated_recommendation,
                },
            ))
        for stage in stages:
            investigation = self._complete(store, investigation, *stage)
        return investigation, predictive

    def test_finalizer_reports_persisted_v2016_without_rerun_or_distributed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory) / "investigations")
            investigation, predictive = self._finalization_history(store)
            frozen_stages = copy.deepcopy(investigation.stages)
            coordinator = types.SimpleNamespace(
                create_project=lambda *args, **kwargs: self.fail("distributed work started")
            )
            engine = build_engine(store, coordinator=coordinator, poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            with patch(
                "workflows.tess.tess_investigation.analyze_predictive_validation",
                side_effect=AssertionError("v20.16 was rerun"),
            ):
                completed, next_request = engine.run_stage(
                    investigation,
                    StageRequest("031-finalize", "openstar.tess.finalize", {},
                                 "030-target-residual-mechanism-predictive-validation"),
                    software_id="integration", software_version="20.33",
                )

            self.assertIsNone(next_request)
            self.assertEqual("COMPLETE", completed.status)
            self.assertEqual(frozen_stages, completed.stages[:-1])
            conclusion = completed.stages[-1].result
            self.assertEqual(predictive,
                conclusion["targetResidualMechanismPredictiveValidation"])
            self.assertEqual(predictive["recommendedNextTest"],
                             conclusion["recommendedNextTest"])
            self.assertEqual(2.5,
                conclusion["periodEvidence"]["recurrentPhotometricPeriodDays"])
            report = Path(conclusion["reportPath"]).read_text(encoding="utf-8")
            self.assertIn("remains unresolved", report)
            self.assertIn("Conservative predictive-validation limitations", report)
            self.assertIn("Sector 69", report)
            self.assertIn("deltaLogLikelihood=1.75", report)
            self.assertIn("no distributed work or archive query", report)
            self.assertIn(
                "v20.16 recommends the next target-residual test: "
                "ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION.",
                report,
            )
            self.assertNotIn("coefficients", report)

    def test_validated_v2016_reports_its_persisted_branch_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory) / "investigations")
            investigation, _ = self._finalization_history(store, predictive_overrides={
                "classification":
                    "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION_PREDICTIVELY_VALIDATED",
                "recommendedNextTest": "ASTROPHYSICAL_MECHANISM_INTERPRETATION",
                "replicatedPredictiveMechanisms": ["SMOOTH_AMPLITUDE_MODULATION"],
                "replicatedPredictiveMechanismSupportingSectorIDs": {
                    "SMOOTH_AMPLITUDE_MODULATION": [10, 69],
                },
                "failClosedReasons": [],
            })
            engine = build_engine(store, coordinator=types.SimpleNamespace(),
                                  poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            completed, _ = engine.run_stage(
                investigation,
                StageRequest("031-finalize", "openstar.tess.finalize", {},
                             "030-target-residual-mechanism-predictive-validation"),
                software_id="integration", software_version="20.33",
            )
            conclusion = completed.stages[-1].result
            report = Path(conclusion["reportPath"]).read_text(encoding="utf-8")
            self.assertEqual("ASTROPHYSICAL_MECHANISM_INTERPRETATION",
                             conclusion["recommendedNextTest"])
            self.assertIn(
                "v20.16 recommends the next target-residual test: "
                "ASTROPHYSICAL_MECHANISM_INTERPRETATION.", report,
            )
            self.assertNotIn("additional independent temporal baseline", report)

    def test_later_unrelated_recommendation_keeps_v2016_branch_rationale(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory) / "investigations")
            later = "DIRECT_COUNTERPART_VARIABILITY_VALIDATION"
            investigation, predictive = self._finalization_history(
                store, later_unrelated_recommendation=later,
            )
            engine = build_engine(store, coordinator=types.SimpleNamespace(),
                                  poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            completed, _ = engine.run_stage(
                investigation,
                StageRequest("033-finalize", "openstar.tess.finalize", {},
                             "032-offset-source-identification"),
                software_id="integration", software_version="20.33",
            )
            conclusion = completed.stages[-1].result
            report = Path(conclusion["reportPath"]).read_text(encoding="utf-8")
            self.assertEqual(later, conclusion["recommendedNextTest"])
            self.assertIn(
                "v20.16 recommends the next target-residual test: "
                f"{predictive['recommendedNextTest']}.", report,
            )
            self.assertNotIn(
                f"The investigation-wide next step is {predictive['recommendedNextTest']}",
                report,
            )

    def test_history_without_v2016_keeps_existing_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory) / "investigations")
            investigation, _ = self._finalization_history(store, include_predictive=False)
            engine = build_engine(store, coordinator=types.SimpleNamespace(),
                                  poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            completed, next_request = engine.run_stage(
                investigation,
                StageRequest("021-finalize", "openstar.tess.finalize", {}, "020-multisource"),
                software_id="integration", software_version="20.33",
            )
            self.assertIsNone(next_request)
            conclusion = completed.stages[-1].result
            self.assertIsNone(conclusion["targetResidualMechanismPredictiveValidation"])
            self.assertEqual("STALE_TEMPORAL_MECHANISM_TEST",
                             conclusion["recommendedNextTest"])


if __name__ == "__main__":
    unittest.main()
