from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from openstar_investigation import (ArtifactReference, InvestigationStage, InvestigationStore,
                                    sha256_file, sha256_json)
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import WORKFLOW_ID, repair_obsolete_terminal_wait
from workflows.tess.tess_target_residual_mechanism_adjudication import (
    adjudicate_frozen_target_residual_mechanism,
)


class TessTargetResidualMechanismAdjudicationTests(unittest.TestCase):
    def frozen(self, root: Path):
        evidence = []
        for sector in (69, 70):
            evidence.append({"sector": sector, "constantAmplitudeBIC": 25000.0,
                "smoothEnvelopeBIC": 24868.075748741066,
                "twoFrequencyBIC": 24839.166068701925,
                "intermittentEnvelopeBIC": 25100.0,
                "episodicSuppressionAndReappearance": False,
                "sectorClassification": "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"})
        result = {"classification": "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION",
                  "physicalMechanismResolved": False, "sectorModelEvidence": evidence,
                  "failClosedReasons": [], "crossSectorPhaseUsed": False}
        path = root / "target-residual-mechanism-v20.14.json"
        path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        artifact = ArtifactReference(str(path), sha256_file(path), "application/json")
        return result, artifact

    def test_verified_frozen_result_is_reinterpreted_without_refitting_or_coordinator(self):
        with tempfile.TemporaryDirectory() as directory:
            result, artifact = self.frozen(Path(directory))
            original = copy.deepcopy(result)
            with patch("workflows.tess.tess_target_residual_mechanism._model_sector",
                       side_effect=AssertionError("must not refit")), \
                 patch("openstar_coordinator_client.OpenStarCoordinatorClient.run_project",
                       side_effect=AssertionError("must not distribute")):
                corrected = adjudicate_frozen_target_residual_mechanism(
                    v2014_result=result, authoritative_v2014_artifacts=(artifact,))
        self.assertEqual(original, result)
        self.assertEqual("COHERENT_TWO_MODE_BEATING_SUPPORTED", corrected["classification"])
        self.assertFalse(corrected["newModelFittingPerformed"])
        self.assertFalse(corrected["distributedWorkPerformed"])
        self.assertEqual(artifact.sha256,
            corrected["inputProvenance"]["frozenV20.14ArtifactSHA256"])

    def test_mutated_artifact_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result, artifact = self.frozen(Path(directory))
            Path(artifact.path).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA verification"):
                adjudicate_frozen_target_residual_mechanism(
                    v2014_result=result, authoritative_v2014_artifacts=(artifact,))

    def test_falsified_persisted_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result, artifact = self.frozen(Path(directory))
            result["classification"] = "FALSIFIED"
            with self.assertRaisesRegex(RuntimeError, "persisted result differ"):
                adjudicate_frozen_target_residual_mechanism(
                    v2014_result=result, authoritative_v2014_artifacts=(artifact,))

    def test_exact_old_v2014_pending_finalizer_repairs_once_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create("real-old-v2014", WORKFLOW_ID, "20.2")
            result, artifact = self.frozen(Path(directory))
            stage = InvestigationStage("028-target-residual-mechanism",
                "openstar.tess.target-residual-mechanism.analyze", "COMPLETE", "027-old", {},
                result=result, artifacts=(artifact,), next_stage=asdict(StageRequest(
                    "029-finalize", "openstar.tess.finalize",
                    {"outputSuffix": "v20.14-intrinsic"},
                    "028-target-residual-mechanism")))
            investigation = replace(investigation, stages=(stage,))
            store.save(investigation)
            finalizer = StageRequest("029-finalize", "openstar.tess.finalize",
                                     {"outputSuffix": "v20.14-intrinsic"}, stage.id)
            investigation = store.set_control_state(investigation, status="RUNNING",
                control_state={"branchAssessments": [], "schedulerAction": "RUN_EXPERIMENT",
                               "selectedExperiment": asdict(finalizer)})
            old_stage_file = store.directory_for(investigation.id).joinpath(
                "stages", f"{stage.id}.json")
            # This synthetic fixture predates append_running_stage, so immutability
            # is asserted over the persisted snapshot and stage tuple.
            old_snapshot = store.path_for(investigation.id).read_bytes()
            repaired = repair_obsolete_terminal_wait(store, investigation)
            repeated = repair_obsolete_terminal_wait(store, repaired)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("openstar.tess.target-residual-mechanism-adjudication.analyze",
                             selected["handler_id"])
            self.assertEqual("029-target-residual-mechanism-adjudication", selected["id"])
            self.assertEqual(investigation.stages, repaired.stages)
            self.assertEqual(repaired, repeated)
            self.assertNotEqual(old_snapshot, store.path_for(investigation.id).read_bytes())
            self.assertFalse(old_stage_file.exists())

    def test_old_v2014_repair_requires_exact_persisted_next_stage_and_control(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            base = store.create("old-v2014-negative", WORKFLOW_ID, "20.2")
            result, artifact = self.frozen(Path(directory))
            exact = asdict(StageRequest("029-finalize", "openstar.tess.finalize",
                {"outputSuffix": "v20.14-intrinsic"}, "028-target-residual-mechanism"))
            stage = InvestigationStage("028-target-residual-mechanism",
                "openstar.tess.target-residual-mechanism.analyze", "COMPLETE", "027-old", {},
                result=result, artifacts=(artifact,), next_stage=exact)
            cases = (
                (replace(stage, next_stage={**exact,
                    "parameters": {"outputSuffix": "altered"}}), exact),
                (replace(stage, next_stage={**exact, "id": "030-finalize"}), exact),
                (stage, {**exact, "id": "030-finalize"}),
                (replace(stage, next_stage=None), exact),
            )
            for index, (candidate_stage, selected) in enumerate(cases):
                with self.subTest(case=index):
                    candidate = replace(base, id=f"negative-{index}", status="RUNNING",
                        stages=(candidate_stage,), metadata={"controlState": {
                            "branchAssessments": [], "schedulerAction": "RUN_EXPERIMENT",
                            "selectedExperiment": selected}})
                    self.assertEqual(candidate,
                                     repair_obsolete_terminal_wait(store, candidate))

    def test_corrected_finalized_and_unrelated_histories_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            base = store.create("not-exact-old-boundary", WORKFLOW_ID, "20.2")
            result, artifact = self.frozen(Path(directory))
            cases = (
                InvestigationStage("028-target-residual-mechanism",
                    "openstar.tess.target-residual-mechanism.analyze", "COMPLETE", None, {},
                    result={**result, "adjudicationVersion": "route-independent-all-models-v1"},
                    artifacts=(artifact,)),
                InvestigationStage("027-target-residual-mechanism",
                    "openstar.tess.target-residual-mechanism.analyze", "COMPLETE", None, {},
                    result=result, artifacts=(artifact,)),
            )
            for index, stage in enumerate(cases):
                candidate = replace(base, id=f"case-{index}", stages=(stage,), status="COMPLETE",
                    metadata={"controlState": {"schedulerAction": "INVESTIGATION_COMPLETE",
                                               "selectedExperiment": None}})
                self.assertEqual(candidate, repair_obsolete_terminal_wait(store, candidate))

    def _v2015_boundary(self, root: Path, *, artifact_value=None, malformed=False,
                        wrong_provenance=False):
        v14, v14_artifact = self.frozen(root)
        v15 = {"classification": "TARGET_RESIDUAL_MECHANISM_UNRESOLVED",
            "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP",
            "physicalMechanismResolved": False, "failClosedReasons": [],
            "inputProvenance": {"frozenV20.14ResultHash": sha256_json(v14),
                "frozenV20.14ArtifactSHA256": v14_artifact.sha256}}
        if wrong_provenance:
            v15["inputProvenance"]["frozenV20.14ResultHash"] = "0" * 64
        path = root / "target-residual-mechanism-adjudication-v20.15.json"
        path.write_text("{" if malformed else json.dumps(
            v15 if artifact_value is None else artifact_value, sort_keys=True), encoding="utf-8")
        v15_artifact = ArtifactReference(str(path), sha256_file(path), "application/json")
        v14_stage = InvestigationStage("028-target-residual-mechanism",
            "openstar.tess.target-residual-mechanism.analyze", "COMPLETE", None, {},
            result=v14, artifacts=(v14_artifact,))
        expected = asdict(StageRequest("030-finalize", "openstar.tess.finalize",
            {"outputSuffix": "v20.15-intrinsic-corrective-adjudication"},
            "029-target-residual-mechanism-adjudication"))
        v15_stage = InvestigationStage("029-target-residual-mechanism-adjudication",
            "openstar.tess.target-residual-mechanism-adjudication.analyze", "COMPLETE",
            v14_stage.id, {}, result=v15, artifacts=(v15_artifact,), next_stage=expected)
        return (v14_stage, v15_stage), expected

    def test_exact_v2015_pending_finalizer_schedules_predictive_validation_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root, store = Path(directory), InvestigationStore(Path(directory) / "store")
            base = store.create("v2015-boundary", WORKFLOW_ID, "20.2")
            stages, expected = self._v2015_boundary(root)
            current = replace(base, status="RUNNING", stages=stages, metadata={"controlState": {
                "branchAssessments": [], "schedulerAction": "RUN_EXPERIMENT",
                "selectedExperiment": expected}})
            repaired = repair_obsolete_terminal_wait(store, current)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("openstar.tess.target-residual-mechanism-predictive-validation.analyze",
                             selected["handler_id"])
            self.assertEqual(stages, repaired.stages)
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))

    def test_v2015_artifact_result_or_provenance_mismatch_leaves_history_unchanged(self):
        cases = ({"artifact_value": {"different": True}}, {"malformed": True},
                 {"wrong_provenance": True})
        for index, options in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                root, store = Path(directory), InvestigationStore(Path(directory) / "store")
                base = store.create(f"negative-v2015-{index}", WORKFLOW_ID, "20.2")
                stages, expected = self._v2015_boundary(root, **options)
                current = replace(base, status="RUNNING", stages=stages,
                    metadata={"controlState": {"branchAssessments": [],
                        "schedulerAction": "RUN_EXPERIMENT", "selectedExperiment": expected}})
                self.assertEqual(current, repair_obsolete_terminal_wait(store, current))


if __name__ == "__main__":
    unittest.main()
