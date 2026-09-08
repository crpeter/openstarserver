import copy
import json
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from openstar_investigation import ArtifactReference, InvestigationStore, sha256_file, sha256_json
from openstar_workflow import StageRequest, RetryableExecutionError
from workflows.tess import tess_eclipse_common_support as common
from workflows.tess import tess_eclipse_common_support_continuation as recorded
from workflows.tess import tess_recorded_eclipse_evidence as evidence
from workflows.tess import tess_investigation as workflow
from workflows.tess.tess_external_companion_evidence import (
    ExternalEvidenceTransientError, acquire_external_evidence, interpret_external_evidence,
    localization_gate, review_source_attribution,
)
from workflows.tess.tess_companion_evidence_synthesis import synthesize_companion_evidence
from workflows.tess.tess_joint_event_phase_model import RESULT_VERSION as MODEL_VERSION
import test_tess_eclipse_common_support_continuation as recorded_fixtures
import test_tess_external_companion_evidence as external_fixtures


class RecordedEclipseEvidenceTests(unittest.TestCase):
    def setUp(self):
        # A real v1 conflict -> v2 reanalysis -> recorded continuation. Only
        # downstream archive/photometry operations are replaced in lifecycle tests.
        self.fixture = recorded_fixtures.CommonSupportContinuationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.run_continuation(True)
        self.base = self.fixture.fixture
        self.store = self.base.store
        self.investigation = self.store.load(self.base.investigation.id)
        self.localization = self.fixture.result
        self.before = self.snapshot()

    def snapshot(self):
        return {str(p): p.read_bytes() for p in self.base.state.rglob("*") if p.is_file()}

    def run_evidence(self, execute=False, retry_failed=False):
        return evidence.run_evidence(self.base.state, self.investigation.id,
                                     execute=execute, retry_failed=retry_failed)

    def review(self):
        return review_source_attribution(self.localization, allow_common_support_v2=True)

    def frozen_external(self, review):
        row = external_fixtures.ExternalCompanionEvidenceTests().row(gaia_dr3_id=None)
        return acquire_external_evidence(
            review, opener=lambda *a, **k: external_fixtures.Response(json.dumps([row]).encode()),
            retrieved_at="2026-01-01T00:00:00+00:00")

    @contextmanager
    def downstream(self, *, archive_error=None):
        """Use existing registered stages with deterministic downstream providers.

The source fixture intentionally has no distributed-search project. Stub those
legacy report inputs; retain the real v2 lineage selector, review, archive parser,
external interpretation, synthesis, workflow engine, and finalizer persistence.
"""
        calls = []
        photometry = {"resultVersion": "openstar.tess-event-depth-photometry-freeze.v1",
                      "status": "FROZEN", "sectors": [], "freezeSHA256": "f" * 64}
        audit = {"resultVersion": "openstar.tess-event-depth-attenuation-audit.v1",
                 "status": "COMPLETE", "externalCatalogInformationUsed": False,
                 "catalogAnswerKeyUsed": False, "suitableForLaterPrecisionModeling": True,
                 "recommendedNextTest": "JOINT_TRANSIT_ECLIPSE_PHASE_CURVE_MODELING"}
        audit["auditSHA256"] = sha256_json(audit)
        model = {"resultVersion": MODEL_VERSION, "status": "UNRESOLVED",
                 "classification": "PRECISION_EMPIRICAL_TRANSIT_DEPTH_UNRESOLVED",
                 "precisionEmpiricalTransitDepthResolved": False, "globalFit": {},
                 "resolutionGates": {}, "unresolvedReasons": ["SYNTHETIC_TEST"]}
        model["modelSHA256"] = sha256_json(model)

        def provider(label, value):
            def call(*args, **kwargs):
                calls.append(label)
                return copy.deepcopy(value)
            return call

        def archive(review):
            calls.append("external")
            if archive_error:
                raise archive_error
            return self.frozen_external(review)

        real_result, real_latest = workflow._result, workflow._latest_result_for_handler

        def prepared(investigation, stage_id):
            if stage_id == "001-prepare-target":
                return {"ticID": 42, "datasetID": "synthetic", "targetName": "Synthetic", "sector": 1}
            return real_result(investigation, stage_id)

        def latest(investigation, handler):
            if handler == "openstar.tess.hypotheses":
                return {"observedPeriodDays": 2.0}
            if handler == "openstar.tess.planner":
                return {"claimDecision": {"claim": "CANDIDATE_PERIOD", "rationale": []}}
            return real_latest(investigation, handler)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(workflow, "_result", side_effect=prepared))
            stack.enter_context(mock.patch.object(workflow, "_latest_result_for_handler", side_effect=latest))
            stack.enter_context(mock.patch.object(workflow, "_build_period_evidence", return_value={
                "physicalPeriodDays": 2.0, "physicalCycleResolved": True,
                "candidateSource": "synthetic-frozen-ephemeris"}))
            stack.enter_context(mock.patch.object(workflow, "_render_report", side_effect=lambda value: json.dumps(value)))
            stack.enter_context(mock.patch.object(workflow, "acquire_full_precision_photometry",
                                                  side_effect=provider("photometry", photometry)))
            stack.enter_context(mock.patch.object(workflow, "audit_depth_attenuation",
                                                  side_effect=provider("audit", audit)))
            stack.enter_context(mock.patch.object(workflow, "fit_joint_event_phase_model",
                                                  side_effect=provider("model", model)))
            stack.enter_context(mock.patch.object(workflow, "acquire_external_evidence", side_effect=archive))
            yield calls

    def test_default_validates_without_writes_or_archive_calls(self):
        with mock.patch.object(workflow, "acquire_full_precision_photometry") as photometry, \
                mock.patch.object(workflow, "acquire_external_evidence") as archive:
            result = self.run_evidence()
        self.assertEqual(result["status"], "VALIDATED_NO_CHANGES")
        self.assertEqual(result["sourceReview"]["classification"], "TARGET_SOURCE_ATTRIBUTION_REVIEW_PASSED")
        self.assertEqual(result["nextStage"]["handler_id"], evidence.REVIEW_HANDLER_ID)
        self.assertEqual(self.before, self.snapshot())
        photometry.assert_not_called()
        archive.assert_not_called()

    def test_v2_requires_explicit_opt_in_and_keeps_its_identity(self):
        self.assertFalse(localization_gate(self.localization))
        with self.assertRaises(ValueError):
            review_source_attribution(self.localization)
        result = self.review()
        self.assertEqual(result["sourceLocalizationVersion"], common.RESULT_VERSION)
        self.assertEqual(result["sourceLocalizationSHA256"], sha256_json(self.localization))
        bare = copy.deepcopy(self.localization)
        bare.pop("methodContract")
        self.assertFalse(localization_gate(bare, allow_common_support_v2=True))

    def test_v1_review_is_unchanged(self):
        fixture = external_fixtures.ExternalCompanionEvidenceTests()
        fixture.setUp()
        self.assertEqual(review_source_attribution(fixture.localization), fixture.review)
        self.assertNotIn("sourceLocalizationVersion", fixture.review)

    def test_conflict_or_primary_cannot_supply_independent_replication(self):
        for kind in ("conflict", "primary"):
            with self.subTest(kind=kind):
                value = copy.deepcopy(self.localization)
                if kind == "conflict":
                    value["sectorResults"][-1].update(matchedCatalogHypothesis="NEIGHBOR",
                                                       classification="CATALOG_CANDIDATE_CONSISTENT")
                else:
                    value["sectorResults"][-1]["role"] = "PRIMARY"
                self.assertFalse(review_source_attribution(value, allow_common_support_v2=True)["sourceAttributionReviewPassed"])

    def test_registered_review_uses_recorded_v2_and_binds_both_ledgers(self):
        request, _, lineage = evidence._next_request(self.store, self.investigation)
        engine = workflow.build_engine(self.store, None, poll_interval=0, timeout=None)
        investigation, next_request = engine.run_stage(self.investigation, request,
                                                       software_id="test", software_version="1")
        stage = investigation.stages[-1]
        self.assertEqual(stage.result, self.review())
        self.assertEqual(stage.provenance.input_hashes, {"sourceLocalization": sha256_json(self.localization), **lineage})
        self.assertEqual(next_request.handler_id, evidence.DEPTH_FREEZE)
        self.assertEqual(investigation.stages[:-1], self.investigation.stages)

    def test_full_lifecycle_preserves_history_and_orders_depth_before_external(self):
        with self.downstream() as calls:
            result = self.run_evidence(True)
        self.assertEqual(calls, ["photometry", "audit", "model", "external"])
        self.assertIn(result["status"], {"COMPLETE", "HUMAN_REVIEW_REQUIRED"})
        current = self.store.load(self.investigation.id)
        self.assertEqual(current.stages[:len(self.investigation.stages)], self.investigation.stages)
        for path, data in self.before.items():
            if path != str(self.store.path_for(current.id)):
                self.assertEqual(Path(path).read_bytes(), data)
        conclusion = json.loads(Path(result["conclusionPath"]).read_text())
        self.assertEqual(conclusion["eclipseEventSourceLocalization"], self.localization)
        self.assertTrue(conclusion["sourceAttributionReview"]["sourceAttributionReviewPassed"])
        self.assertFalse(conclusion["automaticDiscoveryClaim"])
        self.assertTrue(Path(result["reportPath"]).name.endswith("-common-support-v2.md"))
        after = self.snapshot()
        self.assertEqual(self.run_evidence(True)["status"], "ALREADY_COMPLETE")
        self.assertEqual(after, self.snapshot())

    def test_resume_from_completed_review_does_not_repeat_review(self):
        request, _, _ = evidence._next_request(self.store, self.investigation)
        engine = workflow.build_engine(self.store, None, poll_interval=0, timeout=None)
        engine.run_stage(self.investigation, request, software_id="test", software_version="1")
        with self.downstream() as calls:
            self.run_evidence(True)
        self.assertEqual(calls, ["photometry", "audit", "model", "external"])
        current = self.store.load(self.investigation.id)
        self.assertEqual(sum(s.handler_id == evidence.REVIEW_HANDLER_ID for s in current.stages), 1)

    def test_transient_external_retry_preserves_failed_stage_and_frozen_depths(self):
        with self.downstream(archive_error=ExternalEvidenceTransientError("synthetic outage")) as calls:
            with self.assertRaises(RetryableExecutionError):
                self.run_evidence(True)
        self.assertEqual(calls, ["photometry", "audit", "model", "external"])
        failed = self.store.load(self.investigation.id)
        self.assertEqual(failed.status, "FAILED")
        failed_ledger = self.store.stage_path_for(failed.id, failed.stages[-1].id)
        saved_ledger = failed_ledger.read_bytes()
        with self.assertRaisesRegex(ValueError, "retry"):
            self.run_evidence(True)
        with self.downstream() as calls:
            self.run_evidence(True, retry_failed=True)
        self.assertEqual(calls, ["external"])
        self.assertEqual(failed_ledger.read_bytes(), saved_ledger)

    def test_v2_synthesis_requires_opt_in_and_exact_v2_review_hash(self):
        review = self.review()
        frozen = self.frozen_external(review)
        external = interpret_external_evidence(frozen)
        args = (self.base.binary, self.localization, review, frozen, external)
        with self.assertRaisesRegex(ValueError, "localization version"):
            synthesize_companion_evidence(*args)
        result = synthesize_companion_evidence(*args, allow_common_support_v2=True)
        self.assertEqual(result["sourceRelationship"], "TARGET_ASSOCIATED")
        wrong_review = {**review, "sourceLocalizationSHA256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "review"):
            synthesize_companion_evidence(self.base.binary, self.localization, wrong_review,
                                         frozen, external, allow_common_support_v2=True)

    def test_changed_recorded_bytes_block_execution_before_writes(self):
        accepted = self.investigation.stages[-2]
        path = next(Path(a.path) for a in accepted.artifacts if a.path.endswith("reviewed-reanalysis-v2.json"))
        path.write_text("{}")
        before = self.snapshot()
        with mock.patch.object(workflow, "build_engine") as engine:
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                self.run_evidence(True)
            engine.assert_not_called()
        self.assertEqual(before, self.snapshot())

    def test_missing_recorded_stage_cannot_fall_back_to_old_conflict(self):
        broken = replace(self.investigation, stages=self.investigation.stages[:-2] + (self.investigation.stages[-1],))
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            workflow._eclipse_localization_for_evidence(self.store, broken)

    def test_rehashed_parent_change_is_rejected_by_preserved_snapshot(self):
        old_stage = self.investigation.stages[0]
        result = {**old_stage.result, "tampered": True}
        artifacts = []
        for artifact in old_stage.artifacts:
            Path(artifact.path).write_text(json.dumps(result))
            artifacts.append(ArtifactReference(artifact.path, sha256_file(artifact.path), artifact.media_type))
        changed = replace(old_stage, result=result, artifacts=tuple(artifacts),
                          provenance=replace(old_stage.provenance, result_hash=sha256_json(result)))
        self.store.stage_path_for(self.investigation.id, changed.id).write_text(json.dumps(asdict(changed)))
        altered = replace(self.investigation, stages=(changed,) + self.investigation.stages[1:])
        self.store.save(altered)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Preserved investigation prefix"):
            self.run_evidence(True)
        self.assertEqual(before, self.snapshot())

    def test_persisted_request_cannot_skip_depth_chronology(self):
        review = replace(self.investigation.stages[-1], id="008-review",
                         handler_id=evidence.REVIEW_HANDLER_ID, triggered_by_stage_id=self.investigation.stages[-1].id,
                         parameters={}, result=self.review(), stop=False,
                         next_stage={"id": "009-external", "handler_id": evidence.FREEZE_HANDLER_ID,
                                     "parameters": {}, "triggered_by_stage_id": "008-review"})
        with self.assertRaisesRegex(ValueError, "chronology"):
            evidence._verify_tail(self.investigation.stages[-1], (review,))


if __name__ == "__main__":
    unittest.main()
