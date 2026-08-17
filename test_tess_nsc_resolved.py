from __future__ import annotations

import tempfile
import sys
import types
import unittest
import urllib.error
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openstar_investigation import Investigation, InvestigationStage, InvestigationStore
from openstar_autonomy import AutonomousInvestigationEngine
from openstar_dispatch import InvestigationDispatcher
from openstar_lifecycle import InvestigationLifecycleLoop
from openstar_targets import InvestigationTarget, InvestigationTargetPortfolio
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("numpy")
    stub.integer = int
    stub.floating = float
    sys.modules["numpy"] = stub
    _installed_numpy_stub = True
else:
    _installed_numpy_stub = False

from workflows.tess import tess_nsc_resolved as nsc
from workflows.tess.tess_autonomy import plan_tess_branches, repair_obsolete_terminal_wait
from workflows.tess.tess_investigation import build_engine

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


class _SingleTargetSource:
    id = "nsc-test-targets"
    version = "1"

    def __init__(self, target):
        self.target = target

    def enumerate_targets(self):
        return (self.target,)


class CurrentNSCResolvedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pair = {
            "version": "openstar.current-source-pair.v1",
            "target": {"sourceRole": "target-control", "gaiaDR3SourceID": 101,
                       "raDeg": 12.0, "decDeg": -30.0},
            "counterpart": {"sourceRole": "catalog-counterpart", "gaiaDR3SourceID": 202,
                            "raDeg": 12.0005, "decDeg": -30.0},
            "separationArcsec": 1.56,
        }
        self.gaia = {"sourcePair": self.pair, "frequencySearch": {
            "minimumFrequencyPerDay": 1.0, "maximumFrequencyPerDay": 2.0,
            "totalFrequencies": 20, "frequenciesPerWorkUnit": 10}}
        self.sky = {"recommendedNextTest": "NSC_RESOLVED_COUNTERPART_PHOTOMETRY",
                    "physicalMechanismResolved": False}

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return nsc.build_nsc_resolved_project(
            source_project_id="generic-project", source_dataset_id="generic-dataset",
            external_high_resolution_summary=self.gaia, skymapper_summary=self.sky,
            output_dir=self.root, investigation_id="generic-investigation")

    @mock.patch.object(nsc, "_query_object_candidates", return_value=[])
    def test_current_source_pair_is_consumed_and_no_match_is_scientific(self, query):
        spec = self.build()
        self.assertEqual(self.pair, spec["sourcePair"])
        self.assertFalse(spec["available"])
        self.assertEqual([], spec["errors"])
        self.assertEqual([101, 202], [call.args[0]["gaiaDR3SourceID"] for call in query.call_args_list])
        summary = nsc.interpret_nsc_resolved_project(project_status=None, preparation=spec)
        self.assertEqual(nsc.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])

    @mock.patch.object(nsc, "_query_object_candidates")
    def test_one_sided_match_and_unresolved_pair_are_scientific(self, query):
        query.side_effect = [[{"id": "generic-object", "ra": "12", "dec": "-30"}], []]
        spec = self.build()
        self.assertFalse(spec["pairSeparatelyResolvedInNSC"])
        self.assertEqual(nsc.NEXT_ARCHIVE_TEST,
                         nsc.interpret_nsc_resolved_project(project_status=None,
                                                            preparation=spec)["recommendedNextTest"])

    @mock.patch.object(nsc, "_query_object_candidates", side_effect=TimeoutError("outage"))
    def test_transient_archive_failure_is_narrowly_classified(self, _query):
        with self.assertRaises(nsc.NSCArchiveUnavailable):
            self.build()

    @mock.patch.object(nsc, "_query_object_candidates", side_effect=ValueError("bug"))
    def test_programming_error_is_not_retryable_or_no_data(self, _query):
        with self.assertRaises(ValueError):
            self.build()

    def test_flat_photometry_is_scientific_no_series_and_routes_to_noirlab(self):
        constant_magnitudes = [15.0] * nsc.MIN_BAND_SAMPLES
        with mock.patch.object(
            nsc, "_robust_standardize_magnitudes",
            side_effect=RuntimeError("NSC magnitudes have no finite variability scale."),
        ):
            flux, reason = nsc._standardize_series_or_quality_outcome(constant_magnitudes)
        self.assertIsNone(flux)
        self.assertEqual("NO_FINITE_VARIABILITY_SCALE", reason)
        preparation = {
            "pairSeparatelyResolvedInNSC": True,
            "preparedSeries": [],
            "coDetectionDiagnostics": {"unusableSeries": [{
                "sourceRole": "target-control", "band": "g",
                "sampleCount": nsc.MIN_BAND_SAMPLES, "reason": reason}]},
        }
        summary = nsc.interpret_nsc_resolved_project(
            project_status=None, preparation=preparation)
        self.assertEqual("NSC_DR2_NO_QUALIFYING_CODETECTED_RESOLVED_SERIES",
                         summary["classification"])
        self.assertEqual(nsc.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])
        self.assertFalse(summary["physicalMechanismResolved"])

    def test_unexpected_standardization_runtime_error_still_propagates(self):
        with mock.patch.object(nsc, "_robust_standardize_magnitudes",
                               side_effect=RuntimeError("local dependency bug")):
            with self.assertRaisesRegex(RuntimeError, "local dependency bug"):
                nsc._standardize_series_or_quality_outcome([15.0] * nsc.MIN_BAND_SAMPLES)

    def test_http_retry_status_contract(self):
        for code in (408, 425, 429, 500, 503):
            self.assertTrue(nsc._retryable_service_error(
                urllib.error.HTTPError("url", code, "x", {}, None)))
        self.assertFalse(nsc._retryable_service_error(
            urllib.error.HTTPError("url", 400, "x", {}, None)))

    def test_usable_nonrecurrent_continues_to_noirlab_and_worker_is_generic(self):
        preparation = {"pairSeparatelyResolvedInNSC": True,
                       "preparedSeries": [{"datasetID": "d", "sourceRole": "target-control",
                                           "band": "g"}],
                       "workloadID": "openstar.lomb-scargle.v1"}
        summary = nsc.interpret_nsc_resolved_project(
            project_status={"datasets": []}, preparation=preparation)
        self.assertEqual(nsc.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])
        self.assertEqual("openstar.lomb-scargle.v1",
                         summary["distributedValidation"]["workloadID"])

    def test_real_style_045_blocked_reopens_exact_046_without_archive_reruns(self):
        stage = InvestigationStage(
            "045-interpret-skymapper-resolved-counterpart-photometry",
            "openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
            "COMPLETE", "044-prepare", {}, result=self.sky, stop=True)
        investigation = Investigation(
            "generic", "openstar.workflow.tess-investigation.v1", "20.2", "BLOCKED",
            "now", "now", {"datasetID": "generic-dataset", "controlState": {
                "schedulerAction": "WAIT_FOR_PREREQUISITES"}}, (stage,))
        target = InvestigationTarget("generic", "generic", investigation.workflow_id,
                                     investigation.workflow_version)
        branch = plan_tess_branches(investigation, target)[0]
        self.assertEqual("046-prepare-nsc-resolved-counterpart-photometry", branch.experiment.id)
        self.assertEqual((), branch.required_stage_ids)
        self.assertNotIn("gaia", branch.experiment.handler_id)
        self.assertNotIn("skymapper", branch.experiment.handler_id)
        store = InvestigationStore(self.root / "state")
        store.save(investigation)
        repaired = repair_obsolete_terminal_wait(store, investigation)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(asdict(branch.experiment), repaired.metadata["controlState"]["selectedExperiment"])

    def test_completed_nsc_is_idempotent_and_not_replanned(self):
        stages = (InvestigationStage("045-sky",
            "openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
            "COMPLETE", None, {}, result=self.sky), InvestigationStage("046-prepare",
            "openstar.tess.nsc-resolved-photometry.prepare", "COMPLETE", "045-sky", {}, result={}),
            InvestigationStage("047-interpret", "openstar.tess.nsc-resolved-photometry.interpret",
                               "COMPLETE", "046-prepare", {}, result={
                                   "recommendedNextTest": nsc.NEXT_ARCHIVE_TEST,
                                   "physicalMechanismResolved": False}, stop=True))
        investigation = Investigation("i", "openstar.workflow.tess-investigation.v1", "20.2",
                                      "BLOCKED", "now", "now", {}, stages)
        target = InvestigationTarget("t", "i", investigation.workflow_id, investigation.workflow_version)
        branches = plan_tess_branches(investigation, target)
        self.assertEqual(1, len(branches))
        self.assertEqual((), branches[0].required_stage_ids)
        self.assertEqual("openstar.tess.noirlab-image-forced-photometry.prepare",
                         branches[0].experiment.handler_id)

    def _current_evidence(self, investigation_id="handler"):
        return Investigation(
            investigation_id, "openstar.workflow.tess-investigation.v1", "20.2",
            "RUNNING", "now", "now", {"datasetID": "generic-dataset"}, (
                InvestigationStage("001-prepare", "openstar.tess.prepare-target", "COMPLETE",
                    None, {}, result={"sourceProjectID": "generic-project",
                                      "datasetID": "generic-dataset"}),
                InvestigationStage("043-gaia",
                    "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                    "COMPLETE", None, {}, result=self.gaia),
                InvestigationStage("045-sky",
                    "openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
                    "COMPLETE", None, {}, result=self.sky),))

    def _run_registered_handler_chain(self, distributed):
        store = InvestigationStore(self.root / ("handlers-run" if distributed else "handlers-direct"))
        investigation = self._current_evidence("handlers")
        store.save(investigation)
        project_path = self.root / "nsc-project.json"
        if distributed:
            project_path.write_text("{}", encoding="utf-8")
        spec = {"available": distributed,
                "projectPath": str(project_path) if distributed else None,
                "preparedSeries": [], "workloadID": "openstar.lomb-scargle.v1"}
        coordinator = mock.Mock()
        coordinator.run_project.return_value = SimpleNamespace(
            status={"datasets": []}, node_contributions={}, project_id="nsc-project")
        summary = {"recommendedNextTest": nsc.NEXT_ARCHIVE_TEST,
                   "physicalMechanismResolved": False}
        with mock.patch("workflows.tess.tess_investigation.build_nsc_resolved_project",
                        return_value=spec) as prepare, mock.patch(
             "workflows.tess.tess_investigation.interpret_nsc_resolved_project",
             return_value=summary) as interpret:
            completed = build_engine(store, coordinator, poll_interval=0, timeout=1).run(
                investigation, StageRequest(
                    "046-prepare-nsc-resolved-counterpart-photometry",
                    "openstar.tess.nsc-resolved-photometry.prepare", {}, "045-sky"),
                software_id="test", software_version="1")
        expected = ["openstar.tess.nsc-resolved-photometry.prepare"]
        if distributed:
            expected.append("openstar.tess.nsc-resolved-photometry.run")
        expected.append("openstar.tess.nsc-resolved-photometry.interpret")
        self.assertEqual(expected, [stage.handler_id for stage in completed.stages[3:]])
        prepare.assert_called_once()
        interpret.assert_called_once()
        self.assertEqual(1 if distributed else 0, coordinator.run_project.call_count)
        self.assertEqual("BLOCKED", completed.status)

    def test_registered_handlers_prepare_run_interpret(self):
        self._run_registered_handler_chain(True)

    def test_registered_handlers_prepare_interpret_without_project(self):
        self._run_registered_handler_chain(False)

    def _run_persisted_prepare_lifecycle(self, distributed):
        store = InvestigationStore(self.root / ("restart-run" if distributed else "restart-direct"))
        target = InvestigationTarget(
            "generic", "restart", "openstar.workflow.tess-investigation.v1", "20.2",
            metadata={"datasetID": "generic-dataset"})
        next_request = StageRequest(
            "047-run-nsc-resolved-counterpart-photometry" if distributed
            else "047-interpret-nsc-resolved-counterpart-photometry",
            "openstar.tess.nsc-resolved-photometry.run" if distributed
            else "openstar.tess.nsc-resolved-photometry.interpret",
            {"projectPath": "/persisted/nsc.json"} if distributed
            else {"distributedRunExpected": False}, "046-prepare-nsc")
        prepare = InvestigationStage(
            "046-prepare-nsc", "openstar.tess.nsc-resolved-photometry.prepare",
            "COMPLETE", "045-sky", {}, result={"available": distributed},
            next_stage=asdict(next_request))
        investigation = Investigation(
            target.investigation_id, target.workflow_id, target.workflow_version,
            "COMPLETE", "now", "now", target.metadata, (prepare,))
        store.save(investigation)
        investigation, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target))
        executions = []
        workflow = WorkflowEngine(store)
        workflow.register_handler("openstar.tess.nsc-resolved-photometry.prepare",
            lambda *_: self.fail("persisted NSC preparation must not be rerun"))
        if distributed:
            def run_stage(current, request):
                executions.append(request)
                return StageOutcome(result={"datasets": []}, next_stage=StageRequest(
                    "048-interpret-nsc-resolved-counterpart-photometry",
                    "openstar.tess.nsc-resolved-photometry.interpret",
                    {"distributedRunExpected": True}, request.id))
            workflow.register_handler(next_request.handler_id, run_stage)
        def interpret_stage(current, request):
            executions.append(request)
            return StageOutcome(result={"recommendedNextTest": nsc.NEXT_ARCHIVE_TEST,
                                        "physicalMechanismResolved": False},
                                stop=True, final_status="BLOCKED")
        workflow.register_handler("openstar.tess.nsc-resolved-photometry.interpret", interpret_stage)
        workflow.register_handler(
            "openstar.tess.noirlab-image-forced-photometry.prepare",
            lambda current, request: StageOutcome(
                result={"available": False}, next_stage=StageRequest(
                    "050-interpret-noirlab-image-forced-photometry",
                    "openstar.tess.noirlab-image-forced-photometry.interpret",
                    {"distributedRunExpected": False}, request.id)),
        )
        workflow.register_handler(
            "openstar.tess.noirlab-image-forced-photometry.interpret",
            lambda current, request: StageOutcome(result={
                "recommendedNextTest": "DES_DR2_SINGLE_EPOCH_LOCAL_FORCED_PHOTOMETRY",
                "physicalMechanismResolved": False}, stop=True, final_status="BLOCKED"))
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(
            self.root / "nsc-lifecycle.json", store, dispatcher,
            InvestigationTargetPortfolio(self.root / "nsc-portfolio.json", store, dispatcher),
            _SingleTargetSource(target), {target.workflow_id: plan_tess_branches},
            software_id="test", software_version="1")
        lifecycle.start(target)
        result = lifecycle.run(max_transitions=20)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual(next_request, executions[0])
        self.assertEqual(2 if distributed else 1, len(executions))
        persisted = store.load(investigation.id)
        self.assertEqual(prepare, persisted.stages[0])
        self.assertEqual("BLOCKED", persisted.status)
        self.assertEqual("WAIT_FOR_PREREQUISITES",
                         persisted.metadata["controlState"]["schedulerAction"])

    def test_persisted_prepare_run_interpret_without_reprepare_or_loop(self):
        self._run_persisted_prepare_lifecycle(True)

    def test_persisted_prepare_direct_interpret_without_reprepare(self):
        self._run_persisted_prepare_lifecycle(False)

    def test_exact_persisted_nsc_next_stage_is_reused(self):
        expected = StageRequest("099-exact", "openstar.tess.nsc-resolved-photometry.run",
                                {"projectPath": "/exact.json"}, "046-prepare")
        investigation = Investigation("i", "openstar.workflow.tess-investigation.v1", "20.2",
            "COMPLETE", "now", "now", {}, (InvestigationStage(
                "046-prepare", "openstar.tess.nsc-resolved-photometry.prepare", "COMPLETE",
                None, {}, result={}, next_stage=asdict(expected)),))
        target = InvestigationTarget("t", "i", investigation.workflow_id,
                                     investigation.workflow_version)
        self.assertEqual(expected, plan_tess_branches(investigation, target)[0].experiment)

    def test_transient_failed_nsc_run_uses_generic_retry(self):
        store = InvestigationStore(self.root / "retry")
        target = InvestigationTarget("t", "retry", "openstar.workflow.tess-investigation.v1", "20.2")
        investigation = store.create(target.investigation_id, target.workflow_id, target.workflow_version)
        failed = InvestigationStage("047-run-nsc", "openstar.tess.nsc-resolved-photometry.run",
                                    "RUNNING", "046-prepare", {})
        investigation = store.append_running_stage(investigation, failed)
        terminal = store.build_terminal_stage(
            stage_id=failed.id, handler_id=failed.handler_id, status="FAILED",
            triggered_by_stage_id=failed.triggered_by_stage_id, parameters={}, result=None,
            error="RetryableExecutionError: outage", failure_classification="TRANSIENT_INFRASTRUCTURE",
            software_id="test", software_version="1", started_at=failed.started_at)
        store.complete_current_stage(investigation, terminal)
        attempts = []
        workflow = WorkflowEngine(store)
        workflow.register_handler(failed.handler_id,
            lambda current, request: (attempts.append(request.id) or StageOutcome({}, stop=True)))
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(
            self.root / "retry.json", store, dispatcher,
            InvestigationTargetPortfolio(self.root / "retry-portfolio.json", store, dispatcher),
            _SingleTargetSource(target), {target.workflow_id: lambda *_: ()},
            software_id="test", software_version="1")
        lifecycle.start(target)
        result = lifecycle.run(max_transitions=10)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual(["048-run-nsc"], attempts)

    def test_nonretryable_failed_nsc_is_not_freshly_replanned(self):
        stages = (InvestigationStage("045-sky",
            "openstar.tess.skymapper-resolved-counterpart-photometry.interpret", "COMPLETE",
            None, {}, result=self.sky), InvestigationStage(
            "046-prepare", "openstar.tess.nsc-resolved-photometry.prepare", "FAILED",
            "045-sky", {}, error="ValueError: bug", failure_classification="NON_RETRYABLE"))
        investigation = Investigation("i", "openstar.workflow.tess-investigation.v1", "20.2",
                                      "FAILED", "now", "now", {}, stages)
        target = InvestigationTarget("t", "i", investigation.workflow_id,
                                     investigation.workflow_version)
        self.assertFalse(any(
            branch.experiment.handler_id.startswith("openstar.tess.nsc-resolved-photometry.")
            for branch in plan_tess_branches(investigation, target)))


if __name__ == "__main__":
    unittest.main()
