import tempfile
import sys
import types
from types import SimpleNamespace
import unittest
import urllib.error
from dataclasses import asdict
from pathlib import Path
from unittest import mock

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

from workflows.tess import tess_skymapper_resolved as sky
from openstar_investigation import Investigation, InvestigationStage, InvestigationStore
from openstar_autonomy import AutonomousInvestigationEngine
from openstar_dispatch import InvestigationDispatcher
from openstar_lifecycle import InvestigationLifecycleLoop
from openstar_targets import InvestigationTarget, InvestigationTargetPortfolio
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine
from workflows.tess.tess_autonomy import plan_tess_branches, repair_obsolete_terminal_wait
from workflows.tess.tess_investigation import build_engine

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


class _SingleTargetSource:
    id = "skymapper-test-targets"
    version = "1"

    def __init__(self, target):
        self.target = target

    def enumerate_targets(self):
        return (self.target,)


class CurrentSkyMapperResolvedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target_id = 700000000000000011
        self.counterpart_id = 700000000000000099
        self.gaia = {
            "recommendedNextTest": "SKYMAPPER_RESOLVED_COUNTERPART_PHOTOMETRY",
            "physicalMechanismResolved": False,
            "sourcePair": {
                "version": "openstar.current-source-pair.v1",
                "target": {"sourceRole": "target-control", "gaiaDR3SourceID": self.target_id,
                           "raDeg": 12.3, "decDeg": -45.6},
                "counterpart": {"sourceRole": "catalog-counterpart",
                                "gaiaDR3SourceID": self.counterpart_id,
                                "raDeg": 12.3005, "decDeg": -45.6},
                "separationArcsec": 1.26,
            },
            "frequencySearch": {"minimumFrequency": .2, "maximumFrequency": .3,
                                "frequencyStep": .001, "totalFrequencies": 100,
                                "frequenciesPerWorkUnit": 25},
        }

    def build(self):
        return sky.build_skymapper_resolved_project(
            source_project_id="generic-project", source_dataset_id="generic-dataset",
            external_high_resolution_summary=self.gaia, output_dir=self.root,
            investigation_id="generic-investigation")

    def test_current_source_pair_adapter_uses_persisted_generic_ids(self):
        definitions, separation = sky._frozen_source_pair(self.gaia)
        self.assertEqual([self.target_id, self.counterpart_id],
                         [item["gaiaDR3SourceID"] for item in definitions])
        self.assertEqual(1.26, separation)

    @mock.patch.object(sky, "_query_master_matches", return_value=[])
    def test_no_coverage_is_complete_and_routes_to_nsc(self, query):
        preparation = self.build()
        result = sky.interpret_skymapper_resolved_project(
            project_status=None, preparation=preparation)
        self.assertFalse(preparation["available"])
        self.assertEqual("NSC_RESOLVED_COUNTERPART_PHOTOMETRY",
                         result["recommendedNextTest"])
        self.assertFalse(result["physicalMechanismResolved"])
        query.assert_called_once_with([self.target_id, self.counterpart_id])

    def test_insufficient_cadence_and_quality_cuts_are_scientific(self):
        result = sky.interpret_skymapper_resolved_project(project_status=None, preparation={
            "pairSeparatelyResolvedInSkyMapperMaster": True, "preparedSeries": [],
            "sourcePair": self.gaia["sourcePair"]})
        self.assertEqual("NSC_RESOLVED_COUNTERPART_PHOTOMETRY",
                         result["recommendedNextTest"])

    def test_usable_but_nonrecurrent_data_routes_to_nsc(self):
        prep = {"pairSeparatelyResolvedInSkyMapperMaster": True,
                "preparedSeries": [{"datasetID": "d", "sourceRole": "catalog-counterpart",
                                    "band": "g"}], "workloadID": sky.GENERIC_LOMB_SCARGLE_WORKLOAD_ID}
        status = {"datasets": [{"datasetID": "d", "periodStatus": "UNRELIABLE"}]}
        result = sky.interpret_skymapper_resolved_project(project_status=status, preparation=prep)
        self.assertEqual("NSC_RESOLVED_COUNTERPART_PHOTOMETRY",
                         result["recommendedNextTest"])

    def test_incomplete_source_pair_coverage_routes_to_nsc(self):
        result = sky.interpret_skymapper_resolved_project(project_status=None, preparation={
            "pairSeparatelyResolvedInSkyMapperMaster": False, "preparedSeries": []})
        self.assertEqual("NSC_RESOLVED_COUNTERPART_PHOTOMETRY",
                         result["recommendedNextTest"])

    @mock.patch.object(sky, "_query_master_matches",
                       side_effect=urllib.error.URLError("temporary outage"))
    def test_transient_service_failure_is_retryable(self, _query):
        with self.assertRaises(sky.SkyMapperArchiveUnavailable):
            self.build()

    @mock.patch.object(sky, "_query_master_matches", side_effect=ValueError("bug"))
    def test_programming_error_is_not_retryable(self, _query):
        with self.assertRaises(ValueError):
            self.build()

    def test_terminal_gaia_schedules_044_without_rerunning_gaia(self):
        gaia_stage = InvestigationStage(
            id="043-interpret-gaia-source-resolved-counterpart-photometry",
            handler_id="openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
            status="COMPLETE", triggered_by_stage_id="042", parameters={},
            result=self.gaia, stop=True)
        investigation = Investigation(
            id="generic-investigation", workflow_id="openstar.workflow.tess-investigation.v1",
            workflow_version="20.2", status="COMPLETE", created_at="now", updated_at="now",
            metadata={"datasetID": "generic-dataset"}, stages=(gaia_stage,))
        target = InvestigationTarget(id="generic", investigation_id=investigation.id,
                                     workflow_id=investigation.workflow_id,
                                     workflow_version=investigation.workflow_version)
        branches = plan_tess_branches(investigation, target)
        self.assertEqual("044-prepare-skymapper-resolved-counterpart-photometry",
                         branches[0].experiment.id)
        self.assertNotIn("gaia", branches[0].experiment.handler_id)

    def test_completed_skymapper_interpretation_is_idempotent(self):
        stages = (InvestigationStage(
            id="043-gaia", handler_id="openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
            status="COMPLETE", triggered_by_stage_id=None, parameters={}, result=self.gaia),
            InvestigationStage(
            id="046-skymapper", handler_id="openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
            status="COMPLETE", triggered_by_stage_id=None, parameters={}, result={}))
        investigation = Investigation(
            id="generic-investigation", workflow_id="openstar.workflow.tess-investigation.v1",
            workflow_version="20.2", status="COMPLETE", created_at="now", updated_at="now",
            metadata={}, stages=stages)
        target = InvestigationTarget(id="generic", investigation_id=investigation.id,
                                     workflow_id=investigation.workflow_id,
                                     workflow_version=investigation.workflow_version)
        self.assertEqual((), plan_tess_branches(investigation, target))

    def _run_real_handler_chain(self, distributed):
        store = InvestigationStore(self.root / f"handlers-{distributed}")
        investigation = store.create("handler-chain", "openstar.workflow.tess-investigation.v1", "20.2")
        evidence = (
            ("001-target", "openstar.tess.prepare-target",
             {"sourceProjectID": "generic-project", "datasetID": "generic-dataset"}),
            ("043-gaia", "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
             self.gaia),
        )
        for stage_id, handler_id, result in evidence:
            running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
            investigation = store.append_running_stage(investigation, running)
            completed = store.build_terminal_stage(
                stage_id=stage_id, handler_id=handler_id, status="COMPLETE",
                triggered_by_stage_id=None, parameters={}, result=result, error=None,
                software_id="test", software_version="1", started_at=running.started_at)
            investigation = store.complete_current_stage(investigation, completed)
        project_path = self.root / "project.json"
        project_path.write_text("{}", encoding="utf-8")
        spec = {"available": distributed, "projectPath": str(project_path) if distributed else None,
                "preparedSeries": [], "workloadID": "openstar.lomb-scargle.v1"}
        coordinator = mock.Mock()
        coordinator.run_project.return_value = SimpleNamespace(
            status={"datasets": []}, node_contributions={}, project_id="sky-project")
        summary = {"recommendedNextTest": "NSC_RESOLVED_COUNTERPART_PHOTOMETRY",
                   "physicalMechanismResolved": False}
        with mock.patch("workflows.tess.tess_investigation.build_skymapper_resolved_project",
                        return_value=spec) as prepare, mock.patch(
             "workflows.tess.tess_investigation.interpret_skymapper_resolved_project",
             return_value=summary) as interpret:
            engine = build_engine(store, coordinator, poll_interval=0, timeout=1)
            completed = engine.run(
                investigation,
                StageRequest("044-prepare-skymapper-resolved-counterpart-photometry",
                    "openstar.tess.skymapper-resolved-counterpart-photometry.prepare", {}, "043-gaia"),
                software_id="test", software_version="1")
        handlers = [item.handler_id for item in completed.stages[2:]]
        expected = ["openstar.tess.skymapper-resolved-counterpart-photometry.prepare"]
        if distributed:
            expected.append("openstar.tess.skymapper-resolved-counterpart-photometry.run")
        expected.append("openstar.tess.skymapper-resolved-counterpart-photometry.interpret")
        self.assertEqual(expected, handlers)
        prepare.assert_called_once()
        interpret.assert_called_once()
        self.assertEqual(1 if distributed else 0, coordinator.run_project.call_count)
        self.assertEqual("BLOCKED", completed.status)

    def test_handlers_execute_prepare_run_interpret(self):
        self._run_real_handler_chain(True)

    def test_handlers_execute_prepare_then_interpret_without_project(self):
        self._run_real_handler_chain(False)

    def _run_persisted_prepare_lifecycle(self, distributed):
        suffix = "run" if distributed else "direct"
        store = InvestigationStore(self.root / f"prepared-{suffix}")
        target = InvestigationTarget(
            "generic", f"prepared-{suffix}", "openstar.workflow.tess-investigation.v1", "20.2",
            metadata={"datasetID": "generic-dataset"})
        investigation = store.create(target.investigation_id, target.workflow_id,
                                     target.workflow_version, metadata=target.metadata)
        next_request = StageRequest(
            "045-run-skymapper-resolved-counterpart-photometry" if distributed
            else "045-interpret-skymapper-resolved-counterpart-photometry",
            "openstar.tess.skymapper-resolved-counterpart-photometry.run" if distributed
            else "openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
            {"projectPath": "/persisted/project.json"} if distributed
            else {"distributedRunExpected": False},
            "044-prepare-skymapper-resolved-counterpart-photometry")
        evidence = (
            ("043-interpret-gaia-source-resolved-counterpart-photometry",
             "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret", self.gaia, None),
            ("044-prepare-skymapper-resolved-counterpart-photometry",
             "openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
             {"available": distributed}, next_request),
        )
        for stage_id, handler_id, result, continuation in evidence:
            running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
            investigation = store.append_running_stage(investigation, running)
            completed = store.build_terminal_stage(
                stage_id=stage_id, handler_id=handler_id, status="COMPLETE",
                triggered_by_stage_id=None, parameters={}, result=result, error=None,
                software_id="old", software_version="1", started_at=running.started_at,
                next_stage=asdict(continuation) if continuation else None)
            investigation = store.complete_current_stage(investigation, completed)
        investigation, _ = AutonomousInvestigationEngine(store).decide(investigation, ())
        persisted_prepare = investigation.stages[1]

        executions = []
        workflow = WorkflowEngine(store)
        workflow.register_handler(
            "openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
            lambda *_: self.fail("persisted SkyMapper preparation must not be rerun"))
        if distributed:
            def run_stage(current, request):
                executions.append(request)
                return StageOutcome(result={"datasets": []}, next_stage=StageRequest(
                    "046-interpret-skymapper-resolved-counterpart-photometry",
                    "openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
                    {"distributedRunExpected": True}, request.id))
            workflow.register_handler(next_request.handler_id, run_stage)
        def interpret_stage(current, request):
            executions.append(request)
            return StageOutcome(result={"recommendedNextTest": "NSC_RESOLVED_COUNTERPART_PHOTOMETRY",
                                        "physicalMechanismResolved": False}, stop=True,
                                final_status="BLOCKED")
        workflow.register_handler(
            "openstar.tess.skymapper-resolved-counterpart-photometry.interpret", interpret_stage)
        workflow.register_handler(
            "openstar.tess.nsc-resolved-photometry.prepare",
            lambda current, request: StageOutcome(result={}, stop=True,
                                                  final_status="BLOCKED"))
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(
            self.root / f"lifecycle-{suffix}.json", store, dispatcher,
            InvestigationTargetPortfolio(self.root / f"portfolio-{suffix}.json", store, dispatcher),
            _SingleTargetSource(target), {target.workflow_id: plan_tess_branches},
            software_id="test", software_version="1")
        lifecycle.start(target)
        repaired = repair_obsolete_terminal_wait(store, store.load(investigation.id))
        self.assertEqual("RUNNING", repaired.status)
        result = lifecycle.run(max_transitions=20)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual(next_request, executions[0])
        self.assertEqual(persisted_prepare, store.load(investigation.id).stages[1])
        self.assertEqual(2 if distributed else 1, len(executions))
        self.assertEqual("INVESTIGATION_COMPLETE",
                         store.load(investigation.id).metadata["controlState"]["schedulerAction"])

    def test_prepare_run_interpret_restarts_without_repreparing_or_looping(self):
        self._run_persisted_prepare_lifecycle(True)

    def test_prepare_interpret_without_project_restarts_without_repreparing(self):
        self._run_persisted_prepare_lifecycle(False)

    def test_exact_persisted_skymapper_next_stage_is_reused(self):
        expected = StageRequest("099-custom", "openstar.tess.skymapper-resolved-counterpart-photometry.run",
                                {"projectPath": "/exact.json"}, "044-prepare")
        stages = (
            InvestigationStage("043-gaia",
                "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                "COMPLETE", None, {}, result=self.gaia),
            InvestigationStage("044-prepare",
                "openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
                "COMPLETE", "043-gaia", {}, result={}, next_stage=asdict(expected)),)
        investigation = Investigation("i", "openstar.workflow.tess-investigation.v1", "20.2",
                                      "COMPLETE", "now", "now", {}, stages)
        target = InvestigationTarget("t", "i", investigation.workflow_id,
                                     investigation.workflow_version)
        self.assertEqual(expected, plan_tess_branches(investigation, target)[0].experiment)

    def test_transient_failed_skymapper_run_uses_generic_retry(self):
        store = InvestigationStore(self.root / "retry")
        target = InvestigationTarget("t", "retry", "openstar.workflow.tess-investigation.v1", "20.2")
        investigation = store.create(target.investigation_id, target.workflow_id, target.workflow_version)
        failed = InvestigationStage("045-run-skymapper", "openstar.tess.skymapper-resolved-counterpart-photometry.run",
                                    "RUNNING", "044-prepare", {})
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
        self.assertEqual(["046-run-skymapper"], attempts)

    def test_nonretryable_failed_skymapper_stage_is_not_freshly_replanned(self):
        expected = StageRequest("045-run", "openstar.tess.skymapper-resolved-counterpart-photometry.run",
                                {}, "044-prepare")
        stages = (
            InvestigationStage("043-gaia",
                "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                "COMPLETE", None, {}, result=self.gaia),
            InvestigationStage("044-prepare",
                "openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
                "COMPLETE", "043-gaia", {}, result={}, next_stage=asdict(expected)),
            InvestigationStage("045-run", expected.handler_id, "FAILED", "044-prepare", {},
                               error="ValueError: bug", failure_classification="NON_RETRYABLE"),)
        investigation = Investigation("i", "openstar.workflow.tess-investigation.v1", "20.2",
                                      "FAILED", "now", "now", {}, stages)
        target = InvestigationTarget("t", "i", investigation.workflow_id,
                                     investigation.workflow_version)
        self.assertEqual((), plan_tess_branches(investigation, target))

    @mock.patch.object(sky, "_query_master_matches", return_value=[])
    def test_only_generic_lomb_scargle_workload_is_declared(self, _query):
        self.assertEqual("openstar.lomb-scargle.v1", self.build()["workloadID"])


if __name__ == "__main__":
    unittest.main()
