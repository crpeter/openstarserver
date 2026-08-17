import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_autonomy import AutonomousInvestigationEngine
from openstar_dispatch import InvestigationDispatcher
from openstar_lifecycle import InvestigationLifecycleLoop
from openstar_targets import (
    InvestigationTarget,
    InvestigationTargetPortfolio,
)
from openstar_workflow import RetryableExecutionError, StageOutcome, StageRequest, WorkflowEngine
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_autonomy import plan_tess_branches, repair_obsolete_terminal_wait
from workflows.tess import tess_gaia_counterpart as gaia


class _Array(list):
    def tolist(self):
        return list(self)


class _NumpyStub:
    float32 = object()

    @staticmethod
    def asarray(values, dtype=None):
        return _Array(values)


class _SingleTargetSource:
    id = "gaia-test-targets"
    version = "1"

    def __init__(self, target):
        self.target = target

    def enumerate_targets(self):
        return (self.target,)


class CurrentGaiaCounterpartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target_id = 101
        self.target_gaia = 900000000000000001
        self.counterpart_tic = 202
        self.counterpart_gaia = 900000000000000002
        self.prepared = {"ticID": self.target_id, "sourceProjectID": "project-x",
                         "datasetID": "dataset-x"}
        self.identity = {
            "tic": {"metadata": {"raDeg": 10.0, "decDeg": -20.0}},
            "gaiaDR3": {"nearest": {"sourceID": self.target_gaia,
                                     "raDeg": 10.0, "decDeg": -20.0}},
        }
        candidate = {"raDeg": 10.01, "decDeg": -20.0,
                     "catalogIDs": {"ticID": self.counterpart_tic,
                                    "gaiaDR3SourceID": self.counterpart_gaia}}
        self.catalog = {"preferredCandidate": candidate}
        self.variability = {
            "classification": "OFFSET_COUNTERPART_VARIABILITY_SUGGESTIVE",
            "physicalMechanismResolved": False,
            "recommendedNextTest": "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY",
            "catalogCounterpart": {"ticID": self.counterpart_tic,
                                   "gaiaDR3SourceID": self.counterpart_gaia,
                                   "raDeg": 10.01, "decDeg": -20.0},
            "catalogCounterpartEvidence": {"combinedPeriodDays": 4.5,
                                            "combinedFrequency": 1 / 4.5},
            "counterpartPerSectorResults": [
                {"candidateFrequency": 0.221}, {"candidateFrequency": 0.224}],
            "distributedValidation": {"frequencySearch": {"frequencyStep": 0.00001}},
            "targetControl": {"combinedPeriodDays": 4.4},
        }

    def metadata(self, ids=None, *, epochs=True):
        return {
            self.target_gaia: {"sourceID": self.target_gaia, "raDeg": 10.0,
                               "decDeg": -20.0, "hasEpochPhotometry": epochs},
            self.counterpart_gaia: {"sourceID": self.counterpart_gaia, "raDeg": 10.01,
                                    "decDeg": -20.0, "hasEpochPhotometry": epochs},
        }

    def build(self, **overrides):
        arguments = dict(
            source_project_id="project-x", source_dataset_id="dataset-x",
            prepared_target=self.prepared, identity=self.identity,
            catalog_identification=self.catalog, offset_variability=self.variability,
            output_dir=self.root, investigation_id="generic-investigation",
            query_metadata=lambda ids: self.metadata(ids),
            download_epochs=lambda source_id: (b"epoch data", "text/csv"),
        )
        arguments.update(overrides)
        with mock.patch.object(gaia, "np", _NumpyStub), mock.patch.object(
            gaia, "_parse_gaia_g_series",
            return_value=([1.0, 2.0, 3.0], [0.0, 1.0, 0.0],
                          {"baselineDays": 2.0, "sampleCount": 3}),
        ):
            return gaia.build_current_gaia_counterpart_project(**arguments)

    def test_current_variability_result_selects_deterministic_gaia_continuation(self):
        store = InvestigationStore(self.root / "store")
        investigation = store.create("generic", "openstar.workflow.tess-investigation.v1", "20.2")
        investigation = type(investigation)(
            **{**investigation.__dict__, "stages": (
                InvestigationStage("041-interpret-offset-source-variability",
                    "openstar.tess.offset-source-variability.interpret", "COMPLETE", None, {},
                    result=self.variability),
            )}
        )
        branches = plan_tess_branches(
            investigation, InvestigationTarget("t", "generic",
                "openstar.workflow.tess-investigation.v1", "20.2")
        )
        self.assertEqual(1, len(branches))
        self.assertEqual("042-prepare-gaia-source-resolved-counterpart-photometry",
                         branches[0].experiment.id)
        self.assertEqual("openstar.tess.gaia-source-resolved-counterpart-photometry.prepare",
                         branches[0].experiment.handler_id)

    def test_source_pair_comes_only_from_persisted_generic_evidence(self):
        spec = self.build()
        pair = spec["sourcePair"]
        self.assertEqual(self.target_id, pair["target"]["ticID"])
        self.assertEqual(self.target_gaia, pair["target"]["gaiaDR3SourceID"])
        self.assertEqual(self.counterpart_tic, pair["counterpart"]["ticID"])
        self.assertEqual(self.counterpart_gaia, pair["counterpart"]["gaiaDR3SourceID"])
        source = Path(gaia.__file__).read_text()
        self.assertNotIn("Blind C", source)
        self.assertNotIn("736900598", source)
        self.assertNotIn("5284296077579591040", source)

    def test_both_gaia_ids_are_coordinate_validated(self):
        bad = self.metadata()
        bad[self.counterpart_gaia] = {**bad[self.counterpart_gaia], "raDeg": 11.0}
        with self.assertRaisesRegex(RuntimeError, "catalog-counterpart"):
            self.build(query_metadata=lambda ids: bad)

    def test_usable_series_emit_only_generic_lomb_scargle_work(self):
        spec = self.build()
        self.assertEqual("openstar.lomb-scargle.v1", spec["workloadID"])
        manifest = json.loads(Path(spec["projectPath"]).read_text())
        self.assertEqual("openstar.lomb-scargle.v1", manifest["workloadID"])
        self.assertEqual(2, len(manifest["datasets"]))
        for entry in manifest["datasets"]:
            dataset = json.loads(Path(entry["path"]).read_text())
            self.assertEqual({"id", "targetName", "times", "flux", "frequencySearch",
                              "reference", "science", "source"}, set(dataset))
            self.assertFalse(dataset["science"]["tessDriftExtrapolated"])

    @staticmethod
    def status(spec, target=False, counterpart=False):
        rows = []
        for item in spec["preparedSeries"]:
            accepted = target if item["sourceRole"] == "target-control" else counterpart
            rows.append({"datasetID": item["datasetID"], "periodStatus": "RELIABLE",
                         "coverageComplete": True, "candidateFrequency": spec["referenceFrequency"],
                         "candidatePeriodDays": spec["referencePeriodDays"], "candidatePower": 0.2,
                         "candidatePeakProminenceRatio": 2.0 if accepted else 1.0})
        return {"datasets": rows}

    def test_counterpart_and_target_control_outcomes_remain_distinct(self):
        spec = self.build()
        counterpart = gaia.interpret_current_gaia_counterpart_project(
            project_status=self.status(spec, counterpart=True), preparation=spec)
        target = gaia.interpret_current_gaia_counterpart_project(
            project_status=self.status(spec, target=True), preparation=spec)
        both = gaia.interpret_current_gaia_counterpart_project(
            project_status=self.status(spec, target=True, counterpart=True), preparation=spec)
        self.assertEqual("COUNTERPART_RECURRENCE_SUPPORTED", counterpart["classification"])
        self.assertEqual("TARGET_CONTROL_RECURRENCE_ONLY", target["classification"])
        self.assertEqual("BOTH_SOURCES_SHOW_RECURRENCE", both["classification"])
        self.assertFalse(counterpart["physicalMechanismResolved"])

    def test_no_epoch_product_is_complete_and_routes_to_skymapper(self):
        spec = self.build(query_metadata=lambda ids: self.metadata(ids, epochs=False))
        result = gaia.interpret_current_gaia_counterpart_project(
            project_status=None, preparation=spec)
        self.assertEqual("GAIA_NO_EPOCH_PHOTOMETRY", result["classification"])
        self.assertEqual("AVAILABLE", result["externalDataState"])
        self.assertEqual(gaia.NEXT_ARCHIVE_TEST, result["recommendedNextTest"])

    def test_insufficient_epochs_are_complete_and_route_to_skymapper(self):
        with mock.patch.object(
            gaia, "_parse_gaia_g_series",
            side_effect=RuntimeError("Only 3 usable Gaia G-band epoch samples; need at least 20."),
        ):
            spec = gaia.build_current_gaia_counterpart_project(
                source_project_id="project-x", source_dataset_id="dataset-x",
                prepared_target=self.prepared, identity=self.identity,
                catalog_identification=self.catalog, offset_variability=self.variability,
                output_dir=self.root, investigation_id="generic", query_metadata=lambda ids: self.metadata(ids),
                download_epochs=lambda source_id: (b"short", "text/csv"))
        result = gaia.interpret_current_gaia_counterpart_project(project_status=None, preparation=spec)
        self.assertEqual("GAIA_INSUFFICIENT_EPOCH_PHOTOMETRY", result["classification"])
        self.assertEqual(gaia.NEXT_ARCHIVE_TEST, result["recommendedNextTest"])

    def test_usable_no_recurrence_is_complete_and_routes_to_skymapper(self):
        spec = self.build()
        result = gaia.interpret_current_gaia_counterpart_project(
            project_status=self.status(spec), preparation=spec)
        self.assertEqual("GAIA_USABLE_NO_RECURRENCE", result["classification"])
        self.assertTrue(result["archiveExhausted"])
        self.assertEqual(gaia.NEXT_ARCHIVE_TEST, result["recommendedNextTest"])

    def test_transient_metadata_and_download_outages_are_retryable_archive_failures(self):
        def unavailable(ids):
            raise TimeoutError("temporary")
        with self.assertRaises(gaia.GaiaArchiveUnavailable):
            self.build(query_metadata=unavailable)
        with self.assertRaises(gaia.GaiaArchiveUnavailable):
            self.build(download_epochs=lambda source_id: (_ for _ in ()).throw(TimeoutError("temporary")))
        programming_error = RuntimeError("missing local dependency")
        with self.assertRaises(RuntimeError) as raised:
            self.build(query_metadata=lambda ids: (_ for _ in ()).throw(programming_error))
        self.assertIs(programming_error, raised.exception)

    def test_partial_control_recurrence_is_not_mislabeled_no_recurrence(self):
        metadata = self.metadata()
        metadata[self.target_gaia]["hasEpochPhotometry"] = False
        spec = self.build(query_metadata=lambda ids: metadata)
        result = gaia.interpret_current_gaia_counterpart_project(
            project_status=self.status(spec, counterpart=True), preparation=spec)
        self.assertEqual("COUNTERPART_RECURRENCE_CONTROL_UNAVAILABLE", result["classification"])
        self.assertEqual("NO_EPOCH_PHOTOMETRY", result["sourceEpochStates"]["target-control"])
        self.assertTrue(result["catalogCounterpartEvidence"]["acceptedResidualBandVariability"])

        metadata = self.metadata()
        metadata[self.counterpart_gaia]["hasEpochPhotometry"] = False
        spec = self.build(query_metadata=lambda ids: metadata)
        result = gaia.interpret_current_gaia_counterpart_project(
            project_status=self.status(spec, target=True), preparation=spec)
        self.assertEqual(
            "TARGET_CONTROL_RECURRENCE_COUNTERPART_UNAVAILABLE", result["classification"]
        )
        self.assertEqual(
            "NO_EPOCH_PHOTOMETRY", result["sourceEpochStates"]["catalog-counterpart"]
        )

    def test_current_handler_records_transient_gaia_outage_for_generic_retry(self):
        store = InvestigationStore(self.root / "retry-store")
        investigation = store.create("retry-generic", "openstar.workflow.tess-investigation.v1", "20.2")
        evidence = (
            ("001-prepare", "openstar.tess.prepare-target", self.prepared),
            ("002-identity", "openstar.tess.catalog-identity", self.identity),
            ("003-catalog", "openstar.tess.catalog-counterpart-identification.analyze", self.catalog),
            ("004-variability", "openstar.tess.offset-source-variability.interpret", self.variability),
        )
        for stage_id, handler_id, result in evidence:
            investigation = store.load(investigation.id)
            running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
            investigation = store.append_running_stage(investigation, running)
            completed = store.build_terminal_stage(
                stage_id=stage_id, handler_id=handler_id, status="COMPLETE",
                triggered_by_stage_id=None, parameters={}, result=result, error=None,
                software_id="test", software_version="1",
            )
            investigation = store.complete_current_stage(investigation, completed)
        engine = build_engine(store, mock.Mock(), poll_interval=0, timeout=1)
        with mock.patch(
            "workflows.tess.tess_investigation.build_current_gaia_counterpart_project",
            side_effect=gaia.GaiaArchiveUnavailable("temporary Gaia outage"),
        ), self.assertRaises(RetryableExecutionError):
            engine.run_stage(
                store.load(investigation.id),
                StageRequest("005-prepare-gaia-source-resolved-counterpart-photometry",
                             "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare", {}),
                software_id="test", software_version="1",
            )
        failed = store.load(investigation.id).stages[-1]
        self.assertEqual("FAILED", failed.status)
        self.assertEqual("TRANSIENT_INFRASTRUCTURE", failed.failure_classification)

    def test_completed_gaia_attempt_is_not_selected_again_on_restart(self):
        store = InvestigationStore(self.root / "restart-store")
        investigation = store.create("generic", "openstar.workflow.tess-investigation.v1", "20.2")
        stages = (
            InvestigationStage("041-interpret-offset-source-variability",
                "openstar.tess.offset-source-variability.interpret", "COMPLETE", None, {},
                result=self.variability),
            InvestigationStage("044-interpret-gaia-source-resolved-counterpart-photometry",
                "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret", "COMPLETE",
                "043-run", {}, result={"classification": "GAIA_USABLE_NO_RECURRENCE",
                                        "recommendedNextTest": gaia.NEXT_ARCHIVE_TEST}),
        )
        investigation = type(investigation)(**{**investigation.__dict__, "stages": stages})
        branches = plan_tess_branches(investigation, InvestigationTarget(
            "t", "generic", "openstar.workflow.tess-investigation.v1", "20.2"))
        self.assertEqual((), branches)

    def _run_terminal_prepared_regression(self, *, distributed):
        """Exercise the persisted 041/042 production failure shape end to end."""
        suffix = "run" if distributed else "direct"
        store = InvestigationStore(self.root / f"persisted-prepare-{suffix}")
        target = InvestigationTarget(
            "generic-target", f"persisted-prepare-{suffix}",
            "openstar.workflow.tess-investigation.v1", "20.2",
            metadata={"datasetID": "dataset-x"},
        )
        investigation = store.create(
            target.investigation_id, target.workflow_id, target.workflow_version,
            metadata=target.metadata,
        )
        next_request = StageRequest(
            "043-run-gaia-source-resolved-counterpart-photometry" if distributed
            else "043-interpret-gaia-source-resolved-counterpart-photometry",
            "openstar.tess.gaia-source-resolved-counterpart-photometry.run" if distributed
            else "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
            {"projectPath": "/authoritative/project.json"} if distributed
            else {"distributedRunExpected": False},
            "042-prepare-gaia-source-resolved-counterpart-photometry",
        )
        evidence = (
            ("041-interpret-offset-source-variability",
             "openstar.tess.offset-source-variability.interpret", self.variability, None),
            ("042-prepare-gaia-source-resolved-counterpart-photometry",
             "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare",
             {"available": distributed,
              **({"projectPath": "/authoritative/project.json"} if distributed else {})},
             next_request),
        )
        for stage_id, handler_id, result, continuation in evidence:
            running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
            investigation = store.append_running_stage(investigation, running)
            terminal = store.build_terminal_stage(
                stage_id=stage_id, handler_id=handler_id, status="COMPLETE",
                triggered_by_stage_id=running.triggered_by_stage_id, parameters={},
                result=result, error=None, software_id="old", software_version="1",
                started_at=running.started_at,
                next_stage=asdict(continuation) if continuation is not None else None,
            )
            investigation = store.complete_current_stage(investigation, terminal)
        investigation, _ = AutonomousInvestigationEngine(store).decide(investigation, ())
        self.assertEqual("COMPLETE", investigation.status)
        prepare_before = investigation.stages[1]

        executions = []
        workflow = WorkflowEngine(store)
        workflow.register_handler(
            "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare",
            lambda *_: self.fail("persisted Gaia preparation must not be rerun"),
        )
        if distributed:
            def run_stage(current, request):
                executions.append(request)
                return StageOutcome(
                    result={"status": "COMPLETE"},
                    next_stage=StageRequest(
                        "044-interpret-gaia-source-resolved-counterpart-photometry",
                        "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                        {"distributedRunExpected": True}, request.id,
                    ),
                )
            workflow.register_handler(next_request.handler_id, run_stage)

        def interpret_stage(current, request):
            executions.append(request)
            return StageOutcome(result={"classification": "TEST_COMPLETE"}, stop=True)
        workflow.register_handler(
            "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
            interpret_stage,
        )
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(
            self.root / f"persisted-prepare-{suffix}.json", store, dispatcher,
            InvestigationTargetPortfolio(
                self.root / f"persisted-prepare-{suffix}-portfolio.json", store, dispatcher),
            _SingleTargetSource(target), {target.workflow_id: plan_tess_branches},
            software_id="test", software_version="1",
        )
        lifecycle.start(target)
        repaired = repair_obsolete_terminal_wait(store, store.load(investigation.id))
        self.assertEqual("RUNNING", repaired.status)
        result = lifecycle.run(max_transitions=20)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual(next_request, executions[0])
        self.assertEqual(prepare_before, store.load(investigation.id).stages[1])
        self.assertEqual(2 if distributed else 1, len(executions))

    def test_terminal_persisted_prepare_runs_then_interprets_without_reprepare(self):
        self._run_terminal_prepared_regression(distributed=True)

    def test_terminal_persisted_prepare_interprets_directly_without_reprepare(self):
        self._run_terminal_prepared_regression(distributed=False)

    def test_completed_gaia_run_selects_its_exact_persisted_interpretation(self):
        expected = StageRequest(
            "044-custom-persisted-interpret",
            "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
            {"distributedRunExpected": True}, "043-custom-run",
        )
        investigation = InvestigationStore(self.root / "run-resume").create(
            "run-resume", "openstar.workflow.tess-investigation.v1", "20.2")
        investigation = type(investigation)(**{**investigation.__dict__, "stages": (
            InvestigationStage(
                "041-interpret-offset-source-variability",
                "openstar.tess.offset-source-variability.interpret", "COMPLETE",
                None, {}, result=self.variability),
            InvestigationStage(
                "043-custom-run",
                "openstar.tess.gaia-source-resolved-counterpart-photometry.run",
                "COMPLETE", "042-prepare", {}, result={"status": "COMPLETE"},
                next_stage={
                    "id": expected.id, "handler_id": expected.handler_id,
                    "parameters": expected.parameters,
                    "triggered_by_stage_id": expected.triggered_by_stage_id,
                }),
        )})
        planned = plan_tess_branches(investigation, InvestigationTarget(
            "t", investigation.id, investigation.workflow_id, investigation.workflow_version))
        self.assertEqual(expected, planned[0].experiment)

        attempted = type(investigation)(**{**investigation.__dict__, "stages": (
            *investigation.stages,
            InvestigationStage(expected.id, expected.handler_id, "FAILED", expected.triggered_by_stage_id,
                               expected.parameters, error="RuntimeError: permanent",
                               failure_classification="NON_RETRYABLE"),
        )})
        self.assertEqual((), plan_tess_branches(attempted, InvestigationTarget(
            "t", attempted.id, attempted.workflow_id, attempted.workflow_version)))

    def test_any_failed_gaia_attempt_is_not_replanned_as_fresh_science(self):
        for classification in ("TRANSIENT_INFRASTRUCTURE", "NON_RETRYABLE"):
            store = InvestigationStore(self.root / f"failed-{classification}")
            investigation = store.create(
                f"generic-{classification}", "openstar.workflow.tess-investigation.v1", "20.2"
            )
            stages = (
                InvestigationStage("041-interpret-offset-source-variability",
                    "openstar.tess.offset-source-variability.interpret", "COMPLETE", None, {},
                    result=self.variability),
                InvestigationStage("042-prepare-gaia-source-resolved-counterpart-photometry",
                    "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare", "FAILED",
                    "041-interpret-offset-source-variability", {}, error="RuntimeError: failed",
                    failure_classification=classification),
            )
            investigation = type(investigation)(**{**investigation.__dict__, "stages": stages})
            branches = plan_tess_branches(investigation, InvestigationTarget(
                "t", investigation.id, "openstar.workflow.tess-investigation.v1", "20.2"))
            self.assertEqual((), branches)

    def test_terminal_current_investigation_is_durably_repaired_and_lifecycle_runs_gaia(self):
        store = InvestigationStore(self.root / "terminal-resume-store")
        target = InvestigationTarget(
            "generic-target", "terminal-resume", "openstar.workflow.tess-investigation.v1", "20.2",
            metadata={"datasetID": "dataset-x"},
        )
        investigation = store.create(
            target.investigation_id, target.workflow_id, target.workflow_version,
            metadata=target.metadata,
        )
        running = InvestigationStage(
            "041-interpret-offset-source-variability",
            "openstar.tess.offset-source-variability.interpret", "RUNNING", None, {},
        )
        investigation = store.append_running_stage(investigation, running)
        completed = store.build_terminal_stage(
            stage_id=running.id, handler_id=running.handler_id, status="COMPLETE",
            triggered_by_stage_id=None, parameters={}, result=self.variability, error=None,
            software_id="old", software_version="1", started_at=running.started_at,
        )
        investigation = store.complete_current_stage(investigation, completed)
        investigation, _ = AutonomousInvestigationEngine(store).decide(investigation, ())
        self.assertEqual("COMPLETE", investigation.status)
        self.assertEqual(
            "INVESTIGATION_COMPLETE",
            investigation.metadata["controlState"]["schedulerAction"],
        )

        executions = []
        workflow = WorkflowEngine(store)

        def gaia_prepare(current, request):
            executions.append((request.id, request.handler_id))
            return StageOutcome(result={"archiveAttempted": True}, stop=True)

        workflow.register_handler(
            "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare", gaia_prepare
        )
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(
            self.root / "terminal-resume-lifecycle.json", store, dispatcher,
            InvestigationTargetPortfolio(self.root / "terminal-resume-portfolio.json", store, dispatcher),
            _SingleTargetSource(target), {target.workflow_id: plan_tess_branches},
            software_id="test", software_version="1",
        )
        lifecycle.start(target)

        # This is the same narrow startup compatibility hook used by the normal
        # autonomous TESS runner; no control state or prior stage is cleared.
        repaired = repair_obsolete_terminal_wait(store, store.load(target.investigation_id))
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual("RUN_EXPERIMENT", repaired.metadata["controlState"]["schedulerAction"])
        result = lifecycle.run(max_transitions=10)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual([
            ("042-prepare-gaia-source-resolved-counterpart-photometry",
             "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare")
        ], executions)
        self.assertEqual(2, len(store.load(target.investigation_id).stages))

    def test_transient_failed_gaia_attempt_uses_generic_retry_id_without_loop(self):
        store = InvestigationStore(self.root / "transient-retry-store")
        target = InvestigationTarget(
            "generic-target", "transient-retry", "openstar.workflow.tess-investigation.v1", "20.2"
        )
        investigation = store.create(target.investigation_id, target.workflow_id, target.workflow_version)
        running = InvestigationStage(
            "043-run-gaia-source-resolved-counterpart-photometry",
            "openstar.tess.gaia-source-resolved-counterpart-photometry.run", "RUNNING",
            "042-prepare-gaia-source-resolved-counterpart-photometry", {},
        )
        investigation = store.append_running_stage(investigation, running)
        failed = store.build_terminal_stage(
            stage_id=running.id, handler_id=running.handler_id, status="FAILED",
            triggered_by_stage_id=running.triggered_by_stage_id, parameters={}, result=None,
            error="RetryableExecutionError: outage",
            failure_classification="TRANSIENT_INFRASTRUCTURE",
            software_id="test", software_version="1", started_at=running.started_at,
        )
        investigation = store.complete_current_stage(investigation, failed)
        attempts = []
        workflow = WorkflowEngine(store)

        def retry(current, request):
            attempts.append(request.id)
            return StageOutcome(result={"retried": True}, stop=True)

        workflow.register_handler(failed.handler_id, retry)
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(
            self.root / "transient-retry-lifecycle.json", store, dispatcher,
            InvestigationTargetPortfolio(self.root / "transient-retry-portfolio.json", store, dispatcher),
            _SingleTargetSource(target), {target.workflow_id: lambda investigation, target: ()},
            software_id="test", software_version="1",
        )
        lifecycle.start(target)
        result = lifecycle.run(max_transitions=10)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual(["044-run-gaia-source-resolved-counterpart-photometry"], attempts)


if __name__ == "__main__":
    unittest.main()
