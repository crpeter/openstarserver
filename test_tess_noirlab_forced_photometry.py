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
    stub.float64 = float
    stub.asarray = lambda values, dtype=None: list(values)
    stub.median = lambda values: sorted(values)[len(values) // 2]
    sys.modules["numpy"] = stub
    _installed_numpy_stub = True
else:
    _installed_numpy_stub = False

from workflows.tess import tess_noirlab_forced_photometry as noir
from workflows.tess.tess_autonomy import plan_tess_branches, repair_obsolete_terminal_wait
from workflows.tess.tess_investigation import build_engine

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


class _SingleTargetSource:
    id = "noirlab-test-targets"
    version = "1"

    def __init__(self, target):
        self.target = target

    def list_targets(self):
        return (self.target,)


class CurrentNOIRLabForcedPhotometryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pair = {
            "version": "openstar.current-source-pair.v1",
            "target": {"sourceRole": "target-control", "gaiaDR3SourceID": 101,
                       "raDeg": 10.0, "decDeg": -20.0},
            "counterpart": {"sourceRole": "catalog-counterpart", "gaiaDR3SourceID": 202,
                            "raDeg": 10.001, "decDeg": -20.0},
            "separationArcsec": 999.0,
        }
        self.search = {"totalFrequencies": 100, "frequenciesPerWorkUnit": 10,
                       "minimumFrequency": 1.0, "maximumFrequency": 2.0}

    def _build(self, rows=()):
        with mock.patch.object(noir, "_query_sia", return_value=list(rows)):
            return noir.build_noirlab_image_forced_photometry_project(
                source_project_id="project", source_dataset_id="dataset",
                external_high_resolution_summary={},
                nsc_summary={"recommendedNextTest": noir.CURRENT_TRIGGER,
                             "sourcePair": self.pair, "frequencySearch": self.search},
                output_dir=self.root, investigation_id="generic-investigation")

    def test_current_source_pair_adapter_recomputes_gaia_geometry(self):
        sources, separation = noir._frozen_source_pair({"sourcePair": self.pair})
        self.assertEqual([101, 202], [item["gaiaDR3SourceID"] for item in sources])
        expected = noir._angular_separation_arcsec(10.0, -20.0, 10.001, -20.0)
        self.assertAlmostEqual(expected, separation, places=8)
        self.assertNotEqual(self.pair["separationArcsec"], separation)

    def test_current_trigger_and_frequency_search_are_consumed(self):
        result = self._build()
        self.assertEqual(self.search, result["frequencySearch"])
        self.assertEqual("openstar.lomb-scargle.v1", result["workloadID"])
        self.assertFalse(result["tessDriftExtrapolated"])

    def test_no_sia_coverage_is_scientific_complete(self):
        result = self._build()
        summary = noir.interpret_noirlab_image_forced_photometry_project(
            project_status=None, preparation=result)
        self.assertEqual("NOIRLAB_IMAGE_ARCHIVE_NO_SINGLE_EPOCH_CANDIDATES",
                         summary["classification"])
        self.assertEqual(noir.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])

    def test_incomplete_target_control_cannot_decisively_attribute_counterpart(self):
        prepared = [{"datasetID": "tg", "sourceRole": "target-control", "band": "g"},
                    {"datasetID": "cg", "sourceRole": "catalog-counterpart", "band": "g"},
                    {"datasetID": "cr", "sourceRole": "catalog-counterpart", "band": "r"}]
        datasets = [{"datasetID": item["datasetID"], "periodStatus": "RELIABLE",
                     "coverageComplete": True, "candidateFrequency": 1.2,
                     "candidatePeakProminenceRatio": 4.0} for item in prepared
                    if item["sourceRole"] == "catalog-counterpart"]
        minimal_numpy = SimpleNamespace(
            float64=float, asarray=lambda values, dtype=None: list(values),
            median=lambda values: sorted(values)[len(values) // 2])
        with mock.patch.object(noir, "np", minimal_numpy):
            summary = noir.interpret_noirlab_image_forced_photometry_project(
                project_status={"datasets": datasets}, preparation={
                    "preparedSeries": prepared, "candidateExposures": 20,
                    "successfulForcedPhotometryExposures": 20,
                    "workloadID": "openstar.lomb-scargle.v1"})
        self.assertTrue(summary["catalogCounterpartEvidence"]["sourceSupported"])
        self.assertFalse(summary["targetControl"]["scientificallyUsableControl"])
        self.assertEqual(noir.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])

    def test_missing_counterpart_result_is_not_decisive_for_supported_target(self):
        prepared = [{"datasetID": "tg", "sourceRole": "target-control", "band": "g"},
                    {"datasetID": "tr", "sourceRole": "target-control", "band": "r"},
                    {"datasetID": "cg", "sourceRole": "catalog-counterpart", "band": "g"}]
        datasets = [{"datasetID": item["datasetID"], "periodStatus": "RELIABLE",
                     "coverageComplete": True, "candidateFrequency": 1.2,
                     "candidatePeakProminenceRatio": 4.0} for item in prepared
                    if item["sourceRole"] == "target-control"]
        summary = self._interpret_with_minimal_numpy(prepared, datasets)
        self.assertTrue(summary["targetControl"]["sourceSupported"])
        self.assertFalse(summary["catalogCounterpartEvidence"]["scientificallyUsableControl"])
        self.assertEqual(noir.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])

    def _interpret_with_minimal_numpy(self, prepared, datasets):
        minimal_numpy = SimpleNamespace(
            float64=float, asarray=lambda values, dtype=None: list(values),
            median=lambda values: sorted(values)[len(values) // 2])
        with mock.patch.object(noir, "np", minimal_numpy):
            return noir.interpret_noirlab_image_forced_photometry_project(
                project_status={"datasets": datasets}, preparation={
                    "preparedSeries": prepared, "candidateExposures": 20,
                    "successfulForcedPhotometryExposures": 20,
                    "workloadID": "openstar.lomb-scargle.v1"})

    def test_usable_nonrecurrent_control_preserves_decisive_one_sided_routes(self):
        prepared = []
        for role, prefix in (("target-control", "t"),
                             ("catalog-counterpart", "c")):
            for band in ("g", "r"):
                prepared.append({"datasetID": f"{prefix}{band}", "sourceRole": role,
                                 "band": band})

        def supported(role):
            rows = []
            for item in prepared:
                is_supported = item["sourceRole"] == role
                rows.append({"datasetID": item["datasetID"],
                    "periodStatus": "RELIABLE" if is_supported else "LOW_CONFIDENCE",
                    "coverageComplete": True,
                    "candidateFrequency": 1.2 if is_supported else None,
                    "candidatePeakProminenceRatio": 4.0 if is_supported else 1.0})
            return rows

        counterpart = self._interpret_with_minimal_numpy(
            prepared, supported("catalog-counterpart"))
        self.assertTrue(counterpart["targetControl"]["scientificallyUsableControl"])
        self.assertEqual("TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL",
                         counterpart["recommendedNextTest"])
        target = self._interpret_with_minimal_numpy(prepared, supported("target-control"))
        self.assertTrue(target["catalogCounterpartEvidence"]["scientificallyUsableControl"])
        self.assertEqual("TARGET_INTRINSIC_RESIDUAL_MODELING",
                         target["recommendedNextTest"])

    def test_transient_error_classification_is_narrow(self):
        for code, expected in ((408, True), (425, True), (429, True),
                               (500, True), (503, True), (400, False)):
            error = urllib.error.HTTPError("url", code, "x", {}, None)
            self.assertEqual(expected, noir._retryable_service_error(error))
        self.assertFalse(noir._retryable_service_error(ValueError("local bug")))
        self.assertFalse(noir._retryable_service_error(ImportError("astropy")))

    def test_real_047_blocked_reopens_exact_048_without_prior_archive_reruns(self):
        stage = InvestigationStage(
            "047-interpret-nsc-resolved-counterpart-photometry",
            "openstar.tess.nsc-resolved-photometry.interpret", "COMPLETE", None, {},
            result={"recommendedNextTest": noir.CURRENT_TRIGGER,
                    "physicalMechanismResolved": False}, stop=True)
        investigation = Investigation(
            "generic", "openstar.workflow.tess-investigation.v1", "20.2", "BLOCKED",
            "now", "now", {"datasetID": "generic", "controlState": {
                "schedulerAction": "WAIT_FOR_PREREQUISITES"}}, (stage,))
        target = InvestigationTarget("generic", "generic", investigation.workflow_id,
                                     investigation.workflow_version)
        branch = plan_tess_branches(investigation, target)[0]
        self.assertEqual("048-prepare-noirlab-image-level-forced-photometry",
                         branch.experiment.id)
        self.assertEqual((), branch.required_stage_ids)
        self.assertNotIn("gaia", branch.experiment.handler_id)
        self.assertNotIn("skymapper", branch.experiment.handler_id)
        self.assertNotIn("nsc-resolved-photometry.prepare", branch.experiment.handler_id)
        store = InvestigationStore(self.root / "state")
        store.save(investigation)
        repaired = repair_obsolete_terminal_wait(store, investigation)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(branch.experiment.id, selected["id"])
        self.assertEqual("RUNNING", repaired.status)

    def test_repair_ignores_invalid_control_state_and_other_workflows(self):
        store = InvestigationStore(self.root / "guard-state")
        cases = ({}, {"controlState": None}, {"controlState": "invalid"})
        for index, metadata in enumerate(cases):
            investigation = Investigation(
                f"guard-{index}", "openstar.workflow.tess-investigation.v1", "20.2",
                "BLOCKED", "now", "now", metadata, ())
            self.assertIs(investigation,
                          repair_obsolete_terminal_wait(store, investigation))
        other = Investigation("other", "openstar.workflow.other", "1", "BLOCKED",
                              "now", "now", {"controlState": None}, ())
        self.assertIs(other, repair_obsolete_terminal_wait(store, other))

    def _current_evidence(self, investigation_id="handlers"):
        return Investigation(
            investigation_id, "openstar.workflow.tess-investigation.v1", "20.2",
            "RUNNING", "now", "now", {"datasetID": "generic-dataset"}, (
                InvestigationStage("001-prepare", "openstar.tess.prepare-target",
                    "COMPLETE", None, {}, result={"sourceProjectID": "project",
                                                   "datasetID": "dataset"}),
                InvestigationStage("043-gaia",
                    "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                    "COMPLETE", None, {}, result={"sourcePair": self.pair,
                        "frequencySearch": self.search}),
                InvestigationStage("047-nsc", "openstar.tess.nsc-resolved-photometry.interpret",
                    "COMPLETE", None, {}, result={"sourcePair": self.pair,
                        "frequencySearch": self.search,
                        "recommendedNextTest": noir.CURRENT_TRIGGER,
                        "physicalMechanismResolved": False}),))

    def _run_registered_chain(self, distributed):
        store = InvestigationStore(self.root / ("registered-run" if distributed else "registered-direct"))
        investigation = self._current_evidence()
        store.save(investigation)
        project_path = self.root / "noirlab-project.json"
        if distributed:
            project_path.write_text("{}", encoding="utf-8")
        spec = {"available": distributed,
                "projectPath": str(project_path) if distributed else None,
                "preparedSeries": [], "workloadID": "openstar.lomb-scargle.v1",
                "candidateExposures": 0, "successfulForcedPhotometryExposures": 0}
        coordinator = mock.Mock()
        coordinator.run_project.return_value = SimpleNamespace(
            status={"datasets": []}, node_contributions={}, project_id="generic-project")
        summary = {"recommendedNextTest": noir.NEXT_ARCHIVE_TEST,
                   "physicalMechanismResolved": False}
        with mock.patch(
            "workflows.tess.tess_investigation.build_noirlab_image_forced_photometry_project",
            return_value=spec) as prepare, mock.patch(
            "workflows.tess.tess_investigation.interpret_noirlab_image_forced_photometry_project",
            return_value=summary) as interpret:
            completed = build_engine(store, coordinator, poll_interval=0, timeout=1).run(
                investigation, StageRequest(
                    "048-prepare-noirlab-image-level-forced-photometry",
                    "openstar.tess.noirlab-image-forced-photometry.prepare", {}, "047-nsc"),
                software_id="test", software_version="1")
        expected = ["openstar.tess.noirlab-image-forced-photometry.prepare"]
        if distributed:
            expected.append("openstar.tess.noirlab-image-forced-photometry.run")
        expected.append("openstar.tess.noirlab-image-forced-photometry.interpret")
        self.assertEqual(expected, [stage.handler_id for stage in completed.stages[3:]])
        self.assertEqual("BLOCKED", completed.status)
        prepare.assert_called_once()
        interpret.assert_called_once()
        self.assertEqual(1 if distributed else 0, coordinator.run_project.call_count)

    def test_registered_prepare_run_interpret(self):
        self._run_registered_chain(True)

    def test_registered_prepare_interpret_without_project(self):
        self._run_registered_chain(False)

    def test_exact_persisted_noirlab_next_stage_is_reused(self):
        expected = StageRequest("049-custom-persisted-interpret",
            "openstar.tess.noirlab-image-forced-photometry.interpret",
            {"distributedRunExpected": False, "sentinel": 7}, "048-prepare")
        stage = InvestigationStage("048-prepare",
            "openstar.tess.noirlab-image-forced-photometry.prepare", "COMPLETE", None,
            {}, result={}, next_stage=asdict(expected))
        investigation = Investigation("persisted", "openstar.workflow.tess-investigation.v1",
            "20.2", "COMPLETE", "now", "now", {}, (stage,))
        target = InvestigationTarget("persisted", "persisted", investigation.workflow_id,
                                     investigation.workflow_version)
        self.assertEqual(expected, plan_tess_branches(investigation, target)[0].experiment)

    def test_unknown_runtime_and_import_error_propagate_from_image_processing(self):
        exposure = {"access_url": "https://example/image", "mjd_obs": "1.0",
                    "obs_bandpass": "g", "prodtype": "image", "obs_id": "generic-image"}
        for error in (RuntimeError("programming bug"), ImportError("astropy")):
            with self.subTest(error=type(error).__name__), mock.patch.object(
                noir, "_query_sia", return_value=[exposure]), mock.patch.object(
                noir, "_download_fits_cutout", return_value=b"fits"), mock.patch.object(
                noir, "_image_hdu_from_bytes", side_effect=error):
                with self.assertRaises(type(error)):
                    self._build([exposure])

    def test_quality_rejection_and_isolated_transport_failure_are_scientific(self):
        exposures = [{"access_url": f"https://example/{i}", "mjd_obs": str(i),
                      "obs_bandpass": "g", "prodtype": "image",
                      "obs_id": f"image-{i}"} for i in range(2)]
        effects = [TimeoutError("isolated"), noir.NOIRLabImageQualityRejected("target-saturated")]
        with mock.patch.object(noir, "_download_fits_cutout", side_effect=effects):
            result = self._build(exposures)
        self.assertFalse(result["available"])
        self.assertEqual(1, result["failureReasons"]["download-error"])
        self.assertEqual(1, result["failureReasons"]["target-saturated"])

    def test_broad_download_outage_is_retryable_archive_failure(self):
        exposures = [{"access_url": f"https://example/{i}", "mjd_obs": str(i),
                      "obs_bandpass": "g", "prodtype": "image",
                      "obs_id": f"image-{i}"} for i in range(4)]
        with mock.patch.object(noir, "_download_fits_cutout",
                               side_effect=TimeoutError("outage")):
            with self.assertRaises(noir.NOIRLabArchiveUnavailable):
                self._build(exposures)

    def test_sia_timeout_registered_prepare_is_retryable(self):
        store = InvestigationStore(self.root / "sia-retry")
        investigation = self._current_evidence("sia-retry")
        store.save(investigation)
        from openstar_workflow import RetryableExecutionError
        with mock.patch(
            "workflows.tess.tess_investigation.build_noirlab_image_forced_photometry_project",
            side_effect=noir.NOIRLabArchiveUnavailable("timeout")):
            with self.assertRaises(RetryableExecutionError):
                build_engine(store, mock.Mock(), poll_interval=0, timeout=1).run(
                    investigation, StageRequest("048-prepare-noirlab",
                        "openstar.tess.noirlab-image-forced-photometry.prepare", {}, "047-nsc"),
                    software_id="test", software_version="1")
        failed = store.load(investigation.id)
        self.assertEqual("TRANSIENT_INFRASTRUCTURE",
                         failed.stages[-1].failure_classification)

    def test_registered_prepare_records_local_errors_nonretryable(self):
        for index, error in enumerate((RuntimeError("programming bug"),
                                       ImportError("astropy"))):
            store = InvestigationStore(self.root / f"local-error-{index}")
            investigation = self._current_evidence(f"local-error-{index}")
            store.save(investigation)
            with self.subTest(error=type(error).__name__), mock.patch(
                "workflows.tess.tess_investigation.build_noirlab_image_forced_photometry_project",
                side_effect=error):
                with self.assertRaises(type(error)):
                    build_engine(store, mock.Mock(), poll_interval=0, timeout=1).run(
                        investigation, StageRequest("048-prepare-noirlab",
                            "openstar.tess.noirlab-image-forced-photometry.prepare", {},
                            "047-nsc"), software_id="test", software_version="1")
            failed = store.load(investigation.id)
            self.assertEqual("NON_RETRYABLE",
                             failed.stages[-1].failure_classification)

    def test_scientific_empty_fit_cadence_and_flat_outcomes_route_to_des(self):
        cases = (
            {"candidateExposures": 8, "successfulForcedPhotometryExposures": 0,
             "failureReasons": {"target-saturated": 8}},
            {"candidateExposures": 8, "successfulForcedPhotometryExposures": 0,
             "failureReasons": {"two-source-fit-low-explained-variance": 8}},
            {"candidateExposures": 8, "successfulForcedPhotometryExposures": 5,
             "failureReasons": {}},
            {"candidateExposures": 12, "successfulForcedPhotometryExposures": 12,
             "failureReasons": {"target-control-g-flat-magnitude-series": 1}},
        )
        for preparation in cases:
            preparation["preparedSeries"] = []
            summary = noir.interpret_noirlab_image_forced_photometry_project(
                project_status=None, preparation=preparation)
            self.assertEqual(noir.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])
            self.assertFalse(summary["physicalMechanismResolved"])

    def _run_persisted_prepare_lifecycle(self, distributed):
        suffix = "run" if distributed else "direct"
        store = InvestigationStore(self.root / f"persisted-{suffix}")
        target = InvestigationTarget("generic", f"persisted-{suffix}",
            "openstar.workflow.tess-investigation.v1", "20.2",
            metadata={"datasetID": "generic"})
        next_request = StageRequest(
            "049-run-noirlab" if distributed else "049-interpret-noirlab",
            "openstar.tess.noirlab-image-forced-photometry.run" if distributed else
            "openstar.tess.noirlab-image-forced-photometry.interpret",
            {"projectPath": "/persisted/project.json"} if distributed else
            {"distributedRunExpected": False}, "048-prepare-noirlab")
        prepare = InvestigationStage("048-prepare-noirlab",
            "openstar.tess.noirlab-image-forced-photometry.prepare", "COMPLETE", None,
            {}, result={}, next_stage=asdict(next_request))
        investigation = Investigation(target.investigation_id, target.workflow_id,
            target.workflow_version, "COMPLETE", "now", "now", target.metadata, (prepare,))
        store.save(investigation)
        investigation, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target))
        executions = []
        workflow = WorkflowEngine(store)
        workflow.register_handler("openstar.tess.noirlab-image-forced-photometry.prepare",
            lambda *_: self.fail("completed NOIRLab prepare must not rerun"))
        if distributed:
            def run_stage(current, request):
                executions.append(request)
                return StageOutcome(result={"datasets": []}, next_stage=StageRequest(
                    "050-interpret-noirlab",
                    "openstar.tess.noirlab-image-forced-photometry.interpret",
                    {"distributedRunExpected": True}, request.id))
            workflow.register_handler(next_request.handler_id, run_stage)
        def interpret(current, request):
            executions.append(request)
            return StageOutcome(result={"recommendedNextTest": noir.NEXT_ARCHIVE_TEST,
                "physicalMechanismResolved": False}, stop=True, final_status="BLOCKED")
        workflow.register_handler("openstar.tess.noirlab-image-forced-photometry.interpret",
                                  interpret)
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(self.root / f"lifecycle-{suffix}.json",
            store, dispatcher,
            InvestigationTargetPortfolio(self.root / f"portfolio-{suffix}.json", store, dispatcher),
            _SingleTargetSource(target), {target.workflow_id: plan_tess_branches},
            software_id="test", software_version="1")
        lifecycle.start(target)
        result = lifecycle.run(max_transitions=20)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual(next_request, executions[0])
        self.assertEqual(prepare, store.load(investigation.id).stages[0])
        self.assertEqual("BLOCKED", store.load(investigation.id).status)
        self.assertEqual("WAIT_FOR_PREREQUISITES",
            store.load(investigation.id).metadata["controlState"]["schedulerAction"])

    def test_persisted_prepare_run_interpret_without_reprepare_or_loop(self):
        self._run_persisted_prepare_lifecycle(True)

    def test_persisted_prepare_interpret_without_reprepare_or_loop(self):
        self._run_persisted_prepare_lifecycle(False)

    def test_failed_noirlab_planning_and_completed_interpretation_are_idempotent(self):
        target = InvestigationTarget("generic", "failure",
            "openstar.workflow.tess-investigation.v1", "20.2")
        failed = InvestigationStage("049-run-noirlab",
            "openstar.tess.noirlab-image-forced-photometry.run", "FAILED", None, {},
            error="RuntimeError: programming bug", failure_classification="NON_RETRYABLE")
        investigation = Investigation("failure", target.workflow_id, target.workflow_version,
            "FAILED", "now", "now", {}, (failed,))
        branches = plan_tess_branches(investigation, target)
        self.assertFalse(any("noirlab-image-forced-photometry" in branch.experiment.handler_id
                             for branch in branches))
        complete = InvestigationStage("050-interpret-noirlab",
            "openstar.tess.noirlab-image-forced-photometry.interpret", "COMPLETE", None, {},
            result={"recommendedNextTest": noir.NEXT_ARCHIVE_TEST,
                    "physicalMechanismResolved": False,
                    "sourcePair": self.pair}, stop=True)
        completed = Investigation("complete", target.workflow_id, target.workflow_version,
            "BLOCKED", "now", "now", {}, (complete,))
        branch = plan_tess_branches(completed, target)[0]
        self.assertEqual("openstar.tess.des-dr2-se-local-forced-photometry.prepare",
                         branch.experiment.handler_id)
        self.assertEqual((), branch.required_stage_ids)

    def test_transient_failed_noirlab_run_uses_fresh_generic_retry_id(self):
        store = InvestigationStore(self.root / "run-retry")
        target = InvestigationTarget("generic", "run-retry",
            "openstar.workflow.tess-investigation.v1", "20.2")
        prepare = InvestigationStage("048-prepare-noirlab",
            "openstar.tess.noirlab-image-forced-photometry.prepare", "COMPLETE", None,
            {}, result={})
        failed = InvestigationStage("049-run-noirlab",
            "openstar.tess.noirlab-image-forced-photometry.run", "FAILED",
            prepare.id, {"projectPath": "/persisted/project.json"},
            error="RetryableExecutionError: outage",
            failure_classification="TRANSIENT_INFRASTRUCTURE")
        investigation = Investigation(target.investigation_id, target.workflow_id,
            target.workflow_version, "FAILED", "now", "now", {}, (prepare, failed))
        store.save(investigation)
        requests = []
        workflow = WorkflowEngine(store)
        def retry_run(current, request):
            requests.append(request)
            return StageOutcome(result={"datasets": []}, next_stage=StageRequest(
                "051-interpret-noirlab",
                "openstar.tess.noirlab-image-forced-photometry.interpret",
                {"distributedRunExpected": True}, request.id))
        workflow.register_handler(failed.handler_id, retry_run)
        workflow.register_handler("openstar.tess.noirlab-image-forced-photometry.interpret",
            lambda current, request: StageOutcome(result={
                "recommendedNextTest": noir.NEXT_ARCHIVE_TEST,
                "physicalMechanismResolved": False}, stop=True, final_status="BLOCKED"))
        dispatcher = InvestigationDispatcher(store, workflow)
        lifecycle = InvestigationLifecycleLoop(self.root / "retry-lifecycle.json", store,
            dispatcher, InvestigationTargetPortfolio(self.root / "retry-portfolio.json",
                store, dispatcher), _SingleTargetSource(target),
            {target.workflow_id: plan_tess_branches}, software_id="test", software_version="1")
        lifecycle.start(target)
        result = lifecycle.run(max_transitions=20)
        self.assertNotEqual("LIFECYCLE_CHECKPOINT", result.disposition)
        self.assertEqual("050-run-noirlab", requests[0].id)
        self.assertEqual(failed.id, requests[0].triggered_by_stage_id)
        self.assertEqual("WAIT_FOR_PREREQUISITES",
            store.load(investigation.id).metadata["controlState"]["schedulerAction"])


if __name__ == "__main__":
    unittest.main()
