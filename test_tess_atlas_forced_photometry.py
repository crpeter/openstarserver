import os
import sys
import tempfile
import types
import unittest
import urllib.error
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("numpy")
    stub.integer = int
    stub.floating = float
    stub.float64 = float
    stub.float32 = float
    stub.asarray = lambda values, dtype=None: list(values)
    stub.median = lambda values: sorted(values)[len(values) // 2]
    sys.modules["numpy"] = stub
    _installed_numpy_stub = True
else:
    _installed_numpy_stub = False

from openstar_investigation import Investigation, InvestigationStage, InvestigationStore
from openstar_lifecycle import InvestigationLifecycleLoop
from openstar_external_jobs import (ExternalJob, ExternalJobPollUnavailable,
                                    ExternalDependency, ExternalJobStore,
                                    apply_external_job_wakeups)
from openstar_targets import InvestigationTarget
from openstar_workflow import RetryableExecutionError, StageRequest
from workflows.tess import tess_atlas_forced_photometry as atlas
from workflows.tess.tess_autonomy import plan_tess_branches, repair_obsolete_terminal_wait

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


def _workflow_module():
    """Import the registered workflow lazily so discovery does not leak a tiny numpy stub."""
    installed = "numpy" not in sys.modules
    if installed:
        sys.modules["numpy"] = atlas.np
    try:
        from workflows.tess import tess_investigation
    finally:
        if installed:
            sys.modules.pop("numpy", None)
    return tess_investigation


def _build_engine(*args, **kwargs):
    return _workflow_module().build_engine(*args, **kwargs)


class CurrentATLASForcedPhotometryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pair = {
            "version": "openstar.current-source-pair.v1",
            "target": {"sourceRole": "target-control", "gaiaDR3SourceID": 101,
                       "raDeg": 10.0, "decDeg": -20.0},
            "counterpart": {"sourceRole": "catalog-counterpart", "gaiaDR3SourceID": 202,
                            "raDeg": 10.01, "decDeg": -20.0},
            "separationArcsec": 999.0,
        }

    def test_current_source_pair_coordinates_define_geometry_and_guard(self):
        sources, separation = atlas._frozen_sources({"sourcePair": self.pair})
        expected = atlas._angular_separation_arcsec(10, -20, 10.01, -20)
        self.assertAlmostEqual(expected, separation)
        self.assertNotEqual(999, separation)
        self.assertEqual([101, 202], [item["gaiaDR3SourceID"] for item in sources])
        self.assertEqual(15.0, atlas.MIN_GAIA_PAIR_SEPARATION_ARCSEC)

    def test_current_trigger_direct_frequency_search_and_generic_worker(self):
        search = {"totalFrequencies": 10, "frequenciesPerWorkUnit": 2}
        empty = "# MJD uJy duJy F err chi/N\n"
        with mock.patch.object(atlas, "require_atlas_credentials"), \
                mock.patch.object(atlas, "_atlas_headers", return_value={}), \
                mock.patch.object(atlas, "_submit_atlas_job", return_value="task"), \
                mock.patch.object(atlas, "_wait_for_atlas_job", return_value=("result", empty)):
            result = atlas.build_atlas_forced_photometry_project(
                source_project_id="p", source_dataset_id="d",
                external_high_resolution_summary={"sourcePair": self.pair},
                des_dr2_se_summary={"recommendedNextTest": atlas.CURRENT_TRIGGER,
                                    "frequencySearch": search},
                output_dir=self.root, investigation_id="generic")
        self.assertEqual(search, result["frequencySearch"])
        self.assertEqual("openstar.lomb-scargle.v1", result["workloadID"])
        self.assertFalse(result["tessDriftExtrapolated"])
        self.assertTrue(all(Path(item["rawPath"]).exists()
                            for item in result["sourceRecords"]))

    def test_transient_contract_and_no_unbounded_429_retry(self):
        for code in (408, 425, 429, 500, 503):
            self.assertTrue(atlas._retryable_service_error(
                urllib.error.HTTPError("url", code, "x", {}, None)))
        self.assertFalse(atlas._retryable_service_error(RuntimeError("bug")))
        self.assertFalse(atlas._retryable_service_error(ValueError("parse bug")))
        with mock.patch.object(atlas, "_json_request", return_value=(429, {"detail": "wait"})), \
                mock.patch.object(atlas.time, "sleep") as sleep:
            with self.assertRaises(atlas.ATLASArchiveUnavailable):
                atlas._submit_atlas_job({}, ra_deg=1, dec_deg=2)
            sleep.assert_not_called()

    def test_poll_adapter_classifies_only_retryable_service_failures(self):
        job = ExternalJob.create(provider="atlas-forced-photometry",
            investigation_id="inv", trigger_stage_id="052", dependency_id="dep",
            role="target-control")
        job = replace(job, remoteTaskURL="task")
        provider = atlas.ATLASExternalJobProvider()
        with mock.patch.object(atlas, "_atlas_headers", return_value={}), \
                mock.patch.object(atlas, "_json_request", return_value=(503, {})):
            with self.assertRaises(ExternalJobPollUnavailable): provider.poll(job)
        with mock.patch.object(atlas, "_atlas_headers", return_value={}), \
                mock.patch.object(atlas, "_json_request", return_value=(401, {})):
            with self.assertRaises(RuntimeError): provider.poll(job)
        with mock.patch.object(atlas, "_atlas_headers", side_effect=ValueError("bug")):
            with self.assertRaisesRegex(ValueError, "bug"): provider.poll(job)

    def test_partial_submission_retry_reuses_target_and_manifest(self):
        jobs = ExternalJobStore(self.root / "partial" / "external-jobs")
        search = {"totalFrequencies": 10, "frequenciesPerWorkUnit": 2}
        submit = mock.Mock(side_effect=["target-task",
            atlas.ATLASArchiveUnavailable("counterpart unavailable"),
            "counterpart-task"])
        kwargs = dict(source_project_id="p", source_dataset_id="d",
            external_high_resolution_summary={"sourcePair": self.pair},
            des_dr2_se_summary={"recommendedNextTest": atlas.CURRENT_TRIGGER,
                                "frequencySearch": search},
            investigation_id="inv", trigger_stage_id="052-prepare", job_store=jobs)
        with mock.patch.object(atlas, "require_atlas_credentials"), \
                mock.patch.object(atlas, "_atlas_headers", return_value={}), \
                mock.patch.object(atlas, "_submit_atlas_job", submit):
            with self.assertRaises(atlas.ATLASArchiveUnavailable):
                atlas.submit_atlas_forced_photometry_jobs(**kwargs)
            result = atlas.submit_atlas_forced_photometry_jobs(**kwargs)
        self.assertEqual(3, submit.call_count)
        self.assertEqual([10.0, 10.01, 10.01],
                         [call.kwargs["ra_deg"] for call in submit.call_args_list])
        self.assertEqual(2, len(jobs.dependencies()[0].expectedJobIDs))
        self.assertEqual(2, len(jobs.list()))
        self.assertEqual(2, len(result["externalJobIDs"]))

    def test_restart_complete_records_wake_exact_053_collect_without_loop(self):
        dependency = "atlas-forced-photometry:blind:052"
        submission = InvestigationStage("052-prepare-atlas-forced-photometry",
            "openstar.tess.atlas-forced-photometry.prepare", "COMPLETE", "051", {},
            result={"externalDependencyID": dependency,
                    "externalJobIDs": ["target", "counterpart"]}, stop=True)
        inv = Investigation("blind", "openstar.workflow.tess-investigation.v1", "20.2",
            "QUIESCENT_AWAITING_DATA", "now", "now", {"controlState": {
                "schedulerAction": "ADVANCE_TO_NEXT_TARGET"}}, (submission,))
        store = InvestigationStore(self.root / "state" / "investigations"); store.save(inv)
        jobs = ExternalJobStore(self.root / "state" / "external-jobs")
        expected_ids = tuple(ExternalJob.create(provider="atlas-forced-photometry",
            investigation_id="blind", trigger_stage_id=submission.id,
            dependency_id=dependency, role=role).id
            for role in ("target-control", "catalog-counterpart"))
        jobs.save_dependency(ExternalDependency(dependency, "blind", submission.id,
            "atlas-forced-photometry", expected_ids))
        for role in ("target-control", "catalog-counterpart"):
            job = ExternalJob.create(provider="atlas-forced-photometry",
                investigation_id="blind", trigger_stage_id=submission.id,
                dependency_id=dependency, role=role)
            jobs.save(replace(job, state="COMPLETE", remoteTaskURL=f"task-{role}",
                              remoteResultURL=f"result-{role}"))
        restarted = ExternalJobStore(jobs.root)
        self.assertEqual((), restarted.pending())
        apply_external_job_wakeups(store, restarted.ready_dependencies())
        awakened = store.load("blind")
        branch = plan_tess_branches(awakened, InvestigationTarget(
            "blind", "blind", awakened.workflow_id, awakened.workflow_version))[0]
        self.assertEqual("053-collect-atlas-forced-photometry", branch.experiment.id)
        self.assertEqual("openstar.tess.atlas-forced-photometry.collect",
                         branch.experiment.handler_id)
        self.assertTrue(branch.external_data[0].available)
        first = store.path_for("blind").read_bytes()
        apply_external_job_wakeups(store, restarted.ready_dependencies())
        self.assertEqual(first, store.path_for("blind").read_bytes())

    def _registered_collect_evidence(self, name):
        store = InvestigationStore(self.root / name / "investigations")
        dependency = f"atlas-forced-photometry:{name}:052"
        job_store = ExternalJobStore(store.root.parent / "external-jobs")
        job_ids = []
        for role in ("target-control", "catalog-counterpart"):
            job = ExternalJob.create(provider="atlas-forced-photometry",
                investigation_id=name, trigger_stage_id="052-prepare",
                dependency_id=dependency, role=role)
            job_store.save(replace(job, state="COMPLETE",
                remoteTaskURL=f"task-{role}", remoteResultURL=f"result-{role}"))
            job_ids.append(job.id)
        job_store.save_dependency(ExternalDependency(dependency, name, "052-prepare",
            "atlas-forced-photometry", tuple(job_ids)))
        submission = InvestigationStage("052-prepare",
            "openstar.tess.atlas-forced-photometry.prepare", "COMPLETE", "051", {},
            result={"sourceProjectID": "p", "sourceDatasetID": "d",
                "sourcePair": self.pair, "frequencySearch": {
                    "totalFrequencies": 10, "frequenciesPerWorkUnit": 2},
                "externalDependencyID": dependency, "externalJobIDs": job_ids}, stop=True)
        investigation = Investigation(name, "openstar.workflow.tess-investigation.v1",
            "20.2", "RUNNING", "now", "now", {}, (submission,))
        store.save(investigation)
        return store, investigation, job_store

    def test_registered_collect_transient_retries_download_only_with_same_jobs(self):
        store, investigation, jobs = self._registered_collect_evidence("collect-retry")
        workflow = _workflow_module()
        captured = []
        def build(**kwargs):
            captured.append(tuple((job.id, job.remoteTaskURL, job.remoteResultURL)
                                  for job in kwargs["external_jobs"]))
            return atlas.build_atlas_forced_photometry_project(**kwargs)
        engine = _build_engine(store, mock.Mock(), poll_interval=0, timeout=1)
        request = StageRequest("053-collect-atlas-forced-photometry",
            "openstar.tess.atlas-forced-photometry.collect", {}, "052-prepare")
        with mock.patch.object(workflow, "build_atlas_forced_photometry_project",
                               side_effect=build), \
                mock.patch.object(atlas, "require_atlas_credentials"), \
                mock.patch.object(atlas, "_atlas_headers", return_value={}), \
                mock.patch.object(atlas, "_text_request", side_effect=[
                    atlas.ATLASArchiveUnavailable("temporary result download outage"),
                    "# MJD uJy duJy F err chi/N\n",
                    "# MJD uJy duJy F err chi/N\n",
                ]) as download, \
                mock.patch.object(atlas, "_submit_atlas_job") as submit:
            with self.assertRaises(RetryableExecutionError):
                engine.run_stage(investigation, request,
                    software_id="test", software_version="1")
            failed_investigation = store.load(investigation.id)
            failed = failed_investigation.stages[-1]
            self.assertEqual("TRANSIENT_INFRASTRUCTURE", failed.failure_classification)
            retry = InvestigationLifecycleLoop._retry_request(failed_investigation, failed)
            self.assertEqual("054-collect-atlas-forced-photometry", retry.id)
            engine.run_stage(failed_investigation, retry,
                software_id="test", software_version="1")
            submit.assert_not_called()
            self.assertEqual(3, download.call_count)
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(tuple(jobs.load(job_id).remoteTaskURL
                               for job_id in failed_investigation.stages[0].result["externalJobIDs"]),
                         tuple(item[1] for item in captured[1]))

    def test_registered_collect_programming_error_is_non_retryable(self):
        store, investigation, _ = self._registered_collect_evidence("collect-bug")
        workflow = _workflow_module()
        engine = _build_engine(store, mock.Mock(), poll_interval=0, timeout=1)
        with mock.patch.object(workflow, "build_atlas_forced_photometry_project",
                               side_effect=RuntimeError("parser bug")):
            with self.assertRaisesRegex(RuntimeError, "parser bug"):
                engine.run_stage(investigation, StageRequest("053-collect",
                    "openstar.tess.atlas-forced-photometry.collect", {}, "052-prepare"),
                    software_id="test", software_version="1")
        self.assertEqual("NON_RETRYABLE",
                         store.load(investigation.id).stages[-1].failure_classification)

    def _interpret(self, supported_role, include_control):
        prepared, datasets = [], []
        for role, prefix in (("target-control", "t"), ("catalog-counterpart", "c")):
            for band in ("c", "o"):
                prepared.append({"datasetID": prefix + band, "sourceRole": role, "band": band})
                if role == supported_role or include_control:
                    supported = role == supported_role
                    datasets.append({"datasetID": prefix + band, "coverageComplete": True,
                        "periodStatus": "RELIABLE" if supported else "NONRECURRENT",
                        "candidateFrequency": 1.2 if supported else None,
                        "candidatePeakProminenceRatio": 4 if supported else None})
        return atlas.interpret_atlas_forced_photometry_project(
            project_status={"datasets": datasets},
            preparation={"preparedSeries": prepared, "workloadID": "openstar.lomb-scargle.v1"})

    def test_missing_controls_are_nondecisive_and_nonrecurrent_controls_decisive(self):
        self.assertEqual(atlas.HISTORICAL_TRIGGER,
                         self._interpret("target-control", False)["recommendedNextTest"])
        self.assertEqual(atlas.HISTORICAL_TRIGGER,
                         self._interpret("catalog-counterpart", False)["recommendedNextTest"])
        self.assertEqual("TARGET_INTRINSIC_RESIDUAL_MODELING",
                         self._interpret("target-control", True)["recommendedNextTest"])
        self.assertEqual("TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL",
                         self._interpret("catalog-counterpart", True)["recommendedNextTest"])

    def test_search_interpretable_uses_completed_search_status_allowlist_contract(self):
        prepared = {"datasetID": "d", "sourceRole": "target-control", "band": "c"}
        for status in ("NONRECURRENT", "RELIABLE"):
            result = atlas._dataset_result(
                {"periodStatus": status, "coverageComplete": True}, prepared
            )
            self.assertTrue(result["searchInterpretable"], status)
        for status in (
            "SEARCHING", "INCOMPLETE_COVERAGE", "NO_DATASET", "FAILED", "ERROR"
        ):
            result = atlas._dataset_result(
                {"periodStatus": status, "coverageComplete": True}, prepared
            )
            self.assertFalse(result["searchInterpretable"], status)
        self.assertFalse(atlas._dataset_result(
            {"periodStatus": "RELIABLE", "coverageComplete": False}, prepared
        )["searchInterpretable"])

    def test_unusable_opposite_search_statuses_block_one_sided_attribution(self):
        for supported_role, control_status in (
            ("catalog-counterpart", "ERROR"),
            ("target-control", "NO_DATASET"),
        ):
            prepared, datasets = [], []
            for role, prefix in (("target-control", "t"),
                                 ("catalog-counterpart", "c")):
                for band in ("c", "o"):
                    dataset_id = prefix + band
                    prepared.append({"datasetID": dataset_id,
                                     "sourceRole": role, "band": band})
                    supported = role == supported_role
                    datasets.append({
                        "datasetID": dataset_id,
                        "coverageComplete": True,
                        "periodStatus": "RELIABLE" if supported else control_status,
                        "candidateFrequency": 1.2 if supported else None,
                        "candidatePeakProminenceRatio": 4 if supported else None,
                    })
            summary = atlas.interpret_atlas_forced_photometry_project(
                project_status={"datasets": datasets},
                preparation={"preparedSeries": prepared,
                             "workloadID": "openstar.lomb-scargle.v1"},
            )
            self.assertEqual(atlas.HISTORICAL_TRIGGER,
                             summary["recommendedNextTest"])

    def test_no_qualifying_series_routes_to_signed_reanalysis_boundary(self):
        summary = atlas.interpret_atlas_forced_photometry_project(
            project_status=None, preparation={"preparedSeries": [],
                                               "sourceRecords": [{"rawPath": "/immutable/raw"}]})
        self.assertEqual(atlas.SIGNED_REANALYSIS, summary["recommendedNextTest"])
        stage = InvestigationStage("053-interpret-atlas-forced-photometry",
            "openstar.tess.atlas-forced-photometry.interpret", "COMPLETE", None, {},
            result=summary, stop=True)
        inv = Investigation("signed", "openstar.workflow.tess-investigation.v1", "20.2",
                            "BLOCKED", "now", "now", {}, (stage,))
        branch = plan_tess_branches(inv, InvestigationTarget(
            "signed", "signed", inv.workflow_id, inv.workflow_version))[0]
        self.assertEqual(("openstar.capability.current-atlas-signed-reanalysis-adapter",),
                         branch.required_stage_ids)

    def _blocked_051(self):
        prior = tuple(InvestigationStage(str(i), handler, "COMPLETE", None, {}, result={})
                      for i, handler in enumerate((
            "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
            "openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
            "openstar.tess.nsc-resolved-photometry.interpret",
            "openstar.tess.noirlab-image-forced-photometry.interpret"), 43))
        des = InvestigationStage("051-interpret-des-dr2-se-local-forced-photometry",
            "openstar.tess.des-dr2-se-local-forced-photometry.interpret", "COMPLETE", None, {},
            result={"recommendedNextTest": atlas.CURRENT_TRIGGER,
                    "physicalMechanismResolved": False, "sourcePair": self.pair}, stop=True)
        return Investigation("blind", "openstar.workflow.tess-investigation.v1", "20.2",
            "BLOCKED", "now", "now", {"datasetID": "blind", "controlState": {
                "schedulerAction": "WAIT_FOR_PREREQUISITES"}}, prior + (des,))

    def test_credentials_gate_exact_051_to_052_without_prior_archive_reruns(self):
        inv = self._blocked_051()
        target = InvestigationTarget("blind", "blind", inv.workflow_id, inv.workflow_version)
        with mock.patch.dict(os.environ, {}, clear=True):
            branch = plan_tess_branches(inv, target)[0]
            self.assertEqual(("openstar.capability.atlas-forced-photometry-credentials",),
                             branch.required_stage_ids)
            store = InvestigationStore(self.root / "store"); store.save(inv)
            migrated = repair_obsolete_terminal_wait(store, inv)
            control = migrated.metadata["controlState"]
            self.assertEqual("WAIT_FOR_PREREQUISITES", control["schedulerAction"])
            self.assertEqual("052-prepare-atlas-forced-photometry",
                             control["selectedExperiment"]["id"])
            self.assertEqual(
                ["openstar.capability.atlas-forced-photometry-credentials"],
                control["missingPrerequisites"])
            self.assertEqual([stage.id for stage in inv.stages],
                             [stage.id for stage in migrated.stages])
            again = repair_obsolete_terminal_wait(store, migrated)
            self.assertEqual(migrated.metadata["controlState"],
                             again.metadata["controlState"])
            self.assertEqual(len(migrated.stages), len(again.stages))
        with mock.patch.dict(os.environ, {"OPENSTAR_ATLAS_API_TOKEN": "token"}, clear=True):
            branch = plan_tess_branches(inv, target)[0]
            self.assertEqual("052-prepare-atlas-forced-photometry", branch.experiment.id)
            self.assertEqual((), branch.required_stage_ids)
            repaired = repair_obsolete_terminal_wait(store, inv)
        self.assertEqual(branch.experiment.id,
                         repaired.metadata["controlState"]["selectedExperiment"]["id"])
        self.assertEqual([stage.id for stage in inv.stages],
                         [stage.id for stage in repaired.stages])

    def test_old_adapter_prerequisite_is_migrated_without_credentials(self):
        inv = self._blocked_051()
        metadata = dict(inv.metadata)
        metadata["controlState"] = {
            "schedulerAction": "WAIT_FOR_PREREQUISITES",
            "selectedExperiment": {"id": "052-prepare-atlas-forced-photometry"},
            "missingPrerequisites": [
                "openstar.capability.current-atlas-forced-photometry-adapter"
            ],
        }
        inv = Investigation(inv.id, inv.workflow_id, inv.workflow_version, inv.status,
            inv.created_at, inv.updated_at, metadata, inv.stages)
        store = InvestigationStore(self.root / "migration"); store.save(inv)
        with mock.patch.dict(os.environ, {}, clear=True):
            migrated = repair_obsolete_terminal_wait(store, inv)
            again = repair_obsolete_terminal_wait(store, migrated)
        self.assertEqual(["openstar.capability.atlas-forced-photometry-credentials"],
                         migrated.metadata["controlState"]["missingPrerequisites"])
        self.assertEqual("BLOCKED", migrated.status)
        self.assertEqual(migrated.metadata["controlState"], again.metadata["controlState"])
        self.assertEqual(len(inv.stages), len(migrated.stages))

    def test_post_atlas_targeted_observation_is_durable_and_not_completed(self):
        stage = InvestigationStage("053-interpret-atlas-forced-photometry",
            "openstar.tess.atlas-forced-photometry.interpret", "COMPLETE", None, {},
            result={"recommendedNextTest": atlas.HISTORICAL_TRIGGER,
                    "physicalMechanismResolved": False}, stop=True)
        inv = Investigation("targeted", "openstar.workflow.tess-investigation.v1", "20.2",
            "BLOCKED", "now", "now", {"datasetID": "targeted", "controlState": {
                "schedulerAction": "WAIT_FOR_PREREQUISITES"}}, (stage,))
        branch = plan_tess_branches(inv, InvestigationTarget(
            "targeted", "targeted", inv.workflow_id, inv.workflow_version))[0]
        self.assertEqual("openstar.tess.targeted-observation-planning.generate",
                         branch.experiment.handler_id)
        self.assertEqual(("openstar.capability.current-targeted-observation-planning-adapter",),
                         branch.required_stage_ids)
        store = InvestigationStore(self.root / "targeted"); store.save(inv)
        repaired = repair_obsolete_terminal_wait(store, inv)
        self.assertEqual("BLOCKED", repaired.status)
        self.assertEqual("WAIT_FOR_PREREQUISITES",
                         repaired.metadata["controlState"]["schedulerAction"])
        self.assertEqual(
            ("openstar.capability.current-targeted-observation-planning-adapter",),
            repaired.metadata["controlState"]["branchAssessments"][0][
                "missing_stage_ids"
            ])

    def _handler_evidence(self, investigation_id):
        return Investigation(investigation_id,
            "openstar.workflow.tess-investigation.v1", "20.2", "RUNNING",
            "now", "now", {}, (
                InvestigationStage("001", "openstar.tess.prepare-target", "COMPLETE",
                    None, {}, result={"sourceProjectID": "p", "datasetID": "d"}),
                InvestigationStage("043", "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                    "COMPLETE", None, {}, result={"sourcePair": self.pair}),
                InvestigationStage("051-interpret-des-dr2-se-local-forced-photometry",
                    "openstar.tess.des-dr2-se-local-forced-photometry.interpret",
                    "COMPLETE", None, {}, result={"sourcePair": self.pair,
                        "recommendedNextTest": atlas.CURRENT_TRIGGER,
                        "physicalMechanismResolved": False}),
            ))

    def test_registered_current_handler_submits_and_quiesces(self):
        for distributed in (True, False):
            store = InvestigationStore(self.root / f"handler-{distributed}")
            inv = self._handler_evidence(f"handler-{distributed}"); store.save(inv)
            project = self.root / f"atlas-project-{distributed}.json"
            if distributed: project.write_text("{}", encoding="utf-8")
            spec = {"externalDependencyID": "atlas:inv:052",
                    "externalJobIDs": ["target", "counterpart"]}
            coordinator = mock.Mock()
            coordinator.run_project.return_value = SimpleNamespace(
                status={"datasets": []}, node_contributions={}, project_id="generic")
            summary = {"recommendedNextTest": atlas.HISTORICAL_TRIGGER,
                       "physicalMechanismResolved": False}
            workflow = _workflow_module()
            with mock.patch.dict(os.environ, {"OPENSTAR_ATLAS_API_TOKEN": "token"}, clear=True), \
                    mock.patch.object(workflow, "submit_atlas_forced_photometry_jobs",
                                      return_value=spec) as builder, \
                    mock.patch.object(workflow, "interpret_atlas_forced_photometry_project",
                                      return_value=summary):
                completed = _build_engine(store, coordinator, poll_interval=0, timeout=1).run(
                    inv, StageRequest("052-prepare-atlas-forced-photometry",
                        "openstar.tess.atlas-forced-photometry.prepare", {},
                        "051-interpret-des-dr2-se-local-forced-photometry"),
                    software_id="test", software_version="1")
            self.assertEqual(self.pair,
                builder.call_args.kwargs["external_high_resolution_summary"]["sourcePair"])
            self.assertFalse(any("external-high-resolution" in item.handler_id
                                 for item in inv.stages))
            handlers = [item.handler_id for item in completed.stages[len(inv.stages):]]
            self.assertEqual(["openstar.tess.atlas-forced-photometry.prepare"], handlers)
            coordinator.run_project.assert_not_called()
            self.assertEqual("QUIESCENT_AWAITING_DATA", completed.status)

    def test_registered_prepare_failure_classification_is_narrow(self):
        for error, expected, raised in (
            (atlas.ATLASArchiveUnavailable("outage"), "TRANSIENT_INFRASTRUCTURE",
             RetryableExecutionError),
            (RuntimeError("bug"), "NON_RETRYABLE", RuntimeError),
        ):
            store = InvestigationStore(self.root / expected)
            inv = self._handler_evidence(expected); store.save(inv)
            workflow = _workflow_module()
            with mock.patch.object(workflow, "submit_atlas_forced_photometry_jobs",
                                   side_effect=error):
                with self.assertRaises(raised):
                    _build_engine(store, mock.Mock(), poll_interval=0, timeout=1).run(
                        inv, StageRequest("052-prepare-atlas-forced-photometry",
                            "openstar.tess.atlas-forced-photometry.prepare", {}, "051"),
                        software_id="test", software_version="1")
            self.assertEqual(expected, store.load(inv.id).stages[-1].failure_classification)

    def test_registered_persisted_prepare_resumes_without_requery(self):
        store = InvestigationStore(self.root / "persisted-handler")
        preparation = {"preparedSeries": [], "sourceRecords": [{"rawPath": "/raw/immutable"}],
                       "workloadID": "openstar.lomb-scargle.v1"}
        next_stage = StageRequest("053-exact-interpret",
            "openstar.tess.atlas-forced-photometry.interpret",
            {"distributedRunExpected": False, "sentinel": 7}, "052")
        stage = InvestigationStage("052", "openstar.tess.atlas-forced-photometry.prepare",
            "COMPLETE", None, {}, result=preparation, next_stage=asdict(next_stage))
        inv = Investigation("persisted-handler", "openstar.workflow.tess-investigation.v1",
            "20.2", "COMPLETE", "now", "now", {}, (stage,)); store.save(inv)
        request = plan_tess_branches(inv, InvestigationTarget(
            inv.id, inv.id, inv.workflow_id, inv.workflow_version))[0].experiment
        self.assertEqual(next_stage, request)
        workflow = _workflow_module()
        with mock.patch.object(workflow, "build_atlas_forced_photometry_project") as builder, \
                mock.patch.object(workflow, "interpret_atlas_forced_photometry_project",
                    return_value={"recommendedNextTest": atlas.HISTORICAL_TRIGGER,
                                  "physicalMechanismResolved": False}):
            completed = _build_engine(store, mock.Mock(), poll_interval=0, timeout=1).run(
                inv, request, software_id="test", software_version="1")
        builder.assert_not_called()
        self.assertEqual("053-exact-interpret", completed.stages[-1].id)
        self.assertEqual("/raw/immutable",
                         completed.stages[0].result["sourceRecords"][0]["rawPath"])

    def test_persisted_atlas_continuations_reuse_exact_next_stage(self):
        for handler in ("openstar.tess.atlas-forced-photometry.run",
                        "openstar.tess.atlas-forced-photometry.interpret"):
            raw = {"id": "053-custom", "handler_id": handler,
                   "parameters": {"sentinel": 7}, "triggered_by_stage_id": "052"}
            stage = InvestigationStage("052", "openstar.tess.atlas-forced-photometry.prepare",
                "COMPLETE", None, {}, result={"sourceRecords": [{"rawPath": "raw"}]},
                next_stage=raw)
            inv = Investigation("persisted", "openstar.workflow.tess-investigation.v1", "20.2",
                                "COMPLETE", "now", "now", {}, (stage,))
            branch = plan_tess_branches(inv, InvestigationTarget(
                "persisted", "persisted", inv.workflow_id, inv.workflow_version))[0]
            self.assertEqual(raw, asdict(branch.experiment))


if __name__ == "__main__":
    unittest.main()
