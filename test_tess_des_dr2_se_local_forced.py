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
from openstar_targets import InvestigationTarget
from openstar_workflow import RetryableExecutionError, StageRequest
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("numpy")
    stub.integer = int
    stub.floating = float
    stub.float64 = float
    stub.float32 = float
    stub.asarray = lambda values, dtype=None: list(values)
    stub.linalg = SimpleNamespace(LinAlgError=type("LinAlgError", (Exception,), {}))
    sys.modules["numpy"] = stub
    _installed_numpy_stub = True
else:
    _installed_numpy_stub = False

from workflows.tess import tess_des_dr2_se_local_forced as des
from workflows.tess import tess_noirlab_forced_photometry as noir
from workflows.tess.tess_autonomy import plan_tess_branches, repair_obsolete_terminal_wait
from workflows.tess.tess_investigation import build_engine

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


class CurrentDESLocalForcedPhotometryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.pair = {"version": "openstar.current-source-pair.v1",
            "target": {"sourceRole": "target-control", "gaiaDR3SourceID": 101,
                       "raDeg": 10.0, "decDeg": -20.0},
            "counterpart": {"sourceRole": "catalog-counterpart", "gaiaDR3SourceID": 202,
                            "raDeg": 10.01, "decDeg": -20.0},
            "separationArcsec": 999.0}
        self.search = {"totalFrequencies": 100, "frequenciesPerWorkUnit": 10,
                       "minimumFrequency": 1.0, "maximumFrequency": 2.0}

    def _build(self, rows=()):
        with mock.patch.object(des, "_query_sia", return_value=list(rows)):
            return des.build_des_dr2_se_local_forced_project(
                source_project_id="project", source_dataset_id="dataset",
                external_high_resolution_summary={"sourcePair": self.pair},
                noirlab_image_summary={"recommendedNextTest": des.CURRENT_TRIGGER,
                                       "frequencySearch": self.search},
                output_dir=self.root, investigation_id="generic")

    def test_current_pair_coordinates_define_geometry_and_guard_is_unchanged(self):
        sources, separation = des._source_pair({"sourcePair": self.pair})
        expected = des._angular_separation_arcsec(10, -20, 10.01, -20)
        self.assertAlmostEqual(expected, separation)
        self.assertNotEqual(999, separation)
        self.assertEqual([101, 202], [x["gaiaDR3SourceID"] for x in sources])
        self.assertEqual(15.0, des.MIN_INDEPENDENT_LOCAL_SEPARATION_ARCSEC)

    def test_current_trigger_direct_search_and_generic_worker(self):
        result = self._build()
        self.assertEqual(self.search, result["frequencySearch"])
        self.assertEqual("openstar.lomb-scargle.v1", result["workloadID"])
        self.assertFalse(result["tessDriftExtrapolated"])

    def test_scientific_empty_outcomes_continue_to_atlas(self):
        for preparation in ({"candidateExposures": 0},
                            {"candidateExposures": 5, "sourceSuccesses": {}},
                            {"candidateExposures": 5, "sourceSuccesses": {"target-control": 2}}):
            preparation.update({"preparedSeries": [], "workloadID": "openstar.lomb-scargle.v1"})
            summary = des.interpret_des_dr2_se_local_forced_project(
                project_status=None, preparation=preparation)
            self.assertEqual(des.NEXT_ARCHIVE_TEST, summary["recommendedNextTest"])
            self.assertFalse(summary["physicalMechanismResolved"])

    def _interpret(self, supported_role, include_control):
        prepared = []
        datasets = []
        for role, prefix in (("target-control", "t"), ("catalog-counterpart", "c")):
            for band in ("g", "r"):
                item = {"datasetID": prefix + band, "sourceRole": role, "band": band}
                prepared.append(item)
                if role == supported_role or include_control:
                    supported = role == supported_role
                    datasets.append({"datasetID": item["datasetID"], "coverageComplete": True,
                        "periodStatus": "RELIABLE" if supported else "LOW_CONFIDENCE",
                        "candidateFrequency": 1.2 if supported else None,
                        "candidatePeakProminenceRatio": 4.0 if supported else 1.0})
        minimal = SimpleNamespace(float64=float, asarray=lambda x, dtype=None: list(x),
            median=lambda x: sorted(x)[len(x)//2])
        with mock.patch.object(des, "np", minimal), mock.patch.object(noir, "np", minimal):
            return des.interpret_des_dr2_se_local_forced_project(
                project_status={"datasets": datasets}, preparation={"preparedSeries": prepared,
                    "candidateExposures": 20, "sourceSuccesses": {},
                    "workloadID": "openstar.lomb-scargle.v1"})

    def test_missing_controls_block_decisive_routes(self):
        for role in ("target-control", "catalog-counterpart"):
            self.assertEqual(des.NEXT_ARCHIVE_TEST,
                self._interpret(role, False)["recommendedNextTest"])

    def test_usable_nonrecurrent_controls_allow_decisive_routes(self):
        self.assertEqual("TARGET_INTRINSIC_RESIDUAL_MODELING",
            self._interpret("target-control", True)["recommendedNextTest"])
        self.assertEqual("TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL",
            self._interpret("catalog-counterpart", True)["recommendedNextTest"])

    def test_exact_049_blocked_reopens_only_050_des(self):
        prior = tuple(InvestigationStage(f"04{i}-{name}", handler, "COMPLETE", None, {}, result={})
            for i, (name, handler) in enumerate((
                ("gaia", "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret"),
                ("skymapper", "openstar.tess.skymapper-resolved-counterpart-photometry.interpret"),
                ("nsc", "openstar.tess.nsc-resolved-photometry.interpret")), start=3))
        stage = InvestigationStage("049-interpret-noirlab-image-forced-photometry",
            "openstar.tess.noirlab-image-forced-photometry.interpret", "COMPLETE", None, {},
            result={"recommendedNextTest": des.CURRENT_TRIGGER,
                    "physicalMechanismResolved": False,
                    "sourcePair": self.pair}, stop=True)
        inv = Investigation("generic", "openstar.workflow.tess-investigation.v1", "20.2",
            "BLOCKED", "now", "now", {"datasetID": "generic", "controlState": {
                "schedulerAction": "WAIT_FOR_PREREQUISITES"}}, prior + (stage,))
        target = InvestigationTarget("generic", "generic", inv.workflow_id, inv.workflow_version)
        branch = plan_tess_branches(inv, target)[0]
        self.assertEqual("050-prepare-des-dr2-single-epoch-local-forced-photometry", branch.experiment.id)
        self.assertEqual("openstar.tess.des-dr2-se-local-forced-photometry.prepare",
                         branch.experiment.handler_id)
        self.assertEqual((), branch.required_stage_ids)
        store = InvestigationStore(self.root / "repair"); store.save(inv)
        repaired = repair_obsolete_terminal_wait(store, inv)
        self.assertEqual(branch.experiment.id,
            repaired.metadata["controlState"]["selectedExperiment"]["id"])
        self.assertEqual([x.id for x in prior + (stage,)], [x.id for x in repaired.stages])

    def test_persisted_des_continuations_reuse_exact_stage(self):
        for handler in ("openstar.tess.des-dr2-se-local-forced-photometry.run",
                        "openstar.tess.des-dr2-se-local-forced-photometry.interpret"):
            expected = StageRequest("051-custom", handler, {"sentinel": 7}, "050-prepare")
            stage = InvestigationStage("050-prepare",
                "openstar.tess.des-dr2-se-local-forced-photometry.prepare", "COMPLETE",
                None, {}, result={}, next_stage=asdict(expected))
            inv = Investigation("persisted", "openstar.workflow.tess-investigation.v1", "20.2",
                "COMPLETE", "now", "now", {}, (stage,))
            target = InvestigationTarget("persisted", "persisted", inv.workflow_id, inv.workflow_version)
            self.assertEqual(expected, plan_tess_branches(inv, target)[0].experiment)

    def test_transient_classification_is_narrow(self):
        for code, expected in ((408, True), (425, True), (429, True), (500, True), (400, False)):
            self.assertEqual(expected, des._retryable_service_error(
                urllib.error.HTTPError("url", code, "x", {}, None)))
        self.assertFalse(des._retryable_service_error(RuntimeError("bug")))
        self.assertFalse(des._retryable_service_error(ImportError("astropy")))

    def test_isolated_transport_continues_and_unknown_errors_propagate(self):
        rows = [{"access_url": "https://example/one", "mjd_obs": "1", "obs_bandpass": "g",
                 "prodtype": "image", "obs_id": "one"}]
        with mock.patch.object(des, "_download_fits_cutout", side_effect=[TimeoutError("one"),
                des.DESImageQualityRejected("saturated")]):
            result = self._build(rows)
        self.assertEqual(2, sum(result["sourceAttempts"].values()))
        with mock.patch.object(des, "_download_fits_cutout", side_effect=RuntimeError("bug")):
            with self.assertRaises(RuntimeError): self._build(rows)
        with mock.patch.object(des, "_download_fits_cutout", side_effect=ImportError("astropy")):
            with self.assertRaises(ImportError): self._build(rows)

    def test_malformed_fits_is_scientific_rejection_and_processing_continues(self):
        rows = [{"access_url": "https://example/one", "mjd_obs": "1",
                 "obs_bandpass": "g", "prodtype": "image", "obs_id": "one"}]
        with mock.patch.object(des, "_download_fits_cutout", return_value=b"not-fits"), \
                mock.patch.object(des, "_image_hdu_from_bytes",
                                  side_effect=OSError("corrupt FITS")):
            result = self._build(rows)
        self.assertEqual(2, sum(result["sourceAttempts"].values()))
        self.assertEqual(2, result["failureReasons"][
            "target-control:invalid-local-image-product"] + result["failureReasons"][
            "catalog-counterpart:invalid-local-image-product"])

    def test_candidate_linalg_error_is_scientific_fit_rejection(self):
        original = getattr(des.np, "linalg", None)
        if original is None:
            des.np.linalg = SimpleNamespace(
                LinAlgError=type("LinAlgError", (Exception,), {})
            )
        try:
            with mock.patch.object(des, "_scaled_linear_fit",
                                   side_effect=des.np.linalg.LinAlgError("singular")):
                self.assertIsNone(des._candidate_scaled_linear_fit(object(), object()))
            with mock.patch.object(des, "_scaled_linear_fit",
                                   side_effect=TypeError("programming bug")):
                with self.assertRaises(TypeError):
                    des._candidate_scaled_linear_fit(object(), object())
        finally:
            if original is None:
                del des.np.linalg
            else:
                des.np.linalg = original

    def test_broad_transport_outage_raises_archive_unavailable(self):
        rows = [{"access_url": f"https://example/{i}", "mjd_obs": str(i),
                 "obs_bandpass": "g", "prodtype": "image", "obs_id": str(i)} for i in range(3)]
        with mock.patch.object(des, "_download_fits_cutout", side_effect=TimeoutError("outage")):
            with self.assertRaises(des.DESArchiveUnavailable): self._build(rows)

    def test_registered_transient_prepare_uses_generic_retry(self):
        store = InvestigationStore(self.root / "engine")
        stages = (InvestigationStage("001", "openstar.tess.prepare-target", "COMPLETE", None, {},
                    result={"sourceProjectID": "p", "datasetID": "d"}),
            InvestigationStage("043", "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                    "COMPLETE", None, {}, result={"sourcePair": self.pair}),
            InvestigationStage("049", "openstar.tess.noirlab-image-forced-photometry.interpret",
                    "COMPLETE", None, {}, result={"recommendedNextTest": des.CURRENT_TRIGGER,
                                                  "sourcePair": self.pair}))
        inv = Investigation("engine", "openstar.workflow.tess-investigation.v1", "20.2",
                            "RUNNING", "now", "now", {}, stages); store.save(inv)
        with mock.patch("workflows.tess.tess_investigation.build_des_dr2_se_local_forced_project",
                        side_effect=des.DESArchiveUnavailable("outage")):
            with self.assertRaises(RetryableExecutionError):
                build_engine(store, mock.Mock(), poll_interval=0, timeout=1).run(inv,
                    StageRequest("050-prepare-des-dr2-single-epoch-local-forced-photometry",
                    "openstar.tess.des-dr2-se-local-forced-photometry.prepare", {}, "049"),
                    software_id="test", software_version="1")
        self.assertEqual("TRANSIENT_INFRASTRUCTURE", store.load(inv.id).stages[-1].failure_classification)

    def test_registered_prepare_run_interpret_and_direct_interpret(self):
        for distributed in (True, False):
            store = InvestigationStore(self.root / f"chain-{distributed}")
            gaia_pair = {
                **self.pair,
                "target": {**self.pair["target"], "gaiaDR3SourceID": 303},
            }
            stages = (
                InvestigationStage("001", "openstar.tess.prepare-target", "COMPLETE", None, {},
                    result={"sourceProjectID": "p", "datasetID": "d"}),
                InvestigationStage("043", "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
                    "COMPLETE", None, {}, result={"sourcePair": gaia_pair}),
                InvestigationStage("049", "openstar.tess.noirlab-image-forced-photometry.interpret",
                    "COMPLETE", None, {}, result={"recommendedNextTest": des.CURRENT_TRIGGER,
                                                  "sourcePair": self.pair}),
            )
            inv = Investigation(f"chain-{distributed}",
                "openstar.workflow.tess-investigation.v1", "20.2", "RUNNING",
                "now", "now", {}, stages)
            store.save(inv)
            project = self.root / f"project-{distributed}.json"
            if distributed:
                project.write_text("{}", encoding="utf-8")
            spec = {"available": distributed,
                "projectPath": str(project) if distributed else None,
                "preparedSeries": [], "candidateExposures": 0,
                "workloadID": "openstar.lomb-scargle.v1"}
            coordinator = mock.Mock()
            coordinator.run_project.return_value = SimpleNamespace(
                status={"datasets": []}, node_contributions={}, project_id="generic")
            summary = {"recommendedNextTest": des.NEXT_ARCHIVE_TEST,
                       "physicalMechanismResolved": False}
            with mock.patch(
                "workflows.tess.tess_investigation.build_des_dr2_se_local_forced_project",
                return_value=spec) as prepare_des, mock.patch(
                "workflows.tess.tess_investigation.interpret_des_dr2_se_local_forced_project",
                return_value=summary):
                completed = build_engine(store, coordinator, poll_interval=0, timeout=1).run(
                    inv, StageRequest(
                        "050-prepare-des-dr2-single-epoch-local-forced-photometry",
                        "openstar.tess.des-dr2-se-local-forced-photometry.prepare", {}, "049"),
                    software_id="test", software_version="1")
            source_evidence = prepare_des.call_args.kwargs[
                "external_high_resolution_summary"
            ]
            self.assertEqual(self.pair, source_evidence["sourcePair"])
            self.assertFalse(any("external-high-resolution" in stage.handler_id
                                 for stage in stages))
            handlers = [stage.handler_id for stage in completed.stages[3:]]
            expected = ["openstar.tess.des-dr2-se-local-forced-photometry.prepare"]
            if distributed:
                expected.append("openstar.tess.des-dr2-se-local-forced-photometry.run")
            expected.append("openstar.tess.des-dr2-se-local-forced-photometry.interpret")
            self.assertEqual(expected, handlers)
            self.assertEqual("BLOCKED", completed.status)
            self.assertEqual(distributed, bool(coordinator.run_project.call_count))

    def test_nondecisive_interpretation_has_durable_atlas_boundary(self):
        stage = InvestigationStage("052-interpret-des",
            "openstar.tess.des-dr2-se-local-forced-photometry.interpret", "COMPLETE", None, {},
            result={"recommendedNextTest": des.NEXT_ARCHIVE_TEST,
                    "physicalMechanismResolved": False}, stop=True)
        inv = Investigation("atlas", "openstar.workflow.tess-investigation.v1", "20.2",
                            "BLOCKED", "now", "now", {}, (stage,))
        target = InvestigationTarget("atlas", "atlas", inv.workflow_id, inv.workflow_version)
        branch = plan_tess_branches(inv, target)[0]
        self.assertEqual("openstar.tess.atlas-forced-photometry.prepare", branch.experiment.handler_id)
        self.assertEqual(("openstar.capability.current-atlas-forced-photometry-adapter",),
                         branch.required_stage_ids)


if __name__ == "__main__": unittest.main()
