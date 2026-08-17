import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTarget
from openstar_workflow import RetryableExecutionError, StageRequest
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_autonomy import plan_tess_branches
from workflows.tess import tess_gaia_counterpart as gaia


class _Array(list):
    def tolist(self):
        return list(self)


class _NumpyStub:
    float32 = object()

    @staticmethod
    def asarray(values, dtype=None):
        return _Array(values)


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


if __name__ == "__main__":
    unittest.main()
