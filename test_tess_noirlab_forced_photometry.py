import tempfile
import sys
import types
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openstar_investigation import Investigation, InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTarget
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

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


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
        prepared = [{"datasetID": "cg", "sourceRole": "catalog-counterpart", "band": "g"},
                    {"datasetID": "cr", "sourceRole": "catalog-counterpart", "band": "r"}]
        datasets = [{"datasetID": item["datasetID"], "periodStatus": "RELIABLE",
                     "coverageComplete": True, "candidateFrequency": 1.2,
                     "candidatePeakProminenceRatio": 4.0} for item in prepared]
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


if __name__ == "__main__":
    unittest.main()
