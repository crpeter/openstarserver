import tempfile
import sys
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("numpy")
    stub.integer = int
    stub.floating = float
    sys.modules["numpy"] = stub

from workflows.tess import tess_skymapper_resolved as sky
from openstar_investigation import Investigation, InvestigationStage
from openstar_targets import InvestigationTarget
from workflows.tess.tess_autonomy import plan_tess_branches


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

    @mock.patch.object(sky, "_query_master_matches", return_value=[])
    def test_only_generic_lomb_scargle_workload_is_declared(self, _query):
        self.assertEqual("openstar.lomb-scargle.v1", self.build()["workloadID"])


if __name__ == "__main__":
    unittest.main()
