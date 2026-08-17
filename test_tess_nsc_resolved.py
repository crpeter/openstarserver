from __future__ import annotations

import tempfile
import sys
import types
import unittest
import urllib.error
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from openstar_investigation import Investigation, InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTarget
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

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


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
        self.assertEqual(("openstar.capability.current-noirlab-source-pair-adapter",),
                         branches[0].required_stage_ids)
        self.assertEqual("openstar.tess.noirlab-image-forced-photometry.prepare",
                         branches[0].experiment.handler_id)


if __name__ == "__main__":
    unittest.main()
