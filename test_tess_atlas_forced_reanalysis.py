import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openstar_investigation import Investigation, InvestigationStage, InvestigationStore, sha256_json
from openstar_targets import InvestigationTarget
from openstar_workflow import StageRequest
from workflows.tess import tess_atlas_forced_photometry as atlas
from workflows.tess import tess_atlas_forced_reanalysis as reanalysis
from workflows.tess.tess_autonomy import plan_tess_branches, repair_obsolete_terminal_wait
from workflows.tess.tess_investigation import build_engine


class CurrentATLASSignedReanalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pair = {
            "version": "openstar.current-source-pair.v1",
            "target": {"sourceRole": "target-control", "gaiaDR3SourceID": 4676067719930656640,
                       "raDeg": 10.0, "decDeg": -20.0},
            "counterpart": {"sourceRole": "catalog-counterpart", "gaiaDR3SourceID": 4676068475843651200,
                            "raDeg": 10.02, "decDeg": -20.0},
        }
        self.search = {"minimumFrequency": 0.2, "maximumFrequency": 0.4,
                       "frequencyStep": 0.01, "totalFrequencies": 21,
                       "frequenciesPerWorkUnit": 7}
        sources, separation = atlas._frozen_sources({"sourcePair": self.pair})
        records = []
        for source in sources:
            path = self.root / (source["sourceRole"] + ".txt")
            lines = ["# MJD uJy duJy F err chi/N"]
            for night in range(40):
                for band in ("c", "o"):
                    lines.append(f"{59500 + night * 2} {(night % 5) - 2} 10 {band} 0 1")
            path.write_text("\n".join(lines) + "\n")
            records.append({**source, "rawPath": str(path),
                            "rawSha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.summary = atlas.interpret_atlas_forced_photometry_project(
            project_status=None, preparation={"preparedSeries": [], "sourceRecords": records,
                "sourcePair": self.pair, "sourceDefinitions": sources,
                "gaiaPairSeparationArcsec": separation, "frequencySearch": self.search,
                "workloadID": "openstar.lomb-scargle.v1"})

    def build(self, summary=None):
        return reanalysis.build_atlas_forced_photometry_reanalysis_project(
            source_project_id="p", source_dataset_id="d", investigation_id="signed",
            atlas_v20_24_summary=self.summary if summary is None else summary,
            output_dir=self.root / "output")

    def investigation(self):
        prepared = InvestigationStage("001-prepare-target", "openstar.tess.prepare-target",
            "COMPLETE", None, {}, result={"sourceProjectID": "p", "datasetID": "d", "ticID": 123})
        interpreted = InvestigationStage("057-interpret-atlas-forced-photometry",
            "openstar.tess.atlas-forced-photometry.interpret", "COMPLETE", None, {},
            result=copy.deepcopy(self.summary), stop=True)
        return Investigation("signed", "openstar.workflow.tess-investigation.v1", "20.2",
            "BLOCKED", "now", "now", {"datasetID": "d", "controlState": {
                "schedulerAction": "WAIT_FOR_PREREQUISITES",
                "missingPrerequisites": ["openstar.capability.current-atlas-signed-reanalysis-adapter"]}},
            (prepared, interpreted))

    def test_exact_blocked_handoff_reopens_once_and_uses_registered_prepare(self):
        inv = self.investigation()
        store = InvestigationStore(self.root / "store")
        store.save(inv)
        before = copy.deepcopy(inv.stages)
        target = InvestigationTarget("d", inv.id, inv.workflow_id, inv.workflow_version)
        branch = plan_tess_branches(inv, target)[0]
        self.assertEqual((), branch.required_stage_ids)
        self.assertEqual("058-prepare-atlas-forced-photometry-reanalysis", branch.experiment.id)
        self.assertEqual(inv.stages[-1].id, branch.experiment.triggered_by_stage_id)
        repaired = repair_obsolete_terminal_wait(store, inv)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual("RUN_EXPERIMENT", repaired.metadata["controlState"]["schedulerAction"])
        self.assertEqual(before, repaired.stages)
        self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))
        engine = build_engine(store, mock.Mock(), poll_interval=0, timeout=1)
        engine.chain_stages = False
        with mock.patch.object(atlas, "_json_request", side_effect=AssertionError("no network")), \
                mock.patch.object(atlas, "_text_request", side_effect=AssertionError("no download")):
            completed = engine.run(repaired, branch.experiment, software_id="test", software_version="1")
        stage = completed.stages[-1]
        self.assertEqual("COMPLETE", stage.status, stage.error)
        self.assertEqual(before, completed.stages[:-1])
        self.assertEqual(4, len(stage.result["preparedSeries"]))
        self.assertEqual(sha256_json(self.summary), stage.provenance.input_hashes["atlasV20_24Interpretation"])
        self.assertEqual("openstar.tess.atlas-forced-photometry-reanalysis.run", stage.next_stage["handler_id"])
        from workflows.tess.tess_autonomy import _awaiting_atlas_signed_reanalysis_adapter
        self.assertFalse(_awaiting_atlas_signed_reanalysis_adapter(completed))

    def test_signed_rows_reused_with_frozen_grid_and_exact_source_ids(self):
        original = {str(Path(item["rawPath"]).resolve()): Path(item["rawPath"]).read_bytes() for item in self.summary["sourceRecords"]}
        result = self.build()
        self.assertEqual(12, result["totalWorkUnits"])
        self.assertFalse(result["rawArchiveRequeried"])
        self.assertFalse(result["individualDetectionThresholdApplied"])
        self.assertTrue(result["signedForcedFluxRetained"])
        self.assertEqual(self.pair, result["sourcePair"])
        for record in result["sourceRecords"]:
            self.assertEqual(80, record["acceptedSignedRowCount"])
            self.assertEqual(original[record["rawPath"]], Path(record["rawPath"]).read_bytes())
            self.assertEqual(hashlib.sha256(original[record["rawPath"]]).hexdigest(), record["rawSha256"])
        for series in result["preparedSeries"]:
            dataset = json.loads(Path(series["datasetPath"]).read_text())
            self.assertEqual(self.search, dataset["frequencySearch"])
            self.assertEqual(series["gaiaDR3SourceID"], dataset["science"]["gaiaDR3SourceID"])
            self.assertLess(min(dataset["flux"]), 0)
            self.assertGreater(max(dataset["flux"]), 0)

    def test_incomplete_or_mismatched_current_evidence_remains_gated(self):
        variants = []
        for key in ("sourcePair", "sourceDefinitions", "distributedValidation", "useReducedTargetImages"):
            broken = copy.deepcopy(self.summary)
            broken.pop(key)
            variants.append(broken)
        for key, value in (("gaiaDR3SourceID", 999), ("raDeg", 11), ("rawSha256", "")):
            broken = copy.deepcopy(self.summary)
            broken["sourceRecords"][0][key] = value
            variants.append(broken)
        broken = copy.deepcopy(self.summary)
        broken["distributedValidation"]["frequencySearch"]["frequencyStep"] = float("nan")
        variants.append(broken)
        for broken in variants:
            with self.subTest(summary=broken):
                self.assertFalse(reanalysis.current_atlas_signed_reanalysis_ready(broken))
                inv = self.investigation()
                inv.stages[-1].result.clear()
                inv.stages[-1].result.update(broken)
                branch = plan_tess_branches(inv, InvestigationTarget("d", inv.id, inv.workflow_id, inv.workflow_version))[0]
                self.assertEqual(("openstar.capability.current-atlas-signed-reanalysis-adapter",), branch.required_stage_ids)

    def test_missing_or_changed_raw_file_fails_before_output(self):
        path = Path(self.summary["sourceRecords"][-1]["rawPath"])
        path.write_text("changed\n")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            self.build()
        self.assertFalse((self.root / "output").exists())
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "artifact is missing"):
            self.build()
        self.assertFalse((self.root / "output").exists())

    def test_no_usable_rows_still_interprets_without_distributed_work(self):
        for record in self.summary["sourceRecords"]:
            path = Path(record["rawPath"])
            path.write_text("# MJD uJy duJy F err chi/N\n59500 -1 10 c 1 1\n")
            record["rawSha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        preparation = self.build()
        self.assertFalse(preparation["available"])
        self.assertEqual(0, preparation["totalWorkUnits"])
        result = reanalysis.interpret_atlas_forced_photometry_reanalysis_project(
            project_status=None, preparation=preparation)
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertFalse(result["claimLevelChanged"])
        self.assertFalse(result["targetControl"]["sourceSupported"])
        self.assertFalse(result["catalogCounterpartEvidence"]["sourceSupported"])

        inv = self.investigation()
        store = InvestigationStore(self.root / "empty-store")
        store.save(inv)
        repaired = repair_obsolete_terminal_wait(store, inv)
        engine = build_engine(store, mock.Mock(), poll_interval=0, timeout=1)
        engine.chain_stages = False
        request = StageRequest(**repaired.metadata["controlState"]["selectedExperiment"])
        prepared = engine.run(repaired, request, software_id="test", software_version="1")
        self.assertEqual("COMPLETE", prepared.stages[-1].status, prepared.stages[-1].error)
        request = StageRequest(**prepared.stages[-1].next_stage)
        self.assertEqual("openstar.tess.atlas-forced-photometry-reanalysis.interpret", request.handler_id)
        interpreted = engine.run(prepared, request, software_id="test", software_version="1")
        stage = interpreted.stages[-1]
        self.assertEqual("COMPLETE", stage.status, stage.error)
        self.assertEqual("openstar.tess.finalize", stage.next_stage["handler_id"])
        self.assertEqual({"outputSuffix": "v20.25"}, stage.next_stage["parameters"])


if __name__ == "__main__":
    unittest.main()
