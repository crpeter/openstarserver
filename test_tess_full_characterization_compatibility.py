import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from openstar_investigation import (
    ArtifactReference, InvestigationStage, InvestigationStore, StageProvenance,
    sha256_file, sha256_json,
)
from workflows.tess.tess_autonomy import WORKFLOW_ID, repair_obsolete_terminal_wait


class FullCharacterizationCompatibilityTests(unittest.TestCase):
    IDS = ("001-prepare-target", "002-primary-distributed-search",
           "003-catalog-identity", "004-hypotheses", "005-planner", "006-finalize")
    HANDLERS = (
        "openstar.tess.prepare-target", "openstar.tess.primary-project.run",
        "openstar.tess.catalog-identity", "openstar.tess.hypotheses",
        "openstar.tess.planner", "openstar.tess.finalize",
    )

    def _boundary(self, root: Path):
        source = root / "source"
        source.mkdir()
        dataset_path = source / "dataset.json"
        dataset_path.write_text('{"science":{"role":"blind"}}\n')
        entry = {"id": "generic-dataset", "path": str(dataset_path), "role": "blind",
                 "datasetSHA256": sha256_file(dataset_path)}
        project_path = source / "project.json"
        project = {"id": "generic-project", "datasets": [entry], "preparer": {
            "preparerID": "openstar.tess-known-target-blind-benchmark-preparer",
            "schemaVersion": 1, "ownedFiles": ["dataset.json", "project.json"],
            "projectID": "generic-project"}}
        project_path.write_text(json.dumps(project) + "\n")
        artifact_path = root / "frozen-primary-manifest.json"
        artifact_path.write_text('{"frozen":true}\n')
        results = [
            {"sourceProjectPath": str(project_path), "datasetPath": str(dataset_path),
             "sourceDatasetEntry": entry},
            {"datasets": [{"candidatePeriodDays": 2.0}]},
            {"identityResolved": True},
            {"observedPeriodDays": 2.0, "bestCatalogMatch": {"source": "catalog"}},
            {"action": "STOP", "reason": "catalog-period-match"},
            {"claim": "KNOWN_PERIOD_RECOVERED"},
        ]
        store = InvestigationStore(root / "investigations")
        investigation = store.create("generic-investigation", WORKFLOW_ID, "20.2")
        fixture = {"root": root, "store": store, "investigation": investigation,
                   "project_path": project_path, "dataset_path": dataset_path,
                   "artifact_path": artifact_path, "project": project, "results": results,
                   "ids": list(self.IDS), "handlers": list(self.HANDLERS),
                   "statuses": ["COMPLETE"] * 6,
                   "triggers": [None, *self.IDS[:-1]], "stop": True,
                   "investigation_status": "COMPLETE",
                   "control": {"schedulerAction": "INVESTIGATION_COMPLETE",
                               "selectedExperiment": None},
                   "source_project_hash": None, "source_dataset_hash": None,
                   "artifacts": [ArtifactReference(str(artifact_path),
                                                    sha256_file(artifact_path),
                                                    "application/json")]}
        return self._persist(fixture)

    def _persist(self, fixture):
        project_path = fixture["project_path"]
        project_path.write_text(json.dumps(fixture["project"]) + "\n")
        results = fixture["results"]
        hashes = [
            {"sourceProjectManifest": fixture["source_project_hash"] or sha256_file(project_path),
             "sourceDataset": fixture["source_dataset_hash"] or sha256_file(fixture["dataset_path"])},
            {}, {"primaryTargetResult": sha256_json(results[1])},
            {"identity": sha256_json(results[2])},
            {"hypothesisAnalysis": sha256_json(results[3])},
            {"planner": sha256_json(results[4])},
        ]
        stages = tuple(InvestigationStage(
            fixture["ids"][index], fixture["handlers"][index],
            fixture["statuses"][index], fixture["triggers"][index], {},
            result=results[index], provenance=StageProvenance("test", "1", hashes[index]),
            artifacts=tuple(fixture["artifacts"] if index == 0 else ()),
            stop=fixture["stop"] if index == 5 else False,
        ) for index in range(len(fixture["ids"])))
        investigation = replace(fixture["investigation"],
            status=fixture["investigation_status"], stages=stages,
            metadata={"controlState": fixture["control"]})
        fixture["investigation"] = investigation
        fixture["store"].save(investigation)
        stage_dir = fixture["store"].directory_for(investigation.id) / "stages"
        stage_dir.mkdir(parents=True, exist_ok=True)
        for old in stage_dir.glob("*.json"):
            old.unlink()
        for stage in stages:
            fixture["store"]._atomic_write_json(
                fixture["store"].stage_path_for(investigation.id, stage.id),
                asdict(stage), replace=False)
        return fixture

    def _assert_rejected(self, fixture):
        investigation = fixture["investigation"]
        self.assertEqual(investigation,
                         repair_obsolete_terminal_wait(fixture["store"], investigation))

    def test_exact_boundary_is_append_only_artifact_verified_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._boundary(Path(temporary))
            store, investigation = fixture["store"], fixture["investigation"]
            ledgers = {stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                       for stage in investigation.stages}
            hashes = {name: sha256_file(store.stage_path_for(investigation.id, name))
                      for name in ledgers}
            source_files = {path: path.read_bytes() for path in
                            (fixture["project_path"], fixture["dataset_path"],
                             fixture["artifact_path"])}

            repaired = repair_obsolete_terminal_wait(store, investigation)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("007-prepare-independent-sectors", selected["id"])
            self.assertEqual("006-finalize", selected["triggered_by_stage_id"])
            self.assertEqual({"investigationGoal": "FULL_CHARACTERIZATION"},
                             selected["parameters"])
            self.assertEqual(6, len(repaired.stages))
            self.assertEqual(ledgers, {name: store.stage_path_for(investigation.id, name).read_bytes()
                                       for name in ledgers})
            self.assertEqual(hashes, {name: sha256_file(store.stage_path_for(investigation.id, name))
                                      for name in hashes})
            self.assertEqual(source_files, {path: path.read_bytes() for path in source_files})
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))

    def test_internally_consistent_near_matches_fail_closed(self):
        def marker(field, value):
            return lambda f: f["project"]["preparer"].__setitem__(field, value)
        cases = {
            "preparer ID": marker("preparerID", "other.preparer"),
            "schema version": marker("schemaVersion", 2),
            "owned-file marker": marker("ownedFiles", ["project.json"]),
            "marker project ID": marker("projectID", "other-project"),
            "blind role": lambda f: (f["project"]["datasets"][0].__setitem__("role", "science"),
                                      f["results"][0]["sourceDatasetEntry"].__setitem__("role", "science")),
            "source project hash": lambda f: f.__setitem__("source_project_hash", "0" * 64),
            "source dataset hash": lambda f: f.__setitem__("source_dataset_hash", "0" * 64),
            "dataset-entry hash": lambda f: (f["project"]["datasets"][0].__setitem__("datasetSHA256", "0" * 64),
                                               f["results"][0]["sourceDatasetEntry"].__setitem__("datasetSHA256", "0" * 64)),
            "investigation status": lambda f: f.__setitem__("investigation_status", "BLOCKED"),
            "control state": lambda f: f.__setitem__("control", {"schedulerAction": "WAIT_FOR_PREREQUISITES"}),
            "stage count": lambda f: [f[key].pop() for key in ("ids", "handlers", "statuses", "triggers", "results")],
            "stage IDs": lambda f: f["ids"].__setitem__(3, "009-hypotheses"),
            "handler order": lambda f: f["handlers"].__setitem__(3, "openstar.tess.planner"),
            "stage status": lambda f: f["statuses"].__setitem__(3, "FAILED"),
            "triggered-by chain": lambda f: f["triggers"].__setitem__(3, "001-prepare-target"),
            "finalizer stop": lambda f: f.__setitem__("stop", False),
            "planner action": lambda f: f["results"][4].__setitem__("action", "INDEPENDENT_SECTOR_FOLLOWUP"),
            "planner reason": lambda f: f["results"][4].__setitem__("reason", "other"),
            "final claim": lambda f: f["results"][5].__setitem__("claim", "CANDIDATE_PERIOD"),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._boundary(Path(temporary))
                mutate(fixture)
                self._assert_rejected(self._persist(fixture))

    def test_missing_modified_artifact_and_altered_ledger_bytes_fail_closed(self):
        for mode in ("missing", "modified"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                fixture = self._boundary(Path(temporary))
                if mode == "missing":
                    fixture["artifact_path"].unlink()
                else:
                    fixture["artifact_path"].write_bytes(b"modified\n")
                self._assert_rejected(fixture)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._boundary(Path(temporary))
            ledger = fixture["store"].stage_path_for(fixture["investigation"].id,
                                                       "004-hypotheses")
            ledger.write_bytes(ledger.read_bytes() + b"altered")
            self._assert_rejected(fixture)


if __name__ == "__main__":
    unittest.main()
