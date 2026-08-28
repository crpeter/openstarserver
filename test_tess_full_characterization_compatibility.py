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
    def _boundary(self, root: Path):
        source = root / "source"
        source.mkdir()
        dataset_path = source / "dataset.json"
        dataset_path.write_text('{"science":{"role":"blind"}}\n')
        entry = {
            "id": "generic-dataset", "path": str(dataset_path), "role": "blind",
            "datasetSHA256": sha256_file(dataset_path),
        }
        project_path = source / "project.json"
        project = {
            "id": "generic-project", "datasets": [entry],
            "preparer": {
                "preparerID": "openstar.tess-known-target-blind-benchmark-preparer",
                "schemaVersion": 1, "ownedFiles": ["dataset.json", "project.json"],
                "projectID": "generic-project",
            },
        }
        project_path.write_text(json.dumps(project) + "\n")
        prepared = {
            "sourceProjectPath": str(project_path), "datasetPath": str(dataset_path),
            "sourceDatasetEntry": entry,
        }
        results = [
            prepared,
            {"datasets": [{"candidatePeriodDays": 2.0}]},
            {"identityResolved": True},
            {"observedPeriodDays": 2.0, "bestCatalogMatch": {"source": "catalog"}},
            {"action": "STOP", "reason": "catalog-period-match"},
            {"claim": "KNOWN_PERIOD_RECOVERED"},
        ]
        handlers = (
            "openstar.tess.prepare-target", "openstar.tess.primary-project.run",
            "openstar.tess.catalog-identity", "openstar.tess.hypotheses",
            "openstar.tess.planner", "openstar.tess.finalize",
        )
        ids = ("001-prepare-target", "002-primary-distributed-search",
               "003-catalog-identity", "004-hypotheses", "005-planner", "006-finalize")
        input_hashes = [
            {"sourceProjectManifest": sha256_file(project_path),
             "sourceDataset": sha256_file(dataset_path)},
            {}, {"primaryTargetResult": sha256_json(results[1])},
            {"identity": sha256_json(results[2])},
            {"hypothesisAnalysis": sha256_json(results[3])},
            {"planner": sha256_json(results[4])},
        ]
        store = InvestigationStore(root / "investigations")
        investigation = store.create("generic-investigation", WORKFLOW_ID, "20.2")
        stages = tuple(InvestigationStage(
            id=stage_id, handler_id=handler, status="COMPLETE",
            triggered_by_stage_id=(ids[index - 1] if index else None), parameters={},
            result=results[index], provenance=StageProvenance("test", "1", input_hashes[index]),
            stop=index == 5,
        ) for index, (stage_id, handler) in enumerate(zip(ids, handlers)))
        investigation = replace(
            investigation, status="COMPLETE", stages=stages,
            metadata={"controlState": {"schedulerAction": "INVESTIGATION_COMPLETE",
                                       "selectedExperiment": None}},
        )
        store.save(investigation)
        for stage in stages:
            store._atomic_write_json(store.stage_path_for(investigation.id, stage.id),
                                     asdict(stage), replace=False)
        return store, investigation

    def test_exact_boundary_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._boundary(Path(temporary))
            ledgers = {stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                       for stage in investigation.stages}
            hashes = {name: sha256_file(store.stage_path_for(investigation.id, name))
                      for name in ledgers}

            repaired = repair_obsolete_terminal_wait(store, investigation)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("007-prepare-independent-sectors", selected["id"])
            self.assertEqual("006-finalize", selected["triggered_by_stage_id"])
            self.assertEqual("FULL_CHARACTERIZATION",
                             selected["parameters"]["investigationGoal"])
            self.assertEqual(ledgers, {name: store.stage_path_for(investigation.id, name).read_bytes()
                                       for name in ledgers})
            self.assertEqual(hashes, {name: sha256_file(store.stage_path_for(investigation.id, name))
                                      for name in hashes})
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))

    def test_near_matches_fail_closed(self):
        mutations = {
            "role": (0, lambda stages: replace(stages[0], result={**stages[0].result,
                "sourceDatasetEntry": {**stages[0].result["sourceDatasetEntry"], "role": "science"}})),
            "planner": (4, lambda stages: replace(stages[4], result={"action": "STOP", "reason": "other"})),
            "finalizer": (5, lambda stages: replace(stages[5], handler_id="other.finalize")),
            "order": (3, lambda stages: replace(stages[3], id="009-hypotheses")),
        }
        for name, (index, mutate) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                store, investigation = self._boundary(Path(temporary))
                stages = list(investigation.stages)
                stages[index] = mutate(stages)
                altered = replace(investigation, stages=tuple(stages))
                self.assertEqual(altered, repair_obsolete_terminal_wait(store, altered))


if __name__ == "__main__":
    unittest.main()
