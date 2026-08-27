import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from openstar_investigation import (
    InvestigationStage, InvestigationStore, StageProvenance,
)


class HistoricalStageLedgerVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = InvestigationStore(Path(self.temporary.name))
        self.stage = InvestigationStage(
            "011-interpret-broad-independent-search",
            "openstar.tess.independent.broad.interpret",
            "COMPLETE", "010-broad-independent-search", {"sectors": (94, 95)},
            started_at="start", completed_at="complete",
            result={"harmonicFamily": {"supportingSectors": (94, 95),
                "physicalCycleResolved": False}},
            provenance=StageProvenance("openstar", "old",
                project_ids=("project-94", "project-95")),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, mutation=None):
        payload = json.loads(json.dumps(asdict(self.stage)))
        if mutation:
            mutation(payload)
        path = self.store.stage_path_for("investigation", self.stage.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_json_arrays_verify_tuple_backed_complete_stage(self):
        self.write()
        self.assertIsNotNone(self.store.verified_terminal_stage_ledger_hash(
            "investigation", self.stage))

    def test_any_terminal_record_tampering_fails(self):
        mutations = (
            lambda value: value["result"]["harmonicFamily"].update(
                supportingSectors=[94]),
            lambda value: value.update(handler_id="wrong"),
            lambda value: value.update(id="wrong"),
            lambda value: value["provenance"].update(software_version="tampered"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                path = self.store.stage_path_for("investigation", self.stage.id)
                path.unlink(missing_ok=True)
                self.write(mutation)
                self.assertIsNone(self.store.verified_terminal_stage_ledger_hash(
                    "investigation", self.stage))


if __name__ == "__main__":
    unittest.main()
