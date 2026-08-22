import json
import tempfile
import unittest
from pathlib import Path

from backfill_openstar_science_runs import (
    active_sector_sweep_processes,
    backfill_sector_sweeps,
    discover_sector_inventory_paths,
)
from openstar_science_runs import ScienceRunCatalog, ScienceRunRecorder, science_run_id


class ScienceRunCatalogTests(unittest.TestCase):
    def test_register_resume_and_finish_preserve_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "science.sqlite3")
            state = Path(temporary, "state")
            catalog = ScienceRunCatalog(path)
            run_id = science_run_id("test-science", state, identity="alpha")

            started = catalog.register(
                run_id,
                kind="test-science",
                display_name="Test Science",
                state_root=state,
                workflow_id="workflow.test",
                metadata={"target": "alpha"},
                now="2026-08-22T10:00:00Z",
            )
            resumed = catalog.register(
                run_id,
                kind="test-science",
                display_name="Test Science",
                state_root=state,
                workflow_id="workflow.test",
                metadata={"target": "alpha"},
                now="2026-08-22T11:00:00Z",
            )
            finished = catalog.update(
                run_id,
                status="FINISHED",
                summary={"result": "ok"},
                now="2026-08-22T12:00:00Z",
            )

            self.assertEqual(started["startedAt"], resumed["startedAt"])
            self.assertEqual("2026-08-22T10:00:00Z", finished["startedAt"])
            self.assertEqual("2026-08-22T12:00:00Z", finished["completedAt"])
            self.assertEqual({"result": "ok"}, finished["summary"])
            self.assertEqual(str(state.resolve()), finished["stateRoot"])

    def test_multiple_catalog_instances_do_not_clobber_other_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "science.sqlite3")
            first = ScienceRunCatalog(path)
            second = ScienceRunCatalog(path)
            first.register("a", kind="one", display_name="One")
            second.register("b", kind="two", display_name="Two")
            self.assertEqual({"a", "b"}, {item["id"] for item in first.list_runs()})

    def test_read_only_missing_catalog_is_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = ScienceRunCatalog(Path(temporary, "missing.sqlite3"), create=False)
            self.assertEqual([], catalog.list_runs())
            self.assertIsNone(catalog.get("missing"))

    def test_recorder_is_stable_for_same_state_and_updates_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = ScienceRunCatalog(Path(temporary, "science.sqlite3"))
            state = Path(temporary, "state")
            first = ScienceRunRecorder(
                kind="example",
                display_name="Example",
                state_root=state,
                identity="same",
                catalog=catalog,
            )
            second = ScienceRunRecorder(
                kind="example",
                display_name="Example",
                state_root=state,
                identity="same",
                catalog=catalog,
            )
            self.assertEqual(first.run_id, second.run_id)
            second.finish(status="COMPLETE", summary={"done": True})
            self.assertEqual("COMPLETE", catalog.get(first.run_id)["status"])


class ScienceRunBackfillTests(unittest.TestCase):
    def write_inventory(self, root: Path, sector: int, count: int = 1) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"tess-sector-{sector}-inventory.json"
        path.write_text(
            json.dumps({"sector": sector, "entries": [{}] * count}),
            encoding="utf-8",
        )
        return path

    def test_discovery_is_bounded_but_finds_nested_run_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.write_inventory(root / "group" / "sector-state", 2)
            deep = self.write_inventory(root / "a" / "b" / "too-deep", 3)
            found = discover_sector_inventory_paths((root,))
            self.assertIn(expected.resolve(), found)
            self.assertNotIn(deep.resolve(), found)

    def test_process_parser_requires_real_sector_runner_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary, "sector-two").resolve()
            commands = [
                f"python run_openstar_tess_sector_sweep.py --sector 2 --state-dir {state} --coordinator-url http://127.0.0.1:8080",
                "python unrelated.py --sector 4 --state-dir /tmp/not-openstar",
            ]
            self.assertEqual({state: 2}, active_sector_sweep_processes(commands))

    def test_backfill_marks_only_process_proven_active_run_as_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sector_one = root / "sector-one"
            sector_two = root / "sector-two"
            self.write_inventory(sector_one, 1)
            self.write_inventory(sector_two, 2)
            catalog = ScienceRunCatalog(root / "science.sqlite3")

            registered = backfill_sector_sweeps(
                catalog,
                (root,),
                active={sector_two.resolve(): 2},
            )
            by_sector = {item["metadata"]["sector"]: item for item in registered}

            self.assertEqual("DISCOVERED_INCOMPLETE", by_sector[1]["status"])
            self.assertEqual("RUNNING", by_sector[2]["status"])
            self.assertEqual(2, len(catalog.list_runs()))
            self.assertEqual(2, by_sector[2]["summary"]["sectorSweep"]["sector"])


if __name__ == "__main__":
    unittest.main()
