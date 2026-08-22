import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_openstar_autonomous_tess
import run_openstar_tess_ranked_followup
import run_openstar_tess_sector_sweep
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


class ScienceRunRunnerRegistrationTests(unittest.TestCase):
    def catalog_from(self, temporary: str) -> ScienceRunCatalog:
        return ScienceRunCatalog(Path(temporary, "science.sqlite3"), create=False)

    def test_sector_sweep_main_registers_without_dashboard_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary, "science.sqlite3")
            state = Path(temporary, "sector-two")
            projection = {
                "sector": 2,
                "status": "RUNNING",
                "inventory": 10,
                "admitted": 10,
                "complete": 4,
                "remaining": 6,
                "recoveryRequired": 0,
                "inFlightOrRecovery": 1,
                "runnable": 5,
                "waitingExternalData": 0,
                "blockedPrerequisites": 0,
                "failed": 0,
                "unclassified": 0,
                "progress": 0.4,
            }
            with patch.dict(os.environ, {"OPENSTAR_SCIENCE_RUN_CATALOG": str(catalog_path)}), \
                 patch.object(run_openstar_tess_sector_sweep, "run_tess_sector_sweep", return_value=0), \
                 patch.object(run_openstar_tess_sector_sweep, "sector_sweep_projection", return_value=[projection]):
                code = run_openstar_tess_sector_sweep.main([
                    "--sector", "2",
                    "--coordinator-url", "http://127.0.0.1:8080",
                    "--state-dir", str(state),
                ])

            self.assertEqual(0, code)
            runs = self.catalog_from(temporary).list_runs()
            self.assertEqual(1, len(runs))
            self.assertEqual("tess-sector-sweep", runs[0]["kind"])
            self.assertEqual(2, runs[0]["metadata"]["sector"])
            self.assertEqual("FINISHED", runs[0]["status"])
            self.assertEqual(projection, runs[0]["summary"]["sectorSweep"])

    def test_ranked_followup_main_registers_deep_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary, "science.sqlite3")
            shallow = Path(temporary, "shallow")
            deep = Path(temporary, "deep")
            with patch.dict(os.environ, {"OPENSTAR_SCIENCE_RUN_CATALOG": str(catalog_path)}), \
                 patch.object(run_openstar_tess_ranked_followup, "run_tess_ranked_followup", return_value=0):
                code = run_openstar_tess_ranked_followup.main([
                    "--sector", "2",
                    "--sector-state-dir", str(shallow),
                    "--deep-state-dir", str(deep),
                    "--coordinator-url", "http://127.0.0.1:8080",
                    "--promote-top", "5",
                ])

            self.assertEqual(0, code)
            run = self.catalog_from(temporary).list_runs()[0]
            self.assertEqual("tess-ranked-followup", run["kind"])
            self.assertEqual(str(deep.resolve()), run["stateRoot"])
            self.assertEqual("FINISHED", run["status"])

    def test_autonomous_main_registers_state_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary, "science.sqlite3")
            state = Path(temporary, "autonomous")
            project = Path(temporary, "project.json")
            with patch.dict(os.environ, {"OPENSTAR_SCIENCE_RUN_CATALOG": str(catalog_path)}), \
                 patch.object(run_openstar_autonomous_tess, "run_autonomous_tess", return_value=0):
                code = run_openstar_autonomous_tess.main([
                    "--project", str(project),
                    "--coordinator-url", "http://127.0.0.1:8080",
                    "--state-dir", str(state),
                ])

            self.assertEqual(0, code)
            run = self.catalog_from(temporary).list_runs()[0]
            self.assertEqual("autonomous-investigation", run["kind"])
            self.assertEqual(str(state.resolve()), run["stateRoot"])
            self.assertEqual("FINISHED", run["status"])

    def test_generic_investigation_runner_uses_science_recorder(self):
        source = Path("run_investigation.py").read_text(encoding="utf-8")
        self.assertIn("ScienceRunRecorder", source)
        self.assertIn('kind="investigation"', source)


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
