import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from openstar_dashboard import DashboardApplication
from openstar_science_runs import (
    ScienceRunCatalog,
    ScienceRunRecorder,
    backfill_science_runs,
    discover_science_runs,
    stable_run_id,
    catalog_path,
    recorded_science_run,
)


class ScienceRunCatalogTests(unittest.TestCase):
    def test_identity_and_resume_are_stable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = ScienceRunCatalog(Path(temporary) / "catalog.sqlite3")
            root = Path(temporary) / "state"
            first = catalog.record("tess-sector-sweep", root, metadata={"sector": 4})
            second = catalog.record("tess-sector-sweep", root, status="FINISHED", metadata={"sector": 4})
            self.assertEqual(first, second)
            self.assertEqual(stable_run_id("tess-sector-sweep", root), first)
            self.assertEqual(1, len(catalog.list_runs()))
            self.assertEqual("FINISHED", catalog.list_runs()[0].status)

    def test_logical_identity_distinguishes_runs_sharing_a_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            first = stable_run_id("generic-investigation", root, "investigation-a")
            resumed = stable_run_id("generic-investigation", root, "investigation-a")
            second = stable_run_id("generic-investigation", root, "investigation-b")
            self.assertEqual(first, resumed)
            self.assertNotEqual(first, second)

    def test_default_catalog_does_not_depend_on_cwd(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            try:
                os.chdir(first); one = catalog_path()
                os.chdir(second); two = catalog_path()
            finally:
                os.chdir(original)
        self.assertEqual(one, two)
        self.assertEqual(Path(__file__).resolve().parent / "data" / "science-runs.sqlite3", one)

    def test_decorator_records_results_interruptions_and_exceptions(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = os.environ.get("OPENSTAR_SCIENCE_RUN_CATALOG")
            path = Path(temporary) / "catalog.sqlite3"
            os.environ["OPENSTAR_SCIENCE_RUN_CATALOG"] = str(path)
            try:
                @recorded_science_run("test", "root", logical_identity="identity")
                def run(root, identity, outcome):
                    if outcome == "interrupt": raise KeyboardInterrupt
                    if outcome == "exception": raise RuntimeError("science failure")
                    return outcome
                self.assertEqual(0, run(temporary, "success", 0))
                self.assertEqual(2, run(temporary, "nonzero", 2))
                with self.assertRaises(KeyboardInterrupt): run(temporary, "interrupt", "interrupt")
                with self.assertRaisesRegex(RuntimeError, "science failure"): run(temporary, "exception", "exception")
                by_id = {item.run_id: item.status for item in ScienceRunCatalog(path).list_runs()}
                self.assertEqual("FINISHED", by_id[stable_run_id("test", temporary, "success")])
                self.assertEqual("FAILED", by_id[stable_run_id("test", temporary, "nonzero")])
                self.assertEqual("INTERRUPTED", by_id[stable_run_id("test", temporary, "interrupt")])
                self.assertEqual("FAILED", by_id[stable_run_id("test", temporary, "exception")])
            finally:
                if old is None: os.environ.pop("OPENSTAR_SCIENCE_RUN_CATALOG", None)
                else: os.environ["OPENSTAR_SCIENCE_RUN_CATALOG"] = old

    def test_concurrent_catalog_access_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = ScienceRunCatalog(Path(temporary) / "catalog.sqlite3")
            root = Path(temporary) / "state"
            threads = [threading.Thread(target=catalog.record, args=("generic-investigation", root)) for _ in range(12)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(1, len(catalog.list_runs()))

    def test_recorder_failure_is_isolated(self):
        broken = Mock()
        broken.record.side_effect = OSError("disk full")
        recorder = ScienceRunRecorder("generic-investigation", "/gone", catalog=broken)
        recorder.update("RUNNING")
        with recorder:
            pass

    def test_missing_and_partial_roots_are_degraded_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary) / "catalog.sqlite3"
            missing = Path(temporary) / "missing"
            catalog = ScienceRunCatalog(catalog_path)
            catalog.record("tess-sector-sweep", missing, metadata={"sector": 9})
            record = discover_science_runs(catalog_path)[0]
            self.assertEqual("degraded", record["condition"])
            self.assertIn("state_root_missing", record["issues"])
            self.assertFalse(missing.exists())

            partial = Path(temporary) / "partial"
            (partial / "investigations" / "old" / "stages").mkdir(parents=True)
            (partial / "runner.pid").write_text("999999", encoding="utf-8")
            catalog.record("generic-investigation", partial)
            record = next(item for item in discover_science_runs(catalog_path) if item["stateRoot"] == str(partial))
            self.assertIn("investigation_records_missing", record["issues"])
            self.assertTrue((partial / "runner.pid").exists())

    def test_backfill_is_bounded_idempotent_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "science"
            root.mkdir()
            inventory = root / "tess-sector-2-inventory.json"
            inventory.write_text(json.dumps({"sector": 2, "entries": []}), encoding="utf-8")
            before = inventory.read_bytes()
            path = Path(temporary) / "catalog.sqlite3"
            self.assertEqual(1, backfill_science_runs([root], path, limit=1))
            self.assertEqual(1, backfill_science_runs([root], path, limit=1))
            self.assertEqual(1, len(ScienceRunCatalog(path).list_runs()))
            self.assertEqual(before, inventory.read_bytes())

    def test_backfill_recognizes_ranking_only_and_prefers_active_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            scan = Path(temporary)
            ranking_root = scan / "openstar-tess-sector-1-smoke"; ranking_root.mkdir()
            ranking = ranking_root / "tess-sector-1-ranking.json"
            ranking.write_text(json.dumps({"sector": 1, "inventoryCount": 100,
                "completedCount": 80, "remainingCount": 20, "rankingComplete": False}), encoding="utf-8")
            old_duplicate = scan / "sector-2-old"; old_duplicate.mkdir()
            (old_duplicate / "tess-sector-2-ranking.json").write_text(json.dumps({"sector": 2}), encoding="utf-8")
            active = scan / "sector-2-active"; active.mkdir()
            inventory = active / "tess-sector-2-inventory.json"
            inventory.write_text(json.dumps({"sector": 2, "entries": []}), encoding="utf-8")
            (active / "sweep.pid").write_text(str(os.getpid()), encoding="utf-8")
            before = {path: path.read_bytes() for path in (ranking, inventory, active / "sweep.pid")}
            path = scan / "catalog.sqlite3"
            self.assertEqual(2, backfill_science_runs([scan], path))
            runs = {run.metadata["sector"]: run for run in ScienceRunCatalog(path).list_runs()}
            self.assertEqual(str(ranking_root), runs[1].state_root)
            self.assertEqual(100, runs[1].metadata["inventoryCount"])
            self.assertEqual(str(active), runs[2].state_root)
            self.assertEqual("RUNNING", runs[2].status)
            self.assertEqual(2, backfill_science_runs([scan], path))
            self.assertEqual(before, {source: source.read_bytes() for source in before})

    def test_dashboard_discovers_cataloged_sweep_without_manual_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            root.mkdir()
            (root / "tess-sector-3-inventory.json").write_text(
                json.dumps({"sector": 3, "entries": []}), encoding="utf-8")
            path = Path(temporary) / "catalog.sqlite3"
            ScienceRunCatalog(path).record("tess-sector-sweep", root, metadata={"sector": 3})
            coordinator = Mock()
            coordinator.observation.return_value = {"health": {}, "nodes": [], "contributions": {}, "projects": []}
            _, observation = DashboardApplication(coordinator, science_run_catalog=path).snapshot()
            self.assertEqual(3, observation["sectorSweeps"][0]["sector"])
            self.assertEqual("tess-sector-sweep", observation["scienceRuns"][0]["kind"])


if __name__ == "__main__":
    unittest.main()
