import json
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
