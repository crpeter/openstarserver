import json
import tempfile
import unittest
from pathlib import Path

from backfill_openstar_science_runs import backfill_sector_sweeps
from openstar_dashboard import DashboardHandler
from openstar_science_runs import ScienceRunCatalog, science_run_id


class _BrokenWriter:
    def write(self, _body):
        raise BrokenPipeError("client navigated away")


class DashboardPolishTests(unittest.TestCase):
    def test_cancelled_client_write_is_ignored(self):
        handler = object.__new__(DashboardHandler)
        handler.wfile = _BrokenWriter()
        handler._write_body(b"payload")

    def test_sector_ui_uses_exact_counts_and_two_decimal_progress(self):
        source = Path("dashboard/app.js").read_text(encoding="utf-8")
        self.assertIn("const exactFmt", source)
        self.assertIn("toFixed(2)", source)
        self.assertIn("exactFmt(sweep.complete)", source)
        self.assertIn("exactFmt(sweep.inventory)", source)
        self.assertIn('["Remaining", sweep.remaining]', source)
        self.assertIn("exactFmt(values.get(label))", source)

    def test_live_poll_updates_stable_sector_nodes_in_place(self):
        source = Path("dashboard/app.js").read_text(encoding="utf-8")
        self.assertIn("reconcile($(\"#sectors\")", source)
        self.assertIn("updateSectorCard", source)
        self.assertIn("setText(refs.total", source)
        self.assertIn("refs.barFill.style.width", source)
        self.assertIn("setInterval(refreshActivity, 2000)", source)
        self.assertIn("setInterval(refreshFleet, 10000)", source)
        self.assertNotIn("replace($(\"#sectors\")", source)
        self.assertNotIn("replace($(\"#stats\")", source)
        self.assertNotIn("replace($(\"#workers\")", source)

    def test_counts_are_exact_and_compact_notation_is_reserved_for_rates(self):
        source = Path("dashboard/app.js").read_text(encoding="utf-8")
        self.assertIn("exactFmt(worker.completedWorkUnits)", source)
        self.assertIn("exactFmt(project.projectCompletedWorkUnits", source)
        self.assertIn("exactFmt(contribution.acceptedWorkUnits)", source)
        self.assertIn("exactFmt(summary.completedWorkUnits)", source)
        self.assertIn("compactFmt(worker.measuredThroughput)", source)


class CanonicalBackfillTests(unittest.TestCase):
    def write_inventory(self, root: Path, sector: int, count: int) -> None:
        root.mkdir(parents=True, exist_ok=True)
        Path(root, f"tess-sector-{sector}-inventory.json").write_text(
            json.dumps({"sector": sector, "entries": [{}] * count}),
            encoding="utf-8",
        )

    def test_backfill_keeps_one_most_substantial_legacy_run_per_sector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            small = root / "sector-small"
            full = root / "sector-full"
            self.write_inventory(small, 1, 2)
            self.write_inventory(full, 1, 20)
            catalog = ScienceRunCatalog(root / "science.sqlite3")

            stale_id = science_run_id("tess-sector-sweep", small, identity="1")
            catalog.register(
                stale_id,
                kind="tess-sector-sweep",
                display_name="TESS Sector 1 Sweep",
                status="DISCOVERED_INCOMPLETE",
                state_root=small,
                metadata={"sector": 1, "backfilled": True},
            )

            registered = backfill_sector_sweeps(catalog, (root,), active={})

            self.assertEqual(1, len(registered))
            self.assertEqual(str(full.resolve()), registered[0]["stateRoot"])
            runs = catalog.list_runs()
            self.assertEqual(1, len(runs))
            self.assertEqual(str(full.resolve()), runs[0]["stateRoot"])
            self.assertIsNone(catalog.get(stale_id))

    def test_active_legacy_run_wins_even_if_another_root_is_larger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_root = root / "active"
            larger = root / "larger"
            self.write_inventory(active_root, 2, 3)
            self.write_inventory(larger, 2, 30)
            catalog = ScienceRunCatalog(root / "science.sqlite3")

            registered = backfill_sector_sweeps(
                catalog,
                (root,),
                active={active_root.resolve(): 2},
            )

            self.assertEqual(1, len(registered))
            self.assertEqual(str(active_root.resolve()), registered[0]["stateRoot"])
            self.assertEqual("DISCOVERED_ACTIVE", registered[0]["status"])


if __name__ == "__main__":
    unittest.main()
