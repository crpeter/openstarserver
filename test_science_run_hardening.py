import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_openstar_tess_sector_sweep
from openstar_dashboard import DashboardApplication
from openstar_science_runs import ScienceRunCatalog


class TinyCoordinator:
    def observation(self):
        return {
            "health": {"ok": True},
            "nodes": [],
            "contributions": {"allTime": {}, "currentSession": {}},
            "projects": [],
        }


class BrokenCatalog:
    def list_runs(self):
        raise RuntimeError("catalog unavailable")


class ScienceRunHardeningTests(unittest.TestCase):
    def test_broken_science_catalog_does_not_take_down_fleet_dashboard(self):
        application = DashboardApplication(
            TinyCoordinator(),
            science_run_catalog=BrokenCatalog(),
        )

        snapshot, observation = application.snapshot()

        self.assertEqual([], observation["scienceRuns"])
        self.assertEqual([], observation["sectorSweeps"])
        self.assertEqual(0, snapshot["summary"]["knownWorkers"])

    def test_failed_sector_sweep_stays_failed_even_if_projection_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog_path = Path(temporary, "science.sqlite3")
            state = Path(temporary, "sector")
            projection = {
                "sector": 2,
                "status": "COMPLETE",
                "inventory": 1,
                "admitted": 1,
                "complete": 1,
                "remaining": 0,
                "recoveryRequired": 0,
                "inFlightOrRecovery": 0,
                "runnable": 0,
                "waitingExternalData": 0,
                "blockedPrerequisites": 0,
                "failed": 0,
                "unclassified": 0,
                "progress": 1.0,
            }
            with patch.dict(
                os.environ,
                {"OPENSTAR_SCIENCE_RUN_CATALOG": str(catalog_path)},
            ), patch.object(
                run_openstar_tess_sector_sweep,
                "run_tess_sector_sweep",
                return_value=1,
            ), patch.object(
                run_openstar_tess_sector_sweep,
                "sector_sweep_projection",
                return_value=[projection],
            ):
                code = run_openstar_tess_sector_sweep.main(
                    [
                        "--sector",
                        "2",
                        "--coordinator-url",
                        "http://127.0.0.1:8080",
                        "--state-dir",
                        str(state),
                    ]
                )

            self.assertEqual(1, code)
            run = ScienceRunCatalog(catalog_path, create=False).list_runs()[0]
            self.assertEqual("FAILED", run["status"])
            self.assertEqual(1, run["summary"]["exitCode"])


if __name__ == "__main__":
    unittest.main()
