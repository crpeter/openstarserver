import json
import tempfile
import unittest
from pathlib import Path

from openstar_sector_sweep_status import sector_sweep_projection
from workflows.tess.tess_sector_scan import WORKFLOW_ID


class SectorSweepProjectionTests(unittest.TestCase):
    def write_sweep(self, root, sector, inventory, states):
        Path(root, f"tess-sector-{sector}-inventory.json").write_text(
            json.dumps({"sector": sector, "entries": [{}] * inventory})
        )
        for number, (status, running_stage) in enumerate(states):
            path = Path(root, "investigations", str(number), "investigation.json")
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "workflow_id": WORKFLOW_ID,
                        "status": status,
                        "metadata": {"sector": sector},
                        "stages": [{"status": "RUNNING"}] if running_stage else [],
                    }
                )
            )

    def test_progress_remaining_and_recovery_are_independent(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_sweep(
                root,
                1,
                5,
                [
                    ("COMPLETE", False),
                    ("COMPLETE", False),
                    ("FAILED", False),
                    ("RUNNING", True),
                ],
            )
            self.assertEqual(
                {
                    "sector": 1,
                    "status": "RUNNING",
                    "inventory": 5,
                    "admitted": 4,
                    "complete": 2,
                    "remaining": 3,
                    "recoveryRequired": 2,
                    "runnable": 1,
                    "progress": 0.4,
                },
                sector_sweep_projection(root)[0],
            )

    def test_empty_inventory_has_finite_zero_progress(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_sweep(root, 2, 0, [])
            sweep = sector_sweep_projection(root)[0]
            self.assertEqual(
                (0, 0, 0.0, "RUNNING"),
                (
                    sweep["complete"],
                    sweep["remaining"],
                    sweep["progress"],
                    sweep["status"],
                ),
            )

    def test_no_persisted_sweep_is_omitted_and_completed_is_retained(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual([], sector_sweep_projection(root))
            self.write_sweep(root, 3, 1, [("COMPLETE", False)])
            self.assertEqual("COMPLETE", sector_sweep_projection(root)[0]["status"])
