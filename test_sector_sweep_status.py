import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from openstar_investigation import Investigation, InvestigationStage
from openstar_sector_sweep_status import sector_sweep_projection
from workflows.tess.tess_sector_scan import WORKFLOW_ID


class SectorSweepProjectionTests(unittest.TestCase):
    def write_inventory(self, root, sector, count):
        Path(root, f"tess-sector-{sector}-inventory.json").write_text(
            json.dumps({"sector": sector, "entries": [{}] * count})
        )

    def write_investigation(
        self,
        root,
        number,
        *,
        status="RUNNING",
        action=None,
        stage_status=None,
        explicit_recovery=False,
    ):
        stages = ()
        if stage_status:
            stages = (
                InvestigationStage(
                    id="001-work",
                    handler_id="handler",
                    status=stage_status,
                    triggered_by_stage_id=None,
                    parameters={},
                    error="failure" if stage_status == "FAILED" else None,
                ),
            )
        metadata = {"sector": 1}
        if action:
            control = {"schedulerAction": action}
            if action == "RUN_EXPERIMENT":
                control["selectedExperiment"] = {
                    "id": "002-work" if explicit_recovery else "001-work",
                    "handler_id": "handler",
                    "parameters": {},
                    "triggered_by_stage_id": "001-work" if explicit_recovery else None,
                }
            metadata["controlState"] = control
        investigation = Investigation(
            id=f"investigation-{number}",
            workflow_id=WORKFLOW_ID,
            workflow_version="1",
            status=status,
            created_at="now",
            updated_at="now",
            metadata=metadata,
            stages=stages,
        )
        path = Path(root, "investigations", str(number), "investigation.json")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(asdict(investigation)))

    def test_only_authoritative_persisted_scheduler_states_are_counted(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_inventory(root, 1, 8)
            self.write_investigation(
                root, 1, status="COMPLETE", action="INVESTIGATION_COMPLETE"
            )
            self.write_investigation(root, 2, stage_status="FAILED")
            self.write_investigation(root, 3, action="WAIT_FOR_PREREQUISITES")
            self.write_investigation(
                root,
                4,
                status="QUIESCENT_AWAITING_DATA",
                action="ADVANCE_TO_NEXT_TARGET",
            )
            self.write_investigation(root, 5, stage_status="RUNNING")
            self.write_investigation(
                root,
                6,
                stage_status="FAILED",
                action="RUN_EXPERIMENT",
                explicit_recovery=True,
            )
            sweep = sector_sweep_projection(root)[0]
            self.assertEqual(1, sweep["complete"])
            self.assertEqual(7, sweep["remaining"])
            self.assertEqual(1 / 8, sweep["progress"])
            self.assertEqual(1, sweep["failed"])
            self.assertEqual(1, sweep["blockedPrerequisites"])
            self.assertEqual(1, sweep["waitingExternalData"])
            self.assertEqual(0, sweep["recoveryRequired"])
            self.assertEqual(1, sweep["inFlightOrRecovery"])
            self.assertEqual(1, sweep["runnable"])
            self.assertEqual(0, sweep["unclassified"])

    def test_persisted_running_stage_is_not_definite_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_inventory(root, 1, 1)
            self.write_investigation(root, 1, stage_status="RUNNING")
            sweep = sector_sweep_projection(root)[0]
            self.assertEqual(0, sweep["recoveryRequired"])
            self.assertEqual(1, sweep["inFlightOrRecovery"])
            self.assertEqual(0, sweep["runnable"])

    def test_unadmitted_inventory_is_not_runnable(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_inventory(root, 1, 5)
            self.write_investigation(root, 1, action="RUN_EXPERIMENT")
            sweep = sector_sweep_projection(root)[0]
            self.assertEqual(1, sweep["admitted"])
            self.assertEqual(1, sweep["runnable"])
            self.assertEqual(5, sweep["remaining"])

    def test_admitted_without_persisted_decision_is_unclassified(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_inventory(root, 1, 1)
            self.write_investigation(root, 1)
            sweep = sector_sweep_projection(root)[0]
            self.assertEqual(0, sweep["runnable"])
            self.assertEqual(1, sweep["unclassified"])

    def test_empty_inventory_has_finite_zero_progress(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_inventory(root, 1, 0)
            sweep = sector_sweep_projection(root)[0]
            self.assertEqual(
                (0, 0, 0.0), (sweep["complete"], sweep["remaining"], sweep["progress"])
            )

    def test_no_persisted_sweep_is_omitted_and_completed_is_retained(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual([], sector_sweep_projection(root))
            self.write_inventory(root, 1, 1)
            self.write_investigation(
                root, 1, status="COMPLETE", action="INVESTIGATION_COMPLETE"
            )
            self.assertEqual("COMPLETE", sector_sweep_projection(root)[0]["status"])
