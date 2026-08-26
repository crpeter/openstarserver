import json
import tempfile
import unittest
from pathlib import Path

from openstar_autonomy_supervisor import AutonomySupervisor
from openstar_external_jobs import ExternalJobStore
from openstar_investigation import InvestigationStore
from openstar_lifecycle import InvestigationSchedulingState
from openstar_scheduler import InvestigationScheduleOutcome, SchedulingRoundResult
from openstar_targets import InvestigationTarget


class _Monitor:
    def __init__(self):
        self.polls = 0

    def poll_due(self):
        self.polls += 1
        return ()


class _Scheduler:
    def __init__(self, store, results):
        self.store = store
        self.results = iter(results)

    def run_until_idle(self):
        return next(self.results)


class AutonomySupervisorTests(unittest.TestCase):
    def test_bounded_cycles_poll_sleep_and_atomically_report_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InvestigationStore(root / "investigations")
            good = store.create("good", "test", "1")
            bad = store.create("bad", "test", "1")
            target_good = InvestigationTarget("target-good", "good", "test", "1")
            target_bad = InvestigationTarget("target-bad", "bad", "test", "1")
            result = SchedulingRoundResult((
                InvestigationScheduleOutcome(target_good, good, InvestigationSchedulingState.COMPLETE),
                InvestigationScheduleOutcome(target_bad, bad, InvestigationSchedulingState.RECOVERY_REQUIRED),
            ), ("good",))
            monitor = _Monitor()
            sleeps = []
            supervisor = AutonomySupervisor(
                scheduler=_Scheduler(store, [result, result]),
                external_jobs=ExternalJobStore(root / "external-jobs"),
                monitor=monitor,
                heartbeat_path=root / "autonomy-heartbeat.json",
                interval_seconds=2.5,
                sleep=sleeps.append,
            )

            self.assertEqual(0, supervisor.run(max_cycles=2, install_signal_handlers=False))
            heartbeat = json.loads((root / "autonomy-heartbeat.json").read_text())
            self.assertEqual(2, monitor.polls)
            self.assertEqual([2.5], sleeps)
            self.assertEqual(2, heartbeat["cycleNumber"])
            self.assertEqual(["bad"], heartbeat["quarantinedInvestigationIDs"])
            self.assertEqual(1, heartbeat["countsBySchedulerState"]["COMPLETE"])
            self.assertEqual(1, heartbeat["countsBySchedulerState"]["RECOVERY_REQUIRED"])
            self.assertEqual(["good"], heartbeat["lastCycleDispatchedInvestigationIDs"])

    def test_heartbeat_is_overwritten_from_durable_cycle_not_trusted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat = root / "autonomy-heartbeat.json"
            heartbeat.write_text('{"quarantinedInvestigationIDs":["fabricated"]}')
            store = InvestigationStore(root / "investigations")
            current = store.create("current", "test", "1")
            target = InvestigationTarget("target", "current", "test", "1")
            result = SchedulingRoundResult((InvestigationScheduleOutcome(
                target, current, InvestigationSchedulingState.COMPLETE),), ())
            supervisor = AutonomySupervisor(
                scheduler=_Scheduler(store, [result]),
                external_jobs=ExternalJobStore(root / "external-jobs"),
                monitor=_Monitor(), heartbeat_path=heartbeat, interval_seconds=1,
            )
            supervisor.run(max_cycles=1, install_signal_handlers=False)
            payload = json.loads(heartbeat.read_text())
            self.assertEqual([], payload["quarantinedInvestigationIDs"])
            self.assertEqual("current", payload["investigations"][0]["investigationID"])


if __name__ == "__main__":
    unittest.main()
