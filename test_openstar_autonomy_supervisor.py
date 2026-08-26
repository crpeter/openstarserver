import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_autonomy_supervisor import AutonomySupervisor
from openstar_dispatch import InvestigationDispatcher
from openstar_external_jobs import (
    ExternalDependency,
    ExternalJob,
    ExternalJobMonitor,
    ExternalJobStore,
    PollResult,
)
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_lifecycle import InvestigationSchedulingState
from openstar_scheduler import (
    InvestigationScheduler,
    InvestigationScheduleOutcome,
    SchedulingRoundResult,
)
from openstar_targets import InvestigationTarget
from openstar_workflow import (
    RetryableExecutionError,
    StageOutcome,
    StageRequest,
    WorkflowEngine,
)


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


class _Source:
    id = "soak.targets"
    version = "1"

    def __init__(self, targets):
        self.targets = targets

    def enumerate_targets(self):
        return tuple(self.targets)


class _ProgressingProvider:
    """Remain queued once, then durably complete the same remote job."""

    def __init__(self):
        self.calls = []

    def poll(self, job):
        self.calls.append(job.id)
        if len(self.calls) == 1:
            return PollResult("QUEUED")
        return PollResult("COMPLETE", "result://external-data")


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


class MixedPortfolioSupervisorSoakTests(unittest.TestCase):
    """Exercise the supervisor through the real durable scheduling contracts."""

    def test_mixed_portfolio_survives_failures_waits_orphan_and_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            investigations = InvestigationStore(root / "investigations")
            external_jobs = ExternalJobStore(root / "external-jobs")
            workflow = WorkflowEngine(investigations)
            dispatcher = InvestigationDispatcher(investigations, workflow)
            executions = {name: [] for name in (
                "healthy", "transient", "external", "hard", "orphan", "planner-bug"
            )}
            external_handler_control_actions = []
            planner_bug_attempts = []

            def healthy(investigation, request):
                executions["healthy"].append(request.id)
                if request.id == "001-healthy-first":
                    return StageOutcome(
                        {"step": 1},
                        next_stage=StageRequest(
                            "002-healthy-second", "soak.healthy", {}
                        ),
                    )
                return StageOutcome({"step": 2}, stop=True)

            def transient(investigation, request):
                executions["transient"].append(request.id)
                if len(executions["transient"]) == 1:
                    raise RetryableExecutionError("temporary coordinator outage")
                return StageOutcome({"recovered": True}, stop=True)

            def external(investigation, request):
                executions["external"].append(request.id)
                external_handler_control_actions.append(
                    investigation.metadata.get("controlState", {}).get("schedulerAction")
                )
                return StageOutcome({"received": True}, stop=True)

            def hard(investigation, request):
                executions["hard"].append(request.id)
                raise ValueError("invalid scientific input")

            workflow.register_handler("soak.healthy", healthy)
            workflow.register_handler("soak.transient", transient)
            workflow.register_handler("soak.external", external)
            workflow.register_handler("soak.hard", hard)

            dependency_id = "external:external"

            def planner(investigation, target):
                if investigation.status == "COMPLETE":
                    return ()
                if target.id == "healthy":
                    if investigation.stages:
                        return ()
                    request = StageRequest("001-healthy-first", "soak.healthy", {})
                elif target.id == "transient":
                    request = StageRequest("001-transient", "soak.transient", {})
                elif target.id == "external":
                    available = bool(
                        (investigation.metadata.get("externalDataAvailability") or {})
                        .get(dependency_id)
                    )
                    return (ScientificBranch(
                        "external-data",
                        StageRequest("001-external", "soak.external", {}),
                        external_data=(ExternalDataDependency(
                            dependency_id, available, "pending remote data"
                        ),),
                    ),)
                elif target.id == "hard":
                    request = StageRequest("001-hard", "soak.hard", {})
                elif target.id == "planner-bug":
                    planner_bug_attempts.append(target.id)
                    raise RuntimeError("planner bug")
                else:
                    return ()
                return (ScientificBranch(target.id, request),)

            targets = tuple(
                InvestigationTarget(name, name, "soak.workflow", "1")
                for name in (
                    "healthy", "transient", "external", "hard", "orphan", "planner-bug"
                )
            )
            scheduler = InvestigationScheduler(
                investigations,
                dispatcher,
                _Source(targets),
                {"soak.workflow": planner},
                software_id="soak",
                software_version="1",
                max_concurrent_investigations=1,
            )

            # Simulate exactly one external submission before supervision. Its
            # stable durable identity makes duplicate submission observable.
            job = ExternalJob.create(
                provider="soak-provider",
                investigation_id="external",
                trigger_stage_id="000-submit",
                dependency_id=dependency_id,
                role="required",
            )
            external_jobs.save_dependency(ExternalDependency(
                dependency_id, "external", "000-submit", "soak-provider", (job.id,)
            ))
            external_jobs.save(replace(job, remoteTaskURL="task://one"))

            orphan = scheduler.driver.attach(
                next(target for target in targets if target.id == "orphan")
            )
            investigations.append_running_stage(orphan, InvestigationStage(
                "001-orphan", "soak.orphan", "RUNNING", None, {}
            ))
            provider = _ProgressingProvider()
            monitor = ExternalJobMonitor(
                external_jobs, {"soak-provider": provider}, interval_seconds=0
            )
            heartbeat_path = root / "autonomy-heartbeat.json"
            supervisor = AutonomySupervisor(
                scheduler=scheduler,
                external_jobs=external_jobs,
                monitor=monitor,
                heartbeat_path=heartbeat_path,
                interval_seconds=1,
                sleep=lambda _seconds: None,
            )

            supervisor.run_cycle()
            self.assertEqual([job.id], provider.calls)
            self.assertEqual([], executions["external"])
            self.assertEqual("QUEUED", external_jobs.load(job.id).state)
            waiting_external = investigations.load("external")
            self.assertEqual("QUIESCENT_AWAITING_DATA", waiting_external.status)
            self.assertEqual(
                "ADVANCE_TO_NEXT_TARGET",
                waiting_external.metadata["controlState"]["schedulerAction"],
            )
            first_failed_path = investigations.stage_path_for(
                "transient", "001-transient"
            )
            first_failed_evidence = first_failed_path.read_bytes()

            supervisor.run(max_cycles=3, install_signal_handlers=False)

            healthy_record = investigations.load("healthy")
            transient_record = investigations.load("transient")
            external_record = investigations.load("external")
            hard_record = investigations.load("hard")
            orphan_record = investigations.load("orphan")
            planner_bug_record = investigations.load("planner-bug")
            self.assertEqual("COMPLETE", healthy_record.status)
            self.assertEqual(
                ["001-healthy-first", "002-healthy-second"],
                [stage.id for stage in healthy_record.stages],
            )
            self.assertEqual("COMPLETE", transient_record.status)
            self.assertEqual(
                [("001-transient", "FAILED"), ("002-transient", "COMPLETE")],
                [(stage.id, stage.status) for stage in transient_record.stages],
            )
            self.assertEqual(first_failed_evidence, first_failed_path.read_bytes())
            self.assertEqual(
                "TRANSIENT_INFRASTRUCTURE",
                transient_record.stages[0].failure_classification,
            )
            self.assertEqual("001-transient", transient_record.stages[1].triggered_by_stage_id)
            self.assertEqual("COMPLETE", external_record.status)
            self.assertEqual(["001-external"], executions["external"])
            self.assertTrue(
                external_record.metadata["externalDataAvailability"][dependency_id]
            )
            # The stale ADVANCE_TO_NEXT_TARGET wait was cleared by the wakeup;
            # the handler sees only the freshly planned runnable decision.
            self.assertEqual(["RUN_EXPERIMENT"], external_handler_control_actions)
            self.assertEqual("COMPLETE", external_jobs.load(job.id).state)
            self.assertEqual([job.id, job.id], provider.calls)
            self.assertEqual(1, len(external_jobs.list()))
            self.assertEqual("FAILED", hard_record.status)
            self.assertEqual("NON_RETRYABLE", hard_record.stages[-1].failure_classification)
            self.assertEqual(["001-hard"], executions["hard"])
            self.assertEqual("RUNNING", orphan_record.status)
            self.assertEqual(["001-orphan"], [stage.id for stage in orphan_record.stages])
            self.assertEqual([], executions["orphan"])
            self.assertEqual("RUNNING", planner_bug_record.status)
            self.assertEqual((), planner_bug_record.stages)
            self.assertEqual(["planner-bug"], planner_bug_attempts)

            heartbeat = json.loads(heartbeat_path.read_text())
            self.assertEqual(
                ["hard", "orphan", "planner-bug"],
                heartbeat["quarantinedInvestigationIDs"],
            )
            self.assertEqual(3, heartbeat["countsBySchedulerState"]["COMPLETE"])
            self.assertEqual(2, heartbeat["countsBySchedulerState"]["FAILED"])
            self.assertEqual(1, heartbeat["countsBySchedulerState"]["RECOVERY_REQUIRED"])
            planner_summary = next(
                item for item in heartbeat["investigations"]
                if item["investigationID"] == "planner-bug"
            )
            self.assertEqual("RUNNING", planner_summary["status"])
            self.assertEqual("FAILED", planner_summary["schedulerState"])
            self.assertEqual("RuntimeError: planner bug", planner_summary["error"])
            self.assertIsNone(planner_summary["latestStageID"])

            stage_snapshots = {
                name: [stage.id for stage in investigations.load(name).stages]
                for name in (
                    "healthy", "transient", "external", "hard", "orphan", "planner-bug"
                )
            }
            heartbeat_path.write_text(
                '{"quarantinedInvestigationIDs": ["healthy"], "cycleNumber": 999}'
            )

            # Restart every durable contract, not the heartbeat, and run later
            # idle cycles to prove there is no replay or duplicate submission.
            restarted_store = InvestigationStore(investigations.root)
            restarted_jobs = ExternalJobStore(external_jobs.root)
            restarted_workflow = WorkflowEngine(restarted_store)
            restarted_workflow.register_handler("soak.healthy", healthy)
            restarted_workflow.register_handler("soak.transient", transient)
            restarted_workflow.register_handler("soak.external", external)
            restarted_workflow.register_handler("soak.hard", hard)
            restarted_scheduler = InvestigationScheduler(
                restarted_store,
                InvestigationDispatcher(restarted_store, restarted_workflow),
                _Source(targets),
                {"soak.workflow": planner},
                software_id="soak",
                software_version="1",
                max_concurrent_investigations=1,
            )
            restarted_supervisor = AutonomySupervisor(
                scheduler=restarted_scheduler,
                external_jobs=restarted_jobs,
                monitor=ExternalJobMonitor(
                    restarted_jobs, {"soak-provider": provider}, interval_seconds=0
                ),
                heartbeat_path=heartbeat_path,
                interval_seconds=1,
                sleep=lambda _seconds: None,
            )
            restarted_supervisor.run(max_cycles=2, install_signal_handlers=False)

            self.assertEqual(first_failed_evidence, first_failed_path.read_bytes())
            self.assertEqual(1, len(restarted_jobs.list()))
            self.assertEqual([job.id, job.id], provider.calls)
            self.assertEqual(["planner-bug", "planner-bug"], planner_bug_attempts)
            self.assertEqual(
                stage_snapshots,
                {
                    name: [stage.id for stage in restarted_store.load(name).stages]
                    for name in stage_snapshots
                },
            )
            restarted_heartbeat = json.loads(heartbeat_path.read_text())
            self.assertEqual(
                ["hard", "orphan", "planner-bug"],
                restarted_heartbeat["quarantinedInvestigationIDs"],
            )
            self.assertEqual(2, restarted_heartbeat["countsBySchedulerState"]["FAILED"])
            self.assertEqual(
                1,
                restarted_heartbeat["countsBySchedulerState"]["RECOVERY_REQUIRED"],
            )


if __name__ == "__main__":
    unittest.main()
