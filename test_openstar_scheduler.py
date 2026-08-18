import tempfile
import threading
import unittest
from pathlib import Path

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStore
from openstar_lifecycle import InvestigationSchedulingState
from openstar_scheduler import InvestigationScheduler
from openstar_targets import InvestigationTarget
from openstar_workflow import (
    RetryableExecutionError,
    StageOutcome,
    StageRequest,
    WorkflowEngine,
)


class Source:
    id = "test.targets"
    version = "1"

    def __init__(self, targets):
        self.targets = targets

    def enumerate_targets(self):
        return tuple(self.targets)


class InvestigationSchedulerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = InvestigationStore(self.root / "investigations")
        self.workflow = WorkflowEngine(self.store)
        self.dispatcher = InvestigationDispatcher(self.store, self.workflow)

    @staticmethod
    def target(target_id, *, priority=0, eligible=True, investigation_id=None):
        return InvestigationTarget(
            target_id,
            investigation_id or f"investigation-{target_id}",
            "test.workflow",
            "1",
            priority,
            eligible,
        )

    def scheduler(self, targets, planner, **kwargs):
        return InvestigationScheduler(
            self.store,
            self.dispatcher,
            Source(targets),
            {"test.workflow": planner},
            software_id="test",
            software_version="1",
            **kwargs,
        )

    def test_two_investigations_dispatch_concurrently_in_submission_order(self):
        both_started = threading.Event()
        release = threading.Event()
        lock = threading.Lock()
        starts = []

        def execute(investigation, request):
            with lock:
                starts.append(investigation.id)
                if len(starts) == 2:
                    both_started.set()
            self.assertTrue(both_started.wait(2))
            self.assertTrue(release.wait(2))
            return StageOutcome({}, stop=True)

        self.workflow.register_handler("execute", execute)

        def planner(investigation, target):
            if investigation.stages:
                return ()
            return (ScientificBranch("run", StageRequest("001-run", "execute", {})),)

        scheduler = self.scheduler(
            [self.target("b", priority=2), self.target("a", priority=1)], planner
        )
        result_holder = []
        thread = threading.Thread(
            target=lambda: result_holder.append(scheduler.run_round())
        )
        thread.start()
        self.assertTrue(both_started.wait(2))
        self.assertEqual({"investigation-a", "investigation-b"}, set(starts))
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(
            ("investigation-a", "investigation-b"),
            result_holder[0].dispatched_investigation_ids,
        )

    def test_waiting_and_complete_do_not_block_runnable(self):
        ran = []
        self.workflow.register_handler(
            "execute",
            lambda investigation, request: (
                ran.append(investigation.id) or StageOutcome({}, stop=True)
            ),
        )

        def planner(investigation, target):
            if target.id == "waiting":
                return (
                    ScientificBranch(
                        "wait",
                        StageRequest("wait", "execute", {}),
                        external_data=(ExternalDataDependency("remote", False),),
                    ),
                )
            if target.id == "complete" or investigation.stages:
                return ()
            return (ScientificBranch("run", StageRequest("run", "execute", {})),)

        result = self.scheduler(
            [self.target("waiting"), self.target("complete"), self.target("runnable")],
            planner,
        ).run_until_idle()
        self.assertEqual(["investigation-runnable"], ran)
        states = {item.target.id: item.state for item in result.outcomes}
        self.assertEqual(
            InvestigationSchedulingState.WAITING_EXTERNAL_DATA, states["waiting"]
        )
        self.assertEqual(InvestigationSchedulingState.COMPLETE, states["complete"])

    def test_admission_rejects_duplicates_and_ignores_ineligible(self):
        def planner(investigation, target):
            return ()

        with self.assertRaisesRegex(ValueError, "duplicate target ID"):
            self.scheduler(
                [self.target("same"), self.target("same")], planner
            ).run_round()
        with self.assertRaisesRegex(ValueError, "duplicate investigation ID"):
            self.scheduler(
                [
                    self.target("a", investigation_id="shared"),
                    self.target("b", investigation_id="shared"),
                ],
                planner,
            ).run_round()
        result = self.scheduler(
            [self.target("disabled", eligible=False)], planner
        ).run_round()
        self.assertEqual((), result.outcomes)
        self.assertFalse(self.store.path_for("investigation-disabled").exists())

    def test_restart_uses_existing_terminal_history(self):
        self.workflow.register_handler(
            "execute", lambda investigation, request: StageOutcome({}, stop=True)
        )

        def planner(investigation, target):
            if investigation.stages:
                return ()
            return (ScientificBranch("run", StageRequest("run", "execute", {})),)

        targets = [self.target("target")]
        self.scheduler(targets, planner).run_until_idle()
        self.scheduler(targets, planner).run_until_idle()
        investigation = self.store.load("investigation-target")
        self.assertEqual(["run"], [stage.id for stage in investigation.stages])

    def test_retryable_failure_is_deferred_while_other_work_continues(self):
        executions = {"a": 0, "b": 0}

        def execute(investigation, request):
            target_id = investigation.metadata["targetID"]
            executions[target_id] += 1
            if target_id == "a":
                raise RetryableExecutionError("coordinator unavailable")
            return StageOutcome({}, stop=True)

        self.workflow.register_handler("execute", execute)

        def planner(investigation, target):
            if target.id == "a":
                return (ScientificBranch("a", StageRequest("001-run", "execute", {})),)
            completed = sum(
                stage.status == "COMPLETE" for stage in investigation.stages
            )
            if completed >= 2:
                return ()
            return (
                ScientificBranch(
                    f"b-{completed + 1}",
                    StageRequest(f"00{completed + 1}-run", "execute", {}),
                ),
            )

        targets = [
            InvestigationTarget(
                "a",
                "investigation-a",
                "test.workflow",
                "1",
                metadata={"targetID": "a"},
            ),
            InvestigationTarget(
                "b",
                "investigation-b",
                "test.workflow",
                "1",
                metadata={"targetID": "b"},
            ),
        ]
        scheduler = self.scheduler(targets, planner)
        result = scheduler.run_until_idle()

        self.assertEqual({"a": 1, "b": 2}, executions)
        a = self.store.load("investigation-a")
        self.assertEqual("FAILED", a.stages[-1].status)
        self.assertEqual(
            "TRANSIENT_INFRASTRUCTURE", a.stages[-1].failure_classification
        )
        self.assertEqual(
            "RUN_EXPERIMENT", a.metadata["controlState"]["schedulerAction"]
        )
        self.assertEqual(
            a.stages[-1].id,
            a.metadata["controlState"]["selectedExperiment"]["triggered_by_stage_id"],
        )
        self.assertEqual("COMPLETE", self.store.load("investigation-b").status)
        self.assertTrue(any(outcome.error is not None for outcome in result.outcomes))

        scheduler.run_until_idle()
        self.assertEqual(2, executions["a"])


if __name__ == "__main__":
    unittest.main()
