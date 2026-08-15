import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_lifecycle import InvestigationLifecycleLoop
from openstar_targets import InvestigationTarget, InvestigationTargetPortfolio
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine


@dataclass(frozen=True)
class StaticTargetSource:
    targets: tuple[InvestigationTarget, ...]
    id: str = "lifecycle.targets"
    version: str = "1"

    def enumerate_targets(self):
        return self.targets


class InvestigationLifecycleLoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = InvestigationStore(self.root / "investigations")
        self.workflow = WorkflowEngine(self.store)
        self.dispatcher = InvestigationDispatcher(self.store, self.workflow)
        self.executions = []
        self.plans = []

        def execute(investigation, request):
            self.executions.append((investigation.id, request.id))
            return StageOutcome(result={"stage": request.id}, stop=True)

        self.workflow.register_handler("test.execute", execute)
        self.initial = InvestigationTarget("one", "investigation-one", "test", "1")
        self.second = InvestigationTarget("two", "investigation-two", "test", "1")
        self.source = StaticTargetSource((self.initial, self.second))
        self.planner = self._sequential_planner
        self.planners = {"test": self.planner}

    def _sequential_planner(self, investigation, target):
        completed = {
            stage.id
            for stage in investigation.stages
            if stage.status == "COMPLETE"
        }
        self.plans.append((investigation.id, tuple(sorted(completed))))
        if "experiment-1" not in completed:
            stage_id = "experiment-1"
        elif "experiment-2" not in completed:
            stage_id = "experiment-2"
        else:
            return ()
        return (
            ScientificBranch(
                stage_id,
                StageRequest(stage_id, "test.execute", {"targetID": target.id}),
            ),
        )

    def loop(self, planners=None, source=None):
        return InvestigationLifecycleLoop(
            self.root / "lifecycle.json",
            self.store,
            InvestigationDispatcher(self.store, self.workflow),
            InvestigationTargetPortfolio(
                self.root / "portfolio.json", self.store, self.dispatcher
            ),
            source or self.source,
            planners or self.planners,
            software_id="test-suite",
            software_version="1",
        )

    def test_restart_between_plan_dispatch_and_replan_is_idempotent(self):
        self.loop().start(self.initial)

        planned = self.loop().run(max_transitions=1)
        self.assertEqual("LIFECYCLE_CHECKPOINT", planned.disposition)
        persisted = self.store.load(self.initial.investigation_id)
        self.assertEqual(
            "RUN_EXPERIMENT",
            persisted.metadata["controlState"]["schedulerAction"],
        )
        self.assertEqual([], self.executions)

        dispatched = self.loop().run(max_transitions=1)
        self.assertEqual("LIFECYCLE_CHECKPOINT", dispatched.disposition)
        self.assertEqual([("investigation-one", "experiment-1")], self.executions)

        completed = self.loop().run()
        self.assertEqual("INVESTIGATION_COMPLETE", completed.disposition)
        self.assertEqual(
            [
                ("investigation-one", "experiment-1"),
                ("investigation-one", "experiment-2"),
            ],
            self.executions,
        )
        self.assertEqual("COMPLETE", self.store.load("investigation-one").status)
        self.assertEqual(3, len(self.plans))

    def test_wait_returns_without_replanning_or_spinning(self):
        def waiting_planner(investigation, target):
            self.plans.append(investigation.id)
            return (
                ScientificBranch(
                    "later",
                    StageRequest("later", "test.execute", {}),
                    required_stage_ids=("external-prerequisite",),
                ),
            )

        loop = self.loop(planners={"test": waiting_planner})
        loop.start(self.initial)

        first = loop.run()
        restarted = self.loop(planners={"test": waiting_planner}).run()

        self.assertEqual("WAITING_FOR_PREREQUISITES", first.disposition)
        self.assertEqual("WAITING_FOR_PREREQUISITES", restarted.disposition)
        self.assertEqual(["investigation-one"], self.plans)
        self.assertEqual([], self.executions)

    def test_restart_never_reruns_interrupted_running_experiment(self):
        loop = self.loop()
        investigation = loop.start(self.initial)
        loop.run(max_transitions=1)
        investigation = self.store.load(investigation.id)
        self.store.append_running_stage(
            investigation,
            InvestigationStage(
                id="experiment-1",
                handler_id="test.execute",
                status="RUNNING",
                triggered_by_stage_id=None,
                parameters={"targetID": "one"},
            ),
        )

        result = self.loop().run()

        self.assertEqual("EXPERIMENT_RECOVERY_REQUIRED", result.disposition)
        self.assertEqual([], self.executions)
        self.assertEqual("RUNNING", self.store.load(investigation.id).stages[-1].status)

    def test_target_advance_is_durable_and_continues_same_lifecycle(self):
        def advancing_planner(investigation, target):
            self.plans.append(investigation.id)
            if target.id == "one":
                return (
                    ScientificBranch(
                        "unavailable",
                        StageRequest("unavailable", "test.execute", {}),
                        external_data=(
                            ExternalDataDependency(
                                "future-input", False, "not available"
                            ),
                        ),
                    ),
                )
            if investigation.stages:
                return ()
            return (
                ScientificBranch(
                    "second-target-experiment",
                    StageRequest("second-target-experiment", "test.execute", {}),
                ),
            )

        planners = {"test": advancing_planner}
        loop = self.loop(planners=planners)
        loop.start(self.initial)
        loop.run(max_transitions=1)  # persist ADVANCE_TO_NEXT_TARGET

        advanced = self.loop(planners=planners).run(max_transitions=2)
        self.assertEqual("LIFECYCLE_CHECKPOINT", advanced.disposition)
        state = json.loads((self.root / "lifecycle.json").read_text(encoding="utf-8"))
        self.assertEqual("two", state["currentTarget"]["id"])
        self.assertEqual(
            [("investigation-two", "second-target-experiment")], self.executions
        )

        terminal = self.loop(planners=planners).run()
        self.assertEqual("INVESTIGATION_COMPLETE", terminal.disposition)
        self.assertEqual("investigation-two", terminal.investigation.id)
        self.assertEqual(
            "two",
            self.store.load("investigation-two")
            .metadata["targetSelection"]["selectedTargetID"],
        )
        portfolio = json.loads(
            (self.root / "portfolio.json").read_text(encoding="utf-8")
        )
        self.assertEqual(1, len(portfolio["selections"]))


if __name__ == "__main__":
    unittest.main()
