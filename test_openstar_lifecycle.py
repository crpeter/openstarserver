import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_dispatch import InvestigationDispatcher
from openstar_external_jobs import ExternalDependency, ExternalJob, ExternalJobStore
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_lifecycle import InvestigationLifecycleLoop
from openstar_targets import InvestigationTarget, InvestigationTargetPortfolio
from openstar_workflow import (
    RetryableExecutionError,
    StageOutcome,
    StageRequest,
    WorkflowEngine,
)


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
        source = StaticTargetSource((self.initial,))
        self.loop(source=source).start(self.initial)

        planned = self.loop(source=source).run(max_transitions=1)
        self.assertEqual("LIFECYCLE_CHECKPOINT", planned.disposition)
        persisted = self.store.load(self.initial.investigation_id)
        self.assertEqual(
            "RUN_EXPERIMENT",
            persisted.metadata["controlState"]["schedulerAction"],
        )
        self.assertEqual([], self.executions)

        dispatched = self.loop(source=source).run(max_transitions=1)
        self.assertEqual("LIFECYCLE_CHECKPOINT", dispatched.disposition)
        self.assertEqual([("investigation-one", "experiment-1")], self.executions)

        completed = self.loop(source=source).run()
        self.assertEqual(
            "INVESTIGATION_COMPLETE_NO_NEXT_TARGET", completed.disposition
        )
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

    def test_retryable_failed_execution_appends_deterministic_attempt_and_continues(self):
        available = False
        executions = []

        def run_stage(investigation, request):
            executions.append(request.id)
            if not available:
                raise RetryableExecutionError("coordinator is unavailable")
            return StageOutcome(
                result={"computed": True},
                next_stage=StageRequest(
                    f"{int(request.id[:3]) + 1:03d}-interpret",
                    "test.interpret",
                    {},
                    request.id,
                ),
            )

        def interpret(investigation, request):
            return StageOutcome(
                result={"interpreted": True},
                next_stage=StageRequest(
                    f"{int(request.id[:3]) + 1:03d}-finalize",
                    "test.finalize",
                    {},
                    request.id,
                ),
            )

        self.workflow.register_handler("test.run", run_stage)
        self.workflow.register_handler("test.interpret", interpret)
        self.workflow.register_handler(
            "test.finalize", lambda investigation, request: StageOutcome({}, stop=True)
        )
        loop = self.loop(planners={"test": lambda investigation, target: ()})
        investigation = loop.start(self.initial)
        self.store.append_running_stage(
            investigation,
            InvestigationStage("002-prepare", "test.prepare", "RUNNING", None, {}),
        )
        investigation = self.store.load(investigation.id)
        prepared = self.store.build_terminal_stage(
            stage_id="002-prepare", handler_id="test.prepare", status="COMPLETE",
            triggered_by_stage_id=None, parameters={}, result={"prepared": True},
            error=None, software_id="test", software_version="1",
        )
        investigation = self.store.complete_current_stage(investigation, prepared)
        # A sparse historical id proves allocation uses the durable maximum,
        # rather than the failed stage's immediate successor.
        self.store.append_running_stage(
            investigation,
            InvestigationStage("009-audit", "test.audit", "RUNNING", None, {}),
        )
        investigation = self.store.load(investigation.id)
        audit = self.store.build_terminal_stage(
            stage_id="009-audit", handler_id="test.audit", status="COMPLETE",
            triggered_by_stage_id=None, parameters={}, result={}, error=None,
            software_id="test", software_version="1",
        )
        investigation = self.store.complete_current_stage(investigation, audit)

        with self.assertRaises(RetryableExecutionError):
            self.workflow.run(
                investigation, StageRequest("003-run", "test.run", {"project": "p"}),
                software_id="test", software_version="1",
            )
        failed_before = self.store.load(investigation.id).stages[-1]
        legacy_failure = replace(
            failed_before,
            failure_classification=None,
            error="CoordinatorClientError: Coordinator unavailable: connection refused",
        )
        self.assertEqual(
            legacy_failure,
            InvestigationLifecycleLoop._retryable_failure(
                replace(
                    self.store.load(investigation.id),
                    stages=(legacy_failure,),
                )
            ),
        )
        failed_file_before = self.store.stage_path_for(
            investigation.id, failed_before.id
        ).read_bytes()

        # Repeated recovery while the dependency is down appends a distinct,
        # deterministic failed attempt and never rewrites either predecessor.
        with self.assertRaises(RetryableExecutionError):
            loop.run()
        first_retry = self.store.load(investigation.id).stages[-1]
        self.assertEqual("010-run", first_retry.id)
        self.assertEqual("003-run", first_retry.triggered_by_stage_id)
        self.assertEqual(failed_file_before, self.store.stage_path_for(
            investigation.id, failed_before.id
        ).read_bytes())

        available = True
        result = loop.run()
        stages = self.store.load(investigation.id).stages
        self.assertEqual("INVESTIGATION_COMPLETE_NO_NEXT_TARGET", result.disposition)
        self.assertEqual(
            [("003-run", "FAILED"), ("010-run", "FAILED"),
             ("011-run", "COMPLETE"), ("012-interpret", "COMPLETE"),
             ("013-finalize", "COMPLETE")],
            [(stage.id, stage.status) for stage in stages if stage.handler_id.startswith("test.run")
             or stage.handler_id in {"test.interpret", "test.finalize"}],
        )
        self.assertEqual("010-run", stages[-3].triggered_by_stage_id)
        execution_count = len(executions)
        self.assertEqual("INVESTIGATION_COMPLETE_NO_NEXT_TARGET", loop.run().disposition)
        self.assertEqual(execution_count, len(executions))

    def test_non_retryable_failed_stage_is_not_automatically_retried(self):
        loop = self.loop(planners={"test": lambda investigation, target: ()})
        investigation = loop.start(self.initial)
        def fail_science(investigation, request):
            raise ValueError("bad data")

        self.workflow.register_handler("test.scientific-failure", fail_science)
        with self.assertRaises(ValueError):
            self.workflow.run(
                investigation,
                StageRequest("001-science", "test.scientific-failure", {}),
                software_id="test",
                software_version="1",
            )
        failed = self.store.load(investigation.id).stages[-1]
        self.assertEqual("NON_RETRYABLE", failed.failure_classification)
        result = loop.run(max_transitions=1)
        self.assertEqual("NONRETRYABLE_FAILURE_REQUIRES_ATTENTION", result.disposition)
        self.assertEqual(1, len(self.store.load(investigation.id).stages))

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
        planned = loop.run(max_transitions=1)  # persist ADVANCE_TO_NEXT_TARGET
        self.assertEqual("LIFECYCLE_CHECKPOINT", planned.disposition)
        self.assertEqual(
            "ADVANCE_TO_NEXT_TARGET",
            self.store.load(self.initial.investigation_id)
            .metadata["controlState"]["schedulerAction"],
        )
        self.assertFalse((self.root / "portfolio.json").exists())
        state = json.loads((self.root / "lifecycle.json").read_text(encoding="utf-8"))
        self.assertEqual("one", state["currentTarget"]["id"])

        held = self.loop(planners=planners).run(max_transitions=1)
        self.assertEqual("LIFECYCLE_CHECKPOINT", held.disposition)
        self.assertFalse((self.root / "portfolio.json").exists())
        state = json.loads((self.root / "lifecycle.json").read_text(encoding="utf-8"))
        self.assertEqual("one", state["currentTarget"]["id"])
        self.assertFalse(self.store.path_for(self.second.investigation_id).exists())

        advanced = self.loop(planners=planners).run(max_transitions=2)
        self.assertEqual("LIFECYCLE_CHECKPOINT", advanced.disposition)
        state = json.loads((self.root / "lifecycle.json").read_text(encoding="utf-8"))
        self.assertEqual("two", state["currentTarget"]["id"])
        self.assertEqual(
            [("investigation-two", "second-target-experiment")], self.executions
        )

        terminal = self.loop(planners=planners).run()
        self.assertEqual(
            "INVESTIGATION_COMPLETE_NO_NEXT_TARGET", terminal.disposition
        )
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

    def test_persisted_complete_checkpoints_before_portfolio_advance(self):
        planners = {"test": lambda investigation, target: ()}
        loop = self.loop(planners=planners)
        loop.start(self.initial)

        planned = loop.run(max_transitions=1)
        self.assertEqual("LIFECYCLE_CHECKPOINT", planned.disposition)
        self.assertEqual(
            "INVESTIGATION_COMPLETE",
            self.store.load(self.initial.investigation_id)
            .metadata["controlState"]["schedulerAction"],
        )

        held = self.loop(planners=planners).run(max_transitions=1)
        self.assertEqual("LIFECYCLE_CHECKPOINT", held.disposition)
        self.assertFalse((self.root / "portfolio.json").exists())
        state = json.loads((self.root / "lifecycle.json").read_text(encoding="utf-8"))
        self.assertEqual("one", state["currentTarget"]["id"])
        self.assertFalse(self.store.path_for(self.second.investigation_id).exists())

        advanced = self.loop(planners=planners).run(max_transitions=2)
        self.assertEqual("LIFECYCLE_CHECKPOINT", advanced.disposition)
        state = json.loads((self.root / "lifecycle.json").read_text(encoding="utf-8"))
        self.assertEqual("two", state["currentTarget"]["id"])

    def _all_external_wait_loop(self, state):
        def planner(investigation, target):
            dependency = f"external:{investigation.id}"
            return (ScientificBranch("collect", StageRequest("collect", "test.execute", {}),
                external_data=(ExternalDataDependency(dependency, False, "pending"),)),)
        jobs = ExternalJobStore(self.root / "external-jobs")
        for target in self.source.targets:
            job = ExternalJob.create(provider="test", investigation_id=target.investigation_id,
                trigger_stage_id="prepare", dependency_id=f"external:{target.investigation_id}",
                role="required")
            jobs.save_dependency(ExternalDependency(f"external:{target.investigation_id}",
                target.investigation_id, "prepare", "test", (job.id,)))
            jobs.save(replace(job, state=state, remoteTaskURL=f"task:{target.id}"))
        loop = self.loop(planners={"test": planner})
        loop.start(self.initial)
        return loop

    def test_all_targets_pending_externally_is_waiting_not_science_complete(self):
        result = self._all_external_wait_loop("QUEUED").run()
        self.assertEqual("NO_RUNNABLE_TARGETS_WAITING_EXTERNAL_DATA", result.disposition)
        self.assertEqual([], self.executions)

    def test_terminal_external_failure_requires_attention_not_science_complete(self):
        result = self._all_external_wait_loop("REMOTE_FAILED").run()
        self.assertEqual("EXTERNAL_JOB_FAILURE_REQUIRES_ATTENTION", result.disposition)
        self.assertEqual([], self.executions)


if __name__ == "__main__":
    unittest.main()
