import tempfile
import unittest
from pathlib import Path

from openstar_autonomy import AutonomousInvestigationEngine, ScientificBranch
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine


class InvestigationDispatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.store = InvestigationStore(Path(self.temporary_directory.name))
        self.workflow = WorkflowEngine(self.store)
        self.dispatcher = InvestigationDispatcher(self.store, self.workflow)
        self.investigation = self.store.create("dispatch-test", "test", "1")
        self.executions = 0

        def execute(investigation, request):
            self.executions += 1
            return StageOutcome(result={"value": 42}, stop=True)

        self.workflow.register_handler("test.execute", execute)

    def persist_action(self, action, *, status="RUNNING", selected=None):
        return self.store.set_control_state(
            self.investigation,
            status=status,
            control_state={
                "branchAssessments": [],
                "selectedExperiment": selected,
                "schedulerAction": action,
            },
        )

    def dispatch(self):
        return self.dispatcher.dispatch(
            self.investigation.id,
            software_id="test",
            software_version="1",
        )

    def test_run_experiment_reloads_and_uses_workflow_engine(self):
        request = StageRequest("001-experiment", "test.execute", {"seed": 7})
        autonomy = AutonomousInvestigationEngine(self.store)
        self.investigation, _ = autonomy.decide(
            self.investigation,
            (ScientificBranch("experiment", request),),
        )

        restarted_dispatcher = InvestigationDispatcher(self.store, self.workflow)
        result = restarted_dispatcher.dispatch(
            self.investigation.id,
            software_id="test",
            software_version="1",
        )

        self.assertEqual("EXPERIMENT_DISPATCHED", result.disposition)
        self.assertEqual(1, self.executions)
        reloaded = self.store.load(self.investigation.id)
        self.assertEqual("COMPLETE", reloaded.status)
        self.assertEqual("001-experiment", reloaded.stages[-1].id)

    def test_restart_does_not_execute_completed_selection_twice(self):
        selected = {
            "id": "001-experiment",
            "handler_id": "test.execute",
            "parameters": {},
            "triggered_by_stage_id": None,
        }
        self.investigation = self.persist_action("RUN_EXPERIMENT", selected=selected)
        self.assertEqual("EXPERIMENT_DISPATCHED", self.dispatch().disposition)

        restarted_dispatcher = InvestigationDispatcher(self.store, self.workflow)
        restarted = restarted_dispatcher.dispatch(
            self.investigation.id,
            software_id="test",
            software_version="1",
        )

        self.assertEqual("EXPERIMENT_ALREADY_DISPATCHED", restarted.disposition)
        self.assertEqual(1, self.executions)

    def test_restart_does_not_rerun_interrupted_stage(self):
        selected = {
            "id": "001-experiment",
            "handler_id": "test.execute",
            "parameters": {},
            "triggered_by_stage_id": None,
        }
        self.investigation = self.persist_action("RUN_EXPERIMENT", selected=selected)
        self.investigation = self.store.append_running_stage(
            self.investigation,
            InvestigationStage(
                id="001-experiment",
                handler_id="test.execute",
                status="RUNNING",
                triggered_by_stage_id=None,
                parameters={},
            ),
        )

        result = self.dispatch()

        self.assertEqual("EXPERIMENT_RECOVERY_REQUIRED", result.disposition)
        self.assertEqual(0, self.executions)

    def test_non_execution_actions_preserve_investigation(self):
        cases = (
            ("WAIT_FOR_PREREQUISITES", "BLOCKED", "WAITING_FOR_PREREQUISITES", False),
            ("INVESTIGATION_COMPLETE", "COMPLETE", "INVESTIGATION_TERMINAL", False),
            (
                "ADVANCE_TO_NEXT_TARGET",
                "QUIESCENT_AWAITING_DATA",
                "NEXT_TARGET_SELECTION_REQUIRED",
                True,
            ),
        )
        for action, status, disposition, next_target_required in cases:
            with self.subTest(action=action):
                self.investigation = self.persist_action(action, status=status)
                before = self.store.load(self.investigation.id)

                result = self.dispatch()

                after = self.store.load(self.investigation.id)
                self.assertEqual(disposition, result.disposition)
                self.assertEqual(next_target_required, result.next_target_required)
                self.assertEqual(before, after)
                self.assertEqual(0, self.executions)


if __name__ == "__main__":
    unittest.main()
