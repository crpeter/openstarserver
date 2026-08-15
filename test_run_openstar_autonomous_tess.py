import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine
import run_openstar_autonomous_tess as runner


class AutonomousTessEntryPointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project.json"
        self.write_project(
            [
                {"id": "first", "ticID": 1},
                {"id": "second", "ticID": 2},
            ]
        )

    def write_project(self, datasets):
        self.project.write_text(
            json.dumps({"id": "validation", "datasets": datasets}),
            encoding="utf-8",
        )

    @staticmethod
    def workflow(store, coordinator, *, poll_interval, timeout):
        engine = WorkflowEngine(store)

        def execute(investigation, request):
            return StageOutcome(
                result={"target": investigation.metadata["datasetID"]}, stop=True
            )

        engine.register_handler("test.execute", execute)
        return engine

    @staticmethod
    def complete_planner(investigation, target):
        return ()

    def invoke(self, planner=None):
        output = io.StringIO()
        with (
            patch.object(runner, "register_tess_workflow_handlers", self.workflow),
            patch.object(runner, "plan_tess_branches", planner or self.complete_planner),
            contextlib.redirect_stdout(output),
        ):
            code = runner.run_autonomous_tess(
                [self.project], "http://coordinator.test", self.root / "state"
            )
        return code, output.getvalue()

    def test_fresh_startup_selects_first_eligible_target(self):
        code, output = self.invoke()
        self.assertEqual(0, code)
        self.assertIn("disposition=STARTED target=validation:first", output)
        self.assertTrue(
            (self.root / "state/investigations/tess-validation-first/investigation.json").exists()
        )

    def test_restart_resumes_persisted_lifecycle(self):
        def waiting(investigation, target):
            return (
                ScientificBranch(
                    "wait",
                    StageRequest("wait", "test.execute", {}),
                    required_stage_ids=("prerequisite",),
                ),
            )

        self.invoke(waiting)
        code, output = self.invoke(waiting)
        self.assertEqual(0, code)
        self.assertIn("disposition=RESUMING target=validation:first", output)
        self.assertNotIn("disposition=STARTED", output)

    def test_advances_to_another_target(self):
        def planner(investigation, target):
            if target.id.endswith(":first"):
                return (
                    ScientificBranch(
                        "unavailable",
                        StageRequest("unavailable", "test.execute", {}),
                        external_data=(ExternalDataDependency("future", False),),
                    ),
                )
            return ()

        code, output = self.invoke(planner)
        self.assertEqual(0, code)
        state = json.loads((self.root / "state/lifecycle.json").read_text())
        self.assertEqual("validation:second", state["currentTarget"]["id"])
        self.assertIn("investigation=tess-validation-second", output)

    def test_completed_target_advances_and_completes_next_target_in_one_run(self):
        def planner(investigation, target):
            if investigation.stages:
                return ()
            return (
                ScientificBranch(
                    "execute",
                    StageRequest("execute", "test.execute", {}),
                ),
            )

        code, output = self.invoke(planner)

        self.assertEqual(0, code)
        store = InvestigationStore(self.root / "state/investigations")
        first = store.load("tess-validation-first")
        second = store.load("tess-validation-second")
        self.assertEqual("COMPLETE", first.status)
        self.assertEqual("COMPLETE", second.status)
        self.assertEqual(["execute"], [stage.id for stage in first.stages])
        self.assertEqual(["execute"], [stage.id for stage in second.stages])
        self.assertIn("disposition=INVESTIGATION_COMPLETE_NO_NEXT_TARGET", output)
        portfolio = json.loads((self.root / "state/portfolio.json").read_text())
        self.assertEqual(
            ["validation:first", "validation:second"],
            [item["selectedTargetID"] for item in portfolio["selections"]],
        )

    def test_no_eligible_targets_exits_cleanly(self):
        self.write_project([{"id": "disabled", "autonomousEligible": False}])
        code, output = self.invoke()
        self.assertEqual(0, code)
        self.assertIn("disposition=NO_ELIGIBLE_TARGETS", output)
        self.assertFalse((self.root / "state/lifecycle.json").exists())

    def test_recovery_required_returns_distinct_exit_code(self):
        def waiting(investigation, target):
            return (
                ScientificBranch(
                    "wait",
                    StageRequest("wait", "test.execute", {}),
                    required_stage_ids=("prerequisite",),
                ),
            )

        self.invoke(waiting)
        store = InvestigationStore(self.root / "state/investigations")
        investigation = store.load("tess-validation-first")
        store.append_running_stage(
            investigation,
            InvestigationStage("interrupted", "test.execute", "RUNNING", None, {}),
        )
        code, output = self.invoke(waiting)
        self.assertEqual(2, code)
        self.assertIn("disposition=EXPERIMENT_RECOVERY_REQUIRED", output)


if __name__ == "__main__":
    unittest.main()
