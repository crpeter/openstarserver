import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from openstar_autonomy import ScientificBranch
from openstar_dispatch import DispatchResult, InvestigationDispatcher
from openstar_investigation import InvestigationStore
from openstar_targets import InvestigationTarget, InvestigationTargetPortfolio
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine


@dataclass(frozen=True)
class StaticTargetSource:
    targets: tuple[InvestigationTarget, ...]
    id: str = "test.targets"
    version: str = "1"

    def enumerate_targets(self):
        return self.targets


class InvestigationTargetPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.store = InvestigationStore(root / "investigations")
        self.workflow = WorkflowEngine(self.store)
        self.dispatcher = InvestigationDispatcher(self.store, self.workflow)
        self.portfolio_path = root / "portfolio.json"
        self.executions = []
        self.plans = []

        def execute(investigation, request):
            self.executions.append(investigation.id)
            return StageOutcome(result={"target": investigation.id}, stop=True)

        self.workflow.register_handler("test.execute", execute)
        trigger = self.store.create("trigger-investigation", "test", "1")
        trigger = self.store.set_control_state(
            trigger,
            status="QUIESCENT_AWAITING_DATA",
            control_state={
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "ADVANCE_TO_NEXT_TARGET",
            },
        )
        self.trigger = DispatchResult(
            trigger, "NEXT_TARGET_SELECTION_REQUIRED", next_target_required=True
        )

    def planner(self, investigation, target):
        self.plans.append(target.id)
        return (
            ScientificBranch(
                id="primary",
                experiment=StageRequest(
                    id=f"001-{target.id}",
                    handler_id="test.execute",
                    parameters={"targetID": target.id},
                ),
            ),
        )

    def portfolio(self):
        return InvestigationTargetPortfolio(
            self.portfolio_path, self.store, self.dispatcher
        )

    def source(self):
        return StaticTargetSource(
            (
                InvestigationTarget(
                    "trigger", "trigger-investigation", "test", "1", priority=-1
                ),
                InvestigationTarget("target-b", "investigation-b", "test", "1"),
                InvestigationTarget("target-a", "investigation-a", "test", "1"),
                InvestigationTarget(
                    "disabled", "investigation-disabled", "test", "1", eligible=False
                ),
            )
        )

    def test_deterministically_selects_plans_and_dispatches_target(self):
        result = self.portfolio().advance(
            self.trigger,
            self.source(),
            {"test": self.planner},
            software_id="test",
            software_version="1",
        )

        self.assertEqual("target-a", result.target.id)
        self.assertEqual("EXPERIMENT_DISPATCHED", result.dispatch.disposition)
        self.assertEqual(["target-a"], self.plans)
        self.assertEqual(["investigation-a"], self.executions)
        investigation = self.store.load("investigation-a")
        self.assertEqual(
            "target-a", investigation.metadata["targetSelection"]["selectedTargetID"]
        )
        self.assertEqual(
            "RUN_EXPERIMENT",
            investigation.metadata["controlState"]["schedulerAction"],
        )

        provenance = json.loads(self.portfolio_path.read_text(encoding="utf-8"))
        selection = provenance["selections"][0]
        self.assertEqual("priority-then-target-id-v1", selection["algorithm"])
        self.assertEqual("test.targets", selection["sourceID"])
        self.assertEqual("1", selection["sourceVersion"])
        self.assertEqual(
            ["target-a", "target-b"], selection["eligibleTargetIDs"]
        )
        self.assertEqual(
            {"disabled", "trigger"},
            {item["targetID"] for item in selection["excludedTargets"]},
        )
        self.assertEqual(64, len(selection["candidateSetSha256"]))

    def test_restart_reuses_selection_and_does_not_replan_or_redispatch(self):
        first = self.portfolio().advance(
            self.trigger,
            self.source(),
            {"test": self.planner},
            software_id="test",
            software_version="1",
        )
        restarted = InvestigationTargetPortfolio(
            self.portfolio_path,
            self.store,
            InvestigationDispatcher(self.store, self.workflow),
        )

        second = restarted.advance(
            self.trigger,
            self.source(),
            {"test": self.planner},
            software_id="test",
            software_version="1",
        )

        self.assertEqual(first.selection_provenance, second.selection_provenance)
        self.assertEqual("EXPERIMENT_ALREADY_DISPATCHED", second.dispatch.disposition)
        self.assertEqual(["target-a"], self.plans)
        self.assertEqual(["investigation-a"], self.executions)
        state = json.loads(self.portfolio_path.read_text(encoding="utf-8"))
        self.assertEqual(1, len(state["selections"]))

    def test_next_advance_excludes_completed_and_quiescent_targets(self):
        first = self.portfolio().advance(
            self.trigger,
            self.source(),
            {"test": self.planner},
            software_id="test",
            software_version="1",
        )
        next_trigger = DispatchResult(
            first.dispatch.investigation,
            "NEXT_TARGET_SELECTION_REQUIRED",
            next_target_required=True,
        )

        second = self.portfolio().advance(
            next_trigger,
            self.source(),
            {"test": self.planner},
            software_id="test",
            software_version="1",
        )

        self.assertEqual("target-b", second.target.id)
        self.assertEqual(["investigation-a", "investigation-b"], self.executions)


if __name__ == "__main__":
    unittest.main()
