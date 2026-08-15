import tempfile
import unittest
from pathlib import Path

from openstar_autonomy import (
    AutonomousInvestigationEngine,
    ExternalDataDependency,
    ScientificBranch,
)
from openstar_investigation import InvestigationStore
from openstar_workflow import StageRequest


class AutonomousInvestigationEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.store = InvestigationStore(Path(self.temporary_directory.name))
        self.engine = AutonomousInvestigationEngine(self.store)
        self.investigation = self.store.create("blind-c-replay", "tess", "20.28")

    @staticmethod
    def branch(branch_id, *, available, priority=0):
        return ScientificBranch(
            id=branch_id,
            experiment=StageRequest(
                id=f"experiment-{branch_id}",
                handler_id="science.experiment",
                parameters={},
            ),
            external_data=(
                ExternalDataDependency(
                    id="targeted-source-resolved-time-series",
                    available=available,
                    reason=None if available else "Observations have not been collected.",
                ),
            ),
            priority=priority,
        )

    def test_unavailable_external_data_quiesces_and_advances(self):
        updated, decision = self.engine.decide(
            self.investigation,
            (self.branch("targeted-observation-analysis", available=False),),
        )

        self.assertIsNone(decision.selected_experiment)
        self.assertEqual("BLOCKED_EXTERNAL_DATA", decision.branch_assessments[0].state)
        self.assertEqual("QUIESCENT_AWAITING_DATA", updated.status)
        self.assertEqual("ADVANCE_TO_NEXT_TARGET", decision.scheduler_action)
        persisted = self.store.load(updated.id)
        self.assertEqual(
            "BLOCKED_EXTERNAL_DATA",
            persisted.metadata["controlState"]["branchAssessments"][0]["state"],
        )
        self.assertIsNone(persisted.metadata["controlState"]["selectedExperiment"])

    def test_selects_highest_priority_executable_branch(self):
        updated, decision = self.engine.decide(
            self.investigation,
            (
                self.branch("later", available=True, priority=20),
                self.branch("next", available=True, priority=10),
                self.branch("blocked", available=False, priority=0),
            ),
        )

        self.assertEqual("experiment-next", decision.selected_experiment.id)
        self.assertEqual("RUN_EXPERIMENT", decision.scheduler_action)
        self.assertEqual("RUNNING", updated.status)
        persisted = self.store.load(updated.id)
        self.assertEqual(
            "experiment-next",
            persisted.metadata["controlState"]["selectedExperiment"]["id"],
        )
        self.assertEqual(
            "RUN_EXPERIMENT",
            persisted.metadata["controlState"]["schedulerAction"],
        )

    def test_missing_completed_stage_is_not_executable(self):
        branch = ScientificBranch(
            id="needs-prior-evidence",
            experiment=StageRequest("experiment", "science.experiment", {}),
            required_stage_ids=("028-observation-plan",),
        )

        updated, decision = self.engine.decide(self.investigation, (branch,))

        self.assertEqual("NOT_READY", decision.branch_assessments[0].state)
        self.assertIsNone(decision.selected_experiment)
        self.assertEqual("BLOCKED", updated.status)
        self.assertEqual("WAIT_FOR_PREREQUISITES", decision.scheduler_action)

    def test_mixed_external_and_not_ready_branches_waits_for_prerequisites(self):
        not_ready = ScientificBranch(
            id="not-ready",
            experiment=StageRequest("later", "science.experiment", {}),
            required_stage_ids=("prior-stage",),
        )

        updated, decision = self.engine.decide(
            self.investigation,
            (self.branch("external", available=False), not_ready),
        )

        self.assertEqual("BLOCKED", updated.status)
        self.assertEqual("WAIT_FOR_PREREQUISITES", decision.scheduler_action)

    def test_empty_branch_set_completes_investigation(self):
        updated, decision = self.engine.decide(self.investigation, ())

        self.assertEqual("COMPLETE", updated.status)
        self.assertEqual("INVESTIGATION_COMPLETE", decision.scheduler_action)


if __name__ == "__main__":
    unittest.main()
