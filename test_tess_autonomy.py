import json
import tempfile
import unittest
from pathlib import Path

from openstar_autonomy import AutonomousInvestigationEngine
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTargetPortfolio
from openstar_workflow import StageOutcome, WorkflowEngine
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import (
    TessInvestigationTargetSource,
    plan_tess_branches,
)
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION


class TessAutonomyIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = InvestigationStore(self.root / "investigations")
        self.project = self.root / "tess-project.json"
        self.project.write_text(
            json.dumps(
                {
                    "id": "real-tess-project",
                    "workloadID": "openstar.tess-period-search.v1",
                    "datasets": [
                        {
                            "id": "blind-c",
                            "targetName": "Blind C",
                            "path": "c.json",
                            "ticID": 1,
                        },
                        {
                            "id": "next-real-target",
                            "targetName": "Next",
                            "path": "next.json",
                            "ticID": 2,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.source = TessInvestigationTargetSource((self.project,))

    def _complete(self, investigation, stage_id, handler_id, result, next_stage=None):
        running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
        investigation = self.store.append_running_stage(investigation, running)
        terminal = self.store.build_terminal_stage(
            stage_id=stage_id,
            handler_id=handler_id,
            status="COMPLETE",
            triggered_by_stage_id=None,
            parameters={},
            result=result,
            error=None,
            software_id="replay",
            software_version="20.28",
            started_at=running.started_at,
            next_stage=(
                {
                    "id": next_stage.id,
                    "handler_id": next_stage.handler_id,
                    "parameters": next_stage.parameters,
                    "triggered_by_stage_id": next_stage.triggered_by_stage_id,
                }
                if next_stage is not None
                else None
            ),
        )
        return self.store.complete_current_stage(investigation, terminal)

    def test_real_project_targets_have_stable_ids_priority_and_primary_handler(self):
        first = self.source.enumerate_targets()
        second = self.source.enumerate_targets()
        self.assertEqual(first, second)
        self.assertEqual([0, 1], [target.priority for target in first])
        investigation = self.store.create(
            first[0].investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=first[0].metadata,
        )
        branches = plan_tess_branches(investigation, first[0])
        self.assertEqual(
            "openstar.tess.prepare-target", branches[0].experiment.handler_id
        )
        self.assertEqual("blind-c", branches[0].experiment.parameters["datasetID"])

    def test_blind_c_v20_28_replay_is_quiescent_and_next_target_is_eligible(self):
        blind_c, next_target = self.source.enumerate_targets()
        investigation = self.store.create(
            blind_c.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=blind_c.metadata,
        )
        # Replay the persisted scientific milestones required by the real
        # v20.28 handler rather than manufacturing only its final stage.
        replay = (
            (
                "001-prepare-target",
                "openstar.tess.prepare-target",
                {"datasetID": "blind-c"},
            ),
            (
                "087-external-high-resolution",
                "openstar.tess.external-high-resolution-variability-validation.interpret",
                {
                    "recommendedNextTest": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
                },
            ),
            (
                "097-atlas-fixed-window",
                "openstar.tess.atlas-fixed-window.interpret",
                {
                    "recommendedNextTest": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
                },
            ),
            (
                "099-targeted-observation-planning",
                "openstar.tess.targeted-observation-planning.generate",
                {
                    "status": "PREREGISTERED",
                    "recommendedNextTest": "INGEST_TARGETED_OBSERVATIONS",
                },
            ),
        )
        for stage_id, handler_id, result in replay:
            investigation = self._complete(investigation, stage_id, handler_id, result)

        branches = plan_tess_branches(investigation, blind_c)
        autonomy = AutonomousInvestigationEngine(self.store)
        decision = autonomy.inspect(investigation, branches)
        self.assertEqual("BLOCKED_EXTERNAL_DATA", decision.branch_assessments[0].state)
        self.assertEqual("QUIESCENT_AWAITING_DATA", decision.investigation_status)
        self.assertEqual("ADVANCE_TO_NEXT_TARGET", decision.scheduler_action)
        self.assertNotEqual(blind_c.investigation_id, next_target.investigation_id)
        self.assertTrue(next_target.eligible)

        investigation, _ = autonomy.decide(investigation, branches)
        workflow = WorkflowEngine(self.store)
        executions = []

        def existing_prepare_handler(current, request):
            executions.append(request.handler_id)
            return StageOutcome(
                result={"datasetID": request.parameters["datasetID"]}, stop=True
            )

        workflow.register_handler(
            "openstar.tess.prepare-target", existing_prepare_handler
        )
        dispatcher = InvestigationDispatcher(self.store, workflow)
        trigger = dispatcher.dispatch(
            investigation.id, software_id="replay", software_version="20.28"
        )
        advanced = InvestigationTargetPortfolio(
            self.root / "portfolio.json", self.store, dispatcher
        ).advance(
            trigger,
            self.source,
            {WORKFLOW_ID: plan_tess_branches},
            software_id="replay",
            software_version="20.28",
        )
        self.assertEqual(next_target.id, advanced.target.id)
        self.assertEqual(["openstar.tess.prepare-target"], executions)

    def test_persisted_v20_27_evidence_selects_registered_v20_28_handler(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        investigation = self._complete(
            investigation,
            "098-atlas-fixed-window",
            "openstar.tess.atlas-fixed-window.interpret",
            {"recommendedNextTest": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"},
        )
        request = plan_tess_branches(investigation, target)[0].experiment
        self.assertEqual(
            "openstar.tess.targeted-observation-planning.generate", request.handler_id
        )

    def test_fresh_target_progresses_stage_by_stage_using_handler_continuations(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        workflow = WorkflowEngine(self.store)
        workflow.chain_stages = False
        calls = []

        def prepare(current, request):
            calls.append(request.handler_id)
            return StageOutcome(
                result={"projectPath": "prepared-primary.json"},
                next_stage=StageRequest(
                    "002-primary-distributed-search",
                    "openstar.tess.primary-project.run",
                    {"projectPath": "prepared-primary.json"},
                    request.id,
                ),
            )

        def primary(current, request):
            calls.append(request.handler_id)
            return StageOutcome(
                result={"status": "COMPLETE"},
                next_stage=StageRequest(
                    "003-catalog-identity",
                    "openstar.tess.catalog-identity",
                    {},
                    request.id,
                ),
            )

        workflow.register_handler("openstar.tess.prepare-target", prepare)
        workflow.register_handler("openstar.tess.primary-project.run", primary)
        dispatcher = InvestigationDispatcher(self.store, workflow)
        autonomy = AutonomousInvestigationEngine(self.store)

        for expected in (
            "openstar.tess.prepare-target",
            "openstar.tess.primary-project.run",
        ):
            investigation = self.store.load(investigation.id)
            investigation, decision = autonomy.decide(
                investigation, plan_tess_branches(investigation, target)
            )
            self.assertEqual(expected, decision.selected_experiment.handler_id)
            dispatcher.dispatch(
                investigation.id, software_id="test", software_version="20.28"
            )

        investigation = self.store.load(investigation.id)
        continuation = plan_tess_branches(investigation, target)[0].experiment
        self.assertEqual("openstar.tess.catalog-identity", continuation.handler_id)
        self.assertEqual(
            ["openstar.tess.prepare-target", "openstar.tess.primary-project.run"],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
