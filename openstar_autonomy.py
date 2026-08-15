from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from openstar_control import update_investigation_metadata
from openstar_investigation import Investigation, InvestigationStore
from openstar_workflow import StageRequest


@dataclass(frozen=True)
class ExternalDataDependency:
    """An input produced outside OpenStar's current computation graph."""

    id: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class ScientificBranch:
    """A possible experiment and the state required to execute it."""

    id: str
    experiment: StageRequest
    required_stage_ids: tuple[str, ...] = ()
    external_data: tuple[ExternalDataDependency, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class BranchAssessment:
    branch_id: str
    state: str
    missing_stage_ids: tuple[str, ...]
    unavailable_external_data: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class InvestigationDecision:
    branch_assessments: tuple[BranchAssessment, ...]
    selected_experiment: StageRequest | None
    investigation_status: str
    scheduler_action: str

    def control_state(self) -> dict[str, Any]:
        return {
            "branchAssessments": [asdict(item) for item in self.branch_assessments],
            "selectedExperiment": (
                asdict(self.selected_experiment)
                if self.selected_experiment is not None
                else None
            ),
            "schedulerAction": self.scheduler_action,
        }


class AutonomousInvestigationEngine:
    """Select the next scientific experiment from declared branch inputs."""

    def __init__(self, store: InvestigationStore):
        self.store = store

    def inspect(
        self,
        investigation: Investigation,
        branches: tuple[ScientificBranch, ...],
    ) -> InvestigationDecision:
        if any(stage.status == "RUNNING" for stage in investigation.stages):
            raise ValueError("Cannot select a new experiment while a stage is RUNNING.")

        completed = {
            stage.id
            for stage in investigation.stages
            if stage.status == "COMPLETE"
        }
        assessments: list[BranchAssessment] = []
        executable: list[ScientificBranch] = []

        for branch in branches:
            missing = tuple(
                stage_id
                for stage_id in branch.required_stage_ids
                if stage_id not in completed
            )
            unavailable = tuple(
                {
                    "id": dependency.id,
                    "reason": dependency.reason,
                }
                for dependency in branch.external_data
                if not dependency.available
            )
            if unavailable:
                state = "BLOCKED_EXTERNAL_DATA"
            elif missing:
                state = "NOT_READY"
            else:
                state = "EXECUTABLE"
                executable.append(branch)
            assessments.append(
                BranchAssessment(branch.id, state, missing, unavailable)
            )

        selected = None
        if executable:
            selected = min(executable, key=lambda item: (item.priority, item.id))

        if selected is not None:
            return InvestigationDecision(
                tuple(assessments), selected.experiment, "RUNNING", "RUN_EXPERIMENT"
            )

        if not assessments:
            return InvestigationDecision(
                (), None, "COMPLETE", "INVESTIGATION_COMPLETE"
            )

        if all(item.state == "BLOCKED_EXTERNAL_DATA" for item in assessments):
            return InvestigationDecision(
                tuple(assessments),
                None,
                "QUIESCENT_AWAITING_DATA",
                "ADVANCE_TO_NEXT_TARGET",
            )

        return InvestigationDecision(
            tuple(assessments),
            None,
            "BLOCKED",
            "WAIT_FOR_PREREQUISITES",
        )

    def decide(
        self,
        investigation: Investigation,
        branches: tuple[ScientificBranch, ...],
    ) -> tuple[Investigation, InvestigationDecision]:
        decision = self.inspect(investigation, branches)
        investigation = update_investigation_metadata(
            self.store,
            investigation,
            {"controlState": decision.control_state()},
            status=decision.investigation_status,
        )
        return investigation, decision
