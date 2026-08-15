from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openstar_investigation import Investigation, InvestigationStore
from openstar_workflow import StageRequest, WorkflowEngine


@dataclass(frozen=True)
class DispatchResult:
    investigation: Investigation
    disposition: str
    next_target_required: bool = False


class InvestigationDispatcher:
    """Consume one durable investigation decision through the workflow engine."""

    def __init__(self, store: InvestigationStore, workflow_engine: WorkflowEngine):
        self.store = store
        self.workflow_engine = workflow_engine

    @staticmethod
    def _stage_request(raw: Any) -> StageRequest:
        if not isinstance(raw, dict):
            raise ValueError("RUN_EXPERIMENT requires a persisted selectedExperiment.")
        return StageRequest(
            id=str(raw["id"]),
            handler_id=str(raw["handler_id"]),
            parameters=dict(raw.get("parameters") or {}),
            triggered_by_stage_id=raw.get("triggered_by_stage_id"),
        )

    @staticmethod
    def _existing_stage(
        investigation: Investigation,
        request: StageRequest,
    ):
        for stage in investigation.stages:
            if stage.id != request.id:
                continue
            if (
                stage.handler_id != request.handler_id
                or stage.parameters != request.parameters
                or stage.triggered_by_stage_id != request.triggered_by_stage_id
            ):
                raise RuntimeError(
                    f"Persisted experiment id collides with a different stage: {request.id}"
                )
            return stage
        return None

    def dispatch(
        self,
        investigation_id: str,
        *,
        software_id: str,
        software_version: str,
        max_stages: int = 100,
    ) -> DispatchResult:
        # Always reload here: the control decision is the durable handoff
        # between the investigation planner and this process.
        investigation = self.store.load(investigation_id)
        control_state = investigation.metadata.get("controlState")
        if not isinstance(control_state, dict):
            raise ValueError("Investigation has no persisted controlState.")

        action = control_state.get("schedulerAction")
        if action == "RUN_EXPERIMENT":
            request = self._stage_request(control_state.get("selectedExperiment"))
            existing = self._existing_stage(investigation, request)
            if existing is not None:
                disposition = (
                    "EXPERIMENT_RECOVERY_REQUIRED"
                    if existing.status == "RUNNING"
                    else "EXPERIMENT_ALREADY_DISPATCHED"
                )
                return DispatchResult(investigation, disposition)

            investigation = self.workflow_engine.run(
                investigation,
                request,
                software_id=software_id,
                software_version=software_version,
                max_stages=max_stages,
            )
            return DispatchResult(investigation, "EXPERIMENT_DISPATCHED")

        if action == "WAIT_FOR_PREREQUISITES":
            return DispatchResult(investigation, "WAITING_FOR_PREREQUISITES")

        if action == "INVESTIGATION_COMPLETE":
            return DispatchResult(
                investigation,
                "INVESTIGATION_TERMINAL",
                next_target_required=True,
            )

        if action == "ADVANCE_TO_NEXT_TARGET":
            return DispatchResult(
                investigation,
                "NEXT_TARGET_SELECTION_REQUIRED",
                next_target_required=True,
            )

        raise ValueError(f"Unknown persisted scheduler action: {action}")
