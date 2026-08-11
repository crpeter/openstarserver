from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from openstar_investigation import (
    Investigation,
    InvestigationStage,
    InvestigationStore,
    utc_now_iso,
)


@dataclass(frozen=True)
class StageRequest:
    id: str
    handler_id: str
    parameters: dict[str, Any]
    triggered_by_stage_id: str | None = None


@dataclass(frozen=True)
class StageOutcome:
    result: dict[str, Any]
    next_stage: StageRequest | None = None
    stop: bool = False
    final_status: str = "COMPLETE"


StageHandler = Callable[[Investigation, StageRequest], StageOutcome]


class WorkflowEngine:
    """
    Domain-neutral deterministic workflow runner.

    A handler may perform local/controller work, create or inspect ordinary
    OpenStar projects, and deterministically choose the next stage. The engine
    itself knows nothing about astronomy or any other science domain.
    """

    def __init__(self, store: InvestigationStore):
        self.store = store
        self.handlers: dict[str, StageHandler] = {}

    def register_handler(
        self,
        handler_id: str,
        handler: StageHandler,
    ) -> None:
        if handler_id in self.handlers:
            raise ValueError(f"Handler already registered: {handler_id}")
        self.handlers[handler_id] = handler

    def run_stage(
        self,
        investigation: Investigation,
        request: StageRequest,
        *,
        software_id: str,
        software_version: str,
    ) -> tuple[Investigation, StageRequest | None]:
        handler = self.handlers.get(request.handler_id)
        if handler is None:
            raise KeyError(f"Unknown workflow handler: {request.handler_id}")

        pending = InvestigationStage(
            id=request.id,
            handler_id=request.handler_id,
            status="RUNNING",
            triggered_by_stage_id=request.triggered_by_stage_id,
            parameters=dict(request.parameters),
            started_at=utc_now_iso(),
        )
        investigation = self.store.append_stage(
            investigation,
            pending,
        )

        outcome = handler(investigation, request)

        completed = self.store.build_completed_stage(
            stage_id=request.id,
            handler_id=request.handler_id,
            triggered_by_stage_id=request.triggered_by_stage_id,
            parameters=request.parameters,
            result=outcome.result,
            software_id=software_id,
            software_version=software_version,
            started_at=pending.started_at,
        )

        investigation = self.store.replace_last_pending_stage(
            investigation,
            completed,
        )

        if outcome.stop:
            investigation = self.store.set_status(
                investigation,
                outcome.final_status,
            )
            return investigation, None

        return investigation, outcome.next_stage
