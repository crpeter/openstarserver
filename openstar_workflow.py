from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from openstar_investigation import (
    ArtifactReference,
    Investigation,
    InvestigationStage,
    InvestigationStore,
    utc_now_iso,
)


class RetryableExecutionError(RuntimeError):
    """An execution dependency failed transiently; science did not fail."""


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
    input_hashes: dict[str, str] = field(default_factory=dict)
    node_contributions: dict[str, int] = field(default_factory=dict)
    project_ids: tuple[str, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()


StageHandler = Callable[[Investigation, StageRequest], StageOutcome]


class WorkflowEngine:
    """Domain-neutral deterministic workflow runner."""

    def __init__(self, store: InvestigationStore):
        self.store = store
        self.handlers: dict[str, StageHandler] = {}
        self.chain_stages = True

    def register_handler(self, handler_id: str, handler: StageHandler) -> None:
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

        running = InvestigationStage(
            id=request.id,
            handler_id=request.handler_id,
            status="RUNNING",
            triggered_by_stage_id=request.triggered_by_stage_id,
            parameters=dict(request.parameters),
            started_at=utc_now_iso(),
        )
        investigation = self.store.append_running_stage(investigation, running)

        try:
            outcome = handler(investigation, request)
            if not outcome.stop and outcome.next_stage is None:
                raise ValueError(
                    f"Stage {request.id} returned neither stop=True nor next_stage."
                )
        except Exception as error:
            failed = self.store.build_terminal_stage(
                stage_id=request.id,
                handler_id=request.handler_id,
                status="FAILED",
                triggered_by_stage_id=request.triggered_by_stage_id,
                parameters=request.parameters,
                result=None,
                error=f"{type(error).__name__}: {error}",
                failure_classification=(
                    "TRANSIENT_INFRASTRUCTURE"
                    if isinstance(error, RetryableExecutionError)
                    else "NON_RETRYABLE"
                ),
                software_id=software_id,
                software_version=software_version,
                started_at=running.started_at,
            )
            investigation = self.store.complete_current_stage(
                investigation,
                failed,
            )
            self.store.set_status(investigation, "FAILED")
            raise

        completed = self.store.build_terminal_stage(
            stage_id=request.id,
            handler_id=request.handler_id,
            status="COMPLETE",
            triggered_by_stage_id=request.triggered_by_stage_id,
            parameters=request.parameters,
            result=outcome.result,
            error=None,
            software_id=software_id,
            software_version=software_version,
            input_hashes=outcome.input_hashes,
            node_contributions=outcome.node_contributions,
            project_ids=outcome.project_ids,
            artifacts=outcome.artifacts,
            started_at=running.started_at,
            next_stage=(
                asdict(outcome.next_stage) if outcome.next_stage is not None else None
            ),
            stop=outcome.stop,
        )
        investigation = self.store.complete_current_stage(
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

    def run(
        self,
        investigation: Investigation,
        initial_stage: StageRequest,
        *,
        software_id: str,
        software_version: str,
        max_stages: int = 100,
    ) -> Investigation:
        request: StageRequest | None = initial_stage
        count = 0

        while request is not None:
            count += 1
            if count > max_stages:
                investigation = self.store.set_status(
                    investigation,
                    "FAILED",
                )
                raise RuntimeError(f"Workflow exceeded max_stages={max_stages}.")

            investigation, request = self.run_stage(
                investigation,
                request,
                software_id=software_id,
                software_version=software_version,
            )

            if not self.chain_stages:
                return investigation

        return investigation
