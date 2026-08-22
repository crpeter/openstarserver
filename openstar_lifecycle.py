from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from openstar_autonomy import AutonomousInvestigationEngine
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import Investigation, InvestigationStore
from openstar_workflow import StageRequest
from openstar_targets import (
    BranchPlanner,
    InvestigationTarget,
    InvestigationTargetPortfolio,
    InvestigationTargetSource,
    NoEligibleTargetError,
)


@dataclass(frozen=True)
class LifecycleResult:
    investigation: Investigation
    disposition: str
    transitions: int


class InvestigationSchedulingState(str, Enum):
    RUNNABLE = "RUNNABLE"
    WAITING_EXTERNAL_DATA = "WAITING_EXTERNAL_DATA"
    BLOCKED_PREREQUISITES = "BLOCKED_PREREQUISITES"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class InvestigationPreparation:
    investigation: Investigation
    state: InvestigationSchedulingState
    transitions: int = 0


def has_persisted_failed_stage_recovery(
    investigation: Investigation, failed: Any
) -> bool:
    """Recognize an explicit durable replacement for the latest failure."""
    if investigation.status != "RUNNING":
        return False
    control = investigation.metadata.get("controlState")
    if (
        not isinstance(control, dict)
        or control.get("schedulerAction") != "RUN_EXPERIMENT"
    ):
        return False
    selected = control.get("selectedExperiment")
    if not isinstance(selected, dict):
        return False
    stage_id = selected.get("id")
    handler_id = selected.get("handler_id")
    parameters = selected.get("parameters", {})
    return (
        isinstance(stage_id, str)
        and bool(stage_id)
        and isinstance(handler_id, str)
        and bool(handler_id)
        and isinstance(parameters, dict)
        and selected.get("triggered_by_stage_id") == failed.id
        and not any(stage.id == stage_id for stage in investigation.stages)
    )


def persisted_scheduling_state(
    investigation: Investigation,
) -> InvestigationSchedulingState | None:
    """Classify only scheduler decisions already present in durable state.

    Unlike ``prepare``, this helper never plans, synthesizes recovery, or writes.
    ``None`` means a scheduler round is required before a state is authoritative.
    """
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        return InvestigationSchedulingState.RECOVERY_REQUIRED
    latest_failed = (
        investigation.stages[-1]
        if investigation.stages and investigation.stages[-1].status == "FAILED"
        else None
    )
    if latest_failed is not None and not has_persisted_failed_stage_recovery(
        investigation, latest_failed
    ):
        return InvestigationSchedulingState.FAILED
    control = investigation.metadata.get("controlState")
    if not isinstance(control, dict):
        return None
    action = control.get("schedulerAction")
    if action == "RUN_EXPERIMENT":
        return InvestigationSchedulingState.RUNNABLE
    if action == "WAIT_FOR_PREREQUISITES":
        return InvestigationSchedulingState.BLOCKED_PREREQUISITES
    if action == "INVESTIGATION_COMPLETE":
        return InvestigationSchedulingState.COMPLETE
    if (
        action == "ADVANCE_TO_NEXT_TARGET"
        and investigation.status == "QUIESCENT_AWAITING_DATA"
    ):
        return InvestigationSchedulingState.WAITING_EXTERNAL_DATA
    return None


class InvestigationLifecycleDriver:
    """Prepare and execute one explicitly selected durable investigation."""

    def __init__(
        self,
        store: InvestigationStore,
        dispatcher: InvestigationDispatcher,
        planners: dict[str, BranchPlanner],
        *,
        software_id: str,
        software_version: str,
    ):
        self.store = store
        self.dispatcher = dispatcher
        self.planners = planners
        self.software_id = software_id
        self.software_version = software_version
        self.autonomy = AutonomousInvestigationEngine(store)

    def attach(self, target: InvestigationTarget) -> Investigation:
        path = self.store.path_for(target.investigation_id)
        if path.exists():
            investigation = self.store.load(target.investigation_id)
            if (
                investigation.workflow_id != target.workflow_id
                or investigation.workflow_version != target.workflow_version
            ):
                raise RuntimeError(
                    "Target does not match its existing investigation workflow."
                )
            return investigation
        return self.store.create(
            target.investigation_id,
            target.workflow_id,
            target.workflow_version,
            metadata=dict(target.metadata or {}),
        )

    def _plan(
        self, investigation: Investigation, target: InvestigationTarget
    ) -> Investigation:
        planner = self.planners.get(investigation.workflow_id)
        if planner is None:
            raise KeyError(
                f"No branch planner registered for {investigation.workflow_id}"
            )
        updated, _ = self.autonomy.decide(investigation, planner(investigation, target))
        return updated

    @staticmethod
    def _retryable_failure(investigation: Investigation):
        if not investigation.stages:
            return None
        stage = investigation.stages[-1]
        if stage.status != "FAILED":
            return None
        if stage.failure_classification == "TRANSIENT_INFRASTRUCTURE":
            return stage
        if stage.failure_classification is None and stage.error:
            exception_type, separator, _ = stage.error.partition(":")
            if separator and exception_type == "CoordinatorClientError":
                return stage
        return None

    @staticmethod
    def _retry_request(investigation: Investigation, failed) -> StageRequest:
        prefixes = []
        for stage in investigation.stages:
            prefix, separator, _ = stage.id.partition("-")
            if separator and prefix.isdigit():
                prefixes.append(int(prefix))
        failed_prefix, separator, label = failed.id.partition("-")
        if not separator or not failed_prefix.isdigit():
            raise ValueError(f"Stage id must begin with an integer prefix: {failed.id}")
        next_number = max(prefixes, default=int(failed_prefix)) + 1
        return StageRequest(
            id=f"{next_number:03d}-{label}",
            handler_id=failed.handler_id,
            parameters=dict(failed.parameters),
            triggered_by_stage_id=failed.id,
        )

    @staticmethod
    def _has_persisted_failed_stage_recovery(
        investigation: Investigation, failed
    ) -> bool:
        """Recognize a deliberate, durable replacement for the latest failure.

        This does not decide whether a failure is safe to retry.  It only honors
        a decision already persisted by a workflow-specific repair or planner,
        while requiring an unambiguous fresh stage identity.
        """
        return has_persisted_failed_stage_recovery(investigation, failed)

    def prepare(self, target: InvestigationTarget) -> InvestigationPreparation:
        investigation = self.attach(target)
        transitions = 0
        if any(stage.status == "RUNNING" for stage in investigation.stages):
            return InvestigationPreparation(
                investigation, InvestigationSchedulingState.RECOVERY_REQUIRED
            )

        latest_failed = (
            investigation.stages[-1]
            if investigation.stages and investigation.stages[-1].status == "FAILED"
            else None
        )
        persisted_recovery = (
            latest_failed is not None
            and self._has_persisted_failed_stage_recovery(
                investigation, latest_failed
            )
        )
        failed = self._retryable_failure(investigation)
        if failed is not None and not persisted_recovery:
            control = investigation.metadata.get("controlState")
            selected = (
                control.get("selectedExperiment") if isinstance(control, dict) else None
            )
            planned = (
                isinstance(selected, dict)
                and control.get("schedulerAction") == "RUN_EXPERIMENT"
                and selected.get("triggered_by_stage_id") == failed.id
                and not any(
                    stage.id == selected.get("id") for stage in investigation.stages
                )
            )
            if not planned:
                retry = self._retry_request(investigation, failed)
                investigation = self.store.set_control_state(
                    investigation,
                    status="RUNNING",
                    control_state={
                        "branchAssessments": [],
                        "selectedExperiment": asdict(retry),
                        "schedulerAction": "RUN_EXPERIMENT",
                        "recovery": "TRANSIENT_INFRASTRUCTURE_RETRY",
                    },
                )
                transitions += 1
        elif latest_failed is not None and not persisted_recovery:
            return InvestigationPreparation(
                investigation, InvestigationSchedulingState.FAILED
            )

        control = investigation.metadata.get("controlState")
        if not isinstance(control, dict):
            investigation = self._plan(investigation, target)
            transitions += 1
            control = investigation.metadata["controlState"]

        action = control.get("schedulerAction")
        if action == "RUN_EXPERIMENT":
            state = InvestigationSchedulingState.RUNNABLE
        elif action == "WAIT_FOR_PREREQUISITES":
            state = InvestigationSchedulingState.BLOCKED_PREREQUISITES
        elif action in {"INVESTIGATION_COMPLETE"}:
            state = InvestigationSchedulingState.COMPLETE
        elif (
            action == "ADVANCE_TO_NEXT_TARGET"
            and investigation.status == "QUIESCENT_AWAITING_DATA"
        ):
            state = InvestigationSchedulingState.WAITING_EXTERNAL_DATA
        else:
            raise ValueError(f"Unknown persisted scheduler action: {action}")
        return InvestigationPreparation(investigation, state, transitions)

    def classify(self, target: InvestigationTarget) -> InvestigationPreparation:
        """Return the durable scheduler-visible state for an explicit target."""
        return self.prepare(target)

    def dispatch_runnable(
        self, target: InvestigationTarget
    ) -> InvestigationPreparation:
        prepared = self.prepare(target)
        if prepared.state != InvestigationSchedulingState.RUNNABLE:
            return prepared
        dispatched = self.dispatch_prepared(prepared)
        if dispatched.disposition == "EXPERIMENT_RECOVERY_REQUIRED":
            return InvestigationPreparation(
                dispatched.investigation,
                InvestigationSchedulingState.RECOVERY_REQUIRED,
                prepared.transitions + 1,
            )
        self.replan_after_dispatch(target)
        result = self.prepare(target)
        return InvestigationPreparation(
            result.investigation, result.state, prepared.transitions + 2
        )

    def dispatch_prepared(self, prepared: InvestigationPreparation):
        """Dispatch one already-prepared runnable decision without replanning."""
        if prepared.state != InvestigationSchedulingState.RUNNABLE:
            raise ValueError("Only a RUNNABLE investigation may be dispatched.")
        return self.dispatcher.dispatch(
            prepared.investigation.id,
            software_id=self.software_id,
            software_version=self.software_version,
        )

    def replan_after_dispatch(self, target: InvestigationTarget) -> Investigation:
        return self._plan(self.store.load(target.investigation_id), target)


class InvestigationLifecycleLoop:
    """Drive the single durable planner/decision/dispatcher lifecycle.

    The investigation snapshot is the durable action queue.  This small state
    file only identifies which target the lifecycle currently owns, so a
    restarted process always resumes by reloading the investigation itself.
    """

    def __init__(
        self,
        path: str | Path,
        store: InvestigationStore,
        dispatcher: InvestigationDispatcher,
        portfolio: InvestigationTargetPortfolio,
        target_source: InvestigationTargetSource,
        planners: dict[str, BranchPlanner],
        *,
        software_id: str,
        software_version: str,
    ):
        self.path = Path(path)
        self.store = store
        self.dispatcher = dispatcher
        self.portfolio = portfolio
        self.target_source = target_source
        self.planners = planners
        self.software_id = software_id
        self.software_version = software_version
        self.autonomy = AutonomousInvestigationEngine(store)
        self.driver = InvestigationLifecycleDriver(
            store,
            dispatcher,
            planners,
            software_id=software_id,
            software_version=software_version,
        )

    def _save_target(self, target: InvestigationTarget) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {"currentTarget": asdict(target)},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = ""
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _load_target(self) -> InvestigationTarget:
        if not self.path.exists():
            raise FileNotFoundError(
                "Lifecycle has not been started; no current target is persisted."
            )
        with self.path.open("r", encoding="utf-8") as handle:
            state: Any = json.load(handle)
        raw = state.get("currentTarget") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            raise ValueError("Lifecycle state has no valid currentTarget.")
        return InvestigationTarget(**raw)

    def start(self, target: InvestigationTarget) -> Investigation:
        """Create or attach to the initial investigation and persist ownership."""
        investigation = self.driver.attach(target)
        self._save_target(target)
        return investigation

    def _plan(
        self, investigation: Investigation, target: InvestigationTarget
    ) -> Investigation:
        return self.driver._plan(investigation, target)

    @staticmethod
    def _retryable_failure(investigation: Investigation):
        """Return only the latest retryable failure, including legacy records.

        Older snapshots predate ``failure_classification``.  Their serialized
        exception *type* is the narrow compatibility contract; the error
        message itself is deliberately never inspected.
        """
        return InvestigationLifecycleDriver._retryable_failure(investigation)

    @staticmethod
    def _retry_request(investigation: Investigation, failed) -> StageRequest:
        return InvestigationLifecycleDriver._retry_request(investigation, failed)

    def run(self, *, max_transitions: int = 100) -> LifecycleResult:
        """Run until a durable wait, terminal state, recovery, or checkpoint."""
        if max_transitions < 1:
            raise ValueError("max_transitions must be positive.")

        target = self._load_target()
        transitions = 0
        while transitions < max_transitions:
            prepared = self.driver.prepare(target)
            transitions += prepared.transitions

            if transitions >= max_transitions:
                return LifecycleResult(
                    prepared.investigation, "LIFECYCLE_CHECKPOINT", transitions
                )

            if prepared.state == InvestigationSchedulingState.RECOVERY_REQUIRED:
                return LifecycleResult(
                    prepared.investigation, "EXPERIMENT_RECOVERY_REQUIRED", transitions
                )
            if prepared.state == InvestigationSchedulingState.FAILED:
                return LifecycleResult(
                    prepared.investigation,
                    "NONRETRYABLE_FAILURE_REQUIRES_ATTENTION",
                    transitions,
                )
            if prepared.state == InvestigationSchedulingState.RUNNABLE:
                dispatched = self.driver.dispatch_prepared(prepared)
                transitions += 1
                if dispatched.disposition == "EXPERIMENT_RECOVERY_REQUIRED":
                    return LifecycleResult(
                        dispatched.investigation,
                        "EXPERIMENT_RECOVERY_REQUIRED",
                        transitions,
                    )
                if transitions >= max_transitions:
                    return LifecycleResult(
                        self.store.load(prepared.investigation.id),
                        "LIFECYCLE_CHECKPOINT",
                        transitions,
                    )
                investigation = self.driver.replan_after_dispatch(target)
                transitions += 1
                if transitions >= max_transitions:
                    return LifecycleResult(
                        investigation,
                        "LIFECYCLE_CHECKPOINT",
                        transitions,
                    )
                continue
            if prepared.state == InvestigationSchedulingState.BLOCKED_PREREQUISITES:
                dispatched = self.dispatcher.dispatch(
                    prepared.investigation.id,
                    software_id=self.software_id,
                    software_version=self.software_version,
                )
                transitions += 1
                return LifecycleResult(
                    dispatched.investigation, "WAITING_FOR_PREREQUISITES", transitions
                )

            # The legacy lifecycle treats both scientific completion and an
            # external-data wait as a request to select another portfolio target.
            dispatched = self.dispatcher.dispatch(
                prepared.investigation.id,
                software_id=self.software_id,
                software_version=self.software_version,
            )
            transitions += 1
            if transitions >= max_transitions:
                return LifecycleResult(
                    dispatched.investigation,
                    "LIFECYCLE_CHECKPOINT",
                    transitions,
                )
            try:
                advanced = self.portfolio.advance(
                    dispatched,
                    self.target_source,
                    self.planners,
                    software_id=self.software_id,
                    software_version=self.software_version,
                )
            except NoEligibleTargetError:
                from openstar_external_jobs import ExternalJobStore

                external_root = self.store.root.parent / "external-jobs"
                external_jobs = (
                    ExternalJobStore(external_root) if external_root.exists() else None
                )
                if external_jobs and external_jobs.failed_dependencies():
                    disposition = "EXTERNAL_JOB_FAILURE_REQUIRES_ATTENTION"
                elif external_jobs and (
                    external_jobs.pending_dependencies() or external_jobs.pending()
                ):
                    disposition = "NO_RUNNABLE_TARGETS_WAITING_EXTERNAL_DATA"
                else:
                    disposition = "INVESTIGATION_COMPLETE_NO_NEXT_TARGET"
                return LifecycleResult(
                    dispatched.investigation, disposition, transitions
                )

            target = advanced.target
            self._save_target(target)
            transitions += 1
            if advanced.dispatch.disposition == "EXPERIMENT_RECOVERY_REQUIRED":
                return LifecycleResult(
                    advanced.dispatch.investigation,
                    advanced.dispatch.disposition,
                    transitions,
                )

        return LifecycleResult(
            self.store.load(target.investigation_id),
            "LIFECYCLE_CHECKPOINT",
            transitions,
        )
