from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
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
        path = self.store.path_for(target.investigation_id)
        if path.exists():
            investigation = self.store.load(target.investigation_id)
            if (
                investigation.workflow_id != target.workflow_id
                or investigation.workflow_version != target.workflow_version
            ):
                raise RuntimeError(
                    "Initial target does not match its existing investigation workflow."
                )
        else:
            investigation = self.store.create(
                target.investigation_id,
                target.workflow_id,
                target.workflow_version,
                metadata=dict(target.metadata or {}),
            )
        self._save_target(target)
        return investigation

    def _plan(
        self, investigation: Investigation, target: InvestigationTarget
    ) -> Investigation:
        planner = self.planners.get(investigation.workflow_id)
        if planner is None:
            raise KeyError(
                f"No branch planner registered for {investigation.workflow_id}"
            )
        updated, _ = self.autonomy.decide(
            investigation, planner(investigation, target)
        )
        return updated

    @staticmethod
    def _retryable_failure(investigation: Investigation):
        """Return only the latest retryable failure, including legacy records.

        Older snapshots predate ``failure_classification``.  Their serialized
        exception *type* is the narrow compatibility contract; the error
        message itself is deliberately never inspected.
        """
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

    def run(self, *, max_transitions: int = 100) -> LifecycleResult:
        """Run until a durable wait, terminal state, recovery, or checkpoint."""
        if max_transitions < 1:
            raise ValueError("max_transitions must be positive.")

        target = self._load_target()
        transitions = 0
        while transitions < max_transitions:
            investigation = self.store.load(target.investigation_id)

            # An interrupted handler is an explicit recovery boundary.  Never
            # ask the planner or dispatcher to execute it automatically.
            if any(stage.status == "RUNNING" for stage in investigation.stages):
                return LifecycleResult(
                    investigation, "EXPERIMENT_RECOVERY_REQUIRED", transitions
                )

            failed = self._retryable_failure(investigation)
            if failed is not None:
                control = investigation.metadata.get("controlState")
                selected = (
                    control.get("selectedExperiment")
                    if isinstance(control, dict)
                    else None
                )
                retry_already_planned = (
                    isinstance(selected, dict)
                    and control.get("schedulerAction") == "RUN_EXPERIMENT"
                    and selected.get("triggered_by_stage_id") == failed.id
                    and not any(stage.id == selected.get("id") for stage in investigation.stages)
                )
                if retry_already_planned:
                    failed = None
            if failed is not None:
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
                if transitions >= max_transitions:
                    return LifecycleResult(
                        investigation, "LIFECYCLE_CHECKPOINT", transitions
                    )
                continue

            control = investigation.metadata.get("controlState")
            if not isinstance(control, dict):
                investigation = self._plan(investigation, target)
                transitions += 1
                continue

            action = control.get("schedulerAction")
            dispatched = self.dispatcher.dispatch(
                investigation.id,
                software_id=self.software_id,
                software_version=self.software_version,
            )
            transitions += 1

            if dispatched.disposition == "EXPERIMENT_RECOVERY_REQUIRED":
                return LifecycleResult(
                    dispatched.investigation, dispatched.disposition, transitions
                )
            if action == "RUN_EXPERIMENT":
                # Dispatch is synchronous today.  Its terminal result is new
                # scientific state, so deliberately return through the planner.
                if transitions >= max_transitions:
                    return LifecycleResult(
                        self.store.load(investigation.id),
                        "LIFECYCLE_CHECKPOINT",
                        transitions,
                    )
                investigation = self.store.load(investigation.id)
                investigation = self._plan(investigation, target)
                transitions += 1
                continue
            if action == "WAIT_FOR_PREREQUISITES":
                return LifecycleResult(
                    dispatched.investigation, "WAITING_FOR_PREREQUISITES", transitions
                )
            if action in {"INVESTIGATION_COMPLETE", "ADVANCE_TO_NEXT_TARGET"}:
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
                    if external_root.exists() and ExternalJobStore(external_root).pending():
                        return LifecycleResult(
                            dispatched.investigation,
                            "NO_RUNNABLE_TARGETS_WAITING_EXTERNAL_DATA",
                            transitions,
                        )
                    return LifecycleResult(
                        dispatched.investigation,
                        "INVESTIGATION_COMPLETE_NO_NEXT_TARGET",
                        transitions,
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
                if transitions >= max_transitions:
                    return LifecycleResult(
                        self.store.load(target.investigation_id),
                        "LIFECYCLE_CHECKPOINT",
                        transitions,
                    )
                # Whether the selected action ran or was a durable wait/stop,
                # reload it on the next iteration and use the same lifecycle.
                continue

            raise ValueError(f"Unknown persisted scheduler action: {action}")

        return LifecycleResult(
            self.store.load(target.investigation_id),
            "LIFECYCLE_CHECKPOINT",
            transitions,
        )
