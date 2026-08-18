from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

from openstar_autonomy import AutonomousInvestigationEngine, ScientificBranch
from openstar_control import update_investigation_metadata
from openstar_dispatch import DispatchResult, InvestigationDispatcher
from openstar_investigation import Investigation, InvestigationStore, sha256_json


@dataclass(frozen=True)
class InvestigationTarget:
    id: str
    investigation_id: str
    workflow_id: str
    workflow_version: str
    priority: int = 0
    eligible: bool = True
    metadata: dict[str, object] | None = None


class InvestigationTargetSource(Protocol):
    id: str
    version: str

    def enumerate_targets(self) -> Sequence[InvestigationTarget]: ...


BranchPlanner = Callable[
    [Investigation, InvestigationTarget],
    tuple[ScientificBranch, ...],
]


class NoEligibleTargetError(RuntimeError):
    """Raised when a portfolio has exhausted its eligible target set."""


@dataclass(frozen=True)
class TargetAdvanceResult:
    target: InvestigationTarget
    selection_provenance: dict[str, object]
    dispatch: DispatchResult


class InvestigationTargetPortfolio:
    """Select, plan, and dispatch targets without interpreting their science."""

    def __init__(
        self,
        path: str | Path,
        investigation_store: InvestigationStore,
        dispatcher: InvestigationDispatcher,
    ):
        self.path = Path(path)
        self.investigation_store = investigation_store
        self.dispatcher = dispatcher

    def _load_state(self) -> dict[str, object]:
        if not self.path.exists():
            return {"selections": [], "currentSelection": None}
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_state(self, state: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = ""
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _eligibility(
        self,
        targets: Sequence[InvestigationTarget],
        trigger_investigation_id: str,
    ) -> tuple[list[InvestigationTarget], list[dict[str, str]]]:
        eligible: list[InvestigationTarget] = []
        excluded: list[dict[str, str]] = []
        for target in targets:
            reason = None
            if not target.eligible:
                reason = "source-ineligible"
            elif target.investigation_id == trigger_investigation_id:
                reason = "trigger-investigation"
            elif self.investigation_store.path_for(target.investigation_id).exists():
                investigation = self.investigation_store.load(target.investigation_id)
                awakened = bool(investigation.metadata.get("externalJobWakeDependencies"))
                if investigation.status == "COMPLETE" or (
                    investigation.status == "QUIESCENT_AWAITING_DATA" and not awakened
                ):
                    reason = f"investigation-{investigation.status.lower()}"
            if reason is None:
                eligible.append(target)
            else:
                excluded.append({"targetID": target.id, "reason": reason})
        return eligible, excluded

    def _select(
        self,
        source: InvestigationTargetSource,
        trigger_investigation_id: str,
    ) -> tuple[InvestigationTarget, dict[str, object]]:
        state = self._load_state()
        current = state.get("currentSelection")
        targets = tuple(source.enumerate_targets())
        by_id = {target.id: target for target in targets}
        if len(by_id) != len(targets):
            raise ValueError("Target source returned duplicate target IDs.")

        if (
            isinstance(current, dict)
            and current.get("triggerInvestigationID") == trigger_investigation_id
        ):
            selected = by_id.get(str(current.get("selectedTargetID")))
            if selected is None:
                raise RuntimeError("Persisted selected target is absent from its source.")
            return selected, current

        eligible, excluded = self._eligibility(targets, trigger_investigation_id)
        if not eligible:
            raise NoEligibleTargetError(
                "No eligible investigation target is available."
            )
        selected = min(eligible, key=lambda target: (target.priority, target.id))
        selections = list(state.get("selections") or [])
        provenance: dict[str, object] = {
            "selectionID": f"selection-{len(selections) + 1:06d}",
            "selectedAt": datetime.now(timezone.utc).isoformat(),
            "selectedTargetID": selected.id,
            "selectedInvestigationID": selected.investigation_id,
            "triggerInvestigationID": trigger_investigation_id,
            "sourceID": source.id,
            "sourceVersion": source.version,
            "algorithm": "priority-then-target-id-v1",
            "eligibleTargetIDs": [item.id for item in sorted(eligible, key=lambda item: (item.priority, item.id))],
            "excludedTargets": excluded,
            "candidateSetSha256": sha256_json([asdict(item) for item in targets]),
        }
        selections.append(provenance)
        self._save_state({"selections": selections, "currentSelection": provenance})
        return selected, provenance

    def select_initial(
        self, source: InvestigationTargetSource
    ) -> tuple[InvestigationTarget, dict[str, object]]:
        """Select the first lifecycle target through the durable portfolio."""
        return self._select(source, "__lifecycle_start__")

    def advance(
        self,
        trigger: DispatchResult,
        source: InvestigationTargetSource,
        planners: dict[str, BranchPlanner],
        *,
        software_id: str,
        software_version: str,
    ) -> TargetAdvanceResult:
        if not trigger.next_target_required:
            raise ValueError("Dispatch result does not require another target.")
        target, provenance = self._select(source, trigger.investigation.id)
        path = self.investigation_store.path_for(target.investigation_id)
        if path.exists():
            investigation = self.investigation_store.load(target.investigation_id)
            if (
                investigation.workflow_id != target.workflow_id
                or investigation.workflow_version != target.workflow_version
            ):
                raise RuntimeError(
                    "Target candidate does not match its existing investigation workflow."
                )
        else:
            investigation = self.investigation_store.create(
                target.investigation_id,
                target.workflow_id,
                target.workflow_version,
                metadata=dict(target.metadata or {}),
            )

        if investigation.metadata.get("targetSelection") != provenance:
            investigation = update_investigation_metadata(
                self.investigation_store,
                investigation,
                {"targetSelection": provenance},
            )

        if not isinstance(investigation.metadata.get("controlState"), dict):
            planner = planners.get(target.workflow_id)
            if planner is None:
                raise KeyError(f"No branch planner registered for {target.workflow_id}")
            autonomy = AutonomousInvestigationEngine(self.investigation_store)
            investigation, _ = autonomy.decide(
                investigation, planner(investigation, target)
            )

        if investigation.metadata.get("externalJobWakeDependencies"):
            investigation = update_investigation_metadata(
                self.investigation_store, investigation,
                {"externalJobWakeDependencies": []}
            )

        dispatched = self.dispatcher.dispatch(
            investigation.id,
            software_id=software_id,
            software_version=software_version,
        )
        return TargetAdvanceResult(target, provenance, dispatched)
