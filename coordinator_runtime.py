from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from coordinator_state import CoordinatorState, first_value, normalize_id
from openstar_contributions import (
    DEFAULT_ACCOUNTING,
    ContributionStore,
    timing_metrics,
)

MAX_WORK_UNITS_PER_CLAIM = 128
HOT_PATH_PROGRESS_INTERVAL_SECONDS = 10.0


class ProjectBusyError(RuntimeError):
    pass


class ProjectConflictError(RuntimeError):
    pass


class NoActiveProjectError(RuntimeError):
    pass


class CoordinatorRuntime:
    """Multi-project scheduler around isolated, project-local states."""

    def __init__(self, contribution_db: str | Path | None = None):
        self.lock = threading.RLock()
        self._states: dict[str, CoordinatorState] = {}
        self._project_order: list[str] = []
        self._work_project_index: dict[str, str] = {}
        self._node_registrations: dict[str, dict[str, Any]] = {}
        self._node_activity: dict[str, float] = {}
        self._next_project_index = 0
        self._legacy_current_project_id: str | None = None
        self.coordinator_session_id = str(uuid.uuid4())
        self.coordinator_session_started_at = time.time()
        self.contribution_store = (
            ContributionStore(contribution_db) if contribution_db is not None else None
        )
        self._ledger_error: str | None = None
        self._progress_last_logged_at = time.monotonic()
        self._progress_assigned = 0
        self._progress_accepted = 0
        # Terminal edges wait in insertion order for the next distinct project
        # activation.  Activated transitions are keyed by the next project so
        # concurrent investigations cannot overwrite one another before claim.
        self._transition_terminals: dict[str, float] = {}
        self._transition_activations: dict[str, tuple[str, float, float]] = {}

    @staticmethod
    def _operational_print(message: str) -> None:
        try:
            print(message)
        except Exception:
            pass

    def _project_became_terminal(self, project_id: str, terminal_at: float) -> None:
        with self.lock:
            self._transition_terminals[project_id] = terminal_at

    def _record_project_activation(self, project_id: str) -> None:
        with self.lock:
            if project_id in self._transition_activations:
                return
            previous = next(
                (
                    (previous_id, terminal_at)
                    for previous_id, terminal_at in self._transition_terminals.items()
                    if previous_id != project_id
                ),
                None,
            )
            if previous is None:
                return
            previous_id, terminal_at = previous
            del self._transition_terminals[previous_id]
            activated_at = time.monotonic()
            self._transition_activations[project_id] = (
                previous_id,
                terminal_at,
                activated_at,
            )
        self._operational_print(
            "⏱️ Project transition activation: "
            f"previous={previous_id} next={project_id} "
            f"terminal-to-activation={activated_at - terminal_at:.3f}s"
        )

    def _record_first_claim(self, project_id: str) -> None:
        with self.lock:
            transition = self._transition_activations.pop(project_id, None)
            if transition is None:
                return
            previous_id, terminal_at, activated_at = transition
            claimed_at = time.monotonic()
        self._operational_print(
            "⏱️ Project transition: "
            f"previous={previous_id} next={project_id} "
            f"terminal-to-activation={activated_at - terminal_at:.3f}s "
            f"activation-to-first-claim={claimed_at - activated_at:.3f}s "
            f"terminal-to-first-claim={claimed_at - terminal_at:.3f}s"
        )

    def _record_progress(self, *, assigned: int = 0, accepted: int = 0) -> None:
        """Periodically summarize successful hot-path operations in one write."""
        with self.lock:
            self._progress_assigned += assigned
            self._progress_accepted += accepted
            now = time.monotonic()
            if now - self._progress_last_logged_at < HOT_PATH_PROGRESS_INTERVAL_SECONDS:
                return
            message = (
                "📊 Coordinator progress: "
                f"assigned={self._progress_assigned}, "
                f"accepted={self._progress_accepted}, "
                f"liveProjects={len(self._states)}"
            )
            self._progress_assigned = 0
            self._progress_accepted = 0
            self._progress_last_logged_at = now
        print(message)

    def _ledger_failed(self, operation: str, error: Exception) -> None:
        self._ledger_error = f"{operation}: {type(error).__name__}: {error}"
        logging.exception("Contribution ledger %s failed", operation)

    def ledger_health(self) -> dict[str, Any]:
        return {
            "ok": self._ledger_error is None,
            "schema": "openstar-contributions-v1",
            "coordinatorSessionID": self.coordinator_session_id,
            "error": self._ledger_error,
        }

    def registered_nodes(self) -> list[dict[str, Any]]:
        if self.contribution_store is not None:
            try:
                nodes = copy.deepcopy(self.contribution_store.nodes())
                with self.lock:
                    registrations = copy.deepcopy(self._node_registrations)
                    activity = dict(self._node_activity)
                for node in nodes:
                    key = self._node_key(node["nodeID"])
                    live = registrations.get(key, {})
                    if isinstance(live.get("telemetry"), dict):
                        node["telemetry"] = live["telemetry"]
                    node["lastSeenAt"] = max(
                        float(node.get("lastSeenAt") or 0), activity.get(key, 0)
                    )
                return nodes
            except Exception as error:
                self._ledger_failed("node query", error)
        with self.lock:
            nodes = [
                copy.deepcopy(value) for value in self._node_registrations.values()
            ]
            for node in nodes:
                node["lastSeenAt"] = self._node_activity.get(
                    self._node_key(node["nodeID"]), node.get("lastSeenAt")
                )
            return nodes

    def contribution_summary(self) -> dict[str, Any]:
        if self.contribution_store is None:
            empty = {
                "totalAcceptedWorkUnits": 0,
                "totalWorkerComputeSeconds": 0.0,
                "totalMetalSeconds": 0.0,
                "totalSampleFrequencyEvaluations": 0,
                "aggregateSampleFrequencyEvaluationsPerMetalSecond": None,
                "nodes": [],
            }
            summary = {
                "coordinatorSessionID": self.coordinator_session_id,
                "coordinatorSessionStartedAt": self.coordinator_session_started_at,
                "currentSession": copy.deepcopy(empty),
                "allTime": empty,
            }
            return self._with_session_wall_metrics(summary)
        try:
            summary = copy.deepcopy(
                self.contribution_store.summary(self.coordinator_session_id)
            )
            summary["coordinatorSessionStartedAt"] = self.coordinator_session_started_at
            return self._with_session_wall_metrics(summary)
        except Exception as error:
            self._ledger_failed("summary query", error)
            raise

    def _with_session_wall_metrics(self, summary: dict[str, Any]) -> dict[str, Any]:
        elapsed = max(0.0, time.time() - self.coordinator_session_started_at)
        current = summary["currentSession"]
        current["wallElapsedSeconds"] = elapsed
        current["sampleFrequencyEvaluationsPerWallSecond"] = (
            current["totalSampleFrequencyEvaluations"] / elapsed
            if elapsed > 0
            else None
        )
        summary["allTime"]["wallElapsedSeconds"] = None
        summary["allTime"]["sampleFrequencyEvaluationsPerWallSecond"] = None
        return summary

    def close(self) -> None:
        if self.contribution_store is not None:
            self.contribution_store.close()

    @staticmethod
    def _node_key(node_id: Any) -> str:
        return str(node_id).strip().lower()

    @staticmethod
    def _is_terminal_status(status: dict[str, Any]) -> bool:
        total = int(status.get("projectTotalWorkUnits") or 0)
        completed = int(status.get("projectCompletedWorkUnits") or 0)
        failed = int(status.get("projectFailedWorkUnits") or 0)
        assigned = int(status.get("projectAssignedWorkUnits") or 0)
        return total <= 0 or completed + failed >= total and assigned == 0

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {
            "status": "IDLE",
            "projectID": "",
            "workloadID": "",
            "datasetID": "",
            "targetName": "",
            "totalWorkUnits": 0,
            "pendingWorkUnits": 0,
            "assignedWorkUnits": 0,
            "completedWorkUnits": 0,
            "retryCount": 0,
            "failedWorkUnits": 0,
            "projectPendingWorkUnits": 0,
            "projectAssignedWorkUnits": 0,
            "projectCompletedWorkUnits": 0,
            "projectFailedWorkUnits": 0,
            "projectTotalWorkUnits": 0,
            "projectProgress": 1.0,
            "nodeContributions": {},
            "datasets": [],
        }

    def active_state(self) -> CoordinatorState | None:
        with self.lock:
            project_id = self._legacy_current_project_id
            return self._states.get(project_id) if project_id is not None else None

    def active_project_path(self) -> str | None:
        state = self.active_state()
        return None if state is None else str(state.project_path)

    def register_node(self, payload: dict[str, Any]) -> None:
        node_id = first_value(payload, "nodeID", "nodeId", "id")
        if node_id is None:
            raise KeyError("Missing node ID.")
        normalized = dict(payload)
        normalized["nodeID"] = str(node_id)
        now = time.time()
        if self.contribution_store is not None:
            try:
                self.contribution_store.upsert_node(normalized, now)
            except Exception as error:
                self._ledger_failed("node registration", error)
        with self.lock:
            self._node_activity[self._node_key(node_id)] = now
            self._node_registrations[self._node_key(node_id)] = copy.deepcopy(
                normalized
            )
            states = list(self._states.values())
        for state in states:
            state.register_node(normalized)

    def record_node_activity(
        self, node_id: Any, telemetry: dict[str, Any] | None = None
    ) -> None:
        """Track protocol activity; optional telemetry is generic and additive."""
        now = time.time()
        key = self._node_key(node_id)
        with self.lock:
            self._node_activity[key] = now
            registration = self._node_registrations.get(key)
            if registration is not None and isinstance(telemetry, dict):
                registration["telemetry"] = copy.deepcopy(telemetry)
        if self.contribution_store is not None:
            try:
                self.contribution_store.touch_node(
                    str(node_id),
                    now,
                    telemetry if isinstance(telemetry, dict) else None,
                )
            except Exception as error:
                self._ledger_failed("node activity write", error)

    def claim_work(self, node_id: Any, telemetry: dict[str, Any] | None = None):
        self.record_node_activity(node_id, telemetry)
        # Keep the legacy call path intact as well as its single-object result.
        with self.lock:
            if not self._project_order:
                return None
            start = self._next_project_index % len(self._project_order)

            for offset in range(len(self._project_order)):
                position = (start + offset) % len(self._project_order)
                project_id = self._project_order[position]
                state = self._states[project_id]
                work = state.claim_work(node_id)
                if work is None:
                    continue
                self._next_project_index = (position + 1) % len(self._project_order)
                self._record_first_claim(project_id)
                self._record_progress(assigned=1)
                return work

            return None

    def claim_work_batch(
        self,
        node_id: Any,
        max_work_units: int,
        telemetry: dict[str, Any] | None = None,
    ):
        self.record_node_activity(node_id, telemetry)
        if isinstance(max_work_units, bool) or not isinstance(max_work_units, int):
            raise ValueError("maxWorkUnits must be a positive integer.")
        if not 1 <= max_work_units <= MAX_WORK_UNITS_PER_CLAIM:
            raise ValueError(
                f"maxWorkUnits must be between 1 and {MAX_WORK_UNITS_PER_CLAIM}."
            )

        with self.lock:
            if not self._project_order:
                return None
            start = self._next_project_index % len(self._project_order)

            for offset in range(len(self._project_order)):
                position = (start + offset) % len(self._project_order)
                project_id = self._project_order[position]
                state = self._states[project_id]
                work = state.claim_work_batch(node_id, max_work_units)
                if not work:
                    continue
                self._next_project_index = (position + 1) % len(self._project_order)
                self._record_first_claim(project_id)
                self._record_progress(assigned=len(work))
                return work

            return None

    def submit_result(self, work_id: str, payload: dict[str, Any]):
        normalized = normalize_id(work_id)
        with self.lock:
            project_id = self._work_project_index.get(normalized)
            state = self._states.get(project_id) if project_id is not None else None
        if state is None:
            return False, "Unknown work unit.", 404
        with state.lock:
            assignment = state.assigned.get(normalized)
            activity_node_id = assignment.get("nodeID") if assignment else None
        if activity_node_id is not None:
            telemetry = payload.get("telemetry")
            self.record_node_activity(
                activity_node_id, telemetry if isinstance(telemetry, dict) else None
            )
        result_payload = dict(payload)
        # Operational telemetry is deliberately excluded from scientific result
        # storage and retry identity; it belongs to the node activity channel.
        result_payload.pop("telemetry", None)
        response = state.submit_result(work_id, result_payload)
        if response[0] and self.contribution_store is not None:
            normalized_work_id = normalize_id(work_id)
            with state.lock:
                work_unit = state.work_units[normalized_work_id]
                dataset_id = str(work_unit["datasetID"])
                metrics = DEFAULT_ACCOUNTING.metrics(
                    work_unit, state.datasets[dataset_id]
                )
                accepted_result = state.completed[normalized_work_id]
                record = {
                    "project_id": str(work_unit.get("projectID") or ""),
                    "workload_id": str(work_unit.get("workloadID") or ""),
                    "dataset_id": dataset_id,
                    "work_unit_id": str(work_unit["id"]),
                    "node_id": str(accepted_result["nodeID"]),
                    "work_metrics": dict(metrics),
                    "timing_metrics": timing_metrics(accepted_result),
                }
            try:
                self.contribution_store.record(
                    session_id=self.coordinator_session_id,
                    accepted_at=time.time(),
                    **record,
                )
            except Exception as error:
                # Scientific acceptance is never rewritten by telemetry failure.
                self._ledger_failed("accepted contribution write", error)
        if response[0] and response[1] != "Identical result already accepted.":
            self._record_progress(accepted=1)
        return response

    def dataset(
        self, dataset_id: str, project_id: str | None = None
    ) -> dict[str, Any] | None:
        with self.lock:
            selected_id = (
                project_id
                if project_id is not None
                else self._legacy_current_project_id
            )
            state = self._states.get(selected_id) if selected_id is not None else None
        return None if state is None else state.datasets.get(dataset_id)

    def project_status(self, project_id: str | None = None) -> dict[str, Any] | None:
        with self.lock:
            selected_id = (
                project_id
                if project_id is not None
                else self._legacy_current_project_id
            )
            state = self._states.get(selected_id) if selected_id is not None else None
        if state is None:
            return self._idle_status() if project_id is None else None
        status = dict(state.project_status())
        status["status"] = "COMPLETE" if self._is_terminal_status(status) else "RUNNING"
        status["projectPath"] = str(state.project_path)
        return status

    def projects(self) -> list[dict[str, Any]]:
        with self.lock:
            entries = [
                (project_id, self._states[project_id])
                for project_id in self._project_order
            ]
        result = []
        for project_id, state in entries:
            status = self.project_status(project_id)
            if status is not None:
                result.append(
                    {
                        "projectID": project_id,
                        "projectPath": str(state.project_path),
                        "status": status,
                    }
                )
        return result

    def activate_project(
        self, project_path: str | Path, *, require_terminal: bool = True
    ) -> dict[str, Any]:
        resolved = Path(project_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Project manifest not found: {resolved}")
        new_state = CoordinatorState(resolved)
        new_state.terminal_observer = self._project_became_terminal
        project_id = str(new_state.project_id)
        new_work_ids = list(new_state.work_units)

        with self.lock:
            existing = self._states.get(project_id)
            if existing is not None:
                if existing.project_path != resolved:
                    raise ProjectConflictError(
                        f"Project ID {project_id!r} is already loaded from {existing.project_path}."
                    )
                self._legacy_current_project_id = project_id
                status = self.project_status(project_id)
                assert status is not None
                return status

            if require_terminal:
                for state in self._states.values():
                    if not self._is_terminal_status(state.project_status()):
                        raise ProjectBusyError(
                            "Cannot activate a new project while an existing project is still running: "
                            + str(state.project_id)
                        )
            collisions = [
                work_id
                for work_id in new_work_ids
                if normalize_id(work_id) in self._work_project_index
            ]
            if collisions:
                raise ProjectConflictError(
                    f"Work unit ID is already owned by another project: {collisions[0]}"
                )
            registrations = [
                copy.deepcopy(value) for value in self._node_registrations.values()
            ]
            for payload in registrations:
                new_state.register_node(payload)
            self._states[project_id] = new_state
            self._project_order.append(project_id)
            for work_id in new_work_ids:
                self._work_project_index[normalize_id(work_id)] = project_id
            self._legacy_current_project_id = project_id
            self._record_project_activation(project_id)

        return self.project_status(project_id)  # type: ignore[return-value]

    def remove_project(self, project_id: str) -> None:
        with self.lock:
            state = self._states.get(project_id)
            if state is None:
                raise KeyError(project_id)
            if not self._is_terminal_status(state.project_status()):
                raise ProjectBusyError(
                    f"Cannot remove nonterminal project: {project_id}"
                )
            removed_index = self._project_order.index(project_id)
            del self._states[project_id]
            self._project_order.pop(removed_index)
            for work_id, owner in list(self._work_project_index.items()):
                if owner == project_id:
                    del self._work_project_index[work_id]
            if self._project_order:
                if removed_index < self._next_project_index:
                    self._next_project_index -= 1
                self._next_project_index %= len(self._project_order)
            else:
                self._next_project_index = 0
            if self._legacy_current_project_id == project_id:
                self._legacy_current_project_id = (
                    self._project_order[-1] if self._project_order else None
                )
