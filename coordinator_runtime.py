from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

from coordinator_state import CoordinatorState, first_value


class ProjectBusyError(RuntimeError):
    pass


class NoActiveProjectError(RuntimeError):
    pass


class CoordinatorRuntime:
    """
    Generic control-plane wrapper around one active CoordinatorState.

    Workers register with the runtime, not a particular project. Their
    registration is replayed into each newly activated project so Mac/iPhone
    can remain running while a workflow advances from one project to the next.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self._state: CoordinatorState | None = None
        self._node_registrations: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _node_key(node_id: Any) -> str:
        return str(node_id).strip().lower()

    @staticmethod
    def _is_terminal_status(status: dict[str, Any]) -> bool:
        total = int(status.get("projectTotalWorkUnits") or 0)
        completed = int(status.get("projectCompletedWorkUnits") or 0)
        failed = int(status.get("projectFailedWorkUnits") or 0)
        assigned = int(status.get("projectAssignedWorkUnits") or 0)

        if total <= 0:
            return True

        return completed + failed >= total and assigned == 0

    def active_state(self) -> CoordinatorState | None:
        with self.lock:
            return self._state

    def active_project_path(self) -> str | None:
        state = self.active_state()
        if state is None:
            return None
        return str(state.project_path)

    def register_node(self, payload: dict[str, Any]) -> None:
        node_id = first_value(payload, "nodeID", "nodeId", "id")
        if node_id is None:
            raise KeyError("Missing node ID.")

        normalized = dict(payload)
        normalized["nodeID"] = str(node_id)

        with self.lock:
            self._node_registrations[self._node_key(node_id)] = copy.deepcopy(
                normalized
            )
            state = self._state

        if state is not None:
            state.register_node(normalized)

    def claim_work(self, node_id: Any):
        state = self.active_state()
        if state is None:
            return None
        return state.claim_work(node_id)

    def submit_result(self, work_id: str, payload: dict[str, Any]):
        state = self.active_state()
        if state is None:
            return False, "No active project.", 409
        return state.submit_result(work_id, payload)

    def dataset(self, dataset_id: str) -> dict[str, Any] | None:
        state = self.active_state()
        if state is None:
            return None
        return state.datasets.get(dataset_id)

    def project_status(self) -> dict[str, Any]:
        state = self.active_state()
        if state is None:
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

        status = state.project_status()
        status = dict(status)
        status["status"] = (
            "COMPLETE"
            if self._is_terminal_status(status)
            else "RUNNING"
        )
        status["projectPath"] = str(state.project_path)
        return status

    def activate_project(
        self,
        project_path: str | Path,
        *,
        require_terminal: bool = True,
    ) -> dict[str, Any]:
        resolved = Path(project_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Project manifest not found: {resolved}")

        # Validate/build the new project state before replacing the active one.
        new_state = CoordinatorState(resolved)

        with self.lock:
            current = self._state
            if current is not None and require_terminal:
                current_status = current.project_status()
                if not self._is_terminal_status(current_status):
                    raise ProjectBusyError(
                        "Cannot activate a new project while the current "
                        f"project is still running: {current.project_id}"
                    )

            registrations = [
                copy.deepcopy(payload)
                for payload in self._node_registrations.values()
            ]

            # Replay workers into the new state before publishing it. A worker
            # that claims immediately after the swap can therefore be matched
            # by workload capability without needing to re-register.
            for payload in registrations:
                new_state.register_node(payload)

            self._state = new_state

        print()
        print("🧭 Project activated by control plane")
        print(f"   project: {new_state.project_id}")
        print(f"   workload: {new_state.workload_id}")
        print(f"   manifest: {resolved}")
        print(f"   retained nodes: {len(registrations)}")

        return self.project_status()
