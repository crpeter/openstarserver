from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import quote

from openstar_workflow import RetryableExecutionError


class CoordinatorClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CoordinatorUnavailableError(CoordinatorClientError, RetryableExecutionError):
    """The coordinator could not be reached and execution may be retried."""


@dataclass(frozen=True)
class ProjectRunResult:
    project_id: str
    status: dict[str, Any]

    @property
    def node_contributions(self) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in self.status.get("nodeContributions", {}).items()
        }


@dataclass(frozen=True)
class ProjectBatchRunResult:
    runs: tuple[ProjectRunResult, ...]

    @property
    def project_ids(self) -> tuple[str, ...]:
        return tuple(run.project_id for run in self.runs)

    @property
    def node_contributions(self) -> dict[str, int]:
        contributions: dict[str, int] = {}
        for run in self.runs:
            for node_id, count in run.node_contributions.items():
                contributions[node_id] = contributions.get(node_id, 0) + count
        return contributions


class OpenStarCoordinatorClient:
    """Domain-neutral controller client for the OpenStar coordinator."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/v1/health")

    def registered_nodes(self) -> list[dict[str, Any]]:
        return list(self._request_json("GET", "/v1/nodes"))

    def contribution_summary(self) -> dict[str, Any]:
        return dict(self._request_json("GET", "/v1/contributions/summary"))

    def project_status(self, project_id: str | None = None) -> dict[str, Any]:
        if project_id is None:
            path = "/v1/projects/current/status"
        else:
            path = f"/v1/projects/{quote(str(project_id), safe='')}/status"
        return self._request_json("GET", path)

    def activate_project(
        self,
        project_path: str | Path,
        *,
        require_terminal: bool = True,
    ) -> dict[str, Any]:
        response = self._request_json(
            "POST",
            "/v1/projects/activate",
            {
                "projectPath": str(Path(project_path).expanduser().resolve()),
                "requireTerminal": require_terminal,
            },
        )
        return dict(response["status"])

    def remove_project(self, project_id: str) -> None:
        path = f"/v1/projects/{quote(str(project_id), safe='')}"
        for attempt in range(3):
            try:
                self._request_json("DELETE", path)
                return
            except CoordinatorUnavailableError:
                # DELETE is ambiguous after a transport failure: the server may
                # already have removed the terminal project. Retry so a 404 can
                # confirm that cleanup completed.
                if attempt < 2:
                    continue
                logging.warning(
                    "Could not confirm cleanup of terminal project %s; "
                    "preserving its captured result.",
                    project_id,
                )
                return
            except CoordinatorClientError as error:
                if error.status_code == 404:
                    return
                raise

    def run_project(
        self,
        project_path: str | Path,
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> ProjectRunResult:
        status = self.activate_project(project_path, require_terminal=False)
        project_id = str(status["projectID"])
        final_status = self.wait_for_project(
            project_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        result = ProjectRunResult(project_id=project_id, status=final_status)
        self.remove_project(project_id)
        return result

    def run_projects(
        self,
        project_paths: Sequence[str | Path],
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> ProjectBatchRunResult:
        paths = tuple(project_paths)
        if not paths:
            raise ValueError("run_projects requires at least one project path.")

        deadline = time.monotonic() + timeout if timeout is not None else None

        def check_deadline() -> None:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for OpenStar project batch.")

        project_ids: list[str] = []
        seen: set[str] = set()
        for path in paths:
            status = self.activate_project(path, require_terminal=False)
            check_deadline()
            project_id = str(status["projectID"])
            if project_id in seen:
                raise ValueError(
                    f"Coordinator returned duplicate project ID: {project_id}"
                )
            seen.add(project_id)
            project_ids.append(project_id)

        completed: dict[str, dict[str, Any]] = {}
        while len(completed) < len(project_ids):
            check_deadline()
            for project_id in project_ids:
                if project_id in completed:
                    continue
                check_deadline()
                status = self.project_status(project_id)
                check_deadline()
                if self.is_terminal(status):
                    completed[project_id] = status
                    self.remove_project(project_id)

            if len(completed) == len(project_ids):
                check_deadline()
                break
            check_deadline()
            time.sleep(max(0.05, poll_interval))

        check_deadline()
        return ProjectBatchRunResult(
            tuple(
                ProjectRunResult(project_id, completed[project_id])
                for project_id in project_ids
            )
        )

    def wait_for_project(
        self,
        project_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()

        while True:
            status = self.project_status(project_id)

            if self.is_terminal(status):
                return status

            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(
                    f"Timed out waiting for OpenStar project {project_id}."
                )

            time.sleep(max(0.05, poll_interval))

    @staticmethod
    def is_terminal(status: dict[str, Any]) -> bool:
        if status.get("status") == "COMPLETE":
            return True

        total = int(status.get("projectTotalWorkUnits") or 0)
        completed = int(status.get("projectCompletedWorkUnits") or 0)
        failed = int(status.get("projectFailedWorkUnits") or 0)
        assigned = int(status.get("projectAssignedWorkUnits") or 0)
        return total > 0 and completed + failed >= total and assigned == 0

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
        except HTTPError as error:
            raw = error.read()
            try:
                message = json.loads(raw.decode("utf-8")).get("message")
            except Exception:
                message = raw.decode("utf-8", errors="replace")
            raise CoordinatorClientError(
                f"Coordinator HTTP {error.code}: {message}",
                status_code=error.code,
            ) from error
        except URLError as error:
            raise CoordinatorUnavailableError(
                f"Coordinator unavailable: {error.reason}"
            ) from error
        except (ConnectionError, TimeoutError) as error:
            # urlopen normally wraps transport failures in URLError, but some
            # socket and HTTP layers allow these built-in transient failures to
            # escape directly (notably ConnectionResetError while reading).
            raise CoordinatorUnavailableError(
                f"Coordinator unavailable: {error}"
            ) from error

        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CoordinatorClientError(
                "Coordinator returned invalid JSON."
            ) from error
