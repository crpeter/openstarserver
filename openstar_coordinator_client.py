from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openstar_workflow import RetryableExecutionError


class CoordinatorClientError(RuntimeError):
    pass


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


class OpenStarCoordinatorClient:
    """Domain-neutral controller client for the OpenStar coordinator."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/v1/health")

    def project_status(self) -> dict[str, Any]:
        return self._request_json("GET", "/v1/projects/current/status")

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

    def run_project(
        self,
        project_path: str | Path,
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> ProjectRunResult:
        status = self.activate_project(project_path)
        project_id = str(status["projectID"])
        final_status = self.wait_for_project(
            project_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        return ProjectRunResult(project_id=project_id, status=final_status)

    def wait_for_project(
        self,
        project_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()

        while True:
            status = self.project_status()
            active_id = status.get("projectID")
            if str(active_id) != str(project_id):
                raise CoordinatorClientError(
                    "Active project changed while workflow was waiting: "
                    f"expected={project_id}, active={active_id}"
                )

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
    ) -> dict[str, Any]:
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
                f"Coordinator HTTP {error.code}: {message}"
            ) from error
        except URLError as error:
            raise CoordinatorUnavailableError(
                f"Coordinator unavailable: {error.reason}"
            ) from error

        if not raw:
            return {}

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CoordinatorClientError(
                "Coordinator returned invalid JSON."
            ) from error
