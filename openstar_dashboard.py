"""Standalone, optional OpenStar fleet dashboard sidecar."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from dashboard import build_snapshot, history_snapshot
from openstar_sector_sweep_status import sector_sweeps_projection
from openstar_science_runs import catalog_path, discover_science_runs

ROOT = Path(__file__).resolve().parent


class CoordinatorClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> Any:
        with urlopen(self.base_url + path, timeout=self.timeout) as response:
            return json.loads(response.read())

    def observation(self) -> dict[str, Any]:
        health = self.get("/v1/health")
        nodes = self.get("/v1/nodes")
        contributions = self.get("/v1/contributions/summary")
        entries = self.get("/v1/projects")
        # The project listing already contains the complete status snapshot.
        # Consuming it directly is both cheaper and internally consistent.
        projects = [
            copy.deepcopy(entry["status"])
            for entry in entries
            if isinstance(entry.get("status"), dict)
        ]
        return {
            "health": health,
            "nodes": nodes,
            "contributions": contributions,
            "projects": projects,
        }


class TelemetryStore:
    """Ephemeral operational telemetry; never connected to coordinator writes."""

    def __init__(self):
        self.lock = threading.RLock()
        self._heartbeats: dict[str, dict[str, Any]] = {}

    def update(self, payload: dict[str, Any], now: float | None = None) -> None:
        node_id = payload.get("nodeID") or payload.get("nodeId") or payload.get("id")
        if node_id is None or not str(node_id).strip():
            raise ValueError("Missing nodeID.")
        telemetry = payload.get("telemetry")
        if telemetry is None:
            telemetry = {
                key: value
                for key, value in payload.items()
                if key not in {"nodeID", "nodeId", "id"}
            }
        if not isinstance(telemetry, dict):
            raise ValueError("telemetry must be an object.")
        record = {
            "nodeID": str(node_id),
            "receivedAt": time.time() if now is None else now,
            "telemetry": json.loads(json.dumps(telemetry, allow_nan=False)),
        }
        with self.lock:
            self._heartbeats[str(node_id).strip().lower()] = record

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            return json.loads(json.dumps(self._heartbeats))


class DashboardApplication:
    def __init__(
        self,
        coordinator: CoordinatorClient,
        telemetry: TelemetryStore | None = None,
        observation_cache_seconds: float = 1.5,
        sector_sweep_state_dirs: Iterable[str | Path] = (),
        science_run_catalog: str | Path | None = None,
    ):
        self.coordinator = coordinator
        self.telemetry = telemetry or TelemetryStore()
        self.observation_cache_seconds = observation_cache_seconds
        self.sector_sweep_state_dirs = tuple(
            Path(path).expanduser().resolve() for path in sector_sweep_state_dirs
        )
        self.science_run_catalog = catalog_path(science_run_catalog)
        self._observation_lock = threading.Lock()
        self._cached_observation: dict[str, Any] | None = None
        self._cached_until = 0.0

    def observation(self) -> dict[str, Any]:
        """Coalesce concurrent browser reads into one dashboard observation."""
        with self._observation_lock:
            now = time.monotonic()
            if self._cached_observation is None or now >= self._cached_until:
                observation = self.coordinator.observation()
                science_runs = discover_science_runs(self.science_run_catalog)
                discovered_roots = [run["stateRoot"] for run in science_runs
                    if run["kind"] == "tess-sector-sweep" and Path(run["stateRoot"]).is_dir()]
                roots = tuple(dict.fromkeys((*map(str, self.sector_sweep_state_dirs), *discovered_roots)))
                try:
                    observation["sectorSweeps"] = sector_sweeps_projection(roots)
                except (OSError, ValueError, TypeError):
                    observation["sectorSweeps"] = []
                observation["scienceRuns"] = science_runs
                self._cached_observation = observation
                self._cached_until = time.monotonic() + self.observation_cache_seconds
            return self._cached_observation

    def snapshot(self) -> tuple[dict[str, Any], dict[str, Any]]:
        observation = self.observation()
        snapshot = build_snapshot(
            observation["nodes"],
            observation["contributions"],
            observation["projects"],
            self.telemetry.snapshot(),
        )
        return snapshot, observation


class DashboardHandler(BaseHTTPRequestHandler):
    application: DashboardApplication
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def send_json(self, status: int, payload: Any):
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_asset(self, relative: str, content_type: str):
        body = (ROOT / relative).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        assets = {
            "/": ("dashboard/index.html", "text/html; charset=utf-8"),
            "/dashboard": ("dashboard/index.html", "text/html; charset=utf-8"),
            "/dashboard/": ("dashboard/index.html", "text/html; charset=utf-8"),
            "/dashboard/app.css": ("dashboard/app.css", "text/css; charset=utf-8"),
            "/dashboard/app.js": ("dashboard/app.js", "text/javascript; charset=utf-8"),
        }
        if path in assets:
            self.send_asset(*assets[path])
            return
        try:
            snapshot, observation = self.application.snapshot()
            if path == "/api/dashboard/summary":
                self.send_json(200, snapshot)
                return
            if path == "/api/dashboard/workers":
                self.send_json(
                    200,
                    {
                        "workers": snapshot["workers"],
                        "updatedAt": snapshot["summary"]["updatedAt"],
                    },
                )
                return
            prefix = "/api/dashboard/workers/"
            if path.startswith(prefix):
                node_id = unquote(path[len(prefix) :]).strip("/").lower()
                worker = next(
                    (
                        worker
                        for worker in snapshot["workers"]
                        if worker["id"].lower() == node_id
                    ),
                    None,
                )
                (
                    self.send_json(200, worker)
                    if worker
                    else self.send_json(404, {"message": "Unknown worker."})
                )
                return
            if path == "/api/dashboard/activity":
                self.send_json(
                    200,
                    {
                        "projects": observation["projects"],
                        "sectorSweeps": observation.get("sectorSweeps", []),
                        "scienceRuns": observation.get("scienceRuns", []),
                        "health": observation["health"],
                        "updatedAt": time.time(),
                    },
                )
                return
            if path == "/api/dashboard/history":
                self.send_json(200, history_snapshot(
                    observation["contributions"], observation.get("scienceRuns", [])))
                return
            self.send_json(404, {"message": "Not found."})
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            self.send_json(503, {"message": f"Coordinator unavailable: {error}"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/telemetry/heartbeat":
            self.send_json(404, {"message": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Payload must be an object.")
            self.application.telemetry.update(payload)
            self.send_json(202, {"accepted": True})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"accepted": False, "message": str(error)})


def make_server(
    host: str, port: int, application: DashboardApplication
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredDashboardHandler", (DashboardHandler,), {"application": application}
    )
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description="OpenStar dashboard sidecar")
    parser.add_argument("--coordinator", default="http://127.0.0.1:8080")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--sector-sweep-state-dir",
        action="append",
        default=[],
        help="Durable TESS sector-sweep state root to observe read-only (repeatable).",
    )
    parser.add_argument(
        "--science-run-catalog",
        help="Optional catalog path (defaults to OPENSTAR_SCIENCE_RUN_CATALOG or data/science-runs.sqlite3).",
    )
    args = parser.parse_args()
    server = make_server(
        args.host,
        args.port,
        DashboardApplication(
            CoordinatorClient(args.coordinator),
            sector_sweep_state_dirs=args.sector_sweep_state_dir,
            science_run_catalog=args.science_run_catalog,
        ),
    )
    print(f"OpenStar dashboard: http://{args.host}:{args.port}/dashboard/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
