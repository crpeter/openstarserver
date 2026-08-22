import argparse
import builtins
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from coordinator_runtime import (
    CoordinatorRuntime,
    ProjectBusyError,
    ProjectConflictError,
)
from coordinator_state import first_value
from openstar_contributions import DEFAULT_CONTRIBUTION_DB
from openstar_sector_sweep_status import sector_sweeps_projection

DEFAULT_PROJECT_PATH = "data/projects/openstar.tess-validation-v1.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
COORDINATOR_BUILD = "openstar-coordinator-v20.2-workflow-control"

RUNTIME = CoordinatorRuntime()
SECTOR_SWEEP_STATE_DIRS = []
_STATUS_LOG_LOCK = threading.Lock()
_QUIET_STATUS_SIGNATURES: set[tuple[str, str]] = set()


def _install_perf_timing_filter() -> None:
    if os.getenv("OPENSTAR_PERF_TIMING") == "1":
        return

    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if args and str(args[0]).startswith("⏱️"):
            return
        original_print(*args, **kwargs)

    builtins.print = filtered_print


def _should_log_project_status(status):
    state = str(status.get("status") or "")
    project_id = str(status.get("projectID") or "")

    # Workers intentionally keep polling after a project is terminal so they
    # can immediately receive the next workflow project. Repeated COMPLETE/IDLE
    # status lines are operational noise, so log each quiet-state signature once.
    if state not in {"COMPLETE", "IDLE"}:
        return True

    signature = (project_id, state)
    with _STATUS_LOG_LOCK:
        if signature in _QUIET_STATUS_SIGNATURES:
            return False
        _QUIET_STATUS_SIGNATURES.add(signature)
    return True


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "OpenStarCoordinator/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # HTTP/1.1 clients can disappear while the persistent request loop
            # is waiting for the next request line. That ends this connection;
            # it is not a coordinator error.
            pass

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))

        if content_length <= 0:
            return {}

        body = self.rfile.read(content_length)
        if not body:
            return {}

        return json.loads(body.decode("utf-8"))

    def _send_json(self, status_code, payload):
        body = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_no_content(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_error_json(self, status_code, message):
        self._send_json(
            status_code,
            {
                "accepted": False,
                "message": message,
            },
        )

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/v1/health":
            print(f"🌐 GET {path}")
            self._send_json(
                200,
                {
                    "ok": True,
                    "build": COORDINATOR_BUILD,
                    "project": RUNTIME.project_status(),
                    "contributionLedger": RUNTIME.ledger_health(),
                },
            )
            return

        if path == "/v1/nodes":
            self._send_json(200, RUNTIME.registered_nodes())
            return

        if path == "/v1/contributions/summary":
            try:
                summary = RUNTIME.contribution_summary()
            except Exception as error:
                self._send_error_json(503, f"Contribution ledger unavailable: {error}")
                return
            self._send_json(200, summary)
            return

        if path == "/v1/science/tess-sector-sweeps":
            self._send_json(
                200, {"sweeps": sector_sweeps_projection(SECTOR_SWEEP_STATE_DIRS)}
            )
            return

        if path == "/v1/projects/current/status":
            status = RUNTIME.project_status()
            if _should_log_project_status(status):
                print(f"🌐 GET {path}")
                print(f"   status projectID: {status.get('projectID')}")
                print(f"   status targetName: {status.get('targetName')}")
                print(f"   status datasetID: {status.get('datasetID')}")
                print(f"   status: {status.get('status')}")
            self._send_json(200, status)
            return

        if path == "/v1/projects":
            self._send_json(200, RUNTIME.projects())
            return

        project_prefix = "/v1/projects/"
        if path.startswith(project_prefix):
            remainder = path[len(project_prefix) :].strip("/")
            parts = remainder.split("/")
            if len(parts) == 2 and parts[1] == "status":
                status = RUNTIME.project_status(unquote(parts[0]))
                if status is None:
                    self._send_error_json(404, "Unknown project.")
                else:
                    self._send_json(200, status)
                return
            if len(parts) == 3 and parts[1] == "datasets":
                dataset = RUNTIME.dataset(unquote(parts[2]), unquote(parts[0]))
                if dataset is None:
                    self._send_error_json(404, "Unknown project or dataset.")
                else:
                    self._send_json(200, dataset)
                return

        print(f"🌐 GET {path}")
        dataset_prefix = "/v1/datasets/"
        if path.startswith(dataset_prefix):
            dataset_id = unquote(path[len(dataset_prefix) :]).strip("/")
            dataset = RUNTIME.dataset(dataset_id)

            if dataset is None:
                self._send_error_json(404, "Unknown dataset or no active project.")
                return

            print("   dataset targetName: " f"{dataset.get('targetName', dataset_id)}")
            self._send_json(200, dataset)
            return

        self._send_error_json(404, "Not found.")

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            payload = self._read_json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error_json(400, "Invalid JSON.")
            return

        if path == "/v1/projects/activate":
            project_path = first_value(
                payload,
                "projectPath",
                "path",
            )
            if project_path is None:
                self._send_error_json(400, "Missing projectPath.")
                return

            require_terminal = bool(payload.get("requireTerminal", True))

            try:
                status = RUNTIME.activate_project(
                    str(project_path),
                    require_terminal=require_terminal,
                )
            except (ProjectBusyError, ProjectConflictError) as error:
                self._send_error_json(409, str(error))
                return
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                self._send_error_json(400, str(error))
                return

            self._send_json(
                200,
                {
                    "accepted": True,
                    "message": "Project activated.",
                    "status": status,
                },
            )
            return

        if path == "/v1/nodes/register":
            node_id = first_value(payload, "nodeID", "nodeId", "id")
            if node_id is None:
                self._send_error_json(400, "Missing node ID.")
                return

            try:
                RUNTIME.register_node(payload)
            except (KeyError, TypeError, ValueError) as error:
                self._send_error_json(400, str(error))
                return

            self._send_json(
                200,
                {
                    "accepted": True,
                    "message": "Node registered.",
                },
            )
            return

        if path == "/v1/work/claim":
            node_id = first_value(payload, "nodeID", "nodeId", "id")
            if node_id is None:
                self._send_error_json(400, "Missing node ID.")
                return

            max_work_units = payload.get("maxWorkUnits")
            try:
                if max_work_units is None:
                    work_unit = RUNTIME.claim_work(node_id)
                else:
                    work_unit = RUNTIME.claim_work_batch(node_id, max_work_units)
            except ValueError as error:
                self._send_error_json(400, str(error))
                return
            if not work_unit:
                self._send_no_content()
                return

            self._send_json(200, work_unit)
            return

        result_prefix = "/v1/work/"
        result_suffix = "/result"
        if path.startswith(result_prefix) and path.endswith(result_suffix):
            work_id = path[len(result_prefix) : -len(result_suffix)].strip("/")
            if not work_id:
                self._send_error_json(400, "Missing work unit ID.")
                return

            accepted, message, status_code = RUNTIME.submit_result(
                unquote(work_id),
                payload,
            )
            self._send_json(
                status_code,
                {
                    "accepted": accepted,
                    "message": message,
                },
            )
            return

        self._send_error_json(404, "Not found.")

    def do_DELETE(self):
        path = urlparse(self.path).path
        prefix = "/v1/projects/"
        if not path.startswith(prefix) or not path[len(prefix) :].strip("/"):
            self._send_error_json(404, "Not found.")
            return
        project_id = unquote(path[len(prefix) :].strip("/"))
        if "/" in project_id:
            self._send_error_json(404, "Not found.")
            return
        try:
            RUNTIME.remove_project(project_id)
        except KeyError:
            self._send_error_json(404, "Unknown project.")
            return
        except ProjectBusyError as error:
            self._send_error_json(409, str(error))
            return
        self._send_json(200, {"accepted": True, "message": "Project removed."})


def parse_args():
    parser = argparse.ArgumentParser(description="OpenStar coordinator")
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT_PATH,
        help="Path to initial project manifest.",
    )
    parser.add_argument(
        "--idle",
        action="store_true",
        help="Start with no active project; a workflow may activate one later.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--contribution-db",
        default=str(DEFAULT_CONTRIBUTION_DB),
        help="Path to the durable SQLite contribution ledger.",
    )
    parser.add_argument(
        "--sector-sweep-state-dir",
        action="append",
        default=[],
        help="Durable TESS sector-sweep state root to expose read-only (repeatable).",
    )
    return parser.parse_args()


def main():
    global RUNTIME, SECTOR_SWEEP_STATE_DIRS
    args = parse_args()
    SECTOR_SWEEP_STATE_DIRS = tuple(
        Path(path).expanduser().resolve() for path in args.sector_sweep_state_dir
    )
    try:
        RUNTIME = CoordinatorRuntime(args.contribution_db)
    except (OSError, RuntimeError) as error:
        print(f"❌ Contribution ledger failed to initialize: {error}")
        raise SystemExit(1)

    if args.idle:
        print()
        print("⭐ OpenStar Coordinator")
        print(f"Build: {COORDINATOR_BUILD}")
        print(f"Listening on {args.host}:{args.port}")
        print("Project control: IDLE; waiting for workflow activation")
    else:
        try:
            RUNTIME.activate_project(
                args.project,
                require_terminal=False,
            )
            state = RUNTIME.active_state()
            assert state is not None
            state.print_startup_summary(port=args.port, host=args.host)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            print()
            print("❌ OpenStar Coordinator failed to start")
            print(f"   project: {Path(args.project)}")
            print(f"   error: {error}")
            raise SystemExit(1)

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("Stopping OpenStar Coordinator.")
    finally:
        server.server_close()
        RUNTIME.close()


if __name__ == "__main__":
    _install_perf_timing_filter()
    main()
