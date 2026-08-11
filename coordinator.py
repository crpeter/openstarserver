import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from coordinator_runtime import (
    CoordinatorRuntime,
    ProjectBusyError,
)
from coordinator_state import first_value


DEFAULT_PROJECT_PATH = "data/projects/openstar.tess-validation-v1.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
COORDINATOR_BUILD = "openstar-coordinator-v20.0-workflow-control"

RUNTIME = CoordinatorRuntime()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "OpenStarCoordinator/1.0"

    def log_message(self, format, *args):
        return

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
        print(f"🌐 GET {path}")

        if path == "/v1/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "build": COORDINATOR_BUILD,
                    "project": RUNTIME.project_status(),
                },
            )
            return

        if path == "/v1/projects/current/status":
            status = RUNTIME.project_status()
            print(f"   status projectID: {status.get('projectID')}")
            print(f"   status targetName: {status.get('targetName')}")
            print(f"   status datasetID: {status.get('datasetID')}")
            print(f"   status: {status.get('status')}")
            self._send_json(200, status)
            return

        dataset_prefix = "/v1/datasets/"
        if path.startswith(dataset_prefix):
            dataset_id = unquote(path[len(dataset_prefix):]).strip("/")
            dataset = RUNTIME.dataset(dataset_id)

            if dataset is None:
                self._send_error_json(404, "Unknown dataset or no active project.")
                return

            print(
                "   dataset targetName: "
                f"{dataset.get('targetName', dataset_id)}"
            )
            self._send_json(200, dataset)
            return

        self._send_error_json(404, "Not found.")

    def do_POST(self):
        path = urlparse(self.path).path
        print(f"🌐 POST {path}")

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
            except ProjectBusyError as error:
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

            work_unit = RUNTIME.claim_work(node_id)
            if work_unit is None:
                self._send_no_content()
                return

            self._send_json(200, work_unit)
            return

        result_prefix = "/v1/work/"
        result_suffix = "/result"
        if path.startswith(result_prefix) and path.endswith(result_suffix):
            work_id = path[len(result_prefix):-len(result_suffix)].strip("/")
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
    return parser.parse_args()


def main():
    args = parse_args()

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


if __name__ == "__main__":
    main()
