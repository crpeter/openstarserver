#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
import threading
import uuid

HOST = "0.0.0.0"
PORT = 8080

WORKLOAD_ID = "openstar.metal-benchmark.v1"
PROJECT_ID = "openstar.bootstrap"

ELEMENT_COUNT = 262_144
ITERATIONS = 2_048

nodes = {}
work_units = {}
results = {}

lock = threading.Lock()


def canonical_uuid(value):
    return str(uuid.UUID(value)).lower()


class OpenStarCoordinator(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/v1/nodes/register":
            self.register_node()
            return

        if self.path == "/v1/work/claim":
            self.claim_work()
            return

        match = re.fullmatch(r"/v1/work/([0-9a-fA-F-]+)/result", self.path)

        if match:
            self.submit_result(match.group(1))
            return

        self.send_json(404, {"error": "Not found"})

    def register_node(self):
        body = self.read_json()
        node_id = canonical_uuid(body["nodeID"])

        body["nodeID"] = node_id

        with lock:
            nodes[node_id] = body

        capabilities = body.get("capabilities", {})

        print()
        print("⭐ Node registered")
        print(f"   id: {node_id}")
        print(f"   platform: {capabilities.get('platform')}")
        print(f"   hardware: {capabilities.get('hardwareIdentifier')}")
        print(f"   gpu: {capabilities.get('gpuName')}")

        self.send_json(
            200,
            {
                "accepted": True,
                "message": "Node registered."
            }
        )

    def claim_work(self):
        body = self.read_json()
        node_id = canonical_uuid(body["nodeID"])

        if node_id not in nodes:
            self.send_json(
                400,
                {
                    "error": "Node must register first."
                }
            )
            return

        work_id = str(uuid.uuid4()).lower()

        work_unit = {
            "id": work_id,
            "projectID": PROJECT_ID,
            "workloadID": WORKLOAD_ID,
            "elementCount": ELEMENT_COUNT,
            "iterationsPerElement": ITERATIONS
        }

        with lock:
            work_units[work_id] = {
                "work": work_unit,
                "nodeID": node_id
            }

        print()
        print("📦 Work assigned")
        print(f"   work: {work_id}")
        print(f"   node: {node_id}")

        self.send_json(200, work_unit)

    def submit_result(self, work_id):
        body = self.read_json()

        try:
            work_id = canonical_uuid(work_id)
            submitted_node_id = canonical_uuid(body["nodeID"])
        except (ValueError, KeyError):
            self.send_json(
                400,
                {
                    "accepted": False,
                    "message": "Invalid work unit or node ID."
                }
            )
            return

        with lock:
            assignment = work_units.get(work_id)

        if assignment is None:
            self.send_json(
                404,
                {
                    "accepted": False,
                    "message": "Unknown work unit."
                }
            )
            return

        expected_node = assignment["nodeID"]

        if submitted_node_id != expected_node:
            self.send_json(
                200,
                {
                    "accepted": False,
                    "message": "Work unit belongs to another node."
                }
            )
            return

        if body.get("status") != "completed":
            print()
            print("❌ Work failed")
            print(f"   work: {work_id}")
            print(f"   error: {body.get('errorMessage')}")

            with lock:
                work_units.pop(work_id, None)

            self.send_json(
                200,
                {
                    "accepted": False,
                    "message": "Worker reported failure."
                }
            )
            return

        verification_value = body.get("verificationValue")
        checksum = body.get("checksum")
        duration = body.get("duration")
        gflops = body.get("estimatedGFLOPS")

        valid = (
            isinstance(verification_value, (int, float))
            and math.isfinite(verification_value)
            and isinstance(checksum, (int, float))
            and math.isfinite(checksum)
            and isinstance(duration, (int, float))
            and math.isfinite(duration)
            and duration > 0
        )

        if not valid:
            self.send_json(
                200,
                {
                    "accepted": False,
                    "message": "Invalid result data."
                }
            )
            return

        body["nodeID"] = submitted_node_id
        body["workUnitID"] = work_id

        with lock:
            results[work_id] = body
            work_units.pop(work_id, None)

        print()
        print("✅ Result accepted")
        print(f"   work: {work_id}")
        print(f"   node: {expected_node}")
        print(f"   duration: {duration:.4f}s")

        if gflops is not None:
            print(f"   throughput: {gflops:.1f} GFLOP/s")

        print(f"   checksum: {checksum:.4f}")

        self.send_json(
            200,
            {
                "accepted": True,
                "message": "Result accepted."
            }
        )

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def send_json(self, status, value):
        data = json.dumps(value, separators=(",", ":")).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

        self.wfile.write(data)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    print()
    print("⭐ OpenStar Coordinator")
    print(f"Listening on port {PORT}")
    print()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        OpenStarCoordinator
    )

    server.serve_forever()
