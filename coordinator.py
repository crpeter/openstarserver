#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
import threading
import time
import uuid

HOST = "0.0.0.0"
PORT = 8080

PROJECT_ID = "openstar.bootstrap"
WORKLOAD_ID = "openstar.metal-benchmark.v1"

PROJECT_WORK_UNITS = 64

ELEMENT_COUNT = 262_144
ITERATIONS = 2_048

WORK_LEASE_SECONDS = 30

nodes = {}
work_units = {}
results = {}

lock = threading.Lock()


def canonical_uuid(value):
    return str(uuid.UUID(value)).lower()


def create_project_queue():
    with lock:
        work_units.clear()
        results.clear()

        for index in range(PROJECT_WORK_UNITS):
            work_id = str(uuid.uuid4()).lower()

            work_units[work_id] = {
                "work": {
                    "id": work_id,
                    "projectID": PROJECT_ID,
                    "workloadID": WORKLOAD_ID,
                    "elementCount": ELEMENT_COUNT,
                    "iterationsPerElement": ITERATIONS,
                    "seed": index + 1
                },
                "state": "pending",
                "nodeID": None,
                "assignedAt": None,
                "attempts": 0,
                "result": None
            }


def requeue_expired_assignments():
    now = time.time()

    for entry in work_units.values():
        if entry["state"] != "assigned":
            continue

        assigned_at = entry["assignedAt"]

        if assigned_at is None:
            continue

        if now - assigned_at >= WORK_LEASE_SECONDS:
            print()
            print("♻️ Work lease expired")
            print(f"   work: {entry['work']['id']}")
            print(f"   node: {entry['nodeID']}")

            entry["state"] = "pending"
            entry["nodeID"] = None
            entry["assignedAt"] = None


def get_project_status():
    with lock:
        requeue_expired_assignments()

        pending = 0
        assigned = 0
        completed = 0
        retries = 0

        for entry in work_units.values():
            state = entry["state"]

            if state == "pending":
                pending += 1
            elif state == "assigned":
                assigned += 1
            elif state == "completed":
                completed += 1

            retries += max(
                0,
                entry["attempts"] - 1
            )

        return {
            "projectID": PROJECT_ID,
            "totalWorkUnits": len(work_units),
            "pendingWorkUnits": pending,
            "assignedWorkUnits": assigned,
            "completedWorkUnits": completed,
            "retryCount": retries
        }


def expected_verification_value(seed, iterations):
    seed_offset = (seed % 1024) / 4096.0

    value = 0.25 + seed_offset

    for _ in range(iterations):
        value = value * 1.0000001 + 0.0000002
        value = value * 0.9999999 + 0.0000001

    return value


class OpenStarCoordinator(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/v1/projects/current/status":
            self.send_json(
                200,
                get_project_status()
            )
            return

        self.send_json(
            404,
            {
                "error": "Not found"
            }
        )

    def do_POST(self):
        if self.path == "/v1/nodes/register":
            self.register_node()
            return

        if self.path == "/v1/work/claim":
            self.claim_work()
            return

        match = re.fullmatch(
            r"/v1/work/([0-9a-fA-F-]+)/result",
            self.path
        )

        if match:
            self.submit_result(
                match.group(1)
            )
            return

        self.send_json(
            404,
            {
                "error": "Not found"
            }
        )

    def register_node(self):
        body = self.read_json()

        try:
            node_id = canonical_uuid(
                body["nodeID"]
            )
        except (ValueError, KeyError):
            self.send_json(
                400,
                {
                    "accepted": False,
                    "message": "Invalid node ID."
                }
            )
            return

        body["nodeID"] = node_id

        with lock:
            nodes[node_id] = body

        capabilities = body.get(
            "capabilities",
            {}
        )

        print()
        print("⭐ Node registered")
        print(f"   id: {node_id}")
        print(
            f"   platform: "
            f"{capabilities.get('platform')}"
        )
        print(
            f"   hardware: "
            f"{capabilities.get('hardwareIdentifier')}"
        )
        print(
            f"   gpu: "
            f"{capabilities.get('gpuName')}"
        )

        self.send_json(
            200,
            {
                "accepted": True,
                "message": "Node registered."
            }
        )

    def claim_work(self):
        body = self.read_json()

        try:
            node_id = canonical_uuid(
                body["nodeID"]
            )
        except (ValueError, KeyError):
            self.send_json(
                400,
                {
                    "error": "Invalid node ID."
                }
            )
            return

        with lock:
            if node_id not in nodes:
                self.send_json(
                    400,
                    {
                        "error":
                            "Node must register first."
                    }
                )
                return

            requeue_expired_assignments()

            assignment = None

            for entry in work_units.values():
                if entry["state"] == "pending":
                    assignment = entry
                    break

            if assignment is not None:
                assignment["state"] = "assigned"
                assignment["nodeID"] = node_id
                assignment["assignedAt"] = time.time()
                assignment["attempts"] += 1

                work = assignment["work"]
            else:
                work = None

        if work is None:
            self.send_response(204)
            self.end_headers()
            return

        print()
        print("📦 Work assigned")
        print(f"   work: {work['id']}")
        print(f"   node: {node_id}")
        print(f"   seed: {work['seed']}")

        self.send_json(
            200,
            work
        )

    def submit_result(self, work_id):
        body = self.read_json()

        try:
            work_id = canonical_uuid(work_id)
            submitted_node_id = canonical_uuid(
                body["nodeID"]
            )
        except (ValueError, KeyError):
            self.send_json(
                400,
                {
                    "accepted": False,
                    "message":
                        "Invalid work unit or node ID."
                }
            )
            return

        with lock:
            assignment = work_units.get(
                work_id
            )

            if assignment is None:
                self.send_json(
                    404,
                    {
                        "accepted": False,
                        "message": "Unknown work unit."
                    }
                )
                return

            if assignment["state"] == "completed":
                self.send_json(
                    200,
                    {
                        "accepted": True,
                        "message":
                            "Result was already accepted."
                    }
                )
                return

            expected_node = assignment["nodeID"]

            if submitted_node_id != expected_node:
                self.send_json(
                    200,
                    {
                        "accepted": False,
                        "message":
                            "Work unit belongs to another node."
                    }
                )
                return

            work = assignment["work"]

        if body.get("status") != "completed":
            print()
            print("❌ Work failed")
            print(f"   work: {work_id}")
            print(
                f"   error: "
                f"{body.get('errorMessage')}"
            )

            with lock:
                assignment["state"] = "pending"
                assignment["nodeID"] = None
                assignment["assignedAt"] = None

            self.send_json(
                200,
                {
                    "accepted": False,
                    "message":
                        "Work unit requeued."
                }
            )
            return

        verification_value = body.get(
            "verificationValue"
        )

        checksum = body.get(
            "checksum"
        )

        duration = body.get(
            "duration"
        )

        gflops = body.get(
            "estimatedGFLOPS"
        )

        valid = (
            isinstance(
                verification_value,
                (int, float)
            )
            and math.isfinite(
                verification_value
            )
            and isinstance(
                checksum,
                (int, float)
            )
            and math.isfinite(
                checksum
            )
            and isinstance(
                duration,
                (int, float)
            )
            and math.isfinite(
                duration
            )
            and duration > 0
        )

        if not valid:
            self.send_json(
                200,
                {
                    "accepted": False,
                    "message":
                        "Invalid result data."
                }
            )
            return

        expected_value = expected_verification_value(
            work["seed"],
            work["iterationsPerElement"]
        )

        if abs(
            verification_value -
            expected_value
        ) > 0.02:
            print()
            print("❌ Verification rejected")
            print(f"   work: {work_id}")
            print(
                f"   expected: "
                f"{expected_value:.6f}"
            )
            print(
                f"   received: "
                f"{verification_value:.6f}"
            )

            with lock:
                assignment["state"] = "pending"
                assignment["nodeID"] = None
                assignment["assignedAt"] = None

            self.send_json(
                200,
                {
                    "accepted": False,
                    "message":
                        "Result verification failed."
                }
            )
            return

        body["nodeID"] = submitted_node_id
        body["workUnitID"] = work_id

        with lock:
            assignment["state"] = "completed"
            assignment["assignedAt"] = None
            assignment["result"] = body

            results[work_id] = body

        status = get_project_status()

        print()
        print("✅ Result accepted")
        print(f"   work: {work_id}")
        print(f"   node: {submitted_node_id}")
        print(f"   seed: {work['seed']}")
        print(f"   duration: {duration:.4f}s")

        if gflops is not None:
            print(
                f"   throughput: "
                f"{gflops:.1f} GFLOP/s"
            )

        print(
            f"   checksum: "
            f"{checksum:.4f}"
        )

        print(
            f"   project: "
            f"{status['completedWorkUnits']}"
            f"/{status['totalWorkUnits']}"
        )

        if (
            status["completedWorkUnits"] ==
            status["totalWorkUnits"]
        ):
            print()
            print("🌟 PROJECT COMPLETE")
            print(
                f"   project: {PROJECT_ID}"
            )

        self.send_json(
            200,
            {
                "accepted": True,
                "message": "Result accepted."
            }
        )

    def read_json(self):
        length = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        data = self.rfile.read(length)

        return json.loads(
            data.decode("utf-8")
        )

    def send_json(
        self,
        status,
        value
    ):
        data = json.dumps(
            value,
            separators=(",", ":")
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(data))
        )

        self.end_headers()

        self.wfile.write(data)

    def log_message(
        self,
        format,
        *args
    ):
        return


if __name__ == "__main__":
    create_project_queue()

    print()
    print("⭐ OpenStar Coordinator")
    print(f"Listening on port {PORT}")
    print()
    print(f"Project: {PROJECT_ID}")
    print(
        f"Work units: "
        f"{PROJECT_WORK_UNITS}"
    )
    print()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        OpenStarCoordinator
    )

    server.serve_forever()