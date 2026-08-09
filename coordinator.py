#!/usr/bin/env python3

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import math
import re
import threading
import time
import uuid


HOST = "0.0.0.0"
PORT = 8080

DATASET_FILE = Path("data/tess-rr-lyr.json")

PROJECT_ID = "openstar.tess-rr-lyr"
WORKLOAD_ID = "openstar.tess-period-search.v1"

WORK_LEASE_SECONDS = 300
RETRY_DELAY_SECONDS = 2.0

nodes = {}
work_units = {}
results = {}

lock = threading.Lock()


def canonical_uuid(value):
    return str(uuid.UUID(value)).lower()


def load_dataset():
    if not DATASET_FILE.exists():
        raise RuntimeError(
            f"Dataset not found: {DATASET_FILE}"
        )

    with DATASET_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


dataset = load_dataset()


def create_project_queue():
    search = dataset["search"]
    reference_chunks = dataset["reference"]["chunks"]

    minimum_frequency = float(search["minimumFrequency"])
    frequency_step = float(search["frequencyStep"])

    with lock:
        work_units.clear()
        results.clear()

        for reference_chunk in reference_chunks:
            start_index = int(reference_chunk["startIndex"])
            frequency_count = int(reference_chunk["frequencyCount"])

            work_id = str(uuid.uuid4()).lower()

            start_frequency = (
                minimum_frequency +
                start_index * frequency_step
            )

            work_units[work_id] = {
                "work": {
                    "id": work_id,
                    "projectID": PROJECT_ID,
                    "workloadID": WORKLOAD_ID,
                    "datasetID": dataset["id"],
                    "frequencyStartIndex": start_index,
                    "startFrequency": start_frequency,
                    "frequencyStep": frequency_step,
                    "frequencyCount": frequency_count
                },
                "referenceBestFrequency": float(
                    reference_chunk["bestFrequency"]
                ),
                "referenceBestPower": float(
                    reference_chunk["bestPower"]
                ),
                "state": "pending",
                "nodeID": None,
                "assignedAt": None,
                "retryAfter": 0.0,
                "attempts": 0,
                "result": None
            }


def requeue_expired_assignments_locked():
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
            entry["retryAfter"] = 0.0


def get_project_status():
    with lock:
        requeue_expired_assignments_locked()

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
            "targetName": dataset["targetName"],
            "workloadID": WORKLOAD_ID,
            "totalWorkUnits": len(work_units),
            "pendingWorkUnits": pending,
            "assignedWorkUnits": assigned,
            "completedWorkUnits": completed,
            "retryCount": retries
        }


def public_dataset():
    return {
        "id": dataset["id"],
        "targetName": dataset["targetName"],
        "mission": dataset["mission"],
        "timeUnit": dataset["timeUnit"],
        "fluxUnit": dataset["fluxUnit"],
        "times": dataset["times"],
        "flux": dataset["flux"]
    }


def verification_tolerance(frequency_step):
    times = dataset["times"]

    baseline_days = (
        float(times[-1]) -
        float(times[0])
    )

    if baseline_days <= 0:
        return frequency_step * 4.0

    peak_width = 1.0 / baseline_days

    return max(
        frequency_step * 4.0,
        peak_width * 0.005
    )


def print_final_result():
    completed = [
        entry
        for entry in work_units.values()
        if entry["state"] == "completed"
    ]

    if len(completed) != len(work_units):
        return

    winner = max(
        completed,
        key=lambda entry:
            entry["result"]["bestPower"]
    )

    result = winner["result"]

    network_frequency = float(
        result["bestFrequency"]
    )

    network_period = (
        1.0 /
        network_frequency
    )

    reference = dataset["reference"]

    reference_frequency = float(
        reference["bestFrequency"]
    )

    reference_period = float(
        reference["bestPeriodDays"]
    )

    frequency_error = abs(
        network_frequency -
        reference_frequency
    )

    period_error = abs(
        network_period -
        reference_period
    )

    print()
    print("🌟 SCIENCE PROJECT COMPLETE")
    print(f"   target: {dataset['targetName']}")

    print()
    print("🔭 OpenStar network")
    print(
        f"   frequency: "
        f"{network_frequency:.8f} cycles/day"
    )
    print(
        f"   period: "
        f"{network_period:.8f} days"
    )
    print(
        f"   power: "
        f"{result['bestPower']:.8f}"
    )

    print()
    print("🧪 Astropy reference")
    print(
        f"   frequency: "
        f"{reference_frequency:.8f} cycles/day"
    )
    print(
        f"   period: "
        f"{reference_period:.8f} days"
    )

    print()
    print("📐 Difference")
    print(
        f"   frequency error: "
        f"{frequency_error:.8f}"
    )
    print(
        f"   period error: "
        f"{period_error:.8f} days"
    )


class OpenStarCoordinator(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/v1/projects/current/status":
            self.send_json(
                200,
                get_project_status()
            )
            return

        dataset_match = re.fullmatch(
            r"/v1/datasets/([^/]+)",
            self.path
        )

        if dataset_match:
            dataset_id = dataset_match.group(1)

            if dataset_id != dataset["id"]:
                self.send_json(
                    404,
                    {
                        "error": "Unknown dataset."
                    }
                )
                return

            self.send_json(
                200,
                public_dataset()
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
                assignment = None
                node_registered = False
            else:
                node_registered = True

                requeue_expired_assignments_locked()

                now = time.time()

                assignment = next(
                    (
                        entry
                        for entry in work_units.values()
                        if (
                            entry["state"] == "pending"
                            and entry["retryAfter"] <= now
                        )
                    ),
                    None
                )

                if assignment:
                    assignment["state"] = "assigned"
                    assignment["nodeID"] = node_id
                    assignment["assignedAt"] = time.time()
                    assignment["attempts"] += 1

                    work = assignment["work"]
                else:
                    work = None

        if not node_registered:
            self.send_json(
                400,
                {
                    "error": "Node must register first."
                }
            )
            return

        if work is None:
            self.send_response(204)
            self.end_headers()
            return

        print()
        print("📦 Science work assigned")
        print(f"   work: {work['id']}")
        print(f"   node: {node_id}")
        print(
            f"   frequency start: "
            f"{work['startFrequency']:.6f}"
        )
        print(
            f"   frequencies: "
            f"{work['frequencyCount']}"
        )

        self.send_json(
            200,
            work
        )

    def submit_result(self, work_id):
        body = self.read_json()

        try:
            work_id = canonical_uuid(
                work_id
            )

            submitted_node_id = canonical_uuid(
                body["nodeID"]
            )
        except (ValueError, KeyError):
            self.send_json(
                400,
                {
                    "accepted": False,
                    "message": "Invalid identifiers."
                }
            )
            return

        with lock:
            assignment = work_units.get(
                work_id
            )

            if assignment is None:
                response = (
                    404,
                    {
                        "accepted": False,
                        "message": "Unknown work unit."
                    }
                )
            elif assignment["state"] == "completed":
                response = (
                    200,
                    {
                        "accepted": True,
                        "message": "Result was already accepted."
                    }
                )
            elif assignment["nodeID"] != submitted_node_id:
                response = (
                    200,
                    {
                        "accepted": False,
                        "message": "Work belongs to another node."
                    }
                )
            else:
                response = None

        if response:
            self.send_json(
                response[0],
                response[1]
            )
            return

        if body.get("status") != "completed":
            with lock:
                assignment["state"] = "pending"
                assignment["nodeID"] = None
                assignment["assignedAt"] = None
                assignment["retryAfter"] = (
                    time.time() +
                    RETRY_DELAY_SECONDS
                )

            self.send_json(
                200,
                {
                    "accepted": False,
                    "message": "Work unit requeued."
                }
            )
            return

        best_frequency = body.get(
            "bestFrequency"
        )

        best_period = body.get(
            "bestPeriodDays"
        )

        best_power = body.get(
            "bestPower"
        )

        duration = body.get(
            "duration"
        )

        valid = all(
            isinstance(
                value,
                (int, float)
            )
            and math.isfinite(
                value
            )
            for value in (
                best_frequency,
                best_period,
                best_power,
                duration
            )
        )

        if not valid:
            with lock:
                assignment["state"] = "pending"
                assignment["nodeID"] = None
                assignment["assignedAt"] = None
                assignment["retryAfter"] = (
                    time.time() +
                    RETRY_DELAY_SECONDS
                )

            self.send_json(
                200,
                {
                    "accepted": False,
                    "message": "Invalid result values."
                }
            )
            return

        reference_frequency = assignment[
            "referenceBestFrequency"
        ]

        frequency_step = assignment[
            "work"
        ]["frequencyStep"]

        frequency_error = abs(
            best_frequency -
            reference_frequency
        )

        allowed_error = verification_tolerance(
            frequency_step
        )

        if frequency_error > allowed_error:
            print()
            print("❌ Science result rejected")
            print(f"   work: {work_id}")
            print(
                f"   node: "
                f"{submitted_node_id}"
            )
            print(
                f"   OpenStar: "
                f"{best_frequency:.8f}"
            )
            print(
                f"   reference: "
                f"{reference_frequency:.8f}"
            )
            print(
                f"   difference: "
                f"{frequency_error:.8f}"
            )
            print(
                f"   allowed: "
                f"{allowed_error:.8f}"
            )

            with lock:
                assignment["state"] = "pending"
                assignment["nodeID"] = None
                assignment["assignedAt"] = None
                assignment["retryAfter"] = (
                    time.time() +
                    RETRY_DELAY_SECONDS
                )

            self.send_json(
                200,
                {
                    "accepted": False,
                    "message": "Frequency verification failed."
                }
            )
            return

        body["nodeID"] = submitted_node_id
        body["workUnitID"] = work_id

        with lock:
            assignment["state"] = "completed"
            assignment["assignedAt"] = None
            assignment["retryAfter"] = 0.0
            assignment["result"] = body

            results[work_id] = body

        status = get_project_status()

        print()
        print("✅ Science result accepted")
        print(f"   work: {work_id}")
        print(
            f"   node: "
            f"{submitted_node_id}"
        )
        print(
            f"   frequency: "
            f"{best_frequency:.8f}"
        )
        print(
            f"   period: "
            f"{best_period:.8f} days"
        )
        print(
            f"   power: "
            f"{best_power:.8f}"
        )
        print(
            f"   duration: "
            f"{duration:.4f}s"
        )
        print(
            f"   project: "
            f"{status['completedWorkUnits']}"
            f"/"
            f"{status['totalWorkUnits']}"
        )

        if (
            status["completedWorkUnits"]
            ==
            status["totalWorkUnits"]
        ):
            print_final_result()

        self.send_json(
            200,
            {
                "accepted": True,
                "message": "Science result accepted."
            }
        )

    def read_json(self):
        length = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        data = self.rfile.read(
            length
        )

        return json.loads(
            data.decode("utf-8")
        )

    def send_json(self, status, value):
        data = json.dumps(
            value,
            separators=(",", ":")
        ).encode("utf-8")

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(data))
        )

        self.end_headers()

        self.wfile.write(
            data
        )

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
        f"Target: "
        f"{dataset['targetName']}"
    )
    print(
        f"Mission: "
        f"{dataset['mission']}"
    )
    print(
        f"Samples: "
        f"{len(dataset['times'])}"
    )
    print(
        f"Work units: "
        f"{len(work_units)}"
    )
    print()

    reference = dataset["reference"]

    print("Astropy reference:")
    print(
        f"   period: "
        f"{reference['bestPeriodDays']:.8f} days"
    )
    print(
        f"   frequency: "
        f"{reference['bestFrequency']:.8f} cycles/day"
    )
    print()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        OpenStarCoordinator
    )

    server.serve_forever()
