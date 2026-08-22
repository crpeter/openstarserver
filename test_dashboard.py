import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import coordinator
from coordinator_runtime import CoordinatorRuntime
from dashboard import activity_snapshot, dashboard_snapshot


class FakeRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self._states = {}
        self.nodes = [
            {
                "nodeID": "mac-1",
                "lastSeenAt": 990,
                "platform": "macOS",
                "hardwareIdentifier": "Mac14,7",
                "capabilities": {
                    "gpuName": "Apple M2",
                    "workloads": ["openstar.lomb-scargle.v1"],
                },
            },
            {
                "nodeID": "phone-1",
                "lastSeenAt": 800,
                "platform": "iOS",
                "hardwareIdentifier": "iPhone15,2",
                "capabilities": {"batteryLevel": 0.72},
            },
            {"nodeID": "old", "lastSeenAt": 1, "capabilities": {}},
        ]

    def registered_nodes(self):
        return json.loads(json.dumps(self.nodes))

    def contribution_summary(self):
        devices = [
            {
                "nodeID": "mac-1",
                "acceptedWorkUnits": 4,
                "workerComputeSeconds": 12.0,
                "metalSeconds": 10.0,
                "sampleFrequencyEvaluationsPerMetalSecond": 25.0,
            }
        ]
        scope = {
            "totalAcceptedWorkUnits": 4,
            "totalWorkerComputeSeconds": 12.0,
            "aggregateSampleFrequencyEvaluationsPerMetalSecond": 25.0,
            "nodes": devices,
        }
        return {"currentSession": dict(scope), "allTime": dict(scope)}

    def projects(self):
        return []


class DashboardProjectionTests(unittest.TestCase):
    def test_classification_aggregation_and_optional_telemetry(self):
        runtime = FakeRuntime()
        before = json.dumps(runtime.nodes, sort_keys=True)
        result = dashboard_snapshot(runtime, now=1000)
        self.assertEqual(3, result["summary"]["knownWorkers"])
        self.assertEqual(1, result["summary"]["connectedWorkers"])
        self.assertEqual(1, result["summary"]["idleWorkers"])
        self.assertEqual(2, result["summary"]["offlineWorkers"])
        self.assertEqual(1, result["summary"]["recentlyDisconnectedWorkers"])
        self.assertEqual(4, result["summary"]["completedWorkUnits"])
        self.assertEqual(
            {"macOS", "iOS", None}, {w["platform"] for w in result["workers"]}
        )
        self.assertIsNone(result["workers"][1]["osVersion"])
        self.assertEqual(before, json.dumps(runtime.nodes, sort_keys=True))

    def test_active_assignment_wins_over_idle(self):
        runtime = FakeRuntime()
        state = SimpleNamespace(
            lock=threading.RLock(),
            project_id="project",
            workload_id="generic",
            assigned={
                "unit": {"nodeID": "mac-1", "assignedAt": 995, "leaseExpiresAt": 1115}
            },
            work_units={
                "unit": {
                    "projectID": "project",
                    "workloadID": "generic",
                    "datasetID": "data",
                }
            },
            completed={},
            execution_failure_history={},
        )
        runtime._states["project"] = state
        result = dashboard_snapshot(runtime, now=1000)
        self.assertEqual("active", result["workers"][0]["computeState"])
        self.assertEqual(1, result["summary"]["runningWorkUnits"])
        self.assertIsNone(result["workers"][0]["currentAssignment"]["progress"])

    def test_batch_assignments_are_not_overwritten(self):
        runtime = FakeRuntime()
        leases = {
            name: {"nodeID": "mac-1", "assignedAt": 995, "leaseExpiresAt": 1115}
            for name in ("unit-1", "unit-2", "unit-3")
        }
        units = {
            name: {"projectID": "project", "workloadID": "generic", "datasetID": "data"}
            for name in leases
        }
        runtime._states["project"] = SimpleNamespace(
            lock=threading.RLock(),
            project_id="project",
            workload_id="generic",
            assigned=leases,
            work_units=units,
            completed={},
            execution_failure_history={},
        )
        result = dashboard_snapshot(runtime, now=1000)
        worker = result["workers"][0]
        self.assertEqual(3, result["summary"]["runningWorkUnits"])
        self.assertEqual(3, worker["runningWorkUnits"])
        self.assertEqual(
            {"unit-1", "unit-2", "unit-3"},
            {item["workUnitID"] for item in worker["currentAssignments"]},
        )

    def test_claim_activity_after_registration_keeps_worker_connected(self):
        runtime = CoordinatorRuntime()
        with patch("coordinator_runtime.time.time", return_value=10):
            runtime.register_node(
                {"nodeID": "phone", "capabilities": {"platform": "iOS"}}
            )
        with patch("coordinator_runtime.time.time", return_value=500):
            self.assertIsNone(runtime.claim_work("phone"))
        worker = dashboard_snapshot(runtime, now=501)["workers"][0]
        self.assertEqual("connected", worker["connectionState"])
        self.assertEqual(500, worker["lastSeenAt"])

    def test_dynamic_generic_telemetry_refreshes_on_activity(self):
        runtime = CoordinatorRuntime()
        runtime.register_node({"nodeID": "phone", "capabilities": {}})
        runtime.record_node_activity(
            "phone",
            {
                "batteryLevel": 0.4,
                "thermalState": "serious",
                "deviceName": "Chris's iPhone",
                "osVersion": "18.1",
                "appVersion": "1.2",
                "lowPowerMode": True,
            },
        )
        worker = dashboard_snapshot(runtime)["workers"][0]
        self.assertEqual(0.4, worker["batteryLevel"])
        self.assertEqual("serious", worker["thermalState"])
        self.assertEqual("Chris's iPhone", worker["name"])
        self.assertTrue(worker["lowPowerMode"])

    def test_result_submission_refreshes_activity(self):
        runtime = CoordinatorRuntime()
        runtime.register_node({"nodeID": "mac", "capabilities": {}})
        submitted = []

        def reject_result(_work, payload):
            submitted.append(payload)
            return False, "invalid", 400

        state = SimpleNamespace(
            lock=threading.RLock(),
            assigned={"unit": {"nodeID": "mac"}},
            submit_result=reject_result,
        )
        runtime._states["project"] = state
        runtime._work_project_index["unit"] = "project"
        with patch("coordinator_runtime.time.time", return_value=900):
            runtime.submit_result("unit", {"telemetry": {"powerState": "charging"}})
        self.assertEqual(900, runtime.registered_nodes()[0]["lastSeenAt"])
        self.assertEqual(
            "charging", runtime.registered_nodes()[0]["telemetry"]["powerState"]
        )
        self.assertNotIn("telemetry", submitted[0])

    def test_frontend_uses_safe_dom_text_and_fetches_detail(self):
        script = Path("dashboard/app.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)
        self.assertIn("textContent", script)
        self.assertIn("/api/dashboard/workers/${encodeURIComponent(id)}", script)
        runtime = FakeRuntime()
        runtime.nodes[0]["capabilities"][
            "deviceName"
        ] = '<img src=x onerror="alert(1)">'
        self.assertEqual(
            '<img src=x onerror="alert(1)">',
            dashboard_snapshot(runtime, now=1000)["workers"][0]["name"],
        )

    def test_activity_is_read_only(self):
        runtime = FakeRuntime()
        state = SimpleNamespace(
            lock=threading.RLock(),
            project_id="p",
            project_name="Project",
            workload_id="generic",
            work_units={"u": {}},
            pending=["u"],
            assigned={},
            completed={},
            failed={},
        )
        runtime._states["p"] = state
        before = list(state.pending)
        self.assertEqual(
            1, activity_snapshot(runtime)["projects"][0]["projectPendingWorkUnits"]
        )
        self.assertEqual(before, state.pending)


class DashboardAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FakeRuntime()
        cls.runtime.nodes[0]["lastSeenAt"] = 9999999999
        cls.patcher = patch.object(coordinator, "RUNTIME", cls.runtime)
        cls.patcher.start()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), coordinator.RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.patcher.stop()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as response:
            return response.status, response.headers, response.read()

    def test_read_only_endpoints_and_assets(self):
        before = json.dumps(self.runtime.nodes, sort_keys=True)
        status, headers, body = self.get("/api/dashboard/summary")
        self.assertEqual(200, status)
        self.assertEqual(3, json.loads(body)["summary"]["knownWorkers"])
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(before, json.dumps(self.runtime.nodes, sort_keys=True))
        self.assertIn(b"OpenStar", self.get("/dashboard/")[2])

    def test_worker_detail_and_missing_worker(self):
        self.assertEqual(
            "mac-1", json.loads(self.get("/api/dashboard/workers/mac-1")[2])["id"]
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get("/api/dashboard/workers/missing")
        self.assertEqual(404, error.exception.code)


if __name__ == "__main__":
    unittest.main()
