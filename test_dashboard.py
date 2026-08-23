import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from dashboard import build_snapshot, history_snapshot
from openstar_dashboard import (
    DashboardApplication,
    CoordinatorClient,
    TelemetryStore,
    make_server,
)


class FakeCoordinator:
    def __init__(self):
        self.calls = 0
        self.observed = {
            "health": {"ok": True},
            "nodes": [
                {
                    "nodeID": "mac-1",
                    "lastSeenAt": 100,
                    "platform": "macOS",
                    "hardwareIdentifier": "Mac14,7",
                    "capabilities": {"gpuName": "Apple M2"},
                },
                {
                    "nodeID": "phone-1",
                    "lastSeenAt": 100,
                    "platform": "iOS",
                    "hardwareIdentifier": "iPhone15,2",
                    "capabilities": {},
                },
            ],
            "contributions": {
                "currentSession": {
                    "totalWorkerComputeSeconds": 8,
                    "aggregateSampleFrequencyEvaluationsPerMetalSecond": 25,
                },
                "allTime": {
                    "totalAcceptedWorkUnits": 4,
                    "totalWorkerComputeSeconds": 12,
                    "nodes": [
                        {
                            "nodeID": "mac-1",
                            "hardwareIdentifier": "Mac14,7",
                            "acceptedWorkUnits": 4,
                            "workerComputeSeconds": 12,
                            "metalSeconds": 10,
                            "sampleFrequencyEvaluationsPerMetalSecond": 25,
                        }
                    ],
                },
            },
            "projects": [
                {
                    "projectID": "project",
                    "workloadID": "openstar.lomb-scargle.v1",
                    "projectAssignedWorkUnits": 3,
                    "projectCompletedWorkUnits": 7,
                    "projectFailedWorkUnits": 1,
                    "projectTotalWorkUnits": 11,
                    "projectProgress": 8 / 11,
                }
            ],
        }

    def observation(self):
        self.calls += 1
        return json.loads(json.dumps(self.observed))


class ProjectionTests(unittest.TestCase):
    def test_coordinator_and_telemetry_join_only_by_node_id(self):
        coordinator = FakeCoordinator()
        store = TelemetryStore()
        store.update(
            {
                "nodeID": "mac-1",
                "telemetry": {
                    "deviceName": "Build Mac",
                    "batteryLevel": 0.8,
                    "thermalState": "nominal",
                    "computeState": "computing",
                },
            },
            now=990,
        )
        store.update(
            {"nodeID": "unknown", "telemetry": {"deviceName": "Must not appear"}},
            now=999,
        )
        snapshot = build_snapshot(
            coordinator.observed["nodes"],
            coordinator.observed["contributions"],
            coordinator.observed["projects"],
            store.snapshot(),
            now=1000,
        )
        self.assertEqual(
            ["mac-1", "phone-1"], [worker["id"] for worker in snapshot["workers"]]
        )
        mac = snapshot["workers"][0]
        self.assertEqual("Build Mac", mac["name"])
        self.assertEqual("active", mac["computeState"])
        self.assertEqual("dashboard_heartbeat", mac["lastSeenSource"])
        self.assertEqual(3, snapshot["summary"]["runningWorkUnits"])
        self.assertEqual(4, snapshot["summary"]["completedWorkUnits"])

    def test_missing_optional_telemetry_is_unavailable(self):
        coordinator = FakeCoordinator()
        snapshot = build_snapshot(
            coordinator.observed["nodes"],
            coordinator.observed["contributions"],
            coordinator.observed["projects"],
            {},
            now=1001,
        )
        phone = snapshot["workers"][1]
        self.assertIsNone(phone["batteryLevel"])
        self.assertIsNone(phone["osVersion"])
        self.assertEqual("coordinator_registration", phone["lastSeenSource"])
        self.assertEqual("offline", phone["connectionState"])

    def test_heartbeat_refreshes_liveness_and_dynamic_telemetry(self):
        coordinator = FakeCoordinator()
        store = TelemetryStore()
        store.update(
            {
                "nodeID": "phone-1",
                "telemetry": {
                    "osVersion": "18.1",
                    "appVersion": "1.2",
                    "powerState": "charging",
                    "lowPowerMode": True,
                    "workUnitProgress": 0.4,
                },
            },
            now=500,
        )
        snapshot = build_snapshot(
            coordinator.observed["nodes"],
            coordinator.observed["contributions"],
            coordinator.observed["projects"],
            store.snapshot(),
            now=501,
        )
        phone = snapshot["workers"][1]
        self.assertEqual("connected", phone["connectionState"])
        self.assertEqual("charging", phone["powerState"])
        self.assertTrue(phone["lowPowerMode"])

    def test_newer_coordinator_timestamp_wins_over_older_heartbeat(self):
        coordinator = FakeCoordinator()
        coordinator.observed["nodes"][0]["lastSeenAt"] = 990
        store = TelemetryStore()
        store.update(
            {"nodeID": "mac-1", "telemetry": {"deviceName": "Telemetry still joins"}},
            now=500,
        )
        snapshot = build_snapshot(
            coordinator.observed["nodes"],
            coordinator.observed["contributions"],
            coordinator.observed["projects"],
            store.snapshot(),
            now=1000,
        )
        mac = snapshot["workers"][0]
        self.assertEqual(990, mac["lastSeenAt"])
        self.assertEqual("coordinator_registration", mac["lastSeenSource"])
        self.assertEqual("Telemetry still joins", mac["name"])

    def test_telemetry_store_is_ephemeral_copy_and_validates(self):
        store = TelemetryStore()
        payload = {"nodeID": "mac", "telemetry": {"thermalState": "serious"}}
        store.update(payload, now=10)
        payload["telemetry"]["thermalState"] = "changed"
        self.assertEqual(
            "serious", store.snapshot()["mac"]["telemetry"]["thermalState"]
        )
        with self.assertRaises(ValueError):
            store.update({"telemetry": {}})

    def test_frontend_has_no_inner_html_sink(self):
        script = Path("dashboard/app.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)
        self.assertIn("textContent", script)
        self.assertIn("/api/dashboard/workers/${encodeURIComponent(id)}", script)
        self.assertIn("renderSectors(activity.sectorSweeps || [])", script)
        self.assertIn("In flight or recovery", script)
        self.assertNotIn("Recovery required", script)
        self.assertIn("project.projectCompletedWorkUnits", script)
        self.assertIn("reconcileKeyed", script)
        self.assertIn("RECENTLY DISCONNECTED", script)
        self.assertIn("`${Math.floor(seconds)}s ago`", script)
        self.assertIn("setInterval(refreshScience, 3000)", script)
        self.assertIn("setInterval(refreshFleet, 10000)", script)
        self.assertIn('$("#scienceRuns")', script)

    def test_workers_are_sorted_by_measured_throughput(self):
        coordinator = FakeCoordinator()
        coordinator.observed["contributions"]["allTime"]["nodes"].append({
            "nodeID": "phone-1", "acceptedWorkUnits": 1,
            "sampleFrequencyEvaluationsPerMetalSecond": 100,
        })
        snapshot = build_snapshot(coordinator.observed["nodes"],
            coordinator.observed["contributions"], coordinator.observed["projects"], {}, now=100)
        self.assertEqual(["phone-1", "mac-1"], [worker["id"] for worker in snapshot["workers"]])

    def test_contributions_are_sorted_by_exact_accepted_count(self):
        history = history_snapshot({"allTime": {"nodes": [
            {"nodeID": "low", "acceptedWorkUnits": 2},
            {"nodeID": "high", "acceptedWorkUnits": 1001},
        ]}})
        self.assertEqual(["high", "low"], [item["nodeID"] for item in history["contributionByWorker"]])


class CoordinatorClientTests(unittest.TestCase):
    def test_uses_only_documented_read_only_coordinator_routes(self):
        client = CoordinatorClient("http://coordinator")
        calls = []
        replies = {
            "/v1/health": {"ok": True},
            "/v1/nodes": [],
            "/v1/contributions/summary": {"allTime": {}, "currentSession": {}},
            "/v1/projects": [
                {"projectID": "alpha beta", "status": {"projectID": "alpha beta"}}
            ],
        }
        client.get = lambda path: calls.append(path) or replies[path]
        client.observation()
        self.assertEqual(list(replies), calls)
        self.assertTrue(all(path.startswith("/v1/") for path in calls))

    def test_project_list_nested_status_is_used_without_per_project_reads(self):
        client = CoordinatorClient("http://coordinator")
        calls = []
        replies = {
            "/v1/health": {},
            "/v1/nodes": [],
            "/v1/contributions/summary": {},
            "/v1/projects": [
                {
                    "projectID": "one",
                    "status": {"projectID": "one", "status": "RUNNING"},
                },
                {
                    "projectID": "two",
                    "status": {"projectID": "two", "status": "COMPLETE"},
                },
            ],
        }
        client.get = lambda path: calls.append(path) or replies[path]
        observation = client.observation()
        self.assertEqual(
            ["one", "two"], [item["projectID"] for item in observation["projects"]]
        )
        self.assertEqual(4, len(calls))
        self.assertFalse(any(path.endswith("/status") for path in calls))

    def test_dashboard_reads_configured_sector_sweep_state_directly(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "tess-sector-9-inventory.json").write_text(
                json.dumps({"sector": 9, "entries": []}),
                encoding="utf-8",
            )
            coordinator = FakeCoordinator()
            application = DashboardApplication(
                coordinator,
                sector_sweep_state_dirs=(root,),
            )

            _, observation = application.snapshot()

            self.assertEqual(1, coordinator.calls)
            self.assertEqual(9, observation["sectorSweeps"][0]["sector"])

    def test_dashboard_without_configured_sector_state_has_no_sector_sweeps(self):
        coordinator = FakeCoordinator()
        application = DashboardApplication(coordinator)

        _, observation = application.snapshot()

        self.assertEqual([], observation["sectorSweeps"])

    def test_concurrent_snapshots_share_short_lived_observation(self):
        coordinator = FakeCoordinator()
        original = coordinator.observation

        def slow_observation():
            time.sleep(0.05)
            return original()

        coordinator.observation = slow_observation
        application = DashboardApplication(coordinator, observation_cache_seconds=1.5)
        barrier = threading.Barrier(4)
        results = []

        def read_snapshot():
            barrier.wait()
            results.append(application.snapshot()[0])

        threads = [threading.Thread(target=read_snapshot) for _ in range(3)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(3, len(results))
        self.assertEqual(1, coordinator.calls)

    def test_coordinator_modules_have_no_dashboard_dependency(self):
        for path in (
            "coordinator.py",
            "coordinator_runtime.py",
            "coordinator_state.py",
            "openstar_contributions.py",
        ):
            source = Path(path).read_text(encoding="utf-8")
            self.assertNotIn("dashboard", source.lower(), path)
        coordinator_source = Path("coordinator.py").read_text(encoding="utf-8")
        self.assertNotIn("tess-sector-sweeps", coordinator_source)
        self.assertNotIn("sector-sweep-state-dir", coordinator_source)


class SidecarHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coordinator = FakeCoordinator()
        cls.telemetry = TelemetryStore()
        cls.server = make_server(
            "127.0.0.1", 0, DashboardApplication(cls.coordinator, cls.telemetry)
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urllib.request.urlopen(request) as response:
            return response.status, (
                json.loads(response.read())
                if "json" in response.headers.get("Content-Type", "")
                else response.read()
            )

    def test_summary_workers_activity_history_and_assets(self):
        self.assertEqual(
            2, self.request("/api/dashboard/summary")[1]["summary"]["knownWorkers"]
        )
        self.assertEqual(2, len(self.request("/api/dashboard/workers")[1]["workers"]))
        self.assertEqual(
            "project",
            self.request("/api/dashboard/activity")[1]["projects"][0]["projectID"],
        )
        self.assertEqual(
            [], self.request("/api/dashboard/activity")[1]["sectorSweeps"]
        )
        self.assertFalse(self.request("/api/dashboard/history")[1]["available"])
        self.assertIn(b"OpenStar", self.request("/dashboard/")[1])

    def test_heartbeat_and_worker_detail(self):
        self.assertEqual(
            202,
            self.request(
                "/api/telemetry/heartbeat",
                {
                    "nodeID": "mac-1",
                    "telemetry": {
                        "deviceName": '<img src=x onerror="alert(1)">',
                        "recentFailures": [{"message": "hot"}],
                    },
                },
            )[0],
        )
        detail = self.request("/api/dashboard/workers/mac-1")[1]
        self.assertEqual('<img src=x onerror="alert(1)">', detail["name"])
        self.assertEqual("hot", detail["recentFailures"][0]["message"])

    def test_sidecar_failure_or_absence_cannot_write_to_coordinator(self):
        before = json.dumps(self.coordinator.observed, sort_keys=True)
        self.request(
            "/api/telemetry/heartbeat", {"nodeID": "phone-1", "batteryLevel": 0.5}
        )
        self.assertEqual(before, json.dumps(self.coordinator.observed, sort_keys=True))
        self.server.shutdown()
        self.assertEqual(before, json.dumps(self.coordinator.observed, sort_keys=True))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()


if __name__ == "__main__":
    unittest.main()
