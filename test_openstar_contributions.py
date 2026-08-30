import json
import sqlite3
from contextlib import closing
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import coordinator
from coordinator_runtime import CoordinatorRuntime
from openstar_contributions import ContributionStore, timing_metrics
from openstar_coordinator_client import OpenStarCoordinatorClient


class ContributionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.store = ContributionStore(self.path)
        self.addCleanup(self.store.close)
        self.node = {
            "nodeID": "device-a",
            "capabilities": {
                "platform": "macOS",
                "hardwareIdentifier": "MacBookPro18,2",
                "gpuName": "M1 Max",
                "processorCount": 10,
                "memoryGB": 32,
                "workloads": ["openstar.lomb-scargle.v1"],
            },
        }
        self.store.upsert_node(self.node, 10.0)

    @staticmethod
    def work(identifier="work-1", workload="openstar.lomb-scargle.v1"):
        return {
            "id": identifier,
            "projectID": "project",
            "workloadID": workload,
            "datasetID": "dataset",
            "payload": {"frequencyCount": 7},
        }

    def record(self, **changes):
        work_unit = changes.pop("work_unit", self.work())
        changes.pop("dataset", None)
        work_metrics = {"workloadID": work_unit["workloadID"]}
        if work_unit["workloadID"] == "openstar.lomb-scargle.v1":
            work_metrics.update({
                "sampleCount": 4,
                "frequencyCount": 7,
                "sampleFrequencyEvaluations": 28,
            })
        result = changes.pop(
            "result",
            {
                "duration": 5.0,
                "payload": {
                    "metalDurationSeconds": 2.0,
                    "validation": {"durationSeconds": 0.25},
                },
            },
        )
        arguments = {
            "session_id": "session-1",
            "accepted_at": 20.0,
            "project_id": work_unit["projectID"],
            "workload_id": work_unit["workloadID"],
            "dataset_id": work_unit["datasetID"],
            "work_unit_id": work_unit["id"],
            "node_id": "device-a",
            "work_metrics": work_metrics,
            "timing_metrics": timing_metrics(result),
        }
        arguments.update(changes)
        return self.store.record(**arguments)

    def test_node_upsert_preserves_first_seen_and_updates_capabilities(self):
        changed = {
            "nodeID": "device-a",
            "ownerUserID": "untrusted",
            "capabilities": {
                "platform": "iOS",
                "hardwareIdentifier": "iPhone17,1",
                "gpuName": "Apple GPU",
                "processorCount": 6,
                "memoryGB": 8,
            },
        }
        self.store.upsert_node(changed, 30.0)
        node = self.store.nodes()[0]
        self.assertEqual(10.0, node["firstSeenAt"])
        self.assertEqual(30.0, node["lastSeenAt"])
        self.assertEqual("iOS", node["platform"])
        self.assertEqual(changed["capabilities"], node["capabilities"])
        self.assertIsNone(node["ownerUserID"])

    def test_record_is_idempotent_and_derives_work_not_worker_claims(self):
        result = {
            "duration": 5,
            "sampleCount": 999999,
            "payload": {"frequencyCount": 999999, "metalDurationSeconds": 2},
        }
        self.assertTrue(self.record(result=result))
        self.assertFalse(self.record(result=result, accepted_at=99))
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT work_metrics_json,timing_metrics_json FROM contributions"
            ).fetchone()
        metrics, timing = map(json.loads, row)
        self.assertEqual(
            {
                "workloadID": "openstar.lomb-scargle.v1",
                "sampleCount": 4,
                "frequencyCount": 7,
                "sampleFrequencyEvaluations": 28,
            },
            metrics,
        )
        self.assertEqual({"workerTotalSeconds": 5.0, "metalSeconds": 2.0}, timing)

    def test_missing_timing_and_unknown_workload_are_supported(self):
        self.assertTrue(
            self.record(work_unit=self.work(workload="future.work.v9"), result={})
        )
        with closing(sqlite3.connect(self.path)) as connection:
            metrics, timing = connection.execute(
                "SELECT work_metrics_json,timing_metrics_json FROM contributions"
            ).fetchone()
        self.assertEqual({"workloadID": "future.work.v9"}, json.loads(metrics))
        self.assertEqual({}, json.loads(timing))

    def test_sessions_devices_and_restart_aggregate_independently(self):
        self.record()
        other = dict(self.node)
        other["nodeID"] = "device-b"
        self.store.upsert_node(other, 11)
        self.record(
            session_id="old-session", work_unit=self.work("work-2"), node_id="device-b"
        )
        restarted = ContributionStore(self.path)
        self.addCleanup(restarted.close)
        summary = restarted.summary("session-1")
        self.assertEqual(1, summary["currentSession"]["totalAcceptedWorkUnits"])
        self.assertEqual(2, summary["allTime"]["totalAcceptedWorkUnits"])
        self.assertEqual(
            {"device-a", "device-b"},
            {node["nodeID"] for node in summary["allTime"]["nodes"]},
        )
        self.assertEqual(
            14, summary["allTime"]["aggregateSampleFrequencyEvaluationsPerMetalSecond"]
        )

    def test_worker_and_metal_throughput_are_accounted_independently(self):
        cpu = {
            "nodeID": "linux-cpu",
            "capabilities": {
                "platform": "linux",
                "hardwareIdentifier": "x86_64 Linux CPU",
                "gpuName": "none",
                "processorCount": 8,
            },
        }
        zero = {"nodeID": "zero-duration", "capabilities": {}}
        self.store.upsert_node(cpu, 11)
        self.store.upsert_node(zero, 12)
        self.record(
            work_unit=self.work("cpu-work"), node_id="linux-cpu",
            timing_metrics={"workerTotalSeconds": 7.0, "metalSeconds": 0.0},
        )
        self.record(
            work_unit=self.work("zero-work"), node_id="zero-duration",
            timing_metrics={"workerTotalSeconds": 0.0, "metalSeconds": 0.0},
        )
        self.record(work_unit=self.work("metal-work"))

        summary = self.store.summary("session-1")["currentSession"]
        nodes = {node["nodeID"]: node for node in summary["nodes"]}
        self.assertEqual(28, nodes["linux-cpu"]["sampleFrequencyEvaluations"])
        self.assertEqual(7, nodes["linux-cpu"]["workerComputeSeconds"])
        self.assertEqual(0, nodes["linux-cpu"]["metalSeconds"])
        self.assertEqual(4, nodes["linux-cpu"]["sampleFrequencyEvaluationsPerWorkerComputeSecond"])
        self.assertIsNone(nodes["linux-cpu"]["sampleFrequencyEvaluationsPerMetalSecond"])
        self.assertEqual(5.6, nodes["device-a"]["sampleFrequencyEvaluationsPerWorkerComputeSecond"])
        self.assertEqual(14, nodes["device-a"]["sampleFrequencyEvaluationsPerMetalSecond"])
        self.assertIsNone(nodes["zero-duration"]["sampleFrequencyEvaluationsPerWorkerComputeSecond"])
        self.assertEqual(7, summary["aggregateSampleFrequencyEvaluationsPerWorkerComputeSecond"])

    def test_concurrent_writes_are_safe(self):
        errors = []

        def write(worker):
            for index in range(50):
                try:
                    self.record(work_unit=self.work(f"work-{worker}-{index}"))
                    # Every repeated event must remain idempotent under load.
                    self.record(work_unit=self.work(f"work-{worker}-{index}"))
                except Exception as error:
                    errors.append(error)

        threads = [
            threading.Thread(target=write, args=(worker,)) for worker in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([], errors)
        summary = self.store.summary("session-1")["currentSession"]
        self.assertEqual(400, summary["totalAcceptedWorkUnits"])
        self.assertEqual(11_200, summary["totalSampleFrequencyEvaluations"])
        self.assertNotIn("database is locked", " ".join(map(str, errors)).lower())

    def test_contribution_and_aggregate_update_are_atomic(self):
        with self.store._lock, self.store._connection as connection:
            connection.execute("""CREATE TEMP TRIGGER reject_aggregate
                BEFORE INSERT ON contribution_aggregates
                BEGIN SELECT RAISE(ABORT, 'aggregate rejected'); END""")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "aggregate rejected"):
            self.record()
        with closing(sqlite3.connect(self.path)) as connection:
            count = connection.execute("SELECT COUNT(*) FROM contributions").fetchone()[
                0
            ]
        self.assertEqual(0, count)


class ContributionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        dataset = root / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "id": "dataset",
                    "times": [0, 1, 2],
                    "values": [1, 0, 1],
                    "frequencySearch": {
                        "minimumFrequency": 1,
                        "frequencyStep": 0.1,
                        "totalFrequencies": 1,
                        "frequenciesPerWorkUnit": 1,
                    },
                }
            )
        )
        self.manifest = root / "project.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "id": "project",
                    "workloadID": "openstar.lomb-scargle.v1",
                    "datasets": [{"id": "dataset", "path": str(dataset)}],
                }
            )
        )
        self.runtime = CoordinatorRuntime(root / "ledger.sqlite3")
        self.addCleanup(self.runtime.close)
        self.runtime.activate_project(self.manifest, require_terminal=False)
        self.runtime.register_node(
            {"nodeID": "node", "capabilities": {"platform": "macOS"}}
        )

    def test_only_accepted_result_is_recorded_and_duplicate_is_idempotent(self):
        work = self.runtime.claim_work("node")
        failed = {
            "status": "failed",
            "duration": 999,
            "workUnitID": work["id"],
            "nodeID": "node",
        }
        self.assertFalse(self.runtime.submit_result(work["id"], failed)[0])
        self.assertEqual(
            0, self.runtime.contribution_summary()["allTime"]["totalAcceptedWorkUnits"]
        )
        self.runtime.register_node({"nodeID": "node-2", "capabilities": {}})
        self.runtime.active_state().retry_after.clear()
        work = self.runtime.claim_work("node-2")
        result = {
            "status": "completed",
            "workUnitID": work["id"],
            "nodeID": "node-2",
            "bestFrequency": 1.0,
            "bestPower": 0.5,
            "bestFrequencyIndex": 0,
            "duration": 3,
        }
        self.assertTrue(self.runtime.submit_result(work["id"], result)[0])
        self.assertTrue(self.runtime.submit_result(work["id"], result)[0])
        self.assertEqual(
            1, self.runtime.contribution_summary()["allTime"]["totalAcceptedWorkUnits"]
        )

    def test_network_wall_throughput_is_not_summed_metal_time(self):
        second = {"nodeID": "second", "capabilities": {"platform": "iOS"}}
        self.runtime.register_node(second)
        work = {
            "id": "one",
            "projectID": "p",
            "workloadID": "openstar.lomb-scargle.v1",
            "datasetID": "d",
            "payload": {"frequencyCount": 7},
        }
        metrics = {
            "workloadID": work["workloadID"],
            "sampleCount": 4,
            "frequencyCount": 7,
            "sampleFrequencyEvaluations": 28,
        }
        for node, identifier in (("node", "one"), ("second", "two")):
            self.runtime.contribution_store.record(
                session_id=self.runtime.coordinator_session_id,
                accepted_at=101,
                project_id="p",
                workload_id=work["workloadID"],
                dataset_id="d",
                work_unit_id=identifier,
                node_id=node,
                work_metrics=metrics,
                timing_metrics={"workerTotalSeconds": 3.0, "metalSeconds": 2.0},
            )
        self.runtime.coordinator_session_started_at = 100.0
        with patch("coordinator_runtime.time.time", return_value=110.0):
            summary = self.runtime.contribution_summary()
        current = summary["currentSession"]
        self.assertEqual(10.0, current["wallElapsedSeconds"])
        self.assertEqual(5.6, current["sampleFrequencyEvaluationsPerWallSecond"])
        self.assertEqual(
            14.0, current["aggregateSampleFrequencyEvaluationsPerMetalSecond"]
        )
        self.assertEqual(
            [14.0, 14.0],
            [
                node["sampleFrequencyEvaluationsPerMetalSecond"]
                for node in current["nodes"]
            ],
        )
        self.assertIsNone(summary["allTime"]["wallElapsedSeconds"])
        self.assertIsNone(summary["allTime"]["sampleFrequencyEvaluationsPerWallSecond"])

    def test_accepted_recording_does_not_copy_dataset(self):
        class NeverDeepCopied(list):
            def __deepcopy__(self, memo):
                raise AssertionError("large dataset was deep-copied")

        state = self.runtime.active_state()
        state.datasets["dataset"]["times"] = NeverDeepCopied([0, 1, 2])
        state.datasets["dataset"]["values"] = NeverDeepCopied([1, 0, 1])
        work = self.runtime.claim_work("node")
        result = {
            "status": "completed",
            "workUnitID": work["id"],
            "nodeID": "node",
            "bestFrequency": 1.0,
            "bestPower": 0.5,
            "bestFrequencyIndex": 0,
        }
        self.assertTrue(self.runtime.submit_result(work["id"], result)[0])
        self.assertEqual(
            3,
            self.runtime.contribution_summary()["currentSession"][
                "totalSampleFrequencyEvaluations"
            ],
        )

    def test_client_methods_and_defensive_node_results(self):
        client = OpenStarCoordinatorClient()
        with patch.object(
            client, "_request_json", side_effect=[[{"nodeID": "n"}], {"allTime": {}}]
        ) as request:
            nodes = client.registered_nodes()
            summary = client.contribution_summary()
        self.assertEqual([{"nodeID": "n"}], nodes)
        self.assertEqual({"allTime": {}}, summary)
        self.assertEqual(
            [
                unittest.mock.call("GET", "/v1/nodes"),
                unittest.mock.call("GET", "/v1/contributions/summary"),
            ],
            request.call_args_list,
        )
        first = self.runtime.registered_nodes()
        first[0]["capabilities"]["platform"] = "changed"
        self.assertEqual(
            "macOS", self.runtime.registered_nodes()[0]["capabilities"]["platform"]
        )

    def test_nodes_summary_and_health_http_endpoints(self):
        original = coordinator.RUNTIME
        coordinator.RUNTIME = self.runtime
        server = coordinator.ThreadingHTTPServer(
            ("127.0.0.1", 0), coordinator.RequestHandler
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(setattr, coordinator, "RUNTIME", original)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        client = OpenStarCoordinatorClient(base)
        self.assertEqual("node", client.registered_nodes()[0]["nodeID"])
        self.assertIn("currentSession", client.contribution_summary())
        with urlopen(base + "/v1/health") as response:
            health = json.loads(response.read())
        self.assertTrue(health["contributionLedger"]["ok"])

    def test_ledger_failure_does_not_rewrite_scientific_acceptance(self):
        work = self.runtime.claim_work("node")
        result = {
            "status": "completed",
            "workUnitID": work["id"],
            "nodeID": "node",
            "bestFrequency": 1.0,
            "bestPower": 0.5,
            "bestFrequencyIndex": 0,
        }
        with patch.object(
            self.runtime.contribution_store,
            "record",
            side_effect=sqlite3.OperationalError("disk full"),
        ):
            accepted, _, status = self.runtime.submit_result(work["id"], result)
        self.assertTrue(accepted)
        self.assertEqual(200, status)
        self.assertFalse(self.runtime.ledger_health()["ok"])
        self.assertIn("disk full", self.runtime.ledger_health()["error"])


if __name__ == "__main__":
    unittest.main()
