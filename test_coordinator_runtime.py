import json
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import coordinator
from openstar_coordinator_client import (
    CoordinatorClientError, CoordinatorUnavailableError,
    OpenStarCoordinatorClient,
)
from openstar_workflow import RetryableExecutionError
from coordinator_runtime import (
    CoordinatorRuntime,
    ProjectBusyError,
    ProjectConflictError,
)


class CoordinatorRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, project_id, workload="workload", dataset_id="shared", name=None):
        directory = self.root / (name or project_id)
        directory.mkdir()
        dataset = directory / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "marker": project_id,
                    "times": [0.0, 1.0, 2.0],
                    "values": [1.0, 0.0, 1.0],
                    "frequencySearch": {
                        "minimumFrequency": 1.0,
                        "frequencyStep": 0.1,
                        "totalFrequencies": 2,
                        "frequenciesPerWorkUnit": 1,
                    },
                }
            )
        )
        manifest = directory / "project.json"
        manifest.write_text(
            json.dumps(
                {
                    "id": project_id,
                    "workloadID": workload,
                    "datasets": [{"id": dataset_id, "path": str(dataset)}],
                }
            )
        )
        return manifest

    def activate_two(self):
        runtime = CoordinatorRuntime()
        runtime.activate_project(self.manifest("a"), require_terminal=False)
        runtime.activate_project(self.manifest("b"), require_terminal=False)
        return runtime

    def test_projects_coexist_and_legacy_current_does_not_hide_status(self):
        runtime = self.activate_two()
        self.assertEqual(["a", "b"], [item["projectID"] for item in runtime.projects()])
        self.assertEqual("b", runtime.project_status()["projectID"])
        self.assertEqual("a", runtime.project_status("a")["projectID"])

    def test_round_robin_two_and_three_projects(self):
        runtime = self.activate_two()
        runtime.activate_project(self.manifest("c"), require_terminal=False)
        runtime.register_node({"nodeID": "node", "capabilities": {}})
        self.assertEqual(
            ["a", "b", "c", "a", "b", "c"],
            [runtime.claim_work("node")["projectID"] for _ in range(6)],
        )

    def test_concurrent_claims_advance_project_cursor_atomically(self):
        runtime = self.activate_two()
        for node_id in ("node-1", "node-2"):
            runtime.register_node({"nodeID": node_id, "capabilities": {}})

        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        original_claim = runtime._states["a"].claim_work
        call_lock = threading.Lock()
        call_count = 0

        def overlapping_claim(node_id):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                first_entered.set()
                self.assertTrue(release_first.wait(2))
            else:
                second_entered.set()
            return original_claim(node_id)

        claimed_projects = {}
        errors = []

        def claim(node_id):
            try:
                claimed_projects[node_id] = runtime.claim_work(node_id)["projectID"]
            except Exception as error:
                errors.append(error)

        with patch.object(
            runtime._states["a"], "claim_work", side_effect=overlapping_claim
        ):
            first = threading.Thread(target=claim, args=("node-1",))
            second = threading.Thread(target=claim, args=("node-2",))
            first.start()
            self.assertTrue(first_entered.wait(2))
            second.start()
            second_entered.wait(0.1)
            release_first.set()
            first.join(2)
            second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual([], errors)
        self.assertEqual({"node-1": "a", "node-2": "b"}, claimed_projects)

    def test_empty_and_incompatible_projects_are_skipped(self):
        runtime = CoordinatorRuntime()
        runtime.activate_project(
            self.manifest("a", "unsupported"), require_terminal=False
        )
        runtime.activate_project(
            self.manifest("b", "supported"), require_terminal=False
        )
        runtime.register_node(
            {"nodeID": "node", "capabilities": {"workloads": ["supported"]}}
        )
        self.assertEqual("b", runtime.claim_work("node")["projectID"])

    def test_registrations_replay_before_and_after_activation(self):
        runtime = CoordinatorRuntime()
        runtime.register_node({"nodeID": "early", "capabilities": {}})
        runtime.activate_project(self.manifest("a"), require_terminal=False)
        runtime.activate_project(self.manifest("b"), require_terminal=False)
        runtime.register_node({"nodeID": "late", "capabilities": {}})
        self.assertEqual({"early", "late"}, set(runtime._states["a"].nodes))
        self.assertEqual({"early", "late"}, set(runtime._states["b"].nodes))

    def test_results_and_duplicate_dataset_ids_are_project_isolated(self):
        runtime = self.activate_two()
        runtime.register_node({"nodeID": "node", "capabilities": {}})
        work = runtime.claim_work("node")
        owner = work["projectID"]
        other = "b" if owner == "a" else "a"
        result = {
            "status": "completed",
            "bestFrequency": 1.0,
            "bestPower": 0.5,
            "bestFrequencyIndex": 0,
        }
        self.assertTrue(runtime.submit_result(work["id"], result)[0])
        self.assertEqual(1, len(runtime._states[owner].completed))
        self.assertEqual(0, len(runtime._states[other].completed))
        self.assertEqual("a", runtime.dataset("shared", "a")["marker"])
        self.assertEqual("b", runtime.dataset("shared", "b")["marker"])

    def test_idempotence_path_conflict_and_serialized_activation(self):
        runtime = CoordinatorRuntime()
        path = self.manifest("a")
        runtime.activate_project(path, require_terminal=False)
        runtime.activate_project(path, require_terminal=False)
        self.assertEqual(["a"], runtime._project_order)
        with self.assertRaises(ProjectBusyError):
            runtime.activate_project(self.manifest("b"), require_terminal=True)
        with self.assertRaises(ProjectConflictError):
            runtime.activate_project(
                self.manifest("a", name="other-a"), require_terminal=False
            )

    def test_work_id_collision_is_rejected(self):
        runtime = CoordinatorRuntime()
        with patch("coordinator_state.uuid.uuid4", return_value="same-work"):
            runtime.activate_project(self.manifest("a"), require_terminal=False)
            with self.assertRaises(ProjectConflictError):
                runtime.activate_project(self.manifest("b"), require_terminal=False)

    def test_removal_rules_preserve_other_project_nodes_and_cursor(self):
        runtime = self.activate_two()
        runtime.register_node({"nodeID": "node", "capabilities": {}})
        with self.assertRaises(ProjectBusyError):
            runtime.remove_project("a")
        state = runtime._states["a"]
        state.failed.update({work_id: {} for work_id in state.work_units})
        state.pending.clear()
        runtime._next_project_index = 1
        runtime.remove_project("a")
        self.assertEqual(["b"], runtime._project_order)
        self.assertIn("node", runtime._states["b"].nodes)
        self.assertEqual(0, runtime._next_project_index)
        self.assertEqual("b", runtime.claim_work("node")["projectID"])


class CoordinatorHTTPContractTests(unittest.TestCase):
    manifest = CoordinatorRuntimeTests.manifest
    activate_two = CoordinatorRuntimeTests.activate_two

    def setUp(self):
        CoordinatorRuntimeTests.setUp(self)
        self.runtime = self.activate_two()
        self.original_runtime = coordinator.RUNTIME
        coordinator.RUNTIME = self.runtime
        self.addCleanup(setattr, coordinator, "RUNTIME", self.original_runtime)
        self.server = coordinator.ThreadingHTTPServer(
            ("127.0.0.1", 0), coordinator.RequestHandler
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def get(self, path):
        with urlopen(self.base + path) as response:
            return response.status, json.loads(response.read())

    def test_project_list_specific_status_and_dataset_endpoints(self):
        self.assertEqual(
            ["a", "b"], [item["projectID"] for item in self.get("/v1/projects")[1]]
        )
        self.assertEqual("a", self.get("/v1/projects/a/status")[1]["projectID"])
        self.assertEqual("a", self.get("/v1/projects/a/datasets/shared")[1]["marker"])
        with self.assertRaises(HTTPError) as missing:
            self.get("/v1/projects/missing/status")
        self.assertEqual(404, missing.exception.code)

    def test_delete_contract_rejects_running_and_removes_terminal(self):
        with self.assertRaises(HTTPError) as running:
            urlopen(Request(self.base + "/v1/projects/a", method="DELETE"))
        self.assertEqual(409, running.exception.code)
        state = self.runtime._states["a"]
        state.failed.update({work_id: {} for work_id in state.work_units})
        state.pending.clear()
        with urlopen(
            Request(self.base + "/v1/projects/a", method="DELETE")
        ) as response:
            self.assertEqual(200, response.status)
        with self.assertRaises(HTTPError) as missing:
            self.get("/v1/projects/a/status")
        self.assertEqual(404, missing.exception.code)


class CoordinatorClientTests(unittest.TestCase):
    def test_raw_transient_transport_failures_are_retryable(self):
        client = OpenStarCoordinatorClient()
        failures = (
            ConnectionResetError("reset"), ConnectionAbortedError("aborted"),
            BrokenPipeError("broken"), ConnectionRefusedError("refused"),
            TimeoutError("timed out"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), patch(
                "openstar_coordinator_client.urlopen", side_effect=failure
            ), self.assertRaises(CoordinatorUnavailableError) as raised:
                client.health()
            self.assertIsInstance(raised.exception, RetryableExecutionError)

    def test_http_errors_remain_non_retryable_client_errors(self):
        error = HTTPError("http://coordinator", 400, "bad", {}, BytesIO(
            b'{"message":"invalid project"}'
        ))
        with patch("openstar_coordinator_client.urlopen", side_effect=error), \
                self.assertRaises(CoordinatorClientError) as raised:
            OpenStarCoordinatorClient().health()
        self.assertNotIsInstance(raised.exception, CoordinatorUnavailableError)
        self.assertNotIsInstance(raised.exception, RetryableExecutionError)

    def test_malformed_response_remains_non_retryable(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"not json"
        with patch("openstar_coordinator_client.urlopen", return_value=response), \
                self.assertRaises(CoordinatorClientError) as raised:
            OpenStarCoordinatorClient().health()
        self.assertNotIsInstance(raised.exception, RetryableExecutionError)

    def test_run_projects_activates_batch_then_polls_in_order(self):
        client = OpenStarCoordinatorClient()
        activations = [{"projectID": "a"}, {"projectID": "b"}]
        statuses = [
            {"status": "COMPLETE", "nodeContributions": {"node": 2}},
            {"status": "RUNNING"},
            {"status": "COMPLETE", "nodeContributions": {"node": 3}},
        ]
        events = []

        def activate(path, *, require_terminal):
            events.append(("activate", path, require_terminal))
            return activations.pop(0)

        def status(project_id):
            events.append(("status", project_id))
            return statuses.pop(0)

        with patch.object(
            client, "activate_project", side_effect=activate
        ), patch.object(client, "project_status", side_effect=status), patch(
            "openstar_coordinator_client.time.sleep"
        ) as sleep:
            result = client.run_projects(["one.json", "two.json"], poll_interval=0)

        self.assertEqual(("a", "b"), result.project_ids)
        self.assertEqual({"node": 5}, result.node_contributions)
        self.assertEqual(
            [("activate", "one.json", False), ("activate", "two.json", False)],
            events[:2],
        )
        self.assertEqual(
            [("status", "a"), ("status", "b"), ("status", "b")], events[2:]
        )
        sleep.assert_called_once()

    def test_run_projects_rejects_empty_and_duplicate_project_ids(self):
        client = OpenStarCoordinatorClient()
        with self.assertRaises(ValueError):
            client.run_projects([])
        with patch.object(
            client, "activate_project", return_value={"projectID": "same"}
        ), self.assertRaisesRegex(ValueError, "duplicate project ID"):
            client.run_projects(["one.json", "two.json"])

    def test_run_projects_activation_time_counts_toward_overall_timeout(self):
        client = OpenStarCoordinatorClient()
        clock = [0.0]

        def activate(path, *, require_terminal):
            clock[0] = 2.0
            return {"projectID": "a"}

        with patch.object(client, "activate_project", side_effect=activate), patch(
            "openstar_coordinator_client.time.monotonic",
            side_effect=lambda: clock[0],
        ), self.assertRaises(TimeoutError):
            client.run_projects(["one.json"], timeout=1.0)

    def test_run_projects_terminal_status_after_deadline_times_out(self):
        client = OpenStarCoordinatorClient()
        clock = [0.0]

        def status(project_id):
            clock[0] = 2.0
            return {"projectID": project_id, "status": "COMPLETE"}

        with patch.object(
            client, "activate_project", return_value={"projectID": "a"}
        ), patch.object(client, "project_status", side_effect=status), patch(
            "openstar_coordinator_client.time.monotonic",
            side_effect=lambda: clock[0],
        ), self.assertRaises(
            TimeoutError
        ):
            client.run_projects(["one.json"], timeout=1.0)

    def test_run_projects_completion_before_deadline_succeeds(self):
        client = OpenStarCoordinatorClient()
        complete = {
            "projectID": "a",
            "status": "COMPLETE",
            "nodeContributions": {"node": 4},
        }
        with patch.object(
            client, "activate_project", return_value={"projectID": "a"}
        ), patch.object(client, "project_status", return_value=complete), patch(
            "openstar_coordinator_client.time.monotonic", return_value=0.0
        ):
            result = client.run_projects(["one.json"], timeout=1.0)
        self.assertEqual(("a",), result.project_ids)
        self.assertEqual({"node": 4}, result.node_contributions)

    def test_wait_for_project_uses_project_specific_status(self):
        client = OpenStarCoordinatorClient()
        running = {"status": "RUNNING", "projectTotalWorkUnits": 1}
        complete = {"status": "COMPLETE", "projectID": "a"}
        with patch.object(
            client, "project_status", side_effect=[running, complete]
        ) as status, patch("openstar_coordinator_client.time.sleep"):
            self.assertEqual(complete, client.wait_for_project("a", poll_interval=0))
        self.assertEqual(
            [unittest.mock.call("a"), unittest.mock.call("a")], status.call_args_list
        )

    def test_run_project_activates_concurrently_and_waits_for_returned_id(self):
        client = OpenStarCoordinatorClient()
        complete = {"status": "COMPLETE", "projectID": "a"}
        with patch.object(
            client, "activate_project", return_value={"projectID": "a"}
        ) as activate, patch.object(
            client, "wait_for_project", return_value=complete
        ) as wait:
            result = client.run_project("project.json")
        activate.assert_called_once_with("project.json", require_terminal=False)
        wait.assert_called_once_with("a", poll_interval=1.0, timeout=None)
        self.assertEqual(complete, result.status)


if __name__ == "__main__":
    unittest.main()
