import json
import tempfile
import unittest
from pathlib import Path

from coordinator_runtime import CoordinatorRuntime
from openstar_workloads.contract import ResultValidation, WorkloadDefinition
from openstar_workloads.discovery import discover_workloads
from openstar_workloads.registry import WorkloadRegistry


class MinimalStrictPlugin:
    uses_legacy_coordinator_diagnostics = False
    definition = WorkloadDefinition(
        "test.strict.v1", "test.dataset.v1", "test.payload.v1", "test.result.v1"
    )

    def validate_dataset(self, dataset):
        if not isinstance(dataset.get("numbers"), list):
            raise RuntimeError("numbers required")

    def build_work_payloads(self, dataset):
        return [{"numbers": list(dataset["numbers"])}]

    def canonicalize_result(self, work_unit, result):
        return dict(result)

    def validate_result(self, work_unit, result):
        value = result.get("payload", {}).get("sum")
        if not isinstance(value, (int, float)):
            return ResultValidation(False, "numeric sum required", {})
        return ResultValidation(True, "valid", {"sum": value})

    def reduce_dataset(self, dataset, work_units, results, *, terminal):
        return {"status": "COMPLETE" if terminal else "SEARCHING",
                "sum": sum(result["payload"]["sum"] for result in results)}

    def contribution_metrics(self, work_unit, dataset):
        return {"workloadID": self.definition.workload_id,
                "numberCount": len(work_unit["payload"]["numbers"])}


class WorkloadPluginTests(unittest.TestCase):
    def test_discovery_is_deterministic_complete_and_fail_closed(self):
        self.assertEqual([
            "openstar.box-period-search.v1",
            "openstar.lomb-scargle.v1",
            "openstar.tess-period-search.v1",
        ], [plugin.definition.workload_id for plugin in discover_workloads()])
        with self.assertRaisesRegex(RuntimeError, "Unknown workload ID"):
            discover_workloads().require("client.supplied.module")

    def test_duplicate_and_invalid_plugins_fail_registry_startup(self):
        plugin = MinimalStrictPlugin()
        with self.assertRaisesRegex(RuntimeError, "Duplicate workload ID"):
            WorkloadRegistry((plugin, plugin))
        with self.assertRaisesRegex(RuntimeError, "Invalid workload plugin definition"):
            WorkloadRegistry((object(),))

    def _project(self, root, workload_id="test.strict.v1"):
        dataset = root / "dataset.json"
        dataset.write_text(json.dumps({
            "id": "dataset", "datasetSchemaID": "test.dataset.v1", "numbers": [2, 3]
        }))
        project = root / "project.json"
        project.write_text(json.dumps({
            "id": "project", "workloadID": workload_id,
            "datasetSchemaID": "test.dataset.v1", "payloadSchemaID": "test.payload.v1",
            "resultSchemaID": "test.result.v1",
            "datasets": [{"id": "dataset", "path": str(dataset)}],
        }))
        return project

    def test_minimal_strict_plugin_runs_end_to_end(self):
        registry = WorkloadRegistry((MinimalStrictPlugin(),))
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime(workload_registry=registry)
            runtime.activate_project(self._project(Path(directory)), require_terminal=False)
            schemas = {"datasetSchemaID": "test.dataset.v1",
                       "payloadSchemaID": "test.payload.v1",
                       "resultSchemaID": "test.result.v1"}
            runtime.register_node({"nodeID": "partial", "capabilities": {
                "workloads": [{"workloadID": "test.strict.v1",
                               "datasetSchemaID": "test.dataset.v1"}]}})
            runtime.register_node({"nodeID": "wrong", "capabilities": {
                "workloads": [{"workloadID": "test.strict.v1", **schemas,
                               "resultSchemaID": "wrong.result.v1"}]}})
            self.assertIsNone(runtime.claim_work("partial"))
            self.assertIsNone(runtime.claim_work("wrong"))
            runtime.register_node({"nodeID": "strict", "capabilities": {
                "workloads": [{"workloadID": "test.strict.v1", **schemas}]}})
            work = runtime.claim_work("strict")
            self.assertIsNotNone(work)
            self.assertEqual(("test.strict.v1", *schemas.values()),
                             (work["workloadID"], work["datasetSchemaID"],
                              work["payloadSchemaID"], work["resultSchemaID"]))
            self.assertEqual((False, "Result is missing resultSchemaID.", 400),
                             runtime.submit_result(work["id"], {
                                 "status": "failed", "payload": {}}))
            self.assertEqual((False, "Result schema identity mismatch.", 400),
                             runtime.submit_result(work["id"], {
                                 "status": "failed", "resultSchemaID": "wrong.result.v1",
                                 "payload": {}}))
            accepted = runtime.submit_result(work["id"], {
                "status": "completed", "resultSchemaID": "test.result.v1",
                "payload": {"sum": 5},
            })
            self.assertTrue(accepted[0])
            status = runtime.project_status()
            self.assertEqual(5, status["datasets"][0]["payload"]["sum"])
            self.assertEqual("COMPLETE", status["status"])

    def test_capability_matching_accepts_current_apple_json_only_for_legacy(self):
        apple = {"workloadID": "openstar.lomb-scargle.v1",
                 "executionBackends": ["metal"],
                 "validatorID": "openstar.lomb-scargle.validator.v1"}
        plugin = discover_workloads().require("openstar.lomb-scargle.v1")
        self.assertTrue(plugin.definition.allows_legacy_schemaless_workers)
        # Exercise the exact registration shape rather than a string shorthand.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps({"id": "d", "times": [0, 1], "values": [1, 2],
                "frequencySearch": {"minimumFrequency": 1, "frequencyStep": .1,
                                    "totalFrequencies": 1, "frequenciesPerWorkUnit": 1}}))
            project = root / "project.json"
            project.write_text(json.dumps({"id": "p", "workloadID": apple["workloadID"],
                                           "datasets": [{"id": "d", "path": str(dataset)}]}))
            runtime = CoordinatorRuntime()
            runtime.activate_project(project, require_terminal=False)
            runtime.register_node({"nodeID": "apple", "capabilities": {"workloads": [apple]}})
            self.assertIsNotNone(runtime.claim_work("apple"))

    def test_box_workload_remains_distinct(self):
        box = discover_workloads().require("openstar.box-period-search.v1")
        alias = discover_workloads().require("openstar.tess-period-search.v1")
        dataset = {"times": [0, 1, 2], "periodSearch": {
            "minimumPeriodDays": 2, "periodStepDays": .5,
            "totalPeriods": 3, "periodsPerWorkUnit": 2}}
        payloads = box.build_work_payloads(dataset)
        self.assertEqual("periodStartIndex", next(iter(payloads[0])))
        self.assertTrue(alias.uses_legacy_coordinator_diagnostics)
        work = {"payload": payloads[0], **payloads[0]}
        self.assertEqual(6, box.contribution_metrics(work, dataset)["samplePeriodEvaluations"])
        reduced = box.reduce_dataset(dataset, [], [
            {"bestPeriodDays": 2, "bestScore": .4},
            {"bestPeriodDays": 2.5, "bestScore": .8}], terminal=True)
        self.assertEqual((2.5, .8), (reduced["bestPeriodDays"], reduced["bestScore"]))


if __name__ == "__main__":
    unittest.main()
