import json
import tempfile
import unittest
from pathlib import Path

from coordinator_runtime import CoordinatorRuntime
from openstar_workloads.discovery import discover_workloads
from openstar_workloads.registry import WorkloadRegistry


class WorkloadPluginPlatformTests(unittest.TestCase):
    def test_discovery_is_stable_and_unknown_ids_fail_closed(self):
        self.assertEqual(
            ["openstar.tess-period-search.v1", "openstar.lomb-scargle.v1"],
            [plugin.definition.workload_id for plugin in discover_workloads()],
        )
        with self.assertRaisesRegex(RuntimeError, "Unknown workload ID"):
            discover_workloads().require("unknown.science.v1")

    def test_duplicate_registration_fails_closed(self):
        plugin = discover_workloads().require("openstar.lomb-scargle.v1")
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            WorkloadRegistry([plugin, plugin])

    def test_runtime_supplies_one_registry_to_every_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps({"id": "d", "times": [0, 1], "values": [1, 2],
                "frequencySearch": {"minimumFrequency": 1, "frequencyStep": .1,
                                    "totalFrequencies": 1, "frequenciesPerWorkUnit": 1}}))
            project = root / "project.json"
            project.write_text(json.dumps({"id": "p", "workloadID": "openstar.lomb-scargle.v1",
                                           "datasets": [{"id": "d", "path": str(dataset)}]}))
            registry = discover_workloads()
            runtime = CoordinatorRuntime(workload_registry=registry)
            runtime.activate_project(project, require_terminal=False)
            self.assertIs(registry, runtime.active_state().workload_registry)

    def test_schema_capability_is_exact_for_new_workloads(self):
        lomb = discover_workloads().require("openstar.lomb-scargle.v1")
        self.assertTrue(lomb.definition.allows_legacy_schemaless_workers)
        self.assertEqual("openstar.result.period-search-shard.v1",
                         lomb.definition.result_schema_id)


if __name__ == "__main__":
    unittest.main()
