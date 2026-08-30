import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from coordinator_runtime import CoordinatorRuntime
from coordinator_state import CoordinatorState
from openstar_workloads.contract import (
    DatasetReduction,
    ResultValidation,
    WorkloadDefinition,
)
from openstar_workloads.discovery import discover_workloads
from openstar_workloads.registry import WorkloadRegistry


STRICT_WORKLOAD_ID = "test.strict.v1"
STRICT_SCHEMAS = {
    "datasetSchemaID": "test.dataset.v1",
    "payloadSchemaID": "test.payload.v1",
    "resultSchemaID": "test.result.v1",
}


class MinimalStrictPlugin:
    uses_legacy_coordinator_diagnostics = False
    uses_legacy_science_metadata_validation = False
    definition = WorkloadDefinition(
        STRICT_WORKLOAD_ID,
        STRICT_SCHEMAS["datasetSchemaID"],
        STRICT_SCHEMAS["payloadSchemaID"],
        STRICT_SCHEMAS["resultSchemaID"],
    )

    def __init__(self):
        self.build_calls = 0

    def validate_dataset(self, dataset):
        numbers = dataset.get("numbers")
        if not isinstance(numbers, list) or not numbers:
            raise RuntimeError("numbers must be a nonempty list")

    def build_work_payloads(self, dataset):
        self.build_calls += 1
        yield {"numbers": list(dataset["numbers"])}

    def legacy_work_unit_fields(self, payload):
        return {}

    def canonicalize_result(self, work_unit, result):
        return dict(result)

    def validate_result(self, work_unit, result, dataset):
        if result.get("status") != "completed":
            return ResultValidation(False, "Work unit did not complete.", {})
        payload = result.get("payload")
        value = payload.get("sum") if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ResultValidation(False, "numeric sum required", {})
        if value != sum(dataset["numbers"]):
            return ResultValidation(False, "sum does not match the dataset", {})
        return ResultValidation(True, "strict result is valid", {"sum": value})

    def reduce_dataset(self, dataset, work_units, results, *, terminal):
        total = sum(
            result["payload"]["sum"]
            for result in results
            if result is not None
        )
        return DatasetReduction(
            payload={"sum": total},
            status_fields={
                "workloadStatus": "COMPLETE" if terminal else "SEARCHING"
            },
        )

    def contribution_metrics(self, work_unit, dataset):
        return {
            "workloadID": self.definition.workload_id,
            "numberCount": len(work_unit["payload"]["numbers"]),
        }


class EnvelopeMutatingStrictPlugin(MinimalStrictPlugin):
    def __init__(self, mutation):
        super().__init__()
        self.mutation = mutation

    def canonicalize_result(self, work_unit, result):
        canonical = dict(result)
        if self.mutation == "drop-result-schema":
            canonical.pop("resultSchemaID", None)
        elif self.mutation == "alter-result-schema":
            canonical["resultSchemaID"] = "wrong.result.v1"
        elif self.mutation == "drop-status":
            canonical.pop("status", None)
        elif self.mutation == "alter-status":
            canonical["status"] = "failed"
        return canonical


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _strict_project(
    root,
    *,
    project_schemas=STRICT_SCHEMAS,
    dataset_schema=STRICT_SCHEMAS["datasetSchemaID"],
    dataset_fields=None,
    project_id="strict-project",
):
    dataset = {"id": "numbers", "numbers": [2, 3]}
    if dataset_schema is not None:
        dataset["datasetSchemaID"] = dataset_schema
    if dataset_fields is not None:
        dataset.update(dataset_fields)
    dataset_path = root / f"{project_id}-dataset.json"
    _write_json(dataset_path, dataset)

    project = {
        "id": project_id,
        "workloadID": STRICT_WORKLOAD_ID,
        "datasets": [{"id": "numbers", "path": str(dataset_path)}],
    }
    if project_schemas is not None:
        project.update(project_schemas)
    project_path = root / f"{project_id}.json"
    _write_json(project_path, project)
    return project_path


class WorkloadRegistryTests(unittest.TestCase):
    def test_discovery_is_deterministic_ordered_unique_and_fail_closed(self):
        first = [plugin.definition.workload_id for plugin in discover_workloads()]
        second = [plugin.definition.workload_id for plugin in discover_workloads()]
        self.assertTrue(first)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), first)
        self.assertEqual(len(set(first)), len(first))
        with self.assertRaisesRegex(RuntimeError, "Unknown workload ID"):
            discover_workloads().require("client.supplied.module")

    def test_duplicate_and_malformed_plugins_fail_registry_construction(self):
        first = MinimalStrictPlugin()
        second = MinimalStrictPlugin()
        with self.assertRaisesRegex(RuntimeError, "Duplicate workload ID"):
            WorkloadRegistry((first, second))

        class MissingOperations:
            uses_legacy_coordinator_diagnostics = False
            uses_legacy_science_metadata_validation = False
            definition = WorkloadDefinition(
                "test.malformed.v1",
                "test.dataset.malformed.v1",
                "test.payload.malformed.v1",
                "test.result.malformed.v1",
            )

        with self.assertRaises(RuntimeError):
            WorkloadRegistry((MissingOperations(),))
        with self.assertRaises(ValueError):
            WorkloadDefinition(
                " test.not-canonical.v1",
                "test.dataset.v1",
                "test.payload.v1",
                "test.result.v1",
            )


class WorkloadSchemaRoutingTests(unittest.TestCase):
    def test_generic_dataset_owns_opaque_reference_metadata_and_science(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = MinimalStrictPlugin()
            state = CoordinatorState(
                _strict_project(
                    Path(directory),
                    dataset_fields={
                        "reference": "plugin-owned-reference",
                        "metadata": 17,
                        "science": False,
                    },
                ),
                workload_registry=WorkloadRegistry((plugin,)),
            )
            state.validate_startup()

            dataset = state.datasets["numbers"]
            self.assertEqual("plugin-owned-reference", dataset["reference"])
            self.assertEqual(17, dataset["metadata"])
            self.assertIs(False, dataset["science"])
            dataset_status = state.dataset_status("numbers")
            self.assertEqual("SEARCHING", dataset_status["workloadStatus"])
            self.assertEqual({"sum": 0}, dataset_status["payload"])
            project_status = state.project_status()
            self.assertEqual(1, project_status["projectTotalWorkUnits"])
            self.assertEqual(0, project_status["projectCompletedWorkUnits"])

            output = io.StringIO()
            with redirect_stdout(output):
                state.print_startup_summary(8765)
            summary = output.getvalue()
            self.assertIn("registered workload plugin", summary)
            for legacy_term in ("Astropy", "Metal", "TESS", "harmonic"):
                self.assertNotIn(legacy_term, summary)

    def test_strict_project_and_dataset_require_the_exact_schema_identity(self):
        invalid_cases = (
            (None, STRICT_SCHEMAS["datasetSchemaID"]),
            ({"datasetSchemaID": STRICT_SCHEMAS["datasetSchemaID"]},
             STRICT_SCHEMAS["datasetSchemaID"]),
            ({**STRICT_SCHEMAS, "payloadSchemaID": "wrong.payload.v1"},
             STRICT_SCHEMAS["datasetSchemaID"]),
            (STRICT_SCHEMAS, None),
            (STRICT_SCHEMAS, "wrong.dataset.v1"),
        )
        for index, (project_schemas, dataset_schema) in enumerate(invalid_cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                project_path = _strict_project(
                    root,
                    project_schemas=project_schemas,
                    dataset_schema=dataset_schema,
                )
                registry = WorkloadRegistry((MinimalStrictPlugin(),))
                with self.assertRaises(RuntimeError):
                    CoordinatorState(project_path, workload_registry=registry)

        with tempfile.TemporaryDirectory() as directory:
            plugin = MinimalStrictPlugin()
            state = CoordinatorState(
                _strict_project(Path(directory)),
                workload_registry=WorkloadRegistry((plugin,)),
            )
            state.validate_startup()
            self.assertEqual(1, plugin.build_calls)

    def test_capability_schema_tuple_is_atomic_and_stamped_on_work(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = MinimalStrictPlugin()
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((plugin,))
            )
            runtime.activate_project(
                _strict_project(Path(directory)), require_terminal=False
            )
            capabilities = {
                "none": {},
                "legacy-shaped": {
                    "workloads": [{
                        "workloadID": STRICT_WORKLOAD_ID,
                        "executionBackends": [{"id": "metal"}],
                        "validatorID": "test.strict.validator.v1",
                    }]
                },
                "partial": {
                    "workloads": [{
                        "workloadID": STRICT_WORKLOAD_ID,
                        "datasetSchemaID": STRICT_SCHEMAS["datasetSchemaID"],
                    }]
                },
                "wrong": {
                    "workloads": [{
                        "workloadID": STRICT_WORKLOAD_ID,
                        **STRICT_SCHEMAS,
                        "resultSchemaID": "wrong.result.v1",
                    }]
                },
                "exact": {
                    "workloads": [{
                        "workloadID": STRICT_WORKLOAD_ID,
                        **STRICT_SCHEMAS,
                        "executionBackends": [{"id": "metal"}],
                    }]
                },
            }
            for node_id, advertised in capabilities.items():
                runtime.register_node({
                    "nodeID": node_id,
                    "capabilities": advertised,
                })

            for node_id in ("none", "legacy-shaped", "partial", "wrong"):
                self.assertIsNone(runtime.claim_work(node_id), node_id)
            work = runtime.claim_work("exact")
            self.assertIsNotNone(work)
            self.assertEqual(STRICT_WORKLOAD_ID, work["workloadID"])
            for key, value in STRICT_SCHEMAS.items():
                self.assertEqual(value, work[key])
            self.assertEqual({"numbers": [2, 3]}, work["payload"])
            self.assertNotIn("numbers", work)

    def test_schema_less_capability_is_rejected_for_strict_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((MinimalStrictPlugin(),))
            )
            runtime.activate_project(
                _strict_project(Path(directory)), require_terminal=False
            )
            strict_without_schemas = {
                "workloadID": STRICT_WORKLOAD_ID,
                "executionBackends": [{"id": "accelerator"}],
                "validatorID": "test.strict.validator.v1",
            }
            runtime.register_node({
                "nodeID": "schema-less-strict",
                "capabilities": {"workloads": [strict_without_schemas]},
            })
            self.assertIsNone(runtime.claim_work("schema-less-strict"))

    def test_result_schema_is_checked_before_completed_or_failed_semantics(self):
        for status in ("completed", "failed"):
            for supplied_schema in (None, "wrong.result.v1"):
                with (
                    self.subTest(status=status, supplied_schema=supplied_schema),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    runtime = CoordinatorRuntime(
                        workload_registry=WorkloadRegistry((MinimalStrictPlugin(),))
                    )
                    runtime.activate_project(
                        _strict_project(Path(directory)), require_terminal=False
                    )
                    runtime.register_node({
                        "nodeID": "strict",
                        "capabilities": {"workloads": [{
                            "workloadID": STRICT_WORKLOAD_ID,
                            **STRICT_SCHEMAS,
                        }]},
                    })
                    work = runtime.claim_work("strict")
                    result = {"status": status, "payload": {"sum": 5}}
                    if supplied_schema is not None:
                        result["resultSchemaID"] = supplied_schema
                    accepted, message, code = runtime.submit_result(
                        work["id"], result
                    )
                    self.assertFalse(accepted)
                    self.assertEqual(400, code)
                    self.assertIn("schema", message.lower())

    def test_canonicalizer_cannot_mutate_core_result_envelope(self):
        mutations = (
            "drop-result-schema",
            "alter-result-schema",
            "drop-status",
            "alter-status",
        )
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                plugin = EnvelopeMutatingStrictPlugin(mutation)
                runtime = CoordinatorRuntime(
                    workload_registry=WorkloadRegistry((plugin,))
                )
                runtime.activate_project(
                    _strict_project(Path(directory)), require_terminal=False
                )
                runtime.register_node({
                    "nodeID": "strict",
                    "capabilities": {"workloads": [{
                        "workloadID": STRICT_WORKLOAD_ID,
                        **STRICT_SCHEMAS,
                    }]},
                })
                work = runtime.claim_work("strict")
                accepted, _, code = runtime.submit_result(work["id"], {
                    "status": "completed",
                    "resultSchemaID": STRICT_SCHEMAS["resultSchemaID"],
                    "workUnitID": work["id"],
                    "nodeID": "strict",
                    "payload": {"sum": 5},
                })
                self.assertFalse(accepted)
                self.assertEqual(400, code)
                status = runtime.project_status()
                self.assertEqual(0, status["projectCompletedWorkUnits"])

    def test_malformed_capabilities_registration_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((MinimalStrictPlugin(),))
            )
            runtime.activate_project(
                _strict_project(Path(directory)), require_terminal=False
            )
            runtime.register_node({
                "nodeID": "valid",
                "capabilities": {"workloads": [{
                    "workloadID": STRICT_WORKLOAD_ID,
                    **STRICT_SCHEMAS,
                }]},
            })
            before_runtime = [
                node["nodeID"] for node in runtime.registered_nodes()
            ]
            before_state = set(runtime.active_state().nodes)

            with self.assertRaises((TypeError, ValueError)):
                runtime.register_node({
                    "nodeID": "malformed",
                    "capabilities": "not-a-capability-mapping",
                })

            self.assertEqual(
                before_runtime,
                [node["nodeID"] for node in runtime.registered_nodes()],
            )
            self.assertEqual(before_state, set(runtime.active_state().nodes))

    def test_strict_result_reduces_and_generic_project_status_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = MinimalStrictPlugin()
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((plugin,))
            )
            project_path = _strict_project(Path(directory))
            runtime.activate_project(project_path, require_terminal=False)
            self.assertEqual(1, plugin.build_calls)
            runtime.register_node({
                "nodeID": "strict",
                "capabilities": {"workloads": [{
                    "workloadID": STRICT_WORKLOAD_ID,
                    **STRICT_SCHEMAS,
                }]},
            })
            work = runtime.claim_work("strict")
            accepted, _, code = runtime.submit_result(work["id"], {
                "status": "completed",
                "resultSchemaID": STRICT_SCHEMAS["resultSchemaID"],
                "workUnitID": work["id"],
                "nodeID": "strict",
                "payload": {"sum": 5},
            })
            self.assertTrue(accepted)
            self.assertEqual(200, code)

            status = runtime.project_status()
            self.assertEqual("COMPLETE", status["status"])
            self.assertEqual(1, status["projectCompletedWorkUnits"])
            dataset_status = status["datasets"][0]
            self.assertEqual({"sum": 5}, dataset_status["payload"])
            self.assertEqual("COMPLETE", dataset_status["workloadStatus"])

    def test_late_result_cannot_be_credited_to_a_new_lease_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((MinimalStrictPlugin(),))
            )
            runtime.activate_project(
                _strict_project(Path(directory)), require_terminal=False
            )
            capabilities = {"workloads": [{
                "workloadID": STRICT_WORKLOAD_ID,
                **STRICT_SCHEMAS,
            }]}
            runtime.register_node({
                "nodeID": "first",
                "capabilities": capabilities,
            })
            runtime.register_node({
                "nodeID": "second",
                "capabilities": capabilities,
            })

            first = runtime.claim_work("first")
            runtime.active_state().assigned[first["id"].lower()][
                "leaseExpiresAt"
            ] = 0.0
            second = runtime.claim_work("second")
            self.assertEqual(first["id"], second["id"])

            result = {
                "status": "completed",
                "resultSchemaID": STRICT_SCHEMAS["resultSchemaID"],
                "workUnitID": first["id"],
                "nodeID": "first",
                "payload": {"sum": 5},
            }
            accepted, message, code = runtime.submit_result(
                first["id"], result
            )
            self.assertFalse(accepted)
            self.assertEqual(409, code)
            self.assertIn("node identity", message.lower())
            self.assertEqual(
                "second",
                runtime.active_state().assigned[first["id"].lower()]["nodeID"],
            )

            accepted, _, code = runtime.submit_result(
                second["id"],
                {**result, "nodeID": "second"},
            )
            self.assertTrue(accepted)
            self.assertEqual(200, code)

    def test_schema_aware_result_requires_work_and_node_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((MinimalStrictPlugin(),))
            )
            runtime.activate_project(
                _strict_project(Path(directory)), require_terminal=False
            )
            runtime.register_node({
                "nodeID": "strict",
                "capabilities": {"workloads": [{
                    "workloadID": STRICT_WORKLOAD_ID,
                    **STRICT_SCHEMAS,
                }]},
            })
            work = runtime.claim_work("strict")
            result = {
                "status": "completed",
                "resultSchemaID": STRICT_SCHEMAS["resultSchemaID"],
                "workUnitID": work["id"],
                "nodeID": "strict",
                "payload": {"sum": 5},
            }

            for missing_key in ("workUnitID", "nodeID"):
                with self.subTest(missing_key=missing_key):
                    incomplete = dict(result)
                    del incomplete[missing_key]
                    accepted, message, code = runtime.submit_result(
                        work["id"], incomplete
                    )
                    self.assertFalse(accepted)
                    self.assertEqual(400, code)
                    self.assertIn("identity", message.lower())
                    self.assertEqual(
                        0,
                        runtime.project_status()["projectCompletedWorkUnits"],
                    )

            self.assertTrue(runtime.submit_result(work["id"], result)[0])

    def test_operational_output_cannot_change_registration_or_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((MinimalStrictPlugin(),))
            )
            runtime.activate_project(
                _strict_project(Path(directory)), require_terminal=False
            )
            with patch("builtins.print", side_effect=OSError("stdout closed")):
                runtime.register_node({
                    "nodeID": "strict",
                    "capabilities": {"workloads": [{
                        "workloadID": STRICT_WORKLOAD_ID,
                        **STRICT_SCHEMAS,
                    }]},
                })
                work = runtime.claim_work("strict")
                accepted, _, code = runtime.submit_result(work["id"], {
                    "status": "completed",
                    "resultSchemaID": STRICT_SCHEMAS["resultSchemaID"],
                    "workUnitID": work["id"],
                    "nodeID": "strict",
                    "payload": {"sum": 5},
                })

            self.assertTrue(accepted)
            self.assertEqual(200, code)
            self.assertEqual(1, runtime.project_status()["projectCompletedWorkUnits"])

    def test_valid_schema_failures_and_invalid_results_requeue_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime(
                workload_registry=WorkloadRegistry((MinimalStrictPlugin(),))
            )
            runtime.activate_project(
                _strict_project(Path(directory)), require_terminal=False
            )
            exact_capabilities = {"workloads": [{
                "workloadID": STRICT_WORKLOAD_ID,
                **STRICT_SCHEMAS,
            }]}
            runtime.register_node({
                "nodeID": "strict-a",
                "capabilities": exact_capabilities,
            })
            runtime.register_node({
                "nodeID": "strict-b",
                "capabilities": exact_capabilities,
            })

            failed_work = runtime.claim_work("strict-a")
            accepted, _, code = runtime.submit_result(failed_work["id"], {
                "status": "failed",
                "failureKind": "environment-unavailable",
                "resultSchemaID": STRICT_SCHEMAS["resultSchemaID"],
                "workUnitID": failed_work["id"],
                "nodeID": "strict-a",
                "payload": {},
            })
            self.assertFalse(accepted)
            self.assertEqual(200, code)

            retried_work = runtime.claim_work("strict-b")
            self.assertEqual(failed_work["id"], retried_work["id"])
            accepted, _, code = runtime.submit_result(retried_work["id"], {
                "status": "completed",
                "resultSchemaID": STRICT_SCHEMAS["resultSchemaID"],
                "workUnitID": retried_work["id"],
                "nodeID": "strict-b",
                "payload": {"sum": 4},
            })
            self.assertFalse(accepted)
            self.assertEqual(200, code)

            status = runtime.project_status()
            self.assertEqual("RUNNING", status["status"])
            self.assertEqual(1, status["projectPendingWorkUnits"])
            self.assertEqual(0, status["projectAssignedWorkUnits"])
            self.assertEqual(0, status["projectCompletedWorkUnits"])
            self.assertEqual({"sum": 0}, status["datasets"][0]["payload"])


if __name__ == "__main__":
    unittest.main()
