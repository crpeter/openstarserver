import json
import tempfile
import unittest
from pathlib import Path

from coordinator_runtime import CoordinatorRuntime
from openstar_workloads.plugins import curve_grid
from tests.workloads.curve_grid.test_curve_grid_plugin import (
    golden_dataset,
    worker_payload,
)


SCHEMA_TUPLE = {
    "datasetSchemaID": curve_grid.DATASET_SCHEMA_ID,
    "payloadSchemaID": curve_grid.PAYLOAD_SCHEMA_ID,
    "resultSchemaID": curve_grid.RESULT_SCHEMA_ID,
}


def write_project(root):
    dataset_path = root / "curve-grid-dataset.json"
    dataset_path.write_text(
        json.dumps(golden_dataset()),
        encoding="utf-8",
    )
    project_path = root / "curve-grid-project.json"
    project_path.write_text(
        json.dumps(
            {
                "id": "curve-grid-project",
                "workloadID": curve_grid.WORKLOAD_ID,
                **SCHEMA_TUPLE,
                "datasets": [
                    {
                        "id": "golden-curve-grid",
                        "path": str(dataset_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return project_path


class CurveGridCoordinatorIntegrationTests(unittest.TestCase):
    def test_incomplete_schema_tuple_cannot_claim_strict_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime()
            runtime.activate_project(
                write_project(Path(directory)),
                require_terminal=False,
            )
            capabilities = {
                "schema-less": {
                    "workloadID": curve_grid.WORKLOAD_ID,
                },
                "partial": {
                    "workloadID": curve_grid.WORKLOAD_ID,
                    "datasetSchemaID": curve_grid.DATASET_SCHEMA_ID,
                    "payloadSchemaID": curve_grid.PAYLOAD_SCHEMA_ID,
                },
                "wrong": {
                    "workloadID": curve_grid.WORKLOAD_ID,
                    **SCHEMA_TUPLE,
                    "resultSchemaID": "wrong.result.v1",
                },
            }
            for node_id, capability in capabilities.items():
                runtime.register_node(
                    {
                        "nodeID": node_id,
                        "capabilities": {"workloads": [capability]},
                    }
                )
                self.assertIsNone(runtime.claim_work(node_id))

    def test_exact_schema_lifecycle_claims_accepts_reduces_and_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime()
            runtime.activate_project(
                write_project(Path(directory)),
                require_terminal=False,
            )
            runtime.register_node(
                {
                    "nodeID": "curve-grid-node",
                    "capabilities": {
                        "workloads": [
                            {
                                "workloadID": curve_grid.WORKLOAD_ID,
                                **SCHEMA_TUPLE,
                            }
                        ]
                    },
                }
            )

            claimed_ranges = []
            dataset = golden_dataset()
            while True:
                work = runtime.claim_work("curve-grid-node")
                if work is None:
                    break
                self.assertEqual(curve_grid.WORKLOAD_ID, work["workloadID"])
                for key, value in SCHEMA_TUPLE.items():
                    self.assertEqual(value, work[key])
                self.assertEqual(
                    {"familyID", "gridStartIndex", "gridCount"},
                    set(work["payload"]),
                )
                self.assertNotIn("gridStartIndex", work)
                self.assertNotIn("gridCount", work)
                claimed_ranges.append(
                    (
                        work["payload"]["gridStartIndex"],
                        work["payload"]["gridCount"],
                    )
                )
                accepted, message, code = runtime.submit_result(
                    work["id"],
                    {
                        "status": "completed",
                        "resultSchemaID": curve_grid.RESULT_SCHEMA_ID,
                        "workUnitID": work["id"],
                        "nodeID": "curve-grid-node",
                        "payload": worker_payload(dataset, work["payload"]),
                    },
                )
                self.assertTrue(accepted, message)
                self.assertEqual(200, code)

            self.assertEqual(
                [(0, 5), (5, 5), (10, 5), (15, 5), (20, 5), (25, 2)],
                claimed_ranges,
            )
            status = runtime.project_status()
            self.assertEqual("COMPLETE", status["status"])
            self.assertEqual(6, status["projectCompletedWorkUnits"])
            dataset_status = status["datasets"][0]
            self.assertEqual(
                "CURVE_GRID_COMPLETE",
                dataset_status["curveGridStatus"],
            )
            self.assertEqual(
                "CURVE_GRID_COMPLETE",
                dataset_status["workloadStatus"],
            )
            self.assertTrue(dataset_status["coverageComplete"])
            self.assertEqual(27, dataset_status["totalCandidateCount"])
            self.assertEqual(27, dataset_status["completedCandidateCount"])
            self.assertEqual(13, dataset_status["bestGridIndex"])
            self.assertAlmostEqual(0.5, dataset_status["bestOffset"], places=12)
            self.assertAlmostEqual(
                2.0,
                dataset_status["bestAmplitude"],
                places=12,
            )
            self.assertLess(
                dataset_status["bestWeightedResidualSumSquares"],
                1.0e-20,
            )


if __name__ == "__main__":
    unittest.main()
