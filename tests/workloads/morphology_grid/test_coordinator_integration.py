import json
import math
import tempfile
import unittest
from pathlib import Path

from coordinator_runtime import CoordinatorRuntime
from openstar_workloads.plugins import morphology_grid
from tests.workloads.morphology_grid.test_morphology_grid_plugin import (
    explicit_axis,
    linear_axis,
    positive_dataset,
    worker_payload,
)


SCHEMA_TUPLE = {
    "datasetSchemaID": morphology_grid.DATASET_SCHEMA_ID,
    "payloadSchemaID": morphology_grid.PAYLOAD_SCHEMA_ID,
    "resultSchemaID": morphology_grid.RESULT_SCHEMA_ID,
}


def lifecycle_dataset():
    dataset = positive_dataset(series_count=1)
    dataset["morphologyGrid"] = {
        "centerAxis": linear_axis(-0.5, 0.5, 2),
        "logScaleAxis": linear_axis(0.0, 1.0, 1),
        "logShapeAxis": explicit_axis(0.0),
    }
    dataset["candidatesPerWorkUnit"] = 1
    return dataset


def write_project(root):
    dataset_path = root / "morphology-grid-dataset.json"
    dataset_path.write_text(
        json.dumps(lifecycle_dataset()),
        encoding="utf-8",
    )
    project_path = root / "morphology-grid-project.json"
    project_path.write_text(
        json.dumps(
            {
                "id": "morphology-grid-project",
                "workloadID": morphology_grid.WORKLOAD_ID,
                **SCHEMA_TUPLE,
                "datasets": [
                    {
                        "id": "generic-positive-grid",
                        "path": str(dataset_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return project_path


class MorphologyGridCoordinatorIntegrationTests(unittest.TestCase):
    def test_incomplete_schema_tuple_cannot_claim_strict_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            runtime = CoordinatorRuntime()
            runtime.activate_project(
                write_project(root),
                require_terminal=False,
            )
            capabilities = {
                "schema-less": {
                    "workloadID": morphology_grid.WORKLOAD_ID,
                },
                "partial": {
                    "workloadID": morphology_grid.WORKLOAD_ID,
                    "datasetSchemaID": morphology_grid.DATASET_SCHEMA_ID,
                    "payloadSchemaID": morphology_grid.PAYLOAD_SCHEMA_ID,
                },
                "wrong": {
                    "workloadID": morphology_grid.WORKLOAD_ID,
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

    def test_tiny_exact_schema_lifecycle_claims_submits_reduces_and_completes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            runtime = CoordinatorRuntime()
            runtime.activate_project(
                write_project(root),
                require_terminal=False,
            )
            runtime.register_node(
                {
                    "nodeID": "morphology-grid-node",
                    "capabilities": {
                        "workloads": [
                            {
                                "workloadID": morphology_grid.WORKLOAD_ID,
                                **SCHEMA_TUPLE,
                            }
                        ]
                    },
                }
            )

            dataset = lifecycle_dataset()
            claimed_ranges = []
            while True:
                work = runtime.claim_work("morphology-grid-node")
                if work is None:
                    break
                self.assertEqual(
                    morphology_grid.WORKLOAD_ID,
                    work["workloadID"],
                )
                for key, value in SCHEMA_TUPLE.items():
                    self.assertEqual(value, work[key])
                self.assertEqual(
                    {
                        "morphologyFamilyID",
                        "modelClassID",
                        "gridStartIndex",
                        "gridCount",
                    },
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
                        "resultSchemaID": morphology_grid.RESULT_SCHEMA_ID,
                        "workUnitID": work["id"],
                        "nodeID": "morphology-grid-node",
                        "payload": worker_payload(dataset, work["payload"]),
                    },
                )
                self.assertTrue(accepted, message)
                self.assertEqual(200, code)

            self.assertEqual(
                [(0, 1), (1, 1)],
                claimed_ranges,
            )
            status = runtime.project_status()
            self.assertEqual("COMPLETE", status["status"])
            self.assertEqual(2, status["projectCompletedWorkUnits"])
            dataset_status = status["datasets"][0]
            self.assertEqual(
                "MORPHOLOGY_GRID_COMPLETE",
                dataset_status["morphologyGridStatus"],
            )
            self.assertEqual(
                "MORPHOLOGY_GRID_COMPLETE",
                dataset_status["workloadStatus"],
            )
            self.assertTrue(dataset_status["coverageComplete"])
            self.assertEqual(2, dataset_status["totalCandidateCount"])
            self.assertEqual(2, dataset_status["completedCandidateCount"])
            self.assertEqual(0, dataset_status["totalInvalidCandidateCount"])
            self.assertIsInstance(dataset_status["bestGridIndex"], int)
            self.assertIsInstance(dataset_status["bestParameters"], dict)
            self.assertEqual(1, len(dataset_status["bestSeriesFits"]))
            self.assertEqual(4, dataset_status["bestPositiveWeightSampleCount"])
            self.assertEqual(5, dataset_status["bestNominalParameterCount"])
            self.assertGreaterEqual(
                dataset_status["bestWeightedResidualSumSquares"],
                0.0,
            )
            self.assertTrue(
                math.isfinite(
                    dataset_status["bestBayesianInformationCriterion"]
                )
            )
            self.assertIsNone(
                dataset_status["bestCorrectedAkaikeInformationCriterion"]
            )
            self.assertFalse(
                dataset_status[
                    "bestCorrectedAkaikeInformationCriterionDefined"
                ]
            )


if __name__ == "__main__":
    unittest.main()
