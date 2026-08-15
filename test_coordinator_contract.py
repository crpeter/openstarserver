import json
import tempfile
import unittest
from pathlib import Path

from coordinator_state import CoordinatorState


class CoordinatorResultSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        dataset_path = root / "dataset.json"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "dataset-1",
                    "times": [0.0, 1.0, 2.0],
                    "values": [1.0, 0.0, 1.0],
                    "frequencySearch": {
                        "minimumFrequency": 1.0,
                        "frequencyStep": 0.1,
                        "totalFrequencies": 4,
                        "frequenciesPerWorkUnit": 4,
                    },
                }
            ),
            encoding="utf-8",
        )
        project_path = root / "project.json"
        project_path.write_text(
            json.dumps(
                {
                    "id": "project-1",
                    "workloadID": "openstar.lomb-scargle.v1",
                    "datasets": [{"id": "dataset-1", "path": str(dataset_path)}],
                }
            ),
            encoding="utf-8",
        )
        self.state = CoordinatorState(project_path)
        self.state.register_node({"nodeID": "ios-test", "capabilities": {}})
        self.work = self.state.claim_work("ios-test")
        self.result = {
            "status": "completed",
            "bestFrequency": 1.2,
            "bestPower": 0.75,
            "bestFrequencyIndex": 2,
        }

    def test_identical_retry_is_accepted_without_double_counting(self):
        first = self.state.submit_result(self.work["id"], self.result)
        retry = self.state.submit_result(self.work["id"], self.result)

        self.assertTrue(first[0])
        self.assertEqual(200, first[2])
        self.assertEqual((True, "Identical result already accepted.", 200), retry)
        self.assertEqual(1, len(self.state.completed))
        self.assertEqual(1, self.state.project_status()["completedWorkUnits"])

    def test_conflicting_retry_is_rejected_without_replacing_result(self):
        self.assertTrue(self.state.submit_result(self.work["id"], self.result)[0])
        conflicting = dict(self.result, bestPower=0.5)

        retry = self.state.submit_result(self.work["id"], conflicting)

        self.assertEqual(409, retry[2])
        self.assertFalse(retry[0])
        self.assertEqual(0.75, self.state.completed[self.work["id"]]["bestPower"])
        self.assertEqual(1, len(self.state.completed))


if __name__ == "__main__":
    unittest.main()
