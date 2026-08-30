import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import coordinator_state
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
            "workUnitID": self.work["id"],
            "nodeID": "ios-test",
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

    def test_terminal_dataset_diagnostics_time_only_first_uncached_computation(self):
        work_id = self.work["id"]
        self.state.assigned.clear()
        self.state.completed[work_id] = {
            **self.result,
            "bestPeriodDays": 1.0 / self.result["bestFrequency"],
            "nodeID": "ios-test",
        }
        clock = (value / 10.0 for value in range(100))
        with patch(
            "coordinator_state.time.monotonic", side_effect=lambda: next(clock)
        ), patch.object(
            self.state,
            "_independent_candidates_locked",
            wraps=self.state._independent_candidates_locked,
        ) as independent, patch.object(
            self.state,
            "_fold_metrics",
            wraps=self.state._fold_metrics,
        ) as fold, patch.object(
            self.state,
            "_distributed_chunk_mode_coverage_locked",
            wraps=self.state._distributed_chunk_mode_coverage_locked,
        ) as coverage, patch(
            "coordinator_state.estimate_frequency_interval",
            wraps=coordinator_state.estimate_frequency_interval,
        ) as interval, patch("builtins.print") as output:
            first = self.state._dataset_result_diagnostics_locked("dataset-1")
            unchanged = deepcopy(first)
            cached = self.state._dataset_result_diagnostics_locked("dataset-1")

        self.assertEqual(unchanged, first)
        self.assertEqual(unchanged, cached)
        self.assertIs(first, cached)
        self.assertEqual("LOW_CONFIDENCE", first["periodStatus"])
        self.assertEqual(
            {
                "frequency": 1.2,
                "periodDays": 1.0 / 1.2,
                "power": 0.75,
            },
            first["authoritative"],
        )
        self.assertEqual(1.2, first["candidate"]["frequency"])
        self.assertEqual(0.75, first["candidate"]["power"])
        independent.assert_called_once_with("dataset-1")
        self.assertEqual(3, fold.call_count)
        coverage.assert_called_once_with("dataset-1")
        interval.assert_called_once()
        output.assert_called_once_with(
            "⏱️ Dataset diagnostics: dataset=dataset-1 "
            "independent=0.100s primaryFold=0.100s doubleFold=0.100s "
            "halfFold=0.100s coverage=0.100s frequencyInterval=0.100s "
            "total=1.300s"
        )


if __name__ == "__main__":
    unittest.main()
