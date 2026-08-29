import json
import tempfile
import unittest
from pathlib import Path

from coordinator_state import BOX_PERIOD_SEARCH_V1, CoordinatorState


class BoxPeriodSearchCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        dataset_path = root / "dataset.json"
        dataset_path.write_text(json.dumps({
            "id": "series",
            "coordinates": [index * 0.25 for index in range(80)],
            "values": [-2.0 if index % 8 == 0 else 0.25 for index in range(80)],
            "boxPeriodSearch": {
                "phaseBinCount": 20,
                "durationFractions": [0.1, 0.15],
                "minimumInBoxSamples": 4,
                "minimumOutOfBoxSamples": 20,
                "frequencyWindows": [
                    {"familyRank": 13, "familyID": "family-13", "centerFrequency": 0.5,
                     "startFrequency": 0.4, "frequencyStep": 0.05,
                     "frequencyCount": 5, "frequencyStartIndex": 20},
                    {"familyRank": 14, "familyID": "family-14", "centerFrequency": 0.8,
                     "startFrequency": 0.7, "frequencyStep": 0.05,
                     "frequencyCount": 5, "frequencyStartIndex": 25},
                ],
            },
        }), encoding="utf-8")
        project_path = root / "project.json"
        project_path.write_text(json.dumps({
            "id": "box-project",
            "workloadID": BOX_PERIOD_SEARCH_V1,
            "datasets": [{"id": "series", "path": str(dataset_path)}],
        }), encoding="utf-8")
        self.state = CoordinatorState(project_path)
        self.state.validate_startup()

    def test_builds_one_generic_work_unit_per_frequency_window(self):
        self.assertEqual(2, len(self.state.work_units))
        payloads = [unit["payload"] for unit in self.state.work_units.values()]
        self.assertEqual([13, 14], [payload["familyRank"] for payload in payloads])
        self.assertTrue(all(unit["workloadID"] == BOX_PERIOD_SEARCH_V1
                            for unit in self.state.work_units.values()))
        self.assertTrue(all("frequencySearch" not in unit for unit in self.state.work_units.values()))

    def test_capability_filter_and_payload_result_round_trip(self):
        self.state.register_node({"nodeID": "lomb-only", "capabilities": {
            "workloads": [{"workloadID": "openstar.lomb-scargle.v1"}]}})
        self.assertIsNone(self.state.claim_work("lomb-only"))
        self.state.register_node({"nodeID": "box-node", "capabilities": {
            "workloads": [{"workloadID": BOX_PERIOD_SEARCH_V1}]}})
        work = self.state.claim_work("box-node")
        result_payload = {
            "bestFrequency": 0.5, "bestScore": 8.714213,
            "bestPhase": 0.0, "bestDurationFraction": 0.15,
            "bestFrequencyIndex": 22, "bestDurationIndex": 1,
            "bestPhaseBin": 0, "inBoxSamples": 20, "outOfBoxSamples": 60,
        }
        accepted = self.state.submit_result(work["id"], {
            "status": "completed", "payload": result_payload})
        self.assertTrue(accepted[0])
        stored = self.state.completed[work["id"]]
        self.assertEqual(8.714213, stored["bestScore"])
        self.assertNotIn("bestPeriodDays", stored)

    def test_rejects_a_frequency_that_disagrees_with_its_grid_index(self):
        self.state.register_node({"nodeID": "box-node", "capabilities": {}})
        work = self.state.claim_work("box-node")
        accepted = self.state.submit_result(work["id"], {
            "status": "completed", "payload": {
                "bestFrequency": 0.6, "bestScore": 1.0,
                "bestPhase": 0.0, "bestDurationFraction": 0.15,
                "bestFrequencyIndex": 22, "bestDurationIndex": 1,
                "bestPhaseBin": 0, "inBoxSamples": 20, "outOfBoxSamples": 60,
            }})
        self.assertFalse(accepted[0])
        self.assertIn("grid index", accepted[1])

    def test_rejects_fractional_indexes_and_incomplete_sample_partition(self):
        self.state.register_node({"nodeID": "box-node", "capabilities": {}})
        work = self.state.claim_work("box-node")
        base = {
            "bestFrequency": 0.5, "bestScore": 8.7,
            "bestPhase": 0.0, "bestDurationFraction": 0.15,
            "bestFrequencyIndex": 22, "bestDurationIndex": 1,
            "bestPhaseBin": 0, "inBoxSamples": 20, "outOfBoxSamples": 60,
        }
        fractional = self.state.submit_result(work["id"], {
            "status": "completed", "payload": {
                **base, "bestFrequencyIndex": 22.5,
            }})
        self.assertFalse(fractional[0])
        self.state.retry_after.clear()
        self.state.execution_avoid_until.clear()
        work = self.state.claim_work("box-node")
        payload = work["payload"]
        base = {
            **base,
            "bestFrequency": payload["startFrequency"],
            "bestFrequencyIndex": payload["frequencyStartIndex"],
        }
        incomplete = self.state.submit_result(work["id"], {
            "status": "completed", "payload": {
                **base, "inBoxSamples": 19,
            }})
        self.assertFalse(incomplete[0])
        self.assertIn("partition", incomplete[1])

    def test_terminal_status_exposes_all_window_winners_without_period_claim(self):
        self.state.register_node({"nodeID": "box-node", "capabilities": {}})
        while True:
            work = self.state.claim_work("box-node")
            if work is None:
                break
            payload = work["payload"]
            frequency_index = payload["frequencyStartIndex"]
            duration_bins = int(payload["durationFractions"][0] * payload["phaseBinCount"] + 0.5)
            self.assertTrue(self.state.submit_result(work["id"], {
                "status": "completed", "payload": {
                    "bestFrequency": payload["startFrequency"],
                    "bestScore": float(payload["familyRank"]),
                    "bestPhase": 0.0,
                    "bestDurationFraction": duration_bins / payload["phaseBinCount"],
                    "bestFrequencyIndex": frequency_index,
                    "bestDurationIndex": 0, "bestPhaseBin": 0,
                    "inBoxSamples": 10, "outOfBoxSamples": 70,
                }})[0])
        status = self.state.dataset_status("series")
        self.assertEqual("BOX_SEARCH_COMPLETE", status["periodStatus"])
        self.assertTrue(status["coverageComplete"])
        self.assertIsNone(status["bestPeriodDays"])
        self.assertEqual([13, 14], [item["familyRank"] for item in status["boxCandidates"]])


if __name__ == "__main__":
    unittest.main()
