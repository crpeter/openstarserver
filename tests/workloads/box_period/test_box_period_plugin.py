import json
import tempfile
import unittest
from pathlib import Path

from coordinator_runtime import CoordinatorRuntime
from openstar_workloads.discovery import discover_workloads


BOX_WORKLOAD_ID = "openstar.box-period-search.v1"


def _box_dataset():
    return {
        "id": "series",
        "coordinates": [index * 0.25 for index in range(80)],
        "values": [-2.0 if index % 8 == 0 else 0.25 for index in range(80)],
        "boxPeriodSearch": {
            "phaseBinCount": 20,
            "durationFractions": [0.1, 0.15],
            "minimumInBoxSamples": 4,
            "minimumOutOfBoxSamples": 20,
            "frequencyWindows": [
                {
                    "familyRank": 13,
                    "familyID": "family-13",
                    "centerFrequency": 0.5,
                    "startFrequency": 0.4,
                    "frequencyStep": 0.05,
                    "frequencyCount": 5,
                    "frequencyStartIndex": 20,
                },
                {
                    "familyRank": 14,
                    "familyID": "family-14",
                    "centerFrequency": 0.8,
                    "startFrequency": 0.7,
                    "frequencyStep": 0.05,
                    "frequencyCount": 5,
                    "frequencyStartIndex": 25,
                },
            ],
        },
    }


def _expected_payloads():
    shared = {
        "frequencyStep": 0.05,
        "frequencyCount": 5,
        "phaseBinCount": 20,
        "durationFractions": [0.1, 0.15],
        "minimumInBoxSamples": 4,
        "minimumOutOfBoxSamples": 20,
    }
    return [
        {
            "startFrequency": 0.4,
            "frequencyStartIndex": 20,
            "windowIndex": 0,
            "familyRank": 13,
            "familyID": "family-13",
            "centerFrequency": 0.5,
            **shared,
        },
        {
            "startFrequency": 0.7,
            "frequencyStartIndex": 25,
            "windowIndex": 1,
            "familyRank": 14,
            "familyID": "family-14",
            "centerFrequency": 0.8,
            **shared,
        },
    ]


def _valid_result(payload, *, score=None):
    relative_index = 2
    frequency_index = payload["frequencyStartIndex"] + relative_index
    duration_index = 1
    duration_bins = int(
        payload["durationFractions"][duration_index]
        * payload["phaseBinCount"]
        + 0.5
    )
    return {
        "bestFrequency": (
            payload["startFrequency"]
            + relative_index * payload["frequencyStep"]
        ),
        "bestScore": (
            float(payload.get("familyRank", 1)) if score is None else score
        ),
        "bestPhase": 0.0,
        "bestDurationFraction": duration_bins / payload["phaseBinCount"],
        "bestFrequencyIndex": frequency_index,
        "bestDurationIndex": duration_index,
        "bestPhaseBin": 0,
        "inBoxSamples": 20,
        "outOfBoxSamples": 60,
    }


def _write_project(root):
    dataset_path = root / "box-dataset.json"
    dataset_path.write_text(json.dumps(_box_dataset()), encoding="utf-8")
    project_path = root / "box-project.json"
    project_path.write_text(
        json.dumps({
            "id": "box-project",
            "workloadID": BOX_WORKLOAD_ID,
            "datasets": [{"id": "series", "path": str(dataset_path)}],
        }),
        encoding="utf-8",
    )
    return project_path


class BoxPeriodPluginParityTests(unittest.TestCase):
    def test_frequency_window_sharding_is_exact_and_deterministic(self):
        plugin = discover_workloads().require(BOX_WORKLOAD_ID)
        expected = _expected_payloads()
        self.assertEqual(expected, list(plugin.build_work_payloads(_box_dataset())))
        self.assertEqual(expected, list(plugin.build_work_payloads(_box_dataset())))
        self.assertEqual({}, plugin.legacy_work_unit_fields(expected[0]))
        self.assertTrue(plugin.definition.allows_legacy_schemaless_workers)
        self.assertFalse(plugin.uses_legacy_coordinator_diagnostics)
        self.assertTrue(plugin.uses_legacy_science_metadata_validation)
        for payload in expected:
            self.assertNotIn("periodSearch", payload)
            self.assertNotIn("startPeriodDays", payload)
            self.assertNotIn("periodCount", payload)

    def test_complete_result_validation_contract_is_preserved(self):
        plugin = discover_workloads().require(BOX_WORKLOAD_ID)
        dataset = _box_dataset()
        payload = _expected_payloads()[0]
        work = {
            "id": "work-0",
            "workloadID": BOX_WORKLOAD_ID,
            "datasetID": "series",
            "payload": payload,
        }

        def validate(result_payload):
            result = plugin.canonicalize_result(
                work,
                {"status": "completed", "payload": result_payload},
            )
            return plugin.validate_result(work, result, dataset)

        valid = _valid_result(payload)
        self.assertTrue(validate(valid).accepted)

        invalid = []
        missing = dict(valid)
        del missing["bestScore"]
        invalid.append(missing)
        invalid.append({**valid, "bestFrequencyIndex": 22.5})
        invalid.append({**valid, "bestFrequency": 0.55})
        invalid.append({**valid, "bestDurationFraction": 0.1})
        invalid.append({**valid, "bestPhase": 0.1})
        invalid.append({**valid, "inBoxSamples": 3, "outOfBoxSamples": 77})
        invalid.append({**valid, "inBoxSamples": 70, "outOfBoxSamples": 10})
        invalid.append({**valid, "inBoxSamples": 20, "outOfBoxSamples": 59})
        invalid.append({**valid, "bestScore": float("inf")})
        for index, result in enumerate(invalid):
            with self.subTest(index=index):
                self.assertFalse(validate(result).accepted)

    def test_runtime_reduction_status_and_accounting_match_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = CoordinatorRuntime(
                contribution_db=root / "contributions.sqlite3"
            )
            self.addCleanup(runtime.close)
            runtime.activate_project(_write_project(root), require_terminal=False)
            runtime.register_node({
                "nodeID": "box-node",
                "capabilities": {
                    "platform": "macOS",
                    "workloads": [{"workloadID": BOX_WORKLOAD_ID}],
                },
            })

            claimed = []
            while True:
                work = runtime.claim_work("box-node")
                if work is None:
                    break
                claimed.append(work)
                payload = work["payload"]
                accepted, _, code = runtime.submit_result(work["id"], {
                    "status": "completed",
                    "duration": 2.0,
                    "workUnitID": work["id"],
                    "nodeID": "box-node",
                    "payload": _valid_result(payload),
                })
                self.assertTrue(accepted)
                self.assertEqual(200, code)

            self.assertEqual(2, len(claimed))
            self.assertEqual(_expected_payloads(), [work["payload"] for work in claimed])
            self.assertTrue(all("frequencySearch" not in work for work in claimed))

            status = runtime.project_status()
            self.assertEqual("COMPLETE", status["status"])
            self.assertEqual("BOX_SEARCH_COMPLETE", status["periodStatus"])
            self.assertTrue(status["coverageComplete"])
            self.assertIsNone(status["bestPeriodDays"])

            dataset_status = status["datasets"][0]
            legacy_status = {
                "periodStatus": "BOX_SEARCH_COMPLETE",
                "periodConfidence": None,
                "coverageComplete": True,
                "bestFrequency": None,
                "bestPeriodDays": None,
                "bestPower": None,
                "candidateFrequency": None,
                "candidatePeriodDays": None,
                "candidatePower": None,
                "candidateFoldCoherence": None,
                "candidatePeakProminenceRatio": None,
                "candidateFrequencyConfidenceInterval": None,
                "candidateFrequencyUncertaintyDiagnostics": None,
                "preferredPhysicalPeriodDays": None,
                "preferredPhysicalPeriodRelation": None,
                "harmonicCandidates": [],
                "independentCandidates": [],
            }
            self.assertEqual(
                legacy_status,
                {key: dataset_status[key] for key in legacy_status},
            )

            candidates = dataset_status["boxCandidates"]
            self.assertEqual([0, 1], [item["windowIndex"] for item in candidates])
            self.assertEqual([13, 14], [item["familyRank"] for item in candidates])
            self.assertEqual([22, 27], [item["frequencyIndex"] for item in candidates])
            for expected, candidate in zip((0.5, 0.8), candidates):
                self.assertAlmostEqual(expected, candidate["frequency"])
            self.assertEqual(candidates, dataset_status["payload"]["boxCandidates"])

            plugin = discover_workloads().require(BOX_WORKLOAD_ID)
            self.assertEqual(
                {
                    "workloadID": BOX_WORKLOAD_ID,
                    "sampleCount": 80,
                    "frequencyCount": 5,
                    "durationCount": 2,
                    "phaseBinCount": 20,
                    "sampleFrequencyEvaluations": 400,
                },
                dict(plugin.contribution_metrics(claimed[0], _box_dataset())),
            )
            contributions = runtime.contribution_summary()["currentSession"]
            self.assertEqual(2, contributions["totalAcceptedWorkUnits"])
            self.assertEqual(
                800, contributions["totalSampleFrequencyEvaluations"]
            )


if __name__ == "__main__":
    unittest.main()
