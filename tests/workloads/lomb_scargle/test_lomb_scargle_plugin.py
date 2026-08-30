import json
import tempfile
import unittest
from pathlib import Path

from coordinator_runtime import CoordinatorRuntime
from openstar_workloads.discovery import discover_workloads


LOMB_WORKLOAD_ID = "openstar.lomb-scargle.v1"
TESS_ALIAS_WORKLOAD_ID = "openstar.tess-period-search.v1"


def _dataset():
    return {
        "id": "series",
        "times": [0.0, 1.0, 2.0, 3.0],
        "values": [1.0, 0.5, -0.5, 1.0],
        "frequencySearch": {
            "minimumFrequency": 0.5,
            "frequencyStep": 0.25,
            "totalFrequencies": 5,
            "frequenciesPerWorkUnit": 2,
        },
    }


def _write_legacy_project(root, dataset=None):
    dataset_path = root / "legacy-lomb-dataset.json"
    dataset_path.write_text(
        json.dumps(_dataset() if dataset is None else dataset),
        encoding="utf-8",
    )
    project_path = root / "legacy-lomb-project.json"
    project_path.write_text(
        json.dumps({
            "id": "legacy-lomb-project",
            "workloadID": LOMB_WORKLOAD_ID,
            "datasets": [{"id": "series", "path": str(dataset_path)}],
        }),
        encoding="utf-8",
    )
    return project_path


class LombScarglePluginParityTests(unittest.TestCase):
    def test_current_apple_schema_less_capability_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime()
            runtime.activate_project(
                _write_legacy_project(Path(directory)), require_terminal=False
            )
            runtime.register_node({
                "nodeID": "current-apple",
                "capabilities": {
                    "workloads": [{
                        "workloadID": LOMB_WORKLOAD_ID,
                        "executionBackends": [{"id": "metal"}],
                        "validatorID": (
                            "openstar.lomb-scargle.local-double.v1"
                        ),
                    }],
                },
            })
            self.assertIsNotNone(runtime.claim_work("current-apple"))

    def test_schema_less_result_identity_survives_lease_reassignment(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CoordinatorRuntime()
            dataset = _dataset()
            dataset["frequencySearch"].update({
                "totalFrequencies": 1,
                "frequenciesPerWorkUnit": 1,
            })
            runtime.activate_project(
                _write_legacy_project(Path(directory), dataset),
                require_terminal=False,
            )
            capabilities = {
                "workloads": [{"workloadID": LOMB_WORKLOAD_ID}]
            }
            runtime.register_node({
                "nodeID": "first",
                "capabilities": capabilities,
            })
            runtime.register_node({
                "nodeID": "second",
                "capabilities": capabilities,
            })

            first = runtime.claim_work("first")
            result = {
                "status": "completed",
                "workUnitID": first["id"],
                "nodeID": "first",
                "bestFrequency": first["payload"]["startFrequency"],
                "bestPower": 0.75,
            }
            for missing_key in ("workUnitID", "nodeID"):
                with self.subTest(missing_key=missing_key):
                    incomplete = dict(result)
                    del incomplete[missing_key]
                    accepted, message, code = runtime.submit_result(
                        first["id"], incomplete
                    )
                    self.assertFalse(accepted)
                    self.assertEqual(400, code)
                    self.assertIn("identity", message.lower())

            runtime.active_state().assigned[first["id"].lower()][
                "leaseExpiresAt"
            ] = 0.0
            second = runtime.claim_work("second")
            self.assertEqual(first["id"], second["id"])

            accepted, message, code = runtime.submit_result(
                first["id"], result
            )
            self.assertFalse(accepted)
            self.assertEqual(409, code)
            self.assertIn("node identity", message.lower())
            self.assertEqual(
                "second",
                runtime.active_state().assigned[first["id"].lower()][
                    "nodeID"
                ],
            )

            accepted, _, code = runtime.submit_result(
                second["id"], {**result, "nodeID": "second"}
            )
            self.assertTrue(accepted)
            self.assertEqual(200, code)

    def test_frequency_sharding_is_exact_and_deterministic(self):
        registry = discover_workloads()
        lomb = registry.require(LOMB_WORKLOAD_ID)
        alias = registry.require(TESS_ALIAS_WORKLOAD_ID)
        expected = [
            {
                "frequencyStartIndex": 0,
                "startFrequency": 0.5,
                "frequencyStep": 0.25,
                "frequencyCount": 2,
            },
            {
                "frequencyStartIndex": 2,
                "startFrequency": 1.0,
                "frequencyStep": 0.25,
                "frequencyCount": 2,
            },
            {
                "frequencyStartIndex": 4,
                "startFrequency": 1.5,
                "frequencyStep": 0.25,
                "frequencyCount": 1,
            },
        ]
        self.assertEqual(expected, list(lomb.build_work_payloads(_dataset())))
        self.assertEqual(expected, list(lomb.build_work_payloads(_dataset())))
        self.assertEqual(expected, list(alias.build_work_payloads(_dataset())))
        self.assertEqual(expected[0], lomb.legacy_work_unit_fields(expected[0]))
        self.assertEqual(expected[0], alias.legacy_work_unit_fields(expected[0]))

        self.assertTrue(lomb.definition.allows_legacy_schemaless_workers)
        self.assertTrue(alias.definition.allows_legacy_schemaless_workers)
        self.assertTrue(lomb.uses_legacy_coordinator_diagnostics)
        self.assertTrue(alias.uses_legacy_coordinator_diagnostics)
        self.assertTrue(lomb.uses_legacy_science_metadata_validation)
        self.assertTrue(alias.uses_legacy_science_metadata_validation)

    def test_canonical_lomb_accounting_and_alias_omission_are_preserved(self):
        registry = discover_workloads()
        lomb = registry.require(LOMB_WORKLOAD_ID)
        alias = registry.require(TESS_ALIAS_WORKLOAD_ID)
        payload = list(lomb.build_work_payloads(_dataset()))[0]

        canonical_work = {
            "workloadID": LOMB_WORKLOAD_ID,
            "payload": payload,
            **payload,
        }
        self.assertEqual(
            {
                "workloadID": LOMB_WORKLOAD_ID,
                "sampleCount": 4,
                "frequencyCount": 2,
                "sampleFrequencyEvaluations": 8,
            },
            dict(lomb.contribution_metrics(canonical_work, _dataset())),
        )

        alias_work = {
            "workloadID": TESS_ALIAS_WORKLOAD_ID,
            "payload": payload,
            **payload,
        }
        self.assertEqual(
            {"workloadID": TESS_ALIAS_WORKLOAD_ID},
            dict(alias.contribution_metrics(alias_work, _dataset())),
        )


if __name__ == "__main__":
    unittest.main()
