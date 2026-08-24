import contextlib
import io
import json
import tempfile
import unittest
from openstar_test_science_runs import IsolatedScienceRunTestCase
from pathlib import Path

from coordinator_state import CoordinatorState
from run_openstar_tess_sector_sweep import parse_args, run_tess_sector_sweep
from test_tess_sector_sweep import FakeCoordinator, FakeProvider, Prepared, product
from workflows.tess.tess_preprocessing import broad_tess_frequency_search


class TessWorkUnitSizingTests(IsolatedScienceRunTestCase):
    def test_cli_default_and_benchmark_sizes(self):
        base = ["--sector", "7", "--coordinator-url", "unused", "--state-dir", "state"]
        self.assertIsNone(parse_args(base).frequencies_per_work_unit)
        for size in (8192, 16384, 32768):
            with self.subTest(size=size):
                self.assertEqual(
                    size,
                    parse_args(base + ["--frequencies-per-work-unit", str(size)]).frequencies_per_work_unit,
                )

    def test_nonpositive_cli_sizes_are_rejected(self):
        base = ["--sector", "7", "--coordinator-url", "unused", "--state-dir", "state"]
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(SystemExit):
                parse_args(base + ["--frequencies-per-work-unit", str(size)])

    def test_override_changes_only_chunk_size_and_is_persisted_and_reported(self):
        from workflows.tess import tess_sector_scan

        baseline = broad_tess_frequency_search()
        original_preprocessing = tess_sector_scan.read_and_prepare_tess_light_curve
        tess_sector_scan.read_and_prepare_tess_light_curve = lambda path: Prepared()
        try:
            for requested in (None, 8192, 16384, 32768):
                with self.subTest(requested=requested), tempfile.TemporaryDirectory() as tmp:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = run_tess_sector_sweep(
                            7, "unused", tmp, provider=FakeProvider([product(1)]),
                            coordinator=FakeCoordinator(),
                            frequencies_per_work_unit=requested,
                            allow_temporary_state=True)
                    self.assertEqual(0, code)
                    effective = requested or 4096
                    self.assertIn(f"frequencies-per-work-unit={effective}", output.getvalue())
                    dataset_path = next(Path(tmp).glob(
                        "investigations/*/artifacts/scan-input/dataset.json"
                    ))
                    search = json.loads(dataset_path.read_text())["frequencySearch"]
                    self.assertEqual(effective, search["frequenciesPerWorkUnit"])
                    for key in ("minimumFrequency", "maximumFrequency", "frequencyStep", "totalFrequencies"):
                        self.assertEqual(baseline[key], search[key])
        finally:
            tess_sector_scan.read_and_prepare_tess_light_curve = original_preprocessing

    def test_coordinator_chunks_cover_frequency_indices_once_with_partial_tail(self):
        total, chunk_size = 23, 10
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps({
                "id": "dataset", "times": [0.0, 1.0, 2.0], "values": [0.0, 1.0, 0.0],
                "frequencySearch": {"minimumFrequency": 0.1, "maximumFrequency": 2.4,
                    "frequencyStep": 0.1, "totalFrequencies": total,
                    "frequenciesPerWorkUnit": chunk_size},
            }))
            project_path = root / "project.json"
            project_path.write_text(json.dumps({"id": "project", "workloadID": "openstar.lomb-scargle.v1",
                "datasets": [{"id": "dataset", "path": str(dataset_path)}]}))

            state = CoordinatorState(project_path)
            chunks = sorted(
                (unit["payload"] for unit in state.work_units.values()),
                key=lambda payload: payload["frequencyStartIndex"],
            )
            self.assertEqual([0, 10, 20], [chunk["frequencyStartIndex"] for chunk in chunks])
            self.assertEqual([10, 10, 3], [chunk["frequencyCount"] for chunk in chunks])
            covered = [
                index for chunk in chunks
                for index in range(chunk["frequencyStartIndex"],
                                   chunk["frequencyStartIndex"] + chunk["frequencyCount"])
            ]
            self.assertEqual(list(range(total)), covered)
            self.assertEqual(len(covered), len(set(covered)))
            self.assertEqual([0.1, 1.1, 2.1], [chunk["startFrequency"] for chunk in chunks])
            self.assertTrue(all(chunk["frequencyStep"] == 0.1 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
