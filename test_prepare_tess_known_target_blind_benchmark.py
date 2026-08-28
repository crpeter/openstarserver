import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import prepare_tess_known_target_blind_benchmark as preparer
from workflows.tess.tess_autonomy import TessInvestigationTargetSource


class _Column:
    def __init__(self, value):
        self.value = np.asarray(value)


class _LightCurve:
    sector = 7

    def __init__(self):
        self.time = _Column([100.0, 101.0, np.nan, 102.0])
        self.flux = _Column([10.0, 12.0, 99.0, 14.0])
        self.meta = {"SECTOR": 7}

    def __len__(self):
        return len(self.time.value)


class _Search:
    def __init__(self, exposures, light_curve=None):
        self.exptime = list(exposures)
        self.light_curve = light_curve

    def __len__(self):
        return len(self.exptime)

    def __getitem__(self, item):
        indices = list(range(len(self)))[item]
        if isinstance(indices, int):
            indices = [indices]
        return _Search([self.exptime[i] for i in indices], self.light_curve)

    def download(self, **kwargs):
        if kwargs != {"quality_bitmask": "default"}:
            raise AssertionError(kwargs)
        return self.light_curve


class BlindBenchmarkPreparerTests(unittest.TestCase):
    def test_prefers_spoc_120_second_product(self):
        calls = []

        def search(query, **kwargs):
            calls.append((query, kwargs))
            return _Search([120])

        with patch.object(preparer.lk, "search_lightcurve", side_effect=search):
            result, author, cadence = preparer.select_product(123, 7)
        self.assertEqual((author, cadence, len(result)), ("SPOC", 120.0, 1))
        self.assertEqual(calls, [("TIC 123", {
            "mission": "TESS", "sector": 7, "author": "SPOC", "exptime": 120
        })])

    def test_fallback_selects_shortest_tess_spoc_product(self):
        products = [_Search([]), _Search([1800, 600, 1200])]
        with patch.object(preparer.lk, "search_lightcurve", side_effect=products):
            result, author, cadence = preparer.select_product(123, 7)
        self.assertEqual((author, cadence, result.exptime), ("TESS-SPOC", 600.0, [600]))

    def test_preparation_hashes_blind_metadata_and_manifest_compatibility(self):
        reference = {"bestFrequency": 1.25, "bestPeriodDays": 0.8,
                     "bestPower": 0.5, "chunks": []}
        with tempfile.TemporaryDirectory() as root, patch.object(
            preparer, "select_product", return_value=(_Search([120], _LightCurve()), "SPOC", 120.0)
        ), patch.object(preparer, "calculate_astropy_reference", return_value=reference):
            manifest_path = preparer.prepare_benchmark(
                tic=123, primary_sector=7, blind_label="Blind Target",
                project_id="blind-example-v1", output_dir=Path(root) / "project",
            )
            manifest = json.loads(manifest_path.read_text())
            dataset_path = manifest_path.parent / "dataset.json"
            dataset = json.loads(dataset_path.read_text())
            targets = TessInvestigationTargetSource([manifest_path]).enumerate_targets()

            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].metadata["ticID"], 123)
            self.assertEqual(manifest["datasets"][0]["path"], "dataset.json")
            self.assertEqual(
                manifest["datasets"][0]["datasetSHA256"],
                hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            )
            for key, values in (("timesFloat32SHA256", dataset["times"]),
                                ("fluxFloat32SHA256", dataset["flux"])):
                expected = hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()
                self.assertEqual(dataset["hashes"][key], expected)
            serialized = json.dumps({"manifest": manifest, "dataset": dataset}).lower()
            for forbidden in ("classification", "publishedperiod", "publishedfrequency",
                              "answerkeysource", "independentsectors"):
                self.assertNotIn(forbidden, serialized)
            self.assertIs(dataset["science"]["catalogAnswerKeyUsed"], False)

    def test_existing_output_is_refused_before_product_query(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "existing"
            output.mkdir()
            with patch.object(preparer, "select_product") as select:
                with self.assertRaises(FileExistsError):
                    preparer.prepare_benchmark(
                        tic=123, primary_sector=7, blind_label="Blind Target",
                        project_id="blind-example-v1", output_dir=output,
                    )
                select.assert_not_called()

    def test_cli_validation(self):
        required = ["--primary-sector", "7", "--blind-label", "Blind Target",
                    "--project-id", "blind-example-v1", "--output-dir", "/unused"]
        with self.assertRaises(SystemExit):
            preparer.parse_args(["--tic", "0", *required])
        with self.assertRaises(SystemExit):
            preparer.parse_args(["--tic", "123", *required[:-4], "", *required[-3:]])
        with self.assertRaises(SystemExit):
            preparer.parse_args(["--tic", "123", "--primary-sector", "7",
                                  "--blind-label", "Blind", "--project-id", "wasp-18-copy",
                                  "--output-dir", "/unused"])


if __name__ == "__main__":
    unittest.main()
