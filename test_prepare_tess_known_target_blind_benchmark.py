import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

import prepare_tess_known_target_blind_benchmark as preparer
from workflows.tess.tess_autonomy import TessInvestigationTargetSource
from workflows.tess.tess_followup import build_single_target_primary


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
    reference = {"bestFrequency": 1.25, "bestPeriodDays": 0.8,
                 "bestPower": 0.5, "chunks": []}

    def _prepare(self, output, *, label="Blind Target", overwrite=False):
        with patch.object(
            preparer, "select_product",
            return_value=(_Search([120], _LightCurve()), "SPOC", 120.0),
        ), patch.object(
            preparer, "calculate_astropy_reference", return_value=self.reference
        ):
            return preparer.prepare_benchmark(
                tic=123, primary_sector=7, blind_label=label,
                project_id="blind-example-v1", output_dir=output, overwrite=overwrite,
            )

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

    def test_fallback_refuses_all_non_finite_cadences(self):
        products = [_Search([]), _Search([np.nan, np.inf])]
        with patch.object(preparer.lk, "search_lightcurve", side_effect=products):
            with self.assertRaisesRegex(RuntimeError, "no finite cadence"):
                preparer.select_product(123, 7)

    def test_preparation_hashes_blind_metadata_and_manifest_compatibility(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest_path = self._prepare(root / "project")
            manifest = json.loads(manifest_path.read_text())
            dataset_path = manifest_path.parent / "dataset.json"
            dataset = json.loads(dataset_path.read_text())
            targets = TessInvestigationTargetSource([manifest_path]).enumerate_targets()

            # A same-named CWD file must not shadow the absolute frozen path.
            unrelated = root / "cwd"
            unrelated.mkdir()
            (unrelated / "dataset.json").write_text('{"source":{"ticID":999}}')
            previous_cwd = Path.cwd()
            os.chdir(unrelated)
            try:
                prepared = build_single_target_primary(
                    source_project_path=manifest_path,
                    output_dir=root / "artifacts",
                    investigation_id="compatibility-test",
                    dataset_id="tess-tic-123-sector-7",
                    tic_id=123,
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].metadata["ticID"], 123)
            self.assertEqual(Path(manifest["datasets"][0]["path"]), dataset_path)
            self.assertEqual(Path(prepared["datasetPath"]), dataset_path)
            self.assertEqual((prepared["ticID"], prepared["sector"]), (123, 7))
            frozen = json.loads(Path(prepared["projectPath"]).read_text())
            self.assertEqual(frozen["datasets"][0], manifest["datasets"][0])
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
            self.assertEqual(manifest["preparer"], preparer._owned_manifest("blind-example-v1"))

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

    def test_overwrite_refuses_every_unsafe_existing_shape(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cases = []

            not_directory = root / "not-directory"
            not_directory.write_text("x")
            cases.append(("non-symlink output directory", not_directory))

            malformed = root / "malformed"
            malformed.mkdir()
            (malformed / "project.json").write_text("{")
            (malformed / "dataset.json").write_text("{}")
            cases.append(("malformed project.json", malformed))

            missing = root / "missing"
            missing.mkdir()
            (missing / "project.json").write_text("{}")
            cases.append(("missing or unexpected", missing))

            unexpected = root / "unexpected"
            self._prepare(unexpected)
            (unexpected / "notes.txt").write_text("caller data")
            cases.append(("missing or unexpected", unexpected))

            unexpected_directory = root / "unexpected-directory"
            self._prepare(unexpected_directory)
            (unexpected_directory / "caller-data").mkdir()
            cases.append(("missing or unexpected", unexpected_directory))

            marker = root / "marker"
            self._prepare(marker)
            value = json.loads((marker / "project.json").read_text())
            del value["preparer"]
            (marker / "project.json").write_text(json.dumps(value))
            cases.append(("ownership marker", marker))

            schema = root / "schema"
            self._prepare(schema)
            value = json.loads((schema / "project.json").read_text())
            value["preparer"]["schemaVersion"] = 999
            (schema / "project.json").write_text(json.dumps(value))
            cases.append(("ownership marker", schema))

            mismatch = root / "mismatch"
            self._prepare(mismatch)
            cases.append(("project ID mismatch", mismatch))

            symlink_file = root / "symlink-file"
            self._prepare(symlink_file)
            (symlink_file / "dataset.json").unlink()
            (symlink_file / "dataset.json").symlink_to(root / "not-directory")
            cases.append(("symlinks are refused", symlink_file))

            real = root / "real-directory"
            self._prepare(real)
            linked = root / "linked-directory"
            linked.symlink_to(real, target_is_directory=True)
            cases.append(("non-symlink output directory", linked))

            for message, path in cases:
                with self.subTest(path=path.name), patch.object(preparer, "select_product") as select:
                    requested_id = "different-id" if path == mismatch else "blind-example-v1"
                    with self.assertRaisesRegex(RuntimeError, message):
                        preparer.prepare_benchmark(
                            tic=123, primary_sector=7, blind_label="Blind Target",
                            project_id=requested_id, output_dir=path, overwrite=True,
                        )
                    select.assert_not_called()

    def test_successful_safe_replacement_and_failed_build_preserves_existing(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "project"
            manifest_path = self._prepare(output, label="First")
            before = {name: (output / name).read_bytes() for name in preparer.OWNED_FILES}

            with patch.object(preparer, "select_product", side_effect=RuntimeError("download failed")):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    preparer.prepare_benchmark(
                        tic=123, primary_sector=7, blind_label="Broken",
                        project_id="blind-example-v1", output_dir=output, overwrite=True,
                    )
            self.assertEqual(
                {name: (output / name).read_bytes() for name in preparer.OWNED_FILES}, before
            )

            replaced = self._prepare(output, label="Second", overwrite=True)
            self.assertEqual(replaced, manifest_path)
            self.assertEqual(json.loads(replaced.read_text())["name"], "Second")
            self.assertEqual({entry.name for entry in output.iterdir()}, preparer.OWNED_FILES)
            self.assertFalse(output.with_name(f".{output.name}.replace-backup").exists())
            self.assertFalse(any(output.parent.glob(f".{output.name}.prepare-*")))

    def test_cli_validation(self):
        def failure(project_id="blind-example-v1", label="Blind Target", tic="123"):
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                preparer.parse_args([
                    "--tic", tic, "--primary-sector", "7", "--blind-label", label,
                    "--project-id", project_id, "--output-dir", "/unused",
                ])
            return stderr.getvalue()

        self.assertIn("--tic must be positive", failure(tic="0"))
        self.assertIn("--blind-label must not be empty", failure(label=""))
        for unsafe in ("Project ID", "project/id", "../project", "", "project id", "PROJECT-ID"):
            with self.subTest(project_id=unsafe):
                message = failure(project_id=unsafe)
                self.assertTrue("canonical" in message or "invalid --project-id" in message)
        self.assertEqual(preparer.parse_args([
            "--tic", "123", "--primary-sector", "7", "--blind-label", "Blind",
            "--project-id", "project-id", "--output-dir", "/unused",
        ]).project_id, "project-id")


if __name__ == "__main__":
    unittest.main()
