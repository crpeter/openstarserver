import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_workloads.plugins.curve_grid import (
    DATASET_SCHEMA_ID,
    FAMILY_ID,
    MAX_SAFE_INTEGER,
    PAYLOAD_SCHEMA_ID,
    PLUGIN as CURVE_GRID_PLUGIN,
    RESULT_SCHEMA_ID,
    WORKLOAD_ID,
)
from workflows.microlensing.coarse_grid import (
    BUILD_MANIFEST_RELATIVE_PATH,
    CANDIDATES_PER_WORK_UNIT,
    CENTER_AXIS,
    COARSE_GRID_CONTRACT,
    COARSE_GRID_CONTRACT_ID,
    COARSE_GRID_CONTRACT_SHA256,
    CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_AXIS,
    LOG_SHAPE_AXIS,
    PROJECT_RELATIVE_PATH,
    TOTAL_CANDIDATE_COUNT,
    CoarseGridBuildError,
    _safe_product,
    build_coarse_grid_project,
)
from workflows.microlensing.prepare import (
    BLIND_MANIFEST_SCHEMA_ID,
    PREPARATION_CONTRACT_ID,
    PREPARATION_CONTRACT_SHA256,
    SERIES_SCHEMA_ID,
)


BLIND_TARGET_ID = "openstar.generic-recovery-a.v1"
PROJECT_ID = "openstar.generic-recovery-a.coarse-grid.v1"
FORBIDDEN_OUTPUT_TOKENS = (
    "0302608",
    "OGLE",
    "724L",
    "Hirao",
    "UID_",
    "exoplanetarchive.ipac.caltech.edu",
)


def stable_json_bytes(value, *, allow_nan=False):
    return (
        json.dumps(
            value,
            allow_nan=allow_nan,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_bytes(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value, *, allow_nan=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value, allow_nan=allow_nan))


def series_payload(series_id, sample_count, *, start=0.25):
    coordinates = [start + index * 0.5 for index in range(sample_count)]
    values = [1.0 + index * 0.01 for index in range(sample_count)]
    inverse_variances = [100.0 + index for index in range(sample_count)]
    return {
        "blindTargetID": BLIND_TARGET_ID,
        "coordinates": coordinates,
        "inverseVariances": inverse_variances,
        "seriesID": series_id,
        "seriesSchemaID": SERIES_SCHEMA_ID,
        "values": values,
    }


def write_prepared_root(
    root,
    *,
    specifications=(("series-001", 4), ("series-002", 7), ("series-003", 5)),
    ordered_ids=None,
):
    blind = root / "blind"
    series_root = blind / "series"
    series_root.mkdir(parents=True)
    records = []
    default_order = []
    for index, (series_id, sample_count) in enumerate(specifications, 1):
        payload = series_payload(series_id, sample_count, start=index * 0.25)
        relative_path = f"series/input-{index:03d}.json"
        payload_bytes = stable_json_bytes(payload)
        (blind / relative_path).write_bytes(payload_bytes)
        records.append(
            {
                "coordinateRange": {
                    "maximum": max(payload["coordinates"]),
                    "minimum": min(payload["coordinates"]),
                },
                "observableRepresentation": "relative-linear-flux",
                "sampleCount": sample_count,
                "seriesFile": relative_path,
                "seriesID": series_id,
                "sha256": sha256_bytes(payload_bytes),
            }
        )
        default_order.append(series_id)
    selected_order = list(ordered_ids or default_order)
    manifest = {
        "benchmarkKind": "known-event-recovery",
        "blindTargetID": BLIND_TARGET_ID,
        "orderedSeriesIDs": selected_order,
        "preparationContractID": PREPARATION_CONTRACT_ID,
        "preparationContractSHA256": PREPARATION_CONTRACT_SHA256,
        "preparationManifestSchemaID": BLIND_MANIFEST_SCHEMA_ID,
        "series": records,
        "totalSampleCount": sum(record["sampleCount"] for record in records),
        "totalSeriesCount": len(records),
    }
    write_json(blind / "preparation-manifest.json", manifest)
    return manifest


def read_manifest(root):
    return read_json(root / "blind" / "preparation-manifest.json")


def write_manifest(root, manifest):
    write_json(root / "blind" / "preparation-manifest.json", manifest)


def rewrite_series(
    root,
    manifest,
    series_index,
    mutator,
    *,
    allow_nan=False,
):
    record = manifest["series"][series_index]
    path = root / "blind" / record["seriesFile"]
    payload = read_json(path)
    mutator(payload)
    payload_bytes = stable_json_bytes(payload, allow_nan=allow_nan)
    path.write_bytes(payload_bytes)
    record["sha256"] = sha256_bytes(payload_bytes)


def serialized_tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
    }


class CoarseGridBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        write_prepared_root(self.prepared)

    def build(self, name="output"):
        output = self.root / name
        result = build_coarse_grid_project(
            self.prepared,
            project_id=PROJECT_ID,
            output_root=output,
        )
        return output, result

    def test_valid_project_is_deterministic_and_directly_activatable(self):
        output, result = self.build()

        project = read_json(output / PROJECT_RELATIVE_PATH)
        dataset = read_json(output / DATASET_RELATIVE_PATH)
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        self.assertEqual(result["project"], project)
        self.assertEqual(PROJECT_ID, project["id"])
        self.assertEqual(WORKLOAD_ID, project["workloadID"])
        self.assertEqual(DATASET_SCHEMA_ID, project["datasetSchemaID"])
        self.assertEqual(PAYLOAD_SCHEMA_ID, project["payloadSchemaID"])
        self.assertEqual(RESULT_SCHEMA_ID, project["resultSchemaID"])
        self.assertEqual(1, len(project["datasets"]))
        self.assertEqual(DATASET_RELATIVE_PATH, project["datasets"][0]["path"])
        self.assertEqual(dataset["id"], project["datasets"][0]["id"])
        self.assertEqual("series-002", manifest["selectedSeriesID"])
        self.assertEqual(7, manifest["selectedSampleCount"])
        CURVE_GRID_PLUGIN.validate_dataset(dataset)

    def test_primary_selection_uses_greatest_sample_count(self):
        output, _ = self.build()

        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)

        self.assertEqual("series-002", manifest["selectedSeriesID"])
        self.assertEqual(7, manifest["selectedSampleCount"])

    def test_primary_tie_uses_ordered_series_position(self):
        self.prepared = self.root / "prepared-tie"
        self.prepared.mkdir()
        write_prepared_root(
            self.prepared,
            specifications=(("series-a", 6), ("series-b", 6)),
            ordered_ids=("series-b", "series-a"),
        )

        output, _ = self.build("tie-output")
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)

        self.assertEqual("series-b", manifest["selectedSeriesID"])

    def test_frozen_grid_and_accounting_are_exact(self):
        output, _ = self.build()

        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        dataset = read_json(output / DATASET_RELATIVE_PATH)
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        grid = dataset["curveGrid"]
        self.assertEqual(CENTER_AXIS, grid["centerAxis"])
        self.assertEqual(LOG_SCALE_AXIS, grid["logScaleAxis"])
        self.assertEqual(LOG_SHAPE_AXIS, grid["logShapeAxis"])
        self.assertEqual(FAMILY_ID, grid["familyID"])
        self.assertEqual(CANDIDATES_PER_WORK_UNIT, grid["candidatesPerWorkUnit"])
        self.assertEqual(2248.5, contract["curveGrid"]["centerAxis"]["endpoint"])
        self.assertEqual(
            [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
            contract["curveGrid"]["logScaleAxis"]["representedPositiveScales"],
        )
        self.assertEqual(
            1.0,
            contract["curveGrid"]["logShapeAxis"]["endpointPositiveShape"],
        )
        self.assertEqual(
            {
                "datasetSchemaID": DATASET_SCHEMA_ID,
                "payloadSchemaID": PAYLOAD_SCHEMA_ID,
                "resultSchemaID": RESULT_SCHEMA_ID,
                "workloadID": WORKLOAD_ID,
            },
            contract["schemaTuple"],
        )
        self.assertEqual(FAMILY_ID, contract["curveGrid"]["familyID"])
        self.assertEqual(4941, TOTAL_CANDIDATE_COUNT)
        self.assertEqual(TOTAL_CANDIDATE_COUNT, manifest["totalCandidateCount"])
        self.assertEqual(78, EXPECTED_WORK_UNIT_COUNT)
        self.assertEqual(EXPECTED_WORK_UNIT_COUNT, manifest["expectedWorkUnitCount"])
        self.assertEqual(
            7 * TOTAL_CANDIDATE_COUNT,
            manifest["expectedSampleCandidateEvaluationCount"],
        )
        self.assertEqual(
            {
                "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
                "coarseSearchContract": CONTRACT_RELATIVE_PATH,
                "dataset": DATASET_RELATIVE_PATH,
                "project": PROJECT_RELATIVE_PATH,
            },
            manifest["relativeArtifactPaths"],
        )

    def test_safe_accounting_rejects_json_unsafe_products(self):
        self.assertEqual(
            138170124,
            _safe_product(27964, TOTAL_CANDIDATE_COUNT, "evaluations"),
        )
        with self.assertRaisesRegex(CoarseGridBuildError, "safe integer"):
            _safe_product(MAX_SAFE_INTEGER, 2, "evaluations")

    def test_selected_arrays_are_preserved_without_downsampling_or_mutation(self):
        source = read_json(self.prepared / "blind" / "series" / "input-002.json")

        output, _ = self.build()
        dataset = read_json(output / DATASET_RELATIVE_PATH)

        self.assertEqual(source["coordinates"], dataset["coordinates"])
        self.assertEqual(source["values"], dataset["values"])
        self.assertEqual(source["inverseVariances"], dataset["inverseVariances"])

    def test_dataset_has_complete_tuple_and_accepted_opaque_metadata(self):
        output, _ = self.build()
        dataset = read_json(output / DATASET_RELATIVE_PATH)

        self.assertEqual(DATASET_SCHEMA_ID, dataset["datasetSchemaID"])
        self.assertEqual(FAMILY_ID, dataset["curveGrid"]["familyID"])
        self.assertEqual(BLIND_TARGET_ID, dataset["blindTargetID"])
        self.assertEqual("series-002", dataset["sourceGenericSeriesID"])
        self.assertEqual(PREPARATION_CONTRACT_ID, dataset["preparationContractID"])
        self.assertEqual(
            PREPARATION_CONTRACT_SHA256,
            dataset["preparationContractSHA256"],
        )
        self.assertEqual(COARSE_GRID_CONTRACT_ID, dataset["coarseSearchContractID"])
        self.assertEqual(
            COARSE_GRID_CONTRACT_SHA256,
            dataset["coarseSearchContractSHA256"],
        )
        CURVE_GRID_PLUGIN.validate_dataset(dataset)

    def test_contract_and_outputs_have_stable_hashes(self):
        first, _ = self.build("first")
        second, _ = self.build("second")

        self.assertEqual(serialized_tree(first), serialized_tree(second))
        contract = read_json(first / CONTRACT_RELATIVE_PATH)
        manifest = read_json(first / BUILD_MANIFEST_RELATIVE_PATH)
        self.assertEqual(COARSE_GRID_CONTRACT, contract)
        self.assertEqual(
            COARSE_GRID_CONTRACT_SHA256,
            sha256_bytes(canonical_json_bytes(contract)),
        )
        self.assertEqual(
            manifest["outputDatasetSHA256"],
            sha256_bytes((first / DATASET_RELATIVE_PATH).read_bytes()),
        )
        self.assertEqual(
            manifest["inputSeriesSHA256"],
            sha256_bytes(
                (self.prepared / "blind" / "series" / "input-002.json").read_bytes()
            ),
        )

    def test_sealed_directory_is_never_opened_or_read(self):
        sealed = self.prepared / "sealed"
        sealed.mkdir()
        (sealed / "identity-seal.json").write_text(
            "this file must remain unread",
            encoding="utf-8",
        )
        original_read_bytes = Path.read_bytes

        def guarded_read_bytes(path):
            if "sealed" in path.parts:
                raise AssertionError("sealed input was read")
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", guarded_read_bytes):
            self.build()

    def test_outputs_contain_no_source_identity_tokens(self):
        output, _ = self.build()

        serialized = b"\n".join(
            path.read_bytes() for path in sorted(output.rglob("*.json"))
        ).decode("utf-8").casefold()
        for token in FORBIDDEN_OUTPUT_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token.casefold(), serialized)

    def test_existing_output_root_is_rejected(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(CoarseGridBuildError, "already exists"):
            build_coarse_grid_project(
                self.prepared,
                project_id=PROJECT_ID,
                output_root=output,
            )
        self.assertEqual("keep", marker.read_text(encoding="utf-8"))

    def test_existing_output_symlink_is_rejected(self):
        target = self.root / "target"
        target.mkdir()
        output = self.root / "existing-link"
        os.symlink(target, output)

        with self.assertRaisesRegex(CoarseGridBuildError, "already exists"):
            build_coarse_grid_project(
                self.prepared,
                project_id=PROJECT_ID,
                output_root=output,
            )

    def test_existing_output_file_is_rejected(self):
        output = self.root / "existing-file"
        output.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(CoarseGridBuildError, "already exists"):
            build_coarse_grid_project(
                self.prepared,
                project_id=PROJECT_ID,
                output_root=output,
            )
        self.assertEqual("keep", output.read_text(encoding="utf-8"))


class CoarseGridInputRejectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, name):
        prepared = self.root / name
        prepared.mkdir()
        write_prepared_root(prepared)
        return prepared

    def assert_rejected(self, prepared, pattern=None):
        context = (
            self.assertRaisesRegex(CoarseGridBuildError, pattern)
            if pattern is not None
            else self.assertRaises(CoarseGridBuildError)
        )
        with context:
            build_coarse_grid_project(
                prepared,
                project_id=PROJECT_ID,
                output_root=self.root / f"output-{prepared.name}",
            )

    def test_preparation_contract_mismatch_is_rejected(self):
        prepared = self.fixture("contract")
        manifest = read_manifest(prepared)
        manifest["preparationContractSHA256"] = "0" * 64
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "contract SHA-256")

    def test_malformed_manifest_counts_are_rejected(self):
        prepared = self.fixture("counts")
        manifest = read_manifest(prepared)
        manifest["totalSampleCount"] += 1
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "totalSampleCount")

    def test_duplicate_series_ids_are_rejected(self):
        prepared = self.fixture("duplicate-ids")
        manifest = read_manifest(prepared)
        duplicate = copy.deepcopy(manifest["series"][0])
        duplicate["seriesFile"] = manifest["series"][2]["seriesFile"]
        manifest["series"][2] = duplicate
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "duplicate IDs")

    def test_duplicate_series_paths_are_rejected(self):
        prepared = self.fixture("duplicate-paths")
        manifest = read_manifest(prepared)
        manifest["series"][1]["seriesFile"] = manifest["series"][0]["seriesFile"]
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "duplicate paths")

    def test_missing_and_unexpected_ordered_ids_are_rejected(self):
        prepared = self.fixture("ordered-ids")
        manifest = read_manifest(prepared)
        manifest["orderedSeriesIDs"][1] = "series-unexpected"
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "missing or unexpected")

    def test_path_traversal_is_rejected(self):
        prepared = self.fixture("traversal")
        manifest = read_manifest(prepared)
        manifest["series"][0]["seriesFile"] = "../sealed/identity-seal.json"
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "unsafe")

    def test_symlinked_series_input_is_rejected(self):
        prepared = self.fixture("symlink")
        manifest = read_manifest(prepared)
        path = prepared / "blind" / manifest["series"][0]["seriesFile"]
        target = self.root / "symlink-target.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        os.symlink(target, path)

        self.assert_rejected(prepared, "non-symlink")

    def test_symlinked_preparation_manifest_is_rejected(self):
        prepared = self.fixture("manifest-symlink")
        path = prepared / "blind" / "preparation-manifest.json"
        target = self.root / "manifest-target.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        os.symlink(target, path)

        self.assert_rejected(prepared, "non-symlink")

    def test_series_hash_mismatch_is_rejected(self):
        prepared = self.fixture("hash")
        manifest = read_manifest(prepared)
        manifest["series"][0]["sha256"] = "0" * 64
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "SHA-256")

    def test_mismatched_target_and_series_ids_are_rejected(self):
        cases = (
            (
                "target",
                lambda payload: payload.__setitem__("blindTargetID", "other-target"),
                "target ID",
            ),
            (
                "series-id",
                lambda payload: payload.__setitem__("seriesID", "other-series"),
                "series ID",
            ),
        )
        for name, mutator, pattern in cases:
            with self.subTest(name=name):
                prepared = self.fixture(name)
                manifest = read_manifest(prepared)
                rewrite_series(prepared, manifest, 0, mutator)
                write_manifest(prepared, manifest)
                self.assert_rejected(prepared, pattern)

    def test_unequal_arrays_are_rejected(self):
        prepared = self.fixture("unequal")
        manifest = read_manifest(prepared)
        rewrite_series(
            prepared,
            manifest,
            0,
            lambda payload: payload["values"].pop(),
        )
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "equal length")

    def test_nonfinite_values_are_rejected(self):
        prepared = self.fixture("nonfinite")
        manifest = read_manifest(prepared)
        rewrite_series(
            prepared,
            manifest,
            0,
            lambda payload: payload["values"].__setitem__(0, math.nan),
            allow_nan=True,
        )
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "nonfinite")

    def test_nonpositive_inverse_variance_is_rejected(self):
        prepared = self.fixture("weight")
        manifest = read_manifest(prepared)
        rewrite_series(
            prepared,
            manifest,
            0,
            lambda payload: payload["inverseVariances"].__setitem__(0, 0.0),
        )
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "positive")

    def test_incorrect_coordinate_range_is_rejected(self):
        prepared = self.fixture("coordinate-range")
        manifest = read_manifest(prepared)
        manifest["series"][0]["coordinateRange"]["maximum"] += 1.0
        write_manifest(prepared, manifest)

        self.assert_rejected(prepared, "coordinate range")


if __name__ == "__main__":
    unittest.main()
