import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_investigation import sha256_json
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
    BUILD_MANIFEST_RELATIVE_PATH as COARSE_BUILD_MANIFEST_RELATIVE_PATH,
    CENTER_AXIS as COARSE_CENTER_AXIS,
    CONTRACT_RELATIVE_PATH as COARSE_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as COARSE_DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT as COARSE_EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_AXIS as COARSE_LOG_SCALE_AXIS,
    LOG_SHAPE_AXIS as COARSE_LOG_SHAPE_AXIS,
    PROJECT_RELATIVE_PATH as COARSE_PROJECT_RELATIVE_PATH,
    TOTAL_CANDIDATE_COUNT as COARSE_TOTAL_CANDIDATE_COUNT,
    build_coarse_grid_project,
)
from workflows.microlensing.prepare import (
    BLIND_MANIFEST_SCHEMA_ID,
    PREPARATION_CONTRACT_ID,
    PREPARATION_CONTRACT_SHA256,
    SERIES_SCHEMA_ID,
)
from workflows.microlensing.refine_grid import (
    BUILD_MANIFEST_RELATIVE_PATH,
    CANDIDATES_PER_WORK_UNIT,
    CENTER_COUNT,
    CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT,
    PROJECT_RELATIVE_PATH,
    REFINEMENT_GRID_CONTRACT_ID,
    TOTAL_CANDIDATE_COUNT,
    RefinementGridBuildError,
    _safe_product,
    build_refinement_grid_project,
)


BLIND_TARGET_ID = "openstar.generic-recovery-a.v1"
COARSE_PROJECT_ID = "openstar.generic-recovery-a.coarse-grid.v1"
REFINEMENT_PROJECT_ID = "openstar.generic-recovery-a.refinement-grid.v1"
INVESTIGATION_ID = "generic-coarse-investigation"
WINNING_CENTER_INDEX = 30
WINNING_SCALE_INDEX = 4
WINNING_SHAPE_INDEX = 4
WINNING_GRID_INDEX = (
    (
        WINNING_CENTER_INDEX * COARSE_LOG_SCALE_AXIS["count"]
        + WINNING_SCALE_INDEX
    )
    * COARSE_LOG_SHAPE_AXIS["count"]
    + WINNING_SHAPE_INDEX
)
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


def generic_series(series_id, sample_count, start):
    return {
        "blindTargetID": BLIND_TARGET_ID,
        "coordinates": [start + index * 0.25 for index in range(sample_count)],
        "inverseVariances": [25.0 + index for index in range(sample_count)],
        "seriesID": series_id,
        "seriesSchemaID": SERIES_SCHEMA_ID,
        "values": [1.0 + index * 0.02 for index in range(sample_count)],
    }


def write_prepared_root(root):
    blind = root / "blind"
    records = []
    ordered_ids = []
    specifications = (("series-001", 5), ("series-002", 8), ("series-003", 6))
    for index, (series_id, sample_count) in enumerate(specifications, 1):
        payload = generic_series(series_id, sample_count, index * 0.5)
        relative_path = f"series/{series_id}.json"
        payload_bytes = stable_json_bytes(payload)
        path = blind / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload_bytes)
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
        ordered_ids.append(series_id)
    manifest = {
        "benchmarkKind": "known-event-recovery",
        "blindTargetID": BLIND_TARGET_ID,
        "orderedSeriesIDs": ordered_ids,
        "preparationContractID": PREPARATION_CONTRACT_ID,
        "preparationContractSHA256": PREPARATION_CONTRACT_SHA256,
        "preparationManifestSchemaID": BLIND_MANIFEST_SCHEMA_ID,
        "series": records,
        "totalSampleCount": sum(record["sampleCount"] for record in records),
        "totalSeriesCount": len(records),
    }
    write_json(blind / "preparation-manifest.json", manifest)


def stage_record(
    *,
    stage_id,
    handler_id,
    triggered_by,
    parameters,
    result,
    input_hashes,
    project_ids,
    next_stage,
    stop,
    node_contributions=None,
):
    return {
        "artifacts": [],
        "completed_at": "2026-08-30T00:00:01+00:00",
        "error": None,
        "failure_classification": None,
        "handler_id": handler_id,
        "id": stage_id,
        "next_stage": next_stage,
        "parameters": parameters,
        "provenance": {
            "input_hashes": input_hashes,
            "node_contributions": dict(node_contributions or {}),
            "parameters_hash": sha256_json(parameters),
            "project_ids": list(project_ids),
            "result_hash": sha256_json(result),
            "software_id": "openstar.workflow-engine",
            "software_version": "20.0",
        },
        "result": result,
        "started_at": "2026-08-30T00:00:00+00:00",
        "status": "COMPLETE",
        "stop": stop,
        "triggered_by_stage_id": triggered_by,
    }


def winning_result(
    *,
    grid_index=WINNING_GRID_INDEX,
    center=None,
    log_scale=None,
    log_shape=None,
    objective=12.5,
):
    combined, shape_index = divmod(
        grid_index,
        COARSE_LOG_SHAPE_AXIS["count"],
    )
    center_index, scale_index = divmod(
        combined,
        COARSE_LOG_SCALE_AXIS["count"],
    )
    expected_center = (
        COARSE_CENTER_AXIS["start"] + center_index * COARSE_CENTER_AXIS["step"]
    )
    expected_log_scale = (
        COARSE_LOG_SCALE_AXIS["start"]
        + scale_index * COARSE_LOG_SCALE_AXIS["step"]
    )
    expected_log_shape = (
        COARSE_LOG_SHAPE_AXIS["start"]
        + shape_index * COARSE_LOG_SHAPE_AXIS["step"]
    )
    shard_start = (grid_index // 64) * 64
    shard_count = min(64, COARSE_TOTAL_CANDIDATE_COUNT - shard_start)
    return {
        "bestAmplitude": 1.25,
        "bestCenter": expected_center if center is None else center,
        "bestGridIndex": grid_index,
        "bestLogScale": expected_log_scale if log_scale is None else log_scale,
        "bestLogShape": expected_log_shape if log_shape is None else log_shape,
        "bestOffset": -0.25,
        "bestWeightedResidualSumSquares": objective,
        "evaluatedCandidateCount": shard_count,
        "familyID": FAMILY_ID,
        "gridCount": shard_count,
        "gridStartIndex": shard_start,
        "invalidCandidateCount": 0,
    }


def write_investigation(root, coarse_root, *, best=None):
    investigation_root = root / INVESTIGATION_ID
    stages_root = investigation_root / "stages"
    stages_root.mkdir(parents=True)
    project_path = (coarse_root / COARSE_PROJECT_RELATIVE_PATH).resolve()
    project_hash = sha256_bytes(project_path.read_bytes())
    dataset = read_json(coarse_root / COARSE_DATASET_RELATIVE_PATH)
    best = dict(best or winning_result())

    dataset_status = {
        "bestAmplitude": best["bestAmplitude"],
        "bestCenter": best["bestCenter"],
        "bestGridIndex": best["bestGridIndex"],
        "bestLogScale": best["bestLogScale"],
        "bestLogShape": best["bestLogShape"],
        "bestOffset": best["bestOffset"],
        "bestWeightedResidualSumSquares": best[
            "bestWeightedResidualSumSquares"
        ],
        "completedCandidateCount": COARSE_TOTAL_CANDIDATE_COUNT,
        "completedWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "coverageComplete": True,
        "curveGridStatus": "CURVE_GRID_COMPLETE",
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "failedWorkUnits": 0,
        "familyID": FAMILY_ID,
        "id": dataset["id"],
        "payload": {"best": best},
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "totalCandidateCount": COARSE_TOTAL_CANDIDATE_COUNT,
        "totalWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "workloadID": WORKLOAD_ID,
        "workloadStatus": "CURVE_GRID_COMPLETE",
    }
    run_result = {
        "datasets": [dataset_status],
        "nodeContributions": {"generic-node": COARSE_EXPECTED_WORK_UNIT_COUNT},
        "projectAssignedWorkUnits": 0,
        "projectCompletedWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "projectFailedWorkUnits": 0,
        "projectID": COARSE_PROJECT_ID,
        "projectPath": str(project_path),
        "projectPendingWorkUnits": 0,
        "projectTotalWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "status": "COMPLETE",
        "workloadID": WORKLOAD_ID,
    }
    prepare_parameters = {"projectPath": str(project_path)}
    run_parameters = {
        "projectManifestSha256": project_hash,
        "projectPath": str(project_path),
    }
    terminal_parameters = {"expectedProjectID": COARSE_PROJECT_ID}
    terminal_result = {
        "completedWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "failedWorkUnits": 0,
        "passed": True,
        "projectID": COARSE_PROJECT_ID,
        "rule": "projectID matches and completed+failed == total",
        "totalWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
    }
    prepare = stage_record(
        stage_id="001-prepare-project",
        handler_id="local.project.prepare",
        triggered_by=None,
        parameters=prepare_parameters,
        result={
            "projectManifestSha256": project_hash,
            "projectPath": str(project_path),
        },
        input_hashes={"projectManifest": project_hash},
        project_ids=(),
        next_stage={
            "handler_id": "openstar.project.run",
            "id": "002-distributed-project",
            "parameters": run_parameters,
            "triggered_by_stage_id": "001-prepare-project",
        },
        stop=False,
    )
    run = stage_record(
        stage_id="002-distributed-project",
        handler_id="openstar.project.run",
        triggered_by="001-prepare-project",
        parameters=run_parameters,
        result=run_result,
        input_hashes={"projectManifest": project_hash},
        project_ids=(COARSE_PROJECT_ID,),
        next_stage={
            "handler_id": "generic.project.terminal-check",
            "id": "003-terminal-check",
            "parameters": terminal_parameters,
            "triggered_by_stage_id": "002-distributed-project",
        },
        stop=False,
        node_contributions={"generic-node": COARSE_EXPECTED_WORK_UNIT_COUNT},
    )
    terminal = stage_record(
        stage_id="003-terminal-check",
        handler_id="generic.project.terminal-check",
        triggered_by="002-distributed-project",
        parameters=terminal_parameters,
        result=terminal_result,
        input_hashes={},
        project_ids=(COARSE_PROJECT_ID,),
        next_stage=None,
        stop=True,
    )
    stages = [prepare, run, terminal]
    investigation = {
        "created_at": "2026-08-30T00:00:00+00:00",
        "id": INVESTIGATION_ID,
        "metadata": {
            "coordinator": "http://127.0.0.1:8080",
            "projectPath": str(project_path),
        },
        "stages": stages,
        "status": "COMPLETE",
        "updated_at": "2026-08-30T00:00:02+00:00",
        "workflow_id": "openstar.workflow.project-smoke.v1",
        "workflow_version": "20.0",
    }
    record_path = investigation_root / "investigation.json"
    write_json(record_path, investigation)
    for stage in stages:
        write_json(stages_root / f"{stage['id']}.json", stage)
    return record_path


def rewrite_investigation_stage(record_path, stage_index, mutator):
    investigation = read_json(record_path)
    stage = investigation["stages"][stage_index]
    mutator(stage)
    stage["provenance"]["parameters_hash"] = sha256_json(stage["parameters"])
    stage["provenance"]["result_hash"] = sha256_json(stage["result"])
    write_json(record_path, investigation)
    write_json(record_path.parent / "stages" / f"{stage['id']}.json", stage)


def serialized_tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
    }


class RefinementGridFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        write_prepared_root(self.prepared)
        self.coarse = self.root / "coarse"
        build_coarse_grid_project(
            self.prepared,
            project_id=COARSE_PROJECT_ID,
            output_root=self.coarse,
        )
        self.investigation = write_investigation(
            self.root / "investigations",
            self.coarse,
        )

    def build(self, name="refinement"):
        output = self.root / name
        result = build_refinement_grid_project(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.investigation,
            project_id=REFINEMENT_PROJECT_ID,
            output_root=output,
        )
        return output, result


class RefinementGridBuildTests(RefinementGridFixture):
    def test_valid_refinement_is_deterministic_and_directly_activatable(self):
        output, result = self.build()

        project = read_json(output / PROJECT_RELATIVE_PATH)
        dataset = read_json(output / DATASET_RELATIVE_PATH)
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        self.assertEqual(result["project"], project)
        self.assertEqual(REFINEMENT_PROJECT_ID, project["id"])
        self.assertEqual(WORKLOAD_ID, project["workloadID"])
        self.assertEqual(DATASET_SCHEMA_ID, project["datasetSchemaID"])
        self.assertEqual(PAYLOAD_SCHEMA_ID, project["payloadSchemaID"])
        self.assertEqual(RESULT_SCHEMA_ID, project["resultSchemaID"])
        self.assertEqual(1, len(project["datasets"]))
        self.assertEqual(DATASET_RELATIVE_PATH, project["datasets"][0]["path"])
        self.assertEqual(dataset["id"], project["datasets"][0]["id"])
        self.assertEqual("series-002", manifest["selectedSeriesID"])
        self.assertEqual(8, manifest["selectedSampleCount"])
        CURVE_GRID_PLUGIN.validate_dataset(dataset)

    def test_complete_input_provenance_is_recorded(self):
        output, _ = self.build()

        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        coarse_manifest = self.coarse / COARSE_BUILD_MANIFEST_RELATIVE_PATH
        coarse_contract = self.coarse / COARSE_CONTRACT_RELATIVE_PATH
        coarse_dataset = self.coarse / COARSE_DATASET_RELATIVE_PATH
        coarse_project = self.coarse / COARSE_PROJECT_RELATIVE_PATH
        selected_series = self.prepared / "blind" / "series" / "series-002.json"
        preparation_manifest = self.prepared / "blind" / "preparation-manifest.json"
        self.assertEqual(
            sha256_bytes(coarse_manifest.read_bytes()),
            manifest["coarseBuildManifestSHA256"],
        )
        self.assertEqual(
            sha256_bytes(coarse_contract.read_bytes()),
            manifest["coarseContractFileSHA256"],
        )
        self.assertEqual(
            sha256_bytes(coarse_dataset.read_bytes()),
            manifest["coarseDatasetSHA256"],
        )
        self.assertEqual(
            sha256_bytes(coarse_project.read_bytes()),
            manifest["coarseProjectSHA256"],
        )
        self.assertEqual(
            sha256_bytes(self.investigation.read_bytes()),
            manifest["coarseInvestigationRecordSHA256"],
        )
        self.assertEqual(
            sha256_bytes(selected_series.read_bytes()),
            manifest["inputSeriesSHA256"],
        )
        self.assertEqual(
            sha256_bytes(preparation_manifest.read_bytes()),
            manifest["preparationManifestSHA256"],
        )
        for stage_id, digest in manifest["coarseStageLedgerSHA256s"].items():
            self.assertEqual(
                sha256_bytes(
                    (self.investigation.parent / "stages" / f"{stage_id}.json")
                    .read_bytes()
                ),
                digest,
            )
        self.assertEqual(
            manifest["coarseStageLedgerSHA256s"]["002-distributed-project"],
            manifest["coarseRunStageLedgerSHA256"],
        )
        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        self.assertEqual(
            manifest["coarseRunStageLedgerSHA256"],
            contract["verifiedCoarseRun"]["runStageLedgerSHA256"],
        )

    def test_derived_axes_and_exact_counts_are_frozen(self):
        output, _ = self.build()

        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        dataset = read_json(output / DATASET_RELATIVE_PATH)
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        grid = dataset["curveGrid"]
        winner = winning_result()
        expected_center = {
            "count": CENTER_COUNT,
            "start": winner["bestCenter"] - COARSE_CENTER_AXIS["step"],
            "step": COARSE_CENTER_AXIS["step"] / 10.0,
        }
        expected_log_scale = {
            "count": LOG_SCALE_COUNT,
            "start": winner["bestLogScale"] - COARSE_LOG_SCALE_AXIS["step"],
            "step": COARSE_LOG_SCALE_AXIS["step"] / 8.0,
        }
        expected_log_shape = {
            "count": LOG_SHAPE_COUNT,
            "start": winner["bestLogShape"] - COARSE_LOG_SHAPE_AXIS["step"],
            "step": COARSE_LOG_SHAPE_AXIS["step"] / 8.0,
        }
        self.assertEqual(expected_center, grid["centerAxis"])
        self.assertEqual(expected_log_scale, grid["logScaleAxis"])
        self.assertEqual(expected_log_shape, grid["logShapeAxis"])
        self.assertEqual(grid, contract["curveGrid"])
        self.assertEqual(FAMILY_ID, grid["familyID"])
        self.assertEqual(CANDIDATES_PER_WORK_UNIT, grid["candidatesPerWorkUnit"])
        self.assertEqual(
            {
                "datasetSchemaID": DATASET_SCHEMA_ID,
                "payloadSchemaID": PAYLOAD_SCHEMA_ID,
                "resultSchemaID": RESULT_SCHEMA_ID,
                "workloadID": WORKLOAD_ID,
            },
            contract["schemaTuple"],
        )
        self.assertEqual(
            WINNING_GRID_INDEX,
            contract["verifiedCoarseRun"]["bestGridIndex"],
        )
        self.assertEqual(
            12.5,
            contract["verifiedCoarseRun"][
                "bestWeightedResidualSumSquares"
            ],
        )
        self.assertEqual(6069, TOTAL_CANDIDATE_COUNT)
        self.assertEqual(TOTAL_CANDIDATE_COUNT, manifest["totalCandidateCount"])
        self.assertEqual(95, EXPECTED_WORK_UNIT_COUNT)
        self.assertEqual(EXPECTED_WORK_UNIT_COUNT, manifest["expectedWorkUnitCount"])
        self.assertEqual(
            8 * TOTAL_CANDIDATE_COUNT,
            manifest["expectedSampleCandidateEvaluationCount"],
        )
        self.assertEqual(
            {
                "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
                "dataset": DATASET_RELATIVE_PATH,
                "project": PROJECT_RELATIVE_PATH,
                "refinementSearchContract": CONTRACT_RELATIVE_PATH,
            },
            manifest["relativeArtifactPaths"],
        )

    def test_selected_arrays_are_preserved_exactly(self):
        source = read_json(
            self.prepared / "blind" / "series" / "series-002.json"
        )

        output, _ = self.build()
        dataset = read_json(output / DATASET_RELATIVE_PATH)

        self.assertEqual(source["coordinates"], dataset["coordinates"])
        self.assertEqual(source["values"], dataset["values"])
        self.assertEqual(source["inverseVariances"], dataset["inverseVariances"])

    def test_contract_and_artifact_hashes_are_stable(self):
        first, _ = self.build("first")
        second, _ = self.build("second")

        self.assertEqual(serialized_tree(first), serialized_tree(second))
        contract = read_json(first / CONTRACT_RELATIVE_PATH)
        manifest = read_json(first / BUILD_MANIFEST_RELATIVE_PATH)
        self.assertEqual(REFINEMENT_GRID_CONTRACT_ID, contract["contractID"])
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(contract)),
            manifest["refinementSearchContractSHA256"],
        )
        self.assertEqual(
            sha256_bytes((first / CONTRACT_RELATIVE_PATH).read_bytes()),
            manifest["outputContractFileSHA256"],
        )
        self.assertEqual(
            sha256_bytes((first / DATASET_RELATIVE_PATH).read_bytes()),
            manifest["outputDatasetSHA256"],
        )
        self.assertEqual(
            sha256_bytes((first / PROJECT_RELATIVE_PATH).read_bytes()),
            manifest["outputProjectSHA256"],
        )

    def test_safe_evaluation_accounting(self):
        self.assertEqual(
            169713516,
            _safe_product(27964, TOTAL_CANDIDATE_COUNT, "evaluations"),
        )
        with self.assertRaisesRegex(RefinementGridBuildError, "safe integer"):
            _safe_product(MAX_SAFE_INTEGER, 2, "evaluations")

    def test_existing_output_file_directory_and_symlink_are_rejected(self):
        target = self.root / "target"
        target.mkdir()
        cases = []
        directory = self.root / "existing-directory"
        directory.mkdir()
        cases.append(directory)
        file_path = self.root / "existing-file"
        file_path.write_text("keep", encoding="utf-8")
        cases.append(file_path)
        link = self.root / "existing-link"
        os.symlink(target, link)
        cases.append(link)

        for output in cases:
            with self.subTest(output=output.name):
                with self.assertRaisesRegex(
                    RefinementGridBuildError,
                    "already exists",
                ):
                    build_refinement_grid_project(
                        self.prepared,
                        coarse_project_root=self.coarse,
                        coarse_investigation_record=self.investigation,
                        project_id=REFINEMENT_PROJECT_ID,
                        output_root=output,
                    )
        self.assertEqual("keep", file_path.read_text(encoding="utf-8"))

    def test_sealed_directory_is_never_opened(self):
        sealed = self.prepared / "sealed"
        sealed.mkdir()
        (sealed / "identity-seal.json").write_text(
            "must remain unread",
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


class RefinementGridRejectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, name):
        root = self.root / name
        prepared = root / "prepared"
        prepared.mkdir(parents=True)
        write_prepared_root(prepared)
        coarse = root / "coarse"
        build_coarse_grid_project(
            prepared,
            project_id=COARSE_PROJECT_ID,
            output_root=coarse,
        )
        investigation = write_investigation(root / "investigations", coarse)
        return root, prepared, coarse, investigation

    def assert_rejected(self, prepared, coarse, investigation, pattern=None):
        context = (
            self.assertRaisesRegex(RefinementGridBuildError, pattern)
            if pattern is not None
            else self.assertRaises(RefinementGridBuildError)
        )
        with context:
            build_refinement_grid_project(
                prepared,
                coarse_project_root=coarse,
                coarse_investigation_record=investigation,
                project_id=REFINEMENT_PROJECT_ID,
                output_root=prepared.parent / "rejected-output",
            )

    def test_tampered_preparation_is_rejected(self):
        _, prepared, coarse, investigation = self.fixture("preparation")
        manifest_path = prepared / "blind" / "preparation-manifest.json"
        manifest = read_json(manifest_path)
        manifest["preparationContractSHA256"] = "0" * 64
        write_json(manifest_path, manifest)

        self.assert_rejected(prepared, coarse, investigation, "contract SHA-256")

    def test_tampered_coarse_artifacts_are_rejected(self):
        cases = (
            (
                "contract",
                COARSE_CONTRACT_RELATIVE_PATH,
                lambda value: value.__setitem__("candidateCount", 1),
                "contract content",
            ),
            (
                "dataset",
                COARSE_DATASET_RELATIVE_PATH,
                lambda value: value["values"].__setitem__(0, 9.0),
                "dataset SHA-256",
            ),
            (
                "project",
                COARSE_PROJECT_RELATIVE_PATH,
                lambda value: value["datasets"][0].__setitem__(
                    "path",
                    "../outside.json",
                ),
                "dataset entry",
            ),
            (
                "build-manifest",
                COARSE_BUILD_MANIFEST_RELATIVE_PATH,
                lambda value: value.__setitem__("selectedSeriesID", "other"),
                "verified inputs",
            ),
        )
        for name, relative_path, mutator, pattern in cases:
            with self.subTest(name=name):
                _, prepared, coarse, investigation = self.fixture(name)
                path = coarse / relative_path
                value = read_json(path)
                mutator(value)
                write_json(path, value)
                self.assert_rejected(prepared, coarse, investigation, pattern)

    def test_tampered_investigation_and_stage_ledger_are_rejected(self):
        _, prepared, coarse, investigation = self.fixture("investigation")
        record = read_json(investigation)
        record["workflow_id"] = "other.workflow"
        write_json(investigation, record)
        self.assert_rejected(prepared, coarse, investigation, "workflow")

        _, prepared, coarse, investigation = self.fixture("ledger")
        ledger = investigation.parent / "stages" / "002-distributed-project.json"
        raw = read_json(ledger)
        raw["completed_at"] = "tampered"
        write_json(ledger, raw)
        self.assert_rejected(prepared, coarse, investigation, "does not match")

        _, prepared, coarse, investigation = self.fixture("stage-provenance")

        def alter_provenance(stage):
            stage["provenance"]["input_hashes"] = {"projectManifest": "0" * 64}

        rewrite_investigation_stage(investigation, 1, alter_provenance)
        self.assert_rejected(prepared, coarse, investigation, "input provenance")

    def test_tampered_result_counts_objective_and_parameters_are_rejected(self):
        cases = (
            (
                "counts",
                lambda stage: stage["result"].__setitem__(
                    "projectCompletedWorkUnits",
                    77,
                ),
                "exact coverage",
            ),
            (
                "objective",
                lambda stage: (
                    stage["result"]["datasets"][0]["payload"]["best"].__setitem__(
                        "bestWeightedResidualSumSquares",
                        -1.0,
                    ),
                    stage["result"]["datasets"][0].__setitem__(
                        "bestWeightedResidualSumSquares",
                        -1.0,
                    ),
                ),
                "nonnegative",
            ),
            (
                "parameters",
                lambda stage: (
                    stage["result"]["datasets"][0]["payload"]["best"].__setitem__(
                        "bestCenter",
                        100.0,
                    ),
                    stage["result"]["datasets"][0].__setitem__(
                        "bestCenter",
                        100.0,
                    ),
                ),
                "map exactly",
            ),
        )
        for name, mutator, pattern in cases:
            with self.subTest(name=name):
                _, prepared, coarse, investigation = self.fixture(name)
                rewrite_investigation_stage(investigation, 1, mutator)
                self.assert_rejected(prepared, coarse, investigation, pattern)

    def test_boundary_winner_is_rejected_even_when_parameters_match(self):
        _, prepared, coarse, investigation = self.fixture("boundary")
        boundary = winning_result(
            grid_index=(
                (0 * COARSE_LOG_SCALE_AXIS["count"] + WINNING_SCALE_INDEX)
                * COARSE_LOG_SHAPE_AXIS["count"]
                + WINNING_SHAPE_INDEX
            )
        )

        def replace_best(stage):
            stage["result"]["datasets"][0]["payload"]["best"] = boundary
            for key in (
                "bestAmplitude",
                "bestCenter",
                "bestGridIndex",
                "bestLogScale",
                "bestLogShape",
                "bestOffset",
                "bestWeightedResidualSumSquares",
            ):
                stage["result"]["datasets"][0][key] = boundary[key]

        rewrite_investigation_stage(investigation, 1, replace_best)

        self.assert_rejected(prepared, coarse, investigation, "boundary")

    def test_symlinked_and_unsafe_inputs_are_rejected(self):
        _, prepared, coarse, investigation = self.fixture("symlink")
        dataset_path = coarse / COARSE_DATASET_RELATIVE_PATH
        target = dataset_path.with_name("dataset-target.json")
        target.write_bytes(dataset_path.read_bytes())
        dataset_path.unlink()
        os.symlink(target, dataset_path)
        self.assert_rejected(prepared, coarse, investigation, "non-symlink")

        _, prepared, coarse, investigation = self.fixture("traversal")
        project_path = coarse / COARSE_PROJECT_RELATIVE_PATH
        project = read_json(project_path)
        project["datasets"][0]["path"] = "../outside.json"
        write_json(project_path, project)
        self.assert_rejected(prepared, coarse, investigation, "dataset entry")

        _, prepared, coarse, investigation = self.fixture("ledger-symlink")
        ledger = investigation.parent / "stages" / "002-distributed-project.json"
        target = self.root / "run-stage-target.json"
        target.write_bytes(ledger.read_bytes())
        ledger.unlink()
        os.symlink(target, ledger)
        self.assert_rejected(prepared, coarse, investigation, "non-symlink")

    def test_missing_or_unexpected_stage_ledger_is_rejected(self):
        _, prepared, coarse, investigation = self.fixture("missing-ledger")
        ledger = investigation.parent / "stages" / "001-prepare-project.json"
        ledger.unlink()
        self.assert_rejected(prepared, coarse, investigation, "ledger set")

        _, prepared, coarse, investigation = self.fixture("extra-ledger")
        write_json(investigation.parent / "stages" / "unexpected.json", {})
        self.assert_rejected(prepared, coarse, investigation, "ledger set")


if __name__ == "__main__":
    unittest.main()
