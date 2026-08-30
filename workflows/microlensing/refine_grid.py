"""Build a deterministic CurveGrid refinement from a verified coarse run."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openstar_investigation import canonical_json_bytes, sha256_json
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
    BUILD_MANIFEST_SCHEMA_ID as COARSE_BUILD_MANIFEST_SCHEMA_ID,
    CANDIDATES_PER_WORK_UNIT as COARSE_CANDIDATES_PER_WORK_UNIT,
    CENTER_AXIS as COARSE_CENTER_AXIS,
    COARSE_GRID_CONTRACT,
    COARSE_GRID_CONTRACT_ID,
    COARSE_GRID_CONTRACT_SHA256,
    CONTRACT_RELATIVE_PATH as COARSE_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as COARSE_DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT as COARSE_EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_AXIS as COARSE_LOG_SCALE_AXIS,
    LOG_SHAPE_AXIS as COARSE_LOG_SHAPE_AXIS,
    PROJECT_RELATIVE_PATH as COARSE_PROJECT_RELATIVE_PATH,
    TOTAL_CANDIDATE_COUNT as COARSE_TOTAL_CANDIDATE_COUNT,
    CoarseGridBuildError,
    _assert_identity_free,
    _atomic_write_bytes,
    _canonical_compact_json_bytes,
    _decode_json,
    _read_regular_file,
    _select_primary_series,
    _stable_json_bytes,
    _verify_blind_preparation,
)
from workflows.microlensing.prepare import (
    PREPARATION_CONTRACT_ID,
    PREPARATION_CONTRACT_SHA256,
)


REFINEMENT_GRID_CONTRACT_ID = "openstar.microlensing-refinement-grid.v1"
BUILD_MANIFEST_SCHEMA_ID = "openstar.microlensing-refinement-grid-build.v1"
SMOKE_WORKFLOW_ID = "openstar.workflow.project-smoke.v1"
PREPARE_HANDLER_ID = "local.project.prepare"
PROJECT_RUN_HANDLER_ID = "openstar.project.run"
TERMINAL_CHECK_HANDLER_ID = "generic.project.terminal-check"

CENTER_COUNT = 21
LOG_SCALE_COUNT = 17
LOG_SHAPE_COUNT = 17
CANDIDATES_PER_WORK_UNIT = 64
TOTAL_CANDIDATE_COUNT = CENTER_COUNT * LOG_SCALE_COUNT * LOG_SHAPE_COUNT
EXPECTED_WORK_UNIT_COUNT = (
    TOTAL_CANDIDATE_COUNT + CANDIDATES_PER_WORK_UNIT - 1
) // CANDIDATES_PER_WORK_UNIT

CONTRACT_RELATIVE_PATH = "refinement-search-contract.json"
DATASET_RELATIVE_PATH = "datasets/primary-series.json"
PROJECT_RELATIVE_PATH = "project.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"

_SAFE_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_STAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COARSE_BUILD_MANIFEST_FIELDS = frozenset(
    (
        "blindTargetID",
        "buildManifestSchemaID",
        "coarseSearchContractID",
        "coarseSearchContractSHA256",
        "expectedSampleCandidateEvaluationCount",
        "expectedWorkUnitCount",
        "inputSeriesSHA256",
        "outputDatasetSHA256",
        "preparationManifestSHA256",
        "projectID",
        "relativeArtifactPaths",
        "selectedSampleCount",
        "selectedSeriesID",
        "totalCandidateCount",
    )
)
_COARSE_DATASET_FIELDS = frozenset(
    (
        "blindTargetID",
        "coarseSearchContractID",
        "coarseSearchContractSHA256",
        "coordinates",
        "curveGrid",
        "datasetSchemaID",
        "id",
        "inverseVariances",
        "preparationContractID",
        "preparationContractSHA256",
        "sourceGenericSeriesID",
        "values",
    )
)
_PROJECT_FIELDS = frozenset(
    (
        "datasetSchemaID",
        "datasets",
        "id",
        "payloadSchemaID",
        "resultSchemaID",
        "workloadID",
    )
)
_PROJECT_DATASET_FIELDS = frozenset(("id", "path"))
_INVESTIGATION_FIELDS = frozenset(
    (
        "created_at",
        "id",
        "metadata",
        "stages",
        "status",
        "updated_at",
        "workflow_id",
        "workflow_version",
    )
)
_STAGE_FIELDS = frozenset(
    (
        "artifacts",
        "completed_at",
        "error",
        "failure_classification",
        "handler_id",
        "id",
        "next_stage",
        "parameters",
        "provenance",
        "result",
        "started_at",
        "status",
        "stop",
        "triggered_by_stage_id",
    )
)
_PROVENANCE_FIELDS = frozenset(
    (
        "input_hashes",
        "node_contributions",
        "parameters_hash",
        "project_ids",
        "result_hash",
        "software_id",
        "software_version",
    )
)
_ARTIFACT_FIELDS = frozenset(("media_type", "path", "sha256"))
_WINNING_RESULT_FIELDS = frozenset(
    (
        "bestAmplitude",
        "bestCenter",
        "bestGridIndex",
        "bestLogScale",
        "bestLogShape",
        "bestOffset",
        "bestWeightedResidualSumSquares",
        "evaluatedCandidateCount",
        "familyID",
        "gridCount",
        "gridStartIndex",
        "invalidCandidateCount",
    )
)


class RefinementGridBuildError(RuntimeError):
    """The refinement project cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class _VerifiedCoarseProject:
    project_id: str
    dataset_id: str
    selected_series: Any
    blind_target_id: str
    preparation_manifest_sha256: str
    build_manifest_sha256: str
    contract_file_sha256: str
    dataset_sha256: str
    project_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedWinner:
    grid_index: int
    center: float
    log_scale: float
    log_shape: float
    offset: float
    amplitude: float
    objective: float
    investigation_sha256: str
    run_stage_id: str
    run_stage_ledger_sha256: str
    stage_ledger_sha256s: Mapping[str, str]


def _fail(message: str) -> RefinementGridBuildError:
    return RefinementGridBuildError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{field_name} must be a lowercase SHA-256")
    return value


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field_name} must be a nonempty string")
    return value


def _safe_stage_id(value: Any) -> str:
    stage_id = _nonempty_string(value, "stage id")
    if _SAFE_STAGE_ID.fullmatch(stage_id) is None:
        raise _fail("stage id is unsafe")
    return stage_id


def _exact_integer(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{field_name} must be an integer of at least {minimum}")
    if value > MAX_SAFE_INTEGER:
        raise _fail(f"{field_name} exceeds the safe integer range")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _fail(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise _fail(f"{field_name} must be finite")
    return number


def _safe_product(left: int, right: int, field_name: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise _fail(f"{field_name} has invalid factors")
    if left and right > MAX_SAFE_INTEGER // left:
        raise _fail(f"{field_name} exceeds the safe integer range")
    return left * right


def _safe_sum(values: Sequence[int], field_name: str) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise _fail(f"{field_name} has invalid terms")
        if value > MAX_SAFE_INTEGER - total:
            raise _fail(f"{field_name} exceeds the safe integer range")
        total += value
    return total


def _regular_directory(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise _fail(f"{description} is not a regular non-symlink directory")
    return path


def _read_json_file(path: Path, description: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = _read_regular_file(path, description)
        decoded = _decode_json(payload, description)
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error
    return payload, decoded


def _verify_preparation(prepared_root: Path) -> tuple[Any, Any]:
    try:
        preparation = _verify_blind_preparation(prepared_root)
        selected = _select_primary_series(preparation.ordered_series)
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error
    return preparation, selected


def _coarse_grid() -> dict[str, Any]:
    return {
        "candidatesPerWorkUnit": COARSE_CANDIDATES_PER_WORK_UNIT,
        "centerAxis": dict(COARSE_CENTER_AXIS),
        "familyID": FAMILY_ID,
        "logScaleAxis": dict(COARSE_LOG_SCALE_AXIS),
        "logShapeAxis": dict(COARSE_LOG_SHAPE_AXIS),
    }


def _verify_coarse_project(
    prepared_root: Path,
    coarse_project_root: Path,
) -> _VerifiedCoarseProject:
    preparation, selected = _verify_preparation(prepared_root)
    coarse_root = _regular_directory(coarse_project_root, "coarse project root")

    build_bytes, build = _read_json_file(
        coarse_root / COARSE_BUILD_MANIFEST_RELATIVE_PATH,
        "coarse build manifest",
    )
    if set(build) != _COARSE_BUILD_MANIFEST_FIELDS:
        raise _fail("coarse build manifest field set is invalid")
    if build.get("buildManifestSchemaID") != COARSE_BUILD_MANIFEST_SCHEMA_ID:
        raise _fail("coarse build manifest schema ID is invalid")
    if build.get("coarseSearchContractID") != COARSE_GRID_CONTRACT_ID:
        raise _fail("coarse contract ID does not match")
    if build.get("coarseSearchContractSHA256") != COARSE_GRID_CONTRACT_SHA256:
        raise _fail("coarse contract SHA-256 does not match")
    expected_paths = {
        "buildManifest": COARSE_BUILD_MANIFEST_RELATIVE_PATH,
        "coarseSearchContract": COARSE_CONTRACT_RELATIVE_PATH,
        "dataset": COARSE_DATASET_RELATIVE_PATH,
        "project": COARSE_PROJECT_RELATIVE_PATH,
    }
    if build.get("relativeArtifactPaths") != expected_paths:
        raise _fail("coarse artifact paths are invalid")

    contract_bytes, contract = _read_json_file(
        coarse_root / COARSE_CONTRACT_RELATIVE_PATH,
        "coarse search contract",
    )
    if contract != COARSE_GRID_CONTRACT:
        raise _fail("coarse search contract content does not match")
    contract_hash = _sha256_bytes(_canonical_compact_json_bytes(contract))
    if contract_hash != COARSE_GRID_CONTRACT_SHA256:
        raise _fail("coarse search contract canonical hash does not match")

    dataset_bytes, dataset = _read_json_file(
        coarse_root / COARSE_DATASET_RELATIVE_PATH,
        "coarse dataset",
    )
    dataset_hash = _sha256_bytes(dataset_bytes)
    if dataset_hash != _sha256_string(
        build.get("outputDatasetSHA256"),
        "coarse outputDatasetSHA256",
    ):
        raise _fail("coarse dataset SHA-256 does not match")
    if set(dataset) != _COARSE_DATASET_FIELDS:
        raise _fail("coarse dataset field set is invalid")
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(f"coarse CurveGrid dataset is invalid: {error}") from error

    project_bytes, project = _read_json_file(
        coarse_root / COARSE_PROJECT_RELATIVE_PATH,
        "coarse project",
    )
    if set(project) != _PROJECT_FIELDS:
        raise _fail("coarse project field set is invalid")
    project_id = _nonempty_string(project.get("id"), "coarse project ID")
    if build.get("projectID") != project_id:
        raise _fail("coarse project ID does not match its build manifest")
    expected_tuple = {
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "workloadID": WORKLOAD_ID,
    }
    if any(project.get(key) != value for key, value in expected_tuple.items()):
        raise _fail("coarse project CurveGrid schema tuple is invalid")
    project_datasets = project.get("datasets")
    if not isinstance(project_datasets, list) or len(project_datasets) != 1:
        raise _fail("coarse project must contain exactly one dataset")
    project_dataset = project_datasets[0]
    if (
        not isinstance(project_dataset, Mapping)
        or set(project_dataset) != _PROJECT_DATASET_FIELDS
        or project_dataset.get("path") != COARSE_DATASET_RELATIVE_PATH
    ):
        raise _fail("coarse project dataset entry is invalid")
    dataset_id = _nonempty_string(dataset.get("id"), "coarse dataset ID")
    if project_dataset.get("id") != dataset_id:
        raise _fail("coarse project and dataset IDs do not match")

    expected_dataset_metadata = {
        "blindTargetID": preparation.blind_target_id,
        "coarseSearchContractID": COARSE_GRID_CONTRACT_ID,
        "coarseSearchContractSHA256": COARSE_GRID_CONTRACT_SHA256,
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "id": f"{project_id}.primary-series",
        "preparationContractID": PREPARATION_CONTRACT_ID,
        "preparationContractSHA256": PREPARATION_CONTRACT_SHA256,
        "sourceGenericSeriesID": selected.series_id,
    }
    if any(
        dataset.get(key) != value
        for key, value in expected_dataset_metadata.items()
    ):
        raise _fail("coarse dataset generic provenance does not match")
    if dataset.get("curveGrid") != _coarse_grid():
        raise _fail("coarse dataset grid does not match the frozen contract")
    selected_payload = selected.payload
    for field_name in ("coordinates", "values", "inverseVariances"):
        if dataset.get(field_name) != selected_payload[field_name]:
            raise _fail(f"coarse dataset {field_name} differs from blind series")

    selected_count = selected.sample_count
    expected_evaluations = _safe_product(
        selected_count,
        COARSE_TOTAL_CANDIDATE_COUNT,
        "coarse sample-candidate evaluation count",
    )
    expected_build_values = {
        "blindTargetID": preparation.blind_target_id,
        "expectedSampleCandidateEvaluationCount": expected_evaluations,
        "expectedWorkUnitCount": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "inputSeriesSHA256": selected.sha256,
        "preparationManifestSHA256": preparation.manifest_sha256,
        "selectedSampleCount": selected_count,
        "selectedSeriesID": selected.series_id,
        "totalCandidateCount": COARSE_TOTAL_CANDIDATE_COUNT,
    }
    if any(build.get(key) != value for key, value in expected_build_values.items()):
        raise _fail("coarse build manifest does not match verified inputs")

    return _VerifiedCoarseProject(
        project_id=project_id,
        dataset_id=dataset_id,
        selected_series=selected,
        blind_target_id=preparation.blind_target_id,
        preparation_manifest_sha256=preparation.manifest_sha256,
        build_manifest_sha256=_sha256_bytes(build_bytes),
        contract_file_sha256=_sha256_bytes(contract_bytes),
        dataset_sha256=dataset_hash,
        project_sha256=_sha256_bytes(project_bytes),
    )


def _validate_stage_shape(stage: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if set(stage) != _STAGE_FIELDS:
        raise _fail("investigation stage field set is invalid")
    stage_id = _safe_stage_id(stage.get("id"))
    _nonempty_string(stage.get("handler_id"), "stage handler ID")
    _nonempty_string(stage.get("started_at"), "stage start timestamp")
    _nonempty_string(stage.get("completed_at"), "stage completion timestamp")
    if type(stage.get("stop")) is not bool:
        raise _fail("investigation stage stop flag is invalid")
    triggered_by = stage.get("triggered_by_stage_id")
    if triggered_by is not None:
        _safe_stage_id(triggered_by)
    next_stage = stage.get("next_stage")
    if next_stage is not None and not isinstance(next_stage, Mapping):
        raise _fail("investigation stage continuation is invalid")
    if stage.get("status") != "COMPLETE":
        raise _fail("every persisted investigation stage must be COMPLETE")
    if (
        stage.get("error") is not None
        or stage.get("failure_classification") is not None
    ):
        raise _fail("completed investigation stage contains a failure")
    if not isinstance(stage.get("parameters"), Mapping):
        raise _fail("investigation stage parameters are invalid")
    if not isinstance(stage.get("result"), Mapping):
        raise _fail("investigation stage result is invalid")
    artifacts = stage.get("artifacts")
    if not isinstance(artifacts, list):
        raise _fail("investigation stage artifacts are invalid")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
            raise _fail("investigation artifact record is invalid")
        _nonempty_string(artifact.get("path"), "artifact path")
        _sha256_string(artifact.get("sha256"), "artifact SHA-256")

    provenance = stage.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_FIELDS:
        raise _fail("investigation stage provenance is invalid")
    _nonempty_string(provenance.get("software_id"), "stage software ID")
    _nonempty_string(provenance.get("software_version"), "stage software version")
    input_hashes = provenance.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise _fail("stage input hashes are invalid")
    for key, value in input_hashes.items():
        _nonempty_string(key, "stage input hash name")
        _sha256_string(value, "stage input hash")
    contributions = provenance.get("node_contributions")
    if not isinstance(contributions, Mapping):
        raise _fail("stage node contributions are invalid")
    for node_id, count in contributions.items():
        _nonempty_string(node_id, "contribution node ID")
        _exact_integer(count, "node contribution count")
    project_ids = provenance.get("project_ids")
    if not isinstance(project_ids, list):
        raise _fail("stage project IDs are invalid")
    for project_id in project_ids:
        _nonempty_string(project_id, "stage project ID")
    if provenance.get("parameters_hash") != sha256_json(stage["parameters"]):
        raise _fail("stage parameters hash does not match")
    if provenance.get("result_hash") != sha256_json(stage["result"]):
        raise _fail("stage result hash does not match")
    return stage_id, provenance


def _stage_ledgers(
    investigation_path: Path,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    stage_root = _regular_directory(
        investigation_path.parent / "stages",
        "investigation stage directory",
    )
    expected_names = {f"{stage['id']}.json" for stage in stages}
    try:
        entries = list(stage_root.iterdir())
    except OSError as error:
        raise _fail("investigation stage directory is unreadable") from error
    if {entry.name for entry in entries} != expected_names:
        raise _fail("investigation stage ledger set is incomplete or unexpected")

    hashes: dict[str, str] = {}
    for stage in stages:
        stage_id = stage["id"]
        ledger_bytes, ledger = _read_json_file(
            stage_root / f"{stage_id}.json",
            f"stage ledger {stage_id}",
        )
        if canonical_json_bytes(ledger) != canonical_json_bytes(stage):
            raise _fail(f"stage ledger {stage_id} does not match investigation")
        hashes[stage_id] = _sha256_bytes(ledger_bytes)
    return hashes


def _coarse_indices(grid_index: int) -> tuple[int, int, int]:
    if grid_index >= COARSE_TOTAL_CANDIDATE_COUNT:
        raise _fail("coarse best grid index is outside the grid")
    combined, shape_index = divmod(grid_index, COARSE_LOG_SHAPE_AXIS["count"])
    center_index, scale_index = divmod(
        combined,
        COARSE_LOG_SCALE_AXIS["count"],
    )
    return center_index, scale_index, shape_index


def _verify_winning_result(
    dataset_status: Mapping[str, Any],
) -> tuple[int, float, float, float, float, float, float]:
    payload = dataset_status.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"best"}:
        raise _fail("coarse dataset result payload is invalid")
    best = payload.get("best")
    if not isinstance(best, Mapping) or set(best) != _WINNING_RESULT_FIELDS:
        raise _fail("coarse winning result field set is invalid")
    if best.get("familyID") != FAMILY_ID:
        raise _fail("coarse winning result family is invalid")

    grid_index = _exact_integer(best.get("bestGridIndex"), "bestGridIndex")
    center = _finite_number(best.get("bestCenter"), "bestCenter")
    log_scale = _finite_number(best.get("bestLogScale"), "bestLogScale")
    log_shape = _finite_number(best.get("bestLogShape"), "bestLogShape")
    offset = _finite_number(best.get("bestOffset"), "bestOffset")
    amplitude = _finite_number(best.get("bestAmplitude"), "bestAmplitude")
    objective = _finite_number(
        best.get("bestWeightedResidualSumSquares"),
        "bestWeightedResidualSumSquares",
    )
    if objective < 0.0:
        raise _fail("coarse winning objective must be nonnegative")

    center_index, scale_index, shape_index = _coarse_indices(grid_index)
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
    if (center, log_scale, log_shape) != (
        expected_center,
        expected_log_scale,
        expected_log_shape,
    ):
        raise _fail("coarse winner parameters do not map exactly from its index")
    boundary_checks = (
        (center_index, COARSE_CENTER_AXIS["count"]),
        (scale_index, COARSE_LOG_SCALE_AXIS["count"]),
        (shape_index, COARSE_LOG_SHAPE_AXIS["count"]),
    )
    if any(index in {0, count - 1} for index, count in boundary_checks):
        raise _fail("coarse winner lies on a grid boundary")

    shard_start = (grid_index // COARSE_CANDIDATES_PER_WORK_UNIT) * (
        COARSE_CANDIDATES_PER_WORK_UNIT
    )
    shard_count = min(
        COARSE_CANDIDATES_PER_WORK_UNIT,
        COARSE_TOTAL_CANDIDATE_COUNT - shard_start,
    )
    if (
        best.get("gridStartIndex") != shard_start
        or best.get("gridCount") != shard_count
    ):
        raise _fail("coarse winning shard identity is invalid")
    if best.get("evaluatedCandidateCount") != shard_count:
        raise _fail("coarse winning result evaluated count is invalid")
    invalid_count = _exact_integer(
        best.get("invalidCandidateCount"),
        "invalidCandidateCount",
    )
    if invalid_count > shard_count:
        raise _fail("coarse winning result invalid count is impossible")

    status_matches = {
        "bestAmplitude": amplitude,
        "bestCenter": center,
        "bestGridIndex": grid_index,
        "bestLogScale": log_scale,
        "bestLogShape": log_shape,
        "bestOffset": offset,
        "bestWeightedResidualSumSquares": objective,
    }
    if any(dataset_status.get(key) != value for key, value in status_matches.items()):
        raise _fail("coarse dataset status and winning result disagree")
    return grid_index, center, log_scale, log_shape, offset, amplitude, objective


def _verify_investigation(
    investigation_path: Path,
    coarse: _VerifiedCoarseProject,
    coarse_project_path: Path,
) -> _VerifiedWinner:
    if investigation_path.name != "investigation.json":
        raise _fail("coarse investigation record must be named investigation.json")
    _regular_directory(
        investigation_path.parent,
        "coarse investigation directory",
    )
    record_bytes, investigation = _read_json_file(
        investigation_path,
        "coarse investigation record",
    )
    if set(investigation) != _INVESTIGATION_FIELDS:
        raise _fail("coarse investigation field set is invalid")
    investigation_id = _nonempty_string(
        investigation.get("id"),
        "investigation ID",
    )
    if _SAFE_STAGE_ID.fullmatch(investigation_id) is None:
        raise _fail("investigation ID is unsafe")
    if investigation_path.parent.name != investigation_id:
        raise _fail("investigation record directory does not match its ID")
    if investigation.get("workflow_id") != SMOKE_WORKFLOW_ID:
        raise _fail("coarse investigation workflow is invalid")
    _nonempty_string(investigation.get("workflow_version"), "workflow version")
    if investigation.get("status") != "COMPLETE":
        raise _fail("coarse investigation is not COMPLETE")
    metadata = investigation.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("coarse investigation metadata is invalid")
    expected_project_path = coarse_project_path.resolve()
    metadata_project_path = metadata.get("projectPath")
    if not isinstance(metadata_project_path, str):
        raise _fail("investigation project path is missing")
    try:
        if Path(metadata_project_path).expanduser().resolve() != expected_project_path:
            raise _fail("investigation metadata refers to a different project")
    except OSError as error:
        raise _fail("investigation project path cannot be resolved") from error

    stage_values = investigation.get("stages")
    if not isinstance(stage_values, list) or not stage_values:
        raise _fail("coarse investigation stages are invalid")
    stages: list[Mapping[str, Any]] = []
    stage_ids: set[str] = set()
    for value in stage_values:
        if not isinstance(value, Mapping):
            raise _fail("coarse investigation stage is malformed")
        stage_id, _ = _validate_stage_shape(value)
        if stage_id in stage_ids:
            raise _fail("coarse investigation contains duplicate stage IDs")
        stage_ids.add(stage_id)
        stages.append(value)
    if len(stages) != 3:
        raise _fail("project-smoke investigation must contain exactly three stages")
    ledger_hashes = _stage_ledgers(investigation_path, stages)

    prepare_stages = [
        stage for stage in stages if stage.get("handler_id") == PREPARE_HANDLER_ID
    ]
    if len(prepare_stages) != 1:
        raise _fail("investigation must contain exactly one project preparation")
    run_stages = [
        stage for stage in stages if stage.get("handler_id") == PROJECT_RUN_HANDLER_ID
    ]
    if len(run_stages) != 1:
        raise _fail("investigation must contain exactly one completed project run")
    terminal_stages = [
        stage
        for stage in stages
        if stage.get("handler_id") == TERMINAL_CHECK_HANDLER_ID
    ]
    if len(terminal_stages) != 1:
        raise _fail("investigation must contain exactly one terminal check")
    prepare_stage = prepare_stages[0]
    run_stage = run_stages[0]
    terminal_stage = terminal_stages[0]

    if stages != [prepare_stage, run_stage, terminal_stage]:
        raise _fail("project-smoke stages are out of canonical order")
    if prepare_stage.get("triggered_by_stage_id") is not None:
        raise _fail("project preparation stage has invalid causality")
    if run_stage.get("triggered_by_stage_id") != prepare_stage["id"]:
        raise _fail("project run stage has invalid causality")
    if terminal_stage.get("triggered_by_stage_id") != run_stage["id"]:
        raise _fail("terminal check stage has invalid causality")

    expected_project_hash = coarse.project_sha256
    expected_project_path_string = str(expected_project_path)
    prepare_parameters = prepare_stage["parameters"]
    prepare_result = prepare_stage["result"]
    if set(prepare_parameters) != {"projectPath"}:
        raise _fail("project preparation parameters are invalid")
    prepare_parameter_path = _nonempty_string(
        prepare_parameters.get("projectPath"),
        "project preparation path",
    )
    if Path(prepare_parameter_path).expanduser().is_absolute():
        try:
            resolved_prepare_path = (
                Path(prepare_parameter_path).expanduser().resolve()
            )
            if resolved_prepare_path != expected_project_path:
                raise _fail(
                    "project preparation parameters refer to a different project"
                )
        except OSError as error:
            raise _fail("project preparation path cannot be resolved") from error
    if prepare_result != {
        "projectManifestSha256": expected_project_hash,
        "projectPath": expected_project_path_string,
    }:
        raise _fail("project preparation result does not match coarse project")
    prepare_provenance = prepare_stage["provenance"]
    if prepare_provenance.get("input_hashes") != {
        "projectManifest": expected_project_hash
    }:
        raise _fail("project preparation input provenance does not match")
    if prepare_provenance.get("project_ids") != []:
        raise _fail("project preparation provenance has unexpected project IDs")
    if prepare_provenance.get("node_contributions") != {}:
        raise _fail("project preparation has unexpected node contributions")

    run_parameters = run_stage["parameters"]
    if set(run_parameters) != {"projectManifestSha256", "projectPath"}:
        raise _fail("project run parameters are invalid")
    if run_parameters.get("projectManifestSha256") != expected_project_hash:
        raise _fail("project run manifest hash does not match coarse project")
    run_project_path = run_parameters.get("projectPath")
    if not isinstance(run_project_path, str):
        raise _fail("project run path is missing")
    try:
        if Path(run_project_path).expanduser().resolve() != expected_project_path:
            raise _fail("project run refers to a different coarse project")
    except OSError as error:
        raise _fail("project run path cannot be resolved") from error
    run_provenance = run_stage["provenance"]
    if run_provenance.get("input_hashes") != {
        "projectManifest": expected_project_hash
    }:
        raise _fail("project run input provenance does not match")
    if run_provenance.get("project_ids") != [coarse.project_id]:
        raise _fail("project run provenance project ID does not match")
    if _safe_sum(
        list(run_provenance["node_contributions"].values()),
        "project run node contribution count",
    ) != COARSE_EXPECTED_WORK_UNIT_COUNT:
        raise _fail("project run node contributions do not match completed work")

    expected_run_request = {
        "handler_id": PROJECT_RUN_HANDLER_ID,
        "id": run_stage["id"],
        "parameters": dict(run_parameters),
        "triggered_by_stage_id": prepare_stage["id"],
    }
    if prepare_stage.get("next_stage") != expected_run_request:
        raise _fail("project preparation continuation is invalid")
    expected_terminal_request = {
        "handler_id": TERMINAL_CHECK_HANDLER_ID,
        "id": terminal_stage["id"],
        "parameters": dict(terminal_stage["parameters"]),
        "triggered_by_stage_id": run_stage["id"],
    }
    if run_stage.get("next_stage") != expected_terminal_request:
        raise _fail("project run continuation is invalid")
    if prepare_stage.get("stop") is not False or run_stage.get("stop") is not False:
        raise _fail("nonterminal project-smoke stage is marked terminal")

    run_result = run_stage["result"]
    required_run_values = {
        "projectAssignedWorkUnits": 0,
        "projectCompletedWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "projectFailedWorkUnits": 0,
        "projectPendingWorkUnits": 0,
        "projectTotalWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "projectID": coarse.project_id,
        "status": "COMPLETE",
        "workloadID": WORKLOAD_ID,
    }
    if any(run_result.get(key) != value for key, value in required_run_values.items()):
        raise _fail("coarse project run did not complete with exact coverage")
    reported_project_path = run_result.get("projectPath")
    if reported_project_path != expected_project_path_string:
        raise _fail("coarse project run status refers to a different project path")
    if run_result.get("nodeContributions") != run_provenance["node_contributions"]:
        raise _fail("coarse project run contribution provenance does not match")
    dataset_statuses = run_result.get("datasets")
    if not isinstance(dataset_statuses, list) or len(dataset_statuses) != 1:
        raise _fail("coarse project run must report exactly one dataset")
    dataset_status = dataset_statuses[0]
    if not isinstance(dataset_status, Mapping):
        raise _fail("coarse project dataset status is malformed")
    required_dataset_values = {
        "completedCandidateCount": COARSE_TOTAL_CANDIDATE_COUNT,
        "completedWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "coverageComplete": True,
        "curveGridStatus": "CURVE_GRID_COMPLETE",
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "failedWorkUnits": 0,
        "familyID": FAMILY_ID,
        "id": coarse.dataset_id,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "totalCandidateCount": COARSE_TOTAL_CANDIDATE_COUNT,
        "totalWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "workloadID": WORKLOAD_ID,
        "workloadStatus": "CURVE_GRID_COMPLETE",
    }
    if any(
        dataset_status.get(key) != value
        for key, value in required_dataset_values.items()
    ):
        raise _fail("coarse dataset status is incomplete or inconsistent")
    winner_values = _verify_winning_result(dataset_status)

    terminal_result = terminal_stage["result"]
    expected_terminal_result = {
        "completedWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
        "failedWorkUnits": 0,
        "passed": True,
        "projectID": coarse.project_id,
        "rule": "projectID matches and completed+failed == total",
        "totalWorkUnits": COARSE_EXPECTED_WORK_UNIT_COUNT,
    }
    if terminal_result != expected_terminal_result:
        raise _fail("coarse investigation terminal check did not pass exactly")
    if (
        terminal_stage.get("stop") is not True
        or terminal_stage.get("next_stage") is not None
    ):
        raise _fail("coarse investigation terminal stage is not terminal")
    terminal_parameters = terminal_stage["parameters"]
    if terminal_parameters != {"expectedProjectID": coarse.project_id}:
        raise _fail("terminal check expected project ID does not match")
    if terminal_stage["provenance"].get("project_ids") != [coarse.project_id]:
        raise _fail("terminal check provenance project ID does not match")
    if terminal_stage["provenance"].get("input_hashes") != {}:
        raise _fail("terminal check has unexpected input provenance")
    if terminal_stage["provenance"].get("node_contributions") != {}:
        raise _fail("terminal check has unexpected node contributions")

    return _VerifiedWinner(
        grid_index=winner_values[0],
        center=winner_values[1],
        log_scale=winner_values[2],
        log_shape=winner_values[3],
        offset=winner_values[4],
        amplitude=winner_values[5],
        objective=winner_values[6],
        investigation_sha256=_sha256_bytes(record_bytes),
        run_stage_id=run_stage["id"],
        run_stage_ledger_sha256=ledger_hashes[run_stage["id"]],
        stage_ledger_sha256s=dict(sorted(ledger_hashes.items())),
    )


def _derived_axes(winner: _VerifiedWinner) -> dict[str, dict[str, Any]]:
    center_step = COARSE_CENTER_AXIS["step"] / 10.0
    log_scale_step = COARSE_LOG_SCALE_AXIS["step"] / 8.0
    log_shape_step = COARSE_LOG_SHAPE_AXIS["step"] / 8.0
    return {
        "centerAxis": {
            "count": CENTER_COUNT,
            "start": winner.center - COARSE_CENTER_AXIS["step"],
            "step": center_step,
        },
        "logScaleAxis": {
            "count": LOG_SCALE_COUNT,
            "start": winner.log_scale - COARSE_LOG_SCALE_AXIS["step"],
            "step": log_scale_step,
        },
        "logShapeAxis": {
            "count": LOG_SHAPE_COUNT,
            "start": winner.log_shape - COARSE_LOG_SHAPE_AXIS["step"],
            "step": log_shape_step,
        },
    }


def _contract(
    coarse: _VerifiedCoarseProject,
    winner: _VerifiedWinner,
    axes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "axisProvenanceStatement": (
            "Axes were derived only from the verified coarse winner and frozen "
            "coarse-axis steps without consulting sealed identity or expected "
            "published parameters."
        ),
        "benchmarkKind": "known-event-recovery",
        "candidateCount": TOTAL_CANDIDATE_COUNT,
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant whitespace, "
            "non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractID": REFINEMENT_GRID_CONTRACT_ID,
        "curveGrid": {
            "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
            "centerAxis": dict(axes["centerAxis"]),
            "familyID": FAMILY_ID,
            "logScaleAxis": dict(axes["logScaleAxis"]),
            "logShapeAxis": dict(axes["logShapeAxis"]),
        },
        "derivationRules": {
            "center": {
                "count": CENTER_COUNT,
                "start": "coarseBestCenter - coarseCenterStep",
                "step": "coarseCenterStep / 10",
            },
            "logScale": {
                "count": LOG_SCALE_COUNT,
                "start": "coarseBestLogScale - coarseLogScaleStep",
                "step": "coarseLogScaleStep / 8",
            },
            "logShape": {
                "count": LOG_SHAPE_COUNT,
                "start": "coarseBestLogShape - coarseLogShapeStep",
                "step": "coarseLogShapeStep / 8",
            },
        },
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "modelScopeStatement": (
            "This phase still fits only the smooth symmetric radial-amplification "
            "family and does not recover or classify a planetary anomaly."
        ),
        "schemaTuple": {
            "datasetSchemaID": DATASET_SCHEMA_ID,
            "payloadSchemaID": PAYLOAD_SCHEMA_ID,
            "resultSchemaID": RESULT_SCHEMA_ID,
            "workloadID": WORKLOAD_ID,
        },
        "verifiedCoarseRun": {
            "bestGridIndex": winner.grid_index,
            "bestWeightedResidualSumSquares": winner.objective,
            "coarseContractID": COARSE_GRID_CONTRACT_ID,
            "coarseContractSHA256": COARSE_GRID_CONTRACT_SHA256,
            "coarseProjectID": coarse.project_id,
            "runStageID": winner.run_stage_id,
            "runStageLedgerSHA256": winner.run_stage_ledger_sha256,
        },
    }


def _dataset(
    project_id: str,
    coarse: _VerifiedCoarseProject,
    winner: _VerifiedWinner,
    axes: Mapping[str, Mapping[str, Any]],
    contract_sha256: str,
) -> dict[str, Any]:
    source = coarse.selected_series.payload
    dataset = {
        "blindTargetID": coarse.blind_target_id,
        "coarseProjectID": coarse.project_id,
        "coarseRunStageLedgerSHA256": winner.run_stage_ledger_sha256,
        "coarseSearchContractID": COARSE_GRID_CONTRACT_ID,
        "coarseSearchContractSHA256": COARSE_GRID_CONTRACT_SHA256,
        "coordinates": list(source["coordinates"]),
        "curveGrid": {
            "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
            "centerAxis": dict(axes["centerAxis"]),
            "familyID": FAMILY_ID,
            "logScaleAxis": dict(axes["logScaleAxis"]),
            "logShapeAxis": dict(axes["logShapeAxis"]),
        },
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "id": f"{project_id}.primary-series",
        "inverseVariances": list(source["inverseVariances"]),
        "preparationContractID": PREPARATION_CONTRACT_ID,
        "preparationContractSHA256": PREPARATION_CONTRACT_SHA256,
        "refinementSearchContractID": REFINEMENT_GRID_CONTRACT_ID,
        "refinementSearchContractSHA256": contract_sha256,
        "sourceGenericSeriesID": coarse.selected_series.series_id,
        "values": list(source["values"]),
    }
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(f"constructed refinement dataset is invalid: {error}") from error
    return dataset


def _project(project_id: str, dataset_id: str) -> dict[str, Any]:
    return {
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "datasets": [{"id": dataset_id, "path": DATASET_RELATIVE_PATH}],
        "id": project_id,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "workloadID": WORKLOAD_ID,
    }


def build_refinement_grid_project(
    prepared_root: str | Path,
    *,
    coarse_project_root: str | Path,
    coarse_investigation_record: str | Path,
    project_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify the coarse chain and publish one deterministic refinement."""

    if (
        not isinstance(project_id, str)
        or _SAFE_PROJECT_ID.fullmatch(project_id) is None
    ):
        raise _fail("project ID is malformed or unsafe")
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    prepared = Path(prepared_root).expanduser().absolute()
    coarse_root = Path(coarse_project_root).expanduser().absolute()
    investigation_path = Path(coarse_investigation_record).expanduser().absolute()
    _regular_directory(prepared, "prepared root")

    coarse = _verify_coarse_project(prepared, coarse_root)
    winner = _verify_investigation(
        investigation_path,
        coarse,
        coarse_root / COARSE_PROJECT_RELATIVE_PATH,
    )
    axes = _derived_axes(winner)
    contract = _contract(coarse, winner, axes)
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    dataset = _dataset(
        project_id,
        coarse,
        winner,
        axes,
        contract_sha256,
    )
    project = _project(project_id, dataset["id"])

    evaluation_count = _safe_product(
        coarse.selected_series.sample_count,
        TOTAL_CANDIDATE_COUNT,
        "refinement sample-candidate evaluation count",
    )
    contract_bytes = _stable_json_bytes(contract)
    dataset_bytes = _stable_json_bytes(dataset)
    project_bytes = _stable_json_bytes(project)
    build_manifest = {
        "blindTargetID": coarse.blind_target_id,
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID,
        "coarseBuildManifestSHA256": coarse.build_manifest_sha256,
        "coarseContractFileSHA256": coarse.contract_file_sha256,
        "coarseDatasetSHA256": coarse.dataset_sha256,
        "coarseInvestigationRecordSHA256": winner.investigation_sha256,
        "coarseProjectID": coarse.project_id,
        "coarseProjectSHA256": coarse.project_sha256,
        "coarseRunStageLedgerSHA256": winner.run_stage_ledger_sha256,
        "coarseStageLedgerSHA256s": dict(winner.stage_ledger_sha256s),
        "expectedSampleCandidateEvaluationCount": evaluation_count,
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "inputSeriesSHA256": coarse.selected_series.sha256,
        "outputDatasetSHA256": _sha256_bytes(dataset_bytes),
        "outputContractFileSHA256": _sha256_bytes(contract_bytes),
        "outputProjectSHA256": _sha256_bytes(project_bytes),
        "preparationManifestSHA256": coarse.preparation_manifest_sha256,
        "projectID": project_id,
        "refinementSearchContractID": REFINEMENT_GRID_CONTRACT_ID,
        "refinementSearchContractSHA256": contract_sha256,
        "relativeArtifactPaths": {
            "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
            "dataset": DATASET_RELATIVE_PATH,
            "project": PROJECT_RELATIVE_PATH,
            "refinementSearchContract": CONTRACT_RELATIVE_PATH,
        },
        "selectedSampleCount": coarse.selected_series.sample_count,
        "selectedSeriesID": coarse.selected_series.series_id,
        "totalCandidateCount": TOTAL_CANDIDATE_COUNT,
    }
    build_manifest_bytes = _stable_json_bytes(build_manifest)
    try:
        _assert_identity_free(
            (contract_bytes, dataset_bytes, project_bytes, build_manifest_bytes)
        )
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise _fail("output root already exists") from error
    _atomic_write_bytes(output / CONTRACT_RELATIVE_PATH, contract_bytes)
    _atomic_write_bytes(output / DATASET_RELATIVE_PATH, dataset_bytes)
    _atomic_write_bytes(output / PROJECT_RELATIVE_PATH, project_bytes)
    _atomic_write_bytes(output / BUILD_MANIFEST_RELATIVE_PATH, build_manifest_bytes)
    return {
        "buildManifest": build_manifest,
        "contract": contract,
        "dataset": dataset,
        "project": project,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic CurveGrid refinement from a verified blind "
            "preparation and completed coarse investigation."
        )
    )
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--coarse-project-root", required=True, type=Path)
    parser.add_argument("--coarse-investigation-record", required=True, type=Path)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_refinement_grid_project(
        arguments.prepared_root,
        coarse_project_root=arguments.coarse_project_root,
        coarse_investigation_record=arguments.coarse_investigation_record,
        project_id=arguments.project_id,
        output_root=arguments.output_root,
    )
    manifest = result["buildManifest"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind refinement-grid project ready")
    print(f"project ID: {manifest['projectID']}")
    print(f"coarse project ID: {manifest['coarseProjectID']}")
    print(f"selected generic series: {manifest['selectedSeriesID']}")
    print(f"selected samples: {manifest['selectedSampleCount']}")
    print(f"grid candidates: {manifest['totalCandidateCount']}")
    print(f"expected work units: {manifest['expectedWorkUnitCount']}")
    print(f"project: {output / PROJECT_RELATIVE_PATH}")
    print(f"dataset: {output / DATASET_RELATIVE_PATH}")
    print(f"build manifest: {output / BUILD_MANIFEST_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
