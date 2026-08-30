"""Build a deterministic recentered CurveGrid from a verified refinement."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    COARSE_GRID_CONTRACT_ID,
    COARSE_GRID_CONTRACT_SHA256,
    CONTRACT_RELATIVE_PATH as COARSE_CONTRACT_RELATIVE_PATH,
    PROJECT_RELATIVE_PATH as COARSE_PROJECT_RELATIVE_PATH,
    CoarseGridBuildError,
    _assert_identity_free,
    _atomic_write_bytes,
    _canonical_compact_json_bytes,
    _decode_json,
    _read_regular_file,
    _stable_json_bytes,
)
from workflows.microlensing.refine_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
    BUILD_MANIFEST_SCHEMA_ID as REFINEMENT_BUILD_MANIFEST_SCHEMA_ID,
    CANDIDATES_PER_WORK_UNIT as REFINEMENT_CANDIDATES_PER_WORK_UNIT,
    CENTER_COUNT as REFINEMENT_CENTER_COUNT,
    CONTRACT_RELATIVE_PATH as REFINEMENT_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as REFINEMENT_DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT as REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_COUNT as REFINEMENT_LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT as REFINEMENT_LOG_SHAPE_COUNT,
    PROJECT_RELATIVE_PATH as REFINEMENT_PROJECT_RELATIVE_PATH,
    REFINEMENT_GRID_CONTRACT_ID,
    TOTAL_CANDIDATE_COUNT as REFINEMENT_TOTAL_CANDIDATE_COUNT,
    RefinementGridBuildError,
    _VerifiedCoarseProject,
    _VerifiedWinner as _VerifiedCoarseWinner,
    _contract as _expected_refinement_contract,
    _dataset as _expected_refinement_dataset,
    _derived_axes as _expected_refinement_axes,
    _exact_integer,
    _finite_number,
    _nonempty_string,
    _project as _expected_refinement_project,
    _safe_product as _refinement_safe_product,
    _safe_stage_id,
    _safe_sum,
    _stage_ledgers,
    _validate_stage_shape,
    _verify_coarse_project,
    _verify_investigation as _verify_coarse_investigation,
)


RECENTERED_GRID_CONTRACT_ID = "openstar.microlensing-recentered-grid.v1"
BUILD_MANIFEST_SCHEMA_ID = "openstar.microlensing-recentered-grid-build.v1"
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

CONTRACT_RELATIVE_PATH = "recentered-search-contract.json"
DATASET_RELATIVE_PATH = "datasets/primary-series.json"
PROJECT_RELATIVE_PATH = "project.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"

_SAFE_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
_REQUIRED_DATASET_STATUS_FIELDS = frozenset(
    (
        "bestAmplitude",
        "bestCenter",
        "bestGridIndex",
        "bestLogScale",
        "bestLogShape",
        "bestOffset",
        "bestWeightedResidualSumSquares",
        "completedCandidateCount",
        "completedWorkUnits",
        "coverageComplete",
        "curveGridStatus",
        "datasetSchemaID",
        "failedWorkUnits",
        "familyID",
        "id",
        "payload",
        "payloadSchemaID",
        "resultSchemaID",
        "totalCandidateCount",
        "totalWorkUnits",
        "workloadID",
        "workloadStatus",
    )
)
_REQUIRED_RUN_RESULT_FIELDS = frozenset(
    (
        "datasets",
        "nodeContributions",
        "projectAssignedWorkUnits",
        "projectCompletedWorkUnits",
        "projectFailedWorkUnits",
        "projectID",
        "projectPath",
        "projectPendingWorkUnits",
        "projectTotalWorkUnits",
        "status",
        "workloadID",
    )
)
_REFINEMENT_BUILD_MANIFEST_FIELDS = frozenset(
    (
        "blindTargetID",
        "buildManifestSchemaID",
        "coarseBuildManifestSHA256",
        "coarseContractFileSHA256",
        "coarseDatasetSHA256",
        "coarseInvestigationRecordSHA256",
        "coarseProjectID",
        "coarseProjectSHA256",
        "coarseRunStageLedgerSHA256",
        "coarseStageLedgerSHA256s",
        "expectedSampleCandidateEvaluationCount",
        "expectedWorkUnitCount",
        "inputSeriesSHA256",
        "outputContractFileSHA256",
        "outputDatasetSHA256",
        "outputProjectSHA256",
        "preparationManifestSHA256",
        "projectID",
        "refinementSearchContractID",
        "refinementSearchContractSHA256",
        "relativeArtifactPaths",
        "selectedSampleCount",
        "selectedSeriesID",
        "totalCandidateCount",
    )
)


class RecenterGridBuildError(RuntimeError):
    """The recentered project cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class _VerifiedRefinementProject:
    project_id: str
    dataset_id: str
    axes: Mapping[str, Mapping[str, Any]]
    build_manifest_sha256: str
    contract_file_sha256: str
    contract_sha256: str
    dataset_sha256: str
    project_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedRefinementWinner:
    grid_index: int
    center_index: int
    log_scale_index: int
    log_shape_index: int
    center: float
    log_scale: float
    log_shape: float
    offset: float
    amplitude: float
    objective: float
    boundary_axes: tuple[str, ...]
    result_payload: Mapping[str, Any]
    investigation_sha256: str
    run_stage_id: str
    run_stage_ledger_sha256: str
    stage_ledger_sha256s: Mapping[str, str]


def _fail(message: str) -> RecenterGridBuildError:
    return RecenterGridBuildError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_product(left: int, right: int, field_name: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise _fail(f"{field_name} has invalid factors")
    if left and right > MAX_SAFE_INTEGER // left:
        raise _fail(f"{field_name} exceeds the safe integer range")
    return left * right


def _exact_count(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise _fail(f"{field_name} must be a nonnegative integer")
    if value > MAX_SAFE_INTEGER:
        raise _fail(f"{field_name} exceeds the safe integer range")
    return value


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


def _verify_exact_project_tree(root: Path, contract_name: str, label: str) -> None:
    directory = _regular_directory(root, f"{label} root")
    expected_root_names = {
        "build-manifest.json",
        contract_name,
        "datasets",
        "project.json",
    }
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise _fail(f"{label} root is unreadable") from error
    if {entry.name for entry in entries} != expected_root_names:
        raise _fail(f"{label} artifact set is incomplete or unexpected")
    datasets = _regular_directory(directory / "datasets", f"{label} datasets")
    try:
        dataset_entries = list(datasets.iterdir())
    except OSError as error:
        raise _fail(f"{label} datasets directory is unreadable") from error
    if {entry.name for entry in dataset_entries} != {"primary-series.json"}:
        raise _fail(f"{label} dataset artifact set is incomplete or unexpected")


def _verify_investigation_tree(investigation_path: Path, label: str) -> None:
    directory = _regular_directory(investigation_path.parent, f"{label} directory")
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise _fail(f"{label} directory is unreadable") from error
    if {entry.name for entry in entries} != {"investigation.json", "stages"}:
        raise _fail(f"{label} file set is incomplete or unexpected")
    _regular_directory(directory / "stages", f"{label} stage directory")


def _reject_symlink_components(path: Path, description: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise _fail(f"{description} traverses a symlink")


def _verify_prepare_parameter_path(
    investigation_path: Path,
    expected_project_path: Path,
    label: str,
) -> None:
    _, investigation = _read_json_file(investigation_path, f"{label} record")
    stages = investigation.get("stages")
    if not isinstance(stages, list):
        raise _fail(f"{label} stages are invalid")
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("artifacts") != []:
            raise _fail(f"{label} contains unexpected stage artifacts")
    prepare_stages = [
        stage
        for stage in stages
        if isinstance(stage, Mapping)
        and stage.get("handler_id") == PREPARE_HANDLER_ID
    ]
    if len(prepare_stages) != 1:
        raise _fail(f"{label} preparation stage is invalid")
    parameters = prepare_stages[0].get("parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != {"projectPath"}:
        raise _fail(f"{label} preparation parameters are invalid")
    recorded = _nonempty_string(
        parameters.get("projectPath"),
        f"{label} preparation project path",
    )
    try:
        if Path(recorded).expanduser().resolve() != expected_project_path.resolve():
            raise _fail(f"{label} preparation refers to a different project")
    except OSError as error:
        raise _fail(f"{label} preparation path cannot be resolved") from error


def _verify_refinement_project(
    refinement_root: Path,
    coarse: _VerifiedCoarseProject,
    coarse_winner: _VerifiedCoarseWinner,
) -> _VerifiedRefinementProject:
    _verify_exact_project_tree(
        refinement_root,
        REFINEMENT_CONTRACT_RELATIVE_PATH,
        "first-refinement project",
    )
    build_bytes, build = _read_json_file(
        refinement_root / REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
        "first-refinement build manifest",
    )
    if set(build) != _REFINEMENT_BUILD_MANIFEST_FIELDS:
        raise _fail("first-refinement build manifest field set is invalid")
    if build.get("buildManifestSchemaID") != REFINEMENT_BUILD_MANIFEST_SCHEMA_ID:
        raise _fail("first-refinement build manifest schema ID is invalid")
    if build.get("refinementSearchContractID") != REFINEMENT_GRID_CONTRACT_ID:
        raise _fail("first-refinement contract ID does not match")
    expected_paths = {
        "buildManifest": REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
        "dataset": REFINEMENT_DATASET_RELATIVE_PATH,
        "project": REFINEMENT_PROJECT_RELATIVE_PATH,
        "refinementSearchContract": REFINEMENT_CONTRACT_RELATIVE_PATH,
    }
    if build.get("relativeArtifactPaths") != expected_paths:
        raise _fail("first-refinement artifact paths are invalid")

    contract_bytes, contract = _read_json_file(
        refinement_root / REFINEMENT_CONTRACT_RELATIVE_PATH,
        "first-refinement search contract",
    )
    axes = _expected_refinement_axes(coarse_winner)
    expected_contract = _expected_refinement_contract(coarse, coarse_winner, axes)
    if contract != expected_contract:
        raise _fail("first-refinement search contract content does not match")
    if contract_bytes != _stable_json_bytes(expected_contract):
        raise _fail("first-refinement search contract serialization is unstable")
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    if build.get("refinementSearchContractSHA256") != contract_sha256:
        raise _fail("first-refinement contract canonical hash does not match")
    contract_file_sha256 = _sha256_bytes(contract_bytes)
    if build.get("outputContractFileSHA256") != contract_file_sha256:
        raise _fail("first-refinement contract file hash does not match")

    project_bytes, project = _read_json_file(
        refinement_root / REFINEMENT_PROJECT_RELATIVE_PATH,
        "first-refinement project manifest",
    )
    project_id = _nonempty_string(project.get("id"), "first-refinement project ID")
    if _SAFE_PROJECT_ID.fullmatch(project_id) is None:
        raise _fail("first-refinement project ID is unsafe")
    if build.get("projectID") != project_id:
        raise _fail("first-refinement project ID does not match its build manifest")

    expected_dataset = _expected_refinement_dataset(
        project_id,
        coarse,
        coarse_winner,
        axes,
        contract_sha256,
    )
    dataset_bytes, dataset = _read_json_file(
        refinement_root / REFINEMENT_DATASET_RELATIVE_PATH,
        "first-refinement dataset",
    )
    if dataset != expected_dataset:
        raise _fail("first-refinement dataset does not match verified inputs")
    if dataset_bytes != _stable_json_bytes(expected_dataset):
        raise _fail("first-refinement dataset serialization is unstable")
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(f"first-refinement CurveGrid dataset is invalid: {error}") from error

    dataset_id = _nonempty_string(dataset.get("id"), "first-refinement dataset ID")
    expected_project = _expected_refinement_project(project_id, dataset_id)
    if project != expected_project:
        raise _fail("first-refinement project manifest does not match")
    if project_bytes != _stable_json_bytes(expected_project):
        raise _fail("first-refinement project serialization is unstable")

    dataset_sha256 = _sha256_bytes(dataset_bytes)
    project_sha256 = _sha256_bytes(project_bytes)
    evaluation_count = _refinement_safe_product(
        coarse.selected_series.sample_count,
        REFINEMENT_TOTAL_CANDIDATE_COUNT,
        "first-refinement sample-candidate evaluation count",
    )
    expected_build = {
        "blindTargetID": coarse.blind_target_id,
        "buildManifestSchemaID": REFINEMENT_BUILD_MANIFEST_SCHEMA_ID,
        "coarseBuildManifestSHA256": coarse.build_manifest_sha256,
        "coarseContractFileSHA256": coarse.contract_file_sha256,
        "coarseDatasetSHA256": coarse.dataset_sha256,
        "coarseInvestigationRecordSHA256": coarse_winner.investigation_sha256,
        "coarseProjectID": coarse.project_id,
        "coarseProjectSHA256": coarse.project_sha256,
        "coarseRunStageLedgerSHA256": coarse_winner.run_stage_ledger_sha256,
        "coarseStageLedgerSHA256s": dict(coarse_winner.stage_ledger_sha256s),
        "expectedSampleCandidateEvaluationCount": evaluation_count,
        "expectedWorkUnitCount": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "inputSeriesSHA256": coarse.selected_series.sha256,
        "outputContractFileSHA256": contract_file_sha256,
        "outputDatasetSHA256": dataset_sha256,
        "outputProjectSHA256": project_sha256,
        "preparationManifestSHA256": coarse.preparation_manifest_sha256,
        "projectID": project_id,
        "refinementSearchContractID": REFINEMENT_GRID_CONTRACT_ID,
        "refinementSearchContractSHA256": contract_sha256,
        "relativeArtifactPaths": expected_paths,
        "selectedSampleCount": coarse.selected_series.sample_count,
        "selectedSeriesID": coarse.selected_series.series_id,
        "totalCandidateCount": REFINEMENT_TOTAL_CANDIDATE_COUNT,
    }
    if build != expected_build:
        raise _fail("first-refinement build manifest provenance is incomplete")
    if build_bytes != _stable_json_bytes(expected_build):
        raise _fail("first-refinement build manifest serialization is unstable")

    return _VerifiedRefinementProject(
        project_id=project_id,
        dataset_id=dataset_id,
        axes={key: dict(value) for key, value in axes.items()},
        build_manifest_sha256=_sha256_bytes(build_bytes),
        contract_file_sha256=contract_file_sha256,
        contract_sha256=contract_sha256,
        dataset_sha256=dataset_sha256,
        project_sha256=project_sha256,
    )


def _refinement_indices(grid_index: int) -> tuple[int, int, int]:
    if grid_index >= REFINEMENT_TOTAL_CANDIDATE_COUNT:
        raise _fail("first-refinement best grid index is outside the grid")
    combined, shape_index = divmod(grid_index, REFINEMENT_LOG_SHAPE_COUNT)
    center_index, scale_index = divmod(combined, REFINEMENT_LOG_SCALE_COUNT)
    return center_index, scale_index, shape_index


def _verify_winning_result(
    dataset_status: Mapping[str, Any],
    axes: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int, int, int, float, float, float, float, float, float, tuple[str, ...]]:
    payload = dataset_status.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"best"}:
        raise _fail("first-refinement dataset result payload is invalid")
    best = payload.get("best")
    if not isinstance(best, Mapping) or set(best) != _WINNING_RESULT_FIELDS:
        raise _fail("first-refinement winning result field set is invalid")
    if best.get("familyID") != FAMILY_ID:
        raise _fail("first-refinement winning result family is invalid")

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
        raise _fail("first-refinement winning objective must be nonnegative")

    center_index, scale_index, shape_index = _refinement_indices(grid_index)
    center_axis = axes["centerAxis"]
    scale_axis = axes["logScaleAxis"]
    shape_axis = axes["logShapeAxis"]
    expected_values = (
        center_axis["start"] + center_index * center_axis["step"],
        scale_axis["start"] + scale_index * scale_axis["step"],
        shape_axis["start"] + shape_index * shape_axis["step"],
    )
    if (center, log_scale, log_shape) != expected_values:
        raise _fail(
            "first-refinement winner parameters do not map exactly from its index"
        )
    boundary_axes = tuple(
        name
        for name, index, count in (
            ("center", center_index, REFINEMENT_CENTER_COUNT),
            ("logScale", scale_index, REFINEMENT_LOG_SCALE_COUNT),
            ("logShape", shape_index, REFINEMENT_LOG_SHAPE_COUNT),
        )
        if index in {0, count - 1}
    )
    if not boundary_axes:
        raise _fail("first-refinement winner is interior on every axis")

    shard_start = (grid_index // REFINEMENT_CANDIDATES_PER_WORK_UNIT) * (
        REFINEMENT_CANDIDATES_PER_WORK_UNIT
    )
    shard_count = min(
        REFINEMENT_CANDIDATES_PER_WORK_UNIT,
        REFINEMENT_TOTAL_CANDIDATE_COUNT - shard_start,
    )
    result_grid_start = _exact_count(
        best.get("gridStartIndex"),
        "gridStartIndex",
    )
    result_grid_count = _exact_count(best.get("gridCount"), "gridCount")
    evaluated_count = _exact_count(
        best.get("evaluatedCandidateCount"),
        "evaluatedCandidateCount",
    )
    if result_grid_start != shard_start or result_grid_count != shard_count:
        raise _fail("first-refinement winning shard identity is invalid")
    if evaluated_count != shard_count:
        raise _fail("first-refinement winning result evaluated count is invalid")
    invalid_count = _exact_count(
        best.get("invalidCandidateCount"),
        "invalidCandidateCount",
    )
    if invalid_count > shard_count:
        raise _fail("first-refinement winning result invalid count is impossible")

    status_grid_index = _exact_integer(
        dataset_status.get("bestGridIndex"),
        "dataset status bestGridIndex",
    )
    status_numbers = tuple(
        _finite_number(dataset_status.get(field_name), f"dataset status {field_name}")
        for field_name in (
            "bestAmplitude",
            "bestCenter",
            "bestLogScale",
            "bestLogShape",
            "bestOffset",
            "bestWeightedResidualSumSquares",
        )
    )
    if (status_grid_index, *status_numbers) != (
        grid_index,
        amplitude,
        center,
        log_scale,
        log_shape,
        offset,
        objective,
    ):
        raise _fail("first-refinement dataset status and winning result disagree")
    return (
        grid_index,
        center_index,
        scale_index,
        shape_index,
        center,
        log_scale,
        log_shape,
        offset,
        amplitude,
        objective,
        boundary_axes,
    )


def _verify_refinement_investigation(
    investigation_path: Path,
    refinement: _VerifiedRefinementProject,
    refinement_project_path: Path,
) -> _VerifiedRefinementWinner:
    if investigation_path.name != "investigation.json":
        raise _fail(
            "first-refinement investigation record must be named investigation.json"
        )
    _verify_investigation_tree(
        investigation_path,
        "first-refinement investigation",
    )
    record_bytes, investigation = _read_json_file(
        investigation_path,
        "first-refinement investigation record",
    )
    if set(investigation) != _INVESTIGATION_FIELDS:
        raise _fail("first-refinement investigation field set is invalid")
    investigation_id = _nonempty_string(investigation.get("id"), "investigation ID")
    if investigation_path.parent.name != investigation_id:
        raise _fail("first-refinement investigation directory does not match its ID")
    _safe_stage_id(investigation_id)
    if investigation.get("workflow_id") != SMOKE_WORKFLOW_ID:
        raise _fail("first-refinement investigation workflow is invalid")
    _nonempty_string(investigation.get("workflow_version"), "workflow version")
    if investigation.get("status") != "COMPLETE":
        raise _fail("first-refinement investigation is not COMPLETE")
    metadata = investigation.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("first-refinement investigation metadata is invalid")

    expected_project_path = refinement_project_path.resolve()
    metadata_project_path = metadata.get("projectPath")
    if not isinstance(metadata_project_path, str):
        raise _fail("first-refinement investigation project path is missing")
    try:
        if Path(metadata_project_path).expanduser().resolve() != expected_project_path:
            raise _fail(
                "first-refinement investigation metadata refers to a different project"
            )
    except OSError as error:
        raise _fail("first-refinement investigation path cannot be resolved") from error

    stage_values = investigation.get("stages")
    if not isinstance(stage_values, list) or len(stage_values) != 3:
        raise _fail(
            "first-refinement project-smoke investigation must contain exactly three stages"
        )
    stages: list[Mapping[str, Any]] = []
    stage_ids: set[str] = set()
    for value in stage_values:
        if not isinstance(value, Mapping):
            raise _fail("first-refinement investigation stage is malformed")
        stage_id, _ = _validate_stage_shape(value)
        if stage_id in stage_ids:
            raise _fail("first-refinement investigation contains duplicate stage IDs")
        if value.get("artifacts") != []:
            raise _fail(
                "first-refinement investigation contains unexpected stage artifacts"
            )
        stage_ids.add(stage_id)
        stages.append(value)
    ledger_hashes = _stage_ledgers(investigation_path, stages)

    prepare_stages = [
        stage for stage in stages if stage.get("handler_id") == PREPARE_HANDLER_ID
    ]
    run_stages = [
        stage for stage in stages if stage.get("handler_id") == PROJECT_RUN_HANDLER_ID
    ]
    terminal_stages = [
        stage
        for stage in stages
        if stage.get("handler_id") == TERMINAL_CHECK_HANDLER_ID
    ]
    if len(prepare_stages) != 1 or len(run_stages) != 1 or len(terminal_stages) != 1:
        raise _fail("first-refinement project-smoke handler structure is invalid")
    prepare_stage = prepare_stages[0]
    run_stage = run_stages[0]
    terminal_stage = terminal_stages[0]
    if stages != [prepare_stage, run_stage, terminal_stage]:
        raise _fail("first-refinement project-smoke stages are out of canonical order")
    if prepare_stage.get("triggered_by_stage_id") is not None:
        raise _fail("first-refinement preparation stage has invalid causality")
    if run_stage.get("triggered_by_stage_id") != prepare_stage["id"]:
        raise _fail("first-refinement run stage has invalid causality")
    if terminal_stage.get("triggered_by_stage_id") != run_stage["id"]:
        raise _fail("first-refinement terminal stage has invalid causality")

    expected_project_hash = refinement.project_sha256
    expected_path_string = str(expected_project_path)
    prepare_parameters = prepare_stage["parameters"]
    if set(prepare_parameters) != {"projectPath"}:
        raise _fail("first-refinement preparation parameters are invalid")
    prepare_path = _nonempty_string(
        prepare_parameters.get("projectPath"),
        "first-refinement preparation path",
    )
    try:
        if Path(prepare_path).expanduser().resolve() != expected_project_path:
            raise _fail(
                "first-refinement preparation refers to a different project"
            )
    except OSError as error:
        raise _fail("first-refinement preparation path cannot be resolved") from error
    if prepare_stage["result"] != {
        "projectManifestSha256": expected_project_hash,
        "projectPath": expected_path_string,
    }:
        raise _fail("first-refinement preparation result does not match the project")
    prepare_provenance = prepare_stage["provenance"]
    if prepare_provenance.get("input_hashes") != {
        "projectManifest": expected_project_hash
    }:
        raise _fail("first-refinement preparation input provenance does not match")
    if prepare_provenance.get("project_ids") != []:
        raise _fail("first-refinement preparation has unexpected project IDs")
    if prepare_provenance.get("node_contributions") != {}:
        raise _fail("first-refinement preparation has unexpected node contributions")

    run_parameters = run_stage["parameters"]
    if set(run_parameters) != {"projectManifestSha256", "projectPath"}:
        raise _fail("first-refinement run parameters are invalid")
    if run_parameters.get("projectManifestSha256") != expected_project_hash:
        raise _fail("first-refinement run manifest hash does not match")
    run_path = run_parameters.get("projectPath")
    if not isinstance(run_path, str):
        raise _fail("first-refinement run path is missing")
    try:
        if Path(run_path).expanduser().resolve() != expected_project_path:
            raise _fail("first-refinement run refers to a different project")
    except OSError as error:
        raise _fail("first-refinement run path cannot be resolved") from error
    run_provenance = run_stage["provenance"]
    if run_provenance.get("input_hashes") != {
        "projectManifest": expected_project_hash
    }:
        raise _fail("first-refinement run input provenance does not match")
    if run_provenance.get("project_ids") != [refinement.project_id]:
        raise _fail("first-refinement run project ID does not match")
    if _safe_sum(
        list(run_provenance["node_contributions"].values()),
        "first-refinement node contribution count",
    ) != REFINEMENT_EXPECTED_WORK_UNIT_COUNT:
        raise _fail("first-refinement node contributions do not match completed work")

    expected_run_request = {
        "handler_id": PROJECT_RUN_HANDLER_ID,
        "id": run_stage["id"],
        "parameters": dict(run_parameters),
        "triggered_by_stage_id": prepare_stage["id"],
    }
    if prepare_stage.get("next_stage") != expected_run_request:
        raise _fail("first-refinement preparation continuation is invalid")
    expected_terminal_request = {
        "handler_id": TERMINAL_CHECK_HANDLER_ID,
        "id": terminal_stage["id"],
        "parameters": dict(terminal_stage["parameters"]),
        "triggered_by_stage_id": run_stage["id"],
    }
    if run_stage.get("next_stage") != expected_terminal_request:
        raise _fail("first-refinement run continuation is invalid")
    if prepare_stage.get("stop") is not False or run_stage.get("stop") is not False:
        raise _fail("first-refinement nonterminal stage is marked terminal")

    run_result = run_stage["result"]
    missing_run_fields = _REQUIRED_RUN_RESULT_FIELDS.difference(run_result)
    if missing_run_fields:
        raise _fail(
            "first-refinement run result is missing required fields: "
            f"{', '.join(sorted(missing_run_fields))}"
        )
    for field_name in (
        "projectAssignedWorkUnits",
        "projectCompletedWorkUnits",
        "projectFailedWorkUnits",
        "projectPendingWorkUnits",
        "projectTotalWorkUnits",
    ):
        _exact_count(run_result.get(field_name), field_name)
    required_run_values = {
        "projectAssignedWorkUnits": 0,
        "projectCompletedWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "projectFailedWorkUnits": 0,
        "projectPendingWorkUnits": 0,
        "projectTotalWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "projectID": refinement.project_id,
        "projectPath": expected_path_string,
        "status": "COMPLETE",
        "workloadID": WORKLOAD_ID,
    }
    if any(run_result.get(key) != value for key, value in required_run_values.items()):
        raise _fail("first-refinement project run did not complete with exact coverage")
    if run_result.get("nodeContributions") != run_provenance["node_contributions"]:
        raise _fail("first-refinement contribution provenance does not match")
    dataset_statuses = run_result.get("datasets")
    if not isinstance(dataset_statuses, list) or len(dataset_statuses) != 1:
        raise _fail("first-refinement run must report exactly one dataset")
    dataset_status = dataset_statuses[0]
    if not isinstance(dataset_status, Mapping):
        raise _fail("first-refinement dataset status is malformed")
    missing_status_fields = _REQUIRED_DATASET_STATUS_FIELDS.difference(
        dataset_status
    )
    if missing_status_fields:
        raise _fail(
            "first-refinement dataset status is missing required fields: "
            f"{', '.join(sorted(missing_status_fields))}"
        )
    for field_name in (
        "completedCandidateCount",
        "completedWorkUnits",
        "failedWorkUnits",
        "totalCandidateCount",
        "totalWorkUnits",
    ):
        _exact_count(dataset_status.get(field_name), field_name)
    if dataset_status.get("coverageComplete") is not True:
        raise _fail("first-refinement dataset coverage is incomplete or inconsistent")
    required_dataset_values = {
        "completedCandidateCount": REFINEMENT_TOTAL_CANDIDATE_COUNT,
        "completedWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "coverageComplete": True,
        "curveGridStatus": "CURVE_GRID_COMPLETE",
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "failedWorkUnits": 0,
        "familyID": FAMILY_ID,
        "id": refinement.dataset_id,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "totalCandidateCount": REFINEMENT_TOTAL_CANDIDATE_COUNT,
        "totalWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "workloadID": WORKLOAD_ID,
        "workloadStatus": "CURVE_GRID_COMPLETE",
    }
    if any(
        dataset_status.get(key) != value
        for key, value in required_dataset_values.items()
    ):
        raise _fail("first-refinement dataset coverage is incomplete or inconsistent")
    winner_values = _verify_winning_result(dataset_status, refinement.axes)

    expected_terminal_result = {
        "completedWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "failedWorkUnits": 0,
        "passed": True,
        "projectID": refinement.project_id,
        "rule": "projectID matches and completed+failed == total",
        "totalWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
    }
    if terminal_stage["result"] != expected_terminal_result:
        raise _fail("first-refinement terminal check did not pass exactly")
    for field_name in ("completedWorkUnits", "failedWorkUnits", "totalWorkUnits"):
        _exact_count(terminal_stage["result"].get(field_name), field_name)
    if terminal_stage["result"].get("passed") is not True:
        raise _fail("first-refinement terminal check did not pass exactly")
    if terminal_stage.get("stop") is not True or terminal_stage.get("next_stage") is not None:
        raise _fail("first-refinement terminal stage is not terminal")
    if terminal_stage["parameters"] != {"expectedProjectID": refinement.project_id}:
        raise _fail("first-refinement terminal expected project ID does not match")
    terminal_provenance = terminal_stage["provenance"]
    if terminal_provenance.get("project_ids") != [refinement.project_id]:
        raise _fail("first-refinement terminal project ID does not match")
    if terminal_provenance.get("input_hashes") != {}:
        raise _fail("first-refinement terminal has unexpected input provenance")
    if terminal_provenance.get("node_contributions") != {}:
        raise _fail("first-refinement terminal has unexpected node contributions")

    return _VerifiedRefinementWinner(
        grid_index=winner_values[0],
        center_index=winner_values[1],
        log_scale_index=winner_values[2],
        log_shape_index=winner_values[3],
        center=winner_values[4],
        log_scale=winner_values[5],
        log_shape=winner_values[6],
        offset=winner_values[7],
        amplitude=winner_values[8],
        objective=winner_values[9],
        boundary_axes=winner_values[10],
        result_payload=dict(dataset_status["payload"]["best"]),
        investigation_sha256=_sha256_bytes(record_bytes),
        run_stage_id=run_stage["id"],
        run_stage_ledger_sha256=ledger_hashes[run_stage["id"]],
        stage_ledger_sha256s=dict(sorted(ledger_hashes.items())),
    )


def _derived_axes(
    refinement: _VerifiedRefinementProject,
    winner: _VerifiedRefinementWinner,
) -> dict[str, dict[str, Any]]:
    center_step = refinement.axes["centerAxis"]["step"]
    log_scale_step = refinement.axes["logScaleAxis"]["step"]
    log_shape_step = refinement.axes["logShapeAxis"]["step"]
    axes = {
        "centerAxis": {
            "count": CENTER_COUNT,
            "start": winner.center - 10 * center_step,
            "step": center_step,
        },
        "logScaleAxis": {
            "count": LOG_SCALE_COUNT,
            "start": winner.log_scale - 8 * log_scale_step,
            "step": log_scale_step,
        },
        "logShapeAxis": {
            "count": LOG_SHAPE_COUNT,
            "start": winner.log_shape - 8 * log_shape_step,
            "step": log_shape_step,
        },
    }
    return axes


def _provenance_chain(
    coarse: _VerifiedCoarseProject,
    coarse_winner: _VerifiedCoarseWinner,
    refinement: _VerifiedRefinementProject,
    winner: _VerifiedRefinementWinner,
) -> dict[str, Any]:
    return {
        "coarse": {
            "buildManifestSHA256": coarse.build_manifest_sha256,
            "contractFileSHA256": coarse.contract_file_sha256,
            "contractID": COARSE_GRID_CONTRACT_ID,
            "contractSHA256": COARSE_GRID_CONTRACT_SHA256,
            "datasetSHA256": coarse.dataset_sha256,
            "investigationRecordSHA256": coarse_winner.investigation_sha256,
            "projectID": coarse.project_id,
            "projectSHA256": coarse.project_sha256,
            "runStageID": coarse_winner.run_stage_id,
            "runStageLedgerSHA256": coarse_winner.run_stage_ledger_sha256,
            "stageLedgerSHA256s": dict(coarse_winner.stage_ledger_sha256s),
        },
        "firstRefinement": {
            "buildManifestSHA256": refinement.build_manifest_sha256,
            "contractFileSHA256": refinement.contract_file_sha256,
            "contractID": REFINEMENT_GRID_CONTRACT_ID,
            "contractSHA256": refinement.contract_sha256,
            "datasetSHA256": refinement.dataset_sha256,
            "investigationRecordSHA256": winner.investigation_sha256,
            "projectID": refinement.project_id,
            "projectSHA256": refinement.project_sha256,
            "runStageID": winner.run_stage_id,
            "runStageLedgerSHA256": winner.run_stage_ledger_sha256,
            "stageLedgerSHA256s": dict(winner.stage_ledger_sha256s),
        },
        "preparationManifestSHA256": coarse.preparation_manifest_sha256,
    }


def _contract(
    coarse: _VerifiedCoarseProject,
    coarse_winner: _VerifiedCoarseWinner,
    refinement: _VerifiedRefinementProject,
    winner: _VerifiedRefinementWinner,
    axes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "axisProvenanceStatement": (
            "Axes were recentered only from the verified first-refinement grid "
            "and its accepted boundary winner, retaining every axis step."
        ),
        "benchmarkKind": "known-event-recovery",
        "boundaryTrigger": {
            "boundaryAxes": list(winner.boundary_axes),
            "required": "at least one first-refinement winner index is 0 or count - 1",
        },
        "candidateCount": TOTAL_CANDIDATE_COUNT,
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant whitespace, "
            "non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractID": RECENTERED_GRID_CONTRACT_ID,
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
                "start": "firstRefinementBestCenter - 10 * firstRefinementCenterStep",
                "step": "firstRefinementCenterStep",
                "winnerIndex": 10,
            },
            "logScale": {
                "count": LOG_SCALE_COUNT,
                "start": "firstRefinementBestLogScale - 8 * firstRefinementLogScaleStep",
                "step": "firstRefinementLogScaleStep",
                "winnerIndex": 8,
            },
            "logShape": {
                "count": LOG_SHAPE_COUNT,
                "start": "firstRefinementBestLogShape - 8 * firstRefinementLogShapeStep",
                "step": "firstRefinementLogShapeStep",
                "winnerIndex": 8,
            },
        },
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "modelScopeStatement": (
            "This remains smooth-event convergence with the symmetric "
            "radial-amplification family, not planetary-anomaly recovery, "
            "classification, or a discovery claim."
        ),
        "provenanceChain": _provenance_chain(
            coarse,
            coarse_winner,
            refinement,
            winner,
        ),
        "schemaTuple": {
            "datasetSchemaID": DATASET_SCHEMA_ID,
            "payloadSchemaID": PAYLOAD_SCHEMA_ID,
            "resultSchemaID": RESULT_SCHEMA_ID,
            "workloadID": WORKLOAD_ID,
        },
        "verifiedFirstRefinementWinner": {
            "acceptedResultPayload": dict(winner.result_payload),
            "bestAmplitude": winner.amplitude,
            "bestCenter": winner.center,
            "bestGridIndex": winner.grid_index,
            "bestLogScale": winner.log_scale,
            "bestLogShape": winner.log_shape,
            "bestOffset": winner.offset,
            "bestWeightedResidualSumSquares": winner.objective,
            "centerIndex": winner.center_index,
            "logScaleIndex": winner.log_scale_index,
            "logShapeIndex": winner.log_shape_index,
        },
    }


def _dataset(
    project_id: str,
    coarse: _VerifiedCoarseProject,
    refinement: _VerifiedRefinementProject,
    winner: _VerifiedRefinementWinner,
    axes: Mapping[str, Mapping[str, Any]],
    contract_sha256: str,
) -> dict[str, Any]:
    source = coarse.selected_series.payload
    dataset = {
        "blindTargetID": coarse.blind_target_id,
        "coordinates": list(source["coordinates"]),
        "curveGrid": {
            "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
            "centerAxis": dict(axes["centerAxis"]),
            "familyID": FAMILY_ID,
            "logScaleAxis": dict(axes["logScaleAxis"]),
            "logShapeAxis": dict(axes["logShapeAxis"]),
        },
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "firstRefinementProjectID": refinement.project_id,
        "firstRefinementRunStageLedgerSHA256": winner.run_stage_ledger_sha256,
        "id": f"{project_id}.primary-series",
        "inverseVariances": list(source["inverseVariances"]),
        "recenteredSearchContractID": RECENTERED_GRID_CONTRACT_ID,
        "recenteredSearchContractSHA256": contract_sha256,
        "sourceGenericSeriesID": coarse.selected_series.series_id,
        "values": list(source["values"]),
    }
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(f"constructed recentered CurveGrid dataset is invalid: {error}") from error
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


def build_recentered_grid_project(
    prepared_root: str | Path,
    *,
    coarse_project_root: str | Path,
    coarse_investigation_record: str | Path,
    refinement_project_root: str | Path,
    refinement_investigation_record: str | Path,
    project_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify both completed grids and publish one deterministic recentering."""

    if not isinstance(project_id, str) or _SAFE_PROJECT_ID.fullmatch(project_id) is None:
        raise _fail("project ID is malformed or unsafe")
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    _reject_symlink_components(output.parent, "output root")
    prepared = Path(prepared_root).expanduser().absolute()
    coarse_root = Path(coarse_project_root).expanduser().absolute()
    coarse_investigation = Path(coarse_investigation_record).expanduser().absolute()
    refinement_root = Path(refinement_project_root).expanduser().absolute()
    refinement_investigation = (
        Path(refinement_investigation_record).expanduser().absolute()
    )
    for path, description in (
        (prepared, "prepared root"),
        (coarse_root, "coarse project root"),
        (coarse_investigation, "coarse investigation record"),
        (refinement_root, "first-refinement project root"),
        (refinement_investigation, "first-refinement investigation record"),
    ):
        _reject_symlink_components(path, description)
    _regular_directory(prepared, "prepared root")

    try:
        _verify_exact_project_tree(
            coarse_root,
            COARSE_CONTRACT_RELATIVE_PATH,
            "coarse project",
        )
        coarse = _verify_coarse_project(prepared, coarse_root)
        _verify_investigation_tree(
            coarse_investigation,
            "coarse investigation",
        )
        coarse_winner = _verify_coarse_investigation(
            coarse_investigation,
            coarse,
            coarse_root / COARSE_PROJECT_RELATIVE_PATH,
        )
        _verify_prepare_parameter_path(
            coarse_investigation,
            (coarse_root / COARSE_PROJECT_RELATIVE_PATH).resolve(),
            "coarse investigation",
        )
        refinement = _verify_refinement_project(
            refinement_root,
            coarse,
            coarse_winner,
        )
        winner = _verify_refinement_investigation(
            refinement_investigation,
            refinement,
            refinement_root / REFINEMENT_PROJECT_RELATIVE_PATH,
        )
    except RefinementGridBuildError as error:
        raise _fail(str(error)) from error

    axes = _derived_axes(refinement, winner)
    contract = _contract(coarse, coarse_winner, refinement, winner, axes)
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    dataset = _dataset(
        project_id,
        coarse,
        refinement,
        winner,
        axes,
        contract_sha256,
    )
    project = _project(project_id, dataset["id"])

    evaluation_count = _safe_product(
        coarse.selected_series.sample_count,
        TOTAL_CANDIDATE_COUNT,
        "recentered sample-candidate evaluation count",
    )
    contract_bytes = _stable_json_bytes(contract)
    dataset_bytes = _stable_json_bytes(dataset)
    project_bytes = _stable_json_bytes(project)
    provenance = _provenance_chain(coarse, coarse_winner, refinement, winner)
    build_manifest = {
        "blindTargetID": coarse.blind_target_id,
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID,
        "coarseProvenance": provenance["coarse"],
        "expectedSampleCandidateEvaluationCount": evaluation_count,
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "firstRefinementProvenance": provenance["firstRefinement"],
        "inputSeriesSHA256": coarse.selected_series.sha256,
        "outputContractFileSHA256": _sha256_bytes(contract_bytes),
        "outputDatasetSHA256": _sha256_bytes(dataset_bytes),
        "outputProjectSHA256": _sha256_bytes(project_bytes),
        "preparationManifestSHA256": coarse.preparation_manifest_sha256,
        "projectID": project_id,
        "recenteredSearchContractID": RECENTERED_GRID_CONTRACT_ID,
        "recenteredSearchContractSHA256": contract_sha256,
        "relativeArtifactPaths": {
            "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
            "dataset": DATASET_RELATIVE_PATH,
            "project": PROJECT_RELATIVE_PATH,
            "recenteredSearchContract": CONTRACT_RELATIVE_PATH,
        },
        "selectedSampleCount": coarse.selected_series.sample_count,
        "selectedSeriesID": coarse.selected_series.series_id,
        "totalCandidateCount": TOTAL_CANDIDATE_COUNT,
        "verifiedFirstRefinementWinner": contract[
            "verifiedFirstRefinementWinner"
        ],
    }
    build_manifest_bytes = _stable_json_bytes(build_manifest)
    try:
        _assert_identity_free(
            (contract_bytes, dataset_bytes, project_bytes, build_manifest_bytes)
        )
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _fail("output parent cannot be created") from error
    _reject_symlink_components(output.parent, "output root")
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
        )
    except OSError as error:
        raise _fail("atomic output staging cannot be created") from error
    try:
        _atomic_write_bytes(staging / CONTRACT_RELATIVE_PATH, contract_bytes)
        _atomic_write_bytes(staging / DATASET_RELATIVE_PATH, dataset_bytes)
        _atomic_write_bytes(staging / PROJECT_RELATIVE_PATH, project_bytes)
        _atomic_write_bytes(
            staging / BUILD_MANIFEST_RELATIVE_PATH,
            build_manifest_bytes,
        )
        if output.exists() or output.is_symlink():
            raise _fail("output root already exists")
        staging.rename(output)
    except Exception as error:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if isinstance(error, RecenterGridBuildError):
            raise
        raise _fail("atomic output publication failed") from error
    return {
        "buildManifest": build_manifest,
        "contract": contract,
        "dataset": dataset,
        "project": project,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic CurveGrid recentering from a verified blind "
            "preparation, coarse run, and first-refinement boundary winner."
        )
    )
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--coarse-project-root", required=True, type=Path)
    parser.add_argument("--coarse-investigation-record", required=True, type=Path)
    parser.add_argument("--refinement-project-root", required=True, type=Path)
    parser.add_argument(
        "--refinement-investigation-record",
        required=True,
        type=Path,
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_recentered_grid_project(
        arguments.prepared_root,
        coarse_project_root=arguments.coarse_project_root,
        coarse_investigation_record=arguments.coarse_investigation_record,
        refinement_project_root=arguments.refinement_project_root,
        refinement_investigation_record=arguments.refinement_investigation_record,
        project_id=arguments.project_id,
        output_root=arguments.output_root,
    )
    manifest = result["buildManifest"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind recentered-grid project ready")
    print(f"project ID: {manifest['projectID']}")
    print(
        "first-refinement project ID: "
        f"{manifest['firstRefinementProvenance']['projectID']}"
    )
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
