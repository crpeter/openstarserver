"""Build a deterministic second-recentered CurveGrid from verified ancestry."""

from __future__ import annotations

import argparse
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
    _stable_json_bytes,
)
from workflows.microlensing.recenter_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
    BUILD_MANIFEST_SCHEMA_ID as FIRST_RECENTER_BUILD_MANIFEST_SCHEMA_ID,
    CANDIDATES_PER_WORK_UNIT as FIRST_RECENTER_CANDIDATES_PER_WORK_UNIT,
    CENTER_COUNT as FIRST_RECENTER_CENTER_COUNT,
    CONTRACT_RELATIVE_PATH as FIRST_RECENTER_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as FIRST_RECENTER_DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT as FIRST_RECENTER_EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_COUNT as FIRST_RECENTER_LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT as FIRST_RECENTER_LOG_SHAPE_COUNT,
    PROJECT_RELATIVE_PATH as FIRST_RECENTER_PROJECT_RELATIVE_PATH,
    RECENTERED_GRID_CONTRACT_ID,
    TOTAL_CANDIDATE_COUNT as FIRST_RECENTER_TOTAL_CANDIDATE_COUNT,
    RecenterGridBuildError as FirstRecenterGridBuildError,
    _INVESTIGATION_FIELDS,
    _REQUIRED_DATASET_STATUS_FIELDS,
    _REQUIRED_RUN_RESULT_FIELDS,
    _WINNING_RESULT_FIELDS,
    _VerifiedRefinementProject,
    _VerifiedRefinementWinner,
    _contract as _expected_first_recenter_contract,
    _dataset as _expected_first_recenter_dataset,
    _derived_axes as _expected_first_recenter_axes,
    _exact_count,
    _project as _expected_first_recenter_project,
    _provenance_chain as _expected_first_recenter_provenance,
    _read_json_file,
    _regular_directory as _first_recenter_regular_directory,
    _reject_symlink_components as _first_recenter_reject_symlink_components,
    _safe_product as _first_recenter_safe_product,
    _sha256_bytes,
    _verify_exact_project_tree,
    _verify_investigation_tree,
    _verify_prepare_parameter_path,
    _verify_refinement_investigation,
    _verify_refinement_project,
)
from workflows.microlensing.refine_grid import (
    PROJECT_RELATIVE_PATH as REFINEMENT_PROJECT_RELATIVE_PATH,
    RefinementGridBuildError,
    _VerifiedCoarseProject,
    _VerifiedWinner as _VerifiedCoarseWinner,
    _exact_integer,
    _finite_number,
    _nonempty_string,
    _safe_stage_id,
    _safe_sum,
    _stage_ledgers,
    _validate_stage_shape,
    _verify_coarse_project,
    _verify_investigation as _verify_coarse_investigation,
)


SECOND_RECENTER_GRID_CONTRACT_ID = (
    "openstar.microlensing-second-recentered-grid.v1"
)
SECOND_RECENTER_GRID_CONTRACT_VERSION = "1.0"
BUILD_MANIFEST_SCHEMA_ID = (
    "openstar.microlensing-second-recentered-grid-build.v1"
)
BUILD_MANIFEST_VERSION = "1.0"

SMOKE_WORKFLOW_ID = "openstar.workflow.project-smoke.v1"
PREPARE_HANDLER_ID = "local.project.prepare"
PROJECT_RUN_HANDLER_ID = "openstar.project.run"
TERMINAL_CHECK_HANDLER_ID = "generic.project.terminal-check"

CENTER_COUNT = FIRST_RECENTER_CENTER_COUNT
LOG_SCALE_COUNT = FIRST_RECENTER_LOG_SCALE_COUNT
LOG_SHAPE_COUNT = FIRST_RECENTER_LOG_SHAPE_COUNT
CANDIDATES_PER_WORK_UNIT = FIRST_RECENTER_CANDIDATES_PER_WORK_UNIT
TOTAL_CANDIDATE_COUNT = CENTER_COUNT * LOG_SCALE_COUNT * LOG_SHAPE_COUNT
EXPECTED_WORK_UNIT_COUNT = (
    TOTAL_CANDIDATE_COUNT + CANDIDATES_PER_WORK_UNIT - 1
) // CANDIDATES_PER_WORK_UNIT

CONTRACT_RELATIVE_PATH = "second-recentered-search-contract.json"
DATASET_RELATIVE_PATH = "datasets/primary-series.json"
PROJECT_RELATIVE_PATH = "project.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"

_SAFE_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FIRST_RECENTER_BUILD_MANIFEST_FIELDS = frozenset(
    (
        "blindTargetID",
        "buildManifestSchemaID",
        "coarseProvenance",
        "expectedSampleCandidateEvaluationCount",
        "expectedWorkUnitCount",
        "firstRefinementProvenance",
        "inputSeriesSHA256",
        "outputContractFileSHA256",
        "outputDatasetSHA256",
        "outputProjectSHA256",
        "preparationManifestSHA256",
        "projectID",
        "recenteredSearchContractID",
        "recenteredSearchContractSHA256",
        "relativeArtifactPaths",
        "selectedSampleCount",
        "selectedSeriesID",
        "totalCandidateCount",
        "verifiedFirstRefinementWinner",
    )
)


class SecondRecenterGridBuildError(RuntimeError):
    """The second-recenter project cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class _VerifiedFirstRecenterProject:
    project_id: str
    dataset_id: str
    axes: Mapping[str, Mapping[str, Any]]
    build_manifest_sha256: str
    contract_file_sha256: str
    contract_sha256: str
    dataset_sha256: str
    project_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedFirstRecenterWinner:
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
    investigation_id: str
    investigation_sha256: str
    run_stage_id: str
    run_stage_ledger_sha256: str
    stage_ledger_sha256s: Mapping[str, str]


def _fail(message: str) -> SecondRecenterGridBuildError:
    return SecondRecenterGridBuildError(message)


def _reject_symlink_components(path: Path, description: str) -> None:
    try:
        _first_recenter_reject_symlink_components(path, description)
    except FirstRecenterGridBuildError as error:
        raise _fail(str(error)) from error


def _regular_directory(path: Path, description: str) -> Path:
    try:
        return _first_recenter_regular_directory(path, description)
    except FirstRecenterGridBuildError as error:
        raise _fail(str(error)) from error


def _safe_product(left: int, right: int, field_name: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise _fail(f"{field_name} has invalid factors")
    if left and right > MAX_SAFE_INTEGER // left:
        raise _fail(f"{field_name} exceeds the safe integer range")
    return left * right


def _investigation_id(path: Path, label: str) -> str:
    _, investigation = _read_json_file(path, f"{label} record")
    investigation_id = _nonempty_string(
        investigation.get("id"),
        f"{label} ID",
    )
    if path.parent.name != investigation_id:
        raise _fail(f"{label} directory does not match its ID")
    return investigation_id


def _verify_first_recenter_project(
    root: Path,
    coarse: _VerifiedCoarseProject,
    coarse_winner: _VerifiedCoarseWinner,
    refinement: _VerifiedRefinementProject,
    refinement_winner: _VerifiedRefinementWinner,
) -> _VerifiedFirstRecenterProject:
    _verify_exact_project_tree(
        root,
        FIRST_RECENTER_CONTRACT_RELATIVE_PATH,
        "first-recenter project",
    )
    build_bytes, build = _read_json_file(
        root / FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
        "first-recenter build manifest",
    )
    if set(build) != _FIRST_RECENTER_BUILD_MANIFEST_FIELDS:
        raise _fail("first-recenter build manifest field set is invalid")
    if build.get("buildManifestSchemaID") != FIRST_RECENTER_BUILD_MANIFEST_SCHEMA_ID:
        raise _fail("first-recenter build manifest schema ID is invalid")
    if build.get("recenteredSearchContractID") != RECENTERED_GRID_CONTRACT_ID:
        raise _fail("first-recenter contract ID does not match")
    expected_paths = {
        "buildManifest": FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
        "dataset": FIRST_RECENTER_DATASET_RELATIVE_PATH,
        "project": FIRST_RECENTER_PROJECT_RELATIVE_PATH,
        "recenteredSearchContract": FIRST_RECENTER_CONTRACT_RELATIVE_PATH,
    }
    if build.get("relativeArtifactPaths") != expected_paths:
        raise _fail("first-recenter artifact paths are invalid")

    axes = _expected_first_recenter_axes(refinement, refinement_winner)
    expected_contract = _expected_first_recenter_contract(
        coarse,
        coarse_winner,
        refinement,
        refinement_winner,
        axes,
    )
    contract_bytes, contract = _read_json_file(
        root / FIRST_RECENTER_CONTRACT_RELATIVE_PATH,
        "first-recenter search contract",
    )
    if contract != expected_contract:
        raise _fail("first-recenter search contract content does not match")
    if contract_bytes != _stable_json_bytes(expected_contract):
        raise _fail("first-recenter search contract serialization is unstable")
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    contract_file_sha256 = _sha256_bytes(contract_bytes)
    if build.get("recenteredSearchContractSHA256") != contract_sha256:
        raise _fail("first-recenter contract canonical hash does not match")
    if build.get("outputContractFileSHA256") != contract_file_sha256:
        raise _fail("first-recenter contract file hash does not match")

    project_bytes, project = _read_json_file(
        root / FIRST_RECENTER_PROJECT_RELATIVE_PATH,
        "first-recenter project manifest",
    )
    project_id = _nonempty_string(project.get("id"), "first-recenter project ID")
    if _SAFE_PROJECT_ID.fullmatch(project_id) is None:
        raise _fail("first-recenter project ID is unsafe")
    if build.get("projectID") != project_id:
        raise _fail("first-recenter project ID does not match its build manifest")

    expected_dataset = _expected_first_recenter_dataset(
        project_id,
        coarse,
        refinement,
        refinement_winner,
        axes,
        contract_sha256,
    )
    dataset_bytes, dataset = _read_json_file(
        root / FIRST_RECENTER_DATASET_RELATIVE_PATH,
        "first-recenter dataset",
    )
    if dataset != expected_dataset:
        raise _fail("first-recenter dataset does not match verified ancestry")
    if dataset_bytes != _stable_json_bytes(expected_dataset):
        raise _fail("first-recenter dataset serialization is unstable")
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(f"first-recenter CurveGrid dataset is invalid: {error}") from error

    dataset_id = _nonempty_string(dataset.get("id"), "first-recenter dataset ID")
    expected_project = _expected_first_recenter_project(project_id, dataset_id)
    if project != expected_project:
        raise _fail("first-recenter project manifest does not match")
    if project_bytes != _stable_json_bytes(expected_project):
        raise _fail("first-recenter project serialization is unstable")

    dataset_sha256 = _sha256_bytes(dataset_bytes)
    project_sha256 = _sha256_bytes(project_bytes)
    evaluation_count = _first_recenter_safe_product(
        coarse.selected_series.sample_count,
        FIRST_RECENTER_TOTAL_CANDIDATE_COUNT,
        "first-recenter sample-candidate evaluation count",
    )
    provenance = _expected_first_recenter_provenance(
        coarse,
        coarse_winner,
        refinement,
        refinement_winner,
    )
    expected_build = {
        "blindTargetID": coarse.blind_target_id,
        "buildManifestSchemaID": FIRST_RECENTER_BUILD_MANIFEST_SCHEMA_ID,
        "coarseProvenance": provenance["coarse"],
        "expectedSampleCandidateEvaluationCount": evaluation_count,
        "expectedWorkUnitCount": FIRST_RECENTER_EXPECTED_WORK_UNIT_COUNT,
        "firstRefinementProvenance": provenance["firstRefinement"],
        "inputSeriesSHA256": coarse.selected_series.sha256,
        "outputContractFileSHA256": contract_file_sha256,
        "outputDatasetSHA256": dataset_sha256,
        "outputProjectSHA256": project_sha256,
        "preparationManifestSHA256": coarse.preparation_manifest_sha256,
        "projectID": project_id,
        "recenteredSearchContractID": RECENTERED_GRID_CONTRACT_ID,
        "recenteredSearchContractSHA256": contract_sha256,
        "relativeArtifactPaths": expected_paths,
        "selectedSampleCount": coarse.selected_series.sample_count,
        "selectedSeriesID": coarse.selected_series.series_id,
        "totalCandidateCount": FIRST_RECENTER_TOTAL_CANDIDATE_COUNT,
        "verifiedFirstRefinementWinner": expected_contract[
            "verifiedFirstRefinementWinner"
        ],
    }
    if build != expected_build:
        raise _fail("first-recenter build manifest provenance is incomplete")
    if build_bytes != _stable_json_bytes(expected_build):
        raise _fail("first-recenter build manifest serialization is unstable")

    return _VerifiedFirstRecenterProject(
        project_id=project_id,
        dataset_id=dataset_id,
        axes={key: dict(value) for key, value in axes.items()},
        build_manifest_sha256=_sha256_bytes(build_bytes),
        contract_file_sha256=contract_file_sha256,
        contract_sha256=contract_sha256,
        dataset_sha256=dataset_sha256,
        project_sha256=project_sha256,
    )


def _grid_indices(grid_index: int) -> tuple[int, int, int]:
    if grid_index >= TOTAL_CANDIDATE_COUNT:
        raise _fail("first-recenter best grid index is outside the grid")
    combined, shape_index = divmod(grid_index, LOG_SHAPE_COUNT)
    center_index, scale_index = divmod(combined, LOG_SCALE_COUNT)
    return center_index, scale_index, shape_index


def _verify_winning_result(
    dataset_status: Mapping[str, Any],
    axes: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int, int, int, float, float, float, float, float, float, tuple[str, ...]]:
    payload = dataset_status.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"best"}:
        raise _fail("first-recenter dataset result payload is invalid")
    best = payload.get("best")
    if not isinstance(best, Mapping) or set(best) != _WINNING_RESULT_FIELDS:
        raise _fail("first-recenter winning result field set is invalid")
    if best.get("familyID") != FAMILY_ID:
        raise _fail("first-recenter winning result family is invalid")

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
        raise _fail("first-recenter winning objective must be nonnegative")

    center_index, scale_index, shape_index = _grid_indices(grid_index)
    expected_values = (
        axes["centerAxis"]["start"]
        + center_index * axes["centerAxis"]["step"],
        axes["logScaleAxis"]["start"]
        + scale_index * axes["logScaleAxis"]["step"],
        axes["logShapeAxis"]["start"]
        + shape_index * axes["logShapeAxis"]["step"],
    )
    if (center, log_scale, log_shape) != expected_values:
        raise _fail("first-recenter winner is not present in its published grid")

    shard_start = (grid_index // CANDIDATES_PER_WORK_UNIT) * (
        CANDIDATES_PER_WORK_UNIT
    )
    shard_count = min(
        CANDIDATES_PER_WORK_UNIT,
        TOTAL_CANDIDATE_COUNT - shard_start,
    )
    if (
        _exact_count(best.get("gridStartIndex"), "gridStartIndex")
        != shard_start
        or _exact_count(best.get("gridCount"), "gridCount") != shard_count
    ):
        raise _fail("first-recenter winning shard identity is invalid")
    if (
        _exact_count(best.get("evaluatedCandidateCount"), "evaluatedCandidateCount")
        != shard_count
    ):
        raise _fail("first-recenter winning evaluated count is invalid")
    if _exact_count(best.get("invalidCandidateCount"), "invalidCandidateCount") >= shard_count:
        raise _fail("first-recenter winning invalid count is inconsistent")

    status_grid_index = _exact_integer(
        dataset_status.get("bestGridIndex"),
        "dataset status bestGridIndex",
    )
    status_numbers = tuple(
        _finite_number(
            dataset_status.get(field_name),
            f"dataset status {field_name}",
        )
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
        raise _fail("first-recenter aggregate and nested winner disagree")

    boundary_axes = tuple(
        name
        for name, index, count in (
            ("center", center_index, CENTER_COUNT),
            ("logScale", scale_index, LOG_SCALE_COUNT),
            ("logShape", shape_index, LOG_SHAPE_COUNT),
        )
        if index in {0, count - 1}
    )
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


def _verify_first_recenter_investigation(
    path: Path,
    project: _VerifiedFirstRecenterProject,
    project_path: Path,
) -> _VerifiedFirstRecenterWinner:
    label = "first-recenter investigation"
    if path.name != "investigation.json":
        raise _fail(f"{label} record must be named investigation.json")
    _verify_investigation_tree(path, label)
    record_bytes, investigation = _read_json_file(path, f"{label} record")
    if set(investigation) != _INVESTIGATION_FIELDS:
        raise _fail(f"{label} field set is invalid")
    investigation_id = _nonempty_string(investigation.get("id"), f"{label} ID")
    if path.parent.name != investigation_id:
        raise _fail(f"{label} directory does not match its ID")
    _safe_stage_id(investigation_id)
    if investigation.get("workflow_id") != SMOKE_WORKFLOW_ID:
        raise _fail(f"{label} workflow is invalid")
    _nonempty_string(investigation.get("workflow_version"), "workflow version")
    if investigation.get("status") != "COMPLETE":
        raise _fail(f"{label} is not COMPLETE")
    metadata = investigation.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail(f"{label} metadata is invalid")

    expected_project_path = project_path.resolve()
    metadata_path = metadata.get("projectPath")
    if not isinstance(metadata_path, str):
        raise _fail(f"{label} project path is missing")
    try:
        if Path(metadata_path).expanduser().resolve() != expected_project_path:
            raise _fail(f"{label} metadata refers to a different project")
    except OSError as error:
        raise _fail(f"{label} path cannot be resolved") from error

    stage_values = investigation.get("stages")
    if not isinstance(stage_values, list) or len(stage_values) != 3:
        raise _fail(f"{label} must contain exactly three stages")
    stages: list[Mapping[str, Any]] = []
    stage_ids: set[str] = set()
    for value in stage_values:
        if not isinstance(value, Mapping):
            raise _fail(f"{label} stage is malformed")
        stage_id, _ = _validate_stage_shape(value)
        if stage_id in stage_ids:
            raise _fail(f"{label} contains duplicate stage IDs")
        if value.get("artifacts") != []:
            raise _fail(f"{label} contains unexpected stage artifacts")
        stage_ids.add(stage_id)
        stages.append(value)
    ledger_hashes = _stage_ledgers(path, stages)

    prepare_stages = [
        stage for stage in stages if stage.get("handler_id") == PREPARE_HANDLER_ID
    ]
    run_stages = [
        stage for stage in stages if stage.get("handler_id") == PROJECT_RUN_HANDLER_ID
    ]
    terminal_stages = [
        stage for stage in stages if stage.get("handler_id") == TERMINAL_CHECK_HANDLER_ID
    ]
    if len(prepare_stages) != 1 or len(run_stages) != 1 or len(terminal_stages) != 1:
        raise _fail(f"{label} handler structure is invalid")
    prepare_stage = prepare_stages[0]
    run_stage = run_stages[0]
    terminal_stage = terminal_stages[0]
    if stages != [prepare_stage, run_stage, terminal_stage]:
        raise _fail(f"{label} stages are out of canonical order")
    if prepare_stage.get("triggered_by_stage_id") is not None:
        raise _fail(f"{label} preparation causality is invalid")
    if run_stage.get("triggered_by_stage_id") != prepare_stage["id"]:
        raise _fail(f"{label} run causality is invalid")
    if terminal_stage.get("triggered_by_stage_id") != run_stage["id"]:
        raise _fail(f"{label} terminal causality is invalid")

    expected_project_hash = project.project_sha256
    expected_path_string = str(expected_project_path)
    prepare_parameters = prepare_stage["parameters"]
    if set(prepare_parameters) != {"projectPath"}:
        raise _fail(f"{label} preparation parameters are invalid")
    prepare_path = _nonempty_string(
        prepare_parameters.get("projectPath"),
        f"{label} preparation path",
    )
    try:
        if Path(prepare_path).expanduser().resolve() != expected_project_path:
            raise _fail(f"{label} preparation refers to a different project")
    except OSError as error:
        raise _fail(f"{label} preparation path cannot be resolved") from error
    if prepare_stage["result"] != {
        "projectManifestSha256": expected_project_hash,
        "projectPath": expected_path_string,
    }:
        raise _fail(f"{label} preparation result does not match the project")
    prepare_provenance = prepare_stage["provenance"]
    if prepare_provenance.get("input_hashes") != {
        "projectManifest": expected_project_hash
    }:
        raise _fail(f"{label} preparation input provenance does not match")
    if prepare_provenance.get("project_ids") != []:
        raise _fail(f"{label} preparation has unexpected project IDs")
    if prepare_provenance.get("node_contributions") != {}:
        raise _fail(f"{label} preparation has unexpected node contributions")

    run_parameters = run_stage["parameters"]
    if set(run_parameters) != {"projectManifestSha256", "projectPath"}:
        raise _fail(f"{label} run parameters are invalid")
    if run_parameters.get("projectManifestSha256") != expected_project_hash:
        raise _fail(f"{label} run manifest hash does not match")
    run_path = run_parameters.get("projectPath")
    if not isinstance(run_path, str):
        raise _fail(f"{label} run path is missing")
    try:
        if Path(run_path).expanduser().resolve() != expected_project_path:
            raise _fail(f"{label} run refers to a different project")
    except OSError as error:
        raise _fail(f"{label} run path cannot be resolved") from error
    run_provenance = run_stage["provenance"]
    if run_provenance.get("input_hashes") != {
        "projectManifest": expected_project_hash
    }:
        raise _fail(f"{label} run input provenance does not match")
    if run_provenance.get("project_ids") != [project.project_id]:
        raise _fail(f"{label} run project ID does not match")
    if _safe_sum(
        list(run_provenance["node_contributions"].values()),
        "first-recenter node contribution count",
    ) != EXPECTED_WORK_UNIT_COUNT:
        raise _fail(f"{label} node contributions do not match completed work")

    expected_run_request = {
        "handler_id": PROJECT_RUN_HANDLER_ID,
        "id": run_stage["id"],
        "parameters": dict(run_parameters),
        "triggered_by_stage_id": prepare_stage["id"],
    }
    if prepare_stage.get("next_stage") != expected_run_request:
        raise _fail(f"{label} preparation continuation is invalid")
    expected_terminal_request = {
        "handler_id": TERMINAL_CHECK_HANDLER_ID,
        "id": terminal_stage["id"],
        "parameters": dict(terminal_stage["parameters"]),
        "triggered_by_stage_id": run_stage["id"],
    }
    if run_stage.get("next_stage") != expected_terminal_request:
        raise _fail(f"{label} run continuation is invalid")
    if prepare_stage.get("stop") is not False or run_stage.get("stop") is not False:
        raise _fail(f"{label} nonterminal stage is marked terminal")

    run_result = run_stage["result"]
    missing_run_fields = _REQUIRED_RUN_RESULT_FIELDS.difference(run_result)
    if missing_run_fields:
        raise _fail(
            f"{label} run result is missing required fields: "
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
        "projectCompletedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "projectFailedWorkUnits": 0,
        "projectID": project.project_id,
        "projectPath": expected_path_string,
        "projectPendingWorkUnits": 0,
        "projectTotalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "status": "COMPLETE",
        "workloadID": WORKLOAD_ID,
    }
    if any(run_result.get(key) != value for key, value in required_run_values.items()):
        raise _fail(f"{label} project run did not complete with exact coverage")
    if run_result.get("nodeContributions") != run_provenance["node_contributions"]:
        raise _fail(f"{label} contribution provenance does not match")

    dataset_statuses = run_result.get("datasets")
    if not isinstance(dataset_statuses, list) or len(dataset_statuses) != 1:
        raise _fail(f"{label} must report exactly one dataset")
    dataset_status = dataset_statuses[0]
    if not isinstance(dataset_status, Mapping):
        raise _fail(f"{label} dataset status is malformed")
    missing_status_fields = _REQUIRED_DATASET_STATUS_FIELDS.difference(dataset_status)
    if missing_status_fields:
        raise _fail(
            f"{label} dataset status is missing required fields: "
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
    required_dataset_values = {
        "completedCandidateCount": TOTAL_CANDIDATE_COUNT,
        "completedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "coverageComplete": True,
        "curveGridStatus": "CURVE_GRID_COMPLETE",
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "failedWorkUnits": 0,
        "familyID": FAMILY_ID,
        "id": project.dataset_id,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "totalCandidateCount": TOTAL_CANDIDATE_COUNT,
        "totalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "workloadID": WORKLOAD_ID,
        "workloadStatus": "CURVE_GRID_COMPLETE",
    }
    if any(
        dataset_status.get(key) != value
        for key, value in required_dataset_values.items()
    ):
        raise _fail(f"{label} dataset coverage is incomplete or inconsistent")
    winner_values = _verify_winning_result(dataset_status, project.axes)

    expected_terminal_result = {
        "completedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "failedWorkUnits": 0,
        "passed": True,
        "projectID": project.project_id,
        "rule": "projectID matches and completed+failed == total",
        "totalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
    }
    if terminal_stage["result"] != expected_terminal_result:
        raise _fail(f"{label} terminal check did not pass exactly")
    if terminal_stage.get("stop") is not True or terminal_stage.get("next_stage") is not None:
        raise _fail(f"{label} terminal stage is not terminal")
    if terminal_stage["parameters"] != {"expectedProjectID": project.project_id}:
        raise _fail(f"{label} terminal expected project ID does not match")
    terminal_provenance = terminal_stage["provenance"]
    if terminal_provenance.get("project_ids") != [project.project_id]:
        raise _fail(f"{label} terminal project ID does not match")
    if terminal_provenance.get("input_hashes") != {}:
        raise _fail(f"{label} terminal has unexpected input provenance")
    if terminal_provenance.get("node_contributions") != {}:
        raise _fail(f"{label} terminal has unexpected node contributions")

    return _VerifiedFirstRecenterWinner(
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
        investigation_id=investigation_id,
        investigation_sha256=_sha256_bytes(record_bytes),
        run_stage_id=run_stage["id"],
        run_stage_ledger_sha256=ledger_hashes[run_stage["id"]],
        stage_ledger_sha256s=dict(sorted(ledger_hashes.items())),
    )


def _axis_start(winner: float, count: int, step: float, name: str) -> float:
    if type(count) is not int or count <= 0 or count % 2 != 1:
        raise _fail(f"{name} count must be a positive odd integer")
    start = winner - ((count - 1) // 2) * step
    try:
        return _finite_number(start, f"derived {name} start")
    except RefinementGridBuildError as error:
        raise _fail(str(error)) from error


def _derived_axes(
    parent: _VerifiedFirstRecenterProject,
    winner: _VerifiedFirstRecenterWinner,
) -> dict[str, dict[str, Any]]:
    axes = {
        "centerAxis": {
            "count": CENTER_COUNT,
            "start": _axis_start(
                winner.center,
                CENTER_COUNT,
                parent.axes["centerAxis"]["step"],
                "center axis",
            ),
            "step": parent.axes["centerAxis"]["step"],
        },
        "logScaleAxis": {
            "count": LOG_SCALE_COUNT,
            "start": _axis_start(
                winner.log_scale,
                LOG_SCALE_COUNT,
                parent.axes["logScaleAxis"]["step"],
                "log-scale axis",
            ),
            "step": parent.axes["logScaleAxis"]["step"],
        },
        "logShapeAxis": {
            "count": LOG_SHAPE_COUNT,
            "start": _axis_start(
                winner.log_shape,
                LOG_SHAPE_COUNT,
                parent.axes["logShapeAxis"]["step"],
                "log-shape axis",
            ),
            "step": parent.axes["logShapeAxis"]["step"],
        },
    }
    return axes


def _winner_record(winner: _VerifiedFirstRecenterWinner) -> dict[str, Any]:
    return {
        "acceptedResultPayload": dict(winner.result_payload),
        "bestAmplitude": winner.amplitude,
        "bestCenter": winner.center,
        "bestGridIndex": winner.grid_index,
        "bestLogScale": winner.log_scale,
        "bestLogShape": winner.log_shape,
        "bestOffset": winner.offset,
        "bestWeightedResidualSumSquares": winner.objective,
        "boundaryAxes": list(winner.boundary_axes),
        "centerIndex": winner.center_index,
        "logScaleIndex": winner.log_scale_index,
        "logShapeIndex": winner.log_shape_index,
    }


def _parent_hashes(
    coarse: _VerifiedCoarseProject,
    coarse_winner: _VerifiedCoarseWinner,
    refinement: _VerifiedRefinementProject,
    refinement_winner: _VerifiedRefinementWinner,
    first_recenter: _VerifiedFirstRecenterProject,
    winner: _VerifiedFirstRecenterWinner,
) -> dict[str, Any]:
    return {
        "preparationManifestSHA256": coarse.preparation_manifest_sha256,
        "coarse": {
            "buildManifestSHA256": coarse.build_manifest_sha256,
            "contractFileSHA256": coarse.contract_file_sha256,
            "contractID": COARSE_GRID_CONTRACT_ID,
            "contractSHA256": COARSE_GRID_CONTRACT_SHA256,
            "datasetSHA256": coarse.dataset_sha256,
            "investigationRecordSHA256": coarse_winner.investigation_sha256,
            "projectSHA256": coarse.project_sha256,
            "runStageLedgerSHA256": coarse_winner.run_stage_ledger_sha256,
            "stageLedgerSHA256s": dict(coarse_winner.stage_ledger_sha256s),
        },
        "firstRefinement": {
            "buildManifestSHA256": refinement.build_manifest_sha256,
            "contractFileSHA256": refinement.contract_file_sha256,
            "contractSHA256": refinement.contract_sha256,
            "datasetSHA256": refinement.dataset_sha256,
            "investigationRecordSHA256": refinement_winner.investigation_sha256,
            "projectSHA256": refinement.project_sha256,
            "runStageLedgerSHA256": refinement_winner.run_stage_ledger_sha256,
            "stageLedgerSHA256s": dict(refinement_winner.stage_ledger_sha256s),
        },
        "firstRecenter": {
            "buildManifestSHA256": first_recenter.build_manifest_sha256,
            "contractFileSHA256": first_recenter.contract_file_sha256,
            "contractSHA256": first_recenter.contract_sha256,
            "datasetSHA256": first_recenter.dataset_sha256,
            "investigationRecordSHA256": winner.investigation_sha256,
            "projectSHA256": first_recenter.project_sha256,
            "runStageLedgerSHA256": winner.run_stage_ledger_sha256,
            "stageLedgerSHA256s": dict(winner.stage_ledger_sha256s),
        },
    }


def _contract(
    coarse: _VerifiedCoarseProject,
    coarse_winner: _VerifiedCoarseWinner,
    coarse_investigation_id: str,
    refinement: _VerifiedRefinementProject,
    refinement_winner: _VerifiedRefinementWinner,
    refinement_investigation_id: str,
    first_recenter: _VerifiedFirstRecenterProject,
    winner: _VerifiedFirstRecenterWinner,
    axes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "axisDerivationRule": (
            "newStart = acceptedFirstRecenterWinner - "
            "((count - 1) / 2) * retainedParentStep"
        ),
        "benchmarkKind": "known-event-recovery",
        "candidateCount": TOTAL_CANDIDATE_COUNT,
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant whitespace, "
            "non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractSchemaID": SECOND_RECENTER_GRID_CONTRACT_ID,
        "contractVersion": SECOND_RECENTER_GRID_CONTRACT_VERSION,
        "curveGrid": {
            "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
            "centerAxis": dict(axes["centerAxis"]),
            "familyID": FAMILY_ID,
            "logScaleAxis": dict(axes["logScaleAxis"]),
            "logShapeAxis": dict(axes["logShapeAxis"]),
        },
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "identityIsolationStatement": (
            "Sealed identity, source filenames, event names, catalog identifiers, "
            "publications, sky coordinates, and published physical parameters "
            "were not consulted."
        ),
        "modelScopeStatement": (
            "This remains smooth-event convergence with the symmetric "
            "radial-amplification family, not planetary-anomaly recovery, "
            "classification, or a discovery claim."
        ),
        "parentArtifactHashes": _parent_hashes(
            coarse,
            coarse_winner,
            refinement,
            refinement_winner,
            first_recenter,
            winner,
        ),
        "parentAxes": {
            key: dict(value) for key, value in first_recenter.axes.items()
        },
        "parentInvestigationIDs": {
            "coarse": coarse_investigation_id,
            "firstRecenter": winner.investigation_id,
            "firstRefinement": refinement_investigation_id,
        },
        "parentProjectIDs": {
            "coarse": coarse.project_id,
            "firstRecenter": first_recenter.project_id,
            "firstRefinement": refinement.project_id,
        },
        "schemaTuple": {
            "datasetSchemaID": DATASET_SCHEMA_ID,
            "payloadSchemaID": PAYLOAD_SCHEMA_ID,
            "resultSchemaID": RESULT_SCHEMA_ID,
            "workloadID": WORKLOAD_ID,
        },
        "verifiedFirstRecenterWinner": _winner_record(winner),
    }


def _dataset(
    project_id: str,
    coarse: _VerifiedCoarseProject,
    first_recenter: _VerifiedFirstRecenterProject,
    winner: _VerifiedFirstRecenterWinner,
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
        "firstRecenterProjectID": first_recenter.project_id,
        "firstRecenterRunStageLedgerSHA256": winner.run_stage_ledger_sha256,
        "id": f"{project_id}.primary-series",
        "inverseVariances": list(source["inverseVariances"]),
        "secondRecenterSearchContractID": SECOND_RECENTER_GRID_CONTRACT_ID,
        "secondRecenterSearchContractSHA256": contract_sha256,
        "sourceGenericSeriesID": coarse.selected_series.series_id,
        "values": list(source["values"]),
    }
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(
            f"constructed second-recenter CurveGrid dataset is invalid: {error}"
        ) from error
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


def build_second_recenter_grid_project(
    prepared_root: str | Path,
    *,
    coarse_project_root: str | Path,
    coarse_investigation_record: str | Path,
    refinement_project_root: str | Path,
    refinement_investigation_record: str | Path,
    first_recenter_project_root: str | Path,
    first_recenter_investigation_record: str | Path,
    project_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify the complete chain and publish one deterministic second recenter."""

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
    refinement_investigation = Path(
        refinement_investigation_record
    ).expanduser().absolute()
    first_recenter_root = Path(first_recenter_project_root).expanduser().absolute()
    first_recenter_investigation = Path(
        first_recenter_investigation_record
    ).expanduser().absolute()
    for path, description in (
        (prepared, "prepared root"),
        (coarse_root, "coarse project root"),
        (coarse_investigation, "coarse investigation record"),
        (refinement_root, "first-refinement project root"),
        (refinement_investigation, "first-refinement investigation record"),
        (first_recenter_root, "first-recenter project root"),
        (first_recenter_investigation, "first-recenter investigation record"),
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
        _verify_investigation_tree(coarse_investigation, "coarse investigation")
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
        coarse_investigation_id = _investigation_id(
            coarse_investigation,
            "coarse investigation",
        )

        refinement = _verify_refinement_project(
            refinement_root,
            coarse,
            coarse_winner,
        )
        refinement_winner = _verify_refinement_investigation(
            refinement_investigation,
            refinement,
            refinement_root / REFINEMENT_PROJECT_RELATIVE_PATH,
        )
        refinement_investigation_id = _investigation_id(
            refinement_investigation,
            "first-refinement investigation",
        )

        first_recenter = _verify_first_recenter_project(
            first_recenter_root,
            coarse,
            coarse_winner,
            refinement,
            refinement_winner,
        )
        winner = _verify_first_recenter_investigation(
            first_recenter_investigation,
            first_recenter,
            first_recenter_root / FIRST_RECENTER_PROJECT_RELATIVE_PATH,
        )
    except (
        CoarseGridBuildError,
        RefinementGridBuildError,
        FirstRecenterGridBuildError,
    ) as error:
        raise _fail(str(error)) from error

    axes = _derived_axes(first_recenter, winner)
    contract = _contract(
        coarse,
        coarse_winner,
        coarse_investigation_id,
        refinement,
        refinement_winner,
        refinement_investigation_id,
        first_recenter,
        winner,
        axes,
    )
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    dataset = _dataset(
        project_id,
        coarse,
        first_recenter,
        winner,
        axes,
        contract_sha256,
    )
    project = _project(project_id, dataset["id"])

    evaluation_count = _safe_product(
        coarse.selected_series.sample_count,
        TOTAL_CANDIDATE_COUNT,
        "second-recenter sample-candidate evaluation count",
    )
    contract_bytes = _stable_json_bytes(contract)
    dataset_bytes = _stable_json_bytes(dataset)
    project_bytes = _stable_json_bytes(project)
    build_manifest = {
        "acceptedFirstRecenterWinner": _winner_record(winner),
        "blindnessStatement": contract["identityIsolationStatement"],
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID,
        "buildManifestVersion": BUILD_MANIFEST_VERSION,
        "candidateCount": TOTAL_CANDIDATE_COUNT,
        "contractSchemaID": SECOND_RECENTER_GRID_CONTRACT_ID,
        "contractVersion": SECOND_RECENTER_GRID_CONTRACT_VERSION,
        "derivedSecondRecenterAxes": {
            key: dict(value) for key, value in axes.items()
        },
        "expectedSampleCandidateEvaluationCount": evaluation_count,
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "inputSeriesSHA256": coarse.selected_series.sha256,
        "outputContractFileSHA256": _sha256_bytes(contract_bytes),
        "outputDatasetSHA256": _sha256_bytes(dataset_bytes),
        "outputProjectSHA256": _sha256_bytes(project_bytes),
        "parentArtifactHashes": contract["parentArtifactHashes"],
        "parentAxes": contract["parentAxes"],
        "parentInvestigationIDs": contract["parentInvestigationIDs"],
        "parentProjectIDs": contract["parentProjectIDs"],
        "projectID": project_id,
        "relativeArtifactPaths": {
            "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
            "dataset": DATASET_RELATIVE_PATH,
            "project": PROJECT_RELATIVE_PATH,
            "secondRecenterSearchContract": CONTRACT_RELATIVE_PATH,
        },
        "secondRecenterSearchContractSHA256": contract_sha256,
        "selectedSampleCount": coarse.selected_series.sample_count,
        "selectedSeriesID": coarse.selected_series.series_id,
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
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
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
        if isinstance(error, SecondRecenterGridBuildError):
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
            "Build a deterministic second CurveGrid recentering from the "
            "complete verified blind project chain."
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
    parser.add_argument("--first-recenter-project-root", required=True, type=Path)
    parser.add_argument(
        "--first-recenter-investigation-record",
        required=True,
        type=Path,
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_second_recenter_grid_project(
        arguments.prepared_root,
        coarse_project_root=arguments.coarse_project_root,
        coarse_investigation_record=arguments.coarse_investigation_record,
        refinement_project_root=arguments.refinement_project_root,
        refinement_investigation_record=arguments.refinement_investigation_record,
        first_recenter_project_root=arguments.first_recenter_project_root,
        first_recenter_investigation_record=(
            arguments.first_recenter_investigation_record
        ),
        project_id=arguments.project_id,
        output_root=arguments.output_root,
    )
    manifest = result["buildManifest"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind second-recentered-grid project ready")
    print(f"project ID: {manifest['projectID']}")
    print(
        "first-recenter project ID: "
        f"{manifest['parentProjectIDs']['firstRecenter']}"
    )
    print(f"selected generic series: {manifest['selectedSeriesID']}")
    print(f"selected samples: {manifest['selectedSampleCount']}")
    print(f"grid candidates: {manifest['candidateCount']}")
    print(f"expected work units: {manifest['expectedWorkUnitCount']}")
    print(f"project: {output / PROJECT_RELATIVE_PATH}")
    print(f"dataset: {output / DATASET_RELATIVE_PATH}")
    print(f"build manifest: {output / BUILD_MANIFEST_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
