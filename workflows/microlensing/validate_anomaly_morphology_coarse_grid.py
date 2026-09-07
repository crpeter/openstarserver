"""Verify saved bounded morphology winners and report blind interpretation limits.

Only accepted candidates are reproduced; no grid/shard search or replacement
winner selection is performed. Observational support is an interpretation gate,
not an additional MorphologyGrid workload validity rule.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openstar_workloads.plugins import morphology_grid as workload
from openstar_workloads.plugins.morphology_grid import (
    COMPONENT_TEMPLATE_FAMILY_ID, DATASET_SCHEMA_ID, INDEPENDENT_PULSES,
    MORPHOLOGY_FAMILY_ID, ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    PAYLOAD_SCHEMA_ID, POSITIVE_PULSE_ONLY, RESULT_SCHEMA_ID, WORKLOAD_ID,
)
from workflows.microlensing import build_anomaly_morphology_coarse_grid as builder
from workflows.microlensing import prepare_anomaly_morphology as preparation
from workflows.microlensing.build_anomaly_morphology_coarse_grid import (
    _canonical_compact_json_bytes, _exact_count, _finite_number,
    _nonempty_string, _regular_directory, _reject_symlink_components,
    _safe_sum, _sha256_bytes, _stable_json_bytes,
)
from workflows.microlensing.coarse_grid import CoarseGridBuildError, _assert_identity_free, _atomic_write_bytes
from workflows.microlensing.refine_grid import (
    PREPARE_HANDLER_ID, PROJECT_RUN_HANDLER_ID, SMOKE_WORKFLOW_ID,
    TERMINAL_CHECK_HANDLER_ID, RefinementGridBuildError,
    _INVESTIGATION_FIELDS, _stage_ledgers, _validate_stage_shape,
)
from workflows.microlensing.validate_residual_grid import _verify_run_counter_scopes

RESULT_SCHEMA_ID_REPORT = "openstar.microlensing-anomaly-morphology-coarse-validation.v1"
MANIFEST_SCHEMA_ID = "openstar.microlensing-anomaly-morphology-coarse-validation-artifacts.v1"
RESULT_VERSION = "1.0"
RESULT_RELATIVE_PATH = "anomaly-morphology-coarse-validation.json"
MANIFEST_RELATIVE_PATH = "artifact-manifest.json"
UNRESOLVED_SUPPORT = "UNRESOLVED_OBSERVATIONAL_SUPPORT"
SUPPORT_FOLLOW_UP = "SUPPORT_AWARE_BLIND_ANOMALY_MORPHOLOGY_SEARCH"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class AnomalyMorphologyCoarseGridValidationError(RuntimeError):
    """Malformed, inconsistent, or unverifiable input artifacts."""


def _fail(message: str) -> AnomalyMorphologyCoarseGridValidationError:
    return AnomalyMorphologyCoarseGridValidationError(message)


def _finite_tree(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise _fail("nonfinite input")


def _read_json_file(path: Path, label: str) -> tuple[bytes, Mapping[str, Any]]:
    document, payload = builder._read_json(path, label)
    _finite_tree(document)
    return payload, document


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    # JSON equality also rejects boolean-as-integer counters and field-set drift.
    if _canonical_compact_json_bytes(actual) != _canonical_compact_json_bytes(expected):
        raise _fail(f"{label} does not reconstruct exactly")


@dataclass(frozen=True)
class _VerifiedGrid:
    contract: Mapping[str, Any]
    build_manifest: Mapping[str, Any]
    project: Mapping[str, Any]
    datasets: tuple[Mapping[str, Any], ...]
    project_sha256: str
    artifact_sha256s: Mapping[str, str]


@dataclass(frozen=True)
class _VerifiedInvestigation:
    investigation_id: str
    investigation_sha256: str
    run_stage_id: str
    run_stage_ledger_sha256: str
    stage_ledger_sha256s: Mapping[str, str]
    winners: tuple[Mapping[str, Any], ...]


def _verify_grid_root(root: Path, verified: builder._VerifiedPreparation) -> _VerifiedGrid:
    root = _regular_directory(root, "coarse project root")
    if {item.name for item in root.iterdir()} != {
        builder.PROJECT_RELATIVE_PATH, builder.CONTRACT_RELATIVE_PATH,
        builder.BUILD_MANIFEST_RELATIVE_PATH, builder.DATASET_DIRECTORY,
    }:
        raise _fail("coarse project artifact set is incomplete or unexpected")
    project_bytes, project = _read_json_file(root / builder.PROJECT_RELATIVE_PATH, "project")
    contract_bytes, contract = _read_json_file(root / builder.CONTRACT_RELATIVE_PATH, "coarse contract")
    build_bytes, build = _read_json_file(root / builder.BUILD_MANIFEST_RELATIVE_PATH, "build manifest")
    project_id = _nonempty_string(project.get("id"), "project ID")
    if builder._SAFE_PROJECT_ID.fullmatch(project_id) is None:
        raise _fail("unsafe project ID")
    maximum_candidates = _exact_count(contract.get("maximumCandidatesPerSearch"), "candidate limit", positive=True)
    if maximum_candidates > builder.MAXIMUM_ALLOWED_CANDIDATES_PER_SEARCH:
        raise _fail("candidate limit exceeds supported bound")
    searches = builder._searches(project_id, verified, maximum_candidates)
    _require_equal(contract, builder._contract(verified, searches, maximum_candidates), "coarse contract")
    _require_equal(project, builder._project(project_id, searches), "project")
    dataset_root = _regular_directory(root / builder.DATASET_DIRECTORY, "coarse datasets")
    if {item.name for item in dataset_root.iterdir()} != {Path(s.output_file).name for s in searches}:
        raise _fail("coarse dataset artifact set is incomplete or unexpected")
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    series_by_id = {source["genericSeriesID"]: source for source in verified.series}
    dataset_documents = []
    dataset_records = []
    work_unit_counts = []
    evaluation_counts = []
    for search in searches:
        dataset_bytes, dataset = _read_json_file(root / search.output_file, "coarse dataset")
        expected = builder._dataset(
            search=search, series_by_id=series_by_id, verified=verified,
            coarse_contract_sha256=contract_sha256,
        )
        _require_equal(dataset, expected, "coarse dataset")
        sample_count = _safe_sum([len(s["coordinates"]) for s in dataset["series"]], "sample count")
        work_unit_count = builder._safe_work_unit_count(search.coarse_candidate_count)
        evaluation_count = builder._safe_product([sample_count, search.coarse_candidate_count], "evaluation count")
        work_unit_counts.append(work_unit_count)
        evaluation_counts.append(evaluation_count)
        dataset_documents.append((search, dataset, dataset_bytes))
        dataset_records.append({
            "candidatesPerWorkUnit": builder.CANDIDATES_PER_WORK_UNIT,
            "coarseCandidateCount": search.coarse_candidate_count,
            "datasetID": search.dataset_id,
            "expectedSampleCandidateEvaluationCount": evaluation_count,
            "expectedWorkUnitCount": work_unit_count,
            "fullCandidateCount": search.full_candidate_count,
            "genericSeriesIDs": list(search.generic_series_ids),
            "modelClassID": search.model_class_id,
            "outputFile": search.output_file,
            "outputSHA256": _sha256_bytes(dataset_bytes),
            "sampleCount": sample_count,
            "selectedStride": search.stride,
        })
    total_work_units = _safe_sum(work_unit_counts, "total work units")
    total_evaluations = _safe_sum(evaluation_counts, "total evaluations")
    expected_build = {
        "algorithmID": builder.COARSE_GRID_ALGORITHM_ID,
        "algorithmVersion": builder.COARSE_GRID_ALGORITHM_VERSION,
        "buildManifestSchemaID": builder.BUILD_MANIFEST_SCHEMA_ID,
        "buildManifestVersion": builder.BUILD_MANIFEST_VERSION,
        "candidatesPerWorkUnit": builder.CANDIDATES_PER_WORK_UNIT,
        "datasets": dataset_records,
        "identityIsolationStatement": contract["identityIsolationStatement"],
        "inputArtifactManifestFileSHA256": verified.manifest_file_sha256,
        "inputMorphologyContractCanonicalSHA256": (
            verified.contract_canonical_sha256
        ),
        "inputMorphologyContractFileSHA256": verified.contract_file_sha256,
        "inputMorphologyContractID": builder.MORPHOLOGY_CONTRACT_ID,
        "inputMorphologyContractVersion": builder.MORPHOLOGY_CONTRACT_VERSION,
        "inputPreparationID": builder.MORPHOLOGY_PREPARATION_SCHEMA_ID,
        "inputPreparationFileSHA256": verified.preparation_file_sha256,
        "inputPreparationResultSchemaID": builder.MORPHOLOGY_PREPARATION_SCHEMA_ID,
        "inputPreparationResultVersion": builder.MORPHOLOGY_PREPARATION_VERSION,
        "inputPreparationSHA256": verified.preparation_file_sha256,
        "maximumCandidatesPerSearch": maximum_candidates,
        "modelScopeStatement": contract["modelScopeStatement"],
        "noCandidateEvaluationStatement": contract[
            "noCandidateEvaluationStatement"
        ],
        "orderedDatasetIDs": [record["datasetID"] for record in dataset_records],
        "outputHashes": {
            "coarseGridContract": _sha256_bytes(contract_bytes),
            "datasets": {
                record["outputFile"]: record["outputSHA256"]
                for record in dataset_records
            },
            "project": _sha256_bytes(project_bytes),
        },
        "parentHashes": verified.preparation["parentHashes"],
        "parentIDs": verified.preparation["parentIDs"],
        "projectID": project_id,
        "relativeArtifactPaths": {
            "buildManifest": builder.BUILD_MANIFEST_RELATIVE_PATH,
            "coarseGridContract": builder.CONTRACT_RELATIVE_PATH,
            "datasets": [record["outputFile"] for record in dataset_records],
            "project": builder.PROJECT_RELATIVE_PATH,
        },
        "sourceDatasetFileSHA256s": dict(verified.series_file_sha256s),
        "totalCoarseCandidateCount": _safe_sum(
            [search.coarse_candidate_count for search in searches],
            "total coarse candidate count",
        ),
        "totalExpectedSampleCandidateEvaluationCount": total_evaluations,
        "totalExpectedWorkUnitCount": total_work_units,
        "workloadIdentities": contract["workloadIdentities"],
    }
    _require_equal(build, expected_build, "coarse build manifest")
    documents = (project, contract, build, *(item[1] for item in dataset_documents))
    builder._assert_identity_isolated(documents)
    _assert_identity_free((project_bytes, contract_bytes, build_bytes, *(item[2] for item in dataset_documents)))
    hashes = {
        builder.PROJECT_RELATIVE_PATH: _sha256_bytes(project_bytes),
        builder.CONTRACT_RELATIVE_PATH: _sha256_bytes(contract_bytes),
        builder.BUILD_MANIFEST_RELATIVE_PATH: _sha256_bytes(build_bytes),
        **{s.output_file: _sha256_bytes(data) for s, _, data in dataset_documents},
    }
    return _VerifiedGrid(contract, build, project, tuple(item[1] for item in dataset_documents),
                         hashes[builder.PROJECT_RELATIVE_PATH], hashes)


def _verify_investigation(
    path: Path,
    grid: _VerifiedGrid,
    project_path: Path,
) -> _VerifiedInvestigation:
    label = "morphology-grid investigation"
    if path.name != "investigation.json":
        raise _fail(f"{label} record must be named investigation.json")
    directory = _regular_directory(path.parent, f"{label} directory")
    try:
        if {entry.name for entry in directory.iterdir()} != {
            "investigation.json",
            "stages",
        }:
            raise _fail(f"{label} artifact set is incomplete or unexpected")
    except OSError as error:
        raise _fail(f"{label} directory is unreadable") from error
    _regular_directory(directory / "stages", f"{label} stage directory")
    record_bytes, investigation = _read_json_file(path, f"{label} record")
    if set(investigation) != _INVESTIGATION_FIELDS:
        raise _fail(f"{label} field set is invalid")
    investigation_id = _nonempty_string(investigation.get("id"), f"{label} ID")
    if _SAFE_ID.fullmatch(investigation_id) is None:
        raise _fail(f"{label} ID is unsafe")
    if path.parent.name != investigation_id:
        raise _fail(f"{label} directory does not match its ID")
    if investigation.get("workflow_id") != SMOKE_WORKFLOW_ID:
        raise _fail(f"{label} workflow is invalid")
    if investigation.get("workflow_version") != "20.0":
        raise _fail(f"{label} workflow version is unsupported")
    if investigation.get("status") != "COMPLETE":
        raise _fail(f"{label} is not COMPLETE")
    metadata = investigation.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "coordinator",
        "projectPath",
    }:
        raise _fail(f"{label} metadata is invalid")
    _nonempty_string(metadata.get("coordinator"), f"{label} coordinator")
    expected_project_path = project_path.resolve()
    metadata_path = metadata.get("projectPath")
    if not isinstance(metadata_path, str):
        raise _fail(f"{label} metadata project path is missing")
    try:
        if Path(metadata_path).expanduser().resolve() != expected_project_path:
            raise _fail(f"{label} metadata refers to a different project")
    except OSError as error:
        raise _fail(f"{label} project path cannot be resolved") from error

    stage_values = investigation.get("stages")
    if not isinstance(stage_values, list) or len(stage_values) != 3:
        raise _fail(f"{label} must contain exactly three stages")
    stages: list[Mapping[str, Any]] = []
    stage_ids: set[str] = set()
    for value in stage_values:
        if not isinstance(value, Mapping):
            raise _fail(f"{label} stage is malformed")
        try:
            stage_id, _ = _validate_stage_shape(value)
        except RefinementGridBuildError as error:
            raise _fail(str(error)) from error
        if stage_id in stage_ids or value.get("artifacts") != []:
            raise _fail(f"{label} stage ledger structure is invalid")
        stage_ids.add(stage_id)
        stages.append(value)
    for stage in stages:
        _read_json_file(directory / "stages" / f"{stage['id']}.json", "stage ledger")
    try:
        ledger_hashes = _stage_ledgers(path, stages)
    except RefinementGridBuildError as error:
        raise _fail(str(error)) from error
    try:
        _assert_identity_free((record_bytes,))
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error
    prepare = [item for item in stages if item.get("handler_id") == PREPARE_HANDLER_ID]
    run = [item for item in stages if item.get("handler_id") == PROJECT_RUN_HANDLER_ID]
    terminal = [
        item for item in stages if item.get("handler_id") == TERMINAL_CHECK_HANDLER_ID
    ]
    if len(prepare) != 1 or len(run) != 1 or len(terminal) != 1:
        raise _fail(f"{label} handler structure is invalid")
    prepare_stage, run_stage, terminal_stage = prepare[0], run[0], terminal[0]
    if stages != [prepare_stage, run_stage, terminal_stage]:
        raise _fail(f"{label} stages are out of canonical order")
    if (
        prepare_stage.get("triggered_by_stage_id") is not None
        or run_stage.get("triggered_by_stage_id") != prepare_stage["id"]
        or terminal_stage.get("triggered_by_stage_id") != run_stage["id"]
    ):
        raise _fail(f"{label} stage causality is invalid")

    project_id = grid.project["id"]
    project_hash = grid.project_sha256
    expected_path_string = str(expected_project_path)
    prepare_parameters = {"projectPath": expected_path_string}
    run_parameters = {
        "projectManifestSha256": project_hash,
        "projectPath": expected_path_string,
    }
    terminal_parameters = {"expectedProjectID": project_id}
    if prepare_stage["parameters"] != prepare_parameters:
        raise _fail(f"{label} preparation parameters are invalid")
    if prepare_stage["result"] != {
        "projectManifestSha256": project_hash,
        "projectPath": expected_path_string,
    }:
        raise _fail(f"{label} preparation result is invalid")
    if run_stage["parameters"] != run_parameters:
        raise _fail(f"{label} run parameters are invalid")
    if terminal_stage["parameters"] != terminal_parameters:
        raise _fail(f"{label} terminal parameters are invalid")
    if prepare_stage.get("next_stage") != {
        "handler_id": PROJECT_RUN_HANDLER_ID,
        "id": run_stage["id"],
        "parameters": run_parameters,
        "triggered_by_stage_id": prepare_stage["id"],
    }:
        raise _fail(f"{label} preparation continuation is invalid")
    if run_stage.get("next_stage") != {
        "handler_id": TERMINAL_CHECK_HANDLER_ID,
        "id": terminal_stage["id"],
        "parameters": terminal_parameters,
        "triggered_by_stage_id": run_stage["id"],
    }:
        raise _fail(f"{label} run continuation is invalid")
    if (
        prepare_stage.get("stop") is not False
        or run_stage.get("stop") is not False
        or terminal_stage.get("stop") is not True
        or terminal_stage.get("next_stage") is not None
    ):
        raise _fail(f"{label} terminal flags are invalid")
    prepare_provenance = prepare_stage["provenance"]
    run_provenance = run_stage["provenance"]
    terminal_provenance = terminal_stage["provenance"]
    if (
        prepare_provenance.get("input_hashes")
        != {"projectManifest": project_hash}
        or prepare_provenance.get("project_ids") != []
        or prepare_provenance.get("node_contributions") != {}
        or run_provenance.get("input_hashes")
        != {"projectManifest": project_hash}
        or run_provenance.get("project_ids") != [project_id]
        or terminal_provenance.get("input_hashes") != {}
        or terminal_provenance.get("project_ids") != [project_id]
        or terminal_provenance.get("node_contributions") != {}
    ):
        raise _fail(f"{label} stage provenance is invalid")

    expected_work_units = grid.build_manifest["totalExpectedWorkUnitCount"]
    contributions = run_provenance.get("node_contributions")
    if not isinstance(contributions, Mapping) or _safe_sum(
        list(contributions.values()), "node contribution count"
    ) != expected_work_units:
        raise _fail(f"{label} node contributions do not match completed work")
    run_result = run_stage["result"]
    required_run_envelope = {
        "projectID": project_id,
        "projectPath": expected_path_string,
        "status": "COMPLETE",
        "workloadID": WORKLOAD_ID,
    }
    if any(
        run_result.get(key) != value
        for key, value in required_run_envelope.items()
    ):
        raise _fail(f"{label} project run envelope is invalid")
    if run_result.get("nodeContributions") != contributions:
        raise _fail(f"{label} node contributions disagree")
    statuses = run_result.get("datasets")
    if not isinstance(statuses, list) or len(statuses) != len(grid.datasets):
        raise _fail(f"{label} dataset result set is incomplete")
    if [status.get("id") for status in statuses if isinstance(status, Mapping)] != [
        dataset["id"] for dataset in grid.datasets
    ]:
        raise _fail(f"{label} dataset results are out of canonical order")
    winners: list[Mapping[str, Any]] = []
    dataset_contribution_totals: dict[str, int] = {}
    for status, dataset, record in zip(statuses, grid.datasets, grid.build_manifest["datasets"]):
        candidate_count = record["coarseCandidateCount"]
        work_unit_count = record["expectedWorkUnitCount"]
        if not isinstance(status, Mapping):
            raise _fail(f"{label} dataset status is malformed")
        required_status = {
            "assignedWorkUnits": 0,
            "completedCandidateCount": candidate_count,
            "completedWorkUnits": work_unit_count,
            "coverageComplete": True,
            "morphologyGridStatus": "MORPHOLOGY_GRID_COMPLETE",
            "modelClassID": dataset["modelClassID"],
            "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
            "datasetSchemaID": DATASET_SCHEMA_ID,
            "failedWorkUnits": 0,
            "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
            "id": dataset["id"],
            "payloadSchemaID": PAYLOAD_SCHEMA_ID,
            "pendingWorkUnits": 0,
            "resultSchemaID": RESULT_SCHEMA_ID,
            "totalCandidateCount": candidate_count,
            "totalWorkUnits": work_unit_count,
            "workloadID": WORKLOAD_ID,
            "workloadStatus": "MORPHOLOGY_GRID_COMPLETE",
        }
        if any(type(status.get(key)) is not type(value) or status.get(key) != value
               for key, value in required_status.items()):
            raise _fail(f"{label} dataset coverage is incomplete")
        dataset_contributions = status.get("nodeContributions")
        if not isinstance(dataset_contributions, Mapping):
            raise _fail(f"{label} dataset contributions are malformed")
        if _safe_sum(
            list(dataset_contributions.values()),
            "dataset node contribution count",
        ) != work_unit_count:
            raise _fail(f"{label} dataset contributions are incomplete")
        for node_id, count in dataset_contributions.items():
            _nonempty_string(node_id, "dataset contribution node ID")
            dataset_contribution_totals[node_id] = (
                dataset_contribution_totals.get(node_id, 0) + count
            )
        invalid_count = _exact_count(status.get("totalInvalidCandidateCount"), "invalid candidate count")
        if invalid_count >= candidate_count:
            raise _fail("accepted winner conflicts with invalid candidate count")
        winners.append(_winner_from_status(status, dataset))
    if dataset_contribution_totals != contributions:
        raise _fail(f"{label} dataset and project contributions disagree")
    _verify_run_counter_scopes(
        run_result,
        statuses[-1],
        expected_project_work_units=expected_work_units,
    )

    expected_terminal = {
        "completedWorkUnits": expected_work_units,
        "failedWorkUnits": 0,
        "passed": True,
        "projectID": project_id,
        "rule": "projectID matches and completed+failed == total",
        "totalWorkUnits": expected_work_units,
    }
    _require_equal(terminal_stage["result"], expected_terminal, "terminal check")
    return _VerifiedInvestigation(
        investigation_id=investigation_id,
        investigation_sha256=_sha256_bytes(record_bytes),
        run_stage_id=run_stage["id"],
        run_stage_ledger_sha256=ledger_hashes[run_stage["id"]],
        stage_ledger_sha256s=dict(sorted(ledger_hashes.items())),
        winners=tuple(winners),
    )


def _winner_from_status(status: Mapping[str, Any], dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = status.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"bestCandidate"}:
        raise _fail("dataset winner payload is malformed")
    candidate = workload._strict_candidate_payload(payload["bestCandidate"], dataset["modelClassID"])
    index = _exact_count(candidate["gridIndex"], "accepted grid index")
    grid = workload._grid(dataset)
    if index >= grid.total_candidates:
        raise _fail("accepted winner index is outside the grid")
    _require_equal(candidate["parameters"], workload._candidate_parameters(grid, index), "winner grid mapping")
    # Deliberately reproduce only this accepted index. validate_result/reduce_dataset
    # also scan shards; calling those here would silently perform a new search.
    evaluated = workload._evaluate_candidate(dataset, index)
    if evaluated is None or not workload._candidate_payload_matches(candidate, evaluated, dataset["modelClassID"]):
        raise _fail("accepted morphology winner does not reproduce canonically")
    for key, value in candidate.items():
        summary_key = "best" + key[0].upper() + key[1:]
        if summary_key not in status:
            raise _fail(f"missing duplicated winner field {summary_key}")
        _require_equal(status[summary_key], value, f"duplicated winner {summary_key}")
    # Preserve accepted numerical values after checking the workload tolerances.
    return dict(candidate)


def _verify_interpretation_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed on unsupported decision semantics, even with refreshed hashes."""
    requirements = {
        "independentTimingConsistencyTolerance": contract["parameterAxes"]["CENTER"]["step"],
        "timingComparisonTolerance": preparation.TIMING_COMPARISON_TOLERANCE,
        "orderedCentersAndShapesSharedAcrossSeries": True,
        "orderedPerSeriesSigns": ["negative", "positive"],
        "positivePulsePerSeriesSign": "positive",
        "positiveWeightSupportPerComponentPerSeries": preparation.MINIMUM_COMPONENT_POSITIVE_WEIGHT_SUPPORT,
    }
    _require_equal(contract["crossSeriesRequirements"], requirements, "cross-series requirements")
    rules = {
        "preferOrderedDoubletOverPositivePulse": {
            "allPerSeriesDeltaWRSSAtLeast": preparation.ORDERED_OVER_POSITIVE_MINIMUM_PER_SERIES_DELTA_WRSS,
            "globalDeltaBICAtLeast": preparation.ORDERED_OVER_POSITIVE_MINIMUM_DELTA_BIC,
            "globalDeltaWRSSAtLeast": preparation.ORDERED_OVER_POSITIVE_MINIMUM_DELTA_WRSS,
            "signAndSharedTimingRequirementsMustPass": True,
        },
        "rejectOrderedDoubletForIndependentPulses": {
            "globalDeltaBICAtLeast": preparation.INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_BIC,
            "globalDeltaWRSSAtLeast": preparation.INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_WRSS,
            "independentCenterDispersionMustExceedTolerance": True,
            "signRequirementsMustPass": True,
        },
        "tieBreak": (
            "Use deterministicExecution.candidateWinnerOrdering and its "
            "declared relative tolerance. Model ties use modelClassOrder."
        ),
    }
    _require_equal(contract["decisionRules"], rules, "decision rules")
    _require_equal(contract["deterministicExecution"]["decisionThresholdComparison"], (
        "Threshold gates use exact binary64 >= comparisons with no objective "
        "tie tolerance; timing consistency uses its declared tolerance."
    ), "threshold comparison semantics")
    metrics = contract["comparisonMetrics"]
    expected_metrics = {
        "BIC": {"formula": "WRSS + k*ln(N)", "logarithm": "natural logarithm",
                "invalidRule": "N <= 0 or a nonfinite result is invalid"},
        "AICc": {"formula": "WRSS + 2*k + 2*k*(k+1)/(N-k-1)", "undefinedRule": (
            "When N <= k + 1, emit value null and defined false; an "
            "undefined AICc sorts after every finite AICc."
        )},
        "N": (
            "Integer sum, in canonical series order, of samples whose "
            "inverse variance is strictly greater than zero."
        ),
        "WRSS": (
            "Binary64 sum of per-series WRSS in canonical series order; "
            "each per-series WRSS uses source sample order."
        ),
        "zeroConstrainedAmplitudeParameterCountRule": (
            "Nominal k is never reduced when a constrained amplitude is zero."
        ),
    }
    for key, expected in expected_metrics.items():
        _require_equal(metrics[key], expected, f"information criteria {key}")
    series_count = len(contract["admittedGenericSeriesIDs"])
    expected_counts = {
        POSITIVE_PULSE_ONLY: {"linear": 2 * series_count, "nonlinear": 3, "total": 2 * series_count + 3},
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: {"linear": 3 * series_count, "nonlinear": 6, "total": 3 * series_count + 6},
        INDEPENDENT_PULSES: {"linear": 3 * series_count, "nonlinear": 6 * series_count, "total": 9 * series_count},
    }
    _require_equal(metrics["parameterCounts"], expected_counts, "nominal parameter counts")
    independent = contract["independentAggregation"]
    _require_equal(independent["parameterCount"], 9 * series_count, "independent parameter count")
    expected_independent = {
        "WRSS": "sum per-series accepted WRSS in canonical series order",
        "N": "sum per-series positive-weight N in canonical series order",
        "informationCriteria": (
            "Compute global BIC and AICc once from aggregate WRSS, aggregate N, "
            "and nominal k using comparisonMetrics."
        ),
        "negativeCenterDispersion": "max(negative centers) - min(negative centers)",
        "positiveCenterDispersion": "max(positive centers) - min(positive centers)",
        "timingDispersion": (
            "max(negativeCenterDispersion, positiveCenterDispersion); exact "
            "+0.0 when one series is admitted"
        ),
        "timingConsistencyRule": (
            "timingDispersion <= independentTimingConsistencyTolerance + "
            "timingComparisonTolerance"
        ),
    }
    for key, expected in expected_independent.items():
        _require_equal(independent[key], expected, f"independent aggregation {key}")
    limits = contract["interpretationLimits"]
    if limits["discoveryClaim"] is not False or limits["planetaryInterpretationResolved"] is not False:
        raise _fail("unsupported interpretation limits")
    _require_equal(limits["boundaryWinnerRule"], (
        "A winner on any width boundary remains morphology-width "
        "unresolved and cannot be reported as a measured duration."
    ), "width interpretation rule")


def _component_geometries(model: str, parameters: Mapping[str, Any]) -> list[tuple[str, float, float, float]]:
    if model == POSITIVE_PULSE_ONLY:
        return [("positive", parameters["center"], parameters["logScale"], parameters["logShape"])]
    positive_center = (
        parameters["negativeCenter"] + parameters["separation"]
        if model == ORDERED_NEGATIVE_POSITIVE_DOUBLET else parameters["positiveCenter"]
    )
    return [
        ("negative", parameters["negativeCenter"], parameters["negativeLogScale"], parameters["negativeLogShape"]),
        ("positive", positive_center, parameters["positiveLogScale"], parameters["positiveLogShape"]),
    ]


def _component_support(series: Mapping[str, Any], center: float, log_scale: float,
                       log_shape: float, minimum: int) -> dict[str, Any]:
    center = _finite_number(center, "component center")
    # Match preparation._component_geometry: do not reassociate to exp(a+b).
    width = math.exp(_finite_number(log_scale, "log scale")) * math.exp(_finite_number(log_shape, "log shape"))
    threshold = preparation.COMPONENT_SUPPORT_WIDTH_MULTIPLIER * width
    if not math.isfinite(width) or width <= 0.0 or not math.isfinite(threshold):
        raise _fail("component effective width is invalid")
    coordinates, weights = series["coordinates"], series["inverseVariances"]
    if len(coordinates) != len(weights):
        raise _fail("support arrays differ in length")
    distances = []
    for coordinate, weight in zip(coordinates, weights):
        coordinate = _finite_number(coordinate, "support coordinate")
        weight = _finite_number(weight, "support weight")
        if weight < 0.0:
            raise _fail("negative support weight")
        if weight > 0.0:
            distances.append(_finite_number(abs(coordinate - center), "support distance"))
    if not distances:
        raise _fail("series has no positive-weight observations")
    count = sum(distance <= threshold for distance in distances)
    nearest = _finite_number(min(distances) / width, "nearest distance in effective widths")
    return {
        "genericSeriesID": series["genericSeriesID"],
        "center": center,
        "effectiveWidth": width,
        "positiveWeightSamplesWithinTwoEffectiveWidths": count,
        "nearestPositiveWeightDistanceInEffectiveWidths": nearest,
        "minimumRequiredPositiveWeightSamples": minimum,
        "supportRequirementMet": count >= minimum,
    }


def _axis_boundaries(dataset: Mapping[str, Any], grid_index: int) -> list[dict[str, Any]]:
    grid = workload._grid(dataset)
    indices = workload._candidate_indices(grid, grid_index)
    if grid.model_class_id == POSITIVE_PULSE_ONLY:
        names = ("center", "logScale", "logShape")
        axes = grid.axes
    elif grid.model_class_id == ORDERED_NEGATIVE_POSITIVE_DOUBLET:
        names = ("negativeCenter", "separation", "negativeLogScale", "negativeLogShape", "positiveLogScale", "positiveLogShape")
        axes = grid.axes
    else:
        negative, positive = workload.independent_center_pair_indices(grid.axes[0].count, indices[0])
        indices = (negative, positive, *indices[1:])
        names = ("negativeCenter", "positiveCenter", "negativeLogScale", "negativeLogShape", "positiveLogScale", "positiveLogShape")
        axes = (grid.axes[0], grid.axes[0], *grid.axes[1:])
    records = []
    for name, axis, index in zip(names, axes, indices):
        fixed = axis.count == 1
        boundary = not fixed and index in (0, axis.count - 1)
        records.append({
            "axis": name, "index": index, "count": axis.count,
            "value": axis.value(index), "minimum": axis.value(0),
            "maximum": axis.value(axis.count - 1), "fixed": fixed,
            "searchedBoundary": boundary,
            "position": "FIXED" if fixed else "LOWER_BOUNDARY" if index == 0 else "UPPER_BOUNDARY" if boundary else "INTERIOR",
        })
    return records


def _information_criteria(wrss: float, n: int, k: int) -> dict[str, Any]:
    wrss = _finite_number(wrss, "aggregate WRSS")
    n = _exact_count(n, "aggregate N", positive=True)
    k = _exact_count(k, "nominal k", positive=True)
    if wrss < 0.0:
        raise _fail("negative aggregate WRSS")
    bic = _finite_number(wrss + k * math.log(n), "global BIC")
    aicc = None
    if n > k + 1:
        aicc = _finite_number(wrss + 2.0 * k + (2.0 * k * (k + 1) / (n - k - 1)), "global AICc")
    return {
        "weightedResidualSumSquares": wrss, "positiveWeightSampleCount": n,
        "nominalParameterCount": k, "bayesianInformationCriterion": bic,
        "correctedAkaikeInformationCriterion": aicc,
        "correctedAkaikeInformationCriterionDefined": aicc is not None,
    }


def _independent_aggregate(winners: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    expected_ids = contract["admittedGenericSeriesIDs"]
    if (len(winners) != len(expected_ids)
            or any(len(w["seriesFits"]) != 1 for w in winners)
            or [w["seriesFits"][0]["genericSeriesID"] for w in winners] != expected_ids):
        raise _fail("independent accepted winner vector is incomplete or out of order")
    wrss = 0.0
    counts = []
    parameters = []
    for winner in winners:
        wrss = _finite_number(wrss + winner["weightedResidualSumSquares"], "independent WRSS sum")
        counts.append(winner["positiveWeightSampleCount"])
        parameters.append(winner["nominalParameterCount"])
    n = _safe_sum(counts, "independent N")
    k = _safe_sum(parameters, "independent nominal k")
    if k != contract["independentAggregation"]["parameterCount"]:
        raise _fail("independent nominal parameter count disagrees with contract")
    result = _information_criteria(wrss, n, k)
    negative = [w["parameters"]["negativeCenter"] for w in winners]
    positive = [w["parameters"]["positiveCenter"] for w in winners]
    negative_dispersion = _finite_number(max(negative) - min(negative), "negative center dispersion")
    positive_dispersion = _finite_number(max(positive) - min(positive), "positive center dispersion")
    dispersion = max(negative_dispersion, positive_dispersion)
    requirements = contract["crossSeriesRequirements"]
    tolerance = requirements["independentTimingConsistencyTolerance"] + requirements["timingComparisonTolerance"]
    result.update({
        "acceptedWinnerIndices": [w["gridIndex"] for w in winners],
        "genericSeriesIDs": list(expected_ids),
        "negativeCenterDispersion": negative_dispersion,
        "positiveCenterDispersion": positive_dispersion,
        "timingDispersion": dispersion, "timingConsistencyTolerance": tolerance,
        "timingConsistent": dispersion <= tolerance,
    })
    return result


def _comparisons(searches: Sequence[Mapping[str, Any]], independent: Mapping[str, Any],
                 contract: Mapping[str, Any]) -> dict[str, Any]:
    positive, ordered = (item["acceptedWinner"] for item in searches[:2])
    positive_fits = positive["seriesFits"]
    ordered_fits = ordered["seriesFits"]
    independent_fits = [item["acceptedWinner"]["seriesFits"][0] for item in searches[2:]]
    rules = contract["decisionRules"]
    ordered_rule = rules["preferOrderedDoubletOverPositivePulse"]
    independent_rule = rules["rejectOrderedDoubletForIndependentPulses"]
    per_series = []
    for p, o, i in zip(positive_fits, ordered_fits, independent_fits):
        delta = p["weightedResidualSumSquares"] - o["weightedResidualSumSquares"]
        per_series.append({
            "genericSeriesID": p["genericSeriesID"],
            "orderedOverPositiveDeltaWRSS": delta,
            "orderedOverPositiveDeltaWRSSPassed": delta >= ordered_rule["allPerSeriesDeltaWRSSAtLeast"],
            "independentOverOrderedDeltaWRSS": o["weightedResidualSumSquares"] - i["weightedResidualSumSquares"],
            "positiveSign": p["positiveAmplitudeSign"],
            "orderedSigns": [o["negativeAmplitudeSign"], o["positiveAmplitudeSign"]],
            "independentSigns": [i["negativeAmplitudeSign"], i["positiveAmplitudeSign"]],
        })
    ordered_delta = positive["weightedResidualSumSquares"] - ordered["weightedResidualSumSquares"]
    ordered_bic = positive["bayesianInformationCriterion"] - ordered["bayesianInformationCriterion"]
    independent_delta = ordered["weightedResidualSumSquares"] - independent["weightedResidualSumSquares"]
    independent_bic = ordered["bayesianInformationCriterion"] - independent["bayesianInformationCriterion"]
    positive_signs = all(item["positiveSign"] == "positive" for item in per_series)
    ordered_signs = all(item["orderedSigns"] == ["negative", "positive"] for item in per_series)
    independent_signs = all(item["independentSigns"] == ["negative", "positive"] for item in per_series)
    ordered_gates = {
        "globalDeltaWRSSPassed": ordered_delta >= ordered_rule["globalDeltaWRSSAtLeast"],
        "globalDeltaBICPassed": ordered_bic >= ordered_rule["globalDeltaBICAtLeast"],
        "allPerSeriesDeltaWRSSPassed": all(item["orderedOverPositiveDeltaWRSS"] >= ordered_rule["allPerSeriesDeltaWRSSAtLeast"] for item in per_series),
        "signRequirementsPassed": positive_signs and ordered_signs,
        "sharedTimingRequirementsPassed": True,
        "supportRequirementsPassed": all(item["supportRequirementMet"] for item in searches[:2]),
    }
    independent_gates = {
        "globalDeltaWRSSPassed": independent_delta >= independent_rule["globalDeltaWRSSAtLeast"],
        "globalDeltaBICPassed": independent_bic >= independent_rule["globalDeltaBICAtLeast"],
        "centerDispersionExceedsTolerance": not independent["timingConsistent"],
        "signRequirementsPassed": ordered_signs and independent_signs,
        "supportRequirementsPassed": all(item["supportRequirementMet"] for item in searches[1:]),
    }
    return {
        "perSeries": per_series,
        "preferOrderedDoubletOverPositivePulse": {
            "deltaWRSS": ordered_delta, "deltaBIC": ordered_bic,
            "thresholds": dict(ordered_rule), "gates": ordered_gates,
            "passed": all(ordered_gates.values()),
        },
        "rejectOrderedDoubletForIndependentPulses": {
            "deltaWRSS": independent_delta, "deltaBIC": independent_bic,
            "thresholds": dict(independent_rule), "gates": independent_gates,
            "passed": all(independent_gates.values()),
        },
    }


def _report(verified: builder._VerifiedPreparation, grid: _VerifiedGrid,
            investigation: _VerifiedInvestigation) -> dict[str, Any]:
    contract = verified.contract
    minimum = contract["crossSeriesRequirements"]["positiveWeightSupportPerComponentPerSeries"]
    searches = []
    for dataset, winner, record in zip(grid.datasets, investigation.winners, grid.build_manifest["datasets"]):
        components = []
        for name, center, log_scale, log_shape in _component_geometries(dataset["modelClassID"], winner["parameters"]):
            support = [_component_support(series, center, log_scale, log_shape, minimum) for series in dataset["series"]]
            components.append({
                "component": name, "center": center, "logScale": log_scale, "logShape": log_shape,
                "effectiveWidth": support[0]["effectiveWidth"], "seriesSupport": support,
                "supportRequirementMet": all(item["supportRequirementMet"] for item in support),
            })
        boundaries = _axis_boundaries(dataset, winner["gridIndex"])
        width_boundary = any(item["searchedBoundary"] and ("Scale" in item["axis"] or "Shape" in item["axis"]) for item in boundaries)
        searches.append({
            "datasetID": dataset["id"], "modelClassID": dataset["modelClassID"],
            "genericSeriesIDs": dataset["sourceGenericSeriesIDs"], "acceptedWinner": dict(winner),
            "coverageComplete": True, "completedCandidateCount": record["coarseCandidateCount"],
            "completedWorkUnits": record["expectedWorkUnitCount"],
            "components": components, "axes": boundaries,
            "searchedBoundaryAxes": [item["axis"] for item in boundaries if item["searchedBoundary"]],
            "fixedAxes": [item["axis"] for item in boundaries if item["fixed"]],
            "widthInterpretationLimitedByBoundary": width_boundary,
            "supportRequirementMet": all(item["supportRequirementMet"] for item in components),
        })
    independent = _independent_aggregate(investigation.winners[2:], contract)
    comparisons = _comparisons(searches, independent, contract)
    unsupported_count = sum(not item["supportRequirementMet"] for item in searches)
    preference = None
    classification = "MODEL_COMPARISON_UNRESOLVED"
    if unsupported_count:
        classification = UNRESOLVED_SUPPORT
    elif comparisons["rejectOrderedDoubletForIndependentPulses"]["passed"]:
        preference = INDEPENDENT_PULSES
        classification = "INDEPENDENT_PULSES_PREFERRED"
    elif comparisons["preferOrderedDoubletOverPositivePulse"]["passed"]:
        preference = ORDERED_NEGATIVE_POSITIVE_DOUBLET
        classification = "ORDERED_DOUBLET_PREFERRED"
    width_limited = any(item["widthInterpretationLimitedByBoundary"] for item in searches)
    return {
        "resultSchemaID": RESULT_SCHEMA_ID_REPORT, "resultVersion": RESULT_VERSION,
        "overallClassification": classification, "modelPreference": preference,
        "modelPreferenceResolved": preference is not None,
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
        "widthInterpretationResolved": False,
        "widthInterpretationLimitedByBoundary": width_limited,
        "widthInterpretationStatement": "Coarse-grid effective widths are model geometry, not measured physical durations.",
        "unsupportedSearchCount": unsupported_count,
        "allAcceptedWinnersMeetSupportRequirement": unsupported_count == 0,
        "searches": searches, "independentAggregate": independent, "modelComparisons": comparisons,
        "projectID": grid.project["id"], "investigationID": investigation.investigation_id,
        "projectCompletedWorkUnits": grid.build_manifest["totalExpectedWorkUnitCount"],
        "projectTotalWorkUnits": grid.build_manifest["totalExpectedWorkUnitCount"],
        "totalCoarseCandidateCount": grid.build_manifest["totalCoarseCandidateCount"],
        "morphologyContractID": contract["contractID"],
        "morphologyContractSHA256": verified.contract_canonical_sha256,
        "coarseGridContractSHA256": _sha256_bytes(_canonical_compact_json_bytes(grid.contract)),
        "supportRule": {
            "effectiveWidthFormula": "exp(logScale) * exp(logShape)",
            "inclusiveWidthMultiplier": preparation.COMPONENT_SUPPORT_WIDTH_MULTIPLIER,
            "minimumPositiveWeightSamples": minimum, "zeroWeightsExcluded": True,
        },
        "recommendedNextTest": SUPPORT_FOLLOW_UP if unsupported_count else "BLIND_ANOMALY_MORPHOLOGY_REFINEMENT_AND_SUPPORT_REVIEW",
        "recommendation": (
            "Predeclare a follow-up search whose centers and effective widths account for positive-weight "
            "observational support in every applicable series. Preserve the original accepted winners and "
            "workload validity rules; verify new coverage and winners before model comparison."
        ),
        "interpretationStatement": (
            "Saved accepted winners alone do not establish whether a supported candidate exists elsewhere "
            "in the searched grid. No runner-up was verified or promoted. Numerical comparisons are "
            "retained even when support leaves scientific model preference unresolved."
        ),
    }


def _validate_impl(morphology_root: str | Path, *, coarse_project_root: str | Path,
                   coarse_investigation_record: str | Path, output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    _reject_symlink_components(output.parent, "output root")
    morphology = Path(morphology_root).expanduser().absolute()
    coarse = Path(coarse_project_root).expanduser().absolute()
    record = Path(coarse_investigation_record).expanduser().absolute()
    for path in (morphology, coarse, record):
        _reject_symlink_components(path, "input")
    for source in (morphology, coarse, record.parent):
        if output.resolve().is_relative_to(source.resolve()):
            raise _fail("output root must not be inside an input artifact directory")
    verified = builder._verify_preparation(morphology)
    for document in (verified.contract, verified.preparation, verified.manifest, *verified.series):
        _finite_tree(document)
    _verify_interpretation_contract(verified.contract)
    grid = _verify_grid_root(coarse, verified)
    investigation = _verify_investigation(record, grid, coarse / builder.PROJECT_RELATIVE_PATH)
    result = _report(verified, grid, investigation)
    result_bytes = _stable_json_bytes(result)
    manifest = {
        "artifactManifestSchemaID": MANIFEST_SCHEMA_ID, "artifactManifestVersion": RESULT_VERSION,
        "resultSchemaID": RESULT_SCHEMA_ID_REPORT, "resultVersion": RESULT_VERSION,
        "relativeArtifactPaths": {"result": RESULT_RELATIVE_PATH, "artifactManifest": MANIFEST_RELATIVE_PATH},
        "outputSHA256s": {RESULT_RELATIVE_PATH: _sha256_bytes(result_bytes)},
        "inputHashes": {
            "morphologyPreparation": verified.preparation_file_sha256,
            "morphologyContractFile": verified.contract_file_sha256,
            "morphologyContractCanonical": verified.contract_canonical_sha256,
            "morphologyArtifactManifest": verified.manifest_file_sha256,
            "morphologyDatasets": dict(verified.series_file_sha256s),
            "coarseProjectArtifacts": dict(grid.artifact_sha256s),
            "coarseInvestigationRecord": investigation.investigation_sha256,
            "coarseStageLedgers": dict(investigation.stage_ledger_sha256s),
        },
        "parentHashes": verified.preparation["parentHashes"],
        "parentIDs": verified.preparation["parentIDs"],
        "projectID": grid.project["id"], "investigationID": investigation.investigation_id,
        "runStageID": investigation.run_stage_id,
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    manifest_bytes = _stable_json_bytes(manifest)
    _assert_identity_free((result_bytes, manifest_bytes))
    builder._assert_identity_isolated((result, manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent, "output root")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _atomic_write_bytes(staging / RESULT_RELATIVE_PATH, result_bytes)
        _atomic_write_bytes(staging / MANIFEST_RELATIVE_PATH, manifest_bytes)
        if output.exists() or output.is_symlink():
            raise _fail("output root already exists")
        staging.rename(output)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {"result": result, "artifactManifest": manifest}


def validate_anomaly_morphology_coarse_grid(
    morphology_root: str | Path, *, coarse_project_root: str | Path,
    coarse_investigation_record: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    """Verify completed results and atomically publish a new report directory."""
    try:
        return _validate_impl(
            morphology_root, coarse_project_root=coarse_project_root,
            coarse_investigation_record=coarse_investigation_record, output_root=output_root,
        )
    except AnomalyMorphologyCoarseGridValidationError:
        raise
    except (KeyError, IndexError, OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise _fail(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphology-root", required=True, type=Path)
    parser.add_argument("--coarse-project-root", required=True, type=Path)
    parser.add_argument("--coarse-investigation-record", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    published = validate_anomaly_morphology_coarse_grid(**vars(arguments))
    print(f"classification: {published['result']['overallClassification']}")
    print(f"unsupported searches: {published['result']['unsupportedSearchCount']}")
    print(f"result: {arguments.output_root.expanduser().absolute() / RESULT_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
