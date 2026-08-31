"""Validate localized blind residual components across admitted series."""

from __future__ import annotations

import argparse
import hashlib
import math
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
    _agrees as _curve_grid_agrees,
    _evaluate_candidate,
)
from workflows.microlensing.coarse_grid import (
    CoarseGridBuildError,
    _assert_identity_free,
    _atomic_write_bytes,
    _canonical_compact_json_bytes,
    _decode_json,
    _read_regular_file,
    _stable_json_bytes,
)
from workflows.microlensing.prepare_residuals import (
    CONTRACT_RELATIVE_PATH as RESIDUAL_CONTRACT_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH as RESIDUAL_MANIFEST_RELATIVE_PATH,
    RESIDUAL_MANIFEST_SCHEMA_ID,
    RESIDUAL_MANIFEST_VERSION,
    RESIDUAL_PREPARATION_CONTRACT_ID,
    RESIDUAL_PREPARATION_CONTRACT_VERSION,
    RESIDUAL_SERIES_SCHEMA_ID,
    RESIDUAL_SERIES_VERSION,
    ResidualPreparationError,
    _fit_series,
    _residual_contract,
)
from workflows.microlensing.refine_grid import (
    PREPARE_HANDLER_ID,
    PROJECT_RUN_HANDLER_ID,
    SMOKE_WORKFLOW_ID,
    TERMINAL_CHECK_HANDLER_ID,
    RefinementGridBuildError,
    _INVESTIGATION_FIELDS,
    _stage_ledgers,
    _validate_stage_shape,
)
from workflows.microlensing.residual_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as GRID_BUILD_MANIFEST_RELATIVE_PATH,
    BUILD_MANIFEST_SCHEMA_ID as GRID_BUILD_MANIFEST_SCHEMA_ID,
    BUILD_MANIFEST_VERSION as GRID_BUILD_MANIFEST_VERSION,
    CANDIDATES_PER_DATASET,
    CANDIDATES_PER_WORK_UNIT,
    CONTRACT_RELATIVE_PATH as GRID_CONTRACT_RELATIVE_PATH,
    DATASET_DIRECTORY,
    PROJECT_RELATIVE_PATH as GRID_PROJECT_RELATIVE_PATH,
    RESIDUAL_SEARCH_CONTRACT_ID,
    RESIDUAL_SEARCH_CONTRACT_VERSION,
    WORK_UNITS_PER_DATASET,
    ResidualGridBuildError,
    _admission_record as _expected_admission_record,
    _contract as _expected_grid_contract,
    _dataset as _expected_grid_dataset,
    _project as _expected_grid_project,
    _regular_directory as _grid_regular_directory,
    _reject_symlink_components as _grid_reject_symlink_components,
    _search_geometry,
)
from workflows.microlensing.second_recenter_grid import (
    CENTER_COUNT as SECOND_RECENTER_CENTER_COUNT,
    LOG_SCALE_COUNT as SECOND_RECENTER_LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT as SECOND_RECENTER_LOG_SHAPE_COUNT,
)


CROSS_VALIDATION_CONTRACT_ID = (
    "openstar.microlensing-residual-cross-validation-contract.v1"
)
CROSS_VALIDATION_CONTRACT_VERSION = "1.0"
CROSS_VALIDATION_RESULT_SCHEMA_ID = (
    "openstar.microlensing-residual-cross-validation.v1"
)
CROSS_VALIDATION_RESULT_VERSION = "1.0"

DISCOVERY_DELTA_WRSS_THRESHOLD = 30.0
VALIDATION_DELTA_WRSS_THRESHOLD = 9.0
MINIMUM_TWO_WIDTH_SUPPORT = 1

CONTRACT_RELATIVE_PATH = "residual-cross-validation-contract.json"
RESULT_RELATIVE_PATH = "residual-cross-validation.json"

CONFIRMED_STATUS = "CROSS_SERIES_CONFIRMED"
UNCONFIRMED_STATUS = "NOT_CROSS_SERIES_CONFIRMED"
POSITIVE_CLASSIFICATION = "REPRODUCIBLE_LOCALIZED_RESIDUAL_STRUCTURE"
NEGATIVE_CLASSIFICATION = "NO_REPRODUCIBLE_LOCALIZED_RESIDUAL_STRUCTURE"
CONFIRMED_NEXT_TEST = "BLIND_MICROLENSING_ANOMALY_MORPHOLOGY_MODELING"
UNCONFIRMED_NEXT_TEST = "RESIDUAL_SYSTEMATICS_AND_ERROR_MODEL_REVIEW"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_RESIDUAL_MANIFEST_FIELDS = frozenset(
    (
        "canonicalCurveFamilyID",
        "contractID",
        "contractSHA256",
        "contractVersion",
        "convergenceEvidence",
        "frozenGeometry",
        "geometryProvenanceSHA256",
        "identityIsolationStatement",
        "modelScopeStatement",
        "orderedGenericSeriesIDs",
        "parentArtifactHashes",
        "parentInvestigationIDs",
        "parentProjectIDs",
        "preparationManifestSHA256",
        "residualManifestSchemaID",
        "residualManifestVersion",
        "series",
        "totalSampleCount",
        "totalSeriesCount",
        "totalWeightedResidualSumSquares",
        "verifiedFirstRecenterWinner",
        "verifiedSecondRecenterWinner",
    )
)
_RESIDUAL_MANIFEST_SERIES_FIELDS = frozenset(
    (
        "genericSeriesID",
        "inputSeriesSHA256",
        "outputFile",
        "outputSHA256",
        "sampleCount",
        "weightedResidualSumSquares",
    )
)
_RESIDUAL_SERIES_FIELDS = frozenset(
    (
        "canonicalCurveFamilyID",
        "coordinates",
        "fitDiagnostics",
        "fittedAmplitude",
        "fittedOffset",
        "frozenGeometry",
        "geometryProvenanceSHA256",
        "genericSeriesID",
        "inputSeriesSHA256",
        "inverseVariances",
        "modelValues",
        "observedValues",
        "residualPreparationContractID",
        "residualPreparationContractSHA256",
        "residualSeriesSchemaID",
        "residualSeriesVersion",
        "residualValues",
        "sampleCount",
    )
)
_FIT_DIAGNOSTIC_FIELDS = frozenset(
    (
        "amplitudeSign",
        "maximumAbsoluteStandardizedResidual",
        "maximumAbsoluteStandardizedResidualCoordinate",
        "maximumAbsoluteStandardizedResidualIndex",
        "maximumTieRule",
        "positiveWeightSampleCount",
        "weightedResidualSumSquares",
    )
)
_GRID_PROJECT_FIELDS = frozenset(
    (
        "datasetSchemaID",
        "datasets",
        "id",
        "payloadSchemaID",
        "resultSchemaID",
        "workloadID",
    )
)
_GRID_PROJECT_DATASET_FIELDS = frozenset(("id", "path"))
_WINNER_FIELDS = frozenset(
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
_ANCESTRY_WINNER_FIELDS = frozenset(
    (
        "acceptedResultPayload",
        "bestAmplitude",
        "bestCenter",
        "bestGridIndex",
        "bestLogScale",
        "bestLogShape",
        "bestOffset",
        "bestWeightedResidualSumSquares",
        "boundaryAxes",
        "centerIndex",
        "logScaleIndex",
        "logShapeIndex",
    )
)
_ANCESTRY_KEYS = frozenset(
    ("coarse", "firstRecenter", "firstRefinement", "secondRecenter")
)


class ResidualGridValidationError(RuntimeError):
    """The blind residual cross-series validation cannot be trusted."""


@dataclass(frozen=True, slots=True)
class _VerifiedResiduals:
    contract: Mapping[str, Any]
    manifest: Mapping[str, Any]
    series: tuple[Mapping[str, Any], ...]
    contract_file_sha256: str
    manifest_file_sha256: str
    series_file_sha256s: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _VerifiedGrid:
    contract: Mapping[str, Any]
    build_manifest: Mapping[str, Any]
    project: Mapping[str, Any]
    datasets: tuple[Mapping[str, Any], ...]
    generic_series_ids: tuple[str, ...]
    contract_file_sha256: str
    build_manifest_sha256: str
    project_sha256: str
    dataset_sha256s: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _VerifiedWinner:
    dataset_id: str
    generic_series_id: str
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


@dataclass(frozen=True, slots=True)
class _VerifiedInvestigation:
    investigation_id: str
    investigation_sha256: str
    run_stage_id: str
    run_stage_ledger_sha256: str
    stage_ledger_sha256s: Mapping[str, str]
    winners: tuple[_VerifiedWinner, ...]


def _fail(message: str) -> ResidualGridValidationError:
    return ResidualGridValidationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _exact_count(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > MAX_SAFE_INTEGER:
        raise _fail(f"{field_name} must be a nonnegative safe integer")
    return value


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field_name} must be a nonempty string")
    return value


def _sha256_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{field_name} must be a lowercase SHA-256")
    return value


def _safe_sum(values: Sequence[int], field_name: str) -> int:
    total = 0
    for value in values:
        count = _exact_count(value, field_name)
        if count > MAX_SAFE_INTEGER - total:
            raise _fail(f"{field_name} exceeds the safe integer range")
        total += count
    return total


def _amplitude_sign(value: float) -> str:
    return "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"


def _discovery_gate(delta_wrss: Any) -> bool:
    return (
        _finite_number(delta_wrss, "discovery delta WRSS")
        >= DISCOVERY_DELTA_WRSS_THRESHOLD
    )


def _overall_classification(confirmed_component_count: Any) -> str:
    count = _exact_count(confirmed_component_count, "confirmed component count")
    return POSITIVE_CLASSIFICATION if count > 0 else NEGATIVE_CLASSIFICATION


def _reject_symlink_components(path: Path, description: str) -> None:
    try:
        _grid_reject_symlink_components(path, description)
    except ResidualGridBuildError as error:
        raise _fail(str(error)) from error


def _regular_directory(path: Path, description: str) -> Path:
    try:
        return _grid_regular_directory(path, description)
    except ResidualGridBuildError as error:
        raise _fail(str(error)) from error


def _read_json_file(path: Path, description: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = _read_regular_file(path, description)
        decoded = _decode_json(payload, description)
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error
    if payload != _stable_json_bytes(decoded):
        raise _fail(f"{description} is not canonical deterministic JSON")
    return payload, decoded


def _validate_hash_tree(value: Any, field_name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise _fail(f"{field_name} must be a nonempty mapping")
    for key, item in value.items():
        _nonempty_string(key, f"{field_name} key")
        if isinstance(item, Mapping):
            _validate_hash_tree(item, f"{field_name}.{key}")
        elif key.endswith("ID"):
            _nonempty_string(item, f"{field_name}.{key}")
        else:
            _sha256_string(item, f"{field_name}.{key}")


def _validate_id_map(value: Any, field_name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise _fail(f"{field_name} must be a nonempty mapping")
    for key, item in value.items():
        _nonempty_string(key, f"{field_name} key")
        identifier = _nonempty_string(item, f"{field_name}.{key}")
        if _SAFE_ID.fullmatch(identifier) is None:
            raise _fail(f"{field_name}.{key} is unsafe")


def _verified_frozen_geometry(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "center",
        "logScale",
        "logShape",
        "scale",
        "shape",
    }:
        raise _fail("frozen geometry field set is invalid")
    center = _finite_number(value.get("center"), "frozen center")
    log_scale = _finite_number(value.get("logScale"), "frozen log scale")
    log_shape = _finite_number(value.get("logShape"), "frozen log shape")
    scale = _finite_number(value.get("scale"), "frozen scale")
    shape = _finite_number(value.get("shape"), "frozen shape")
    try:
        if scale != math.exp(log_scale) or shape != math.exp(log_shape):
            raise _fail("frozen geometry does not match its logarithms")
    except OverflowError as error:
        raise _fail("frozen geometry exponentiation is invalid") from error
    if scale <= 0.0 or shape <= 0.0:
        raise _fail("frozen scale and shape must be positive")
    return {
        "center": center,
        "logScale": log_scale,
        "logShape": log_shape,
        "scale": scale,
        "shape": shape,
    }


def _verify_convergence(manifest: Mapping[str, Any]) -> None:
    evidence = manifest.get("convergenceEvidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "comparedFields",
        "exactEquality",
        "firstRecenterWinner",
        "secondRecenterInteriorOnEveryAxis",
        "secondRecenterWinner",
    }:
        raise _fail("residual convergence evidence is malformed")
    if evidence.get("comparedFields") != [
        "bestCenter",
        "bestLogScale",
        "bestLogShape",
        "bestWeightedResidualSumSquares",
    ]:
        raise _fail("residual convergence compared fields are invalid")
    if evidence.get("exactEquality") is not True:
        raise _fail("residual convergence equality is not established")
    if evidence.get("secondRecenterInteriorOnEveryAxis") is not True:
        raise _fail("residual convergence winner is not interior")
    first = evidence.get("firstRecenterWinner")
    second = evidence.get("secondRecenterWinner")
    for label, winner in (("first", first), ("second", second)):
        if not isinstance(winner, Mapping) or set(winner) != (
            _ANCESTRY_WINNER_FIELDS
        ):
            raise _fail(f"residual {label}-recenter winner is malformed")
        for field_name in (
            "bestAmplitude",
            "bestCenter",
            "bestLogScale",
            "bestLogShape",
            "bestOffset",
            "bestWeightedResidualSumSquares",
        ):
            _finite_number(winner.get(field_name), f"{label} {field_name}")
        for field_name in (
            "bestGridIndex",
            "centerIndex",
            "logScaleIndex",
            "logShapeIndex",
        ):
            _exact_count(winner.get(field_name), f"{label} {field_name}")
        boundary_axes = winner.get("boundaryAxes")
        if (
            not isinstance(boundary_axes, list)
            or len(set(boundary_axes)) != len(boundary_axes)
            or any(
                axis not in {"center", "logScale", "logShape"}
                for axis in boundary_axes
            )
        ):
            raise _fail(f"residual {label}-recenter boundary axes are invalid")
        accepted = winner.get("acceptedResultPayload")
        if not isinstance(accepted, Mapping) or set(accepted) != _WINNER_FIELDS:
            raise _fail(f"residual {label}-recenter payload is malformed")
        if accepted.get("familyID") != FAMILY_ID:
            raise _fail(f"residual {label}-recenter family is invalid")
        if any(
            accepted.get(payload_field) != winner.get(winner_field)
            for payload_field, winner_field in (
                ("bestAmplitude", "bestAmplitude"),
                ("bestCenter", "bestCenter"),
                ("bestGridIndex", "bestGridIndex"),
                ("bestLogScale", "bestLogScale"),
                ("bestLogShape", "bestLogShape"),
                ("bestOffset", "bestOffset"),
                (
                    "bestWeightedResidualSumSquares",
                    "bestWeightedResidualSumSquares",
                ),
            )
        ):
            raise _fail(f"residual {label}-recenter payload disagrees")
    fields = (
        "bestCenter",
        "bestLogScale",
        "bestLogShape",
        "bestWeightedResidualSumSquares",
    )
    first_values = tuple(_finite_number(first.get(key), key) for key in fields)
    second_values = tuple(_finite_number(second.get(key), key) for key in fields)
    if first_values != second_values:
        raise _fail("residual convergence winners differ")
    if first_values[-1] < 0.0:
        raise _fail("residual converged objective must be nonnegative")
    second_counts = (
        SECOND_RECENTER_CENTER_COUNT,
        SECOND_RECENTER_LOG_SCALE_COUNT,
        SECOND_RECENTER_LOG_SHAPE_COUNT,
    )
    for label, winner in (("first", first), ("second", second)):
        indices = (
            winner["centerIndex"],
            winner["logScaleIndex"],
            winner["logShapeIndex"],
        )
        if any(index >= count for index, count in zip(indices, second_counts)):
            raise _fail(f"residual {label}-recenter index is outside its grid")
        expected_grid_index = (
            (
                winner["centerIndex"] * SECOND_RECENTER_LOG_SCALE_COUNT
                + winner["logScaleIndex"]
            )
            * SECOND_RECENTER_LOG_SHAPE_COUNT
            + winner["logShapeIndex"]
        )
        if winner["bestGridIndex"] != expected_grid_index:
            raise _fail(f"residual {label}-recenter indices are inconsistent")
    second_indices = (
        second["centerIndex"],
        second["logScaleIndex"],
        second["logShapeIndex"],
    )
    if second.get("boundaryAxes") != [] or any(
        index in {0, count - 1}
        for index, count in zip(second_indices, second_counts)
    ):
        raise _fail("residual second-recenter winner is on a boundary")
    if manifest.get("verifiedFirstRecenterWinner") != first:
        raise _fail("verified first-recenter winner disagrees with convergence")
    if manifest.get("verifiedSecondRecenterWinner") != second:
        raise _fail("verified second-recenter winner disagrees with convergence")


def _verify_residual_root(root: Path) -> _VerifiedResiduals:
    residual_root = _regular_directory(root, "residual preparation root")
    try:
        root_names = {entry.name for entry in residual_root.iterdir()}
    except OSError as error:
        raise _fail("residual preparation root is unreadable") from error
    if root_names != {
        RESIDUAL_CONTRACT_RELATIVE_PATH,
        RESIDUAL_MANIFEST_RELATIVE_PATH,
        "series",
    }:
        raise _fail("residual preparation artifact set is incomplete or unexpected")

    contract_bytes, contract = _read_json_file(
        residual_root / RESIDUAL_CONTRACT_RELATIVE_PATH,
        "residual preparation contract",
    )
    expected_contract = _residual_contract()
    if contract != expected_contract:
        raise _fail("residual preparation contract is invalid")
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))

    manifest_bytes, manifest = _read_json_file(
        residual_root / RESIDUAL_MANIFEST_RELATIVE_PATH,
        "residual preparation manifest",
    )
    if set(manifest) != _RESIDUAL_MANIFEST_FIELDS:
        raise _fail("residual preparation manifest field set is invalid")
    required_manifest_values = {
        "canonicalCurveFamilyID": FAMILY_ID,
        "contractID": RESIDUAL_PREPARATION_CONTRACT_ID,
        "contractSHA256": contract_sha256,
        "contractVersion": RESIDUAL_PREPARATION_CONTRACT_VERSION,
        "identityIsolationStatement": contract["identityIsolationStatement"],
        "modelScopeStatement": contract["modelScopeStatement"],
        "residualManifestSchemaID": RESIDUAL_MANIFEST_SCHEMA_ID,
        "residualManifestVersion": RESIDUAL_MANIFEST_VERSION,
    }
    if any(
        manifest.get(key) != value
        for key, value in required_manifest_values.items()
    ):
        raise _fail("residual preparation manifest schema or contract is invalid")
    geometry = _verified_frozen_geometry(manifest.get("frozenGeometry"))
    _verify_convergence(manifest)
    geometry_provenance_sha256 = _sha256_string(
        manifest.get("geometryProvenanceSHA256"),
        "geometry provenance SHA-256",
    )
    _sha256_string(
        manifest.get("preparationManifestSHA256"),
        "preparation manifest SHA-256",
    )
    parent_hashes = manifest.get("parentArtifactHashes")
    parent_project_ids = manifest.get("parentProjectIDs")
    parent_investigation_ids = manifest.get("parentInvestigationIDs")
    _validate_hash_tree(parent_hashes, "parent hashes")
    _validate_id_map(parent_project_ids, "parent project IDs")
    _validate_id_map(
        parent_investigation_ids,
        "parent investigation IDs",
    )
    if (
        not isinstance(parent_hashes, Mapping)
        or set(parent_hashes)
        != {*_ANCESTRY_KEYS, "preparationManifestSHA256"}
        or not isinstance(parent_project_ids, Mapping)
        or set(parent_project_ids) != _ANCESTRY_KEYS
        or not isinstance(parent_investigation_ids, Mapping)
        or set(parent_investigation_ids) != _ANCESTRY_KEYS
    ):
        raise _fail("residual ancestry is incomplete or unexpected")
    expected_geometry_provenance_sha256 = _sha256_bytes(
        _canonical_compact_json_bytes(
            {
                "convergenceEvidence": manifest["convergenceEvidence"],
                "curveFamilyID": FAMILY_ID,
                "parentArtifactHashes": parent_hashes,
                "parentInvestigationIDs": parent_investigation_ids,
                "parentProjectIDs": parent_project_ids,
            }
        )
    )
    if geometry_provenance_sha256 != expected_geometry_provenance_sha256:
        raise _fail("residual geometry provenance hash is invalid")

    ordered_ids = manifest.get("orderedGenericSeriesIDs")
    records = manifest.get("series")
    if (
        not isinstance(ordered_ids, list)
        or not ordered_ids
        or any(not isinstance(value, str) or not value for value in ordered_ids)
        or ordered_ids != sorted(ordered_ids)
        or len(set(ordered_ids)) != len(ordered_ids)
        or ordered_ids
        != [f"series-{ordinal:03d}" for ordinal in range(1, len(ordered_ids) + 1)]
    ):
        raise _fail("ordered residual generic series IDs are invalid")
    if not isinstance(records, list) or len(records) != len(ordered_ids):
        raise _fail("residual manifest series records are incomplete")
    series_root = _regular_directory(
        residual_root / "series", "residual preparation series directory"
    )
    expected_names = {
        f"residual-series-{ordinal:03d}.json"
        for ordinal in range(1, len(records) + 1)
    }
    try:
        if {entry.name for entry in series_root.iterdir()} != expected_names:
            raise _fail("residual series artifact set is incomplete or unexpected")
    except OSError as error:
        raise _fail("residual series directory is unreadable") from error

    documents: list[Mapping[str, Any]] = []
    file_hashes: dict[str, str] = {}
    sample_counts: list[int] = []
    wrss_values: list[float] = []
    identity_payloads: list[bytes] = [contract_bytes, manifest_bytes]
    for ordinal, (generic_id, record) in enumerate(
        zip(ordered_ids, records), start=1
    ):
        if not isinstance(record, Mapping) or set(record) != (
            _RESIDUAL_MANIFEST_SERIES_FIELDS
        ):
            raise _fail("residual manifest series record is malformed")
        relative_path = f"series/residual-series-{ordinal:03d}.json"
        if (
            record.get("genericSeriesID") != generic_id
            or record.get("outputFile") != relative_path
        ):
            raise _fail("residual series order or path is invalid")
        series_bytes, document = _read_json_file(
            residual_root / relative_path,
            f"residual series {generic_id}",
        )
        series_sha256 = _sha256_bytes(series_bytes)
        if record.get("outputSHA256") != series_sha256:
            raise _fail("residual series hash does not match its manifest")
        if set(document) != _RESIDUAL_SERIES_FIELDS:
            raise _fail("residual series field set is invalid")
        if (
            document.get("genericSeriesID") != generic_id
            or document.get("canonicalCurveFamilyID") != FAMILY_ID
            or document.get("residualPreparationContractID")
            != RESIDUAL_PREPARATION_CONTRACT_ID
            or document.get("residualPreparationContractSHA256")
            != contract_sha256
            or document.get("residualSeriesSchemaID")
            != RESIDUAL_SERIES_SCHEMA_ID
            or document.get("residualSeriesVersion") != RESIDUAL_SERIES_VERSION
            or document.get("frozenGeometry") != geometry
            or document.get("geometryProvenanceSHA256")
            != manifest["geometryProvenanceSHA256"]
            or document.get("inputSeriesSHA256") != record.get("inputSeriesSHA256")
        ):
            raise _fail("residual series schema or provenance is invalid")
        _sha256_string(document.get("inputSeriesSHA256"), "input series SHA-256")
        sample_count = _exact_count(document.get("sampleCount"), "sample count")
        if sample_count < 2 or record.get("sampleCount") != sample_count:
            raise _fail("residual series sample count is invalid")
        arrays = []
        for field_name in (
            "coordinates",
            "observedValues",
            "inverseVariances",
            "modelValues",
            "residualValues",
        ):
            values = document.get(field_name)
            if not isinstance(values, list) or len(values) != sample_count:
                raise _fail(f"residual series {field_name} is malformed")
            arrays.append(
                tuple(
                    _finite_number(item, f"{field_name}[{index}]")
                    for index, item in enumerate(values)
                )
            )
        coordinates, observed, weights, models, residuals = arrays
        if any(weight < 0.0 for weight in weights):
            raise _fail("residual inverse variances must be nonnegative")
        fit = _fit_series(
            coordinates,
            observed,
            weights,
            center=geometry["center"],
            scale=geometry["scale"],
            shape=geometry["shape"],
        )
        maximum_index = fit.maximum_absolute_standardized_residual_index
        diagnostics = document.get("fitDiagnostics")
        expected_diagnostics = {
            "amplitudeSign": _amplitude_sign(fit.amplitude),
            "maximumAbsoluteStandardizedResidual": (
                fit.maximum_absolute_standardized_residual
            ),
            "maximumAbsoluteStandardizedResidualCoordinate": coordinates[
                maximum_index
            ],
            "maximumAbsoluteStandardizedResidualIndex": maximum_index,
            "maximumTieRule": contract["maximumResidualTieRule"],
            "positiveWeightSampleCount": fit.positive_weight_sample_count,
            "weightedResidualSumSquares": fit.weighted_residual_sum_squares,
        }
        if (
            not isinstance(diagnostics, Mapping)
            or set(diagnostics) != _FIT_DIAGNOSTIC_FIELDS
            or diagnostics != expected_diagnostics
            or document.get("fittedOffset") != fit.offset
            or document.get("fittedAmplitude") != fit.amplitude
            or models != fit.model_values
            or residuals != fit.residual_values
            or any(
                residual != value - model
                for residual, value, model in zip(residuals, observed, models)
            )
            or record.get("weightedResidualSumSquares")
            != fit.weighted_residual_sum_squares
        ):
            raise _fail("residual series fit does not reproduce exactly")
        documents.append(dict(document))
        file_hashes[relative_path] = series_sha256
        sample_counts.append(sample_count)
        wrss_values.append(fit.weighted_residual_sum_squares)
        identity_payloads.append(series_bytes)

    if (
        manifest.get("totalSeriesCount") != len(documents)
        or manifest.get("totalSampleCount")
        != _safe_sum(sample_counts, "total residual sample count")
    ):
        raise _fail("residual manifest count totals are invalid")
    total_wrss = 0.0
    for value in wrss_values:
        total_wrss += value
        if not math.isfinite(total_wrss):
            raise _fail("residual manifest WRSS total is non-finite")
    if manifest.get("totalWeightedResidualSumSquares") != total_wrss:
        raise _fail("residual manifest WRSS total is invalid")
    try:
        _assert_identity_free(tuple(identity_payloads))
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error
    return _VerifiedResiduals(
        contract=dict(contract),
        manifest=dict(manifest),
        series=tuple(documents),
        contract_file_sha256=_sha256_bytes(contract_bytes),
        manifest_file_sha256=_sha256_bytes(manifest_bytes),
        series_file_sha256s=dict(sorted(file_hashes.items())),
    )


def _verify_grid_root(root: Path, residuals: _VerifiedResiduals) -> _VerifiedGrid:
    grid_root = _regular_directory(root, "residual-grid project root")
    try:
        if {entry.name for entry in grid_root.iterdir()} != {
            GRID_CONTRACT_RELATIVE_PATH,
            GRID_BUILD_MANIFEST_RELATIVE_PATH,
            GRID_PROJECT_RELATIVE_PATH,
            DATASET_DIRECTORY,
        }:
            raise _fail("residual-grid artifact set is incomplete or unexpected")
    except OSError as error:
        raise _fail("residual-grid root is unreadable") from error

    frozen_geometry, axes, core_width = _search_geometry(
        residuals.manifest["frozenGeometry"]
    )
    expected_contract = _expected_grid_contract(
        frozen_geometry, axes, core_width
    )
    contract_bytes, contract = _read_json_file(
        grid_root / GRID_CONTRACT_RELATIVE_PATH,
        "residual-grid contract",
    )
    if contract != expected_contract:
        raise _fail("residual-grid contract does not match residual ancestry")
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))

    build_bytes, build = _read_json_file(
        grid_root / GRID_BUILD_MANIFEST_RELATIVE_PATH,
        "residual-grid build manifest",
    )
    project_bytes, project = _read_json_file(
        grid_root / GRID_PROJECT_RELATIVE_PATH,
        "residual-grid project",
    )
    if set(project) != _GRID_PROJECT_FIELDS:
        raise _fail("residual-grid project field set is invalid")
    project_id = _nonempty_string(project.get("id"), "residual-grid project ID")
    if _SAFE_ID.fullmatch(project_id) is None:
        raise _fail("residual-grid project ID is unsafe")

    interval_minimum = frozen_geometry["center"] - 4.0 * core_width
    interval_maximum = frozen_geometry["center"] + 4.0 * core_width
    admission_records: list[dict[str, Any]] = []
    admitted: list[tuple[int, Mapping[str, Any], str]] = []
    for ordinal, source in enumerate(residuals.series, start=1):
        record = _expected_admission_record(
            source["genericSeriesID"],
            source["coordinates"],
            source["inverseVariances"],
            interval_minimum=interval_minimum,
            interval_maximum=interval_maximum,
        )
        admission_records.append(record)
        if record["admissionDecision"] == "ADMITTED":
            admitted.append(
                (ordinal, source, f"series/residual-series-{ordinal:03d}.json")
            )
    if not admitted:
        raise _fail("residual-grid project has no deterministically admitted series")

    dataset_root = _regular_directory(
        grid_root / DATASET_DIRECTORY, "residual-grid dataset directory"
    )
    expected_dataset_names = {
        f"residual-series-{ordinal:03d}.json"
        for ordinal in range(1, len(admitted) + 1)
    }
    try:
        if {entry.name for entry in dataset_root.iterdir()} != expected_dataset_names:
            raise _fail("residual-grid dataset set is incomplete or unexpected")
    except OSError as error:
        raise _fail("residual-grid dataset directory is unreadable") from error

    datasets: list[Mapping[str, Any]] = []
    dataset_hashes: dict[str, str] = {}
    project_references: list[dict[str, str]] = []
    dataset_records: list[dict[str, Any]] = []
    evaluation_counts: list[int] = []
    for admitted_ordinal, (_, source, source_path) in enumerate(admitted, start=1):
        dataset_id = f"{project_id}.residual-series-{admitted_ordinal:03d}"
        relative_path = (
            f"{DATASET_DIRECTORY}/residual-series-{admitted_ordinal:03d}.json"
        )
        expected_dataset = _expected_grid_dataset(
            dataset_id=dataset_id,
            source=source,
            axes=axes,
            residual_manifest_sha256=residuals.manifest_file_sha256,
            residual_search_contract_sha256=contract_sha256,
            residual_series_file_sha256=residuals.series_file_sha256s[
                source_path
            ],
        )
        dataset_bytes, dataset = _read_json_file(
            grid_root / relative_path, f"residual-grid dataset {dataset_id}"
        )
        if dataset != expected_dataset:
            raise _fail("residual-grid dataset does not match verified residuals")
        try:
            CURVE_GRID_PLUGIN.validate_dataset(dataset)
        except (RuntimeError, TypeError, ValueError, OverflowError) as error:
            raise _fail(f"residual-grid dataset is invalid: {error}") from error
        dataset_sha256 = _sha256_bytes(dataset_bytes)
        sample_count = len(dataset["coordinates"])
        evaluation_count = sample_count * CANDIDATES_PER_DATASET
        if evaluation_count > MAX_SAFE_INTEGER:
            raise _fail("residual-grid evaluation count exceeds safe range")
        datasets.append(dict(dataset))
        dataset_hashes[relative_path] = dataset_sha256
        project_references.append({"id": dataset_id, "path": relative_path})
        evaluation_counts.append(evaluation_count)
        dataset_records.append(
            {
                "candidateCount": CANDIDATES_PER_DATASET,
                "datasetID": dataset_id,
                "expectedSampleCandidateEvaluationCount": evaluation_count,
                "expectedWorkUnitCount": WORK_UNITS_PER_DATASET,
                "genericSeriesID": source["genericSeriesID"],
                "outputFile": relative_path,
                "outputSHA256": dataset_sha256,
                "sampleCount": sample_count,
                "sourceResidualSeriesFileSHA256": (
                    residuals.series_file_sha256s[source_path]
                ),
            }
        )
    expected_project = _expected_grid_project(project_id, project_references)
    if project != expected_project:
        raise _fail("residual-grid project does not match admitted datasets")

    dataset_count = len(datasets)
    expected_build = {
        "admissionRecords": admission_records,
        "axisDerivationRules": contract["axisDerivationRules"],
        "buildManifestSchemaID": GRID_BUILD_MANIFEST_SCHEMA_ID,
        "buildManifestVersion": GRID_BUILD_MANIFEST_VERSION,
        "candidateCountPerDataset": CANDIDATES_PER_DATASET,
        "canonicalCurveFamilyID": FAMILY_ID,
        "derivedCoreWidth": core_width,
        "datasets": dataset_records,
        "frozenGeometry": frozen_geometry,
        "identityIsolationStatement": contract["identityIsolationStatement"],
        "modelScopeStatement": contract["modelScopeStatement"],
        "orderedAdmittedDatasetIDs": [item["datasetID"] for item in dataset_records],
        "outputHashes": {
            "datasets": {
                item["outputFile"]: item["outputSHA256"]
                for item in dataset_records
            },
            "project": _sha256_bytes(project_bytes),
            "residualSearchContractFile": _sha256_bytes(contract_bytes),
        },
        "parentArtifactHashes": residuals.manifest["parentArtifactHashes"],
        "parentInvestigationIDs": residuals.manifest["parentInvestigationIDs"],
        "parentProjectIDs": residuals.manifest["parentProjectIDs"],
        "preparationManifestSHA256": residuals.manifest[
            "preparationManifestSHA256"
        ],
        "projectID": project_id,
        "publishedAxes": {key: dict(value) for key, value in axes.items()},
        "relativeArtifactPaths": {
            "buildManifest": GRID_BUILD_MANIFEST_RELATIVE_PATH,
            "datasets": [item["outputFile"] for item in dataset_records],
            "project": GRID_PROJECT_RELATIVE_PATH,
            "residualSearchContract": GRID_CONTRACT_RELATIVE_PATH,
        },
        "residualPreparationContractCanonicalSHA256": residuals.manifest[
            "contractSHA256"
        ],
        "residualPreparationContractFileSHA256": (
            residuals.contract_file_sha256
        ),
        "residualPreparationManifestFileSHA256": residuals.manifest_file_sha256,
        "residualVerificationRule": contract["residualVerificationRule"],
        "verifiedResidualSeriesFileSHA256s": dict(
            residuals.series_file_sha256s
        ),
        "residualSearchContractID": RESIDUAL_SEARCH_CONTRACT_ID,
        "residualSearchContractSHA256": contract_sha256,
        "residualSearchContractVersion": RESIDUAL_SEARCH_CONTRACT_VERSION,
        "totalAdmittedDatasetCount": dataset_count,
        "totalCandidateCount": dataset_count * CANDIDATES_PER_DATASET,
        "totalExpectedSampleCandidateEvaluationCount": _safe_sum(
            evaluation_counts, "total sample-candidate evaluation count"
        ),
        "totalExpectedWorkUnitCount": dataset_count * WORK_UNITS_PER_DATASET,
        "verifiedConvergenceEvidence": residuals.manifest["convergenceEvidence"],
        "workUnitsPerDataset": WORK_UNITS_PER_DATASET,
    }
    if build != expected_build:
        raise _fail("residual-grid build manifest does not reconstruct exactly")
    try:
        _assert_identity_free(
            (
                contract_bytes,
                build_bytes,
                project_bytes,
                *(
                    _stable_json_bytes(dataset)
                    for dataset in datasets
                ),
            )
        )
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error
    return _VerifiedGrid(
        contract=dict(contract),
        build_manifest=dict(build),
        project=dict(project),
        datasets=tuple(datasets),
        generic_series_ids=tuple(
            item["genericSeriesID"] for item in dataset_records
        ),
        contract_file_sha256=_sha256_bytes(contract_bytes),
        build_manifest_sha256=_sha256_bytes(build_bytes),
        project_sha256=_sha256_bytes(project_bytes),
        dataset_sha256s=dict(sorted(dataset_hashes.items())),
    )


def _winner_from_status(
    status: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> _VerifiedWinner:
    payload = status.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"best"}:
        raise _fail("residual-grid dataset result payload is invalid")
    best = payload.get("best")
    if not isinstance(best, Mapping) or set(best) != _WINNER_FIELDS:
        raise _fail("residual-grid winning result field set is invalid")
    if best.get("familyID") != FAMILY_ID:
        raise _fail("residual-grid winning result family is invalid")
    grid_index = _exact_count(best.get("bestGridIndex"), "best grid index")
    if grid_index >= CANDIDATES_PER_DATASET:
        raise _fail("residual-grid winner is outside the published grid")
    grid = dataset["curveGrid"]
    shape_count = grid["logShapeAxis"]["count"]
    scale_count = grid["logScaleAxis"]["count"]
    combined, shape_index = divmod(grid_index, shape_count)
    center_index, scale_index = divmod(combined, scale_count)
    center = _finite_number(best.get("bestCenter"), "best center")
    log_scale = _finite_number(best.get("bestLogScale"), "best log scale")
    log_shape = _finite_number(best.get("bestLogShape"), "best log shape")
    offset = _finite_number(best.get("bestOffset"), "best offset")
    amplitude = _finite_number(best.get("bestAmplitude"), "best amplitude")
    objective = _finite_number(
        best.get("bestWeightedResidualSumSquares"), "best WRSS"
    )
    if objective < 0.0:
        raise _fail("residual-grid winning WRSS must be nonnegative")
    expected_geometry = (
        grid["centerAxis"]["start"]
        + center_index * grid["centerAxis"]["step"],
        grid["logScaleAxis"]["start"]
        + scale_index * grid["logScaleAxis"]["step"],
        grid["logShapeAxis"]["start"]
        + shape_index * grid["logShapeAxis"]["step"],
    )
    if (center, log_scale, log_shape) != expected_geometry:
        raise _fail("residual-grid winner is not on its published grid")
    shard_start = (
        grid_index // CANDIDATES_PER_WORK_UNIT
    ) * CANDIDATES_PER_WORK_UNIT
    shard_count = min(
        CANDIDATES_PER_WORK_UNIT,
        CANDIDATES_PER_DATASET - shard_start,
    )
    if (
        best.get("gridStartIndex") != shard_start
        or best.get("gridCount") != shard_count
        or best.get("evaluatedCandidateCount") != shard_count
        or type(best.get("invalidCandidateCount")) is not int
        or best.get("invalidCandidateCount") < 0
        or best.get("invalidCandidateCount") >= shard_count
    ):
        raise _fail("residual-grid winning shard accounting is invalid")
    aggregate = (
        status.get("bestGridIndex"),
        status.get("bestCenter"),
        status.get("bestLogScale"),
        status.get("bestLogShape"),
        status.get("bestOffset"),
        status.get("bestAmplitude"),
        status.get("bestWeightedResidualSumSquares"),
    )
    if aggregate != (
        grid_index,
        center,
        log_scale,
        log_shape,
        offset,
        amplitude,
        objective,
    ):
        raise _fail("residual-grid aggregate and nested winners disagree")
    evaluated = _evaluate_candidate(dataset, grid_index)
    if evaluated is None:
        raise _fail("canonical CurveGrid evaluator rejects the accepted winner")
    recomputed = (
        evaluated.center,
        evaluated.log_scale,
        evaluated.log_shape,
        evaluated.offset,
        evaluated.amplitude,
        evaluated.weighted_residual_sum_squares,
    )
    accepted = (center, log_scale, log_shape, offset, amplitude, objective)
    if not all(
        actual == expected if index < 3 else _curve_grid_agrees(actual, expected)
        for index, (actual, expected) in enumerate(zip(accepted, recomputed))
    ):
        raise _fail("accepted residual-grid winner does not reproduce canonically")
    boundary_axes = tuple(
        name
        for name, index, count in (
            ("center", center_index, grid["centerAxis"]["count"]),
            ("logScale", scale_index, grid["logScaleAxis"]["count"]),
            ("logShape", shape_index, grid["logShapeAxis"]["count"]),
        )
        if count > 1 and index in {0, count - 1}
    )
    return _VerifiedWinner(
        dataset_id=dataset["id"],
        generic_series_id=dataset["sourceGenericSeriesID"],
        grid_index=grid_index,
        center_index=center_index,
        log_scale_index=scale_index,
        log_shape_index=shape_index,
        center=center,
        log_scale=log_scale,
        log_shape=log_shape,
        offset=offset,
        amplitude=amplitude,
        objective=objective,
        boundary_axes=boundary_axes,
        result_payload=dict(best),
    )


def _verify_run_counter_scopes(
    run_result: Mapping[str, Any],
    final_dataset_status: Mapping[str, Any],
    *,
    expected_project_work_units: int,
) -> None:
    project_counters = {
        "projectAssignedWorkUnits": 0,
        "projectCompletedWorkUnits": expected_project_work_units,
        "projectFailedWorkUnits": 0,
        "projectPendingWorkUnits": 0,
        "projectTotalWorkUnits": expected_project_work_units,
    }
    if any(
        _exact_count(
            run_result.get(field_name),
            f"project run {field_name}",
        )
        != expected
        for field_name, expected in project_counters.items()
    ):
        raise _fail(
            "residual-grid investigation project counters lack exact "
            "complete coverage"
        )

    final_dataset_counters = {
        field_name: _exact_count(
            final_dataset_status.get(field_name),
            f"final dataset {field_name}",
        )
        for field_name in (
            "assignedWorkUnits",
            "completedWorkUnits",
            "failedWorkUnits",
            "pendingWorkUnits",
            "totalWorkUnits",
        )
    }
    if any(
        _exact_count(
            run_result.get(field_name),
            f"project run {field_name}",
        )
        != expected
        for field_name, expected in final_dataset_counters.items()
    ):
        raise _fail(
            "residual-grid investigation current-dataset counters disagree "
            "with the final canonical dataset"
        )


def _verify_investigation(
    path: Path,
    grid: _VerifiedGrid,
    project_path: Path,
) -> _VerifiedInvestigation:
    label = "residual-grid investigation"
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
    _nonempty_string(investigation.get("workflow_version"), "workflow version")
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
    winners: list[_VerifiedWinner] = []
    dataset_contribution_totals: dict[str, int] = {}
    for status, dataset in zip(statuses, grid.datasets):
        if not isinstance(status, Mapping):
            raise _fail(f"{label} dataset status is malformed")
        required_status = {
            "assignedWorkUnits": 0,
            "completedCandidateCount": CANDIDATES_PER_DATASET,
            "completedWorkUnits": WORK_UNITS_PER_DATASET,
            "coverageComplete": True,
            "curveGridStatus": "CURVE_GRID_COMPLETE",
            "datasetSchemaID": DATASET_SCHEMA_ID,
            "failedWorkUnits": 0,
            "familyID": FAMILY_ID,
            "id": dataset["id"],
            "payloadSchemaID": PAYLOAD_SCHEMA_ID,
            "pendingWorkUnits": 0,
            "resultSchemaID": RESULT_SCHEMA_ID,
            "totalCandidateCount": CANDIDATES_PER_DATASET,
            "totalWorkUnits": WORK_UNITS_PER_DATASET,
            "workloadID": WORKLOAD_ID,
            "workloadStatus": "CURVE_GRID_COMPLETE",
        }
        if any(status.get(key) != value for key, value in required_status.items()):
            raise _fail(f"{label} dataset coverage is incomplete")
        dataset_contributions = status.get("nodeContributions")
        if not isinstance(dataset_contributions, Mapping):
            raise _fail(f"{label} dataset contributions are malformed")
        if _safe_sum(
            list(dataset_contributions.values()),
            "dataset node contribution count",
        ) != WORK_UNITS_PER_DATASET:
            raise _fail(f"{label} dataset contributions are incomplete")
        for node_id, count in dataset_contributions.items():
            _nonempty_string(node_id, "dataset contribution node ID")
            dataset_contribution_totals[node_id] = (
                dataset_contribution_totals.get(node_id, 0) + count
            )
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
    if terminal_stage["result"] != expected_terminal:
        raise _fail(f"{label} terminal check is invalid")
    return _VerifiedInvestigation(
        investigation_id=investigation_id,
        investigation_sha256=_sha256_bytes(record_bytes),
        run_stage_id=run_stage["id"],
        run_stage_ledger_sha256=ledger_hashes[run_stage["id"]],
        stage_ledger_sha256s=dict(sorted(ledger_hashes.items())),
        winners=tuple(winners),
    )


def _null_fit(values: Sequence[Any], weights: Sequence[Any]) -> tuple[float, float]:
    if len(values) != len(weights) or not values:
        raise _fail("null-fit arrays are inconsistent")
    total_weight = 0.0
    weighted_value = 0.0
    numeric_values: list[float] = []
    numeric_weights: list[float] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        number = _finite_number(value, f"null values[{index}]")
        numeric_weight = _finite_number(weight, f"null weights[{index}]")
        if numeric_weight < 0.0:
            raise _fail("null-fit weights must be nonnegative")
        total_weight += numeric_weight
        weighted_value += numeric_weight * number
        if not math.isfinite(total_weight) or not math.isfinite(weighted_value):
            raise _fail("null-fit accumulation is non-finite")
        numeric_values.append(number)
        numeric_weights.append(numeric_weight)
    if total_weight <= 0.0:
        raise _fail("null fit has no positive-weight support")
    offset = weighted_value / total_weight
    wrss = 0.0
    for value, weight in zip(numeric_values, numeric_weights):
        residual = value - offset
        wrss += weight * residual * residual
        if not math.isfinite(wrss):
            raise _fail("null-fit WRSS is non-finite")
    return offset, wrss


def _support_counts(
    coordinates: Sequence[Any],
    weights: Sequence[Any],
    *,
    center: float,
    effective_width: float,
) -> tuple[int, int]:
    if len(coordinates) != len(weights):
        raise _fail("support-count arrays are inconsistent")
    two_widths = 2.0 * effective_width
    if not math.isfinite(two_widths) or effective_width <= 0.0:
        raise _fail("frozen effective width is invalid")
    within_one = 0
    within_two = 0
    for index, (coordinate_value, weight_value) in enumerate(
        zip(coordinates, weights)
    ):
        coordinate = _finite_number(coordinate_value, f"coordinates[{index}]")
        weight = _finite_number(weight_value, f"inverseVariances[{index}]")
        if weight > 0.0:
            distance = abs(coordinate - center)
            if distance <= effective_width:
                within_one += 1
            if distance <= two_widths:
                within_two += 1
    return within_one, within_two


def _contract() -> dict[str, Any]:
    return {
        "canonicalCurveFamilyID": FAMILY_ID,
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant "
            "whitespace, non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractID": CROSS_VALIDATION_CONTRACT_ID,
        "contractVersion": CROSS_VALIDATION_CONTRACT_VERSION,
        "crossSeriesDecisionRule": {
            "confirmedStatus": CONFIRMED_STATUS,
            "rule": (
                "Discovery gate passes and at least one distinct admitted "
                "validation series passes the held-out validation gate."
            ),
            "unconfirmedStatus": UNCONFIRMED_STATUS,
        },
        "discoveryGate": {
            "completeCoverageRequired": True,
            "minimumDeltaWRSS": DISCOVERY_DELTA_WRSS_THRESHOLD,
            "searchedBoundaryReportedButNotRejected": True,
        },
        "effectiveWidthRule": "exp(frozenLogScale) * exp(frozenLogShape)",
        "heldOutValidationGate": {
            "minimumDeltaWRSS": VALIDATION_DELTA_WRSS_THRESHOLD,
            "minimumPositiveWeightSamplesWithinTwoWidths": (
                MINIMUM_TWO_WIDTH_SUPPORT
            ),
            "sameNonzeroAmplitudeSignRequired": True,
            "selfValidationForbidden": True,
        },
        "identityIsolationStatement": (
            "Sealed identity, archive sources, source filenames, event names, "
            "catalog identifiers, publications, sky coordinates, and "
            "published physical parameters were not read or consulted."
        ),
        "interpretationStatement": (
            "This validates generic temporal and directional residual "
            "reproducibility; it does not classify a planetary anomaly or "
            "make a discovery claim."
        ),
        "overallClassificationRule": {
            "negative": NEGATIVE_CLASSIFICATION,
            "positive": POSITIVE_CLASSIFICATION,
            "rule": "Positive when at least one component is cross-series confirmed.",
        },
        "validationFitRule": (
            "Freeze discovery center, log scale, and log shape; evaluate the "
            "canonical unit basis on each distinct admitted validation series; "
            "fit only validation offset and unconstrained signed amplitude."
        ),
    }


def _held_out_result(
    discovery: _VerifiedWinner,
    validation_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    validation_id = validation_dataset["sourceGenericSeriesID"]
    scale = math.exp(discovery.log_scale)
    shape = math.exp(discovery.log_shape)
    effective_width = scale * shape
    if not math.isfinite(effective_width) or effective_width <= 0.0:
        raise _fail("frozen discovery effective width is invalid")
    coordinates = validation_dataset["coordinates"]
    values = validation_dataset["values"]
    weights = validation_dataset["inverseVariances"]
    within_one, within_two = _support_counts(
        coordinates,
        weights,
        center=discovery.center,
        effective_width=effective_width,
    )
    null_offset, null_wrss = _null_fit(values, weights)
    result = {
        "amplitudeSignMatchesDiscovery": False,
        "decisionReasons": [],
        "deltaWRSS": None,
        "discoveryGenericSeriesID": discovery.generic_series_id,
        "fittedAmplitude": None,
        "fittedAmplitudeSign": "unavailable",
        "fittedOffset": None,
        "frozenDiscoveryAmplitudeSign": _amplitude_sign(discovery.amplitude),
        "frozenCenter": discovery.center,
        "frozenEffectiveWidth": effective_width,
        "frozenLogScale": discovery.log_scale,
        "frozenLogShape": discovery.log_shape,
        "heldOutValidationGatePassed": False,
        "nullFittedOffset": null_offset,
        "nullWeightedResidualSumSquares": null_wrss,
        "positiveWeightSamplesWithinOneEffectiveWidth": within_one,
        "positiveWeightSamplesWithinTwoEffectiveWidths": within_two,
        "status": "EVALUATED",
        "templateWeightedResidualSumSquares": None,
        "validationGenericSeriesID": validation_id,
    }
    try:
        fit = _fit_series(
            coordinates,
            values,
            weights,
            center=discovery.center,
            scale=scale,
            shape=shape,
        )
    except ResidualPreparationError as error:
        result["status"] = "FIT_FAILED"
        result["decisionReasons"] = [f"frozen template fit failed: {error}"]
        return result
    delta_wrss = null_wrss - fit.weighted_residual_sum_squares
    if not math.isfinite(delta_wrss):
        raise _fail("held-out delta WRSS is non-finite")
    discovery_sign = _amplitude_sign(discovery.amplitude)
    validation_sign = _amplitude_sign(fit.amplitude)
    sign_matches = discovery_sign != "zero" and validation_sign == discovery_sign
    support_passes = within_two >= MINIMUM_TWO_WIDTH_SUPPORT
    delta_passes = delta_wrss >= VALIDATION_DELTA_WRSS_THRESHOLD
    passed = support_passes and sign_matches and delta_passes
    reasons = [
        (
            "two-effective-width positive-weight support passes"
            if support_passes
            else "insufficient positive-weight support within two effective widths"
        ),
        (
            "held-out amplitude has the same nonzero sign"
            if sign_matches
            else "held-out amplitude sign does not match discovery"
        ),
        (
            "held-out delta WRSS meets threshold"
            if delta_passes
            else "held-out delta WRSS is below threshold"
        ),
    ]
    result.update(
        {
            "amplitudeSignMatchesDiscovery": sign_matches,
            "decisionReasons": reasons,
            "deltaWRSS": delta_wrss,
            "fittedAmplitude": fit.amplitude,
            "fittedAmplitudeSign": validation_sign,
            "fittedOffset": fit.offset,
            "heldOutValidationGatePassed": passed,
            "templateWeightedResidualSumSquares": (
                fit.weighted_residual_sum_squares
            ),
        }
    )
    return result


def _result(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    residuals: _VerifiedResiduals,
    grid: _VerifiedGrid,
    investigation: _VerifiedInvestigation,
) -> dict[str, Any]:
    dataset_by_id = {
        dataset["sourceGenericSeriesID"]: dataset for dataset in grid.datasets
    }
    components: list[dict[str, Any]] = []
    for discovery in investigation.winners:
        discovery_dataset = dataset_by_id[discovery.generic_series_id]
        _, null_wrss = _null_fit(
            discovery_dataset["values"],
            discovery_dataset["inverseVariances"],
        )
        delta_wrss = null_wrss - discovery.objective
        if not math.isfinite(delta_wrss):
            raise _fail("discovery delta WRSS is non-finite")
        discovery_gate_passed = _discovery_gate(delta_wrss)
        validations = [
            _held_out_result(discovery, validation_dataset)
            for validation_dataset in grid.datasets
            if validation_dataset["sourceGenericSeriesID"]
            != discovery.generic_series_id
        ]
        if len(validations) != len(grid.datasets) - 1:
            raise _fail("self-validation exclusion is inconsistent")
        held_out_passes = sum(
            item["heldOutValidationGatePassed"] for item in validations
        )
        confirmed = discovery_gate_passed and held_out_passes >= 1
        boundary_limited = bool(discovery.boundary_axes)
        components.append(
            {
                "componentStatus": (
                    CONFIRMED_STATUS if confirmed else UNCONFIRMED_STATUS
                ),
                "crossSeriesConfirmed": confirmed,
                "discoveryAmplitudeSign": _amplitude_sign(discovery.amplitude),
                "discoveryCoverageComplete": True,
                "discoveryDeltaWRSS": delta_wrss,
                "discoveryGatePassed": discovery_gate_passed,
                "discoveryGateReasons": [
                    "complete grid coverage verified",
                    (
                        "discovery delta WRSS meets threshold"
                        if discovery_gate_passed
                        else "discovery delta WRSS is below threshold"
                    ),
                    (
                        "searched-axis boundary reported; width interpretation limited"
                        if boundary_limited
                        else "winner is interior on every searched axis"
                    ),
                ],
                "discoveryGenericSeriesID": discovery.generic_series_id,
                "discoveryNullWeightedResidualSumSquares": null_wrss,
                "discoveryWinner": dict(discovery.result_payload),
                "heldOutPassingSeriesCount": held_out_passes,
                "heldOutValidations": validations,
                "searchedAxisBoundaryReported": boundary_limited,
                "searchedBoundaryAxes": list(discovery.boundary_axes),
                "widthInterpretationLimitedByBoundary": boundary_limited,
            }
        )
    confirmed_count = sum(item["crossSeriesConfirmed"] for item in components)
    classification = _overall_classification(confirmed_count)
    positive = classification == POSITIVE_CLASSIFICATION
    admitted_count = len(grid.generic_series_ids)
    return {
        "admittedSeriesCount": admitted_count,
        "confirmedComponentCount": confirmed_count,
        "contractID": CROSS_VALIDATION_CONTRACT_ID,
        "contractSHA256": contract_sha256,
        "contractVersion": CROSS_VALIDATION_CONTRACT_VERSION,
        "discoveryClaim": False,
        "heldOutValidationPairCount": admitted_count * (admitted_count - 1),
        "identityIsolationStatement": contract["identityIsolationStatement"],
        "noDiscoveryStatement": (
            "This result validates only generic cross-series residual "
            "reproducibility and makes no planetary discovery claim."
        ),
        "orderedAdmittedGenericSeriesIDs": list(grid.generic_series_ids),
        "overallClassification": classification,
        "parentHashes": {
            "ancestryArtifactHashes": residuals.manifest[
                "parentArtifactHashes"
            ],
            "residualGridBuildManifestSHA256": grid.build_manifest_sha256,
            "residualGridContractFileSHA256": grid.contract_file_sha256,
            "residualGridDatasetSHA256s": dict(grid.dataset_sha256s),
            "residualGridInvestigationRecordSHA256": (
                investigation.investigation_sha256
            ),
            "residualGridProjectSHA256": grid.project_sha256,
            "residualGridRunStageLedgerSHA256": (
                investigation.run_stage_ledger_sha256
            ),
            "residualGridStageLedgerSHA256s": dict(
                investigation.stage_ledger_sha256s
            ),
            "residualPreparationContractFileSHA256": (
                residuals.contract_file_sha256
            ),
            "residualPreparationContractCanonicalSHA256": (
                residuals.manifest["contractSHA256"]
            ),
            "residualPreparationManifestFileSHA256": (
                residuals.manifest_file_sha256
            ),
            "residualSeriesFileSHA256s": dict(residuals.series_file_sha256s),
            "residualSearchContractCanonicalSHA256": (
                grid.build_manifest["residualSearchContractSHA256"]
            ),
        },
        "parentIDs": {
            "ancestryInvestigationIDs": residuals.manifest[
                "parentInvestigationIDs"
            ],
            "ancestryProjectIDs": residuals.manifest["parentProjectIDs"],
            "residualGridInvestigationID": investigation.investigation_id,
            "residualGridProjectID": grid.project["id"],
            "residualPreparationContractID": RESIDUAL_PREPARATION_CONTRACT_ID,
            "residualSearchContractID": RESIDUAL_SEARCH_CONTRACT_ID,
        },
        "planetaryInterpretationResolved": False,
        "predeclaredThresholds": {
            "discoveryMinimumDeltaWRSS": DISCOVERY_DELTA_WRSS_THRESHOLD,
            "heldOutMinimumDeltaWRSS": VALIDATION_DELTA_WRSS_THRESHOLD,
            "minimumPositiveWeightSamplesWithinTwoEffectiveWidths": (
                MINIMUM_TWO_WIDTH_SUPPORT
            ),
            "sameNonzeroAmplitudeSignRequired": True,
        },
        "recommendedNextTest": (
            CONFIRMED_NEXT_TEST if positive else UNCONFIRMED_NEXT_TEST
        ),
        "resultSchemaID": CROSS_VALIDATION_RESULT_SCHEMA_ID,
        "resultVersion": CROSS_VALIDATION_RESULT_VERSION,
        "validatedComponents": components,
    }


def _validate_residual_grid_impl(
    residual_root: str | Path,
    *,
    residual_grid_root: str | Path,
    residual_grid_investigation_record: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    paths = (
        (Path(residual_root).expanduser().absolute(), "residual preparation root"),
        (Path(residual_grid_root).expanduser().absolute(), "residual-grid root"),
        (
            Path(residual_grid_investigation_record).expanduser().absolute(),
            "residual-grid investigation",
        ),
    )
    _reject_symlink_components(output.parent, "output root")
    for path, description in paths:
        _reject_symlink_components(path, description)
    residual_path, grid_path, investigation_path = (item[0] for item in paths)
    residuals = _verify_residual_root(residual_path)
    grid = _verify_grid_root(grid_path, residuals)
    investigation = _verify_investigation(
        investigation_path,
        grid,
        grid_path / GRID_PROJECT_RELATIVE_PATH,
    )

    contract = _contract()
    contract_bytes = _stable_json_bytes(contract)
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    try:
        _assert_identity_free((contract_bytes,))
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
        result = _result(
            contract=contract,
            contract_sha256=contract_sha256,
            residuals=residuals,
            grid=grid,
            investigation=investigation,
        )
        result_bytes = _stable_json_bytes(result)
        try:
            _assert_identity_free((result_bytes,))
        except CoarseGridBuildError as error:
            raise _fail(str(error)) from error
        _atomic_write_bytes(staging / RESULT_RELATIVE_PATH, result_bytes)
        if output.exists() or output.is_symlink():
            raise _fail("output root already exists")
        staging.rename(output)
    except Exception as error:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if isinstance(error, ResidualGridValidationError):
            raise
        raise _fail("atomic cross-validation publication failed") from error
    return {"contract": contract, "result": result}


def validate_residual_grid(
    residual_root: str | Path,
    *,
    residual_grid_root: str | Path,
    residual_grid_investigation_record: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Validate every frozen discovery component on every other series."""

    try:
        return _validate_residual_grid_impl(
            residual_root,
            residual_grid_root=residual_grid_root,
            residual_grid_investigation_record=(
                residual_grid_investigation_record
            ),
            output_root=output_root,
        )
    except ResidualGridValidationError:
        raise
    except (
        CoarseGridBuildError,
        RefinementGridBuildError,
        ResidualGridBuildError,
        ResidualPreparationError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise _fail(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate localized blind residual components across admitted "
            "generic series."
        )
    )
    parser.add_argument("--residual-root", required=True, type=Path)
    parser.add_argument("--residual-grid-root", required=True, type=Path)
    parser.add_argument(
        "--residual-grid-investigation-record", required=True, type=Path
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    published = validate_residual_grid(
        arguments.residual_root,
        residual_grid_root=arguments.residual_grid_root,
        residual_grid_investigation_record=(
            arguments.residual_grid_investigation_record
        ),
        output_root=arguments.output_root,
    )
    result = published["result"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind residual cross-series validation ready")
    print(f"classification: {result['overallClassification']}")
    print(f"confirmed components: {result['confirmedComponentCount']}")
    print(f"result: {output / RESULT_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
