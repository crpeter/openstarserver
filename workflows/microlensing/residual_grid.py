"""Build a deterministic localized CurveGrid from verified residual series."""

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
)
from workflows.microlensing.coarse_grid import (
    CoarseGridBuildError,
    _assert_identity_free,
    _atomic_write_bytes,
    _canonical_compact_json_bytes,
    _read_regular_file,
    _stable_json_bytes,
)
from workflows.microlensing.prepare_residuals import (
    CONTRACT_RELATIVE_PATH as RESIDUAL_CONTRACT_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH as RESIDUAL_MANIFEST_RELATIVE_PATH,
    RESIDUAL_PREPARATION_CONTRACT_ID,
    ResidualPreparationError,
    _regular_directory as _residual_regular_directory,
    _reject_symlink_components as _residual_reject_symlink_components,
    prepare_blind_microlensing_residuals,
)


RESIDUAL_SEARCH_CONTRACT_ID = "openstar.microlensing-residual-search-grid.v1"
RESIDUAL_SEARCH_CONTRACT_VERSION = "1.0"
BUILD_MANIFEST_SCHEMA_ID = "openstar.microlensing-residual-grid-build.v1"
BUILD_MANIFEST_VERSION = "1.0"

CENTER_COUNT = 129
LOG_SCALE_COUNT = 17
LOG_SHAPE_COUNT = 1
LOG_SHAPE_STEP = 1.0
CANDIDATES_PER_WORK_UNIT = 64
MINIMUM_IN_WINDOW_POSITIVE_WEIGHT_SAMPLES = 8
CANDIDATES_PER_DATASET = CENTER_COUNT * LOG_SCALE_COUNT * LOG_SHAPE_COUNT
WORK_UNITS_PER_DATASET = (
    CANDIDATES_PER_DATASET + CANDIDATES_PER_WORK_UNIT - 1
) // CANDIDATES_PER_WORK_UNIT

CONTRACT_RELATIVE_PATH = "residual-search-contract.json"
PROJECT_RELATIVE_PATH = "project.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"
DATASET_DIRECTORY = "datasets"

_SAFE_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ResidualGridBuildError(RuntimeError):
    """The blind residual-search project cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class _VerifiedResidualPreparation:
    contract: Mapping[str, Any]
    manifest: Mapping[str, Any]
    series: tuple[Mapping[str, Any], ...]
    contract_file_sha256: str
    manifest_file_sha256: str
    series_file_sha256s: Mapping[str, str]


def _fail(message: str) -> ResidualGridBuildError:
    return ResidualGridBuildError(message)


def _reject_symlink_components(path: Path, description: str) -> None:
    try:
        _residual_reject_symlink_components(path, description)
    except ResidualPreparationError as error:
        raise _fail(str(error)) from error


def _regular_directory(path: Path, description: str) -> Path:
    try:
        return _residual_regular_directory(path, description)
    except ResidualPreparationError as error:
        raise _fail(str(error)) from error


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
            raise _fail(f"{field_name} contains an invalid count")
        if value > MAX_SAFE_INTEGER - total:
            raise _fail(f"{field_name} exceeds the safe integer range")
        total += value
    return total


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return _read_regular_file(path, description)
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error


def _verify_residual_tree(
    residual_root: Path,
    expected_root: Path,
    expected_result: Mapping[str, Any],
) -> _VerifiedResidualPreparation:
    actual = _regular_directory(residual_root, "residual root")
    expected = _regular_directory(expected_root, "expected residual root")
    expected_series = expected_result.get("series")
    if not isinstance(expected_series, list) or not expected_series:
        raise _fail("reconstructed residual series are missing")

    expected_root_names = {
        RESIDUAL_CONTRACT_RELATIVE_PATH,
        RESIDUAL_MANIFEST_RELATIVE_PATH,
        "series",
    }
    try:
        if {entry.name for entry in actual.iterdir()} != expected_root_names:
            raise _fail("residual artifact set is incomplete or unexpected")
    except OSError as error:
        raise _fail("residual root is unreadable") from error
    actual_series_root = _regular_directory(
        actual / "series", "residual series directory"
    )
    expected_series_root = _regular_directory(
        expected / "series", "expected residual series directory"
    )
    expected_series_names = {
        f"residual-series-{ordinal:03d}.json"
        for ordinal in range(1, len(expected_series) + 1)
    }
    try:
        if {entry.name for entry in actual_series_root.iterdir()} != (
            expected_series_names
        ):
            raise _fail("residual series artifact set is incomplete or unexpected")
    except OSError as error:
        raise _fail("residual series directory is unreadable") from error

    relative_paths = (
        RESIDUAL_CONTRACT_RELATIVE_PATH,
        RESIDUAL_MANIFEST_RELATIVE_PATH,
        *(
            f"series/residual-series-{ordinal:03d}.json"
            for ordinal in range(1, len(expected_series) + 1)
        ),
    )
    verified_bytes: dict[str, bytes] = {}
    for relative_path in relative_paths:
        actual_bytes = _read_bytes(actual / relative_path, relative_path)
        expected_bytes = _read_bytes(
            expected / relative_path,
            f"expected {relative_path}",
        )
        if actual_bytes != expected_bytes:
            raise _fail(
                f"residual artifact differs from deterministic reconstruction: "
                f"{relative_path}"
            )
        verified_bytes[relative_path] = actual_bytes

    manifest = expected_result.get("manifest")
    contract = expected_result.get("contract")
    if not isinstance(manifest, Mapping) or not isinstance(contract, Mapping):
        raise _fail("reconstructed residual metadata is malformed")
    series_hashes = {
        relative_path: _sha256_bytes(payload)
        for relative_path, payload in verified_bytes.items()
        if relative_path.startswith("series/")
    }
    return _VerifiedResidualPreparation(
        contract=dict(contract),
        manifest=dict(manifest),
        series=tuple(dict(value) for value in expected_series),
        contract_file_sha256=_sha256_bytes(
            verified_bytes[RESIDUAL_CONTRACT_RELATIVE_PATH]
        ),
        manifest_file_sha256=_sha256_bytes(
            verified_bytes[RESIDUAL_MANIFEST_RELATIVE_PATH]
        ),
        series_file_sha256s=dict(sorted(series_hashes.items())),
    )


def _reconstruct_and_verify_residuals(
    residual_root: Path,
    *,
    prepared_root: Path,
    coarse_project_root: Path,
    coarse_investigation_record: Path,
    refinement_project_root: Path,
    refinement_investigation_record: Path,
    first_recenter_project_root: Path,
    first_recenter_investigation_record: Path,
    second_recenter_project_root: Path,
    second_recenter_investigation_record: Path,
) -> _VerifiedResidualPreparation:
    try:
        with tempfile.TemporaryDirectory(
            prefix=".openstar-residual-reconstruction."
        ) as temporary:
            temporary_root = Path(temporary).resolve(strict=True)
            expected_root = temporary_root / "expected-residuals"
            expected_result = prepare_blind_microlensing_residuals(
                prepared_root,
                coarse_project_root=coarse_project_root,
                coarse_investigation_record=coarse_investigation_record,
                refinement_project_root=refinement_project_root,
                refinement_investigation_record=refinement_investigation_record,
                first_recenter_project_root=first_recenter_project_root,
                first_recenter_investigation_record=(
                    first_recenter_investigation_record
                ),
                second_recenter_project_root=second_recenter_project_root,
                second_recenter_investigation_record=(
                    second_recenter_investigation_record
                ),
                output_root=expected_root,
            )
            return _verify_residual_tree(
                residual_root,
                expected_root,
                expected_result,
            )
    except ResidualGridBuildError:
        raise
    except ResidualPreparationError as error:
        raise _fail(str(error)) from error
    except (OSError, TypeError, ValueError, OverflowError) as error:
        raise _fail(f"residual reconstruction failed: {error}") from error


def _search_geometry(
    frozen_geometry: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], float]:
    center = _finite_number(frozen_geometry.get("center"), "frozen center")
    log_scale = _finite_number(
        frozen_geometry.get("logScale"), "frozen log scale"
    )
    log_shape = _finite_number(
        frozen_geometry.get("logShape"), "frozen log shape"
    )
    scale = _finite_number(frozen_geometry.get("scale"), "frozen scale")
    shape = _finite_number(frozen_geometry.get("shape"), "frozen shape")
    try:
        expected_scale = math.exp(log_scale)
        expected_shape = math.exp(log_shape)
    except OverflowError as error:
        raise _fail("frozen logarithmic geometry is invalid") from error
    if (
        scale != expected_scale
        or shape != expected_shape
        or scale <= 0.0
        or shape <= 0.0
    ):
        raise _fail("frozen positive geometry is inconsistent")
    core_width = scale * shape
    if not math.isfinite(core_width) or core_width <= 0.0:
        raise _fail("derived core width is non-finite or non-positive")
    center_step = core_width / 16.0
    log_sixteen = math.log(16.0)
    axes = {
        "centerAxis": {
            "count": CENTER_COUNT,
            "start": center - 4.0 * core_width,
            "step": center_step,
        },
        "logScaleAxis": {
            "count": LOG_SCALE_COUNT,
            "start": log_scale - log_sixteen,
            "step": log_sixteen / 16.0,
        },
        "logShapeAxis": {
            "count": LOG_SHAPE_COUNT,
            "start": log_shape,
            "step": LOG_SHAPE_STEP,
        },
    }
    for axis_name, axis in axes.items():
        endpoint = axis["start"] + (axis["count"] - 1) * axis["step"]
        if not all(
            math.isfinite(value)
            for value in (axis["start"], axis["step"], endpoint)
        ):
            raise _fail(f"derived {axis_name} is non-finite")
    normalized_geometry = {
        "center": center,
        "logScale": log_scale,
        "logShape": log_shape,
        "scale": scale,
        "shape": shape,
    }
    return normalized_geometry, axes, core_width


def _admission_record(
    generic_series_id: Any,
    coordinates: Sequence[Any],
    inverse_variances: Sequence[Any],
    *,
    interval_minimum: float,
    interval_maximum: float,
) -> dict[str, Any]:
    if not isinstance(generic_series_id, str) or not generic_series_id:
        raise _fail("generic series ID must be a nonempty string")
    if len(coordinates) != len(inverse_variances) or not coordinates:
        raise _fail("admission coordinate and weight arrays are inconsistent")
    lower = _finite_number(interval_minimum, "search interval minimum")
    upper = _finite_number(interval_maximum, "search interval maximum")
    if lower > upper:
        raise _fail("search interval is reversed")
    in_window = 0
    for index, (coordinate_value, weight_value) in enumerate(
        zip(coordinates, inverse_variances)
    ):
        coordinate = _finite_number(
            coordinate_value, f"coordinates[{index}]"
        )
        weight = _finite_number(
            weight_value, f"inverseVariances[{index}]"
        )
        if weight < 0.0:
            raise _fail("inverse variances must be nonnegative")
        if weight > 0.0 and lower <= coordinate <= upper:
            in_window += 1
    admitted = in_window >= MINIMUM_IN_WINDOW_POSITIVE_WEIGHT_SAMPLES
    return {
        "admissionDecision": "ADMITTED" if admitted else "EXCLUDED",
        "admissionReason": (
            "meets predetermined in-window positive-weight sample threshold"
            if admitted
            else (
                "does not meet predetermined in-window positive-weight "
                "sample threshold"
            )
        ),
        "admissionThreshold": MINIMUM_IN_WINDOW_POSITIVE_WEIGHT_SAMPLES,
        "genericSeriesID": generic_series_id,
        "inWindowPositiveWeightSampleCount": in_window,
    }


def _curve_grid(axes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "centerAxis": dict(axes["centerAxis"]),
        "familyID": FAMILY_ID,
        "logScaleAxis": dict(axes["logScaleAxis"]),
        "logShapeAxis": dict(axes["logShapeAxis"]),
    }


def _contract(
    frozen_geometry: Mapping[str, float],
    axes: Mapping[str, Mapping[str, Any]],
    core_width: float,
) -> dict[str, Any]:
    center = frozen_geometry["center"]
    return {
        "admissionRule": {
            "decisionInputs": ["coordinates", "inverseVariances"],
            "inclusiveInterval": {
                "maximum": center + 4.0 * core_width,
                "minimum": center - 4.0 * core_width,
            },
            "minimumPositiveWeightSamples": (
                MINIMUM_IN_WINDOW_POSITIVE_WEIGHT_SAMPLES
            ),
            "residualValuesUsed": False,
        },
        "axisDerivationRules": {
            "centerAxis": (
                "count=129; start=eventCenter-4*coreWidth; "
                "step=coreWidth/16"
            ),
            "coreWidth": "exp(frozenLogScale) * exp(frozenLogShape)",
            "logScaleAxis": (
                "count=17; start=frozenLogScale-log(16); "
                "step=log(16)/16"
            ),
            "logShapeAxis": (
                "count=1; start=frozenLogShape; positive finite step=1.0 "
                "does not alter the sole value"
            ),
        },
        "amplitudeConstraint": "unconstrained-signed",
        "benchmarkKind": "known-event-recovery",
        "candidatesPerDataset": CANDIDATES_PER_DATASET,
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant "
            "whitespace, non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractID": RESIDUAL_SEARCH_CONTRACT_ID,
        "contractVersion": RESIDUAL_SEARCH_CONTRACT_VERSION,
        "curveGrid": _curve_grid(axes),
        "derivedCoreWidth": core_width,
        "frozenGeometry": dict(frozen_geometry),
        "identityIsolationStatement": (
            "Sealed identity, archive sources, source filenames, event names, "
            "catalog identifiers, publications, identity coordinates, and "
            "published physical parameters were not read or consulted."
        ),
        "modelScopeStatement": (
            "This is generic localized residual modeling, not planetary "
            "classification or a discovery claim."
        ),
        "residualPreparationContractID": RESIDUAL_PREPARATION_CONTRACT_ID,
        "residualVerificationRule": (
            "The supplied residual contract, manifest, and every ordered "
            "series file must match a fresh deterministic reconstruction "
            "from the verified immutable ancestry byte for byte."
        ),
        "workUnitsPerDataset": WORK_UNITS_PER_DATASET,
    }


def _dataset(
    *,
    dataset_id: str,
    source: Mapping[str, Any],
    axes: Mapping[str, Mapping[str, Any]],
    residual_manifest_sha256: str,
    residual_search_contract_sha256: str,
    residual_series_file_sha256: str,
) -> dict[str, Any]:
    dataset = {
        "coordinates": list(source["coordinates"]),
        "curveGrid": _curve_grid(axes),
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "id": dataset_id,
        "inverseVariances": list(source["inverseVariances"]),
        "residualManifestSHA256": residual_manifest_sha256,
        "residualSearchContractID": RESIDUAL_SEARCH_CONTRACT_ID,
        "residualSearchContractSHA256": residual_search_contract_sha256,
        "sourceGenericSeriesID": source["genericSeriesID"],
        "sourceResidualSeriesFileSHA256": residual_series_file_sha256,
        "values": list(source["residualValues"]),
    }
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(
            f"constructed residual CurveGrid dataset is invalid: {error}"
        ) from error
    return dataset


def _project(project_id: str, datasets: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "datasets": [dict(value) for value in datasets],
        "id": project_id,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "workloadID": WORKLOAD_ID,
    }


def _build_residual_grid_project(
    prepared_root: str | Path,
    *,
    coarse_project_root: str | Path,
    coarse_investigation_record: str | Path,
    refinement_project_root: str | Path,
    refinement_investigation_record: str | Path,
    first_recenter_project_root: str | Path,
    first_recenter_investigation_record: str | Path,
    second_recenter_project_root: str | Path,
    second_recenter_investigation_record: str | Path,
    residual_root: str | Path,
    project_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify residual ancestry and publish one localized multi-dataset grid."""

    if (
        not isinstance(project_id, str)
        or _SAFE_PROJECT_ID.fullmatch(project_id) is None
    ):
        raise _fail("project ID is malformed or unsafe")
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    _reject_symlink_components(output.parent, "output root")

    inputs = {
        "prepared_root": Path(prepared_root).expanduser().absolute(),
        "coarse_project_root": Path(coarse_project_root).expanduser().absolute(),
        "coarse_investigation_record": Path(
            coarse_investigation_record
        ).expanduser().absolute(),
        "refinement_project_root": Path(
            refinement_project_root
        ).expanduser().absolute(),
        "refinement_investigation_record": Path(
            refinement_investigation_record
        ).expanduser().absolute(),
        "first_recenter_project_root": Path(
            first_recenter_project_root
        ).expanduser().absolute(),
        "first_recenter_investigation_record": Path(
            first_recenter_investigation_record
        ).expanduser().absolute(),
        "second_recenter_project_root": Path(
            second_recenter_project_root
        ).expanduser().absolute(),
        "second_recenter_investigation_record": Path(
            second_recenter_investigation_record
        ).expanduser().absolute(),
    }
    residual = Path(residual_root).expanduser().absolute()
    for path, description in (
        *(
            (path, name.replace("_", " "))
            for name, path in inputs.items()
        ),
        (residual, "residual root"),
    ):
        _reject_symlink_components(path, description)

    verified = _reconstruct_and_verify_residuals(
        residual,
        **inputs,
    )
    manifest = verified.manifest
    frozen_value = manifest.get("frozenGeometry")
    if not isinstance(frozen_value, Mapping):
        raise _fail("verified residual frozen geometry is malformed")
    frozen_geometry, axes, core_width = _search_geometry(frozen_value)
    contract = _contract(frozen_geometry, axes, core_width)
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    contract_bytes = _stable_json_bytes(contract)

    interval_minimum = frozen_geometry["center"] - 4.0 * core_width
    interval_maximum = frozen_geometry["center"] + 4.0 * core_width
    admission_records: list[dict[str, Any]] = []
    admitted_sources: list[tuple[int, Mapping[str, Any], str]] = []
    for source_ordinal, source in enumerate(verified.series, start=1):
        record = _admission_record(
            source.get("genericSeriesID"),
            source.get("coordinates", ()),
            source.get("inverseVariances", ()),
            interval_minimum=interval_minimum,
            interval_maximum=interval_maximum,
        )
        admission_records.append(record)
        if record["admissionDecision"] == "ADMITTED":
            relative_path = f"series/residual-series-{source_ordinal:03d}.json"
            admitted_sources.append((source_ordinal, source, relative_path))
    if not admitted_sources:
        raise _fail("no residual series meet the predetermined admission gate")

    dataset_documents: list[tuple[str, dict[str, Any], bytes]] = []
    project_datasets: list[dict[str, str]] = []
    dataset_records: list[dict[str, Any]] = []
    evaluation_counts: list[int] = []
    for admitted_ordinal, (_, source, source_relative_path) in enumerate(
        admitted_sources, start=1
    ):
        dataset_id = f"{project_id}.residual-series-{admitted_ordinal:03d}"
        relative_path = (
            f"{DATASET_DIRECTORY}/residual-series-{admitted_ordinal:03d}.json"
        )
        dataset = _dataset(
            dataset_id=dataset_id,
            source=source,
            axes=axes,
            residual_manifest_sha256=verified.manifest_file_sha256,
            residual_search_contract_sha256=contract_sha256,
            residual_series_file_sha256=verified.series_file_sha256s[
                source_relative_path
            ],
        )
        dataset_bytes = _stable_json_bytes(dataset)
        dataset_sha256 = _sha256_bytes(dataset_bytes)
        sample_count = len(dataset["coordinates"])
        evaluation_count = _safe_product(
            sample_count,
            CANDIDATES_PER_DATASET,
            "sample-candidate evaluation count",
        )
        evaluation_counts.append(evaluation_count)
        dataset_documents.append((relative_path, dataset, dataset_bytes))
        project_datasets.append({"id": dataset_id, "path": relative_path})
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
                    verified.series_file_sha256s[source_relative_path]
                ),
            }
        )

    project = _project(project_id, project_datasets)
    project_bytes = _stable_json_bytes(project)
    total_dataset_count = len(dataset_documents)
    total_candidate_count = _safe_product(
        total_dataset_count,
        CANDIDATES_PER_DATASET,
        "total candidate count",
    )
    total_work_unit_count = _safe_product(
        total_dataset_count,
        WORK_UNITS_PER_DATASET,
        "total expected work-unit count",
    )
    build_manifest = {
        "admissionRecords": admission_records,
        "axisDerivationRules": contract["axisDerivationRules"],
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID,
        "buildManifestVersion": BUILD_MANIFEST_VERSION,
        "candidateCountPerDataset": CANDIDATES_PER_DATASET,
        "canonicalCurveFamilyID": FAMILY_ID,
        "derivedCoreWidth": core_width,
        "datasets": dataset_records,
        "frozenGeometry": frozen_geometry,
        "identityIsolationStatement": contract["identityIsolationStatement"],
        "modelScopeStatement": contract["modelScopeStatement"],
        "orderedAdmittedDatasetIDs": [
            value["datasetID"] for value in dataset_records
        ],
        "outputHashes": {
            "datasets": {
                value["outputFile"]: value["outputSHA256"]
                for value in dataset_records
            },
            "project": _sha256_bytes(project_bytes),
            "residualSearchContractFile": _sha256_bytes(contract_bytes),
        },
        "parentArtifactHashes": manifest["parentArtifactHashes"],
        "parentInvestigationIDs": manifest["parentInvestigationIDs"],
        "parentProjectIDs": manifest["parentProjectIDs"],
        "preparationManifestSHA256": manifest["preparationManifestSHA256"],
        "projectID": project_id,
        "publishedAxes": {
            key: dict(value) for key, value in axes.items()
        },
        "relativeArtifactPaths": {
            "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
            "datasets": [value["outputFile"] for value in dataset_records],
            "project": PROJECT_RELATIVE_PATH,
            "residualSearchContract": CONTRACT_RELATIVE_PATH,
        },
        "residualPreparationContractCanonicalSHA256": manifest[
            "contractSHA256"
        ],
        "residualPreparationContractFileSHA256": (
            verified.contract_file_sha256
        ),
        "residualPreparationManifestFileSHA256": verified.manifest_file_sha256,
        "residualVerificationRule": contract["residualVerificationRule"],
        "verifiedResidualSeriesFileSHA256s": dict(
            verified.series_file_sha256s
        ),
        "residualSearchContractID": RESIDUAL_SEARCH_CONTRACT_ID,
        "residualSearchContractSHA256": contract_sha256,
        "residualSearchContractVersion": RESIDUAL_SEARCH_CONTRACT_VERSION,
        "totalAdmittedDatasetCount": total_dataset_count,
        "totalCandidateCount": total_candidate_count,
        "totalExpectedSampleCandidateEvaluationCount": _safe_sum(
            evaluation_counts,
            "total sample-candidate evaluation count",
        ),
        "totalExpectedWorkUnitCount": total_work_unit_count,
        "verifiedConvergenceEvidence": manifest["convergenceEvidence"],
        "workUnitsPerDataset": WORK_UNITS_PER_DATASET,
    }
    build_manifest_bytes = _stable_json_bytes(build_manifest)
    try:
        _assert_identity_free(
            (
                contract_bytes,
                project_bytes,
                build_manifest_bytes,
                *(value[2] for value in dataset_documents),
            )
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
        for relative_path, _, dataset_bytes in dataset_documents:
            _atomic_write_bytes(staging / relative_path, dataset_bytes)
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
        if isinstance(error, ResidualGridBuildError):
            raise
        raise _fail("atomic output publication failed") from error

    return {
        "buildManifest": build_manifest,
        "contract": contract,
        "datasets": [value[1] for value in dataset_documents],
        "project": project,
    }


def build_residual_grid_project(
    prepared_root: str | Path,
    *,
    coarse_project_root: str | Path,
    coarse_investigation_record: str | Path,
    refinement_project_root: str | Path,
    refinement_investigation_record: str | Path,
    first_recenter_project_root: str | Path,
    first_recenter_investigation_record: str | Path,
    second_recenter_project_root: str | Path,
    second_recenter_investigation_record: str | Path,
    residual_root: str | Path,
    project_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Publish a blind residual grid behind one stable public error type."""

    try:
        return _build_residual_grid_project(
            prepared_root,
            coarse_project_root=coarse_project_root,
            coarse_investigation_record=coarse_investigation_record,
            refinement_project_root=refinement_project_root,
            refinement_investigation_record=refinement_investigation_record,
            first_recenter_project_root=first_recenter_project_root,
            first_recenter_investigation_record=(
                first_recenter_investigation_record
            ),
            second_recenter_project_root=second_recenter_project_root,
            second_recenter_investigation_record=(
                second_recenter_investigation_record
            ),
            residual_root=residual_root,
            project_id=project_id,
            output_root=output_root,
        )
    except ResidualGridBuildError:
        raise
    except (
        CoarseGridBuildError,
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
            "Build a deterministic localized CurveGrid project from verified "
            "blind residual series."
        )
    )
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--coarse-project-root", required=True, type=Path)
    parser.add_argument("--coarse-investigation-record", required=True, type=Path)
    parser.add_argument("--refinement-project-root", required=True, type=Path)
    parser.add_argument(
        "--refinement-investigation-record", required=True, type=Path
    )
    parser.add_argument("--first-recenter-project-root", required=True, type=Path)
    parser.add_argument(
        "--first-recenter-investigation-record", required=True, type=Path
    )
    parser.add_argument("--second-recenter-project-root", required=True, type=Path)
    parser.add_argument(
        "--second-recenter-investigation-record", required=True, type=Path
    )
    parser.add_argument("--residual-root", required=True, type=Path)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_residual_grid_project(
        arguments.prepared_root,
        coarse_project_root=arguments.coarse_project_root,
        coarse_investigation_record=arguments.coarse_investigation_record,
        refinement_project_root=arguments.refinement_project_root,
        refinement_investigation_record=arguments.refinement_investigation_record,
        first_recenter_project_root=arguments.first_recenter_project_root,
        first_recenter_investigation_record=(
            arguments.first_recenter_investigation_record
        ),
        second_recenter_project_root=arguments.second_recenter_project_root,
        second_recenter_investigation_record=(
            arguments.second_recenter_investigation_record
        ),
        residual_root=arguments.residual_root,
        project_id=arguments.project_id,
        output_root=arguments.output_root,
    )
    manifest = result["buildManifest"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind localized residual CurveGrid project ready")
    print(f"project ID: {manifest['projectID']}")
    print(f"admitted datasets: {manifest['totalAdmittedDatasetCount']}")
    print(f"total candidates: {manifest['totalCandidateCount']}")
    print(f"expected work units: {manifest['totalExpectedWorkUnitCount']}")
    print(f"project: {output / PROJECT_RELATIVE_PATH}")
    print(f"build manifest: {output / BUILD_MANIFEST_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
