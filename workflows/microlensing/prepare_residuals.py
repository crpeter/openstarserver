"""Prepare identity-free residual series after verified grid convergence."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openstar_workloads.plugins.curve_grid import (
    FAMILY_ID,
    MAX_SAFE_INTEGER,
)
from workflows.microlensing.coarse_grid import (
    CONTRACT_RELATIVE_PATH as COARSE_CONTRACT_RELATIVE_PATH,
    PROJECT_RELATIVE_PATH as COARSE_PROJECT_RELATIVE_PATH,
    CoarseGridBuildError,
    _assert_identity_free,
    _atomic_write_bytes,
    _canonical_compact_json_bytes,
    _stable_json_bytes,
    _verify_blind_preparation,
)
from workflows.microlensing.recenter_grid import (
    PROJECT_RELATIVE_PATH as FIRST_RECENTER_PROJECT_RELATIVE_PATH,
    RecenterGridBuildError as FirstRecenterGridBuildError,
    _verify_exact_project_tree,
    _verify_investigation_tree,
    _verify_prepare_parameter_path,
    _verify_refinement_investigation,
    _verify_refinement_project,
)
from workflows.microlensing.refine_grid import (
    PROJECT_RELATIVE_PATH as REFINEMENT_PROJECT_RELATIVE_PATH,
    RefinementGridBuildError,
    _verify_coarse_project,
    _verify_investigation as _verify_coarse_investigation,
)
from workflows.microlensing.second_recenter_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as SECOND_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
    BUILD_MANIFEST_SCHEMA_ID as SECOND_RECENTER_BUILD_MANIFEST_SCHEMA_ID,
    BUILD_MANIFEST_VERSION as SECOND_RECENTER_BUILD_MANIFEST_VERSION,
    CONTRACT_RELATIVE_PATH as SECOND_RECENTER_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as SECOND_RECENTER_DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT,
    PROJECT_RELATIVE_PATH as SECOND_RECENTER_PROJECT_RELATIVE_PATH,
    SECOND_RECENTER_GRID_CONTRACT_ID,
    SECOND_RECENTER_GRID_CONTRACT_VERSION,
    TOTAL_CANDIDATE_COUNT,
    SecondRecenterGridBuildError,
    _VerifiedFirstRecenterProject,
    _VerifiedFirstRecenterWinner,
    _contract as _expected_second_recenter_contract,
    _dataset as _expected_second_recenter_dataset,
    _derived_axes as _expected_second_recenter_axes,
    _investigation_id,
    _parent_hashes as _ancestry_hashes_through_first_recenter,
    _project as _expected_second_recenter_project,
    _read_json_file,
    _regular_directory as _second_recenter_regular_directory,
    _reject_symlink_components as _second_recenter_reject_symlink_components,
    _sha256_bytes,
    _verify_first_recenter_investigation,
    _verify_first_recenter_project,
    _winner_record,
)


RESIDUAL_PREPARATION_CONTRACT_ID = (
    "openstar.microlensing-residual-preparation-contract.v1"
)
RESIDUAL_PREPARATION_CONTRACT_VERSION = "1.0"
RESIDUAL_MANIFEST_SCHEMA_ID = "openstar.microlensing-residual-manifest.v1"
RESIDUAL_MANIFEST_VERSION = "1.0"
RESIDUAL_SERIES_SCHEMA_ID = "openstar.microlensing-residual-series.v1"
RESIDUAL_SERIES_VERSION = "1.0"

CONTRACT_RELATIVE_PATH = "residual-preparation-contract.json"
MANIFEST_RELATIVE_PATH = "residual-manifest.json"
SERIES_DIRECTORY = "series"

_SINGULARITY_RELATIVE_TOLERANCE = 1.0e-12
_SAFE_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SECOND_RECENTER_BUILD_FIELDS = frozenset(
    (
        "acceptedFirstRecenterWinner",
        "blindnessStatement",
        "buildManifestSchemaID",
        "buildManifestVersion",
        "candidateCount",
        "contractSchemaID",
        "contractVersion",
        "derivedSecondRecenterAxes",
        "expectedSampleCandidateEvaluationCount",
        "expectedWorkUnitCount",
        "inputSeriesSHA256",
        "outputContractFileSHA256",
        "outputDatasetSHA256",
        "outputProjectSHA256",
        "parentArtifactHashes",
        "parentAxes",
        "parentInvestigationIDs",
        "parentProjectIDs",
        "projectID",
        "relativeArtifactPaths",
        "secondRecenterSearchContractSHA256",
        "selectedSampleCount",
        "selectedSeriesID",
    )
)


class ResidualPreparationError(RuntimeError):
    """Blind residual preparation cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class _SeriesFit:
    basis: tuple[float, ...]
    offset: float
    amplitude: float
    model_values: tuple[float, ...]
    residual_values: tuple[float, ...]
    weighted_residual_sum_squares: float
    positive_weight_sample_count: int
    maximum_absolute_standardized_residual: float
    maximum_absolute_standardized_residual_index: int


def _fail(message: str) -> ResidualPreparationError:
    return ResidualPreparationError(message)


def _reject_symlink_components(path: Path, description: str) -> None:
    try:
        _second_recenter_reject_symlink_components(path, description)
    except SecondRecenterGridBuildError as error:
        raise _fail(str(error)) from error


def _regular_directory(path: Path, description: str) -> Path:
    try:
        return _second_recenter_regular_directory(path, description)
    except SecondRecenterGridBuildError as error:
        raise _fail(str(error)) from error


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


def _safe_sum(values: Sequence[int], field_name: str) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise _fail(f"{field_name} contains an invalid count")
        if value > MAX_SAFE_INTEGER - total:
            raise _fail(f"{field_name} exceeds the safe integer range")
        total += value
    return total


def _canonical_curve_basis(
    coordinates: Sequence[Any],
    *,
    center: float,
    scale: float,
    shape: float,
) -> tuple[float, ...]:
    """Evaluate the exact CurveGrid symmetric radial-amplification basis."""

    normalized_center = _finite_number(center, "frozen center")
    normalized_scale = _finite_number(scale, "frozen scale")
    normalized_shape = _finite_number(shape, "frozen shape")
    if normalized_scale <= 0.0 or normalized_shape <= 0.0:
        raise _fail("frozen scale and shape must be positive")

    bases: list[float] = []
    for index, value in enumerate(coordinates):
        coordinate = _finite_number(value, f"coordinates[{index}]")
        try:
            difference = coordinate - normalized_center
            z = difference / normalized_scale
            shape_squared = normalized_shape * normalized_shape
            z_squared = z * z
            u_squared = shape_squared + z_squared
            u = math.sqrt(u_squared)
            numerator = u_squared + 2.0
            rooted = math.sqrt(u_squared + 4.0)
            denominator = u * rooted
            basis = numerator / denominator
        except (OverflowError, ValueError, ZeroDivisionError) as error:
            raise _fail("canonical curve basis is numerically invalid") from error
        intermediates = (
            difference,
            z,
            shape_squared,
            z_squared,
            u_squared,
            u,
            numerator,
            rooted,
            denominator,
            basis,
        )
        if not all(math.isfinite(item) for item in intermediates):
            raise _fail("canonical curve basis is non-finite")
        bases.append(basis)
    return tuple(bases)


def _fit_series(
    coordinates: Sequence[Any],
    values: Sequence[Any],
    inverse_variances: Sequence[Any],
    *,
    center: float,
    scale: float,
    shape: float,
) -> _SeriesFit:
    if not (
        len(coordinates) == len(values) == len(inverse_variances)
        and len(coordinates) >= 2
    ):
        raise _fail("series arrays must have equal length and at least two samples")

    numeric_values = tuple(
        _finite_number(value, f"values[{index}]")
        for index, value in enumerate(values)
    )
    weights = tuple(
        _finite_number(value, f"inverseVariances[{index}]")
        for index, value in enumerate(inverse_variances)
    )
    if any(weight < 0.0 for weight in weights):
        raise _fail("inverse variances must be nonnegative")
    positive_count = sum(weight > 0.0 for weight in weights)
    if positive_count < 2:
        raise _fail("series has insufficient positive-weight samples")

    bases = _canonical_curve_basis(
        coordinates,
        center=center,
        scale=scale,
        shape=shape,
    )
    total_weight = 0.0
    weighted_basis = 0.0
    weighted_basis_squared = 0.0
    weighted_value = 0.0
    weighted_basis_value = 0.0
    for weight, basis, value in zip(weights, bases, numeric_values):
        terms = (
            weight,
            weight * basis,
            weight * basis * basis,
            weight * value,
            weight * basis * value,
        )
        if not all(math.isfinite(item) for item in terms):
            raise _fail("weighted normal-equation term is non-finite")
        total_weight += terms[0]
        weighted_basis += terms[1]
        weighted_basis_squared += terms[2]
        weighted_value += terms[3]
        weighted_basis_value += terms[4]
        if not all(
            math.isfinite(item)
            for item in (
                total_weight,
                weighted_basis,
                weighted_basis_squared,
                weighted_value,
                weighted_basis_value,
            )
        ):
            raise _fail("weighted normal-equation accumulation is non-finite")

    determinant_left = total_weight * weighted_basis_squared
    determinant_right = weighted_basis * weighted_basis
    determinant = determinant_left - determinant_right
    determinant_limit = _SINGULARITY_RELATIVE_TOLERANCE * max(
        abs(determinant_left),
        abs(determinant_right),
        1.0,
    )
    if not all(
        math.isfinite(item)
        for item in (
            determinant_left,
            determinant_right,
            determinant,
            determinant_limit,
        )
    ) or determinant <= determinant_limit:
        raise _fail("weighted two-parameter fit is singular or numerically invalid")

    offset_numerator_left = weighted_value * weighted_basis_squared
    offset_numerator_right = weighted_basis_value * weighted_basis
    amplitude_numerator_left = total_weight * weighted_basis_value
    amplitude_numerator_right = weighted_basis * weighted_value
    numerator_terms = (
        offset_numerator_left,
        offset_numerator_right,
        amplitude_numerator_left,
        amplitude_numerator_right,
    )
    if not all(math.isfinite(item) for item in numerator_terms):
        raise _fail("weighted fit numerator is non-finite")
    offset = (offset_numerator_left - offset_numerator_right) / determinant
    amplitude = (
        amplitude_numerator_left - amplitude_numerator_right
    ) / determinant
    if not math.isfinite(offset) or not math.isfinite(amplitude):
        raise _fail("fitted nuisance parameters are non-finite")

    model_values: list[float] = []
    residual_values: list[float] = []
    weighted_residual_sum_squares = 0.0
    maximum_standardized = -1.0
    maximum_index = 0
    for index, (weight, basis, value) in enumerate(
        zip(weights, bases, numeric_values)
    ):
        model = offset + amplitude * basis
        residual = value - model
        weighted_residual = weight * residual * residual
        standardized = abs(residual) * math.sqrt(weight)
        if not all(
            math.isfinite(item)
            for item in (model, residual, weighted_residual, standardized)
        ):
            raise _fail("fitted model or residual is non-finite")
        weighted_residual_sum_squares += weighted_residual
        if not math.isfinite(weighted_residual_sum_squares):
            raise _fail("weighted residual sum of squares is non-finite")
        model_values.append(model)
        residual_values.append(residual)
        if standardized > maximum_standardized:
            maximum_standardized = standardized
            maximum_index = index

    return _SeriesFit(
        basis=bases,
        offset=offset,
        amplitude=amplitude,
        model_values=tuple(model_values),
        residual_values=tuple(residual_values),
        weighted_residual_sum_squares=weighted_residual_sum_squares,
        positive_weight_sample_count=positive_count,
        maximum_absolute_standardized_residual=maximum_standardized,
        maximum_absolute_standardized_residual_index=maximum_index,
    )


def _verify_second_recenter_project(
    root: Path,
    coarse: Any,
    coarse_winner: Any,
    coarse_investigation_id: str,
    refinement: Any,
    refinement_winner: Any,
    refinement_investigation_id: str,
    first_recenter: _VerifiedFirstRecenterProject,
    first_winner: _VerifiedFirstRecenterWinner,
) -> _VerifiedFirstRecenterProject:
    _verify_exact_project_tree(
        root,
        SECOND_RECENTER_CONTRACT_RELATIVE_PATH,
        "second-recenter project",
    )
    build_bytes, build = _read_json_file(
        root / SECOND_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
        "second-recenter build manifest",
    )
    if set(build) != _SECOND_RECENTER_BUILD_FIELDS:
        raise _fail("second-recenter build manifest field set is invalid")
    if (
        build.get("buildManifestSchemaID")
        != SECOND_RECENTER_BUILD_MANIFEST_SCHEMA_ID
        or build.get("buildManifestVersion")
        != SECOND_RECENTER_BUILD_MANIFEST_VERSION
    ):
        raise _fail("second-recenter build manifest schema is invalid")

    axes = _expected_second_recenter_axes(first_recenter, first_winner)
    expected_contract = _expected_second_recenter_contract(
        coarse,
        coarse_winner,
        coarse_investigation_id,
        refinement,
        refinement_winner,
        refinement_investigation_id,
        first_recenter,
        first_winner,
        axes,
    )
    contract_bytes, contract = _read_json_file(
        root / SECOND_RECENTER_CONTRACT_RELATIVE_PATH,
        "second-recenter search contract",
    )
    if contract != expected_contract or contract_bytes != _stable_json_bytes(
        expected_contract
    ):
        raise _fail("second-recenter contract does not match verified ancestry")
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    contract_file_sha256 = _sha256_bytes(contract_bytes)

    project_bytes, project = _read_json_file(
        root / SECOND_RECENTER_PROJECT_RELATIVE_PATH,
        "second-recenter project manifest",
    )
    project_id = project.get("id")
    if (
        not isinstance(project_id, str)
        or _SAFE_PROJECT_ID.fullmatch(project_id) is None
    ):
        raise _fail("second-recenter project ID is malformed or unsafe")
    expected_dataset = _expected_second_recenter_dataset(
        project_id,
        coarse,
        first_recenter,
        first_winner,
        axes,
        contract_sha256,
    )
    dataset_bytes, dataset = _read_json_file(
        root / SECOND_RECENTER_DATASET_RELATIVE_PATH,
        "second-recenter dataset",
    )
    if dataset != expected_dataset or dataset_bytes != _stable_json_bytes(
        expected_dataset
    ):
        raise _fail("second-recenter dataset does not match verified ancestry")
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise _fail("second-recenter dataset ID is invalid")
    expected_project = _expected_second_recenter_project(project_id, dataset_id)
    if project != expected_project or project_bytes != _stable_json_bytes(
        expected_project
    ):
        raise _fail("second-recenter project manifest does not match")

    dataset_sha256 = _sha256_bytes(dataset_bytes)
    project_sha256 = _sha256_bytes(project_bytes)
    expected_paths = {
        "buildManifest": SECOND_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
        "dataset": SECOND_RECENTER_DATASET_RELATIVE_PATH,
        "project": SECOND_RECENTER_PROJECT_RELATIVE_PATH,
        "secondRecenterSearchContract": SECOND_RECENTER_CONTRACT_RELATIVE_PATH,
    }
    evaluation_count = (
        coarse.selected_series.sample_count * TOTAL_CANDIDATE_COUNT
    )
    if evaluation_count > MAX_SAFE_INTEGER:
        raise _fail("second-recenter evaluation count exceeds the safe range")
    expected_build = {
        "acceptedFirstRecenterWinner": _winner_record(first_winner),
        "blindnessStatement": expected_contract["identityIsolationStatement"],
        "buildManifestSchemaID": SECOND_RECENTER_BUILD_MANIFEST_SCHEMA_ID,
        "buildManifestVersion": SECOND_RECENTER_BUILD_MANIFEST_VERSION,
        "candidateCount": TOTAL_CANDIDATE_COUNT,
        "contractSchemaID": SECOND_RECENTER_GRID_CONTRACT_ID,
        "contractVersion": SECOND_RECENTER_GRID_CONTRACT_VERSION,
        "derivedSecondRecenterAxes": {
            key: dict(value) for key, value in axes.items()
        },
        "expectedSampleCandidateEvaluationCount": evaluation_count,
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "inputSeriesSHA256": coarse.selected_series.sha256,
        "outputContractFileSHA256": contract_file_sha256,
        "outputDatasetSHA256": dataset_sha256,
        "outputProjectSHA256": project_sha256,
        "parentArtifactHashes": expected_contract["parentArtifactHashes"],
        "parentAxes": expected_contract["parentAxes"],
        "parentInvestigationIDs": expected_contract["parentInvestigationIDs"],
        "parentProjectIDs": expected_contract["parentProjectIDs"],
        "projectID": project_id,
        "relativeArtifactPaths": expected_paths,
        "secondRecenterSearchContractSHA256": contract_sha256,
        "selectedSampleCount": coarse.selected_series.sample_count,
        "selectedSeriesID": coarse.selected_series.series_id,
    }
    if build != expected_build or build_bytes != _stable_json_bytes(expected_build):
        raise _fail("second-recenter build manifest provenance is incomplete")

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


def _investigation_hashes(winner: Any) -> dict[str, Any]:
    return {
        "investigationRecordSHA256": winner.investigation_sha256,
        "runStageLedgerSHA256": winner.run_stage_ledger_sha256,
        "stageLedgerSHA256s": dict(winner.stage_ledger_sha256s),
    }


def _residual_contract() -> dict[str, Any]:
    return {
        "canonicalCurveFamilyID": FAMILY_ID,
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant "
            "whitespace, non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractID": RESIDUAL_PREPARATION_CONTRACT_ID,
        "contractVersion": RESIDUAL_PREPARATION_CONTRACT_VERSION,
        "convergenceRule": (
            "The verified second-recenter winner is interior on every axis "
            "and its bestCenter, bestLogScale, bestLogShape, and "
            "bestWeightedResidualSumSquares exactly equal the verified "
            "first-recenter accepted winner."
        ),
        "fitRule": (
            "For every prepared generic series in orderedSeriesIDs order, "
            "fit only offset and amplitude by deterministic weighted two-"
            "parameter normal equations at the frozen shared geometry."
        ),
        "identityIsolationStatement": (
            "Sealed identity, archive sources, source filenames, event names, "
            "catalog identifiers, publications, sky coordinates, and "
            "published event parameters were not read or consulted."
        ),
        "maximumResidualTieRule": (
            "Maximum absolute standardized residual ties select the earliest "
            "sample in the source series order."
        ),
        "modelBasisRule": (
            "uSquared = exp(bestLogShape)^2 + "
            "((coordinate - bestCenter) / exp(bestLogScale))^2; "
            "basis = (uSquared + 2) / (sqrt(uSquared) * "
            "sqrt(uSquared + 4))."
        ),
        "modelScopeStatement": (
            "This phase prepares residuals after smooth-model convergence; it "
            "does not detect, classify, or claim a planetary anomaly."
        ),
        "residualSignRule": "residual = observed - model",
        "singularityRelativeTolerance": _SINGULARITY_RELATIVE_TOLERANCE,
    }


def prepare_blind_microlensing_residuals(
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
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify convergence and atomically publish all generic residual series."""

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
    second_recenter_root = Path(second_recenter_project_root).expanduser().absolute()
    second_recenter_investigation = Path(
        second_recenter_investigation_record
    ).expanduser().absolute()
    paths = (
        (prepared, "prepared root"),
        (coarse_root, "coarse project root"),
        (coarse_investigation, "coarse investigation record"),
        (refinement_root, "first-refinement project root"),
        (refinement_investigation, "first-refinement investigation record"),
        (first_recenter_root, "first-recenter project root"),
        (first_recenter_investigation, "first-recenter investigation record"),
        (second_recenter_root, "second-recenter project root"),
        (second_recenter_investigation, "second-recenter investigation record"),
    )
    for path, description in paths:
        _reject_symlink_components(path, description)
    _regular_directory(prepared, "prepared root")

    try:
        preparation = _verify_blind_preparation(prepared)
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
        first_winner = _verify_first_recenter_investigation(
            first_recenter_investigation,
            first_recenter,
            first_recenter_root / FIRST_RECENTER_PROJECT_RELATIVE_PATH,
        )

        second_recenter = _verify_second_recenter_project(
            second_recenter_root,
            coarse,
            coarse_winner,
            coarse_investigation_id,
            refinement,
            refinement_winner,
            refinement_investigation_id,
            first_recenter,
            first_winner,
        )
        second_winner = _verify_first_recenter_investigation(
            second_recenter_investigation,
            second_recenter,
            second_recenter_root / SECOND_RECENTER_PROJECT_RELATIVE_PATH,
        )
    except ResidualPreparationError:
        raise
    except (
        CoarseGridBuildError,
        RefinementGridBuildError,
        FirstRecenterGridBuildError,
        SecondRecenterGridBuildError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise _fail(str(error)) from error

    if second_winner.boundary_axes:
        raise _fail("second-recenter winner is on a grid boundary")
    first_convergence_values = (
        first_winner.center,
        first_winner.log_scale,
        first_winner.log_shape,
        first_winner.objective,
    )
    second_convergence_values = (
        second_winner.center,
        second_winner.log_scale,
        second_winner.log_shape,
        second_winner.objective,
    )
    if second_convergence_values != first_convergence_values:
        raise _fail("first- and second-recenter winners or objectives differ")

    try:
        scale = math.exp(second_winner.log_scale)
        shape = math.exp(second_winner.log_shape)
    except OverflowError as error:
        raise _fail("frozen geometry cannot be exponentiated safely") from error
    if (
        not math.isfinite(scale)
        or not math.isfinite(shape)
        or scale <= 0.0
        or shape <= 0.0
    ):
        raise _fail("frozen geometry is non-finite or non-positive")

    second_investigation_id = second_winner.investigation_id
    first_investigation_id = first_winner.investigation_id
    parent_artifact_hashes = _ancestry_hashes_through_first_recenter(
        coarse,
        coarse_winner,
        refinement,
        refinement_winner,
        first_recenter,
        first_winner,
    )
    parent_artifact_hashes["secondRecenter"] = {
        "buildManifestSHA256": second_recenter.build_manifest_sha256,
        "contractFileSHA256": second_recenter.contract_file_sha256,
        "contractSHA256": second_recenter.contract_sha256,
        "datasetSHA256": second_recenter.dataset_sha256,
        "projectSHA256": second_recenter.project_sha256,
        **_investigation_hashes(second_winner),
    }
    parent_project_ids = {
        "coarse": coarse.project_id,
        "firstRecenter": first_recenter.project_id,
        "firstRefinement": refinement.project_id,
        "secondRecenter": second_recenter.project_id,
    }
    parent_investigation_ids = {
        "coarse": coarse_investigation_id,
        "firstRecenter": first_investigation_id,
        "firstRefinement": refinement_investigation_id,
        "secondRecenter": second_investigation_id,
    }
    convergence_evidence = {
        "comparedFields": [
            "bestCenter",
            "bestLogScale",
            "bestLogShape",
            "bestWeightedResidualSumSquares",
        ],
        "exactEquality": True,
        "firstRecenterWinner": _winner_record(first_winner),
        "secondRecenterInteriorOnEveryAxis": True,
        "secondRecenterWinner": _winner_record(second_winner),
    }
    frozen_geometry = {
        "center": second_winner.center,
        "logScale": second_winner.log_scale,
        "logShape": second_winner.log_shape,
        "scale": scale,
        "shape": shape,
    }
    geometry_provenance = {
        "convergenceEvidence": convergence_evidence,
        "curveFamilyID": FAMILY_ID,
        "parentArtifactHashes": parent_artifact_hashes,
        "parentInvestigationIDs": parent_investigation_ids,
        "parentProjectIDs": parent_project_ids,
    }
    geometry_provenance_sha256 = _sha256_bytes(
        _canonical_compact_json_bytes(geometry_provenance)
    )

    contract = _residual_contract()
    contract_bytes = _stable_json_bytes(contract)
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    series_documents: list[tuple[str, dict[str, Any], bytes]] = []
    series_manifest_records: list[dict[str, Any]] = []
    wrss_values: list[float] = []
    sample_counts: list[int] = []
    for ordinal, series in enumerate(preparation.ordered_series, start=1):
        payload = series.payload
        coordinates = payload["coordinates"]
        values = payload["values"]
        inverse_variances = payload["inverseVariances"]
        fit = _fit_series(
            coordinates,
            values,
            inverse_variances,
            center=second_winner.center,
            scale=scale,
            shape=shape,
        )
        relative_path = f"{SERIES_DIRECTORY}/residual-series-{ordinal:03d}.json"
        maximum_index = fit.maximum_absolute_standardized_residual_index
        document = {
            "canonicalCurveFamilyID": FAMILY_ID,
            "coordinates": list(coordinates),
            "fitDiagnostics": {
                "amplitudeSign": (
                    "positive"
                    if fit.amplitude > 0.0
                    else "negative"
                    if fit.amplitude < 0.0
                    else "zero"
                ),
                "maximumAbsoluteStandardizedResidual": (
                    fit.maximum_absolute_standardized_residual
                ),
                "maximumAbsoluteStandardizedResidualCoordinate": coordinates[
                    maximum_index
                ],
                "maximumAbsoluteStandardizedResidualIndex": maximum_index,
                "maximumTieRule": contract["maximumResidualTieRule"],
                "positiveWeightSampleCount": fit.positive_weight_sample_count,
                "weightedResidualSumSquares": (
                    fit.weighted_residual_sum_squares
                ),
            },
            "fittedAmplitude": fit.amplitude,
            "fittedOffset": fit.offset,
            "frozenGeometry": dict(frozen_geometry),
            "geometryProvenanceSHA256": geometry_provenance_sha256,
            "genericSeriesID": series.series_id,
            "inputSeriesSHA256": series.sha256,
            "inverseVariances": list(inverse_variances),
            "modelValues": list(fit.model_values),
            "observedValues": list(values),
            "residualPreparationContractID": RESIDUAL_PREPARATION_CONTRACT_ID,
            "residualPreparationContractSHA256": contract_sha256,
            "residualSeriesSchemaID": RESIDUAL_SERIES_SCHEMA_ID,
            "residualSeriesVersion": RESIDUAL_SERIES_VERSION,
            "residualValues": list(fit.residual_values),
            "sampleCount": series.sample_count,
        }
        document_bytes = _stable_json_bytes(document)
        output_sha256 = _sha256_bytes(document_bytes)
        series_documents.append((relative_path, document, document_bytes))
        series_manifest_records.append(
            {
                "genericSeriesID": series.series_id,
                "inputSeriesSHA256": series.sha256,
                "outputFile": relative_path,
                "outputSHA256": output_sha256,
                "sampleCount": series.sample_count,
                "weightedResidualSumSquares": fit.weighted_residual_sum_squares,
            }
        )
        wrss_values.append(fit.weighted_residual_sum_squares)
        sample_counts.append(series.sample_count)

    total_wrss = 0.0
    for value in wrss_values:
        total_wrss += value
        if not math.isfinite(total_wrss):
            raise _fail("total residual WRSS is non-finite")
    manifest = {
        "canonicalCurveFamilyID": FAMILY_ID,
        "contractID": RESIDUAL_PREPARATION_CONTRACT_ID,
        "contractSHA256": contract_sha256,
        "contractVersion": RESIDUAL_PREPARATION_CONTRACT_VERSION,
        "convergenceEvidence": convergence_evidence,
        "frozenGeometry": frozen_geometry,
        "geometryProvenanceSHA256": geometry_provenance_sha256,
        "identityIsolationStatement": contract["identityIsolationStatement"],
        "modelScopeStatement": contract["modelScopeStatement"],
        "orderedGenericSeriesIDs": [
            series.series_id for series in preparation.ordered_series
        ],
        "parentArtifactHashes": parent_artifact_hashes,
        "parentInvestigationIDs": parent_investigation_ids,
        "parentProjectIDs": parent_project_ids,
        "preparationManifestSHA256": preparation.manifest_sha256,
        "residualManifestSchemaID": RESIDUAL_MANIFEST_SCHEMA_ID,
        "residualManifestVersion": RESIDUAL_MANIFEST_VERSION,
        "series": series_manifest_records,
        "totalSampleCount": _safe_sum(sample_counts, "total sample count"),
        "totalSeriesCount": len(series_manifest_records),
        "totalWeightedResidualSumSquares": total_wrss,
        "verifiedFirstRecenterWinner": _winner_record(first_winner),
        "verifiedSecondRecenterWinner": _winner_record(second_winner),
    }
    manifest_bytes = _stable_json_bytes(manifest)
    try:
        _assert_identity_free(
            (
                contract_bytes,
                manifest_bytes,
                *(item[2] for item in series_documents),
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
        for relative_path, _, document_bytes in series_documents:
            _atomic_write_bytes(staging / relative_path, document_bytes)
        _atomic_write_bytes(staging / MANIFEST_RELATIVE_PATH, manifest_bytes)
        if output.exists() or output.is_symlink():
            raise _fail("output root already exists")
        staging.rename(output)
    except Exception as error:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if isinstance(error, ResidualPreparationError):
            raise
        raise _fail("atomic output publication failed") from error

    return {
        "contract": contract,
        "manifest": manifest,
        "series": [item[1] for item in series_documents],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify blind smooth-model convergence and prepare all generic "
            "residual series."
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
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = prepare_blind_microlensing_residuals(
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
        output_root=arguments.output_root,
    )
    manifest = result["manifest"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind microlensing residual preparation ready")
    print(f"generic series: {manifest['totalSeriesCount']}")
    print(f"total samples: {manifest['totalSampleCount']}")
    print(f"manifest: {output / MANIFEST_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
