"""Prepare blind residual evidence for later anomaly-morphology modeling."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openstar_workloads.plugins.curve_grid import FAMILY_ID, MAX_SAFE_INTEGER
from workflows.microlensing.coarse_grid import (
    CoarseGridBuildError,
    _assert_identity_free,
    _atomic_write_bytes,
    _canonical_compact_json_bytes,
    _stable_json_bytes,
)
from workflows.microlensing.prepare_residuals import ResidualPreparationError
from workflows.microlensing.refine_grid import RefinementGridBuildError
from workflows.microlensing.residual_grid import (
    PROJECT_RELATIVE_PATH as GRID_PROJECT_RELATIVE_PATH,
    ResidualGridBuildError,
)
from workflows.microlensing.validate_residual_grid import (
    CONFIRMED_NEXT_TEST,
    CONFIRMED_STATUS,
    CONTRACT_RELATIVE_PATH as CROSS_VALIDATION_CONTRACT_RELATIVE_PATH,
    CROSS_VALIDATION_CONTRACT_ID,
    CROSS_VALIDATION_CONTRACT_VERSION,
    MINIMUM_TWO_WIDTH_SUPPORT,
    POSITIVE_CLASSIFICATION,
    RESULT_RELATIVE_PATH as CROSS_VALIDATION_RESULT_RELATIVE_PATH,
    ResidualGridValidationError,
    VALIDATION_DELTA_WRSS_THRESHOLD,
    _VerifiedGrid,
    _VerifiedInvestigation,
    _VerifiedResiduals,
    _contract as _expected_cross_validation_contract,
    _exact_count as _validation_exact_count,
    _finite_number as _validation_finite_number,
    _read_json_file,
    _regular_directory,
    _reject_symlink_components,
    _result as _expected_cross_validation_result,
    _sha256_bytes,
    _verify_grid_root,
    _verify_investigation,
    _verify_residual_root,
)


MORPHOLOGY_PREPARATION_SCHEMA_ID = (
    "openstar.microlensing-anomaly-morphology-preparation.v1"
)
MORPHOLOGY_PREPARATION_VERSION = "1.0"
MORPHOLOGY_CONTRACT_ID = (
    "openstar.microlensing-anomaly-morphology-contract.v1"
)
MORPHOLOGY_CONTRACT_VERSION = "1.0"
MORPHOLOGY_FAMILY_ID = "openstar.microlensing-residual-morphology.v1"
MORPHOLOGY_DATASET_SCHEMA_ID = (
    "openstar.microlensing-anomaly-morphology-dataset.v1"
)
MORPHOLOGY_DATASET_VERSION = "1.0"
ARTIFACT_MANIFEST_SCHEMA_ID = (
    "openstar.microlensing-anomaly-morphology-artifact-manifest.v1"
)
ARTIFACT_MANIFEST_VERSION = "1.0"

PREPARATION_RELATIVE_PATH = "anomaly-morphology-preparation.json"
CONTRACT_RELATIVE_PATH = "morphology-contract.json"
MANIFEST_RELATIVE_PATH = "artifact-manifest.json"
DATASET_DIRECTORY = "datasets"

POSITIVE_PULSE_ONLY = "POSITIVE_PULSE_ONLY"
ORDERED_NEGATIVE_POSITIVE_DOUBLET = "ORDERED_NEGATIVE_POSITIVE_DOUBLET"
INDEPENDENT_PULSES = "INDEPENDENT_PULSES"
MODEL_CLASS_IDS = (
    POSITIVE_PULSE_ONLY,
    ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    INDEPENDENT_PULSES,
)
NEXT_TEST = "DISTRIBUTED_BLIND_MICROLENSING_ANOMALY_MORPHOLOGY_GRID"

COMPONENT_SUPPORT_WIDTH_MULTIPLIER = 2.0
MINIMUM_COMPONENT_POSITIVE_WEIGHT_SUPPORT = 1
MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE = 3
MINIMUM_WIDTH_DIVISOR = 16.0
MAXIMUM_WIDTH_WINDOW_DIVISOR = 4.0

ORDERED_OVER_POSITIVE_MINIMUM_DELTA_WRSS = 30.0
ORDERED_OVER_POSITIVE_MINIMUM_PER_SERIES_DELTA_WRSS = 9.0
ORDERED_OVER_POSITIVE_MINIMUM_DELTA_BIC = 10.0
INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_WRSS = 18.0
INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_BIC = 10.0

RANK_RELATIVE_TOLERANCE = 1.0e-12
OBJECTIVE_RELATIVE_TOLERANCE = 1.0e-9
CONSTRAINT_TOLERANCE = 0.0
TIMING_COMPARISON_TOLERANCE = 0.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "archivefilename",
        "archiveurl",
        "catalogidentifier",
        "citation",
        "eventidentity",
        "eventname",
        "observatory",
        "observatoryid",
        "sealedmetadata",
        "skycoordinates",
        "sourcefilename",
        "starid",
        "uid",
    }
)


class AnomalyMorphologyPreparationError(RuntimeError):
    """Blind anomaly-morphology evidence cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class _VerifiedCrossValidation:
    contract: Mapping[str, Any]
    result: Mapping[str, Any]
    contract_file_sha256: str
    result_file_sha256: str


@dataclass(frozen=True, slots=True)
class _ComponentGeometry:
    generic_series_id: str
    center: float
    log_scale: float
    log_shape: float
    effective_width: float
    boundary_axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedWindow:
    minimum: float
    maximum: float
    core_minimum: float
    core_maximum: float
    negative: _ComponentGeometry
    positive: _ComponentGeometry
    datasets: tuple[Mapping[str, Any], ...]


def _fail(message: str) -> AnomalyMorphologyPreparationError:
    return AnomalyMorphologyPreparationError(message)


def _finite_number(value: Any, field_name: str) -> float:
    try:
        return _validation_finite_number(value, field_name)
    except ResidualGridValidationError as error:
        raise _fail(str(error)) from error


def _exact_count(value: Any, field_name: str) -> int:
    try:
        return _validation_exact_count(value, field_name)
    except ResidualGridValidationError as error:
        raise _fail(str(error)) from error


def _sha256_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{field_name} must be a lowercase SHA-256")
    return value


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field_name} must be a nonempty string")
    return value


def _safe_product(values: Sequence[int], field_name: str) -> int:
    product = 1
    for value in values:
        count = _exact_count(value, field_name)
        if count == 0:
            return 0
        if product > MAX_SAFE_INTEGER // count:
            raise _fail(f"{field_name} exceeds the safe integer range")
        product *= count
    return product


def _safe_sum(values: Sequence[int], field_name: str) -> int:
    total = 0
    for value in values:
        count = _exact_count(value, field_name)
        if total > MAX_SAFE_INTEGER - count:
            raise _fail(f"{field_name} exceeds the safe integer range")
        total += count
    return total


def _independent_candidate_layout(
    *,
    center_count: int,
    log_scale_count: int,
    log_shape_count: int,
    admitted_ids: Sequence[str],
    global_offset: int,
) -> dict[str, Any]:
    """Build concatenated per-series searches, never a cross-series product."""

    center_count = _exact_count(center_count, "independent center count")
    log_scale_count = _exact_count(
        log_scale_count, "independent log-scale count"
    )
    log_shape_count = _exact_count(
        log_shape_count, "independent log-shape count"
    )
    global_offset = _exact_count(global_offset, "independent global offset")
    if center_count < 2 or log_scale_count < 1 or log_shape_count < 1:
        raise _fail("independent search axes are empty")
    if len(set(admitted_ids)) != len(admitted_ids) or not admitted_ids:
        raise _fail("independent series order is invalid")

    ordered_center_pair_count = _safe_product(
        [center_count, center_count - 1], "ordered center-pair numerator"
    ) // 2
    per_series_count = _safe_product(
        [
            ordered_center_pair_count,
            log_scale_count,
            log_shape_count,
            log_scale_count,
            log_shape_count,
        ],
        "independent per-series candidate count",
    )
    total_count = _safe_product(
        [per_series_count, len(admitted_ids)],
        "independent total candidate count",
    )
    if global_offset > MAX_SAFE_INTEGER - total_count:
        raise _fail("independent candidate offset exceeds the safe integer range")

    searches: list[dict[str, Any]] = []
    for ordinal, series_id in enumerate(admitted_ids):
        relative_offset = _safe_product(
            [ordinal, per_series_count], "series offset"
        )
        start = _safe_sum(
            [global_offset, relative_offset],
            "independent series offset",
        )
        end = _safe_sum([start, per_series_count], "independent series end")
        searches.append(
            {
                "candidateCount": per_series_count,
                "canonicalSeriesIndex": ordinal,
                "genericSeriesID": _nonempty_string(
                    series_id, "independent generic series ID"
                ),
                "globalEndExclusive": end,
                "globalStartIndex": start,
            }
        )
    return {
        "centerPairIndexFormula": (
            "pairIndex = negativeCenterIndex * "
            "(2 * centerCount - negativeCenterIndex - 1) // 2 + "
            "(positiveCenterIndex - negativeCenterIndex - 1)"
        ),
        "independentSeriesSearches": searches,
        "localMixedRadixFormula": (
            "localIndex = ((((pairIndex * logScaleCount + "
            "negativeLogScaleIndex) * logShapeCount + "
            "negativeLogShapeIndex) * logScaleCount + "
            "positiveLogScaleIndex) * logShapeCount + "
            "positiveLogShapeIndex)"
        ),
        "orderedCenterPairCount": ordered_center_pair_count,
        "orderedCenterPairRule": (
            "Enumerate negativeCenterIndex ascending, then positiveCenterIndex "
            "ascending, retaining exactly 0 <= negativeCenterIndex < "
            "positiveCenterIndex < centerCount."
        ),
        "perSeriesCandidateCount": per_series_count,
        "totalCandidateCount": total_count,
    }


def _assert_identity_isolated(value: Any) -> None:
    """Reject exact identity-bearing field names without short-token scans."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail("blind output keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            if normalized in _FORBIDDEN_IDENTITY_KEYS:
                raise _fail(f"blind output contains forbidden identity field {key}")
            _assert_identity_isolated(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_identity_isolated(item)


def _identity_check(documents: Sequence[Mapping[str, Any]]) -> None:
    for document in documents:
        _assert_identity_isolated(document)
    try:
        _assert_identity_free(tuple(_stable_json_bytes(item) for item in documents))
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error


def _verify_cross_validation_root(
    root: Path,
    *,
    residuals: _VerifiedResiduals,
    grid: _VerifiedGrid,
    investigation: _VerifiedInvestigation,
) -> _VerifiedCrossValidation:
    cross_root = _regular_directory(root, "residual cross-validation root")
    try:
        names = {entry.name for entry in cross_root.iterdir()}
    except OSError as error:
        raise _fail("residual cross-validation root is unreadable") from error
    if names != {
        CROSS_VALIDATION_CONTRACT_RELATIVE_PATH,
        CROSS_VALIDATION_RESULT_RELATIVE_PATH,
    }:
        raise _fail("residual cross-validation artifact set is invalid")

    contract_bytes, contract = _read_json_file(
        cross_root / CROSS_VALIDATION_CONTRACT_RELATIVE_PATH,
        "residual cross-validation contract",
    )
    expected_contract = _expected_cross_validation_contract()
    if contract != expected_contract:
        raise _fail("residual cross-validation contract is invalid")
    canonical_contract_sha256 = _sha256_bytes(
        _canonical_compact_json_bytes(contract)
    )

    result_bytes, result = _read_json_file(
        cross_root / CROSS_VALIDATION_RESULT_RELATIVE_PATH,
        "residual cross-validation result",
    )
    expected_result = _expected_cross_validation_result(
        contract=contract,
        contract_sha256=canonical_contract_sha256,
        residuals=residuals,
        grid=grid,
        investigation=investigation,
    )
    if result != expected_result:
        raise _fail(
            "residual cross-validation result does not match fresh reconstruction"
        )
    if result.get("contractID") != CROSS_VALIDATION_CONTRACT_ID:
        raise _fail("residual cross-validation contract ID is invalid")
    if result.get("contractVersion") != CROSS_VALIDATION_CONTRACT_VERSION:
        raise _fail("residual cross-validation contract version is invalid")
    if result.get("contractSHA256") != canonical_contract_sha256:
        raise _fail("residual cross-validation contract hash is invalid")
    return _VerifiedCrossValidation(
        contract=contract,
        result=result,
        contract_file_sha256=_sha256_bytes(contract_bytes),
        result_file_sha256=_sha256_bytes(result_bytes),
    )


def _component_geometry(component: Mapping[str, Any]) -> _ComponentGeometry:
    winner = component.get("discoveryWinner")
    if not isinstance(winner, Mapping):
        raise _fail("component discovery winner is malformed")
    center = _finite_number(winner.get("bestCenter"), "component center")
    log_scale = _finite_number(
        winner.get("bestLogScale"), "component log scale"
    )
    log_shape = _finite_number(
        winner.get("bestLogShape"), "component log shape"
    )
    try:
        effective_width = math.exp(log_scale) * math.exp(log_shape)
    except OverflowError as error:
        raise _fail("component effective width is invalid") from error
    if not math.isfinite(effective_width) or effective_width <= 0.0:
        raise _fail("component effective width must be positive and finite")
    axes = component.get("searchedBoundaryAxes")
    if (
        not isinstance(axes, list)
        or len(set(axes)) != len(axes)
        or any(axis not in {"center", "logScale", "logShape"} for axis in axes)
    ):
        raise _fail("component boundary-axis evidence is malformed")
    return _ComponentGeometry(
        generic_series_id=_nonempty_string(
            component.get("discoveryGenericSeriesID"),
            "component discovery series ID",
        ),
        center=center,
        log_scale=log_scale,
        log_shape=log_shape,
        effective_width=effective_width,
        boundary_axes=tuple(axes),
    )


def _confirmed_evidence(
    cross_validation: Mapping[str, Any],
    ordered_series_ids: Sequence[str],
) -> tuple[Mapping[str, Any], tuple[str, ...], _ComponentGeometry]:
    if cross_validation.get("overallClassification") != POSITIVE_CLASSIFICATION:
        raise _fail("cross-validation classification is not reproducible structure")
    if cross_validation.get("recommendedNextTest") != CONFIRMED_NEXT_TEST:
        raise _fail("cross-validation recommended next test is invalid")
    if cross_validation.get("planetaryInterpretationResolved") is not False:
        raise _fail("cross-validation resolved a planetary interpretation")
    if cross_validation.get("discoveryClaim") is not False:
        raise _fail("cross-validation made a discovery claim")
    components = cross_validation.get("validatedComponents")
    if not isinstance(components, list):
        raise _fail("cross-validation components are malformed")
    confirmed = [
        component
        for component in components
        if isinstance(component, Mapping)
        and component.get("crossSeriesConfirmed") is True
        and component.get("componentStatus") == CONFIRMED_STATUS
    ]
    if _exact_count(
        cross_validation.get("confirmedComponentCount"),
        "confirmed component count",
    ) != len(confirmed):
        raise _fail("cross-validation confirmed-component count disagrees")
    if len(confirmed) != 1:
        raise _fail("exactly one cross-series-confirmed component is required")
    component = confirmed[0]
    if component.get("discoveryCoverageComplete") is not True:
        raise _fail("confirmed component lacks complete discovery coverage")
    if component.get("discoveryGatePassed") is not True:
        raise _fail("confirmed component did not pass the discovery gate")
    positive = _component_geometry(component)
    winner = component["discoveryWinner"]
    if (
        component.get("discoveryAmplitudeSign") != "positive"
        or _finite_number(winner.get("bestAmplitude"), "confirmed amplitude")
        <= 0.0
    ):
        raise _fail("confirmed morphology component must be positive")
    validations = component.get("heldOutValidations")
    if not isinstance(validations, list):
        raise _fail("confirmed held-out validations are malformed")
    passed: list[str] = []
    for validation in validations:
        if not isinstance(validation, Mapping):
            raise _fail("held-out validation record is malformed")
        discovery_id = _nonempty_string(
            validation.get("discoveryGenericSeriesID"),
            "held-out discovery series ID",
        )
        validation_id = _nonempty_string(
            validation.get("validationGenericSeriesID"),
            "held-out validation series ID",
        )
        if discovery_id != positive.generic_series_id:
            raise _fail("held-out validation has contradictory discovery series")
        if validation_id == discovery_id:
            raise _fail("confirmed component attempts self-validation")
        if validation_id not in ordered_series_ids:
            raise _fail("held-out validation series is not admitted")
        if validation.get("heldOutValidationGatePassed") is True:
            if validation.get("status") != "EVALUATED":
                raise _fail("passed held-out validation was not evaluated")
            if validation.get("amplitudeSignMatchesDiscovery") is not True:
                raise _fail("passed held-out validation has contradictory sign")
            if validation.get("fittedAmplitudeSign") != "positive":
                raise _fail("passed held-out validation is not positive")
            if _finite_number(
                validation.get("deltaWRSS"), "held-out delta WRSS"
            ) < VALIDATION_DELTA_WRSS_THRESHOLD:
                raise _fail("passed held-out validation is below its threshold")
            if _exact_count(
                validation.get(
                    "positiveWeightSamplesWithinTwoEffectiveWidths"
                ),
                "held-out support count",
            ) < MINIMUM_TWO_WIDTH_SUPPORT:
                raise _fail("passed held-out validation lacks support")
            passed.append(validation_id)
    if len(passed) != _exact_count(
        component.get("heldOutPassingSeriesCount"),
        "held-out passing-series count",
    ):
        raise _fail("held-out passing-series count is contradictory")
    if not passed:
        raise _fail("confirmed component has no passed held-out validation")
    selected = tuple(
        series_id
        for series_id in ordered_series_ids
        if series_id == positive.generic_series_id or series_id in passed
    )
    if len(selected) != 1 + len(set(passed)):
        raise _fail("confirmed discovery and validation series are inconsistent")
    return component, selected, positive


def _preceding_negative_component(
    cross_validation: Mapping[str, Any],
    positive: _ComponentGeometry,
    canonical_ids: Sequence[str],
) -> tuple[Mapping[str, Any], _ComponentGeometry]:
    canonical_index = {
        series_id: index for index, series_id in enumerate(canonical_ids)
    }
    candidates: list[tuple[float, int, Mapping[str, Any], _ComponentGeometry]] = []
    for component in cross_validation["validatedComponents"]:
        if not isinstance(component, Mapping):
            raise _fail("cross-validation component is malformed")
        if (
            component.get("discoveryCoverageComplete") is not True
            or component.get("discoveryGatePassed") is not True
            or component.get("discoveryAmplitudeSign") != "negative"
        ):
            continue
        geometry = _component_geometry(component)
        if geometry.center >= positive.center:
            continue
        amplitude = _finite_number(
            component["discoveryWinner"].get("bestAmplitude"),
            "preceding negative amplitude",
        )
        if amplitude >= 0.0 or geometry.generic_series_id not in canonical_index:
            raise _fail("preceding negative component evidence is contradictory")
        candidates.append(
            (
                geometry.center,
                -canonical_index[geometry.generic_series_id],
                component,
                geometry,
            )
        )
    if not candidates:
        raise _fail("no discovery-gated negative component precedes the positive")
    _, _, component, geometry = max(candidates, key=lambda item: item[:2])
    return component, geometry


def _source_arrays(
    source: Mapping[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    coordinates = source.get("coordinates")
    residuals = source.get("residualValues")
    weights = source.get("inverseVariances")
    if not all(isinstance(item, list) for item in (coordinates, residuals, weights)):
        raise _fail("source residual arrays are malformed")
    if (
        not coordinates
        or len(coordinates) != len(residuals)
        or len(coordinates) != len(weights)
    ):
        raise _fail("source residual arrays are inconsistent")
    numeric_coordinates: list[float] = []
    numeric_residuals: list[float] = []
    numeric_weights: list[float] = []
    previous: float | None = None
    for index, (coordinate, residual, weight) in enumerate(
        zip(coordinates, residuals, weights)
    ):
        x = _finite_number(coordinate, f"coordinates[{index}]")
        y = _finite_number(residual, f"residualValues[{index}]")
        w = _finite_number(weight, f"inverseVariances[{index}]")
        if w < 0.0:
            raise _fail("source inverse variances must be nonnegative")
        if previous is not None and x <= previous:
            raise _fail("source coordinates must be strictly increasing")
        previous = x
        numeric_coordinates.append(x)
        numeric_residuals.append(y)
        numeric_weights.append(w)
    return numeric_coordinates, numeric_residuals, numeric_weights


def _support_count(
    coordinates: Sequence[float],
    weights: Sequence[float],
    *,
    center: float,
    width: float,
) -> int:
    radius = COMPONENT_SUPPORT_WIDTH_MULTIPLIER * width
    return sum(
        weight > 0.0 and abs(coordinate - center) <= radius
        for coordinate, weight in zip(coordinates, weights)
    )


def _prepare_window(
    sources: Sequence[Mapping[str, Any]],
    *,
    source_sha256s: Mapping[str, str],
    negative: _ComponentGeometry,
    positive: _ComponentGeometry,
) -> _PreparedWindow:
    if negative.center >= positive.center:
        raise _fail("negative component must strictly precede positive component")
    core_minimum = (
        negative.center
        - COMPONENT_SUPPORT_WIDTH_MULTIPLIER * negative.effective_width
    )
    core_maximum = (
        positive.center
        + COMPONENT_SUPPORT_WIDTH_MULTIPLIER * positive.effective_width
    )
    if not all(math.isfinite(value) for value in (core_minimum, core_maximum)):
        raise _fail("component support bounds are non-finite")

    parsed: list[tuple[Mapping[str, Any], list[float], list[float], list[float]]] = []
    left_bounds: list[float] = []
    right_bounds: list[float] = []
    for source in sources:
        series_id = _nonempty_string(
            source.get("genericSeriesID"), "source generic series ID"
        )
        coordinates, residuals, weights = _source_arrays(source)
        for label, component in (("negative", negative), ("positive", positive)):
            if _support_count(
                coordinates,
                weights,
                center=component.center,
                width=component.effective_width,
            ) < MINIMUM_COMPONENT_POSITIVE_WEIGHT_SUPPORT:
                raise _fail(f"{series_id} lacks positive-weight {label} support")
        left = [
            coordinate
            for coordinate, weight in zip(coordinates, weights)
            if weight > 0.0 and coordinate < core_minimum
        ]
        right = [
            coordinate
            for coordinate, weight in zip(coordinates, weights)
            if weight > 0.0 and coordinate > core_maximum
        ]
        if len(left) < MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE:
            raise _fail(f"{series_id} lacks left baseline support")
        if len(right) < MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE:
            raise _fail(f"{series_id} lacks right baseline support")
        left_bounds.append(
            left[-MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE]
        )
        right_bounds.append(
            right[MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE - 1]
        )
        parsed.append((source, coordinates, residuals, weights))

    window_minimum = min(left_bounds)
    window_maximum = max(right_bounds)
    if not window_minimum < core_minimum < core_maximum < window_maximum:
        raise _fail("prepared window does not bracket anomaly and baselines")

    datasets: list[Mapping[str, Any]] = []
    for source, coordinates, residuals, weights in parsed:
        series_id = source["genericSeriesID"]
        indices = [
            index
            for index, coordinate in enumerate(coordinates)
            if window_minimum <= coordinate <= window_maximum
        ]
        if not indices:
            raise _fail(f"{series_id} has no samples in prepared window")
        inclusion_reasons: list[list[str]] = []
        for index in indices:
            coordinate = coordinates[index]
            reasons: list[str] = []
            if coordinate < core_minimum:
                reasons.append("LEFT_LOCAL_BASELINE_PADDING")
            if (
                abs(coordinate - negative.center)
                <= COMPONENT_SUPPORT_WIDTH_MULTIPLIER * negative.effective_width
            ):
                reasons.append("PRECEDING_NEGATIVE_COMPONENT_SUPPORT")
            if core_minimum <= coordinate <= core_maximum:
                reasons.append("INTERCOMPONENT_LOCAL_CONTEXT")
            if (
                abs(coordinate - positive.center)
                <= COMPONENT_SUPPORT_WIDTH_MULTIPLIER * positive.effective_width
            ):
                reasons.append("CONFIRMED_POSITIVE_COMPONENT_SUPPORT")
            if coordinate > core_maximum:
                reasons.append("RIGHT_LOCAL_BASELINE_PADDING")
            if not reasons:
                raise _fail("prepared sample lacks an inclusion reason")
            inclusion_reasons.append(reasons)
        selected_coordinates = [coordinates[index] for index in indices]
        selected_residuals = [residuals[index] for index in indices]
        selected_weights = [weights[index] for index in indices]
        left_count = sum(
            weight > 0.0 and coordinate < core_minimum
            for coordinate, weight in zip(selected_coordinates, selected_weights)
        )
        right_count = sum(
            weight > 0.0 and coordinate > core_maximum
            for coordinate, weight in zip(selected_coordinates, selected_weights)
        )
        dataset = {
            "coordinates": selected_coordinates,
            "genericSeriesID": series_id,
            "inclusionReasons": inclusion_reasons,
            "inverseVariances": selected_weights,
            "morphologyDatasetSchemaID": MORPHOLOGY_DATASET_SCHEMA_ID,
            "morphologyDatasetVersion": MORPHOLOGY_DATASET_VERSION,
            "positiveWeightSupport": {
                "confirmedPositiveWithinTwoEffectiveWidths": _support_count(
                    selected_coordinates,
                    selected_weights,
                    center=positive.center,
                    width=positive.effective_width,
                ),
                "leftBaseline": left_count,
                "precedingNegativeWithinTwoEffectiveWidths": _support_count(
                    selected_coordinates,
                    selected_weights,
                    center=negative.center,
                    width=negative.effective_width,
                ),
                "rightBaseline": right_count,
            },
            "preparedCoordinateBounds": {
                "maximum": window_maximum,
                "minimum": window_minimum,
            },
            "residualValues": selected_residuals,
            "sampleCount": len(indices),
            "sourceResidualSeriesSHA256": _sha256_string(
                source_sha256s.get(series_id),
                f"{series_id} source residual series SHA-256",
            ),
            "sourceSampleIndices": indices,
        }
        datasets.append(dataset)
    return _PreparedWindow(
        minimum=window_minimum,
        maximum=window_maximum,
        core_minimum=core_minimum,
        core_maximum=core_maximum,
        negative=negative,
        positive=positive,
        datasets=tuple(datasets),
    )


def _axis(start: float, step: float, count: int) -> dict[str, Any]:
    start = _finite_number(start, "axis start")
    step = _finite_number(step, "axis step")
    count = _exact_count(count, "axis count")
    if step <= 0.0 or count < 1:
        raise _fail("axis requires positive step and count")
    endpoint = start + (count - 1) * step
    if not math.isfinite(endpoint):
        raise _fail("axis endpoint is non-finite")
    return {"count": count, "start": start, "step": step}


def _model_axes(
    window: _PreparedWindow,
    grid: _VerifiedGrid,
    admitted_ids: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    previous_axes = grid.contract.get("curveGrid")
    if not isinstance(previous_axes, Mapping):
        raise _fail("residual-grid contract axes are malformed")
    center_axis = previous_axes.get("centerAxis")
    log_scale_axis = previous_axes.get("logScaleAxis")
    if not isinstance(center_axis, Mapping) or not isinstance(log_scale_axis, Mapping):
        raise _fail("residual-grid search axes are malformed")
    previous_center_start = _finite_number(
        center_axis.get("start"), "previous center-axis start"
    )
    center_step = _finite_number(
        center_axis.get("step"), "previous center-axis step"
    )
    previous_log_scale_start = _finite_number(
        log_scale_axis.get("start"), "previous log-scale start"
    )
    log_scale_step = _finite_number(
        log_scale_axis.get("step"), "previous log-scale step"
    )
    if center_step <= 0.0 or log_scale_step <= 0.0:
        raise _fail("previous residual-grid steps must be positive")
    for component in (window.negative, window.positive):
        if (
            "logScale" not in component.boundary_axes
            or component.log_scale != previous_log_scale_start
        ):
            raise _fail("localized component is not on the minimum width boundary")

    first_center_index = max(
        0,
        math.ceil((window.minimum - previous_center_start) / center_step),
    )
    center_start = previous_center_start + first_center_index * center_step
    center_count = math.floor((window.maximum - center_start) / center_step) + 1
    center_parameters = _axis(center_start, center_step, center_count)
    center_endpoint = center_start + (center_count - 1) * center_step
    if not (
        center_start
        <= window.negative.center
        < window.positive.center
        <= center_endpoint
    ):
        raise _fail("component centers do not map onto morphology center axis")

    separation_count = math.floor((center_endpoint - center_start) / center_step)
    separation_parameters = _axis(center_step, center_step, separation_count)

    fixed_log_shapes = sorted({window.negative.log_shape, window.positive.log_shape})
    if not fixed_log_shapes or not all(
        math.isfinite(item) for item in fixed_log_shapes
    ):
        raise _fail("fixed morphology log shapes are invalid")
    shape_axis = {
        "count": len(fixed_log_shapes),
        "ordering": "strictly ascending explicit values",
        "values": fixed_log_shapes,
    }
    minimum_previous_log_scale = min(
        window.negative.log_scale, window.positive.log_scale
    )
    log_scale_start = minimum_previous_log_scale - math.log(MINIMUM_WIDTH_DIVISOR)
    maximum_width = (window.maximum - window.minimum) / MAXIMUM_WIDTH_WINDOW_DIVISOR
    if not math.isfinite(maximum_width) or maximum_width <= 0.0:
        raise _fail("maximum morphology width is invalid")
    maximum_log_scale = math.log(maximum_width) - max(fixed_log_shapes)
    log_scale_count = math.floor(
        (maximum_log_scale - log_scale_start) / log_scale_step
    ) + 1
    log_scale_parameters = _axis(log_scale_start, log_scale_step, log_scale_count)
    log_scale_endpoint = (
        log_scale_start + (log_scale_count - 1) * log_scale_step
    )
    if log_scale_endpoint < minimum_previous_log_scale:
        raise _fail("morphology width axis does not include previous boundary")

    axes: dict[str, Mapping[str, Any]] = {
        "CENTER": center_parameters,
        "LOG_SCALE": log_scale_parameters,
        "LOG_SHAPE": shape_axis,
        "SEPARATION": separation_parameters,
    }
    positive_order = ["CENTER", "LOG_SCALE", "LOG_SHAPE"]
    ordered_order = [
        "NEGATIVE_CENTER",
        "SEPARATION",
        "NEGATIVE_LOG_SCALE",
        "NEGATIVE_LOG_SHAPE",
        "POSITIVE_LOG_SCALE",
        "POSITIVE_LOG_SHAPE",
    ]
    independent_order = [
        "NEGATIVE_CENTER_PAIR_POSITIVE_CENTER",
        "NEGATIVE_LOG_SCALE",
        "NEGATIVE_LOG_SHAPE",
        "POSITIVE_LOG_SCALE",
        "POSITIVE_LOG_SHAPE",
    ]
    counts = {
        "CENTER": center_count,
        "LOG_SCALE": log_scale_count,
        "LOG_SHAPE": len(fixed_log_shapes),
        "SEPARATION": separation_count,
    }
    positive_count = _safe_product(
        [counts[item] for item in positive_order], "positive model candidates"
    )
    ordered_count = _safe_product(
        [
            center_count,
            separation_count,
            log_scale_count,
            len(fixed_log_shapes),
            log_scale_count,
            len(fixed_log_shapes),
        ],
        "ordered-doublet candidates",
    )
    ordered_offset = positive_count
    independent_offset = _safe_sum(
        [positive_count, ordered_count], "independent class offset"
    )
    independent_layout = _independent_candidate_layout(
        center_count=center_count,
        log_scale_count=log_scale_count,
        log_shape_count=len(fixed_log_shapes),
        admitted_ids=admitted_ids,
        global_offset=independent_offset,
    )
    model_counts = {
        POSITIVE_PULSE_ONLY: positive_count,
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: ordered_count,
        INDEPENDENT_PULSES: independent_layout["totalCandidateCount"],
    }
    offsets = {
        POSITIVE_PULSE_ONLY: 0,
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: ordered_offset,
        INDEPENDENT_PULSES: independent_offset,
    }
    global_count = _safe_sum(
        [positive_count, ordered_count, independent_layout["totalCandidateCount"]],
        "global candidate count",
    )
    mapping = {
        "axisSourceByModelClass": {
            INDEPENDENT_PULSES: {
                "NEGATIVE_CENTER_PAIR_POSITIVE_CENTER": "CENTER",
                "NEGATIVE_LOG_SCALE": "LOG_SCALE",
                "NEGATIVE_LOG_SHAPE": "LOG_SHAPE",
                "POSITIVE_LOG_SCALE": "LOG_SCALE",
                "POSITIVE_LOG_SHAPE": "LOG_SHAPE",
            },
            ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
                "NEGATIVE_CENTER": "CENTER",
                "NEGATIVE_LOG_SCALE": "LOG_SCALE",
                "NEGATIVE_LOG_SHAPE": "LOG_SHAPE",
                "POSITIVE_LOG_SCALE": "LOG_SCALE",
                "POSITIVE_LOG_SHAPE": "LOG_SHAPE",
                "SEPARATION": "SEPARATION",
            },
            POSITIVE_PULSE_ONLY: {
                "CENTER": "CENTER",
                "LOG_SCALE": "LOG_SCALE",
                "LOG_SHAPE": "LOG_SHAPE",
            },
        },
        "axisOrderingByModelClass": {
            INDEPENDENT_PULSES: independent_order,
            ORDERED_NEGATIVE_POSITIVE_DOUBLET: ordered_order,
            POSITIVE_PULSE_ONLY: positive_order,
        },
        "candidateCounts": model_counts,
        "globalCandidateOffsets": offsets,
        "globalCandidateCount": global_count,
        "independentPerSeriesMapping": independent_layout,
        "linearizationRule": (
            "Shared model classes use the declared class order and rightmost-"
            "fastest mixed radix. INDEPENDENT_PULSES concatenates canonical "
            "per-series searches; it never forms a product across series."
        ),
        "maximumSafeInteger": MAX_SAFE_INTEGER,
    }
    return axes, mapping


def _morphology_contract(
    window: _PreparedWindow,
    grid: _VerifiedGrid,
    admitted_ids: Sequence[str],
) -> dict[str, Any]:
    axes, candidate_mapping = _model_axes(window, grid, admitted_ids)
    series_count = len(admitted_ids)
    log_scale_endpoint = axes["LOG_SCALE"]["start"] + (
        axes["LOG_SCALE"]["count"] - 1
    ) * axes["LOG_SCALE"]["step"]
    effective_width_minimum = min(
        math.exp(axes["LOG_SCALE"]["start"] + log_shape)
        for log_shape in axes["LOG_SHAPE"]["values"]
    )
    effective_width_maximum = max(
        math.exp(log_scale_endpoint + log_shape)
        for log_shape in axes["LOG_SHAPE"]["values"]
    )
    prepared_width_limit = (
        window.maximum - window.minimum
    ) / MAXIMUM_WIDTH_WINDOW_DIVISOR
    if (
        not math.isfinite(effective_width_minimum)
        or not math.isfinite(effective_width_maximum)
        or effective_width_minimum <= 0.0
        or effective_width_maximum > prepared_width_limit
    ):
        raise _fail("derived effective-width bounds are invalid")
    parameter_counts = {
        POSITIVE_PULSE_ONLY: {
            "linear": 2 * series_count,
            "nonlinear": 3,
            "total": 2 * series_count + 3,
        },
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
            "linear": 3 * series_count,
            "nonlinear": 6,
            "total": 3 * series_count + 6,
        },
        INDEPENDENT_PULSES: {
            "linear": 3 * series_count,
            "nonlinear": 6 * series_count,
            "total": 9 * series_count,
        },
    }
    return {
        "admittedGenericSeriesIDs": list(admitted_ids),
        "axisRules": {
            "centerStepSource": "verified residual-grid center-axis step",
            "maximumEffectiveWidth": (
                "prepared coordinate span divided by four; combinations "
                "exceeding it are invalid"
            ),
            "minimumEffectiveWidth": (
                "at least sixteen times below the previous minimum-boundary "
                "winner for every fixed log shape"
            ),
            "strictlyPositiveFiniteWidthsRequired": True,
        },
        "benchmarkKind": "known-event-recovery",
        "candidateIndexMapping": candidate_mapping,
        "comparisonMetrics": {
            "AICc": {
                "formula": "WRSS + 2*k + 2*k*(k+1)/(N-k-1)",
                "undefinedRule": (
                    "When N <= k + 1, emit value null and defined false; an "
                    "undefined AICc sorts after every finite AICc."
                ),
            },
            "BIC": {
                "formula": "WRSS + k*ln(N)",
                "logarithm": "natural logarithm",
                "invalidRule": "N <= 0 or a nonfinite result is invalid",
            },
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
            "parameterCounts": parameter_counts,
        },
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant "
            "whitespace, non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractID": MORPHOLOGY_CONTRACT_ID,
        "contractVersion": MORPHOLOGY_CONTRACT_VERSION,
        "familyIdentities": {
            "componentTemplateFamilyID": FAMILY_ID,
            "componentTemplateScope": (
                "Identifies only one unit symmetric radial component, not any "
                "compound morphology model class."
            ),
            "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
            "morphologyFamilyScope": (
                "Identifies the three compound residual-morphology classes and "
                "their deterministic execution contract."
            ),
        },
        "crossSeriesRequirements": {
            "independentTimingConsistencyTolerance": axes["CENTER"]["step"],
            "timingComparisonTolerance": TIMING_COMPARISON_TOLERANCE,
            "orderedCentersAndShapesSharedAcrossSeries": True,
            "orderedPerSeriesSigns": ["negative", "positive"],
            "positivePulsePerSeriesSign": "positive",
            "positiveWeightSupportPerComponentPerSeries": (
                MINIMUM_COMPONENT_POSITIVE_WEIGHT_SUPPORT
            ),
        },
        "deterministicExecution": {
            "arithmetic": {
                "format": "IEEE-754 binary64",
                "roundingMode": "roundTiesToEven",
                "operationRules": (
                    "Evaluate operations in the written order; do not use fused "
                    "multiply-add, reassociate expressions, or parallelize reductions."
                ),
            },
            "componentBasis": {
                "equation": (
                    "scale=exp(logScale); shape=exp(logShape); "
                    "z=(coordinate-center)/scale; uSquared=shape*shape+z*z; "
                    "basis=(uSquared+2)/(sqrt(uSquared)*sqrt(uSquared+4))"
                ),
                "evaluationOrder": [
                    "scale = exp(logScale)",
                    "shape = exp(logShape)",
                    "difference = coordinate - center",
                    "z = difference / scale",
                    "shapeSquared = shape * shape",
                    "zSquared = z * z",
                    "uSquared = shapeSquared + zSquared",
                    "u = sqrt(uSquared)",
                    "numerator = uSquared + 2.0",
                    "rooted = sqrt(uSquared + 4.0)",
                    "denominator = u * rooted",
                    "basis = numerator / denominator",
                ],
                "geometryRule": (
                    "center is used directly; scale=exp(logScale) and "
                    "shape=exp(logShape) must be positive and finite."
                ),
                "nonfiniteRule": (
                    "Any nonfinite intermediate makes the candidate invalid."
                ),
            },
            "designMatrices": {
                POSITIVE_PULSE_ONLY: {
                    "columnOrder": ["INTERCEPT", "POSITIVE_COMPONENT"],
                    "rowOrder": "sourceSampleIndices order",
                },
                ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
                    "columnOrder": [
                        "INTERCEPT",
                        "NEGATIVE_COMPONENT",
                        "POSITIVE_COMPONENT",
                    ],
                    "rowOrder": "sourceSampleIndices order",
                },
                INDEPENDENT_PULSES: {
                    "columnOrder": [
                        "INTERCEPT",
                        "NEGATIVE_COMPONENT",
                        "POSITIVE_COMPONENT",
                    ],
                    "rowOrder": "sourceSampleIndices order",
                    "scope": "one independent design matrix per canonical series",
                },
            },
            "weightRules": {
                "negative": "INVALID_NEGATIVE_WEIGHT",
                "positive": (
                    "Contributes to normal equations, rank support, N, and WRSS."
                ),
                "zero": (
                    "Retain the row and published value, but do not update normal "
                    "equations or N; its WRSS contribution is exact +0.0."
                ),
            },
            "amplitudeConstraints": {
                "NEGATIVE_COMPONENT": "amplitude <= 0.0",
                "POSITIVE_COMPONENT": "amplitude >= 0.0",
                "constraintTolerance": CONSTRAINT_TOLERANCE,
                "zeroRule": (
                    "Exact zero satisfies either feasibility constraint, is labeled "
                    "zero, and fails every strict sign-evidence decision requirement."
                ),
            },
            "normalEquations": {
                "accumulation": (
                    "Initialize every Gram and right-hand-side entry to +0.0. "
                    "For each positive-weight sample in source order, update the "
                    "upper Gram triangle with j ascending from 0 and k ascending "
                    "from j, then update the right-hand side with j ascending from "
                    "0. Mirror Gram[j][k] into Gram[k][j] only after accumulation."
                ),
                "gramTerm": "weight * column[j] * column[k]",
                "rightHandSideTerm": "weight * column[j] * residualValue",
            },
            "constrainedLinearFit": {
                "activeSetOrder": {
                    POSITIVE_PULSE_ONLY: ["FREE", "ZERO"],
                    ORDERED_NEGATIVE_POSITIVE_DOUBLET: [
                        "FREE_FREE",
                        "ZERO_FREE",
                        "FREE_ZERO",
                        "ZERO_ZERO",
                    ],
                    INDEPENDENT_PULSES: [
                        "FREE_FREE",
                        "ZERO_FREE",
                        "FREE_ZERO",
                        "ZERO_ZERO",
                    ],
                },
                "twoComponentStateOrder": [
                    "NEGATIVE_COMPONENT",
                    "POSITIVE_COMPONENT",
                ],
                "algorithm": (
                    "Enumerate active states in the declared order. The offset is "
                    "always free; ZERO amplitudes are exact +0.0. Solve remaining "
                    "free columns, discard sign-infeasible states, and choose the "
                    "lowest WRSS; tolerance ties choose the lower state ordinal."
                ),
                "activeSetTieBreak": "lower declared active-state ordinal",
            },
            "linearSolve": {
                "algorithm": "Gaussian elimination with partial pivoting",
                "eliminationOrder": (
                    "Pivot columns ascend; elimination rows and columns ascend; "
                    "back substitution uses descending row order."
                ),
                "pivotTieBreak": "largest absolute pivot, then lowest row index",
                "operationSteps": (
                    "For pivot p, swap the chosen row with p. For each row r>p "
                    "ascending, factor=A[r][p]/A[p][p], set A[r][p] to exact "
                    "+0.0, update A[r][c]=A[r][c]-factor*A[p][c] for c>p "
                    "ascending, then b[r]=b[r]-factor*b[p]. Back-substitute rows "
                    "descending, subtracting A[r][c]*beta[c] for c>r ascending, "
                    "then divide by A[r][r]."
                ),
                "rankLimit": (
                    "rankRelativeTolerance * max(1.0, maximum absolute entry of "
                    "the original reduced Gram matrix for that active state)"
                ),
                "rankRelativeTolerance": RANK_RELATIVE_TOLERANCE,
                "singularRule": (
                    "A nonfinite pivot or abs(pivot) <= rankLimit is rank deficient "
                    "and invalid for that active state."
                ),
            },
            "objective": {
                "prediction": (
                    "Initialize prediction=offset, then for each amplitude-bearing "
                    "design column ascending set prediction=prediction+amplitude*basis."
                ),
                "residualSign": "observed residual value - prediction",
                "wrssAccumulation": (
                    "Initialize +0.0 and add weight * residual * residual once per "
                    "sample in source order; every intermediate must be finite."
                ),
            },
            "comparisonTolerances": {
                "objectiveRelativeTolerance": OBJECTIVE_RELATIVE_TOLERANCE,
                "comparisonLimit": (
                    "objectiveRelativeTolerance * max(1.0, abs(left), abs(right))"
                ),
                "constraintTolerance": CONSTRAINT_TOLERANCE,
                "rankRelativeTolerance": RANK_RELATIVE_TOLERANCE,
                "timingComparisonTolerance": TIMING_COMPARISON_TOLERANCE,
            },
            "decisionThresholdComparison": (
                "Threshold gates use exact binary64 >= comparisons with no objective "
                "tie tolerance; timing consistency uses its declared tolerance."
            ),
            "candidateWinnerOrdering": [
                "finite WRSS ascending within relative tolerance",
                "finite BIC ascending within relative tolerance",
                "finite AICc ascending within relative tolerance; null after finite",
                "global candidate index ascending",
            ],
            "modelWinnerOrdering": (
                "Apply the same WRSS, BIC, and AICc ordering, then declared "
                "modelClassOrder; null AICc sorts after finite and null equals null."
            ),
        },
        "decisionRules": {
            "preferOrderedDoubletOverPositivePulse": {
                "allPerSeriesDeltaWRSSAtLeast": (
                    ORDERED_OVER_POSITIVE_MINIMUM_PER_SERIES_DELTA_WRSS
                ),
                "globalDeltaBICAtLeast": ORDERED_OVER_POSITIVE_MINIMUM_DELTA_BIC,
                "globalDeltaWRSSAtLeast": (
                    ORDERED_OVER_POSITIVE_MINIMUM_DELTA_WRSS
                ),
                "signAndSharedTimingRequirementsMustPass": True,
            },
            "rejectOrderedDoubletForIndependentPulses": {
                "globalDeltaBICAtLeast": (
                    INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_BIC
                ),
                "globalDeltaWRSSAtLeast": (
                    INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_WRSS
                ),
                "independentCenterDispersionMustExceedTolerance": True,
                "signRequirementsMustPass": True,
            },
            "tieBreak": (
                "Use deterministicExecution.candidateWinnerOrdering and its "
                "declared relative tolerance. Model ties use modelClassOrder."
            ),
        },
        "effectiveWidthBounds": {
            "maximum": effective_width_maximum,
            "minimum": effective_width_minimum,
            "preparedWindowMaximum": prepared_width_limit,
            "previousBoundaryWidths": {
                "confirmedPositive": window.positive.effective_width,
                "precedingNegative": window.negative.effective_width,
            },
        },
        "finiteValueRules": (
            "All consumed inputs and all basis, solve, prediction, objective, "
            "penalty, and aggregate intermediates must be finite. Negative "
            "weights are invalid; zero weights follow deterministicExecution."
        ),
        "identityIsolationStatement": (
            "Only generic series identifiers and verified identity-free "
            "numerical residual evidence are admitted."
        ),
        "interpretationLimits": {
            "boundaryWinnerRule": (
                "A winner on any width boundary remains morphology-width "
                "unresolved and cannot be reported as a measured duration."
            ),
            "discoveryClaim": False,
            "planetaryInterpretationResolved": False,
            "statement": (
                "These are generic residual morphology models, not a physical "
                "binary-lens model; no model outcome alone is a planetary "
                "interpretation or discovery."
            ),
        },
        "invalidCandidateBehavior": {
            "reasonPriority": [
                "NONFINITE_INPUT",
                "NEGATIVE_WEIGHT",
                "INVALID_GEOMETRY",
                "INSUFFICIENT_POSITIVE_WEIGHT_SUPPORT",
                "RANK_DEFICIENT",
                "NONFINITE_FIT",
                "NO_FEASIBLE_ACTIVE_SET",
                "NONFINITE_OBJECTIVE",
            ],
            "resultRule": (
                "Emit status INVALID and the first applicable reason in priority "
                "order; retain model class, independent series ID when applicable, "
                "and global and local indices; emit fitted parameters, WRSS, BIC, "
                "and AICc as null; the result cannot win or aggregate."
            ),
        },
        "independentAggregation": {
            "acceptedWinnerIdentity": (
                "Canonical ordered vector of one accepted per-series global "
                "candidate index; it is not a cross-series Cartesian candidate."
            ),
            "requiredInputs": (
                "Exactly one finite accepted independent winner for every admitted "
                "series in canonical order; otherwise the aggregate is invalid."
            ),
            "WRSS": "sum per-series accepted WRSS in canonical series order",
            "N": "sum per-series positive-weight N in canonical series order",
            "parameterCount": 9 * series_count,
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
            "decisionRule": (
                "Compare this aggregate with the ordered-doublet result only by "
                "the predeclared rejectOrderedDoubletForIndependentPulses rule."
            ),
        },
        "modelClassOrder": list(MODEL_CLASS_IDS),
        "modelClasses": {
            INDEPENDENT_PULSES: {
                "amplitudeConstraints": ["negative", "positive"],
                "centersSharedAcrossSeries": False,
                "description": (
                    "Each series receives a separate six-axis negative-positive "
                    "search. No nonlinear parameter and no candidate Cartesian "
                    "product is shared between different series."
                ),
                "independentNonlinearParametersPerSeries": [
                    "negativeCenter",
                    "positiveCenter",
                    "negativeLogScale",
                    "negativeLogShape",
                    "positiveLogScale",
                    "positiveLogShape",
                ],
                "linearNuisancePerSeries": [
                    "offset",
                    "negativeAmplitude",
                    "positiveAmplitude",
                ],
                "strictTemporalOrderingPerSeries": True,
            },
            ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
                "amplitudeConstraints": ["negative", "positive"],
                "description": (
                    "A negative component is followed by a positive component "
                    "with shared nonlinear timing and shape across series."
                ),
                "linearNuisancePerSeries": [
                    "offset",
                    "negativeAmplitude",
                    "positiveAmplitude",
                ],
                "sharedNonlinearParameters": [
                    "negativeCenter",
                    "separation",
                    "negativeLogScale",
                    "negativeLogShape",
                    "positiveLogScale",
                    "positiveLogShape",
                ],
                "strictTemporalOrdering": True,
            },
            POSITIVE_PULSE_ONLY: {
                "amplitudeConstraint": "positive",
                "description": (
                    "One positive localized component with shared nonlinear "
                    "timing and shape across series."
                ),
                "linearNuisancePerSeries": ["offset", "amplitude"],
                "sharedNonlinearParameters": ["center", "logScale", "logShape"],
            },
        },
        "parameterAxes": axes,
        "preparedCoordinateBounds": {
            "maximum": window.maximum,
            "minimum": window.minimum,
        },
        "separationRules": {
            "maximum": axes["SEPARATION"]["start"]
            + (axes["SEPARATION"]["count"] - 1) * axes["SEPARATION"]["step"],
            "minimum": axes["SEPARATION"]["start"],
            "strictNegativeBeforePositive": True,
        },
    }


def _component_provenance(
    component: Mapping[str, Any], geometry: _ComponentGeometry
) -> dict[str, Any]:
    return {
        "boundaryAxes": list(geometry.boundary_axes),
        "discoveryDeltaWRSS": _finite_number(
            component.get("discoveryDeltaWRSS"), "discovery delta WRSS"
        ),
        "discoveryGenericSeriesID": geometry.generic_series_id,
        "effectiveWidth": geometry.effective_width,
        "frozenCenter": geometry.center,
        "frozenLogScale": geometry.log_scale,
        "frozenLogShape": geometry.log_shape,
        "widthInterpretationLimitedByBoundary": component.get(
            "widthInterpretationLimitedByBoundary"
        ),
    }


def _prepare_result(
    *,
    contract_sha256: str,
    cross: _VerifiedCrossValidation,
    window: _PreparedWindow,
    positive_component: Mapping[str, Any],
    negative_component: Mapping[str, Any],
    dataset_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_samples = sum(record["sampleCount"] for record in dataset_records)
    parent_hashes = dict(cross.result["parentHashes"])
    parent_hashes.update(
        {
            "crossValidationContractFileSHA256": cross.contract_file_sha256,
            "crossValidationResultFileSHA256": cross.result_file_sha256,
        }
    )
    parent_ids = dict(cross.result["parentIDs"])
    parent_ids["crossValidationContractID"] = CROSS_VALIDATION_CONTRACT_ID
    return {
        "admittedGenericSeriesIDs": [
            dataset["genericSeriesID"] for dataset in window.datasets
        ],
        "confirmedComponentProvenance": {
            "heldOutPassingSeriesIDs": [
                validation["validationGenericSeriesID"]
                for validation in positive_component["heldOutValidations"]
                if validation["heldOutValidationGatePassed"] is True
            ],
            "positive": _component_provenance(
                positive_component, window.positive
            ),
            "precedingNegative": _component_provenance(
                negative_component, window.negative
            ),
        },
        "discoveryClaim": False,
        "modelClassIDs": list(MODEL_CLASS_IDS),
        "morphologyContractID": MORPHOLOGY_CONTRACT_ID,
        "morphologyContractSHA256": contract_sha256,
        "morphologyContractVersion": MORPHOLOGY_CONTRACT_VERSION,
        "parentHashes": parent_hashes,
        "parentIDs": parent_ids,
        "planetaryInterpretationResolved": False,
        "preparedCoordinateBounds": {
            "anomalyCoreMaximum": window.core_maximum,
            "anomalyCoreMinimum": window.core_minimum,
            "maximum": window.maximum,
            "minimum": window.minimum,
        },
        "preparedDatasets": list(dataset_records),
        "recommendedNextTest": NEXT_TEST,
        "resultSchemaID": MORPHOLOGY_PREPARATION_SCHEMA_ID,
        "resultVersion": MORPHOLOGY_PREPARATION_VERSION,
        "sampleCounts": {
            "perSeries": {
                item["genericSeriesID"]: item["sampleCount"]
                for item in dataset_records
            },
            "total": total_samples,
        },
        "widthInterpretationResolved": False,
    }


def _artifact_manifest(
    *,
    contract_sha256: str,
    contract_file_sha256: str,
    preparation_file_sha256: str,
    preparation: Mapping[str, Any],
    dataset_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "artifactManifestSchemaID": ARTIFACT_MANIFEST_SCHEMA_ID,
        "artifactManifestVersion": ARTIFACT_MANIFEST_VERSION,
        "identityIsolationStatement": (
            "Artifacts contain generic identifiers and identity-free numerical "
            "evidence only."
        ),
        "modelScopeStatement": (
            "This prepares generic morphology evidence and a predeclared model "
            "contract; it performs no fit, planetary interpretation, or discovery."
        ),
        "morphologyContractFileSHA256": contract_file_sha256,
        "morphologyContractID": MORPHOLOGY_CONTRACT_ID,
        "morphologyContractSHA256": contract_sha256,
        "morphologyPreparationFileSHA256": preparation_file_sha256,
        "orderedDatasetFiles": [item["outputFile"] for item in dataset_records],
        "orderedGenericSeriesIDs": preparation["admittedGenericSeriesIDs"],
        "outputSHA256s": {
            item["outputFile"]: item["outputSHA256"] for item in dataset_records
        },
        "parentHashes": preparation["parentHashes"],
        "parentIDs": preparation["parentIDs"],
        "totalSampleCount": preparation["sampleCounts"]["total"],
    }


def _prepare_anomaly_morphology_impl(
    residual_root: str | Path,
    *,
    residual_grid_root: str | Path,
    residual_grid_investigation_record: str | Path,
    cross_validation_root: str | Path,
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
            "residual-grid investigation record",
        ),
        (
            Path(cross_validation_root).expanduser().absolute(),
            "residual cross-validation root",
        ),
    )
    _reject_symlink_components(output.parent, "output root")
    for path, description in paths:
        _reject_symlink_components(path, description)
    residual_path, grid_path, investigation_path, cross_path = (
        item[0] for item in paths
    )

    residuals = _verify_residual_root(residual_path)
    grid = _verify_grid_root(grid_path, residuals)
    investigation = _verify_investigation(
        investigation_path,
        grid,
        grid_path / GRID_PROJECT_RELATIVE_PATH,
    )
    cross = _verify_cross_validation_root(
        cross_path,
        residuals=residuals,
        grid=grid,
        investigation=investigation,
    )
    positive_component, selected_ids, positive = _confirmed_evidence(
        cross.result, grid.generic_series_ids
    )
    negative_component, negative = _preceding_negative_component(
        cross.result, positive, grid.generic_series_ids
    )
    if negative.generic_series_id not in selected_ids:
        raise _fail(
            "preceding negative discovery series is not a passed morphology series"
        )
    residual_by_id = {
        item["genericSeriesID"]: item for item in residuals.series
    }
    if any(series_id not in residual_by_id for series_id in selected_ids):
        raise _fail("selected morphology series is absent from residual preparation")
    source_sha256s = {
        record["genericSeriesID"]: residuals.series_file_sha256s[
            record["outputFile"]
        ]
        for record in residuals.manifest["series"]
    }
    window = _prepare_window(
        [residual_by_id[series_id] for series_id in selected_ids],
        source_sha256s=source_sha256s,
        negative=negative,
        positive=positive,
    )
    contract = _morphology_contract(window, grid, selected_ids)
    contract_bytes = _stable_json_bytes(contract)
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    _identity_check((contract,))

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
        dataset_records: list[dict[str, Any]] = []
        dataset_documents: list[Mapping[str, Any]] = []
        for ordinal, dataset in enumerate(window.datasets, start=1):
            relative_path = (
                f"{DATASET_DIRECTORY}/morphology-series-{ordinal:03d}.json"
            )
            dataset_document = dict(dataset)
            dataset_document.update(
                {
                    "morphologyContractID": MORPHOLOGY_CONTRACT_ID,
                    "morphologyContractSHA256": contract_sha256,
                    "morphologyContractVersion": MORPHOLOGY_CONTRACT_VERSION,
                }
            )
            dataset_bytes = _stable_json_bytes(dataset_document)
            _identity_check((dataset_document,))
            _atomic_write_bytes(staging / relative_path, dataset_bytes)
            dataset_documents.append(dataset_document)
            dataset_records.append(
                {
                    "genericSeriesID": dataset["genericSeriesID"],
                    "outputFile": relative_path,
                    "outputSHA256": _sha256_bytes(dataset_bytes),
                    "sampleCount": dataset["sampleCount"],
                    "sourceResidualSeriesSHA256": dataset[
                        "sourceResidualSeriesSHA256"
                    ],
                }
            )
        preparation = _prepare_result(
            contract_sha256=contract_sha256,
            cross=cross,
            window=window,
            positive_component=positive_component,
            negative_component=negative_component,
            dataset_records=dataset_records,
        )
        _identity_check((preparation,))
        preparation_bytes = _stable_json_bytes(preparation)
        _atomic_write_bytes(staging / PREPARATION_RELATIVE_PATH, preparation_bytes)
        manifest = _artifact_manifest(
            contract_sha256=contract_sha256,
            contract_file_sha256=_sha256_bytes(contract_bytes),
            preparation_file_sha256=_sha256_bytes(preparation_bytes),
            preparation=preparation,
            dataset_records=dataset_records,
        )
        _identity_check((manifest,))
        manifest_bytes = _stable_json_bytes(manifest)
        _atomic_write_bytes(staging / MANIFEST_RELATIVE_PATH, manifest_bytes)
        if output.exists() or output.is_symlink():
            raise _fail("output root already exists")
        staging.rename(output)
    except Exception as error:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if isinstance(error, AnomalyMorphologyPreparationError):
            raise
        raise _fail("atomic anomaly-morphology publication failed") from error
    return {
        "contract": contract,
        "datasets": dataset_documents,
        "manifest": manifest,
        "preparation": preparation,
    }


def prepare_anomaly_morphology(
    residual_root: str | Path,
    *,
    residual_grid_root: str | Path,
    residual_grid_investigation_record: str | Path,
    cross_validation_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Prepare deterministic blind evidence without evaluating morphology fits."""

    try:
        return _prepare_anomaly_morphology_impl(
            residual_root,
            residual_grid_root=residual_grid_root,
            residual_grid_investigation_record=residual_grid_investigation_record,
            cross_validation_root=cross_validation_root,
            output_root=output_root,
        )
    except AnomalyMorphologyPreparationError:
        raise
    except (
        CoarseGridBuildError,
        RefinementGridBuildError,
        ResidualGridBuildError,
        ResidualGridValidationError,
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
            "Prepare verified blind residual evidence and a predeclared "
            "generic anomaly-morphology contract."
        )
    )
    parser.add_argument("--residual-root", required=True, type=Path)
    parser.add_argument("--residual-grid-root", required=True, type=Path)
    parser.add_argument(
        "--residual-grid-investigation-record", required=True, type=Path
    )
    parser.add_argument("--cross-validation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    published = prepare_anomaly_morphology(
        arguments.residual_root,
        residual_grid_root=arguments.residual_grid_root,
        residual_grid_investigation_record=(
            arguments.residual_grid_investigation_record
        ),
        cross_validation_root=arguments.cross_validation_root,
        output_root=arguments.output_root,
    )
    preparation = published["preparation"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind anomaly-morphology preparation ready")
    print(f"series: {len(preparation['admittedGenericSeriesIDs'])}")
    print(f"samples: {preparation['sampleCounts']['total']}")
    print(f"result: {output / PREPARATION_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
