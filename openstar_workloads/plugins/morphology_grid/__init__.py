"""Strict, generic residual-morphology grid workload."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from openstar_workloads.contract import (
    DatasetReduction,
    ResultValidation,
    WorkloadDefinition,
)


WORKLOAD_ID = "openstar.morphology-grid.v1"
DATASET_SCHEMA_ID = "openstar.dataset.morphology-grid.v1"
PAYLOAD_SCHEMA_ID = "openstar.payload.morphology-grid-shard.v1"
RESULT_SCHEMA_ID = "openstar.result.morphology-grid-shard.v1"
MORPHOLOGY_FAMILY_ID = "openstar.microlensing-residual-morphology.v1"
COMPONENT_TEMPLATE_FAMILY_ID = (
    "openstar.curve-family.symmetric-radial-amplification.v1"
)
EXECUTION_CONTRACT_ID = "openstar.morphology-grid-execution.v1"
EXECUTION_CONTRACT_VERSION = "1.0"

POSITIVE_PULSE_ONLY = "POSITIVE_PULSE_ONLY"
ORDERED_NEGATIVE_POSITIVE_DOUBLET = "ORDERED_NEGATIVE_POSITIVE_DOUBLET"
INDEPENDENT_PULSES = "INDEPENDENT_PULSES"
MODEL_CLASS_IDS = (
    POSITIVE_PULSE_ONLY,
    ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    INDEPENDENT_PULSES,
)

MAX_SAFE_INTEGER = (1 << 53) - 1
RANK_RELATIVE_TOLERANCE = 1.0e-12
RESULT_RELATIVE_TOLERANCE = 1.0e-9

_LINEAR_AXIS_FIELDS = frozenset(("start", "step", "count"))
_EXPLICIT_AXIS_FIELDS = frozenset(("values",))
_SERIES_FIELDS = frozenset(
    ("genericSeriesID", "coordinates", "values", "inverseVariances")
)
_DATASET_FIELDS = frozenset(
    (
        "id",
        "datasetSchemaID",
        "morphologyFamilyID",
        "componentTemplateFamilyID",
        "modelClassID",
        "series",
        "morphologyGrid",
        "candidatesPerWorkUnit",
        "executionContractID",
        "executionContractVersion",
    )
)
_POSITIVE_GRID_FIELDS = frozenset(
    ("centerAxis", "logScaleAxis", "logShapeAxis")
)
_ORDERED_GRID_FIELDS = frozenset(
    (
        "negativeCenterAxis",
        "separationAxis",
        "negativeLogScaleAxis",
        "negativeLogShapeAxis",
        "positiveLogScaleAxis",
        "positiveLogShapeAxis",
    )
)
_INDEPENDENT_GRID_FIELDS = frozenset(
    (
        "centerAxis",
        "negativeLogScaleAxis",
        "negativeLogShapeAxis",
        "positiveLogScaleAxis",
        "positiveLogShapeAxis",
    )
)
_WORK_PAYLOAD_FIELDS = frozenset(
    ("morphologyFamilyID", "modelClassID", "gridStartIndex", "gridCount")
)
_RESULT_PAYLOAD_FIELDS = frozenset(
    (
        "morphologyFamilyID",
        "modelClassID",
        "gridStartIndex",
        "gridCount",
        "bestCandidate",
        "evaluatedCandidateCount",
        "invalidCandidateCount",
    )
)
_BEST_CANDIDATE_FIELDS = frozenset(
    (
        "gridIndex",
        "parameters",
        "seriesFits",
        "positiveWeightSampleCount",
        "weightedResidualSumSquares",
        "nominalParameterCount",
        "bayesianInformationCriterion",
        "correctedAkaikeInformationCriterion",
        "correctedAkaikeInformationCriterionDefined",
    )
)
_POSITIVE_PARAMETER_FIELDS = frozenset(("center", "logScale", "logShape"))
_DOUBLET_PARAMETER_FIELDS = frozenset(
    (
        "negativeCenter",
        "separation",
        "negativeLogScale",
        "negativeLogShape",
        "positiveLogScale",
        "positiveLogShape",
    )
)
_INDEPENDENT_PARAMETER_FIELDS = frozenset(
    (
        "negativeCenter",
        "positiveCenter",
        "negativeLogScale",
        "negativeLogShape",
        "positiveLogScale",
        "positiveLogShape",
    )
)
_POSITIVE_FIT_FIELDS = frozenset(
    (
        "genericSeriesID",
        "positiveWeightSampleCount",
        "offset",
        "positiveAmplitude",
        "positiveAmplitudeSign",
        "weightedResidualSumSquares",
    )
)
_DOUBLET_FIT_FIELDS = frozenset(
    (
        "genericSeriesID",
        "positiveWeightSampleCount",
        "offset",
        "negativeAmplitude",
        "negativeAmplitudeSign",
        "positiveAmplitude",
        "positiveAmplitudeSign",
        "weightedResidualSumSquares",
    )
)
_FLATTENABLE_RESULT_FIELDS = frozenset(
    (
        *_RESULT_PAYLOAD_FIELDS,
        *_BEST_CANDIDATE_FIELDS,
        *_POSITIVE_PARAMETER_FIELDS,
        *_DOUBLET_PARAMETER_FIELDS,
        *_INDEPENDENT_PARAMETER_FIELDS,
        *_POSITIVE_FIT_FIELDS,
        *_DOUBLET_FIT_FIELDS,
    )
)


@dataclass(frozen=True, slots=True)
class _Axis:
    start: float | None
    step: float | None
    count: int
    explicit_values: tuple[float, ...] | None

    def value(self, index: int) -> float:
        if type(index) is not int or index < 0 or index >= self.count:
            raise ValueError("axis index is outside the configured axis")
        if self.explicit_values is not None:
            return self.explicit_values[index]
        if self.start is None or self.step is None:
            raise ValueError("axis representation is invalid")
        value = self.start + index * self.step
        if not math.isfinite(value):
            raise ValueError("axis value is nonfinite")
        return value


@dataclass(frozen=True, slots=True)
class _Series:
    generic_series_id: str
    coordinates: tuple[float, ...]
    values: tuple[float, ...]
    inverse_variances: tuple[float, ...]
    positive_weight_count: int


@dataclass(frozen=True, slots=True)
class _Grid:
    model_class_id: str
    axes: tuple[_Axis, ...]
    candidates_per_work_unit: int
    total_candidates: int


@dataclass(frozen=True, slots=True)
class _ValidatedDataset:
    series: tuple[_Series, ...]
    grid: _Grid


@dataclass(frozen=True, slots=True)
class _SeriesFit:
    generic_series_id: str
    positive_weight_sample_count: int
    offset: float
    negative_amplitude: float | None
    positive_amplitude: float
    weighted_residual_sum_squares: float


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    grid_index: int
    parameters: Mapping[str, float]
    series_fits: tuple[_SeriesFit, ...]
    positive_weight_sample_count: int
    weighted_residual_sum_squares: float
    nominal_parameter_count: int
    bayesian_information_criterion: float
    corrected_akaike_information_criterion: float | None
    corrected_akaike_information_criterion_defined: bool


def _runtime_error(message: str) -> RuntimeError:
    return RuntimeError(f"morphology-grid dataset: {message}")


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _runtime_error(f"{field_name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _runtime_error(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise _runtime_error(f"{field_name} must be finite")
    return number


def _positive_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise _runtime_error(f"{field_name} must be a positive integer")
    if value > MAX_SAFE_INTEGER:
        raise _runtime_error(f"{field_name} exceeds the safe integer range")
    return value


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    if value > MAX_SAFE_INTEGER:
        raise ValueError(f"{field_name} exceeds the safe integer range")
    return value


def _safe_product(field_name: str, *values: int) -> int:
    product = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise _runtime_error(f"{field_name} contains an invalid count")
        if value and product > MAX_SAFE_INTEGER // value:
            raise _runtime_error(f"{field_name} exceeds the safe integer range")
        product *= value
    return product


def _safe_pair_count(center_count: int) -> int:
    if type(center_count) is not int or center_count < 2:
        raise ValueError("center count must be an integer of at least two")
    if center_count > MAX_SAFE_INTEGER:
        raise ValueError("center count exceeds the safe integer range")
    if center_count > MAX_SAFE_INTEGER // (center_count - 1):
        raise ValueError("center pair count exceeds the safe integer range")
    pair_count = center_count * (center_count - 1) // 2
    if pair_count > MAX_SAFE_INTEGER:
        raise ValueError("center pair count exceeds the safe integer range")
    return pair_count


def independent_center_pair_index(
    center_count: int,
    negative_center_index: int,
    positive_center_index: int,
) -> int:
    """Return the lexicographic strict ordered-pair index."""

    _safe_pair_count(center_count)
    if (
        type(negative_center_index) is not int
        or type(positive_center_index) is not int
        or negative_center_index < 0
        or positive_center_index >= center_count
        or negative_center_index >= positive_center_index
    ):
        raise ValueError("center indices must satisfy 0 <= negative < positive < count")
    return (
        negative_center_index
        * (2 * center_count - negative_center_index - 1)
        // 2
        + positive_center_index
        - negative_center_index
        - 1
    )


def independent_center_pair_indices(
    center_count: int,
    pair_index: int,
) -> tuple[int, int]:
    """Invert the lexicographic strict ordered-pair index."""

    pair_count = _safe_pair_count(center_count)
    if type(pair_index) is not int or pair_index < 0 or pair_index >= pair_count:
        raise ValueError("center pair index is outside the configured axis")
    low = 0
    high = center_count - 2
    while low <= high:
        middle = (low + high) // 2
        row_start = middle * (2 * center_count - middle - 1) // 2
        next_start = (
            (middle + 1) * (2 * center_count - middle - 2) // 2
            if middle + 1 < center_count - 1
            else pair_count
        )
        if pair_index < row_start:
            high = middle - 1
        elif pair_index >= next_start:
            low = middle + 1
        else:
            positive = middle + 1 + pair_index - row_start
            return middle, positive
    raise ValueError("center pair index could not be inverted")


def mixed_radix_index(indices: Sequence[int], counts: Sequence[int]) -> int:
    """Flatten rightmost-fastest indices using JSON-safe integer checks."""

    if len(indices) != len(counts) or not counts:
        raise ValueError("indices and counts must have equal nonzero length")
    total = 1
    for count in counts:
        if type(count) is not int or count <= 0 or count > MAX_SAFE_INTEGER:
            raise ValueError("axis count is invalid")
        if total > MAX_SAFE_INTEGER // count:
            raise ValueError("mixed-radix count exceeds the safe integer range")
        total *= count
    result = 0
    for index, count in zip(indices, counts):
        if type(index) is not int or index < 0 or index >= count:
            raise ValueError("candidate index is outside its axis")
        if result > (MAX_SAFE_INTEGER - index) // count:
            raise ValueError("mixed-radix index exceeds the safe integer range")
        result = result * count + index
    return result


def mixed_radix_indices(index: int, counts: Sequence[int]) -> tuple[int, ...]:
    """Invert a rightmost-fastest JSON-safe mixed-radix index."""

    if not counts:
        raise ValueError("counts must be nonempty")
    total = 1
    for count in counts:
        if type(count) is not int or count <= 0 or count > MAX_SAFE_INTEGER:
            raise ValueError("axis count is invalid")
        if total > MAX_SAFE_INTEGER // count:
            raise ValueError("mixed-radix count exceeds the safe integer range")
        total *= count
    if type(index) is not int or index < 0 or index >= total:
        raise ValueError("mixed-radix index is outside the configured grid")
    remaining = index
    reversed_indices = []
    for count in reversed(counts):
        remaining, axis_index = divmod(remaining, count)
        reversed_indices.append(axis_index)
    return tuple(reversed(reversed_indices))


def candidate_index(indices: Sequence[int], counts: Sequence[int]) -> int:
    """Public forward candidate mapping for Python/Swift golden vectors."""

    return mixed_radix_index(indices, counts)


def candidate_indices(index: int, counts: Sequence[int]) -> tuple[int, ...]:
    """Public inverse candidate mapping for Python/Swift golden vectors."""

    return mixed_radix_indices(index, counts)


def _axis(
    value: Any,
    field_name: str,
    *,
    exponentiated: bool,
    allow_explicit: bool = False,
    strictly_positive: bool = False,
) -> _Axis:
    if not isinstance(value, Mapping):
        raise _runtime_error(f"{field_name} must be a mapping")
    if set(value) == _LINEAR_AXIS_FIELDS:
        start = _finite_number(value["start"], f"{field_name}.start")
        step = _finite_number(value["step"], f"{field_name}.step")
        count = _positive_integer(value["count"], f"{field_name}.count")
        if step <= 0.0:
            raise _runtime_error(f"{field_name}.step must be positive")
        last = start + (count - 1) * step
        if not math.isfinite(last):
            raise _runtime_error(f"{field_name} has a nonfinite endpoint")
        values = (start, last)
        explicit_values = None
    elif allow_explicit and set(value) == _EXPLICIT_AXIS_FIELDS:
        raw_values = value["values"]
        if not isinstance(raw_values, list) or not raw_values:
            raise _runtime_error(f"{field_name}.values must be a nonempty array")
        values = tuple(
            _finite_number(item, f"{field_name}.values[{index}]")
            for index, item in enumerate(raw_values)
        )
        if any(right <= left for left, right in zip(values, values[1:])):
            raise _runtime_error(f"{field_name}.values must be strictly increasing")
        start = None
        step = None
        count = len(values)
        explicit_values = values
    else:
        expected = "start, step, and count"
        if allow_explicit:
            expected += " or exactly values"
        raise _runtime_error(f"{field_name} must contain exactly {expected}")

    for item in values:
        if strictly_positive and item <= 0.0:
            raise _runtime_error(f"{field_name} values must be strictly positive")
        if exponentiated:
            try:
                expanded = math.exp(item)
            except OverflowError as error:
                raise _runtime_error(
                    f"{field_name} has an invalid exponentiated value"
                ) from error
            if not math.isfinite(expanded) or expanded <= 0.0:
                raise _runtime_error(
                    f"{field_name} has an invalid exponentiated value"
                )
    return _Axis(
        start=start,
        step=step,
        count=count,
        explicit_values=explicit_values,
    )


def _series(value: Any, index: int) -> _Series:
    field_name = f"series[{index}]"
    if not isinstance(value, Mapping) or set(value) != _SERIES_FIELDS:
        raise _runtime_error(f"{field_name} does not match the published field set")
    series_id = value["genericSeriesID"]
    if not isinstance(series_id, str) or not series_id.strip():
        raise _runtime_error(f"{field_name}.genericSeriesID must be nonempty")

    arrays = []
    for array_name in ("coordinates", "values", "inverseVariances"):
        raw = value[array_name]
        if not isinstance(raw, list) or not raw:
            raise _runtime_error(f"{field_name}.{array_name} must be nonempty")
        arrays.append(
            tuple(
                _finite_number(item, f"{field_name}.{array_name}[{item_index}]")
                for item_index, item in enumerate(raw)
            )
        )
    coordinates, values, inverse_variances = arrays
    if not (len(coordinates) == len(values) == len(inverse_variances)):
        raise _runtime_error(f"{field_name} arrays must have equal length")
    if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
        raise _runtime_error(f"{field_name}.coordinates must be strictly increasing")
    if any(weight < 0.0 for weight in inverse_variances):
        raise _runtime_error(f"{field_name}.inverseVariances must be nonnegative")
    return _Series(
        generic_series_id=series_id,
        coordinates=coordinates,
        values=values,
        inverse_variances=inverse_variances,
        positive_weight_count=sum(weight > 0.0 for weight in inverse_variances),
    )


def _grid(dataset: Mapping[str, Any]) -> _Grid:
    model_class_id = dataset.get("modelClassID")
    grid = dataset.get("morphologyGrid")
    if not isinstance(grid, Mapping):
        raise _runtime_error("morphologyGrid must be a mapping")

    if model_class_id == POSITIVE_PULSE_ONLY:
        if set(grid) != _POSITIVE_GRID_FIELDS:
            raise _runtime_error("positive morphologyGrid field set is invalid")
        axes = (
            _axis(grid["centerAxis"], "morphologyGrid.centerAxis", exponentiated=False),
            _axis(
                grid["logScaleAxis"],
                "morphologyGrid.logScaleAxis",
                exponentiated=True,
            ),
            _axis(
                grid["logShapeAxis"],
                "morphologyGrid.logShapeAxis",
                exponentiated=True,
                allow_explicit=True,
            ),
        )
        counts = tuple(axis.count for axis in axes)
    elif model_class_id == ORDERED_NEGATIVE_POSITIVE_DOUBLET:
        if set(grid) != _ORDERED_GRID_FIELDS:
            raise _runtime_error("ordered morphologyGrid field set is invalid")
        axes = (
            _axis(
                grid["negativeCenterAxis"],
                "morphologyGrid.negativeCenterAxis",
                exponentiated=False,
            ),
            _axis(
                grid["separationAxis"],
                "morphologyGrid.separationAxis",
                exponentiated=False,
                strictly_positive=True,
            ),
            _axis(
                grid["negativeLogScaleAxis"],
                "morphologyGrid.negativeLogScaleAxis",
                exponentiated=True,
            ),
            _axis(
                grid["negativeLogShapeAxis"],
                "morphologyGrid.negativeLogShapeAxis",
                exponentiated=True,
                allow_explicit=True,
            ),
            _axis(
                grid["positiveLogScaleAxis"],
                "morphologyGrid.positiveLogScaleAxis",
                exponentiated=True,
            ),
            _axis(
                grid["positiveLogShapeAxis"],
                "morphologyGrid.positiveLogShapeAxis",
                exponentiated=True,
                allow_explicit=True,
            ),
        )
        counts = tuple(axis.count for axis in axes)
    elif model_class_id == INDEPENDENT_PULSES:
        if set(grid) != _INDEPENDENT_GRID_FIELDS:
            raise _runtime_error("independent morphologyGrid field set is invalid")
        center = _axis(
            grid["centerAxis"],
            "morphologyGrid.centerAxis",
            exponentiated=False,
        )
        try:
            pair_count = _safe_pair_count(center.count)
        except ValueError as error:
            raise _runtime_error(str(error)) from error
        remaining = (
            _axis(
                grid["negativeLogScaleAxis"],
                "morphologyGrid.negativeLogScaleAxis",
                exponentiated=True,
            ),
            _axis(
                grid["negativeLogShapeAxis"],
                "morphologyGrid.negativeLogShapeAxis",
                exponentiated=True,
                allow_explicit=True,
            ),
            _axis(
                grid["positiveLogScaleAxis"],
                "morphologyGrid.positiveLogScaleAxis",
                exponentiated=True,
            ),
            _axis(
                grid["positiveLogShapeAxis"],
                "morphologyGrid.positiveLogShapeAxis",
                exponentiated=True,
                allow_explicit=True,
            ),
        )
        axes = (center, *remaining)
        counts = (pair_count, *(axis.count for axis in remaining))
    else:
        raise _runtime_error("modelClassID is invalid")

    candidates_per_work_unit = _positive_integer(
        dataset.get("candidatesPerWorkUnit"),
        "candidatesPerWorkUnit",
    )
    total_candidates = _safe_product("total candidate count", *counts)
    return _Grid(
        model_class_id=model_class_id,
        axes=axes,
        candidates_per_work_unit=candidates_per_work_unit,
        total_candidates=total_candidates,
    )


def _validated_dataset(dataset: Mapping[str, Any]) -> _ValidatedDataset:
    if not isinstance(dataset, Mapping):
        raise _runtime_error("dataset must be a mapping")
    missing = _DATASET_FIELDS.difference(dataset)
    if missing:
        raise _runtime_error("missing required fields: " + ", ".join(sorted(missing)))
    if not isinstance(dataset["id"], str) or not dataset["id"].strip():
        raise _runtime_error("id must be a nonempty string")
    identities = (
        ("datasetSchemaID", DATASET_SCHEMA_ID),
        ("morphologyFamilyID", MORPHOLOGY_FAMILY_ID),
        ("componentTemplateFamilyID", COMPONENT_TEMPLATE_FAMILY_ID),
        ("executionContractID", EXECUTION_CONTRACT_ID),
        ("executionContractVersion", EXECUTION_CONTRACT_VERSION),
    )
    for field_name, expected in identities:
        if dataset[field_name] != expected:
            raise _runtime_error(f"{field_name} is invalid")

    raw_series = dataset["series"]
    if not isinstance(raw_series, list) or not raw_series:
        raise _runtime_error("series must be a nonempty array")
    series = tuple(_series(item, index) for index, item in enumerate(raw_series))
    series_ids = tuple(item.generic_series_id for item in series)
    if len(set(series_ids)) != len(series_ids):
        raise _runtime_error("generic series IDs must be unique")
    if series_ids != tuple(sorted(series_ids)):
        raise _runtime_error("series must be in canonical generic-series-ID order")

    grid = _grid(dataset)
    required_rank = 2 if grid.model_class_id == POSITIVE_PULSE_ONLY else 3
    if grid.model_class_id == INDEPENDENT_PULSES and len(series) != 1:
        raise _runtime_error("INDEPENDENT_PULSES requires exactly one series")
    for item in series:
        if item.positive_weight_count < required_rank:
            raise _runtime_error(
                f"{item.generic_series_id} lacks positive-weight rows for rank"
            )
    total_samples = sum(len(item.coordinates) for item in series)
    if total_samples > MAX_SAFE_INTEGER:
        raise _runtime_error("total sample count exceeds the safe integer range")
    _safe_product(
        "sample-candidate evaluation count",
        total_samples,
        grid.total_candidates,
    )
    return _ValidatedDataset(series=series, grid=grid)


def _candidate_counts(grid: _Grid) -> tuple[int, ...]:
    if grid.model_class_id == INDEPENDENT_PULSES:
        pair_count = _safe_pair_count(grid.axes[0].count)
        return (pair_count, *(axis.count for axis in grid.axes[1:]))
    return tuple(axis.count for axis in grid.axes)


def _candidate_indices(grid: _Grid, grid_index: int) -> tuple[int, ...]:
    return candidate_indices(grid_index, _candidate_counts(grid))


def _candidate_index(grid: _Grid, *indices: int) -> int:
    return candidate_index(indices, _candidate_counts(grid))


def _candidate_parameters(grid: _Grid, grid_index: int) -> Mapping[str, float]:
    indices = _candidate_indices(grid, grid_index)
    if grid.model_class_id == POSITIVE_PULSE_ONLY:
        return {
            "center": grid.axes[0].value(indices[0]),
            "logScale": grid.axes[1].value(indices[1]),
            "logShape": grid.axes[2].value(indices[2]),
        }
    if grid.model_class_id == ORDERED_NEGATIVE_POSITIVE_DOUBLET:
        negative_center = grid.axes[0].value(indices[0])
        separation = grid.axes[1].value(indices[1])
        positive_center = negative_center + separation
        if not math.isfinite(positive_center) or positive_center <= negative_center:
            raise ValueError("ordered candidate centers are invalid")
        return {
            "negativeCenter": negative_center,
            "separation": separation,
            "negativeLogScale": grid.axes[2].value(indices[2]),
            "negativeLogShape": grid.axes[3].value(indices[3]),
            "positiveLogScale": grid.axes[4].value(indices[4]),
            "positiveLogShape": grid.axes[5].value(indices[5]),
        }

    negative_index, positive_index = independent_center_pair_indices(
        grid.axes[0].count,
        indices[0],
    )
    return {
        "negativeCenter": grid.axes[0].value(negative_index),
        "positiveCenter": grid.axes[0].value(positive_index),
        "negativeLogScale": grid.axes[1].value(indices[1]),
        "negativeLogShape": grid.axes[2].value(indices[2]),
        "positiveLogScale": grid.axes[3].value(indices[3]),
        "positiveLogShape": grid.axes[4].value(indices[4]),
    }


def _finite_calculation(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _component_basis(
    coordinate: float,
    center: float,
    log_scale: float,
    log_shape: float,
) -> float | None:
    try:
        scale = math.exp(log_scale)
        shape = math.exp(log_shape)
        difference = coordinate - center
        z = difference / scale
        shape_squared = shape * shape
        z_squared = z * z
        u_squared = shape_squared + z_squared
        u = math.sqrt(u_squared)
        numerator = u_squared + 2.0
        rooted = math.sqrt(u_squared + 4.0)
        denominator = u * rooted
        basis = numerator / denominator
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    calculations = (
        scale,
        shape,
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
    if not _finite_calculation(*calculations) or scale <= 0.0 or shape <= 0.0:
        return None
    return basis


def _solve_normal_equations(
    matrix: Sequence[Sequence[float]],
    right_hand_side: Sequence[float],
) -> tuple[float, ...] | None:
    size = len(matrix)
    if size == 0 or len(right_hand_side) != size:
        return None
    working = [list(row) for row in matrix]
    right = list(right_hand_side)
    if any(len(row) != size for row in working):
        return None
    original_maximum = max(abs(value) for row in working for value in row)
    rank_limit = RANK_RELATIVE_TOLERANCE * max(1.0, original_maximum)
    if not _finite_calculation(original_maximum, rank_limit, *right):
        return None

    for pivot in range(size):
        pivot_row = pivot
        pivot_absolute = abs(working[pivot][pivot])
        for row in range(pivot + 1, size):
            candidate_absolute = abs(working[row][pivot])
            if candidate_absolute > pivot_absolute:
                pivot_row = row
                pivot_absolute = candidate_absolute
        if not math.isfinite(pivot_absolute) or pivot_absolute <= rank_limit:
            return None
        if pivot_row != pivot:
            working[pivot], working[pivot_row] = working[pivot_row], working[pivot]
            right[pivot], right[pivot_row] = right[pivot_row], right[pivot]
        for row in range(pivot + 1, size):
            factor = working[row][pivot] / working[pivot][pivot]
            working[row][pivot] = 0.0
            for column in range(pivot + 1, size):
                working[row][column] = (
                    working[row][column] - factor * working[pivot][column]
                )
            right[row] = right[row] - factor * right[pivot]
            if not _finite_calculation(*working[row], right[row]):
                return None

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        numerator = right[row]
        for column in range(row + 1, size):
            numerator = numerator - working[row][column] * solution[column]
        diagonal = working[row][row]
        if not _finite_calculation(numerator, diagonal) or abs(diagonal) <= rank_limit:
            return None
        solution[row] = numerator / diagonal
        if not math.isfinite(solution[row]):
            return None
    return tuple(solution)


def _objective_limit(left: float, right: float) -> float:
    return RESULT_RELATIVE_TOLERANCE * max(1.0, abs(left), abs(right))


def _amplitude_sign(value: float) -> str:
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "zero"


def _nominal_parameter_count(model_class_id: str, series_count: int) -> int:
    if model_class_id == POSITIVE_PULSE_ONLY:
        return 2 * series_count + 3
    if model_class_id == ORDERED_NEGATIVE_POSITIVE_DOUBLET:
        return 3 * series_count + 6
    if model_class_id == INDEPENDENT_PULSES and series_count == 1:
        return 9
    raise ValueError("model class and series count are inconsistent")


def _candidate_precedes(
    left: _CandidateEvaluation,
    right: _CandidateEvaluation,
) -> bool:
    comparisons = (
        (
            left.weighted_residual_sum_squares,
            right.weighted_residual_sum_squares,
        ),
        (
            left.bayesian_information_criterion,
            right.bayesian_information_criterion,
        ),
    )
    for left_value, right_value in comparisons:
        limit = _objective_limit(left_value, right_value)
        if left_value < right_value - limit:
            return True
        if right_value < left_value - limit:
            return False

    if (
        left.corrected_akaike_information_criterion_defined
        and right.corrected_akaike_information_criterion_defined
    ):
        left_aicc = left.corrected_akaike_information_criterion
        right_aicc = right.corrected_akaike_information_criterion
        if left_aicc is None or right_aicc is None:
            raise ValueError("defined AICc must be finite")
        limit = _objective_limit(left_aicc, right_aicc)
        if left_aicc < right_aicc - limit:
            return True
        if right_aicc < left_aicc - limit:
            return False
    elif (
        left.corrected_akaike_information_criterion_defined
        != right.corrected_akaike_information_criterion_defined
    ):
        return left.corrected_akaike_information_criterion_defined

    return left.grid_index < right.grid_index


def _fit_series(
    series: _Series,
    component_bases: Sequence[Sequence[float]],
    signs: Sequence[int],
) -> tuple[float, tuple[float, ...], float] | None:
    component_count = len(component_bases)
    if component_count != len(signs) or component_count not in (1, 2):
        return None
    if any(len(bases) != len(series.coordinates) for bases in component_bases):
        return None
    active_states = (
        ((True,), (False,))
        if component_count == 1
        else ((True, True), (False, True), (True, False), (False, False))
    )
    accepted: tuple[float, tuple[float, ...], float, int] | None = None
    for state_ordinal, state in enumerate(active_states):
        free_components = [index for index, free in enumerate(state) if free]
        column_count = 1 + len(free_components)
        if series.positive_weight_count < column_count:
            continue
        gram = [[0.0 for _ in range(column_count)] for _ in range(column_count)]
        right = [0.0 for _ in range(column_count)]
        valid = True
        for sample_index, (value, weight) in enumerate(
            zip(series.values, series.inverse_variances)
        ):
            if weight == 0.0:
                continue
            columns = [1.0]
            columns.extend(
                component_bases[component_index][sample_index]
                for component_index in free_components
            )
            for left in range(column_count):
                for right_index in range(left, column_count):
                    term = weight * columns[left] * columns[right_index]
                    gram[left][right_index] += term
                right[left] += weight * columns[left] * value
            if not _finite_calculation(
                *right,
                *(item for row in gram for item in row),
            ):
                valid = False
                break
        if not valid:
            continue
        for left in range(column_count):
            for right_index in range(left + 1, column_count):
                gram[right_index][left] = gram[left][right_index]
        solved = _solve_normal_equations(gram, right)
        if solved is None:
            continue
        offset = solved[0]
        amplitudes = [0.0] * component_count
        for solution_index, component_index in enumerate(free_components, start=1):
            amplitudes[component_index] = solved[solution_index]
        if any(
            (sign < 0 and amplitude > 0.0)
            or (sign > 0 and amplitude < 0.0)
            for sign, amplitude in zip(signs, amplitudes)
        ):
            continue

        objective = 0.0
        for sample_index, (value, weight) in enumerate(
            zip(series.values, series.inverse_variances)
        ):
            if weight == 0.0:
                continue
            prediction = offset
            for amplitude, bases in zip(amplitudes, component_bases):
                prediction = prediction + amplitude * bases[sample_index]
            residual = value - prediction
            term = weight * residual * residual
            if not _finite_calculation(prediction, residual, term):
                valid = False
                break
            objective += term
            if not math.isfinite(objective):
                valid = False
                break
        if not valid:
            continue
        candidate = (offset, tuple(amplitudes), objective, state_ordinal)
        if accepted is None:
            accepted = candidate
        else:
            limit = _objective_limit(candidate[2], accepted[2])
            if candidate[2] < accepted[2] - limit:
                accepted = candidate
            elif (
                abs(candidate[2] - accepted[2]) <= limit
                and candidate[3] < accepted[3]
            ):
                accepted = candidate
    if accepted is None:
        return None
    return accepted[0], accepted[1], accepted[2]


def _evaluate_candidate(
    dataset: Mapping[str, Any],
    grid_index: int,
) -> _CandidateEvaluation | None:
    validated = _validated_dataset(dataset)
    grid = validated.grid
    try:
        parameters = _candidate_parameters(grid, grid_index)
    except (RuntimeError, TypeError, ValueError, OverflowError):
        return None

    if grid.model_class_id == POSITIVE_PULSE_ONLY:
        geometries = (
            (
                parameters["center"],
                parameters["logScale"],
                parameters["logShape"],
            ),
        )
        signs = (1,)
    else:
        positive_center = (
            parameters["negativeCenter"] + parameters["separation"]
            if grid.model_class_id == ORDERED_NEGATIVE_POSITIVE_DOUBLET
            else parameters["positiveCenter"]
        )
        if (
            not math.isfinite(positive_center)
            or positive_center <= parameters["negativeCenter"]
        ):
            return None
        geometries = (
            (
                parameters["negativeCenter"],
                parameters["negativeLogScale"],
                parameters["negativeLogShape"],
            ),
            (
                positive_center,
                parameters["positiveLogScale"],
                parameters["positiveLogShape"],
            ),
        )
        signs = (-1, 1)

    fits = []
    total_objective = 0.0
    for series in validated.series:
        bases_by_component = []
        for center, log_scale, log_shape in geometries:
            bases = []
            for coordinate, weight in zip(
                series.coordinates,
                series.inverse_variances,
            ):
                if weight == 0.0:
                    bases.append(0.0)
                    continue
                basis = _component_basis(
                    coordinate,
                    center,
                    log_scale,
                    log_shape,
                )
                if basis is None:
                    return None
                bases.append(basis)
            bases_by_component.append(tuple(bases))
        fitted = _fit_series(series, tuple(bases_by_component), signs)
        if fitted is None:
            return None
        offset, amplitudes, objective = fitted
        fits.append(
            _SeriesFit(
                generic_series_id=series.generic_series_id,
                positive_weight_sample_count=series.positive_weight_count,
                offset=offset,
                negative_amplitude=(
                    amplitudes[0] if len(amplitudes) == 2 else None
                ),
                positive_amplitude=(
                    amplitudes[1] if len(amplitudes) == 2 else amplitudes[0]
                ),
                weighted_residual_sum_squares=objective,
            )
        )
        total_objective += objective
        if not math.isfinite(total_objective):
            return None

    positive_weight_sample_count = sum(
        fit.positive_weight_sample_count for fit in fits
    )
    nominal_parameter_count = _nominal_parameter_count(
        grid.model_class_id,
        len(fits),
    )
    try:
        bayesian_information_criterion = (
            total_objective
            + nominal_parameter_count * math.log(positive_weight_sample_count)
        )
        if positive_weight_sample_count > nominal_parameter_count + 1:
            corrected_akaike_information_criterion = (
                total_objective
                + 2.0 * nominal_parameter_count
                + (
                    2.0
                    * nominal_parameter_count
                    * (nominal_parameter_count + 1)
                    / (
                        positive_weight_sample_count
                        - nominal_parameter_count
                        - 1
                    )
                )
            )
            corrected_akaike_information_criterion_defined = True
        else:
            corrected_akaike_information_criterion = None
            corrected_akaike_information_criterion_defined = False
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    metric_values = (total_objective, bayesian_information_criterion)
    if corrected_akaike_information_criterion is not None:
        metric_values = (*metric_values, corrected_akaike_information_criterion)
    if not _finite_calculation(*metric_values):
        return None
    return _CandidateEvaluation(
        grid_index=grid_index,
        parameters=dict(parameters),
        series_fits=tuple(fits),
        positive_weight_sample_count=positive_weight_sample_count,
        weighted_residual_sum_squares=total_objective,
        nominal_parameter_count=nominal_parameter_count,
        bayesian_information_criterion=bayesian_information_criterion,
        corrected_akaike_information_criterion=(
            corrected_akaike_information_criterion
        ),
        corrected_akaike_information_criterion_defined=(
            corrected_akaike_information_criterion_defined
        ),
    )


def _candidate_payload(
    evaluation: _CandidateEvaluation,
    model_class_id: str,
) -> dict[str, Any]:
    series_fits = []
    for fit in evaluation.series_fits:
        record = {
            "genericSeriesID": fit.generic_series_id,
            "positiveWeightSampleCount": fit.positive_weight_sample_count,
            "offset": fit.offset,
            "positiveAmplitude": fit.positive_amplitude,
            "positiveAmplitudeSign": _amplitude_sign(fit.positive_amplitude),
            "weightedResidualSumSquares": fit.weighted_residual_sum_squares,
        }
        if model_class_id != POSITIVE_PULSE_ONLY:
            record["negativeAmplitude"] = fit.negative_amplitude
            if fit.negative_amplitude is None:
                raise ValueError("doublet fit lacks a negative amplitude")
            record["negativeAmplitudeSign"] = _amplitude_sign(
                fit.negative_amplitude
            )
        series_fits.append(record)
    return {
        "gridIndex": evaluation.grid_index,
        "parameters": dict(evaluation.parameters),
        "seriesFits": series_fits,
        "positiveWeightSampleCount": evaluation.positive_weight_sample_count,
        "weightedResidualSumSquares": evaluation.weighted_residual_sum_squares,
        "nominalParameterCount": evaluation.nominal_parameter_count,
        "bayesianInformationCriterion": (
            evaluation.bayesian_information_criterion
        ),
        "correctedAkaikeInformationCriterion": (
            evaluation.corrected_akaike_information_criterion
        ),
        "correctedAkaikeInformationCriterionDefined": (
            evaluation.corrected_akaike_information_criterion_defined
        ),
    }


def _strict_work_payload(work_unit: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = work_unit.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != _WORK_PAYLOAD_FIELDS:
        raise ValueError("work payload does not match the published field set")
    if _WORK_PAYLOAD_FIELDS.intersection(
        key for key in work_unit if key != "payload"
    ):
        raise ValueError("work payload fields must not be flattened")
    return payload


def _result_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _agrees(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= RESULT_RELATIVE_TOLERANCE * max(
        1.0,
        abs(actual),
        abs(expected),
    )


def _invalid_result(message: str) -> ResultValidation:
    return ResultValidation(False, message, {"method": "morphology-grid-invalid"})


def _strict_candidate_payload(value: Any, model_class_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BEST_CANDIDATE_FIELDS:
        raise ValueError("bestCandidate does not match the published field set")
    parameters = value["parameters"]
    expected_parameters = {
        POSITIVE_PULSE_ONLY: _POSITIVE_PARAMETER_FIELDS,
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: _DOUBLET_PARAMETER_FIELDS,
        INDEPENDENT_PULSES: _INDEPENDENT_PARAMETER_FIELDS,
    }[model_class_id]
    if not isinstance(parameters, Mapping) or set(parameters) != expected_parameters:
        raise ValueError("bestCandidate.parameters field set is invalid")
    series_fits = value["seriesFits"]
    expected_fit_fields = (
        _POSITIVE_FIT_FIELDS
        if model_class_id == POSITIVE_PULSE_ONLY
        else _DOUBLET_FIT_FIELDS
    )
    if not isinstance(series_fits, list) or not series_fits:
        raise ValueError("bestCandidate.seriesFits must be nonempty")
    if any(
        not isinstance(item, Mapping) or set(item) != expected_fit_fields
        for item in series_fits
    ):
        raise ValueError("bestCandidate.seriesFits field set is invalid")
    return value


def _candidate_payload_matches(
    candidate: Mapping[str, Any],
    expected: _CandidateEvaluation,
    model_class_id: str,
) -> bool:
    grid_index = _nonnegative_integer(candidate["gridIndex"], "best gridIndex")
    if grid_index != expected.grid_index:
        return False

    returned_parameters = {
        key: _result_number(value, f"parameters.{key}")
        for key, value in candidate["parameters"].items()
    }
    if returned_parameters != dict(expected.parameters):
        return False

    positive_weight_sample_count = _positive_integer(
        candidate["positiveWeightSampleCount"],
        "positiveWeightSampleCount",
    )
    nominal_parameter_count = _positive_integer(
        candidate["nominalParameterCount"],
        "nominalParameterCount",
    )
    returned_objective = _result_number(
        candidate["weightedResidualSumSquares"],
        "weightedResidualSumSquares",
    )
    returned_bic = _result_number(
        candidate["bayesianInformationCriterion"],
        "bayesianInformationCriterion",
    )
    returned_defined = candidate["correctedAkaikeInformationCriterionDefined"]
    if type(returned_defined) is not bool:
        return False
    returned_aicc_value = candidate["correctedAkaikeInformationCriterion"]
    if returned_defined:
        returned_aicc = _result_number(
            returned_aicc_value,
            "correctedAkaikeInformationCriterion",
        )
    elif returned_aicc_value is None:
        returned_aicc = None
    else:
        return False

    if (
        positive_weight_sample_count != expected.positive_weight_sample_count
        or nominal_parameter_count != expected.nominal_parameter_count
        or returned_objective < 0.0
        or not _agrees(
            returned_objective,
            expected.weighted_residual_sum_squares,
        )
        or not _agrees(returned_bic, expected.bayesian_information_criterion)
        or returned_defined
        != expected.corrected_akaike_information_criterion_defined
    ):
        return False
    if returned_defined:
        expected_aicc = expected.corrected_akaike_information_criterion
        if (
            returned_aicc is None
            or expected_aicc is None
            or not _agrees(returned_aicc, expected_aicc)
        ):
            return False
    elif (
        returned_aicc is not None
        or expected.corrected_akaike_information_criterion is not None
    ):
        return False

    returned_fits = candidate["seriesFits"]
    if len(returned_fits) != len(expected.series_fits):
        return False
    returned_sample_count = 0
    returned_series_objective = 0.0
    for returned, expected_fit in zip(returned_fits, expected.series_fits):
        if returned["genericSeriesID"] != expected_fit.generic_series_id:
            return False
        fit_sample_count = _positive_integer(
            returned["positiveWeightSampleCount"],
            "series positiveWeightSampleCount",
        )
        if fit_sample_count != expected_fit.positive_weight_sample_count:
            return False
        returned_sample_count += fit_sample_count
        offset = _result_number(returned["offset"], "offset")
        positive_amplitude = _result_number(
            returned["positiveAmplitude"],
            "positiveAmplitude",
        )
        series_objective = _result_number(
            returned["weightedResidualSumSquares"],
            "series weightedResidualSumSquares",
        )
        if (
            positive_amplitude < 0.0
            or series_objective < 0.0
            or returned["positiveAmplitudeSign"]
            != _amplitude_sign(positive_amplitude)
            or returned["positiveAmplitudeSign"]
            != _amplitude_sign(expected_fit.positive_amplitude)
            or not _agrees(offset, expected_fit.offset)
            or not _agrees(
                positive_amplitude,
                expected_fit.positive_amplitude,
            )
            or not _agrees(
                series_objective,
                expected_fit.weighted_residual_sum_squares,
            )
        ):
            return False
        if model_class_id != POSITIVE_PULSE_ONLY:
            negative_amplitude = _result_number(
                returned["negativeAmplitude"],
                "negativeAmplitude",
            )
            if (
                expected_fit.negative_amplitude is None
                or negative_amplitude > 0.0
                or returned["negativeAmplitudeSign"]
                != _amplitude_sign(negative_amplitude)
                or returned["negativeAmplitudeSign"]
                != _amplitude_sign(expected_fit.negative_amplitude)
                or not _agrees(
                    negative_amplitude,
                    expected_fit.negative_amplitude,
                )
            ):
                return False
        returned_series_objective += series_objective
        if not math.isfinite(returned_series_objective):
            return False
    return (
        returned_sample_count == positive_weight_sample_count
        and _agrees(returned_series_objective, returned_objective)
    )


def _recompute_shard(
    dataset: Mapping[str, Any],
    start_index: int,
    candidate_count: int,
) -> tuple[int, _CandidateEvaluation | None]:
    invalid_count = 0
    best = None
    for grid_index in range(start_index, start_index + candidate_count):
        evaluation = _evaluate_candidate(dataset, grid_index)
        if evaluation is None:
            invalid_count += 1
        elif best is None or _candidate_precedes(evaluation, best):
            best = evaluation
    return invalid_count, best


class MorphologyGridPlugin:
    """Strict server contract for deterministic generic morphology grids."""

    uses_legacy_coordinator_diagnostics = False
    uses_legacy_science_metadata_validation = False
    definition = WorkloadDefinition(
        workload_id=WORKLOAD_ID,
        dataset_schema_id=DATASET_SCHEMA_ID,
        payload_schema_id=PAYLOAD_SCHEMA_ID,
        result_schema_id=RESULT_SCHEMA_ID,
        allows_legacy_schemaless_workers=False,
    )

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        _validated_dataset(dataset)

    def build_work_payloads(
        self,
        dataset: Mapping[str, Any],
    ) -> Iterator[Mapping[str, Any]]:
        grid = _validated_dataset(dataset).grid
        for start_index in range(
            0,
            grid.total_candidates,
            grid.candidates_per_work_unit,
        ):
            yield {
                "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
                "modelClassID": grid.model_class_id,
                "gridStartIndex": start_index,
                "gridCount": min(
                    grid.candidates_per_work_unit,
                    grid.total_candidates - start_index,
                ),
            }

    def legacy_work_unit_fields(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        del payload
        return {}

    def canonicalize_result(
        self,
        work_unit: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del work_unit
        return dict(result)

    def validate_result(
        self,
        work_unit: Mapping[str, Any],
        result: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> ResultValidation:
        if result.get("status") != "completed":
            return _invalid_result("Work unit did not complete.")
        payload = result.get("payload")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _RESULT_PAYLOAD_FIELDS
        ):
            return _invalid_result(
                "Result payload does not match the published field set."
            )
        if _FLATTENABLE_RESULT_FIELDS.intersection(
            key for key in result if key != "payload"
        ):
            return _invalid_result("Workload result fields must not be flattened.")

        try:
            validated = _validated_dataset(dataset)
            grid = validated.grid
            work_payload = _strict_work_payload(work_unit)
            server_start = _nonnegative_integer(
                work_payload["gridStartIndex"], "work gridStartIndex"
            )
            server_count = _positive_integer(
                work_payload["gridCount"],
                "work gridCount",
            )
            returned_start = _nonnegative_integer(
                payload["gridStartIndex"],
                "gridStartIndex",
            )
            returned_count = _positive_integer(payload["gridCount"], "gridCount")
            evaluated_count = _positive_integer(
                payload["evaluatedCandidateCount"], "evaluatedCandidateCount"
            )
            invalid_count = _nonnegative_integer(
                payload["invalidCandidateCount"], "invalidCandidateCount"
            )
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return _invalid_result("Result structure or shard counts are invalid.")

        if (
            work_payload.get("morphologyFamilyID") != MORPHOLOGY_FAMILY_ID
            or payload.get("morphologyFamilyID") != MORPHOLOGY_FAMILY_ID
            or work_payload.get("modelClassID") != grid.model_class_id
            or payload.get("modelClassID") != grid.model_class_id
        ):
            return _invalid_result("Result identity does not match the work unit.")
        if returned_start != server_start or returned_count != server_count:
            return _invalid_result("Result shard does not match the work unit.")
        if server_start > MAX_SAFE_INTEGER - server_count:
            return _invalid_result("Work-unit shard range is not safe.")
        if server_start + server_count > grid.total_candidates:
            return _invalid_result("Work-unit shard exceeds the configured grid.")
        if evaluated_count != server_count:
            return _invalid_result(
                "Evaluated candidate count is inconsistent."
            )
        try:
            expected_invalid_count, expected = _recompute_shard(
                dataset,
                server_start,
                server_count,
            )
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return _invalid_result("Shard recomputation failed.")
        if invalid_count != expected_invalid_count:
            return _invalid_result(
                "Invalid candidate count failed full-shard recomputation."
            )

        raw_candidate = payload["bestCandidate"]
        if expected is None:
            if raw_candidate is not None:
                return _invalid_result(
                    "An all-invalid shard must report a null winner."
                )
            return ResultValidation(
                True,
                "All morphology-grid shard candidates were invalid.",
                {
                    "method": "morphology-grid-full-shard-recomputation",
                    "invalidCandidateCount": expected_invalid_count,
                },
            )
        if raw_candidate is None:
            return _invalid_result("Result omitted the canonical shard winner.")
        try:
            candidate = _strict_candidate_payload(
                raw_candidate,
                grid.model_class_id,
            )
            if not _candidate_payload_matches(
                candidate,
                expected,
                grid.model_class_id,
            ):
                return _invalid_result(
                    "Returned winner failed full-shard recomputation."
                )
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return _invalid_result("Returned winner payload is invalid.")
        return ResultValidation(
            True,
            "Morphology-grid result accepted.",
            {
                "method": "morphology-grid-full-shard-recomputation",
                "bestGridIndex": expected.grid_index,
                "invalidCandidateCount": expected_invalid_count,
            },
        )

    def reduce_dataset(
        self,
        dataset: Mapping[str, Any],
        work_units: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any] | None],
        terminal: bool,
    ) -> DatasetReduction:
        if len(work_units) != len(results):
            raise RuntimeError(
                "Morphology-grid reduction requires aligned work and result sequences"
            )
        grid = _validated_dataset(dataset).grid
        expected_payloads = list(self.build_work_payloads(dataset))
        work_matches = []
        for index, work_unit in enumerate(work_units):
            matches = index < len(expected_payloads)
            if matches:
                try:
                    actual_payload = _strict_work_payload(work_unit)
                    matches = dict(actual_payload) == dict(expected_payloads[index])
                except (KeyError, RuntimeError, TypeError, ValueError):
                    matches = False
            work_matches.append(matches)
        exact_work_coverage = (
            len(work_units) == len(expected_payloads)
            and all(work_matches)
        )
        completed_candidate_count = 0
        total_invalid_candidate_count = 0
        result_accepted = []
        accepted_winners: list[_CandidateEvaluation] = []
        for matches, work_unit, result in zip(work_matches, work_units, results):
            if not matches or result is None:
                result_accepted.append(False)
                continue
            try:
                work_payload = _strict_work_payload(work_unit)
                candidate_count = _positive_integer(
                    work_payload["gridCount"],
                    "work gridCount",
                )
                validation = self.validate_result(work_unit, result, dataset)
                if not validation.accepted:
                    result_accepted.append(False)
                    continue
                result_payload = result["payload"]
                invalid_count = _nonnegative_integer(
                    result_payload["invalidCandidateCount"],
                    "invalidCandidateCount",
                )
                raw_candidate = result_payload["bestCandidate"]
                winner = None
                if raw_candidate is not None:
                    candidate = _strict_candidate_payload(
                        raw_candidate,
                        grid.model_class_id,
                    )
                    best_index = _nonnegative_integer(
                        candidate["gridIndex"],
                        "best gridIndex",
                    )
                    winner = _evaluate_candidate(dataset, best_index)
                    if winner is None:
                        result_accepted.append(False)
                        continue
            except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
                result_accepted.append(False)
                continue
            result_accepted.append(True)
            completed_candidate_count += candidate_count
            total_invalid_candidate_count += invalid_count
            if winner is not None:
                accepted_winners.append(winner)

        coverage_complete = bool(
            terminal
            and exact_work_coverage
            and work_units
            and len(result_accepted) == len(expected_payloads)
            and all(result_accepted)
            and completed_candidate_count == grid.total_candidates
        )
        status = (
            "MORPHOLOGY_GRID_COMPLETE"
            if coverage_complete
            else "MORPHOLOGY_GRID_INCOMPLETE"
        )
        best = None
        for item in sorted(
            accepted_winners,
            key=lambda evaluation: evaluation.grid_index,
        ):
            if best is None or _candidate_precedes(item, best):
                best = item
        best_candidate = (
            _candidate_payload(best, grid.model_class_id)
            if best is not None
            else None
        )
        return DatasetReduction(
            payload={"bestCandidate": dict(best_candidate) if best_candidate else None},
            status_fields={
                "workloadStatus": status,
                "morphologyGridStatus": status,
                "coverageComplete": coverage_complete,
                "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
                "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
                "modelClassID": grid.model_class_id,
                "totalCandidateCount": grid.total_candidates,
                "completedCandidateCount": completed_candidate_count,
                "totalInvalidCandidateCount": total_invalid_candidate_count,
                "bestGridIndex": (
                    best_candidate["gridIndex"] if best_candidate else None
                ),
                "bestParameters": (
                    best_candidate["parameters"] if best_candidate else None
                ),
                "bestSeriesFits": (
                    best_candidate["seriesFits"] if best_candidate else None
                ),
                "bestPositiveWeightSampleCount": (
                    best_candidate["positiveWeightSampleCount"]
                    if best_candidate
                    else None
                ),
                "bestWeightedResidualSumSquares": (
                    best_candidate["weightedResidualSumSquares"]
                    if best_candidate
                    else None
                ),
                "bestNominalParameterCount": (
                    best_candidate["nominalParameterCount"]
                    if best_candidate
                    else None
                ),
                "bestBayesianInformationCriterion": (
                    best_candidate["bayesianInformationCriterion"]
                    if best_candidate
                    else None
                ),
                "bestCorrectedAkaikeInformationCriterion": (
                    best_candidate["correctedAkaikeInformationCriterion"]
                    if best_candidate
                    else None
                ),
                "bestCorrectedAkaikeInformationCriterionDefined": (
                    best_candidate[
                        "correctedAkaikeInformationCriterionDefined"
                    ]
                    if best_candidate
                    else None
                ),
            },
        )

    def contribution_metrics(
        self,
        work_unit: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        validated = _validated_dataset(dataset)
        payload = _strict_work_payload(work_unit)
        if (
            payload.get("morphologyFamilyID") != MORPHOLOGY_FAMILY_ID
            or payload.get("modelClassID") != validated.grid.model_class_id
        ):
            raise RuntimeError("Morphology-grid work payload identity is invalid")
        candidate_count = _positive_integer(payload["gridCount"], "work gridCount")
        start_index = _nonnegative_integer(
            payload["gridStartIndex"],
            "work gridStartIndex",
        )
        if (
            start_index > MAX_SAFE_INTEGER - candidate_count
            or start_index + candidate_count > validated.grid.total_candidates
        ):
            raise RuntimeError("Morphology-grid work payload range is invalid")
        sample_count = sum(len(series.coordinates) for series in validated.series)
        evaluations = _safe_product(
            "sample-candidate evaluation count",
            sample_count,
            candidate_count,
        )
        return {
            "workloadID": WORKLOAD_ID,
            "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
            "modelClassID": validated.grid.model_class_id,
            "seriesCount": len(validated.series),
            "sampleCount": sample_count,
            "candidateCount": candidate_count,
            "sampleCandidateEvaluations": evaluations,
        }


PLUGIN = MorphologyGridPlugin()
