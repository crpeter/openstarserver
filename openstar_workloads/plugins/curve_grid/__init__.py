"""Strict, domain-neutral curve-grid workload."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from openstar_workloads.contract import (
    DatasetReduction,
    ResultValidation,
    WorkloadDefinition,
)


WORKLOAD_ID = "openstar.curve-grid.v1"
DATASET_SCHEMA_ID = "openstar.dataset.curve-grid.v1"
PAYLOAD_SCHEMA_ID = "openstar.payload.curve-grid-shard.v1"
RESULT_SCHEMA_ID = "openstar.result.curve-grid-shard.v1"
FAMILY_ID = (
    "openstar.curve-family.symmetric-radial-amplification.v1"
)

MAX_SAFE_INTEGER = (1 << 53) - 1
_SINGULARITY_RELATIVE_TOLERANCE = 1.0e-12
_RESULT_RELATIVE_TOLERANCE = 1.0e-9

_AXIS_FIELDS = frozenset(("start", "step", "count"))
_GRID_FIELDS = frozenset(
    (
        "familyID",
        "centerAxis",
        "logScaleAxis",
        "logShapeAxis",
        "candidatesPerWorkUnit",
    )
)
_DATASET_FIELDS = frozenset(
    (
        "id",
        "datasetSchemaID",
        "coordinates",
        "values",
        "inverseVariances",
        "curveGrid",
    )
)
_WORK_PAYLOAD_FIELDS = frozenset(
    ("familyID", "gridStartIndex", "gridCount")
)
_RESULT_PAYLOAD_FIELDS = frozenset(
    (
        "familyID",
        "gridStartIndex",
        "gridCount",
        "bestGridIndex",
        "bestCenter",
        "bestLogScale",
        "bestLogShape",
        "bestOffset",
        "bestAmplitude",
        "bestWeightedResidualSumSquares",
        "evaluatedCandidateCount",
        "invalidCandidateCount",
    )
)


@dataclass(frozen=True, slots=True)
class _Axis:
    start: float
    step: float
    count: int


@dataclass(frozen=True, slots=True)
class _Grid:
    center: _Axis
    log_scale: _Axis
    log_shape: _Axis
    candidates_per_work_unit: int
    total_candidates: int


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    grid_index: int
    center: float
    log_scale: float
    log_shape: float
    offset: float
    amplitude: float
    weighted_residual_sum_squares: float


def _runtime_error(message: str) -> RuntimeError:
    return RuntimeError(f"curve-grid dataset: {message}")


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
        if value > MAX_SAFE_INTEGER // product:
            raise _runtime_error(
                f"{field_name} exceeds the safe integer range"
            )
        product *= value
    return product


def _axis(value: Any, field_name: str, *, exponentiated: bool) -> _Axis:
    if not isinstance(value, Mapping) or set(value) != _AXIS_FIELDS:
        raise _runtime_error(
            f"{field_name} must contain exactly start, step, and count"
        )

    start = _finite_number(value["start"], f"{field_name}.start")
    step = _finite_number(value["step"], f"{field_name}.step")
    count = _positive_integer(value["count"], f"{field_name}.count")
    if count > 1 and step == 0.0:
        raise _runtime_error(
            f"{field_name}.step must be nonzero when count is greater than one"
        )

    last = start + (count - 1) * step
    if not math.isfinite(last):
        raise _runtime_error(f"{field_name} has a nonfinite endpoint")

    if exponentiated:
        for endpoint in (start, last):
            try:
                expanded = math.exp(endpoint)
            except OverflowError as error:
                raise _runtime_error(
                    f"{field_name} has an invalid exponentiated endpoint"
                ) from error
            if not math.isfinite(expanded) or expanded <= 0.0:
                raise _runtime_error(
                    f"{field_name} has an invalid exponentiated endpoint"
                )

    return _Axis(start=start, step=step, count=count)


def _grid(dataset: Mapping[str, Any]) -> _Grid:
    curve_grid = dataset.get("curveGrid")
    if not isinstance(curve_grid, Mapping) or set(curve_grid) != _GRID_FIELDS:
        raise _runtime_error(
            "curveGrid does not match the published field set"
        )
    if curve_grid.get("familyID") != FAMILY_ID:
        raise _runtime_error("curveGrid.familyID is invalid")

    center = _axis(
        curve_grid["centerAxis"],
        "curveGrid.centerAxis",
        exponentiated=False,
    )
    log_scale = _axis(
        curve_grid["logScaleAxis"],
        "curveGrid.logScaleAxis",
        exponentiated=True,
    )
    log_shape = _axis(
        curve_grid["logShapeAxis"],
        "curveGrid.logShapeAxis",
        exponentiated=True,
    )
    candidates_per_work_unit = _positive_integer(
        curve_grid["candidatesPerWorkUnit"],
        "curveGrid.candidatesPerWorkUnit",
    )
    total_candidates = _safe_product(
        "total candidate count",
        center.count,
        log_scale.count,
        log_shape.count,
    )
    return _Grid(
        center=center,
        log_scale=log_scale,
        log_shape=log_shape,
        candidates_per_work_unit=candidates_per_work_unit,
        total_candidates=total_candidates,
    )


def _validated_dataset(
    dataset: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], _Grid]:
    if not isinstance(dataset, Mapping):
        raise _runtime_error("dataset must be a mapping")
    missing = _DATASET_FIELDS.difference(dataset)
    if missing:
        raise _runtime_error(
            "missing required fields: " + ", ".join(sorted(missing))
        )
    if not isinstance(dataset["id"], str):
        raise _runtime_error("id must be a string")
    if dataset["datasetSchemaID"] != DATASET_SCHEMA_ID:
        raise _runtime_error("datasetSchemaID is invalid")

    arrays = []
    for field_name in ("coordinates", "values", "inverseVariances"):
        values = dataset[field_name]
        if not isinstance(values, list):
            raise _runtime_error(f"{field_name} must be an array")
        arrays.append(
            tuple(
                _finite_number(item, f"{field_name}[{index}]")
                for index, item in enumerate(values)
            )
        )

    coordinates, values, inverse_variances = arrays
    if len(coordinates) < 3:
        raise _runtime_error("arrays must contain at least three samples")
    if not (
        len(coordinates) == len(values) == len(inverse_variances)
    ):
        raise _runtime_error("arrays must have equal length")
    if any(weight <= 0.0 for weight in inverse_variances):
        raise _runtime_error("inverseVariances must be strictly positive")

    grid = _grid(dataset)
    _safe_product(
        "sample-candidate evaluation count",
        len(coordinates),
        grid.total_candidates,
    )
    return coordinates, values, inverse_variances, grid


def _grid_indices(grid: _Grid, grid_index: int) -> tuple[int, int, int]:
    if type(grid_index) is not int:
        raise ValueError("grid index must be an integer")
    if grid_index < 0 or grid_index >= grid.total_candidates:
        raise ValueError("grid index is outside the configured grid")

    combined_index, shape_index = divmod(
        grid_index,
        grid.log_shape.count,
    )
    center_index, scale_index = divmod(
        combined_index,
        grid.log_scale.count,
    )
    return center_index, scale_index, shape_index


def _grid_index(
    grid: _Grid,
    center_index: int,
    scale_index: int,
    shape_index: int,
) -> int:
    indices = (
        (center_index, grid.center.count, "center"),
        (scale_index, grid.log_scale.count, "scale"),
        (shape_index, grid.log_shape.count, "shape"),
    )
    for index, count, label in indices:
        if type(index) is not int or index < 0 or index >= count:
            raise ValueError(f"{label} index is outside its axis")
    return (
        (
            center_index * grid.log_scale.count
            + scale_index
        )
        * grid.log_shape.count
        + shape_index
    )


def _grid_parameters(
    grid: _Grid,
    grid_index: int,
) -> tuple[float, float, float]:
    center_index, scale_index, shape_index = _grid_indices(
        grid,
        grid_index,
    )
    return (
        grid.center.start + center_index * grid.center.step,
        grid.log_scale.start + scale_index * grid.log_scale.step,
        grid.log_shape.start + shape_index * grid.log_shape.step,
    )


def _finite_calculation(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _evaluate_candidate(
    dataset: Mapping[str, Any],
    grid_index: int,
) -> _CandidateEvaluation | None:
    coordinates, values, weights, grid = _validated_dataset(dataset)
    center, log_scale, log_shape = _grid_parameters(grid, grid_index)

    try:
        scale = math.exp(log_scale)
        shape = math.exp(log_shape)
    except OverflowError:
        return None
    if not _finite_calculation(scale, shape) or scale <= 0.0 or shape <= 0.0:
        return None

    bases = []
    for coordinate in coordinates:
        try:
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
        if not _finite_calculation(
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
        ):
            return None
        bases.append(basis)

    total_weight = 0.0
    weighted_basis = 0.0
    weighted_basis_squared = 0.0
    weighted_value = 0.0
    weighted_basis_value = 0.0
    for weight, basis, value in zip(weights, bases, values):
        terms = (
            weight,
            weight * basis,
            weight * basis * basis,
            weight * value,
            weight * basis * value,
        )
        if not _finite_calculation(*terms):
            return None
        total_weight += terms[0]
        weighted_basis += terms[1]
        weighted_basis_squared += terms[2]
        weighted_value += terms[3]
        weighted_basis_value += terms[4]
        if not _finite_calculation(
            total_weight,
            weighted_basis,
            weighted_basis_squared,
            weighted_value,
            weighted_basis_value,
        ):
            return None

    determinant_left = total_weight * weighted_basis_squared
    determinant_right = weighted_basis * weighted_basis
    determinant = determinant_left - determinant_right
    determinant_limit = _SINGULARITY_RELATIVE_TOLERANCE * max(
        abs(determinant_left),
        abs(determinant_right),
        1.0,
    )
    if not _finite_calculation(
        determinant_left,
        determinant_right,
        determinant,
        determinant_limit,
    ) or determinant <= determinant_limit:
        return None

    offset_numerator_left = weighted_value * weighted_basis_squared
    offset_numerator_right = weighted_basis_value * weighted_basis
    amplitude_numerator_left = total_weight * weighted_basis_value
    amplitude_numerator_right = weighted_basis * weighted_value
    if not _finite_calculation(
        offset_numerator_left,
        offset_numerator_right,
        amplitude_numerator_left,
        amplitude_numerator_right,
    ):
        return None

    offset_numerator = offset_numerator_left - offset_numerator_right
    amplitude_numerator = (
        amplitude_numerator_left - amplitude_numerator_right
    )
    offset = offset_numerator / determinant
    amplitude = amplitude_numerator / determinant
    if not _finite_calculation(
        offset_numerator,
        amplitude_numerator,
        offset,
        amplitude,
    ):
        return None

    weighted_residual_sum_squares = 0.0
    for weight, basis, value in zip(weights, bases, values):
        predicted = offset + amplitude * basis
        residual = value - predicted
        weighted_residual = weight * residual * residual
        if not _finite_calculation(predicted, residual, weighted_residual):
            return None
        weighted_residual_sum_squares += weighted_residual
        if not math.isfinite(weighted_residual_sum_squares):
            return None

    return _CandidateEvaluation(
        grid_index=grid_index,
        center=center,
        log_scale=log_scale,
        log_shape=log_shape,
        offset=offset,
        amplitude=amplitude,
        weighted_residual_sum_squares=weighted_residual_sum_squares,
    )


def _strict_work_payload(work_unit: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = work_unit.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != _WORK_PAYLOAD_FIELDS:
        raise ValueError("work payload does not match the published field set")
    return payload


def _result_number(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _agrees(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= _RESULT_RELATIVE_TOLERANCE * max(
        1.0,
        abs(expected),
    )


def _invalid_result(message: str) -> ResultValidation:
    return ResultValidation(False, message, {"method": "curve-grid-invalid"})


class CurveGridPlugin:
    """Strict server-side contract for one deterministic curve-grid lane."""

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
        _, _, _, grid = _validated_dataset(dataset)
        for start_index in range(
            0,
            grid.total_candidates,
            grid.candidates_per_work_unit,
        ):
            yield {
                "familyID": FAMILY_ID,
                "gridStartIndex": start_index,
                "gridCount": min(
                    grid.candidates_per_work_unit,
                    grid.total_candidates - start_index,
                ),
            }

    def legacy_work_unit_fields(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
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
        if _RESULT_PAYLOAD_FIELDS.intersection(
            key for key in result if key != "payload"
        ):
            return _invalid_result(
                "Workload result fields must not be flattened."
            )

        try:
            work_payload = _strict_work_payload(work_unit)
            server_start = _nonnegative_integer(
                work_payload["gridStartIndex"],
                "work gridStartIndex",
            )
            server_count = _positive_integer(
                work_payload["gridCount"],
                "work gridCount",
            )
            returned_start = _nonnegative_integer(
                payload["gridStartIndex"],
                "gridStartIndex",
            )
            returned_count = _positive_integer(
                payload["gridCount"],
                "gridCount",
            )
            best_index = _nonnegative_integer(
                payload["bestGridIndex"],
                "bestGridIndex",
            )
            evaluated_count = _positive_integer(
                payload["evaluatedCandidateCount"],
                "evaluatedCandidateCount",
            )
            invalid_count = _nonnegative_integer(
                payload["invalidCandidateCount"],
                "invalidCandidateCount",
            )
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return _invalid_result("Result shard counts are invalid.")

        if (
            work_payload.get("familyID") != FAMILY_ID
            or payload.get("familyID") != FAMILY_ID
        ):
            return _invalid_result("Result family does not match the work unit.")
        if returned_start != server_start or returned_count != server_count:
            return _invalid_result("Result shard does not match the work unit.")
        if server_start > MAX_SAFE_INTEGER - server_count:
            return _invalid_result("Work-unit shard range is not safe.")
        if not server_start <= best_index < server_start + server_count:
            return _invalid_result("Best grid index is outside the work unit.")
        if evaluated_count != server_count:
            return _invalid_result(
                "Evaluated candidate count does not match the work unit."
            )
        if invalid_count >= evaluated_count:
            return _invalid_result("Invalid candidate count is inconsistent.")

        try:
            _, _, _, grid = _validated_dataset(dataset)
            expected_parameters = _grid_parameters(grid, best_index)
            returned_parameters = (
                _result_number(payload, "bestCenter"),
                _result_number(payload, "bestLogScale"),
                _result_number(payload, "bestLogShape"),
            )
            returned_offset = _result_number(payload, "bestOffset")
            returned_amplitude = _result_number(payload, "bestAmplitude")
            returned_objective = _result_number(
                payload,
                "bestWeightedResidualSumSquares",
            )
        except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
            return _invalid_result("Result contains invalid numerical fields.")

        if returned_parameters != expected_parameters:
            return _invalid_result(
                "Returned grid parameters do not match the best grid index."
            )
        if returned_objective < 0.0:
            return _invalid_result(
                "Weighted residual sum of squares must be nonnegative."
            )

        expected = _evaluate_candidate(dataset, best_index)
        if expected is None:
            return _invalid_result("Reported best candidate is invalid.")
        if not _agrees(returned_offset, expected.offset):
            return _invalid_result("Returned offset failed recomputation.")
        if not _agrees(returned_amplitude, expected.amplitude):
            return _invalid_result("Returned amplitude failed recomputation.")
        if not _agrees(
            returned_objective,
            expected.weighted_residual_sum_squares,
        ):
            return _invalid_result("Returned objective failed recomputation.")

        return ResultValidation(
            True,
            "Curve-grid result accepted.",
            {
                "method": "curve-grid-recomputation",
                "bestGridIndex": best_index,
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
                "Curve-grid reduction requires aligned work and result sequences"
            )

        _, _, _, grid = _validated_dataset(dataset)
        expected_payloads = list(self.build_work_payloads(dataset))
        exact_work_coverage = len(work_units) == len(expected_payloads)
        if exact_work_coverage:
            for work_unit, expected_payload in zip(
                work_units,
                expected_payloads,
            ):
                try:
                    actual_payload = _strict_work_payload(work_unit)
                except ValueError:
                    exact_work_coverage = False
                    break
                if dict(actual_payload) != dict(expected_payload):
                    exact_work_coverage = False
                    break

        completed_candidate_count = 0
        accepted_winners = []
        for work_unit, result in zip(work_units, results):
            if result is None:
                continue
            try:
                work_payload = _strict_work_payload(work_unit)
                candidate_count = _positive_integer(
                    work_payload["gridCount"],
                    "work gridCount",
                )
                result_payload = result["payload"]
                if (
                    not isinstance(result_payload, Mapping)
                    or set(result_payload) != _RESULT_PAYLOAD_FIELDS
                ):
                    continue
                objective = _result_number(
                    result_payload,
                    "bestWeightedResidualSumSquares",
                )
                best_index = _nonnegative_integer(
                    result_payload["bestGridIndex"],
                    "bestGridIndex",
                )
            except (
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                continue
            completed_candidate_count += candidate_count
            accepted_winners.append(
                (objective, best_index, dict(result_payload))
            )

        coverage_complete = bool(
            terminal
            and exact_work_coverage
            and work_units
            and all(result is not None for result in results)
            and completed_candidate_count == grid.total_candidates
            and len(accepted_winners) == len(work_units)
        )
        curve_status = (
            "CURVE_GRID_COMPLETE"
            if coverage_complete
            else "CURVE_GRID_INCOMPLETE"
        )
        best_payload = (
            min(accepted_winners, key=lambda item: (item[0], item[1]))[2]
            if accepted_winners
            else None
        )
        status_fields = {
            "workloadStatus": curve_status,
            "curveGridStatus": curve_status,
            "coverageComplete": coverage_complete,
            "familyID": FAMILY_ID,
            "totalCandidateCount": grid.total_candidates,
            "completedCandidateCount": completed_candidate_count,
            "bestGridIndex": (
                best_payload["bestGridIndex"] if best_payload else None
            ),
            "bestCenter": best_payload["bestCenter"] if best_payload else None,
            "bestLogScale": (
                best_payload["bestLogScale"] if best_payload else None
            ),
            "bestLogShape": (
                best_payload["bestLogShape"] if best_payload else None
            ),
            "bestOffset": best_payload["bestOffset"] if best_payload else None,
            "bestAmplitude": (
                best_payload["bestAmplitude"] if best_payload else None
            ),
            "bestWeightedResidualSumSquares": (
                best_payload["bestWeightedResidualSumSquares"]
                if best_payload
                else None
            ),
        }
        return DatasetReduction(
            payload={
                "best": dict(best_payload) if best_payload is not None else None
            },
            status_fields=status_fields,
        )

    def contribution_metrics(
        self,
        work_unit: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        coordinates, _, _, _ = _validated_dataset(dataset)
        payload = _strict_work_payload(work_unit)
        if payload.get("familyID") != FAMILY_ID:
            raise RuntimeError("Curve-grid work payload has an invalid family")
        candidate_count = _positive_integer(
            payload["gridCount"],
            "work gridCount",
        )
        sample_count = len(coordinates)
        evaluations = _safe_product(
            "sample-candidate evaluation count",
            sample_count,
            candidate_count,
        )
        return {
            "workloadID": WORKLOAD_ID,
            "familyID": FAMILY_ID,
            "sampleCount": sample_count,
            "candidateCount": candidate_count,
            "sampleCandidateEvaluations": evaluations,
        }


PLUGIN = CurveGridPlugin()
