"""Existing Lomb-Scargle workload and historical TESS alias."""

from __future__ import annotations

import math
from typing import Any, Iterator, Mapping, Sequence

from openstar_workloads.contract import (
    DatasetReduction,
    ResultValidation,
    WorkloadDefinition,
)


LOMB_SCARGLE_V1 = "openstar.lomb-scargle.v1"
TESS_PERIOD_SEARCH_V1 = "openstar.tess-period-search.v1"

_DATASET_SCHEMA_ID = "openstar.dataset.lomb-scargle.v1"
_PAYLOAD_SCHEMA_ID = "openstar.payload.lomb-scargle-shard.v1"
_RESULT_SCHEMA_ID = "openstar.result.lomb-scargle-shard.v1"
_LEGACY_FIELDS = (
    "frequencyStartIndex",
    "startFrequency",
    "frequencyStep",
    "frequencyCount",
)


def _first_value(
    mapping: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _frequency_grid(
    dataset: Mapping[str, Any],
) -> tuple[float, float, int, int]:
    search = dataset.get("frequencySearch")
    if not isinstance(search, Mapping):
        raise RuntimeError("Dataset is missing a frequencySearch object")

    try:
        minimum_frequency = float(
            _first_value(
                search,
                "minimumFrequency",
                "minFrequency",
                "startFrequency",
            )
        )
        total_frequencies = int(
            _first_value(
                search,
                "totalFrequencies",
                "frequencyCount",
            )
        )
        frequencies_per_work_unit = int(
            _first_value(
                search,
                "frequenciesPerWorkUnit",
                "workUnitFrequencyCount",
                "chunkSize",
            )
        )
        frequency_step_value = _first_value(
            search,
            "frequencyStep",
            "step",
        )
        if frequency_step_value is None:
            maximum_frequency = float(
                _first_value(
                    search,
                    "maximumFrequency",
                    "maxFrequency",
                    "endFrequency",
                )
            )
            frequency_step_value = (
                0.0
                if total_frequencies <= 1
                else (
                    maximum_frequency - minimum_frequency
                ) / total_frequencies
            )
        frequency_step = float(frequency_step_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("Invalid Lomb-Scargle frequency grid") from error

    if not math.isfinite(minimum_frequency):
        raise RuntimeError("minimumFrequency must be finite.")
    if not math.isfinite(frequency_step) or frequency_step <= 0:
        raise RuntimeError("frequencyStep must be finite and > 0.")
    if total_frequencies <= 0:
        raise RuntimeError("totalFrequencies must be > 0.")
    if frequencies_per_work_unit <= 0:
        raise RuntimeError("frequenciesPerWorkUnit must be > 0.")

    return (
        minimum_frequency,
        frequency_step,
        total_frequencies,
        frequencies_per_work_unit,
    )


def _work_value(work_unit: Mapping[str, Any], key: str) -> Any:
    if key in work_unit and work_unit[key] is not None:
        return work_unit[key]
    payload = work_unit.get("payload")
    if isinstance(payload, Mapping):
        return payload[key]
    raise KeyError(key)


class LombScarglePlugin:
    """Stateless adapter preserving the existing Lomb worker contract."""

    uses_legacy_coordinator_diagnostics = True
    uses_legacy_science_metadata_validation = True

    def __init__(self, workload_id: str) -> None:
        self.definition = WorkloadDefinition(
            workload_id=workload_id,
            dataset_schema_id=_DATASET_SCHEMA_ID,
            payload_schema_id=_PAYLOAD_SCHEMA_ID,
            result_schema_id=_RESULT_SCHEMA_ID,
            allows_legacy_schemaless_workers=True,
        )

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        _frequency_grid(dataset)

    def build_work_payloads(
        self,
        dataset: Mapping[str, Any],
    ) -> Iterator[Mapping[str, Any]]:
        (
            minimum_frequency,
            frequency_step,
            total_frequencies,
            frequencies_per_work_unit,
        ) = _frequency_grid(dataset)

        for start_index in range(
            0,
            total_frequencies,
            frequencies_per_work_unit,
        ):
            frequency_count = min(
                frequencies_per_work_unit,
                total_frequencies - start_index,
            )
            yield {
                "frequencyStartIndex": start_index,
                "startFrequency": (
                    minimum_frequency + start_index * frequency_step
                ),
                "frequencyStep": frequency_step,
                "frequencyCount": frequency_count,
            }

    def legacy_work_unit_fields(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {key: payload[key] for key in _LEGACY_FIELDS}

    def canonicalize_result(
        self,
        work_unit: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del work_unit
        canonical = dict(result)
        payload = canonical.get("payload")
        if isinstance(payload, Mapping):
            for key in (
                "bestFrequency",
                "bestPeriodDays",
                "bestPower",
            ):
                if canonical.get(key) is None and payload.get(key) is not None:
                    canonical[key] = payload[key]
        return canonical

    def validate_result(
        self,
        work_unit: Mapping[str, Any],
        result: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> ResultValidation:
        del dataset
        details = {
            "method": "metal-result-invalid",
            "deviceFrequency": None,
            "devicePower": None,
        }

        if result.get("status") != "completed":
            return ResultValidation(
                False,
                "Work unit did not complete.",
                details,
            )
        if result.get("bestFrequency") is None:
            return ResultValidation(False, "Missing best frequency.", details)
        if result.get("bestPower") is None:
            return ResultValidation(False, "Missing best power.", details)

        try:
            best_frequency = float(result["bestFrequency"])
            best_power = float(result["bestPower"])
        except (TypeError, ValueError):
            return ResultValidation(
                False,
                "Best frequency/power must be numeric.",
                details,
            )

        details.update(
            {
                "deviceFrequency": best_frequency,
                "devicePower": best_power,
            }
        )
        if not math.isfinite(best_frequency):
            return ResultValidation(
                False,
                "Best frequency is not finite.",
                details,
            )
        if not math.isfinite(best_power):
            return ResultValidation(
                False,
                "Best power is not finite.",
                details,
            )

        try:
            start_frequency = float(_work_value(work_unit, "startFrequency"))
            frequency_step = float(_work_value(work_unit, "frequencyStep"))
            frequency_count = int(_work_value(work_unit, "frequencyCount"))
        except (KeyError, TypeError, ValueError, OverflowError):
            return ResultValidation(
                False,
                "Assigned frequency grid is malformed.",
                details,
            )

        end_frequency = (
            start_frequency
            + max(frequency_count - 1, 0) * frequency_step
        )
        grid_tolerance = max(abs(frequency_step) * 2.0, 1.0e-7)
        if (
            best_frequency < start_frequency - grid_tolerance
            or best_frequency > end_frequency + grid_tolerance
        ):
            return ResultValidation(
                False,
                "Best frequency is outside work-unit range.",
                details,
            )

        return ResultValidation(
            True,
            "Metal result is structurally valid.",
            details,
        )

    def reduce_dataset(
        self,
        dataset: Mapping[str, Any],
        work_units: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any] | None],
        terminal: bool,
    ) -> DatasetReduction:
        del dataset, work_units, terminal
        best = None
        for result in results:
            if result is None or result.get("bestPower") is None:
                continue
            if best is None or float(result["bestPower"]) > float(
                best["bestPower"]
            ):
                best = result
        return DatasetReduction(
            payload={"best": dict(best) if best is not None else None},
            status_fields={},
        )

    def contribution_metrics(
        self,
        work_unit: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        metrics = {"workloadID": self.definition.workload_id}
        if self.definition.workload_id != LOMB_SCARGLE_V1:
            return metrics

        samples = dataset.get("times")
        sample_count = len(samples) if isinstance(samples, list) else 0
        payload = work_unit.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
        frequency_count = int(
            payload.get(
                "frequencyCount",
                work_unit.get("frequencyCount", 0),
            )
        )
        metrics.update(
            {
                "sampleCount": sample_count,
                "frequencyCount": frequency_count,
                "sampleFrequencyEvaluations": (
                    sample_count * frequency_count
                ),
            }
        )
        return metrics


PLUGIN = (
    LombScarglePlugin(LOMB_SCARGLE_V1),
    LombScarglePlugin(TESS_PERIOD_SEARCH_V1),
)
