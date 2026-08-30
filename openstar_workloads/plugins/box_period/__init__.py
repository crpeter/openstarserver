"""Existing frequency-window periodic-box workload."""

from __future__ import annotations

import math
from typing import Any, Iterator, Mapping, Sequence

from openstar_workloads.contract import (
    DatasetReduction,
    ResultValidation,
    WorkloadDefinition,
)


BOX_PERIOD_SEARCH_V1 = "openstar.box-period-search.v1"

_RESULT_FIELDS = (
    "bestFrequency",
    "bestScore",
    "bestPhase",
    "bestDurationFraction",
    "bestFrequencyIndex",
    "bestDurationIndex",
    "bestPhaseBin",
    "inBoxSamples",
    "outOfBoxSamples",
)


def _dataset_label(dataset: Mapping[str, Any]) -> str:
    return str(dataset.get("id") or "dataset")


def _window_payload(
    dataset_label: str,
    window: Mapping[str, Any],
    window_index: int,
    phase_bin_count: int,
    durations: tuple[float, ...],
    minimum_in: int,
    minimum_out: int,
) -> dict[str, Any]:
    try:
        start_frequency = float(window["startFrequency"])
        frequency_step = float(window["frequencyStep"])
        frequency_count = int(window["frequencyCount"])
        frequency_start_index = int(
            window.get("frequencyStartIndex", 0)
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{dataset_label}: invalid frequency window {window_index}"
        ) from error

    if (
        not math.isfinite(start_frequency)
        or start_frequency <= 0
        or not math.isfinite(frequency_step)
        or frequency_step <= 0
        or frequency_count <= 0
    ):
        raise RuntimeError(
            f"{dataset_label}: invalid frequency window {window_index}"
        )

    payload = {
        "startFrequency": start_frequency,
        "frequencyStep": frequency_step,
        "frequencyCount": frequency_count,
        "frequencyStartIndex": frequency_start_index,
        "phaseBinCount": phase_bin_count,
        "durationFractions": list(durations),
        "minimumInBoxSamples": minimum_in,
        "minimumOutOfBoxSamples": minimum_out,
        "windowIndex": window_index,
    }
    for key in ("familyRank", "familyID", "centerFrequency"):
        if window.get(key) is not None:
            payload[key] = window[key]
    return payload


def _box_configuration(
    dataset: Mapping[str, Any],
    *,
    validate_windows: bool = True,
) -> tuple[
    list[Any],
    int,
    tuple[float, ...],
    int,
    int,
]:
    label = _dataset_label(dataset)
    search = dataset.get("boxPeriodSearch")
    if not isinstance(search, Mapping):
        raise RuntimeError(f"{label}: missing boxPeriodSearch object")

    windows = search.get("frequencyWindows")
    if not isinstance(windows, list) or not windows:
        raise RuntimeError(f"{label}: frequencyWindows must be nonempty")

    try:
        phase_bin_count = int(search.get("phaseBinCount", 0))
        minimum_in = int(search.get("minimumInBoxSamples", 0))
        minimum_out = int(search.get("minimumOutOfBoxSamples", 0))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{label}: periodic-box integer settings are invalid"
        ) from error

    duration_values = search.get("durationFractions")
    if not isinstance(duration_values, list) or not duration_values:
        raise RuntimeError(
            f"{label}: durationFractions must be nonempty"
        )
    try:
        durations = tuple(float(value) for value in duration_values)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{label}: durationFractions must be numeric"
        ) from error

    if phase_bin_count < 2:
        raise RuntimeError(f"{label}: phaseBinCount must be >= 2")
    if any(
        not math.isfinite(value) or value <= 0 or value >= 1
        for value in durations
    ):
        raise RuntimeError(
            f"{label}: durationFractions must be finite and between 0 and 1"
        )
    if minimum_in <= 0 or minimum_out <= 0:
        raise RuntimeError(
            f"{label}: sample-count gates must be positive"
        )

    if validate_windows:
        for window_index, window in enumerate(windows):
            if not isinstance(window, Mapping):
                raise RuntimeError(
                    f"{label}: invalid frequency window {window_index}"
                )
            _window_payload(
                label,
                window,
                window_index,
                phase_bin_count,
                durations,
                minimum_in,
                minimum_out,
            )

    return (
        windows,
        phase_bin_count,
        durations,
        minimum_in,
        minimum_out,
    )


def _exact_integer(result: Mapping[str, Any], key: str) -> int:
    value = result[key]
    if isinstance(value, bool):
        raise ValueError(key)
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(key)
    return int(numeric)


class BoxPeriodPlugin:
    """Stateless adapter preserving the existing periodic-box contract."""

    uses_legacy_coordinator_diagnostics = False
    uses_legacy_science_metadata_validation = True
    definition = WorkloadDefinition(
        workload_id=BOX_PERIOD_SEARCH_V1,
        dataset_schema_id="openstar.dataset.box-period-search.v1",
        payload_schema_id="openstar.payload.box-period-shard.v1",
        result_schema_id="openstar.result.box-period-shard.v1",
        allows_legacy_schemaless_workers=True,
    )

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        _box_configuration(dataset)

    def build_work_payloads(
        self,
        dataset: Mapping[str, Any],
    ) -> Iterator[Mapping[str, Any]]:
        (
            windows,
            phase_bin_count,
            durations,
            minimum_in,
            minimum_out,
        ) = _box_configuration(dataset, validate_windows=False)
        label = _dataset_label(dataset)

        for window_index, window in enumerate(windows):
            if not isinstance(window, Mapping):
                raise RuntimeError(
                    f"{label}: invalid frequency window {window_index}"
                )
            yield _window_payload(
                label,
                window,
                window_index,
                phase_bin_count,
                durations,
                minimum_in,
                minimum_out,
            )

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
        canonical = dict(result)
        payload = canonical.get("payload")
        if isinstance(payload, Mapping):
            for key in _RESULT_FIELDS:
                if canonical.get(key) is None and payload.get(key) is not None:
                    canonical[key] = payload[key]
        return canonical

    def validate_result(
        self,
        work_unit: Mapping[str, Any],
        result: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> ResultValidation:
        details = {
            "method": "metal-result-invalid",
            "deviceFrequency": None,
            "deviceScore": None,
        }
        if result.get("status") != "completed":
            return ResultValidation(
                False,
                "Work unit did not complete.",
                details,
            )
        if any(result.get(key) is None for key in _RESULT_FIELDS):
            return ResultValidation(
                False,
                "Missing periodic-box result field.",
                details,
            )

        try:
            frequency = float(result["bestFrequency"])
            score = float(result["bestScore"])
            phase = float(result["bestPhase"])
            duration = float(result["bestDurationFraction"])
            frequency_index = _exact_integer(
                result,
                "bestFrequencyIndex",
            )
            duration_index = _exact_integer(result, "bestDurationIndex")
            phase_bin = _exact_integer(result, "bestPhaseBin")
            in_count = _exact_integer(result, "inBoxSamples")
            out_count = _exact_integer(result, "outOfBoxSamples")
        except (TypeError, ValueError, OverflowError):
            return ResultValidation(
                False,
                "Periodic-box result fields must be numeric.",
                details,
            )

        details.update(
            {
                "deviceFrequency": frequency,
                "deviceScore": score,
            }
        )
        if not all(
            math.isfinite(value)
            for value in (frequency, score, phase, duration)
        ):
            return ResultValidation(
                False,
                "Periodic-box result contains a non-finite value.",
                details,
            )

        payload = work_unit.get("payload")
        if not isinstance(payload, Mapping):
            return ResultValidation(
                False,
                "Periodic-box work payload is malformed.",
                details,
            )
        try:
            start = float(payload["startFrequency"])
            step = float(payload["frequencyStep"])
            count = int(payload["frequencyCount"])
            start_index = int(payload.get("frequencyStartIndex", 0))
            bins = int(payload["phaseBinCount"])
            durations = tuple(
                float(value) for value in payload["durationFractions"]
            )
            minimum_in = int(payload["minimumInBoxSamples"])
            minimum_out = int(payload["minimumOutOfBoxSamples"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return ResultValidation(
                False,
                "Periodic-box work payload is malformed.",
                details,
            )

        relative_index = frequency_index - start_index
        if relative_index < 0 or relative_index >= count:
            return ResultValidation(
                False,
                "Best frequency index is outside work-unit range.",
                details,
            )
        expected_frequency = start + relative_index * step
        frequency_tolerance = max(
            abs(expected_frequency) * 2.0e-6,
            abs(step) * 2.0e-5,
            1.0e-7,
        )
        if abs(frequency - expected_frequency) > frequency_tolerance:
            return ResultValidation(
                False,
                "Best frequency does not match its grid index.",
                details,
            )

        if duration_index < 0 or duration_index >= len(durations):
            return ResultValidation(
                False,
                "Best duration index is outside configured durations.",
                details,
            )
        expected_bins = max(
            1,
            min(
                bins - 1,
                int(math.floor(durations[duration_index] * bins + 0.5)),
            ),
        )
        expected_duration = expected_bins / bins
        if (
            phase_bin < 0
            or phase_bin >= bins
            or abs(phase - phase_bin / bins) > 1.0e-6
        ):
            return ResultValidation(
                False,
                "Best phase does not match its phase-bin index.",
                details,
            )
        if abs(duration - expected_duration) > 1.0e-6:
            return ResultValidation(
                False,
                "Best duration does not match its configured index.",
                details,
            )
        if in_count < minimum_in:
            return ResultValidation(
                False,
                "In-box sample gate was not satisfied.",
                details,
            )
        if out_count < minimum_out:
            return ResultValidation(
                False,
                "Out-of-box sample gate was not satisfied.",
                details,
            )

        series = dataset.get("coordinates")
        if not isinstance(series, list):
            series = dataset.get("times")
        if (
            not isinstance(series, list)
            or in_count + out_count != len(series)
        ):
            return ResultValidation(
                False,
                "Periodic-box sample counts do not partition the dataset.",
                details,
            )

        details.update(
            {
                "method": "periodic-box-result",
                "referenceComparisonStatus": "not-applicable",
            }
        )
        return ResultValidation(
            True,
            "Periodic-box result accepted.",
            details,
        )

    def reduce_dataset(
        self,
        dataset: Mapping[str, Any],
        work_units: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any] | None],
        terminal: bool,
    ) -> DatasetReduction:
        del dataset
        if len(work_units) != len(results):
            raise RuntimeError(
                "Periodic-box reduction requires aligned work and result "
                "sequences"
            )

        candidates = []
        for work_unit, result in zip(work_units, results):
            if result is None or result.get("bestScore") is None:
                continue
            payload = work_unit["payload"]
            work_id = str(work_unit["id"]).strip().lower()
            candidates.append(
                {
                    "workID": work_id,
                    "windowIndex": payload.get("windowIndex"),
                    "familyRank": payload.get("familyRank"),
                    "familyID": payload.get("familyID"),
                    "centerFrequency": payload.get("centerFrequency"),
                    "frequency": float(result["bestFrequency"]),
                    "score": float(result["bestScore"]),
                    "phase": float(result["bestPhase"]),
                    "durationFraction": float(
                        result["bestDurationFraction"]
                    ),
                    "frequencyIndex": int(result["bestFrequencyIndex"]),
                    "durationIndex": int(result["bestDurationIndex"]),
                    "phaseBin": int(result["bestPhaseBin"]),
                    "inBoxSamples": int(result["inBoxSamples"]),
                    "outOfBoxSamples": int(result["outOfBoxSamples"]),
                }
            )

        candidates.sort(
            key=lambda item: (
                item["windowIndex"]
                if item["windowIndex"] is not None
                else math.inf,
                item["frequencyIndex"],
            )
        )
        coverage_complete = bool(
            terminal
            and work_units
            and all(result is not None for result in results)
        )
        period_status = (
            "BOX_SEARCH_COMPLETE"
            if coverage_complete
            else "INCOMPLETE_COVERAGE"
            if terminal
            else "SEARCHING"
        )
        status_fields = {
            "periodStatus": period_status,
            "periodConfidence": None,
            "coverageComplete": coverage_complete,
            "bestFrequency": None,
            "bestPeriodDays": None,
            "bestPower": None,
            "candidateFrequency": None,
            "candidatePeriodDays": None,
            "candidatePower": None,
            "candidateFoldCoherence": None,
            "candidatePeakProminenceRatio": None,
            "candidateFrequencyConfidenceInterval": None,
            "candidateFrequencyUncertaintyDiagnostics": None,
            "preferredPhysicalPeriodDays": None,
            "preferredPhysicalPeriodRelation": None,
            "harmonicCandidates": [],
            "independentCandidates": [],
            "boxCandidates": candidates,
        }
        return DatasetReduction(
            payload={"boxCandidates": candidates},
            status_fields=status_fields,
        )

    def contribution_metrics(
        self,
        work_unit: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        samples = dataset.get("coordinates")
        if not isinstance(samples, list):
            samples = dataset.get("times")
        sample_count = len(samples) if isinstance(samples, list) else 0

        payload = work_unit.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
        frequency_count = int(payload.get("frequencyCount", 0))
        return {
            "workloadID": self.definition.workload_id,
            "sampleCount": sample_count,
            "frequencyCount": frequency_count,
            "durationCount": len(payload.get("durationFractions") or []),
            "phaseBinCount": int(payload.get("phaseBinCount", 0)),
            "sampleFrequencyEvaluations": (
                sample_count * frequency_count
            ),
        }


PLUGIN = BoxPeriodPlugin()
