"""Fixed-clock, coordinator-side replication of narrow orbital events.

This is deliberately a geometry and timing experiment.  Standardized box
depths are detection statistics, not physical transit depths, and are never
converted into companion properties here.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


RESULT_VERSION = "1.0"
DURATION_FRACTIONS = (0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12)
MIN_INDEPENDENT_SUPPORTERS = 3
MIN_SAMPLES = 80
MIN_EVENT_SAMPLES = 5
MIN_EVENT_SNR = 6.0


def physical_interpretation_continuation(physical: dict[str, Any],
                                         morphology: dict[str, Any]) -> bool:
    """Return true only at the exact persisted scientific boundary."""
    try:
        period = float(morphology.get("resolvedPhysicalPeriodDays"))
    except (TypeError, ValueError):
        return False
    return (physical.get("recommendedNextTest") == "INDEPENDENT_BINARY_CONFIRMATION"
            and physical.get("physicalMechanismResolved") is False
            and physical.get("preferredPhotometricHypothesis") == "BINARY_LIKE_DOUBLE_WAVE"
            and morphology.get("physicalCycleResolved") is True
            and math.isfinite(period) and period > 0)


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        return json.load(handle)


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for column in range(len(b)):
        pivot = max(range(column, len(b)), key=lambda row: abs(aug[row][column]))
        if abs(aug[pivot][column]) < 1e-12:
            raise ValueError("singular smooth-waveform fit")
        aug[column], aug[pivot] = aug[pivot], aug[column]
        scale = aug[column][column]
        aug[column] = [value / scale for value in aug[column]]
        for row in range(len(b)):
            if row != column:
                scale = aug[row][column]
                aug[row] = [x - scale * y for x, y in zip(aug[row], aug[column])]
    return [aug[i][-1] for i in range(len(b))]


def _smooth_residual(times: list[float], flux: list[float], period: float,
                     excluded_phase: tuple[float, float] | None = None) -> tuple[list[float], int]:
    rows, used = [], []
    for index, (time, value) in enumerate(zip(times, flux)):
        phase = (time / period) % 1.0
        if excluded_phase is not None:
            distance = abs((phase - excluded_phase[0] + 0.5) % 1.0 - 0.5)
            if distance <= excluded_phase[1] / 2.0:
                continue
        angle = 2.0 * math.pi * time / period
        rows.append(([1.0, math.sin(angle), math.cos(angle),
                      math.sin(2 * angle), math.cos(2 * angle)], value))
        used.append(index)
    if len(rows) < MIN_SAMPLES:
        raise ValueError("insufficient finite samples for smooth-waveform fit")
    normal = [[0.0] * 5 for _ in range(5)]
    rhs = [0.0] * 5
    for row, value in rows:
        for i in range(5):
            rhs[i] += row[i] * value
            for j in range(5):
                normal[i][j] += row[i] * row[j]
    beta = _solve(normal, rhs)
    residual = []
    for time, value in zip(times, flux):
        angle = 2.0 * math.pi * time / period
        row = [1.0, math.sin(angle), math.cos(angle),
               math.sin(2 * angle), math.cos(2 * angle)]
        residual.append(value - sum(x * y for x, y in zip(row, beta)))
    return residual, len(used)


def _mad_sigma(values: list[float]) -> float:
    if not values:
        return 0.0
    median = statistics.median(values)
    return 1.4826 * statistics.median(abs(value - median) for value in values)


def _box_search(times: list[float], residual: list[float], period: float,
                durations: tuple[float, ...] = DURATION_FRACTIONS,
                phase_window: tuple[float, float] | None = None) -> dict[str, Any]:
    # A fixed-duration box optimum changes only when one of its edges crosses
    # an observed phase.  Sorting once and sweeping a duplicated circular
    # phase array therefore evaluates every relevant window in O(D*N), rather
    # than rescanning N samples for each of N possible centers.
    ordered = sorted(((time / period) % 1.0, value)
                     for time, value in zip(times, residual))
    phases = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    n = len(values)
    doubled_phases = phases + [phase + 1.0 for phase in phases]
    doubled_values = values + values
    prefix = [0.0]
    for value in doubled_values:
        prefix.append(prefix[-1] + value)
    baseline = statistics.median(values)
    sigma = _mad_sigma(values)
    best = None
    for duty in durations:
        right = 0
        for left in range(n):
            right = max(right, left)
            edge = doubled_phases[left] + duty
            while right < left + n and doubled_phases[right] <= edge:
                right += 1
            count = right - left
            if count < MIN_EVENT_SAMPLES or count >= n // 2:
                continue
            center = (doubled_phases[left] + duty / 2.0) % 1.0
            if phase_window and abs((center - phase_window[0] + 0.5) % 1.0 - 0.5) > phase_window[1]:
                continue
            inside_mean = (prefix[right] - prefix[left]) / count
            depth = baseline - inside_mean
            uncertainty = sigma / math.sqrt(count) if sigma > 1e-15 else None
            snr = depth / uncertainty if uncertainty else (float("inf") if depth > 1e-12 else 0.0)
            candidate = (snr, depth, -duty, center, duty, count, uncertainty, sigma)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return {"usable": False, "reason": "NO_BOX_WITH_ENOUGH_SAMPLES"}
    snr, depth, _, phase, duty, count, uncertainty, sigma = best
    boundary = duty in (durations[0], durations[-1])
    usable = bool(depth > 0 and snr >= MIN_EVENT_SNR and not boundary)
    return {"usable": usable, "eventPhase": phase, "durationDays": duty * period,
            "dutyCycle": duty, "depthStandardized": depth,
            "depthUncertaintyStandardized": uncertainty, "detectionSnr": snr,
            "eventSampleCount": count, "residualScatterStandardized": sigma,
            "durationBoundaryHit": boundary,
            "durationBoundary": ("MINIMUM" if duty == durations[0] else
                                 "MAXIMUM" if duty == durations[-1] else None),
            "depthIsPhysical": False}


def _sector(dataset: dict[str, Any], period: float, role: str) -> dict[str, Any]:
    try:
        origin = float((dataset.get("source") or {})["originalTimeOriginDays"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen dataset lacks originalTimeOriginDays") from error
    if not math.isfinite(origin):
        raise ValueError("frozen dataset has nonfinite originalTimeOriginDays")
    pairs = []
    for relative_raw, flux_raw in zip(dataset.get("times") or [], dataset.get("flux") or []):
        try:
            relative, flux = float(relative_raw), float(flux_raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(relative) and math.isfinite(flux):
            pairs.append((origin + relative, flux))
    result = {"datasetID": dataset.get("id"), "sector": (dataset.get("source") or {}).get("sector"),
              "role": role, "sampleCount": len(pairs),
              "originalTimeOriginDays": origin,
              "timeReference": "BTJD_RECONSTRUCTED_FROM_FROZEN_RELATIVE_TIME"}
    if len(pairs) < MIN_SAMPLES:
        return {**result, "usable": False, "reason": "INSUFFICIENT_SAMPLES"}
    times, flux = map(list, zip(*pairs))
    residual, initial_count = _smooth_residual(times, flux, period)
    initial = _box_search(times, residual, period)
    if initial.get("eventPhase") is not None:
        # Protect the candidate from the smooth fit, then measure it again.
        residual, refit_count = _smooth_residual(
            times, flux, period, (initial["eventPhase"], initial["dutyCycle"] * 1.5))
        event = _box_search(times, residual, period)
    else:
        refit_count, event = initial_count, initial
    result.update(event)
    result["smoothModel"] = {"harmonics": [1, 2], "initialFitSampleCount": initial_count,
                             "protectedRefitSampleCount": refit_count,
                             "candidateEventMaskedDuringRefit": True}
    if event.get("eventPhase") is not None:
        median_time = statistics.median(times)
        cycle = round(median_time / period - event["eventPhase"])
        result["eventEpoch"] = (cycle + event["eventPhase"]) * period
    result["_times"] = times
    result["_residual"] = residual
    return result


def _ephemeris(events: list[dict[str, Any]], input_period: float) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: item["eventEpoch"])
    anchor = ordered[0]["eventEpoch"]
    cycles = [round((item["eventEpoch"] - anchor) / input_period) for item in ordered]
    if len(set(cycles)) != len(cycles):
        return {"coherent": False, "reason": "INTEGER_CYCLE_ASSIGNMENT_NOT_UNIQUE"}
    def fit(indices: list[int]) -> tuple[float, float]:
        xs = [cycles[i] for i in indices]; ys = [ordered[i]["eventEpoch"] for i in indices]
        xbar, ybar = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        if denom <= 0: raise ValueError("insufficient ephemeris baseline")
        period = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
        return ybar - period * xbar, period
    try:
        epoch, refined = fit(list(range(len(ordered))))
    except ValueError as error:
        return {"coherent": False, "reason": str(error)}
    residuals = [item["eventEpoch"] - (epoch + cycle * refined)
                 for item, cycle in zip(ordered, cycles)]
    phase_residuals = [abs(value) / input_period for value in residuals]
    duration_scales = [item["dutyCycle"] for item in ordered]
    coherent = (refined > 0 and abs(refined - input_period) / input_period <= 0.25 * statistics.median(duration_scales)
                and max(phase_residuals) <= 0.5 * statistics.median(duration_scales))
    loso = []
    for omitted in range(len(ordered)):
        try:
            _, period = fit([i for i in range(len(ordered)) if i != omitted])
            loso.append({"omittedSector": ordered[omitted].get("sector"), "periodDays": period})
        except ValueError:
            loso.append({"omittedSector": ordered[omitted].get("sector"), "periodDays": None})
    assignments = [{"sector": item.get("sector"), "eventEpoch": item["eventEpoch"],
                    "cycleNumber": cycle, "oMinusCDays": residual,
                    "oMinusCInInputPeriod": residual / input_period}
                   for item, cycle, residual in zip(ordered, cycles, residuals)]
    return {"coherent": coherent, "referenceEpoch": epoch, "refinedPeriodDays": refined,
            "cycleAssignments": assignments,
            "rmsOMinusCDays": math.sqrt(statistics.mean(value * value for value in residuals)),
            "maximumAbsoluteOMinusCDays": max(abs(value) for value in residuals),
            "leaveOneSectorOutPeriodSolutions": loso,
            "coherenceScale": "EVENT_DURATION_AND_INPUT_PERIOD"}


def analyze_binary_confirmation(*, primary_dataset_path: str | Path,
                                independent_spec: dict[str, Any], morphology: dict[str, Any],
                                physical_interpretation: dict[str, Any]) -> dict[str, Any]:
    if not physical_interpretation_continuation(physical_interpretation, morphology):
        raise ValueError("authoritative binary-confirmation input gate is not satisfied")
    prepared = [item for item in independent_spec.get("preparedSectors") or []
                if item.get("datasetPath")]
    if len(prepared) < MIN_INDEPENDENT_SUPPORTERS:
        raise ValueError("at least three frozen independent-sector datasets are required")
    period = float(morphology["resolvedPhysicalPeriodDays"])
    results = [_sector(_load(primary_dataset_path), period, "PRIMARY")]
    results += [_sector(_load(item["datasetPath"]), period, "INDEPENDENT") for item in prepared]
    supporters = [item for item in results if item["role"] == "INDEPENDENT" and item.get("usable")]
    ephemeris = _ephemeris(supporters, period) if len(supporters) >= 3 else {
        "coherent": False, "reason": "FEWER_THAN_THREE_INDEPENDENT_SUPPORTERS"}
    supported = len(supporters) >= 3 and ephemeris.get("coherent") is True
    secondary_results = []
    if supported:
        primary_phase = (ephemeris["referenceEpoch"] / period) % 1.0
        median_duty = statistics.median(item["dutyCycle"] for item in supporters)
        durations = tuple(median_duty * factor for factor in (0.5, 0.75, 1.0, 1.25, 1.5))
        for item in results:
            if item.get("eventPhase") is None: continue
            search = _box_search(item["_times"], item["_residual"], period, durations,
                                 ((primary_phase + 0.5) % 1.0, 0.12))
            search.update({"sector": item.get("sector"), "role": item["role"]})
            if item.get("depthStandardized") and search.get("depthStandardized"):
                search["secondaryToPrimaryDepthRatioStandardized"] = (
                    search["depthStandardized"] / item["depthStandardized"])
            secondary_results.append(search)
    secondary_support = [item for item in secondary_results
                         if item["role"] == "INDEPENDENT" and item.get("usable")]
    secondary_phases = [item["eventPhase"] for item in secondary_support]
    secondary_coherent = (len(secondary_support) >= 3 and
        max((abs((phase - secondary_phases[0] + .5) % 1 - .5)
             for phase in secondary_phases), default=1) <= 0.5 * statistics.median(
                 item["dutyCycle"] for item in secondary_support))
    for item in results:
        item.pop("_times", None); item.pop("_residual", None)
    return {"resultVersion": RESULT_VERSION, "experiment": "FIXED_PERIOD_ECLIPSE_GEOMETRY_REPLICATION",
            "physicalPeriodInputDays": period, "sectorResults": results,
            "independentEvidence": {"classification": (
                "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED" if supported else
                "ECLIPSE_LIKE_EVENT_UNRESOLVED"),
                "supportingIndependentSectorCount": len(supporters),
                "minimumSupportingIndependentSectors": MIN_INDEPENDENT_SUPPORTERS,
                "supportingSectors": [item.get("sector") for item in supporters]},
            "linearEphemeris": ephemeris,
            "oppositeConjunctionEvidence": {"classification": (
                "OPPOSITE_CONJUNCTION_EVENT_SUPPORTED" if secondary_coherent else
                "OPPOSITE_CONJUNCTION_EVENT_UNRESOLVED"), "sectorResults": secondary_results,
                "supportingIndependentSectorCount": len(secondary_support),
                "requiredForPrimaryClassification": False,
                "depthNormalization": "SAME_STANDARDIZED_LIGHT_CURVE_NONPHYSICAL"},
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "catalogAnswerKeyUsed": False,
            "claim": "Narrow-event replication tests orbital geometry only; companion nature remains unresolved.",
            "recommendedNextTest": ("ECLIPSE_EVENT_SOURCE_LOCALIZATION" if supported else
                                    "BINARY_ROTATION_EXTERNAL_EVIDENCE")}
