"""Software-blind search for a repeated narrow dimming period.

This fallback is intentionally narrower than companion confirmation.  It uses
only the already-frozen light curves, searches no catalog period, and reports a
transit-like *candidate* even when the available independent-sector count is
below the stronger source-localization and companion-evidence gates.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


HANDLER_ID = "openstar.tess.blind-transit-search.analyze"
RESULT_VERSION = "1.5"
ENTRY_BOUNDARY = "FULL_CHARACTERIZATION_UNRESOLVED_BROAD_VARIABILITY"
TARGETED_BOUNDARY_ENTRY = "FULL_CHARACTERIZATION_NONRECURRENT_BOUNDARY_PERIOD"
UNRELIABLE_PRIMARY_ENTRY = "FULL_CHARACTERIZATION_NONRECURRENT_UNRELIABLE_PRIMARY"
MINIMUM_INDEPENDENT_SECTORS = 2
MINIMUM_SECTOR_SNR = 7.0
MINIMUM_JOINT_SECTOR_SNR = 6.0
MINIMUM_JOINT_RECURRENCE_SNR = (
    MINIMUM_SECTOR_SNR * math.sqrt(1 + MINIMUM_INDEPENDENT_SECTORS)
)
MINIMUM_POOLED_INDEPENDENT_SECTORS = 6
MINIMUM_POOLED_INDEPENDENT_SNR = 10.0
MINIMUM_POOLED_SPLIT_SNR = 5.5
MINIMUM_POOLED_LEAVE_ONE_OUT_SNR = 8.5
MINIMUM_POOLED_CONTRIBUTING_SECTORS = 4
MINIMUM_POOLED_SECTOR_ALIGNED_SNR = 1.5
MAXIMUM_TRANSIT_CLAIM_DUTY_CYCLE = 0.07
MINIMUM_PERIOD_DAYS = 0.2
MAXIMUM_PERIOD_DAYS = 10.0
PHASE_BIN_COUNT = 400
OVERSAMPLING = 8.0
COARSE_FAMILY_COUNT = 12
JOINT_REFINEMENT_FAMILY_COUNT = 4
JOINT_REFINEMENT_HALF_WIDTH_STEPS = 12
MAXIMUM_ITERATIVE_CANDIDATES = 4
TRANSIT_MASK_DURATION_MULTIPLIER = 1.5
DISTINCT_FREQUENCY_TOLERANCE_STEPS = 4.0
MINIMUM_PARITY_CYCLES = 2
MINIMUM_PARITY_SNR_SEPARATION = 4.0
MAXIMUM_ALTERNATE_PARITY_SNR_FRACTION = 0.5
DUTY_CYCLES = (
    0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10,
)


def blind_transit_search_continuation(
    morphology: dict[str, Any] | None,
    independent_spec: dict[str, Any],
    broad_interpretation: dict[str, Any] | None,
    targeted_interpretation: dict[str, Any] | None = None,
) -> bool:
    """Enter after an exact unresolved full-characterization boundary."""
    prepared = [
        item for item in independent_spec.get("preparedSectors") or []
        if item.get("datasetPath")
    ]
    broad_path_spent = (
        broad_interpretation is not None
        and morphology is not None
        and morphology.get("physicalCycleResolved") is False
    )
    targeted_claim = ((targeted_interpretation or {}).get("claimDecision") or {}).get("claim")
    contradiction = (targeted_interpretation or {}).get("contradictionPlan") or {}
    targeted_boundary_spent = (
        targeted_claim == "HUMAN_REVIEW_REQUIRED"
        and (targeted_interpretation or {}).get("primaryBoundaryHit") is True
        and (targeted_interpretation or {}).get("supportingSectorCount") == 0
        and contradiction.get("action") == "STOP"
        and contradiction.get("reason")
        == "insufficient-independent-evidence-for-broad-contradiction-search"
    )
    unreliable_primary_spent = (
        targeted_claim in {"CANDIDATE_PERIOD", "HUMAN_REVIEW_REQUIRED"}
        and (targeted_interpretation or {}).get("primaryReliable") is False
    )
    return (
        independent_spec.get("investigationGoal") == "FULL_CHARACTERIZATION"
        and len(prepared) >= MINIMUM_INDEPENDENT_SECTORS
        and (
            broad_path_spent
            or targeted_boundary_spent
            or unreliable_primary_spent
        )
    )


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite_light_curve(dataset: dict[str, Any]):
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover - production guard
        raise RuntimeError("blind transit search requires NumPy") from error

    source = dataset.get("source") or {}
    try:
        origin = float(source["originalTimeOriginDays"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("frozen dataset lacks originalTimeOriginDays") from error
    if not math.isfinite(origin):
        raise ValueError("frozen dataset has nonfinite originalTimeOriginDays")
    relative = np.asarray(dataset.get("times") or [], dtype=float)
    flux = np.asarray(dataset.get("flux") or [], dtype=float)
    if relative.shape != flux.shape:
        raise ValueError("frozen dataset time and flux arrays differ in length")
    finite = np.isfinite(relative) & np.isfinite(flux)
    times, flux = origin + relative[finite], flux[finite]
    if times.size < 200:
        raise ValueError("frozen dataset has fewer than 200 finite samples")
    order = np.argsort(times)
    return times[order], flux[order], origin


def _detrend(times, flux):
    try:
        import numpy as np
        from scipy.signal import savgol_filter
    except ModuleNotFoundError as error:  # pragma: no cover - production guard
        raise RuntimeError("blind transit search requires NumPy and SciPy") from error

    differences = np.diff(times)
    positive = differences[differences > 0]
    if positive.size == 0:
        raise ValueError("frozen dataset has no increasing time samples")
    cadence = float(np.median(positive))
    # A 0.75-day smooth model removes multi-day stellar variability while
    # leaving ordinary hour-scale TESS transit profiles in the residual.
    window = max(7, int(round(0.75 / cadence)))
    if window % 2 == 0:
        window += 1
    window = min(window, len(flux) - (1 - len(flux) % 2))
    if window < 7:
        raise ValueError("frozen dataset is too short for blind transit detrending")
    trend = savgol_filter(flux, window_length=window, polyorder=2, mode="interp")
    residual = flux - trend
    residual -= np.median(residual)
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if not math.isfinite(sigma) or sigma <= 1e-12:
        raise ValueError("detrended frozen dataset has no finite robust scatter")
    return residual, sigma, cadence, window


def _robust_scatter(values) -> float:
    import numpy as np

    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if not math.isfinite(sigma) or sigma <= 1e-12:
        raise ValueError("masked frozen dataset has no finite robust scatter")
    return sigma


def _box_score(times, residual, sigma: float, frequency: float) -> dict[str, Any]:
    import numpy as np

    period = 1.0 / frequency
    phases = np.remainder(times * frequency, 1.0)
    indices = np.minimum((phases * PHASE_BIN_COUNT).astype(int), PHASE_BIN_COUNT - 1)
    counts = np.bincount(indices, minlength=PHASE_BIN_COUNT).astype(float)
    sums = np.bincount(indices, weights=residual, minlength=PHASE_BIN_COUNT)
    doubled_counts = np.concatenate((counts, counts))
    doubled_sums = np.concatenate((sums, sums))
    count_prefix = np.concatenate(([0.0], np.cumsum(doubled_counts)))
    sum_prefix = np.concatenate(([0.0], np.cumsum(doubled_sums)))
    best: tuple[float, float, int, int, float] | None = None
    for duty in DUTY_CYCLES:
        width = max(1, int(round(duty * PHASE_BIN_COUNT)))
        inside_count = count_prefix[width:width + PHASE_BIN_COUNT] - count_prefix[:PHASE_BIN_COUNT]
        inside_sum = sum_prefix[width:width + PHASE_BIN_COUNT] - sum_prefix[:PHASE_BIN_COUNT]
        valid = inside_count >= 5
        means = np.full(PHASE_BIN_COUNT, np.inf)
        means[valid] = inside_sum[valid] / inside_count[valid]
        start = int(np.argmin(means))
        count = float(inside_count[start])
        depth = -float(means[start])
        snr = depth * math.sqrt(count) / sigma if depth > 0 else 0.0
        candidate = (snr, depth, -width, start, count)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return {"snr": 0.0, "periodDays": period}
    snr, depth, negative_width, start, count = best
    width = -negative_width
    return {
        "snr": snr,
        "periodDays": period,
        "frequencyPerDay": frequency,
        "eventPhase": ((start + width / 2.0) / PHASE_BIN_COUNT) % 1.0,
        "dutyCycle": width / PHASE_BIN_COUNT,
        "durationDays": width / PHASE_BIN_COUNT * period,
        "depthStandardized": depth,
        "eventSampleCount": int(count),
        "depthIsPhysical": False,
    }


def _pooled_box_score(
    sectors: list[dict[str, Any]], frequency: float
) -> dict[str, Any]:
    """Measure a shared clock after normalizing every sector by its scatter."""
    import numpy as np

    if not sectors:
        return {"snr": 0.0, "periodDays": 1.0 / frequency}
    times = np.concatenate([item["times"] for item in sectors])
    residual = np.concatenate([
        item["residual"] / item["sigma"] for item in sectors
    ])
    return _box_score(times, residual, 1.0, frequency)


def _phase_box_arrays(sector: dict[str, Any], frequency: float, width: int):
    """Return fixed-phase box counts, depths, and SNRs for one sector."""
    import numpy as np

    phases = np.remainder(sector["times"] * frequency, 1.0)
    indices = np.minimum(
        (phases * PHASE_BIN_COUNT).astype(int), PHASE_BIN_COUNT - 1
    )
    counts = np.bincount(indices, minlength=PHASE_BIN_COUNT).astype(float)
    sums = np.bincount(
        indices, weights=sector["residual"], minlength=PHASE_BIN_COUNT
    )
    doubled_counts = np.concatenate((counts, counts))
    doubled_sums = np.concatenate((sums, sums))
    count_prefix = np.concatenate(([0.0], np.cumsum(doubled_counts)))
    sum_prefix = np.concatenate(([0.0], np.cumsum(doubled_sums)))
    inside_count = (
        count_prefix[width:width + PHASE_BIN_COUNT]
        - count_prefix[:PHASE_BIN_COUNT]
    )
    inside_sum = (
        sum_prefix[width:width + PHASE_BIN_COUNT]
        - sum_prefix[:PHASE_BIN_COUNT]
    )
    valid = inside_count >= 5
    depth = np.zeros(PHASE_BIN_COUNT, dtype=float)
    depth[valid] = -inside_sum[valid] / inside_count[valid]
    snr = np.zeros(PHASE_BIN_COUNT, dtype=float)
    positive = valid & (depth > 0)
    snr[positive] = (
        depth[positive] * np.sqrt(inside_count[positive]) / sector["sigma"]
    )
    return inside_count, depth, snr


def _joint_box_score(
    sectors: list[dict[str, Any]], frequency: float
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """Rank one shared period, epoch, and duration across normalized sectors.

    A loud event in one sector cannot choose the epoch.  The strict objective
    requires the primary and two independent sectors at the same phase.  With
    enough sectors, a second robust objective requires evidence in both the
    early and late halves and caps every sector's ranking contribution.  Claim
    thresholds remain the responsibility of the existing recurrence gates.
    """
    import numpy as np

    if not sectors:
        return 0.0, [], {
            "method": "SHARED_PERIOD_EPOCH_DURATION_BOX_SEARCH",
            "frequencyPerDay": frequency,
        }
    if sectors[0]["role"] != "PRIMARY":
        raise ValueError("joint transit search requires the primary sector first")

    period = 1.0 / frequency
    independent_indices = [
        index for index, item in enumerate(sectors)
        if item["role"] == "INDEPENDENT"
    ]
    ordered_independent = sorted(
        independent_indices,
        key=lambda index: float(statistics.median(sectors[index]["times"])),
    )
    midpoint = len(ordered_independent) // 2
    early_indices = ordered_independent[:midpoint]
    late_indices = ordered_independent[midpoint:]
    best = None

    for duty in DUTY_CYCLES:
        width = max(1, int(round(duty * PHASE_BIN_COUNT)))
        profiles = [
            _phase_box_arrays(item, frequency, width) for item in sectors
        ]
        snrs = np.stack([item[2] for item in profiles])
        primary_snr = snrs[0]

        if len(independent_indices) >= MINIMUM_INDEPENDENT_SECTORS:
            independent_snrs = np.sort(snrs[independent_indices], axis=0)
            top = independent_snrs[-1]
            second = independent_snrs[-MINIMUM_INDEPENDENT_SECTORS]
            strict_score = (
                np.minimum(primary_snr, second)
                + 0.05 * (primary_snr + top + second)
            )
        else:
            strict_score = np.zeros(PHASE_BIN_COUNT, dtype=float)

        pooled_score = np.zeros(PHASE_BIN_COUNT, dtype=float)
        if len(independent_indices) >= MINIMUM_POOLED_INDEPENDENT_SECTORS:
            capped = np.minimum(snrs, MINIMUM_SECTOR_SNR)

            def robust(indices):
                if not indices:
                    return np.zeros(PHASE_BIN_COUNT, dtype=float)
                return np.sqrt(np.sum(capped[indices] ** 2, axis=0))

            all_independent = robust(independent_indices)
            early = robust(early_indices)
            late = robust(late_indices)
            pooled_score = (
                np.minimum(np.minimum(primary_snr, early), late)
                + 0.02 * (primary_snr + all_independent + early + late)
            )

        objective = np.maximum(strict_score, pooled_score)
        start = int(np.argmax(objective))
        candidate = (
            float(objective[start]),
            float(primary_snr[start]),
            int(np.count_nonzero(snrs[independent_indices, start] > 0.0)),
            -width,
            -start,
        )
        if best is None or candidate > best[0]:
            best = (candidate, width, start, profiles, strict_score, pooled_score)

    if best is None:  # pragma: no cover - DUTY_CYCLES is a nonempty contract
        return 0.0, [], {
            "method": "SHARED_PERIOD_EPOCH_DURATION_BOX_SEARCH",
            "frequencyPerDay": frequency,
        }

    candidate, width, start, profiles, strict_score, pooled_score = best
    event_phase = ((start + width / 2.0) / PHASE_BIN_COUNT) % 1.0
    duty_cycle = width / PHASE_BIN_COUNT
    measurements = []
    for sector, (counts, depths, snrs) in zip(sectors, profiles):
        measurements.append({
            "snr": float(snrs[start]),
            "periodDays": period,
            "frequencyPerDay": frequency,
            "eventPhase": event_phase,
            "dutyCycle": duty_cycle,
            "durationDays": duty_cycle * period,
            "depthStandardized": float(depths[start]),
            "eventSampleCount": int(counts[start]),
            "depthIsPhysical": False,
        })
    audit = {
        "method": "SHARED_PERIOD_EPOCH_DURATION_BOX_SEARCH",
        "frequencyPerDay": frequency,
        "periodDays": period,
        "eventPhase": event_phase,
        "dutyCycle": duty_cycle,
        "durationDays": duty_cycle * period,
        "objectiveScore": candidate[0],
        "strictObjectiveScore": float(strict_score[start]),
        "pooledSplitObjectiveScore": float(pooled_score[start]),
        "perSectorNormalization": "ROBUST_SCATTER",
        "pooledRankingSnrCapPerSector": MINIMUM_SECTOR_SNR,
        "maximumClaimDutyCycle": MAXIMUM_TRANSIT_CLAIM_DUTY_CYCLE,
    }
    return candidate[0], measurements, audit


def _phase_distance(first: float, second: float) -> float:
    difference = abs(first - second) % 1.0
    return min(difference, 1.0 - difference)


def _aligned_sector_measurement(
    sector: dict[str, Any], frequency: float, event_phase: float,
    duty_cycle: float,
) -> dict[str, Any]:
    """Score one sector at a fixed pooled phase without refitting its event."""
    import numpy as np

    phases = np.remainder(sector["times"] * frequency, 1.0)
    distance = np.abs(phases - event_phase)
    distance = np.minimum(distance, 1.0 - distance)
    selected = distance <= duty_cycle / 2.0
    count = int(np.count_nonzero(selected))
    standardized = sector["residual"] / sector["sigma"]
    depth = -float(np.mean(standardized[selected])) if count else 0.0
    snr = depth * math.sqrt(count) if depth > 0 else 0.0
    return {
        "role": sector["role"],
        "sector": sector["sector"],
        "datasetID": sector["datasetID"],
        "snr": snr,
        "depthStandardized": depth,
        "eventSampleCount": count,
    }


def _pooled_recurrence_evidence(
    sectors: list[dict[str, Any]], sector_results: list[dict[str, Any]],
    frequency: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], bool]:
    """Fail closed unless weak events recur across time and survive jackknifing."""
    independent = [item for item in sectors if item["role"] == "INDEPENDENT"]
    period = 1.0 / frequency
    unavailable = {
        "mode": "NOT_SATISFIED",
        "minimumIndependentSectorCount": MINIMUM_POOLED_INDEPENDENT_SECTORS,
        "independentSectorCount": len(independent),
        "minimumPooledIndependentSnr": MINIMUM_POOLED_INDEPENDENT_SNR,
        "minimumSplitSnr": MINIMUM_POOLED_SPLIT_SNR,
        "minimumLeaveOneOutSnr": MINIMUM_POOLED_LEAVE_ONE_OUT_SNR,
        "minimumContributingSectorCount": MINIMUM_POOLED_CONTRIBUTING_SECTORS,
        "catalogAnswerKeyUsed": False,
    }
    if len(independent) < MINIMUM_POOLED_INDEPENDENT_SECTORS:
        unavailable["reason"] = "INSUFFICIENT_INDEPENDENT_SECTORS_FOR_POOLING"
        return [], {
            "coherent": False,
            "reason": "POOLED_RECURRENCE_GATE_NOT_SATISFIED",
        }, unavailable, False

    ordered = sorted(
        independent, key=lambda item: float(statistics.median(item["times"]))
    )
    midpoint = len(ordered) // 2
    early, late = ordered[:midpoint], ordered[midpoint:]
    pooled = _pooled_box_score(independent, frequency)
    early_measurement = _pooled_box_score(early, frequency)
    late_measurement = _pooled_box_score(late, frequency)
    primary = next(
        item for item in sector_results if item["role"] == "PRIMARY"
    )
    measurements = [primary, pooled, early_measurement, late_measurement]
    phase_tolerance = 0.75 * max(
        float(item.get("dutyCycle") or 0.0) for item in measurements
    )
    maximum_phase_offset = max(
        _phase_distance(
            float(primary["eventPhase"]), float(item["eventPhase"])
        )
        for item in measurements[1:]
    )
    aligned = [
        _aligned_sector_measurement(
            item, frequency, float(pooled["eventPhase"]),
            float(pooled["dutyCycle"]),
        )
        for item in independent
    ]
    contributing = [
        item for item in aligned
        if item["snr"] >= MINIMUM_POOLED_SECTOR_ALIGNED_SNR
    ]
    early_ids = {item["datasetID"] for item in early}
    late_ids = {item["datasetID"] for item in late}
    contributing_early = [
        item for item in contributing if item["datasetID"] in early_ids
    ]
    contributing_late = [
        item for item in contributing if item["datasetID"] in late_ids
    ]
    leave_one_out = [
        {
            "omittedSector": omitted["sector"],
            "snr": _pooled_box_score(
                [item for item in independent if item is not omitted], frequency
            )["snr"],
        }
        for omitted in independent
    ]
    minimum_leave_one_out = min(item["snr"] for item in leave_one_out)
    combined_snr = math.sqrt(primary["snr"] ** 2 + pooled["snr"] ** 2)
    supported = bool(
        primary.get("usable") is True
        and pooled["snr"] >= MINIMUM_POOLED_INDEPENDENT_SNR
        and early_measurement["snr"] >= MINIMUM_POOLED_SPLIT_SNR
        and late_measurement["snr"] >= MINIMUM_POOLED_SPLIT_SNR
        and minimum_leave_one_out >= MINIMUM_POOLED_LEAVE_ONE_OUT_SNR
        and len(contributing) >= MINIMUM_POOLED_CONTRIBUTING_SECTORS
        and contributing_early
        and contributing_late
        and maximum_phase_offset <= phase_tolerance
        and combined_snr >= MINIMUM_JOINT_RECURRENCE_SNR
    )
    gate = {
        **unavailable,
        "mode": "POOLED_SPLIT_RECURRENCE" if supported else "NOT_SATISFIED",
        "pooledIndependentSnr": pooled["snr"],
        "earlyIndependentSnr": early_measurement["snr"],
        "lateIndependentSnr": late_measurement["snr"],
        "combinedPrimaryAndPooledSnr": combined_snr,
        "minimumCombinedRecurrenceSnr": MINIMUM_JOINT_RECURRENCE_SNR,
        "minimumObservedLeaveOneOutSnr": minimum_leave_one_out,
        "maximumPhaseOffset": maximum_phase_offset,
        "phaseCoherenceTolerance": phase_tolerance,
        "pooledEventPhase": pooled["eventPhase"],
        "pooledDutyCycle": pooled["dutyCycle"],
        "contributingIndependentSectors": [
            item["sector"] for item in contributing
        ],
        "contributingEarlySectors": [item["sector"] for item in contributing_early],
        "contributingLateSectors": [item["sector"] for item in contributing_late],
        "alignedSectorEvidence": aligned,
        "leaveOneSectorOut": leave_one_out,
    }
    if not supported:
        return [], {
            "coherent": False,
            "reason": "POOLED_RECURRENCE_GATE_NOT_SATISFIED",
        }, gate, False
    contributing_ids = {item["datasetID"] for item in contributing}
    supporting_results = [
        item for item in sector_results
        if item["role"] == "INDEPENDENT" and item["datasetID"] in contributing_ids
    ]
    ephemeris = {
        "coherent": True,
        "method": "SHARED_PHASE_EARLY_LATE_POOLED_RECURRENCE",
        "refinedPeriodDays": period,
        "referenceEpoch": pooled["eventPhase"] * period,
        "maximumPhaseOffset": maximum_phase_offset,
        "coherenceTolerancePhase": phase_tolerance,
    }
    return supporting_results, ephemeris, gate, True


def _sector_results_for_frequency(
    sectors: list[dict[str, Any]], measurements: list[dict[str, Any]], frequency: float
) -> list[dict[str, Any]]:
    period = 1.0 / frequency
    results = []
    for sector, measurement in zip(sectors, measurements):
        cycle_coverage = (sector["times"][-1] - sector["times"][0]) / period
        usable = bool(
            measurement["snr"] >= MINIMUM_SECTOR_SNR
            and cycle_coverage >= 2.0
            and measurement["dutyCycle"] <= MAXIMUM_TRANSIT_CLAIM_DUTY_CYCLE
        )
        median_time = float(statistics.median(sector["times"]))
        epoch = (round(median_time / period - measurement["eventPhase"])
                 + measurement["eventPhase"]) * period
        results.append({
            **measurement,
            "role": sector["role"],
            "sector": sector["sector"],
            "datasetID": sector["datasetID"],
            "usable": usable,
            "cycleCoverage": cycle_coverage,
            "eventEpoch": epoch,
            "sampleCount": len(sector["times"]),
            "originalTimeOriginDays": sector["origin"],
            "detrending": {
                "method": "SAVITZKY_GOLAY_LOCAL_BASELINE",
                "windowDays": 0.75,
                "windowSamples": sector["detrendWindowSamples"],
                "cadenceDays": sector["cadence"],
            },
        })
    return results


def _alternating_cycle_evidence(
    sector: dict[str, Any], measurement: dict[str, Any], period: float
) -> dict[str, Any]:
    """Measure whether a candidate clock contains events on only one parity."""
    import numpy as np

    event_phase = float(measurement["eventPhase"])
    duty_cycle = float(measurement["dutyCycle"])
    cycle_coordinates = sector["times"] / period - event_phase
    cycle_numbers = np.rint(cycle_coordinates).astype(np.int64)
    distance = np.abs(cycle_coordinates - cycle_numbers)
    inside = distance <= duty_cycle / 2.0
    parities = []
    for parity in (0, 1):
        selected = inside & (np.remainder(cycle_numbers, 2) == parity)
        count = int(np.count_nonzero(selected))
        observed_cycles = int(np.unique(cycle_numbers[selected]).size)
        depth = -float(np.mean(sector["residual"][selected])) if count else 0.0
        snr = depth * math.sqrt(count) / sector["sigma"] if depth > 0 else 0.0
        parities.append({
            "parity": parity,
            "snr": snr,
            "depthStandardized": depth,
            "eventSampleCount": count,
            "observedCycleCount": observed_cycles,
        })

    dominant, alternate = sorted(parities, key=lambda item: item["snr"], reverse=True)
    sufficiently_observed = all(
        item["observedCycleCount"] >= MINIMUM_PARITY_CYCLES for item in parities
    )
    decisive = bool(
        sufficiently_observed
        and dominant["snr"] >= MINIMUM_SECTOR_SNR
        and dominant["snr"] - alternate["snr"] >= MINIMUM_PARITY_SNR_SEPARATION
        and alternate["snr"]
        <= MAXIMUM_ALTERNATE_PARITY_SNR_FRACTION * dominant["snr"]
    )
    return {
        "role": sector["role"],
        "sector": sector["sector"],
        "datasetID": sector["datasetID"],
        "decisiveAlternatingEvents": decisive,
        "dominantParity": dominant["parity"],
        "dominantParitySnr": dominant["snr"],
        "alternateParitySnr": alternate["snr"],
        "parities": parities,
    }


def _candidate_evidence(
    sectors: list[dict[str, Any]], frequency: float
) -> tuple[
    list[dict[str, Any]], bool, list[dict[str, Any]], dict[str, Any], bool,
    dict[str, Any],
]:
    period = 1.0 / frequency
    _, measurements, _ = _joint_box_score(sectors, frequency)
    results = _sector_results_for_frequency(sectors, measurements, frequency)
    (
        primary_supported, independent_supporters, ephemeris, supported,
        support_gate,
    ) = _evaluate_recurrence_support(results, period)
    if not supported:
        (
            pooled_supporters, pooled_ephemeris, pooled_gate, pooled_supported,
        ) = _pooled_recurrence_evidence(sectors, results, frequency)
        support_gate["pooledRecurrence"] = pooled_gate
        if pooled_supported:
            independent_supporters = pooled_supporters
            ephemeris = pooled_ephemeris
            supported = True
            support_gate = pooled_gate
    return (
        results, primary_supported, independent_supporters, ephemeris, supported,
        support_gate,
    )


def _evaluate_recurrence_support(
    results: list[dict[str, Any]], period: float
) -> tuple[bool, list[dict[str, Any]], dict[str, Any], bool, dict[str, Any]]:
    """Apply strict-sector evidence first, then a conservative joint gate."""
    primary = next((item for item in results if item["role"] == "PRIMARY"), None)
    primary_supported = bool(primary and primary["usable"])
    strict_independent = [
        item for item in results if item["role"] == "INDEPENDENT" and item["usable"]
    ]
    strict_timing = [item for item in results if item["usable"]]
    strict_ephemeris = (
        _linear_ephemeris(strict_timing, period) if len(strict_timing) >= 3 else {
            "coherent": False, "reason": "FEWER_THAN_THREE_TOTAL_SECTOR_EVENTS"
        }
    )
    strict_supported = bool(
        primary_supported
        and len(strict_independent) >= MINIMUM_INDEPENDENT_SECTORS
        and strict_ephemeris.get("coherent") is True
    )
    if strict_supported:
        return primary_supported, strict_independent, strict_ephemeris, True, {
            "mode": "STRICT_INDIVIDUAL_SECTORS",
            "strictSectorSnrThreshold": MINIMUM_SECTOR_SNR,
            "jointIndependentSectorSnrFloor": MINIMUM_JOINT_SECTOR_SNR,
            "minimumJointRecurrenceSnr": MINIMUM_JOINT_RECURRENCE_SNR,
            "linearEphemerisCoherent": True,
        }

    joint_candidates = sorted(
        (
            item for item in results
            if item["role"] == "INDEPENDENT"
            and item.get("cycleCoverage", 0.0) >= 2.0
            and item["snr"] >= MINIMUM_JOINT_SECTOR_SNR
        ),
        key=lambda item: item["snr"],
        reverse=True,
    )
    selected_joint = joint_candidates[:MINIMUM_INDEPENDENT_SECTORS]
    joint_recurrence_snr = (
        math.sqrt(
            primary["snr"] ** 2
            + sum(item["snr"] ** 2 for item in selected_joint)
        )
        if primary_supported and len(selected_joint) == MINIMUM_INDEPENDENT_SECTORS
        else None
    )
    joint_ephemeris = (
        _linear_ephemeris([primary, *selected_joint], period)
        if joint_recurrence_snr is not None else {
            "coherent": False, "reason": "FEWER_THAN_THREE_TOTAL_SECTOR_EVENTS"
        }
    )
    # The joint path is only for near-threshold evidence.  If two independent
    # sectors already passed the strict gate but disagreed in event timing, do
    # not discard contradictory strict evidence by selecting a smaller subset.
    joint_supported = bool(
        primary_supported
        and len(strict_independent) < MINIMUM_INDEPENDENT_SECTORS
        and joint_recurrence_snr is not None
        and joint_recurrence_snr >= MINIMUM_JOINT_RECURRENCE_SNR
        and joint_ephemeris.get("coherent") is True
    )
    support_gate = {
        "mode": (
            "JOINT_NEAR_THRESHOLD_SECTORS" if joint_supported
            else "NOT_SATISFIED"
        ),
        "strictSectorSnrThreshold": MINIMUM_SECTOR_SNR,
        "jointIndependentSectorSnrFloor": MINIMUM_JOINT_SECTOR_SNR,
        "minimumJointRecurrenceSnr": MINIMUM_JOINT_RECURRENCE_SNR,
        "jointRecurrenceSnr": joint_recurrence_snr,
        "selectedIndependentSectors": [item.get("sector") for item in selected_joint],
        "selectedIndependentSectorSnrs": [item["snr"] for item in selected_joint],
        "linearEphemerisCoherent": joint_ephemeris.get("coherent") is True,
    }
    if joint_supported:
        return primary_supported, selected_joint, joint_ephemeris, True, support_gate
    return (
        primary_supported, strict_independent, strict_ephemeris, False, support_gate,
    )


def _resolve_alternating_cycle_alias(
    sectors: list[dict[str, Any]], measurements: list[dict[str, Any]],
    frequency: float, minimum_frequency: float,
) -> tuple[float, dict[str, Any]]:
    """Promote P to 2P only when alternating-cycle and recurrence gates agree."""
    period = 1.0 / frequency
    doubled_period = 2.0 * period
    audit = [
        _alternating_cycle_evidence(sector, measurement, period)
        for sector, measurement in zip(sectors, measurements)
    ]
    primary_decisive = any(
        item["role"] == "PRIMARY" and item["decisiveAlternatingEvents"]
        for item in audit
    )
    independent_decisive = [
        item for item in audit
        if item["role"] == "INDEPENDENT" and item["decisiveAlternatingEvents"]
    ]
    eligible = bool(
        doubled_period <= MAXIMUM_PERIOD_DAYS
        and frequency / 2.0 >= minimum_frequency
        and primary_decisive
        and len(independent_decisive) >= MINIMUM_INDEPENDENT_SECTORS
    )
    result = {
        "testedBasePeriodDays": period,
        "testedDoublePeriodDays": doubled_period,
        "decision": "RETAIN_BASE_PERIOD",
        "reason": "ALTERNATING_CYCLE_EVIDENCE_NOT_DECISIVE",
        "minimumParityCycles": MINIMUM_PARITY_CYCLES,
        "minimumParitySnrSeparation": MINIMUM_PARITY_SNR_SEPARATION,
        "maximumAlternateParitySnrFraction": MAXIMUM_ALTERNATE_PARITY_SNR_FRACTION,
        "sectorEvidence": audit,
    }
    if not eligible:
        if doubled_period > MAXIMUM_PERIOD_DAYS or frequency / 2.0 < minimum_frequency:
            result["reason"] = "DOUBLE_PERIOD_OUTSIDE_SEARCH_RANGE"
        return frequency, result

    _, _, doubled_independent, doubled_ephemeris, doubled_supported, doubled_gate = (
        _candidate_evidence(sectors, frequency / 2.0)
    )
    result["doublePeriodValidation"] = {
        "supported": doubled_supported,
        "supportingIndependentSectorCount": len(doubled_independent),
        "linearEphemeris": doubled_ephemeris,
        "recurrenceSupportGate": doubled_gate,
    }
    if not doubled_supported:
        result["reason"] = "DOUBLE_PERIOD_FAILED_RECURRENCE_OR_EPHEMERIS_GATE"
        return frequency, result

    result["decision"] = "PROMOTE_DOUBLE_PERIOD"
    result["reason"] = "TRANSITS_OCCUR_ON_ONLY_ONE_ALTERNATING_CYCLE_PARITY"
    return frequency / 2.0, result


def _search_grid(sectors: list[dict[str, Any]], minimum: float, maximum: float):
    import numpy as np

    longest_baseline = max(float(item["times"][-1] - item["times"][0]) for item in sectors)
    full_span = max(float(item["times"][-1]) for item in sectors) - min(
        float(item["times"][0]) for item in sectors
    )
    coarse_step = 1.0 / (longest_baseline * OVERSAMPLING)
    requested_fine_step = min(DUTY_CYCLES) / (full_span * OVERSAMPLING)
    coarse = np.arange(minimum, maximum + 0.5 * coarse_step, coarse_step)

    def evaluate(frequencies):
        combined = []
        details = []
        for frequency in frequencies:
            per_sector = [
                _box_score(item["times"], item["residual"], item["sigma"], float(frequency))
                for item in sectors
            ]
            # Candidate selection must match the claim gate: the primary plus
            # two independent sectors.  Requiring every available sector to
            # score well lets one noisy or transit-free sector hide a real
            # recurrence that already satisfies the evidence contract.
            primary_snr = per_sector[0]["snr"]
            independent_snrs = sorted(
                (result["snr"] for result in per_sector[1:]), reverse=True
            )
            required = [primary_snr, *independent_snrs[:MINIMUM_INDEPENDENT_SECTORS]]
            if len(sectors) - 1 >= MINIMUM_POOLED_INDEPENDENT_SECTORS:
                ordered = sorted(
                    sectors[1:],
                    key=lambda item: float(statistics.median(item["times"])),
                )
                midpoint = len(ordered) // 2
                pooled_snr = _pooled_box_score(
                    sectors[1:], float(frequency)
                )["snr"]
                early_snr = _pooled_box_score(
                    ordered[:midpoint], float(frequency)
                )["snr"]
                late_snr = _pooled_box_score(
                    ordered[midpoint:], float(frequency)
                )["snr"]
                pooled_score = (
                    min(primary_snr, pooled_snr)
                    + 0.05 * (primary_snr + pooled_snr)
                    + 0.02 * (early_snr + late_snr)
                )
                strict_score = min(required) + 0.05 * sum(required)
                combined.append(max(strict_score, pooled_score))
            else:
                combined.append(min(required) + 0.05 * sum(required))
            details.append(per_sector)
        return np.asarray(combined), details

    coarse_scores, coarse_details = evaluate(coarse)
    ranked_indices = np.argsort(coarse_scores)[::-1]
    family_indices = []
    for raw_index in ranked_indices:
        index = int(raw_index)
        if any(abs(index - selected) <= 1 for selected in family_indices):
            continue
        family_indices.append(index)
        if len(family_indices) >= COARSE_FAMILY_COUNT:
            break

    hypotheses = []
    for index in family_indices:
        center = float(coarse[index])
        hypotheses.append(center)
        measurements = coarse_details[index]
        event_epochs = []
        for sector, measurement in zip(sectors, measurements):
            median_time = float(statistics.median(sector["times"]))
            phase = float(measurement["eventPhase"])
            epoch = (round(median_time * center - phase) + phase) / center
            event_epochs.append(epoch)
        for first in range(len(event_epochs)):
            for second in range(first + 1, len(event_epochs)):
                separation = event_epochs[second] - event_epochs[first]
                cycles = round(separation * center)
                if cycles == 0:
                    continue
                frequency = cycles / separation
                if (
                    minimum <= frequency <= maximum
                    and abs(frequency - center) <= coarse_step
                ):
                    hypotheses.append(float(frequency))

    resolution = max(requested_fine_step, np.finfo(float).eps)
    unique = {}
    for frequency in hypotheses:
        unique.setdefault(round(frequency / resolution), frequency)

    ranked_joint = []
    for frequency in unique.values():
        score, measurements, _ = _joint_box_score(sectors, frequency)
        ranked_joint.append((score, frequency, measurements))
    ranked_joint.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    refined = []
    for _, center, _ in ranked_joint[:JOINT_REFINEMENT_FAMILY_COUNT]:
        for offset in range(
            -JOINT_REFINEMENT_HALF_WIDTH_STEPS,
            JOINT_REFINEMENT_HALF_WIDTH_STEPS + 1,
        ):
            frequency = center + offset * requested_fine_step
            if minimum <= frequency <= maximum:
                score, measurements, _ = _joint_box_score(sectors, frequency)
                refined.append((score, frequency, measurements))
    if not refined:
        refined = ranked_joint
    score, frequency, measurements = max(
        refined, key=lambda item: (item[0], -item[1])
    )
    actual_fine_step = requested_fine_step
    return (
        float(frequency), float(score), measurements,
        coarse_step, actual_fine_step, full_span,
    )


def _linear_ephemeris(sector_results: list[dict[str, Any]], input_period: float) -> dict[str, Any]:
    import numpy as np

    epochs = [float(item["eventEpoch"]) for item in sector_results]
    anchor = min(epochs)
    cycles = np.asarray([round((epoch - anchor) / input_period) for epoch in epochs], dtype=float)
    if len(set(int(value) for value in cycles)) != len(cycles):
        return {"coherent": False, "reason": "INTEGER_CYCLE_ASSIGNMENT_NOT_UNIQUE"}
    design = np.column_stack((np.ones(len(cycles)), cycles))
    epoch, period = np.linalg.lstsq(design, np.asarray(epochs), rcond=None)[0]
    predictions = epoch + period * cycles
    residuals = np.asarray(epochs) - predictions
    tolerance = 0.75 * statistics.median(float(item["durationDays"]) for item in sector_results)
    coherent = bool(period > 0 and np.max(np.abs(residuals)) <= tolerance)
    return {
        "coherent": coherent,
        "referenceEpoch": float(epoch),
        "refinedPeriodDays": float(period),
        "rmsOMinusCDays": float(np.sqrt(np.mean(residuals ** 2))),
        "maximumAbsoluteOMinusCDays": float(np.max(np.abs(residuals))),
        "coherenceToleranceDays": tolerance,
        "cycleAssignments": [
            {"sector": item.get("sector"), "cycleNumber": int(cycle),
             "eventEpoch": measured, "oMinusCDays": float(residual)}
            for item, cycle, measured, residual in zip(sector_results, cycles, epochs, residuals)
        ],
    }


def _entry_boundary(
    broad_interpretation: dict[str, Any] | None,
    targeted_interpretation: dict[str, Any] | None,
) -> str:
    if broad_interpretation is not None:
        return ENTRY_BOUNDARY
    if (targeted_interpretation or {}).get("primaryReliable") is False:
        return UNRELIABLE_PRIMARY_ENTRY
    return TARGETED_BOUNDARY_ENTRY


def _prepare_analysis_sectors(
    primary_dataset_path: str | Path,
    independent_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_dataset = _load(primary_dataset_path)
    entries = [("PRIMARY", None, primary_dataset)] + [
        ("INDEPENDENT", item.get("sector"), _load(item["datasetPath"]))
        for item in independent_spec.get("preparedSectors") or []
        if item.get("datasetPath")
    ]
    sectors = []
    for role, frozen_sector, dataset in entries:
        times, flux, origin = _finite_light_curve(dataset)
        residual, sigma, cadence, window = _detrend(times, flux)
        source = dataset.get("source") or {}
        sectors.append({
            "role": role,
            "sector": source.get("sector", frozen_sector),
            "datasetID": dataset.get("id"),
            "times": times,
            "residual": residual,
            "sigma": sigma,
            "origin": origin,
            "cadence": cadence,
            "detrendWindowSamples": window,
        })
    return sectors, primary_dataset


def _frequency_bounds(primary_dataset: dict[str, Any]) -> tuple[float, float]:
    search = primary_dataset.get("frequencySearch") or {}
    try:
        source_minimum = float(search.get("minimumFrequency", 1.0 / MAXIMUM_PERIOD_DAYS))
        source_maximum = float(search.get("maximumFrequency", 1.0 / MINIMUM_PERIOD_DAYS))
    except (TypeError, ValueError) as error:
        raise ValueError("primary frozen frequency search is malformed") from error
    minimum = max(1.0 / MAXIMUM_PERIOD_DAYS, source_minimum)
    maximum = min(1.0 / MINIMUM_PERIOD_DAYS, source_maximum)
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum < maximum):
        raise ValueError("primary frozen frequency search does not overlap the transit search")
    return minimum, maximum


def _analyze_prepared_sectors(
    sectors: list[dict[str, Any]], minimum: float, maximum: float,
    entry_boundary: str,
) -> dict[str, Any]:
    """Run one authoritative shared-clock search over prepared sectors."""

    (
        frequency, combined_score, measurements, coarse_step,
        fine_step, full_span,
    ) = _search_grid(sectors, minimum, maximum)
    raw_period = 1.0 / frequency
    frequency, alias_resolution = _resolve_alternating_cycle_alias(
        sectors, measurements, frequency, minimum
    )
    _, _, joint_search = _joint_box_score(sectors, frequency)
    (
        sector_results, primary_supported, independent_supporters,
        ephemeris, supported, support_gate,
    ) = _candidate_evidence(sectors, frequency)
    refined = ephemeris.get("refinedPeriodDays") if supported else None
    return {
        "resultVersion": RESULT_VERSION,
        "experiment": "SOFTWARE_BLIND_MULTI_SECTOR_BOX_PERIOD_SEARCH",
        "entryBoundary": entry_boundary,
        "classification": (
            "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE" if supported
            else "BLIND_TRANSIT_PERIOD_UNRESOLVED"
        ),
        "candidatePeriodDays": refined,
        "coarseCandidatePeriodDays": raw_period,
        "candidateFrequencyPerDay": (1.0 / refined if refined else None),
        "combinedRecurrenceScore": combined_score,
        "jointTransitSearch": joint_search,
        "alternatingCycleAliasResolution": alias_resolution,
        "sectorResults": sector_results,
        "primarySectorSupported": primary_supported,
        "supportingIndependentSectorCount": len(independent_supporters),
        "supportingIndependentSectors": [item.get("sector") for item in independent_supporters],
        "minimumSupportingIndependentSectors": MINIMUM_INDEPENDENT_SECTORS,
        "linearEphemeris": ephemeris,
        "recurrenceSupportGate": support_gate,
        "searchGrid": {
            "minimumFrequencyPerDay": minimum,
            "maximumFrequencyPerDay": maximum,
            "coarseFrequencyStepPerDay": coarse_step,
            "fullObservationSpanDays": full_span,
            "fineFrequencyStepPerDay": fine_step,
            "frequencyResolutionBasis": (
                "MINIMUM_TRANSIT_DUTY_CYCLE_OVER_FULL_OBSERVATION_SPAN"
            ),
            "selectionSupportRule": (
                "SHARED_PERIOD_EPOCH_DURATION_PRIMARY_PLUS_TWO_INDEPENDENT_OR_POOLED_SPLIT"
                if len(sectors) - 1 >= MINIMUM_POOLED_INDEPENDENT_SECTORS
                else "SHARED_PERIOD_EPOCH_DURATION_PRIMARY_PLUS_TWO_INDEPENDENT"
            ),
            "phaseBinCount": PHASE_BIN_COUNT,
            "dutyCycles": list(DUTY_CYCLES),
        },
        "physicalCycleResolved": False,
        "companionNatureResolved": False,
        "catalogAnswerKeyUsed": False,
        "claimDecision": {
            "claim": "CANDIDATE_PERIOD" if supported else "HUMAN_REVIEW_REQUIRED",
            "rationale": ([
                "A software-blind box search recovered the same narrow dimming clock in the primary and at least two frozen independent TESS sectors.",
                "The event period is a transit-like candidate; source attribution and companion nature remain unresolved.",
            ] if supported else [
                "The full-characterization variability path did not yield a coherently repeated blind narrow-event period.",
            ]),
        },
        "recommendedNextTest": (
            "ADDITIONAL_INDEPENDENT_SECTOR_TRANSIT_CONFIRMATION" if supported
            else "HUMAN_SCIENTIFIC_REVIEW"
        ),
    }


def analyze_blind_transit_search(
    *, primary_dataset_path: str | Path, independent_spec: dict[str, Any],
    morphology: dict[str, Any] | None,
    broad_interpretation: dict[str, Any] | None,
    targeted_interpretation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search one shared blind box period across frozen sectors."""
    if not blind_transit_search_continuation(
        morphology, independent_spec, broad_interpretation, targeted_interpretation
    ):
        raise ValueError("authoritative blind-transit-search input gate is not satisfied")
    sectors, primary_dataset = _prepare_analysis_sectors(
        primary_dataset_path, independent_spec
    )
    minimum, maximum = _frequency_bounds(primary_dataset)
    return _analyze_prepared_sectors(
        sectors,
        minimum,
        maximum,
        _entry_boundary(broad_interpretation, targeted_interpretation),
    )


def _candidate_signal(result: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    """Persist the complete evidence needed to audit one accepted clock."""
    return {
        "candidateIndex": candidate_index,
        "classification": result.get("classification"),
        "candidatePeriodDays": result.get("candidatePeriodDays"),
        "coarseCandidatePeriodDays": result.get("coarseCandidatePeriodDays"),
        "candidateFrequencyPerDay": result.get("candidateFrequencyPerDay"),
        "combinedRecurrenceScore": result.get("combinedRecurrenceScore"),
        "jointTransitSearch": result.get("jointTransitSearch"),
        "alternatingCycleAliasResolution": result.get(
            "alternatingCycleAliasResolution"
        ),
        "sectorResults": result.get("sectorResults"),
        "primarySectorSupported": result.get("primarySectorSupported"),
        "supportingIndependentSectorCount": result.get(
            "supportingIndependentSectorCount"
        ),
        "supportingIndependentSectors": result.get(
            "supportingIndependentSectors"
        ),
        "linearEphemeris": result.get("linearEphemeris"),
        "recurrenceSupportGate": result.get("recurrenceSupportGate"),
        "searchGrid": result.get("searchGrid"),
        "physicalCycleResolved": False,
        "companionNatureResolved": False,
        "catalogAnswerKeyUsed": False,
    }


def _mask_candidate_clock(
    sectors: list[dict[str, Any]], result: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove one accepted clock's predicted windows from every sector."""
    import numpy as np

    ephemeris = result.get("linearEphemeris") or {}
    joint = result.get("jointTransitSearch") or {}
    period = float(
        ephemeris.get("refinedPeriodDays") or result["candidatePeriodDays"]
    )
    frequency = float(joint.get("frequencyPerDay") or 1.0 / period)
    reference_epoch = ephemeris.get("referenceEpoch")
    if reference_epoch is None:
        reference_epoch = float(joint["eventPhase"]) / frequency
    reference_epoch = float(reference_epoch)
    duration = float(joint.get("durationDays") or 0.0)
    if duration <= 0.0:
        durations = [
            float(item.get("durationDays") or 0.0)
            for item in result.get("sectorResults") or []
            if float(item.get("durationDays") or 0.0) > 0.0
        ]
        duration = statistics.median(durations) if durations else 0.0
    if not all(math.isfinite(value) and value > 0.0 for value in (period, duration)):
        raise ValueError("accepted blind transit candidate has no finite masking clock")
    if not math.isfinite(reference_epoch):
        raise ValueError("accepted blind transit candidate has no finite masking epoch")

    mask_duration = TRANSIT_MASK_DURATION_MULTIPLIER * duration
    masked = []
    sector_audit = []
    for sector in sectors:
        distance = np.abs(
            np.remainder(
                sector["times"] - reference_epoch + period / 2.0, period
            ) - period / 2.0
        )
        keep = distance > mask_duration / 2.0
        remaining = int(np.count_nonzero(keep))
        removed = int(len(keep) - remaining)
        if remaining < 200:
            raise ValueError(
                "iterative transit masking leaves fewer than 200 samples"
            )
        residual = sector["residual"][keep]
        residual = residual - np.median(residual)
        updated = dict(sector)
        updated["times"] = sector["times"][keep]
        updated["residual"] = residual
        updated["sigma"] = _robust_scatter(residual)
        masked.append(updated)
        sector_audit.append({
            "role": sector["role"],
            "sector": sector["sector"],
            "datasetID": sector["datasetID"],
            "inputSampleCount": len(keep),
            "removedSampleCount": removed,
            "remainingSampleCount": remaining,
        })
    return masked, {
        "method": "REMOVE_PREDICTED_SHARED_CLOCK_TRANSIT_WINDOWS",
        "periodDays": period,
        "referenceEpoch": reference_epoch,
        "detectedDurationDays": duration,
        "maskDurationMultiplier": TRANSIT_MASK_DURATION_MULTIPLIER,
        "maskDurationDays": mask_duration,
        "perSector": sector_audit,
        "totalRemovedSampleCount": sum(
            item["removedSampleCount"] for item in sector_audit
        ),
        "detrendingRecomputed": False,
        "robustScatterRecomputedAfterMasking": True,
        "catalogAnswerKeyUsed": False,
    }


def _search_frequency(result: dict[str, Any]) -> float | None:
    joint = result.get("jointTransitSearch") or {}
    value = joint.get("frequencyPerDay") or result.get("candidateFrequencyPerDay")
    try:
        frequency = float(value)
    except (TypeError, ValueError):
        return None
    return frequency if math.isfinite(frequency) and frequency > 0.0 else None


def _distinct_frequency_family(
    candidate: dict[str, Any], accepted: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    frequency = _search_frequency(candidate)
    if frequency is None:
        return False, {"distinct": False, "reason": "MALFORMED_CANDIDATE_FREQUENCY"}
    fine_step = float((candidate.get("searchGrid") or {}).get(
        "fineFrequencyStepPerDay", 0.0
    ))
    comparisons = []
    harmonic_multipliers = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    for prior in accepted:
        prior_frequency = _search_frequency(prior)
        if prior_frequency is None:
            continue
        prior_step = float((prior.get("searchGrid") or {}).get(
            "fineFrequencyStepPerDay", 0.0
        ))
        tolerance = DISTINCT_FREQUENCY_TOLERANCE_STEPS * max(
            fine_step, prior_step, 1e-12
        )
        for multiplier in harmonic_multipliers:
            distance = abs(frequency - multiplier * prior_frequency)
            comparison = {
                "priorCandidateIndex": prior.get("candidateIndex"),
                "harmonicMultiplier": multiplier,
                "absoluteFrequencyDistancePerDay": distance,
                "tolerancePerDay": tolerance,
            }
            comparisons.append(comparison)
            if distance <= tolerance:
                return False, {
                    "distinct": False,
                    "reason": "DUPLICATE_OR_EXACT_HARMONIC_FREQUENCY_FAMILY",
                    "matchingComparison": comparison,
                }
    return True, {
        "distinct": True,
        "reason": "SEPARATE_FREQUENCY_FAMILY",
        "comparisons": comparisons,
    }


def analyze_iterative_blind_transit_search(
    *, primary_dataset_path: str | Path, independent_spec: dict[str, Any],
    morphology: dict[str, Any] | None,
    broad_interpretation: dict[str, Any] | None,
    targeted_interpretation: dict[str, Any] | None = None,
    initial_result: dict[str, Any] | None = None,
    maximum_candidates: int = MAXIMUM_ITERATIVE_CANDIDATES,
) -> dict[str, Any]:
    """Repeatedly mask accepted shared clocks and search the residuals."""
    if not isinstance(maximum_candidates, int) or not (
        1 <= maximum_candidates <= MAXIMUM_ITERATIVE_CANDIDATES
    ):
        raise ValueError("maximum_candidates is outside the iterative search contract")
    if not blind_transit_search_continuation(
        morphology, independent_spec, broad_interpretation, targeted_interpretation
    ):
        raise ValueError("authoritative blind-transit-search input gate is not satisfied")

    sectors, primary_dataset = _prepare_analysis_sectors(
        primary_dataset_path, independent_spec
    )
    minimum, maximum = _frequency_bounds(primary_dataset)
    entry_boundary = _entry_boundary(broad_interpretation, targeted_interpretation)
    first = dict(initial_result) if initial_result is not None else (
        _analyze_prepared_sectors(sectors, minimum, maximum, entry_boundary)
    )
    iterations = [{
        "iteration": 1,
        "accepted": first.get("classification")
        == "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
        "classification": first.get("classification"),
        "candidatePeriodDays": first.get("candidatePeriodDays"),
        "input": "ORIGINAL_DETRENDED_FROZEN_SECTORS",
        "catalogAnswerKeyUsed": False,
    }]
    accepted_results = []
    accepted_signals = []
    termination_reason = "INITIAL_SIGNAL_UNRESOLVED"
    if first.get("classification") == "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE":
        accepted_results.append(first)
        accepted_signals.append(_candidate_signal(first, 1))
        working = sectors
        current = first
        while len(accepted_results) < maximum_candidates:
            try:
                working, mask_audit = _mask_candidate_clock(working, current)
            except ValueError as error:
                termination_reason = "MASKED_DATA_INSUFFICIENT"
                iterations.append({
                    "iteration": len(accepted_results) + 1,
                    "accepted": False,
                    "classification": "BLIND_TRANSIT_PERIOD_UNRESOLVED",
                    "reason": str(error),
                    "catalogAnswerKeyUsed": False,
                })
                break
            accepted_signals[-1]["residualSearchMask"] = mask_audit
            residual_result = _analyze_prepared_sectors(
                working, minimum, maximum, entry_boundary
            )
            iteration = {
                "iteration": len(accepted_results) + 1,
                "accepted": False,
                "classification": residual_result.get("classification"),
                "candidatePeriodDays": residual_result.get("candidatePeriodDays"),
                "coarseCandidatePeriodDays": residual_result.get(
                    "coarseCandidatePeriodDays"
                ),
                "maskingAppliedBeforeSearch": mask_audit,
                "catalogAnswerKeyUsed": False,
            }
            if residual_result.get("classification") != (
                "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE"
            ):
                termination_reason = "NEXT_RESIDUAL_SIGNAL_UNRESOLVED"
                iterations.append(iteration)
                break
            distinct, distinct_audit = _distinct_frequency_family(
                residual_result, accepted_signals
            )
            iteration["frequencyFamilySeparation"] = distinct_audit
            if not distinct:
                termination_reason = "NEXT_SIGNAL_NOT_DISTINCT"
                iterations.append(iteration)
                break
            iteration["accepted"] = True
            iterations.append(iteration)
            accepted_results.append(residual_result)
            accepted_signals.append(
                _candidate_signal(residual_result, len(accepted_results))
            )
            current = residual_result
        else:
            termination_reason = "MAXIMUM_CANDIDATE_COUNT_REACHED"

    enriched = dict(first)
    enriched["candidateSignals"] = accepted_signals
    enriched["iterativeSearch"] = {
        "method": "REPEATED_SHARED_CLOCK_SEARCH_AFTER_TRANSIT_WINDOW_MASKING",
        "maximumCandidateCount": maximum_candidates,
        "acceptedCandidateCount": len(accepted_signals),
        "terminationReason": termination_reason,
        "maskDurationMultiplier": TRANSIT_MASK_DURATION_MULTIPLIER,
        "exactHarmonicToleranceFineSteps": DISTINCT_FREQUENCY_TOLERANCE_STEPS,
        "iterations": iterations,
        "catalogAnswerKeyUsed": False,
    }
    if len(accepted_signals) > 1:
        claim = dict(enriched.get("claimDecision") or {})
        rationale = list(claim.get("rationale") or [])
        rationale.append(
            "Additional distinct transit-like clocks remained independently "
            "replicated after masking every previously accepted event window."
        )
        claim["rationale"] = rationale
        enriched["claimDecision"] = claim
    return enriched
