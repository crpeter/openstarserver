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
RESULT_VERSION = "1.2"
ENTRY_BOUNDARY = "FULL_CHARACTERIZATION_UNRESOLVED_BROAD_VARIABILITY"
TARGETED_BOUNDARY_ENTRY = "FULL_CHARACTERIZATION_NONRECURRENT_BOUNDARY_PERIOD"
UNRELIABLE_PRIMARY_ENTRY = "FULL_CHARACTERIZATION_NONRECURRENT_UNRELIABLE_PRIMARY"
MINIMUM_INDEPENDENT_SECTORS = 2
MINIMUM_SECTOR_SNR = 7.0
MINIMUM_JOINT_SECTOR_SNR = 6.0
MINIMUM_JOINT_RECURRENCE_SNR = (
    MINIMUM_SECTOR_SNR * math.sqrt(1 + MINIMUM_INDEPENDENT_SECTORS)
)
MINIMUM_PERIOD_DAYS = 0.2
MAXIMUM_PERIOD_DAYS = 10.0
PHASE_BIN_COUNT = 200
OVERSAMPLING = 8.0
MAXIMUM_FINE_GRID_SIZE = 2001
MINIMUM_PARITY_CYCLES = 2
MINIMUM_PARITY_SNR_SEPARATION = 4.0
MAXIMUM_ALTERNATE_PARITY_SNR_FRACTION = 0.5
DUTY_CYCLES = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10)


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
    measurements = [
        _box_score(item["times"], item["residual"], item["sigma"], frequency)
        for item in sectors
    ]
    results = _sector_results_for_frequency(sectors, measurements, frequency)
    (
        primary_supported, independent_supporters, ephemeris, supported,
        support_gate,
    ) = _evaluate_recurrence_support(results, period)
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
    requested_fine_step = 1.0 / (full_span * OVERSAMPLING)
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
            combined.append(min(required) + 0.05 * sum(required))
            details.append(per_sector)
        return np.asarray(combined), details

    coarse_scores, _ = evaluate(coarse)
    center = float(coarse[int(np.argmax(coarse_scores))])
    lower = max(minimum, center - coarse_step)
    upper = min(maximum, center + coarse_step)
    requested_count = int(math.ceil((upper - lower) / requested_fine_step)) + 1
    fine_count = min(MAXIMUM_FINE_GRID_SIZE, max(201, requested_count))
    fine = np.linspace(lower, upper, fine_count)
    fine_scores, fine_details = evaluate(fine)
    index = int(np.argmax(fine_scores))
    actual_fine_step = float(fine[1] - fine[0]) if fine.size > 1 else 0.0
    return (
        float(fine[index]), float(fine_scores[index]), fine_details[index],
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


def analyze_blind_transit_search(
    *, primary_dataset_path: str | Path, independent_spec: dict[str, Any],
    morphology: dict[str, Any] | None,
    broad_interpretation: dict[str, Any] | None,
    targeted_interpretation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search a shared blind box period across primary plus frozen sectors."""
    if not blind_transit_search_continuation(
        morphology, independent_spec, broad_interpretation, targeted_interpretation
    ):
        raise ValueError("authoritative blind-transit-search input gate is not satisfied")

    entries = [("PRIMARY", None, primary_dataset_path)] + [
        ("INDEPENDENT", item.get("sector"), item["datasetPath"])
        for item in independent_spec.get("preparedSectors") or []
        if item.get("datasetPath")
    ]
    sectors = []
    for role, frozen_sector, path in entries:
        dataset = _load(path)
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

    primary_dataset = _load(primary_dataset_path)
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

    (
        frequency, combined_score, measurements, coarse_step,
        fine_step, full_span,
    ) = _search_grid(sectors, minimum, maximum)
    raw_period = 1.0 / frequency
    frequency, alias_resolution = _resolve_alternating_cycle_alias(
        sectors, measurements, frequency, minimum
    )
    period = 1.0 / frequency
    (
        sector_results, primary_supported, independent_supporters,
        ephemeris, supported, support_gate,
    ) = _candidate_evidence(sectors, frequency)
    refined = ephemeris.get("refinedPeriodDays") if supported else None
    return {
        "resultVersion": RESULT_VERSION,
        "experiment": "SOFTWARE_BLIND_MULTI_SECTOR_BOX_PERIOD_SEARCH",
        "entryBoundary": (
            ENTRY_BOUNDARY if broad_interpretation is not None
            else (
                UNRELIABLE_PRIMARY_ENTRY
                if (targeted_interpretation or {}).get("primaryReliable") is False
                else TARGETED_BOUNDARY_ENTRY
            )
        ),
        "classification": (
            "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE" if supported
            else "BLIND_TRANSIT_PERIOD_UNRESOLVED"
        ),
        "candidatePeriodDays": refined,
        "coarseCandidatePeriodDays": raw_period,
        "candidateFrequencyPerDay": (1.0 / refined if refined else None),
        "combinedRecurrenceScore": combined_score,
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
            "selectionSupportRule": "PRIMARY_PLUS_TWO_INDEPENDENT_SECTORS",
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
