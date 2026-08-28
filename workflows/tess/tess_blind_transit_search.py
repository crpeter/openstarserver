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
RESULT_VERSION = "1.0"
ENTRY_BOUNDARY = "FULL_CHARACTERIZATION_UNRESOLVED_BROAD_VARIABILITY"
TARGETED_BOUNDARY_ENTRY = "FULL_CHARACTERIZATION_NONRECURRENT_BOUNDARY_PERIOD"
MINIMUM_INDEPENDENT_SECTORS = 2
MINIMUM_SECTOR_SNR = 7.0
MINIMUM_PERIOD_DAYS = 0.2
MAXIMUM_PERIOD_DAYS = 10.0
PHASE_BIN_COUNT = 200
OVERSAMPLING = 8.0
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
    return (
        independent_spec.get("investigationGoal") == "FULL_CHARACTERIZATION"
        and len(prepared) >= MINIMUM_INDEPENDENT_SECTORS
        and (broad_path_spent or targeted_boundary_spent)
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


def _search_grid(sectors: list[dict[str, Any]], minimum: float, maximum: float):
    import numpy as np

    longest_baseline = max(float(item["times"][-1] - item["times"][0]) for item in sectors)
    step = 1.0 / (longest_baseline * OVERSAMPLING)
    coarse = np.arange(minimum, maximum + 0.5 * step, step)

    def evaluate(frequencies):
        combined = []
        details = []
        for frequency in frequencies:
            per_sector = [
                _box_score(item["times"], item["residual"], item["sigma"], float(frequency))
                for item in sectors
            ]
            # The weakest sector is authoritative for recurrence.  A small
            # all-sector term breaks near-ties without allowing one unusually
            # deep sector to overpower a non-detection elsewhere.
            snrs = [result["snr"] for result in per_sector]
            combined.append(min(snrs) + 0.05 * sum(snrs))
            details.append(per_sector)
        return np.asarray(combined), details

    coarse_scores, _ = evaluate(coarse)
    best_frequency = float(coarse[int(np.argmax(coarse_scores))])
    fine = np.linspace(max(minimum, best_frequency - step),
                       min(maximum, best_frequency + step), 201)
    fine_scores, fine_details = evaluate(fine)
    index = int(np.argmax(fine_scores))
    return float(fine[index]), float(fine_scores[index]), fine_details[index], step


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

    frequency, combined_score, measurements, coarse_step = _search_grid(
        sectors, minimum, maximum
    )
    period = 1.0 / frequency
    sector_results = []
    for sector, measurement in zip(sectors, measurements):
        usable = bool(
            measurement["snr"] >= MINIMUM_SECTOR_SNR
            and (sector["times"][-1] - sector["times"][0]) / period >= 2.0
        )
        median_time = float(statistics.median(sector["times"]))
        epoch = (round(median_time / period - measurement["eventPhase"])
                 + measurement["eventPhase"]) * period
        sector_results.append({
            **measurement,
            "role": sector["role"],
            "sector": sector["sector"],
            "datasetID": sector["datasetID"],
            "usable": usable,
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

    primary_supported = any(item["role"] == "PRIMARY" and item["usable"] for item in sector_results)
    independent_supporters = [
        item for item in sector_results if item["role"] == "INDEPENDENT" and item["usable"]
    ]
    timing = [item for item in sector_results if item["usable"]]
    ephemeris = _linear_ephemeris(timing, period) if len(timing) >= 3 else {
        "coherent": False, "reason": "FEWER_THAN_THREE_TOTAL_SECTOR_EVENTS"
    }
    supported = (
        primary_supported
        and len(independent_supporters) >= MINIMUM_INDEPENDENT_SECTORS
        and ephemeris.get("coherent") is True
    )
    refined = ephemeris.get("refinedPeriodDays") if supported else None
    return {
        "resultVersion": RESULT_VERSION,
        "experiment": "SOFTWARE_BLIND_MULTI_SECTOR_BOX_PERIOD_SEARCH",
        "entryBoundary": (
            ENTRY_BOUNDARY if broad_interpretation is not None
            else TARGETED_BOUNDARY_ENTRY
        ),
        "classification": (
            "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE" if supported
            else "BLIND_TRANSIT_PERIOD_UNRESOLVED"
        ),
        "candidatePeriodDays": refined,
        "coarseCandidatePeriodDays": period,
        "candidateFrequencyPerDay": (1.0 / refined if refined else None),
        "combinedRecurrenceScore": combined_score,
        "sectorResults": sector_results,
        "primarySectorSupported": primary_supported,
        "supportingIndependentSectorCount": len(independent_supporters),
        "supportingIndependentSectors": [item.get("sector") for item in independent_supporters],
        "minimumSupportingIndependentSectors": MINIMUM_INDEPENDENT_SECTORS,
        "linearEphemeris": ephemeris,
        "searchGrid": {
            "minimumFrequencyPerDay": minimum,
            "maximumFrequencyPerDay": maximum,
            "coarseFrequencyStepPerDay": coarse_step,
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
