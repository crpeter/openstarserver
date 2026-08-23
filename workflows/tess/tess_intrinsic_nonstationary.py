"""Temporal classification of the frozen v20.12 target component.

Only spatially decomposed coefficient artifacts admitted through their immutable
v20.12 ``ArtifactReference`` records are used.  Local per-sector time axes are
never interpreted as a common clock.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

MIN_SECTORS = 2
MIN_SAMPLES = 80
WINDOW_COUNT = 4
MIN_WINDOW_SAMPLES = 20
FREQUENCY_GRID_SIZE = 2048
FREQUENCY_HALF_WIDTH_FRACTION = 0.20
BIC_DECISIVE_DELTA = 10.0
FREQUENCY_DRIFT_FRACTION = 0.02
MIN_DRIFT_SECTORS = 3
MIN_DRIFT_LINEAR_CORRELATION = 0.80
MAX_COHERENT_PHASE_CIRCULAR_STD_RAD = 0.35
AMPLITUDE_EVOLUTION_FRACTION = 0.35
INTERMITTENT_AMPLITUDE_RATIO = 3.0
SECONDARY_PEAK_POWER_RATIO = 0.70
INDEPENDENT_PEAK_SEPARATION_BINS = 8


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solve3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    values = [row[:] + [item] for row, item in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(values[row][column]))
        if abs(values[pivot][column]) < 1e-20:
            return [0.0, 0.0, 0.0]
        values[column], values[pivot] = values[pivot], values[column]
        scale = values[column][column]
        values[column] = [item / scale for item in values[column]]
        for row in range(3):
            if row == column:
                continue
            scale = values[row][column]
            values[row] = [left - scale * right for left, right in zip(values[row], values[column])]
    return [values[row][3] for row in range(3)]


def _fit(times: list[float], flux: list[float], frequency: float) -> tuple[float, float, float]:
    rows = [(1.0, math.sin(2 * math.pi * frequency * time),
             math.cos(2 * math.pi * frequency * time)) for time in times]
    normal = [[sum(row[i] * row[j] for row in rows) for j in range(3)] for i in range(3)]
    rhs = [sum(row[i] * value for row, value in zip(rows, flux)) for i in range(3)]
    beta = _solve3(normal, rhs)
    rss = sum((value - sum(a * b for a, b in zip(beta, row))) ** 2
              for row, value in zip(rows, flux))
    return rss, math.hypot(beta[1], beta[2]), math.atan2(beta[2], beta[1])


def _scan(times: list[float], flux: list[float], reference: float, *,
          minimum_frequency: float | None = None,
          maximum_frequency: float | None = None) -> dict[str, float]:
    low = (reference * (1 - FREQUENCY_HALF_WIDTH_FRACTION)
           if minimum_frequency is None else minimum_frequency)
    high = (reference * (1 + FREQUENCY_HALF_WIDTH_FRACTION)
            if maximum_frequency is None else maximum_frequency)
    if low <= 0 or high <= low:
        raise ValueError("The pre-registered physical frequency interval is invalid.")
    frequencies = [low + (high - low) * index / (FREQUENCY_GRID_SIZE - 1)
                   for index in range(FREQUENCY_GRID_SIZE)]
    fits = [_fit(times, flux, frequency) for frequency in frequencies]
    winner = min(range(len(fits)), key=lambda index: fits[index][0])
    mean = sum(flux) / len(flux)
    null_rss = sum((value - mean) ** 2 for value in flux)
    power = 1 - fits[winner][0] / null_rss if null_rss > 0 else 0.0
    minima = [index for index in range(1, len(fits) - 1)
              if fits[index][0] < fits[index - 1][0] and fits[index][0] < fits[index + 1][0]]
    secondary = next((index for index in sorted(minima, key=lambda index: fits[index][0])
                      if abs(index - winner) >= INDEPENDENT_PEAK_SEPARATION_BINS), None)
    secondary_power = 1 - fits[secondary][0] / null_rss if secondary is not None and null_rss > 0 else 0.0
    return {"frequency": frequencies[winner], "power": power, "rss": fits[winner][0],
            "amplitude": fits[winner][1], "phase": fits[winner][2],
            "secondaryPeakPowerRatio": secondary_power / power if power > 0 else 0.0,
            "minimumFrequency": low, "maximumFrequency": high,
            "winnerGridIndex": winner,
            "winnerAtSearchBoundary": winner <= 1 or winner >= FREQUENCY_GRID_SIZE - 2}


def _correlation(x: list[float], y: list[float]) -> float:
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - x_mean) ** 2 for a in x)
                            * sum((b - y_mean) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def _authoritative_hashes(references: Iterable[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for reference in references:
        path = reference.path if hasattr(reference, "path") else reference.get("path")
        digest = reference.sha256 if hasattr(reference, "sha256") else reference.get("sha256")
        if path and digest:
            result[str(Path(path).resolve())] = str(digest)
    return result


def classify_target_component(*, preparation: dict[str, Any], decomposition: dict[str, Any],
                              authoritative_artifacts: Iterable[Any] = (),
                              preparation_link_verified: bool = False) -> dict[str, Any]:
    """Classify an exactly qualified target component, failing closed on lineage."""
    if not (decomposition.get("recommendedNextTest") == "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"
            and decomposition.get("classification") == "TARGET_RESIDUAL_COMPONENT_DOMINANT"
            and decomposition.get("residualModeOrigin") == "TARGET_DOMINANT"
            and decomposition.get("physicalMechanismResolved") is False):
        raise RuntimeError("Intrinsic classification requires the exact unresolved v20.12 target boundary.")

    reasons: list[str] = []
    timing_limitations: list[str] = []
    target_id = decomposition.get("targetComponentID")
    summary = next((item for item in decomposition.get("componentSummaries") or []
                    if item.get("componentID") == target_id), None)
    if target_id != "target" or summary is None:
        reasons.append("target component identity is absent or inconsistent")
    elif int(summary.get("independentSupportCount") or 0) < MIN_SECTORS:
        reasons.append("target component lacks independent-sector support")
    if not preparation_link_verified:
        reasons.append("v20.12 interpretation-to-preparation hash linkage is absent or inconsistent")

    frozen = _authoritative_hashes(authoritative_artifacts)
    entries = [item for item in preparation.get("preparedSeries") or []
               if item.get("componentID") == target_id and item.get("componentType") == "TARGET"
               and not item.get("combined") and item.get("sector") is not None]
    evidence: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    reference = float(preparation.get("referenceFrequency") or 0)
    if reference <= 0:
        reasons.append("v20.12 reference frequency is missing")

    for entry in sorted(entries, key=lambda item: str(item["datasetID"])):
        paths = (("targetCoefficientSeries", entry.get("coefficientSeriesPath")),
                 ("targetDataset", entry.get("datasetPath")))
        verified = True
        for role, raw_path in paths:
            resolved = str(Path(raw_path).resolve()) if raw_path else ""
            authoritative = frozen.get(resolved)
            current = _file_hash(resolved) if resolved and Path(resolved).is_file() else None
            provenance.append({"role": role, "path": resolved or None,
                               "authoritativeSha256": authoritative, "verifiedCurrentSha256": current,
                               "verified": authoritative is not None and authoritative == current})
            if authoritative is None:
                reasons.append(f"sector {entry.get('sector')} {role} has no authoritative v20.12 ArtifactReference")
                verified = False
            elif current != authoritative:
                reasons.append(f"sector {entry.get('sector')} {role} failed frozen hash verification")
                verified = False
        if not verified:
            continue
        component, dataset = _load(entry["coefficientSeriesPath"]), _load(entry["datasetPath"])
        if component.get("componentID") != target_id or (dataset.get("science") or {}).get("componentID") != target_id:
            reasons.append(f"sector {entry.get('sector')} artifact component identity disagrees")
            continue
        local_times = [float(value) for value in component.get("times") or []]
        flux = [float(value) for value in component.get("coefficients") or []]
        common = component.get("commonWarpedTimes")
        common_times = [float(value) for value in common] if isinstance(common, list) else None
        absolute = component.get("absoluteTimes")
        absolute_times = [float(value) for value in absolute] if isinstance(absolute, list) else None
        if (len(local_times) != len(flux) or len(local_times) < MIN_SAMPLES
                or not all(map(math.isfinite, local_times + flux))):
            reasons.append(f"sector {entry.get('sector')} coefficient series is incomplete")
            continue
        if (common_times is None or absolute_times is None
                or len(common_times) != len(local_times) or len(absolute_times) != len(local_times)
                or not all(map(math.isfinite, common_times + absolute_times))):
            timing_limitations.append(f"sector {entry.get('sector')} has only independently zeroed local times")
            common_times = None
            absolute_times = None
        elif any(abs((common_times[index] - min(common_times)) - local_times[index]) > 1e-9
                 for index in range(len(local_times))):
            reasons.append(f"sector {entry.get('sector')} common timing disagrees with the frozen local axis")
            continue
        elif component.get("timeReferenceDays") is None or component.get("fractionalFrequencyDriftPerDay") is None:
            reasons.append(f"sector {entry.get('sector')} common timing lacks its frozen warp parameters")
            continue
        elif any(abs(((absolute_times[index] - float(component["timeReferenceDays"]))
                      + 0.5 * float(component["fractionalFrequencyDriftPerDay"])
                      * (absolute_times[index] - float(component["timeReferenceDays"])) ** 2)
                     - common_times[index]) > 1e-9 for index in range(len(local_times))):
            reasons.append(f"sector {entry.get('sector')} absolute-to-warped timing reconstruction disagrees")
            continue
        mean = sum(flux) / len(flux)
        physical_times = absolute_times if absolute_times is not None else local_times
        if absolute_times is not None:
            time_reference = float(component["timeReferenceDays"])
            q = float(component["fractionalFrequencyDriftPerDay"])
            upstream_physical_frequencies = [
                reference * (1.0 + q * (time - time_reference))
                for time in absolute_times
            ]
            expected_minimum = min(upstream_physical_frequencies)
            expected_maximum = max(upstream_physical_frequencies)
            physical_minimum = expected_minimum * (1.0 - FREQUENCY_HALF_WIDTH_FRACTION)
            physical_maximum = expected_maximum * (1.0 + FREQUENCY_HALF_WIDTH_FRACTION)
        else:
            q = None
            expected_minimum = expected_maximum = None
            physical_minimum = physical_maximum = None
        scan = _scan(physical_times, [value - mean for value in flux], reference,
                     minimum_frequency=physical_minimum,
                     maximum_frequency=physical_maximum)
        if scan["winnerAtSearchBoundary"]:
            reasons.append(f"sector {entry.get('sector')} physical-frequency winner is search-boundary truncated")
            continue
        windows = []
        for window in range(WINDOW_COUNT):
            start, end = len(flux) * window // WINDOW_COUNT, len(flux) * (window + 1) // WINDOW_COUNT
            if end - start >= MIN_WINDOW_SAMPLES:
                _, amplitude, phase = _fit(physical_times[start:end], flux[start:end], scan["frequency"])
                windows.append({"sampleCount": end - start, "amplitude": amplitude,
                                "phaseRadians": phase,
                                "measurementCoordinate": ("ORIGINAL_ABSOLUTE_TIME"
                                    if absolute_times is not None else
                                    "SECTOR_LOCAL_WARPED_TIME_NO_CROSS_SECTOR_CLOCK")})
        scan.update({"sector": int(entry["sector"]), "datasetID": entry["datasetID"],
                     "sampleCount": len(flux), "windows": windows,
                     "observationEpochCommonWarpedDays": (sum(common_times) / len(common_times)
                                                           if common_times else None),
                     "observationEpochAbsoluteDays": (sum(absolute_times) / len(absolute_times)
                                                       if absolute_times else None),
                     "frequencyCoordinate": ("ORIGINAL_ABSOLUTE_TIME"
                                             if absolute_times is not None else
                                             "SECTOR_LOCAL_WARPED_TIME_NO_CROSS_SECTOR_CLOCK"),
                     "upstreamFractionalFrequencyDriftPerDay": q,
                     "upstreamExpectedPhysicalFrequencyRange": (
                         {"minimumFrequency": expected_minimum,
                          "maximumFrequency": expected_maximum}
                         if expected_minimum is not None else None),
                     "commonTimes": common_times, "physicalTimes": physical_times, "flux": flux})
        evidence.append(scan)
    if len(evidence) < MIN_SECTORS:
        reasons.append("fewer than two frozen, valid target-component sectors remain")

    classification = "INSUFFICIENT_TARGET_COMPONENT_TEMPORAL_EVIDENCE"
    diagnostics: dict[str, Any] = {"commonTimingAvailable": bool(evidence) and
                                  all(item["commonTimes"] is not None for item in evidence)}
    if not reasons:
        frequencies = [item["frequency"] for item in evidence]
        frequency_span = (max(frequencies) - min(frequencies)) / (sum(frequencies) / len(frequencies))
        amplitudes = [window["amplitude"] for item in evidence for window in item["windows"]]
        amplitude_ratio = max(amplitudes) / max(min(amplitudes), 1e-15)
        multi = any(item["secondaryPeakPowerRatio"] >= SECONDARY_PEAK_POWER_RATIO for item in evidence)
        count = sum(item["sampleCount"] for item in evidence)
        shared_frequency = sum(frequencies) / len(frequencies)
        shared_rss = sum(_fit(item["physicalTimes"], item["flux"], shared_frequency)[0]
                         for item in evidence)
        free_rss = sum(item["rss"] for item in evidence)
        bic_shared = count * math.log(max(shared_rss / count, 1e-30)) + (2 * len(evidence) + 1) * math.log(count)
        bic_free = count * math.log(max(free_rss / count, 1e-30)) + 3 * len(evidence) * math.log(count)
        diagnostics.update({"frequencyFractionalSpan": frequency_span, "windowAmplitudeRatio": amplitude_ratio,
                            "stationaryFrequencyBIC": bic_shared, "sectorFrequencyBIC": bic_free,
                            "sectorFrequencyDeltaBIC": bic_shared - bic_free})
        common_available = diagnostics["commonTimingAvailable"]
        if common_available:
            epochs = [item["observationEpochAbsoluteDays"] for item in evidence]
            correlation = abs(_correlation(epochs, frequencies))
            epoch_mean, frequency_mean = sum(epochs) / len(epochs), sum(frequencies) / len(frequencies)
            slope_denominator = sum((epoch - epoch_mean) ** 2 for epoch in epochs)
            slope = (sum((epoch - epoch_mean) * (frequency - frequency_mean)
                         for epoch, frequency in zip(epochs, frequencies)) / slope_denominator
                     if slope_denominator else 0.0)
            phases = [_fit(item["physicalTimes"], item["flux"], shared_frequency)[2]
                      for item in evidence]
            resultant = math.hypot(sum(math.cos(value) for value in phases),
                                   sum(math.sin(value) for value in phases)) / len(phases)
            phase_std = math.sqrt(max(0.0, -2 * math.log(max(resultant, 1e-15))))
            diagnostics.update({"frequencyDriftPerAbsoluteDay": slope,
                                "frequencyEpochLinearCorrelation": correlation,
                                "crossSectorPhaseCircularStdRadians": phase_std,
                                "phaseCoherenceCoordinate": "ORIGINAL_ABSOLUTE_TIME"})
        else:
            correlation, phase_std = None, None
        if multi:
            classification = "MULTIPLE_UNRESOLVED_TARGET_MODES"
        elif amplitude_ratio >= INTERMITTENT_AMPLITUDE_RATIO:
            classification = "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"
        elif (common_available and len(evidence) >= MIN_DRIFT_SECTORS
              and correlation >= MIN_DRIFT_LINEAR_CORRELATION
              and frequency_span >= FREQUENCY_DRIFT_FRACTION
              and bic_shared - bic_free >= BIC_DECISIVE_DELTA):
            classification = "SMOOTHLY_FREQUENCY_DRIFTING_TARGET_RESIDUAL_MODE"
        elif amplitude_ratio - 1 >= AMPLITUDE_EVOLUTION_FRACTION:
            classification = "AMPLITUDE_EVOLVING_TARGET_RESIDUAL"
        elif common_available and phase_std <= MAX_COHERENT_PHASE_CIRCULAR_STD_RAD:
            classification = "STATIONARY_PHASE_COHERENT_TARGET_RESIDUAL_MODE"
        else:
            classification = "STATIONARY_FREQUENCY_COMPATIBLE_TARGET_RESIDUAL"

    for item in evidence:
        item.pop("commonTimes", None); item.pop("physicalTimes", None); item.pop("flux", None)
    return {"classification": classification, "targetComponentID": target_id,
            "sectorsUsed": [item["sector"] for item in evidence], "temporalModelEvidence": evidence,
            "modelSelectionDiagnostics": diagnostics, "failClosedReasons": reasons,
            "timingLimitations": timing_limitations, "claimLevelChanged": False,
            "physicalMechanismResolved": False,
            "recommendedNextTest": ("RESTORE_TARGET_COMPONENT_PROVENANCE" if reasons
                                    else "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"),
            "observable": "v20.12 spatially-decomposed target coefficient series",
            "inputProvenance": {"preparationArtifacts": provenance},
            "preRegisteredRules": {"minimumSectors": MIN_SECTORS, "minimumSamples": MIN_SAMPLES,
                "windowCount": WINDOW_COUNT, "frequencyGridSize": FREQUENCY_GRID_SIZE,
                "frequencyHalfWidthFraction": FREQUENCY_HALF_WIDTH_FRACTION,
                "physicalFrequencySearchRule": (
                    "[min(referenceFrequency*(1+q*(absoluteTime-timeReference)))*0.8, "
                    "max(referenceFrequency*(1+q*(absoluteTime-timeReference)))*1.2]"),
                "decisiveDeltaBIC": BIC_DECISIVE_DELTA, "frequencyDriftFraction": FREQUENCY_DRIFT_FRACTION,
                "minimumDriftSectors": MIN_DRIFT_SECTORS,
                "minimumDriftLinearCorrelation": MIN_DRIFT_LINEAR_CORRELATION,
                "maximumCoherentPhaseCircularStdRadians": MAX_COHERENT_PHASE_CIRCULAR_STD_RAD,
                "amplitudeEvolutionFraction": AMPLITUDE_EVOLUTION_FRACTION,
                "intermittentAmplitudeRatio": INTERMITTENT_AMPLITUDE_RATIO,
                "secondaryPeakPowerRatio": SECONDARY_PEAK_POWER_RATIO}}
