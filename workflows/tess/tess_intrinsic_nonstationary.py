"""Phenomenological temporal classification of a v20.12 target component.

This module deliberately operates on the persisted spatial coefficients, not on
the aperture photometry.  All thresholds below are target-independent and are
declared here (rather than selected after inspecting a periodogram winner).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


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
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-20:
            return [0.0, 0.0, 0.0]
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[column])]
    return [augmented[row][3] for row in range(3)]


def _fit(times: list[float], values: list[float], frequency: float) -> tuple[float, float, float]:
    rows = [(1.0, math.sin(2.0 * math.pi * frequency * time),
             math.cos(2.0 * math.pi * frequency * time)) for time in times]
    normal = [[sum(row[i] * row[j] for row in rows) for j in range(3)] for i in range(3)]
    rhs = [sum(row[i] * value for row, value in zip(rows, values)) for i in range(3)]
    beta = _solve3(normal, rhs)
    rss = sum((value - sum(coefficient * item for coefficient, item in zip(beta, row))) ** 2
              for row, value in zip(rows, values))
    return rss, math.hypot(beta[1], beta[2]), math.atan2(beta[2], beta[1])


def _scan(times: list[float], values: list[float], reference: float) -> dict[str, Any]:
    low = reference * (1.0 - FREQUENCY_HALF_WIDTH_FRACTION)
    high = reference * (1.0 + FREQUENCY_HALF_WIDTH_FRACTION)
    frequencies = [low + (high - low) * index / (FREQUENCY_GRID_SIZE - 1)
                   for index in range(FREQUENCY_GRID_SIZE)]
    fits = [_fit(times, values, float(frequency)) for frequency in frequencies]
    rss = [item[0] for item in fits]
    winner = min(range(len(rss)), key=rss.__getitem__)
    mean = sum(values) / len(values)
    null_rss = sum((value - mean) ** 2 for value in values)
    power = 1.0 - float(rss[winner]) / null_rss if null_rss > 0 else 0.0
    local = [index for index in range(1, len(rss) - 1) if rss[index] < rss[index - 1] and rss[index] < rss[index + 1]]
    ranked = sorted(local, key=lambda index: float(rss[index]))
    second = next((index for index in ranked if abs(index - winner) >= INDEPENDENT_PEAK_SEPARATION_BINS), None)
    second_power = (1.0 - float(rss[second]) / null_rss) if second is not None and null_rss > 0 else 0.0
    return {"frequency": float(frequencies[winner]), "power": power, "rss": float(rss[winner]),
            "amplitude": fits[winner][1], "phase": fits[winner][2],
            "secondaryPeakPowerRatio": second_power / power if power > 0 else 0.0}


def classify_target_component(*, preparation: dict[str, Any], decomposition: dict[str, Any]) -> dict[str, Any]:
    """Classify only an exactly qualified v20.12 target-dominant boundary."""
    boundary = (
        decomposition.get("recommendedNextTest") == "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"
        and decomposition.get("classification") == "TARGET_RESIDUAL_COMPONENT_DOMINANT"
        and decomposition.get("residualModeOrigin") == "TARGET_DOMINANT"
        and decomposition.get("physicalMechanismResolved") is False
    )
    reasons: list[str] = []
    if not boundary:
        raise RuntimeError("Intrinsic target classification requires the exact unresolved v20.12 target-dominant boundary.")
    target_id = decomposition.get("targetComponentID")
    target_summary = next((item for item in decomposition.get("componentSummaries") or [] if item.get("componentID") == target_id), None)
    if target_id != "target" or not target_summary:
        reasons.append("target component identity is absent or inconsistent")
    elif int(target_summary.get("independentSupportCount") or 0) < MIN_SECTORS:
        reasons.append("target component lacks independent-sector support")

    entries = [item for item in preparation.get("preparedSeries") or []
               if item.get("componentID") == target_id and item.get("componentType") == "TARGET"
               and not item.get("combined") and item.get("sector") is not None]
    evidence: list[dict[str, Any]] = []
    provenance_files: list[dict[str, Any]] = []
    reference = float(preparation.get("referenceFrequency") or 0.0)
    if reference <= 0:
        reasons.append("v20.12 reference frequency is missing")
    for entry in sorted(entries, key=lambda item: int(item["sector"])):
        path = entry.get("coefficientSeriesPath")
        dataset_path = entry.get("datasetPath")
        if not path or not dataset_path or not Path(path).is_file() or not Path(dataset_path).is_file():
            reasons.append(f"sector {entry.get('sector')} component artifact is missing")
            continue
        component = _load(path)
        dataset = _load(dataset_path)
        if component.get("componentID") not in (None, target_id) or (dataset.get("science") or {}).get("componentID") != target_id:
            reasons.append(f"sector {entry.get('sector')} artifact component identity disagrees")
            continue
        if (dataset.get("source") or {}).get("timeReferenceDays") is None:
            reasons.append(f"sector {entry.get('sector')} timing origin is not reconstructible")
            continue
        times = [float(value) for value in component.get("times") or []]
        values = [float(value) for value in component.get("coefficients") or []]
        if len(times) != len(values) or len(times) < MIN_SAMPLES or not all(map(math.isfinite, times + values)):
            reasons.append(f"sector {entry.get('sector')} coefficient series is incomplete")
            continue
        value_mean = sum(values) / len(values)
        scan = _scan(times, [value - value_mean for value in values], reference)
        windows = []
        for window in range(WINDOW_COUNT):
            start, end = len(times) * window // WINDOW_COUNT, len(times) * (window + 1) // WINDOW_COUNT
            indices = range(start, end)
            if len(indices) >= MIN_WINDOW_SAMPLES:
                rss, amplitude, phase = _fit([times[index] for index in indices],
                                             [values[index] for index in indices], scan["frequency"])
                windows.append({"sampleCount": len(indices), "amplitude": amplitude, "phaseRadians": phase, "rss": rss})
        scan.update({"sector": int(entry["sector"]), "sampleCount": len(times), "windows": windows})
        evidence.append(scan)
        provenance_files.extend(({"role": "targetCoefficientSeries", "path": str(Path(path).resolve()), "sha256": _file_hash(path)},
                                 {"role": "targetDataset", "path": str(Path(dataset_path).resolve()), "sha256": _file_hash(dataset_path)}))
    if len(evidence) < MIN_SECTORS:
        reasons.append("fewer than two valid target-component sectors remain")

    classification = "INSUFFICIENT_TARGET_COMPONENT_TEMPORAL_EVIDENCE"
    diagnostics: dict[str, Any] = {}
    if not reasons:
        frequencies = [item["frequency"] for item in evidence]
        amplitudes = [window["amplitude"] for item in evidence for window in item["windows"]]
        frequency_span = (max(frequencies) - min(frequencies)) / (sum(frequencies) / len(frequencies))
        positions = list(range(len(frequencies)))
        position_mean = sum(positions) / len(positions)
        frequency_mean = sum(frequencies) / len(frequencies)
        covariance = sum((x - position_mean) * (y - frequency_mean)
                         for x, y in zip(positions, frequencies))
        denominator = math.sqrt(sum((x - position_mean) ** 2 for x in positions)
                                * sum((y - frequency_mean) ** 2 for y in frequencies))
        drift_correlation = abs(covariance / denominator) if denominator > 0 else 0.0
        amplitude_ratio = max(amplitudes) / max(min(amplitudes), 1e-15)
        multi = any(item["secondaryPeakPowerRatio"] >= SECONDARY_PEAK_POWER_RATIO for item in evidence)
        # BIC comparison: one shared frequency versus one frequency per sector.
        shared_frequency = sum(frequencies) / len(frequencies)
        shared_rss = sum(_fit(_load(entry["coefficientSeriesPath"])["times"],
                              _load(entry["coefficientSeriesPath"])["coefficients"], shared_frequency)[0]
                         for entry in sorted(entries, key=lambda item: int(item["sector"])))
        free_rss = sum(item["rss"] for item in evidence)
        count = sum(item["sampleCount"] for item in evidence)
        bic_shared = count * math.log(max(shared_rss / count, 1e-30)) + (2 * len(evidence) + 1) * math.log(count)
        bic_free = count * math.log(max(free_rss / count, 1e-30)) + (3 * len(evidence)) * math.log(count)
        diagnostics = {"frequencyFractionalSpan": frequency_span, "windowAmplitudeRatio": amplitude_ratio,
                       "stationaryBIC": bic_shared, "sectorFrequencyBIC": bic_free,
                       "sectorFrequencyDeltaBIC": bic_shared - bic_free,
                       "frequencySequenceLinearCorrelation": drift_correlation}
        if multi:
            classification = "MULTIPLE_UNRESOLVED_TARGET_MODES"
        elif amplitude_ratio >= INTERMITTENT_AMPLITUDE_RATIO:
            classification = "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"
        elif (len(evidence) >= MIN_DRIFT_SECTORS
              and drift_correlation >= MIN_DRIFT_LINEAR_CORRELATION
              and frequency_span >= FREQUENCY_DRIFT_FRACTION
              and bic_shared - bic_free >= BIC_DECISIVE_DELTA):
            classification = "SMOOTHLY_FREQUENCY_DRIFTING_TARGET_RESIDUAL_MODE"
        elif amplitude_ratio - 1.0 >= AMPLITUDE_EVOLUTION_FRACTION:
            classification = "AMPLITUDE_PHASE_EVOLVING_TARGET_RESIDUAL"
        else:
            classification = "STATIONARY_COHERENT_TARGET_RESIDUAL_MODE"

    return {"classification": classification, "targetComponentID": target_id,
            "sectorsUsed": [item["sector"] for item in evidence], "temporalModelEvidence": evidence,
            "modelSelectionDiagnostics": diagnostics, "failClosedReasons": reasons,
            "preRegisteredRules": {"minimumSectors": MIN_SECTORS, "minimumSamples": MIN_SAMPLES,
                "windowCount": WINDOW_COUNT, "frequencyGridSize": FREQUENCY_GRID_SIZE,
                "frequencyHalfWidthFraction": FREQUENCY_HALF_WIDTH_FRACTION, "decisiveDeltaBIC": BIC_DECISIVE_DELTA,
                "frequencyDriftFraction": FREQUENCY_DRIFT_FRACTION, "minimumDriftSectors": MIN_DRIFT_SECTORS,
                "minimumDriftLinearCorrelation": MIN_DRIFT_LINEAR_CORRELATION,
                "amplitudeEvolutionFraction": AMPLITUDE_EVOLUTION_FRACTION,
                "intermittentAmplitudeRatio": INTERMITTENT_AMPLITUDE_RATIO, "secondaryPeakPowerRatio": SECONDARY_PEAK_POWER_RATIO},
            "claimLevelChanged": False, "physicalMechanismResolved": False,
            "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP" if not reasons else "RESTORE_TARGET_COMPONENT_PROVENANCE",
            "observable": "v20.12 spatially-decomposed target coefficient series",
            "inputProvenance": {"preparationArtifacts": provenance_files}}
