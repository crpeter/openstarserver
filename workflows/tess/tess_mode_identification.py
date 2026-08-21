"""Conservative harmonic-versus-independent-mode tests for frozen TESS data.

This module is server-side science.  It deliberately contains no worker or
TESS-download logic: callers pass the immutable dataset paths recorded by the
investigation and the linear fits are deterministic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable



MIN_BIC_IMPROVEMENT = 10.0
MIN_INDEPENDENT_SECTOR_SUPPORT = 3
GENERIC_REFINEMENT_WORKLOAD_ID = "openstar.lomb-scargle.v1"


def _load_dataset(path: str | Path) -> tuple[list[float], list[float]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    raw_times = dataset.get("times") or []
    raw_flux = dataset.get("flux") or []
    if len(raw_times) != len(raw_flux):
        raise RuntimeError("Frozen dataset times/flux lengths do not match.")
    pairs = [(float(time), float(value)) for time, value in zip(raw_times, raw_flux)
             if math.isfinite(float(time)) and math.isfinite(float(value))]
    if len(pairs) < 16:
        raise RuntimeError("Frozen dataset has too few finite samples for mode identification.")
    origin = (dataset.get("source") or {}).get("originalTimeOriginDays")
    if origin is None:
        origin = (dataset.get("source") or {}).get("timeOriginDays")
    offset = float(origin) if origin is not None and math.isfinite(float(origin)) else 0.0
    return [item[0] + offset for item in pairs], [item[1] for item in pairs]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(len(vector)):
        pivot = max(range(column, len(vector)), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise RuntimeError("Mode-identification design is singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(len(vector)):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * other for value, other in zip(augmented[row], augmented[column])]
    return [row[-1] for row in augmented]


def _fit(datasets: list[tuple[list[float], list[float]]], frequencies: Iterable[float], *, frequency_parameters: int = 0) -> dict[str, Any]:
    frequencies = tuple(float(value) for value in frequencies)
    rss = 0.0
    samples = 0
    for times, flux in datasets:
        rows = []
        for time in times:
            row = [1.0]
            for frequency in frequencies:
                phase = 2.0 * math.pi * frequency * time
                row.extend((math.sin(phase), math.cos(phase)))
            rows.append(row)
        size = len(rows[0])
        normal = [[sum(row[i] * row[j] for row in rows) for j in range(size)] for i in range(size)]
        rhs = [sum(row[i] * value for row, value in zip(rows, flux)) for i in range(size)]
        coefficients = _solve(normal, rhs)
        rss += sum((value - sum(coefficient * basis for coefficient, basis in zip(coefficients, row))) ** 2
                   for row, value in zip(rows, flux))
        samples += len(times)
    parameters = len(datasets) * (1 + 2 * len(frequencies)) + frequency_parameters
    safe_rss = max(rss, 1e-300)
    bic = samples * math.log(safe_rss / samples) + parameters * math.log(samples)
    return {"bic": float(bic), "rss": rss, "sampleCount": samples, "parameterCount": parameters}


def identify_residual_mode(*, dataset_paths: Iterable[str | Path], established_period_days: float,
                           residual_period_days: float, independent_sectors: Iterable[int]) -> dict[str, Any]:
    """Compare an established family, higher harmonics, and a free residual.

    Frequency commensurability uses the Rayleigh resolution of the actual
    frozen-photometry time span.  The free frequency is deterministically
    refined within one Rayleigh element around the measured candidate.
    """
    if established_period_days <= 0 or residual_period_days <= 0:
        raise ValueError("Periods must be positive.")
    paths = tuple(str(Path(path).expanduser().resolve()) for path in dataset_paths)
    if not paths:
        raise ValueError("At least one frozen dataset is required.")
    datasets = [_load_dataset(path) for path in paths]
    all_times = [value for item in datasets for value in item[0]]
    baseline = max(all_times) - min(all_times)
    if baseline <= 0:
        raise RuntimeError("Frozen photometry has no positive time baseline.")
    resolution = 1.0 / baseline
    family_frequency = 1.0 / established_period_days
    measured_frequency = 1.0 / residual_period_days
    ratio = measured_frequency / family_frequency
    harmonic_order = max(3, int(round(ratio)))
    harmonic_frequency = harmonic_order * family_frequency
    separation = abs(measured_frequency - harmonic_frequency)
    commensurate = separation <= resolution

    family_frequencies = [family_frequency, 2.0 * family_frequency]
    extended_frequencies = family_frequencies + [harmonic_frequency]
    model_a = _fit(datasets, family_frequencies)
    model_b = _fit(datasets, extended_frequencies)
    grid = [measured_frequency - resolution + 2.0 * resolution * index / 100 for index in range(101)]
    positive_grid = [value for value in grid if value > 0]
    candidates = [(float(frequency), _fit(datasets, family_frequencies + [float(frequency)], frequency_parameters=1))
                  for frequency in positive_grid]
    refined_frequency, model_c = min(candidates, key=lambda item: (item[1]["bic"], item[0]))

    delta_bic_a_b = model_a["bic"] - model_b["bic"]
    delta_bic_a_c = model_a["bic"] - model_c["bic"]
    delta_bic_b_c = model_b["bic"] - model_c["bic"]
    sectors = sorted({int(value) for value in independent_sectors})
    enough_support = len(sectors) >= MIN_INDEPENDENT_SECTOR_SUPPORT
    independent_survives = (not commensurate and enough_support
                            and delta_bic_a_c >= MIN_BIC_IMPROVEMENT
                            and delta_bic_b_c >= MIN_BIC_IMPROVEMENT)
    if independent_survives:
        classification = "INDEPENDENT_STABLE_MODE"
        recommended = "RESIDUAL_MODE_PIXEL_LOCALIZATION"
    elif commensurate and delta_bic_a_b >= MIN_BIC_IMPROVEMENT and delta_bic_b_c <= MIN_BIC_IMPROVEMENT:
        classification = "HIGHER_ORDER_HARMONIC_STRUCTURE"
        recommended = "DYNAMIC_HARMONIC_MODELING"
    elif max(delta_bic_a_b, delta_bic_a_c) < MIN_BIC_IMPROVEMENT:
        classification = "NO_COMPELLING_RESIDUAL_MODE"
        recommended = "BINARY_ROTATION_EXTERNAL_EVIDENCE"
    else:
        classification = "AMBIGUOUS_HARMONIC_OR_MODE"
        recommended = "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"

    return {
        "classification": classification,
        "establishedPeriodFamily": {"referencePeriodDays": established_period_days,
                                      "referenceFrequencyCyclesPerDay": family_frequency,
                                      "modeledHarmonicOrders": [1, 2, harmonic_order]},
        "residualCandidate": {"measuredPeriodDays": residual_period_days,
                              "measuredFrequencyCyclesPerDay": measured_frequency,
                              "refinedPeriodDays": 1.0 / refined_frequency,
                              "refinedFrequencyCyclesPerDay": refined_frequency},
        "harmonicRelation": {"testedOrder": harmonic_order, "harmonicFrequencyCyclesPerDay": harmonic_frequency,
                             "absoluteFrequencySeparation": separation, "frequencyResolutionCyclesPerDay": resolution,
                             "baselineDays": baseline, "commensurateWithinResolution": commensurate},
        "modelComparison": {"criterion": "BIC", "conservativeThreshold": MIN_BIC_IMPROVEMENT,
                            "models": {"establishedFamily": model_a, "extendedHigherHarmonics": model_b,
                                       "familyPlusIndependentFreeFrequency": model_c},
                            "bicImprovementExtendedOverFamily": delta_bic_a_b,
                            "bicImprovementIndependentOverFamily": delta_bic_a_c,
                            "bicImprovementIndependentOverExtended": delta_bic_b_c},
        "independentSectorSupport": {"sectors": sectors, "count": len(sectors),
                                     "requiredCount": MIN_INDEPENDENT_SECTOR_SUPPORT,
                                     "sufficient": enough_support},
        "independentModeEvidenceSurvived": independent_survives,
        "modeCandidate": ({"periodDays": 1.0 / refined_frequency, "frequencyCyclesPerDay": refined_frequency,
                           "supportingSectors": sectors} if independent_survives else None),
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommended,
        "dataReuse": {"frozenDatasetPaths": list(paths), "downloadPerformed": False},
        "frequencyRefinement": {"execution": "PYTHON_SERVER", "genericDistributedWorkloadIfNeeded": GENERIC_REFINEMENT_WORKLOAD_ID},
    }
