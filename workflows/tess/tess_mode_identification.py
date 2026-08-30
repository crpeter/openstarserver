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
MULTIMODE_MODE_EVIDENCE_LINEAGE = (
    "MULTIMODE_RECURRENT_SECONDARY_FREQUENCY"
)


def validated_multimode_mode_evidence(
    summary: dict[str, Any] | None,
    *,
    physical_period_days: float,
    target_supporting_sectors: Iterable[int],
    iteration_count: int | None = None,
) -> dict[str, Any] | None:
    """Validate the persisted v20.7 recurrent-mode contract fail closed."""
    value = summary or {}
    recurrent = value.get("bestRecurrentSecondaryMode") or {}
    members = recurrent.get("members") or []
    clusters = value.get("frequencyClusters") or []
    accepted = value.get("acceptedResidualModes") or []
    try:
        established_period = float(physical_period_days)
        reported_period = float(value.get("physicalPeriodDays"))
        reported_frequency = float(value.get("physicalFrequency"))
        first_harmonic = float(value.get("firstHarmonicFrequency"))
        residual_period = float(recurrent.get("medianPeriodDays"))
        residual_frequency = float(recurrent.get("medianFrequency"))
        support = sorted(int(item) for item in (
            recurrent.get("independentSectors") or []))
        target_support = {int(item) for item in target_supporting_sectors}
        reported_support_count = int(
            recurrent.get("independentSectorCount"))
        minimum_support = int(
            value.get("minimumRecurrentIndependentSectorCount"))
        reported_iterations = int(value.get("iterationsCompleted"))
        cluster_tolerance = float(value.get("clusterRelativeTolerance"))
        accepted_support = sorted(int(item) for item in (
            value.get("independentSectorsWithAcceptedResidualModes") or []))
    except (TypeError, ValueError):
        return None

    finite_positive = (
        established_period,
        reported_period,
        reported_frequency,
        first_harmonic,
        residual_period,
        residual_frequency,
        cluster_tolerance,
    )
    if not all(math.isfinite(item) and item > 0 for item in finite_positive):
        return None
    if not (
        value.get("classification") == "MULTI_MODE_RECURRENT"
        and value.get("physicalMechanismResolved") is False
        and value.get("claimLevelChanged") is False
        and value.get("recommendedNextTest")
        == "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
        and math.isclose(
            reported_period, established_period,
            rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            reported_frequency, 1.0 / established_period,
            rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            first_harmonic, 2.0 / established_period,
            rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            residual_period, 1.0 / residual_frequency,
            rel_tol=1e-9, abs_tol=1e-12)
        and reported_iterations >= 1
        and reported_iterations <= 3
        and (iteration_count is None
             or reported_iterations == int(iteration_count))
        and minimum_support == MIN_INDEPENDENT_SECTOR_SUPPORT
        and math.isclose(
            cluster_tolerance, 0.05, rel_tol=0.0, abs_tol=1e-12)
        and len(support) == reported_support_count
        and len(set(support)) == len(support)
        and len(support) >= minimum_support
        and set(support).issubset(target_support)
        and len(set(accepted_support)) == len(accepted_support)
        and set(support).issubset(accepted_support)
        and any(item == recurrent for item in clusters)
    ):
        return None

    observed_support: set[int] = set()
    for member in members:
        if not isinstance(member, dict):
            return None
        try:
            frequency = float(member.get("frequency"))
            period = float(member.get("periodDays"))
            iteration = int(member.get("iteration"))
        except (TypeError, ValueError):
            return None
        if not (
            math.isfinite(frequency)
            and frequency > 0
            and math.isfinite(period)
            and period > 0
            and math.isclose(
                period, 1.0 / frequency,
                rel_tol=1e-9, abs_tol=1e-12)
            and 1 <= iteration <= reported_iterations
            and abs(frequency - residual_frequency) / residual_frequency
            <= cluster_tolerance
        ):
            return None
        if (
            member.get("role") == "independent-residual-multimode"
            and member.get("sector") is not None
        ):
            try:
                observed_support.add(int(member["sector"]))
            except (TypeError, ValueError):
                return None
        matching_point = next((
            point for point in accepted
            if isinstance(point, dict)
            and point.get("iteration") == member.get("iteration")
            and point.get("sector") == member.get("sector")
            and point.get("role") == member.get("role")
            and point.get("candidateFrequency") == member.get("frequency")
            and point.get("candidatePeriodDays") == member.get("periodDays")
            and point.get("candidatePeakProminenceRatio")
            == member.get("prominence")
            and point.get("acceptedDistinctMode") is True
        ), None)
        if matching_point is None:
            return None
    if observed_support != set(support):
        return None
    return {
        "evidenceLineage": MULTIMODE_MODE_EVIDENCE_LINEAGE,
        "establishedPeriodDays": established_period,
        "residualPeriodDays": residual_period,
        "independentSectors": support,
    }


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
