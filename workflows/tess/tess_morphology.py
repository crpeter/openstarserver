from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


RAW_BINS = 32
DOUBLE_BINS = 64
MIN_VALID_BIN_FRACTION = 0.70
MIN_DOUBLE_CYCLES = 1.5
DOUBLE_SUPPORT_MIN_IMPROVEMENT = 0.04
DOUBLE_SUPPORT_MIN_HALF_DIFFERENCE = 0.12
RAW_SUPPORT_MAX_IMPROVEMENT = 0.025
RAW_SUPPORT_MAX_HALF_DIFFERENCE = 0.10
MIN_RESOLUTION_SECTORS = 3
EVOLUTION_FOLLOWUP_MIN_INDEPENDENT_SECTORS = 2


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = min(max(float(q), 0.0), 1.0)
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _phase_profile(
    times: list[float],
    flux: list[float],
    *,
    period_days: float,
    bins: int,
) -> dict[str, Any]:
    if period_days <= 0:
        raise ValueError("period_days must be positive")
    if bins < 8:
        raise ValueError("bins must be >= 8")

    sums = [0.0] * bins
    counts = [0] * bins
    sample_bins: list[tuple[float, int]] = []
    finite_flux: list[float] = []

    for t_raw, f_raw in zip(times, flux):
        t = _float(t_raw)
        f = _float(f_raw)
        if t is None or f is None:
            continue
        phase = (t / period_days) % 1.0
        index = min(int(phase * bins), bins - 1)
        sums[index] += f
        counts[index] += 1
        sample_bins.append((f, index))
        finite_flux.append(f)

    means: list[float | None] = [
        (sums[i] / counts[i]) if counts[i] else None for i in range(bins)
    ]
    valid_means = [value for value in means if value is not None]
    valid_fraction = len(valid_means) / bins if bins else 0.0

    residuals: list[float] = []
    for value, index in sample_bins:
        mean = means[index]
        if mean is not None:
            residuals.append(value - mean)

    total_variance = _variance(finite_flux)
    residual_variance = _variance(residuals)
    explained = (
        1.0 - residual_variance / total_variance
        if total_variance > 1e-15
        else 0.0
    )

    p05 = _percentile(valid_means, 0.05)
    p95 = _percentile(valid_means, 0.95)
    amplitude = (
        max((p95 or 0.0) - (p05 or 0.0), 0.0)
        if p05 is not None and p95 is not None
        else 0.0
    )
    median = _percentile(valid_means, 0.50)
    minimum = min(valid_means) if valid_means else None
    maximum = max(valid_means) if valid_means else None
    minimum_index = means.index(minimum) if minimum is not None else None
    maximum_index = means.index(maximum) if maximum is not None else None

    minimum_duty_cycle = None
    if median is not None and minimum is not None and amplitude > 1e-12:
        threshold = median - 0.5 * (median - minimum)
        below = sum(1 for value in valid_means if value <= threshold)
        minimum_duty_cycle = below / len(valid_means) if valid_means else None

    roughness = None
    if len(valid_means) >= 8 and amplitude > 1e-12 and all(v is not None for v in means):
        full = [float(v) for v in means]
        second = []
        for i in range(bins):
            prev = full[(i - 1) % bins]
            cur = full[i]
            nxt = full[(i + 1) % bins]
            second.append(abs(prev - 2.0 * cur + nxt))
        roughness = statistics.median(second) / amplitude

    return {
        "periodDays": period_days,
        "bins": bins,
        "sampleCount": len(finite_flux),
        "validBinFraction": valid_fraction,
        "explainedVariance": explained,
        "profileAmplitude": amplitude,
        "minimumPhase": ((minimum_index + 0.5) / bins) if minimum_index is not None else None,
        "maximumPhase": ((maximum_index + 0.5) / bins) if maximum_index is not None else None,
        "minimumDutyCycle": minimum_duty_cycle,
        "profileRoughness": roughness,
        "binCounts": counts,
        "binMeans": means,
    }


def _double_wave_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    means = profile.get("binMeans") or []
    bins = int(profile.get("bins") or 0)
    if bins <= 0 or bins % 2:
        return {"available": False}
    half = bins // 2
    amplitude = float(profile.get("profileAmplitude") or 0.0)

    pairs: list[tuple[float, float]] = []
    for i in range(half):
        a = means[i]
        b = means[i + half]
        if a is not None and b is not None:
            pairs.append((float(a), float(b)))
    if not pairs or amplitude <= 1e-12:
        return {"available": False}

    rms_difference = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))
    half_difference_ratio = rms_difference / amplitude

    first = [float(v) for v in means[:half] if v is not None]
    second = [float(v) for v in means[half:] if v is not None]
    all_valid = first + second
    median = _percentile(all_valid, 0.5)
    depth1 = (median - min(first)) if median is not None and first else None
    depth2 = (median - min(second)) if median is not None and second else None
    alternating_depth_difference_ratio = None
    if depth1 is not None and depth2 is not None:
        alternating_depth_difference_ratio = abs(depth1 - depth2) / amplitude

    return {
        "available": True,
        "pairedBinCount": len(pairs),
        "halfCycleDifferenceRms": rms_difference,
        "halfCycleDifferenceRatio": half_difference_ratio,
        "firstHalfMinimumDepth": depth1,
        "secondHalfMinimumDepth": depth2,
        "alternatingMinimumDepthDifferenceRatio": alternating_depth_difference_ratio,
    }


def _dataset_descriptor(path: str | Path, *, role: str) -> dict[str, Any]:
    dataset = _load_json(path)
    source = dataset.get("source") or {}
    times = [float(value) for value in (dataset.get("times") or [])]
    flux = [float(value) for value in (dataset.get("flux") or [])]
    baseline = _float(source.get("baselineDays"))
    if baseline is None and len(times) > 1:
        baseline = max(times) - min(times)
    return {
        "datasetID": dataset.get("id"),
        "targetName": dataset.get("targetName"),
        "sector": source.get("sector"),
        "role": role,
        "path": str(Path(path).expanduser().resolve()),
        "baselineDays": baseline,
        "times": times,
        "flux": flux,
    }


def analyze_morphology(
    *,
    primary_dataset_path: str | Path,
    independent_spec: dict[str, Any],
    raw_period_days: float,
    possible_double_cycle_days: float,
) -> dict[str, Any]:
    raw_period = float(raw_period_days)
    double_period = float(possible_double_cycle_days)
    if raw_period <= 0 or double_period <= 0:
        raise ValueError("Morphology periods must be positive")

    datasets = [_dataset_descriptor(primary_dataset_path, role="primary")]
    for item in independent_spec.get("preparedSectors") or []:
        path = item.get("datasetPath")
        if path:
            datasets.append(_dataset_descriptor(path, role="independent"))

    sector_results: list[dict[str, Any]] = []
    eligible_results: list[dict[str, Any]] = []
    double_supporters: list[int] = []
    raw_supporters: list[int] = []

    for item in datasets:
        baseline = _float(item.get("baselineDays")) or 0.0
        raw_profile = _phase_profile(
            item["times"],
            item["flux"],
            period_days=raw_period,
            bins=RAW_BINS,
        )
        double_profile = _phase_profile(
            item["times"],
            item["flux"],
            period_days=double_period,
            bins=DOUBLE_BINS,
        )
        double_metrics = _double_wave_metrics(double_profile)
        raw_cycles = baseline / raw_period if raw_period > 0 else 0.0
        double_cycles = baseline / double_period if double_period > 0 else 0.0
        improvement = (
            float(double_profile.get("explainedVariance") or 0.0)
            - float(raw_profile.get("explainedVariance") or 0.0)
        )
        half_difference = _float(double_metrics.get("halfCycleDifferenceRatio"))
        eligible = (
            double_cycles >= MIN_DOUBLE_CYCLES
            and float(raw_profile.get("validBinFraction") or 0.0) >= MIN_VALID_BIN_FRACTION
            and float(double_profile.get("validBinFraction") or 0.0) >= MIN_VALID_BIN_FRACTION
            and bool(double_metrics.get("available"))
        )
        supports_double = bool(
            eligible
            and improvement >= DOUBLE_SUPPORT_MIN_IMPROVEMENT
            and half_difference is not None
            and half_difference >= DOUBLE_SUPPORT_MIN_HALF_DIFFERENCE
        )
        supports_raw = bool(
            eligible
            and improvement <= RAW_SUPPORT_MAX_IMPROVEMENT
            and half_difference is not None
            and half_difference <= RAW_SUPPORT_MAX_HALF_DIFFERENCE
        )

        result = {
            "sector": item.get("sector"),
            "role": item.get("role"),
            "datasetID": item.get("datasetID"),
            "datasetPath": item.get("path"),
            "baselineDays": baseline,
            "rawObservedCycles": raw_cycles,
            "doubleObservedCycles": double_cycles,
            "rawProfile": raw_profile,
            "doubleProfile": double_profile,
            "doubleWaveMetrics": double_metrics,
            "doubleExplainedVarianceImprovement": improvement,
            "eligibleForPhysicalCycleDiscrimination": eligible,
            "supportsDoubleCycle": supports_double,
            "supportsRawCycle": supports_raw,
        }
        sector_results.append(result)
        if eligible:
            eligible_results.append(result)
            sector = item.get("sector")
            if supports_double and sector is not None:
                double_supporters.append(int(sector))
            if supports_raw and sector is not None:
                raw_supporters.append(int(sector))

    eligible_count = len(eligible_results)
    independent_eligible = [
        item for item in eligible_results if item.get("role") == "independent"
    ]
    independent_double_supporters = [
        int(item["sector"])
        for item in independent_eligible
        if item.get("supportsDoubleCycle") and item.get("sector") is not None
    ]
    independent_raw_supporters = [
        int(item["sector"])
        for item in independent_eligible
        if item.get("supportsRawCycle") and item.get("sector") is not None
    ]
    independent_eligible_count = len(independent_eligible)
    required_support = (
        max(MIN_RESOLUTION_SECTORS, independent_eligible_count // 2 + 1)
        if independent_eligible_count
        else MIN_RESOLUTION_SECTORS
    )
    double_resolved = len(independent_double_supporters) >= required_support
    raw_resolved = len(independent_raw_supporters) >= required_support

    if double_resolved and not raw_resolved:
        physical_cycle_resolved = True
        resolved_period = double_period
        morphology_class = "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED"
        rationale = (
            "Across a strict majority of sufficiently covered sectors, folding at the possible full cycle explains materially more variance and reveals non-identical half-cycles.",
        )
    elif raw_resolved and not double_resolved:
        physical_cycle_resolved = True
        resolved_period = raw_period
        morphology_class = "RAW_PERIODICITY_PHYSICAL_CYCLE_SUPPORTED"
        rationale = (
            "Across a strict majority of sufficiently covered sectors, the two halves of the doubled fold are nearly interchangeable and the doubled period does not materially improve morphology coherence.",
        )
    else:
        physical_cycle_resolved = False
        resolved_period = None
        morphology_class = "EVOLVING_OR_MIXED_MORPHOLOGY_UNRESOLVED"
        rationale = (
            "Sector morphology does not produce a strict-majority, three-sector discrimination between the recurrent raw family and its possible doubled physical cycle.",
        )

    independent_classes = {
        "doubleSupport": sum(1 for item in independent_eligible if item.get("supportsDoubleCycle")),
        "rawSupport": sum(1 for item in independent_eligible if item.get("supportsRawCycle")),
        "unresolved": sum(
            1
            for item in independent_eligible
            if not item.get("supportsDoubleCycle") and not item.get("supportsRawCycle")
        ),
    }

    # A failed majority vote is not, by itself, evidence for evolution.  It may
    # simply mean that too few sectors were informative.  Continue only when
    # multiple independent, well-covered sectors give genuinely different
    # morphology answers.  The doubled candidate is the least-assumptive
    # reference cycle for the next experiment: its fundamental plus first
    # harmonic spans both members of the unresolved raw/double family.
    represented_classes = sum(1 for count in independent_classes.values() if count)
    evolution_followup_warranted = bool(
        not physical_cycle_resolved
        and independent_eligible_count >= EVOLUTION_FOLLOWUP_MIN_INDEPENDENT_SECTORS
        and represented_classes >= 2
    )

    duty_cycles = [
        _float((item.get("doubleProfile") or {}).get("minimumDutyCycle"))
        for item in eligible_results
    ]
    duty_cycles = [value for value in duty_cycles if value is not None]
    median_duty = statistics.median(duty_cycles) if duty_cycles else None
    roughness_values = [
        _float((item.get("doubleProfile") or {}).get("profileRoughness"))
        for item in eligible_results
    ]
    roughness_values = [value for value in roughness_values if value is not None]
    median_roughness = statistics.median(roughness_values) if roughness_values else None

    if physical_cycle_resolved and morphology_class.startswith("DOUBLE_WAVE"):
        if median_duty is not None and median_duty <= 0.18 and (median_roughness or 0.0) >= 0.08:
            phenomenology = "SHARP_DOUBLE_WAVE_FEATURES"
        else:
            phenomenology = "BROAD_DOUBLE_WAVE_MODULATION"
    elif physical_cycle_resolved and morphology_class.startswith("RAW_"):
        phenomenology = "SINGLE_WAVE_RECURRENCE"
    else:
        phenomenology = "SECTOR_EVOLVING_OR_MULTI_COMPONENT_VARIABILITY"

    return {
        "version": "openstar.tess-morphology.v1",
        "rawPeriodDays": raw_period,
        "possibleDoubleCycleDays": double_period,
        "physicalCycleResolved": physical_cycle_resolved,
        "resolvedPhysicalPeriodDays": resolved_period,
        "morphologyClass": morphology_class,
        "phenomenology": phenomenology,
        "continuationEvidence": {
            "timeFrequencyEvolutionWarranted": evolution_followup_warranted,
            "analysisReferencePeriodDays": double_period if evolution_followup_warranted else None,
            "independentEvidenceClassCount": represented_classes,
            "minimumIndependentSectors": EVOLUTION_FOLLOWUP_MIN_INDEPENDENT_SECTORS,
            "scientificQuestion": (
                "Does temporal evolution/nonstationarity or an additional periodic component "
                "explain the cross-sector morphology inconsistency?"
                if evolution_followup_warranted else None
            ),
        },
        "rationale": list(rationale),
        "eligibleSectorCount": eligible_count,
        "independentEligibleSectorCount": independent_eligible_count,
        "requiredIndependentSupportCount": required_support,
        "doubleCycleSupportingSectors": sorted(double_supporters),
        "rawCycleSupportingSectors": sorted(raw_supporters),
        "independentDoubleCycleSupportingSectors": sorted(independent_double_supporters),
        "independentRawCycleSupportingSectors": sorted(independent_raw_supporters),
        "independentSectorSummary": independent_classes,
        "thresholds": {
            "minimumDoubleObservedCycles": MIN_DOUBLE_CYCLES,
            "minimumValidBinFraction": MIN_VALID_BIN_FRACTION,
            "doubleMinimumExplainedVarianceImprovement": DOUBLE_SUPPORT_MIN_IMPROVEMENT,
            "doubleMinimumHalfCycleDifferenceRatio": DOUBLE_SUPPORT_MIN_HALF_DIFFERENCE,
            "rawMaximumExplainedVarianceImprovement": RAW_SUPPORT_MAX_IMPROVEMENT,
            "rawMaximumHalfCycleDifferenceRatio": RAW_SUPPORT_MAX_HALF_DIFFERENCE,
            "minimumResolutionSectors": MIN_RESOLUTION_SECTORS,
        },
        "summaryMorphology": {
            "medianDoubleMinimumDutyCycle": median_duty,
            "medianDoubleProfileRoughness": median_roughness,
        },
        "sectorResults": sector_results,
    }
