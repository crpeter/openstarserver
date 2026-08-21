"""Deterministic, server-side dynamic harmonic-family characterization.

The routine consumes only frozen light-curve JSON files.  It deliberately has
no archive client and emits no specialized worker work: a subsequent frequency
refinement, if warranted, is an ordinary ``openstar.lomb-scargle.v1`` job.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .tess_mode_identification import GENERIC_REFINEMENT_WORKLOAD_ID, MIN_BIC_IMPROVEMENT, _solve


def _read(path: str | Path, position: int) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    source = raw.get("source") or {}
    offset = source.get("originalTimeOriginDays", source.get("timeOriginDays", 0.0))
    pairs = [(float(t) + float(offset or 0), float(y)) for t, y in zip(raw.get("times") or [], raw.get("flux") or [])
             if math.isfinite(float(t)) and math.isfinite(float(y))]
    if len(pairs) < 16:
        raise RuntimeError("Frozen dataset has too few finite samples for dynamic harmonic modeling.")
    sector = source.get("sector", raw.get("sector", position))
    return {"path": str(resolved), "sector": int(sector), "times": [p[0] for p in pairs],
            "flux": [p[1] for p in pairs]}


def _linear_fit(rows: list[list[float]], values: list[float], parameters: int | None = None) -> dict[str, Any]:
    size = len(rows[0])
    normal = [[sum(r[i] * r[j] for r in rows) for j in range(size)] for i in range(size)]
    rhs = [sum(r[i] * y for r, y in zip(rows, values)) for i in range(size)]
    coefficients = _solve(normal, rhs)
    residuals = [y - sum(a * x for a, x in zip(coefficients, row)) for row, y in zip(rows, values)]
    rss = sum(value * value for value in residuals)
    n = len(values)
    k = size if parameters is None else parameters
    return {"coefficients": coefficients, "residuals": residuals, "rss": rss,
            "sampleCount": n, "parameterCount": k,
            "bic": n * math.log(max(rss, 1e-300) / n) + k * math.log(n)}


def _sector_full(item: dict[str, Any], frequency: float, orders: tuple[int, ...]) -> dict[str, Any]:
    reference = sum(item["times"]) / len(item["times"])
    rows = [[1.0] + [v for order in orders for v in
            (math.sin(2 * math.pi * order * frequency * (time - reference)),
             math.cos(2 * math.pi * order * frequency * (time - reference)))] for time in item["times"]]
    fit = _linear_fit(rows, item["flux"])
    variance = sum((y - sum(item["flux"]) / len(item["flux"])) ** 2 for y in item["flux"])
    sigma = math.sqrt(fit["rss"] / max(1, len(item["flux"]) - len(rows[0])))
    harmonics = []
    for index, order in enumerate(orders):
        sine, cosine = fit["coefficients"][1 + 2 * index:3 + 2 * index]
        amplitude = math.hypot(sine, cosine)
        local_phase = math.atan2(cosine, sine)
        # Convert from the numerically stable sector-centered basis to one
        # common absolute epoch (BJD zero, or the persisted absolute origin).
        phase = (local_phase - 2 * math.pi * order * frequency * reference + math.pi) % (2 * math.pi) - math.pi
        harmonics.append({"order": order, "amplitude": amplitude, "phaseRadiansAtAbsoluteEpoch": phase,
                          "localPhaseRadians": local_phase,
                          "amplitudeUncertaintyApprox": sigma * math.sqrt(2 / len(item["times"]))})
    return {"sector": item["sector"], "datasetPath": item["path"], "timeReferenceDays": reference,
            "sampleCount": len(item["times"]), "rss": fit["rss"],
            "explainedVariance": 1 - fit["rss"] / max(variance, 1e-300),
            "fitNoiseRms": math.sqrt(fit["rss"] / len(item["times"])), "harmonics": harmonics,
            "_fit": fit}


def _bic(rss: float, n: int, k: int) -> dict[str, Any]:
    return {"rss": rss, "sampleCount": n, "parameterCount": k,
            "bic": n * math.log(max(rss, 1e-300) / n) + k * math.log(n)}


def _circular_scatter(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = math.atan2(sum(math.sin(v) for v in values), sum(math.cos(v) for v in values))
    return math.sqrt(sum(((v - mean + math.pi) % (2 * math.pi) - math.pi) ** 2 for v in values) / len(values))


def model_dynamic_harmonics(*, dataset_paths: Iterable[str | Path], reference_period_days: float,
                            harmonic_orders: Iterable[int] = (1, 2, 3, 4)) -> dict[str, Any]:
    """Compare static and sector-evolving representations of one frequency family."""
    if reference_period_days <= 0:
        raise ValueError("Reference period must be positive.")
    paths = tuple(dataset_paths)
    orders = tuple(sorted({int(order) for order in harmonic_orders}))
    if not paths or not orders or orders[0] < 1:
        raise ValueError("Frozen datasets and positive harmonic orders are required.")
    data = [_read(path, index) for index, path in enumerate(paths)]
    frequency = 1 / reference_period_days
    sectors = [_sector_full(item, frequency, orders) for item in data]
    n = sum(len(item["times"]) for item in data)
    sector_count = len(data)

    # Static coefficients share an absolute epoch; sector offsets remain nuisance
    # parameters so differing normalization cannot masquerade as waveform change.
    rows, values = [], []
    for s, item in enumerate(data):
        for time, value in zip(item["times"], item["flux"]):
            row = [1.0 if j == s else 0.0 for j in range(sector_count)]
            row += [v for order in orders for v in (math.sin(2 * math.pi * order * frequency * time),
                                                     math.cos(2 * math.pi * order * frequency * time))]
            rows.append(row); values.append(value)
    static = _linear_fit(rows, values)
    dynamic_rss = sum(item["rss"] for item in sectors)
    dynamic = _bic(dynamic_rss, n, sector_count * (1 + 2 * len(orders)))

    # Amplitude-only uses the static phase and one signed amplitude per sector.
    phases = [math.atan2(static["coefficients"][sector_count + 2*i + 1],
                         static["coefficients"][sector_count + 2*i]) for i in range(len(orders))]
    amp_rows, amp_values = [], []
    for s, item in enumerate(data):
        for time, value in zip(item["times"], item["flux"]):
            row = [1.0 if j == s else 0.0 for j in range(sector_count)] + [
                math.sin(2 * math.pi * order * frequency * time + phase) if j == s else 0.0
                for j in range(sector_count) for order, phase in zip(orders, phases)]
            amp_rows.append(row); amp_values.append(value)
    amplitude_model = _linear_fit(amp_rows, amp_values)

    models = {"staticGlobalAmplitudeAndPhase": _bic(static["rss"], n, len(rows[0])),
              "sectorVaryingAmplitude": _bic(amplitude_model["rss"], n, len(amp_rows[0])),
              "sectorVaryingAmplitudeAndPhase": dynamic}
    if len(orders) > 1:
        reduced = [_sector_full(item, frequency, orders[:-1]) for item in data]
        reduced_model = _bic(sum(item["rss"] for item in reduced), n,
                             sector_count * (1 + 2 * len(orders[:-1])))
        models["dynamicWithoutHighestTestedHarmonic"] = reduced_model
        highest_gain = reduced_model["bic"] - dynamic["bic"]
    else:
        highest_gain = None

    amplitude_evolution, phase_evolution, ratios = [], [], []
    for index, order in enumerate(orders):
        amplitudes = [s["harmonics"][index]["amplitude"] for s in sectors]
        phase_values = [s["harmonics"][index]["phaseRadiansAtAbsoluteEpoch"] for s in sectors]
        mean_amp = sum(amplitudes) / len(amplitudes)
        spread = math.sqrt(sum((a - mean_amp) ** 2 for a in amplitudes) / len(amplitudes))
        amplitude_evolution.append({"order": order, "meanAmplitude": mean_amp, "standardDeviation": spread,
                                    "fractionalVariation": spread / max(mean_amp, 1e-15)})
        phase_evolution.append({"order": order, "circularScatterRadians": _circular_scatter(phase_values)})
    fundamental = [s["harmonics"][0]["amplitude"] for s in sectors]
    relative_phases = []
    for sector in sectors:
        base_phase = sector["harmonics"][0]["phaseRadiansAtAbsoluteEpoch"]
        relative_phases.append({
            "sector": sector["sector"],
            "relativePhasesRadians": [
                {"order": harmonic["order"],
                 "phaseMinusOrderTimesFundamental": (
                     harmonic["phaseRadiansAtAbsoluteEpoch"]
                     - harmonic["order"] * base_phase + math.pi
                 ) % (2 * math.pi) - math.pi}
                for harmonic in sector["harmonics"][1:]
            ],
        })
    for index, order in enumerate(orders[1:], 1):
        values = [s["harmonics"][index]["amplitude"] / max(fundamental[j], 1e-15) for j, s in enumerate(sectors)]
        ratios.append({"numeratorOrder": order, "denominatorOrder": orders[0], "perSector": values,
                       "mean": sum(values) / len(values)})

    amp_changed = any(item["fractionalVariation"] > 0.05 for item in amplitude_evolution
                      if item["meanAmplitude"] > 5 * sum(s["fitNoiseRms"] for s in sectors) / len(sectors) / math.sqrt(n / sector_count))
    phase_changed = any(item["circularScatterRadians"] > 0.08 for item in phase_evolution)
    static_gain = models["staticGlobalAmplitudeAndPhase"]["bic"] - dynamic["bic"]

    # A common linear phase slope divided by harmonic order is the signature of
    # a small reference-frequency error.  This is a refinement flag, not a claim
    # of physical period evolution.
    slopes = []
    times = [s["timeReferenceDays"] for s in sectors]
    for index, order in enumerate(orders):
        raw = [s["harmonics"][index]["phaseRadiansAtAbsoluteEpoch"] for s in sectors]
        unwrapped = [raw[0]]
        for value in raw[1:]:
            unwrapped.append(unwrapped[-1] + ((value - unwrapped[-1] + math.pi) % (2 * math.pi) - math.pi))
        mt, mp = sum(times)/len(times), sum(unwrapped)/len(unwrapped)
        denominator = sum((t-mt)**2 for t in times)
        if denominator:
            slope = sum((t-mt)*(p-mp) for t, p in zip(times, unwrapped)) / denominator
            residual = sum((p-mp-slope*(t-mt))**2 for t, p in zip(times, unwrapped))
            total = sum((p-mp)**2 for p in unwrapped)
            slopes.append((slope/order, 1-residual/max(total, 1e-30)))
    good_slopes = [s for s, r2 in slopes if r2 > 0.9 and abs(s) * (max(times)-min(times)) > 0.15]
    refinement = len(good_slopes) >= max(2, len(orders)//2)

    residual_variance = dynamic_rss / max(sum((y-sum(item["flux"])/len(item["flux"]))**2
                                               for item in data for y in item["flux"]), 1e-300)
    additional = residual_variance > 0.15
    changed_sector_count = sum(
        any(abs(sectors[index]["harmonics"][j]["amplitude"]
                - sectors[index - 1]["harmonics"][j]["amplitude"])
            > 0.1 * max(amplitude_evolution[j]["meanAmplitude"], 1e-15)
            for j in range(len(orders)))
        for index in range(1, len(sectors))
    )
    evolution_pattern = ("STATIC" if not amp_changed and not phase_changed else
                         "SECTOR_LOCALIZED_OR_DISCONTINUOUS" if changed_sector_count <= 1 else
                         "SMOOTH_OR_MULTI_SECTOR")
    if refinement:
        classification, recommended = "HARMONIC_FAMILY_REQUIRES_FREQUENCY_REFINEMENT", "LOMB_SCARGLE_FREQUENCY_REFINEMENT"
    elif additional:
        classification, recommended = "ADDITIONAL_VARIABILITY_REMAINS", "RESIDUAL_MULTIMODE_LOCALIZATION"
    elif static_gain < MIN_BIC_IMPROVEMENT:
        classification, recommended = "COHERENT_STATIC_HARMONIC_FAMILY", "BINARY_ROTATION_EXTERNAL_EVIDENCE"
    elif amp_changed and phase_changed:
        classification, recommended = "COHERENT_HARMONIC_FAMILY_WITH_AMPLITUDE_AND_PHASE_EVOLUTION", "BINARY_ROTATION_EXTERNAL_EVIDENCE"
    elif amp_changed:
        classification, recommended = "COHERENT_HARMONIC_FAMILY_WITH_AMPLITUDE_EVOLUTION", "BINARY_ROTATION_EXTERNAL_EVIDENCE"
    elif phase_changed:
        classification, recommended = "COHERENT_HARMONIC_FAMILY_WITH_PHASE_EVOLUTION", "BINARY_ROTATION_EXTERNAL_EVIDENCE"
    else:
        classification, recommended = "DYNAMIC_HARMONIC_MODEL_UNRESOLVED", "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"

    return {"referenceFamilyPeriodDays": reference_period_days, "referenceFrequencyCyclesPerDay": frequency,
            "harmonicOrdersTested": list(orders), "sectorFits": [{k: v for k, v in s.items() if k != "_fit"} for s in sectors],
            "modelComparison": {"criterion": "BIC", "conservativeThreshold": MIN_BIC_IMPROVEMENT,
                                "models": models, "bicImprovementDynamicOverStatic": static_gain,
                                "bicImprovementFromHighestTestedHarmonic": highest_gain,
                                "highestTestedHarmonicSupported": highest_gain is not None and highest_gain >= MIN_BIC_IMPROVEMENT},
            "amplitudeEvolution": amplitude_evolution, "phaseEvolution": phase_evolution,
            "harmonicAmplitudeRatios": ratios,
            "translationInvariantRelativeHarmonicPhases": relative_phases,
            "coherenceAssessment": {"sameFrequenciesModeledAcrossSectors": True,
                                    "consistentWithSmallReferenceFrequencyError": refinement,
                                    "normalizedPhaseSlopesRadiansPerDay": [s for s, _ in slopes],
                                    "evolutionPattern": evolution_pattern},
            "residualUnexplainedVarianceFraction": residual_variance, "classification": classification,
            "physicalMechanismResolved": False, "recommendedNextTest": recommended,
            "dataReuse": {"frozenDatasetPaths": [str(Path(p).expanduser().resolve()) for p in paths], "downloadPerformed": False},
            "frequencyRefinement": {"genericDistributedWorkloadIfNeeded": GENERIC_REFINEMENT_WORKLOAD_ID}}


def refine_harmonic_family_frequency(dynamic_result: dict[str, Any]) -> dict[str, Any]:
    """Refine a coherent family frequency from harmonic-normalized phase slopes.

    This is the deterministic interpretation of the frequency evidence already
    measured by the dynamic fit.  If a wider search is subsequently required,
    it remains an ordinary generic Lomb--Scargle workload.
    """
    if dynamic_result.get("recommendedNextTest") != "LOMB_SCARGLE_FREQUENCY_REFINEMENT":
        raise ValueError("Frequency refinement was not recommended.")
    coherence = dynamic_result.get("coherenceAssessment") or {}
    slopes = [float(value) for value in coherence.get("normalizedPhaseSlopesRadiansPerDay") or []
              if math.isfinite(float(value))]
    if not slopes:
        raise RuntimeError("Dynamic harmonic result has no coherent phase-slope evidence.")
    slopes.sort()
    slope = slopes[len(slopes) // 2]
    old_frequency = float(dynamic_result["referenceFrequencyCyclesPerDay"])
    refined_frequency = old_frequency + slope / (2 * math.pi)
    if refined_frequency <= 0:
        raise RuntimeError("Phase-slope refinement produced a non-positive frequency.")
    return {
        "classification": "COHERENT_HARMONIC_FAMILY_FREQUENCY_REFINED",
        "originalFrequencyCyclesPerDay": old_frequency,
        "refinedFrequencyCyclesPerDay": refined_frequency,
        "originalPeriodDays": 1 / old_frequency,
        "refinedPeriodDays": 1 / refined_frequency,
        "frequencyCorrectionCyclesPerDay": refined_frequency - old_frequency,
        "evidence": "COMMON_HARMONIC_NORMALIZED_PHASE_SLOPE",
        "physicalPeriodChangeClaimed": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": "BINARY_ROTATION_EXTERNAL_EVIDENCE",
        "distributedRefinement": {
            "workloadID": GENERIC_REFINEMENT_WORKLOAD_ID,
            "workerSemantics": "GENERIC_LOMB_SCARGLE",
            "specializedTessWorkerLogic": False,
        },
    }
