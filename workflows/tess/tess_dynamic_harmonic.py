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


MIN_ALIAS_SUPPORTING_HELD_OUT_SECTORS = 3
NESTED_ALIAS_METHOD = (
    "NESTED_EVEN_ONLY_VS_EVEN_PLUS_ODD_LEAVE_ONE_SECTOR_OUT_PREDICTION"
)
NESTED_ALIAS_RESOLVED_CLASSIFICATION = (
    "DOUBLE_CYCLE_ODD_HARMONICS_PREDICTIVELY_SUPPORTED"
)
NESTED_ALIAS_EVIDENCE_LINEAGES = frozenset({
    "UNRESOLVED_FAMILY_TIME_FREQUENCY_RECOMMENDATION",
    "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT",
})


def read_frozen_light_curve(path: str | Path, position: int = 0) -> dict[str, Any]:
    """Normalize OpenStar's authoritative frozen ``times``/``flux`` schema.

    Relative times are translated only by an explicitly persisted source time
    origin.  The original source mapping is retained for downstream provenance.
    """
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    source = raw.get("source") or {}
    metadata = raw.get("metadata") or {}
    offset = source.get("originalTimeOriginDays")
    if offset is None:
        offset = source.get("timeOriginDays")
    if offset is None:
        offset = metadata.get("originalTimeOriginDays")
    if offset is None:
        offset = metadata.get("timeOriginDays", 0.0)
    pairs = [(float(t) + float(offset or 0), float(y)) for t, y in zip(raw.get("times") or [], raw.get("flux") or [])
             if math.isfinite(float(t)) and math.isfinite(float(y))]
    if len(pairs) < 16:
        raise RuntimeError("Frozen dataset has too few finite samples for dynamic harmonic modeling.")
    sector = source.get("sector")
    if sector is None:
        sector = raw.get("sector")
    if sector is None:
        sector = metadata.get("sector", position)
    return {"path": str(resolved), "sector": int(sector), "times": [p[0] for p in pairs],
            "flux": [p[1] for p in pairs], "source": dict(source),
            "metadata": dict(metadata),
            "appliedTimeOriginDays": float(offset or 0)}


_read = read_frozen_light_curve  # private compatibility for earlier callers


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
    order_bic_gains: dict[int, float] = {}
    for order in orders:
        reduced_orders = tuple(value for value in orders if value != order)
        if not reduced_orders:
            continue
        reduced = [_sector_full(item, frequency, reduced_orders) for item in data]
        reduced_model = _bic(sum(item["rss"] for item in reduced), n,
                             sector_count * (1 + 2 * len(reduced_orders)))
        models[f"dynamicWithoutHarmonicOrder{order}"] = reduced_model
        order_bic_gains[order] = reduced_model["bic"] - dynamic["bic"]
    highest_gain = order_bic_gains.get(orders[-1])
    if highest_gain is not None:
        models["dynamicWithoutHighestTestedHarmonic"] = models[
            f"dynamicWithoutHarmonicOrder{orders[-1]}"
        ]

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

    supported_orders = [order for order in orders
                        if order_bic_gains.get(order, MIN_BIC_IMPROVEMENT) >= MIN_BIC_IMPROVEMENT]
    return {"referenceFamilyPeriodDays": reference_period_days, "referenceFrequencyCyclesPerDay": frequency,
            "harmonicOrdersTested": list(orders), "supportedHarmonicOrders": supported_orders,
            "sectorFits": [{k: v for k, v in s.items() if k != "_fit"} for s in sectors],
            "modelComparison": {"criterion": "BIC", "conservativeThreshold": MIN_BIC_IMPROVEMENT,
                                "models": models, "bicImprovementDynamicOverStatic": static_gain,
                                "bicImprovementFromHighestTestedHarmonic": highest_gain,
                                "bicImprovementByHarmonicOrder": {
                                    str(order): gain for order, gain in order_bic_gains.items()
                                },
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


def _held_out_sector_prediction(
    training: list[dict[str, Any]],
    held_out: dict[str, Any],
    frequency: float,
    orders: tuple[int, ...],
    phase_learning_orders: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Predict one sector with phases learned only from the other sectors.

    The held-out fit may choose its own offset and signed amplitude for each
    harmonic.  Its harmonic phases remain frozen from the training sectors, so
    evolving amplitude cannot masquerade as predictive phase coherence.
    """
    learned_orders = phase_learning_orders or orders
    if any(order not in learned_orders for order in orders):
        raise ValueError("Prediction orders must be present in phase-learning orders.")
    rows: list[list[float]] = []
    values: list[float] = []
    for sector_index, item in enumerate(training):
        for time, value in zip(item["times"], item["flux"]):
            row = [1.0 if index == sector_index else 0.0
                   for index in range(len(training))]
            row.extend(
                component
                for order in learned_orders
                for component in (
                    math.sin(2 * math.pi * order * frequency * time),
                    math.cos(2 * math.pi * order * frequency * time),
                )
            )
            rows.append(row)
            values.append(value)
    training_fit = _linear_fit(rows, values)
    coefficient_offset = len(training)
    phases = [
        math.atan2(
            training_fit["coefficients"][coefficient_offset + 2 * index + 1],
            training_fit["coefficients"][coefficient_offset + 2 * index],
        )
        for index in range(len(learned_orders))
    ]
    phase_by_order = dict(zip(learned_orders, phases))

    held_out_rows = [
        [1.0] + [
            math.sin(2 * math.pi * order * frequency * time + phase)
            for order in orders
            for phase in (phase_by_order[order],)
        ]
        for time in held_out["times"]
    ]
    predictive = _linear_fit(held_out_rows, held_out["flux"])
    null = _linear_fit([[1.0] for _ in held_out["times"]], held_out["flux"])
    return {
        "sector": held_out["sector"],
        "sampleCount": predictive["sampleCount"],
        "predictiveRss": predictive["rss"],
        "predictiveBic": predictive["bic"],
        "nullRss": null["rss"],
        "nullBic": null["bic"],
        "bicImprovementOverNull": null["bic"] - predictive["bic"],
        "trainingSectorIDs": sorted(item["sector"] for item in training),
        "predictedHarmonicOrders": list(orders),
        "phaseLearningHarmonicOrders": list(learned_orders),
        "heldOutParametersFitted": ["offset", "signed-amplitude-per-harmonic"],
        "phasesLearnedFromTrainingSectorsOnly": True,
    }


def _leave_one_sector_out_predictions(
    data: list[dict[str, Any]],
    period_days: float,
    orders: tuple[int, ...],
    phase_learning_orders: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    frequency = 1.0 / period_days
    return [
        _held_out_sector_prediction(
            [item for index, item in enumerate(data) if index != held_out_index],
            held_out,
            frequency,
            orders,
            phase_learning_orders,
        )
        for held_out_index, held_out in enumerate(data)
    ]


def compare_unresolved_family_dynamic_harmonics(
    *,
    dataset_paths: Iterable[str | Path],
    raw_period_days: float,
    double_cycle_period_days: float,
    primary_sector: int,
    harmonic_orders: Iterable[int] = (1, 2, 3, 4),
) -> dict[str, Any]:
    """Test whether odd harmonics establish the doubled photometric cycle.

    Both nested hypotheses use the doubled-cycle frequency and the same maximum
    absolute frequency.  The equal-half null contains only even orders (the raw
    family expressed at the doubled period); the full model adds the intervening
    odd orders.  Absence of odd-order support cannot establish the shorter
    physical cycle, so only the doubled cycle can be resolved by this test.
    """
    raw_period = float(raw_period_days)
    double_period = float(double_cycle_period_days)
    if not (
        math.isfinite(raw_period)
        and math.isfinite(double_period)
        and raw_period > 0.0
        and double_period > 0.0
        and math.isclose(double_period, 2.0 * raw_period, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise ValueError(
            "Unresolved-family dynamic modeling requires an exact raw/2x cycle pair."
        )
    paths = tuple(dataset_paths)
    orders = tuple(sorted({int(order) for order in harmonic_orders}))
    if not paths or not orders or orders[0] < 1:
        raise ValueError("Frozen datasets and positive harmonic orders are required.")
    if orders != tuple(range(1, orders[-1] + 1)):
        raise ValueError(
            "Raw-family harmonic orders must be contiguous from one for a "
            "matched-frequency nested alias test."
        )
    data = [_read(path, index) for index, path in enumerate(paths)]
    sectors = [item["sector"] for item in data]
    primary = int(primary_sector)
    if primary not in sectors:
        raise RuntimeError(
            "Unresolved-family alias comparison requires the frozen primary sector."
        )
    if len(data) < MIN_ALIAS_SUPPORTING_HELD_OUT_SECTORS + 1:
        raise RuntimeError(
            "Unresolved-family alias comparison requires a primary and at least "
            "three independent frozen sectors."
        )
    if len(set(sectors)) != len(sectors):
        raise RuntimeError(
            "Unresolved-family alias comparison requires distinct frozen sectors."
        )

    even_orders = tuple(2 * order for order in orders)
    full_orders = tuple(range(1, even_orders[-1] + 1))
    odd_orders = tuple(order for order in full_orders if order % 2 == 1)
    full_double_model = model_dynamic_harmonics(
        dataset_paths=paths,
        reference_period_days=double_period,
        harmonic_orders=full_orders,
    )
    even_predictions = _leave_one_sector_out_predictions(
        data, double_period, even_orders, full_orders)
    full_predictions = _leave_one_sector_out_predictions(
        data, double_period, full_orders, full_orders)
    comparisons = []
    odd_support: list[int] = []
    independent_odd_support: list[int] = []
    for even, full in zip(even_predictions, full_predictions):
        if even["sector"] != full["sector"]:
            raise RuntimeError("Held-out sector identity changed between hypotheses.")
        # Positive values favor the full model.  BIC includes the additional
        # held-out signed amplitudes, so this does not waive their complexity.
        delta = even["predictiveBic"] - full["predictiveBic"]
        supported = bool(
            delta >= MIN_BIC_IMPROVEMENT
            and full["bicImprovementOverNull"] >= MIN_BIC_IMPROVEMENT
        )
        if supported:
            odd_support.append(even["sector"])
            if even["sector"] != primary:
                independent_odd_support.append(even["sector"])
        comparisons.append({
            "sector": even["sector"],
            "role": "PRIMARY" if even["sector"] == primary else "INDEPENDENT",
            "equalHalfEvenOnlyHypothesis": even,
            "fullDoubleCycleHypothesis": full,
            "deltaBicFullMinusEvenOnly": delta,
            "oddHarmonicStructureSupported": supported,
        })

    independent_comparisons = [
        item for item in comparisons if item["role"] == "INDEPENDENT"
    ]
    aggregate_independent_delta = sum(
        item["deltaBicFullMinusEvenOnly"] for item in independent_comparisons
    )
    double_resolved = bool(
        aggregate_independent_delta >= MIN_BIC_IMPROVEMENT
        and len(independent_odd_support)
        >= MIN_ALIAS_SUPPORTING_HELD_OUT_SECTORS
    )
    if double_resolved:
        selected_relation = "DOUBLE_CYCLE"
        selected_period = double_period
        selected_model = full_double_model
        classification = NESTED_ALIAS_RESOLVED_CLASSIFICATION
    else:
        selected_relation = None
        selected_period = None
        selected_model = None
        classification = "UNRESOLVED_FAMILY_DYNAMIC_HARMONIC_ALIAS_AMBIGUOUS"

    alias_resolution = {
        "method": NESTED_ALIAS_METHOD,
        "criterion": "BIC",
        "conservativeThreshold": MIN_BIC_IMPROVEMENT,
        "equalHalfEvenHarmonicOrders": list(even_orders),
        "discriminatingOddHarmonicOrders": list(odd_orders),
        "fullDoubleCycleHarmonicOrders": list(full_orders),
        "maximumAbsoluteFrequencyMatched": True,
        "primarySector": primary,
        "minimumSupportingIndependentHeldOutSectors": (
            MIN_ALIAS_SUPPORTING_HELD_OUT_SECTORS
        ),
        "aggregateIndependentDeltaBicFullMinusEvenOnly": (
            aggregate_independent_delta
        ),
        "oddHarmonicSupportingHeldOutSectors": sorted(odd_support),
        "oddHarmonicSupportingIndependentHeldOutSectors": sorted(
            independent_odd_support
        ),
        "equalHalfOutcomeInterpretation": (
            "NON_RESOLUTION_ONLY; ABSENCE_OF_ODD_HARMONICS_DOES_NOT_ESTABLISH_"
            "THE_SHORTER_PHYSICAL_CYCLE"
        ),
        "selectedPeriodRelation": selected_relation,
        "selectedPeriodDays": selected_period,
        "physicalCycleResolved": selected_model is not None,
        "comparisons": comparisons,
    }
    common = {
        "classification": classification,
        "rawFamilyPeriodDays": raw_period,
        "possibleDoubleCycleDays": double_period,
        "periodAliasResolution": alias_resolution,
        "periodHypothesisModels": {
            "equalHalfEvenOnly": {
                "referencePeriodDays": double_period,
                "harmonicOrdersTested": list(even_orders),
                "absoluteFrequencySpanMatched": True,
            },
            "fullDoubleCycle": full_double_model,
        },
        "physicalCycleResolved": selected_model is not None,
        "resolvedPhysicalPeriodDays": selected_period,
        "physicalMechanismResolved": False,
        "dataReuse": {
            "frozenDatasetPaths": [
                str(Path(path).expanduser().resolve()) for path in paths
            ],
            "downloadPerformed": False,
        },
    }
    if selected_model is None:
        return {
            **common,
            "referenceFamilyPeriodDays": double_period,
            "referencePeriodRole": "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE",
            "recommendedNextTest": (
                "ADDITIONAL_INDEPENDENT_SECTOR_CYCLE_ALIAS_CONFIRMATION"
            ),
        }
    return {
        **selected_model,
        **common,
        "referenceFamilyPeriodDays": selected_period,
        "referencePeriodRole": "PREDICTIVELY_RESOLVED_PHOTOMETRIC_CYCLE",
        "recommendedNextTest": selected_model.get("recommendedNextTest"),
    }


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
