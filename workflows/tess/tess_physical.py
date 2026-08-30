from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .tess_resolved_cycle import (
    MORPHOLOGY_SOURCE,
    authoritative_resolved_cycle,
    validated_cycle_period,
)

from .tess_hypotheses import rotational_sanity


MIN_MODEL_SAMPLES = 200
MIN_MODEL_EXPLAINED_VARIANCE = 0.05
HARMONIC_DOMINANCE_RATIO = 1.0
PHASE_COHERENCE_STRONG = 0.60
PHASE_COHERENCE_WEAK = 0.35
AMPLITUDE_EVOLUTION_FRACTION = 0.20
BROAD_PERIOD_SPREAD_FRACTION = 0.20
TIC_CONTAMINATION_FLAG = 0.10


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def physical_source_localization_continuation(
    physical: dict[str, Any],
    resolved_cycle: dict[str, Any] | None,
) -> bool:
    """Route only an exact unresolved contamination-attribution boundary."""
    period = validated_cycle_period(resolved_cycle)
    reported_period = _float(physical.get("physicalPeriodDays"))
    reported_harmonic = _float(
        physical.get("photometricFirstHarmonicPeriodDays"))
    return bool(
        period is not None
        and reported_period is not None
        and reported_harmonic is not None
        and math.isclose(
            reported_period, period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            reported_harmonic, period / 2.0,
            rel_tol=1e-9, abs_tol=1e-12)
        and physical.get("version") in {
            "openstar.tess-physical-interpretation.v1",
            "openstar.tess-physical-interpretation.v2",
        }
        and physical.get("physicalCycleEvidence") == resolved_cycle
        and physical.get("physicalMechanismResolved") is False
        and (physical.get("contaminationScreen") or {}).get(
            "flaggedByExistingMetadata") is True
        and physical.get("recommendedNextTest")
        == "PIXEL_LEVEL_SOURCE_LOCALIZATION"
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _median_absolute_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def _wrap_angle(value: float) -> float:
    while value <= -math.pi:
        value += 2.0 * math.pi
    while value > math.pi:
        value -= 2.0 * math.pi
    return value


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [list(matrix[i]) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("Singular Fourier normal matrix")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= divisor
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-18:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def _fit_two_harmonic_model(
    times: list[float],
    flux: list[float],
    *,
    physical_period_days: float,
) -> dict[str, Any]:
    finite: list[tuple[float, float]] = []
    for t_raw, f_raw in zip(times, flux):
        t = _float(t_raw)
        f = _float(f_raw)
        if t is not None and f is not None:
            finite.append((t, f))
    if len(finite) < MIN_MODEL_SAMPLES:
        return {
            "available": False,
            "sampleCount": len(finite),
            "reason": f"At least {MIN_MODEL_SAMPLES} finite samples are required.",
        }

    epoch = finite[0][0]
    omega = 2.0 * math.pi / physical_period_days
    normal = [[0.0] * 5 for _ in range(5)]
    rhs = [0.0] * 5
    design_rows: list[tuple[list[float], float]] = []

    for time_value, flux_value in finite:
        x = time_value - epoch
        row = [
            1.0,
            math.sin(omega * x),
            math.cos(omega * x),
            math.sin(2.0 * omega * x),
            math.cos(2.0 * omega * x),
        ]
        design_rows.append((row, flux_value))
        for i in range(5):
            rhs[i] += row[i] * flux_value
            for j in range(5):
                normal[i][j] += row[i] * row[j]

    try:
        beta = _solve_linear_system(normal, rhs)
    except ValueError as error:
        return {
            "available": False,
            "sampleCount": len(finite),
            "reason": str(error),
        }

    residuals = []
    observed = []
    for row, flux_value in design_rows:
        prediction = sum(beta[i] * row[i] for i in range(5))
        residuals.append(flux_value - prediction)
        observed.append(flux_value)

    total_variance = _variance(observed)
    residual_variance = _variance(residuals)
    explained_variance = (
        1.0 - residual_variance / total_variance
        if total_variance > 1e-15
        else 0.0
    )

    fundamental_sin = beta[1]
    fundamental_cos = beta[2]
    harmonic_sin = beta[3]
    harmonic_cos = beta[4]
    fundamental_amplitude = math.hypot(fundamental_sin, fundamental_cos)
    first_harmonic_amplitude = math.hypot(harmonic_sin, harmonic_cos)
    harmonic_ratio = (
        first_harmonic_amplitude / fundamental_amplitude
        if fundamental_amplitude > 1e-12
        else None
    )
    fundamental_phase = math.atan2(fundamental_cos, fundamental_sin)
    first_harmonic_phase = math.atan2(harmonic_cos, harmonic_sin)
    relative_phase = _wrap_angle(first_harmonic_phase - 2.0 * fundamental_phase)

    if harmonic_ratio is None:
        dominant = "FIRST_HARMONIC" if first_harmonic_amplitude > 1e-12 else "UNRESOLVED"
    elif harmonic_ratio >= HARMONIC_DOMINANCE_RATIO:
        dominant = "FIRST_HARMONIC"
    elif harmonic_ratio < 0.67:
        dominant = "FUNDAMENTAL"
    else:
        dominant = "MIXED"

    return {
        "available": True,
        "sampleCount": len(finite),
        "referenceEpoch": epoch,
        "physicalPeriodDays": physical_period_days,
        "photometricFirstHarmonicPeriodDays": physical_period_days / 2.0,
        "coefficients": {
            "offset": beta[0],
            "fundamentalSin": fundamental_sin,
            "fundamentalCos": fundamental_cos,
            "firstHarmonicSin": harmonic_sin,
            "firstHarmonicCos": harmonic_cos,
        },
        "fundamentalAmplitude": fundamental_amplitude,
        "firstHarmonicAmplitude": first_harmonic_amplitude,
        "firstHarmonicToFundamentalAmplitudeRatio": harmonic_ratio,
        "fundamentalPhaseRad": fundamental_phase,
        "firstHarmonicPhaseRad": first_harmonic_phase,
        "translationInvariantRelativeHarmonicPhaseRad": relative_phase,
        "explainedVariance": explained_variance,
        "residualRms": math.sqrt(max(residual_variance, 0.0)),
        "dominantFourierComponent": dominant,
        "eligibleForCrossSectorComparison": bool(
            explained_variance >= MIN_MODEL_EXPLAINED_VARIANCE
        ),
    }


def _dataset_descriptor(path: str | Path, *, role: str) -> dict[str, Any]:
    dataset = _load_json(path)
    source = dataset.get("source") or {}
    return {
        "datasetID": dataset.get("id"),
        "sector": source.get("sector"),
        "role": role,
        "path": str(Path(path).expanduser().resolve()),
        "times": [float(value) for value in (dataset.get("times") or [])],
        "flux": [float(value) for value in (dataset.get("flux") or [])],
    }


def _phase_concentration(values: list[float]) -> float | None:
    if not values:
        return None
    x = sum(math.cos(value) for value in values) / len(values)
    y = sum(math.sin(value) for value in values) / len(values)
    return math.hypot(x, y)


def _catalog_hints(identity: dict[str, Any]) -> dict[str, Any]:
    simbad = identity.get("simbad") or {}
    vsx = identity.get("vsx") or {}
    gaia_variability = identity.get("gaiaVariability") or {}

    texts: list[str] = []
    for value in (
        simbad.get("objectType"),
        simbad.get("spectralType"),
        gaia_variability.get("classification"),
    ):
        if value is not None:
            texts.append(str(value))
    for match in vsx.get("matches") or []:
        if match.get("type") is not None:
            texts.append(str(match.get("type")))
    joined = " | ".join(texts).upper()
    tokens = {
        token
        for token in joined.replace("/", " ").replace("-", " ").replace(";", " ").replace(",", " ").split()
        if token
    }

    binary_hint = (
        any(word in joined for word in ("BINARY", "ECLIPS", "ELLIPSOID"))
        or bool(tokens & {"EA", "EB", "EW", "ELL"})
    )
    pulsation_hint = (
        any(word in joined for word in ("PULSAT", "CEPHEID", "DELTA SCUTI", "SPB", "RR LYRAE", "LPV"))
        or bool(tokens & {"DSCT", "DCEP", "RR", "RRAB", "RRC", "SPB", "LPV"})
    )
    rotation_hint = any(word in joined for word in ("ROT", "SPOT"))

    return {
        "sourceText": texts,
        "binaryHint": binary_hint,
        "pulsationHint": pulsation_hint,
        "rotationHint": rotation_hint,
    }


def _contamination_screen(identity: dict[str, Any]) -> dict[str, Any]:
    tic_metadata = ((identity.get("tic") or {}).get("metadata") or {})
    contamination_ratio = _float(tic_metadata.get("contaminationRatio"))
    gaia = identity.get("gaiaDR3") or {}
    sources = gaia.get("sources") or []
    nearby_count = len(sources)
    additional_sources = max(nearby_count - 1, 0)

    flag_reasons = []
    if contamination_ratio is not None and contamination_ratio >= TIC_CONTAMINATION_FLAG:
        flag_reasons.append("tic-contamination-ratio-non-negligible")
    if additional_sources > 0:
        flag_reasons.append("additional-gaia-source-within-existing-5arcsec-query")

    return {
        "ticContaminationRatio": contamination_ratio,
        "existingGaiaQueryRadiusArcsec": 5.0,
        "gaiaSourcesWithinExistingQuery": nearby_count,
        "additionalGaiaSourcesWithinExistingQuery": additional_sources,
        "flaggedByExistingMetadata": bool(flag_reasons),
        "flagReasons": flag_reasons,
        "canExcludeTessApertureContamination": False,
        "reason": (
            "Existing identity metadata is only a screening layer. The stored Gaia query covers 5 arcsec, "
            "which is not a pixel/aperture-level source-localization test for TESS photometry."
        ),
    }


def _broad_period_spread(broad_interpretation: dict[str, Any] | None) -> dict[str, Any]:
    periods = []
    sectors = []
    for item in (broad_interpretation or {}).get("sectorResults") or []:
        if not item.get("eligibleForClustering"):
            continue
        period = _float(item.get("candidatePeriodDays"))
        if period is None or period <= 0:
            continue
        periods.append(period)
        sectors.append(item.get("sector"))
    if not periods:
        return {
            "available": False,
            "eligibleSectorCount": 0,
            "relativeRange": None,
            "periodsDays": [],
            "sectors": [],
        }
    median = statistics.median(periods)
    relative_range = (max(periods) - min(periods)) / median if median > 0 else None
    return {
        "available": True,
        "eligibleSectorCount": len(periods),
        "periodsDays": periods,
        "sectors": sectors,
        "medianPeriodDays": median,
        "relativeRange": relative_range,
    }


def _evidence_level(score: int) -> str:
    if score >= 4:
        return "STRONG_PHOTOMETRIC_SUPPORT"
    if score >= 2:
        return "MODERATE_PHOTOMETRIC_SUPPORT"
    if score == 1:
        return "WEAK_PHOTOMETRIC_SUPPORT"
    return "NO_POSITIVE_PHOTOMETRIC_SUPPORT"


def _mechanism_scores(
    *,
    morphology: dict[str, Any],
    rotation: dict[str, Any],
    catalog_hints: dict[str, Any],
    independent_models: list[dict[str, Any]],
    phase_concentration: float | None,
    amplitude_variation_fraction: float | None,
    broad_spread: dict[str, Any],
) -> list[dict[str, Any]]:
    double_wave = morphology.get("morphologyClass") == "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED"
    harmonic_dominant_count = sum(
        1
        for item in independent_models
        if item.get("dominantFourierComponent") == "FIRST_HARMONIC"
    )
    independent_count = len(independent_models)
    broad_relative_range = _float(broad_spread.get("relativeRange"))

    mechanisms: dict[str, dict[str, Any]] = {
        "BINARY_LIKE_DOUBLE_WAVE": {"score": 0, "reasons": [], "cautions": []},
        "ROTATIONAL_DOUBLE_WAVE": {"score": 0, "reasons": [], "cautions": []},
        "PULSATION_OR_MULTIMODE": {"score": 0, "reasons": [], "cautions": []},
        "EVOLVING_OR_MIXED_VARIABILITY": {"score": 0, "reasons": [], "cautions": []},
    }

    binary = mechanisms["BINARY_LIKE_DOUBLE_WAVE"]
    if double_wave:
        binary["score"] += 2
        binary["reasons"].append("multi-sector-double-wave-morphology")
    if harmonic_dominant_count >= 3:
        binary["score"] += 1
        binary["reasons"].append("first-harmonic-dominant-in-at-least-three-independent-sectors")
    if phase_concentration is not None and phase_concentration >= PHASE_COHERENCE_STRONG:
        binary["score"] += 1
        binary["reasons"].append("cross-sector-relative-harmonic-phase-coherent")
    if catalog_hints.get("binaryHint"):
        binary["score"] += 2
        binary["reasons"].append("existing-catalog-binary-hint")
    if phase_concentration is not None and phase_concentration < PHASE_COHERENCE_WEAK:
        binary["score"] -= 1
        binary["cautions"].append("double-wave-shape-phase-varies-substantially-by-sector")
    binary["cautions"].append("photometric-double-wave-shape-alone-does-not-establish-binarity")

    rotation_score = mechanisms["ROTATIONAL_DOUBLE_WAVE"]
    if double_wave:
        rotation_score["score"] += 1
        rotation_score["reasons"].append("double-wave-morphology-can-be-produced-by-longitudinal-surface-structure")
    if harmonic_dominant_count >= 3:
        rotation_score["score"] += 1
        rotation_score["reasons"].append("strong-first-harmonic-content")
    if (
        (amplitude_variation_fraction is not None and amplitude_variation_fraction >= AMPLITUDE_EVOLUTION_FRACTION)
        or (phase_concentration is not None and phase_concentration < PHASE_COHERENCE_STRONG)
    ):
        rotation_score["score"] += 1
        rotation_score["reasons"].append("sector-to-sector-waveform-evolution")
    if catalog_hints.get("rotationHint"):
        rotation_score["score"] += 2
        rotation_score["reasons"].append("existing-catalog-rotation-hint")
    if rotation.get("status") == "ruled-out":
        rotation_score["score"] -= 3
        rotation_score["cautions"].append("resolved-period-rotation-exceeds-classical-critical-speed")
    elif rotation.get("status") == "strongly-disfavored":
        rotation_score["score"] -= 2
        rotation_score["cautions"].append("resolved-period-rotation-near-classical-critical-speed")
    elif rotation.get("status") == "unknown":
        rotation_score["cautions"].append("catalog-mass-unavailable-so-breakup-test-is-mass-dependent")

    pulsation = mechanisms["PULSATION_OR_MULTIMODE"]
    if broad_relative_range is not None and broad_relative_range >= BROAD_PERIOD_SPREAD_FRACTION:
        pulsation["score"] += 2
        pulsation["reasons"].append("independent-sector-broad-search-peaks-span-a-wide-period-range")
    if phase_concentration is not None and phase_concentration < PHASE_COHERENCE_STRONG:
        pulsation["score"] += 1
        pulsation["reasons"].append("relative-harmonic-phase-varies-by-sector")
    if amplitude_variation_fraction is not None and amplitude_variation_fraction >= AMPLITUDE_EVOLUTION_FRACTION:
        pulsation["score"] += 1
        pulsation["reasons"].append("fourier-amplitude-varies-by-sector")
    if catalog_hints.get("pulsationHint"):
        pulsation["score"] += 2
        pulsation["reasons"].append("existing-catalog-pulsation-hint")

    mixed = mechanisms["EVOLVING_OR_MIXED_VARIABILITY"]
    if broad_relative_range is not None and broad_relative_range >= BROAD_PERIOD_SPREAD_FRACTION:
        mixed["score"] += 2
        mixed["reasons"].append("independent-sector-dominant-period-changes-materially")
    if amplitude_variation_fraction is not None and amplitude_variation_fraction >= AMPLITUDE_EVOLUTION_FRACTION:
        mixed["score"] += 1
        mixed["reasons"].append("sector-amplitudes-evolve")
    if phase_concentration is not None and phase_concentration < PHASE_COHERENCE_STRONG:
        mixed["score"] += 1
        mixed["reasons"].append("sector-relative-harmonic-phases-are-not-tightly-locked")
    if independent_count >= 3 and 0 < harmonic_dominant_count < independent_count:
        mixed["score"] += 1
        mixed["reasons"].append("fourier-dominance-is-not-uniform-across-independent-sectors")

    ranked = []
    for name, value in mechanisms.items():
        score = max(int(value["score"]), 0)
        ranked.append({
            "hypothesis": name,
            "score": score,
            "evidenceLevel": _evidence_level(score),
            "reasons": value["reasons"],
            "cautions": value["cautions"],
        })
    ranked.sort(key=lambda item: (-item["score"], item["hypothesis"]))
    return ranked


def analyze_physical_interpretation(
    *,
    primary_dataset_path: str | Path,
    independent_spec: dict[str, Any],
    identity: dict[str, Any],
    morphology: dict[str, Any],
    broad_interpretation: dict[str, Any] | None,
    resolved_cycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cycle = resolved_cycle or authoritative_resolved_cycle(
        morphology=morphology,
    )
    physical_period = validated_cycle_period(cycle)
    if physical_period is None:
        raise ValueError(
            "physical interpretation requires an authoritative resolved-cycle contract."
        )

    descriptors = [_dataset_descriptor(primary_dataset_path, role="primary")]
    for item in independent_spec.get("preparedSectors") or []:
        path = item.get("datasetPath")
        if path:
            descriptors.append(_dataset_descriptor(path, role="independent"))

    sector_results = []
    for descriptor in descriptors:
        fit = _fit_two_harmonic_model(
            descriptor["times"],
            descriptor["flux"],
            physical_period_days=physical_period,
        )
        sector_results.append({
            "sector": descriptor.get("sector"),
            "role": descriptor.get("role"),
            "datasetID": descriptor.get("datasetID"),
            "datasetPath": descriptor.get("path"),
            "fourierFit": fit,
        })

    independent_models = [
        item["fourierFit"] | {"sector": item.get("sector")}
        for item in sector_results
        if item.get("role") == "independent"
        and (item.get("fourierFit") or {}).get("available")
        and (item.get("fourierFit") or {}).get("eligibleForCrossSectorComparison")
    ]

    relative_phases = [
        float(item["translationInvariantRelativeHarmonicPhaseRad"])
        for item in independent_models
        if _float(item.get("translationInvariantRelativeHarmonicPhaseRad")) is not None
    ]
    phase_concentration = _phase_concentration(relative_phases)

    harmonic_amplitudes = [
        float(item["firstHarmonicAmplitude"])
        for item in independent_models
        if _float(item.get("firstHarmonicAmplitude")) is not None
    ]
    amplitude_variation_fraction = None
    if harmonic_amplitudes:
        median_amplitude = statistics.median(harmonic_amplitudes)
        mad = _median_absolute_deviation(harmonic_amplitudes)
        if median_amplitude > 1e-12 and mad is not None:
            amplitude_variation_fraction = mad / median_amplitude

    harmonic_dominant_sectors = sorted(
        int(item["sector"])
        for item in independent_models
        if item.get("sector") is not None
        and item.get("dominantFourierComponent") == "FIRST_HARMONIC"
    )

    rotation = rotational_sanity(identity, physical_period)
    catalog_hints = _catalog_hints(identity)
    contamination = _contamination_screen(identity)
    broad_spread = _broad_period_spread(broad_interpretation)
    rankings = _mechanism_scores(
        morphology=morphology,
        rotation=rotation,
        catalog_hints=catalog_hints,
        independent_models=independent_models,
        phase_concentration=phase_concentration,
        amplitude_variation_fraction=amplitude_variation_fraction,
        broad_spread=broad_spread,
    )

    preferred = rankings[0] if rankings else None
    runner_up = rankings[1] if len(rankings) > 1 else None
    margin = (
        preferred["score"] - runner_up["score"]
        if preferred is not None and runner_up is not None
        else None
    )
    preferred_photometric_hypothesis = (
        preferred["hypothesis"]
        if preferred is not None and preferred["score"] >= 3 and (margin is None or margin >= 1)
        else None
    )

    if contamination.get("flaggedByExistingMetadata"):
        next_test = "PIXEL_LEVEL_SOURCE_LOCALIZATION"
    elif preferred_photometric_hypothesis == "BINARY_LIKE_DOUBLE_WAVE":
        next_test = "INDEPENDENT_BINARY_CONFIRMATION"
    elif preferred_photometric_hypothesis == "ROTATIONAL_DOUBLE_WAVE":
        next_test = "SPECTROSCOPIC_ROTATION_CONSTRAINT"
    else:
        next_test = "MULTI_MODE_FREQUENCY_DECOMPOSITION"

    return {
        "version": (
            "openstar.tess-physical-interpretation.v1"
            if cycle.get("sourceKind") == MORPHOLOGY_SOURCE
            else "openstar.tess-physical-interpretation.v2"
        ),
        "physicalPeriodDays": physical_period,
        "photometricFirstHarmonicPeriodDays": physical_period / 2.0,
        "physicalCycleEvidence": cycle,
        "physicalMechanismResolved": False,
        "preferredPhotometricHypothesis": preferred_photometric_hypothesis,
        "preferredHypothesisScoreMargin": margin,
        "mechanismRankings": rankings,
        "rotationConstraint": rotation,
        "catalogHints": catalog_hints,
        "contaminationScreen": contamination,
        "broadIndependentPeriodSpread": broad_spread,
        "crossSectorFourierSummary": {
            "independentEligibleSectorCount": len(independent_models),
            "independentHarmonicDominantSectors": harmonic_dominant_sectors,
            "relativeHarmonicPhaseConcentration": phase_concentration,
            "firstHarmonicAmplitudeVariationFraction": amplitude_variation_fraction,
            "minimumModelExplainedVariance": MIN_MODEL_EXPLAINED_VARIANCE,
            "harmonicDominanceRatioThreshold": HARMONIC_DOMINANCE_RATIO,
        },
        "sectorResults": sector_results,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "The authoritative photometric cycle is resolved, but this stage only ranks mechanism hypotheses and does not convert photometry alone into a physical binary, rotation, or pulsation classification."
        ),
    }
