"""Preregistered fixed-frequency reassessment of a frozen TESS period family.

This is coordinator-local hypothesis testing, not a period discovery search.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

ROTATION_MULTICYCLE = "ROTATION_MULTICYCLE_FREQUENCY_STRUCTURE_SUPPORTED"
WINDOW_ALIAS = "SAMPLING_WINDOW_ALIAS_SUPPORTED"
FAMILY_SURVIVES = "PERSISTED_MAIN_FAMILY_FREQUENCY_SUPPORT_SURVIVES"
UNRESOLVED = "MAIN_FAMILY_FREQUENCY_DOMAIN_REASSESSMENT_UNRESOLVED"

METHOD = {
    "frequencyResolutionMethod": "Rayleigh 1 / observed time baseline",
    "fitMethod": "floating-mean fixed-frequency sinusoid normalized by constant-model RSS",
    "windowMethod": "squared modulus of mean exp(2*pi*i*delta_f*time)",
    "minimumReplicatedSectorCount": 2,
    "minimumNormalizedPowerAdvantage": 0.05,
    "minimumDirectWindowResponse": 0.50,
    "supportIntervalHalfWidthResolutionMultiples": 1.0,
    "contradictionsFailClosed": True,
}


def _power(time: np.ndarray, flux: np.ndarray, frequency: float) -> float:
    phase = 2.0 * np.pi * frequency * time
    design = np.column_stack((np.ones(len(time)), np.sin(phase), np.cos(phase)))
    residual = flux - design @ np.linalg.lstsq(design, flux, rcond=None)[0]
    null = flux - np.mean(flux)
    denominator = float(null @ null)
    return 0.0 if denominator <= 0 else float(max(0.0, 1.0 - (residual @ residual) / denominator))


def _window(time: np.ndarray, separation: float) -> float:
    response = np.mean(np.exp(2j * np.pi * separation * (time - time.min())))
    return float(abs(response) ** 2)


def analyze_sector(time: Iterable[float], flux: Iterable[float], *, sector_id: int,
                   rotation_period_days: float, family_period_days: float,
                   possible_double_days: float, method=METHOD) -> dict[str, Any]:
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    order = np.argsort(time); time, flux = time[order], flux[order]
    if len(time) < 16 or time[-1] <= time[0]:
        raise ValueError("fixed-hypothesis reassessment requires 16 samples and a positive baseline")
    baseline = float(time[-1] - time[0]); resolution = 1.0 / baseline
    periods = {"frozenFamily": float(family_period_days),
        "frozenPossibleDouble": float(possible_double_days),
        "authoritativeRotation": float(rotation_period_days),
        "expectedTwoRotation": 2.0 * rotation_period_days,
        "expectedFourRotation": 4.0 * rotation_period_days}
    frequencies = {name: 1.0 / value for name, value in periods.items()}
    fits = {name: {"frequencyCyclesPerDay": frequencies[name], "periodDays": periods[name],
        "normalizedLombScargleEquivalentPower": _power(time, flux, frequencies[name])}
        for name in periods}
    family_score = fits["frozenFamily"]["normalizedLombScargleEquivalentPower"] + fits["frozenPossibleDouble"]["normalizedLombScargleEquivalentPower"]
    rotation_score = fits["expectedTwoRotation"]["normalizedLombScargleEquivalentPower"] + fits["expectedFourRotation"]["normalizedLombScargleEquivalentPower"]
    margin = method["minimumNormalizedPowerAdvantage"]
    preference = ("PERSISTED_FAMILY" if family_score-rotation_score >= margin else
        "ROTATION_MULTICYCLE" if rotation_score-family_score >= margin else "INDETERMINATE")
    diagnostics = {}
    for family_name, rotation_name in (("frozenFamily", "expectedTwoRotation"),
                                       ("frozenPossibleDouble", "expectedFourRotation")):
        separation = abs(frequencies[family_name] - frequencies[rotation_name])
        separation_in_resolution_units = separation / resolution
        response = _window(time, separation)
        separation_passes_resolution_gate = (
            separation_in_resolution_units
            >= method["supportIntervalHalfWidthResolutionMultiples"]
        )
        direct_window_support = (
            separation_passes_resolution_gate
            and response >= method["minimumDirectWindowResponse"]
        )
        diagnostics[family_name] = {"rotationBranch": rotation_name,
            "frequencySeparationCyclesPerDay": separation,
            "separationInResolutionUnits": separation_in_resolution_units,
            "samplingWindowResponse": response,
            "separationPassesResolutionGate": separation_passes_resolution_gate,
            "directWindowSupport": direct_window_support}
    alias = preference != "PERSISTED_FAMILY" and all(
        item["directWindowSupport"] for item in diagnostics.values())
    return {"sectorID": int(sector_id), "baselineDays": baseline,
        "sampleCount": int(len(time)), "frequencyResolutionCyclesPerDay": resolution,
        "frozenFamilyFrequency": frequencies["frozenFamily"], "frozenFamilyPeriodDays": periods["frozenFamily"],
        "frozenPossibleDoubleFrequency": frequencies["frozenPossibleDouble"], "frozenPossibleDoublePeriodDays": periods["frozenPossibleDouble"],
        "authoritativeRotationFrequency": frequencies["authoritativeRotation"], "authoritativeRotationPeriodDays": periods["authoritativeRotation"],
        "expectedTwoRotationFrequency": frequencies["expectedTwoRotation"], "expectedTwoRotationPeriodDays": periods["expectedTwoRotation"],
        "expectedFourRotationFrequency": frequencies["expectedFourRotation"], "expectedFourRotationPeriodDays": periods["expectedFourRotation"],
        "fixedHypothesisFits": fits, "familyCombinedPower": family_score,
        "rotationMulticycleCombinedPower": rotation_score, "sectorPreference": preference,
        "samplingWindowDiagnostics": diagnostics, "samplingWindowAliasSupported": alias,
        "empiricalSupportIntervalHalfWidthCyclesPerDay": resolution * method["supportIntervalHalfWidthResolutionMultiples"]}


def combine_sector_results(results: list[dict[str, Any]], method=METHOD) -> dict[str, Any]:
    n = method["minimumReplicatedSectorCount"]
    family = [r["sectorID"] for r in results if r["sectorPreference"] == "PERSISTED_FAMILY"]
    rotation = [r["sectorID"] for r in results if r["sectorPreference"] == "ROTATION_MULTICYCLE"]
    aliases = [r["sectorID"] for r in results if r["samplingWindowAliasSupported"]]
    contradiction = bool(family and rotation)
    if contradiction:
        classification = UNRESOLVED
    elif len(aliases) >= n:
        classification = WINDOW_ALIAS
    elif len(rotation) >= n:
        classification = ROTATION_MULTICYCLE
    elif len(family) >= n:
        classification = FAMILY_SURVIVES
    else:
        classification = UNRESOLVED
    return {"classification": classification, "replicationThreshold": n,
        "persistedFamilyPreferredSectorIDs": family,
        "rotationMulticyclePreferredSectorIDs": rotation,
        "samplingWindowAliasSectorIDs": aliases, "contradictionDetected": contradiction,
        "physicalCycleResolved": False, "exactPhysicalCycleResolved": False,
        "recommendedNextTest": "FINALIZE_FREQUENCY_DOMAIN_REASSESSMENT"}


def analyze_frequency_domain_reassessment(sectors, *, rotation_period_days,
        family_period_days, possible_double_days, prior_time_domain, method=METHOD):
    results = [analyze_sector(s["time"], s["flux"], sector_id=s["sectorID"],
        rotation_period_days=rotation_period_days, family_period_days=family_period_days,
        possible_double_days=possible_double_days, method=method) for s in sectors]
    combined = combine_sector_results(results, method)
    return {"schemaVersion": "main-family-frequency-domain-reassessment-v1",
        "method": method, "sectorResults": results, "combinedEvidence": combined,
        **combined,
        "priorTimeDomainClassification": prior_time_domain["classification"],
        "priorTimeDomainRawFamilyRecurrenceSectorIDs": prior_time_domain["rawFamilyRecurrenceSectorIDs"],
        "priorTimeDomainPossibleDoubleRecurrenceSectorIDs": prior_time_domain["possibleDoubleRecurrenceSectorIDs"],
        "priorTimeDomainRawFamilyCoverageSectorIDs": prior_time_domain["rawFamilyCoverageSectorIDs"],
        "priorTimeDomainPossibleDoubleCoverageSectorIDs": prior_time_domain["possibleDoubleCoverageSectorIDs"],
        "interpretationNotes": ["This fixed-frequency diagnostic did not perform a broad period search.",
            "Frequency-domain reassessment cannot promote physicalCycleResolved."]}
