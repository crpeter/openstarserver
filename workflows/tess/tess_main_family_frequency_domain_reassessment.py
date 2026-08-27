"""Preregistered frequency-domain reassessment of a frozen period family.

The spectral window is used only to explain *resolved* displacements.  Inside
one empirical resolution element its response describes the width of the same
finite-baseline feature and is not independent evidence for an alias.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

ROTATION_MULTICYCLE = "ROTATION_MULTICYCLE_FREQUENCY_STRUCTURE_SUPPORTED"
SAMPLING_ALIAS = "SAMPLING_WINDOW_ALIAS_SUPPORTED"
FAMILY_SURVIVES = "PERSISTED_MAIN_FAMILY_FREQUENCY_SUPPORT_SURVIVES"
UNRESOLVED = "MAIN_FAMILY_FREQUENCY_DOMAIN_REASSESSMENT_UNRESOLVED"

METHOD = {
    "frequencyGridOversampling": 12,
    "supportIntervalHalfWidthResolutionMultiples": 1.0,
    "minimumDirectWindowResponse": 0.50,
    "minimumRelativeBranchPower": 0.20,
    "replicatedSectorCount": 2,
}


def _normalized_projection(time: np.ndarray, flux: np.ndarray, frequency: float) -> float:
    """Return floating-mean sinusoidal least-squares power in [0, 1]."""
    phase = 2 * np.pi * frequency * (time - time.min())
    design = np.column_stack((np.ones(len(time)), np.sin(phase), np.cos(phase)))
    model = design @ np.linalg.lstsq(design, flux, rcond=None)[0]
    denominator = np.sum((flux - np.mean(flux)) ** 2)
    return 0.0 if denominator <= 0 else float(max(0.0, 1 - np.sum((flux - model) ** 2) / denominator))


def _window_response(time: np.ndarray, separation: float) -> float:
    phase = 2 * np.pi * separation * (time - time.min())
    return float(abs(np.mean(np.exp(1j * phase))) ** 2)


def analyze_sector(time: Iterable[float], flux: Iterable[float], *, sector_id,
                   rotation_period_days: float, family_period_days: float,
                   method=METHOD):
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    order = np.argsort(time); time, flux = time[order], flux[order]
    baseline = float(np.ptp(time)) if len(time) else 0.0
    resolution = 1.0 / baseline if baseline > 0 else float("inf")
    family_frequency = 1.0 / float(family_period_days)
    rotation_branch_frequency = 1.0 / (2.0 * float(rotation_period_days))
    separation = abs(family_frequency - rotation_branch_frequency)
    separation_units = separation / resolution if np.isfinite(resolution) else 0.0
    response = _window_response(time, separation) if len(time) else 0.0
    family_power = _normalized_projection(time, flux, family_frequency) if len(time) >= 3 else 0.0
    rotation_power = _normalized_projection(time, flux, rotation_branch_frequency) if len(time) >= 3 else 0.0
    strongest = max(family_power, rotation_power)
    relative_floor = method["minimumRelativeBranchPower"] * strongest
    resolved = bool(separation_units >= method["supportIntervalHalfWidthResolutionMultiples"])
    family_supported = bool(strongest > 0 and family_power >= relative_floor)
    rotation_supported = bool(strongest > 0 and rotation_power >= relative_floor)
    if not resolved:
        # Within one support interval the two evaluations are measurements of
        # one feature. Attribute that feature to its higher preregistered
        # branch rather than double-counting correlated powers as two signals.
        family_supported = family_power > rotation_power
        rotation_supported = not family_supported and strongest > 0

    # This resolution gate is essential: a high window response within the
    # main lobe is not a resolved alias of a distinct source frequency.
    direct_window_support = bool(
        resolved and response >= method["minimumDirectWindowResponse"]
        and family_supported and rotation_supported
    )
    return {
        "sectorID": sector_id,
        "timeBaselineDays": baseline,
        "empiricalRayleighResolutionCyclesPerDay": resolution,
        "familyFrequencyCyclesPerDay": family_frequency,
        "rotationDerivedTwoCycleFrequencyCyclesPerDay": rotation_branch_frequency,
        "familyRotationFrequencySeparationCyclesPerDay": separation,
        "familyRotationSeparationResolutionUnits": separation_units,
        "familyPower": family_power,
        "rotationDerivedPower": rotation_power,
        "familyFrequencySupported": family_supported,
        "rotationMulticycleFrequencySupported": rotation_supported,
        "samplingWindowResponse": response,
        "separationEmpiricallyResolvable": resolved,
        "samplingWindowAliasSupported": direct_window_support,
        "directWindowRule": {
            "minimumSeparationResolutionUnits": method["supportIntervalHalfWidthResolutionMultiples"],
            "minimumWindowResponse": method["minimumDirectWindowResponse"],
        },
    }


def combine_sector_results(results, *, method=METHOD):
    n = method["replicatedSectorCount"]
    aliases = [r["sectorID"] for r in results if r.get("samplingWindowAliasSupported")]
    family = [r["sectorID"] for r in results if r.get("familyFrequencySupported")]
    rotation = [r["sectorID"] for r in results if r.get("rotationMulticycleFrequencySupported")]
    family_only = [r["sectorID"] for r in results if r.get("familyFrequencySupported")
                   and not r.get("rotationMulticycleFrequencySupported")]
    rotation_only = [r["sectorID"] for r in results if r.get("rotationMulticycleFrequencySupported")
                     and not r.get("familyFrequencySupported")]
    contradiction = bool(family_only and rotation_only)
    if contradiction:
        classification = UNRESOLVED
    elif len(aliases) >= n:
        classification = SAMPLING_ALIAS
    elif len(rotation) >= n and not family_only:
        classification = ROTATION_MULTICYCLE
    elif len(family_only) >= n:
        classification = FAMILY_SURVIVES
    else:
        classification = UNRESOLVED
    return {"classification": classification, "physicalCycleResolved": False,
            "samplingWindowAliasSectorIDs": aliases,
            "familyFrequencySupportSectorIDs": family,
            "rotationMulticycleSupportSectorIDs": rotation,
            "noStrongerContradiction": not contradiction}


def analyze_frequency_domain_reassessment(sectors, *, rotation_period_days: float,
                                          family_period_days: float, method=METHOD):
    rows = [analyze_sector(s["time"], s["flux"], sector_id=s["sectorID"],
                           rotation_period_days=rotation_period_days,
                           family_period_days=family_period_days, method=method)
            for s in sectors]
    combined = combine_sector_results(rows, method=method)
    return {"schemaVersion": "main-family-frequency-domain-reassessment-v1",
            "method": dict(method), "sectorResults": rows, **combined}
