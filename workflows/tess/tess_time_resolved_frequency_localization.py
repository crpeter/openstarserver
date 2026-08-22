"""Independent, time-resolved fixed-frequency coherent pixel localization.

The persisted stage-054 windows and residual ephemeris are inputs, not fit
parameters.  Unlike stages 055-056, this experiment never forms a high-minus-
low image: it localizes the complex pixel response at the frozen frequency.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_catalog_guided_localization import COMPONENT_IDS
from .tess_frequency_localized_pixel import (
    _jackknife_uncertainty, _response_map,
)
from .tess_residual_localization import _time_warp, _write_json
from .tess_residual_phase_difference_image import _production_difference_image_inputs

_PRESERVED = ("referenceFamilyPeriodDays", "subtractedHarmonicOrders",
              "physicalCycleResolved", "residualReferenceFrequency",
              "residualTimeReferenceDays", "fractionalFrequencyDriftPerDay",
              "targetSky", "catalogCandidates", "sectors")
_IDENTITIES = {"TARGET_SUPPORTED", "CANDIDATE_1_SUPPORTED", "CANDIDATE_2_SUPPORTED"}
# TESS times are persisted as JSON doubles.  This tolerance is tight enough to
# reject a different cadence while allowing a final-bit serialization change.
WINDOW_TIME_TOLERANCE_DAYS = 1e-7


def prepare_time_resolved_frequency_localization(
    *, stage054: dict[str, Any], stage055: dict[str, Any], stage056: dict[str, Any],
    output_dir: Path, investigation_id: str,
) -> dict[str, Any]:
    if (stage056.get("classification") != "TIME_VARIABLE_LOCALIZATION"
            or stage056.get("sourceAttributionResolved") is not False
            or stage056.get("physicalMechanismResolved") is not False
            or stage056.get("recommendedNextTest")
            != "TIME_VARIABLE_SOURCE_LOCALIZATION_FOLLOWUP"):
        raise RuntimeError("Stage 056 does not authorize the fixed-frequency follow-up.")
    if any(key not in stage054 for key in _PRESERVED):
        raise RuntimeError("Persisted stage 054 is incomplete.")
    prior = {int(s["sector"]): s for s in stage055.get("sectorResults") or []}
    windows = []
    for sector in stage054["sectors"]:
        item = prior.get(int(sector))
        if item is None:
            raise RuntimeError(f"Stage 055 has no persisted windows for sector {sector}.")
        for window in item.get("windowResults") or []:
            windows.append({"sector": int(sector), "windowIndex": window["windowIndex"],
                "cadenceStartIndex": window["cadenceStartIndex"],
                "cadenceEndIndex": window["cadenceEndIndex"],
                "cadenceCount": window["cadenceCount"], "timeRangeDays": window["timeRangeDays"],
                "stage056Classification": window["classification"]})
    root = Path(output_dir) / "time-resolved-frequency-localization"
    root.mkdir(parents=True, exist_ok=True)
    result = {"version": "openstar.tess-time-resolved-frequency-localization-preparation.v1",
        "investigationID": investigation_id, "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-fixed-frequency-complex-pixel-response",
        "spatialHypotheses": list(stage054.get("spatialHypotheses") or []),
        "frozenWindows": windows, "priorStage056Classification": stage056["classification"]}
    result.update({key: stage054[key] for key in _PRESERVED})
    _write_json(Path(result["preparationPath"]), result)
    return result


def _fixed_frequency_power(times: np.ndarray, cube: np.ndarray, valid: np.ndarray,
                           frequency: float) -> np.ndarray:
    centered = times - float(np.mean(times))
    design = np.column_stack((np.ones(len(times)), np.sin(2*np.pi*frequency*centered),
                              np.cos(2*np.pi*frequency*centered)))
    flat = cube.reshape(len(times), -1)
    beta = np.linalg.pinv(design) @ flat
    fitted = design[:, 1:] @ beta[1:]
    variance = np.sum((flat - np.mean(flat, axis=0)) ** 2, axis=0)
    power = np.sum(fitted**2, axis=0) / np.maximum(variance, 1e-12)
    return np.where(valid, power.reshape(valid.shape), np.nan)


def _classify(centroid: tuple[float, float], centers: list[dict[str, Any]],
              uncertainty: float, usable: bool) -> tuple[str, dict[str, float]]:
    distances = {c["componentID"]: math.hypot(centroid[0]-float(c["x"]),
                                               centroid[1]-float(c["y"])) for c in centers}
    if not usable:
        return "NO_QUALITY_LOCALIZATION", distances
    ordered = sorted(distances.items(), key=lambda item: item[1])
    margin = max(.25, 2*float(uncertainty))
    if ordered[0][1] <= 1.05 and ordered[1][1] - ordered[0][1] >= margin:
        return {"target": "TARGET_SUPPORTED", "candidate-1": "CANDIDATE_1_SUPPORTED",
                "candidate-2": "CANDIDATE_2_SUPPORTED"}[ordered[0][0]], distances
    if ordered[0][1] <= 1.05 and ordered[1][1] <= 1.05:
        return "MULTIPLE_OR_BLENDED", distances
    return "UNRESOLVED", distances


def run_time_resolved_frequency_localization(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    inputs = (_production_difference_image_inputs(preparation) if sector_inputs is None
              else sector_inputs)
    frozen = {(w["sector"], w["windowIndex"]): w for w in preparation["frozenWindows"]}
    sectors = []
    for item in inputs:
        sector = int(item["sector"]); times = np.asarray(item["times"], float)
        cube = np.asarray(item["prewhitened"], float); valid = np.asarray(item["valid"], bool)
        centers = list(item.get("componentPixelCenters") or
                       (item.get("acquisitionProvenance") or {}).get("componentPixelCenters") or [])
        if [c.get("componentID") for c in centers] != list(COMPONENT_IDS):
            raise RuntimeError("Source positions must remain frozen for all three components.")
        results = []
        for number in sorted(n for s, n in frozen if s == sector):
            spec = frozen[(sector, number)]
            start, end = int(spec["cadenceStartIndex"]), int(spec["cadenceEndIndex"])
            indices = np.arange(start, end + 1)
            persisted_range = [float(value) for value in spec["timeRangeDays"]]
            in_range = start >= 0 and end >= start and end < len(times)
            reacquired_count = int(len(indices)) if in_range else 0
            reacquired_range = ([float(times[start]), float(times[end])]
                                if in_range else None)
            monotonic = bool(in_range and np.all(np.diff(times[indices]) > 0.0))
            reproduced = bool(
                in_range
                and reacquired_count == int(spec["cadenceCount"])
                and monotonic
                and np.allclose(reacquired_range, persisted_range, rtol=0.0,
                                atol=WINDOW_TIME_TOLERANCE_DAYS)
            )
            diagnostic = {**spec,
                "persistedTimeRangeDays": persisted_range,
                "reacquiredTimeRangeDays": reacquired_range,
                "persistedCadenceCount": int(spec["cadenceCount"]),
                "reacquiredCadenceCount": reacquired_count,
                "windowReproduced": reproduced}
            if not reproduced:
                diagnostic.update({"classification": "NO_QUALITY_LOCALIZATION",
                    "qualityState": "WINDOW_REPRODUCTION_MISMATCH",
                    "qualityFailure": (
                        "Reacquired cadences do not reproduce the persisted stage-055 window."
                    )})
                results.append(diagnostic)
                continue
            try:
                wt, wc = times[indices], cube[indices]
                warped = _time_warp(wt-float(preparation["residualTimeReferenceDays"]),
                                    float(preparation["fractionalFrequencyDriftPerDay"]))
                frequency = float(preparation["residualReferenceFrequency"])
                powers = _fixed_frequency_power(warped, wc, valid, frequency)
                response = _response_map(times=warped, residual_cube=wc, valid_pixels=valid,
                                         frequency=frequency, power_map=powers)
                uncertainty, jackknife = _jackknife_uncertainty(
                    times=warped, residual_cube=wc, valid_pixels=valid,
                    frequency=frequency, power_map=powers)
                label, distances = _classify((response["centroidX"], response["centroidY"]),
                                              centers, uncertainty, response["mapUsable"])
                diagnostic.update({"classification": label, "qualityState":
                    "QUALITY_LOCALIZATION" if response["mapUsable"] else "NO_QUALITY_LOCALIZATION",
                    "response": response, "centroidUncertaintyPixels": uncertainty,
                    "jackknifeCentroids": jackknife, "distancesPixels": distances,
                    "catalogPixelPositions": centers})
            except (RuntimeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
                diagnostic.update({"classification": "NO_QUALITY_LOCALIZATION",
                                   "qualityState": "NO_QUALITY_LOCALIZATION",
                                   "qualityFailure": str(exc)})
            results.append(diagnostic)
        sectors.append({"sector": sector, "windowResults": results})
    if [s["sector"] for s in sectors] != list(preparation["sectors"]):
        raise RuntimeError("Inputs do not match the frozen persisted sector ordering.")
    return {"version": "openstar.tess-time-resolved-frequency-localization-run.v1",
            "sectorResults": sectors, "physicalCycleResolved": False}


def _motion(windows: list[dict[str, Any]]) -> bool:
    quality = [w for w in windows if w.get("qualityState") == "QUALITY_LOCALIZATION"]
    for i, a in enumerate(quality):
        for b in quality[i+1:]:
            distance = math.hypot(a["response"]["centroidX"]-b["response"]["centroidX"],
                                  a["response"]["centroidY"]-b["response"]["centroidY"])
            sigma = math.hypot(a["centroidUncertaintyPixels"], b["centroidUncertaintyPixels"])
            if distance > max(.30, 2*sigma): return True
    return False


def interpret_time_resolved_frequency_localization(preparation: dict[str, Any],
                                                   run: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for sector in run.get("sectorResults") or []:
        windows = sector.get("windowResults") or []; quality = [w for w in windows
            if w.get("qualityState") == "QUALITY_LOCALIZATION"]
        identities = {w["classification"] for w in quality if w["classification"] in _IDENTITIES}
        if len(identities) > 1: label = "WITHIN_SECTOR_SOURCE_SWITCHING_CONFIRMED"
        elif _motion(windows): label = "TIME_VARIABLE_LOCALIZATION_CONFIRMED"
        elif len(quality) >= 2 and len(identities) == 1 and all(w["classification"] in identities for w in quality):
            label = "STABLE_" + next(iter(identities)).replace("_SUPPORTED", "_LOCALIZATION")
        elif any(w["classification"] == "MULTIPLE_OR_BLENDED" for w in quality): label = "MULTI_SOURCE_OR_BLENDED"
        else: label = "UNRESOLVED"
        comparisons = []
        for w in windows:
            prior, current = w["stage056Classification"], w["classification"]
            agreement = (prior == current and current in _IDENTITIES)
            conflict = prior in _IDENTITIES and current in _IDENTITIES and prior != current
            comparisons.append({"windowIndex": w["windowIndex"],
                "stage056DifferenceImageClassification": prior,
                "frequencyResponseClassification": current,
                "assessment": "REINFORCED" if agreement else "CONFLICTING_UNRESOLVED" if conflict else "INCONCLUSIVE"})
        evidence.append({"sector": sector["sector"], "classification": label,
                         "windowResults": windows, "stage056Comparison": comparisons})
    labels = [s["classification"] for s in evidence]
    conflicts = any(c["assessment"] == "CONFLICTING_UNRESOLVED" for s in evidence for c in s["stage056Comparison"])
    stable = {x for x in labels if x.startswith("STABLE_")}
    if conflicts: classification = "UNRESOLVED"
    elif "WITHIN_SECTOR_SOURCE_SWITCHING_CONFIRMED" in labels: classification = "WITHIN_SECTOR_SOURCE_SWITCHING_CONFIRMED"
    elif len(stable) > 1: classification = "CROSS_SECTOR_SOURCE_SWITCHING_CONFIRMED"
    # Raw TPF pixel coordinates have sector-specific WCS frames.  Cross-sector
    # conclusions therefore use only frozen source identities; unmatched
    # centroid motion is confirmed only within a sector above.
    elif "TIME_VARIABLE_LOCALIZATION_CONFIRMED" in labels: classification = "TIME_VARIABLE_LOCALIZATION_CONFIRMED"
    elif len(stable) == 1 and all(x == next(iter(stable)) for x in labels): classification = next(iter(stable))
    elif "MULTI_SOURCE_OR_BLENDED" in labels: classification = "MULTI_SOURCE_OR_BLENDED"
    else: classification = "UNRESOLVED"
    candidates = preparation["catalogCandidates"]
    preferred = candidates[0] if classification == "STABLE_CANDIDATE_1_LOCALIZATION" else candidates[1] if classification == "STABLE_CANDIDATE_2_LOCALIZATION" else None
    recommendations = {"STABLE_TARGET_LOCALIZATION": "TARGET_INTRINSIC_RESIDUAL_MODELING",
        "STABLE_CANDIDATE_1_LOCALIZATION": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        "STABLE_CANDIDATE_2_LOCALIZATION": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        "MULTI_SOURCE_OR_BLENDED": "JOINT_MULTI_SOURCE_VARIABILITY_MODEL",
        "UNRESOLVED": "ADDITIONAL_INDEPENDENT_SOURCE_LOCALIZATION_DATA"}
    result = {"version": "openstar.tess-time-resolved-frequency-localization-interpretation.v1",
        "classification": classification, "sectorEvidence": evidence,
        "stage056DifferenceImageClassification": preparation["priorStage056Classification"],
        "sourceAttributionResolved": classification.startswith("STABLE_"),
        "preferredCandidate": preferred, "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommendations.get(classification,
            "TIME_VARIABLE_LOCALIZATION_PHYSICAL_MECHANISM_FOLLOWUP")}
    result.update({key: preparation[key] for key in _PRESERVED})
    return result
