"""Time-resolved residual-phase difference-image localization.

This is an image-domain experiment: it applies the persisted residual ephemeris
inside deterministic contiguous sector windows and reuses the established
high-minus-low centroid and jackknife implementation.  It does not fit source
amplitudes, move catalog positions, or search frequency per window.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_catalog_guided_localization import COMPONENT_IDS
from .tess_difference_image import (
    MIN_IMAGE_PEAK_SNR, MIN_PHASE_BIN_CADENCES, _centroid_from_frames,
    _extreme_indices, _jackknife_uncertainty, _phase_model,
)
from .tess_residual_localization import _time_warp, _write_json
from .tess_residual_phase_difference_image import (
    EXPECTED_SECTORS, _production_difference_image_inputs, _sector_classification,
)

MIN_WINDOW_CYCLES = 1.5
WINDOW_COUNT = 2

_PRESERVED = ("referenceFamilyPeriodDays", "subtractedHarmonicOrders",
              "physicalCycleResolved", "residualReferenceFrequency",
              "residualTimeReferenceDays", "fractionalFrequencyDriftPerDay",
              "catalogCandidates", "targetSky", "sectors")


def prepare_time_resolved_residual_phase_localization(
    *, temporal_interpretation: dict[str, Any], temporal_preparation: dict[str, Any],
    output_dir: Path, investigation_id: str,
) -> dict[str, Any]:
    if not (temporal_interpretation.get("classification") == "SECTOR_VARIABLE_MULTI_SOURCE"
            and temporal_interpretation.get("sourceIdentifiable") is True
            and temporal_interpretation.get("sourceAttributionResolved") is False
            and temporal_interpretation.get("physicalMechanismResolved") is False
            and temporal_interpretation.get("recommendedNextTest")
            == "ADDITIONAL_SOURCE_LOCALIZATION_DATA"):
        raise RuntimeError("Stage 053 does not authorize time-resolved residual-phase localization.")
    if any(key not in temporal_preparation for key in _PRESERVED):
        raise RuntimeError("The persisted stage-051 bridge is incomplete.")
    if tuple(temporal_preparation["sectors"]) != EXPECTED_SECTORS:
        raise RuntimeError("The continuation requires persisted sectors 94, 95, 102, and 103.")
    if temporal_preparation["physicalCycleResolved"] is not False:
        raise RuntimeError("The unresolved physical-cycle evidence must be preserved.")
    root = Path(output_dir) / "time-resolved-residual-phase-localization"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-time-resolved-residual-phase-localization-preparation.v1",
        "investigationID": investigation_id, "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-windowed-difference-image-centroiding",
        "ticID": temporal_preparation.get("ticID"),
        "spatialHypotheses": list(temporal_preparation.get("spatialHypotheses") or []),
        "priorStage053Classification": temporal_interpretation["classification"],
        "windowPolicy": {"kind": "deterministic-contiguous-equal-cadence",
                         "desiredWindowCount": WINDOW_COUNT,
                         "minimumCadences": 2 * MIN_PHASE_BIN_CADENCES,
                         "minimumResidualCycles": MIN_WINDOW_CYCLES},
    }
    # Values (including harmonic order and sector ordering) are deliberately not
    # normalized or reconstructed: this is the persisted scientific bridge.
    result.update({key: temporal_preparation[key] for key in _PRESERVED})
    _write_json(Path(result["preparationPath"]), result)
    return result


def _window_indices(times: np.ndarray, frequency: float, drift: float,
                    reference: float) -> list[np.ndarray]:
    finite = np.flatnonzero(np.isfinite(times))
    if len(finite) < 2 * MIN_PHASE_BIN_CADENCES:
        return []
    windows = []
    for indices in np.array_split(finite, WINDOW_COUNT):
        if len(indices) < 2 * MIN_PHASE_BIN_CADENCES:
            continue
        warped = _time_warp(times[indices] - reference, drift)
        cycles = abs(float(frequency) * float(warped[-1] - warped[0]))
        if cycles >= MIN_WINDOW_CYCLES:
            windows.append(indices)
    return windows


def run_time_resolved_residual_phase_localization(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    inputs = (_production_difference_image_inputs(preparation)
              if sector_inputs is None else sector_inputs)
    results = []
    for item in inputs:
        times = np.asarray(item["times"], float)
        cube = np.asarray(item["prewhitened"], float)
        valid = np.asarray(item["valid"], bool)
        centers = list(item.get("componentPixelCenters") or
                       (item.get("acquisitionProvenance") or {}).get("componentPixelCenters") or [])
        if [x.get("componentID") for x in centers] != list(COMPONENT_IDS):
            raise RuntimeError("Source positions must remain frozen for all three components.")
        windows = _window_indices(times, float(preparation["residualReferenceFrequency"]),
                                  float(preparation["fractionalFrequencyDriftPerDay"]),
                                  float(preparation["residualTimeReferenceDays"]))
        window_results = []
        for number, indices in enumerate(windows, 1):
            wt, wc = times[indices], cube[indices]
            diagnostic = {"windowIndex": number, "cadenceStartIndex": int(indices[0]),
                          "cadenceEndIndex": int(indices[-1]), "cadenceCount": len(indices),
                          "timeRangeDays": [float(wt[0]), float(wt[-1])]}
            try:
                warped = _time_warp(wt - float(preparation["residualTimeReferenceDays"]),
                                    float(preparation["fractionalFrequencyDriftPerDay"]))
                phase = _phase_model(warped, np.sum(wc[:, valid], axis=1),
                                     float(preparation["residualReferenceFrequency"]))
                high, low = _extreme_indices(phase["model"])
                image = _centroid_from_frames(wc, valid, high, low)
                uncertainty, jackknife = _jackknife_uncertainty(wc, valid, high, low)
                usable = float(image["peakSNR"]) >= MIN_IMAGE_PEAK_SNR
                label, distances = _sector_classification(
                    (image["centroidX"], image["centroidY"]), centers, uncertainty, usable)
                if not usable:
                    label = "NO_QUALITY_LOCALIZATION"
                diagnostic.update({"classification": label, "differenceImageUsable": usable,
                    "differenceImage": image, "centroidUncertaintyPixels": uncertainty,
                    "jackknifeCentroids": jackknife, "distancesPixels": distances,
                    "catalogPixelPositions": centers,
                    "phaseModel": {"amplitude": phase["amplitude"],
                        "phaseRadians": phase["phaseRadians"],
                        "explainedVariance": phase["explainedVariance"],
                        "highCadences": len(high), "lowCadences": len(low)}})
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                diagnostic.update({"classification": "NO_QUALITY_LOCALIZATION",
                                   "differenceImageUsable": False,
                                   "qualityFailure": str(exc)})
            window_results.append(diagnostic)
        results.append({"sector": int(item["sector"]), "windowResults": window_results,
                        "usableWindowCount": sum(w["differenceImageUsable"] for w in window_results),
                        "qualityDiagnostics": {"eligibleWindowCount": len(windows),
                            "inputCadenceCount": len(times)},
                        "acquisitionProvenance": item.get("acquisitionProvenance")})
    if [x["sector"] for x in results] != list(preparation["sectors"]):
        raise RuntimeError("Inputs do not match the frozen persisted sector ordering.")
    return {"version": "openstar.tess-time-resolved-residual-phase-localization-run.v1",
            "execution": "coordinator-local-windowed-difference-image-centroiding",
            "sectorResults": results, "physicalCycleResolved": False,
            "frozenResidualEphemeris": {"frequency": preparation["residualReferenceFrequency"],
                "timeReferenceDays": preparation["residualTimeReferenceDays"],
                "fractionalFrequencyDriftPerDay": preparation["fractionalFrequencyDriftPerDay"]}}


def _centroid_changed(windows: list[dict[str, Any]]) -> bool:
    quality = [w for w in windows if w.get("differenceImageUsable")]
    for i, left in enumerate(quality):
        for right in quality[i + 1:]:
            a, b = left["differenceImage"], right["differenceImage"]
            distance = math.hypot(a["centroidX"] - b["centroidX"],
                                  a["centroidY"] - b["centroidY"])
            sigma = math.hypot(float(left["centroidUncertaintyPixels"]),
                               float(right["centroidUncertaintyPixels"]))
            if distance > max(0.30, 2.0 * sigma):
                return True
    return False


def interpret_time_resolved_residual_phase_localization(
    preparation: dict[str, Any], run: dict[str, Any],
) -> dict[str, Any]:
    identity_labels = {"TARGET_SUPPORTED", "CANDIDATE_1_SUPPORTED", "CANDIDATE_2_SUPPORTED"}
    sector_evidence = []
    for sector in run.get("sectorResults") or []:
        windows = list(sector.get("windowResults") or [])
        quality = [w for w in windows if w.get("differenceImageUsable")]
        identities = {w["classification"] for w in quality if w["classification"] in identity_labels}
        if len(identities) >= 2:  # multiple quality, uncertainty-discriminated identities
            label = "WITHIN_SECTOR_SOURCE_SWITCHING"
        elif len(identities) == 1 and len(quality) >= 2 and all(
                w["classification"] in identities for w in quality):
            label = "STABLE_" + next(iter(identities)).replace("_SUPPORTED", "_LOCALIZATION")
        elif any(w["classification"] == "MULTIPLE_OR_BLENDED" for w in quality):
            label = "MULTI_SOURCE_OR_BLENDED"
        elif _centroid_changed(quality):
            label = "TIME_VARIABLE_LOCALIZATION"
        else:
            label = "UNRESOLVED"
        sector_evidence.append({"sector": sector["sector"], "classification": label,
                                "windowResults": windows})
    sector_labels = [x["classification"] for x in sector_evidence]
    stable = {x for x in sector_labels if x.startswith("STABLE_")}
    if "WITHIN_SECTOR_SOURCE_SWITCHING" in sector_labels:
        classification = "WITHIN_SECTOR_SOURCE_SWITCHING"
    elif len(stable) >= 2:
        classification = "CROSS_SECTOR_SOURCE_SWITCHING"
    elif len(stable) == 1 and all(x == next(iter(stable)) for x in sector_labels):
        classification = next(iter(stable))
    elif "TIME_VARIABLE_LOCALIZATION" in sector_labels:
        classification = "TIME_VARIABLE_LOCALIZATION"
    elif "MULTI_SOURCE_OR_BLENDED" in sector_labels:
        classification = "MULTI_SOURCE_OR_BLENDED"
    else:
        classification = "UNRESOLVED"
    result = {"version": "openstar.tess-time-resolved-residual-phase-localization-interpretation.v1",
        "classification": classification, "sectorEvidence": sector_evidence,
        "sourceAttributionResolved": classification.startswith("STABLE_"),
        "sourceAttributionResolvedByWindows": classification.startswith("STABLE_"),
        "physicalCycleResolved": False, "physicalMechanismResolved": False,
        "recommendedNextTest": None if classification.startswith("STABLE_") else
                               "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
        "interpretationGuard": "Switching requires multiple high-quality, uncertainty-discriminated source identities; nearest-centroid labels alone are insufficient."}
    result.update({key: preparation[key] for key in _PRESERVED})
    return result
