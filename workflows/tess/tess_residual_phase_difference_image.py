"""Residual-phase difference imaging for the unresolved three-source branch.

This adapts the established difference-image centroid measurement, while the
catalog/PRF bridge is used only to freeze the three positions to test.  No PRF
source-amplitude fit is performed here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_catalog_guided_localization import (
    COMPONENT_IDS, HARMONIC_ORDERS, _production_sector_inputs,
)
from .tess_difference_image import (
    MIN_IMAGE_PEAK_SNR, SOURCE_MARGIN_FLOOR_PIXELS, SOURCE_MATCH_MAX_PIXELS,
    _centroid_from_frames, _extreme_indices, _jackknife_uncertainty, _phase_model,
)
from .tess_residual_localization import _time_warp, _write_json

EXPECTED_SECTORS = (94, 95, 102, 103)


def prepare_residual_phase_difference_imaging(
    *, localization_summary: dict[str, Any], localization_preparation: dict[str, Any],
    output_dir: Path, investigation_id: str,
) -> dict[str, Any]:
    """Freeze the authoritative unresolved bridge without rewriting its evidence."""
    candidates = list(localization_summary.get("catalogCandidates") or
                      localization_preparation.get("catalogCandidates") or [])
    if not (localization_summary.get("recommendedNextTest") ==
            "ADDITIONAL_SOURCE_LOCALIZATION_DATA"
            and localization_summary.get("classification") == "UNRESOLVED"
            and localization_summary.get("sourceAttributionResolved") is False):
        raise RuntimeError("Residual-phase difference imaging requires an unresolved localization recommendation.")
    if len(candidates) < 2:
        raise RuntimeError("Both persisted catalog candidates are required as spatial hypotheses.")
    required = {
        "referenceFamilyPeriodDays": localization_preparation.get("referenceFamilyPeriodDays"),
        "residualReferenceFrequency": localization_preparation.get("residualReferenceFrequency"),
        "residualTimeReferenceDays": localization_preparation.get("residualTimeReferenceDays"),
        "fractionalFrequencyDriftPerDay": localization_preparation.get("fractionalFrequencyDriftPerDay"),
    }
    if any(value is None for value in required.values()):
        raise RuntimeError("The persisted catalog-guided/PRF bridge lacks residual ephemeris values.")
    sectors = tuple(int(value) for value in localization_preparation.get("sectors") or [])
    root = Path(output_dir) / "residual-phase-difference-imaging"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-residual-phase-difference-imaging-preparation.v1",
        "investigationID": investigation_id, "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-difference-image-centroiding",
        "ticID": localization_preparation.get("ticID"),
        "targetSky": localization_preparation.get("targetSky"),
        "catalogCandidates": candidates[:2],
        "spatialHypotheses": [
            {"componentID": "target", **(localization_preparation.get("targetSky") or {})},
            {"componentID": "candidate-1", **candidates[0]},
            {"componentID": "candidate-2", **candidates[1]},
        ],
        "sectors": list(sectors), "referenceFamilyPeriodDays": float(required["referenceFamilyPeriodDays"]),
        "subtractedHarmonicOrders": list(HARMONIC_ORDERS), "physicalCycleResolved": False,
        "residualReferenceFrequency": float(required["residualReferenceFrequency"]),
        "residualTimeReferenceDays": float(required["residualTimeReferenceDays"]),
        "fractionalFrequencyDriftPerDay": float(required["fractionalFrequencyDriftPerDay"]),
        "priorEvidenceReferences": {
            "catalogGuidedPreparationVersion": localization_preparation.get("version"),
            "catalogGuidedInterpretationVersion": localization_summary.get("version"),
        },
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def _sector_classification(centroid: tuple[float, float], centers: list[dict[str, Any]],
                           uncertainty: float, usable: bool) -> tuple[str, dict[str, float]]:
    distances = {str(item["componentID"]): math.hypot(
        centroid[0] - float(item["x"]), centroid[1] - float(item["y"])) for item in centers}
    if not usable:
        return "UNRESOLVED", distances
    ordered = sorted(distances, key=distances.get)
    margin = max(SOURCE_MARGIN_FLOOR_PIXELS, 2.0 * float(uncertainty))
    close = [name for name, distance in distances.items() if distance <= SOURCE_MATCH_MAX_PIXELS]
    if len(close) != 1 or distances[ordered[1]] - distances[ordered[0]] < margin:
        return ("MULTIPLE_OR_BLENDED" if len(close) > 1 else "UNRESOLVED"), distances
    labels = {"target": "TARGET_SUPPORTED", "candidate-1": "CANDIDATE_1_SUPPORTED",
              "candidate-2": "CANDIDATE_2_SUPPORTED"}
    return labels[close[0]], distances


def run_residual_phase_difference_imaging(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct high-minus-low residual-phase images and jackknife centroids."""
    inputs = _production_sector_inputs(preparation) if sector_inputs is None else sector_inputs
    results = []
    for item in inputs:
        times = np.asarray(item["times"], dtype=float)
        cube = np.asarray(item["prewhitened"], dtype=float)
        valid = np.asarray(item["valid"], dtype=bool)
        relative = times - float(preparation["residualTimeReferenceDays"])
        warped = _time_warp(relative, float(preparation["fractionalFrequencyDriftPerDay"]))
        aperture = np.sum(cube[:, valid], axis=1)
        phase = _phase_model(warped, aperture, float(preparation["residualReferenceFrequency"]))
        high, low = _extreme_indices(phase["model"])
        image = _centroid_from_frames(cube, valid, high, low)
        uncertainty, jackknife = _jackknife_uncertainty(cube, valid, high, low)
        centers = list(item.get("componentPixelCenters") or
                       (item.get("acquisitionProvenance") or {}).get("componentPixelCenters") or [])
        if [center.get("componentID") for center in centers] != list(COMPONENT_IDS):
            raise RuntimeError("Sector input must retain target and both candidate pixel positions.")
        usable = float(image["peakSNR"]) >= MIN_IMAGE_PEAK_SNR
        classification, distances = _sector_classification(
            (float(image["centroidX"]), float(image["centroidY"])), centers,
            uncertainty, usable)
        results.append({
            "sector": int(item["sector"]), "classification": classification,
            "differenceImageUsable": usable, "differenceImage": image,
            "centroidUncertaintyPixels": uncertainty, "jackknifeCentroids": jackknife,
            "distancesPixels": distances, "catalogPixelPositions": centers,
            "phaseModel": {"amplitude": phase["amplitude"],
                           "phaseRadians": phase["phaseRadians"],
                           "explainedVariance": phase["explainedVariance"],
                           "highCadences": len(high), "lowCadences": len(low)},
        })
    return {"version": "openstar.tess-residual-phase-difference-imaging-run.v1",
            "execution": "coordinator-local-difference-image-centroiding",
            "sectorResults": results, "physicalCycleResolved": False}


def interpret_residual_phase_difference_imaging(preparation: dict[str, Any],
                                                run: dict[str, Any]) -> dict[str, Any]:
    sectors = list(run.get("sectorResults") or [])
    supported = [item["classification"] for item in sectors
                 if item.get("classification") in {"TARGET_SUPPORTED", "CANDIDATE_1_SUPPORTED",
                                                    "CANDIDATE_2_SUPPORTED"}]
    distinct = set(supported)
    if len(distinct) > 1:
        classification = "SOURCE_SWITCHING_BY_SECTOR"
    elif len(supported) >= 3:
        classification = supported[0]
    elif any(item.get("classification") == "MULTIPLE_OR_BLENDED" for item in sectors):
        classification = "MULTIPLE_OR_BLENDED"
    else:
        classification = "UNRESOLVED"
    return {
        "version": "openstar.tess-residual-phase-difference-imaging-interpretation.v1",
        "classification": classification, "sectorResults": sectors,
        "sourceAttributionResolved": classification in {
            "TARGET_SUPPORTED", "CANDIDATE_1_SUPPORTED", "CANDIDATE_2_SUPPORTED"},
        "physicalCycleResolved": False, "physicalMechanismResolved": False,
        "referenceFamilyPeriodDays": preparation["referenceFamilyPeriodDays"],
        "subtractedHarmonicOrders": preparation["subtractedHarmonicOrders"],
        "residualReferenceFrequency": preparation["residualReferenceFrequency"],
        "recommendedNextTest": None if classification.endswith("SUPPORTED")
                               else "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
        "interpretationGuard": (
            "Centroids are uncertainty-discriminated against all three frozen catalog positions. "
            "Fewer than three consistent sector localizations cannot resolve source attribution."),
    }
