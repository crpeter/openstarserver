"""Generic, fixed-ephemeris localization of replicated eclipse-like events.

The pixel data used here are deliberately not allowed to alter the scientific
clock.  They answer only the spatial question by forming out-of-event minus
in-event images at the ephemeris and durations persisted by binary confirmation.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from openstar_investigation import sha256_json

from .tess_difference_image import MIN_IMAGE_PEAK_SNR, SOURCE_MATCH_MAX_PIXELS, _centroid_from_frames
from .tess_localization import (MAX_CADENCES, MIN_VALID_CADENCES, OFF_TARGET_MIN_PIXELS,
                                _background_subtract_cube, _download_tpf, _pixel_scale_arcsec,
                                _uniform_indices)
from .tess_offset_variability import _skycoord
from .tess_period_family_difference_image import _filled_cube
from .tess_sector_archive import TessArchiveTransientError


RESULT_VERSION = "openstar.tess-eclipse-event-source-localization.v1"
HANDLER_ID = "openstar.tess.eclipse-event-source-localization.analyze"
MIN_INDEPENDENT_SECTORS = 3
MIN_BIN_CADENCES = 6
MIN_EVENTS = 3
MAX_JACKKNIFE_UNCERTAINTY_PIXELS = 0.75
CATALOG_OVERLAP_PIXELS = 0.5


def authoritative_binary_gate(result: dict[str, Any]) -> bool:
    """Accept only the exact completed binary-confirmation-v2 boundary."""
    evidence = result.get("independentEvidence") or {}
    ephemeris = result.get("linearEphemeris") or {}
    return (
        result.get("resultVersion") == "2.0"
        and evidence.get("classification") == "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED"
        and int(evidence.get("supportingIndependentSectorCount") or 0) >= 3
        and ephemeris.get("coherent") is True
        and ephemeris.get("primaryTimingConsistent") is True
        and result.get("recommendedNextTest") == "ECLIPSE_EVENT_SOURCE_LOCALIZATION"
        and result.get("catalogAnswerKeyUsed") is False
    )


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _frozen_sector(binary: dict[str, Any], sector: int) -> dict[str, Any]:
    matches = [item for item in binary.get("sectorResults") or []
               if item.get("sector") is not None and int(item["sector"]) == sector]
    if len(matches) != 1 or matches[0].get("usable") is not True:
        raise ValueError("sector lacks one usable frozen binary-confirmation result")
    return matches[0]


def _event_bins(times: np.ndarray, period: float, epoch: float, duration: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return immutable in/control masks and event cycle labels.

    Controls are symmetric, one duration wide, immediately outside a protected
    half-duration buffer.  The opposite conjunction receives the same exclusion.
    """
    cycles = np.rint((times - epoch) / period).astype(np.int64)
    primary_delta = times - (epoch + cycles * period)
    opposite_cycles = np.rint((times - (epoch + 0.5 * period)) / period).astype(np.int64)
    opposite_delta = times - (epoch + 0.5 * period + opposite_cycles * period)
    half = duration / 2.0
    inside = np.abs(primary_delta) <= half
    control = ((np.abs(primary_delta) >= 1.5 * half) & (np.abs(primary_delta) <= 3.5 * half)
               & (np.abs(opposite_delta) > 1.5 * half))
    return inside, control, cycles


def _catalog_pixels(item: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for hypothesis in item.get("catalogHypotheses") or []:
        pixel = hypothesis.get("pixel") or hypothesis
        x, y = _finite(pixel.get("x")), _finite(pixel.get("y"))
        if x is not None and y is not None:
            result.append({"id": str(hypothesis.get("id") or hypothesis.get("label") or "CATALOG_SOURCE"),
                           "isTarget": hypothesis.get("isTarget") is True, "x": x, "y": y,
                           "sky": hypothesis.get("sky")})
    if not result and item.get("targetPixel"):
        target = item["targetPixel"]
        result.append({"id": "TARGET", "isTarget": True,
                       "x": float(target["x"]), "y": float(target["y"]), "sky": None})
    if sum(source["isTarget"] for source in result) != 1:
        raise ValueError("frozen catalog hypotheses require exactly one target")
    return result


def _event_jackknife(cube: np.ndarray, valid: np.ndarray, inside: np.ndarray,
                     control: np.ndarray, cycles: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    labels = sorted(set(int(value) for value in cycles[inside]))
    centroids = []
    for omitted in labels:
        keep_in = np.flatnonzero(inside & (cycles != omitted))
        # Control windows belonging to the omitted orbital cycle are removed too.
        keep_out = np.flatnonzero(control & (cycles != omitted))
        if len(keep_in) < 2 or len(keep_out) < 2:
            continue
        image = _centroid_from_frames(cube, valid, keep_out, keep_in)
        centroids.append({"omittedCycle": omitted, "centroidX": image["centroidX"],
                          "centroidY": image["centroidY"]})
    if len(centroids) < 2:
        return float("inf"), centroids
    xs = [item["centroidX"] for item in centroids]
    ys = [item["centroidY"] for item in centroids]
    center_x, center_y = statistics.mean(xs), statistics.mean(ys)
    uncertainty = max(0.12, math.sqrt(statistics.mean(
        (x - center_x) ** 2 + (y - center_y) ** 2 for x, y in zip(xs, ys))))
    return uncertainty, centroids


def measure_eclipse_sector(item: dict[str, Any], frozen: dict[str, Any],
                           ephemeris: dict[str, Any]) -> dict[str, Any]:
    sector, role = int(frozen["sector"]), str(frozen["role"])
    period = float(ephemeris["refinedPeriodDays"])
    epoch = float(ephemeris["referenceEpoch"])
    duration = float(frozen["durationDays"])
    times = np.asarray(item["times"], dtype=np.float64)
    cube = np.asarray(item["fluxCube"], dtype=np.float64)
    if cube.ndim != 3 or len(cube) != len(times):
        raise ValueError("fluxCube must match the cadence times")
    keep = np.isfinite(times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
    quality = item.get("qualityMask")
    if quality is not None:
        keep &= np.asarray(quality, dtype=bool)
    selected = _uniform_indices(int(np.count_nonzero(keep)), MAX_CADENCES)
    indices = np.flatnonzero(keep)[selected]
    times, cube = times[indices], cube[indices]
    if len(times) < MIN_VALID_CADENCES:
        raise RuntimeError("inadequate cadence coverage")
    corrected, background = _background_subtract_cube(cube)
    corrected, valid = _filled_cube(corrected)
    inside, control, cycles = _event_bins(times, period, epoch, duration)
    event_count = len(set(int(value) for value in cycles[inside]))
    if np.count_nonzero(inside) < MIN_BIN_CADENCES or np.count_nonzero(control) < MIN_BIN_CADENCES or event_count < MIN_EVENTS:
        raise RuntimeError("inadequate frozen eclipse/control-window coverage")
    image = _centroid_from_frames(corrected, valid, np.flatnonzero(control), np.flatnonzero(inside))
    uncertainty, jackknife = _event_jackknife(corrected, valid, inside, control, cycles)
    catalog = _catalog_pixels(item)
    distances = [{"id": source["id"], "isTarget": source["isTarget"],
                  "distancePixels": math.hypot(image["centroidX"] - source["x"],
                                                image["centroidY"] - source["y"])}
                 for source in catalog]
    ordered = sorted(distances, key=lambda value: value["distancePixels"])
    overlap = any(math.hypot(first["x"] - second["x"], first["y"] - second["y"]) < CATALOG_OVERLAP_PIXELS
                  for index, first in enumerate(catalog) for second in catalog[index + 1:])
    usable = image["peakSNR"] >= MIN_IMAGE_PEAK_SNR and uncertainty <= MAX_JACKKNIFE_UNCERTAINTY_PIXELS and not overlap
    classification = "AMBIGUOUS"
    matched = None
    if usable and ordered[0]["distancePixels"] + 2 * uncertainty <= SOURCE_MATCH_MAX_PIXELS:
        matched = ordered[0]["id"]
        classification = "TARGET_CONSISTENT" if ordered[0]["isTarget"] else "CATALOG_CANDIDATE_CONSISTENT"
    elif usable and ordered[0]["distancePixels"] - 2 * uncertainty >= OFF_TARGET_MIN_PIXELS:
        classification = "OFF_CATALOG"
    reasons = []
    if image["peakSNR"] < MIN_IMAGE_PEAK_SNR: reasons.append("WEAK_DIFFERENCE_IMAGE_SNR")
    if uncertainty > MAX_JACKKNIFE_UNCERTAINTY_PIXELS: reasons.append("EVENT_JACKKNIFE_UNSTABLE")
    if overlap: reasons.append("CATALOG_POSITIONS_OVERLAP")
    return {"sector": sector, "role": role, "inEventCadenceCount": int(np.count_nonzero(inside)),
            "controlCadenceCount": int(np.count_nonzero(control)), "eclipseEventCount": event_count,
            "differenceImagePeakSNR": image["peakSNR"], "differenceImage": image,
            "measuredPixelCentroid": {"x": image["centroidX"], "y": image["centroidY"]},
            "centroidUncertaintyPixels": uncertainty, "eventJackknifeCentroids": jackknife,
            "centroidSky": item.get("centroidSky"), "catalogDistances": distances,
            "matchedCatalogHypothesis": matched, "classification": classification,
            "usable": usable and classification != "AMBIGUOUS", "qualityRejectionReasons": reasons,
            "backgroundCorrection": background, "acquisitionProvenance": item.get("acquisitionProvenance"),
            "inputProvenance": item.get("inputProvenance"), "frozenMask": {"periodDays": period,
                "referenceEpoch": epoch, "durationDays": duration, "oppositeConjunctionExcluded": True,
                "phaseOrDurationSearched": False}}


def _production_input(tic_id: int, identity: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    metadata = ((identity.get("tic") or {}).get("metadata") or {})
    ra, dec = float(metadata["raDeg"]), float(metadata["decDeg"])
    tpf, provenance = _download_tpf(tic_id=tic_id, sector=int(frozen["sector"]), ra_deg=ra, dec_deg=dec)
    flux = getattr(tpf.flux, "value", tpf.flux)
    cube = np.ma.filled(flux, np.nan) if np.ma.isMaskedArray(flux) else np.asarray(flux)
    target = _skycoord(ra, dec); x, y = tpf.wcs.world_to_pixel(target)
    hypotheses = [{"id": "TARGET", "isTarget": True, "pixel": {"x": x, "y": y},
                   "sky": {"raDeg": ra, "decDeg": dec}}]
    # Identity collectors may freeze neighboring sources under either name.
    candidates = (identity.get("catalogHypotheses") or identity.get("nearbySources") or
                  (identity.get("gaiaDR3") or {}).get("sources") or [])
    nearest_gaia = ((identity.get("gaiaDR3") or {}).get("nearest") or {}).get("sourceID")
    for candidate in candidates:
        # The nearest Gaia counterpart represents the target already inserted
        # above rather than a distinct blending hypothesis.
        if nearest_gaia is not None and candidate.get("sourceID") == nearest_gaia:
            continue
        sky = candidate.get("sky") or candidate
        cra, cdec = _finite(sky.get("raDeg")), _finite(sky.get("decDeg"))
        if cra is not None and cdec is not None:
            cx, cy = tpf.wcs.world_to_pixel(_skycoord(cra, cdec))
            hypotheses.append({"id": str(candidate.get("id") or candidate.get("sourceID") or "CATALOG_SOURCE"),
                               "isTarget": False, "pixel": {"x": cx, "y": cy}, "sky": sky})
    quality = np.asarray(getattr(tpf, "quality", np.zeros(len(cube))), dtype=np.int64) == 0
    return {"sector": int(frozen["sector"]), "times": np.asarray(tpf.time.value), "fluxCube": cube,
            "qualityMask": quality, "targetPixel": {"x": x, "y": y},
            "catalogHypotheses": hypotheses, "pixelScaleArcsec": _pixel_scale_arcsec(tpf.wcs),
            "acquisitionProvenance": provenance}


def localize_eclipse_events(*, binary_confirmation: dict[str, Any], identity: dict[str, Any],
                            tic_id: int, sector_inputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not authoritative_binary_gate(binary_confirmation):
        raise ValueError("authoritative binary-confirmation-v2 localization gate is not satisfied")
    ephemeris = binary_confirmation["linearEphemeris"]
    supplied = None if sector_inputs is None else {int(item["sector"]): item for item in sector_inputs}
    results, rejections = [], []
    for frozen in binary_confirmation.get("sectorResults") or []:
        if frozen.get("usable") is not True: continue
        sector = int(frozen["sector"])
        try:
            item = _production_input(tic_id, identity, frozen) if supplied is None else supplied[sector]
            results.append(measure_eclipse_sector(item, frozen, ephemeris))
        except TessArchiveTransientError:
            raise
        except Exception as error:
            rejections.append({"sector": sector, "role": frozen.get("role"),
                               "reason": f"{type(error).__name__}: {error}"})
    independent = [item for item in results if item["role"] == "INDEPENDENT" and item["usable"]]
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in independent:
        key = item["matchedCatalogHypothesis"] or item["classification"]
        groups.setdefault(key, []).append(item)
    winner = max(groups, key=lambda key: len(groups[key])) if groups else None
    strong_conflict = any(key != winner and values for key, values in groups.items()) if winner else False
    support = len(groups.get(winner, [])) if winner else 0
    if support >= MIN_INDEPENDENT_SECTORS and not strong_conflict:
        sample = groups[winner][0]
        if sample["classification"] == "TARGET_CONSISTENT": classification = "TARGET_CONSISTENT_ECLIPSE_SOURCE"
        elif sample["classification"] == "CATALOG_CANDIDATE_CONSISTENT": classification = "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE"
        else: classification = "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE"
    elif strong_conflict:
        classification = "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND"
    else:
        classification = "INSUFFICIENT_OR_AMBIGUOUS_LOCALIZATION"
    resolved = classification in {"TARGET_CONSISTENT_ECLIPSE_SOURCE", "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE",
                                  "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE"}
    return {"resultVersion": RESULT_VERSION, "classification": classification,
            "sourceAttributionResolved": resolved, "attributedCatalogHypothesis": winner if resolved else None,
            "usableIndependentSectorCount": len(independent), "requiredIndependentSectorCount": MIN_INDEPENDENT_SECTORS,
            "primarySectorCanSatisfyReplication": False, "sectorResults": results, "sectorRejections": rejections,
            "frozenEphemeris": {"refinedPeriodDays": ephemeris["refinedPeriodDays"],
                                "referenceEpoch": ephemeris["referenceEpoch"],
                                "cycleAssignments": ephemeris.get("cycleAssignments")},
            "pixelDataChangedFrozenEventDefinition": False, "catalogAnswerKeyUsed": False,
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "recommendedNextTest": ("SOURCE_ATTRIBUTION_REVIEW" if resolved else "ADDITIONAL_SPATIAL_EVIDENCE"),
            "binaryConfirmationSHA256": sha256_json(binary_confirmation), "identitySHA256": sha256_json(identity)}
