from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    LOMB_SCARGLE_WORKLOAD_ALIASES,
    _float,
    _load_json,
    _safe,
    _write_json,
)

NARROW_FREQUENCY_HALF_WIDTH_FRACTION = 0.02
TOTAL_FREQUENCIES = 2_048
FREQUENCIES_PER_WORK_UNIT = 2_048
MIN_VALID_PIXELS = 9
MIN_PEAK_POWER = 0.05
MIN_POWER_CONTRAST = 1.5
MIN_PHASE_CONCENTRATION = 0.35
MIN_CLUSTER_WEIGHT_FRACTION = 0.30
CLUSTER_RADIUS_PIXELS = 2.5
CLUSTER_RELATIVE_WEIGHT = 0.20
SOURCE_MATCH_MAX_PIXELS = 1.05
SOURCE_MARGIN_FLOOR_PIXELS = 0.25
CENTROID_UNCERTAINTY_FLOOR_PIXELS = 0.10
JACKKNIFE_GROUPS = 4
MIN_CROSS_SECTOR_SUPPORT = 3


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - float(np.mean(values))
    scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("Frequency-localized pixel series has zero/invalid variance.")
    return values / scale


def _narrow_search(frequency: float) -> dict[str, Any]:
    frequency = float(frequency)
    minimum = frequency * (1.0 - NARROW_FREQUENCY_HALF_WIDTH_FRACTION)
    maximum = frequency * (1.0 + NARROW_FREQUENCY_HALF_WIDTH_FRACTION)
    if minimum <= 0 or maximum <= minimum:
        raise RuntimeError("Invalid v20.17 narrow frequency range.")
    return {
        "minimumFrequency": float(minimum),
        "maximumFrequency": float(maximum),
        "frequencyStep": float((maximum - minimum) / (TOTAL_FREQUENCIES - 1)),
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def _usable_sector_frequency(
    sector_result: dict[str, Any] | None,
    reference_frequency: float,
) -> tuple[float, str]:
    if sector_result:
        result = sector_result.get("frequencyResult") or {}
        candidate = _float(result.get("candidateFrequency"))
        status = str(result.get("periodStatus") or "").upper()
        confidence = str(result.get("periodConfidence") or "none").lower()
        reference_consistent = bool(result.get("referenceConsistent"))
        if (
            candidate is not None
            and candidate > 0
            and status == "RELIABLE"
            and confidence in {"high", "medium"}
            and reference_consistent
        ):
            return float(candidate), "v20.16-sector-refined"
    return float(reference_frequency), "v20.16-reference"


def _complex_coefficients(
    times: np.ndarray,
    residual_cube: np.ndarray,
    valid_pixels: np.ndarray,
    frequency: float,
) -> tuple[np.ndarray, complex]:
    times = np.asarray(times, dtype=np.float64)
    cube = np.asarray(residual_cube, dtype=np.float64)
    valid = np.asarray(valid_pixels, dtype=bool)
    centered = times - float(np.mean(times))
    omega = 2.0 * math.pi * float(frequency)
    design = np.column_stack(
        [
            np.ones(len(times), dtype=np.float64),
            np.sin(omega * centered),
            np.cos(omega * centered),
        ]
    )
    flat = cube.reshape(len(times), -1)
    beta = np.linalg.pinv(design) @ flat
    z = beta[1] + 1j * beta[2]
    z = z.reshape(valid.shape)
    z = np.where(valid, z, 0.0 + 0.0j)

    aperture = np.sum(cube[:, valid], axis=1)
    aperture_beta = np.linalg.lstsq(design, aperture, rcond=None)[0]
    aperture_z = complex(float(aperture_beta[1]), float(aperture_beta[2]))
    if not math.isfinite(abs(aperture_z)) or abs(aperture_z) <= 1e-12:
        raise RuntimeError("Aperture has no finite fixed-frequency complex response.")
    return z, aperture_z


def _response_map(
    *,
    times: np.ndarray,
    residual_cube: np.ndarray,
    valid_pixels: np.ndarray,
    frequency: float,
    power_map: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray(valid_pixels, dtype=bool)
    powers = np.asarray(power_map, dtype=np.float64)
    if powers.shape != valid.shape:
        raise RuntimeError("Pixel power map shape does not match cached TPF pixels.")
    usable = valid & np.isfinite(powers) & (powers >= 0.0)
    if int(np.count_nonzero(usable)) < MIN_VALID_PIXELS:
        raise RuntimeError("Too few usable pixel responses for frequency-localized localization.")

    z, aperture_z = _complex_coefficients(times, residual_cube, valid, float(frequency))
    amplitude = np.abs(z)
    phase_alignment = np.zeros_like(amplitude, dtype=np.float64)
    nonzero = usable & (amplitude > 1e-12)
    phase_alignment[nonzero] = np.real(
        z[nonzero] * np.conjugate(aperture_z)
    ) / (amplitude[nonzero] * abs(aperture_z))
    coherent_amplitude = np.zeros_like(amplitude, dtype=np.float64)
    coherent_amplitude[nonzero] = np.maximum(
        np.real(z[nonzero] * np.conjugate(aperture_z)) / abs(aperture_z),
        0.0,
    )

    peak_power = float(np.max(powers[usable]))
    median_power = float(np.median(powers[usable]))
    power_contrast = peak_power / max(median_power, 1e-12)
    normalized_power = np.zeros_like(powers, dtype=np.float64)
    if peak_power > 0:
        normalized_power[usable] = np.clip(powers[usable] / peak_power, 0.0, 1.0)

    weights = np.where(
        usable,
        coherent_amplitude
        * np.sqrt(np.maximum(normalized_power, 0.0))
        * np.maximum(phase_alignment, 0.0),
        0.0,
    )
    if float(np.sum(weights)) <= 0:
        raise RuntimeError("Frequency-localized coherent response has zero positive weight.")

    yy, xx = np.mgrid[0:weights.shape[0], 0:weights.shape[1]]
    peak_flat = int(np.argmax(weights))
    peak_y, peak_x = np.unravel_index(peak_flat, weights.shape)
    peak_weight = float(weights[peak_y, peak_x])
    radius = np.hypot(xx - float(peak_x), yy - float(peak_y))
    cluster = usable & (radius <= CLUSTER_RADIUS_PIXELS) & (
        weights >= max(peak_weight * CLUSTER_RELATIVE_WEIGHT, 0.0)
    )
    if int(np.count_nonzero(cluster)) < 1:
        cluster = usable & (radius <= 1.5) & (weights > 0)
    cluster_weight = float(np.sum(weights[cluster]))
    total_weight = float(np.sum(weights))
    cluster_fraction = cluster_weight / total_weight if total_weight > 0 else 0.0
    if cluster_weight <= 0:
        raise RuntimeError("Frequency-localized response cluster has zero weight.")

    centroid_x = float(np.sum(weights[cluster] * xx[cluster]) / cluster_weight)
    centroid_y = float(np.sum(weights[cluster] * yy[cluster]) / cluster_weight)

    strong = usable & (normalized_power >= 0.30) & (amplitude > 1e-12)
    if np.any(strong):
        phase_vectors = z[strong] / np.maximum(amplitude[strong], 1e-12)
        phase_weights = amplitude[strong] * np.sqrt(np.maximum(normalized_power[strong], 0.0))
        phase_concentration = float(
            abs(np.sum(phase_weights * phase_vectors)) / max(float(np.sum(phase_weights)), 1e-12)
        )
    else:
        phase_concentration = 0.0

    map_usable = bool(
        peak_power >= MIN_PEAK_POWER
        and power_contrast >= MIN_POWER_CONTRAST
        and phase_concentration >= MIN_PHASE_CONCENTRATION
        and cluster_fraction >= MIN_CLUSTER_WEIGHT_FRACTION
    )
    return {
        "centroidX": centroid_x,
        "centroidY": centroid_y,
        "peakX": int(peak_x),
        "peakY": int(peak_y),
        "peakPower": peak_power,
        "medianPower": median_power,
        "powerContrast": power_contrast,
        "phaseConcentration": phase_concentration,
        "clusterWeightFraction": cluster_fraction,
        "clusterPixelCount": int(np.count_nonzero(cluster)),
        "mapUsable": map_usable,
        "powerMap": np.asarray(powers, dtype=np.float32).tolist(),
        "coherentResponseMap": np.asarray(weights, dtype=np.float32).tolist(),
        "phaseAlignmentMap": np.asarray(phase_alignment, dtype=np.float32).tolist(),
    }


def _jackknife_uncertainty(
    *,
    times: np.ndarray,
    residual_cube: np.ndarray,
    valid_pixels: np.ndarray,
    frequency: float,
    power_map: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    centroids: list[dict[str, float]] = []
    indices = np.arange(len(times), dtype=int)
    for group in range(JACKKNIFE_GROUPS):
        keep = indices % JACKKNIFE_GROUPS != group
        if int(np.count_nonzero(keep)) < max(100, len(times) // 2):
            continue
        try:
            result = _response_map(
                times=np.asarray(times)[keep],
                residual_cube=np.asarray(residual_cube)[keep],
                valid_pixels=valid_pixels,
                frequency=frequency,
                power_map=power_map,
            )
        except Exception:
            continue
        centroids.append({"x": float(result["centroidX"]), "y": float(result["centroidY"])})
    if len(centroids) < 2:
        return CENTROID_UNCERTAINTY_FLOOR_PIXELS, centroids
    mx = statistics.mean(item["x"] for item in centroids)
    my = statistics.mean(item["y"] for item in centroids)
    scatter = math.sqrt(
        statistics.mean(
            (item["x"] - mx) ** 2 + (item["y"] - my) ** 2
            for item in centroids
        )
    )
    return max(CENTROID_UNCERTAINTY_FLOOR_PIXELS, float(scatter)), centroids


def _classify_sector(
    *,
    target_distance_pixels: float,
    counterpart_distance_pixels: float,
    uncertainty_pixels: float,
    map_usable: bool,
) -> str:
    if not map_usable:
        return "NO_QUALITY_LOCALIZATION"
    margin = max(SOURCE_MARGIN_FLOOR_PIXELS, 2.0 * float(uncertainty_pixels))
    if (
        counterpart_distance_pixels <= SOURCE_MATCH_MAX_PIXELS
        and target_distance_pixels - counterpart_distance_pixels >= margin
    ):
        return "COUNTERPART_CONSISTENT"
    if (
        target_distance_pixels <= SOURCE_MATCH_MAX_PIXELS
        and counterpart_distance_pixels - target_distance_pixels >= margin
    ):
        return "TARGET_CONSISTENT"
    return "AMBIGUOUS"


def _classify_cross_sector(sector_results: list[dict[str, Any]]) -> tuple[str, str, str]:
    quality = [item for item in sector_results if item.get("classification") != "NO_QUALITY_LOCALIZATION"]
    counterpart = [item for item in quality if item.get("classification") == "COUNTERPART_CONSISTENT"]
    target = [item for item in quality if item.get("classification") == "TARGET_CONSISTENT"]
    if len(counterpart) >= MIN_CROSS_SECTOR_SUPPORT and len(counterpart) > len(target):
        return (
            "FREQUENCY_LOCALIZED_COUNTERPART_SUPPORTED",
            "CATALOG_COUNTERPART_SUPPORTED_BY_FREQUENCY_LOCALIZED_PIXEL_RESPONSE",
            "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL",
        )
    if len(target) >= MIN_CROSS_SECTOR_SUPPORT and len(target) > len(counterpart):
        return (
            "FREQUENCY_LOCALIZED_TARGET_SUPPORTED",
            "TARGET_SUPPORTED_BY_FREQUENCY_LOCALIZED_PIXEL_RESPONSE",
            "TARGET_INTRINSIC_RESIDUAL_MODELING",
        )
    if len(counterpart) >= 2 and len(target) >= 2:
        return (
            "FREQUENCY_LOCALIZED_MIXED_OR_BLENDED",
            "TARGET_AND_COUNTERPART_FREQUENCY_LOCALIZED_SUPPORT",
            "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL",
        )
    if len(counterpart) >= 2 and len(counterpart) > len(target):
        return (
            "FREQUENCY_LOCALIZED_COUNTERPART_SUGGESTIVE",
            "CATALOG_COUNTERPART_SUGGESTIVE_BY_FREQUENCY_LOCALIZED_PIXEL_RESPONSE",
            "OFFICIAL_SPOC_PRF_FORWARD_MODELING",
        )
    if len(target) >= 2 and len(target) > len(counterpart):
        return (
            "FREQUENCY_LOCALIZED_TARGET_SUGGESTIVE",
            "TARGET_SUGGESTIVE_BY_FREQUENCY_LOCALIZED_PIXEL_RESPONSE",
            "OFFICIAL_SPOC_PRF_FORWARD_MODELING",
        )
    return (
        "FREQUENCY_LOCALIZED_CONFIRMATION_UNRESOLVED",
        "UNRESOLVED_AFTER_FREQUENCY_LOCALIZED_PIXEL_RESPONSE",
        "OFFICIAL_SPOC_PRF_FORWARD_MODELING",
    )


def build_frequency_localized_pixel_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    difference_image_preparation: dict[str, Any],
    difference_image_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.17 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )
    if difference_image_summary.get("recommendedNextTest") != "FREQUENCY_LOCALIZED_PIXEL_RESPONSE_CONFIRMATION":
        raise RuntimeError("v20.17 requires v20.16 to recommend frequency-localized pixel-response confirmation.")

    reference_frequency = _float(difference_image_summary.get("referenceFrequency"))
    if reference_frequency is None or reference_frequency <= 0:
        raise RuntimeError("v20.17 requires the v20.16 reference residual frequency.")
    prior_by_sector = {
        int(item["sector"]): item
        for item in difference_image_summary.get("sectorResults") or []
        if item.get("sector") is not None
    }

    root = Path(output_dir) / "frequency-localized-pixel-response"
    root.mkdir(parents=True, exist_ok=True)
    source_base_id = str(source_dataset_entry.get("id") or "tess-target")
    dataset_entries: list[dict[str, Any]] = []
    prepared_pixels: list[dict[str, Any]] = []
    sectors: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for cache in difference_image_preparation.get("sectorCaches") or []:
        sector = int(cache["sector"])
        try:
            with np.load(cache["cachePath"]) as payload:
                warped_times = np.asarray(payload["warped_times"], dtype=np.float64)
                residual_cube = np.asarray(payload["residual_cube"], dtype=np.float64)
                valid_pixels = np.asarray(payload["valid_pixels"], dtype=bool)
            if len(warped_times) != residual_cube.shape[0]:
                raise RuntimeError("Cached time and pixel-cube lengths differ.")
            frequency, frequency_source = _usable_sector_frequency(
                prior_by_sector.get(sector), float(reference_frequency)
            )
            search = _narrow_search(float(frequency))
            local_times = warped_times - float(np.min(warped_times))
            valid_count = 0
            for row, col in np.argwhere(valid_pixels):
                series = residual_cube[:, int(row), int(col)]
                try:
                    normalized = _normalize(series)
                except Exception:
                    continue
                dataset_id = (
                    f"{source_base_id}-frequency-localized-sector-{sector}-"
                    f"r{int(row)}-c{int(col)}-v1"
                )
                target_name = (
                    f"{source_dataset_entry.get('targetName') or source_base_id} "
                    f"frequency-localized residual sector {sector} pixel ({int(row)},{int(col)})"
                )
                dataset_path = root / f"{_safe(dataset_id)}.json"
                dataset = {
                    "id": dataset_id,
                    "targetName": target_name,
                    "times": np.asarray(local_times, dtype=np.float32).tolist(),
                    "flux": np.asarray(normalized, dtype=np.float32).tolist(),
                    "frequencySearch": search,
                    "reference": {},
                    "science": {
                        "role": "frequency-localized-pixel-response",
                        "sector": int(sector),
                        "row": int(row),
                        "column": int(col),
                        "targetFrequency": float(frequency),
                        "frequencySource": frequency_source,
                    },
                    "source": {
                        "mission": "TESS",
                        "sector": int(sector),
                        "distributedSamples": int(len(local_times)),
                    },
                }
                _write_json(dataset_path, dataset)
                dataset_entries.append(
                    {"id": dataset_id, "path": str(dataset_path.resolve()), "targetName": target_name}
                )
                prepared_pixels.append(
                    {
                        "datasetID": dataset_id,
                        "datasetPath": str(dataset_path.resolve()),
                        "sector": int(sector),
                        "row": int(row),
                        "column": int(col),
                        "targetFrequency": float(frequency),
                        "frequencySource": frequency_source,
                    }
                )
                valid_count += 1
            if valid_count < MIN_VALID_PIXELS:
                raise RuntimeError(f"Only {valid_count} usable pixel datasets were prepared.")
            sectors.append(
                {
                    "sector": int(sector),
                    "role": cache.get("role"),
                    "cachePath": cache.get("cachePath"),
                    "targetPixel": cache.get("targetPixel"),
                    "counterpartPixel": cache.get("counterpartPixel"),
                    "localSkyJacobian": cache.get("localSkyJacobian"),
                    "targetFrequency": float(frequency),
                    "frequencySource": frequency_source,
                    "search": search,
                    "pixelDatasetCount": int(valid_count),
                }
            )
            print(
                f"      sector {sector}: frequency={frequency:.8f} c/d ({frequency_source}), "
                f"pixel datasets={valid_count}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"sector": int(sector), "error": f"{type(exc).__name__}: {exc}"})
            print(f"      sector {sector} unavailable: {type(exc).__name__}: {exc}", flush=True)

    if not dataset_entries or not sectors:
        raise RuntimeError("v20.17 could not prepare any frequency-localized pixel datasets.")

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "frequency-localized-pixel-response-v1"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — frequency-localized pixel response",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "frequency-localized-pixel-response-confirmation",
            "workerSemantics": (
                "Each dataset is one established-family-prewhitened TESS pixel residual series with a narrow "
                "frequency band centered on the sector/reference residual frequency. Workers execute ordinary "
                "Lomb-Scargle and have no source-localization semantics."
            ),
            "referenceFrequency": float(reference_frequency),
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)
    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": True,
        "version": "openstar.tess-frequency-localized-pixel-response-preparation.v1",
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-narrow-band-lomb-scargle-per-pixel",
        "targetTIC": difference_image_summary.get("targetTIC"),
        "catalogCounterpart": difference_image_summary.get("catalogCounterpart"),
        "referenceFrequency": float(reference_frequency),
        "referencePeriodDays": float(1.0 / reference_frequency),
        "preparedPixels": prepared_pixels,
        "sectorPreparations": sectors,
        "errors": errors,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "interpretationGuard": (
            "This stage is a source-localization confirmation, not a new independent period detection. The generic "
            "worker searches only a narrow band around a pre-existing residual frequency; the TESS workflow later "
            "combines distributed pixel power with fixed-frequency phase coherence."
        ),
    }


def interpret_frequency_localized_pixel_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {str(item["datasetID"]): item for item in preparation.get("preparedPixels") or []}
    by_sector_results: dict[int, list[dict[str, Any]]] = {}
    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id)
        if meta is None:
            continue
        power = _float(dataset.get("candidatePower"))
        frequency = _float(dataset.get("candidateFrequency"))
        status = str(dataset.get("periodStatus") or "").upper()
        confidence = str(dataset.get("periodConfidence") or "none").lower()
        by_sector_results.setdefault(int(meta["sector"]), []).append(
            {
                **meta,
                "candidatePower": power,
                "candidateFrequency": frequency,
                "periodStatus": status,
                "periodConfidence": confidence,
            }
        )

    sector_results: list[dict[str, Any]] = []
    analysis_errors: list[dict[str, Any]] = []
    for sector_meta in preparation.get("sectorPreparations") or []:
        sector = int(sector_meta["sector"])
        try:
            with np.load(sector_meta["cachePath"]) as payload:
                warped_times = np.asarray(payload["warped_times"], dtype=np.float64)
                residual_cube = np.asarray(payload["residual_cube"], dtype=np.float64)
                valid_pixels = np.asarray(payload["valid_pixels"], dtype=bool)
            powers = np.full(valid_pixels.shape, np.nan, dtype=np.float64)
            frequencies = np.full(valid_pixels.shape, np.nan, dtype=np.float64)
            reliable_pixels = 0
            for item in by_sector_results.get(sector, []):
                row = int(item["row"])
                col = int(item["column"])
                power = _float(item.get("candidatePower"))
                frequency = _float(item.get("candidateFrequency"))
                if power is not None:
                    powers[row, col] = float(power)
                if frequency is not None:
                    frequencies[row, col] = float(frequency)
                if item.get("periodStatus") == "RELIABLE" and item.get("periodConfidence") in {"high", "medium"}:
                    reliable_pixels += 1
            response = _response_map(
                times=warped_times,
                residual_cube=residual_cube,
                valid_pixels=valid_pixels,
                frequency=float(sector_meta["targetFrequency"]),
                power_map=powers,
            )
            uncertainty, jackknife = _jackknife_uncertainty(
                times=warped_times,
                residual_cube=residual_cube,
                valid_pixels=valid_pixels,
                frequency=float(sector_meta["targetFrequency"]),
                power_map=powers,
            )
            target = sector_meta.get("targetPixel") or {}
            counterpart = sector_meta.get("counterpartPixel") or {}
            tx, ty = float(target["x"]), float(target["y"])
            cx, cy = float(counterpart["x"]), float(counterpart["y"])
            target_distance = float(math.hypot(response["centroidX"] - tx, response["centroidY"] - ty))
            counterpart_distance = float(math.hypot(response["centroidX"] - cx, response["centroidY"] - cy))
            classification = _classify_sector(
                target_distance_pixels=target_distance,
                counterpart_distance_pixels=counterpart_distance,
                uncertainty_pixels=float(uncertainty),
                map_usable=bool(response["mapUsable"]),
            )
            finite_freq = frequencies[np.isfinite(frequencies)]
            sector_results.append(
                {
                    "sector": sector,
                    "role": sector_meta.get("role"),
                    "targetFrequency": sector_meta.get("targetFrequency"),
                    "frequencySource": sector_meta.get("frequencySource"),
                    "pixelDatasetCount": sector_meta.get("pixelDatasetCount"),
                    "reliablePixelCount": int(reliable_pixels),
                    "medianDistributedPixelFrequency": (
                        float(np.median(finite_freq)) if len(finite_freq) else None
                    ),
                    "response": response,
                    "centroidUncertaintyPixels": float(uncertainty),
                    "jackknifeCentroids": jackknife,
                    "targetDistancePixels": target_distance,
                    "counterpartDistancePixels": counterpart_distance,
                    "classification": classification,
                }
            )
        except Exception as exc:
            analysis_errors.append({"sector": sector, "error": f"{type(exc).__name__}: {exc}"})

    quality = [item for item in sector_results if item.get("classification") != "NO_QUALITY_LOCALIZATION"]
    counterpart = [item for item in quality if item.get("classification") == "COUNTERPART_CONSISTENT"]
    target = [item for item in quality if item.get("classification") == "TARGET_CONSISTENT"]
    ambiguous = [item for item in quality if item.get("classification") == "AMBIGUOUS"]
    classification, origin, next_test = _classify_cross_sector(sector_results)

    return {
        "version": "openstar.tess-frequency-localized-pixel-response.v1",
        "distributedPixelSearch": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "datasetCount": len(preparation.get("preparedPixels") or []),
            "frequencyHalfWidthFraction": NARROW_FREQUENCY_HALF_WIDTH_FRACTION,
        },
        "targetTIC": preparation.get("targetTIC"),
        "catalogCounterpart": preparation.get("catalogCounterpart"),
        "referenceFrequency": preparation.get("referenceFrequency"),
        "referencePeriodDays": preparation.get("referencePeriodDays"),
        "sectorResults": sector_results,
        "qualitySectorCount": len(quality),
        "counterpartSupportingSectors": sorted(int(item["sector"]) for item in counterpart),
        "targetSupportingSectors": sorted(int(item["sector"]) for item in target),
        "ambiguousSectors": sorted(int(item["sector"]) for item in ambiguous),
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "preparationErrors": preparation.get("errors") or [],
        "analysisErrors": analysis_errors,
        "interpretationGuard": (
            "Frequency-localized pixel-response attribution requires recurrent independent-sector spatial support. "
            "The narrow-band searches are conditioned on an already established residual family and therefore do not "
            "constitute independent period discovery or modify the v20.6 target association of the main 13.72-day family."
        ),
    }
