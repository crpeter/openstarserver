from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_sector_archive import TessArchiveTransientError
from .tess_difference_image_constants import (MIN_IMAGE_PEAK_SNR,
    SOURCE_MATCH_MAX_PIXELS, SOURCE_MARGIN_FLOOR_PIXELS)

from .tess_multisource_residual import MIN_COMPONENT_SAMPLES, _prewhiten_cube_raw
from .tess_offset_variability import (
    MIN_CANDIDATE_POWER,
    MIN_OBSERVED_CYCLES,
    MIN_PEAK_PROMINENCE,
    REFERENCE_FREQUENCY_TOLERANCE_FRACTION,
    TOTAL_FREQUENCIES,
    FREQUENCIES_PER_WORK_UNIT,
    _best_offset_summary,
    _boundary_hit,
    _catalog_candidate,
    _frequency_search,
    _skycoord,
)
from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    LOMB_SCARGLE_WORKLOAD_ALIASES,
    MAX_CADENCES,
    _background_subtract_cube,
    _download_tpf,
    _float,
    _int,
    _load_json,
    _local_sky_jacobian,
    _safe,
    _sector_candidates,
    _time_warp,
    _uniform_indices,
    _write_json,
)

PHASE_EXTREME_FRACTION = 0.25
MIN_PHASE_BIN_CADENCES = 80
CLUSTER_RELATIVE_SNR = 0.30
CLUSTER_MIN_SNR = 2.0
CLUSTER_RADIUS_PIXELS = 2.5
CENTROID_UNCERTAINTY_FLOOR_PIXELS = 0.12
MIN_CROSS_SECTOR_SUPPORT = 3
JACKKNIFE_GROUPS = 4


def _normalize_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - float(np.mean(values))
    scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("Difference-image aperture residual has zero/invalid variance.")
    return values / scale


def _phase_model(times: np.ndarray, flux: np.ndarray, frequency: float) -> dict[str, Any]:
    times = np.asarray(times, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    centered = times - float(np.mean(times))
    omega = 2.0 * math.pi * float(frequency)
    design = np.column_stack(
        [
            np.ones(len(times), dtype=np.float64),
            np.sin(omega * centered),
            np.cos(omega * centered),
        ]
    )
    beta, *_ = np.linalg.lstsq(design, flux, rcond=None)
    model = design[:, 1:] @ beta[1:]
    residual = flux - design @ beta
    total_var = float(np.var(flux))
    residual_var = float(np.var(residual))
    explained = 1.0 - residual_var / total_var if total_var > 1e-12 else 0.0
    return {
        "model": model,
        "amplitude": float(math.hypot(float(beta[1]), float(beta[2]))),
        "phaseRadians": float(math.atan2(float(beta[2]), float(beta[1]))),
        "explainedVariance": float(explained),
    }


def _extreme_indices(model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = np.asarray(model, dtype=np.float64)
    count = len(model)
    take = max(MIN_PHASE_BIN_CADENCES, int(math.floor(count * PHASE_EXTREME_FRACTION)))
    take = min(take, count // 2)
    if take < MIN_PHASE_BIN_CADENCES:
        raise RuntimeError("Too few cadences for high/low residual difference-image bins.")
    order = np.argsort(model)
    low = np.sort(order[:take])
    high = np.sort(order[-take:])
    return high, low


def _centroid_from_frames(
    residual_cube: np.ndarray,
    valid_pixels: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> dict[str, Any]:
    cube = np.asarray(residual_cube, dtype=np.float64)
    valid_pixels = np.asarray(valid_pixels, dtype=bool)
    if len(high) < 2 or len(low) < 2:
        raise RuntimeError("Difference image requires at least two cadences per phase bin.")

    high_frames = cube[high]
    low_frames = cube[low]
    difference = np.nanmean(high_frames, axis=0) - np.nanmean(low_frames, axis=0)
    variance = (
        np.nanvar(high_frames, axis=0, ddof=1) / max(len(high), 1)
        + np.nanvar(low_frames, axis=0, ddof=1) / max(len(low), 1)
    )
    noise = np.sqrt(np.maximum(variance, 0.0))
    snr = np.divide(
        difference,
        noise,
        out=np.zeros_like(difference),
        where=np.isfinite(noise) & (noise > 1e-12),
    )
    difference = np.where(np.isfinite(difference) & valid_pixels, difference, 0.0)
    snr = np.where(np.isfinite(snr) & valid_pixels, snr, 0.0)

    if not np.any(valid_pixels):
        raise RuntimeError("No valid pixels remain for difference-image localization.")
    peak_flat = int(np.argmax(np.where(valid_pixels, snr, -np.inf)))
    peak_y, peak_x = np.unravel_index(peak_flat, snr.shape)
    peak_snr = float(snr[peak_y, peak_x])
    if not math.isfinite(peak_snr) or peak_snr <= 0:
        raise RuntimeError("Difference image has no positive residual source peak.")

    yy, xx = np.mgrid[0:snr.shape[0], 0:snr.shape[1]]
    radius = np.hypot(xx - float(peak_x), yy - float(peak_y))
    threshold = max(CLUSTER_MIN_SNR, peak_snr * CLUSTER_RELATIVE_SNR)
    centered_difference = difference - float(np.median(difference[valid_pixels]))
    cluster = (
        valid_pixels
        & (radius <= CLUSTER_RADIUS_PIXELS)
        & (snr >= threshold)
        & (centered_difference > 0)
    )
    if int(np.count_nonzero(cluster)) < 1:
        cluster = valid_pixels & (radius <= 1.5) & (centered_difference > 0)
    weights = np.where(cluster, np.maximum(centered_difference, 0.0) * np.maximum(snr, 0.0), 0.0)
    weight_sum = float(np.sum(weights))
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise RuntimeError("Difference-image source cluster has zero positive weight.")

    centroid_x = float(np.sum(weights * xx) / weight_sum)
    centroid_y = float(np.sum(weights * yy) / weight_sum)
    integrated_snr = float(math.sqrt(float(np.sum(np.square(np.maximum(snr[cluster], 0.0))))))
    return {
        "centroidX": centroid_x,
        "centroidY": centroid_y,
        "peakX": int(peak_x),
        "peakY": int(peak_y),
        "peakSNR": peak_snr,
        "integratedSNR": integrated_snr,
        "clusterPixelCount": int(np.count_nonzero(cluster)),
        "differenceImage": np.asarray(difference, dtype=np.float32).tolist(),
        "snrImage": np.asarray(snr, dtype=np.float32).tolist(),
    }


def _jackknife_uncertainty(
    residual_cube: np.ndarray,
    valid_pixels: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    centroids: list[dict[str, float]] = []
    for group in range(JACKKNIFE_GROUPS):
        high_keep = np.asarray([value for index, value in enumerate(high) if index % JACKKNIFE_GROUPS != group], dtype=int)
        low_keep = np.asarray([value for index, value in enumerate(low) if index % JACKKNIFE_GROUPS != group], dtype=int)
        if len(high_keep) < MIN_PHASE_BIN_CADENCES // 2 or len(low_keep) < MIN_PHASE_BIN_CADENCES // 2:
            continue
        try:
            result = _centroid_from_frames(residual_cube, valid_pixels, high_keep, low_keep)
        except Exception:
            continue
        centroids.append({"x": float(result["centroidX"]), "y": float(result["centroidY"])})
    if len(centroids) < 2:
        return CENTROID_UNCERTAINTY_FLOOR_PIXELS, centroids
    xs = [item["x"] for item in centroids]
    ys = [item["y"] for item in centroids]
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    scatter = math.sqrt(statistics.mean((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)))
    return max(CENTROID_UNCERTAINTY_FLOOR_PIXELS, float(scatter)), centroids


def _classify_sector_localization(
    *,
    target_distance_pixels: float,
    counterpart_distance_pixels: float,
    uncertainty_pixels: float,
    image_usable: bool,
    frequency_accepted: bool,
) -> str:
    if not image_usable or not frequency_accepted:
        return "NO_QUALITY_LOCALIZATION"
    margin = max(SOURCE_MARGIN_FLOOR_PIXELS, 2.0 * float(uncertainty_pixels))
    if (
        float(counterpart_distance_pixels) <= SOURCE_MATCH_MAX_PIXELS
        and float(target_distance_pixels) - float(counterpart_distance_pixels) >= margin
    ):
        return "COUNTERPART_CONSISTENT"
    if (
        float(target_distance_pixels) <= SOURCE_MATCH_MAX_PIXELS
        and float(counterpart_distance_pixels) - float(target_distance_pixels) >= margin
    ):
        return "TARGET_CONSISTENT"
    return "AMBIGUOUS"


def _classify_cross_sector(sector_results: list[dict[str, Any]]) -> tuple[str, str, str]:
    quality = [item for item in sector_results if item.get("classification") != "NO_QUALITY_LOCALIZATION"]
    counterpart_supported = [item for item in quality if item.get("classification") == "COUNTERPART_CONSISTENT"]
    target_supported = [item for item in quality if item.get("classification") == "TARGET_CONSISTENT"]
    if (
        len(quality) >= MIN_CROSS_SECTOR_SUPPORT
        and len(counterpart_supported) >= MIN_CROSS_SECTOR_SUPPORT
        and len(counterpart_supported) > len(target_supported)
    ):
        return (
            "DIFFERENCE_IMAGE_COUNTERPART_SUPPORTED",
            "CATALOG_COUNTERPART_SUPPORTED_BY_DIFFERENCE_IMAGES",
            "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL",
        )
    if (
        len(quality) >= MIN_CROSS_SECTOR_SUPPORT
        and len(target_supported) >= MIN_CROSS_SECTOR_SUPPORT
        and len(target_supported) > len(counterpart_supported)
    ):
        return (
            "DIFFERENCE_IMAGE_TARGET_SUPPORTED",
            "TARGET_SUPPORTED_BY_DIFFERENCE_IMAGES",
            "TARGET_INTRINSIC_RESIDUAL_MODELING",
        )
    if len(counterpart_supported) >= 2 and len(target_supported) >= 2:
        return (
            "DIFFERENCE_IMAGE_MIXED_OR_BLENDED",
            "TARGET_AND_COUNTERPART_DIFFERENCE_IMAGE_SUPPORT",
            "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL",
        )
    if len(counterpart_supported) >= 2 and len(counterpart_supported) > len(target_supported):
        return (
            "DIFFERENCE_IMAGE_COUNTERPART_SUGGESTIVE",
            "CATALOG_COUNTERPART_SUGGESTIVE_BY_DIFFERENCE_IMAGES",
            "FREQUENCY_LOCALIZED_PIXEL_RESPONSE_CONFIRMATION",
        )
    return (
        "DIFFERENCE_IMAGE_LOCALIZATION_UNRESOLVED",
        "UNRESOLVED_AFTER_DIFFERENCE_IMAGING",
        "FREQUENCY_LOCALIZED_PIXEL_RESPONSE_CONFIRMATION",
    )


def _pixel_to_sky_offsets(
    *,
    centroid_x: float,
    centroid_y: float,
    target_x: float,
    target_y: float,
    jacobian: dict[str, Any],
) -> tuple[float | None, float | None]:
    values = [
        _float(jacobian.get("xToEastArcsec")),
        _float(jacobian.get("xToNorthArcsec")),
        _float(jacobian.get("yToEastArcsec")),
        _float(jacobian.get("yToNorthArcsec")),
    ]
    if any(value is None for value in values):
        return None, None
    x_east, x_north, y_east, y_north = [float(value) for value in values]
    dx = float(centroid_x) - float(target_x)
    dy = float(centroid_y) - float(target_y)
    return dx * x_east + dy * y_east, dx * x_north + dy * y_north


def _frequency_result(dataset: dict[str, Any], meta: dict[str, Any], preparation: dict[str, Any]) -> dict[str, Any]:
    frequency = _float(dataset.get("candidateFrequency"))
    period = _float(dataset.get("candidatePeriodDays"))
    power = _float(dataset.get("candidatePower"))
    prominence = _float(dataset.get("candidatePeakProminenceRatio"))
    status = str(dataset.get("periodStatus") or "").upper()
    confidence = str(dataset.get("periodConfidence") or "none").lower()
    baseline = _float(meta.get("baselineDays")) or 0.0
    observed_cycles = baseline / period if period and period > 0 else 0.0
    reference = float(preparation["referenceFrequency"])
    relative = abs(float(frequency) - reference) / reference if frequency is not None else None
    rayleigh = 1.0 / baseline if baseline > 0 else None
    reference_consistent = bool(
        frequency is not None
        and (
            (relative is not None and relative <= REFERENCE_FREQUENCY_TOLERANCE_FRACTION)
            or (rayleigh is not None and abs(float(frequency) - reference) <= rayleigh)
        )
    )
    accepted = bool(
        status == "RELIABLE"
        and confidence in {"high", "medium"}
        and power is not None
        and power >= MIN_CANDIDATE_POWER
        and prominence is not None
        and prominence >= MIN_PEAK_PROMINENCE
        and observed_cycles >= MIN_OBSERVED_CYCLES
        and not _boundary_hit(frequency, preparation.get("frequencySearch") or {})
        and reference_consistent
    )
    return {
        **meta,
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "periodStatus": status,
        "periodConfidence": confidence,
        "observedCycles": observed_cycles,
        "relativeFrequencyDifferenceFromReference": relative,
        "referenceConsistent": reference_consistent,
        "acceptedFrequencyRefinement": accepted,
    }


def build_difference_image_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    target_tic_id: int,
    identity: dict[str, Any],
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    nonstationary_summary: dict[str, Any],
    multisource_summary: dict[str, Any],
    offset_source_identification: dict[str, Any],
    calibrated_prf_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.16 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )
    if calibrated_prf_summary.get("recommendedNextTest") != "DIFFERENCE_IMAGE_SOURCE_LOCALIZATION":
        raise RuntimeError("v20.16 requires v20.15 to recommend difference-image source localization.")

    best_offset = _best_offset_summary(multisource_summary)
    reference_frequency = _float(best_offset.get("combinedFrequency"))
    if reference_frequency is None or reference_frequency <= 0:
        reference_frequency = _float(calibrated_prf_summary.get("referenceFrequency"))
    q = _float(nonstationary_summary.get("fractionalFrequencyDriftPerDay"))
    time_reference = _float(nonstationary_summary.get("timeReferenceDays"))
    if reference_frequency is None or reference_frequency <= 0 or q is None or time_reference is None:
        raise RuntimeError("v20.16 requires the v20.9 drift model and v20.12/v20.15 residual frequency.")

    target_meta = ((identity.get("tic") or {}).get("metadata") or {})
    target_ra = _float(target_meta.get("raDeg"))
    target_dec = _float(target_meta.get("decDeg"))
    if target_ra is None or target_dec is None:
        raise RuntimeError("v20.16 requires TIC RA/Dec from the identity stage.")
    candidate = _catalog_candidate(offset_source_identification)
    candidate_ra = float(candidate["raDeg"])
    candidate_dec = float(candidate["decDeg"])
    candidate_ids = candidate.get("catalogIDs") or {}

    signal_sectors = [
        int(value)
        for value in ((nonstationary_summary.get("preferredModel") or {}).get("signalSectors") or [])
        if _int(value) is not None
    ]
    sectors = _sector_candidates(
        primary_sector=primary_sector,
        independent_spec=independent_spec,
        signal_sectors=signal_sectors,
    )
    if not sectors:
        raise RuntimeError("v20.16 found no frozen signal sectors for difference imaging.")

    physical_frequency = 1.0 / float(physical_period_days)
    search = _frequency_search(float(reference_frequency))
    target_sky = _skycoord(float(target_ra), float(target_dec))
    candidate_sky = _skycoord(float(candidate_ra), float(candidate_dec))
    root = Path(output_dir) / "difference-image-localization"
    root.mkdir(parents=True, exist_ok=True)
    source_base_id = str(source_dataset_entry.get("id") or f"tic-{target_tic_id}")

    dataset_entries: list[dict[str, Any]] = []
    prepared_series: list[dict[str, Any]] = []
    sector_caches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    combined_times: list[np.ndarray] = []
    combined_flux: list[np.ndarray] = []

    for sector_index, (sector, role) in enumerate(sectors, start=1):
        print(
            f"   Sector {sector} ({sector_index}/{len(sectors)}): preparing residual aperture + difference-image cache",
            flush=True,
        )
        try:
            tpf, source = _download_tpf(
                tic_id=int(target_tic_id),
                sector=int(sector),
                ra_deg=float(target_ra),
                dec_deg=float(target_dec),
            )
            absolute_times = np.asarray(tpf.time.value, dtype=np.float64)
            flux = getattr(tpf.flux, "value", tpf.flux)
            if np.ma.isMaskedArray(flux):
                flux = np.ma.filled(flux, np.nan)
            cube = np.asarray(flux, dtype=np.float64)
            keep = np.isfinite(absolute_times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
            absolute_times = absolute_times[keep]
            cube = cube[keep]
            if len(absolute_times) < MIN_COMPONENT_SAMPLES:
                raise RuntimeError(f"Only {len(absolute_times)} usable cadences.")

            indices = _uniform_indices(len(absolute_times), MAX_CADENCES)
            absolute_times = absolute_times[indices]
            cube = cube[indices]
            corrected, _ = _background_subtract_cube(cube)
            residual_cube, valid_pixels = _prewhiten_cube_raw(
                absolute_times=absolute_times,
                cube=corrected,
                physical_frequency=physical_frequency,
            )
            aperture = np.sum(residual_cube[:, valid_pixels], axis=1)
            aperture = _normalize_series(aperture)
            relative_times = absolute_times - float(time_reference)
            warped = _time_warp(relative_times, float(q))
            local_times = warped - float(np.min(warped))

            target_x, target_y = tpf.wcs.world_to_pixel(target_sky)
            candidate_x, candidate_y = tpf.wcs.world_to_pixel(candidate_sky)
            target_x = float(target_x)
            target_y = float(target_y)
            candidate_x = float(candidate_x)
            candidate_y = float(candidate_y)
            jacobian = _local_sky_jacobian(tpf.wcs, target_sky, target_x, target_y)

            cache_path = root / f"sector-{sector}-difference-image-cache-v1.npz"
            np.savez_compressed(
                cache_path,
                absolute_times=np.asarray(absolute_times, dtype=np.float64),
                warped_times=np.asarray(warped, dtype=np.float64),
                aperture=np.asarray(aperture, dtype=np.float32),
                residual_cube=np.asarray(residual_cube, dtype=np.float32),
                valid_pixels=np.asarray(valid_pixels, dtype=np.uint8),
            )
            dataset_id = f"{source_base_id}-difference-image-frequency-sector-{sector}-v1"
            target_name = (
                f"{source_dataset_entry.get('targetName') or source_base_id} "
                f"difference-image residual frequency sector {sector}"
            )
            dataset_path = root / f"{_safe(dataset_id)}.json"
            dataset = {
                "id": dataset_id,
                "targetName": target_name,
                "times": np.asarray(local_times, dtype=np.float32).tolist(),
                "flux": np.asarray(aperture, dtype=np.float32).tolist(),
                "frequencySearch": search,
                "reference": {},
                "science": {
                    "role": "difference-image-frequency-refinement",
                    "referenceFrequency": float(reference_frequency),
                    "fractionalFrequencyDriftPerDay": float(q),
                    "sector": int(sector),
                },
                "source": {
                    "mission": "TESS",
                    "sector": int(sector),
                    "distributedSamples": int(len(local_times)),
                    "baselineDays": float(np.max(absolute_times) - np.min(absolute_times)),
                    "timeReferenceDays": float(time_reference),
                    "sourceType": source.get("sourceType"),
                    "author": source.get("author"),
                    "cadenceSeconds": source.get("cadenceSeconds"),
                },
            }
            _write_json(dataset_path, dataset)
            dataset_entries.append({"id": dataset_id, "path": str(dataset_path.resolve()), "targetName": target_name})
            prepared_series.append(
                {
                    "datasetID": dataset_id,
                    "datasetPath": str(dataset_path.resolve()),
                    "sector": int(sector),
                    "role": role,
                    "combined": False,
                    "baselineDays": float(np.max(absolute_times) - np.min(absolute_times)),
                }
            )
            sector_caches.append(
                {
                    "sector": int(sector),
                    "role": role,
                    "cachePath": str(cache_path.resolve()),
                    "sourceType": source.get("sourceType"),
                    "author": source.get("author"),
                    "cadenceSeconds": source.get("cadenceSeconds"),
                    "targetPixel": {"x": target_x, "y": target_y},
                    "counterpartPixel": {"x": candidate_x, "y": candidate_y},
                    "localSkyJacobian": jacobian,
                }
            )
            combined_times.append(np.asarray(warped, dtype=np.float64))
            combined_flux.append(np.asarray(aperture, dtype=np.float64))
            print(
                f"      cached {len(absolute_times)} cadences; target/counterpart separation="
                f"{math.hypot(candidate_x-target_x, candidate_y-target_y):.3f} px",
                flush=True,
            )
        except TessArchiveTransientError:
            raise
        except Exception as exc:
            errors.append({"sector": int(sector), "error": f"{type(exc).__name__}: {exc}"})
            print(f"      unavailable: {type(exc).__name__}: {exc}", flush=True)

    if len(combined_times) >= 2:
        all_times = np.concatenate(combined_times)
        all_flux = np.concatenate(combined_flux)
        order = np.argsort(all_times)
        all_times = all_times[order]
        all_flux = all_flux[order]
        local_times = all_times - float(np.min(all_times))
        dataset_id = f"{source_base_id}-difference-image-frequency-combined-v1"
        target_name = f"{source_dataset_entry.get('targetName') or source_base_id} difference-image residual frequency combined"
        dataset_path = root / f"{_safe(dataset_id)}.json"
        dataset = {
            "id": dataset_id,
            "targetName": target_name,
            "times": np.asarray(local_times, dtype=np.float32).tolist(),
            "flux": np.asarray(_normalize_series(all_flux), dtype=np.float32).tolist(),
            "frequencySearch": search,
            "reference": {},
            "science": {
                "role": "difference-image-frequency-refinement-combined",
                "referenceFrequency": float(reference_frequency),
                "fractionalFrequencyDriftPerDay": float(q),
            },
            "source": {
                "mission": "TESS",
                "distributedSamples": int(len(local_times)),
                "baselineDays": float(np.max(all_times) - np.min(all_times)),
                "timeReferenceDays": float(time_reference),
                "combinedSectors": True,
            },
        }
        _write_json(dataset_path, dataset)
        dataset_entries.append({"id": dataset_id, "path": str(dataset_path.resolve()), "targetName": target_name})
        prepared_series.append(
            {
                "datasetID": dataset_id,
                "datasetPath": str(dataset_path.resolve()),
                "sector": None,
                "role": "combined",
                "combined": True,
                "baselineDays": float(np.max(all_times) - np.min(all_times)),
            }
        )

    if not dataset_entries or not sector_caches:
        raise RuntimeError("v20.16 could not prepare any difference-image frequency datasets.")

    project_id = f"{source_project['id']}.investigation.{_safe(investigation_id)}.difference-image-source-localization-v1"
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — difference-image source localization",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "difference-image-source-localization-frequency-refinement",
            "workerSemantics": (
                "Each dataset is a drift-corrected residual aperture light curve used only to refine the residual frequency. "
                "Workers execute ordinary Lomb-Scargle. TESS difference images are constructed by the workflow afterward."
            ),
            "referenceFrequency": float(reference_frequency),
            "fractionalFrequencyDriftPerDay": float(q),
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)
    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": True,
        "version": "openstar.tess-difference-image-localization-preparation.v1",
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-frequency-refinement-before-domain-specific-difference-imaging",
        "targetTIC": int(target_tic_id),
        "targetCoordinate": {"raDeg": float(target_ra), "decDeg": float(target_dec)},
        "catalogCounterpart": {
            "ticID": _int(candidate_ids.get("ticID")),
            "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
            "raDeg": float(candidate_ra),
            "decDeg": float(candidate_dec),
            "catalogSeparationArcsec": _float(candidate.get("separationArcsec")),
        },
        "referenceFrequency": float(reference_frequency),
        "referencePeriodDays": float(1.0 / reference_frequency),
        "fractionalFrequencyDriftPerDay": float(q),
        "timeReferenceDays": float(time_reference),
        "frequencySearch": search,
        "preparedSeries": prepared_series,
        "sectorCaches": sector_caches,
        "errors": errors,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "interpretationGuard": (
            "v20.16 uses distributed Lomb-Scargle only to refine the drift-corrected residual frequency. "
            "Localization is then performed from high-minus-low residual TESS images, an observable independent "
            "of v20.14 Gaussian and v20.15 empirical-ePRF source-amplitude deblending."
        ),
    }


def interpret_difference_image_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {str(item.get("datasetID")): item for item in preparation.get("preparedSeries") or []}
    frequency_results: list[dict[str, Any]] = []
    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id)
        if meta is None:
            continue
        frequency_results.append(_frequency_result(dataset, meta, preparation))
    by_sector = {
        int(item["sector"]): item
        for item in frequency_results
        if item.get("sector") is not None and not item.get("combined")
    }
    combined = next((item for item in frequency_results if item.get("combined")), None)

    target_coord = preparation.get("targetCoordinate") or {}
    counterpart = preparation.get("catalogCounterpart") or {}
    target_sky = _skycoord(float(target_coord["raDeg"]), float(target_coord["decDeg"]))
    counterpart_sky = _skycoord(float(counterpart["raDeg"]), float(counterpart["decDeg"]))
    from astropy import units as u

    sector_results: list[dict[str, Any]] = []
    analysis_errors: list[dict[str, Any]] = []
    for cache in preparation.get("sectorCaches") or []:
        sector = int(cache["sector"])
        frequency_result = by_sector.get(sector)
        if frequency_result is None:
            analysis_errors.append({"sector": sector, "error": "No distributed frequency result."})
            continue
        frequency = _float(frequency_result.get("candidateFrequency"))
        if frequency is None or frequency <= 0:
            analysis_errors.append({"sector": sector, "error": "No finite distributed candidate frequency."})
            continue
        try:
            with np.load(cache["cachePath"]) as payload:
                warped_times = np.asarray(payload["warped_times"], dtype=np.float64)
                aperture = np.asarray(payload["aperture"], dtype=np.float64)
                residual_cube = np.asarray(payload["residual_cube"], dtype=np.float64)
                valid_pixels = np.asarray(payload["valid_pixels"], dtype=bool)
            phase = _phase_model(warped_times, aperture, float(frequency))
            high, low = _extreme_indices(np.asarray(phase["model"], dtype=np.float64))
            image = _centroid_from_frames(residual_cube, valid_pixels, high, low)
            uncertainty, jackknife = _jackknife_uncertainty(residual_cube, valid_pixels, high, low)

            target_pixel = cache["targetPixel"]
            counterpart_pixel = cache["counterpartPixel"]
            tx = float(target_pixel["x"])
            ty = float(target_pixel["y"])
            cx = float(counterpart_pixel["x"])
            cy = float(counterpart_pixel["y"])
            dx_t = float(image["centroidX"]) - tx
            dy_t = float(image["centroidY"]) - ty
            dx_c = float(image["centroidX"]) - cx
            dy_c = float(image["centroidY"]) - cy
            target_distance_pixels = float(math.hypot(dx_t, dy_t))
            counterpart_distance_pixels = float(math.hypot(dx_c, dy_c))
            east, north = _pixel_to_sky_offsets(
                centroid_x=float(image["centroidX"]),
                centroid_y=float(image["centroidY"]),
                target_x=tx,
                target_y=ty,
                jacobian=cache.get("localSkyJacobian") or {},
            )
            centroid_sky = None
            target_sep_arcsec = None
            counterpart_sep_arcsec = None
            if east is not None and north is not None:
                centroid_sky = target_sky.spherical_offsets_by(float(east) * u.arcsec, float(north) * u.arcsec)
                target_sep_arcsec = float(target_sky.separation(centroid_sky).arcsec)
                counterpart_sep_arcsec = float(counterpart_sky.separation(centroid_sky).arcsec)

            image_usable = bool(float(image["peakSNR"]) >= MIN_IMAGE_PEAK_SNR)
            frequency_accepted = bool(frequency_result.get("acceptedFrequencyRefinement"))
            classification = _classify_sector_localization(
                target_distance_pixels=target_distance_pixels,
                counterpart_distance_pixels=counterpart_distance_pixels,
                uncertainty_pixels=float(uncertainty),
                image_usable=image_usable,
                frequency_accepted=frequency_accepted,
            )

            sector_results.append(
                {
                    "sector": sector,
                    "role": cache.get("role"),
                    "frequencyResult": frequency_result,
                    "phaseModel": {
                        "amplitude": phase["amplitude"],
                        "phaseRadians": phase["phaseRadians"],
                        "explainedVariance": phase["explainedVariance"],
                        "highCadences": int(len(high)),
                        "lowCadences": int(len(low)),
                    },
                    "differenceImage": image,
                    "centroidUncertaintyPixels": float(uncertainty),
                    "jackknifeCentroids": jackknife,
                    "targetDistancePixels": target_distance_pixels,
                    "counterpartDistancePixels": counterpart_distance_pixels,
                    "targetSeparationArcsec": target_sep_arcsec,
                    "counterpartSeparationArcsec": counterpart_sep_arcsec,
                    "centroidSky": (
                        {"raDeg": float(centroid_sky.ra.deg), "decDeg": float(centroid_sky.dec.deg)}
                        if centroid_sky is not None
                        else None
                    ),
                    "differenceImageUsable": image_usable,
                    "classification": classification,
                }
            )
        except Exception as exc:
            analysis_errors.append({"sector": sector, "error": f"{type(exc).__name__}: {exc}"})

    quality = [item for item in sector_results if item.get("classification") != "NO_QUALITY_LOCALIZATION"]
    counterpart_supported = [item for item in quality if item.get("classification") == "COUNTERPART_CONSISTENT"]
    target_supported = [item for item in quality if item.get("classification") == "TARGET_CONSISTENT"]
    ambiguous = [item for item in quality if item.get("classification") == "AMBIGUOUS"]

    classification, origin, next_test = _classify_cross_sector(sector_results)

    counterpart_separations = [
        float(item["counterpartSeparationArcsec"])
        for item in counterpart_supported
        if _float(item.get("counterpartSeparationArcsec")) is not None
    ]
    target_separations = [
        float(item["targetSeparationArcsec"])
        for item in target_supported
        if _float(item.get("targetSeparationArcsec")) is not None
    ]
    return {
        "version": "openstar.tess-difference-image-source-localization.v1",
        "distributedFrequencyRefinement": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
            "combinedResult": combined,
        },
        "targetTIC": preparation.get("targetTIC"),
        "catalogCounterpart": counterpart,
        "referenceFrequency": preparation.get("referenceFrequency"),
        "referencePeriodDays": preparation.get("referencePeriodDays"),
        "fractionalFrequencyDriftPerDay": preparation.get("fractionalFrequencyDriftPerDay"),
        "sectorResults": sector_results,
        "qualitySectorCount": len(quality),
        "counterpartSupportingSectors": sorted(int(item["sector"]) for item in counterpart_supported),
        "targetSupportingSectors": sorted(int(item["sector"]) for item in target_supported),
        "ambiguousSectors": sorted(int(item["sector"]) for item in ambiguous),
        "medianCounterpartSeparationArcsec": statistics.median(counterpart_separations) if counterpart_separations else None,
        "medianTargetSeparationArcsec": statistics.median(target_separations) if target_separations else None,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "preparationErrors": preparation.get("errors") or [],
        "analysisErrors": analysis_errors,
        "interpretationGuard": (
            "Difference images are constructed from established-family-prewhitened TESS pixel cubes using high-minus-low "
            "phase bins at the distributed drift-corrected residual frequency. A source attribution requires the same "
            "direction to recur across independent sectors; this stage does not modify the v20.6 main-family target association."
        ),
    }
