from __future__ import annotations

import copy
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_localization import (
    _background_subtract_cube,
    _download_tpf,
    _pixel_scale_arcsec,
)


GENERIC_LOMB_SCARGLE_WORKLOAD_ID = "openstar.lomb-scargle.v1"
LOMB_SCARGLE_WORKLOAD_ALIASES = {
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    "openstar.tess-period-search.v1",
}
MAX_CADENCES = 3_000
MIN_VALID_CADENCES = 300
MIN_VALID_PIXEL_FRACTION = 0.90
FREQUENCY_HALF_WIDTH_FRACTION = 0.10
TOTAL_FREQUENCIES = 2_048
FREQUENCIES_PER_WORK_UNIT = 2_048
MIN_PEAK_POWER = 0.08
MIN_POWER_CONTRAST = 1.5
SIGNAL_CLUSTER_FRACTION = 0.20
MIN_CLUSTER_WEIGHT_FRACTION = 0.25
TARGET_SUPPORT_MAX_PIXELS = 1.0
OFF_TARGET_MIN_PIXELS = 1.5
MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR = 3
MAX_OFF_TARGET_SKY_SCATTER_ARCSEC = 15.0


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _safe(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return text or "artifact"


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _uniform_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def _design_matrix(
    times: np.ndarray, physical_frequency: float, harmonic_orders: tuple[int, ...] = (1, 2)
) -> np.ndarray:
    centered = times - float(np.mean(times))
    scale = float(np.std(centered))
    trend = centered / scale if scale > 0 else centered
    omega = 2.0 * math.pi * float(physical_frequency)
    columns = [np.ones(len(times), dtype=np.float64), trend]
    for order in harmonic_orders:
        columns.extend((
            np.sin(float(order) * omega * times),
            np.cos(float(order) * omega * times),
        ))
    return np.column_stack(columns)


def _prewhiten_cube(
    *,
    absolute_times: np.ndarray,
    cube: np.ndarray,
    physical_frequency: float,
    harmonic_orders: tuple[int, ...] = (1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _design_matrix(absolute_times, physical_frequency, harmonic_orders)
    pinv = np.linalg.pinv(matrix)
    flat = cube.reshape(len(absolute_times), -1).astype(np.float64)
    finite_fraction = np.mean(np.isfinite(flat), axis=0)
    valid = finite_fraction >= MIN_VALID_PIXEL_FRACTION

    medians = np.nanmedian(flat, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    if np.any(~np.isfinite(flat)):
        flat = np.where(np.isfinite(flat), flat, medians[None, :])

    beta = pinv @ flat
    residual = flat - matrix @ beta
    residual -= np.mean(residual, axis=0, keepdims=True)
    std = np.std(residual, axis=0)
    valid &= np.isfinite(std) & (std > 1e-12)
    normalized = np.divide(
        residual,
        std[None, :],
        out=np.zeros_like(residual),
        where=std[None, :] > 1e-12,
    )
    normalized[:, ~valid] = 0.0
    return normalized.reshape(cube.shape), valid.reshape(cube.shape[1:])


def _time_warp(relative_times: np.ndarray, q: float) -> np.ndarray:
    warped = relative_times + 0.5 * float(q) * np.square(relative_times)
    if np.any(~np.isfinite(warped)):
        raise RuntimeError("Residual-mode pixel time warp produced non-finite values.")
    order = np.argsort(relative_times)
    if len(order) > 1 and np.any(np.diff(warped[order]) <= 0):
        raise RuntimeError("Residual-mode drift time warp is non-monotonic.")
    return warped


def _frequency_search(reference_frequency: float) -> dict[str, Any]:
    minimum = reference_frequency * (1.0 - FREQUENCY_HALF_WIDTH_FRACTION)
    maximum = reference_frequency * (1.0 + FREQUENCY_HALF_WIDTH_FRACTION)
    if minimum <= 0 or maximum <= minimum:
        raise RuntimeError("Invalid residual-mode localization frequency range.")
    return {
        "minimumFrequency": float(minimum),
        "maximumFrequency": float(maximum),
        "frequencyStep": float((maximum - minimum) / (TOTAL_FREQUENCIES - 1)),
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def _local_sky_jacobian(wcs: Any, target: Any, target_x: float, target_y: float) -> dict[str, float | None]:
    try:
        x_world = wcs.pixel_to_world(float(target_x) + 1.0, float(target_y))
        y_world = wcs.pixel_to_world(float(target_x), float(target_y) + 1.0)
        x_east, x_north = target.spherical_offsets_to(x_world)
        y_east, y_north = target.spherical_offsets_to(y_world)
        return {
            "xToEastArcsec": float(x_east.arcsec),
            "xToNorthArcsec": float(x_north.arcsec),
            "yToEastArcsec": float(y_east.arcsec),
            "yToNorthArcsec": float(y_north.arcsec),
        }
    except Exception:
        return {
            "xToEastArcsec": None,
            "xToNorthArcsec": None,
            "yToEastArcsec": None,
            "yToNorthArcsec": None,
        }


def _sector_candidates(
    *,
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    signal_sectors: list[int],
) -> list[tuple[int, str]]:
    allowed = {int(value) for value in signal_sectors}
    sectors: list[tuple[int, str]] = []
    if primary_sector is not None and int(primary_sector) in allowed:
        sectors.append((int(primary_sector), "primary"))
    for item in independent_spec.get("preparedSectors") or []:
        sector = _int(item.get("sector"))
        if sector is None or sector not in allowed:
            continue
        if all(existing != sector for existing, _ in sectors):
            sectors.append((sector, "independent"))
    return sectors


def _target_coordinate(ra_deg: float, dec_deg: float) -> Any:
    """Construct the astronomy-library coordinate used by a real TPF WCS."""

    from astropy.coordinates import SkyCoord
    from astropy import units as u

    return SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")


def build_residual_mode_pixel_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    tic_id: int,
    identity: dict[str, Any],
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    nonstationary_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
    harmonic_orders: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.10 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )

    preferred = nonstationary_summary.get("preferredModel") or {}
    reference_frequency = _float(nonstationary_summary.get("preferredFrequencyAtReference"))
    q = _float(nonstationary_summary.get("fractionalFrequencyDriftPerDay"))
    time_reference = _float(nonstationary_summary.get("timeReferenceDays"))
    signal_sectors = [
        int(value)
        for value in preferred.get("signalSectors") or []
        if _int(value) is not None
    ]
    if reference_frequency is None or reference_frequency <= 0:
        raise RuntimeError("v20.10 requires the v20.9 preferred residual frequency.")
    if q is None:
        raise RuntimeError("v20.10 requires the v20.9 preferred fractional frequency drift.")
    if time_reference is None:
        raise RuntimeError("v20.10 requires the v20.9 time reference.")
    if len(signal_sectors) < 1:
        raise RuntimeError("v20.10 requires at least one v20.9 signal sector.")
    if physical_period_days <= 0:
        raise RuntimeError("v20.10 requires a positive resolved physical period.")

    tic_metadata = ((identity.get("tic") or {}).get("metadata") or {})
    ra_deg = _float(tic_metadata.get("raDeg"))
    dec_deg = _float(tic_metadata.get("decDeg"))
    if ra_deg is None or dec_deg is None:
        raise RuntimeError("v20.10 requires TIC RA/Dec from the identity stage.")

    target = _target_coordinate(float(ra_deg), float(dec_deg))
    sectors = _sector_candidates(
        primary_sector=primary_sector,
        independent_spec=independent_spec,
        signal_sectors=signal_sectors,
    )
    if not sectors:
        raise RuntimeError("No frozen TESS sectors overlap the v20.9 signal-sector set.")

    family_suffix = "-h" + "-".join(str(value) for value in harmonic_orders)
    corrected_family = tuple(harmonic_orders) != (1, 2)
    root = Path(output_dir) / "residual-mode-localization"
    if corrected_family:
        root = root / family_suffix.lstrip("-")
    root.mkdir(parents=True, exist_ok=True)
    search = _frequency_search(reference_frequency)
    physical_frequency = 1.0 / float(physical_period_days)
    source_base_id = str(source_dataset_entry.get("id") or f"tic-{tic_id}")

    dataset_entries: list[dict[str, Any]] = []
    prepared_pixels: list[dict[str, Any]] = []
    sector_metadata: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for sector_index, (sector, role) in enumerate(sectors, start=1):
        print(f"   Sector {sector} ({sector_index}/{len(sectors)}): preparing residual-mode pixel work", flush=True)
        try:
            tpf, source = _download_tpf(
                tic_id=int(tic_id),
                sector=int(sector),
                ra_deg=float(ra_deg),
                dec_deg=float(dec_deg),
            )
            absolute_times = np.asarray(tpf.time.value, dtype=np.float64)
            flux = getattr(tpf.flux, "value", tpf.flux)
            if np.ma.isMaskedArray(flux):
                flux = np.ma.filled(flux, np.nan)
            cube = np.asarray(flux, dtype=np.float64)
            finite_time = np.isfinite(absolute_times)
            finite_frame = np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
            keep = finite_time & finite_frame
            absolute_times = absolute_times[keep]
            cube = cube[keep]
            if len(absolute_times) < MIN_VALID_CADENCES:
                raise RuntimeError(
                    f"Only {len(absolute_times)} usable cadences; need {MIN_VALID_CADENCES}."
                )
            indices = _uniform_indices(len(absolute_times), MAX_CADENCES)
            absolute_times = absolute_times[indices]
            cube = cube[indices]
            corrected, background = _background_subtract_cube(cube)
            residual_cube, valid_pixels = _prewhiten_cube(
                absolute_times=absolute_times,
                cube=corrected,
                physical_frequency=physical_frequency,
                harmonic_orders=harmonic_orders,
            )
            target_x, target_y = tpf.wcs.world_to_pixel(target)
            pixel_scale = _pixel_scale_arcsec(tpf.wcs)
            jacobian = _local_sky_jacobian(tpf.wcs, target, float(target_x), float(target_y))
            relative_times = absolute_times - float(time_reference)
            warped = _time_warp(relative_times, float(q))
            local_times = warped - float(np.min(warped))

            rows, cols = valid_pixels.shape
            sector_pixel_count = 0
            for row in range(rows):
                for col in range(cols):
                    if not bool(valid_pixels[row, col]):
                        continue
                    pixel_flux = residual_cube[:, row, col]
                    if not np.all(np.isfinite(pixel_flux)):
                        continue
                    dataset_id = (
                        f"{source_base_id}-residual-localization-sector-{sector}-"
                        f"r{row:02d}-c{col:02d}-v1{family_suffix if corrected_family else ''}"
                    )
                    target_name = (
                        f"{source_dataset_entry.get('targetName') or source_base_id} "
                        f"residual localization sector {sector} pixel ({row},{col})"
                    )
                    output_path = root / f"{_safe(dataset_id)}.json"
                    dataset = {
                        "id": dataset_id,
                        "targetName": target_name,
                        "times": np.asarray(local_times, dtype=np.float32).tolist(),
                        "flux": np.asarray(pixel_flux, dtype=np.float32).tolist(),
                        "frequencySearch": search,
                        "reference": {},
                        "science": {
                            "role": "residual-mode-pixel-localization",
                            "purpose": "generic-lomb-scargle-on-time-warped-prewhitened-pixel-light-curve",
                            "sector": int(sector),
                            "pixelRow": int(row),
                            "pixelColumn": int(col),
                            "referenceFrequency": float(reference_frequency),
                            "fractionalFrequencyDriftPerDay": float(q),
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
                    _write_json(output_path, dataset)
                    meta = {
                        "datasetID": dataset_id,
                        "datasetPath": str(output_path.resolve()),
                        "sector": int(sector),
                        "role": role,
                        "row": int(row),
                        "column": int(col),
                    }
                    prepared_pixels.append(meta)
                    manifest_entry = copy.deepcopy(source_dataset_entry)
                    manifest_entry.update({
                        "id": dataset_id,
                        "path": meta["datasetPath"],
                        "targetName": target_name,
                        "role": "residual-mode-pixel-localization",
                    })
                    dataset_entries.append(manifest_entry)
                    sector_pixel_count += 1

            sector_metadata.append({
                "sector": int(sector),
                "role": role,
                "shape": [int(rows), int(cols)],
                "targetPixel": {"x": float(target_x), "y": float(target_y)},
                "pixelScaleArcsec": pixel_scale,
                "pixelToSkyJacobianArcsec": jacobian,
                "source": source,
                "backgroundCorrection": background,
                "usableCadences": int(len(absolute_times)),
                "preparedPixelCount": int(sector_pixel_count),
            })
            print(
                f"      source={source.get('sourceType')} | shape={rows}x{cols} | "
                f"pixel datasets={sector_pixel_count}",
                flush=True,
            )
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            errors.append({"sector": int(sector), "role": role, "error": text})
            print(f"      residual-mode pixel preparation unavailable: {text}", flush=True)

    if not dataset_entries:
        raise RuntimeError("v20.10 could not prepare any residual-mode pixel datasets.")

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        f"residual-mode-pixel-localization-v1{family_suffix if corrected_family else ''}"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — residual-mode pixel localization",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "residual-mode-pixel-localization",
            "workerSemantics": (
                "Each dataset is one prewhitened TESS pixel light curve after the v20.9 "
                "deterministic drift time warp. Workers execute ordinary Lomb-Scargle only."
            ),
            "referenceFrequency": float(reference_frequency),
            "fractionalFrequencyDriftPerDay": float(q),
            "signalSectors": signal_sectors,
            "subtractedHarmonicOrders": list(harmonic_orders),
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": True,
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-time-warped-prewhitened-pixel-light-curves",
        "ticID": int(tic_id),
        "targetSky": {"raDeg": float(ra_deg), "decDeg": float(dec_deg)},
        "physicalPeriodDays": float(physical_period_days),
        "subtractedHarmonicOrders": list(harmonic_orders),
        "residualFrequencyAtReference": float(reference_frequency),
        "residualPeriodAtReferenceDays": 1.0 / float(reference_frequency),
        "fractionalFrequencyDriftPerDay": float(q),
        "timeReferenceDays": float(time_reference),
        "signalSectors": signal_sectors,
        "frequencySearch": search,
        "sectorMetadata": sector_metadata,
        "preparedPixels": prepared_pixels,
        "errors": errors,
        "workUnitsPerDataset": work_units_per_dataset,
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
    }


def _connected_component(mask: np.ndarray, seed_y: int, seed_x: int) -> np.ndarray:
    rows, cols = mask.shape
    output = np.zeros_like(mask, dtype=bool)
    if not mask[seed_y, seed_x]:
        return output
    stack = [(seed_y, seed_x)]
    output[seed_y, seed_x] = True
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not output[ny, nx]:
                    output[ny, nx] = True
                    stack.append((ny, nx))
    return output


def _localize_power_map(
    *,
    power_map: np.ndarray,
    target_x: float,
    target_y: float,
    pixel_scale_arcsec: float | None,
    jacobian: dict[str, Any],
) -> dict[str, Any]:
    values = np.where(np.isfinite(power_map) & (power_map > 0), power_map, 0.0)
    positive = values[values > 0]
    if positive.size < 4:
        return {
            "classification": "AMBIGUOUS",
            "localizationQualityPass": False,
            "reason": "too-few-positive-pixel-powers",
        }
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median)))
    floor = median + 1.4826 * mad
    weights = np.clip(values - floor, 0.0, None)
    if not np.any(weights > 0):
        weights = values.copy()
        floor = 0.0

    peak_y, peak_x = np.unravel_index(int(np.argmax(weights)), weights.shape)
    peak_weight = float(weights[peak_y, peak_x])
    cluster = _connected_component(weights >= peak_weight * SIGNAL_CLUSTER_FRACTION, peak_y, peak_x)
    cluster_weights = np.where(cluster, weights, 0.0)
    cluster_total = float(np.sum(cluster_weights))
    total_weight = float(np.sum(weights))
    if cluster_total <= 0 or total_weight <= 0:
        return {
            "classification": "AMBIGUOUS",
            "localizationQualityPass": False,
            "reason": "no-spatially-concentrated-power",
        }

    yy, xx = np.indices(weights.shape, dtype=np.float64)
    centroid_x = float(np.sum(xx * cluster_weights) / cluster_total)
    centroid_y = float(np.sum(yy * cluster_weights) / cluster_total)
    peak_power = float(values[peak_y, peak_x])
    contrast = peak_power / median if median > 0 else None
    cluster_fraction = cluster_total / total_weight
    dx = centroid_x - float(target_x)
    dy = centroid_y - float(target_y)
    offset_pixels = float(math.hypot(dx, dy))

    x_e = _float(jacobian.get("xToEastArcsec"))
    x_n = _float(jacobian.get("xToNorthArcsec"))
    y_e = _float(jacobian.get("yToEastArcsec"))
    y_n = _float(jacobian.get("yToNorthArcsec"))
    east = dx * x_e + dy * y_e if None not in (x_e, y_e) else None
    north = dx * x_n + dy * y_n if None not in (x_n, y_n) else None
    sky_sep = math.hypot(east, north) if east is not None and north is not None else (
        offset_pixels * float(pixel_scale_arcsec) if pixel_scale_arcsec is not None else None
    )

    quality = bool(
        peak_power >= MIN_PEAK_POWER
        and contrast is not None
        and contrast >= MIN_POWER_CONTRAST
        and cluster_fraction >= MIN_CLUSTER_WEIGHT_FRACTION
    )
    if not quality:
        classification = "AMBIGUOUS"
    elif offset_pixels <= TARGET_SUPPORT_MAX_PIXELS:
        classification = "TARGET_CONSISTENT"
    elif offset_pixels >= OFF_TARGET_MIN_PIXELS:
        classification = "OFF_TARGET"
    else:
        classification = "AMBIGUOUS"

    return {
        "classification": classification,
        "localizationQualityPass": quality,
        "signalCentroidPixel": {"x": centroid_x, "y": centroid_y},
        "signalPeakPixel": {"x": int(peak_x), "y": int(peak_y)},
        "offsetPixels": offset_pixels,
        "skyOffsetEastArcsec": east,
        "skyOffsetNorthArcsec": north,
        "skySeparationArcsec": sky_sep,
        "peakPower": peak_power,
        "medianPositivePixelPower": median,
        "powerContrast": contrast,
        "backgroundPowerFloor": floor,
        "signalClusterPixelCount": int(np.sum(cluster)),
        "signalClusterWeightFraction": cluster_fraction,
        "thresholds": {
            "minimumPeakPower": MIN_PEAK_POWER,
            "minimumPowerContrast": MIN_POWER_CONTRAST,
            "minimumClusterWeightFraction": MIN_CLUSTER_WEIGHT_FRACTION,
            "targetSupportMaxPixels": TARGET_SUPPORT_MAX_PIXELS,
            "offTargetMinPixels": OFF_TARGET_MIN_PIXELS,
        },
    }


def _cross_sector_summary(sector_results: list[dict[str, Any]]) -> dict[str, Any]:
    independent = [item for item in sector_results if item.get("role") == "independent"]
    eligible = len(independent)
    target_support = sorted(
        int(item["sector"]) for item in independent
        if item.get("classification") == "TARGET_CONSISTENT"
    )
    off_target = sorted(
        int(item["sector"]) for item in independent
        if item.get("classification") == "OFF_TARGET"
    )
    ambiguous = sorted(
        int(item["sector"]) for item in independent
        if item.get("classification") == "AMBIGUOUS"
    )
    required = max(
        MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR,
        eligible // 2 + 1,
    ) if eligible else MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR

    off_offsets = [
        (float(item["skyOffsetEastArcsec"]), float(item["skyOffsetNorthArcsec"]))
        for item in independent
        if item.get("classification") == "OFF_TARGET"
        and _float(item.get("skyOffsetEastArcsec")) is not None
        and _float(item.get("skyOffsetNorthArcsec")) is not None
    ]
    off_scatter = None
    if off_offsets:
        med_e = statistics.median(value[0] for value in off_offsets)
        med_n = statistics.median(value[1] for value in off_offsets)
        off_scatter = statistics.median(
            math.hypot(value[0] - med_e, value[1] - med_n)
            for value in off_offsets
        )

    if eligible >= MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR and len(target_support) >= required:
        classification = "RESIDUAL_MODE_TARGET_SUPPORTED"
        origin = "TARGET_CONSISTENT"
        next_test = "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
    elif (
        eligible >= MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR
        and len(off_target) >= required
        and off_scatter is not None
        and off_scatter <= MAX_OFF_TARGET_SKY_SCATTER_ARCSEC
    ):
        classification = "RESIDUAL_MODE_OFF_TARGET_SUPPORTED"
        origin = "OFF_TARGET"
        next_test = "IDENTIFY_OFFSET_RESIDUAL_VARIABLE_SOURCE"
    else:
        classification = "RESIDUAL_MODE_LOCALIZATION_UNRESOLVED"
        origin = "UNRESOLVED"
        next_test = "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"

    separations = [
        float(item["skySeparationArcsec"])
        for item in independent
        if _float(item.get("skySeparationArcsec")) is not None
    ]
    return {
        "classification": classification,
        "residualModeOrigin": origin,
        "independentEligibleSectorCount": eligible,
        "requiredIndependentSupportCount": required,
        "targetSupportingSectors": target_support,
        "offTargetSectors": off_target,
        "ambiguousSectors": ambiguous,
        "medianSkySeparationArcsec": statistics.median(separations) if separations else None,
        "offTargetSkyOffsetScatterArcsec": off_scatter,
        "maximumOffTargetSkyOffsetScatterArcsec": MAX_OFF_TARGET_SKY_SCATTER_ARCSEC,
        "recommendedNextTest": next_test,
    }


def interpret_residual_mode_pixel_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedPixels") or []
    }
    sector_meta = {
        int(item["sector"]): item
        for item in preparation.get("sectorMetadata") or []
    }
    by_sector: dict[int, list[dict[str, Any]]] = {}

    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id)
        if meta is None:
            continue
        power = _float(dataset.get("candidatePower"))
        frequency = _float(dataset.get("candidateFrequency"))
        result = {
            **meta,
            "candidatePower": power,
            "candidateFrequency": frequency,
            "candidatePeriodDays": _float(dataset.get("candidatePeriodDays")),
            "periodStatus": str(dataset.get("periodStatus") or "").upper(),
            "periodConfidence": str(dataset.get("periodConfidence") or "none").lower(),
        }
        by_sector.setdefault(int(meta["sector"]), []).append(result)

    sector_results: list[dict[str, Any]] = []
    for sector in sorted(sector_meta):
        meta = sector_meta[sector]
        shape = meta.get("shape") or []
        if len(shape) != 2:
            continue
        rows, cols = int(shape[0]), int(shape[1])
        power_map = np.zeros((rows, cols), dtype=np.float64)
        frequency_map = np.full((rows, cols), np.nan, dtype=np.float64)
        for item in by_sector.get(sector, []):
            row = int(item["row"])
            col = int(item["column"])
            if 0 <= row < rows and 0 <= col < cols:
                if item.get("candidatePower") is not None:
                    power_map[row, col] = max(0.0, float(item["candidatePower"]))
                if item.get("candidateFrequency") is not None:
                    frequency_map[row, col] = float(item["candidateFrequency"])

        target_pixel = meta.get("targetPixel") or {}
        localization = _localize_power_map(
            power_map=power_map,
            target_x=float(target_pixel.get("x")),
            target_y=float(target_pixel.get("y")),
            pixel_scale_arcsec=_float(meta.get("pixelScaleArcsec")),
            jacobian=meta.get("pixelToSkyJacobianArcsec") or {},
        )
        sector_results.append({
            "sector": int(sector),
            "role": meta.get("role"),
            "source": meta.get("source"),
            "shape": [rows, cols],
            "targetPixel": target_pixel,
            "preparedPixelCount": meta.get("preparedPixelCount"),
            **localization,
            "candidatePowerMap": power_map.tolist(),
            "candidateFrequencyMap": [
                [float(value) if math.isfinite(float(value)) else None for value in row]
                for row in frequency_map
            ],
        })

    cross = _cross_sector_summary(sector_results)
    return {
        "version": "openstar.tess-residual-mode-pixel-localization.v1",
        "distributedLocalization": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "ticID": preparation.get("ticID"),
        "targetSky": preparation.get("targetSky"),
        "physicalPeriodDays": preparation.get("physicalPeriodDays"),
        "residualFrequencyAtReference": preparation.get("residualFrequencyAtReference"),
        "residualPeriodAtReferenceDays": preparation.get("residualPeriodAtReferenceDays"),
        "fractionalFrequencyDriftPerDay": preparation.get("fractionalFrequencyDriftPerDay"),
        "timeReferenceDays": preparation.get("timeReferenceDays"),
        "signalSectors": preparation.get("signalSectors"),
        "sectorResults": sector_results,
        "crossSector": cross,
        "errors": preparation.get("errors") or [],
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": cross.get("recommendedNextTest"),
        "interpretationGuard": (
            "This localizes the v20.9 drifting residual component after subtracting the established "
            "13.72-day family. It does not alter the already-established localization of the main periodic family."
        ),
    }
