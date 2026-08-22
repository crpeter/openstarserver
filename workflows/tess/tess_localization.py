from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_sector_archive import (
    TessArchiveTransientError,
    _is_transient_transport_error,
    configure_tess_archive_timeout,
)


OFFICIAL_AUTHORS = ("SPOC", "TESS-SPOC")
TESScut_SIZE = (11, 11)
MAX_CADENCES = 18_000
MIN_VALID_CADENCES = 300
TARGET_SUPPORT_MAX_PIXELS = 1.0
OFF_TARGET_MIN_PIXELS = 1.5
MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR = 3
MAX_OFF_TARGET_SKY_SCATTER_ARCSEC = 15.0
SIGNAL_CLUSTER_FRACTION = 0.20
MIN_AMPLITUDE_CONTRAST = 1.5
MIN_CLUSTER_WEIGHT_FRACTION = 0.30


def _archive_operation(description: str, operation):
    # Lightkurve has no timeout parameter on these public methods; its supported
    # Astroquery/Astropy transports are configured immediately before each call.
    configure_tess_archive_timeout()
    try:
        return operation()
    except Exception as error:
        if _is_transient_transport_error(error):
            raise TessArchiveTransientError(
                f"Transient MAST transport failure during {description}"
            ) from error
        raise


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


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if np.ma.is_masked(value):
            return None
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _sector_from_row(table: Any, index: int) -> int | None:
    columns = set(getattr(table, "colnames", []))
    if "sequence_number" in columns:
        value = _int(table["sequence_number"][index])
        if value is not None:
            return value
    if "mission" in columns:
        import re
        match = re.search(r"sector\s*0*(\d+)", str(table["mission"][index]), flags=re.I)
        if match:
            return int(match.group(1))
    return None


def _exptime_seconds(value: Any) -> float | None:
    try:
        from astropy import units as u
        if hasattr(value, "to_value"):
            return float(value.to_value(u.s))
    except Exception:
        pass
    if hasattr(value, "value"):
        value = value.value
    return _float(value)


def _select_official_tpf(search: Any, sector: int):
    table = getattr(search, "table", None)
    if table is None or len(table) == 0:
        return None
    columns = set(getattr(table, "colnames", []))
    candidates: list[tuple[int, float, int, str, Any]] = []
    for index in range(len(table)):
        row_sector = _sector_from_row(table, index)
        if row_sector is not None and row_sector != int(sector):
            continue
        author = str(table["author"][index]).strip().upper() if "author" in columns else ""
        if author not in OFFICIAL_AUTHORS:
            continue
        cadence = _exptime_seconds(table["exptime"][index]) if "exptime" in columns else None
        candidates.append((
            OFFICIAL_AUTHORS.index(author),
            cadence if cadence is not None else float("inf"),
            index,
            author,
            search[index:index + 1],
        ))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, cadence, _, author, selected = candidates[0]
    return selected, author, None if not math.isfinite(cadence) else float(cadence)


def _download_tpf(*, tic_id: int, sector: int, ra_deg: float, dec_deg: float):
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    print(f"      searching official target-pixel products for Sector {sector}...", flush=True)
    try:
        search = _archive_operation(
            "official target-pixel search",
            lambda: lk.search_targetpixelfile(
                f"TIC {int(tic_id)}",
                mission="TESS",
                sector=int(sector),
            ),
        )
        selected = _select_official_tpf(search, sector)
    except TessArchiveTransientError:
        raise
    except Exception as error:
        selected = None
        official_error = f"{type(error).__name__}: {error}"
    else:
        official_error = None

    if selected is not None:
        result, author, cadence_seconds = selected
        print(
            f"      selected official TPF: {author} | "
            f"{cadence_seconds:.0f}s" if cadence_seconds is not None else f"      selected official TPF: {author}",
            flush=True,
        )
        print("      downloading target pixel file...", flush=True)
        tpf = _archive_operation(
            "official target-pixel download",
            lambda: result.download(quality_bitmask="default"),
        )
        if tpf is None:
            raise RuntimeError(f"Official TPF download returned no data for Sector {sector}.")
        return tpf, {
            "sourceType": "OFFICIAL_TPF",
            "author": author,
            "cadenceSeconds": cadence_seconds,
            "officialSearchError": official_error,
        }

    print("      no official TPF selected; falling back to an 11x11 TESScut FFI cutout...", flush=True)
    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    search = _archive_operation(
        "TESScut search",
        lambda: lk.search_tesscut(target, sector=int(sector)),
    )
    if getattr(search, "table", None) is None or len(search.table) == 0:
        raise RuntimeError(f"No official TPF or TESScut coverage available for Sector {sector}.")
    print("      downloading TESScut pixels...", flush=True)
    tpf = _archive_operation(
        "TESScut download",
        lambda: search[0:1].download(
            quality_bitmask="default", cutout_size=TESScut_SIZE
        ),
    )
    if tpf is None:
        raise RuntimeError(f"TESScut download returned no data for Sector {sector}.")
    cadence_seconds = None
    table = getattr(search[0:1], "table", None)
    if table is not None and len(table) and "exptime" in table.colnames:
        cadence_seconds = _exptime_seconds(table["exptime"][0])
    return tpf, {
        "sourceType": "TESSCUT_FFI",
        "author": "TESScut",
        "cadenceSeconds": cadence_seconds,
        "officialSearchError": official_error,
    }


def _uniform_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def _background_subtract_cube(cube: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    # Remove a spatially uniform residual/background at every cadence. Official
    # TPFs are pipeline background-subtracted already; this step only removes a
    # residual common mode. It is important for TESScut, whose pixels are not
    # pipeline background-subtracted.
    rows, cols = cube.shape[1:]
    border = np.zeros((rows, cols), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_values = cube[:, border]
    background = np.nanmedian(border_values, axis=1)
    corrected = cube - background[:, None, None]
    return corrected, {
        "method": "per-cadence-border-median",
        "borderPixelCount": int(np.sum(border)),
    }


def _design_matrix(times: np.ndarray, physical_period_days: float) -> np.ndarray:
    centered = times - np.mean(times)
    scale = np.std(centered)
    trend = centered / scale if scale > 0 else centered
    omega = 2.0 * math.pi / physical_period_days
    return np.column_stack([
        np.ones(len(times), dtype=np.float64),
        trend,
        np.sin(omega * times),
        np.cos(omega * times),
        np.sin(2.0 * omega * times),
        np.cos(2.0 * omega * times),
    ])


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
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not output[ny, nx]:
                    output[ny, nx] = True
                    stack.append((ny, nx))
    return output


def _pixel_fit_maps(
    times: np.ndarray,
    cube: np.ndarray,
    *,
    physical_period_days: float,
) -> dict[str, Any]:
    if len(times) < MIN_VALID_CADENCES:
        raise RuntimeError(
            f"Only {len(times)} usable cadences; need at least {MIN_VALID_CADENCES} for localization."
        )

    matrix = _design_matrix(times, physical_period_days)
    pinv = np.linalg.pinv(matrix)
    flat = cube.reshape(len(times), -1).astype(np.float64)
    finite_fraction = np.mean(np.isfinite(flat), axis=0)
    valid_pixels = finite_fraction >= 0.90

    # Fill the small number of missing values per pixel with its time median so
    # all pixels use the same deterministic Fourier design matrix.
    medians = np.nanmedian(flat, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(flat)
    if np.any(missing):
        flat = np.where(missing, medians[None, :], flat)

    beta = pinv @ flat
    model = matrix @ beta
    residual = flat - model
    residual_rms = np.sqrt(np.mean(residual * residual, axis=0))

    fundamental = np.sqrt(beta[2] ** 2 + beta[3] ** 2)
    harmonic = np.sqrt(beta[4] ** 2 + beta[5] ** 2)
    combined = np.sqrt(fundamental ** 2 + harmonic ** 2)
    periodic_rms = combined / math.sqrt(2.0)
    relative_to_residual = np.divide(
        periodic_rms,
        residual_rms,
        out=np.zeros_like(periodic_rms),
        where=residual_rms > 0,
    )

    combined = np.where(valid_pixels, combined, 0.0)
    fundamental = np.where(valid_pixels, fundamental, 0.0)
    harmonic = np.where(valid_pixels, harmonic, 0.0)
    relative_to_residual = np.where(valid_pixels, relative_to_residual, 0.0)

    shape = cube.shape[1:]
    return {
        "fundamentalAmplitude": fundamental.reshape(shape),
        "firstHarmonicAmplitude": harmonic.reshape(shape),
        "combinedAmplitude": combined.reshape(shape),
        "periodicToResidualRms": relative_to_residual.reshape(shape),
        "validPixelMask": valid_pixels.reshape(shape),
    }


def _signal_centroid(amplitude: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    values = np.where(valid & np.isfinite(amplitude), amplitude, 0.0)
    if not np.any(values > 0):
        raise RuntimeError("Periodic amplitude map contains no positive valid pixels.")

    positive = values[values > 0]
    median = float(np.median(positive))
    mad = float(np.median(np.abs(positive - median))) if positive.size else 0.0
    floor = median + 1.4826 * mad
    weights = np.clip(values - floor, 0.0, None)
    if not np.any(weights > 0):
        weights = values.copy()
        floor = 0.0

    peak_y, peak_x = np.unravel_index(int(np.argmax(weights)), weights.shape)
    peak_weight = float(weights[peak_y, peak_x])
    threshold = peak_weight * SIGNAL_CLUSTER_FRACTION
    cluster_mask = weights >= threshold
    cluster_mask = _connected_component(cluster_mask, peak_y, peak_x)
    cluster_weights = np.where(cluster_mask, weights, 0.0)
    total = float(np.sum(cluster_weights))
    if total <= 0:
        raise RuntimeError("Could not isolate a connected periodic-signal pixel cluster.")

    yy, xx = np.indices(weights.shape, dtype=np.float64)
    x_centroid = float(np.sum(xx * cluster_weights) / total)
    y_centroid = float(np.sum(yy * cluster_weights) / total)
    total_all = float(np.sum(weights))
    cluster_fraction = total / total_all if total_all > 0 else None
    return {
        "x": x_centroid,
        "y": y_centroid,
        "peakX": int(peak_x),
        "peakY": int(peak_y),
        "backgroundFloor": floor,
        "clusterThresholdFractionOfPeak": SIGNAL_CLUSTER_FRACTION,
        "clusterPixelCount": int(np.sum(cluster_mask)),
        "clusterWeightFraction": cluster_fraction,
        "clusterMask": cluster_mask,
    }


def analyze_pixel_cube(
    *,
    times: np.ndarray,
    flux_cube: np.ndarray,
    physical_period_days: float,
    target_x: float,
    target_y: float,
    pixel_scale_arcsec: float | None = None,
) -> dict[str, Any]:
    times = np.asarray(times, dtype=np.float64)
    cube = np.asarray(flux_cube, dtype=np.float64)
    if cube.ndim != 3 or cube.shape[0] != len(times):
        raise ValueError("flux_cube must have shape (cadence, row, column) matching times.")
    finite_time = np.isfinite(times)
    finite_frame = np.any(np.isfinite(cube.reshape(len(times), -1)), axis=1)
    keep = finite_time & finite_frame
    times = times[keep]
    cube = cube[keep]
    indices = _uniform_indices(len(times), MAX_CADENCES)
    times = times[indices]
    cube = cube[indices]

    corrected, background = _background_subtract_cube(cube)
    maps = _pixel_fit_maps(times, corrected, physical_period_days=physical_period_days)
    centroid = _signal_centroid(maps["combinedAmplitude"], maps["validPixelMask"])

    amplitude_map = maps["combinedAmplitude"]
    valid_amplitudes = amplitude_map[maps["validPixelMask"] & np.isfinite(amplitude_map) & (amplitude_map > 0)]
    median_amplitude = float(np.median(valid_amplitudes)) if valid_amplitudes.size else 0.0
    peak_amplitude = float(np.max(valid_amplitudes)) if valid_amplitudes.size else 0.0
    amplitude_contrast = (peak_amplitude / median_amplitude) if median_amplitude > 0 else None
    peak_ratio_map = maps["periodicToResidualRms"]
    peak_periodic_to_residual = float(peak_ratio_map[centroid["peakY"], centroid["peakX"]])
    localization_quality = (
        amplitude_contrast is not None
        and amplitude_contrast >= MIN_AMPLITUDE_CONTRAST
        and centroid["clusterWeightFraction"] is not None
        and centroid["clusterWeightFraction"] >= MIN_CLUSTER_WEIGHT_FRACTION
    )

    dx = centroid["x"] - float(target_x)
    dy = centroid["y"] - float(target_y)
    distance_pixels = float(math.hypot(dx, dy))
    distance_arcsec = (
        distance_pixels * float(pixel_scale_arcsec)
        if pixel_scale_arcsec is not None
        else None
    )

    if not localization_quality:
        classification = "AMBIGUOUS"
    elif distance_pixels <= TARGET_SUPPORT_MAX_PIXELS:
        classification = "TARGET_CONSISTENT"
    elif distance_pixels >= OFF_TARGET_MIN_PIXELS:
        classification = "OFF_TARGET"
    else:
        classification = "AMBIGUOUS"

    return {
        "available": True,
        "usableCadences": int(len(times)),
        "shape": [int(corrected.shape[1]), int(corrected.shape[2])],
        "backgroundCorrection": background,
        "targetPixel": {"x": float(target_x), "y": float(target_y)},
        "signalCentroidPixel": {"x": centroid["x"], "y": centroid["y"]},
        "signalPeakPixel": {"x": centroid["peakX"], "y": centroid["peakY"]},
        "signalClusterPixelCount": centroid["clusterPixelCount"],
        "signalClusterWeightFraction": centroid["clusterWeightFraction"],
        "peakPeriodicAmplitude": peak_amplitude,
        "medianPeriodicAmplitude": median_amplitude,
        "periodicAmplitudeContrast": amplitude_contrast,
        "peakPeriodicToResidualRms": peak_periodic_to_residual,
        "localizationQualityPass": localization_quality,
        "offsetPixels": distance_pixels,
        "offsetArcsecApprox": distance_arcsec,
        "classification": classification,
        "thresholds": {
            "targetSupportMaxPixels": TARGET_SUPPORT_MAX_PIXELS,
            "offTargetMinPixels": OFF_TARGET_MIN_PIXELS,
            "minimumAmplitudeContrast": MIN_AMPLITUDE_CONTRAST,
            "minimumClusterWeightFraction": MIN_CLUSTER_WEIGHT_FRACTION,
        },
        "fundamentalAmplitudeMap": maps["fundamentalAmplitude"].tolist(),
        "firstHarmonicAmplitudeMap": maps["firstHarmonicAmplitude"].tolist(),
        "combinedPeriodicAmplitudeMap": maps["combinedAmplitude"].tolist(),
        "periodicToResidualRmsMap": maps["periodicToResidualRms"].tolist(),
    }


def _pixel_scale_arcsec(wcs: Any) -> float | None:
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(wcs)
        values = [abs(float(value)) * 3600.0 for value in scales[:2]]
        values = [value for value in values if math.isfinite(value) and value > 0]
        return statistics.median(values) if values else None
    except Exception:
        return None


def _world_offsets_arcsec(target: Any, signal: Any) -> tuple[float | None, float | None, float | None]:
    try:
        dlon, dlat = target.spherical_offsets_to(signal)
        separation = target.separation(signal)
        return float(dlon.arcsec), float(dlat.arcsec), float(separation.arcsec)
    except Exception:
        return None, None, None


def _sector_localization(
    *,
    tic_id: int,
    sector: int,
    role: str,
    ra_deg: float,
    dec_deg: float,
    physical_period_days: float,
) -> dict[str, Any]:
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    tpf, source = _download_tpf(
        tic_id=tic_id,
        sector=sector,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )
    try:
        header = tpf.get_header(0)
        source.update({
            "sector": _int(header.get("SECTOR")) or int(sector),
            "camera": _int(header.get("CAMERA")),
            "ccd": _int(header.get("CCD")),
            "object": _python_value(header.get("OBJECT")),
            "origin": _python_value(header.get("ORIGIN")),
            "dataRelease": _int(header.get("DATA_REL")),
        })
    except Exception:
        pass

    times = np.asarray(tpf.time.value, dtype=np.float64)
    flux = getattr(tpf.flux, "value", tpf.flux)
    if np.ma.isMaskedArray(flux):
        flux = np.ma.filled(flux, np.nan)
    cube = np.asarray(flux, dtype=np.float64)
    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    target_x, target_y = tpf.wcs.world_to_pixel(target)
    pixel_scale = _pixel_scale_arcsec(tpf.wcs)

    result = analyze_pixel_cube(
        times=times,
        flux_cube=cube,
        physical_period_days=physical_period_days,
        target_x=float(target_x),
        target_y=float(target_y),
        pixel_scale_arcsec=pixel_scale,
    )

    signal_world = tpf.wcs.pixel_to_world(
        result["signalCentroidPixel"]["x"],
        result["signalCentroidPixel"]["y"],
    )
    east, north, separation = _world_offsets_arcsec(target, signal_world)
    result.update({
        "sector": int(sector),
        "role": role,
        "source": source,
        "pixelScaleArcsec": pixel_scale,
        "targetSky": {"raDeg": ra_deg, "decDeg": dec_deg},
        "signalCentroidSky": {
            "raDeg": _float(getattr(signal_world, "ra", None).deg if hasattr(getattr(signal_world, "ra", None), "deg") else None),
            "decDeg": _float(getattr(signal_world, "dec", None).deg if hasattr(getattr(signal_world, "dec", None), "deg") else None),
        },
        "skyOffsetEastArcsec": east,
        "skyOffsetNorthArcsec": north,
        "skySeparationArcsec": separation,
    })
    return result


def _cross_sector_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    independent = [
        item for item in results
        if item.get("role") == "independent" and item.get("available")
    ]
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

    east = [
        float(item["skyOffsetEastArcsec"])
        for item in independent
        if _float(item.get("skyOffsetEastArcsec")) is not None
    ]
    north = [
        float(item["skyOffsetNorthArcsec"])
        for item in independent
        if _float(item.get("skyOffsetNorthArcsec")) is not None
    ]
    separations = [
        float(item["skySeparationArcsec"])
        for item in independent
        if _float(item.get("skySeparationArcsec")) is not None
    ]

    off_target_offsets = [
        (float(item["skyOffsetEastArcsec"]), float(item["skyOffsetNorthArcsec"]))
        for item in independent
        if item.get("classification") == "OFF_TARGET"
        and _float(item.get("skyOffsetEastArcsec")) is not None
        and _float(item.get("skyOffsetNorthArcsec")) is not None
    ]
    off_target_scatter = None
    if off_target_offsets:
        med_east = statistics.median(value[0] for value in off_target_offsets)
        med_north = statistics.median(value[1] for value in off_target_offsets)
        distances = [
            math.hypot(value[0] - med_east, value[1] - med_north)
            for value in off_target_offsets
        ]
        off_target_scatter = statistics.median(distances)

    if eligible >= MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR and len(target_support) >= required:
        classification = "TARGET_SOURCE_SUPPORTED"
        variable_origin = "TARGET_CONSISTENT"
        next_test = "MULTI_MODE_FREQUENCY_DECOMPOSITION"
    elif (
        eligible >= MIN_INDEPENDENT_SECTORS_FOR_CROSS_SECTOR
        and len(off_target) >= required
        and off_target_scatter is not None
        and off_target_scatter <= MAX_OFF_TARGET_SKY_SCATTER_ARCSEC
    ):
        classification = "OFF_TARGET_VARIABLE_SOURCE_SUPPORTED"
        variable_origin = "OFF_TARGET"
        next_test = "IDENTIFY_OFFSET_VARIABLE_SOURCE"
    else:
        classification = "SOURCE_LOCALIZATION_UNRESOLVED"
        variable_origin = "UNRESOLVED"
        next_test = "SOURCE_LOCALIZATION_REVIEW"

    return {
        "classification": classification,
        "variableSignalOrigin": variable_origin,
        "independentEligibleSectorCount": eligible,
        "requiredIndependentSupportCount": required,
        "targetSupportingSectors": target_support,
        "offTargetSectors": off_target,
        "ambiguousSectors": ambiguous,
        "medianSkyOffsetEastArcsec": statistics.median(east) if east else None,
        "medianSkyOffsetNorthArcsec": statistics.median(north) if north else None,
        "medianSkySeparationArcsec": statistics.median(separations) if separations else None,
        "offTargetSkyOffsetScatterArcsec": off_target_scatter,
        "maximumOffTargetSkyOffsetScatterArcsec": MAX_OFF_TARGET_SKY_SCATTER_ARCSEC,
        "recommendedNextTest": next_test,
    }


def localize_periodic_source(
    *,
    tic_id: int,
    identity: dict[str, Any],
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    artifact_root: str | Path,
) -> dict[str, Any]:
    tic_metadata = ((identity.get("tic") or {}).get("metadata") or {})
    ra_deg = _float(tic_metadata.get("raDeg"))
    dec_deg = _float(tic_metadata.get("decDeg"))
    if ra_deg is None or dec_deg is None:
        raise RuntimeError("Pixel localization requires TIC RA/Dec from the identity stage.")

    sectors: list[tuple[int, str]] = []
    if primary_sector is not None:
        sectors.append((int(primary_sector), "primary"))
    for item in independent_spec.get("preparedSectors") or []:
        sector = _int(item.get("sector"))
        if sector is not None and all(existing != sector for existing, _ in sectors):
            sectors.append((sector, "independent"))

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    root = Path(artifact_root)
    for index, (sector, role) in enumerate(sectors, start=1):
        print(f"   Sector {sector} ({index}/{len(sectors)}): localizing periodic signal", flush=True)
        try:
            result = _sector_localization(
                tic_id=tic_id,
                sector=sector,
                role=role,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                physical_period_days=physical_period_days,
            )
            results.append(result)
            sector_path = root / f"sector-{sector}-localization.json"
            _write_json(sector_path, result)
            print(
                f"      source={result.get('source', {}).get('sourceType')} | "
                f"offset={result.get('offsetPixels'):.3f} px | "
                f"classification={result.get('classification')}",
                flush=True,
            )
        except TessArchiveTransientError:
            raise
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            errors.append({"sector": sector, "role": role, "error": text})
            print(f"      localization unavailable: {text}", flush=True)

    summary = _cross_sector_summary(results)
    result = {
        "version": "openstar.tess-pixel-localization.v1",
        "ticID": int(tic_id),
        "physicalPeriodDays": float(physical_period_days),
        "photometricFirstHarmonicPeriodDays": float(physical_period_days) / 2.0,
        "targetSky": {"raDeg": ra_deg, "decDeg": dec_deg},
        "sectorResults": results,
        "errors": errors,
        "crossSector": summary,
        "contaminationInterpretation": {
            "existingCatalogContaminationCanBeCleared": False,
            "reason": (
                "Pixel localization tests the origin of the variable signal; it does not prove "
                "that static flux contamination in the aperture is absent."
            ),
        },
        "recommendedNextTest": summary["recommendedNextTest"],
        "interpretationGuard": (
            "A periodic-signal centroid consistent with the TIC position supports target origin, "
            "but does not by itself identify the physical variability mechanism."
        ),
    }
    _write_json(root / "pixel-localization-v20.6.json", result)
    return result
