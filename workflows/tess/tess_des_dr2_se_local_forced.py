from __future__ import annotations

import io
import math
import statistics
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    _float,
    _int,
    _safe,
    _write_json,
)
from .tess_noirlab_forced_photometry import (
    _angular_separation_arcsec,
    _dataset_result,
    _header_float,
    _image_hdu_from_bytes,
    _photometric_magnitude,
    _pixel_scale_arcsec,
    _python_value,
    _row_value,
    _scaled_linear_fit,
    _source_records_by_role,
    _summarize_source,
    _text,
    _world_to_pixel,
)

DES_SIA_URL = "https://datalab.noirlab.edu/sia/des_dr2_se"
HTTP_TIMEOUT_SECONDS = 120
USER_AGENT = "OpenStar/20.23 DES-DR2-single-epoch-local-forced-photometry"

SUPPORTED_BANDS = ("g", "r", "i", "z", "y")
DISCOVERY_FIELD_SIZE_DEG = 0.022
LOCAL_CUTOUT_SIZE_DEG = 0.006
MIN_INDEPENDENT_LOCAL_SEPARATION_ARCSEC = 15.0

MAX_EXPOSURES_PER_BAND = 40
MAX_TOTAL_EXPOSURES = 180

MIN_BAND_EPOCHS = 10
MIN_BAND_BASELINE_DAYS = 15.0
MAX_PIXEL_SCALE_ARCSEC = 0.55
MAX_ABSOLUTE_FWHM_ARCSEC = 2.20
MAX_SCALED_DESIGN_CONDITION = 120.0
MIN_FIT_R2 = 0.25
MIN_TARGET_AMPLITUDE_SNR = 8.0
MIN_COUNTERPART_AMPLITUDE_SNR = 3.0
SATURATION_FRACTION = 0.90
LOCAL_FIT_RADIUS_PIXELS = 10

MIN_PEAK_PROMINENCE_RATIO = 2.0
MIN_CROSS_BAND_SUPPORT = 2
MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD = 0.12


def _parse_float(value: Any) -> float | None:
    value = _python_value(value)
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _source_pair(
    external_high_resolution_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    records = _source_records_by_role(external_high_resolution_summary)
    pair = external_high_resolution_summary.get("sourcePair") or {}
    target_id = _int(pair.get("targetGaiaDR3SourceID"))
    counterpart_id = _int(pair.get("counterpartGaiaDR3SourceID"))
    if target_id is None or counterpart_id is None:
        raise RuntimeError("v20.23 requires both frozen Gaia DR3 source IDs from v20.19.")

    sources: list[dict[str, Any]] = []
    for role, source_id in (
        ("target-control", int(target_id)),
        ("catalog-counterpart", int(counterpart_id)),
    ):
        record = records.get(role) or {}
        metadata = record.get("metadata") or {}
        ra = _float(metadata.get("raDeg"))
        dec = _float(metadata.get("decDeg"))
        if ra is None or dec is None:
            raise RuntimeError(
                f"v20.23 requires frozen Gaia RA/Dec metadata for {role} from v20.19."
            )
        sources.append(
            {
                "sourceRole": role,
                "gaiaDR3SourceID": source_id,
                "raDeg": float(ra),
                "decDeg": float(dec),
                "gMag": _float(metadata.get("gMag")),
                "bpMag": _float(metadata.get("bpMag")),
                "rpMag": _float(metadata.get("rpMag")),
            }
        )

    separation = _angular_separation_arcsec(
        sources[0]["raDeg"],
        sources[0]["decDeg"],
        sources[1]["raDeg"],
        sources[1]["decDeg"],
    )
    return sources, float(separation)


def _pair_center(sources: list[dict[str, Any]]) -> tuple[float, float]:
    ra = statistics.fmean(float(item["raDeg"]) for item in sources)
    dec = statistics.fmean(float(item["decDeg"]) for item in sources)
    return float(ra), float(dec)


def _sia_size(size_deg: float, dec_deg: float) -> tuple[float, float]:
    cos_dec = max(0.05, abs(math.cos(math.radians(float(dec_deg)))))
    return float(size_deg / cos_dec), float(size_deg)


def _query_sia(
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> list[dict[str, Any]]:
    try:
        from astropy.io.votable import parse_single_table
    except Exception as exc:
        raise RuntimeError(
            "v20.23 requires astropy.io.votable for the DES DR2 single-epoch SIA response."
        ) from exc

    ra_size_deg, dec_size_deg = _sia_size(DISCOVERY_FIELD_SIZE_DEG, center_dec_deg)
    params = urllib.parse.urlencode(
        {
            "POS": f"{center_ra_deg:.10f},{center_dec_deg:.10f}",
            "SIZE": f"{ra_size_deg:.6f},{dec_size_deg:.6f}",
            "VERB": "2",
        }
    )
    request = urllib.request.Request(
        f"{DES_SIA_URL}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        body = response.read()
    if not body:
        raise RuntimeError("DES DR2 single-epoch SIA returned an empty response.")

    try:
        table = parse_single_table(io.BytesIO(body)).to_table(use_names_over_ids=True)
    except Exception as exc:
        prefix = body[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"DES DR2 single-epoch SIA response was not a readable VOTable: {prefix}"
        ) from exc

    return [
        {name: _python_value(row[name]) for name in table.colnames}
        for row in table
    ]


def _canonical_band(value: Any) -> str | None:
    text = (_text(value) or "").lower()
    if not text:
        return None
    for band in SUPPORTED_BANDS:
        if (
            text == band
            or text.startswith(band)
            or f" {band}" in text
            or f"_{band}" in text
        ):
            return band
    return None


def _candidate_exposures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        access_url = _text(_row_value(row, "access_url", "accessurl"))
        mjd = _parse_float(_row_value(row, "mjd_obs", "mjdobs", "mjd"))
        band = _canonical_band(_row_value(row, "obs_bandpass", "filter", "bandpass"))
        prodtype = (_text(_row_value(row, "prodtype", "product_type")) or "").lower()
        proctype = (_text(_row_value(row, "proctype", "processing_type")) or "").lower()
        if access_url is None or mjd is None or band is None:
            continue
        if prodtype and prodtype != "image":
            continue
        if "stack" in proctype or "coadd" in proctype:
            continue

        identifier = _text(
            _row_value(
                row,
                "title",
                "obs_id",
                "obsid",
                "reference",
                "siaref",
                "image_title",
            )
        ) or access_url

        candidates.append(
            {
                "accessURL": access_url,
                "mjd": float(mjd),
                "band": band,
                "proctype": proctype or None,
                "prodtype": prodtype or None,
                "identifier": identifier,
                "instrument": _text(
                    _row_value(row, "instrument_name", "instrument", "instrume")
                ),
                "seeingMetadataArcsec": _parse_float(
                    _row_value(row, "seeing", "fwhm", "psf_fwhm")
                ),
            }
        )

    dedup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in candidates:
        key = (
            str(item["identifier"]),
            str(item["band"]),
            int(round(float(item["mjd"]) * 1_000_000)),
        )
        dedup.setdefault(key, item)

    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dedup.values():
        by_band[str(item["band"])].append(item)

    selected: list[dict[str, Any]] = []
    for band in SUPPORTED_BANDS:
        items = sorted(by_band.get(band, []), key=lambda item: float(item["mjd"]))
        if len(items) > MAX_EXPOSURES_PER_BAND:
            indices = np.linspace(
                0,
                len(items) - 1,
                MAX_EXPOSURES_PER_BAND,
                dtype=int,
            )
            items = [items[int(index)] for index in indices]
        selected.extend(items)

    selected.sort(key=lambda item: float(item["mjd"]))
    if len(selected) > MAX_TOTAL_EXPOSURES:
        indices = np.linspace(
            0,
            len(selected) - 1,
            MAX_TOTAL_EXPOSURES,
            dtype=int,
        )
        selected = [selected[int(index)] for index in indices]
    return selected


def _source_cutout_url(
    access_url: str,
    *,
    ra_deg: float,
    dec_deg: float,
) -> str:
    parts = urllib.parse.urlsplit(access_url)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    ra_size_deg, dec_size_deg = _sia_size(LOCAL_CUTOUT_SIZE_DEG, dec_deg)
    query["POS"] = f"{float(ra_deg):.10f},{float(dec_deg):.10f}"
    query["SIZE"] = f"{ra_size_deg:.6f},{dec_size_deg:.6f}"
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urllib.parse.urlencode(query),
            parts.fragment,
        )
    )


def _download_fits_cutout(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        body = response.read()
    if not body:
        raise RuntimeError("DES DR2 single-epoch source cutout was empty.")
    return body


def _saturation_level(headers: list[Any]) -> float | None:
    values: list[float] = []
    for key in ("SATURATE", "SATURATA", "SATURATB", "SATLEVEL", "DATAMAX"):
        value = _header_float(headers, key)
        if value is not None and value > 0:
            values.append(float(value))
    return min(values) if values else None


def _gaussian_template(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x0: float,
    y0: float,
    sigma_pixels: float,
) -> np.ndarray:
    values = np.exp(
        -0.5
        * (
            np.square((x_grid - float(x0)) / sigma_pixels)
            + np.square((y_grid - float(y0)) / sigma_pixels)
        )
    )
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("Local forced-photometry Gaussian template is degenerate.")
    return values / total


def _local_source_fit(
    data: np.ndarray,
    header: Any,
    primary_header: Any,
    *,
    source: dict[str, Any],
) -> dict[str, Any]:
    source_x, source_y = _world_to_pixel(
        header,
        float(source["raDeg"]),
        float(source["decDeg"]),
    )
    pixel_scale = _pixel_scale_arcsec(header)
    if pixel_scale > MAX_PIXEL_SCALE_ARCSEC:
        raise RuntimeError(
            f"pixel-scale-too-coarse:{pixel_scale:.4f}-arcsec-per-pixel"
        )

    radius = LOCAL_FIT_RADIUS_PIXELS
    ix = int(round(source_x))
    iy = int(round(source_y))
    x0 = max(0, ix - radius)
    x1 = min(data.shape[1], ix + radius + 1)
    y0 = max(0, iy - radius)
    y1 = min(data.shape[0], iy + radius + 1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        raise RuntimeError("source-too-close-to-local-cutout-edge")

    crop = np.asarray(data[y0:y1, x0:x1], dtype=np.float64)
    finite = np.isfinite(crop)
    if int(np.count_nonzero(finite)) < 120:
        raise RuntimeError("too-few-finite-local-fit-pixels")

    headers = [header, primary_header]
    saturation = _saturation_level(headers)
    if saturation is not None:
        saturated_local = finite & (crop >= saturation * SATURATION_FRACTION)
        if bool(np.any(saturated_local)):
            raise RuntimeError("local-cutout-saturation-contamination")

    yy, xx = np.indices(crop.shape, dtype=np.float64)
    xx += float(x0)
    yy += float(y0)
    normalized_x = (xx - float(np.mean(xx))) / max(1.0, float(np.std(xx)))
    normalized_y = (yy - float(np.mean(yy))) / max(1.0, float(np.std(yy)))

    min_fwhm_arcsec = max(0.55, pixel_scale * 2.0)
    if min_fwhm_arcsec >= MAX_ABSOLUTE_FWHM_ARCSEC:
        raise RuntimeError("no-valid-local-psf-width-range")
    fwhm_grid = np.linspace(
        min_fwhm_arcsec,
        MAX_ABSOLUTE_FWHM_ARCSEC,
        17,
    )
    shift_grid = (-0.60, 0.0, 0.60)

    best: dict[str, Any] | None = None
    mask = finite.ravel()
    values = crop.ravel()[mask]

    for fwhm_arcsec in fwhm_grid:
        sigma_pixels = float(fwhm_arcsec / pixel_scale / 2.354820045)
        for dx in shift_grid:
            for dy in shift_grid:
                source_template = _gaussian_template(
                    xx,
                    yy,
                    source_x + dx,
                    source_y + dy,
                    sigma_pixels,
                )
                design_full = np.column_stack(
                    [
                        source_template.ravel(),
                        np.ones(source_template.size, dtype=np.float64),
                        normalized_x.ravel(),
                        normalized_y.ravel(),
                    ]
                )
                design = design_full[mask]
                try:
                    coefficients, covariance, condition, _ = _scaled_linear_fit(
                        design,
                        values,
                    )
                except Exception:
                    continue

                model = design @ coefficients
                residual = values - model
                center = float(np.median(residual))
                mad = float(np.median(np.abs(residual - center)))
                fit_design = design
                fit_values = values
                if math.isfinite(mad) and mad > 0:
                    robust_sigma = 1.4826 * mad
                    robust = np.abs(residual - center) <= 6.0 * robust_sigma
                    if int(np.count_nonzero(robust)) >= 100:
                        try:
                            coefficients, covariance, condition, _ = _scaled_linear_fit(
                                design[robust],
                                values[robust],
                            )
                            fit_design = design[robust]
                            fit_values = values[robust]
                            model = fit_design @ coefficients
                            residual = fit_values - model
                        except Exception:
                            continue

                amplitude = float(coefficients[0])
                if not math.isfinite(amplitude) or amplitude <= 0:
                    continue
                variances = np.diag(covariance)
                if len(variances) < 1 or not math.isfinite(float(variances[0])) or variances[0] <= 0:
                    continue
                amplitude_snr = amplitude / math.sqrt(float(variances[0]))
                denominator = float(
                    np.sum(np.square(fit_values - np.mean(fit_values)))
                )
                r2 = (
                    1.0 - float(np.sum(np.square(residual))) / denominator
                    if denominator > 0
                    else 0.0
                )
                score = float(np.mean(np.square(residual)))
                candidate = {
                    "score": score,
                    "amplitude": amplitude,
                    "amplitudeSNR": float(amplitude_snr),
                    "fwhmArcsec": float(fwhm_arcsec),
                    "sigmaPixels": sigma_pixels,
                    "shiftX": float(dx),
                    "shiftY": float(dy),
                    "scaledDesignCondition": float(condition),
                    "fitR2": float(r2),
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    if best is None:
        raise RuntimeError("no-positive-local-source-psf-fit")
    if best["scaledDesignCondition"] > MAX_SCALED_DESIGN_CONDITION:
        raise RuntimeError("local-source-fit-poorly-conditioned")
    if best["fitR2"] < MIN_FIT_R2:
        raise RuntimeError("local-source-fit-low-explained-variance")

    minimum_snr = (
        MIN_TARGET_AMPLITUDE_SNR
        if source["sourceRole"] == "target-control"
        else MIN_COUNTERPART_AMPLITUDE_SNR
    )
    if best["amplitudeSNR"] < minimum_snr:
        raise RuntimeError("local-source-amplitude-snr-too-low")

    magnitude, calibration = _photometric_magnitude(
        best["amplitude"],
        headers,
    )
    best.update(
        {
            "sourceRole": source["sourceRole"],
            "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
            "gaiaGMag": source.get("gMag"),
            "pixelScaleArcsec": pixel_scale,
            "sourcePixel": {"x": source_x, "y": source_y},
            "saturationADU": saturation,
            "magnitude": magnitude,
            "photometricCalibration": calibration,
            "exposureTimeSeconds": _header_float(headers, "EXPTIME", "EXPOSURE"),
            "magZero": _header_float(headers, "MAGZERO"),
            "magZpt": _header_float(headers, "MAGZPT"),
            "mjdHeader": _header_float(headers, "MJD-OBS", "MJD_OBS", "MJD"),
        }
    )
    return best


def _robust_standardize_magnitudes(magnitudes: np.ndarray) -> np.ndarray:
    values = np.asarray(magnitudes, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Local forced-photometry magnitudes have no finite variability scale.")
    return -(values - median) / scale


def _prepare_band_series(
    successful: list[dict[str, Any]],
    *,
    role: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for band in SUPPORTED_BANDS:
        items = [
            item
            for item in successful
            if item.get("band") == band and item.get("sourceRole") == role
        ]
        items.sort(key=lambda item: float(item["mjd"]))
        if len(items) < MIN_BAND_EPOCHS:
            continue

        times = np.asarray([float(item["mjd"]) for item in items], dtype=np.float64)
        magnitudes = np.asarray(
            [float(item["fit"]["magnitude"]) for item in items],
            dtype=np.float64,
        )

        median = float(np.median(magnitudes))
        mad = float(np.median(np.abs(magnitudes - median)))
        keep = np.ones(len(magnitudes), dtype=bool)
        if math.isfinite(mad) and mad > 0:
            keep = np.abs(magnitudes - median) <= 5.0 * 1.4826 * mad

        kept_times = times[keep]
        kept_magnitudes = magnitudes[keep]
        kept_items = [item for item, accepted in zip(items, keep.tolist()) if accepted]
        if len(kept_times) < MIN_BAND_EPOCHS:
            continue

        baseline = float(kept_times[-1] - kept_times[0])
        if baseline < MIN_BAND_BASELINE_DAYS:
            continue

        flux = _robust_standardize_magnitudes(kept_magnitudes)
        local_times = kept_times - float(kept_times[0])
        result[band] = {
            "times": local_times,
            "flux": flux,
            "sampleCount": int(len(local_times)),
            "baselineDays": baseline,
            "medianMagnitude": float(np.median(kept_magnitudes)),
            "medianFwhmArcsec": float(
                np.median([float(item["fit"]["fwhmArcsec"]) for item in kept_items])
            ),
            "medianDesignCondition": float(
                np.median(
                    [float(item["fit"]["scaledDesignCondition"]) for item in kept_items]
                )
            ),
            "medianAmplitudeSNR": float(
                np.median([float(item["fit"]["amplitudeSNR"]) for item in kept_items])
            ),
        }
    return result


def build_des_dr2_se_local_forced_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    external_high_resolution_summary: dict[str, Any],
    noirlab_image_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if noirlab_image_summary.get("recommendedNextTest") != "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY":
        raise RuntimeError(
            "v20.23 requires v20.22 to remain on the targeted time-series follow-up branch."
        )

    sources, pair_separation = _source_pair(external_high_resolution_summary)
    if pair_separation < MIN_INDEPENDENT_LOCAL_SEPARATION_ARCSEC:
        raise RuntimeError(
            "v20.23 independent local cutouts are only preregistered for a widely separated Gaia source pair."
        )

    source_by_role = {str(item["sourceRole"]): item for item in sources}
    target = source_by_role["target-control"]
    counterpart = source_by_role["catalog-counterpart"]
    center_ra, center_dec = _pair_center(sources)

    search = dict(
        ((noirlab_image_summary.get("distributedValidation") or {}).get("frequencySearch") or {})
    )
    if not search:
        search = dict(
            ((external_high_resolution_summary.get("distributedValidation") or {}).get("frequencySearch") or {})
        )
    total_frequencies = _int(search.get("totalFrequencies"))
    per_work = _int(search.get("frequenciesPerWorkUnit"))
    if (
        not search
        or total_frequencies is None
        or per_work is None
        or total_frequencies <= 0
        or per_work <= 0
    ):
        raise RuntimeError("v20.23 requires the frozen residual-frequency search definition.")

    print("   querying DES DR2 single-epoch SIA coverage", flush=True)
    sia_rows = _query_sia(
        center_ra_deg=center_ra,
        center_dec_deg=center_dec,
    )
    exposures = _candidate_exposures(sia_rows)
    print(f"   DES DR2 SE rows: {len(sia_rows)}", flush=True)
    print(f"   candidate single-epoch images: {len(exposures)}", flush=True)
    print(f"   actual Gaia pair separation: {pair_separation:.3f} arcsec", flush=True)
    print(
        "   target/counterpart are fit in separate local cutouts, so saturation of one source does not automatically reject the other",
        flush=True,
    )

    root = Path(output_dir) / "des-dr2-se-local-forced-photometry"
    root.mkdir(parents=True, exist_ok=True)

    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    source_attempts: Counter[str] = Counter()
    source_successes: Counter[str] = Counter()

    for index, exposure in enumerate(exposures, start=1):
        band = str(exposure["band"])
        print(
            f"      image {index}/{len(exposures)}: band={band} mjd={float(exposure['mjd']):.5f}",
            flush=True,
        )

        for role in ("target-control", "catalog-counterpart"):
            source = source_by_role[role]
            source_attempts[role] += 1
            cutout_url = _source_cutout_url(
                str(exposure["accessURL"]),
                ra_deg=float(source["raDeg"]),
                dec_deg=float(source["decDeg"]),
            )
            try:
                body = _download_fits_cutout(cutout_url)
                data, header, primary_header = _image_hdu_from_bytes(body)
                fit = _local_source_fit(
                    data,
                    header,
                    primary_header,
                    source=source,
                )
                mjd = fit.get("mjdHeader")
                if mjd is None:
                    mjd = float(exposure["mjd"])
                record = dict(exposure)
                record["mjd"] = float(mjd)
                record["sourceRole"] = role
                record["gaiaDR3SourceID"] = int(source["gaiaDR3SourceID"])
                record["cutoutURL"] = cutout_url
                record["fit"] = fit
                successful.append(record)
                source_successes[role] += 1
                print(
                    f"         {role}: accepted | "
                    f"FWHM={fit['fwhmArcsec']:.3f}\" | "
                    f"SNR={fit['amplitudeSNR']:.1f} | "
                    f"mag={fit['magnitude']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                message = str(exc)
                reason = message.split(":", 1)[0] if message else type(exc).__name__
                if isinstance(exc, (urllib.error.URLError, TimeoutError)):
                    reason = "download-error"
                failure_counts[f"{role}:{reason}"] += 1
                failures.append(
                    {
                        "identifier": exposure.get("identifier"),
                        "band": band,
                        "mjd": exposure.get("mjd"),
                        "sourceRole": role,
                        "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                        "reason": reason,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"         {role}: rejected: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    total_attempts = int(sum(source_attempts.values()))
    transport_failures = sum(
        count
        for key, count in failure_counts.items()
        if (
            "download-error" in key
            or "HTTP Error 500" in key
            or "HTTP Error 502" in key
            or "HTTP Error 503" in key
            or "HTTP Error 504" in key
            or "timed out" in key.lower()
        )
    )
    if total_attempts >= 6 and transport_failures >= max(6, total_attempts // 2):
        raise RuntimeError(
            "v20.23 found DES DR2 single-epoch coverage but image retrieval failed broadly; retry rather than converting a service outage into a scientific non-detection."
        )

    series_by_role = {
        role: _prepare_band_series(successful, role=role)
        for role in ("target-control", "catalog-counterpart")
    }

    diagnostics_path = root / "des-dr2-se-local-forced-diagnostics-v20.23.json"
    diagnostics = {
        "archive": "DES DR2 single-epoch SIA",
        "siaURL": DES_SIA_URL,
        "pairCenter": {"raDeg": center_ra, "decDeg": center_dec},
        "pairSeparationArcsec": pair_separation,
        "sourceDefinitions": sources,
        "siaRows": len(sia_rows),
        "candidateExposures": len(exposures),
        "sourceAttempts": dict(source_attempts),
        "sourceSuccesses": dict(source_successes),
        "failureReasons": dict(sorted(failure_counts.items())),
        "failures": failures,
        "successfulFits": [
            {
                "identifier": item.get("identifier"),
                "band": item.get("band"),
                "mjd": item.get("mjd"),
                "sourceRole": item.get("sourceRole"),
                "gaiaDR3SourceID": item.get("gaiaDR3SourceID"),
                "fit": item.get("fit"),
            }
            for item in successful
        ],
    }
    _write_json(diagnostics_path, diagnostics)

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    for role in ("target-control", "catalog-counterpart"):
        source = source_by_role[role]
        for band, series in series_by_role[role].items():
            dataset_id = f"{source_dataset_id}-des-dr2-se-local-{role}-{band}-v1"
            target_name = f"{source_dataset_id} DES DR2 SE local {role} {band}-band"
            dataset_path = root / f"{_safe(dataset_id)}.json"
            dataset = {
                "id": dataset_id,
                "targetName": target_name,
                "times": np.asarray(series["times"], dtype=np.float32).tolist(),
                "flux": np.asarray(series["flux"], dtype=np.float32).tolist(),
                "frequencySearch": search,
                "reference": {},
                "science": {
                    "role": "des-dr2-single-epoch-local-forced-photometry",
                    "sourceRole": role,
                    "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                    "band": band,
                    "pairSeparationArcsec": pair_separation,
                    "independentLocalCutout": True,
                    "otherSourceSaturationDoesNotVetoThisSeries": True,
                    "tessDriftExtrapolated": False,
                },
                "source": {
                    "mission": "Dark Energy Survey",
                    "dataRelease": "DR2",
                    "product": "single-epoch",
                    "archive": "NOIRLab Data Lab SIA",
                    "band": band,
                    "distributedSamples": int(series["sampleCount"]),
                    "baselineDays": float(series["baselineDays"]),
                },
            }
            _write_json(dataset_path, dataset)
            prepared = {
                "datasetID": dataset_id,
                "datasetPath": str(dataset_path.resolve()),
                "sourceRole": role,
                "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                "gaiaGMag": source.get("gMag"),
                "band": band,
                "sampleCount": int(series["sampleCount"]),
                "baselineDays": float(series["baselineDays"]),
                "medianMagnitude": float(series["medianMagnitude"]),
                "medianFwhmArcsec": float(series["medianFwhmArcsec"]),
                "medianDesignCondition": float(series["medianDesignCondition"]),
                "medianAmplitudeSNR": float(series["medianAmplitudeSNR"]),
            }
            prepared_series.append(prepared)
            dataset_entries.append(
                {
                    "id": dataset_id,
                    "path": str(dataset_path.resolve()),
                    "targetName": target_name,
                }
            )
            print(
                f"      {role} {band}: {series['sampleCount']} local forced-photometry epochs | baseline={series['baselineDays']:.1f} d",
                flush=True,
            )

    project_id: str | None = None
    project_path: str | None = None
    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "des-dr2-se-local-forced-photometry-v1"
        )
        manifest = {
            "id": project_id,
            "name": f"{source_project_id} — DES DR2 single-epoch local forced photometry",
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": "des-dr2-single-epoch-independent-local-forced-photometry",
                "archive": "DES DR2 single-epoch SIA",
                "workerSemantics": (
                    "Preparation performs source-local image/WCS/PSF/calibration work. "
                    "A saturated fit at one Gaia source does not veto the other source's independent local cutout. "
                    "Workers receive only accepted single-source band light curves and execute ordinary Lomb-Scargle "
                    "over the frozen residual-frequency band."
                ),
                "tessDriftExtrapolated": False,
            },
        }
        manifest_path = root / f"{_safe(project_id)}.json"
        _write_json(manifest_path, manifest)
        project_path = str(manifest_path.resolve())

    work_units_per_dataset = math.ceil(total_frequencies / per_work)
    return {
        "available": bool(dataset_entries),
        "version": "openstar.tess-des-dr2-se-local-forced-photometry-preparation.v1",
        "archive": "DES DR2 single-epoch SIA",
        "siaURL": DES_SIA_URL,
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-des-dr2-se-local-source-series",
        "sourcePair": external_high_resolution_summary.get("sourcePair"),
        "sourceDefinitions": sources,
        "pairCenter": {"raDeg": center_ra, "decDeg": center_dec},
        "pairSeparationArcsec": pair_separation,
        "frequencySearch": search,
        "tessDriftExtrapolated": False,
        "siaRows": len(sia_rows),
        "candidateExposures": len(exposures),
        "sourceAttempts": dict(source_attempts),
        "sourceSuccesses": dict(source_successes),
        "failureReasons": dict(sorted(failure_counts.items())),
        "diagnosticsPath": str(diagnostics_path.resolve()),
        "preparedSeries": prepared_series,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "qualityGuard": {
            "minimumIndependentLocalSeparationArcsec": MIN_INDEPENDENT_LOCAL_SEPARATION_ARCSEC,
            "localCutoutSizeDeg": LOCAL_CUTOUT_SIZE_DEG,
            "maximumPixelScaleArcsec": MAX_PIXEL_SCALE_ARCSEC,
            "maximumFwhmArcsec": MAX_ABSOLUTE_FWHM_ARCSEC,
            "maximumScaledDesignCondition": MAX_SCALED_DESIGN_CONDITION,
            "minimumFitR2": MIN_FIT_R2,
            "minimumTargetAmplitudeSNR": MIN_TARGET_AMPLITUDE_SNR,
            "minimumCounterpartAmplitudeSNR": MIN_COUNTERPART_AMPLITUDE_SNR,
            "saturationFraction": SATURATION_FRACTION,
            "minimumBandEpochs": MIN_BAND_EPOCHS,
            "minimumBandBaselineDays": MIN_BAND_BASELINE_DAYS,
            "minimumCrossBandSupport": MIN_CROSS_BAND_SUPPORT,
        },
        "interpretationGuard": (
            "v20.23 uses the DES DR2 single-epoch SIA archive rather than the sparse NSC image subset. "
            "Because the frozen Gaia sources are separated by tens of arcseconds, each source is fit in an independent "
            "small cutout. Saturation or fit failure at Blind C does not invalidate the counterpart measurement unless "
            "saturation contamination is present inside the counterpart's own local pixels. Each accepted local cutout "
            "must independently pass pixel-scale, saturation, PSF, conditioning, fit-quality, and source-SNR guards. "
            "A source is called supported only when residual-band variability recurs at consistent frequency in at least "
            "two independent bands. The TESS frequency-drift law is not extrapolated into DES epochs."
        ),
    }


def interpret_des_dr2_se_local_forced_project(
    *,
    project_status: dict[str, Any] | None,
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared_by_id = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedSeries") or []
        if item.get("datasetID")
    }
    results: list[dict[str, Any]] = []
    if project_status is not None:
        for dataset in project_status.get("datasets") or []:
            dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
            prepared = prepared_by_id.get(dataset_id)
            if prepared is not None:
                results.append(_dataset_result(dataset, prepared))

    target = _summarize_source("target-control", results)
    counterpart = _summarize_source("catalog-counterpart", results)

    target_supported = bool(target.get("sourceSupported"))
    counterpart_supported = bool(counterpart.get("sourceSupported"))
    target_suggestive = bool(target.get("sourceSuggestive"))
    counterpart_suggestive = bool(counterpart.get("sourceSuggestive"))

    candidate_exposures = int(preparation.get("candidateExposures") or 0)
    target_successes = int((preparation.get("sourceSuccesses") or {}).get("target-control") or 0)
    counterpart_successes = int(
        (preparation.get("sourceSuccesses") or {}).get("catalog-counterpart") or 0
    )
    prepared_roles = {
        str(item.get("sourceRole"))
        for item in preparation.get("preparedSeries") or []
    }

    if candidate_exposures == 0:
        classification = "DES_DR2_SE_NO_FIELD_COVERAGE"
        origin = "UNRESOLVED_DES_DR2_SE_NO_SINGLE_EPOCH_COVERAGE"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif counterpart_supported and target_supported:
        classification = "DES_DR2_SE_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_AND_COUNTERPART_SUPPORTED_BY_DES_DR2_SE_LOCAL_PHOTOMETRY"
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    elif counterpart_supported:
        classification = "DES_DR2_SE_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_BY_DES_DR2_SE_LOCAL_PHOTOMETRY"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_supported:
        classification = "DES_DR2_SE_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_SUPPORTED_BY_DES_DR2_SE_LOCAL_PHOTOMETRY"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif counterpart_suggestive and not target_suggestive:
        classification = "DES_DR2_SE_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE_BY_DES_DR2_SE_LOCAL_PHOTOMETRY"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif target_suggestive and not counterpart_suggestive:
        classification = "DES_DR2_SE_TARGET_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "TARGET_SUGGESTIVE_BY_DES_DR2_SE_LOCAL_PHOTOMETRY"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif not prepared_roles:
        if counterpart_successes > 0 or target_successes > 0:
            classification = "DES_DR2_SE_LOCAL_PHOTOMETRY_INSUFFICIENT_TIME_SERIES"
            origin = "UNRESOLVED_DES_DR2_SE_TIME_SERIES_LIMIT"
        else:
            classification = "DES_DR2_SE_NO_QUALIFYING_LOCAL_SOURCE_FITS"
            origin = "UNRESOLVED_DES_DR2_SE_IMAGE_FIT_QUALITY_LIMIT"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    else:
        classification = "DES_DR2_SE_LOCAL_SOURCE_ATTRIBUTION_UNRESOLVED"
        origin = "ARCHIVAL_DES_DR2_SE_SOURCE_ATTRIBUTION_UNRESOLVED"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"

    return {
        "version": "openstar.tess-des-dr2-se-local-forced-photometry.v1",
        "archive": preparation.get("archive"),
        "siaURL": preparation.get("siaURL"),
        "sourcePair": preparation.get("sourcePair"),
        "sourceDefinitions": preparation.get("sourceDefinitions"),
        "pairCenter": preparation.get("pairCenter"),
        "pairSeparationArcsec": preparation.get("pairSeparationArcsec"),
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "tessDriftExtrapolated": False,
        "siaRows": preparation.get("siaRows"),
        "candidateExposures": preparation.get("candidateExposures"),
        "sourceAttempts": preparation.get("sourceAttempts"),
        "sourceSuccesses": preparation.get("sourceSuccesses"),
        "failureReasons": preparation.get("failureReasons"),
        "diagnosticsPath": preparation.get("diagnosticsPath"),
        "componentResults": results,
        "targetControl": target,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "qualityGuard": preparation.get("qualityGuard"),
        "interpretationGuard": preparation.get("interpretationGuard"),
    }
