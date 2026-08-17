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

SIA_URL = "https://datalab.noirlab.edu/sia/nsc_dr2"
HTTP_TIMEOUT_SECONDS = 120
USER_AGENT = "OpenStar/20.22 NOIRLab-image-level-forced-photometry"
SUPPORTED_BANDS = ("g", "r", "i", "z")
SIA_FIELD_SIZE_DEG = 0.020
MAX_EXPOSURES_PER_BAND = 28
MAX_TOTAL_EXPOSURES = 96
MIN_BAND_EPOCHS = 10
MIN_BAND_BASELINE_DAYS = 15.0
MAX_PIXEL_SCALE_ARCSEC = 0.55
MAX_FWHM_PAIR_FRACTION = 0.72
MAX_ABSOLUTE_FWHM_ARCSEC = 1.80
MAX_TEMPLATE_CORRELATION = 0.45
MAX_SCALED_DESIGN_CONDITION = 200.0
MIN_FIT_R2 = 0.35
MIN_TARGET_AMPLITUDE_SNR = 8.0
MIN_COUNTERPART_AMPLITUDE_SNR = 3.0
MIN_PEAK_PROMINENCE_RATIO = 2.0
MIN_CROSS_BAND_SUPPORT = 2
MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD = 0.12
CUTOUT_MARGIN_PIXELS = 11
SATURATION_FRACTION = 0.90
CURRENT_TRIGGER = "NOIRLAB_IMAGE_LEVEL_FORCED_PHOTOMETRY"
HISTORICAL_TRIGGER = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
NEXT_ARCHIVE_TEST = "DES_DR2_SINGLE_EPOCH_LOCAL_FORCED_PHOTOMETRY"


class NOIRLabArchiveUnavailable(RuntimeError):
    """A transient NOIRLab service failure, rather than a scientific outcome."""


def _retryable_service_error(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, (TimeoutError, ConnectionError, urllib.error.URLError)):
            if isinstance(current, urllib.error.HTTPError):
                return current.code in {408, 425, 429} or current.code >= 500
            return True
        module = type(current).__module__.lower()
        name = type(current).__name__.lower()
        if module.startswith(("requests.", "httpx", "urllib3.")) and any(
            token in name for token in ("timeout", "connection", "network")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if np.ma.is_masked(value):
            return None
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _text(value: Any) -> str | None:
    value = _python_value(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_float(value: Any) -> float | None:
    value = _python_value(value)
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _row_value(row: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            value = _python_value(lowered[name.lower()])
            if value is not None:
                return value
    return None


def _angular_separation_arcsec(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    ra1 = math.radians(float(ra1_deg))
    dec1 = math.radians(float(dec1_deg))
    ra2 = math.radians(float(ra2_deg))
    dec2 = math.radians(float(dec2_deg))
    cos_sep = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep)) * 3600.0


def _source_records_by_role(
    external_high_resolution_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in external_high_resolution_summary.get("sourceRecords") or []:
        role = str(item.get("sourceRole") or "")
        if role:
            records[role] = item
    return records


def _frozen_source_pair(
    external_high_resolution_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    pair = external_high_resolution_summary.get("sourcePair") or {}
    if pair.get("version") == "openstar.current-source-pair.v1":
        sources = []
        for key, expected_role in (("target", "target-control"),
                                   ("counterpart", "catalog-counterpart")):
            current = pair.get(key) or {}
            source_id = _int(current.get("gaiaDR3SourceID"))
            ra = _float(current.get("raDeg"))
            dec = _float(current.get("decDeg"))
            if source_id is None or ra is None or dec is None:
                raise RuntimeError(f"Current source pair lacks frozen Gaia data for {key}.")
            if current.get("sourceRole") not in (None, expected_role):
                raise RuntimeError(f"Current source pair has an invalid role for {key}.")
            sources.append({"sourceRole": expected_role,
                            "gaiaDR3SourceID": int(source_id), "raDeg": float(ra),
                            "decDeg": float(dec)})
        separation = _angular_separation_arcsec(
            sources[0]["raDeg"], sources[0]["decDeg"],
            sources[1]["raDeg"], sources[1]["decDeg"])
        return sources, float(separation)

    records = _source_records_by_role(external_high_resolution_summary)
    target_id = _int(pair.get("targetGaiaDR3SourceID"))
    counterpart_id = _int(pair.get("counterpartGaiaDR3SourceID"))
    if target_id is None or counterpart_id is None:
        raise RuntimeError("v20.22 requires both frozen Gaia DR3 source IDs from v20.19.")

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
                f"v20.22 requires frozen Gaia RA/Dec metadata for {role} from v20.19."
            )
        sources.append(
            {
                "sourceRole": role,
                "gaiaDR3SourceID": source_id,
                "raDeg": float(ra),
                "decDeg": float(dec),
            }
        )

    # IMPORTANT: sourcePair.catalogSeparationArcsec from v20.19 is the
    # separation between the inferred TESS offset component and its matched
    # catalog counterpart. It is NOT the Blind-C-to-counterpart separation.
    # Image-level source geometry must therefore be recomputed directly from
    # the two frozen Gaia coordinates.
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


def _sia_size(center_dec_deg: float) -> tuple[float, float]:
    # Data Lab SIA SIZE is expressed in RA/Dec degrees rather than an
    # automatically convergence-corrected square. Preserve a roughly square
    # on-sky cutout by expanding the RA coordinate span by cos(dec).
    cos_dec = max(0.05, abs(math.cos(math.radians(float(center_dec_deg)))))
    return float(SIA_FIELD_SIZE_DEG / cos_dec), float(SIA_FIELD_SIZE_DEG)


def _query_sia(center_ra_deg: float, center_dec_deg: float) -> list[dict[str, Any]]:
    try:
        from astropy.io.votable import parse_single_table
    except Exception as exc:
        raise RuntimeError(
            "v20.22 requires astropy.io.votable for the public NOIRLab SIA response."
        ) from exc

    ra_size_deg, dec_size_deg = _sia_size(center_dec_deg)
    params = urllib.parse.urlencode(
        {
            "POS": f"{center_ra_deg:.10f},{center_dec_deg:.10f}",
            "SIZE": f"{ra_size_deg:.6f},{dec_size_deg:.6f}",
            "VERB": "2",
        }
    )
    request = urllib.request.Request(
        f"{SIA_URL}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = response.read()
    except Exception as exc:
        if _retryable_service_error(exc):
            raise NOIRLabArchiveUnavailable(
                f"NOIRLab SIA service unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        raise
    if not body:
        raise RuntimeError("NOIRLab NSC DR2 SIA returned an empty response.")

    try:
        table = parse_single_table(io.BytesIO(body)).to_table(use_names_over_ids=True)
    except Exception as exc:
        prefix = body[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"NOIRLab NSC DR2 SIA response was not a readable VOTable: {prefix}"
        ) from exc

    rows: list[dict[str, Any]] = []
    for row in table:
        rows.append({name: _python_value(row[name]) for name in table.colnames})
    return rows


def _canonical_band(value: Any) -> str | None:
    text = (_text(value) or "").lower()
    if not text:
        return None
    for band in SUPPORTED_BANDS:
        if text == band or text.startswith(band) or f" {band}" in text or f"_{band}" in text:
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
            _row_value(row, "title", "obs_id", "obsid", "reference", "siaref")
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

    selected: list[dict[str, Any]] = []
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dedup.values():
        by_band[str(item["band"])].append(item)

    for band in SUPPORTED_BANDS:
        items = sorted(by_band.get(band, []), key=lambda item: float(item["mjd"]))
        if len(items) > MAX_EXPOSURES_PER_BAND:
            indices = np.linspace(0, len(items) - 1, MAX_EXPOSURES_PER_BAND, dtype=int)
            items = [items[int(index)] for index in indices]
        selected.extend(items)

    selected.sort(key=lambda item: float(item["mjd"]))
    if len(selected) > MAX_TOTAL_EXPOSURES:
        indices = np.linspace(0, len(selected) - 1, MAX_TOTAL_EXPOSURES, dtype=int)
        selected = [selected[int(index)] for index in indices]
    return selected


def _rewrite_cutout_url(
    access_url: str,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> str:
    parts = urllib.parse.urlsplit(access_url)
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    ra_size_deg, dec_size_deg = _sia_size(center_dec_deg)
    query["POS"] = f"{center_ra_deg:.10f},{center_dec_deg:.10f}"
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
        raise RuntimeError("NOIRLab SIA image cutout was empty.")
    return body


def _header_float(headers: list[Any], *keys: str) -> float | None:
    for header in headers:
        if header is None:
            continue
        for key in keys:
            try:
                value = header.get(key)
            except Exception:
                value = None
            result = _parse_float(value)
            if result is not None:
                return float(result)
    return None


def _image_hdu_from_bytes(body: bytes):
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception as exc:
        raise ImportError("v20.22 requires astropy FITS/WCS support.") from exc

    hdul = fits.open(io.BytesIO(body), memmap=False)
    try:
        primary_header = hdul[0].header.copy() if len(hdul) else None
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            array = np.asarray(data)
            if array.ndim != 2 or min(array.shape) < 8:
                continue
            header = hdu.header.copy()
            try:
                wcs = WCS(header)
                if not bool(wcs.has_celestial):
                    continue
            except Exception:
                continue
            return np.asarray(array, dtype=np.float64), header, primary_header
    finally:
        hdul.close()
    raise RuntimeError("Downloaded NOIRLab FITS cutout has no usable 2-D celestial image HDU.")


def _pixel_scale_arcsec(header: Any) -> float:
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
    except Exception as exc:
        raise ImportError("v20.22 requires astropy WCS utilities.") from exc
    wcs = WCS(header).celestial
    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=np.float64) * 3600.0
    scales = scales[np.isfinite(scales) & (scales > 0)]
    if len(scales) == 0:
        raise RuntimeError("NOIRLab cutout has no finite celestial pixel scale.")
    return float(np.median(scales))


def _world_to_pixel(header: Any, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    try:
        from astropy.wcs import WCS
    except Exception as exc:
        raise ImportError("v20.22 requires astropy WCS support.") from exc
    wcs = WCS(header).celestial
    x, y = wcs.world_to_pixel_values(float(ra_deg), float(dec_deg))
    x = float(x)
    y = float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise RuntimeError("Frozen Gaia coordinate does not map to a finite cutout pixel.")
    return x, y


def _local_peak(data: np.ndarray, x: float, y: float, radius: int = 2) -> float | None:
    ix = int(round(x))
    iy = int(round(y))
    x0 = max(0, ix - radius)
    x1 = min(data.shape[1], ix + radius + 1)
    y0 = max(0, iy - radius)
    y1 = min(data.shape[0], iy + radius + 1)
    values = np.asarray(data[y0:y1, x0:x1], dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.max(values))


def _saturation_level(headers: list[Any]) -> float | None:
    values: list[float] = []
    for key in ("SATURATE", "SATURATA", "SATURATB", "SATLEVEL", "DATAMAX"):
        value = _header_float(headers, key)
        if value is not None and value > 0:
            values.append(float(value))
    return min(values) if values else None


def _photometric_magnitude(amplitude: float, headers: list[Any]) -> tuple[float, str]:
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise RuntimeError("Forced-photometry amplitude is not positive and finite.")
    magzero = _header_float(headers, "MAGZERO")
    if magzero is not None:
        return float(magzero - 2.5 * math.log10(amplitude)), "MAGZERO"
    magzpt = _header_float(headers, "MAGZPT")
    exptime = _header_float(headers, "EXPTIME", "EXPOSURE")
    if magzpt is not None and exptime is not None and exptime > 0:
        return float(magzpt - 2.5 * math.log10(amplitude / exptime)), "MAGZPT"
    raise RuntimeError("NOIRLab cutout lacks usable MAGZERO or MAGZPT+EXPTIME calibration.")


def _scaled_linear_fit(design: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    norms = np.linalg.norm(design, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("Forced-photometry design matrix has a degenerate column.")
    scaled = design / norms
    condition = float(np.linalg.cond(scaled))
    coefficients_scaled, _, _, _ = np.linalg.lstsq(scaled, values, rcond=None)
    coefficients = coefficients_scaled / norms
    model = design @ coefficients
    residuals = values - model
    dof = max(1, len(values) - design.shape[1])
    variance = float(np.sum(np.square(residuals)) / dof)
    covariance_scaled = variance * np.linalg.pinv(scaled.T @ scaled)
    scale_matrix = np.diag(1.0 / norms)
    covariance = scale_matrix @ covariance_scaled @ scale_matrix
    return coefficients, covariance, condition, variance


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
        raise RuntimeError("Forced-photometry Gaussian template is degenerate.")
    return values / total


def _forced_two_source_fit(
    data: np.ndarray,
    header: Any,
    primary_header: Any,
    *,
    target: dict[str, Any],
    counterpart: dict[str, Any],
    pair_separation_arcsec: float,
) -> dict[str, Any]:
    target_x, target_y = _world_to_pixel(header, target["raDeg"], target["decDeg"])
    counterpart_x, counterpart_y = _world_to_pixel(
        header, counterpart["raDeg"], counterpart["decDeg"]
    )
    pixel_scale = _pixel_scale_arcsec(header)
    if pixel_scale > MAX_PIXEL_SCALE_ARCSEC:
        raise RuntimeError(
            f"pixel-scale-too-coarse:{pixel_scale:.4f}-arcsec-per-pixel"
        )

    margin = CUTOUT_MARGIN_PIXELS
    x0 = max(0, int(math.floor(min(target_x, counterpart_x))) - margin)
    x1 = min(data.shape[1], int(math.ceil(max(target_x, counterpart_x))) + margin + 1)
    y0 = max(0, int(math.floor(min(target_y, counterpart_y))) - margin)
    y1 = min(data.shape[0], int(math.ceil(max(target_y, counterpart_y))) + margin + 1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        raise RuntimeError("pair-too-close-to-cutout-edge")

    headers = [header, primary_header]
    saturation = _saturation_level(headers)
    target_peak = _local_peak(data, target_x, target_y)
    counterpart_peak = _local_peak(data, counterpart_x, counterpart_y)
    if saturation is not None:
        if target_peak is not None and target_peak >= saturation * SATURATION_FRACTION:
            raise RuntimeError("target-saturated")
        if counterpart_peak is not None and counterpart_peak >= saturation * SATURATION_FRACTION:
            raise RuntimeError("counterpart-saturated")

    crop = np.asarray(data[y0:y1, x0:x1], dtype=np.float64)
    yy, xx = np.indices(crop.shape, dtype=np.float64)
    xx += float(x0)
    yy += float(y0)
    finite = np.isfinite(crop)
    if int(np.count_nonzero(finite)) < 120:
        raise RuntimeError("too-few-finite-fit-pixels")

    normalized_x = (xx - float(np.mean(xx))) / max(1.0, float(np.std(xx)))
    normalized_y = (yy - float(np.mean(yy))) / max(1.0, float(np.std(yy)))

    max_fwhm_arcsec = min(
        MAX_ABSOLUTE_FWHM_ARCSEC,
        float(pair_separation_arcsec) * MAX_FWHM_PAIR_FRACTION,
    )
    min_fwhm_arcsec = max(0.55, pixel_scale * 2.2)
    if min_fwhm_arcsec >= max_fwhm_arcsec:
        raise RuntimeError("no-valid-psf-width-range-for-pair-separation")
    fwhm_grid = np.linspace(min_fwhm_arcsec, max_fwhm_arcsec, 17)
    shift_grid = (-0.60, 0.0, 0.60)

    best: dict[str, Any] | None = None
    values_all = crop[finite]
    for fwhm_arcsec in fwhm_grid:
        sigma_pixels = float(fwhm_arcsec / pixel_scale / 2.354820045)
        for dx in shift_grid:
            for dy in shift_grid:
                t = _gaussian_template(
                    xx,
                    yy,
                    target_x + dx,
                    target_y + dy,
                    sigma_pixels,
                )
                c = _gaussian_template(
                    xx,
                    yy,
                    counterpart_x + dx,
                    counterpart_y + dy,
                    sigma_pixels,
                )
                design_full = np.column_stack(
                    [
                        t.ravel(),
                        c.ravel(),
                        np.ones(t.size, dtype=np.float64),
                        normalized_x.ravel(),
                        normalized_y.ravel(),
                    ]
                )
                mask = finite.ravel()
                design = design_full[mask]
                values = crop.ravel()[mask]
                try:
                    coefficients, covariance, condition, _ = _scaled_linear_fit(design, values)
                except Exception:
                    continue
                model = design @ coefficients
                residual = values - model
                center = float(np.median(residual))
                mad = float(np.median(np.abs(residual - center)))
                if math.isfinite(mad) and mad > 0:
                    robust_sigma = 1.4826 * mad
                    robust = np.abs(residual - center) <= 6.0 * robust_sigma
                    if int(np.count_nonzero(robust)) >= 100:
                        try:
                            coefficients, covariance, condition, _ = _scaled_linear_fit(
                                design[robust], values[robust]
                            )
                            model = design[robust] @ coefficients
                            residual = values[robust] - model
                            fit_values = values[robust]
                        except Exception:
                            continue
                    else:
                        fit_values = values
                else:
                    fit_values = values

                target_amp = float(coefficients[0])
                counterpart_amp = float(coefficients[1])
                if target_amp <= 0 or counterpart_amp <= 0:
                    continue
                variances = np.diag(covariance)
                if len(variances) < 2 or variances[0] <= 0 or variances[1] <= 0:
                    continue
                target_snr = target_amp / math.sqrt(float(variances[0]))
                counterpart_snr = counterpart_amp / math.sqrt(float(variances[1]))
                tvec = t[finite]
                cvec = c[finite]
                correlation = float(
                    np.dot(tvec, cvec)
                    / max(1e-20, np.linalg.norm(tvec) * np.linalg.norm(cvec))
                )
                denominator = float(np.sum(np.square(fit_values - np.mean(fit_values))))
                r2 = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0 else 0.0
                score = float(np.mean(np.square(residual)))
                candidate = {
                    "score": score,
                    "targetAmplitude": target_amp,
                    "counterpartAmplitude": counterpart_amp,
                    "targetAmplitudeSNR": float(target_snr),
                    "counterpartAmplitudeSNR": float(counterpart_snr),
                    "fwhmArcsec": float(fwhm_arcsec),
                    "sigmaPixels": sigma_pixels,
                    "sharedShiftX": float(dx),
                    "sharedShiftY": float(dy),
                    "templateCorrelation": correlation,
                    "scaledDesignCondition": float(condition),
                    "fitR2": float(r2),
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    if best is None:
        raise RuntimeError("no-positive-two-source-psf-fit")
    if best["fwhmArcsec"] > max_fwhm_arcsec + 1e-9:
        raise RuntimeError("psf-too-broad-for-source-resolution")
    if best["templateCorrelation"] > MAX_TEMPLATE_CORRELATION:
        raise RuntimeError("source-templates-too-correlated")
    if best["scaledDesignCondition"] > MAX_SCALED_DESIGN_CONDITION:
        raise RuntimeError("two-source-fit-poorly-conditioned")
    if best["fitR2"] < MIN_FIT_R2:
        raise RuntimeError("two-source-fit-low-explained-variance")
    if best["targetAmplitudeSNR"] < MIN_TARGET_AMPLITUDE_SNR:
        raise RuntimeError("target-amplitude-snr-too-low")
    if best["counterpartAmplitudeSNR"] < MIN_COUNTERPART_AMPLITUDE_SNR:
        raise RuntimeError("counterpart-amplitude-snr-too-low")

    target_mag, target_calibration = _photometric_magnitude(
        best["targetAmplitude"], headers
    )
    counterpart_mag, counterpart_calibration = _photometric_magnitude(
        best["counterpartAmplitude"], headers
    )
    if target_calibration != counterpart_calibration:
        raise RuntimeError("inconsistent-photometric-calibration-path")

    best.update(
        {
            "pixelScaleArcsec": pixel_scale,
            "targetPixel": {"x": target_x, "y": target_y},
            "counterpartPixel": {"x": counterpart_x, "y": counterpart_y},
            "pixelPairSeparation": float(
                math.hypot(target_x - counterpart_x, target_y - counterpart_y)
            ),
            "targetPeakADU": target_peak,
            "counterpartPeakADU": counterpart_peak,
            "saturationADU": saturation,
            "targetMagnitude": target_mag,
            "counterpartMagnitude": counterpart_mag,
            "photometricCalibration": target_calibration,
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
        raise RuntimeError("Forced-photometry magnitudes have no finite variability scale.")
    return -(values - median) / scale


def _prepare_band_series(
    successful: list[dict[str, Any]],
    *,
    role: str,
    failure_counts: Counter[str] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for band in SUPPORTED_BANDS:
        items = [item for item in successful if item.get("band") == band]
        items.sort(key=lambda item: float(item["mjd"]))
        if len(items) < MIN_BAND_EPOCHS:
            continue
        magnitudes = np.asarray(
            [
                float(item["fit"]["targetMagnitude" if role == "target-control" else "counterpartMagnitude"])
                for item in items
            ],
            dtype=np.float64,
        )
        times = np.asarray([float(item["mjd"]) for item in items], dtype=np.float64)
        median = float(np.median(magnitudes))
        mad = float(np.median(np.abs(magnitudes - median)))
        keep = np.ones(len(items), dtype=bool)
        if math.isfinite(mad) and mad > 0:
            keep = np.abs(magnitudes - median) <= 8.0 * 1.4826 * mad
        kept_times = times[keep]
        kept_mags = magnitudes[keep]
        kept_items = [item for item, accepted in zip(items, keep.tolist()) if accepted]
        if len(kept_times) < MIN_BAND_EPOCHS:
            continue
        baseline = float(kept_times[-1] - kept_times[0])
        if baseline < MIN_BAND_BASELINE_DAYS:
            continue
        try:
            flux = _robust_standardize_magnitudes(kept_mags)
        except RuntimeError as exc:
            if str(exc) != "Forced-photometry magnitudes have no finite variability scale.":
                raise
            if failure_counts is not None:
                failure_counts[f"{role}-{band}-flat-magnitude-series"] += 1
            continue
        result[band] = {
            "times": kept_times - float(kept_times[0]),
            "flux": flux,
            "sampleCount": int(len(kept_times)),
            "baselineDays": baseline,
            "medianMagnitude": float(np.median(kept_mags)),
            "medianFwhmArcsec": float(
                np.median([float(item["fit"]["fwhmArcsec"]) for item in kept_items])
            ),
            "medianTemplateCorrelation": float(
                np.median([float(item["fit"]["templateCorrelation"]) for item in kept_items])
            ),
            "medianDesignCondition": float(
                np.median([float(item["fit"]["scaledDesignCondition"]) for item in kept_items])
            ),
        }
    return result


def _dataset_result(project_dataset: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    frequency = _float(
        project_dataset.get("candidateFrequency")
        if project_dataset.get("candidateFrequency") is not None
        else project_dataset.get("bestFrequency")
    )
    period = _float(
        project_dataset.get("candidatePeriodDays")
        if project_dataset.get("candidatePeriodDays") is not None
        else project_dataset.get("bestPeriodDays")
    )
    power = _float(
        project_dataset.get("candidatePower")
        if project_dataset.get("candidatePower") is not None
        else project_dataset.get("bestPower")
    )
    prominence = _float(project_dataset.get("candidatePeakProminenceRatio"))
    status = str(project_dataset.get("periodStatus") or "")
    coverage = project_dataset.get("coverageComplete")
    accepted = bool(
        status == "RELIABLE"
        and (coverage is None or bool(coverage))
        and prominence is not None
        and prominence >= MIN_PEAK_PROMINENCE_RATIO
        and frequency is not None
        and frequency > 0
    )
    return {
        "datasetID": prepared.get("datasetID"),
        "sourceRole": prepared.get("sourceRole"),
        "gaiaDR3SourceID": prepared.get("gaiaDR3SourceID"),
        "band": prepared.get("band"),
        "sampleCount": prepared.get("sampleCount"),
        "baselineDays": prepared.get("baselineDays"),
        "periodStatus": status or None,
        "periodConfidence": project_dataset.get("periodConfidence"),
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "acceptedResidualBandVariability": accepted,
    }


def _summarize_source(role: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    source_results = [item for item in results if item.get("sourceRole") == role]
    accepted = [item for item in source_results if item.get("acceptedResidualBandVariability")]
    frequencies = [
        float(item["candidateFrequency"])
        for item in accepted
        if item.get("candidateFrequency") is not None
    ]
    median_frequency = None
    relative_spread = None
    supported = False
    if frequencies:
        median_frequency = float(np.median(np.asarray(frequencies, dtype=np.float64)))
        if median_frequency > 0:
            relative_spread = float((max(frequencies) - min(frequencies)) / median_frequency)
        supported = bool(
            len(frequencies) >= MIN_CROSS_BAND_SUPPORT
            and relative_spread is not None
            and relative_spread <= MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD
        )
    return {
        "sourceRole": role,
        "bandResults": source_results,
        "acceptedBands": sorted(str(item.get("band")) for item in accepted),
        "acceptedBandCount": len(accepted),
        "medianAcceptedFrequency": median_frequency,
        "crossBandRelativeFrequencySpread": relative_spread,
        "sourceSupported": supported,
        "sourceSuggestive": bool(accepted) and not supported,
    }


def build_noirlab_image_forced_photometry_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    external_high_resolution_summary: dict[str, Any],
    nsc_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if nsc_summary.get("recommendedNextTest") not in {CURRENT_TRIGGER, HISTORICAL_TRIGGER}:
        raise RuntimeError(
            "v20.22 requires v20.21 to leave the investigation at targeted high-resolution follow-up."
        )

    pair_evidence = (nsc_summary if (nsc_summary.get("sourcePair") or {}).get("version")
                     == "openstar.current-source-pair.v1" else external_high_resolution_summary)
    sources, pair_separation = _frozen_source_pair(pair_evidence)
    external_pair = dict(pair_evidence.get("sourcePair") or {})
    catalog_association_separation = _float(external_pair.get("catalogSeparationArcsec"))
    source_pair = dict(external_pair)
    source_pair["gaiaPairSeparationArcsec"] = float(pair_separation)
    source_pair["catalogAssociationSeparationArcsec"] = catalog_association_separation
    center_ra, center_dec = _pair_center(sources)
    source_by_role = {str(item["sourceRole"]): item for item in sources}
    target = source_by_role["target-control"]
    counterpart = source_by_role["catalog-counterpart"]

    search = {}
    for evidence in (nsc_summary, external_high_resolution_summary):
        search = dict(evidence.get("frequencySearch") or
                      (evidence.get("distributedValidation") or {}).get("frequencySearch") or {})
        if search:
            break
    total_frequencies = _int(search.get("totalFrequencies"))
    per_work = _int(search.get("frequenciesPerWorkUnit"))
    if not search or total_frequencies is None or per_work is None or total_frequencies <= 0 or per_work <= 0:
        raise RuntimeError(
            "v20.22 requires the frozen residual-frequency search definition from v20.19."
        )

    print("   querying the public NSC DR2 SIA service for single-epoch image cutouts", flush=True)
    sia_rows = _query_sia(center_ra, center_dec)
    exposures = _candidate_exposures(sia_rows)
    print(f"   SIA image candidates after single-epoch/filter guards: {len(exposures)}", flush=True)

    if not exposures:
        return {
            "available": False,
            "version": "openstar.tess-noirlab-image-forced-photometry-preparation.v1",
            "archive": "NOIRLab NSC DR2 SIA calibrated images",
            "siaURL": SIA_URL,
            "projectID": None,
            "projectPath": None,
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "sourcePair": source_pair,
            "pairSeparationArcsec": pair_separation,
            "catalogAssociationSeparationArcsec": catalog_association_separation,
            "frequencySearch": search,
            "tessDriftExtrapolated": False,
            "siaRows": len(sia_rows),
            "candidateExposures": 0,
            "successfulForcedPhotometryExposures": 0,
            "failureReasons": {},
            "preparedSeries": [],
            "totalWorkUnits": 0,
            "qualityGuard": _quality_guard(pair_separation),
            "interpretationGuard": _interpretation_guard(),
        }

    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    for index, exposure in enumerate(exposures, start=1):
        band = str(exposure["band"])
        print(
            f"      image {index}/{len(exposures)}: band={band} mjd={float(exposure['mjd']):.5f}",
            flush=True,
        )
        cutout_url = _rewrite_cutout_url(
            str(exposure["accessURL"]),
            center_ra_deg=center_ra,
            center_dec_deg=center_dec,
        )
        try:
            body = _download_fits_cutout(cutout_url)
            data, header, primary_header = _image_hdu_from_bytes(body)
            fit = _forced_two_source_fit(
                data,
                header,
                primary_header,
                target=target,
                counterpart=counterpart,
                pair_separation_arcsec=pair_separation,
            )
            mjd = fit.get("mjdHeader")
            if mjd is None:
                mjd = float(exposure["mjd"])
            record = dict(exposure)
            record["mjd"] = float(mjd)
            record["cutoutURL"] = cutout_url
            record["fit"] = fit
            successful.append(record)
            print(
                "         accepted: "
                f"FWHM={fit['fwhmArcsec']:.3f}\" | "
                f"corr={fit['templateCorrelation']:.3f} | "
                f"SNR=({fit['targetAmplitudeSNR']:.1f},{fit['counterpartAmplitudeSNR']:.1f})",
                flush=True,
            )
        except (ImportError, ValueError):
            # Missing local science dependencies and programming errors are
            # not archive-quality rejections and must remain non-retryable.
            raise
        except Exception as exc:
            message = str(exc)
            reason = message.split(":", 1)[0] if message else type(exc).__name__
            if _retryable_service_error(exc):
                reason = "download-error"
            failure_counts[reason] += 1
            failures.append(
                {
                    "identifier": exposure.get("identifier"),
                    "band": band,
                    "mjd": exposure.get("mjd"),
                    "reason": reason,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"         rejected: {type(exc).__name__}: {exc}", flush=True)

    if len(exposures) >= 3:
        transport_like = failure_counts["download-error"]
        if transport_like >= max(3, math.ceil(len(exposures) / 2)):
            raise NOIRLabArchiveUnavailable(
                "v20.22 discovered public NOIRLab images but image retrieval failed broadly; retry rather than converting an archive/service failure into a scientific non-detection."
            )

    series_by_role = {
        role: _prepare_band_series(successful, role=role, failure_counts=failure_counts)
        for role in ("target-control", "catalog-counterpart")
    }

    root = Path(output_dir) / "noirlab-image-forced-photometry"
    root.mkdir(parents=True, exist_ok=True)
    diagnostics_path = root / "forced-photometry-diagnostics-v20.22.json"
    diagnostics = {
        "archive": "NOIRLab NSC DR2 SIA calibrated images",
        "pairCenter": {"raDeg": center_ra, "decDeg": center_dec},
        "pairSeparationArcsec": pair_separation,
        "siaRows": len(sia_rows),
        "candidateExposures": len(exposures),
        "successfulForcedPhotometryExposures": len(successful),
        "failureReasons": dict(sorted(failure_counts.items())),
        "failures": failures,
        "successfulFits": [
            {
                "identifier": item.get("identifier"),
                "band": item.get("band"),
                "mjd": item.get("mjd"),
                "proctype": item.get("proctype"),
                "instrument": item.get("instrument"),
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
            dataset_id = f"{source_dataset_id}-noirlab-forced-{role}-{band}-v1"
            target_name = f"{source_dataset_id} NOIRLab forced {role} {band}-band"
            dataset_path = root / f"{_safe(dataset_id)}.json"
            dataset = {
                "id": dataset_id,
                "targetName": target_name,
                "times": np.asarray(series["times"], dtype=np.float32).tolist(),
                "flux": np.asarray(series["flux"], dtype=np.float32).tolist(),
                "frequencySearch": search,
                "reference": {},
                "science": {
                    "role": "noirlab-image-level-two-source-forced-photometry",
                    "sourceRole": role,
                    "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                    "band": band,
                    "pairSeparationArcsec": pair_separation,
                    "tessDriftExtrapolated": False,
                },
                "source": {
                    "mission": "NOIRLab archival imaging",
                    "archive": "NSC DR2 SIA",
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
                "band": band,
                "sampleCount": int(series["sampleCount"]),
                "baselineDays": float(series["baselineDays"]),
                "medianMagnitude": float(series["medianMagnitude"]),
                "medianFwhmArcsec": float(series["medianFwhmArcsec"]),
                "medianTemplateCorrelation": float(series["medianTemplateCorrelation"]),
                "medianDesignCondition": float(series["medianDesignCondition"]),
            }
            prepared_series.append(prepared)
            dataset_entries.append(
                {"id": dataset_id, "path": str(dataset_path.resolve()), "targetName": target_name}
            )
            print(
                f"      {role} {band}: {series['sampleCount']} calibrated forced-photometry epochs | baseline={series['baselineDays']:.1f} d",
                flush=True,
            )

    project_id: str | None = None
    project_path: str | None = None
    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "noirlab-image-forced-photometry-v1"
        )
        manifest = {
            "id": project_id,
            "name": f"{source_project_id} — NOIRLab image-level forced photometry",
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": "noirlab-image-level-two-source-forced-photometry",
                "archive": "NOIRLab NSC DR2 SIA calibrated images",
                "workerSemantics": (
                    "Preparation performs image/WCS/PSF/calibration work locally. Workers receive only frozen, "
                    "source-resolved single-band light curves and execute ordinary Lomb-Scargle over the frozen residual-frequency band."
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
        "version": "openstar.tess-noirlab-image-forced-photometry-preparation.v1",
        "archive": "NOIRLab NSC DR2 SIA calibrated images",
        "siaURL": SIA_URL,
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-image-level-forced-source-series",
        "sourcePair": source_pair,
        "pairCenter": {"raDeg": center_ra, "decDeg": center_dec},
        "pairSeparationArcsec": pair_separation,
        "catalogAssociationSeparationArcsec": catalog_association_separation,
        "frequencySearch": search,
        "tessDriftExtrapolated": False,
        "siaRows": len(sia_rows),
        "candidateExposures": len(exposures),
        "successfulForcedPhotometryExposures": len(successful),
        "failureReasons": dict(sorted(failure_counts.items())),
        "diagnosticsPath": str(diagnostics_path.resolve()),
        "preparedSeries": prepared_series,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "qualityGuard": _quality_guard(pair_separation),
        "interpretationGuard": _interpretation_guard(),
    }


def _quality_guard(pair_separation_arcsec: float) -> dict[str, Any]:
    return {
        "maximumPixelScaleArcsec": MAX_PIXEL_SCALE_ARCSEC,
        "maximumFwhmArcsec": min(
            MAX_ABSOLUTE_FWHM_ARCSEC,
            float(pair_separation_arcsec) * MAX_FWHM_PAIR_FRACTION,
        ),
        "maximumFwhmPairFraction": MAX_FWHM_PAIR_FRACTION,
        "maximumTemplateCorrelation": MAX_TEMPLATE_CORRELATION,
        "maximumScaledDesignCondition": MAX_SCALED_DESIGN_CONDITION,
        "minimumFitR2": MIN_FIT_R2,
        "minimumTargetAmplitudeSNR": MIN_TARGET_AMPLITUDE_SNR,
        "minimumCounterpartAmplitudeSNR": MIN_COUNTERPART_AMPLITUDE_SNR,
        "saturationFraction": SATURATION_FRACTION,
        "minimumBandEpochs": MIN_BAND_EPOCHS,
        "minimumBandBaselineDays": MIN_BAND_BASELINE_DAYS,
        "minimumCrossBandSupport": MIN_CROSS_BAND_SUPPORT,
    }


def _interpretation_guard() -> str:
    return (
        "v20.22 bypasses the NSC extracted-object catalog and returns to public calibrated NOIRLab image pixels. "
        "The two source positions are frozen from Gaia before image fitting. Each exposure uses a shared PSF width and "
        "astrometric shift while solving separate source amplitudes plus a local background plane; saturated, coarse-pixel, "
        "broad-PSF, highly correlated, poorly conditioned, low-SNR, or poorly fit exposures are rejected. Flux amplitudes are "
        "calibrated with the Community Pipeline MAGZERO or MAGZPT convention before band light curves are formed. A source is "
        "called supported only when residual-band variability recurs at consistent frequency in at least two independent bands. "
        "The TESS frequency-drift law is not extrapolated into the archival image epochs."
    )


def interpret_noirlab_image_forced_photometry_project(
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
    prepared = preparation.get("preparedSeries") or []
    target_usable = any(item.get("sourceRole") == "target-control" for item in prepared)
    counterpart_usable = any(item.get("sourceRole") == "catalog-counterpart" for item in prepared)
    target["scientificallyUsableControl"] = target_usable
    counterpart["scientificallyUsableControl"] = counterpart_usable

    failure_reasons = preparation.get("failureReasons") or {}
    successful_exposures = int(preparation.get("successfulForcedPhotometryExposures") or 0)
    candidate_exposures = int(preparation.get("candidateExposures") or 0)

    if candidate_exposures == 0:
        classification = "NOIRLAB_IMAGE_ARCHIVE_NO_SINGLE_EPOCH_CANDIDATES"
        origin = "UNRESOLVED_NOIRLAB_IMAGE_ARCHIVE_EMPTY_FOR_STRICT_SCREEN"
        next_test = NEXT_ARCHIVE_TEST
    elif not preparation.get("preparedSeries"):
        saturation_failures = sum(
            int(value)
            for key, value in failure_reasons.items()
            if "saturated" in str(key)
        )
        if saturation_failures >= max(1, candidate_exposures // 2):
            classification = "NOIRLAB_IMAGE_FORCED_PHOTOMETRY_SATURATION_LIMIT"
            origin = "UNRESOLVED_NOIRLAB_ARCHIVE_SATURATION_LIMIT"
        elif successful_exposures == 0:
            classification = "NOIRLAB_IMAGE_FORCED_PHOTOMETRY_NO_QUALIFYING_EXPOSURES"
            origin = "UNRESOLVED_NOIRLAB_IMAGE_FIT_QUALITY_LIMIT"
        else:
            classification = "NOIRLAB_IMAGE_FORCED_PHOTOMETRY_INSUFFICIENT_TIME_SERIES"
            origin = "UNRESOLVED_NOIRLAB_IMAGE_TIME_SERIES_LIMIT"
        next_test = NEXT_ARCHIVE_TEST
    elif counterpart_supported and not target_supported and target_usable:
        classification = "NOIRLAB_IMAGE_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_BY_IMAGE_LEVEL_FORCED_PHOTOMETRY"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_supported and not counterpart_supported and counterpart_usable:
        classification = "NOIRLAB_IMAGE_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_SUPPORTED_BY_IMAGE_LEVEL_FORCED_PHOTOMETRY"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif target_supported and counterpart_supported:
        classification = "NOIRLAB_IMAGE_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_AND_COUNTERPART_SUPPORTED_BY_IMAGE_LEVEL_FORCED_PHOTOMETRY"
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    elif counterpart_suggestive and not target_suggestive:
        classification = "NOIRLAB_IMAGE_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE_BY_IMAGE_LEVEL_FORCED_PHOTOMETRY"
        next_test = NEXT_ARCHIVE_TEST
    elif target_suggestive and not counterpart_suggestive:
        classification = "NOIRLAB_IMAGE_TARGET_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "TARGET_SUGGESTIVE_BY_IMAGE_LEVEL_FORCED_PHOTOMETRY"
        next_test = NEXT_ARCHIVE_TEST
    else:
        classification = "NOIRLAB_IMAGE_FORCED_PHOTOMETRY_SOURCE_ATTRIBUTION_UNRESOLVED"
        origin = "ARCHIVAL_IMAGE_LEVEL_SOURCE_ATTRIBUTION_UNRESOLVED"
        next_test = NEXT_ARCHIVE_TEST

    return {
        "version": "openstar.tess-noirlab-image-level-forced-photometry.v1",
        "archive": preparation.get("archive"),
        "siaURL": preparation.get("siaURL"),
        "sourcePair": preparation.get("sourcePair"),
        "pairCenter": preparation.get("pairCenter"),
        "pairSeparationArcsec": preparation.get("pairSeparationArcsec"),
        "catalogAssociationSeparationArcsec": preparation.get("catalogAssociationSeparationArcsec"),
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "tessDriftExtrapolated": False,
        "siaRows": preparation.get("siaRows"),
        "candidateExposures": preparation.get("candidateExposures"),
        "successfulForcedPhotometryExposures": preparation.get(
            "successfulForcedPhotometryExposures"
        ),
        "failureReasons": failure_reasons,
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
