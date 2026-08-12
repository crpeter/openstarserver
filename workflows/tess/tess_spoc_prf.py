from __future__ import annotations

import html
import math
import os
import re
import statistics
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .tess_multisource_residual import MIN_COMPONENT_SAMPLES, _prewhiten_cube_raw
from .tess_offset_variability import (
    DOMINANCE_POWER_RATIO,
    MIN_INDEPENDENT_SUPPORT,
    TOTAL_FREQUENCIES,
    FREQUENCIES_PER_WORK_UNIT,
    _best_offset_summary,
    _catalog_candidate,
    _candidate_label,
    _frequency_search,
    _nuisance_catalog_sources,
    _skycoord,
)
from .tess_prf_deblend import (
    MAX_DESIGN_CONDITION,
    MAX_NUISANCE_SOURCES,
    MAX_TEMPLATE_CORRELATION,
    _background_columns,
    _component_summary,
    _extract_calibrated_series,
    _result_record,
    _source_positions,
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
    _safe,
    _sector_candidates,
    _uniform_indices,
    _write_json,
)

# Official MAST/SPOC PRF archive.  NASA/TESS documentation states that the
# sector-1--3 calibration is separate and the sector-4+ model is rooted at
# start_s0004.  Blind C's frozen sectors are all later than sector 4.
MAST_PRF_ROOT = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles"
HTTP_TIMEOUT_SECONDS = 60
USER_AGENT = "OpenStar/20.18 official-SPOC-PRF-forward-modeling"

# Static forward-model quality guards.  These do not decide variability;
# they only decide whether a sector is suitable for source separation.
MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE = 0.25
MAX_OFFICIAL_PRF_DESIGN_CONDITION = MAX_DESIGN_CONDITION
MAX_OFFICIAL_PRF_TEMPLATE_CORRELATION = MAX_TEMPLATE_CORRELATION
SHARED_ASTROMETRIC_SHIFT_GRID = (-0.20, 0.0, 0.20)

# The public PRF FITS products are supersampled.  Prefer explicit metadata,
# then the common SPOC sampling factors.  A native-pixel product still works
# with factor 1.
OVERSAMPLE_CANDIDATES = (9, 10, 5, 4, 3, 2, 1)
MIN_NATIVE_PRF_WIDTH = 5.0
MAX_NATIVE_PRF_WIDTH = 31.0


def _drift_corrected_times(
    absolute_times: np.ndarray,
    *,
    time_reference_days: float,
    fractional_frequency_drift_per_day: float,
) -> np.ndarray:
    """
    Map the v20.9 linearly drifting-frequency model onto a constant-frequency
    time coordinate.

    v20.9 defines the instantaneous residual frequency as

        f(t) = f_ref * (1 + q * (t - t_ref))

    where q is fractionalFrequencyDriftPerDay. Integrating phase relative to
    f_ref gives the drift-corrected coordinate

        tau = dt + 0.5 * q * dt^2

    with dt = t - t_ref.

    Any additive constant is irrelevant to Lomb-Scargle; per-sector and
    combined datasets are shifted to start at zero downstream.
    """
    times = np.asarray(absolute_times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("Drift correction requires a one-dimensional time array.")
    if len(times) == 0:
        raise ValueError("Drift correction requires at least one cadence.")
    if not np.all(np.isfinite(times)):
        raise ValueError("Drift correction received non-finite cadence times.")

    t_ref = float(time_reference_days)
    q = float(fractional_frequency_drift_per_day)
    if not math.isfinite(t_ref) or not math.isfinite(q):
        raise ValueError("Drift correction requires finite t_ref and q.")

    dt = times - t_ref
    warped = dt + 0.5 * q * np.square(dt)

    if not np.all(np.isfinite(warped)):
        raise ValueError("Drift correction produced non-finite times.")
    if len(warped) > 1 and np.any(np.diff(warped) <= 0):
        raise ValueError(
            "Drift correction produced a non-monotonic time axis; "
            "the fitted v20.9 drift is outside the valid transform range."
        )

    return warped


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.read()


def _model_start_sector(sector: int) -> int:
    return 1 if int(sector) <= 3 else 4


def _prf_directory_url(*, sector: int, camera: int, ccd: int) -> str:
    start_sector = _model_start_sector(int(sector))
    return (
        f"{MAST_PRF_ROOT}/start_s{start_sector:04d}/"
        f"cam{int(camera)}_ccd{int(ccd)}/"
    )


def _list_official_prf_grid(*, sector: int, camera: int, ccd: int) -> list[dict[str, Any]]:
    directory = _prf_directory_url(sector=sector, camera=camera, ccd=ccd)
    body = _http_get(directory).decode("utf-8", errors="replace")
    pattern = re.compile(
        rf'href=["\']([^"\']*prf-{int(camera)}-{int(ccd)}-row(\d+)-col(\d+)\.fits)["\']',
        re.IGNORECASE,
    )
    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for href, row_text, col_text in pattern.findall(body):
        row = int(row_text)
        col = int(col_text)
        key = (row, col)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "row": row,
                "column": col,
                "url": urllib.parse.urljoin(directory, html.unescape(href)),
            }
        )
    if not entries:
        raise RuntimeError(
            "MAST official SPOC PRF directory returned no row/column FITS models: "
            f"{directory}"
        )
    return sorted(entries, key=lambda item: (int(item["row"]), int(item["column"])))


def _download_cached(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    payload = _http_get(url)
    if len(payload) < 2880:
        raise RuntimeError(f"Official PRF download was unexpectedly small ({len(payload)} bytes): {url}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(payload)
    os.replace(tmp, destination)
    return destination


def _fits_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from astropy.io import fits
    except Exception as exc:  # pragma: no cover - exercised in user's astronomy environment
        raise RuntimeError("v20.18 requires astropy.io.fits to read the official SPOC PRF files.") from exc

    with fits.open(path, memmap=False) as hdul:
        primary = dict(hdul[0].header) if len(hdul) else {}
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            image = np.asarray(data, dtype=np.float64).squeeze()
            if image.ndim != 2 or min(image.shape) < 5:
                continue
            header = dict(primary)
            header.update(dict(hdu.header))
            finite = np.isfinite(image)
            if not np.any(finite):
                continue
            image = np.where(finite, image, 0.0)
            # Tiny negative calibration lobes can occur in fitted PRFs; the
            # source-flux forward model uses the non-negative response only.
            image = np.maximum(image, 0.0)
            total = float(np.sum(image))
            if not math.isfinite(total) or total <= 1e-12:
                continue
            return image / total, header
    raise RuntimeError(f"No usable 2-D PRF image found in {path}.")


def _infer_oversample(image: np.ndarray, header: dict[str, Any]) -> int:
    for key in (
        "OVERSAMP",
        "OVERSAMPLE",
        "SUBSAMP",
        "SUBSAMPLE",
        "SAMPFAC",
        "SAMPLEFAC",
    ):
        value = _float(header.get(key))
        if value is not None:
            rounded = int(round(value))
            if 1 <= rounded <= 32:
                return rounded

    rows, cols = image.shape
    for candidate in OVERSAMPLE_CANDIDATES:
        native_rows = rows / candidate
        native_cols = cols / candidate
        if (
            MIN_NATIVE_PRF_WIDTH <= native_rows <= MAX_NATIVE_PRF_WIDTH
            and MIN_NATIVE_PRF_WIDTH <= native_cols <= MAX_NATIVE_PRF_WIDTH
            and abs(native_rows - round(native_rows)) < 1e-6
            and abs(native_cols - round(native_cols)) < 1e-6
        ):
            return int(candidate)
    return 1


def _prf_center(image: np.ndarray, header: dict[str, Any]) -> tuple[float, float]:
    rows, cols = image.shape
    x = _float(header.get("CRPIX1"))
    y = _float(header.get("CRPIX2"))
    if x is not None and y is not None:
        # FITS CRPIX is 1-indexed.
        x0 = float(x) - 1.0
        y0 = float(y) - 1.0
        if -0.5 <= x0 <= cols - 0.5 and -0.5 <= y0 <= rows - 0.5:
            return x0, y0
    return (cols - 1.0) / 2.0, (rows - 1.0) / 2.0


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rows, cols = image.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < cols) & (y1 < rows)
    out = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)
    if not np.any(inside):
        return out
    xx = x[inside]
    yy = y[inside]
    xa = x0[inside]
    xb = x1[inside]
    ya = y0[inside]
    yb = y1[inside]
    fx = xx - xa
    fy = yy - ya
    out[inside] = (
        image[ya, xa] * (1.0 - fx) * (1.0 - fy)
        + image[ya, xb] * fx * (1.0 - fy)
        + image[yb, xa] * (1.0 - fx) * fy
        + image[yb, xb] * fx * fy
    )
    return out


def _render_prf_template(
    *,
    image: np.ndarray,
    header: dict[str, Any],
    source_x: float,
    source_y: float,
    rows: int,
    cols: int,
    valid_pixels: np.ndarray,
) -> np.ndarray:
    oversample = _infer_oversample(image, header)
    center_x, center_y = _prf_center(image, header)
    # Integrate each TPF pixel using a modest supersampled quadrature.  If the
    # archive image is already native-pixel sampled this naturally reduces to
    # one sample per pixel.
    quadrature = max(1, min(int(oversample), 9))
    offsets = (np.arange(quadrature, dtype=np.float64) + 0.5) / quadrature - 0.5
    yy, xx = np.mgrid[0:rows, 0:cols]
    template = np.zeros((rows, cols), dtype=np.float64)
    for dy in offsets:
        for dx in offsets:
            sample_x = center_x + ((xx.astype(np.float64) + dx) - float(source_x)) * oversample
            sample_y = center_y + ((yy.astype(np.float64) + dy) - float(source_y)) * oversample
            template += _bilinear_sample(image, sample_x, sample_y)
    template /= float(quadrature * quadrature)
    template *= np.asarray(valid_pixels, dtype=np.float64)
    total = float(np.sum(template))
    if not math.isfinite(total) or total <= 1e-12:
        raise RuntimeError("Official SPOC PRF rendered zero flux into the TPF stamp.")
    return (template / total).reshape(-1)


def _bracket(values: list[int], target: float) -> tuple[int, int]:
    ordered = sorted(set(int(v) for v in values))
    lower = [v for v in ordered if v <= target]
    upper = [v for v in ordered if v >= target]
    lo = max(lower) if lower else ordered[0]
    hi = min(upper) if upper else ordered[-1]
    return int(lo), int(hi)


def _grid_neighbors(entries: list[dict[str, Any]], row: float, col: float) -> list[tuple[dict[str, Any], float]]:
    rows = [int(item["row"]) for item in entries]
    cols = [int(item["column"]) for item in entries]
    r0, r1 = _bracket(rows, row)
    c0, c1 = _bracket(cols, col)
    by_key = {(int(item["row"]), int(item["column"])): item for item in entries}

    tr = 0.0 if r1 == r0 else min(max((float(row) - r0) / (r1 - r0), 0.0), 1.0)
    tc = 0.0 if c1 == c0 else min(max((float(col) - c0) / (c1 - c0), 0.0), 1.0)
    candidates = [
        ((r0, c0), (1.0 - tr) * (1.0 - tc)),
        ((r0, c1), (1.0 - tr) * tc),
        ((r1, c0), tr * (1.0 - tc)),
        ((r1, c1), tr * tc),
    ]
    accumulated: dict[tuple[int, int], float] = {}
    for key, weight in candidates:
        if key in by_key:
            accumulated[key] = accumulated.get(key, 0.0) + float(weight)
    if not accumulated:
        # Irregular archive grids: fall back to the four nearest detector-grid
        # models with inverse-distance weights.
        nearest = sorted(
            entries,
            key=lambda item: math.hypot(float(item["row"]) - row, float(item["column"]) - col),
        )[:4]
        weights = []
        for item in nearest:
            distance = math.hypot(float(item["row"]) - row, float(item["column"]) - col)
            weights.append(1.0 / max(distance, 1.0))
        norm = sum(weights)
        return [(item, weight / norm) for item, weight in zip(nearest, weights)]
    norm = sum(accumulated.values())
    if norm <= 0:
        key = min(
            by_key,
            key=lambda value: math.hypot(float(value[0]) - row, float(value[1]) - col),
        )
        return [(by_key[key], 1.0)]
    return [(by_key[key], weight / norm) for key, weight in accumulated.items() if weight > 0]


def _official_prf_at_detector_position(
    *,
    sector: int,
    camera: int,
    ccd: int,
    detector_row: float,
    detector_col: float,
    archive_cache: Path,
    grid_entries: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    pieces = _grid_neighbors(grid_entries, detector_row, detector_col)
    weighted: np.ndarray | None = None
    representative_header: dict[str, Any] = {}
    used: list[dict[str, Any]] = []
    for entry, weight in pieces:
        filename = Path(urllib.parse.urlparse(str(entry["url"])).path).name
        local_path = _download_cached(str(entry["url"]), archive_cache / filename)
        image, header = _fits_image(local_path)
        if weighted is None:
            weighted = np.zeros_like(image, dtype=np.float64)
            representative_header = header
        if image.shape != weighted.shape:
            raise RuntimeError("Official SPOC PRF grid files have inconsistent image dimensions.")
        weighted += float(weight) * image
        used.append(
            {
                "row": int(entry["row"]),
                "column": int(entry["column"]),
                "weight": float(weight),
                "url": str(entry["url"]),
                "localPath": str(local_path.resolve()),
            }
        )
    if weighted is None:
        raise RuntimeError("No official SPOC PRF grid model could be loaded.")
    total = float(np.sum(weighted))
    if total <= 1e-12:
        raise RuntimeError("Interpolated official SPOC PRF has zero flux.")
    return weighted / total, representative_header, used


def _tpf_detector_geometry(tpf: Any) -> tuple[int, int, float, float]:
    header = tpf.hdu[0].header
    camera = _int(getattr(tpf, "camera", None)) or _int(header.get("CAMERA"))
    ccd = _int(getattr(tpf, "ccd", None)) or _int(header.get("CCD"))
    column = _float(getattr(tpf, "column", None))
    row = _float(getattr(tpf, "row", None))
    # Lightkurve exposes the physical lower-left CCD pixel as column/row.  The
    # header fallbacks cover raw TessTargetPixelFile implementations.
    if column is None:
        column = _float(tpf.hdu[1].header.get("1CRV5P"))
    if row is None:
        row = _float(tpf.hdu[1].header.get("2CRV5P"))
    if camera is None or ccd is None or column is None or row is None:
        raise RuntimeError("TPF is missing camera/CCD or physical row/column metadata required by the SPOC PRF.")
    return int(camera), int(ccd), float(column), float(row)


def _fit_static_image(design: np.ndarray, image: np.ndarray, source_count: int) -> tuple[float, np.ndarray, float]:
    y = np.asarray(image, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    if int(np.count_nonzero(valid)) < max(12, design.shape[1] + 2):
        return float("inf"), np.zeros(design.shape[1]), float("-inf")
    a = design[valid]
    b = y[valid]
    coefficients, *_ = np.linalg.lstsq(a, b, rcond=None)
    model = a @ coefficients
    residual = b - model
    source_coefficients = coefficients[:source_count]
    negative_penalty = float(np.sum(np.square(np.minimum(source_coefficients, 0.0))))
    sse = float(np.sum(residual * residual)) + negative_penalty * max(float(np.var(b)), 1.0) * 50.0
    centered = b - float(np.mean(b))
    sst = float(np.sum(centered * centered))
    explained = 1.0 - float(np.sum(residual * residual)) / sst if sst > 1e-12 else 0.0
    return sse, coefficients, explained


def _calibrate_official_spoc_design(
    *,
    sector: int,
    corrected_cube: np.ndarray,
    valid_pixels: np.ndarray,
    tpf: Any,
    target_sky: Any,
    candidate_sky: Any,
    nuisance_sources: list[dict[str, Any]],
    cache_root: Path,
) -> dict[str, Any]:
    rows, cols = valid_pixels.shape
    sources = _source_positions(
        tpf=tpf,
        target_sky=target_sky,
        candidate_sky=candidate_sky,
        nuisance_sources=nuisance_sources,
        rows=rows,
        cols=cols,
    )
    camera, ccd, tpf_column, tpf_row = _tpf_detector_geometry(tpf)
    start_sector = _model_start_sector(int(sector))
    grid_entries = _list_official_prf_grid(sector=sector, camera=camera, ccd=ccd)
    archive_cache = cache_root / f"start_s{start_sector:04d}" / f"cam{camera}_ccd{ccd}"

    # Load/interpolate the official PRF shape once per catalog source.  The
    # detector-position interpolation is independent of the small shared WCS
    # shift optimized below.
    source_models: list[dict[str, Any]] = []
    for source in sources:
        center = source["pixelCenter"]
        detector_col = float(tpf_column) + float(center["x"])
        detector_row = float(tpf_row) + float(center["y"])
        image, header, model_files = _official_prf_at_detector_position(
            sector=sector,
            camera=camera,
            ccd=ccd,
            detector_row=detector_row,
            detector_col=detector_col,
            archive_cache=archive_cache,
            grid_entries=grid_entries,
        )
        source_models.append(
            {
                "componentID": source["componentID"],
                "pixelCenter": dict(center),
                "detectorRow": detector_row,
                "detectorColumn": detector_col,
                "image": image,
                "header": header,
                "modelFiles": model_files,
                "oversample": _infer_oversample(image, header),
            }
        )

    median_image = np.nanmedian(np.asarray(corrected_cube, dtype=np.float64), axis=0)
    best: dict[str, Any] | None = None
    for dx in SHARED_ASTROMETRIC_SHIFT_GRID:
        for dy in SHARED_ASTROMETRIC_SHIFT_GRID:
            columns: list[np.ndarray] = []
            modeled: list[dict[str, Any]] = []
            for source in source_models:
                center = source["pixelCenter"]
                template = _render_prf_template(
                    image=source["image"],
                    header=source["header"],
                    source_x=float(center["x"]) + float(dx),
                    source_y=float(center["y"]) + float(dy),
                    rows=rows,
                    cols=cols,
                    valid_pixels=valid_pixels,
                )
                columns.append(template)
                modeled.append(
                    {
                        "componentID": source["componentID"],
                        "catalogPixelCenter": dict(center),
                        "forwardModelPixelCenter": {
                            "x": float(center["x"]) + float(dx),
                            "y": float(center["y"]) + float(dy),
                        },
                        "detectorRow": source["detectorRow"],
                        "detectorColumn": source["detectorColumn"],
                        "oversample": source["oversample"],
                        "officialPRFModelFiles": source["modelFiles"],
                    }
                )
            columns.extend(_background_columns(rows, cols, valid_pixels))
            design = np.column_stack(columns)
            score, coefficients, explained = _fit_static_image(design, median_image, len(source_models))
            if best is None or score < best["score"]:
                best = {
                    "score": float(score),
                    "coefficients": coefficients,
                    "explainedVariance": float(explained),
                    "dx": float(dx),
                    "dy": float(dy),
                    "design": design,
                    "sources": modeled,
                }
    if best is None:
        raise RuntimeError("Unable to fit the official SPOC PRF forward model to this TPF.")

    design = np.asarray(best.pop("design"), dtype=np.float64)
    coefficients = np.asarray(best.pop("coefficients"), dtype=np.float64)
    component_ids = [item["componentID"] for item in best["sources"]]
    target_index = component_ids.index("target-control")
    counterpart_index = component_ids.index("catalog-counterpart")
    target_template = design[:, target_index]
    counterpart_template = design[:, counterpart_index]
    denom = float(np.linalg.norm(target_template) * np.linalg.norm(counterpart_template))
    correlation = float(np.dot(target_template, counterpart_template) / denom) if denom > 1e-12 else 1.0
    condition = float(np.linalg.cond(design))

    best.update(
        {
            "backend": "official-spoc-prf-forward-model-v1",
            "officialPRFRoot": MAST_PRF_ROOT,
            "officialPRFStartSector": int(start_sector),
            "camera": int(camera),
            "ccd": int(ccd),
            "tpfPhysicalColumn": float(tpf_column),
            "tpfPhysicalRow": float(tpf_row),
            "designConditionNumber": condition,
            "targetCounterpartTemplateCorrelation": correlation,
            "medianSourceCoefficients": {
                component_ids[index]: float(coefficients[index])
                for index in range(len(component_ids))
            },
            "design": design,
            "componentIDs": component_ids,
        }
    )

    if not math.isfinite(condition) or condition > MAX_OFFICIAL_PRF_DESIGN_CONDITION:
        raise RuntimeError(f"Official SPOC PRF forward design is ill-conditioned ({condition}).")
    if abs(correlation) >= MAX_OFFICIAL_PRF_TEMPLATE_CORRELATION:
        raise RuntimeError(
            "Official SPOC target/counterpart templates are too degenerate "
            f"(correlation={correlation})."
        )
    if float(best.get("explainedVariance") or 0.0) < MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE:
        raise RuntimeError(
            "Official SPOC PRF forward model does not explain enough of the sector median image "
            f"(R2={best.get('explainedVariance')})."
        )
    return best


def build_official_spoc_prf_project(
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
    frequency_localized_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.18 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )
    if frequency_localized_summary.get("recommendedNextTest") != "OFFICIAL_SPOC_PRF_FORWARD_MODELING":
        raise RuntimeError("v20.18 requires v20.17 to recommend official SPOC PRF forward modeling.")

    best_offset = _best_offset_summary(multisource_summary)
    reference_frequency = _float(best_offset.get("combinedFrequency"))
    if reference_frequency is None or reference_frequency <= 0:
        reference_frequency = _float(nonstationary_summary.get("preferredFrequencyAtReference"))
    q = _float(nonstationary_summary.get("fractionalFrequencyDriftPerDay"))
    time_reference = _float(nonstationary_summary.get("timeReferenceDays"))
    if reference_frequency is None or reference_frequency <= 0 or q is None or time_reference is None:
        raise RuntimeError("v20.18 requires the completed v20.9 drift model and v20.12 offset frequency.")

    target_meta = ((identity.get("tic") or {}).get("metadata") or {})
    target_ra = _float(target_meta.get("raDeg"))
    target_dec = _float(target_meta.get("decDeg"))
    if target_ra is None or target_dec is None:
        raise RuntimeError("v20.18 requires TIC RA/Dec from the identity stage.")

    candidate = _catalog_candidate(offset_source_identification)
    candidate_ra = float(candidate["raDeg"])
    candidate_dec = float(candidate["decDeg"])
    candidate_ids = candidate.get("catalogIDs") or {}
    nuisance_sources = _nuisance_catalog_sources(
        offset_source_identification=offset_source_identification,
        best_candidate=candidate,
    )[:MAX_NUISANCE_SOURCES]

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
        raise RuntimeError("v20.18 found no frozen sectors for official SPOC PRF modeling.")

    physical_frequency = 1.0 / float(physical_period_days)
    search = _frequency_search(float(reference_frequency))
    target_sky = _skycoord(float(target_ra), float(target_dec))
    candidate_sky = _skycoord(candidate_ra, candidate_dec)
    root = Path(output_dir) / "official-spoc-prf-forward-modeling"
    root.mkdir(parents=True, exist_ok=True)
    source_base_id = str(source_dataset_entry.get("id") or f"tic-{target_tic_id}")

    dataset_entries: list[dict[str, Any]] = []
    prepared_series: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    combined: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "target-control": [],
        "catalog-counterpart": [],
    }

    for sector_index, (sector, role) in enumerate(sectors, start=1):
        print(
            f"   Sector {sector} ({sector_index}/{len(sectors)}): official SPOC PRF forward fit + target/counterpart extraction",
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
            calibration = _calibrate_official_spoc_design(
                sector=int(sector),
                corrected_cube=corrected,
                valid_pixels=valid_pixels,
                tpf=tpf,
                target_sky=target_sky,
                candidate_sky=candidate_sky,
                nuisance_sources=nuisance_sources,
                cache_root=root / "official-prf-cache",
            )
            component_series = _extract_calibrated_series(
                residual_cube=residual_cube,
                calibration=calibration,
            )
            diagnostic = {key: value for key, value in calibration.items() if key not in {"design"}}
            diagnostic.update(
                {
                    "sector": int(sector),
                    "role": role,
                    "sourceType": source.get("sourceType"),
                    "author": source.get("author"),
                    "cadenceSeconds": source.get("cadenceSeconds"),
                }
            )
            diagnostics.append(diagnostic)

            warped = _drift_corrected_times(
                absolute_times,
                time_reference_days=float(time_reference),
                fractional_frequency_drift_per_day=float(q),
            )
            local_times = warped - float(np.min(warped))
            for component_id in ("target-control", "catalog-counterpart"):
                values = component_series[component_id]
                dataset_id = f"{source_base_id}-spoc-prf-{component_id}-sector-{sector}-v1"
                target_name = (
                    f"{source_dataset_entry.get('targetName') or source_base_id} "
                    f"official SPOC PRF {component_id} sector {sector}"
                )
                output_path = root / f"{_safe(dataset_id)}.json"
                dataset = {
                    "id": dataset_id,
                    "targetName": target_name,
                    "times": np.asarray(local_times, dtype=np.float32).tolist(),
                    "flux": np.asarray(values, dtype=np.float32).tolist(),
                    "frequencySearch": search,
                    "reference": {},
                    "science": {
                        "role": "official-spoc-prf-forward-modeling",
                        "componentID": component_id,
                        "sector": int(sector),
                        "referenceFrequency": float(reference_frequency),
                        "fractionalFrequencyDriftPerDay": float(q),
                        "deblendBackend": "official-spoc-prf-forward-model-v1",
                        "catalogCounterpart": {
                            "ticID": _int(candidate_ids.get("ticID")),
                            "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
                            "raDeg": candidate_ra,
                            "decDeg": candidate_dec,
                        },
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
                dataset_entries.append(
                    {"id": dataset_id, "path": str(output_path.resolve()), "targetName": target_name}
                )
                prepared_series.append(
                    {
                        "datasetID": dataset_id,
                        "datasetPath": str(output_path.resolve()),
                        "componentID": component_id,
                        "sector": int(sector),
                        "role": role,
                        "combined": False,
                        "baselineDays": float(np.max(absolute_times) - np.min(absolute_times)),
                    }
                )
                combined[component_id].append(
                    (np.asarray(warped, dtype=np.float64), np.asarray(values, dtype=np.float64))
                )
            print(
                f"      official PRF start_s{calibration.get('officialPRFStartSector'):04d} "
                f"cam{calibration.get('camera')} ccd{calibration.get('ccd')} | "
                f"R2={calibration.get('explainedVariance'):.3f}, "
                f"templateCorr={calibration.get('targetCounterpartTemplateCorrelation'):.3f}; "
                f"extracted target-control + {_candidate_label(candidate)}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"sector": int(sector), "error": f"{type(exc).__name__}: {exc}"})
            print(f"      unavailable: {type(exc).__name__}: {exc}", flush=True)

    for component_id in ("target-control", "catalog-counterpart"):
        pieces = combined.get(component_id) or []
        if len(pieces) < 2:
            continue
        all_times = np.concatenate([item[0] for item in pieces])
        all_flux = np.concatenate([item[1] for item in pieces])
        order = np.argsort(all_times)
        all_times = all_times[order]
        all_flux = all_flux[order]
        local_times = all_times - float(np.min(all_times))
        dataset_id = f"{source_base_id}-spoc-prf-{component_id}-combined-v1"
        target_name = (
            f"{source_dataset_entry.get('targetName') or source_base_id} "
            f"official SPOC PRF {component_id} combined"
        )
        output_path = root / f"{_safe(dataset_id)}.json"
        dataset = {
            "id": dataset_id,
            "targetName": target_name,
            "times": np.asarray(local_times, dtype=np.float32).tolist(),
            "flux": np.asarray(all_flux, dtype=np.float32).tolist(),
            "frequencySearch": search,
            "reference": {},
            "science": {
                "role": "official-spoc-prf-forward-modeling-combined",
                "componentID": component_id,
                "referenceFrequency": float(reference_frequency),
                "fractionalFrequencyDriftPerDay": float(q),
                "deblendBackend": "official-spoc-prf-forward-model-v1",
                "catalogCounterpart": {
                    "ticID": _int(candidate_ids.get("ticID")),
                    "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
                    "raDeg": candidate_ra,
                    "decDeg": candidate_dec,
                },
            },
            "source": {
                "mission": "TESS",
                "distributedSamples": int(len(local_times)),
                "baselineDays": float(np.max(all_times) - np.min(all_times)),
                "timeReferenceDays": float(time_reference),
                "combinedSectors": True,
            },
        }
        _write_json(output_path, dataset)
        dataset_entries.append(
            {"id": dataset_id, "path": str(output_path.resolve()), "targetName": target_name}
        )
        prepared_series.append(
            {
                "datasetID": dataset_id,
                "datasetPath": str(output_path.resolve()),
                "componentID": component_id,
                "sector": None,
                "role": "combined",
                "combined": True,
                "baselineDays": float(np.max(all_times) - np.min(all_times)),
            }
        )

    if not dataset_entries:
        detail = "; ".join(item["error"] for item in errors[:3])
        raise RuntimeError(
            "v20.18 could not prepare any official SPOC PRF source datasets. "
            + (f"First errors: {detail}" if detail else "")
        )

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "official-spoc-prf-forward-modeling-v1"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — official SPOC PRF forward modeling",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "official-spoc-prf-forward-modeling",
            "workerSemantics": (
                "Each dataset is an official-SPOC-PRF-forward-modeled, established-family-prewhitened, "
                "v20.9 drift-corrected source-component light curve. Workers execute ordinary Lomb-Scargle only."
            ),
            "referenceFrequency": float(reference_frequency),
            "fractionalFrequencyDriftPerDay": float(q),
            "catalogCounterpart": {
                "ticID": _int(candidate_ids.get("ticID")),
                "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
                "raDeg": candidate_ra,
                "decDeg": candidate_dec,
            },
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)
    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": True,
        "version": "openstar.tess-official-spoc-prf-forward-modeling-preparation.v1",
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-official-spoc-prf-forward-modeled-drift-corrected-source-series",
        "deblendBackend": "official-spoc-prf-forward-model-v1",
        "officialPRFRoot": MAST_PRF_ROOT,
        "targetTIC": int(target_tic_id),
        "catalogCounterpart": {
            "ticID": _int(candidate_ids.get("ticID")),
            "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
            "raDeg": candidate_ra,
            "decDeg": candidate_dec,
            "catalogSeparationArcsec": _float(candidate.get("separationArcsec")),
        },
        "bestOffsetComponentID": multisource_summary.get("bestOffsetComponentID"),
        "referenceFrequency": float(reference_frequency),
        "referencePeriodDays": float(1.0 / reference_frequency),
        "fractionalFrequencyDriftPerDay": float(q),
        "timeReferenceDays": float(time_reference),
        "frequencySearch": search,
        "preparedSeries": prepared_series,
        "calibrationDiagnostics": diagnostics,
        "nuisanceCatalogSources": nuisance_sources,
        "errors": errors,
        "workUnitsPerDataset": work_units_per_dataset,
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "interpretationGuard": (
            "v20.18 uses the official public SPOC TESS PRF FITS calibration appropriate to each camera/CCD, "
            "interpolates the detector-position PRF grid, forward-fits fixed catalog source positions, and only "
            "then exposes separated residual light curves to the generic Lomb-Scargle worker. Failed PRF fit, "
            "conditioning, or template-separation guards exclude a sector rather than relaxing source attribution."
        ),
    }


def interpret_official_spoc_prf_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedSeries") or []
    }
    results: list[dict[str, Any]] = []
    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id)
        if meta is None:
            continue
        results.append(_result_record(dataset, meta, preparation))

    target = _component_summary("target-control", results)
    counterpart = _component_summary("catalog-counterpart", results)
    target_support = int(target.get("independentSupportCount") or 0)
    counterpart_support = int(counterpart.get("independentSupportCount") or 0)
    target_power = float(target.get("combinedPower") or 0.0)
    counterpart_power = float(counterpart.get("combinedPower") or 0.0)
    target_present = target_support >= MIN_INDEPENDENT_SUPPORT and bool(target.get("combinedAccepted"))
    counterpart_present = counterpart_support >= MIN_INDEPENDENT_SUPPORT and bool(counterpart.get("combinedAccepted"))
    counterpart_dominant = bool(
        counterpart_present
        and (
            not target_present
            or target_power <= 0
            or counterpart_power >= target_power * DOMINANCE_POWER_RATIO
        )
    )

    if counterpart_dominant:
        classification = "SPOC_PRF_COUNTERPART_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_BY_OFFICIAL_SPOC_PRF"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif counterpart_present and target_present:
        classification = "SPOC_PRF_TARGET_AND_COUNTERPART_SUPPORTED"
        origin = "TARGET_AND_COUNTERPART_SUPPORTED_BY_OFFICIAL_SPOC_PRF"
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    elif target_present and not counterpart_present:
        classification = "SPOC_PRF_TARGET_SUPPORTED"
        origin = "TARGET_SUPPORTED_BY_OFFICIAL_SPOC_PRF"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif counterpart_support >= 2 and counterpart_power > target_power:
        classification = "SPOC_PRF_COUNTERPART_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE_BY_OFFICIAL_SPOC_PRF"
        next_test = "EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION"
    elif target_support >= 2 and target_power >= counterpart_power:
        classification = "SPOC_PRF_TARGET_SUGGESTIVE"
        origin = "TARGET_SUGGESTIVE_BY_OFFICIAL_SPOC_PRF"
        next_test = "EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION"
    else:
        classification = "SPOC_PRF_SOURCE_ATTRIBUTION_UNRESOLVED"
        origin = "TESS_SPATIAL_ATTRIBUTION_LIMIT_AFTER_OFFICIAL_SPOC_PRF"
        next_test = "EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION"

    return {
        "version": "openstar.tess-official-spoc-prf-forward-modeling.v1",
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "deblendBackend": preparation.get("deblendBackend"),
        "officialPRFRoot": preparation.get("officialPRFRoot"),
        "catalogCounterpart": preparation.get("catalogCounterpart"),
        "bestOffsetComponentID": preparation.get("bestOffsetComponentID"),
        "referenceFrequency": preparation.get("referenceFrequency"),
        "referencePeriodDays": preparation.get("referencePeriodDays"),
        "fractionalFrequencyDriftPerDay": preparation.get("fractionalFrequencyDriftPerDay"),
        "componentResults": results,
        "targetControl": target,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "officialSPOCPRFCounterpartSupported": classification == "SPOC_PRF_COUNTERPART_SUPPORTED",
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "preparationErrors": preparation.get("errors") or [],
        "calibrationDiagnostics": preparation.get("calibrationDiagnostics") or [],
        "interpretationGuard": (
            "v20.18 uses the official public SPOC TESS PRF calibration rather than an empirical or Gaussian "
            "source template. Secure source attribution still requires recurring independent-sector residual-frequency "
            "support and an accepted combined result. Failure to resolve the source after this stage is reported as "
            "a TESS spatial-attribution limit, not converted into evidence for either source."
        ),
    }
