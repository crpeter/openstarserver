from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_multisource_residual import MIN_COMPONENT_SAMPLES, _prewhiten_cube_raw
from .tess_offset_variability import (
    DOMINANCE_POWER_RATIO,
    MIN_CANDIDATE_POWER,
    MIN_INDEPENDENT_SUPPORT,
    MIN_OBSERVED_CYCLES,
    MIN_PEAK_PROMINENCE,
    REFERENCE_FREQUENCY_TOLERANCE_FRACTION,
    TOTAL_FREQUENCIES,
    FREQUENCIES_PER_WORK_UNIT,
    _best_offset_summary,
    _boundary_hit,
    _catalog_candidate,
    _candidate_label,
    _frequency_search,
    _nuisance_catalog_sources,
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
    _safe,
    _sector_candidates,
    _time_warp,
    _uniform_indices,
    _write_json,
)

# The calibrated ePRF model is inferred independently in every TESS sector
# from that sector's actual background-corrected pixel stamp.  This avoids
# the fixed-Gaussian assumption used by v20.14 while preserving a deterministic
# and dependency-light workflow.  The fit uses catalog positions as fixed
# source centers plus one shared astrometric offset and a shared elliptical
# core+wing response shape.
MAX_NUISANCE_SOURCES = 3
MIN_TEMPLATE_SEPARATION_PIXELS = 0.45
MAX_TEMPLATE_CORRELATION = 0.985
MAX_DESIGN_CONDITION = 5.0e5
MIN_CALIBRATION_EXPLAINED_VARIANCE = 0.35

CORE_SIGMA_GRID = (0.55, 0.70, 0.85, 1.00, 1.15)
AXIS_RATIO_GRID = (0.75, 0.90, 1.00, 1.10, 1.30)
THETA_GRID = (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)
WING_FRACTION_GRID = (0.0, 0.08, 0.16)
WING_SCALE = 2.15
ASTROMETRIC_SHIFT_GRID = (-0.30, -0.15, 0.0, 0.15, 0.30)


def _source_positions(
    *,
    tpf: Any,
    target_sky: Any,
    candidate_sky: Any,
    nuisance_sources: list[dict[str, Any]],
    rows: int,
    cols: int,
) -> list[dict[str, Any]]:
    source_defs: list[tuple[str, Any]] = [
        ("target-control", target_sky),
        ("catalog-counterpart", candidate_sky),
    ]
    for index, source in enumerate(nuisance_sources[:MAX_NUISANCE_SOURCES], start=1):
        ra = _float(source.get("raDeg"))
        dec = _float(source.get("decDeg"))
        if ra is None or dec is None:
            continue
        source_defs.append((f"nuisance-{index}", _skycoord(ra, dec)))

    used: list[dict[str, Any]] = []
    centers: list[tuple[float, float]] = []
    for component_id, coordinate in source_defs:
        try:
            x, y = tpf.wcs.world_to_pixel(coordinate)
            x = float(x)
            y = float(y)
        except Exception:
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if x < -2.5 or x > cols - 1 + 2.5 or y < -2.5 or y > rows - 1 + 2.5:
            continue
        if component_id.startswith("nuisance-") and any(
            math.hypot(x - px, y - py) < MIN_TEMPLATE_SEPARATION_PIXELS
            for px, py in centers
        ):
            continue
        centers.append((x, y))
        used.append({"componentID": component_id, "pixelCenter": {"x": x, "y": y}})

    ids = {item["componentID"] for item in used}
    if not {"target-control", "catalog-counterpart"}.issubset(ids):
        raise RuntimeError("Target and catalog counterpart are not both usable in this TPF.")
    return used


def _eprf_template(
    *,
    rows: int,
    cols: int,
    x: float,
    y: float,
    valid_pixels: np.ndarray,
    sigma_x: float,
    axis_ratio: float,
    theta: float,
    wing_fraction: float,
) -> np.ndarray:
    sigma_y = max(0.35, float(sigma_x) * float(axis_ratio))
    yy, xx = np.mgrid[0:rows, 0:cols]
    dx = xx.astype(np.float64) - float(x)
    dy = yy.astype(np.float64) - float(y)
    ct = math.cos(float(theta))
    st = math.sin(float(theta))
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    core = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))

    wing_sigma_x = float(sigma_x) * WING_SCALE
    wing_sigma_y = float(sigma_y) * WING_SCALE
    wing = np.exp(-0.5 * ((xr / wing_sigma_x) ** 2 + (yr / wing_sigma_y) ** 2))
    template = (1.0 - float(wing_fraction)) * core + float(wing_fraction) * wing
    template *= np.asarray(valid_pixels, dtype=np.float64)
    total = float(np.sum(template))
    if not math.isfinite(total) or total <= 1e-12:
        return np.zeros(rows * cols, dtype=np.float64)
    return (template / total).reshape(-1)


def _background_columns(rows: int, cols: int, valid_pixels: np.ndarray) -> list[np.ndarray]:
    yy, xx = np.mgrid[0:rows, 0:cols]
    valid = np.asarray(valid_pixels, dtype=np.float64).reshape(-1)
    x = xx.astype(np.float64).reshape(-1)
    y = yy.astype(np.float64).reshape(-1)
    if cols > 1:
        x = (x - (cols - 1) / 2.0) / ((cols - 1) / 2.0)
    else:
        x = x * 0.0
    if rows > 1:
        y = (y - (rows - 1) / 2.0) / ((rows - 1) / 2.0)
    else:
        y = y * 0.0
    return [valid, valid * x, valid * y]


def _build_design(
    *,
    sources: list[dict[str, Any]],
    rows: int,
    cols: int,
    valid_pixels: np.ndarray,
    sigma_x: float,
    axis_ratio: float,
    theta: float,
    wing_fraction: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    columns: list[np.ndarray] = []
    modeled: list[dict[str, Any]] = []
    for source in sources:
        center = source["pixelCenter"]
        x = float(center["x"]) + float(dx)
        y = float(center["y"]) + float(dy)
        template = _eprf_template(
            rows=rows,
            cols=cols,
            x=x,
            y=y,
            valid_pixels=valid_pixels,
            sigma_x=float(sigma_x),
            axis_ratio=float(axis_ratio),
            theta=float(theta),
            wing_fraction=float(wing_fraction),
        )
        if float(np.linalg.norm(template)) <= 1e-12:
            raise RuntimeError(f"Zero ePRF template for {source['componentID']}.")
        columns.append(template)
        modeled.append(
            {
                "componentID": source["componentID"],
                "catalogPixelCenter": dict(center),
                "calibratedPixelCenter": {"x": x, "y": y},
            }
        )
    columns.extend(_background_columns(rows, cols, valid_pixels))
    return np.column_stack(columns), modeled


def _fit_static_image(design: np.ndarray, image: np.ndarray, source_count: int) -> tuple[float, np.ndarray, float]:
    y = np.asarray(image, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    if int(np.count_nonzero(valid)) < max(12, design.shape[1] + 2):
        return float("inf"), np.zeros(design.shape[1]), float("-inf")
    a = design[valid]
    b = y[valid]
    coefficients, *_ = np.linalg.lstsq(a, b, rcond=None)
    # Median image source fluxes should not be strongly negative. Penalize such
    # solutions rather than using a constrained solver in the calibration loop.
    source_coefficients = coefficients[:source_count]
    negative_penalty = float(np.sum(np.square(np.minimum(source_coefficients, 0.0))))
    model = a @ coefficients
    residual = b - model
    sse = float(np.sum(residual * residual)) + negative_penalty * max(float(np.var(b)), 1.0) * 50.0
    centered = b - float(np.mean(b))
    sst = float(np.sum(centered * centered))
    explained = 1.0 - float(np.sum(residual * residual)) / sst if sst > 1e-12 else 0.0
    return sse, coefficients, explained


def _calibrate_sector_eprf(
    *,
    corrected_cube: np.ndarray,
    valid_pixels: np.ndarray,
    tpf: Any,
    target_sky: Any,
    candidate_sky: Any,
    nuisance_sources: list[dict[str, Any]],
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
    median_image = np.nanmedian(np.asarray(corrected_cube, dtype=np.float64), axis=0)

    best: dict[str, Any] | None = None
    source_count = len(sources)
    # Stage 1: calibrate the ePRF shape at the catalog positions.
    for sigma_x in CORE_SIGMA_GRID:
        for axis_ratio in AXIS_RATIO_GRID:
            for theta in THETA_GRID:
                for wing_fraction in WING_FRACTION_GRID:
                    design, modeled = _build_design(
                        sources=sources,
                        rows=rows,
                        cols=cols,
                        valid_pixels=valid_pixels,
                        sigma_x=sigma_x,
                        axis_ratio=axis_ratio,
                        theta=theta,
                        wing_fraction=wing_fraction,
                        dx=0.0,
                        dy=0.0,
                    )
                    score, coefficients, explained = _fit_static_image(
                        design, median_image, source_count
                    )
                    if best is None or score < best["score"]:
                        best = {
                            "score": score,
                            "coefficients": coefficients,
                            "explainedVariance": explained,
                            "sigmaX": float(sigma_x),
                            "axisRatio": float(axis_ratio),
                            "thetaRadians": float(theta),
                            "wingFraction": float(wing_fraction),
                            "dx": 0.0,
                            "dy": 0.0,
                            "design": design,
                            "sources": modeled,
                        }
    if best is None:
        raise RuntimeError("Unable to calibrate a sector ePRF model.")

    # Stage 2: refine one shared sub-pixel astrometric offset.  A shared offset
    # avoids fitting the desired target/counterpart amplitudes by moving the two
    # catalog sources independently.
    for dx in ASTROMETRIC_SHIFT_GRID:
        for dy in ASTROMETRIC_SHIFT_GRID:
            design, modeled = _build_design(
                sources=sources,
                rows=rows,
                cols=cols,
                valid_pixels=valid_pixels,
                sigma_x=best["sigmaX"],
                axis_ratio=best["axisRatio"],
                theta=best["thetaRadians"],
                wing_fraction=best["wingFraction"],
                dx=dx,
                dy=dy,
            )
            score, coefficients, explained = _fit_static_image(design, median_image, source_count)
            if score < best["score"]:
                best.update(
                    {
                        "score": score,
                        "coefficients": coefficients,
                        "explainedVariance": explained,
                        "dx": float(dx),
                        "dy": float(dy),
                        "design": design,
                        "sources": modeled,
                    }
                )

    design = np.asarray(best.pop("design"), dtype=np.float64)
    coefficients = np.asarray(best.pop("coefficients"), dtype=np.float64)
    condition_number = float(np.linalg.cond(design))
    component_ids = [item["componentID"] for item in best["sources"]]
    target_index = component_ids.index("target-control")
    candidate_index = component_ids.index("catalog-counterpart")
    tvec = design[:, target_index]
    cvec = design[:, candidate_index]
    denom = float(np.linalg.norm(tvec) * np.linalg.norm(cvec))
    template_correlation = float(np.dot(tvec, cvec) / denom) if denom > 1e-12 else 1.0

    best.update(
        {
            "backend": "sector-calibrated-empirical-eprf-v1",
            "designConditionNumber": condition_number,
            "targetCounterpartTemplateCorrelation": template_correlation,
            "medianSourceCoefficients": {
                component_ids[index]: float(coefficients[index])
                for index in range(len(component_ids))
            },
        }
    )

    if not math.isfinite(condition_number) or condition_number > MAX_DESIGN_CONDITION:
        raise RuntimeError(f"Calibrated ePRF design is ill-conditioned ({condition_number}).")
    if abs(template_correlation) >= MAX_TEMPLATE_CORRELATION:
        raise RuntimeError(
            f"Target/counterpart ePRF templates are too degenerate (correlation={template_correlation})."
        )
    if float(best.get("explainedVariance") or 0.0) < MIN_CALIBRATION_EXPLAINED_VARIANCE:
        raise RuntimeError(
            "Sector-calibrated ePRF does not explain enough of the median TPF image "
            f"(R2={best.get('explainedVariance')})."
        )
    return {**best, "design": design, "componentIDs": component_ids}


def _extract_calibrated_series(
    *,
    residual_cube: np.ndarray,
    calibration: dict[str, Any],
) -> dict[str, np.ndarray]:
    design = np.asarray(calibration["design"], dtype=np.float64)
    pinv = np.linalg.pinv(design)
    flat = np.asarray(residual_cube, dtype=np.float64).reshape(len(residual_cube), -1)
    finite = np.all(np.isfinite(flat), axis=0) & np.all(np.isfinite(design), axis=1)
    if int(np.count_nonzero(finite)) < max(12, design.shape[1] + 2):
        raise RuntimeError("Too few finite pixels for calibrated ePRF time-series extraction.")
    # Recompute the inverse using only stable finite pixels.
    reduced = design[finite]
    reduced_pinv = np.linalg.pinv(reduced)
    coefficients = (reduced_pinv @ flat[:, finite].T).T

    series: dict[str, np.ndarray] = {}
    for index, component_id in enumerate(calibration["componentIDs"]):
        values = np.asarray(coefficients[:, index], dtype=np.float64)
        values -= float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std <= 1e-12:
            continue
        series[component_id] = values / std
    if "target-control" not in series or "catalog-counterpart" not in series:
        raise RuntimeError("Calibrated ePRF extraction did not produce both validation series.")
    return series


def build_calibrated_prf_deblending_project(
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
    offset_source_variability: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.15 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )
    if offset_source_variability.get("recommendedNextTest") != "CALIBRATED_PRF_SOURCE_DEBLENDING":
        raise RuntimeError("v20.15 requires v20.14 to recommend calibrated PRF source deblending.")

    best_offset = _best_offset_summary(multisource_summary)
    reference_frequency = _float(best_offset.get("combinedFrequency"))
    if reference_frequency is None or reference_frequency <= 0:
        reference_frequency = _float(nonstationary_summary.get("preferredFrequencyAtReference"))
    q = _float(nonstationary_summary.get("fractionalFrequencyDriftPerDay"))
    time_reference = _float(nonstationary_summary.get("timeReferenceDays"))
    if reference_frequency is None or reference_frequency <= 0 or q is None or time_reference is None:
        raise RuntimeError("v20.15 requires the completed v20.9 drift model and v20.12 offset frequency.")

    target_meta = ((identity.get("tic") or {}).get("metadata") or {})
    target_ra = _float(target_meta.get("raDeg"))
    target_dec = _float(target_meta.get("decDeg"))
    if target_ra is None or target_dec is None:
        raise RuntimeError("v20.15 requires TIC RA/Dec from the identity stage.")

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
        raise RuntimeError("v20.15 found no frozen sectors for calibrated ePRF deblending.")

    physical_frequency = 1.0 / float(physical_period_days)
    search = _frequency_search(float(reference_frequency))
    target_sky = _skycoord(float(target_ra), float(target_dec))
    candidate_sky = _skycoord(candidate_ra, candidate_dec)
    root = Path(output_dir) / "calibrated-prf-deblending"
    root.mkdir(parents=True, exist_ok=True)
    source_base_id = str(source_dataset_entry.get("id") or f"tic-{target_tic_id}")

    dataset_entries: list[dict[str, Any]] = []
    prepared_series: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    calibration_diagnostics: list[dict[str, Any]] = []
    combined: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "target-control": [],
        "catalog-counterpart": [],
    }

    for sector_index, (sector, role) in enumerate(sectors, start=1):
        print(
            f"   Sector {sector} ({sector_index}/{len(sectors)}): calibrating sector ePRF + deblending target/counterpart",
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
            finite_time = np.isfinite(absolute_times)
            finite_frame = np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
            keep = finite_time & finite_frame
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
            calibration = _calibrate_sector_eprf(
                corrected_cube=corrected,
                valid_pixels=valid_pixels,
                tpf=tpf,
                target_sky=target_sky,
                candidate_sky=candidate_sky,
                nuisance_sources=nuisance_sources,
            )
            component_series = _extract_calibrated_series(
                residual_cube=residual_cube,
                calibration=calibration,
            )
            diagnostic = {
                key: value
                for key, value in calibration.items()
                if key not in {"design"}
            }
            diagnostic.update(
                {
                    "sector": int(sector),
                    "role": role,
                    "sourceType": source.get("sourceType"),
                    "author": source.get("author"),
                    "cadenceSeconds": source.get("cadenceSeconds"),
                }
            )
            calibration_diagnostics.append(diagnostic)

            relative_times = absolute_times - float(time_reference)
            warped = _time_warp(relative_times, float(q))
            local_times = warped - float(np.min(warped))
            for component_id in ("target-control", "catalog-counterpart"):
                values = component_series[component_id]
                dataset_id = f"{source_base_id}-prf-validation-{component_id}-sector-{sector}-v1"
                target_name = (
                    f"{source_dataset_entry.get('targetName') or source_base_id} "
                    f"calibrated ePRF {component_id} sector {sector}"
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
                        "role": "calibrated-prf-source-deblending",
                        "componentID": component_id,
                        "sector": int(sector),
                        "sectorRole": role,
                        "referenceFrequency": float(reference_frequency),
                        "fractionalFrequencyDriftPerDay": float(q),
                        "deblendBackend": "sector-calibrated-empirical-eprf-v1",
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
                f"      ePRF R2={calibration.get('explainedVariance'):.3f}, "
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
        dataset_id = f"{source_base_id}-prf-validation-{component_id}-combined-v1"
        target_name = (
            f"{source_dataset_entry.get('targetName') or source_base_id} "
            f"calibrated ePRF {component_id} combined"
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
                "role": "calibrated-prf-source-deblending-combined",
                "componentID": component_id,
                "referenceFrequency": float(reference_frequency),
                "fractionalFrequencyDriftPerDay": float(q),
                "deblendBackend": "sector-calibrated-empirical-eprf-v1",
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
        raise RuntimeError("v20.15 could not prepare any calibrated ePRF validation datasets.")

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "calibrated-prf-source-deblending-v1"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — calibrated ePRF source deblending",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "calibrated-prf-source-deblending",
            "workerSemantics": (
                "Each dataset is a sector-calibrated empirical-ePRF-deblended, established-family-prewhitened, "
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
        "version": "openstar.tess-calibrated-prf-source-deblending-preparation.v1",
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-sector-calibrated-eprf-deblended-drift-corrected-source-series",
        "deblendBackend": "sector-calibrated-empirical-eprf-v1",
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
        "calibrationDiagnostics": calibration_diagnostics,
        "nuisanceCatalogSources": nuisance_sources,
        "errors": errors,
        "workUnitsPerDataset": work_units_per_dataset,
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "interpretationGuard": (
            "v20.15 replaces v20.14's fixed Gaussian templates with a sector-calibrated empirical pixel-response "
            "model fit to the actual TPF median image at fixed catalog source positions. This is substantially more "
            "source-specific than the Gaussian test, but it is not claimed to be the SPOC engineering PRF calibration."
        ),
    }


def _result_record(dataset: dict[str, Any], meta: dict[str, Any], preparation: dict[str, Any]) -> dict[str, Any]:
    frequency = _float(dataset.get("candidateFrequency"))
    period = _float(dataset.get("candidatePeriodDays"))
    power = _float(dataset.get("candidatePower"))
    prominence = _float(dataset.get("candidatePeakProminenceRatio"))
    status = str(dataset.get("periodStatus") or "").upper()
    confidence = str(dataset.get("periodConfidence") or "none").lower()
    baseline = _float(meta.get("baselineDays")) or 0.0
    observed_cycles = (baseline / period) if period and period > 0 else 0.0
    reference_frequency = float(preparation["referenceFrequency"])
    relative_difference = (
        abs(float(frequency) - reference_frequency) / reference_frequency
        if frequency is not None
        else None
    )
    rayleigh = (1.0 / baseline) if baseline > 0 else None
    reference_consistent = bool(
        frequency is not None
        and (
            (relative_difference is not None and relative_difference <= REFERENCE_FREQUENCY_TOLERANCE_FRACTION)
            or (rayleigh is not None and abs(float(frequency) - reference_frequency) <= rayleigh)
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
        "relativeFrequencyDifferenceFromReference": relative_difference,
        "referenceConsistent": reference_consistent,
        "acceptedResidualMatch": accepted,
    }


def _component_summary(component_id: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    component = [item for item in results if item.get("componentID") == component_id]
    sectors = [item for item in component if not item.get("combined")]
    independent = [item for item in sectors if item.get("role") == "independent"]
    accepted_independent = [item for item in independent if item.get("acceptedResidualMatch")]
    accepted_all = [item for item in sectors if item.get("acceptedResidualMatch")]
    combined = next((item for item in component if item.get("combined")), None)
    powers = [
        float(item["candidatePower"])
        for item in accepted_all
        if _float(item.get("candidatePower")) is not None
    ]
    return {
        "componentID": component_id,
        "independentSupportCount": len(accepted_independent),
        "independentSupportingSectors": sorted(int(item["sector"]) for item in accepted_independent),
        "allSupportingSectors": sorted(
            int(item["sector"]) for item in accepted_all if item.get("sector") is not None
        ),
        "medianAcceptedSectorPower": statistics.median(powers) if powers else None,
        "combinedAccepted": bool(combined and combined.get("acceptedResidualMatch")),
        "combinedPower": combined.get("candidatePower") if combined else None,
        "combinedPeriodDays": combined.get("candidatePeriodDays") if combined else None,
        "combinedFrequency": combined.get("candidateFrequency") if combined else None,
        "combinedProminence": combined.get("candidatePeakProminenceRatio") if combined else None,
    }


def interpret_calibrated_prf_deblending_project(
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
    candidate_support = int(counterpart.get("independentSupportCount") or 0)
    target_power = float(target.get("combinedPower") or 0.0)
    candidate_power = float(counterpart.get("combinedPower") or 0.0)
    target_present = target_support >= MIN_INDEPENDENT_SUPPORT and bool(target.get("combinedAccepted"))
    candidate_present = candidate_support >= MIN_INDEPENDENT_SUPPORT and bool(counterpart.get("combinedAccepted"))
    candidate_dominant = bool(
        candidate_present
        and (
            not target_present
            or target_power <= 0
            or candidate_power >= target_power * DOMINANCE_POWER_RATIO
        )
    )

    if candidate_dominant:
        classification = "PRF_OFFSET_COUNTERPART_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_AFTER_PRF_DEBLENDING"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif candidate_present and target_present:
        classification = "PRF_TARGET_AND_OFFSET_VARIABILITY_SUPPORTED"
        origin = "BLENDED_TARGET_AND_COUNTERPART_AFTER_PRF_DEBLENDING"
        next_test = "JOINT_TARGET_OFFSET_VARIABILITY_MODEL"
    elif target_present and not candidate_present:
        classification = "PRF_TARGET_CONTROL_DOMINANT"
        origin = "TARGET_CONTROL_SUPPORTED_AFTER_PRF_DEBLENDING"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif candidate_support >= 2 and candidate_power > target_power:
        classification = "PRF_OFFSET_COUNTERPART_VARIABILITY_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE_AFTER_PRF_DEBLENDING"
        next_test = "DIFFERENCE_IMAGE_SOURCE_LOCALIZATION"
    else:
        classification = "PRF_DEBLENDING_UNRESOLVED"
        origin = "UNRESOLVED_AFTER_PRF_DEBLENDING"
        next_test = "DIFFERENCE_IMAGE_SOURCE_LOCALIZATION"

    return {
        "version": "openstar.tess-calibrated-prf-source-deblending.v1",
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "deblendBackend": preparation.get("deblendBackend"),
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
        "offsetCounterpartVariabilitySupported": classification == "PRF_OFFSET_COUNTERPART_VARIABILITY_SUPPORTED",
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "preparationErrors": preparation.get("errors") or [],
        "calibrationDiagnostics": preparation.get("calibrationDiagnostics") or [],
        "interpretationGuard": (
            "v20.15 tests the v20.13 counterpart using a sector-calibrated empirical TESS pixel-response model "
            "instead of v20.14's fixed Gaussian templates, then applies the same residual-frequency evidence guards. "
            "It does not alter the v20.6 target association for the established 13.72-day family and does not claim "
            "the empirical ePRF is identical to the SPOC engineering PRF calibration."
        ),
    }
