from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    LOMB_SCARGLE_WORKLOAD_ALIASES,
    MAX_CADENCES,
    MIN_VALID_PIXEL_FRACTION,
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

TOTAL_FREQUENCIES = 8_192
FREQUENCIES_PER_WORK_UNIT = 2_048
FREQUENCY_HALF_WIDTH_FRACTION = 0.20
SOURCE_CLUSTER_RADIUS_ARCSEC = 30.0
MAX_OFFSET_COMPONENTS = 3
GAUSSIAN_SIGMA_PIXELS = 0.85
GAUSSIAN_RADIUS_PIXELS = 2.75
MIN_COMPONENT_SAMPLES = 80
MIN_COMPONENT_POWER = 0.08
MIN_INDEPENDENT_SUPPORT = 2
DOMINANCE_POWER_RATIO = 1.25
MIN_SECTOR_COMPONENT_RMS_FRACTION = 0.25


def _design_matrix(times: np.ndarray, physical_frequency: float) -> np.ndarray:
    centered = times - float(np.mean(times))
    scale = float(np.std(centered))
    trend = centered / scale if scale > 0 else centered
    omega = 2.0 * math.pi * float(physical_frequency)
    return np.column_stack(
        [
            np.ones(len(times), dtype=np.float64),
            trend,
            np.sin(omega * times),
            np.cos(omega * times),
            np.sin(2.0 * omega * times),
            np.cos(2.0 * omega * times),
        ]
    )


def _prewhiten_cube_raw(
    *,
    absolute_times: np.ndarray,
    cube: np.ndarray,
    physical_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _design_matrix(absolute_times, physical_frequency)
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
    residual[:, ~valid] = 0.0
    return residual.reshape(cube.shape), valid.reshape(cube.shape[1:])


def _frequency_search(reference_frequency: float) -> dict[str, Any]:
    minimum = float(reference_frequency) * (1.0 - FREQUENCY_HALF_WIDTH_FRACTION)
    maximum = float(reference_frequency) * (1.0 + FREQUENCY_HALF_WIDTH_FRACTION)
    if minimum <= 0 or maximum <= minimum:
        raise RuntimeError("Invalid v20.12 residual-component frequency search.")
    return {
        "minimumFrequency": minimum,
        "maximumFrequency": maximum,
        "frequencyStep": (maximum - minimum) / (TOTAL_FREQUENCIES - 1),
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def _cluster_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = [dict(item) for item in points]
    clusters: list[list[dict[str, Any]]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            keep: list[dict[str, Any]] = []
            for item in remaining:
                east = float(item["eastArcsec"])
                north = float(item["northArcsec"])
                if any(
                    math.hypot(
                        east - float(member["eastArcsec"]),
                        north - float(member["northArcsec"]),
                    ) <= SOURCE_CLUSTER_RADIUS_ARCSEC
                    for member in cluster
                ):
                    cluster.append(item)
                    changed = True
                else:
                    keep.append(item)
            remaining = keep
        clusters.append(cluster)

    ranked = sorted(clusters, key=lambda group: (-len(group), statistics.median(
        math.hypot(float(item["eastArcsec"]), float(item["northArcsec"])) for item in group
    )))
    result: list[dict[str, Any]] = []
    for index, group in enumerate(ranked[:MAX_OFFSET_COMPONENTS], start=1):
        east = statistics.median(float(item["eastArcsec"]) for item in group)
        north = statistics.median(float(item["northArcsec"]) for item in group)
        scatter = statistics.median(
            math.hypot(float(item["eastArcsec"]) - east, float(item["northArcsec"]) - north)
            for item in group
        )
        result.append(
            {
                "componentID": f"offset-{index}",
                "componentType": "OFFSET",
                "eastArcsec": float(east),
                "northArcsec": float(north),
                "supportingWindows": len(group),
                "supportingSectors": sorted({int(item["sector"]) for item in group}),
                "skyScatterArcsec": float(scatter),
            }
        )
    return result


def identify_spatial_components(review: dict[str, Any]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = [
        {
            "componentID": "target",
            "componentType": "TARGET",
            "eastArcsec": 0.0,
            "northArcsec": 0.0,
            "supportingWindows": sum(
                1
                for item in review.get("windowResults") or []
                if item.get("localizationQualityPass")
                and item.get("classification") == "TARGET_CONSISTENT"
            ),
            "supportingSectors": sorted(
                {
                    int(item["sector"])
                    for item in review.get("windowResults") or []
                    if item.get("localizationQualityPass")
                    and item.get("classification") == "TARGET_CONSISTENT"
                    and _int(item.get("sector")) is not None
                }
            ),
            "skyScatterArcsec": 0.0,
        }
    ]
    off_points: list[dict[str, Any]] = []
    for item in review.get("windowResults") or []:
        if not item.get("localizationQualityPass") or item.get("classification") != "OFF_TARGET":
            continue
        east = _float(item.get("skyOffsetEastArcsec"))
        north = _float(item.get("skyOffsetNorthArcsec"))
        sector = _int(item.get("sector"))
        if east is None or north is None or sector is None:
            continue
        off_points.append(
            {
                "eastArcsec": east,
                "northArcsec": north,
                "sector": sector,
                "windowIndex": _int(item.get("windowIndex")),
            }
        )
    components.extend(_cluster_points(off_points))
    return components


def _pixel_delta_from_sky(
    *,
    east_arcsec: float,
    north_arcsec: float,
    jacobian: dict[str, Any],
) -> tuple[float, float] | None:
    x_e = _float(jacobian.get("xToEastArcsec"))
    x_n = _float(jacobian.get("xToNorthArcsec"))
    y_e = _float(jacobian.get("yToEastArcsec"))
    y_n = _float(jacobian.get("yToNorthArcsec"))
    if None in (x_e, x_n, y_e, y_n):
        return None
    matrix = np.asarray([[x_e, y_e], [x_n, y_n]], dtype=np.float64)
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        return None
    delta = np.linalg.solve(matrix, np.asarray([east_arcsec, north_arcsec], dtype=np.float64))
    return float(delta[0]), float(delta[1])


def _gaussian_template(
    *,
    rows: int,
    cols: int,
    x: float,
    y: float,
    valid_pixels: np.ndarray,
) -> np.ndarray:
    yy, xx = np.indices((rows, cols), dtype=np.float64)
    distance2 = np.square(xx - float(x)) + np.square(yy - float(y))
    weights = np.exp(-0.5 * distance2 / (GAUSSIAN_SIGMA_PIXELS**2))
    weights[distance2 > GAUSSIAN_RADIUS_PIXELS**2] = 0.0
    weights *= np.asarray(valid_pixels, dtype=np.float64)
    norm = float(np.linalg.norm(weights))
    if norm <= 1e-12:
        return np.zeros(rows * cols, dtype=np.float64)
    return (weights / norm).reshape(-1)


def _decompose_residual_cube(
    *,
    residual_cube: np.ndarray,
    valid_pixels: np.ndarray,
    target_x: float,
    target_y: float,
    jacobian: dict[str, Any],
    components: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows, cols = valid_pixels.shape
    templates: list[np.ndarray] = []
    usable_components: list[dict[str, Any]] = []
    for component in components:
        delta = _pixel_delta_from_sky(
            east_arcsec=float(component.get("eastArcsec") or 0.0),
            north_arcsec=float(component.get("northArcsec") or 0.0),
            jacobian=jacobian,
        )
        if delta is None:
            continue
        center_x = float(target_x) + delta[0]
        center_y = float(target_y) + delta[1]
        template = _gaussian_template(
            rows=rows,
            cols=cols,
            x=center_x,
            y=center_y,
            valid_pixels=valid_pixels,
        )
        if float(np.linalg.norm(template)) <= 1e-12:
            continue
        templates.append(template)
        usable_components.append(
            {
                **component,
                "pixelCenter": {"x": center_x, "y": center_y},
            }
        )

    if not templates:
        return {}, []

    background = np.asarray(valid_pixels, dtype=np.float64).reshape(-1)
    bg_norm = float(np.linalg.norm(background))
    if bg_norm > 0:
        background /= bg_norm
        templates.append(background)

    spatial = np.column_stack(templates)
    component_spatial = spatial[:, :len(usable_components)]
    template_overlap = (
        float(np.dot(component_spatial[:, 0], component_spatial[:, 1]))
        if component_spatial.shape[1] == 2 else None
    )
    condition_number = float(np.linalg.cond(spatial))
    pinv = np.linalg.pinv(spatial)
    flat = residual_cube.reshape(len(residual_cube), -1).astype(np.float64)
    coefficients = (pinv @ flat.T).T

    series: dict[str, np.ndarray] = {}
    for index, component in enumerate(usable_components):
        values = np.asarray(coefficients[:, index], dtype=np.float64)
        values -= float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std <= 1e-12:
            continue
        series[str(component["componentID"])] = values / std
        component["coefficientRMS"] = std
        component["spatialDesignConditionNumber"] = condition_number
        component["normalizedTemplateOverlap"] = template_overlap
    return series, usable_components


def build_multisource_residual_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    tic_id: int,
    identity: dict[str, Any],
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    nonstationary_summary: dict[str, Any],
    localization_review: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.12 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )
    if localization_review.get("recommendedNextTest") != "MULTI_SOURCE_RESIDUAL_DECOMPOSITION":
        raise RuntimeError("v20.12 requires v20.11 to recommend MULTI_SOURCE_RESIDUAL_DECOMPOSITION.")

    preferred = nonstationary_summary.get("preferredModel") or {}
    reference_frequency = _float(nonstationary_summary.get("preferredFrequencyAtReference"))
    q = _float(nonstationary_summary.get("fractionalFrequencyDriftPerDay"))
    time_reference = _float(nonstationary_summary.get("timeReferenceDays"))
    signal_sectors = [
        int(value)
        for value in preferred.get("signalSectors") or []
        if _int(value) is not None
    ]
    if reference_frequency is None or reference_frequency <= 0 or q is None or time_reference is None:
        raise RuntimeError("v20.12 requires the completed v20.9 residual drift model.")

    tic_metadata = ((identity.get("tic") or {}).get("metadata") or {})
    ra_deg = _float(tic_metadata.get("raDeg"))
    dec_deg = _float(tic_metadata.get("decDeg"))
    if ra_deg is None or dec_deg is None:
        raise RuntimeError("v20.12 requires TIC RA/Dec from the identity stage.")

    components = identify_spatial_components(localization_review)
    if len(components) < 2:
        raise RuntimeError(
            "v20.12 did not recover an offset spatial component from the v20.11 review."
        )

    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    sectors = _sector_candidates(
        primary_sector=primary_sector,
        independent_spec=independent_spec,
        signal_sectors=signal_sectors,
    )
    if not sectors:
        raise RuntimeError("No frozen sectors overlap the v20.9 signal-sector set.")

    root = Path(output_dir) / "multi-source-residual"
    root.mkdir(parents=True, exist_ok=True)
    search = _frequency_search(reference_frequency)
    physical_frequency = 1.0 / float(physical_period_days)
    source_base_id = str(source_dataset_entry.get("id") or f"tic-{tic_id}")

    dataset_entries: list[dict[str, Any]] = []
    prepared_series: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    combined: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        str(component["componentID"]): [] for component in components
    }

    for sector_index, (sector, role) in enumerate(sectors, start=1):
        print(f"   Sector {sector} ({sector_index}/{len(sectors)}): decomposing spatial residual components", flush=True)
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

            target_x, target_y = tpf.wcs.world_to_pixel(target)
            jacobian = _local_sky_jacobian(tpf.wcs, target, float(target_x), float(target_y))
            component_series, usable_components = _decompose_residual_cube(
                residual_cube=residual_cube,
                valid_pixels=valid_pixels,
                target_x=float(target_x),
                target_y=float(target_y),
                jacobian=jacobian,
                components=components,
            )

            relative_times = absolute_times - float(time_reference)
            warped = _time_warp(relative_times, float(q))
            local_times = warped - float(np.min(warped))
            for component in usable_components:
                component_id = str(component["componentID"])
                values = component_series.get(component_id)
                if values is None or len(values) < MIN_COMPONENT_SAMPLES:
                    continue
                dataset_id = f"{source_base_id}-multisource-{component_id}-sector-{sector}-v1"
                target_name = (
                    f"{source_dataset_entry.get('targetName') or source_base_id} "
                    f"multi-source residual {component_id} sector {sector}"
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
                        "role": "multi-source-residual-component",
                        "componentID": component_id,
                        "componentType": component.get("componentType"),
                        "sector": int(sector),
                        "sectorRole": role,
                        "referenceFrequency": float(reference_frequency),
                        "fractionalFrequencyDriftPerDay": float(q),
                        "coefficientRMS": component.get("coefficientRMS"),
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
                dataset_entries.append({"id": dataset_id, "path": str(output_path.resolve()), "targetName": target_name})
                prepared_series.append(
                    {
                        "datasetID": dataset_id,
                        "datasetPath": str(output_path.resolve()),
                        "componentID": component_id,
                        "componentType": component.get("componentType"),
                        "sector": int(sector),
                        "role": role,
                        "combined": False,
                        "sampleCount": int(len(values)),
                        "pixelCenter": component.get("pixelCenter"),
                        "coefficientRMS": component.get("coefficientRMS"),
                        "spatialDesignConditionNumber": component.get("spatialDesignConditionNumber"),
                        "normalizedTemplateOverlap": component.get("normalizedTemplateOverlap"),
                    }
                )
                combined.setdefault(component_id, []).append((np.asarray(warped, dtype=np.float64), np.asarray(values, dtype=np.float64)))
            print(f"      extracted components: {sorted(component_series)}", flush=True)
        except Exception as exc:
            errors.append({"sector": int(sector), "error": f"{type(exc).__name__}: {exc}"})
            print(f"      unavailable: {type(exc).__name__}: {exc}", flush=True)

    for component in components:
        component_id = str(component["componentID"])
        pieces = combined.get(component_id) or []
        if len(pieces) < 2:
            continue
        all_times = np.concatenate([item[0] for item in pieces])
        all_flux = np.concatenate([item[1] for item in pieces])
        order = np.argsort(all_times)
        all_times = all_times[order]
        all_flux = all_flux[order]
        all_times = all_times - float(np.min(all_times))
        all_flux = all_flux - float(np.mean(all_flux))
        std = float(np.std(all_flux))
        if not math.isfinite(std) or std <= 1e-12:
            continue
        all_flux /= std
        dataset_id = f"{source_base_id}-multisource-{component_id}-combined-v1"
        target_name = f"{source_dataset_entry.get('targetName') or source_base_id} multi-source residual {component_id} combined"
        output_path = root / f"{_safe(dataset_id)}.json"
        dataset = {
            "id": dataset_id,
            "targetName": target_name,
            "times": np.asarray(all_times, dtype=np.float32).tolist(),
            "flux": np.asarray(all_flux, dtype=np.float32).tolist(),
            "frequencySearch": search,
            "reference": {},
            "science": {
                "role": "multi-source-residual-component-combined",
                "componentID": component_id,
                "componentType": component.get("componentType"),
                "referenceFrequency": float(reference_frequency),
                "fractionalFrequencyDriftPerDay": float(q),
                "coefficientRMS": std,
            },
            "source": {
                "mission": "TESS",
                "distributedSamples": int(len(all_times)),
                "timeReferenceDays": float(time_reference),
                "combinedSectors": True,
            },
        }
        _write_json(output_path, dataset)
        dataset_entries.append({"id": dataset_id, "path": str(output_path.resolve()), "targetName": target_name})
        prepared_series.append(
            {
                "datasetID": dataset_id,
                "datasetPath": str(output_path.resolve()),
                "componentID": component_id,
                "componentType": component.get("componentType"),
                "sector": None,
                "role": "combined",
                "combined": True,
                "sampleCount": int(len(all_flux)),
                "coefficientRMS": std,
            }
        )

    if not dataset_entries:
        raise RuntimeError("v20.12 could not prepare any multi-source residual component datasets.")

    project_id = f"{source_project['id']}.investigation.{_safe(investigation_id)}.multi-source-residual-decomposition-v1"
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — multi-source residual decomposition",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "multi-source-residual-decomposition",
            "workerSemantics": (
                "Each dataset is one spatially decomposed, established-family-prewhitened, "
                "v20.9 drift-corrected component light curve. Workers execute ordinary Lomb-Scargle only."
            ),
            "referenceFrequency": float(reference_frequency),
            "fractionalFrequencyDriftPerDay": float(q),
            "spatialComponents": components,
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
        "workerSemantics": "generic-lomb-scargle-on-spatially-decomposed-drift-corrected-residual-components",
        "ticID": int(tic_id),
        "targetSky": {"raDeg": float(ra_deg), "decDeg": float(dec_deg)},
        "physicalPeriodDays": float(physical_period_days),
        "referenceFrequency": float(reference_frequency),
        "referencePeriodDays": float(1.0 / reference_frequency),
        "fractionalFrequencyDriftPerDay": float(q),
        "timeReferenceDays": float(time_reference),
        "frequencySearch": search,
        "spatialComponents": components,
        "preparedSeries": prepared_series,
        "errors": errors,
        "workUnitsPerDataset": work_units_per_dataset,
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
    }


def _accepted(dataset: dict[str, Any]) -> bool:
    power = _float(dataset.get("candidatePower"))
    status = str(dataset.get("periodStatus") or "").upper()
    confidence = str(dataset.get("periodConfidence") or "none").lower()
    return bool(
        power is not None
        and power >= MIN_COMPONENT_POWER
        and status == "RELIABLE"
        and confidence in {"high", "medium"}
    )


def interpret_multisource_residual_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {str(item.get("datasetID")): item for item in preparation.get("preparedSeries") or []}
    results: list[dict[str, Any]] = []
    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id)
        if meta is None:
            continue
        result = {
            **meta,
            "candidatePower": _float(dataset.get("candidatePower")),
            "candidateFrequency": _float(dataset.get("candidateFrequency")),
            "candidatePeriodDays": _float(dataset.get("candidatePeriodDays")),
            "periodStatus": str(dataset.get("periodStatus") or "").upper(),
            "periodConfidence": str(dataset.get("periodConfidence") or "none").lower(),
        }
        result["accepted"] = _accepted(dataset)
        results.append(result)

    sector_max_rms: dict[int, float] = {}
    for result in results:
        sector = _int(result.get("sector"))
        rms = _float(result.get("coefficientRMS"))
        if sector is not None and rms is not None:
            sector_max_rms[sector] = max(sector_max_rms.get(sector, 0.0), rms)
    for result in results:
        sector = _int(result.get("sector"))
        rms = _float(result.get("coefficientRMS"))
        maximum = sector_max_rms.get(sector, 0.0) if sector is not None else 0.0
        fraction = rms / maximum if rms is not None and maximum > 0 else None
        result["componentRMSFractionOfSectorMaximum"] = fraction
        result["supportRMSFractionThreshold"] = MIN_SECTOR_COMPONENT_RMS_FRACTION
        result["amplitudeQualified"] = bool(
            result.get("combined") or (
                fraction is not None and fraction >= MIN_SECTOR_COMPONENT_RMS_FRACTION
            )
        )
        result["countedAsIndependentSupport"] = bool(
            result.get("role") == "independent"
            and result.get("accepted")
            and result.get("amplitudeQualified")
        )

    summaries: list[dict[str, Any]] = []
    for component in preparation.get("spatialComponents") or []:
        component_id = str(component.get("componentID"))
        component_results = [item for item in results if item.get("componentID") == component_id]
        sector_results = [item for item in component_results if not item.get("combined")]
        independent = [item for item in sector_results if item.get("role") == "independent"]
        accepted_independent = [
            item for item in independent if item.get("countedAsIndependentSupport")
        ]
        accepted_all = [
            item for item in sector_results
            if item.get("accepted") and item.get("amplitudeQualified")
        ]
        combined = next((item for item in component_results if item.get("combined")), None)
        powers = [float(item["candidatePower"]) for item in accepted_all if _float(item.get("candidatePower")) is not None]
        summaries.append(
            {
                **component,
                "independentSupportCount": len(accepted_independent),
                "independentSupportingSectors": sorted(int(item["sector"]) for item in accepted_independent),
                "allSupportingSectors": sorted(int(item["sector"]) for item in accepted_all if item.get("sector") is not None),
                "medianAcceptedSectorPower": statistics.median(powers) if powers else None,
                "combinedAccepted": bool(combined and combined.get("accepted")),
                "combinedPower": combined.get("candidatePower") if combined else None,
                "combinedPeriodDays": combined.get("candidatePeriodDays") if combined else None,
                "combinedFrequency": combined.get("candidateFrequency") if combined else None,
            }
        )

    target = next((item for item in summaries if item.get("componentType") == "TARGET"), None)
    offsets = [item for item in summaries if item.get("componentType") == "OFFSET"]
    best_offset = max(
        offsets,
        key=lambda item: (
            int(item.get("independentSupportCount") or 0),
            float(item.get("combinedPower") or 0.0),
            float(item.get("medianAcceptedSectorPower") or 0.0),
        ),
        default=None,
    )

    target_support = int((target or {}).get("independentSupportCount") or 0)
    offset_support = int((best_offset or {}).get("independentSupportCount") or 0)
    target_power = float((target or {}).get("combinedPower") or 0.0)
    offset_power = float((best_offset or {}).get("combinedPower") or 0.0)
    target_present = target_support >= MIN_INDEPENDENT_SUPPORT and bool((target or {}).get("combinedAccepted"))
    offset_present = offset_support >= MIN_INDEPENDENT_SUPPORT and bool((best_offset or {}).get("combinedAccepted"))

    if target_present and offset_present:
        classification = "MULTIPLE_RESIDUAL_SOURCES_SUPPORTED"
        origin = "TARGET_AND_OFFSET_COMPONENTS"
        next_test = "NEIGHBOR_SOURCE_IDENTIFICATION_AND_CATALOG_CROSSMATCH"
    elif target_present and (offset_power <= 0 or target_power >= offset_power * DOMINANCE_POWER_RATIO):
        classification = "TARGET_RESIDUAL_COMPONENT_DOMINANT"
        origin = "TARGET_DOMINANT"
        next_test = "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"
    elif offset_present and (target_power <= 0 or offset_power >= target_power * DOMINANCE_POWER_RATIO):
        classification = "OFF_TARGET_RESIDUAL_COMPONENT_DOMINANT"
        origin = "OFFSET_DOMINANT"
        next_test = "IDENTIFY_OFFSET_RESIDUAL_VARIABLE_SOURCE"
    else:
        classification = "MULTI_SOURCE_DECOMPOSITION_UNRESOLVED"
        origin = "UNRESOLVED"
        next_test = "PIXEL_RESPONSE_FUNCTION_DEBLENDING"

    return {
        "version": "openstar.tess-multi-source-residual-decomposition.v1",
        "distributedDecomposition": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "supportQualification": {
            "minimumCandidatePower": MIN_COMPONENT_POWER,
            "minimumIndependentSupport": MIN_INDEPENDENT_SUPPORT,
            "minimumSectorComponentRMSFraction": MIN_SECTOR_COMPONENT_RMS_FRACTION,
            "reason": (
                "A normalized component periodogram is not independent-source evidence unless "
                "the component's pre-normalization coefficient RMS is also a material fraction "
                "of the strongest fitted spatial component in that sector."
            ),
        },
        "spatialComponents": preparation.get("spatialComponents") or [],
        "componentResults": results,
        "componentSummaries": summaries,
        "classification": classification,
        "residualModeOrigin": origin,
        "targetComponentID": (target or {}).get("componentID"),
        "bestOffsetComponentID": (best_offset or {}).get("componentID"),
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "This decomposes only the v20.9 drifting residual structure. It does not alter "
            "v20.6's target association for the established 13.72-day family. Spatial templates "
            "are deterministic Gaussian approximations, not calibrated TESS PRF solutions."
        ),
    }
