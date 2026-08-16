from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    LOMB_SCARGLE_WORKLOAD_ALIASES,
    MAX_CADENCES,
    MIN_VALID_CADENCES,
    MIN_VALID_PIXEL_FRACTION,
    TOTAL_FREQUENCIES,
    FREQUENCIES_PER_WORK_UNIT,
    _download_tpf,
    _background_subtract_cube,
    _frequency_search,
    _float,
    _int,
    _load_json,
    _local_sky_jacobian,
    _localize_power_map,
    _pixel_scale_arcsec,
    _prewhiten_cube,
    _safe,
    _sector_candidates,
    _time_warp,
    _uniform_indices,
    _write_json,
)

WINDOW_LENGTH_DAYS = 12.0
WINDOWS_PER_SECTOR = 3
MIN_WINDOW_CADENCES = 180
MIN_WINDOW_CYCLES = 2.0
MAX_OFF_TARGET_SKY_SCATTER_ARCSEC = 15.0
MIN_INDEPENDENT_SECTORS_FOR_RESOLUTION = 3


def _window_bounds(times: np.ndarray, *, residual_period_days: float) -> list[tuple[int, float, float, float]]:
    if len(times) < MIN_WINDOW_CADENCES:
        return []
    start = float(np.min(times))
    end = float(np.max(times))
    baseline = end - start
    minimum_length = max(float(residual_period_days) * MIN_WINDOW_CYCLES, 8.0)
    length = max(WINDOW_LENGTH_DAYS, minimum_length)
    length = min(length, baseline)
    if baseline <= 0 or length <= 0:
        return []

    if baseline <= length * 1.05:
        starts = [start]
    else:
        max_start = end - length
        starts = np.linspace(start, max_start, WINDOWS_PER_SECTOR).tolist()

    windows: list[tuple[int, float, float, float]] = []
    for index, w_start in enumerate(starts, start=1):
        w_end = min(end, float(w_start) + length)
        midpoint = 0.5 * (float(w_start) + float(w_end))
        windows.append((index, float(w_start), float(w_end), float(midpoint)))
    return windows


def build_residual_mode_localization_review_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    tic_id: int,
    identity: dict[str, Any],
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    nonstationary_summary: dict[str, Any],
    residual_localization_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.11 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )

    if residual_localization_summary.get("recommendedNextTest") != "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW":
        raise RuntimeError(
            "v20.11 requires v20.10 to recommend RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW."
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
        raise RuntimeError("v20.11 requires the v20.9 preferred residual frequency.")
    if q is None:
        raise RuntimeError("v20.11 requires the v20.9 fractional frequency drift.")
    if time_reference is None:
        raise RuntimeError("v20.11 requires the v20.9 time reference.")
    if physical_period_days <= 0:
        raise RuntimeError("v20.11 requires a positive resolved physical period.")

    tic_metadata = ((identity.get("tic") or {}).get("metadata") or {})
    ra_deg = _float(tic_metadata.get("raDeg"))
    dec_deg = _float(tic_metadata.get("decDeg"))
    if ra_deg is None or dec_deg is None:
        raise RuntimeError("v20.11 requires TIC RA/Dec from the identity stage.")

    target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    sectors = _sector_candidates(
        primary_sector=primary_sector,
        independent_spec=independent_spec,
        signal_sectors=signal_sectors,
    )
    if not sectors:
        raise RuntimeError("No frozen sectors overlap the v20.9 signal-sector set.")

    root = Path(output_dir) / "residual-mode-localization-review"
    root.mkdir(parents=True, exist_ok=True)
    search = _frequency_search(reference_frequency)
    physical_frequency = 1.0 / float(physical_period_days)
    residual_period = 1.0 / float(reference_frequency)
    source_base_id = str(source_dataset_entry.get("id") or f"tic-{tic_id}")

    static_by_sector = {
        int(item["sector"]): item
        for item in residual_localization_summary.get("sectorResults") or []
        if _int(item.get("sector")) is not None
    }

    dataset_entries: list[dict[str, Any]] = []
    prepared_pixels: list[dict[str, Any]] = []
    window_metadata: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for sector_index, (sector, role) in enumerate(sectors, start=1):
        print(
            f"   Sector {sector} ({sector_index}/{len(sectors)}): preparing time-resolved residual localization",
            flush=True,
        )
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
            corrected, _ = _background_subtract_cube(cube)
            residual_cube, valid_pixels = _prewhiten_cube(
                absolute_times=absolute_times,
                cube=corrected,
                physical_frequency=physical_frequency,
            )

            target_x, target_y = tpf.wcs.world_to_pixel(target)
            pixel_scale = _pixel_scale_arcsec(tpf.wcs)
            jacobian = _local_sky_jacobian(tpf.wcs, target, float(target_x), float(target_y))
            rows, cols = valid_pixels.shape
            windows = _window_bounds(absolute_times, residual_period_days=residual_period)
            if not windows:
                raise RuntimeError("Could not form usable time windows.")

            for window_index, start_day, end_day, midpoint_day in windows:
                if window_index == len(windows):
                    mask = (absolute_times >= start_day) & (absolute_times <= end_day)
                else:
                    mask = (absolute_times >= start_day) & (absolute_times < end_day)
                sample_count = int(np.sum(mask))
                if sample_count < MIN_WINDOW_CADENCES:
                    errors.append(
                        {
                            "sector": int(sector),
                            "role": role,
                            "stage": "window-selection",
                            "windowIndex": int(window_index),
                            "windowStartDays": float(start_day),
                            "windowEndDays": float(end_day),
                            "sampleCount": sample_count,
                            "minimumSampleCount": MIN_WINDOW_CADENCES,
                            "error": "insufficient-window-cadences",
                        }
                    )
                    continue

                window_times = absolute_times[mask]
                window_cube = residual_cube[mask]
                relative_times = window_times - float(time_reference)
                warped = _time_warp(relative_times, float(q))
                local_times = warped - float(np.min(warped))

                window_key = f"s{sector}-w{window_index}"
                prepared_count = 0
                for row in range(rows):
                    for col in range(cols):
                        if not bool(valid_pixels[row, col]):
                            continue
                        pixel_flux = np.asarray(window_cube[:, row, col], dtype=np.float64)
                        if not np.all(np.isfinite(pixel_flux)):
                            continue
                        std = float(np.std(pixel_flux))
                        if not math.isfinite(std) or std <= 1e-12:
                            continue
                        pixel_flux = (pixel_flux - float(np.mean(pixel_flux))) / std

                        dataset_id = (
                            f"{source_base_id}-residual-review-sector-{sector}-"
                            f"w{window_index}-r{row:02d}-c{col:02d}-v1"
                        )
                        target_name = (
                            f"{source_dataset_entry.get('targetName') or source_base_id} "
                            f"residual review sector {sector} window {window_index} pixel ({row},{col})"
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
                                "role": "residual-mode-time-resolved-pixel-localization",
                                "sector": int(sector),
                                "windowIndex": int(window_index),
                                "pixelRow": int(row),
                                "pixelColumn": int(col),
                                "referenceFrequency": float(reference_frequency),
                                "fractionalFrequencyDriftPerDay": float(q),
                            },
                            "source": {
                                "mission": "TESS",
                                "sector": int(sector),
                                "distributedSamples": int(len(local_times)),
                                "baselineDays": float(np.max(window_times) - np.min(window_times)),
                                "windowStartDays": float(start_day),
                                "windowEndDays": float(end_day),
                                "windowMidpointDays": float(midpoint_day),
                                "timeReferenceDays": float(time_reference),
                                "sourceType": source.get("sourceType"),
                                "author": source.get("author"),
                                "cadenceSeconds": source.get("cadenceSeconds"),
                            },
                        }
                        _write_json(output_path, dataset)
                        dataset_entries.append(
                            {
                                "id": dataset_id,
                                "path": str(output_path.resolve()),
                                "targetName": target_name,
                            }
                        )
                        prepared_pixels.append(
                            {
                                "datasetID": dataset_id,
                                "datasetPath": str(output_path.resolve()),
                                "windowKey": window_key,
                                "sector": int(sector),
                                "role": role,
                                "windowIndex": int(window_index),
                                "row": int(row),
                                "column": int(col),
                            }
                        )
                        prepared_count += 1

                if prepared_count:
                    window_metadata.append(
                        {
                            "windowKey": window_key,
                            "sector": int(sector),
                            "role": role,
                            "windowIndex": int(window_index),
                            "windowStartDays": float(start_day),
                            "windowEndDays": float(end_day),
                            "windowMidpointDays": float(midpoint_day),
                            "sampleCount": sample_count,
                            "shape": [int(rows), int(cols)],
                            "targetPixel": {"x": float(target_x), "y": float(target_y)},
                            "pixelScaleArcsec": pixel_scale,
                            "skyJacobian": jacobian,
                            "preparedPixelCount": int(prepared_count),
                            "source": source,
                            "v20_10StaticClassification": (
                                static_by_sector.get(int(sector), {}).get("classification")
                            ),
                            "v20_10StaticOffsetPixels": (
                                static_by_sector.get(int(sector), {}).get("offsetPixels")
                            ),
                        }
                    )
                    print(
                        f"      window {window_index}: samples={sample_count}, pixel datasets={prepared_count}",
                        flush=True,
                    )
        except Exception as exc:
            errors.append(
                {
                    "sector": int(sector),
                    "role": role,
                    "stage": "sector-preparation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"      unavailable: {type(exc).__name__}: {exc}", flush=True)

    if not dataset_entries:
        raise RuntimeError("v20.11 could not prepare any time-resolved residual pixel datasets.")

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "residual-mode-source-localization-review-v1"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — time-resolved residual localization review",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "residual-mode-source-localization-review",
            "workerSemantics": (
                "Each dataset is one time-windowed, prewhitened TESS pixel light curve after the "
                "v20.9 deterministic drift time warp. Workers execute ordinary Lomb-Scargle only."
            ),
            "referenceFrequency": float(reference_frequency),
            "fractionalFrequencyDriftPerDay": float(q),
            "signalSectors": signal_sectors,
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
        "workerSemantics": "generic-lomb-scargle-on-time-windowed-drift-corrected-pixel-light-curves",
        "ticID": int(tic_id),
        "targetSky": {"raDeg": float(ra_deg), "decDeg": float(dec_deg)},
        "physicalPeriodDays": float(physical_period_days),
        "residualFrequencyAtReference": float(reference_frequency),
        "residualPeriodAtReferenceDays": float(residual_period),
        "fractionalFrequencyDriftPerDay": float(q),
        "timeReferenceDays": float(time_reference),
        "signalSectors": signal_sectors,
        "frequencySearch": search,
        "windowLengthTargetDays": WINDOW_LENGTH_DAYS,
        "windowMetadata": window_metadata,
        "preparedPixels": prepared_pixels,
        "errors": errors,
        "workUnitsPerDataset": work_units_per_dataset,
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
    }


def _sky_scatter(points: list[tuple[float, float]]) -> float | None:
    if not points:
        return None
    med_e = statistics.median(value[0] for value in points)
    med_n = statistics.median(value[1] for value in points)
    return statistics.median(
        math.hypot(value[0] - med_e, value[1] - med_n)
        for value in points
    )


def _sector_temporal_summary(window_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sector: dict[int, list[dict[str, Any]]] = {}
    for item in window_results:
        by_sector.setdefault(int(item["sector"]), []).append(item)

    summaries: list[dict[str, Any]] = []
    for sector in sorted(by_sector):
        items = sorted(by_sector[sector], key=lambda item: int(item.get("windowIndex") or 0))
        quality = [item for item in items if item.get("localizationQualityPass")]
        target = [item for item in quality if item.get("classification") == "TARGET_CONSISTENT"]
        off = [item for item in quality if item.get("classification") == "OFF_TARGET"]
        ambiguous = [item for item in items if item.get("classification") == "AMBIGUOUS"]

        off_points = [
            (float(item["skyOffsetEastArcsec"]), float(item["skyOffsetNorthArcsec"]))
            for item in off
            if _float(item.get("skyOffsetEastArcsec")) is not None
            and _float(item.get("skyOffsetNorthArcsec")) is not None
        ]
        off_scatter = _sky_scatter(off_points)

        if target and off:
            classification = "SOURCE_SWITCHING"
        elif len(target) >= 2 and len(target) > len(off):
            classification = "TARGET_DOMINANT"
        elif (
            len(off) >= 2
            and len(off) > len(target)
            and off_scatter is not None
            and off_scatter <= MAX_OFF_TARGET_SKY_SCATTER_ARCSEC
        ):
            classification = "OFF_TARGET_DOMINANT"
        elif quality:
            classification = "MIXED_OR_INSUFFICIENT"
        else:
            classification = "NO_QUALITY_WINDOWS"

        summaries.append(
            {
                "sector": int(sector),
                "role": items[0].get("role"),
                "classification": classification,
                "qualityWindowCount": len(quality),
                "targetWindowCount": len(target),
                "offTargetWindowCount": len(off),
                "ambiguousWindowCount": len(ambiguous),
                "offTargetSkyScatterArcsec": off_scatter,
                "windowClassifications": [
                    {
                        "windowIndex": item.get("windowIndex"),
                        "classification": item.get("classification"),
                        "offsetPixels": item.get("offsetPixels"),
                        "skySeparationArcsec": item.get("skySeparationArcsec"),
                    }
                    for item in items
                ],
            }
        )
    return summaries


def _cross_time_summary(
    *,
    window_results: list[dict[str, Any]],
    sector_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    independent_sectors = [item for item in sector_summaries if item.get("role") == "independent"]
    eligible = len(independent_sectors)
    required = max(
        MIN_INDEPENDENT_SECTORS_FOR_RESOLUTION,
        eligible // 2 + 1,
    ) if eligible else MIN_INDEPENDENT_SECTORS_FOR_RESOLUTION

    target_sectors = [
        int(item["sector"])
        for item in independent_sectors
        if item.get("classification") == "TARGET_DOMINANT"
    ]
    off_sectors = [
        int(item["sector"])
        for item in independent_sectors
        if item.get("classification") == "OFF_TARGET_DOMINANT"
    ]
    switching_sectors = [
        int(item["sector"])
        for item in independent_sectors
        if item.get("classification") == "SOURCE_SWITCHING"
    ]

    off_points = [
        (float(item["skyOffsetEastArcsec"]), float(item["skyOffsetNorthArcsec"]))
        for item in window_results
        if item.get("role") == "independent"
        and item.get("classification") == "OFF_TARGET"
        and item.get("localizationQualityPass")
        and _float(item.get("skyOffsetEastArcsec")) is not None
        and _float(item.get("skyOffsetNorthArcsec")) is not None
    ]
    off_scatter = _sky_scatter(off_points)

    has_independent_target_window = any(
        item.get("role") == "independent"
        and item.get("classification") == "TARGET_CONSISTENT"
        and item.get("localizationQualityPass")
        for item in window_results
    )
    has_independent_off_window = any(
        item.get("role") == "independent"
        and item.get("classification") == "OFF_TARGET"
        and item.get("localizationQualityPass")
        for item in window_results
    )

    if eligible >= MIN_INDEPENDENT_SECTORS_FOR_RESOLUTION and len(target_sectors) >= required:
        classification = "RESIDUAL_MODE_TARGET_SUPPORTED_TIME_RESOLVED"
        origin = "TARGET_CONSISTENT"
        next_test = "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
    elif (
        eligible >= MIN_INDEPENDENT_SECTORS_FOR_RESOLUTION
        and len(off_sectors) >= required
        and off_scatter is not None
        and off_scatter <= MAX_OFF_TARGET_SKY_SCATTER_ARCSEC
    ):
        classification = "RESIDUAL_MODE_OFF_TARGET_SUPPORTED_TIME_RESOLVED"
        origin = "OFF_TARGET"
        next_test = "IDENTIFY_OFFSET_RESIDUAL_VARIABLE_SOURCE"
    elif len(switching_sectors) >= 1 or (
        has_independent_target_window and has_independent_off_window
    ):
        classification = "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND"
        origin = "TIME_VARIABLE_OR_BLENDED"
        next_test = "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"
    else:
        classification = "RESIDUAL_MODE_TIME_RESOLVED_LOCALIZATION_UNRESOLVED"
        origin = "UNRESOLVED"
        next_test = "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW"

    return {
        "classification": classification,
        "residualModeOrigin": origin,
        "independentEligibleSectorCount": eligible,
        "requiredIndependentSupportCount": required,
        "targetDominantSectors": sorted(target_sectors),
        "offTargetDominantSectors": sorted(off_sectors),
        "sourceSwitchingSectors": sorted(switching_sectors),
        "offTargetSkyOffsetScatterArcsec": off_scatter,
        "maximumOffTargetSkyOffsetScatterArcsec": MAX_OFF_TARGET_SKY_SCATTER_ARCSEC,
        "recommendedNextTest": next_test,
    }


def interpret_residual_mode_localization_review_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedPixels") or []
    }
    windows = {
        str(item.get("windowKey")): item
        for item in preparation.get("windowMetadata") or []
    }
    by_window: dict[str, list[dict[str, Any]]] = {}

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
        by_window.setdefault(str(meta["windowKey"]), []).append(result)

    window_results: list[dict[str, Any]] = []
    for window_key in sorted(
        windows,
        key=lambda key: (
            int(windows[key].get("sector") or 0),
            int(windows[key].get("windowIndex") or 0),
        ),
    ):
        meta = windows[window_key]
        shape = meta.get("shape") or []
        if len(shape) != 2:
            continue
        rows, cols = int(shape[0]), int(shape[1])
        power_map = np.zeros((rows, cols), dtype=np.float64)
        frequency_map = np.full((rows, cols), np.nan, dtype=np.float64)
        for item in by_window.get(window_key, []):
            row = int(item["row"])
            col = int(item["column"])
            if 0 <= row < rows and 0 <= col < cols:
                power = item.get("candidatePower")
                frequency = item.get("candidateFrequency")
                if power is not None:
                    power_map[row, col] = max(0.0, float(power))
                if frequency is not None:
                    frequency_map[row, col] = float(frequency)

        target_pixel = meta.get("targetPixel") or {}
        localization = _localize_power_map(
            power_map=power_map,
            target_x=float(target_pixel.get("x")),
            target_y=float(target_pixel.get("y")),
            pixel_scale_arcsec=_float(meta.get("pixelScaleArcsec")),
            jacobian=meta.get("skyJacobian") or {},
        )
        window_results.append(
            {
                "windowKey": window_key,
                "sector": int(meta["sector"]),
                "role": meta.get("role"),
                "windowIndex": int(meta["windowIndex"]),
                "windowStartDays": meta.get("windowStartDays"),
                "windowEndDays": meta.get("windowEndDays"),
                "windowMidpointDays": meta.get("windowMidpointDays"),
                "sampleCount": meta.get("sampleCount"),
                "shape": [rows, cols],
                "targetPixel": target_pixel,
                "v20_10StaticClassification": meta.get("v20_10StaticClassification"),
                "v20_10StaticOffsetPixels": meta.get("v20_10StaticOffsetPixels"),
                **localization,
                "candidatePowerMap": power_map.tolist(),
                "candidateFrequencyMap": [
                    [float(value) if math.isfinite(float(value)) else None for value in row]
                    for row in frequency_map
                ],
            }
        )

    sector_summaries = _sector_temporal_summary(window_results)
    cross = _cross_time_summary(
        window_results=window_results,
        sector_summaries=sector_summaries,
    )
    return {
        "version": "openstar.tess-residual-mode-source-localization-review.v1",
        "distributedLocalizationReview": {
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
        "windowResults": window_results,
        "sectorTemporalSummaries": sector_summaries,
        "crossTime": cross,
        "errors": preparation.get("errors") or [],
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": cross.get("recommendedNextTest"),
        "interpretationGuard": (
            "This time-resolves only the v20.9 drifting residual component. "
            "It does not alter v20.6's target association for the established 13.72-day family."
        ),
    }
