from __future__ import annotations

import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WINDOW_LENGTH_DAYS = 12.0
WINDOWS_PER_SECTOR = 3
MIN_WINDOW_SAMPLES = 256
MIN_WINDOW_SPAN_FRACTION = 0.70
MINIMUM_FREQUENCY = 0.10
MAXIMUM_FREQUENCY = 0.50
TOTAL_FREQUENCIES = 131_072
FREQUENCIES_PER_WORK_UNIT = 2_048
MIN_PEAK_PROMINENCE = 1.5
MIN_OBSERVED_CYCLES = 1.5
BOUNDARY_FRACTION = 0.004
FREQUENCY_CLUSTER_RELATIVE_TOLERANCE = 0.08
FAMILY_NEAR_RELATIVE_TOLERANCE = 0.25
MIN_STABLE_CLUSTER_WINDOWS = 4
MIN_STABLE_CLUSTER_INDEPENDENT_SECTORS = 3
MIN_DRIFT_WINDOWS = 4
MIN_DRIFT_INDEPENDENT_SECTORS = 3
MIN_DRIFT_R_SQUARED = 0.60
MIN_DRIFT_FRACTION = 0.08
MIN_TRANSIENT_CLUSTER_WINDOWS = 2
FAMILY_AMPLITUDE_EVOLUTION_THRESHOLD = 0.50
FAMILY_PHASE_CONCENTRATION_THRESHOLD = 0.55
FAMILY_PHASE_DRIFT_TURNS_THRESHOLD = 0.25
FAMILY_PHASE_DRIFT_R_SQUARED_THRESHOLD = 0.45


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


def _dataset_arrays(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(dataset.get("times") or [], dtype=np.float64)
    flux = np.asarray(dataset.get("flux") or [], dtype=np.float64)
    if len(times) != len(flux):
        raise RuntimeError("Dataset times/flux lengths do not match.")
    finite = np.isfinite(times) & np.isfinite(flux)
    times = times[finite]
    flux = flux[finite]
    if len(times) < MIN_WINDOW_SAMPLES:
        raise RuntimeError("Dataset has too few finite samples for time-frequency analysis.")
    order = np.argsort(times)
    return times[order], flux[order]


def _source_time_origin(dataset: dict[str, Any]) -> float | None:
    source = dataset.get("source") or {}
    for key in ("originalTimeOriginDays", "timeOriginDays"):
        value = _float(source.get(key))
        if value is not None:
            return value
    return None


def _baseline(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    return float(np.max(times) - np.min(times))


def _frequency_search() -> dict[str, Any]:
    step = (MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY) / (TOTAL_FREQUENCIES - 1)
    return {
        "minimumFrequency": MINIMUM_FREQUENCY,
        "maximumFrequency": MAXIMUM_FREQUENCY,
        "frequencyStep": step,
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def _design_matrix(times: np.ndarray, frequencies: list[float]) -> np.ndarray:
    columns = [np.ones(len(times), dtype=np.float64)]
    for frequency in frequencies:
        omega_t = 2.0 * math.pi * float(frequency) * times
        columns.append(np.sin(omega_t))
        columns.append(np.cos(omega_t))
    return np.column_stack(columns)


def _fit_family(
    *,
    phase_times: np.ndarray,
    flux: np.ndarray,
    physical_frequency: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    frequencies = [physical_frequency, 2.0 * physical_frequency]
    design = _design_matrix(phase_times, frequencies)
    coefficients, _, _, _ = np.linalg.lstsq(design, flux, rcond=None)
    model = design @ coefficients
    residual = flux - model

    total_ss = float(np.sum(np.square(flux - np.mean(flux))))
    residual_ss = float(np.sum(np.square(residual)))
    explained = None
    if total_ss > 1e-18:
        explained = max(0.0, min(1.0, 1.0 - residual_ss / total_ss))

    a1, b1 = float(coefficients[1]), float(coefficients[2])
    a2, b2 = float(coefficients[3]), float(coefficients[4])
    amp1 = math.hypot(a1, b1)
    amp2 = math.hypot(a2, b2)
    phase1 = math.atan2(b1, a1)
    phase2 = math.atan2(b2, a2)
    relative_phase = math.atan2(
        math.sin(phase2 - 2.0 * phase1),
        math.cos(phase2 - 2.0 * phase1),
    )

    residual_std = float(np.std(residual))
    normalized = residual.copy()
    if math.isfinite(residual_std) and residual_std > 1e-12:
        normalized = residual / residual_std

    return normalized, {
        "physicalFrequency": physical_frequency,
        "firstHarmonicFrequency": 2.0 * physical_frequency,
        "fundamentalAmplitude": amp1,
        "firstHarmonicAmplitude": amp2,
        "firstHarmonicToFundamentalAmplitudeRatio": (
            amp2 / amp1 if amp1 > 1e-12 else None
        ),
        "fundamentalPhaseRad": phase1,
        "firstHarmonicPhaseRad": phase2,
        "translationInvariantRelativeHarmonicPhaseRad": relative_phase,
        "explainedVariance": explained,
        "residualRMS": float(np.sqrt(np.mean(np.square(residual)))),
        "residualStdDevBeforeNormalization": residual_std,
    }


def _window_bounds(times: np.ndarray) -> list[tuple[float, float]]:
    minimum = float(np.min(times))
    maximum = float(np.max(times))
    baseline = maximum - minimum
    if baseline <= 0:
        return []
    if baseline <= WINDOW_LENGTH_DAYS * 1.10:
        return [(minimum, maximum)]

    last_start = maximum - WINDOW_LENGTH_DAYS
    starts = np.linspace(minimum, last_start, WINDOWS_PER_SECTOR)
    bounds: list[tuple[float, float]] = []
    for start in starts:
        end = float(start + WINDOW_LENGTH_DAYS)
        pair = (float(start), min(end, maximum))
        if any(abs(pair[0] - existing[0]) < 1e-6 for existing in bounds):
            continue
        bounds.append(pair)
    return bounds


def _source_items(
    *,
    primary_dataset_path: str | Path,
    primary_sector: int | None,
    independent_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [{
        "sectorKey": str(primary_sector if primary_sector is not None else "primary"),
        "sector": _int(primary_sector),
        "datasetPath": str(Path(primary_dataset_path).expanduser().resolve()),
        "role": "primary-time-frequency-window",
    }]
    for item in independent_spec.get("preparedSectors") or []:
        sector = _int(item.get("sector"))
        path = item.get("datasetPath")
        if sector is None or not path:
            continue
        items.append({
            "sectorKey": str(sector),
            "sector": sector,
            "datasetPath": str(Path(path).expanduser().resolve()),
            "role": "independent-time-frequency-window",
        })
    return items


def build_time_frequency_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    primary_dataset_path: str | Path,
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if physical_period_days <= 0:
        raise ValueError("physical_period_days must be positive")

    source_project = _load_json(source_project_path)
    physical_frequency = 1.0 / float(physical_period_days)
    root = Path(output_dir) / "time-frequency"
    root.mkdir(parents=True, exist_ok=True)

    source_items = _source_items(
        primary_dataset_path=primary_dataset_path,
        primary_sector=primary_sector,
        independent_spec=independent_spec,
    )

    origins: list[float] = []
    loaded: list[tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]] = []
    for item in source_items:
        dataset = _load_json(item["datasetPath"])
        times, flux = _dataset_arrays(dataset)
        origin = _source_time_origin(dataset)
        if origin is not None:
            origins.append(float(origin + np.min(times)))
        loaded.append((item, dataset, times, flux))

    absolute_time_reference = min(origins) if origins else None
    prepared_windows: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    family_track: list[dict[str, Any]] = []

    # Track the established family on each full sector. The ~25-day sector
    # baselines cover substantially more of the ~13.7-day physical cycle than
    # the shorter residual-search windows, so amplitude/phase evolution is not
    # inferred from sub-cycle windows.
    for source_item, source_dataset, times, flux in loaded:
        origin = _source_time_origin(source_dataset)
        if origin is not None and absolute_time_reference is not None:
            phase_times = times + origin - absolute_time_reference
            absolute_center = float(np.median(times + origin))
        else:
            phase_times = times - np.min(times)
            absolute_center = None
        _, fit = _fit_family(
            phase_times=phase_times,
            flux=flux,
            physical_frequency=physical_frequency,
        )
        family_track.append({
            "sectorKey": source_item["sectorKey"],
            "sector": source_item["sector"],
            "role": source_item["role"],
            "absoluteWindowCenterDays": absolute_center,
            "baselineDays": _baseline(times),
            "sampleCount": int(len(times)),
            "familyFit": fit,
        })

    for source_item, source_dataset, times, flux in loaded:
        sector = source_item["sector"]
        sector_key = source_item["sectorKey"]
        origin = _source_time_origin(source_dataset)
        for index, (start, end) in enumerate(_window_bounds(times), start=1):
            # Include the right edge only for the final window to avoid a trivial
            # duplicate sample at an overlap boundary.
            if index == len(_window_bounds(times)):
                mask = (times >= start) & (times <= end)
            else:
                mask = (times >= start) & (times < end)
            window_times = times[mask]
            window_flux = flux[mask]
            if len(window_times) < MIN_WINDOW_SAMPLES:
                continue
            span = _baseline(window_times)
            nominal_span = max(end - start, 1e-12)
            if span < nominal_span * MIN_WINDOW_SPAN_FRACTION:
                continue

            if origin is not None and absolute_time_reference is not None:
                phase_times = window_times + origin - absolute_time_reference
                absolute_center = float(np.median(window_times + origin))
                absolute_origin = float(origin + np.min(window_times))
            else:
                phase_times = window_times - np.min(window_times)
                absolute_center = None
                absolute_origin = None

            residual, family_fit = _fit_family(
                phase_times=phase_times,
                flux=window_flux,
                physical_frequency=physical_frequency,
            )
            local_times = window_times - np.min(window_times)
            base_id = str(source_dataset.get("id") or source_dataset_entry["id"])
            dataset_id = f"{base_id}-tf-window-{index}-v1"
            target_name = (
                f"{source_dataset_entry.get('targetName') or source_dataset_entry['id']} "
                f"sector {sector_key} time-frequency window {index}"
            )
            output_path = root / f"{_safe(dataset_id)}.json"

            residual_dataset = copy.deepcopy(source_dataset)
            residual_dataset["id"] = dataset_id
            residual_dataset["targetName"] = target_name
            residual_dataset["times"] = np.asarray(local_times, dtype=np.float32).tolist()
            residual_dataset["flux"] = np.asarray(residual, dtype=np.float32).tolist()
            residual_dataset["frequencySearch"] = _frequency_search()
            residual_dataset["reference"] = {}
            science = dict(residual_dataset.get("science") or {})
            science.update({
                "role": source_item["role"],
                "purpose": "sliding-window-time-frequency-evolution",
                "sectorKey": sector_key,
                "windowIndex": index,
                "physicalFundamentalFrequency": physical_frequency,
                "firstHarmonicFrequency": 2.0 * physical_frequency,
            })
            residual_dataset["science"] = science
            source = dict(residual_dataset.get("source") or {})
            source.update({
                "sector": sector,
                "baselineDays": span,
                "distributedSamples": int(len(local_times)),
                "timeFrequencyWindowIndex": index,
                "windowStartDatasetDays": float(np.min(window_times)),
                "windowEndDatasetDays": float(np.max(window_times)),
                "windowCenterDatasetDays": float(np.median(window_times)),
                "absoluteWindowCenterDays": absolute_center,
                "familyFit": family_fit,
            })
            if absolute_origin is not None:
                source["originalTimeOriginDays"] = absolute_origin
            residual_dataset["source"] = source
            _write_json(output_path, residual_dataset)

            meta = {
                "datasetID": dataset_id,
                "datasetPath": str(output_path.resolve()),
                "sectorKey": sector_key,
                "sector": sector,
                "role": source_item["role"],
                "windowIndex": index,
                "sampleCount": int(len(local_times)),
                "baselineDays": span,
                "windowStartDatasetDays": float(np.min(window_times)),
                "windowEndDatasetDays": float(np.max(window_times)),
                "windowCenterDatasetDays": float(np.median(window_times)),
                "absoluteWindowCenterDays": absolute_center,
                "familyFit": family_fit,
            }
            prepared_windows.append(meta)

            entry = copy.deepcopy(source_dataset_entry)
            entry.update({
                "id": dataset_id,
                "path": meta["datasetPath"],
                "targetName": target_name,
                "sector": sector,
                "role": source_item["role"],
            })
            dataset_entries.append(entry)

    if not dataset_entries:
        raise RuntimeError("No usable sliding windows were produced for time-frequency analysis.")

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "time-frequency-evolution-v1"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — time-frequency evolution",
        "workloadID": source_project["workloadID"],
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry["id"],
            "purpose": "sliding-window-time-frequency-evolution",
            "physicalPeriodDays": float(physical_period_days),
            "physicalFrequency": physical_frequency,
            "windowLengthDays": WINDOW_LENGTH_DAYS,
            "windowsPerSector": WINDOWS_PER_SECTOR,
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": True,
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "physicalPeriodDays": float(physical_period_days),
        "physicalFrequency": physical_frequency,
        "firstHarmonicFrequency": 2.0 * physical_frequency,
        "absoluteTimeReferenceDays": absolute_time_reference,
        "frequencySearch": _frequency_search(),
        "windowLengthDays": WINDOW_LENGTH_DAYS,
        "windowsPerSector": WINDOWS_PER_SECTOR,
        "preparedWindows": prepared_windows,
        "familyTrack": family_track,
        "totalWorkUnits": len(dataset_entries) * work_units_per_dataset,
        "workUnitsPerDataset": work_units_per_dataset,
    }


def _candidate_boundary_hit(candidate_frequency: float | None, search: dict[str, Any]) -> bool:
    if candidate_frequency is None:
        return False
    minimum = _float(search.get("minimumFrequency"))
    maximum = _float(search.get("maximumFrequency"))
    step = _float(search.get("frequencyStep")) or 0.0
    if minimum is None or maximum is None or maximum <= minimum:
        return False
    guard = max(step * 4.0, (maximum - minimum) * BOUNDARY_FRACTION, 1e-12)
    return candidate_frequency <= minimum + guard or candidate_frequency >= maximum - guard


def interpret_time_frequency_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedWindows") or []
    }
    search = preparation.get("frequencySearch") or {}
    physical_frequency = float(preparation["physicalFrequency"])
    harmonic_frequency = 2.0 * physical_frequency
    results: list[dict[str, Any]] = []

    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id) or {}
        frequency = _float(dataset.get("candidateFrequency"))
        period = _float(dataset.get("candidatePeriodDays"))
        prominence = _float(dataset.get("candidatePeakProminenceRatio"))
        status = str(dataset.get("periodStatus") or "").upper()
        confidence = str(dataset.get("periodConfidence") or "none").lower()
        baseline = _float(meta.get("baselineDays")) or 0.0
        observed_cycles = (baseline / period) if period and period > 0 else 0.0
        reliable = status == "RELIABLE" and confidence in {"high", "medium"}
        prominence_ok = prominence is not None and prominence >= MIN_PEAK_PROMINENCE
        coverage_ok = observed_cycles >= MIN_OBSERVED_CYCLES
        boundary_hit = _candidate_boundary_hit(frequency, search)

        near_family = False
        nearest_family_frequency = None
        family_relative_separation = None
        rayleigh = (1.0 / baseline) if baseline > 0 else None
        if frequency is not None:
            family_candidates = [physical_frequency, harmonic_frequency]
            nearest_family_frequency = min(
                family_candidates,
                key=lambda value: abs(frequency - value),
            )
            family_relative_separation = (
                abs(frequency - nearest_family_frequency)
                / max(abs(nearest_family_frequency), 1e-12)
            )
            near_family = family_relative_separation <= FAMILY_NEAR_RELATIVE_TOLERANCE

        accepted = bool(
            reliable
            and prominence_ok
            and coverage_ok
            and not boundary_hit
            and frequency is not None
            and period is not None
        )

        results.append({
            "datasetID": dataset_id,
            "sectorKey": meta.get("sectorKey"),
            "sector": meta.get("sector"),
            "role": meta.get("role"),
            "windowIndex": meta.get("windowIndex"),
            "windowCenterDatasetDays": meta.get("windowCenterDatasetDays"),
            "absoluteWindowCenterDays": meta.get("absoluteWindowCenterDays"),
            "baselineDays": baseline,
            "periodStatus": status,
            "periodConfidence": confidence,
            "candidateFrequency": frequency,
            "candidatePeriodDays": period,
            "candidatePower": _float(dataset.get("candidatePower")),
            "candidatePeakProminenceRatio": prominence,
            "observedCycles": observed_cycles,
            "reliable": reliable,
            "prominenceOK": prominence_ok,
            "coverageOK": coverage_ok,
            "boundaryHit": boundary_hit,
            "acceptedTimeFrequencyFeature": accepted,
            "rayleighFrequency": rayleigh,
            "nearestEstablishedFamilyFrequency": nearest_family_frequency,
            "relativeSeparationFromEstablishedFamily": family_relative_separation,
            "nearEstablishedFamily": near_family,
            "familyFit": meta.get("familyFit") or {},
        })

    return {
        "physicalFrequency": physical_frequency,
        "firstHarmonicFrequency": harmonic_frequency,
        "windowResults": results,
        "acceptedFeatureCount": sum(1 for item in results if item["acceptedTimeFrequencyFeature"]),
        "acceptedNearFamilyCount": sum(
            1 for item in results
            if item["acceptedTimeFrequencyFeature"] and item["nearEstablishedFamily"]
        ),
        "familyTrack": preparation.get("familyTrack") or [],
        "minimumPeakProminenceRatio": MIN_PEAK_PROMINENCE,
        "minimumObservedCycles": MIN_OBSERVED_CYCLES,
    }


def _cluster_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    for point in sorted(points, key=lambda item: float(item["candidateFrequency"])):
        frequency = float(point["candidateFrequency"])
        placed = False
        for cluster in clusters:
            center = float(np.median([float(item["candidateFrequency"]) for item in cluster]))
            if abs(frequency - center) / max(abs(center), 1e-12) <= FREQUENCY_CLUSTER_RELATIVE_TOLERANCE:
                cluster.append(point)
                placed = True
                break
        if not placed:
            clusters.append([point])

    summaries: list[dict[str, Any]] = []
    for cluster in clusters:
        frequencies = np.asarray([float(item["candidateFrequency"]) for item in cluster], dtype=np.float64)
        independent_sectors = sorted({
            int(item["sector"])
            for item in cluster
            if item.get("sector") is not None
            and item.get("role") == "independent-time-frequency-window"
        })
        median_frequency = float(np.median(frequencies))
        summaries.append({
            "medianFrequency": median_frequency,
            "medianPeriodDays": 1.0 / median_frequency,
            "windowCount": len(cluster),
            "independentSectors": independent_sectors,
            "independentSectorCount": len(independent_sectors),
            "relativeSpan": (
                float((np.max(frequencies) - np.min(frequencies)) / median_frequency)
                if median_frequency > 0 and len(frequencies) > 1
                else 0.0
            ),
            "nearEstablishedFamilyFraction": (
                sum(1 for item in cluster if item.get("nearEstablishedFamily")) / len(cluster)
            ),
            "members": [
                {
                    "sector": item.get("sector"),
                    "windowIndex": item.get("windowIndex"),
                    "absoluteWindowCenterDays": item.get("absoluteWindowCenterDays"),
                    "frequency": item.get("candidateFrequency"),
                    "periodDays": item.get("candidatePeriodDays"),
                    "prominence": item.get("candidatePeakProminenceRatio"),
                    "nearEstablishedFamily": item.get("nearEstablishedFamily"),
                }
                for item in sorted(
                    cluster,
                    key=lambda value: (
                        value.get("absoluteWindowCenterDays") is None,
                        value.get("absoluteWindowCenterDays") or 0.0,
                        value.get("sector") or 0,
                        value.get("windowIndex") or 0,
                    ),
                )
            ],
        })
    summaries.sort(
        key=lambda item: (item["independentSectorCount"], item["windowCount"]),
        reverse=True,
    )
    return summaries


def _linear_track(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [
        item for item in points
        if _float(item.get("absoluteWindowCenterDays")) is not None
        and _float(item.get("candidateFrequency")) is not None
    ]
    if len(usable) < 3:
        return None
    x = np.asarray([float(item["absoluteWindowCenterDays"]) for item in usable], dtype=np.float64)
    y = np.asarray([float(item["candidateFrequency"]) for item in usable], dtype=np.float64)
    x0 = float(np.mean(x))
    centered = x - x0
    design = np.column_stack([np.ones(len(x)), centered])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    total_ss = float(np.sum(np.square(y - np.mean(y))))
    residual_ss = float(np.sum(np.square(y - predicted)))
    r_squared = 1.0 if total_ss <= 1e-18 else max(0.0, min(1.0, 1.0 - residual_ss / total_ss))
    span_days = float(np.max(x) - np.min(x))
    slope = float(coefficients[1])
    median_frequency = float(np.median(y))
    drift_fraction = (
        abs(slope) * span_days / max(abs(median_frequency), 1e-12)
        if span_days > 0
        else 0.0
    )
    return {
        "sampleCount": len(usable),
        "slopeCyclesPerDayPerDay": slope,
        "interceptFrequencyAtMeanTime": float(coefficients[0]),
        "meanTimeDays": x0,
        "timeSpanDays": span_days,
        "rSquared": r_squared,
        "fractionalFrequencyChangeAcrossSpan": drift_fraction,
    }


def _circular_concentration(phases: list[float], weights: list[float] | None = None) -> float | None:
    if not phases:
        return None
    z = np.exp(1j * np.asarray(phases, dtype=np.float64))
    if weights is not None and len(weights) == len(phases):
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        if float(np.sum(w)) > 0:
            return float(abs(np.sum(w * z) / np.sum(w)))
    return float(abs(np.mean(z)))


def _phase_drift(points: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    usable = []
    for item in points:
        center = _float(item.get("absoluteWindowCenterDays"))
        fit = item.get("familyFit") or {}
        phase = _float(fit.get(key))
        amplitude = _float(
            fit.get("fundamentalAmplitude")
            if key == "fundamentalPhaseRad"
            else fit.get("firstHarmonicAmplitude")
        )
        if center is not None and phase is not None:
            usable.append((center, phase, amplitude or 0.0))
    if len(usable) < 3:
        return None
    usable.sort(key=lambda item: item[0])
    x = np.asarray([item[0] for item in usable], dtype=np.float64)
    phase = np.unwrap(np.asarray([item[1] for item in usable], dtype=np.float64))
    x0 = float(np.mean(x))
    centered = x - x0
    design = np.column_stack([np.ones(len(x)), centered])
    coefficients, _, _, _ = np.linalg.lstsq(design, phase, rcond=None)
    predicted = design @ coefficients
    total_ss = float(np.sum(np.square(phase - np.mean(phase))))
    residual_ss = float(np.sum(np.square(phase - predicted)))
    r_squared = 1.0 if total_ss <= 1e-18 else max(0.0, min(1.0, 1.0 - residual_ss / total_ss))
    span = float(np.max(x) - np.min(x))
    slope = float(coefficients[1])
    return {
        "sampleCount": len(usable),
        "phaseSlopeRadPerDay": slope,
        "timeSpanDays": span,
        "phaseTurnsAcrossSpan": abs(slope) * span / (2.0 * math.pi),
        "rSquared": r_squared,
        "circularConcentration": _circular_concentration(
            [item[1] for item in usable],
            [item[2] for item in usable],
        ),
    }


def _amplitude_variation(points: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = []
    for item in points:
        fit = item.get("familyFit") or {}
        value = _float(fit.get(key))
        if value is not None and value >= 0:
            values.append(value)
    if len(values) < 3:
        return None
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    p10, p90 = np.percentile(array, [10.0, 90.0])
    variation = (
        float((p90 - p10) / median)
        if median > 1e-12
        else None
    )
    return {
        "sampleCount": len(values),
        "medianAmplitude": median,
        "p10Amplitude": float(p10),
        "p90Amplitude": float(p90),
        "variationFraction": variation,
    }


def summarize_time_frequency_evolution(
    *,
    interpretation: dict[str, Any],
    physical_period_days: float,
) -> dict[str, Any]:
    windows = interpretation.get("windowResults") or []
    accepted = [item for item in windows if item.get("acceptedTimeFrequencyFeature")]
    accepted_independent = [
        item for item in accepted
        if item.get("role") == "independent-time-frequency-window"
    ]
    clusters = _cluster_points(accepted)
    best_cluster = clusters[0] if clusters else None

    # Residual-feature classification.
    residual_classification = "NO_SIGNIFICANT_TIME_FREQUENCY_STRUCTURE"
    residual_rationale = (
        "No sliding-window residual feature survives the reliability, prominence, coverage, and boundary guards."
    )
    drift = _linear_track(accepted_independent)

    stable_cluster = next(
        (
            cluster for cluster in clusters
            if cluster["windowCount"] >= MIN_STABLE_CLUSTER_WINDOWS
            and cluster["independentSectorCount"] >= MIN_STABLE_CLUSTER_INDEPENDENT_SECTORS
            and cluster["relativeSpan"] <= FREQUENCY_CLUSTER_RELATIVE_TOLERANCE
            and cluster["nearEstablishedFamilyFraction"] < 0.5
        ),
        None,
    )
    independent_sector_count = len({
        int(item["sector"])
        for item in accepted_independent
        if item.get("sector") is not None
    })
    drifting = bool(
        drift is not None
        and len(accepted_independent) >= MIN_DRIFT_WINDOWS
        and independent_sector_count >= MIN_DRIFT_INDEPENDENT_SECTORS
        and drift.get("rSquared", 0.0) >= MIN_DRIFT_R_SQUARED
        and drift.get("fractionalFrequencyChangeAcrossSpan", 0.0) >= MIN_DRIFT_FRACTION
    )

    transient_clusters = [
        cluster for cluster in clusters
        if cluster["windowCount"] >= MIN_TRANSIENT_CLUSTER_WINDOWS
        and cluster["independentSectorCount"] <= 2
        and cluster["nearEstablishedFamilyFraction"] < 0.5
    ]

    if stable_cluster is not None:
        residual_classification = "STABLE_RESIDUAL_MODE"
        residual_rationale = (
            "A residual frequency cluster recurs in multiple time windows across at least three independent sectors."
        )
    elif drifting:
        residual_classification = "DRIFTING_RESIDUAL_MODE"
        residual_rationale = (
            "Accepted residual-window peaks follow a coherent frequency drift across multiple independent sectors."
        )
    elif len(transient_clusters) >= 2:
        residual_classification = "MULTIPLE_TRANSIENT_MODES"
        residual_rationale = (
            "Multiple residual-frequency clusters appear only in limited time/sector intervals rather than recurring stably."
        )
    elif len(transient_clusters) == 1:
        residual_classification = "TRANSIENT_RESIDUAL_MODE"
        residual_rationale = (
            "A residual-frequency cluster is significant in a limited interval but does not recur across enough independent sectors."
        )
    elif len(accepted_independent) >= 3:
        residual_classification = "NONSTATIONARY_RESIDUAL_VARIABILITY"
        residual_rationale = (
            "Several independent time windows contain significant residual structure, but it does not form a stable or coherently drifting mode."
        )

    # Established-family evolution is measured on full-sector fits, not on
    # the shorter residual-search windows.
    family_track = interpretation.get("familyTrack") or []
    fundamental_amplitude = _amplitude_variation(family_track, "fundamentalAmplitude")
    harmonic_amplitude = _amplitude_variation(family_track, "firstHarmonicAmplitude")
    fundamental_phase = _phase_drift(family_track, "fundamentalPhaseRad")
    harmonic_phase = _phase_drift(family_track, "firstHarmonicPhaseRad")

    amplitude_evolving = any(
        metric is not None
        and metric.get("variationFraction") is not None
        and metric["variationFraction"] >= FAMILY_AMPLITUDE_EVOLUTION_THRESHOLD
        for metric in (fundamental_amplitude, harmonic_amplitude)
    )
    phase_evolving = any(
        metric is not None
        and (
            (
                metric.get("phaseTurnsAcrossSpan", 0.0) >= FAMILY_PHASE_DRIFT_TURNS_THRESHOLD
                and metric.get("rSquared", 0.0) >= FAMILY_PHASE_DRIFT_R_SQUARED_THRESHOLD
            )
            or (
                metric.get("circularConcentration") is not None
                and metric["circularConcentration"] < FAMILY_PHASE_CONCENTRATION_THRESHOLD
            )
        )
        for metric in (fundamental_phase, harmonic_phase)
    )

    if amplitude_evolving and phase_evolving:
        family_classification = "FAMILY_AMPLITUDE_AND_PHASE_EVOLUTION"
    elif phase_evolving:
        family_classification = "FAMILY_PHASE_EVOLUTION"
    elif amplitude_evolving:
        family_classification = "FAMILY_AMPLITUDE_EVOLUTION"
    else:
        family_classification = "STABLE_ESTABLISHED_FAMILY"

    accepted_near_family = [item for item in accepted if item.get("nearEstablishedFamily")]
    if residual_classification == "DRIFTING_RESIDUAL_MODE":
        overall = "DRIFTING_RESIDUAL_MODE"
        recommended = "LONG_BASELINE_NONSTATIONARY_MODE_MODELING"
    elif residual_classification == "STABLE_RESIDUAL_MODE":
        overall = "STABLE_RESIDUAL_MODE"
        recommended = "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
    elif residual_classification in {"MULTIPLE_TRANSIENT_MODES", "TRANSIENT_RESIDUAL_MODE"}:
        overall = residual_classification
        recommended = "TRANSIENT_MODE_VALIDATION"
    elif family_classification != "STABLE_ESTABLISHED_FAMILY" and accepted_near_family:
        overall = family_classification
        recommended = "DYNAMIC_HARMONIC_MODELING"
    elif residual_classification == "NONSTATIONARY_RESIDUAL_VARIABILITY":
        overall = "NONSTATIONARY_VARIABILITY"
        recommended = "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
    elif family_classification != "STABLE_ESTABLISHED_FAMILY":
        overall = family_classification
        recommended = "DYNAMIC_HARMONIC_MODELING"
    else:
        overall = "NO_SIGNIFICANT_TIME_FREQUENCY_STRUCTURE"
        recommended = "BINARY_ROTATION_EXTERNAL_EVIDENCE"

    return {
        "classification": overall,
        "physicalPeriodDays": float(physical_period_days),
        "physicalFrequency": 1.0 / float(physical_period_days),
        "firstHarmonicFrequency": 2.0 / float(physical_period_days),
        "windowCount": len(windows),
        "acceptedFeatureCount": len(accepted),
        "acceptedIndependentFeatureCount": len(accepted_independent),
        "acceptedIndependentSectors": sorted({
            int(item["sector"])
            for item in accepted_independent
            if item.get("sector") is not None
        }),
        "acceptedNearEstablishedFamilyCount": len(accepted_near_family),
        "residualEvolution": {
            "classification": residual_classification,
            "rationale": residual_rationale,
            "frequencyClusters": clusters,
            "bestCluster": best_cluster,
            "linearFrequencyTrack": drift,
        },
        "familyEvolution": {
            "classification": family_classification,
            "sectorTrack": family_track,
            "fundamentalAmplitude": fundamental_amplitude,
            "firstHarmonicAmplitude": harmonic_amplitude,
            "fundamentalPhase": fundamental_phase,
            "firstHarmonicPhase": harmonic_phase,
            "amplitudeEvolutionThreshold": FAMILY_AMPLITUDE_EVOLUTION_THRESHOLD,
            "phaseConcentrationThreshold": FAMILY_PHASE_CONCENTRATION_THRESHOLD,
            "phaseDriftTurnsThreshold": FAMILY_PHASE_DRIFT_TURNS_THRESHOLD,
        },
        "windowResults": windows,
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommended,
    }
