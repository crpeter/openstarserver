from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DRIFT_GRID_COUNT = 33
MAX_EDGE_FRACTIONAL_DRIFT = 0.30
MINIMUM_FREQUENCY_HARD = 0.10
MAXIMUM_FREQUENCY_HARD = 0.50
FREQUENCY_HALF_WIDTH_FRACTION = 0.30
TOTAL_FREQUENCIES = 131_072
FREQUENCIES_PER_WORK_UNIT = 2_048
MIN_PEAK_PROMINENCE = 1.5
MIN_BIC_IMPROVEMENT = 10.0
GENERIC_LOMB_SCARGLE_WORKLOAD_ID = "openstar.lomb-scargle.v1"
LOMB_SCARGLE_WORKLOAD_ALIASES = {
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    "openstar.tess-period-search.v1",
}


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
    if len(times) < 256:
        raise RuntimeError("Dataset has too few finite samples for nonstationary modeling.")
    order = np.argsort(times)
    return times[order], flux[order]


def _source_time_origin(dataset: dict[str, Any]) -> float | None:
    source = dataset.get("source") or {}
    for key in ("originalTimeOriginDays", "timeOriginDays"):
        value = _float(source.get(key))
        if value is not None:
            return value
    return None


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
        "role": "primary-long-baseline",
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
            "role": "independent-long-baseline",
        })
    return items


def _design_matrix(times: np.ndarray, frequencies: list[float]) -> np.ndarray:
    columns = [np.ones(len(times), dtype=np.float64)]
    for frequency in frequencies:
        omega_t = 2.0 * math.pi * float(frequency) * times
        columns.append(np.sin(omega_t))
        columns.append(np.cos(omega_t))
    return np.column_stack(columns)


def _fit_established_family(
    *,
    phase_times: np.ndarray,
    flux: np.ndarray,
    physical_frequency: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    design = _design_matrix(
        phase_times,
        [physical_frequency, 2.0 * physical_frequency],
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, flux, rcond=None)
    model = design @ coefficients
    residual = flux - model
    residual_std = float(np.std(residual))
    if math.isfinite(residual_std) and residual_std > 1e-12:
        normalized = residual / residual_std
    else:
        normalized = residual.copy()
    return normalized, {
        "residualStdDevBeforeNormalization": residual_std,
        "sampleCount": int(len(flux)),
    }


def _frequency_search(center_frequency: float) -> dict[str, Any]:
    minimum = max(
        MINIMUM_FREQUENCY_HARD,
        center_frequency * (1.0 - FREQUENCY_HALF_WIDTH_FRACTION),
    )
    maximum = min(
        MAXIMUM_FREQUENCY_HARD,
        center_frequency * (1.0 + FREQUENCY_HALF_WIDTH_FRACTION),
    )
    if maximum <= minimum:
        raise RuntimeError("Invalid long-baseline frequency-search range.")
    step = (maximum - minimum) / (TOTAL_FREQUENCIES - 1)
    return {
        "minimumFrequency": minimum,
        "maximumFrequency": maximum,
        "frequencyStep": step,
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def _drift_grid(relative_times: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(relative_times)))
    if not math.isfinite(max_abs) or max_abs <= 0:
        raise RuntimeError("Long-baseline time span is invalid.")
    q_max = min(0.01, MAX_EDGE_FRACTIONAL_DRIFT / max_abs)
    values = np.linspace(-q_max, q_max, DRIFT_GRID_COUNT, dtype=np.float64)
    # Force the stationary model to be represented exactly.
    values[np.argmin(np.abs(values))] = 0.0
    return values


def _time_warp(relative_times: np.ndarray, fractional_drift_per_day: float) -> np.ndarray:
    q = float(fractional_drift_per_day)
    warped = relative_times + 0.5 * q * np.square(relative_times)
    if np.any(~np.isfinite(warped)):
        raise RuntimeError("Nonstationary time warp produced non-finite values.")
    order = np.argsort(relative_times)
    ordered = warped[order]
    if len(ordered) > 1 and np.any(np.diff(ordered) <= 0):
        raise RuntimeError(
            "Candidate fractional drift makes warped time non-monotonic; reduce the drift grid."
        )
    return warped


def build_nonstationary_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    primary_dataset_path: str | Path,
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    time_frequency_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if physical_period_days <= 0:
        raise ValueError("physical_period_days must be positive")

    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.9 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )
    workload_id = GENERIC_LOMB_SCARGLE_WORKLOAD_ID
    physical_frequency = 1.0 / float(physical_period_days)
    root = Path(output_dir) / "nonstationary"
    root.mkdir(parents=True, exist_ok=True)

    best_cluster = ((time_frequency_summary.get("residualEvolution") or {}).get("bestCluster") or {})
    center_frequency = _float(best_cluster.get("medianFrequency"))
    if center_frequency is None or center_frequency <= 0:
        center_period = _float(best_cluster.get("medianPeriodDays"))
        if center_period is None or center_period <= 0:
            raise RuntimeError("v20.9 requires a v20.8 best residual cluster.")
        center_frequency = 1.0 / center_period

    supporting_sectors = sorted({
        int(value)
        for value in (best_cluster.get("independentSectors") or [])
        if _int(value) is not None
    })

    source_items = _source_items(
        primary_dataset_path=primary_dataset_path,
        primary_sector=primary_sector,
        independent_spec=independent_spec,
    )

    loaded: list[dict[str, Any]] = []
    all_absolute_times: list[np.ndarray] = []
    for item in source_items:
        dataset = _load_json(item["datasetPath"])
        times, flux = _dataset_arrays(dataset)
        origin = _source_time_origin(dataset)
        if origin is None:
            raise RuntimeError(
                "v20.9 long-baseline modeling requires originalTimeOriginDays/timeOriginDays "
                f"for sector {item.get('sectorKey')}."
            )
        absolute_times = times + float(origin)
        all_absolute_times.append(absolute_times)
        loaded.append({
            "item": item,
            "dataset": dataset,
            "times": times,
            "flux": flux,
            "absoluteTimes": absolute_times,
        })

    reference_time = float(np.median(np.concatenate(all_absolute_times)))
    combined_time_parts: list[np.ndarray] = []
    combined_flux_parts: list[np.ndarray] = []
    combined_sector_parts: list[np.ndarray] = []
    sector_residuals: list[dict[str, Any]] = []

    for entry in loaded:
        item = entry["item"]
        absolute_times = entry["absoluteTimes"]
        residual, residual_meta = _fit_established_family(
            phase_times=absolute_times - reference_time,
            flux=entry["flux"],
            physical_frequency=physical_frequency,
        )
        sector_value = item.get("sector")
        sector_numeric = -1 if sector_value is None else int(sector_value)
        combined_time_parts.append(absolute_times)
        combined_flux_parts.append(residual)
        combined_sector_parts.append(np.full(len(residual), sector_numeric, dtype=np.int32))
        sector_residuals.append({
            "sectorKey": item["sectorKey"],
            "sector": sector_value,
            "role": item["role"],
            "sampleCount": int(len(residual)),
            "absoluteStartDays": float(np.min(absolute_times)),
            "absoluteEndDays": float(np.max(absolute_times)),
            **residual_meta,
        })

    absolute_times = np.concatenate(combined_time_parts)
    residual_flux = np.concatenate(combined_flux_parts)
    sector_ids = np.concatenate(combined_sector_parts)
    order = np.argsort(absolute_times)
    absolute_times = absolute_times[order]
    residual_flux = residual_flux[order]
    sector_ids = sector_ids[order]
    relative_times = absolute_times - reference_time

    drift_values = _drift_grid(relative_times)
    search = _frequency_search(center_frequency)
    source_base_id = str(source_dataset_entry.get("id") or "target")

    groups: list[dict[str, Any]] = [{
        "groupID": "all-sectors",
        "label": "all frozen sectors",
        "mask": np.ones(len(relative_times), dtype=bool),
        "sectors": sorted({int(value) for value in sector_ids if int(value) >= 0}),
    }]
    if len(supporting_sectors) >= 2:
        support_mask = np.isin(sector_ids, np.asarray(supporting_sectors, dtype=np.int32))
        if int(np.count_nonzero(support_mask)) >= 256:
            groups.append({
                "groupID": "v20.8-support-sectors",
                "label": "v20.8 best-cluster supporting sectors",
                "mask": support_mask,
                "sectors": supporting_sectors,
            })

    prepared_datasets: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    for group in groups:
        mask = group["mask"]
        group_times = relative_times[mask]
        group_flux = residual_flux[mask]
        group_sector_ids = sector_ids[mask]
        for drift_index, q in enumerate(drift_values):
            warped = _time_warp(group_times, float(q))
            local_times = warped - float(np.min(warped))
            drift_tag = f"{drift_index:02d}"
            dataset_id = (
                f"{source_base_id}-nonstationary-{group['groupID']}-drift-{drift_tag}-v1"
            )
            target_name = (
                f"{source_dataset_entry.get('targetName') or source_base_id} "
                f"long-baseline {group['groupID']} drift candidate {drift_index + 1}/{len(drift_values)}"
            )
            output_path = root / f"{_safe(dataset_id)}.json"

            template_dataset = copy.deepcopy(loaded[0]["dataset"])
            template_dataset["id"] = dataset_id
            template_dataset["targetName"] = target_name
            template_dataset["times"] = np.asarray(local_times, dtype=np.float32).tolist()
            template_dataset["flux"] = np.asarray(group_flux, dtype=np.float32).tolist()
            template_dataset["frequencySearch"] = search
            template_dataset["reference"] = {}
            science = dict(template_dataset.get("science") or {})
            science.update({
                "role": "long-baseline-nonstationary-drift-grid",
                "purpose": "generic-lomb-scargle-on-time-warped-residuals",
                "groupID": group["groupID"],
                "groupSectors": group["sectors"],
                "fractionalFrequencyDriftPerDay": float(q),
                "physicalFundamentalFrequency": physical_frequency,
                "firstHarmonicFrequency": 2.0 * physical_frequency,
            })
            template_dataset["science"] = science
            template_dataset["source"] = {
                "mission": "TESS",
                "baselineDays": float(np.max(group_times) - np.min(group_times)),
                "distributedSamples": int(len(group_times)),
                "timeReferenceDays": reference_time,
                "fractionalFrequencyDriftPerDay": float(q),
                "groupID": group["groupID"],
                "groupSectors": group["sectors"],
                "originalTimeOriginDays": float(np.min(warped)),
            }
            _write_json(output_path, template_dataset)

            meta = {
                "datasetID": dataset_id,
                "datasetPath": str(output_path.resolve()),
                "groupID": group["groupID"],
                "groupLabel": group["label"],
                "groupSectors": group["sectors"],
                "driftIndex": int(drift_index),
                "fractionalFrequencyDriftPerDay": float(q),
                "sampleCount": int(len(group_times)),
                "baselineDays": float(np.max(group_times) - np.min(group_times)),
            }
            prepared_datasets.append(meta)

            manifest_entry = copy.deepcopy(source_dataset_entry)
            manifest_entry.update({
                "id": dataset_id,
                "path": meta["datasetPath"],
                "targetName": target_name,
                "role": "long-baseline-nonstationary-drift-grid",
            })
            dataset_entries.append(manifest_entry)

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "long-baseline-nonstationary-v1"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — long-baseline nonstationary modeling",
        "workloadID": workload_id,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "long-baseline-nonstationary-mode-modeling",
            "workerSemantics": (
                "Each dataset is an ordinary Lomb-Scargle search on a deterministic time warp; "
                "workers and coordinator do not interpret TESS or nonstationary astrophysics."
            ),
            "physicalPeriodDays": float(physical_period_days),
            "residualCenterFrequency": float(center_frequency),
            "supportingSectors": supporting_sectors,
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    analysis_series_path = root / "long-baseline-series-v20.9.json"
    _write_json(analysis_series_path, {
        "timeReferenceDays": reference_time,
        "physicalPeriodDays": float(physical_period_days),
        "centerFrequency": float(center_frequency),
        "supportingSectors": supporting_sectors,
        "absoluteTimes": np.asarray(absolute_times, dtype=np.float64).tolist(),
        "relativeTimes": np.asarray(relative_times, dtype=np.float64).tolist(),
        "residualFlux": np.asarray(residual_flux, dtype=np.float64).tolist(),
        "sectorIDs": np.asarray(sector_ids, dtype=np.int32).tolist(),
        "sectorResiduals": sector_residuals,
    })

    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": True,
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "analysisSeriesPath": str(analysis_series_path.resolve()),
        "workloadID": workload_id,
        "workerSemantics": "generic-lomb-scargle-on-time-warped-residuals",
        "physicalPeriodDays": float(physical_period_days),
        "physicalFrequency": physical_frequency,
        "residualCenterFrequency": float(center_frequency),
        "residualCenterPeriodDays": 1.0 / float(center_frequency),
        "supportingSectors": supporting_sectors,
        "timeReferenceDays": reference_time,
        "timeSpanDays": float(np.max(relative_times) - np.min(relative_times)),
        "frequencySearch": search,
        "driftGrid": {
            "count": int(len(drift_values)),
            "minimumFractionalFrequencyDriftPerDay": float(np.min(drift_values)),
            "maximumFractionalFrequencyDriftPerDay": float(np.max(drift_values)),
            "values": [float(value) for value in drift_values],
        },
        "groups": [
            {
                "groupID": group["groupID"],
                "groupLabel": group["label"],
                "groupSectors": group["sectors"],
                "sampleCount": int(np.count_nonzero(group["mask"])),
            }
            for group in groups
        ],
        "preparedDatasets": prepared_datasets,
        "sectorResiduals": sector_residuals,
        "workUnitsPerDataset": work_units_per_dataset,
        "totalWorkUnits": len(dataset_entries) * work_units_per_dataset,
    }


def _boundary_hit(frequency: float | None, search: dict[str, Any]) -> bool:
    if frequency is None:
        return False
    minimum = _float(search.get("minimumFrequency"))
    maximum = _float(search.get("maximumFrequency"))
    step = _float(search.get("frequencyStep")) or 0.0
    if minimum is None or maximum is None or maximum <= minimum:
        return False
    guard = max(step * 4.0, (maximum - minimum) * 0.004, 1e-12)
    return frequency <= minimum + guard or frequency >= maximum - guard


def interpret_nonstationary_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedDatasets") or []
    }
    search = preparation.get("frequencySearch") or {}
    results: list[dict[str, Any]] = []

    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id)
        if meta is None:
            continue
        frequency = _float(dataset.get("candidateFrequency"))
        period = _float(dataset.get("candidatePeriodDays"))
        power = _float(dataset.get("candidatePower"))
        prominence = _float(dataset.get("candidatePeakProminenceRatio"))
        status = str(dataset.get("periodStatus") or "").upper()
        confidence = str(dataset.get("periodConfidence") or "none").lower()
        reliable = status == "RELIABLE" and confidence in {"high", "medium"}
        boundary = _boundary_hit(frequency, search)
        prominence_ok = prominence is None or prominence >= MIN_PEAK_PROMINENCE
        accepted = bool(
            reliable
            and frequency is not None
            and period is not None
            and power is not None
            and not boundary
            and prominence_ok
        )
        q = float(meta["fractionalFrequencyDriftPerDay"])
        result = {
            **meta,
            "candidateFrequency": frequency,
            "candidatePeriodDays": period,
            "candidatePower": power,
            "candidatePeakProminenceRatio": prominence,
            "periodStatus": status,
            "periodConfidence": confidence,
            "boundaryHit": boundary,
            "accepted": accepted,
            "frequencyDerivativeCyclesPerDaySquared": (
                frequency * q if frequency is not None else None
            ),
        }
        results.append(result)

    groups: dict[str, dict[str, Any]] = {}
    for group_id in sorted({str(item.get("groupID")) for item in results}):
        items = [item for item in results if str(item.get("groupID")) == group_id]
        accepted = [item for item in items if item.get("accepted")]
        stationary = min(
            items,
            key=lambda item: abs(float(item.get("fractionalFrequencyDriftPerDay") or 0.0)),
            default=None,
        )
        best = max(
            accepted,
            key=lambda item: float(item.get("candidatePower") or -math.inf),
            default=None,
        )
        groups[group_id] = {
            "groupID": group_id,
            "groupSectors": (items[0].get("groupSectors") if items else []),
            "candidateCount": len(items),
            "acceptedCandidateCount": len(accepted),
            "stationaryCandidate": stationary,
            "bestCandidate": best,
            "distributedPowerGainOverStationary": (
                float(best["candidatePower"]) - float(stationary["candidatePower"])
                if best is not None
                and stationary is not None
                and best.get("candidatePower") is not None
                and stationary.get("candidatePower") is not None
                else None
            ),
        }

    return {
        "candidateResults": results,
        "groups": groups,
        "acceptedCandidateCount": sum(1 for item in results if item.get("accepted")),
    }


def _sector_offset_matrix(sector_ids: np.ndarray) -> tuple[np.ndarray, list[int]]:
    sectors = sorted({int(value) for value in sector_ids})
    columns = [(sector_ids == sector).astype(np.float64) for sector in sectors]
    return np.column_stack(columns), sectors


def _phase(relative_times: np.ndarray, frequency: float, q: float) -> np.ndarray:
    return 2.0 * math.pi * frequency * (
        relative_times + 0.5 * q * np.square(relative_times)
    )


def _fit_model(
    *,
    relative_times: np.ndarray,
    flux: np.ndarray,
    sector_ids: np.ndarray,
    frequency: float | None,
    q: float,
    signal_sectors: set[int] | None,
    sector_specific_amplitude_phase: bool,
    free_frequency_parameters: int,
    model_id: str,
) -> dict[str, Any]:
    offsets, sectors = _sector_offset_matrix(sector_ids)
    columns = [offsets]
    parameter_count = len(sectors)

    if frequency is not None:
        phase = _phase(relative_times, float(frequency), float(q))
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        if sector_specific_amplitude_phase:
            selected = sectors if signal_sectors is None else [s for s in sectors if s in signal_sectors]
            for sector in selected:
                mask = (sector_ids == sector).astype(np.float64)
                columns.append((mask * sin_phase)[:, None])
                columns.append((mask * cos_phase)[:, None])
                parameter_count += 2
        else:
            mask = np.ones(len(flux), dtype=np.float64)
            if signal_sectors is not None:
                mask = np.isin(sector_ids, np.asarray(sorted(signal_sectors), dtype=np.int32)).astype(np.float64)
            columns.append((mask * sin_phase)[:, None])
            columns.append((mask * cos_phase)[:, None])
            parameter_count += 2
        parameter_count += int(free_frequency_parameters)

    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(design, flux, rcond=None)
    residual = flux - design @ coefficients
    sse = max(float(np.sum(np.square(residual))), 1e-18)
    n = int(len(flux))
    bic = n * math.log(sse / n) + parameter_count * math.log(max(n, 2))
    aic = n * math.log(sse / n) + 2.0 * parameter_count
    return {
        "modelID": model_id,
        "frequency": frequency,
        "periodDays": (1.0 / frequency if frequency is not None and frequency > 0 else None),
        "fractionalFrequencyDriftPerDay": float(q),
        "frequencyDerivativeCyclesPerDaySquared": (
            float(frequency) * float(q) if frequency is not None else None
        ),
        "signalSectors": sorted(signal_sectors) if signal_sectors is not None else sectors,
        "sectorSpecificAmplitudePhase": bool(sector_specific_amplitude_phase),
        "parameterCount": int(parameter_count),
        "sampleCount": n,
        "sse": sse,
        "rms": math.sqrt(sse / n),
        "bic": bic,
        "aic": aic,
    }


def summarize_nonstationary_modeling(
    *,
    interpretation: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    series = _load_json(preparation["analysisSeriesPath"])
    relative_times = np.asarray(series["relativeTimes"], dtype=np.float64)
    flux = np.asarray(series["residualFlux"], dtype=np.float64)
    sector_ids = np.asarray(series["sectorIDs"], dtype=np.int32)
    support_sectors = {int(value) for value in preparation.get("supportingSectors") or []}
    groups = interpretation.get("groups") or {}

    all_group = groups.get("all-sectors") or {}
    support_group = groups.get("v20.8-support-sectors") or {}
    all_stationary = all_group.get("stationaryCandidate") or {}
    all_best = all_group.get("bestCandidate") or all_stationary
    support_stationary = support_group.get("stationaryCandidate") or {}
    support_best = support_group.get("bestCandidate") or support_stationary

    def candidate_frequency(candidate: dict[str, Any]) -> float | None:
        value = _float(candidate.get("candidateFrequency"))
        return value if value is not None and value > 0 else None

    def candidate_q(candidate: dict[str, Any]) -> float:
        return float(_float(candidate.get("fractionalFrequencyDriftPerDay")) or 0.0)

    model_results: list[dict[str, Any]] = []
    model_results.append(_fit_model(
        relative_times=relative_times,
        flux=flux,
        sector_ids=sector_ids,
        frequency=None,
        q=0.0,
        signal_sectors=None,
        sector_specific_amplitude_phase=False,
        free_frequency_parameters=0,
        model_id="NULL_SECTOR_OFFSETS",
    ))

    f_stationary = candidate_frequency(all_stationary)
    if f_stationary is not None:
        model_results.append(_fit_model(
            relative_times=relative_times,
            flux=flux,
            sector_ids=sector_ids,
            frequency=f_stationary,
            q=0.0,
            signal_sectors=None,
            sector_specific_amplitude_phase=False,
            free_frequency_parameters=1,
            model_id="STATIONARY_GLOBAL_MODE",
        ))
        model_results.append(_fit_model(
            relative_times=relative_times,
            flux=flux,
            sector_ids=sector_ids,
            frequency=f_stationary,
            q=0.0,
            signal_sectors=None,
            sector_specific_amplitude_phase=True,
            free_frequency_parameters=1,
            model_id="STATIONARY_SECTOR_EVOLVING_MODE",
        ))

    f_drift = candidate_frequency(all_best)
    q_drift = candidate_q(all_best)
    if f_drift is not None:
        model_results.append(_fit_model(
            relative_times=relative_times,
            flux=flux,
            sector_ids=sector_ids,
            frequency=f_drift,
            q=q_drift,
            signal_sectors=None,
            sector_specific_amplitude_phase=False,
            free_frequency_parameters=2,
            model_id="FREQUENCY_DRIFT_GLOBAL_MODE",
        ))
        model_results.append(_fit_model(
            relative_times=relative_times,
            flux=flux,
            sector_ids=sector_ids,
            frequency=f_drift,
            q=q_drift,
            signal_sectors=None,
            sector_specific_amplitude_phase=True,
            free_frequency_parameters=2,
            model_id="DRIFT_SECTOR_EVOLVING_MODE",
        ))

    if len(support_sectors) >= 2:
        f_support_stationary = candidate_frequency(support_stationary)
        if f_support_stationary is not None:
            model_results.append(_fit_model(
                relative_times=relative_times,
                flux=flux,
                sector_ids=sector_ids,
                frequency=f_support_stationary,
                q=0.0,
                signal_sectors=support_sectors,
                sector_specific_amplitude_phase=False,
                free_frequency_parameters=1,
                model_id="STATIONARY_SUPPORT_SECTORS_MODE",
            ))
        f_support_drift = candidate_frequency(support_best)
        q_support_drift = candidate_q(support_best)
        if f_support_drift is not None:
            model_results.append(_fit_model(
                relative_times=relative_times,
                flux=flux,
                sector_ids=sector_ids,
                frequency=f_support_drift,
                q=q_support_drift,
                signal_sectors=support_sectors,
                sector_specific_amplitude_phase=False,
                free_frequency_parameters=2,
                model_id="FREQUENCY_DRIFT_SUPPORT_SECTORS_MODE",
            ))

    model_results.sort(key=lambda item: float(item["bic"]))
    best = model_results[0]
    null = next(item for item in model_results if item["modelID"] == "NULL_SECTOR_OFFSETS")
    bic_improvement = float(null["bic"] - best["bic"])

    if best["modelID"] == "NULL_SECTOR_OFFSETS" or bic_improvement < MIN_BIC_IMPROVEMENT:
        classification = "NO_COMPELLING_LONG_BASELINE_MODE"
        recommended = "BINARY_ROTATION_EXTERNAL_EVIDENCE"
    else:
        classification = {
            "STATIONARY_GLOBAL_MODE": "STATIONARY_RESIDUAL_MODE",
            "FREQUENCY_DRIFT_GLOBAL_MODE": "FREQUENCY_DRIFT_MODE",
            "STATIONARY_SECTOR_EVOLVING_MODE": "AMPLITUDE_PHASE_EVOLVING_MODE",
            "DRIFT_SECTOR_EVOLVING_MODE": "NONSTATIONARY_DRIFT_WITH_SECTOR_EVOLUTION",
            "STATIONARY_SUPPORT_SECTORS_MODE": "TRANSIENT_SECTOR_LOCALIZED_MODE",
            "FREQUENCY_DRIFT_SUPPORT_SECTORS_MODE": "DRIFTING_TRANSIENT_MODE",
        }.get(best["modelID"], "NONSTATIONARY_MODEL_UNRESOLVED")
        recommended = "RESIDUAL_MODE_PIXEL_LOCALIZATION"

    q = float(best.get("fractionalFrequencyDriftPerDay") or 0.0)
    frequency = _float(best.get("frequency"))
    start_t = float(np.min(relative_times))
    end_t = float(np.max(relative_times))
    frequency_start = (
        frequency * (1.0 + q * start_t) if frequency is not None else None
    )
    frequency_end = (
        frequency * (1.0 + q * end_t) if frequency is not None else None
    )

    return {
        "classification": classification,
        "distributedModeling": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "driftGrid": preparation.get("driftGrid"),
            "frequencySearch": preparation.get("frequencySearch"),
            "groups": groups,
            "totalWorkUnits": preparation.get("totalWorkUnits"),
        },
        "timeReferenceDays": preparation.get("timeReferenceDays"),
        "timeSpanDays": preparation.get("timeSpanDays"),
        "supportingSectors": preparation.get("supportingSectors"),
        "modelComparison": {
            "criterion": "BIC",
            "minimumImprovementForCompellingMode": MIN_BIC_IMPROVEMENT,
            "nullModelID": "NULL_SECTOR_OFFSETS",
            "bestModelID": best.get("modelID"),
            "bicImprovementOverNull": bic_improvement,
            "models": model_results,
        },
        "preferredModel": best,
        "preferredFrequencyAtStart": frequency_start,
        "preferredFrequencyAtReference": frequency,
        "preferredFrequencyAtEnd": frequency_end,
        "preferredPeriodAtReferenceDays": (
            1.0 / frequency if frequency is not None and frequency > 0 else None
        ),
        "frequencyDerivativeCyclesPerDaySquared": (
            frequency * q if frequency is not None else None
        ),
        "fractionalFrequencyDriftPerDay": q if frequency is not None else None,
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommended,
    }
