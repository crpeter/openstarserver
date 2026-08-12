from __future__ import annotations

import copy
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MINIMUM_FREQUENCY = 0.04
MAXIMUM_FREQUENCY = 5.0
TOTAL_FREQUENCIES = 262_144
FREQUENCIES_PER_WORK_UNIT = 2_048
MAX_RESIDUAL_ITERATIONS = 3
MAX_COMBINED_SAMPLES = 18_000
MIN_PEAK_PROMINENCE = 1.5
MIN_OBSERVED_CYCLES = 2.0
BOUNDARY_FRACTION = 0.002
CLUSTER_RELATIVE_TOLERANCE = 0.05
MIN_RECURRENT_INDEPENDENT_SECTORS = 3
NEAR_PRIMARY_RELATIVE_LIMIT = 0.25


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


def _baseline(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    return float(np.max(times) - np.min(times))


def _source_time_origin(dataset: dict[str, Any]) -> float | None:
    source = dataset.get("source") or {}
    for key in ("originalTimeOriginDays", "timeOriginDays"):
        value = _float(source.get(key))
        if value is not None:
            return value
    return None


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


def _prewhiten(
    times: np.ndarray,
    flux: np.ndarray,
    frequencies: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    finite = np.isfinite(times) & np.isfinite(flux)
    times = np.asarray(times[finite], dtype=np.float64)
    flux = np.asarray(flux[finite], dtype=np.float64)
    if len(times) < 32:
        raise RuntimeError("Residual decomposition dataset has too few finite samples.")

    unique_frequencies: list[float] = []
    for value in frequencies:
        frequency = _float(value)
        if frequency is None or frequency <= 0:
            continue
        if any(abs(frequency - existing) <= 1e-10 for existing in unique_frequencies):
            continue
        unique_frequencies.append(frequency)

    design = _design_matrix(times, unique_frequencies)
    coefficients, _, _, _ = np.linalg.lstsq(design, flux, rcond=None)
    model = design @ coefficients
    residual = flux - model

    input_rms = float(np.sqrt(np.mean(np.square(flux))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    residual_std = float(np.std(residual))
    if not math.isfinite(residual_std) or residual_std <= 1e-12:
        raise RuntimeError("Residual decomposition produced zero/invalid residual variance.")

    normalized = residual / residual_std
    explained_fraction = None
    if input_rms > 0:
        explained_fraction = max(0.0, min(1.0, 1.0 - (residual_rms * residual_rms) / (input_rms * input_rms)))

    return normalized, {
        "removedFrequencies": unique_frequencies,
        "inputRMS": input_rms,
        "residualRMS": residual_rms,
        "residualStdDevBeforeNormalization": residual_std,
        "explainedVarianceFraction": explained_fraction,
    }


def _dataset_arrays(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(dataset.get("times") or [], dtype=np.float64)
    flux = np.asarray(dataset.get("flux") or [], dtype=np.float64)
    if len(times) != len(flux):
        raise RuntimeError("Dataset times/flux lengths do not match.")
    finite = np.isfinite(times) & np.isfinite(flux)
    times = times[finite]
    flux = flux[finite]
    if len(times) < 32:
        raise RuntimeError("Dataset has too few finite samples for residual decomposition.")
    return times, flux


def _previous_sector_modes(
    prior_iterations: list[dict[str, Any]],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for iteration in prior_iterations:
        for item in iteration.get("datasetResults") or []:
            if not item.get("acceptedDistinctMode"):
                continue
            frequency = _float(item.get("candidateFrequency"))
            key = str(item.get("sectorKey") or "")
            if frequency is not None and key:
                result[key].append(frequency)
    return dict(result)


def _copy_residual_dataset(
    *,
    source_dataset: dict[str, Any],
    output_path: Path,
    dataset_id: str,
    target_name: str,
    role: str,
    iteration: int,
    sector_key: str,
    sector: int | None,
    physical_frequency: float,
    prior_modes: list[float],
) -> dict[str, Any]:
    times, flux = _dataset_arrays(source_dataset)
    removed = [physical_frequency, 2.0 * physical_frequency, *prior_modes]
    residual, fit = _prewhiten(times, flux, removed)

    residual_dataset = copy.deepcopy(source_dataset)
    residual_dataset["id"] = dataset_id
    residual_dataset["targetName"] = target_name
    residual_dataset["times"] = np.asarray(times, dtype=np.float32).tolist()
    residual_dataset["flux"] = np.asarray(residual, dtype=np.float32).tolist()
    residual_dataset["frequencySearch"] = _frequency_search()
    residual_dataset["reference"] = {}
    science = dict(residual_dataset.get("science") or {})
    science.update({
        "role": role,
        "purpose": "iterative-residual-frequency-decomposition",
        "residualIteration": int(iteration),
        "sectorKey": sector_key,
        "physicalFundamentalFrequency": physical_frequency,
        "firstHarmonicFrequency": 2.0 * physical_frequency,
        "prewhitenedFrequencies": fit["removedFrequencies"],
    })
    residual_dataset["science"] = science
    source = dict(residual_dataset.get("source") or {})
    source["baselineDays"] = _baseline(times)
    source["residualIteration"] = int(iteration)
    source["prewhitening"] = fit
    if sector is not None:
        source["sector"] = int(sector)
    residual_dataset["source"] = source
    _write_json(output_path, residual_dataset)

    return {
        "datasetID": dataset_id,
        "datasetPath": str(output_path.resolve()),
        "sectorKey": sector_key,
        "sector": sector,
        "role": role,
        "baselineDays": _baseline(times),
        "sampleCount": int(len(times)),
        "prewhitenedFrequencies": fit["removedFrequencies"],
        "prewhitening": fit,
        "absoluteTimeOriginDays": _source_time_origin(source_dataset),
    }


def _combined_residual_dataset(
    *,
    prepared: list[dict[str, Any]],
    output_path: Path,
    dataset_id: str,
    target_name: str,
    iteration: int,
    physical_frequency: float,
) -> dict[str, Any] | None:
    chunks: list[tuple[np.ndarray, np.ndarray, int | None]] = []
    origins: list[float] = []
    for item in prepared:
        origin = _float(item.get("absoluteTimeOriginDays"))
        if origin is None:
            return None
        dataset = _load_json(item["datasetPath"])
        times, flux = _dataset_arrays(dataset)
        absolute = times + origin
        chunks.append((absolute, flux, item.get("sector")))
        origins.append(float(np.min(absolute)))

    if not chunks:
        return None

    absolute_origin = min(origins)
    selected_times: list[np.ndarray] = []
    selected_flux: list[np.ndarray] = []
    per_chunk_cap = max(64, MAX_COMBINED_SAMPLES // len(chunks))
    for absolute, flux, _ in chunks:
        if len(absolute) > per_chunk_cap:
            indices = np.linspace(0, len(absolute) - 1, per_chunk_cap, dtype=np.int64)
            absolute = absolute[indices]
            flux = flux[indices]
        selected_times.append(absolute - absolute_origin)
        selected_flux.append(flux)

    times = np.concatenate(selected_times)
    flux = np.concatenate(selected_flux)
    order = np.argsort(times)
    times = times[order]
    flux = flux[order]
    # Individual residual datasets are already normalized and have the current
    # iteration's modes removed. The combined search should only remove any
    # remaining global constant offset; using an empty frequency list does that.
    residual, fit = _prewhiten(times, flux, [])

    source = {
        "mission": "TESS",
        "sector": None,
        "baselineDays": _baseline(times),
        "originalTimeOriginDays": absolute_origin,
        "distributedSamples": int(len(times)),
        "combinedSectors": [item.get("sector") for item in prepared],
        "residualIteration": int(iteration),
        "prewhitening": fit,
    }
    dataset = {
        "id": dataset_id,
        "targetName": target_name,
        "times": np.asarray(times, dtype=np.float32).tolist(),
        "flux": np.asarray(residual, dtype=np.float32).tolist(),
        "frequencySearch": _frequency_search(),
        "reference": {},
        "source": source,
        "science": {
            "role": "combined-residual-multimode",
            "purpose": "multi-sector-residual-frequency-decomposition",
            "residualIteration": int(iteration),
            "sectorKey": "combined",
            "physicalFundamentalFrequency": physical_frequency,
            "firstHarmonicFrequency": 2.0 * physical_frequency,
        },
    }
    _write_json(output_path, dataset)
    return {
        "datasetID": dataset_id,
        "datasetPath": str(output_path.resolve()),
        "sectorKey": "combined",
        "sector": None,
        "role": "combined-residual-multimode",
        "baselineDays": _baseline(times),
        "sampleCount": int(len(times)),
        "prewhitenedFrequencies": [],
        "prewhitening": fit,
        "absoluteTimeOriginDays": absolute_origin,
        "combinedSectors": source["combinedSectors"],
    }


def build_residual_search_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    primary_dataset_path: str | Path,
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    physical_period_days: float,
    prior_iterations: list[dict[str, Any]],
    iteration: int,
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if physical_period_days <= 0:
        raise ValueError("physical_period_days must be positive")
    if iteration < 1 or iteration > MAX_RESIDUAL_ITERATIONS:
        raise ValueError(f"iteration must be within 1..{MAX_RESIDUAL_ITERATIONS}")

    source_project = _load_json(source_project_path)
    physical_frequency = 1.0 / float(physical_period_days)
    prior_modes = _previous_sector_modes(prior_iterations)
    root = Path(output_dir) / "multimode" / f"iteration-{iteration}"
    root.mkdir(parents=True, exist_ok=True)

    source_items: list[dict[str, Any]] = [{
        "sectorKey": str(primary_sector if primary_sector is not None else "primary"),
        "sector": _int(primary_sector),
        "datasetPath": str(Path(primary_dataset_path).expanduser().resolve()),
        "role": "primary-residual-multimode",
    }]
    for item in independent_spec.get("preparedSectors") or []:
        sector = _int(item.get("sector"))
        path = item.get("datasetPath")
        if sector is None or not path:
            continue
        source_items.append({
            "sectorKey": str(sector),
            "sector": sector,
            "datasetPath": str(Path(path).expanduser().resolve()),
            "role": "independent-residual-multimode",
        })

    prepared: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    for source_item in source_items:
        sector_key = source_item["sectorKey"]
        sector = source_item["sector"]
        source_dataset = _load_json(source_item["datasetPath"])
        suffix = f"sector-{sector_key}" if sector is not None else sector_key
        dataset_id = f"{source_dataset.get('id', source_dataset_entry['id'])}-residual-mode-{iteration}-v1"
        target_name = (
            f"{source_dataset_entry.get('targetName') or source_dataset_entry['id']} "
            f"{suffix} residual mode iteration {iteration}"
        )
        output_path = root / f"{_safe(dataset_id)}.json"
        item = _copy_residual_dataset(
            source_dataset=source_dataset,
            output_path=output_path,
            dataset_id=dataset_id,
            target_name=target_name,
            role=source_item["role"],
            iteration=iteration,
            sector_key=sector_key,
            sector=sector,
            physical_frequency=physical_frequency,
            prior_modes=prior_modes.get(sector_key, []),
        )
        prepared.append(item)

        entry = copy.deepcopy(source_dataset_entry)
        entry.update({
            "id": dataset_id,
            "path": item["datasetPath"],
            "targetName": target_name,
            "sector": sector,
            "role": source_item["role"],
        })
        dataset_entries.append(entry)

    combined_id = f"{source_dataset_entry['id']}-combined-residual-mode-{iteration}-v1"
    combined_path = root / f"{_safe(combined_id)}.json"
    combined = _combined_residual_dataset(
        prepared=prepared,
        output_path=combined_path,
        dataset_id=combined_id,
        target_name=(
            f"{source_dataset_entry.get('targetName') or source_dataset_entry['id']} "
            f"combined residual mode iteration {iteration}"
        ),
        iteration=iteration,
        physical_frequency=physical_frequency,
    )
    if combined is not None:
        prepared.append(combined)
        entry = copy.deepcopy(source_dataset_entry)
        entry.update({
            "id": combined_id,
            "path": combined["datasetPath"],
            "targetName": (
                f"{source_dataset_entry.get('targetName') or source_dataset_entry['id']} "
                f"combined residual mode iteration {iteration}"
            ),
            "sector": None,
            "role": "combined-residual-multimode",
        })
        dataset_entries.append(entry)

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        f"residual-multimode-{iteration}-v1"
    )
    manifest = {
        "id": project_id,
        "name": (
            f"{source_project.get('name', source_project['id'])} — "
            f"residual multi-mode iteration {iteration}"
        ),
        "workloadID": source_project["workloadID"],
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry["id"],
            "purpose": "iterative-residual-frequency-decomposition",
            "iteration": int(iteration),
            "physicalPeriodDays": float(physical_period_days),
            "physicalFrequency": physical_frequency,
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": bool(dataset_entries),
        "iteration": int(iteration),
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "physicalPeriodDays": float(physical_period_days),
        "physicalFrequency": physical_frequency,
        "frequencySearch": _frequency_search(),
        "preparedDatasets": prepared,
        "totalWorkUnits": len(dataset_entries) * work_units_per_dataset,
        "workUnitsPerDataset": work_units_per_dataset,
        "maximumIterations": MAX_RESIDUAL_ITERATIONS,
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


def _nearest_removed_frequency(candidate_frequency: float, removed: list[float]) -> tuple[float | None, float | None]:
    if not removed:
        return None, None
    nearest = min(removed, key=lambda value: abs(candidate_frequency - value))
    return nearest, abs(candidate_frequency - nearest)


def interpret_residual_iteration(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedDatasets") or []
    }
    search = preparation.get("frequencySearch") or {}
    physical_frequency = float(preparation["physicalFrequency"])
    results: list[dict[str, Any]] = []
    accepted_independent = 0
    accepted_total = 0
    unresolved_family_residuals = 0
    combined_accepted = False

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
        boundary_hit = _candidate_boundary_hit(frequency, search)
        reliable = status == "RELIABLE" and confidence in {"high", "medium"}
        prominence_ok = prominence is not None and prominence >= MIN_PEAK_PROMINENCE
        coverage_ok = observed_cycles >= MIN_OBSERVED_CYCLES

        removed = [
            float(value)
            for value in (meta.get("prewhitenedFrequencies") or [])
            if _float(value) is not None
        ]
        nearest_removed = None
        removed_separation = None
        rayleigh = (1.0 / baseline) if baseline > 0 else None
        distinct_from_removed = False
        near_primary_family = False
        resolved_near_primary = False
        if frequency is not None:
            nearest_removed, removed_separation = _nearest_removed_frequency(frequency, removed)
            distinct_from_removed = (
                nearest_removed is None
                or rayleigh is None
                or removed_separation >= rayleigh
            )
            primary_distance = min(
                abs(frequency - physical_frequency),
                abs(frequency - 2.0 * physical_frequency),
            )
            primary_reference = (
                physical_frequency
                if abs(frequency - physical_frequency) <= abs(frequency - 2.0 * physical_frequency)
                else 2.0 * physical_frequency
            )
            near_primary_family = primary_distance <= max(
                primary_reference * NEAR_PRIMARY_RELATIVE_LIMIT,
                rayleigh or 0.0,
            )
            resolved_near_primary = bool(
                near_primary_family
                and rayleigh is not None
                and primary_distance >= rayleigh
            )

        accepted = bool(
            reliable
            and prominence_ok
            and coverage_ok
            and not boundary_hit
            and frequency is not None
            and period is not None
            and distinct_from_removed
        )
        if reliable and prominence_ok and coverage_ok and not boundary_hit and not distinct_from_removed:
            unresolved_family_residuals += 1

        role = str(meta.get("role") or "")
        is_independent = role == "independent-residual-multimode"
        is_combined = role == "combined-residual-multimode"
        if accepted:
            accepted_total += 1
            if is_independent:
                accepted_independent += 1
            if is_combined:
                combined_accepted = True

        result = {
            "datasetID": dataset_id,
            "sectorKey": meta.get("sectorKey"),
            "sector": meta.get("sector"),
            "role": role,
            "periodStatus": status,
            "periodConfidence": confidence,
            "candidateFrequency": frequency,
            "candidatePeriodDays": period,
            "candidatePower": _float(dataset.get("candidatePower")),
            "candidatePeakProminenceRatio": prominence,
            "baselineDays": baseline,
            "observedCycles": observed_cycles,
            "boundaryHit": boundary_hit,
            "reliable": reliable,
            "prominenceOK": prominence_ok,
            "coverageOK": coverage_ok,
            "rayleighFrequency": rayleigh,
            "nearestPrewhitenedFrequency": nearest_removed,
            "separationFromNearestPrewhitenedFrequency": removed_separation,
            "distinctFromPrewhitenedModes": distinct_from_removed,
            "nearPrimaryFamily": near_primary_family,
            "resolvedNearPrimaryFamily": resolved_near_primary,
            "acceptedDistinctMode": accepted,
        }
        results.append(result)

    # Continue if there is credible residual structure to remove. A single
    # sector-only peak is retained diagnostically but does not force another
    # expensive distributed iteration unless the combined baseline also sees it.
    continue_recommended = bool(
        preparation.get("iteration", 1) < MAX_RESIDUAL_ITERATIONS
        and (accepted_independent >= 2 or combined_accepted)
    )

    return {
        "iteration": preparation.get("iteration"),
        "physicalFrequency": physical_frequency,
        "datasetResults": results,
        "acceptedDistinctModeCount": accepted_total,
        "acceptedIndependentSectorCount": accepted_independent,
        "combinedAcceptedDistinctMode": combined_accepted,
        "unresolvedPrimaryFamilyResidualCount": unresolved_family_residuals,
        "continueRecommended": continue_recommended,
        "minimumPeakProminenceRatio": MIN_PEAK_PROMINENCE,
        "minimumObservedCycles": MIN_OBSERVED_CYCLES,
    }


def _cluster_frequency_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    for point in sorted(points, key=lambda item: float(item["candidateFrequency"])):
        frequency = float(point["candidateFrequency"])
        placed = False
        for cluster in clusters:
            center = float(np.median([float(item["candidateFrequency"]) for item in cluster]))
            if abs(frequency - center) / max(abs(center), 1e-12) <= CLUSTER_RELATIVE_TOLERANCE:
                cluster.append(point)
                placed = True
                break
        if not placed:
            clusters.append([point])

    summaries: list[dict[str, Any]] = []
    for cluster in clusters:
        frequencies = [float(item["candidateFrequency"]) for item in cluster]
        sectors = sorted({
            int(item["sector"])
            for item in cluster
            if item.get("sector") is not None
            and item.get("role") == "independent-residual-multimode"
        })
        combined_support = any(item.get("role") == "combined-residual-multimode" for item in cluster)
        median_frequency = float(np.median(frequencies))
        summaries.append({
            "medianFrequency": median_frequency,
            "medianPeriodDays": 1.0 / median_frequency,
            "independentSectors": sectors,
            "independentSectorCount": len(sectors),
            "combinedSupport": combined_support,
            "members": [
                {
                    "iteration": item.get("iteration"),
                    "sector": item.get("sector"),
                    "role": item.get("role"),
                    "frequency": item.get("candidateFrequency"),
                    "periodDays": item.get("candidatePeriodDays"),
                    "prominence": item.get("candidatePeakProminenceRatio"),
                }
                for item in cluster
            ],
        })
    summaries.sort(
        key=lambda item: (
            item["independentSectorCount"],
            1 if item["combinedSupport"] else 0,
        ),
        reverse=True,
    )
    return summaries


def summarize_multimode_decomposition(
    *,
    iteration_results: list[dict[str, Any]],
    physical_period_days: float,
) -> dict[str, Any]:
    physical_frequency = 1.0 / float(physical_period_days)
    points: list[dict[str, Any]] = []
    independent_sectors_with_modes: set[int] = set()
    resolved_near_primary_sectors: set[int] = set()
    combined_near_primary = False
    for iteration in iteration_results:
        iteration_number = iteration.get("iteration")
        for item in iteration.get("datasetResults") or []:
            if not item.get("acceptedDistinctMode"):
                continue
            point = dict(item)
            point["iteration"] = iteration_number
            points.append(point)
            if item.get("role") == "independent-residual-multimode" and item.get("sector") is not None:
                independent_sectors_with_modes.add(int(item["sector"]))
                if item.get("resolvedNearPrimaryFamily"):
                    resolved_near_primary_sectors.add(int(item["sector"]))
            if item.get("role") == "combined-residual-multimode" and item.get("resolvedNearPrimaryFamily"):
                combined_near_primary = True

    clusters = _cluster_frequency_points(points)
    recurrent = next(
        (
            cluster
            for cluster in clusters
            if cluster["independentSectorCount"] >= MIN_RECURRENT_INDEPENDENT_SECTORS
        ),
        None,
    )

    if recurrent is not None:
        classification = "MULTI_MODE_RECURRENT"
        rationale = (
            "A secondary residual frequency recurs across at least three independent sectors "
            "after prewhitening the resolved physical fundamental and first harmonic."
        )
        recommended = "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
    elif len(resolved_near_primary_sectors) >= 2 or combined_near_primary:
        classification = "POSSIBLE_BEATING"
        rationale = (
            "Resolved residual power remains close to the established physical-frequency family, "
            "consistent with a nearby mode or beating rather than a fully removed stationary signal."
        )
        recommended = "LONG_BASELINE_BEATING_CONFIRMATION"
    elif len(independent_sectors_with_modes) >= 3:
        classification = "EVOLVING_QUASI_PERIODIC_VARIABILITY"
        rationale = (
            "Several independent sectors retain significant residual periodic structure, but the "
            "frequencies do not form a stable recurrent cross-sector mode."
        )
        recommended = "TIME_FREQUENCY_EVOLUTION_ANALYSIS"
    elif not points:
        classification = "SINGLE_FAMILY_WITH_EVOLVING_MORPHOLOGY"
        rationale = (
            "No distinct residual mode survives the deterministic reliability, prominence, coverage, "
            "boundary, and frequency-resolution guards after removing the established family."
        )
        recommended = "BINARY_ROTATION_EXTERNAL_EVIDENCE"
    else:
        classification = "MULTI_MODE_OR_EVOLVING_UNRESOLVED"
        rationale = (
            "Residual periodic structure is present, but it is too sparse or inconsistent across "
            "independent sectors to classify as a recurrent secondary mode."
        )
        recommended = "TIME_FREQUENCY_EVOLUTION_ANALYSIS"

    return {
        "classification": classification,
        "rationale": rationale,
        "physicalPeriodDays": float(physical_period_days),
        "physicalFrequency": physical_frequency,
        "firstHarmonicFrequency": 2.0 * physical_frequency,
        "iterationsCompleted": len(iteration_results),
        "acceptedResidualModes": points,
        "frequencyClusters": clusters,
        "bestRecurrentSecondaryMode": recurrent,
        "independentSectorsWithAcceptedResidualModes": sorted(independent_sectors_with_modes),
        "resolvedNearPrimaryFamilySectors": sorted(resolved_near_primary_sectors),
        "combinedResolvedNearPrimaryFamily": combined_near_primary,
        "minimumRecurrentIndependentSectorCount": MIN_RECURRENT_INDEPENDENT_SECTORS,
        "clusterRelativeTolerance": CLUSTER_RELATIVE_TOLERANCE,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": recommended,
    }
