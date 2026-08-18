from __future__ import annotations

import copy
import json
import math
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np


MAX_SAMPLES_PER_SECTOR = 18_000
TOTAL_FREQUENCIES = 262_144
FREQUENCIES_PER_WORK_UNIT = 4_096
MAX_INDEPENDENT_SECTORS = 4
BROAD_MINIMUM_FREQUENCY = 0.04
BROAD_MAXIMUM_FREQUENCY = 0.20
OFFICIAL_AUTHORS = ("SPOC", "TESS-SPOC")

# Lightkurve search results retain Astropy table/file-backed state and its MAST
# downloader uses process-global machinery.  Keep those objects inside one
# process-local lifecycle; frozen numpy arrays are safe after this lock exits.
_MAST_LIGHTKURVE_LOCK = threading.RLock()


class TessArchiveInfrastructureError(RuntimeError):
    """A transient MAST/Lightkurve failure with durable preparation evidence."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _archive_io_failure(error: Exception) -> bool:
    """Classify I/O only while executing an archive lifecycle operation."""
    return isinstance(error, (OSError, ConnectionError, TimeoutError)) or (
        isinstance(error, ValueError)
        and str(error) == "I/O operation on closed file."
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")


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


def _sector_from_text(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"sector\s*0*(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _sector_from_search_row(table: Any, index: int) -> int | None:
    colnames = set(getattr(table, "colnames", []))
    if "sequence_number" in colnames:
        value = _int(table["sequence_number"][index])
        if value is not None and value > 0:
            return value
    if "mission" in colnames:
        return _sector_from_text(table["mission"][index])
    return None


def _exptime_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        from astropy import units as u
        if hasattr(value, "to_value"):
            return float(value.to_value(u.s))
    except Exception:
        pass
    if hasattr(value, "value"):
        return _float(value.value)
    return _float(value)


def discover_official_sectors(tic_id: int) -> list[int]:
    import lightkurve as lk

    with _MAST_LIGHTKURVE_LOCK:
        search = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
        table = getattr(search, "table", None)
        if table is None or len(table) == 0:
            return []

        sectors: set[int] = set()
        colnames = set(getattr(table, "colnames", []))
        for index in range(len(table)):
            author = None
            if "author" in colnames:
                author = str(table["author"][index]).strip().upper()
            if author not in OFFICIAL_AUTHORS:
                continue
            sector = _sector_from_search_row(table, index)
            if sector is not None:
                sectors.add(sector)
    return sorted(sectors)


def _search_lightcurves(tic_id: int):
    import lightkurve as lk

    return lk.search_lightcurve(
        f"TIC {int(tic_id)}",
        mission="TESS",
    )


def _select_product_from_search(
    search: Any,
    sector: int,
):
    table = getattr(search, "table", None)
    if table is None or len(table) == 0:
        raise RuntimeError("MAST returned no TESS light-curve products.")

    colnames = set(getattr(table, "colnames", []))
    candidates: list[tuple[int, float, int, str, Any]] = []

    for index in range(len(table)):
        row_sector = _sector_from_search_row(table, index)
        if row_sector != int(sector):
            continue

        author = None
        if "author" in colnames:
            author = str(table["author"][index]).strip().upper()
        if author not in OFFICIAL_AUTHORS:
            continue

        author_priority = OFFICIAL_AUTHORS.index(author)
        exptime = (
            _exptime_seconds(table["exptime"][index])
            if "exptime" in colnames
            else None
        )
        cadence_rank = exptime if exptime is not None else float("inf")
        candidates.append(
            (
                author_priority,
                cadence_rank,
                index,
                author,
                search[index:index + 1],
            )
        )

    if not candidates:
        raise RuntimeError(
            "No official SPOC/TESS-SPOC light curve found for "
            f"Sector {sector}."
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, cadence, _, author, selected = candidates[0]
    cadence_seconds = None if not math.isfinite(cadence) else float(cadence)
    return selected, author, cadence_seconds


def _download_selected_sector(
    selected: Any,
    *,
    tic_id: int,
    sector: int,
    author: str,
    cadence_seconds: float | None,
) -> tuple[Any, dict[str, Any]]:
    light_curve = selected.download(quality_bitmask="default")
    if light_curve is None:
        raise RuntimeError(
            f"MAST download returned no light curve for Sector {sector}."
        )

    actual_sector = getattr(light_curve, "sector", None)
    if actual_sector is None:
        actual_sector = getattr(light_curve, "meta", {}).get("SECTOR")
    actual_sector = _int(actual_sector) or int(sector)
    if actual_sector != int(sector):
        raise RuntimeError(
            f"Downloaded unexpected sector for TIC {tic_id}: "
            f"requested={sector}, actual={actual_sector}."
        )

    return light_curve, {
        "author": author,
        "cadenceSeconds": cadence_seconds,
        "sector": actual_sector,
    }


def _rank_independent_sectors(
    sectors: list[int],
    primary_sector: int | None,
) -> list[int]:
    unique = sorted({int(value) for value in sectors})

    if primary_sector is None:
        return unique

    primary = int(primary_sector)
    return sorted(
        unique,
        key=lambda sector: (
            -abs(sector - primary),
            sector,
        ),
    )



def _prepare_samples(light_curve: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    times64 = np.asarray(light_curve.time.value, dtype=np.float64)
    flux64 = np.asarray(light_curve.flux.value, dtype=np.float64)
    finite = np.isfinite(times64) & np.isfinite(flux64)
    times64 = times64[finite]
    flux64 = flux64[finite]

    if len(times64) < 32:
        raise RuntimeError("Independent sector contains too few finite samples.")

    order = np.argsort(times64)
    times64 = times64[order]
    flux64 = flux64[order]

    source_count = len(times64)
    selected_count = min(source_count, MAX_SAMPLES_PER_SECTOR)
    if selected_count < source_count:
        indices = np.linspace(0, source_count - 1, selected_count, dtype=np.int64)
        times64 = times64[indices]
        flux64 = flux64[indices]

    flux_mean = float(np.mean(flux64))
    flux_stddev = float(np.std(flux64))
    if not math.isfinite(flux_stddev) or flux_stddev <= 0:
        raise RuntimeError("Independent sector has invalid flux standard deviation.")

    normalized64 = (flux64 - flux_mean) / flux_stddev
    time_origin = float(times64[0])
    relative64 = times64 - time_origin
    times = np.asarray(relative64, dtype=np.float32)
    flux = np.asarray(normalized64, dtype=np.float32)
    times[0] = np.float32(0.0)

    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(flux)):
        raise RuntimeError("Float32 conversion produced non-finite values.")

    baseline = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    return times, flux, {
        "originalSamples": int(source_count),
        "distributedSamples": int(len(times)),
        "originalTimeOriginDays": time_origin,
        "sourceFluxMean": flux_mean,
        "sourceFluxStddev": flux_stddev,
        "baselineDays": baseline,
    }


def _frequency_window(target_period_days: float) -> dict[str, Any]:
    if target_period_days <= 0:
        raise ValueError("target_period_days must be positive")

    target_frequency = 1.0 / target_period_days
    minimum_frequency = max(0.005, target_frequency * 0.65)
    maximum_frequency = target_frequency * 1.45
    if maximum_frequency <= minimum_frequency:
        raise ValueError("Independent-sector frequency window is invalid.")

    step = (maximum_frequency - minimum_frequency) / (TOTAL_FREQUENCIES - 1)
    return {
        "minimumFrequency": minimum_frequency,
        "maximumFrequency": maximum_frequency,
        "frequencyStep": step,
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def build_independent_sector_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    tic_id: int,
    primary_sector: int | None,
    target_period_days: float,
    candidate_sectors: list[int] | None,
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    source_project_path = Path(source_project_path).expanduser().resolve()
    with source_project_path.open("r", encoding="utf-8") as handle:
        source_project = json.load(handle)

    sectors = sorted(
        {
            int(value)
            for value in (candidate_sectors or [])
            if value is not None
        }
    )
    if not sectors:
        try:
            sectors = discover_official_sectors(tic_id)
        except Exception as error:
            diagnostic = {
                "sector": None,
                "operation": "archive-sector-discovery",
                "error": f"{type(error).__name__}: {error}",
            }
            if _archive_io_failure(error):
                raise TessArchiveInfrastructureError(
                    "MAST sector discovery is temporarily unavailable.",
                    {"candidateSectors": [], "preparedSectors": [],
                     "errors": [diagnostic]},
                ) from error
            raise
    if primary_sector is not None:
        sectors = [
            value
            for value in sectors
            if value != int(primary_sector)
        ]

    sectors = _rank_independent_sectors(
        sectors,
        primary_sector,
    )

    frequency_search = _frequency_window(float(target_period_days))
    artifact_root = Path(output_dir) / "independent-sectors"
    dataset_entries: list[dict[str, Any]] = []
    prepared_sectors: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    selected_sectors = sectors[:MAX_INDEPENDENT_SECTORS]

    print(
        "   querying MAST light-curve catalog once for independent sectors...",
        flush=True,
    )
    infrastructure_errors: list[dict[str, Any]] = []
    try:
        with _MAST_LIGHTKURVE_LOCK:
            search = _search_lightcurves(tic_id)
            materialized: list[tuple[int, str, float | None, Any, Any, dict[str, Any]]] = []
            for position, sector in enumerate(selected_sectors, start=1):
                try:
                    print(
                        f"   sector {sector} ({position}/{len(selected_sectors)}): "
                        "selecting official product",
                        flush=True,
                    )
                    selected, author, cadence_seconds = _select_product_from_search(search, sector)
                    print("      downloading light curve...", flush=True)
                    light_curve, source = _download_selected_sector(
                        selected, tic_id=tic_id, sector=sector, author=author,
                        cadence_seconds=cadence_seconds,
                    )
                    # Materialize while the archive lifecycle remains isolated;
                    # no Lightkurve/Astropy object escapes the critical section.
                    times, flux, prep = _prepare_samples(light_curve)
                    materialized.append((sector, author, cadence_seconds, times, flux, prep))
                except Exception as error:
                    diagnostic = {"sector": int(sector), "operation": "archive-materialization",
                                  "error": f"{type(error).__name__}: {error}"}
                    errors.append(diagnostic)
                    if _archive_io_failure(error):
                        infrastructure_errors.append(diagnostic)
    except Exception as error:
        diagnostic = {"sector": None, "operation": "archive-search",
                      "error": f"{type(error).__name__}: {error}"}
        if _archive_io_failure(error):
            raise TessArchiveInfrastructureError(
                "MAST light-curve search is temporarily unavailable.",
                {"candidateSectors": sectors, "preparedSectors": [], "errors": [diagnostic]},
            ) from error
        return {
            "available": False,
            "projectID": None,
            "projectPath": None,
            "targetPeriodDays": float(target_period_days),
            "primarySector": primary_sector,
            "candidateSectors": sectors,
            "preparedSectors": [],
            "errors": [diagnostic],
            "frequencySearch": frequency_search,
        }

    print(
        f"   sector preparation order: {selected_sectors}",
        flush=True,
    )

    if infrastructure_errors:
        raise TessArchiveInfrastructureError(
            "MAST light-curve download or materialization is temporarily unavailable.",
            {"candidateSectors": sectors, "preparedSectors": [],
             "errors": errors, "frequencySearch": frequency_search},
        )

    for sector, author, cadence_seconds, times, flux, prep in materialized:
        try:
            source = {"author": author, "cadenceSeconds": cadence_seconds}

            dataset_id = f"{source_dataset_entry['id']}-sector-{sector}-independent-v1"
            target_name = (
                f"{source_dataset_entry.get('targetName') or source_dataset_entry['id']} "
                f"independent Sector {sector}"
            )
            dataset = {
                "id": dataset_id,
                "targetName": target_name,
                "mission": "TESS",
                "source": {
                    "archive": "MAST",
                    "author": source["author"],
                    "ticID": int(tic_id),
                    "sector": int(sector),
                    "cadenceSeconds": source.get("cadenceSeconds"),
                    "originalSamples": prep["originalSamples"],
                    "distributedSamples": prep["distributedSamples"],
                    "originalTimeOriginDays": prep["originalTimeOriginDays"],
                    "baselineDays": prep["baselineDays"],
                },
                "science": {
                    "role": "independent-follow-up",
                    "targetPeriodDays": float(target_period_days),
                    "sourceSector": primary_sector,
                },
                "timeUnit": "days",
                "timeReference": "relative-to-first-distributed-sample",
                "numericRepresentation": "Float32",
                "fluxUnit": "normalized",
                "fluxNormalization": "mean-stddev",
                "sampleAllocation": "evenly-spaced-across-finite-rows",
                "times": [float(value) for value in times],
                "flux": [float(value) for value in flux],
                "frequencySearch": dict(frequency_search),
                "reference": {},
            }

            dataset_path = artifact_root / f"{_safe(dataset_id)}.json"
            _write_json(dataset_path, dataset)

            entry = copy.deepcopy(source_dataset_entry)
            entry.update({
                "id": dataset_id,
                "path": str(dataset_path.resolve()),
                "targetName": target_name,
                "ticID": int(tic_id),
                "sector": int(sector),
                "author": source["author"],
                "cadenceSeconds": source.get("cadenceSeconds"),
                "role": "independent-follow-up",
            })
            dataset_entries.append(entry)
            prepared_sectors.append({
                "sector": int(sector),
                "datasetID": dataset_id,
                "datasetPath": str(dataset_path.resolve()),
                "author": source["author"],
                "cadenceSeconds": source.get("cadenceSeconds"),
                "baselineDays": prep["baselineDays"],
                "distributedSamples": prep["distributedSamples"],
            })
            print(
                f"      prepared: {prep['distributedSamples']} samples | "
                f"baseline {prep['baselineDays']:.3f} days",
                flush=True,
            )
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            errors.append({
                "sector": int(sector),
                "error": error_text,
            })
            print(
                f"      skipped: {error_text}",
                flush=True,
            )

    if not dataset_entries:
        return {
            "available": False,
            "projectID": None,
            "projectPath": None,
            "targetPeriodDays": float(target_period_days),
            "primarySector": primary_sector,
            "candidateSectors": sectors,
            "preparedSectors": [],
            "errors": errors,
            "frequencySearch": frequency_search,
        }

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "independent-sectors-v1"
    )
    manifest = {
        "id": project_id,
        "name": (
            f"{source_project.get('name', source_project['id'])} — "
            "independent TESS sector verification"
        ),
        "workloadID": source_project["workloadID"],
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry["id"],
            "purpose": "independent-sector-period-recurrence",
            "targetPeriodDays": float(target_period_days),
            "primarySector": primary_sector,
        },
    }
    manifest_path = artifact_root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    return {
        "available": True,
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "targetPeriodDays": float(target_period_days),
        "primarySector": primary_sector,
        "candidateSectors": sectors,
        "preparedSectors": prepared_sectors,
        "errors": errors,
        "frequencySearch": frequency_search,
        "totalWorkUnits": len(dataset_entries)
        * math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT),
    }



def _broad_frequency_window() -> dict[str, Any]:
    """
    Target-independent contradiction-resolution grid.

    0.04-0.20 cycles/day corresponds to 25-5 days. Scientific eligibility
    is still decided later from each sector's actual baseline/cycle coverage,
    so the long-period end cannot become a claim merely because it was
    searched.
    """
    minimum_frequency = BROAD_MINIMUM_FREQUENCY
    maximum_frequency = BROAD_MAXIMUM_FREQUENCY
    step = (maximum_frequency - minimum_frequency) / (TOTAL_FREQUENCIES - 1)
    return {
        "minimumFrequency": minimum_frequency,
        "maximumFrequency": maximum_frequency,
        "frequencyStep": step,
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def build_broad_independent_sector_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    independent_spec: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    """
    Reuse already-frozen independent TESS sectors for a target-independent
    broad search. No MAST query or download occurs here.
    """
    source_project_path = Path(source_project_path).expanduser().resolve()
    with source_project_path.open("r", encoding="utf-8") as handle:
        source_project = json.load(handle)

    frequency_search = _broad_frequency_window()
    artifact_root = Path(output_dir) / "independent-broad"
    dataset_entries: list[dict[str, Any]] = []
    prepared_sectors: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for item in independent_spec.get("preparedSectors") or []:
        sector = _int(item.get("sector"))
        dataset_path_value = item.get("datasetPath")
        if sector is None or not dataset_path_value:
            continue

        try:
            source_dataset_path = Path(dataset_path_value).expanduser().resolve()
            with source_dataset_path.open("r", encoding="utf-8") as handle:
                dataset = json.load(handle)

            old_dataset_id = str(dataset.get("id") or item.get("datasetID"))
            dataset_id = re.sub(
                r"-independent-v1$",
                "-independent-broad-v1",
                old_dataset_id,
            )
            if dataset_id == old_dataset_id:
                dataset_id = f"{old_dataset_id}-broad-v1"

            target_name = (
                f"{source_dataset_entry.get('targetName') or source_dataset_entry['id']} "
                f"independent Sector {sector} broad search"
            )
            dataset["id"] = dataset_id
            dataset["targetName"] = target_name
            dataset["frequencySearch"] = dict(frequency_search)
            dataset["reference"] = {}
            science = dict(dataset.get("science") or {})
            science.update({
                "role": "independent-broad-follow-up",
                "purpose": "contradiction-resolution-broad-period-search",
                "sourceIndependentDatasetID": old_dataset_id,
            })
            science.pop("targetPeriodDays", None)
            dataset["science"] = science

            broad_dataset_path = artifact_root / f"{_safe(dataset_id)}.json"
            _write_json(broad_dataset_path, dataset)

            entry = copy.deepcopy(source_dataset_entry)
            source = dataset.get("source") or {}
            entry.update({
                "id": dataset_id,
                "path": str(broad_dataset_path.resolve()),
                "targetName": target_name,
                "ticID": source.get("ticID"),
                "sector": int(sector),
                "author": source.get("author"),
                "cadenceSeconds": source.get("cadenceSeconds"),
                "role": "independent-broad-follow-up",
            })
            dataset_entries.append(entry)
            prepared_sectors.append({
                "sector": int(sector),
                "datasetID": dataset_id,
                "datasetPath": str(broad_dataset_path.resolve()),
                "sourceDatasetID": old_dataset_id,
                "baselineDays": _float(source.get("baselineDays"))
                or _float(item.get("baselineDays")),
                "distributedSamples": source.get("distributedSamples")
                or item.get("distributedSamples"),
            })
        except Exception as error:
            errors.append({
                "sector": sector,
                "error": f"{type(error).__name__}: {error}",
            })

    if not dataset_entries:
        return {
            "available": False,
            "projectID": None,
            "projectPath": None,
            "preparedSectors": [],
            "errors": errors,
            "frequencySearch": frequency_search,
            "sourceIndependentProjectID": independent_spec.get("projectID"),
        }

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "independent-broad-v1"
    )
    manifest = {
        "id": project_id,
        "name": (
            f"{source_project.get('name', source_project['id'])} — "
            "independent TESS contradiction-resolution broad search"
        ),
        "workloadID": source_project["workloadID"],
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry["id"],
            "sourceIndependentProjectID": independent_spec.get("projectID"),
            "purpose": "independent-sector-contradiction-resolution",
            "frequencySearchMode": "target-independent-broad",
        },
    }
    manifest_path = artifact_root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    return {
        "available": True,
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "preparedSectors": prepared_sectors,
        "errors": errors,
        "frequencySearch": frequency_search,
        "sourceIndependentProjectID": independent_spec.get("projectID"),
        "totalWorkUnits": len(dataset_entries)
        * math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT),
    }
