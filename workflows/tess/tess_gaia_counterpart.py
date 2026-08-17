from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .tess_external_highres import (
    GAIA_DATA_RELEASE,
    MIN_GAIA_G_SAMPLES,
    MIN_PEAK_PROMINENCE_RATIO,
    _angular_separation_arcsec,
    _dataset_result,
    _download_gaia_epoch_csv,
    _parse_gaia_g_series,
    _query_gaia_source_metadata,
    _validate_position,
)
from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    _float,
    _int,
    _safe,
    _write_json,
)


NEXT_ARCHIVE_TEST = "SKYMAPPER_RESOLVED_COUNTERPART_PHOTOMETRY"
CURRENT_TRIGGER = "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
TOTAL_FREQUENCIES = 8_192
FREQUENCIES_PER_WORK_UNIT = 2_048
MIN_SEARCH_HALF_WIDTH_FRACTION = 0.02
MAX_SEARCH_HALF_WIDTH_FRACTION = 0.20


class GaiaArchiveUnavailable(RuntimeError):
    """The Gaia service failed transiently before a scientific result existed."""


def _frequency_search(offset_variability: dict[str, Any]) -> tuple[dict[str, Any], float]:
    counterpart = offset_variability.get("catalogCounterpartEvidence") or {}
    period = _float(counterpart.get("combinedPeriodDays"))
    frequency = _float(counterpart.get("combinedFrequency"))
    if frequency is None and period is not None and period > 0:
        frequency = 1.0 / period
    if frequency is None or frequency <= 0:
        raise RuntimeError("Current counterpart evidence has no positive residual-period hypothesis.")

    sector_frequencies = [
        value
        for item in offset_variability.get("counterpartPerSectorResults") or []
        if (value := _float(item.get("candidateFrequency"))) is not None and value > 0
    ]
    fractional_uncertainty = MIN_SEARCH_HALF_WIDTH_FRACTION
    if len(sector_frequencies) >= 2:
        median = statistics.median(sector_frequencies)
        mad = statistics.median(abs(value - median) for value in sector_frequencies)
        fractional_uncertainty = max(fractional_uncertainty, 3.0 * 1.4826 * mad / frequency)
    old_search = (offset_variability.get("distributedValidation") or {}).get("frequencySearch") or {}
    old_step = _float(old_search.get("frequencyStep"))
    if old_step is not None and old_step > 0:
        fractional_uncertainty = max(fractional_uncertainty, 3.0 * old_step / frequency)
    fractional_uncertainty = min(MAX_SEARCH_HALF_WIDTH_FRACTION, fractional_uncertainty)
    minimum = frequency * (1.0 - fractional_uncertainty)
    maximum = frequency * (1.0 + fractional_uncertainty)
    step = (maximum - minimum) / (TOTAL_FREQUENCIES - 1)
    return ({
        "minimumFrequency": minimum,
        "maximumFrequency": maximum,
        "frequencyStep": step,
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
        "preregisteredFrom": "current-counterpart-residual-period-hypothesis",
        "uncertaintyHalfWidthFraction": fractional_uncertainty,
    }, frequency)


def _source_pair(
    *, prepared_target: dict[str, Any], identity: dict[str, Any],
    catalog_identification: dict[str, Any], offset_variability: dict[str, Any],
) -> dict[str, Any]:
    target_tic = _int(prepared_target.get("ticID"))
    target_gaia_record = (identity.get("gaiaDR3") or {}).get("nearest") or {}
    target_gaia = _int(target_gaia_record.get("sourceID"))
    target_metadata = (identity.get("tic") or {}).get("metadata") or {}
    target_ra = _float(target_gaia_record.get("raDeg"))
    target_dec = _float(target_gaia_record.get("decDeg"))
    if target_ra is None:
        target_ra = _float(target_metadata.get("raDeg"))
    if target_dec is None:
        target_dec = _float(target_metadata.get("decDeg"))

    preferred = catalog_identification.get("preferredCandidate") or {}
    preferred_ids = preferred.get("catalogIDs") or {}
    persisted_counterpart = offset_variability.get("catalogCounterpart") or {}
    counterpart_tic = _int(persisted_counterpart.get("ticID"))
    counterpart_gaia = _int(persisted_counterpart.get("gaiaDR3SourceID"))
    counterpart_ra = _float(persisted_counterpart.get("raDeg"))
    counterpart_dec = _float(persisted_counterpart.get("decDeg"))
    if counterpart_tic is None:
        counterpart_tic = _int(preferred_ids.get("ticID"))
    if counterpart_gaia is None:
        counterpart_gaia = _int(preferred_ids.get("gaiaDR3SourceID"))
    if counterpart_ra is None:
        counterpart_ra = _float(preferred.get("raDeg"))
    if counterpart_dec is None:
        counterpart_dec = _float(preferred.get("decDeg"))

    required = (target_tic, target_gaia, target_ra, target_dec,
                counterpart_gaia, counterpart_ra, counterpart_dec)
    if any(value is None for value in required):
        raise RuntimeError("Persisted evidence does not define a complete target/counterpart Gaia source pair.")
    if target_gaia == counterpart_gaia:
        raise RuntimeError("Target and counterpart resolve to the same Gaia DR3 source ID.")
    separation = _angular_separation_arcsec(target_ra, target_dec, counterpart_ra, counterpart_dec)
    return {
        "version": "openstar.current-source-pair.v1",
        "target": {"sourceRole": "target-control", "ticID": target_tic,
                   "gaiaDR3SourceID": target_gaia, "raDeg": target_ra, "decDeg": target_dec},
        "counterpart": {"sourceRole": "catalog-counterpart", "ticID": counterpart_tic,
                        "gaiaDR3SourceID": counterpart_gaia,
                        "raDeg": counterpart_ra, "decDeg": counterpart_dec},
        "separationArcsec": separation,
        "frozenFromPersistedEvidence": True,
    }


def build_current_gaia_counterpart_project(
    *, source_project_id: str, source_dataset_id: str, prepared_target: dict[str, Any],
    identity: dict[str, Any], catalog_identification: dict[str, Any],
    offset_variability: dict[str, Any], output_dir: str | Path, investigation_id: str,
    query_metadata: Callable[[list[int]], dict[int, dict[str, Any]]] = _query_gaia_source_metadata,
    download_epochs: Callable[[int], tuple[bytes, str]] = _download_gaia_epoch_csv,
) -> dict[str, Any]:
    if offset_variability.get("recommendedNextTest") != CURRENT_TRIGGER:
        raise RuntimeError(f"Current Gaia continuation requires {CURRENT_TRIGGER}.")
    pair = _source_pair(prepared_target=prepared_target, identity=identity,
                        catalog_identification=catalog_identification,
                        offset_variability=offset_variability)
    search, reference_frequency = _frequency_search(offset_variability)
    sources = [pair["target"], pair["counterpart"]]
    ids = [int(source["gaiaDR3SourceID"]) for source in sources]
    try:
        metadata = query_metadata(ids)
    except Exception as exc:
        raise GaiaArchiveUnavailable(f"Gaia DR3 metadata query failed: {type(exc).__name__}: {exc}") from exc
    missing = sorted(set(ids) - set(metadata))
    if missing:
        raise RuntimeError(f"Gaia DR3 metadata did not contain persisted source IDs: {missing}")

    root = Path(output_dir) / "current-gaia-counterpart-photometry"
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    prepared_series: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    for source in sources:
        role = str(source["sourceRole"])
        source_id = int(source["gaiaDR3SourceID"])
        source_metadata = metadata[source_id]
        position_match = _validate_position(source_metadata, source["raDeg"], source["decDeg"], label=role)
        record = {"sourceRole": role, "gaiaDR3SourceID": source_id,
                  "positionMatchArcsec": position_match, "metadata": source_metadata,
                  "epochState": "NO_EPOCH_PHOTOMETRY", "prepared": False}
        if not source_metadata.get("hasEpochPhotometry"):
            records.append(record)
            continue
        try:
            payload, content_type = download_epochs(source_id)
        except Exception as exc:
            raise GaiaArchiveUnavailable(
                f"Gaia DR3 epoch download failed for {role}: {type(exc).__name__}: {exc}"
            ) from exc
        raw_path = root / f"gaia-dr3-{source_id}-epoch-photometry.csv"
        raw_path.write_bytes(payload)
        try:
            times, flux, diagnostics = _parse_gaia_g_series(payload)
        except Exception as exc:
            message = str(exc)
            scientific_limit = (
                message.startswith("Only ")
                or message == "Gaia G-band epoch flux has no finite variability scale."
            )
            if not scientific_limit:
                raise GaiaArchiveUnavailable(
                    f"Gaia DR3 epoch response was unusable for {role}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            record.update({"epochState": "INSUFFICIENT_EPOCH_PHOTOMETRY",
                           "rawEpochPath": str(raw_path.resolve()),
                           "parseDiagnostic": f"{type(exc).__name__}: {exc}"})
            records.append(record)
            continue
        record.update({"epochState": "USABLE_EPOCH_PHOTOMETRY", "prepared": True,
                       "contentType": content_type, "rawEpochPath": str(raw_path.resolve()),
                       "diagnostics": diagnostics})
        dataset_id = f"{source_dataset_id}-gaia-dr3-current-{role}-g-v1"
        dataset_path = root / f"{_safe(dataset_id)}.json"
        dataset = {"id": dataset_id, "targetName": f"{source_dataset_id} Gaia DR3 {role} G epoch photometry",
                   "times": np.asarray(times, dtype=np.float32).tolist(),
                   "flux": np.asarray(flux, dtype=np.float32).tolist(),
                   "frequencySearch": search, "reference": {},
                   "science": {"sourceRole": role, "gaiaDR3SourceID": source_id,
                               "sourceResolvedArchive": "Gaia DR3 epoch photometry",
                               "tessDriftExtrapolated": False},
                   "source": {"mission": "Gaia", "dataRelease": GAIA_DATA_RELEASE, "band": "G",
                              "distributedSamples": len(times), "baselineDays": diagnostics["baselineDays"]}}
        _write_json(dataset_path, dataset)
        item = {"datasetID": dataset_id, "datasetPath": str(dataset_path.resolve()),
                "sourceRole": role, "gaiaDR3SourceID": source_id, "band": "G",
                "sampleCount": len(times), "baselineDays": diagnostics["baselineDays"],
                "referenceFrequency": reference_frequency, "rawEpochPath": str(raw_path.resolve())}
        prepared_series.append(item)
        datasets.append({"id": dataset_id, "path": str(dataset_path.resolve()),
                         "targetName": dataset["targetName"]})
        records.append(record)

    project_id = project_path = None
    if datasets:
        project_id = f"{source_project_id}.investigation.{_safe(investigation_id)}.current-gaia-counterpart-v1"
        manifest_path = root / f"{_safe(project_id)}.json"
        _write_json(manifest_path, {"id": project_id, "name": "Gaia DR3 source-resolved counterpart validation",
                                    "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID, "datasets": datasets})
        project_path = str(manifest_path.resolve())
    work_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {"version": "openstar.tess-current-gaia-counterpart-preparation.v1",
            "archive": "Gaia DR3 epoch photometry", "archiveAttempted": True,
            "available": bool(datasets), "projectID": project_id, "projectPath": project_path,
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "workerSemantics": "generic-lomb-scargle-on-source-resolved-gaia-g-epoch-series",
            "sourcePair": pair, "referenceFrequency": reference_frequency,
            "referencePeriodDays": 1.0 / reference_frequency, "frequencySearch": search,
            "tessDriftExtrapolated": False, "sourceRecords": records,
            "preparedSeries": prepared_series, "totalWorkUnits": len(datasets) * work_per_dataset}


def interpret_current_gaia_counterpart_project(
    *, project_status: dict[str, Any] | None, preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {str(item["datasetID"]): item for item in preparation.get("preparedSeries") or []}
    results = []
    for dataset in (project_status or {}).get("datasets") or []:
        item = prepared.get(str(dataset.get("datasetID") or dataset.get("id") or ""))
        if item is not None:
            results.append(_dataset_result(dataset, item))
    by_role = {str(item["sourceRole"]): item for item in results}
    target = by_role.get("target-control")
    counterpart = by_role.get("catalog-counterpart")
    target_recurs = bool(target and target.get("acceptedResidualBandVariability"))
    counterpart_recurs = bool(counterpart and counterpart.get("acceptedResidualBandVariability"))
    records = preparation.get("sourceRecords") or []
    states = {str(item.get("sourceRole")): item.get("epochState") for item in records}
    both_usable = all(
        states.get(role) == "USABLE_EPOCH_PHOTOMETRY"
        for role in ("target-control", "catalog-counterpart")
    )
    if counterpart_recurs and not target_recurs and both_usable:
        outcome = "COUNTERPART_RECURRENCE_SUPPORTED"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_recurs and not counterpart_recurs:
        outcome = "TARGET_CONTROL_RECURRENCE_ONLY"
        next_test = NEXT_ARCHIVE_TEST
    elif target_recurs and counterpart_recurs:
        outcome = "BOTH_SOURCES_SHOW_RECURRENCE"
        next_test = NEXT_ARCHIVE_TEST
    elif not any(state == "USABLE_EPOCH_PHOTOMETRY" for state in states.values()):
        outcome = ("GAIA_INSUFFICIENT_EPOCH_PHOTOMETRY"
                   if any(state == "INSUFFICIENT_EPOCH_PHOTOMETRY" for state in states.values())
                   else "GAIA_NO_EPOCH_PHOTOMETRY")
        next_test = NEXT_ARCHIVE_TEST
    elif results:
        outcome = "GAIA_USABLE_NO_RECURRENCE"
        next_test = NEXT_ARCHIVE_TEST
    else:
        outcome = "GAIA_INSUFFICIENT_EPOCH_PHOTOMETRY"
        next_test = NEXT_ARCHIVE_TEST
    return {"version": "openstar.tess-current-gaia-counterpart-interpretation.v1",
            "archive": preparation.get("archive"), "archiveAttempted": True,
            "archiveExhausted": next_test == NEXT_ARCHIVE_TEST,
            "externalDataState": "AVAILABLE", "sourcePair": preparation.get("sourcePair"),
            "frequencySearch": preparation.get("frequencySearch"), "tessDriftExtrapolated": False,
            "targetControl": target, "catalogCounterpartEvidence": counterpart,
            "componentResults": results, "sourceRecords": records, "classification": outcome,
            "counterpartRecurrenceSupported": outcome == "COUNTERPART_RECURRENCE_SUPPORTED",
            "physicalMechanismResolved": False, "claimLevelChanged": False,
            "recommendedNextTest": next_test,
            "qualityGuard": {"minimumGaiaGSamples": MIN_GAIA_G_SAMPLES,
                             "minimumPeakProminenceRatio": MIN_PEAK_PROMINENCE_RATIO}}
