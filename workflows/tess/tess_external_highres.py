from __future__ import annotations

import csv
import io
import json
import math
import urllib.parse
import urllib.request
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


GAIA_DATA_RELEASE = "Gaia DR3"
GAIA_DATALINK_ENDPOINT = "https://gea.esac.esa.int/data-server/data"
HTTP_TIMEOUT_SECONDS = 90
USER_AGENT = "OpenStar/20.19 external-high-resolution-variability-validation"
MIN_GAIA_G_SAMPLES = 20
MIN_PEAK_PROMINENCE_RATIO = 1.5
MAX_POSITION_MISMATCH_ARCSEC = 1.0


def _python_value(value: Any) -> Any:
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _row_value(row: Any, names: tuple[str, ...]) -> Any:
    colnames = set(getattr(row, "colnames", []) or [])
    for name in names:
        if name not in colnames:
            continue
        try:
            value = _python_value(row[name])
        except Exception:
            continue
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
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(math.acos(cosine)) * 3600.0


def _query_gaia_source_metadata(source_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not source_ids:
        return {}
    try:
        from astroquery.gaia import Gaia
    except Exception as exc:  # pragma: no cover - user's astronomy environment
        raise RuntimeError(
            "v20.19 requires astroquery.gaia for Gaia DR3 source metadata."
        ) from exc

    identifiers = ",".join(str(int(value)) for value in sorted(set(source_ids)))
    query = f"""
SELECT
    source_id,
    ra,
    dec,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    phot_variable_flag,
    has_epoch_photometry,
    duplicated_source,
    ruwe,
    ipd_frac_multi_peak,
    ipd_frac_odd_win,
    phot_bp_rp_excess_factor
FROM gaiadr3.gaia_source
WHERE source_id IN ({identifiers})
"""
    job = Gaia.launch_job(query)
    table = job.get_results()
    output: dict[int, dict[str, Any]] = {}
    for row in table:
        source_id = _int(_row_value(row, ("source_id", "SOURCE_ID")))
        if source_id is None:
            continue
        output[int(source_id)] = {
            "sourceID": int(source_id),
            "raDeg": _float(_row_value(row, ("ra", "RA"))),
            "decDeg": _float(_row_value(row, ("dec", "DEC"))),
            "gMag": _float(_row_value(row, ("phot_g_mean_mag", "PHOT_G_MEAN_MAG"))),
            "bpMag": _float(_row_value(row, ("phot_bp_mean_mag", "PHOT_BP_MEAN_MAG"))),
            "rpMag": _float(_row_value(row, ("phot_rp_mean_mag", "PHOT_RP_MEAN_MAG"))),
            "photVariableFlag": _python_value(
                _row_value(row, ("phot_variable_flag", "PHOT_VARIABLE_FLAG"))
            ),
            "hasEpochPhotometry": bool(
                _python_value(
                    _row_value(row, ("has_epoch_photometry", "HAS_EPOCH_PHOTOMETRY"))
                )
                or False
            ),
            "duplicatedSource": bool(
                _python_value(
                    _row_value(row, ("duplicated_source", "DUPLICATED_SOURCE"))
                )
                or False
            ),
            "ruwe": _float(_row_value(row, ("ruwe", "RUWE"))),
            "ipdFracMultiPeak": _float(
                _row_value(row, ("ipd_frac_multi_peak", "IPD_FRAC_MULTI_PEAK"))
            ),
            "ipdFracOddWin": _float(
                _row_value(row, ("ipd_frac_odd_win", "IPD_FRAC_ODD_WIN"))
            ),
            "photBpRpExcessFactor": _float(
                _row_value(
                    row,
                    ("phot_bp_rp_excess_factor", "PHOT_BP_RP_EXCESS_FACTOR"),
                )
            ),
        }
    return output


def _gaia_epoch_url(source_id: int) -> str:
    params = {
        "RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY",
        "ID": str(int(source_id)),
        "RELEASE": GAIA_DATA_RELEASE,
        "VALID_DATA": "TRUE",
        "FORMAT": "CSV",
        "DATA_STRUCTURE": "INDIVIDUAL",
    }
    return GAIA_DATALINK_ENDPOINT + "?" + urllib.parse.urlencode(params)


def _download_gaia_epoch_csv(source_id: int) -> tuple[bytes, str]:
    url = _gaia_epoch_url(source_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read()
        content_type = str(response.headers.get("Content-Type") or "")
    if not payload:
        raise RuntimeError(f"Gaia DataLink returned an empty epoch-photometry response for {source_id}.")
    return payload, content_type


def _normalized_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    normalized = {_normalized_key(key): value for key, value in row.items()}
    for name in names:
        key = _normalized_key(name)
        if key in normalized and str(normalized[key]).strip() != "":
            return normalized[key]
    return None


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_gaia_g_series(payload: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # The single-source INDIVIDUAL response is plain text CSV. Guard against an
    # HTML/XML service error so it cannot silently become a bogus light curve.
    text = payload.decode("utf-8-sig", errors="replace")
    prefix = text.lstrip()[:200].lower()
    if prefix.startswith("<") or "<html" in prefix or "<votable" in prefix:
        raise RuntimeError("Gaia DataLink did not return CSV epoch photometry.")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise RuntimeError("Gaia epoch-photometry CSV has no header.")

    times: list[float] = []
    fluxes: list[float] = []
    total_rows = 0
    g_rows = 0
    rejected_rows = 0
    for row in reader:
        total_rows += 1
        band = _first_present(row, ("band", "photometric_band"))
        if band is not None and str(band).strip().upper() != "G":
            continue
        g_rows += 1

        rejected_phot = _parse_bool(
            _first_present(row, ("rejected_by_photometry", "photometry_rejected"))
        )
        rejected_vari = _parse_bool(
            _first_present(row, ("rejected_by_variability", "variability_rejected"))
        )
        if rejected_phot is True or rejected_vari is True:
            rejected_rows += 1
            continue

        time = _parse_float(
            _first_present(
                row,
                ("time", "observation_time", "g_transit_time", "g_time"),
            )
        )
        flux = _parse_float(
            _first_present(
                row,
                ("flux", "g_transit_flux", "g_flux"),
            )
        )
        if time is None or flux is None or flux <= 0:
            continue
        times.append(time)
        fluxes.append(flux)

    if len(times) < MIN_GAIA_G_SAMPLES:
        raise RuntimeError(
            f"Only {len(times)} usable Gaia G-band epoch samples; need at least {MIN_GAIA_G_SAMPLES}."
        )

    time_array = np.asarray(times, dtype=np.float64)
    flux_array = np.asarray(fluxes, dtype=np.float64)
    order = np.argsort(time_array)
    time_array = time_array[order]
    flux_array = flux_array[order]

    scale = float(np.std(flux_array))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Gaia G-band epoch flux has no finite variability scale.")
    normalized_flux = (flux_array - float(np.mean(flux_array))) / scale
    local_times = time_array - float(time_array[0])

    return local_times, normalized_flux, {
        "totalRows": int(total_rows),
        "gBandRows": int(g_rows),
        "rejectedRows": int(rejected_rows),
        "usableSamples": int(len(local_times)),
        "baselineDays": float(local_times[-1] - local_times[0]),
        "gaiaTimeSystem": "BJD(TCB)-2455197.5 days, shifted locally to start at zero",
    }


def _validate_position(
    metadata: dict[str, Any],
    expected_ra: float | None,
    expected_dec: float | None,
    *,
    label: str,
) -> float | None:
    if expected_ra is None or expected_dec is None:
        return None
    actual_ra = _float(metadata.get("raDeg"))
    actual_dec = _float(metadata.get("decDeg"))
    if actual_ra is None or actual_dec is None:
        raise RuntimeError(f"Gaia metadata for {label} is missing RA/Dec.")
    separation = _angular_separation_arcsec(
        actual_ra,
        actual_dec,
        float(expected_ra),
        float(expected_dec),
    )
    if separation > MAX_POSITION_MISMATCH_ARCSEC:
        raise RuntimeError(
            f"Gaia source ID for {label} is {separation:.3f} arcsec from the frozen catalog position; "
            "refusing source-resolved validation with a mismatched identifier."
        )
    return float(separation)


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
    reference_frequency = _float(prepared.get("referenceFrequency"))
    relative_offset = None
    if frequency is not None and reference_frequency is not None and reference_frequency > 0:
        relative_offset = abs(frequency - reference_frequency) / reference_frequency

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
        "relativeFrequencyOffsetFromTessReference": relative_offset,
        "acceptedResidualBandVariability": accepted,
    }


def build_external_high_resolution_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    identity: dict[str, Any],
    offset_source_identification: dict[str, Any],
    official_spoc_prf_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if official_spoc_prf_summary.get("recommendedNextTest") != "EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION":
        raise RuntimeError(
            "v20.19 requires v20.18 to recommend EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION."
        )

    target_gaia = _int((((identity.get("gaiaDR3") or {}).get("nearest") or {}).get("sourceID")))
    target_meta_frozen = ((identity.get("gaiaDR3") or {}).get("nearest") or {})
    candidate = offset_source_identification.get("bestCandidate") or {}
    candidate_ids = candidate.get("catalogIDs") or {}
    counterpart_gaia = _int(
        (official_spoc_prf_summary.get("catalogCounterpart") or {}).get("gaiaDR3SourceID")
    )
    if counterpart_gaia is None:
        counterpart_gaia = _int(candidate_ids.get("gaiaDR3SourceID"))

    if target_gaia is None or counterpart_gaia is None:
        raise RuntimeError(
            "v20.19 requires resolved Gaia DR3 source IDs for both Blind C and the catalog counterpart."
        )
    if int(target_gaia) == int(counterpart_gaia):
        raise RuntimeError("v20.19 target and counterpart resolve to the same Gaia DR3 source ID.")

    source_definitions = [
        {
            "sourceRole": "target-control",
            "gaiaDR3SourceID": int(target_gaia),
            "expectedRaDeg": _float(target_meta_frozen.get("raDeg")),
            "expectedDecDeg": _float(target_meta_frozen.get("decDeg")),
        },
        {
            "sourceRole": "catalog-counterpart",
            "gaiaDR3SourceID": int(counterpart_gaia),
            "expectedRaDeg": _float(candidate.get("raDeg")),
            "expectedDecDeg": _float(candidate.get("decDeg")),
        },
    ]

    print("   querying Gaia DR3 source metadata for the frozen target/counterpart pair", flush=True)
    metadata_by_id = _query_gaia_source_metadata(
        [int(target_gaia), int(counterpart_gaia)]
    )
    if len(metadata_by_id) != 2:
        missing = sorted(
            {int(target_gaia), int(counterpart_gaia)} - set(metadata_by_id)
        )
        raise RuntimeError(f"Gaia DR3 metadata query did not return source IDs: {missing}")

    search = dict(
        ((official_spoc_prf_summary.get("distributedValidation") or {}).get("frequencySearch") or {})
    )
    if not search:
        raise RuntimeError("v20.19 requires the v20.18 residual-frequency search definition.")
    total_frequencies = _int(search.get("totalFrequencies"))
    per_work = _int(search.get("frequenciesPerWorkUnit"))
    if total_frequencies is None or per_work is None or total_frequencies <= 0 or per_work <= 0:
        raise RuntimeError("v20.18 frequencySearch is missing total/per-work frequency counts.")

    reference_frequency = _float(official_spoc_prf_summary.get("referenceFrequency"))
    root = Path(output_dir) / "external-high-resolution-variability"
    root.mkdir(parents=True, exist_ok=True)

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for source in source_definitions:
        role = str(source["sourceRole"])
        source_id = int(source["gaiaDR3SourceID"])
        metadata = dict(metadata_by_id[source_id])
        position_match = _validate_position(
            metadata,
            source.get("expectedRaDeg"),
            source.get("expectedDecDeg"),
            label=role,
        )
        record: dict[str, Any] = {
            "sourceRole": role,
            "gaiaDR3SourceID": source_id,
            "metadata": metadata,
            "positionMatchArcsec": position_match,
            "prepared": False,
        }

        if not metadata.get("hasEpochPhotometry"):
            record["availability"] = "NO_GAIA_DR3_EPOCH_PHOTOMETRY"
            source_records.append(record)
            print(f"      {role}: Gaia DR3 has_epoch_photometry=False", flush=True)
            continue

        print(f"      {role}: downloading Gaia DR3 epoch photometry for {source_id}", flush=True)
        try:
            payload, content_type = _download_gaia_epoch_csv(source_id)
            raw_path = root / f"gaia-dr3-{source_id}-epoch-photometry.csv"
            raw_path.write_bytes(payload)
            times, flux, diagnostics = _parse_gaia_g_series(payload)

            dataset_id = f"{source_dataset_id}-gaia-dr3-{role}-g-v1"
            dataset_path = root / f"{_safe(dataset_id)}.json"
            target_name = f"{source_dataset_id} Gaia DR3 {role} G-band epoch photometry"
            dataset = {
                "id": dataset_id,
                "targetName": target_name,
                "times": np.asarray(times, dtype=np.float32).tolist(),
                "flux": np.asarray(flux, dtype=np.float32).tolist(),
                "frequencySearch": search,
                "reference": {},
                "science": {
                    "role": "external-high-resolution-variability-validation",
                    "sourceRole": role,
                    "gaiaDR3SourceID": source_id,
                    "band": "G",
                    "tessReferenceFrequency": reference_frequency,
                    "tessReferencePeriodDays": (
                        float(1.0 / reference_frequency)
                        if reference_frequency is not None and reference_frequency > 0
                        else None
                    ),
                    "driftExtrapolatedToGaiaEpoch": False,
                    "sourceResolvedArchive": "Gaia DR3 epoch photometry",
                },
                "source": {
                    "mission": "Gaia",
                    "dataRelease": GAIA_DATA_RELEASE,
                    "band": "G",
                    "distributedSamples": int(len(times)),
                    "baselineDays": diagnostics["baselineDays"],
                    "rawEpochPath": str(raw_path.resolve()),
                },
            }
            _write_json(dataset_path, dataset)
            prepared = {
                "datasetID": dataset_id,
                "datasetPath": str(dataset_path.resolve()),
                "sourceRole": role,
                "gaiaDR3SourceID": source_id,
                "band": "G",
                "sampleCount": int(len(times)),
                "baselineDays": diagnostics["baselineDays"],
                "referenceFrequency": reference_frequency,
                "rawEpochPath": str(raw_path.resolve()),
            }
            prepared_series.append(prepared)
            dataset_entries.append(
                {"id": dataset_id, "path": str(dataset_path.resolve()), "targetName": target_name}
            )
            record.update(
                {
                    "prepared": True,
                    "availability": "GAIA_DR3_EPOCH_PHOTOMETRY_AVAILABLE",
                    "contentType": content_type,
                    "rawEpochPath": str(raw_path.resolve()),
                    "diagnostics": diagnostics,
                    "datasetID": dataset_id,
                    "datasetPath": str(dataset_path.resolve()),
                }
            )
            print(
                f"         G-band samples={len(times)}, baseline={diagnostics['baselineDays']:.1f} days",
                flush=True,
            )
        except Exception as exc:
            record["availability"] = "GAIA_DR3_EPOCH_PHOTOMETRY_PREPARATION_FAILED"
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(
                {"sourceRole": role, "gaiaDR3SourceID": source_id, "error": record["error"]}
            )
            print(f"         unavailable: {type(exc).__name__}: {exc}", flush=True)
        source_records.append(record)

    project_path: str | None = None
    project_id: str | None = None
    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "external-high-resolution-gaia-dr3-v1"
        )
        manifest = {
            "id": project_id,
            "name": f"{source_project_id} — Gaia DR3 source-resolved residual-band validation",
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": "external-high-resolution-variability-validation",
                "archive": "Gaia DR3 epoch photometry",
                "workerSemantics": (
                    "Each dataset is source-resolved Gaia DR3 G-band epoch photometry. "
                    "Workers execute ordinary Lomb-Scargle over the frozen TESS residual-frequency band."
                ),
                "driftExtrapolatedToGaiaEpoch": False,
            },
        }
        manifest_path = root / f"{_safe(project_id)}.json"
        _write_json(manifest_path, manifest)
        project_path = str(manifest_path.resolve())

    work_units_per_dataset = math.ceil(total_frequencies / per_work)
    return {
        "available": bool(dataset_entries),
        "version": "openstar.tess-external-high-resolution-variability-preparation.v1",
        "archive": "Gaia DR3 epoch photometry",
        "dataRelease": GAIA_DATA_RELEASE,
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-source-resolved-gaia-dr3-g-band-epoch-photometry",
        "sourcePair": {
            "targetGaiaDR3SourceID": int(target_gaia),
            "counterpartGaiaDR3SourceID": int(counterpart_gaia),
            "catalogSeparationArcsec": _float(
                (official_spoc_prf_summary.get("catalogCounterpart") or {}).get(
                    "catalogSeparationArcsec"
                )
            ),
        },
        "referenceFrequency": reference_frequency,
        "referencePeriodDays": _float(official_spoc_prf_summary.get("referencePeriodDays")),
        "frequencySearch": search,
        "driftExtrapolatedToGaiaEpoch": False,
        "preparedSeries": prepared_series,
        "sourceRecords": source_records,
        "errors": errors,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "interpretationGuard": (
            "Gaia DR3 epoch photometry is source-resolved external evidence. The TESS v20.9 drift law is not "
            "extrapolated years backward; Gaia data are searched only within the frozen residual-frequency band. "
            "Absence of Gaia epoch photometry or absence of a Gaia-band signal is not evidence that a source was "
            "non-variable during the TESS sectors."
        ),
    }


def interpret_external_high_resolution_project(
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
            if prepared is None:
                continue
            results.append(_dataset_result(dataset, prepared))

    by_role = {str(item.get("sourceRole")): item for item in results}
    target = by_role.get("target-control")
    counterpart = by_role.get("catalog-counterpart")
    target_accepted = bool(target and target.get("acceptedResidualBandVariability"))
    counterpart_accepted = bool(
        counterpart and counterpart.get("acceptedResidualBandVariability")
    )

    prepared_roles = {
        str(item.get("sourceRole")) for item in preparation.get("preparedSeries") or []
    }
    both_available = {"target-control", "catalog-counterpart"}.issubset(prepared_roles)

    if not prepared_roles:
        classification = "GAIA_DR3_EPOCH_PHOTOMETRY_UNAVAILABLE"
        origin = "UNRESOLVED_EXTERNAL_ARCHIVE_NO_SOURCE_RESOLVED_EPOCH_DATA"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif not both_available:
        classification = "GAIA_DR3_SOURCE_RESOLVED_COVERAGE_INCOMPLETE"
        origin = "UNRESOLVED_EXTERNAL_ARCHIVE_INCOMPLETE_SOURCE_PAIR"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif counterpart_accepted and not target_accepted:
        classification = "GAIA_DR3_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_BY_EXTERNAL_SOURCE_RESOLVED_PHOTOMETRY"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_accepted and not counterpart_accepted:
        classification = "GAIA_DR3_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_SUPPORTED_BY_EXTERNAL_SOURCE_RESOLVED_PHOTOMETRY"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif target_accepted and counterpart_accepted:
        classification = "GAIA_DR3_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_AND_COUNTERPART_SUPPORTED_BY_EXTERNAL_SOURCE_RESOLVED_PHOTOMETRY"
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    else:
        classification = "GAIA_DR3_SOURCE_RESOLVED_VARIABILITY_UNRESOLVED"
        origin = "EXTERNAL_SOURCE_RESOLVED_VALIDATION_UNRESOLVED"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"

    return {
        "version": "openstar.tess-external-high-resolution-variability-validation.v1",
        "archive": preparation.get("archive"),
        "dataRelease": preparation.get("dataRelease"),
        "sourcePair": preparation.get("sourcePair"),
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "referenceFrequency": preparation.get("referenceFrequency"),
        "referencePeriodDays": preparation.get("referencePeriodDays"),
        "driftExtrapolatedToGaiaEpoch": False,
        "sourceRecords": preparation.get("sourceRecords") or [],
        "componentResults": results,
        "targetControl": target,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "preparationErrors": preparation.get("errors") or [],
        "interpretationGuard": preparation.get("interpretationGuard"),
    }
