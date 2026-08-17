from __future__ import annotations

import csv
import io
import math
import urllib.parse
import urllib.error
import urllib.request
from collections import defaultdict
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

NSC_RELEASE = "DR2"
NSC_QUERY_URL = "https://datalab.noirlab.edu/query/query"
HTTP_TIMEOUT_SECONDS = 90
USER_AGENT = "OpenStar/20.21 NOIRLab-NSC-resolved-epoch-photometry"

SUPPORTED_BANDS = ("g", "r", "i", "z")
MAX_OBJECT_MATCH_ARCSEC = 0.80
MAX_MEASUREMENT_MATCH_ARCSEC = 0.85
MIN_PAIR_SEPARATION_FRACTION = 0.60
MAX_MAG_ERROR = 0.15
MIN_BAND_SAMPLES = 10
MIN_PEAK_PROMINENCE_RATIO = 2.0
MIN_CROSS_BAND_SUPPORT = 2
MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD = 0.12
MAX_ROWS = 10000
CURRENT_TRIGGER = "NSC_RESOLVED_COUNTERPART_PHOTOMETRY"
HISTORICAL_TRIGGER = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
NEXT_ARCHIVE_TEST = "NOIRLAB_IMAGE_LEVEL_FORCED_PHOTOMETRY"


class NSCArchiveUnavailable(RuntimeError):
    """The NSC service failed transiently before a scientific result existed."""


def _retryable_service_error(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, (TimeoutError, ConnectionError, urllib.error.URLError)):
            if isinstance(current, urllib.error.HTTPError):
                return current.code in {408, 425, 429} or current.code >= 500
            return True
        module = type(current).__module__.lower()
        name = type(current).__name__.lower()
        if module.startswith(("requests.", "httpx", "urllib3.")) and any(
            token in name for token in ("timeout", "connection", "network")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _archive_query(query, *args):
    try:
        return query(*args)
    except Exception as exc:
        if _retryable_service_error(exc):
            raise NSCArchiveUnavailable(
                f"NSC DR2 service unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        raise


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    numpy_scalars = tuple(
        kind for kind in (getattr(np, "integer", None), getattr(np, "floating", None))
        if isinstance(kind, type)
    )
    if numpy_scalars and isinstance(value, numpy_scalars):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_float(value: Any) -> float | None:
    value = _python_value(value)
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_int(value: Any) -> int | None:
    value = _python_value(value)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
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
    cos_sep = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep)) * 3600.0


def _query_csv(sql: str) -> list[dict[str, str]]:
    query = urllib.parse.urlencode(
        {
            "sql": sql,
            "ofmt": "csv",
            "out": "",
            "async": "False",
            "drop": "False",
            "profile": "default",
        }
    )
    request = urllib.request.Request(
        f"{NSC_QUERY_URL}?{query}",
        headers={
            "User-Agent": USER_AGENT,
            "X-DL-AuthToken": "anonymous.0.0.anon_access",
        },
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        body = response.read()
        content_type = str(response.headers.get("Content-Type") or "")
    if not body:
        raise RuntimeError("NOIRLab Data Lab returned an empty query response.")
    text = body.decode("utf-8-sig", errors="replace")
    prefix = text.lstrip()[:300].lower()
    if prefix.startswith("<") or "<html" in prefix:
        raise RuntimeError(
            f"NOIRLab Data Lab did not return CSV (Content-Type={content_type or '[none]'})."
        )
    if text.lower().startswith("error"):
        raise RuntimeError(f"NOIRLab Data Lab query error: {text[:500]}")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise RuntimeError("NOIRLab Data Lab CSV has no header.")
    return [dict(row) for row in reader]


def _source_records_by_role(
    external_high_resolution_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in external_high_resolution_summary.get("sourceRecords") or []:
        role = str(item.get("sourceRole") or "")
        if role:
            records[role] = item
    return records


def _frozen_source_pair(
    external_high_resolution_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    pair = external_high_resolution_summary.get("sourcePair") or {}
    if pair.get("version") == "openstar.current-source-pair.v1":
        definitions = []
        for key, expected_role in (
            ("target", "target-control"),
            ("counterpart", "catalog-counterpart"),
        ):
            source = pair.get(key) or {}
            source_id = _int(source.get("gaiaDR3SourceID"))
            ra = _float(source.get("raDeg"))
            dec = _float(source.get("decDeg"))
            if source_id is None or ra is None or dec is None:
                raise RuntimeError(f"Current sourcePair lacks Gaia ID/coordinates for {key}.")
            definitions.append({
                "sourceRole": str(source.get("sourceRole") or expected_role),
                "gaiaDR3SourceID": int(source_id), "raDeg": float(ra), "decDeg": float(dec),
            })
        separation = _float(pair.get("separationArcsec"))
        if separation is None or separation <= 0:
            separation = _angular_separation_arcsec(
                definitions[0]["raDeg"], definitions[0]["decDeg"],
                definitions[1]["raDeg"], definitions[1]["decDeg"],
            )
        return definitions, float(separation)

    records = _source_records_by_role(external_high_resolution_summary)
    target_id = _int(pair.get("targetGaiaDR3SourceID"))
    counterpart_id = _int(pair.get("counterpartGaiaDR3SourceID"))
    if target_id is None or counterpart_id is None:
        raise RuntimeError("v20.21 requires both frozen Gaia DR3 source IDs from v20.19.")

    definitions: list[dict[str, Any]] = []
    for role, source_id in (
        ("target-control", int(target_id)),
        ("catalog-counterpart", int(counterpart_id)),
    ):
        record = records.get(role) or {}
        metadata = record.get("metadata") or {}
        ra = _float(metadata.get("raDeg"))
        dec = _float(metadata.get("decDeg"))
        if ra is None or dec is None:
            raise RuntimeError(
                f"v20.21 requires frozen Gaia RA/Dec metadata for {role} from v20.19."
            )
        definitions.append(
            {
                "sourceRole": role,
                "gaiaDR3SourceID": source_id,
                "raDeg": float(ra),
                "decDeg": float(dec),
            }
        )

    separation = _float(pair.get("catalogSeparationArcsec"))
    if separation is None or separation <= 0:
        separation = _angular_separation_arcsec(
            definitions[0]["raDeg"],
            definitions[0]["decDeg"],
            definitions[1]["raDeg"],
            definitions[1]["decDeg"],
        )
    return definitions, float(separation)


def _query_object_candidates(source: dict[str, Any]) -> list[dict[str, str]]:
    ra = float(source["raDeg"])
    dec = float(source["decDeg"])
    radius_arcsec = 2.0
    dec_half = radius_arcsec / 3600.0
    cos_dec = max(0.05, abs(math.cos(math.radians(dec))))
    ra_half = radius_arcsec / 3600.0 / cos_dec
    sql = f"""
SELECT id, ra, dec, ndet, class_star, flags, gmag, rmag, imag, zmag
FROM nsc_dr2.object
WHERE ra BETWEEN {ra - ra_half:.10f} AND {ra + ra_half:.10f}
  AND dec BETWEEN {dec - dec_half:.10f} AND {dec + dec_half:.10f}
LIMIT 100
"""
    return _query_csv(sql)


def _select_object_match(
    rows: list[dict[str, str]],
    source: dict[str, Any],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        object_id = row.get("id")
        ra = _parse_float(row.get("ra"))
        dec = _parse_float(row.get("dec"))
        if not object_id or ra is None or dec is None:
            continue
        separation = _angular_separation_arcsec(
            ra,
            dec,
            float(source["raDeg"]),
            float(source["decDeg"]),
        )
        if separation > MAX_OBJECT_MATCH_ARCSEC:
            continue
        candidates.append(
            {
                "objectID": str(object_id),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                "gaiaDistanceArcsec": float(separation),
                "ndet": _parse_int(row.get("ndet")),
                "classStar": _parse_float(row.get("class_star")),
                "flags": _parse_int(row.get("flags")),
                "meanMagnitudes": {
                    "g": _parse_float(row.get("gmag")),
                    "r": _parse_float(row.get("rmag")),
                    "i": _parse_float(row.get("imag")),
                    "z": _parse_float(row.get("zmag")),
                },
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["gaiaDistanceArcsec"])


def _sql_literal_object_id(value: str) -> str:
    text = str(value).strip()
    if text.replace(".", "", 1).isdigit():
        return text
    return "'" + text.replace("'", "''") + "'"


def _query_measurements(object_ids: list[str]) -> list[dict[str, str]]:
    identifiers = ",".join(_sql_literal_object_id(value) for value in object_ids)
    bands = ",".join(f"'{band}'" for band in SUPPORTED_BANDS)
    sql = f"""
SELECT objectid, filter, ra, dec, mag_auto, magerr_auto, mjd, exposure
FROM nsc_dr2.meas
WHERE objectid IN ({identifiers})
  AND filter IN ({bands})
ORDER BY mjd
LIMIT {MAX_ROWS}
"""
    return _query_csv(sql)


def _robust_standardize_magnitudes(magnitudes: np.ndarray) -> np.ndarray:
    values = np.asarray(magnitudes, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("NSC magnitudes have no finite variability scale.")
    return -(values - median) / scale


def _standardize_series_or_quality_outcome(
    magnitudes: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Return a scientific quality outcome only for the known zero-scale case."""
    try:
        return _robust_standardize_magnitudes(magnitudes), None
    except RuntimeError as exc:
        if str(exc) != "NSC magnitudes have no finite variability scale.":
            raise
        return None, "NO_FINITE_VARIABILITY_SCALE"


def _closest_valid_measurement(
    rows: list[dict[str, str]],
    *,
    source: dict[str, Any],
    object_id: str,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in rows:
        if str(row.get("objectid") or "").strip() != str(object_id):
            continue
        ra = _parse_float(row.get("ra"))
        dec = _parse_float(row.get("dec"))
        mjd = _parse_float(row.get("mjd"))
        magnitude = _parse_float(row.get("mag_auto"))
        magnitude_error = _parse_float(row.get("magerr_auto"))
        if (
            ra is None
            or dec is None
            or mjd is None
            or magnitude is None
            or magnitude_error is None
            or magnitude_error <= 0
            or magnitude_error > MAX_MAG_ERROR
        ):
            continue
        position_distance = _angular_separation_arcsec(
            ra,
            dec,
            float(source["raDeg"]),
            float(source["decDeg"]),
        )
        if position_distance > MAX_MEASUREMENT_MATCH_ARCSEC:
            continue
        candidate = {
            "raDeg": float(ra),
            "decDeg": float(dec),
            "mjd": float(mjd),
            "magnitude": float(magnitude),
            "magnitudeError": float(magnitude_error),
            "positionDistanceArcsec": float(position_distance),
        }
        if best is None or candidate["positionDistanceArcsec"] < best["positionDistanceArcsec"]:
            best = candidate
    return best


def _prepare_codetected_series(
    rows: list[dict[str, str]],
    *,
    sources: list[dict[str, Any]],
    matches: dict[str, dict[str, Any]],
    pair_separation_arcsec: float,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    by_exposure_band: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        band = str(row.get("filter") or "").strip().lower()
        exposure = str(row.get("exposure") or "").strip()
        if band in SUPPORTED_BANDS and exposure:
            by_exposure_band[(exposure, band)].append(row)

    source_by_role = {str(item["sourceRole"]): item for item in sources}
    minimum_measured_separation = float(pair_separation_arcsec) * MIN_PAIR_SEPARATION_FRACTION
    accepted_pairs: dict[str, list[dict[str, Any]]] = {band: [] for band in SUPPORTED_BANDS}
    diagnostics = {
        band: {
            "exposureGroups": 0,
            "bothObjectsMeasured": 0,
            "bothPositionsMatched": 0,
            "pairSeparationPassed": 0,
            "usableCoDetections": 0,
        }
        for band in SUPPORTED_BANDS
    }
    unusable_series: list[dict[str, Any]] = []

    for (_, band), group_rows in by_exposure_band.items():
        diag = diagnostics[band]
        diag["exposureGroups"] += 1
        target_source = source_by_role["target-control"]
        counterpart_source = source_by_role["catalog-counterpart"]
        target = _closest_valid_measurement(
            group_rows,
            source=target_source,
            object_id=str(matches["target-control"]["objectID"]),
        )
        counterpart = _closest_valid_measurement(
            group_rows,
            source=counterpart_source,
            object_id=str(matches["catalog-counterpart"]["objectID"]),
        )
        if target is None or counterpart is None:
            continue
        diag["bothObjectsMeasured"] += 1
        diag["bothPositionsMatched"] += 1

        measured_separation = _angular_separation_arcsec(
            target["raDeg"],
            target["decDeg"],
            counterpart["raDeg"],
            counterpart["decDeg"],
        )
        if measured_separation < minimum_measured_separation:
            continue
        diag["pairSeparationPassed"] += 1

        accepted_pairs[band].append(
            {
                "mjd": float((target["mjd"] + counterpart["mjd"]) / 2.0),
                "measuredPairSeparationArcsec": float(measured_separation),
                "target": target,
                "counterpart": counterpart,
            }
        )
        diag["usableCoDetections"] += 1

    result: dict[str, dict[str, dict[str, Any]]] = {
        "target-control": {},
        "catalog-counterpart": {},
    }
    for band, pairs in accepted_pairs.items():
        if len(pairs) < MIN_BAND_SAMPLES:
            continue
        pairs.sort(key=lambda item: item["mjd"])
        times = np.asarray([item["mjd"] for item in pairs], dtype=np.float64)
        for role in ("target-control", "catalog-counterpart"):
            mags = np.asarray(
                [item["target" if role == "target-control" else "counterpart"]["magnitude"] for item in pairs],
                dtype=np.float64,
            )
            errors = np.asarray(
                [item["target" if role == "target-control" else "counterpart"]["magnitudeError"] for item in pairs],
                dtype=np.float64,
            )
            positions = np.asarray(
                [item["target" if role == "target-control" else "counterpart"]["positionDistanceArcsec"] for item in pairs],
                dtype=np.float64,
            )
            median = float(np.median(mags))
            mad = float(np.median(np.abs(mags - median)))
            keep = np.ones(len(mags), dtype=bool)
            if math.isfinite(mad) and mad > 0:
                keep = np.abs(mags - median) <= 5.0 * 1.4826 * mad
            kept_times = times[keep]
            kept_mags = mags[keep]
            kept_errors = errors[keep]
            kept_positions = positions[keep]
            kept_pairs = [pair for pair, accepted in zip(pairs, keep.tolist()) if accepted]
            if len(kept_times) < MIN_BAND_SAMPLES:
                continue
            flux, quality_outcome = _standardize_series_or_quality_outcome(kept_mags)
            if flux is None:
                unusable_series.append({
                    "sourceRole": role,
                    "band": band,
                    "sampleCount": int(len(kept_times)),
                    "reason": quality_outcome,
                })
                continue
            local_times = kept_times - float(kept_times[0])
            result[role][band] = {
                "times": local_times,
                "flux": flux,
                "sampleCount": int(len(local_times)),
                "baselineDays": float(local_times[-1] - local_times[0]),
                "medianMagnitudeError": float(np.median(kept_errors)),
                "medianPositionDistanceArcsec": float(np.median(kept_positions)),
                "medianMeasuredPairSeparationArcsec": float(
                    np.median([item["measuredPairSeparationArcsec"] for item in kept_pairs])
                ),
                "qualityDiagnostics": diagnostics[band],
            }

    return result, {
        "minimumMeasuredPairSeparationArcsec": minimum_measured_separation,
        "bandDiagnostics": diagnostics,
        "unusableSeries": unusable_series,
    }


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
    return {
        "datasetID": prepared.get("datasetID"),
        "sourceRole": prepared.get("sourceRole"),
        "gaiaDR3SourceID": prepared.get("gaiaDR3SourceID"),
        "nscObjectID": prepared.get("nscObjectID"),
        "band": prepared.get("band"),
        "sampleCount": prepared.get("sampleCount"),
        "baselineDays": prepared.get("baselineDays"),
        "periodStatus": status or None,
        "periodConfidence": project_dataset.get("periodConfidence"),
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "acceptedResidualBandVariability": accepted,
    }


def _summarize_source(role: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    source_results = [item for item in results if item.get("sourceRole") == role]
    accepted = [item for item in source_results if item.get("acceptedResidualBandVariability")]
    frequencies = [
        float(item["candidateFrequency"])
        for item in accepted
        if item.get("candidateFrequency") is not None
    ]
    median_frequency = None
    relative_spread = None
    supported = False
    if frequencies:
        median_frequency = float(np.median(np.asarray(frequencies, dtype=np.float64)))
        if median_frequency > 0:
            relative_spread = float((max(frequencies) - min(frequencies)) / median_frequency)
        supported = bool(
            len(frequencies) >= MIN_CROSS_BAND_SUPPORT
            and relative_spread is not None
            and relative_spread <= MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD
        )
    return {
        "sourceRole": role,
        "bandResults": source_results,
        "acceptedBands": sorted(str(item.get("band")) for item in accepted),
        "acceptedBandCount": len(accepted),
        "medianAcceptedFrequency": median_frequency,
        "crossBandRelativeFrequencySpread": relative_spread,
        "sourceSupported": supported,
        "sourceSuggestive": bool(accepted) and not supported,
    }


def build_nsc_resolved_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    external_high_resolution_summary: dict[str, Any],
    skymapper_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if skymapper_summary.get("recommendedNextTest") not in {CURRENT_TRIGGER, HISTORICAL_TRIGGER}:
        raise RuntimeError(
            "v20.21 requires v20.20 to leave the investigation at targeted high-resolution follow-up."
        )

    sources, pair_separation = _frozen_source_pair(external_high_resolution_summary)
    search = dict(
        external_high_resolution_summary.get("frequencySearch")
        or (external_high_resolution_summary.get("distributedValidation") or {}).get("frequencySearch")
        or skymapper_summary.get("frequencySearch")
        or (skymapper_summary.get("distributedValidation") or {}).get("frequencySearch")
        or {}
    )
    total_frequencies = _int(search.get("totalFrequencies"))
    per_work = _int(search.get("frequenciesPerWorkUnit"))
    if not search or total_frequencies is None or per_work is None or total_frequencies <= 0 or per_work <= 0:
        raise RuntimeError("v20.21 requires the frozen residual-frequency search definition from v20.19.")

    print("   matching the frozen Gaia pair against NSC DR2 objects", flush=True)
    matches: dict[str, dict[str, Any] | None] = {}
    errors: list[dict[str, Any]] = []
    for source in sources:
        role = str(source["sourceRole"])
        rows = _archive_query(_query_object_candidates, source)
        match = _select_object_match(rows, source)
        matches[role] = match
        if match is None:
            print(f"      {role}: no NSC DR2 object within {MAX_OBJECT_MATCH_ARCSEC:.2f} arcsec", flush=True)
        else:
            print(
                f"      {role}: NSC object {match['objectID']} | Gaia distance={match['gaiaDistanceArcsec']:.3f} arcsec | ndet={match.get('ndet')}",
                flush=True,
            )

    target_match = matches.get("target-control")
    counterpart_match = matches.get("catalog-counterpart")
    pair_separately_resolved = bool(
        target_match is not None
        and counterpart_match is not None
        and str(target_match["objectID"]) != str(counterpart_match["objectID"])
    )

    observed_object_separation = None
    if pair_separately_resolved:
        observed_object_separation = _angular_separation_arcsec(
            float(target_match["raDeg"]),
            float(target_match["decDeg"]),
            float(counterpart_match["raDeg"]),
            float(counterpart_match["decDeg"]),
        )
        if observed_object_separation < pair_separation * MIN_PAIR_SEPARATION_FRACTION:
            pair_separately_resolved = False

    root = Path(output_dir) / "nsc-resolved-photometry"
    root.mkdir(parents=True, exist_ok=True)

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    co_detection_diagnostics: dict[str, Any] = {}

    if pair_separately_resolved:
        object_ids = [str(target_match["objectID"]), str(counterpart_match["objectID"])]
        print("   loading NSC per-exposure measurements for both resolved objects", flush=True)
        rows = _archive_query(_query_measurements, object_ids)
        source_series, co_detection_diagnostics = _prepare_codetected_series(
            rows, sources=sources,
            matches={"target-control": target_match, "catalog-counterpart": counterpart_match},
            pair_separation_arcsec=pair_separation,
        )

        source_by_role = {str(item["sourceRole"]): item for item in sources}
        for role in ("target-control", "catalog-counterpart"):
            source = source_by_role[role]
            match = target_match if role == "target-control" else counterpart_match
            for band, series in source_series.get(role, {}).items():
                dataset_id = f"{source_dataset_id}-nsc-dr2-{role}-{band}-v1"
                target_name = f"{source_dataset_id} NSC DR2 {role} {band}-band"
                dataset_path = root / f"{_safe(dataset_id)}.json"
                dataset = {
                    "id": dataset_id,
                    "targetName": target_name,
                    "times": np.asarray(series["times"], dtype=np.float32).tolist(),
                    "flux": np.asarray(series["flux"], dtype=np.float32).tolist(),
                    "frequencySearch": search,
                    "reference": {},
                    "science": {
                        "role": "nsc-dr2-resolved-co-detected-photometry-screen",
                        "sourceRole": role,
                        "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                        "nscObjectID": str(match["objectID"]),
                        "band": band,
                        "pairSeparationArcsec": float(pair_separation),
                        "tessDriftExtrapolated": False,
                    },
                    "source": {
                        "mission": "NOIRLab Source Catalog",
                        "dataRelease": NSC_RELEASE,
                        "band": band,
                        "distributedSamples": int(series["sampleCount"]),
                        "baselineDays": float(series["baselineDays"]),
                    },
                }
                _write_json(dataset_path, dataset)
                prepared = {
                    "datasetID": dataset_id,
                    "datasetPath": str(dataset_path.resolve()),
                    "sourceRole": role,
                    "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                    "nscObjectID": str(match["objectID"]),
                    "band": band,
                    "sampleCount": int(series["sampleCount"]),
                    "baselineDays": float(series["baselineDays"]),
                    "medianMagnitudeError": float(series["medianMagnitudeError"]),
                    "medianPositionDistanceArcsec": float(series["medianPositionDistanceArcsec"]),
                    "medianMeasuredPairSeparationArcsec": float(series["medianMeasuredPairSeparationArcsec"]),
                    "qualityDiagnostics": series["qualityDiagnostics"],
                }
                prepared_series.append(prepared)
                dataset_entries.append(
                    {"id": dataset_id, "path": str(dataset_path.resolve()), "targetName": target_name}
                )
                print(
                    f"      {role} {band}: {series['sampleCount']} co-detected resolved epochs | baseline={series['baselineDays']:.1f} d",
                    flush=True,
                )
    else:
        print(
            "   NSC DR2 does not provide two safely distinct object matches for the frozen pair; no variability claim will be attempted",
            flush=True,
        )

    project_id: str | None = None
    project_path: str | None = None
    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "nsc-dr2-resolved-photometry-v1"
        )
        manifest = {
            "id": project_id,
            "name": f"{source_project_id} — NSC DR2 resolved residual-band screen",
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": "nsc-dr2-resolved-co-detected-photometry-screen",
                "archive": "NOIRLab Source Catalog DR2",
                "workerSemantics": (
                    "Each dataset contains one source from a frozen Gaia pair, using only NSC epochs where both "
                    "distinct NSC objects are independently position-matched in the same exposure/filter. "
                    "Workers execute ordinary Lomb-Scargle over the frozen residual-frequency band."
                ),
                "tessDriftExtrapolated": False,
            },
        }
        manifest_path = root / f"{_safe(project_id)}.json"
        _write_json(manifest_path, manifest)
        project_path = str(manifest_path.resolve())

    work_units_per_dataset = math.ceil(total_frequencies / per_work)
    return {
        "available": bool(dataset_entries),
        "version": "openstar.tess-nsc-dr2-resolved-photometry-preparation.v1",
        "archive": "NOIRLab Source Catalog DR2",
        "dataRelease": NSC_RELEASE,
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-nsc-dr2-resolved-co-detected-single-band-series",
        "sourcePair": external_high_resolution_summary.get("sourcePair"),
        "pairSeparationArcsec": float(pair_separation),
        "pairSeparatelyResolvedInNSC": bool(pair_separately_resolved),
        "observedNSCObjectSeparationArcsec": observed_object_separation,
        "objectMatches": matches,
        "frequencySearch": search,
        "tessDriftExtrapolated": False,
        "preparedSeries": prepared_series,
        "coDetectionDiagnostics": co_detection_diagnostics,
        "errors": errors,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "qualityGuard": {
            "maximumObjectMatchArcsec": MAX_OBJECT_MATCH_ARCSEC,
            "maximumMeasurementMatchArcsec": MAX_MEASUREMENT_MATCH_ARCSEC,
            "minimumMeasuredPairSeparationFraction": MIN_PAIR_SEPARATION_FRACTION,
            "maximumMagnitudeError": MAX_MAG_ERROR,
            "minimumBandSamples": MIN_BAND_SAMPLES,
            "minimumCrossBandSupport": MIN_CROSS_BAND_SUPPORT,
            "requiresSameExposureCoDetection": True,
        },
        "interpretationGuard": (
            "NSC DR2 is used as an archival source-resolved screen because it contains individual measurements from "
            "public NOIRLab imaging with substantially finer typical image quality than TESS. OpenStar requires two "
            "distinct NSC objects and admits only same-exposure/filter epochs where both objects are independently "
            "position-matched to the frozen Gaia coordinates and remain spatially separated. A source is supported "
            "only when accepted residual-band variability recurs at consistent frequency in at least two bands. "
            "The TESS drift law is not extrapolated into the NSC observing epochs."
        ),
    }


def interpret_nsc_resolved_project(
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
            if prepared is not None:
                results.append(_dataset_result(dataset, prepared))

    target = _summarize_source("target-control", results)
    counterpart = _summarize_source("catalog-counterpart", results)
    target_supported = bool(target.get("sourceSupported"))
    counterpart_supported = bool(counterpart.get("sourceSupported"))
    target_suggestive = bool(target.get("sourceSuggestive"))
    counterpart_suggestive = bool(counterpart.get("sourceSuggestive"))

    if not preparation.get("pairSeparatelyResolvedInNSC"):
        classification = "NSC_DR2_PAIR_NOT_SEPARATELY_RESOLVED"
        origin = "UNRESOLVED_NSC_SPATIAL_MATCH_LIMIT"
        next_test = NEXT_ARCHIVE_TEST
    elif not preparation.get("preparedSeries"):
        classification = "NSC_DR2_NO_QUALIFYING_CODETECTED_RESOLVED_SERIES"
        origin = "UNRESOLVED_NSC_NO_CLEAN_CODETECTED_TIME_SERIES"
        next_test = NEXT_ARCHIVE_TEST
    elif counterpart_supported and not target_supported:
        classification = "NSC_DR2_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_BY_NSC_RESOLVED_PHOTOMETRY"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_supported and not counterpart_supported:
        classification = "NSC_DR2_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_SUPPORTED_BY_NSC_RESOLVED_PHOTOMETRY"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif target_supported and counterpart_supported:
        classification = "NSC_DR2_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_AND_COUNTERPART_SUPPORTED_BY_NSC_RESOLVED_PHOTOMETRY"
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    elif counterpart_suggestive and not target_suggestive:
        classification = "NSC_DR2_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE_BY_NSC_RESOLVED_PHOTOMETRY"
        next_test = NEXT_ARCHIVE_TEST
    elif target_suggestive and not counterpart_suggestive:
        classification = "NSC_DR2_TARGET_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "TARGET_SUGGESTIVE_BY_NSC_RESOLVED_PHOTOMETRY"
        next_test = NEXT_ARCHIVE_TEST
    else:
        classification = "NSC_DR2_RESOLVED_VARIABILITY_UNRESOLVED"
        origin = "ARCHIVAL_NSC_SOURCE_ATTRIBUTION_UNRESOLVED"
        next_test = NEXT_ARCHIVE_TEST

    return {
        "version": "openstar.tess-nsc-dr2-resolved-photometry-screen.v1",
        "archive": preparation.get("archive"),
        "dataRelease": preparation.get("dataRelease"),
        "sourcePair": preparation.get("sourcePair"),
        "pairSeparationArcsec": preparation.get("pairSeparationArcsec"),
        "pairSeparatelyResolvedInNSC": preparation.get("pairSeparatelyResolvedInNSC"),
        "observedNSCObjectSeparationArcsec": preparation.get("observedNSCObjectSeparationArcsec"),
        "objectMatches": preparation.get("objectMatches"),
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "tessDriftExtrapolated": False,
        "componentResults": results,
        "targetControl": target,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "preparationErrors": preparation.get("errors") or [],
        "coDetectionDiagnostics": preparation.get("coDetectionDiagnostics"),
        "qualityGuard": preparation.get("qualityGuard"),
        "interpretationGuard": preparation.get("interpretationGuard"),
    }
