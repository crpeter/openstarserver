from __future__ import annotations

import csv
import io
import math
import urllib.parse
import urllib.error
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

SKYMAPPER_RELEASE = "DR4"
SKYMAPPER_TAP_SYNC = "https://api.skymapper.nci.org.au/public/tap/sync/"
HTTP_TIMEOUT_SECONDS = 90
USER_AGENT = "OpenStar/20.20 SkyMapper-resolved-epoch-photometry"

# v20.20 is deliberately conservative because the frozen Gaia pair is only
# ~2.44 arcsec apart and SkyMapper's typical seeing is comparable.  We only
# retain detections from exposures whose CCD seeing is materially tighter than
# the pair separation and whose Source Extractor / image-mask flags are clean.
MAX_MASTER_MATCH_ARCSEC = 1.0
MAX_DETECTION_MATCH_ARCSEC = 0.8
SEEING_FRACTION_OF_PAIR = 0.90
MAX_SEEING_ARCSEC = 2.20
MAX_PSF_MAG_ERROR = 0.15
MIN_BAND_SAMPLES = 10
MIN_PEAK_PROMINENCE_RATIO = 2.0
MIN_CROSS_BAND_SUPPORT = 2
MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD = 0.12
SUPPORTED_BANDS = ("g", "r", "i", "z")
BAND_NEIGHBOR_BIAS_BITS = {"g": 8, "r": 4, "i": 2, "z": 1}
NEXT_ARCHIVE_TEST = "NSC_RESOLVED_COUNTERPART_PHOTOMETRY"
CURRENT_TRIGGER = "SKYMAPPER_RESOLVED_COUNTERPART_PHOTOMETRY"


class SkyMapperArchiveUnavailable(RuntimeError):
    """The SkyMapper service failed transiently before a scientific result existed."""


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
            raise SkyMapperArchiveUnavailable(
                f"SkyMapper DR4 service unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        raise


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.integer, np.floating)):
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


def _tap_csv(query: str) -> list[dict[str, str]]:
    payload = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        SKYMAPPER_TAP_SYNC,
        data=payload,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        body = response.read()
        content_type = str(response.headers.get("Content-Type") or "")
    if not body:
        raise RuntimeError("SkyMapper TAP returned an empty response.")
    text = body.decode("utf-8-sig", errors="replace")
    prefix = text.lstrip()[:300].lower()
    if prefix.startswith("<") or "<votable" in prefix or "<html" in prefix:
        raise RuntimeError(
            f"SkyMapper TAP did not return CSV (Content-Type={content_type or '[none]'})."
        )
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise RuntimeError("SkyMapper TAP CSV has no header.")
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
        for key, default_role in (("target", "target-control"),
                                  ("counterpart", "catalog-counterpart")):
            source = pair.get(key) or {}
            source_id = _int(source.get("gaiaDR3SourceID"))
            ra = _float(source.get("raDeg"))
            dec = _float(source.get("decDeg"))
            if source_id is None or ra is None or dec is None:
                raise RuntimeError("Current Gaia sourcePair has incomplete archive coordinates.")
            definitions.append({"sourceRole": str(source.get("sourceRole") or default_role),
                                "gaiaDR3SourceID": source_id, "raDeg": ra, "decDeg": dec})
        separation = _float(pair.get("separationArcsec"))
        if separation is None or separation <= 0:
            separation = _angular_separation_arcsec(
                definitions[0]["raDeg"], definitions[0]["decDeg"],
                definitions[1]["raDeg"], definitions[1]["decDeg"])
        return definitions, float(separation)

    records = _source_records_by_role(external_high_resolution_summary)
    target_id = _int(pair.get("targetGaiaDR3SourceID"))
    counterpart_id = _int(pair.get("counterpartGaiaDR3SourceID"))
    if target_id is None or counterpart_id is None:
        raise RuntimeError("v20.20 requires both frozen Gaia DR3 source IDs from v20.19.")

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
                f"v20.20 requires frozen Gaia RA/Dec metadata for {role} from v20.19."
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


def _query_master_matches(source_ids: list[int]) -> list[dict[str, str]]:
    identifiers = ",".join(str(int(value)) for value in sorted(set(source_ids)))
    query = f"""
SELECT TOP 50
    object_id,
    raj2000,
    dej2000,
    flags,
    nimaflags,
    flags_psf,
    ngood,
    g_ngood,
    r_ngood,
    i_ngood,
    z_ngood,
    gaia_dr3_id1,
    gaia_dr3_dist1,
    gaia_dr3_id2,
    gaia_dr3_dist2,
    self_id1,
    self_dist1,
    self_id2,
    self_dist2,
    cnt_self_15,
    cnt_gaia_dr3_15
FROM dr4.master
WHERE gaia_dr3_id1 IN ({identifiers})
"""
    return _tap_csv(query)


def _select_master_match(
    rows: list[dict[str, str]],
    source: dict[str, Any],
) -> dict[str, Any] | None:
    source_id = int(source["gaiaDR3SourceID"])
    candidates: list[dict[str, Any]] = []
    for row in rows:
        gaia_id = _parse_int(row.get("gaia_dr3_id1"))
        if gaia_id != source_id:
            continue
        object_id = _parse_int(row.get("object_id"))
        ra = _parse_float(row.get("raj2000"))
        dec = _parse_float(row.get("dej2000"))
        catalog_dist = _parse_float(row.get("gaia_dr3_dist1"))
        if object_id is None or ra is None or dec is None:
            continue
        positional_dist = _angular_separation_arcsec(
            ra,
            dec,
            float(source["raDeg"]),
            float(source["decDeg"]),
        )
        distance = catalog_dist if catalog_dist is not None else positional_dist
        candidates.append(
            {
                "objectID": int(object_id),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "gaiaDR3SourceID": source_id,
                "gaiaDR3DistanceArcsec": float(distance),
                "positionDistanceArcsec": float(positional_dist),
                "flags": _parse_int(row.get("flags")),
                "nimaflags": _parse_int(row.get("nimaflags")),
                "flagsPSF": _parse_int(row.get("flags_psf")),
                "ngood": _parse_int(row.get("ngood")),
                "bandGoodCounts": {
                    band: _parse_int(row.get(f"{band}_ngood")) for band in SUPPORTED_BANDS
                },
                "nearestSelfObjectID": _parse_int(row.get("self_id1")),
                "nearestSelfDistanceArcsec": _parse_float(row.get("self_dist1")),
                "secondSelfObjectID": _parse_int(row.get("self_id2")),
                "secondSelfDistanceArcsec": _parse_float(row.get("self_dist2")),
                "gaiaDR3SecondSourceID": _parse_int(row.get("gaia_dr3_id2")),
                "gaiaDR3SecondDistanceArcsec": _parse_float(row.get("gaia_dr3_dist2")),
                "sourceCountWithin15Arcsec": _parse_int(row.get("cnt_self_15")),
                "gaiaCountWithin15Arcsec": _parse_int(row.get("cnt_gaia_dr3_15")),
            }
        )
    if not candidates:
        return None
    best = min(candidates, key=lambda item: item["gaiaDR3DistanceArcsec"])
    if best["gaiaDR3DistanceArcsec"] > MAX_MASTER_MATCH_ARCSEC:
        return None
    return best


def _query_epoch_detections(object_ids: list[int]) -> list[dict[str, str]]:
    identifiers = ",".join(str(int(value)) for value in sorted(set(object_ids)))
    bands = ",".join(f"'{band}'" for band in SUPPORTED_BANDS)
    query = f"""
SELECT TOP 4000
    p.object_id,
    p.image_id,
    p.ccd,
    p.filter,
    p.ra_img,
    p.decl_img,
    p.flags,
    p.nimaflags,
    p.chi2_psf,
    p.mag_psf,
    p.e_mag_psf,
    c.mjd_obs,
    c.fwhm_ccd,
    c.elong_ccd
FROM dr4.photometry AS p
JOIN dr4.ccds AS c
  ON p.image_id = c.image_id AND p.ccd = c.ccd
WHERE p.object_id IN ({identifiers})
  AND p.filter IN ({bands})
ORDER BY c.mjd_obs
"""
    return _tap_csv(query)


def _robust_standardize_magnitudes(magnitudes: np.ndarray) -> np.ndarray:
    values = np.asarray(magnitudes, dtype=np.float64)
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("SkyMapper PSF magnitudes have no finite variability scale.")
    # Magnitudes run backwards, so negate to produce a flux-like normalized series.
    return -(values - median) / scale


def _prepare_band_series(
    rows: list[dict[str, str]],
    *,
    source: dict[str, Any],
    master: dict[str, Any],
    pair_separation_arcsec: float,
) -> dict[str, dict[str, Any]]:
    object_id = int(master["objectID"])
    seeing_limit = min(
        MAX_SEEING_ARCSEC,
        float(pair_separation_arcsec) * SEEING_FRACTION_OF_PAIR,
    )
    by_band: dict[str, list[dict[str, float]]] = {band: [] for band in SUPPORTED_BANDS}
    diagnostics = {
        band: {
            "totalRows": 0,
            "cleanFlagRows": 0,
            "goodSeeingRows": 0,
            "positionMatchedRows": 0,
            "usableRows": 0,
        }
        for band in SUPPORTED_BANDS
    }

    master_flags_psf = _int(master.get("flagsPSF")) or 0

    for row in rows:
        if _parse_int(row.get("object_id")) != object_id:
            continue
        band = str(row.get("filter") or "").strip().lower()
        if band not in by_band:
            continue
        diag = diagnostics[band]
        diag["totalRows"] += 1

        # DR4 flags_psf marks bands where a neighbour is expected to bias PSF
        # photometry by >1%.  Reject that entire band before looking at epochs.
        neighbor_bit = BAND_NEIGHBOR_BIAS_BITS[band]
        if master_flags_psf & neighbor_bit:
            diag["masterNeighborBiasFlag"] = True
            continue
        diag["masterNeighborBiasFlag"] = False

        flags = _parse_int(row.get("flags"))
        nimaflags = _parse_int(row.get("nimaflags"))
        if flags != 0 or nimaflags != 0:
            continue
        diag["cleanFlagRows"] += 1

        seeing = _parse_float(row.get("fwhm_ccd"))
        if seeing is None or seeing > seeing_limit:
            continue
        diag["goodSeeingRows"] += 1

        ra = _parse_float(row.get("ra_img"))
        dec = _parse_float(row.get("decl_img"))
        if ra is None or dec is None:
            continue
        separation = _angular_separation_arcsec(
            ra,
            dec,
            float(source["raDeg"]),
            float(source["decDeg"]),
        )
        if separation > MAX_DETECTION_MATCH_ARCSEC:
            continue
        diag["positionMatchedRows"] += 1

        mjd = _parse_float(row.get("mjd_obs"))
        magnitude = _parse_float(row.get("mag_psf"))
        magnitude_error = _parse_float(row.get("e_mag_psf"))
        if (
            mjd is None
            or magnitude is None
            or magnitude_error is None
            or magnitude_error <= 0
            or magnitude_error > MAX_PSF_MAG_ERROR
        ):
            continue
        by_band[band].append(
            {
                "mjd": float(mjd),
                "magnitude": float(magnitude),
                "magnitudeError": float(magnitude_error),
                "seeingArcsec": float(seeing),
                "positionDistanceArcsec": float(separation),
            }
        )
        diag["usableRows"] += 1

    output: dict[str, dict[str, Any]] = {}
    for band, items in by_band.items():
        if len(items) < MIN_BAND_SAMPLES:
            continue
        items.sort(key=lambda item: item["mjd"])
        times = np.asarray([item["mjd"] for item in items], dtype=np.float64)
        magnitudes = np.asarray([item["magnitude"] for item in items], dtype=np.float64)

        # Remove only extreme photometric outliers after the archive quality cuts.
        median = float(np.median(magnitudes))
        mad = float(np.median(np.abs(magnitudes - median)))
        if math.isfinite(mad) and mad > 0:
            robust_sigma = 1.4826 * mad
            keep = np.abs(magnitudes - median) <= 5.0 * robust_sigma
            times = times[keep]
            magnitudes = magnitudes[keep]
            items = [item for item, accepted in zip(items, keep.tolist()) if accepted]
        if len(times) < MIN_BAND_SAMPLES:
            continue

        flux = _robust_standardize_magnitudes(magnitudes)
        local_times = times - float(times[0])
        output[band] = {
            "times": local_times,
            "flux": flux,
            "sampleCount": int(len(local_times)),
            "baselineDays": float(local_times[-1] - local_times[0]),
            "medianSeeingArcsec": float(np.median([item["seeingArcsec"] for item in items])),
            "maximumSeeingArcsec": float(max(item["seeingArcsec"] for item in items)),
            "medianPositionDistanceArcsec": float(
                np.median([item["positionDistanceArcsec"] for item in items])
            ),
            "medianMagnitudeError": float(
                np.median([item["magnitudeError"] for item in items])
            ),
            "qualityDiagnostics": diagnostics[band],
        }
    return output


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
        "skyMapperObjectID": prepared.get("skyMapperObjectID"),
        "band": prepared.get("band"),
        "sampleCount": prepared.get("sampleCount"),
        "baselineDays": prepared.get("baselineDays"),
        "medianSeeingArcsec": prepared.get("medianSeeingArcsec"),
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
    frequencies = [float(item["candidateFrequency"]) for item in accepted if item.get("candidateFrequency")]
    consistent = False
    relative_spread = None
    median_frequency = None
    if frequencies:
        median_frequency = float(np.median(np.asarray(frequencies, dtype=np.float64)))
        if median_frequency > 0:
            relative_spread = float((max(frequencies) - min(frequencies)) / median_frequency)
        if len(frequencies) >= MIN_CROSS_BAND_SUPPORT and relative_spread is not None:
            consistent = relative_spread <= MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD

    strongest = None
    if source_results:
        strongest = max(source_results, key=lambda item: float(item.get("candidatePower") or 0.0))

    return {
        "sourceRole": role,
        "bandResults": source_results,
        "acceptedBands": [item.get("band") for item in accepted],
        "acceptedBandCount": int(len(accepted)),
        "crossBandFrequencyConsistent": bool(consistent),
        "crossBandRelativeFrequencySpread": relative_spread,
        "medianAcceptedFrequency": median_frequency,
        "sourceSupported": bool(consistent and len(accepted) >= MIN_CROSS_BAND_SUPPORT),
        "sourceSuggestive": bool(len(accepted) >= 1),
        "strongestBand": strongest,
    }


def build_skymapper_resolved_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    external_high_resolution_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if external_high_resolution_summary.get("recommendedNextTest") not in {
        CURRENT_TRIGGER, "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    }:
        raise RuntimeError(
            "v20.20 requires v20.19 to recommend TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY."
        )

    source_definitions, pair_separation = _frozen_source_pair(external_high_resolution_summary)
    source_ids = [int(item["gaiaDR3SourceID"]) for item in source_definitions]
    print("   querying SkyMapper DR4 master objects for the frozen Gaia pair", flush=True)
    master_rows = _archive_query(_query_master_matches, source_ids)

    master_matches: dict[str, dict[str, Any] | None] = {}
    for source in source_definitions:
        role = str(source["sourceRole"])
        match = _select_master_match(master_rows, source)
        master_matches[role] = match
        if match is None:
            print(f"      {role}: no distinct SkyMapper DR4 master match", flush=True)
        else:
            print(
                f"      {role}: SkyMapper object {match['objectID']} | Gaia distance={match['gaiaDR3DistanceArcsec']:.3f} arcsec",
                flush=True,
            )

    target_master = master_matches.get("target-control")
    counterpart_master = master_matches.get("catalog-counterpart")
    pair_separately_resolved = bool(
        target_master is not None
        and counterpart_master is not None
        and int(target_master["objectID"]) != int(counterpart_master["objectID"])
    )

    search = dict(external_high_resolution_summary.get("frequencySearch") or
                  ((external_high_resolution_summary.get("distributedValidation") or {}).get("frequencySearch") or {}))
    total_frequencies = _int(search.get("totalFrequencies"))
    per_work = _int(search.get("frequenciesPerWorkUnit"))
    if not search or total_frequencies is None or per_work is None or total_frequencies <= 0 or per_work <= 0:
        raise RuntimeError("v20.20 requires the frozen residual-frequency search definition from v20.19.")

    root = Path(output_dir) / "skymapper-resolved-photometry"
    root.mkdir(parents=True, exist_ok=True)
    seeing_limit = min(MAX_SEEING_ARCSEC, pair_separation * SEEING_FRACTION_OF_PAIR)

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    preparation_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if pair_separately_resolved:
        object_ids = [int(target_master["objectID"]), int(counterpart_master["objectID"])]
        print(
            f"   pair has distinct SkyMapper objects; loading per-image PSF photometry with seeing <= {seeing_limit:.3f} arcsec",
            flush=True,
        )
        rows = _archive_query(_query_epoch_detections, object_ids)
        for source in source_definitions:
            role = str(source["sourceRole"])
            master = master_matches[role]
            assert master is not None
            try:
                band_series = _prepare_band_series(
                    rows,
                    source=source,
                    master=master,
                    pair_separation_arcsec=pair_separation,
                )
            except RuntimeError as exc:
                if "no finite variability scale" not in str(exc):
                    raise
                errors.append({"sourceRole": role, "error": f"{type(exc).__name__}: {exc}"})
                band_series = {}

            record = {
                "sourceRole": role,
                "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                "skyMapperMaster": master,
                "preparedBands": sorted(band_series),
            }
            preparation_records.append(record)

            for band, series in band_series.items():
                dataset_id = f"{source_dataset_id}-skymapper-dr4-{role}-{band}-v1"
                target_name = f"{source_dataset_id} SkyMapper DR4 {role} {band}-band"
                dataset_path = root / f"{_safe(dataset_id)}.json"
                dataset = {
                    "id": dataset_id,
                    "targetName": target_name,
                    "times": np.asarray(series["times"], dtype=np.float32).tolist(),
                    "flux": np.asarray(series["flux"], dtype=np.float32).tolist(),
                    "frequencySearch": search,
                    "reference": {},
                    "science": {
                        "role": "skymapper-resolved-epoch-photometry-screen",
                        "sourceRole": role,
                        "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                        "skyMapperObjectID": int(master["objectID"]),
                        "band": band,
                        "pairSeparationArcsec": float(pair_separation),
                        "seeingLimitArcsec": float(seeing_limit),
                        "tessDriftExtrapolated": False,
                    },
                    "source": {
                        "mission": "SkyMapper Southern Survey",
                        "dataRelease": SKYMAPPER_RELEASE,
                        "band": band,
                        "distributedSamples": int(series["sampleCount"]),
                        "baselineDays": float(series["baselineDays"]),
                        "medianSeeingArcsec": float(series["medianSeeingArcsec"]),
                    },
                }
                _write_json(dataset_path, dataset)
                prepared = {
                    "datasetID": dataset_id,
                    "datasetPath": str(dataset_path.resolve()),
                    "sourceRole": role,
                    "gaiaDR3SourceID": int(source["gaiaDR3SourceID"]),
                    "skyMapperObjectID": int(master["objectID"]),
                    "band": band,
                    "sampleCount": int(series["sampleCount"]),
                    "baselineDays": float(series["baselineDays"]),
                    "medianSeeingArcsec": float(series["medianSeeingArcsec"]),
                    "maximumSeeingArcsec": float(series["maximumSeeingArcsec"]),
                    "medianPositionDistanceArcsec": float(series["medianPositionDistanceArcsec"]),
                    "medianMagnitudeError": float(series["medianMagnitudeError"]),
                    "qualityDiagnostics": series["qualityDiagnostics"],
                }
                prepared_series.append(prepared)
                dataset_entries.append(
                    {"id": dataset_id, "path": str(dataset_path.resolve()), "targetName": target_name}
                )
                print(
                    f"      {role} {band}: {series['sampleCount']} clean epochs | median seeing={series['medianSeeingArcsec']:.3f} arcsec",
                    flush=True,
                )
    else:
        print(
            "   SkyMapper does not provide two distinct master objects for the frozen pair; no variability claim will be attempted",
            flush=True,
        )

    project_id: str | None = None
    project_path: str | None = None
    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "skymapper-dr4-resolved-photometry-v1"
        )
        manifest = {
            "id": project_id,
            "name": f"{source_project_id} — SkyMapper DR4 resolved residual-band screen",
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": "skymapper-resolved-epoch-photometry-screen",
                "archive": "SkyMapper Southern Survey DR4",
                "workerSemantics": (
                    "Each dataset is a single-band, clean-flag, good-seeing SkyMapper DR4 PSF time series "
                    "for one frozen Gaia source. Workers execute ordinary Lomb-Scargle over the frozen residual-frequency band."
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
        "version": "openstar.tess-skymapper-resolved-photometry-preparation.v1",
        "archive": "SkyMapper Southern Survey DR4",
        "dataRelease": SKYMAPPER_RELEASE,
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-clean-good-seeing-skymapper-dr4-single-band-psf-series",
        "sourcePair": external_high_resolution_summary.get("sourcePair"),
        "pairSeparationArcsec": float(pair_separation),
        "seeingLimitArcsec": float(seeing_limit),
        "pairSeparatelyResolvedInSkyMapperMaster": bool(pair_separately_resolved),
        "masterMatches": master_matches,
        "frequencySearch": search,
        "tessDriftExtrapolated": False,
        "preparedSeries": prepared_series,
        "preparationRecords": preparation_records,
        "errors": errors,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "qualityGuard": {
            "sourceExtractorFlagsRequired": 0,
            "imageMaskFlagsRequired": 0,
            "maximumDetectionPositionDistanceArcsec": MAX_DETECTION_MATCH_ARCSEC,
            "maximumSeeingArcsec": float(seeing_limit),
            "maximumPsfMagnitudeError": MAX_PSF_MAG_ERROR,
            "minimumBandSamples": MIN_BAND_SAMPLES,
            "minimumCrossBandSupport": MIN_CROSS_BAND_SUPPORT,
        },
        "interpretationGuard": (
            "SkyMapper DR4 is an opportunistic archival screen, not equivalent to dedicated high-resolution follow-up. "
            "The frozen pair is only a few arcseconds apart and typical SkyMapper seeing is comparable, so OpenStar "
            "uses only distinct master objects, clean Source Extractor/image-mask flags, per-detection positional agreement, "
            "and exposures with seeing materially tighter than the pair separation. A source is called supported only when "
            "the residual-band signal recurs at consistent frequency in at least two independent SkyMapper bands. "
            "The TESS drift law is not extrapolated into the SkyMapper epoch."
        ),
    }


def interpret_skymapper_resolved_project(
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

    if not preparation.get("pairSeparatelyResolvedInSkyMapperMaster"):
        classification = "SKYMAPPER_DR4_PAIR_NOT_SEPARATELY_RESOLVED"
        origin = "UNRESOLVED_SKYMAPPER_SPATIAL_RESOLUTION_LIMIT"
        next_test = NEXT_ARCHIVE_TEST
    elif not preparation.get("preparedSeries"):
        classification = "SKYMAPPER_DR4_NO_QUALIFYING_RESOLVED_EPOCH_SERIES"
        origin = "UNRESOLVED_SKYMAPPER_NO_CLEAN_GOOD_SEEING_TIME_SERIES"
        next_test = NEXT_ARCHIVE_TEST
    elif counterpart_supported and not target_supported:
        classification = "SKYMAPPER_DR4_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_BY_SKYMAPPER_RESOLVED_PHOTOMETRY"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_supported and not counterpart_supported:
        classification = "SKYMAPPER_DR4_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_SUPPORTED_BY_SKYMAPPER_RESOLVED_PHOTOMETRY"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif target_supported and counterpart_supported:
        classification = "SKYMAPPER_DR4_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_AND_COUNTERPART_SUPPORTED_BY_SKYMAPPER_RESOLVED_PHOTOMETRY"
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    elif counterpart_suggestive and not target_suggestive:
        classification = "SKYMAPPER_DR4_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE_BY_SKYMAPPER_RESOLVED_PHOTOMETRY"
        next_test = NEXT_ARCHIVE_TEST
    elif target_suggestive and not counterpart_suggestive:
        classification = "SKYMAPPER_DR4_TARGET_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "TARGET_SUGGESTIVE_BY_SKYMAPPER_RESOLVED_PHOTOMETRY"
        next_test = NEXT_ARCHIVE_TEST
    else:
        classification = "SKYMAPPER_DR4_RESOLVED_VARIABILITY_UNRESOLVED"
        origin = "ARCHIVAL_SKYMAPPER_SOURCE_ATTRIBUTION_UNRESOLVED"
        next_test = NEXT_ARCHIVE_TEST

    return {
        "version": "openstar.tess-skymapper-resolved-photometry-screen.v1",
        "archive": preparation.get("archive"),
        "dataRelease": preparation.get("dataRelease"),
        "sourcePair": preparation.get("sourcePair"),
        "pairSeparationArcsec": preparation.get("pairSeparationArcsec"),
        "seeingLimitArcsec": preparation.get("seeingLimitArcsec"),
        "pairSeparatelyResolvedInSkyMapperMaster": preparation.get(
            "pairSeparatelyResolvedInSkyMapperMaster"
        ),
        "masterMatches": preparation.get("masterMatches"),
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
        "qualityGuard": preparation.get("qualityGuard"),
        "interpretationGuard": preparation.get("interpretationGuard"),
    }
