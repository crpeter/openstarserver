from __future__ import annotations

import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
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
from .tess_noirlab_forced_photometry import (
    _angular_separation_arcsec,
    _python_value,
    _source_records_by_role,
    _summarize_source,
)

ATLAS_BASE_URL = "https://fallingstar-data.com/forcedphot"
ATLAS_AUTH_URL = f"{ATLAS_BASE_URL}/api-token-auth/"
ATLAS_QUEUE_URL = f"{ATLAS_BASE_URL}/queue/"
HTTP_TIMEOUT_SECONDS = 120
JOB_POLL_SECONDS = 10
JOB_TIMEOUT_SECONDS = 2400
USER_AGENT = "OpenStar/20.24 ATLAS-forced-photometry"

SUPPORTED_BANDS = ("c", "o", "g", "r", "i")
ATLAS_MJD_MIN = 59500.0
MIN_GAIA_PAIR_SEPARATION_ARCSEC = 15.0

MIN_RAW_SNR = 3.0
MAX_REDUCED_CHI_SQUARED = 5.0
MIN_NIGHTLY_POINTS = 1
MIN_BAND_NIGHTS = 20
MIN_BAND_BASELINE_DAYS = 30.0
MAX_ROBUST_OUTLIER_SIGMA = 5.0

MIN_PEAK_PROMINENCE_RATIO = 2.0
MIN_CROSS_BAND_SUPPORT = 2
MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD = 0.12


def atlas_credentials_available() -> bool:
    token = (os.environ.get("OPENSTAR_ATLAS_API_TOKEN") or "").strip()
    username = (os.environ.get("OPENSTAR_ATLAS_USERNAME") or "").strip()
    password = (os.environ.get("OPENSTAR_ATLAS_PASSWORD") or "").strip()
    return bool(token or (username and password))


def require_atlas_credentials() -> None:
    if atlas_credentials_available():
        return
    raise RuntimeError(
        "v20.24 requires ATLAS forced-photometry credentials. Set OPENSTAR_ATLAS_API_TOKEN, "
        "or set both OPENSTAR_ATLAS_USERNAME and OPENSTAR_ATLAS_PASSWORD. "
        "Register at https://fallingstar-data.com/forcedphot/register/."
    )


def _json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    form: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    if form is not None:
        body = urllib.parse.urlencode(form).encode("utf-8")
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"

    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        status = int(exc.code)

    text = payload.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ATLAS API returned non-JSON response for {url}: {text[:500]}"
        ) from exc
    return status, parsed


def _text_request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> str:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        headers=request_headers,
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("ATLAS result download returned an empty response.")
    return payload.decode("utf-8", errors="replace")


def _atlas_headers() -> dict[str, str]:
    require_atlas_credentials()

    token = (os.environ.get("OPENSTAR_ATLAS_API_TOKEN") or "").strip()
    if not token:
        username = (os.environ.get("OPENSTAR_ATLAS_USERNAME") or "").strip()
        password = (os.environ.get("OPENSTAR_ATLAS_PASSWORD") or "").strip()
        status, payload = _json_request(
            ATLAS_AUTH_URL,
            method="POST",
            form={"username": username, "password": password},
        )
        if status != 200:
            raise RuntimeError(
                f"ATLAS authentication failed with HTTP {status}: {payload}"
            )
        token = str(payload.get("token") or "").strip()
        if not token:
            raise RuntimeError("ATLAS authentication response did not include a token.")

    return {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    }


def _throttle_wait_seconds(message: str) -> int:
    seconds = re.search(r"available in (\d+) seconds", message, re.IGNORECASE)
    if seconds:
        return max(1, int(seconds.group(1)))
    minutes = re.search(r"available in (\d+) minutes", message, re.IGNORECASE)
    if minutes:
        return max(1, int(minutes.group(1)) * 60)
    return JOB_POLL_SECONDS


def _submit_atlas_job(
    headers: dict[str, str],
    *,
    ra_deg: float,
    dec_deg: float,
) -> str:
    while True:
        status, payload = _json_request(
            ATLAS_QUEUE_URL,
            method="POST",
            headers=headers,
            form={
                "ra": f"{float(ra_deg):.10f}",
                "dec": f"{float(dec_deg):.10f}",
                "mjd_min": f"{ATLAS_MJD_MIN:.1f}",
                "send_email": "False",
                "use_reduced": "True",
            },
        )
        if status == 201:
            task_url = str(payload.get("url") or "").strip()
            if not task_url:
                raise RuntimeError("ATLAS queued a task but returned no task URL.")
            return task_url
        if status == 429:
            message = str(payload.get("detail") or "")
            wait_seconds = _throttle_wait_seconds(message)
            print(
                f"      ATLAS queue throttled; retrying after {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            continue
        raise RuntimeError(
            f"ATLAS queue submission failed with HTTP {status}: {payload}"
        )


def _wait_for_atlas_job(
    headers: dict[str, str],
    task_url: str,
) -> tuple[str, str]:
    started = time.monotonic()
    last_state = None

    while True:
        if time.monotonic() - started > JOB_TIMEOUT_SECONDS:
            raise RuntimeError(
                "ATLAS forced-photometry task exceeded the OpenStar polling timeout. "
                "Rerun the v20.24 continuation; the prior FAILED stage will remain provenance."
            )

        status, payload = _json_request(
            task_url,
            method="GET",
            headers=headers,
        )
        if status != 200:
            raise RuntimeError(
                f"ATLAS task polling failed with HTTP {status}: {payload}"
            )

        finish = payload.get("finishtimestamp")
        result_url = str(payload.get("result_url") or "").strip()
        if finish:
            if not result_url:
                raise RuntimeError(
                    "ATLAS task reports completion but did not provide result_url."
                )
            return result_url, _text_request(result_url, headers=headers)

        start = payload.get("starttimestamp")
        state = "RUNNING" if start else "QUEUED"
        if state != last_state:
            if state == "RUNNING":
                print(f"      ATLAS task running: {task_url}", flush=True)
            else:
                print(f"      ATLAS task queued: {task_url}", flush=True)
            last_state = state
        time.sleep(JOB_POLL_SECONDS)


def _parse_float(value: Any) -> float | None:
    value = _python_value(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _parse_atlas_output(text: str) -> list[dict[str, Any]]:
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_candidate = line.lstrip("#").strip()
        if header is None and "MJD" in header_candidate and "uJy" in header_candidate:
            header = header_candidate.split()
            continue

        if line.startswith("#"):
            continue
        if header is None:
            continue

        values = line.split()
        if len(values) < len(header):
            continue
        if len(values) > len(header):
            values = values[: len(header)]

        row = dict(zip(header, values))
        rows.append(row)

    if header is None:
        raise RuntimeError("ATLAS result text did not contain a recognizable photometry header.")
    return rows


def _atlas_row_value(row: dict[str, Any], *names: str) -> Any:
    normalized = {
        str(key).lstrip("#").lower(): value
        for key, value in row.items()
    }
    for name in names:
        key = str(name).lstrip("#").lower()
        if key in normalized:
            return normalized[key]
    return None


def _clean_atlas_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        mjd = _parse_float(_atlas_row_value(row, "MJD"))
        flux = _parse_float(_atlas_row_value(row, "uJy"))
        flux_error = _parse_float(_atlas_row_value(row, "duJy"))
        band = str(_atlas_row_value(row, "F") or "").strip().lower()
        error_code = _parse_int(_atlas_row_value(row, "err"))
        chi = _parse_float(_atlas_row_value(row, "chi/N", "chi"))

        if mjd is None or flux is None or flux_error is None or flux_error <= 0:
            rejection_counts["invalid-numeric-row"] += 1
            continue
        if band not in SUPPORTED_BANDS:
            rejection_counts["unsupported-band"] += 1
            continue
        if error_code not in (None, 0):
            rejection_counts["tphot-error"] += 1
            continue
        if chi is not None and chi > MAX_REDUCED_CHI_SQUARED:
            rejection_counts["high-reduced-chi-square"] += 1
            continue

        snr = flux / flux_error
        if not math.isfinite(snr) or snr < MIN_RAW_SNR:
            rejection_counts["low-snr"] += 1
            continue

        accepted.append(
            {
                "mjd": float(mjd),
                "uJy": float(flux),
                "duJy": float(flux_error),
                "snr": float(snr),
                "band": band,
                "reducedChiSquared": chi,
                "magnitude": _parse_float(_atlas_row_value(row, "m")),
                "magnitudeError": _parse_float(_atlas_row_value(row, "dm")),
                "mag5sig": _parse_float(_atlas_row_value(row, "mag5sig")),
                "obs": str(_atlas_row_value(row, "Obs") or "").strip() or None,
            }
        )

    return accepted, {
        "rawRows": int(len(rows)),
        "acceptedRawRows": int(len(accepted)),
        "rejectionCounts": dict(sorted(rejection_counts.items())),
    }


def _nightly_band_series(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_band_night: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        night = int(math.floor(float(row["mjd"])))
        by_band_night[(str(row["band"]), night)].append(row)

    result: dict[str, dict[str, Any]] = {}

    for band in SUPPORTED_BANDS:
        nightly: list[dict[str, Any]] = []
        for (group_band, _), group in by_band_night.items():
            if group_band != band or len(group) < MIN_NIGHTLY_POINTS:
                continue
            fluxes = np.asarray([float(item["uJy"]) for item in group], dtype=np.float64)
            errors = np.asarray([float(item["duJy"]) for item in group], dtype=np.float64)
            times = np.asarray([float(item["mjd"]) for item in group], dtype=np.float64)
            weights = 1.0 / np.square(errors)
            total_weight = float(np.sum(weights))
            if not math.isfinite(total_weight) or total_weight <= 0:
                continue
            nightly.append(
                {
                    "mjd": float(np.sum(weights * times) / total_weight),
                    "uJy": float(np.sum(weights * fluxes) / total_weight),
                    "duJy": float(math.sqrt(1.0 / total_weight)),
                    "rawPoints": int(len(group)),
                }
            )

        nightly.sort(key=lambda item: float(item["mjd"]))
        if len(nightly) < MIN_BAND_NIGHTS:
            continue

        times = np.asarray([float(item["mjd"]) for item in nightly], dtype=np.float64)
        fluxes = np.asarray([float(item["uJy"]) for item in nightly], dtype=np.float64)
        errors = np.asarray([float(item["duJy"]) for item in nightly], dtype=np.float64)

        median_flux = float(np.median(fluxes))
        mad = float(np.median(np.abs(fluxes - median_flux)))
        keep = np.ones(len(fluxes), dtype=bool)
        if math.isfinite(mad) and mad > 0:
            robust_sigma = 1.4826 * mad
            keep = np.abs(fluxes - median_flux) <= MAX_ROBUST_OUTLIER_SIGMA * robust_sigma

        times = times[keep]
        fluxes = fluxes[keep]
        errors = errors[keep]
        kept_nightly = [
            item
            for item, accepted in zip(nightly, keep.tolist())
            if accepted
        ]
        if len(times) < MIN_BAND_NIGHTS:
            continue

        baseline = float(times[-1] - times[0])
        if baseline < MIN_BAND_BASELINE_DAYS:
            continue

        center = float(np.median(fluxes))
        scale = float(np.median(np.abs(fluxes - center))) * 1.4826
        if not math.isfinite(scale) or scale <= 0:
            scale = float(np.std(fluxes))
        if not math.isfinite(scale) or scale <= 0:
            continue

        standardized = (fluxes - center) / scale
        local_times = times - float(times[0])

        result[band] = {
            "times": local_times,
            "flux": standardized,
            "sampleCount": int(len(local_times)),
            "baselineDays": baseline,
            "medianFluxUJy": float(center),
            "medianFluxErrorUJy": float(np.median(errors)),
            "medianNightlySNR": float(np.median(fluxes / errors)),
            "medianRawPointsPerNight": float(
                np.median([item["rawPoints"] for item in kept_nightly])
            ),
        }

    return result


def _dataset_result(
    project_dataset: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
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
        "band": prepared.get("band"),
        "sampleCount": prepared.get("sampleCount"),
        "baselineDays": prepared.get("baselineDays"),
        "medianFluxUJy": prepared.get("medianFluxUJy"),
        "medianNightlySNR": prepared.get("medianNightlySNR"),
        "periodStatus": status or None,
        "periodConfidence": project_dataset.get("periodConfidence"),
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "acceptedResidualBandVariability": accepted,
    }


def _summarize_atlas_source(
    role: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    source_results = [
        item
        for item in results
        if item.get("sourceRole") == role
    ]
    accepted = [
        item
        for item in source_results
        if item.get("acceptedResidualBandVariability")
    ]
    frequencies = [
        float(item["candidateFrequency"])
        for item in accepted
        if item.get("candidateFrequency") is not None
    ]

    median_frequency = None
    relative_spread = None
    supported = False

    if frequencies:
        median_frequency = float(
            np.median(np.asarray(frequencies, dtype=np.float64))
        )
        if median_frequency > 0:
            relative_spread = float(
                (max(frequencies) - min(frequencies)) / median_frequency
            )
        supported = bool(
            len(frequencies) >= MIN_CROSS_BAND_SUPPORT
            and relative_spread is not None
            and relative_spread <= MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD
        )

    return {
        "sourceRole": role,
        "bandResults": source_results,
        "acceptedBands": sorted(str(item.get("band")) for item in accepted),
        "acceptedBandCount": int(len(accepted)),
        "medianAcceptedFrequency": median_frequency,
        "crossBandRelativeFrequencySpread": relative_spread,
        "sourceSupported": supported,
        "sourceSuggestive": bool(accepted) and not supported,
    }


def _frozen_sources(
    external_high_resolution_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    records = _source_records_by_role(external_high_resolution_summary)
    pair = external_high_resolution_summary.get("sourcePair") or {}
    target_id = _int(pair.get("targetGaiaDR3SourceID"))
    counterpart_id = _int(pair.get("counterpartGaiaDR3SourceID"))
    if target_id is None or counterpart_id is None:
        raise RuntimeError("v20.24 requires both frozen Gaia DR3 source IDs from v20.19.")

    sources: list[dict[str, Any]] = []
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
                f"v20.24 requires frozen Gaia coordinates for {role} from v20.19."
            )
        sources.append(
            {
                "sourceRole": role,
                "gaiaDR3SourceID": int(source_id),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "gMag": _float(metadata.get("gMag")),
                "bpMag": _float(metadata.get("bpMag")),
                "rpMag": _float(metadata.get("rpMag")),
            }
        )

    separation = _angular_separation_arcsec(
        sources[0]["raDeg"],
        sources[0]["decDeg"],
        sources[1]["raDeg"],
        sources[1]["decDeg"],
    )
    if separation < MIN_GAIA_PAIR_SEPARATION_ARCSEC:
        raise RuntimeError(
            "v20.24 is preregistered only for the corrected widely separated Gaia source pair."
        )

    return sources, float(separation)


def build_atlas_forced_photometry_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    external_high_resolution_summary: dict[str, Any],
    des_dr2_se_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if des_dr2_se_summary.get("recommendedNextTest") != "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY":
        raise RuntimeError(
            "v20.24 requires v20.23 to remain on the targeted high-resolution time-series branch."
        )

    require_atlas_credentials()
    headers = _atlas_headers()
    sources, pair_separation = _frozen_sources(external_high_resolution_summary)

    search = dict(
        ((des_dr2_se_summary.get("distributedValidation") or {}).get("frequencySearch") or {})
    )
    if not search:
        search = dict(
            ((external_high_resolution_summary.get("distributedValidation") or {}).get("frequencySearch") or {})
        )
    total_frequencies = _int(search.get("totalFrequencies"))
    per_work = _int(search.get("frequenciesPerWorkUnit"))
    if (
        not search
        or total_frequencies is None
        or per_work is None
        or total_frequencies <= 0
        or per_work <= 0
    ):
        raise RuntimeError("v20.24 requires the frozen residual-frequency search definition.")

    root = Path(output_dir) / "atlas-forced-photometry"
    root.mkdir(parents=True, exist_ok=True)

    source_records: list[dict[str, Any]] = []
    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []

    print(f"   corrected Gaia source separation: {pair_separation:.3f} arcsec", flush=True)
    print("   ATLAS mode: calibrated target-image forced photometry (use_reduced=True)", flush=True)
    print(f"   ATLAS minimum MJD: {ATLAS_MJD_MIN:.1f}", flush=True)

    for source in sources:
        role = str(source["sourceRole"])
        source_id = int(source["gaiaDR3SourceID"])

        print(
            f"   requesting ATLAS forced photometry for {role} | Gaia DR3 {source_id}",
            flush=True,
        )
        task_url = _submit_atlas_job(
            headers,
            ra_deg=float(source["raDeg"]),
            dec_deg=float(source["decDeg"]),
        )
        result_url, text = _wait_for_atlas_job(headers, task_url)

        raw_path = root / f"atlas-{role}-gaia-{source_id}-target-image.txt"
        raw_path.write_text(text, encoding="utf-8")

        parsed_rows = _parse_atlas_output(text)
        clean_rows, quality = _clean_atlas_rows(parsed_rows)
        band_series = _nightly_band_series(clean_rows)

        record = {
            "sourceRole": role,
            "gaiaDR3SourceID": source_id,
            "gaiaGMag": source.get("gMag"),
            "raDeg": source["raDeg"],
            "decDeg": source["decDeg"],
            "taskURL": task_url,
            "resultURL": result_url,
            "rawPath": str(raw_path.resolve()),
            "rawRowCount": quality["rawRows"],
            "acceptedRawRowCount": quality["acceptedRawRows"],
            "rejectionCounts": quality["rejectionCounts"],
            "preparedBands": sorted(band_series),
        }
        source_records.append(record)

        print(
            f"      raw rows={record['rawRowCount']} | quality rows={record['acceptedRawRowCount']} | "
            f"prepared bands={record['preparedBands']}",
            flush=True,
        )

        for band, series in band_series.items():
            dataset_id = f"{source_dataset_id}-atlas-{role}-{band}-nightly-v1"
            target_name = f"{source_dataset_id} ATLAS {role} {band}-band nightly forced photometry"
            dataset_path = root / f"{_safe(dataset_id)}.json"

            dataset = {
                "id": dataset_id,
                "targetName": target_name,
                "times": np.asarray(series["times"], dtype=np.float32).tolist(),
                "flux": np.asarray(series["flux"], dtype=np.float32).tolist(),
                "frequencySearch": search,
                "reference": {},
                "science": {
                    "role": "atlas-target-image-forced-photometry",
                    "sourceRole": role,
                    "gaiaDR3SourceID": source_id,
                    "band": band,
                    "gaiaPairSeparationArcsec": pair_separation,
                    "nightlyBinned": True,
                    "useReducedTargetImages": True,
                    "differenceImagingUsed": False,
                    "tessDriftExtrapolated": False,
                },
                "source": {
                    "mission": "ATLAS",
                    "archive": "ATLAS Forced Photometry",
                    "filter": band,
                    "distributedSamples": int(series["sampleCount"]),
                    "baselineDays": float(series["baselineDays"]),
                },
            }
            _write_json(dataset_path, dataset)

            prepared = {
                "datasetID": dataset_id,
                "datasetPath": str(dataset_path.resolve()),
                "sourceRole": role,
                "gaiaDR3SourceID": source_id,
                "gaiaGMag": source.get("gMag"),
                "band": band,
                "sampleCount": int(series["sampleCount"]),
                "baselineDays": float(series["baselineDays"]),
                "medianFluxUJy": float(series["medianFluxUJy"]),
                "medianFluxErrorUJy": float(series["medianFluxErrorUJy"]),
                "medianNightlySNR": float(series["medianNightlySNR"]),
                "medianRawPointsPerNight": float(series["medianRawPointsPerNight"]),
            }
            prepared_series.append(prepared)
            dataset_entries.append(
                {
                    "id": dataset_id,
                    "path": str(dataset_path.resolve()),
                    "targetName": target_name,
                }
            )
            print(
                f"      {band}: {series['sampleCount']} nights | "
                f"baseline={series['baselineDays']:.1f} d | "
                f"median nightly SNR={series['medianNightlySNR']:.1f}",
                flush=True,
            )

    project_id: str | None = None
    project_path: str | None = None

    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "atlas-forced-photometry-v1"
        )
        manifest = {
            "id": project_id,
            "name": f"{source_project_id} — ATLAS source-resolved forced photometry",
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": "atlas-source-resolved-forced-photometry",
                "archive": "ATLAS Forced Photometry",
                "workerSemantics": (
                    "ATLAS target-image PSF forced photometry is extracted at the two frozen Gaia coordinates, "
                    "quality filtered, nightly binned per filter, and only then distributed as ordinary "
                    "Lomb-Scargle light curves over the frozen residual-frequency band."
                ),
                "differenceImagingUsed": False,
                "tessDriftExtrapolated": False,
            },
        }
        manifest_path = root / f"{_safe(project_id)}.json"
        _write_json(manifest_path, manifest)
        project_path = str(manifest_path.resolve())

    work_units_per_dataset = math.ceil(total_frequencies / per_work)

    return {
        "available": bool(dataset_entries),
        "version": "openstar.tess-atlas-forced-photometry-preparation.v1",
        "archive": "ATLAS Forced Photometry",
        "apiBaseURL": ATLAS_BASE_URL,
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-atlas-nightly-source-resolved-forced-photometry",
        "sourcePair": external_high_resolution_summary.get("sourcePair"),
        "sourceDefinitions": sources,
        "gaiaPairSeparationArcsec": pair_separation,
        "frequencySearch": search,
        "mjdMinimum": ATLAS_MJD_MIN,
        "useReducedTargetImages": True,
        "differenceImagingUsed": False,
        "tessDriftExtrapolated": False,
        "sourceRecords": source_records,
        "preparedSeries": prepared_series,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "qualityGuard": {
            "minimumRawSNR": MIN_RAW_SNR,
            "maximumReducedChiSquared": MAX_REDUCED_CHI_SQUARED,
            "minimumBandNights": MIN_BAND_NIGHTS,
            "minimumBandBaselineDays": MIN_BAND_BASELINE_DAYS,
            "minimumCrossBandSupport": MIN_CROSS_BAND_SUPPORT,
            "maximumCrossBandRelativeFrequencySpread": MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD,
            "nightlyBinning": True,
            "differenceImagingUsed": False,
        },
        "interpretationGuard": (
            "v20.24 uses ATLAS calibrated target-image forced photometry rather than the southern difference-image "
            "light curve, avoiding dependence on the documented southern template change. The corrected Gaia source "
            "separation is used directly. Raw measurements must pass tphot error, reduced-chi-square, positive-flux, "
            "and SNR guards before nightly inverse-variance binning. Each ATLAS filter is searched independently only "
            "inside the frozen TESS residual-frequency band. A source is supported only when accepted residual-band "
            "variability recurs at consistent frequency in at least two independent filters. The TESS frequency-drift "
            "law is not extrapolated into ATLAS epochs."
        ),
    }


def interpret_atlas_forced_photometry_project(
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

    target = _summarize_atlas_source("target-control", results)
    counterpart = _summarize_atlas_source("catalog-counterpart", results)

    target_supported = bool(target.get("sourceSupported"))
    counterpart_supported = bool(counterpart.get("sourceSupported"))
    target_suggestive = bool(target.get("sourceSuggestive"))
    counterpart_suggestive = bool(counterpart.get("sourceSuggestive"))

    prepared_roles = {
        str(item.get("sourceRole"))
        for item in preparation.get("preparedSeries") or []
    }

    if counterpart_supported and target_supported:
        classification = "ATLAS_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_AND_COUNTERPART_SUPPORTED_BY_ATLAS_FORCED_PHOTOMETRY"
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    elif counterpart_supported:
        classification = "ATLAS_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED_BY_ATLAS_FORCED_PHOTOMETRY"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_supported:
        classification = "ATLAS_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        origin = "TARGET_SUPPORTED_BY_ATLAS_FORCED_PHOTOMETRY"
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif counterpart_suggestive and not target_suggestive:
        classification = "ATLAS_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE_BY_ATLAS_FORCED_PHOTOMETRY"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif target_suggestive and not counterpart_suggestive:
        classification = "ATLAS_TARGET_RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        origin = "TARGET_SUGGESTIVE_BY_ATLAS_FORCED_PHOTOMETRY"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif not prepared_roles:
        classification = "ATLAS_NO_QUALIFYING_FORCED_PHOTOMETRY_TIME_SERIES"
        origin = "UNRESOLVED_ATLAS_FORCED_PHOTOMETRY_QUALITY_OR_SENSITIVITY_LIMIT"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    else:
        classification = "ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        origin = "ARCHIVAL_ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"

    return {
        "version": "openstar.tess-atlas-forced-photometry.v1",
        "archive": preparation.get("archive"),
        "sourcePair": preparation.get("sourcePair"),
        "sourceDefinitions": preparation.get("sourceDefinitions"),
        "gaiaPairSeparationArcsec": preparation.get("gaiaPairSeparationArcsec"),
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "mjdMinimum": preparation.get("mjdMinimum"),
        "useReducedTargetImages": True,
        "differenceImagingUsed": False,
        "tessDriftExtrapolated": False,
        "sourceRecords": preparation.get("sourceRecords") or [],
        "componentResults": results,
        "targetControl": target,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "qualityGuard": preparation.get("qualityGuard"),
        "interpretationGuard": preparation.get("interpretationGuard"),
    }
