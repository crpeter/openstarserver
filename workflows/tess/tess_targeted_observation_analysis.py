from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
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

ANALYSIS_VERSION = "openstar.tess-targeted-observation-analysis.v1"

ALLOWED_SOURCE_ROLES = (
    "target-control",
    "catalog-counterpart",
)
ALLOWED_EXPOSURE_TIERS = (
    "short",
    "deep",
)
ACCEPTED_QUALITY_FLAGS = {
    "",
    "OK",
    "GOOD",
    "PASS",
    "VALID",
}

TARGET_ROLE = "target-control"
COUNTERPART_ROLE = "catalog-counterpart"

TARGET_ANALYSIS_TIER = "short"
COUNTERPART_ANALYSIS_TIER = "deep"

MINIMUM_GLOBAL_PEAK_PROMINENCE = 2.0


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _parse_float(value: Any) -> float | None:
    text = _text(value)
    if text is None:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _parse_bool(value: Any) -> bool | None:
    text = (_text(value) or "").lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input_csv(observations_path: str | Path) -> Path:
    path = Path(observations_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise RuntimeError(
                "v20.29 observations input must be a CSV file or a directory "
                "containing exactly one observation CSV."
            )
        return path

    preferred_names = (
        "openstar-targeted-observations.csv",
        "targeted-observations.csv",
        "observations.csv",
    )
    for name in preferred_names:
        candidate = path / name
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    candidates = sorted(
        item.resolve()
        for item in path.glob("*.csv")
        if item.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise RuntimeError(
            f"No observation CSV found in {path}. "
            "Use the v20.28 ingest template as the starting schema."
        )

    raise RuntimeError(
        f"Multiple CSV files found in {path}; name the intended manifest "
        "openstar-targeted-observations.csv or pass the CSV path directly."
    )


def _load_observatory_sidecar(csv_path: Path) -> dict[str, dict[str, float]]:
    candidates = (
        csv_path.parent / "observatories.json",
        csv_path.parent / "openstar-observatories.json",
    )
    sidecar = next(
        (path for path in candidates if path.exists()),
        None,
    )
    if sidecar is None:
        return {}

    with sidecar.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Observatory sidecar must be a JSON object: {sidecar}"
        )

    result: dict[str, dict[str, float]] = {}
    for code, value in payload.items():
        if not isinstance(value, dict):
            continue
        lat = _parse_float(
            value.get("latitudeDeg")
            if value.get("latitudeDeg") is not None
            else value.get("latDeg")
        )
        lon = _parse_float(
            value.get("longitudeDeg")
            if value.get("longitudeDeg") is not None
            else value.get("lonDeg")
        )
        elevation = _parse_float(
            value.get("elevationM")
            if value.get("elevationM") is not None
            else value.get("heightM")
        )
        if lat is None or lon is None or elevation is None:
            continue
        result[str(code).strip()] = {
            "latitudeDeg": float(lat),
            "longitudeDeg": float(lon),
            "elevationM": float(elevation),
        }

    return result


def _utc_to_bjd_tdb(
    utc_mid: str,
    *,
    source_ra_deg: float,
    source_dec_deg: float,
    observatory: dict[str, float],
) -> float:
    try:
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        import astropy.units as u
    except Exception as exc:
        raise RuntimeError(
            "UTC->BJD_TDB fallback requires astropy. "
            "Prefer supplying time_bjd_tdb directly."
        ) from exc

    location = EarthLocation.from_geodetic(
        lon=float(observatory["longitudeDeg"]) * u.deg,
        lat=float(observatory["latitudeDeg"]) * u.deg,
        height=float(observatory["elevationM"]) * u.m,
    )
    source = SkyCoord(
        ra=float(source_ra_deg) * u.deg,
        dec=float(source_dec_deg) * u.deg,
        frame="icrs",
    )
    time = Time(
        utc_mid,
        scale="utc",
        location=location,
    )
    correction = time.light_travel_time(source)
    return float((time.tdb + correction).jd)


def _source_by_role(
    observation_plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    geometry = observation_plan.get("sourceGeometry") or {}
    target = geometry.get("target") or {}
    counterpart = geometry.get("counterpart") or {}

    result = {
        TARGET_ROLE: target,
        COUNTERPART_ROLE: counterpart,
    }

    for role, source in result.items():
        source_id = _int(source.get("gaiaDR3SourceID"))
        ra = _float(source.get("raDeg"))
        dec = _float(source.get("decDeg"))
        if source_id is None or ra is None or dec is None:
            raise RuntimeError(
                f"v20.29 frozen source metadata missing for {role}."
            )

    return result


def _load_rows(
    csv_path: Path,
) -> list[dict[str, str]]:
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = {
            str(name).strip()
            for name in (reader.fieldnames or [])
            if name is not None
        }

        required = {
            "exposure_id",
            "visit_id",
            "source_role",
            "gaia_dr3_source_id",
            "filter",
            "exposure_tier",
            "exposure_seconds",
            "flux",
            "flux_error",
            "differential_mag",
            "differential_mag_error",
            "fwhm_arcsec",
            "saturated",
            "contaminated",
            "quality_flag",
            "fits_path",
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise RuntimeError(
                "Observation CSV is missing v20.28 contract columns: "
                + ", ".join(missing)
            )

        if (
            "time_bjd_tdb" not in fieldnames
            and "time_utc_mid" not in fieldnames
        ):
            raise RuntimeError(
                "Observation CSV must contain time_bjd_tdb or time_utc_mid."
            )

        return [
            {
                str(key).strip(): (
                    "" if value is None else str(value).strip()
                )
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]


def _signal_and_error(
    row: dict[str, str],
) -> tuple[float, float, float, str] | None:
    flux = _parse_float(row.get("flux"))
    flux_error = _parse_float(row.get("flux_error"))

    if (
        flux is not None
        and flux_error is not None
        and flux_error > 0
    ):
        snr = abs(float(flux)) / float(flux_error)
        return (
            float(flux),
            float(flux_error),
            float(snr),
            "flux",
        )

    differential_mag = _parse_float(
        row.get("differential_mag")
    )
    differential_mag_error = _parse_float(
        row.get("differential_mag_error")
    )

    if (
        differential_mag is not None
        and differential_mag_error is not None
        and differential_mag_error > 0
    ):
        # Magnitude decreases as flux increases; negate it so positive
        # variability has the same sense as a flux-like series.
        snr = 1.0857362047581296 / float(
            differential_mag_error
        )
        return (
            -float(differential_mag),
            float(differential_mag_error),
            float(snr),
            "negative-differential-magnitude",
        )

    return None


def _analysis_night(bjd_tdb: float) -> int:
    # Deterministic UTC-like Julian-day boundary. This is intentionally
    # independent of the measured periodogram and of observatory longitude.
    return int(math.floor(float(bjd_tdb) + 0.5))


def _normalized_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad

    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))

    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            "Qualified targeted-observation light curve has zero finite "
            "variability scale."
        )

    return (values - center) / scale


def _weighted_mean(
    values: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float]:
    weights = 1.0 / np.square(errors)
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise RuntimeError(
            "Invalid inverse-variance weight in targeted observations."
        )
    return (
        float(np.sum(weights * values) / total_weight),
        float(math.sqrt(1.0 / total_weight)),
    )


def _copy_input_snapshot(
    csv_path: Path,
    artifact_root: Path,
) -> tuple[Path, str]:
    digest = _sha256_file(csv_path)
    destination = (
        artifact_root
        / "input"
        / f"targeted-observations-{digest[:16]}.csv"
    )
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if not destination.exists():
        shutil.copy2(csv_path, destination)
    return destination.resolve(), digest


def _fits_manifest(
    *,
    csv_path: Path,
    rows: list[dict[str, str]],
    output_path: Path,
) -> dict[str, Any]:
    unique_paths: dict[str, Path] = {}
    for row in rows:
        raw = _text(row.get("fits_path"))
        if raw is None:
            continue

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (csv_path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()

        unique_paths[str(candidate)] = candidate

    entries: list[dict[str, Any]] = []
    for path in unique_paths.values():
        if not path.exists() or not path.is_file():
            entries.append(
                {
                    "path": str(path),
                    "exists": False,
                    "sha256": None,
                    "sizeBytes": None,
                }
            )
            continue

        entries.append(
            {
                "path": str(path),
                "exists": True,
                "sha256": _sha256_file(path),
                "sizeBytes": int(path.stat().st_size),
            }
        )

    manifest = {
        "version": "openstar.targeted-observation-fits-manifest.v1",
        "csvPath": str(csv_path.resolve()),
        "fitsFileCount": len(entries),
        "missingFitsFileCount": sum(
            1 for item in entries if not item["exists"]
        ),
        "files": entries,
    }
    _write_json(output_path, manifest)
    return manifest


def _qualify_rows(
    *,
    rows: list[dict[str, str]],
    csv_path: Path,
    observation_plan: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    source_by_role = _source_by_role(observation_plan)
    observatories = _load_observatory_sidecar(csv_path)

    geometry = observation_plan.get("sourceGeometry") or {}
    maximum_fwhm = _float(
        geometry.get("maximumFwhmArcsec")
    )
    if maximum_fwhm is None:
        raise RuntimeError(
            "v20.29 observation plan is missing maximumFwhmArcsec."
        )

    exposure = observation_plan.get("exposureStrategy") or {}
    target_short = exposure.get("targetShortTier") or {}
    counterpart_deep = exposure.get("counterpartDeepTier") or {}

    target_minimum_snr = _float(
        target_short.get("minimumSNR")
    )
    counterpart_minimum_snr = _float(
        counterpart_deep.get("minimumSNR")
    )
    if (
        target_minimum_snr is None
        or counterpart_minimum_snr is None
    ):
        raise RuntimeError(
            "v20.29 observation plan is missing frozen SNR requirements."
        )

    diagnostics: dict[str, Any] = {
        "inputRows": int(len(rows)),
        "rejectionReasons": defaultdict(int),
        "roleTierAcceptedRows": defaultdict(int),
        "timing": {
            "directBjdRows": 0,
            "utcConvertedRows": 0,
            "utcRowsMissingObservatory": 0,
        },
    }

    parsed_rows: list[dict[str, Any]] = []
    seen_source_exposure: set[tuple[str, str]] = set()

    for row_index, row in enumerate(rows, start=2):
        if (
            (_text(row.get("quality_flag")) or "").upper()
            == "TEMPLATE_ROW_DELETE_ME"
        ):
            diagnostics["rejectionReasons"][
                "template-row"
            ] += 1
            continue

        exposure_id = _text(row.get("exposure_id"))
        visit_id = _text(row.get("visit_id"))
        role = _text(row.get("source_role"))
        source_id = _parse_int(
            row.get("gaia_dr3_source_id")
        )
        band = _text(row.get("filter"))
        tier = (
            _text(row.get("exposure_tier")) or ""
        ).lower()
        exposure_seconds = _parse_float(
            row.get("exposure_seconds")
        )
        fwhm_arcsec = _parse_float(
            row.get("fwhm_arcsec")
        )
        saturated = _parse_bool(
            row.get("saturated")
        )
        contaminated = _parse_bool(
            row.get("contaminated")
        )
        quality_flag = (
            _text(row.get("quality_flag")) or ""
        ).upper()

        if (
            exposure_id is None
            or visit_id is None
            or role is None
            or source_id is None
            or band is None
            or exposure_seconds is None
            or exposure_seconds <= 0
        ):
            diagnostics["rejectionReasons"][
                "missing-required-identity-or-exposure-field"
            ] += 1
            continue

        if role not in ALLOWED_SOURCE_ROLES:
            diagnostics["rejectionReasons"][
                "unexpected-source-role"
            ] += 1
            continue

        expected_source = source_by_role[role]
        expected_id = int(
            expected_source["gaiaDR3SourceID"]
        )
        if source_id != expected_id:
            diagnostics["rejectionReasons"][
                "gaia-source-id-mismatch"
            ] += 1
            continue

        if tier not in ALLOWED_EXPOSURE_TIERS:
            diagnostics["rejectionReasons"][
                "unexpected-exposure-tier"
            ] += 1
            continue

        duplicate_key = (
            exposure_id,
            role,
        )
        if duplicate_key in seen_source_exposure:
            diagnostics["rejectionReasons"][
                "duplicate-source-exposure"
            ] += 1
            continue
        seen_source_exposure.add(duplicate_key)

        if quality_flag not in ACCEPTED_QUALITY_FLAGS:
            diagnostics["rejectionReasons"][
                "quality-flag-rejected"
            ] += 1
            continue

        if fwhm_arcsec is None:
            diagnostics["rejectionReasons"][
                "missing-fwhm"
            ] += 1
            continue
        if fwhm_arcsec > float(maximum_fwhm):
            diagnostics["rejectionReasons"][
                "fwhm-above-frozen-maximum"
            ] += 1
            continue

        if saturated is None:
            diagnostics["rejectionReasons"][
                "saturation-state-unknown"
            ] += 1
            continue

        if contaminated is None:
            diagnostics["rejectionReasons"][
                "contamination-state-unknown"
            ] += 1
            continue

        if contaminated:
            diagnostics["rejectionReasons"][
                "contaminated"
            ] += 1
            continue

        signal = _signal_and_error(row)
        if signal is None:
            diagnostics["rejectionReasons"][
                "missing-valid-photometry"
            ] += 1
            continue
        signal_value, signal_error, snr, signal_kind = signal

        bjd_tdb = _parse_float(
            row.get("time_bjd_tdb")
        )
        if bjd_tdb is not None:
            diagnostics["timing"][
                "directBjdRows"
            ] += 1
        else:
            utc_mid = _text(row.get("time_utc_mid"))
            observatory_code = _text(
                row.get("observatory_code")
            )
            observatory = (
                observatories.get(observatory_code)
                if observatory_code is not None
                else None
            )
            if utc_mid is None or observatory is None:
                diagnostics["timing"][
                    "utcRowsMissingObservatory"
                ] += 1
                diagnostics["rejectionReasons"][
                    "missing-bjd-and-observatory-fallback"
                ] += 1
                continue

            bjd_tdb = _utc_to_bjd_tdb(
                utc_mid,
                source_ra_deg=float(
                    expected_source["raDeg"]
                ),
                source_dec_deg=float(
                    expected_source["decDeg"]
                ),
                observatory=observatory,
            )
            diagnostics["timing"][
                "utcConvertedRows"
            ] += 1

        analysis_tier = (
            TARGET_ANALYSIS_TIER
            if role == TARGET_ROLE
            else COUNTERPART_ANALYSIS_TIER
        )

        minimum_snr = (
            float(target_minimum_snr)
            if role == TARGET_ROLE
            else float(counterpart_minimum_snr)
        )

        # Rows from the non-analysis tier remain useful for visit-pair
        # validation, but they are not admitted to the source light curve.
        analysis_eligible = bool(
            tier == analysis_tier
            and not saturated
            and snr >= minimum_snr
        )

        if tier == analysis_tier and saturated:
            diagnostics["rejectionReasons"][
                f"{role}-analysis-tier-saturated"
            ] += 1
        elif tier == analysis_tier and snr < minimum_snr:
            diagnostics["rejectionReasons"][
                f"{role}-analysis-tier-snr-below-frozen-minimum"
            ] += 1

        parsed = {
            "rowIndex": int(row_index),
            "exposureID": exposure_id,
            "visitID": visit_id,
            "sourceRole": role,
            "gaiaDR3SourceID": source_id,
            "band": band,
            "exposureTier": tier,
            "exposureSeconds": float(
                exposure_seconds
            ),
            "bjdTdb": float(bjd_tdb),
            "analysisNight": _analysis_night(
                float(bjd_tdb)
            ),
            "signal": float(signal_value),
            "signalError": float(signal_error),
            "signalKind": signal_kind,
            "snr": float(snr),
            "fwhmArcsec": float(fwhm_arcsec),
            "saturated": bool(saturated),
            "contaminated": bool(contaminated),
            "qualityFlag": quality_flag,
            "fitsPath": _text(
                row.get("fits_path")
            ),
            "analysisEligibleBeforeVisitPair": (
                analysis_eligible
            ),
        }
        parsed_rows.append(parsed)

        if analysis_eligible:
            diagnostics["roleTierAcceptedRows"][
                f"{role}:{tier}"
            ] += 1

    # A v20.28 visit is paired only when the same visit + filter contains
    # both a short and a deep exposure. This is checked without using the
    # measured signal.
    tiers_by_visit_band: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    for item in parsed_rows:
        tiers_by_visit_band[
            (
                str(item["visitID"]),
                str(item["band"]),
            )
        ].add(
            str(item["exposureTier"])
        )

    paired_visit_bands = {
        key
        for key, tiers in tiers_by_visit_band.items()
        if {"short", "deep"}.issubset(tiers)
    }

    qualified: list[dict[str, Any]] = []
    for item in parsed_rows:
        if not item[
            "analysisEligibleBeforeVisitPair"
        ]:
            continue

        key = (
            str(item["visitID"]),
            str(item["band"]),
        )
        if key not in paired_visit_bands:
            diagnostics["rejectionReasons"][
                "analysis-row-from-unpaired-visit"
            ] += 1
            continue

        qualified.append(item)

    diagnostics["parsedRows"] = int(
        len(parsed_rows)
    )
    diagnostics["pairedVisitFilterCount"] = int(
        len(paired_visit_bands)
    )
    diagnostics["qualifiedAnalysisRows"] = int(
        len(qualified)
    )
    diagnostics["rejectionReasons"] = dict(
        sorted(
            diagnostics["rejectionReasons"].items()
        )
    )
    diagnostics["roleTierAcceptedRows"] = dict(
        sorted(
            diagnostics["roleTierAcceptedRows"].items()
        )
    )

    return qualified, diagnostics


def _nightly_series(
    qualified_rows: list[dict[str, Any]],
    *,
    minimum_visits_per_night: int,
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[str, Any],
]:
    grouped: dict[
        tuple[str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    signal_kinds_by_source_band: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    for row in qualified_rows:
        source_band = (
            str(row["sourceRole"]),
            str(row["band"]),
        )
        signal_kinds_by_source_band[
            source_band
        ].add(
            str(row["signalKind"])
        )
        grouped[
            (
                source_band[0],
                source_band[1],
                int(row["analysisNight"]),
            )
        ].append(row)

    mixed_signal_kinds = {
        key: sorted(kinds)
        for key, kinds in signal_kinds_by_source_band.items()
        if len(kinds) > 1
    }
    if mixed_signal_kinds:
        raise RuntimeError(
            "v20.29 refuses to combine incompatible photometry representations "
            "within one source/filter series. Use either calibrated flux+error "
            "or differential_mag+error consistently for each source/filter. "
            f"Mixed series: {mixed_signal_kinds}"
        )

    by_source_band: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    diagnostics = {
        "nightsRejectedForTooFewVisits": 0,
        "qualifiedNightCountBySourceBand": {},
    }

    for (
        role,
        band,
        night,
    ), rows in grouped.items():
        unique_visits = {
            str(item["visitID"])
            for item in rows
        }
        if len(unique_visits) < minimum_visits_per_night:
            diagnostics[
                "nightsRejectedForTooFewVisits"
            ] += 1
            continue

        values = np.asarray(
            [float(item["signal"]) for item in rows],
            dtype=np.float64,
        )
        errors = np.asarray(
            [float(item["signalError"]) for item in rows],
            dtype=np.float64,
        )
        times = np.asarray(
            [float(item["bjdTdb"]) for item in rows],
            dtype=np.float64,
        )

        nightly_signal, nightly_error = _weighted_mean(
            values,
            errors,
        )
        weights = 1.0 / np.square(errors)
        total_weight = float(np.sum(weights))
        nightly_time = float(
            np.sum(weights * times) / total_weight
        )

        by_source_band[(role, band)].append(
            {
                "analysisNight": int(night),
                "bjdTdb": nightly_time,
                "signal": nightly_signal,
                "signalError": nightly_error,
                "visitCount": int(
                    len(unique_visits)
                ),
                "rowCount": int(len(rows)),
            }
        )

    for key in by_source_band:
        by_source_band[key].sort(
            key=lambda item: float(
                item["bjdTdb"]
            )
        )
        role, band = key
        diagnostics[
            "qualifiedNightCountBySourceBand"
        ][f"{role}:{band}"] = int(
            len(by_source_band[key])
        )

    return by_source_band, diagnostics


def _series_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "nightCount": 0,
            "baselineDays": 0.0,
            "firstBjdTdb": None,
            "lastBjdTdb": None,
        }

    times = [
        float(item["bjdTdb"])
        for item in rows
    ]
    return {
        "nightCount": int(len(rows)),
        "baselineDays": float(
            max(times) - min(times)
        ),
        "firstBjdTdb": float(min(times)),
        "lastBjdTdb": float(max(times)),
    }


def _campaign_contract(
    *,
    by_source_band: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ],
    observation_plan: dict[str, Any],
) -> dict[str, Any]:
    cadence = observation_plan.get("cadence") or {}
    filter_strategy = (
        observation_plan.get("filterStrategy") or {}
    )

    minimum_baseline = _float(
        cadence.get("minimumBaselineDays")
    )
    minimum_nights = _int(
        cadence.get("minimumDistinctNights")
    )
    minimum_filters = _int(
        filter_strategy.get("minimumFilters")
    )

    if (
        minimum_baseline is None
        or minimum_nights is None
        or minimum_filters is None
    ):
        raise RuntimeError(
            "v20.29 observation plan is missing frozen campaign requirements."
        )

    source_summaries: dict[str, Any] = {}
    complete_roles: list[str] = []

    for role in ALLOWED_SOURCE_ROLES:
        filters: dict[str, Any] = {}
        qualified_filters: list[str] = []

        for (
            source_role,
            band,
        ), rows in sorted(
            by_source_band.items()
        ):
            if source_role != role:
                continue

            summary = _series_summary(rows)
            summary["globallyQualified"] = bool(
                summary["nightCount"] >= minimum_nights
                and summary["baselineDays"] >= minimum_baseline
            )
            filters[band] = summary

            if summary["globallyQualified"]:
                qualified_filters.append(band)

        role_complete = bool(
            len(qualified_filters) >= minimum_filters
        )
        if role_complete:
            complete_roles.append(role)

        source_summaries[role] = {
            "filters": filters,
            "globallyQualifiedFilters": sorted(
                qualified_filters
            ),
            "globallyQualifiedFilterCount": int(
                len(qualified_filters)
            ),
            "roleComplete": role_complete,
        }

    campaign_complete = bool(
        set(ALLOWED_SOURCE_ROLES).issubset(
            set(complete_roles)
        )
    )

    return {
        "campaignComplete": campaign_complete,
        "minimumBaselineDays": float(
            minimum_baseline
        ),
        "minimumDistinctNights": int(
            minimum_nights
        ),
        "minimumFiltersPerSource": int(
            minimum_filters
        ),
        "sourceSummaries": source_summaries,
        "completeRoles": sorted(
            complete_roles
        ),
    }


def _frozen_frequency_search(
    observation_plan: dict[str, Any],
) -> dict[str, Any]:
    provenance = (
        observation_plan.get("provenance") or {}
    )
    search = dict(
        provenance.get("frozenFrequencySearch")
        or {}
    )

    total = _int(
        search.get("totalFrequencies")
        if search.get("totalFrequencies") is not None
        else search.get("frequencyCount")
    )
    per_work = _int(
        search.get("frequenciesPerWorkUnit")
        if search.get("frequenciesPerWorkUnit") is not None
        else search.get("workUnitFrequencyCount")
    )

    if (
        not search
        or total is None
        or total <= 0
        or per_work is None
        or per_work <= 0
    ):
        raise RuntimeError(
            "v20.29 observation plan does not contain the frozen "
            "residual-frequency work definition."
        )

    return search


def _write_dataset(
    *,
    root: Path,
    source_dataset_id: str,
    role: str,
    source_id: int,
    band: str,
    scope: str,
    rows: list[dict[str, Any]],
    frequency_search: dict[str, Any],
    window_index: int | None = None,
    window_start_bjd: float | None = None,
    window_end_bjd: float | None = None,
) -> dict[str, Any]:
    times = np.asarray(
        [float(item["bjdTdb"]) for item in rows],
        dtype=np.float64,
    )
    signals = np.asarray(
        [float(item["signal"]) for item in rows],
        dtype=np.float64,
    )

    standardized = _normalized_signal(
        signals
    )
    local_times = times - float(
        times[0]
    )

    suffix = (
        f"{scope}-{role}-{band}"
        if window_index is None
        else (
            f"{scope}-{role}-{band}-"
            f"window-{window_index}"
        )
    )
    dataset_id = (
        f"{source_dataset_id}-targeted-{suffix}-v1"
    )
    target_name = (
        f"{source_dataset_id} targeted observations "
        f"{scope} {role} {band}"
        + (
            ""
            if window_index is None
            else f" window {window_index}"
        )
    )

    path = root / f"{_safe(dataset_id)}.json"
    dataset = {
        "id": dataset_id,
        "targetName": target_name,
        "times": np.asarray(
            local_times,
            dtype=np.float32,
        ).tolist(),
        "flux": np.asarray(
            standardized,
            dtype=np.float32,
        ).tolist(),
        "frequencySearch": frequency_search,
        "reference": {},
        "science": {
            "role": (
                "targeted-source-resolved-time-series-photometry"
            ),
            "sourceRole": role,
            "gaiaDR3SourceID": int(source_id),
            "filter": band,
            "analysisScope": scope,
            "windowIndex": window_index,
            "windowStartBjdTdb": (
                window_start_bjd
            ),
            "windowEndBjdTdb": (
                window_end_bjd
            ),
            "preregisteredByPlanVersion": (
                "openstar.tess-targeted-observation-plan.v1"
            ),
            "tessDriftExtrapolated": False,
        },
        "source": {
            "mission": "Ground-based targeted photometry",
            "archive": "User-supplied targeted observations",
            "filter": band,
            "distributedSamples": int(
                len(local_times)
            ),
            "baselineDays": float(
                times[-1] - times[0]
            ),
        },
    }
    _write_json(path, dataset)

    return {
        "datasetID": dataset_id,
        "datasetPath": str(path.resolve()),
        "targetName": target_name,
        "sourceRole": role,
        "gaiaDR3SourceID": int(source_id),
        "band": band,
        "analysisScope": scope,
        "windowIndex": window_index,
        "windowStartBjdTdb": (
            window_start_bjd
        ),
        "windowEndBjdTdb": (
            window_end_bjd
        ),
        "nightCount": int(len(rows)),
        "baselineDays": float(
            times[-1] - times[0]
        ),
        "firstBjdTdb": float(times[0]),
        "lastBjdTdb": float(times[-1]),
    }


def build_targeted_observation_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    investigation_id: str,
    observations_path: str | Path,
    observation_plan: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if observation_plan.get("status") != "OBSERVATION_PLAN_READY":
        raise RuntimeError(
            "v20.29 requires a completed v20.28 OBSERVATION_PLAN_READY result."
        )

    if observation_plan.get("recommendedNextTest") != (
        "COLLECT_TARGETED_TIME_SERIES_PHOTOMETRY"
    ):
        raise RuntimeError(
            "v20.28 does not point to targeted time-series collection."
        )

    csv_path = _resolve_input_csv(
        observations_path
    )
    raw_rows = _load_rows(csv_path)

    root = (
        Path(output_dir)
        / "targeted-observation-analysis"
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_path, input_sha256 = (
        _copy_input_snapshot(
            csv_path,
            root,
        )
    )

    fits_manifest_path = (
        root
        / f"fits-manifest-{input_sha256[:16]}.json"
    )
    fits_manifest = _fits_manifest(
        csv_path=csv_path,
        rows=raw_rows,
        output_path=fits_manifest_path,
    )

    qualified_rows, row_diagnostics = (
        _qualify_rows(
            rows=raw_rows,
            csv_path=csv_path,
            observation_plan=observation_plan,
        )
    )

    cadence = observation_plan.get("cadence") or {}
    minimum_visits = _int(
        cadence.get(
            "minimumVisitsPerObservedNight"
        )
    )
    if minimum_visits is None or minimum_visits <= 0:
        raise RuntimeError(
            "v20.29 observation plan is missing minimum visits per night."
        )

    by_source_band, nightly_diagnostics = (
        _nightly_series(
            qualified_rows,
            minimum_visits_per_night=int(
                minimum_visits
            ),
        )
    )

    contract = _campaign_contract(
        by_source_band=by_source_band,
        observation_plan=observation_plan,
    )

    source_by_role = _source_by_role(
        observation_plan
    )
    frequency_search = _frozen_frequency_search(
        observation_plan
    )

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []

    global_root = root / "datasets" / "global"
    window_root = root / "datasets" / "windows"

    for role in ALLOWED_SOURCE_ROLES:
        role_summary = (
            contract["sourceSummaries"][role]
        )
        globally_qualified_filters = set(
            role_summary[
                "globallyQualifiedFilters"
            ]
        )
        source_id = int(
            source_by_role[role][
                "gaiaDR3SourceID"
            ]
        )

        for band in sorted(
            globally_qualified_filters
        ):
            rows = by_source_band.get(
                (role, band),
                [],
            )
            if not rows:
                continue

            prepared = _write_dataset(
                root=global_root,
                source_dataset_id=source_dataset_id,
                role=role,
                source_id=source_id,
                band=band,
                scope="global",
                rows=rows,
                frequency_search=frequency_search,
            )
            prepared_series.append(
                prepared
            )
            dataset_entries.append(
                {
                    "id": prepared["datasetID"],
                    "path": prepared[
                        "datasetPath"
                    ],
                    "targetName": prepared[
                        "targetName"
                    ],
                }
            )

    time_resolved = (
        cadence.get("timeResolvedAnalysis") or {}
    )
    window_days = _float(
        time_resolved.get("fixedWindowDays")
    )
    minimum_window_nights = _int(
        time_resolved.get(
            "minimumQualifiedNightsPerWindow"
        )
    )
    if (
        window_days is None
        or window_days <= 0
        or minimum_window_nights is None
        or minimum_window_nights <= 0
    ):
        raise RuntimeError(
            "v20.29 observation plan is missing fixed time-resolved window rules."
        )

    all_qualified_nights = [
        item
        for rows in by_source_band.values()
        for item in rows
    ]
    campaign_anchor_bjd = (
        min(
            float(item["bjdTdb"])
            for item in all_qualified_nights
        )
        if all_qualified_nights
        else None
    )

    if campaign_anchor_bjd is not None:
        for (
            role,
            band,
        ), rows in sorted(
            by_source_band.items()
        ):
            source_id = int(
                source_by_role[role][
                    "gaiaDR3SourceID"
                ]
            )

            by_window: dict[
                int,
                list[dict[str, Any]],
            ] = defaultdict(list)

            for item in rows:
                index = int(
                    math.floor(
                        (
                            float(
                                item["bjdTdb"]
                            )
                            - campaign_anchor_bjd
                        )
                        / float(window_days)
                    )
                )
                by_window[index].append(
                    item
                )

            for index, window_rows in sorted(
                by_window.items()
            ):
                if (
                    len(window_rows)
                    < minimum_window_nights
                ):
                    continue

                window_rows = sorted(
                    window_rows,
                    key=lambda item: float(
                        item["bjdTdb"]
                    ),
                )
                start = (
                    campaign_anchor_bjd
                    + index * float(
                        window_days
                    )
                )
                end = start + float(
                    window_days
                )

                prepared = _write_dataset(
                    root=window_root,
                    source_dataset_id=source_dataset_id,
                    role=role,
                    source_id=source_id,
                    band=band,
                    scope="window",
                    rows=window_rows,
                    frequency_search=frequency_search,
                    window_index=index,
                    window_start_bjd=start,
                    window_end_bjd=end,
                )
                prepared_series.append(
                    prepared
                )
                dataset_entries.append(
                    {
                        "id": prepared[
                            "datasetID"
                        ],
                        "path": prepared[
                            "datasetPath"
                        ],
                        "targetName": prepared[
                            "targetName"
                        ],
                    }
                )

    total_frequencies = _int(
        frequency_search.get(
            "totalFrequencies"
        )
        if frequency_search.get(
            "totalFrequencies"
        )
        is not None
        else frequency_search.get(
            "frequencyCount"
        )
    )
    per_work = _int(
        frequency_search.get(
            "frequenciesPerWorkUnit"
        )
        if frequency_search.get(
            "frequenciesPerWorkUnit"
        )
        is not None
        else frequency_search.get(
            "workUnitFrequencyCount"
        )
    )
    if (
        total_frequencies is None
        or per_work is None
        or total_frequencies <= 0
        or per_work <= 0
    ):
        raise RuntimeError(
            "Invalid frozen frequency work geometry."
        )

    work_units_per_dataset = int(
        math.ceil(
            total_frequencies / per_work
        )
    )

    project_id: str | None = None
    project_path: str | None = None

    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation."
            f"{_safe(investigation_id)}."
            f"targeted-observation-analysis-"
            f"{input_sha256[:12]}"
        )

        manifest = {
            "id": project_id,
            "name": (
                f"{source_project_id} — targeted "
                "source-resolved observation analysis"
            ),
            "workloadID": (
                GENERIC_LOMB_SCARGLE_WORKLOAD_ID
            ),
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": (
                    "preregistered-targeted-source-"
                    "resolved-photometry-analysis"
                ),
                "observationInputSHA256": (
                    input_sha256
                ),
                "workerSemantics": (
                    "Input photometry is validated locally "
                    "against the frozen v20.28 contract. "
                    "Workers execute ordinary Lomb-Scargle "
                    "only over the frozen residual-frequency "
                    "band for globally qualified source/filter "
                    "series and preregistered fixed windows."
                ),
                "tessDriftExtrapolated": False,
            },
        }

        manifest_path = (
            root
            / f"{_safe(project_id)}.json"
        )
        _write_json(
            manifest_path,
            manifest,
        )
        project_path = str(
            manifest_path.resolve()
        )

    diagnostics = {
        "version": (
            "openstar.tess-targeted-observation-ingest-diagnostics.v1"
        ),
        "inputCSV": str(csv_path),
        "snapshotCSV": str(snapshot_path),
        "inputSHA256": input_sha256,
        "fitsManifestPath": str(
            fits_manifest_path.resolve()
        ),
        "fitsManifest": fits_manifest,
        "rowDiagnostics": row_diagnostics,
        "nightlyDiagnostics": nightly_diagnostics,
        "campaignContract": contract,
        "campaignAnchorBjdTdb": (
            campaign_anchor_bjd
        ),
        "fixedWindowDays": float(
            window_days
        ),
        "preparedDatasetCount": int(
            len(dataset_entries)
        ),
    }

    diagnostics_path = (
        root
        / f"ingest-diagnostics-{input_sha256[:16]}.json"
    )
    _write_json(
        diagnostics_path,
        diagnostics,
    )

    return {
        "version": (
            "openstar.tess-targeted-observation-analysis-preparation.v1"
        ),
        "status": (
            "CAMPAIGN_READY_FOR_PREREGISTERED_ANALYSIS"
            if contract["campaignComplete"]
            else "CAMPAIGN_INCOMPLETE"
        ),
        "sourceProjectID": source_project_id,
        "sourceDatasetID": source_dataset_id,
        "observationInputPath": str(
            csv_path.resolve()
        ),
        "observationInputSHA256": (
            input_sha256
        ),
        "snapshotCSVPath": str(
            snapshot_path
        ),
        "fitsManifestPath": str(
            fits_manifest_path.resolve()
        ),
        "diagnosticsPath": str(
            diagnostics_path.resolve()
        ),
        "campaignContract": contract,
        "campaignAnchorBjdTdb": (
            campaign_anchor_bjd
        ),
        "frequencySearch": frequency_search,
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": (
            GENERIC_LOMB_SCARGLE_WORKLOAD_ID
        ),
        "workerSemantics": (
            "generic-lomb-scargle-on-preregistered-targeted-photometry"
        ),
        "preparedSeries": prepared_series,
        "workUnitsPerDataset": (
            work_units_per_dataset
        ),
        "totalWorkUnits": int(
            len(dataset_entries)
            * work_units_per_dataset
        ),
        "analysisContract": (
            observation_plan.get(
                "analysisContract"
            )
            or {}
        ),
        "tessDriftExtrapolated": False,
        "recommendedNextTest": (
            "RUN_PREREGISTERED_TARGETED_PHOTOMETRY_ANALYSIS"
            if contract["campaignComplete"]
            else "CONTINUE_COLLECTING_TARGETED_TIME_SERIES_PHOTOMETRY"
        ),
    }


def _dataset_result(
    project_dataset: dict[str, Any],
    prepared: dict[str, Any],
    *,
    minimum_prominence: float,
) -> dict[str, Any]:
    frequency = _float(
        project_dataset.get(
            "candidateFrequency"
        )
        if project_dataset.get(
            "candidateFrequency"
        )
        is not None
        else project_dataset.get(
            "bestFrequency"
        )
    )
    period = _float(
        project_dataset.get(
            "candidatePeriodDays"
        )
        if project_dataset.get(
            "candidatePeriodDays"
        )
        is not None
        else project_dataset.get(
            "bestPeriodDays"
        )
    )
    power = _float(
        project_dataset.get(
            "candidatePower"
        )
        if project_dataset.get(
            "candidatePower"
        )
        is not None
        else project_dataset.get(
            "bestPower"
        )
    )
    prominence = _float(
        project_dataset.get(
            "candidatePeakProminenceRatio"
        )
    )
    status = str(
        project_dataset.get(
            "periodStatus"
        )
        or ""
    )
    coverage = project_dataset.get(
        "coverageComplete"
    )
    boundary_hit = bool(
        project_dataset.get(
            "boundaryHit"
        )
        or False
    )

    accepted = bool(
        status == "RELIABLE"
        and (
            coverage is None
            or bool(coverage)
        )
        and not boundary_hit
        and prominence is not None
        and prominence >= minimum_prominence
        and frequency is not None
        and frequency > 0
    )

    return {
        "datasetID": prepared.get(
            "datasetID"
        ),
        "sourceRole": prepared.get(
            "sourceRole"
        ),
        "gaiaDR3SourceID": prepared.get(
            "gaiaDR3SourceID"
        ),
        "band": prepared.get("band"),
        "analysisScope": prepared.get(
            "analysisScope"
        ),
        "windowIndex": prepared.get(
            "windowIndex"
        ),
        "windowStartBjdTdb": (
            prepared.get(
                "windowStartBjdTdb"
            )
        ),
        "windowEndBjdTdb": (
            prepared.get(
                "windowEndBjdTdb"
            )
        ),
        "nightCount": prepared.get(
            "nightCount"
        ),
        "baselineDays": prepared.get(
            "baselineDays"
        ),
        "periodStatus": (
            status or None
        ),
        "periodConfidence": (
            project_dataset.get(
                "periodConfidence"
            )
        ),
        "coverageComplete": coverage,
        "boundaryHit": boundary_hit,
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": (
            prominence
        ),
        "accepted": accepted,
    }


def _relative_frequency_spread(
    frequencies: list[float],
) -> float | None:
    if not frequencies:
        return None
    median = float(
        np.median(
            np.asarray(
                frequencies,
                dtype=np.float64,
            )
        )
    )
    if median <= 0:
        return None
    return float(
        (
            max(frequencies)
            - min(frequencies)
        )
        / median
    )


def _source_summary(
    *,
    role: str,
    results: list[dict[str, Any]],
    analysis_contract: dict[str, Any],
) -> dict[str, Any]:
    cross_filter = (
        analysis_contract.get(
            "crossFilterAcceptance"
        )
        or {}
    )
    time_resolved = (
        analysis_contract.get(
            "timeResolvedAcceptance"
        )
        or {}
    )

    minimum_filters = _int(
        cross_filter.get(
            "minimumAcceptedFilters"
        )
    )
    maximum_spread = _float(
        cross_filter.get(
            "maximumRelativeFrequencySpread"
        )
    )
    minimum_windows = _int(
        time_resolved.get(
            "minimumAcceptedRecurrentWindows"
        )
    )

    if (
        minimum_filters is None
        or maximum_spread is None
        or minimum_windows is None
    ):
        raise RuntimeError(
            "v20.29 analysis contract is incomplete."
        )

    source_results = [
        item
        for item in results
        if item.get("sourceRole") == role
    ]
    global_results = [
        item
        for item in source_results
        if item.get("analysisScope")
        == "global"
    ]
    window_results = [
        item
        for item in source_results
        if item.get("analysisScope")
        == "window"
    ]

    accepted_global = [
        item
        for item in global_results
        if item.get("accepted")
    ]
    accepted_global_frequencies = [
        float(item["candidateFrequency"])
        for item in accepted_global
        if item.get(
            "candidateFrequency"
        )
        is not None
    ]
    global_spread = (
        _relative_frequency_spread(
            accepted_global_frequencies
        )
    )
    global_cross_filter_supported = bool(
        len(accepted_global)
        >= minimum_filters
        and global_spread is not None
        and global_spread
        <= float(maximum_spread)
    )

    accepted_windows = [
        item
        for item in window_results
        if item.get("accepted")
    ]
    recurrent_window_indices = sorted(
        {
            int(item["windowIndex"])
            for item in accepted_windows
            if item.get(
                "windowIndex"
            )
            is not None
        }
    )
    recurrent_windows_supported = bool(
        len(recurrent_window_indices)
        >= minimum_windows
    )

    source_supported = bool(
        global_cross_filter_supported
        and recurrent_windows_supported
    )

    return {
        "sourceRole": role,
        "globalResults": global_results,
        "acceptedGlobalFilters": sorted(
            str(item.get("band"))
            for item in accepted_global
        ),
        "acceptedGlobalFilterCount": int(
            len(accepted_global)
        ),
        "globalRelativeFrequencySpread": (
            global_spread
        ),
        "globalCrossFilterSupported": (
            global_cross_filter_supported
        ),
        "windowResults": window_results,
        "acceptedWindowResults": (
            accepted_windows
        ),
        "acceptedRecurrentWindows": (
            recurrent_window_indices
        ),
        "acceptedRecurrentWindowCount": int(
            len(recurrent_window_indices)
        ),
        "recurrentWindowsSupported": (
            recurrent_windows_supported
        ),
        "sourceSupported": (
            source_supported
        ),
    }


def interpret_targeted_observation_project(
    *,
    project_status: dict[str, Any] | None,
    preparation: dict[str, Any],
) -> dict[str, Any]:
    analysis_contract = (
        preparation.get("analysisContract")
        or {}
    )
    global_acceptance = (
        analysis_contract.get(
            "globalAcceptance"
        )
        or {}
    )
    minimum_prominence = _float(
        global_acceptance.get(
            "minimumIndependentPeakProminenceRatio"
        )
    )
    if minimum_prominence is None:
        minimum_prominence = (
            MINIMUM_GLOBAL_PEAK_PROMINENCE
        )

    prepared_by_id = {
        str(item.get("datasetID")): item
        for item in preparation.get(
            "preparedSeries"
        )
        or []
        if item.get("datasetID")
    }

    results: list[dict[str, Any]] = []

    if project_status is not None:
        for dataset in (
            project_status.get("datasets")
            or []
        ):
            dataset_id = str(
                dataset.get("datasetID")
                or dataset.get("id")
                or ""
            )
            prepared = prepared_by_id.get(
                dataset_id
            )
            if prepared is None:
                continue

            results.append(
                _dataset_result(
                    dataset,
                    prepared,
                    minimum_prominence=float(
                        minimum_prominence
                    ),
                )
            )

    target = _source_summary(
        role=TARGET_ROLE,
        results=results,
        analysis_contract=analysis_contract,
    )
    counterpart = _source_summary(
        role=COUNTERPART_ROLE,
        results=results,
        analysis_contract=analysis_contract,
    )

    campaign_complete = bool(
        (
            preparation.get(
                "campaignContract"
            )
            or {}
        ).get("campaignComplete")
    )

    target_supported = bool(
        target.get("sourceSupported")
    )
    counterpart_supported = bool(
        counterpart.get(
            "sourceSupported"
        )
    )

    if not campaign_complete:
        classification = (
            "TARGETED_OBSERVATION_CAMPAIGN_INCOMPLETE"
        )
        origin = (
            "TARGETED_PHOTOMETRY_NOT_YET_INTERPRETABLE"
        )
        next_test = (
            "CONTINUE_COLLECTING_TARGETED_TIME_SERIES_PHOTOMETRY"
        )
    elif target_supported and counterpart_supported:
        classification = (
            "TARGETED_PHOTOMETRY_TARGET_AND_COUNTERPART_VARIABILITY_SUPPORTED"
        )
        origin = (
            "TARGET_AND_COUNTERPART_SUPPORTED_BY_TARGETED_PHOTOMETRY"
        )
        next_test = (
            "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
        )
    elif counterpart_supported:
        classification = (
            "TARGETED_PHOTOMETRY_COUNTERPART_VARIABILITY_SUPPORTED"
        )
        origin = (
            "CATALOG_COUNTERPART_SUPPORTED_BY_TARGETED_PHOTOMETRY"
        )
        next_test = (
            "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
        )
    elif target_supported:
        classification = (
            "TARGETED_PHOTOMETRY_TARGET_VARIABILITY_SUPPORTED"
        )
        origin = (
            "TARGET_SUPPORTED_BY_TARGETED_PHOTOMETRY"
        )
        next_test = (
            "TARGET_INTRINSIC_RESIDUAL_MODELING"
        )
    else:
        classification = (
            "TARGETED_PHOTOMETRY_RESIDUAL_SOURCE_NOT_CONFIRMED"
        )
        origin = (
            "TARGETED_SOURCE_RESOLVED_CAMPAIGN_DOES_NOT_CONFIRM_TESTED_SOURCES"
        )
        next_test = (
            "REASSESS_RESIDUAL_SOURCE_MODEL"
        )

    return {
        "version": ANALYSIS_VERSION,
        "status": "COMPLETE",
        "observationInputSHA256": (
            preparation.get(
                "observationInputSHA256"
            )
        ),
        "snapshotCSVPath": (
            preparation.get(
                "snapshotCSVPath"
            )
        ),
        "fitsManifestPath": (
            preparation.get(
                "fitsManifestPath"
            )
        ),
        "diagnosticsPath": (
            preparation.get(
                "diagnosticsPath"
            )
        ),
        "campaignContract": (
            preparation.get(
                "campaignContract"
            )
        ),
        "distributedValidation": {
            "workloadID": (
                preparation.get(
                    "workloadID"
                )
            ),
            "workerSemantics": (
                preparation.get(
                    "workerSemantics"
                )
            ),
            "totalWorkUnits": (
                preparation.get(
                    "totalWorkUnits"
                )
            ),
            "frequencySearch": (
                preparation.get(
                    "frequencySearch"
                )
            ),
        },
        "componentResults": results,
        "targetControl": target,
        "catalogCounterpartEvidence": (
            counterpart
        ),
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "mainFamilyAssociationChanged": False,
        "tessDriftExtrapolated": False,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "v20.29 analyzes only measurements that satisfy the frozen v20.28 "
            "source identity, paired-visit, FWHM, exposure-tier, saturation, "
            "contamination, per-exposure SNR, visit-count, baseline, filter, "
            "and cadence requirements. Global and fixed-window datasets use "
            "the already-frozen residual-frequency search band. A source is "
            "supported only when the global search passes the preregistered "
            "cross-filter rule and strict accepted peaks recur in the required "
            "number of fixed campaign windows. An incomplete campaign is never "
            "converted into a negative source result. The established TESS "
            "main-family source association is not changed by this residual-only "
            "experiment."
        ),
    }
