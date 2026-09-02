"""Deeper catalog discovery at a frozen PRF residual position.

This continuation is coordinator-local.  It queries only catalog object tables
and never creates worker work units or reads TESS flux.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from .tess_offset_source import (
    CATALOG_MERGE_RADIUS_ARCSEC,
    MAX_PLAUSIBLE_RADIUS_ARCSEC,
    TARGET_EXCLUSION_RADIUS_ARCSEC,
    _coordinate_separation_arcsec,
)
from .tess_nsc_resolved import _parse_float as _nsc_float
from .tess_nsc_resolved import _parse_int as _nsc_int
from .tess_nsc_resolved import _query_csv as _nsc_query_csv
from .tess_skymapper_resolved import _parse_float as _sky_float
from .tess_skymapper_resolved import _parse_int as _sky_int
from .tess_skymapper_resolved import _tap_csv as _skymapper_tap_csv

METHOD_VERSION = "openstar.tess-deep-catalog-counterpart-identification.v1"
HANDLER_ID = "openstar.tess.deep-catalog-counterpart-identification.analyze"
CURRENT_TRIGGER = "DEEPER_CATALOG_OR_HIGH_RESOLUTION_IMAGING"
MIN_PREFERENCE_MARGIN_ARCSEC = 3.0


def validate_catalog_boundary(summary: dict[str, Any]) -> dict[str, float | str]:
    """Accept only the exact persisted TIC/Gaia no-candidate boundary."""
    if not (
        summary.get("version")
        == "openstar.tess-catalog-counterpart-identification.v1"
        and summary.get("classification") == "NO_USABLE_CATALOG_CANDIDATES"
        and summary.get("counterpartIdentified") is False
        and summary.get("preferredCandidate") is None
        and not (summary.get("plausibleCatalogCandidates") or [])
        and summary.get("physicalMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and summary.get("recommendedNextTest") == CURRENT_TRIGGER
    ):
        raise RuntimeError(
            "Deep-catalog identification requires the exact finalized "
            "TIC/Gaia NO_USABLE_CATALOG_CANDIDATES boundary."
        )
    position = summary.get("searchPosition") or {}
    component_id = str(position.get("componentID") or "").strip()
    try:
        values = {
            "raDeg": float(position["raDeg"]),
            "decDeg": float(position["decDeg"]),
            "targetRaDeg": float(position["targetRaDeg"]),
            "targetDecDeg": float(position["targetDecDeg"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Catalog boundary lacks its persisted PRF search position.") from exc
    if not component_id or not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Catalog boundary has an invalid persisted PRF search position.")
    if not (0.0 <= values["raDeg"] < 360.0 and -90.0 <= values["decDeg"] <= 90.0):
        raise RuntimeError("Catalog boundary search coordinates are outside ICRS bounds.")
    return {"componentID": component_id, **values}


def _query_skymapper(position: dict[str, Any]) -> list[dict[str, str]]:
    radius_deg = MAX_PLAUSIBLE_RADIUS_ARCSEC / 3600.0
    query = f"""
SELECT TOP 200
    object_id, raj2000, dej2000, flags, nimaflags, flags_psf, ngood,
    g_ngood, r_ngood, i_ngood, z_ngood,
    gaia_dr3_id1, gaia_dr3_dist1
FROM dr4.master
WHERE 1 = CONTAINS(
    POINT('ICRS', raj2000, dej2000),
    CIRCLE('ICRS', {float(position['raDeg']):.10f},
                   {float(position['decDeg']):.10f}, {radius_deg:.10f})
)
"""
    return _skymapper_tap_csv(query)


def _query_nsc(position: dict[str, Any]) -> list[dict[str, str]]:
    ra = float(position["raDeg"])
    dec = float(position["decDeg"])
    dec_half = MAX_PLAUSIBLE_RADIUS_ARCSEC / 3600.0
    ra_half = dec_half / max(0.05, abs(math.cos(math.radians(dec))))
    query = f"""
SELECT id, ra, dec, ndet, class_star, flags, gmag, rmag, imag, zmag
FROM nsc_dr2.object
WHERE ra BETWEEN {ra - ra_half:.10f} AND {ra + ra_half:.10f}
  AND dec BETWEEN {dec - dec_half:.10f} AND {dec + dec_half:.10f}
LIMIT 200
"""
    return _nsc_query_csv(query)


def _candidate(
    *, catalog: str, identifier: int | str, ra: float, dec: float,
    position: dict[str, Any], metadata: dict[str, Any],
) -> dict[str, Any] | None:
    residual_separation = _coordinate_separation_arcsec(
        float(position["raDeg"]), float(position["decDeg"]), ra, dec)
    target_separation = _coordinate_separation_arcsec(
        float(position["targetRaDeg"]), float(position["targetDecDeg"]), ra, dec)
    if residual_separation > MAX_PLAUSIBLE_RADIUS_ARCSEC:
        return None
    return {
        "raDeg": float(ra),
        "decDeg": float(dec),
        "separationArcsec": float(residual_separation),
        "targetSeparationArcsec": float(target_separation),
        "isTarget": target_separation <= TARGET_EXCLUSION_RADIUS_ARCSEC,
        "catalogIDs": {
            "skyMapperDR4ObjectID": identifier if catalog == "SkyMapperDR4" else None,
            "nscDR2ObjectID": identifier if catalog == "NSCDR2" else None,
        },
        "catalogRecords": {catalog: metadata},
    }


def _skymapper_candidates(
    rows: list[dict[str, Any]], position: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        object_id = _sky_int(row.get("object_id"))
        ra = _sky_float(row.get("raj2000"))
        dec = _sky_float(row.get("dej2000"))
        if object_id is None or ra is None or dec is None:
            continue
        item = _candidate(
            catalog="SkyMapperDR4", identifier=int(object_id), ra=ra, dec=dec,
            position=position,
            metadata={
                "objectID": int(object_id),
                "flags": _sky_int(row.get("flags")),
                "nimaflags": _sky_int(row.get("nimaflags")),
                "flagsPSF": _sky_int(row.get("flags_psf")),
                "ngood": _sky_int(row.get("ngood")),
                "bandGoodCounts": {
                    band: _sky_int(row.get(f"{band}_ngood"))
                    for band in ("g", "r", "i", "z")
                },
                "gaiaDR3SourceID": _sky_int(row.get("gaia_dr3_id1")),
                "gaiaDR3DistanceArcsec": _sky_float(row.get("gaia_dr3_dist1")),
            },
        )
        if item is not None:
            candidates.append(item)
    return candidates


def _nsc_candidates(
    rows: list[dict[str, Any]], position: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        object_id = str(row.get("id") or "").strip()
        ra = _nsc_float(row.get("ra"))
        dec = _nsc_float(row.get("dec"))
        if not object_id or ra is None or dec is None:
            continue
        item = _candidate(
            catalog="NSCDR2", identifier=object_id, ra=ra, dec=dec,
            position=position,
            metadata={
                "objectID": object_id,
                "ndet": _nsc_int(row.get("ndet")),
                "classStar": _nsc_float(row.get("class_star")),
                "flags": _nsc_int(row.get("flags")),
                "meanMagnitudes": {
                    band: _nsc_float(row.get(column))
                    for band, column in (
                        ("g", "gmag"), ("r", "rmag"),
                        ("i", "imag"), ("z", "zmag"),
                    )
                },
            },
        )
        if item is not None:
            candidates.append(item)
    return candidates


def _merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    ordered = sorted(candidates, key=lambda item: (
        item["separationArcsec"],
        str(item["catalogIDs"].get("skyMapperDR4ObjectID") or ""),
        str(item["catalogIDs"].get("nscDR2ObjectID") or ""),
    ))
    for item in ordered:
        match = next((
            existing for existing in merged
            if _coordinate_separation_arcsec(
                existing["raDeg"], existing["decDeg"], item["raDeg"], item["decDeg"]
            ) <= CATALOG_MERGE_RADIUS_ARCSEC
            and not set(existing["catalogRecords"]).intersection(item["catalogRecords"])
        ), None)
        if match is None:
            merged.append(item)
            continue
        match["catalogIDs"].update({
            key: value for key, value in item["catalogIDs"].items() if value is not None
        })
        match["catalogRecords"].update(item["catalogRecords"])
        if item["separationArcsec"] < match["separationArcsec"]:
            for key in ("raDeg", "decDeg", "separationArcsec", "targetSeparationArcsec", "isTarget"):
                match[key] = item[key]
    for item in merged:
        item["motivatingComponentID"] = None
        item["variabilityConfirmed"] = False
        item["rankingEvidence"] = {
            "residualPositionSeparationArcsec": item["separationArcsec"],
            "targetSeparationArcsec": item["targetSeparationArcsec"],
            "catalogCount": len(item["catalogRecords"]),
        }
    return sorted(merged, key=lambda item: (
        item["separationArcsec"], -len(item["catalogRecords"]),
        str(item["catalogIDs"]),
    ))


def identify_deep_catalog_counterparts(
    *, catalog_summary: dict[str, Any],
    query_skymapper: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
    query_nsc: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    position = validate_catalog_boundary(catalog_summary)
    query_skymapper = query_skymapper or _query_skymapper
    query_nsc = query_nsc or _query_nsc
    rows: dict[str, list[dict[str, Any]]] = {}
    errors = []
    for name, query in (("SkyMapperDR4", query_skymapper), ("NSCDR2", query_nsc)):
        try:
            rows[name] = list(query(position))
        except Exception as exc:
            rows[name] = []
            errors.append({"catalog": name, "error": f"{type(exc).__name__}: {exc}"})

    candidates = _merge_candidates(
        _skymapper_candidates(rows["SkyMapperDR4"], position)
        + _nsc_candidates(rows["NSCDR2"], position)
    )
    for item in candidates:
        item["motivatingComponentID"] = position["componentID"]
    plausible = [item for item in candidates if not item["isTarget"]]
    preferred = None
    if errors:
        classification = "EXTERNAL_DEEP_CATALOG_DATA_UNAVAILABLE"
        next_test = "RETRY_DEEP_CATALOG_COUNTERPART_IDENTIFICATION"
    elif not plausible:
        classification = "NO_DEEP_CATALOG_COUNTERPART"
        next_test = "DEDICATED_HIGH_RESOLUTION_IMAGING"
    elif len(plausible) == 1:
        classification = "DEEP_CATALOG_COUNTERPART_IDENTIFIED"
        preferred = plausible[0]
        next_test = "DEEP_CATALOG_GUIDED_SOURCE_LOCALIZATION"
    else:
        first, second = plausible[:2]
        if (
            first["rankingEvidence"]["catalogCount"] >= 2
            and second["separationArcsec"] - first["separationArcsec"]
            >= MIN_PREFERENCE_MARGIN_ARCSEC
        ):
            classification = "DEEP_CATALOG_COUNTERPART_IDENTIFIED"
            preferred = first
            next_test = "DEEP_CATALOG_GUIDED_SOURCE_LOCALIZATION"
        else:
            classification = "AMBIGUOUS_DEEP_CATALOG_COUNTERPARTS"
            next_test = "HIGH_RESOLUTION_RESIDUAL_SOURCE_LOCALIZATION"

    return {
        "version": METHOD_VERSION,
        "evidenceLineage": "NO_TIC_GAIA_COUNTERPART_DEEP_CATALOG_SEARCH",
        "searchPosition": dict(catalog_summary["searchPosition"]),
        "search": {
            "catalogs": ["SkyMapperDR4", "NSCDR2"],
            "maximumPlausibleRadiusArcsec": MAX_PLAUSIBLE_RADIUS_ARCSEC,
            "targetExclusionRadiusArcsec": TARGET_EXCLUSION_RADIUS_ARCSEC,
            "catalogMergeRadiusArcsec": CATALOG_MERGE_RADIUS_ARCSEC,
            "minimumPreferenceMarginArcsec": MIN_PREFERENCE_MARGIN_ARCSEC,
        },
        "catalogQueries": {
            "skyMapperDR4": {"rows": rows["SkyMapperDR4"]},
            "nscDR2": {"rows": rows["NSCDR2"]},
        },
        "queryProvenance": {
            "SkyMapperDR4": {"service": "SkyMapper public TAP", "table": "dr4.master"},
            "NSCDR2": {"service": "NOIRLab Data Lab", "table": "nsc_dr2.object"},
            "center": {"raDeg": position["raDeg"], "decDeg": position["decDeg"]},
            "responsesPersistedVerbatim": True,
        },
        "queryErrors": errors,
        "catalogCandidates": candidates,
        "plausibleCatalogCandidates": plausible,
        "preferredCandidate": preferred,
        "classification": classification,
        "counterpartIdentified": preferred is not None,
        "variabilityConfirmed": False,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "externalDataState": "BLOCKED_EXTERNAL_DATA" if errors else "AVAILABLE",
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "A deeper-catalog position is only a source hypothesis. It does not attribute "
            "the TESS residual variability without independent source localization."
        ),
    }
