"""Conservative catalog identification for PRF-localized residual components.

Catalog access is coordinator-local: this module never creates worker work units.
The query functions are deliberately narrow boundaries so tests can freeze the
external responses while exercising all ranking and interpretation code.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from .tess_offset_source import (
    CATALOG_MERGE_RADIUS_ARCSEC,
    MAX_PLAUSIBLE_RADIUS_ARCSEC,
    TARGET_EXCLUSION_RADIUS_ARCSEC,
    _coordinate_separation_arcsec,
    _merge_catalog_candidates,
    _query_gaia_region,
    _query_tic_region,
    _skycoord,
)

NEXT_VARIABILITY_TEST = "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"


def _search_position(preparation: dict[str, Any], prf: dict[str, Any]) -> dict[str, Any]:
    target = preparation.get("targetSky") or {}
    ra = target.get("raDeg")
    dec = target.get("decDeg")
    offset = (preparation.get("offset") or {}).get("initialGeometry") or {}
    east = offset.get("eastArcsec")
    north = offset.get("northArcsec")
    if None in (ra, dec, east, north):
        raise RuntimeError("Catalog identification requires persisted target sky and offset geometry.")

    # Exact spherical offset, expressed without an astronomy dependency so
    # frozen-response interpretation also runs on dependency-minimal servers.
    ra0, dec0 = math.radians(float(ra)), math.radians(float(dec))
    distance = math.radians(math.hypot(float(east), float(north)) / 3600.0)
    bearing = math.atan2(float(east), float(north))
    residual_dec = math.asin(math.sin(dec0) * math.cos(distance)
                             + math.cos(dec0) * math.sin(distance) * math.cos(bearing))
    residual_ra = ra0 + math.atan2(
        math.sin(bearing) * math.sin(distance) * math.cos(dec0),
        math.cos(distance) - math.sin(dec0) * math.sin(residual_dec))
    return {
        "componentID": (preparation.get("offset") or {}).get("componentID"),
        "raDeg": math.degrees(residual_ra) % 360.0,
        "decDeg": math.degrees(residual_dec),
        "targetRaDeg": float(ra),
        "targetDecDeg": float(dec),
        "targetSeparationArcsec": math.hypot(float(east), float(north)),
        "eastArcsec": float(east),
        "northArcsec": float(north),
        "supportingSectors": offset.get("supportingSectors") or [],
        "supportingWindows": offset.get("supportingWindows"),
        "prfClassification": prf.get("classification"),
        "provenance": {
            "target": "official PRF preparation targetSky from persisted TIC identity",
            "offset": "best-offset initialGeometry persisted before official SPOC PRF fitting",
            "transform": "ICRS spherical_offsets_by(eastArcsec, northArcsec)",
        },
    }


def identify_catalog_counterparts(
    *, tic_id: int, preparation: dict[str, Any], prf_summary: dict[str, Any],
    query_tic: Callable[[Any, int], dict[str, Any]] | None = None,
    query_gaia: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Query TIC/Gaia and retain positional candidates without claiming variability."""
    if (prf_summary.get("recommendedNextTest") != "CATALOG_COUNTERPART_IDENTIFICATION"
            or prf_summary.get("physicalMechanismResolved") is not False):
        raise RuntimeError("Official PRF evidence did not request catalog counterpart identification.")

    position = _search_position(preparation, prf_summary)
    if query_tic is None or query_gaia is None:
        query_tic = query_tic or _query_tic_region
        query_gaia = query_gaia or _query_gaia_region
        try:
            coordinate = _skycoord(position["raDeg"], position["decDeg"])
        except Exception as error:
            unavailable = {"found": False, "sources": [],
                           "queryError": f"{type(error).__name__}: {error}"}
            coordinate = position
            query_tic = lambda *_args: dict(unavailable)
            query_gaia = lambda *_args: dict(unavailable)
    else:
        coordinate = position
    try:
        tic = query_tic(coordinate, int(tic_id))
    except Exception as error:
        tic = {"found": False, "sources": [],
               "queryError": f"{type(error).__name__}: {error}"}
    try:
        gaia = query_gaia(coordinate)
    except Exception as error:
        gaia = {"found": False, "sources": [],
                "queryError": f"{type(error).__name__}: {error}"}
    errors = []
    for name, response in (("TIC", tic), ("GaiaDR3", gaia)):
        if response.get("queryError"):
            errors.append({"catalog": name, "error": str(response["queryError"])})

    candidates = _merge_catalog_candidates(
        tic_sources=tic.get("sources") or [], gaia_sources=gaia.get("sources") or [],
        target_sky={"raDeg": position["targetRaDeg"], "decDeg": position["targetDecDeg"]},
    )
    raw_sources = list(tic.get("sources") or []) + list(gaia.get("sources") or [])
    target_only = bool(raw_sources) and not candidates and all(
        bool(item.get("isTargetTIC"))
        or _coordinate_separation_arcsec(
            position["targetRaDeg"], position["targetDecDeg"],
            float(item["raDeg"]), float(item["decDeg"]),
        ) <= TARGET_EXCLUSION_RADIUS_ARCSEC
        for item in raw_sources
    )
    plausible = [item for item in candidates
                 if float(item["separationArcsec"]) <= MAX_PLAUSIBLE_RADIUS_ARCSEC]
    for item in candidates:
        item["motivatingComponentID"] = position["componentID"]
        item["motivatingSectors"] = position["supportingSectors"]
        item["isTarget"] = False
        item["variabilityConfirmed"] = False
        item["rankingEvidence"] = {
            "residualPositionSeparationArcsec": item["separationArcsec"],
            "targetSeparationArcsec": item["targetSeparationArcsec"],
            "catalogCount": sum(item.get(name) is not None for name in ("tic", "gaiaDR3")),
        }

    if not candidates and len(errors) == 2:
        classification = "EXTERNAL_CATALOG_DATA_UNAVAILABLE"
        preferred = None
        next_test = "RETRY_CATALOG_COUNTERPART_IDENTIFICATION"
    elif target_only:
        classification = "TARGET_CONSISTENT_ONLY"
        preferred = None
        next_test = "HIGHER_RESOLUTION_RESIDUAL_SOURCE_LOCALIZATION"
    elif not plausible:
        classification = "NO_USABLE_CATALOG_CANDIDATES"
        preferred = None
        next_test = "DEEPER_CATALOG_OR_HIGH_RESOLUTION_IMAGING"
    elif len(plausible) == 1:
        classification = "PLAUSIBLE_NEARBY_CATALOG_COUNTERPART"
        preferred = plausible[0]
        next_test = NEXT_VARIABILITY_TEST
    else:
        first, second = plausible[:2]
        # A preferred positional hypothesis needs more than merely being first:
        # require a cross-catalog association and clear localization margin.
        supported = (first["rankingEvidence"]["catalogCount"] >= 2
                     and float(second["separationArcsec"]) - float(first["separationArcsec"]) >= 3.0)
        classification = ("PLAUSIBLE_NEARBY_CATALOG_COUNTERPARTS"
                          if supported else "AMBIGUOUS_MULTIPLE_CATALOG_COUNTERPARTS")
        preferred = first if supported else None
        next_test = NEXT_VARIABILITY_TEST if supported else "CATALOG_GUIDED_SOURCE_LOCALIZATION"

    return {
        "version": "openstar.tess-catalog-counterpart-identification.v1",
        "searchPosition": position,
        "search": {
            "catalogs": ["TIC", "GaiaDR3"],
            "maximumPlausibleRadiusArcsec": MAX_PLAUSIBLE_RADIUS_ARCSEC,
            "targetExclusionRadiusArcsec": TARGET_EXCLUSION_RADIUS_ARCSEC,
            "catalogMergeRadiusArcsec": CATALOG_MERGE_RADIUS_ARCSEC,
        },
        "catalogQueries": {"tic": tic, "gaiaDR3": gaia},
        "queryProvenance": {
            "TIC": {"service": "MAST Catalogs", "catalog": "TIC",
                    "center": {"raDeg": position["raDeg"], "decDeg": position["decDeg"]}},
            "GaiaDR3": {"service": "VizieR", "catalog": "I/355/gaiadr3",
                        "center": {"raDeg": position["raDeg"], "decDeg": position["decDeg"]}},
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
        "externalDataState": ("BLOCKED_EXTERNAL_DATA"
                              if classification == "EXTERNAL_CATALOG_DATA_UNAVAILABLE"
                              else "AVAILABLE"),
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "Catalog position and brightness establish candidates, not the source of variability. "
            "Independent source-resolved photometry is required before variability attribution."
        ),
    }
