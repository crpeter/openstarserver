from __future__ import annotations

import math
from typing import Any

from .tess_identity import (
    _float,
    _int,
    _python_value,
    _query_gaia_variability,
    _row_value,
    _separation_arcsec,
)

TIC_QUERY_RADIUS_ARCSEC = 45.0
GAIA_QUERY_RADIUS_ARCSEC = 45.0
SIMBAD_QUERY_RADIUS_ARCSEC = 30.0
VSX_QUERY_RADIUS_ARCSEC = 30.0
CATALOG_MERGE_RADIUS_ARCSEC = 1.5
TARGET_EXCLUSION_RADIUS_ARCSEC = 3.0
MAX_PLAUSIBLE_RADIUS_ARCSEC = 25.0
MIN_SECURE_RADIUS_ARCSEC = 8.0
MAX_SECURE_RADIUS_ARCSEC = 18.0
SECURE_SEPARATION_MARGIN_ARCSEC = 6.0
SECURE_SEPARATION_RATIO = 1.6
VARIABLE_MATCH_RADIUS_ARCSEC = 4.0


def _skycoord(ra_deg: float, dec_deg: float):
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    return SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")


def _coordinate_separation_arcsec(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    return float(
        _skycoord(ra1_deg, dec1_deg)
        .separation(_skycoord(ra2_deg, dec2_deg))
        .arcsec
    )


def _component_position(
    *,
    identity: dict[str, Any],
    multisource_summary: dict[str, Any],
) -> dict[str, Any]:
    tic_metadata = ((identity.get("tic") or {}).get("metadata") or {})
    target_ra = _float(tic_metadata.get("raDeg"))
    target_dec = _float(tic_metadata.get("decDeg"))
    if target_ra is None or target_dec is None:
        raise RuntimeError("v20.13 requires TIC RA/Dec from the completed identity stage.")

    component_id = str(multisource_summary.get("bestOffsetComponentID") or "").strip()
    if not component_id:
        raise RuntimeError("v20.13 requires v20.12 to select a best offset component.")

    components = multisource_summary.get("spatialComponents") or []
    component = next(
        (item for item in components if str(item.get("componentID")) == component_id),
        None,
    )
    if component is None:
        component = next(
            (
                item
                for item in multisource_summary.get("componentSummaries") or []
                if str(item.get("componentID")) == component_id
            ),
            None,
        )
    if component is None:
        raise RuntimeError(f"v20.13 cannot find spatial metadata for {component_id}.")

    east_arcsec = _float(component.get("eastArcsec"))
    north_arcsec = _float(component.get("northArcsec"))
    if east_arcsec is None or north_arcsec is None:
        raise RuntimeError(f"v20.13 requires sky offsets for {component_id}.")

    from astropy import units as u

    target = _skycoord(target_ra, target_dec)
    offset = target.spherical_offsets_by(east_arcsec * u.arcsec, north_arcsec * u.arcsec)
    sky_scatter = _float(component.get("skyScatterArcsec"))
    return {
        "componentID": component_id,
        "componentType": component.get("componentType"),
        "eastArcsec": float(east_arcsec),
        "northArcsec": float(north_arcsec),
        "skyScatterArcsec": sky_scatter,
        "supportingWindows": component.get("supportingWindows"),
        "supportingSectors": component.get("supportingSectors") or [],
        "targetSky": {"raDeg": float(target_ra), "decDeg": float(target_dec)},
        "componentSky": {"raDeg": float(offset.ra.deg), "decDeg": float(offset.dec.deg)},
        "targetSeparationArcsec": float(target.separation(offset).arcsec),
    }


def _query_tic_region(coordinate: Any, target_tic_id: int) -> dict[str, Any]:
    from astroquery.mast import Catalogs

    try:
        table = Catalogs.query_region(
            coordinate,
            radius=TIC_QUERY_RADIUS_ARCSEC / 3600.0,
            catalog="TIC",
        )
    except Exception as error:
        return {"found": False, "sources": [], "queryError": f"{type(error).__name__}: {error}"}

    sources: list[dict[str, Any]] = []
    if table is None:
        return {"found": False, "sources": []}
    center_ra = float(coordinate.ra.deg)
    center_dec = float(coordinate.dec.deg)
    for row in table:
        tic_id = _int(_row_value(row, ("ID", "id")))
        ra = _float(_row_value(row, ("ra", "RA")))
        dec = _float(_row_value(row, ("dec", "DEC")))
        if tic_id is None or ra is None or dec is None:
            continue
        sep = _float(_row_value(row, ("dstArcSec", "dstarcsec")))
        if sep is None:
            sep = _coordinate_separation_arcsec(center_ra, center_dec, ra, dec)
        sources.append(
            {
                "catalog": "TIC",
                "ticID": int(tic_id),
                "isTargetTIC": int(tic_id) == int(target_tic_id),
                "gaiaSourceID": _int(_row_value(row, ("GAIA", "Gaia"))),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "separationArcsec": float(sep),
                "tmag": _float(_row_value(row, ("Tmag",))),
                "teffK": _float(_row_value(row, ("Teff",))),
                "radiusRsun": _float(_row_value(row, ("rad",))),
                "distancePc": _float(_row_value(row, ("d",))),
                "objectType": _row_value(row, ("objType",)),
                "contaminationRatio": _float(_row_value(row, ("contratio",))),
            }
        )
    sources.sort(key=lambda item: float(item["separationArcsec"]))
    return {"found": bool(sources), "sources": sources}


def _query_gaia_region(coordinate: Any) -> dict[str, Any]:
    from astropy import units as u
    from astroquery.vizier import Vizier

    try:
        result = Vizier(columns=["*", "+_r"], row_limit=100).query_region(
            coordinate,
            radius=GAIA_QUERY_RADIUS_ARCSEC * u.arcsec,
            catalog="I/355/gaiadr3",
        )
    except Exception as error:
        return {"found": False, "sources": [], "queryError": f"{type(error).__name__}: {error}"}
    if len(result) == 0 or len(result[0]) == 0:
        return {"found": False, "sources": []}

    table = result[0]
    sources: list[dict[str, Any]] = []
    for row in table:
        source_id = _int(_row_value(row, ("Source", "source_id")))
        ra = _float(_row_value(row, ("RA_ICRS", "RAJ2000", "ra")))
        dec = _float(_row_value(row, ("DE_ICRS", "DEJ2000", "dec")))
        if source_id is None or ra is None or dec is None:
            continue
        sep = _separation_arcsec(table, row)
        if sep is None:
            sep = _coordinate_separation_arcsec(
                float(coordinate.ra.deg), float(coordinate.dec.deg), ra, dec
            )
        sources.append(
            {
                "catalog": "GaiaDR3",
                "gaiaSourceID": int(source_id),
                "designation": _row_value(row, ("DR3Name", "designation")),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "separationArcsec": float(sep),
                "gMag": _float(_row_value(row, ("Gmag", "phot_g_mean_mag"))),
                "bpMag": _float(_row_value(row, ("BPmag", "phot_bp_mean_mag"))),
                "rpMag": _float(_row_value(row, ("RPmag", "phot_rp_mean_mag"))),
                "bpRp": _float(_row_value(row, ("BP-RP", "bp_rp"))),
                "parallaxMas": _float(_row_value(row, ("Plx", "parallax"))),
                "ruwe": _float(_row_value(row, ("RUWE", "ruwe"))),
            }
        )
    sources.sort(key=lambda item: float(item["separationArcsec"]))
    return {"found": bool(sources), "sources": sources}


def _query_vsx_region(coordinate: Any) -> dict[str, Any]:
    from astropy import units as u
    from astroquery.vizier import Vizier

    try:
        result = Vizier(columns=["*", "+_r"], row_limit=100).query_region(
            coordinate,
            radius=VSX_QUERY_RADIUS_ARCSEC * u.arcsec,
            catalog="B/vsx/vsx",
        )
    except Exception as error:
        return {"found": False, "matches": [], "queryError": f"{type(error).__name__}: {error}"}
    if len(result) == 0 or len(result[0]) == 0:
        return {"found": False, "matches": []}

    table = result[0]
    matches: list[dict[str, Any]] = []
    for row in table:
        ra = _float(_row_value(row, ("RAJ2000", "_RAJ2000")))
        dec = _float(_row_value(row, ("DEJ2000", "_DEJ2000")))
        sep = _separation_arcsec(table, row)
        if ra is None or dec is None:
            continue
        if sep is None:
            sep = _coordinate_separation_arcsec(
                float(coordinate.ra.deg), float(coordinate.dec.deg), ra, dec
            )
        matches.append(
            {
                "name": _row_value(row, ("Name", "name")),
                "type": _row_value(row, ("Type", "type")),
                "periodDays": _float(_row_value(row, ("Period", "period"))),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "separationArcsec": float(sep),
                "maxMag": _float(_row_value(row, ("max", "Max"))),
                "minMag": _float(_row_value(row, ("min", "Min"))),
            }
        )
    matches.sort(key=lambda item: float(item["separationArcsec"]))
    return {"found": bool(matches), "matches": matches}


def _query_simbad_region(coordinate: Any) -> dict[str, Any]:
    from astropy import units as u
    from astroquery.simbad import Simbad

    simbad = Simbad()
    simbad.ROW_LIMIT = 100
    for field in ("otype", "sp_type"):
        try:
            simbad.add_votable_fields(field)
        except Exception:
            pass
    try:
        table = simbad.query_region(coordinate, radius=SIMBAD_QUERY_RADIUS_ARCSEC * u.arcsec)
    except Exception as error:
        return {"found": False, "matches": [], "queryError": f"{type(error).__name__}: {error}"}
    if table is None or len(table) == 0:
        return {"found": False, "matches": []}

    matches: list[dict[str, Any]] = []
    for row in table:
        ra = _float(_row_value(row, ("ra", "RA")))
        dec = _float(_row_value(row, ("dec", "DEC")))
        if ra is None or dec is None:
            continue
        matches.append(
            {
                "mainID": _row_value(row, ("main_id", "MAIN_ID")),
                "objectType": _row_value(row, ("otype", "OTYPE", "otype_txt", "OTYPE_TXT")),
                "spectralType": _row_value(row, ("sp_type", "SP_TYPE")),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "separationArcsec": _coordinate_separation_arcsec(
                    float(coordinate.ra.deg), float(coordinate.dec.deg), ra, dec
                ),
            }
        )
    matches.sort(key=lambda item: float(item["separationArcsec"]))
    return {"found": bool(matches), "matches": matches}


def _new_group(source: dict[str, Any], target_sky: dict[str, float]) -> dict[str, Any]:
    return {
        "raDeg": float(source["raDeg"]),
        "decDeg": float(source["decDeg"]),
        "separationArcsec": float(source["separationArcsec"]),
        "targetSeparationArcsec": _coordinate_separation_arcsec(
            float(target_sky["raDeg"]),
            float(target_sky["decDeg"]),
            float(source["raDeg"]),
            float(source["decDeg"]),
        ),
        "tic": None,
        "gaiaDR3": None,
    }


def _merge_catalog_candidates(
    *,
    tic_sources: list[dict[str, Any]],
    gaia_sources: list[dict[str, Any]],
    target_sky: dict[str, float],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for source in gaia_sources:
        group = _new_group(source, target_sky)
        group["gaiaDR3"] = source
        groups.append(group)

    for source in tic_sources:
        matching = None
        gaia_id = _int(source.get("gaiaSourceID"))
        if gaia_id is not None:
            matching = next(
                (
                    group
                    for group in groups
                    if _int(((group.get("gaiaDR3") or {}).get("gaiaSourceID"))) == gaia_id
                ),
                None,
            )
        if matching is None:
            nearest = None
            nearest_sep = float("inf")
            for group in groups:
                sep = _coordinate_separation_arcsec(
                    float(group["raDeg"]),
                    float(group["decDeg"]),
                    float(source["raDeg"]),
                    float(source["decDeg"]),
                )
                if sep < nearest_sep:
                    nearest_sep = sep
                    nearest = group
            if nearest is not None and nearest_sep <= CATALOG_MERGE_RADIUS_ARCSEC:
                matching = nearest
        if matching is None:
            matching = _new_group(source, target_sky)
            groups.append(matching)
        matching["tic"] = source
        if float(source["separationArcsec"]) < float(matching["separationArcsec"]):
            matching["raDeg"] = float(source["raDeg"])
            matching["decDeg"] = float(source["decDeg"])
            matching["separationArcsec"] = float(source["separationArcsec"])
            matching["targetSeparationArcsec"] = _coordinate_separation_arcsec(
                float(target_sky["raDeg"]),
                float(target_sky["decDeg"]),
                float(source["raDeg"]),
                float(source["decDeg"]),
            )

    groups = [
        group
        for group in groups
        if float(group.get("targetSeparationArcsec") or 0.0) > TARGET_EXCLUSION_RADIUS_ARCSEC
        and not bool(((group.get("tic") or {}).get("isTargetTIC")))
    ]
    groups.sort(key=lambda item: float(item["separationArcsec"]))
    for index, group in enumerate(groups, start=1):
        group["candidateRank"] = index
        group["catalogIDs"] = {
            "ticID": (group.get("tic") or {}).get("ticID"),
            "gaiaDR3SourceID": (group.get("gaiaDR3") or {}).get("gaiaSourceID"),
        }
    return groups


def _secure_radius(component: dict[str, Any]) -> float:
    scatter = _float(component.get("skyScatterArcsec"))
    if scatter is None:
        return MIN_SECURE_RADIUS_ARCSEC
    return max(
        MIN_SECURE_RADIUS_ARCSEC,
        min(MAX_SECURE_RADIUS_ARCSEC, float(scatter) + 6.0),
    )


def _select_candidate(
    candidates: list[dict[str, Any]],
    *,
    secure_radius_arcsec: float,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    plausible = [
        item
        for item in candidates
        if float(item.get("separationArcsec") or float("inf")) <= MAX_PLAUSIBLE_RADIUS_ARCSEC
    ]
    if not plausible:
        return "NO_SECURE_CATALOG_COUNTERPART", None, []

    best = plausible[0]
    best_sep = float(best["separationArcsec"])
    if best_sep > secure_radius_arcsec:
        return "MULTIPLE_PLAUSIBLE_CATALOG_COUNTERPARTS", None, plausible

    if len(plausible) == 1:
        return "CATALOG_COUNTERPART_IDENTIFIED", best, plausible

    second_sep = float(plausible[1]["separationArcsec"])
    strongly_separated = (
        best_sep <= 3.0
        or second_sep - best_sep >= SECURE_SEPARATION_MARGIN_ARCSEC
        or second_sep >= max(SECURE_SEPARATION_RATIO * best_sep, best_sep + 2.0)
    )
    if strongly_separated:
        return "CATALOG_COUNTERPART_IDENTIFIED", best, plausible
    return "MULTIPLE_PLAUSIBLE_CATALOG_COUNTERPARTS", None, plausible


def _nearest_match(
    matches: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
    radius_arcsec: float,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in matches:
        ra = _float(item.get("raDeg"))
        dec = _float(item.get("decDeg"))
        if ra is None or dec is None:
            continue
        sep = _coordinate_separation_arcsec(
            float(candidate["raDeg"]), float(candidate["decDeg"]), ra, dec
        )
        if sep <= radius_arcsec:
            ranked.append((sep, item))
    if not ranked:
        return None
    ranked.sort(key=lambda pair: pair[0])
    result = dict(ranked[0][1])
    result["candidateSeparationArcsec"] = float(ranked[0][0])
    return result


def _simbad_variable(match: dict[str, Any] | None) -> bool:
    if not match:
        return False
    text = " ".join(
        str(match.get(key) or "")
        for key in ("objectType", "mainID")
    ).lower()
    return any(token in text for token in ("v*", "variable", "pulsating", "eclipsing", "rotv"))


def identify_offset_residual_source(
    *,
    tic_id: int,
    identity: dict[str, Any],
    multisource_summary: dict[str, Any],
) -> dict[str, Any]:
    if multisource_summary.get("recommendedNextTest") not in {
        "IDENTIFY_OFFSET_RESIDUAL_VARIABLE_SOURCE",
        "NEIGHBOR_SOURCE_IDENTIFICATION_AND_CATALOG_CROSSMATCH",
    }:
        raise RuntimeError(
            "v20.13 requires v20.12 to recommend offset/neighbor source identification."
        )

    component = _component_position(identity=identity, multisource_summary=multisource_summary)
    component_sky = component["componentSky"]
    target_sky = component["targetSky"]
    coordinate = _skycoord(component_sky["raDeg"], component_sky["decDeg"])

    query_errors: list[str] = []
    tic = _query_tic_region(coordinate, int(tic_id))
    gaia = _query_gaia_region(coordinate)
    simbad = _query_simbad_region(coordinate)
    vsx = _query_vsx_region(coordinate)
    for name, result in (("TIC", tic), ("GaiaDR3", gaia), ("SIMBAD", simbad), ("VSX", vsx)):
        if result.get("queryError"):
            query_errors.append(f"{name}: {result['queryError']}")

    candidates = _merge_catalog_candidates(
        tic_sources=tic.get("sources") or [],
        gaia_sources=gaia.get("sources") or [],
        target_sky=target_sky,
    )
    secure_radius = _secure_radius(component)
    classification, best, plausible = _select_candidate(
        candidates,
        secure_radius_arcsec=secure_radius,
    )

    nearest_vsx = _nearest_match(vsx.get("matches") or [], best, VARIABLE_MATCH_RADIUS_ARCSEC)
    nearest_simbad = _nearest_match(simbad.get("matches") or [], best, VARIABLE_MATCH_RADIUS_ARCSEC)
    gaia_variability: dict[str, Any] = {
        "classification": None,
        "tablesFound": [],
        "periodCandidates": [],
    }
    gaia_id = _int(((best or {}).get("gaiaDR3") or {}).get("gaiaSourceID"))
    if gaia_id is not None:
        try:
            gaia_variability = _query_gaia_variability(int(gaia_id))
        except Exception as error:
            gaia_variability = {
                "classification": None,
                "tablesFound": [],
                "periodCandidates": [],
                "queryError": f"{type(error).__name__}: {error}",
            }
            query_errors.append(f"Gaia variability: {gaia_variability['queryError']}")

    known_variable = bool(
        best is not None
        and (
            nearest_vsx is not None
            or (gaia_variability.get("classification") is not None)
            or bool(gaia_variability.get("tablesFound"))
            or _simbad_variable(nearest_simbad)
        )
    )
    if classification == "CATALOG_COUNTERPART_IDENTIFIED" and known_variable:
        classification = "KNOWN_VARIABLE_CATALOG_COUNTERPART_IDENTIFIED"

    if classification == "KNOWN_VARIABLE_CATALOG_COUNTERPART_IDENTIFIED":
        next_test = "OFFSET_SOURCE_VARIABILITY_MATCH_TEST"
    elif classification == "CATALOG_COUNTERPART_IDENTIFIED":
        next_test = "OFFSET_SOURCE_VARIABILITY_VALIDATION"
    elif classification == "MULTIPLE_PLAUSIBLE_CATALOG_COUNTERPARTS":
        next_test = "CATALOG_GUIDED_PIXEL_SOURCE_FIT"
    elif query_errors and not candidates:
        classification = "CATALOG_QUERY_INCOMPLETE"
        next_test = "RETRY_OFFSET_SOURCE_CATALOG_IDENTIFICATION"
    else:
        next_test = "DEEP_CATALOG_IMAGE_REVIEW"

    if best is not None:
        best = dict(best)
        best["vsxMatch"] = nearest_vsx
        best["simbadMatch"] = nearest_simbad
        best["gaiaVariability"] = gaia_variability
        best["knownVariableCatalogEvidence"] = known_variable

    return {
        "version": "openstar.tess-offset-residual-source-identification.v1",
        "component": component,
        "search": {
            "ticRadiusArcsec": TIC_QUERY_RADIUS_ARCSEC,
            "gaiaRadiusArcsec": GAIA_QUERY_RADIUS_ARCSEC,
            "simbadRadiusArcsec": SIMBAD_QUERY_RADIUS_ARCSEC,
            "vsxRadiusArcsec": VSX_QUERY_RADIUS_ARCSEC,
            "secureAssociationRadiusArcsec": secure_radius,
            "maximumPlausibleRadiusArcsec": MAX_PLAUSIBLE_RADIUS_ARCSEC,
        },
        "catalogs": {
            "tic": tic,
            "gaiaDR3": gaia,
            "simbad": simbad,
            "vsx": vsx,
        },
        "catalogCandidates": candidates,
        "plausibleCatalogCandidates": plausible,
        "bestCandidate": best,
        "classification": classification,
        "offsetSourceIdentificationResolved": classification in {
            "KNOWN_VARIABLE_CATALOG_COUNTERPART_IDENTIFIED",
            "CATALOG_COUNTERPART_IDENTIFIED",
        },
        "knownVariableCatalogEvidence": known_variable,
        "queryErrors": query_errors,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "A positional catalog counterpart is not automatically proven to be the source of "
            "the TESS residual variability. v20.13 identifies plausible sky counterparts only; "
            "photometric variability validation remains a separate test. This result does not "
            "alter v20.6's target association for the established 13.72-day family."
        ),
    }
