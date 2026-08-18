from __future__ import annotations

import math
import re
import socket
from typing import Any


VSX_CATALOG = "B/vsx/vsx"
GAIA_MAIN_CATALOG = "I/355/gaiadr3"
GAIA_VARIABILITY_CATALOGS = (
    "I/358/vclassre",
    "I/358/veb",
    "I/358/vcep",
    "I/358/vrrlyr",
    "I/358/vrm",
    "I/358/vlpv",
    "I/358/vmsosc",
    "I/358/vst",
)
VSX_RADIUS_ARCSEC = 10.0
GAIA_RADIUS_ARCSEC = 5.0
TRANSIENT_INFRASTRUCTURE = "TRANSIENT_INFRASTRUCTURE"


def classify_query_exception(error: BaseException) -> str | None:
    """Classify transport failures without interpreting exception messages."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is None:
        status = getattr(error, "status", None) or getattr(error, "code", None)
    if isinstance(status, int) and (status in {408, 425, 429} or 500 <= status <= 599):
        return TRANSIENT_INFRASTRUCTURE

    transient_names = {
        "Timeout", "ReadTimeout", "ConnectTimeout", "TimeoutError",
        "ConnectionError", "ConnectError", "NewConnectionError",
        "MaxRetryError", "NameResolutionError", "NetworkError",
    }
    if isinstance(error, (TimeoutError, ConnectionError, socket.gaierror)) or any(
        cls.__name__ in transient_names
        and cls.__module__.split(".", 1)[0] in {"requests", "httpx", "urllib3", "socket", "builtins"}
        for cls in type(error).__mro__
    ):
        return TRANSIENT_INFRASTRUCTURE
    return None


def _query_failure(error: BaseException) -> dict[str, Any]:
    failure = {
        "queryError": f"{type(error).__name__}: {error}",
        "queryErrorType": type(error).__name__,
    }
    classification = classify_query_exception(error)
    if classification:
        failure["queryErrorClassification"] = classification
    return failure


def _python_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        import numpy as np
        if np.ma.is_masked(value):
            return None
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _float(value: Any) -> float | None:
    value = _python_value(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    value = _python_value(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _row_value(row: Any, names: tuple[str, ...], default: Any = None) -> Any:
    columns = set(getattr(row, "colnames", []))
    for name in names:
        if name in columns:
            value = _python_value(row[name])
            if value is not None:
                return value
    return default


def _query_tic(tic_id: int) -> dict[str, Any]:
    from astroquery.mast import Catalogs

    table = None
    errors: list[BaseException] = []
    for catalog_name in ("TIC", "Tic"):
        try:
            table = Catalogs.query_criteria(catalog=catalog_name, ID=int(tic_id))
            if table is not None and len(table) > 0:
                break
        except Exception as error:
            errors.append(error)

    if table is None or len(table) == 0:
        failure = {
            "found": False,
            "queryError": "; ".join(f"{type(e).__name__}: {e}" for e in errors) if errors else "TIC ID not found.",
        }
        if errors:
            failure["queryErrorType"] = type(errors[-1]).__name__
            if any(classify_query_exception(error) for error in errors):
                failure["queryErrorClassification"] = TRANSIENT_INFRASTRUCTURE
        return failure

    selected = None
    for row in table:
        if _int(_row_value(row, ("ID",))) == int(tic_id):
            selected = row
            break
    if selected is None:
        selected = table[0]

    aliases = {
        "TIC": int(tic_id),
        "GAIA_field": _int(_row_value(selected, ("GAIA", "Gaia"))),
        "2MASS": _row_value(selected, ("TWOMASS", "2MASS")),
        "HIP": _int(_row_value(selected, ("HIP",))),
        "TYC": _row_value(selected, ("TYC",)),
        "UCAC": _row_value(selected, ("UCAC",)),
        "ALLWISE": _row_value(selected, ("ALLWISE",)),
        "APASS": _row_value(selected, ("APASS",)),
        "KIC": _int(_row_value(selected, ("KIC",))),
    }
    metadata = {
        "raDeg": _float(_row_value(selected, ("ra", "RA"))),
        "decDeg": _float(_row_value(selected, ("dec", "DEC"))),
        "tmag": _float(_row_value(selected, ("Tmag",))),
        "teffK": _float(_row_value(selected, ("Teff",))),
        "logg": _float(_row_value(selected, ("logg",))),
        "radiusRsun": _float(_row_value(selected, ("rad",))),
        "massMsun": _float(_row_value(selected, ("mass",))),
        "luminosity": _float(_row_value(selected, ("lum",))),
        "distancePc": _float(_row_value(selected, ("d",))),
        "contaminationRatio": _float(_row_value(selected, ("contratio",))),
        "objectType": _row_value(selected, ("objType",)),
    }
    return {"found": True, "aliases": aliases, "metadata": metadata}


def _coordinate(tic: dict[str, Any]):
    from astropy.coordinates import SkyCoord
    metadata = tic.get("metadata") or {}
    ra = _float(metadata.get("raDeg"))
    dec = _float(metadata.get("decDeg"))
    if ra is None or dec is None:
        return None
    return SkyCoord(ra, dec, unit="deg", frame="icrs")


def _query_simbad(tic_id: int) -> dict[str, Any]:
    from astroquery.simbad import Simbad

    result: dict[str, Any] = {
        "found": False,
        "mainID": None,
        "objectType": None,
        "spectralType": None,
        "identifiers": [],
    }
    try:
        table = Simbad.query_object(f"TIC {int(tic_id)}")
    except Exception as error:
        result.update(_query_failure(error))
        return result
    if table is None or len(table) == 0:
        return result

    row = table[0]
    result["found"] = True
    result["mainID"] = _row_value(row, ("main_id", "MAIN_ID"))
    result["objectType"] = _row_value(row, ("otype", "OTYPE", "otype_txt", "OTYPE_TXT"))
    result["spectralType"] = _row_value(row, ("sp_type", "SP_TYPE", "sptype", "SP_TYPE_TXT"))

    try:
        ids = Simbad.query_objectids(f"TIC {int(tic_id)}")
        if ids is not None and len(ids) > 0:
            columns = list(getattr(ids, "colnames", []))
            id_column = next((name for name in ("id", "ID", "identifier", "IDENTIFIER") if name in columns), None)
            if id_column is None and columns:
                id_column = columns[0]
            if id_column is not None:
                result["identifiers"] = sorted({
                    str(_python_value(value)).strip()
                    for value in ids[id_column]
                    if _python_value(value) is not None
                })
    except Exception as error:
        result["identifierQueryError"] = f"{type(error).__name__}: {error}"
    return result


def _separation_arcsec(table: Any, row: Any) -> float | None:
    value = _float(_row_value(row, ("_r",)))
    if value is None:
        return None
    try:
        from astropy import units as u
        unit = getattr(table["_r"], "unit", None)
        if unit is not None:
            return float((value * unit).to_value(u.arcsec))
    except Exception:
        pass
    return value


def _query_vsx(coordinate: Any) -> dict[str, Any]:
    from astropy import units as u
    from astroquery.vizier import Vizier

    if coordinate is None:
        return {"found": False, "queryError": "No TIC coordinate available for VSX query."}
    vizier = Vizier(columns=["**", "+_r"], row_limit=20)
    try:
        result = vizier.query_region(
            coordinate,
            radius=VSX_RADIUS_ARCSEC * u.arcsec,
            catalog=VSX_CATALOG,
        )
    except Exception as error:
        return {"found": False, **_query_failure(error)}
    if len(result) == 0 or len(result[0]) == 0:
        return {"found": False, "matches": []}

    table = result[0]
    matches = []
    for row in table:
        matches.append({
            "name": _row_value(row, ("Name", "name")),
            "type": _row_value(row, ("Type", "type")),
            "periodDays": _float(_row_value(row, ("Period", "period"))),
            "separationArcsec": _separation_arcsec(table, row),
            "maxMag": _float(_row_value(row, ("max", "Max"))),
            "minMag": _float(_row_value(row, ("min", "Min"))),
        })
    matches.sort(key=lambda item: item["separationArcsec"] if item["separationArcsec"] is not None else float("inf"))
    return {"found": bool(matches), "matches": matches, "nearest": matches[0] if matches else None}


def _query_gaia_main(coordinate: Any) -> dict[str, Any]:
    from astropy import units as u
    from astroquery.vizier import Vizier

    if coordinate is None:
        return {"found": False, "queryError": "No TIC coordinate available for Gaia query."}
    vizier = Vizier(columns=["**", "+_r"], row_limit=20)
    try:
        result = vizier.query_region(
            coordinate,
            radius=GAIA_RADIUS_ARCSEC * u.arcsec,
            catalog=GAIA_MAIN_CATALOG,
        )
    except Exception as error:
        return {"found": False, **_query_failure(error)}
    if len(result) == 0 or len(result[0]) == 0:
        return {"found": False, "sources": []}

    table = result[0]
    sources = []
    for row in table:
        source_id = _int(_row_value(row, ("Source", "source_id")))
        if source_id is None:
            continue
        sources.append({
            "sourceID": source_id,
            "separationArcsec": _separation_arcsec(table, row),
            "raDeg": _float(_row_value(row, ("RA_ICRS", "RAJ2000"))),
            "decDeg": _float(_row_value(row, ("DE_ICRS", "DEJ2000"))),
            "gMag": _float(_row_value(row, ("Gmag",))),
            "bpRp": _float(_row_value(row, ("BP-RP", "BP_RP"))),
            "parallaxMas": _float(_row_value(row, ("Plx", "parallax"))),
        })
    sources.sort(key=lambda item: item["separationArcsec"] if item["separationArcsec"] is not None else float("inf"))
    return {"found": bool(sources), "sources": sources, "nearest": sources[0] if sources else None}


def _period_fields(table: Any, row: Any, source: str) -> list[dict[str, Any]]:
    result = []
    period_fields = {"period", "pf", "p1o", "p2o", "prot"}
    frequency_fields = {"freq", "frequency", "fundfreq1", "fundfreq2"}
    for name in getattr(table, "colnames", []):
        normalized = name.lower().replace("_", "")
        value = _float(row[name])
        if value is None or value <= 0:
            continue
        is_period = normalized in period_fields or (
            "period" in normalized
            and "error" not in normalized
            and "percent" not in normalized
        )
        is_frequency = normalized in frequency_fields or (
            "frequency" in normalized and "error" not in normalized
        )
        if is_period:
            period_days = value
            try:
                from astropy import units as u
                unit = getattr(table[name], "unit", None)
                if unit is not None:
                    period_days = float((value * unit).to_value(u.day))
            except Exception:
                pass
            result.append({"periodDays": period_days, "source": source, "field": name})
        elif is_frequency:
            frequency_per_day = value
            try:
                from astropy import units as u
                unit = getattr(table[name], "unit", None)
                if unit is not None:
                    frequency_per_day = float((value * unit).to_value(1 / u.day))
            except Exception:
                pass
            if frequency_per_day > 0:
                result.append({"periodDays": 1.0 / frequency_per_day, "source": source, "field": f"1/{name}"})

    deduplicated = []
    for candidate in result:
        period = candidate["periodDays"]
        if any(
            abs(period - old["periodDays"]) <= max(1e-10, abs(period) * 1e-9)
            for old in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    return deduplicated


def _query_gaia_variability(source_id: int) -> dict[str, Any]:
    from astroquery.vizier import Vizier

    output: dict[str, Any] = {
        "classification": None,
        "tablesFound": [],
        "periodCandidates": [],
    }
    try:
        result = Vizier(columns=["**"], row_limit=20).query_constraints(
            catalog=list(GAIA_VARIABILITY_CATALOGS),
            Source=str(int(source_id)),
        )
    except Exception as error:
        output.update(_query_failure(error))
        return output

    for key in result.keys():
        table = result[key]
        if len(table) == 0:
            continue
        source = str(key)
        output["tablesFound"].append(source)
        row = table[0]
        if "vclassre" in source:
            output["classification"] = {
                "class": _row_value(row, ("Class",)),
                "score": _float(_row_value(row, ("ClassSc",))),
                "classifier": _row_value(row, ("Classifier",)),
                "source": source,
            }
            continue
        output["periodCandidates"].extend(_period_fields(table, row, source))
    return output




def _sector_from_row(table: Any, index: int) -> int | None:
    colnames = set(getattr(table, "colnames", []))

    if "sequence_number" in colnames:
        value = _int(table["sequence_number"][index])
        if value is not None and value > 0:
            return value

    if "mission" in colnames:
        text = str(_python_value(table["mission"][index]) or "")
        match = re.search(r"sector\s*0*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None

def _query_tess_products(tic_id: int) -> dict[str, Any]:
    import lightkurve as lk

    try:
        search = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
    except Exception as error:
        return {"found": False, "products": [], "officialProducts": [], **_query_failure(error)}
    table = getattr(search, "table", None)
    if table is None or len(table) == 0:
        return {"found": False, "products": [], "officialProducts": [], "officialSectors": []}

    products = []
    for index in range(len(table)):
        author = str(_python_value(table["author"][index])).strip() if "author" in table.colnames else None
        exptime = None
        if "exptime" in table.colnames:
            value = table["exptime"][index]
            try:
                from astropy import units as u
                exptime = float(value.to_value(u.s)) if hasattr(value, "to_value") else _float(value)
            except Exception:
                exptime = _float(value)
        products.append({
            "sector": _sector_from_row(table, index),
            "author": author,
            "exptimeSeconds": exptime,
            "mission": str(_python_value(table["mission"][index])) if "mission" in table.colnames else None,
        })

    unique = []
    seen = set()
    for product in products:
        key = (product["sector"], product["author"], round(product["exptimeSeconds"], 3) if product["exptimeSeconds"] is not None else None)
        if key in seen:
            continue
        seen.add(key)
        unique.append(product)
    unique.sort(key=lambda item: (item["sector"] if item["sector"] is not None else 10**9, item["author"] or ""))
    official = [item for item in unique if (item["author"] or "").upper() in {"SPOC", "TESS-SPOC"}]
    return {
        "found": bool(unique),
        "products": unique,
        "officialProducts": official,
        "officialSectors": sorted({item["sector"] for item in official if item["sector"] is not None}),
    }


def collect_identity(tic_id: int) -> dict[str, Any]:
    query_errors: list[str] = []

    try:
        tic = _query_tic(tic_id)
    except Exception as error:
        tic = {"found": False, **_query_failure(error)}
    if tic.get("queryError"):
        query_errors.append(f"TIC: {tic['queryError']}")

    coordinate = None
    try:
        coordinate = _coordinate(tic)
    except Exception as error:
        query_errors.append(f"coordinate: {type(error).__name__}: {error}")

    try:
        simbad = _query_simbad(tic_id)
    except Exception as error:
        simbad = {"found": False, **_query_failure(error)}
    if simbad.get("queryError"):
        query_errors.append(f"SIMBAD: {simbad['queryError']}")

    try:
        vsx = _query_vsx(coordinate)
    except Exception as error:
        vsx = {"found": False, **_query_failure(error)}
    if vsx.get("queryError"):
        query_errors.append(f"VSX: {vsx['queryError']}")

    try:
        gaia = _query_gaia_main(coordinate)
    except Exception as error:
        gaia = {"found": False, **_query_failure(error)}
    if gaia.get("queryError"):
        query_errors.append(f"Gaia: {gaia['queryError']}")

    gaia_variability: dict[str, Any] = {"classification": None, "tablesFound": [], "periodCandidates": []}
    nearest = gaia.get("nearest") if gaia.get("found") else None
    if nearest and nearest.get("sourceID") is not None:
        try:
            gaia_variability = _query_gaia_variability(int(nearest["sourceID"]))
        except Exception as error:
            gaia_variability = {"classification": None, "tablesFound": [], "periodCandidates": [], **_query_failure(error)}
    for error in gaia_variability.get("queryErrors") or []:
        query_errors.append(f"Gaia variability: {error}")
    if gaia_variability.get("queryError"):
        query_errors.append(f"Gaia variability: {gaia_variability['queryError']}")

    try:
        tess = _query_tess_products(tic_id)
    except Exception as error:
        tess = {"found": False, "products": [], "officialProducts": [], **_query_failure(error)}
    if tess.get("queryError"):
        query_errors.append(f"TESS inventory: {tess['queryError']}")

    return {
        "ticID": int(tic_id),
        "tic": tic,
        "simbad": simbad,
        "vsx": vsx,
        "gaiaDR3": gaia,
        "gaiaVariability": gaia_variability,
        "tess": tess,
        "queryErrors": query_errors,
        "identityResolved": bool(tic.get("found")),
    }


def transient_required_catalog_failures(identity: dict[str, Any]) -> list[str]:
    """Return failed required period-catalog paths; optional enrichment is excluded."""
    required = (("TIC", "tic"), ("VSX", "vsx"), ("Gaia", "gaiaDR3"),
                ("Gaia variability", "gaiaVariability"))
    return [name for name, key in required if
            (identity.get(key) or {}).get("queryErrorClassification") == TRANSIENT_INFRASTRUCTURE]
