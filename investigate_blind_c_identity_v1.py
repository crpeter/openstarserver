#!/usr/bin/env python3

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import lightkurve as lk

# ============================================================
# Blind C identity / catalog reconnaissance
#
# This does NOT modify OpenStar data or coordinator state.
# It performs no new period search. Its job is to resolve the
# exact preregistered target and collect the identifiers/catalog
# information needed for a literature-first astrophysical follow-up.
# ============================================================

LOCK_PATH = Path(
    "data/projects/openstar.tess-blind-validation-set-v1.lock.json"
)
OUTPUT_DIR = Path("blind_c_identity_v1")

BLIND_NAME = "Blind C"
TIC_ID = 41460085

# Frozen from the completed distributed blind run.
OPENSTAR_FREQUENCY = 0.13380100
OPENSTAR_PERIOD_DAYS = 7.47378546
OPENSTAR_POWER = 0.40587783

ASTROPY_FREQUENCY = 0.13425896
ASTROPY_PERIOD_DAYS = 7.44829237
ASTROPY_POWER = 0.41856196

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


def python_value(value):
    if value is None or np.ma.is_masked(value):
        return None

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, (np.integer, np.floating)):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def float_or_none(value):
    value = python_value(value)
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def int_or_none(value):
    value = python_value(value)
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def row_value(row, names, default=None):
    colnames = set(getattr(row, "colnames", []))

    for name in names:
        if name not in colnames:
            continue

        value = python_value(row[name])
        if value is not None:
            return value

    return default


def format_value(value, digits=6):
    if value is None:
        return "[none]"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_locked_target():
    if not LOCK_PATH.exists():
        raise RuntimeError(
            f"Missing preregistered blind lock: {LOCK_PATH}\n"
            "Run this from the openstarserver project root."
        )

    with LOCK_PATH.open("r", encoding="utf-8") as file:
        document = json.load(file)

    matches = [
        target
        for target in document.get("targets", [])
        if target.get("blindName") == BLIND_NAME
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {BLIND_NAME} entry in the lock; "
            f"found {len(matches)}."
        )

    target = matches[0]

    if int(target.get("ticID")) != TIC_ID:
        raise RuntimeError(
            "Blind C TIC mismatch. Refusing to query a different target: "
            f"lock has TIC {target.get('ticID')}, expected TIC {TIC_ID}."
        )

    return document, target


def target_coordinate(target):
    return SkyCoord(
        float(target["ra"]),
        float(target["dec"]),
        unit="deg",
        frame="icrs",
    )


def query_tic(target):
    table = Catalogs.query_region(
        target_coordinate(target),
        radius=5.0 * u.arcsec,
        catalog="TIC",
    )

    selected = None
    for row in table:
        if int_or_none(row_value(row, ("ID",))) == TIC_ID:
            selected = row
            break

    if selected is None:
        return {
            "found": False,
            "error": "Exact TIC ID not found inside 5 arcsec locked-coordinate query",
        }

    # Keep identifiers plus a small set of useful stellar/aperture metadata.
    aliases = {
        "TIC": TIC_ID,
        "GAIA_field": int_or_none(row_value(selected, ("GAIA", "Gaia"))),
        "2MASS": row_value(selected, ("TWOMASS", "2MASS")),
        "HIP": int_or_none(row_value(selected, ("HIP",))),
        "TYC": row_value(selected, ("TYC",)),
        "UCAC": row_value(selected, ("UCAC",)),
        "ALLWISE": row_value(selected, ("ALLWISE",)),
        "APASS": row_value(selected, ("APASS",)),
        "KIC": int_or_none(row_value(selected, ("KIC",))),
    }

    metadata = {
        "raDeg": float_or_none(row_value(selected, ("ra", "RA"))),
        "decDeg": float_or_none(row_value(selected, ("dec", "DEC"))),
        "tmag": float_or_none(row_value(selected, ("Tmag",))),
        "teffK": float_or_none(row_value(selected, ("Teff",))),
        "logg": float_or_none(row_value(selected, ("logg",))),
        "radiusRsun": float_or_none(row_value(selected, ("rad",))),
        "massMsun": float_or_none(row_value(selected, ("mass",))),
        "luminosity": float_or_none(row_value(selected, ("lum",))),
        "distancePc": float_or_none(row_value(selected, ("d",))),
        "contaminationRatio": float_or_none(
            row_value(selected, ("contratio", "contratio"))
        ),
        "objectType": row_value(selected, ("objType",)),
    }

    return {
        "found": True,
        "aliases": aliases,
        "metadata": metadata,
    }


def query_simbad():
    result = {
        "found": False,
        "mainID": None,
        "objectType": None,
        "spectralType": None,
        "identifiers": [],
    }

    try:
        table = Simbad.query_object(f"TIC {TIC_ID}")
    except Exception as error:
        result["queryError"] = f"{type(error).__name__}: {error}"
        return result

    if table is None or len(table) == 0:
        return result

    row = table[0]
    result["found"] = True
    result["mainID"] = row_value(row, ("main_id", "MAIN_ID"))
    result["objectType"] = row_value(
        row,
        ("otype", "OTYPE", "otype_txt", "OTYPE_TXT"),
    )
    result["spectralType"] = row_value(
        row,
        ("sp_type", "SP_TYPE", "sptype", "SP_TYPE_TXT"),
    )

    try:
        ids = Simbad.query_objectids(f"TIC {TIC_ID}")
        if ids is not None and len(ids) > 0:
            columns = list(getattr(ids, "colnames", []))
            id_column = None
            for candidate in ("id", "ID", "identifier", "IDENTIFIER"):
                if candidate in columns:
                    id_column = candidate
                    break
            if id_column is None and columns:
                id_column = columns[0]

            if id_column is not None:
                values = []
                for value in ids[id_column]:
                    converted = python_value(value)
                    if converted is not None:
                        values.append(str(converted).strip())
                result["identifiers"] = sorted(set(values))
    except Exception as error:
        result["identifierQueryError"] = f"{type(error).__name__}: {error}"

    return result


def separation_arcsec(table, row):
    value = float_or_none(row_value(row, ("_r",)))
    if value is None:
        return None

    try:
        unit = getattr(table["_r"], "unit", None)
        if unit is not None:
            return float((value * unit).to_value(u.arcsec))
    except Exception:
        pass

    return value


def query_vsx(target):
    vizier = Vizier(columns=["**", "+_r"], row_limit=20)

    try:
        result = vizier.query_region(
            target_coordinate(target),
            radius=VSX_RADIUS_ARCSEC * u.arcsec,
            catalog=VSX_CATALOG,
        )
    except Exception as error:
        return {
            "found": False,
            "queryError": f"{type(error).__name__}: {error}",
        }

    if len(result) == 0 or len(result[0]) == 0:
        return {"found": False}

    table = result[0]
    matches = []

    for row in table:
        matches.append(
            {
                "name": row_value(row, ("Name", "name")),
                "type": row_value(row, ("Type", "type")),
                "periodDays": float_or_none(
                    row_value(row, ("Period", "period"))
                ),
                "separationArcsec": separation_arcsec(table, row),
                "maxMag": float_or_none(row_value(row, ("max", "Max"))),
                "minMag": float_or_none(row_value(row, ("min", "Min"))),
            }
        )

    matches.sort(
        key=lambda item: (
            item["separationArcsec"]
            if item["separationArcsec"] is not None
            else float("inf")
        )
    )

    return {
        "found": bool(matches),
        "matches": matches,
        "nearest": matches[0] if matches else None,
    }


def query_gaia_main(target):
    vizier = Vizier(columns=["**", "+_r"], row_limit=20)

    try:
        result = vizier.query_region(
            target_coordinate(target),
            radius=GAIA_RADIUS_ARCSEC * u.arcsec,
            catalog=GAIA_MAIN_CATALOG,
        )
    except Exception as error:
        return {
            "found": False,
            "queryError": f"{type(error).__name__}: {error}",
        }

    if len(result) == 0 or len(result[0]) == 0:
        return {"found": False}

    table = result[0]
    sources = []

    for row in table:
        source_id = int_or_none(row_value(row, ("Source", "source_id")))
        if source_id is None:
            continue

        sources.append(
            {
                "sourceID": source_id,
                "separationArcsec": separation_arcsec(table, row),
                "raDeg": float_or_none(row_value(row, ("RA_ICRS", "RAdeg", "ra"))),
                "decDeg": float_or_none(row_value(row, ("DE_ICRS", "DEdeg", "dec"))),
                "gMag": float_or_none(row_value(row, ("Gmag",))),
                "bpMag": float_or_none(row_value(row, ("BPmag",))),
                "rpMag": float_or_none(row_value(row, ("RPmag",))),
                "bpRp": float_or_none(row_value(row, ("BP-RP", "BP_RP"))),
                "parallaxMas": float_or_none(row_value(row, ("Plx", "parallax"))),
                "ruwe": float_or_none(row_value(row, ("RUWE", "ruwe"))),
            }
        )

    sources.sort(
        key=lambda item: (
            item["separationArcsec"]
            if item["separationArcsec"] is not None
            else float("inf")
        )
    )

    return {
        "found": bool(sources),
        "nearest": sources[0] if sources else None,
        "sourcesWithin5Arcsec": sources,
    }


def extract_period_candidates(table, source_label):
    if len(table) == 0:
        return []

    row = table[0]
    candidates = []

    period_field_names = {"period", "pf", "p1o", "p2o", "prot"}
    frequency_field_names = {"freq", "frequency", "fundfreq1", "fundfreq2"}

    for column in table.colnames:
        normalized = column.lower().replace("_", "")
        value = float_or_none(row[column])

        if value is None or value <= 0:
            continue

        is_period = normalized in period_field_names or (
            "period" in normalized
            and "error" not in normalized
            and "percent" not in normalized
        )
        is_frequency = normalized in frequency_field_names or (
            "frequency" in normalized and "error" not in normalized
        )

        if is_period:
            period_days = value
            try:
                unit = getattr(table[column], "unit", None)
                if unit is not None:
                    period_days = float((value * unit).to_value(u.day))
            except Exception:
                pass

            candidates.append(
                {
                    "source": source_label,
                    "field": column,
                    "periodDays": period_days,
                }
            )

        elif is_frequency:
            frequency_per_day = value
            try:
                unit = getattr(table[column], "unit", None)
                if unit is not None:
                    frequency_per_day = float((value * unit).to_value(1 / u.day))
            except Exception:
                pass

            if frequency_per_day > 0:
                candidates.append(
                    {
                        "source": source_label,
                        "field": f"1/{column}",
                        "periodDays": 1.0 / frequency_per_day,
                    }
                )

    deduplicated = []
    for candidate in candidates:
        period = candidate["periodDays"]
        if any(
            abs(period - old["periodDays"])
            <= max(1e-10, abs(period) * 1e-9)
            for old in deduplicated
        ):
            continue
        deduplicated.append(candidate)

    return deduplicated


def query_gaia_variability(source_id):
    if source_id is None:
        return {
            "classification": None,
            "tablesFound": [],
            "periodCandidates": [],
        }

    vizier = Vizier(columns=["**"], row_limit=20)

    try:
        result = vizier.query_constraints(
            catalog=list(GAIA_VARIABILITY_CATALOGS),
            Source=str(int(source_id)),
        )
    except Exception as error:
        return {
            "queryError": f"{type(error).__name__}: {error}",
            "classification": None,
            "tablesFound": [],
            "periodCandidates": [],
        }

    classification = None
    tables_found = []
    periods = []

    for key in result.keys():
        table = result[key]
        if len(table) == 0:
            continue

        tables_found.append(str(key))

        if "vclassre" in str(key):
            row = table[0]
            classification = {
                "class": row_value(row, ("Class",)),
                "score": float_or_none(row_value(row, ("ClassSc",))),
                "classifier": row_value(row, ("Classifier",)),
            }
            continue

        periods.extend(extract_period_candidates(table, str(key)))

    return {
        "classification": classification,
        "tablesFound": tables_found,
        "periodCandidates": periods,
    }


def sector_from_row(table, index):
    if "sequence_number" in table.colnames:
        value = int_or_none(table["sequence_number"][index])
        if value is not None and value > 0:
            return value

    if "mission" in table.colnames:
        text = str(python_value(table["mission"][index]) or "")
        match = re.search(r"sector\s*0*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def exposure_seconds(value):
    if value is None or np.ma.is_masked(value):
        return None

    try:
        if hasattr(value, "to_value"):
            return float(value.to_value(u.s))
    except Exception:
        pass

    try:
        if hasattr(value, "value"):
            return float(value.value)
    except Exception:
        pass

    return float_or_none(value)


def query_tess_products():
    search = lk.search_lightcurve(
        f"TIC {TIC_ID}",
        mission="TESS",
    )

    table = getattr(search, "table", None)
    if table is None or len(table) == 0:
        return {
            "found": False,
            "products": [],
            "officialProducts": [],
        }

    products = []

    for index in range(len(table)):
        author = None
        if "author" in table.colnames:
            author = python_value(table["author"][index])
        if author is not None:
            author = str(author).strip()

        exptime = None
        if "exptime" in table.colnames:
            exptime = exposure_seconds(table["exptime"][index])

        products.append(
            {
                "sector": sector_from_row(table, index),
                "author": author,
                "exptimeSeconds": exptime,
                "mission": (
                    str(python_value(table["mission"][index]))
                    if "mission" in table.colnames
                    else None
                ),
            }
        )

    # De-duplicate repeated search rows at the level relevant to this inventory.
    unique = []
    seen = set()
    for product in products:
        key = (
            product["sector"],
            product["author"],
            round(product["exptimeSeconds"], 3)
            if product["exptimeSeconds"] is not None
            else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(product)

    unique.sort(
        key=lambda item: (
            item["sector"] if item["sector"] is not None else 10**9,
            item["author"] or "",
            item["exptimeSeconds"] if item["exptimeSeconds"] is not None else 10**9,
        )
    )

    official_authors = {"SPOC", "TESS-SPOC"}
    official = [
        item for item in unique
        if (item["author"] or "").upper() in official_authors
    ]

    return {
        "found": bool(unique),
        "products": unique,
        "officialProducts": official,
        "officialSectors": sorted(
            {
                item["sector"]
                for item in official
                if item["sector"] is not None
            }
        ),
    }


def write_tess_csv(products):
    path = OUTPUT_DIR / "tess_lightcurve_products.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("sector", "author", "exptimeSeconds", "mission"),
        )
        writer.writeheader()
        writer.writerows(products)
    return path


def print_aliases(tic, simbad, gaia):
    print()
    print("🔎 Resolved identity")

    aliases = tic.get("aliases", {}) if tic.get("found") else {}
    print(f"   TIC: {TIC_ID}")
    print(f"   2MASS: {format_value(aliases.get('2MASS'))}")
    print(f"   HIP: {format_value(aliases.get('HIP'), 0)}")
    print(f"   TYC: {format_value(aliases.get('TYC'))}")
    print(f"   TIC GAIA field: {format_value(aliases.get('GAIA_field'), 0)}")

    if simbad.get("found"):
        print(f"   SIMBAD main ID: {format_value(simbad.get('mainID'))}")
        print(f"   SIMBAD object type: {format_value(simbad.get('objectType'))}")
        print(f"   SIMBAD spectral type: {format_value(simbad.get('spectralType'))}")
    else:
        print("   SIMBAD TIC lookup: [none]")

    nearest = gaia.get("nearest") if gaia.get("found") else None
    if nearest:
        print(f"   Gaia DR3 source: {nearest['sourceID']}")
        print(
            "   Gaia separation from locked coordinate: "
            f"{format_value(nearest.get('separationArcsec'), 4)} arcsec"
        )
    else:
        print("   Gaia DR3 source: [none]")

    ids = simbad.get("identifiers") or []
    if ids:
        print("   SIMBAD identifiers:")
        for identifier in ids:
            print(f"      {identifier}")


def main():
    print("⭐ OpenStar Blind-C astrophysical identity reconnaissance")
    print("No new period search is performed in this script.")
    print()

    lock_document, target = load_locked_target()

    print("🔒 Frozen blind target")
    print(f"   project: {lock_document.get('projectID')}")
    print(f"   blind name: {BLIND_NAME}")
    print(f"   TIC: {TIC_ID}")
    print(f"   locked RA: {float(target['ra']):.8f} deg")
    print(f"   locked Dec: {float(target['dec']):.8f} deg")
    print(f"   locked sector: {target.get('sector')}")
    print(f"   locked author: {target.get('author')}")
    print(f"   locked cadence: {float(target.get('cadenceSeconds')):.0f}s")
    print(f"   Tmag: {float(target.get('tmag')):.2f}")

    print()
    print("⭐ Frozen Blind-C result")
    print(
        f"   OpenStar: f={OPENSTAR_FREQUENCY:.8f} c/d  "
        f"P={OPENSTAR_PERIOD_DAYS:.8f} d  power={OPENSTAR_POWER:.8f}"
    )
    print(
        f"   Astropy : f={ASTROPY_FREQUENCY:.8f} c/d  "
        f"P={ASTROPY_PERIOD_DAYS:.8f} d  power={ASTROPY_POWER:.8f}"
    )
    print(
        "   period difference: "
        f"{abs(OPENSTAR_PERIOD_DAYS - ASTROPY_PERIOD_DAYS):.8f} d"
    )

    print()
    print("🌐 Querying TIC metadata...")
    tic = query_tic(target)

    print("🌐 Querying SIMBAD identity and aliases...")
    simbad = query_simbad()

    print("🌐 Querying AAVSO VSX around locked coordinate...")
    vsx = query_vsx(target)

    print("🌐 Resolving Gaia DR3 from locked coordinate via VizieR...")
    gaia = query_gaia_main(target)

    gaia_variability = {
        "classification": None,
        "tablesFound": [],
        "periodCandidates": [],
    }
    if gaia.get("found") and gaia.get("nearest"):
        print("🌐 Querying Gaia DR3 variability tables...")
        gaia_variability = query_gaia_variability(
            gaia["nearest"]["sourceID"]
        )

    print("🌐 Inventorying TESS light-curve products...")
    tess = query_tess_products()

    print_aliases(tic, simbad, gaia)

    if tic.get("found"):
        metadata = tic.get("metadata", {})
        print()
        print("⭐ TIC stellar metadata")
        print(f"   Tmag: {format_value(metadata.get('tmag'), 3)}")
        print(f"   Teff: {format_value(metadata.get('teffK'), 0)} K")
        print(f"   log g: {format_value(metadata.get('logg'), 3)}")
        print(f"   radius: {format_value(metadata.get('radiusRsun'), 3)} R_sun")
        print(f"   mass: {format_value(metadata.get('massMsun'), 3)} M_sun")
        print(f"   distance: {format_value(metadata.get('distancePc'), 2)} pc")
        print(
            "   TIC contamination ratio: "
            f"{format_value(metadata.get('contaminationRatio'), 6)}"
        )

    print()
    print("📚 AAVSO VSX")
    if vsx.get("queryError"):
        print(f"   QUERY ERROR: {vsx['queryError']}")
    elif not vsx.get("found"):
        print(f"   no object found within {VSX_RADIUS_ARCSEC:.0f} arcsec")
    else:
        for index, match in enumerate(vsx.get("matches", []), start=1):
            print(
                f"   match {index}: {format_value(match.get('name'))} | "
                f"type={format_value(match.get('type'))} | "
                f"P={format_value(match.get('periodDays'), 8)} d | "
                f"sep={format_value(match.get('separationArcsec'), 4)} arcsec"
            )

    print()
    print("🛰 Gaia DR3 variability")
    if gaia_variability.get("queryError"):
        print(f"   QUERY ERROR: {gaia_variability['queryError']}")
    else:
        classification = gaia_variability.get("classification")
        if classification:
            print(
                "   classification: "
                f"{format_value(classification.get('class'))}"
            )
            print(
                "   score: "
                f"{format_value(classification.get('score'), 6)}"
            )
        else:
            print("   classification: [none found]")

        tables = gaia_variability.get("tablesFound") or []
        if tables:
            print("   tables found:")
            for table in tables:
                print(f"      {table}")

        periods = gaia_variability.get("periodCandidates") or []
        if periods:
            print("   published period candidates:")
            for candidate in periods:
                print(
                    f"      {candidate['periodDays']:.8f} d | "
                    f"{candidate['source']} | {candidate['field']}"
                )
        else:
            print("   published period/frequency: [none found]")

    print()
    print("🔭 TESS light-curve inventory")
    official = tess.get("officialProducts") or []
    official_sectors = tess.get("officialSectors") or []

    print(
        "   official SPOC/TESS-SPOC sectors: "
        + (", ".join(str(value) for value in official_sectors)
           if official_sectors else "[none]")
    )

    if official:
        print("   official products:")
        for product in official:
            exposure = product.get("exptimeSeconds")
            exposure_text = (
                f"{exposure:.0f}s" if exposure is not None else "[unknown cadence]"
            )
            print(
                f"      Sector {product.get('sector')} | "
                f"{product.get('author')} | {exposure_text}"
            )

    other = [
        product
        for product in (tess.get("products") or [])
        if product not in official
    ]
    if other:
        print("   other available light-curve products:")
        for product in other:
            exposure = product.get("exptimeSeconds")
            exposure_text = (
                f"{exposure:.0f}s" if exposure is not None else "[unknown cadence]"
            )
            print(
                f"      Sector {product.get('sector')} | "
                f"{product.get('author')} | {exposure_text}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "investigation": "Blind C identity reconnaissance v1",
        "blindTarget": target,
        "frozenResult": {
            "openstar": {
                "frequencyCyclesPerDay": OPENSTAR_FREQUENCY,
                "periodDays": OPENSTAR_PERIOD_DAYS,
                "power": OPENSTAR_POWER,
            },
            "astropy": {
                "frequencyCyclesPerDay": ASTROPY_FREQUENCY,
                "periodDays": ASTROPY_PERIOD_DAYS,
                "power": ASTROPY_POWER,
            },
        },
        "tic": tic,
        "simbad": simbad,
        "vsx": vsx,
        "gaiaDR3": gaia,
        "gaiaVariability": gaia_variability,
        "tess": tess,
    }

    json_path = OUTPUT_DIR / "identity_inventory.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)

    csv_path = write_tess_csv(tess.get("products") or [])

    print()
    print("✅ Blind-C reconnaissance complete")
    print(f"   JSON: {json_path}")
    print(f"   TESS products CSV: {csv_path}")
    print()
    print("Next scientific step:")
    print(
        "Use the resolved aliases and catalog result above to search the "
        "literature before choosing a multi-sector period test."
    )


if __name__ == "__main__":
    main()
