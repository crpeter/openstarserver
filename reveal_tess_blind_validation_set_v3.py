import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


LOCK_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-validation-set-v1.lock.json"
)

OUTPUT_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-validation-set-v1.reveal-v3.json"
)

VSX_CATALOG = "B/vsx/vsx"
VSX_RADIUS_ARCSEC = 10.0
GAIA_CONE_RADIUS_ARCSEC = 5.0

GAIA_DETAIL_TABLES = (
    "vari_cepheid",
    "vari_eclipsing_binary",
    "vari_long_period_variable",
    "vari_ms_oscillator",
    "vari_rotation_modulation",
    "vari_rrlyrae",
    "vari_short_timescale",
)

OPENSTAR_RESULTS = {
    "Blind B": {
        "ticID": 468621617,
        "frequency": 2.49966595,
        "periodDays": 0.40005345,
        "power": 0.17376943,
    },
    "Blind C": {
        "ticID": 41460085,
        "frequency": 0.13380100,
        "periodDays": 7.47378546,
        "power": 0.40587783,
    },
    "Blind D": {
        "ticID": 329328236,
        "frequency": 0.13901141,
        "periodDays": 7.19365395,
        "power": 0.20524040,
    },
    "Blind E": {
        "ticID": 468620943,
        "frequency": 0.58712796,
        "periodDays": 1.70320623,
        "power": 0.11195137,
    },
    "Blind F": {
        "ticID": 404927661,
        "frequency": 0.21400495,
        "periodDays": 4.67278907,
        "power": 0.04895498,
    },
    "Blind G": {
        "ticID": 233064123,
        "frequency": 2.45372098,
        "periodDays": 0.40754430,
        "power": 0.00704431,
    },
    "Blind H": {
        "ticID": 233065677,
        "frequency": 0.17618632,
        "periodDays": 5.67580965,
        "power": 0.00588597,
    },
    "Blind I": {
        "ticID": 149935043,
        "frequency": 0.42238473,
        "periodDays": 2.36750985,
        "power": 0.00461870,
    },
}


def python_value(value):
    if value is None:
        return None
    if np.ma.is_masked(value):
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
    return result if math.isfinite(result) else None


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
        try:
            value = python_value(row[name])
        except Exception:
            continue
        if value is not None:
            return value
    return default


def percent_error(measured, reference):
    if measured is None or reference is None or reference == 0:
        return None
    return abs(measured - reference) / abs(reference) * 100.0


def harmonic_note(openstar_period, catalog_period):
    if catalog_period is None or catalog_period <= 0:
        return None

    ratio = openstar_period / catalog_period
    candidates = (
        (0.25, "OpenStar ≈ 1/4 catalog period"),
        (0.50, "OpenStar ≈ 1/2 catalog period"),
        (1.00, "direct period match"),
        (2.00, "OpenStar ≈ 2× catalog period"),
        (3.00, "OpenStar ≈ 3× catalog period"),
        (4.00, "OpenStar ≈ 4× catalog period"),
    )

    best_ratio, best_label = min(
        candidates,
        key=lambda item: abs(ratio - item[0]),
    )

    if abs(ratio - best_ratio) / best_ratio <= 0.03:
        return best_label

    return None


def comparison_record(openstar_period, catalog_period, source, field):
    return {
        "source": source,
        "field": field,
        "catalogPeriodDays": catalog_period,
        "openstarPeriodDays": openstar_period,
        "directPercentError": percent_error(
            openstar_period,
            catalog_period,
        ),
        "periodRatioOpenStarToCatalog": (
            openstar_period / catalog_period
            if catalog_period else None
        ),
        "harmonicNote": harmonic_note(
            openstar_period,
            catalog_period,
        ),
    }


def query_tic_metadata(target):
    tic_id = int(target["ticID"])

    coordinate = SkyCoord(
        float(target["ra"]),
        float(target["dec"]),
        unit="deg",
        frame="icrs",
    )

    table = Catalogs.query_region(
        coordinate,
        radius=5.0 * u.arcsec,
        catalog="TIC",
    )

    exact_row = None

    for row in table:
        row_id = int_or_none(row_value(row, ("ID",)))
        if row_id == tic_id:
            exact_row = row
            break

    if exact_row is None:
        return {
            "gaiaField": None,
            "twoMassID": None,
            "tychoID": None,
        }

    return {
        "gaiaField": int_or_none(
            row_value(exact_row, ("GAIA", "Gaia"))
        ),
        "twoMassID": row_value(
            exact_row,
            ("TWOMASS", "2MASS"),
        ),
        "tychoID": row_value(exact_row, ("TYC",)),
    }


def query_simbad(tic_id):
    try:
        table = Simbad.query_object(f"TIC {tic_id}")
    except Exception as error:
        return {
            "found": False,
            "error": f"{type(error).__name__}: {error}",
        }

    if table is None or len(table) == 0:
        return {"found": False}

    row = table[0]

    return {
        "found": True,
        "mainID": row_value(
            row,
            ("main_id", "MAIN_ID"),
        ),
        "objectType": row_value(
            row,
            ("otype", "OTYPE", "otype_txt"),
        ),
        "spectralType": row_value(
            row,
            ("sp_type", "SP_TYPE"),
        ),
    }


def query_vsx(target):
    coordinate = SkyCoord(
        float(target["ra"]),
        float(target["dec"]),
        unit="deg",
        frame="icrs",
    )

    vizier = Vizier(
        columns=["**", "+_r"],
        row_limit=25,
    )

    try:
        result = vizier.query_region(
            coordinate,
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
    row = table[0]

    separation_arcsec = None

    if "_r" in table.colnames:
        raw_distance = float_or_none(row["_r"])
        if raw_distance is not None:
            unit = getattr(table["_r"], "unit", None)
            try:
                if unit is not None:
                    separation_arcsec = float(
                        (raw_distance * unit).to_value(u.arcsec)
                    )
                else:
                    separation_arcsec = raw_distance * 60.0
            except Exception:
                separation_arcsec = raw_distance * 60.0

    return {
        "found": True,
        "name": row_value(row, ("Name", "name")),
        "type": row_value(row, ("Type", "type")),
        "periodDays": float_or_none(
            row_value(row, ("Period", "period"))
        ),
        "separationArcsec": separation_arcsec,
    }


def query_gaia_dr3_by_coordinate(target):
    coordinate = SkyCoord(
        float(target["ra"]),
        float(target["dec"]),
        unit="deg",
        frame="icrs",
    )

    old_table = Gaia.MAIN_GAIA_TABLE
    old_limit = Gaia.ROW_LIMIT

    try:
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = 10

        job = Gaia.cone_search_async(
            coordinate,
            radius=GAIA_CONE_RADIUS_ARCSEC * u.arcsec,
        )

        table = job.get_results()

    except Exception as error:
        return {
            "found": False,
            "queryError": f"{type(error).__name__}: {error}",
        }

    finally:
        Gaia.MAIN_GAIA_TABLE = old_table
        Gaia.ROW_LIMIT = old_limit

    if table is None or len(table) == 0:
        return {
            "found": False,
            "reason": "No Gaia DR3 source inside coordinate cone",
        }

    candidate_coordinates = SkyCoord(
        np.asarray(table["ra"], dtype=np.float64),
        np.asarray(table["dec"], dtype=np.float64),
        unit="deg",
        frame="icrs",
    )

    separations = coordinate.separation(
        candidate_coordinates
    ).arcsec

    best_index = int(np.argmin(separations))
    row = table[best_index]

    candidates = []

    for index in range(len(table)):
        candidates.append(
            {
                "sourceID": int_or_none(
                    table["source_id"][index]
                ),
                "separationArcsec": float(
                    separations[index]
                ),
                "gMag": float_or_none(
                    table["phot_g_mean_mag"][index]
                    if "phot_g_mean_mag" in table.colnames
                    else None
                ),
            }
        )

    return {
        "found": True,
        "sourceID": int_or_none(row["source_id"]),
        "ra": float_or_none(row["ra"]),
        "dec": float_or_none(row["dec"]),
        "gMag": float_or_none(
            row["phot_g_mean_mag"]
            if "phot_g_mean_mag" in table.colnames
            else None
        ),
        "separationArcsec": float(
            separations[best_index]
        ),
        "candidateCount": int(len(table)),
        "candidates": candidates,
    }


def gaia_adql(query):
    job = Gaia.launch_job_async(
        query,
        dump_to_file=False,
    )
    return job.get_results()


def query_gaia_classifier(source_id):
    query = f"""
SELECT
    source_id,
    classifier_name,
    best_class_name,
    best_class_score
FROM gaiadr3.vari_classifier_result
WHERE source_id = {int(source_id)}
"""

    try:
        table = gaia_adql(query)
    except Exception as error:
        return {
            "found": False,
            "queryError": f"{type(error).__name__}: {error}",
        }

    if table is None or len(table) == 0:
        return {"found": False}

    row = table[0]

    return {
        "found": True,
        "classifierName": row_value(
            row,
            ("classifier_name",),
        ),
        "class": row_value(
            row,
            ("best_class_name",),
        ),
        "score": float_or_none(
            row_value(
                row,
                ("best_class_score",),
            )
        ),
    }


def query_gaia_summary(source_id):
    query = f"""
SELECT *
FROM gaiadr3.vari_summary
WHERE source_id = {int(source_id)}
"""

    try:
        table = gaia_adql(query)
    except Exception as error:
        return {
            "found": False,
            "queryError": f"{type(error).__name__}: {error}",
        }

    if table is None or len(table) == 0:
        return {"found": False}

    row = table[0]
    memberships = {}

    for column in table.colnames:
        if column.startswith("in_vari_"):
            memberships[column] = bool(
                python_value(row[column])
            )

    return {
        "found": True,
        "memberships": memberships,
    }


def extract_period_candidates(table, row, source_label):
    candidates = []

    for column in table.colnames:
        lower = column.lower()

        if lower in ("source_id", "solution_id"):
            continue

        value = float_or_none(row[column])

        if value is None or value <= 0:
            continue

        if (
            "period" in lower
            and "error" not in lower
            and "percent" not in lower
            and "percentile" not in lower
        ):
            unit = getattr(table[column], "unit", None)
            period_days = value

            try:
                if unit is not None:
                    period_days = float(
                        (value * unit).to_value(u.day)
                    )
            except Exception:
                pass

            candidates.append(
                {
                    "source": source_label,
                    "field": column,
                    "periodDays": period_days,
                    "derivedFromFrequency": False,
                }
            )

        elif (
            "frequency" in lower
            and "error" not in lower
            and "fap" not in lower
        ):
            unit = getattr(table[column], "unit", None)
            frequency_per_day = value

            try:
                if unit is not None:
                    frequency_per_day = float(
                        (value * unit).to_value(1.0 / u.day)
                    )
            except Exception:
                pass

            if frequency_per_day > 0:
                candidates.append(
                    {
                        "source": source_label,
                        "field": f"1/{column}",
                        "periodDays": 1.0 / frequency_per_day,
                        "derivedFromFrequency": True,
                        "frequencyPerDay": frequency_per_day,
                    }
                )

    deduplicated = []

    for candidate in candidates:
        duplicate = any(
            abs(
                candidate["periodDays"]
                - previous["periodDays"]
            )
            <= max(
                1e-10,
                abs(candidate["periodDays"]) * 1e-9,
            )
            for previous in deduplicated
        )

        if not duplicate:
            deduplicated.append(candidate)

    return deduplicated


def query_gaia_detail_table(source_id, table_name):
    query = f"""
SELECT *
FROM gaiadr3.{table_name}
WHERE source_id = {int(source_id)}
"""

    try:
        table = gaia_adql(query)
    except Exception as error:
        return {
            "table": table_name,
            "found": False,
            "queryError": f"{type(error).__name__}: {error}",
            "periodCandidates": [],
        }

    if table is None or len(table) == 0:
        return {
            "table": table_name,
            "found": False,
            "periodCandidates": [],
        }

    row = table[0]

    return {
        "table": table_name,
        "found": True,
        "periodCandidates": extract_period_candidates(
            table,
            row,
            f"Gaia DR3 {table_name}",
        ),
    }


def relevant_gaia_detail_tables(summary):
    if not summary.get("found"):
        return ()

    memberships = summary["memberships"]
    relevant = []

    for table_name in GAIA_DETAIL_TABLES:
        if memberships.get(
            f"in_{table_name}",
            False,
        ):
            relevant.append(table_name)

    return tuple(relevant)


def query_gaia_variability(source_id):
    classifier = query_gaia_classifier(
        source_id
    )

    summary = query_gaia_summary(
        source_id
    )

    details = []

    for table_name in relevant_gaia_detail_tables(
        summary
    ):
        details.append(
            query_gaia_detail_table(
                source_id,
                table_name,
            )
        )

    period_candidates = []

    for detail in details:
        period_candidates.extend(
            detail["periodCandidates"]
        )

    return {
        "classifier": classifier,
        "summary": summary,
        "detailTables": details,
        "periodCandidates": period_candidates,
    }


def print_comparison(comparison):
    print(
        "      catalog period: "
        f"{comparison['catalogPeriodDays']:.8f} d"
    )
    print(
        "      direct difference from OpenStar: "
        f"{comparison['directPercentError']:.4f}%"
    )

    if comparison.get("harmonicNote"):
        print(
            "      relation: "
            f"{comparison['harmonicNote']}"
        )


def reveal_target(target):
    blind_name = target["blindName"]
    tic_id = int(target["ticID"])
    openstar = OPENSTAR_RESULTS[blind_name]

    print()
    print(
        "════════════════════════════════════════════════════════"
    )
    print(f"⭐ {blind_name} — TIC {tic_id}")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "   OpenStar period: "
        f"{openstar['periodDays']:.8f} d"
    )
    print(
        "   OpenStar power: "
        f"{openstar['power']:.8f}"
    )
    print(
        "   locked coordinate: "
        f"{float(target['ra']):.8f}, "
        f"{float(target['dec']):.8f}"
    )

    tic = query_tic_metadata(target)

    print()
    print("🔎 Identity")
    print(
        "   TIC GAIA field: "
        f"{tic['gaiaField'] or '[none]'}"
    )
    print(
        "   2MASS: "
        f"{tic['twoMassID'] or '[none]'}"
    )

    simbad = query_simbad(tic_id)

    if simbad.get("found"):
        print(
            "   SIMBAD main ID: "
            f"{simbad.get('mainID')}"
        )
        print(
            "   SIMBAD object type: "
            f"{simbad.get('objectType')}"
        )
    else:
        print("   SIMBAD TIC lookup: [none]")

    comparisons = []

    print()
    print("📚 AAVSO VSX")

    vsx = query_vsx(target)

    if vsx.get("queryError"):
        print(
            "   QUERY ERROR: "
            f"{vsx['queryError']}"
        )
    elif vsx.get("found"):
        print(
            "   match: "
            f"{vsx.get('name')}"
        )
        print(
            "   type: "
            f"{vsx.get('type')}"
        )

        if vsx.get("separationArcsec") is not None:
            print(
                "   separation: "
                f"{vsx['separationArcsec']:.3f} arcsec"
            )

        if vsx.get("periodDays") is not None:
            comparison = comparison_record(
                openstar["periodDays"],
                vsx["periodDays"],
                "AAVSO VSX",
                "Period",
            )

            comparisons.append(comparison)
            print_comparison(comparison)
        else:
            print("   published period: [none]")
    else:
        print(
            "   no VSX object within "
            f"{VSX_RADIUS_ARCSEC:.0f} arcsec"
        )

    print()
    print("🔗 Locked coordinate → Gaia DR3")

    gaia_match = query_gaia_dr3_by_coordinate(
        target
    )

    gaia_variability = None

    if gaia_match.get("queryError"):
        print(
            "   QUERY ERROR: "
            f"{gaia_match['queryError']}"
        )

    elif not gaia_match.get("found"):
        print(
            "   Gaia DR3 counterpart: [none]"
        )

        if gaia_match.get("reason"):
            print(
                "   reason: "
                f"{gaia_match['reason']}"
            )

    else:
        print(
            "   Gaia DR3 source_id: "
            f"{gaia_match['sourceID']}"
        )
        print(
            "   separation from locked TIC coordinate: "
            f"{gaia_match['separationArcsec']:.4f} arcsec"
        )
        print(
            "   Gaia sources in 5 arcsec cone: "
            f"{gaia_match['candidateCount']}"
        )

        if tic["gaiaField"] is not None:
            same = (
                int(tic["gaiaField"])
                == int(gaia_match["sourceID"])
            )

            print(
                "   TIC GAIA field equals selected "
                "Gaia DR3 source_id: "
                f"{'YES' if same else 'NO'}"
            )

        gaia_variability = query_gaia_variability(
            gaia_match["sourceID"]
        )

    print()
    print("🛰 Gaia DR3 variability")

    if gaia_variability is None:
        print(
            "   not evaluated because Gaia source "
            "cross-match failed"
        )
    else:
        classifier = gaia_variability["classifier"]

        if classifier.get("queryError"):
            print(
                "   classifier QUERY ERROR: "
                f"{classifier['queryError']}"
            )
        elif classifier.get("found"):
            print(
                "   class: "
                f"{classifier.get('class')}"
            )

            if classifier.get("score") is not None:
                print(
                    "   class score: "
                    f"{classifier['score']:.6f}"
                )
        else:
            print(
                "   classification: [none found]"
            )

        summary = gaia_variability["summary"]

        if summary.get("queryError"):
            print(
                "   vari_summary QUERY ERROR: "
                f"{summary['queryError']}"
            )
        elif summary.get("found"):
            memberships = [
                name
                for name, present
                in summary["memberships"].items()
                if present
            ]

            if memberships:
                print(
                    "   vari_summary memberships:"
                )
                for membership in memberships:
                    print(f"      {membership}")
            else:
                print(
                    "   vari_summary row exists "
                    "without SOS memberships"
                )
        else:
            print(
                "   vari_summary: [none found]"
            )

        for detail in gaia_variability[
            "detailTables"
        ]:
            if detail.get("queryError"):
                print(
                    "   "
                    f"{detail['table']} QUERY ERROR: "
                    f"{detail['queryError']}"
                )

        gaia_periods = gaia_variability[
            "periodCandidates"
        ]

        if len(gaia_periods) == 0:
            print(
                "   published Gaia period/frequency: "
                "[none found]"
            )
        else:
            for candidate in gaia_periods:
                print(
                    "   "
                    f"{candidate['source']} "
                    f"({candidate['field']})"
                )

                comparison = comparison_record(
                    openstar["periodDays"],
                    candidate["periodDays"],
                    candidate["source"],
                    candidate["field"],
                )

                comparisons.append(comparison)
                print_comparison(comparison)

    query_errors = []

    if vsx.get("queryError"):
        query_errors.append("VSX")

    if gaia_match.get("queryError"):
        query_errors.append(
            "Gaia DR3 cone search"
        )

    if gaia_variability is not None:
        if gaia_variability[
            "classifier"
        ].get("queryError"):
            query_errors.append(
                "Gaia classifier"
            )

        if gaia_variability[
            "summary"
        ].get("queryError"):
            query_errors.append(
                "Gaia vari_summary"
            )

        for detail in gaia_variability[
            "detailTables"
        ]:
            if detail.get("queryError"):
                query_errors.append(
                    "Gaia " + detail["table"]
                )

    if query_errors:
        reveal_status = (
            "CATALOG QUERY ERROR — RESULT INCOMPLETE"
        )
    elif len(comparisons) == 0:
        reveal_status = (
            "NO TRUSTWORTHY CATALOG PERIOD FOUND"
        )
    elif any(
        item["directPercentError"] is not None
        and item["directPercentError"] <= 3.0
        for item in comparisons
    ):
        reveal_status = (
            "CATALOG PERIOD COMPATIBLE"
        )
    elif any(
        item.get("harmonicNote") is not None
        for item in comparisons
    ):
        reveal_status = (
            "CATALOG HARMONIC RELATION FOUND"
        )
    else:
        reveal_status = (
            "CATALOG PERIOD FOUND — NOT A DIRECT MATCH"
        )

    print()
    print(
        f"🏷 Reveal status: {reveal_status}"
    )

    return {
        "blindName": blind_name,
        "ticID": tic_id,
        "openstar": openstar,
        "tic": tic,
        "simbad": simbad,
        "vsx": vsx,
        "gaiaDR3CoordinateMatch": gaia_match,
        "gaiaVariability": gaia_variability,
        "comparisons": comparisons,
        "queryErrors": query_errors,
        "revealStatus": reveal_status,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpenStar blind catalog reveal using "
            "direct locked-coordinate cross-match "
            "into Gaia DR3."
        )
    )

    parser.add_argument(
        "--lock",
        type=Path,
        default=LOCK_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )

    args = parser.parse_args()

    if not args.lock.exists():
        raise RuntimeError(
            f"Missing blind lock: {args.lock}"
        )

    with args.lock.open(
        "r",
        encoding="utf-8",
    ) as file:
        lock_document = json.load(file)

    targets = lock_document.get(
        "targets",
        []
    )

    if len(targets) != len(
        OPENSTAR_RESULTS
    ):
        raise RuntimeError(
            "Locked target count does not match "
            "the completed OpenStar result set."
        )

    print(
        "🔓 OpenStar Blind Validation — CATALOG REVEAL v3"
    )
    print(
        "OpenStar results remain frozen from the "
        "completed distributed run."
    )
    print(
        "Gaia DR3 identity is selected directly "
        "from each target's locked sky coordinate."
    )
    print(
        f"targets: {len(targets)}"
    )
    print(
        "catalogs: AAVSO VSX + Gaia DR3 + SIMBAD"
    )

    revealed = []

    for target in targets:
        try:
            result = reveal_target(target)
        except Exception as error:
            print()
            print(
                "❌ Reveal failed for "
                f"{target.get('blindName')}: "
                f"{type(error).__name__}: {error}"
            )

            result = {
                "blindName": target.get(
                    "blindName"
                ),
                "ticID": target.get("ticID"),
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
                "revealStatus": (
                    "REVEAL QUERY FAILED"
                ),
            }

        revealed.append(result)

    output = {
        "projectID": lock_document.get(
            "projectID"
        ),
        "selectionLockSHA256": (
            lock_document.get(
                "lockSHA256"
            )
        ),
        "revealVersion": 3,
        "gaiaCrossmatchMethod": (
            "locked TIC coordinate -> "
            "5 arcsec Gaia DR3 cone search -> "
            "nearest source"
        ),
        "targets": revealed,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            allow_nan=False,
        )

    print()
    print()
    print(
        "🏁 BLIND REVEAL v3 SUMMARY"
    )
    print(
        "════════════════════════════════════════════════════════"
    )

    for item in revealed:
        print(
            f"{item.get('blindName')}: "
            f"{item.get('revealStatus')}"
        )

    print()
    print(
        f"💾 Reveal record: {args.output}"
    )
    print(
        "✅ Coordinate-based catalog reveal complete"
    )


if __name__ == "__main__":
    main()
