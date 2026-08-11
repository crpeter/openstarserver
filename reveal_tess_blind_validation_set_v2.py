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


# ============================================================
# OpenStar blind-validation reveal v2
#
# IMPORTANT FIX FROM v1:
#   TIC's GAIA field is a Gaia DR2 source_id. Gaia explicitly
#   warns that DR2 source IDs must NOT be assumed to identify
#   the same source in DR3.
#
#   v2 therefore performs:
#
#     TIC Gaia DR2 source_id
#              ↓
#     gaiadr3.dr2_neighbourhood
#              ↓
#     Gaia DR3 source_id
#              ↓
#     Gaia DR3 variability tables
#
# AAVSO VSX remains a coordinate-based cross-match.
#
# Run only after the preregistered OpenStar project has
# completed. This script does not modify any OpenStar science
# data, lock file, or coordinator state.
# ============================================================


LOCK_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-validation-set-v1.lock.json"
)

OUTPUT_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-validation-set-v1.reveal-v2.json"
)

VSX_CATALOG = "B/vsx/vsx"
VSX_RADIUS_ARCSEC = 10.0

GAIA_VARIABILITY_TABLES = (
    "vari_eclipsing_binary",
    "vari_cepheid",
    "vari_long_period_variable",
    "vari_ms_oscillator",
    "vari_rotation_modulation",
    "vari_rrlyrae",
    "vari_short_timescale",
)

# Frozen from the already-completed coordinator run.
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


# ============================================================
# Generic helpers
# ============================================================


def python_value(value):
    if value is None:
        return None

    if np.ma.is_masked(value):
        return None

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def float_or_none(value):
    value = python_value(
        value
    )

    if value is None:
        return None

    try:
        result = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def int_or_none(value):
    value = python_value(
        value
    )

    if value is None:
        return None

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        try:
            return int(
                float(
                    value
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return None


def row_value(
    row,
    names,
    default=None,
):
    colnames = set(
        getattr(
            row,
            "colnames",
            [],
        )
    )

    for name in names:
        if name not in colnames:
            continue

        try:
            value = python_value(
                row[name]
            )
        except Exception:
            continue

        if value is not None:
            return value

    return default


def percent_error(
    measured,
    reference,
):
    if (
        measured is None
        or reference is None
        or reference == 0
    ):
        return None

    return (
        abs(
            measured - reference
        )
        / abs(reference)
        * 100.0
    )


def harmonic_note(
    openstar_period,
    catalog_period,
):
    if (
        catalog_period is None
        or catalog_period <= 0
    ):
        return None

    ratio = (
        openstar_period
        / catalog_period
    )

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
        key=lambda item: abs(
            ratio - item[0]
        ),
    )

    relative_distance = abs(
        ratio - best_ratio
    ) / best_ratio

    if relative_distance <= 0.03:
        return best_label

    return None


def comparison_record(
    openstar_period,
    catalog_period,
    source,
    field,
):
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
            openstar_period
            / catalog_period
            if catalog_period
            else None
        ),
        "harmonicNote": harmonic_note(
            openstar_period,
            catalog_period,
        ),
    }


# ============================================================
# TIC metadata
# ============================================================


def query_tic_metadata(
    target,
):
    tic_id = int(
        target["ticID"]
    )

    coordinate = SkyCoord(
        float(
            target["ra"]
        ),
        float(
            target["dec"]
        ),
        unit="deg",
        frame="icrs",
    )

    table = Catalogs.query_region(
        coordinate,
        radius=(
            5.0 * u.arcsec
        ),
        catalog="TIC",
    )

    exact_row = None

    for row in table:
        row_id = int_or_none(
            row_value(
                row,
                ("ID",),
            )
        )

        if row_id == tic_id:
            exact_row = row
            break

    if exact_row is None:
        return {
            "gaiaDR2SourceID": None,
            "twoMassID": None,
            "tychoID": None,
        }

    return {
        "gaiaDR2SourceID": int_or_none(
            row_value(
                exact_row,
                (
                    "GAIA",
                    "Gaia",
                ),
            )
        ),
        "twoMassID": row_value(
            exact_row,
            (
                "TWOMASS",
                "2MASS",
            ),
        ),
        "tychoID": row_value(
            exact_row,
            ("TYC",),
        ),
    }


# ============================================================
# SIMBAD / VSX
# ============================================================


def query_simbad(
    tic_id,
):
    try:
        table = Simbad.query_object(
            f"TIC {tic_id}"
        )
    except Exception as error:
        return {
            "found": False,
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }

    if table is None or len(table) == 0:
        return {
            "found": False,
        }

    row = table[0]

    return {
        "found": True,
        "mainID": row_value(
            row,
            (
                "main_id",
                "MAIN_ID",
            ),
        ),
        "objectType": row_value(
            row,
            (
                "otype",
                "OTYPE",
                "otype_txt",
            ),
        ),
        "spectralType": row_value(
            row,
            (
                "sp_type",
                "SP_TYPE",
            ),
        ),
    }


def query_vsx(
    target,
):
    coordinate = SkyCoord(
        float(
            target["ra"]
        ),
        float(
            target["dec"]
        ),
        unit="deg",
        frame="icrs",
    )

    vizier = Vizier(
        columns=[
            "**",
            "+_r",
        ],
        row_limit=25,
    )

    try:
        result = vizier.query_region(
            coordinate,
            radius=(
                VSX_RADIUS_ARCSEC
                * u.arcsec
            ),
            catalog=VSX_CATALOG,
        )
    except Exception as error:
        return {
            "found": False,
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }

    if len(result) == 0:
        return {
            "found": False,
        }

    table = result[0]

    if len(table) == 0:
        return {
            "found": False,
        }

    row = table[0]

    separation_arcsec = None

    if "_r" in table.colnames:
        raw_distance = float_or_none(
            row["_r"]
        )

        if raw_distance is not None:
            unit = getattr(
                table["_r"],
                "unit",
                None,
            )

            try:
                if unit is not None:
                    separation_arcsec = float(
                        (
                            raw_distance
                            * unit
                        ).to_value(
                            u.arcsec
                        )
                    )
                else:
                    separation_arcsec = (
                        raw_distance
                        * 60.0
                    )
            except Exception:
                separation_arcsec = (
                    raw_distance
                    * 60.0
                )

    return {
        "found": True,
        "name": row_value(
            row,
            (
                "Name",
                "name",
            ),
        ),
        "type": row_value(
            row,
            (
                "Type",
                "type",
            ),
        ),
        "periodDays": float_or_none(
            row_value(
                row,
                (
                    "Period",
                    "period",
                ),
            )
        ),
        "separationArcsec": (
            separation_arcsec
        ),
    }


# ============================================================
# Gaia DR2 → DR3 authoritative cross-match
# ============================================================


def gaia_query(
    query,
):
    job = Gaia.launch_job_async(
        query,
        dump_to_file=False,
    )

    return job.get_results()


def query_dr3_from_dr2(
    dr2_source_id,
):
    if dr2_source_id is None:
        return {
            "found": False,
            "reason": "TIC has no Gaia DR2 source_id",
        }

    query = f"""
SELECT TOP 10
    x.dr2_source_id,
    x.dr3_source_id,
    x.angular_distance,
    x.magnitude_difference,
    x.proper_motion_propagation,
    g.ra,
    g.dec,
    g.phot_g_mean_mag
FROM gaiadr3.dr2_neighbourhood AS x
JOIN gaiadr3.gaia_source AS g
    ON g.source_id = x.dr3_source_id
WHERE x.dr2_source_id = {int(dr2_source_id)}
ORDER BY x.angular_distance ASC
"""

    try:
        table = gaia_query(
            query
        )
    except Exception as error:
        return {
            "found": False,
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }

    if table is None or len(table) == 0:
        return {
            "found": False,
            "reason": (
                "No Gaia DR3 neighbour for TIC Gaia DR2 source"
            ),
        }

    row = table[0]

    candidates = []

    for candidate_row in table:
        candidates.append(
            {
                "dr3SourceID": int_or_none(
                    row_value(
                        candidate_row,
                        ("dr3_source_id",),
                    )
                ),
                "angularDistanceMas": float_or_none(
                    row_value(
                        candidate_row,
                        ("angular_distance",),
                    )
                ),
                "magnitudeDifference": float_or_none(
                    row_value(
                        candidate_row,
                        ("magnitude_difference",),
                    )
                ),
            }
        )

    return {
        "found": True,
        "dr2SourceID": int(
            dr2_source_id
        ),
        "dr3SourceID": int_or_none(
            row_value(
                row,
                ("dr3_source_id",),
            )
        ),
        "angularDistanceMas": float_or_none(
            row_value(
                row,
                ("angular_distance",),
            )
        ),
        "magnitudeDifference": float_or_none(
            row_value(
                row,
                ("magnitude_difference",),
            )
        ),
        "properMotionPropagation": python_value(
            row_value(
                row,
                ("proper_motion_propagation",),
            )
        ),
        "ra": float_or_none(
            row_value(
                row,
                ("ra",),
            )
        ),
        "dec": float_or_none(
            row_value(
                row,
                ("dec",),
            )
        ),
        "photGMeanMag": float_or_none(
            row_value(
                row,
                ("phot_g_mean_mag",),
            )
        ),
        "candidateCount": int(
            len(table)
        ),
        "candidates": candidates,
    }


# ============================================================
# Gaia DR3 variability
# ============================================================


def query_gaia_classifier(
    dr3_source_id,
):
    if dr3_source_id is None:
        return None

    query = f"""
SELECT
    source_id,
    classifier_name,
    best_class_name,
    best_class_score
FROM gaiadr3.vari_classifier_result
WHERE source_id = {int(dr3_source_id)}
"""

    try:
        table = gaia_query(
            query
        )
    except Exception:
        return None

    if table is None or len(table) == 0:
        return None

    row = table[0]

    return {
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


def query_gaia_vari_summary(
    dr3_source_id,
):
    if dr3_source_id is None:
        return None

    query = f"""
SELECT *
FROM gaiadr3.vari_summary
WHERE source_id = {int(dr3_source_id)}
"""

    try:
        table = gaia_query(
            query
        )
    except Exception:
        return None

    if table is None or len(table) == 0:
        return None

    row = table[0]

    membership = {}

    for column in table.colnames:
        if not column.startswith(
            "in_vari_"
        ):
            continue

        membership[column] = bool(
            python_value(
                row[column]
            )
        )

    return {
        "found": True,
        "membership": membership,
    }


def extract_period_candidates(
    row,
    table_name,
):
    candidates = []

    for column in getattr(
        row,
        "colnames",
        [],
    ):
        lower = column.lower()

        if lower in (
            "source_id",
            "solution_id",
        ):
            continue

        value = float_or_none(
            row[column]
        )

        if (
            value is None
            or value <= 0
        ):
            continue

        if (
            "period" in lower
            and "error" not in lower
            and "percent" not in lower
            and "percentile" not in lower
        ):
            candidates.append(
                {
                    "source": (
                        f"Gaia DR3 {table_name}"
                    ),
                    "field": column,
                    "periodDays": value,
                    "derivedFromFrequency": False,
                }
            )

        elif (
            "frequency" in lower
            and "error" not in lower
            and "fap" not in lower
        ):
            candidates.append(
                {
                    "source": (
                        f"Gaia DR3 {table_name}"
                    ),
                    "field": (
                        f"1/{column}"
                    ),
                    "periodDays": (
                        1.0 / value
                    ),
                    "derivedFromFrequency": True,
                    "frequencyPerDay": value,
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
                abs(
                    candidate[
                        "periodDays"
                    ]
                ) * 1e-9,
            )
            for previous
            in deduplicated
        )

        if not duplicate:
            deduplicated.append(
                candidate
            )

    return deduplicated


def query_gaia_variability_table(
    dr3_source_id,
    table_name,
):
    query = f"""
SELECT *
FROM gaiadr3.{table_name}
WHERE source_id = {int(dr3_source_id)}
"""

    try:
        table = gaia_query(
            query
        )
    except Exception as error:
        return {
            "table": table_name,
            "found": False,
            "queryError": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
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
        "periodCandidates": (
            extract_period_candidates(
                row,
                table_name,
            )
        ),
    }


def query_gaia_variability(
    dr3_source_id,
):
    classifier = (
        query_gaia_classifier(
            dr3_source_id
        )
    )

    summary = (
        query_gaia_vari_summary(
            dr3_source_id
        )
    )

    detail_tables = []

    for table_name in GAIA_VARIABILITY_TABLES:
        detail_tables.append(
            query_gaia_variability_table(
                dr3_source_id,
                table_name,
            )
        )

    period_candidates = []

    for detail in detail_tables:
        period_candidates.extend(
            detail[
                "periodCandidates"
            ]
        )

    return {
        "classifier": classifier,
        "summary": summary,
        "detailTables": detail_tables,
        "periodCandidates": (
            period_candidates
        ),
    }


# ============================================================
# Reporting
# ============================================================


def print_comparison(
    comparison,
):
    print(
        "      period: "
        f"{comparison['catalogPeriodDays']:.8f} d"
    )
    print(
        "      direct difference from OpenStar: "
        f"{comparison['directPercentError']:.4f}%"
    )

    if comparison.get(
        "harmonicNote"
    ):
        print(
            "      relation: "
            f"{comparison['harmonicNote']}"
        )


def reveal_target(
    target,
):
    blind_name = target[
        "blindName"
    ]
    tic_id = int(
        target["ticID"]
    )

    openstar = (
        OPENSTAR_RESULTS[
            blind_name
        ]
    )

    print()
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"⭐ {blind_name} — TIC {tic_id}"
    )
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

    tic = query_tic_metadata(
        target
    )

    print()
    print("🔎 Identity")
    print(
        "   TIC Gaia DR2 source_id: "
        f"{tic['gaiaDR2SourceID'] or '[none]'}"
    )
    print(
        "   2MASS: "
        f"{tic['twoMassID'] or '[none]'}"
    )

    simbad = query_simbad(
        tic_id
    )

    if simbad.get(
        "found"
    ):
        print(
            "   SIMBAD main ID: "
            f"{simbad.get('mainID')}"
        )
        print(
            "   SIMBAD object type: "
            f"{simbad.get('objectType')}"
        )
    else:
        print(
            "   SIMBAD TIC lookup: [none]"
        )

    comparisons = []

    # --------------------------------------------------------
    # AAVSO VSX
    # --------------------------------------------------------

    print()
    print("📚 AAVSO VSX")

    vsx = query_vsx(
        target
    )

    if vsx.get(
        "found"
    ):
        print(
            "   match: "
            f"{vsx.get('name')}"
        )
        print(
            "   type: "
            f"{vsx.get('type')}"
        )

        if (
            vsx.get(
                "separationArcsec"
            )
            is not None
        ):
            print(
                "   separation: "
                f"{vsx['separationArcsec']:.3f} arcsec"
            )

        if (
            vsx.get(
                "periodDays"
            )
            is not None
        ):
            comparison = (
                comparison_record(
                    openstar[
                        "periodDays"
                    ],
                    vsx[
                        "periodDays"
                    ],
                    "AAVSO VSX",
                    "Period",
                )
            )

            comparisons.append(
                comparison
            )

            print_comparison(
                comparison
            )
        else:
            print(
                "   published period: [none]"
            )
    else:
        print(
            "   no VSX object within "
            f"{VSX_RADIUS_ARCSEC:.0f} arcsec"
        )

    # --------------------------------------------------------
    # Gaia DR2 -> DR3 crossmatch
    # --------------------------------------------------------

    print()
    print("🔗 Gaia DR2 → DR3 cross-match")

    dr3 = query_dr3_from_dr2(
        tic[
            "gaiaDR2SourceID"
        ]
    )

    if not dr3.get(
        "found"
    ):
        print(
            "   DR3 counterpart: [none]"
        )

        if dr3.get(
            "reason"
        ):
            print(
                "   reason: "
                f"{dr3['reason']}"
            )

        gaia_variability = {
            "classifier": None,
            "summary": None,
            "detailTables": [],
            "periodCandidates": [],
        }

    else:
        print(
            "   Gaia DR3 source_id: "
            f"{dr3['dr3SourceID']}"
        )
        print(
            "   DR2↔DR3 angular distance: "
            f"{dr3['angularDistanceMas']:.3f} mas"
        )
        print(
            "   candidate counterparts: "
            f"{dr3['candidateCount']}"
        )

        if (
            dr3.get(
                "magnitudeDifference"
            )
            is not None
        ):
            print(
                "   G magnitude difference: "
                f"{dr3['magnitudeDifference']:+.5f}"
            )

        gaia_variability = (
            query_gaia_variability(
                dr3[
                    "dr3SourceID"
                ]
            )
        )

    # --------------------------------------------------------
    # Gaia variability reveal
    # --------------------------------------------------------

    print()
    print("🛰 Gaia DR3 variability")

    classifier = (
        gaia_variability[
            "classifier"
        ]
    )

    if classifier is None:
        print(
            "   classification: [none found]"
        )
    else:
        print(
            "   class: "
            f"{classifier.get('class')}"
        )

        if (
            classifier.get(
                "score"
            )
            is not None
        ):
            print(
                "   class score: "
                f"{classifier['score']:.6f}"
            )

    summary = (
        gaia_variability[
            "summary"
        ]
    )

    if summary is not None:
        memberships = [
            name
            for name, present
            in summary[
                "membership"
            ].items()
            if present
        ]

        if memberships:
            print(
                "   vari_summary memberships:"
            )

            for membership in memberships:
                print(
                    f"      {membership}"
                )

    gaia_periods = (
        gaia_variability[
            "periodCandidates"
        ]
    )

    if len(
        gaia_periods
    ) == 0:
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

            comparison = (
                comparison_record(
                    openstar[
                        "periodDays"
                    ],
                    candidate[
                        "periodDays"
                    ],
                    candidate[
                        "source"
                    ],
                    candidate[
                        "field"
                    ],
                )
            )

            comparisons.append(
                comparison
            )

            print_comparison(
                comparison
            )

    # --------------------------------------------------------
    # Do not cherry-pick a "closest" external answer.
    # --------------------------------------------------------

    if len(comparisons) == 0:
        reveal_status = (
            "NO TRUSTWORTHY CATALOG PERIOD FOUND"
        )
    elif any(
        item[
            "directPercentError"
        ] <= 3.0
        for item in comparisons
        if (
            item[
                "directPercentError"
            ]
            is not None
        )
    ):
        reveal_status = (
            "CATALOG PERIOD COMPATIBLE"
        )
    elif any(
        item.get(
            "harmonicNote"
        )
        is not None
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
        "gaiaDR2ToDR3": dr3,
        "gaiaVariability": gaia_variability,
        "comparisons": comparisons,
        "revealStatus": reveal_status,
    }


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Corrected OpenStar blind-validation "
            "catalog reveal using Gaia's official "
            "DR2-to-DR3 neighbourhood cross-match."
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
            f"Missing blind lock: "
            f"{args.lock}"
        )

    with args.lock.open(
        "r",
        encoding="utf-8",
    ) as file:
        lock_document = json.load(
            file
        )

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
        "🔓 OpenStar Blind Validation — CATALOG REVEAL v2"
    )
    print(
        "OpenStar results remain frozen from the "
        "completed distributed run."
    )
    print(
        "Gaia lookup uses the official DR2 → DR3 "
        "neighbourhood cross-match."
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
            result = reveal_target(
                target
            )
        except Exception as error:
            print()
            print(
                "❌ Reveal failed for "
                f"{target.get('blindName')}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            result = {
                "blindName": target.get(
                    "blindName"
                ),
                "ticID": target.get(
                    "ticID"
                ),
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
                "revealStatus": (
                    "REVEAL QUERY FAILED"
                ),
            }

        revealed.append(
            result
        )

    output = {
        "projectID": lock_document.get(
            "projectID"
        ),
        "selectionLockSHA256": (
            lock_document.get(
                "lockSHA256"
            )
        ),
        "revealVersion": 2,
        "gaiaCrossmatchMethod": (
            "TIC Gaia DR2 source_id -> "
            "gaiadr3.dr2_neighbourhood -> "
            "Gaia DR3 source_id"
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
        "🏁 BLIND REVEAL v2 SUMMARY"
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
        "✅ Corrected catalog reveal complete"
    )


if __name__ == "__main__":
    main()
