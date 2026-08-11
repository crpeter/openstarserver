import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


LOCK_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-validation-set-v1.lock.json"
)

OUTPUT_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-validation-set-v1.reveal-v4.json"
)

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

OPENSTAR_RESULTS = {
    "Blind B": {
        "ticID": 468621617,
        "periodDays": 0.40005345,
        "power": 0.17376943,
    },
    "Blind C": {
        "ticID": 41460085,
        "periodDays": 7.47378546,
        "power": 0.40587783,
    },
    "Blind D": {
        "ticID": 329328236,
        "periodDays": 7.19365395,
        "power": 0.20524040,
    },
    "Blind E": {
        "ticID": 468620943,
        "periodDays": 1.70320623,
        "power": 0.11195137,
    },
    "Blind F": {
        "ticID": 404927661,
        "periodDays": 4.67278907,
        "power": 0.04895498,
    },
    "Blind G": {
        "ticID": 233064123,
        "periodDays": 0.40754430,
        "power": 0.00704431,
    },
    "Blind H": {
        "ticID": 233065677,
        "periodDays": 5.67580965,
        "power": 0.00588597,
    },
    "Blind I": {
        "ticID": 149935043,
        "periodDays": 2.36750985,
        "power": 0.00461870,
    },
}


def python_value(value):
    if value is None or np.ma.is_masked(value):
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

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
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

        value = python_value(
            row[name]
        )

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

    if (
        abs(
            ratio - best_ratio
        )
        / best_ratio
        <= 0.03
    ):
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
        "catalogPeriodDays": (
            catalog_period
        ),
        "directPercentError": (
            percent_error(
                openstar_period,
                catalog_period,
            )
        ),
        "harmonicNote": (
            harmonic_note(
                openstar_period,
                catalog_period,
            )
        ),
    }


def coordinate_for_target(target):
    return SkyCoord(
        float(target["ra"]),
        float(target["dec"]),
        unit="deg",
        frame="icrs",
    )


def query_tic_metadata(target):
    coordinate = coordinate_for_target(
        target
    )

    tic_id = int(
        target["ticID"]
    )

    table = Catalogs.query_region(
        coordinate,
        radius=5.0 * u.arcsec,
        catalog="TIC",
    )

    for row in table:
        if int_or_none(
            row_value(
                row,
                ("ID",),
            )
        ) == tic_id:
            return {
                "gaiaField": int_or_none(
                    row_value(
                        row,
                        ("GAIA", "Gaia"),
                    )
                ),
                "twoMassID": row_value(
                    row,
                    ("TWOMASS", "2MASS"),
                ),
            }

    return {
        "gaiaField": None,
        "twoMassID": None,
    }


def query_simbad(tic_id):
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

    if (
        table is None
        or len(table) == 0
    ):
        return {
            "found": False,
        }

    row = table[0]

    return {
        "found": True,
        "mainID": row_value(
            row,
            ("main_id", "MAIN_ID"),
        ),
        "objectType": row_value(
            row,
            (
                "otype",
                "OTYPE",
                "otype_txt",
            ),
        ),
    }


def query_vsx(target):
    vizier = Vizier(
        columns=[
            "**",
            "+_r",
        ],
        row_limit=20,
    )

    try:
        result = vizier.query_region(
            coordinate_for_target(
                target
            ),
            radius=(
                VSX_RADIUS_ARCSEC
                * u.arcsec
            ),
            catalog=VSX_CATALOG,
        )
    except Exception as error:
        return {
            "found": False,
            "queryError": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }

    if (
        len(result) == 0
        or len(result[0]) == 0
    ):
        return {
            "found": False,
        }

    table = result[0]
    row = table[0]

    separation = float_or_none(
        row_value(
            row,
            ("_r",),
        )
    )

    if separation is not None:
        unit = getattr(
            table["_r"],
            "unit",
            None,
        )

        try:
            if unit is not None:
                separation = float(
                    (
                        separation * unit
                    ).to_value(
                        u.arcsec
                    )
                )
        except Exception:
            pass

    return {
        "found": True,
        "name": row_value(
            row,
            ("Name", "name"),
        ),
        "type": row_value(
            row,
            ("Type", "type"),
        ),
        "periodDays": float_or_none(
            row_value(
                row,
                ("Period", "period"),
            )
        ),
        "separationArcsec": separation,
    }


def query_gaia_main(target):
    vizier = Vizier(
        columns=[
            "**",
            "+_r",
        ],
        row_limit=20,
    )

    try:
        result = vizier.query_region(
            coordinate_for_target(
                target
            ),
            radius=(
                GAIA_RADIUS_ARCSEC
                * u.arcsec
            ),
            catalog=GAIA_MAIN_CATALOG,
        )
    except Exception as error:
        return {
            "found": False,
            "queryError": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }

    if (
        len(result) == 0
        or len(result[0]) == 0
    ):
        return {
            "found": False,
        }

    table = result[0]
    row = table[0]

    source_id = int_or_none(
        row_value(
            row,
            ("Source", "source_id"),
        )
    )

    separation = float_or_none(
        row_value(
            row,
            ("_r",),
        )
    )

    if separation is not None:
        unit = getattr(
            table["_r"],
            "unit",
            None,
        )

        try:
            if unit is not None:
                separation = float(
                    (
                        separation * unit
                    ).to_value(
                        u.arcsec
                    )
                )
        except Exception:
            pass

    return {
        "found": (
            source_id is not None
        ),
        "sourceID": source_id,
        "separationArcsec": separation,
        "gMag": float_or_none(
            row_value(
                row,
                ("Gmag",),
            )
        ),
    }


def extract_period_candidates(
    table,
    source_label,
):
    if len(table) == 0:
        return []

    row = table[0]
    candidates = []

    period_field_names = {
        "period",
        "pf",
        "p1o",
        "p2o",
        "prot",
    }

    frequency_field_names = {
        "freq",
        "frequency",
        "fundfreq1",
        "fundfreq2",
    }

    for column in table.colnames:
        normalized = (
            column
            .lower()
            .replace("_", "")
        )

        value = float_or_none(
            row[column]
        )

        if (
            value is None
            or value <= 0
        ):
            continue

        is_period = (
            normalized in period_field_names
            or (
                "period" in normalized
                and "error" not in normalized
                and "percent" not in normalized
            )
        )

        is_frequency = (
            normalized in frequency_field_names
            or (
                "frequency" in normalized
                and "error" not in normalized
            )
        )

        if is_period:
            period_days = value

            unit = getattr(
                table[column],
                "unit",
                None,
            )

            try:
                if unit is not None:
                    period_days = float(
                        (
                            value * unit
                        ).to_value(
                            u.day
                        )
                    )
            except Exception:
                pass

            candidates.append(
                {
                    "source": source_label,
                    "field": column,
                    "periodDays": (
                        period_days
                    ),
                }
            )

        elif is_frequency:
            frequency_per_day = value

            unit = getattr(
                table[column],
                "unit",
                None,
            )

            try:
                if unit is not None:
                    frequency_per_day = float(
                        (
                            value * unit
                        ).to_value(
                            1 / u.day
                        )
                    )
            except Exception:
                pass

            if frequency_per_day > 0:
                candidates.append(
                    {
                        "source": source_label,
                        "field": (
                            f"1/{column}"
                        ),
                        "periodDays": (
                            1.0
                            / frequency_per_day
                        ),
                    }
                )

    deduplicated = []

    for candidate in candidates:
        period = candidate[
            "periodDays"
        ]

        if any(
            abs(
                period
                - previous[
                    "periodDays"
                ]
            )
            <= max(
                1e-10,
                abs(period) * 1e-9,
            )
            for previous
            in deduplicated
        ):
            continue

        deduplicated.append(
            candidate
        )

    return deduplicated


def query_gaia_variability(
    source_id,
):
    vizier = Vizier(
        columns=["**"],
        row_limit=20,
    )

    try:
        result = (
            vizier.query_constraints(
                catalog=list(
                    GAIA_VARIABILITY_CATALOGS
                ),
                Source=str(
                    int(source_id)
                ),
            )
        )
    except Exception as error:
        return {
            "queryError": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
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

        tables_found.append(
            str(key)
        )

        if "vclassre" in str(key):
            row = table[0]

            classification = {
                "class": row_value(
                    row,
                    ("Class",),
                ),
                "score": float_or_none(
                    row_value(
                        row,
                        ("ClassSc",),
                    )
                ),
                "classifier": row_value(
                    row,
                    ("Classifier",),
                ),
            }

            continue

        periods.extend(
            extract_period_candidates(
                table,
                str(key),
            )
        )

    return {
        "classification": (
            classification
        ),
        "tablesFound": tables_found,
        "periodCandidates": periods,
    }


def print_comparison(
    comparison,
):
    print(
        "      catalog period: "
        f"{comparison['catalogPeriodDays']:.8f} d"
    )
    print(
        "      difference from OpenStar: "
        f"{comparison['directPercentError']:.4f}%"
    )

    if comparison.get(
        "harmonicNote"
    ):
        print(
            "      relation: "
            f"{comparison['harmonicNote']}"
        )


def reveal_target(target):
    blind_name = target[
        "blindName"
    ]
    tic_id = int(
        target["ticID"]
    )

    openstar = OPENSTAR_RESULTS[
        blind_name
    ]

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
        "   TIC GAIA field: "
        f"{tic['gaiaField'] or '[none]'}"
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
    else:
        print(
            "   SIMBAD TIC lookup: [none]"
        )

    comparisons = []
    query_errors = []

    print()
    print("📚 AAVSO VSX")

    vsx = query_vsx(
        target
    )

    if vsx.get(
        "queryError"
    ):
        query_errors.append(
            "VSX"
        )
        print(
            "   QUERY ERROR: "
            f"{vsx['queryError']}"
        )

    elif vsx.get(
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

        if vsx.get(
            "periodDays"
        ) is not None:
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

    print()
    print(
        "🔗 Locked coordinate → Gaia DR3 via VizieR"
    )

    gaia = query_gaia_main(
        target
    )

    variability = None

    if gaia.get(
        "queryError"
    ):
        query_errors.append(
            "Gaia DR3 VizieR"
        )
        print(
            "   QUERY ERROR: "
            f"{gaia['queryError']}"
        )

    elif not gaia.get(
        "found"
    ):
        print(
            "   Gaia DR3 counterpart: [none]"
        )

    else:
        print(
            "   Gaia DR3 source_id: "
            f"{gaia['sourceID']}"
        )

        if gaia.get(
            "separationArcsec"
        ) is not None:
            print(
                "   separation: "
                f"{gaia['separationArcsec']:.4f} arcsec"
            )

        if tic.get(
            "gaiaField"
        ) is not None:
            same = (
                int(
                    tic["gaiaField"]
                )
                == int(
                    gaia["sourceID"]
                )
            )

            print(
                "   TIC GAIA field equals Gaia DR3 source_id: "
                f"{'YES' if same else 'NO'}"
            )

        variability = (
            query_gaia_variability(
                gaia[
                    "sourceID"
                ]
            )
        )

    print()
    print(
        "🛰 Gaia DR3 variability via VizieR"
    )

    if variability is None:
        print(
            "   not evaluated"
        )

    elif variability.get(
        "queryError"
    ):
        query_errors.append(
            "Gaia variability VizieR"
        )

        print(
            "   QUERY ERROR: "
            f"{variability['queryError']}"
        )

    else:
        classification = (
            variability[
                "classification"
            ]
        )

        if classification is None:
            print(
                "   classification: [none found]"
            )
        else:
            print(
                "   class: "
                f"{classification.get('class')}"
            )

            if classification.get(
                "score"
            ) is not None:
                print(
                    "   class score: "
                    f"{classification['score']:.6f}"
                )

        if variability[
            "tablesFound"
        ]:
            print(
                "   Gaia variability tables:"
            )

            for table_name in variability[
                "tablesFound"
            ]:
                print(
                    f"      {table_name}"
                )

        periods = variability[
            "periodCandidates"
        ]

        if not periods:
            print(
                "   published Gaia period/frequency: "
                "[none found]"
            )

        for candidate in periods:
            print(
                "   source: "
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

    if query_errors:
        status = (
            "CATALOG QUERY ERROR — RESULT INCOMPLETE"
        )

    elif not comparisons:
        status = (
            "NO TRUSTWORTHY CATALOG PERIOD FOUND"
        )

    elif any(
        item[
            "directPercentError"
        ] is not None
        and item[
            "directPercentError"
        ] <= 3.0
        for item in comparisons
    ):
        status = (
            "CATALOG PERIOD COMPATIBLE"
        )

    elif any(
        item.get(
            "harmonicNote"
        ) is not None
        for item in comparisons
    ):
        status = (
            "CATALOG HARMONIC RELATION FOUND"
        )

    else:
        status = (
            "CATALOG PERIOD FOUND — NOT A DIRECT MATCH"
        )

    print()
    print(
        f"🏷 Reveal status: {status}"
    )

    return {
        "blindName": blind_name,
        "ticID": tic_id,
        "openstar": openstar,
        "tic": tic,
        "simbad": simbad,
        "vsx": vsx,
        "gaiaDR3": gaia,
        "gaiaVariability": variability,
        "comparisons": comparisons,
        "queryErrors": query_errors,
        "revealStatus": status,
    }


def main():
    parser = argparse.ArgumentParser()

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

    with args.lock.open(
        "r",
        encoding="utf-8",
    ) as file:
        lock_document = json.load(
            file
        )

    targets = lock_document[
        "targets"
    ]

    print(
        "🔓 OpenStar Blind Validation — CATALOG REVEAL v4"
    )
    print(
        "Gaia lookup: VizieR only "
        "(no Gaia TAP/Archive calls)"
    )
    print(
        f"targets: {len(targets)}"
    )

    revealed = []

    for target in targets:
        revealed.append(
            reveal_target(
                target
            )
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
        "revealVersion": 4,
        "gaiaLookup": (
            "VizieR I/355/gaiadr3 + I/358"
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
    print(
        "🏁 BLIND REVEAL v4 SUMMARY"
    )
    print(
        "════════════════════════════════════════════════════════"
    )

    for item in revealed:
        print(
            f"{item['blindName']}: "
            f"{item['revealStatus']}"
        )

    print()
    print(
        f"💾 Reveal record: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
