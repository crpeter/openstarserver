import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import lightkurve as lk
import numpy as np
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind Published-Period Validation Set v3
#
# Goal:
#   Build a genuinely blind external-validation set where every
#   selected target is guaranteed to have an independently
#   cataloged AAVSO VSX period, but the selector NEVER requests,
#   reads, stores, prints, or ranks by the actual period value.
#
# Selection knows only:
#   - a VSX row exists
#   - VSX Period is inside the already-frozen OpenStar range
#   - sky coordinates
#   - TESS/TIC identity and light-curve availability/quality
#
# Selection deliberately does NOT use:
#   - published period value
#   - published frequency
#   - VSX variable name
#   - VSX variability classification
#   - TESS periodogram strength
#   - any TESS-derived winning frequency/period
#
# Stages:
#
#   --select
#       Server-side VSX period-existence/range filter
#       -> coordinate cross-match to TIC
#       -> TESS data-quality preflight only
#       -> deterministic selection of eight targets
#       -> SHA-256 lock
#
#   --prepare
#       Verify the exact lock
#       -> build Float32 TESS datasets
#       -> build exact Astropy chunk references
#       -> create multi-target OpenStar project
#
# Only after the distributed OpenStar project completes should a
# separate reveal script request the VSX Period field.
# ============================================================


PROJECT_ID = "openstar.tess-blind-published-v3"
PROJECT_NAME = "OpenStar Blind Published-Period Validation Set v3"
WORKLOAD_ID = "openstar.tess-period-search.v1"

TARGET_COUNT = 8

BLIND_NAMES = (
    "Blind V2-A",
    "Blind V2-B",
    "Blind V2-C",
    "Blind V2-D",
    "Blind V2-E",
    "Blind V2-F",
    "Blind V2-G",
    "Blind V2-H",
)

DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"

LOCK_PATH = (
    PROJECTS_DIR
    / "openstar.tess-blind-published-v3.lock.json"
)

MANIFEST_PATH = (
    PROJECTS_DIR
    / "openstar.tess-blind-published-v3.json"
)

VSX_CATALOG = "B/vsx/vsx"

# This is a SERVER-SIDE eligibility constraint.
#
# The selector does not request the Period column itself.
#
# The range exactly mirrors the frozen OpenStar frequency range:
#
#   0.1 cycles/day -> 10 days
#   5.0 cycles/day -> 0.2 days
#
# VizieR supports interval constraints in min..max form.
VSX_PERIOD_FILTER = "0.2..10"

DISCOVERY_RANDOM_SEED = 20260810

# Fixed sky fields so the candidate search is reproducible and
# does not depend on manually choosing famous variable stars.
DISCOVERY_FIELDS = (
    (30.0, -60.0),
    (90.0, -60.0),
    (150.0, -60.0),
    (210.0, -60.0),
    (270.0, -60.0),
    (330.0, -60.0),
    (30.0, -20.0),
    (90.0, -20.0),
    (150.0, -20.0),
    (210.0, -20.0),
    (270.0, -20.0),
    (330.0, -20.0),
    (30.0, 20.0),
    (90.0, 20.0),
    (150.0, 20.0),
    (210.0, 20.0),
    (270.0, 20.0),
    (330.0, 20.0),
    (30.0, 60.0),
    (90.0, 60.0),
    (150.0, 60.0),
    (210.0, 60.0),
    (270.0, 60.0),
    (330.0, 60.0),
)

DISCOVERY_RADIUS_DEG = 5.0
MAX_VSX_ROWS_PER_FIELD = 80
MAX_PREFLIGHT_CANDIDATES = 1200

# VSX coordinates are commonly J2000 while TIC/Gaia positions can
# reflect a later epoch. A tiny 2-arcsec hard cutoff rejects valid
# higher-proper-motion matches. Search wider, but only accept a
# clearly nearest TIC source.
TIC_MATCH_RADIUS_ARCSEC = 12.0
MAX_TIC_MATCH_SEPARATION_ARCSEC = 8.0
MIN_TIC_MATCH_MARGIN_ARCSEC = 1.0
MIN_TIC_MATCH_RATIO = 1.8

MIN_TMAG = 7.0
MAX_TMAG = 13.5

MIN_FINITE_SAMPLES = 8_000
MIN_BASELINE_DAYS = 20.0

PREFERRED_AUTHOR = "SPOC"
FALLBACK_AUTHOR = "TESS-SPOC"
PREFERRED_EXPTIME_SECONDS = 120

# Frozen OpenStar science workload.
MAX_SAMPLES = 18_000
MINIMUM_FREQUENCY = 0.10
MAXIMUM_FREQUENCY = 5.00
TOTAL_FREQUENCIES = 4_194_304
FREQUENCIES_PER_WORK_UNIT = 4_096


# ============================================================
# Generic helpers
# ============================================================


def frequency_step() -> float:
    return (
        MAXIMUM_FREQUENCY
        - MINIMUM_FREQUENCY
    ) / TOTAL_FREQUENCIES


def expected_work_unit_count() -> int:
    return math.ceil(
        TOTAL_FREQUENCIES
        / FREQUENCIES_PER_WORK_UNIT
    )


def canonical_payload(
    lock_document: dict,
) -> dict:
    payload = dict(
        lock_document
    )

    payload.pop(
        "lockSHA256",
        None,
    )

    return payload


def lock_sha256(
    lock_document: dict,
) -> str:
    encoded = json.dumps(
        canonical_payload(
            lock_document
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def write_json(
    path: Path,
    value: dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
            allow_nan=False,
        )


def python_value(
    value,
):
    if value is None:
        return None

    if np.ma.is_masked(
        value
    ):
        return None

    if isinstance(
        value,
        bytes,
    ):
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
        isinstance(
            value,
            float,
        )
        and not math.isfinite(
            value
        )
    ):
        return None

    return value


def float_or_none(
    value,
):
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


def int_or_none(
    value,
):
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


def evenly_spaced_indices(
    source_count: int,
    selected_count: int,
) -> np.ndarray:
    if (
        selected_count
        >= source_count
    ):
        return np.arange(
            source_count,
            dtype=np.int64,
        )

    return np.linspace(
        0,
        source_count - 1,
        selected_count,
        dtype=np.int64,
    )


def deterministic_candidate_key(
    candidate: dict,
) -> str:
    payload = (
        f"{DISCOVERY_RANDOM_SEED}|"
        f"{candidate['ra']:.10f}|"
        f"{candidate['dec']:.10f}"
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# VSX discovery
# ============================================================


def vsx_coordinate_from_row(
    row,
) -> SkyCoord:
    ra_value = row_value(
        row,
        (
            "RAJ2000",
            "RA_ICRS",
        ),
    )

    dec_value = row_value(
        row,
        (
            "DEJ2000",
            "DE_ICRS",
        ),
    )

    if (
        ra_value is None
        or dec_value is None
    ):
        raise ValueError(
            "VSX row has no usable coordinates."
        )

    # VizieR may expose these as numeric degrees or formatted
    # sexagesimal strings depending on table metadata/version.
    try:
        ra_float = float(
            ra_value
        )
        dec_float = float(
            dec_value
        )

        return SkyCoord(
            ra_float,
            dec_float,
            unit="deg",
            frame="icrs",
        )

    except (
        TypeError,
        ValueError,
    ):
        return SkyCoord(
            str(
                ra_value
            ),
            str(
                dec_value
            ),
            unit=(
                u.hourangle,
                u.deg,
            ),
            frame="icrs",
        )


def query_vsx_field(
    ra_deg: float,
    dec_deg: float,
):
    # CRITICAL BLINDNESS PROPERTY:
    #
    # Period is used only as a server-side eligibility filter.
    # It is deliberately omitted from requested columns.
    #
    # Also deliberately omitted:
    #   Name
    #   Type
    #   Epoch
    #
    # We only ask the server for coordinates and an opaque VSX
    # object ID for de-duplication.
    vizier = Vizier(
        columns=[
            "OID",
            "RAJ2000",
            "DEJ2000",
        ],
        column_filters={
            "Period": (
                VSX_PERIOD_FILTER
            ),
        },
        row_limit=(
            MAX_VSX_ROWS_PER_FIELD
        ),
    )

    coordinate = SkyCoord(
        ra_deg,
        dec_deg,
        unit="deg",
        frame="icrs",
    )

    result = vizier.query_region(
        coordinate,
        radius=(
            DISCOVERY_RADIUS_DEG
            * u.deg
        ),
        catalog=VSX_CATALOG,
    )

    if len(
        result
    ) == 0:
        return None

    table = result[0]

    # Defense in depth: even if a future astroquery/VizieR
    # behavior unexpectedly returns Period, abort rather than
    # allowing the selector to receive answer-key values.
    forbidden_columns = {
        "Period",
        "period",
        "Name",
        "name",
        "Type",
        "type",
    }

    leaked = (
        forbidden_columns
        .intersection(
            set(
                table.colnames
            )
        )
    )

    if leaked:
        raise RuntimeError(
            "Blindness guard triggered: VSX returned "
            "forbidden answer-key columns: "
            + ", ".join(
                sorted(
                    leaked
                )
            )
        )

    return table


def build_vsx_candidate_pool() -> list[dict]:
    print()
    print(
        "📚 Discovering hidden-answer VSX targets"
    )
    print(
        "   server-side published-period filter: "
        f"{VSX_PERIOD_FILTER} days"
    )
    print(
        "   Period column requested: NO"
    )
    print(
        "   Name/Type columns requested: NO"
    )
    print(
        f"   fixed sky fields: "
        f"{len(DISCOVERY_FIELDS)}"
    )

    candidates_by_identity = {}

    for field_index, (
        ra_deg,
        dec_deg,
    ) in enumerate(
        DISCOVERY_FIELDS,
        start=1,
    ):
        print(
            "   field "
            f"{field_index}/"
            f"{len(DISCOVERY_FIELDS)}: "
            f"RA={ra_deg:.1f}, "
            f"Dec={dec_deg:.1f}"
        )

        try:
            table = query_vsx_field(
                ra_deg,
                dec_deg,
            )
        except Exception as error:
            print(
                "      VSX query skipped: "
                f"{type(error).__name__}: "
                f"{error}"
            )
            continue

        if (
            table is None
            or len(table) == 0
        ):
            continue

        for row in table:
            try:
                coordinate = (
                    vsx_coordinate_from_row(
                        row
                    )
                )
            except Exception:
                continue

            oid = int_or_none(
                row_value(
                    row,
                    ("OID",),
                )
            )

            if oid is not None:
                identity = (
                    f"oid:{oid}"
                )
            else:
                identity = (
                    "coord:"
                    f"{coordinate.ra.deg:.8f}:"
                    f"{coordinate.dec.deg:.8f}"
                )

            candidates_by_identity.setdefault(
                identity,
                {
                    "vsxOpaqueID": oid,
                    "ra": float(
                        coordinate.ra.deg
                    ),
                    "dec": float(
                        coordinate.dec.deg
                    ),
                },
            )

    candidates = list(
        candidates_by_identity.values()
    )

    candidates.sort(
        key=deterministic_candidate_key
    )

    print()
    print(
        "   hidden-period VSX candidates: "
        f"{len(candidates)}"
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "VSX returned no eligible candidates. "
            "If this happens, inspect whether the VizieR "
            "Period interval-filter syntax has changed."
        )

    return candidates[
        :MAX_PREFLIGHT_CANDIDATES
    ]


# ============================================================
# TIC cross-match
# ============================================================


def nearest_tic_match(
    candidate: dict,
):
    coordinate = SkyCoord(
        candidate["ra"],
        candidate["dec"],
        unit="deg",
        frame="icrs",
    )

    table = Catalogs.query_region(
        coordinate,
        radius=(
            TIC_MATCH_RADIUS_ARCSEC
            * u.arcsec
        ),
        catalog="TIC",
    )

    matches = []

    for row in table:
        tic_id = int_or_none(
            row_value(
                row,
                ("ID",),
            )
        )

        ra = float_or_none(
            row_value(
                row,
                ("ra",),
            )
        )

        dec = float_or_none(
            row_value(
                row,
                ("dec",),
            )
        )

        tmag = float_or_none(
            row_value(
                row,
                ("Tmag",),
            )
        )

        if (
            tic_id is None
            or ra is None
            or dec is None
            or tmag is None
        ):
            continue

        if not (
            MIN_TMAG
            <= tmag
            <= MAX_TMAG
        ):
            continue

        tic_coordinate = SkyCoord(
            ra,
            dec,
            unit="deg",
            frame="icrs",
        )

        separation = float(
            coordinate.separation(
                tic_coordinate
            ).arcsec
        )

        matches.append(
            {
                "ticID": tic_id,
                "ra": ra,
                "dec": dec,
                "tmag": tmag,
                "vsxToTicSeparationArcsec": (
                    separation
                ),
            }
        )

    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            item[
                "vsxToTicSeparationArcsec"
            ],
            item[
                "ticID"
            ],
        )
    )

    best = matches[0]

    best_separation = float(
        best[
            "vsxToTicSeparationArcsec"
        ]
    )

    if (
        best_separation
        > MAX_TIC_MATCH_SEPARATION_ARCSEC
    ):
        return None

    # Wider radius is only for epoch/proper-motion tolerance.
    # We still reject ambiguous positional matches.
    if len(matches) > 1:
        second_separation = float(
            matches[1][
                "vsxToTicSeparationArcsec"
            ]
        )

        separation_margin = (
            second_separation
            - best_separation
        )

        separation_ratio = (
            second_separation
            / max(
                best_separation,
                0.001,
            )
        )

        if (
            separation_margin
            < MIN_TIC_MATCH_MARGIN_ARCSEC
            and separation_ratio
            < MIN_TIC_MATCH_RATIO
        ):
            return None

    return best


# ============================================================
# TESS product selection
# ============================================================


def sector_from_search_row(
    search_result,
    index: int,
):
    table = getattr(
        search_result,
        "table",
        None,
    )

    if table is not None:
        colnames = set(
            getattr(
                table,
                "colnames",
                [],
            )
        )

        if "sequence_number" in colnames:
            try:
                value = int(
                    table[
                        "sequence_number"
                    ][index]
                )

                if value > 0:
                    return value
            except (
                TypeError,
                ValueError,
            ):
                pass

        if "mission" in colnames:
            text = str(
                table[
                    "mission"
                ][index]
            )

            match = re.search(
                r"sector\s*0*(\d+)",
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return int(
                    match.group(1)
                )

    try:
        text = str(
            search_result.mission[
                index
            ]
        )
    except Exception:
        return None

    match = re.search(
        r"sector\s*0*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def exptime_seconds(
    search_result,
    index: int,
) -> float:
    value = (
        search_result.exptime[
            index
        ]
    )

    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        if hasattr(
            value,
            "value",
        ):
            return float(
                value.value
            )

        return math.inf


def choose_search_row(
    search_result,
):
    if len(
        search_result
    ) == 0:
        return None

    ranked = []

    for index in range(
        len(
            search_result
        )
    ):
        sector = (
            sector_from_search_row(
                search_result,
                index,
            )
        )

        cadence = (
            exptime_seconds(
                search_result,
                index,
            )
        )

        # Earliest supported sector first, then shortest cadence.
        ranked.append(
            (
                sector
                if sector is not None
                else 10**9,
                cadence,
                index,
            )
        )

    _, _, selected_index = min(
        ranked
    )

    return {
        "searchResult": (
            search_result[
                selected_index:
                selected_index + 1
            ]
        ),
        "sector": (
            sector_from_search_row(
                search_result,
                selected_index,
            )
        ),
        "cadenceSeconds": (
            exptime_seconds(
                search_result,
                selected_index,
            )
        ),
    }


def search_candidate_product(
    tic_id: int,
):
    query = (
        f"TIC {tic_id}"
    )

    preferred = (
        lk.search_lightcurve(
            query,
            mission="TESS",
            author=(
                PREFERRED_AUTHOR
            ),
            exptime=(
                PREFERRED_EXPTIME_SECONDS
            ),
        )
    )

    selected = (
        choose_search_row(
            preferred
        )
    )

    if selected is not None:
        selected[
            "author"
        ] = PREFERRED_AUTHOR

        return selected

    fallback = (
        lk.search_lightcurve(
            query,
            mission="TESS",
            author=(
                FALLBACK_AUTHOR
            ),
        )
    )

    selected = (
        choose_search_row(
            fallback
        )
    )

    if selected is not None:
        selected[
            "author"
        ] = FALLBACK_AUTHOR

        return selected

    spoc_any = (
        lk.search_lightcurve(
            query,
            mission="TESS",
            author=(
                PREFERRED_AUTHOR
            ),
        )
    )

    selected = (
        choose_search_row(
            spoc_any
        )
    )

    if selected is not None:
        selected[
            "author"
        ] = PREFERRED_AUTHOR

        return selected

    return None


def download_product(
    selection: dict,
):
    return (
        selection[
            "searchResult"
        ]
        .download(
            quality_bitmask="default"
        )
    )


def finite_light_curve_arrays(
    light_curve,
):
    times = np.asarray(
        light_curve.time.value,
        dtype=np.float64,
    )

    flux = np.asarray(
        light_curve.flux.value,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(
            times
        )
        & np.isfinite(
            flux
        )
    )

    times = times[
        finite
    ]

    flux = flux[
        finite
    ]

    order = np.argsort(
        times
    )

    return (
        times[
            order
        ],
        flux[
            order
        ],
    )


# ============================================================
# Blind preflight
# ============================================================


def evaluate_candidates(
    vsx_candidates: list[dict],
):
    print()
    print(
        "🧪 TESS data-quality preflight"
    )
    print(
        "   periodogram computed for selection: NO"
    )
    print(
        "   published period read by selector: NO"
    )
    print(
        "   max candidates to inspect: "
        f"{len(vsx_candidates)}"
    )
    print(
        "   TIC search radius: "
        f"{TIC_MATCH_RADIUS_ARCSEC:.1f} arcsec"
    )
    print(
        "   max accepted nearest separation: "
        f"{MAX_TIC_MATCH_SEPARATION_ARCSEC:.1f} arcsec"
    )
    print(
        "   ambiguous nearest matches: REJECTED"
    )

    accepted = []
    seen_tic_ids = set()

    for index, candidate in enumerate(
        vsx_candidates,
        start=1,
    ):
        print()
        print(
            f"   candidate "
            f"{index}/"
            f"{len(vsx_candidates)}"
        )

        try:
            tic = nearest_tic_match(
                candidate
            )

            if tic is None:
                print(
                    "      skip: no sufficiently close "
                    "TIC match"
                )
                continue

            tic_id = int(
                tic[
                    "ticID"
                ]
            )

            if tic_id in seen_tic_ids:
                print(
                    "      skip: duplicate TIC"
                )
                continue

            selection = (
                search_candidate_product(
                    tic_id
                )
            )

            if selection is None:
                print(
                    "      skip: no supported "
                    "TESS light curve"
                )
                continue

            if (
                selection[
                    "sector"
                ]
                is None
            ):
                print(
                    "      skip: sector unavailable"
                )
                continue

            light_curve = (
                download_product(
                    selection
                )
            )

            if light_curve is None:
                print(
                    "      skip: TESS download failed"
                )
                continue

            times64, flux64 = (
                finite_light_curve_arrays(
                    light_curve
                )
            )

            if (
                len(
                    times64
                )
                < MIN_FINITE_SAMPLES
            ):
                print(
                    "      skip: only "
                    f"{len(times64)} "
                    "finite samples"
                )
                continue

            baseline = float(
                times64[-1]
                - times64[0]
            )

            if (
                baseline
                < MIN_BASELINE_DAYS
            ):
                print(
                    "      skip: baseline only "
                    f"{baseline:.3f} days"
                )
                continue

            accepted_candidate = {
                "ticID": tic_id,
                "ra": float(
                    tic[
                        "ra"
                    ]
                ),
                "dec": float(
                    tic[
                        "dec"
                    ]
                ),
                "tmag": float(
                    tic[
                        "tmag"
                    ]
                ),
                "vsxToTicSeparationArcsec": (
                    float(
                        tic[
                            "vsxToTicSeparationArcsec"
                        ]
                    )
                ),
                "sector": int(
                    selection[
                        "sector"
                    ]
                ),
                "author": str(
                    selection[
                        "author"
                    ]
                ),
                "cadenceSeconds": float(
                    selection[
                        "cadenceSeconds"
                    ]
                ),
                "finiteSamples": int(
                    len(
                        times64
                    )
                ),
                "baselineDays": baseline,
                "answerKeyEligibility": {
                    "catalog": (
                        "AAVSO VSX"
                    ),
                    "catalogID": (
                        VSX_CATALOG
                    ),
                    "serverSidePeriodFilterDays": (
                        VSX_PERIOD_FILTER
                    ),
                    "periodValueRetrieved": False,
                },
            }

            accepted.append(
                accepted_candidate
            )

            seen_tic_ids.add(
                tic_id
            )

            print(
                "      accepted"
            )
            print(
                f"      TIC: {tic_id}"
            )
            print(
                "      VSX↔TIC separation: "
                f"{accepted_candidate['vsxToTicSeparationArcsec']:.3f}\""
            )
            print(
                "      TESS sector: "
                f"{accepted_candidate['sector']}"
            )
            print(
                "      cadence: "
                f"{accepted_candidate['cadenceSeconds']:.0f}s"
            )
            print(
                "      finite samples: "
                f"{accepted_candidate['finiteSamples']}"
            )
            print(
                "      baseline: "
                f"{baseline:.3f} days"
            )
            print(
                "      published period value: [HIDDEN]"
            )

            if (
                len(
                    accepted
                )
                >= TARGET_COUNT
            ):
                break

        except Exception as error:
            print(
                "      skip: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    if (
        len(
            accepted
        )
        < TARGET_COUNT
    ):
        raise RuntimeError(
            "Only "
            f"{len(accepted)} "
            "eligible hidden-answer targets survived "
            "preflight; eight are required after scanning "
            f"{len(vsx_candidates)} candidates. "
            "Increase the discovery pool rather than changing "
            "the hidden-answer or TESS-quality rules."
        )

    return accepted[
        :TARGET_COUNT
    ]


# ============================================================
# Lock manifest
# ============================================================


def build_lock_document(
    selected: list[dict],
):
    targets = []

    for blind_name, item in zip(
        BLIND_NAMES,
        selected,
    ):
        targets.append(
            {
                "blindName": (
                    blind_name
                ),
                "ticID": int(
                    item[
                        "ticID"
                    ]
                ),
                "ra": float(
                    item[
                        "ra"
                    ]
                ),
                "dec": float(
                    item[
                        "dec"
                    ]
                ),
                "tmag": float(
                    item[
                        "tmag"
                    ]
                ),
                "vsxToTicSeparationArcsec": (
                    float(
                        item[
                            "vsxToTicSeparationArcsec"
                        ]
                    )
                ),
                "sector": int(
                    item[
                        "sector"
                    ]
                ),
                "author": str(
                    item[
                        "author"
                    ]
                ),
                "cadenceSeconds": float(
                    item[
                        "cadenceSeconds"
                    ]
                ),
                "preflightFiniteSamples": int(
                    item[
                        "finiteSamples"
                    ]
                ),
                "preflightBaselineDays": float(
                    item[
                        "baselineDays"
                    ]
                ),
                "answerKeyEligibility": (
                    item[
                        "answerKeyEligibility"
                    ]
                ),
            }
        )

    document = {
        "schemaVersion": 3,
        "projectID": (
            PROJECT_ID
        ),
        "purpose": (
            "Preregistered blind validation against "
            "independently published AAVSO VSX periods"
        ),
        "selectionPolicy": {
            "targetCount": (
                TARGET_COUNT
            ),
            "discoveryRandomSeed": (
                DISCOVERY_RANDOM_SEED
            ),
            "sourceCatalog": (
                VSX_CATALOG
            ),
            "publishedPeriodEligibility": (
                "server-side VSX Period constraint only"
            ),
            "publishedPeriodFilterDays": (
                VSX_PERIOD_FILTER
            ),
            "actualPublishedPeriodRequested": False,
            "actualPublishedPeriodStored": False,
            "vsxNameRequested": False,
            "vsxCoordinateEpochTolerance": {
                "ticSearchRadiusArcsec": TIC_MATCH_RADIUS_ARCSEC,
                "maximumAcceptedNearestSeparationArcsec": MAX_TIC_MATCH_SEPARATION_ARCSEC,
                "minimumNearestMatchMarginArcsec": MIN_TIC_MATCH_MARGIN_ARCSEC,
                "minimumNearestMatchRatio": MIN_TIC_MATCH_RATIO,
            },
            "vsxTypeRequested": False,
            "tessPeriodogramUsedForSelection": False,
            "selectionInputs": [
                "VSX period existence/range eligibility",
                "VSX sky coordinates",
                "TIC positional cross-match",
                "T magnitude",
                "TESS light-curve availability",
                "TESS finite sample count",
                "TESS baseline",
                "TESS cadence/product metadata",
            ],
            "forbiddenSelectionInputs": [
                "actual VSX period value",
                "actual VSX frequency",
                "VSX variable name",
                "VSX variability type",
                "TESS Lomb-Scargle winner",
                "TESS periodicity strength",
                "Astropy final period",
            ],
        },
        "analysisPolicy": {
            "mission": "TESS",
            "maximumDistributedSamples": (
                MAX_SAMPLES
            ),
            "numericRepresentation": (
                "Float32"
            ),
            "timeOrigin": (
                "relative-to-first-distributed-sample"
            ),
            "fluxNormalization": (
                "mean/stddev in Float64 before "
                "Float32 conversion"
            ),
            "sampleSelection": (
                "evenly spaced across finite rows "
                "when source exceeds sample cap"
            ),
            "minimumFrequency": (
                MINIMUM_FREQUENCY
            ),
            "maximumFrequency": (
                MAXIMUM_FREQUENCY
            ),
            "totalFrequencies": (
                TOTAL_FREQUENCIES
            ),
            "frequenciesPerWorkUnit": (
                FREQUENCIES_PER_WORK_UNIT
            ),
            "workUnitsPerTarget": (
                expected_work_unit_count()
            ),
            "reference": (
                "Astropy LombScargle from exact "
                "distributed Float32 samples"
            ),
        },
        "revealPolicy": {
            "revealOnlyAfterOpenStarCompletion": True,
            "revealCatalog": (
                "AAVSO VSX"
            ),
            "revealField": (
                "Period"
            ),
        },
        "targets": targets,
    }

    document[
        "lockSHA256"
    ] = lock_sha256(
        document
    )

    return document


def select_and_lock():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROJECTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if LOCK_PATH.exists():
        raise RuntimeError(
            "Blind lock already exists:\n"
            f"{LOCK_PATH}\n"
            "Refusing to silently replace a preregistered "
            "target set."
        )

    print()
    print(
        "⭐ OpenStar Blind Published-Period Validation v3"
    )
    print(
        "🔒 SELECT + LOCK"
    )
    print()
    print(
        "The selector is allowed to know only that a "
        "published period exists inside the frozen search range."
    )
    print(
        "The actual period value is never requested."
    )

    vsx_candidates = (
        build_vsx_candidate_pool()
    )

    selected = (
        evaluate_candidates(
            vsx_candidates
        )
    )

    lock_document = (
        build_lock_document(
            selected
        )
    )

    write_json(
        LOCK_PATH,
        lock_document,
    )

    print()
    print(
        "🔒 BLIND PUBLISHED-PERIOD SET LOCKED"
    )
    print(
        f"   file: {LOCK_PATH}"
    )
    print(
        "   SHA-256: "
        f"{lock_document['lockSHA256']}"
    )
    print(
        f"   targets: "
        f"{len(selected)}"
    )

    for target in lock_document[
        "targets"
    ]:
        print(
            "   "
            f"{target['blindName']}: "
            f"TIC {target['ticID']} | "
            f"Sector {target['sector']} | "
            f"{target['cadenceSeconds']:.0f}s | "
            f"Tmag {target['tmag']:.2f} | "
            "published period [HIDDEN]"
        )

    print()
    print(
        "✅ Eight external answer keys are guaranteed "
        "to exist, but their values remain hidden."
    )
    print(
        "Next: run this file with --prepare."
    )


# ============================================================
# Locked preparation
# ============================================================


def load_and_verify_lock():
    if not LOCK_PATH.exists():
        raise RuntimeError(
            "Missing lock file:\n"
            f"{LOCK_PATH}\n"
            "Run --select first."
        )

    with LOCK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(
            file
        )

    stored_hash = (
        document.get(
            "lockSHA256"
        )
    )

    calculated_hash = (
        lock_sha256(
            document
        )
    )

    if (
        stored_hash
        != calculated_hash
    ):
        raise RuntimeError(
            "Blind lock hash mismatch.\n"
            f"stored:     {stored_hash}\n"
            f"calculated: {calculated_hash}"
        )

    if (
        document.get(
            "projectID"
        )
        != PROJECT_ID
    ):
        raise RuntimeError(
            "Lock project ID mismatch."
        )

    if (
        len(
            document.get(
                "targets",
                [],
            )
        )
        != TARGET_COUNT
    ):
        raise RuntimeError(
            "Locked target count mismatch."
        )

    return document


def search_locked_product(
    target: dict,
):
    tic_id = int(
        target[
            "ticID"
        ]
    )

    sector = int(
        target[
            "sector"
        ]
    )

    author = str(
        target[
            "author"
        ]
    )

    cadence_seconds = float(
        target[
            "cadenceSeconds"
        ]
    )

    query = (
        f"TIC {tic_id}"
    )

    result = (
        lk.search_lightcurve(
            query,
            mission="TESS",
            author=author,
            sector=sector,
            exptime=int(
                round(
                    cadence_seconds
                )
            ),
        )
    )

    selected = (
        choose_search_row(
            result
        )
    )

    if selected is not None:
        selected[
            "author"
        ] = author

        return selected

    result = (
        lk.search_lightcurve(
            query,
            mission="TESS",
            author=author,
            sector=sector,
        )
    )

    selected = (
        choose_search_row(
            result
        )
    )

    if selected is not None:
        selected[
            "author"
        ] = author

    return selected


def prepare_distributed_samples(
    times64: np.ndarray,
    flux64: np.ndarray,
):
    source_count = int(
        len(
            times64
        )
    )

    selected_count = min(
        source_count,
        MAX_SAMPLES,
    )

    indices = (
        evenly_spaced_indices(
            source_count,
            selected_count,
        )
    )

    selected_times64 = (
        times64[
            indices
        ]
    )

    selected_flux64 = (
        flux64[
            indices
        ]
    )

    flux_mean = float(
        np.mean(
            selected_flux64
        )
    )

    flux_stddev = float(
        np.std(
            selected_flux64
        )
    )

    if (
        not math.isfinite(
            flux_stddev
        )
        or flux_stddev <= 0
    ):
        raise RuntimeError(
            "Invalid flux standard deviation."
        )

    normalized_flux64 = (
        selected_flux64
        - flux_mean
    ) / flux_stddev

    time_origin_days = float(
        selected_times64[
            0
        ]
    )

    relative_times64 = (
        selected_times64
        - time_origin_days
    )

    times = np.asarray(
        relative_times64,
        dtype=np.float32,
    )

    flux = np.asarray(
        normalized_flux64,
        dtype=np.float32,
    )

    times[
        0
    ] = np.float32(
        0.0
    )

    if not np.all(
        np.isfinite(
            times
        )
    ):
        raise RuntimeError(
            "Float32 time conversion produced "
            "non-finite values."
        )

    if not np.all(
        np.isfinite(
            flux
        )
    ):
        raise RuntimeError(
            "Float32 flux conversion produced "
            "non-finite values."
        )

    return (
        times,
        flux,
        {
            "originalSamples": (
                source_count
            ),
            "distributedSamples": int(
                len(
                    times
                )
            ),
            "originalTimeOriginDays": (
                time_origin_days
            ),
            "sourceFluxMean": (
                flux_mean
            ),
            "sourceFluxStddev": (
                flux_stddev
            ),
            "baselineDays": (
                float(
                    times[-1]
                    - times[0]
                )
                if len(
                    times
                ) > 1
                else 0.0
            ),
        },
    )


def calculate_astropy_reference(
    times: np.ndarray,
    flux: np.ndarray,
    blind_name: str,
):
    astropy_times = (
        np.asarray(
            times,
            dtype=np.float32,
        )
        .astype(
            np.float64
        )
    )

    astropy_flux = (
        np.asarray(
            flux,
            dtype=np.float32,
        )
        .astype(
            np.float64
        )
    )

    frequencies = (
        MINIMUM_FREQUENCY
        + np.arange(
            TOTAL_FREQUENCIES,
            dtype=np.float64,
        )
        * frequency_step()
    )

    print()
    print(
        "🧪 Building Astropy references for "
        f"{blind_name}"
    )
    print(
        "   exact distributed Float32 samples"
    )
    print(
        "   external published period: [HIDDEN]"
    )
    print(
        f"   frequencies: "
        f"{TOTAL_FREQUENCIES:,}"
    )

    powers = LombScargle(
        astropy_times,
        astropy_flux,
    ).power(
        frequencies
    )

    global_index = int(
        np.nanargmax(
            powers
        )
    )

    best_frequency = float(
        frequencies[
            global_index
        ]
    )

    best_power = float(
        powers[
            global_index
        ]
    )

    chunks = []

    for start_index in range(
        0,
        TOTAL_FREQUENCIES,
        FREQUENCIES_PER_WORK_UNIT,
    ):
        end_index = min(
            start_index
            + FREQUENCIES_PER_WORK_UNIT,
            TOTAL_FREQUENCIES,
        )

        chunk = powers[
            start_index:
            end_index
        ]

        local_index = int(
            np.nanargmax(
                chunk
            )
        )

        absolute_index = (
            start_index
            + local_index
        )

        chunk_frequency = float(
            frequencies[
                absolute_index
            ]
        )

        chunk_power = float(
            powers[
                absolute_index
            ]
        )

        chunks.append(
            {
                "frequencyStartIndex": (
                    start_index
                ),
                "frequencyCount": (
                    end_index
                    - start_index
                ),
                "bestFrequency": (
                    chunk_frequency
                ),
                "bestPeriodDays": (
                    1.0
                    / chunk_frequency
                ),
                "bestPower": (
                    chunk_power
                ),
            }
        )

    if (
        len(
            chunks
        )
        != expected_work_unit_count()
    ):
        raise RuntimeError(
            "Astropy chunk-reference count mismatch."
        )

    print(
        "   references ready: "
        f"{len(chunks)}/"
        f"{expected_work_unit_count()}"
    )

    # The Astropy answer is required by the coordinator for
    # scientific verification. The external VSX answer remains
    # absent from the dataset.
    return {
        "bestFrequency": (
            best_frequency
        ),
        "bestPeriodDays": (
            1.0
            / best_frequency
        ),
        "bestPower": (
            best_power
        ),
        "chunks": chunks,
    }


def slugify(
    value: str,
):
    value = (
        value
        .strip()
        .lower()
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    return value.strip(
        "-"
    )


def dataset_id_for_target(
    target: dict,
):
    return (
        f"tess-"
        f"{slugify(target['blindName'])}-"
        f"tic-{int(target['ticID'])}"
    )


def build_dataset(
    target: dict,
    lock_document: dict,
    times: np.ndarray,
    flux: np.ndarray,
    preprocess_metadata: dict,
    reference: dict,
):
    dataset_id = (
        dataset_id_for_target(
            target
        )
    )

    return {
        "id": (
            dataset_id
        ),
        "targetName": (
            target[
                "blindName"
            ]
        ),
        "mission": "TESS",
        "source": {
            "archive": "MAST",
            "author": (
                target[
                    "author"
                ]
            ),
            "ticID": int(
                target[
                    "ticID"
                ]
            ),
            "sector": int(
                target[
                    "sector"
                ]
            ),
            "cadenceSeconds": float(
                target[
                    "cadenceSeconds"
                ]
            ),
            "originalSamples": (
                preprocess_metadata[
                    "originalSamples"
                ]
            ),
            "distributedSamples": (
                preprocess_metadata[
                    "distributedSamples"
                ]
            ),
            "originalTimeOriginDays": (
                preprocess_metadata[
                    "originalTimeOriginDays"
                ]
            ),
            "selectionLockSHA256": (
                lock_document[
                    "lockSHA256"
                ]
            ),
        },
        "science": {
            "role": "blind",
            "externalAnswerKey": (
                "AAVSO VSX Period"
            ),
            "externalAnswerKeyPresent": True,
            "externalAnswerKeyRevealed": False,
        },
        "timeUnit": "days",
        "timeReference": (
            "relative-to-first-distributed-sample"
        ),
        "numericRepresentation": (
            "Float32"
        ),
        "fluxUnit": (
            "normalized"
        ),
        "fluxNormalization": (
            "mean-stddev"
        ),
        "sampleAllocation": (
            "evenly-spaced-across-finite-rows"
        ),
        "times": [
            float(
                value
            )
            for value in times
        ],
        "flux": [
            float(
                value
            )
            for value in flux
        ],
        "frequencySearch": {
            "minimumFrequency": (
                MINIMUM_FREQUENCY
            ),
            "maximumFrequency": (
                MAXIMUM_FREQUENCY
            ),
            "frequencyStep": (
                frequency_step()
            ),
            "totalFrequencies": (
                TOTAL_FREQUENCIES
            ),
            "frequenciesPerWorkUnit": (
                FREQUENCIES_PER_WORK_UNIT
            ),
        },
        "reference": (
            reference
        ),
    }


def prepare_locked_project():
    lock_document = (
        load_and_verify_lock()
    )

    print()
    print(
        "⭐ OpenStar Blind Published-Period Validation v3"
    )
    print(
        "🧪 PREPARE"
    )
    print()
    print(
        "   verified lock SHA-256: "
        f"{lock_document['lockSHA256']}"
    )
    print(
        f"   locked targets: "
        f"{len(lock_document['targets'])}"
    )
    print(
        "   external periods guaranteed: YES"
    )
    print(
        "   external period values available here: NO"
    )
    print(
        "   frequency search: "
        f"{MINIMUM_FREQUENCY:.3f} - "
        f"{MAXIMUM_FREQUENCY:.3f} cycles/day"
    )
    print(
        "   work units per target: "
        f"{expected_work_unit_count()}"
    )

    manifest_datasets = []

    for target_index, target in enumerate(
        lock_document[
            "targets"
        ],
        start=1,
    ):
        print()
        print(
            "════════════════════════════════════════"
        )
        print(
            f"⭐ Target "
            f"{target_index}/"
            f"{TARGET_COUNT}: "
            f"{target['blindName']}"
        )
        print(
            f"   TIC: "
            f"{target['ticID']}"
        )
        print(
            "   published period: [HIDDEN]"
        )
        print(
            f"   sector: "
            f"{target['sector']}"
        )
        print(
            f"   author: "
            f"{target['author']}"
        )
        print(
            f"   cadence: "
            f"{target['cadenceSeconds']:.0f}s"
        )

        selection = (
            search_locked_product(
                target
            )
        )

        if selection is None:
            raise RuntimeError(
                "Locked TESS product could not be "
                f"re-located for TIC "
                f"{target['ticID']}."
            )

        light_curve = (
            download_product(
                selection
            )
        )

        if light_curve is None:
            raise RuntimeError(
                "Locked light-curve download failed "
                f"for TIC "
                f"{target['ticID']}."
            )

        times64, flux64 = (
            finite_light_curve_arrays(
                light_curve
            )
        )

        (
            times,
            flux,
            preprocess_metadata,
        ) = prepare_distributed_samples(
            times64,
            flux64,
        )

        print(
            "   finite source samples: "
            f"{len(times64)}"
        )
        print(
            "   distributed samples: "
            f"{len(times)}"
        )
        print(
            "   baseline: "
            f"{preprocess_metadata['baselineDays']:.4f} days"
        )
        print(
            "   precision: Float32"
        )

        reference = (
            calculate_astropy_reference(
                times,
                flux,
                target[
                    "blindName"
                ],
            )
        )

        dataset = build_dataset(
            target,
            lock_document,
            times,
            flux,
            preprocess_metadata,
            reference,
        )

        dataset_path = (
            DATA_DIR
            / (
                dataset[
                    "id"
                ]
                + ".json"
            )
        )

        write_json(
            dataset_path,
            dataset,
        )

        print(
            f"💾 Dataset saved: "
            f"{dataset_path}"
        )

        manifest_datasets.append(
            {
                "id": (
                    dataset[
                        "id"
                    ]
                ),
                "path": str(
                    dataset_path
                ),
                "targetName": (
                    target[
                        "blindName"
                    ]
                ),
                "ticID": int(
                    target[
                        "ticID"
                    ]
                ),
                "sector": int(
                    target[
                        "sector"
                    ]
                ),
                "author": (
                    target[
                        "author"
                    ]
                ),
                "cadenceSeconds": float(
                    target[
                        "cadenceSeconds"
                    ]
                ),
                "role": "blind",
            }
        )

    manifest = {
        "id": (
            PROJECT_ID
        ),
        "name": (
            PROJECT_NAME
        ),
        "workloadID": (
            WORKLOAD_ID
        ),
        "blindSelectionLockSHA256": (
            lock_document[
                "lockSHA256"
            ]
        ),
        "externalAnswerKey": {
            "catalog": "AAVSO VSX",
            "field": "Period",
            "guaranteedPresent": True,
            "revealed": False,
        },
        "datasets": (
            manifest_datasets
        ),
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    print()
    print(
        "✅ OpenStar hidden-answer project ready"
    )
    print(
        f"   project: "
        f"{PROJECT_ID}"
    )
    print(
        f"   manifest: "
        f"{MANIFEST_PATH}"
    )
    print(
        f"   datasets: "
        f"{len(manifest_datasets)}"
    )
    print(
        "   work units per target: "
        f"{expected_work_unit_count()}"
    )
    print(
        "   total work units: "
        f"{expected_work_unit_count() * len(manifest_datasets)}"
    )
    print(
        "   published periods remain hidden"
    )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a preregistered OpenStar blind "
            "validation set where every target has "
            "a hidden published VSX period."
        )
    )

    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--select",
        action="store_true",
        help=(
            "Select and SHA-256 lock eight targets "
            "without requesting their published periods."
        ),
    )

    mode.add_argument(
        "--prepare",
        action="store_true",
        help=(
            "Verify the existing blind lock and build "
            "all OpenStar datasets/project."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.select:
        select_and_lock()
        return

    prepare_locked_project()


if __name__ == "__main__":
    main()
