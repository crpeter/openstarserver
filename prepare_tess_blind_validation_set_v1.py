import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import lightkurve as lk
import numpy as np
from astroquery.mast import Catalogs
from astropy import units as u
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar multi-target blind validation set
#
# Stage 1:
#   --select
#   Deterministically discovers candidate TIC stars using only:
#     - TIC identity / sky position / T magnitude
#     - TESS light-curve availability
#     - light-curve quality / variability
#     - a coarse TESS-only periodicity-strength score
#
#   It DOES NOT query or store any external published period,
#   classification, or answer-key catalog.
#
#   The selected targets + frozen analysis settings are written
#   to a SHA-256 protected lock manifest.
#
# Stage 2:
#   --prepare
#   Reads that exact lock and creates the OpenStar datasets and
#   multi-target project using the already validated Float32 /
#   Astropy-reference pipeline.
#
# External catalog periods should only be looked up after the
# distributed OpenStar project is complete.
# ============================================================


PROJECT_ID = "openstar.tess-blind-validation-set-v1"
PROJECT_NAME = "OpenStar Multi-Target Blind TESS Validation Set v1"
WORKLOAD_ID = "openstar.tess-period-search.v1"

TARGET_COUNT = 8
BLIND_NAMES = (
    "Blind B",
    "Blind C",
    "Blind D",
    "Blind E",
    "Blind F",
    "Blind G",
    "Blind H",
    "Blind I",
)

DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"

LOCK_PATH = (
    PROJECTS_DIR
    / "openstar.tess-blind-validation-set-v1.lock.json"
)

# Existing Blind A is excluded from candidate selection.
EXCLUDED_TIC_IDS = {
    25165839,
}

# Fixed discovery fields. These are simply deterministic sky
# positions used to obtain a candidate pool from the TIC.
DISCOVERY_FIELDS = (
    (90.0, -66.0),
    (94.0, -66.0),
    (86.0, -66.0),
    (90.0, -62.5),
    (270.0, 66.0),
    (274.0, 66.0),
    (266.0, 66.0),
    (270.0, 62.5),
    (45.0, -45.0),
    (135.0, -45.0),
    (225.0, 45.0),
    (315.0, 45.0),
)

DISCOVERY_RADIUS_DEG = 0.35
DISCOVERY_RANDOM_SEED = 20260810
MAX_CATALOG_CANDIDATES_PER_FIELD = 6
MAX_LIGHT_CURVE_CANDIDATES = 48

MIN_TMAG = 8.0
MAX_TMAG = 12.5
MIN_FINITE_SAMPLES = 4_000

# Selection only. The frequency that wins is deliberately not
# written into the lock or printed.
SELECTION_MIN_FREQUENCY = 0.20
SELECTION_MAX_FREQUENCY = 5.00
SELECTION_FREQUENCIES = 8_192
SELECTION_MAX_SAMPLES = 6_000

# Frozen OpenStar science workload.
MAX_SAMPLES = 18_000
MINIMUM_FREQUENCY = 0.10
MAXIMUM_FREQUENCY = 5.00
TOTAL_FREQUENCIES = 4_194_304
FREQUENCIES_PER_WORK_UNIT = 4_096

PREFERRED_AUTHOR = "SPOC"
FALLBACK_AUTHOR = "TESS-SPOC"
PREFERRED_EXPTIME_SECONDS = 120


# ============================================================
# Generic helpers
# ============================================================


def frequency_step() -> float:
    return (
        MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY
    ) / TOTAL_FREQUENCIES


def expected_work_unit_count() -> int:
    return math.ceil(
        TOTAL_FREQUENCIES
        / FREQUENCIES_PER_WORK_UNIT
    )


def canonical_payload(lock_document: dict) -> dict:
    payload = dict(lock_document)
    payload.pop("lockSHA256", None)
    return payload


def lock_sha256(lock_document: dict) -> str:
    encoded = json.dumps(
        canonical_payload(lock_document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

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


def evenly_spaced_indices(
    source_count: int,
    selected_count: int,
) -> np.ndarray:
    if selected_count >= source_count:
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


def sector_from_search_row(
    search_result,
    index: int,
) -> int | None:
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
                table["mission"][index]
            )

            match = re.search(
                r"sector\s*0*(\d+)",
                text,
                flags=re.IGNORECASE,
            )

            if match is not None:
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
        return float(value)
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
    if len(search_result) == 0:
        return None

    ranked = []

    for index in range(
        len(search_result)
    ):
        sector = sector_from_search_row(
            search_result,
            index,
        )

        cadence = exptime_seconds(
            search_result,
            index,
        )

        ranked.append(
            (
                (
                    sector
                    if sector is not None
                    else 10**9
                ),
                cadence,
                index,
            )
        )

    _, _, selected_index = min(
        ranked
    )

    sector = sector_from_search_row(
        search_result,
        selected_index,
    )

    cadence = exptime_seconds(
        search_result,
        selected_index,
    )

    return {
        "searchResult": search_result[
            selected_index:
            selected_index + 1
        ],
        "sector": sector,
        "cadenceSeconds": cadence,
    }


# ============================================================
# Candidate discovery
# ============================================================


def tic_value(
    row,
    name: str,
    default=None,
):
    try:
        value = row[name]
    except Exception:
        return default

    if np.ma.is_masked(value):
        return default

    return value


def build_catalog_candidate_pool() -> list[dict]:
    print()
    print("🔭 Building deterministic TIC candidate pool")
    print(
        f"   fixed discovery fields: "
        f"{len(DISCOVERY_FIELDS)}"
    )
    print(
        f"   radius per field: "
        f"{DISCOVERY_RADIUS_DEG:.2f} deg"
    )
    print(
        f"   T magnitude range: "
        f"{MIN_TMAG:.1f} - "
        f"{MAX_TMAG:.1f}"
    )
    print(
        "   no period/classification catalog is queried"
    )

    rng = np.random.default_rng(
        DISCOVERY_RANDOM_SEED
    )

    candidate_by_tic = {}

    for field_index, (
        ra,
        dec,
    ) in enumerate(
        DISCOVERY_FIELDS,
        start=1,
    ):
        print(
            f"   field {field_index}/"
            f"{len(DISCOVERY_FIELDS)}: "
            f"RA={ra:.2f}, Dec={dec:.2f}"
        )

        table = Catalogs.query_region(
            f"{ra} {dec}",
            radius=(
                DISCOVERY_RADIUS_DEG
                * u.deg
            ),
            catalog="TIC",
        )

        eligible = []

        for row in table:
            tic_raw = tic_value(
                row,
                "ID",
            )
            tmag_raw = tic_value(
                row,
                "Tmag",
            )
            ra_raw = tic_value(
                row,
                "ra",
            )
            dec_raw = tic_value(
                row,
                "dec",
            )
            object_type = tic_value(
                row,
                "objType",
                "STAR",
            )

            try:
                tic_id = int(
                    tic_raw
                )
                tmag = float(
                    tmag_raw
                )
                candidate_ra = float(
                    ra_raw
                )
                candidate_dec = float(
                    dec_raw
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if tic_id in EXCLUDED_TIC_IDS:
                continue

            if not math.isfinite(
                tmag
            ):
                continue

            if not (
                MIN_TMAG
                <= tmag
                <= MAX_TMAG
            ):
                continue

            if (
                object_type is not None
                and str(
                    object_type
                ).upper()
                not in (
                    "STAR",
                    "ST",
                )
            ):
                continue

            eligible.append(
                {
                    "ticID": tic_id,
                    "ra": candidate_ra,
                    "dec": candidate_dec,
                    "tmag": tmag,
                    "discoveryField": field_index,
                }
            )

        # Randomized with a frozen seed so target selection is
        # deterministic but not "pick the brightest known stars".
        rng.shuffle(
            eligible
        )

        for candidate in eligible[
            :MAX_CATALOG_CANDIDATES_PER_FIELD
        ]:
            candidate_by_tic.setdefault(
                candidate["ticID"],
                candidate,
            )

    candidates = list(
        candidate_by_tic.values()
    )

    rng.shuffle(
        candidates
    )

    candidates = candidates[
        :MAX_LIGHT_CURVE_CANDIDATES
    ]

    print()
    print(
        "   unique TIC candidates queued for "
        f"TESS preflight: {len(candidates)}"
    )

    if len(candidates) < TARGET_COUNT:
        raise RuntimeError(
            "TIC discovery produced fewer candidates "
            "than the requested blind target count."
        )

    return candidates


# ============================================================
# TESS product selection
# ============================================================


def search_candidate_product(
    tic_id: int,
):
    query = f"TIC {tic_id}"

    preferred = lk.search_lightcurve(
        query,
        mission="TESS",
        author=PREFERRED_AUTHOR,
        exptime=PREFERRED_EXPTIME_SECONDS,
    )

    selected = choose_search_row(
        preferred
    )

    if selected is not None:
        selected["author"] = (
            PREFERRED_AUTHOR
        )
        return selected

    fallback = lk.search_lightcurve(
        query,
        mission="TESS",
        author=FALLBACK_AUTHOR,
    )

    selected = choose_search_row(
        fallback
    )

    if selected is not None:
        selected["author"] = (
            FALLBACK_AUTHOR
        )
        return selected

    spoc_any = lk.search_lightcurve(
        query,
        mission="TESS",
        author=PREFERRED_AUTHOR,
    )

    selected = choose_search_row(
        spoc_any
    )

    if selected is not None:
        selected["author"] = (
            PREFERRED_AUTHOR
        )
        return selected

    return None


def download_product(
    selection: dict,
):
    light_curve = (
        selection["searchResult"]
        .download(
            quality_bitmask="default"
        )
    )

    return light_curve


def finite_light_curve_arrays(
    light_curve,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(
        light_curve.time.value,
        dtype=np.float64,
    )
    flux = np.asarray(
        light_curve.flux.value,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(times)
        & np.isfinite(flux)
    )

    times = times[finite]
    flux = flux[finite]

    order = np.argsort(
        times
    )

    return (
        times[order],
        flux[order],
    )


# ============================================================
# Blind selection metrics
# ============================================================


def selection_metrics(
    times64: np.ndarray,
    flux64: np.ndarray,
) -> dict:
    if len(times64) < MIN_FINITE_SAMPLES:
        raise RuntimeError(
            "Insufficient finite samples."
        )

    median_flux = float(
        np.median(
            flux64
        )
    )

    p05 = float(
        np.percentile(
            flux64,
            5,
        )
    )
    p95 = float(
        np.percentile(
            flux64,
            95,
        )
    )

    fractional_range = (
        (p95 - p05)
        / max(
            abs(
                median_flux
            ),
            np.finfo(
                np.float64
            ).eps,
        )
    )

    selected_count = min(
        len(times64),
        SELECTION_MAX_SAMPLES,
    )

    indices = evenly_spaced_indices(
        len(times64),
        selected_count,
    )

    selected_times = (
        times64[indices]
        - times64[indices][0]
    )

    selected_flux = (
        flux64[indices]
    )

    flux_mean = float(
        np.mean(
            selected_flux
        )
    )
    flux_stddev = float(
        np.std(
            selected_flux
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

    selected_flux = (
        selected_flux
        - flux_mean
    ) / flux_stddev

    frequencies = np.linspace(
        SELECTION_MIN_FREQUENCY,
        SELECTION_MAX_FREQUENCY,
        SELECTION_FREQUENCIES,
        dtype=np.float64,
    )

    powers = LombScargle(
        selected_times,
        selected_flux,
    ).power(
        frequencies
    )

    peak_power = float(
        np.nanmax(
            powers
        )
    )

    if not math.isfinite(
        peak_power
    ):
        raise RuntimeError(
            "Coarse periodogram returned no finite maximum."
        )

    baseline_days = float(
        times64[-1]
        - times64[0]
    )

    return {
        "finiteSamples": int(
            len(times64)
        ),
        "baselineDays": baseline_days,
        "robustFractionalRange": fractional_range,
        "coarsePeakPower": peak_power,
    }


def evaluate_candidates(
    candidates: list[dict],
) -> list[dict]:
    print()
    print("🧪 TESS-only blind candidate preflight")
    print(
        "   selection uses periodicity strength, "
        "not the winning period"
    )

    evaluated = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        tic_id = int(
            candidate["ticID"]
        )

        print()
        print(
            f"   candidate {index}/"
            f"{len(candidates)}: "
            f"TIC {tic_id}"
        )

        try:
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
                selection["sector"]
                is None
            ):
                print(
                    "      skip: sector could not "
                    "be determined"
                )
                continue

            light_curve = download_product(
                selection
            )

            if light_curve is None:
                print(
                    "      skip: download returned "
                    "no light curve"
                )
                continue

            times64, flux64 = (
                finite_light_curve_arrays(
                    light_curve
                )
            )

            if (
                len(times64)
                < MIN_FINITE_SAMPLES
            ):
                print(
                    "      skip: only "
                    f"{len(times64)} finite samples"
                )
                continue

            metrics = selection_metrics(
                times64,
                flux64,
            )

            result = dict(
                candidate
            )

            result.update(
                {
                    "sector": int(
                        selection["sector"]
                    ),
                    "author": str(
                        selection["author"]
                    ),
                    "cadenceSeconds": float(
                        selection[
                            "cadenceSeconds"
                        ]
                    ),
                    "selectionMetrics": metrics,
                }
            )

            evaluated.append(
                result
            )

            print(
                "      accepted for ranking"
            )
            print(
                "      sector: "
                f"{result['sector']}"
            )
            print(
                "      cadence: "
                f"{result['cadenceSeconds']:.0f}s"
            )
            print(
                "      finite samples: "
                f"{metrics['finiteSamples']}"
            )
            print(
                "      robust fractional range: "
                f"{metrics['robustFractionalRange']:.6f}"
            )
            print(
                "      coarse periodicity strength: "
                f"{metrics['coarsePeakPower']:.6f}"
            )

        except Exception as error:
            print(
                "      skip: "
                f"{type(error).__name__}: "
                f"{error}"
            )

        if (
            len(evaluated)
            >= TARGET_COUNT * 3
        ):
            # Enough viable targets to make the final ranking
            # meaningful without downloading the entire pool.
            break

    if len(evaluated) < TARGET_COUNT:
        raise RuntimeError(
            "Fewer than eight viable TESS candidates "
            "survived blind preflight. Increase the "
            "candidate pool or discovery radius."
        )

    ranked = sorted(
        evaluated,
        key=lambda item: (
            -float(
                item[
                    "selectionMetrics"
                ][
                    "coarsePeakPower"
                ]
            ),
            -float(
                item[
                    "selectionMetrics"
                ][
                    "robustFractionalRange"
                ]
            ),
            int(
                item["ticID"]
            ),
        ),
    )

    selected = ranked[
        :TARGET_COUNT
    ]

    return selected


# ============================================================
# Lock manifest
# ============================================================


def build_lock_document(
    selected: list[dict],
) -> dict:
    targets = []

    for blind_name, item in zip(
        BLIND_NAMES,
        selected,
    ):
        targets.append(
            {
                "blindName": blind_name,
                "ticID": int(
                    item["ticID"]
                ),
                "ra": float(
                    item["ra"]
                ),
                "dec": float(
                    item["dec"]
                ),
                "tmag": float(
                    item["tmag"]
                ),
                "sector": int(
                    item["sector"]
                ),
                "author": str(
                    item["author"]
                ),
                "cadenceSeconds": float(
                    item[
                        "cadenceSeconds"
                    ]
                ),
                "selectionMetrics": {
                    "finiteSamples": int(
                        item[
                            "selectionMetrics"
                        ][
                            "finiteSamples"
                        ]
                    ),
                    "baselineDays": float(
                        item[
                            "selectionMetrics"
                        ][
                            "baselineDays"
                        ]
                    ),
                    "robustFractionalRange": float(
                        item[
                            "selectionMetrics"
                        ][
                            "robustFractionalRange"
                        ]
                    ),
                    "coarsePeakPower": float(
                        item[
                            "selectionMetrics"
                        ][
                            "coarsePeakPower"
                        ]
                    ),
                },
            }
        )

    lock_document = {
        "schemaVersion": 1,
        "projectID": PROJECT_ID,
        "purpose": (
            "Blind multi-target validation before "
            "external published-period lookup"
        ),
        "selectionPolicy": {
            "targetCount": TARGET_COUNT,
            "discoveryRandomSeed": (
                DISCOVERY_RANDOM_SEED
            ),
            "discoveryFields": [
                {
                    "ra": float(ra),
                    "dec": float(dec),
                }
                for ra, dec
                in DISCOVERY_FIELDS
            ],
            "discoveryRadiusDeg": (
                DISCOVERY_RADIUS_DEG
            ),
            "minimumTmag": MIN_TMAG,
            "maximumTmag": MAX_TMAG,
            "minimumFiniteSamples": (
                MIN_FINITE_SAMPLES
            ),
            "selectionMinimumFrequency": (
                SELECTION_MIN_FREQUENCY
            ),
            "selectionMaximumFrequency": (
                SELECTION_MAX_FREQUENCY
            ),
            "selectionFrequencyCount": (
                SELECTION_FREQUENCIES
            ),
            "ranking": (
                "descending coarsePeakPower, "
                "then descending robustFractionalRange, "
                "then ascending TIC ID"
            ),
            "forbiddenSelectionInputs": [
                "published period",
                "published frequency",
                "published variability classification",
                "answer-key catalog",
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
                "mean/stddev in Float64 "
                "before Float32 conversion"
            ),
            "sampleSelection": (
                "evenly spaced across all finite rows "
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
        "answerKeyPolicy": {
            "externalPeriodLookupBeforeProjectCompletion": False,
            "externalClassificationLookupBeforeProjectCompletion": False,
            "revealAfterOpenStarProjectCompletion": True,
        },
        "targets": targets,
    }

    lock_document[
        "lockSHA256"
    ] = lock_sha256(
        lock_document
    )

    return lock_document


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
            f"Lock already exists: {LOCK_PATH}\n"
            "Refusing to silently replace a preregistered "
            "blind target set. Delete it manually only if "
            "you intentionally want a new experiment."
        )

    print()
    print(
        "⭐ OpenStar Blind Validation Set — SELECT + LOCK"
    )
    print(
        "No external published period/classification "
        "lookup is performed in this stage."
    )

    candidates = (
        build_catalog_candidate_pool()
    )

    selected = evaluate_candidates(
        candidates
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
    print("🔒 BLIND TARGET SET LOCKED")
    print(
        f"   file: {LOCK_PATH}"
    )
    print(
        "   SHA-256: "
        f"{lock_document['lockSHA256']}"
    )
    print(
        f"   targets: {len(selected)}"
    )

    for target in lock_document[
        "targets"
    ]:
        metrics = target[
            "selectionMetrics"
        ]

        print(
            "   "
            f"{target['blindName']}: "
            f"TIC {target['ticID']} | "
            f"Sector {target['sector']} | "
            f"{target['cadenceSeconds']:.0f}s | "
            f"Tmag {target['tmag']:.2f} | "
            "coarse strength "
            f"{metrics['coarsePeakPower']:.4f}"
        )

    print()
    print(
        "✅ Selection is preregistered."
    )
    print(
        "Next command: run this same file with --prepare."
    )


# ============================================================
# Locked-target validation
# ============================================================


def load_and_verify_lock() -> dict:
    if not LOCK_PATH.exists():
        raise RuntimeError(
            f"Missing lock file: {LOCK_PATH}\n"
            "Run --select first."
        )

    with LOCK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        lock_document = json.load(
            file
        )

    stored_hash = (
        lock_document.get(
            "lockSHA256"
        )
    )

    calculated_hash = (
        lock_sha256(
            lock_document
        )
    )

    if stored_hash != calculated_hash:
        raise RuntimeError(
            "Blind lock hash mismatch.\n"
            f"stored:     {stored_hash}\n"
            f"calculated: {calculated_hash}"
        )

    if (
        lock_document.get(
            "projectID"
        )
        != PROJECT_ID
    ):
        raise RuntimeError(
            "Lock project ID does not match this script."
        )

    targets = lock_document.get(
        "targets",
        []
    )

    if len(targets) != TARGET_COUNT:
        raise RuntimeError(
            "Locked target count mismatch."
        )

    return lock_document


def search_locked_product(
    target: dict,
):
    tic_id = int(
        target["ticID"]
    )
    sector = int(
        target["sector"]
    )
    author = str(
        target["author"]
    )
    cadence_seconds = float(
        target["cadenceSeconds"]
    )

    query = f"TIC {tic_id}"

    preferred_exptime = int(
        round(
            cadence_seconds
        )
    )

    result = lk.search_lightcurve(
        query,
        mission="TESS",
        author=author,
        sector=sector,
        exptime=preferred_exptime,
    )

    selected = choose_search_row(
        result
    )

    if selected is not None:
        return selected

    # Some archive products expose a cadence value with tiny
    # representation differences. The fallback is still locked
    # to the same TIC / author / sector.
    result = lk.search_lightcurve(
        query,
        mission="TESS",
        author=author,
        sector=sector,
    )

    return choose_search_row(
        result
    )


def prepare_distributed_samples(
    times64: np.ndarray,
    flux64: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    source_count = len(
        times64
    )

    selected_count = min(
        source_count,
        MAX_SAMPLES,
    )

    indices = evenly_spaced_indices(
        source_count,
        selected_count,
    )

    selected_times64 = (
        times64[indices]
    )
    selected_flux64 = (
        flux64[indices]
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
        selected_times64[0]
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

    times[0] = np.float32(
        0.0
    )

    if not np.all(
        np.isfinite(
            times
        )
    ):
        raise RuntimeError(
            "Float32 time conversion produced non-finite values."
        )

    if not np.all(
        np.isfinite(
            flux
        )
    ):
        raise RuntimeError(
            "Float32 flux conversion produced non-finite values."
        )

    metadata = {
        "originalSamples": int(
            source_count
        ),
        "distributedSamples": int(
            len(times)
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
            if len(times) > 1
            else 0.0
        ),
    }

    return (
        times,
        flux,
        metadata,
    )


def calculate_astropy_reference(
    times: np.ndarray,
    flux: np.ndarray,
    blind_name: str,
) -> dict:
    astropy_times = np.asarray(
        times,
        dtype=np.float32,
    ).astype(
        np.float64
    )

    astropy_flux = np.asarray(
        flux,
        dtype=np.float32,
    ).astype(
        np.float64
    )

    step = frequency_step()

    frequencies = (
        MINIMUM_FREQUENCY
        + np.arange(
            TOTAL_FREQUENCIES,
            dtype=np.float64,
        ) * step
    )

    print()
    print(
        f"🧪 Building Astropy references for {blind_name}"
    )
    print(
        "   exact distributed Float32 samples"
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
        len(chunks)
        != expected_work_unit_count()
    ):
        raise RuntimeError(
            "Astropy chunk-reference count mismatch."
        )

    # Keep the reference inside the dataset for coordinator
    # verification, but do not print the global period here.
    print(
        "   references ready: "
        f"{len(chunks)}/"
        f"{expected_work_unit_count()}"
    )

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


def dataset_id_for_target(
    target: dict,
) -> str:
    blind_suffix = (
        str(
            target["blindName"]
        )
        .lower()
        .replace(
            " ",
            "-",
        )
    )

    return (
        f"tess-{blind_suffix}-"
        f"tic-{int(target['ticID'])}"
    )


def build_dataset(
    target: dict,
    selection: dict,
    times: np.ndarray,
    flux: np.ndarray,
    preprocess_metadata: dict,
    reference: dict,
) -> dict:
    dataset_id = (
        dataset_id_for_target(
            target
        )
    )

    return {
        "id": dataset_id,
        "targetName": (
            target["blindName"]
        ),
        "mission": "TESS",
        "source": {
            "archive": "MAST",
            "author": (
                target["author"]
            ),
            "ticID": int(
                target["ticID"]
            ),
            "sector": int(
                target["sector"]
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
                selection[
                    "lockSHA256"
                ]
            ),
        },
        "science": {
            "role": "blind",
        },
        "timeUnit": "days",
        "timeReference": (
            "relative-to-first-distributed-sample"
        ),
        "numericRepresentation": (
            "Float32"
        ),
        "fluxUnit": "normalized",
        "fluxNormalization": (
            "mean-stddev"
        ),
        "sampleAllocation": (
            "evenly-spaced-across-finite-rows"
        ),
        "times": [
            float(value)
            for value in times
        ],
        "flux": [
            float(value)
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
        "reference": reference,
    }


def prepare_locked_project():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROJECTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_document = (
        load_and_verify_lock()
    )

    print()
    print(
        "⭐ OpenStar Blind Validation Set — PREPARE"
    )
    print(
        "   verified lock SHA-256: "
        f"{lock_document['lockSHA256']}"
    )
    print(
        f"   locked targets: "
        f"{len(lock_document['targets'])}"
    )
    print(
        "   external answer key remains absent"
    )
    print(
        "   frequency search: "
        f"{MINIMUM_FREQUENCY:.3f} - "
        f"{MAXIMUM_FREQUENCY:.3f} cycles/day"
    )
    print(
        f"   work units per target: "
        f"{expected_work_unit_count()}"
    )

    manifest_datasets = []

    for target_index, target in enumerate(
        lock_document["targets"],
        start=1,
    ):
        print()
        print(
            "════════════════════════════════════════"
        )
        print(
            f"⭐ Target {target_index}/"
            f"{TARGET_COUNT}: "
            f"{target['blindName']}"
        )
        print(
            f"   TIC: {target['ticID']}"
        )
        print(
            f"   locked sector: "
            f"{target['sector']}"
        )
        print(
            f"   locked author: "
            f"{target['author']}"
        )
        print(
            f"   locked cadence: "
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
                f"re-located for TIC {target['ticID']}."
            )

        light_curve = download_product(
            selection
        )

        if light_curve is None:
            raise RuntimeError(
                "Locked light curve download failed "
                f"for TIC {target['ticID']}."
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
            "   distributed precision: Float32"
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

        dataset_id = (
            dataset["id"]
        )

        dataset_path = (
            DATA_DIR
            / f"{dataset_id}.json"
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
                "id": dataset_id,
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
        "id": PROJECT_ID,
        "name": PROJECT_NAME,
        "workloadID": WORKLOAD_ID,
        "blindSelectionLockSHA256": (
            lock_document[
                "lockSHA256"
            ]
        ),
        "datasets": (
            manifest_datasets
        ),
    }

    manifest_path = (
        PROJECTS_DIR
        / f"{PROJECT_ID}.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print(
        "✅ OpenStar multi-target blind project ready"
    )
    print(
        f"   project: {PROJECT_ID}"
    )
    print(
        f"   manifest: {manifest_path}"
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
        "   external published periods remain unqueried"
    )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a preregistered multi-target "
            "blind TESS validation set for OpenStar."
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--select",
        action="store_true",
        help=(
            "Discover, rank, and SHA-256 lock "
            "the blind target set."
        ),
    )

    mode.add_argument(
        "--prepare",
        action="store_true",
        help=(
            "Verify the existing lock and build "
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
