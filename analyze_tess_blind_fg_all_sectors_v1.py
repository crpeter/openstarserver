import argparse
import csv
import json
import math
import re
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar F/G all-sector diagnostic v1
#
# DEVELOPMENT ANALYSIS ONLY.
#
# F/G have already been revealed. This script does not alter the
# completed blind score. It asks whether the published VSX period
# appears in other TESS epochs even though the blind sector did not
# support it.
#
# Preprocessing intentionally mirrors the OpenStar validation rules:
#   - TESS only
#   - prefer SPOC 120 s
#   - else TESS-SPOC shortest cadence
#   - else SPOC shortest cadence
#   - quality_bitmask="default"
#   - finite samples only
#   - sort by time
#   - downsample before normalization
#   - normalize each sector independently in Float64
#   - Float32 quantization before period analysis
#
# Analyses:
#   1. exact OpenStar frequency grid per sector
#   2. exact VSX-frequency power/rank per sector
#   3. phase-independent consensus periodogram:
#         mean LS power across sectors
#   4. combined-gap LS using all sectors, independently normalized,
#      downsampled to the same 18,000-sample payload cap
#   5. folds at sector winner and VSX period
# ============================================================


REVEAL_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-published-v3.reveal-v1.json"
)

OUTPUT_DIR = Path(
    "data/analysis/"
    "openstar-blind-published-v3-fg-all-sectors-v1"
)

TARGETS = (
    "Blind V2-F",
    "Blind V2-G",
)

PREFERRED_AUTHOR = "SPOC"
FALLBACK_AUTHOR = "TESS-SPOC"
PREFERRED_EXPTIME_SECONDS = 120

MAX_SAMPLES_PER_SECTOR = 18000
MAX_COMBINED_SAMPLES = 18000

MINIMUM_FREQUENCY = 0.1
MAXIMUM_FREQUENCY = 5.0
TOTAL_FREQUENCIES = 4194304

TOP_PEAK_COUNT = 10
FOLD_BINS = 100
MIN_POINTS_PER_BIN = 3


def frequency_step():
    return (
        MAXIMUM_FREQUENCY
        - MINIMUM_FREQUENCY
    ) / (
        TOTAL_FREQUENCIES
        - 1
    )


def frequency_grid():
    return (
        MINIMUM_FREQUENCY
        + np.arange(
            TOTAL_FREQUENCIES,
            dtype=np.float64,
        )
        * frequency_step()
    )


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def parse_sector_from_mission(value):
    match = re.search(
        r"Sector\s+(\d+)",
        str(value),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def search_metadata_row(search, index):
    table = search.table

    def value(column, default=None):
        if column not in table.colnames:
            return default

        result = table[column][index]

        if np.ma.is_masked(result):
            return default

        if hasattr(result, "value"):
            result = result.value

        return result

    sector = value(
        "sequence_number",
        None,
    )

    try:
        if sector is not None:
            sector = int(sector)
    except Exception:
        sector = None

    if sector is None:
        sector = parse_sector_from_mission(
            value(
                "mission",
                "",
            )
        )

    author = str(
        value(
            "author",
            "",
        )
    ).strip()

    exptime = value(
        "exptime",
        math.nan,
    )

    try:
        exptime = float(exptime)
    except Exception:
        exptime = math.nan

    return {
        "index": int(index),
        "sector": sector,
        "author": author,
        "exptime": exptime,
        "mission": str(
            value(
                "mission",
                "",
            )
        ),
    }


def choose_product_for_sector(
    search,
    rows,
):
    def shortest(
        candidates,
    ):
        if not candidates:
            return None

        finite = [
            item
            for item in candidates
            if math.isfinite(
                item["exptime"]
            )
        ]

        if finite:
            return min(
                finite,
                key=lambda item: (
                    item["exptime"],
                    item["index"],
                ),
            )

        return min(
            candidates,
            key=lambda item: (
                item["index"]
            ),
        )

    preferred = [
        item
        for item in rows
        if (
            item["author"]
            == PREFERRED_AUTHOR
            and math.isfinite(
                item["exptime"]
            )
            and abs(
                item["exptime"]
                - PREFERRED_EXPTIME_SECONDS
            )
            < 0.5
        )
    ]

    if preferred:
        return (
            shortest(
                preferred
            ),
            "SPOC 120s",
        )

    tess_spoc = [
        item
        for item in rows
        if item[
            "author"
        ] == FALLBACK_AUTHOR
    ]

    if tess_spoc:
        return (
            shortest(
                tess_spoc
            ),
            "TESS-SPOC shortest",
        )

    spoc = [
        item
        for item in rows
        if item[
            "author"
        ] == PREFERRED_AUTHOR
    ]

    if spoc:
        return (
            shortest(
                spoc
            ),
            "SPOC shortest",
        )

    return (
        None,
        None,
    )


def discover_products(
    tic_id,
):
    query = (
        f"TIC {tic_id}"
    )

    print()
    print(
        f"🔭 Searching all TESS light curves for "
        f"{query}"
    )

    search = lk.search_lightcurve(
        query,
        mission="TESS",
    )

    if len(search) == 0:
        raise RuntimeError(
            f"No TESS light curves found for "
            f"{query}."
        )

    rows = [
        search_metadata_row(
            search,
            index,
        )
        for index in range(
            len(search)
        )
    ]

    by_sector = {}

    for row in rows:
        sector = row[
            "sector"
        ]

        if sector is None:
            continue

        by_sector.setdefault(
            sector,
            [],
        ).append(
            row
        )

    selected = []

    for sector in sorted(
        by_sector
    ):
        row, rule = (
            choose_product_for_sector(
                search,
                by_sector[
                    sector
                ],
            )
        )

        if row is None:
            print(
                f"   Sector {sector}: "
                "no SPOC-family product"
            )
            continue

        selected.append(
            {
                **row,
                "selectionRule": (
                    rule
                ),
            }
        )

        cadence = (
            f"{row['exptime']:.0f}s"
            if math.isfinite(
                row[
                    "exptime"
                ]
            )
            else "unknown cadence"
        )

        print(
            f"   Sector {sector}: "
            f"{row['author']}, "
            f"{cadence} "
            f"({rule})"
        )

    if not selected:
        raise RuntimeError(
            "No supported SPOC-family "
            "TESS products found."
        )

    return (
        search,
        selected,
    )


def downsample_indices(
    count,
    maximum,
):
    if count <= maximum:
        return np.arange(
            count,
            dtype=np.int64,
        )

    return np.linspace(
        0,
        count - 1,
        maximum,
        dtype=np.int64,
    )


def load_selected_sector(
    search,
    product,
):
    sector = int(
        product[
            "sector"
        ]
    )

    print()
    print(
        f"⬇️ Sector {sector}"
    )
    print(
        f"   author: "
        f"{product['author']}"
    )
    print(
        "   cadence: "
        f"{product['exptime']:.0f}s"
        if math.isfinite(
            product[
                "exptime"
            ]
        )
        else "   cadence: unknown"
    )

    selected_search = search[
        product[
            "index"
        ]:
        product[
            "index"
        ]
        + 1
    ]

    light_curve = (
        selected_search[
            0
        ].download(
            quality_bitmask="default"
        )
    )

    if light_curve is None:
        raise RuntimeError(
            f"Download failed for "
            f"Sector {sector}."
        )

    times64 = np.asarray(
        light_curve.time.value,
        dtype=np.float64,
    )

    flux64 = np.asarray(
        light_curve.flux.value,
        dtype=np.float64,
    )

    original_samples = int(
        len(
            times64
        )
    )

    finite = (
        np.isfinite(
            times64
        )
        & np.isfinite(
            flux64
        )
    )

    times64 = times64[
        finite
    ]

    flux64 = flux64[
        finite
    ]

    order = np.argsort(
        times64
    )

    times64 = times64[
        order
    ]

    flux64 = flux64[
        order
    ]

    finite_samples = int(
        len(
            times64
        )
    )

    if finite_samples < 2:
        raise RuntimeError(
            f"Sector {sector} contains "
            "too few finite samples."
        )

    selected_indices = (
        downsample_indices(
            finite_samples,
            MAX_SAMPLES_PER_SECTOR,
        )
    )

    times64 = times64[
        selected_indices
    ]

    flux64 = flux64[
        selected_indices
    ]

    source_mean = float(
        np.mean(
            flux64
        )
    )

    source_stddev = float(
        np.std(
            flux64
        )
    )

    if (
        not math.isfinite(
            source_stddev
        )
        or source_stddev <= 0
    ):
        raise RuntimeError(
            f"Sector {sector} has "
            "invalid flux standard deviation."
        )

    normalized_flux64 = (
        flux64
        - source_mean
    ) / source_stddev

    time_origin = float(
        times64[0]
    )

    relative_times64 = (
        times64
        - time_origin
    )

    # Match the distributed representation before analysis.
    relative_times32 = np.asarray(
        relative_times64,
        dtype=np.float32,
    )

    flux32 = np.asarray(
        normalized_flux64,
        dtype=np.float32,
    )

    # Cast back to Float64 only for Astropy.
    analysis_times = np.asarray(
        relative_times32,
        dtype=np.float64,
    )

    analysis_flux = np.asarray(
        flux32,
        dtype=np.float64,
    )

    print(
        f"   original samples: "
        f"{original_samples}"
    )
    print(
        f"   finite samples: "
        f"{finite_samples}"
    )
    print(
        f"   analyzed samples: "
        f"{len(analysis_times)}"
    )
    print(
        "   baseline: "
        f"{analysis_times[-1] - analysis_times[0]:.4f} d"
    )

    return {
        "sector": sector,
        "author": (
            product[
                "author"
            ]
        ),
        "cadenceSeconds": (
            product[
                "exptime"
            ]
        ),
        "selectionRule": (
            product[
                "selectionRule"
            ]
        ),
        "originalSamples": (
            original_samples
        ),
        "finiteSamples": (
            finite_samples
        ),
        "selectedSamples": int(
            len(
                analysis_times
            )
        ),
        "absoluteTimes64": (
            times64
        ),
        "analysisTimes": (
            analysis_times
        ),
        "analysisFlux": (
            analysis_flux
        ),
        "sourceFluxMean": (
            source_mean
        ),
        "sourceFluxStddev": (
            source_stddev
        ),
    }


def nearest_grid_diagnostic(
    frequencies,
    powers,
    frequency,
):
    index = int(
        np.argmin(
            np.abs(
                frequencies
                - frequency
            )
        )
    )

    power = float(
        powers[
            index
        ]
    )

    rank = (
        int(
            np.count_nonzero(
                powers
                > power
            )
        )
        + 1
    )

    return {
        "requestedFrequency": (
            float(
                frequency
            )
        ),
        "gridFrequency": float(
            frequencies[
                index
            ]
        ),
        "gridPeriodDays": (
            1.0
            / float(
                frequencies[
                    index
                ]
            )
        ),
        "power": power,
        "rank": rank,
        "rankFraction": (
            rank
            / len(
                powers
            )
        ),
    }


def independent_peaks(
    frequencies,
    powers,
    baseline,
    count=TOP_PEAK_COUNT,
):
    rayleigh = (
        1.0
        / baseline
    )

    order = np.argsort(
        powers
    )[::-1]

    result = []

    for index in order:
        frequency = float(
            frequencies[
                index
            ]
        )

        if any(
            abs(
                frequency
                - item[
                    "frequency"
                ]
            )
            < rayleigh
            for item in result
        ):
            continue

        result.append(
            {
                "frequency": (
                    frequency
                ),
                "periodDays": (
                    1.0
                    / frequency
                ),
                "power": float(
                    powers[
                        index
                    ]
                ),
                "gridIndex": int(
                    index
                ),
            }
        )

        if len(result) >= count:
            break

    return result


def fold_rms(
    times,
    flux,
    period_days,
):
    phase = np.mod(
        times
        / period_days,
        1.0,
    )

    indices = np.floor(
        phase
        * FOLD_BINS
    ).astype(
        np.int64
    )

    indices = np.clip(
        indices,
        0,
        FOLD_BINS - 1,
    )

    residuals = []

    for bin_index in range(
        FOLD_BINS
    ):
        mask = (
            indices
            == bin_index
        )

        count = int(
            np.count_nonzero(
                mask
            )
        )

        if count < MIN_POINTS_PER_BIN:
            continue

        values = flux[
            mask
        ]

        residuals.append(
            values
            - np.mean(
                values
            )
        )

    if not residuals:
        return None

    residual = np.concatenate(
        residuals
    )

    return float(
        np.sqrt(
            np.mean(
                residual
                * residual
            )
        )
    )


def harmonic_relation(
    measured_period,
    vsx_period,
):
    ratio = (
        measured_period
        / vsx_period
    )

    candidates = (
        (0.25, "1/4×"),
        (0.50, "1/2×"),
        (1.00, "direct"),
        (2.00, "2×"),
        (3.00, "3×"),
        (4.00, "4×"),
        (0.20, "1/5×"),
        (5.00, "5×"),
        (6.00, "6×"),
        (7.00, "7×"),
        (8.00, "8×"),
    )

    expected, label = min(
        candidates,
        key=lambda item: abs(
            ratio
            - item[0]
        ),
    )

    error_percent = (
        abs(
            ratio
            - expected
        )
        / expected
        * 100.0
    )

    return {
        "label": label,
        "ratio": ratio,
        "relationErrorPercent": (
            error_percent
        ),
    }


def save_fold_plot(
    path,
    times,
    flux,
    period_days,
    title,
):
    phase = np.mod(
        times
        / period_days,
        1.0,
    )

    plot_phase = np.concatenate(
        (
            phase,
            phase + 1.0,
        )
    )

    plot_flux = np.concatenate(
        (
            flux,
            flux,
        )
    )

    figure = plt.figure(
        figsize=(
            9,
            5,
        )
    )

    axes = figure.add_subplot(
        111
    )

    axes.scatter(
        plot_phase,
        plot_flux,
        s=3,
        alpha=0.16,
    )

    axes.set_xlim(
        0.0,
        2.0,
    )

    axes.set_xlabel(
        "Phase"
    )

    axes.set_ylabel(
        "Normalized flux"
    )

    axes.set_title(
        f"{title}: "
        f"{period_days:.8f} d"
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(
        figure
    )


def save_sector_summary_plot(
    path,
    rows,
    vsx_period,
    title,
):
    sectors = [
        row[
            "sector"
        ]
        for row in rows
    ]

    winner_periods = [
        row[
            "winnerPeriodDays"
        ]
        for row in rows
    ]

    figure = plt.figure(
        figsize=(
            10,
            5,
        )
    )

    axes = figure.add_subplot(
        111
    )

    axes.scatter(
        sectors,
        winner_periods,
        s=50,
        label="Sector LS winner",
    )

    axes.axhline(
        vsx_period,
        linestyle=":",
        label="VSX",
    )

    axes.set_xlabel(
        "TESS sector"
    )

    axes.set_ylabel(
        "Period (days)"
    )

    axes.set_title(
        title
    )

    axes.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(
        figure
    )


def analyze_sector(
    target_slug,
    sector_data,
    frequencies,
    vsx_period,
    output_dir,
):
    sector = int(
        sector_data[
            "sector"
        ]
    )

    times = sector_data[
        "analysisTimes"
    ]

    flux = sector_data[
        "analysisFlux"
    ]

    baseline = float(
        times[-1]
        - times[0]
    )

    model = LombScargle(
        times,
        flux,
        fit_mean=True,
        center_data=True,
    )

    powers = np.asarray(
        model.power(
            frequencies
        ),
        dtype=np.float64,
    )

    winner_index = int(
        np.argmax(
            powers
        )
    )

    winner_frequency = float(
        frequencies[
            winner_index
        ]
    )

    winner_period = (
        1.0
        / winner_frequency
    )

    winner_power = float(
        powers[
            winner_index
        ]
    )

    vsx_frequency = (
        1.0
        / vsx_period
    )

    vsx_grid = (
        nearest_grid_diagnostic(
            frequencies,
            powers,
            vsx_frequency,
        )
    )

    vsx_exact_power = float(
        model.power(
            vsx_frequency
        )
    )

    relation = (
        harmonic_relation(
            winner_period,
            vsx_period,
        )
    )

    top_peaks = (
        independent_peaks(
            frequencies,
            powers,
            baseline,
        )
    )

    winner_fold_rms = (
        fold_rms(
            times,
            flux,
            winner_period,
        )
    )

    vsx_fold_rms = (
        fold_rms(
            times,
            flux,
            vsx_period,
        )
    )

    power_ratio = (
        vsx_exact_power
        / winner_power
        if winner_power > 0
        else None
    )

    print()
    print(
        f"📐 Sector {sector} result"
    )
    print(
        "   winner: "
        f"{winner_period:.8f} d | "
        f"{winner_frequency:.8f} c/d | "
        f"power {winner_power:.8f}"
    )
    print(
        "   VSX exact power: "
        f"{vsx_exact_power:.8f}"
    )
    print(
        "   VSX rank: "
        f"{vsx_grid['rank']:,}/"
        f"{len(frequencies):,}"
    )
    print(
        "   VSX/winner power ratio: "
        f"{power_ratio:.6f}"
        if power_ratio is not None
        else "   VSX/winner power ratio: [n/a]"
    )
    print(
        "   winner relation to VSX: "
        f"{relation['label']} "
        f"(error "
        f"{relation['relationErrorPercent']:.3f}%)"
    )
    print(
        "   fold RMS — winner / VSX: "
        f"{winner_fold_rms:.6f} / "
        f"{vsx_fold_rms:.6f}"
        if (
            winner_fold_rms is not None
            and vsx_fold_rms is not None
        )
        else "   fold RMS: [n/a]"
    )

    sector_dir = (
        output_dir
        / target_slug
        / (
            f"sector-{sector}"
        )
    )

    sector_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_fold_plot(
        sector_dir
        / "winner-fold.png",
        times,
        flux,
        winner_period,
        (
            f"{target_slug} Sector {sector} "
            "winner"
        ),
    )

    save_fold_plot(
        sector_dir
        / "vsx-fold.png",
        times,
        flux,
        vsx_period,
        (
            f"{target_slug} Sector {sector} "
            "VSX"
        ),
    )

    return {
        "sector": sector,
        "author": (
            sector_data[
                "author"
            ]
        ),
        "cadenceSeconds": (
            sector_data[
                "cadenceSeconds"
            ]
        ),
        "selectionRule": (
            sector_data[
                "selectionRule"
            ]
        ),
        "originalSamples": (
            sector_data[
                "originalSamples"
            ]
        ),
        "finiteSamples": (
            sector_data[
                "finiteSamples"
            ]
        ),
        "selectedSamples": (
            sector_data[
                "selectedSamples"
            ]
        ),
        "baselineDays": baseline,
        "winnerFrequency": (
            winner_frequency
        ),
        "winnerPeriodDays": (
            winner_period
        ),
        "winnerPower": (
            winner_power
        ),
        "winnerRelationToVSX": (
            relation
        ),
        "vsxFrequency": (
            vsx_frequency
        ),
        "vsxExactPower": (
            vsx_exact_power
        ),
        "vsxGridPower": (
            vsx_grid[
                "power"
            ]
        ),
        "vsxRank": (
            vsx_grid[
                "rank"
            ]
        ),
        "vsxRankFraction": (
            vsx_grid[
                "rankFraction"
            ]
        ),
        "vsxToWinnerPowerRatio": (
            power_ratio
        ),
        "winnerFoldRMS": (
            winner_fold_rms
        ),
        "vsxFoldRMS": (
            vsx_fold_rms
        ),
        "topIndependentPeaks": (
            top_peaks
        ),
        # Returned only in memory for cross-sector consensus.
        "_powers": powers,
    }


def combined_gap_dataset(
    sector_data_items,
):
    all_times = np.concatenate(
        [
            item[
                "absoluteTimes64"
            ]
            for item in sector_data_items
        ]
    )

    all_flux = []

    sector_labels = []

    for item in sector_data_items:
        source_flux = np.asarray(
            item[
                "analysisFlux"
            ],
            dtype=np.float64,
        )

        source_times = np.asarray(
            item[
                "absoluteTimes64"
            ],
            dtype=np.float64,
        )

        # absoluteTimes64 may have more samples than analysisFlux when
        # a sector was downsampled. Recreate the same selected indices.
        selected_indices = (
            downsample_indices(
                len(
                    source_times
                ),
                MAX_SAMPLES_PER_SECTOR,
            )
        )

        source_times = source_times[
            selected_indices
        ]

        # analysisFlux is already normalized/Float32-equivalent and
        # corresponds to these selected indices.
        if len(
            source_times
        ) != len(
            source_flux
        ):
            raise RuntimeError(
                "Combined-sector selected sample "
                "count mismatch."
            )

        all_flux.append(
            source_flux
        )

        sector_labels.append(
            np.full(
                len(
                    source_flux
                ),
                int(
                    item[
                        "sector"
                    ]
                ),
                dtype=np.int32,
            )
        )

    selected_times = np.concatenate(
        [
            np.asarray(
                item[
                    "absoluteTimes64"
                ],
                dtype=np.float64,
            )[
                downsample_indices(
                    len(
                        item[
                            "absoluteTimes64"
                        ]
                    ),
                    MAX_SAMPLES_PER_SECTOR,
                )
            ]
            for item in sector_data_items
        ]
    )

    selected_flux = np.concatenate(
        all_flux
    )

    labels = np.concatenate(
        sector_labels
    )

    order = np.argsort(
        selected_times
    )

    selected_times = (
        selected_times[
            order
        ]
    )

    selected_flux = (
        selected_flux[
            order
        ]
    )

    labels = labels[
        order
    ]

    if len(
        selected_times
    ) > MAX_COMBINED_SAMPLES:
        indices = (
            downsample_indices(
                len(
                    selected_times
                ),
                MAX_COMBINED_SAMPLES,
            )
        )

        selected_times = (
            selected_times[
                indices
            ]
        )

        selected_flux = (
            selected_flux[
                indices
            ]
        )

        labels = labels[
            indices
        ]

    origin = float(
        selected_times[0]
    )

    times32 = np.asarray(
        selected_times
        - origin,
        dtype=np.float32,
    )

    flux32 = np.asarray(
        selected_flux,
        dtype=np.float32,
    )

    return {
        "times": np.asarray(
            times32,
            dtype=np.float64,
        ),
        "flux": np.asarray(
            flux32,
            dtype=np.float64,
        ),
        "sectorLabels": labels,
        "baselineDays": float(
            times32[-1]
            - times32[0]
        ),
    }


def analyze_target(
    reveal_target,
    frequencies,
    output_dir,
):
    blind_name = reveal_target[
        "blindName"
    ]

    tic_id = int(
        reveal_target[
            "ticID"
        ]
    )

    vsx_period = float(
        reveal_target[
            "vsx"
        ][
            "periodDays"
        ]
    )

    vsx_frequency = (
        1.0
        / vsx_period
    )

    target_slug = (
        blind_name
        .lower()
        .replace(
            " ",
            "-",
        )
    )

    print()
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
        f"   VSX period: "
        f"{vsx_period:.8f} d"
    )
    print(
        f"   VSX frequency: "
        f"{vsx_frequency:.8f} c/d"
    )

    search, products = (
        discover_products(
            tic_id
        )
    )

    print(
        f"   supported sectors found: "
        f"{len(products)}"
    )

    sector_data_items = []

    sector_results = []

    consensus_power_sum = np.zeros(
        len(
            frequencies
        ),
        dtype=np.float64,
    )

    for product in products:
        sector_data = (
            load_selected_sector(
                search,
                product,
            )
        )

        sector_data_items.append(
            sector_data
        )

        result = analyze_sector(
            target_slug,
            sector_data,
            frequencies,
            vsx_period,
            output_dir,
        )

        consensus_power_sum += (
            result[
                "_powers"
            ]
        )

        sector_results.append(
            result
        )

    consensus_powers = (
        consensus_power_sum
        / len(
            sector_results
        )
    )

    consensus_index = int(
        np.argmax(
            consensus_powers
        )
    )

    consensus_frequency = float(
        frequencies[
            consensus_index
        ]
    )

    consensus_period = (
        1.0
        / consensus_frequency
    )

    consensus_power = float(
        consensus_powers[
            consensus_index
        ]
    )

    consensus_vsx = (
        nearest_grid_diagnostic(
            frequencies,
            consensus_powers,
            vsx_frequency,
        )
    )

    consensus_relation = (
        harmonic_relation(
            consensus_period,
            vsx_period,
        )
    )

    print()
    print(
        "🧭 Phase-independent sector consensus"
    )
    print(
        "   mean-power winner: "
        f"{consensus_period:.8f} d | "
        f"{consensus_frequency:.8f} c/d | "
        f"mean power {consensus_power:.8f}"
    )
    print(
        "   VSX consensus power: "
        f"{consensus_vsx['power']:.8f}"
    )
    print(
        "   VSX consensus rank: "
        f"{consensus_vsx['rank']:,}/"
        f"{len(frequencies):,}"
    )
    print(
        "   consensus relation to VSX: "
        f"{consensus_relation['label']} "
        f"(error "
        f"{consensus_relation['relationErrorPercent']:.3f}%)"
    )

    combined = (
        combined_gap_dataset(
            sector_data_items
        )
    )

    print()
    print(
        "🌉 Combined-gap LS"
    )
    print(
        f"   samples: "
        f"{len(combined['times'])}"
    )
    print(
        f"   full baseline: "
        f"{combined['baselineDays']:.3f} d"
    )

    combined_model = (
        LombScargle(
            combined[
                "times"
            ],
            combined[
                "flux"
            ],
            fit_mean=True,
            center_data=True,
        )
    )

    combined_powers = np.asarray(
        combined_model.power(
            frequencies
        ),
        dtype=np.float64,
    )

    combined_index = int(
        np.argmax(
            combined_powers
        )
    )

    combined_frequency = float(
        frequencies[
            combined_index
        ]
    )

    combined_period = (
        1.0
        / combined_frequency
    )

    combined_power = float(
        combined_powers[
            combined_index
        ]
    )

    combined_vsx = (
        nearest_grid_diagnostic(
            frequencies,
            combined_powers,
            vsx_frequency,
        )
    )

    combined_vsx_exact_power = float(
        combined_model.power(
            vsx_frequency
        )
    )

    combined_relation = (
        harmonic_relation(
            combined_period,
            vsx_period,
        )
    )

    print(
        "   winner: "
        f"{combined_period:.8f} d | "
        f"{combined_frequency:.8f} c/d | "
        f"power {combined_power:.8f}"
    )
    print(
        "   VSX exact power: "
        f"{combined_vsx_exact_power:.8f}"
    )
    print(
        "   VSX rank: "
        f"{combined_vsx['rank']:,}/"
        f"{len(frequencies):,}"
    )
    print(
        "   winner relation to VSX: "
        f"{combined_relation['label']} "
        f"(error "
        f"{combined_relation['relationErrorPercent']:.3f}%)"
    )

    save_fold_plot(
        output_dir
        / target_slug
        / "combined-winner-fold.png",
        combined[
            "times"
        ],
        combined[
            "flux"
        ],
        combined_period,
        (
            f"{blind_name} combined winner"
        ),
    )

    save_fold_plot(
        output_dir
        / target_slug
        / "combined-vsx-fold.png",
        combined[
            "times"
        ],
        combined[
            "flux"
        ],
        vsx_period,
        (
            f"{blind_name} combined VSX"
        ),
    )

    serializable_sector_results = []

    for result in sector_results:
        item = dict(
            result
        )

        item.pop(
            "_powers",
            None,
        )

        serializable_sector_results.append(
            item
        )

    save_sector_summary_plot(
        output_dir
        / target_slug
        / "sector-winner-periods.png",
        serializable_sector_results,
        vsx_period,
        (
            f"{blind_name}: sector LS winners"
        ),
    )

    sectors_supporting_vsx = []

    for item in serializable_sector_results:
        relation = item[
            "winnerRelationToVSX"
        ]

        if (
            relation[
                "relationErrorPercent"
            ]
            <= 2.0
            and relation[
                "label"
            ]
            in (
                "direct",
                "1/2×",
                "2×",
                "3×",
                "4×",
            )
        ):
            sectors_supporting_vsx.append(
                {
                    "sector": (
                        item[
                            "sector"
                        ]
                    ),
                    "relation": (
                        relation[
                            "label"
                        ]
                    ),
                    "errorPercent": (
                        relation[
                            "relationErrorPercent"
                        ]
                    ),
                }
            )

    print()
    print(
        "📋 Target summary"
    )
    print(
        f"   sectors analyzed: "
        f"{len(serializable_sector_results)}"
    )
    print(
        "   sectors with winner within 2% "
        "of VSX/simple harmonic: "
        f"{len(sectors_supporting_vsx)}"
    )

    for item in (
        sectors_supporting_vsx
    ):
        print(
            f"      Sector {item['sector']}: "
            f"{item['relation']}, "
            f"{item['errorPercent']:.3f}%"
        )

    return {
        "blindName": (
            blind_name
        ),
        "ticID": (
            tic_id
        ),
        "vsxName": (
            reveal_target[
                "vsx"
            ].get(
                "name"
            )
        ),
        "vsxType": (
            reveal_target[
                "vsx"
            ].get(
                "type"
            )
        ),
        "vsxPeriodDays": (
            vsx_period
        ),
        "vsxFrequency": (
            vsx_frequency
        ),
        "sectorCount": (
            len(
                serializable_sector_results
            )
        ),
        "sectors": (
            serializable_sector_results
        ),
        "sectorsSupportingVSXOrSimpleHarmonic": (
            sectors_supporting_vsx
        ),
        "phaseIndependentConsensus": {
            "frequency": (
                consensus_frequency
            ),
            "periodDays": (
                consensus_period
            ),
            "meanPower": (
                consensus_power
            ),
            "vsxPower": (
                consensus_vsx[
                    "power"
                ]
            ),
            "vsxRank": (
                consensus_vsx[
                    "rank"
                ]
            ),
            "relationToVSX": (
                consensus_relation
            ),
        },
        "combinedGap": {
            "samples": int(
                len(
                    combined[
                        "times"
                    ]
                )
            ),
            "baselineDays": (
                combined[
                    "baselineDays"
                ]
            ),
            "winnerFrequency": (
                combined_frequency
            ),
            "winnerPeriodDays": (
                combined_period
            ),
            "winnerPower": (
                combined_power
            ),
            "vsxExactPower": (
                combined_vsx_exact_power
            ),
            "vsxRank": (
                combined_vsx[
                    "rank"
                ]
            ),
            "relationToVSX": (
                combined_relation
            ),
        },
    }


def write_csv(
    output_path,
    target_results,
):
    fields = (
        "blindName",
        "ticID",
        "sector",
        "author",
        "cadenceSeconds",
        "selectedSamples",
        "baselineDays",
        "winnerPeriodDays",
        "winnerFrequency",
        "winnerPower",
        "winnerRelation",
        "winnerRelationErrorPercent",
        "vsxPeriodDays",
        "vsxExactPower",
        "vsxRank",
        "vsxRankFraction",
        "vsxToWinnerPowerRatio",
        "winnerFoldRMS",
        "vsxFoldRMS",
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                fields
            ),
        )

        writer.writeheader()

        for target in target_results:
            for sector in target[
                "sectors"
            ]:
                relation = sector[
                    "winnerRelationToVSX"
                ]

                writer.writerow(
                    {
                        "blindName": (
                            target[
                                "blindName"
                            ]
                        ),
                        "ticID": (
                            target[
                                "ticID"
                            ]
                        ),
                        "sector": (
                            sector[
                                "sector"
                            ]
                        ),
                        "author": (
                            sector[
                                "author"
                            ]
                        ),
                        "cadenceSeconds": (
                            sector[
                                "cadenceSeconds"
                            ]
                        ),
                        "selectedSamples": (
                            sector[
                                "selectedSamples"
                            ]
                        ),
                        "baselineDays": (
                            sector[
                                "baselineDays"
                            ]
                        ),
                        "winnerPeriodDays": (
                            sector[
                                "winnerPeriodDays"
                            ]
                        ),
                        "winnerFrequency": (
                            sector[
                                "winnerFrequency"
                            ]
                        ),
                        "winnerPower": (
                            sector[
                                "winnerPower"
                            ]
                        ),
                        "winnerRelation": (
                            relation[
                                "label"
                            ]
                        ),
                        "winnerRelationErrorPercent": (
                            relation[
                                "relationErrorPercent"
                            ]
                        ),
                        "vsxPeriodDays": (
                            target[
                                "vsxPeriodDays"
                            ]
                        ),
                        "vsxExactPower": (
                            sector[
                                "vsxExactPower"
                            ]
                        ),
                        "vsxRank": (
                            sector[
                                "vsxRank"
                            ]
                        ),
                        "vsxRankFraction": (
                            sector[
                                "vsxRankFraction"
                            ]
                        ),
                        "vsxToWinnerPowerRatio": (
                            sector[
                                "vsxToWinnerPowerRatio"
                            ]
                        ),
                        "winnerFoldRMS": (
                            sector[
                                "winnerFoldRMS"
                            ]
                        ),
                        "vsxFoldRMS": (
                            sector[
                                "vsxFoldRMS"
                            ]
                        ),
                    }
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze every supported TESS sector for "
            "revealed Blind V2-F and V2-G."
        )
    )

    parser.add_argument(
        "--reveal",
        type=Path,
        default=REVEAL_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    reveal = load_json(
        args.reveal
    )

    reveal_by_name = {
        item[
            "blindName"
        ]: item
        for item in reveal[
            "targets"
        ]
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "🔬 OpenStar F/G all-sector diagnostic v1"
    )
    print(
        "DEVELOPMENT ANALYSIS — blind score remains frozen."
    )
    print(
        "Frequency range: "
        f"{MINIMUM_FREQUENCY:.3f} - "
        f"{MAXIMUM_FREQUENCY:.3f} c/d"
    )
    print(
        "Frequencies per periodogram: "
        f"{TOTAL_FREQUENCIES:,}"
    )
    print(
        "Per-sector preprocessing: "
        "OpenStar-compatible Float32 boundary"
    )

    frequencies = (
        frequency_grid()
    )

    target_results = []

    for blind_name in TARGETS:
        reveal_target = (
            reveal_by_name.get(
                blind_name
            )
        )

        if reveal_target is None:
            raise RuntimeError(
                f"Reveal JSON is missing "
                f"{blind_name}."
            )

        if (
            reveal_target.get(
                "status"
            )
            != "REVEALED"
        ):
            raise RuntimeError(
                f"{blind_name} was not "
                "successfully revealed."
            )

        target_results.append(
            analyze_target(
                reveal_target,
                frequencies,
                args.output_dir,
            )
        )

    json_path = (
        args.output_dir
        / (
            "fg-all-sector-"
            "diagnostic-summary.json"
        )
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "projectID": (
                    reveal.get(
                        "projectID"
                    )
                ),
                "analysis": (
                    "post-reveal F/G all-sector diagnostic"
                ),
                "developmentOnly": True,
                "frequencySearch": {
                    "minimumFrequency": (
                        MINIMUM_FREQUENCY
                    ),
                    "maximumFrequency": (
                        MAXIMUM_FREQUENCY
                    ),
                    "totalFrequencies": (
                        TOTAL_FREQUENCIES
                    ),
                    "frequencyStep": (
                        frequency_step()
                    ),
                },
                "targets": (
                    target_results
                ),
            },
            file,
            indent=2,
            allow_nan=False,
        )

    csv_path = (
        args.output_dir
        / (
            "fg-all-sector-"
            "diagnostic-summary.csv"
        )
    )

    write_csv(
        csv_path,
        target_results,
    )

    print()
    print()
    print(
        "🏁 F/G all-sector diagnostic complete"
    )
    print(
        f"   JSON: {json_path}"
    )
    print(
        f"   CSV:  {csv_path}"
    )
    print(
        f"   plots: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
