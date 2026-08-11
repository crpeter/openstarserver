import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle

try:
    from scipy.optimize import minimize_scalar
except ImportError as error:
    raise RuntimeError(
        "This diagnostic requires scipy. "
        "Install it in the OpenStar virtualenv with: "
        "python3 -m pip install scipy"
    ) from error


# ============================================================
# OpenStar F/G mismatch diagnostic v1
#
# DEVELOPMENT ANALYSIS ONLY.
#
# Blind V2-F and V2-G have already been revealed, so results from
# this script MUST NOT be used to retroactively change the score
# of the completed blind validation set.
#
# Goal:
#   Determine whether the published VSX period is genuinely
#   present in the exact TESS data but missed by the current
#   single-harmonic/global-maximum resolver, or whether this TESS
#   sector itself does not strongly support the catalog period.
#
# Tests:
#   1. Exact full-grid Lomb-Scargle top 20 independent peaks
#   2. Power/rank at the exact VSX frequency and simple harmonics
#   3. Raw / linear-detrended / quadratic-detrended LS
#   4. Local multi-harmonic regression (1-4 Fourier terms)
#      around:
#          - OpenStar/Astropy winner
#          - VSX fundamental
#   5. PDM-style phase-dispersion score
#   6. Phase-fold within-bin RMS
#   7. Approximate gap-aware autocorrelation peaks
#   8. Diagnostic plots
#
# No tolerance widening. No catalog-based peak selection is
# applied to the original blind result.
# ============================================================


REVEAL_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-published-v3.reveal-v1.json"
)

OUTPUT_DIR = Path(
    "data/analysis/"
    "openstar-blind-published-v3-fg-diagnostic-v1"
)

TARGETS = {
    "Blind V2-F": Path(
        "data/tess-blind-v2-f-tic-315229214.json"
    ),
    "Blind V2-G": Path(
        "data/tess-blind-v2-g-tic-164697828.json"
    ),
}

TOP_PEAK_COUNT = 20

# Peaks closer than one Rayleigh resolution are treated as one
# independent peak family.
INDEPENDENT_PEAK_SEPARATION_RAYLEIGH = 1.0

PDM_BINS = 60
COHERENCE_BINS = 100
MIN_POINTS_PER_BIN = 3

MULTIHARMONIC_MAX_TERMS = 4

# Local refinement window around each seed, in Rayleigh widths.
MULTIHARMONIC_WINDOW_RAYLEIGH = 3.0

# ACF output.
ACF_TOP_PEAK_COUNT = 10
ACF_MIN_PERIOD_DAYS = 0.2
ACF_MAX_PERIOD_DAYS = 10.0


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def finite_arrays(dataset):
    times = np.asarray(
        dataset["times"],
        dtype=np.float64,
    )

    flux = np.asarray(
        dataset["flux"],
        dtype=np.float64,
    )

    finite = (
        np.isfinite(times)
        & np.isfinite(flux)
    )

    times = times[finite]
    flux = flux[finite]

    order = np.argsort(times)

    return (
        times[order],
        flux[order],
    )


def baseline_days(times):
    if len(times) < 2:
        return 0.0

    return float(
        times[-1]
        - times[0]
    )


def rayleigh_resolution(times):
    baseline = baseline_days(times)

    if baseline <= 0:
        raise RuntimeError(
            "Invalid time baseline."
        )

    return 1.0 / baseline


def frequency_grid(dataset):
    search = dataset[
        "frequencySearch"
    ]

    minimum = float(
        search[
            "minimumFrequency"
        ]
    )

    step = float(
        search[
            "frequencyStep"
        ]
    )

    count = int(
        search[
            "totalFrequencies"
        ]
    )

    return (
        minimum
        + np.arange(
            count,
            dtype=np.float64,
        )
        * step
    )


def detrend_variants(
    times,
    flux,
):
    centered_time = (
        times
        - np.mean(times)
    )

    variants = {
        "raw": np.asarray(
            flux,
            dtype=np.float64,
        ),
    }

    for order, name in (
        (1, "linear"),
        (2, "quadratic"),
    ):
        coefficients = np.polyfit(
            centered_time,
            flux,
            order,
        )

        trend = np.polyval(
            coefficients,
            centered_time,
        )

        residual = (
            flux
            - trend
        )

        residual -= np.mean(
            residual
        )

        variants[
            name
        ] = residual

    return variants


def select_independent_peaks(
    frequencies,
    powers,
    *,
    minimum_separation,
    count,
):
    order = np.argsort(
        powers
    )[::-1]

    selected = []

    for index in order:
        frequency = float(
            frequencies[index]
        )

        if any(
            abs(
                frequency
                - item[
                    "frequency"
                ]
            )
            < minimum_separation
            for item in selected
        ):
            continue

        selected.append(
            {
                "gridIndex": int(index),
                "frequency": frequency,
                "periodDays": (
                    1.0 / frequency
                ),
                "power": float(
                    powers[index]
                ),
            }
        )

        if len(selected) >= count:
            break

    return selected


def nearest_grid_diagnostic(
    frequencies,
    powers,
    target_frequency,
):
    index = int(
        np.argmin(
            np.abs(
                frequencies
                - target_frequency
            )
        )
    )

    power = float(
        powers[index]
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

    percentile = (
        100.0
        * (
            1.0
            - (
                rank
                - 1
            )
            / len(powers)
        )
    )

    return {
        "requestedFrequency": (
            float(
                target_frequency
            )
        ),
        "gridIndex": index,
        "gridFrequency": float(
            frequencies[index]
        ),
        "gridPeriodDays": (
            1.0
            / float(
                frequencies[index]
            )
        ),
        "power": power,
        "rank": rank,
        "percentile": (
            percentile
        ),
    }


def candidate_frequency_set(
    vsx_period,
):
    vsx_frequency = (
        1.0
        / vsx_period
    )

    return {
        "VSX fundamental": (
            vsx_frequency
        ),
        "VSX 1/2 frequency": (
            0.5
            * vsx_frequency
        ),
        "VSX 2x frequency": (
            2.0
            * vsx_frequency
        ),
        "VSX 3x frequency": (
            3.0
            * vsx_frequency
        ),
        "VSX 4x frequency": (
            4.0
            * vsx_frequency
        ),
    }


def run_full_lomb_scargle(
    times,
    flux,
    frequencies,
):
    model = LombScargle(
        times,
        flux,
    )

    powers = model.power(
        frequencies
    )

    return (
        model,
        np.asarray(
            powers,
            dtype=np.float64,
        ),
    )


def phase_bins(
    times,
    flux,
    period_days,
    bin_count,
):
    phase = np.mod(
        times
        / period_days,
        1.0,
    )

    indices = np.floor(
        phase
        * bin_count
    ).astype(
        np.int64
    )

    indices = np.clip(
        indices,
        0,
        bin_count - 1,
    )

    means = np.full(
        bin_count,
        np.nan,
        dtype=np.float64,
    )

    variances = np.full(
        bin_count,
        np.nan,
        dtype=np.float64,
    )

    counts = np.zeros(
        bin_count,
        dtype=np.int64,
    )

    for index in range(
        bin_count
    ):
        mask = (
            indices
            == index
        )

        count = int(
            np.count_nonzero(
                mask
            )
        )

        counts[index] = count

        if count == 0:
            continue

        values = flux[
            mask
        ]

        means[index] = float(
            np.mean(values)
        )

        if count >= 2:
            variances[index] = float(
                np.var(
                    values,
                    ddof=1,
                )
            )

    return {
        "phase": phase,
        "indices": indices,
        "means": means,
        "variances": variances,
        "counts": counts,
    }


def fold_within_bin_rms(
    times,
    flux,
    period_days,
):
    profile = phase_bins(
        times,
        flux,
        period_days,
        COHERENCE_BINS,
    )

    residuals = []

    for index in range(
        COHERENCE_BINS
    ):
        if (
            profile[
                "counts"
            ][index]
            < MIN_POINTS_PER_BIN
        ):
            continue

        mean = profile[
            "means"
        ][index]

        if not math.isfinite(
            mean
        ):
            continue

        mask = (
            profile[
                "indices"
            ]
            == index
        )

        residuals.append(
            flux[mask]
            - mean
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


def pdm_theta(
    times,
    flux,
    period_days,
):
    profile = phase_bins(
        times,
        flux,
        period_days,
        PDM_BINS,
    )

    total_variance = float(
        np.var(
            flux,
            ddof=1,
        )
    )

    if (
        not math.isfinite(
            total_variance
        )
        or total_variance <= 0
    ):
        return None

    numerator = 0.0
    denominator_weight = 0

    for index in range(
        PDM_BINS
    ):
        count = int(
            profile[
                "counts"
            ][index]
        )

        variance = profile[
            "variances"
        ][index]

        if (
            count < 2
            or not math.isfinite(
                variance
            )
        ):
            continue

        weight = (
            count
            - 1
        )

        numerator += (
            weight
            * variance
        )

        denominator_weight += weight

    if denominator_weight <= 0:
        return None

    return float(
        numerator
        / (
            denominator_weight
            * total_variance
        )
    )


def multiharmonic_score(
    times,
    flux,
    frequency,
    nterms,
):
    columns = [
        np.ones(
            len(times),
            dtype=np.float64,
        )
    ]

    angular = (
        2.0
        * np.pi
        * frequency
        * times
    )

    for harmonic in range(
        1,
        nterms + 1,
    ):
        columns.append(
            np.sin(
                harmonic
                * angular
            )
        )

        columns.append(
            np.cos(
                harmonic
                * angular
            )
        )

    design = np.column_stack(
        columns
    )

    coefficients, _, _, _ = (
        np.linalg.lstsq(
            design,
            flux,
            rcond=None,
        )
    )

    model = (
        design
        @ coefficients
    )

    residual = (
        flux
        - model
    )

    sse = float(
        np.dot(
            residual,
            residual,
        )
    )

    centered = (
        flux
        - np.mean(
            flux
        )
    )

    sst = float(
        np.dot(
            centered,
            centered,
        )
    )

    if sst <= 0:
        return None

    return (
        1.0
        - sse
        / sst
    )


def refine_multiharmonic(
    times,
    flux,
    seed_frequency,
    nterms,
    *,
    rayleigh,
    minimum_frequency,
    maximum_frequency,
):
    half_width = (
        MULTIHARMONIC_WINDOW_RAYLEIGH
        * rayleigh
    )

    lower = max(
        minimum_frequency,
        seed_frequency
        - half_width,
    )

    upper = min(
        maximum_frequency,
        seed_frequency
        + half_width,
    )

    if lower >= upper:
        score = multiharmonic_score(
            times,
            flux,
            seed_frequency,
            nterms,
        )

        return {
            "seedFrequency": (
                seed_frequency
            ),
            "bestFrequency": (
                seed_frequency
            ),
            "bestPeriodDays": (
                1.0
                / seed_frequency
            ),
            "score": score,
        }

    def objective(
        frequency,
    ):
        score = multiharmonic_score(
            times,
            flux,
            frequency,
            nterms,
        )

        if score is None:
            return math.inf

        return -score

    result = minimize_scalar(
        objective,
        bounds=(
            lower,
            upper,
        ),
        method="bounded",
        options={
            "xatol": 1e-10,
            "maxiter": 80,
        },
    )

    best_frequency = float(
        result.x
    )

    best_score = (
        -float(
            result.fun
        )
    )

    return {
        "seedFrequency": float(
            seed_frequency
        ),
        "searchMinimumFrequency": (
            lower
        ),
        "searchMaximumFrequency": (
            upper
        ),
        "bestFrequency": (
            best_frequency
        ),
        "bestPeriodDays": (
            1.0
            / best_frequency
        ),
        "score": (
            best_score
        ),
        "optimizerSuccess": bool(
            result.success
        ),
    }


def gap_aware_acf(
    times,
    flux,
):
    differences = np.diff(
        times
    )

    positive = differences[
        differences > 0
    ]

    if len(
        positive
    ) == 0:
        raise RuntimeError(
            "Cannot determine cadence for ACF."
        )

    cadence = float(
        np.median(
            positive
        )
    )

    if cadence <= 0:
        raise RuntimeError(
            "Invalid median cadence."
        )

    sample_index = np.rint(
        (
            times
            - times[0]
        )
        / cadence
    ).astype(
        np.int64
    )

    grid_count = (
        int(
            sample_index[-1]
        )
        + 1
    )

    values = np.zeros(
        grid_count,
        dtype=np.float64,
    )

    weights = np.zeros(
        grid_count,
        dtype=np.float64,
    )

    # Multiple points should not normally map to one cadence
    # slot, but averaging makes the method robust if they do.
    for index, value in zip(
        sample_index,
        flux,
    ):
        values[index] += value
        weights[index] += 1.0

    occupied = (
        weights
        > 0
    )

    values[
        occupied
    ] /= weights[
        occupied
    ]

    mean = float(
        np.mean(
            values[
                occupied
            ]
        )
    )

    centered = np.zeros_like(
        values
    )

    centered[
        occupied
    ] = (
        values[
            occupied
        ]
        - mean
    )

    mask = occupied.astype(
        np.float64
    )

    nfft = 1

    while nfft < (
        2
        * grid_count
    ):
        nfft *= 2

    signal_fft = np.fft.rfft(
        centered,
        n=nfft,
    )

    mask_fft = np.fft.rfft(
        mask,
        n=nfft,
    )

    numerator = np.fft.irfft(
        signal_fft
        * np.conjugate(
            signal_fft
        ),
        n=nfft,
    )[
        :grid_count
    ]

    pair_count = np.fft.irfft(
        mask_fft
        * np.conjugate(
            mask_fft
        ),
        n=nfft,
    )[
        :grid_count
    ]

    variance = float(
        np.var(
            values[
                occupied
            ]
        )
    )

    acf = np.full(
        grid_count,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        pair_count
        >= 5
    )

    if variance > 0:
        acf[
            valid
        ] = (
            numerator[
                valid
            ]
            / pair_count[
                valid
            ]
            / variance
        )

    lags = (
        np.arange(
            grid_count,
            dtype=np.float64,
        )
        * cadence
    )

    return (
        lags,
        acf,
        cadence,
        pair_count,
    )


def local_maxima(
    x,
    y,
    *,
    minimum_x,
    maximum_x,
    count,
    minimum_separation,
):
    finite = (
        np.isfinite(y)
        & (x >= minimum_x)
        & (x <= maximum_x)
    )

    indices = np.where(
        finite
    )[0]

    candidates = []

    for index in indices:
        if (
            index <= 0
            or index
            >= len(y) - 1
        ):
            continue

        if (
            not np.isfinite(
                y[index - 1]
            )
            or not np.isfinite(
                y[index + 1]
            )
        ):
            continue

        if (
            y[index]
            >= y[index - 1]
            and y[index]
            >= y[index + 1]
        ):
            candidates.append(
                index
            )

    candidates.sort(
        key=lambda index: (
            y[index]
        ),
        reverse=True,
    )

    selected = []

    for index in candidates:
        value_x = float(
            x[index]
        )

        if any(
            abs(
                value_x
                - item["periodDays"]
            )
            < minimum_separation
            for item in selected
        ):
            continue

        selected.append(
            {
                "periodDays": (
                    value_x
                ),
                "acf": float(
                    y[index]
                ),
            }
        )

        if len(selected) >= count:
            break

    return selected


def candidate_metrics(
    times,
    flux,
    periods,
):
    results = []

    for label, period in periods:
        if (
            period <= 0
            or not math.isfinite(
                period
            )
        ):
            continue

        results.append(
            {
                "label": label,
                "periodDays": (
                    float(
                        period
                    )
                ),
                "frequency": (
                    1.0
                    / float(
                        period
                    )
                ),
                "pdmTheta": (
                    pdm_theta(
                        times,
                        flux,
                        period,
                    )
                ),
                "foldWithinBinRMS": (
                    fold_within_bin_rms(
                        times,
                        flux,
                        period,
                    )
                ),
            }
        )

    return results


def save_periodogram_plot(
    path,
    frequencies,
    powers,
    *,
    title,
    openstar_frequency,
    vsx_frequency,
):
    figure = plt.figure(
        figsize=(
            10,
            5,
        )
    )

    axes = figure.add_subplot(
        111
    )

    axes.plot(
        frequencies,
        powers,
        linewidth=0.7,
    )

    axes.axvline(
        openstar_frequency,
        linestyle="--",
        label="OpenStar",
    )

    axes.axvline(
        vsx_frequency,
        linestyle=":",
        label="VSX",
    )

    axes.set_xlabel(
        "Frequency (cycles/day)"
    )

    axes.set_ylabel(
        "Lomb–Scargle power"
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


def save_fold_plot(
    path,
    times,
    flux,
    period,
    *,
    title,
):
    phase = np.mod(
        times
        / period,
        1.0,
    )

    profile = phase_bins(
        times,
        flux,
        period,
        COHERENCE_BINS,
    )

    centers = (
        np.arange(
            COHERENCE_BINS,
            dtype=np.float64,
        )
        + 0.5
    ) / COHERENCE_BINS

    figure = plt.figure(
        figsize=(
            9,
            5,
        )
    )

    axes = figure.add_subplot(
        111
    )

    doubled_phase = np.concatenate(
        (
            phase,
            phase + 1.0,
        )
    )

    doubled_flux = np.concatenate(
        (
            flux,
            flux,
        )
    )

    axes.scatter(
        doubled_phase,
        doubled_flux,
        s=3,
        alpha=0.16,
    )

    means = profile[
        "means"
    ]

    finite = np.isfinite(
        means
    )

    if np.any(
        finite
    ):
        axes.plot(
            np.concatenate(
                (
                    centers[finite],
                    centers[finite]
                    + 1.0,
                )
            ),
            np.concatenate(
                (
                    means[finite],
                    means[finite],
                )
            ),
            linewidth=2.0,
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
        f"{period:.8f} d"
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(
        figure
    )


def save_acf_plot(
    path,
    lags,
    acf,
    *,
    title,
    openstar_period,
    vsx_period,
):
    mask = (
        np.isfinite(acf)
        & (lags >= ACF_MIN_PERIOD_DAYS)
        & (lags <= ACF_MAX_PERIOD_DAYS)
    )

    figure = plt.figure(
        figsize=(
            10,
            5,
        )
    )

    axes = figure.add_subplot(
        111
    )

    axes.plot(
        lags[mask],
        acf[mask],
        linewidth=1.0,
    )

    axes.axvline(
        openstar_period,
        linestyle="--",
        label="OpenStar",
    )

    axes.axvline(
        vsx_period,
        linestyle=":",
        label="VSX",
    )

    axes.set_xlabel(
        "Lag (days)"
    )

    axes.set_ylabel(
        "Approx. autocorrelation"
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


def analyze_target(
    blind_name,
    dataset_path,
    reveal_target,
    output_dir,
):
    dataset = load_json(
        dataset_path
    )

    times, flux = finite_arrays(
        dataset
    )

    baseline = baseline_days(
        times
    )

    rayleigh = (
        rayleigh_resolution(
            times
        )
    )

    frequencies = frequency_grid(
        dataset
    )

    minimum_frequency = float(
        frequencies[0]
    )

    maximum_frequency = float(
        frequencies[-1]
    )

    frozen = reveal_target[
        "frozen"
    ]

    openstar_frequency = float(
        frozen[
            "openstarFrequency"
        ]
    )

    openstar_period = float(
        frozen[
            "openstarPeriodDays"
        ]
    )

    astropy_frequency = float(
        frozen[
            "astropyFrequency"
        ]
    )

    astropy_period = float(
        frozen[
            "astropyPeriodDays"
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

    print()
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"⭐ {blind_name} — "
        f"TIC {reveal_target['ticID']}"
    )
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"   samples: {len(times)}"
    )
    print(
        f"   baseline: {baseline:.6f} d"
    )
    print(
        "   Rayleigh resolution: "
        f"{rayleigh:.8f} c/d"
    )
    print(
        "   OpenStar: "
        f"{openstar_period:.8f} d "
        f"({openstar_frequency:.8f} c/d)"
    )
    print(
        "   Astropy: "
        f"{astropy_period:.8f} d "
        f"({astropy_frequency:.8f} c/d)"
    )
    print(
        "   VSX: "
        f"{vsx_period:.8f} d "
        f"({vsx_frequency:.8f} c/d)"
    )

    variants = detrend_variants(
        times,
        flux,
    )

    variant_results = {}

    raw_powers = None

    for variant_name, variant_flux in (
        variants.items()
    ):
        print()
        print(
            f"🔭 Full-grid LS — "
            f"{variant_name}"
        )

        model, powers = (
            run_full_lomb_scargle(
                times,
                variant_flux,
                frequencies,
            )
        )

        if variant_name == "raw":
            raw_powers = powers

        independent_peaks = (
            select_independent_peaks(
                frequencies,
                powers,
                minimum_separation=(
                    rayleigh
                    * INDEPENDENT_PEAK_SEPARATION_RAYLEIGH
                ),
                count=TOP_PEAK_COUNT,
            )
        )

        global_peak = (
            independent_peaks[0]
        )

        candidate_diagnostics = {}

        for label, candidate_frequency in (
            candidate_frequency_set(
                vsx_period
            ).items()
        ):
            if not (
                minimum_frequency
                <= candidate_frequency
                <= maximum_frequency
            ):
                candidate_diagnostics[
                    label
                ] = {
                    "inSearchRange": (
                        False
                    ),
                    "frequency": (
                        candidate_frequency
                    ),
                }

                continue

            grid = nearest_grid_diagnostic(
                frequencies,
                powers,
                candidate_frequency,
            )

            exact_power = float(
                model.power(
                    candidate_frequency
                )
            )

            grid[
                "inSearchRange"
            ] = True

            grid[
                "exactFrequencyPower"
            ] = (
                exact_power
            )

            candidate_diagnostics[
                label
            ] = grid

        vsx_diagnostic = (
            candidate_diagnostics[
                "VSX fundamental"
            ]
        )

        print(
            "   global winner: "
            f"{global_peak['periodDays']:.8f} d | "
            f"{global_peak['frequency']:.8f} c/d | "
            f"power {global_peak['power']:.8f}"
        )

        if vsx_diagnostic.get(
            "inSearchRange"
        ):
            print(
                "   VSX-frequency power: "
                f"{vsx_diagnostic['exactFrequencyPower']:.8f}"
            )
            print(
                "   VSX grid rank: "
                f"{vsx_diagnostic['rank']:,}/"
                f"{len(frequencies):,}"
            )
            print(
                "   VSX power percentile: "
                f"{vsx_diagnostic['percentile']:.5f}%"
            )

        print(
            "   top independent peaks:"
        )

        for peak_index, peak in enumerate(
            independent_peaks[
                :10
            ],
            start=1,
        ):
            print(
                f"      {peak_index:2d}. "
                f"{peak['periodDays']:.8f} d | "
                f"{peak['frequency']:.8f} c/d | "
                f"{peak['power']:.8f}"
            )

        variant_results[
            variant_name
        ] = {
            "globalPeak": (
                global_peak
            ),
            "topIndependentPeaks": (
                independent_peaks
            ),
            "catalogFrequencyDiagnostics": (
                candidate_diagnostics
            ),
        }

        if variant_name == "raw":
            save_periodogram_plot(
                output_dir
                / (
                    f"{blind_name.lower().replace(' ', '-')}"
                    "-raw-periodogram.png"
                ),
                frequencies,
                powers,
                title=(
                    f"{blind_name} raw "
                    "Lomb–Scargle"
                ),
                openstar_frequency=(
                    openstar_frequency
                ),
                vsx_frequency=(
                    vsx_frequency
                ),
            )

    print()
    print(
        "🎼 Multi-harmonic local fits"
    )

    multiharmonic = {}

    for seed_label, seed_frequency in (
        (
            (
                "OpenStar/Astropy winner",
                astropy_frequency,
            ),
            (
                "VSX fundamental",
                vsx_frequency,
            ),
        )
    ):
        seed_results = {}

        print()
        print(
            f"   seed: {seed_label}"
        )

        for nterms in range(
            1,
            MULTIHARMONIC_MAX_TERMS
            + 1,
        ):
            refined = (
                refine_multiharmonic(
                    times,
                    flux,
                    seed_frequency,
                    nterms,
                    rayleigh=(
                        rayleigh
                    ),
                    minimum_frequency=(
                        minimum_frequency
                    ),
                    maximum_frequency=(
                        maximum_frequency
                    ),
                )
            )

            seed_results[
                str(
                    nterms
                )
            ] = refined

            print(
                f"      nterms={nterms}: "
                f"{refined['bestPeriodDays']:.8f} d | "
                f"{refined['bestFrequency']:.8f} c/d | "
                f"score {refined['score']:.8f}"
            )

        multiharmonic[
            seed_label
        ] = seed_results

    candidate_periods = [
        (
            "OpenStar",
            openstar_period,
        ),
        (
            "Astropy",
            astropy_period,
        ),
        (
            "VSX",
            vsx_period,
        ),
        (
            "1/2 × VSX",
            0.5
            * vsx_period,
        ),
        (
            "2 × VSX",
            2.0
            * vsx_period,
        ),
        (
            "3 × VSX",
            3.0
            * vsx_period,
        ),
        (
            "4 × VSX",
            4.0
            * vsx_period,
        ),
    ]

    for peak_index, peak in enumerate(
        variant_results[
            "raw"
        ][
            "topIndependentPeaks"
        ][
            :10
        ],
        start=1,
    ):
        candidate_periods.append(
            (
                f"Raw LS peak {peak_index}",
                peak[
                    "periodDays"
                ],
            )
        )

    # Deduplicate periods before PDM/coherence scoring.
    unique_periods = []

    for label, period in (
        candidate_periods
    ):
        minimum_period = (
            1.0
            / maximum_frequency
        )

        maximum_period = (
            1.0
            / minimum_frequency
        )

        if not (
            minimum_period
            <= period
            <= maximum_period
        ):
            continue

        if any(
            abs(
                period
                - existing_period
            )
            / period
            < 1e-6
            for _,
            existing_period
            in unique_periods
        ):
            continue

        unique_periods.append(
            (
                label,
                period,
            )
        )

    print()
    print(
        "📉 PDM / fold-coherence candidates"
    )
    print(
        "   lower PDM theta = better"
    )
    print(
        "   lower fold RMS = better"
    )

    phase_metrics = (
        candidate_metrics(
            times,
            flux,
            unique_periods,
        )
    )

    phase_metrics.sort(
        key=lambda item: (
            math.inf
            if item[
                "pdmTheta"
            ] is None
            else item[
                "pdmTheta"
            ]
        )
    )

    for item in phase_metrics[
        :12
    ]:
        print(
            "      "
            f"{item['label']}: "
            f"{item['periodDays']:.8f} d | "
            f"PDM {item['pdmTheta']:.6f} | "
            f"fold RMS "
            f"{item['foldWithinBinRMS']:.6f}"
        )

    print()
    print(
        "🔁 Gap-aware ACF"
    )

    (
        acf_lags,
        acf,
        cadence,
        pair_count,
    ) = gap_aware_acf(
        times,
        flux,
    )

    acf_peaks = local_maxima(
        acf_lags,
        acf,
        minimum_x=(
            ACF_MIN_PERIOD_DAYS
        ),
        maximum_x=(
            ACF_MAX_PERIOD_DAYS
        ),
        count=(
            ACF_TOP_PEAK_COUNT
        ),
        minimum_separation=(
            max(
                cadence * 3.0,
                rayleigh,
            )
        ),
    )

    print(
        "   inferred cadence: "
        f"{cadence * 86400.0:.2f} s"
    )

    for peak_index, peak in enumerate(
        acf_peaks,
        start=1,
    ):
        print(
            f"      {peak_index:2d}. "
            f"{peak['periodDays']:.8f} d | "
            f"ACF {peak['acf']:.6f}"
        )

    slug = (
        blind_name
        .lower()
        .replace(
            " ",
            "-",
        )
    )

    save_fold_plot(
        output_dir
        / (
            f"{slug}-openstar-fold.png"
        ),
        times,
        flux,
        openstar_period,
        title=(
            f"{blind_name} OpenStar fold"
        ),
    )

    save_fold_plot(
        output_dir
        / (
            f"{slug}-vsx-fold.png"
        ),
        times,
        flux,
        vsx_period,
        title=(
            f"{blind_name} VSX fold"
        ),
    )

    save_acf_plot(
        output_dir
        / (
            f"{slug}-acf.png"
        ),
        acf_lags,
        acf,
        title=(
            f"{blind_name} gap-aware ACF"
        ),
        openstar_period=(
            openstar_period
        ),
        vsx_period=(
            vsx_period
        ),
    )

    return {
        "blindName": blind_name,
        "ticID": int(
            reveal_target[
                "ticID"
            ]
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
        "samples": int(
            len(times)
        ),
        "baselineDays": (
            baseline
        ),
        "rayleighResolution": (
            rayleigh
        ),
        "openstar": {
            "frequency": (
                openstar_frequency
            ),
            "periodDays": (
                openstar_period
            ),
            "power": (
                float(
                    frozen[
                        "openstarPower"
                    ]
                )
            ),
        },
        "astropy": {
            "frequency": (
                astropy_frequency
            ),
            "periodDays": (
                astropy_period
            ),
            "power": (
                float(
                    frozen[
                        "astropyPower"
                    ]
                )
            ),
        },
        "vsx": {
            "frequency": (
                vsx_frequency
            ),
            "periodDays": (
                vsx_period
            ),
        },
        "lombScargleVariants": (
            variant_results
        ),
        "multiharmonic": (
            multiharmonic
        ),
        "phaseMetrics": (
            phase_metrics
        ),
        "acf": {
            "cadenceDays": (
                cadence
            ),
            "topPeaks": (
                acf_peaks
            ),
        },
        "plots": {
            "rawPeriodogram": str(
                output_dir
                / (
                    f"{slug}"
                    "-raw-periodogram.png"
                )
            ),
            "openstarFold": str(
                output_dir
                / (
                    f"{slug}"
                    "-openstar-fold.png"
                )
            ),
            "vsxFold": str(
                output_dir
                / (
                    f"{slug}"
                    "-vsx-fold.png"
                )
            ),
            "acf": str(
                output_dir
                / (
                    f"{slug}"
                    "-acf.png"
                )
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deep diagnostic of Blind V2-F/G external "
            "period mismatches."
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
        "🔬 OpenStar F/G mismatch diagnostic v1"
    )
    print(
        "DEVELOPMENT ANALYSIS — blind results remain frozen."
    )
    print(
        "Tests: full LS + detrending + multi-harmonic + "
        "PDM/coherence + ACF."
    )

    results = []

    for blind_name, dataset_path in (
        TARGETS.items()
    ):
        if not dataset_path.exists():
            raise RuntimeError(
                f"Missing dataset: "
                f"{dataset_path}"
            )

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

        results.append(
            analyze_target(
                blind_name,
                dataset_path,
                reveal_target,
                args.output_dir,
            )
        )

    output_path = (
        args.output_dir
        / "fg-mismatch-diagnostic-summary.json"
    )

    with output_path.open(
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
                    "post-reveal F/G mismatch diagnostic"
                ),
                "developmentOnly": True,
                "targets": (
                    results
                ),
            },
            file,
            indent=2,
            allow_nan=False,
        )

    print()
    print()
    print(
        "🏁 F/G diagnostic complete"
    )
    print(
        f"   summary: {output_path}"
    )
    print(
        f"   plots:   {args.output_dir}"
    )
    print()
    print(
        "Interpretation rule:"
    )
    print(
        "   If VSX becomes competitive under detrending, "
        "multi-harmonic, PDM, or ACF, improve the resolver."
    )
    print(
        "   If VSX remains weak across all methods, this "
        "TESS sector itself does not strongly reproduce "
        "the catalog period."
    )


if __name__ == "__main__":
    main()
