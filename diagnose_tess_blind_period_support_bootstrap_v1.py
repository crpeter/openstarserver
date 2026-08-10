import math

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind A period-support + bootstrap diagnostic
#
# Uses the independent-phase model:
#   - one shared trial frequency
#   - each TESS sector gets its own mean/amplitude/phase
#
# Reports:
#   - exact 99% / 98% / 95% support bands
#   - TARS position inside those bands
#   - moving-block residual bootstrap distribution of the
#     preferred independent-phase period
#
# This is a stability/support diagnostic, not a formal
# frequentist confidence interval calculation.
# ============================================================

TARGET_NAME = "Blind A"
TARGET_QUERY = "TIC 25165839"
SECTORS = (1, 28)

PREFERRED_AUTHOR = "SPOC"
PREFERRED_EXPTIME_SECONDS = 120

MINIMUM_FREQUENCY = 0.03
MAXIMUM_FREQUENCY = 5.0
TOTAL_FREQUENCIES = 4_194_304

WINDOW_MIN_PERIOD_DAYS = 7.0
WINDOW_MAX_PERIOD_DAYS = 12.0

TARS_ADOPTED_PERIOD_DAYS = 9.0381
TARS_QUOTED_UNCERTAINTY_DAYS = 0.3342

SUPPORT_THRESHOLDS = (
    0.99,
    0.98,
    0.95,
)

BOOTSTRAP_REPLICATES = 64
BOOTSTRAP_BLOCK_SAMPLES = 64
BOOTSTRAP_GRID_STRIDE = 8
BOOTSTRAP_RANDOM_SEED = 20260810


# ============================================================
# Frequency grid
# ============================================================


def frequency_step() -> float:
    return (
        MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY
    ) / TOTAL_FREQUENCIES


def full_analysis_grid() -> tuple[np.ndarray, np.ndarray]:
    step = frequency_step()

    low_frequency = 1.0 / WINDOW_MAX_PERIOD_DAYS
    high_frequency = 1.0 / WINDOW_MIN_PERIOD_DAYS

    first_index = max(
        0,
        int(math.ceil(
            (low_frequency - MINIMUM_FREQUENCY) / step
        )),
    )
    last_index = min(
        TOTAL_FREQUENCIES - 1,
        int(math.floor(
            (high_frequency - MINIMUM_FREQUENCY) / step
        )),
    )

    absolute_indices = np.arange(
        first_index,
        last_index + 1,
        dtype=np.int64,
    )

    frequencies = (
        MINIMUM_FREQUENCY
        + absolute_indices.astype(np.float64) * step
    )

    return absolute_indices, frequencies


# ============================================================
# TESS loading
# ============================================================


def load_sector(sector: int) -> dict:
    print()
    print(f"🔭 Sector {sector}")

    search = lk.search_lightcurve(
        TARGET_QUERY,
        mission="TESS",
        author=PREFERRED_AUTHOR,
        sector=sector,
        exptime=PREFERRED_EXPTIME_SECONDS,
    )

    if len(search) == 0:
        raise RuntimeError(
            f"No {PREFERRED_AUTHOR} "
            f"{PREFERRED_EXPTIME_SECONDS}s light curve "
            f"found for Sector {sector}."
        )

    light_curve = search[0].download(
        quality_bitmask="default"
    )

    if light_curve is None:
        raise RuntimeError(
            f"Download failed for Sector {sector}."
        )

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

    order = np.argsort(times)
    times = times[order]
    flux = flux[order]

    if len(times) == 0:
        raise RuntimeError(
            f"Sector {sector} has no finite samples."
        )

    mean = float(np.mean(flux))
    stddev = float(np.std(flux))

    if not math.isfinite(stddev) or stddev <= 0:
        raise RuntimeError(
            f"Sector {sector} has invalid flux stddev."
        )

    flux = (
        (flux - mean) / stddev
    ).astype(np.float64)

    print(f"   finite samples: {len(times)}")
    print(f"   first time: {times[0]:.8f}")
    print(f"   last time: {times[-1]:.8f}")
    print(
        "   baseline: "
        f"{times[-1] - times[0]:.4f} days"
    )

    return {
        "sector": int(sector),
        "times": times,
        "flux": flux,
    }


# ============================================================
# Independent-phase periodogram
# ============================================================


def sector_periodogram(
    times: np.ndarray,
    flux: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, float]:
    model = LombScargle(
        times,
        flux,
        fit_mean=True,
        center_data=True,
    )

    powers = model.power(
        frequencies,
    )

    reference_chi2 = float(
        np.sum(
            (
                flux
                - np.mean(flux)
            ) ** 2
        )
    )

    return powers, reference_chi2


def independent_phase_power(
    sector_data: dict[int, dict],
    frequencies: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    sector_powers = {}
    sector_reference_chi2 = {}

    for sector in SECTORS:
        powers, reference_chi2 = (
            sector_periodogram(
                sector_data[sector]["times"],
                sector_data[sector]["flux"],
                frequencies,
            )
        )

        sector_powers[sector] = powers
        sector_reference_chi2[
            sector
        ] = reference_chi2

    total_reference_chi2 = sum(
        sector_reference_chi2.values()
    )

    combined = (
        sector_powers[1]
        * sector_reference_chi2[1]
        + sector_powers[28]
        * sector_reference_chi2[28]
    ) / total_reference_chi2

    return combined, sector_powers


# ============================================================
# Support intervals
# ============================================================


def contiguous_true_runs(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return []

    breaks = np.flatnonzero(
        np.diff(indices) > 1
    )

    starts = np.concatenate(
        (
            np.asarray([0]),
            breaks + 1,
        )
    )
    ends = np.concatenate(
        (
            breaks,
            np.asarray(
                [len(indices) - 1]
            ),
        )
    )

    return [
        (
            int(indices[start]),
            int(indices[end]),
        )
        for start, end
        in zip(starts, ends)
    ]


def period_interval_from_indices(
    frequencies: np.ndarray,
    start_index: int,
    end_index: int,
) -> tuple[float, float]:
    high_period = (
        1.0 / float(
            frequencies[start_index]
        )
    )
    low_period = (
        1.0 / float(
            frequencies[end_index]
        )
    )

    return low_period, high_period


def print_support_profile(
    frequencies: np.ndarray,
    powers: np.ndarray,
    best_index: int,
    tars_index: int,
):
    maximum = float(
        powers[best_index]
    )

    print()
    print("📏 PERIOD-SUPPORT PROFILE")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "These are relative-power support bands, "
        "not formal confidence intervals."
    )

    for threshold in SUPPORT_THRESHOLDS:
        mask = (
            powers
            >= maximum * threshold
        )

        runs = contiguous_true_runs(mask)

        winner_run = None

        for run in runs:
            if (
                run[0]
                <= best_index
                <= run[1]
            ):
                winner_run = run
                break

        tars_supported = bool(
            mask[tars_index]
        )

        print()
        print(
            f"🔹 ≥ {threshold * 100:.0f}% of maximum"
        )
        print(
            f"   connected support regions: {len(runs)}"
        )
        print(
            "   TARS 9.0381d included: "
            f"{'YES' if tars_supported else 'NO'}"
        )

        if winner_run is not None:
            low_period, high_period = (
                period_interval_from_indices(
                    frequencies,
                    winner_run[0],
                    winner_run[1],
                )
            )

            print(
                "   winner-containing interval: "
                f"{low_period:.8f} - "
                f"{high_period:.8f} days"
            )
            print(
                "   interval width: "
                f"{high_period - low_period:.8f} days"
            )

        if len(runs) > 1:
            print("   all intervals:")

            for number, run in enumerate(
                runs,
                start=1,
            ):
                low_period, high_period = (
                    period_interval_from_indices(
                        frequencies,
                        run[0],
                        run[1],
                    )
                )

                print(
                    f"      {number}. "
                    f"{low_period:.8f} - "
                    f"{high_period:.8f} days"
                )


# ============================================================
# Sinusoid fit + moving-block residual bootstrap
# ============================================================


def fitted_model_and_residuals(
    times: np.ndarray,
    flux: np.ndarray,
    frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    omega_t = (
        2.0
        * math.pi
        * frequency
        * times
    )

    design = np.column_stack(
        (
            np.sin(omega_t),
            np.cos(omega_t),
            np.ones_like(omega_t),
        )
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        flux,
        rcond=None,
    )

    fitted = (
        design @ coefficients
    )

    residuals = (
        flux - fitted
    )

    residuals = (
        residuals
        - np.mean(residuals)
    )

    return fitted, residuals


def circular_block_resample(
    values: np.ndarray,
    block_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(values)

    if count == 0:
        return values.copy()

    block_samples = max(
        1,
        min(
            int(block_samples),
            count,
        ),
    )

    block_count = int(
        math.ceil(
            count / block_samples
        )
    )

    starts = rng.integers(
        0,
        count,
        size=block_count,
    )

    output = np.empty(
        block_count * block_samples,
        dtype=np.float64,
    )

    cursor = 0

    offsets = np.arange(
        block_samples,
        dtype=np.int64,
    )

    for start in starts:
        indices = (
            int(start)
            + offsets
        ) % count

        output[
            cursor:cursor + block_samples
        ] = values[indices]

        cursor += block_samples

    return output[:count]


def bootstrap_independent_power(
    sector_data: dict[int, dict],
    bootstrap_flux: dict[int, np.ndarray],
    frequencies: np.ndarray,
) -> np.ndarray:
    powers_by_sector = {}
    chi2_by_sector = {}

    for sector in SECTORS:
        flux = bootstrap_flux[sector]

        powers, reference_chi2 = (
            sector_periodogram(
                sector_data[sector]["times"],
                flux,
                frequencies,
            )
        )

        powers_by_sector[sector] = powers
        chi2_by_sector[
            sector
        ] = reference_chi2

    total_chi2 = sum(
        chi2_by_sector.values()
    )

    return (
        powers_by_sector[1]
        * chi2_by_sector[1]
        + powers_by_sector[28]
        * chi2_by_sector[28]
    ) / total_chi2


def refine_bootstrap_peak(
    sector_data: dict[int, dict],
    bootstrap_flux: dict[int, np.ndarray],
    full_frequencies: np.ndarray,
    coarse_best_full_position: int,
) -> tuple[float, float]:
    radius = (
        BOOTSTRAP_GRID_STRIDE * 2
    )

    start = max(
        0,
        coarse_best_full_position - radius,
    )
    end = min(
        len(full_frequencies),
        coarse_best_full_position + radius + 1,
    )

    fine_frequencies = (
        full_frequencies[start:end]
    )

    fine_power = bootstrap_independent_power(
        sector_data,
        bootstrap_flux,
        fine_frequencies,
    )

    local_best = int(
        np.nanargmax(
            fine_power
        )
    )

    frequency = float(
        fine_frequencies[
            local_best
        ]
    )

    power = float(
        fine_power[
            local_best
        ]
    )

    return frequency, power


def percentile(
    values: np.ndarray,
    q: float,
) -> float:
    return float(
        np.percentile(
            values,
            q,
        )
    )


def run_bootstrap(
    sector_data: dict[int, dict],
    full_frequencies: np.ndarray,
    best_frequency: float,
):
    print()
    print("🎲 MOVING-BLOCK RESIDUAL BOOTSTRAP")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"replicates: {BOOTSTRAP_REPLICATES}"
    )
    print(
        "block size: "
        f"{BOOTSTRAP_BLOCK_SAMPLES} samples"
    )
    print(
        "coarse grid stride: "
        f"{BOOTSTRAP_GRID_STRIDE} OpenStar bins"
    )
    print(
        "Each replicate is locally refined on the full grid."
    )

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_SEED
    )

    fitted = {}
    residuals = {}

    for sector in SECTORS:
        sector_fitted, sector_residuals = (
            fitted_model_and_residuals(
                sector_data[sector]["times"],
                sector_data[sector]["flux"],
                best_frequency,
            )
        )

        fitted[sector] = sector_fitted
        residuals[sector] = sector_residuals

    coarse_positions = np.arange(
        0,
        len(full_frequencies),
        BOOTSTRAP_GRID_STRIDE,
        dtype=np.int64,
    )

    if (
        coarse_positions[-1]
        != len(full_frequencies) - 1
    ):
        coarse_positions = np.append(
            coarse_positions,
            len(full_frequencies) - 1,
        )

    coarse_frequencies = (
        full_frequencies[
            coarse_positions
        ]
    )

    tars_frequency = (
        1.0 / TARS_ADOPTED_PERIOD_DAYS
    )

    bootstrap_periods = []
    tars_relative_support = []

    progress_interval = max(
        1,
        BOOTSTRAP_REPLICATES // 8,
    )

    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):
        bootstrap_flux = {}

        for sector in SECTORS:
            sampled_residuals = (
                circular_block_resample(
                    residuals[sector],
                    BOOTSTRAP_BLOCK_SAMPLES,
                    rng,
                )
            )

            bootstrap_flux[sector] = (
                fitted[sector]
                + sampled_residuals
            )

        coarse_power = (
            bootstrap_independent_power(
                sector_data,
                bootstrap_flux,
                coarse_frequencies,
            )
        )

        coarse_best_index = int(
            np.nanargmax(
                coarse_power
            )
        )

        coarse_best_full_position = int(
            coarse_positions[
                coarse_best_index
            ]
        )

        refined_frequency, refined_power = (
            refine_bootstrap_peak(
                sector_data,
                bootstrap_flux,
                full_frequencies,
                coarse_best_full_position,
            )
        )

        bootstrap_periods.append(
            1.0 / refined_frequency
        )

        # Evaluate exact TARS frequency directly rather than
        # relying on the coarse bootstrap grid.
        tars_power = float(
            bootstrap_independent_power(
                sector_data,
                bootstrap_flux,
                np.asarray(
                    [tars_frequency],
                    dtype=np.float64,
                ),
            )[0]
        )

        tars_relative_support.append(
            (
                tars_power / refined_power
                if refined_power != 0
                else math.nan
            )
        )

        if (
            (replicate + 1)
            % progress_interval
            == 0
            or replicate
            == BOOTSTRAP_REPLICATES - 1
        ):
            print(
                "   completed "
                f"{replicate + 1}/"
                f"{BOOTSTRAP_REPLICATES}"
            )

    periods = np.asarray(
        bootstrap_periods,
        dtype=np.float64,
    )

    tars_support = np.asarray(
        tars_relative_support,
        dtype=np.float64,
    )

    print()
    print("📊 Bootstrap preferred-period distribution")
    print(
        "   median: "
        f"{percentile(periods, 50):.8f} days"
    )
    print(
        "   central 68%: "
        f"{percentile(periods, 16):.8f} - "
        f"{percentile(periods, 84):.8f} days"
    )
    print(
        "   central 95%: "
        f"{percentile(periods, 2.5):.8f} - "
        f"{percentile(periods, 97.5):.8f} days"
    )
    print(
        "   minimum winner: "
        f"{float(np.min(periods)):.8f} days"
    )
    print(
        "   maximum winner: "
        f"{float(np.max(periods)):.8f} days"
    )

    quoted_low = (
        TARS_ADOPTED_PERIOD_DAYS
        - TARS_QUOTED_UNCERTAINTY_DAYS
    )
    quoted_high = (
        TARS_ADOPTED_PERIOD_DAYS
        + TARS_QUOTED_UNCERTAINTY_DAYS
    )

    in_tars_interval = np.mean(
        (
            periods >= quoted_low
        )
        & (
            periods <= quoted_high
        )
    )

    print()
    print("🎯 TARS within bootstrap")
    print(
        "   TARS adopted period: "
        f"{TARS_ADOPTED_PERIOD_DAYS:.8f} days"
    )
    print(
        "   quoted TARS interval: "
        f"{quoted_low:.8f} - "
        f"{quoted_high:.8f} days"
    )
    print(
        "   bootstrap winners inside quoted TARS interval: "
        f"{in_tars_interval * 100:.2f}%"
    )

    for threshold in SUPPORT_THRESHOLDS:
        fraction = float(
            np.mean(
                tars_support
                >= threshold
            )
        )

        print(
            "   replicates where exact TARS has "
            f"≥{threshold * 100:.0f}% of replicate max: "
            f"{fraction * 100:.2f}%"
        )

    print(
        "   median TARS relative support: "
        f"{percentile(tars_support, 50) * 100:.2f}%"
    )

    return periods, tars_support


# ============================================================
# Main
# ============================================================


def main():
    print(
        "🔬 OpenStar Blind A Period-Support + Bootstrap Diagnostic"
    )
    print(f"target: {TARGET_NAME}")
    print("sectors: 1, 28")
    print(
        "model: shared frequency, independent "
        "mean/amplitude/phase per sector"
    )
    print(
        "analysis window: "
        f"{WINDOW_MIN_PERIOD_DAYS:.1f}-"
        f"{WINDOW_MAX_PERIOD_DAYS:.1f} days"
    )
    print(
        "frequency grid step: "
        f"{frequency_step():.12f} cycles/day"
    )
    print(
        "this diagnostic does not modify "
        "OpenStar datasets/projects"
    )

    sector_data = {
        sector: load_sector(sector)
        for sector in SECTORS
    }

    _, frequencies = (
        full_analysis_grid()
    )

    print()
    print(
        "📐 Full support-grid bins: "
        f"{len(frequencies):,}"
    )

    print()
    print(
        "🧪 Computing exact independent-phase support profile"
    )

    powers, sector_powers = (
        independent_phase_power(
            sector_data,
            frequencies,
        )
    )

    best_index = int(
        np.nanargmax(
            powers
        )
    )

    best_frequency = float(
        frequencies[
            best_index
        ]
    )
    best_period = (
        1.0 / best_frequency
    )
    best_power = float(
        powers[
            best_index
        ]
    )

    tars_frequency = (
        1.0 / TARS_ADOPTED_PERIOD_DAYS
    )
    tars_index = int(
        np.argmin(
            np.abs(
                frequencies
                - tars_frequency
            )
        )
    )

    tars_grid_period = (
        1.0
        / float(
            frequencies[
                tars_index
            ]
        )
    )
    tars_power = float(
        powers[
            tars_index
        ]
    )

    print()
    print("⭐ Independent-phase maximum")
    print(
        "   frequency: "
        f"{best_frequency:.8f} cycles/day"
    )
    print(
        "   period: "
        f"{best_period:.8f} days"
    )
    print(
        "   power: "
        f"{best_power:.8f}"
    )
    print(
        "   Sector 1 power there: "
        f"{float(sector_powers[1][best_index]):.8f}"
    )
    print(
        "   Sector 28 power there: "
        f"{float(sector_powers[28][best_index]):.8f}"
    )

    print()
    print("🎯 TARS exact-grid support")
    print(
        "   adopted period: "
        f"{TARS_ADOPTED_PERIOD_DAYS:.8f} days"
    )
    print(
        "   nearest grid period: "
        f"{tars_grid_period:.8f} days"
    )
    print(
        "   power: "
        f"{tars_power:.8f}"
    )
    print(
        "   relative to maximum: "
        f"{tars_power / best_power * 100:.2f}%"
    )

    print_support_profile(
        frequencies,
        powers,
        best_index,
        tars_index,
    )

    periods, tars_support = (
        run_bootstrap(
            sector_data,
            frequencies,
            best_frequency,
        )
    )

    print()
    print()
    print("🏁 PERIOD-SUPPORT SUMMARY")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "best independent-phase period: "
        f"{best_period:.8f} days"
    )
    print(
        "TARS relative support in original data: "
        f"{tars_power / best_power * 100:.2f}%"
    )
    print(
        "bootstrap median winner: "
        f"{percentile(periods, 50):.8f} days"
    )
    print(
        "bootstrap central 68%: "
        f"{percentile(periods, 16):.8f} - "
        f"{percentile(periods, 84):.8f} days"
    )
    print(
        "bootstrap central 95%: "
        f"{percentile(periods, 2.5):.8f} - "
        f"{percentile(periods, 97.5):.8f} days"
    )
    print(
        "median bootstrap TARS support: "
        f"{percentile(tars_support, 50) * 100:.2f}%"
    )
    print()
    print(
        "Reminder: relative-power bands and this block bootstrap "
        "are descriptive stability measures, not formal publication-grade "
        "confidence intervals."
    )
    print()
    print("✅ Diagnostic complete")


if __name__ == "__main__":
    main()
