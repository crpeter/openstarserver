import math

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind A two-hypothesis injection/recovery diagnostic
#
# Hypothesis A:
#   TARS adopted period = 9.0381 d
#
# Hypothesis B:
#   TESS independent-phase best period = 9.29803505 d
#
# For each hypothesis:
#   1. Fit each TESS sector independently at the injected period
#   2. Resample residuals with circular moving blocks
#   3. Add those residuals back to the fitted signal
#   4. Re-run the independent-phase period search
#   5. Compare support at A versus B
#
# This is a simulation-based discrimination/stability test.
# It is not a formal Bayes factor or publication-grade p-value.
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

HYPOTHESIS_A_NAME = "TARS"
HYPOTHESIS_A_PERIOD = 9.0381

HYPOTHESIS_B_NAME = "TESS-best"
HYPOTHESIS_B_PERIOD = 9.29803505

REPLICATES_PER_HYPOTHESIS = 48
BOOTSTRAP_BLOCK_SAMPLES = 64
COARSE_GRID_STRIDE = 8
RANDOM_SEED = 20260810


# ============================================================
# Frequency grid
# ============================================================


def frequency_step() -> float:
    return (
        MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY
    ) / TOTAL_FREQUENCIES


def full_analysis_grid() -> np.ndarray:
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

    indices = np.arange(
        first_index,
        last_index + 1,
        dtype=np.int64,
    )

    return (
        MINIMUM_FREQUENCY
        + indices.astype(np.float64) * step
    )


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
# Independent-phase score
# ============================================================


def sector_power(
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
    flux_by_sector: dict[int, np.ndarray],
    frequencies: np.ndarray,
) -> np.ndarray:
    powers = {}
    chi2 = {}

    for sector in SECTORS:
        sector_powers, reference_chi2 = (
            sector_power(
                sector_data[sector]["times"],
                flux_by_sector[sector],
                frequencies,
            )
        )

        powers[sector] = sector_powers
        chi2[sector] = reference_chi2

    total_chi2 = sum(
        chi2.values()
    )

    return (
        powers[1] * chi2[1]
        + powers[28] * chi2[28]
    ) / total_chi2


# ============================================================
# Fitting and residual resampling
# ============================================================


def fit_at_frequency(
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

    fitted = design @ coefficients

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

    offsets = np.arange(
        block_samples,
        dtype=np.int64,
    )

    output = np.empty(
        block_count * block_samples,
        dtype=np.float64,
    )

    cursor = 0

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


# ============================================================
# Full-window recovery
# ============================================================


def refine_winner(
    sector_data: dict[int, dict],
    flux_by_sector: dict[int, np.ndarray],
    full_frequencies: np.ndarray,
    coarse_positions: np.ndarray,
    coarse_power: np.ndarray,
) -> tuple[float, float]:
    coarse_best = int(
        np.nanargmax(
            coarse_power
        )
    )

    full_position = int(
        coarse_positions[
            coarse_best
        ]
    )

    radius = (
        COARSE_GRID_STRIDE * 2
    )

    start = max(
        0,
        full_position - radius,
    )
    end = min(
        len(full_frequencies),
        full_position + radius + 1,
    )

    fine_frequencies = (
        full_frequencies[start:end]
    )

    fine_power = (
        independent_phase_power(
            sector_data,
            flux_by_sector,
            fine_frequencies,
        )
    )

    fine_best = int(
        np.nanargmax(
            fine_power
        )
    )

    return (
        float(
            fine_frequencies[
                fine_best
            ]
        ),
        float(
            fine_power[
                fine_best
            ]
        ),
    )


# ============================================================
# Statistics helpers
# ============================================================


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


def describe_distribution(
    label: str,
    values: np.ndarray,
    suffix: str = "",
):
    print(f"   {label}:")
    print(
        "      median: "
        f"{percentile(values, 50):.8f}{suffix}"
    )
    print(
        "      central 68%: "
        f"{percentile(values, 16):.8f} - "
        f"{percentile(values, 84):.8f}{suffix}"
    )
    print(
        "      central 95%: "
        f"{percentile(values, 2.5):.8f} - "
        f"{percentile(values, 97.5):.8f}{suffix}"
    )


# ============================================================
# Simulation
# ============================================================


def prepare_hypothesis_model(
    sector_data: dict[int, dict],
    period: float,
) -> dict:
    frequency = 1.0 / period

    fitted = {}
    residuals = {}

    for sector in SECTORS:
        sector_fitted, sector_residuals = (
            fit_at_frequency(
                sector_data[sector]["times"],
                sector_data[sector]["flux"],
                frequency,
            )
        )

        fitted[sector] = sector_fitted
        residuals[sector] = sector_residuals

    return {
        "period": period,
        "frequency": frequency,
        "fitted": fitted,
        "residuals": residuals,
    }


def run_hypothesis(
    *,
    name: str,
    model: dict,
    sector_data: dict[int, dict],
    hypothesis_frequencies: np.ndarray,
    full_frequencies: np.ndarray,
    coarse_positions: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    print()
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"🧪 Injecting {name}: "
        f"{model['period']:.8f} days"
    )
    print(
        "════════════════════════════════════════════════════════"
    )

    coarse_frequencies = (
        full_frequencies[
            coarse_positions
        ]
    )

    deltas = []
    recovered_periods = []
    recovered_powers = []
    correct_fixed_choice = []
    correct_winner_closeness = []

    progress_interval = 8

    for replicate in range(
        REPLICATES_PER_HYPOTHESIS
    ):
        synthetic_flux = {}

        for sector in SECTORS:
            sampled_residuals = (
                circular_block_resample(
                    model["residuals"][sector],
                    BOOTSTRAP_BLOCK_SAMPLES,
                    rng,
                )
            )

            synthetic_flux[sector] = (
                model["fitted"][sector]
                + sampled_residuals
            )

        fixed_power = independent_phase_power(
            sector_data,
            synthetic_flux,
            hypothesis_frequencies,
        )

        power_a = float(
            fixed_power[0]
        )
        power_b = float(
            fixed_power[1]
        )

        delta = (
            power_b - power_a
        )

        deltas.append(delta)

        if name == HYPOTHESIS_A_NAME:
            correct_fixed_choice.append(
                power_a > power_b
            )
        else:
            correct_fixed_choice.append(
                power_b > power_a
            )

        coarse_power = (
            independent_phase_power(
                sector_data,
                synthetic_flux,
                coarse_frequencies,
            )
        )

        winner_frequency, winner_power = (
            refine_winner(
                sector_data,
                synthetic_flux,
                full_frequencies,
                coarse_positions,
                coarse_power,
            )
        )

        winner_period = (
            1.0 / winner_frequency
        )

        recovered_periods.append(
            winner_period
        )
        recovered_powers.append(
            winner_power
        )

        distance_a = abs(
            winner_period
            - HYPOTHESIS_A_PERIOD
        )
        distance_b = abs(
            winner_period
            - HYPOTHESIS_B_PERIOD
        )

        if name == HYPOTHESIS_A_NAME:
            correct_winner_closeness.append(
                distance_a < distance_b
            )
        else:
            correct_winner_closeness.append(
                distance_b < distance_a
            )

        if (
            (replicate + 1)
            % progress_interval
            == 0
            or replicate
            == REPLICATES_PER_HYPOTHESIS - 1
        ):
            print(
                "   completed "
                f"{replicate + 1}/"
                f"{REPLICATES_PER_HYPOTHESIS}"
            )

    return {
        "name": name,
        "period": model["period"],
        "deltas": np.asarray(
            deltas,
            dtype=np.float64,
        ),
        "recoveredPeriods": np.asarray(
            recovered_periods,
            dtype=np.float64,
        ),
        "recoveredPowers": np.asarray(
            recovered_powers,
            dtype=np.float64,
        ),
        "fixedCorrect": np.asarray(
            correct_fixed_choice,
            dtype=bool,
        ),
        "winnerCloserCorrect": np.asarray(
            correct_winner_closeness,
            dtype=bool,
        ),
    }


def print_hypothesis_summary(
    result: dict,
):
    print()
    print(
        f"📊 {result['name']} injection summary"
    )

    describe_distribution(
        "Δsupport = power(B) - power(A)",
        result["deltas"],
    )

    describe_distribution(
        "recovered full-window period",
        result["recoveredPeriods"],
        " days",
    )

    print(
        "   fixed A-vs-B comparison chooses "
        "the injected hypothesis: "
        f"{np.mean(result['fixedCorrect']) * 100:.2f}%"
    )

    print(
        "   full-window winner is closer to "
        "the injected hypothesis: "
        f"{np.mean(result['winnerCloserCorrect']) * 100:.2f}%"
    )


# ============================================================
# Main
# ============================================================


def main():
    print(
        "🔬 OpenStar Blind A Two-Hypothesis Injection/Recovery"
    )
    print(f"target: {TARGET_NAME}")
    print("sectors: 1, 28")
    print()
    print(
        f"Hypothesis A ({HYPOTHESIS_A_NAME}): "
        f"{HYPOTHESIS_A_PERIOD:.8f} days"
    )
    print(
        f"Hypothesis B ({HYPOTHESIS_B_NAME}): "
        f"{HYPOTHESIS_B_PERIOD:.8f} days"
    )
    print(
        "statistic: Δsupport = "
        "independent-phase power(B) - power(A)"
    )
    print(
        "positive Δ favors B; negative Δ favors A"
    )
    print(
        "simulation noise: moving-block residual resampling "
        "within each sector"
    )
    print(
        "this diagnostic does not modify "
        "OpenStar datasets/projects"
    )

    sector_data = {
        sector: load_sector(sector)
        for sector in SECTORS
    }

    full_frequencies = (
        full_analysis_grid()
    )

    coarse_positions = np.arange(
        0,
        len(full_frequencies),
        COARSE_GRID_STRIDE,
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

    frequency_a = (
        1.0 / HYPOTHESIS_A_PERIOD
    )
    frequency_b = (
        1.0 / HYPOTHESIS_B_PERIOD
    )

    hypothesis_frequencies = np.asarray(
        [
            frequency_a,
            frequency_b,
        ],
        dtype=np.float64,
    )

    observed_flux = {
        sector: sector_data[sector]["flux"]
        for sector in SECTORS
    }

    observed_fixed_power = (
        independent_phase_power(
            sector_data,
            observed_flux,
            hypothesis_frequencies,
        )
    )

    observed_power_a = float(
        observed_fixed_power[0]
    )
    observed_power_b = float(
        observed_fixed_power[1]
    )
    observed_delta = (
        observed_power_b
        - observed_power_a
    )

    print()
    print("🎯 Observed data")
    print(
        f"   power at A ({HYPOTHESIS_A_NAME}): "
        f"{observed_power_a:.8f}"
    )
    print(
        f"   power at B ({HYPOTHESIS_B_NAME}): "
        f"{observed_power_b:.8f}"
    )
    print(
        "   observed Δsupport B-A: "
        f"{observed_delta:+.8f}"
    )
    print(
        "   fixed comparison favors: "
        f"{HYPOTHESIS_B_NAME if observed_delta > 0 else HYPOTHESIS_A_NAME}"
    )

    observed_coarse_power = (
        independent_phase_power(
            sector_data,
            observed_flux,
            full_frequencies[
                coarse_positions
            ],
        )
    )

    observed_winner_frequency, observed_winner_power = (
        refine_winner(
            sector_data,
            observed_flux,
            full_frequencies,
            coarse_positions,
            observed_coarse_power,
        )
    )

    observed_winner_period = (
        1.0 / observed_winner_frequency
    )

    print(
        "   full-window independent-phase winner: "
        f"{observed_winner_period:.8f} days"
    )
    print(
        "   winner power: "
        f"{observed_winner_power:.8f}"
    )

    model_a = prepare_hypothesis_model(
        sector_data,
        HYPOTHESIS_A_PERIOD,
    )
    model_b = prepare_hypothesis_model(
        sector_data,
        HYPOTHESIS_B_PERIOD,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    result_a = run_hypothesis(
        name=HYPOTHESIS_A_NAME,
        model=model_a,
        sector_data=sector_data,
        hypothesis_frequencies=hypothesis_frequencies,
        full_frequencies=full_frequencies,
        coarse_positions=coarse_positions,
        rng=rng,
    )

    result_b = run_hypothesis(
        name=HYPOTHESIS_B_NAME,
        model=model_b,
        sector_data=sector_data,
        hypothesis_frequencies=hypothesis_frequencies,
        full_frequencies=full_frequencies,
        coarse_positions=coarse_positions,
        rng=rng,
    )

    print_hypothesis_summary(
        result_a
    )
    print_hypothesis_summary(
        result_b
    )

    a_deltas = result_a[
        "deltas"
    ]
    b_deltas = result_b[
        "deltas"
    ]

    # Direct sign-based discrimination:
    #   delta < 0 -> A
    #   delta > 0 -> B
    a_misclassification = float(
        np.mean(
            a_deltas > 0
        )
    )
    b_misclassification = float(
        np.mean(
            b_deltas < 0
        )
    )

    if observed_delta >= 0:
        tail_under_a = float(
            np.mean(
                a_deltas
                >= observed_delta
            )
        )
        tail_under_b = float(
            np.mean(
                b_deltas
                <= observed_delta
            )
        )
    else:
        tail_under_a = float(
            np.mean(
                a_deltas
                <= observed_delta
            )
        )
        tail_under_b = float(
            np.mean(
                b_deltas
                >= observed_delta
            )
        )

    print()
    print()
    print(
        "🏁 TWO-HYPOTHESIS DISCRIMINATION SUMMARY"
    )
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"A = {HYPOTHESIS_A_NAME} "
        f"{HYPOTHESIS_A_PERIOD:.8f} d"
    )
    print(
        f"B = {HYPOTHESIS_B_NAME} "
        f"{HYPOTHESIS_B_PERIOD:.8f} d"
    )
    print()
    print(
        "observed power(A): "
        f"{observed_power_a:.8f}"
    )
    print(
        "observed power(B): "
        f"{observed_power_b:.8f}"
    )
    print(
        "observed Δsupport B-A: "
        f"{observed_delta:+.8f}"
    )
    print(
        "observed full-window winner: "
        f"{observed_winner_period:.8f} days"
    )
    print()
    print(
        "A injections misclassified as B by fixed support: "
        f"{a_misclassification * 100:.2f}%"
    )
    print(
        "B injections misclassified as A by fixed support: "
        f"{b_misclassification * 100:.2f}%"
    )
    print(
        "mean sign-classification error: "
        f"{(a_misclassification + b_misclassification) * 50:.2f}%"
    )
    print()
    print(
        "fraction of A injections with a Δsupport "
        "at least as extreme as observed in the observed direction: "
        f"{tail_under_a * 100:.2f}%"
    )
    print(
        "fraction of B injections on the opposite side "
        "of the observed Δsupport: "
        f"{tail_under_b * 100:.2f}%"
    )
    print()
    print(
        "A full-window recoveries closer to A: "
        f"{np.mean(result_a['winnerCloserCorrect']) * 100:.2f}%"
    )
    print(
        "B full-window recoveries closer to B: "
        f"{np.mean(result_b['winnerCloserCorrect']) * 100:.2f}%"
    )
    print()
    print(
        "Interpretation guide:"
    )
    print(
        "If the A and B Δsupport distributions overlap heavily "
        "and sign misclassification is common, these two TESS "
        "windows cannot reliably distinguish the periods."
    )
    print(
        "If the distributions separate cleanly and each injected "
        "period is recovered with low misclassification, the data "
        "do contain enough information to discriminate them."
    )
    print()
    print(
        "Reminder: this residual-injection experiment is a "
        "descriptive simulation test, not a formal Bayes factor "
        "or publication-grade significance calculation."
    )
    print()
    print("✅ Diagnostic complete")


if __name__ == "__main__":
    main()
