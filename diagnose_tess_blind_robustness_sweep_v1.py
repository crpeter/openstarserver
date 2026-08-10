import math

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind A robustness sweep
#
# Tests whether discrimination between:
#   A = TARS adopted period
#   B = TESS independent-phase best period
#
# survives:
#   - different residual correlation lengths
#   - a more flexible two-harmonic light-curve model
#
# Each configuration:
#   1. Fits each sector independently under A and B
#   2. Resamples residuals in circular moving blocks
#   3. Injects fitted signal + correlated residuals
#   4. Scores both fixed hypotheses using the same model
#   5. Measures sign misclassification and the observed-data tail
#
# The full 7-12 day periodogram is also computed once for each
# harmonic order on the real data.
#
# This is a descriptive robustness simulation, not a formal
# Bayes factor or publication-grade significance calculation.
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

HARMONIC_ORDERS = (1, 2)

# At 120-second cadence these correspond approximately to:
# 16   -> 32 minutes
# 64   -> 2.1 hours
# 256  -> 8.5 hours
# 1024 -> 34.1 hours
BLOCK_SIZES = (16, 64, 256, 1024)

REPLICATES_PER_HYPOTHESIS = 64
RANDOM_SEED = 20260810


# ============================================================
# Frequency grid
# ============================================================


def frequency_step() -> float:
    return (
        MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY
    ) / TOTAL_FREQUENCIES


def analysis_frequency_grid() -> np.ndarray:
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
# Harmonic linear model
# ============================================================


def design_matrix(
    times: np.ndarray,
    frequency: float,
    harmonics: int,
) -> np.ndarray:
    columns = []

    for harmonic in range(
        1,
        harmonics + 1,
    ):
        phase = (
            2.0
            * math.pi
            * frequency
            * harmonic
            * times
        )

        columns.append(
            np.sin(phase)
        )
        columns.append(
            np.cos(phase)
        )

    columns.append(
        np.ones_like(times)
    )

    return np.column_stack(
        columns
    )


def orthonormal_basis(
    times: np.ndarray,
    frequency: float,
    harmonics: int,
) -> np.ndarray:
    design = design_matrix(
        times,
        frequency,
        harmonics,
    )

    q, _ = np.linalg.qr(
        design,
        mode="reduced",
    )

    return q


def fitted_and_residuals_from_basis(
    flux: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fitted = (
        q @ (q.T @ flux)
    )

    residuals = (
        flux - fitted
    )

    residuals = (
        residuals
        - np.mean(residuals)
    )

    return fitted, residuals


def power_from_basis(
    flux: np.ndarray,
    q: np.ndarray,
) -> tuple[float, float]:
    total_energy = float(
        np.dot(
            flux,
            flux,
        )
    )

    mean = float(
        np.mean(flux)
    )

    reference_chi2 = (
        total_energy
        - len(flux)
        * mean
        * mean
    )

    projected = (
        q.T @ flux
    )

    fitted_energy = float(
        np.dot(
            projected,
            projected,
        )
    )

    residual_chi2 = max(
        0.0,
        total_energy
        - fitted_energy,
    )

    if reference_chi2 <= 0:
        return math.nan, reference_chi2

    power = (
        reference_chi2
        - residual_chi2
    ) / reference_chi2

    return float(power), float(reference_chi2)


def independent_fixed_powers(
    flux_by_sector: dict[int, np.ndarray],
    bases_by_hypothesis: dict[str, dict[int, np.ndarray]],
) -> tuple[float, float]:
    weighted_a = 0.0
    weighted_b = 0.0
    total_chi2 = 0.0

    for sector in SECTORS:
        flux = flux_by_sector[sector]

        power_a, chi2_a = power_from_basis(
            flux,
            bases_by_hypothesis[
                HYPOTHESIS_A_NAME
            ][sector],
        )

        power_b, chi2_b = power_from_basis(
            flux,
            bases_by_hypothesis[
                HYPOTHESIS_B_NAME
            ][sector],
        )

        # Both reference chi2 values are the same for a given flux;
        # average only protects against tiny floating-point differences.
        reference_chi2 = (
            chi2_a + chi2_b
        ) / 2.0

        weighted_a += (
            power_a * reference_chi2
        )
        weighted_b += (
            power_b * reference_chi2
        )
        total_chi2 += reference_chi2

    return (
        weighted_a / total_chi2,
        weighted_b / total_chi2,
    )


# ============================================================
# Full observed periodogram
# ============================================================


def lomb_power(
    model: LombScargle,
    frequencies: np.ndarray,
) -> np.ndarray:
    try:
        return model.power(
            frequencies,
            method="fastchi2",
        )
    except Exception:
        return model.power(
            frequencies,
        )


def observed_independent_periodogram(
    sector_data: dict[int, dict],
    frequencies: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    powers = {}
    chi2 = {}

    for sector in SECTORS:
        flux = sector_data[
            sector
        ]["flux"]

        model = LombScargle(
            sector_data[sector]["times"],
            flux,
            fit_mean=True,
            center_data=True,
            nterms=harmonics,
        )

        sector_powers = lomb_power(
            model,
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
# Residual block bootstrap
# ============================================================


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
# Simulation setup
# ============================================================


def prepare_configuration(
    sector_data: dict[int, dict],
    harmonics: int,
) -> dict:
    frequencies = {
        HYPOTHESIS_A_NAME:
            1.0 / HYPOTHESIS_A_PERIOD,
        HYPOTHESIS_B_NAME:
            1.0 / HYPOTHESIS_B_PERIOD,
    }

    bases = {
        HYPOTHESIS_A_NAME: {},
        HYPOTHESIS_B_NAME: {},
    }

    fitted = {
        HYPOTHESIS_A_NAME: {},
        HYPOTHESIS_B_NAME: {},
    }

    residuals = {
        HYPOTHESIS_A_NAME: {},
        HYPOTHESIS_B_NAME: {},
    }

    for hypothesis_name in (
        HYPOTHESIS_A_NAME,
        HYPOTHESIS_B_NAME,
    ):
        frequency = frequencies[
            hypothesis_name
        ]

        for sector in SECTORS:
            q = orthonormal_basis(
                sector_data[sector]["times"],
                frequency,
                harmonics,
            )

            sector_fitted, sector_residuals = (
                fitted_and_residuals_from_basis(
                    sector_data[sector]["flux"],
                    q,
                )
            )

            bases[
                hypothesis_name
            ][sector] = q

            fitted[
                hypothesis_name
            ][sector] = sector_fitted

            residuals[
                hypothesis_name
            ][sector] = sector_residuals

    observed_flux = {
        sector: sector_data[
            sector
        ]["flux"]
        for sector in SECTORS
    }

    observed_power_a, observed_power_b = (
        independent_fixed_powers(
            observed_flux,
            bases,
        )
    )

    return {
        "harmonics": harmonics,
        "bases": bases,
        "fitted": fitted,
        "residuals": residuals,
        "observedPowerA": observed_power_a,
        "observedPowerB": observed_power_b,
        "observedDelta": (
            observed_power_b
            - observed_power_a
        ),
    }


def simulate_hypothesis(
    configuration: dict,
    sector_data: dict[int, dict],
    hypothesis_name: str,
    block_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    deltas = np.empty(
        REPLICATES_PER_HYPOTHESIS,
        dtype=np.float64,
    )

    for replicate in range(
        REPLICATES_PER_HYPOTHESIS
    ):
        synthetic_flux = {}

        for sector in SECTORS:
            sampled_residuals = (
                circular_block_resample(
                    configuration[
                        "residuals"
                    ][hypothesis_name][sector],
                    block_samples,
                    rng,
                )
            )

            synthetic_flux[sector] = (
                configuration[
                    "fitted"
                ][hypothesis_name][sector]
                + sampled_residuals
            )

        power_a, power_b = (
            independent_fixed_powers(
                synthetic_flux,
                configuration[
                    "bases"
                ],
            )
        )

        deltas[replicate] = (
            power_b - power_a
        )

    return deltas


# ============================================================
# Reporting
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


def block_duration_text(
    block_samples: int,
) -> str:
    minutes = (
        block_samples
        * PREFERRED_EXPTIME_SECONDS
        / 60.0
    )

    if minutes < 60:
        return f"{minutes:.0f} min"

    hours = minutes / 60.0

    if hours < 24:
        return f"{hours:.1f} h"

    return f"{hours / 24.0:.2f} d"


def plus_one_tail(
    count: int,
    total: int,
) -> float:
    return (
        count + 1
    ) / (
        total + 1
    )


def run_robustness_sweep(
    sector_data: dict[int, dict],
    configuration: dict,
    rng: np.random.Generator,
) -> list[dict]:
    harmonics = configuration[
        "harmonics"
    ]

    observed_delta = configuration[
        "observedDelta"
    ]

    results = []

    print()
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"🧪 MODEL: {harmonics} harmonic"
        f"{'' if harmonics == 1 else 's'}"
    )
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "observed power(A): "
        f"{configuration['observedPowerA']:.8f}"
    )
    print(
        "observed power(B): "
        f"{configuration['observedPowerB']:.8f}"
    )
    print(
        "observed Δsupport B-A: "
        f"{observed_delta:+.8f}"
    )

    for block_samples in BLOCK_SIZES:
        print()
        print(
            "🔹 block size "
            f"{block_samples} samples "
            f"(~{block_duration_text(block_samples)})"
        )

        a_deltas = simulate_hypothesis(
            configuration,
            sector_data,
            HYPOTHESIS_A_NAME,
            block_samples,
            rng,
        )

        b_deltas = simulate_hypothesis(
            configuration,
            sector_data,
            HYPOTHESIS_B_NAME,
            block_samples,
            rng,
        )

        a_wrong = int(
            np.count_nonzero(
                a_deltas > 0
            )
        )
        b_wrong = int(
            np.count_nonzero(
                b_deltas < 0
            )
        )

        if observed_delta >= 0:
            a_extreme = int(
                np.count_nonzero(
                    a_deltas
                    >= observed_delta
                )
            )
            b_percentile_count = int(
                np.count_nonzero(
                    b_deltas
                    <= observed_delta
                )
            )
        else:
            a_extreme = int(
                np.count_nonzero(
                    a_deltas
                    <= observed_delta
                )
            )
            b_percentile_count = int(
                np.count_nonzero(
                    b_deltas
                    >= observed_delta
                )
            )

        a_max = float(
            np.max(a_deltas)
        )
        b_min = float(
            np.min(b_deltas)
        )

        separated = (
            a_max < b_min
        )

        print(
            "   A Δ median / 95%: "
            f"{percentile(a_deltas, 50):+.8f} / "
            f"{percentile(a_deltas, 2.5):+.8f} to "
            f"{percentile(a_deltas, 97.5):+.8f}"
        )
        print(
            "   B Δ median / 95%: "
            f"{percentile(b_deltas, 50):+.8f} / "
            f"{percentile(b_deltas, 2.5):+.8f} to "
            f"{percentile(b_deltas, 97.5):+.8f}"
        )
        print(
            "   A misclassified as B: "
            f"{a_wrong}/{REPLICATES_PER_HYPOTHESIS} "
            f"({a_wrong / REPLICATES_PER_HYPOTHESIS * 100:.2f}%)"
        )
        print(
            "   B misclassified as A: "
            f"{b_wrong}/{REPLICATES_PER_HYPOTHESIS} "
            f"({b_wrong / REPLICATES_PER_HYPOTHESIS * 100:.2f}%)"
        )
        print(
            "   A simulations at least as B-favoring "
            "as observed: "
            f"{a_extreme}/{REPLICATES_PER_HYPOTHESIS}"
        )
        print(
            "   plus-one empirical A-tail probability: "
            f"{plus_one_tail(a_extreme, REPLICATES_PER_HYPOTHESIS) * 100:.2f}%"
        )
        print(
            "   observed Δ percentile within B injections: "
            f"{b_percentile_count / REPLICATES_PER_HYPOTHESIS * 100:.2f}%"
        )
        print(
            "   complete simulated A/B Δ separation: "
            f"{'YES' if separated else 'NO'}"
        )

        results.append(
            {
                "harmonics": harmonics,
                "blockSamples": block_samples,
                "blockText": block_duration_text(
                    block_samples
                ),
                "observedDelta": observed_delta,
                "aWrong": a_wrong,
                "bWrong": b_wrong,
                "aExtreme": a_extreme,
                "aTailPlusOne": plus_one_tail(
                    a_extreme,
                    REPLICATES_PER_HYPOTHESIS,
                ),
                "bObservedPercentile": (
                    b_percentile_count
                    / REPLICATES_PER_HYPOTHESIS
                ),
                "aMedian": percentile(
                    a_deltas,
                    50,
                ),
                "bMedian": percentile(
                    b_deltas,
                    50,
                ),
                "aLow95": percentile(
                    a_deltas,
                    2.5,
                ),
                "aHigh95": percentile(
                    a_deltas,
                    97.5,
                ),
                "bLow95": percentile(
                    b_deltas,
                    2.5,
                ),
                "bHigh95": percentile(
                    b_deltas,
                    97.5,
                ),
                "fullySeparated": separated,
            }
        )

    return results


def print_full_periodogram_summary(
    sector_data: dict[int, dict],
    frequencies: np.ndarray,
):
    print()
    print("📈 REAL-DATA FULL-WINDOW MODEL CHECK")
    print(
        "════════════════════════════════════════════════════════"
    )

    frequency_a = (
        1.0 / HYPOTHESIS_A_PERIOD
    )

    for harmonics in HARMONIC_ORDERS:
        print()
        print(
            f"🧪 {harmonics} harmonic"
            f"{'' if harmonics == 1 else 's'}"
        )

        powers = (
            observed_independent_periodogram(
                sector_data,
                frequencies,
                harmonics,
            )
        )

        best_index = int(
            np.nanargmax(
                powers
            )
        )

        tars_index = int(
            np.argmin(
                np.abs(
                    frequencies
                    - frequency_a
                )
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

        tars_power = float(
            powers[
                tars_index
            ]
        )

        print(
            "   full-window winner: "
            f"{best_period:.8f} days"
        )
        print(
            "   winner power: "
            f"{best_power:.8f}"
        )
        print(
            "   TARS relative support: "
            f"{tars_power / best_power * 100:.2f}%"
        )


def print_final_summary(
    configurations: dict[int, dict],
    results: list[dict],
):
    print()
    print()
    print("🏁 ROBUSTNESS SWEEP SUMMARY")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "harm  block       observed Δ    "
        "A→B       B→A       A-tail+1    separated"
    )
    print(
        "--------------------------------------------------------"
        "--------------------"
    )

    for result in results:
        print(
            f"{result['harmonics']:>4}  "
            f"{result['blockText']:<9} "
            f"{result['observedDelta']:+.8f}  "
            f"{result['aWrong']:>2}/"
            f"{REPLICATES_PER_HYPOTHESIS:<2}    "
            f"{result['bWrong']:>2}/"
            f"{REPLICATES_PER_HYPOTHESIS:<2}    "
            f"{result['aTailPlusOne'] * 100:>7.2f}%     "
            f"{'YES' if result['fullySeparated'] else 'NO'}"
        )

    all_zero_misclassification = all(
        result["aWrong"] == 0
        and result["bWrong"] == 0
        for result in results
    )

    all_separated = all(
        result["fullySeparated"]
        for result in results
    )

    all_observed_favors_b = all(
        configurations[harmonics][
            "observedDelta"
        ] > 0
        for harmonics in HARMONIC_ORDERS
    )

    print()
    print(
        "observed data favors B under every model: "
        f"{'YES' if all_observed_favors_b else 'NO'}"
    )
    print(
        "zero sign misclassifications in every configuration: "
        f"{'YES' if all_zero_misclassification else 'NO'}"
    )
    print(
        "complete A/B simulated Δ separation in every configuration: "
        f"{'YES' if all_separated else 'NO'}"
    )

    print()
    print(
        "Interpretation:"
    )
    print(
        "Robust discrimination requires the result to survive "
        "both longer residual-correlation blocks and the two-harmonic "
        "signal model. A zero count is limited by the finite number "
        "of simulations, so the plus-one A-tail probability is reported "
        "instead of treating 0/N as a literal zero probability."
    )
    print()
    print(
        "This remains a descriptive residual-injection robustness test, "
        "not a formal model-selection probability or confidence statement."
    )


# ============================================================
# Main
# ============================================================


def main():
    print(
        "🔬 OpenStar Blind A Noise/Shape Robustness Sweep"
    )
    print(f"target: {TARGET_NAME}")
    print("sectors: 1, 28")
    print()
    print(
        f"A ({HYPOTHESIS_A_NAME}): "
        f"{HYPOTHESIS_A_PERIOD:.8f} days"
    )
    print(
        f"B ({HYPOTHESIS_B_NAME}): "
        f"{HYPOTHESIS_B_PERIOD:.8f} days"
    )
    print(
        "statistic: Δsupport = power(B) - power(A)"
    )
    print(
        f"replicates per hypothesis/configuration: "
        f"{REPLICATES_PER_HYPOTHESIS}"
    )
    print(
        "harmonic orders: "
        + ", ".join(
            str(value)
            for value in HARMONIC_ORDERS
        )
    )
    print(
        "moving-block sizes: "
        + ", ".join(
            str(value)
            for value in BLOCK_SIZES
        )
        + " samples"
    )
    print(
        "this diagnostic does not modify OpenStar datasets/projects"
    )

    sector_data = {
        sector: load_sector(sector)
        for sector in SECTORS
    }

    frequencies = (
        analysis_frequency_grid()
    )

    print()
    print(
        "📐 Full real-data frequency bins: "
        f"{len(frequencies):,}"
    )

    print_full_periodogram_summary(
        sector_data,
        frequencies,
    )

    configurations = {
        harmonics: prepare_configuration(
            sector_data,
            harmonics,
        )
        for harmonics in HARMONIC_ORDERS
    }

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    all_results = []

    for harmonics in HARMONIC_ORDERS:
        model_results = (
            run_robustness_sweep(
                sector_data,
                configurations[
                    harmonics
                ],
                rng,
            )
        )

        all_results.extend(
            model_results
        )

    print_final_summary(
        configurations,
        all_results,
    )

    print()
    print("✅ Diagnostic complete")


if __name__ == "__main__":
    main()
