import math

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind A phase-coherence diagnostic
#
# Compares:
#   1. Normal coherent Lomb-Scargle on Sectors 1 + 28
#   2. Shared frequency with each sector allowed its own
#      mean, amplitude, and phase
#
# This diagnostic does not modify OpenStar datasets/projects.
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

KNOWN_CANDIDATE_PERIODS = (
    ("TARS adopted", 9.0381),
    ("near-TARS alias", 9.06755908),
    ("Sector 1 winner", 9.21257156),
    ("combined coherent alias", 9.29813749),
    ("equal-weight alias", 9.41788085),
    ("S28-heavy alias", 9.54064075),
    ("Sector 28 winner", 9.99286041),
)


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

    times64 = np.asarray(
        light_curve.time.value,
        dtype=np.float64,
    )
    flux64 = np.asarray(
        light_curve.flux.value,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(times64)
        & np.isfinite(flux64)
    )

    times64 = times64[finite]
    flux64 = flux64[finite]

    order = np.argsort(times64)
    times64 = times64[order]
    flux64 = flux64[order]

    if len(times64) == 0:
        raise RuntimeError(
            f"Sector {sector} has no finite samples."
        )

    mean = float(np.mean(flux64))
    stddev = float(np.std(flux64))

    if not math.isfinite(stddev) or stddev <= 0:
        raise RuntimeError(
            f"Sector {sector} has invalid flux stddev."
        )

    normalized64 = (
        (flux64 - mean) / stddev
    ).astype(np.float64)

    print(f"   finite samples: {len(times64)}")
    print(f"   first time: {times64[0]:.8f}")
    print(f"   last time: {times64[-1]:.8f}")
    print(
        "   baseline: "
        f"{times64[-1] - times64[0]:.4f} days"
    )

    return {
        "sector": sector,
        "times": times64,
        "flux": normalized64,
    }


def local_maxima_indices(
    powers: np.ndarray,
) -> np.ndarray:
    if len(powers) < 3:
        return np.asarray([], dtype=np.int64)

    mask = (
        (powers[1:-1] > powers[:-2])
        & (powers[1:-1] >= powers[2:])
        & np.isfinite(powers[1:-1])
    )

    return np.flatnonzero(mask) + 1


def top_local_peaks(
    frequencies: np.ndarray,
    powers: np.ndarray,
    *,
    limit: int = 12,
) -> list[dict]:
    indices = local_maxima_indices(powers)

    ranked = sorted(
        indices,
        key=lambda index: float(
            powers[index]
        ),
        reverse=True,
    )

    global_max = float(
        np.nanmax(powers)
    )

    peaks = []

    for index in ranked[:limit]:
        frequency = float(
            frequencies[index]
        )
        power = float(
            powers[index]
        )

        peaks.append(
            {
                "frequency": frequency,
                "period": 1.0 / frequency,
                "power": power,
                "relative": (
                    power / global_max
                    if global_max != 0
                    else math.nan
                ),
            }
        )

    return peaks


def wrap_phase_radians(value: float) -> float:
    return (
        (value + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def sinusoid_fit(
    times: np.ndarray,
    flux: np.ndarray,
    frequency: float,
    common_origin: float,
) -> dict:
    relative_time = (
        times - common_origin
    )

    omega_t = (
        2.0
        * math.pi
        * frequency
        * relative_time
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

    sin_coefficient = float(
        coefficients[0]
    )
    cos_coefficient = float(
        coefficients[1]
    )
    offset = float(
        coefficients[2]
    )

    fitted = design @ coefficients
    residual = flux - fitted

    amplitude = math.hypot(
        sin_coefficient,
        cos_coefficient,
    )

    # A*sin(wt + phi)
    phase = math.atan2(
        cos_coefficient,
        sin_coefficient,
    )

    rms = float(
        np.sqrt(
            np.mean(
                residual * residual
            )
        )
    )

    return {
        "amplitude": amplitude,
        "phaseRadians": phase,
        "phaseCycles": phase / (
            2.0 * math.pi
        ),
        "offset": offset,
        "rms": rms,
    }


def print_peak_table(
    title: str,
    peaks: list[dict],
):
    print()
    print(title)
    print(
        "#   period (days)    frequency (c/d)      "
        "power       % of max"
    )
    print(
        "--------------------------------------------------------"
    )

    for rank, peak in enumerate(
        peaks,
        start=1,
    ):
        print(
            f"{rank:>2}  "
            f"{peak['period']:>14.8f}  "
            f"{peak['frequency']:>16.8f}  "
            f"{peak['power']:>10.8f}  "
            f"{peak['relative'] * 100:>8.2f}%"
        )


def main():
    print(
        "🔬 OpenStar Blind A Phase-Coherence Diagnostic"
    )
    print(f"target: {TARGET_NAME}")
    print("sectors: 1, 28")
    print(
        "analysis window: "
        f"{WINDOW_MIN_PERIOD_DAYS:.1f}-"
        f"{WINDOW_MAX_PERIOD_DAYS:.1f} days"
    )
    print(
        "frequency grid step: "
        f"{frequency_step():.12f} cycles/day"
    )
    print()
    print(
        "coherent model: one global mean/amplitude/phase"
    )
    print(
        "independent-phase model: one shared frequency, "
        "but each sector gets its own mean/amplitude/phase"
    )
    print(
        "input: all finite samples, each sector normalized "
        "independently in Float64"
    )
    print(
        "this diagnostic does not modify OpenStar datasets/projects"
    )

    sector_data = {
        sector: load_sector(sector)
        for sector in SECTORS
    }

    common_origin = min(
        float(
            sector_data[sector]["times"][0]
        )
        for sector in SECTORS
    )

    combined_times = np.concatenate(
        [
            sector_data[sector]["times"]
            for sector in SECTORS
        ]
    )
    combined_flux = np.concatenate(
        [
            sector_data[sector]["flux"]
            for sector in SECTORS
        ]
    )

    order = np.argsort(
        combined_times
    )
    combined_times = combined_times[order]
    combined_flux = combined_flux[order]

    baseline = float(
        combined_times[-1]
        - combined_times[0]
    )

    gap = float(
        sector_data[28]["times"][0]
        - sector_data[1]["times"][-1]
    )

    print()
    print("📏 Combined timing")
    print(
        f"   full baseline: {baseline:.8f} days"
    )
    print(
        f"   no-observation gap: {gap:.8f} days"
    )
    print(
        "   Rayleigh resolution: "
        f"{1.0 / baseline:.8f} cycles/day"
    )
    print(
        "   gap-alias spacing: "
        f"{1.0 / gap:.8f} cycles/day"
    )

    frequencies = analysis_frequency_grid()

    print()
    print(
        "📐 Frequency bins evaluated: "
        f"{len(frequencies):,}"
    )

    coherent_ls = LombScargle(
        combined_times - common_origin,
        combined_flux,
        fit_mean=True,
        center_data=True,
    )

    print()
    print(
        "🧪 Computing globally coherent Lomb-Scargle"
    )

    coherent_power = coherent_ls.power(
        frequencies
    )

    sector_powers = {}
    sector_reference_chi2 = {}

    print(
        "🧪 Computing per-sector Lomb-Scargle powers"
    )

    for sector in SECTORS:
        times = (
            sector_data[sector]["times"]
            - common_origin
        )
        flux = sector_data[sector]["flux"]

        model = LombScargle(
            times,
            flux,
            fit_mean=True,
            center_data=True,
        )

        powers = model.power(
            frequencies
        )

        reference_chi2 = float(
            np.sum(
                (
                    flux
                    - np.mean(flux)
                ) ** 2
            )
        )

        sector_powers[sector] = powers
        sector_reference_chi2[
            sector
        ] = reference_chi2

    total_reference_chi2 = sum(
        sector_reference_chi2.values()
    )

    # Astropy standard normalization:
    # power = (chi2_ref - chi2_f) / chi2_ref
    #
    # Summing each sector's chi2 reduction produces the
    # block-diagonal shared-frequency / independent-phase fit.
    independent_power = (
        sector_powers[1]
        * sector_reference_chi2[1]
        + sector_powers[28]
        * sector_reference_chi2[28]
    ) / total_reference_chi2

    coherent_best_index = int(
        np.nanargmax(
            coherent_power
        )
    )
    independent_best_index = int(
        np.nanargmax(
            independent_power
        )
    )

    coherent_best_frequency = float(
        frequencies[
            coherent_best_index
        ]
    )
    independent_best_frequency = float(
        frequencies[
            independent_best_index
        ]
    )

    coherent_best_period = (
        1.0 / coherent_best_frequency
    )
    independent_best_period = (
        1.0 / independent_best_frequency
    )

    print()
    print("⭐ Best coherent solution")
    print(
        "   frequency: "
        f"{coherent_best_frequency:.8f} cycles/day"
    )
    print(
        "   period: "
        f"{coherent_best_period:.8f} days"
    )
    print(
        "   power: "
        f"{float(coherent_power[coherent_best_index]):.8f}"
    )

    print()
    print("⭐ Best independent-phase solution")
    print(
        "   frequency: "
        f"{independent_best_frequency:.8f} cycles/day"
    )
    print(
        "   period: "
        f"{independent_best_period:.8f} days"
    )
    print(
        "   combined independent-phase power: "
        f"{float(independent_power[independent_best_index]):.8f}"
    )
    print(
        "   Sector 1 power there: "
        f"{float(sector_powers[1][independent_best_index]):.8f}"
    )
    print(
        "   Sector 28 power there: "
        f"{float(sector_powers[28][independent_best_index]):.8f}"
    )

    coherent_peaks = top_local_peaks(
        frequencies,
        coherent_power,
    )
    independent_peaks = top_local_peaks(
        frequencies,
        independent_power,
    )

    print_peak_table(
        "📈 Top coherent peaks",
        coherent_peaks,
    )

    print_peak_table(
        "📈 Top independent-phase peaks",
        independent_peaks,
    )

    diagnostic_periods = list(
        KNOWN_CANDIDATE_PERIODS
    )

    diagnostic_periods.extend(
        [
            (
                "coherent winner",
                coherent_best_period,
            ),
            (
                "independent-phase winner",
                independent_best_period,
            ),
        ]
    )

    unique_candidates = []

    for name, period in diagnostic_periods:
        if any(
            abs(
                period - existing_period
            ) < 0.00001
            for _, existing_period
            in unique_candidates
        ):
            continue

        unique_candidates.append(
            (name, period)
        )

    print()
    print()
    print(
        "🧭 PHASE COHERENCE AT IMPORTANT CANDIDATES"
    )
    print(
        "════════════════════════════════════════════════════════"
    )

    for name, period in unique_candidates:
        frequency = 1.0 / period

        nearest_index = int(
            np.argmin(
                np.abs(
                    frequencies
                    - frequency
                )
            )
        )

        grid_frequency = float(
            frequencies[
                nearest_index
            ]
        )
        grid_period = (
            1.0 / grid_frequency
        )

        fit1 = sinusoid_fit(
            sector_data[1]["times"],
            sector_data[1]["flux"],
            grid_frequency,
            common_origin,
        )
        fit28 = sinusoid_fit(
            sector_data[28]["times"],
            sector_data[28]["flux"],
            grid_frequency,
            common_origin,
        )

        phase_difference = wrap_phase_radians(
            fit28["phaseRadians"]
            - fit1["phaseRadians"]
        )

        phase_cycles = (
            phase_difference
            / (2.0 * math.pi)
        )

        phase_time_days = (
            phase_cycles
            / grid_frequency
        )

        print()
        print(f"🔹 {name}")
        print(
            "   grid period: "
            f"{grid_period:.8f} days"
        )
        print(
            "   coherent power: "
            f"{float(coherent_power[nearest_index]):.8f}"
        )
        print(
            "   independent-phase power: "
            f"{float(independent_power[nearest_index]):.8f}"
        )
        print(
            "   Sector 1 power: "
            f"{float(sector_powers[1][nearest_index]):.8f}"
        )
        print(
            "   Sector 28 power: "
            f"{float(sector_powers[28][nearest_index]):.8f}"
        )
        print(
            "   Sector 1 amplitude: "
            f"{fit1['amplitude']:.8f}"
        )
        print(
            "   Sector 28 amplitude: "
            f"{fit28['amplitude']:.8f}"
        )
        print(
            "   Sector 1 RMS: "
            f"{fit1['rms']:.8f}"
        )
        print(
            "   Sector 28 RMS: "
            f"{fit28['rms']:.8f}"
        )
        print(
            "   fitted phase difference S28-S1: "
            f"{phase_cycles:+.6f} cycles"
        )
        print(
            "   equivalent phase-time offset: "
            f"{phase_time_days:+.6f} days"
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

    coherent_max = float(
        np.nanmax(coherent_power)
    )
    independent_max = float(
        np.nanmax(independent_power)
    )

    print()
    print()
    print("🏁 PHASE-COHERENCE SUMMARY")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "coherent winner: "
        f"{coherent_best_period:.8f} days"
    )
    print(
        "independent-phase winner: "
        f"{independent_best_period:.8f} days"
    )
    print(
        "winner separation: "
        f"{abs(coherent_best_period - independent_best_period):.8f} days"
    )
    print()
    print(
        "TARS nearest-grid coherent power: "
        f"{float(coherent_power[tars_index]):.8f} "
        f"({float(coherent_power[tars_index]) / coherent_max * 100:.2f}% of coherent max)"
    )
    print(
        "TARS nearest-grid independent-phase power: "
        f"{float(independent_power[tars_index]):.8f} "
        f"({float(independent_power[tars_index]) / independent_max * 100:.2f}% of independent-phase max)"
    )
    print()
    print(
        "Interpretation target:"
    )
    print(
        "If the independent-phase score strongly favors the "
        "TARS neighborhood while the coherent score keeps the "
        "alias comb, the long-gap phase constraint is the main "
        "reason the normal combined periodogram picks alternate aliases."
    )
    print()
    print("✅ Diagnostic complete")


if __name__ == "__main__":
    main()
