import math

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind A Sector 1 / Sector 28 weighting stability
# ============================================================

TARGET_NAME = "Blind A"
TARGET_QUERY = "TIC 25165839"
SECTORS = (1, 28)

PREFERRED_AUTHOR = "SPOC"
FALLBACK_AUTHOR = "TESS-SPOC"
PREFERRED_EXPTIME_SECONDS = 120

MAX_SAMPLES = 18_000

# Use the exact OpenStar frequency-grid definition, but only evaluate the
# scientifically relevant 7-12 day window for this stability diagnostic.
MINIMUM_FREQUENCY = 0.03
MAXIMUM_FREQUENCY = 5.0
TOTAL_FREQUENCIES = 4_194_304

WINDOW_MIN_PERIOD_DAYS = 7.0
WINDOW_MAX_PERIOD_DAYS = 12.0

# External comparison only. This diagnostic does not create or modify a
# blind OpenStar dataset/project.
TARS_ADOPTED_PERIOD_DAYS = 9.0381


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
# TESS product selection
# ============================================================


def _search_products(
    sector: int,
    *,
    author: str,
    exptime: int | None,
):
    kwargs = {
        "mission": "TESS",
        "author": author,
        "sector": int(sector),
    }

    if exptime is not None:
        kwargs["exptime"] = exptime

    return lk.search_lightcurve(
        TARGET_QUERY,
        **kwargs,
    )


def _select_shortest_cadence(search_result):
    if len(search_result) == 0:
        return search_result

    exposures = np.asarray(
        search_result.exptime,
        dtype=np.float64,
    )

    finite_indices = np.flatnonzero(
        np.isfinite(exposures)
    )

    if len(finite_indices) == 0:
        return search_result[0:1]

    shortest_local_index = int(
        np.argmin(exposures[finite_indices])
    )
    selected_index = int(
        finite_indices[shortest_local_index]
    )

    return search_result[
        selected_index:selected_index + 1
    ]


def _selected_exptime_seconds(search_result) -> float:
    value = search_result.exptime[0]

    try:
        return float(value)
    except (TypeError, ValueError):
        if hasattr(value, "value"):
            return float(value.value)
        raise


def find_light_curve(sector: int):
    preferred = _search_products(
        sector,
        author=PREFERRED_AUTHOR,
        exptime=PREFERRED_EXPTIME_SECONDS,
    )

    if len(preferred) > 0:
        selected = _select_shortest_cadence(preferred)
        return (
            selected,
            PREFERRED_AUTHOR,
            _selected_exptime_seconds(selected),
        )

    fallback = _search_products(
        sector,
        author=FALLBACK_AUTHOR,
        exptime=None,
    )

    if len(fallback) > 0:
        selected = _select_shortest_cadence(fallback)
        return (
            selected,
            FALLBACK_AUTHOR,
            _selected_exptime_seconds(selected),
        )

    raise RuntimeError(
        f"No SPOC-family light curve found for Sector {sector}."
    )


def load_sector(sector: int) -> dict:
    print()
    print(f"🔭 Sector {sector}")

    search_result, author, cadence_seconds = (
        find_light_curve(sector)
    )

    print(f"   author: {author}")
    print(f"   cadence: {cadence_seconds:.0f}s")

    light_curve = search_result.download(
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
            f"Sector {sector} contains no finite samples."
        )

    print(f"   finite samples: {len(times64)}")
    print(f"   first time: {times64[0]:.8f}")
    print(f"   last time: {times64[-1]:.8f}")
    print(
        "   baseline: "
        f"{times64[-1] - times64[0]:.4f} days"
    )

    return {
        "sector": int(sector),
        "times64": times64,
        "flux64": flux64,
    }


# ============================================================
# Sampling and exact Float32 preparation
# ============================================================


def evenly_spaced_indices(
    source_count: int,
    selected_count: int,
) -> np.ndarray:
    if selected_count <= 0:
        return np.asarray([], dtype=np.int64)

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


def normalize_selected_flux(
    flux64: np.ndarray,
) -> np.ndarray:
    mean = float(np.mean(flux64))
    stddev = float(np.std(flux64))

    if not math.isfinite(stddev) or stddev <= 0:
        raise RuntimeError(
            "Selected flux has invalid standard deviation."
        )

    return (
        (flux64 - mean) / stddev
    ).astype(np.float64)


def prepare_from_counts(
    sector_data: dict[int, dict],
    counts: dict[int, int],
):
    selected_times = []
    selected_flux = []
    selected_labels = []

    actual_counts = {}

    for sector in SECTORS:
        data = sector_data[sector]
        requested = int(counts.get(sector, 0))
        requested = max(0, requested)

        indices = evenly_spaced_indices(
            len(data["times64"]),
            requested,
        )

        times = data["times64"][indices]
        flux = data["flux64"][indices]

        if len(times) > 0:
            flux = normalize_selected_flux(flux)

            selected_times.append(times)
            selected_flux.append(flux)
            selected_labels.append(
                np.full(
                    len(times),
                    sector,
                    dtype=np.int16,
                )
            )

        actual_counts[sector] = len(times)

    if not selected_times:
        raise RuntimeError(
            "Variant selected zero samples."
        )

    combined_times64 = np.concatenate(selected_times)
    combined_flux64 = np.concatenate(selected_flux)
    combined_labels = np.concatenate(selected_labels)

    order = np.argsort(combined_times64)
    combined_times64 = combined_times64[order]
    combined_flux64 = combined_flux64[order]
    combined_labels = combined_labels[order]

    time_origin = float(combined_times64[0])

    relative_times64 = (
        combined_times64 - time_origin
    )

    times32 = np.asarray(
        relative_times64,
        dtype=np.float32,
    )
    flux32 = np.asarray(
        combined_flux64,
        dtype=np.float32,
    )

    if len(times32) > 0:
        times32[0] = np.float32(0.0)

    return (
        times32,
        flux32,
        combined_labels,
        actual_counts,
    )


def prepare_original_proportional(
    sector_data: dict[int, dict],
):
    all_times = []
    all_flux = []
    all_labels = []

    for sector in SECTORS:
        data = sector_data[sector]
        all_times.append(data["times64"])
        all_flux.append(data["flux64"])
        all_labels.append(
            np.full(
                len(data["times64"]),
                sector,
                dtype=np.int16,
            )
        )

    times64 = np.concatenate(all_times)
    flux64 = np.concatenate(all_flux)
    labels = np.concatenate(all_labels)

    order = np.argsort(times64)
    times64 = times64[order]
    flux64 = flux64[order]
    labels = labels[order]

    indices = evenly_spaced_indices(
        len(times64),
        MAX_SAMPLES,
    )

    times64 = times64[indices]
    flux64 = flux64[indices]
    labels = labels[indices]

    normalized_flux = np.empty_like(
        flux64,
        dtype=np.float64,
    )

    actual_counts = {}

    for sector in SECTORS:
        mask = labels == sector
        sector_flux = flux64[mask]

        normalized_flux[mask] = (
            normalize_selected_flux(
                sector_flux
            )
        )

        actual_counts[sector] = int(
            np.count_nonzero(mask)
        )

    time_origin = float(times64[0])

    times32 = np.asarray(
        times64 - time_origin,
        dtype=np.float32,
    )
    flux32 = np.asarray(
        normalized_flux,
        dtype=np.float32,
    )

    times32[0] = np.float32(0.0)

    return (
        times32,
        flux32,
        labels,
        actual_counts,
    )


def prepare_all_finite(
    sector_data: dict[int, dict],
):
    counts = {
        sector: len(sector_data[sector]["times64"])
        for sector in SECTORS
    }

    return prepare_from_counts(
        sector_data,
        counts,
    )


# ============================================================
# Lomb-Scargle analysis
# ============================================================


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


def analyze_variant(
    name: str,
    times32: np.ndarray,
    flux32: np.ndarray,
    actual_counts: dict[int, int],
    frequencies: np.ndarray,
):
    times64 = np.asarray(
        times32,
        dtype=np.float32,
    ).astype(np.float64)

    flux64 = np.asarray(
        flux32,
        dtype=np.float32,
    ).astype(np.float64)

    model = LombScargle(
        times64,
        flux64,
    )

    powers = model.power(
        frequencies
    )

    best_index = int(
        np.nanargmax(powers)
    )

    best_frequency = float(
        frequencies[best_index]
    )
    best_period = 1.0 / best_frequency
    best_power = float(
        powers[best_index]
    )

    local_indices = local_maxima_indices(
        powers
    )

    local_indices = sorted(
        local_indices,
        key=lambda index: float(
            powers[index]
        ),
        reverse=True,
    )

    top_peaks = []

    for index in local_indices[:8]:
        frequency = float(
            frequencies[index]
        )
        power = float(
            powers[index]
        )

        top_peaks.append(
            {
                "frequency": frequency,
                "period": 1.0 / frequency,
                "power": power,
                "relative": (
                    power / best_power
                    if best_power != 0
                    else math.nan
                ),
            }
        )

    tars_frequency = (
        1.0 / TARS_ADOPTED_PERIOD_DAYS
    )

    nearest_tars_index = int(
        np.argmin(
            np.abs(
                frequencies - tars_frequency
            )
        )
    )

    tars_grid_power = float(
        powers[nearest_tars_index]
    )

    nearest_local = None

    if local_indices:
        nearest_local_index = min(
            local_indices,
            key=lambda index: abs(
                float(frequencies[index])
                - tars_frequency
            ),
        )

        nearest_frequency = float(
            frequencies[
                nearest_local_index
            ]
        )

        nearest_local = {
            "frequency": nearest_frequency,
            "period": 1.0 / nearest_frequency,
            "power": float(
                powers[
                    nearest_local_index
                ]
            ),
        }

    return {
        "name": name,
        "sampleCount": len(times32),
        "sectorCounts": dict(actual_counts),
        "baselineDays": (
            float(times64[-1] - times64[0])
            if len(times64) > 1
            else 0.0
        ),
        "bestFrequency": best_frequency,
        "bestPeriod": best_period,
        "bestPower": best_power,
        "tarsGridPower": tars_grid_power,
        "tarsRelativePower": (
            tars_grid_power / best_power
            if best_power != 0
            else math.nan
        ),
        "nearestTarsLocal": nearest_local,
        "topPeaks": top_peaks,
    }


def print_variant(result: dict):
    sector_counts = result[
        "sectorCounts"
    ]

    print()
    print(
        "════════════════════════════════════════════════════════"
    )
    print(f"🧪 {result['name']}")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "samples: "
        f"{result['sampleCount']} "
        f"(S1={sector_counts.get(1, 0)}, "
        f"S28={sector_counts.get(28, 0)})"
    )
    print(
        "baseline: "
        f"{result['baselineDays']:.4f} days"
    )
    print()
    print("winner in 7-12 day window:")
    print(
        "   period: "
        f"{result['bestPeriod']:.8f} days"
    )
    print(
        "   frequency: "
        f"{result['bestFrequency']:.8f} cycles/day"
    )
    print(
        "   power: "
        f"{result['bestPower']:.8f}"
    )
    print()
    print(
        "TARS 9.0381d nearest-grid power: "
        f"{result['tarsGridPower']:.8f}"
    )
    print(
        "TARS nearest-grid power relative to winner: "
        f"{result['tarsRelativePower'] * 100:.2f}%"
    )

    nearest = result["nearestTarsLocal"]

    if nearest is not None:
        print(
            "nearest local maximum to TARS:"
        )
        print(
            "   period: "
            f"{nearest['period']:.8f} days"
        )
        print(
            "   power: "
            f"{nearest['power']:.8f}"
        )
        print(
            "   difference from TARS: "
            f"{abs(nearest['period'] - TARS_ADOPTED_PERIOD_DAYS):.8f} days"
        )

    print()
    print("top local peaks:")

    for rank, peak in enumerate(
        result["topPeaks"],
        start=1,
    ):
        print(
            f"   {rank:>2}. "
            f"{peak['period']:.8f} d   "
            f"{peak['power']:.8f}   "
            f"{peak['relative'] * 100:6.2f}%"
        )


def print_summary(results: list[dict]):
    print()
    print()
    print("🏁 WEIGHTING STABILITY SUMMARY")
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "variant                         "
        "S1      S28      winner(d)      "
        "TARS-near(d)   TARS-power"
    )
    print(
        "--------------------------------------------------------"
        "------------------------------"
    )

    for result in results:
        counts = result["sectorCounts"]
        nearest = result["nearestTarsLocal"]

        nearest_period = (
            nearest["period"]
            if nearest is not None
            else math.nan
        )

        print(
            f"{result['name'][:30]:<30} "
            f"{counts.get(1, 0):>6} "
            f"{counts.get(28, 0):>8} "
            f"{result['bestPeriod']:>13.8f} "
            f"{nearest_period:>13.8f} "
            f"{result['tarsRelativePower'] * 100:>9.2f}%"
        )

    rounded_winners = {}

    for result in results:
        key = round(
            result["bestPeriod"],
            3,
        )
        rounded_winners.setdefault(
            key,
            [],
        ).append(
            result["name"]
        )

    print()
    print("winner families rounded to 0.001 day:")

    for period in sorted(
        rounded_winners
    ):
        names = ", ".join(
            rounded_winners[period]
        )
        print(
            f"   {period:.3f} d -> {names}"
        )

    winner_periods = np.asarray(
        [
            result["bestPeriod"]
            for result in results
        ],
        dtype=np.float64,
    )

    print()
    print(
        "winner-period spread across variants: "
        f"{np.min(winner_periods):.8f} - "
        f"{np.max(winner_periods):.8f} days"
    )
    print(
        "spread width: "
        f"{np.ptp(winner_periods):.8f} days"
    )

    closest_result = min(
        results,
        key=lambda result: abs(
            result["bestPeriod"]
            - TARS_ADOPTED_PERIOD_DAYS
        ),
    )

    print()
    print(
        "winner closest to TARS adopted period:"
    )
    print(
        f"   variant: {closest_result['name']}"
    )
    print(
        "   period: "
        f"{closest_result['bestPeriod']:.8f} days"
    )
    print(
        "   difference: "
        f"{abs(closest_result['bestPeriod'] - TARS_ADOPTED_PERIOD_DAYS):.8f} days"
    )


# ============================================================
# Main
# ============================================================


def main():
    print(
        "🔬 OpenStar Blind A Sector-Weighting Stability Diagnostic"
    )
    print(f"target: {TARGET_NAME}")
    print("sectors: 1, 28")
    print(
        "analysis window: "
        f"{WINDOW_MIN_PERIOD_DAYS:.1f}-"
        f"{WINDOW_MAX_PERIOD_DAYS:.1f} days"
    )
    print(
        "frequency-grid step: "
        f"{frequency_step():.12f} cycles/day"
    )
    print(
        "external comparison period: "
        f"{TARS_ADOPTED_PERIOD_DAYS:.4f} days"
    )
    print()
    print(
        "This diagnostic does not modify OpenStar datasets or projects."
    )

    sector_data = {
        sector: load_sector(sector)
        for sector in SECTORS
    }

    frequencies = analysis_frequency_grid()

    print()
    print(
        "📐 Analysis frequency bins: "
        f"{len(frequencies):,}"
    )
    print(
        "   first: "
        f"{frequencies[0]:.8f} cycles/day"
    )
    print(
        "   last: "
        f"{frequencies[-1]:.8f} cycles/day"
    )

    variants = []

    variants.append(
        (
            "original proportional 18k",
            prepare_original_proportional(
                sector_data
            ),
        )
    )

    variants.append(
        (
            "equal 50/50",
            prepare_from_counts(
                sector_data,
                {
                    1: 9_000,
                    28: 9_000,
                },
            ),
        )
    )

    variants.append(
        (
            "S1-heavy 75/25",
            prepare_from_counts(
                sector_data,
                {
                    1: 13_500,
                    28: 4_500,
                },
            ),
        )
    )

    variants.append(
        (
            "S28-heavy 25/75",
            prepare_from_counts(
                sector_data,
                {
                    1: 4_500,
                    28: 13_500,
                },
            ),
        )
    )

    variants.append(
        (
            "Sector 1 only",
            prepare_from_counts(
                sector_data,
                {
                    1: min(
                        MAX_SAMPLES,
                        len(
                            sector_data[1][
                                "times64"
                            ]
                        ),
                    ),
                    28: 0,
                },
            ),
        )
    )

    variants.append(
        (
            "Sector 28 only",
            prepare_from_counts(
                sector_data,
                {
                    1: 0,
                    28: len(
                        sector_data[28][
                            "times64"
                        ]
                    ),
                },
            ),
        )
    )

    variants.append(
        (
            "all finite samples",
            prepare_all_finite(
                sector_data
            ),
        )
    )

    results = []

    for name, prepared in variants:
        (
            times32,
            flux32,
            _labels,
            actual_counts,
        ) = prepared

        result = analyze_variant(
            name,
            times32,
            flux32,
            actual_counts,
            frequencies,
        )

        results.append(result)
        print_variant(result)

    print_summary(results)

    print()
    print("✅ Diagnostic complete")
    print(
        "Paste the WEIGHTING STABILITY SUMMARY back into ChatGPT."
    )


if __name__ == "__main__":
    main()
