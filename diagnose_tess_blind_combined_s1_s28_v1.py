import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar TIC 25165839 combined Sector 1 + Sector 28 diagnostic
# ============================================================
#
# This is a post-run diagnostic only. It does not download TESS data,
# regenerate the OpenStar dataset, or change coordinator behavior.
#
# It reads the exact combined JSON dataset that was distributed to the
# Swift/Metal clients, re-quantizes the stored values to Float32, converts
# those exact values back to Float64 for Astropy, reconstructs the original
# OpenStar frequency grid, and examines how the combined TESS periodogram
# relates to the external TARS adopted period.
# ============================================================

DEFAULT_DATASET_PATH = Path("data/tess-blind-a-s1-s28.json")

# External post-run comparison values already used for this blind target.
EXTERNAL_TARS_ADOPTED_PERIOD_DAYS = 9.0381
EXTERNAL_TARS_ADOPTED_UNCERTAINTY_DAYS = 0.3342

TOP_DISTINCT_PEAKS = 12
TOP_FOCUS_LOCAL_MAXIMA = 20
FOCUS_MIN_PERIOD_DAYS = 7.0
FOCUS_MAX_PERIOD_DAYS = 12.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the combined TIC 25165839 Sector 1 + Sector 28 "
            "Lomb-Scargle periodogram using the exact OpenStar dataset."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to the prepared combined dataset JSON "
            f"(default: {DEFAULT_DATASET_PATH})"
        ),
    )

    return parser.parse_args()


def load_dataset(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run the combined Sector 1 + Sector 28 preparation first, "
            "or pass the dataset path explicitly."
        )

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    required = (
        "id",
        "times",
        "flux",
        "frequencySearch",
        "reference",
    )

    missing = [key for key in required if key not in dataset]

    if missing:
        raise RuntimeError(
            "Dataset is missing required fields: " + ", ".join(missing)
        )

    return dataset


def exact_distributed_samples(dataset: dict):
    times_float32 = np.asarray(dataset["times"], dtype=np.float32)
    flux_float32 = np.asarray(dataset["flux"], dtype=np.float32)

    if len(times_float32) != len(flux_float32):
        raise RuntimeError("Time/flux sample count mismatch in dataset.")

    if len(times_float32) < 2:
        raise RuntimeError("At least two samples are required.")

    if not np.all(np.isfinite(times_float32)):
        raise RuntimeError("Dataset contains non-finite Float32 time values.")

    if not np.all(np.isfinite(flux_float32)):
        raise RuntimeError("Dataset contains non-finite Float32 flux values.")

    return (
        times_float32.astype(np.float64),
        flux_float32.astype(np.float64),
    )


def build_frequency_grid(dataset: dict):
    search = dataset["frequencySearch"]

    minimum_frequency = float(search["minimumFrequency"])
    maximum_frequency = float(search["maximumFrequency"])
    frequency_step = float(search["frequencyStep"])
    total_frequencies = int(search["totalFrequencies"])

    if total_frequencies <= 0:
        raise RuntimeError("Dataset frequency count must be positive.")

    if not math.isfinite(frequency_step) or frequency_step <= 0:
        raise RuntimeError("Dataset frequency step is invalid.")

    frequencies = (
        minimum_frequency
        + np.arange(total_frequencies, dtype=np.float64) * frequency_step
    )

    if frequencies[-1] > maximum_frequency + frequency_step:
        raise RuntimeError(
            "Generated frequency grid exceeds the dataset search range."
        )

    return frequencies


def calculate_periodogram(times, flux, frequencies):
    periodogram = LombScargle(times, flux)
    powers = np.asarray(periodogram.power(frequencies), dtype=np.float64)

    if len(powers) != len(frequencies):
        raise RuntimeError(
            "Astropy returned an unexpected number of power values."
        )

    if not np.any(np.isfinite(powers)):
        raise RuntimeError("Astropy returned no finite power values.")

    return periodogram, powers


def local_maximum_indices(powers: np.ndarray):
    if len(powers) < 3:
        return np.asarray([int(np.nanargmax(powers))], dtype=np.int64)

    finite = np.isfinite(powers)
    middle_is_peak = (
        finite[1:-1]
        & finite[:-2]
        & finite[2:]
        & (powers[1:-1] > powers[:-2])
        & (powers[1:-1] >= powers[2:])
    )

    indices = np.flatnonzero(middle_is_peak) + 1
    global_index = int(np.nanargmax(powers))

    if global_index not in indices:
        indices = np.append(indices, global_index)

    return indices.astype(np.int64, copy=False)


def select_distinct_peaks(
    frequencies,
    powers,
    candidate_indices,
    minimum_frequency_separation,
    limit,
):
    order = candidate_indices[np.argsort(powers[candidate_indices])[::-1]]
    selected = []

    for index in order:
        frequency = float(frequencies[index])

        if all(
            abs(frequency - float(frequencies[existing]))
            >= minimum_frequency_separation
            for existing in selected
        ):
            selected.append(int(index))

        if len(selected) >= limit:
            break

    return selected


def nearest_grid_index(frequencies: np.ndarray, frequency: float):
    index = int(np.searchsorted(frequencies, frequency))

    if index <= 0:
        return 0

    if index >= len(frequencies):
        return len(frequencies) - 1

    before = index - 1
    after = index

    if abs(frequencies[before] - frequency) <= abs(
        frequencies[after] - frequency
    ):
        return before

    return after


def strongest_in_frequency_window(
    frequencies,
    powers,
    center_frequency,
    half_width,
):
    mask = (
        frequencies >= center_frequency - half_width
    ) & (
        frequencies <= center_frequency + half_width
    )

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return None

    local = int(np.nanargmax(powers[indices]))
    return int(indices[local])


def strongest_in_period_interval(
    frequencies,
    powers,
    minimum_period,
    maximum_period,
):
    frequency_min = 1.0 / maximum_period
    frequency_max = 1.0 / minimum_period

    mask = (
        frequencies >= frequency_min
    ) & (
        frequencies <= frequency_max
    )

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return None

    local = int(np.nanargmax(powers[indices]))
    return int(indices[local])


def power_rank(powers: np.ndarray, power: float):
    finite_powers = powers[np.isfinite(powers)]
    stronger_count = int(np.count_nonzero(finite_powers > power))
    return stronger_count + 1, len(finite_powers)


def print_peak_table(
    title,
    indices,
    frequencies,
    powers,
    global_power,
):
    print()
    print(title)
    print(
        "   #   period (days)    frequency (c/d)      power       % of max"
    )
    print(
        "   --  ---------------  -----------------  ----------  --------"
    )

    for rank, index in enumerate(indices, start=1):
        frequency = float(frequencies[index])
        period = 1.0 / frequency
        power = float(powers[index])
        percent = (
            100.0 * power / global_power
            if global_power > 0
            else float("nan")
        )

        print(
            f"   {rank:>2}  "
            f"{period:>15.8f}  "
            f"{frequency:>17.8f}  "
            f"{power:>10.8f}  "
            f"{percent:>7.2f}%"
        )


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset)
    times, flux = exact_distributed_samples(dataset)
    frequencies = build_frequency_grid(dataset)

    baseline_days = float(times[-1] - times[0])

    if baseline_days <= 0:
        raise RuntimeError("Dataset time baseline must be positive.")

    rayleigh_frequency = 1.0 / baseline_days

    source = dataset.get("source", {})
    sectors = source.get("sectors")
    inter_sector_gap_days = source.get("interSectorGapDays")

    gap_alias_spacing = None
    if inter_sector_gap_days is not None:
        inter_sector_gap_days = float(inter_sector_gap_days)
        if inter_sector_gap_days > 0:
            gap_alias_spacing = 1.0 / inter_sector_gap_days

    print()
    print("🔬 OpenStar Combined S1+S28 Periodogram Diagnostic")
    print(f"   dataset: {dataset['id']}")
    print(f"   target: {dataset.get('targetName', 'unknown')}")
    print(f"   sectors: {sectors if sectors is not None else 'unknown'}")
    print(f"   samples: {len(times)}")
    print(
        "   numeric representation: "
        f"{dataset.get('numericRepresentation', 'unknown')}"
    )
    print(f"   full baseline: {baseline_days:.8f} days")
    print(
        "   Rayleigh frequency resolution (1/baseline): "
        f"{rayleigh_frequency:.8f} cycles/day"
    )

    if inter_sector_gap_days is not None:
        print(
            "   inter-sector no-observation gap: "
            f"{inter_sector_gap_days:.8f} days"
        )

    if gap_alias_spacing is not None:
        print(
            "   gap alias spacing estimate (1/gap): "
            f"{gap_alias_spacing:.8f} cycles/day"
        )

    print(
        "   frequency grid: "
        f"{frequencies[0]:.8f} - {frequencies[-1]:.8f} cycles/day"
    )
    print(f"   frequency bins: {len(frequencies):,}")

    print()
    print("🧪 Recomputing full Astropy periodogram")
    print("   input: exact distributed Float32 samples")

    periodogram, powers = calculate_periodogram(times, flux, frequencies)

    global_index = int(np.nanargmax(powers))
    global_frequency = float(frequencies[global_index])
    global_period = 1.0 / global_frequency
    global_power = float(powers[global_index])

    print()
    print("⭐ Recomputed Astropy maximum")
    print(f"   frequency: {global_frequency:.8f} cycles/day")
    print(f"   period: {global_period:.8f} days")
    print(f"   power: {global_power:.8f}")

    stored = dataset.get("reference", {})
    stored_frequency = stored.get("bestFrequency")
    stored_period = stored.get("bestPeriodDays")
    stored_power = stored.get("bestPower")

    if (
        stored_frequency is not None
        and stored_period is not None
        and stored_power is not None
    ):
        print()
        print("✅ Stored-reference consistency check")
        print(f"   stored frequency: {float(stored_frequency):.8f} cycles/day")
        print(
            "   recomputed frequency error: "
            f"{abs(global_frequency - float(stored_frequency)):.12f} cycles/day"
        )
        print(f"   stored period: {float(stored_period):.8f} days")
        print(
            "   recomputed period error: "
            f"{abs(global_period - float(stored_period)):.12f} days"
        )
        print(f"   stored power: {float(stored_power):.8f}")
        print(
            "   recomputed power error: "
            f"{abs(global_power - float(stored_power)):.12f}"
        )

    tars_period = EXTERNAL_TARS_ADOPTED_PERIOD_DAYS
    tars_uncertainty = EXTERNAL_TARS_ADOPTED_UNCERTAINTY_DAYS
    tars_frequency = 1.0 / tars_period

    tars_exact_power = float(periodogram.power(tars_frequency))
    tars_grid_index = nearest_grid_index(frequencies, tars_frequency)
    tars_grid_frequency = float(frequencies[tars_grid_index])
    tars_grid_period = 1.0 / tars_grid_frequency
    tars_grid_power = float(powers[tars_grid_index])

    frequency_difference = abs(tars_frequency - global_frequency)
    rayleigh_units = frequency_difference / rayleigh_frequency

    gap_alias_units = None
    if gap_alias_spacing is not None and gap_alias_spacing > 0:
        gap_alias_units = frequency_difference / gap_alias_spacing

    tars_power_fraction = (
        tars_exact_power / global_power
        if global_power > 0
        else float("nan")
    )

    rank, rank_total = power_rank(powers, tars_grid_power)

    tars_period_min = tars_period - tars_uncertainty
    tars_period_max = tars_period + tars_uncertainty

    print()
    print("🎯 External TARS adopted-period diagnostic")
    print(f"   TARS adopted period: {tars_period:.8f} days")
    print(f"   quoted uncertainty: ±{tars_uncertainty:.8f} days")
    print(
        "   quoted period interval: "
        f"{tars_period_min:.8f} - {tars_period_max:.8f} days"
    )
    print(f"   TARS frequency: {tars_frequency:.8f} cycles/day")
    print(
        "   exact Astropy power at TARS frequency: "
        f"{tars_exact_power:.8f}"
    )
    print(
        "   power relative to Astropy maximum: "
        f"{100.0 * tars_power_fraction:.2f}%"
    )
    print(f"   nearest grid period: {tars_grid_period:.8f} days")
    print(f"   nearest grid frequency: {tars_grid_frequency:.8f} cycles/day")
    print(f"   nearest grid power: {tars_grid_power:.8f}")
    print(f"   nearest-grid power rank: {rank:,}/{rank_total:,}")
    print(
        "   frequency difference from global maximum: "
        f"{frequency_difference:.8f} cycles/day"
    )
    print(
        "   difference in Rayleigh-resolution units: "
        f"{rayleigh_units:.4f}"
    )

    if gap_alias_units is not None:
        print(
            "   difference in inter-sector-gap alias spacings: "
            f"{gap_alias_units:.4f}"
        )

    in_tars_interval = tars_period_min <= global_period <= tars_period_max
    print(
        "   Astropy global period inside quoted TARS interval: "
        f"{'YES' if in_tars_interval else 'NO'}"
    )

    half_rayleigh_index = strongest_in_frequency_window(
        frequencies,
        powers,
        tars_frequency,
        0.5 * rayleigh_frequency,
    )

    one_rayleigh_index = strongest_in_frequency_window(
        frequencies,
        powers,
        tars_frequency,
        rayleigh_frequency,
    )

    for label, index in (
        ("±0.5 Rayleigh", half_rayleigh_index),
        ("±1.0 Rayleigh", one_rayleigh_index),
    ):
        if index is None:
            continue

        frequency = float(frequencies[index])
        period = 1.0 / frequency
        power = float(powers[index])

        print()
        print(f"🔎 Strongest point within {label} of TARS")
        print(f"   frequency: {frequency:.8f} cycles/day")
        print(f"   period: {period:.8f} days")
        print(f"   power: {power:.8f}")
        print(
            "   power relative to global maximum: "
            f"{100.0 * power / global_power:.2f}%"
        )

    interval_index = strongest_in_period_interval(
        frequencies,
        powers,
        tars_period_min,
        tars_period_max,
    )

    if interval_index is not None:
        interval_frequency = float(frequencies[interval_index])
        interval_period = 1.0 / interval_frequency
        interval_power = float(powers[interval_index])

        print()
        print("📏 Strongest point inside quoted TARS period interval")
        print(f"   frequency: {interval_frequency:.8f} cycles/day")
        print(f"   period: {interval_period:.8f} days")
        print(f"   power: {interval_power:.8f}")
        print(
            "   power relative to global maximum: "
            f"{100.0 * interval_power / global_power:.2f}%"
        )

    candidate_indices = local_maximum_indices(powers)

    distinct_indices = select_distinct_peaks(
        frequencies,
        powers,
        candidate_indices,
        minimum_frequency_separation=rayleigh_frequency,
        limit=TOP_DISTINCT_PEAKS,
    )

    print_peak_table(
        "📈 Strongest distinct periodogram peaks",
        distinct_indices,
        frequencies,
        powers,
        global_power,
    )

    focus_frequency_min = 1.0 / FOCUS_MAX_PERIOD_DAYS
    focus_frequency_max = 1.0 / FOCUS_MIN_PERIOD_DAYS

    focus_candidates = candidate_indices[
        (
            frequencies[candidate_indices] >= focus_frequency_min
        )
        & (
            frequencies[candidate_indices] <= focus_frequency_max
        )
    ]

    focus_indices = focus_candidates[
        np.argsort(powers[focus_candidates])[::-1]
    ][:TOP_FOCUS_LOCAL_MAXIMA]

    print_peak_table(
        (
            "🔭 Local maxima in the "
            f"{FOCUS_MIN_PERIOD_DAYS:.0f}-{FOCUS_MAX_PERIOD_DAYS:.0f} day window"
        ),
        focus_indices,
        frequencies,
        powers,
        global_power,
    )

    if len(focus_indices) > 0:
        nearest_local = min(
            focus_indices,
            key=lambda index: abs(float(frequencies[index]) - tars_frequency),
        )

        nearest_local_frequency = float(frequencies[nearest_local])
        nearest_local_period = 1.0 / nearest_local_frequency
        nearest_local_power = float(powers[nearest_local])
        nearest_local_difference = abs(
            nearest_local_frequency - tars_frequency
        )

        print()
        print("🧭 Local maximum nearest the TARS adopted frequency")
        print(f"   frequency: {nearest_local_frequency:.8f} cycles/day")
        print(f"   period: {nearest_local_period:.8f} days")
        print(f"   power: {nearest_local_power:.8f}")
        print(
            "   power relative to global maximum: "
            f"{100.0 * nearest_local_power / global_power:.2f}%"
        )
        print(
            "   separation from TARS: "
            f"{nearest_local_difference / rayleigh_frequency:.4f} Rayleigh units"
        )

        if gap_alias_spacing is not None:
            print(
                "   separation from TARS: "
                f"{nearest_local_difference / gap_alias_spacing:.4f} gap-alias spacings"
            )

    print()
    print("🏁 Diagnostic complete")
    print(
        "   Main question: is 9.0381 d itself strongly supported, or does the "
        "combined two-sector window favor a nearby alias/local maximum?"
    )
    print(
        "   Paste this output back into ChatGPT and compare the TARS power, "
        "Rayleigh separation, gap-alias spacing, and 7-12 day local maxima."
    )


if __name__ == "__main__":
    main()
