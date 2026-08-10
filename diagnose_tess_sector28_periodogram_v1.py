import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar TIC 25165839 Sector 28 periodogram diagnostic
# ============================================================
#
# This script is intentionally separate from the blind validation
# preparation/run. The OpenStar Sector 28 run has already completed,
# so it is now safe for this post-run diagnostic to explicitly examine
# the external TARS Sector 28 period.
#
# It reads the exact JSON dataset distributed to the Swift/Metal clients,
# reconstructs the stored samples as Float32, converts those exact values
# to Float64 for Astropy, and uses the dataset's original frequency grid.
#
# No MAST download or preprocessing is performed here.
# ============================================================

DEFAULT_DATASET_PATH = Path("data/tess-blind-a-sector28.json")
EXTERNAL_TARS_PERIOD_DAYS = 8.81

TOP_DISTINCT_PEAKS = 12
FOCUS_MIN_PERIOD_DAYS = 5.0
FOCUS_MAX_PERIOD_DAYS = 15.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the TIC 25165839 Sector 28 Lomb-Scargle "
            "periodogram using the exact OpenStar distributed dataset."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to the prepared Sector 28 dataset JSON "
            f"(default: {DEFAULT_DATASET_PATH})"
        ),
    )

    return parser.parse_args()


def load_dataset(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run prepare_tess_sector28_blind_v1.py first, or pass the "
            "dataset path explicitly."
        )

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    required_top_level = (
        "id",
        "times",
        "flux",
        "frequencySearch",
        "reference",
    )

    missing = [
        key
        for key in required_top_level
        if key not in dataset
    ]

    if missing:
        raise RuntimeError(
            "Dataset is missing required fields: "
            + ", ".join(missing)
        )

    return dataset


def exact_distributed_samples(dataset: dict):
    # The JSON numbers originated from np.float32 values in the preparation
    # script. Re-quantize them to Float32 first so this diagnostic starts from
    # the exact same numeric representation received by Swift/Metal clients.
    times_float32 = np.asarray(
        dataset["times"],
        dtype=np.float32,
    )

    flux_float32 = np.asarray(
        dataset["flux"],
        dtype=np.float32,
    )

    if len(times_float32) != len(flux_float32):
        raise RuntimeError(
            "Time/flux sample count mismatch in dataset."
        )

    if len(times_float32) < 2:
        raise RuntimeError(
            "At least two samples are required."
        )

    if not np.all(np.isfinite(times_float32)):
        raise RuntimeError(
            "Dataset contains non-finite Float32 time values."
        )

    if not np.all(np.isfinite(flux_float32)):
        raise RuntimeError(
            "Dataset contains non-finite Float32 flux values."
        )

    # Astropy performs its numerical work in Float64, but these Float64 arrays
    # are derived from the exact distributed Float32 values rather than from
    # the original unquantized TESS samples.
    times = times_float32.astype(np.float64)
    flux = flux_float32.astype(np.float64)

    return times, flux


def build_frequency_grid(dataset: dict):
    search = dataset["frequencySearch"]

    minimum_frequency = float(
        search["minimumFrequency"]
    )
    maximum_frequency = float(
        search["maximumFrequency"]
    )
    frequency_step = float(
        search["frequencyStep"]
    )
    total_frequencies = int(
        search["totalFrequencies"]
    )

    if total_frequencies <= 0:
        raise RuntimeError(
            "Dataset frequency count must be positive."
        )

    if not math.isfinite(frequency_step) or frequency_step <= 0:
        raise RuntimeError(
            "Dataset frequency step is invalid."
        )

    frequencies = (
        minimum_frequency
        + np.arange(
            total_frequencies,
            dtype=np.float64,
        ) * frequency_step
    )

    expected_last_frequency = (
        minimum_frequency
        + (total_frequencies - 1) * frequency_step
    )

    if frequencies[-1] > maximum_frequency + frequency_step:
        raise RuntimeError(
            "Generated frequency grid exceeds the dataset search range."
        )

    if not math.isclose(
        float(frequencies[-1]),
        expected_last_frequency,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Frequency grid construction mismatch."
        )

    return frequencies


def calculate_periodogram(
    times: np.ndarray,
    flux: np.ndarray,
    frequencies: np.ndarray,
):
    periodogram = LombScargle(
        times,
        flux,
    )

    powers = np.asarray(
        periodogram.power(frequencies),
        dtype=np.float64,
    )

    if len(powers) != len(frequencies):
        raise RuntimeError(
            "Astropy returned an unexpected number of power values."
        )

    if not np.any(np.isfinite(powers)):
        raise RuntimeError(
            "Astropy returned no finite power values."
        )

    return periodogram, powers


def local_maximum_indices(powers: np.ndarray):
    if len(powers) < 3:
        return np.asarray(
            [int(np.nanargmax(powers))],
            dtype=np.int64,
        )

    finite = np.isfinite(powers)

    middle_is_peak = (
        finite[1:-1]
        & finite[:-2]
        & finite[2:]
        & (powers[1:-1] > powers[:-2])
        & (powers[1:-1] >= powers[2:])
    )

    indices = np.flatnonzero(
        middle_is_peak
    ) + 1

    global_index = int(
        np.nanargmax(powers)
    )

    if global_index not in indices:
        indices = np.append(
            indices,
            global_index,
        )

    return indices.astype(
        np.int64,
        copy=False,
    )


def select_distinct_peaks(
    frequencies: np.ndarray,
    powers: np.ndarray,
    candidate_indices: np.ndarray,
    minimum_frequency_separation: float,
    limit: int,
):
    order = candidate_indices[
        np.argsort(
            powers[candidate_indices]
        )[::-1]
    ]

    selected = []

    for index in order:
        frequency = float(
            frequencies[index]
        )

        if all(
            abs(
                frequency
                - float(frequencies[existing_index])
            ) >= minimum_frequency_separation
            for existing_index in selected
        ):
            selected.append(int(index))

        if len(selected) >= limit:
            break

    return selected


def nearest_grid_index(
    frequencies: np.ndarray,
    frequency: float,
):
    index = int(
        np.searchsorted(
            frequencies,
            frequency,
        )
    )

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
    frequencies: np.ndarray,
    powers: np.ndarray,
    center_frequency: float,
    half_width: float,
):
    mask = (
        frequencies
        >= center_frequency - half_width
    ) & (
        frequencies
        <= center_frequency + half_width
    )

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return None

    local = int(
        np.nanargmax(
            powers[indices]
        )
    )

    return int(indices[local])


def power_rank(powers: np.ndarray, power: float):
    finite_powers = powers[
        np.isfinite(powers)
    ]

    stronger_count = int(
        np.count_nonzero(
            finite_powers > power
        )
    )

    return (
        stronger_count + 1,
        len(finite_powers),
    )


def print_peak_table(
    title: str,
    indices,
    frequencies: np.ndarray,
    powers: np.ndarray,
    global_power: float,
):
    print()
    print(title)
    print(
        "   #   period (days)    frequency (c/d)      power       % of max"
    )
    print(
        "   --  ---------------  -----------------  ----------  --------"
    )

    for rank, index in enumerate(
        indices,
        start=1,
    ):
        frequency = float(
            frequencies[index]
        )
        period = 1.0 / frequency
        power = float(
            powers[index]
        )
        percent_of_max = (
            100.0 * power / global_power
            if global_power > 0
            else float("nan")
        )

        print(
            f"   {rank:>2}  "
            f"{period:>15.8f}  "
            f"{frequency:>17.8f}  "
            f"{power:>10.8f}  "
            f"{percent_of_max:>7.2f}%"
        )


def main():
    args = parse_args()

    dataset = load_dataset(
        args.dataset
    )

    times, flux = exact_distributed_samples(
        dataset
    )

    frequencies = build_frequency_grid(
        dataset
    )

    baseline_days = float(
        times[-1] - times[0]
    )

    if baseline_days <= 0:
        raise RuntimeError(
            "Dataset time baseline must be positive."
        )

    rayleigh_frequency = (
        1.0 / baseline_days
    )

    print()
    print("🔬 OpenStar Sector 28 Periodogram Diagnostic")
    print(f"   dataset: {dataset['id']}")
    print(
        "   target: "
        f"{dataset.get('targetName', 'unknown')}"
    )
    print(
        "   sector: "
        f"{dataset.get('source', {}).get('sector', 'unknown')}"
    )
    print(f"   samples: {len(times)}")
    print(
        "   numeric representation: "
        f"{dataset.get('numericRepresentation', 'unknown')}"
    )
    print(
        "   baseline: "
        f"{baseline_days:.8f} days"
    )
    print(
        "   Rayleigh frequency resolution (1/baseline): "
        f"{rayleigh_frequency:.8f} cycles/day"
    )
    print(
        "   frequency grid: "
        f"{frequencies[0]:.8f} - "
        f"{frequencies[-1]:.8f} cycles/day"
    )
    print(
        "   frequency bins: "
        f"{len(frequencies):,}"
    )

    print()
    print("🧪 Recomputing full Astropy periodogram")
    print("   input: exact distributed Float32 samples")

    periodogram, powers = calculate_periodogram(
        times,
        flux,
        frequencies,
    )

    global_index = int(
        np.nanargmax(powers)
    )
    global_frequency = float(
        frequencies[global_index]
    )
    global_period = (
        1.0 / global_frequency
    )
    global_power = float(
        powers[global_index]
    )

    print()
    print("⭐ Recomputed Astropy maximum")
    print(
        "   frequency: "
        f"{global_frequency:.8f} cycles/day"
    )
    print(
        "   period: "
        f"{global_period:.8f} days"
    )
    print(
        "   power: "
        f"{global_power:.8f}"
    )

    stored_reference = dataset.get(
        "reference",
        {},
    )

    stored_frequency = stored_reference.get(
        "bestFrequency"
    )
    stored_period = stored_reference.get(
        "bestPeriodDays"
    )
    stored_power = stored_reference.get(
        "bestPower"
    )

    if (
        stored_frequency is not None
        and stored_period is not None
        and stored_power is not None
    ):
        print()
        print("✅ Stored-reference consistency check")
        print(
            "   stored frequency: "
            f"{float(stored_frequency):.8f} cycles/day"
        )
        print(
            "   recomputed frequency error: "
            f"{abs(global_frequency - float(stored_frequency)):.12f} cycles/day"
        )
        print(
            "   stored period: "
            f"{float(stored_period):.8f} days"
        )
        print(
            "   recomputed period error: "
            f"{abs(global_period - float(stored_period)):.12f} days"
        )
        print(
            "   stored power: "
            f"{float(stored_power):.8f}"
        )
        print(
            "   recomputed power error: "
            f"{abs(global_power - float(stored_power)):.12f}"
        )

    external_period = EXTERNAL_TARS_PERIOD_DAYS
    external_frequency = (
        1.0 / external_period
    )
    external_exact_power = float(
        periodogram.power(
            external_frequency
        )
    )

    external_grid_index = nearest_grid_index(
        frequencies,
        external_frequency,
    )
    external_grid_frequency = float(
        frequencies[external_grid_index]
    )
    external_grid_period = (
        1.0 / external_grid_frequency
    )
    external_grid_power = float(
        powers[external_grid_index]
    )

    frequency_difference = abs(
        external_frequency
        - global_frequency
    )

    resolution_units = (
        frequency_difference
        / rayleigh_frequency
    )

    external_power_fraction = (
        external_exact_power / global_power
        if global_power > 0
        else float("nan")
    )

    rank, rank_total = power_rank(
        powers,
        external_grid_power,
    )

    print()
    print("🎯 External TARS Sector 28 diagnostic")
    print(
        "   TARS period: "
        f"{external_period:.8f} days"
    )
    print(
        "   TARS frequency: "
        f"{external_frequency:.8f} cycles/day"
    )
    print(
        "   exact Astropy power at TARS frequency: "
        f"{external_exact_power:.8f}"
    )
    print(
        "   power relative to Astropy maximum: "
        f"{100.0 * external_power_fraction:.2f}%"
    )
    print(
        "   nearest grid frequency: "
        f"{external_grid_frequency:.8f} cycles/day"
    )
    print(
        "   nearest grid period: "
        f"{external_grid_period:.8f} days"
    )
    print(
        "   nearest grid power: "
        f"{external_grid_power:.8f}"
    )
    print(
        "   nearest-grid power rank: "
        f"{rank:,}/{rank_total:,}"
    )
    print(
        "   frequency difference from global maximum: "
        f"{frequency_difference:.8f} cycles/day"
    )
    print(
        "   difference in Rayleigh-resolution units: "
        f"{resolution_units:.4f}"
    )

    if resolution_units < 1.0:
        print(
            "   interpretation: TARS 8.81 d and the Astropy maximum "
            "are separated by less than one Rayleigh frequency resolution."
        )
    else:
        print(
            "   interpretation: TARS 8.81 d and the Astropy maximum "
            "are separated by at least one Rayleigh frequency resolution."
        )

    nearby_index = strongest_in_frequency_window(
        frequencies,
        powers,
        external_frequency,
        rayleigh_frequency,
    )

    if nearby_index is not None:
        nearby_frequency = float(
            frequencies[nearby_index]
        )
        nearby_period = (
            1.0 / nearby_frequency
        )
        nearby_power = float(
            powers[nearby_index]
        )

        print()
        print("🔎 Strongest point within ±1 Rayleigh resolution of TARS")
        print(
            "   frequency: "
            f"{nearby_frequency:.8f} cycles/day"
        )
        print(
            "   period: "
            f"{nearby_period:.8f} days"
        )
        print(
            "   power: "
            f"{nearby_power:.8f}"
        )
        print(
            "   power relative to global maximum: "
            f"{100.0 * nearby_power / global_power:.2f}%"
        )

    candidate_indices = local_maximum_indices(
        powers
    )

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

    focus_frequency_min = (
        1.0 / FOCUS_MAX_PERIOD_DAYS
    )
    focus_frequency_max = (
        1.0 / FOCUS_MIN_PERIOD_DAYS
    )

    focus_candidates = candidate_indices[
        (
            frequencies[candidate_indices]
            >= focus_frequency_min
        )
        & (
            frequencies[candidate_indices]
            <= focus_frequency_max
        )
    ]

    focus_indices = focus_candidates[
        np.argsort(
            powers[focus_candidates]
        )[::-1]
    ][:TOP_DISTINCT_PEAKS]

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

    print()
    print("🏁 Diagnostic complete")
    print(
        "   Key question: compare the TARS power percentage and the "
        "Rayleigh-resolution separation above."
    )
    print(
        "   If 8.81 d retains power close to the maximum while lying "
        "inside one resolution element, the two reported periods are likely "
        "different estimates of the same broad single-sector feature."
    )


if __name__ == "__main__":
    main()
