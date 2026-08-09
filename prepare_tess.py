#!/usr/bin/env python3

from pathlib import Path
import json

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


TARGET = "RR Lyr"
DATASET_ID = "tess-rr-lyr"

OUTPUT_DIRECTORY = Path("data")
OUTPUT_FILE = OUTPUT_DIRECTORY / f"{DATASET_ID}.json"

MAX_SAMPLES = 5_000

MIN_FREQUENCY = 0.5
MAX_FREQUENCY = 5.0
TOTAL_FREQUENCIES = 32_768
FREQUENCIES_PER_WORK_UNIT = 512


def download_light_curve():
    print()
    print("🔭 Searching MAST")
    print(f"   target: {TARGET}")

    search = lk.search_lightcurve(
        TARGET,
        mission="TESS",
        author="SPOC",
        exptime=120
    )

    if len(search) == 0:
        print("   no 120-second SPOC result; trying any cadence")

        search = lk.search_lightcurve(
            TARGET,
            mission="TESS",
            author="SPOC"
        )

    if len(search) == 0:
        raise RuntimeError(
            f"No TESS light curves found for {TARGET}"
        )

    print(f"   products found: {len(search)}")
    print()
    print("⬇️ Downloading first light curve")

    light_curve = search[0].download()

    if light_curve is None:
        raise RuntimeError(
            "MAST returned no downloadable light curve."
        )

    return light_curve


def prepare_samples(light_curve):
    light_curve = light_curve.remove_nans()
    light_curve = light_curve.normalize()

    times = np.asarray(
        light_curve.time.value,
        dtype=np.float64
    )

    flux = np.asarray(
        light_curve.flux.value,
        dtype=np.float64
    )

    finite = (
        np.isfinite(times) &
        np.isfinite(flux)
    )

    times = times[finite]
    flux = flux[finite]

    if len(times) == 0:
        raise RuntimeError(
            "No valid samples remained after cleaning."
        )

    # Keep time values close to zero. This will matter when
    # the Metal implementation performs trig using Float.
    times = times - times[0]

    # Mean-center the normalized brightness measurements.
    flux = flux - np.mean(flux)

    original_count = len(times)

    if original_count > MAX_SAMPLES:
        indices = np.linspace(
            0,
            original_count - 1,
            MAX_SAMPLES,
            dtype=np.int64
        )

        times = times[indices]
        flux = flux[indices]

    print()
    print("✨ Light curve prepared")
    print(f"   original samples: {original_count}")
    print(f"   distributed samples: {len(times)}")
    print(f"   baseline: {times[-1]:.4f} days")

    return times, flux


def calculate_reference_period(times, flux):
    frequency_step = (
        MAX_FREQUENCY - MIN_FREQUENCY
    ) / TOTAL_FREQUENCIES

    frequencies = (
        MIN_FREQUENCY +
        np.arange(
            TOTAL_FREQUENCIES,
            dtype=np.float64
        ) * frequency_step
    )

    print()
    print("🧪 Calculating reference period")
    print(
        f"   frequency range: "
        f"{MIN_FREQUENCY:.3f} - "
        f"{MAX_FREQUENCY:.3f} cycles/day"
    )
    print(f"   frequencies: {TOTAL_FREQUENCIES}")

    periodogram = LombScargle(
        times,
        flux,
        center_data=False,
        fit_mean=False
    )

    powers = periodogram.power(
        frequencies,
        method="fast",
        normalization="standard"
    )

    best_index = int(
        np.argmax(powers)
    )

    best_frequency = float(
        frequencies[best_index]
    )

    best_power = float(
        powers[best_index]
    )

    best_period = (
        1.0 /
        best_frequency
    )

    print()
    print("⭐ Reference result")
    print(
        f"   frequency: "
        f"{best_frequency:.8f} cycles/day"
    )
    print(
        f"   period: "
        f"{best_period:.8f} days"
    )
    print(
        f"   power: "
        f"{best_power:.8f}"
    )

    return (
        frequency_step,
        best_frequency,
        best_period,
        best_power
    )


def save_dataset(
    times,
    flux,
    frequency_step,
    reference_frequency,
    reference_period,
    reference_power
):
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    work_unit_count = (
        TOTAL_FREQUENCIES +
        FREQUENCIES_PER_WORK_UNIT -
        1
    ) // FREQUENCIES_PER_WORK_UNIT

    dataset = {
        "id": DATASET_ID,
        "targetName": TARGET,
        "mission": "TESS",
        "source": "MAST",
        "timeUnit": "day",
        "fluxUnit": "normalized",
        "times": times.astype(
            np.float32
        ).tolist(),
        "flux": flux.astype(
            np.float32
        ).tolist(),
        "search": {
            "minimumFrequency": MIN_FREQUENCY,
            "maximumFrequency": MAX_FREQUENCY,
            "frequencyStep": frequency_step,
            "totalFrequencies": TOTAL_FREQUENCIES,
            "frequenciesPerWorkUnit":
                FREQUENCIES_PER_WORK_UNIT,
            "workUnitCount": work_unit_count
        },
        "reference": {
            "bestFrequency": reference_frequency,
            "bestPeriodDays": reference_period,
            "bestPower": reference_power
        }
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            dataset,
            file,
            separators=(",", ":")
        )

    print()
    print("💾 Dataset saved")
    print(f"   file: {OUTPUT_FILE}")
    print(
        f"   work units: "
        f"{work_unit_count}"
    )


def main():
    light_curve = download_light_curve()

    times, flux = prepare_samples(
        light_curve
    )

    (
        frequency_step,
        reference_frequency,
        reference_period,
        reference_power
    ) = calculate_reference_period(
        times,
        flux
    )

    save_dataset(
        times,
        flux,
        frequency_step,
        reference_frequency,
        reference_period,
        reference_power
    )

    print()
    print("🌟 TESS dataset ready for OpenStar")
    print()


if __name__ == "__main__":
    main()
