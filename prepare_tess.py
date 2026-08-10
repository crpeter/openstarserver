#!/usr/bin/env python3

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


TARGETS_FILE = Path("targets.json")

DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"

MAX_SAMPLES = 18_000

MIN_FREQUENCY = 0.5
MAX_FREQUENCY = 5.0

TOTAL_FREQUENCIES = 4_194_304
FREQUENCIES_PER_WORK_UNIT = 4_096


def load_target_catalog():
    with TARGETS_FILE.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    required = (
        "projectID",
        "projectName",
        "workloadID",
        "targets",
    )

    missing = [
        key
        for key in required
        if key not in catalog
    ]

    if missing:
        raise ValueError(
            f"targets.json is missing: {', '.join(missing)}"
        )

    if not catalog["targets"]:
        raise ValueError(
            "targets.json contains no targets"
        )

    return catalog


def search_light_curve(target):
    query = target["query"]
    author = target.get("author", "SPOC")
    sector = target.get("sector")

    search_kwargs = {
        "mission": "TESS",
        "author": author,
    }

    if sector is not None:
        search_kwargs["sector"] = sector

    print()
    print("🔭 Searching MAST")
    print(f"   target: {target['targetName']}")
    print(f"   query: {query}")

    result = lk.search_lightcurve(
        query,
        **search_kwargs,
    )

    print(f"   products found: {len(result)}")

    if len(result) == 0:
        raise RuntimeError(
            f"No TESS light curves found for {query}"
        )

    product_index = int(
        target.get("productIndex", 0)
    )

    if (
            product_index < 0
            or product_index >= len(result)
    ):
        raise RuntimeError(
            f"productIndex {product_index} is invalid "
            f"for {query}; {len(result)} products were found"
        )

    print(
        f"⬇️ Downloading product {product_index}"
    )

    light_curve = result[
        product_index
    ].download()

    if light_curve is None:
        raise RuntimeError(
            f"Failed to download TESS light curve for {query}"
        )

    return light_curve, product_index


def prepare_light_curve(light_curve):
    light_curve = light_curve.remove_nans()

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

    if len(times) < 3:
        raise RuntimeError(
            "Light curve has fewer than 3 usable samples"
        )

    order = np.argsort(times)

    times = times[order]
    flux = flux[order]

    original_sample_count = len(times)

    # Keep values near zero so Float precision remains
    # good on the Apple-device Metal implementation.
    times = times - times[0]

    if len(times) > MAX_SAMPLES:
        indices = np.linspace(
            0,
            len(times) - 1,
            MAX_SAMPLES,
            dtype=np.int64,
            )

        times = times[indices]
        flux = flux[indices]

    flux = flux - np.mean(flux)

    flux_stddev = float(
        np.std(flux)
    )

    if (
            not math.isfinite(flux_stddev)
            or flux_stddev <= 0
    ):
        raise RuntimeError(
            "Light curve flux has zero or invalid standard deviation"
        )

    flux = flux / flux_stddev

    baseline_days = float(
        times[-1] - times[0]
    )

    print("✨ Light curve prepared")
    print(
        f"   original samples: "
        f"{original_sample_count}"
    )
    print(
        f"   distributed samples: "
        f"{len(times)}"
    )
    print(
        f"   baseline: "
        f"{baseline_days:.4f} days"
    )
    print(
        f"   flux mean: "
        f"{np.mean(flux):.8f}"
    )
    print(
        f"   flux stddev: "
        f"{np.std(flux):.8f}"
    )

    return (
        times,
        flux,
        original_sample_count,
        baseline_days,
    )


def calculate_reference(
        times,
        flux,
):
    frequency_step = (
                             MAX_FREQUENCY
                             - MIN_FREQUENCY
                     ) / TOTAL_FREQUENCIES

    frequencies = (
            MIN_FREQUENCY
            + np.arange(
        TOTAL_FREQUENCIES,
        dtype=np.float64,
    ) * frequency_step
    )

    print(
        "🧪 Calculating reference periodogram"
    )
    print(
        "   frequency range: "
        f"{MIN_FREQUENCY:.3f} - "
        f"{MAX_FREQUENCY:.3f} cycles/day"
    )
    print(
        f"   frequencies: "
        f"{TOTAL_FREQUENCIES}"
    )

    powers = LombScargle(
        times,
        flux,
    ).power(
        frequencies
    )

    best_index = int(
        np.argmax(powers)
    )

    best_frequency = float(
        frequencies[best_index]
    )

    best_period_days = (
            1.0 / best_frequency
    )

    best_power = float(
        powers[best_index]
    )

    chunk_references = []

    for start_index in range(
            0,
            TOTAL_FREQUENCIES,
            FREQUENCIES_PER_WORK_UNIT,
    ):
        end_index = min(
            start_index
            + FREQUENCIES_PER_WORK_UNIT,
            TOTAL_FREQUENCIES,
            )

        local_powers = powers[
            start_index:end_index
        ]

        local_offset = int(
            np.argmax(local_powers)
        )

        local_index = (
                start_index
                + local_offset
        )

        chunk_references.append(
            {
                "frequencyStartIndex":
                    start_index,
                "bestFrequency":
                    float(
                        frequencies[
                            local_index
                        ]
                    ),
                "bestPower":
                    float(
                        powers[
                            local_index
                        ]
                    ),
            }
        )

    print("⭐ Reference result")
    print(
        f"   frequency: "
        f"{best_frequency:.8f} cycles/day"
    )
    print(
        f"   period: "
        f"{best_period_days:.8f} days"
    )
    print(
        f"   power: "
        f"{best_power:.8f}"
    )

    return {
        "frequencyStep":
            frequency_step,
        "globalBestFrequency":
            best_frequency,
        "globalBestPeriodDays":
            best_period_days,
        "globalBestPower":
            best_power,
        "chunks":
            chunk_references,
    }


def metadata_value(
        light_curve,
        *keys,
):
    for key in keys:
        value = light_curve.meta.get(
            key
        )

        if value is None:
            continue

        if isinstance(
                value,
                np.generic,
        ):
            value = value.item()

        if isinstance(
                value,
                (
                        str,
                        int,
                        float,
                        bool,
                ),
        ):
            return value

    return None


def build_dataset(target):
    (
        light_curve,
        product_index,
    ) = search_light_curve(
        target
    )

    (
        times,
        flux,
        original_sample_count,
        baseline_days,
    ) = prepare_light_curve(
        light_curve
    )

    reference = calculate_reference(
        times,
        flux,
    )

    dataset_id = target[
        "datasetID"
    ]

    tic_id = target.get(
        "ticID"
    )

    if tic_id is None:
        tic_id = metadata_value(
            light_curve,
            "TICID",
            "TIC_ID",
            "TARGETID",
            "TARGET_ID",
        )

    sector = target.get(
        "sector"
    )

    if sector is None:
        sector = metadata_value(
            light_curve,
            "SECTOR",
        )

    dataset = {
        "id":
            dataset_id,

        "targetName":
            target["targetName"],

        "mission":
            "TESS",

        "timeUnit":
            "days",

        "fluxUnit":
            "standardized relative flux",

        "times":
            times.astype(
                np.float32
            ).tolist(),

        "flux":
            flux.astype(
                np.float32
            ).tolist(),

        "metadata": {
            "query":
                target["query"],

            "ticID":
                tic_id,

            "sector":
                sector,

            "author":
                target.get(
                    "author",
                    "SPOC",
                ),

            "productIndex":
                product_index,

            "originalSampleCount":
                original_sample_count,

            "distributedSampleCount":
                len(times),

            "baselineDays":
                baseline_days,
        },

        "frequencySearch": {
            "minimumFrequency":
                MIN_FREQUENCY,

            "maximumFrequency":
                MAX_FREQUENCY,

            "totalFrequencies":
                TOTAL_FREQUENCIES,

            "frequenciesPerWorkUnit":
                FREQUENCIES_PER_WORK_UNIT,

            "frequencyStep":
                reference[
                    "frequencyStep"
                ],
        },

        "reference": {
            "globalBestFrequency":
                reference[
                    "globalBestFrequency"
                ],

            "globalBestPeriodDays":
                reference[
                    "globalBestPeriodDays"
                ],

            "globalBestPower":
                reference[
                    "globalBestPower"
                ],

            "chunks":
                reference["chunks"],
        },
    }

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
            DATA_DIR
            / f"{dataset_id}.json"
    )

    with dataset_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            dataset,
            file,
            separators=(",", ":"),
        )

    work_unit_count = math.ceil(
        TOTAL_FREQUENCIES
        / FREQUENCIES_PER_WORK_UNIT
    )

    print("💾 Dataset saved")
    print(
        f"   file: "
        f"{dataset_path}"
    )
    print(
        f"   work units: "
        f"{work_unit_count}"
    )

    return {
        "id":
            dataset_id,

        "targetName":
            target["targetName"],

        "mission":
            "TESS",

        "ticID":
            tic_id,

        "sector":
            sector,

        "path":
            str(dataset_path),

        "sampleCount":
            len(times),

        "workUnitCount":
            work_unit_count,
    }


def build_project_manifest(
        catalog,
        dataset_entries,
):
    PROJECTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "id":
            catalog["projectID"],

        "name":
            catalog["projectName"],

        "workloadID":
            catalog["workloadID"],

        "createdAt":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "datasets":
            dataset_entries,
    }

    project_path = (
            PROJECTS_DIR
            / f"{catalog['projectID']}.json"
    )

    with project_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print(
        "📋 Project manifest saved"
    )
    print(
        f"   file: "
        f"{project_path}"
    )
    print(
        f"   datasets: "
        f"{len(dataset_entries)}"
    )
    print(
        "   work units: "
        f"{sum(
            entry['workUnitCount']
            for entry
            in dataset_entries
        )}"
    )

    return project_path


def main():
    catalog = load_target_catalog()

    print(
        "⭐ OpenStar TESS Multi-Target Preprocessor"
    )
    print(
        f"   project: "
        f"{catalog['projectID']}"
    )
    print(
        f"   targets: "
        f"{len(catalog['targets'])}"
    )

    dataset_entries = []

    for target in catalog[
        "targets"
    ]:
        try:
            dataset_entries.append(
                build_dataset(
                    target
                )
            )

        except Exception as error:
            print()
            print("❌ Target failed")
            print(
                "   target: "
                f"{target.get(
                    'targetName',
                    target.get('query')
                )}"
            )
            print(
                f"   error: "
                f"{error}"
            )

    if not dataset_entries:
        raise RuntimeError(
            "No datasets were successfully prepared"
        )

    project_path = (
        build_project_manifest(
            catalog,
            dataset_entries,
        )
    )

    print()
    print(
        "🌟 Multi-target TESS project ready"
    )
    print(
        "   start with: "
        "python3 coordinator.py "
        f"--project {project_path}"
    )


if __name__ == "__main__":
    main()