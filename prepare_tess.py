import json
import math
import re
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


def load_targets():
    with TARGETS_FILE.open(
            "r",
            encoding="utf-8",
    ) as file:
        return json.load(file)


def json_number(value):
    if value is None:
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def extract_tic_id(*values):
    for value in values:
        if value is None:
            continue

        match = re.search(
            r"\bTIC\s+(\d+)\b",
            str(value),
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def evenly_downsample(
        times,
        flux,
        max_samples,
):
    if len(times) <= max_samples:
        return times, flux

    indices = np.linspace(
        0,
        len(times) - 1,
        max_samples,
        dtype=np.int64,
        )

    return (
        times[indices],
        flux[indices],
    )


def prepare_light_curve(light_curve):
    original_samples = len(light_curve)

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

    if len(times) < 2:
        raise RuntimeError(
            "Light curve has too few finite samples."
        )

    order = np.argsort(times)

    times = times[order]
    flux = flux[order]

    times, flux = evenly_downsample(
        times,
        flux,
        MAX_SAMPLES,
    )

    # Shift the observation timeline to zero.
    times = times - times[0]

    flux_mean = float(
        np.mean(flux)
    )

    flux_stddev = float(
        np.std(flux)
    )

    if (
            not math.isfinite(flux_stddev)
            or flux_stddev <= 0
    ):
        raise RuntimeError(
            "Light curve flux has zero or invalid variance."
        )

    # Center and standardize the light curve.
    flux = (
                   flux - flux_mean
           ) / flux_stddev

    baseline_days = float(
        times[-1] - times[0]
    )

    return {
        "times": times,
        "flux": flux,
        "originalSamples": original_samples,
        "distributedSamples": len(times),
        "baselineDays": baseline_days,
        "originalFluxMean": flux_mean,
        "originalFluxStddev": flux_stddev,
    }


def build_frequency_grid():
    frequency_step = (
                             MAX_FREQUENCY - MIN_FREQUENCY
                     ) / TOTAL_FREQUENCIES

    frequencies = (
            MIN_FREQUENCY
            + np.arange(
        TOTAL_FREQUENCIES,
        dtype=np.float64,
    )
            * frequency_step
    )

    return frequencies, frequency_step


def calculate_reference(
        times,
        flux,
):
    print()
    print("🧪 Calculating Astropy reference")
    print(
        "   frequency range: "
        f"{MIN_FREQUENCY:.3f} - "
        f"{MAX_FREQUENCY:.3f} cycles/day"
    )
    print(
        f"   frequencies: "
        f"{TOTAL_FREQUENCIES}"
    )

    frequencies, frequency_step = (
        build_frequency_grid()
    )

    model = LombScargle(
        times,
        flux,
    )

    powers = model.power(
        frequencies
    )

    powers = np.asarray(
        powers,
        dtype=np.float64,
    )

    finite = np.isfinite(powers)

    if not np.any(finite):
        raise RuntimeError(
            "Astropy produced no finite powers."
        )

    safe_powers = np.where(
        finite,
        powers,
        -np.inf,
    )

    global_index = int(
        np.argmax(safe_powers)
    )

    best_frequency = float(
        frequencies[global_index]
    )

    best_period_days = float(
        1.0 / best_frequency
    )

    best_power = float(
        powers[global_index]
    )

    chunks = []

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

        chunk_powers = safe_powers[
            start_index:end_index
        ]

        relative_index = int(
            np.argmax(chunk_powers)
        )

        absolute_index = (
                start_index
                + relative_index
        )

        chunk_frequency = float(
            frequencies[absolute_index]
        )

        chunk_power = float(
            powers[absolute_index]
        )

        chunks.append(
            {
                "frequencyStartIndex": (
                    start_index
                ),
                "frequencyCount": (
                        end_index
                        - start_index
                ),
                "bestFrequency": (
                    chunk_frequency
                ),
                "bestPeriodDays": float(
                    1.0 / chunk_frequency
                ),
                "bestPower": (
                    chunk_power
                ),
            }
        )

    expected_chunks = math.ceil(
        TOTAL_FREQUENCIES
        / FREQUENCIES_PER_WORK_UNIT
    )

    if len(chunks) != expected_chunks:
        raise RuntimeError(
            "Astropy chunk reference generation "
            f"failed: {len(chunks)}/"
            f"{expected_chunks}"
        )

    print()
    print("⭐ Astropy reference result")
    print(
        "   frequency: "
        f"{best_frequency:.8f} cycles/day"
    )
    print(
        "   period: "
        f"{best_period_days:.8f} days"
    )
    print(
        "   power: "
        f"{best_power:.8f}"
    )
    print(
        "   work-unit references: "
        f"{len(chunks)}/{expected_chunks}"
    )

    return {
        "bestFrequency": (
            best_frequency
        ),
        "bestPeriodDays": (
            best_period_days
        ),
        "bestPower": (
            best_power
        ),
        "chunks": chunks,
    }, frequency_step


def search_light_curve(
        target,
):
    query = target["query"]

    author = target.get(
        "author",
        "SPOC",
    )

    sector = target.get(
        "sector"
    )

    print()
    print("🔭 Searching MAST")
    print(
        f"   target: "
        f"{target['targetName']}"
    )
    print(
        f"   query: "
        f"{query}"
    )
    print(
        f"   author: "
        f"{author}"
    )

    if sector is not None:
        print(
            f"   sector: "
            f"{sector}"
        )

    search_kwargs = {
        "mission": "TESS",
        "author": author,
    }

    if sector is not None:
        search_kwargs[
            "sector"
        ] = sector

    search_result = (
        lk.search_lightcurve(
            query,
            **search_kwargs,
        )
    )

    print(
        f"   products found: "
        f"{len(search_result)}"
    )

    if len(search_result) == 0:
        raise RuntimeError(
            "No TESS light curves found."
        )

    product_index = int(
        target.get(
            "productIndex",
            0,
        )
    )

    if (
            product_index < 0
            or product_index
            >= len(search_result)
    ):
        raise RuntimeError(
            f"productIndex {product_index} "
            f"is outside 0..{len(search_result) - 1}"
        )

    print()
    print(
        "⬇️ Downloading selected light curve"
    )
    print(
        f"   product index: "
        f"{product_index}"
    )

    selected = search_result[
        product_index
    ]

    light_curve = selected.download()

    if light_curve is None:
        raise RuntimeError(
            "Lightkurve download returned no light curve."
        )

    return (
        search_result,
        selected,
        light_curve,
    )


def selected_product_metadata(
        selected,
):
    metadata = {}

    try:
        table = selected.table

        if len(table) > 0:
            row = table[0]

            for key in (
                    "target_name",
                    "sequence_number",
                    "author",
                    "mission",
                    "exptime",
                    "distance",
            ):
                if key not in row.colnames:
                    continue

                value = row[key]

                try:
                    if np.ma.is_masked(value):
                        continue
                except TypeError:
                    pass

                metadata[key] = (
                    json_number(value)
                )

    except Exception:
        # Search-table metadata is useful but is not
        # required for the distributed science work.
        pass

    return metadata


def prepare_target(
        project,
        target,
):
    dataset_id = target[
        "datasetID"
    ]

    target_name = target[
        "targetName"
    ]

    (
        _,
        selected,
        light_curve,
    ) = search_light_curve(
        target
    )

    prepared = prepare_light_curve(
        light_curve
    )

    times = prepared[
        "times"
    ]

    flux = prepared[
        "flux"
    ]

    print()
    print("✨ Light curve prepared")
    print(
        "   original samples: "
        f"{prepared['originalSamples']}"
    )
    print(
        "   distributed samples: "
        f"{prepared['distributedSamples']}"
    )
    print(
        "   baseline: "
        f"{prepared['baselineDays']:.4f} days"
    )
    print(
        "   flux mean: "
        f"{float(np.mean(flux)):.8f}"
    )
    print(
        "   flux stddev: "
        f"{float(np.std(flux)):.8f}"
    )

    (
        reference,
        frequency_step,
    ) = calculate_reference(
        times,
        flux,
    )

    product_metadata = (
        selected_product_metadata(
            selected
        )
    )

    tic_id = target.get(
        "ticID"
    )

    if tic_id is None:
        tic_id = extract_tic_id(
            target_name,
            target.get("query"),
            product_metadata.get(
                "target_name"
            ),
        )

    sector = target.get(
        "sector"
    )

    if sector is None:
        sector = product_metadata.get(
            "sequence_number"
        )

    if sector is not None:
        try:
            sector = int(
                sector
            )
        except (
                TypeError,
                ValueError,
        ):
            sector = None

    work_unit_count = (
        len(
            reference["chunks"]
        )
    )

    dataset_payload = {
        # ------------------------------------------------------------------
        # Swift AstronomyDataset contract
        # ------------------------------------------------------------------
        "id": dataset_id,
        "targetName": target_name,
        "mission": "TESS",
        "timeUnit": "days",
        "fluxUnit": "standardized normalized flux",

        "times": [
            float(value)
            for value
            in times.astype(
                np.float32
            )
        ],

        "flux": [
            float(value)
            for value
            in flux.astype(
                np.float32
            )
        ],

        # ------------------------------------------------------------------
        # Server/science metadata
        # ------------------------------------------------------------------
        "metadata": {
            "query": target[
                "query"
            ],
            "author": target.get(
                "author",
                "SPOC",
            ),
            "ticID": tic_id,
            "sector": sector,
            "sampleCount": (
                prepared[
                    "distributedSamples"
                ]
            ),
            "originalSampleCount": (
                prepared[
                    "originalSamples"
                ]
            ),
            "baselineDays": (
                prepared[
                    "baselineDays"
                ]
            ),
        },

        # ------------------------------------------------------------------
        # Distributed frequency-search definition
        # ------------------------------------------------------------------
        "frequencySearch": {
            "minimumFrequency": (
                MIN_FREQUENCY
            ),
            "maximumFrequency": (
                MAX_FREQUENCY
            ),
            "frequencyStep": (
                frequency_step
            ),
            "totalFrequencies": (
                TOTAL_FREQUENCIES
            ),
            "frequenciesPerWorkUnit": (
                FREQUENCIES_PER_WORK_UNIT
            ),
            "workUnitCount": (
                work_unit_count
            ),
        },

        # ------------------------------------------------------------------
        # Canonical Astropy validation data.
        #
        # coordinator_v5 reads THIS exact object.
        # ------------------------------------------------------------------
        "reference": reference,
    }

    output_path = (
            DATA_DIR
            / f"{dataset_id}.json"
    )

    with output_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            dataset_payload,
            file,
            separators=(",", ":"),
            allow_nan=False,
        )

    print()
    print("💾 Dataset saved")
    print(
        f"   file: "
        f"{output_path}"
    )
    print(
        f"   work units: "
        f"{work_unit_count}"
    )
    print(
        "   Astropy references: "
        f"{len(reference['chunks'])}/"
        f"{work_unit_count}"
    )

    return {
        "id": dataset_id,
        "targetName": target_name,
        "mission": "TESS",
        "ticID": tic_id,
        "sector": sector,
        "path": str(
            output_path
        ),
        "sampleCount": (
            prepared[
                "distributedSamples"
            ]
        ),
        "workUnitCount": (
            work_unit_count
        ),
    }


def write_project_manifest(
        project,
        datasets,
):
    manifest = {
        "id": project[
            "projectID"
        ],
        "name": project.get(
            "projectName",
            project["projectID"],
        ),
        "workloadID": project[
            "workloadID"
        ],
        "createdAt": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "datasets": datasets,
    }

    output_path = (
            PROJECTS_DIR
            / (
                f"{project['projectID']}"
                ".json"
            )
    )

    with output_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            allow_nan=False,
        )

    print()
    print("📦 Project manifest saved")
    print(
        f"   file: "
        f"{output_path}"
    )
    print(
        f"   datasets: "
        f"{len(datasets)}"
    )
    print(
        "   total work units: "
        f"{sum(item['workUnitCount'] for item in datasets)}"
    )

    return output_path


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROJECTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    project = load_targets()

    required_project_fields = (
        "projectID",
        "workloadID",
        "targets",
    )

    for field in required_project_fields:
        if field not in project:
            raise RuntimeError(
                f"targets.json missing "
                f"required field: {field}"
            )

    datasets = []
    failures = []

    print()
    print("⭐ OpenStar TESS Preprocessor")
    print(
        f"Project: "
        f"{project['projectID']}"
    )
    print(
        f"Targets: "
        f"{len(project['targets'])}"
    )

    for target in project[
        "targets"
    ]:
        try:
            dataset_entry = (
                prepare_target(
                    project,
                    target,
                )
            )

            datasets.append(
                dataset_entry
            )

            print()
            print(
                "🌟 Target ready for OpenStar"
            )
            print(
                f"   target: "
                f"{target['targetName']}"
            )

        except Exception as error:
            dataset_id = target.get(
                "datasetID",
                "unknown",
            )

            target_name = target.get(
                "targetName",
                dataset_id,
            )

            failures.append(
                (
                    target_name,
                    str(error),
                )
            )

            print()
            print(
                "❌ Target preparation failed"
            )
            print(
                f"   target: "
                f"{target_name}"
            )
            print(
                f"   error: "
                f"{error}"
            )

    if not datasets:
        raise RuntimeError(
            "All TESS targets failed."
        )

    manifest_path = (
        write_project_manifest(
            project,
            datasets,
        )
    )

    print()
    print("✅ OpenStar TESS project ready")
    print(
        f"   project: "
        f"{project['projectID']}"
    )
    print(
        f"   manifest: "
        f"{manifest_path}"
    )
    print(
        f"   datasets prepared: "
        f"{len(datasets)}"
    )

    if failures:
        print(
            f"   datasets failed: "
            f"{len(failures)}"
        )

        for (
                target_name,
                error,
        ) in failures:
            print(
                f"      {target_name}: "
                f"{error}"
            )


if __name__ == "__main__":
    main()