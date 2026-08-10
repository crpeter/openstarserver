import json
import math
from pathlib import Path

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar TESS scientific validation project
# ============================================================

PROJECT_ID = "openstar.tess-validation-v1"
PROJECT_NAME = "OpenStar TESS Scientific Validation v1"
WORKLOAD_ID = "openstar.tess-period-search.v1"

DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"

MAX_SAMPLES = 18_000

# The previous 0.5 cycles/day lower bound could not detect periods
# longer than 2 days. This range covers periods up to 10 days.
MINIMUM_FREQUENCY = 0.1
MAXIMUM_FREQUENCY = 5.0

TOTAL_FREQUENCIES = 4_194_304
FREQUENCIES_PER_WORK_UNIT = 4_096

PREFERRED_AUTHOR = "SPOC"
FALLBACK_AUTHOR = "TESS-SPOC"
PREFERRED_EXPTIME_SECONDS = 120

# Product selections are populated by the startup preflight so expensive
# Astropy work never begins until every target has a usable TESS product.
SEARCH_SELECTION_CACHE = {}


# ============================================================
# Targets
# ============================================================

TARGETS = [
    {
        "id": "tess-rr-lyr",
        "targetName": "RR Lyr",
        "query": "TIC 159717514",
        "ticID": 159717514,
        "sector": None,
        "science": {
            "role": "known",
            "classification": "RRab",
            "publishedPeriodDays": 0.5668,
            "answerKeySource": "published/catalog RR Lyr period",
        },
    },
    {
        "id": "tess-v473-lyr",
        "targetName": "V473 Lyr",
        "query": "TIC 403786081",
        "ticID": 403786081,
        "sector": 14,
        "science": {
            "role": "known",
            "classification": "DCEP (second overtone)",
            "publishedPeriodDays": 1.490780,
            "answerKeySource": "published V473 Lyr pulsation period",
        },
    },
    {
        "id": "tess-au-mic",
        "targetName": "AU Mic",
        "query": "TIC 441420236",
        "ticID": 441420236,
        "sector": 1,
        "science": {
            "role": "known",
            "classification": "rotational variable",
            "publishedPeriodDays": 4.848,
            "answerKeySource": "catalog stellar rotation period",
        },
    },
    {
        "id": "tess-tic-199716496",
        "targetName": "TIC 199716496",
        "query": "TIC 199716496",
        "ticID": 199716496,
        "sector": 14,
        "science": {
            "role": "known",
            "classification": "EA eclipsing binary",
            "publishedPeriodDays": 1.04583737,
            "answerKeySource": "published eclipsing-binary orbital period",
        },
    },
    {
        "id": "tess-pi-men",
        "targetName": "Pi Mensae",
        "query": "TIC 261136679",
        "ticID": 261136679,
        "sector": 1,
        "science": {
            "role": "control",
        },
    },
    {
        "id": "tess-toi-1080",
        "targetName": "TOI-1080",
        "query": "TIC 161032923",
        "ticID": 161032923,
        "sector": 13,
        "science": {
            "role": "control",
        },
    },
    {
        "id": "tess-toi-561",
        "targetName": "TOI-561",
        "query": "TIC 377064495",
        "ticID": 377064495,
        "sector": 8,
        "science": {
            "role": "control",
        },
    },
    {
        "id": "tess-blind-a",
        "targetName": "Blind A",
        "query": "TIC 25165839",
        "ticID": 25165839,
        "sector": 1,
        "science": {
            "role": "blind",
        },
    },
]


def frequency_step() -> float:
    return (
            MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY
    ) / TOTAL_FREQUENCIES


def expected_work_unit_count() -> int:
    return math.ceil(
        TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT
    )


def _search_products(
        target: dict,
        *,
        author: str,
        exptime: int | None,
):
    kwargs = {
        "mission": "TESS",
        "author": author,
    }

    if target.get("sector") is not None:
        kwargs["sector"] = int(target["sector"])

    if exptime is not None:
        kwargs["exptime"] = exptime

    return lk.search_lightcurve(
        target["query"],
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
    if len(search_result) == 0:
        raise RuntimeError(
            "Cannot read cadence from an empty search result."
        )

    value = search_result.exptime[0]

    try:
        return float(value)
    except (TypeError, ValueError):
        if hasattr(value, "value"):
            return float(value.value)
        raise


def _diagnostic_search(target: dict):
    kwargs = {
        "mission": "TESS",
    }

    if target.get("sector") is not None:
        kwargs["sector"] = int(target["sector"])

    return lk.search_lightcurve(
        target["query"],
        **kwargs,
    )


def search_light_curve(target: dict):
    cached = SEARCH_SELECTION_CACHE.get(target["id"])

    if cached is not None:
        print()
        print("🔭 Using preflight MAST selection")
        print(f"   target: {target['targetName']}")
        print(f"   author: {cached[1]}")
        print(f"   cadence: {cached[2]:.0f}s")
        return cached

    print()
    print("🔭 Searching MAST")
    print(f"   target: {target['targetName']}")
    print(f"   query: {target['query']}")

    if target.get("sector") is not None:
        print(f"   requested sector: {target['sector']}")

    print(f"   preferred author: {PREFERRED_AUTHOR}")
    print(
        "   preferred cadence: "
        f"{PREFERRED_EXPTIME_SECONDS}s"
    )

    preferred = _search_products(
        target,
        author=PREFERRED_AUTHOR,
        exptime=PREFERRED_EXPTIME_SECONDS,
    )

    if len(preferred) > 0:
        print(
            f"   preferred products found: "
            f"{len(preferred)}"
        )

        selected = _select_shortest_cadence(
            preferred
        )

        cadence_seconds = (
            _selected_exptime_seconds(selected)
        )

        print(
            f"   selected author: "
            f"{PREFERRED_AUTHOR}"
        )
        print(
            "   selected cadence: "
            f"{cadence_seconds:.0f}s"
        )

        selection = (
            selected,
            PREFERRED_AUTHOR,
            cadence_seconds,
        )

        SEARCH_SELECTION_CACHE[target["id"]] = selection
        return selection

    print(
        "   no 120s SPOC product; "
        "trying TESS-SPOC FFI light curves"
    )

    fallback = _search_products(
        target,
        author=FALLBACK_AUTHOR,
        exptime=None,
    )

    if len(fallback) > 0:
        print(
            f"   fallback products found: "
            f"{len(fallback)}"
        )

        selected = _select_shortest_cadence(
            fallback
        )

        cadence_seconds = (
            _selected_exptime_seconds(selected)
        )

        print(
            f"   selected author: "
            f"{FALLBACK_AUTHOR}"
        )
        print(
            "   selected cadence: "
            f"{cadence_seconds:.0f}s"
        )

        selection = (
            selected,
            FALLBACK_AUTHOR,
            cadence_seconds,
        )

        SEARCH_SELECTION_CACHE[target["id"]] = selection
        return selection

    available = _diagnostic_search(target)

    print()
    print("❌ No usable SPOC-family product")
    print(f"   target: {target['targetName']}")

    if len(available) > 0:
        print()
        print("Available TESS light-curve products:")
        print(available)

    raise RuntimeError(
        "No SPOC or TESS-SPOC TESS light curve "
        f"found for {target['targetName']}."
    )


def preflight_targets():
    print()
    print("🔎 Preflighting all TESS targets")
    print("   no downloads or Astropy calculations yet")

    failures = []

    for target in TARGETS:
        try:
            search_result, author, cadence_seconds = search_light_curve(target)

            if len(search_result) == 0:
                raise RuntimeError(
                    "empty search result"
                )

            print(
                "   preflight ready: "
                f"{target['targetName']} -> "
                f"{author}, {cadence_seconds:.0f}s"
            )

        except Exception as error:
            failures.append(
                f"{target['targetName']}: {error}"
            )

    if failures:
        raise RuntimeError(
            "TESS target preflight failed. "
            "No datasets were processed:\n - "
            + "\n - ".join(failures)
        )

    print()
    print("✅ TESS target preflight complete")
    print(
        f"   usable targets: "
        f"{len(TARGETS)}/{len(TARGETS)}"
    )


def _extract_sector(
        light_curve,
        target: dict,
):
    sector = getattr(
        light_curve,
        "sector",
        None,
    )

    if sector is None:
        meta = getattr(
            light_curve,
            "meta",
            {},
        )

        sector = meta.get(
            "SECTOR"
        )

    if sector is None:
        sector = target.get(
            "sector"
        )

    if sector is None:
        return None

    try:
        return int(sector)
    except (TypeError, ValueError):
        return sector


def download_light_curve(target: dict):
    (
        search_result,
        source_author,
        cadence_seconds,
    ) = search_light_curve(target)

    print()
    print("⬇️ Downloading selected light curve")
    print(f"   author: {source_author}")
    print(
        "   cadence: "
        f"{cadence_seconds:.0f}s"
    )

    light_curve = search_result.download(
        quality_bitmask="default"
    )

    if light_curve is None:
        raise RuntimeError(
            "Download failed for "
            f"{target['targetName']}."
        )

    actual_sector = _extract_sector(
        light_curve,
        target,
    )

    print(
        "   selected sector: "
        f"{actual_sector if actual_sector is not None else 'unknown'}"
    )

    return (
        light_curve,
        {
            "author": source_author,
            "cadenceSeconds": cadence_seconds,
            "sector": actual_sector,
        },
    )


def prepare_light_curve(light_curve):
    original_samples = len(light_curve)

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

    if len(times) == 0:
        raise RuntimeError(
            "Light curve contains no finite samples."
        )

    order = np.argsort(times)

    times = times[order]
    flux = flux[order]

    finite_samples = len(times)

    # Preserve the full time baseline. If we need to cap the payload,
    # select evenly spaced samples across the entire light curve instead
    # of truncating the end of the sector.
    if len(times) > MAX_SAMPLES:
        indices = np.linspace(
            0,
            len(times) - 1,
            MAX_SAMPLES,
            dtype=np.int64,
            )

        times = times[indices]
        flux = flux[indices]

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
            "Light curve flux has invalid standard deviation."
        )

    flux = (
                   flux - flux_mean
           ) / flux_stddev

    baseline_days = (
        float(times[-1] - times[0])
        if len(times) > 1
        else 0.0
    )

    print()
    print("✨ Light curve prepared")
    print(
        f"   original samples: "
        f"{original_samples}"
    )
    print(
        f"   finite samples: "
        f"{finite_samples}"
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
        f"{float(np.mean(flux)):.8f}"
    )
    print(
        f"   flux stddev: "
        f"{float(np.std(flux)):.8f}"
    )

    return times, flux


def calculate_astropy_reference(
        times: np.ndarray,
        flux: np.ndarray,
):
    step = frequency_step()

    frequencies = (
            MINIMUM_FREQUENCY
            + np.arange(
        TOTAL_FREQUENCIES,
        dtype=np.float64,
    ) * step
    )

    print()
    print("🧪 Calculating Astropy reference")
    print(
        "   frequency range: "
        f"{MINIMUM_FREQUENCY:.3f} - "
        f"{MAXIMUM_FREQUENCY:.3f} cycles/day"
    )
    print(
        f"   frequencies: "
        f"{len(frequencies)}"
    )

    periodogram = LombScargle(
        times,
        flux,
    )

    powers = periodogram.power(
        frequencies
    )

    if len(powers) != TOTAL_FREQUENCIES:
        raise RuntimeError(
            "Astropy returned unexpected frequency count."
        )

    finite_power = np.isfinite(powers)

    if not np.any(finite_power):
        raise RuntimeError(
            "Astropy returned no finite Lomb-Scargle powers."
        )

    global_index = int(
        np.nanargmax(powers)
    )

    best_frequency = float(
        frequencies[global_index]
    )

    best_power = float(
        powers[global_index]
    )

    best_period_days = (
            1.0 / best_frequency
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

        chunk_powers = powers[
            start_index:end_index
        ]

        if not np.any(
                np.isfinite(chunk_powers)
        ):
            raise RuntimeError(
                "Astropy returned no finite power values for "
                f"frequency chunk starting at {start_index}."
            )

        local_index = int(
            np.nanargmax(chunk_powers)
        )

        absolute_index = (
                start_index + local_index
        )

        chunk_frequency = float(
            frequencies[absolute_index]
        )

        chunk_power = float(
            powers[absolute_index]
        )

        chunks.append(
            {
                "frequencyStartIndex": start_index,
                "frequencyCount": (
                        end_index - start_index
                ),
                "bestFrequency": chunk_frequency,
                "bestPeriodDays": (
                        1.0 / chunk_frequency
                ),
                "bestPower": chunk_power,
            }
        )

    expected_chunks = expected_work_unit_count()

    if len(chunks) != expected_chunks:
        raise RuntimeError(
            "Astropy chunk-reference count mismatch: "
            f"{len(chunks)}/{expected_chunks}"
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
        "bestFrequency": best_frequency,
        "bestPeriodDays": best_period_days,
        "bestPower": best_power,
        "chunks": chunks,
    }


def validate_science_metadata(
        target: dict,
):
    science = dict(
        target.get(
            "science",
            {},
        )
    )

    role = science.get("role")

    if role not in (
            "known",
            "control",
            "blind",
    ):
        raise RuntimeError(
            "Invalid science role for "
            f"{target['targetName']}: {role}"
        )

    if role == "known":
        if not science.get(
                "classification"
        ):
            raise RuntimeError(
                "Known target is missing classification: "
                f"{target['targetName']}"
            )

        if science.get(
                "publishedPeriodDays"
        ) is None:
            raise RuntimeError(
                "Known target is missing published period: "
                f"{target['targetName']}"
            )

    if role == "blind":
        forbidden = (
            "classification",
            "publishedPeriodDays",
            "publishedFrequency",
            "answerKeySource",
        )

        leaked = [
            key
            for key in forbidden
            if science.get(key) is not None
        ]

        if leaked:
            raise RuntimeError(
                "Blind target contains answer-key metadata: "
                + ", ".join(leaked)
            )

    return science


def build_dataset(
        target: dict,
        times: np.ndarray,
        flux: np.ndarray,
        reference: dict,
        source_metadata: dict,
):
    science = validate_science_metadata(
        target
    )

    if len(times) != len(flux):
        raise RuntimeError(
            "Time/flux sample count mismatch."
        )

    if len(times) == 0:
        raise RuntimeError(
            "Cannot build an empty dataset."
        )

    return {
        "id": target["id"],
        "targetName": target["targetName"],
        "mission": "TESS",
        "source": {
            "archive": "MAST",
            "author": source_metadata[
                "author"
            ],
            "ticID": target["ticID"],
            "sector": source_metadata[
                "sector"
            ],
            "cadenceSeconds": source_metadata[
                "cadenceSeconds"
            ],
        },
        "science": science,
        "timeUnit": "days",
        "fluxUnit": "normalized",
        "times": [
            float(value)
            for value in times
        ],
        "flux": [
            float(value)
            for value in flux
        ],
        "frequencySearch": {
            "minimumFrequency": MINIMUM_FREQUENCY,
            "maximumFrequency": MAXIMUM_FREQUENCY,
            "frequencyStep": frequency_step(),
            "totalFrequencies": TOTAL_FREQUENCIES,
            "frequenciesPerWorkUnit": (
                FREQUENCIES_PER_WORK_UNIT
            ),
        },
        "reference": reference,
    }


def write_dataset(
        target: dict,
        dataset: dict,
):
    output_path = (
            DATA_DIR
            / f"{target['id']}.json"
    )

    with output_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            dataset,
            file,
            indent=2,
            allow_nan=False,
        )

    reference_count = len(
        dataset["reference"]["chunks"]
    )

    print()
    print("💾 Dataset saved")
    print(f"   file: {output_path}")
    print(
        "   work units: "
        f"{expected_work_unit_count()}"
    )
    print(
        "   Astropy references: "
        f"{reference_count}/"
        f"{expected_work_unit_count()}"
    )

    return output_path


def prepare_target(target: dict):
    validate_science_metadata(target)

    (
        light_curve,
        source_metadata,
    ) = download_light_curve(
        target
    )

    times, flux = prepare_light_curve(
        light_curve
    )

    reference = calculate_astropy_reference(
        times,
        flux,
    )

    dataset = build_dataset(
        target,
        times,
        flux,
        reference,
        source_metadata,
    )

    output_path = write_dataset(
        target,
        dataset,
    )

    print()
    print("🌟 Target ready for OpenStar")
    print(
        f"   target: "
        f"{target['targetName']}"
    )
    print(
        "   source: "
        f"{source_metadata['author']}"
    )
    print(
        "   cadence: "
        f"{source_metadata['cadenceSeconds']:.0f}s"
    )
    print(
        "   sector: "
        f"{source_metadata['sector']}"
    )

    return dataset, output_path


def write_project_manifest(
        prepared_targets,
):
    datasets = []

    for (
            target,
            dataset,
            output_path,
    ) in prepared_targets:
        datasets.append(
            {
                "id": target["id"],
                "path": str(output_path),
                "targetName": target[
                    "targetName"
                ],
                "ticID": target["ticID"],
                "sector": dataset[
                    "source"
                ].get("sector"),
                "author": dataset[
                    "source"
                ].get("author"),
                "cadenceSeconds": dataset[
                    "source"
                ].get("cadenceSeconds"),
                "role": dataset[
                    "science"
                ].get("role"),
            }
        )

    total_work_units = (
            len(datasets)
            * expected_work_unit_count()
    )

    manifest = {
        "id": PROJECT_ID,
        "name": PROJECT_NAME,
        "workloadID": WORKLOAD_ID,
        "datasets": datasets,
    }

    output_path = (
            PROJECTS_DIR
            / f"{PROJECT_ID}.json"
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
    print(f"   file: {output_path}")
    print(f"   datasets: {len(datasets)}")
    print(
        "   total work units: "
        f"{total_work_units}"
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

    for target in TARGETS:
        validate_science_metadata(target)

    print()
    print("⭐ OpenStar TESS Preprocessor")
    print(f"Project: {PROJECT_ID}")
    print(f"Targets: {len(TARGETS)}")
    print(
        "Frequency range: "
        f"{MINIMUM_FREQUENCY:.3f} - "
        f"{MAXIMUM_FREQUENCY:.3f} cycles/day"
    )
    print(
        "Frequencies per target: "
        f"{TOTAL_FREQUENCIES}"
    )
    print(
        "Work units per target: "
        f"{expected_work_unit_count()}"
    )

    preflight_targets()

    prepared_targets = []

    for target in TARGETS:
        dataset, output_path = prepare_target(
            target
        )

        prepared_targets.append(
            (
                target,
                dataset,
                output_path,
            )
        )

    manifest_path = write_project_manifest(
        prepared_targets
    )

    print()
    print("✅ OpenStar TESS validation project ready")
    print(f"   project: {PROJECT_ID}")
    print(f"   manifest: {manifest_path}")
    print(
        "   datasets prepared: "
        f"{len(prepared_targets)}"
    )
    print(
        "   total work units: "
        f"{len(prepared_targets) * expected_work_unit_count()}"
    )


if __name__ == "__main__":
    main()