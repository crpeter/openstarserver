import json
import math
from pathlib import Path

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind A combined Sector 1 + Sector 28 validation
# ============================================================

PROJECT_ID = "openstar.tess-blind-combined-s1-s28-v1"
PROJECT_NAME = "OpenStar Blind A Combined S1+S28 Validation v1"
WORKLOAD_ID = "openstar.tess-period-search.v1"

DATASET_ID = "tess-blind-a-s1-s28"
TARGET_NAME = "Blind A"
TARGET_QUERY = "TIC 25165839"
TIC_ID = 25165839
SECTORS = (1, 28)

SCIENCE = {
    "role": "blind",
}

DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"

# Keep the same distributed payload cap used by the validated tests.
MAX_SAMPLES = 18_000

# Keep the same wider range used by the successful Blind-A long-period test.
MINIMUM_FREQUENCY = 0.03
MAXIMUM_FREQUENCY = 5.0

TOTAL_FREQUENCIES = 4_194_304
FREQUENCIES_PER_WORK_UNIT = 4_096

PREFERRED_AUTHOR = "SPOC"
FALLBACK_AUTHOR = "TESS-SPOC"
PREFERRED_EXPTIME_SECONDS = 120

# Preflight selections are cached so the exact product selected during
# preflight is the product downloaded during preparation.
SEARCH_SELECTION_CACHE = {}


# ============================================================
# Frequency grid
# ============================================================


def frequency_step() -> float:
    return (
        MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY
    ) / TOTAL_FREQUENCIES


def expected_work_unit_count() -> int:
    return math.ceil(
        TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT
    )


# ============================================================
# Blind metadata validation
# ============================================================


def validate_science_metadata() -> dict:
    science = dict(SCIENCE)

    role = science.get("role")

    if role not in (
        "known",
        "control",
        "blind",
    ):
        raise RuntimeError(
            f"Invalid science role: {role}"
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


# ============================================================
# TESS product selection
# ============================================================


def _cache_key(sector: int) -> str:
    return f"{DATASET_ID}-sector-{sector}"


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


def _diagnostic_search(sector: int):
    return lk.search_lightcurve(
        TARGET_QUERY,
        mission="TESS",
        sector=int(sector),
    )


def search_light_curve(sector: int):
    key = _cache_key(sector)
    cached = SEARCH_SELECTION_CACHE.get(key)

    if cached is not None:
        print()
        print("🔭 Using preflight MAST selection")
        print(f"   target: {TARGET_NAME}")
        print(f"   sector: {sector}")
        print(f"   author: {cached[1]}")
        print(f"   cadence: {cached[2]:.0f}s")
        return cached

    print()
    print("🔭 Searching MAST")
    print(f"   target: {TARGET_NAME}")
    print(f"   query: {TARGET_QUERY}")
    print(f"   requested sector: {sector}")
    print(f"   preferred author: {PREFERRED_AUTHOR}")
    print(
        "   preferred cadence: "
        f"{PREFERRED_EXPTIME_SECONDS}s"
    )

    preferred = _search_products(
        sector,
        author=PREFERRED_AUTHOR,
        exptime=PREFERRED_EXPTIME_SECONDS,
    )

    if len(preferred) > 0:
        print(
            "   preferred products found: "
            f"{len(preferred)}"
        )

        selected = _select_shortest_cadence(
            preferred
        )

        cadence_seconds = (
            _selected_exptime_seconds(selected)
        )

        print(
            "   selected author: "
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

        SEARCH_SELECTION_CACHE[key] = selection
        return selection

    print(
        "   no 120s SPOC product; "
        "trying TESS-SPOC FFI light curves"
    )

    fallback = _search_products(
        sector,
        author=FALLBACK_AUTHOR,
        exptime=None,
    )

    if len(fallback) > 0:
        print(
            "   fallback products found: "
            f"{len(fallback)}"
        )

        selected = _select_shortest_cadence(
            fallback
        )

        cadence_seconds = (
            _selected_exptime_seconds(selected)
        )

        print(
            "   selected author: "
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

        SEARCH_SELECTION_CACHE[key] = selection
        return selection

    available = _diagnostic_search(sector)

    print()
    print("❌ No usable SPOC-family product")
    print(f"   target: {TARGET_NAME}")
    print(f"   sector: {sector}")

    if len(available) > 0:
        print()
        print("Available TESS light-curve products:")
        print(available)

    raise RuntimeError(
        "No SPOC or TESS-SPOC TESS light curve "
        f"found for {TARGET_NAME}, Sector {sector}."
    )


def preflight_sectors():
    print()
    print("🔎 Preflighting both TESS sectors")
    print("   no downloads or Astropy calculations yet")

    failures = []

    for sector in SECTORS:
        try:
            search_result, author, cadence_seconds = (
                search_light_curve(sector)
            )

            if len(search_result) == 0:
                raise RuntimeError("empty search result")

            print(
                "   preflight ready: "
                f"Sector {sector} -> "
                f"{author}, {cadence_seconds:.0f}s"
            )
        except Exception as error:
            failures.append(
                f"Sector {sector}: {error}"
            )

    if failures:
        raise RuntimeError(
            "TESS sector preflight failed. "
            "No dataset was processed:\n - "
            + "\n - ".join(failures)
        )

    print()
    print("✅ TESS sector preflight complete")
    print(f"   usable sectors: {len(SECTORS)}/{len(SECTORS)}")


# ============================================================
# Download and source extraction
# ============================================================


def _extract_sector(
    light_curve,
    requested_sector: int,
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

        sector = meta.get("SECTOR")

    if sector is None:
        sector = requested_sector

    try:
        return int(sector)
    except (TypeError, ValueError):
        return sector


def download_light_curve(sector: int):
    (
        search_result,
        source_author,
        cadence_seconds,
    ) = search_light_curve(sector)

    print()
    print("⬇️ Downloading selected light curve")
    print(f"   requested sector: {sector}")
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
            f"{TARGET_NAME}, Sector {sector}."
        )

    actual_sector = _extract_sector(
        light_curve,
        sector,
    )

    if actual_sector != sector:
        raise RuntimeError(
            "Downloaded unexpected TESS sector: "
            f"requested={sector}, actual={actual_sector}."
        )

    print(
        "   selected sector: "
        f"{actual_sector}"
    )

    return (
        light_curve,
        {
            "author": source_author,
            "cadenceSeconds": cadence_seconds,
            "sector": actual_sector,
        },
    )


def extract_finite_sector_samples(
    light_curve,
    sector: int,
):
    original_samples = len(light_curve)

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

    if len(times64) == 0:
        raise RuntimeError(
            f"Sector {sector} contains no finite samples."
        )

    order = np.argsort(times64)

    times64 = times64[order]
    flux64 = flux64[order]

    if len(times64) > 1 and np.any(
        np.diff(times64) < 0
    ):
        raise RuntimeError(
            f"Sector {sector} times are not sorted."
        )

    print()
    print(f"✨ Sector {sector} source prepared")
    print(
        "   original samples: "
        f"{original_samples}"
    )
    print(
        "   finite samples: "
        f"{len(times64)}"
    )
    print(
        "   first TESS time: "
        f"{float(times64[0]):.8f} days"
    )
    print(
        "   last TESS time: "
        f"{float(times64[-1]):.8f} days"
    )
    print(
        "   sector baseline: "
        f"{float(times64[-1] - times64[0]):.4f} days"
    )

    return {
        "sector": int(sector),
        "times64": times64,
        "flux64": flux64,
        "originalSamples": int(original_samples),
        "finiteSamples": int(len(times64)),
    }


# ============================================================
# Combined preprocessing
# ============================================================


def combine_and_prepare_samples(
    sector_samples: list[dict],
):
    if len(sector_samples) != len(SECTORS):
        raise RuntimeError(
            "Combined preparation did not receive every requested sector."
        )

    combined_times64 = np.concatenate(
        [item["times64"] for item in sector_samples]
    )

    combined_flux64 = np.concatenate(
        [item["flux64"] for item in sector_samples]
    )

    combined_sector_labels = np.concatenate(
        [
            np.full(
                len(item["times64"]),
                int(item["sector"]),
                dtype=np.int16,
            )
            for item in sector_samples
        ]
    )

    if not (
        len(combined_times64)
        == len(combined_flux64)
        == len(combined_sector_labels)
    ):
        raise RuntimeError(
            "Combined time/flux/sector sample count mismatch."
        )

    order = np.argsort(combined_times64)

    combined_times64 = combined_times64[order]
    combined_flux64 = combined_flux64[order]
    combined_sector_labels = combined_sector_labels[order]

    combined_finite_samples = len(combined_times64)

    # Keep the exact same overall payload cap as the successful validation
    # tests. Because there are no synthetic samples in the inter-sector gap,
    # sampling by observation index simply retains real observations from both
    # sectors in proportion to the available source samples.
    if len(combined_times64) > MAX_SAMPLES:
        indices = np.linspace(
            0,
            len(combined_times64) - 1,
            MAX_SAMPLES,
            dtype=np.int64,
        )

        combined_times64 = combined_times64[indices]
        combined_flux64 = combined_flux64[indices]
        combined_sector_labels = combined_sector_labels[indices]

    # Match the existing preprocessing order: downsample first, normalize in
    # Float64 second. For a multi-sector dataset, normalize each observing
    # sector independently so a different flux zero-point in one sector does
    # not create an artificial step across the long gap.
    normalization_metadata = {}

    for sector in SECTORS:
        mask = combined_sector_labels == sector

        if not np.any(mask):
            raise RuntimeError(
                f"No distributed samples remain for Sector {sector}."
            )

        sector_flux = combined_flux64[mask]

        flux_mean = float(
            np.mean(sector_flux)
        )

        flux_stddev = float(
            np.std(sector_flux)
        )

        if (
            not math.isfinite(flux_stddev)
            or flux_stddev <= 0
        ):
            raise RuntimeError(
                "Sector flux has invalid standard deviation: "
                f"Sector {sector}."
            )

        combined_flux64[mask] = (
            sector_flux - flux_mean
        ) / flux_stddev

        normalization_metadata[str(sector)] = {
            "selectedSamples": int(np.count_nonzero(mask)),
            "sourceFluxMean": flux_mean,
            "sourceFluxStddev": flux_stddev,
        }

    # This is the important combined-sector time rule: subtract ONE common
    # origin only after the two sectors have been merged on their real TESS
    # timestamps. The Sector 1 -> Sector 28 gap therefore remains in the
    # distributed time array.
    time_origin_days = float(
        combined_times64[0]
    )

    relative_times64 = (
        combined_times64 - time_origin_days
    )

    # THIS IS THE VALIDATION BOUNDARY:
    # Quantize the complete combined dataset to the exact numeric
    # representation consumed by Swift/Metal BEFORE Astropy is calculated.
    times = np.asarray(
        relative_times64,
        dtype=np.float32,
    )

    flux = np.asarray(
        combined_flux64,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(times)):
        raise RuntimeError(
            "Float32 time conversion produced non-finite values."
        )

    if not np.all(np.isfinite(flux)):
        raise RuntimeError(
            "Float32 flux conversion produced non-finite values."
        )

    if len(times) == 0:
        raise RuntimeError(
            "Combined dataset is empty after preprocessing."
        )

    times[0] = np.float32(0.0)

    if len(times) > 1 and np.any(
        np.diff(times.astype(np.float64)) < 0
    ):
        raise RuntimeError(
            "Float32 combined time array is not sorted."
        )

    baseline_days = (
        float(times[-1] - times[0])
        if len(times) > 1
        else 0.0
    )

    # Measure the actual no-observation gap using the selected, absolute
    # Float64 timestamps before the common-origin conversion.
    first_sector = SECTORS[0]
    second_sector = SECTORS[1]

    first_mask = combined_sector_labels == first_sector
    second_mask = combined_sector_labels == second_sector

    first_sector_last_time = float(
        np.max(combined_times64[first_mask])
    )
    second_sector_first_time = float(
        np.min(combined_times64[second_mask])
    )

    inter_sector_gap_days = (
        second_sector_first_time - first_sector_last_time
    )

    if inter_sector_gap_days <= 0:
        raise RuntimeError(
            "Expected Sector 28 observations to occur after Sector 1."
        )

    print()
    print("✨ Combined light curve prepared")
    print(
        "   finite source samples across sectors: "
        f"{combined_finite_samples}"
    )
    print(
        "   distributed samples: "
        f"{len(times)}"
    )

    for sector in SECTORS:
        count = normalization_metadata[
            str(sector)
        ]["selectedSamples"]
        print(
            f"   Sector {sector} distributed samples: "
            f"{count}"
        )

    print(
        "   common original time origin: "
        f"{time_origin_days:.8f} days"
    )
    print(
        "   distributed time origin: "
        f"{float(times[0]):.8f} days"
    )
    print(
        "   full Sector 1 -> Sector 28 baseline: "
        f"{baseline_days:.4f} days"
    )
    print(
        "   real no-observation gap between sectors: "
        f"{inter_sector_gap_days:.4f} days"
    )
    print("   distributed precision: Float32")
    print("   normalization: independent per sector in Float64")
    print(
        "   combined flux mean after Float32: "
        f"{float(np.mean(flux, dtype=np.float64)):.8f}"
    )
    print(
        "   combined flux stddev after Float32: "
        f"{float(np.std(flux, dtype=np.float64)):.8f}"
    )

    return (
        times,
        flux,
        time_origin_days,
        normalization_metadata,
        inter_sector_gap_days,
    )


# ============================================================
# Astropy reference from exact distributed Float32 samples
# ============================================================


def calculate_astropy_reference(
    times: np.ndarray,
    flux: np.ndarray,
):
    # Convert the exact quantized Float32 values back to Float64 only for
    # Astropy's arithmetic. Both implementations therefore begin with the
    # same distributed samples.
    astropy_times = np.asarray(
        times,
        dtype=np.float32,
    ).astype(np.float64)

    astropy_flux = np.asarray(
        flux,
        dtype=np.float32,
    ).astype(np.float64)

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
        "   frequencies: "
        f"{len(frequencies)}"
    )
    print("   input samples: exact distributed Float32 values")

    periodogram = LombScargle(
        astropy_times,
        astropy_flux,
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

    baseline_days = (
        float(astropy_times[-1] - astropy_times[0])
        if len(astropy_times) > 1
        else 0.0
    )

    rayleigh_resolution = (
        1.0 / baseline_days
        if baseline_days > 0
        else None
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

    if rayleigh_resolution is not None:
        print(
            "   combined Rayleigh resolution: "
            f"{rayleigh_resolution:.8f} cycles/day"
        )

    return {
        "bestFrequency": best_frequency,
        "bestPeriodDays": best_period_days,
        "bestPower": best_power,
        "chunks": chunks,
    }


# ============================================================
# Dataset/project output
# ============================================================


def _combined_author(
    source_metadata: list[dict],
) -> str:
    authors = []

    for item in source_metadata:
        author = str(item["author"])
        if author not in authors:
            authors.append(author)

    return "+".join(authors)


def _combined_cadence_seconds(
    source_metadata: list[dict],
) -> float:
    cadences = [
        float(item["cadenceSeconds"])
        for item in source_metadata
    ]

    if all(
        math.isclose(
            cadence,
            cadences[0],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for cadence in cadences[1:]
    ):
        return cadences[0]

    # The legacy scalar field remains numeric for compatibility. Per-sector
    # cadence values are retained in source.products below.
    return min(cadences)


def build_dataset(
    times: np.ndarray,
    flux: np.ndarray,
    reference: dict,
    source_metadata: list[dict],
    sector_samples: list[dict],
    time_origin_days: float,
    normalization_metadata: dict,
    inter_sector_gap_days: float,
):
    science = validate_science_metadata()

    if len(times) != len(flux):
        raise RuntimeError(
            "Time/flux sample count mismatch."
        )

    if len(times) == 0:
        raise RuntimeError(
            "Cannot build an empty dataset."
        )

    products = []

    for metadata, samples in zip(
        source_metadata,
        sector_samples,
    ):
        sector_key = str(metadata["sector"])

        products.append(
            {
                "sector": metadata["sector"],
                "author": metadata["author"],
                "cadenceSeconds": metadata[
                    "cadenceSeconds"
                ],
                "originalSamples": samples[
                    "originalSamples"
                ],
                "finiteSamples": samples[
                    "finiteSamples"
                ],
                "distributedSamples": (
                    normalization_metadata[
                        sector_key
                    ]["selectedSamples"]
                ),
            }
        )

    return {
        "id": DATASET_ID,
        "targetName": TARGET_NAME,
        "mission": "TESS",
        "source": {
            "archive": "MAST",
            "author": _combined_author(
                source_metadata
            ),
            "ticID": TIC_ID,
            # Keep the legacy scalar sector field numeric for existing clients.
            # The authoritative combined-sector metadata is `sectors` below.
            "sector": int(SECTORS[0]),
            "sectors": [
                int(sector)
                for sector in SECTORS
            ],
            "sectorLabel": "+".join(
                str(sector)
                for sector in SECTORS
            ),
            "cadenceSeconds": (
                _combined_cadence_seconds(
                    source_metadata
                )
            ),
            "originalTimeOriginDays": time_origin_days,
            "interSectorGapDays": inter_sector_gap_days,
            "products": products,
        },
        "science": science,
        "timeUnit": "days",
        "timeReference": "relative-to-first-distributed-sample",
        "numericRepresentation": "Float32",
        "fluxUnit": "normalized",
        "fluxNormalization": "independent-per-sector",
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


def write_dataset(dataset: dict):
    output_path = (
        DATA_DIR
        / f"{DATASET_ID}.json"
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


def write_project_manifest(
    dataset: dict,
    output_path: Path,
):
    manifest = {
        "id": PROJECT_ID,
        "name": PROJECT_NAME,
        "workloadID": WORKLOAD_ID,
        "datasets": [
            {
                "id": DATASET_ID,
                "path": str(output_path),
                "targetName": TARGET_NAME,
                "ticID": TIC_ID,
                # Legacy scalar compatibility field.
                "sector": int(SECTORS[0]),
                "sectors": [
                    int(sector)
                    for sector in SECTORS
                ],
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
        ],
    }

    manifest_path = (
        PROJECTS_DIR
        / f"{PROJECT_ID}.json"
    )

    with manifest_path.open(
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
    print(f"   file: {manifest_path}")
    print("   datasets: 1")
    print(
        "   total work units: "
        f"{expected_work_unit_count()}"
    )

    return manifest_path


# ============================================================
# Main
# ============================================================


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROJECTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_science_metadata()

    print()
    print("⭐ OpenStar Blind A Combined S1+S28 Preprocessor")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Target: {TARGET_NAME}")
    print(
        "Sectors: "
        + ", ".join(
            str(sector)
            for sector in SECTORS
        )
    )
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
    print(
        "Maximum distributed samples: "
        f"{MAX_SAMPLES}"
    )
    print("Distributed numeric representation: Float32")
    print("Time origin: one common origin after sector merge")
    print("Inter-sector time gap: preserved")
    print("Flux normalization: independent per sector in Float64")
    print("Astropy reference input: exact distributed Float32 samples")
    print("Science role: blind")

    preflight_sectors()

    sector_samples = []
    source_metadata = []

    for sector in SECTORS:
        (
            light_curve,
            metadata,
        ) = download_light_curve(sector)

        samples = extract_finite_sector_samples(
            light_curve,
            sector,
        )

        sector_samples.append(samples)
        source_metadata.append(metadata)

    (
        times,
        flux,
        time_origin_days,
        normalization_metadata,
        inter_sector_gap_days,
    ) = combine_and_prepare_samples(
        sector_samples
    )

    reference = calculate_astropy_reference(
        times,
        flux,
    )

    dataset = build_dataset(
        times,
        flux,
        reference,
        source_metadata,
        sector_samples,
        time_origin_days,
        normalization_metadata,
        inter_sector_gap_days,
    )

    dataset_path = write_dataset(
        dataset
    )

    manifest_path = write_project_manifest(
        dataset,
        dataset_path,
    )

    print()
    print("✅ OpenStar combined blind project ready")
    print(f"   project: {PROJECT_ID}")
    print(f"   dataset: {DATASET_ID}")
    print(f"   manifest: {manifest_path}")
    print(
        "   sectors: "
        + ", ".join(
            str(sector)
            for sector in SECTORS
        )
    )
    print(
        "   distributed samples: "
        f"{len(times)}"
    )
    print(
        "   total work units: "
        f"{expected_work_unit_count()}"
    )
    print("   external period remains absent from project metadata")


if __name__ == "__main__":
    main()
