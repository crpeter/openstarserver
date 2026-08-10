import json
import math
import re
from pathlib import Path

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle


# ============================================================
# OpenStar Blind A all-available-sector validation
# ============================================================

PROJECT_ID = "openstar.tess-blind-all-sectors-v1"
PROJECT_NAME = "OpenStar Blind A All Available TESS Sectors Validation v1"
WORKLOAD_ID = "openstar.tess-period-search.v1"

DATASET_ID = "tess-blind-a-all-sectors"
TARGET_NAME = "Blind A"
TARGET_QUERY = "TIC 25165839"
TIC_ID = 25165839

SCIENCE = {
    "role": "blind",
}

DATA_DIR = Path("data")
PROJECTS_DIR = DATA_DIR / "projects"

# Keep the validated client payload size. For this all-sector test the sample
# budget is distributed across sectors so every usable observing visit remains
# represented instead of allowing one dense sector to dominate the payload.
MAX_SAMPLES = 18_000

# Same Blind-A long-period search used by the successful earlier runs.
MINIMUM_FREQUENCY = 0.03
MAXIMUM_FREQUENCY = 5.0

TOTAL_FREQUENCIES = 4_194_304
FREQUENCIES_PER_WORK_UNIT = 4_096

PREFERRED_AUTHOR = "SPOC"
FALLBACK_AUTHOR = "TESS-SPOC"
PREFERRED_EXPTIME_SECONDS = 120

# Populated after discovery/preflight.
USABLE_SECTORS = []
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

    if role not in ("known", "control", "blind"):
        raise RuntimeError(f"Invalid science role: {role}")

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
# TESS sector discovery and product selection
# ============================================================


def _sector_from_text(value):
    if value is None:
        return None

    match = re.search(
        r"sector\s*0*(\d+)",
        str(value),
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(match.group(1))


def discover_tess_sectors() -> list[int]:
    print()
    print("🔭 Discovering available TESS sectors")
    print(f"   target: {TARGET_NAME}")
    print(f"   query: {TARGET_QUERY}")

    result = lk.search_lightcurve(
        TARGET_QUERY,
        mission="TESS",
    )

    if len(result) == 0:
        raise RuntimeError(
            f"No TESS light-curve products found for {TARGET_NAME}."
        )

    sectors = set()
    table = getattr(result, "table", None)

    if table is not None:
        colnames = set(getattr(table, "colnames", []))

        if "sequence_number" in colnames:
            for value in table["sequence_number"]:
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    continue

                if numeric > 0:
                    sectors.add(numeric)

        if "mission" in colnames:
            for value in table["mission"]:
                sector = _sector_from_text(value)
                if sector is not None:
                    sectors.add(sector)

    if not sectors:
        missions = getattr(result, "mission", [])
        for value in missions:
            sector = _sector_from_text(value)
            if sector is not None:
                sectors.add(sector)

    discovered = sorted(sectors)

    if not discovered:
        print()
        print("Available TESS light-curve products:")
        print(result)
        raise RuntimeError(
            "TESS products were found, but sector numbers could not be parsed."
        )

    print(
        "   discovered sectors: "
        + ", ".join(str(sector) for sector in discovered)
    )
    print(f"   discovered sector count: {len(discovered)}")

    return discovered


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

    selected_index = int(
        finite_indices[
            int(np.argmin(exposures[finite_indices]))
        ]
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


def search_light_curve(sector: int):
    key = _cache_key(sector)
    cached = SEARCH_SELECTION_CACHE.get(key)

    if cached is not None:
        return cached

    preferred = _search_products(
        sector,
        author=PREFERRED_AUTHOR,
        exptime=PREFERRED_EXPTIME_SECONDS,
    )

    if len(preferred) > 0:
        selected = _select_shortest_cadence(preferred)
        cadence_seconds = _selected_exptime_seconds(selected)
        selection = (
            selected,
            PREFERRED_AUTHOR,
            cadence_seconds,
        )
        SEARCH_SELECTION_CACHE[key] = selection
        return selection

    # A sector is still scientifically useful if SPOC exists at a different
    # cadence. Prefer the shortest available SPOC product before falling back
    # to the FFI-derived TESS-SPOC light curve.
    any_spoc = _search_products(
        sector,
        author=PREFERRED_AUTHOR,
        exptime=None,
    )

    if len(any_spoc) > 0:
        selected = _select_shortest_cadence(any_spoc)
        cadence_seconds = _selected_exptime_seconds(selected)
        selection = (
            selected,
            PREFERRED_AUTHOR,
            cadence_seconds,
        )
        SEARCH_SELECTION_CACHE[key] = selection
        return selection

    fallback = _search_products(
        sector,
        author=FALLBACK_AUTHOR,
        exptime=None,
    )

    if len(fallback) > 0:
        selected = _select_shortest_cadence(fallback)
        cadence_seconds = _selected_exptime_seconds(selected)
        selection = (
            selected,
            FALLBACK_AUTHOR,
            cadence_seconds,
        )
        SEARCH_SELECTION_CACHE[key] = selection
        return selection

    raise RuntimeError(
        "no SPOC or TESS-SPOC light curve found"
    )


def preflight_available_sectors(discovered_sectors: list[int]) -> list[int]:
    print()
    print("🔎 Preflighting discovered TESS sectors")
    print("   no downloads or Astropy calculations yet")

    usable = []
    skipped = []

    for sector in discovered_sectors:
        try:
            search_result, author, cadence_seconds = search_light_curve(sector)

            if len(search_result) == 0:
                raise RuntimeError("empty search result")

            usable.append(sector)
            print(
                f"   ✅ Sector {sector}: "
                f"{author}, {cadence_seconds:.0f}s"
            )
        except Exception as error:
            skipped.append((sector, str(error)))
            print(
                f"   ⏭️ Sector {sector}: skipped ({error})"
            )

    if not usable:
        raise RuntimeError(
            "No usable SPOC-family TESS sectors were found."
        )

    print()
    print("✅ TESS sector preflight complete")
    print(
        "   usable sectors: "
        + ", ".join(str(sector) for sector in usable)
    )
    print(f"   usable sector count: {len(usable)}")
    print(f"   skipped sector count: {len(skipped)}")

    return usable


# ============================================================
# Download and source extraction
# ============================================================


def _extract_sector(light_curve, requested_sector: int):
    sector = getattr(light_curve, "sector", None)

    if sector is None:
        meta = getattr(light_curve, "meta", {})
        sector = meta.get("SECTOR")

    if sector is None:
        sector = requested_sector

    try:
        return int(sector)
    except (TypeError, ValueError):
        return sector


def download_light_curve(sector: int):
    search_result, source_author, cadence_seconds = search_light_curve(sector)

    print()
    print("⬇️ Downloading selected light curve")
    print(f"   requested sector: {sector}")
    print(f"   author: {source_author}")
    print(f"   cadence: {cadence_seconds:.0f}s")

    light_curve = search_result.download(
        quality_bitmask="default"
    )

    if light_curve is None:
        raise RuntimeError(
            f"Download failed for {TARGET_NAME}, Sector {sector}."
        )

    actual_sector = _extract_sector(light_curve, sector)

    if actual_sector != sector:
        raise RuntimeError(
            "Downloaded unexpected TESS sector: "
            f"requested={sector}, actual={actual_sector}."
        )

    print(f"   selected sector: {actual_sector}")

    return (
        light_curve,
        {
            "author": source_author,
            "cadenceSeconds": cadence_seconds,
            "sector": actual_sector,
        },
    )


def extract_finite_sector_samples(light_curve, sector: int):
    original_samples = len(light_curve)

    times64 = np.asarray(
        light_curve.time.value,
        dtype=np.float64,
    )

    flux64 = np.asarray(
        light_curve.flux.value,
        dtype=np.float64,
    )

    finite = np.isfinite(times64) & np.isfinite(flux64)
    times64 = times64[finite]
    flux64 = flux64[finite]

    if len(times64) == 0:
        raise RuntimeError(
            f"Sector {sector} contains no finite samples."
        )

    order = np.argsort(times64)
    times64 = times64[order]
    flux64 = flux64[order]

    print()
    print(f"✨ Sector {sector} source prepared")
    print(f"   original samples: {original_samples}")
    print(f"   finite samples: {len(times64)}")
    print(f"   first TESS time: {float(times64[0]):.8f} days")
    print(f"   last TESS time: {float(times64[-1]):.8f} days")
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
# Balanced multi-sector preprocessing
# ============================================================


def balanced_sample_counts(lengths: list[int], budget: int) -> list[int]:
    total_available = sum(lengths)

    if total_available <= budget:
        return list(lengths)

    if budget < len(lengths):
        raise RuntimeError(
            "MAX_SAMPLES is too small to retain at least one sample per sector."
        )

    counts = [0] * len(lengths)
    remaining_indices = set(range(len(lengths)))
    remaining_budget = budget

    while remaining_indices:
        share = remaining_budget // len(remaining_indices)

        if share <= 0:
            raise RuntimeError("Balanced sample allocation failed.")

        saturated = [
            index
            for index in remaining_indices
            if lengths[index] <= share
        ]

        if saturated:
            for index in saturated:
                counts[index] = lengths[index]
                remaining_budget -= counts[index]
                remaining_indices.remove(index)
            continue

        ordered = sorted(remaining_indices)
        remainder = remaining_budget - share * len(ordered)

        for position, index in enumerate(ordered):
            counts[index] = share + (1 if position < remainder else 0)

        remaining_budget = 0
        remaining_indices.clear()

    if sum(counts) != budget:
        raise RuntimeError(
            "Balanced sample allocation did not consume the full sample budget."
        )

    return counts


def _evenly_select(values: np.ndarray, count: int) -> np.ndarray:
    if count >= len(values):
        return values.copy()

    indices = np.linspace(
        0,
        len(values) - 1,
        count,
        dtype=np.int64,
    )

    return values[indices]


def combine_and_prepare_samples(sector_samples: list[dict]):
    if not sector_samples:
        raise RuntimeError("No sector samples were supplied.")

    lengths = [
        len(item["times64"])
        for item in sector_samples
    ]

    selected_counts = balanced_sample_counts(
        lengths,
        MAX_SAMPLES,
    )

    selected_times = []
    selected_flux = []
    selected_labels = []
    normalization_metadata = {}

    for item, selected_count in zip(sector_samples, selected_counts):
        sector = int(item["sector"])
        times64 = _evenly_select(item["times64"], selected_count)
        flux64 = _evenly_select(item["flux64"], selected_count)

        flux_mean = float(np.mean(flux64))
        flux_stddev = float(np.std(flux64))

        if not math.isfinite(flux_stddev) or flux_stddev <= 0:
            raise RuntimeError(
                f"Sector {sector} has invalid flux standard deviation."
            )

        flux64 = (flux64 - flux_mean) / flux_stddev

        selected_times.append(times64)
        selected_flux.append(flux64)
        selected_labels.append(
            np.full(
                len(times64),
                sector,
                dtype=np.int16,
            )
        )

        normalization_metadata[str(sector)] = {
            "selectedSamples": int(len(times64)),
            "sourceFluxMean": flux_mean,
            "sourceFluxStddev": flux_stddev,
        }

    combined_times64 = np.concatenate(selected_times)
    combined_flux64 = np.concatenate(selected_flux)
    combined_sector_labels = np.concatenate(selected_labels)

    order = np.argsort(combined_times64)
    combined_times64 = combined_times64[order]
    combined_flux64 = combined_flux64[order]
    combined_sector_labels = combined_sector_labels[order]

    time_origin_days = float(combined_times64[0])
    relative_times64 = combined_times64 - time_origin_days

    # Validation boundary: these are the exact arrays consumed by Swift/Metal.
    times = np.asarray(relative_times64, dtype=np.float32)
    flux = np.asarray(combined_flux64, dtype=np.float32)

    if not np.all(np.isfinite(times)):
        raise RuntimeError("Float32 time conversion produced non-finite values.")

    if not np.all(np.isfinite(flux)):
        raise RuntimeError("Float32 flux conversion produced non-finite values.")

    if len(times) == 0:
        raise RuntimeError("Combined dataset is empty after preprocessing.")

    times[0] = np.float32(0.0)

    if len(times) > 1 and np.any(np.diff(times.astype(np.float64)) < 0):
        raise RuntimeError("Float32 combined time array is not sorted.")

    baseline_days = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    # Build real no-observation gaps between chronologically adjacent sectors.
    chronological = sorted(
        sector_samples,
        key=lambda item: float(item["times64"][0]),
    )

    gaps = []

    for previous, current in zip(chronological, chronological[1:]):
        previous_last = float(previous["times64"][-1])
        current_first = float(current["times64"][0])
        gap_days = current_first - previous_last

        if gap_days > 0:
            gaps.append(
                {
                    "fromSector": int(previous["sector"]),
                    "toSector": int(current["sector"]),
                    "gapDays": gap_days,
                }
            )

    print()
    print("✨ All-sector light curve prepared")
    print(
        "   finite source samples across sectors: "
        f"{sum(lengths)}"
    )
    print(f"   distributed samples: {len(times)}")
    print(f"   sectors represented: {len(sector_samples)}")

    for item in chronological:
        sector = int(item["sector"])
        count = normalization_metadata[str(sector)]["selectedSamples"]
        print(f"   Sector {sector} distributed samples: {count}")

    print(f"   common original time origin: {time_origin_days:.8f} days")
    print(f"   distributed time origin: {float(times[0]):.8f} days")
    print(f"   full all-sector baseline: {baseline_days:.4f} days")

    if gaps:
        largest_gap = max(gaps, key=lambda item: item["gapDays"])
        print(
            "   largest no-observation gap: "
            f"{largest_gap['gapDays']:.4f} days "
            f"(Sector {largest_gap['fromSector']} -> "
            f"Sector {largest_gap['toSector']})"
        )

    print("   distributed precision: Float32")
    print("   normalization: independent per sector in Float64")
    print("   sample allocation: balanced across usable sectors")
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
        gaps,
    )


# ============================================================
# Astropy reference from exact distributed Float32 samples
# ============================================================


def calculate_astropy_reference(times: np.ndarray, flux: np.ndarray):
    astropy_times = np.asarray(times, dtype=np.float32).astype(np.float64)
    astropy_flux = np.asarray(flux, dtype=np.float32).astype(np.float64)

    step = frequency_step()
    frequencies = (
        MINIMUM_FREQUENCY
        + np.arange(TOTAL_FREQUENCIES, dtype=np.float64) * step
    )

    print()
    print("🧪 Calculating Astropy reference")
    print(
        "   frequency range: "
        f"{MINIMUM_FREQUENCY:.3f} - "
        f"{MAXIMUM_FREQUENCY:.3f} cycles/day"
    )
    print(f"   frequencies: {len(frequencies)}")
    print("   input samples: exact distributed Float32 values")

    periodogram = LombScargle(astropy_times, astropy_flux)
    powers = periodogram.power(frequencies)

    if len(powers) != TOTAL_FREQUENCIES:
        raise RuntimeError("Astropy returned unexpected frequency count.")

    if not np.any(np.isfinite(powers)):
        raise RuntimeError("Astropy returned no finite Lomb-Scargle powers.")

    global_index = int(np.nanargmax(powers))
    best_frequency = float(frequencies[global_index])
    best_power = float(powers[global_index])
    best_period_days = 1.0 / best_frequency

    chunks = []

    for start_index in range(
        0,
        TOTAL_FREQUENCIES,
        FREQUENCIES_PER_WORK_UNIT,
    ):
        end_index = min(
            start_index + FREQUENCIES_PER_WORK_UNIT,
            TOTAL_FREQUENCIES,
        )

        chunk_powers = powers[start_index:end_index]

        if not np.any(np.isfinite(chunk_powers)):
            raise RuntimeError(
                "Astropy returned no finite power values for "
                f"frequency chunk starting at {start_index}."
            )

        local_index = int(np.nanargmax(chunk_powers))
        absolute_index = start_index + local_index
        chunk_frequency = float(frequencies[absolute_index])
        chunk_power = float(powers[absolute_index])

        chunks.append(
            {
                "frequencyStartIndex": start_index,
                "frequencyCount": end_index - start_index,
                "bestFrequency": chunk_frequency,
                "bestPeriodDays": 1.0 / chunk_frequency,
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
    print(f"   frequency: {best_frequency:.8f} cycles/day")
    print(f"   period: {best_period_days:.8f} days")
    print(f"   power: {best_power:.8f}")
    print(f"   work-unit references: {len(chunks)}/{expected_chunks}")

    if rayleigh_resolution is not None:
        print(
            "   all-sector Rayleigh resolution: "
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


def _combined_author(source_metadata: list[dict]) -> str:
    authors = []

    for item in source_metadata:
        author = str(item["author"])
        if author not in authors:
            authors.append(author)

    return "+".join(authors)


def _combined_cadence_seconds(source_metadata: list[dict]) -> float:
    return min(float(item["cadenceSeconds"]) for item in source_metadata)


def build_dataset(
    times: np.ndarray,
    flux: np.ndarray,
    reference: dict,
    source_metadata: list[dict],
    sector_samples: list[dict],
    time_origin_days: float,
    normalization_metadata: dict,
    gaps: list[dict],
):
    science = validate_science_metadata()
    sectors = [int(item["sector"]) for item in sector_samples]

    products = []

    for metadata, samples in zip(source_metadata, sector_samples):
        sector_key = str(metadata["sector"])

        products.append(
            {
                "sector": metadata["sector"],
                "author": metadata["author"],
                "cadenceSeconds": metadata["cadenceSeconds"],
                "originalSamples": samples["originalSamples"],
                "finiteSamples": samples["finiteSamples"],
                "distributedSamples": normalization_metadata[
                    sector_key
                ]["selectedSamples"],
            }
        )

    largest_gap_days = (
        max(item["gapDays"] for item in gaps)
        if gaps
        else 0.0
    )

    return {
        "id": DATASET_ID,
        "targetName": TARGET_NAME,
        "mission": "TESS",
        "source": {
            "archive": "MAST",
            "author": _combined_author(source_metadata),
            "ticID": TIC_ID,
            # Legacy scalar compatibility field for existing clients.
            "sector": int(sectors[0]),
            "sectors": sectors,
            "sectorLabel": "+".join(str(sector) for sector in sectors),
            "cadenceSeconds": _combined_cadence_seconds(source_metadata),
            "originalTimeOriginDays": time_origin_days,
            "interSectorGapDays": largest_gap_days,
            "interSectorGaps": gaps,
            "products": products,
        },
        "science": science,
        "timeUnit": "days",
        "timeReference": "relative-to-first-distributed-sample",
        "numericRepresentation": "Float32",
        "fluxUnit": "normalized",
        "fluxNormalization": "independent-per-sector",
        "sampleAllocation": "balanced-across-sectors",
        "times": [float(value) for value in times],
        "flux": [float(value) for value in flux],
        "frequencySearch": {
            "minimumFrequency": MINIMUM_FREQUENCY,
            "maximumFrequency": MAXIMUM_FREQUENCY,
            "frequencyStep": frequency_step(),
            "totalFrequencies": TOTAL_FREQUENCIES,
            "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
        },
        "reference": reference,
    }


def write_dataset(dataset: dict):
    output_path = DATA_DIR / f"{DATASET_ID}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, allow_nan=False)

    reference_count = len(dataset["reference"]["chunks"])

    print()
    print("💾 Dataset saved")
    print(f"   file: {output_path}")
    print(f"   work units: {expected_work_unit_count()}")
    print(
        "   Astropy references: "
        f"{reference_count}/{expected_work_unit_count()}"
    )

    return output_path


def write_project_manifest(dataset: dict, output_path: Path):
    sectors = list(dataset["source"]["sectors"])

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
                "sector": int(sectors[0]),
                "sectors": sectors,
                "author": dataset["source"].get("author"),
                "cadenceSeconds": dataset["source"].get("cadenceSeconds"),
                "role": dataset["science"].get("role"),
            }
        ],
    }

    manifest_path = PROJECTS_DIR / f"{PROJECT_ID}.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)

    print()
    print("📦 Project manifest saved")
    print(f"   file: {manifest_path}")
    print("   datasets: 1")
    print(f"   total work units: {expected_work_unit_count()}")

    return manifest_path


# ============================================================
# Main
# ============================================================


def main():
    global USABLE_SECTORS

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    validate_science_metadata()

    print()
    print("⭐ OpenStar Blind A All Available TESS Sectors Preprocessor")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Target: {TARGET_NAME}")
    print(
        "Frequency range: "
        f"{MINIMUM_FREQUENCY:.3f} - "
        f"{MAXIMUM_FREQUENCY:.3f} cycles/day"
    )
    print(f"Frequencies per target: {TOTAL_FREQUENCIES}")
    print(f"Work units per target: {expected_work_unit_count()}")
    print(f"Maximum distributed samples: {MAX_SAMPLES}")
    print("Distributed numeric representation: Float32")
    print("Time origin: one common origin after all-sector merge")
    print("Inter-sector time gaps: preserved")
    print("Flux normalization: independent per sector in Float64")
    print("Sample allocation: balanced across usable sectors")
    print("Astropy reference input: exact distributed Float32 samples")
    print("Science role: blind")

    discovered_sectors = discover_tess_sectors()
    USABLE_SECTORS = preflight_available_sectors(discovered_sectors)

    sector_samples = []
    source_metadata = []

    for sector in USABLE_SECTORS:
        light_curve, metadata = download_light_curve(sector)
        samples = extract_finite_sector_samples(light_curve, sector)
        sector_samples.append(samples)
        source_metadata.append(metadata)

    (
        times,
        flux,
        time_origin_days,
        normalization_metadata,
        gaps,
    ) = combine_and_prepare_samples(sector_samples)

    reference = calculate_astropy_reference(times, flux)

    dataset = build_dataset(
        times,
        flux,
        reference,
        source_metadata,
        sector_samples,
        time_origin_days,
        normalization_metadata,
        gaps,
    )

    dataset_path = write_dataset(dataset)
    manifest_path = write_project_manifest(dataset, dataset_path)

    print()
    print("✅ OpenStar all-sector blind project ready")
    print(f"   project: {PROJECT_ID}")
    print(f"   dataset: {DATASET_ID}")
    print(f"   manifest: {manifest_path}")
    print(
        "   sectors: "
        + ", ".join(str(sector) for sector in USABLE_SECTORS)
    )
    print(f"   sector count: {len(USABLE_SECTORS)}")
    print(f"   distributed samples: {len(times)}")
    print(f"   total work units: {expected_work_unit_count()}")
    print("   external period remains absent from project metadata")


if __name__ == "__main__":
    main()
