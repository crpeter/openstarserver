"""Reusable server-side preparation for generic TESS time-series workloads."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterable

MAX_SAMPLES = 18_000
MINIMUM_FREQUENCY = 0.1
MAXIMUM_FREQUENCY = 5.0
TOTAL_FREQUENCIES = 4_194_304
FREQUENCIES_PER_WORK_UNIT = 4_096


@dataclass(frozen=True)
class PreparedLightCurve:
    coordinates: tuple[float, ...]
    values: tuple[float, ...]
    source_sample_count: int
    finite_sample_count: int
    sample_count: int
    time_origin_days: float
    baseline_days: float


def broad_tess_frequency_search() -> dict[str, int | float]:
    """The validated broad profile, kept independent of archive admission."""
    return {
        "minimumFrequency": MINIMUM_FREQUENCY,
        "maximumFrequency": MAXIMUM_FREQUENCY,
        "frequencyStep": (MAXIMUM_FREQUENCY - MINIMUM_FREQUENCY) / TOTAL_FREQUENCIES,
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def prepare_tess_samples(
    times: Iterable[float], fluxes: Iterable[float], *, max_samples: int = MAX_SAMPLES
) -> PreparedLightCurve:
    """Clean in Float64, normalize and shift, then quantize for workers."""
    time = [float(value) for value in times]
    flux = [float(value) for value in fluxes]
    if len(time) != len(flux):
        raise RuntimeError("Time/flux sample count mismatch.")
    source_count = len(time)
    pairs = [(x, y) for x, y in zip(time, flux) if math.isfinite(x) and math.isfinite(y)]
    time, flux = [x for x, _ in pairs], [y for _, y in pairs]
    if not len(time):
        raise RuntimeError("Light curve contains no finite samples.")
    pairs = sorted(zip(time, flux), key=lambda item: item[0])
    time, flux = [x for x, _ in pairs], [y for _, y in pairs]
    finite_count = len(time)
    if max_samples < 2:
        raise ValueError("max_samples must be at least two.")
    if len(time) > max_samples:
        indices = [int(position * (len(time) - 1) / (max_samples - 1)) for position in range(max_samples)]
        time, flux = [time[i] for i in indices], [flux[i] for i in indices]
    mean = sum(flux) / len(flux)
    deviation = math.sqrt(sum((value - mean) ** 2 for value in flux) / len(flux))
    if not math.isfinite(deviation) or deviation <= 0:
        raise RuntimeError("Light curve flux has invalid standard deviation.")
    flux = [(value - mean) / deviation for value in flux]
    origin = float(time[0])
    quantize = lambda value: struct.unpack("!f", struct.pack("!f", value))[0]
    time32 = [quantize(value - origin) for value in time]
    flux32 = [quantize(value) for value in flux]
    if not all(map(math.isfinite, time32)) or not all(map(math.isfinite, flux32)):
        raise RuntimeError("Float32 conversion produced non-finite values.")
    time32[0] = 0.0
    return PreparedLightCurve(
        tuple(time32), tuple(flux32),
        source_count, finite_count, len(time32), origin,
        time32[-1] - time32[0] if len(time32) > 1 else 0.0,
    )


def read_and_prepare_tess_light_curve(path, *, max_samples: int = MAX_SAMPLES):
    """Read a downloaded product using the established Lightkurve quality policy."""
    try:
        import lightkurve as lk
    except ImportError as error:  # pragma: no cover - installation boundary
        raise RuntimeError("lightkurve is required to materialize TESS products") from error
    curve = lk.read(path, quality_bitmask="default")
    return prepare_tess_samples(curve.time.value, curve.flux.value, max_samples=max_samples)
