#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk
from astropy.timeseries import LombScargle

TIC = 404927661
TARGET = f"TIC {TIC}"

# TESS EB catalog ephemeris for TIC 404927661.
ORBITAL_PERIOD_D = 18.536282847200237
T0_BTJD = 1333.9773604614618
SECONDARY_PHASE = 0.235

# Blind-F / Astropy frequency from the exact OpenStar distributed Float32 input.
OPENSTAR_SIGNAL_F = 0.21384606
OPENSTAR_SIGNAL_P = 1.0 / OPENSTAR_SIGNAL_F

# Exact fourth harmonic of the known EB orbital frequency.
FOURTH_ORBITAL_HARMONIC_F = 4.0 / ORBITAL_PERIOD_D
FOURTH_ORBITAL_HARMONIC_P = 1.0 / FOURTH_ORBITAL_HARMONIC_F

# Catalog eclipse widths are ~0.015 and ~0.017 in phase.  +/- 0.03 is
# intentionally generous so the first pass removes the eclipse bodies/wings.
ECLIPSE_MASK_HALF_WIDTH_PHASE = 0.03

MIN_FREQUENCY = 0.10
MAX_FREQUENCY = 5.00
SAMPLES_PER_PEAK = 50
LOCAL_F_MIN = 0.17
LOCAL_F_MAX = 0.26

DEFAULT_SECTORS = [1, 27, 35, 61, 87]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the ~4.67-day Blind-F signal survives after masking "
            "the known 18.5363-day eclipses of TIC 404927661."
        )
    )
    parser.add_argument(
        "--sectors",
        default=",".join(str(s) for s in DEFAULT_SECTORS),
        help=(
            "Comma-separated SPOC 120-s sectors to test, or 'all'. "
            f"Default: {','.join(str(s) for s in DEFAULT_SECTORS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="blind_f_decisive_v1",
        help="Directory for plots and CSV output.",
    )
    return parser.parse_args()


def circular_phase_distance(phase, center):
    return np.abs(((phase - center + 0.5) % 1.0) - 0.5)


def orbital_phase(time_btjd):
    return ((time_btjd - T0_BTJD) / ORBITAL_PERIOD_D) % 1.0


def eclipse_keep_mask(time_btjd):
    phase = orbital_phase(time_btjd)
    primary_distance = circular_phase_distance(phase, 0.0)
    secondary_distance = circular_phase_distance(phase, SECONDARY_PHASE)

    in_primary = primary_distance <= ECLIPSE_MASK_HALF_WIDTH_PHASE
    in_secondary = secondary_distance <= ECLIPSE_MASK_HALF_WIDTH_PHASE
    return ~(in_primary | in_secondary), in_primary, in_secondary


def sector_from_mission(value):
    match = re.search(r"Sector\s*0*(\d+)", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def available_sector_indices(search_result):
    by_sector = {}
    for index, mission in enumerate(search_result.mission):
        sector = sector_from_mission(mission)
        if sector is not None:
            by_sector.setdefault(sector, []).append(index)
    return by_sector


def clean_arrays(light_curve):
    time = np.asarray(light_curve.time.value, dtype=np.float64)
    flux = np.asarray(light_curve.flux.value, dtype=np.float64)

    if getattr(light_curve, "flux_err", None) is not None:
        flux_err = np.asarray(light_curve.flux_err.value, dtype=np.float64)
    else:
        flux_err = np.full_like(flux, np.nan)

    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]
    flux_err = flux_err[finite]

    median_flux = np.nanmedian(flux)
    if not np.isfinite(median_flux) or median_flux == 0:
        raise RuntimeError("Could not determine a finite non-zero median flux")

    relative_flux = flux / median_flux - 1.0
    relative_err = flux_err / median_flux

    good_err = np.isfinite(relative_err) & (relative_err > 0)
    if np.count_nonzero(good_err) < 0.9 * len(relative_err):
        relative_err = None
    else:
        # Avoid one pathological uncertainty giving enormous weight.
        finite_positive = relative_err[good_err]
        floor = np.nanpercentile(finite_positive, 1)
        relative_err = np.where(good_err, np.maximum(relative_err, floor), floor)

    order = np.argsort(time)
    time = time[order]
    relative_flux = relative_flux[order]
    if relative_err is not None:
        relative_err = relative_err[order]

    return time, relative_flux, relative_err


def make_frequency_grid(time):
    baseline = float(np.max(time) - np.min(time))
    df = 1.0 / (baseline * SAMPLES_PER_PEAK)
    n = 1 + int(math.ceil((MAX_FREQUENCY - MIN_FREQUENCY) / df))
    return MIN_FREQUENCY + df * np.arange(n, dtype=np.float64)


def compute_periodogram(time, flux, flux_err, frequency):
    ls = LombScargle(time, flux, dy=flux_err, center_data=True, fit_mean=True)
    power = ls.power(
        frequency,
        method="fast",
        assume_regular_frequency=True,
        normalization="standard",
    )
    return ls, np.asarray(power, dtype=np.float64)


def fixed_frequency_power(ls, frequency):
    value = ls.power(float(frequency), normalization="standard")
    return float(np.asarray(value))


def sinusoid_semi_amplitude_ppt(time, flux, frequency, flux_err=None):
    omega_t = 2.0 * np.pi * frequency * time
    design = np.column_stack(
        [np.ones_like(time), np.sin(omega_t), np.cos(omega_t)]
    )

    if flux_err is not None:
        weights = 1.0 / np.square(flux_err)
        root_w = np.sqrt(weights)
        weighted_design = design * root_w[:, None]
        weighted_flux = flux * root_w
        beta, *_ = np.linalg.lstsq(weighted_design, weighted_flux, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(design, flux, rcond=None)

    amplitude_relative = math.hypot(float(beta[1]), float(beta[2]))
    return 1000.0 * amplitude_relative


def best_peak(frequency, power, fmin=None, fmax=None):
    mask = np.ones_like(frequency, dtype=bool)
    if fmin is not None:
        mask &= frequency >= fmin
    if fmax is not None:
        mask &= frequency <= fmax

    if not np.any(mask):
        raise RuntimeError("Peak search window contains no frequency bins")

    idxs = np.flatnonzero(mask)
    local_idx = idxs[int(np.argmax(power[idxs]))]
    f = float(frequency[local_idx])
    return f, 1.0 / f, float(power[local_idx])


def top_distinct_peaks(frequency, power, baseline, count=5):
    minimum_separation = 1.0 / baseline
    accepted = []

    for idx in np.argsort(power)[::-1]:
        f = float(frequency[idx])
        if all(abs(f - existing[0]) >= minimum_separation for existing in accepted):
            accepted.append((f, float(power[idx])))
            if len(accepted) >= count:
                break

    return [(f, 1.0 / f, p) for f, p in accepted]


def plot_periodogram(output_path, sector, frequency, raw_power, masked_power):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(frequency, raw_power, linewidth=1.0, label="Before eclipse mask")
    ax.plot(frequency, masked_power, linewidth=1.0, label="After eclipse mask")
    ax.axvline(
        OPENSTAR_SIGNAL_F,
        linestyle="--",
        linewidth=1.2,
        label=f"Blind-F Astropy: {OPENSTAR_SIGNAL_F:.8f} d$^{{-1}}$",
    )
    ax.axvline(
        FOURTH_ORBITAL_HARMONIC_F,
        linestyle=":",
        linewidth=1.6,
        label=f"4 f_orb: {FOURTH_ORBITAL_HARMONIC_F:.8f} d$^{{-1}}$",
    )
    ax.set_xlim(0.10, 1.00)
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Lomb-Scargle power")
    ax.set_title(f"TIC {TIC} - TESS Sector {sector} - SPOC 120 s")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_orbital_mask(output_path, sector, time, flux, keep, in_primary, in_secondary):
    phase = orbital_phase(time)
    centered_phase = ((phase + 0.5) % 1.0) - 0.5

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(centered_phase[keep], 1.0 + flux[keep], s=2, alpha=0.35, label="Kept")
    ax.scatter(
        centered_phase[in_primary],
        1.0 + flux[in_primary],
        s=5,
        alpha=0.65,
        label="Masked primary",
    )
    ax.scatter(
        centered_phase[in_secondary],
        1.0 + flux[in_secondary],
        s=5,
        alpha=0.65,
        label="Masked secondary",
    )
    ax.set_xlabel("Orbital phase (primary at 0)")
    ax.set_ylabel("Normalized PDCSAP flux")
    ax.set_title(f"Sector {sector} eclipse mask at P_orb = {ORBITAL_PERIOD_D:.8f} d")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_blind_f_fold(output_path, sector, time, flux, keep):
    phase = ((time - time[0]) * OPENSTAR_SIGNAL_F) % 1.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(phase, 1.0 + flux, s=2, alpha=0.22, label="All points")
    ax.scatter(phase[keep], 1.0 + flux[keep], s=2, alpha=0.45, label="After eclipse mask")
    ax.set_xlabel(f"Phase at {OPENSTAR_SIGNAL_P:.6f} d")
    ax.set_ylabel("Normalized PDCSAP flux")
    ax.set_title(f"Sector {sector} folded at Blind-F ~4.67-day signal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_sector(sector, light_curve, output_dir):
    time, flux, flux_err = clean_arrays(light_curve)
    baseline = float(time[-1] - time[0])

    keep, in_primary, in_secondary = eclipse_keep_mask(time)
    if np.count_nonzero(keep) < 100:
        raise RuntimeError(f"Too few points remain after masking Sector {sector}")

    masked_time = time[keep]
    masked_flux = flux[keep]
    masked_err = flux_err[keep] if flux_err is not None else None

    # Identical frequency grid before/after masking is important for a fair comparison.
    frequency = make_frequency_grid(time)
    raw_ls, raw_power = compute_periodogram(time, flux, flux_err, frequency)
    masked_ls, masked_power = compute_periodogram(
        masked_time, masked_flux, masked_err, frequency
    )

    best_raw_f, best_raw_p, best_raw_pow = best_peak(frequency, raw_power)
    best_masked_f, best_masked_p, best_masked_pow = best_peak(frequency, masked_power)

    local_raw_f, local_raw_p, local_raw_pow = best_peak(
        frequency, raw_power, LOCAL_F_MIN, LOCAL_F_MAX
    )
    local_masked_f, local_masked_p, local_masked_pow = best_peak(
        frequency, masked_power, LOCAL_F_MIN, LOCAL_F_MAX
    )

    raw_openstar_power = fixed_frequency_power(raw_ls, OPENSTAR_SIGNAL_F)
    masked_openstar_power = fixed_frequency_power(masked_ls, OPENSTAR_SIGNAL_F)
    raw_4forb_power = fixed_frequency_power(raw_ls, FOURTH_ORBITAL_HARMONIC_F)
    masked_4forb_power = fixed_frequency_power(masked_ls, FOURTH_ORBITAL_HARMONIC_F)

    raw_openstar_amp = sinusoid_semi_amplitude_ppt(
        time, flux, OPENSTAR_SIGNAL_F, flux_err
    )
    masked_openstar_amp = sinusoid_semi_amplitude_ppt(
        masked_time, masked_flux, OPENSTAR_SIGNAL_F, masked_err
    )
    raw_4forb_amp = sinusoid_semi_amplitude_ppt(
        time, flux, FOURTH_ORBITAL_HARMONIC_F, flux_err
    )
    masked_4forb_amp = sinusoid_semi_amplitude_ppt(
        masked_time, masked_flux, FOURTH_ORBITAL_HARMONIC_F, masked_err
    )

    plot_periodogram(
        output_dir / f"sector_{sector:02d}_periodogram.png",
        sector,
        frequency,
        raw_power,
        masked_power,
    )
    plot_orbital_mask(
        output_dir / f"sector_{sector:02d}_orbital_mask.png",
        sector,
        time,
        flux,
        keep,
        in_primary,
        in_secondary,
    )
    plot_blind_f_fold(
        output_dir / f"sector_{sector:02d}_fold_4p67.png",
        sector,
        time,
        flux,
        keep,
    )

    raw_top = top_distinct_peaks(frequency, raw_power, baseline)
    masked_top = top_distinct_peaks(frequency, masked_power, baseline)

    return {
        "sector": sector,
        "samples_raw": len(time),
        "samples_masked": int(np.count_nonzero(keep)),
        "masked_fraction": 1.0 - np.count_nonzero(keep) / len(time),
        "baseline_d": baseline,
        "best_raw_frequency_cpd": best_raw_f,
        "best_raw_period_d": best_raw_p,
        "best_raw_power": best_raw_pow,
        "best_masked_frequency_cpd": best_masked_f,
        "best_masked_period_d": best_masked_p,
        "best_masked_power": best_masked_pow,
        "local_raw_frequency_cpd": local_raw_f,
        "local_raw_period_d": local_raw_p,
        "local_raw_power": local_raw_pow,
        "local_masked_frequency_cpd": local_masked_f,
        "local_masked_period_d": local_masked_p,
        "local_masked_power": local_masked_pow,
        "openstar_raw_power": raw_openstar_power,
        "openstar_masked_power": masked_openstar_power,
        "openstar_raw_amp_ppt": raw_openstar_amp,
        "openstar_masked_amp_ppt": masked_openstar_amp,
        "fourth_orbital_raw_power": raw_4forb_power,
        "fourth_orbital_masked_power": masked_4forb_power,
        "fourth_orbital_raw_amp_ppt": raw_4forb_amp,
        "fourth_orbital_masked_amp_ppt": masked_4forb_amp,
        "raw_top_peaks": raw_top,
        "masked_top_peaks": masked_top,
    }


def write_summary_csv(path, results):
    fields = [
        key
        for key in results[0].keys()
        if key not in {"raw_top_peaks", "masked_top_peaks"}
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})


def fmt_peak(peak):
    f, p, power = peak
    return f"f={f:.8f} c/d  P={p:.6f} d  power={power:.6f}"


def print_sector_result(result):
    print()
    print("=" * 78)
    print(f"SECTOR {result['sector']}")
    print("=" * 78)
    print(
        f"samples: {result['samples_raw']} -> {result['samples_masked']} after mask "
        f"({100.0 * result['masked_fraction']:.1f}% removed)"
    )
    print(f"baseline: {result['baseline_d']:.4f} d")
    print()
    print("Blind-F frequency from exact distributed samples")
    print(f"  f = {OPENSTAR_SIGNAL_F:.8f} c/d  P = {OPENSTAR_SIGNAL_P:.8f} d")
    print(
        f"  power: {result['openstar_raw_power']:.6f} -> "
        f"{result['openstar_masked_power']:.6f}"
    )
    print(
        f"  sine semi-amplitude: {result['openstar_raw_amp_ppt']:.4f} -> "
        f"{result['openstar_masked_amp_ppt']:.4f} ppt"
    )
    print()
    print("Exact fourth orbital harmonic")
    print(
        f"  f = {FOURTH_ORBITAL_HARMONIC_F:.8f} c/d  "
        f"P = {FOURTH_ORBITAL_HARMONIC_P:.8f} d"
    )
    print(
        f"  power: {result['fourth_orbital_raw_power']:.6f} -> "
        f"{result['fourth_orbital_masked_power']:.6f}"
    )
    print(
        f"  sine semi-amplitude: {result['fourth_orbital_raw_amp_ppt']:.4f} -> "
        f"{result['fourth_orbital_masked_amp_ppt']:.4f} ppt"
    )
    print()
    print("Best peak in 0.17-0.26 c/d neighborhood")
    print(
        f"  before: f={result['local_raw_frequency_cpd']:.8f} c/d  "
        f"P={result['local_raw_period_d']:.6f} d  "
        f"power={result['local_raw_power']:.6f}"
    )
    print(
        f"  after : f={result['local_masked_frequency_cpd']:.8f} c/d  "
        f"P={result['local_masked_period_d']:.6f} d  "
        f"power={result['local_masked_power']:.6f}"
    )
    print()
    print("Top 5 distinct peaks BEFORE masking")
    for peak in result["raw_top_peaks"]:
        print(f"  {fmt_peak(peak)}")
    print("Top 5 distinct peaks AFTER masking")
    for peak in result["masked_top_peaks"]:
        print(f"  {fmt_peak(peak)}")


def print_cross_sector_summary(results):
    print()
    print("#" * 78)
    print("CROSS-SECTOR DECISIVE SUMMARY")
    print("#" * 78)
    print(
        "sector | local ~4.67 d peak before -> after | "
        "4 f_orb amplitude before -> after (ppt)"
    )
    print("-" * 78)
    for result in results:
        print(
            f"{result['sector']:>6} | "
            f"P {result['local_raw_period_d']:.4f} / pow {result['local_raw_power']:.4f} "
            f"-> P {result['local_masked_period_d']:.4f} / "
            f"pow {result['local_masked_power']:.4f} | "
            f"{result['fourth_orbital_raw_amp_ppt']:.3f} -> "
            f"{result['fourth_orbital_masked_amp_ppt']:.3f}"
        )

    print()
    print("Interpretation rule for this first pass:")
    print(
        "  * If the ~0.214-0.216 c/d feature is strong before masking but collapses "
        "after masking in independent sectors, the EB-harmonic explanation wins."
    )
    print(
        "  * If a ~4.67-day feature remains after masking at similar frequency and "
        "non-trivial amplitude in multiple widely separated sectors, we continue "
        "with a dedicated rotational/contamination test."
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("⭐ OpenStar Blind-F decisive astrophysical test")
    print(f"Target: {TARGET}")
    print(f"Known EB period: {ORBITAL_PERIOD_D:.12f} d")
    print(f"Known EB t0: {T0_BTJD:.12f} BTJD")
    print(f"Secondary eclipse phase: {SECONDARY_PHASE:.3f}")
    print(f"Eclipse mask: +/- {ECLIPSE_MASK_HALF_WIDTH_PHASE:.3f} in phase")
    print(
        f"Blind-F signal: {OPENSTAR_SIGNAL_F:.8f} c/d = "
        f"{OPENSTAR_SIGNAL_P:.8f} d"
    )
    print(
        f"4th orbital harmonic: {FOURTH_ORBITAL_HARMONIC_F:.8f} c/d = "
        f"{FOURTH_ORBITAL_HARMONIC_P:.8f} d"
    )
    print()
    print("🔭 Searching MAST for official SPOC 120-second light curves...")

    search_result = lk.search_lightcurve(
        TARGET,
        mission="TESS",
        author="SPOC",
        exptime=120,
    )

    if len(search_result) == 0:
        raise RuntimeError("No SPOC 120-second TESS light curves found")

    by_sector = available_sector_indices(search_result)
    available = sorted(by_sector.keys())
    print("Available SPOC 120-s sectors:")
    print(", ".join(str(s) for s in available))

    if args.sectors.strip().lower() == "all":
        requested = available
    else:
        requested = [
            int(value.strip())
            for value in args.sectors.split(",")
            if value.strip()
        ]

    selected = [sector for sector in requested if sector in by_sector]
    missing = [sector for sector in requested if sector not in by_sector]

    if missing:
        print(f"Skipping unavailable requested sectors: {missing}")
    if not selected:
        raise RuntimeError("None of the requested sectors are available as SPOC 120-s")

    print(f"Testing sectors: {selected}")

    results = []
    for sector in selected:
        print()
        print(f"⬇️ Downloading Sector {sector}...")
        product = search_result[by_sector[sector][0]]
        light_curve = product.download(
            quality_bitmask="default",
            flux_column="pdcsap_flux",
        )
        if light_curve is None:
            print(f"WARNING: download returned no light curve for Sector {sector}; skipping")
            continue

        result = analyze_sector(sector, light_curve, output_dir)
        results.append(result)
        print_sector_result(result)

    if not results:
        raise RuntimeError("No sectors were successfully analyzed")

    summary_path = output_dir / "summary.csv"
    write_summary_csv(summary_path, results)
    print_cross_sector_summary(results)
    print(f"Results CSV: {summary_path}")
    print(f"Plots: {output_dir}/sector_*_periodogram.png")
    print(f"       {output_dir}/sector_*_orbital_mask.png")
    print(f"       {output_dir}/sector_*_fold_4p67.png")


if __name__ == "__main__":
    main()
