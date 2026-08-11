#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle

TIC = 404927661
TARGET = f"TIC {TIC}"

# Published TESS eclipsing-binary ephemeris used in the decisive follow-up.
ORBITAL_PERIOD_D = 18.536282847200237
T0_BTJD = 1333.9773604614618
SECONDARY_PHASE = 0.235
ORBITAL_FREQUENCY_CPD = 1.0 / ORBITAL_PERIOD_D

# Blind-F Astropy reference from the exact Float32 samples distributed by OpenStar.
BLIND_F_FREQUENCY_CPD = 0.21384606
BLIND_F_PERIOD_D = 1.0 / BLIND_F_FREQUENCY_CPD

FOURTH_HARMONIC_FREQUENCY_CPD = 4.0 * ORBITAL_FREQUENCY_CPD
FOURTH_HARMONIC_PERIOD_D = 1.0 / FOURTH_HARMONIC_FREQUENCY_CPD

# Keep this identical to the decisive experiment.
ECLIPSE_MASK_HALF_WIDTH_PHASE = 0.03

# Wide enough to show the orbital fundamental and the harmonic comb that dominated
# the single-sector periodograms, without filling the figure with irrelevant
# high-frequency structure.
MIN_FREQUENCY_CPD = 0.03
MAX_FREQUENCY_CPD = 1.20
SAMPLES_PER_PEAK = 25

DEFAULT_SECTORS = [1, 27, 35, 61, 87]


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a presentation-quality multi-sector verification figure for "
            "the Blind-F ~4.67-day signal in TIC 404927661."
        )
    )
    parser.add_argument(
        "--sectors",
        default=",".join(str(s) for s in DEFAULT_SECTORS),
        help=(
            "Comma-separated SPOC 120-s sectors to combine, or 'all'. "
            f"Default: {','.join(str(s) for s in DEFAULT_SECTORS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="blind_f_harmonic_verification_v2",
        help="Directory for the verification figure and supporting CSV files.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Light-curve handling
# -----------------------------------------------------------------------------

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


def clean_sector_light_curve(light_curve):
    """
    Return BTJD, per-sector relative PDCSAP flux, and relative uncertainty.

    Each sector is normalized by its own median before sectors are combined.
    This avoids stitching offsets from masquerading as long-period variability.
    """
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
        raise RuntimeError("Could not determine a finite non-zero sector median flux")

    relative_flux = flux / median_flux - 1.0
    relative_err = flux_err / median_flux

    good_err = np.isfinite(relative_err) & (relative_err > 0)
    if np.count_nonzero(good_err) < 0.9 * len(relative_err):
        relative_err = None
    else:
        finite_positive = relative_err[good_err]
        floor = np.nanpercentile(finite_positive, 1)
        relative_err = np.where(good_err, np.maximum(relative_err, floor), floor)

    order = np.argsort(time)
    time = time[order]
    relative_flux = relative_flux[order]
    if relative_err is not None:
        relative_err = relative_err[order]

    return time, relative_flux, relative_err


def circular_phase_distance(phase, center):
    return np.abs(((phase - center + 0.5) % 1.0) - 0.5)


def orbital_phase(time_btjd):
    return ((time_btjd - T0_BTJD) / ORBITAL_PERIOD_D) % 1.0


def eclipse_keep_mask(time_btjd):
    phase = orbital_phase(time_btjd)
    in_primary = (
        circular_phase_distance(phase, 0.0)
        <= ECLIPSE_MASK_HALF_WIDTH_PHASE
    )
    in_secondary = (
        circular_phase_distance(phase, SECONDARY_PHASE)
        <= ECLIPSE_MASK_HALF_WIDTH_PHASE
    )
    keep = ~(in_primary | in_secondary)
    return keep, in_primary, in_secondary


# -----------------------------------------------------------------------------
# Periodogram / amplitudes
# -----------------------------------------------------------------------------

def make_frequency_grid(time):
    baseline = float(np.max(time) - np.min(time))
    if baseline <= 0:
        raise RuntimeError("Combined baseline is not positive")

    df = 1.0 / (baseline * SAMPLES_PER_PEAK)
    count = 1 + int(
        math.ceil((MAX_FREQUENCY_CPD - MIN_FREQUENCY_CPD) / df)
    )
    return MIN_FREQUENCY_CPD + df * np.arange(count, dtype=np.float64)


def compute_periodogram(time, flux, flux_err, frequency):
    ls = LombScargle(
        time,
        flux,
        dy=flux_err,
        center_data=True,
        fit_mean=True,
    )
    power = ls.power(
        frequency,
        method="fast",
        assume_regular_frequency=True,
        normalization="standard",
    )
    return ls, np.asarray(power, dtype=np.float64)


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


def fixed_frequency_power(ls, frequency):
    return float(
        np.asarray(ls.power(float(frequency), normalization="standard"))
    )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def binned_phase_curve(phase, flux_ppt, bins=500):
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.digitize(phase, edges) - 1

    medians = np.full(bins, np.nan, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)

    for i in range(bins):
        values = flux_ppt[indices == i]
        finite = values[np.isfinite(values)]
        if len(finite) > 0:
            medians[i] = np.nanmedian(finite)
            counts[i] = len(finite)

    valid = np.isfinite(medians)
    return centers[valid], medians[valid], counts[valid]


def add_harmonic_lines(ax):
    max_n = int(math.floor(MAX_FREQUENCY_CPD / ORBITAL_FREQUENCY_CPD))

    for n in range(1, max_n + 1):
        frequency = n * ORBITAL_FREQUENCY_CPD
        ax.axvline(frequency, linewidth=0.7, alpha=0.20)

    # Label only the low-order harmonics to keep the figure readable.
    y0, y1 = ax.get_ylim()
    label_y = y0 + 0.94 * (y1 - y0)
    for n in range(1, min(max_n, 10) + 1):
        frequency = n * ORBITAL_FREQUENCY_CPD
        ax.text(
            frequency,
            label_y,
            f"{n}f",
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
            alpha=0.65,
        )

    # Make the specific Blind-F interpretation explicit.
    ax.axvline(
        FOURTH_HARMONIC_FREQUENCY_CPD,
        linestyle="--",
        linewidth=1.8,
        label=(
            f"4f_orb = {FOURTH_HARMONIC_FREQUENCY_CPD:.6f} c/d "
            f"({FOURTH_HARMONIC_PERIOD_D:.6f} d)"
        ),
    )
    ax.axvline(
        BLIND_F_FREQUENCY_CPD,
        linestyle=":",
        linewidth=1.8,
        label=(
            f"Blind-F Astropy = {BLIND_F_FREQUENCY_CPD:.6f} c/d "
            f"({BLIND_F_PERIOD_D:.6f} d)"
        ),
    )


def make_verification_figure(
    output_base,
    sectors,
    time,
    flux,
    keep,
    frequency,
    raw_power,
    masked_power,
    raw_amp_4f,
    masked_amp_4f,
):
    phase = orbital_phase(time)
    centered_phase = ((phase + 0.5) % 1.0) - 0.5
    flux_ppt = 1000.0 * flux

    bin_phase, bin_flux, _ = binned_phase_curve(centered_phase, flux_ppt)

    fig = plt.figure(figsize=(12.5, 13.0))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.0], hspace=0.28)

    # Panel A: actual eclipsing-binary waveform.
    ax_fold = fig.add_subplot(grid[0, 0])
    ax_fold.scatter(
        centered_phase,
        flux_ppt,
        s=1.0,
        alpha=0.08,
        rasterized=True,
        label="SPOC 120-s samples",
    )
    ax_fold.plot(
        bin_phase,
        bin_flux,
        linewidth=1.5,
        label="Phase-bin median",
    )
    ax_fold.axvspan(
        -ECLIPSE_MASK_HALF_WIDTH_PHASE,
        ECLIPSE_MASK_HALF_WIDTH_PHASE,
        alpha=0.10,
        label="Masked eclipse windows",
    )
    secondary_left = SECONDARY_PHASE - ECLIPSE_MASK_HALF_WIDTH_PHASE
    secondary_right = SECONDARY_PHASE + ECLIPSE_MASK_HALF_WIDTH_PHASE
    ax_fold.axvspan(secondary_left, secondary_right, alpha=0.10)
    ax_fold.set_xlim(-0.5, 0.5)
    ax_fold.set_xlabel("Orbital phase")
    ax_fold.set_ylabel("Relative PDCSAP flux (ppt)")
    ax_fold.set_title(
        "A. Multi-sector light curve folded on the known eclipsing-binary orbit"
    )
    ax_fold.legend(loc="lower right", fontsize=8)

    # Panels B/C: same frequencies and same y-scale for direct visual comparison.
    shared_power_max = 1.05 * max(
        float(np.nanmax(raw_power)),
        float(np.nanmax(masked_power)),
    )

    ax_raw = fig.add_subplot(grid[1, 0])
    ax_raw.plot(frequency, raw_power, linewidth=0.9)
    ax_raw.set_xlim(MIN_FREQUENCY_CPD, MAX_FREQUENCY_CPD)
    ax_raw.set_ylim(0.0, shared_power_max)
    ax_raw.set_ylabel("Lomb-Scargle power")
    ax_raw.set_title(
        "B. Before eclipse masking: integer harmonics dominate the spectrum"
    )
    add_harmonic_lines(ax_raw)
    ax_raw.legend(loc="upper right", fontsize=8)

    ax_masked = fig.add_subplot(grid[2, 0])
    ax_masked.plot(frequency, masked_power, linewidth=0.9)
    ax_masked.set_xlim(MIN_FREQUENCY_CPD, MAX_FREQUENCY_CPD)
    ax_masked.set_ylim(0.0, shared_power_max)
    ax_masked.set_xlabel("Frequency (cycles/day)")
    ax_masked.set_ylabel("Lomb-Scargle power")
    ax_masked.set_title(
        "C. After masking both eclipses: the ~4.67-day / 4f feature collapses"
    )
    add_harmonic_lines(ax_masked)
    ax_masked.legend(loc="upper right", fontsize=8)

    sector_text = ", ".join(str(s) for s in sectors)
    reduction = 100.0 * (1.0 - masked_amp_4f / raw_amp_4f)
    fig.suptitle(
        "TIC 404927661 / HD 38600 — Blind-F harmonic verification\n"
        f"TESS Sectors {sector_text} | P_orb = {ORBITAL_PERIOD_D:.9f} d | "
        f"4f semi-amplitude {raw_amp_4f:.3f} → {masked_amp_4f:.3f} ppt "
        f"({reduction:.1f}% reduction)",
        fontsize=14,
        y=0.985,
    )

    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.09, right=0.98)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


# -----------------------------------------------------------------------------
# Supporting outputs
# -----------------------------------------------------------------------------

def write_harmonic_csv(path, raw_ls, masked_ls, time, flux, masked_time, masked_flux,
                       flux_err, masked_err):
    max_n = int(math.floor(MAX_FREQUENCY_CPD / ORBITAL_FREQUENCY_CPD))
    rows = []

    for n in range(1, max_n + 1):
        frequency = n * ORBITAL_FREQUENCY_CPD
        rows.append(
            {
                "harmonic_n": n,
                "frequency_cpd": frequency,
                "period_d": 1.0 / frequency,
                "raw_power": fixed_frequency_power(raw_ls, frequency),
                "masked_power": fixed_frequency_power(masked_ls, frequency),
                "raw_semi_amplitude_ppt": sinusoid_semi_amplitude_ppt(
                    time, flux, frequency, flux_err
                ),
                "masked_semi_amplitude_ppt": sinusoid_semi_amplitude_ppt(
                    masked_time, masked_flux, frequency, masked_err
                ),
            }
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_sector_csv(path, sector_rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sector_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sector_rows)


def write_summary_text(
    path,
    sectors,
    total_samples,
    kept_samples,
    combined_baseline,
    raw_power_blind_f,
    masked_power_blind_f,
    raw_amp_blind_f,
    masked_amp_blind_f,
    raw_power_4f,
    masked_power_4f,
    raw_amp_4f,
    masked_amp_4f,
):
    reduction_4f = 100.0 * (1.0 - masked_amp_4f / raw_amp_4f)
    reduction_blind_f = 100.0 * (1.0 - masked_amp_blind_f / raw_amp_blind_f)

    text = "OpenStar Blind-F harmonic verification\n\n"
    text += f"Target: {TARGET} / HD 38600\n"
    text += f"Sectors: {', '.join(str(s) for s in sectors)}\n"
    text += f"Total samples: {total_samples}\n"
    text += f"Samples after eclipse mask: {kept_samples}\n"
    text += f"Combined time span: {combined_baseline:.4f} d\n\n"
    text += f"Known EB period: {ORBITAL_PERIOD_D:.12f} d\n"
    text += f"Orbital frequency: {ORBITAL_FREQUENCY_CPD:.12f} c/d\n"
    text += f"Fourth harmonic: {FOURTH_HARMONIC_FREQUENCY_CPD:.12f} c/d = {FOURTH_HARMONIC_PERIOD_D:.12f} d\n"
    text += f"Blind-F Astropy: {BLIND_F_FREQUENCY_CPD:.8f} c/d = {BLIND_F_PERIOD_D:.8f} d\n\n"
    text += "Blind-F fixed frequency\n"
    text += f"  LS power: {raw_power_blind_f:.8f} -> {masked_power_blind_f:.8f}\n"
    text += f"  sine semi-amplitude: {raw_amp_blind_f:.6f} -> {masked_amp_blind_f:.6f} ppt\n"
    text += f"  amplitude reduction: {reduction_blind_f:.2f}%\n\n"
    text += "Exact fourth orbital harmonic\n"
    text += f"  LS power: {raw_power_4f:.8f} -> {masked_power_4f:.8f}\n"
    text += f"  sine semi-amplitude: {raw_amp_4f:.6f} -> {masked_amp_4f:.6f} ppt\n"
    text += f"  amplitude reduction: {reduction_4f:.2f}%\n\n"
    text += "Interpretation:\n"
    text += "  The ~4.67-day Blind-F feature is tested against the known eclipsing-binary ephemeris.\n"
    text += "  A harmonic origin is supported when the integer harmonic comb is present before masking\n"
    text += "  and the 4f / ~4.67-day component collapses after the primary and secondary eclipses are removed.\n"

    path.write_text(text)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("⭐ OpenStar Blind-F presentation verification")
    print(f"Target: {TARGET} / HD 38600")
    print(f"Known EB period: {ORBITAL_PERIOD_D:.12f} d")
    print(f"Orbital frequency: {ORBITAL_FREQUENCY_CPD:.12f} c/d")
    print(
        f"Fourth harmonic: {FOURTH_HARMONIC_FREQUENCY_CPD:.12f} c/d "
        f"= {FOURTH_HARMONIC_PERIOD_D:.12f} d"
    )
    print(
        f"Blind-F Astropy: {BLIND_F_FREQUENCY_CPD:.8f} c/d "
        f"= {BLIND_F_PERIOD_D:.8f} d"
    )
    print(f"Eclipse mask: +/- {ECLIPSE_MASK_HALF_WIDTH_PHASE:.3f} orbital phase")
    print()
    print("🔭 Searching MAST for SPOC 120-second TESS light curves...")

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

    print(f"Combining sectors: {selected}")

    all_time = []
    all_flux = []
    all_err = []
    have_all_errors = True
    sector_rows = []

    for sector in selected:
        print(f"⬇️ Downloading Sector {sector}...")
        product = search_result[by_sector[sector][0]]
        light_curve = product.download(
            quality_bitmask="default",
            flux_column="pdcsap_flux",
        )
        if light_curve is None:
            print(f"WARNING: no light curve returned for Sector {sector}; skipping")
            continue

        time, flux, flux_err = clean_sector_light_curve(light_curve)
        keep, _, _ = eclipse_keep_mask(time)

        sector_rows.append(
            {
                "sector": sector,
                "samples": len(time),
                "samples_after_mask": int(np.count_nonzero(keep)),
                "masked_fraction": 1.0 - np.count_nonzero(keep) / len(time),
                "sector_baseline_d": float(time[-1] - time[0]),
            }
        )

        all_time.append(time)
        all_flux.append(flux)
        if flux_err is None:
            have_all_errors = False
        else:
            all_err.append(flux_err)

    if not all_time:
        raise RuntimeError("No sectors were successfully downloaded")

    selected = [row["sector"] for row in sector_rows]
    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)
    flux_err = np.concatenate(all_err) if have_all_errors else None

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    if flux_err is not None:
        flux_err = flux_err[order]

    keep, in_primary, in_secondary = eclipse_keep_mask(time)
    masked_time = time[keep]
    masked_flux = flux[keep]
    masked_err = flux_err[keep] if flux_err is not None else None

    combined_baseline = float(time[-1] - time[0])
    print()
    print(f"Combined samples: {len(time):,}")
    print(f"After eclipse mask: {len(masked_time):,}")
    print(f"Removed: {100.0 * (1.0 - len(masked_time) / len(time)):.1f}%")
    print(f"Combined time span: {combined_baseline:.4f} d")

    print()
    print("📈 Computing identical before/after multi-sector Lomb-Scargle spectra...")
    frequency = make_frequency_grid(time)
    raw_ls, raw_power = compute_periodogram(time, flux, flux_err, frequency)
    masked_ls, masked_power = compute_periodogram(
        masked_time,
        masked_flux,
        masked_err,
        frequency,
    )

    raw_power_blind_f = fixed_frequency_power(raw_ls, BLIND_F_FREQUENCY_CPD)
    masked_power_blind_f = fixed_frequency_power(masked_ls, BLIND_F_FREQUENCY_CPD)
    raw_amp_blind_f = sinusoid_semi_amplitude_ppt(
        time, flux, BLIND_F_FREQUENCY_CPD, flux_err
    )
    masked_amp_blind_f = sinusoid_semi_amplitude_ppt(
        masked_time, masked_flux, BLIND_F_FREQUENCY_CPD, masked_err
    )

    raw_power_4f = fixed_frequency_power(raw_ls, FOURTH_HARMONIC_FREQUENCY_CPD)
    masked_power_4f = fixed_frequency_power(masked_ls, FOURTH_HARMONIC_FREQUENCY_CPD)
    raw_amp_4f = sinusoid_semi_amplitude_ppt(
        time, flux, FOURTH_HARMONIC_FREQUENCY_CPD, flux_err
    )
    masked_amp_4f = sinusoid_semi_amplitude_ppt(
        masked_time, masked_flux, FOURTH_HARMONIC_FREQUENCY_CPD, masked_err
    )

    blind_reduction = 100.0 * (1.0 - masked_amp_blind_f / raw_amp_blind_f)
    fourth_reduction = 100.0 * (1.0 - masked_amp_4f / raw_amp_4f)

    print()
    print("Blind-F frequency")
    print(
        f"  power: {raw_power_blind_f:.8f} -> {masked_power_blind_f:.8f}"
    )
    print(
        f"  sine semi-amplitude: {raw_amp_blind_f:.6f} -> "
        f"{masked_amp_blind_f:.6f} ppt ({blind_reduction:.2f}% reduction)"
    )

    print("Exact fourth orbital harmonic")
    print(f"  power: {raw_power_4f:.8f} -> {masked_power_4f:.8f}")
    print(
        f"  sine semi-amplitude: {raw_amp_4f:.6f} -> "
        f"{masked_amp_4f:.6f} ppt ({fourth_reduction:.2f}% reduction)"
    )

    harmonic_csv = output_dir / "harmonics_before_after.csv"
    write_harmonic_csv(
        harmonic_csv,
        raw_ls,
        masked_ls,
        time,
        flux,
        masked_time,
        masked_flux,
        flux_err,
        masked_err,
    )

    sector_csv = output_dir / "sector_inputs.csv"
    write_sector_csv(sector_csv, sector_rows)

    summary_txt = output_dir / "verification_summary.txt"
    write_summary_text(
        summary_txt,
        selected,
        len(time),
        len(masked_time),
        combined_baseline,
        raw_power_blind_f,
        masked_power_blind_f,
        raw_amp_blind_f,
        masked_amp_blind_f,
        raw_power_4f,
        masked_power_4f,
        raw_amp_4f,
        masked_amp_4f,
    )

    output_base = output_dir / "blind_f_harmonic_verification"
    png_path, pdf_path = make_verification_figure(
        output_base,
        selected,
        time,
        flux,
        keep,
        frequency,
        raw_power,
        masked_power,
        raw_amp_4f,
        masked_amp_4f,
    )

    print()
    print("✅ Verification outputs")
    print(f"Figure PNG: {png_path}")
    print(f"Figure PDF: {pdf_path}")
    print(f"Harmonic table: {harmonic_csv}")
    print(f"Sector inputs: {sector_csv}")
    print(f"Summary: {summary_txt}")
    print()
    print("What should be visually true if the EB-harmonic interpretation is correct:")
    print("  1. The folded light curve shows the deep primary and shallow secondary eclipse.")
    print("  2. Before masking, periodogram peaks align with integer multiples of f_orb.")
    print("  3. The ~4.67-day / 4f component collapses after both eclipses are removed.")


if __name__ == "__main__":
    main()
