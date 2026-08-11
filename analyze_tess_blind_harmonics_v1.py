import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REVEAL_PATH = Path(
    "data/projects/"
    "openstar.tess-blind-published-v3.reveal-v1.json"
)

OUTPUT_DIR = Path(
    "data/analysis/"
    "openstar-blind-published-v3-harmonics-v1"
)

TARGETS = (
    "Blind V2-B",
    "Blind V2-D",
    "Blind V2-E",
    "Blind V2-H",
)

DATASET_PATHS = {
    "Blind V2-B": Path(
        "data/tess-blind-v2-b-tic-356108440.json"
    ),
    "Blind V2-D": Path(
        "data/tess-blind-v2-d-tic-233684019.json"
    ),
    "Blind V2-E": Path(
        "data/tess-blind-v2-e-tic-233679640.json"
    ),
    "Blind V2-H": Path(
        "data/tess-blind-v2-h-tic-349231109.json"
    ),
}

PHASE_BINS = 120
MIN_POINTS_PER_BIN = 3


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def safe_float(value):
    if value is None:
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def phase_for_period(
    times: np.ndarray,
    period_days: float,
) -> np.ndarray:
    return np.mod(
        times / period_days,
        1.0,
    )


def binned_profile(
    phase: np.ndarray,
    flux: np.ndarray,
    bin_count: int = PHASE_BINS,
):
    edges = np.linspace(
        0.0,
        1.0,
        bin_count + 1,
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) * 0.5

    indices = np.floor(
        phase * bin_count
    ).astype(np.int64)

    indices = np.clip(
        indices,
        0,
        bin_count - 1,
    )

    means = np.full(
        bin_count,
        np.nan,
        dtype=np.float64,
    )

    counts = np.zeros(
        bin_count,
        dtype=np.int64,
    )

    for bin_index in range(
        bin_count
    ):
        mask = (
            indices
            == bin_index
        )

        count = int(
            np.count_nonzero(mask)
        )

        counts[
            bin_index
        ] = count

        if count == 0:
            continue

        means[
            bin_index
        ] = float(
            np.mean(
                flux[mask]
            )
        )

    return {
        "edges": edges,
        "centers": centers,
        "indices": indices,
        "means": means,
        "counts": counts,
    }


def circular_smooth(
    values: np.ndarray,
    width: int = 5,
):
    result = np.array(
        values,
        dtype=np.float64,
        copy=True,
    )

    finite = np.isfinite(
        result
    )

    if not np.any(
        finite
    ):
        return result

    # Fill missing bins circularly with the nearest finite
    # interpolation before applying a short moving average.
    x = np.arange(
        len(result),
        dtype=np.float64,
    )

    finite_x = x[
        finite
    ]

    finite_y = result[
        finite
    ]

    extended_x = np.concatenate(
        (
            finite_x
            - len(result),
            finite_x,
            finite_x
            + len(result),
        )
    )

    extended_y = np.concatenate(
        (
            finite_y,
            finite_y,
            finite_y,
        )
    )

    filled = np.interp(
        x,
        extended_x,
        extended_y,
    )

    half = width // 2

    padded = np.concatenate(
        (
            filled[
                -half:
            ],
            filled,
            filled[
                :half
            ],
        )
    )

    kernel = np.ones(
        width,
        dtype=np.float64,
    ) / width

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def within_bin_rms(
    flux: np.ndarray,
    profile,
):
    indices = profile[
        "indices"
    ]

    means = profile[
        "means"
    ]

    residuals = []

    for bin_index in range(
        len(means)
    ):
        if (
            profile[
                "counts"
            ][
                bin_index
            ]
            < MIN_POINTS_PER_BIN
        ):
            continue

        mask = (
            indices
            == bin_index
        )

        residuals.append(
            flux[mask]
            - means[
                bin_index
            ]
        )

    if not residuals:
        return None

    residual = np.concatenate(
        residuals
    )

    return float(
        np.sqrt(
            np.mean(
                residual * residual
            )
        )
    )


def two_minimum_metrics(
    profile,
):
    means = profile[
        "means"
    ]

    centers = profile[
        "centers"
    ]

    smoothed = circular_smooth(
        means,
        width=5,
    )

    if not np.any(
        np.isfinite(
            smoothed
        )
    ):
        return None

    primary_index = int(
        np.nanargmin(
            smoothed
        )
    )

    primary_phase = float(
        centers[
            primary_index
        ]
    )

    shifted_phase = np.mod(
        centers
        - primary_phase,
        1.0,
    )

    median_flux = float(
        np.nanmedian(
            smoothed
        )
    )

    primary_flux = float(
        smoothed[
            primary_index
        ]
    )

    primary_depth = (
        median_flux
        - primary_flux
    )

    secondary_mask = (
        (shifted_phase >= 0.35)
        & (shifted_phase <= 0.65)
        & np.isfinite(smoothed)
    )

    if not np.any(
        secondary_mask
    ):
        return None

    secondary_indices = np.where(
        secondary_mask
    )[0]

    secondary_index = int(
        secondary_indices[
            np.argmin(
                smoothed[
                    secondary_indices
                ]
            )
        ]
    )

    secondary_flux = float(
        smoothed[
            secondary_index
        ]
    )

    secondary_depth = (
        median_flux
        - secondary_flux
    )

    secondary_separation = float(
        shifted_phase[
            secondary_index
        ]
    )

    if primary_depth > 0:
        secondary_to_primary = (
            secondary_depth
            / primary_depth
        )
    else:
        secondary_to_primary = None

    return {
        "primaryPhaseBeforeShift": (
            primary_phase
        ),
        "primaryDepth": (
            primary_depth
        ),
        "secondaryDepth": (
            secondary_depth
        ),
        "secondaryPhaseSeparation": (
            secondary_separation
        ),
        "secondaryToPrimaryDepthRatio": (
            secondary_to_primary
        ),
    }


def odd_even_half_cycle_metrics(
    times: np.ndarray,
    flux: np.ndarray,
    half_period_days: float,
):
    cycle_number = np.floor(
        times
        / half_period_days
    ).astype(np.int64)

    phase = phase_for_period(
        times,
        half_period_days,
    )

    even_mask = (
        cycle_number % 2
        == 0
    )

    odd_mask = ~even_mask

    even = binned_profile(
        phase[
            even_mask
        ],
        flux[
            even_mask
        ],
    )

    odd = binned_profile(
        phase[
            odd_mask
        ],
        flux[
            odd_mask
        ],
    )

    valid = (
        (even["counts"] >= MIN_POINTS_PER_BIN)
        & (odd["counts"] >= MIN_POINTS_PER_BIN)
        & np.isfinite(
            even[
                "means"
            ]
        )
        & np.isfinite(
            odd[
                "means"
            ]
        )
    )

    if not np.any(valid):
        return {
            "comparableBins": 0,
            "rmsProfileDifference": None,
            "maxAbsProfileDifference": None,
        }

    difference = (
        even[
            "means"
        ][valid]
        - odd[
            "means"
        ][valid]
    )

    return {
        "comparableBins": int(
            np.count_nonzero(
                valid
            )
        ),
        "rmsProfileDifference": float(
            np.sqrt(
                np.mean(
                    difference
                    * difference
                )
            )
        ),
        "maxAbsProfileDifference": float(
            np.max(
                np.abs(
                    difference
                )
            )
        ),
    }


def profile_metrics(
    times: np.ndarray,
    flux: np.ndarray,
    period_days: float,
):
    phase = phase_for_period(
        times,
        period_days,
    )

    profile = binned_profile(
        phase,
        flux,
    )

    return {
        "periodDays": (
            period_days
        ),
        "withinBinRMS": (
            within_bin_rms(
                flux,
                profile,
            )
        ),
        "occupiedBins": int(
            np.count_nonzero(
                profile[
                    "counts"
                ]
            )
        ),
        "twoMinima": (
            two_minimum_metrics(
                profile
            )
        ),
        "profile": profile,
        "phase": phase,
    }


def save_fold_plot(
    output_path: Path,
    *,
    blind_name: str,
    variable_type: str,
    label: str,
    period_days: float,
    phase: np.ndarray,
    flux: np.ndarray,
    profile,
):
    figure = plt.figure(
        figsize=(
            9,
            5,
        )
    )

    axes = figure.add_subplot(
        111
    )

    # Show two cycles because eclipse/spot morphology is easier
    # to inspect around the phase boundary.
    plot_phase = np.concatenate(
        (
            phase,
            phase + 1.0,
        )
    )

    plot_flux = np.concatenate(
        (
            flux,
            flux,
        )
    )

    axes.scatter(
        plot_phase,
        plot_flux,
        s=3,
        alpha=0.16,
    )

    means = profile[
        "means"
    ]

    centers = profile[
        "centers"
    ]

    finite = np.isfinite(
        means
    )

    if np.any(
        finite
    ):
        profile_phase = np.concatenate(
            (
                centers[
                    finite
                ],
                centers[
                    finite
                ]
                + 1.0,
            )
        )

        profile_flux = np.concatenate(
            (
                means[
                    finite
                ],
                means[
                    finite
                ],
            )
        )

        axes.plot(
            profile_phase,
            profile_flux,
            linewidth=2.0,
        )

    axes.set_xlim(
        0.0,
        2.0,
    )

    axes.set_xlabel(
        "Phase"
    )

    axes.set_ylabel(
        "Normalized flux"
    )

    axes.set_title(
        f"{blind_name} ({variable_type}) — "
        f"{label}: {period_days:.8f} d"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        figure
    )


def format_optional(
    value,
    digits=6,
):
    if value is None:
        return "[n/a]"

    return (
        f"{value:.{digits}f}"
    )


def analyze_target(
    reveal_target,
    dataset_path: Path,
    output_dir: Path,
):
    blind_name = reveal_target[
        "blindName"
    ]

    dataset = load_json(
        dataset_path
    )

    if (
        dataset.get(
            "targetName"
        )
        != blind_name
    ):
        raise RuntimeError(
            f"Dataset target mismatch for "
            f"{blind_name}: "
            f"{dataset.get('targetName')}"
        )

    times = np.asarray(
        dataset[
            "times"
        ],
        dtype=np.float64,
    )

    flux = np.asarray(
        dataset[
            "flux"
        ],
        dtype=np.float64,
    )

    finite = (
        np.isfinite(
            times
        )
        & np.isfinite(
            flux
        )
    )

    times = times[
        finite
    ]

    flux = flux[
        finite
    ]

    openstar_period = float(
        reveal_target[
            "frozen"
        ][
            "openstarPeriodDays"
        ]
    )

    vsx_period = float(
        reveal_target[
            "vsx"
        ][
            "periodDays"
        ]
    )

    variable_type = str(
        reveal_target[
            "vsx"
        ].get(
            "type"
        )
        or "?"
    )

    doubled_period = (
        2.0
        * openstar_period
    )

    half_metrics = profile_metrics(
        times,
        flux,
        openstar_period,
    )

    doubled_metrics = profile_metrics(
        times,
        flux,
        doubled_period,
    )

    vsx_metrics = profile_metrics(
        times,
        flux,
        vsx_period,
    )

    odd_even = (
        odd_even_half_cycle_metrics(
            times,
            flux,
            openstar_period,
        )
    )

    harmonic_error_percent = (
        abs(
            doubled_period
            - vsx_period
        )
        / vsx_period
        * 100.0
    )

    baseline_days = float(
        np.max(
            times
        )
        - np.min(
            times
        )
    )

    phase_drift_cycles = (
        baseline_days
        * abs(
            (1.0 / doubled_period)
            - (1.0 / vsx_period)
        )
    )

    half_rms = half_metrics[
        "withinBinRMS"
    ]

    doubled_rms = doubled_metrics[
        "withinBinRMS"
    ]

    if (
        half_rms is not None
        and doubled_rms is not None
        and half_rms > 0
    ):
        full_period_coherence_gain = (
            (half_rms - doubled_rms)
            / half_rms
            * 100.0
        )
    else:
        full_period_coherence_gain = None

    target_slug = (
        blind_name
        .lower()
        .replace(
            " ",
            "-",
        )
    )

    plot_specs = (
        (
            "openstar-half",
            "OpenStar",
            half_metrics,
        ),
        (
            "openstar-doubled",
            "2 × OpenStar",
            doubled_metrics,
        ),
        (
            "vsx",
            "VSX",
            vsx_metrics,
        ),
    )

    plot_paths = {}

    for key, label, metrics in plot_specs:
        output_path = (
            output_dir
            / (
                f"{target_slug}-"
                f"{key}.png"
            )
        )

        save_fold_plot(
            output_path,
            blind_name=blind_name,
            variable_type=(
                variable_type
            ),
            label=label,
            period_days=(
                metrics[
                    "periodDays"
                ]
            ),
            phase=metrics[
                "phase"
            ],
            flux=flux,
            profile=metrics[
                "profile"
            ],
        )

        plot_paths[
            key
        ] = str(
            output_path
        )

    result = {
        "blindName": blind_name,
        "ticID": int(
            reveal_target[
                "ticID"
            ]
        ),
        "vsxName": reveal_target[
            "vsx"
        ].get(
            "name"
        ),
        "vsxType": variable_type,
        "samples": int(
            len(
                times
            )
        ),
        "baselineDays": (
            baseline_days
        ),
        "openstarPeriodDays": (
            openstar_period
        ),
        "doubledOpenstarPeriodDays": (
            doubled_period
        ),
        "vsxPeriodDays": (
            vsx_period
        ),
        "doubledVsVSXErrorPercent": (
            harmonic_error_percent
        ),
        "doubledVsVSXPhaseDriftAcrossBaselineCycles": (
            phase_drift_cycles
        ),
        "openstarHalfWithinBinRMS": (
            half_rms
        ),
        "doubledOpenstarWithinBinRMS": (
            doubled_rms
        ),
        "vsxWithinBinRMS": (
            vsx_metrics[
                "withinBinRMS"
            ]
        ),
        "fullPeriodCoherenceGainPercent": (
            full_period_coherence_gain
        ),
        "oddEvenHalfCycle": (
            odd_even
        ),
        "doubledOpenstarTwoMinima": (
            doubled_metrics[
                "twoMinima"
            ]
        ),
        "vsxTwoMinima": (
            vsx_metrics[
                "twoMinima"
            ]
        ),
        "plots": (
            plot_paths
        ),
    }

    print()
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        f"⭐ {blind_name} — "
        f"{variable_type}"
    )
    print(
        "════════════════════════════════════════════════════════"
    )
    print(
        "   OpenStar: "
        f"{openstar_period:.8f} d"
    )
    print(
        "   2 × OpenStar: "
        f"{doubled_period:.8f} d"
    )
    print(
        "   VSX: "
        f"{vsx_period:.8f} d"
    )
    print(
        "   2×OpenStar vs VSX error: "
        f"{harmonic_error_percent:.4f}%"
    )
    print(
        "   phase drift across TESS baseline: "
        f"{phase_drift_cycles:.4f} cycles"
    )
    print()
    print(
        "   fold within-bin RMS"
    )
    print(
        "      OpenStar: "
        f"{format_optional(half_rms)}"
    )
    print(
        "      2×OpenStar: "
        f"{format_optional(doubled_rms)}"
    )
    print(
        "      VSX: "
        f"{format_optional(vsx_metrics['withinBinRMS'])}"
    )
    print(
        "      full-period coherence gain: "
        f"{format_optional(full_period_coherence_gain, 3)}%"
    )
    print()
    print(
        "   alternating OpenStar half-cycles"
    )
    print(
        "      comparable phase bins: "
        f"{odd_even['comparableBins']}"
    )
    print(
        "      RMS profile difference: "
        f"{format_optional(odd_even['rmsProfileDifference'])}"
    )
    print(
        "      max profile difference: "
        f"{format_optional(odd_even['maxAbsProfileDifference'])}"
    )

    two_minima = result[
        "doubledOpenstarTwoMinima"
    ]

    if two_minima is not None:
        print()
        print(
            "   2×OpenStar two-minimum morphology"
        )
        print(
            "      second-minimum phase separation: "
            f"{two_minima['secondaryPhaseSeparation']:.4f}"
        )
        print(
            "      secondary/primary depth ratio: "
            f"{format_optional(two_minima['secondaryToPrimaryDepthRatio'], 4)}"
        )

    print()
    print(
        "   plots:"
    )

    for path in plot_paths.values():
        print(
            f"      {path}"
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether Blind V2-B/D/E/H OpenStar "
            "periods are first harmonics of the VSX "
            "physical periods."
        )
    )

    parser.add_argument(
        "--reveal",
        type=Path,
        default=REVEAL_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    reveal = load_json(
        args.reveal
    )

    reveal_by_name = {
        item[
            "blindName"
        ]: item
        for item in reveal[
            "targets"
        ]
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "🔬 OpenStar blind harmonic validation"
    )
    print(
        "Targets: B, D, E, H"
    )
    print(
        "Tests: fold coherence + alternating half-cycles "
        "+ two-minimum morphology"
    )

    results = []

    for blind_name in TARGETS:
        reveal_target = (
            reveal_by_name.get(
                blind_name
            )
        )

        if reveal_target is None:
            raise RuntimeError(
                f"Reveal JSON is missing "
                f"{blind_name}."
            )

        if (
            reveal_target.get(
                "status"
            )
            != "REVEALED"
        ):
            raise RuntimeError(
                f"{blind_name} was not "
                "successfully revealed."
            )

        dataset_path = (
            DATASET_PATHS[
                blind_name
            ]
        )

        if not dataset_path.exists():
            raise RuntimeError(
                f"Missing dataset: "
                f"{dataset_path}"
            )

        results.append(
            analyze_target(
                reveal_target,
                dataset_path,
                args.output_dir,
            )
        )

    json_path = (
        args.output_dir
        / "harmonic-validation-summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "projectID": (
                    reveal.get(
                        "projectID"
                    )
                ),
                "analysis": (
                    "first-harmonic phase-fold validation"
                ),
                "targets": (
                    results
                ),
            },
            file,
            indent=2,
            allow_nan=False,
        )

    csv_path = (
        args.output_dir
        / "harmonic-validation-summary.csv"
    )

    fieldnames = (
        "blindName",
        "ticID",
        "vsxType",
        "openstarPeriodDays",
        "doubledOpenstarPeriodDays",
        "vsxPeriodDays",
        "doubledVsVSXErrorPercent",
        "doubledVsVSXPhaseDriftAcrossBaselineCycles",
        "openstarHalfWithinBinRMS",
        "doubledOpenstarWithinBinRMS",
        "vsxWithinBinRMS",
        "fullPeriodCoherenceGainPercent",
        "oddEvenRMSProfileDifference",
        "oddEvenMaxProfileDifference",
        "secondMinimumPhaseSeparation",
        "secondaryToPrimaryDepthRatio",
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                fieldnames
            ),
        )

        writer.writeheader()

        for item in results:
            odd_even = (
                item[
                    "oddEvenHalfCycle"
                ]
            )

            minima = (
                item[
                    "doubledOpenstarTwoMinima"
                ]
                or {}
            )

            writer.writerow(
                {
                    "blindName": (
                        item[
                            "blindName"
                        ]
                    ),
                    "ticID": (
                        item[
                            "ticID"
                        ]
                    ),
                    "vsxType": (
                        item[
                            "vsxType"
                        ]
                    ),
                    "openstarPeriodDays": (
                        item[
                            "openstarPeriodDays"
                        ]
                    ),
                    "doubledOpenstarPeriodDays": (
                        item[
                            "doubledOpenstarPeriodDays"
                        ]
                    ),
                    "vsxPeriodDays": (
                        item[
                            "vsxPeriodDays"
                        ]
                    ),
                    "doubledVsVSXErrorPercent": (
                        item[
                            "doubledVsVSXErrorPercent"
                        ]
                    ),
                    "doubledVsVSXPhaseDriftAcrossBaselineCycles": (
                        item[
                            "doubledVsVSXPhaseDriftAcrossBaselineCycles"
                        ]
                    ),
                    "openstarHalfWithinBinRMS": (
                        item[
                            "openstarHalfWithinBinRMS"
                        ]
                    ),
                    "doubledOpenstarWithinBinRMS": (
                        item[
                            "doubledOpenstarWithinBinRMS"
                        ]
                    ),
                    "vsxWithinBinRMS": (
                        item[
                            "vsxWithinBinRMS"
                        ]
                    ),
                    "fullPeriodCoherenceGainPercent": (
                        item[
                            "fullPeriodCoherenceGainPercent"
                        ]
                    ),
                    "oddEvenRMSProfileDifference": (
                        odd_even[
                            "rmsProfileDifference"
                        ]
                    ),
                    "oddEvenMaxProfileDifference": (
                        odd_even[
                            "maxAbsProfileDifference"
                        ]
                    ),
                    "secondMinimumPhaseSeparation": (
                        minima.get(
                            "secondaryPhaseSeparation"
                        )
                    ),
                    "secondaryToPrimaryDepthRatio": (
                        minima.get(
                            "secondaryToPrimaryDepthRatio"
                        )
                    ),
                }
            )

    print()
    print()
    print(
        "🏁 Harmonic validation complete"
    )
    print(
        f"   JSON: {json_path}"
    )
    print(
        f"   CSV:  {csv_path}"
    )
    print(
        f"   plots: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
