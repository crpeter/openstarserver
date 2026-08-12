from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .tess_residual_localization import (
    _float,
    _int,
    _safe,
    _write_json,
)
from .tess_noirlab_forced_photometry import (
    _angular_separation_arcsec,
    _source_records_by_role,
)

PLAN_VERSION = "openstar.tess-targeted-observation-plan.v1"

MINIMUM_BASELINE_FLOOR_DAYS = 45.0
PREFERRED_BASELINE_FLOOR_DAYS = 90.0
MINIMUM_LONGEST_PERIOD_CYCLES = 10.0
PREFERRED_LONGEST_PERIOD_CYCLES = 18.0

MINIMUM_NIGHTS_FLOOR = 24
PREFERRED_NIGHTS_FLOOR = 40
MINIMUM_VISITS_PER_NIGHT = 2
PREFERRED_VISIT_SEPARATION_HOURS = (1.0, 3.0)

MINIMUM_FILTERS = 2
PREFERRED_FILTERS = ("r", "i")
ALTERNATE_FILTER_PAIR = ("g", "r")

TARGET_SHORT_MINIMUM_SNR = 100.0
COUNTERPART_DEEP_MINIMUM_SNR = 20.0
COUNTERPART_DEEP_PREFERRED_SNR = 30.0
TARGET_MAXIMUM_LINEAR_FRACTION = 0.70

PREFERRED_FWHM_ARCSEC = 3.0
MINIMUM_SOURCE_SEPARATION_IN_FWHM = 6.0
PREFERRED_PIXEL_SCALE_ARCSEC = 1.0
ABSOLUTE_MAXIMUM_PIXEL_SCALE_ARCSEC = 2.0

MINIMUM_GLOBAL_PEAK_PROMINENCE = 2.0
MAXIMUM_CROSS_FILTER_RELATIVE_FREQUENCY_SPREAD = 0.12
MINIMUM_RECURRENT_WINDOWS = 2
MINIMUM_WINDOW_NIGHTS = 8
WINDOW_LONGEST_PERIOD_CYCLES = 5.0
WINDOW_DAYS_FLOOR = 30.0


def _search_float(search: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if search.get(key) is None:
            continue
        value = _float(search.get(key))
        if value is not None:
            return float(value)
    return None


def _frequency_bounds(search: dict[str, Any]) -> tuple[float, float]:
    minimum = _search_float(
        search,
        "minimumFrequency",
        "minFrequency",
        "startFrequency",
    )
    maximum = _search_float(
        search,
        "maximumFrequency",
        "maxFrequency",
        "endFrequency",
    )

    total = _int(
        search.get("totalFrequencies")
        if search.get("totalFrequencies") is not None
        else search.get("frequencyCount")
    )
    step = _search_float(
        search,
        "frequencyStep",
        "step",
    )

    if maximum is None and minimum is not None and total is not None and step is not None:
        maximum = minimum + max(total - 1, 0) * step

    if (
        minimum is None
        or maximum is None
        or not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0
        or maximum <= minimum
    ):
        raise RuntimeError(
            "v20.28 requires a valid frozen residual-frequency search range."
        )

    return float(minimum), float(maximum)


def _frozen_sources(
    external_high_resolution_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    records = _source_records_by_role(external_high_resolution_summary)

    sources: list[dict[str, Any]] = []
    for role in ("target-control", "catalog-counterpart"):
        record = records.get(role) or {}
        metadata = record.get("metadata") or {}
        source_id = _int(record.get("gaiaDR3SourceID"))
        ra = _float(metadata.get("raDeg"))
        dec = _float(metadata.get("decDeg"))

        if source_id is None or ra is None or dec is None:
            raise RuntimeError(
                f"v20.28 requires frozen Gaia identity and coordinates for {role}."
            )

        sources.append(
            {
                "sourceRole": role,
                "gaiaDR3SourceID": int(source_id),
                "raDeg": float(ra),
                "decDeg": float(dec),
                "gMag": _float(metadata.get("gMag")),
                "bpMag": _float(metadata.get("bpMag")),
                "rpMag": _float(metadata.get("rpMag")),
            }
        )

    if sources[0]["gaiaDR3SourceID"] == sources[1]["gaiaDR3SourceID"]:
        raise RuntimeError(
            "v20.28 target and counterpart must remain distinct frozen Gaia sources."
        )

    return sources


def _source_geometry(
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    target = next(
        item for item in sources
        if item["sourceRole"] == "target-control"
    )
    counterpart = next(
        item for item in sources
        if item["sourceRole"] == "catalog-counterpart"
    )

    separation = _angular_separation_arcsec(
        float(target["raDeg"]),
        float(target["decDeg"]),
        float(counterpart["raDeg"]),
        float(counterpart["decDeg"]),
    )

    hard_fwhm = float(
        separation / MINIMUM_SOURCE_SEPARATION_IN_FWHM
    )
    maximum_fwhm = min(
        hard_fwhm,
        5.0,
    )

    maximum_pixel_scale = min(
        ABSOLUTE_MAXIMUM_PIXEL_SCALE_ARCSEC,
        separation / 12.0,
    )

    return {
        "target": target,
        "counterpart": counterpart,
        "separationArcsec": float(separation),
        "preferredFwhmArcsec": PREFERRED_FWHM_ARCSEC,
        "maximumFwhmArcsec": float(maximum_fwhm),
        "preferredPixelScaleArcsec": PREFERRED_PIXEL_SCALE_ARCSEC,
        "maximumPixelScaleArcsec": float(maximum_pixel_scale),
    }


def _cadence_plan(
    minimum_frequency: float,
    maximum_frequency: float,
) -> dict[str, Any]:
    longest_period = 1.0 / minimum_frequency
    shortest_period = 1.0 / maximum_frequency

    minimum_baseline = max(
        MINIMUM_BASELINE_FLOOR_DAYS,
        MINIMUM_LONGEST_PERIOD_CYCLES * longest_period,
    )
    preferred_baseline = max(
        PREFERRED_BASELINE_FLOOR_DAYS,
        PREFERRED_LONGEST_PERIOD_CYCLES * longest_period,
    )

    minimum_nights = max(
        MINIMUM_NIGHTS_FLOOR,
        int(math.ceil(minimum_baseline / 2.0)),
    )
    preferred_nights = max(
        PREFERRED_NIGHTS_FLOOR,
        int(math.ceil(preferred_baseline / 2.0)),
    )

    window_days = max(
        WINDOW_DAYS_FLOOR,
        WINDOW_LONGEST_PERIOD_CYCLES * longest_period,
    )

    return {
        "frozenFrequencyRangeCyclesPerDay": {
            "minimum": float(minimum_frequency),
            "maximum": float(maximum_frequency),
        },
        "frozenPeriodRangeDays": {
            "minimum": float(shortest_period),
            "maximum": float(longest_period),
        },
        "minimumBaselineDays": float(minimum_baseline),
        "preferredBaselineDays": float(preferred_baseline),
        "minimumDistinctNights": int(minimum_nights),
        "preferredDistinctNights": int(preferred_nights),
        "minimumVisitsPerObservedNight": MINIMUM_VISITS_PER_NIGHT,
        "preferredWithinNightVisitSeparationHours": {
            "minimum": PREFERRED_VISIT_SEPARATION_HOURS[0],
            "maximum": PREFERRED_VISIT_SEPARATION_HOURS[1],
        },
        "cadenceAdvice": (
            "Vary the nightly observation time when practical rather than "
            "sampling at the same sidereal/local time every night. This reduces "
            "daily-window aliases without using the expected signal frequency "
            "to schedule individual observations."
        ),
        "timeResolvedAnalysis": {
            "enabled": True,
            "fixedWindowDays": float(window_days),
            "windowAnchor": "first-qualified-campaign-observation",
            "minimumQualifiedNightsPerWindow": MINIMUM_WINDOW_NIGHTS,
            "minimumAcceptedRecurrentWindows": MINIMUM_RECURRENT_WINDOWS,
            "windowBoundariesDependOnMeasuredPeriodogram": False,
        },
    }


def _exposure_plan(
    geometry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pairedExposureTiersRequired": True,
        "targetShortTier": {
            "purpose": (
                "Keep Blind C in the detector/photometry linear regime while "
                "obtaining high-S/N source-resolved target photometry."
            ),
            "minimumSNR": TARGET_SHORT_MINIMUM_SNR,
            "maximumPeakFractionOfDocumentedLinearOrSaturationLevel": (
                TARGET_MAXIMUM_LINEAR_FRACTION
            ),
            "saturationAllowed": False,
        },
        "counterpartDeepTier": {
            "purpose": (
                "Reach useful precision on the frozen Gaia counterpart. Blind C "
                "may saturate in this tier only if its saturation/bleed/halo does "
                "not contaminate the counterpart measurement region."
            ),
            "minimumSNR": COUNTERPART_DEEP_MINIMUM_SNR,
            "preferredSNR": COUNTERPART_DEEP_PREFERRED_SNR,
            "targetMaySaturate": True,
            "counterpartMaySaturate": False,
            "hardContaminationRule": (
                "Reject any deep exposure when target saturation, bleed, "
                "persistence, ghosts, or scattered-light structure reaches the "
                "counterpart photometry/PSF-fitting region."
            ),
        },
        "sameVisitRequirement": (
            "A visit should contain both short-tier and deep-tier imaging close "
            "enough in time to represent the same variability state."
        ),
        "geometryGuard": {
            "gaiaSourceSeparationArcsec": geometry["separationArcsec"],
            "preferredFwhmArcsec": geometry["preferredFwhmArcsec"],
            "maximumFwhmArcsec": geometry["maximumFwhmArcsec"],
            "preferredPixelScaleArcsec": (
                geometry["preferredPixelScaleArcsec"]
            ),
            "maximumPixelScaleArcsec": (
                geometry["maximumPixelScaleArcsec"]
            ),
        },
    }


def _filter_plan() -> dict[str, Any]:
    return {
        "minimumFilters": MINIMUM_FILTERS,
        "preferredFilters": list(PREFERRED_FILTERS),
        "acceptableAlternatePair": list(ALTERNATE_FILTER_PAIR),
        "requireStandardOrWellCharacterizedPassbands": True,
        "sameFilterSetThroughoutCampaignPreferred": True,
        "reason": (
            "Cross-filter recurrence is part of the preregistered attribution "
            "guard. A red optical filter is preferred because it helps manage "
            "the bright target while retaining counterpart sensitivity."
        ),
    }


def _calibration_plan() -> dict[str, Any]:
    return {
        "preferredDataProduct": (
            "Calibrated/reduced FITS images for every short/deep exposure, "
            "retaining both frozen Gaia sources and several non-variable "
            "comparison stars in the field."
        ),
        "photometricMethod": (
            "Use source-resolved PSF or aperture photometry with the same "
            "comparison ensemble for the target and counterpart when possible."
        ),
        "requiredImageCalibrations": [
            "bias/overscan treatment as appropriate",
            "dark correction when material for the detector/exposure",
            "flat-field correction",
            "bad-pixel/cosmetic masking",
            "time metadata",
            "filter metadata",
        ],
        "preferredTiming": "BJD_TDB",
        "timingFallback": (
            "UTC mid-exposure time plus observatory longitude, latitude, and "
            "elevation sufficient for OpenStar to compute barycentric time."
        ),
        "comparisonStarGuard": (
            "Do not select comparison stars because they make the candidate "
            "period stronger. Freeze the comparison ensemble using brightness, "
            "color, isolation, detector linearity, and empirical stability "
            "criteria before examining the residual-band periodogram."
        ),
    }


def _analysis_contract() -> dict[str, Any]:
    return {
        "claimLevelAutomaticallyChanged": False,
        "frozenSearchBandOnly": True,
        "globalAcceptance": {
            "periodStatusRequired": "RELIABLE",
            "minimumIndependentPeakProminenceRatio": (
                MINIMUM_GLOBAL_PEAK_PROMINENCE
            ),
            "boundaryHitAllowed": False,
            "coverageCompleteRequiredWhenReported": True,
        },
        "crossFilterAcceptance": {
            "minimumAcceptedFilters": 2,
            "maximumRelativeFrequencySpread": (
                MAXIMUM_CROSS_FILTER_RELATIVE_FREQUENCY_SPREAD
            ),
        },
        "timeResolvedAcceptance": {
            "minimumAcceptedRecurrentWindows": (
                MINIMUM_RECURRENT_WINDOWS
            ),
            "minimumQualifiedNightsPerWindow": (
                MINIMUM_WINDOW_NIGHTS
            ),
            "sameProminenceThresholdAsGlobal": True,
            "tessDriftLawUsedToChooseWindowFrequency": False,
        },
        "sourceAttributionDecision": {
            "counterpartOnly": {
                "classification": (
                    "TARGETED_PHOTOMETRY_COUNTERPART_VARIABILITY_SUPPORTED"
                ),
                "recommendedNextTest": (
                    "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
                ),
            },
            "targetOnly": {
                "classification": (
                    "TARGETED_PHOTOMETRY_TARGET_VARIABILITY_SUPPORTED"
                ),
                "recommendedNextTest": (
                    "TARGET_INTRINSIC_RESIDUAL_MODELING"
                ),
            },
            "both": {
                "classification": (
                    "TARGETED_PHOTOMETRY_TARGET_AND_COUNTERPART_VARIABILITY_SUPPORTED"
                ),
                "recommendedNextTest": (
                    "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
                ),
            },
            "neither": {
                "classification": (
                    "TARGETED_PHOTOMETRY_RESIDUAL_SOURCE_NOT_CONFIRMED"
                ),
                "recommendedNextTest": (
                    "REASSESS_RESIDUAL_SOURCE_MODEL"
                ),
            },
        },
        "importantGuard": (
            "The established main periodic family remains a separate target-"
            "associated result. This experiment is designed specifically to "
            "resolve the drifting residual component and must not overwrite "
            "the main-family source association."
        ),
    }


def _ingest_columns() -> list[str]:
    return [
        "exposure_id",
        "visit_id",
        "time_bjd_tdb",
        "time_utc_mid",
        "observatory_code",
        "source_role",
        "gaia_dr3_source_id",
        "filter",
        "exposure_tier",
        "exposure_seconds",
        "flux",
        "flux_error",
        "differential_mag",
        "differential_mag_error",
        "fwhm_arcsec",
        "airmass",
        "saturated",
        "contaminated",
        "quality_flag",
        "fits_path",
    ]


def _write_csv_template(
    path: Path,
    sources: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_ingest_columns(),
        )
        writer.writeheader()

        for source in sources:
            writer.writerow(
                {
                    "source_role": source["sourceRole"],
                    "gaia_dr3_source_id": (
                        source["gaiaDR3SourceID"]
                    ),
                    "quality_flag": "TEMPLATE_ROW_DELETE_ME",
                }
            )


def _markdown_plan(
    plan: dict[str, Any],
) -> str:
    geometry = plan["sourceGeometry"]
    cadence = plan["cadence"]
    exposure = plan["exposureStrategy"]
    filters = plan["filterStrategy"]
    analysis = plan["analysisContract"]

    target = geometry["target"]
    counterpart = geometry["counterpart"]

    lines = [
        "# OpenStar targeted high-resolution time-series photometry plan",
        "",
        f"- Plan version: {plan['version']}",
        f"- Investigation: {plan['investigationID']}",
        f"- Source project: {plan['sourceProjectID']}",
        f"- Source dataset: {plan['sourceDatasetID']}",
        "",
        "## Frozen source pair",
        "",
        (
            f"- Blind C / target-control: Gaia DR3 "
            f"{target['gaiaDR3SourceID']} at "
            f"RA={target['raDeg']} deg, Dec={target['decDeg']} deg"
        ),
        (
            f"- Catalog counterpart: Gaia DR3 "
            f"{counterpart['gaiaDR3SourceID']} at "
            f"RA={counterpart['raDeg']} deg, "
            f"Dec={counterpart['decDeg']} deg"
        ),
        (
            f"- Gaia-to-Gaia separation: "
            f"{geometry['separationArcsec']:.6f} arcsec"
        ),
        "",
        "## Scientific question",
        "",
        (
            "Which frozen Gaia source produces the drifting TESS residual "
            "component: Blind C, the catalog counterpart, both, or neither?"
        ),
        "",
        "The established main photometric family remains separately associated "
        "with Blind C and is not being re-litigated by this experiment.",
        "",
        "## Frequency / period test frozen before observations",
        "",
        (
            f"- Frequency range: "
            f"{cadence['frozenFrequencyRangeCyclesPerDay']['minimum']:.9f} "
            f"to "
            f"{cadence['frozenFrequencyRangeCyclesPerDay']['maximum']:.9f} "
            "cycles/day"
        ),
        (
            f"- Corresponding period range: "
            f"{cadence['frozenPeriodRangeDays']['minimum']:.6f} "
            f"to {cadence['frozenPeriodRangeDays']['maximum']:.6f} days"
        ),
        "- The TESS drift law will not be extrapolated to choose the new-data frequency.",
        "",
        "## Campaign cadence",
        "",
        (
            f"- Minimum baseline: "
            f"{cadence['minimumBaselineDays']:.1f} days"
        ),
        (
            f"- Preferred baseline: "
            f"{cadence['preferredBaselineDays']:.1f} days"
        ),
        (
            f"- Minimum distinct nights: "
            f"{cadence['minimumDistinctNights']}"
        ),
        (
            f"- Preferred distinct nights: "
            f"{cadence['preferredDistinctNights']}"
        ),
        (
            f"- Minimum visits per observed night: "
            f"{cadence['minimumVisitsPerObservedNight']}"
        ),
        (
            "- Preferred visit separation within a night: "
            f"{cadence['preferredWithinNightVisitSeparationHours']['minimum']:.1f}"
            "–"
            f"{cadence['preferredWithinNightVisitSeparationHours']['maximum']:.1f} h"
        ),
        (
            f"- Time-resolved fixed windows: "
            f"{cadence['timeResolvedAnalysis']['fixedWindowDays']:.1f} days"
        ),
        "",
        "## Image quality / source resolution",
        "",
        (
            f"- Preferred FWHM: <= "
            f"{geometry['preferredFwhmArcsec']:.2f} arcsec"
        ),
        (
            f"- Hard maximum FWHM for source attribution: <= "
            f"{geometry['maximumFwhmArcsec']:.2f} arcsec"
        ),
        (
            f"- Preferred pixel scale: <= "
            f"{geometry['preferredPixelScaleArcsec']:.2f} arcsec/pixel"
        ),
        (
            f"- Hard maximum pixel scale: <= "
            f"{geometry['maximumPixelScaleArcsec']:.2f} arcsec/pixel"
        ),
        "",
        "## Exposure strategy",
        "",
        "- Every visit should contain paired short + deep exposures.",
        (
            f"- Short tier: keep Blind C unsaturated/linear; "
            f"target S/N >= {exposure['targetShortTier']['minimumSNR']:.0f}; "
            f"peak <= "
            f"{exposure['targetShortTier']['maximumPeakFractionOfDocumentedLinearOrSaturationLevel']:.2f} "
            "of the documented detector linear/saturation level."
        ),
        (
            f"- Deep tier: counterpart S/N >= "
            f"{exposure['counterpartDeepTier']['minimumSNR']:.0f}, "
            f"preferred >= "
            f"{exposure['counterpartDeepTier']['preferredSNR']:.0f}."
        ),
        (
            "- Blind C may saturate in a deep frame only when its saturation "
            "structure does not contaminate the counterpart measurement."
        ),
        "",
        "## Filters",
        "",
        (
            f"- Minimum filters: {filters['minimumFilters']}"
        ),
        (
            f"- Preferred: {', '.join(filters['preferredFilters'])}"
        ),
        (
            f"- Acceptable alternate pair: "
            f"{', '.join(filters['acceptableAlternatePair'])}"
        ),
        "",
        "## Pre-registered acceptance",
        "",
        (
            "- Each accepted periodogram must be RELIABLE, not a boundary hit, "
            f"and have independent-peak prominence >= "
            f"{analysis['globalAcceptance']['minimumIndependentPeakProminenceRatio']:.1f}."
        ),
        (
            "- Strong source support requires at least two accepted filters "
            f"with relative frequency spread <= "
            f"{analysis['crossFilterAcceptance']['maximumRelativeFrequencySpread']:.2f}."
        ),
        (
            "- Because the residual is nonstationary, source support must also "
            f"recur in at least "
            f"{analysis['timeResolvedAcceptance']['minimumAcceptedRecurrentWindows']} "
            "predefined non-overlapping campaign windows."
        ),
        "",
        "## Deliverables for OpenStar ingest",
        "",
        "- Preferred: calibrated/reduced FITS images for every short/deep exposure.",
        "- Also provide the generated CSV schema when source-resolved photometry is exported.",
        "- Preserve rejected/flagged measurements; do not remove them silently.",
        "",
        "## Stop condition",
        "",
        (
            "Do not keep changing cadence windows, prominence thresholds, or "
            "filter rules after seeing the new periodograms. If this pre-"
            "registered campaign does not resolve the source, OpenStar should "
            "reassess the residual-source model rather than tune acceptance "
            "until one source wins."
        ),
        "",
    ]

    return "\n".join(lines)


def build_targeted_observation_plan(
    *,
    source_project_id: str,
    source_dataset_id: str,
    investigation_id: str,
    external_high_resolution_summary: dict[str, Any],
    atlas_fixed_window_summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if atlas_fixed_window_summary.get("recommendedNextTest") != (
        "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    ):
        raise RuntimeError(
            "v20.28 requires v20.27 to recommend targeted high-resolution "
            "time-series photometry."
        )

    search = dict(
        (
            (atlas_fixed_window_summary.get("distributedValidation") or {})
            .get("frequencySearch")
            or {}
        )
    )
    minimum_frequency, maximum_frequency = _frequency_bounds(search)

    sources = _frozen_sources(external_high_resolution_summary)
    geometry = _source_geometry(sources)
    cadence = _cadence_plan(
        minimum_frequency,
        maximum_frequency,
    )

    plan = {
        "version": PLAN_VERSION,
        "investigationID": investigation_id,
        "sourceProjectID": source_project_id,
        "sourceDatasetID": source_dataset_id,
        "status": "OBSERVATION_PLAN_READY",
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "scientificObjective": (
            "Resolve which frozen Gaia source produces the drifting residual "
            "TESS component using new source-resolved time-series photometry."
        ),
        "mainFamilyPolicy": (
            "The established main photometric family remains target-associated. "
            "The new campaign tests only the residual source attribution."
        ),
        "sourceGeometry": geometry,
        "cadence": cadence,
        "filterStrategy": _filter_plan(),
        "exposureStrategy": _exposure_plan(geometry),
        "calibrationStrategy": _calibration_plan(),
        "analysisContract": _analysis_contract(),
        "ingestContract": {
            "format": "one row per source per exposure",
            "columns": _ingest_columns(),
            "preferredRawProduct": (
                "calibrated/reduced FITS frames plus the CSV metadata/photometry table"
            ),
            "requiredSourceRoles": [
                "target-control",
                "catalog-counterpart",
            ],
            "requiredGaiaDR3SourceIDs": [
                int(item["gaiaDR3SourceID"])
                for item in sources
            ],
            "retainRejectedMeasurements": True,
        },
        "provenance": {
            "priorStage": "v20.27 ATLAS fixed-window recurrence",
            "priorClassification": (
                atlas_fixed_window_summary.get("classification")
            ),
            "priorRecommendedNextTest": (
                atlas_fixed_window_summary.get("recommendedNextTest")
            ),
            "frozenFrequencySearch": search,
            "archiveBranchConsideredExhaustedForThisQuestion": True,
        },
        "recommendedNextTest": "COLLECT_TARGETED_TIME_SERIES_PHOTOMETRY",
    }

    root = Path(output_dir) / "targeted-observation-plan"
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / "targeted-observation-plan-v20.28.json"
    markdown_path = root / "targeted-observation-plan-v20.28.md"
    csv_path = root / "targeted-observation-ingest-template-v20.28.csv"

    _write_json(json_path, plan)
    markdown_path.write_text(
        _markdown_plan(plan),
        encoding="utf-8",
    )
    _write_csv_template(
        csv_path,
        sources,
    )

    result = dict(plan)
    result["artifacts"] = {
        "jsonPlanPath": str(json_path.resolve()),
        "markdownPlanPath": str(markdown_path.resolve()),
        "csvIngestTemplatePath": str(csv_path.resolve()),
    }
    return result
