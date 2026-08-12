from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    _float,
    _int,
    _safe,
    _write_json,
)
from .tess_atlas_forced_photometry import (
    MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD,
    MIN_PEAK_PROMINENCE_RATIO,
    _parse_atlas_output,
)
from .tess_atlas_forced_reanalysis import (
    _clean_signed_forced_rows,
)
from .tess_atlas_time_resolved import (
    _nightly_absolute_rows,
)

FIXED_WINDOW_DAYS = 180.0
MIN_WINDOW_BAND_NIGHTS = 12
MIN_WINDOW_BAND_BASELINE_DAYS = 30.0

MIN_SUPPORTED_WINDOWS = 2
MIN_SUPPORTED_WINDOW_BAND_RESULTS = 3
MIN_SUGGESTIVE_WINDOWS = 2


def _fixed_window_index(mjd: float) -> int:
    """
    Deterministic absolute-MJD window index.

    MJD zero is the anchor. No data-dependent epoch, signal, period,
    cadence gap, or previously measured ATLAS frequency affects a boundary.
    """
    return int(math.floor(float(mjd) / FIXED_WINDOW_DAYS))


def _fixed_windows(
    nightly_by_band: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    indices = sorted(
        {
            _fixed_window_index(float(item["mjd"]))
            for rows in nightly_by_band.values()
            for item in rows
        }
    )

    return [
        {
            "windowIndex": int(index),
            "startMJD": float(index * FIXED_WINDOW_DAYS),
            "endMJD": float((index + 1) * FIXED_WINDOW_DAYS),
            "windowDays": FIXED_WINDOW_DAYS,
        }
        for index in indices
    ]


def _rows_for_window(
    rows: list[dict[str, Any]],
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    start = float(window["startMJD"])
    end = float(window["endMJD"])

    return [
        item
        for item in rows
        if start <= float(item["mjd"]) < end
    ]


def _standardize_flux(fluxes: np.ndarray) -> np.ndarray:
    values = np.asarray(fluxes, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad

    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))

    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            "ATLAS fixed-window flux series has no finite variability scale."
        )

    return (values - center) / scale


def _prepare_window_band(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(rows) < MIN_WINDOW_BAND_NIGHTS:
        return None

    times = np.asarray(
        [float(item["mjd"]) for item in rows],
        dtype=np.float64,
    )
    fluxes = np.asarray(
        [float(item["uJy"]) for item in rows],
        dtype=np.float64,
    )
    errors = np.asarray(
        [float(item["duJy"]) for item in rows],
        dtype=np.float64,
    )

    order = np.argsort(times)
    times = times[order]
    fluxes = fluxes[order]
    errors = errors[order]

    baseline = float(times[-1] - times[0])
    if baseline < MIN_WINDOW_BAND_BASELINE_DAYS:
        return None

    standardized = _standardize_flux(fluxes)
    local_times = times - float(times[0])

    return {
        "times": local_times,
        "flux": standardized,
        "sampleCount": int(len(local_times)),
        "baselineDays": baseline,
        "startMJD": float(times[0]),
        "endMJD": float(times[-1]),
        "midMJD": float((times[0] + times[-1]) / 2.0),
        "medianFluxUJy": float(np.median(fluxes)),
        "medianFluxErrorUJy": float(np.median(errors)),
        "medianAbsoluteNightlySNR": float(
            np.median(np.abs(fluxes / errors))
        ),
    }


def _dataset_result(
    project_dataset: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    frequency = _float(
        project_dataset.get("candidateFrequency")
        if project_dataset.get("candidateFrequency") is not None
        else project_dataset.get("bestFrequency")
    )
    period = _float(
        project_dataset.get("candidatePeriodDays")
        if project_dataset.get("candidatePeriodDays") is not None
        else project_dataset.get("bestPeriodDays")
    )
    power = _float(
        project_dataset.get("candidatePower")
        if project_dataset.get("candidatePower") is not None
        else project_dataset.get("bestPower")
    )
    prominence = _float(
        project_dataset.get("candidatePeakProminenceRatio")
    )
    status = str(project_dataset.get("periodStatus") or "")
    coverage = project_dataset.get("coverageComplete")
    boundary_hit = bool(project_dataset.get("boundaryHit") or False)

    accepted = bool(
        status == "RELIABLE"
        and (coverage is None or bool(coverage))
        and not boundary_hit
        and prominence is not None
        and prominence >= MIN_PEAK_PROMINENCE_RATIO
        and frequency is not None
        and frequency > 0
    )

    return {
        "datasetID": prepared.get("datasetID"),
        "sourceRole": "catalog-counterpart",
        "gaiaDR3SourceID": prepared.get("gaiaDR3SourceID"),
        "windowIndex": prepared.get("windowIndex"),
        "windowStartMJD": prepared.get("windowStartMJD"),
        "windowEndMJD": prepared.get("windowEndMJD"),
        "band": prepared.get("band"),
        "sampleCount": prepared.get("sampleCount"),
        "baselineDays": prepared.get("baselineDays"),
        "startMJD": prepared.get("startMJD"),
        "endMJD": prepared.get("endMJD"),
        "midMJD": prepared.get("midMJD"),
        "periodStatus": status or None,
        "periodConfidence": project_dataset.get("periodConfidence"),
        "boundaryHit": boundary_hit,
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "acceptedWindowResidualBandVariability": accepted,
    }


def _relative_frequency_difference(
    a: float,
    b: float,
) -> float:
    center = (float(a) + float(b)) / 2.0
    if center <= 0:
        return float("inf")
    return abs(float(a) - float(b)) / center


def _counterpart_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [
        item
        for item in results
        if item.get("acceptedWindowResidualBandVariability")
    ]

    accepted_windows = sorted(
        {
            int(item["windowIndex"])
            for item in accepted
            if item.get("windowIndex") is not None
        }
    )
    accepted_bands = sorted(
        {
            str(item["band"])
            for item in accepted
            if item.get("band")
        }
    )

    by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in accepted:
        if item.get("windowIndex") is not None:
            by_window[int(item["windowIndex"])].append(item)

    cross_band_consistent_windows: list[dict[str, Any]] = []

    for window_index, items in sorted(by_window.items()):
        by_band = {
            str(item["band"]): item
            for item in items
            if item.get("band")
            and item.get("candidateFrequency") is not None
        }

        c_item = by_band.get("c")
        o_item = by_band.get("o")

        if c_item is None or o_item is None:
            continue

        relative_difference = _relative_frequency_difference(
            float(c_item["candidateFrequency"]),
            float(o_item["candidateFrequency"]),
        )

        if (
            relative_difference
            <= MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD
        ):
            cross_band_consistent_windows.append(
                {
                    "windowIndex": int(window_index),
                    "windowStartMJD": c_item.get("windowStartMJD"),
                    "windowEndMJD": c_item.get("windowEndMJD"),
                    "cFrequency": float(
                        c_item["candidateFrequency"]
                    ),
                    "oFrequency": float(
                        o_item["candidateFrequency"]
                    ),
                    "relativeFrequencyDifference": float(
                        relative_difference
                    ),
                }
            )

    supported = bool(
        len(accepted_windows) >= MIN_SUPPORTED_WINDOWS
        and len(accepted) >= MIN_SUPPORTED_WINDOW_BAND_RESULTS
        and len(accepted_bands) >= 2
        and len(cross_band_consistent_windows) >= 1
    )

    suggestive = bool(
        not supported
        and len(accepted_windows) >= MIN_SUGGESTIVE_WINDOWS
        and len(accepted) >= 2
    )

    frequency_trend = None
    if len(accepted) >= 3:
        x = np.asarray(
            [float(item["midMJD"]) for item in accepted],
            dtype=np.float64,
        )
        y = np.asarray(
            [float(item["candidateFrequency"]) for item in accepted],
            dtype=np.float64,
        )

        if float(np.ptp(x)) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            residuals = y - predicted

            frequency_trend = {
                "slopeCyclesPerDayPerDay": float(slope),
                "interceptCyclesPerDay": float(intercept),
                "rmsCyclesPerDay": float(
                    np.sqrt(np.mean(np.square(residuals)))
                ),
                "fitCount": int(len(x)),
                "tessDriftLawUsedInFit": False,
            }

    return {
        "windowBandResults": results,
        "acceptedWindowBandResults": accepted,
        "acceptedWindowBandCount": int(len(accepted)),
        "acceptedWindows": accepted_windows,
        "acceptedWindowCount": int(len(accepted_windows)),
        "acceptedBands": accepted_bands,
        "crossBandConsistentWindows": (
            cross_band_consistent_windows
        ),
        "crossBandConsistentWindowCount": int(
            len(cross_band_consistent_windows)
        ),
        "independentATLASFrequencyTrend": frequency_trend,
        "sourceSupported": supported,
        "sourceSuggestive": suggestive,
    }


def build_atlas_fixed_window_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    atlas_v20_24_summary: dict[str, Any],
    atlas_v20_26_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if atlas_v20_26_summary.get("classification") != (
        "ATLAS_TIME_RESOLVED_COUNTERPART_RECURRENCE_NOT_CONFIRMED"
    ):
        raise RuntimeError(
            "v20.27 is preregistered for the completed v20.26 "
            "single-gap-season branch."
        )

    search = dict(
        (
            (atlas_v20_26_summary.get("distributedValidation") or {})
            .get("frequencySearch")
            or {}
        )
    )
    total_frequencies = _int(search.get("totalFrequencies"))
    per_work = _int(search.get("frequenciesPerWorkUnit"))

    if (
        not search
        or total_frequencies is None
        or per_work is None
        or total_frequencies <= 0
        or per_work <= 0
    ):
        raise RuntimeError(
            "v20.27 requires the unchanged frozen residual-frequency "
            "search definition."
        )

    v20_24_records = atlas_v20_24_summary.get("sourceRecords") or []
    counterpart_record = next(
        (
            item
            for item in v20_24_records
            if item.get("sourceRole") == "catalog-counterpart"
        ),
        None,
    )

    if counterpart_record is None:
        raise RuntimeError(
            "v20.27 cannot find the immutable v20.24 counterpart "
            "ATLAS source record."
        )

    raw_path_text = str(
        counterpart_record.get("rawPath") or ""
    ).strip()
    if not raw_path_text:
        raise RuntimeError(
            "v20.27 counterpart raw ATLAS artifact path is missing."
        )

    raw_path = Path(raw_path_text)
    if not raw_path.exists():
        raise RuntimeError(
            f"v20.27 raw counterpart ATLAS artifact is missing: "
            f"{raw_path}."
        )

    raw_rows = _parse_atlas_output(
        raw_path.read_text(encoding="utf-8")
    )
    clean_rows, signed_quality = _clean_signed_forced_rows(
        raw_rows
    )
    nightly_by_band = _nightly_absolute_rows(clean_rows)
    windows = _fixed_windows(nightly_by_band)

    print(
        f"   immutable counterpart raw rows: {len(raw_rows)}",
        flush=True,
    )
    print(
        "   signed quality-valid rows: "
        f"{signed_quality['acceptedSignedRows']}",
        flush=True,
    )
    print(
        f"   absolute-MJD window size: {FIXED_WINDOW_DAYS:.1f} days",
        flush=True,
    )
    print(
        f"   fixed windows intersecting data: {len(windows)}",
        flush=True,
    )

    root = Path(output_dir) / "atlas-fixed-windows"
    root.mkdir(parents=True, exist_ok=True)

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    window_diagnostics: list[dict[str, Any]] = []

    for window in windows:
        window_index = int(window["windowIndex"])
        diag = {
            "windowIndex": window_index,
            "startMJD": float(window["startMJD"]),
            "endMJD": float(window["endMJD"]),
            "bands": {},
        }

        for band in ("c", "o"):
            rows = _rows_for_window(
                nightly_by_band.get(band, []),
                window,
            )
            series = _prepare_window_band(rows)

            diag["bands"][band] = {
                "nightCount": int(len(rows)),
                "prepared": bool(series is not None),
                "observedBaselineDays": (
                    float(series["baselineDays"])
                    if series is not None
                    else (
                        float(rows[-1]["mjd"] - rows[0]["mjd"])
                        if len(rows) >= 2
                        else 0.0
                    )
                ),
            }

            if series is None:
                continue

            dataset_id = (
                f"{source_dataset_id}-atlas-fixed-window-"
                f"{window_index}-{band}-v1"
            )
            target_name = (
                f"{source_dataset_id} ATLAS counterpart fixed "
                f"window {window_index} {band}-band"
            )
            dataset_path = root / f"{_safe(dataset_id)}.json"

            dataset = {
                "id": dataset_id,
                "targetName": target_name,
                "times": np.asarray(
                    series["times"],
                    dtype=np.float32,
                ).tolist(),
                "flux": np.asarray(
                    series["flux"],
                    dtype=np.float32,
                ).tolist(),
                "frequencySearch": search,
                "reference": {},
                "science": {
                    "role": (
                        "atlas-fixed-window-counterpart-recurrence"
                    ),
                    "sourceRole": "catalog-counterpart",
                    "gaiaDR3SourceID": int(
                        counterpart_record["gaiaDR3SourceID"]
                    ),
                    "windowIndex": window_index,
                    "windowStartMJD": float(
                        window["startMJD"]
                    ),
                    "windowEndMJD": float(
                        window["endMJD"]
                    ),
                    "windowDays": FIXED_WINDOW_DAYS,
                    "windowAnchor": "absolute-mjd-zero",
                    "band": band,
                    "signedForcedFluxRetained": True,
                    "individualDetectionThresholdApplied": False,
                    "nightlyInverseVarianceBinning": True,
                    "tessDriftExtrapolated": False,
                    "tessDriftLawUsedToChooseWindowFrequency": (
                        False
                    ),
                },
                "source": {
                    "mission": "ATLAS",
                    "archive": "ATLAS Forced Photometry",
                    "filter": band,
                    "distributedSamples": int(
                        series["sampleCount"]
                    ),
                    "baselineDays": float(
                        series["baselineDays"]
                    ),
                },
            }
            _write_json(dataset_path, dataset)

            prepared = {
                "datasetID": dataset_id,
                "datasetPath": str(dataset_path.resolve()),
                "sourceRole": "catalog-counterpart",
                "gaiaDR3SourceID": int(
                    counterpart_record["gaiaDR3SourceID"]
                ),
                "windowIndex": window_index,
                "windowStartMJD": float(
                    window["startMJD"]
                ),
                "windowEndMJD": float(
                    window["endMJD"]
                ),
                "band": band,
                "sampleCount": int(series["sampleCount"]),
                "baselineDays": float(
                    series["baselineDays"]
                ),
                "startMJD": float(series["startMJD"]),
                "endMJD": float(series["endMJD"]),
                "midMJD": float(series["midMJD"]),
                "medianFluxUJy": float(
                    series["medianFluxUJy"]
                ),
                "medianFluxErrorUJy": float(
                    series["medianFluxErrorUJy"]
                ),
                "medianAbsoluteNightlySNR": float(
                    series["medianAbsoluteNightlySNR"]
                ),
            }
            prepared_series.append(prepared)
            dataset_entries.append(
                {
                    "id": dataset_id,
                    "path": str(dataset_path.resolve()),
                    "targetName": target_name,
                }
            )

            print(
                f"      window {window_index} {band}: "
                f"{series['sampleCount']} nights | "
                f"baseline={series['baselineDays']:.1f} d",
                flush=True,
            )

        window_diagnostics.append(diag)

    project_id: str | None = None
    project_path: str | None = None

    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation."
            f"{_safe(investigation_id)}."
            "atlas-fixed-window-counterpart-v1"
        )

        manifest = {
            "id": project_id,
            "name": (
                f"{source_project_id} — ATLAS fixed-window "
                "counterpart recurrence"
            ),
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": (
                    "atlas-fixed-window-counterpart-recurrence"
                ),
                "archive": "ATLAS Forced Photometry",
                "workerSemantics": (
                    "The immutable ATLAS counterpart target-image "
                    "forced-photometry series is divided into fixed "
                    "non-overlapping 180-day bins anchored at absolute "
                    "MJD zero. Every qualifying window/filter is searched "
                    "independently by ordinary Lomb-Scargle over the "
                    "unchanged frozen residual-frequency band."
                ),
                "tessDriftExtrapolated": False,
            },
        }

        manifest_path = root / f"{_safe(project_id)}.json"
        _write_json(manifest_path, manifest)
        project_path = str(manifest_path.resolve())

    work_units_per_dataset = math.ceil(
        total_frequencies / per_work
    )

    return {
        "available": bool(dataset_entries),
        "version": (
            "openstar.tess-atlas-fixed-window-preparation.v1"
        ),
        "archive": "ATLAS Forced Photometry",
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": (
            "generic-lomb-scargle-on-atlas-counterpart-fixed-window-filter-series"
        ),
        "sourcePair": atlas_v20_26_summary.get("sourcePair"),
        "gaiaPairSeparationArcsec": (
            atlas_v20_26_summary.get(
                "gaiaPairSeparationArcsec"
            )
        ),
        "counterpartGaiaDR3SourceID": int(
            counterpart_record["gaiaDR3SourceID"]
        ),
        "frequencySearch": search,
        "rawArchiveRequeried": False,
        "rawArtifactsReusedFromV20_24": True,
        "v20_26GapSeasonResult": (
            atlas_v20_26_summary.get("classification")
        ),
        "tessDriftExtrapolated": False,
        "windowDays": FIXED_WINDOW_DAYS,
        "windowAnchor": "absolute-mjd-zero",
        "windows": windows,
        "windowDiagnostics": window_diagnostics,
        "preparedSeries": prepared_series,
        "workUnitsPerDataset": int(
            work_units_per_dataset
        ),
        "totalWorkUnits": int(
            len(dataset_entries) * work_units_per_dataset
        ),
        "acceptanceGuard": {
            "minimumPeakProminenceRatio": (
                MIN_PEAK_PROMINENCE_RATIO
            ),
            "rejectBoundaryHit": True,
            "minimumWindowBandNights": (
                MIN_WINDOW_BAND_NIGHTS
            ),
            "minimumWindowBandBaselineDays": (
                MIN_WINDOW_BAND_BASELINE_DAYS
            ),
            "maximumCrossBandRelativeFrequencySpread": (
                MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD
            ),
            "minimumSupportedWindows": (
                MIN_SUPPORTED_WINDOWS
            ),
            "minimumSupportedWindowBandResults": (
                MIN_SUPPORTED_WINDOW_BAND_RESULTS
            ),
            "minimumSuggestiveWindows": (
                MIN_SUGGESTIVE_WINDOWS
            ),
        },
        "interpretationGuard": (
            "v20.27 is a correction to the failed v20.26 cadence-gap "
            "segmentation, which collapsed the full ATLAS baseline into "
            "one season. It does not lower any signal-acceptance threshold "
            "and does not query the archive again. Window boundaries are "
            "fixed before looking at any periodogram: consecutive "
            "non-overlapping 180-day intervals anchored to absolute MJD "
            "zero. Every window/filter must independently be RELIABLE, "
            "avoid a search-grid boundary hit, have complete coverage when "
            "reported, and satisfy the unchanged prominence>=2.0 rule. "
            "Strong counterpart support requires accepted recurrence in "
            "multiple fixed windows, evidence in both c and o overall, "
            "and at least one fixed window with accepted c/o frequencies "
            "agreeing within the existing cross-band tolerance. The TESS "
            "drift law is not extrapolated into ATLAS epochs."
        ),
    }


def interpret_atlas_fixed_window_project(
    *,
    project_status: dict[str, Any] | None,
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared_by_id = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedSeries") or []
        if item.get("datasetID")
    }

    results: list[dict[str, Any]] = []

    if project_status is not None:
        for dataset in project_status.get("datasets") or []:
            dataset_id = str(
                dataset.get("datasetID")
                or dataset.get("id")
                or ""
            )
            prepared = prepared_by_id.get(dataset_id)
            if prepared is not None:
                results.append(
                    _dataset_result(dataset, prepared)
                )

    counterpart = _counterpart_summary(results)

    if counterpart.get("sourceSupported"):
        classification = (
            "ATLAS_FIXED_WINDOW_COUNTERPART_VARIABILITY_SUPPORTED"
        )
        origin = (
            "CATALOG_COUNTERPART_SUPPORTED_BY_"
            "FIXED_WINDOW_ATLAS_FORCED_PHOTOMETRY"
        )
        next_test = (
            "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
        )
    elif counterpart.get("sourceSuggestive"):
        classification = (
            "ATLAS_FIXED_WINDOW_COUNTERPART_VARIABILITY_SUGGESTIVE"
        )
        origin = (
            "CATALOG_COUNTERPART_SUGGESTIVE_BY_"
            "FIXED_WINDOW_ATLAS_FORCED_PHOTOMETRY"
        )
        next_test = (
            "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        )
    elif not preparation.get("preparedSeries"):
        classification = (
            "ATLAS_FIXED_WINDOW_NO_QUALIFYING_WINDOW_SERIES"
        )
        origin = (
            "UNRESOLVED_ATLAS_FIXED_WINDOW_CADENCE_LIMIT"
        )
        next_test = (
            "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        )
    else:
        classification = (
            "ATLAS_FIXED_WINDOW_COUNTERPART_RECURRENCE_NOT_CONFIRMED"
        )
        origin = (
            "UNRESOLVED_ATLAS_FIXED_WINDOW_RECURRENCE_TEST"
        )
        next_test = (
            "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        )

    return {
        "version": "openstar.tess-atlas-fixed-window.v1",
        "archive": preparation.get("archive"),
        "sourcePair": preparation.get("sourcePair"),
        "gaiaPairSeparationArcsec": (
            preparation.get("gaiaPairSeparationArcsec")
        ),
        "counterpartGaiaDR3SourceID": (
            preparation.get("counterpartGaiaDR3SourceID")
        ),
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get(
                "workerSemantics"
            ),
            "totalWorkUnits": preparation.get(
                "totalWorkUnits"
            ),
            "frequencySearch": preparation.get(
                "frequencySearch"
            ),
        },
        "rawArchiveRequeried": False,
        "rawArtifactsReusedFromV20_24": True,
        "tessDriftExtrapolated": False,
        "windowDays": preparation.get("windowDays"),
        "windowAnchor": preparation.get("windowAnchor"),
        "windows": preparation.get("windows") or [],
        "windowDiagnostics": (
            preparation.get("windowDiagnostics") or []
        ),
        "componentResults": results,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "acceptanceGuard": preparation.get(
            "acceptanceGuard"
        ),
        "interpretationGuard": preparation.get(
            "interpretationGuard"
        ),
    }
