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
    SUPPORTED_BANDS,
    MIN_PEAK_PROMINENCE_RATIO,
    MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD,
    _parse_atlas_output,
)
from .tess_atlas_forced_reanalysis import (
    _clean_signed_forced_rows,
)

SEASON_GAP_DAYS = 75.0
MIN_SEASON_BAND_NIGHTS = 12
MIN_SEASON_BAND_BASELINE_DAYS = 30.0
MAX_NIGHTLY_ERROR_MULTIPLIER = 3.0

MIN_SUPPORTED_SEASONS = 2
MIN_SUPPORTED_SEASON_BAND_RESULTS = 3
MIN_SUGGESTIVE_SEASONS = 2


def _nightly_absolute_rows(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_band_night: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        band = str(row["band"])
        night = int(math.floor(float(row["mjd"])))
        by_band_night[(band, night)].append(row)

    output: dict[str, list[dict[str, Any]]] = {
        band: [] for band in SUPPORTED_BANDS
    }

    for (band, _), group in by_band_night.items():
        fluxes = np.asarray(
            [float(item["uJy"]) for item in group],
            dtype=np.float64,
        )
        errors = np.asarray(
            [float(item["duJy"]) for item in group],
            dtype=np.float64,
        )
        times = np.asarray(
            [float(item["mjd"]) for item in group],
            dtype=np.float64,
        )

        weights = 1.0 / np.square(errors)
        total_weight = float(np.sum(weights))
        if not math.isfinite(total_weight) or total_weight <= 0:
            continue

        output[band].append(
            {
                "mjd": float(np.sum(weights * times) / total_weight),
                "uJy": float(np.sum(weights * fluxes) / total_weight),
                "duJy": float(math.sqrt(1.0 / total_weight)),
                "rawPoints": int(len(group)),
            }
        )

    for band in output:
        output[band].sort(key=lambda item: float(item["mjd"]))

        if not output[band]:
            continue

        errors = np.asarray(
            [float(item["duJy"]) for item in output[band]],
            dtype=np.float64,
        )
        median_error = float(np.median(errors))

        if not math.isfinite(median_error) or median_error <= 0:
            output[band] = []
            continue

        maximum_error = median_error * MAX_NIGHTLY_ERROR_MULTIPLIER
        output[band] = [
            item
            for item in output[band]
            if 0 < float(item["duJy"]) <= maximum_error
        ]

    return output


def _global_seasons(
    nightly_by_band: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    all_times = sorted(
        {
            float(item["mjd"])
            for rows in nightly_by_band.values()
            for item in rows
        }
    )

    if not all_times:
        return []

    groups: list[list[float]] = [[all_times[0]]]

    for value in all_times[1:]:
        if value - groups[-1][-1] > SEASON_GAP_DAYS:
            groups.append([value])
        else:
            groups[-1].append(value)

    seasons: list[dict[str, Any]] = []

    for index, values in enumerate(groups, start=1):
        start = float(min(values))
        end = float(max(values))
        seasons.append(
            {
                "seasonIndex": int(index),
                "startMJD": start,
                "endMJD": end,
                "baselineDays": float(end - start),
            }
        )

    return seasons


def _rows_for_season(
    rows: list[dict[str, Any]],
    season: dict[str, Any],
) -> list[dict[str, Any]]:
    start = float(season["startMJD"])
    end = float(season["endMJD"])
    return [
        item
        for item in rows
        if start <= float(item["mjd"]) <= end
    ]


def _standardize_flux(fluxes: np.ndarray) -> np.ndarray:
    values = np.asarray(fluxes, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad

    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))

    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("ATLAS season flux series has no finite variability scale.")

    return (values - center) / scale


def _prepare_season_band(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(rows) < MIN_SEASON_BAND_NIGHTS:
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
    if baseline < MIN_SEASON_BAND_BASELINE_DAYS:
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
    prominence = _float(project_dataset.get("candidatePeakProminenceRatio"))
    status = str(project_dataset.get("periodStatus") or "")
    coverage = project_dataset.get("coverageComplete")

    accepted = bool(
        status == "RELIABLE"
        and (coverage is None or bool(coverage))
        and prominence is not None
        and prominence >= MIN_PEAK_PROMINENCE_RATIO
        and frequency is not None
        and frequency > 0
    )

    return {
        "datasetID": prepared.get("datasetID"),
        "sourceRole": prepared.get("sourceRole"),
        "gaiaDR3SourceID": prepared.get("gaiaDR3SourceID"),
        "seasonIndex": prepared.get("seasonIndex"),
        "band": prepared.get("band"),
        "sampleCount": prepared.get("sampleCount"),
        "baselineDays": prepared.get("baselineDays"),
        "startMJD": prepared.get("startMJD"),
        "endMJD": prepared.get("endMJD"),
        "midMJD": prepared.get("midMJD"),
        "periodStatus": status or None,
        "periodConfidence": project_dataset.get("periodConfidence"),
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "acceptedSeasonalResidualBandVariability": accepted,
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
    counterpart_results = [
        item
        for item in results
        if item.get("sourceRole") == "catalog-counterpart"
    ]
    accepted = [
        item
        for item in counterpart_results
        if item.get("acceptedSeasonalResidualBandVariability")
    ]

    accepted_seasons = sorted(
        {
            int(item["seasonIndex"])
            for item in accepted
            if item.get("seasonIndex") is not None
        }
    )
    accepted_bands = sorted(
        {
            str(item["band"])
            for item in accepted
            if item.get("band")
        }
    )

    by_season: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in accepted:
        if item.get("seasonIndex") is not None:
            by_season[int(item["seasonIndex"])].append(item)

    cross_band_consistent_seasons: list[dict[str, Any]] = []

    for season_index, items in sorted(by_season.items()):
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

        if relative_difference <= MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD:
            cross_band_consistent_seasons.append(
                {
                    "seasonIndex": int(season_index),
                    "cFrequency": float(c_item["candidateFrequency"]),
                    "oFrequency": float(o_item["candidateFrequency"]),
                    "relativeFrequencyDifference": float(relative_difference),
                }
            )

    supported = bool(
        len(accepted_seasons) >= MIN_SUPPORTED_SEASONS
        and len(accepted) >= MIN_SUPPORTED_SEASON_BAND_RESULTS
        and len(accepted_bands) >= 2
        and len(cross_band_consistent_seasons) >= 1
    )

    suggestive = bool(
        not supported
        and len(accepted_seasons) >= MIN_SUGGESTIVE_SEASONS
        and len(accepted) >= 2
    )

    frequencies = [
        float(item["candidateFrequency"])
        for item in accepted
        if item.get("candidateFrequency") is not None
    ]

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
            rms = float(np.sqrt(np.mean(np.square(residuals))))
            frequency_trend = {
                "slopeCyclesPerDayPerDay": float(slope),
                "interceptCyclesPerDay": float(intercept),
                "rmsCyclesPerDay": rms,
                "fitCount": int(len(x)),
                "tessDriftLawUsedInFit": False,
            }

    return {
        "seasonBandResults": counterpart_results,
        "acceptedSeasonBandResults": accepted,
        "acceptedSeasonBandCount": int(len(accepted)),
        "acceptedSeasons": accepted_seasons,
        "acceptedSeasonCount": int(len(accepted_seasons)),
        "acceptedBands": accepted_bands,
        "crossBandConsistentSeasons": cross_band_consistent_seasons,
        "crossBandConsistentSeasonCount": int(
            len(cross_band_consistent_seasons)
        ),
        "independentFrequencyTrend": frequency_trend,
        "sourceSupported": supported,
        "sourceSuggestive": suggestive,
        "acceptedFrequencies": frequencies,
    }


def build_atlas_time_resolved_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    atlas_v20_24_summary: dict[str, Any],
    atlas_v20_25_preparation: dict[str, Any],
    atlas_v20_25_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if atlas_v20_25_summary.get("classification") != (
        "ATLAS_REANALYSIS_SOURCE_ATTRIBUTION_UNRESOLVED"
    ):
        raise RuntimeError(
            "v20.26 is preregistered for the completed v20.25 unresolved "
            "global ATLAS reanalysis branch."
        )

    search = dict(
        (
            (atlas_v20_25_summary.get("distributedValidation") or {})
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
            "v20.26 requires the frozen residual-frequency search definition."
        )

    root = Path(output_dir) / "atlas-time-resolved"
    root.mkdir(parents=True, exist_ok=True)

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
            "v20.26 cannot find the immutable v20.24 counterpart raw ATLAS artifact."
        )

    raw_path_text = str(counterpart_record.get("rawPath") or "").strip()
    if not raw_path_text:
        raise RuntimeError(
            "v20.26 counterpart raw ATLAS artifact path is missing."
        )

    raw_path = Path(raw_path_text)
    if not raw_path.exists():
        raise RuntimeError(
            f"v20.26 raw counterpart ATLAS artifact is missing: {raw_path}."
        )

    raw_rows = _parse_atlas_output(
        raw_path.read_text(encoding="utf-8")
    )
    clean_rows, signed_quality = _clean_signed_forced_rows(raw_rows)
    nightly_by_band = _nightly_absolute_rows(clean_rows)
    seasons = _global_seasons(nightly_by_band)

    print(
        f"   immutable counterpart raw rows: {len(raw_rows)}",
        flush=True,
    )
    print(
        f"   signed quality-valid rows: {signed_quality['acceptedSignedRows']}",
        flush=True,
    )
    print(
        f"   shared ATLAS observing seasons: {len(seasons)}",
        flush=True,
    )

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []

    for season in seasons:
        season_index = int(season["seasonIndex"])

        for band in ("c", "o"):
            band_rows = _rows_for_season(
                nightly_by_band.get(band, []),
                season,
            )
            series = _prepare_season_band(band_rows)

            if series is None:
                continue

            dataset_id = (
                f"{source_dataset_id}-atlas-time-resolved-"
                f"catalog-counterpart-season-{season_index}-{band}-v1"
            )
            target_name = (
                f"{source_dataset_id} ATLAS counterpart season "
                f"{season_index} {band}-band"
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
                    "role": "atlas-time-resolved-counterpart-recurrence",
                    "sourceRole": "catalog-counterpart",
                    "gaiaDR3SourceID": int(
                        counterpart_record["gaiaDR3SourceID"]
                    ),
                    "seasonIndex": season_index,
                    "band": band,
                    "signedForcedFluxRetained": True,
                    "individualDetectionThresholdApplied": False,
                    "nightlyInverseVarianceBinning": True,
                    "tessDriftExtrapolated": False,
                    "tessDriftLawUsedToChooseSeasonFrequency": False,
                },
                "source": {
                    "mission": "ATLAS",
                    "archive": "ATLAS Forced Photometry",
                    "filter": band,
                    "distributedSamples": int(series["sampleCount"]),
                    "baselineDays": float(series["baselineDays"]),
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
                "seasonIndex": season_index,
                "band": band,
                "sampleCount": int(series["sampleCount"]),
                "baselineDays": float(series["baselineDays"]),
                "startMJD": float(series["startMJD"]),
                "endMJD": float(series["endMJD"]),
                "midMJD": float(series["midMJD"]),
                "medianFluxUJy": float(series["medianFluxUJy"]),
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
                f"      season {season_index} {band}: "
                f"{series['sampleCount']} nights | "
                f"baseline={series['baselineDays']:.1f} d",
                flush=True,
            )

    project_id: str | None = None
    project_path: str | None = None

    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "atlas-time-resolved-counterpart-v1"
        )
        manifest = {
            "id": project_id,
            "name": (
                f"{source_project_id} — ATLAS time-resolved "
                "counterpart recurrence"
            ),
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": (
                    "atlas-time-resolved-counterpart-recurrence"
                ),
                "archive": "ATLAS Forced Photometry",
                "workerSemantics": (
                    "The immutable ATLAS counterpart target-image "
                    "forced-photometry series is split into shared observing "
                    "seasons. Each season/filter is searched independently "
                    "with ordinary Lomb-Scargle over the same frozen residual "
                    "frequency band. The TESS drift law is not used to choose "
                    "or predict ATLAS season frequencies."
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
        "version": "openstar.tess-atlas-time-resolved-preparation.v1",
        "archive": "ATLAS Forced Photometry",
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": (
            "generic-lomb-scargle-on-atlas-counterpart-season-filter-series"
        ),
        "sourcePair": atlas_v20_25_summary.get("sourcePair"),
        "gaiaPairSeparationArcsec": (
            atlas_v20_25_summary.get("gaiaPairSeparationArcsec")
        ),
        "counterpartGaiaDR3SourceID": int(
            counterpart_record["gaiaDR3SourceID"]
        ),
        "frequencySearch": search,
        "rawArchiveRequeried": False,
        "rawArtifactsReusedFromV20_24": True,
        "v20_25GlobalResultReusedAsMotivationOnly": True,
        "tessDriftExtrapolated": False,
        "seasonGapDays": SEASON_GAP_DAYS,
        "seasons": seasons,
        "preparedSeries": prepared_series,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(
            len(dataset_entries) * work_units_per_dataset
        ),
        "acceptanceGuard": {
            "minimumPeakProminenceRatio": MIN_PEAK_PROMINENCE_RATIO,
            "minimumSeasonBandNights": MIN_SEASON_BAND_NIGHTS,
            "minimumSeasonBandBaselineDays": MIN_SEASON_BAND_BASELINE_DAYS,
            "maximumCrossBandRelativeFrequencySpread": (
                MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD
            ),
            "minimumSupportedSeasons": MIN_SUPPORTED_SEASONS,
            "minimumSupportedSeasonBandResults": (
                MIN_SUPPORTED_SEASON_BAND_RESULTS
            ),
            "minimumSuggestiveSeasons": MIN_SUGGESTIVE_SEASONS,
        },
        "interpretationGuard": (
            "v20.26 does not lower the v20.25 peak-prominence threshold and "
            "does not perform a new archive query. It addresses the known "
            "nonstationary nature of the TESS residual by testing the "
            "immutable ATLAS counterpart light curve in independent observing "
            "seasons. Every season/filter must independently satisfy the same "
            "RELIABLE, coverage, positive-frequency, and prominence>=2.0 "
            "requirements used previously. Strong counterpart support requires "
            "accepted recurrence across multiple seasons, multiple filters, "
            "and at least one season in which accepted c/o frequencies agree "
            "within the existing cross-band tolerance. An independent ATLAS "
            "frequency trend may be summarized only after the season fits; "
            "the TESS drift law is not extrapolated into ATLAS epochs."
        ),
    }


def interpret_atlas_time_resolved_project(
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
            "ATLAS_TIME_RESOLVED_COUNTERPART_VARIABILITY_SUPPORTED"
        )
        origin = (
            "CATALOG_COUNTERPART_SUPPORTED_BY_"
            "TIME_RESOLVED_ATLAS_FORCED_PHOTOMETRY"
        )
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif counterpart.get("sourceSuggestive"):
        classification = (
            "ATLAS_TIME_RESOLVED_COUNTERPART_VARIABILITY_SUGGESTIVE"
        )
        origin = (
            "CATALOG_COUNTERPART_SUGGESTIVE_BY_"
            "TIME_RESOLVED_ATLAS_FORCED_PHOTOMETRY"
        )
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif not preparation.get("preparedSeries"):
        classification = (
            "ATLAS_TIME_RESOLVED_NO_QUALIFYING_SEASON_SERIES"
        )
        origin = (
            "UNRESOLVED_ATLAS_TIME_RESOLVED_CADENCE_LIMIT"
        )
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    else:
        classification = (
            "ATLAS_TIME_RESOLVED_COUNTERPART_RECURRENCE_NOT_CONFIRMED"
        )
        origin = (
            "UNRESOLVED_ATLAS_TIME_RESOLVED_RECURRENCE_TEST"
        )
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"

    return {
        "version": "openstar.tess-atlas-time-resolved.v1",
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
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "rawArchiveRequeried": False,
        "rawArtifactsReusedFromV20_24": True,
        "tessDriftExtrapolated": False,
        "seasonGapDays": preparation.get("seasonGapDays"),
        "seasons": preparation.get("seasons") or [],
        "componentResults": results,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "acceptanceGuard": preparation.get("acceptanceGuard"),
        "interpretationGuard": preparation.get(
            "interpretationGuard"
        ),
    }
