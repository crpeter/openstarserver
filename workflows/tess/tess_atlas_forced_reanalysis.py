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
    MAX_REDUCED_CHI_SQUARED,
    MIN_PEAK_PROMINENCE_RATIO,
    MIN_CROSS_BAND_SUPPORT,
    MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD,
    _atlas_row_value,
    _parse_atlas_output,
    _parse_float,
    _parse_int,
)

MIN_BAND_NIGHTS = 20
MIN_BAND_BASELINE_DAYS = 30.0
MAX_NIGHTLY_ERROR_MULTIPLIER = 3.0


def _clean_signed_forced_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Preserve statistically valid signed forced-photometry measurements.

    v20.24 incorrectly required uJy/duJy >= 3 before nightly binning.
    That discards negative and low-S/N forced measurements and preferentially
    retains positive noise excursions. v20.25 removes that detection threshold.

    Rows are rejected only for:
      - missing/invalid numeric values,
      - unsupported filters,
      - tphot error flags,
      - excessive reported fit chi/N.

    Positive or negative flux is allowed.
    """
    accepted: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        mjd = _parse_float(_atlas_row_value(row, "MJD"))
        flux = _parse_float(_atlas_row_value(row, "uJy"))
        flux_error = _parse_float(_atlas_row_value(row, "duJy"))
        band = str(_atlas_row_value(row, "F") or "").strip().lower()
        error_code = _parse_int(_atlas_row_value(row, "err"))
        chi = _parse_float(_atlas_row_value(row, "chi/N", "chi"))

        if mjd is None or flux is None or flux_error is None or flux_error <= 0:
            rejection_counts["invalid-numeric-row"] += 1
            continue

        if band not in SUPPORTED_BANDS:
            rejection_counts["unsupported-band"] += 1
            continue

        if error_code not in (None, 0):
            rejection_counts["tphot-error"] += 1
            continue

        if chi is not None and chi > MAX_REDUCED_CHI_SQUARED:
            rejection_counts["high-reduced-chi-square"] += 1
            continue

        accepted.append(
            {
                "mjd": float(mjd),
                "uJy": float(flux),
                "duJy": float(flux_error),
                "snr": float(flux / flux_error),
                "band": band,
                "reducedChiSquared": chi,
                "magnitude": _parse_float(_atlas_row_value(row, "m")),
                "magnitudeError": _parse_float(_atlas_row_value(row, "dm")),
                "mag5sig": _parse_float(_atlas_row_value(row, "mag5sig")),
                "obs": str(_atlas_row_value(row, "Obs") or "").strip() or None,
            }
        )

    return accepted, {
        "rawRows": int(len(rows)),
        "acceptedSignedRows": int(len(accepted)),
        "rejectionCounts": dict(sorted(rejection_counts.items())),
    }


def _nightly_signed_series(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_band_night: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        night = int(math.floor(float(row["mjd"])))
        by_band_night[(str(row["band"]), night)].append(row)

    provisional: dict[str, list[dict[str, Any]]] = {
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

        provisional[band].append(
            {
                "mjd": float(np.sum(weights * times) / total_weight),
                "uJy": float(np.sum(weights * fluxes) / total_weight),
                "duJy": float(math.sqrt(1.0 / total_weight)),
                "rawPoints": int(len(group)),
            }
        )

    output: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}

    for band in SUPPORTED_BANDS:
        nights = sorted(
            provisional.get(band, []),
            key=lambda item: float(item["mjd"]),
        )

        band_diag = {
            "provisionalNights": int(len(nights)),
            "medianProvisionalNightlyErrorUJy": None,
            "maximumAcceptedNightlyErrorUJy": None,
            "acceptedNights": 0,
            "baselineDays": None,
        }

        if not nights:
            diagnostics[band] = band_diag
            continue

        nightly_errors = np.asarray(
            [float(item["duJy"]) for item in nights],
            dtype=np.float64,
        )
        median_error = float(np.median(nightly_errors))
        band_diag["medianProvisionalNightlyErrorUJy"] = median_error

        if not math.isfinite(median_error) or median_error <= 0:
            diagnostics[band] = band_diag
            continue

        maximum_error = median_error * MAX_NIGHTLY_ERROR_MULTIPLIER
        band_diag["maximumAcceptedNightlyErrorUJy"] = float(maximum_error)

        nights = [
            item
            for item in nights
            if math.isfinite(float(item["duJy"]))
            and 0 < float(item["duJy"]) <= maximum_error
        ]

        band_diag["acceptedNights"] = int(len(nights))

        if len(nights) < MIN_BAND_NIGHTS:
            diagnostics[band] = band_diag
            continue

        times = np.asarray(
            [float(item["mjd"]) for item in nights],
            dtype=np.float64,
        )
        fluxes = np.asarray(
            [float(item["uJy"]) for item in nights],
            dtype=np.float64,
        )
        errors = np.asarray(
            [float(item["duJy"]) for item in nights],
            dtype=np.float64,
        )

        baseline = float(times[-1] - times[0])
        band_diag["baselineDays"] = baseline

        if baseline < MIN_BAND_BASELINE_DAYS:
            diagnostics[band] = band_diag
            continue

        center = float(np.median(fluxes))
        mad = float(np.median(np.abs(fluxes - center)))
        scale = 1.4826 * mad

        if not math.isfinite(scale) or scale <= 0:
            scale = float(np.std(fluxes))

        if not math.isfinite(scale) or scale <= 0:
            diagnostics[band] = band_diag
            continue

        standardized = (fluxes - center) / scale
        local_times = times - float(times[0])

        output[band] = {
            "times": local_times,
            "flux": standardized,
            "sampleCount": int(len(local_times)),
            "baselineDays": baseline,
            "medianFluxUJy": center,
            "medianFluxErrorUJy": float(np.median(errors)),
            "medianAbsoluteNightlySNR": float(
                np.median(np.abs(fluxes / errors))
            ),
            "medianRawPointsPerNight": float(
                np.median([int(item["rawPoints"]) for item in nights])
            ),
        }

        diagnostics[band] = band_diag

    return output, diagnostics


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
        "band": prepared.get("band"),
        "sampleCount": prepared.get("sampleCount"),
        "baselineDays": prepared.get("baselineDays"),
        "medianFluxUJy": prepared.get("medianFluxUJy"),
        "medianAbsoluteNightlySNR": prepared.get("medianAbsoluteNightlySNR"),
        "periodStatus": status or None,
        "periodConfidence": project_dataset.get("periodConfidence"),
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "acceptedResidualBandVariability": accepted,
    }


def _summarize_source(
    role: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    source_results = [
        item
        for item in results
        if item.get("sourceRole") == role
    ]
    accepted = [
        item
        for item in source_results
        if item.get("acceptedResidualBandVariability")
    ]
    frequencies = [
        float(item["candidateFrequency"])
        for item in accepted
        if item.get("candidateFrequency") is not None
    ]

    median_frequency = None
    relative_spread = None
    supported = False

    if frequencies:
        median_frequency = float(
            np.median(np.asarray(frequencies, dtype=np.float64))
        )
        if median_frequency > 0:
            relative_spread = float(
                (max(frequencies) - min(frequencies)) / median_frequency
            )

        supported = bool(
            len(frequencies) >= MIN_CROSS_BAND_SUPPORT
            and relative_spread is not None
            and relative_spread <= MAX_CROSS_BAND_RELATIVE_FREQUENCY_SPREAD
        )

    return {
        "sourceRole": role,
        "bandResults": source_results,
        "acceptedBands": sorted(
            str(item.get("band")) for item in accepted
        ),
        "acceptedBandCount": int(len(accepted)),
        "medianAcceptedFrequency": median_frequency,
        "crossBandRelativeFrequencySpread": relative_spread,
        "sourceSupported": supported,
        "sourceSuggestive": bool(accepted) and not supported,
    }


def build_atlas_forced_photometry_reanalysis_project(
    *,
    source_project_id: str,
    source_dataset_id: str,
    atlas_v20_24_summary: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if atlas_v20_24_summary.get("classification") != (
        "ATLAS_NO_QUALIFYING_FORCED_PHOTOMETRY_TIME_SERIES"
    ):
        raise RuntimeError(
            "v20.25 is preregistered specifically to repair the v20.24 "
            "individual-SNR selection gate after ATLAS returned raw photometry."
        )

    search = dict(
        (
            (atlas_v20_24_summary.get("distributedValidation") or {})
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
            "v20.25 requires the frozen residual-frequency search definition "
            "from v20.24."
        )

    root = Path(output_dir) / "atlas-forced-photometry-reanalysis"
    root.mkdir(parents=True, exist_ok=True)

    prepared_series: list[dict[str, Any]] = []
    dataset_entries: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    v20_24_records = atlas_v20_24_summary.get("sourceRecords") or []
    if not v20_24_records:
        raise RuntimeError(
            "v20.25 cannot find the v20.24 ATLAS source records/raw artifact paths."
        )

    print(
        "   reusing immutable v20.24 ATLAS raw target-image photometry artifacts",
        flush=True,
    )
    print(
        "   individual uJy/duJy detection threshold removed before nightly binning",
        flush=True,
    )
    print(
        "   signed flux values, including negative and sub-3-sigma measurements, are retained",
        flush=True,
    )

    for record in v20_24_records:
        role = str(record.get("sourceRole") or "")
        source_id = _int(record.get("gaiaDR3SourceID"))
        raw_path_text = str(record.get("rawPath") or "").strip()

        if not role or source_id is None or not raw_path_text:
            raise RuntimeError(
                "v20.25 encountered an incomplete v20.24 ATLAS source record."
            )

        raw_path = Path(raw_path_text)
        if not raw_path.exists():
            raise RuntimeError(
                f"v20.25 raw ATLAS artifact is missing: {raw_path}. "
                "Do not silently re-query or substitute new archive data."
            )

        text = raw_path.read_text(encoding="utf-8")
        raw_rows = _parse_atlas_output(text)
        clean_rows, quality = _clean_signed_forced_rows(raw_rows)
        band_series, band_diagnostics = _nightly_signed_series(clean_rows)

        reanalysis_record = {
            "sourceRole": role,
            "gaiaDR3SourceID": int(source_id),
            "gaiaGMag": record.get("gaiaGMag"),
            "rawPath": str(raw_path.resolve()),
            "rawFileReusedFromV20_24": True,
            "rawRowCount": int(quality["rawRows"]),
            "acceptedSignedRowCount": int(quality["acceptedSignedRows"]),
            "rejectionCounts": quality["rejectionCounts"],
            "bandDiagnostics": band_diagnostics,
            "preparedBands": sorted(band_series),
        }
        source_records.append(reanalysis_record)

        print(
            f"      {role}: raw={reanalysis_record['rawRowCount']} | "
            f"signed-quality={reanalysis_record['acceptedSignedRowCount']} | "
            f"prepared bands={reanalysis_record['preparedBands']}",
            flush=True,
        )
        print(
            f"         rejection counts: {reanalysis_record['rejectionCounts']}",
            flush=True,
        )

        for band, series in band_series.items():
            dataset_id = (
                f"{source_dataset_id}-atlas-reanalysis-{role}-{band}-nightly-v1"
            )
            target_name = (
                f"{source_dataset_id} ATLAS reanalysis {role} "
                f"{band}-band nightly forced photometry"
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
                    "role": "atlas-target-image-forced-photometry-reanalysis",
                    "sourceRole": role,
                    "gaiaDR3SourceID": int(source_id),
                    "band": band,
                    "rawArtifactReusedFromV20_24": True,
                    "signedForcedFluxRetained": True,
                    "individualDetectionThresholdApplied": False,
                    "nightlyInverseVarianceBinning": True,
                    "differenceImagingUsed": False,
                    "tessDriftExtrapolated": False,
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
                "sourceRole": role,
                "gaiaDR3SourceID": int(source_id),
                "gaiaGMag": record.get("gaiaGMag"),
                "band": band,
                "sampleCount": int(series["sampleCount"]),
                "baselineDays": float(series["baselineDays"]),
                "medianFluxUJy": float(series["medianFluxUJy"]),
                "medianFluxErrorUJy": float(series["medianFluxErrorUJy"]),
                "medianAbsoluteNightlySNR": float(
                    series["medianAbsoluteNightlySNR"]
                ),
                "medianRawPointsPerNight": float(
                    series["medianRawPointsPerNight"]
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
                f"         {band}: {series['sampleCount']} nights | "
                f"baseline={series['baselineDays']:.1f} d | "
                f"median |nightly SNR|={series['medianAbsoluteNightlySNR']:.2f}",
                flush=True,
            )

    project_id: str | None = None
    project_path: str | None = None

    if dataset_entries:
        project_id = (
            f"{source_project_id}.investigation.{_safe(investigation_id)}."
            "atlas-forced-photometry-reanalysis-v1"
        )
        manifest = {
            "id": project_id,
            "name": (
                f"{source_project_id} — ATLAS statistically valid "
                "forced-photometry reanalysis"
            ),
            "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
            "datasets": dataset_entries,
            "investigation": {
                "sourceProjectID": source_project_id,
                "sourceDatasetID": source_dataset_id,
                "purpose": (
                    "atlas-signed-forced-photometry-nightly-reanalysis"
                ),
                "archive": "ATLAS Forced Photometry",
                "workerSemantics": (
                    "The immutable v20.24 target-image forced-photometry "
                    "files are reprocessed without an individual detection "
                    "threshold. Signed quality-valid fluxes are "
                    "inverse-variance binned nightly per source/filter; "
                    "workers execute ordinary Lomb-Scargle only over the "
                    "previously frozen residual-frequency band."
                ),
                "rawArchiveRequeried": False,
                "differenceImagingUsed": False,
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
            "openstar.tess-atlas-forced-photometry-reanalysis-preparation.v1"
        ),
        "archive": "ATLAS Forced Photometry",
        "sourcePair": atlas_v20_24_summary.get("sourcePair"),
        "sourceDefinitions": atlas_v20_24_summary.get("sourceDefinitions"),
        "gaiaPairSeparationArcsec": (
            atlas_v20_24_summary.get("gaiaPairSeparationArcsec")
        ),
        "projectID": project_id,
        "projectPath": project_path,
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": (
            "generic-lomb-scargle-on-atlas-nightly-signed-forced-photometry"
        ),
        "frequencySearch": search,
        "rawArchiveRequeried": False,
        "rawArtifactsReusedFromV20_24": True,
        "individualDetectionThresholdApplied": False,
        "signedForcedFluxRetained": True,
        "differenceImagingUsed": False,
        "tessDriftExtrapolated": False,
        "sourceRecords": source_records,
        "preparedSeries": prepared_series,
        "workUnitsPerDataset": int(work_units_per_dataset),
        "totalWorkUnits": int(
            len(dataset_entries) * work_units_per_dataset
        ),
        "qualityGuard": {
            "tphotErrorRequiredZeroOrMissing": True,
            "maximumReducedChiSquared": MAX_REDUCED_CHI_SQUARED,
            "individualFluxSNRThreshold": None,
            "signedFluxAllowed": True,
            "nightlyInverseVarianceBinning": True,
            "maximumNightlyErrorMultiplier": MAX_NIGHTLY_ERROR_MULTIPLIER,
            "minimumBandNights": MIN_BAND_NIGHTS,
            "minimumBandBaselineDays": MIN_BAND_BASELINE_DAYS,
            "minimumCrossBandSupport": MIN_CROSS_BAND_SUPPORT,
        },
        "interpretationGuard": (
            "v20.25 is a methodological correction to v20.24, not a new "
            "archive query. Forced photometry estimates flux at a fixed "
            "position whether or not an individual exposure contains a "
            "formal detection; therefore low-S/N and negative signed flux "
            "measurements are not discarded merely for failing a positive "
            "detection threshold. Only invalid numeric rows, tphot error "
            "flags, and excessive fit chi/N are removed before nightly "
            "inverse-variance binning. Nights with unusually poor "
            "uncertainty are removed by a preregistered per-band error "
            "threshold. The resulting source/filter time series are searched "
            "independently only inside the frozen TESS residual-frequency "
            "band. Source attribution still requires recurring accepted "
            "frequency support in at least two independent filters. The "
            "TESS drift law is not extrapolated into ATLAS epochs."
        ),
    }


def interpret_atlas_forced_photometry_reanalysis_project(
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

    target = _summarize_source(
        "target-control",
        results,
    )
    counterpart = _summarize_source(
        "catalog-counterpart",
        results,
    )

    target_supported = bool(target.get("sourceSupported"))
    counterpart_supported = bool(
        counterpart.get("sourceSupported")
    )
    target_suggestive = bool(target.get("sourceSuggestive"))
    counterpart_suggestive = bool(
        counterpart.get("sourceSuggestive")
    )

    prepared_roles = {
        str(item.get("sourceRole"))
        for item in preparation.get("preparedSeries") or []
    }

    if counterpart_supported and target_supported:
        classification = (
            "ATLAS_REANALYSIS_TARGET_AND_COUNTERPART_"
            "RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        )
        origin = (
            "TARGET_AND_COUNTERPART_SUPPORTED_BY_"
            "ATLAS_SIGNED_FORCED_PHOTOMETRY"
        )
        next_test = "JOINT_TARGET_COUNTERPART_VARIABILITY_MODEL"
    elif counterpart_supported:
        classification = (
            "ATLAS_REANALYSIS_COUNTERPART_"
            "RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        )
        origin = (
            "CATALOG_COUNTERPART_SUPPORTED_BY_"
            "ATLAS_SIGNED_FORCED_PHOTOMETRY"
        )
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif target_supported:
        classification = (
            "ATLAS_REANALYSIS_TARGET_"
            "RESIDUAL_BAND_VARIABILITY_SUPPORTED"
        )
        origin = (
            "TARGET_SUPPORTED_BY_ATLAS_SIGNED_FORCED_PHOTOMETRY"
        )
        next_test = "TARGET_INTRINSIC_RESIDUAL_MODELING"
    elif counterpart_suggestive and not target_suggestive:
        classification = (
            "ATLAS_REANALYSIS_COUNTERPART_"
            "RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        )
        origin = (
            "CATALOG_COUNTERPART_SUGGESTIVE_BY_"
            "ATLAS_SIGNED_FORCED_PHOTOMETRY"
        )
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif target_suggestive and not counterpart_suggestive:
        classification = (
            "ATLAS_REANALYSIS_TARGET_"
            "RESIDUAL_BAND_VARIABILITY_SUGGESTIVE"
        )
        origin = (
            "TARGET_SUGGESTIVE_BY_ATLAS_SIGNED_FORCED_PHOTOMETRY"
        )
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    elif not prepared_roles:
        classification = (
            "ATLAS_REANALYSIS_NO_QUALIFYING_NIGHTLY_TIME_SERIES"
        )
        origin = (
            "UNRESOLVED_ATLAS_SIGNED_FORCED_PHOTOMETRY_"
            "QUALITY_OR_CADENCE_LIMIT"
        )
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    else:
        classification = (
            "ATLAS_REANALYSIS_SOURCE_ATTRIBUTION_UNRESOLVED"
        )
        origin = (
            "ARCHIVAL_ATLAS_SIGNED_FORCED_PHOTOMETRY_"
            "SOURCE_ATTRIBUTION_UNRESOLVED"
        )
        next_test = "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"

    return {
        "version": (
            "openstar.tess-atlas-forced-photometry-reanalysis.v1"
        ),
        "archive": preparation.get("archive"),
        "sourcePair": preparation.get("sourcePair"),
        "sourceDefinitions": preparation.get("sourceDefinitions"),
        "gaiaPairSeparationArcsec": (
            preparation.get("gaiaPairSeparationArcsec")
        ),
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "rawArchiveRequeried": False,
        "rawArtifactsReusedFromV20_24": True,
        "individualDetectionThresholdApplied": False,
        "signedForcedFluxRetained": True,
        "differenceImagingUsed": False,
        "tessDriftExtrapolated": False,
        "sourceRecords": preparation.get("sourceRecords") or [],
        "componentResults": results,
        "targetControl": target,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "qualityGuard": preparation.get("qualityGuard"),
        "interpretationGuard": preparation.get(
            "interpretationGuard"
        ),
    }
