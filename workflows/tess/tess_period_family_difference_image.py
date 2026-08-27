"""Append-only source localization for an unresolved recurrent period family.

This experiment deliberately does not search for a period.  It freezes the
sector-local frequencies already persisted by the targeted independent-sector
search, then asks a different question of the TESS pixels: where on the sky is
the high-minus-low phase signal centered in each sector?
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from openstar_investigation import Investigation, InvestigationStore, sha256_json
from openstar_workflow import StageRequest

from .tess_difference_image import (
    MIN_IMAGE_PEAK_SNR,
    SOURCE_MATCH_MAX_PIXELS,
    _centroid_from_frames,
    _extreme_indices,
    _jackknife_uncertainty,
    _phase_model,
)
from .tess_localization import (
    MAX_CADENCES,
    MIN_VALID_CADENCES,
    OFF_TARGET_MIN_PIXELS,
    _background_subtract_cube,
    _download_tpf,
    _pixel_scale_arcsec,
    _uniform_indices,
    _world_offsets_arcsec,
)
from .tess_offset_variability import _skycoord
from .tess_sector_archive import TessArchiveTransientError
from .tess_residual_localization import _write_json


HANDLER_PREFIX = "openstar.tess.period-family-difference-imaging."
PREPARE_HANDLER = HANDLER_PREFIX + "prepare"
RUN_HANDLER = HANDLER_PREFIX + "run"
INTERPRET_HANDLER = HANDLER_PREFIX + "interpret"
MIN_CROSS_SECTOR_SUPPORT = 3
MAX_OFF_TARGET_SKY_SCATTER_ARCSEC = 15.0

_TERMINAL_CONTROL = {
    "branchAssessments": [],
    "selectedExperiment": None,
    "schedulerAction": "INVESTIGATION_COMPLETE",
}

_BOUNDARY_STAGES = (
    ("001-prepare-target", "openstar.tess.prepare-target"),
    ("002-primary-distributed-search", "openstar.tess.primary-project.run"),
    ("003-catalog-identity", "openstar.tess.catalog-identity"),
    ("004-hypotheses", "openstar.tess.hypotheses"),
    ("005-planner", "openstar.tess.planner"),
    ("006-prepare-independent-sectors", "openstar.tess.independent.prepare"),
    ("007-run-independent-sectors", "openstar.tess.independent.run"),
    ("008-interpret-independent-sectors", "openstar.tess.independent.interpret"),
    ("009-prepare-broad-independent-search", "openstar.tess.independent.broad.prepare"),
    ("010-run-broad-independent-search", "openstar.tess.independent.broad.run"),
    ("011-interpret-broad-independent-search", "openstar.tess.independent.broad.interpret"),
    ("012-finalize", "openstar.tess.finalize"),
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _claim(result: dict[str, Any]) -> str | None:
    value = result.get("claim")
    if isinstance(value, dict):
        value = value.get("claim")
    return str(value) if isinstance(value, str) else None


def _base_stages(investigation: Investigation) -> tuple[Any, ...]:
    stages = investigation.stages
    if len(stages) == len(_BOUNDARY_STAGES) + 1:
        current = stages[-1]
        if not (
            current.id == "013-prepare-period-family-difference-imaging"
            and current.handler_id == PREPARE_HANDLER
            and current.status == "RUNNING"
            and current.triggered_by_stage_id == "012-finalize"
        ):
            raise RuntimeError("Unexpected stage after the frozen 12-stage boundary.")
        stages = stages[:-1]
    if len(stages) != len(_BOUNDARY_STAGES):
        raise RuntimeError("The manual localization bridge requires the exact 12-stage terminal boundary.")
    for stage, (stage_id, handler_id) in zip(stages, _BOUNDARY_STAGES):
        if not (
            stage.id == stage_id
            and stage.handler_id == handler_id
            and stage.status == "COMPLETE"
            and stage.result is not None
        ):
            raise RuntimeError(f"Frozen boundary mismatch at {stage_id}.")
    if not stages[-1].stop:
        raise RuntimeError("The frozen stage-012 finalizer is not terminal.")
    return stages


def freeze_period_family_boundary(investigation: Investigation) -> dict[str, Any]:
    """Validate and freeze the narrow unresolved-family scientific boundary."""
    stages = _base_stages(investigation)
    by_id = {stage.id: stage for stage in stages}
    prepared = by_id["001-prepare-target"].result or {}
    primary = by_id["002-primary-distributed-search"].result or {}
    identity = by_id["003-catalog-identity"].result or {}
    independent_preparation = by_id["006-prepare-independent-sectors"].result or {}
    independent_run = by_id["007-run-independent-sectors"].result or {}
    independent = by_id["008-interpret-independent-sectors"].result or {}
    broad_preparation = by_id["009-prepare-broad-independent-search"].result or {}
    broad = by_id["011-interpret-broad-independent-search"].result or {}
    final = by_id["012-finalize"].result or {}

    tic_id = prepared.get("ticID")
    primary_sector = prepared.get("sector")
    primary_frequency = _finite(primary.get("candidateFrequency"))
    primary_period = _finite(primary.get("candidatePeriodDays"))
    if not (
        isinstance(tic_id, int)
        and isinstance(primary_sector, int)
        and primary_frequency is not None
        and primary_frequency > 0
        and primary_period is not None
        and primary_period > 0
        and math.isclose(primary_frequency * primary_period, 1.0, rel_tol=1e-6)
        and str(primary.get("periodStatus") or "").upper() == "RELIABLE"
        and str(primary.get("periodConfidence") or "").lower() == "high"
    ):
        raise RuntimeError("The primary period evidence is not the required reliable frozen detection.")

    tic_metadata = ((identity.get("tic") or {}).get("metadata") or {})
    ra_deg = _finite(tic_metadata.get("raDeg"))
    dec_deg = _finite(tic_metadata.get("decDeg"))
    if identity.get("ticID") != tic_id or ra_deg is None or dec_deg is None:
        raise RuntimeError("The catalog identity lacks the target TIC sky position.")

    sector_results = list(independent.get("sectorResults") or [])
    run_datasets = list(independent_run.get("datasets") or [])
    prepared_sectors = list(independent_preparation.get("preparedSectors") or [])
    result_by_sector = {int(item["sector"]): item for item in sector_results if item.get("sector") is not None}
    dataset_by_sector = {int(item["sector"]): item for item in run_datasets if item.get("sector") is not None}
    prepared_ids = [int(item["sector"]) for item in prepared_sectors if item.get("sector") is not None]
    sector_ids = [int(item["sector"]) for item in sector_results if item.get("sector") is not None]
    if not (
        len(sector_ids) >= MIN_CROSS_SECTOR_SUPPORT
        and len(set(sector_ids)) == len(sector_ids) == len(sector_results) == len(dataset_by_sector)
        and prepared_ids == sector_ids
        and independent.get("eligibleSectorCount") == len(sector_ids)
        and independent.get("supportingSectorCount") == 0
        and independent.get("resolutionLimitedSectorCount") == len(sector_ids)
        and _claim(independent.get("claimDecision") or {}) == "HUMAN_REVIEW_REQUIRED"
    ):
        raise RuntimeError("Independent-sector evidence is not the unresolved resolution-limited boundary.")

    contradiction = independent.get("contradictionPlan") or {}
    if not (
        contradiction.get("action") == "BROAD_INDEPENDENT_SEARCH"
        and contradiction.get("reason")
        == "targeted-candidate-not-recurrent-independent-sectors-contain-alternate-reliable-structure"
        and contradiction.get("reliableSectorCount") == len(sector_ids)
    ):
        raise RuntimeError("The persisted contradiction plan did not authorize broad independent search.")

    detections: list[dict[str, Any]] = []
    for sector in sector_ids:
        result = result_by_sector[sector]
        dataset = dataset_by_sector[sector]
        frequency = _finite(result.get("candidateFrequency"))
        period = _finite(result.get("candidatePeriodDays"))
        if frequency is None or period is None or frequency <= 0 or period <= 0:
            raise RuntimeError(f"Sector {sector} lacks a finite persisted candidate.")
        if not (
            math.isclose(frequency * period, 1.0, rel_tol=1e-6)
            and math.isclose(frequency, float(dataset.get("candidateFrequency")), rel_tol=1e-12)
            and math.isclose(period, float(dataset.get("candidatePeriodDays")), rel_tol=1e-12)
            and result.get("recurrenceClassification") == "RESOLUTION_LIMITED"
            and result.get("resolutionLimited") is True
            and result.get("supportsTarget") is False
            and result.get("eligibleForRecurrence") is True
            and result.get("boundaryHit") is False
            and str(dataset.get("periodStatus") or "").upper() == "RELIABLE"
            and str(dataset.get("periodConfidence") or "").lower() == "high"
        ):
            raise RuntimeError(f"Sector {sector} is not a reliable resolution-limited period-family member.")
        detections.append(
            {
                "sector": sector,
                "datasetID": result.get("datasetID"),
                "frequencyCyclesPerDay": frequency,
                "periodDays": period,
                "power": _finite(dataset.get("candidatePower")),
                "peakProminenceRatio": _finite(dataset.get("candidatePeakProminenceRatio")),
                "foldCoherence": _finite(dataset.get("candidateFoldCoherence")),
                "recurrenceClassification": "RESOLUTION_LIMITED",
                "supportsOriginalCandidate": False,
            }
        )

    broad_results = list(broad.get("sectorResults") or [])
    broad_ids = [int(item["sector"]) for item in broad_results if item.get("sector") is not None]
    boundary_hits = sum(item.get("boundaryHit") is True for item in broad_results)
    best_cluster = broad.get("bestCluster") or {}
    if not (
        [int(item["sector"]) for item in broad_preparation.get("preparedSectors") or []] == sector_ids
        and broad_ids == sector_ids
        and _claim(broad.get("claimDecision") or {}) == "HUMAN_REVIEW_REQUIRED"
        and broad.get("promotionEligible") is False
        and broad.get("selectedPeriodDays") is None
        and broad.get("eligibleSectorCount") == 1
        and boundary_hits == len(sector_ids) - 1
        and best_cluster.get("count") == 1
        and broad.get("promotionBlockers") == ["insufficient-independent-sector-support"]
    ):
        raise RuntimeError("Broad-search evidence is not the frozen no-stable-cluster boundary.")

    final_claim = _claim(final)
    if not (
        final_claim == "HUMAN_REVIEW_REQUIRED"
        and final.get("automaticDiscoveryClaim") is False
        and final.get("selectedPeriodDays") is None
        and final.get("recommendedNextTest") is None
    ):
        raise RuntimeError("The final claim is not the required fail-closed unresolved result.")

    official_sectors = {int(value) for value in (identity.get("tess") or {}).get("officialSectors") or []}
    if not set(sector_ids).issubset(official_sectors):
        raise RuntimeError("A frozen independent sector is absent from the official-sector identity evidence.")

    return {
        "version": "openstar.tess-period-family-localization-boundary.v1",
        "investigationID": investigation.id,
        "ticID": tic_id,
        "targetSky": {"raDeg": ra_deg, "decDeg": dec_deg},
        "primaryDetection": {
            "sector": primary_sector,
            "frequencyCyclesPerDay": primary_frequency,
            "periodDays": primary_period,
            "power": _finite(primary.get("candidatePower")),
        },
        "independentSectorDetections": detections,
        "broadSearchOutcome": {
            "eligibleSectorCount": broad.get("eligibleSectorCount"),
            "boundaryHitCount": boundary_hits,
            "bestClusterSectorCount": best_cluster.get("count"),
            "promotionEligible": False,
        },
        "claim": "HUMAN_REVIEW_REQUIRED",
        "periodFamilyResolved": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "periodDetectionRecomputed": False,
    }


def verified_period_family_boundary(
    store: InvestigationStore, investigation: Investigation
) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind every authoritative stage to its immutable on-disk ledger."""
    frozen = freeze_period_family_boundary(investigation)
    stages = _base_stages(investigation)
    hashes: dict[str, str] = {}
    for stage in stages:
        ledger_hash = store.verified_terminal_stage_ledger_hash(investigation.id, stage)
        if ledger_hash is None:
            raise RuntimeError(f"Immutable ledger verification failed for {stage.id}.")
        hashes[stage.id] = ledger_hash
    return frozen, hashes


def admit_period_family_difference_imaging(
    store: InvestigationStore, investigation: Investigation
) -> Investigation:
    """Explicitly reopen only the verified manual boundary, without adding a stage."""
    if any(stage.handler_id.startswith(HANDLER_PREFIX) for stage in investigation.stages):
        return investigation
    control = investigation.metadata.get("controlState")
    if (
        investigation.status == "RUNNING"
        and isinstance(control, dict)
        and control.get("recovery") == "TESS_MANUAL_PERIOD_FAMILY_DIFFERENCE_IMAGING_V1"
        and control.get("schedulerAction") == "RUN_EXPERIMENT"
        and (control.get("selectedExperiment") or {}).get("id")
        == "013-prepare-period-family-difference-imaging"
        and (control.get("selectedExperiment") or {}).get("handler_id") == PREPARE_HANDLER
    ):
        verified_period_family_boundary(store, investigation)
        return investigation
    if not (
        investigation.status == "COMPLETE"
        and control == _TERMINAL_CONTROL
    ):
        raise RuntimeError("Manual localization admission requires the exact completed control state.")
    verified_period_family_boundary(store, investigation)
    request = StageRequest(
        "013-prepare-period-family-difference-imaging",
        PREPARE_HANDLER,
        {},
        "012-finalize",
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_MANUAL_PERIOD_FAMILY_DIFFERENCE_IMAGING_V1",
        },
    )


def prepare_period_family_difference_imaging(
    *, frozen_boundary: dict[str, Any], ledger_hashes: dict[str, str],
    output_dir: Path, investigation_id: str,
) -> dict[str, Any]:
    root = Path(output_dir) / "period-family-difference-imaging"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-period-family-difference-imaging-preparation.v1",
        "investigationID": investigation_id,
        "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-phase-difference-image-centroiding",
        "workerWorkload": None,
        "ticID": frozen_boundary["ticID"],
        "targetSky": frozen_boundary["targetSky"],
        "primaryDetection": frozen_boundary["primaryDetection"],
        "sectorDetections": frozen_boundary["independentSectorDetections"],
        "authoritativeStageLedgerSHA256": dict(ledger_hashes),
        "claimBeforeExperiment": "HUMAN_REVIEW_REQUIRED",
        "periodFamilyResolved": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "periodDetectionRecomputed": False,
        "scientificQuestion": (
            "Is the persisted approximately 4.5-day period-family signal spatially "
            "centered on the TIC target in at least three independent sectors?"
        ),
        "interpretationGuard": (
            "Each sector uses its own previously persisted frequency only as a phase reference. "
            "No Lomb-Scargle search is rerun, and source localization cannot resolve the physical cycle or mechanism."
        ),
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def _filled_cube(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = cube.reshape(len(cube), -1).astype(np.float64)
    valid = np.mean(np.isfinite(flat), axis=0) >= 0.90
    if not np.any(valid):
        raise RuntimeError("No pixel has at least 90 percent finite cadences.")
    medians = np.nanmedian(flat, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    flat = np.where(np.isfinite(flat), flat, medians[None, :])
    return flat.reshape(cube.shape), valid.reshape(cube.shape[1:])


def _sector_classification(distance: float, uncertainty: float, usable: bool) -> str:
    if not usable:
        return "NO_QUALITY_LOCALIZATION"
    if distance + 2.0 * uncertainty <= SOURCE_MATCH_MAX_PIXELS:
        return "TARGET_CONSISTENT"
    if distance - 2.0 * uncertainty >= OFF_TARGET_MIN_PIXELS:
        return "OFF_TARGET"
    return "AMBIGUOUS"


def _measure_sector(item: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    times = np.asarray(item["times"], dtype=np.float64)
    cube = np.asarray(item["fluxCube"], dtype=np.float64)
    if cube.ndim != 3 or cube.shape[0] != len(times):
        raise ValueError("fluxCube must have shape (cadence, row, column) matching times.")
    keep = np.isfinite(times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
    times, cube = times[keep], cube[keep]
    indices = _uniform_indices(len(times), MAX_CADENCES)
    times, cube = times[indices], cube[indices]
    if len(times) < MIN_VALID_CADENCES:
        raise RuntimeError(f"Only {len(times)} usable cadences; need {MIN_VALID_CADENCES}.")
    corrected, background = _background_subtract_cube(cube)
    corrected, valid = _filled_cube(corrected)
    aperture = np.sum(corrected[:, valid], axis=1)
    frequency = float(detection["frequencyCyclesPerDay"])
    phase = _phase_model(times, aperture, frequency)
    high, low = _extreme_indices(phase["model"])
    image = _centroid_from_frames(corrected, valid, high, low)
    uncertainty, jackknife = _jackknife_uncertainty(corrected, valid, high, low)
    target = item["targetPixel"]
    distance = math.hypot(
        float(image["centroidX"]) - float(target["x"]),
        float(image["centroidY"]) - float(target["y"]),
    )
    usable = float(image["peakSNR"]) >= MIN_IMAGE_PEAK_SNR
    classification = _sector_classification(distance, float(uncertainty), usable)
    pixel_scale = _finite(item.get("pixelScaleArcsec"))
    return {
        "sector": int(detection["sector"]),
        "persistedFrequencyCyclesPerDay": frequency,
        "persistedPeriodDays": float(detection["periodDays"]),
        "frequencySource": "stage-007-targeted-independent-search",
        "periodDetectionRecomputed": False,
        "usableCadences": int(len(times)),
        "backgroundCorrection": background,
        "phaseModel": {
            "amplitude": phase["amplitude"],
            "phaseRadians": phase["phaseRadians"],
            "explainedVariance": phase["explainedVariance"],
            "highCadences": int(len(high)),
            "lowCadences": int(len(low)),
        },
        "differenceImage": image,
        "differenceImageUsable": usable,
        "centroidUncertaintyPixels": float(uncertainty),
        "jackknifeCentroids": jackknife,
        "targetPixel": {"x": float(target["x"]), "y": float(target["y"])},
        "targetDistancePixels": float(distance),
        "targetSeparationArcsec": distance * pixel_scale if pixel_scale is not None else None,
        "skyOffsetEastArcsec": _finite(item.get("skyOffsetEastArcsec")),
        "skyOffsetNorthArcsec": _finite(item.get("skyOffsetNorthArcsec")),
        "centroidSky": item.get("centroidSky"),
        "classification": classification,
        "acquisitionProvenance": item.get("acquisitionProvenance"),
        "thresholds": {
            "minimumPeakSNR": MIN_IMAGE_PEAK_SNR,
            "targetMatchMaximumPixels": SOURCE_MATCH_MAX_PIXELS,
            "offTargetMinimumPixels": OFF_TARGET_MIN_PIXELS,
            "uncertaintyMultiplier": 2.0,
        },
    }


def _production_sector_input(preparation: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    target = preparation["targetSky"]
    tpf, source = _download_tpf(
        tic_id=int(preparation["ticID"]),
        sector=int(detection["sector"]),
        ra_deg=float(target["raDeg"]),
        dec_deg=float(target["decDeg"]),
    )
    times = np.asarray(tpf.time.value, dtype=np.float64)
    flux = getattr(tpf.flux, "value", tpf.flux)
    if np.ma.isMaskedArray(flux):
        flux = np.ma.filled(flux, np.nan)
    cube = np.asarray(flux, dtype=np.float64)
    coordinate = _skycoord(float(target["raDeg"]), float(target["decDeg"]))
    target_x, target_y = tpf.wcs.world_to_pixel(coordinate)

    # The centroid is not known until after measurement.  Preserve the WCS in
    # this private runtime-only input so the public result can record sky offsets.
    return {
        "sector": int(detection["sector"]),
        "times": times,
        "fluxCube": cube,
        "targetPixel": {"x": float(target_x), "y": float(target_y)},
        "pixelScaleArcsec": _pixel_scale_arcsec(tpf.wcs),
        "_wcs": tpf.wcs,
        "_targetCoordinate": coordinate,
        "acquisitionProvenance": source,
    }


def run_period_family_difference_imaging(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure phase-difference centroids without any new period search."""
    supplied = None if sector_inputs is None else {
        int(item["sector"]): item for item in sector_inputs
    }
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for detection in preparation.get("sectorDetections") or []:
        sector = int(detection["sector"])
        try:
            item = (_production_sector_input(preparation, detection)
                    if supplied is None else supplied[sector])
            result = _measure_sector(item, detection)
            wcs = item.get("_wcs")
            target_coordinate = item.get("_targetCoordinate")
            if wcs is not None and target_coordinate is not None:
                centroid = result["differenceImage"]
                signal = wcs.pixel_to_world(
                    float(centroid["centroidX"]), float(centroid["centroidY"])
                )
                east, north, separation = _world_offsets_arcsec(target_coordinate, signal)
                result["skyOffsetEastArcsec"] = east
                result["skyOffsetNorthArcsec"] = north
                result["targetSeparationArcsec"] = separation
                result["centroidSky"] = {
                    "raDeg": _finite(getattr(getattr(signal, "ra", None), "deg", None)),
                    "decDeg": _finite(getattr(getattr(signal, "dec", None), "deg", None)),
                }
            results.append(result)
        except TessArchiveTransientError:
            raise
        except Exception as error:
            errors.append({"sector": sector, "error": f"{type(error).__name__}: {error}"})
    return {
        "version": "openstar.tess-period-family-difference-imaging-run.v1",
        "execution": "coordinator-local-phase-difference-image-centroiding",
        "workerWorkload": None,
        "periodDetectionRecomputed": False,
        "sectorResults": results,
        "errors": errors,
    }


def _off_target_scatter(results: list[dict[str, Any]]) -> float | None:
    offsets = [
        (float(item["skyOffsetEastArcsec"]), float(item["skyOffsetNorthArcsec"]))
        for item in results
        if _finite(item.get("skyOffsetEastArcsec")) is not None
        and _finite(item.get("skyOffsetNorthArcsec")) is not None
    ]
    if not offsets:
        return None
    median_east = statistics.median(item[0] for item in offsets)
    median_north = statistics.median(item[1] for item in offsets)
    return statistics.median(
        math.hypot(east - median_east, north - median_north)
        for east, north in offsets
    )


def interpret_period_family_difference_imaging(
    preparation: dict[str, Any], run: dict[str, Any]
) -> dict[str, Any]:
    sectors = list(run.get("sectorResults") or [])
    target = [item for item in sectors if item.get("classification") == "TARGET_CONSISTENT"]
    off_target = [item for item in sectors if item.get("classification") == "OFF_TARGET"]
    ambiguous = [item for item in sectors if item.get("classification") == "AMBIGUOUS"]
    no_quality = [item for item in sectors if item.get("classification") == "NO_QUALITY_LOCALIZATION"]
    quality_count = len(target) + len(off_target) + len(ambiguous)
    required = max(MIN_CROSS_SECTOR_SUPPORT, quality_count // 2 + 1)
    scatter = _off_target_scatter(off_target)

    if len(target) >= required and not off_target:
        classification = "TARGET_PERIOD_FAMILY_SUPPORTED"
    elif (
        len(off_target) >= required
        and not target
        and scatter is not None
        and scatter <= MAX_OFF_TARGET_SKY_SCATTER_ARCSEC
    ):
        classification = "OFF_TARGET_PERIOD_FAMILY_SUPPORTED"
    elif len(target) >= 2 and len(off_target) >= 2:
        classification = "SOURCE_SWITCHING_BY_SECTOR"
    else:
        classification = "PERIOD_FAMILY_LOCALIZATION_UNRESOLVED"

    recommendations = {
        "TARGET_PERIOD_FAMILY_SUPPORTED": "UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION",
        "OFF_TARGET_PERIOD_FAMILY_SUPPORTED": "OFFSET_SOURCE_CATALOG_IDENTIFICATION",
        "SOURCE_SWITCHING_BY_SECTOR": "SOURCE_SWITCHING_TEMPORAL_MODEL",
        "PERIOD_FAMILY_LOCALIZATION_UNRESOLVED": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
    }
    return {
        "version": "openstar.tess-period-family-difference-imaging-interpretation.v1",
        "classification": classification,
        "claimDecision": {
            "claim": "HUMAN_REVIEW_REQUIRED",
            "rationale": [
                "Phase-difference imaging tests source location, not period recurrence or physical interpretation.",
                "The unresolved period-family claim therefore remains fail-closed regardless of localization outcome.",
            ],
        },
        "sourceAttributionResolved": classification in {
            "TARGET_PERIOD_FAMILY_SUPPORTED", "OFF_TARGET_PERIOD_FAMILY_SUPPORTED"
        },
        "variableSignalOrigin": (
            "TARGET" if classification == "TARGET_PERIOD_FAMILY_SUPPORTED"
            else "OFF_TARGET" if classification == "OFF_TARGET_PERIOD_FAMILY_SUPPORTED"
            else "MULTIPLE_OR_TIME_VARIABLE" if classification == "SOURCE_SWITCHING_BY_SECTOR"
            else "UNRESOLVED"
        ),
        "qualitySectorCount": quality_count,
        "requiredSupportCount": required,
        "targetSupportingSectors": sorted(int(item["sector"]) for item in target),
        "offTargetSectors": sorted(int(item["sector"]) for item in off_target),
        "ambiguousSectors": sorted(int(item["sector"]) for item in ambiguous),
        "noQualitySectors": sorted(int(item["sector"]) for item in no_quality),
        "offTargetSkyOffsetScatterArcsec": scatter,
        "maximumOffTargetSkyOffsetScatterArcsec": MAX_OFF_TARGET_SKY_SCATTER_ARCSEC,
        "sectorResults": sectors,
        "errors": list(run.get("errors") or []),
        "periodDetectionRecomputed": False,
        "periodFamilyResolved": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommendations[classification],
        "interpretationGuard": (
            "At least three consistent independent-sector centroids and no conflicting strong "
            "localization are required. Localization does not turn the period family into a confirmed physical period."
        ),
        "preparationSHA256": sha256_json(preparation),
    }
