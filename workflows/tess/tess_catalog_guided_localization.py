"""Coordinator-local, catalog-guided official-SPOC-PRF source localization."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_multisource_residual import _prewhiten_cube_raw
from .tess_offset_variability import _skycoord
from .tess_prf_refinement import (
    COHERENT_VECTOR_CHI2_95, MIN_SAMPLES, _calibration,
    _chi_square_survival_even, _weighted_hypothesis,
)
from .tess_prf_deblend import _background_columns
from .tess_residual_localization import (
    MAX_CADENCES, _background_subtract_cube, _download_tpf, _uniform_indices,
    _write_json,
)
from .tess_spoc_prf import (
    MAST_PRF_ROOT, MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE,
    SHARED_ASTROMETRIC_SHIFT_GRID, _drift_corrected_times, _fit_static_image,
    _render_prf_template, _tpf_detector_geometry,
)

HYPOTHESES = {
    "TARGET_ONLY": (0,), "CANDIDATE_1_ONLY": (1,), "CANDIDATE_2_ONLY": (2,),
    "TARGET_PLUS_CANDIDATE_1": (0, 1), "TARGET_PLUS_CANDIDATE_2": (0, 2),
    "CANDIDATE_1_PLUS_CANDIDATE_2": (1, 2), "TARGET_PLUS_BOTH": (0, 1, 2),
}


def prepare_catalog_guided_localization(*, evidence: dict[str, dict[str, Any]],
                                        output_dir: Path, investigation_id: str) -> dict[str, Any]:
    """Freeze the two stage-044 candidates and pre-existing residual model verbatim."""
    catalog = evidence["catalogCounterpartIdentification"]
    candidates = catalog.get("plausibleCatalogCandidates") or []
    if (catalog.get("recommendedNextTest") != "CATALOG_GUIDED_SOURCE_LOCALIZATION"
            or catalog.get("physicalMechanismResolved") is not False or len(candidates) < 2):
        raise RuntimeError("Catalog-guided localization requires the unresolved multi-candidate result.")
    if catalog.get("preferredCandidate") is not None:
        raise RuntimeError("Catalog-guided localization must not promote a preferred candidate during preparation.")
    decomposition = evidence["decompositionPreparation"]
    period = decomposition.get("referenceFamilyPeriodDays")
    orders = decomposition.get("subtractedHarmonicOrders")
    cycle = decomposition.get("physicalCycleResolved")
    residual = decomposition.get("residualModelProvenance") or {}
    frequency = residual.get("referenceFrequency")
    reference = residual.get("timeReferenceDays")
    drift = residual.get("fractionalFrequencyDriftPerDay")
    if period is None or orders != [1, 2, 3, 4] or cycle is not False:
        raise RuntimeError("Catalog-guided localization requires persisted unresolved [1,2,3,4] family evidence.")
    if None in (frequency, reference, drift):
        raise RuntimeError("Catalog-guided localization requires the persisted residual frequency/drift reference.")
    target = (evidence["prfPreparation"].get("targetSky") or {})
    if target.get("raDeg") is None or target.get("decDeg") is None:
        raise RuntimeError("Catalog-guided localization requires stage-041 target sky coordinates.")
    selected = []
    for index, candidate in enumerate(candidates[:2], 1):
        if candidate.get("raDeg") is None or candidate.get("decDeg") is None:
            raise RuntimeError("Every plausible catalog candidate requires persisted sky coordinates.")
        selected.append({"componentID": f"candidate-{index}", **candidate})
    sectors = sorted({int(x) for x in (evidence["prfPreparation"].get("sectors") or [])})
    if not sectors:
        sectors = sorted({int(x) for x in ((target.get("supportingSectors") or []))})
    root = Path(output_dir) / "catalog-guided-source-localization"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-catalog-guided-source-localization-preparation.v1",
        "investigationID": investigation_id, "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-small-coupled-spatial-fit", "workerProtocolUsed": False,
        "modelSource": "official-public-SPOC-TESS-PRF-FITS", "officialPRFRoot": MAST_PRF_ROOT,
        "ticID": evidence["targetPreparation"].get("ticID"), "sectors": sectors,
        "target": {"componentID": "target", "raDeg": float(target["raDeg"]), "decDeg": float(target["decDeg"])},
        "plausibleCatalogCandidates": selected, "preferredCandidate": None,
        "referenceFamilyPeriodDays": float(period), "subtractedHarmonicOrders": list(orders),
        "physicalCycleResolved": False, "residualReferenceFrequency": float(frequency),
        "residualTimeReferenceDays": float(reference), "fractionalFrequencyDriftPerDay": float(drift),
        "sourceHypotheses": list(HYPOTHESES),
        "provenance": {"catalogCandidates": "stage-044 persisted result",
                       "prfPreparation": "stage-041 persisted result",
                       "prfInterpretation": "stage-043 persisted result",
                       "familyResidualModel": "stage-038 persisted preparation"},
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def _comparison(coefficients: np.ndarray, covariances: np.ndarray,
                templates: np.ndarray, component_ids: list[str]) -> dict[str, Any]:
    observations = coefficients.reshape(-1)
    models = {name: _weighted_hypothesis(
        observations=observations, pixel_covariances=covariances,
        templates=templates[:, indices], component_ids=[component_ids[i] for i in indices])
        for name, indices in HYPOTHESES.items()}
    ranked = sorted(models, key=lambda name: models[name]["bic"])
    for name in models:
        models[name]["deltaBIC"] = float(models[name]["bic"] - models[ranked[0]]["bic"])
    best = models[ranked[0]]
    identifiable = bool(best["fullRank"] and all(x["individuallyIdentifiable"]
                                                  for x in best["sourceEstimates"]))
    return {"models": models, "ranking": ranked, "bestModel": ranked[0],
            "bestModelIdentifiable": identifiable}


def _calibrate_shared_astrometric_offset(*, corrected_cube: np.ndarray,
                                         valid_pixels: np.ndarray,
                                         source_models: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit one deterministic dx/dy shared by every fixed catalog source."""
    rows, cols = valid_pixels.shape
    median_image = np.nanmedian(np.asarray(corrected_cube, dtype=float), axis=0)
    trials = []
    best = None
    for dx in SHARED_ASTROMETRIC_SHIFT_GRID:
        for dy in SHARED_ASTROMETRIC_SHIFT_GRID:
            templates = [
                _render_prf_template(
                    image=item["image"], header=item["header"],
                    source_x=float(item["x"]) + float(dx),
                    source_y=float(item["y"]) + float(dy), rows=rows, cols=cols,
                    valid_pixels=valid_pixels,
                )
                for item in source_models
            ]
            design = np.column_stack([
                *templates, *_background_columns(rows, cols, valid_pixels)
            ])
            objective, coefficients, explained = _fit_static_image(
                design, median_image, len(source_models)
            )
            trial = {"dxPixels": float(dx), "dyPixels": float(dy),
                     "objectiveSSE": float(objective),
                     "explainedVariance": float(explained)}
            trials.append(trial)
            if math.isfinite(objective) and (best is None or objective < best["objectiveSSE"]):
                best = {**trial, "templates": templates,
                        "medianImageSourceCoefficients": coefficients[:len(source_models)].tolist()}
    finite_objectives = sorted(trial["objectiveSSE"] for trial in trials
                               if math.isfinite(trial["objectiveSSE"]))
    numerical_tolerance = (np.finfo(float).eps
                           * max(abs(finite_objectives[0]), 1.0) * 16.0
                           if finite_objectives else float("inf"))
    uniquely_determined = bool(
        len(finite_objectives) == 1
        or finite_objectives[1] - finite_objectives[0] > numerical_tolerance
    )
    if (best is None or not uniquely_determined
            or best["explainedVariance"] < MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE):
        raise RuntimeError(
            "No scientifically valid common astrometric offset fits the sector median image."
        )
    return {"method": "deterministic-shared-dx-dy-median-image-SSE",
            "sharedAcrossComponentIDs": [item["componentID"] for item in source_models],
            "dxPixels": best["dxPixels"], "dyPixels": best["dyPixels"],
            "objectiveSSE": best["objectiveSSE"],
            "uniqueObjectiveMinimum": True,
            "objectiveNumericalTieTolerance": float(numerical_tolerance),
            "explainedVariance": best["explainedVariance"],
            "minimumExplainedVariance": MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE,
            "medianImageSourceCoefficients": best["medianImageSourceCoefficients"],
            "objectiveTrials": trials, "templates": best["templates"]}


def _temporal_predictive_validation(*, times: np.ndarray, pixels: np.ndarray,
                                    coherent_basis: np.ndarray,
                                    templates: np.ndarray,
                                    component_ids: list[str],
                                    block_count: int = 4) -> dict[str, Any]:
    """Train source vectors on contiguous-fold complements and predict unseen samples."""
    blocks = [block for block in np.array_split(np.arange(len(times)), block_count)
              if len(block) >= 10]
    if len(blocks) < 2:
        return {"available": False, "reason": "insufficient contiguous temporal folds",
                "folds": [], "predictiveWinner": None, "consistent": False}
    folds = []
    all_indices = np.arange(len(times))
    for fold_index, held_indices in enumerate(blocks):
        train_indices = np.setdiff1d(all_indices, held_indices)
        train_design = np.column_stack((coherent_basis[train_indices],
                                        np.ones(len(train_indices))))
        train_coefficients, _, train_rank, train_singular = np.linalg.lstsq(
            train_design, pixels[train_indices], rcond=None
        )
        train_errors = pixels[train_indices] - train_design @ train_coefficients
        train_dof = len(train_indices) - int(train_rank)
        if train_dof <= 0:
            raise RuntimeError("Temporal predictive training fit has no residual degrees of freedom.")
        variances = np.sum(train_errors * train_errors, axis=0) / train_dof
        floor = np.finfo(float).eps * max(float(np.nanmedian(variances)), 1.0)
        variances = np.maximum(variances, floor)
        normal = np.linalg.pinv(train_design.T @ train_design, hermitian=True)[:2, :2]
        covariances = np.asarray([normal * float(value) for value in variances])
        training = _comparison(train_coefficients[:2].T, covariances,
                               templates, component_ids)

        # This independent held-out fit is diagnostic only. Model selection below
        # exclusively evaluates parameters frozen from the training interval.
        held_design = np.column_stack((coherent_basis[held_indices],
                                       np.ones(len(held_indices))))
        held_coefficients, _, held_rank, _ = np.linalg.lstsq(
            held_design, pixels[held_indices], rcond=None
        )
        held_normal = np.linalg.pinv(held_design.T @ held_design, hermitian=True)[:2, :2]
        held_errors = pixels[held_indices] - held_design @ held_coefficients
        held_dof = max(1, len(held_indices) - int(held_rank))
        held_variances = np.sum(held_errors * held_errors, axis=0) / held_dof
        held_covariances = np.asarray([
            held_normal * max(float(value), floor) for value in held_variances
        ])
        held_diagnostic = _comparison(held_coefficients[:2].T, held_covariances,
                                      templates, component_ids)

        records = {}
        log_normalization = float(np.sum(
            len(held_indices) * np.log(2.0 * math.pi * variances)
        ))
        for model_id, indices in HYPOTHESES.items():
            parameters = np.asarray(
                training["models"][model_id]["parameterEstimates"], dtype=float
            ).reshape(len(indices), 2)
            frozen_pixel_vectors = templates[:, indices] @ parameters
            prediction = (coherent_basis[held_indices] @ frozen_pixel_vectors.T
                          + train_coefficients[2][None, :])
            residual = pixels[held_indices] - prediction
            chi_square = float(np.sum(np.square(residual) / variances[None, :]))
            records[model_id] = {
                "trainingParameterEstimates": parameters.reshape(-1).tolist(),
                "frozenTrainingPixelVectors": frozen_pixel_vectors.tolist(),
                "heldOutChiSquare": chi_square,
                "heldOutLogLikelihood": -0.5 * (chi_square + log_normalization),
            }
        winner = max(records, key=lambda name: records[name]["heldOutLogLikelihood"])
        compatibility = {}
        for model_id in HYPOTHESES:
            train_sources = training["models"][model_id]["sourceEstimates"]
            held_sources = held_diagnostic["models"][model_id]["sourceEstimates"]
            source_records = []
            for train_source, held_source in zip(train_sources, held_sources):
                train_vector = np.asarray([train_source["sinA"], train_source["cosB"]])
                held_vector = np.asarray([held_source["sinA"], held_source["cosB"]])
                covariance = (np.asarray(train_source["covariance"])
                              + np.asarray(held_source["covariance"]))
                difference = held_vector - train_vector
                statistic = float(
                    difference @ np.linalg.pinv(covariance, hermitian=True) @ difference
                )
                source_records.append({
                    "componentID": train_source["componentID"],
                    "trainingVector": train_vector.tolist(),
                    "independentHeldOutVector": held_vector.tolist(),
                    "differenceChiSquare": statistic,
                    "degreesOfFreedom": 2,
                    "chiSquareSurvivalProbability": _chi_square_survival_even(statistic, 2),
                })
            compatibility[model_id] = source_records
        folds.append({
            "foldIndex": fold_index,
            "trainingTimeRanges": [
                {"start": float(times[block[0]]), "end": float(times[block[-1]])}
                for block in blocks if not np.array_equal(block, held_indices)
            ],
            "heldOutTimeRange": {"start": float(times[held_indices[0]]),
                                 "end": float(times[held_indices[-1]])},
            "trainingSampleCount": len(train_indices),
            "heldOutSampleCount": len(held_indices),
            "trainingTemporalRank": int(train_rank),
            "trainingTemporalSingularValues": train_singular.tolist(),
            "models": records, "predictiveWinner": winner,
            "sourceVectorTemporalCompatibility": compatibility,
            "independentHeldOutDiagnostic": {
                "bestModel": held_diagnostic["bestModel"],
                "sourceVectorsByHypothesis": {
                    name: value["sourceEstimates"]
                    for name, value in held_diagnostic["models"].items()
                },
            },
        })
    winners = [fold["predictiveWinner"] for fold in folds]
    consistent = len(set(winners)) == 1
    return {"available": True, "method": "contiguous-fold-frozen-parameter-prediction",
            "folds": folds, "predictiveWinner": winners[0] if consistent else None,
            "consistent": consistent}


def run_catalog_guided_localization(preparation: dict[str, Any]) -> dict[str, Any]:
    results, errors = [], []
    cache = Path(preparation["artifactRoot"]) / "official-prf-cache"
    definitions = [preparation["target"], *preparation["plausibleCatalogCandidates"]]
    component_ids = [x["componentID"] for x in definitions]
    for sector in preparation["sectors"]:
        try:
            tpf, source = _download_tpf(tic_id=int(preparation["ticID"]), sector=int(sector),
                                        ra_deg=float(definitions[0]["raDeg"]), dec_deg=float(definitions[0]["decDeg"]))
            times = np.asarray(tpf.time.value, dtype=float)
            cube = np.asarray(getattr(tpf.flux, "value", tpf.flux), dtype=float)
            keep = np.isfinite(times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
            times, cube = times[keep], cube[keep]
            indices = _uniform_indices(len(times), MAX_CADENCES); times, cube = times[indices], cube[indices]
            if len(times) < MIN_SAMPLES: raise RuntimeError(f"Only {len(times)} usable cadences.")
            corrected, _ = _background_subtract_cube(cube)
            residual, valid = _prewhiten_cube_raw(
                absolute_times=times, cube=corrected,
                physical_frequency=1.0 / preparation["referenceFamilyPeriodDays"],
                harmonic_orders=tuple(preparation["subtractedHarmonicOrders"]))
            camera, ccd, tpf_col, tpf_row = _tpf_detector_geometry(tpf)
            source_models = []
            for definition in definitions:
                coordinate = _skycoord(float(definition["raDeg"]), float(definition["decDeg"]))
                x, y = map(float, tpf.wcs.world_to_pixel(coordinate))
                image, header, files = _calibration(sector=int(sector), camera=camera, ccd=ccd,
                    row=tpf_row + y, column=tpf_col + x, cache=cache)
                source_models.append({"componentID": definition["componentID"],
                    "image": image, "header": header, "x": x, "y": y,
                    "catalogSky": {
                    "raDeg": definition["raDeg"], "decDeg": definition["decDeg"]},
                    "officialPRFModelFiles": files})
            shared = _calibrate_shared_astrometric_offset(
                corrected_cube=corrected, valid_pixels=valid,
                source_models=source_models,
            )
            templates = shared.pop("templates")
            calibration = []
            for item in source_models:
                calibration.append({"componentID": item["componentID"],
                    "catalogSky": item["catalogSky"],
                    "catalogPixelCenter": {"x": item["x"], "y": item["y"]},
                    "renderedPixelCenter": {
                        "x": item["x"] + shared["dxPixels"],
                        "y": item["y"] + shared["dyPixels"],
                    },
                    "officialPRFModelFiles": item["officialPRFModelFiles"]})
            valid_flat = valid.reshape(-1)
            template_matrix = np.column_stack(templates)[valid_flat]
            warped = _drift_corrected_times(times, time_reference_days=preparation["residualTimeReferenceDays"],
                fractional_frequency_drift_per_day=preparation["fractionalFrequencyDriftPerDay"])
            phase = 2 * math.pi * preparation["residualReferenceFrequency"] * warped
            temporal = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(phase))))
            pixels = residual.reshape(len(times), -1)[:, valid_flat]
            coefficients, _, rank, _ = np.linalg.lstsq(temporal, pixels, rcond=None)
            noise = pixels - temporal @ coefficients; dof = len(times) - rank
            variances = np.sum(noise * noise, axis=0) / dof
            normal = np.linalg.pinv(temporal.T @ temporal, hermitian=True)[:2, :2]
            floor = np.finfo(float).eps * max(float(np.nanmedian(variances)), 1.0)
            covariances = np.asarray([normal * max(float(v), floor) for v in variances])
            comparison = _comparison(coefficients[:2].T, covariances, template_matrix, component_ids)
            predictive = _temporal_predictive_validation(
                times=times, pixels=pixels, coherent_basis=temporal[:, :2],
                templates=template_matrix, component_ids=component_ids,
            )
            decisive = bool(
                comparison["bestModelIdentifiable"]
                and predictive["available"] and predictive["consistent"]
                and predictive["predictiveWinner"] == comparison["bestModel"]
            )
            results.append({"sector": int(sector), "source": source, "componentIDs": component_ids,
                "calibration": calibration,
                "sharedAstrometricCalibration": {**shared, "executed": True,
                    "relativeCatalogPositionsFixed": True,
                    "independentSourceMotionAllowed": False},
                "models": comparison["models"], "modelRanking": comparison["ranking"],
                "bestModel": comparison["bestModel"], "bestModelIdentifiable": comparison["bestModelIdentifiable"],
                "heldOutTemporalValidation": predictive,
                "decisive": bool(decisive), "officialPRFFiles": [f for c in calibration for f in c["officialPRFModelFiles"]],
                "renderedComponentIDs": component_ids})
        except Exception as exc:
            errors.append({"sector": int(sector), "error": f"{type(exc).__name__}: {exc}"})
    return {"version": "openstar.tess-catalog-guided-source-localization-run.v1",
            "sectorResults": results, "errors": errors, "workerProtocolUsed": False}


def interpret_catalog_guided_localization(preparation: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    decisive = [x for x in run.get("sectorResults", []) if x.get("decisive")]
    winners = [x["bestModel"] for x in decisive]
    consistent = winners[0] if len(winners) >= 2 and len(set(winners)) == 1 else None
    candidate_number = None
    if consistent in {"CANDIDATE_1_ONLY", "TARGET_PLUS_CANDIDATE_1"}: candidate_number = 1
    if consistent in {"CANDIDATE_2_ONLY", "TARGET_PLUS_CANDIDATE_2"}: candidate_number = 2
    preferred = (preparation["plausibleCatalogCandidates"][candidate_number - 1]
                 if candidate_number else None)
    if preferred:
        classification = "CATALOG_CANDIDATE_LOCALIZED"
        recommended = "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
    elif len(set(winners)) > 1:
        classification = "CATALOG_SOURCE_SWITCHING_BY_SECTOR"; recommended = "SOURCE_SWITCHING_TEMPORAL_MODEL"
    elif consistent in {"CANDIDATE_1_PLUS_CANDIDATE_2", "TARGET_PLUS_BOTH"}:
        classification = "MULTIPLE_CATALOG_SOURCES_CONTRIBUTE"; recommended = "JOINT_MULTI_SOURCE_VARIABILITY_VALIDATION"
    else:
        classification = "CATALOG_GUIDED_LOCALIZATION_UNRESOLVED"; recommended = "HIGHER_RESOLUTION_SPATIAL_FOLLOWUP"
    return {"version": "openstar.tess-catalog-guided-source-localization-interpretation.v1",
        "sourceHypotheses": list(HYPOTHESES), "sectorResults": run.get("sectorResults", []),
        "errors": run.get("errors", []), "classification": classification,
        "preferredCandidate": preferred, "counterpartIdentified": preferred is not None,
        "physicalMechanismResolved": False, "physicalCycleResolved": False,
        "referenceFamilyPeriodDays": preparation["referenceFamilyPeriodDays"],
        "subtractedHarmonicOrders": preparation["subtractedHarmonicOrders"],
        "recommendedNextTest": recommended,
        "modelSelection": {"metric": "Gaussian block-weighted likelihood BIC",
            "coherentVectorChiSquare95": COHERENT_VECTOR_CHI2_95,
            "identifiability": "full numerical rank and conditional coherent-vector significance",
            "heldOutTemporalValidation": (
                "contiguous folds with training source vectors frozen during held-out "
                "Gaussian likelihood evaluation; no new score cutoff"
            )}}
