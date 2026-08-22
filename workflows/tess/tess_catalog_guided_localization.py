"""Catalog-guided, three-source localization of an unresolved TESS residual.

This continuation is intentionally coordinator local.  Catalog candidates are
scientific hypotheses, not worker instructions, and are therefore kept out of
generic work-unit payloads.  All temporal validation freezes parameters fitted
on the training complement before scoring a contiguous held-out fold.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .tess_prf_refinement import (
    _coherent_pixel_fit,
    _prewhitened_coherent_basis,
    _weighted_hypothesis,
)
from .tess_residual_localization import _write_json


HARMONIC_ORDERS = (1, 2, 3, 4)
COMPONENT_IDS = ("target", "candidate-1", "candidate-2")
MODEL_COMPONENTS = {
    "TARGET_ONLY": (0,),
    "CANDIDATE_1_ONLY": (1,),
    "CANDIDATE_2_ONLY": (2,),
    "TARGET_PLUS_CANDIDATE_1": (0, 1),
    "TARGET_PLUS_CANDIDATE_2": (0, 2),
    "CANDIDATE_1_PLUS_CANDIDATE_2": (1, 2),
    "TARGET_PLUS_BOTH": (0, 1, 2),
}


def _compare_hypotheses(coefficients: np.ndarray, covariances: np.ndarray,
                        templates: np.ndarray) -> dict[str, Any]:
    observations = np.asarray(coefficients, dtype=float).reshape(-1)
    models = {}
    for model_id, indices in MODEL_COMPONENTS.items():
        models[model_id] = _weighted_hypothesis(
            observations=observations,
            pixel_covariances=np.asarray(covariances, dtype=float),
            templates=np.asarray(templates, dtype=float)[:, indices],
            component_ids=[COMPONENT_IDS[index] for index in indices],
        )
    winner = min(models, key=lambda item: models[item]["bic"])
    for model_id, model in models.items():
        model["deltaBIC"] = float(model["bic"] - models[winner]["bic"])
    selected = models[winner]
    identifiable = bool(selected["fullRank"] and all(
        source["individuallyIdentifiable"] for source in selected["sourceEstimates"]
    ))
    return {"models": models, "bestModel": winner,
            "bestModelIdentifiable": identifiable}


def _source_vector_compatibility(folds: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    """Test whether each relevant source has one common coherent vector."""
    relevant = MODEL_COMPONENTS[model_id]
    by_source: dict[str, Any] = {}
    all_compatible = True
    for source_index in relevant:
        component_id = COMPONENT_IDS[source_index]
        estimates = []
        for fold in folds:
            source = next((item for item in fold["independentHeldOutSourceEstimates"]
                           if item["componentID"] == component_id), None)
            if source is not None:
                estimates.append((np.asarray([source["sinA"], source["cosB"]]),
                                  np.asarray(source["covariance"])))
        if len(estimates) < 2 or any(np.linalg.matrix_rank(cov) < 2 for _, cov in estimates):
            record = {"available": False, "compatible": False,
                      "reason": "fewer than two identifiable block vectors"}
        else:
            precision = sum((np.linalg.inv(cov) for _, cov in estimates), np.zeros((2, 2)))
            covariance = np.linalg.inv(precision)
            vector = covariance @ sum((np.linalg.inv(cov) @ value
                                       for value, cov in estimates), np.zeros(2))
            statistic = float(sum((value - vector) @ np.linalg.inv(cov) @ (value - vector)
                                  for value, cov in estimates))
            dof = 2 * (len(estimates) - 1)
            # dof is even for two-dimensional source vectors.
            x = statistic / 2.0
            p_value = float(math.exp(-x) * sum(x ** k / math.factorial(k)
                                               for k in range(dof // 2)))
            record = {"available": True, "compatible": p_value >= 0.05,
                      "commonVector": vector.tolist(), "covariance": covariance.tolist(),
                      "heterogeneityStatistic": statistic, "degreesOfFreedom": dof,
                      "pValue": p_value, "compatibilityLevel": 0.95}
        by_source[component_id] = record
        all_compatible = all_compatible and bool(record["compatible"])
    return {"bySource": by_source, "relevantSources": [COMPONENT_IDS[i] for i in relevant],
            "compatible": all_compatible}


def _temporal_predictive_validation(*, times: np.ndarray, prewhitened: np.ndarray,
                                    valid: np.ndarray, templates: np.ndarray,
                                    residual_frequency: float, time_reference: float,
                                    drift: float, block_count: int = 4,
                                    coherent_basis: np.ndarray | None = None) -> dict[str, Any]:
    """Select from all seven hypotheses using aggregate frozen held-out evidence."""
    times = np.asarray(times, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    templates = np.asarray(templates, dtype=float)
    folds_indices = [part for part in np.array_split(np.arange(len(times)), block_count)
                     if len(part) >= 10]
    if len(folds_indices) < 2:
        raise RuntimeError("Predictive validation requires at least two contiguous folds.")
    totals = {model_id: 0.0 for model_id in MODEL_COMPONENTS}
    folds = []
    for fold_index, held_out in enumerate(folds_indices):
        training = np.setdiff1d(np.arange(len(times)), held_out)
        kwargs = {"frequency": residual_frequency, "time_reference": time_reference,
                  "drift": drift}
        train_fit = _coherent_pixel_fit(
            times=times[training], cube=prewhitened[training][:, valid], **kwargs,
            coherent_basis=None if coherent_basis is None else coherent_basis[training])
        test_fit = _coherent_pixel_fit(
            times=times[held_out], cube=prewhitened[held_out][:, valid], **kwargs,
            coherent_basis=None if coherent_basis is None else coherent_basis[held_out])
        train_models = _compare_hypotheses(train_fit["coefficients"],
                                           train_fit["covariances"], templates)
        held_out_models = _compare_hypotheses(test_fit["coefficients"],
                                              test_fit["covariances"], templates)
        observed = test_fit["coefficients"].reshape(-1)
        inverses = [np.linalg.pinv(item, hermitian=True) for item in test_fit["covariances"]]
        determinants = [np.linalg.slogdet(item) for item in test_fit["covariances"]]
        if any(sign <= 0 for sign, _ in determinants):
            raise RuntimeError("Held-out coherent covariance is not positive definite.")
        constant = float(sum(value for _, value in determinants)
                         + observed.size * math.log(2.0 * math.pi))
        records = {}
        for model_id, indices in MODEL_COMPONENTS.items():
            parameters = np.asarray(train_models["models"][model_id]["parameterEstimates"])
            prediction = (templates[:, indices, None]
                          * parameters.reshape(len(indices), 2)[None, :, :]).sum(axis=1)
            residual = test_fit["coefficients"] - prediction
            chi_square = float(sum(value @ inverse @ value
                                   for value, inverse in zip(residual, inverses)))
            log_likelihood = -0.5 * (chi_square + constant)
            records[model_id] = {
                "trainingParameterEstimates": parameters.tolist(),
                "heldOutChiSquare": chi_square,
                "heldOutLogLikelihood": log_likelihood,
            }
            totals[model_id] += log_likelihood
        folds.append({
            "foldIndex": fold_index,
            "trainingSampleCount": int(len(training)),
            "heldOutSampleCount": int(len(held_out)),
            "heldOutTimeRange": {"start": float(times[held_out[0]]),
                                 "end": float(times[held_out[-1]])},
            "models": records,
            # This refit is diagnostic only and never enters predictive selection.
            "independentHeldOutBestModel": held_out_models["bestModel"],
            "independentHeldOutSourceEstimates": held_out_models["models"]
                ["TARGET_PLUS_BOTH"]["sourceEstimates"],
        })
    winner = max(totals, key=totals.get)
    compatibility = _source_vector_compatibility(folds, winner)
    return {
        "method": "aggregate contiguous-fold frozen-training-parameter Gaussian predictive likelihood",
        "folds": folds,
        "totalHeldOutLogLikelihoodByModel": totals,
        "predictiveModel": winner,
        "sourceVectorTemporalCompatibility": compatibility,
        "predictiveSupport": True,
        "interpretationGuard": (
            "The predictive model is selected from summed held-out evidence; it need not win "
            "every independently refitted fold."),
    }


def _fit_shared_astrometric_shift(*, observations: np.ndarray, covariances: np.ndarray,
                                  render_templates: Callable[[float, float], np.ndarray],
                                  shift_grid: tuple[float, ...] = (-0.2, 0.0, 0.2)) -> dict[str, Any]:
    """Calibrate exactly one detector dx/dy shared by all three catalog sources."""
    trials = []
    for dx in shift_grid:
        for dy in shift_grid:
            templates = np.asarray(render_templates(float(dx), float(dy)), dtype=float)
            if templates.ndim != 2 or templates.shape[1] != 3:
                raise ValueError("PRF renderer must return target + two candidate templates.")
            comparison = _compare_hypotheses(observations, covariances, templates)
            trials.append((comparison["models"]["TARGET_PLUS_BOTH"]["chiSquare"], dx, dy,
                           templates, comparison))
    _, dx, dy, templates, comparison = min(trials, key=lambda item: item[0])
    return {"sharedAstrometricCalibration": {"dxPixels": float(dx), "dyPixels": float(dy),
            "appliedToComponents": list(COMPONENT_IDS), "independentSourceMotion": False},
            "templates": templates, "comparison": comparison}


def analyze_catalog_guided_sector(*, sector: int, times: np.ndarray,
                                  prewhitened: np.ndarray, valid: np.ndarray,
                                  render_templates: Callable[[float, float], np.ndarray],
                                  residual_frequency: float, time_reference: float,
                                  drift: float, physical_frequency: float,
                                  block_count: int = 4) -> dict[str, Any]:
    """Run the real coupled three-source fit for one sector."""
    times = np.asarray(times, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    coherent_basis = _prewhitened_coherent_basis(
        times=times, physical_frequency=float(physical_frequency),
        residual_frequency=float(residual_frequency), time_reference=float(time_reference),
        drift=float(drift), harmonic_orders=HARMONIC_ORDERS)
    fit = _coherent_pixel_fit(
        times=times, cube=np.asarray(prewhitened, dtype=float)[:, valid],
        frequency=float(residual_frequency), time_reference=float(time_reference),
        drift=float(drift), coherent_basis=coherent_basis)
    calibrated = _fit_shared_astrometric_shift(
        observations=fit["coefficients"], covariances=fit["covariances"],
        render_templates=render_templates)
    temporal = _temporal_predictive_validation(
        times=times, prewhitened=np.asarray(prewhitened, dtype=float), valid=valid,
        templates=calibrated["templates"], residual_frequency=float(residual_frequency),
        time_reference=float(time_reference), drift=float(drift), block_count=block_count,
        coherent_basis=coherent_basis)
    return {
        "sector": int(sector),
        "sharedAstrometricCalibration": calibrated["sharedAstrometricCalibration"],
        "fullDataComparison": calibrated["comparison"],
        "temporalPredictiveValidation": temporal,
        "subtractedHarmonicOrders": list(HARMONIC_ORDERS),
        "physicalCycleResolved": False,
    }


def run_catalog_guided_localization(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Execute supplied calibrated sector inputs locally (never through workers)."""
    results = []
    for item in sector_inputs:
        renderer = item.get("renderTemplates")
        if not callable(renderer):
            raise RuntimeError("Each sector input requires an official-SPOC PRF renderer.")
        results.append(analyze_catalog_guided_sector(
            sector=int(item["sector"]), times=item["times"],
            prewhitened=item["prewhitened"], valid=item["valid"],
            render_templates=renderer,
            residual_frequency=float(preparation["residualReferenceFrequency"]),
            time_reference=float(preparation["residualTimeReferenceDays"]),
            drift=float(preparation["fractionalFrequencyDriftPerDay"]),
            physical_frequency=float(item["physicalFrequency"]),
            block_count=int(item.get("blockCount", 4))))
    return {"version": "openstar.tess-catalog-guided-source-localization-run.v1",
            "execution": "coordinator-local-small-coupled-spatial-fit",
            "sectorResults": results, "physicalCycleResolved": False}


def prepare_catalog_guided_localization(*, catalog_summary: dict[str, Any],
                                        prf_preparation: dict[str, Any], output_dir: Path,
                                        investigation_id: str) -> dict[str, Any]:
    candidates = list(catalog_summary.get("plausibleCatalogCandidates") or [])
    if (catalog_summary.get("recommendedNextTest") != "CATALOG_GUIDED_SOURCE_LOCALIZATION"
            or catalog_summary.get("physicalMechanismResolved") is not False
            or len(candidates) < 2):
        raise RuntimeError("Catalog-guided localization requires two unresolved candidates.")
    root = Path(output_dir) / "catalog-guided-source-localization"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-catalog-guided-source-localization.v1",
        "investigationID": investigation_id, "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-small-coupled-spatial-fit",
        "target": prf_preparation.get("target"), "catalogCandidates": candidates[:2],
        "preferredCandidate": None, "componentIDs": list(COMPONENT_IDS),
        "modelHypotheses": list(MODEL_COMPONENTS),
        "subtractedHarmonicOrders": list(HARMONIC_ORDERS),
        "physicalCycleResolved": False,
        "residualReferenceFrequency": prf_preparation.get("residualReferenceFrequency"),
        "residualTimeReferenceDays": prf_preparation.get("residualTimeReferenceDays"),
        "fractionalFrequencyDriftPerDay": prf_preparation.get("fractionalFrequencyDriftPerDay"),
        "priorEvidence": {"catalogCounterpart": catalog_summary,
                          "officialSPOCPRFPreparation": prf_preparation},
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def interpret_catalog_guided_localization(preparation: dict[str, Any],
                                          run: dict[str, Any]) -> dict[str, Any]:
    sectors = list(run.get("sectorResults") or [])
    decisive = []
    for sector in sectors:
        full = sector["fullDataComparison"]
        temporal = sector["temporalPredictiveValidation"]
        same = temporal["predictiveModel"] == full["bestModel"]
        compatible = bool(temporal["sourceVectorTemporalCompatibility"]["compatible"])
        is_decisive = bool(full["bestModelIdentifiable"] and same and compatible)
        sector["decisive"] = is_decisive
        sector["decisivenessCriteria"] = {
            "fullDataModelIdentifiable": bool(full["bestModelIdentifiable"]),
            "aggregateFrozenHeldOutSupportsFullDataAttribution": same,
            "relevantSourceVectorTemporalCompatibilityPasses": compatible,
        }
        if is_decisive:
            decisive.append(sector)
    attributions = {item["fullDataComparison"]["bestModel"] for item in decisive}
    resolved = bool(decisive and len(decisive) == len(sectors) and len(attributions) == 1)
    return {
        "version": "openstar.tess-catalog-guided-source-localization-interpretation.v1",
        "sectorResults": sectors, "decisiveSectorCount": len(decisive),
        "classification": "CONSISTENT_SOURCE_ATTRIBUTION" if resolved else "UNRESOLVED",
        "preferredModel": next(iter(attributions)) if resolved else None,
        "physicalCycleResolved": False, "sourceAttributionResolved": resolved,
        "recommendedNextTest": None if resolved else "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
        "claimLevelChanged": resolved,
        "interpretationGuard": "Unresolved aggregate predictive evidence must not finalize attribution.",
    }
