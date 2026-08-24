"""Catalog-guided, three-source localization of an unresolved TESS residual.

This continuation is intentionally coordinator local.  Catalog candidates are
scientific hypotheses, not worker instructions, and are therefore kept out of
generic work-unit payloads.  All temporal validation freezes parameters fitted
on the training complement before scoring a contiguous held-out fold.
"""
from __future__ import annotations

import math
import itertools
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .tess_prf_refinement import (
    _coherent_pixel_fit,
    _prewhitened_coherent_basis,
    _weighted_hypothesis,
)
from .tess_multisource_residual import _prewhiten_cube_raw
from .tess_offset_variability import _skycoord
from .tess_prf_deblend import _background_columns
from .tess_residual_localization import (
    MAX_CADENCES, _background_subtract_cube, _download_tpf, _uniform_indices, _write_json,
)
from .tess_spoc_prf import (
    MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE, SHARED_ASTROMETRIC_SHIFT_GRID,
    _fit_static_image,
    _list_official_prf_grid, _official_prf_at_detector_position,
    _render_prf_template, _tpf_detector_geometry,
)


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


def generate_source_hypotheses(component_ids: list[str] | tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Return every non-empty source subset in stable cardinality/input order.

    The legacy three-source constants above intentionally remain untouched.  New
    continuations may supply a bounded frozen catalog without changing serialized
    model identifiers produced for old callers.
    """
    ids = tuple(str(value) for value in component_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("component IDs must be a non-empty unique sequence")
    return {
        "SOURCE_SUBSET_" + "__".join(ids[index] for index in indices): indices
        for size in range(1, len(ids) + 1)
        for indices in itertools.combinations(range(len(ids)), size)
    }


def compare_source_hypotheses(coefficients: np.ndarray, covariances: np.ndarray,
                              templates: np.ndarray, component_ids: list[str] | tuple[str, ...]
                              ) -> dict[str, Any]:
    """Fit exhaustive hypotheses and require attribution in the complete model."""
    ids = tuple(component_ids)
    hypotheses = generate_source_hypotheses(ids)
    observations = np.asarray(coefficients, dtype=float).reshape(-1)
    template_array = np.asarray(templates, dtype=float)
    if template_array.ndim != 2 or template_array.shape[1] != len(ids):
        raise ValueError("template count must equal the supplied source count")
    models = {}
    for model_id, indices in hypotheses.items():
        models[model_id] = _weighted_hypothesis(
            observations=observations,
            pixel_covariances=np.asarray(covariances, dtype=float),
            templates=template_array[:, indices],
            component_ids=[ids[index] for index in indices])
        models[model_id]["sourceIDs"] = [ids[index] for index in indices]
    winner = min(models, key=lambda key: models[key]["bic"])
    for model in models.values():
        model["deltaBIC"] = float(model["bic"] - models[winner]["bic"])
    all_model_id = next(reversed(hypotheses))
    complete = models[all_model_id]
    conditional = {row["componentID"]: row for row in complete["sourceEstimates"]}
    selected = models[winner]
    selected_ids = selected["sourceIDs"]
    identifiable = bool(
        selected["fullRank"] and complete["fullRank"]
        and all(row["individuallyIdentifiable"] for row in selected["sourceEstimates"])
        and all(conditional[source_id]["individuallyIdentifiable"] for source_id in selected_ids))
    return {"models": models, "bestModel": winner,
            "bestModelSourceIDs": selected_ids,
            "bestModelIdentifiable": identifiable,
            "conditionalIdentifiabilityModel": all_model_id,
            "completeModelFullRank": bool(complete["fullRank"]),
            "conditionallyIdentifiableSources": sorted(
                source_id for source_id, row in conditional.items()
                if row["individuallyIdentifiable"])}


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
    selected_indices = set(MODEL_COMPONENTS[winner])
    conditional = models["TARGET_PLUS_BOTH"]
    conditional_sources = {
        item["componentID"]: item for item in conditional["sourceEstimates"]}
    # Attribution is conditional on every plausible source. A source that is
    # significant only after its close competitor is omitted is not localized.
    identifiable = bool(
        selected["fullRank"] and conditional["fullRank"]
        and all(source["individuallyIdentifiable"] for source in selected["sourceEstimates"])
        and all(conditional_sources[COMPONENT_IDS[index]]["individuallyIdentifiable"]
                for index in selected_indices)
        and all(not conditional_sources[COMPONENT_IDS[index]]["individuallyIdentifiable"]
                for index in set(range(3)) - selected_indices))
    return {"models": models, "bestModel": winner,
            "bestModelIdentifiable": identifiable,
            "conditionalIdentifiabilityModel": "TARGET_PLUS_BOTH"}


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


def _fit_shared_astrometric_shift(*, calibration_image: np.ndarray,
                                  background_columns: list[np.ndarray],
                                  render_templates: Callable[[float, float], np.ndarray],
                                  shift_grid: tuple[float, ...] = SHARED_ASTROMETRIC_SHIFT_GRID
                                  ) -> dict[str, Any]:
    """Calibrate one shared dx/dy from the static sector image, independently of attribution."""
    trials = []
    for dx in shift_grid:
        for dy in shift_grid:
            templates = np.asarray(render_templates(float(dx), float(dy)), dtype=float)
            if templates.ndim != 2 or templates.shape[1] != 3:
                raise ValueError("PRF renderer must return target + two candidate templates.")
            design = np.column_stack([templates, *background_columns])
            objective, coefficients, explained = _fit_static_image(
                design, np.asarray(calibration_image, dtype=float), len(COMPONENT_IDS))
            trials.append((objective, dx, dy, templates, coefficients, explained))
    finite = [item for item in trials if math.isfinite(float(item[0]))]
    if not finite:
        return {"available": False, "reason": "no finite static-image calibration objective"}
    minimum = min(float(item[0]) for item in finite)
    minima = [item for item in finite if np.isclose(float(item[0]), minimum)]
    if len(minima) != 1:
        return {"available": False, "reason": "shared astrometric objective has no unique minimum",
                "minimumObjective": minimum, "tiedMinimumCount": len(minima)}
    _, dx, dy, templates, coefficients, explained = minima[0]
    if float(explained) < MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE:
        return {"available": False, "reason": "official PRF static-image calibration inadequate",
                "explainedVariance": float(explained),
                "minimumExplainedVariance": MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE}
    return {"available": True,
            "sharedAstrometricCalibration": {"dxPixels": float(dx), "dyPixels": float(dy),
            "appliedToComponents": list(COMPONENT_IDS), "independentSourceMotion": False,
            "objective": minimum, "explainedVariance": float(explained),
            "minimumExplainedVariance": MIN_OFFICIAL_PRF_EXPLAINED_VARIANCE,
            "sourceFluxCoefficients": coefficients[:3].tolist()},
            "templates": templates}


def analyze_catalog_guided_sector(*, sector: int, times: np.ndarray,
                                  prewhitened: np.ndarray, valid: np.ndarray,
                                  calibration_image: np.ndarray,
                                  background_columns: list[np.ndarray],
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
        calibration_image=calibration_image, background_columns=background_columns,
        render_templates=render_templates)
    if not calibrated.get("available"):
        return {"sector": int(sector), "calibrationResolved": False,
                "calibrationFailure": calibrated,
                "fullDataComparison": {"bestModel": None, "bestModelIdentifiable": False},
                "temporalPredictiveValidation": {
                    "predictiveModel": None,
                    "sourceVectorTemporalCompatibility": {"compatible": False}},
                "subtractedHarmonicOrders": list(HARMONIC_ORDERS),
                "physicalCycleResolved": False}
    comparison = _compare_hypotheses(
        fit["coefficients"], fit["covariances"], calibrated["templates"])
    temporal = _temporal_predictive_validation(
        times=times, prewhitened=np.asarray(prewhitened, dtype=float), valid=valid,
        templates=calibrated["templates"], residual_frequency=float(residual_frequency),
        time_reference=float(time_reference), drift=float(drift), block_count=block_count,
        coherent_basis=coherent_basis)
    return {
        "sector": int(sector),
        "sharedAstrometricCalibration": calibrated["sharedAstrometricCalibration"],
        "calibrationResolved": True, "fullDataComparison": comparison,
        "temporalPredictiveValidation": temporal,
        "subtractedHarmonicOrders": list(HARMONIC_ORDERS),
        "physicalCycleResolved": False,
    }


def _prewhiten_production_cube(*, times: np.ndarray, corrected: np.ndarray,
                                reference_family_period_days: float,
                                harmonic_orders: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Prewhiten acquisition pixels with the caller's frozen harmonic evidence."""
    return _prewhiten_cube_raw(
        absolute_times=times, cube=corrected,
        physical_frequency=1.0 / float(reference_family_period_days),
        harmonic_orders=tuple(harmonic_orders))


def _production_sector_inputs(
    preparation: dict[str, Any], *, harmonic_orders: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Acquire TPF/WCS and official PRFs from the persisted scientific evidence."""
    orders = HARMONIC_ORDERS if harmonic_orders is None else tuple(harmonic_orders)
    target_sky = preparation["targetSky"]
    candidates = preparation["catalogCandidates"]
    coordinates = [
        _skycoord(float(target_sky["raDeg"]), float(target_sky["decDeg"])),
        *[_skycoord(float(item["raDeg"]), float(item["decDeg"])) for item in candidates],
    ]
    inputs = []
    cache_root = Path(preparation["artifactRoot"]) / "official-prf-cache"
    for sector in preparation["sectors"]:
        tpf, source = _download_tpf(
            tic_id=int(preparation["ticID"]), sector=int(sector),
            ra_deg=float(target_sky["raDeg"]), dec_deg=float(target_sky["decDeg"]))
        times = np.asarray(tpf.time.value, dtype=float)
        cube = np.asarray(getattr(tpf.flux, "value", tpf.flux), dtype=float)
        keep = np.isfinite(times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
        times, cube = times[keep], cube[keep]
        indices = _uniform_indices(len(times), MAX_CADENCES)
        times, cube = times[indices], cube[indices]
        if len(times) < 100:
            raise RuntimeError(f"Sector {sector} has only {len(times)} usable cadences.")
        corrected, background = _background_subtract_cube(cube)
        residual, valid = _prewhiten_production_cube(
            times=times, corrected=corrected,
            reference_family_period_days=float(preparation["referenceFamilyPeriodDays"]),
            harmonic_orders=orders)
        rows, cols = valid.shape
        centers = []
        for component_id, coordinate in zip(COMPONENT_IDS, coordinates):
            x, y = tpf.wcs.world_to_pixel(coordinate)
            if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                raise RuntimeError(f"{component_id} has no finite WCS position in sector {sector}.")
            centers.append({"componentID": component_id, "x": float(x), "y": float(y)})
        camera, ccd, tpf_col, tpf_row = _tpf_detector_geometry(tpf)
        grid = _list_official_prf_grid(sector=int(sector), camera=camera, ccd=ccd)
        models = []
        for center in centers:
            image, header, files = _official_prf_at_detector_position(
                sector=int(sector), camera=camera, ccd=ccd,
                detector_row=tpf_row + center["y"], detector_col=tpf_col + center["x"],
                archive_cache=cache_root / f"sector-{int(sector):04d}", grid_entries=grid)
            models.append({**center, "image": image, "header": header, "modelFiles": files})
        valid_flat = valid.reshape(-1)
        median_image = np.nanmedian(corrected, axis=0).reshape(-1)[valid_flat]
        static_background = [column[valid_flat]
                             for column in _background_columns(rows, cols, valid)]

        def render(dx: float, dy: float, *, source_models=models,
                   mask=valid, selection=valid_flat) -> np.ndarray:
            return np.column_stack([
                _render_prf_template(
                    image=model["image"], header=model["header"],
                    source_x=model["x"] + dx, source_y=model["y"] + dy,
                    rows=mask.shape[0], cols=mask.shape[1], valid_pixels=mask)
                for model in source_models
            ])[selection]

        inputs.append({
            "sector": int(sector), "times": times, "prewhitened": residual,
            "valid": valid, "renderTemplates": render,
            "calibrationImage": median_image, "backgroundColumns": static_background,
            "physicalFrequency": 1.0 / float(preparation["referenceFamilyPeriodDays"]),
            "acquisitionProvenance": {"tpf": source, "backgroundSubtraction": background,
                                      "componentPixelCenters": centers,
                                      "officialPRFModels": [item["modelFiles"] for item in models],
                                      "subtractedHarmonicOrders": list(orders)},
        })
    return inputs


def run_catalog_guided_localization(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Acquire and execute calibrated sector inputs locally (never through workers)."""
    if sector_inputs is None:
        sector_inputs = _production_sector_inputs(preparation)
    results = []
    for item in sector_inputs:
        renderer = item.get("renderTemplates")
        if not callable(renderer):
            raise RuntimeError("Each sector input requires an official-SPOC PRF renderer.")
        result = analyze_catalog_guided_sector(
            sector=int(item["sector"]), times=item["times"],
            prewhitened=item["prewhitened"], valid=item["valid"],
            calibration_image=item["calibrationImage"],
            background_columns=item["backgroundColumns"],
            render_templates=renderer,
            residual_frequency=float(preparation["residualReferenceFrequency"]),
            time_reference=float(preparation["residualTimeReferenceDays"]),
            drift=float(preparation["fractionalFrequencyDriftPerDay"]),
            physical_frequency=float(item["physicalFrequency"]),
            block_count=int(item.get("blockCount", 4)))
        result["acquisitionProvenance"] = item.get("acquisitionProvenance")
        results.append(result)
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
        "ticID": prf_preparation.get("ticID"),
        "targetSky": prf_preparation.get("targetSky"),
        "sectors": list(prf_preparation.get("sectors") or []),
        "referenceFamilyPeriodDays": prf_preparation.get(
            "referenceFamilyPeriodDays", prf_preparation.get("physicalPeriodDays")),
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
    consistent = bool(decisive and len(decisive) == len(sectors) and len(attributions) == 1)
    preferred_by_model = {
        "CANDIDATE_1_ONLY": 0, "TARGET_PLUS_CANDIDATE_1": 0,
        "CANDIDATE_2_ONLY": 1, "TARGET_PLUS_CANDIDATE_2": 1,
    }
    model = next(iter(attributions)) if consistent else None
    candidate_index = preferred_by_model.get(model)
    candidates = preparation.get("catalogCandidates") or []
    preferred = (dict(candidates[candidate_index])
                 if candidate_index is not None and candidate_index < len(candidates) else None)
    resolved = preferred is not None
    return {
        "version": "openstar.tess-catalog-guided-source-localization-interpretation.v1",
        "sectorResults": sectors, "decisiveSectorCount": len(decisive),
        "classification": "SINGLE_CATALOG_CANDIDATE_ATTRIBUTED" if resolved else "UNRESOLVED",
        "preferredModel": model, "preferredCandidate": preferred,
        # Preserve the frozen stage-044 hypothesis set in its original order. The losing
        # candidate remains a required nuisance source in direct variability deblending.
        "catalogCandidates": candidates,
        "physicalCycleResolved": False, "physicalMechanismResolved": False,
        "sourceAttributionResolved": resolved,
        "recommendedNextTest": (
            "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION" if resolved
            else "ADDITIONAL_SOURCE_LOCALIZATION_DATA"),
        "claimLevelChanged": resolved,
        "interpretationGuard": "Unresolved aggregate predictive evidence must not finalize attribution.",
    }
