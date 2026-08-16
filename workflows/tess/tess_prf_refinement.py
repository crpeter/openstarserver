"""Evidence-seeded official SPOC PRF refinement of residual source geometry.

This is the first historical mission-PRF implementation, adapted to accept the
positions already measured by v20.11/v20.12 instead of a catalog counterpart.
The only replaceable test boundary is acquisition of the official calibration
image; rendering, fitting, BIC comparison, and interpretation remain real.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_multisource_residual import _prewhiten_cube_raw
from .tess_residual_localization import (
    _background_subtract_cube, _download_tpf, _uniform_indices, MAX_CADENCES,
    _write_json,
)
from .tess_spoc_prf import (
    MAST_PRF_ROOT, _drift_corrected_times, _list_official_prf_grid,
    _official_prf_at_detector_position, _render_prf_template,
    _tpf_detector_geometry, MAX_OFFICIAL_PRF_DESIGN_CONDITION,
    MAX_OFFICIAL_PRF_TEMPLATE_CORRELATION,
)

MIN_SAMPLES = 100
# 95% chi-square quantile for a two-parameter coherent [sin, cos] vector.
# Unlike the removed correlation/condition/delta-BIC cutoffs, this has a
# defined false-positive interpretation under the persisted Gaussian model.
COHERENT_VECTOR_CHI2_95 = 5.991464547107979


def _calibration(*, sector: int, camera: int, ccd: int, row: float, column: float,
                 cache: Path) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """External boundary: official MAST grid listing/download, freeze only this in tests."""
    grid = _list_official_prf_grid(sector=sector, camera=camera, ccd=ccd)
    return _official_prf_at_detector_position(
        sector=sector, camera=camera, ccd=ccd, detector_row=row,
        detector_col=column, archive_cache=cache, grid_entries=grid,
    )


def prepare_prf_deblending(*, evidence: dict[str, dict[str, Any]], output_dir: Path,
                           investigation_id: str) -> dict[str, Any]:
    decomposition = evidence["multiSourceDecomposition"]
    decomposition_preparation = evidence["decompositionPreparation"]
    components = decomposition.get("spatialComponents") or []
    target = next((x for x in components if x.get("componentType") == "TARGET"), None)
    offset_id = decomposition.get("bestOffsetComponentID")
    offset = next((x for x in components if x.get("componentID") == offset_id), None)
    if target is None or offset is None:
        raise RuntimeError("PRF deblending requires persisted target and best-offset geometry.")
    centers: dict[str, dict[str, dict[str, float]]] = {}
    for item in decomposition_preparation.get("preparedSeries") or []:
        if item.get("combined") or not item.get("pixelCenter"):
            continue
        centers.setdefault(str(item["sector"]), {})[str(item["componentID"])] = dict(item["pixelCenter"])
    sectors = [int(s) for s, value in centers.items()
               if target["componentID"] in value and offset_id in value]
    if not sectors:
        raise RuntimeError("PRF deblending requires per-sector target and offset pixel positions.")
    prepared = evidence["targetPreparation"]
    identity_metadata = ((evidence["targetIdentity"].get("tic") or {}).get("metadata") or {})
    ra_deg = identity_metadata.get("raDeg")
    dec_deg = identity_metadata.get("decDeg")
    if ra_deg is None or dec_deg is None:
        raise RuntimeError("PRF deblending requires the persisted target sky position.")
    morphology = evidence["physicalMorphology"]
    nonstationary = evidence["nonstationaryResidual"]
    root = Path(output_dir) / "prf-deblending"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-prf-deblending.v1",
        "investigationID": investigation_id,
        "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "prf-deblending-preparation.json").resolve()),
        "modelSource": "official-public-SPOC-TESS-PRF-FITS",
        "officialPRFRoot": MAST_PRF_ROOT,
        "externalBoundary": "official PRF grid listing and FITS acquisition only",
        "execution": "coordinator-local-small-coupled-spatial-fit",
        "target": {"componentID": target["componentID"], "initialGeometry": target},
        "offset": {"componentID": offset_id, "initialGeometry": offset},
        "sectorPixelCenters": centers,
        "sectors": sorted(sectors),
        "ticID": prepared.get("ticID"),
        "targetSky": {"raDeg": float(ra_deg), "decDeg": float(dec_deg),
                      "provenance": "persisted target TIC identity; no neighbor lookup"},
        "physicalPeriodDays": morphology["resolvedPhysicalPeriodDays"],
        "residualReferenceFrequency": nonstationary["preferredFrequencyAtReference"],
        "residualTimeReferenceDays": nonstationary["timeReferenceDays"],
        "fractionalFrequencyDriftPerDay": nonstationary["fractionalFrequencyDriftPerDay"],
        "priorEvidence": evidence,
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def _weighted_hypothesis(
    *, observations: np.ndarray, pixel_covariances: np.ndarray,
    templates: np.ndarray, component_ids: list[str],
) -> dict[str, Any]:
    """Fit coherent source vectors using block-whitened per-pixel covariance."""
    pixel_count = len(pixel_covariances)
    source_count = templates.shape[1]
    design = np.zeros((pixel_count * 2, source_count * 2), dtype=float)
    design[0::2, 0::2] = templates
    design[1::2, 1::2] = templates
    whitened_design = np.empty_like(design)
    whitened_observations = np.empty_like(observations)
    log_determinant = 0.0
    for pixel in range(pixel_count):
        rows = slice(2 * pixel, 2 * pixel + 2)
        covariance = pixel_covariances[pixel]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise RuntimeError("Temporal coherent-vector covariance is not positive definite.")
        chol = np.linalg.cholesky(covariance)
        whitened_design[rows] = np.linalg.solve(chol, design[rows])
        whitened_observations[rows] = np.linalg.solve(chol, observations[rows])
        log_determinant += float(logdet)

    coefficients, _, rank, singular_values = np.linalg.lstsq(
        whitened_design, whitened_observations, rcond=None
    )
    residual = whitened_observations - whitened_design @ coefficients
    chi_square = float(residual @ residual)
    n = int(observations.size)
    k = int(coefficients.size)
    log_likelihood = -0.5 * (chi_square + log_determinant + n * math.log(2.0 * math.pi))
    normal = whitened_design.T @ whitened_design
    parameter_covariance = np.linalg.pinv(normal, hermitian=True)
    source_estimates = []
    for index, component_id in enumerate(component_ids):
        selection = slice(2 * index, 2 * index + 2)
        vector = coefficients[selection]
        covariance = parameter_covariance[selection, selection]
        vector_chi_square = float(vector @ np.linalg.pinv(covariance, hermitian=True) @ vector)
        source_estimates.append({
            "componentID": component_id,
            "sinA": float(vector[0]), "cosB": float(vector[1]),
            "coherentAmplitude": float(np.linalg.norm(vector)),
            "covariance": covariance.tolist(),
            "amplitudeStandardErrorDeltaMethod": (
                float(math.sqrt(max(vector @ covariance @ vector, 0.0)) / np.linalg.norm(vector))
                if np.linalg.norm(vector) > 0 else None
            ),
            "coherentVectorChiSquare": vector_chi_square,
            "individuallyIdentifiable": bool(
                rank == k and vector_chi_square >= COHERENT_VECTOR_CHI2_95
            ),
        })
    return {
        "chiSquare": chi_square, "logLikelihood": log_likelihood,
        "bic": float(k * math.log(n) - 2.0 * log_likelihood),
        "observationCount": n, "parameterCount": k,
        "parameterEstimates": coefficients.tolist(),
        "parameterCovariance": parameter_covariance.tolist(),
        "rank": int(rank), "fullRank": bool(rank == k),
        "singularValues": singular_values.tolist(),
        "sourceEstimates": source_estimates,
    }


def compare_prf_hypotheses(*, coherent_coefficients: np.ndarray,
                           pixel_covariances: np.ndarray,
                           template_matrix: np.ndarray,
                           component_ids: list[str]) -> dict[str, Any]:
    observations = np.asarray(coherent_coefficients, dtype=float).reshape(-1)
    definitions = {
        "TARGET_ONLY": [0], "OFFSET_ONLY": [1], "TARGET_PLUS_OFFSET": [0, 1],
    }
    models = {}
    for name, indices in definitions.items():
        models[name] = _weighted_hypothesis(
            observations=observations, pixel_covariances=pixel_covariances,
            templates=template_matrix[:, indices],
            component_ids=[component_ids[index] for index in indices],
        )
    ranked = sorted(models, key=lambda key: models[key]["bic"])
    for name in models:
        models[name]["deltaBIC"] = float(models[name]["bic"] - models[ranked[0]]["bic"])
    best = models[ranked[0]]
    identifiable = bool(best["fullRank"] and all(
        item["individuallyIdentifiable"] for item in best["sourceEstimates"]
    ))
    joint = models["TARGET_PLUS_OFFSET"]
    joint_by_component = {item["componentID"]: item for item in joint["sourceEstimates"]}
    if ranked[0] in {"TARGET_ONLY", "OFFSET_ONLY"}:
        selected = component_ids[0] if ranked[0] == "TARGET_ONLY" else component_ids[1]
        other = component_ids[1] if ranked[0] == "TARGET_ONLY" else component_ids[0]
        # Source attribution, unlike mere signal detection, must survive the
        # conditional two-source fit. This naturally becomes unresolved as
        # PRFs become collinear because the joint covariance inflates.
        identifiable = bool(
            identifiable and joint["fullRank"]
            and joint_by_component[selected]["individuallyIdentifiable"]
            and not joint_by_component[other]["individuallyIdentifiable"]
        )
    return {"models": models, "bestModel": ranked[0],
            "bestModelIdentifiable": identifiable}


def _primary_residual_diagnostics(*, coherent_coefficients: np.ndarray,
                                  pixel_covariances: np.ndarray,
                                  template_matrix: np.ndarray,
                                  models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Describe whether the alternate PRF absorbs a one-source model residual."""
    primary_id = min(("TARGET_ONLY", "OFFSET_ONLY"), key=lambda name: models[name]["bic"])
    primary_index = 0 if primary_id == "TARGET_ONLY" else 1
    alternative_index = 1 - primary_index
    observed = np.asarray(coherent_coefficients, dtype=float)
    vector = np.asarray(models[primary_id]["parameterEstimates"], dtype=float)
    prediction = template_matrix[:, primary_index, None] * vector[None, :]
    residual = observed - prediction
    alternative = template_matrix[:, alternative_index]
    projection_vectors = []
    weighted_improvement = 0.0
    for phase_index in range(2):
        weights = np.asarray([np.linalg.pinv(cov, hermitian=True)[phase_index, phase_index]
                              for cov in pixel_covariances])
        denominator = float(np.sum(weights * alternative * alternative))
        coefficient = (float(np.sum(weights * alternative * residual[:, phase_index])) / denominator
                       if denominator > 0 else 0.0)
        projected = alternative * coefficient
        projection_vectors.append({"coefficient": coefficient,
                                   "projectedNorm": float(np.linalg.norm(projected))})
        weighted_improvement += float(np.sum(weights * (
            residual[:, phase_index] ** 2 - (residual[:, phase_index] - projected) ** 2
        )))
    joint_improvement = float(models[primary_id]["chiSquare"]
                              - models["TARGET_PLUS_OFFSET"]["chiSquare"])
    return {
        "primaryModel": primary_id,
        "observedCoherentMapNorm": float(np.linalg.norm(observed)),
        "primaryPredictionNorm": float(np.linalg.norm(prediction)),
        "primaryOnlyResidualNorm": float(np.linalg.norm(residual)),
        "residualToPrimaryPredictionNormRatio": (
            float(np.linalg.norm(residual) / np.linalg.norm(prediction))
            if np.linalg.norm(prediction) > 0 else None
        ),
        "alternativeTemplateProjectionByPhase": projection_vectors,
        "alternativeProjectionWeightedChiSquareImprovement": weighted_improvement,
        "jointModelChiSquareImprovement": joint_improvement,
        "projectionFractionOfJointImprovement": (
            weighted_improvement / joint_improvement if joint_improvement > 0 else None
        ),
        "observedSinMap": observed[:, 0].tolist(),
        "observedCosMap": observed[:, 1].tolist(),
        "primaryPredictionSinMap": prediction[:, 0].tolist(),
        "primaryPredictionCosMap": prediction[:, 1].tolist(),
        "primaryResidualSinMap": residual[:, 0].tolist(),
        "primaryResidualCosMap": residual[:, 1].tolist(),
    }


def _empirical_cross_pixel_diagnostic(*, temporal_errors: np.ndarray,
                                      temporal_normal_inverse: np.ndarray,
                                      coherent_coefficients: np.ndarray,
                                      template_matrix: np.ndarray,
                                      component_ids: list[str]) -> dict[str, Any]:
    """Compare joint-source Q using empirical cross-pixel covariance, without regularization."""
    spatial_covariance = np.cov(temporal_errors, rowvar=False, ddof=1)
    covariance = np.kron(spatial_covariance, temporal_normal_inverse)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = (np.finfo(float).eps * max(covariance.shape)
                 * max(float(np.max(np.abs(eigenvalues))), 1.0))
    keep = eigenvalues > tolerance
    if int(np.count_nonzero(keep)) < covariance.shape[0]:
        return {"available": False, "reason": "empirical covariance is not full rank",
                "rank": int(np.count_nonzero(keep)), "dimension": int(covariance.shape[0]),
                "eigenvalues": eigenvalues.tolist()}
    whitener = ((eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))
                @ eigenvectors[:, keep].T)
    pixel_count = len(template_matrix)
    design = np.zeros((pixel_count * 2, 4), dtype=float)
    design[0::2, 0::2] = template_matrix
    design[1::2, 1::2] = template_matrix
    whitened_design = whitener @ design
    whitened_observed = whitener @ np.asarray(coherent_coefficients).reshape(-1)
    coefficients, _, rank, singular = np.linalg.lstsq(
        whitened_design, whitened_observed, rcond=None
    )
    parameter_covariance = np.linalg.pinv(
        whitened_design.T @ whitened_design, hermitian=True
    )
    sources = []
    for index, component_id in enumerate(component_ids):
        selection = slice(2 * index, 2 * index + 2)
        vector = coefficients[selection]
        source_covariance = parameter_covariance[selection, selection]
        q = float(vector @ np.linalg.pinv(source_covariance, hermitian=True) @ vector)
        sources.append({"componentID": component_id, "coherentVectorChiSquare": q,
                        "covariance": source_covariance.tolist()})
    return {"available": True, "rank": int(rank), "singularValues": singular.tolist(),
            "sources": sources,
            "assumption": "Empirical cadence-residual cross-pixel covariance; no shrinkage or regularization."}


def run_prf_deblending(preparation: dict[str, Any]) -> dict[str, Any]:
    sector_results = []
    errors = []
    cache = Path(preparation["artifactRoot"]) / "official-prf-cache"
    for sector in preparation["sectors"]:
        try:
            tpf, source = _download_tpf(tic_id=int(preparation["ticID"]), sector=int(sector),
                                        ra_deg=float(preparation["targetSky"]["raDeg"]),
                                        dec_deg=float(preparation["targetSky"]["decDeg"]))
            times = np.asarray(tpf.time.value, dtype=float)
            cube = np.asarray(getattr(tpf.flux, "value", tpf.flux), dtype=float)
            keep = np.isfinite(times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
            times, cube = times[keep], cube[keep]
            indices = _uniform_indices(len(times), MAX_CADENCES)
            times, cube = times[indices], cube[indices]
            if len(times) < MIN_SAMPLES:
                raise RuntimeError(f"Only {len(times)} usable cadences.")
            corrected, _ = _background_subtract_cube(cube)
            residual, valid = _prewhiten_cube_raw(
                absolute_times=times, cube=corrected,
                physical_frequency=1.0 / float(preparation["physicalPeriodDays"]),
            )
            camera, ccd, tpf_col, tpf_row = _tpf_detector_geometry(tpf)
            centers = preparation["sectorPixelCenters"][str(sector)]
            component_ids = [preparation["target"]["componentID"], preparation["offset"]["componentID"]]
            templates = []
            calibration_provenance = []
            for component_id in component_ids:
                center = centers[component_id]
                detector_row = tpf_row + float(center["y"])
                detector_col = tpf_col + float(center["x"])
                image, header, files = _calibration(
                    sector=sector, camera=camera, ccd=ccd, row=detector_row,
                    column=detector_col, cache=cache,
                )
                templates.append(_render_prf_template(
                    image=image, header=header, source_x=float(center["x"]),
                    source_y=float(center["y"]), rows=valid.shape[0], cols=valid.shape[1],
                    valid_pixels=valid,
                ))
                calibration_provenance.append({"componentID": component_id,
                    "initialPixelPosition": center, "detectorRow": detector_row,
                    "detectorColumn": detector_col, "modelFiles": files})
            valid_flat = valid.reshape(-1)
            template_matrix = np.column_stack(templates)[valid_flat]
            correlation = float(np.corrcoef(template_matrix.T)[0, 1])
            condition = float(np.linalg.cond(template_matrix))
            warped = _drift_corrected_times(
                times, time_reference_days=float(preparation["residualTimeReferenceDays"]),
                fractional_frequency_drift_per_day=float(preparation["fractionalFrequencyDriftPerDay"]),
            )
            phase = 2 * math.pi * float(preparation["residualReferenceFrequency"]) * warped
            temporal = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(phase))))
            flat_residual = residual.reshape(len(times), -1)[:, valid_flat]
            temporal_coefficients, _, temporal_rank, temporal_singular = np.linalg.lstsq(
                temporal, flat_residual, rcond=None
            )
            temporal_errors = flat_residual - temporal @ temporal_coefficients
            degrees_freedom = len(times) - int(temporal_rank)
            if degrees_freedom <= 0:
                raise RuntimeError("Temporal coherent fit has no residual degrees of freedom.")
            residual_variances = np.sum(temporal_errors * temporal_errors, axis=0) / degrees_freedom
            temporal_normal_inverse = np.linalg.pinv(temporal.T @ temporal, hermitian=True)[:2, :2]
            covariance_floor = np.finfo(float).eps * max(float(np.nanmedian(residual_variances)), 1.0)
            pixel_covariances = np.asarray([
                temporal_normal_inverse * max(float(variance), covariance_floor)
                for variance in residual_variances
            ])
            comparison = compare_prf_hypotheses(
                coherent_coefficients=temporal_coefficients[:2].T,
                pixel_covariances=pixel_covariances,
                template_matrix=template_matrix,
                component_ids=component_ids,
            )
            models = comparison["models"]
            best_model = comparison["bestModel"]
            primary_residual_diagnostics = _primary_residual_diagnostics(
                coherent_coefficients=temporal_coefficients[:2].T,
                pixel_covariances=pixel_covariances,
                template_matrix=template_matrix,
                models=models,
            )
            empirical_cross_pixel = _empirical_cross_pixel_diagnostic(
                temporal_errors=temporal_errors,
                temporal_normal_inverse=temporal_normal_inverse,
                coherent_coefficients=temporal_coefficients[:2].T,
                template_matrix=template_matrix,
                component_ids=component_ids,
            )
            historical_guard = bool(
                math.isfinite(condition)
                and condition <= MAX_OFFICIAL_PRF_DESIGN_CONDITION
                and abs(correlation) < MAX_OFFICIAL_PRF_TEMPLATE_CORRELATION
            )
            sector_results.append({
                "sector": sector, "source": source, "camera": camera, "ccd": ccd,
                "tpfPhysicalRow": tpf_row, "tpfPhysicalColumn": tpf_col,
                "sampleCount": len(times), "actualResidualPixelCount": int(np.count_nonzero(valid)),
                "calibration": calibration_provenance, "models": models,
                "primaryOnlyResidualDiagnostics": primary_residual_diagnostics,
                "empiricalCrossPixelCovarianceDiagnostic": empirical_cross_pixel,
                "executionProvenance": {
                    "officialDetectorGridInterpolationExecuted": True,
                    "prfStampRenderingExecuted": True,
                    "renderedComponentIDs": component_ids,
                },
                "bestModel": best_model,
                "templateCorrelation": correlation, "designConditionNumber": condition,
                "historicalForwardDesignGuard": {
                    "passed": historical_guard,
                    "maximumConditionNumber": MAX_OFFICIAL_PRF_DESIGN_CONDITION,
                    "maximumAbsoluteTemplateCorrelation": MAX_OFFICIAL_PRF_TEMPLATE_CORRELATION,
                    "provenance": "tess_spoc_prf historical official-forward-model guard",
                },
                "temporalFit": {"rank": int(temporal_rank),
                    "singularValues": temporal_singular.tolist(),
                    "degreesOfFreedomPerPixel": degrees_freedom,
                    "coherentCoefficientCovariances": pixel_covariances.tolist(),
                    "crossPixelCovarianceAssumption": (
                        "Block diagonal across pixels. Per-pixel heteroscedastic residual variance and "
                        "within-pixel sin/cos covariance are estimated; preprocessing-induced cross-pixel "
                        "covariance is not estimated."
                    )},
                "degenerate": not (comparison["bestModelIdentifiable"] and historical_guard),
                "decisive": bool(comparison["bestModelIdentifiable"] and historical_guard),
            })
        except Exception as exc:
            errors.append({"sector": sector, "error": f"{type(exc).__name__}: {exc}"})
    return {"version": "openstar.tess-prf-deblending-run.v1", "sectorResults": sector_results,
            "errors": errors, "workerProtocolUsed": False}


def interpret_prf_deblending(preparation: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    if not run.get("sectorResults") and run.get("errors"):
        classification = "BLOCKED_EXTERNAL_DATA"
        recommended = "RETRY_PIXEL_RESPONSE_FUNCTION_DEBLENDING"
        decisive = []
        winners = []
    else:
        classification = None
        recommended = None
    decisive = [x for x in run.get("sectorResults", []) if x.get("decisive")]
    winners = [x["bestModel"] for x in decisive]
    if classification is not None:
        pass
    elif len(winners) >= 2 and set(winners) == {"TARGET_ONLY", "OFFSET_ONLY"}:
        classification = "PRF_SOURCE_SWITCHING"
    elif len(winners) >= 2 and len(set(winners)) == 1:
        classification = {"TARGET_ONLY": "PRF_TARGET_DOMINANT", "OFFSET_ONLY": "PRF_OFFSET_DOMINANT",
                          "TARGET_PLUS_OFFSET": "PRF_MULTIPLE_SOURCES"}[winners[0]]
    else:
        classification = "PRF_DEBLENDING_UNRESOLVED"
    if recommended is None:
        recommended = ("CATALOG_COUNTERPART_IDENTIFICATION"
                       if classification != "PRF_DEBLENDING_UNRESOLVED"
                       else "HIGHER_RESOLUTION_SPATIAL_FOLLOWUP")
    return {
        "version": "openstar.tess-prf-deblending-interpretation.v1",
        "modelSource": preparation["modelSource"], "officialPRFRoot": preparation["officialPRFRoot"],
        "sourceHypotheses": ["TARGET_ONLY", "OFFSET_ONLY", "TARGET_PLUS_OFFSET"],
        "modelSelection": {
            "metric": "Gaussian block-weighted likelihood BIC",
            "categoricalRule": (
                "minimum BIC model with full numerical rank and every included source's "
                "conditional two-parameter coherent vector significant at chi-square 95%"
            ),
            "coherentVectorChiSquare95": COHERENT_VECTOR_CHI2_95,
            "rankTolerance": "numpy.linalg.lstsq default machine-precision tolerance",
            "noNewCorrelationOrConditionNumberCutoff": True,
            "historicalForwardDesignGuardsRetained": {
                "maximumConditionNumber": MAX_OFFICIAL_PRF_DESIGN_CONDITION,
                "maximumAbsoluteTemplateCorrelation": MAX_OFFICIAL_PRF_TEMPLATE_CORRELATION,
            },
        },
        "sectorResults": run.get("sectorResults", []), "errors": run.get("errors", []),
        "classification": classification,
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommended,
        "followingHandlerActivated": False,
        "interpretationGuard": "PRF evidence refines residual spatial attribution only; it does not reinterpret the established physical cycle.",
    }
