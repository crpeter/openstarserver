"""Coordinator-local, catalog-guided official-SPOC-PRF source localization."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_multisource_residual import _prewhiten_cube_raw
from .tess_offset_variability import _skycoord
from .tess_prf_refinement import (
    COHERENT_VECTOR_CHI2_95, MIN_SAMPLES, _calibration, _weighted_hypothesis,
)
from .tess_residual_localization import (
    MAX_CADENCES, _background_subtract_cube, _download_tpf, _uniform_indices,
    _write_json,
)
from .tess_spoc_prf import (
    MAST_PRF_ROOT, _drift_corrected_times, _render_prf_template,
    _tpf_detector_geometry,
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
            templates, calibration = [], []
            for definition in definitions:
                coordinate = _skycoord(float(definition["raDeg"]), float(definition["decDeg"]))
                x, y = map(float, tpf.wcs.world_to_pixel(coordinate))
                image, header, files = _calibration(sector=int(sector), camera=camera, ccd=ccd,
                    row=tpf_row + y, column=tpf_col + x, cache=cache)
                templates.append(_render_prf_template(image=image, header=header, source_x=x, source_y=y,
                    rows=valid.shape[0], cols=valid.shape[1], valid_pixels=valid))
                calibration.append({"componentID": definition["componentID"], "catalogSky": {
                    "raDeg": definition["raDeg"], "decDeg": definition["decDeg"]},
                    "catalogPixelCenter": {"x": x, "y": y}, "sharedAstrometricOffsetPixels": {"x": 0.0, "y": 0.0},
                    "officialPRFModelFiles": files})
            valid_flat = valid.reshape(-1); template_matrix = np.column_stack(templates)[valid_flat]
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
            # Existing framework's contiguous temporal-validation principle: an attribution
            # must independently win in both chronological halves.
            held_out = []
            for block in np.array_split(np.arange(len(times)), 2):
                fit, _, block_rank, _ = np.linalg.lstsq(temporal[block], pixels[block], rcond=None)
                err = pixels[block] - temporal[block] @ fit
                var = np.sum(err * err, axis=0) / max(1, len(block) - block_rank)
                cov = np.asarray([np.linalg.pinv(temporal[block].T @ temporal[block], hermitian=True)[:2, :2]
                                  * max(float(v), floor) for v in var])
                held_out.append(_comparison(fit[:2].T, cov, template_matrix, component_ids)["bestModel"])
            decisive = comparison["bestModelIdentifiable"] and len(set(held_out)) == 1 and held_out[0] == comparison["bestModel"]
            results.append({"sector": int(sector), "source": source, "componentIDs": component_ids,
                "calibration": calibration, "sharedAstrometricCalibration": True,
                "models": comparison["models"], "modelRanking": comparison["ranking"],
                "bestModel": comparison["bestModel"], "bestModelIdentifiable": comparison["bestModelIdentifiable"],
                "heldOutTemporalValidation": {"chronologicalBlockWinners": held_out, "consistent": len(set(held_out)) == 1},
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
            "heldOutTemporalValidation": "two chronological blocks; no new score cutoff"}}
