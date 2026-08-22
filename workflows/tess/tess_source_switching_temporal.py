"""Frozen-frequency temporal source model for the residual-localization branch.

All source labels and pixel positions are coordinator-side scientific inputs.  The
calculation deliberately does not create a generic worker work unit.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_catalog_guided_localization import (
    COMPONENT_IDS, _fit_shared_astrometric_shift, _production_sector_inputs,
)
from .tess_residual_localization import _time_warp, _write_json

EXPECTED_SECTORS = (94, 95, 102, 103)


def prepare_source_switching_temporal_model(*, difference_interpretation: dict[str, Any],
                                            difference_preparation: dict[str, Any],
                                            output_dir: Path,
                                            investigation_id: str) -> dict[str, Any]:
    if (difference_interpretation.get("classification") != "SOURCE_SWITCHING_BY_SECTOR"
            or difference_interpretation.get("recommendedNextTest")
            != "SOURCE_SWITCHING_TEMPORAL_MODEL"):
        raise RuntimeError("The persisted stage-050 result does not recommend temporal modeling.")
    candidates = list(difference_preparation.get("catalogCandidates") or [])
    if len(candidates) < 2:
        raise RuntimeError("Both frozen stage-044 catalog candidates are required.")
    required = ("referenceFamilyPeriodDays", "residualReferenceFrequency",
                "residualTimeReferenceDays", "fractionalFrequencyDriftPerDay",
                "subtractedHarmonicOrders")
    if any(difference_preparation.get(key) is None for key in required):
        raise RuntimeError("The residual frequency/drift bridge is incomplete.")
    sectors = tuple(int(x) for x in difference_preparation.get("sectors") or [])
    if sectors != EXPECTED_SECTORS:
        raise RuntimeError("The real continuation requires persisted sectors 94, 95, 102, and 103.")
    root = Path(output_dir) / "source-switching-temporal-model"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-source-switching-temporal-preparation.v1",
        "investigationID": investigation_id, "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-fixed-frequency-source-temporal-model",
        "ticID": difference_preparation.get("ticID"),
        "targetSky": difference_preparation.get("targetSky"),
        "catalogCandidates": candidates[:2],
        "spatialHypotheses": list(difference_preparation.get("spatialHypotheses") or []),
        "sectors": list(sectors), "physicalCycleResolved": False,
        **{key: difference_preparation[key] for key in required},
        "priorStage050Classification": difference_interpretation["classification"],
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def _templates(item: dict[str, Any], valid: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    supplied = item.get("sourceTemplates")
    if supplied is not None:
        templates = np.asarray(supplied, dtype=float)
        if templates.shape[0] != 3:
            raise RuntimeError("sourceTemplates must contain target and both candidates.")
        templates = templates.reshape(3, -1)[:, valid.ravel()].T
        calibration = {"available": True, "method": "injected-calibrated-test-templates"}
    else:
        renderer = item.get("renderTemplates")
        calibration_image = item.get("calibrationImage")
        background = item.get("backgroundColumns")
        if renderer is None or calibration_image is None or background is None:
            raise RuntimeError(
                "Production temporal modeling requires calibrated official SPOC PRF inputs; "
                "synthetic spatial-template fallback is forbidden.")
        calibrated = _fit_shared_astrometric_shift(
            calibration_image=np.asarray(calibration_image, float),
            background_columns=[np.asarray(x, float) for x in background],
            render_templates=renderer)
        if not calibrated.get("available"):
            return None, calibrated
        templates = np.asarray(calibrated["templates"], float)
        calibration = {"available": True, "method": "official-spoc-prf-shared-astrometry",
                       "sharedAstrometricCalibration": calibrated["sharedAstrometricCalibration"]}
    norms = np.sqrt(np.sum(templates * templates, axis=0))
    if np.any(norms <= 0):
        raise RuntimeError("A frozen source template has zero support.")
    return templates / norms, calibration


def _design(times: np.ndarray, templates: np.ndarray, preparation: dict[str, Any]) -> np.ndarray:
    warped = _time_warp(times - float(preparation["residualTimeReferenceDays"]),
                        float(preparation["fractionalFrequencyDriftPerDay"]))
    angle = 2 * np.pi * float(preparation["residualReferenceFrequency"]) * warped
    # time-major flattening matches cube[:, valid].reshape(-1)
    return np.column_stack([(np.sin(angle)[:, None] * templates[:, j]).reshape(-1)
                            for j in range(3)] +
                           [(np.cos(angle)[:, None] * templates[:, j]).reshape(-1)
                            for j in range(3)])


def _fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    beta, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    rss = max(float(residual @ residual), np.finfo(float).tiny)
    dof = max(1, len(y) - x.shape[1])
    covariance = np.linalg.pinv(x.T @ x) * rss / dof
    return {"beta": beta, "rss": rss, "rank": int(rank), "singular": singular,
            "covariance": covariance, "n": len(y), "k": x.shape[1]}


def run_source_switching_temporal_model(preparation: dict[str, Any], *,
                                        sector_inputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # The catalog-guided acquisition path is the authoritative official-PRF path:
    # it preserves frozen WCS positions and exposes one shared astrometric renderer.
    inputs = _production_sector_inputs(preparation) if sector_inputs is None else sector_inputs
    sectors = []
    for item in inputs:
        times = np.asarray(item["times"], float)
        cube = np.asarray(item["prewhitened"], float)
        valid = np.asarray(item["valid"], bool)
        finite = np.isfinite(times) & np.all(np.isfinite(cube[:, valid]), axis=1)
        times, values = times[finite], cube[finite][:, valid]
        template, calibration = _templates(item, valid)
        if template is None:
            sectors.append({"sector": int(item["sector"]), "usable": False,
                            "calibrationFailure": calibration})
            continue
        x = _design(times, template, preparation)
        y = values.reshape(-1)
        sectors.append({"sector": int(item["sector"]), "times": times, "x": x, "y": y,
                        "usable": True, "spatialCalibration": calibration,
                        "templateRank": int(np.linalg.matrix_rank(template)),
                        "templateConditionNumber": float(np.linalg.cond(template))})
    if [x["sector"] for x in sectors] != list(preparation["sectors"]):
        raise RuntimeError("Temporal model inputs do not match the frozen sector ordering.")

    usable_sectors = [sector for sector in sectors if sector.get("usable")]
    source_sets = {"TARGET_STATIONARY": (0,), "CANDIDATE_1_STATIONARY": (1,),
                   "CANDIDATE_2_STATIONARY": (2,),
                   "TARGET_PLUS_CANDIDATE_1_STATIONARY": (0, 1),
                   "TARGET_PLUS_CANDIDATE_2_STATIONARY": (0, 2),
                   "CANDIDATE_1_PLUS_CANDIDATE_2_STATIONARY": (1, 2),
                   "ALL_SOURCES_STATIONARY": (0, 1, 2)}
    models: dict[str, Any] = {}
    if not usable_sectors:
        return {"version": "openstar.tess-source-switching-temporal-run.v1", "models": {},
                "perSectorSourceCoherentVectors": [], "sectorUsability": sectors,
                "sourceIdentifiable": False, "physicalCycleResolved": False,
                "frozenResidualEphemeris": {"frequency": preparation["residualReferenceFrequency"],
                    "timeReferenceDays": preparation["residualTimeReferenceDays"],
                    "fractionalFrequencyDriftPerDay": preparation["fractionalFrequencyDriftPerDay"]}}
    all_y = np.concatenate([s["y"] for s in usable_sectors])
    for name, sources in source_sets.items():
        columns = list(sources) + [3 + j for j in sources]
        fit = _fit(np.vstack([s["x"][:, columns] for s in usable_sectors]), all_y)
        models[name] = {"rss": fit["rss"], "bic": len(all_y) * math.log(fit["rss"] / len(all_y))
                        + fit["k"] * math.log(len(all_y)), "parameterCount": fit["k"],
                        "rank": fit["rank"]}

    vectors = []
    varying_rss = 0.0
    varying_k = 0
    identifiable = True
    for sector in usable_sectors:
        fit = _fit(sector["x"], sector["y"])
        varying_rss += fit["rss"]; varying_k += fit["k"]
        identifiable &= fit["rank"] == 6 and sector["templateRank"] == 3
        entry = {"sector": sector["sector"], "templateRank": sector["templateRank"],
                 "templateConditionNumber": sector["templateConditionNumber"],
                 "spatialCalibration": sector["spatialCalibration"],
                 "jointCoherentVectorCovariance": fit["covariance"].tolist(), "sources": {}}
        for j, component in enumerate(COMPONENT_IDS):
            sin, cos = float(fit["beta"][j]), float(fit["beta"][3 + j])
            cov = fit["covariance"][np.ix_([j, 3 + j], [j, 3 + j])]
            entry["sources"][component] = {
                "sinAmplitude": sin, "cosAmplitude": cos,
                "coherentAmplitude": math.hypot(sin, cos),
                "phaseRadians": math.atan2(cos, sin),
                "sinUncertainty": math.sqrt(max(0., float(cov[0, 0]))),
                "cosUncertainty": math.sqrt(max(0., float(cov[1, 1]))),
                "covariance": cov.tolist()}
        vectors.append(entry)
    models["SECTOR_VARYING_SOURCE_AMPLITUDES"] = {
        "rss": varying_rss, "bic": len(all_y) * math.log(varying_rss / len(all_y))
        + varying_k * math.log(len(all_y)), "parameterCount": varying_k,
        "rank": sum(6 for _ in usable_sectors) if identifiable else None}

    # Frozen-parameter prediction over contiguous time blocks.  Every cadence in a
    # held-out block is absent from the amplitude/phase fit used to predict it.
    cv = {name: 0.0 for name in models}
    fold_evidence = []
    block_count = 4
    for name, sources in {**source_sets, "SECTOR_VARYING_SOURCE_AMPLITUDES": (0, 1, 2)}.items():
        columns = list(sources) + [3 + j for j in sources]
        for fold_index in range(block_count):
            partitions = []
            for s in usable_sectors:
                pixels = len(s["y"]) // len(s["times"])
                blocks = np.array_split(np.arange(len(s["times"])), block_count)
                held_cadences = blocks[fold_index]
                held = np.isin(np.repeat(np.arange(len(s["times"])), pixels), held_cadences)
                partitions.append((s, ~held, held, held_cadences))
            records = []
            if name == "SECTOR_VARYING_SOURCE_AMPLITUDES":
                fold_rss = 0.0
                for s, train, held, held_cadences in partitions:
                    fit = _fit(s["x"][train][:, columns], s["y"][train])
                    rss = float(np.sum((s["y"][held] -
                                        s["x"][held][:, columns] @ fit["beta"]) ** 2))
                    fold_rss += rss
                    records.append({"sector": s["sector"], "heldOutRSS": rss,
                                    "heldOutTimeRange": [float(s["times"][held_cadences[0]]),
                                                         float(s["times"][held_cadences[-1]])]})
            else:
                fit = _fit(np.vstack([s["x"][train][:, columns]
                                      for s, train, _, _ in partitions]),
                           np.concatenate([s["y"][train]
                                           for s, train, _, _ in partitions]))
                fold_rss = 0.0
                for s, _, held, held_cadences in partitions:
                    rss = float(np.sum((s["y"][held] -
                                        s["x"][held][:, columns] @ fit["beta"]) ** 2))
                    fold_rss += rss
                    records.append({"sector": s["sector"], "heldOutRSS": rss,
                                    "heldOutTimeRange": [float(s["times"][held_cadences[0]]),
                                                         float(s["times"][held_cadences[-1]])]})
            cv[name] += fold_rss
            fold_evidence.append({"model": name, "foldIndex": fold_index,
                                  "heldOutRSS": fold_rss, "sectorEvidence": records})
        models[name]["heldOutRSS"] = cv[name]
    return {"version": "openstar.tess-source-switching-temporal-run.v1",
            "models": models, "perSectorSourceCoherentVectors": vectors,
            "sourceIdentifiable": bool(identifiable), "physicalCycleResolved": False,
            "sectorUsability": [{key: value for key, value in sector.items()
                                  if key not in {"times", "x", "y"}} for sector in sectors],
            "heldOutTemporalValidation": {
                "method": "four contiguous temporal blocks with frozen amplitude/phase fits",
                "folds": fold_evidence},
            "frozenResidualEphemeris": {"frequency": preparation["residualReferenceFrequency"],
                "timeReferenceDays": preparation["residualTimeReferenceDays"],
                "fractionalFrequencyDriftPerDay": preparation["fractionalFrequencyDriftPerDay"]}}


def interpret_source_switching_temporal_model(preparation: dict[str, Any],
                                              run: dict[str, Any]) -> dict[str, Any]:
    models = run["models"]
    bic_winner = min(models, key=lambda name: models[name]["bic"]) if models else None
    predictive_winner = min(models, key=lambda name: models[name]["heldOutRSS"]) if models else None
    stationary_labels = {"TARGET_STATIONARY": "STATIONARY_TARGET_SOURCE",
                         "CANDIDATE_1_STATIONARY": "STATIONARY_CANDIDATE_1_SOURCE",
                         "CANDIDATE_2_STATIONARY": "STATIONARY_CANDIDATE_2_SOURCE",
                         "TARGET_PLUS_CANDIDATE_1_STATIONARY": "MULTI_SOURCE_STATIONARY_BLEND",
                         "TARGET_PLUS_CANDIDATE_2_STATIONARY": "MULTI_SOURCE_STATIONARY_BLEND",
                         "CANDIDATE_1_PLUS_CANDIDATE_2_STATIONARY": "MULTI_SOURCE_STATIONARY_BLEND",
                         "ALL_SOURCES_STATIONARY": "MULTI_SOURCE_STATIONARY_BLEND"}
    classification = "UNRESOLVED"
    if run.get("sourceIdentifiable"):
        if bic_winner in stationary_labels and predictive_winner == bic_winner:
            classification = stationary_labels[bic_winner]
        elif bic_winner == predictive_winner == "SECTOR_VARYING_SOURCE_AMPLITUDES":
            dominant = []
            for sector in run["perSectorSourceCoherentVectors"]:
                sources = sector["sources"]
                ordered = sorted(sources, key=lambda key: sources[key]["coherentAmplitude"],
                                 reverse=True)
                winner = ordered[0]
                vector = np.array([sources[winner]["sinAmplitude"],
                                   sources[winner]["cosAmplitude"]])
                covariance = np.asarray(sources[winner]["covariance"], float)
                # A two-dimensional coherent vector is conditionally supported by
                # the conventional 95% chi-square confidence ellipse (2 dof).
                supported = float(vector @ np.linalg.pinv(covariance) @ vector) > 5.991464547
                winner_amplitude = sources[winner]["coherentAmplitude"]
                winner_gradient = vector / winner_amplitude if winner_amplitude else np.zeros(2)
                joint_covariance = np.asarray(sector["jointCoherentVectorCovariance"], float)
                winner_index = list(COMPONENT_IDS).index(winner)
                separated = True
                for competitor in ordered[1:]:
                    other = sources[competitor]
                    other_vector = np.array([other["sinAmplitude"], other["cosAmplitude"]])
                    other_amplitude = other["coherentAmplitude"]
                    other_gradient = (other_vector / other_amplitude
                                      if other_amplitude else np.zeros(2))
                    competitor_index = list(COMPONENT_IDS).index(competitor)
                    gradient = np.zeros(6)
                    gradient[[winner_index, 3 + winner_index]] = winner_gradient
                    gradient[[competitor_index, 3 + competitor_index]] = -other_gradient
                    difference_sigma = math.sqrt(max(0., float(
                        gradient @ joint_covariance @ gradient)))
                    # A near tie whose confidence intervals overlap is ambiguous.
                    separated &= winner_amplitude - other_amplitude > 1.959963985 * difference_sigma
                if supported and separated:
                    dominant.append({"sector": sector["sector"], "componentID": winner,
                                     "uncertaintySupported": True})
            identities = {item["componentID"] for item in dominant}
            classification = ("SOURCE_SWITCHING_CONFIRMED"
                              if len(dominant) >= 2 and len(identities) >= 2
                              else "SECTOR_VARIABLE_MULTI_SOURCE")
    candidates = list(preparation.get("catalogCandidates") or [])
    preferred = (candidates[0] if classification == "STATIONARY_CANDIDATE_1_SOURCE"
                 else candidates[1] if classification == "STATIONARY_CANDIDATE_2_SOURCE"
                 else None)
    recommendations = {
        "STATIONARY_TARGET_SOURCE": "TARGET_INTRINSIC_RESIDUAL_MODELING",
        "STATIONARY_CANDIDATE_1_SOURCE": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        "STATIONARY_CANDIDATE_2_SOURCE": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        "MULTI_SOURCE_STATIONARY_BLEND": "JOINT_MULTI_SOURCE_VARIABILITY_MODEL",
        "SECTOR_VARIABLE_MULTI_SOURCE": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
        "SOURCE_SWITCHING_CONFIRMED": "SOURCE_SWITCHING_PHYSICAL_MECHANISM_MODELING",
        "UNRESOLVED": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
    }
    return {"version": "openstar.tess-source-switching-temporal-interpretation.v1",
            "classification": classification, "bicWinningModel": bic_winner,
            "heldOutWinningModel": predictive_winner, "modelComparisons": models,
            "perSectorSourceCoherentVectors": run["perSectorSourceCoherentVectors"],
            "sourceIdentifiable": run.get("sourceIdentifiable", False),
            "preferredCandidate": preferred, "catalogCandidates": candidates,
            "sourceAttributionResolved": classification in {
                "STATIONARY_TARGET_SOURCE", "STATIONARY_CANDIDATE_1_SOURCE",
                "STATIONARY_CANDIDATE_2_SOURCE"},
            "physicalCycleResolved": False, "physicalMechanismResolved": False,
            "referenceFamilyPeriodDays": preparation["referenceFamilyPeriodDays"],
            "subtractedHarmonicOrders": preparation["subtractedHarmonicOrders"],
            "recommendedNextTest": recommendations[classification],
            "validationGuard": "A variable-source result requires both BIC and frozen-ephemeris held-out prediction to select it."}
