"""Frozen-frequency temporal source model for the residual-localization branch.

All source labels and pixel positions are coordinator-side scientific inputs.  The
calculation deliberately does not create a generic worker work unit.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tess_catalog_guided_localization import COMPONENT_IDS
from .tess_residual_localization import _time_warp, _write_json
from .tess_residual_phase_difference_image import _production_difference_image_inputs

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


def _templates(item: dict[str, Any], valid: np.ndarray) -> np.ndarray:
    supplied = item.get("sourceTemplates")
    if supplied is not None:
        templates = np.asarray(supplied, dtype=float)
        if templates.shape[0] != 3:
            raise RuntimeError("sourceTemplates must contain target and both candidates.")
        templates = templates.reshape(3, -1)[:, valid.ravel()].T
    else:
        centers = list(item.get("componentPixelCenters") or
                       (item.get("acquisitionProvenance") or {}).get("componentPixelCenters") or [])
        if [x.get("componentID") for x in centers] != list(COMPONENT_IDS):
            raise RuntimeError("All three frozen component pixel positions are required.")
        yy, xx = np.indices(valid.shape)
        templates = np.column_stack([
            np.exp(-.5 * ((xx[valid] - float(c["x"])) ** 2 +
                          (yy[valid] - float(c["y"])) ** 2) / .7 ** 2) for c in centers])
    norms = np.sqrt(np.sum(templates * templates, axis=0))
    if np.any(norms <= 0):
        raise RuntimeError("A frozen source template has zero support.")
    return templates / norms


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
    inputs = _production_difference_image_inputs(preparation) if sector_inputs is None else sector_inputs
    sectors = []
    for item in inputs:
        times = np.asarray(item["times"], float)
        cube = np.asarray(item["prewhitened"], float)
        valid = np.asarray(item["valid"], bool)
        finite = np.isfinite(times) & np.all(np.isfinite(cube[:, valid]), axis=1)
        times, values = times[finite], cube[finite][:, valid]
        template = _templates(item, valid)
        x = _design(times, template, preparation)
        y = values.reshape(-1)
        sectors.append({"sector": int(item["sector"]), "times": times, "x": x, "y": y,
                        "templateRank": int(np.linalg.matrix_rank(template)),
                        "templateConditionNumber": float(np.linalg.cond(template))})
    if [x["sector"] for x in sectors] != list(preparation["sectors"]):
        raise RuntimeError("Temporal model inputs do not match the frozen sector ordering.")

    source_sets = {"TARGET_STATIONARY": (0,), "CANDIDATE_1_STATIONARY": (1,),
                   "CANDIDATE_2_STATIONARY": (2,),
                   "TARGET_PLUS_CANDIDATE_1_STATIONARY": (0, 1),
                   "ALL_SOURCES_STATIONARY": (0, 1, 2)}
    models: dict[str, Any] = {}
    all_y = np.concatenate([s["y"] for s in sectors])
    for name, sources in source_sets.items():
        columns = list(sources) + [3 + j for j in sources]
        fit = _fit(np.vstack([s["x"][:, columns] for s in sectors]), all_y)
        models[name] = {"rss": fit["rss"], "bic": len(all_y) * math.log(fit["rss"] / len(all_y))
                        + fit["k"] * math.log(len(all_y)), "parameterCount": fit["k"],
                        "rank": fit["rank"]}

    vectors = []
    varying_rss = 0.0
    varying_k = 0
    identifiable = True
    for sector in sectors:
        fit = _fit(sector["x"], sector["y"])
        varying_rss += fit["rss"]; varying_k += fit["k"]
        identifiable &= fit["rank"] == 6 and sector["templateRank"] == 3
        entry = {"sector": sector["sector"], "templateRank": sector["templateRank"],
                 "templateConditionNumber": sector["templateConditionNumber"], "sources": {}}
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
        "rank": sum(6 for _ in sectors) if identifiable else None}

    # Frozen-parameter prediction: train on alternating cadences, predict the held-out
    # cadences. This tests amplitude generalization without frequency refitting.
    cv = {name: 0.0 for name in models}
    for name, sources in {**source_sets, "SECTOR_VARYING_SOURCE_AMPLITUDES": (0, 1, 2)}.items():
        columns = list(sources) + [3 + j for j in sources]
        if name == "SECTOR_VARYING_SOURCE_AMPLITUDES":
            for s in sectors:
                pixels = len(s["y"]) // len(s["times"]); cadence = np.repeat(np.arange(len(s["times"])), pixels)
                train = cadence % 2 == 0
                fit = _fit(s["x"][train][:, columns], s["y"][train])
                cv[name] += float(np.sum((s["y"][~train] - s["x"][~train][:, columns] @ fit["beta"]) ** 2))
        else:
            train_x=[]; train_y=[]; test=[]
            for s in sectors:
                pixels=len(s["y"])//len(s["times"]); cadence=np.repeat(np.arange(len(s["times"])), pixels)
                mask=cadence%2==0; train_x.append(s["x"][mask][:, columns]); train_y.append(s["y"][mask])
                test.append((s["x"][~mask][:, columns], s["y"][~mask]))
            fit=_fit(np.vstack(train_x), np.concatenate(train_y))
            cv[name]=sum(float(np.sum((y-x@fit["beta"])**2)) for x,y in test)
        models[name]["heldOutRSS"] = cv[name]
    return {"version": "openstar.tess-source-switching-temporal-run.v1",
            "models": models, "perSectorSourceCoherentVectors": vectors,
            "sourceIdentifiable": bool(identifiable), "physicalCycleResolved": False,
            "frozenResidualEphemeris": {"frequency": preparation["residualReferenceFrequency"],
                "timeReferenceDays": preparation["residualTimeReferenceDays"],
                "fractionalFrequencyDriftPerDay": preparation["fractionalFrequencyDriftPerDay"]}}


def interpret_source_switching_temporal_model(preparation: dict[str, Any],
                                              run: dict[str, Any]) -> dict[str, Any]:
    models = run["models"]
    bic_winner = min(models, key=lambda name: models[name]["bic"])
    predictive_winner = min(models, key=lambda name: models[name]["heldOutRSS"])
    stationary_labels = {"TARGET_STATIONARY": "STATIONARY_TARGET_SOURCE",
                         "CANDIDATE_1_STATIONARY": "STATIONARY_CANDIDATE_1_SOURCE",
                         "CANDIDATE_2_STATIONARY": "STATIONARY_CANDIDATE_2_SOURCE",
                         "TARGET_PLUS_CANDIDATE_1_STATIONARY": "MULTI_SOURCE_STATIONARY_BLEND",
                         "ALL_SOURCES_STATIONARY": "MULTI_SOURCE_STATIONARY_BLEND"}
    classification = "UNRESOLVED"
    if run.get("sourceIdentifiable"):
        if bic_winner in stationary_labels and predictive_winner == bic_winner:
            classification = stationary_labels[bic_winner]
        elif bic_winner == predictive_winner == "SECTOR_VARYING_SOURCE_AMPLITUDES":
            dominant = []
            for sector in run["perSectorSourceCoherentVectors"]:
                dominant.append(max(sector["sources"], key=lambda key:
                                    sector["sources"][key]["coherentAmplitude"]))
            classification = ("SOURCE_SWITCHING_CONFIRMED" if len(set(dominant)) > 1
                              else "SECTOR_VARIABLE_MULTI_SOURCE")
    return {"version": "openstar.tess-source-switching-temporal-interpretation.v1",
            "classification": classification, "bicWinningModel": bic_winner,
            "heldOutWinningModel": predictive_winner, "modelComparisons": models,
            "perSectorSourceCoherentVectors": run["perSectorSourceCoherentVectors"],
            "sourceIdentifiable": run.get("sourceIdentifiable", False),
            "sourceAttributionResolved": classification in {
                "STATIONARY_TARGET_SOURCE", "STATIONARY_CANDIDATE_1_SOURCE",
                "STATIONARY_CANDIDATE_2_SOURCE"},
            "physicalCycleResolved": False, "physicalMechanismResolved": False,
            "referenceFamilyPeriodDays": preparation["referenceFamilyPeriodDays"],
            "subtractedHarmonicOrders": preparation["subtractedHarmonicOrders"],
            "recommendedNextTest": None,
            "validationGuard": "A variable-source result requires both BIC and frozen-ephemeris held-out prediction to select it."}
