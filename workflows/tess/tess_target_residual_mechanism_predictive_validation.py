"""Local, held-out validation of the frozen v20.14 temporal model families.

Every domain quantity is constructed from the complete, frozen sector series;
only after that construction are deterministic temporal blocks masked.  This is
important: this module must never turn a holdout into a different model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from openstar_investigation import sha256_file, sha256_json
from .tess_target_residual_mechanism import (
    BEAT_GRID_SIZE, BEAT_MAX_FRACTIONAL_SEPARATION,
    BEAT_MIN_RESOLUTION_CYCLES, ENVELOPE_SEGMENTS, PHASE_GRID_SIZE,
)

PREDICTIVE_FOLDS = 4
DECISIVE_PREDICTIVE_DELTA_LOG_LIKELIHOOD = 10.0
MIN_PREDICTIVE_FOLD_WINS = 3
MIN_REPLICATING_SECTORS = 2
VALIDATION_VERSION = "held-out-stratified-blocked-all-models-v1"
UNRESOLVED = "TARGET_RESIDUAL_MECHANISM_PREDICTIVE_VALIDATION_UNRESOLVED"
MODEL_NAMES = ("CONSTANT_AMPLITUDE", "SMOOTH_AMPLITUDE_MODULATION",
               "COHERENT_TWO_MODE_BEATING", "EPISODIC_ACTIVATION")
MODEL_LABELS = {
    "SMOOTH_AMPLITUDE_MODULATION":
        "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION_PREDICTIVELY_VALIDATED",
    "COHERENT_TWO_MODE_BEATING":
        "COHERENT_TWO_MODE_BEATING_PREDICTIVELY_VALIDATED",
    "EPISODIC_ACTIVATION":
        "EPISODIC_TARGET_MODE_ACTIVATION_PREDICTIVELY_VALIDATED",
}


def v2013_lineage_matches(*, stage_input_hashes: dict[str, Any],
                          result_input_provenance: dict[str, Any],
                          preparation: dict[str, Any], interpretation: dict[str, Any]) -> bool:
    """Independently bind v20.13 to the exact v20.12 snapshots consumed now."""
    preparation_hash = sha256_json(preparation)
    interpretation_hash = sha256_json(interpretation)
    return (stage_input_hashes.get("v20.12Preparation") == preparation_hash
            and stage_input_hashes.get("v20.12Interpretation") == interpretation_hash
            and result_input_provenance.get("v20.12PreparationResultHash") == preparation_hash
            and result_input_provenance.get("v20.12InterpretationResultHash") == interpretation_hash)


def freeze_model_domain(times: list[float], frequency: float) -> dict[str, Any]:
    low, high = min(times), max(times)
    span = high - low
    if not math.isfinite(low) or not math.isfinite(high) or span <= 0:
        raise ValueError("sector has no finite nonzero time domain")
    boundaries = [low + span * index / ENVELOPE_SEGMENTS
                  for index in range(ENVELOPE_SEGMENTS + 1)]
    minimum = BEAT_MIN_RESOLUTION_CYCLES / span
    maximum = frequency * BEAT_MAX_FRACTIONAL_SEPARATION
    if not math.isfinite(frequency) or frequency <= 0 or minimum >= maximum:
        raise ValueError("sector has no valid preregistered beating separation domain")
    grid = [minimum + (maximum - minimum) * index / (BEAT_GRID_SIZE - 1)
            for index in range(BEAT_GRID_SIZE)]
    return {"fullDomainStart": low, "fullDomainEnd": high,
            "fullDomainMidpoint": (low + high) / 2,
            "fullDomainHalfSpan": span / 2,
            "intermittentSegmentBoundaries": boundaries,
            "beatMinimumDelta": minimum, "beatMaximumDelta": maximum,
            "beatDeltaGrid": grid}


def _segment(time: float, domain: dict[str, Any]) -> int:
    low, high = domain["fullDomainStart"], domain["fullDomainEnd"]
    return min(ENVELOPE_SEGMENTS - 1,
               max(0, int((time - low) / (high - low) * ENVELOPE_SEGMENTS)))


def construct_folds(times: list[float], domain: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic, segment-stratified chronological blocked folds."""
    by_segment = [[] for _ in range(ENVELOPE_SEGMENTS)]
    for index in sorted(range(len(times)), key=times.__getitem__):
        by_segment[_segment(times[index], domain)].append(index)
    if any(len(indices) < PREDICTIVE_FOLDS for indices in by_segment):
        raise ValueError("every full-domain segment must contain at least four samples")
    blocks = []
    for indices in by_segment:
        blocks.append([indices[len(indices) * fold // PREDICTIVE_FOLDS:
                               len(indices) * (fold + 1) // PREDICTIVE_FOLDS]
                       for fold in range(PREDICTIVE_FOLDS)])
    folds = []
    universe = set(range(len(times)))
    for fold in range(PREDICTIVE_FOLDS):
        held = sorted(index for segment in blocks for index in segment[fold])
        if any(not segment[fold] for segment in blocks):
            raise ValueError("a held-out segment sub-block is empty")
        folds.append({"fold": fold, "trainingIndices": sorted(universe - set(held)),
                      "heldOutIndices": held,
                      "heldOutBlocks": [{"segment": segment,
                          "count": len(blocks[segment][fold]),
                          "timeStart": times[blocks[segment][fold][0]],
                          "timeEnd": times[blocks[segment][fold][-1]]}
                          for segment in range(ENVELOPE_SEGMENTS)]})
    return folds


def _fit(rows: list[list[float]], values: list[float]) -> tuple[float, list[float]]:
    if not rows or len(rows) <= len(rows[0]):
        raise ValueError("insufficient training samples")
    width = len(rows[0])
    a = [[sum(row[i] * row[j] for row in rows) for j in range(width)]
         for i in range(width)]
    b = [sum(row[i] * value for row, value in zip(rows, values)) for i in range(width)]
    scale = max((abs(value) for row in a for value in row), default=1.0)
    for column in range(width):
        pivot = max(range(column, width), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) <= max(scale * 1e-12, 1e-18):
            raise ValueError("singular training design")
        a[column], a[pivot], b[column], b[pivot] = a[pivot], a[column], b[pivot], b[column]
        divisor = a[column][column]
        for item in range(column, width): a[column][item] /= divisor
        b[column] /= divisor
        for row in range(width):
            if row == column: continue
            factor = a[row][column]
            for item in range(column, width): a[row][item] -= factor * a[column][item]
            b[row] -= factor * b[column]
    rss = sum((value - sum(x * y for x, y in zip(row, b))) ** 2
              for row, value in zip(rows, values))
    return rss, b


def _rows(model: str, times: list[float], frequency: float, domain: dict[str, Any],
          *, phase: float | None = None, delta: float | None = None) -> list[list[float]]:
    if model == "CONSTANT_AMPLITUDE":
        return [[1., math.sin(2*math.pi*frequency*t), math.cos(2*math.pi*frequency*t)] for t in times]
    if model == "COHERENT_TWO_MODE_BEATING":
        return [[1., math.sin(2*math.pi*(frequency-delta/2)*t),
                 math.cos(2*math.pi*(frequency-delta/2)*t),
                 math.sin(2*math.pi*(frequency+delta/2)*t),
                 math.cos(2*math.pi*(frequency+delta/2)*t)] for t in times]
    rows = []
    for t in times:
        carrier = math.sin(2*math.pi*frequency*t + phase)
        if model == "SMOOTH_AMPLITUDE_MODULATION":
            x = (t-domain["fullDomainMidpoint"])/domain["fullDomainHalfSpan"]
            rows.append([1., carrier, x*carrier, x*x*carrier])
        else:
            row = [1.] + [0.] * ENVELOPE_SEGMENTS
            row[1 + _segment(t, domain)] = carrier
            rows.append(row)
    return rows


def fit_training_model(model: str, times: list[float], values: list[float],
                       frequency: float, domain: dict[str, Any]) -> dict[str, Any]:
    candidates = [(None, None)]
    if model == "COHERENT_TWO_MODE_BEATING":
        candidates = [(None, delta) for delta in domain["beatDeltaGrid"]]
    elif model in {"SMOOTH_AMPLITUDE_MODULATION", "EPISODIC_ACTIVATION"}:
        candidates = [(2*math.pi*index/PHASE_GRID_SIZE, None)
                      for index in range(PHASE_GRID_SIZE)]
    best = None
    for phase, delta in candidates:
        try:
            rows = _rows(model, times, frequency, domain, phase=phase, delta=delta)
            rss, beta = _fit(rows, values)
        except ValueError:
            continue
        if model == "SMOOTH_AMPLITUDE_MODULATION":
            envelope = [beta[1]+beta[2]*(-1+2*i/100)+beta[3]*(-1+2*i/100)**2
                        for i in range(101)]
            if min(envelope) < -1e-12: continue
        if model == "EPISODIC_ACTIVATION" and min(beta[1:]) < -1e-12: continue
        if best is None or rss < best["trainingRSS"]:
            best = {"trainingRSS": rss, "coefficients": beta,
                    "carrierPhaseRadians": phase, "beatFrequencySeparation": delta}
    if best is None:
        raise ValueError(f"no identifiable valid nonnegative {model} training fit")
    return best


def validate_sector(times: list[float], values: list[float], frequency: float,
                    *, sector: int, dataset_id: str, timing_coordinate: str,
                    episodic_morphology: bool) -> dict[str, Any]:
    domain = freeze_model_domain(times, frequency)
    folds = construct_folds(times, domain)
    model_folds = {model: [] for model in MODEL_NAMES}
    reasons = []
    for fold in folds:
        train, held = fold["trainingIndices"], fold["heldOutIndices"]
        train_times, train_values = [times[i] for i in train], [values[i] for i in train]
        held_times, held_values = [times[i] for i in held], [values[i] for i in held]
        for model in MODEL_NAMES:
            try:
                fitted = fit_training_model(model, train_times, train_values, frequency, domain)
                test_rows = _rows(model, held_times, frequency, domain,
                                  phase=fitted["carrierPhaseRadians"],
                                  delta=fitted["beatFrequencySeparation"])
                residuals = [value-sum(a*b for a,b in zip(row, fitted["coefficients"]))
                             for row,value in zip(test_rows, held_values)]
                held_rss = sum(value*value for value in residuals)
                variance = max(fitted["trainingRSS"]/len(train), 1e-30)
                loglike = -.5*(len(held)*math.log(2*math.pi*variance)+held_rss/variance)
                fitted.update({"fold": fold["fold"], "trainingSampleCount": len(train),
                    "heldOutSampleCount": len(held), "trainingVariance": variance,
                    "heldOutRSS": held_rss, "heldOutLogLikelihood": loglike})
                if model == "EPISODIC_ACTIVATION":
                    fitted["segmentAmplitudes"] = fitted["coefficients"][1:]
                model_folds[model].append(fitted)
            except ValueError as error:
                reasons.append(f"fold {fold['fold']} {model}: {error}")
    totals = {model: sum(item["heldOutLogLikelihood"] for item in rows)
              for model, rows in model_folds.items() if len(rows) == PREDICTIVE_FOLDS}
    wins = {model: 0 for model in MODEL_NAMES}
    if len(totals) == len(MODEL_NAMES):
        for fold in range(PREDICTIVE_FOLDS):
            scores = {model: model_folds[model][fold]["heldOutLogLikelihood"] for model in MODEL_NAMES}
            maximum = max(scores.values())
            winners = [model for model, score in scores.items() if score == maximum]
            if len(winners) == 1: wins[winners[0]] += 1
    fair = not reasons and all(len(model_folds[model]) == PREDICTIVE_FOLDS
                               for model in MODEL_NAMES)
    ranking = sorted(totals, key=lambda model: totals[model], reverse=True)
    best, second = (ranking + [None, None])[:2]
    delta = totals[best]-totals[second] if best and second else None
    predictive_winner_meets_rules = bool(best and delta >= DECISIVE_PREDICTIVE_DELTA_LOG_LIKELIHOOD
                    and wins[best] >= MIN_PREDICTIVE_FOLD_WINS)
    decisive = fair and predictive_winner_meets_rules
    blocked = decisive and best == "EPISODIC_ACTIVATION" and episodic_morphology is not True
    label = MODEL_LABELS.get(best, UNRESOLVED) if decisive and not blocked else UNRESOLVED
    return {"sector": sector, "datasetID": dataset_id, "timingCoordinate": timing_coordinate,
        "baseFrequency": frequency, "frozenModelDomain": domain,
        "foldConstruction": [{"fold": f["fold"], "heldOutBlocks": f["heldOutBlocks"],
                              "trainingSampleCount": len(f["trainingIndices"]),
                              "heldOutSampleCount": len(f["heldOutIndices"])} for f in folds],
        "modelFoldEvidence": model_folds, "totalHeldOutLogLikelihoodByModel": totals,
        "bestPredictiveModel": best, "secondBestPredictiveModel": second,
        "bestTotalHeldOutLogLikelihood": totals.get(best),
        "secondBestTotalHeldOutLogLikelihood": totals.get(second),
        "predictiveDeltaLogLikelihood": delta, "foldWinsByModel": wins,
        "fairAllModelComparisonCompleted": fair,
        "predictiveWinnerMeetsNumericalRules": predictive_winner_meets_rules,
        "decisivePredictiveWinner": decisive, "frozenEpisodicSuppressionAndReappearance": episodic_morphology,
        "morphologyGateBlockedPromotion": blocked, "sectorClassification": label,
        "failClosedReasons": reasons}


def _verified_artifact(result: dict[str, Any], artifacts: Iterable[Any], filename: str | tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    filenames = (filename,) if isinstance(filename, str) else filename
    for reference in artifacts:
        path = Path(reference.path if hasattr(reference, "path") else reference.get("path", ""))
        expected = str(reference.sha256 if hasattr(reference, "sha256") else reference.get("sha256", ""))
        if path.name not in filenames or not expected or not path.is_file() or sha256_file(path) != expected:
            continue
        with path.open(encoding="utf-8") as handle: frozen = json.load(handle)
        if frozen != result: raise RuntimeError(f"{filenames[0]} frozen artifact and persisted result differ")
        return expected, frozen
    raise RuntimeError(f"{filenames[0]} SHA verification failed")


def adjudicate_predictive_sectors(sectors: Iterable[dict[str, Any]], *,
                                  fail_closed_reasons: Iterable[str] = ()) -> dict[str, Any]:
    """Apply only the preregistered distinct-sector target promotion rule."""
    rows = list(sectors)
    reasons = list(fail_closed_reasons)
    ids = []
    support: dict[str, set[int]] = {}
    for index, row in enumerate(rows):
        sector = row.get("sector")
        if not isinstance(sector, int) or isinstance(sector, bool) or sector <= 0:
            reasons.append(f"sector predictive row {index} lacks a valid persisted sector ID")
            continue
        ids.append(sector)
        label = row.get("sectorClassification")
        if (row.get("fairAllModelComparisonCompleted") is True
                and not row.get("failClosedReasons") and label != UNRESOLVED):
            support.setdefault(label, set()).add(sector)
    duplicates = sorted({sector for sector in ids if ids.count(sector) > 1})
    if duplicates:
        reasons.append("duplicate persisted sector IDs: " + ", ".join(map(str, duplicates)))
    replicated_support = {key: sorted(value) for key, value in support.items()
                          if len(value) >= MIN_REPLICATING_SECTORS}
    replicated = sorted(replicated_support)
    promoted = len(replicated) == 1 and not reasons
    return {"classification": replicated[0] if promoted else UNRESOLVED,
        "recommendedNextTest": ("ASTROPHYSICAL_MECHANISM_INTERPRETATION" if promoted else
                                 "ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION"),
        "replicatedPredictiveMechanisms": replicated,
        "replicatedPredictiveMechanismSupportingSectorIDs": replicated_support,
        "failClosedReasons": reasons}


def analyze_predictive_validation(*, preparation: dict[str, Any], v2013_result: dict[str, Any],
        v2014_result: dict[str, Any], adjudication_result: dict[str, Any],
        adjudication_stage_id: str, adjudication_handler_id: str,
        preparation_artifacts: Iterable[Any], v2013_artifacts: Iterable[Any],
        v2014_artifacts: Iterable[Any], adjudication_artifacts: Iterable[Any],
        lineage_verified: bool) -> dict[str, Any]:
    """Verify all snapshots, then perform the only new work: local training fits."""
    if adjudication_handler_id not in {
            "openstar.tess.target-residual-mechanism.analyze",
            "openstar.tess.target-residual-mechanism-adjudication.analyze"}:
        raise RuntimeError("v20.16 adjudication source handler is not authoritative")
    if (adjudication_result.get("classification") != "TARGET_RESIDUAL_MECHANISM_UNRESOLVED"
            or adjudication_result.get("recommendedNextTest") != "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
            or adjudication_result.get("physicalMechanismResolved") is not False
            or adjudication_result.get("failClosedReasons")):
        raise RuntimeError("v20.16 requires the exact fail-open unresolved adjudication boundary")
    if (adjudication_handler_id == "openstar.tess.target-residual-mechanism.analyze"
            and adjudication_result.get("adjudicationVersion")
                != "route-independent-all-models-v1"):
        raise RuntimeError("direct v20.14 admission requires corrected route-independent semantics")
    adjudication_name = ("target-residual-mechanism-adjudication-v20.15.json"
        if adjudication_handler_id.endswith("adjudication.analyze") else "target-residual-mechanism-v20.14.json")
    adjudication_sha, _ = _verified_artifact(adjudication_result, adjudication_artifacts, adjudication_name)
    v14_sha, _ = _verified_artifact(v2014_result, v2014_artifacts, "target-residual-mechanism-v20.14.json")
    v13_sha, _ = _verified_artifact(v2013_result, v2013_artifacts,
        ("intrinsic-nonstationary-v20.13.json", "intrinsic-nonstationary-v20.31.json"))
    if adjudication_handler_id.endswith("adjudication.analyze"):
        source = adjudication_result.get("inputProvenance") or {}
        if (source.get("frozenV20.14ResultHash") != sha256_json(v2014_result)
                or source.get("frozenV20.14ArtifactSHA256") != v14_sha):
            raise RuntimeError("v20.15 input provenance does not identify the frozen v20.14 snapshot")
    if not lineage_verified: raise RuntimeError("frozen dataset/frequency/adjudication lineage is inconsistent")
    frozen_paths = {}
    for ref in preparation_artifacts:
        path = Path(ref.path if hasattr(ref, "path") else ref.get("path", ""))
        sha = str(ref.sha256 if hasattr(ref, "sha256") else ref.get("sha256", ""))
        if path.is_file() and sha and sha256_file(path) == sha: frozen_paths[str(path.resolve())] = sha
    frequencies = {str(row.get("datasetID")): float(row["frequency"])
                   for row in v2013_result.get("temporalModelEvidence") or [] if row.get("frequency") is not None}
    morphology = {str(row.get("datasetID")): row.get("episodicSuppressionAndReappearance") is True
                  for row in v2014_result.get("sectorModelEvidence") or []}
    sectors, hashes, reasons = [], {}, []
    for entry in preparation.get("preparedSeries") or []:
        if entry.get("componentID") != "target" or entry.get("componentType") != "TARGET" or entry.get("combined"): continue
        sector, dataset_id = entry.get("sector"), str(entry.get("datasetID"))
        if not isinstance(sector, int) or isinstance(sector, bool) or sector <= 0:
            reasons.append("invalid persisted sector ID"); continue
        coefficient, dataset = Path(entry.get("coefficientSeriesPath", "")), Path(entry.get("datasetPath", ""))
        coefficient_sha, dataset_sha = frozen_paths.get(str(coefficient.resolve())), frozen_paths.get(str(dataset.resolve()))
        if not coefficient_sha or not dataset_sha:
            reasons.append(f"sector {sector} failed frozen v20.12 artifact SHA verification"); continue
        with coefficient.open(encoding="utf-8") as handle: series = json.load(handle)
        with dataset.open(encoding="utf-8") as handle: dataset_value = json.load(handle)
        times, values = series.get("absoluteTimes") or series.get("times"), series.get("coefficients")
        if (series.get("componentID") != "target" or (dataset_value.get("science") or {}).get("componentID") != "target"
                or not times or len(times) != len(values or []) or dataset_id not in frequencies):
            reasons.append(f"sector {sector} lacks matching target series/frequency lineage"); continue
        hashes[dataset_id] = {"sector": sector, "coefficientSeriesSHA256": coefficient_sha,
                              "datasetSHA256": dataset_sha}
        try:
            sectors.append(validate_sector([float(v) for v in times], [float(v) for v in values],
                frequencies[dataset_id], sector=sector, dataset_id=dataset_id,
                timing_coordinate=("ORIGINAL_ABSOLUTE_TIME" if series.get("absoluteTimes") else "SECTOR_LOCAL_WARPED_TIME"),
                episodic_morphology=morphology.get(dataset_id, False)))
        except ValueError as error: reasons.append(f"sector {sector}: {error}")
    reasons.extend(reason for sector in sectors for reason in sector["failClosedReasons"])
    adjudication = adjudicate_predictive_sectors(sectors, fail_closed_reasons=reasons)
    return {"validationVersion": VALIDATION_VERSION,
        "classification": adjudication["classification"],
        "physicalMechanismResolved": False,
        "recommendedNextTest": adjudication["recommendedNextTest"],
        "observable": "frozen v20.12 spatially-decomposed target coefficient series",
        "adjudicationSource": {"stageID": adjudication_stage_id, "handlerID": adjudication_handler_id,
            "resultHash": sha256_json(adjudication_result), "artifactSHA256": adjudication_sha},
        "frozenV20.14ResultHash": sha256_json(v2014_result), "frozenV20.14ArtifactSHA256": v14_sha,
        "frozenV20.13ResultHash": sha256_json(v2013_result), "frozenV20.13ArtifactSHA256": v13_sha,
        "frozenV20.12ArtifactsByDataset": hashes, "sectorPredictiveEvidence": sectors,
        "replicatedPredictiveMechanisms": adjudication["replicatedPredictiveMechanisms"],
        "replicatedPredictiveMechanismSupportingSectorIDs":
            adjudication["replicatedPredictiveMechanismSupportingSectorIDs"],
        "failClosedReasons": adjudication["failClosedReasons"], "crossSectorPhaseUsed": False,
        "foldConstruction": {"method": "five-segment-stratified-chronological-contiguous-blocks",
                             "foldCount": PREDICTIVE_FOLDS},
        "preregisteredRules": {"predictiveFoldCount": PREDICTIVE_FOLDS,
            "decisivePredictiveDeltaLogLikelihood": DECISIVE_PREDICTIVE_DELTA_LOG_LIKELIHOOD,
            "minimumPredictiveFoldWins": MIN_PREDICTIVE_FOLD_WINS,
            "minimumReplicatingSectors": MIN_REPLICATING_SECTORS, "models": list(MODEL_NAMES),
            "phaseGridSize": PHASE_GRID_SIZE, "beatGridSize": BEAT_GRID_SIZE,
            "beatMinimumResolutionCycles": BEAT_MIN_RESOLUTION_CYCLES,
            "beatMaximumFractionalSeparation": BEAT_MAX_FRACTIONAL_SEPARATION,
            "envelopeSegments": ENVELOPE_SEGMENTS,
            "modelDomainFrozenBeforeHoldoutMasking": True,
            "episodicMorphologyVetoRequired": True, "crossSectorPhaseForbidden": True},
        "localModelFittingPerformed": True, "distributedWorkPerformed": False,
        "archiveQueryPerformed": False}
