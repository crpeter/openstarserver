"""Append-only temporal-phenomenology follow-up for v20.13 target residuals.

The fits in this module are deliberately sector local.  Sector labels and the
origins of historical warped clocks are metadata, never fit coordinates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

DECISIVE_DELTA_BIC = 10.0
MIN_REPLICATING_SECTORS = 2
ENVELOPE_SEGMENTS = 5
BEAT_GRID_SIZE = 81
BEAT_MIN_RESOLUTION_CYCLES = 0.6
BEAT_MAX_FRACTIONAL_SEPARATION = 0.12
INTERMITTENT_SUPPRESSION_RATIO = 3.0
CONSTANT_MODEL_PARAMETERS = 3
SMOOTH_MODEL_PARAMETERS = 7
BEAT_MODEL_PARAMETERS = 6
INTERMITTENT_MODEL_PARAMETERS = 1 + 2 * ENVELOPE_SEGMENTS


def _hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-18:
            return [0.0] * n
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row != column:
                scale = augmented[row][column]
                augmented[row] = [left - scale * right
                                  for left, right in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(n)]


def _linear_fit(rows: list[list[float]], values: list[float]) -> tuple[float, list[float]]:
    width = len(rows[0])
    normal = [[sum(row[i] * row[j] for row in rows) for j in range(width)]
              for i in range(width)]
    rhs = [sum(row[i] * value for row, value in zip(rows, values)) for i in range(width)]
    beta = _solve(normal, rhs)
    rss = sum((value - sum(coefficient * item for coefficient, item in zip(beta, row))) ** 2
              for row, value in zip(rows, values))
    return max(rss, 1e-30), beta


def _bic(rss: float, count: int, parameters: int) -> float:
    return count * math.log(rss / count) + parameters * math.log(count)


def _basis(times: list[float], frequency: float, powers: int = 1) -> list[list[float]]:
    middle = (min(times) + max(times)) / 2
    half_span = max((max(times) - min(times)) / 2, 1e-12)
    rows = []
    for time in times:
        x = (time - middle) / half_span
        sine, cosine = math.sin(2 * math.pi * frequency * time), math.cos(2 * math.pi * frequency * time)
        row = [1.0]
        for power in range(powers):
            row.extend(((x ** power) * sine, (x ** power) * cosine))
        rows.append(row)
    return rows


def _model_sector(times: list[float], values: list[float], frequency: float) -> dict[str, Any]:
    count, span = len(times), max(times) - min(times)
    constant_rss, _ = _linear_fit(_basis(times, frequency), values)
    smooth_rss, _ = _linear_fit(_basis(times, frequency, 3), values)
    minimum_delta = BEAT_MIN_RESOLUTION_CYCLES / max(span, 1e-12)
    maximum_delta = frequency * BEAT_MAX_FRACTIONAL_SEPARATION
    beat_rss, beat_delta = float("inf"), None
    if minimum_delta < maximum_delta:
        for index in range(BEAT_GRID_SIZE):
            delta = minimum_delta + (maximum_delta - minimum_delta) * index / (BEAT_GRID_SIZE - 1)
            rows = [[1.0,
                     math.sin(2 * math.pi * (frequency - delta / 2) * time),
                     math.cos(2 * math.pi * (frequency - delta / 2) * time),
                     math.sin(2 * math.pi * (frequency + delta / 2) * time),
                     math.cos(2 * math.pi * (frequency + delta / 2) * time)] for time in times]
            rss, _ = _linear_fit(rows, values)
            if rss < beat_rss:
                beat_rss, beat_delta = rss, delta

    segment_rows = []
    low, width = min(times), max(span, 1e-12)
    for time in times:
        segment = min(ENVELOPE_SEGMENTS - 1, int((time - low) / width * ENVELOPE_SEGMENTS))
        row = [1.0] + [0.0] * (2 * ENVELOPE_SEGMENTS)
        row[1 + 2 * segment] = math.sin(2 * math.pi * frequency * time)
        row[2 + 2 * segment] = math.cos(2 * math.pi * frequency * time)
        segment_rows.append(row)
    intermittent_rss, intermittent_beta = _linear_fit(segment_rows, values)
    amplitudes = [math.hypot(intermittent_beta[1 + 2 * index],
                             intermittent_beta[2 + 2 * index])
                  for index in range(ENVELOPE_SEGMENTS)]
    weakest = min(range(ENVELOPE_SEGMENTS), key=amplitudes.__getitem__)
    episodic_shape = (0 < weakest < ENVELOPE_SEGMENTS - 1
                      and min(max(amplitudes[:weakest]), max(amplitudes[weakest + 1:]))
                      / max(amplitudes[weakest], 1e-15) >= INTERMITTENT_SUPPRESSION_RATIO)
    return {
        "constantAmplitudeBIC": _bic(constant_rss, count, CONSTANT_MODEL_PARAMETERS),
        "smoothEnvelopeBIC": _bic(smooth_rss, count, SMOOTH_MODEL_PARAMETERS),
        # The separation is selected by minimizing RSS over the preregistered
        # grid, so it is the sixth fitted parameter; it is not a free search.
        "twoFrequencyBIC": (_bic(beat_rss, count, BEAT_MODEL_PARAMETERS)
                            if beat_delta is not None else None),
        "intermittentEnvelopeBIC": _bic(intermittent_rss, count, INTERMITTENT_MODEL_PARAMETERS),
        "beatFrequencySeparation": beat_delta,
        "intermittentSegmentAmplitudes": amplitudes,
        "episodicSuppressionAndReappearance": episodic_shape,
    }


def analyze_target_residual_mechanism(*, preparation: dict[str, Any],
        decomposition: dict[str, Any], v2013_result: dict[str, Any],
        authoritative_artifacts: Iterable[Any], v2013_lineage_verified: bool,
        authoritative_v2013_artifacts: Iterable[Any] = ()) -> dict[str, Any]:
    """Compare preregistered models using only frozen target coefficients."""
    allowed = {"AMPLITUDE_EVOLVING_TARGET_RESIDUAL", "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL"}
    if (v2013_result.get("classification") not in allowed
            or v2013_result.get("recommendedNextTest") != "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
            or v2013_result.get("physicalMechanismResolved") is not False):
        raise RuntimeError("v20.14 requires the exact unresolved v20.13 target-residual boundary.")
    reasons = []
    if not v2013_lineage_verified:
        reasons.append("v20.13 stage/result hash lineage is absent or inconsistent")
    v2013_verified = False
    for reference in authoritative_v2013_artifacts:
        path = reference.path if hasattr(reference, "path") else reference.get("path")
        sha = reference.sha256 if hasattr(reference, "sha256") else reference.get("sha256")
        if not path or not sha or not Path(path).is_file() or _hash(path) != str(sha):
            continue
        try:
            with Path(path).open(encoding="utf-8") as handle:
                frozen_result = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if frozen_result == v2013_result:
            v2013_verified = True
            break
    if not v2013_verified:
        reasons.append("frozen v20.13 artifact SHA or result content is absent or inconsistent")
    frozen = {}
    for reference in authoritative_artifacts:
        path = reference.path if hasattr(reference, "path") else reference.get("path")
        sha = reference.sha256 if hasattr(reference, "sha256") else reference.get("sha256")
        if path and sha:
            frozen[str(Path(path).resolve())] = str(sha)
    evidence = []
    v13_frequencies = {str(item.get("datasetID")): float(item["frequency"])
                       for item in v2013_result.get("temporalModelEvidence") or []
                       if item.get("datasetID") is not None and item.get("frequency") is not None}
    for entry in preparation.get("preparedSeries") or []:
        if (entry.get("componentID") != "target" or entry.get("componentType") != "TARGET"
                or entry.get("combined") or entry.get("sector") is None):
            continue
        paths = [entry.get("coefficientSeriesPath"), entry.get("datasetPath")]
        if any(not path or not Path(path).is_file()
               or frozen.get(str(Path(path).resolve())) != _hash(path) for path in paths):
            reasons.append(f"sector {entry.get('sector')} failed frozen v20.12 SHA verification")
            continue
        with Path(paths[0]).open(encoding="utf-8") as handle:
            component = json.load(handle)
        with Path(paths[1]).open(encoding="utf-8") as handle:
            dataset = json.load(handle)
        if (component.get("componentID") != "target"
                or (dataset.get("science") or {}).get("componentID") != "target"):
            reasons.append(f"sector {entry.get('sector')} is not the target component")
            continue
        times = component.get("absoluteTimes") or component.get("times")
        coordinate = "ORIGINAL_ABSOLUTE_TIME" if component.get("absoluteTimes") else "SECTOR_LOCAL_WARPED_TIME"
        values = component.get("coefficients") or []
        frequency = v13_frequencies.get(str(entry.get("datasetID")))
        if frequency is None or len(times) != len(values) or len(times) < 80:
            reasons.append(f"sector {entry.get('sector')} lacks matching frozen v20.13 evidence")
            continue
        models = _model_sector([float(item) for item in times], [float(item) for item in values], frequency)
        models.update({"sector": entry["sector"], "datasetID": entry["datasetID"],
                       "timingCoordinate": coordinate})
        evidence.append(models)

    sector_labels = []
    mode = v2013_result["classification"]
    for item in evidence:
        constant, smooth = item["constantAmplitudeBIC"], item["smoothEnvelopeBIC"]
        if mode == "AMPLITUDE_EVOLVING_TARGET_RESIDUAL":
            beat = item["twoFrequencyBIC"]
            if beat is not None and min(constant, smooth) - beat >= DECISIVE_DELTA_BIC:
                label = "COHERENT_TWO_MODE_BEATING_SUPPORTED"
            elif min(constant, beat if beat is not None else float("inf")) - smooth >= DECISIVE_DELTA_BIC:
                label = "SMOOTH_SINGLE_MODE_AMPLITUDE_EVOLUTION"
            else:
                label = "AMPLITUDE_EVOLUTION_MECHANISM_UNRESOLVED"
        else:
            episodic = item["intermittentEnvelopeBIC"]
            if (item["episodicSuppressionAndReappearance"]
                    and min(constant, smooth) - episodic >= DECISIVE_DELTA_BIC):
                label = "EPISODIC_TARGET_MODE_ACTIVATION"
            elif min(constant, episodic) - smooth >= DECISIVE_DELTA_BIC:
                label = "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"
            else:
                label = "INTERMITTENCY_MECHANISM_UNRESOLVED"
        item["sectorClassification"] = label
        sector_labels.append(label)
    candidates = [label for label in set(sector_labels) if "UNRESOLVED" not in label
                  and sector_labels.count(label) >= MIN_REPLICATING_SECTORS]
    unresolved = ("AMPLITUDE_EVOLUTION_MECHANISM_UNRESOLVED" if mode.startswith("AMPLITUDE")
                  else "INTERMITTENCY_MECHANISM_UNRESOLVED")
    promoted = len(candidates) == 1 and not reasons
    classification = candidates[0] if promoted else unresolved
    return {"classification": classification, "physicalMechanismResolved": False,
            "recommendedNextTest": "ASTROPHYSICAL_MECHANISM_INTERPRETATION" if promoted
                                   else "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP",
            "observable": "frozen v20.12 spatially-decomposed target coefficient series",
            "sectorModelEvidence": evidence, "failClosedReasons": reasons,
            "crossSectorPhaseUsed": False,
            "preRegisteredRules": {"decisiveDeltaBIC": DECISIVE_DELTA_BIC,
                "minimumReplicatingSectors": MIN_REPLICATING_SECTORS,
                "beatGridSize": BEAT_GRID_SIZE, "envelopeSegments": ENVELOPE_SEGMENTS,
                "intermittentSuppressionRatio": INTERMITTENT_SUPPRESSION_RATIO,
                "modelParameterCounts": {"constantAmplitude": CONSTANT_MODEL_PARAMETERS,
                    "smoothEnvelope": SMOOTH_MODEL_PARAMETERS,
                    "twoFrequencyIncludingGridOptimizedSeparation": BEAT_MODEL_PARAMETERS,
                    "intermittentEnvelope": INTERMITTENT_MODEL_PARAMETERS}}}
