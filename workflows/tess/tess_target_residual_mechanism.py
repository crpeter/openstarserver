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
PHASE_GRID_SIZE = 181
CONSTANT_MODEL_PARAMETERS = 3
SMOOTH_MODEL_PARAMETERS = 5
BEAT_MODEL_PARAMETERS = 6
INTERMITTENT_MODEL_PARAMETERS = 2 + ENVELOPE_SEGMENTS
ADJUDICATION_VERSION = "route-independent-all-models-v1"
UNRESOLVED_CLASSIFICATION = "TARGET_RESIDUAL_MECHANISM_UNRESOLVED"

MODEL_BIC_FIELDS = (
    ("CONSTANT_AMPLITUDE", "constantAmplitudeBIC"),
    ("SMOOTH_AMPLITUDE_MODULATION", "smoothEnvelopeBIC"),
    ("COHERENT_TWO_MODE_BEATING", "twoFrequencyBIC"),
    ("EPISODIC_ACTIVATION", "intermittentEnvelopeBIC"),
)


def adjudicate_sector_model_evidence(evidence: Iterable[dict[str, Any]], *,
        fail_closed_reasons: Iterable[str] = ()) -> dict[str, Any]:
    """Adjudicate already-computed sector models without regard to admission route."""
    sectors = []
    labels = []
    sector_ids = []
    adjudication_reasons = list(fail_closed_reasons)
    for source in evidence:
        item = dict(source)
        eligible = [(model, field, float(item[field])) for model, field in MODEL_BIC_FIELDS
                    if item.get(field) is not None]
        eligible.sort(key=lambda candidate: candidate[2])
        best = eligible[0] if eligible else None
        second = eligible[1] if len(eligible) > 1 else None
        delta = second[2] - best[2] if best is not None and second is not None else None
        decisive = delta is not None and delta >= DECISIVE_DELTA_BIC
        gate_blocked = False
        label = UNRESOLVED_CLASSIFICATION
        if decisive and best[0] == "COHERENT_TWO_MODE_BEATING":
            label = "COHERENT_TWO_MODE_BEATING_SUPPORTED"
        elif decisive and best[0] == "SMOOTH_AMPLITUDE_MODULATION":
            label = "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"
        elif decisive and best[0] == "EPISODIC_ACTIVATION":
            if item.get("episodicSuppressionAndReappearance") is True:
                label = "EPISODIC_TARGET_MODE_ACTIVATION"
            else:
                gate_blocked = True
        # A decisively best constant model is recorded but cannot manufacture a
        # physical explanation for a target admitted on nonstationary evidence.
        item.update({
            "bestModel": best[0] if best else None,
            "bestModelBICField": best[1] if best else None,
            "secondBestModel": second[0] if second else None,
            "secondBestModelBICField": second[1] if second else None,
            "deltaBICToSecondBest": delta,
            "decisiveBestModel": decisive,
            "extraMorphologyGateBlockedPromotion": gate_blocked,
            "sectorClassification": label,
        })
        sectors.append(item)
        labels.append(label)
        sector = item.get("sector")
        if not isinstance(sector, int) or isinstance(sector, bool) or sector <= 0:
            sector_ids.append(None)
            adjudication_reasons.append(
                f"sector evidence row {len(sectors) - 1} lacks a valid persisted sector ID")
        else:
            sector_ids.append(sector)

    seen = set()
    duplicates = set()
    for sector in sector_ids:
        if sector is not None and sector in seen:
            duplicates.add(sector)
        elif sector is not None:
            seen.add(sector)
    if duplicates:
        adjudication_reasons.append(
            "duplicate persisted sector evidence IDs: "
            + ", ".join(str(sector) for sector in sorted(duplicates)))

    supporting = {}
    for label, sector in zip(labels, sector_ids):
        if label != UNRESOLVED_CLASSIFICATION and sector is not None:
            supporting.setdefault(label, set()).add(sector)
    replicated_support = {label: sorted(ids) for label, ids in supporting.items()
                          if len(ids) >= MIN_REPLICATING_SECTORS}
    replicated = set(replicated_support)
    promoted = len(replicated) == 1 and not adjudication_reasons
    classification = next(iter(replicated)) if promoted else UNRESOLVED_CLASSIFICATION
    return {
        "classification": classification,
        "recommendedNextTest": ("ASTROPHYSICAL_MECHANISM_INTERPRETATION" if promoted
                                else "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"),
        "sectorModelEvidence": sectors,
        "replicatedMechanisms": sorted(replicated),
        "replicatedMechanismSupportingSectorIDs": replicated_support,
        "failClosedReasons": adjudication_reasons,
    }


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


def _constant_basis(times: list[float], frequency: float) -> list[list[float]]:
    return [[1.0, math.sin(2 * math.pi * frequency * time),
             math.cos(2 * math.pi * frequency * time)] for time in times]


def _shared_phase_envelope_fit(times: list[float], values: list[float], frequency: float,
                               *, segmented: bool) -> tuple[float, list[float], float]:
    """Fit nonnegative scalar amplitudes against one sector-wide carrier phase."""
    low, high = min(times), max(times)
    middle, half_span = (low + high) / 2, max((high - low) / 2, 1e-12)
    best = (float("inf"), [], 0.0)
    for index in range(PHASE_GRID_SIZE):
        phase = 2 * math.pi * index / PHASE_GRID_SIZE
        rows = []
        for time in times:
            carrier = math.sin(2 * math.pi * frequency * time + phase)
            if segmented:
                segment = min(ENVELOPE_SEGMENTS - 1,
                              int((time - low) / max(high - low, 1e-12) * ENVELOPE_SEGMENTS))
                row = [1.0] + [0.0] * ENVELOPE_SEGMENTS
                row[1 + segment] = carrier
            else:
                x = (time - middle) / half_span
                row = [1.0, carrier, x * carrier, x * x * carrier]
            rows.append(row)
        rss, beta = _linear_fit(rows, values)
        if segmented:
            envelope = beta[1:]
        else:
            envelope = [beta[1] + beta[2] * (2 * sample / 100 - 1)
                        + beta[3] * (2 * sample / 100 - 1) ** 2 for sample in range(101)]
        # Negative scalar amplitudes are forbidden: they are hidden pi phase jumps.
        if envelope and min(envelope) >= -1e-12 and rss < best[0]:
            best = (rss, beta, phase)
    return best


def _model_sector(times: list[float], values: list[float], frequency: float) -> dict[str, Any]:
    count, span = len(times), max(times) - min(times)
    constant_rss, _ = _linear_fit(_constant_basis(times, frequency), values)
    smooth_rss, _, smooth_phase = _shared_phase_envelope_fit(
        times, values, frequency, segmented=False)
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
    # Independent-phase segment amplitudes are diagnostic only: they prevent a
    # phase jump from masquerading as suppression.  They do not enter the BIC.
    _, intermittent_beta = _linear_fit(segment_rows, values)
    amplitudes = [math.hypot(intermittent_beta[1 + 2 * index],
                             intermittent_beta[2 + 2 * index]) for index in range(ENVELOPE_SEGMENTS)]
    intermittent_rss, shared_beta, intermittent_phase = _shared_phase_envelope_fit(
        times, values, frequency, segmented=True)
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
        "smoothCarrierPhaseRadians": smooth_phase,
        "intermittentCarrierPhaseRadians": intermittent_phase,
        "intermittentSharedPhaseSegmentAmplitudes": shared_beta[1:] if shared_beta else [],
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

    adjudication = adjudicate_sector_model_evidence(evidence, fail_closed_reasons=reasons)
    return {"classification": adjudication["classification"], "physicalMechanismResolved": False,
            "recommendedNextTest": adjudication["recommendedNextTest"],
            "observable": "frozen v20.12 spatially-decomposed target coefficient series",
            "sectorModelEvidence": adjudication["sectorModelEvidence"],
            "replicatedMechanisms": adjudication["replicatedMechanisms"],
            "replicatedMechanismSupportingSectorIDs":
                adjudication["replicatedMechanismSupportingSectorIDs"],
            "failClosedReasons": adjudication["failClosedReasons"],
            "crossSectorPhaseUsed": False,
            "adjudicationVersion": ADJUDICATION_VERSION,
            "admissionClassification": v2013_result["classification"],
            "preRegisteredRules": {"adjudicationVersion": ADJUDICATION_VERSION,
                "decisiveDeltaBIC": DECISIVE_DELTA_BIC,
                "minimumReplicatingSectors": MIN_REPLICATING_SECTORS,
                "beatGridSize": BEAT_GRID_SIZE, "envelopeSegments": ENVELOPE_SEGMENTS,
                "carrierPhaseGridSize": PHASE_GRID_SIZE,
                "intermittentSuppressionRatio": INTERMITTENT_SUPPRESSION_RATIO,
                "amplitudeEnvelopeConstraint": (
                    "one shared sector carrier phase; scalar envelope nonnegative over fitted domain"),
                "episodicPhaseJumpVeto": (
                    "suppression/reappearance is measured from phase-free segment amplitudes, "
                    "while only the shared-phase envelope enters model BIC"),
                "modelParameterCounts": {"constantAmplitude": CONSTANT_MODEL_PARAMETERS,
                    "smoothEnvelope": SMOOTH_MODEL_PARAMETERS,
                    "twoFrequencyIncludingGridOptimizedSeparation": BEAT_MODEL_PARAMETERS,
                    "intermittentEnvelope": INTERMITTENT_MODEL_PARAMETERS}}}
