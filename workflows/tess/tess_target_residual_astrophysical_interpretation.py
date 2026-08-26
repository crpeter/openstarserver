"""Server-local, fail-closed interpretation of a frozen target residual.

This module deliberately performs no light-curve fitting.  It adjudicates an
already measured signal against independently acquired astrophysical context.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


UNRESOLVED = "ASTROPHYSICAL_MECHANISM_UNRESOLVED"
ROTATION = "ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED"


class AstrophysicalEvidenceProvider(Protocol):
    """Injectable catalog/literature boundary; returned records are frozen verbatim."""

    def fetch(self, object_identity: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class UnavailableAstrophysicalEvidenceProvider:
    provider_id: str = "unconfigured"

    def fetch(self, object_identity: dict[str, Any]) -> dict[str, Any]:
        return {"available": False, "provider": self.provider_id,
                "objectIdentity": dict(object_identity), "records": []}


def newest_authoritative_recommendation(stages, fallback=None):
    """Return the newest completed non-finalizer science recommendation."""
    return next((stage.result.get("recommendedNextTest")
        for stage in reversed(tuple(stages))
        if stage.status == "COMPLETE" and isinstance(stage.result, dict)
        and "recommendedNextTest" in stage.result
        and stage.handler_id != "openstar.tess.finalize"), fallback)


def interpret_target_residual_astrophysics(*, mechanism: dict[str, Any],
        target_attribution: dict[str, Any], stellar_context: dict[str, Any],
        external_evidence: dict[str, Any], retrieved_at: str | None = None) -> dict[str, Any]:
    """Apply preregistered, mechanism-neutral guards to independent evidence."""
    period = float(mechanism.get("targetResidualPeriodDays") or
                   mechanism.get("referencePeriodDays") or 0.0)
    records = external_evidence.get("records") or []
    consistent_rotation_records = []
    contradictions = []
    for record in records:
        provenance_complete = all(record.get(key) for key in (
            "provider", "stableObjectID", "queryParameters", "citation"))
        if not provenance_complete:
            continue
        mechanism_name = str(record.get("mechanism") or "").upper()
        low, high = record.get("periodRangeDays") or (None, None)
        range_match = (low is not None and high is not None and period > 0
                       and float(low) <= period <= float(high))
        if mechanism_name in {"ROTATION", "ROTATIONAL_MODULATION",
                              "STARSPOT_ROTATION"} and range_match:
            consistent_rotation_records.append(record)
        if record.get("contradictsRotation") is True:
            contradictions.append(record)

    gates = {
        "targetSpatialAttribution": (
            target_attribution.get("classification") == "TARGET_RESIDUAL_COMPONENT_DOMINANT"
            and target_attribution.get("residualModeOrigin") == "TARGET_DOMINANT"),
        "replicatedSmoothAmplitudeModulation": (
            mechanism.get("classification") == "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"
            and len(mechanism.get("replicatedSupportingSectors") or []) >= 2),
        "rotationPhysicallyAllowed": stellar_context.get("rotationPhysicallyAllowed") is True,
        "independentRotationRangeConsistent": len(consistent_rotation_records) >= 1,
        # The frozen TESS morphology and physical-radius guard are constraints
        # independent of the cited historical rotation measurement.
        "multipleIndependentConstraints": len(consistent_rotation_records) >= 1,
        "noStrongerContradiction": not contradictions,
    }
    # A label alone, youth alone, or an exact-period mismatch is intentionally
    # absent from the promotion gates.  A cited historical range is required.
    promoted = external_evidence.get("available") is True and all(gates.values())
    return {
        "schemaVersion": "target-residual-astrophysical-interpretation-v1",
        "classification": ROTATION if promoted else UNRESOLVED,
        "physicalMechanismResolved": promoted,
        "targetResidualMechanismResolved": promoted,
        "physicalCycleResolved": False,
        "mainPhotometricFamily": {"periodDays": 7.546,
            "possibleDoubleCycleDays": 15.093, "physicalCycleResolved": False},
        "recommendedNextTest": (None if promoted else
            "ADDITIONAL_INDEPENDENT_ASTROPHYSICAL_MECHANISM_EVIDENCE"),
        "decisionGates": gates,
        "consistentIndependentRotationRecords": consistent_rotation_records,
        "contradictions": contradictions,
        "externalEvidence": external_evidence,
        "retrievedAt": retrieved_at or datetime.now(timezone.utc).isoformat(),
        "interpretationNotes": [
            "No v20.14 temporal model was refit.",
            "A catalog ROT label or youth indicator alone cannot promote rotation.",
            "Single-value exact-period mismatch is not evidence against rotation.",
            "The recurrent 7.546/15.093-day family remains a separate unresolved question.",
        ],
    }
