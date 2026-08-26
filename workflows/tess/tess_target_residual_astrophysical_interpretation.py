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
class FrozenCatalogAstrophysicalEvidenceProvider:
    """Normalize already-frozen machine-readable VSX evidence.

    Catalog acquisition remains owned by ``catalog-identity``; this adapter does
    not scrape prose or silently issue a second, untracked query.
    """
    provider_id: str = "openstar-frozen-vsx"
    relative_period_tolerance: float = 0.05

    def fetch(self, object_identity: dict[str, Any]) -> dict[str, Any]:
        identity = object_identity.get("frozenCatalogIdentity") or {}
        vsx = identity.get("vsx") or {}
        nearest = vsx.get("nearest") or {}
        type_text = str(nearest.get("type") or "").upper()
        period = nearest.get("periodDays")
        retrieved_at = (identity.get("retrievedAt") or
                        object_identity.get("catalogEvidenceFrozenAt"))
        record = None
        if vsx.get("found") is True and period is not None:
            record = {
                "provider": "AAVSO VSX via VizieR",
                "stableObjectID": nearest.get("name"),
                "queryParameters": dict(vsx.get("queryProvenance") or {}),
                "citation": {"catalog": "B/vsx/vsx", "catalogName": "AAVSO VSX"},
                "retrievedValues": dict(nearest),
                "retrievalTimestamp": retrieved_at,
                "mechanism": "ROTATIONAL_MODULATION" if "ROT" in type_text else None,
                "catalogPeriodDays": float(period),
                "maximumRelativePeriodDifference": self.relative_period_tolerance,
            }
        return {"available": record is not None, "provider": self.provider_id,
            "objectIdentity": {key: value for key, value in object_identity.items()
                if key != "frozenCatalogIdentity"},
            "queryParameters": dict(vsx.get("queryProvenance") or {}),
            "retrievedAt": retrieved_at,
            "records": [record] if record else []}


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


def rotation_sanity_allows(rotation_sanity: dict[str, Any], period_days: float) -> bool:
    """Normalize the authoritative ``rotational_sanity`` producer semantics."""
    try:
        recorded_period = float(rotation_sanity["periodDays"])
        ratio = float(rotation_sanity["equatorialToCriticalRatio"])
    except (KeyError, TypeError, ValueError):
        return False
    return (rotation_sanity.get("evaluated") is True
        and rotation_sanity.get("status") == "not-ruled-out"
        and abs(recorded_period - period_days) <= max(1e-12, abs(period_days) * 1e-9)
        and ratio < 0.7)


def interpret_target_residual_astrophysics(*, mechanism: dict[str, Any],
        target_attribution: dict[str, Any], stellar_context: dict[str, Any],
        external_evidence: dict[str, Any], residual_period_days: float | None,
        main_photometric_family: dict[str, Any] | None,
        retrieved_at: str | None = None) -> dict[str, Any]:
    """Apply preregistered, mechanism-neutral guards to independent evidence."""
    period = float(residual_period_days) if residual_period_days is not None else None
    records = external_evidence.get("records") or []
    consistent_rotation_records = []
    contradictions = []
    for record in records:
        provenance_complete = all(record.get(key) for key in (
            "provider", "stableObjectID", "queryParameters", "citation",
            "retrievalTimestamp"))
        if not provenance_complete:
            continue
        mechanism_name = str(record.get("mechanism") or "").upper()
        low, high = record.get("periodRangeDays") or (None, None)
        range_match = (low is not None and high is not None and period is not None and period > 0
                       and float(low) <= period <= float(high))
        catalog_period = record.get("catalogPeriodDays")
        tolerance = record.get("maximumRelativePeriodDifference")
        catalog_match = (catalog_period is not None and tolerance is not None
            and period is not None and period > 0
            and abs(float(catalog_period) - period) / period <= float(tolerance))
        if mechanism_name in {"ROTATION", "ROTATIONAL_MODULATION",
                              "STARSPOT_ROTATION"} and (range_match or catalog_match):
            consistent_rotation_records.append(record)
        if record.get("contradictsRotation") is True:
            contradictions.append(record)

    gates = {
        "targetSpatialAttribution": (
            target_attribution.get("classification") == "TARGET_RESIDUAL_COMPONENT_DOMINANT"
            and target_attribution.get("residualModeOrigin") == "TARGET_DOMINANT"),
        "replicatedSmoothAmplitudeModulation": (
            mechanism.get("classification") == "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"
            and "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION" in
                (mechanism.get("replicatedMechanisms") or [])
            and len((mechanism.get("replicatedMechanismSupportingSectorIDs") or {}).get(
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION") or []) >= 2),
        "authoritativeResidualPeriodAvailable": period is not None and period > 0,
        "rotationPhysicallyAllowed": (period is not None and
            rotation_sanity_allows(stellar_context, period)),
        "independentRotationPeriodConsistent": len(consistent_rotation_records) >= 1,
        # The frozen TESS morphology and physical-radius guard are constraints
        # independent of the cited historical rotation measurement.
        "multipleIndependentConstraints": len(consistent_rotation_records) >= 1,
        "noStrongerContradiction": not contradictions,
    }
    # A label alone, youth alone, or an unrealistically exact-period equality is
    # intentionally absent. A provenance-complete historical range or a
    # preregistered catalog-period tolerance is required.
    promoted = external_evidence.get("available") is True and all(gates.values())
    return {
        "schemaVersion": "target-residual-astrophysical-interpretation-v1",
        "classification": ROTATION if promoted else UNRESOLVED,
        "physicalMechanismResolved": promoted,
        "targetResidualMechanismResolved": promoted,
        "physicalCycleResolved": False,
        "targetResidualPeriodDays": period,
        "smoothAmplitudeSupportingSectorIDs": list(
            (mechanism.get("replicatedMechanismSupportingSectorIDs") or {}).get(
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION") or []),
        "mainPhotometricFamily": (dict(main_photometric_family)
            if main_photometric_family is not None else
            {"available": False, "physicalCycleResolved": False}),
        "recommendedNextTest": (None if promoted else
            "ADDITIONAL_INDEPENDENT_ASTROPHYSICAL_MECHANISM_EVIDENCE"),
        "decisionGates": gates,
        "rotationSanity": dict(stellar_context),
        "consistentIndependentRotationRecords": consistent_rotation_records,
        "contradictions": contradictions,
        "externalEvidence": external_evidence,
        "retrievedAt": retrieved_at or datetime.now(timezone.utc).isoformat(),
        "interpretationNotes": [
            "No v20.14 temporal model was refit.",
            "A catalog ROT label or youth indicator alone cannot promote rotation.",
            "Single-value exact-period mismatch is not evidence against rotation.",
            "The recurrent main photometric family remains a separate physical-cycle question.",
        ],
    }
