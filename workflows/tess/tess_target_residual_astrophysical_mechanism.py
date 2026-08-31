"""Conservative mechanism follow-up for a target-associated residual signal.

This server-local stage consumes only the frozen v20.10.1 adjudication and the
already persisted catalog identity.  It performs no catalog query and no flux
fit.  A period-consistent catalog label can support a hypothesis, but cannot by
itself resolve the physical mechanism or upgrade the investigation claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .tess_hypotheses import rotational_sanity


HANDLER_ID = "openstar.tess.target-residual-astrophysical-mechanism.analyze"
RESULT_VERSION = "openstar.tess-target-residual-astrophysical-mechanism.v1"
METHOD_CONTRACT_ID = (
    "openstar.tess.target-residual-astrophysical-mechanism."
    "frozen-period-classification-adjudication.v1"
)

ROTATION_SUPPORTED = "TARGET_RESIDUAL_ROTATION_HYPOTHESIS_SUPPORTED"
PULSATION_SUPPORTED = "TARGET_RESIDUAL_PULSATION_HYPOTHESIS_SUPPORTED"
ESTABLISHED_FAMILY = "EXTERNAL_VARIABILITY_TRACES_ESTABLISHED_FAMILY"
INCONCLUSIVE = "TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_INCONCLUSIVE"


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def build_method_contract(*, external_evidence: dict[str, Any]) -> dict[str, Any]:
    """Freeze deterministic decision rules before catalog labels are inspected."""
    spatial = external_evidence.get("spatialEvidence") or {}
    return {
        "methodContractID": METHOD_CONTRACT_ID,
        "resultVersion": RESULT_VERSION,
        "evidenceBoundary": {
            "upstreamVersion": external_evidence.get("version"),
            "upstreamMethodContractID": external_evidence.get("methodContractID"),
            "upstreamMethodContractHash": external_evidence.get("methodContractHash"),
            "residualPeriodAtReferenceDays": external_evidence.get(
                "residualPeriodAtReferenceDays"
            ),
            "establishedPhysicalPeriodDays": external_evidence.get(
                "establishedPhysicalPeriodDays"
            ),
            "targetSupportingSectors": list(
                spatial.get("targetSupportingSectors") or []
            ),
            "offTargetSectors": list(spatial.get("offTargetSectors") or []),
        },
        "inputs": {
            "catalogAcquisition": "REUSE_V20_10_1_FROZEN_RECORDS_ONLY",
            "fluxValuesRead": False,
            "networkQueries": False,
            "allowedTargetAssociatedFamilies": [
                "ROTATION_LIKE",
                "PULSATION_LIKE",
                "OTHER_VARIABLE",
            ],
        },
        "decisionRules": {
            "residualHypothesisRequiresPersistedResidualPeriodMatch": True,
            "residualHypothesisRejectsEstablishedFamilyMatch": True,
            "rotationAndPulsationConflictIsInconclusive": True,
            "otherVariableLabelCannotIdentifyMechanism": True,
            "rotationSanityIsComputedFromFrozenTICMassAndRadius": True,
            "rotationRuledOutOrStronglyDisfavoredIsInconclusive": True,
            "catalogClassificationCannotResolvePhysicalMechanism": True,
            "catalogClassificationCannotUpgradeClaim": True,
            "offTargetSectorEvidenceIsRetainedAsCaution": True,
        },
    }


def method_contract_hash(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_mechanism_followup_boundary(
    *, external_evidence: dict[str, Any], identity: dict[str, Any], expected_tic_id: int
) -> None:
    """Fail closed unless the exact unresolved v20.10.1 boundary is intact."""
    spatial = external_evidence.get("spatialEvidence") or {}
    records = external_evidence.get("catalogEvidence") or []
    nonbinary = external_evidence.get(
        "targetAssociatedNonbinaryVariabilityEvidence"
    ) or []
    residual_period = _positive(external_evidence.get("residualPeriodAtReferenceDays"))
    physical_period = _positive(external_evidence.get("establishedPhysicalPeriodDays"))
    upstream_contract = external_evidence.get("methodContract")
    derived_nonbinary = [
        record for record in records
        if isinstance(record, dict)
        and record.get("targetAssociated") is True
        and record.get("classificationFamily") in {
            "ROTATION_LIKE", "PULSATION_LIKE", "OTHER_VARIABLE"
        }
    ]
    associated_binary = [
        record for record in records
        if isinstance(record, dict)
        and record.get("targetAssociated") is True
        and record.get("classificationFamily") == "BINARY_LIKE"
    ]
    try:
        tic_id = int(external_evidence.get("ticID"))
        identity_tic = int(identity.get("ticID"))
        expected_tic = int(expected_tic_id)
        target_sectors = sorted(
            int(value) for value in spatial.get("targetSupportingSectors") or []
        )
        off_target_sectors = sorted(
            int(value) for value in spatial.get("offTargetSectors") or []
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            "Target-residual astrophysical mechanism lineage is incomplete."
        ) from None

    valid_records = all(
        isinstance(record, dict)
        and record.get("targetAssociated") is True
        and record.get("classificationFamily") in {
            "ROTATION_LIKE", "PULSATION_LIKE", "OTHER_VARIABLE"
        }
        and record in records
        and record.get("source")
        and record.get("stableObjectID") is not None
        and record.get("classification")
        and isinstance(record.get("periodComparisons"), dict)
        for record in nonbinary
    )
    exact = (
        external_evidence.get("version")
        == "openstar.tess-residual-external-evidence.v1"
        and external_evidence.get("methodContractID")
        == "openstar.tess.residual-external-evidence.frozen-catalog-adjudication.v1"
        and isinstance(external_evidence.get("methodContractHash"), str)
        and len(external_evidence["methodContractHash"]) == 64
        and isinstance(upstream_contract, dict)
        and method_contract_hash(upstream_contract)
        == external_evidence.get("methodContractHash")
        and external_evidence.get("classification")
        == "TARGET_ASSOCIATED_NONBINARY_VARIABILITY_EVIDENCE_PRESENT"
        and external_evidence.get("recommendedNextTest")
        == "TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_FOLLOWUP"
        and external_evidence.get("physicalMechanismResolved") is False
        and external_evidence.get("claimLevelChanged") is False
        and spatial.get("classification") == "RESIDUAL_MODE_TARGET_SUPPORTED"
        and len(target_sectors) >= 3
        and not set(target_sectors) & set(off_target_sectors)
        and residual_period is not None
        and physical_period is not None
        and not math.isclose(
            residual_period, physical_period, rel_tol=1e-9, abs_tol=1e-12
        )
        and bool(nonbinary)
        and nonbinary == derived_nonbinary
        and not associated_binary
        and valid_records
        and tic_id == identity_tic == expected_tic
        and identity.get("identityResolved") is True
        and (identity.get("tic") or {}).get("found") is True
    )
    if not exact:
        raise RuntimeError(
            "Target-residual astrophysical mechanism follow-up requires the exact "
            "unresolved target-associated nonbinary v20.10.1 boundary."
        )


def _persisted_match(record: dict[str, Any], key: str) -> bool:
    comparison = (record.get("periodComparisons") or {}).get(key)
    return isinstance(comparison, dict) and comparison.get("matches") is True


def analyze_target_residual_astrophysical_mechanism(
    *, external_evidence: dict[str, Any], identity: dict[str, Any], expected_tic_id: int
) -> dict[str, Any]:
    validate_mechanism_followup_boundary(
        external_evidence=external_evidence,
        identity=identity,
        expected_tic_id=expected_tic_id,
    )
    contract = build_method_contract(external_evidence=external_evidence)
    contract_hash = method_contract_hash(contract)

    residual_period = float(external_evidence["residualPeriodAtReferenceDays"])
    rotation_constraint = rotational_sanity(identity, residual_period)
    records = external_evidence[
        "targetAssociatedNonbinaryVariabilityEvidence"
    ]
    adjudicated = []
    residual_rotation = []
    residual_pulsation = []
    established_matches = []
    insufficiencies = list(external_evidence.get("insufficiencyReasons") or [])

    for source in records:
        record = dict(source)
        residual_match = _persisted_match(record, "residualPeriod")
        established_match = _persisted_match(record, "establishedPhysicalPeriod")
        record["supportsResidualPeriod"] = residual_match
        record["supportsEstablishedFamily"] = established_match
        if residual_match and established_match:
            record["adjudication"] = "MATCHES_BOTH_PERIOD_FAMILIES"
        elif residual_match:
            record["adjudication"] = "RESIDUAL_PERIOD_SPECIFIC"
        elif established_match:
            record["adjudication"] = "ESTABLISHED_FAMILY_SPECIFIC"
        else:
            record["adjudication"] = "NO_PERSISTED_PERIOD_MATCH"
        adjudicated.append(record)

        if established_match:
            established_matches.append(record)
        if residual_match and not established_match:
            if record.get("classificationFamily") == "ROTATION_LIKE":
                residual_rotation.append(record)
            elif record.get("classificationFamily") == "PULSATION_LIKE":
                residual_pulsation.append(record)

    rotation_disfavored = rotation_constraint.get("status") in {
        "ruled-out", "strongly-disfavored"
    }
    ambiguous_period_match = any(
        item.get("adjudication") == "MATCHES_BOTH_PERIOD_FAMILIES"
        for item in adjudicated
    )
    if (
        residual_rotation
        and not residual_pulsation
        and not rotation_disfavored
        and not ambiguous_period_match
    ):
        classification = ROTATION_SUPPORTED
        next_test = "SPECTROSCOPIC_ROTATION_CONSTRAINT"
    elif (
        residual_pulsation
        and not residual_rotation
        and not ambiguous_period_match
    ):
        classification = PULSATION_SUPPORTED
        next_test = "ASTEROSPECTROSCOPIC_MODE_CLASSIFICATION"
    elif established_matches and not residual_rotation and not residual_pulsation:
        classification = ESTABLISHED_FAMILY
        next_test = "TARGET_RESIDUAL_TIME_RESOLVED_SPECTROSCOPY"
    else:
        classification = INCONCLUSIVE
        next_test = "HUMAN_SCIENTIFIC_REVIEW"
        if rotation_disfavored and residual_rotation:
            insufficiencies.append("ROTATION_PHYSICALLY_DISFAVORED_FOR_RESIDUAL_PERIOD")
        if residual_rotation and residual_pulsation:
            insufficiencies.append("CONFLICTING_RESIDUAL_MECHANISM_FAMILIES")
        if ambiguous_period_match:
            insufficiencies.append("CATALOG_PERIOD_MATCHES_BOTH_PERIOD_FAMILIES")
        if not residual_rotation and not residual_pulsation and not established_matches:
            insufficiencies.append("NO_MECHANISM_SPECIFIC_PERSISTED_PERIOD_MATCH")

    spatial = external_evidence.get("spatialEvidence") or {}
    return {
        "version": RESULT_VERSION,
        "methodContractID": METHOD_CONTRACT_ID,
        "methodContractHash": contract_hash,
        "methodContract": contract,
        "ticID": int(expected_tic_id),
        "classification": classification,
        "residualPeriodAtReferenceDays": residual_period,
        "establishedPhysicalPeriodDays": float(
            external_evidence["establishedPhysicalPeriodDays"]
        ),
        "adjudicatedCatalogEvidence": adjudicated,
        "residualRotationEvidence": residual_rotation,
        "residualPulsationEvidence": residual_pulsation,
        "establishedFamilyEvidence": established_matches,
        "rotationConstraintAtResidualPeriod": rotation_constraint,
        "spatialEvidence": {
            "classification": spatial.get("classification"),
            "targetSupportingSectors": list(
                spatial.get("targetSupportingSectors") or []
            ),
            "offTargetSectors": list(spatial.get("offTargetSectors") or []),
            "ambiguousSectors": list(spatial.get("ambiguousSectors") or []),
            "cautions": list(spatial.get("cautions") or []),
        },
        "insufficiencyReasons": sorted(set(insufficiencies)),
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "This stage supports or rejects catalog-informed hypotheses only. "
            "It does not prove that a catalog label produces the TESS residual, "
            "does not erase discordant spatial evidence, does not resolve the "
            "physical mechanism, and does not upgrade the claim."
        ),
    }
