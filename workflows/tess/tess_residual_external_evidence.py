"""Frozen external classification evidence for a target-supported TESS residual.

This append-only stage does not query catalogs and does not refit photometry.  It
adjudicates the catalog responses frozen by the original identity stage against
the exact v20.9.2/v20.10 residual lineage.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .tess_nonstationary import (
    CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
    validate_confirmed_nonstationary_localization_boundary,
)


HANDLER_ID = "openstar.tess.residual-external-evidence.analyze"
RESULT_VERSION = "openstar.tess-residual-external-evidence.v1"
METHOD_CONTRACT_ID = (
    "openstar.tess.residual-external-evidence.frozen-catalog-adjudication.v1"
)
TARGET_ASSOCIATION_MAX_ARCSEC = 1.0
PERIOD_ABSOLUTE_TOLERANCE_DAYS = 0.02
PERIOD_RELATIVE_TOLERANCE = 0.03
MIN_TARGET_SUPPORTING_INDEPENDENT_SECTORS = 3

BINARY_TOKENS = frozenset({
    "BINARY", "ECLIPSING", "ECLIPSE", "ELLIPSOIDAL", "EA", "EB", "EW",
    "ELL", "ECL",
})
ROTATION_TOKENS = frozenset({"ROT", "ROTATION", "ROTATIONAL", "SPOT", "BY", "RS"})
PULSATION_TOKENS = frozenset({
    "PULSATING", "PULSATION", "DSCT", "GDOR", "SPB", "CEPHEID", "DCEP",
    "RR", "RRAB", "RRC", "LPV",
})


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0.0 else None


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.split(r"[^A-Z0-9]+", str(value or "").upper())
        if token
    }


def _classification_family(value: Any) -> str | None:
    text = str(value or "").upper()
    tokens = _tokens(text)
    if tokens & BINARY_TOKENS or any(
        marker in text for marker in ("BINARY", "ECLIPS", "ELLIPSOID")
    ):
        return "BINARY_LIKE"
    if tokens & ROTATION_TOKENS or "ROTAT" in text:
        return "ROTATION_LIKE"
    if tokens & PULSATION_TOKENS or "PULSAT" in text:
        return "PULSATION_LIKE"
    return "OTHER_VARIABLE" if tokens else None


def _period_comparison(observed_days: float, catalog_days: float) -> dict[str, Any]:
    candidates = (
        ("0.5x", observed_days * 0.5),
        ("1x", observed_days),
        ("2x", observed_days * 2.0),
    )
    relation, equivalent = min(
        candidates, key=lambda item: abs(item[1] - catalog_days)
    )
    absolute = abs(equivalent - catalog_days)
    relative = absolute / max(abs(catalog_days), 1e-12)
    return {
        "relation": relation,
        "observedEquivalentDays": equivalent,
        "catalogPeriodDays": catalog_days,
        "absoluteErrorDays": absolute,
        "relativeError": relative,
        "matches": absolute <= max(
            PERIOD_ABSOLUTE_TOLERANCE_DAYS,
            PERIOD_RELATIVE_TOLERANCE * catalog_days,
        ),
    }


def build_method_contract(
    *, localization: dict[str, Any], nonstationary: dict[str, Any]
) -> dict[str, Any]:
    """Freeze all interpretation rules before inspecting catalog classifications."""
    return {
        "methodContractID": METHOD_CONTRACT_ID,
        "resultVersion": RESULT_VERSION,
        "evidenceBoundary": {
            "localizationVersion": localization.get("version"),
            "localizationClassification": (
                (localization.get("crossSector") or {}).get("classification")
            ),
            "residualPeriodAtReferenceDays": localization.get(
                "residualPeriodAtReferenceDays"
            ),
            "physicalPeriodDays": localization.get("physicalPeriodDays"),
            "targetSupportingSectors": list(
                (localization.get("crossSector") or {}).get(
                    "targetSupportingSectors"
                ) or []
            ),
            "offTargetSectors": list(
                (localization.get("crossSector") or {}).get("offTargetSectors")
                or []
            ),
            "nonstationaryEvidenceLineage": nonstationary.get("evidenceLineage"),
        },
        "catalogInputs": {
            "sources": ["TIC", "SIMBAD", "AAVSO_VSX", "GAIA_DR3_VARIABILITY"],
            "acquisition": "REUSE_COMPLETED_CATALOG_IDENTITY_RESULT_ONLY",
            "networkQueries": False,
            "targetAssociationMaxArcsec": TARGET_ASSOCIATION_MAX_ARCSEC,
            "gaiaAssociation": "TIC_GAIA_ALIAS_EQUALS_NEAREST_GAIA_SOURCE_ID",
        },
        "periodComparison": {
            "relations": ["0.5x", "1x", "2x"],
            "absoluteToleranceDays": PERIOD_ABSOLUTE_TOLERANCE_DAYS,
            "relativeTolerance": PERIOD_RELATIVE_TOLERANCE,
            "comparisonTargets": ["RESIDUAL_PERIOD", "ESTABLISHED_PHYSICAL_PERIOD"],
        },
        "classificationRules": {
            "families": [
                "BINARY_LIKE", "ROTATION_LIKE", "PULSATION_LIKE", "OTHER_VARIABLE"
            ],
            "positiveCatalogInterpretationRequiresTargetAssociation": True,
            "catalogClassificationDoesNotResolvePhysicalMechanism": True,
            "minimumTargetSupportingIndependentSectors": (
                MIN_TARGET_SUPPORTING_INDEPENDENT_SECTORS
            ),
            "offTargetSectorEvidenceIsRetainedAsCaution": True,
        },
    }


def method_contract_hash(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_target_supported_boundary(
    *,
    localization: dict[str, Any],
    nonstationary: dict[str, Any],
    confirmation: dict[str, Any],
    physical_cycle: dict[str, Any] | None,
    identity: dict[str, Any],
    expected_tic_id: int,
) -> None:
    """Fail closed unless the exact confirmed target-supported lineage is intact."""
    cycle_period = validate_confirmed_nonstationary_localization_boundary(
        nonstationary, confirmation, physical_cycle
    )
    cross = localization.get("crossSector") or {}
    target = list(cross.get("targetSupportingSectors") or [])
    off_target = list(cross.get("offTargetSectors") or [])
    ambiguous = list(cross.get("ambiguousSectors") or [])
    sector_rows = [
        item for item in localization.get("sectorResults") or []
        if item.get("role") == "independent"
    ]
    row_target = sorted(
        int(item["sector"]) for item in sector_rows
        if item.get("classification") == "TARGET_CONSISTENT"
    )
    row_off_target = sorted(
        int(item["sector"]) for item in sector_rows
        if item.get("classification") == "OFF_TARGET"
    )
    row_ambiguous = sorted(
        int(item["sector"]) for item in sector_rows
        if item.get("classification") == "AMBIGUOUS"
    )
    residual_period = _positive(localization.get("residualPeriodAtReferenceDays"))
    residual_frequency = _positive(localization.get("residualFrequencyAtReference"))
    expected_period = _positive(nonstationary.get("preferredPeriodAtReferenceDays"))
    expected_frequency = _positive(nonstationary.get("preferredFrequencyAtReference"))
    expected_drift = _finite(nonstationary.get("fractionalFrequencyDriftPerDay"))
    actual_drift = _finite(localization.get("fractionalFrequencyDriftPerDay"))
    localized_physical_period = _positive(localization.get("physicalPeriodDays"))
    expected_signal_sectors = sorted(
        int(value) for value in
        ((nonstationary.get("preferredModel") or {}).get("signalSectors") or [])
    )
    localized_signal_sectors = sorted(
        int(value) for value in (localization.get("signalSectors") or [])
    )
    try:
        tic_id = int(localization.get("ticID"))
        identity_tic = int(identity.get("ticID"))
        expected_tic = int(expected_tic_id)
        eligible = int(cross.get("independentEligibleSectorCount"))
        required = int(cross.get("requiredIndependentSupportCount"))
    except (TypeError, ValueError):
        raise RuntimeError("Residual external-evidence target lineage is incomplete.") from None

    exact = (
        localization.get("version")
        == "openstar.tess-residual-mode-pixel-localization.v1"
        and localization.get("recommendedNextTest")
        == "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
        and localization.get("physicalMechanismResolved") is False
        and localization.get("claimLevelChanged") is False
        and cross.get("classification") == "RESIDUAL_MODE_TARGET_SUPPORTED"
        and cross.get("residualModeOrigin") == "TARGET_CONSISTENT"
        and cross.get("recommendedNextTest")
        == "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
        and eligible == len(sector_rows)
        and eligible >= MIN_TARGET_SUPPORTING_INDEPENDENT_SECTORS
        and required >= MIN_TARGET_SUPPORTING_INDEPENDENT_SECTORS
        and len(target) >= required
        and sorted(int(value) for value in target) == row_target
        and sorted(int(value) for value in off_target) == row_off_target
        and sorted(int(value) for value in ambiguous) == row_ambiguous
        and not set(row_target) & set(row_off_target)
        and not set(row_target) & set(row_ambiguous)
        and not set(row_off_target) & set(row_ambiguous)
        and nonstationary.get("evidenceLineage")
        == CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE
        and residual_period is not None
        and residual_frequency is not None
        and expected_period is not None
        and expected_frequency is not None
        and expected_drift is not None
        and actual_drift is not None
        and localized_physical_period is not None
        and math.isclose(
            localized_physical_period, cycle_period,
            rel_tol=1e-9, abs_tol=1e-12,
        )
        and localized_signal_sectors == expected_signal_sectors
        and math.isclose(residual_period, expected_period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            residual_frequency, expected_frequency, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(actual_drift, expected_drift, rel_tol=1e-9, abs_tol=1e-12)
        and tic_id == identity_tic == expected_tic
        and identity.get("identityResolved") is True
        and (identity.get("tic") or {}).get("found") is True
    )
    if not exact:
        raise RuntimeError(
            "Residual external evidence requires the exact confirmed "
            "RESIDUAL_MODE_TARGET_SUPPORTED boundary."
        )


def _catalog_records(identity: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    insufficiencies: list[str] = []

    simbad = identity.get("simbad") or {}
    simbad_family = _classification_family(simbad.get("objectType"))
    if simbad.get("found") is True and simbad_family == "BINARY_LIKE":
        records.append({
            "source": "SIMBAD",
            "stableObjectID": simbad.get("mainID"),
            "classification": simbad.get("objectType"),
            "classificationFamily": simbad_family,
            "targetAssociated": True,
            "targetAssociation": {"method": "DIRECT_TIC_OBJECT_QUERY"},
            "catalogPeriodDays": None,
            "queryProvenance": dict(simbad.get("queryProvenance") or {}),
        })
    elif simbad.get("queryError"):
        insufficiencies.append("SIMBAD_QUERY_WAS_NOT_AVAILABLE_IN_FROZEN_IDENTITY")

    vsx = identity.get("vsx") or {}
    for match in vsx.get("matches") or []:
        family = _classification_family(match.get("type"))
        separation = _finite(match.get("separationArcsec"))
        associated = separation is not None and separation <= TARGET_ASSOCIATION_MAX_ARCSEC
        if family is None:
            continue
        records.append({
            "source": "AAVSO_VSX",
            "stableObjectID": match.get("name"),
            "classification": match.get("type"),
            "classificationFamily": family,
            "targetAssociated": associated,
            "targetAssociation": {
                "method": "ANGULAR_SEPARATION",
                "separationArcsec": separation,
                "maximumArcsec": TARGET_ASSOCIATION_MAX_ARCSEC,
            },
            "catalogPeriodDays": _positive(match.get("periodDays")),
            "queryProvenance": dict(vsx.get("queryProvenance") or {}),
        })
    if vsx.get("queryError"):
        insufficiencies.append("VSX_QUERY_WAS_NOT_AVAILABLE_IN_FROZEN_IDENTITY")

    gaia = identity.get("gaiaDR3") or {}
    gaia_variability = identity.get("gaiaVariability") or {}
    nearest = gaia.get("nearest") or {}
    tic_gaia = ((identity.get("tic") or {}).get("aliases") or {}).get("GAIA_field")
    source_id = nearest.get("sourceID")
    separation = _finite(nearest.get("separationArcsec"))
    associated = (
        tic_gaia is not None
        and source_id is not None
        and str(tic_gaia) == str(source_id)
        and separation is not None
        and separation <= TARGET_ASSOCIATION_MAX_ARCSEC
    )
    classification = gaia_variability.get("classification") or {}
    family = _classification_family(classification.get("class"))
    periods = [
        _positive(item.get("periodDays"))
        for item in gaia_variability.get("periodCandidates") or []
    ]
    periods = [value for value in periods if value is not None]
    if family is not None:
        if periods:
            for period in periods:
                records.append({
                    "source": "GAIA_DR3_VARIABILITY",
                    "stableObjectID": source_id,
                    "classification": classification.get("class"),
                    "classificationFamily": family,
                    "targetAssociated": associated,
                    "targetAssociation": {
                        "method": "TIC_GAIA_ALIAS_AND_ANGULAR_SEPARATION",
                        "ticGaiaSourceID": tic_gaia,
                        "gaiaSourceID": source_id,
                        "separationArcsec": separation,
                        "maximumArcsec": TARGET_ASSOCIATION_MAX_ARCSEC,
                    },
                    "catalogPeriodDays": period,
                    "queryProvenance": dict(
                        gaia_variability.get("queryProvenance") or {}
                    ),
                })
        else:
            records.append({
                "source": "GAIA_DR3_VARIABILITY",
                "stableObjectID": source_id,
                "classification": classification.get("class"),
                "classificationFamily": family,
                "targetAssociated": associated,
                "targetAssociation": {
                    "method": "TIC_GAIA_ALIAS_AND_ANGULAR_SEPARATION",
                    "ticGaiaSourceID": tic_gaia,
                    "gaiaSourceID": source_id,
                    "separationArcsec": separation,
                    "maximumArcsec": TARGET_ASSOCIATION_MAX_ARCSEC,
                },
                "catalogPeriodDays": None,
                "queryProvenance": dict(
                    gaia_variability.get("queryProvenance") or {}
                ),
            })
    if gaia.get("queryError"):
        insufficiencies.append("GAIA_DR3_QUERY_WAS_NOT_AVAILABLE_IN_FROZEN_IDENTITY")
    if gaia_variability.get("queryError"):
        insufficiencies.append(
            "GAIA_VARIABILITY_QUERY_WAS_NOT_AVAILABLE_IN_FROZEN_IDENTITY"
        )
    if identity.get("catalogCoverageComplete") is not True:
        insufficiencies.append("FROZEN_CATALOG_COVERAGE_INCOMPLETE")
    return records, sorted(set(insufficiencies))


def analyze_residual_external_evidence(
    *,
    localization: dict[str, Any],
    nonstationary: dict[str, Any],
    confirmation: dict[str, Any],
    physical_cycle: dict[str, Any] | None,
    identity: dict[str, Any],
    expected_tic_id: int,
) -> dict[str, Any]:
    validate_target_supported_boundary(
        localization=localization,
        nonstationary=nonstationary,
        confirmation=confirmation,
        physical_cycle=physical_cycle,
        identity=identity,
        expected_tic_id=expected_tic_id,
    )
    contract = build_method_contract(
        localization=localization, nonstationary=nonstationary
    )
    contract_hash = method_contract_hash(contract)
    residual_period = float(localization["residualPeriodAtReferenceDays"])
    physical_period = float(localization["physicalPeriodDays"])
    records, insufficiencies = _catalog_records(identity)
    compared = []
    for record in records:
        item = dict(record)
        catalog_period = _positive(item.get("catalogPeriodDays"))
        item["periodComparisons"] = {
            "residualPeriod": (
                _period_comparison(residual_period, catalog_period)
                if catalog_period is not None else None
            ),
            "establishedPhysicalPeriod": (
                _period_comparison(physical_period, catalog_period)
                if catalog_period is not None else None
            ),
        }
        compared.append(item)

    associated = [item for item in compared if item.get("targetAssociated") is True]
    binary = [
        item for item in associated
        if item.get("classificationFamily") == "BINARY_LIKE"
    ]
    nonbinary = [
        item for item in associated
        if item.get("classificationFamily") in {
            "ROTATION_LIKE", "PULSATION_LIKE", "OTHER_VARIABLE"
        }
    ]
    if binary and nonbinary:
        classification = "CONFLICTING_TARGET_EXTERNAL_CLASSIFICATIONS"
        next_test = "HUMAN_SCIENTIFIC_REVIEW"
    elif binary:
        classification = "TARGET_ASSOCIATED_BINARY_EVIDENCE_PRESENT"
        next_test = "SPECTROSCOPIC_BINARY_CONFIRMATION"
    elif nonbinary:
        classification = (
            "TARGET_ASSOCIATED_NONBINARY_VARIABILITY_EVIDENCE_PRESENT"
        )
        next_test = "TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_FOLLOWUP"
    else:
        classification = (
            "EXTERNAL_VARIABILITY_AND_BINARY_EVIDENCE_INCONCLUSIVE"
        )
        next_test = "SPECTROSCOPIC_VARIABILITY_AND_BINARY_FOLLOWUP"
        if not compared:
            insufficiencies.append("NO_FROZEN_EXTERNAL_VARIABILITY_CLASSIFICATION")
        elif not associated:
            insufficiencies.append("NO_EXTERNAL_CLASSIFICATION_SECURELY_ASSOCIATED_WITH_TARGET")

    cross = localization.get("crossSector") or {}
    off_target = list(cross.get("offTargetSectors") or [])
    spatial_cautions = []
    if off_target:
        spatial_cautions.append({
            "reason": "RESIDUAL_LOCALIZATION_HAS_OFF_TARGET_INDEPENDENT_SECTOR",
            "sectors": off_target,
            "effect": (
                "Retained as discordant spatial evidence; catalog classification "
                "cannot erase it or resolve the physical mechanism."
            ),
        })

    return {
        "version": RESULT_VERSION,
        "methodContractID": METHOD_CONTRACT_ID,
        "methodContractHash": contract_hash,
        "methodContract": contract,
        "ticID": int(expected_tic_id),
        "classification": classification,
        "residualPeriodAtReferenceDays": residual_period,
        "establishedPhysicalPeriodDays": physical_period,
        "catalogCoverageComplete": identity.get("catalogCoverageComplete") is True,
        "catalogEvidence": compared,
        "targetAssociatedBinaryEvidence": binary,
        "targetAssociatedNonbinaryVariabilityEvidence": nonbinary,
        "spatialEvidence": {
            "classification": cross.get("classification"),
            "targetSupportingSectors": list(cross.get("targetSupportingSectors") or []),
            "offTargetSectors": off_target,
            "ambiguousSectors": list(cross.get("ambiguousSectors") or []),
            "cautions": spatial_cautions,
        },
        "insufficiencyReasons": sorted(set(insufficiencies)),
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "Frozen catalog classifications are external context, not proof that "
            "the cataloged mechanism produces either TESS period. The target-supported "
            "residual localization, its off-target sector, the claim level, and the "
            "unresolved physical mechanism are preserved."
        ),
    }
