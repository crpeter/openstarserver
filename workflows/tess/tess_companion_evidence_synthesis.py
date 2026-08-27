"""Pure final synthesis of the persisted companion-evidence chain."""
from __future__ import annotations

import math
from typing import Any

from openstar_investigation import sha256_json
from .tess_external_companion_evidence import (
    FREEZE_VERSION, LOCALIZATION_VERSION, RESULT_VERSION as EXTERNAL_RESULT_VERSION,
    REVIEW_VERSION, canonical_gaia_dr3_id, canonical_tic_id,
)

RESULT_VERSION = "openstar.final-companion-evidence-synthesis.v1"
HANDLER_ID = "openstar.tess.final-companion-evidence-synthesis"
BINARY_VERSION = "2.0"
MIN_INDEPENDENT_SUPPORT = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"invalid {name}")
    number = float(value)
    _require(math.isfinite(number), f"invalid {name}")
    return number


def synthesize_companion_evidence(binary_confirmation: dict[str, Any],
                                  localization: dict[str, Any],
                                  source_review: dict[str, Any],
                                  frozen_external_response: dict[str, Any],
                                  external_result: dict[str, Any]) -> dict[str, Any]:
    """Validate and synthesize immutable evidence; never repairs an invalid chain."""
    artifacts = (binary_confirmation, localization, source_review,
                 frozen_external_response, external_result)
    _require(all(isinstance(item, dict) for item in artifacts), "evidence artifact is malformed")
    _require(all(item.get("catalogAnswerKeyUsed") is False for item in artifacts),
             "catalog answer-key flag must be exactly false")
    _require(binary_confirmation.get("resultVersion") == BINARY_VERSION, "invalid binary version")
    independent = binary_confirmation.get("independentEvidence") or {}
    _require(independent.get("classification") == "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
             "binary confirmation is unresolved")
    _require(localization.get("resultVersion") == LOCALIZATION_VERSION, "invalid localization version")
    _require(localization.get("sourceAttributionResolved") is True
             and localization.get("pixelDataChangedFrozenEventDefinition") is False,
             "localization did not preserve the frozen event")
    _require(localization.get("binaryConfirmationSHA256") == sha256_json(binary_confirmation),
             "binary/localization hash chain mismatch")
    relationship_by_class = {
        "TARGET_CONSISTENT_ECLIPSE_SOURCE": "TARGET_ASSOCIATED",
        "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE": "OFF_TARGET",
    }
    relationship = relationship_by_class.get(localization.get("classification"))
    _require(relationship is not None, "localization has no catalog-attributed source")

    _require(source_review.get("resultVersion") == REVIEW_VERSION
             and source_review.get("sourceAttributionReviewPassed") is True,
             "source-attribution review did not pass")
    expected_review_class = ("TARGET_SOURCE_ATTRIBUTION_REVIEW_PASSED" if relationship == "TARGET_ASSOCIATED"
                             else "OFF_TARGET_CATALOG_ATTRIBUTION_REVIEW_PASSED")
    _require(source_review.get("classification") == expected_review_class,
             "review/source relationship conflict")
    _require(source_review.get("sourceLocalizationSHA256") == sha256_json(localization),
             "localization/review hash chain mismatch")
    _require(source_review.get("binaryConfirmationSHA256") == sha256_json(binary_confirmation),
             "review binary hash mismatch")
    sectors = source_review.get("supportingIndependentSectors")
    _require(isinstance(sectors, list) and len(sectors) >= MIN_INDEPENDENT_SUPPORT
             and len(sectors) == len(set(sectors))
             and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in sectors),
             "invalid independent-sector support")
    _require(source_review.get("supportingIndependentSectorCount") == len(sectors)
             and source_review.get("primarySectorCanSatisfyReplication") is False
             and not source_review.get("conflictingIndependentSectors")
             and not source_review.get("duplicateIndependentSectors"),
             "review replication gate failed")

    _require(frozen_external_response.get("resultVersion") == FREEZE_VERSION, "invalid freeze version")
    _require(frozen_external_response.get("sourceAttributionReviewSHA256") == sha256_json(source_review),
             "review/freeze hash chain mismatch")
    _require(external_result.get("resultVersion") == EXTERNAL_RESULT_VERSION
             and external_result.get("externalEvidenceFreezeSHA256") == sha256_json(frozen_external_response),
             "freeze/result hash chain mismatch")
    _require(external_result.get("externalCompanionEvidenceResolved") is True
             and external_result.get("recommendedNextTest") == "FINAL_COMPANION_EVIDENCE_SYNTHESIS",
             "external companion evidence is unresolved")
    _require(external_result.get("softwareBlindPhotometricEvidencePreserved") is True
             and external_result.get("externalKnownObjectCatalogUsed") is True,
             "evidence separation gate failed")

    source = source_review.get("attributedSource") or {}
    identifiers = frozen_external_response.get("attributedSourceIdentifiers") or {}
    row = external_result.get("selectedExternalRow")
    _require(isinstance(row, dict), "selected external row is missing")
    tic = canonical_tic_id(source.get("ticID")); freeze_tic = canonical_tic_id(identifiers.get("ticID"))
    row_tic = canonical_tic_id(row.get("tic_id"))
    _require(tic == freeze_tic == row_tic, "TIC identity changed across evidence")
    source_gaia = canonical_gaia_dr3_id(source.get("gaiaDR3SourceID"))
    freeze_gaia = canonical_gaia_dr3_id(identifiers.get("gaiaDR3SourceID"))
    row_gaia = canonical_gaia_dr3_id(row.get("gaia_dr3_id"))
    _require(source_gaia == freeze_gaia == row_gaia, "Gaia identity changed across evidence")
    _require(source_review.get("attributedCatalogHypothesis") == localization.get("attributedCatalogHypothesis"),
             "attributed source changed")

    regime = external_result.get("supportedCompanionMassRegime")
    expected_external = {
        "PLANETARY": "PERIOD_MATCHED_PLANETARY_MASS_COMPANION_SUPPORTED",
        "BROWN_DWARF": "PERIOD_MATCHED_BROWN_DWARF_MASS_COMPANION_SUPPORTED",
        "STELLAR": "PERIOD_MATCHED_STELLAR_MASS_COMPANION_SUPPORTED",
    }
    _require(regime in expected_external and external_result.get("classification") == expected_external[regime],
             "classification/mass-regime conflict")
    period = _finite(external_result.get("externalOrbitalPeriodDays"), "external period")
    difference = _finite(external_result.get("externalOrbitalPeriodDifferenceDays"), "period difference")
    mass = _finite(external_result.get("externalMassJupiter"), "external mass")
    interval = external_result.get("externalMassIntervalJupiter")
    _require(period > 0 and difference >= 0 and isinstance(interval, list) and len(interval) == 2,
             "invalid period or mass interval")
    low, high = (_finite(v, "mass interval") for v in interval)
    _require(0 < low <= mass <= high and low < high, "invalid mass interval")
    row_period = _finite(row.get("pl_orbper"), "selected-row period")
    row_mass = _finite(row.get("pl_bmassj"), "selected-row mass")
    row_down = _finite(row.get("pl_bmassjerr2"), "selected-row lower mass error")
    row_up = _finite(row.get("pl_bmassjerr1"), "selected-row upper mass error")
    _require(row_period == period and row_mass == mass
             and math.isclose(row_mass + row_down, low, abs_tol=1e-12)
             and math.isclose(row_mass + row_up, high, abs_tol=1e-12),
             "selected external row was mutated or summarized inconsistently")
    refined = _finite((localization.get("frozenEphemeris") or {}).get("refinedPeriodDays"), "refined period")
    _require(refined > 0 and math.isclose(abs(period - refined), difference, abs_tol=1e-12),
             "period evidence is inconsistent")
    _require(frozen_external_response.get("frozenEphemeris") == localization.get("frozenEphemeris")
             and source_review.get("frozenEphemeris") == localization.get("frozenEphemeris"),
             "frozen ephemeris changed")

    nature = {"PLANETARY": "PLANETARY", "BROWN_DWARF": "BROWN_DWARF", "STELLAR": "STELLAR"}[regime]
    classification = (f"TARGET_ASSOCIATED_KNOWN_{nature}_COMPANION_SUPPORTED" if relationship == "TARGET_ASSOCIATED"
                      else f"OFF_TARGET_KNOWN_{nature}_COMPANION_IDENTIFIED")
    return {
        "resultVersion": RESULT_VERSION, "classification": classification,
        "sourceRelationship": relationship, "sourceAttributionResolved": True,
        "externalCompanionEvidenceResolved": True, "supportedCompanionMassRegime": regime,
        "externalOrbitalPeriodDays": period, "externalOrbitalPeriodDifferenceDays": difference,
        "externalMassJupiter": mass, "externalMassIntervalJupiter": [low, high],
        "attributedTICID": tic, "attributedGaiaDR3SourceID": source_gaia,
        "supportingIndependentSectors": sectors, "supportingIndependentSectorCount": len(sectors),
        "frozenRefinedEphemeris": localization["frozenEphemeris"],
        "binaryConfirmationSHA256": sha256_json(binary_confirmation),
        "sourceLocalizationSHA256": sha256_json(localization),
        "sourceAttributionReviewSHA256": sha256_json(source_review),
        "externalEvidenceFreezeSHA256": sha256_json(frozen_external_response),
        "externalCompanionEvidenceSHA256": sha256_json(external_result),
        "autonomousCompanionEvidenceComplete": True,
        "softwareBlindPhotometricEvidencePreserved": True, "externalKnownObjectCatalogUsed": True,
        "catalogAnswerKeyUsed": False, "companionNatureResolved": True,
        "physicalMechanismResolved": False, "automaticDiscoveryClaim": False,
        "scientificStatement": ("OpenStar independently recovered software-blind photometric and spatial evidence "
                                "consistent with a previously known companion; this is not a new-object discovery claim."),
        "recommendedNextTest": "HUMAN_SCIENTIFIC_REVIEW",
    }
