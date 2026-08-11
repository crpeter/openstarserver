from __future__ import annotations

import math
from typing import Any

from .tess_claims import ClaimDecision, decision


G = 6.67430e-11
M_SUN_KG = 1.98847e30
R_SUN_M = 6.957e8
SECONDS_PER_DAY = 86400.0


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _period_match(observed_days: float, published_days: float) -> dict[str, Any]:
    candidates = (
        ("0.5x", observed_days * 0.5),
        ("1x", observed_days),
        ("2x", observed_days * 2.0),
    )
    relation, candidate = min(
        candidates,
        key=lambda item: abs(item[1] - published_days),
    )
    absolute_error = abs(candidate - published_days)
    relative_error = absolute_error / max(abs(published_days), 1e-12)
    return {
        "relation": relation,
        "observedEquivalentDays": candidate,
        "publishedPeriodDays": published_days,
        "absoluteErrorDays": absolute_error,
        "relativeError": relative_error,
        "matches": absolute_error <= max(0.02, 0.03 * published_days),
    }


def catalog_period_evidence(identity: dict[str, Any], observed_period: float) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    vsx = identity.get("vsx") or {}
    for match in vsx.get("matches") or []:
        period = _float(match.get("periodDays"))
        if period is None or period <= 0:
            continue
        comparison = _period_match(observed_period, period)
        comparison.update({
            "source": "AAVSO VSX",
            "name": match.get("name"),
            "classification": match.get("type"),
            "separationArcsec": match.get("separationArcsec"),
        })
        evidence.append(comparison)

    variability = identity.get("gaiaVariability") or {}
    for candidate in variability.get("periodCandidates") or []:
        period = _float(candidate.get("periodDays"))
        if period is None or period <= 0:
            continue
        comparison = _period_match(observed_period, period)
        comparison.update({
            "source": candidate.get("source") or "Gaia DR3",
            "field": candidate.get("field"),
            "classification": (variability.get("classification") or {}).get("class"),
        })
        evidence.append(comparison)

    evidence.sort(key=lambda item: (not item["matches"], item["relativeError"]))
    return evidence


def rotational_sanity(identity: dict[str, Any], period_days: float) -> dict[str, Any]:
    metadata = ((identity.get("tic") or {}).get("metadata") or {})
    radius_rsun = _float(metadata.get("radiusRsun"))
    mass_msun = _float(metadata.get("massMsun"))

    result: dict[str, Any] = {
        "evaluated": False,
        "periodDays": period_days,
        "radiusRsun": radius_rsun,
        "massMsun": mass_msun,
        "status": "unknown",
    }

    if radius_rsun is None or radius_rsun <= 0 or mass_msun is None or mass_msun <= 0:
        result["reason"] = "TIC radius and mass are both required for the deterministic rotation sanity test."
        return result

    radius_m = radius_rsun * R_SUN_M
    mass_kg = mass_msun * M_SUN_KG
    period_seconds = period_days * SECONDS_PER_DAY
    equatorial_speed = 2.0 * math.pi * radius_m / period_seconds
    critical_speed = math.sqrt(G * mass_kg / radius_m)
    ratio = equatorial_speed / critical_speed

    if ratio >= 1.0:
        status = "ruled-out"
    elif ratio >= 0.7:
        status = "strongly-disfavored"
    else:
        status = "not-ruled-out"

    result.update({
        "evaluated": True,
        "equatorialSpeedKmS": equatorial_speed / 1000.0,
        "criticalSpeedKmS": critical_speed / 1000.0,
        "equatorialToCriticalRatio": ratio,
        "status": status,
        "rule": "rotation is ruled out when v_eq >= sqrt(GM/R); strongly disfavored above 0.7 of critical speed",
    })
    return result


def analyze(primary: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    period = _float(primary.get("preferredPhysicalPeriodDays"))
    if period is None:
        period = _float(primary.get("candidatePeriodDays"))

    status = str(primary.get("periodStatus") or "").upper()
    confidence = str(primary.get("periodConfidence") or "none").lower()
    reliable = status == "RELIABLE" and confidence in {"high", "medium"}

    catalog_evidence = catalog_period_evidence(identity, period) if period else []
    best_catalog_match = next((item for item in catalog_evidence if item["matches"]), None)
    rotation = rotational_sanity(identity, period) if period else {
        "evaluated": False,
        "status": "unknown",
        "reason": "No usable period.",
    }

    harmonic_candidates = primary.get("harmonicCandidates") or []
    preferred_relation = primary.get("preferredPhysicalPeriodRelation") or "1x"

    hypotheses: list[dict[str, Any]] = []
    if best_catalog_match is not None:
        hypotheses.append({
            "id": "catalog-period",
            "status": "supported",
            "evidence": best_catalog_match,
        })
    else:
        hypotheses.append({
            "id": "catalog-period",
            "status": "not-found",
            "evidence": catalog_evidence[:5],
        })

    hypotheses.append({
        "id": "rotation-at-primary-period",
        "status": rotation["status"],
        "evidence": rotation,
    })

    if preferred_relation != "1x":
        hypotheses.append({
            "id": "harmonic-or-multiplicity",
            "status": "supported",
            "evidence": {
                "preferredRelation": preferred_relation,
                "harmonicCandidates": harmonic_candidates,
            },
        })
    else:
        hypotheses.append({
            "id": "harmonic-or-multiplicity",
            "status": "not-preferred",
            "evidence": {
                "preferredRelation": preferred_relation,
                "harmonicCandidates": harmonic_candidates,
            },
        })

    return {
        "primaryReliable": reliable,
        "periodStatus": status,
        "periodConfidence": confidence,
        "observedPeriodDays": period,
        "bestCatalogMatch": best_catalog_match,
        "catalogPeriodEvidence": catalog_evidence,
        "rotationSanity": rotation,
        "preferredPhysicalPeriodRelation": preferred_relation,
        "hypotheses": hypotheses,
    }


def plan(analysis: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    period = _float(analysis.get("observedPeriodDays"))
    best_match = analysis.get("bestCatalogMatch")
    rotation = analysis.get("rotationSanity") or {}
    query_errors = identity.get("queryErrors") or []

    if best_match is not None:
        claim = decision(
            "KNOWN_PERIOD_RECOVERED",
            "OpenStar recovered a period consistent with an external catalog period or its 0.5x/2x harmonic relation.",
            f"Best catalog source: {best_match.get('source')}",
        )
        return {
            "action": "STOP",
            "claimDecision": claim.as_dict(),
            "reason": "catalog-period-match",
        }

    if not analysis.get("primaryReliable"):
        claim = decision(
            "CANDIDATE_PERIOD" if period is not None else "HUMAN_REVIEW_REQUIRED",
            "The primary distributed search did not meet the reliable-period threshold.",
        )
        return {
            "action": "STOP",
            "claimDecision": claim.as_dict(),
            "reason": "primary-period-not-reliable",
        }

    if rotation.get("status") in {"ruled-out", "strongly-disfavored"}:
        return {
            "action": "LOW_FREQUENCY_FOLLOWUP",
            "reason": "primary-period-rotation-physically-disfavored",
            "claimDecision": None,
        }

    if analysis.get("preferredPhysicalPeriodRelation") != "1x":
        return {
            "action": "LOW_FREQUENCY_FOLLOWUP",
            "reason": "harmonic-cycle-preferred",
            "claimDecision": None,
        }

    if query_errors and not (identity.get("tic") or {}).get("found"):
        claim = decision(
            "HUMAN_REVIEW_REQUIRED",
            "Catalog identity could not be established because the required identity query failed.",
        )
        return {
            "action": "STOP",
            "claimDecision": claim.as_dict(),
            "reason": "identity-unresolved",
        }

    claim = decision(
        "INDEPENDENT_PERIOD_ESTIMATE",
        "The distributed period is reliable and no matching external catalog period was found in the queried sources.",
        "The deterministic rotation sanity test did not rule out the period.",
    )
    return {
        "action": "STOP",
        "claimDecision": claim.as_dict(),
        "reason": "reliable-uncataloged-period-no-followup-trigger",
    }


def interpret_followup(
    primary_analysis: dict[str, Any],
    followup_result: dict[str, Any],
) -> dict[str, Any]:
    datasets = followup_result.get("datasets") or []
    followup = datasets[0] if datasets else followup_result

    followup_status = str(followup.get("periodStatus") or "").upper()
    followup_confidence = str(followup.get("periodConfidence") or "none").lower()
    followup_period = _float(followup.get("preferredPhysicalPeriodDays"))
    if followup_period is None:
        followup_period = _float(followup.get("candidatePeriodDays"))

    reliable = followup_status == "RELIABLE" and followup_confidence in {"high", "medium"}

    if reliable and followup_period is not None:
        claim: ClaimDecision = decision(
            "INDEPENDENT_PERIOD_ESTIMATE",
            "A deterministic follow-up search found a reliable period after the primary interpretation triggered a lower-frequency test.",
            "The estimate remains independent because no matching catalog period was used to choose the numeric result.",
        )
        return {
            "claimDecision": claim.as_dict(),
            "selectedPeriodDays": followup_period,
            "selectedSource": "low-frequency-followup",
            "followupReliable": True,
            "followupDataset": followup,
        }

    claim = decision(
        "HUMAN_REVIEW_REQUIRED",
        "The primary interpretation required a decisive lower-frequency follow-up, but that follow-up did not produce a reliable period.",
    )
    return {
        "claimDecision": claim.as_dict(),
        "selectedPeriodDays": primary_analysis.get("observedPeriodDays"),
        "selectedSource": "primary-unresolved",
        "followupReliable": False,
        "followupDataset": followup,
    }
