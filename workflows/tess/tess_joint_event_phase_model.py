"""Software-blind coordinator-side joint empirical event/phase modelling.

The fitter deliberately operates on the immutable Float64 cadence arrays.  It
has no archive client and no target identity: fixed ephemerides and generic
trapezoids are the only scientific assumptions made here.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .tess_event_depth_accuracy import validate_audit_hash, validate_freeze

HANDLER_ID = "openstar.tess.joint-event-phase-model.fit"
RESULT_VERSION = "openstar.tess-joint-transit-eclipse-phase-curve-model.v1"
DURATION_MULTIPLIERS = (0.70, 0.85, 1.00, 1.15, 1.30)
INGRESS_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
SECONDARY_PHASE_OFFSETS = (-0.02, 0.0, 0.02)
MIN_INDEPENDENT_SECTORS = 3
MAX_CONDITION_NUMBER = 1.0e10


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _finalize(value: dict[str, Any]) -> dict[str, Any]:
    answer = dict(value); answer.pop("modelSHA256", None)
    answer["modelSHA256"] = _sha(answer)
    return answer


def validate_model_hash(value: dict[str, Any]) -> str:
    copy = dict(value); claimed = copy.pop("modelSHA256", None)
    if value.get("resultVersion") != RESULT_VERSION or claimed != _sha(copy):
        raise ValueError("persisted joint event/phase model hash mismatch")
    return claimed


def model_required(audit: dict[str, Any]) -> bool:
    return (audit.get("status") == "COMPLETE"
            and audit.get("suitableForLaterPrecisionModeling") is True
            and audit.get("recommendedNextTest") == "JOINT_TRANSIT_ECLIPSE_PHASE_CURVE_MODELING"
            and audit.get("externalCatalogInformationUsed") is False
            and audit.get("catalogAnswerKeyUsed") is False)


def _template(phase: np.ndarray, center: float, duration_phase: float,
              ingress_fraction: float, exposure_phase: float) -> np.ndarray:
    import numpy as np
    # Eleven midpoint samples make the exposure convolution deterministic and
    # preserve mixed-cadence integrations without constructing huge matrices.
    offsets = (np.arange(11, dtype=float) + .5) / 11 - .5
    p = ((phase[:, None] + offsets * exposure_phase - center + .5) % 1) - .5
    distance = np.abs(p); half = duration_phase / 2
    ingress = max(half * ingress_fraction, np.finfo(float).eps)
    instantaneous = np.clip((half - distance) / ingress, 0, 1)
    return instantaneous.mean(axis=1)


def _matrix(rows: list[dict[str, Any]], period: float, epoch: float, duration: float,
            ingress: float, secondary_offset: float, include_transit: bool = True) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    import numpy as np
    blocks, values, slices, start = [], [], [], 0
    count = len(rows)
    for index, row in enumerate(rows):
        time = np.asarray(row["timeBTJDFloat64"], dtype=float)
        flux = np.asarray(row["relativeFluxFloat64"], dtype=float)
        phase = ((time - epoch) / period) % 1
        exposure = float(row["cadenceSeconds"]) / 86400 / period
        primary = _template(phase, 0, duration / period, ingress, exposure)
        secondary = _template(phase, .5 + secondary_offset, duration / period, ingress, exposure)
        angle = 2*np.pi*phase
        local = (time - np.median(time)) / max(np.ptp(time), period)
        base = np.zeros((len(time), 2*count)); base[:, 2*index] = 1; base[:, 2*index+1] = local
        columns = ([primary] if include_transit else []) + [secondary, np.sin(angle), np.cos(angle),
                   np.sin(2*angle), np.cos(2*angle)]
        blocks.append(np.column_stack(columns + [base])); values.append(flux)
        slices.append(slice(start, start+len(time))); start += len(time)
    return np.vstack(blocks), np.concatenate(values), slices


def _solve(rows: list[dict[str, Any]], period: float, epoch: float, duration: float,
           ingress: float, secondary_offset: float, include_transit: bool = True) -> dict[str, Any]:
    import numpy as np
    matrix, flux, slices = _matrix(rows, period, epoch, duration, ingress, secondary_offset, include_transit)
    # Initial residual clipping applies only far from both events. Event samples
    # always retain unit weight, preventing real deficits from clipping themselves.
    beta, _, rank, singular = np.linalg.lstsq(matrix, flux, rcond=None)
    residual = flux-matrix@beta
    phase = np.concatenate([((np.asarray(r["timeBTJDFloat64"])-epoch)/period) % 1 for r in rows])
    protected = (np.abs((phase+.5)%1-.5) <= duration/period
                 ) | (np.abs((phase-.5-secondary_offset+.5)%1-.5) <= duration/period)
    med = np.median(residual[~protected]) if np.any(~protected) else 0
    scale = 1.4826*np.median(np.abs(residual[~protected]-med)) if np.any(~protected) else 0
    keep = protected | (np.abs(residual-med) <= 5*max(scale, np.finfo(float).eps))
    weighted = matrix[keep]; selected = flux[keep]
    beta, _, rank, singular = np.linalg.lstsq(weighted, selected, rcond=None)
    residual = flux-matrix@beta; rss = float(residual@residual)
    dof = max(int(keep.sum())-matrix.shape[1], 1); variance = float((residual[keep]@residual[keep])/dof)
    covariance = np.linalg.pinv(weighted.T@weighted)*variance
    condition = float(singular[0]/singular[-1]) if len(singular) and singular[-1] > 0 else math.inf
    bic = len(flux)*math.log(max(rss/len(flux), np.finfo(float).tiny))+matrix.shape[1]*math.log(len(flux))
    return {"beta": beta, "covariance": covariance, "residual": residual, "rss": rss, "bic": bic,
            "condition": condition, "rank": int(rank), "columnCount": matrix.shape[1], "slices": slices,
            "retained": int(keep.sum()), "sampleCount": len(flux)}


def _unresolved(base: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return _finalize({**base, "status": "UNRESOLVED", "classification": "PRECISION_EMPIRICAL_TRANSIT_DEPTH_UNRESOLVED",
        "precisionEmpiricalTransitDepthResolved": False, "unresolvedReasons": sorted(set(reasons)),
        "recommendedNextTest": "ADDITIONAL_PRECISION_PHOTOMETRY", "workflowNextStage": "EXTERNAL_EVIDENCE_FREEZE"})


def fit_joint_event_phase_model(freeze: dict[str, Any], binary: dict[str, Any], audit: dict[str, Any], *,
                                binary_confirmation_sha256: str, chronology_proof: dict[str, Any]) -> dict[str, Any]:
    """Fit a deterministic joint model after validating every upstream binding."""
    import numpy as np
    base = {"resultVersion": RESULT_VERSION, "binaryConfirmationSHA256": binary_confirmation_sha256,
        "photometryFreezeSHA256": freeze.get("freezeSHA256"), "depthAttenuationAuditSHA256": audit.get("auditSHA256"),
        "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False,
        "companionRadiusInferred": False, "planetToStarRadiusRatioInferred": False,
        "limbDarkenedPhysicalGeometryClaimed": False, "stellarDensityInferred": False,
        "fullPhysicalTransitSolutionClaimed": False, "uniqueReflectionThermalInterpretationClaimed": False,
        "automaticDiscoveryClaimed": False, "chronologyProof": chronology_proof}
    try:
        if not model_required(audit): raise ValueError("DEPTH_AUDIT_MODEL_GATE_NOT_SATISFIED")
        validate_audit_hash(audit); validate_freeze(freeze, binary, binary_confirmation_sha256)
        if audit.get("binaryConfirmationSHA256") != binary_confirmation_sha256:
            raise ValueError("AUDIT_BINARY_HASH_BINDING_MISMATCH")
        chronology = chronology_proof
        handlers = chronology.get("completedStageHandlerIDs") or []
        if (chronology.get("verifiedFromCompletedStages") is not True
                or chronology.get("externalEvidenceStageAlreadyCompleted") is not False
                or any("external-companion-evidence" in str(x) for x in handlers)):
            raise ValueError("MODEL_BEFORE_EXTERNAL_QUERY_CHRONOLOGY_UNPROVEN")
        ephemeris = binary["linearEphemeris"]; period = float(ephemeris["refinedPeriodDays"]); epoch = float(ephemeris["referenceEpoch"])
        if ephemeris.get("coherent") is not True or not math.isfinite(period) or period <= 0 or not math.isfinite(epoch):
            raise ValueError("INVALID_FROZEN_EPHEMERIS")
        supporting = list((binary.get("independentEvidence") or {}).get("supportingSectors") or [])
        rows = list(freeze["sectors"]); eligible = [r["sector"] for r in rows]
        independent = [x for x in supporting if x in eligible]
        if len(set(independent)) < MIN_INDEPENDENT_SECTORS: raise ValueError("INSUFFICIENT_INDEPENDENT_SUPPORTING_SECTORS")
        established = float(audit["eventDurationDays"])
    except (KeyError, TypeError, ValueError) as error:
        return _unresolved(base, [str(error)])

    candidates = []
    for multiplier in DURATION_MULTIPLIERS:
        for ingress in INGRESS_FRACTIONS:
            for offset in SECONDARY_PHASE_OFFSETS:
                duration = established*multiplier
                fit = _solve(rows, period, epoch, duration, ingress, offset)
                candidates.append((fit["bic"], duration, ingress, offset, fit))
    _, duration, ingress, offset, fit = min(candidates, key=lambda x: (x[0], x[1], x[2], abs(x[3])))
    beta = fit["beta"]; depth = float(-beta[0]); eclipse = float(-beta[1])
    formal = float(math.sqrt(max(fit["covariance"][0, 0], 0)))
    no_transit = _solve(rows, period, epoch, duration, ingress, offset, False)
    delta_bic = float(no_transit["bic"]-fit["bic"])
    jackknife, jk_depths = [], []
    for sector in independent:
        subset = [row for row in rows if row["sector"] != sector]
        value = _solve(subset, period, epoch, duration, ingress, offset)
        estimate = float(-value["beta"][0]); jk_depths.append(estimate)
        jackknife.append({"omittedIndependentSector": sector, "transitDepthFractionalFlux": estimate,
                          "conditionNumber": value["condition"]})
    jk_unc = float(math.sqrt((len(jk_depths)-1)/len(jk_depths)*sum((x-np.mean(jk_depths))**2 for x in jk_depths)))
    per_sector, sector_depths = [], []
    for row in rows:
        one = _solve([row], period, epoch, duration, ingress, offset); d = float(-one["beta"][0])
        sector_depths.append(d); per_sector.append({"sector": row["sector"], "role": "INDEPENDENT" if row["sector"] in independent else "PRIMARY",
            "transitDepthFractionalFlux": d, "sampleCount": one["sampleCount"], "conditionNumber": one["condition"]})
    scatter = float(np.std(sector_depths, ddof=1)/math.sqrt(len(sector_depths))) if len(sector_depths)>1 else 0
    conservative = max(formal, jk_unc, scatter, np.finfo(float).eps)
    equivalent = depth*(1-ingress/2)
    boundary = (duration in (established*DURATION_MULTIPLIERS[0], established*DURATION_MULTIPLIERS[-1])
                or ingress in (INGRESS_FRACTIONS[0], INGRESS_FRACTIONS[-1]) or offset in (SECONDARY_PHASE_OFFSETS[0], SECONDARY_PHASE_OFFSETS[-1]))
    independent_depths = [p["transitDepthFractionalFlux"] for p in per_sector if p["role"] == "INDEPENDENT"]
    consistency = bool(independent_depths and max(abs(x-depth) for x in independent_depths) <= max(5*conservative, .5*depth))
    stability = bool(jk_depths and max(abs(x-depth) for x in jk_depths) <= max(3*conservative, .35*depth))
    gates = {"positiveFiniteTransitDepth": math.isfinite(depth) and depth > 0,
        "adequateTransitSignificance": depth/conservative >= 5, "atLeastThreeIndependentSupportingSectors": len(set(independent)) >= 3,
        "crossSectorDepthConsistency": consistency, "leaveOneSectorOutStable": stability,
        "acceptableFitConditioning": fit["condition"] <= MAX_CONDITION_NUMBER and fit["rank"] == fit["columnCount"],
        "adequatePrimaryAndBaselineCoverage": all(p["sampleCount"] >= 20 for p in per_sector),
        "nonlinearSolutionNotBoundaryPinned": not boundary, "finitePositiveUncertainty": math.isfinite(conservative) and conservative > 0,
        "meaningfulNoTransitImprovement": delta_bic >= 10}
    unresolved = [name for name, passed in gates.items() if not passed]
    eclipse_unc = float(math.sqrt(max(fit["covariance"][1, 1], 0))); eclipse_resolved = eclipse > 0 and eclipse_unc > 0 and eclipse/eclipse_unc >= 3
    phase = beta[2:6]; phase_unc = np.sqrt(np.maximum(np.diag(fit["covariance"])[2:6], 0))
    fundamental_resolved = bool(np.linalg.norm(phase[:2]) >= 3*np.linalg.norm(phase_unc[:2]))
    second_resolved = bool(np.linalg.norm(phase[2:]) >= 3*np.linalg.norm(phase_unc[2:]))
    result = {**base, "status": "COMPLETE" if not unresolved else "UNRESOLVED",
        "classification": "PRECISION_EMPIRICAL_TRANSIT_DEPTH_RESOLVED" if not unresolved else "PRECISION_EMPIRICAL_TRANSIT_DEPTH_UNRESOLVED",
        "precisionEmpiricalTransitDepthResolved": not unresolved, "unresolvedReasons": unresolved,
        "frozenPeriodDays": period, "frozenReferenceEpochBTJD": epoch, "timingSectors": eligible,
        "independentSupportingSectors": independent, "independentSupportingSectorCount": len(set(independent)),
        "modelSpecification": {"primaryTemplate": "SHARED_EXPOSURE_INTEGRATED_TRAPEZOID", "oppositeConjunctionTemplate": "SHARED_EXPOSURE_INTEGRATED_TRAPEZOID",
            "phaseTerms": ["ORBITAL_SINE", "ORBITAL_COSINE", "TWICE_ORBITAL_SINE", "TWICE_ORBITAL_COSINE"],
            "perSectorBaseline": "INTERCEPT_PLUS_LINEAR_TIME", "durationMultipliers": list(DURATION_MULTIPLIERS),
            "ingressEgressFractionsOfHalfDuration": list(INGRESS_FRACTIONS), "secondaryPhaseOffsetGrid": list(SECONDARY_PHASE_OFFSETS),
            "exposureIntegration": "ELEVEN_DETERMINISTIC_MIDPOINT_SUBEXPOSURES_PER_FULL_CADENCE", "ephemerisHeldFixed": True},
        "globalFit": {"midTransitFractionalFluxDeficit": depth, "conservativeTransitDepthUncertainty": conservative,
            "equivalentBoxTransitDepthFractionalFlux": equivalent, "oppositeConjunctionEclipseDepthFractionalFlux": eclipse,
            "oppositeConjunctionEclipseUncertainty": eclipse_unc, "oppositeConjunctionEclipseStatus": "RESOLVED" if eclipse_resolved else "UNRESOLVED",
            "orbitalFrequencySineCoefficient": float(phase[0]), "orbitalFrequencyCosineCoefficient": float(phase[1]),
            "twiceOrbitalFrequencySineCoefficient": float(phase[2]), "twiceOrbitalFrequencyCosineCoefficient": float(phase[3]),
            "fundamentalPhaseCurveStatus": "RESOLVED" if fundamental_resolved else "UNRESOLVED", "secondHarmonicPhaseCurveStatus": "RESOLVED" if second_resolved else "UNRESOLVED",
            "eventDurationDays": duration, "ingressEgressDurationDays": duration*ingress/2, "secondaryPhaseOffset": offset},
        "uncertaintyDiagnostics": {"formalCovarianceTransitUncertainty": formal, "independentSectorJackknifeUncertainty": jk_unc,
            "sectorScatterUncertainty": scatter, "conservativeRule": "MAX_FORMAL_JACKKNIFE_SECTOR_SCATTER"},
        "perSectorDiagnostics": per_sector, "independentSectorJackknife": jackknife,
        "nestedModelComparisons": {"noTransitBIC": no_transit["bic"], "jointModelBIC": fit["bic"], "deltaBICForTransit": delta_bic},
        "resolutionGates": gates, "fitDiagnostics": {"conditionNumber": fit["condition"], "retainedSampleCount": fit["retained"], "sampleCount": fit["sampleCount"],
            "eventCadencesProtectedFromClipping": True}, "modelRanBeforeExternalKnownObjectQuery": True,
        "workflowNextStage": "EXTERNAL_EVIDENCE_FREEZE", "recommendedNextTest": "EXTERNAL_EVIDENCE_FREEZE" if not unresolved else "ADDITIONAL_PRECISION_PHOTOMETRY"}
    return _finalize(result)


# Stable descriptive alias for callers that prefer the scientific verb.
model_joint_transit_eclipse_phase_curve = fit_joint_event_phase_model
