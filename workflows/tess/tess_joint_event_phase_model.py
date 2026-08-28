"""Coordinator-side, software-blind joint empirical event/phase modelling.

Only immutable full-cadence Float64 photometry and its verified scientific
provenance enter this module.  It has deliberately no archive or catalog client.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .tess_event_depth_accuracy import AUDIT_VERSION, validate_audit_hash, validate_freeze

HANDLER_ID = "openstar.tess.joint-event-phase-model.fit"
RESULT_VERSION = "openstar.tess-joint-transit-eclipse-phase-curve-model.v1"
DURATION_MULTIPLIERS = (0.70, 0.85, 1.00, 1.15, 1.30)
INGRESS_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
SECONDARY_PHASE_OFFSETS = (-0.02, 0.0, 0.02)
MIN_INDEPENDENT_SECTORS = 3
MAX_CONDITION_NUMBER = 1.0e10
MAX_REDUCED_HETEROGENEITY = 4.0
MAX_FRACTIONAL_SECTOR_DEVIATION = 0.50
MAX_FRACTIONAL_JACKKNIFE_MOVEMENT = 0.25


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
    return (audit.get("resultVersion") == AUDIT_VERSION
            and audit.get("status") == "COMPLETE"
            and audit.get("suitableForLaterPrecisionModeling") is True
            and audit.get("recommendedNextTest") == "JOINT_TRANSIT_ECLIPSE_PHASE_CURVE_MODELING"
            and audit.get("externalCatalogInformationUsed") is False
            and audit.get("catalogAnswerKeyUsed") is False)


def chronology_from_completed_stages(completed, required_handlers, external_handlers):
    """Derive a strict, unique pre-model ledger proof from completed stages."""
    required_handlers=list(required_handlers); external_handlers=set(external_handlers)
    positions=[[index for index,stage in enumerate(completed) if stage.handler_id==handler]
               for handler in required_handlers]
    external=[stage for stage in completed if stage.handler_id in external_handlers]
    unique_positions=[items[0] for items in positions if len(items)==1]
    verified=(len(unique_positions)==len(required_handlers)
              and unique_positions==sorted(unique_positions) and not external)
    return {"verifiedFromCompletedStages":verified,
        "externalEvidenceStageAlreadyCompleted":bool(external),
        "requiredPreModelStageHandlerIDs":required_handlers,
        "requiredPreModelStageIDs":[completed[items[0]].id if len(items)==1 else None for items in positions],
        "requiredPreModelStageOccurrenceCounts":[len(items) for items in positions],
        "completedStageHandlerIDs":[stage.handler_id for stage in completed],
        "completedStageIDs":[stage.id for stage in completed]}


def _template(phase, center: float, duration_phase: float, ingress_fraction: float,
              exposure_phase: float, *, integrate: bool = True):
    import numpy as np
    offsets = ((np.arange(11, dtype=float)+.5)/11-.5) if integrate else np.array([0.])
    p = ((phase[:, None]+offsets*exposure_phase-center+.5)%1)-.5
    half = duration_phase/2; ramp = max(half*ingress_fraction, np.finfo(float).eps)
    return np.clip((half-np.abs(p))/ramp, 0, 1).mean(axis=1)


def _matrix(rows, period, epoch, duration, ingress, secondary_offset, components, *, integrate=True):
    import numpy as np
    blocks, values, slices, start = [], [], [], 0; count = len(rows)
    names = [name for name in ("transit", "eclipse", "fundamentalSine", "fundamentalCosine",
                                "secondSine", "secondCosine") if name in components]
    for index, row in enumerate(rows):
        time=np.asarray(row["timeBTJDFloat64"], dtype=float); flux=np.asarray(row["relativeFluxFloat64"], dtype=float)
        phase=((time-epoch)/period)%1; exposure=float(row["cadenceSeconds"])/86400/period; angle=2*np.pi*phase
        columns={"transit": _template(phase,0,duration/period,ingress,exposure,integrate=integrate),
                 "eclipse": _template(phase,.5+secondary_offset,duration/period,ingress,exposure,integrate=integrate),
                 "fundamentalSine":np.sin(angle), "fundamentalCosine":np.cos(angle),
                 "secondSine":np.sin(2*angle), "secondCosine":np.cos(2*angle)}
        base=np.zeros((len(time),2*count)); base[:,2*index]=1
        base[:,2*index+1]=(time-np.median(time))/max(float(np.ptp(time)),period)
        blocks.append(np.column_stack([columns[name] for name in names]+[base])); values.append(flux)
        slices.append(slice(start,start+len(time))); start += len(time)
    return np.vstack(blocks),np.concatenate(values),slices,names


def _solve(rows, period, epoch, duration, ingress, secondary_offset, components=None, *, integrate=True):
    import numpy as np
    components=set(components or {"transit","eclipse","fundamentalSine","fundamentalCosine","secondSine","secondCosine"})
    matrix,flux,slices,names=_matrix(rows,period,epoch,duration,ingress,secondary_offset,components,integrate=integrate)
    beta,_,rank,singular=np.linalg.lstsq(matrix,flux,rcond=None); residual=flux-matrix@beta
    phase=np.concatenate([((np.asarray(r["timeBTJDFloat64"])-epoch)/period)%1 for r in rows])
    protected=(np.abs((phase+.5)%1-.5)<=duration/period)|(np.abs((phase-.5-secondary_offset+.5)%1-.5)<=duration/period)
    outside=residual[~protected]; med=np.median(outside) if len(outside) else 0
    scale=1.4826*np.median(np.abs(outside-med)) if len(outside) else 0
    keep=protected|(np.abs(residual-med)<=5*max(scale,np.finfo(float).eps))
    selected_matrix=matrix[keep]; selected_flux=flux[keep]
    beta,_,rank,singular=np.linalg.lstsq(selected_matrix,selected_flux,rcond=None)
    residual=flux-matrix@beta; rss=float(residual@residual); dof=max(int(keep.sum())-matrix.shape[1],1)
    covariance=np.linalg.pinv(selected_matrix.T@selected_matrix)*float((residual[keep]@residual[keep])/dof)
    condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>0 else math.inf
    bic=len(flux)*math.log(max(rss/len(flux),np.finfo(float).tiny))+matrix.shape[1]*math.log(len(flux))
    coefficients={name:float(beta[i]) for i,name in enumerate(names)}
    uncertainties={name:float(math.sqrt(max(covariance[i,i],0))) for i,name in enumerate(names)}
    return {"coefficients":coefficients,"uncertainties":uncertainties,"covariance":covariance,"rss":rss,"bic":float(bic),
            "condition":condition,"rank":int(rank),"columnCount":matrix.shape[1],"retained":int(keep.sum()),
            "sampleCount":len(flux),"eventCadencesProtectedFromClipping":True}


def _select(rows,period,epoch,established,components=None,*,integrate=True):
    candidates=[]
    for multiplier in DURATION_MULTIPLIERS:
        for ingress in INGRESS_FRACTIONS:
            for offset in SECONDARY_PHASE_OFFSETS:
                fit=_solve(rows,period,epoch,established*multiplier,ingress,offset,components,integrate=integrate)
                candidates.append((fit["bic"],established*multiplier,ingress,offset,fit))
    return min(candidates,key=lambda x:(x[0],x[1],x[2],abs(x[3])))


def _unresolved(base,reasons):
    return _finalize({**base,"status":"UNRESOLVED","classification":"PRECISION_EMPIRICAL_TRANSIT_DEPTH_UNRESOLVED",
        "precisionEmpiricalTransitDepthResolved":False,"unresolvedReasons":sorted(set(reasons)),
        "recommendedNextTest":"ADDITIONAL_PRECISION_PHOTOMETRY","workflowNextStage":"EXTERNAL_EVIDENCE_FREEZE"})


def _validate_upstream(freeze,binary,audit,binary_hash):
    if audit.get("resultVersion") != AUDIT_VERSION: raise ValueError("AUDIT_RESULT_VERSION_MISMATCH")
    validate_audit_hash(audit)
    if audit.get("binaryConfirmationSHA256") != binary_hash: raise ValueError("AUDIT_BINARY_HASH_BINDING_MISMATCH")
    if audit.get("catalogAnswerKeyUsed") is not False or audit.get("externalCatalogInformationUsed") is not False:
        raise ValueError("AUDIT_BLINDNESS_GATE_FAILED")
    if not model_required(audit): raise ValueError("DEPTH_AUDIT_MODEL_GATE_NOT_SATISFIED")
    if binary.get("catalogAnswerKeyUsed") is not False:
        raise ValueError("BINARY_BLINDNESS_GATE_FAILED")
    independent=binary.get("independentEvidence") or {}; support=independent.get("supportingIndependentSectorCount")
    sectors=independent.get("supportingSectors")
    if independent.get("classification") != "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED":
        raise ValueError("INDEPENDENT_REPLICATION_CLASSIFICATION_MISMATCH")
    if (isinstance(support,bool) or not isinstance(support,int) or support<MIN_INDEPENDENT_SECTORS
            or not isinstance(sectors,list) or support!=len(sectors)
            or any(isinstance(value,bool) or not isinstance(value,int) or value<=0 for value in sectors)
            or len(set(sectors))!=len(sectors)):
        raise ValueError("INDEPENDENT_SUPPORT_COUNT_OR_SECTOR_LIST_INVALID")
    if (independent.get("independentLinearEphemeris") or {}).get("coherent") is not True:
        raise ValueError("INDEPENDENT_EPHEMERIS_INCOHERENT")
    ephemeris=binary.get("linearEphemeris") or {}
    if ephemeris.get("coherent") is not True: raise ValueError("FINAL_EPHEMERIS_INCOHERENT")
    validate_freeze(freeze,binary,binary_hash)
    timing_sectors=ephemeris.get("timingSectors")
    frozen_sectors=[row.get("sector") for row in freeze.get("sectors") or []]
    if (not isinstance(timing_sectors,list) or not set(sectors).issubset(set(timing_sectors))
            or not set(sectors).issubset(set(frozen_sectors))):
        raise ValueError("INDEPENDENT_SUPPORT_SECTORS_NOT_IN_FROZEN_TIMING_SET")
    independent_rows=[row for row in binary.get("sectorResults") or []
                      if row.get("role")=="INDEPENDENT" and row.get("usable") is True]
    row_sectors=[row.get("sector") for row in independent_rows]
    if (any(isinstance(value,bool) or not isinstance(value,int) or value<=0 for value in row_sectors)
            or len(row_sectors)!=len(set(row_sectors)) or len(row_sectors)!=len(sectors)
            or set(row_sectors)!=set(sectors)):
        raise ValueError("INDEPENDENT_BINARY_SECTOR_RESULTS_INVALID")
    return list(sectors),ephemeris


def _event_block_uncertainty(row,period,epoch,duration,ingress,offset):
    """Within-sector orbit-block jackknife, independent of other sectors."""
    import numpy as np
    times=np.asarray(row["timeBTJDFloat64"],dtype=float)
    cycles=np.floor((times-epoch)/period+.5).astype(int)
    unique=sorted(set(int(value) for value in cycles))
    estimates=[]
    for cycle in unique:
        keep=cycles!=cycle
        if int(keep.sum())<20: continue
        subset={**row,"timeBTJDFloat64":times[keep].tolist(),
                "relativeFluxFloat64":np.asarray(row["relativeFluxFloat64"],dtype=float)[keep].tolist()}
        fit=_solve([subset],period,epoch,duration,ingress,offset)
        estimates.append(-fit["coefficients"]["transit"])
    if len(estimates)<2: return 0.0,len(estimates)
    mean=float(np.mean(estimates))
    uncertainty=math.sqrt((len(estimates)-1)/len(estimates)*sum((value-mean)**2 for value in estimates))
    return float(uncertainty),len(estimates)


def fit_joint_event_phase_model(freeze,binary,audit,*,binary_confirmation_sha256,chronology_proof):
    base={"resultVersion":RESULT_VERSION,"binaryConfirmationSHA256":binary_confirmation_sha256,
          "photometryFreezeSHA256":freeze.get("freezeSHA256"),"depthAttenuationAuditSHA256":audit.get("auditSHA256"),
          "externalCatalogInformationUsed":False,"catalogAnswerKeyUsed":False,"companionRadiusInferred":False,
          "planetToStarRadiusRatioInferred":False,"limbDarkenedPhysicalGeometryClaimed":False,"stellarDensityInferred":False,
          "fullPhysicalTransitSolutionClaimed":False,"uniqueReflectionThermalInterpretationClaimed":False,
          "automaticDiscoveryClaimed":False,"chronologyProof":chronology_proof}
    try:
        independent,ephemeris=_validate_upstream(freeze,binary,audit,binary_confirmation_sha256)
        handlers=chronology_proof.get("completedStageHandlerIDs") or []
        if (chronology_proof.get("verifiedFromCompletedStages") is not True
                or chronology_proof.get("externalEvidenceStageAlreadyCompleted") is not False
                or any("external-companion-evidence" in str(x) for x in handlers)):
            raise ValueError("MODEL_BEFORE_EXTERNAL_QUERY_CHRONOLOGY_UNPROVEN")
        period=float(ephemeris["refinedPeriodDays"]); epoch=float(ephemeris["referenceEpoch"])
        if not math.isfinite(period) or period<=0 or not math.isfinite(epoch): raise ValueError("INVALID_FROZEN_EPHEMERIS")
        rows=list(freeze["sectors"]); established=float(audit["eventDurationDays"])
        if not math.isfinite(established) or established<=0: raise ValueError("INVALID_EVENT_DURATION")
    except (KeyError,TypeError,ValueError) as error: return _unresolved(base,[str(error)])
    import numpy as np

    _,duration,ingress,offset,fit=_select(rows,period,epoch,established)
    depth=-fit["coefficients"]["transit"]; formal=fit["uncertainties"]["transit"]
    per_sector=[]
    for row in rows:
        _,sd,si,so,one=_select([row],period,epoch,established)
        block_uncertainty,block_count=_event_block_uncertainty(row,period,epoch,sd,si,so)
        formal_uncertainty=one["uncertainties"]["transit"]
        per_sector.append({"sector":row["sector"],"role":"INDEPENDENT" if row["sector"] in independent else "PRIMARY",
            "transitDepthFractionalFlux":-one["coefficients"]["transit"],
            "formalTransitDepthUncertainty":formal_uncertainty,
            "eventBlockTransitDepthUncertainty":block_uncertainty,
            "consistencyTransitDepthUncertainty":max(formal_uncertainty,block_uncertainty,np.finfo(float).eps),
            "eventBlockJackknifeCount":block_count,"selectedDurationDays":sd,
            "selectedIngressFraction":si,"sampleCount":one["sampleCount"],"conditionNumber":one["condition"]})
    independent_rows=[x for x in per_sector if x["role"]=="INDEPENDENT"]
    if (set(row["sector"] for row in independent_rows) != set(independent)
            or len(independent_rows) != len(independent)):
        return _unresolved(base,["FITTED_INDEPENDENT_DIAGNOSTIC_SECTOR_BINDING_MISMATCH"])
    weights=[1/x["consistencyTransitDepthUncertainty"]**2 for x in independent_rows]
    if not weights or not math.isfinite(sum(weights)) or sum(weights)<=0:
        return _unresolved(base,["INVALID_INDEPENDENT_CONSISTENCY_WEIGHTS"])
    weighted_depth=sum(w*x["transitDepthFractionalFlux"] for w,x in zip(weights,independent_rows))/sum(weights)
    heterogeneity=sum(w*(x["transitDepthFractionalFlux"]-weighted_depth)**2 for w,x in zip(weights,independent_rows))
    reduced_heterogeneity=heterogeneity/max(len(independent_rows)-1,1)
    fractional_sector_deviation=max(abs(x["transitDepthFractionalFlux"]-weighted_depth) for x in independent_rows)/max(abs(weighted_depth),np.finfo(float).eps)
    consistency=(reduced_heterogeneity<=MAX_REDUCED_HETEROGENEITY and fractional_sector_deviation<=MAX_FRACTIONAL_SECTOR_DEVIATION)

    jackknife=[]
    for sector in independent:
        subset=[row for row in rows if row["sector"]!=sector]
        _,jd,ji,jo,jfit=_select(subset,period,epoch,established)
        jackknife.append({"omittedIndependentSector":sector,"transitDepthFractionalFlux":-jfit["coefficients"]["transit"],
            "formalTransitDepthUncertainty":jfit["uncertainties"]["transit"],"selectedDurationDays":jd,
            "selectedIngressFraction":ji,"selectedSecondaryPhaseOffset":jo,"conditionNumber":jfit["condition"]})
    jk_depths=[x["transitDepthFractionalFlux"] for x in jackknife]
    jk_unc=math.sqrt((len(jk_depths)-1)/len(jk_depths)*sum((x-np.mean(jk_depths))**2 for x in jk_depths))
    geometry_stable=all(abs(x["selectedDurationDays"]-duration)<=established*.15 and abs(x["selectedIngressFraction"]-ingress)<=.10 for x in jackknife)
    movement=max(abs(x-depth) for x in jk_depths)
    # This threshold contains neither jackknife nor sector scatter: disagreement cannot authorize itself.
    stability_scale=max(3*formal,MAX_FRACTIONAL_JACKKNIFE_MOVEMENT*abs(depth),np.finfo(float).eps)
    stability=geometry_stable and movement<=stability_scale
    sector_depths=[x["transitDepthFractionalFlux"] for x in per_sector]
    sector_scatter=float(np.std(sector_depths,ddof=1)/math.sqrt(len(sector_depths))) if len(sector_depths)>1 else 0
    conservative=max(formal,jk_unc,sector_scatter,np.finfo(float).eps)

    all_components={"transit","eclipse","fundamentalSine","fundamentalCosine","secondSine","secondCosine"}
    comparisons={}
    for label,removed in (("noTransit",{"transit"}),("noEclipse",{"eclipse"}),
                          ("noFundamentalPhase",{"fundamentalSine","fundamentalCosine"}),
                          ("noSecondHarmonic",{"secondSine","secondCosine"})):
        nested=_solve(rows,period,epoch,duration,ingress,offset,all_components-removed)
        comparisons[label]={"bic":nested["bic"],"deltaBICVersusJoint":nested["bic"]-fit["bic"]}
    comparisons["jointModelBIC"]=fit["bic"]
    primary_boundary=(duration in (established*DURATION_MULTIPLIERS[0],established*DURATION_MULTIPLIERS[-1])
                      or ingress in (INGRESS_FRACTIONS[0],INGRESS_FRACTIONS[-1]))
    secondary_boundary=offset in (SECONDARY_PHASE_OFFSETS[0],SECONDARY_PHASE_OFFSETS[-1])
    gates={"positiveFiniteTransitDepth":math.isfinite(depth) and depth>0,"adequateTransitSignificance":depth/conservative>=5,
           "atLeastThreeIndependentSupportingSectors":len(independent)>=3,"crossSectorDepthConsistency":consistency,
           "leaveOneSectorOutStable":stability,"acceptableFitConditioning":fit["condition"]<=MAX_CONDITION_NUMBER and fit["rank"]==fit["columnCount"],
           "adequatePrimaryAndBaselineCoverage":all(x["sampleCount"]>=20 for x in per_sector),
           "primaryDurationIngressNotBoundaryPinned":not primary_boundary,"finitePositiveUncertainty":math.isfinite(conservative) and conservative>0,
           "meaningfulNoTransitImprovement":comparisons["noTransit"]["deltaBICVersusJoint"]>=10}
    reasons=[]
    reason_names={"crossSectorDepthConsistency":"CROSS_SECTOR_DEPTH_INCONSISTENCY",
                  "leaveOneSectorOutStable":"LEAVE_ONE_SECTOR_OUT_INSTABILITY",
                  "primaryDurationIngressNotBoundaryPinned":"PRIMARY_GEOMETRY_BOUNDARY_PINNED"}
    for name,passed in gates.items():
        if not passed: reasons.append(reason_names.get(name,name))
    eclipse=-fit["coefficients"]["eclipse"]; eclipse_unc=fit["uncertainties"]["eclipse"]
    eclipse_resolved=(not secondary_boundary and eclipse>0 and eclipse_unc>0 and eclipse/eclipse_unc>=3
                      and comparisons["noEclipse"]["deltaBICVersusJoint"]>=6)
    fs=np.array([fit["coefficients"]["fundamentalSine"],fit["coefficients"]["fundamentalCosine"]]); fu=np.array([fit["uncertainties"]["fundamentalSine"],fit["uncertainties"]["fundamentalCosine"]])
    ss=np.array([fit["coefficients"]["secondSine"],fit["coefficients"]["secondCosine"]]); su=np.array([fit["uncertainties"]["secondSine"],fit["uncertainties"]["secondCosine"]])
    fundamental=bool(np.linalg.norm(fs)>=3*np.linalg.norm(fu) and comparisons["noFundamentalPhase"]["deltaBICVersusJoint"]>=6)
    second=bool(np.linalg.norm(ss)>=3*np.linalg.norm(su) and comparisons["noSecondHarmonic"]["deltaBICVersusJoint"]>=6)
    result={**base,"status":"COMPLETE" if not reasons else "UNRESOLVED",
      "classification":"PRECISION_EMPIRICAL_TRANSIT_DEPTH_RESOLVED" if not reasons else "PRECISION_EMPIRICAL_TRANSIT_DEPTH_UNRESOLVED",
      "precisionEmpiricalTransitDepthResolved":not reasons,"unresolvedReasons":reasons,"frozenPeriodDays":period,"frozenReferenceEpochBTJD":epoch,
      "timingSectors":[r["sector"] for r in rows],"independentSupportingSectors":independent,"independentSupportingSectorCount":len(independent),
      "modelSpecification":{"primaryTemplate":"SHARED_EXPOSURE_INTEGRATED_TRAPEZOID","oppositeConjunctionTemplate":"SHARED_EXPOSURE_INTEGRATED_TRAPEZOID",
        "phaseTerms":["ORBITAL_SINE","ORBITAL_COSINE","TWICE_ORBITAL_SINE","TWICE_ORBITAL_COSINE"],"perSectorBaseline":"INTERCEPT_PLUS_LINEAR_TIME",
        "durationMultipliers":list(DURATION_MULTIPLIERS),"ingressEgressFractionsOfHalfDuration":list(INGRESS_FRACTIONS),"secondaryPhaseOffsetGrid":list(SECONDARY_PHASE_OFFSETS),
        "exposureIntegration":"ELEVEN_DETERMINISTIC_MIDPOINT_SUBEXPOSURES_PER_FULL_CADENCE","ephemerisHeldFixed":True},
      "globalFit":{"midTransitFractionalFluxDeficit":depth,"conservativeTransitDepthUncertainty":conservative,
        "equivalentBoxTransitDepthFractionalFlux":depth*(1-ingress/2),"oppositeConjunctionEclipseDepthFractionalFlux":eclipse,
        "oppositeConjunctionEclipseUncertainty":eclipse_unc,"oppositeConjunctionEclipseStatus":"RESOLVED" if eclipse_resolved else "UNRESOLVED",
        "orbitalFrequencySineCoefficient":float(fs[0]),"orbitalFrequencyCosineCoefficient":float(fs[1]),
        "twiceOrbitalFrequencySineCoefficient":float(ss[0]),"twiceOrbitalFrequencyCosineCoefficient":float(ss[1]),
        "fundamentalPhaseCurveStatus":"RESOLVED" if fundamental else "UNRESOLVED","secondHarmonicPhaseCurveStatus":"RESOLVED" if second else "UNRESOLVED",
        "eventDurationDays":duration,"ingressEgressDurationDays":duration*ingress/2,"secondaryPhaseOffset":offset},
      "uncertaintyDiagnostics":{"formalCovarianceTransitUncertainty":formal,"independentSectorJackknifeUncertainty":jk_unc,"sectorScatterUncertainty":sector_scatter,
        "conservativeRule":"MAX_FORMAL_JACKKNIFE_SECTOR_SCATTER","gateStatisticsExcludeTestedScatter":True},
      "crossSectorConsistencyDiagnostics":{"weightedMeanDepth":weighted_depth,"chiSquare":heterogeneity,"reducedChiSquare":reduced_heterogeneity,
        "maximumFractionalDeviation":fractional_sector_deviation,"maximumReducedChiSquare":MAX_REDUCED_HETEROGENEITY,"maximumFractionalDeviationAllowed":MAX_FRACTIONAL_SECTOR_DEVIATION},
      "jackknifeStabilityDiagnostics":{"maximumDepthMovement":movement,"independentAcceptanceScale":stability_scale,"geometryStable":geometry_stable},
      "perSectorDiagnostics":per_sector,"independentSectorJackknife":jackknife,"nestedModelComparisons":comparisons,"resolutionGates":gates,
      "componentResolutionGates":{"secondaryPhaseOffsetNotBoundaryPinned":not secondary_boundary,"eclipseEvidenceIndependentOfTransit":True,"phaseEvidenceIndependentOfTransit":True},
      "fitDiagnostics":{"conditionNumber":fit["condition"],"retainedSampleCount":fit["retained"],"sampleCount":fit["sampleCount"],"eventCadencesProtectedFromClipping":True},
      "modelRanBeforeExternalKnownObjectQuery":True,"workflowNextStage":"EXTERNAL_EVIDENCE_FREEZE",
      "recommendedNextTest":"EXTERNAL_EVIDENCE_FREEZE" if not reasons else "ADDITIONAL_PRECISION_PHOTOMETRY"}
    return _finalize(result)

model_joint_transit_eclipse_phase_curve=fit_joint_event_phase_model
