"""v20.18 source localization of the v20.17 archival residual recurrence.

This module deliberately consumes frozen sector frequencies.  It contains no
period search, cross-sector ephemeris, drift correction, or worker workload.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Iterable

from openstar_path_relocation import HistoricalPathResolver, NO_HISTORICAL_PATH_RELOCATION
from .tess_target_residual_archival_baseline import (MAX_PIXEL_FOLLOWUP_SECTORS,
    verify_frozen_science_lineage, verified_json_result)

# Public established thresholds mirrored here so lineage/admission remains
# usable on installations without optional numerical astronomy packages.
MIN_IMAGE_PEAK_SNR = 4.0
SOURCE_MATCH_MAX_PIXELS = 1.10
SOURCE_MARGIN_FLOOR_PIXELS = 0.30

PREFIX = "openstar.tess.target-residual-pixel-recurrence."
RECOMMENDATION = "PIXEL_LEVEL_SOURCE_RESOLVED_RESIDUAL_RECURRENCE_VALIDATION"


def verify_v2017_lineage(stages: Iterable[Any], *, resolver: HistoricalPathResolver | None = None) -> dict[str, Any]:
    """Verify the exact immutable 032--035 continuation boundary."""
    rows = list(stages); resolver = resolver or NO_HISTORICAL_PATH_RELOCATION
    expected = (
        ("032-target-residual-archival-baseline-prepare", "openstar.tess.target-residual-archival-baseline.prepare"),
        ("033-target-residual-archival-baseline-run", "openstar.tess.target-residual-archival-baseline.run"),
        ("034-target-residual-archival-baseline-interpret", "openstar.tess.target-residual-archival-baseline.interpret"),
        ("035-finalize", "openstar.tess.finalize"))
    boundary_index=next((i for i,s in enumerate(rows) if s.id==expected[0][0]),None)
    if boundary_index is None: raise RuntimeError("missing v20.17 boundary")
    predecessor=verify_frozen_science_lineage(rows[:boundary_index], resolver=resolver)
    found=[]
    for stage_id, handler in expected:
        matches=[s for s in rows if s.id == stage_id]
        if len(matches) != 1 or matches[0].handler_id != handler or matches[0].status != "COMPLETE":
            raise RuntimeError(f"invalid v20.17 stage {stage_id}")
        found.append(matches[0])
    if any(found[i].triggered_by_stage_id != found[i-1].id for i in range(1,4)):
        raise RuntimeError("invalid v20.17 trigger chain")
    prepare, run, science, final = found
    verified_json_result(prepare, "target-residual-archival-baseline-prepare-v20.17.json", resolver=resolver)
    verified_json_result(science, "target-residual-archival-baseline-v20.17.json", resolver=resolver)
    verified_json_result(final, "conclusion-v20.17-target-residual-archival-baseline.json", resolver=resolver)
    result=science.result
    if result.get("classification") not in {"ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED", "ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUGGESTIVE"}:
        raise RuntimeError("v20.17 recurrence does not admit localization")
    required={"recommendedNextTest":RECOMMENDATION,"sourceAttributionResolved":False,
        "physicalMechanismResolved":False,"crossSectorPhaseUsed":False,
        "historicalResidualDriftExtrapolated":False}
    if any(result.get(k) != v for k,v in required.items()): raise RuntimeError("altered v20.17 science boundary")
    if final.parameters != {"outputSuffix":"v20.17-target-residual-archival-baseline"} or final.result.get("targetResidualArchivalBaselineExtension") != result:
        raise RuntimeError("altered v20.17 finalizer")
    selected=result.get("selectedFuturePixelFollowupSectors") or []
    if not 0 < len(selected) <= MAX_PIXEL_FOLLOWUP_SECTORS: raise RuntimeError("invalid selected sector list")
    evidence=result.get("sectorEvidence") or []; by_sector={}
    for item in evidence: by_sector.setdefault(item.get("sector"),[]).append(item)
    frozen=[]
    for raw in selected:
        sector=int(raw.get("sector") if isinstance(raw,dict) else raw)
        if len(by_sector.get(sector,[])) != 1: raise RuntimeError("selected sector must exist exactly once")
        item=by_sector[sector][0]
        if item.get("supportsHistoricalResidualFamily") is not True or item.get("recurrenceClassification") != "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY":
            raise RuntimeError("selected sector is not supporting")
        if item.get("candidateFrequency") is None: raise RuntimeError("selected sector lacks frozen frequency")
        selection = raw if isinstance(raw,dict) else next((x for x in result.get("selectedFuturePixelFollowupSectorEvidence",[]) if x.get("sector")==sector), {})
        frozen.append({"sector":sector,"candidateFrequency":item["candidateFrequency"],
            "candidateFrequencyConfidenceInterval":copy.deepcopy(item.get("candidateFrequencyConfidenceInterval")),
            "originalTimeOriginDays":item.get("originalTimeOriginDays"),
            "selectionReason":selection.get("selectionReason") or item.get("selectionReason")})
    return {"prepare":prepare,"run":run,"science":science,"finalizer":final,
            "frozenScienceLineage":predecessor,"selectedSectorEvidence":frozen}


def classify_centroid(centroid: tuple[float,float], hypotheses: list[dict[str,Any]], uncertainty: float, peak_snr: float) -> dict[str,Any]:
    distances={str(h["sourceID"]):math.hypot(centroid[0]-float(h["x"]),centroid[1]-float(h["y"])) for h in hypotheses}
    if peak_snr < MIN_IMAGE_PEAK_SNR or not distances:
        return {"classification":"UNRESOLVED","preferredSource":None,"distancesPixels":distances}
    ordered=sorted(distances,key=distances.get); close=[x for x in ordered if distances[x] <= SOURCE_MATCH_MAX_PIXELS]
    margin=max(SOURCE_MARGIN_FLOOR_PIXELS,2*float(uncertainty))
    unique=len(close)==1 and (len(ordered)==1 or distances[ordered[1]]-distances[ordered[0]] >= margin)
    return {"classification":"UNIQUE_SOURCE_SUPPORTED" if unique else "AMBIGUOUS_OR_BLENDED",
        "preferredSource":close[0] if unique else None,"distancesPixels":distances,"requiredMarginPixels":margin}


def measure_sector(times, cube, valid, *, established_frequency: float, candidate_frequency: float,
                   hypotheses: list[dict[str,Any]]) -> dict[str,Any]:
    """Prewhiten physical harmonics and localize one frozen sector clock."""
    import numpy as np
    from .tess_difference_image import (_centroid_from_frames, _extreme_indices,
        _jackknife_uncertainty, _phase_model)
    from .tess_multisource_residual import _prewhiten_cube_raw
    times=np.asarray(times,float); cube=np.asarray(cube,float); valid=np.asarray(valid,bool)
    residual, fitted_valid=_prewhiten_cube_raw(absolute_times=times,cube=cube,
        physical_frequency=float(established_frequency),harmonic_orders=(1,2))
    valid=valid & fitted_valid
    aperture=np.nansum(residual[:,valid],axis=1)
    phase=_phase_model(times-times[0],aperture,float(candidate_frequency)); high,low=_extreme_indices(phase["model"])
    image=_centroid_from_frames(residual,valid,high,low); uncertainty,jackknife=_jackknife_uncertainty(residual,valid,high,low)
    attribution=classify_centroid((image["centroidX"],image["centroidY"]),hypotheses,uncertainty,image["peakSNR"])
    return {"candidateFrequencyUsed":float(candidate_frequency),"establishedFamilyPrewhitening":{"frequency":float(established_frequency),"harmonicOrders":[1,2],"sectorLocalIntercept":True,"sectorLocalTrend":True},
        "highCadenceCount":len(high),"lowCadenceCount":len(low),"differenceImage":image.get("differenceImage"),"snrImage":image.get("snrImage"),
        "peakSNR":image["peakSNR"],"centroidX":image["centroidX"],"centroidY":image["centroidY"],
        "centroidUncertaintyPixels":uncertainty,"jackknifeCentroids":jackknife,**attribution,
        "crossSectorPhaseUsed":False,"historicalResidualDriftExtrapolated":False}


def interpret_sectors(sectors: list[dict[str,Any]], target_source_id: str) -> dict[str,Any]:
    quality=[x for x in sectors if x.get("classification") in {"UNIQUE_SOURCE_SUPPORTED","AMBIGUOUS_OR_BLENDED"}]
    counts={}
    for x in quality:
        if x.get("classification")=="UNIQUE_SOURCE_SUPPORTED": counts[x["preferredSource"]]=counts.get(x["preferredSource"],0)+1
    repeated=[k for k,v in counts.items() if v>=2]
    winner=max(counts,key=counts.get) if counts else None
    resolved=bool(winner and counts[winner]>=3 and counts[winner] > sum(counts.values())/2 and len(repeated)<2)
    classification=("PIXEL_RECURRENCE_TARGET_SUPPORTED" if resolved and winner==target_source_id else
        "PIXEL_RECURRENCE_CATALOG_SOURCE_SUPPORTED" if resolved else
        "PIXEL_RECURRENCE_SOURCE_SWITCHING_OR_BLEND" if len(repeated)>=2 else "PIXEL_RECURRENCE_LOCALIZATION_UNRESOLVED")
    recommendation=("ARCHIVAL_RECURRENCE_INFORMED_TARGET_MECHANISM_MODELING" if classification.endswith("TARGET_SUPPORTED") else
        "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION" if classification.endswith("CATALOG_SOURCE_SUPPORTED") else
        "SOURCE_SWITCHING_TEMPORAL_MODEL" if "SWITCHING" in classification else "ADDITIONAL_SOURCE_LOCALIZATION_DATA")
    return {"classification":classification,"qualitySectorCount":len(quality),"targetSupportingSectors":[x["sector"] for x in quality if x.get("preferredSource")==target_source_id],
        "supportByCatalogSource":counts,"ambiguousSectors":[x["sector"] for x in sectors if x.get("classification")=="AMBIGUOUS_OR_BLENDED"],
        "unavailableSectors":[x["sector"] for x in sectors if x.get("classification")=="UNAVAILABLE"],"preferredSource":winner if resolved else None,
        "sourceAttributionResolved":resolved,"physicalMechanismResolved":False,"crossSectorPhaseUsed":False,
        "historicalResidualDriftExtrapolated":False,"recommendedNextTest":recommendation,"sectorResults":sectors}


def frozen_catalog_hypotheses(identity: dict[str,Any], *, tic_id: int, ra_deg: float, dec_deg: float) -> list[dict[str,Any]]:
    """Freeze all already-query-bounded TIC/Gaia records, with target explicit."""
    result=[{"sourceID":f"TIC-{tic_id}","isTarget":True,"ticID":tic_id,
             "gaiaDR3SourceID":None,"raDeg":float(ra_deg),"decDeg":float(dec_deg)}]
    seen={(round(float(ra_deg),9),round(float(dec_deg),9))}
    def visit(value):
        if isinstance(value,dict):
            ra=value.get("raDeg"); dec=value.get("decDeg")
            if isinstance(ra,(int,float)) and isinstance(dec,(int,float)):
                key=(round(float(ra),9),round(float(dec),9))
                if key not in seen:
                    tic=value.get("ticID"); gaia=value.get("gaiaDR3SourceID",value.get("gaiaSourceID"))
                    source=f"TIC-{tic}" if tic is not None else f"GaiaDR3-{gaia}" if gaia is not None else f"sky-{ra:.9f}-{dec:.9f}"
                    result.append({"sourceID":source,"isTarget":False,"ticID":tic,"gaiaDR3SourceID":gaia,
                                   "raDeg":float(ra),"decDeg":float(dec)}); seen.add(key)
            for child in value.values(): visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    visit(identity)
    return result
