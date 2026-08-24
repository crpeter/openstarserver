"""v20.18 source localization of the v20.17 archival residual recurrence.

This module deliberately consumes frozen sector frequencies.  It contains no
period search, cross-sector ephemeris, drift correction, or worker workload.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Iterable

from openstar_path_relocation import HistoricalPathResolver, NO_HISTORICAL_PATH_RELOCATION
from .tess_target_residual_archival_baseline import (MAX_PIXEL_FOLLOWUP_SECTORS,
    verify_frozen_science_lineage, verified_json_result)
from .tess_offset_source import (CATALOG_MERGE_RADIUS_ARCSEC, GAIA_QUERY_RADIUS_ARCSEC,
    TIC_QUERY_RADIUS_ARCSEC, _merge_catalog_candidates)
from .tess_difference_image_constants import (MIN_IMAGE_PEAK_SNR,
    SOURCE_MATCH_MAX_PIXELS, SOURCE_MARGIN_FLOOR_PIXELS)

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
    hashes=science.provenance.input_hashes if science.provenance else {}
    from openstar_investigation import sha256_json
    if (hashes.get("preparation") != sha256_json(prepare.result)
            or hashes.get("distributedResult") != sha256_json(run.result)):
        raise RuntimeError("v20.17 interpretation is not bound to preparation and run")
    if result.get("classification") not in {"ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED", "ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUGGESTIVE"}:
        raise RuntimeError("v20.17 recurrence does not admit localization")
    required={"recommendedNextTest":RECOMMENDATION,"sourceAttributionResolved":False,
        "physicalMechanismResolved":False,"crossSectorPhaseUsed":False,
        "historicalResidualDriftExtrapolated":False}
    if any(result.get(k) != v for k,v in required.items()): raise RuntimeError("altered v20.17 science boundary")
    if not isinstance(final.result,dict): raise RuntimeError("malformed v20.17 finalizer result")
    if (final.parameters != {"outputSuffix":"v20.17-target-residual-archival-baseline"}
            or final.result.get("targetResidualArchivalBaselineExtension") != result
            or final.result.get("recommendedNextTest") != RECOMMENDATION):
        raise RuntimeError("altered v20.17 finalizer")
    selected=result.get("selectedFuturePixelFollowupSectors") or []
    if not 0 < len(selected) <= MAX_PIXEL_FOLLOWUP_SECTORS: raise RuntimeError("invalid selected sector list")
    selected_ids=[int(x.get("sector") if isinstance(x,dict) else x) for x in selected]
    if len(selected_ids) != len(set(selected_ids)): raise RuntimeError("selected sector IDs must be unique")
    evidence=result.get("sectorEvidence") or []; by_sector={}
    for item in evidence: by_sector.setdefault(item.get("sector"),[]).append(item)
    frozen=[]
    for raw in selected:
        sector=int(raw.get("sector") if isinstance(raw,dict) else raw)
        if len(by_sector.get(sector,[])) != 1: raise RuntimeError("selected sector must exist exactly once")
        item=by_sector[sector][0]
        if item.get("supportsHistoricalResidualFamily") is not True or item.get("recurrenceClassification") != "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY":
            raise RuntimeError("selected sector is not supporting")
        frequency=item.get("candidateFrequency")
        if not isinstance(frequency,(int,float)) or not math.isfinite(float(frequency)) or float(frequency)<=0:
            raise RuntimeError("selected sector lacks valid frozen frequency")
        selection = raw if isinstance(raw,dict) else next((x for x in result.get("selectedFuturePixelFollowupSectorEvidence",[]) if x.get("sector")==sector), {})
        if (selection.get("originalTimeOriginDays") != item.get("originalTimeOriginDays")
                or not isinstance(selection.get("selectionReason"),str)
                or not selection.get("selectionReason").strip()):
            raise RuntimeError("selected-sector metadata disagrees with sector evidence")
        frozen.append({"sector":sector,"candidateFrequency":frequency,
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
        "supportByCatalogSource":{key:value for key,value in counts.items() if key != target_source_id},"ambiguousSectors":[x["sector"] for x in sectors if x.get("classification")=="AMBIGUOUS_OR_BLENDED"],
        "unavailableSectors":[x["sector"] for x in sectors if x.get("classification")=="UNAVAILABLE"],"preferredSource":winner if resolved else None,
        "sourceAttributionResolved":resolved,"physicalMechanismResolved":False,"crossSectorPhaseUsed":False,
        "historicalResidualDriftExtrapolated":False,"recommendedNextTest":recommendation,"sectorResults":sectors}


class CatalogInfrastructureError(RuntimeError):
    def __init__(self,message,diagnostics=None):
        super().__init__(message); self.diagnostics=diagnostics or {}
class NoPixelCoverageError(RuntimeError): pass


def freeze_catalog_hypotheses(*, tic_id:int, ra_deg:float, dec_deg:float,
        query_tic=None, query_gaia=None, coordinate_factory=None) -> dict[str,Any]:
    """Query the preregistered regions and freeze the complete merged catalog."""
    from .tess_offset_source import _query_tic_region, _query_gaia_region, _skycoord
    try:
        coordinate=(coordinate_factory or _skycoord)(ra_deg,dec_deg)
        tic=(query_tic or _query_tic_region)(coordinate,int(tic_id))
        gaia=(query_gaia or _query_gaia_region)(coordinate)
    except Exception as error:
        raise CatalogInfrastructureError(f"catalog query failed: {type(error).__name__}: {error}") from error
    errors=[{"catalog":name,"error":response.get("queryError")} for name,response in
            (("TIC",tic),("GaiaDR3",gaia)) if response.get("queryError")]
    if errors:
        raise CatalogInfrastructureError(str(errors),{"catalogQueries":{"tic":tic,"gaiaDR3":gaia},"queryErrors":errors})
    target_sky={"raDeg":float(ra_deg),"decDeg":float(dec_deg)}
    candidates=_merge_catalog_candidates(tic_sources=list(tic.get("sources") or []),
        gaia_sources=list(gaia.get("sources") or []),target_sky=target_sky,
        exclude_target_neighborhood=False)
    candidates.sort(key=lambda item:(float(item.get("targetSeparationArcsec") or 0.0),
        str((item.get("catalogIDs") or {}).get("ticID") or ""),
        str((item.get("catalogIDs") or {}).get("gaiaDR3SourceID") or ""),
        float(item["raDeg"]),float(item["decDeg"])))
    target_tic=next((x for x in tic.get("sources") or [] if x.get("isTargetTIC") or x.get("ticID")==tic_id),{})
    target_gaia=target_tic.get("gaiaSourceID")
    hypotheses=[{"sourceID":f"TIC-{tic_id}","isTarget":True,"ticID":tic_id,
        "gaiaDR3SourceID":target_gaia,"raDeg":float(ra_deg),"decDeg":float(dec_deg)}]
    for item in candidates:
        ids=item.get("catalogIDs") or {}; tic_value=ids.get("ticID"); gaia_value=ids.get("gaiaDR3SourceID")
        source_id=f"TIC-{tic_value}" if tic_value is not None else f"GaiaDR3-{gaia_value}"
        hypotheses.append({"sourceID":source_id,"isTarget":False,"ticID":tic_value,
            "gaiaDR3SourceID":gaia_value,"raDeg":item["raDeg"],"decDeg":item["decDeg"]})
    return {"catalogHypotheses":hypotheses,"catalogQueries":{"tic":tic,"gaiaDR3":gaia},
        "queryProvenance":{"TIC":{"service":"MAST Catalogs","catalog":"TIC","radiusArcsec":TIC_QUERY_RADIUS_ARCSEC,"center":target_sky},
        "GaiaDR3":{"service":"VizieR","catalog":"I/355/gaiadr3","radiusArcsec":GAIA_QUERY_RADIUS_ARCSEC,"center":target_sky},
        "catalogMergeRadiusArcsec":CATALOG_MERGE_RADIUS_ARCSEC,"responsesPersistedVerbatim":True}}


_NO_COVERAGE = re.compile(r"^No official TPF or TESScut coverage available for Sector [1-9][0-9]*\.$")
def acquire_selected_sector(download, **kwargs):
    """Narrowly translate only the established no-coverage downloader result."""
    try: return download(**kwargs)
    except RuntimeError as error:
        if type(error) is RuntimeError and _NO_COVERAGE.fullmatch(str(error)):
            raise NoPixelCoverageError(str(error)) from error
        raise


def tpf_flux_cube(tpf):
    """Convert TPF flux without converting masked samples into measurements."""
    import numpy as np
    flux=getattr(tpf.flux,"value",tpf.flux)
    if np.ma.isMaskedArray(flux): flux=np.ma.filled(flux,np.nan)
    return np.asarray(flux,dtype=np.float64)
