"""Append-only v20.17 archival recurrence screen for a frozen TESS residual family.

Archive access and all TESS interpretation live here on the server.  The
generated project contains only generic Lomb--Scargle datasets.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

from openstar_investigation import sha256_file, sha256_json

OFFICIAL_AUTHORS = ("SPOC", "TESS-SPOC")

MAX_ARCHIVAL_BASELINE_SECTORS = 64
FREQUENCY_HALF_WIDTH_FRACTION = .20
TOTAL_FREQUENCIES = 8192
FREQUENCIES_PER_WORK_UNIT = 2048
HARMONIC_ORDERS = (1, 2)
MIN_CYCLE_COVERAGE = 1.5
MIN_ELIGIBLE_NEW_SECTORS = 3
MIN_SUPPORTING_NEW_SECTORS = 3
MIN_EPOCH_SEPARATION_DAYS = 300.0
MAX_PIXEL_FOLLOWUP_SECTORS = 6
WORKLOAD_ID = "openstar.lomb-scargle.v1"

# Explicit schemas only.  Values are paths within a stage result which contain
# either a sector object/list or a scalar primary sector.  This deliberately
# is not a recursive JSON integer search.
CONSUMED_SECTOR_SCHEMAS: dict[str, tuple[tuple[str, ...], ...]] = {
    "openstar.tess.prepare-target": (("sector",), ("source", "sector"),
                                      ("dataset", "source", "sector")),
    "openstar.tess.independent.prepare": (("preparedSectors",),),
    "openstar.tess.independent.broad.prepare": (("preparedSectors",),),
    "openstar.tess.residual-mode-localization.prepare": (("preparedSectors",), ("sector",)),
    "openstar.tess.residual-mode-localization-review.prepare": (("preparedSectors",),),
    "openstar.tess.time-resolved-frequency-localization.prepare": (("preparedSectors",),),
    "openstar.tess.time-resolved-residual-phase-localization.prepare": (("preparedSectors",),),
    "openstar.tess.multi-source-residual.prepare": (("preparedSeries",), ("preparedSectors",)),
}


def _at(value: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(value, dict): return None
        value = value.get(key)
    return value


def _valid_sector(value: Any) -> int | None:
    if isinstance(value, bool): return None
    try: sector = int(value)
    except (TypeError, ValueError): return None
    return sector if sector > 0 and str(value).strip() in {str(sector), f"{sector}.0"} else None


def previously_consumed_tess_sectors(stages: Iterable[Any]) -> dict[int, list[dict[str, str]]]:
    """Return durable exclusions with exact stage/schema reasons."""
    consumed: dict[int, list[dict[str, str]]] = {}
    for stage in stages:
        handler = getattr(stage, "handler_id", None) or (stage.get("handler_id") if isinstance(stage, dict) else None)
        status = getattr(stage, "status", None) or (stage.get("status") if isinstance(stage, dict) else None)
        result = getattr(stage, "result", None) if not isinstance(stage, dict) else stage.get("result")
        stage_id = getattr(stage, "id", None) or (stage.get("id") if isinstance(stage, dict) else None)
        if status != "COMPLETE" or handler not in CONSUMED_SECTOR_SCHEMAS or not isinstance(result, dict): continue
        for path in CONSUMED_SECTOR_SCHEMAS[handler]:
            value = _at(result, path)
            values = value if isinstance(value, list) else [value]
            for item in values:
                raw = item.get("sector") if isinstance(item, dict) else item
                sector = _valid_sector(raw)
                if sector is not None:
                    reason = {"stageID": str(stage_id), "handlerID": handler,
                              "schemaPath": ".".join(path),
                              "reason": "TESS data previously materialized as scientific evidence"}
                    if reason not in consumed.setdefault(sector, []): consumed[sector].append(reason)
    return dict(sorted(consumed.items()))


def frozen_search_grid(reference_frequency: float) -> dict[str, Any]:
    if not math.isfinite(reference_frequency) or reference_frequency <= 0: raise ValueError("invalid frozen residual frequency")
    low = reference_frequency * (1-FREQUENCY_HALF_WIDTH_FRACTION)
    high = reference_frequency * (1+FREQUENCY_HALF_WIDTH_FRACTION)
    return {"minimumFrequency": low, "maximumFrequency": high,
            "frequencyStep": (high-low)/(TOTAL_FREQUENCIES-1),
            "totalFrequencies": TOTAL_FREQUENCIES,
            "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT}


def prewhiten_established_family(times: Iterable[float], flux: Iterable[float],
                                 established_frequency: float) -> tuple[Any, dict[str, Any]]:
    """Fit only sector-local coefficients at the frozen physical frequency."""
    import numpy as np
    t = np.asarray(list(times), dtype=np.float64); y = np.asarray(list(flux), dtype=np.float64)
    if len(t) != len(y) or len(t) < 32 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)): raise ValueError("invalid prewhitening samples")
    if not math.isfinite(established_frequency) or established_frequency <= 0: raise ValueError("invalid frozen established frequency")
    local = t-t[0]; trend = local-float(np.mean(local))
    columns = [np.ones(len(t)), trend]
    for order in HARMONIC_ORDERS:
        angle = 2*np.pi*established_frequency*order*local
        columns.extend((np.sin(angle), np.cos(angle)))
    design = np.column_stack(columns)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y-design@beta; residual -= np.mean(residual)
    std = float(np.std(residual))
    if not math.isfinite(std) or std <= 0: raise ValueError("invalid prewhitened residual variance")
    residual /= std
    return np.asarray(residual, dtype=np.float32), {
        "frozenEstablishedPhysicalFrequency": float(established_frequency),
        "frozenEstablishedPhysicalPeriodDays": 1/float(established_frequency),
        "harmonicOrders": list(HARMONIC_ORDERS), "linearTrendIncluded": True,
        "fitCoefficients": [float(x) for x in beta], "prewhitenedResidualStddev": std,
        "historicalResidualDriftExtrapolated": False}


def adjudicate_sector(candidate: dict[str, Any], envelope: tuple[float, float]) -> dict[str, Any]:
    low, high = map(float, envelope); result = copy.deepcopy(candidate)
    f = candidate.get("candidateFrequency"); ci = candidate.get("candidateFrequencyConfidenceInterval")
    if isinstance(ci, dict): ci = (ci.get("lower"), ci.get("upper"))
    baseline = float(candidate.get("baselineDays") or 0); rayleigh = 1/baseline if baseline > 0 else None
    valid_ci = isinstance(ci, (list, tuple)) and len(ci)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in ci) and ci[0] <= ci[1]
    width = ci[1]-ci[0] if valid_ci else None
    cycles = float(candidate.get("cycleCoverage") or (float(f)*baseline if isinstance(f,(int,float)) else 0))
    reliable = candidate.get("periodStatus") == "RELIABLE" and str(candidate.get("periodConfidence", "")).lower() in {"high","medium"}
    boundary = bool(candidate.get("boundaryHit")); inside = isinstance(f,(int,float)) and low <= f <= high
    overlap = valid_ci and ci[1] >= low and ci[0] <= high
    resolved = valid_ci and rayleigh is not None and width <= rayleigh
    supports = bool(reliable and cycles >= MIN_CYCLE_COVERAGE and not boundary and resolved and inside and overlap)
    if supports: classification = "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY"
    elif valid_ci and not resolved: classification = "RESOLUTION_LIMITED"
    elif reliable and not boundary and isinstance(f,(int,float)) and not inside: classification = "INTERIOR_RESIDUAL_BAND_PEAK_OUTSIDE_HISTORICAL_ENVELOPE"
    else: classification = "NONSUPPORTING"
    result.update({"candidateFrequency": f, "candidatePeriodDays": 1/f if isinstance(f,(int,float)) and f>0 else None,
        "candidateFrequencyConfidenceInterval": list(ci) if valid_ci else None,
        "rayleighFrequencyResolution": rayleigh, "boundaryHit": boundary, "cycleCoverage": cycles,
        "eligibleForResidualRecurrence": True, "historicalFrequencyEnvelope": {"minimum":low,"maximum":high},
        "candidateInsideHistoricalEnvelope": inside, "confidenceIntervalOverlapsHistoricalEnvelope": bool(overlap),
        "frequencyIntervalResolved": bool(resolved), "supportsHistoricalResidualFamily": supports,
        "recurrenceClassification": classification})
    return result


def _followups(supporting: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = sorted(supporting, key=lambda x:(float(x["originalTimeOriginDays"]), int(x["sector"])))
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < MAX_PIXEL_FOLLOWUP_SECTORS:
        if not selected: pick = remaining[0]; why = "earliest supporting observation epoch"
        elif len(selected)==1: pick = remaining[-1]; why = "latest supporting observation epoch"
        else:
            pick = max(remaining, key=lambda x:(min(abs(float(x["originalTimeOriginDays"])-float(s["originalTimeOriginDays"])) for s in selected), -int(x["sector"])))
            why = "deterministic farthest-point temporal spacing"
        remaining.remove(pick); selected.append({"sector":int(pick["sector"]), "originalTimeOriginDays":float(pick["originalTimeOriginDays"]), "selectionReason":why})
    return selected


def adjudicate_target(sectors: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows=list(sectors); ids=[_valid_sector(x.get("sector")) for x in rows]
    if None in ids or len(ids)!=len(set(ids)): raise ValueError("sector IDs must be positive and unique")
    eligible=[x for x in rows if x.get("eligibleForResidualRecurrence") is True]
    support=[x for x in eligible if x.get("supportsHistoricalResidualFamily") is True]
    span=max((float(x["originalTimeOriginDays"]) for x in support),default=0)-min((float(x["originalTimeOriginDays"]) for x in support),default=0) if support else 0
    if len(eligible)<MIN_ELIGIBLE_NEW_SECTORS: classification="ARCHIVAL_TARGET_RESIDUAL_BASELINE_INSUFFICIENT"
    elif len(support)>=MIN_SUPPORTING_NEW_SECTORS and len(support)>len(eligible)/2 and span>=MIN_EPOCH_SEPARATION_DAYS: classification="ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED"
    elif len(support)>=2: classification="ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUGGESTIVE"
    else: classification="ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_NOT_ESTABLISHED"
    recommendation = ("PIXEL_LEVEL_SOURCE_RESOLVED_RESIDUAL_RECURRENCE_VALIDATION" if classification.endswith(("SUPPORTED","SUGGESTIVE")) else
        "FFI_ONLY_RESIDUAL_BASELINE_EXTENSION" if classification.endswith("INSUFFICIENT") else "EXTERNAL_LONG_BASELINE_OR_FFI_RESIDUAL_VALIDATION")
    return {"classification":classification,"recommendedNextTest":recommendation,"eligibleSectorCount":len(eligible),"supportingSectorCount":len(support),
        "resolutionLimitedSectorCount":sum(x.get("recurrenceClassification")=="RESOLUTION_LIMITED" for x in eligible),
        "nonSupportingSectorCount":sum(not x.get("supportsHistoricalResidualFamily") for x in eligible),"supportingTemporalSpanDays":span,
        "selectedFuturePixelFollowupSectors":_followups(support),"sectorEvidence":rows,"physicalMechanismResolved":False,"sourceAttributionResolved":False,
        "crossSectorPhaseUsed":False,"historicalResidualDriftExtrapolated":False}


def verified_json_result(stage: Any, filenames: str|tuple[str,...]) -> tuple[str,dict[str,Any]]:
    names=(filenames,) if isinstance(filenames,str) else filenames
    for ref in stage.artifacts:
        path=Path(ref.path); expected=str(ref.sha256)
        if path.name in names and expected and path.is_file() and sha256_file(path)==expected:
            frozen=json.loads(path.read_text(encoding="utf-8"))
            if frozen != stage.result: raise RuntimeError("persisted result and frozen artifact JSON differ")
            return expected,frozen
    raise RuntimeError(f"artifact SHA verification failed for {names[0]}")


def build_archival_baseline_project(*, source_project_path: str|Path,
        source_dataset_entry: dict[str,Any], tic_id: int, candidate_sectors: Iterable[int]|None,
        previously_consumed: dict[int,list[dict[str,str]]], residual_reference_frequency: float,
        established_frequency: float, historical_frequency_envelope: tuple[float,float],
        output_dir: str|Path, investigation_id: str) -> dict[str,Any]:
    """Query once and materialize every unseen official sector, without a science cap."""
    from .tess_multisector import (TessArchiveInfrastructureError,
        _MAST_LIGHTKURVE_LOCK, _archive_io_failure, _download_selected_sector,
        _prepare_samples, _safe, _search_lightcurves, _select_product_from_search)
    grid=frozen_search_grid(residual_reference_frequency)  # freeze before archive I/O
    sectors=sorted({_valid_sector(x) for x in candidate_sectors}) if candidate_sectors is not None else []
    if None in sectors: raise ValueError("invalid archival sector ID")
    source_project=json.loads(Path(source_project_path).read_text(encoding="utf-8"))
    materialized=[]; errors=[]
    try:
        with _MAST_LIGHTKURVE_LOCK:
            search=_search_lightcurves(tic_id)
            if candidate_sectors is None:
                from .tess_multisector import _sector_from_search_row
                table=getattr(search,"table",None); colnames=set(getattr(table,"colnames",[]))
                sectors=sorted({_sector_from_search_row(table,index) for index in range(0 if table is None else len(table))
                    if "author" in colnames and str(table["author"][index]).strip().upper() in OFFICIAL_AUTHORS})
                sectors=[x for x in sectors if x is not None]
            eligible=[x for x in sectors if x not in previously_consumed]
            if len(eligible)>MAX_ARCHIVAL_BASELINE_SECTORS:
                raise RuntimeError("eligible official sectors exceed MAX_ARCHIVAL_BASELINE_SECTORS; refusing to truncate")
            for sector in eligible:
                try:
                    selected,author,cadence=_select_product_from_search(search,sector)
                    lc,_=_download_selected_sector(selected,tic_id=tic_id,sector=sector,author=author,cadence_seconds=cadence)
                    # Reuse the established immutable finite/sort/downsample path.
                    times,normalized,prep=_prepare_samples(lc)
                    residual,prewhite=prewhiten_established_family(times,normalized,established_frequency)
                    materialized.append((sector,author,cadence,times,residual,prep,prewhite))
                except Exception as error:
                    diagnostic={"sector":sector,"operation":"archive-materialization","error":f"{type(error).__name__}: {error}"}
                    if _archive_io_failure(error):
                        raise TessArchiveInfrastructureError("TESS archive materialization is temporarily unavailable",{"errors":[diagnostic]}) from error
                    errors.append(diagnostic)
    except TessArchiveInfrastructureError: raise
    except Exception as error:
        if _archive_io_failure(error):
            raise TessArchiveInfrastructureError("TESS archive search is temporarily unavailable",{"errors":[{"sector":None,"operation":"archive-search","error":f"{type(error).__name__}: {error}"}]}) from error
        raise
    root=Path(output_dir)/"target-residual-archival-baseline"; entries=[]; prepared=[]
    for sector,author,cadence,times,residual,prep,prewhite in materialized:
        dataset_id=f"{source_dataset_entry['id']}-sector-{sector}-archival-residual-v1"
        dataset={"id":dataset_id,"targetName":f"TIC {tic_id} archival residual Sector {sector}","timeUnit":"days",
            "timeReference":"relative-to-first-distributed-sample","numericRepresentation":"Float32","fluxUnit":"normalized-prewhitened-residual",
            "times":[float(x) for x in times],"flux":[float(x) for x in residual],"frequencySearch":dict(grid),"reference":{}}
        path=root/f"{_safe(dataset_id)}.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(dataset,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        entry=copy.deepcopy(source_dataset_entry); entry.update({"id":dataset_id,"path":str(path.resolve()),"ticID":int(tic_id),"sector":sector,"author":author,"cadenceSeconds":cadence,"role":"archival-target-residual-baseline"}); entries.append(entry)
        prepared.append({"sector":sector,"datasetID":dataset_id,"datasetPath":str(path.resolve()),"author":author,"cadenceSeconds":cadence,
            "originalSamples":prep["originalSamples"],"finiteSampleCount":prep["originalSamples"],"distributedSamples":prep["distributedSamples"],
            "originalTimeOriginDays":prep["originalTimeOriginDays"],"baselineDays":prep["baselineDays"],"priorSectorExclusionStatus":"UNSEEN_ADMITTED",
            "productSelectionRule":"official-author-priority-SPOC-then-TESS-SPOC; shortest-cadence; catalog-order",
            "prewhitening":prewhite,"frozenResidualReferenceFrequency":residual_reference_frequency,"frequencySearch":dict(grid)})
    project_id=f"{source_project['id']}.investigation.{_safe(investigation_id)}.archival-residual-v1"
    manifest={"id":project_id,"name":"Archival target-residual recurrence screen","workloadID":WORKLOAD_ID,"datasets":entries,
        "investigation":{"purpose":"frozen-target-residual-archival-recurrence","sourceProjectID":source_project["id"]}}
    manifest_path=root/f"{_safe(project_id)}.json"; manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return {"available":bool(entries),"projectID":project_id,"projectPath":str(manifest_path.resolve()),"preparedSectors":prepared,"errors":errors,
        "previouslyConsumedTessSectors":[{"sector":sector,"exclusionReasons":reasons} for sector,reasons in sorted(previously_consumed.items())],
        "candidateSectors":sectors,"eligibleSectors":eligible,"frequencySearch":grid,"frozenResidualReferenceFrequency":residual_reference_frequency,
        "frozenResidualReferencePeriodDays":1/residual_reference_frequency,"historicalFrequencyEnvelope":{"minimum":historical_frequency_envelope[0],"maximum":historical_frequency_envelope[1]},
        "frozenEstablishedPhysicalFrequency":established_frequency,"frozenEstablishedPhysicalPeriodDays":1/established_frequency,
        "crossSectorPhaseUsed":False,"historicalResidualDriftExtrapolated":False,"totalWorkUnits":len(entries)*math.ceil(TOTAL_FREQUENCIES/FREQUENCIES_PER_WORK_UNIT)}
