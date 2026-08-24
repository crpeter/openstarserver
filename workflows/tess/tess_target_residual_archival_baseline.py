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


def _product_provenance(selected: Any, *, author: str,
                        cadence_seconds: float | None) -> dict[str, Any]:
    table = getattr(selected, "table", None)
    columns = set(getattr(table, "colnames", []))
    def value(*names: str) -> Any:
        for name in names:
            if name in columns and len(table):
                item = table[name][0]
                return item.item() if hasattr(item, "item") else str(item)
        return None
    return {"mission": value("mission") or "TESS", "author": author,
        "cadenceSeconds": cadence_seconds,
        "observationID": value("obs_id", "obsID", "observation_id"),
        "productFilename": value("productFilename", "productFilename", "dataURI"),
        "productURI": value("dataURI", "data_uri"),
        "selectionRule": "official-author-priority-SPOC-then-TESS-SPOC; shortest-cadence; catalog-order"}


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
    coverage_ok = cycles >= MIN_CYCLE_COVERAGE
    eligible = coverage_ok and not boundary
    overlap = valid_ci and ci[1] >= low and ci[0] <= high
    resolved = valid_ci and rayleigh is not None and width <= rayleigh
    supports = bool(eligible and reliable and resolved and inside and overlap)
    if not eligible: classification = "INELIGIBLE"
    elif supports: classification = "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY"
    elif valid_ci and not resolved: classification = "RESOLUTION_LIMITED"
    elif reliable and not boundary and isinstance(f,(int,float)) and not inside: classification = "INTERIOR_RESIDUAL_BAND_PEAK_OUTSIDE_HISTORICAL_ENVELOPE"
    else: classification = "NONSUPPORTING"
    result.update({"candidateFrequency": f, "candidatePeriodDays": 1/f if isinstance(f,(int,float)) and f>0 else None,
        "candidateFrequencyConfidenceInterval": list(ci) if valid_ci else None,
        "rayleighFrequencyResolution": rayleigh, "boundaryHit": boundary, "cycleCoverage": cycles,
        "eligibleForResidualRecurrence": eligible, "historicalFrequencyEnvelope": {"minimum":low,"maximum":high},
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
        "nonSupportingSectorCount":sum(x.get("recurrenceClassification") not in {
            "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY", "RESOLUTION_LIMITED"} for x in eligible),
        "supportingTemporalSpanDays":span,
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


def _all_artifacts_verified(stage: Any) -> dict[str, str]:
    verified = {}
    for ref in stage.artifacts:
        path = Path(ref.path)
        if not ref.sha256 or not path.is_file() or sha256_file(path) != ref.sha256:
            raise RuntimeError(f"artifact SHA verification failed for {path.name}")
        verified[str(path.resolve())] = ref.sha256
    if not verified:
        raise RuntimeError(f"stage {stage.id} has no authoritative artifacts")
    return verified


def verify_frozen_science_lineage(stages: Iterable[Any]) -> dict[str, Any]:
    """Verify the connected v20.12--v20.16 chain before any archive I/O."""
    rows = list(stages)
    def latest(handler: str) -> Any:
        stage = next((s for s in reversed(rows) if s.handler_id == handler and
                      s.status == "COMPLETE" and isinstance(s.result, dict)), None)
        if stage is None: raise RuntimeError(f"missing frozen predecessor {handler}")
        return stage
    prepare_target = latest("openstar.tess.prepare-target")
    morphology = latest("openstar.tess.morphology.analyze")
    v12_prepare = latest("openstar.tess.multi-source-residual.prepare")
    v12_interpret = latest("openstar.tess.multi-source-residual.interpret")
    v13 = latest("openstar.tess.intrinsic-nonstationary.analyze")
    v14 = latest("openstar.tess.target-residual-mechanism.analyze")
    v16 = latest("openstar.tess.target-residual-mechanism-predictive-validation.analyze")
    final = latest("openstar.tess.finalize")

    artifact_hashes = {"prepareTargetArtifacts": _all_artifacts_verified(prepare_target),
        "v20.12PreparationArtifacts": _all_artifacts_verified(v12_prepare)}
    for item in v12_prepare.result.get("preparedSeries") or []:
        for key in ("coefficientSeriesPath", "datasetPath"):
            path = item.get(key)
            if path and str(Path(path).resolve()) not in artifact_hashes["v20.12PreparationArtifacts"]:
                raise RuntimeError(f"v20.12 preparedSeries {key} lacks an authoritative ArtifactReference")
    target_hashes = prepare_target.provenance.input_hashes if prepare_target.provenance else {}
    source_project = Path(str(prepare_target.result.get("sourceProjectPath") or ""))
    source_dataset = Path(str(prepare_target.result.get("datasetPath") or ""))
    primary_project_path = str(Path(str(prepare_target.result.get("projectPath") or "")).resolve())
    if not (primary_project_path in artifact_hashes["prepareTargetArtifacts"]
            and source_project.is_file() and source_dataset.is_file()
            and target_hashes.get("sourceProjectManifest") == sha256_file(source_project)
            and target_hashes.get("sourceDataset") == sha256_file(source_dataset)):
        raise RuntimeError("prepare-target source project/dataset provenance is not authoritative")
    morphology_sha, _ = verified_json_result(morphology, "morphology-v20.4.json")
    v12_interpret_sha, _ = verified_json_result(v12_interpret, "multi-source-residual-v20.12.json")
    v13_sha, _ = verified_json_result(v13, ("intrinsic-nonstationary-v20.13.json",
                                             "intrinsic-nonstationary-v20.31.json"))
    v14_sha, _ = verified_json_result(v14, "target-residual-mechanism-v20.14.json")
    v16_sha, _ = verified_json_result(v16,
        "target-residual-mechanism-predictive-validation-v20.16.json")
    final_sha, _ = verified_json_result(final,
        "conclusion-v20.16-target-residual-predictive-validation.json")

    from .tess_target_residual_mechanism_predictive_validation import v2013_lineage_matches
    v12_interpret_hashes = (v12_interpret.provenance.input_hashes
                            if v12_interpret.provenance else {})
    if v12_interpret_hashes.get("preparation") != sha256_json(v12_prepare.result):
        raise RuntimeError("v20.12 interpretation does not bind its preparation")
    v13_hashes = v13.provenance.input_hashes if v13.provenance else {}
    if not v2013_lineage_matches(stage_input_hashes=v13_hashes,
            result_input_provenance=v13.result.get("inputProvenance") or {},
            preparation=v12_prepare.result, interpretation=v12_interpret.result):
        raise RuntimeError("v20.13 does not bind the exact v20.12 snapshots")
    v14_hashes = v14.provenance.input_hashes if v14.provenance else {}
    if not (v14_hashes.get("v20.12Preparation") == sha256_json(v12_prepare.result)
            and v14_hashes.get("v20.12Interpretation") == sha256_json(v12_interpret.result)
            and v14_hashes.get("v20.13Result") == sha256_json(v13.result)):
        raise RuntimeError("v20.14 lineage does not bind v20.12/v20.13")

    source = v16.result.get("adjudicationSource") or {}
    source_handler = source.get("handlerID")
    if source_handler == "openstar.tess.target-residual-mechanism-adjudication.analyze":
        v15 = latest(source_handler)
        v15_sha, _ = verified_json_result(v15,
            "target-residual-mechanism-adjudication-v20.15.json")
        v15_hashes = v15.provenance.input_hashes if v15.provenance else {}
        v15_input = v15.result.get("inputProvenance") or {}
        if not (v15_hashes.get("v20.14Result") == sha256_json(v14.result)
                and v15_hashes.get("v20.14Artifact") == v14_sha
                and v15_input.get("frozenV20.14ResultHash") == sha256_json(v14.result)
                and v15_input.get("frozenV20.14ArtifactSHA256") == v14_sha):
            raise RuntimeError("v20.15 lineage does not bind frozen v20.14")
        adjudication, adjudication_sha = v15, v15_sha
    elif source_handler == "openstar.tess.target-residual-mechanism.analyze":
        if v14.result.get("adjudicationVersion") != "route-independent-all-models-v1":
            raise RuntimeError("direct v20.14 source lacks corrected semantics")
        adjudication, adjudication_sha = v14, v14_sha
    else:
        raise RuntimeError("v20.16 adjudication source is not authoritative")
    if not (source.get("stageID") == adjudication.id
            and source.get("resultHash") == sha256_json(adjudication.result)
            and source.get("artifactSHA256") == adjudication_sha
            and v16.result.get("frozenV20.14ResultHash") == sha256_json(v14.result)
            and v16.result.get("frozenV20.14ArtifactSHA256") == v14_sha
            and v16.result.get("frozenV20.13ResultHash") == sha256_json(v13.result)
            and v16.result.get("frozenV20.13ArtifactSHA256") == v13_sha):
        raise RuntimeError("v20.16 does not bind the authoritative frozen chain")
    v16_hashes = v16.provenance.input_hashes if v16.provenance else {}
    if not (v16_hashes.get("adjudicationResult") == sha256_json(adjudication.result)
            and v16_hashes.get("adjudicationArtifact") == adjudication_sha
            and v16_hashes.get("v20.14Result") == sha256_json(v14.result)
            and v16_hashes.get("v20.14Artifact") == v14_sha
            and v16_hashes.get("v20.13Result") == sha256_json(v13.result)
            and v16_hashes.get("v20.13Artifact") == v13_sha):
        raise RuntimeError("v20.16 stage provenance does not bind the frozen chain")
    if not (final.id == "031-finalize" and final.triggered_by_stage_id == v16.id
            and final.parameters.get("outputSuffix") == "v20.16-target-residual-predictive-validation"
            and final.result.get("targetResidualMechanismPredictiveValidation") == v16.result
            and final.result.get("recommendedNextTest") ==
                "ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION"):
        raise RuntimeError("031 conclusion is not the exact unresolved v20.16 boundary")
    return {"prepareTarget": prepare_target, "morphology": morphology,
        "v20.12Preparation": v12_prepare, "v20.12Interpretation": v12_interpret,
        "v20.13": v13, "v20.14": v14, "adjudication": adjudication,
        "v20.16": v16, "finalizer": final, "artifactHashes": artifact_hashes,
        "verifiedArtifactSHA256": {"morphology": morphology_sha,
            "v20.12Interpretation": v12_interpret_sha, "v20.13": v13_sha,
            "v20.14": v14_sha, "adjudication": adjudication_sha,
            "v20.16": v16_sha, "finalizer": final_sha}}


def build_archival_baseline_project(*, source_project_path: str|Path,
        source_dataset_entry: dict[str,Any], tic_id: int, candidate_sectors: Iterable[int]|None,
        previously_consumed: dict[int,list[dict[str,str]]], residual_reference_frequency: float,
        established_frequency: float, historical_frequency_envelope: tuple[float,float],
        output_dir: str|Path, investigation_id: str) -> dict[str,Any]:
    """Query once and materialize every unseen official sector, without a science cap."""
    from .tess_multisector import (TessArchiveInfrastructureError,
        _MAST_LIGHTKURVE_LOCK, _archive_io_failure, _download_selected_sector,
        _prepare_samples_float64, _safe, _search_lightcurves, _select_product_from_search)
    grid=frozen_search_grid(residual_reference_frequency)  # freeze before archive I/O
    parsed=[_valid_sector(x) for x in candidate_sectors] if candidate_sectors is not None else []
    if None in parsed: raise ValueError("invalid archival sector ID")
    sectors=sorted(set(parsed))
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
                    product = _product_provenance(selected, author=author,
                                                  cadence_seconds=cadence)
                    lc,_=_download_selected_sector(selected,tic_id=tic_id,sector=sector,author=author,cadence_seconds=cadence)
                    # Fit in Float64, then and only then quantize the generic payload.
                    times64,flux64,prep=_prepare_samples_float64(lc)
                    residual,prewhite=prewhiten_established_family(times64,flux64,established_frequency)
                    import numpy as np
                    times=np.asarray(times64-times64[0],dtype=np.float32); times[0]=np.float32(0)
                    materialized.append((sector,author,cadence,product,times,residual,prep,prewhite))
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
    for sector,author,cadence,product,times,residual,prep,prewhite in materialized:
        dataset_id=f"{source_dataset_entry['id']}-sector-{sector}-archival-residual-v1"
        dataset={"id":dataset_id,"targetName":f"TIC {tic_id} archival residual Sector {sector}","timeUnit":"days",
            "timeReference":"relative-to-first-distributed-sample","numericRepresentation":"Float32","fluxUnit":"normalized-prewhitened-residual",
            "times":[float(x) for x in times],"flux":[float(x) for x in residual],"frequencySearch":dict(grid),"reference":{}}
        path=root/f"{_safe(dataset_id)}.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(dataset,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        entry=copy.deepcopy(source_dataset_entry); entry.update({"id":dataset_id,"path":str(path.resolve()),"ticID":int(tic_id),"sector":sector,"author":author,"cadenceSeconds":cadence,"role":"archival-target-residual-baseline"}); entries.append(entry)
        prepared.append({"sector":sector,"datasetID":dataset_id,"datasetPath":str(path.resolve()),"author":author,"cadenceSeconds":cadence,
            "originalSamples":prep["originalSamples"],"finiteSamples":prep["finiteSamples"],
            "finiteSampleCount":prep["finiteSamples"],"distributedSamples":prep["distributedSamples"],
            "originalTimeOriginDays":prep["originalTimeOriginDays"],"baselineDays":prep["baselineDaysFloat64"],"priorSectorExclusionStatus":"UNSEEN_ADMITTED",
            "selectedProduct":product,
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
