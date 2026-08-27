"""Provider-neutral, coordinator-local external long-baseline experiment.

The observable is seasonal predictive phase coherence, never Lomb--Scargle.
Providers return frozen measurements; future Gaia/ATLAS/ZTF adapters can honor
the same contract without changing the scientific interpretation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SUPPORTED_SKYPATROL_VERSION = "0.6.21"

class ProviderTransientError(RuntimeError): pass
class ProviderUnavailable(RuntimeError): pass
class MalformedProviderData(RuntimeError): pass
class ProviderConfigurationUnavailable(RuntimeError): pass


class PhotometryProvider(Protocol):
    name: str
    def coverage(self, target: dict[str, Any]) -> dict[str, Any]: ...
    def acquire(self, target: dict[str, Any], request: dict[str, Any]) -> bytes: ...
    def parse(self, raw: bytes) -> list[dict[str, Any]]: ...


@dataclass
class ASASSNSkyPatrolProvider:
    """Public-interface adapter with injected transport (no hidden credential)."""
    transport: Any = None
    name: str = "ASAS-SN_SKY_PATROL"

    def coverage(self, target):
        if self.transport is None:
            raise ProviderUnavailable("ASAS-SN public transport is not configured")
        return self.transport.coverage(target)

    def acquire(self, target, request):
        if self.transport is None:
            raise ProviderUnavailable("ASAS-SN public transport is not configured")
        return self.transport.acquire(target, request)

    def parse(self, raw):
        try:
            rows = json.loads(raw.decode("utf-8"))
            if not isinstance(rows, list): raise ValueError
            result = []
            for row in rows:
                item = {"time": float(row["time"]), "flux": float(row["flux"]),
                        "uncertainty": float(row["uncertainty"]),
                        "band": str(row["band"]), "quality": str(row.get("quality", "GOOD")),
                        "sourceID": (str(row["asasSnID"]) if row.get("asasSnID") is not None
                                     else None)}
                if not all(math.isfinite(item[k]) for k in ("time", "flux", "uncertainty")):
                    raise ValueError
                result.append(item)
            return result
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise MalformedProviderData("Malformed external light curve") from exc

    @classmethod
    def from_environment(cls):
        """Build the public official-client transport; no credential is used."""
        return cls(OfficialASASSNTransport())


class OfficialASASSNTransport:
    """Thin mapping to the optional official ``pyasassn`` Sky Patrol client."""
    def __init__(self):
        try:
            from pyasassn.client import SkyPatrolClient
        except ImportError as exc:
            raise ProviderConfigurationUnavailable(
                "The optional skypatrol package is not installed") from exc
        try:
            self.client_version = importlib.metadata.version("skypatrol")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProviderConfigurationUnavailable(
                "The skypatrol distribution metadata is unavailable") from exc
        if self.client_version != SUPPORTED_SKYPATROL_VERSION:
            raise ProviderConfigurationUnavailable(
                f"skypatrol=={SUPPORTED_SKYPATROL_VERSION} is required; "
                f"found {self.client_version}")
        try:
            self._client = SkyPatrolClient()
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise ProviderTransientError("Transient ASAS-SN client initialization failure") from exc

    def coverage(self, target):
        if target.get("ticID") is None:
            return {"available": False, "reason": "tic-id-required",
                    "clientVersion": self.client_version}
        lookup = {"catalog": "stellar_main", "idColumn": "tic_id",
                  "ids": [int(target["ticID"])]}
        # The official client performs the deterministic catalog/coordinate
        # lookup.  Keep its returned identity for positional validation.
        try:
            result = self._client.query_list(
                lookup["ids"], catalog="stellar_main", id_col="tic_id", download=False)
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise ProviderTransientError("Transient ASAS-SN lookup failure") from exc
        if result is None or len(result) == 0:
            return {"available": False, "reason": "source-not-found",
                    "lookup": lookup, "clientVersion": self.client_version}
        try:
            records = result.to_dict(orient="records")
        except (AttributeError, TypeError, ValueError) as exc:
            raise MalformedProviderData("ASAS-SN catalog lookup is not tabular") from exc
        if len(records) != 1:
            return {"available": False, "reason": "ambiguous-source-match",
                    "matchCount": len(records), "lookup": lookup,
                    "clientVersion": self.client_version}
        row = records[0]
        try:
            source_id = int(row["asas_sn_id"])
            source_ra = float(row["ra_deg"])
            source_dec = float(row["dec_deg"])
            separation = _angular_separation_arcsec(
                float(target["raDeg"]), float(target["decDeg"]), source_ra, source_dec)
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedProviderData("ASAS-SN catalog identity is incomplete") from exc
        if separation > 2.0:
            return {"available": False, "reason": "positional-match-exceeds-2-arcsec",
                    "matchSeparationArcsec": separation, "lookup": lookup,
                    "clientVersion": self.client_version}
        return {"available": True, "lookup": lookup,
                "selectedSource": str(source_id), "asasSnID": source_id,
                "sourceRaDeg": source_ra, "sourceDecDeg": source_dec,
                "matchSeparationArcsec": separation,
                "clientVersion": self.client_version,
                "product": "ASAS-SN Sky Patrol light curve"}

    def acquire(self, target, request):
        try:
            result = self._client.query_list([int(target["ticID"])],
                catalog="stellar_main", id_col="tic_id", download=True)
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise ProviderTransientError("Transient ASAS-SN acquisition failure") from exc
        # Canonical lossless JSON is the frozen provider response.  Tokens and
        # usernames are never included in request or provenance dictionaries.
        data = getattr(result, "data", None)
        if data is None:
            raise MalformedProviderData("ASAS-SN LightCurveCollection lacks .data")
        try:
            rows = data.to_dict(orient="records")
        except (AttributeError, TypeError, ValueError) as exc:
            raise MalformedProviderData("ASAS-SN LightCurveCollection .data is not tabular") from exc
        try:
            mapped = [{"time": _required_number(r, "jd", "hjd"),
                       "flux": _required_number(r, "flux", "mag"),
                       "uncertainty": _required_number(r, "flux_err", "mag_err"),
                       "band": str(r["phot_filter"]),
                       "quality": str(r["quality"]),
                       "asasSnID": int(r["asas_sn_id"])} for r in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedProviderData("ASAS-SN photometry schema is incomplete") from exc
        return json.dumps(mapped, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")


def _required_number(row: dict[str, Any], primary: str, fallback: str) -> float:
    value = row.get(primary, row.get(fallback))
    result = float(value)
    if not math.isfinite(result):
        raise MalformedProviderData(f"ASAS-SN {primary} is not finite")
    return result


def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1r, dec1r, ra2r, dec2r = map(math.radians, (ra1, dec1, ra2, dec2))
    value = (math.sin(dec1r) * math.sin(dec2r)
             + math.cos(dec1r) * math.cos(dec2r) * math.cos(ra1r - ra2r))
    return math.degrees(math.acos(max(-1.0, min(1.0, value)))) * 3600.0


DEFAULT_CONTRACT = {"minimumMeasurements": 80, "minimumBaselineDays": 730.0,
                    "minimumSeasons": 3, "acceptedBands": ["g", "V"],
                    "acceptedQualityFlags": ["G", "GOOD"], "minimumPhaseBins": 6,
                    "maximumNeighborFluxFraction": 0.10}


def run_external_experiment(*, target: dict[str, Any], family_window: list[float],
                            neighbors: list[dict[str, Any]], providers: list[PhotometryProvider],
                            artifact_root: Path, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Select the first objectively usable provider and freeze its raw response."""
    policy = {**DEFAULT_CONTRACT, **(contract or {}),
              "frozenFamilyCenterDays": float(sum(family_window) / 2.0)}; attempts = []
    # Freeze the family before any provider call.
    preregistration = {"familyWindowDays": list(family_window), "providerPriority": [p.name for p in providers],
                       "qualityContract": policy, "observable": "blocked-seasonal-phase-prediction"}
    root = Path(artifact_root); root.mkdir(parents=True, exist_ok=True)
    preregistration_path = root / "execution-preregistration.json"
    _write_once_json(preregistration_path, preregistration)
    artifacts = [_artifact_manifest_entry(preregistration_path, "EXECUTION_PREREGISTRATION")]
    if neighbors is None:
        result = _decision("EXTERNAL_CONTAMINATION_AMBIGUOUS", attempts, None,
                           "Authoritative catalog-neighbor evidence is missing.", False, False, False)
        return _complete_execution_result(result, artifacts, analysis_available=False)
    try:
        crowding = sum(float(n["fluxFraction"]) for n in neighbors
                       if float(n["separationArcsec"]) <= float(n.get("providerRadiusArcsec", 16.0)))
    except (KeyError, TypeError, ValueError):
        result = _decision("EXTERNAL_CONTAMINATION_AMBIGUOUS", attempts, None,
                           "Catalog-neighbor evidence is incomplete.", False, False, False)
        return _complete_execution_result(result, artifacts, analysis_available=False)
    if not math.isfinite(crowding):
        result = _decision("EXTERNAL_CONTAMINATION_AMBIGUOUS", attempts, None,
                           "Catalog-neighbor flux evidence is non-finite.", False, False, False)
        return _complete_execution_result(result, artifacts, analysis_available=False)
    if crowding > policy["maximumNeighborFluxFraction"]:
        result = _decision("EXTERNAL_CONTAMINATION_AMBIGUOUS", attempts, None,
                           "Persisted catalog neighbors exceed the external-survey blending gate.", False, False, False)
        result["crowdingFluxFraction"] = crowding
        return _complete_execution_result(result, artifacts, analysis_available=False)
    for provider in providers:
        try:
            coverage = provider.coverage(target)
            coverage_path = root / f"{provider.name.lower()}-coverage.json"
            _write_once_json(coverage_path, coverage)
            artifacts.append(_artifact_manifest_entry(coverage_path, "PROVIDER_COVERAGE"))
            if coverage.get("available") is not True:
                attempts.append({"provider": provider.name, "availability": "UNAVAILABLE", "rejectionReason": coverage.get("reason")}); continue
            request_parameters = {"familyWindowDays": family_window,
                                  "acceptedBands": policy["acceptedBands"],
                                  "acceptedQualityFlags": policy["acceptedQualityFlags"],
                                  "selectedSource": coverage.get("selectedSource")}
            raw = provider.acquire(target, request_parameters)
            raw_path = root / f"{provider.name.lower()}-raw.json"
            if raw_path.exists():
                if hashlib.sha256(raw_path.read_bytes()).digest() != hashlib.sha256(raw).digest():
                    raise RuntimeError("Frozen provider response differs on recovery")
            else: raw_path.write_bytes(raw)
            artifacts.append(_artifact_manifest_entry(raw_path, "RAW_PROVIDER_RESPONSE"))
            parsed = provider.parse(raw)
            selected_source = coverage.get("selectedSource")
            parsed_sources = {row.get("sourceID") for row in parsed if row.get("sourceID") is not None}
            if selected_source is not None and parsed_sources != {str(selected_source)}:
                raise MalformedProviderData(
                    "ASAS-SN photometry source identity differs from the frozen catalog match")
            rows = [r for r in parsed if r["band"] in policy["acceptedBands"] and r["quality"] in policy["acceptedQualityFlags"]]
            cleaned = {"measurements": rows, "rawMeasurementCount": len(parsed),
                       "acceptedMeasurementCount": len(rows),
                       "excludedMeasurementCount": len(parsed) - len(rows),
                       "qualityFilter":{"acceptedBands":policy["acceptedBands"],
                                        "acceptedQualityFlags":policy["acceptedQualityFlags"]}}
            cleaned_path = root / f"{provider.name.lower()}-cleaned.json"
            _write_once_json(cleaned_path, cleaned)
            artifacts.append(_artifact_manifest_entry(cleaned_path, "CLEANED_MEASUREMENTS"))
            quality = objective_coverage(rows, policy)
            quality_path = root / f"{provider.name.lower()}-quality.json"
            _write_once_json(quality_path, quality)
            artifacts.append(_artifact_manifest_entry(quality_path, "OBJECTIVE_QUALITY_GATE"))
            if quality["status"] != "QUALIFIED":
                attempts.append({"provider": provider.name, "availability": "AVAILABLE",
                                 "rejectionReason": quality["reason"]})
                continue
            attempts.append({"provider": provider.name, "availability": "AVAILABLE", "rejectionReason": None})
            result = analyze_seasonal_coherence(rows, family_window, policy)
            result.update({"providersAttempted": attempts, "selectedProvider": provider.name,
                           "rawResponsePath": str(raw_path), "rawResponseSHA256": hashlib.sha256(raw).hexdigest(),
                           "requestParameters": request_parameters,
                           "providerProvenance": coverage, "crowdingFluxFraction": crowding})
            acquisition_path = root / f"{provider.name.lower()}-acquisition.json"
            if acquisition_path.exists():
                acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
            else:
                acquisition = {"acquiredAt": datetime.now(timezone.utc).isoformat(),
                    "provider": provider.name, "rawSHA256": hashlib.sha256(raw).hexdigest()}
                _write_once_json(acquisition_path, acquisition)
            artifacts.append(_artifact_manifest_entry(acquisition_path, "ACQUISITION_METADATA"))
            result["acquiredAt"] = acquisition["acquiredAt"]
            result["cleanedMeasurementsPath"] = str(cleaned_path)
            result["cleanedMeasurementsSHA256"] = hashlib.sha256(cleaned_path.read_bytes()).hexdigest()
            result["qualityContract"] = policy
            result["catalogNeighbors"] = neighbors
            return _complete_execution_result(result, artifacts, analysis_available=True)
        except (ProviderUnavailable, MalformedProviderData) as exc:
            attempts.append({"provider": provider.name, "availability": "UNAVAILABLE", "rejectionReason": str(exc)})
    result = _decision("EXTERNAL_DATA_INSUFFICIENT", attempts, None,
                       "No preregistered provider passed objective availability gates.", False, False, False)
    return _complete_execution_result(result, artifacts, analysis_available=False)


def _artifact_manifest_entry(path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _complete_execution_result(result: dict[str, Any], artifacts: list[dict[str, Any]],
                               *, analysis_available: bool) -> dict[str, Any]:
    result["artifactManifest"] = list(artifacts)
    result["analysisAvailable"] = analysis_available
    result["operationalOutcome"] = (
        "ANALYSIS_COMPLETE" if analysis_available else result["classification"]
    )
    return result


def _write_once_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Immutable artifact differs on recovery: {path.name}")
        return
    path.write_bytes(encoded)


def analyze_seasonal_coherence(rows, family_window, policy=DEFAULT_CONTRACT):
    """Analyze calibrated bands separately and require agreement."""
    bands=sorted({str(r["band"]) for r in rows})
    if not bands:
        return _decision("EXTERNAL_DATA_INSUFFICIENT",[],None,"No accepted photometric band.",False,False,False)
    results={band:_analyze_single_band([r for r in rows if str(r["band"])==band],family_window,policy)
             for band in bands}
    qualified={b:r for b,r in results.items() if r["classification"]!="EXTERNAL_DATA_INSUFFICIENT"}
    if not qualified:
        result=_decision("EXTERNAL_DATA_INSUFFICIENT",[],None,"No band independently passes coverage gates.",False,False,False)
    elif len({r["classification"] for r in qualified.values()})>1:
        result=_decision("EXTERNAL_PIPELINE_OR_BAND_DEPENDENT",[],None,"Independent accepted bands disagree.",False,False,False)
    else:
        result=dict(next(iter(qualified.values())))
        periods=[float(item["bestPeriodDays"]) for item in qualified.values()
                 if item.get("bestPeriodDays") is not None]
        tolerances=[float(item.get("periodUncertaintyDays") or 0.0)
                    for item in qualified.values()]
        if len(periods)>1 and max(periods)-min(periods)>max(2e-4,sum(tolerances)):
            result=_decision("EXTERNAL_PIPELINE_OR_BAND_DEPENDENT",[],None,
                             "Accepted bands select incompatible periods.",False,False,False)
    result["bandResults"]=results; result["bandsAnalyzedSeparately"]=True
    return result


def _analyze_single_band(rows, family_window, policy=DEFAULT_CONTRACT):
    """Narrow frozen-family fit with blocked prediction in every season."""
    coverage=objective_coverage(rows,{**policy,
        "frozenFamilyCenterDays":float(sum(family_window)/2.0)})
    if coverage["status"] != "QUALIFIED":
        result=_decision("EXTERNAL_DATA_INSUFFICIENT", [], None,
                         f"Band coverage failed: {coverage['reason']}.",False,False,False)
        result["coverageAssessment"]=coverage
        return result
    times=np.array([r["time"] for r in rows],dtype=float)
    flux=np.array([r["flux"] for r in rows],dtype=float)
    err=np.array([r["uncertainty"] for r in rows],dtype=float)
    # A new season begins at an objective 90-day sampling gap.
    order=np.argsort(times); season_ids=np.zeros(len(times),dtype=int); current=0
    for left,right in zip(order[:-1],order[1:]):
        if times[right]-times[left] >= 90.0: current += 1
        season_ids[right]=current
    seasons=season_ids; unique=np.unique(seasons)
    # Fixed finite grid is a narrow preregistered fit, not a blind period search.
    grid=np.linspace(float(family_window[0]), float(family_window[1]), 401)
    def score(period, mask):
        x=np.column_stack([np.ones(mask.sum()), np.sin(2*np.pi*times[mask]/period), np.cos(2*np.pi*times[mask]/period)])
        w=1/np.maximum(err[mask],1e-9)
        beta=np.linalg.lstsq(x*w[:,None],flux[mask]*w,rcond=None)[0]
        residual=(flux[mask]-x@beta)/np.maximum(err[mask],1e-9)
        return beta,float(np.sum(residual**2))
    def predict_chi2(period,beta,mask):
        x=np.column_stack([np.ones(mask.sum()),np.sin(2*np.pi*times[mask]/period),
                           np.cos(2*np.pi*times[mask]/period)])
        return float(np.sum(((flux[mask]-x@beta)/np.maximum(err[mask],1e-9))**2))
    full=np.ones(len(times),dtype=bool)
    profile=np.array([score(float(p),full)[1] for p in grid])
    best_index=int(np.argmin(profile)); period=float(grid[best_index])
    best_chi2=float(profile[best_index])
    folds=[]; jack=[]
    for held in unique:
        train=seasons != held; test=~train
        train_profile=np.array([score(float(p),train)[1] for p in grid])
        fold_period=float(grid[int(np.argmin(train_profile))]); jack.append(fold_period)
        beta,_=score(fold_period,train)
        signal_chi2=predict_chi2(fold_period,beta,test)
        train_weights=1/np.maximum(err[train],1e-9)**2
        null_level=float(np.average(flux[train],weights=train_weights))
        null_chi2=float(np.sum(((flux[test]-null_level)/np.maximum(err[test],1e-9))**2))
        folds.append({"heldOutSeason":int(held),"trainingSeasonCount":int(len(unique)-1),
                      "selectedPeriodDays":fold_period,"testMeasurementCount":int(test.sum()),
                      "signalChiSquare":signal_chi2,"nullChiSquare":null_chi2,
                      "predictiveReducedChiSquare":signal_chi2/max(1,int(test.sum())-3)})
    predictive=float(np.median([item["predictiveReducedChiSquare"] for item in folds]))
    signal_total=float(sum(item["signalChiSquare"] for item in folds))
    null_total=float(sum(item["nullChiSquare"] for item in folds))
    null_improvement=(1.0-signal_total/null_total) if null_total>0 else 0.0
    # Measure phase independently in every season at the preregistered family
    # centre.  This detects discrete seasonal O-C steps rather than allowing a
    # slightly shifted global period to absorb evolution across annual gaps.
    reference=float(sum(family_window)/2.0); seasonal=[]
    for season in unique:
        mask=seasons==season; b,_=score(reference,mask)
        seasonal.append({"season":int(season),"phaseRadians":float(math.atan2(b[2],b[1])),
                         "amplitude":float(math.hypot(b[1],b[2])),"measurements":int(mask.sum())})
    unwrapped=np.unwrap([x["phaseRadians"] for x in seasonal])
    phase_range=float(np.ptp(unwrapped)); amplitude_range=float(np.ptp([x["amplitude"] for x in seasonal]))
    candidate_frequency=1.0/period
    alias_periods=[]
    for label,delta in (("DAILY_PLUS",1.0),("DAILY_MINUS",-1.0),
                        ("YEARLY_PLUS",1.0/365.25),("YEARLY_MINUS",-1.0/365.25)):
        frequency=candidate_frequency+delta
        if frequency>0:
            alias_period=1.0/frequency
            alias_periods.append({"alias":label,"periodDays":alias_period,
                                  "chiSquare":score(alias_period,full)[1]})
    alias_threshold=best_chi2+max(1.0,0.01*best_chi2)
    alias_ambiguous=any(item["chiSquare"]<=alias_threshold for item in alias_periods)
    replicated=predictive <= 9.0 and null_improvement >= 0.10
    stable=replicated and predictive <= 4.0 and phase_range <= 0.30 and not alias_ambiguous
    if alias_ambiguous and replicated:
        classification="EXTERNAL_ALIAS_AMBIGUOUS"
    elif stable:
        classification="EXTERNAL_STABLE_CLOCK_SUPPORTED"
    elif replicated:
        classification="EXTERNAL_EVOLVING_RECURRENCE_SUPPORTED"
    else:
        classification="EXTERNAL_RECURRENCE_NOT_REPLICATED"
    result=_decision(classification, [], None,
                     "Every season was held out once for phase prediction; physical interpretation remains unresolved.",
                     replicated,stable,classification=="EXTERNAL_EVOLVING_RECURRENCE_SUPPORTED")
    # Seasonal jackknife is an uncertainty estimate, unlike the numerical grid
    # spacing.  Each leave-one-season-out fit is an independent long-baseline
    # perturbation of the shared-clock estimate.
    failures=[]
    interval=[float(np.percentile(jack,16)),float(np.percentile(jack,84))] if len(jack)>=3 else [None,None]
    grid_resolution=float(grid[1]-grid[0])
    jackknife_half_width=((interval[1]-interval[0])/2 if interval[0] is not None else None)
    # A finite profile grid cannot support precision below half a cell.  Report
    # that resolution floor explicitly instead of false zero precision when all
    # seasonal jackknife optima occupy the same cell.
    profile_mask=profile<=best_chi2+1.0
    profile_interval=[float(grid[profile_mask][0]),float(grid[profile_mask][-1])]
    profile_half_width=max(period-profile_interval[0],profile_interval[1]-period,
                           grid_resolution/2)
    uncertainty=(max(jackknife_half_width or 0.0,profile_half_width,grid_resolution/2))
    result.update({"bestPeriodDays": float(period), "periodGridStepDays": float(grid[1]-grid[0]), "predictiveReducedChiSquare": predictive,
                   "periodUncertaintyDays": uncertainty, "periodUncertainty":{"method":"leave-one-season-out-jackknife-plus-delta-chi-square-profile","successfulResamples":len(jack),"jackknifeIntervalDays":interval,"jackknifeHalfWidthDays":jackknife_half_width,"profileLikelihood68IntervalDays":profile_interval,"profileHalfWidthDays":profile_half_width,"gridResolutionFloorDays":grid_resolution/2,"failures":failures,"frozenFamilyWindowDays":family_window},
                   "seasonalPhaseOC":seasonal,"seasonalPhaseRangeRadians":phase_range,"seasonalAmplitudeRange":amplitude_range,
                   "seasonCount": int(len(unique)), "measurementCount": len(rows),
                   "blockedSeasonFolds":folds,"allSeasonsHeldOut":len(folds)==len(unique),
                   "nullModel":{"model":"weighted-constant-flux","blockedChiSquare":null_total,
                                "signalBlockedChiSquare":signal_total,
                                "fractionalImprovement":null_improvement},
                   "aliasAssessment":{"dailyAndSeasonalAliasesTested":True,
                                      "competitors":alias_periods,"ambiguous":alias_ambiguous,
                                      "ambiguityThresholdChiSquare":alias_threshold},
                   "coverageAssessment":coverage,
                   "blindPeriodSearchPerformed": False, "lombScarglePerformed": False})
    return result


def objective_coverage(rows, policy):
    """Signal-blind provider gate: sampling, bands, seasons and phase coverage."""
    if len(rows) < policy["minimumMeasurements"]:
        return {"status":"REJECTED","reason":"insufficient-measurements"}
    times=sorted(float(r["time"]) for r in rows)
    if times[-1]-times[0] < policy["minimumBaselineDays"]:
        return {"status":"REJECTED","reason":"insufficient-baseline"}
    seasons=1+sum(b-a>=90.0 for a,b in zip(times,times[1:]))
    if seasons < policy["minimumSeasons"]:
        return {"status":"REJECTED","reason":"insufficient-seasons"}
    bands={r["band"] for r in rows}
    if not bands: return {"status":"REJECTED","reason":"no-accepted-band"}
    # Sampling-only phase coverage uses the frozen window centre and never
    # examines flux.  Each accepted season must populate enough phase bins.
    period=float(policy.get("frozenFamilyCenterDays") or 1.0)
    season=0; grouped={0:[]}
    for index,time in enumerate(times):
        if index and time-times[index-1]>=90.0:
            season+=1; grouped[season]=[]
        grouped[season].append(time)
    bins=int(policy["minimumPhaseBins"])
    populated={str(key):len({min(bins-1,int(((t/period)%1)*bins)) for t in values})
               for key,values in grouped.items()}
    if any(value < bins for value in populated.values()):
        return {"status":"REJECTED","reason":"inadequate-phase-coverage",
                "phaseBinsRequired":bins,"phaseBinsPopulatedBySeason":populated}
    return {"status":"QUALIFIED","reason":None,"seasonCount":seasons,
            "baselineDays":times[-1]-times[0],"phaseBinsRequired":bins,
            "phaseBinsPopulatedBySeason":populated}


def _decision(classification, attempts, selected, rationale, replicated, stable, evolving):
    next_tests = {
        "EXTERNAL_DATA_INSUFFICIENT": "MANUAL_EXTERNAL_DATA_REVIEW",
        "EXTERNAL_CONTAMINATION_AMBIGUOUS": "HIGHER_RESOLUTION_SOURCE_LOCALIZATION",
        "EXTERNAL_ALIAS_AMBIGUOUS": "INDEPENDENT_SAMPLING_WINDOW_PHOTOMETRY",
        "EXTERNAL_PIPELINE_OR_BAND_DEPENDENT": "INDEPENDENT_CALIBRATED_BAND_PHOTOMETRY",
    }
    return {"classification": classification, "providersAttempted": attempts, "selectedProvider": selected,
            "externalRecurrenceReplicated": replicated, "stableClockSupported": stable,
            "waveformEvolutionSupported": evolving, "sourceAttributionReliableAtExternalResolution": False,
            "periodFamilyResolved": False, "physicalCycleResolved": False, "physicalMechanismResolved": False,
            "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"}, "rationale": rationale,
            "recommendedNextTest": next_tests.get(classification)}
