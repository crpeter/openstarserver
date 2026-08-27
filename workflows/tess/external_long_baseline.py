"""Provider-neutral, coordinator-local external long-baseline experiment.

The observable is seasonal predictive phase coherence, never Lomb--Scargle.
Providers return frozen measurements; future Gaia/ATLAS/ZTF adapters can honor
the same contract without changing the scientific interpretation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


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
                        "band": str(row["band"]), "quality": str(row.get("quality", "GOOD"))}
                if not all(math.isfinite(item[k]) for k in ("time", "flux", "uncertainty")):
                    raise ValueError
                result.append(item)
            return result
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise MalformedProviderData("Malformed external light curve") from exc

    @classmethod
    def from_environment(cls):
        """Build the official-client transport without exposing credentials."""
        username = os.environ.get("OPENSTAR_ASASSN_USERNAME")
        token = os.environ.get("OPENSTAR_ASASSN_TOKEN")
        if not username or not token:
            raise ProviderConfigurationUnavailable(
                "OPENSTAR_ASASSN_USERNAME and OPENSTAR_ASASSN_TOKEN are required")
        return cls(OfficialASASSNTransport(username, token))


class OfficialASASSNTransport:
    """Thin mapping to the optional official ``pyasassn`` Sky Patrol client."""
    def __init__(self, username, token):
        try:
            from pyasassn.client import SkyPatrolClient
            import pyasassn
        except ImportError as exc:
            raise ProviderConfigurationUnavailable(
                "The optional pyasassn package is not installed") from exc
        self._client = SkyPatrolClient(username, token)
        self.client_version = getattr(pyasassn, "__version__", "unknown")

    def coverage(self, target):
        lookup = ({"catalog": "tic", "id": target["ticID"]}
                  if target.get("ticID") is not None else
                  {"ra_deg": target.get("raDeg"), "dec_deg": target.get("decDeg")})
        # The official client performs the deterministic catalog/coordinate
        # lookup.  Keep its returned identity for positional validation.
        result = self._client.query_list(lookup, catalog="master_list")
        if result is None or len(result) == 0:
            return {"available": False, "reason": "source-not-found",
                    "lookup": lookup, "clientVersion": self.client_version}
        return {"available": True, "lookup": lookup,
                "selectedSource": str(result.iloc[0].get("asas_sn_id", "unknown")),
                "clientVersion": self.client_version,
                "product": "ASAS-SN Sky Patrol light curve"}

    def acquire(self, target, request):
        result = self._client.query_list(
            {"ra_deg": target.get("raDeg"), "dec_deg": target.get("decDeg")},
            catalog="stellar_main", download=True)
        # Canonical lossless JSON is the frozen provider response.  Tokens and
        # usernames are never included in request or provenance dictionaries.
        rows = result.to_dict(orient="records")
        mapped = [{"time": r.get("jd", r.get("hjd")),
                   "flux": r.get("flux", r.get("mag")),
                   "uncertainty": r.get("flux_err", r.get("mag_err")),
                   "band": r.get("filter", r.get("band")),
                   "quality": r.get("quality", "GOOD")} for r in rows]
        return json.dumps(mapped, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")


DEFAULT_CONTRACT = {"minimumMeasurements": 80, "minimumBaselineDays": 730.0,
                    "minimumSeasons": 3, "acceptedBands": ["g", "V"],
                    "acceptedQualityFlags": ["GOOD"], "minimumPhaseBins": 6,
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
    (root / "preregistration.json").write_text(json.dumps(preregistration, sort_keys=True), encoding="utf-8")
    if neighbors is None:
        return _decision("EXTERNAL_CONTAMINATION_AMBIGUOUS", attempts, None,
                         "Authoritative catalog-neighbor evidence is missing.", False, False, False)
    crowding = sum(float(n.get("fluxFraction", 0.0)) for n in neighbors
                   if float(n.get("separationArcsec", 1e9)) <= float(n.get("providerRadiusArcsec", 16.0)))
    if crowding > policy["maximumNeighborFluxFraction"]:
        return _decision("EXTERNAL_CONTAMINATION_AMBIGUOUS", attempts, None,
                         "Persisted catalog neighbors exceed the external-survey blending gate.", False, False, False)
    for provider in providers:
        try:
            coverage = provider.coverage(target)
            if coverage.get("available") is not True:
                attempts.append({"provider": provider.name, "availability": "UNAVAILABLE", "rejectionReason": coverage.get("reason")}); continue
            raw = provider.acquire(target, {"familyWindowDays": family_window, "acceptedBands": policy["acceptedBands"]})
            raw_path = root / f"{provider.name.lower()}-raw.json"
            if raw_path.exists():
                if hashlib.sha256(raw_path.read_bytes()).digest() != hashlib.sha256(raw).digest():
                    raise RuntimeError("Frozen provider response differs on recovery")
            else: raw_path.write_bytes(raw)
            rows = [r for r in provider.parse(raw) if r["band"] in policy["acceptedBands"] and r["quality"] in policy["acceptedQualityFlags"]]
            quality = objective_coverage(rows, policy)
            if quality["status"] != "QUALIFIED":
                attempts.append({"provider": provider.name, "availability": "AVAILABLE",
                                 "rejectionReason": quality["reason"]})
                continue
            attempts.append({"provider": provider.name, "availability": "AVAILABLE", "rejectionReason": None})
            result = analyze_seasonal_coherence(rows, family_window, policy)
            result.update({"providersAttempted": attempts, "selectedProvider": provider.name,
                           "rawResponsePath": str(raw_path), "rawResponseSHA256": hashlib.sha256(raw).hexdigest(),
                           "requestParameters": {"familyWindowDays": family_window, "acceptedBands": policy["acceptedBands"]},
                           "providerProvenance": coverage, "crowdingFluxFraction": crowding})
            result["acquiredAt"] = datetime.now(timezone.utc).isoformat()
            result["qualityContract"] = policy
            result["catalogNeighbors"] = neighbors
            return result
        except ProviderUnavailable as exc:
            attempts.append({"provider": provider.name, "availability": "UNAVAILABLE", "rejectionReason": str(exc)})
    return _decision("EXTERNAL_DATA_INSUFFICIENT", attempts, None,
                     "No preregistered provider passed objective availability gates.", False, False, False)


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
    result["bandResults"]=results; result["bandsAnalyzedSeparately"]=True
    return result


def _analyze_single_band(rows, family_window, policy=DEFAULT_CONTRACT):
    """Narrow family fit with leave-last-season-out phase prediction."""
    if len(rows) < policy["minimumMeasurements"]:
        return _decision("EXTERNAL_DATA_INSUFFICIENT", [], None, "Too few quality measurements.", False, False, False)
    times=np.array([r["time"] for r in rows]); flux=np.array([r["flux"] for r in rows]); err=np.array([r["uncertainty"] for r in rows])
    # A new season begins at an objective 90-day sampling gap.
    order=np.argsort(times); season_ids=np.zeros(len(times),dtype=int); current=0
    for left,right in zip(order[:-1],order[1:]):
        if times[right]-times[left] >= 90.0: current += 1
        season_ids[right]=current
    seasons=season_ids; unique=np.unique(seasons)
    if np.ptp(times) < policy["minimumBaselineDays"] or len(unique) < policy["minimumSeasons"]:
        return _decision("EXTERNAL_DATA_INSUFFICIENT", [], None, "Baseline or seasonal coverage is inadequate.", False, False, False)
    # Fixed finite grid is a narrow preregistered parameter fit, not a blind search.
    grid=np.linspace(float(family_window[0]), float(family_window[1]), 101)
    def score(period, mask):
        x=np.column_stack([np.ones(mask.sum()), np.sin(2*np.pi*times[mask]/period), np.cos(2*np.pi*times[mask]/period)])
        w=1/np.maximum(err[mask],1e-9); beta=np.linalg.lstsq(x*w[:,None],flux[mask]*w,rcond=None)[0]
        return beta, float(np.mean(((flux[mask]-x@beta)/np.maximum(err[mask],1e-9))**2))
    train=seasons != unique[-1]; period=min(grid,key=lambda p: score(p,train)[1]); beta,_=score(period,train)
    test=~train; x=np.column_stack([np.ones(test.sum()),np.sin(2*np.pi*times[test]/period),np.cos(2*np.pi*times[test]/period)])
    predictive=float(np.mean(((flux[test]-x@beta)/np.maximum(err[test],1e-9))**2))
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
    replicated=predictive <= 9.0
    stable=predictive <= 4.0 and phase_range <= 0.30
    classification="EXTERNAL_STABLE_CLOCK_SUPPORTED" if stable else ("EXTERNAL_EVOLVING_RECURRENCE_SUPPORTED" if replicated else "EXTERNAL_RECURRENCE_NOT_REPLICATED")
    result=_decision(classification, [], None, "Blocked final-season phase prediction; physical interpretation remains unresolved.", replicated, stable, replicated and not stable)
    # Seasonal jackknife is an uncertainty estimate, unlike the numerical grid
    # spacing.  Each leave-one-season-out fit is an independent long-baseline
    # perturbation of the shared-clock estimate.
    jack=[]; failures=[]
    for season in unique:
        mask=seasons != season
        try: jack.append(float(min(grid,key=lambda p: score(p,mask)[1])))
        except (ValueError,np.linalg.LinAlgError) as exc: failures.append({"season":int(season),"error":type(exc).__name__})
    interval=[float(np.percentile(jack,16)),float(np.percentile(jack,84))] if len(jack)>=3 else [None,None]
    uncertainty=(interval[1]-interval[0])/2 if interval[0] is not None else None
    result.update({"bestPeriodDays": float(period), "periodGridStepDays": float(grid[1]-grid[0]), "predictiveReducedChiSquare": predictive,
                   "periodUncertaintyDays": uncertainty, "periodUncertainty":{"method":"leave-one-season-out-jackknife-profile-grid","successfulResamples":len(jack),"intervalDays":interval,"failures":failures,"frozenFamilyWindowDays":family_window},
                   "seasonalPhaseOC":seasonal,"seasonalPhaseRangeRadians":phase_range,"seasonalAmplitudeRange":amplitude_range,
                   "seasonCount": int(len(unique)), "measurementCount": len(rows),
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
    if any(len({min(bins-1,int(((t/period)%1)*bins)) for t in values}) < bins
           for values in grouped.values()):
        return {"status":"REJECTED","reason":"inadequate-phase-coverage"}
    return {"status":"QUALIFIED","reason":None}


def _decision(classification, attempts, selected, rationale, replicated, stable, evolving):
    return {"classification": classification, "providersAttempted": attempts, "selectedProvider": selected,
            "externalRecurrenceReplicated": replicated, "stableClockSupported": stable,
            "waveformEvolutionSupported": evolving, "sourceAttributionReliableAtExternalResolution": False,
            "periodFamilyResolved": False, "physicalCycleResolved": False, "physicalMechanismResolved": False,
            "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"}, "rationale": rationale,
            "recommendedNextTest": None if classification != "EXTERNAL_DATA_INSUFFICIENT" else "MANUAL_EXTERNAL_DATA_REVIEW"}
