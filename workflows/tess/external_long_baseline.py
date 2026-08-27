"""Provider-neutral, coordinator-local external long-baseline experiment.

The observable is seasonal predictive phase coherence, never Lomb--Scargle.
Providers return frozen measurements; future Gaia/ATLAS/ZTF adapters can honor
the same contract without changing the scientific interpretation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class ProviderTransientError(RuntimeError): pass
class ProviderUnavailable(RuntimeError): pass
class MalformedProviderData(RuntimeError): pass


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


DEFAULT_CONTRACT = {"minimumMeasurements": 80, "minimumBaselineDays": 730.0,
                    "minimumSeasons": 3, "acceptedBands": ["g", "V"],
                    "acceptedQualityFlags": ["GOOD"], "minimumPhaseBins": 6,
                    "maximumNeighborFluxFraction": 0.10}


def run_external_experiment(*, target: dict[str, Any], family_window: list[float],
                            neighbors: list[dict[str, Any]], providers: list[PhotometryProvider],
                            artifact_root: Path, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Select the first objectively usable provider and freeze its raw response."""
    policy = {**DEFAULT_CONTRACT, **(contract or {})}; attempts = []
    # Freeze the family before any provider call.
    preregistration = {"familyWindowDays": list(family_window), "providerPriority": [p.name for p in providers],
                       "qualityContract": policy, "observable": "blocked-seasonal-phase-prediction"}
    root = Path(artifact_root); root.mkdir(parents=True, exist_ok=True)
    (root / "preregistration.json").write_text(json.dumps(preregistration, sort_keys=True), encoding="utf-8")
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
            attempts.append({"provider": provider.name, "availability": "AVAILABLE", "rejectionReason": None})
            result = analyze_seasonal_coherence(rows, family_window, policy)
            result.update({"providersAttempted": attempts, "selectedProvider": provider.name,
                           "rawResponsePath": str(raw_path), "rawResponseSHA256": hashlib.sha256(raw).hexdigest(),
                           "requestParameters": {"familyWindowDays": family_window, "acceptedBands": policy["acceptedBands"]},
                           "providerProvenance": coverage, "crowdingFluxFraction": crowding})
            return result
        except ProviderUnavailable as exc:
            attempts.append({"provider": provider.name, "availability": "UNAVAILABLE", "rejectionReason": str(exc)})
    return _decision("EXTERNAL_DATA_INSUFFICIENT", attempts, None,
                     "No preregistered provider passed objective availability gates.", False, False, False)


def analyze_seasonal_coherence(rows, family_window, policy=DEFAULT_CONTRACT):
    """Narrow family fit with leave-last-season-out phase prediction."""
    if len(rows) < policy["minimumMeasurements"]:
        return _decision("EXTERNAL_DATA_INSUFFICIENT", [], None, "Too few quality measurements.", False, False, False)
    times=np.array([r["time"] for r in rows]); flux=np.array([r["flux"] for r in rows]); err=np.array([r["uncertainty"] for r in rows])
    seasons=np.floor((times-times.min())/365.25).astype(int); unique=np.unique(seasons)
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
    replicated=predictive <= 9.0; stable=predictive <= 4.0
    classification="EXTERNAL_STABLE_CLOCK_SUPPORTED" if stable else ("EXTERNAL_EVOLVING_RECURRENCE_SUPPORTED" if replicated else "EXTERNAL_RECURRENCE_NOT_REPLICATED")
    result=_decision(classification, [], None, "Blocked final-season phase prediction; physical interpretation remains unresolved.", replicated, stable, replicated and not stable)
    result.update({"bestPeriodDays": float(period), "periodGridStepDays": float(grid[1]-grid[0]), "predictiveReducedChiSquare": predictive,
                   "periodUncertaintyDays": float(grid[1]-grid[0]), "seasonCount": int(len(unique)), "measurementCount": len(rows),
                   "blindPeriodSearchPerformed": False, "lombScarglePerformed": False})
    return result


def _decision(classification, attempts, selected, rationale, replicated, stable, evolving):
    return {"classification": classification, "providersAttempted": attempts, "selectedProvider": selected,
            "externalRecurrenceReplicated": replicated, "stableClockSupported": stable,
            "waveformEvolutionSupported": evolving, "sourceAttributionReliableAtExternalResolution": False,
            "periodFamilyResolved": bool(stable), "physicalCycleResolved": False, "physicalMechanismResolved": False,
            "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"}, "rationale": rationale,
            "recommendedNextTest": None if classification != "EXTERNAL_DATA_INSUFFICIENT" else "MANUAL_EXTERNAL_DATA_REVIEW"}
