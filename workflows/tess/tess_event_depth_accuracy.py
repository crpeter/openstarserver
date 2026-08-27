"""Software-blind, coordinator-side audit of narrow-event depth attenuation.

The values produced here are diagnostic fractional-flux estimates.  They are
not a transit model and must not be converted to companion properties.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any, Iterable

from .tess_binary_confirmation import _solve
from .tess_preprocessing import MAX_SAMPLES

FREEZE_HANDLER_ID = "openstar.tess.event-depth-photometry.freeze"
AUDIT_HANDLER_ID = "openstar.tess.event-depth-attenuation.audit"
FREEZE_VERSION = "openstar.tess-event-depth-photometry-freeze.v1"
AUDIT_VERSION = "openstar.tess-event-depth-attenuation-audit.v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()


def _finite_pairs(times: Iterable[Any], flux: Iterable[Any]) -> list[tuple[float, float]]:
    if not isinstance(times, (list, tuple)) or not isinstance(flux, (list, tuple)):
        raise ValueError("full-precision time and flux arrays are required")
    if len(times) != len(flux):
        raise ValueError("full-precision time/flux length mismatch")
    pairs = []
    for x, y in zip(times, flux):
        try: x, y = float(x), float(y)
        except (TypeError, ValueError): continue
        if math.isfinite(x) and math.isfinite(y): pairs.append((x, y))
    pairs.sort()
    if len(pairs) < 20 or len({x for x, _ in pairs}) != len(pairs):
        raise ValueError("insufficient or duplicate full-precision cadences")
    return pairs


def freeze_photometry(products: list[dict[str, Any]], timing_sectors: list[int], *,
                      before_external_known_object_query: bool) -> dict[str, Any]:
    """Freeze already-selected official products; never performs acquisition."""
    if before_external_known_object_query is not True:
        raise ValueError("photometry must be frozen before external known-object access")
    if not timing_sectors or any(isinstance(x, bool) or not isinstance(x, int) for x in timing_sectors):
        raise ValueError("coherent timing sectors are required")
    by_sector = {item.get("sector"): item for item in products}
    frozen = []
    for sector in timing_sectors:
        item = by_sector.get(sector)
        required = ("cadenceSeconds", "author", "productIdentity", "sourceProductProvenance",
                    "fluxColumn", "fluxUnits", "qualityMaskPolicy")
        if not item or any(item.get(key) in (None, "") for key in required):
            raise ValueError(f"sector {sector} lacks required official-product provenance")
        pairs = _finite_pairs(item.get("time"), item.get("flux"))
        times, raw = map(list, zip(*pairs))
        normalization = item.get("normalization", "DIVIDE_BY_MEDIAN")
        if normalization == "ORIGINAL_FLUX":
            relative, scale = list(raw), None
        elif normalization == "DIVIDE_BY_MEDIAN":
            scale = statistics.median(raw)
            if not math.isfinite(scale) or scale == 0: raise ValueError("invalid normalization scale")
            relative = [value / scale for value in raw]
        else:
            raise ValueError("undocumented relative-flux normalization")
        payload = {"sector": sector, "timeBTJDFloat64": times,
                   "relativeFluxFloat64": relative, "originalFluxFloat64": list(raw),
                   "normalization": normalization, "normalizationScale": scale,
                   **{key: item[key] for key in required}}
        payload["sampleCount"] = len(times)
        payload["timeSHA256"] = _sha(times); payload["originalFluxSHA256"] = _sha(list(raw))
        payload["relativeFluxSHA256"] = _sha(relative)
        payload["frozenInputSHA256"] = _sha(payload)
        frozen.append(payload)
    return {"resultVersion": FREEZE_VERSION, "status": "FROZEN",
            "timingSectors": list(timing_sectors), "sectors": frozen,
            "fullFiniteCadencePreserved": True, "distributedSearchDatasetReplaced": False,
            "frozenBeforeExternalKnownObjectQuery": True, "externalCatalogInformationUsed": False,
            "catalogAnswerKeyUsed": False, "freezeSHA256": _sha(frozen)}


def unresolved_freeze(reasons: list[str]) -> dict[str, Any]:
    return {"resultVersion": FREEZE_VERSION, "status": "UNRESOLVED", "reasons": reasons,
            "sectors": [], "frozenBeforeExternalKnownObjectQuery": True,
            "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False}


def _distance(time: float, epoch: float, period: float) -> float:
    return abs((time - epoch + period / 2) % period - period / 2)


def _measure(times: list[float], flux: list[float], epoch: float, period: float,
             duration: float, *, baseline_width: float = 4.0) -> dict[str, Any]:
    inside = [y for x, y in zip(times, flux) if _distance(x, epoch, period) <= duration / 2]
    outside = [y for x, y in zip(times, flux)
               if duration <= _distance(x, epoch, period) <= baseline_width * duration]
    if len(inside) < 3 or len(outside) < 8:
        return {"usable": False, "reason": "INSUFFICIENT_LOCAL_EVENT_OR_BASELINE_SAMPLES",
                "inEventSampleCount": len(inside), "outOfEventSampleCount": len(outside)}
    base = statistics.median(outside); event = statistics.median(inside)
    scatter = 1.4826 * statistics.median(abs(x - base) for x in outside)
    uncertainty = scatter * math.sqrt(1 / len(inside) + 1 / len(outside))
    return {"usable": True, "depthFractionalFlux": base - event,
            "depthUncertaintyFractionalFlux": uncertainty, "localBaseline": base,
            "inEventSampleCount": len(inside), "outOfEventSampleCount": len(outside)}


def _subset(times: list[float], flux: list[float], cap: int) -> tuple[list[float], list[float]]:
    if len(times) <= cap: return list(times), list(flux)
    idx = [int(p * (len(times) - 1) / (cap - 1)) for p in range(cap)]
    return [times[i] for i in idx], [flux[i] for i in idx]


def _standardize32(values: list[float]) -> list[float]:
    mean = statistics.mean(values); sigma = statistics.pstdev(values)
    if sigma <= 0: raise ValueError("zero flux variance")
    q = lambda x: struct.unpack("!f", struct.pack("!f", x))[0]
    return [q((x - mean) / sigma) for x in values]


def _protected_harmonic(times: list[float], flux: list[float], period: float,
                        masks: list[bool]) -> list[float]:
    rows = []
    for time, value, masked in zip(times, flux, masks):
        if masked: continue
        angle = 2 * math.pi * time / period
        rows.append(([1.0, math.sin(angle), math.cos(angle),
                      math.sin(2 * angle), math.cos(2 * angle)], value))
    if len(rows) < 20: raise ValueError("insufficient protected harmonic samples")
    normal = [[0.0] * 5 for _ in range(5)]; rhs = [0.0] * 5
    for row, value in rows:
        for i in range(5):
            rhs[i] += row[i] * value
            for j in range(5): normal[i][j] += row[i] * row[j]
    beta = _solve(normal, rhs)
    answer = []
    for time, value in zip(times, flux):
        angle = 2 * math.pi * time / period
        row = [1.0, math.sin(angle), math.cos(angle), math.sin(2 * angle), math.cos(2 * angle)]
        answer.append(value - sum(x * y for x, y in zip(row, beta)))
    return answer


def _attenuation(before: dict[str, Any], after: dict[str, Any], *, scale: float = 1.0) -> float | None:
    a, b = before.get("depthFractionalFlux"), after.get("depthFractionalFlux")
    if not before.get("usable") or not after.get("usable") or not a: return None
    return 1.0 - (b * scale / a)


def audit_depth_attenuation(freeze: dict[str, Any], binary: dict[str, Any], *,
                            downsampling_cap: int = MAX_SAMPLES) -> dict[str, Any]:
    base = {"resultVersion": AUDIT_VERSION, "classification": "DIAGNOSTIC_ONLY",
            "detectionOnlyStandardizedBoxDepthIsPhysical": False,
            "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False,
            "companionRadiusInferred": False, "precisionPhysicalTransitSolutionClaimed": False}
    ephem = binary.get("linearEphemeris") or {}
    if freeze.get("status") != "FROZEN" or ephem.get("coherent") is not True:
        return {**base, "status": "UNRESOLVED", "reasons": [
            "FULL_PRECISION_PHOTOMETRY_UNAVAILABLE" if freeze.get("status") != "FROZEN"
            else "COHERENT_EPHEMERIS_UNAVAILABLE"],
            "suitableForLaterPrecisionModeling": False,
            "recommendedNextTest": "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY"}
    period = float(ephem["refinedPeriodDays"]); epoch = float(ephem["referenceEpoch"])
    duties = [x.get("dutyCycle") for x in binary.get("sectorResults", []) if x.get("usable")]
    if period <= 0 or not duties:
        return {**base, "status": "UNRESOLVED", "reasons": ["EVENT_DURATION_UNAVAILABLE"],
                "suitableForLaterPrecisionModeling": False,
                "recommendedNextTest": "PRECISION_EVENT_PHOTOMETRY_REVIEW"}
    duration = statistics.median(map(float, duties)) * period
    results = []
    for sector in freeze["sectors"]:
        times = list(map(float, sector["timeBTJDFloat64"])); flux = list(map(float, sector["relativeFluxFloat64"]))
        full = _measure(times, flux, epoch, period, duration)
        dt, df = _subset(times, flux, downsampling_cap)
        down = _measure(dt, df, epoch, period, duration)
        standardized_flux = _standardize32(df); sigma = statistics.pstdev(df)
        standardized_raw = _measure(dt, standardized_flux, epoch, period, duration)
        standardized = dict(standardized_raw)
        if standardized.get("usable"):
            standardized["depthStandardized"] = standardized.pop("depthFractionalFlux")
            standardized["depthUncertaintyStandardized"] = standardized.pop("depthUncertaintyFractionalFlux")
        # Protect both conjunctions with a deliberately wider mask before harmonic fitting.
        protected = [(_distance(x, epoch, period) <= duration or
                      _distance(x, epoch + period / 2, period) <= duration) for x in times]
        fit_t = [x for x, mask in zip(times, protected) if not mask]
        fit_f = [y for y, mask in zip(flux, protected) if not mask]
        try:
            residual = _protected_harmonic(times, flux, period, protected)
            harmonic = _measure(times, residual, epoch, period, duration)
        except ValueError:
            harmonic = {"usable": False, "reason": "HARMONIC_FIT_FAILED"}
        half = _measure(times, flux, epoch, period, duration / 2)
        result = {"sector": sector["sector"], "fullPrecisionLocalBaseline": full,
                  "postDownsampling": down, "postStandardizationFloat32": standardized,
                  "postProtectedHarmonicSubtraction": harmonic,
                  "discreteHalfDurationDiagnostic": half,
                  "sampleCounts": {"full": len(times), "downsampled": len(dt),
                                   "harmonicFitProtected": len(fit_t)},
                  "primaryAndOppositeConjunctionProtected": True,
                  "attenuationFractions": {
                      "downsampling": _attenuation(full, down),
                      "standardizationFloat32": _attenuation(down, standardized_raw, scale=sigma),
                      "harmonicSubtraction": _attenuation(full, harmonic),
                      "discreteBoxDuration": _attenuation(full, half)}}
        results.append(result)
    usable = [r for r in results if r["fullPrecisionLocalBaseline"].get("usable")]
    summary = {}
    for key in ("downsampling", "standardizationFloat32", "harmonicSubtraction", "discreteBoxDuration"):
        vals = [r["attenuationFractions"][key] for r in usable if r["attenuationFractions"][key] is not None]
        summary[key] = {"medianAttenuationFraction": statistics.median(vals), "sectorCount": len(vals)} if vals else None
    return {**base, "status": "COMPLETE" if usable else "UNRESOLVED", "sectorResults": results,
            "crossSectorRobustSummary": summary, "eventDurationDays": duration,
            "eventWindowSource": "ALREADY_ESTABLISHED_COHERENT_BINARY_EPHEMERIS",
            "suitableForLaterPrecisionModeling": len(usable) == len(results) and len(usable) >= 3,
            "recommendedNextTest": ("JOINT_TRANSIT_ECLIPSE_PHASE_CURVE_MODELING" if len(usable) >= 3
                                    else "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY")}
