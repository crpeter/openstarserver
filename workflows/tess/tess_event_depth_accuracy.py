"""Coordinator-side, software-blind TESS event-depth attenuation audit."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from typing import Any, Callable

from .tess_binary_confirmation import DURATION_FRACTIONS, _solve
from .tess_preprocessing import MAX_SAMPLES
from .tess_sector_archive import TessArchiveTransientError, _is_transient_transport_error

FREEZE_HANDLER_ID = "openstar.tess.event-depth-photometry.freeze"
AUDIT_HANDLER_ID = "openstar.tess.event-depth-attenuation.audit"
FREEZE_VERSION = "openstar.tess-event-depth-photometry-freeze.v1"
AUDIT_VERSION = "openstar.tess-event-depth-attenuation-audit.v1"
QUALITY_POLICY = "Lightkurve quality_bitmask='default'"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def _finite_pairs(times: Any, flux: Any) -> list[tuple[float, float]]:
    try:
        xs, ys = list(times), list(flux)
    except TypeError as error:
        raise ValueError("full-precision time and flux arrays are required") from error
    if len(xs) != len(ys):
        raise ValueError("full-precision time/flux length mismatch")
    pairs = []
    for x, y in zip(xs, ys):
        try: x, y = float(x), float(y)
        except (TypeError, ValueError): continue
        if math.isfinite(x) and math.isfinite(y): pairs.append((x, y))
    pairs.sort()
    if len(pairs) < 20 or len({x for x, _ in pairs}) != len(pairs):
        raise ValueError("insufficient or duplicate full-precision cadences")
    return pairs


def _search_identity(selected: Any) -> dict[str, Any]:
    table = getattr(selected, "table", None)
    if table is None or len(table) != 1:
        raise ValueError("selected official product lacks a unique catalog row")
    def clean(value: Any) -> Any:
        if hasattr(value, "item"): value = value.item()
        if isinstance(value, float) and not math.isfinite(value): return None
        return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)
    row = {name: clean(table[name][0]) for name in getattr(table, "colnames", [])}
    identity = {key: row.get(key) for key in
                ("obs_id", "obsid", "productFilename", "dataURI", "mission", "author", "exptime")}
    if not any(identity.values()): raise ValueError("selected official product identity is empty")
    return {"catalogRow": row, "identity": identity,
            "selectionRule": "official-author-priority-SPOC-then-TESS-SPOC; shortest-cadence; catalog-order"}


def _curve_arrays(curve: Any) -> tuple[list[float], list[float], str, str]:
    times = getattr(getattr(curve, "time", None), "value", None)
    flux_object = getattr(curve, "flux", None)
    flux = getattr(flux_object, "value", None)
    pairs = _finite_pairs(times, flux)
    xs, ys = map(list, zip(*pairs))
    meta = getattr(curve, "meta", {}) or {}
    column = str(meta.get("FLUX_ORIGIN") or meta.get("FLUX_COLUMN") or "PDCSAP_FLUX")
    units = str(getattr(flux_object, "unit", None) or meta.get("BUNIT") or "UNKNOWN")
    return xs, ys, column, units


def freeze_photometry(products: list[dict[str, Any]], timing_sectors: list[int], *,
                      binary_confirmation_sha256: str,
                      chronology_proof: dict[str, Any]) -> dict[str, Any]:
    """Freeze already-selected official products and bind them to workflow history."""
    if not isinstance(binary_confirmation_sha256, str) or len(binary_confirmation_sha256) != 64:
        raise ValueError("exact binary-confirmation hash is required")
    if chronology_proof.get("verifiedFromCompletedStages") is not True:
        raise ValueError("completed-stage chronology proof is required")
    if chronology_proof.get("externalEvidenceStageAlreadyCompleted") is not False:
        raise ValueError("external evidence already exists or chronology is ambiguous")
    if not timing_sectors or len(set(timing_sectors)) != len(timing_sectors) or any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in timing_sectors):
        raise ValueError("unique coherent timing sectors are required")
    product_sectors = [item.get("sector") for item in products]
    if len(product_sectors) != len(set(product_sectors)) or set(product_sectors) != set(timing_sectors):
        raise ValueError("official product sectors do not exactly match coherent timing sectors")
    frozen = []
    required = ("cadenceSeconds", "author", "productIdentity", "sourceProductProvenance",
                "fluxColumn", "fluxUnits", "qualityMaskPolicy")
    for sector in timing_sectors:
        item = next(x for x in products if x.get("sector") == sector)
        if any(item.get(key) in (None, "") for key in required):
            raise ValueError(f"sector {sector} lacks required official-product provenance")
        pairs = _finite_pairs(item.get("time"), item.get("flux")); times, raw = map(list, zip(*pairs))
        scale = statistics.median(raw)
        if not math.isfinite(scale) or scale <= 0: raise ValueError("invalid relative-flux normalization scale")
        relative = [value / scale for value in raw]
        payload = {"sector": sector, "timeBTJDFloat64": times,
                   "originalFluxFloat64": list(raw), "relativeFluxFloat64": relative,
                   "normalization": "DIVIDE_BY_POSITIVE_MEDIAN", "normalizationScale": scale,
                   **{key: item[key] for key in required}, "sampleCount": len(times)}
        payload["timeSHA256"] = _sha(times)
        payload["originalFluxSHA256"] = _sha(list(raw))
        payload["relativeFluxSHA256"] = _sha(relative)
        payload["frozenInputSHA256"] = _sha(payload)
        frozen.append(payload)
    result = {"resultVersion": FREEZE_VERSION, "status": "FROZEN",
              "binaryConfirmationSHA256": binary_confirmation_sha256,
              "timingSectors": list(timing_sectors), "sectors": frozen,
              "fullFiniteCadencePreserved": True, "distributedSearchDatasetReplaced": False,
              "chronologyProof": chronology_proof, "frozenBeforeExternalKnownObjectQuery": True,
              "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False}
    result["freezeSHA256"] = _sha(result)
    return result


def acquire_full_precision_photometry(*, tic_id: int, timing_sectors: list[int],
                                      binary_confirmation_sha256: str,
                                      chronology_proof: dict[str, Any],
                                      search: Callable[[int], Any] | None = None,
                                      selector: Callable[[Any, int], Any] | None = None,
                                      downloader: Callable[..., Any] | None = None,
                                      archive_lock: Any = None) -> dict[str, Any]:
    """Select/download every official timing-sector product under the shared MAST lock."""
    products = []
    archive_failure = _is_transient_transport_error
    if search is None or selector is None or downloader is None or archive_lock is None:
        from . import tess_multisector as archive
        search = search or archive._search_lightcurves
        selector = selector or archive._select_product_from_search
        downloader = downloader or archive._download_selected_sector
        archive_lock = archive_lock or archive._MAST_LIGHTKURVE_LOCK
        archive_failure = archive._archive_io_failure
    try:
        with archive_lock:
            catalog = search(int(tic_id))
            for sector in timing_sectors:
                selected, author, cadence = selector(catalog, int(sector))
                provenance = _search_identity(selected)
                curve, source = downloader(selected, tic_id=int(tic_id), sector=int(sector),
                                                          author=author, cadence_seconds=cadence)
                times, flux, column, units = _curve_arrays(curve)
                products.append({"sector": int(sector), "time": times, "flux": flux,
                                 "cadenceSeconds": source["cadenceSeconds"], "author": source["author"],
                                 "productIdentity": provenance["identity"],
                                 "sourceProductProvenance": provenance, "fluxColumn": column,
                                 "fluxUnits": units, "qualityMaskPolicy": QUALITY_POLICY})
    except Exception as error:
        if isinstance(error, TessArchiveTransientError) or archive_failure(error):
            transient = TessArchiveTransientError("TESS event-depth photometry acquisition failed transiently")
            transient.diagnostics = {"operation": "event-depth-photometry-freeze", "ticID": int(tic_id),
                                     "timingSectors": list(timing_sectors),
                                     "completedSectorCount": len(products),
                                     "failure": f"{type(error).__name__}: {error}"}
            raise transient from error
        result = {"resultVersion": FREEZE_VERSION, "status": "UNRESOLVED",
                  "unresolvedReasons": [f"{type(error).__name__}: {error}"],
                  "binaryConfirmationSHA256": binary_confirmation_sha256,
                  "timingSectors": list(timing_sectors), "sectors": [],
                  "chronologyProof": chronology_proof,
                  "frozenBeforeExternalKnownObjectQuery": True,
                  "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False}
        result["freezeSHA256"] = _sha(result)
        return result
    return freeze_photometry(products, timing_sectors,
                             binary_confirmation_sha256=binary_confirmation_sha256,
                             chronology_proof=chronology_proof)


def validate_freeze(freeze: dict[str, Any], binary: dict[str, Any], binary_sha256: str) -> None:
    if freeze.get("resultVersion") != FREEZE_VERSION or freeze.get("status") != "FROZEN":
        raise ValueError("complete full-precision photometry freeze is required")
    copy = dict(freeze); claimed = copy.pop("freezeSHA256", None)
    if claimed != _sha(copy): raise ValueError("top-level photometry freeze hash mismatch")
    if freeze.get("binaryConfirmationSHA256") != binary_sha256:
        raise ValueError("photometry freeze is not bound to this binary confirmation")
    expected = (binary.get("linearEphemeris") or {}).get("timingSectors") or []
    actual = freeze.get("timingSectors") or []
    sectors = freeze.get("sectors") or []; ids = [item.get("sector") for item in sectors]
    if len(ids) != len(set(ids)) or set(ids) != set(expected) or actual != list(expected) or ids != list(expected):
        raise ValueError("frozen sectors do not exactly match coherent ephemeris timing sectors")
    for item in sectors:
        checks = (("timeBTJDFloat64", "timeSHA256"), ("originalFluxFloat64", "originalFluxSHA256"),
                  ("relativeFluxFloat64", "relativeFluxSHA256"))
        if any(item.get(hash_key) != _sha(item.get(value_key)) for value_key, hash_key in checks):
            raise ValueError(f"sector {item.get('sector')} frozen array hash mismatch")
        sector_copy = dict(item); sector_hash = sector_copy.pop("frozenInputSHA256", None)
        if sector_hash != _sha(sector_copy):
            raise ValueError(f"sector {item.get('sector')} frozen-input hash mismatch")


def _distance(time: float, epoch: float, period: float) -> float:
    return abs((time - epoch + period / 2) % period - period / 2)


def _measure(times: list[float], flux: list[float], center: float, duration: float,
             excluded_centers: list[float]) -> dict[str, Any]:
    inside = [y for x, y in zip(times, flux) if abs(x - center) <= duration / 2]
    outside = [y for x, y in zip(times, flux) if duration <= abs(x - center) <= 4 * duration
               and all(abs(x - other) > duration for other in excluded_centers)]
    if len(inside) < 3 or len(outside) < 8:
        return {"usable": False, "reason": "INSUFFICIENT_LOCAL_EVENT_OR_BASELINE_SAMPLES",
                "inEventSampleCount": len(inside), "outOfEventSampleCount": len(outside)}
    baseline, event = statistics.median(outside), statistics.median(inside)
    if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(event):
        return {"usable": False, "reason": "INVALID_LOCAL_BASELINE",
                "inEventSampleCount": len(inside), "outOfEventSampleCount": len(outside)}
    scatter = 1.4826 * statistics.median(abs(x - baseline) for x in outside)
    uncertainty = scatter * math.sqrt(1 / len(inside) + 1 / len(outside)) / baseline
    return {"usable": True, "localBaseline": baseline, "eventLevel": event,
            "depthFractionalFlux": (baseline - event) / baseline,
            "depthUncertaintyFractionalFlux": uncertainty,
            "inEventSampleCount": len(inside), "outOfEventSampleCount": len(outside)}


def _subset(times: list[float], flux: list[float], cap: int) -> tuple[list[float], list[float]]:
    if len(times) <= cap: return list(times), list(flux)
    indices = [int(p * (len(times) - 1) / (cap - 1)) for p in range(cap)]
    return [times[i] for i in indices], [flux[i] for i in indices]


def _standardize_roundtrip(values: list[float]) -> tuple[list[float], dict[str, float]]:
    mean, sigma = statistics.mean(values), statistics.pstdev(values)
    if not math.isfinite(sigma) or sigma <= 0: raise ValueError("zero flux variance")
    quantize = lambda x: struct.unpack("!f", struct.pack("!f", x))[0]
    standardized = [quantize((x - mean) / sigma) for x in values]
    reconstructed = [float(x) * sigma + mean for x in standardized]
    return reconstructed, {"sourceMean": mean, "sourceStandardDeviation": sigma}


def _harmonic_on_original_scale(times: list[float], flux: list[float], period: float,
                                masks: list[bool]) -> list[float]:
    rows = []
    for time, value, masked in zip(times, flux, masks):
        if masked: continue
        angle = 2 * math.pi * time / period
        rows.append(([1.0, math.sin(angle), math.cos(angle), math.sin(2*angle), math.cos(2*angle)], value))
    if len(rows) < 20: raise ValueError("insufficient protected harmonic samples")
    normal = [[0.0]*5 for _ in range(5)]; rhs = [0.0]*5
    for row, value in rows:
        for i in range(5):
            rhs[i] += row[i]*value
            for j in range(5): normal[i][j] += row[i]*row[j]
    beta = _solve(normal, rhs)
    # Remove only varying terms. Keeping the fitted intercept preserves the original fractional scale.
    answer = []
    for time, value in zip(times, flux):
        angle = 2 * math.pi * time / period
        varying = beta[1]*math.sin(angle)+beta[2]*math.cos(angle)+beta[3]*math.sin(2*angle)+beta[4]*math.cos(2*angle)
        answer.append(value-varying)
    return answer


def _events(times: list[float], epoch: float, period: float) -> list[tuple[str, float]]:
    low, high = min(times), max(times); first = math.floor((low-epoch)/period)-1
    result = []
    for cycle in range(first, math.ceil((high-epoch)/period)+2):
        for kind, center in (("PRIMARY", epoch+cycle*period), ("OPPOSITE_CONJUNCTION", epoch+(cycle+.5)*period)):
            if low <= center <= high: result.append((kind, center))
    return sorted(result, key=lambda item: item[1])


def _attenuation(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    a, b = before.get("depthFractionalFlux"), after.get("depthFractionalFlux")
    return None if not before.get("usable") or not after.get("usable") or not a else 1-b/a


def _finalize_audit(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the canonical hash to every terminal audit result."""
    finalized = dict(result)
    finalized.pop("auditSHA256", None)
    finalized["auditSHA256"] = _sha(finalized)
    return finalized


def validate_audit_hash(result: dict[str, Any]) -> str:
    value = dict(result)
    claimed = value.pop("auditSHA256", None)
    if claimed != _sha(value):
        raise ValueError("persisted event-depth attenuation audit hash mismatch")
    return claimed


def _unresolved(base: dict[str, Any], reasons: list[str], recommendation: str) -> dict[str, Any]:
    return _finalize_audit({**base, "status": "UNRESOLVED", "unresolvedReasons": reasons,
                            "suitableForLaterPrecisionModeling": False,
                            "recommendedNextTest": recommendation})


def audit_depth_attenuation(freeze: dict[str, Any], binary: dict[str, Any], *,
                            binary_confirmation_sha256: str, downsampling_cap: int = MAX_SAMPLES) -> dict[str, Any]:
    base = {"resultVersion": AUDIT_VERSION, "classification": "DIAGNOSTIC_ONLY",
            "detectionOnlyStandardizedBoxDepthIsPhysical": False,
            "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False,
            "companionRadiusInferred": False, "precisionPhysicalTransitSolutionClaimed": False,
            "binaryConfirmationSHA256": binary_confirmation_sha256}
    ephemeris = binary.get("linearEphemeris") or {}
    independent = binary.get("independentEvidence") or {}
    independent_ephemeris = independent.get("independentLinearEphemeris") or {}
    if binary.get("catalogAnswerKeyUsed") is not False:
        return _unresolved(base, ["BINARY_CONFIRMATION_BLINDNESS_GATE_FAILED"],
                           "PRECISION_EVENT_PHOTOMETRY_REVIEW")
    if independent.get("classification") != "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED":
        return _unresolved(base, ["INDEPENDENT_EVENT_REPLICATION_UNAVAILABLE"],
                           "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY")
    support = independent.get("supportingIndependentSectorCount")
    supporting_sectors = independent.get("supportingSectors")
    if (isinstance(support, bool) or not isinstance(support, int) or support < 3
            or not isinstance(supporting_sectors, list) or len(supporting_sectors) < 3
            or support != len(supporting_sectors)
            or len(set(supporting_sectors)) != len(supporting_sectors)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in supporting_sectors)):
        return _unresolved(base, ["INSUFFICIENT_INDEPENDENT_EVENT_SUPPORT"],
                           "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY")
    if independent_ephemeris.get("coherent") is not True:
        return _unresolved(base, ["INDEPENDENT_EPHEMERIS_INCOHERENT"],
                           "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY")
    if ephemeris.get("coherent") is not True:
        return _unresolved(base, ["COHERENT_EPHEMERIS_UNAVAILABLE"],
                           "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY")
    if freeze.get("status") != "FROZEN":
        return _unresolved(base, list(freeze.get("unresolvedReasons") or
                                     ["FULL_PRECISION_PHOTOMETRY_UNAVAILABLE"]),
                           "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY")
    try: validate_freeze(freeze, binary, binary_confirmation_sha256)
    except ValueError as error:
        return _unresolved(base, [str(error)], "PRECISION_EVENT_PHOTOMETRY_REVIEW")
    period, epoch = float(ephemeris["refinedPeriodDays"]), float(ephemeris["referenceEpoch"])
    duration_rows = []
    for row in binary.get("sectorResults", []):
        if row.get("role") != "INDEPENDENT" or row.get("usable") is not True:
            continue
        try: duty = float(row.get("dutyCycle"))
        except (TypeError, ValueError): continue
        sector = row.get("sector")
        if (math.isfinite(duty) and duty > 0 and isinstance(sector, int)
                and not isinstance(sector, bool) and sector > 0):
            if sector in supporting_sectors:
                duration_rows.append((sector, duty))
    if (not math.isfinite(period) or period <= 0 or len(duration_rows) < 3
            or len({sector for sector, _ in duration_rows}) < 3):
        return _unresolved(base, ["INSUFFICIENT_INDEPENDENT_EVENT_DURATION_SUPPORT"],
                           "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY")
    duties = [duty for _, duty in duration_rows]
    duration = statistics.median(duties)*period; sectors = []
    for frozen_sector in freeze["sectors"]:
        times = list(map(float, frozen_sector["timeBTJDFloat64"])); flux = list(map(float, frozen_sector["relativeFluxFloat64"]))
        event_defs = _events(times, epoch, period); centers = [center for _, center in event_defs]
        dt, df = _subset(times, flux, downsampling_cap)
        reconstructed, standardization = _standardize_roundtrip(df)
        masks = [any(abs(x-center) <= duration for center in centers) for x in times]
        protected_flux = _harmonic_on_original_scale(times, flux, period, masks)
        sector_events = []
        for kind, center in event_defs:
            full = _measure(times, flux, center, duration, centers)
            down = _measure(dt, df, center, duration, centers)
            standardized = _measure(dt, reconstructed, center, duration, centers)
            harmonic = _measure(times, protected_flux, center, duration, centers)
            candidates = [(duty*period, _measure(times, flux, center, duty*period, centers)) for duty in DURATION_FRACTIONS]
            usable_candidates = [(d, m) for d, m in candidates if m.get("usable")]
            # Mirror the existing box search's signal-to-noise duration choice,
            # while holding the already-established event center fixed.
            selected_duration, selected = max(
                usable_candidates,
                key=lambda pair: (pair[1]["depthFractionalFlux"] /
                                  max(pair[1]["depthUncertaintyFractionalFlux"], 1e-15),
                                  pair[1]["depthFractionalFlux"], -pair[0]),
            ) if usable_candidates else (None, {"usable": False})
            sector_events.append({"eventType": kind, "eventCenterBTJD": center,
                "establishedDurationDays": duration, "fullPrecisionLocalBaseline": full,
                "postDownsampling": down, "postStandardizationFloat32RoundTrip": standardized,
                "standardizationRoundTrip": standardization,
                "postProtectedHarmonicSubtraction": harmonic,
                "durationGridSelection": {"durationGridFractions": list(DURATION_FRACTIONS),
                    "selectedDurationDays": selected_duration, "measurement": selected},
                "attenuationFractions": {"downsampling": _attenuation(full, down),
                    "standardizationFloat32": _attenuation(down, standardized),
                    "harmonicSubtraction": _attenuation(full, harmonic),
                    "discreteBoxDuration": _attenuation(full, selected)}})
        sectors.append({"sector": frozen_sector["sector"], "sampleCounts": {"full": len(times), "downsampled": len(dt),
                        "harmonicFitProtected": len(times)-sum(masks)},
                        "primaryAndOppositeConjunctionProtected": True, "eventResults": sector_events})
    primary = [e for s in sectors for e in s["eventResults"] if e["eventType"] == "PRIMARY" and e["fullPrecisionLocalBaseline"].get("usable")]
    summary = {}
    for key in ("downsampling", "standardizationFloat32", "harmonicSubtraction", "discreteBoxDuration"):
        values = [e["attenuationFractions"][key] for e in primary if e["attenuationFractions"][key] is not None]
        if values:
            med = statistics.median(values); mad = 1.4826*statistics.median(abs(x-med) for x in values)
            summary[key] = {"medianAttenuationFraction": med, "robustUncertainty": mad/math.sqrt(len(values)),
                            "eventCount": len(values), "sectorCount": len({s["sector"] for s in sectors if any(e in primary and e["attenuationFractions"][key] is not None for e in s["eventResults"])})}
        else: summary[key] = None
    complete = len(primary) >= 3 and len({s["sector"] for s in sectors if any(e in primary for e in s["eventResults"])}) >= 3
    result = {**base, "status": "COMPLETE" if complete else "UNRESOLVED", "unresolvedReasons": [] if complete else ["INSUFFICIENT_USABLE_PRIMARY_EVENTS"],
              "sectorResults": sectors, "crossSectorRobustSummary": summary, "eventDurationDays": duration,
              "suitableForLaterPrecisionModeling": complete,
              "recommendedNextTest": "JOINT_TRANSIT_ECLIPSE_PHASE_CURVE_MODELING" if complete else "ADDITIONAL_PRECISION_EVENT_PHOTOMETRY"}
    return _finalize_audit(result)
