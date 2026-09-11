"""Manual append-only time-domain test of a target-localized period family.

This experiment consumes untouched official TESS light curves and deliberately
does not compute a Lomb-Scargle periodogram.  It asks whether a previously
persisted period family recurs in the flux time series and whether its waveform
is stable or evolves.  Detection remains distinct from physical interpretation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from openstar_investigation import (
    Investigation,
    InvestigationStore,
    sha256_file,
    sha256_json,
)
from openstar_workflow import StageRequest

from .tess_main_family_time_domain_recurrence import (
    METHOD as REUSED_ACF_METHOD,
    _jackknife_peak,
    _nearest,
    _peaks,
    gap_aware_acf,
)
from .tess_multisector import (
    _MAST_LIGHTKURVE_LOCK,
    _archive_io_failure,
    _download_selected_sector,
    _exptime_seconds,
    _search_lightcurves,
    _sector_from_search_row,
)
from .tess_period_family_difference_image import freeze_period_family_boundary
from .tess_residual_localization import _write_json
from .tess_sector_archive import TessArchiveTransientError
from .tess_target_residual_archival_baseline import _product_provenance


HANDLER_PREFIX = "openstar.tess.period-family-time-domain-evolution."
PREPARE_HANDLER = HANDLER_PREFIX + "prepare"
RUN_HANDLER = HANDLER_PREFIX + "run"
INTERPRET_HANDLER = HANDLER_PREFIX + "interpret"

# These sectors were selected before looking at their flux.  They provide three
# separated observing epochs while holding the archive product/cadence contract
# fixed.  Lower-cadence sectors are intentionally excluded from this first test.
DEFAULT_UNTOUCHED_SECTORS = (5, 6, 7, 8, 11, 61, 65, 66, 67, 68, 69, 87, 88)
CAMPAIGNS = {
    "EARLY_TESS": (5, 6, 7, 8, 11),
    "MIDDLE_EPOCH": (61, 65, 66, 67, 68, 69),
    "RECENT_EPOCH": (87, 88),
}

METHOD = {
    **REUSED_ACF_METHOD,
    # The reused generic implementation measures prominence against shoulders
    # only five 0.05-day cells away.  A coherent sinusoid therefore has a broad
    # ACF maximum with about 0.02-0.03 local rise, rather than a narrow 0.08
    # peak.  Cross-product, jackknife, sector, and epoch replication provide
    # the stronger false-positive controls for this targeted experiment.
    "minimumPeakProminence": 0.02,
    "maximumSamplesPerSector": 8_000,
    "minimumFiniteSamples": 500,
    "minimumBaselineDays": 20.0,
    "familyWindowMinimumPaddingDays": 0.10,
    "sapPdcsapMaximumLagDifferenceDays": 0.15,
    "crossSectorStableLagRangeDays": 0.15,
    "stableWaveformCorrelation": 0.75,
    "stableAmplitudeFractionalChange": 0.35,
    "minimumSupportingSectors": 3,
    "minimumCampaignsForPersistentResult": 3,
    "minimumQualitySectorsForNonreplication": 6,
    "preprocessing": {
        "qualityBitmask": "default",
        "commonFiniteSapPdcsapCadences": True,
        "normalization": "divide-by-median-minus-one",
        "linearTrendRemoved": True,
        "smoothing": None,
        "outlierClipping": None,
        "downsampling": "uniform-index-deterministic",
    },
}

_BASE_HANDLERS = (
    ("001-prepare-target", "openstar.tess.prepare-target"),
    ("002-primary-distributed-search", "openstar.tess.primary-project.run"),
    ("003-catalog-identity", "openstar.tess.catalog-identity"),
    ("004-hypotheses", "openstar.tess.hypotheses"),
    ("005-planner", "openstar.tess.planner"),
    ("006-prepare-independent-sectors", "openstar.tess.independent.prepare"),
    ("007-run-independent-sectors", "openstar.tess.independent.run"),
    ("008-interpret-independent-sectors", "openstar.tess.independent.interpret"),
    ("009-prepare-broad-independent-search", "openstar.tess.independent.broad.prepare"),
    ("010-run-broad-independent-search", "openstar.tess.independent.broad.run"),
    ("011-interpret-broad-independent-search", "openstar.tess.independent.broad.interpret"),
    ("012-finalize", "openstar.tess.finalize"),
    ("013-prepare-period-family-difference-imaging",
     "openstar.tess.period-family-difference-imaging.prepare"),
    ("014-run-period-family-difference-imaging",
     "openstar.tess.period-family-difference-imaging.run"),
    ("015-interpret-period-family-difference-imaging",
     "openstar.tess.period-family-difference-imaging.interpret"),
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _campaign_for_sector(sector: int) -> str:
    for name, sectors in CAMPAIGNS.items():
        if int(sector) in sectors:
            return name
    raise ValueError(f"Sector {sector} is not in the preregistered campaign map.")


def _campaign_from_preparation(sector: int, preparation: dict[str, Any]) -> str:
    """Resolve the campaign frozen by either the manual or reusable selector."""
    campaigns = preparation.get("campaigns") or {}
    if not campaigns:
        # Historical/manual preparation fixtures predate the persisted map and
        # retain the original fixed-sector campaign contract.
        return _campaign_for_sector(sector)
    matches = [name for name, sectors in campaigns.items()
               if int(sector) in {int(value) for value in sectors}]
    if len(matches) != 1:
        raise ValueError(
            f"Sector {sector} does not have exactly one frozen campaign assignment.")
    return matches[0]


def _boundary_stages(investigation: Investigation) -> tuple[Any, ...]:
    stages = investigation.stages
    if len(stages) == len(_BASE_HANDLERS) + 1:
        current = stages[-1]
        if not (
            current.id == "016-prepare-period-family-time-domain-evolution"
            and current.handler_id == PREPARE_HANDLER
            and current.status == "RUNNING"
            and current.triggered_by_stage_id == "015-interpret-period-family-difference-imaging"
        ):
            raise RuntimeError("Unexpected stage after the frozen 15-stage boundary.")
        stages = stages[:-1]
    if len(stages) != len(_BASE_HANDLERS):
        raise RuntimeError("Time-domain evolution requires the exact 15-stage localization boundary.")
    previous = None
    for stage, (stage_id, handler_id) in zip(stages, _BASE_HANDLERS):
        if not (
            stage.id == stage_id
            and stage.handler_id == handler_id
            and stage.status == "COMPLETE"
            and stage.result is not None
            and stage.triggered_by_stage_id == previous
        ):
            raise RuntimeError(f"Frozen time-domain boundary mismatch at {stage_id}.")
        previous = stage_id
    if not stages[-1].stop:
        raise RuntimeError("Stage 015 is not an explicit terminal localization result.")
    return stages


def freeze_time_domain_evolution_boundary(
    investigation: Investigation,
    *, sectors: Iterable[int] = DEFAULT_UNTOUCHED_SECTORS,
) -> dict[str, Any]:
    """Validate stage 015 and freeze an untouched-sector time-domain plan."""
    stages = _boundary_stages(investigation)
    by_id = {stage.id: stage for stage in stages}
    original = freeze_period_family_boundary(replace(investigation, stages=stages[:12]))
    preparation = by_id["013-prepare-period-family-difference-imaging"].result or {}
    run = by_id["014-run-period-family-difference-imaging"].result or {}
    localization = by_id["015-interpret-period-family-difference-imaging"].result or {}

    expected = sorted(item["sector"] for item in original["independentSectorDetections"])
    if not (
        preparation.get("ticID") == original["ticID"]
        and preparation.get("primaryDetection") == original["primaryDetection"]
        and sorted(item.get("sector") for item in preparation.get("sectorDetections") or [])
        == expected
        and run.get("periodDetectionRecomputed") is False
        and sorted(item.get("sector") for item in run.get("sectorResults") or []) == expected
        and not (run.get("errors") or [])
    ):
        raise RuntimeError("Stages 013-014 do not bind the frozen period-family evidence.")

    claim = (localization.get("claimDecision") or {}).get("claim")
    if not (
        localization.get("classification") == "TARGET_PERIOD_FAMILY_SUPPORTED"
        and claim == "HUMAN_REVIEW_REQUIRED"
        and localization.get("sourceAttributionResolved") is True
        and localization.get("variableSignalOrigin") == "TARGET"
        and localization.get("targetSupportingSectors") == expected
        and not (localization.get("offTargetSectors") or [])
        and not (localization.get("ambiguousSectors") or [])
        and not (localization.get("noQualitySectors") or [])
        and not (localization.get("errors") or [])
        and localization.get("periodDetectionRecomputed") is False
        and localization.get("periodFamilyResolved") is False
        and localization.get("physicalCycleResolved") is False
        and localization.get("physicalMechanismResolved") is False
        and localization.get("recommendedNextTest")
        == "UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION"
        and localization.get("preparationSHA256") == sha256_json(preparation)
        and localization.get("sectorResults") == run.get("sectorResults")
    ):
        raise RuntimeError("Stage 015 is not the target-supported unresolved-family boundary.")

    selected = tuple(int(value) for value in sectors)
    if not selected or len(set(selected)) != len(selected):
        raise RuntimeError("Untouched sectors must be a non-empty unique sequence.")
    consumed = {int(original["primaryDetection"]["sector"]), *expected}
    official = {
        int(value)
        for value in (((by_id["003-catalog-identity"].result or {}).get("tess") or {})
                      .get("officialSectors") or [])
    }
    if set(selected) & consumed:
        raise RuntimeError("The time-domain plan includes a previously consumed sector.")
    if not set(selected).issubset(official):
        raise RuntimeError("The time-domain plan includes a sector absent from official identity evidence.")
    campaigns = sorted({_campaign_for_sector(value) for value in selected})
    if campaigns != sorted(CAMPAIGNS):
        raise RuntimeError("The frozen sectors do not cover all preregistered observing epochs.")

    periods = [float(original["primaryDetection"]["periodDays"])] + [
        float(item["periodDays"]) for item in original["independentSectorDetections"]
    ]
    family_min, family_max = min(periods), max(periods)
    padding = max(
        float(METHOD["familyWindowMinimumPaddingDays"]),
        family_max - family_min,
    )
    return {
        "version": "openstar.tess-period-family-time-domain-boundary.v1",
        "investigationID": investigation.id,
        "ticID": original["ticID"],
        "sourceAttribution": "TARGET",
        "primaryPeriodDays": float(original["primaryDetection"]["periodDays"]),
        "persistedPeriodFamilyDays": sorted(periods),
        "familyCenterDays": float(np.median(periods)),
        "familyAcceptanceWindowDays": [family_min - padding, family_max + padding],
        "previouslyConsumedSectors": sorted(consumed),
        "untouchedSectors": list(selected),
        "campaigns": {name: [x for x in selected if x in values]
                      for name, values in CAMPAIGNS.items()},
        "archiveContract": {
            "mission": "TESS",
            "author": "SPOC",
            "cadenceSeconds": 120.0,
            "requiredFluxProducts": ["SAP", "PDCSAP"],
            "fallbackAllowed": False,
        },
        "periodSearchPerformed": False,
        "claim": "HUMAN_REVIEW_REQUIRED",
        "periodFamilyResolved": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
    }


def verified_time_domain_evolution_boundary(
    store: InvestigationStore,
    investigation: Investigation,
    *, sectors: Iterable[int] = DEFAULT_UNTOUCHED_SECTORS,
) -> tuple[dict[str, Any], dict[str, str]]:
    frozen = freeze_time_domain_evolution_boundary(investigation, sectors=sectors)
    hashes: dict[str, str] = {}
    for stage in _boundary_stages(investigation):
        digest = store.verified_terminal_stage_ledger_hash(investigation.id, stage)
        if digest is None:
            raise RuntimeError(f"Immutable ledger verification failed for {stage.id}.")
        hashes[stage.id] = digest
    return frozen, hashes


def admit_period_family_time_domain_evolution(
    store: InvestigationStore,
    investigation: Investigation,
) -> Investigation:
    """Explicitly reopen only the verified stage-015 manual boundary."""
    if any(stage.handler_id.startswith(HANDLER_PREFIX) for stage in investigation.stages):
        return investigation
    control = investigation.metadata.get("controlState")
    if (
        investigation.status == "RUNNING"
        and isinstance(control, dict)
        and control.get("recovery") == "TESS_MANUAL_PERIOD_FAMILY_TIME_DOMAIN_EVOLUTION_V1"
        and (control.get("selectedExperiment") or {}).get("id")
        == "016-prepare-period-family-time-domain-evolution"
        and (control.get("selectedExperiment") or {}).get("handler_id") == PREPARE_HANDLER
    ):
        verified_time_domain_evolution_boundary(store, investigation)
        return investigation
    if not (
        investigation.status == "QUIESCENT_AWAITING_DATA"
        and isinstance(control, dict)
        and control.get("recovery") == "TESS_MANUAL_PERIOD_FAMILY_DIFFERENCE_IMAGING_V1"
        and (control.get("selectedExperiment") or {}).get("id")
        == "013-prepare-period-family-difference-imaging"
    ):
        raise RuntimeError("Manual time-domain admission requires the exact stage-015 control state.")
    verified_time_domain_evolution_boundary(store, investigation)
    request = StageRequest(
        "016-prepare-period-family-time-domain-evolution",
        PREPARE_HANDLER,
        {},
        "015-interpret-period-family-difference-imaging",
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_MANUAL_PERIOD_FAMILY_TIME_DOMAIN_EVOLUTION_V1",
        },
    )


def prepare_period_family_time_domain_evolution(
    *, frozen_boundary: dict[str, Any], ledger_hashes: dict[str, str],
    output_dir: Path, investigation_id: str,
) -> dict[str, Any]:
    root = Path(output_dir) / "period-family-time-domain-evolution"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-period-family-time-domain-evolution-preparation.v1",
        "investigationID": investigation_id,
        "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "ticID": frozen_boundary["ticID"],
        "sourceAttribution": "TARGET",
        "primaryPeriodDays": frozen_boundary["primaryPeriodDays"],
        "persistedPeriodFamilyDays": frozen_boundary["persistedPeriodFamilyDays"],
        "familyCenterDays": frozen_boundary["familyCenterDays"],
        "familyAcceptanceWindowDays": frozen_boundary["familyAcceptanceWindowDays"],
        "previouslyConsumedSectors": frozen_boundary["previouslyConsumedSectors"],
        "untouchedSectors": frozen_boundary["untouchedSectors"],
        "campaigns": frozen_boundary["campaigns"],
        "archiveContract": frozen_boundary["archiveContract"],
        "method": METHOD,
        "execution": "coordinator-local-time-domain-analysis",
        "workerWorkload": None,
        "periodSearchPerformed": False,
        "authoritativeStageLedgerSHA256": dict(ledger_hashes),
        "claimBeforeExperiment": "HUMAN_REVIEW_REQUIRED",
        "scientificQuestion": (
            "Does the target-centered approximately 4.5-day family recur in untouched "
            "TESS flux, and is its cycle waveform stable or evolving?"
        ),
        "interpretationGuard": (
            "The persisted family defines only a preregistered lag window. ACF recurrence "
            "is an independent time-domain observable, not a new periodogram and not proof "
            "that the family is rotation, an orbit, or any other physical mechanism."
        ),
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def _column_values(light_curve: Any, name: str) -> np.ndarray:
    value = getattr(light_curve, name, None)
    if value is None:
        try:
            value = light_curve[name]
        except Exception as error:
            raise RuntimeError(f"Official light curve lacks required {name} column.") from error
    value = getattr(value, "value", value)
    if np.ma.isMaskedArray(value):
        value = np.ma.filled(value, np.nan)
    return np.asarray(value, dtype=np.float64)


def _prepare_flux_pair(light_curve: Any, *, maximum_samples: int) -> dict[str, Any]:
    time = np.asarray(getattr(light_curve.time, "value", light_curve.time), dtype=np.float64)
    sap = _column_values(light_curve, "sap_flux")
    pdcsap = _column_values(light_curve, "pdcsap_flux")
    if not (len(time) == len(sap) == len(pdcsap)):
        raise RuntimeError("TESS time, SAP, and PDCSAP columns have different lengths.")
    original = len(time)
    keep = np.isfinite(time) & np.isfinite(sap) & np.isfinite(pdcsap)
    time, sap, pdcsap = time[keep], sap[keep], pdcsap[keep]
    order = np.argsort(time)
    time, sap, pdcsap = time[order], sap[order], pdcsap[order]
    if len(time) < int(METHOD["minimumFiniteSamples"]):
        raise RuntimeError("Official light curve contains too few common finite SAP/PDCSAP cadences.")
    if len(time) > maximum_samples:
        indices = np.linspace(0, len(time) - 1, maximum_samples, dtype=np.int64)
        time, sap, pdcsap = time[indices], sap[indices], pdcsap[indices]
    time = time - time[0]
    baseline = float(time[-1])
    if baseline < float(METHOD["minimumBaselineDays"]):
        raise RuntimeError("Official light curve baseline is insufficient for the preregistered test.")

    def normalize(flux: np.ndarray) -> np.ndarray:
        median = float(np.median(flux))
        if not math.isfinite(median) or median == 0:
            raise RuntimeError("Official flux has an invalid median.")
        relative = flux / median - 1.0
        centered_time = time - float(np.mean(time))
        design = np.column_stack((np.ones(len(time)), centered_time))
        coefficients, _, _, _ = np.linalg.lstsq(design, relative, rcond=None)
        detrended = relative - design @ coefficients
        scale = float(np.std(detrended))
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError("Official flux has no finite time-domain variance.")
        return np.asarray(detrended / scale, dtype=np.float64)

    return {
        "time": np.asarray(time, dtype=np.float64),
        "sap": normalize(sap),
        "pdcsap": normalize(pdcsap),
        "originalSamples": int(original),
        "commonFiniteSamples": int(np.count_nonzero(keep)),
        "analysisSamples": int(len(time)),
        "baselineDays": baseline,
    }


def _select_spoc_120s(search: Any, sector: int) -> tuple[Any, str, float]:
    table = getattr(search, "table", None)
    if table is None or len(table) == 0:
        raise RuntimeError("MAST returned no TESS light-curve products.")
    columns = set(getattr(table, "colnames", []))
    candidates = []
    for index in range(len(table)):
        if _sector_from_search_row(table, index) != int(sector):
            continue
        author = str(table["author"][index]).strip().upper() if "author" in columns else ""
        cadence = _exptime_seconds(table["exptime"][index]) if "exptime" in columns else None
        if author == "SPOC" and cadence is not None and math.isclose(cadence, 120.0, abs_tol=1.0):
            candidates.append((abs(cadence - 120.0), index, search[index:index + 1]))
    if not candidates:
        raise RuntimeError(f"No official SPOC 120-second light curve found for Sector {sector}.")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2], "SPOC", 120.0


def _production_sector_inputs(preparation: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        with _MAST_LIGHTKURVE_LOCK:
            search = _search_lightcurves(int(preparation["ticID"]))
            sectors = list(preparation["untouchedSectors"])
            for position, sector in enumerate(sectors, start=1):
                print(f"      Sector {sector} ({position}/{len(sectors)}): selecting SPOC 120s light curve...", flush=True)
                try:
                    selected, author, cadence = _select_spoc_120s(search, int(sector))
                    provenance = _product_provenance(selected, author=author,
                                                     cadence_seconds=cadence)
                    provenance["selectionRule"] = "SPOC-exact-120-second-cadence; catalog-order; no-fallback"
                    print("        downloading official light curve...", flush=True)
                    light_curve, _ = _download_selected_sector(
                        selected,
                        tic_id=int(preparation["ticID"]),
                        sector=int(sector),
                        author=author,
                        cadence_seconds=cadence,
                    )
                    prepared = _prepare_flux_pair(
                        light_curve,
                        maximum_samples=int(METHOD["maximumSamplesPerSector"]),
                    )
                    results.append({"sector": int(sector), "provenance": provenance, **prepared})
                except Exception as error:
                    if _archive_io_failure(error):
                        raise TessArchiveTransientError(
                            f"TESS light-curve acquisition failed transiently in Sector {sector}."
                        ) from error
                    errors.append({"sector": int(sector),
                                   "error": f"{type(error).__name__}: {error}"})
    except TessArchiveTransientError:
        raise
    except Exception as error:
        if _archive_io_failure(error):
            raise TessArchiveTransientError("TESS light-curve catalog search failed transiently.") from error
        raise
    return results, errors


def _phase_profile(time: np.ndarray, flux: np.ndarray, period: float,
                   *, phase_bins: int) -> np.ndarray:
    phase = np.mod(time / period, 1.0)
    values = []
    for index in range(phase_bins):
        selected = flux[(phase >= index / phase_bins) & (phase < (index + 1) / phase_bins)]
        values.append(float(np.median(selected)) if len(selected) else np.nan)
    return np.asarray(values, dtype=np.float64)


def _profile_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    keep = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(keep) < max(3, int(0.75 * len(first))):
        return None
    x, y = first[keep], second[keep]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _waveform_evolution(time: np.ndarray, flux: np.ndarray, period: float) -> dict[str, Any]:
    bins = int(METHOD["phaseBins"])
    midpoint = 0.5 * (float(time[0]) + float(time[-1]))
    first = time <= midpoint
    second = time > midpoint
    first_profile = _phase_profile(time[first], flux[first], period, phase_bins=bins)
    second_profile = _phase_profile(time[second], flux[second], period, phase_bins=bins)
    correlation = _profile_correlation(first_profile, second_profile)
    first_amplitude = float(np.nanpercentile(flux[first], 95) - np.nanpercentile(flux[first], 5))
    second_amplitude = float(np.nanpercentile(flux[second], 95) - np.nanpercentile(flux[second], 5))
    denominator = max(first_amplitude, second_amplitude)
    fractional_change = (abs(first_amplitude - second_amplitude) / denominator
                         if denominator > 0 else None)

    cycle = np.floor(time / period).astype(int)
    profiles = {
        int(value): _phase_profile(time[cycle == value],
                                   flux[cycle == value], period, phase_bins=bins)
        for value in np.unique(cycle)
        if np.count_nonzero(cycle == value) >= bins
    }
    adjacent = []
    for value in sorted(profiles):
        if value + 1 in profiles:
            correlation_value = _profile_correlation(profiles[value], profiles[value + 1])
            if correlation_value is not None:
                adjacent.append(correlation_value)
    return {
        "referencePeriodDays": float(period),
        "referenceIsPhysicalInterpretation": False,
        "acceptedCycleCount": len(profiles),
        "adjacentCycleCorrelations": adjacent,
        "medianAdjacentCycleCorrelation": float(np.median(adjacent)) if adjacent else None,
        "firstHalfSecondHalfProfileCorrelation": correlation,
        "firstHalfAmplitude": first_amplitude,
        "secondHalfAmplitude": second_amplitude,
        "amplitudeFractionalChange": fractional_change,
        "evolutionEvidenceAvailable": bool(
            adjacent and correlation is not None and fractional_change is not None
        ),
    }


def _alternate_peak_family(peak: dict[str, Any], center: float) -> dict[str, Any]:
    ratio = float(peak["lagDays"]) / center
    candidates = [(0.5, "HALF_FAMILY"), (2.0, "DOUBLE_FAMILY"),
                  (3.0, "TRIPLE_FAMILY"), (4.0, "QUADRUPLE_FAMILY")]
    nearest, label = min(candidates, key=lambda item: abs(ratio - item[0]))
    relation = label if abs(ratio - nearest) <= 0.08 * nearest else "NON_HARMONIC_RECURRENCE_CANDIDATE"
    return {**peak, "ratioToFamilyCenter": ratio, "relationship": relation,
            "physicalModeClaim": False}


def analyze_flux_product(time: Iterable[float], flux: Iterable[float], *,
                         family_center_days: float,
                         family_window_days: Iterable[float]) -> dict[str, Any]:
    method = dict(METHOD)
    lags, acf, support, clean_time, clean_flux = gap_aware_acf(time, flux, method)
    peaks = _peaks(lags, acf, support, method)
    low, high = [float(value) for value in family_window_days]
    family_peaks = [item for item in peaks if low <= item["lagDays"] <= high]
    candidate = max(family_peaks, key=lambda item: item["peakCorrelation"], default=None)
    uncertainty = _jackknife_peak(clean_time, clean_flux, candidate, method) if candidate else None
    if candidate is not None:
        candidate = {**candidate, "uncertaintyEstimate": uncertainty}
    reliable = bool(
        candidate
        and uncertainty
        and uncertainty["successfulResamples"] >= int(method["minimumJackknifeDetections"])
        and uncertainty["intervalDays"]
        and uncertainty["intervalDays"][1] >= low
        and uncertainty["intervalDays"][0] <= high
    )
    waveform = _waveform_evolution(clean_time, clean_flux, family_center_days)
    stable = bool(
        reliable
        and waveform["medianAdjacentCycleCorrelation"] is not None
        and waveform["medianAdjacentCycleCorrelation"] >= METHOD["stableWaveformCorrelation"]
        and waveform["firstHalfSecondHalfProfileCorrelation"] is not None
        and waveform["firstHalfSecondHalfProfileCorrelation"] >= METHOD["stableWaveformCorrelation"]
        and waveform["amplitudeFractionalChange"] is not None
        and waveform["amplitudeFractionalChange"] <= METHOD["stableAmplitudeFractionalChange"]
    )
    alternates = [
        _alternate_peak_family(item, family_center_days)
        for item in peaks
        if not (low <= item["lagDays"] <= high)
    ]
    return {
        "familyRecurrenceSupported": reliable,
        "qualifyingFamilyPeak": candidate,
        "waveformEvolution": waveform,
        "stableWaveform": stable,
        "qualifyingAlternateRecurrencePeaks": alternates,
        "acfMethod": {
            "name": "gap-aware-normalized-slot-autocorrelation",
            "parameters": method,
            "reusedFrom": "tess_main_family_time_domain_recurrence",
        },
        "periodogramComputed": False,
    }


def analyze_sector_flux_pair(item: dict[str, Any], preparation: dict[str, Any]) -> dict[str, Any]:
    time = np.asarray(item["time"], dtype=np.float64)
    sap = analyze_flux_product(
        time, item["sap"],
        family_center_days=float(preparation["familyCenterDays"]),
        family_window_days=preparation["familyAcceptanceWindowDays"],
    )
    pdcsap = analyze_flux_product(
        time, item["pdcsap"],
        family_center_days=float(preparation["familyCenterDays"]),
        family_window_days=preparation["familyAcceptanceWindowDays"],
    )
    sap_peak = sap.get("qualifyingFamilyPeak") or {}
    pdcsap_peak = pdcsap.get("qualifyingFamilyPeak") or {}
    both_support = sap["familyRecurrenceSupported"] and pdcsap["familyRecurrenceSupported"]
    lag_difference = (abs(float(sap_peak["lagDays"]) - float(pdcsap_peak["lagDays"]))
                      if both_support else None)
    agreement = bool(
        both_support
        and lag_difference is not None
        and lag_difference <= METHOD["sapPdcsapMaximumLagDifferenceDays"]
    )
    if agreement:
        if sap["stableWaveform"] and pdcsap["stableWaveform"]:
            classification = "STABLE_TIME_DOMAIN_RECURRENCE"
        elif (sap["waveformEvolution"]["evolutionEvidenceAvailable"]
              and pdcsap["waveformEvolution"]["evolutionEvidenceAvailable"]):
            classification = "EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE"
        else:
            classification = "TIME_DOMAIN_RECURRENCE_WAVEFORM_UNRESOLVED"
    elif sap["familyRecurrenceSupported"] != pdcsap["familyRecurrenceSupported"] or both_support:
        classification = "SAP_PDCSAP_DISAGREEMENT"
    else:
        classification = "TIME_DOMAIN_RECURRENCE_NOT_DETECTED"
    return {
        "sector": int(item["sector"]),
        "campaign": _campaign_from_preparation(int(item["sector"]),preparation),
        "classification": classification,
        "sapPdcsapAgreement": agreement,
        "sapPdcsapPeakLagDifferenceDays": lag_difference,
        "consensusFamilyLagDays": (
            0.5 * (float(sap_peak["lagDays"]) + float(pdcsap_peak["lagDays"]))
            if agreement else None
        ),
        "sap": sap,
        "pdcsap": pdcsap,
        "baselineDays": float(item["baselineDays"]),
        "originalSamples": int(item["originalSamples"]),
        "commonFiniteSamples": int(item["commonFiniteSamples"]),
        "analysisSamples": int(item["analysisSamples"]),
        "acquisitionProvenance": item.get("provenance"),
        "frozenDatasetPath": item.get("frozenDatasetPath"),
        "frozenDatasetSHA256": item.get("frozenDatasetSHA256"),
        "periodSearchPerformed": False,
    }


def _freeze_sector_input(item: dict[str, Any], root: Path) -> dict[str, Any]:
    sector = int(item["sector"])
    path = root / "inputs" / f"sector-{sector}-sap-pdcsap.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=np.asarray(item["time"], dtype=np.float64),
        sap=np.asarray(item["sap"], dtype=np.float64),
        pdcsap=np.asarray(item["pdcsap"], dtype=np.float64),
    )
    return {**item, "frozenDatasetPath": str(path.resolve()),
            "frozenDatasetSHA256": sha256_file(path)}


def run_period_family_time_domain_evolution(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Acquire, freeze, and analyze both official flux products per sector."""
    acquisition_errors: list[dict[str, Any]] = []
    if sector_inputs is None:
        inputs, acquisition_errors = _production_sector_inputs(preparation)
    else:
        supplied = {int(item["sector"]): item for item in sector_inputs}
        unexpected = set(supplied) - set(preparation["untouchedSectors"])
        if unexpected:
            raise RuntimeError(f"Supplied unregistered sectors: {sorted(unexpected)}")
        inputs = [supplied[sector] for sector in preparation["untouchedSectors"] if sector in supplied]
    root = Path(preparation["artifactRoot"])
    results = []
    errors = list(acquisition_errors)
    frozen = []
    for item in inputs:
        sector = int(item["sector"])
        try:
            frozen_item = _freeze_sector_input(item, root)
            frozen.append({
                "sector": sector,
                "path": frozen_item["frozenDatasetPath"],
                "sha256": frozen_item["frozenDatasetSHA256"],
                "source": frozen_item.get("provenance"),
            })
            results.append(analyze_sector_flux_pair(frozen_item, preparation))
        except Exception as error:
            errors.append({"sector": sector, "error": f"{type(error).__name__}: {error}"})
    return {
        "version": "openstar.tess-period-family-time-domain-evolution-run.v1",
        "execution": "coordinator-local-time-domain-analysis",
        "workerWorkload": None,
        "periodSearchPerformed": False,
        "sectorsRequested": list(preparation["untouchedSectors"]),
        "frozenDatasets": frozen,
        "sectorResults": results,
        "errors": errors,
    }


def interpret_period_family_time_domain_evolution(
    preparation: dict[str, Any], run: dict[str, Any],
) -> dict[str, Any]:
    sectors = list(run.get("sectorResults") or [])
    supporting = [item for item in sectors if item.get("classification") in {
        "STABLE_TIME_DOMAIN_RECURRENCE", "EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE",
        "TIME_DOMAIN_RECURRENCE_WAVEFORM_UNRESOLVED",
    }]
    stable = [item for item in supporting
              if item.get("classification") == "STABLE_TIME_DOMAIN_RECURRENCE"]
    evolving = [item for item in supporting
                if item.get("classification") == "EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE"]
    waveform_unresolved = [item for item in supporting
                           if item.get("classification") ==
                           "TIME_DOMAIN_RECURRENCE_WAVEFORM_UNRESOLVED"]
    disagreements = [item for item in sectors
                     if item.get("classification") == "SAP_PDCSAP_DISAGREEMENT"]
    nondetections = [item for item in sectors
                     if item.get("classification") == "TIME_DOMAIN_RECURRENCE_NOT_DETECTED"]
    campaigns = sorted({item["campaign"] for item in supporting})
    quality_campaigns = sorted({item["campaign"] for item in sectors})
    required_sectors = int(METHOD["minimumSupportingSectors"])
    required_campaigns = int(METHOD["minimumCampaignsForPersistentResult"])
    persistent = len(supporting) >= required_sectors and len(campaigns) >= required_campaigns
    consensus_lags = [float(item["consensusFamilyLagDays"]) for item in supporting
                      if _finite(item.get("consensusFamilyLagDays")) is not None]
    lag_range = max(consensus_lags) - min(consensus_lags) if consensus_lags else None

    alternate_agreements = []
    for item in sectors:
        sap_alternates = (item.get("sap") or {}).get("qualifyingAlternateRecurrencePeaks") or []
        pdcsap_alternates = (item.get("pdcsap") or {}).get("qualifyingAlternateRecurrencePeaks") or []
        for sap_peak in sap_alternates:
            matches = [peak for peak in pdcsap_alternates
                       if abs(float(peak["lagDays"]) - float(sap_peak["lagDays"]))
                       <= METHOD["sapPdcsapMaximumLagDifferenceDays"]]
            if not matches:
                continue
            pdcsap_peak = min(matches, key=lambda peak:
                              abs(float(peak["lagDays"]) - float(sap_peak["lagDays"])))
            alternate_agreements.append({
                "sector": int(item["sector"]),
                "lagDays": 0.5 * (float(sap_peak["lagDays"]) +
                                  float(pdcsap_peak["lagDays"])),
                "sapRelationship": sap_peak["relationship"],
                "pdcsapRelationship": pdcsap_peak["relationship"],
                "physicalModeClaim": False,
            })

    stable_lag = lag_range is not None and lag_range <= METHOD["crossSectorStableLagRangeDays"]
    if persistent and not evolving and not waveform_unresolved and not disagreements and stable_lag:
        classification = "PERSISTENT_STABLE_TIME_DOMAIN_RECURRENCE"
    elif persistent and not waveform_unresolved and not disagreements:
        classification = "PERSISTENT_EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE"
    elif persistent and not disagreements:
        classification = "PERSISTENT_TIME_DOMAIN_RECURRENCE_WAVEFORM_UNRESOLVED"
    elif len(disagreements) >= required_sectors:
        classification = "PIPELINE_DEPENDENT_TIME_DOMAIN_RESULT"
    elif (len(sectors) >= int(METHOD["minimumQualitySectorsForNonreplication"])
          and len(quality_campaigns) >= required_campaigns
          and len(supporting) < required_sectors):
        classification = "TIME_DOMAIN_RECURRENCE_NOT_REPLICATED"
    else:
        classification = "TIME_DOMAIN_EVOLUTION_UNRESOLVED"

    recommendations = {
        "PERSISTENT_STABLE_TIME_DOMAIN_RECURRENCE": "LONG_BASELINE_PHASE_STABILITY_TEST",
        "PERSISTENT_EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE": "ACTIVE_REGION_OR_CLOSE_MODE_DISCRIMINATION",
        "PERSISTENT_TIME_DOMAIN_RECURRENCE_WAVEFORM_UNRESOLVED": "ADDITIONAL_LONG_BASELINE_TIME_DOMAIN_DATA",
        "PIPELINE_DEPENDENT_TIME_DOMAIN_RESULT": "INDEPENDENT_PHOTOMETRY_PIPELINE_VALIDATION",
        "TIME_DOMAIN_RECURRENCE_NOT_REPLICATED": "PERIOD_FAMILY_MODEL_REASSESSMENT",
        "TIME_DOMAIN_EVOLUTION_UNRESOLVED": "ADDITIONAL_LONG_BASELINE_TIME_DOMAIN_DATA",
    }
    return {
        "version": "openstar.tess-period-family-time-domain-evolution-interpretation.v1",
        "classification": classification,
        "claimDecision": {
            "claim": "HUMAN_REVIEW_REQUIRED",
            "rationale": [
                "Time-domain recurrence tests persistence and waveform evolution, not physical mechanism.",
                "No classification promotes the candidate family to a solved rotation or orbital period.",
            ],
        },
        "sourceAttribution": "TARGET",
        "timeDomainFamilyReplicated": persistent,
        "waveformEvolutionOrComplexitySupported": classification ==
            "PERSISTENT_EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE",
        "stableWithinSectorRecurrenceSupported": classification ==
            "PERSISTENT_STABLE_TIME_DOMAIN_RECURRENCE",
        "stableClockSupported": False,
        "supportingSectors": sorted(int(item["sector"]) for item in supporting),
        "stableSectors": sorted(int(item["sector"]) for item in stable),
        "evolvingOrComplexSectors": sorted(int(item["sector"]) for item in evolving),
        "waveformUnresolvedSectors": sorted(int(item["sector"])
                                            for item in waveform_unresolved),
        "sapPdcsapDisagreementSectors": sorted(int(item["sector"]) for item in disagreements),
        "nonDetectionSectors": sorted(int(item["sector"]) for item in nondetections),
        "supportingCampaigns": campaigns,
        "qualityCampaigns": quality_campaigns,
        "requiredSupportingSectors": required_sectors,
        "requiredSupportingCampaigns": required_campaigns,
        "consensusFamilyLagRangeDays": lag_range,
        "maximumStableLagRangeDays": METHOD["crossSectorStableLagRangeDays"],
        "sapPdcsapAlternateRecurrenceAgreements": alternate_agreements,
        "sectorResults": sectors,
        "errors": list(run.get("errors") or []),
        "periodSearchPerformed": False,
        "periodFamilyResolved": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommendations[classification],
        "interpretationGuard": (
            "Persistent recurrence is a detection-level result only. Stable recurrence can be "
            "consistent with an orbital clock or long-lived activity; evolution can be consistent "
            "with active regions, differential rotation, beating, or other nonstationarity."
        ),
        "preparationSHA256": sha256_json(preparation),
        "runSHA256": sha256_json(run),
    }
