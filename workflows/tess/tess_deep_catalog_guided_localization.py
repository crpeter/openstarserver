"""Frozen deep-catalog, multi-source official-PRF localization.

The SkyMapper/NSC rows are scientific hypotheses and never enter a generic
worker payload.  This coordinator-local continuation fits every bounded source
subset at the persisted residual frequency.  It is sub-pixel TESS PRF
localization constrained by higher-resolution catalog coordinates, not a claim
that TESS has acquired new high-angular-resolution observations.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from .tess_catalog_guided_localization import (
    _production_sector_inputs,
    _write_json,
    analyze_generalized_catalog_guided_sector,
    generate_source_hypotheses,
)


METHOD_VERSION = "openstar.tess-deep-catalog-guided-prf-localization.v1"
PREPARE_HANDLER_ID = "openstar.tess.deep-catalog-guided-prf-localization.prepare"
RUN_HANDLER_ID = "openstar.tess.deep-catalog-guided-prf-localization.run"
INTERPRET_HANDLER_ID = "openstar.tess.deep-catalog-guided-prf-localization.interpret"
CURRENT_TRIGGER = "HIGH_RESOLUTION_RESIDUAL_SOURCE_LOCALIZATION"
MIN_CANDIDATES = 2
MAX_CANDIDATES = 5


def validate_deep_catalog_boundary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept only the exact successful ambiguous PR182 boundary, without truncation."""
    candidates = list(summary.get("plausibleCatalogCandidates") or [])
    if not (
        summary.get("version")
        == "openstar.tess-deep-catalog-counterpart-identification.v1"
        and summary.get("classification") == "AMBIGUOUS_DEEP_CATALOG_COUNTERPARTS"
        and summary.get("counterpartIdentified") is False
        and summary.get("preferredCandidate") is None
        and MIN_CANDIDATES <= len(candidates) <= MAX_CANDIDATES
        and summary.get("variabilityConfirmed") is False
        and summary.get("physicalMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and summary.get("externalDataState") == "AVAILABLE"
        and not (summary.get("queryErrors") or [])
        and summary.get("recommendedNextTest") == CURRENT_TRIGGER
    ):
        raise RuntimeError(
            "Deep-catalog PRF localization requires the exact available, ambiguous PR182 boundary."
        )
    frozen = []
    coordinates = set()
    for candidate in candidates:
        try:
            ra = float(candidate["raDeg"])
            dec = float(candidate["decDeg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("A deep-catalog candidate lacks finite ICRS coordinates.") from exc
        coordinate = (ra, dec)
        if (
            not math.isfinite(ra) or not math.isfinite(dec)
            or not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0
            or candidate.get("isTarget") is not False
            or candidate.get("variabilityConfirmed") is not False
            or coordinate in coordinates
        ):
            raise RuntimeError("Deep-catalog candidate coordinates are invalid or duplicated.")
        coordinates.add(coordinate)
        frozen.append(copy.deepcopy(candidate))
    return frozen


def prepare_deep_catalog_guided_localization(
    *, deep_catalog_summary: dict[str, Any], prf_preparation: dict[str, Any],
    output_dir: Path, investigation_id: str,
) -> dict[str, Any]:
    candidates = validate_deep_catalog_boundary(deep_catalog_summary)
    target_sky = prf_preparation.get("targetSky") or {}
    sectors = list(prf_preparation.get("sectors") or [])
    harmonic_orders = tuple(
        int(value) for value in prf_preparation.get("subtractedHarmonicOrders") or []
    )
    required_numbers = (
        prf_preparation.get("referenceFamilyPeriodDays", prf_preparation.get("physicalPeriodDays")),
        prf_preparation.get("residualReferenceFrequency"),
        prf_preparation.get("residualTimeReferenceDays"),
        prf_preparation.get("fractionalFrequencyDriftPerDay"),
        target_sky.get("raDeg"), target_sky.get("decDeg"),
    )
    try:
        finite = all(math.isfinite(float(value)) for value in required_numbers)
    except (TypeError, ValueError):
        finite = False
    if not (
        finite
        and prf_preparation.get("version")
        == "openstar.tess-prf-deblending.v1"
        and prf_preparation.get("modelSource")
        == "official-public-SPOC-TESS-PRF-FITS"
        and sectors and len(sectors) == len(set(sectors))
        and all(int(sector) > 0 for sector in sectors)
        and prf_preparation.get("ticID") is not None
        and harmonic_orders
        and len(harmonic_orders) == len(set(harmonic_orders))
        and all(order > 0 for order in harmonic_orders)
        and float(required_numbers[0]) > 0.0
        and float(required_numbers[1]) > 0.0
        and abs(float(required_numbers[3])) <= 1e-15
    ):
        raise RuntimeError("Deep-catalog localization requires complete frozen official-PRF evidence.")

    component_ids = ["target", *[f"candidate-{index}" for index in range(1, len(candidates) + 1)]]
    hypotheses = generate_source_hypotheses(component_ids)
    root = Path(output_dir) / "deep-catalog-guided-prf-localization"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": METHOD_VERSION,
        "investigationID": investigation_id,
        "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-bounded-multi-source-official-prf-fit",
        "methodScope": "TESS_PRF_WITH_FROZEN_HIGHER_RESOLUTION_CATALOG_COORDINATES",
        "ticID": prf_preparation["ticID"],
        "target": copy.deepcopy(prf_preparation.get("target")),
        "targetSky": copy.deepcopy(target_sky),
        "catalogCandidates": candidates,
        "componentIDs": component_ids,
        "candidateComponentMap": [
            {"componentID": component_ids[index], "catalogCandidate": candidate}
            for index, candidate in enumerate(candidates, start=1)
        ],
        "modelHypothesisCount": len(hypotheses),
        "sectors": sectors,
        "referenceFamilyPeriodDays": float(required_numbers[0]),
        "residualReferenceFrequency": float(required_numbers[1]),
        "residualTimeReferenceDays": float(required_numbers[2]),
        "fractionalFrequencyDriftPerDay": float(required_numbers[3]),
        "subtractedHarmonicOrders": list(harmonic_orders),
        "crossSectorPhaseUsed": False,
        "historicalResidualDriftExtrapolated": False,
        "catalogQueriesRepeated": False,
        "officialPRFCacheRoot": str(
            Path(prf_preparation["artifactRoot"]) / "official-prf-cache"
        ),
        "physicalCycleResolved": False,
        "priorEvidence": {
            "deepCatalogCounterpart": copy.deepcopy(deep_catalog_summary),
            "officialPRFDeblendingPreparation": copy.deepcopy(prf_preparation),
        },
    }
    _write_json(Path(result["preparationPath"]), result)
    return result


def run_deep_catalog_guided_localization(
    preparation: dict[str, Any], *, sector_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    orders = tuple(int(value) for value in preparation["subtractedHarmonicOrders"])
    if sector_inputs is None:
        sector_inputs = _production_sector_inputs(preparation, harmonic_orders=orders)
    results = []
    for item in sector_inputs:
        renderer = item.get("renderTemplates")
        if not callable(renderer):
            raise RuntimeError("Each sector input requires an official-SPOC PRF renderer.")
        result = analyze_generalized_catalog_guided_sector(
            sector=int(item["sector"]), times=item["times"],
            prewhitened=item["prewhitened"], valid=item["valid"],
            calibration_image=item["calibrationImage"],
            background_columns=item["backgroundColumns"], render_templates=renderer,
            candidate_frequency=float(preparation["residualReferenceFrequency"]),
            original_time_origin=float(preparation["residualTimeReferenceDays"]),
            physical_frequency=1.0 / float(preparation["referenceFamilyPeriodDays"]),
            component_ids=preparation["componentIDs"],
            block_count=int(item.get("blockCount", 4)), harmonic_orders=orders,
        )
        result["acquisitionProvenance"] = item.get("acquisitionProvenance")
        results.append(result)
    return {
        "version": "openstar.tess-deep-catalog-guided-prf-localization-run.v1",
        "execution": preparation["execution"],
        "sectorResults": results,
        "catalogQueriesRepeated": False,
        "physicalCycleResolved": False,
    }


def interpret_deep_catalog_guided_localization(
    preparation: dict[str, Any], run: dict[str, Any],
) -> dict[str, Any]:
    sectors = copy.deepcopy(list(run.get("sectorResults") or []))
    decisive = []
    for sector in sectors:
        full = sector.get("fullDataComparison") or {}
        temporal = sector.get("temporalPredictiveValidation") or {}
        full_sources = list(full.get("bestModelSourceIDs") or [])
        predictive_sources = list(temporal.get("predictiveModelSourceIDs") or [])
        compatibility = temporal.get("sourceVectorTemporalCompatibility") or {}
        criteria = {
            "completeModelFullRank": full.get("completeModelFullRank") is True,
            "fullDataModelIdentifiable": full.get("bestModelIdentifiable") is True,
            "aggregateFrozenHeldOutSupportsFullDataSources": predictive_sources == full_sources,
            "relevantSourceVectorTemporalCompatibilityPasses": compatibility.get("compatible") is True,
            "scientificallyValid": sector.get("scientificallyValid") is True,
        }
        sector["decisivenessCriteria"] = criteria
        sector["decisive"] = all(criteria.values())
        sector["attributedCandidateComponentIDs"] = [
            value for value in full_sources if value != "target"
        ]
        if sector["decisive"]:
            decisive.append(sector)

    all_decisive = bool(sectors) and len(decisive) == len(sectors)
    candidate_sets = {
        tuple(item["attributedCandidateComponentIDs"]) for item in decisive
    }
    stable_component = None
    if all_decisive and len(candidate_sets) == 1:
        only = next(iter(candidate_sets))
        if len(only) == 1:
            stable_component = only[0]
    switching = bool(all_decisive and stable_component is None and len(candidate_sets) > 1)
    target_only = bool(all_decisive and candidate_sets == {()})

    candidate_map = {
        item["componentID"]: item["catalogCandidate"]
        for item in preparation["candidateComponentMap"]
    }
    preferred = copy.deepcopy(candidate_map.get(stable_component))
    if preferred is not None:
        classification = "DEEP_CATALOG_RESIDUAL_SOURCE_LOCALIZED"
        next_test = "INDEPENDENT_DEEP_COUNTERPART_VARIABILITY_VALIDATION"
    elif switching:
        classification = "DEEP_CATALOG_RESIDUAL_SOURCE_SWITCHING_OR_BLEND"
        next_test = "DEDICATED_HIGH_RESOLUTION_TIME_SERIES_IMAGING"
    elif target_only:
        classification = "TARGET_CONSISTENT_DEEP_CATALOG_HYPOTHESES_REJECTED"
        next_test = "RESIDUAL_SOURCE_ATTRIBUTION_RECONCILIATION"
    else:
        classification = "DEEP_CATALOG_PRF_LOCALIZATION_UNRESOLVED"
        next_test = "DEDICATED_HIGH_RESOLUTION_TIME_SERIES_IMAGING"
    return {
        "version": "openstar.tess-deep-catalog-guided-prf-localization-interpretation.v1",
        "methodScope": preparation["methodScope"],
        "classification": classification,
        "sectorResults": sectors,
        "sectorCount": len(sectors),
        "decisiveSectorCount": len(decisive),
        "allSectorsDecisive": all_decisive,
        "stableComponentID": stable_component,
        "preferredCandidate": preferred,
        "catalogCandidates": copy.deepcopy(preparation["catalogCandidates"]),
        "sourceAttributionResolved": preferred is not None,
        "variabilityConfirmed": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "claimLevelChanged": preferred is not None,
        "catalogQueriesRepeated": False,
        "recommendedNextTest": next_test,
        "interpretationGuard": (
            "Catalog-constrained TESS PRF evidence is not external high-resolution imaging; "
            "unresolved or changing sector attribution must remain unresolved."
        ),
    }
