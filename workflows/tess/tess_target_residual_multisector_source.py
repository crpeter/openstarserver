"""v20.19 catalog-constrained localization in unused recurrence sectors.

Only frozen v20.17/v20.18 evidence enters preparation.  Pixel acquisition and
the scientific fit are coordinator responsibilities; generic workers remain
unaware of TESS or catalog sources.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Iterable

from openstar_path_relocation import HistoricalPathResolver, NO_HISTORICAL_PATH_RELOCATION
from openstar_investigation import sha256_json
from .tess_difference_image_constants import SOURCE_MATCH_MAX_PIXELS
from .tess_target_residual_archival_baseline import verified_json_result
from .tess_target_residual_pixel_recurrence import verify_v2017_lineage

PREFIX = "openstar.tess.target-residual-multisector-source."
MAX_COMPETING_SOURCES = 8
MAX_ADDITIONAL_SOURCE_LOCALIZATION_SECTORS = 12
V2018_SECTOR_IDS = (2, 65, 33, 13, 27, 39)


def verify_v2018_lineage(stages: Iterable[Any], *, resolver: HistoricalPathResolver | None = None
                         ) -> dict[str, Any]:
    """Verify the complete immutable v20.17 -> v20.18 terminal boundary."""
    rows = list(stages)
    resolver = resolver or NO_HISTORICAL_PATH_RELOCATION
    v17 = verify_v2017_lineage(rows, resolver=resolver)
    expected = (
        ("036-target-residual-pixel-recurrence-prepare", PREFIX.replace("multisector-source", "pixel-recurrence") + "prepare"),
        ("037-target-residual-pixel-recurrence-run", PREFIX.replace("multisector-source", "pixel-recurrence") + "run"),
        ("038-target-residual-pixel-recurrence-interpret", PREFIX.replace("multisector-source", "pixel-recurrence") + "interpret"),
        ("039-finalize", "openstar.tess.finalize"))
    found = []
    for stage_id, handler in expected:
        matches = [stage for stage in rows if stage.id == stage_id]
        if len(matches) != 1 or matches[0].handler_id != handler or matches[0].status != "COMPLETE":
            raise RuntimeError(f"invalid v20.18 stage {stage_id}")
        found.append(matches[0])
    if found[-1] is not rows[-1] or found[0].triggered_by_stage_id != v17["finalizer"].id:
        raise RuntimeError("v20.18 is not the final, contiguous persisted boundary")
    if any(found[index].triggered_by_stage_id != found[index - 1].id for index in range(1, 4)):
        raise RuntimeError("invalid v20.18 trigger chain")
    prepare, run, science, finalizer = found
    verified_json_result(prepare, "target-residual-pixel-recurrence-prepare-v20.18.json", resolver=resolver)
    verified_json_result(run, "target-residual-pixel-recurrence-run-v20.18.json", resolver=resolver)
    verified_json_result(science, "target-residual-pixel-recurrence-v20.18.json", resolver=resolver)
    verified_json_result(finalizer, "conclusion-v20.18-target-residual-pixel-recurrence-validation.json", resolver=resolver)
    hashes = science.provenance.input_hashes if science.provenance else {}
    if hashes.get("preparation") != sha256_json(prepare.result) or hashes.get("run") != sha256_json(run.result):
        raise RuntimeError("v20.18 interpretation input binding is damaged")
    required = {"classification": "PIXEL_RECURRENCE_LOCALIZATION_UNRESOLVED",
        "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
        "sourceAttributionResolved": False, "physicalMechanismResolved": False,
        "crossSectorPhaseUsed": False, "historicalResidualDriftExtrapolated": False}
    if any(science.result.get(key) != value for key, value in required.items()):
        raise RuntimeError("altered v20.18 science boundary")
    if (finalizer.parameters != {"outputSuffix": "v20.18-target-residual-pixel-recurrence-validation"}
            or finalizer.result.get("targetResidualPixelRecurrenceValidation") != science.result):
        raise RuntimeError("altered v20.18 finalizer")
    return {"v20.17": v17, "prepare": prepare, "run": run,
            "science": science, "finalizer": finalizer}


def derive_competing_sources(catalog_hypotheses: list[dict[str, Any]],
                             sector_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select frozen hypotheses within the match radius in any quality sector."""
    qualifying = set()
    for sector in sector_results:
        if sector.get("classification") == "UNAVAILABLE":
            continue
        distances = sector.get("distancesPixels") or {}
        for source_id, distance in distances.items():
            if isinstance(distance, (int, float)) and math.isfinite(float(distance)) \
                    and float(distance) <= SOURCE_MATCH_MAX_PIXELS:
                qualifying.add(str(source_id))
    selected = [copy.deepcopy(row) for row in catalog_hypotheses
                if str(row.get("sourceID")) in qualifying]
    if len(selected) > MAX_COMPETING_SOURCES:
        raise RuntimeError("more than eight frozen catalog competitors; refusing truncation")
    if not selected:
        raise RuntimeError("no frozen catalog competitors satisfy the source-match rule")
    return selected


def derive_additional_sectors(sector_evidence: list[dict[str, Any]],
                              excluded: Iterable[int] = V2018_SECTOR_IDS) -> list[dict[str, Any]]:
    excluded_ids = {int(value) for value in excluded}
    eligible = []
    for row in sector_evidence:
        frequency = row.get("candidateFrequency")
        if (row.get("supportsHistoricalResidualFamily") is True
                and row.get("recurrenceClassification") == "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY"
                and int(row["sector"]) not in excluded_ids
                and isinstance(frequency, (int, float)) and math.isfinite(float(frequency))
                and float(frequency) > 0):
            eligible.append(copy.deepcopy(row))
    eligible.sort(key=lambda row: int(row["sector"]))
    for row in eligible:
        row["selectionReason"] = "unused-v20.17-supporting-sector-with-frozen-positive-frequency"
    if len(eligible) <= MAX_ADDITIONAL_SOURCE_LOCALIZATION_SECTORS:
        return eligible
    # Preregistered deterministic coverage: strongest recurrence evidence, then sector ID.
    ranked = sorted(eligible, key=lambda row: (-float(row.get("recurrenceScore", 0.0)), int(row["sector"])))
    chosen = ranked[:MAX_ADDITIONAL_SOURCE_LOCALIZATION_SECTORS]
    for row in chosen:
        row["selectionReason"] = "bounded-by-descending-frozen-recurrence-score-then-sector-id"
    return chosen


def classify_sector_model(model_evidence: dict[str, Any]) -> dict[str, Any]:
    if model_evidence.get("availability") == "UNAVAILABLE":
        return {"classification": "UNAVAILABLE", "supportedSources": []}
    full = model_evidence.get("fullDataComparison") or {}
    predictive = model_evidence.get("temporalPredictiveValidation") or {}
    sources = list(full.get("bestModelSourceIDs") or [])
    conditional = set(full.get("conditionallyIdentifiableSources") or [])
    supported = [source for source in sources if source in conditional]
    valid = bool(full.get("bestModelIdentifiable") and full.get("completeModelFullRank")
                 and predictive.get("predictiveSupport")
                 and predictive.get("predictiveModel") == full.get("bestModel"))
    classification = ("UNRESOLVED" if not valid else
        "UNIQUE_SOURCE_SUPPORTED" if len(supported) == 1 and len(sources) == 1 else
        "MULTIPLE_SOURCES_SUPPORTED" if len(supported) >= 2 else "UNRESOLVED")
    return {"classification": classification, "supportedSources": supported}


def interpret_multisector(sectors: list[dict[str, Any]], target_source_id: str) -> dict[str, Any]:
    rows = []
    for sector in sectors:
        classified = classify_sector_model(sector)
        rows.append({**sector, **classified})
    unique = [row for row in rows if row["classification"] == "UNIQUE_SOURCE_SUPPORTED"]
    counts: dict[str, list[int]] = {}
    for row in unique:
        counts.setdefault(row["supportedSources"][0], []).append(int(row["sector"]))
    repeated = [source for source, sector_ids in counts.items() if len(sector_ids) >= 2]
    multis = [row for row in rows if row["classification"] == "MULTIPLE_SOURCES_SUPPORTED"]
    blends: dict[tuple[str, ...], int] = {}
    for row in multis:
        key = tuple(row["supportedSources"]); blends[key] = blends.get(key, 0) + 1
    winner = max(counts, key=lambda source: len(counts[source])) if counts else None
    resolved = bool(winner and len(counts[winner]) >= 3
                    and len(counts[winner]) > len(unique) / 2 and len(repeated) < 2)
    switching = len(repeated) >= 2 or any(value >= 2 for value in blends.values())
    decision = ("TARGET_SUPPORTED" if resolved and winner == target_source_id else
                "CATALOG_SOURCE_SUPPORTED" if resolved else
                "SOURCE_SWITCHING_OR_BLEND" if switching else "UNRESOLVED")
    recommendation = {"TARGET_SUPPORTED": "ARCHIVAL_RECURRENCE_INFORMED_TARGET_MECHANISM_MODELING",
        "CATALOG_SOURCE_SUPPORTED": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        "SOURCE_SWITCHING_OR_BLEND": "SOURCE_SWITCHING_TEMPORAL_MODEL",
        "UNRESOLVED": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"}[decision]
    return {"classification": decision,
        "validSectorCount": sum(row["classification"] != "UNAVAILABLE" for row in rows),
        "uniquelyResolvedSectorCount": len(unique),
        "targetSupportingSectors": counts.get(target_source_id, []),
        "supportByCatalogSource": {source: ids for source, ids in counts.items() if source != target_source_id},
        "multiSourceSupportingSectors": [int(row["sector"]) for row in multis],
        "unresolvedSectors": [int(row["sector"]) for row in rows if row["classification"] == "UNRESOLVED"],
        "unavailableSectors": [int(row["sector"]) for row in rows if row["classification"] == "UNAVAILABLE"],
        "sourceSupportTable": counts, "sectorResults": rows, "preferredSource": winner if resolved else None,
        "sourceAttributionResolved": resolved or switching, "physicalMechanismResolved": False,
        "crossSectorPhaseUsed": False, "historicalResidualDriftExtrapolated": False,
        "recommendedNextTest": recommendation}


def run_multisector_source_localization(preparation: dict[str, Any], *, sector_inputs=None
                                        ) -> dict[str, Any]:
    """Execute supplied/acquired sector inputs with the generalized PRF comparison.

    Acquisition is deliberately injected at this narrow boundary in tests and by
    the archive adapter.  This function never performs a period search and checks
    every input against its persisted sector clock.
    """
    from .tess_catalog_guided_localization import compare_source_hypotheses
    if sector_inputs is None:
        raise RuntimeError("v20.19 requires the official TPF/PRF archive acquisition adapter")
    frozen = {int(row["sector"]): row for row in preparation["additionalSectorEvidence"]}
    source_ids = [row["sourceID"] for row in preparation["catalogHypotheses"]]
    results = []
    for item in sector_inputs:
        sector = int(item["sector"])
        if sector not in frozen or sector in preparation["excludedV2018SectorIDs"]:
            raise RuntimeError("sector input was not preregistered for v20.19")
        if item.get("availability") == "UNAVAILABLE":
            results.append({"sector": sector, "availability": "UNAVAILABLE",
                            "acquisitionProvenance": item.get("acquisitionProvenance")})
            continue
        comparison = compare_source_hypotheses(item["coefficients"], item["covariances"],
                                               item["templates"], source_ids)
        results.append({"sector": sector, "candidateFrequencyUsed": frozen[sector]["candidateFrequency"],
            "establishedFamilyPrewhitening": {"frequency": preparation["establishedPhysicalFamilyFrequency"],
                "harmonicOrders": [1, 2], "sectorLocalIntercept": True, "sectorLocalTrend": True},
            "fullDataComparison": comparison,
            "temporalPredictiveValidation": copy.deepcopy(item["temporalPredictiveValidation"]),
            "acquisitionProvenance": copy.deepcopy(item.get("acquisitionProvenance")),
            "crossSectorPhaseUsed": False, "historicalResidualDriftExtrapolated": False})
    return {"version": "openstar.tess-target-residual-multisector-source-run.v1",
            "sectorResults": results, "crossSectorPhaseUsed": False,
            "historicalResidualDriftExtrapolated": False}
