"""Append-only localization of a frozen residual in unused official sectors."""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from .tess_catalog_guided_localization import (
    COMPONENT_IDS,
    _production_sector_inputs,
    analyze_generalized_catalog_guided_sector,
)
from .tess_residual_localization import _write_json
from .tess_sector_archive import TessArchiveTransientError

AUTHORIZATION = "ADDITIONAL_INDEPENDENT_SOURCE_LOCALIZATION_DATA"
PRESERVED = (
    "ticID",
    "referenceFamilyPeriodDays",
    "residualReferenceFrequency",
    "residualTimeReferenceDays",
    "fractionalFrequencyDriftPerDay",
    "subtractedHarmonicOrders",
    "catalogCandidates",
    "targetSky",
)


def unused_official_sectors(
    identity: dict[str, Any], bridge: dict[str, Any]
) -> list[int]:
    official = {
        int(v) for v in ((identity.get("tess") or {}).get("officialSectors") or [])
    }
    used = {int(v) for v in (bridge.get("sectors") or [])}
    return sorted(official - used)


def boundary_authorized(result: dict[str, Any]) -> bool:
    return (
        result.get("classification") == "UNRESOLVED"
        and result.get("recommendedNextTest") == AUTHORIZATION
        and result.get("sourceAttributionResolved") is False
        and result.get("physicalMechanismResolved") is False
    )


def bridge_is_complete(bridge: dict[str, Any]) -> bool:
    try:
        return (
            all(key in bridge and bridge[key] is not None for key in PRESERVED)
            and len(bridge["catalogCandidates"]) == 2
            and bool(bridge.get("sectors"))
            and float(bridge["fractionalFrequencyDriftPerDay"]) == 0.0
            and list(bridge["subtractedHarmonicOrders"]) == [1, 2, 3, 4]
        )
    except (TypeError, ValueError):
        return False


def prepare_additional_sector_source_localization(
    *, interpretation, localization_bridge, identity, output_dir, investigation_id
):
    if not boundary_authorized(interpretation):
        raise RuntimeError("Interpretation does not authorize continuation.")
    if not bridge_is_complete(localization_bridge):
        raise RuntimeError(
            "Frozen localization bridge is incomplete or unsafe for old sectors."
        )
    if float(localization_bridge["fractionalFrequencyDriftPerDay"]) != 0.0:
        raise RuntimeError(
            "Old-sector localization refuses to extrapolate nonzero residual drift."
        )
    sectors = unused_official_sectors(identity, localization_bridge)
    if not sectors:
        raise RuntimeError("No unused official TESS sectors are available.")
    root = Path(output_dir) / "additional-sector-source-localization"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "version": "openstar.tess-additional-sector-source-localization-preparation.v1",
        "investigationID": investigation_id,
        "artifactRoot": str(root.resolve()),
        "preparationPath": str((root / "preparation.json").resolve()),
        "execution": "coordinator-local-official-spoc-tpf-prf-fit",
        "componentIDs": list(COMPONENT_IDS),
        "sectors": sectors,
        "sectorSelection": [
            {
                "sector": s,
                "reason": "official TESS sector absent from frozen residual-localization bridge",
            }
            for s in sectors
        ],
        "usedFrozenBridgeSectors": sorted(
            {int(v) for v in localization_bridge["sectors"]}
        ),
        "authorizationStageResult": copy.deepcopy(interpretation),
    }
    result.update({k: copy.deepcopy(localization_bridge[k]) for k in PRESERVED})
    _write_json(Path(result["preparationPath"]), result)
    return result


def run_additional_sector_source_localization(preparation, *, sector_inputs=None):
    if float(preparation["fractionalFrequencyDriftPerDay"]) != 0.0:
        raise RuntimeError(
            "Nonzero historical drift cannot be extrapolated into old sectors."
        )
    orders = tuple(preparation["subtractedHarmonicOrders"])
    supplied = {int(i["sector"]): i for i in sector_inputs or []}
    results = []
    for sector in preparation["sectors"]:
        try:
            item = supplied.get(int(sector))
            if item is None:
                one = dict(preparation)
                one["sectors"] = [int(sector)]
                item = _production_sector_inputs(one, harmonic_orders=orders)[0]
            if not callable(item.get("renderTemplates")):
                raise RuntimeError("official-SPOC PRF renderer unavailable")
            result = analyze_generalized_catalog_guided_sector(
                sector=int(sector),
                times=item["times"],
                prewhitened=item["prewhitened"],
                valid=item["valid"],
                calibration_image=item["calibrationImage"],
                background_columns=item["backgroundColumns"],
                render_templates=item["renderTemplates"],
                candidate_frequency=float(preparation["residualReferenceFrequency"]),
                original_time_origin=float(preparation["residualTimeReferenceDays"]),
                physical_frequency=1 / float(preparation["referenceFamilyPeriodDays"]),
                component_ids=COMPONENT_IDS,
                harmonic_orders=orders,
                block_count=int(item.get("blockCount", 4)),
            )
            result["acquisitionProvenance"] = item.get("acquisitionProvenance")
        except (
            TessArchiveTransientError,
            ConnectionError,
            TimeoutError,
        ) as exc:
            result = {
                "sector": int(sector),
                "availability": "UNAVAILABLE",
                "scientificallyValid": False,
                "reason": str(exc),
            }
        except RuntimeError as exc:
            unavailable = (
                str(exc).startswith("No official TPF or TESScut coverage available")
                or str(exc).startswith("Official TPF download returned no data")
                or str(exc).startswith("TESScut download returned no data")
            )
            if not unavailable:
                raise
            result = {
                "sector": int(sector),
                "availability": "UNAVAILABLE",
                "scientificallyValid": False,
                "reason": str(exc),
            }
        results.append(result)
    return {
        "version": "openstar.tess-additional-sector-source-localization-run.v1",
        "execution": "coordinator-local",
        "sectorResults": results,
    }


def _sector_classification(sector):
    if sector.get("availability") == "UNAVAILABLE":
        return "UNAVAILABLE", []
    full = sector.get("fullDataComparison") or {}
    pred = sector.get("temporalPredictiveValidation") or {}
    if (
        not sector.get("calibrationResolved")
        or not sector.get("scientificallyValid")
        or not full.get("completeModelFullRank")
    ):
        return "SCIENTIFICALLY_INVALID", []
    sources = list(full.get("bestModelSourceIDs") or [])
    valid = (
        full.get("bestModelIdentifiable") is True
        and pred.get("predictiveSupport") is True
        and pred.get("predictiveModel") == full.get("bestModel")
        and pred.get("sourceVectorTemporalCompatibility", {}).get("compatible") is True
    )
    if not valid:
        return "UNRESOLVED", []
    if not sources:
        return "UNRESOLVED", []
    if len(sources) == 1:
        return "UNIQUE_SOURCE_SUPPORTED", sources
    return "MULTIPLE_SOURCES_SUPPORTED", sources


def interpret_additional_sector_source_localization(preparation, run):
    rows = copy.deepcopy(list(run.get("sectorResults") or []))
    unique = []
    multi = 0
    for row in rows:
        label, sources = _sector_classification(row)
        row["sourceSupportClassification"] = label
        row["supportedSourceIDs"] = sources
        if label == "UNIQUE_SOURCE_SUPPORTED":
            unique.append(sources[0])
        elif label == "MULTIPLE_SOURCES_SUPPORTED":
            multi += 1
    counts = {s: unique.count(s) for s in COMPONENT_IDS}
    repeated = {s for s, n in counts.items() if n >= 2}
    winner = next(
        (s for s, n in counts.items() if n >= 3 and n > len(unique) / 2), None
    )
    if len(repeated) >= 2 or multi >= 2:
        classification, winner = "SOURCE_SWITCHING_OR_BLEND", None
    elif winner == "target":
        classification = "TARGET_SUPPORTED"
    elif winner in {"candidate-1", "candidate-2"}:
        classification = "CATALOG_SOURCE_SUPPORTED"
    else:
        classification, winner = "UNRESOLVED", None
    resolved = classification in {"TARGET_SUPPORTED", "CATALOG_SOURCE_SUPPORTED"}
    next_tests = {
        "TARGET_SUPPORTED": "TARGET_INTRINSIC_RESIDUAL_MODELING",
        "CATALOG_SOURCE_SUPPORTED": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        "SOURCE_SWITCHING_OR_BLEND": "SOURCE_SWITCHING_PHYSICAL_MECHANISM_MODELING",
        "UNRESOLVED": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY",
    }
    preferred = (
        copy.deepcopy(preparation["catalogCandidates"][int(winner[-1]) - 1])
        if winner in {"candidate-1", "candidate-2"}
        else None
    )
    return {
        "version": "openstar.tess-additional-sector-source-localization-interpretation.v1",
        "classification": classification,
        "sectorResults": rows,
        "uniqueSourceSupportCounts": counts,
        "multipleSourceSupportCount": multi,
        "supportedSourceID": winner,
        "preferredCandidate": preferred,
        "catalogCandidates": copy.deepcopy(preparation["catalogCandidates"]),
        "sourceAttributionResolved": resolved,
        "physicalMechanismResolved": False,
        "recommendedNextTest": next_tests[classification],
        "claimLevelChanged": resolved,
    }
