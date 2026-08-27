"""Target-independent admission and sector selection for period-family follow-up.

This module contains policy only.  It never downloads or examines flux; that is
important because selection must be frozen before an archive product is opened.
"""
from __future__ import annotations

from typing import Any, Iterable

from openstar_investigation import Investigation, InvestigationStore, sha256_json


DIFFERENCE_TRIGGER = "PERIOD_FAMILY_DIFFERENCE_IMAGE_LOCALIZATION"
TIME_DOMAIN_TRIGGER = "UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION"
EXTERNAL_TRIGGER = "ADDITIONAL_LONG_BASELINE_TIME_DOMAIN_DATA"
SUPPORTED_PRODUCT = "SPOC_LIGHTCURVE"
SUPPORTED_CADENCE_SECONDS = 120


def latest_semantic_result(investigation: Investigation) -> tuple[Any, dict[str, Any]] | None:
    """Return the newest complete result carrying a recognized follow-up trigger."""
    for stage in reversed(investigation.stages):
        result = stage.result or {}
        if stage.status == "COMPLETE" and result.get("recommendedNextTest") in {
            DIFFERENCE_TRIGGER, TIME_DOMAIN_TRIGGER, EXTERNAL_TRIGGER,
        }:
            return stage, result
    return None


def verify_semantic_boundary(store: InvestigationStore, investigation: Investigation) -> dict[str, Any]:
    """Verify all authoritative completed ledgers and return target-independent evidence."""
    found = latest_semantic_result(investigation)
    if found is None:
        raise RuntimeError("No persisted period-family follow-up recommendation.")
    stage, result = found
    hashes = {}
    for predecessor in investigation.stages[: investigation.stages.index(stage) + 1]:
        if predecessor.status != "COMPLETE" or predecessor.result is None:
            continue
        digest = store.verified_terminal_stage_ledger_hash(investigation.id, predecessor)
        if digest is None:
            raise RuntimeError(f"Authoritative ledger verification failed: {predecessor.id}")
        hashes[predecessor.id] = digest
    if result.get("periodFamilyResolved") is not False:
        raise RuntimeError("The period family is not explicitly unresolved.")
    if result.get("claimDecision", {}).get("claim", result.get("claim")) != "HUMAN_REVIEW_REQUIRED":
        raise RuntimeError("Follow-up requires a conservative human-review claim.")
    return {"triggerStageID": stage.id, "trigger": result["recommendedNextTest"],
            "result": result, "ledgerSHA256": hashes,
            "boundarySHA256": sha256_json(result)}


def select_untouched_sectors(catalog_sectors: Iterable[dict[str, Any]],
                             consumed_sectors: Iterable[int], *,
                             minimum_epochs: int = 3,
                             maximum_sectors: int = 12) -> dict[str, Any]:
    """Preregister a deterministic epoch-separated SPOC 120-second selection.

    ``epoch`` is official persisted catalog metadata (normally a TESS cycle or
    campaign), not a value inferred from flux.  One sector per epoch is chosen
    first, then remaining eligible sectors are filled in numeric order.
    """
    consumed = {int(x) for x in consumed_sectors}
    eligible, rejected = [], []
    for raw in sorted(catalog_sectors, key=lambda x: int(x["sector"])):
        sector = int(raw["sector"])
        reason = None
        if sector in consumed:
            reason = "already-consumed-by-time-domain-observable"
        elif raw.get("product") != SUPPORTED_PRODUCT:
            reason = "unsupported-archive-product"
        elif int(raw.get("cadenceSeconds") or 0) != SUPPORTED_CADENCE_SECONDS:
            reason = "unsupported-cadence"
        elif raw.get("available") is not True:
            reason = "product-not-explicitly-available"
        elif not str(raw.get("epoch") or ""):
            reason = "missing-official-epoch-identity"
        if reason:
            rejected.append({"sector": sector, "reason": reason})
        else:
            eligible.append({**raw, "sector": sector, "epoch": str(raw["epoch"])})
    by_epoch: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_epoch.setdefault(item["epoch"], []).append(item)
    if len(by_epoch) < minimum_epochs:
        return {"status": "INSUFFICIENT_EPOCH_COVERAGE", "selectedSectors": [],
                "rejectedSectors": rejected, "eligibleEpochs": sorted(by_epoch),
                "selectionRule": "lowest-sector-per-official-epoch-then-numeric-fill",
                "fluxInspectedDuringSelection": False}
    first = [items[0] for _, items in sorted(by_epoch.items())]
    chosen = first + [x for x in eligible if x not in first]
    selected = [x["sector"] for x in chosen[:maximum_sectors]]
    return {"status": "SELECTED", "selectedSectors": selected,
            "selectedEpochs": sorted({x["epoch"] for x in chosen[:maximum_sectors]}),
            "rejectedSectors": rejected,
            "selectionRule": "lowest-sector-per-official-epoch-then-numeric-fill",
            "product": SUPPORTED_PRODUCT, "cadenceSeconds": SUPPORTED_CADENCE_SECONDS,
            "fluxInspectedDuringSelection": False}
