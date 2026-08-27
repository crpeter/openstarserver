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
SUPPORTED_AUTHOR = "SPOC"
SUPPORTED_MISSION = "TESS"
SUPPORTED_CADENCE_SECONDS = 120


POLICY_VERSION = "openstar.period-family-followup-policy.v2"
CONTRACT_VERSION = "openstar.period-family-contract.v1"


def freeze_period_family_contract(evidence: dict[str, Any], origin_stage_id: str) -> dict[str, Any]:
    family = evidence.get("frozenPeriodFamily") or evidence.get("periodFamilyContract") or {}
    contract = {"schemaVersion": CONTRACT_VERSION,
        "originStageID": family.get("originStageID") or origin_stage_id,
        "primaryPeriodDays": family.get("primaryPeriodDays") or (family.get("primaryDetection") or {}).get("periodDays"),
        "familyCenterDays": family.get("familyCenterDays"),
        "periodFamilyMembersDays": family.get("periodFamilyMembersDays") or
            [x.get("periodDays") for x in family.get("independentSectorDetections") or []],
        "acceptanceWindowDays": family.get("acceptanceWindowDays") or family.get("familyAcceptanceWindowDays"),
        "consumedSectors": family.get("consumedSectors") or
            [x.get("sector") for x in family.get("independentSectorDetections") or []],
        "observableDefinition": family.get("observableDefinition") or "persisted-period-family-phase-reference",
        "selectionPolicyVersion": POLICY_VERSION}
    if not (contract["primaryPeriodDays"] and contract["familyCenterDays"]
            and isinstance(contract["acceptanceWindowDays"], list)
            and len(contract["acceptanceWindowDays"]) == 2
            and contract["periodFamilyMembersDays"]):
        raise RuntimeError("Frozen period-family contract is incomplete.")
    return contract


def verified_contract(result: dict[str, Any]) -> dict[str, Any]:
    contract = result.get("periodFamilyContract")
    if not isinstance(contract, dict) or contract.get("schemaVersion") != CONTRACT_VERSION:
        raise RuntimeError("Missing versioned period-family contract.")
    if result.get("periodFamilyContractSHA256") != sha256_json(contract):
        raise RuntimeError("Frozen period-family contract hash mismatch.")
    return contract


def latest_semantic_result(investigation: Investigation) -> tuple[Any, dict[str, Any]] | None:
    """Return only the current authoritative terminal scientific result.

    A workflow engine appends its RUNNING admission stage before invoking a
    handler, so that sole non-terminal tail is ignored.  Older recommendations
    are never searched for through intervening science.
    """
    stages = investigation.stages
    if stages and stages[-1].status == "RUNNING":
        stages = stages[:-1]
    if not stages:
        return None
    stage = stages[-1]
    result = stage.result or {}
    if (stage.status == "COMPLETE" and stage.stop
            and result.get("recommendedNextTest") in {
            DIFFERENCE_TRIGGER, TIME_DOMAIN_TRIGGER, EXTERNAL_TRIGGER,
            }):
        return stage, result
    return None


def verify_semantic_boundary(store: InvestigationStore, investigation: Investigation) -> dict[str, Any]:
    """Verify all authoritative completed ledgers and return target-independent evidence."""
    found = latest_semantic_result(investigation)
    if found is None:
        raise RuntimeError("No persisted period-family follow-up recommendation.")
    stage, result = found
    hashes = {}
    previous = None
    for predecessor in investigation.stages[: investigation.stages.index(stage) + 1]:
        if predecessor.status != "COMPLETE" or predecessor.result is None:
            raise RuntimeError(f"Non-terminal stage in authoritative chain: {predecessor.id}")
        if previous is not None and predecessor.triggered_by_stage_id != previous.id:
            raise RuntimeError(f"Broken authoritative stage linkage at {predecessor.id}")
        digest = store.verified_terminal_stage_ledger_hash(investigation.id, predecessor)
        if digest is None:
            raise RuntimeError(f"Authoritative ledger verification failed: {predecessor.id}")
        hashes[predecessor.id] = digest
        previous = predecessor
    if result.get("periodFamilyResolved") is not False:
        raise RuntimeError("The period family is not explicitly unresolved.")
    if result.get("claimDecision", {}).get("claim", result.get("claim")) != "HUMAN_REVIEW_REQUIRED":
        raise RuntimeError("Follow-up requires a conservative human-review claim.")
    observable_prefix = {
        DIFFERENCE_TRIGGER: "openstar.tess.generic-period-family-difference-imaging.",
        TIME_DOMAIN_TRIGGER: "openstar.tess.generic-period-family-time-domain-evolution.",
        EXTERNAL_TRIGGER: "openstar.tess.external-long-baseline.",
    }[result["recommendedNextTest"]]
    # The engine has already appended the current RUNNING preparation.  Reuse
    # detection intentionally considers only prior COMPLETE stages belonging to
    # this exact observable; difference imaging does not consume time-domain
    # evolution, nor vice versa.
    consumed = [s.id for s in investigation.stages
                if s.status == "COMPLETE" and s.handler_id.startswith(observable_prefix)]
    if consumed:
        raise RuntimeError("The recommended observable was already consumed.")
    return {"policyVersion": POLICY_VERSION, "admissionMode": "AUTONOMOUS_SEMANTIC",
            "triggerStageID": stage.id, "trigger": result["recommendedNextTest"],
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
    products = list(catalog_sectors)
    eligible_sector_ids = sorted({int(x["sector"]) for x in products
        if x.get("sector") is not None and str(x.get("author") or "").upper() == SUPPORTED_AUTHOR
        and str(x.get("mission") or SUPPORTED_MISSION).upper().startswith(SUPPORTED_MISSION)
        and int(x.get("exptimeSeconds") or 0) == SUPPORTED_CADENCE_SECONDS})
    # Three chronological quantile campaigns provide separated epochs without
    # pretending that every later extended-mission TESS cycle has 13 sectors.
    campaign = {}
    for index, sector in enumerate(eligible_sector_ids):
        bin_index = min(2, (3 * index) // max(1, len(eligible_sector_ids)))
        campaign[sector] = ("EARLY", "MIDDLE", "RECENT")[bin_index]
    for raw in sorted(products, key=lambda x: int(x["sector"])):
        sector = int(raw["sector"])
        reason = None
        if sector in consumed:
            reason = "already-consumed-by-time-domain-observable"
        elif str(raw.get("author") or "").upper() != SUPPORTED_AUTHOR:
            reason = "unsupported-author"
        elif not str(raw.get("mission") or SUPPORTED_MISSION).upper().startswith(SUPPORTED_MISSION):
            reason = "unsupported-mission"
        elif int(raw.get("exptimeSeconds") or 0) != SUPPORTED_CADENCE_SECONDS:
            reason = "unsupported-cadence"
        if reason:
            rejected.append({"sector": sector, "reason": reason})
        else:
            eligible.append({**raw, "sector": sector, "epoch": campaign[sector]})
    by_epoch: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_epoch.setdefault(item["epoch"], []).append(item)
    if len(by_epoch) < minimum_epochs:
        return {"policyVersion": POLICY_VERSION, "status": "INSUFFICIENT_EPOCH_COVERAGE", "selectedSectors": [],
                "rejectedSectors": rejected, "eligibleEpochs": sorted(by_epoch),
                "selectionRule": "lowest-sector-per-official-epoch-then-numeric-fill",
                "fluxInspectedDuringSelection": False}
    first = [items[0] for _, items in sorted(by_epoch.items())]
    chosen = first + [x for x in eligible if x not in first]
    selected = [x["sector"] for x in chosen[:maximum_sectors]]
    return {"policyVersion": POLICY_VERSION, "status": "SELECTED", "selectedSectors": selected,
            "selectedEpochs": sorted({x["epoch"] for x in chosen[:maximum_sectors]}),
            "selectedSectorEpochs": {str(x["sector"]): x["epoch"] for x in chosen[:maximum_sectors]},
            "rejectedSectors": rejected,
            "selectionRule": "lowest-sector-per-official-epoch-then-numeric-fill",
            "author": SUPPORTED_AUTHOR, "mission": SUPPORTED_MISSION,
            "exptimeSeconds": SUPPORTED_CADENCE_SECONDS,
            "fluxInspectedDuringSelection": False}


def extract_gaia_context(identity: dict[str, Any], *, aperture_arcsec: float = 16.0) -> dict[str, Any]:
    """Normalize the persisted ``gaiaDR3.sources`` identity without re-querying."""
    gaia = identity.get("gaiaDR3") or {}
    sources = list(gaia.get("sources") or [])
    nearest = gaia.get("nearest") or {}
    target_id = nearest.get("sourceID") or nearest.get("sourceId")
    target = next((s for s in sources if str(s.get("sourceID") or s.get("sourceId")) == str(target_id)), None)
    ambiguous = target is None or gaia.get("ambiguous") is True
    target_mag = float(target.get("photGMeanMag")) if target and target.get("photGMeanMag") is not None else None
    neighbors = []
    for source in sources:
        source_id = source.get("sourceID") or source.get("sourceId")
        if target is not None and str(source_id) == str(target_id):
            continue
        separation = source.get("separationArcsec")
        magnitude = source.get("photGMeanMag")
        fraction = (10 ** (-0.4 * (float(magnitude) - target_mag))
                    if target_mag is not None and magnitude is not None else None)
        neighbors.append({"sourceID": source_id, "raDeg": source.get("raDeg"),
            "decDeg": source.get("decDeg"), "separationArcsec": separation,
            "photGMeanMag": magnitude, "fluxFraction": fraction,
            "providerRadiusArcsec": aperture_arcsec,
            "relativeFluxAssumption": "10**(-0.4*(neighborG-targetG))"})
    return {"targetGaiaDR3SourceID": target_id, "target": target,
            "neighbors": neighbors if not ambiguous else None,
            "identityAmbiguous": ambiguous, "apertureRadiusArcsec": aperture_arcsec,
            "missingMagnitudeData": any(n["fluxFraction"] is None for n in neighbors)}
