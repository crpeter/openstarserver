"""Target-independent admission and sector selection for period-family follow-up.

This module contains policy only.  It never downloads or examines flux; that is
important because selection must be frozen before an archive product is opened.
"""
from __future__ import annotations

import math
import statistics
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


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _latest_result(investigation: Investigation, handler_id: str) -> dict[str, Any] | None:
    return next((stage.result for stage in reversed(investigation.stages)
                 if stage.handler_id == handler_id and stage.status == "COMPLETE"
                 and stage.result is not None), None)


def build_period_family_followup_recommendation(
    investigation: Investigation,
    broad_interpretation: dict[str, Any],
    *,
    origin_stage_id: str,
) -> dict[str, Any] | None:
    """Build the reusable localization boundary from production-stage evidence.

    This is deliberately narrower than merely seeing a failed broad search.  It
    requires a reliable primary detection plus at least three reliable targeted
    independent-sector detections that were explicitly classified as
    resolution-limited.  The later broad search must also have failed closed.
    """
    prepared = _latest_result(investigation, "openstar.tess.prepare-target") or {}
    primary = _latest_result(investigation, "openstar.tess.primary-project.run") or {}
    identity = _latest_result(investigation, "openstar.tess.catalog-identity") or {}
    targeted_preparation = _latest_result(
        investigation, "openstar.tess.independent.prepare") or {}
    targeted_run = _latest_result(investigation, "openstar.tess.independent.run") or {}
    targeted = _latest_result(investigation, "openstar.tess.independent.interpret") or {}

    contradiction = targeted.get("contradictionPlan") or {}
    if not (
        (broad_interpretation.get("claimDecision") or {}).get("claim")
        == "HUMAN_REVIEW_REQUIRED"
        and broad_interpretation.get("promotionEligible") is False
        and broad_interpretation.get("selectedPeriodDays") is None
        and (targeted.get("claimDecision") or {}).get("claim")
        == "HUMAN_REVIEW_REQUIRED"
        and int(targeted.get("supportingSectorCount") or 0) == 0
        and contradiction.get("action") == "BROAD_INDEPENDENT_SEARCH"
        and contradiction.get("reason")
        == "targeted-candidate-not-recurrent-independent-sectors-contain-alternate-reliable-structure"
    ):
        return None

    tic_id = prepared.get("ticID")
    primary_sector = prepared.get("sector")
    primary_frequency = _finite(primary.get("candidateFrequency"))
    primary_period = _finite(primary.get("candidatePeriodDays"))
    metadata = ((identity.get("tic") or {}).get("metadata") or {})
    ra_deg = _finite(metadata.get("raDeg"))
    dec_deg = _finite(metadata.get("decDeg"))
    if not (
        isinstance(tic_id, int)
        and isinstance(primary_sector, int)
        and identity.get("ticID") == tic_id
        and primary_frequency is not None and primary_frequency > 0
        and primary_period is not None and primary_period > 0
        and math.isclose(primary_frequency * primary_period, 1.0, rel_tol=1e-6)
        and str(primary.get("periodStatus") or "").upper() == "RELIABLE"
        and str(primary.get("periodConfidence") or "").lower() == "high"
        and ra_deg is not None and dec_deg is not None
    ):
        return None

    datasets = {str(item.get("datasetID") or item.get("id") or ""): item
                for item in targeted_run.get("datasets") or []}
    detections: list[dict[str, Any]] = []
    for item in targeted.get("sectorResults") or []:
        sector = item.get("sector")
        dataset = datasets.get(str(item.get("datasetID") or "")) or {}
        frequency = _finite(item.get("candidateFrequency"))
        period = _finite(item.get("candidatePeriodDays"))
        dataset_frequency = _finite(dataset.get("candidateFrequency"))
        dataset_period = _finite(dataset.get("candidatePeriodDays"))
        if not (
            isinstance(sector, int)
            and frequency is not None and frequency > 0
            and period is not None and period > 0
            and math.isclose(frequency * period, 1.0, rel_tol=1e-6)
            and item.get("recurrenceClassification") == "RESOLUTION_LIMITED"
            and item.get("resolutionLimited") is True
            and item.get("supportsTarget") is False
            and item.get("eligibleForRecurrence") is True
            and item.get("boundaryHit") is False
            and str(dataset.get("periodStatus") or "").upper() == "RELIABLE"
            and str(dataset.get("periodConfidence") or "").lower() == "high"
            and dataset_frequency is not None
            and dataset_period is not None
            and math.isclose(frequency, dataset_frequency, rel_tol=1e-12)
            and math.isclose(period, dataset_period, rel_tol=1e-12)
        ):
            continue
        detections.append({
            "sector": sector,
            "datasetID": item.get("datasetID"),
            "frequencyCyclesPerDay": frequency,
            "periodDays": period,
            "power": _finite(dataset.get("candidatePower")),
            "peakProminenceRatio": _finite(dataset.get("candidatePeakProminenceRatio")),
            "foldCoherence": _finite(dataset.get("candidateFoldCoherence")),
            "recurrenceClassification": "RESOLUTION_LIMITED",
            "supportsOriginalCandidate": False,
        })
    detections.sort(key=lambda item: int(item["sector"]))
    if len(detections) < 3:
        return None
    prepared_sectors = [int(item["sector"]) for item in
                        targeted_preparation.get("preparedSectors") or []
                        if item.get("sector") is not None]
    result_sectors = [int(item["sector"]) for item in targeted.get("sectorResults") or []
                      if item.get("sector") is not None]
    if not (
        prepared_sectors == result_sectors
        and sorted(result_sectors) == [int(item["sector"]) for item in detections]
        and len(result_sectors) == len(set(result_sectors)) == len(detections)
        and int(targeted.get("eligibleSectorCount") or 0) == len(detections)
        and int(targeted.get("resolutionLimitedSectorCount") or 0) == len(detections)
        and int(contradiction.get("reliableSectorCount") or 0) == len(detections)
    ):
        return None

    official = {int(value) for value in (identity.get("tess") or {}).get("officialSectors") or []}
    consumed = sorted({primary_sector, *(int(item["sector"]) for item in detections)})
    if not set(consumed).issubset(official):
        return None

    members = sorted({primary_period, *(float(item["periodDays"]) for item in detections)})
    family_min, family_max = min(members), max(members)
    padding = max(0.10, family_max - family_min)
    center = float(statistics.median(members))
    window = [family_min - padding, family_max + padding]
    family = {
        "version": "openstar.tess-period-family-localization-boundary.v2",
        "originStageID": origin_stage_id,
        "investigationID": investigation.id,
        "ticID": tic_id,
        "targetSky": {"raDeg": ra_deg, "decDeg": dec_deg},
        "primaryDetection": {
            "sector": primary_sector,
            "frequencyCyclesPerDay": primary_frequency,
            "periodDays": primary_period,
            "power": _finite(primary.get("candidatePower")),
        },
        "independentSectorDetections": detections,
        "primaryPeriodDays": primary_period,
        "familyCenterDays": center,
        "periodFamilyMembersDays": members,
        "familyAcceptanceWindowDays": window,
        "consumedSectors": consumed,
        "observableDefinition": "persisted-sector-period-phase-reference",
        "claim": "HUMAN_REVIEW_REQUIRED",
        "periodFamilyResolved": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "periodDetectionRecomputed": False,
    }
    contract = freeze_period_family_contract(
        {"frozenPeriodFamily": family}, origin_stage_id
    )
    return {
        "recommendedNextTest": DIFFERENCE_TRIGGER,
        "frozenPeriodFamily": family,
        "periodFamilyContract": contract,
        "periodFamilyContractSHA256": sha256_json(contract),
        "periodFamilyResolved": False,
        "physicalCycleResolved": False,
        "physicalMechanismResolved": False,
        "autonomousContinuationEligible": True,
    }


def freeze_period_family_contract(evidence: dict[str, Any], origin_stage_id: str) -> dict[str, Any]:
    family = evidence.get("frozenPeriodFamily") or evidence.get("periodFamilyContract") or {}
    contract = {"schemaVersion": CONTRACT_VERSION,
        "originStageID": family.get("originStageID") or origin_stage_id,
        "primaryPeriodDays": family.get("primaryPeriodDays") or (family.get("primaryDetection") or {}).get("periodDays"),
        "familyCenterDays": family.get("familyCenterDays"),
        "periodFamilyMembersDays": family.get("periodFamilyMembersDays") or
            [family.get("primaryPeriodDays") or
             (family.get("primaryDetection") or {}).get("periodDays"),
             *[x.get("periodDays") for x in family.get("independentSectorDetections") or []]],
        "acceptanceWindowDays": family.get("acceptanceWindowDays") or family.get("familyAcceptanceWindowDays"),
        "consumedSectors": family.get("consumedSectors") or
            [(family.get("primaryDetection") or {}).get("sector"),
             *[x.get("sector") for x in family.get("independentSectorDetections") or []]],
        "observableDefinition": family.get("observableDefinition") or "persisted-period-family-phase-reference",
        "selectionPolicyVersion": POLICY_VERSION}
    contract["periodFamilyMembersDays"] = sorted({float(value) for value in
        contract["periodFamilyMembersDays"] if _finite(value) is not None})
    contract["consumedSectors"] = sorted({int(value) for value in
        contract["consumedSectors"] if value is not None})
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
    seen_sectors: set[int] = set()
    for raw in sorted(products, key=lambda x: int(x.get("sector") or 10**9)):
        if raw.get("sector") is None:
            rejected.append({"sector": None, "reason": "missing-sector"})
            continue
        sector = int(raw["sector"])
        reason = None
        observation_year = raw.get("observationYear", raw.get("year"))
        if sector in seen_sectors:
            reason = "duplicate-sector-product"
        elif sector in consumed:
            reason = "already-consumed-by-time-domain-observable"
        elif str(raw.get("author") or "").upper() != SUPPORTED_AUTHOR:
            reason = "unsupported-author"
        elif not str(raw.get("mission") or SUPPORTED_MISSION).upper().startswith(SUPPORTED_MISSION):
            reason = "unsupported-mission"
        elif int(raw.get("exptimeSeconds") or 0) != SUPPORTED_CADENCE_SECONDS:
            reason = "unsupported-cadence"
        elif observation_year is None:
            reason = "missing-observation-epoch-metadata"
        if reason:
            rejected.append({"sector": sector, "reason": reason})
        else:
            seen_sectors.add(sector)
            eligible.append({**raw, "sector": sector,
                             "epoch": f"CALENDAR_YEAR_{int(observation_year)}"})
    by_epoch: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        by_epoch.setdefault(item["epoch"], []).append(item)
    if len(by_epoch) < minimum_epochs:
        return {"policyVersion": POLICY_VERSION, "status": "INSUFFICIENT_EPOCH_COVERAGE", "selectedSectors": [],
                "rejectedSectors": rejected, "eligibleEpochs": sorted(by_epoch),
                "selectionRule": "lowest-sector-per-calendar-observation-year-then-numeric-fill",
                "fluxInspectedDuringSelection": False}
    first = [items[0] for _, items in sorted(by_epoch.items())]
    chosen = first + [x for x in eligible if x not in first]
    selected = [x["sector"] for x in chosen[:maximum_sectors]]
    return {"policyVersion": POLICY_VERSION, "status": "SELECTED", "selectedSectors": selected,
            "selectedEpochs": sorted({x["epoch"] for x in chosen[:maximum_sectors]}),
            "selectedSectorEpochs": {str(x["sector"]): x["epoch"] for x in chosen[:maximum_sectors]},
            "rejectedSectors": rejected,
            "selectionRule": "lowest-sector-per-calendar-observation-year-then-numeric-fill",
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
    catalog_radius = _finite(((gaia.get("queryProvenance") or {}).get("radiusArcsec")))
    coverage_complete = catalog_radius is not None and catalog_radius >= float(aperture_arcsec)
    ambiguous = target is None or gaia.get("ambiguous") is True or not coverage_complete
    target_mag = _finite(target.get("gMag")) if target else None
    neighbors = []
    for source in sources:
        source_id = source.get("sourceID") or source.get("sourceId")
        if target is not None and str(source_id) == str(target_id):
            continue
        separation = source.get("separationArcsec")
        magnitude = _finite(source.get("gMag"))
        fraction = (10 ** (-0.4 * (float(magnitude) - target_mag))
                    if target_mag is not None and magnitude is not None else None)
        neighbors.append({"sourceID": source_id, "raDeg": source.get("raDeg"),
            "decDeg": source.get("decDeg"), "separationArcsec": separation,
            "gMag": magnitude, "fluxFraction": fraction,
            "providerRadiusArcsec": aperture_arcsec,
            "relativeFluxAssumption": "10**(-0.4*(neighborG-targetG))"})
    missing_neighbor_data = any(n["fluxFraction"] is None or
                                _finite(n["separationArcsec"]) is None for n in neighbors)
    if target_mag is None or missing_neighbor_data:
        ambiguous = True
    return {"targetGaiaDR3SourceID": target_id, "target": target,
            "neighbors": neighbors if not ambiguous else None,
            "identityAmbiguous": ambiguous, "apertureRadiusArcsec": aperture_arcsec,
            "catalogQueryRadiusArcsec": catalog_radius,
            "catalogCoverageCompleteForAperture": coverage_complete,
            "missingMagnitudeData": target_mag is None or missing_neighbor_data}
