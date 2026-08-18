"""Deterministic follow-up prioritization from immutable shallow sector scans.

This module deliberately performs local evidence analysis only.  It neither
contacts MAST/the coordinator nor starts a deep investigation.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openstar_investigation import InvestigationStore, sha256_file, sha256_json
from workflows.tess.tess_autonomy import WORKFLOW_ID as DEEP_WORKFLOW_ID
from workflows.tess.tess_sector_archive import TessSectorInventory
from workflows.tess.tess_sector_scan import EVIDENCE_HANDLER, MATERIALIZE_HANDLER, WORKFLOW_ID, WORKFLOW_VERSION

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class TessSectorRankingPolicy:
    id: str = "openstar.tess-sector-follow-up-priority"
    version: str = "1"
    confidence_order: tuple[str, ...] = ("high", "medium", "low", "none")


@dataclass(frozen=True)
class TessSectorRankingEntry:
    state: str
    tic_id: int
    target_name: str | None
    exclusion_reasons: tuple[str, ...] = ()
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TessSectorRanking:
    sector: int
    content: dict[str, Any]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class TessSectorRankingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, ranking: TessSectorRanking) -> None:
        _atomic_json(self.path, ranking.content)

    def load(self) -> TessSectorRanking:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        return TessSectorRanking(int(value["sector"]), value)


def _finite(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or (positive and result <= 0):
        return None
    return result


def _percentile(value: float, population: list[float]) -> float:
    # An empirical CDF is transparent, stable under ties, and deterministic.
    return 100.0 * sum(item <= value for item in population) / len(population)


def _verify_file(path_value: Any, expected_hash: Any, label: str, reasons: list[str]) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        reasons.append(f"MISSING_{label}_PATH")
        return None
    path = Path(path_value)
    if not path.is_file():
        reasons.append(f"MISSING_{label}_ARTIFACT")
        return None
    if not isinstance(expected_hash, str) or not expected_hash:
        reasons.append(f"MISSING_{label}_SHA256")
    elif sha256_file(path) != expected_hash:
        reasons.append(f"{label}_SHA256_MISMATCH")
    return path


def _entry(inventory: TessSectorInventory, item, store: InvestigationStore) -> TessSectorRankingEntry:
    product, sector = item.product, inventory.sector
    tic = int(product.tic_id)
    investigation_id = f"tess-sector-scan-{sector}-tic-{tic}"
    path = store.path_for(investigation_id)
    base = {"scanInvestigationID": investigation_id}
    if not path.is_file():
        return TessSectorRankingEntry("MISSING", tic, product.target_name, ("MISSING_INVESTIGATION",), base)
    try:
        investigation = store.load(investigation_id)
    except Exception as error:
        return TessSectorRankingEntry("FAILED", tic, product.target_name,
                                      (f"MALFORMED_INVESTIGATION:{type(error).__name__}",), base)
    historical_failures = tuple(stage for stage in investigation.stages if stage.status == "FAILED")
    base.update({
        "historicalFailedAttemptCount": len(historical_failures),
        "historicalFailedStageIDs": [stage.id for stage in historical_failures],
        "historicalFailureClassifications": [
            stage.failure_classification for stage in historical_failures
        ],
    })
    # Terminal stages are immutable attempt provenance.  A durable retry can
    # therefore be COMPLETE while still containing earlier FAILED stages; only
    # the investigation's current outcome determines whether ranking failed.
    if investigation.status == "FAILED":
        return TessSectorRankingEntry("FAILED", tic, product.target_name, ("FAILED_INVESTIGATION",), base)
    if investigation.status != "COMPLETE":
        return TessSectorRankingEntry("INCOMPLETE", tic, product.target_name,
                                      (f"INVESTIGATION_STATUS_{investigation.status}",), base)

    reasons: list[str] = []
    if investigation.workflow_id != WORKFLOW_ID or investigation.workflow_version != WORKFLOW_VERSION:
        reasons.append("UNEXPECTED_SCAN_WORKFLOW")
    if investigation.metadata.get("sector") != sector or investigation.metadata.get("ticID") != tic:
        reasons.append("INVESTIGATION_IDENTITY_MISMATCH")
    evidence_stage = investigation.stages[-1] if investigation.stages else None
    if (evidence_stage is None or evidence_stage.id != "003-persist-scan-evidence"
            or evidence_stage.handler_id != EVIDENCE_HANDLER or evidence_stage.status != "COMPLETE"
            or not evidence_stage.stop or not isinstance(evidence_stage.result, dict)):
        return TessSectorRankingEntry("COMPLETE_NO_RELIABLE_PERIOD", tic, product.target_name,
                                      tuple(reasons + ["INVALID_TERMINAL_EVIDENCE_STAGE"]), base)
    evidence = evidence_stage.result
    if evidence.get("sector") != sector or evidence.get("ticID") != tic:
        reasons.append("EVIDENCE_IDENTITY_MISMATCH")
    if not evidence_stage.artifacts:
        reasons.append("MISSING_EVIDENCE_ARTIFACT_PROVENANCE")
        evidence_sha = None
    else:
        artifact = evidence_stage.artifacts[0]
        evidence_path = _verify_file(artifact.path, artifact.sha256, "EVIDENCE", reasons)
        evidence_sha = artifact.sha256
        if evidence_path is not None:
            try:
                if json.loads(evidence_path.read_text(encoding="utf-8")) != evidence:
                    reasons.append("EVIDENCE_ARTIFACT_RESULT_MISMATCH")
            except (OSError, UnicodeError, json.JSONDecodeError):
                reasons.append("MALFORMED_EVIDENCE_ARTIFACT")
    materialize = next((s for s in investigation.stages if s.handler_id == MATERIALIZE_HANDLER and s.status == "COMPLETE"), None)
    prepared = materialize.result if materialize and isinstance(materialize.result, dict) else {}
    if not prepared:
        reasons.append("MISSING_MATERIALIZATION_PROVENANCE")
    dataset = _verify_file(evidence.get("datasetArtifact"), evidence.get("datasetSha256"), "DATASET", reasons)
    project = _verify_file(prepared.get("projectPath"), prepared.get("projectManifestSha256"), "SOURCE_PROJECT_MANIFEST", reasons)
    source_dataset = None
    source_project_id = None
    if project is not None:
        try:
            project_value = json.loads(project.read_text(encoding="utf-8"))
            source_project_id = project_value.get("id")
            project_datasets = project_value.get("datasets")
            matching = [value for value in project_datasets if isinstance(value, dict)
                        and value.get("ticID") == tic and value.get("sector") == sector] \
                if isinstance(project_datasets, list) else []
            if len(matching) != 1:
                reasons.append("SOURCE_PROJECT_IDENTITY_MISMATCH")
            else:
                source_dataset = matching[0]
                if dataset is not None and Path(str(source_dataset.get("path"))).resolve() != dataset.resolve():
                    reasons.append("SOURCE_PROJECT_DATASET_MISMATCH")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            reasons.append("MALFORMED_SOURCE_PROJECT_MANIFEST")
    project_ids = evidence.get("computeProjectIDs")
    contributions = evidence.get("nodeContributions")
    if not isinstance(project_ids, list) or not project_ids or not all(isinstance(x, str) and x for x in project_ids):
        reasons.append("MISSING_COMPUTE_PROJECT_PROVENANCE")
    if not isinstance(contributions, dict):
        reasons.append("INVALID_NODE_CONTRIBUTIONS")

    frequency = _finite(evidence.get("bestFrequency"), positive=True)
    period = _finite(evidence.get("bestPeriodDays"), positive=True)
    power = _finite(evidence.get("bestPower"))
    confidence = evidence.get("periodConfidence")
    if evidence.get("coverageComplete") is not True: reasons.append("INCOMPLETE_COVERAGE")
    if frequency is None: reasons.append("INVALID_BEST_FREQUENCY")
    if period is None: reasons.append("INVALID_BEST_PERIOD_DAYS")
    if power is None: reasons.append("INVALID_BEST_POWER")
    if confidence not in ("high", "medium", "low", "none"): reasons.append("INVALID_PERIOD_CONFIDENCE")
    fold = _finite(evidence.get("foldCoherence"))
    fold_missing = fold is None
    if evidence.get("foldCoherence") is not None and fold is None:
        reasons.append("INVALID_FOLD_COHERENCE")

    values = {**base, "periodStatus": evidence.get("periodStatus"), "periodConfidence": confidence,
              "bestFrequency": frequency, "bestPeriodDays": period, "bestPower": power,
              "foldCoherence": fold, "foldCoherenceMissing": fold_missing,
              "sampleCount": evidence.get("sampleCount"), "baselineDays": evidence.get("baselineDays"),
              "cadenceSeconds": evidence.get("cadenceSeconds"), "datasetArtifact": str(dataset) if dataset else evidence.get("datasetArtifact"),
              "datasetSha256": evidence.get("datasetSha256"), "sourceProjectPath": str(project) if project else prepared.get("projectPath"),
              "sourceProjectManifestSha256": prepared.get("projectManifestSha256"),
              "sourceProjectID": source_project_id,
              "datasetID": source_dataset.get("id") if source_dataset else evidence.get("datasetID"),
              "computeProjectIDs": project_ids, "nodeContributions": contributions,
              "sourceEvidenceSha256": evidence_sha}
    state = "COMPLETE_ELIGIBLE" if not reasons else "COMPLETE_NO_RELIABLE_PERIOD"
    return TessSectorRankingEntry(state, tic, product.target_name, tuple(reasons), values)


def aggregate_tess_sector_ranking(inventory: TessSectorInventory, store: InvestigationStore,
                                  policy: TessSectorRankingPolicy | None = None) -> TessSectorRanking:
    policy = policy or TessSectorRankingPolicy()
    entries = [_entry(inventory, item, store) for item in inventory.entries]
    eligible = [entry for entry in entries if entry.state == "COMPLETE_ELIGIBLE"]
    confidence_rank = {value: position for position, value in enumerate(policy.confidence_order)}
    eligible.sort(key=lambda e: (confidence_rank[e.values["periodConfidence"]],
                                 -(e.values["foldCoherence"] if e.values["foldCoherence"] is not None else float("-inf")),
                                 -e.values["bestPower"], e.tic_id))
    powers = [e.values["bestPower"] for e in eligible]
    folds = [e.values["foldCoherence"] for e in eligible if e.values["foldCoherence"] is not None]
    ranked = []
    for rank, entry in enumerate(eligible, 1):
        value = dict(entry.values)
        value.update({"rank": rank, "ticID": entry.tic_id, "targetName": entry.target_name,
                      "powerPercentile": _percentile(value["bestPower"], powers),
                      "foldCoherencePercentile": (_percentile(value["foldCoherence"], folds)
                                                   if value["foldCoherence"] is not None else None),
                      "rankingPolicyID": policy.id, "rankingPolicyVersion": policy.version,
                      "rankingKey": [value["periodConfidence"], value["foldCoherence"],
                                     value["bestPower"], entry.tic_id]})
        ranked.append(value)
    excluded = [{"ticID": e.tic_id, "targetName": e.target_name, "state": e.state,
                 "exclusionReasons": list(e.exclusion_reasons), **e.values}
                for e in entries if e.state != "COMPLETE_ELIGIBLE"]
    counts = {name: sum(e.state == name for e in entries) for name in
              ("FAILED", "INCOMPLETE", "MISSING")}
    completed = sum(e.state.startswith("COMPLETE_") for e in entries)
    inventory_value = asdict(inventory)
    input_evidence = [{"ticID": e.tic_id, "state": e.state, "values": e.values,
                       "exclusionReasons": list(e.exclusion_reasons)} for e in entries]
    content = {"schemaVersion": SCHEMA_VERSION, "sector": inventory.sector,
               "rankingPolicyID": policy.id, "rankingPolicyVersion": policy.version,
               "inventorySha256": sha256_json(inventory_value),
               "inputEvidenceFingerprint": sha256_json(input_evidence),
               "inventoryTargetCount": len(entries), "inventoryCount": len(entries),
               "completedScanCount": completed, "completedCount": completed,
               "eligibleRankedCount": len(ranked), "failedCount": counts["FAILED"],
               "incompleteCount": counts["INCOMPLETE"], "missingCount": counts["MISSING"],
               "excludedCount": len(excluded),
               "remainingCount": counts["INCOMPLETE"] + counts["MISSING"],
               "rankingComplete": counts["INCOMPLETE"] == counts["MISSING"] == 0,
               "rankedEntries": ranked, "excludedEntries": excluded}
    return TessSectorRanking(inventory.sector, content)


def write_promotion_manifest(ranking: TessSectorRanking, top_n: int, output: str | Path) -> dict[str, Any]:
    if top_n < 1: raise ValueError("top_n must be positive")
    selected = ranking.content["rankedEntries"][:top_n]
    datasets = []
    for entry in selected:
        source = json.loads(Path(entry["sourceProjectPath"]).read_text(encoding="utf-8"))
        source_dataset = next(item for item in source["datasets"]
                              if str(item.get("ticID")) == str(entry["ticID"]))
        dataset = dict(source_dataset)
        dataset.update({"investigationID": f"tess-discovery-sector-{ranking.sector}-tic-{entry['ticID']}",
                        "autonomousPriority": entry["rank"], "autonomousEligible": True,
                        "ticID": entry["ticID"], "targetName": entry["targetName"],
                        "sourceScanInvestigationID": entry["scanInvestigationID"],
                        "sourceRankingRank": entry["rank"],
                        "sourceRankingPolicyVersion": ranking.content["rankingPolicyVersion"],
                        "sourceEvidenceSha256": entry["sourceEvidenceSha256"]})
        datasets.append(dataset)
    manifest = {"id": f"tess-sector-{ranking.sector}-promoted-top-{top_n}",
                "name": f"TESS sector {ranking.sector} ranked follow-up candidates",
                "workloadID": "openstar.lomb-scargle.v1", "workflowID": DEEP_WORKFLOW_ID,
                "sourceRankingSha256": sha256_json(ranking.content), "datasets": datasets}
    _atomic_json(Path(output), manifest)
    return manifest
