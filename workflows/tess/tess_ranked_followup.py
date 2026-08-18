"""Durable admissions bridging sector rankings to deep TESS investigations."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from openstar_investigation import sha256_file, sha256_json
from openstar_targets import InvestigationTarget
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION
from workflows.tess.tess_sector_ranking import TessSectorRanking
from workflows.tess.tess_sector_scan import (
    EVIDENCE_HANDLER, MATERIALIZE_HANDLER, SCAN_HANDLER,
    WORKFLOW_ID as SCAN_WORKFLOW_ID, WORKFLOW_VERSION as SCAN_WORKFLOW_VERSION,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); temporary = ""
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)


@dataclass(frozen=True)
class TessDeepAdmission:
    sector: int
    ticID: int
    targetName: str | None
    deepInvestigationID: str
    sourceScanInvestigationID: str
    sourceProjectPath: str
    sourceProjectID: str
    sourceProjectManifestSha256: str
    datasetID: str
    datasetArtifact: str
    datasetSha256: str
    sourceEvidenceSha256: str
    admittedRankingRank: int
    rankingPolicyID: str
    rankingPolicyVersion: str
    sourceRankingSha256: str


class TessDeepAdmissionStore:
    """An append-only-by-TIC admission ledger."""
    def __init__(self, path: str | Path, sector: int | None = None):
        self.path, self.sector = Path(path), sector

    def load(self) -> tuple[TessDeepAdmission, ...]:
        if not self.path.exists(): return ()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if self.sector is not None and int(raw["sector"]) != self.sector:
            raise RuntimeError("Admission ledger sector does not match requested sector")
        return tuple(TessDeepAdmission(**item) for item in raw["admissions"])

    def save(self, admissions: Sequence[TessDeepAdmission]) -> None:
        sector = self.sector if self.sector is not None else (admissions[0].sector if admissions else None)
        if sector is None: raise ValueError("Cannot save an empty ledger without a sector")
        _atomic_json(self.path, {"schemaVersion": "1", "sector": sector,
                                "admissions": [asdict(item) for item in admissions]})

    def admit(self, ranking: TessSectorRanking, top_n: int):
        if top_n < 1: raise ValueError("top_n must be positive")
        existing = list(self.load()); known = {item.ticID for item in existing}
        new, excluded = [], []
        ranking_hash = sha256_json(ranking.content)
        for entry in ranking.content["rankedEntries"][:top_n]:
            tic = int(entry["ticID"])
            if tic in known: continue
            try: admission = _verified_admission(ranking, entry, ranking_hash)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                excluded.append({"ticID": tic, "reason": str(error)}); continue
            existing.append(admission); known.add(tic); new.append(admission)
        if new: self.save(existing)
        return tuple(existing), tuple(new), tuple(excluded)


def _verified_admission(ranking: TessSectorRanking, entry: dict[str, Any], ranking_hash: str) -> TessDeepAdmission:
    sector, tic = ranking.sector, int(entry["ticID"])
    project_path = Path(entry["sourceProjectPath"]).resolve()
    if not project_path.is_file() or sha256_file(project_path) != entry["sourceProjectManifestSha256"]:
        raise ValueError("SOURCE_PROJECT_MANIFEST_SHA256_MISMATCH")
    project = json.loads(project_path.read_text(encoding="utf-8"))
    if str(project.get("id")) != str(entry["sourceProjectID"]): raise ValueError("SOURCE_PROJECT_ID_MISMATCH")
    matches = [item for item in project.get("datasets", []) if isinstance(item, dict)
               and str(item.get("id")) == str(entry["datasetID"])
               and int(item.get("ticID", -1)) == tic and int(item.get("sector", -1)) == sector]
    if len(matches) != 1: raise ValueError("SOURCE_PROJECT_DATASET_IDENTITY_MISMATCH")
    artifact = Path(str(matches[0].get("path"))).resolve()
    if artifact != Path(entry["datasetArtifact"]).resolve(): raise ValueError("SOURCE_PROJECT_DATASET_PATH_MISMATCH")
    if not artifact.is_file() or sha256_file(artifact) != entry["datasetSha256"]: raise ValueError("DATASET_SHA256_MISMATCH")
    return TessDeepAdmission(sector, tic, entry.get("targetName"),
        f"tess-discovery-sector-{sector}-tic-{tic}", str(entry["scanInvestigationID"]),
        str(project_path), str(project["id"]), str(entry["sourceProjectManifestSha256"]),
        str(entry["datasetID"]), str(artifact), str(entry["datasetSha256"]),
        str(entry["sourceEvidenceSha256"]), int(entry["rank"]),
        str(ranking.content["rankingPolicyID"]), str(ranking.content["rankingPolicyVersion"]), ranking_hash)


def verified_reusable_primary(store, admission: TessDeepAdmission) -> dict[str, Any] | None:
    """Return self-contained shallow compute provenance, or no reuse on any doubt."""
    try:
        investigation = store.load(admission.sourceScanInvestigationID)
        if (investigation.workflow_id != SCAN_WORKFLOW_ID
                or investigation.workflow_version != SCAN_WORKFLOW_VERSION
                or investigation.status != "COMPLETE"):
            return None
        materialized = [s for s in investigation.stages if s.id == "001-materialize-light-curve"
                        and s.handler_id == MATERIALIZE_HANDLER and s.status == "COMPLETE"]
        scans = [s for s in investigation.stages if s.id == "002-broad-distributed-scan"
                 and s.handler_id == SCAN_HANDLER and s.status == "COMPLETE"]
        evidences = [s for s in investigation.stages if s.id == "003-persist-scan-evidence"
                     and s.handler_id == EVIDENCE_HANDLER and s.status == "COMPLETE" and s.stop]
        if len(materialized) != 1 or len(scans) != 1 or len(evidences) != 1:
            return None
        prepared, scan_stage, evidence_stage = materialized[0].result, scans[0], evidences[0]
        scan = scan_stage.result
        evidence = evidence_stage.result
        if not all(isinstance(value, dict) for value in (prepared, scan, evidence)):
            return None
        if scan.get("status") != "COMPLETE":
            return None
        if investigation.metadata.get("ticID") != admission.ticID or investigation.metadata.get("sector") != admission.sector:
            return None
        if (str(prepared.get("projectPath")) != admission.sourceProjectPath
                or prepared.get("projectManifestSha256") != admission.sourceProjectManifestSha256
                or prepared.get("datasetID") != admission.datasetID
                or str(prepared.get("datasetPath")) != admission.datasetArtifact
                or prepared.get("datasetSha256") != admission.datasetSha256):
            return None
        project_path, dataset_path = Path(admission.sourceProjectPath), Path(admission.datasetArtifact)
        dispatched_path = scan_stage.parameters.get("projectPath")
        if (not isinstance(dispatched_path, str)
                or Path(dispatched_path).expanduser().resolve() != project_path.resolve()):
            return None
        if (not project_path.is_file() or sha256_file(project_path) != admission.sourceProjectManifestSha256
                or not dataset_path.is_file() or sha256_file(dataset_path) != admission.datasetSha256):
            return None
        project, dataset = json.loads(project_path.read_text()), json.loads(dataset_path.read_text())
        if project.get("id") != admission.sourceProjectID or project.get("workloadID") != "openstar.lomb-scargle.v1":
            return None
        entries = [x for x in project.get("datasets", []) if isinstance(x, dict)
                   and str(x.get("id")) == admission.datasetID]
        if len(entries) != 1 or Path(str(entries[0].get("path"))).resolve() != dataset_path.resolve():
            return None
        if entries[0].get("ticID") != admission.ticID or entries[0].get("sector") != admission.sector:
            return None
        results = scan.get("datasets")
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            return None
        target = results[0]
        if str(target.get("datasetID")) != admission.datasetID:
            return None
        if target.get("ticID") is not None and target.get("ticID") != admission.ticID:
            return None
        if target.get("sector") is not None and target.get("sector") != admission.sector:
            return None
        expected = {
            "bestFrequency": target.get("bestFrequency", target.get("candidateFrequency")),
            "bestPeriodDays": target.get("bestPeriodDays", target.get("candidatePeriodDays")),
            "bestPower": target.get("bestPower", target.get("candidatePower")),
            "periodStatus": target.get("periodStatus"), "periodConfidence": target.get("periodConfidence"),
            "foldCoherence": target.get("candidateFoldCoherence"), "coverageComplete": target.get("coverageComplete"),
        }
        if any(evidence.get(key) != value for key, value in expected.items()):
            return None
        if (evidence.get("ticID") != admission.ticID or evidence.get("sector") != admission.sector
                or evidence.get("datasetArtifact") != admission.datasetArtifact
                or evidence.get("datasetSha256") != admission.datasetSha256):
            return None
        if len(evidence_stage.artifacts) != 1 or evidence_stage.artifacts[0].sha256 != admission.sourceEvidenceSha256:
            return None
        evidence_path = Path(evidence_stage.artifacts[0].path)
        if (not evidence_path.is_file() or sha256_file(evidence_path) != admission.sourceEvidenceSha256
                or json.loads(evidence_path.read_text()) != evidence):
            return None
        provenance = scan_stage.provenance
        if provenance is None or list(provenance.project_ids) != evidence.get("computeProjectIDs") \
                or provenance.node_contributions != evidence.get("nodeContributions"):
            return None
        if scan.get("projectID") is not None and scan.get("projectID") not in provenance.project_ids:
            return None
        return {"schemaVersion": "1", "verification": "EXACT_FROZEN_SHALLOW_PRIMARY",
                "sourceScanInvestigationID": investigation.id, "sourceWorkflowID": investigation.workflow_id,
                "sourceWorkflowVersion": investigation.workflow_version, "sourceProjectID": admission.sourceProjectID,
                "sourceProjectManifestSha256": admission.sourceProjectManifestSha256,
                "datasetID": admission.datasetID, "datasetArtifact": admission.datasetArtifact,
                "datasetSha256": admission.datasetSha256, "frequencySearchSha256": sha256_json(dataset.get("frequencySearch")),
                "sourceEvidenceSha256": admission.sourceEvidenceSha256,
                "coordinatorResult": scan, "coordinatorResultSha256": sha256_json(scan),
                "computeProjectIDs": list(provenance.project_ids),
                "nodeContributions": dict(provenance.node_contributions)}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


class TessRankedFollowupTargetSource:
    id, version = "openstar.tess-ranked-followup-targets", "1"
    def __init__(self, admissions: Sequence[TessDeepAdmission], reusable_primary=None):
        self.admissions = tuple(admissions); self.reusable_primary = reusable_primary or {}
    def enumerate_targets(self) -> tuple[InvestigationTarget, ...]:
        return tuple(InvestigationTarget(
            id=f"tess-sector-{a.sector}-ranked-followup-tic-{a.ticID}",
            investigation_id=a.deepInvestigationID, workflow_id=WORKFLOW_ID,
            workflow_version=WORKFLOW_VERSION, priority=a.admittedRankingRank,
            metadata={"sourceProjectPath": a.sourceProjectPath, "sourceProjectID": a.sourceProjectID,
                      "datasetID": a.datasetID, "ticID": a.ticID, "targetName": a.targetName,
                      "sourceScanInvestigationID": a.sourceScanInvestigationID,
                      "sourceEvidenceSha256": a.sourceEvidenceSha256,
                      "sourceProjectManifestSha256": a.sourceProjectManifestSha256,
                      "datasetSha256": a.datasetSha256,
                      **({"reusablePrimary": self.reusable_primary[a.deepInvestigationID]}
                         if a.deepInvestigationID in self.reusable_primary else {}),
                      "sourceRankingRank": a.admittedRankingRank,
                      "sourceRankingPolicyID": a.rankingPolicyID,
                      "sourceRankingPolicyVersion": a.rankingPolicyVersion,
                      "sourceRankingSha256": a.sourceRankingSha256}) for a in self.admissions)
