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


class TessRankedFollowupTargetSource:
    id, version = "openstar.tess-ranked-followup-targets", "1"
    def __init__(self, admissions: Sequence[TessDeepAdmission]): self.admissions = tuple(admissions)
    def enumerate_targets(self) -> tuple[InvestigationTarget, ...]:
        return tuple(InvestigationTarget(
            id=f"tess-sector-{a.sector}-ranked-followup-tic-{a.ticID}",
            investigation_id=a.deepInvestigationID, workflow_id=WORKFLOW_ID,
            workflow_version=WORKFLOW_VERSION, priority=a.admittedRankingRank,
            metadata={"sourceProjectPath": a.sourceProjectPath, "sourceProjectID": a.sourceProjectID,
                      "datasetID": a.datasetID, "ticID": a.ticID, "targetName": a.targetName,
                      "sourceScanInvestigationID": a.sourceScanInvestigationID,
                      "sourceEvidenceSha256": a.sourceEvidenceSha256,
                      "sourceRankingRank": a.admittedRankingRank,
                      "sourceRankingPolicyID": a.rankingPolicyID,
                      "sourceRankingPolicyVersion": a.rankingPolicyVersion,
                      "sourceRankingSha256": a.sourceRankingSha256}) for a in self.admissions)
