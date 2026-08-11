from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str
    media_type: str | None = None


@dataclass(frozen=True)
class StageProvenance:
    software_id: str
    software_version: str
    input_hashes: dict[str, str] = field(default_factory=dict)
    parameters_hash: str | None = None
    result_hash: str | None = None
    node_contributions: dict[str, int] = field(default_factory=dict)
    project_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvestigationStage:
    id: str
    handler_id: str
    status: str
    triggered_by_stage_id: str | None
    parameters: dict[str, Any]
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    provenance: StageProvenance | None = None


@dataclass(frozen=True)
class Investigation:
    id: str
    workflow_id: str
    workflow_version: str
    status: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any]
    stages: tuple[InvestigationStage, ...] = ()


class InvestigationStore:
    """
    Domain-neutral immutable-stage investigation store.

    Existing stage records are never edited in place. Updating an
    investigation writes a new complete JSON snapshot atomically, while every
    completed stage contains its own immutable result/provenance hashes.
    """

    def __init__(self, root: str | Path = "data/investigations"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, investigation_id: str) -> Path:
        return self.root / f"{investigation_id}.json"

    def create(
        self,
        investigation_id: str,
        workflow_id: str,
        workflow_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> Investigation:
        path = self.path_for(investigation_id)
        if path.exists():
            raise FileExistsError(f"Investigation already exists: {path}")

        now = utc_now_iso()
        investigation = Investigation(
            id=investigation_id,
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            status="RUNNING",
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
            stages=(),
        )
        self.save(investigation)
        return investigation

    def load(self, investigation_id: str) -> Investigation:
        path = self.path_for(investigation_id)
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return self._decode(raw)

    def save(self, investigation: Investigation) -> None:
        payload = self._encode(investigation)
        path = self.path_for(investigation.id)
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def append_stage(
        self,
        investigation: Investigation,
        stage: InvestigationStage,
    ) -> Investigation:
        if any(existing.id == stage.id for existing in investigation.stages):
            raise ValueError(f"Duplicate stage id: {stage.id}")

        updated = Investigation(
            id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            status=investigation.status,
            created_at=investigation.created_at,
            updated_at=utc_now_iso(),
            metadata=investigation.metadata,
            stages=investigation.stages + (stage,),
        )
        self.save(updated)
        return updated

    def replace_last_pending_stage(
        self,
        investigation: Investigation,
        completed_stage: InvestigationStage,
    ) -> Investigation:
        if not investigation.stages:
            raise ValueError("Investigation has no stage to complete.")

        previous = investigation.stages[-1]
        if previous.id != completed_stage.id:
            raise ValueError("Only the most recently appended stage may complete.")
        if previous.status not in ("PENDING", "RUNNING"):
            raise ValueError("The current stage is already terminal.")

        updated = Investigation(
            id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            status=investigation.status,
            created_at=investigation.created_at,
            updated_at=utc_now_iso(),
            metadata=investigation.metadata,
            stages=investigation.stages[:-1] + (completed_stage,),
        )
        self.save(updated)
        return updated

    def set_status(
        self,
        investigation: Investigation,
        status: str,
    ) -> Investigation:
        updated = Investigation(
            id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            status=status,
            created_at=investigation.created_at,
            updated_at=utc_now_iso(),
            metadata=investigation.metadata,
            stages=investigation.stages,
        )
        self.save(updated)
        return updated

    @staticmethod
    def build_completed_stage(
        *,
        stage_id: str,
        handler_id: str,
        triggered_by_stage_id: str | None,
        parameters: dict[str, Any],
        result: dict[str, Any],
        software_id: str,
        software_version: str,
        input_hashes: dict[str, str] | None = None,
        node_contributions: dict[str, int] | None = None,
        project_ids: tuple[str, ...] = (),
        artifacts: tuple[ArtifactReference, ...] = (),
        started_at: str | None = None,
    ) -> InvestigationStage:
        return InvestigationStage(
            id=stage_id,
            handler_id=handler_id,
            status="COMPLETE",
            triggered_by_stage_id=triggered_by_stage_id,
            parameters=dict(parameters),
            started_at=started_at or utc_now_iso(),
            completed_at=utc_now_iso(),
            result=dict(result),
            artifacts=artifacts,
            provenance=StageProvenance(
                software_id=software_id,
                software_version=software_version,
                input_hashes=dict(input_hashes or {}),
                parameters_hash=sha256_json(parameters),
                result_hash=sha256_json(result),
                node_contributions=dict(node_contributions or {}),
                project_ids=project_ids,
            ),
        )

    @staticmethod
    def _encode(investigation: Investigation) -> dict[str, Any]:
        return asdict(investigation)

    @staticmethod
    def _decode(raw: dict[str, Any]) -> Investigation:
        stages = []
        for stage_raw in raw.get("stages", []):
            provenance_raw = stage_raw.get("provenance")
            provenance = (
                StageProvenance(
                    software_id=provenance_raw["software_id"],
                    software_version=provenance_raw["software_version"],
                    input_hashes=dict(provenance_raw.get("input_hashes", {})),
                    parameters_hash=provenance_raw.get("parameters_hash"),
                    result_hash=provenance_raw.get("result_hash"),
                    node_contributions=dict(
                        provenance_raw.get("node_contributions", {})
                    ),
                    project_ids=tuple(provenance_raw.get("project_ids", [])),
                )
                if provenance_raw
                else None
            )

            artifacts = tuple(
                ArtifactReference(
                    path=item["path"],
                    sha256=item["sha256"],
                    media_type=item.get("media_type"),
                )
                for item in stage_raw.get("artifacts", [])
            )

            stages.append(
                InvestigationStage(
                    id=stage_raw["id"],
                    handler_id=stage_raw["handler_id"],
                    status=stage_raw["status"],
                    triggered_by_stage_id=stage_raw.get(
                        "triggered_by_stage_id"
                    ),
                    parameters=dict(stage_raw.get("parameters", {})),
                    started_at=stage_raw.get("started_at"),
                    completed_at=stage_raw.get("completed_at"),
                    result=stage_raw.get("result"),
                    artifacts=artifacts,
                    provenance=provenance,
                )
            )

        return Investigation(
            id=raw["id"],
            workflow_id=raw["workflow_id"],
            workflow_version=raw["workflow_version"],
            status=raw["status"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            metadata=dict(raw.get("metadata", {})),
            stages=tuple(stages),
        )
