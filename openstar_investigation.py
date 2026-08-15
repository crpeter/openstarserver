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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    error: str | None = None
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
    Domain-neutral investigation store with immutable terminal stage records.

    Layout:
      <root>/<investigation-id>/investigation.json
      <root>/<investigation-id>/stages/<stage-id>.json

    `investigation.json` is the current snapshot. Once a stage becomes COMPLETE
    or FAILED, its individual stage file is written exactly once and is never
    replaced.
    """

    def __init__(self, root: str | Path = "data/investigations"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value: str) -> str:
        if not value or value in (".", ".."):
            raise ValueError("ID must not be empty.")
        if "/" in value or "\\" in value:
            raise ValueError(f"ID may not contain path separators: {value}")
        return value

    def directory_for(self, investigation_id: str) -> Path:
        return self.root / self._safe_id(investigation_id)

    def path_for(self, investigation_id: str) -> Path:
        return self.directory_for(investigation_id) / "investigation.json"

    def stage_path_for(self, investigation_id: str, stage_id: str) -> Path:
        return (
            self.directory_for(investigation_id)
            / "stages"
            / f"{self._safe_id(stage_id)}.json"
        )

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
            return self._decode(json.load(handle))

    def save(self, investigation: Investigation) -> None:
        self._atomic_write_json(
            self.path_for(investigation.id),
            self._encode(investigation),
            replace=True,
        )

    def append_running_stage(
        self,
        investigation: Investigation,
        stage: InvestigationStage,
    ) -> Investigation:
        if stage.status != "RUNNING":
            raise ValueError("New stage must start as RUNNING.")
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

    def complete_current_stage(
        self,
        investigation: Investigation,
        terminal_stage: InvestigationStage,
    ) -> Investigation:
        if terminal_stage.status not in ("COMPLETE", "FAILED"):
            raise ValueError("Terminal stage must be COMPLETE or FAILED.")
        if not investigation.stages:
            raise ValueError("Investigation has no stage to complete.")

        current = investigation.stages[-1]
        if current.id != terminal_stage.id or current.status != "RUNNING":
            raise ValueError("Only the current RUNNING stage may become terminal.")

        stage_path = self.stage_path_for(investigation.id, terminal_stage.id)
        self._atomic_write_json(
            stage_path,
            asdict(terminal_stage),
            replace=False,
        )

        updated = Investigation(
            id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            status=investigation.status,
            created_at=investigation.created_at,
            updated_at=utc_now_iso(),
            metadata=investigation.metadata,
            stages=investigation.stages[:-1] + (terminal_stage,),
        )
        self.save(updated)
        return updated

    def restart_current_running_stage(
        self,
        investigation: Investigation,
    ) -> tuple[Investigation, InvestigationStage]:
        """
        Remove only the current non-terminal RUNNING stage from the mutable
        investigation snapshot and return it as the stage to execute again.

        Terminal stage files remain immutable. This is intended for explicit
        recovery after process interruption (for example Ctrl+C or host loss)
        while a handler was still running.
        """
        if investigation.status != "RUNNING":
            raise ValueError(
                "Only a RUNNING investigation can restart an interrupted stage."
            )

        if not investigation.stages:
            raise ValueError("Investigation has no stage to resume.")

        current = investigation.stages[-1]
        if current.status != "RUNNING":
            raise ValueError(
                "Investigation has no interrupted RUNNING stage to resume."
            )

        stage_path = self.stage_path_for(investigation.id, current.id)
        if stage_path.exists():
            raise RuntimeError(
                "Cannot restart a RUNNING snapshot whose immutable terminal "
                f"stage file already exists: {stage_path}"
            )

        updated = Investigation(
            id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            status=investigation.status,
            created_at=investigation.created_at,
            updated_at=utc_now_iso(),
            metadata=investigation.metadata,
            stages=investigation.stages[:-1],
        )
        self.save(updated)
        return updated, current

    def set_status(self, investigation: Investigation, status: str) -> Investigation:
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

    def set_control_state(
        self,
        investigation: Investigation,
        *,
        status: str,
        control_state: dict[str, Any],
    ) -> Investigation:
        """Persist a scheduler-facing decision with its investigation status."""
        metadata = dict(investigation.metadata)
        metadata["controlState"] = dict(control_state)
        updated = Investigation(
            id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            status=status,
            created_at=investigation.created_at,
            updated_at=utc_now_iso(),
            metadata=metadata,
            stages=investigation.stages,
        )
        self.save(updated)
        return updated

    @staticmethod
    def build_terminal_stage(
        *,
        stage_id: str,
        handler_id: str,
        status: str,
        triggered_by_stage_id: str | None,
        parameters: dict[str, Any],
        result: dict[str, Any] | None,
        error: str | None,
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
            status=status,
            triggered_by_stage_id=triggered_by_stage_id,
            parameters=dict(parameters),
            started_at=started_at,
            completed_at=utc_now_iso(),
            result=dict(result) if result is not None else None,
            error=error,
            artifacts=artifacts,
            provenance=StageProvenance(
                software_id=software_id,
                software_version=software_version,
                input_hashes=dict(input_hashes or {}),
                parameters_hash=sha256_json(parameters),
                result_hash=(
                    sha256_json(result)
                    if result is not None
                    else None
                ),
                node_contributions=dict(node_contributions or {}),
                project_ids=project_ids,
            ),
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any, *, replace: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if not replace and path.exists():
            raise FileExistsError(f"Immutable stage already exists: {path}")

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

            if replace:
                os.replace(temp_path, path)
            else:
                # Atomic publish that fails if an immutable stage path already
                # exists. The temporary file is on the same filesystem.
                os.link(temp_path, path)
                os.unlink(temp_path)
                temp_path = ""
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _encode(investigation: Investigation) -> dict[str, Any]:
        return asdict(investigation)

    @staticmethod
    def _decode(raw: dict[str, Any]) -> Investigation:
        stages: list[InvestigationStage] = []
        for stage_raw in raw.get("stages", []):
            provenance_raw = stage_raw.get("provenance")
            provenance = None
            if provenance_raw:
                provenance = StageProvenance(
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
                    triggered_by_stage_id=stage_raw.get("triggered_by_stage_id"),
                    parameters=dict(stage_raw.get("parameters", {})),
                    started_at=stage_raw.get("started_at"),
                    completed_at=stage_raw.get("completed_at"),
                    result=stage_raw.get("result"),
                    error=stage_raw.get("error"),
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
