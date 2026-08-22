"""Durable operational catalog of OpenStar science runs.

The catalog records which science processes have run and where their authoritative
state lives. It is deliberately separate from investigation/scientific history:
runners may update catalog metadata, but the catalog never owns or rewrites science
results.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
DEFAULT_SCIENCE_RUN_CATALOG = ROOT / "data" / "science-runs.sqlite3"
_TERMINAL_STATUSES = {"COMPLETE", "FINISHED", "FAILED", "INTERRUPTED"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def science_run_id(
    kind: str, state_root: str | Path | None = None, *, identity: str | None = None
) -> str:
    """Return a stable run id for one durable science state root/identity."""
    if not kind or not kind.strip():
        raise ValueError("Science run kind is required.")
    resolved = ""
    if state_root is not None:
        resolved = str(Path(state_root).expanduser().resolve())
    material = "\0".join((kind.strip(), resolved, identity or ""))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{kind.strip()}:{digest}"


class ScienceRunCatalog:
    """Small process-safe SQLite registry for science-run observability."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        create: bool = True,
    ):
        configured = path or os.getenv("OPENSTAR_SCIENCE_RUN_CATALOG")
        self.path = Path(configured or DEFAULT_SCIENCE_RUN_CATALOG).expanduser().resolve()
        self.create = create
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        if not write and not self.path.exists():
            raise FileNotFoundError(self.path)
        if write:
            connection = sqlite3.connect(self.path, timeout=30)
        else:
            connection = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, timeout=30
            )
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a connection and always close it after commit/rollback handling."""
        connection = self._connect(write=write)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection(write=True) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS science_runs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    workflow_id TEXT,
                    status TEXT NOT NULL,
                    state_root TEXT,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS science_runs_updated_idx "
                "ON science_runs(updated_at DESC)"
            )

    @staticmethod
    def _json(value: dict[str, Any] | None) -> str:
        return json.dumps(
            value or {}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def register(
        self,
        run_id: str,
        *,
        kind: str,
        display_name: str,
        status: str = "RUNNING",
        state_root: str | Path | None = None,
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        now: str | None = None,
    ) -> dict[str, Any]:
        """Create or resume one durable science run without touching science state."""
        if not self.create:
            raise RuntimeError("ScienceRunCatalog is read-only.")
        if not run_id.strip() or not kind.strip() or not display_name.strip():
            raise ValueError("run_id, kind, and display_name are required.")
        timestamp = now or _utc_now()
        resolved_root = (
            str(Path(state_root).expanduser().resolve()) if state_root is not None else None
        )
        completed_at = timestamp if status in _TERMINAL_STATUSES else None
        with self._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO science_runs (
                    id, kind, display_name, workflow_id, status, state_root,
                    started_at, updated_at, completed_at, metadata_json, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    display_name = excluded.display_name,
                    workflow_id = excluded.workflow_id,
                    status = excluded.status,
                    state_root = excluded.state_root,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at,
                    metadata_json = excluded.metadata_json,
                    summary_json = excluded.summary_json
                """,
                (
                    run_id,
                    kind,
                    display_name,
                    workflow_id,
                    status,
                    resolved_root,
                    timestamp,
                    timestamp,
                    completed_at,
                    self._json(metadata),
                    self._json(summary),
                ),
            )
        return self.get(run_id) or {}

    def update(
        self,
        run_id: str,
        *,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        completed: bool | None = None,
        now: str | None = None,
    ) -> dict[str, Any]:
        if not self.create:
            raise RuntimeError("ScienceRunCatalog is read-only.")
        existing = self.get(run_id)
        if existing is None:
            raise KeyError(f"Unknown science run: {run_id}")
        timestamp = now or _utc_now()
        next_status = status or existing["status"]
        if completed is None:
            completed = next_status in _TERMINAL_STATUSES
        completed_at = timestamp if completed else None
        with self._connection(write=True) as connection:
            connection.execute(
                """
                UPDATE science_runs
                SET status = ?, updated_at = ?, completed_at = ?,
                    metadata_json = ?, summary_json = ?
                WHERE id = ?
                """,
                (
                    next_status,
                    timestamp,
                    completed_at,
                    self._json(
                        metadata if metadata is not None else existing["metadata"]
                    ),
                    self._json(summary if summary is not None else existing["summary"]),
                    run_id,
                ),
            )
        return self.get(run_id) or {}

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "displayName": row["display_name"],
            "workflowID": row["workflow_id"],
            "status": row["status"],
            "stateRoot": row["state_root"],
            "startedAt": row["started_at"],
            "updatedAt": row["updated_at"],
            "completedAt": row["completed_at"],
            "metadata": json.loads(row["metadata_json"]),
            "summary": json.loads(row["summary_json"]),
        }

    def get(self, run_id: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM science_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return None if row is None else self._decode(row)

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM science_runs
                ORDER BY CASE WHEN status = 'RUNNING' THEN 0 ELSE 1 END,
                         updated_at DESC, id
                """
            ).fetchall()
        return [self._decode(row) for row in rows]


class ScienceRunRecorder:
    """Best-effort runner hook; catalog failures never block science execution."""

    def __init__(
        self,
        *,
        kind: str,
        display_name: str,
        state_root: str | Path | None = None,
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        identity: str | None = None,
        catalog: ScienceRunCatalog | None = None,
    ):
        self.run_id = science_run_id(kind, state_root, identity=identity)
        self.catalog: ScienceRunCatalog | None = catalog
        try:
            self.catalog = self.catalog or ScienceRunCatalog()
            self.catalog.register(
                self.run_id,
                kind=kind,
                display_name=display_name,
                status="RUNNING",
                state_root=state_root,
                workflow_id=workflow_id,
                metadata=metadata,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            self.catalog = None

    @property
    def enabled(self) -> bool:
        return self.catalog is not None

    def finish(
        self,
        *,
        status: str = "FINISHED",
        summary: dict[str, Any] | None = None,
    ) -> None:
        if self.catalog is None:
            return
        try:
            self.catalog.update(
                self.run_id,
                status=status,
                summary=summary,
                completed=True,
            )
        except (OSError, sqlite3.Error, KeyError, TypeError, ValueError):
            return

    def fail(self, error: BaseException) -> None:
        self.finish(
            status="FAILED",
            summary={"error": f"{type(error).__name__}: {error}"},
        )

    def interrupt(self) -> None:
        self.finish(status="INTERRUPTED")
