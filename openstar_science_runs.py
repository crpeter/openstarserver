"""Best-effort, durable discovery records for science executions.

The catalog is an observability index, not science state.  Nothing in this
module creates, repairs, or decides whether an investigation should run.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CATALOG_ENV = "OPENSTAR_SCIENCE_RUN_CATALOG"
DEFAULT_CATALOG = Path("data/science-runs.sqlite3")


def catalog_path(value: str | Path | None = None) -> Path:
    return Path(value or os.environ.get(CATALOG_ENV) or DEFAULT_CATALOG).expanduser().resolve()


def stable_run_id(kind: str, state_root: str | Path) -> str:
    """Return an identity stable across process restarts and relative paths."""
    root = str(Path(state_root).expanduser().resolve())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"openstar:science-run:{kind}:{root}"))


@dataclass(frozen=True)
class ScienceRun:
    run_id: str
    kind: str
    state_root: str
    status: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any]


class ScienceRunCatalog:
    """Small concurrent SQLite index containing no authoritative state."""

    def __init__(self, path: str | Path | None = None):
        self.path = catalog_path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS science_runs (
               run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state_root TEXT NOT NULL,
               status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
               metadata_json TEXT NOT NULL)"""
        )
        return connection

    def record(self, kind: str, state_root: str | Path, *, status: str = "RUNNING",
               metadata: dict[str, Any] | None = None) -> str:
        root = str(Path(state_root).expanduser().resolve())
        identity = stable_run_id(kind, root)
        now = time.time()
        encoded = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), allow_nan=False)
        connection = self._connect()
        try:
            connection.execute(
                """INSERT INTO science_runs
                   (run_id,kind,state_root,status,created_at,updated_at,metadata_json)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET status=excluded.status,
                     updated_at=excluded.updated_at, metadata_json=excluded.metadata_json""",
                (identity, kind, root, status, now, now, encoded),
            )
            connection.commit()
        finally:
            connection.close()
        return identity

    def list_runs(self) -> list[ScienceRun]:
        if not self.path.exists():
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT run_id,kind,state_root,status,created_at,updated_at,metadata_json "
                "FROM science_runs ORDER BY updated_at DESC, run_id"
            ).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            try:
                metadata = json.loads(row[6])
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            result.append(ScienceRun(*row[:6], metadata if isinstance(metadata, dict) else {}))
        return result


class ScienceRunRecorder:
    """Failure-isolating hook around a science runner.

    All methods intentionally swallow catalog errors: losing visibility must
    never alter the runner's result.
    """

    def __init__(self, kind: str, state_root: str | Path, *, metadata: dict[str, Any] | None = None,
                 catalog: ScienceRunCatalog | None = None):
        self.kind, self.state_root, self.metadata = kind, state_root, metadata or {}
        self.catalog = catalog or ScienceRunCatalog()
        self.run_id = stable_run_id(kind, state_root)

    def update(self, status: str) -> None:
        try:
            self.run_id = self.catalog.record(self.kind, self.state_root, status=status, metadata=self.metadata)
        except Exception:
            pass

    def __enter__(self) -> "ScienceRunRecorder":
        self.update("RUNNING")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.update("INTERRUPTED" if exc_type is KeyboardInterrupt else "FAILED" if exc_type else "FINISHED")
        return False


def science_run_projection(run: ScienceRun) -> dict[str, Any]:
    """Read a catalog entry without changing its possibly damaged root."""
    root = Path(run.state_root)
    record = {
        "runID": run.run_id, "kind": run.kind, "stateRoot": run.state_root,
        "status": run.status, "createdAt": run.created_at, "updatedAt": run.updated_at,
        "metadata": run.metadata, "condition": "available", "issues": [],
    }
    if not root.exists():
        record["condition"] = "degraded"
        record["issues"].append("state_root_missing")
        return record
    if run.kind == "tess-sector-sweep":
        sector = run.metadata.get("sector")
        inventory = root / f"tess-sector-{sector}-inventory.json"
        if not inventory.is_file():
            record["condition"] = "degraded"
            record["issues"].append("inventory_missing")
    investigations = root / "investigations"
    if investigations.is_dir():
        total = partial = 0
        for child in investigations.iterdir():
            if not child.is_dir():
                continue
            total += 1
            if not (child / "investigation.json").is_file():
                partial += 1
        record["investigationCount"] = total
        if partial:
            record["condition"] = "degraded"
            record["issues"].append("investigation_records_missing")
            record["partialInvestigationCount"] = partial
    return record


def discover_science_runs(path: str | Path | None = None) -> list[dict[str, Any]]:
    try:
        return [science_run_projection(run) for run in ScienceRunCatalog(path).list_runs()]
    except Exception:
        return []


def backfill_science_runs(roots: Iterable[str | Path], path: str | Path | None = None,
                          *, limit: int = 250) -> int:
    """Bounded/idempotent indexing; source roots are exclusively read."""
    catalog = ScienceRunCatalog(path)
    count = 0
    for supplied in roots:
        if count >= limit:
            break
        root = Path(supplied).expanduser().resolve()
        if not root.is_dir():
            continue
        inventories = list(root.glob("tess-sector-*-inventory.json"))[: max(0, limit - count)]
        for inventory in inventories:
            try:
                payload = json.loads(inventory.read_text(encoding="utf-8"))
                sector = int(payload.get("sector"))
                catalog.record("tess-sector-sweep", root, status="HISTORICAL", metadata={"sector": sector})
                count += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError, sqlite3.Error):
                continue
    return count
