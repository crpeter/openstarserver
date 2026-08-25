"""Best-effort, durable discovery records for science executions.

The catalog is an observability index, not science state.  Nothing in this
module creates, repairs, or decides whether an investigation should run.
"""

from __future__ import annotations

import json
import hashlib
import functools
import inspect
import os
import sqlite3
import time
import uuid
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openstar_state_storage import is_temporary_state_path

CATALOG_ENV = "OPENSTAR_SCIENCE_RUN_CATALOG"
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = MODULE_ROOT / "data" / "science-runs.sqlite3"
RECONSTRUCTION_SCHEMA = "openstar.tess-sector-reconstruction.v1"
RECONSTRUCTION_KIND = "tess-sector-reconstruction"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class ReconstructionManifestError(ValueError):
    """A reconstruction manifest is not valid authoritative metadata."""


def load_reconstruction_manifest(manifest_path: str | Path,
                                 *, verify_artifacts: bool = False) -> dict[str, Any]:
    """Strictly validate reconstruction metadata, without writing any source file."""
    path = Path(manifest_path).expanduser().resolve()
    match = re.fullmatch(r"sector-(\d+)-reconstruction-manifest\.json", path.name)
    if not match:
        raise ReconstructionManifestError("invalid reconstruction manifest filename")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconstructionManifestError(f"unable to read reconstruction manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ReconstructionManifestError("reconstruction manifest must be an object")
    if payload.get("schemaVersion") != RECONSTRUCTION_SCHEMA:
        raise ReconstructionManifestError("unsupported reconstruction schemaVersion")

    def integer(name: str, *, positive: bool = False) -> int:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
            raise ReconstructionManifestError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
        return value

    sector = integer("sector", positive=True)
    if sector != int(match.group(1)):
        raise ReconstructionManifestError("manifest filename sector does not match sector")
    inventory = integer("inventoryCount")
    original = integer("originalCompleteCount")
    recovery = integer("recoveryCompleteCount")
    combined = integer("combinedCompleteCount")
    if payload.get("coverageComplete") is not True:
        raise ReconstructionManifestError("coverageComplete must be true")
    if combined != inventory:
        raise ReconstructionManifestError("combinedCompleteCount must equal inventoryCount")
    if original + recovery != combined:
        raise ReconstructionManifestError("component counts do not equal combinedCompleteCount")
    original_ids = payload.get("originalInvestigationIDs")
    recovered_ids = payload.get("recoveredInvestigationIDs")
    if not isinstance(original_ids, list) or len(original_ids) != original:
        raise ReconstructionManifestError("originalInvestigationIDs length mismatch")
    if not isinstance(recovered_ids, list) or len(recovered_ids) != recovery:
        raise ReconstructionManifestError("recoveredInvestigationIDs length mismatch")
    try:
        original_set, recovered_set = set(original_ids), set(recovered_ids)
    except TypeError as error:
        raise ReconstructionManifestError("investigation IDs must be scalar and unique") from error
    if len(original_set) != original or len(recovered_set) != recovery:
        raise ReconstructionManifestError("investigation IDs must be unique within each component")
    if original_set & recovered_set:
        raise ReconstructionManifestError("original and recovered investigation IDs overlap")
    if len(original_set | recovered_set) != combined:
        raise ReconstructionManifestError("investigation ID union count mismatch")
    for name in ("originalHistoryModified", "recoveryHistoryModified"):
        if payload.get(name) is not False:
            raise ReconstructionManifestError(f"{name} must be false")
    for name in ("originalRoot", "recoveryRoot", "inventoryPath", "rankingPath"):
        if not isinstance(payload.get(name), str) or not payload[name].strip():
            raise ReconstructionManifestError(f"{name} must be present")
    for name in ("inventorySHA256", "rankingSHA256"):
        if not isinstance(payload.get(name), str) or not _SHA256.fullmatch(payload[name]):
            raise ReconstructionManifestError(f"{name} must be a SHA-256 hex string")
    if verify_artifacts:
        for name in ("originalRoot", "recoveryRoot"):
            if not _manifest_reference(path, payload[name]).is_dir():
                raise ReconstructionManifestError(f"{name} does not exist")
        for path_name, hash_name in (("inventoryPath", "inventorySHA256"),
                                     ("rankingPath", "rankingSHA256")):
            artifact = _manifest_reference(path, payload[path_name])
            try:
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            except OSError as error:
                raise ReconstructionManifestError(f"unable to read {path_name}: {error}") from error
            if digest.lower() != payload[hash_name].lower():
                raise ReconstructionManifestError(f"{path_name} SHA-256 mismatch")
    return payload


def _manifest_reference(manifest: Path, value: str) -> Path:
    reference = Path(value).expanduser()
    return reference if reference.is_absolute() else manifest.parent / reference


def register_sector_reconstruction(manifest_path: str | Path,
                                   catalog: str | Path | None = None) -> str:
    """Validate and idempotently index one immutable sector reconstruction."""
    manifest = Path(manifest_path).expanduser().resolve()
    payload = load_reconstruction_manifest(manifest, verify_artifacts=True)
    sector = payload["sector"]
    return ScienceRunCatalog(catalog).record(
        RECONSTRUCTION_KIND, manifest.parent, status="COMPLETE", logical_identity=sector,
        metadata={"sector": sector, "manifestPath": str(manifest)},
    )


def catalog_path(value: str | Path | None = None) -> Path:
    return Path(value or os.environ.get(CATALOG_ENV) or DEFAULT_CATALOG).expanduser().resolve()


def stable_run_id(kind: str, state_root: str | Path, logical_identity: Any = None) -> str:
    """Return an identity stable across process restarts and relative paths."""
    root = str(Path(state_root).expanduser().resolve())
    logical = json.dumps(logical_identity, sort_keys=True, separators=(",", ":"), default=str)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"openstar:science-run:{kind}:{root}:{logical}"))


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
        columns = {row[1] for row in connection.execute("PRAGMA table_info(science_runs)")}
        if columns and "run_id" not in columns:
            try:
                self._migrate_legacy(connection, columns)
            except Exception:
                connection.rollback()
                connection.close()
                raise
        connection.execute(
            """CREATE TABLE IF NOT EXISTS science_runs (
               run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state_root TEXT NOT NULL,
               status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
               metadata_json TEXT NOT NULL)"""
        )
        return connection

    @staticmethod
    def _migrate_legacy(connection: sqlite3.Connection, columns: set[str]) -> None:
        """Idempotently preserve the catalog used by the #72 observability branch."""
        required = {"id", "kind", "state_root", "status", "started_at",
                    "updated_at", "metadata_json"}
        if not required.issubset(columns):
            raise sqlite3.DatabaseError("Unsupported science_runs catalog schema")
        connection.execute("BEGIN IMMEDIATE")
        # Another process may have completed the migration while this process
        # waited for the write lock.
        current = {row[1] for row in connection.execute("PRAGMA table_info(science_runs)")}
        if "run_id" in current:
            connection.commit()
            return
        rows = connection.execute("SELECT * FROM science_runs").fetchall()
        names = [item[0] for item in connection.execute("SELECT * FROM science_runs LIMIT 0").description]
        connection.execute("ALTER TABLE science_runs RENAME TO science_runs_legacy_72")
        connection.execute(
            """CREATE TABLE science_runs (
               run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state_root TEXT NOT NULL,
               status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
               metadata_json TEXT NOT NULL)"""
        )
        for values in rows:
            old = dict(zip(names, values))
            try:
                metadata = json.loads(old.get("metadata_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            legacy = {name: old.get(name) for name in
                      ("display_name", "workflow_id", "completed_at") if old.get(name) is not None}
            summary = old.get("summary_json")
            if summary:
                try:
                    legacy["summary"] = json.loads(summary)
                except (TypeError, json.JSONDecodeError):
                    legacy["summary"] = summary
            if legacy:
                metadata.setdefault("legacy72", {}).update(legacy)
            old_kind = str(old["kind"])
            kind = {
                "investigation": "generic-investigation",
                "autonomous-investigation": "tess-autonomous",
            }.get(old_kind, old_kind)
            if kind != old_kind:
                metadata.setdefault("legacy72", {})["kind"] = old_kind
            legacy_summary = legacy.get("summary")
            if (isinstance(legacy_summary, dict)
                    and isinstance(legacy_summary.get("sectorSweep"), dict)):
                metadata.setdefault("sectorSweep", legacy_summary["sectorSweep"])
            state_root = str(old["state_root"])
            logical_identity = None
            if kind in {"tess-sector-sweep", "tess-ranked-followup"}:
                logical_identity = metadata.get("sector")
            elif kind == "generic-investigation":
                logical_identity = metadata.get("investigationID")
            can_resynthesize = kind == "tess-autonomous" or logical_identity is not None
            identity = (stable_run_id(kind, state_root, logical_identity)
                        if can_resynthesize else str(old["id"]))
            connection.execute(
                """INSERT INTO science_runs VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     status=CASE WHEN excluded.updated_at >= updated_at
                                 THEN excluded.status ELSE status END,
                     updated_at=MAX(updated_at, excluded.updated_at),
                     metadata_json=CASE WHEN excluded.updated_at >= updated_at
                                        THEN excluded.metadata_json ELSE metadata_json END""",
                (identity, kind, state_root, str(old["status"]),
                 _epoch_seconds(old.get("started_at") or old.get("updated_at")),
                 _epoch_seconds(old.get("updated_at") or old.get("started_at")),
                 json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)),
            )
        connection.execute("DROP TABLE science_runs_legacy_72")
        connection.commit()

    def record(self, kind: str, state_root: str | Path, *, status: str = "RUNNING",
               logical_identity: Any = None,
               metadata: dict[str, Any] | None = None) -> str:
        root = str(Path(state_root).expanduser().resolve())
        identity = stable_run_id(kind, root, logical_identity)
        now = time.time()
        connection = self._connect()
        try:
            combined_metadata: dict[str, Any] = {}
            existing = connection.execute(
                "SELECT metadata_json FROM science_runs WHERE run_id=?", (identity,)
            ).fetchone()
            if existing is not None:
                try:
                    decoded = json.loads(existing[0])
                    if isinstance(decoded, dict):
                        combined_metadata.update(decoded)
                except (TypeError, json.JSONDecodeError):
                    pass
            combined_metadata.update(metadata or {})
            encoded = json.dumps(combined_metadata, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False, default=str)
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

    def __init__(self, kind: str, state_root: str | Path, *, logical_identity: Any = None,
                 metadata: dict[str, Any] | None = None,
                 catalog: ScienceRunCatalog | None = None):
        self.kind, self.state_root, self.metadata = kind, state_root, metadata or {}
        self.logical_identity = logical_identity
        self.catalog = catalog or ScienceRunCatalog()
        self.run_id = stable_run_id(kind, state_root, logical_identity)
        self._terminal = False

    def update(self, status: str) -> None:
        try:
            self.run_id = self.catalog.record(self.kind, self.state_root, status=status,
                logical_identity=self.logical_identity, metadata=self.metadata)
        except Exception:
            pass
        if status != "RUNNING":
            self._terminal = True

    def __enter__(self) -> "ScienceRunRecorder":
        self.update("RUNNING")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is KeyboardInterrupt:
            self.update("INTERRUPTED")
        elif exc_type is not None:
            self.update("FAILED")
        elif not self._terminal:
            self.update("FINISHED")
        return False


def recorded_science_run(kind: str, state_root: str,
                         *, logical_identity: str | None = None,
                         metadata: tuple[str, ...] = ()):
    """Decorate a runner with exception-safe, result-aware observability."""
    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            recorder = ScienceRunRecorder(
                kind, values[state_root],
                logical_identity=values.get(logical_identity) if logical_identity else None,
                metadata={name: values.get(name) for name in metadata},
            )
            with recorder:
                result = function(*args, **kwargs)
                if isinstance(result, int) and not isinstance(result, bool) and result != 0:
                    recorder.update("FAILED")
                else:
                    recorder.update("FINISHED")
                return result
        return wrapped
    return decorate


def science_run_projection(run: ScienceRun) -> dict[str, Any]:
    """Read a catalog entry without changing its possibly damaged root."""
    root = Path(run.state_root)
    record = {
        "runID": run.run_id, "kind": run.kind, "stateRoot": run.state_root,
        "status": run.status, "createdAt": run.created_at, "updatedAt": run.updated_at,
        "metadata": dict(run.metadata), "condition": "available", "issues": [],
    }
    if is_temporary_state_path(root):
        record["condition"] = "degraded"
        record["issues"].append("unsafe_temporary_state_root")
    if not root.exists():
        record["condition"] = "degraded"
        record["issues"].append("state_root_missing")
        return record
    if run.kind == RECONSTRUCTION_KIND:
        configured = run.metadata.get("manifestPath")
        manifests = ([Path(configured)] if isinstance(configured, str) and configured
                     else list(root.glob("sector-*-reconstruction-manifest.json")))
        if len(manifests) != 1:
            record["condition"] = "degraded"
            record["issues"].append("reconstruction_manifest_missing_or_ambiguous")
            return record
        manifest_path = manifests[0].expanduser().resolve()
        try:
            payload = load_reconstruction_manifest(manifest_path)
        except ReconstructionManifestError as error:
            record["condition"] = "degraded"
            record["issues"].append(f"reconstruction_manifest_invalid: {error}")
            return record
        missing = [name for name in ("originalRoot", "recoveryRoot")
                   if not _manifest_reference(manifest_path, payload[name]).is_dir()]
        if missing:
            record["condition"] = "degraded"
            record["issues"].extend(f"reconstruction_component_missing: {name}"
                                    for name in missing)
        complete = payload["combinedCompleteCount"]
        inventory = payload["inventoryCount"]
        reconstruction = {
            "sector": payload["sector"], "inventory": inventory, "complete": complete,
            "originalComplete": payload["originalCompleteCount"],
            "recoveredComplete": payload["recoveryCompleteCount"],
            "coverageComplete": payload["coverageComplete"],
            "originalRoot": payload["originalRoot"], "recoveryRoot": payload["recoveryRoot"],
            "manifestPath": str(manifest_path),
        }
        record["reconstruction"] = reconstruction
        record["metadata"]["sectorSweep"] = {
            "sector": payload["sector"], "inventory": inventory, "complete": complete,
            "remaining": inventory - complete,
            "progress": complete / inventory if inventory else 1.0,
            "status": "COMPLETE", "historical": True, "reconstructed": True,
            "originalComplete": payload["originalCompleteCount"],
            "recoveredComplete": payload["recoveryCompleteCount"],
        }
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
    """Index one best shallow/ranking state root per sector without mutating it."""
    catalog = ScienceRunCatalog(path)
    candidates: dict[int, tuple[tuple[int, int, float], Path, dict[str, Any]]] = {}
    inspected = 0
    for supplied in roots:
        if inspected >= limit:
            break
        root = Path(supplied).expanduser().resolve()
        if not root.is_dir():
            continue
        try:
            children = [root, *(item for item in root.iterdir() if item.is_dir())]
        except OSError:
            continue
        for candidate_root in children:
            if inspected >= limit:
                break
            inspected += 1
            artifacts = [
                *(candidate_root.glob("tess-sector-*-inventory.json")),
                *(candidate_root.glob("tess-sector-*-ranking.json")),
            ]
            for artifact in artifacts:
                try:
                    payload = json.loads(artifact.read_text(encoding="utf-8"))
                    sector = int(payload["sector"])
                    if sector < 1:
                        continue
                    inventory_backed = artifact.name.endswith("-inventory.json")
                    active = _root_has_live_pid(candidate_root)
                    metadata = {"sector": sector, "historicalArtifact": str(artifact),
                                "inventoryBacked": inventory_backed}
                    for name in ("inventoryCount", "completedCount", "remainingCount", "rankingComplete"):
                        if name in payload:
                            metadata[name] = payload[name]
                    if not inventory_backed:
                        inventory_count = _nonnegative_int(payload.get("inventoryCount"))
                        completed_count = _nonnegative_int(payload.get("completedCount"))
                        remaining_count = _nonnegative_int(payload.get("remainingCount"))
                        if inventory_count is not None and completed_count is not None:
                            remaining_count = (max(0, inventory_count - completed_count)
                                               if remaining_count is None else remaining_count)
                            metadata["sectorSweep"] = {
                                "sector": sector,
                                "inventory": inventory_count,
                                "complete": completed_count,
                                "remaining": remaining_count,
                                "progress": completed_count / inventory_count if inventory_count else 0.0,
                                "status": "COMPLETE" if remaining_count == 0 else "HISTORICAL",
                                "historical": True,
                                "rankingComplete": bool(payload.get("rankingComplete")),
                            }
                    score = (int(active), int(inventory_backed), artifact.stat().st_mtime)
                    if sector not in candidates or score > candidates[sector][0]:
                        candidates[sector] = (score, candidate_root, metadata)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
    count = 0
    for sector, (score, root, metadata) in sorted(candidates.items()):
        status = "RUNNING" if score[0] else "HISTORICAL"
        try:
            catalog.record("tess-sector-sweep", root, status=status,
                           logical_identity=sector, metadata=metadata)
            count += 1
        except (OSError, ValueError, TypeError, sqlite3.Error):
            continue
    return count


def _root_has_live_pid(root: Path) -> bool:
    """Treat PID files only as presence hints; never clean or rewrite them."""
    for path in root.glob("*.pid"):
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
            if pid > 0:
                os.kill(pid, 0)
                return True
        except (OSError, ValueError):
            continue
    return False


def _nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _epoch_seconds(value: Any) -> float:
    """Normalize legacy numeric or ISO-8601 UTC/offset timestamps."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            parsed = datetime.fromisoformat(
                stripped[:-1] + "+00:00" if stripped.endswith(("Z", "z")) else stripped
            )
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    raise ValueError(f"Unsupported legacy catalog timestamp: {value!r}")
