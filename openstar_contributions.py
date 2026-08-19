"""Durable, domain-neutral accounting for accepted OpenStar work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
DEFAULT_CONTRIBUTION_DB = Path(
    "data/contributions/openstar-contributions.sqlite3"
)


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


AccountingAdapter = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class WorkloadAccountingRegistry:
    """Derives trusted work dimensions from server-owned inputs."""

    def __init__(self) -> None:
        self._adapters: dict[str, AccountingAdapter] = {}

    def register(self, workload_id: str, adapter: AccountingAdapter) -> None:
        self._adapters[str(workload_id)] = adapter

    def metrics(
        self, work_unit: dict[str, Any], dataset: dict[str, Any]
    ) -> dict[str, Any]:
        workload_id = str(work_unit.get("workloadID") or "")
        metrics: dict[str, Any] = {"workloadID": workload_id}
        adapter = self._adapters.get(workload_id)
        if adapter is not None:
            metrics.update(adapter(work_unit, dataset))
        return metrics


def _lomb_scargle_metrics(
    work_unit: dict[str, Any], dataset: dict[str, Any]
) -> dict[str, Any]:
    # Both dimensions come from the coordinator's immutable work and dataset,
    # never from the submitted result.
    samples = dataset.get("times")
    sample_count = len(samples) if isinstance(samples, list) else 0
    payload = work_unit.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    frequency_count = int(
        payload.get("frequencyCount", work_unit.get("frequencyCount", 0))
    )
    return {
        "sampleCount": sample_count,
        "frequencyCount": frequency_count,
        "sampleFrequencyEvaluations": sample_count * frequency_count,
    }


DEFAULT_ACCOUNTING = WorkloadAccountingRegistry()
DEFAULT_ACCOUNTING.register("openstar.lomb-scargle.v1", _lomb_scargle_metrics)


class ContributionStore:
    """SQLite implementation of the contribution repository boundary."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY COLLATE NOCASE,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    platform TEXT,
                    hardware_identifier TEXT,
                    gpu_name TEXT,
                    processor_count INTEGER,
                    memory_gb REAL,
                    capabilities_json TEXT NOT NULL,
                    owner_user_id TEXT NULL
                );
                CREATE TABLE IF NOT EXISTS contributions (
                    schema_version INTEGER NOT NULL,
                    contribution_id TEXT PRIMARY KEY,
                    coordinator_session_id TEXT NOT NULL,
                    accepted_at REAL NOT NULL,
                    project_id TEXT NOT NULL,
                    workload_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    work_unit_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    owner_user_id TEXT NULL,
                    worker_duration_seconds REAL,
                    work_metrics_json TEXT NOT NULL,
                    timing_metrics_json TEXT NOT NULL,
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                );
                CREATE INDEX IF NOT EXISTS contributions_session_idx
                    ON contributions(coordinator_session_id);
                CREATE INDEX IF NOT EXISTS contributions_node_time_idx
                    ON contributions(node_id, accepted_at);
                CREATE INDEX IF NOT EXISTS contributions_project_dataset_idx
                    ON contributions(project_id, dataset_id);
                CREATE TABLE IF NOT EXISTS contribution_aggregates (
                    coordinator_session_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    accepted_work_units INTEGER NOT NULL,
                    worker_compute_seconds REAL NOT NULL,
                    metal_seconds REAL NOT NULL,
                    sample_frequency_evaluations INTEGER NOT NULL,
                    PRIMARY KEY(coordinator_session_id, node_id),
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                );
                """
            )
            row = connection.execute(
                "SELECT value FROM ledger_metadata WHERE key='schema_version'"
            ).fetchone()
            if row is not None and int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported contribution schema: {row[0]}")
            connection.execute(
                "INSERT OR IGNORE INTO ledger_metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _node_values(payload: dict[str, Any]) -> tuple[Any, ...]:
        node_id = str(payload.get("nodeID") or payload.get("nodeId") or payload["id"])
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}
        def value(*names: str) -> Any:
            for name in names:
                if payload.get(name) is not None:
                    return payload[name]
                if capabilities.get(name) is not None:
                    return capabilities[name]
            return None
        return (
            node_id,
            value("platform"), value("hardwareIdentifier"), value("gpuName"),
            value("processorCount"), value("memoryGB", "memoryGb"), _json(capabilities),
        )

    def upsert_node(self, payload: dict[str, Any], seen_at: float) -> None:
        values = self._node_values(payload)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO nodes(
                    node_id,first_seen_at,last_seen_at,platform,hardware_identifier,
                    gpu_name,processor_count,memory_gb,capabilities_json,owner_user_id
                ) VALUES(?,?,?,?,?,?,?,?,?,NULL)
                ON CONFLICT(node_id) DO UPDATE SET
                    last_seen_at=excluded.last_seen_at,
                    platform=excluded.platform,
                    hardware_identifier=excluded.hardware_identifier,
                    gpu_name=excluded.gpu_name,
                    processor_count=excluded.processor_count,
                    memory_gb=excluded.memory_gb,
                    capabilities_json=excluded.capabilities_json""",
                (values[0], seen_at, seen_at, *values[1:]),
            )

    def nodes(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT * FROM nodes ORDER BY node_id COLLATE NOCASE").fetchall()
        return [self._public_node(row) for row in rows]

    @staticmethod
    def _public_node(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "nodeID": row["node_id"], "firstSeenAt": row["first_seen_at"],
            "lastSeenAt": row["last_seen_at"], "platform": row["platform"],
            "hardwareIdentifier": row["hardware_identifier"], "gpuName": row["gpu_name"],
            "processorCount": row["processor_count"], "memoryGB": row["memory_gb"],
            "capabilities": json.loads(row["capabilities_json"]),
            "ownerUserID": row["owner_user_id"],
        }

    def record(
        self, *, session_id: str, accepted_at: float, work_unit: dict[str, Any],
        dataset: dict[str, Any], node_id: str, result: dict[str, Any],
        accounting: WorkloadAccountingRegistry = DEFAULT_ACCOUNTING,
    ) -> bool:
        identity = "\0".join((session_id, str(work_unit.get("projectID", "")),
            str(work_unit.get("datasetID", "")), str(work_unit["id"]), str(node_id).lower()))
        contribution_id = hashlib.sha256(identity.encode()).hexdigest()
        metrics = accounting.metrics(work_unit, dataset)
        result_payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
        validation = result_payload.get("validation") if isinstance(result_payload.get("validation"), dict) else {}
        worker = _number(result.get("duration"))
        metal = _number(result_payload.get("metalDurationSeconds"))
        validation_seconds = _number(validation.get("durationSeconds"))
        timings = {key: value for key, value in {
            "workerTotalSeconds": worker, "metalSeconds": metal,
            "validationSeconds": validation_seconds,
        }.items() if value is not None}
        evaluations = int(metrics.get("sampleFrequencyEvaluations") or 0)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO contributions VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (SCHEMA_VERSION, contribution_id, session_id, accepted_at,
                 str(work_unit.get("projectID") or ""), str(work_unit.get("workloadID") or ""),
                 str(work_unit.get("datasetID") or ""), str(work_unit["id"]), str(node_id),
                 None, worker, _json(metrics), _json(timings)),
            )
            if cursor.rowcount == 0:
                return False
            connection.execute(
                """INSERT INTO contribution_aggregates VALUES(?,?,?,?,?,?)
                ON CONFLICT(coordinator_session_id,node_id) DO UPDATE SET
                  accepted_work_units=accepted_work_units+1,
                  worker_compute_seconds=worker_compute_seconds+excluded.worker_compute_seconds,
                  metal_seconds=metal_seconds+excluded.metal_seconds,
                  sample_frequency_evaluations=sample_frequency_evaluations+excluded.sample_frequency_evaluations""",
                (session_id, str(node_id), 1, worker or 0.0, metal or 0.0, evaluations),
            )
        return True

    def summary(self, current_session_id: str) -> dict[str, Any]:
        return {"coordinatorSessionID": current_session_id,
                "currentSession": self._summary_scope(current_session_id),
                "allTime": self._summary_scope(None)}

    def _summary_scope(self, session_id: str | None) -> dict[str, Any]:
        where = "WHERE a.coordinator_session_id=?" if session_id is not None else ""
        parameters = (session_id,) if session_id is not None else ()
        query = f"""SELECT n.*, SUM(a.accepted_work_units) accepted_work_units,
            SUM(a.worker_compute_seconds) worker_seconds, SUM(a.metal_seconds) metal_seconds,
            SUM(a.sample_frequency_evaluations) evaluations
            FROM contribution_aggregates a JOIN nodes n ON n.node_id=a.node_id
            {where} GROUP BY a.node_id ORDER BY n.node_id COLLATE NOCASE"""
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(query, parameters).fetchall()
        devices = []
        for row in rows:
            metal = float(row["metal_seconds"] or 0)
            evaluations = int(row["evaluations"] or 0)
            device = self._public_node(row)
            device.update({"acceptedWorkUnits": int(row["accepted_work_units"] or 0),
                "workerComputeSeconds": float(row["worker_seconds"] or 0),
                "metalSeconds": metal, "sampleFrequencyEvaluations": evaluations,
                "sampleFrequencyEvaluationsPerMetalSecond": evaluations / metal if metal > 0 else None})
            devices.append(device)
        total_metal = sum(item["metalSeconds"] for item in devices)
        total_evaluations = sum(item["sampleFrequencyEvaluations"] for item in devices)
        return {"totalAcceptedWorkUnits": sum(item["acceptedWorkUnits"] for item in devices),
            "totalWorkerComputeSeconds": sum(item["workerComputeSeconds"] for item in devices),
            "totalMetalSeconds": total_metal,
            "totalSampleFrequencyEvaluations": total_evaluations,
            "sampleFrequencyEvaluationsPerMetalSecond": total_evaluations / total_metal if total_metal > 0 else None,
            "nodes": devices}
