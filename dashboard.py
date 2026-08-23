"""Pure projections for the standalone OpenStar dashboard sidecar."""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from typing import Any

CONNECTED_SECONDS = 150.0
RECENTLY_OFFLINE_SECONDS = 900.0


def _timestamp(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _value(node: dict[str, Any], telemetry: dict[str, Any], *names: str) -> Any:
    capabilities = (
        node.get("capabilities") if isinstance(node.get("capabilities"), dict) else {}
    )
    for name in names:
        if telemetry.get(name) is not None:
            return telemetry[name]
        if node.get(name) is not None:
            return node[name]
        if capabilities.get(name) is not None:
            return capabilities[name]
    return None


def build_snapshot(
    nodes: list[dict[str, Any]],
    contributions: dict[str, Any],
    projects: list[dict[str, Any]],
    telemetry_by_node: dict[str, dict[str, Any]],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Join coordinator observations and optional sidecar telemetry by node ID."""
    now = time.time() if now is None else now
    all_time = contributions.get("allTime", {})
    current = contributions.get("currentSession", {})
    totals = {
        str(item.get("nodeID", "")).lower(): item for item in all_time.get("nodes", [])
    }
    running = sum(
        int(
            project.get("projectAssignedWorkUnits")
            or project.get("assignedWorkUnits")
            or 0
        )
        for project in projects
    )
    failures = sum(
        int(
            project.get("projectFailedWorkUnits") or project.get("failedWorkUnits") or 0
        )
        for project in projects
    )
    workers = []
    for node in nodes:
        node_id = str(node.get("nodeID") or node.get("id") or "")
        heartbeat = telemetry_by_node.get(node_id.lower(), {})
        telemetry = (
            heartbeat.get("telemetry")
            if isinstance(heartbeat.get("telemetry"), dict)
            else {}
        )
        heartbeat_at = _timestamp(heartbeat.get("receivedAt"))
        coordinator_seen = _timestamp(node.get("lastSeenAt"))
        if coordinator_seen is None:
            coordinator_seen = _timestamp(node.get("registeredAt"))
        valid_timestamps = [
            timestamp
            for timestamp in (heartbeat_at, coordinator_seen)
            if timestamp is not None
        ]
        last_seen = max(valid_timestamps) if valid_timestamps else None
        age = max(0.0, now - float(last_seen)) if last_seen else None
        connection = (
            "connected"
            if age is not None and age <= CONNECTED_SECONDS
            else (
                "recently_disconnected"
                if age is not None and age <= RECENTLY_OFFLINE_SECONDS
                else "offline"
            )
        )
        local_state = str(
            _value(node, telemetry, "computeState", "workerState", "state") or ""
        ).lower()
        compute = (
            "active"
            if connection == "connected"
            and local_state in {"active", "computing", "running"}
            else "idle" if connection == "connected" else "offline"
        )
        contribution = totals.get(node_id.lower(), {})
        workers.append(
            {
                "id": node_id,
                "name": _value(node, telemetry, "deviceName", "friendlyName", "name"),
                "hardwareModel": _value(
                    node, telemetry, "hardwareIdentifier", "hardwareModel"
                ),
                "platform": _value(node, telemetry, "platform"),
                "osVersion": _value(
                    node, telemetry, "osVersion", "operatingSystemVersion"
                ),
                "workerVersion": _value(
                    node, telemetry, "appVersion", "workerVersion", "version"
                ),
                "connectionState": connection,
                "computeState": compute,
                "lastSeenAt": last_seen,
                "lastSeenSource": (
                    "dashboard_heartbeat"
                    if heartbeat_at is not None and heartbeat_at == last_seen
                    else "coordinator_registration"
                ),
                "currentAssignments": copy.deepcopy(
                    telemetry.get("currentAssignments") or []
                ),
                "currentAssignment": copy.deepcopy(telemetry.get("currentAssignment")),
                "workUnitProgress": telemetry.get("workUnitProgress"),
                "completedWorkUnits": int(contribution.get("acceptedWorkUnits") or 0),
                "failedWorkUnits": int(telemetry.get("failedWorkUnits") or 0),
                "sessionRuntimeSeconds": telemetry.get("sessionRuntimeSeconds"),
                "cumulativeRuntimeSeconds": contribution.get("workerComputeSeconds"),
                "metalSeconds": contribution.get("metalSeconds"),
                "measuredThroughput": contribution.get(
                    "sampleFrequencyEvaluationsPerMetalSecond"
                ),
                "throughputUnit": "sample-frequency evaluations / Metal second",
                "gpuName": _value(node, telemetry, "gpuName"),
                "processorCount": _value(node, telemetry, "processorCount"),
                "memoryGB": _value(node, telemetry, "memoryGB", "memoryGb"),
                "batteryLevel": _value(node, telemetry, "batteryLevel"),
                "powerState": _value(node, telemetry, "powerState", "batteryState"),
                "thermalState": _value(node, telemetry, "thermalState"),
                "lowPowerMode": _value(
                    node, telemetry, "lowPowerMode", "isLowPowerModeEnabled"
                ),
                "network": _value(node, telemetry, "network", "connectionType"),
                "latestError": _value(node, telemetry, "latestError", "lastError"),
                "capabilities": copy.deepcopy(node.get("capabilities") or {}),
                "recentCompleted": copy.deepcopy(
                    telemetry.get("recentCompleted") or []
                ),
                "recentFailures": copy.deepcopy(telemetry.get("recentFailures") or []),
            }
        )
    workers.sort(key=lambda worker: (
        -(worker["measuredThroughput"] if isinstance(worker["measuredThroughput"], (int, float)) else -1),
        worker["id"].lower(),
    ))
    states = Counter(worker["computeState"] for worker in workers)
    connected = sum(worker["connectionState"] == "connected" for worker in workers)
    summary = {
        "knownWorkers": len(workers),
        "connectedWorkers": connected,
        "activeWorkers": states["active"],
        "idleWorkers": states["idle"],
        "offlineWorkers": states["offline"],
        "recentlyDisconnectedWorkers": sum(
            w["connectionState"] == "recently_disconnected" for w in workers
        ),
        "completedWorkUnits": int(all_time.get("totalAcceptedWorkUnits") or 0),
        "runningWorkUnits": running,
        "workerComputeSeconds": float(all_time.get("totalWorkerComputeSeconds") or 0),
        "currentSessionComputeSeconds": float(
            current.get("totalWorkerComputeSeconds") or 0
        ),
        "measuredThroughput": current.get(
            "aggregateSampleFrequencyEvaluationsPerMetalSecond"
        ),
        "throughputUnit": "sample-frequency evaluations / Metal second",
        "failureCount": failures,
        "health": "healthy" if connected else "quiet",
        "updatedAt": now,
    }
    return {"summary": summary, "workers": workers}


def history_snapshot(contributions: dict[str, Any], science_runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    all_time = contributions.get("allTime", {})
    contribution_by_worker = sorted(
        copy.deepcopy(all_time.get("nodes", [])),
        key=lambda node: (-int(node.get("acceptedWorkUnits") or 0), str(node.get("nodeID") or "")),
    )
    by_model: Counter = Counter()
    for node in all_time.get("nodes", []):
        by_model[node.get("hardwareIdentifier") or "Unknown"] += int(
            node.get("acceptedWorkUnits") or 0
        )
    return {
        "available": bool(science_runs),
        "reason": None if science_runs else "The coordinator ledger does not retain time-series buckets.",
        "scienceRuns": copy.deepcopy(science_runs or []),
        "contributionByWorker": contribution_by_worker,
        "completedByDeviceModel": dict(by_model),
        "series": [],
    }
