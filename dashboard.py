"""Read-only, domain-neutral fleet telemetry projections for the web dashboard."""

from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Any

CONNECTED_SECONDS = 150.0
RECENTLY_OFFLINE_SECONDS = 900.0


def _value(source: dict[str, Any], *names: str) -> Any:
    capabilities = source.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    telemetry = source.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = capabilities.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
    for name in names:
        if source.get(name) is not None:
            return source[name]
        if capabilities.get(name) is not None:
            return capabilities[name]
        if telemetry.get(name) is not None:
            return telemetry[name]
    return None


def _assignment_snapshot(
    runtime: Any, now: float
) -> tuple[dict[str, list[dict[str, Any]]], Counter, list, list]:
    assignments: dict[str, list[dict[str, Any]]] = {}
    failures: Counter = Counter()
    recent: list[dict[str, Any]] = []
    recent_failures: list[dict[str, Any]] = []
    with runtime.lock:
        states = list(runtime._states.values())
    for state in states:
        with state.lock:
            for work_id, lease in state.assigned.items():
                if float(lease.get("leaseExpiresAt") or 0) < now:
                    continue
                work = state.work_units.get(work_id, {})
                assignments.setdefault(str(lease.get("nodeID", "")).lower(), []).append(
                    {
                        "workUnitID": work_id,
                        "projectID": work.get("projectID", state.project_id),
                        "datasetID": work.get("datasetID"),
                        "workloadID": work.get("workloadID", state.workload_id),
                        "assignedAt": lease.get("assignedAt"),
                        "leaseExpiresAt": lease.get("leaseExpiresAt"),
                        "progress": None,
                    }
                )
            for work_id, result in state.completed.items():
                if result.get("nodeID") is not None:
                    recent.append(
                        {
                            "nodeID": str(result["nodeID"]),
                            "workUnitID": result.get("id")
                            or result.get("workUnitID")
                            or work_id,
                            "projectID": state.project_id,
                            "workloadID": state.workload_id,
                            "durationSeconds": result.get("duration"),
                        }
                    )
            histories = [state.execution_failure_history]
            for name in (
                "environment_unavailable_history",
                "transport_unavailable_history",
            ):
                history = getattr(state, name, None)
                if isinstance(history, dict):
                    histories.append(history)
            for history in histories:
                for items in history.values():
                    for item in items:
                        failures[str(item.get("nodeID", "")).lower()] += 1
                        recent_failures.append(copy.deepcopy(item))
    return assignments, failures, recent[-100:], recent_failures[-100:]


def dashboard_snapshot(runtime: Any, now: float | None = None) -> dict[str, Any]:
    """Build a consistent copy without changing scheduler or investigation state."""
    now = time.time() if now is None else now
    nodes = runtime.registered_nodes()
    contributions = runtime.contribution_summary()
    totals_by_node = {
        str(item["nodeID"]).lower(): item
        for item in contributions["allTime"].get("nodes", [])
    }
    assignments, failures, recent, recent_failures = _assignment_snapshot(runtime, now)
    workers = []
    for source in nodes:
        node_id = str(source.get("nodeID") or source.get("id") or "")
        last_seen = float(source.get("lastSeenAt") or source.get("registeredAt") or 0)
        age = max(0.0, now - last_seen) if last_seen else None
        worker_assignments = assignments.get(node_id.lower(), [])
        if age is not None and age <= CONNECTED_SECONDS:
            connection = "connected"
            compute = "active" if worker_assignments else "idle"
        elif age is not None and age <= RECENTLY_OFFLINE_SECONDS:
            connection, compute = "recently_disconnected", "offline"
        else:
            connection, compute = "offline", "offline"
        totals = totals_by_node.get(node_id.lower(), {})
        workers.append(
            {
                "id": node_id,
                "name": _value(source, "deviceName", "friendlyName", "name"),
                "hardwareModel": _value(source, "hardwareIdentifier", "hardwareModel"),
                "platform": _value(source, "platform"),
                "osVersion": _value(source, "osVersion", "operatingSystemVersion"),
                "workerVersion": _value(
                    source, "appVersion", "workerVersion", "version"
                ),
                "connectionState": connection,
                "computeState": compute,
                "lastSeenAt": last_seen or None,
                "firstSeenAt": source.get("firstSeenAt") or source.get("registeredAt"),
                "currentAssignment": (
                    copy.deepcopy(worker_assignments[0]) if worker_assignments else None
                ),
                "currentAssignments": copy.deepcopy(worker_assignments),
                "runningWorkUnits": len(worker_assignments),
                "completedWorkUnits": int(totals.get("acceptedWorkUnits") or 0),
                "failedWorkUnits": int(failures[node_id.lower()]),
                "cumulativeRuntimeSeconds": totals.get("workerComputeSeconds"),
                "metalSeconds": totals.get("metalSeconds"),
                "measuredThroughput": totals.get(
                    "sampleFrequencyEvaluationsPerMetalSecond"
                ),
                "throughputUnit": "sample-frequency evaluations / Metal second",
                "gpuName": _value(source, "gpuName"),
                "processorCount": _value(source, "processorCount"),
                "memoryGB": _value(source, "memoryGB", "memoryGb"),
                "batteryLevel": _value(source, "batteryLevel"),
                "powerState": _value(source, "powerState", "batteryState"),
                "thermalState": _value(source, "thermalState"),
                "lowPowerMode": _value(source, "lowPowerMode", "isLowPowerModeEnabled"),
                "network": _value(source, "network", "connectionType"),
                "latestError": _value(source, "latestError", "lastError"),
                "capabilities": copy.deepcopy(source.get("capabilities") or {}),
            }
        )
    counts = Counter(item["computeState"] for item in workers)
    connected = sum(item["connectionState"] == "connected" for item in workers)
    current = contributions["currentSession"]
    summary = {
        "knownWorkers": len(workers),
        "connectedWorkers": connected,
        "activeWorkers": counts["active"],
        "idleWorkers": counts["idle"],
        "offlineWorkers": counts["offline"],
        "recentlyDisconnectedWorkers": sum(
            w["connectionState"] == "recently_disconnected" for w in workers
        ),
        "completedWorkUnits": contributions["allTime"]["totalAcceptedWorkUnits"],
        "runningWorkUnits": sum(len(items) for items in assignments.values()),
        "workerComputeSeconds": contributions["allTime"]["totalWorkerComputeSeconds"],
        "currentSessionComputeSeconds": current["totalWorkerComputeSeconds"],
        "measuredThroughput": current[
            "aggregateSampleFrequencyEvaluationsPerMetalSecond"
        ],
        "throughputUnit": "sample-frequency evaluations / Metal second",
        "failureCount": sum(failures.values()),
        "health": (
            "degraded"
            if failures and connected == 0
            else ("healthy" if connected else "quiet")
        ),
        "updatedAt": now,
    }
    return {
        "summary": summary,
        "workers": workers,
        "recentCompleted": recent,
        "recentFailures": recent_failures,
    }


def activity_snapshot(runtime: Any) -> dict[str, Any]:
    projects = []
    with runtime.lock:
        states = list(runtime._states.values())
    for state in states:
        with state.lock:
            total = len(state.work_units)
            completed = len(state.completed)
            failed = len(state.failed)
            assigned = len(state.assigned)
            pending = len(state.pending)
            projects.append(
                {
                    "projectID": state.project_id,
                    "projectName": state.project_name,
                    "workloadID": state.workload_id,
                    "status": "COMPLETE" if completed + failed >= total else "RUNNING",
                    "projectPendingWorkUnits": pending,
                    "projectAssignedWorkUnits": assigned,
                    "projectCompletedWorkUnits": completed,
                    "projectFailedWorkUnits": failed,
                    "projectTotalWorkUnits": total,
                    "projectProgress": (completed + failed) / total if total else 1.0,
                }
            )
    return {"projects": projects, "updatedAt": time.time()}


def history_snapshot(runtime: Any) -> dict[str, Any]:
    # The v1 ledger retains accurate cumulative contributions, but no time-series
    # buckets. Be explicit rather than synthesizing historical points.
    summary = runtime.contribution_summary()["allTime"]
    by_model: Counter = Counter()
    for node in summary.get("nodes", []):
        by_model[node.get("hardwareIdentifier") or "Unknown"] += node[
            "acceptedWorkUnits"
        ]
    return {
        "available": False,
        "reason": "Time-series telemetry begins with a future ledger schema.",
        "contributionByWorker": summary.get("nodes", []),
        "completedByDeviceModel": dict(by_model),
        "series": [],
    }
