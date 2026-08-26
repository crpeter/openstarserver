"""Unattended portfolio supervision built on the durable scheduler and stores.

The heartbeat written here is observability only.  Scientific state is always
reconstructed from InvestigationStore and ExternalJobStore on every cycle.
"""
from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable

from openstar_external_jobs import ExternalJobMonitor, ExternalJobStore, apply_external_job_wakeups
from openstar_lifecycle import InvestigationSchedulingState
from openstar_scheduler import InvestigationScheduler, SchedulingRoundResult

HEARTBEAT_SCHEMA_VERSION = "openstar.autonomy-heartbeat.v1"
QUARANTINE_STATES = frozenset({
    InvestigationSchedulingState.FAILED,
    InvestigationSchedulingState.RECOVERY_REQUIRED,
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class AutonomySupervisor:
    """Poll dependencies and drain runnable investigations once per cycle."""

    def __init__(self, *, scheduler: InvestigationScheduler,
                 external_jobs: ExternalJobStore, monitor: ExternalJobMonitor,
                 heartbeat_path: str | Path, interval_seconds: float,
                 sleep: Callable[[float], None] = time.sleep):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.scheduler = scheduler
        self.external_jobs = external_jobs
        self.monitor = monitor
        self.heartbeat_path = Path(heartbeat_path)
        self.interval_seconds = interval_seconds
        self.sleep = sleep
        self.shutdown = Event()
        self.cycle_number = 0
        self.last_heartbeat: dict | None = None

    def request_shutdown(self, *_args) -> None:
        self.shutdown.set()

    def run_cycle(self) -> SchedulingRoundResult:
        started = _now()
        # poll_due itself enforces provider nextCheckAt; this supervisor never
        # introduces a competing external-job state machine or polling clock.
        self.monitor.poll_due()
        apply_external_job_wakeups(
            self.scheduler.store, self.external_jobs.ready_dependencies()
        )
        result = self.scheduler.run_until_idle()
        self.cycle_number += 1
        self._write_heartbeat(result, started, shutdown_requested=False)
        return result

    def _write_heartbeat(self, result: SchedulingRoundResult, started: str,
                         *, shutdown_requested: bool) -> None:
        states = {state.value: 0 for state in InvestigationSchedulingState}
        summaries = []
        quarantined = []
        for outcome in result.outcomes:
            states[outcome.state.value] += 1
            investigation = outcome.investigation
            stage = investigation.stages[-1] if investigation.stages else None
            error = None
            if outcome.error is not None:
                error = f"{type(outcome.error).__name__}: {outcome.error}"
            summaries.append({
                "investigationID": investigation.id,
                "status": investigation.status,
                "schedulerState": outcome.state.value,
                "latestStageID": stage.id if stage else None,
                "latestStageStatus": stage.status if stage else None,
                "error": error,
            })
            if outcome.state in QUARANTINE_STATES:
                quarantined.append(investigation.id)
        pending = self.external_jobs.pending()
        checks = [job.nextCheckAt for job in pending if job.nextCheckAt]
        payload = {
            "schemaVersion": HEARTBEAT_SCHEMA_VERSION,
            "updatedAt": _now(),
            "processID": os.getpid(),
            "mode": "daemon-multi-investigation",
            "cycleNumber": self.cycle_number,
            "cycleStartedAt": started,
            "cycleCompletedAt": _now(),
            "sleepIntervalSeconds": self.interval_seconds,
            "shutdownRequested": shutdown_requested,
            "countsBySchedulerState": states,
            "investigations": summaries,
            "quarantinedInvestigationIDs": sorted(quarantined),
            "externalJobs": {
                "pendingCount": len(pending),
                "readyDependencyCount": len(self.external_jobs.ready_dependencies()),
                "failedDependencyCount": len(self.external_jobs.failed_dependencies()),
                "nextCheckAt": min(checks) if checks else None,
            },
            "lastCycleDispatchedInvestigationIDs": list(result.dispatched_investigation_ids),
        }
        _atomic_json(self.heartbeat_path, payload)
        self.last_heartbeat = payload

    def run(self, *, max_cycles: int | None = None,
            install_signal_handlers: bool = True) -> int:
        if max_cycles is not None and max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        previous = {}
        if install_signal_handlers:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, self.request_shutdown)
        last_result = None
        last_started = _now()
        try:
            while not self.shutdown.is_set() and (
                max_cycles is None or self.cycle_number < max_cycles
            ):
                last_started = _now()
                last_result = self.run_cycle()
                if not self.shutdown.is_set() and (
                    max_cycles is None or self.cycle_number < max_cycles
                ):
                    self.shutdown.wait(self.interval_seconds) if self.sleep is time.sleep else self.sleep(self.interval_seconds)
        finally:
            if self.shutdown.is_set() and last_result is not None:
                self._write_heartbeat(last_result, last_started, shutdown_requested=True)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return 0

