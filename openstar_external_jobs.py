"""Durable, provider-neutral asynchronous external job coordination."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from openstar_investigation import InvestigationStore

VERSION = "openstar.external-job.v1"
PENDING_STATES = frozenset({"SUBMITTED", "QUEUED", "RUNNING"})
TERMINAL_STATES = frozenset({"COMPLETE", "REMOTE_FAILED"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_job_id(provider: str, investigation_id: str, trigger_stage_id: str,
                  dependency_id: str, role: str) -> str:
    identity = "\0".join((provider, investigation_id, trigger_stage_id,
                           dependency_id, role)).encode()
    return f"external-job-{hashlib.sha256(identity).hexdigest()[:24]}"


@dataclass(frozen=True)
class ExternalJob:
    id: str
    provider: str
    investigationID: str
    triggerStageID: str
    dependencyID: str
    role: str
    state: str
    remoteTaskURL: str | None = None
    remoteResultURL: str | None = None
    submittedAt: str | None = None
    lastCheckedAt: str | None = None
    nextCheckAt: str | None = None
    completedAt: str | None = None
    lastOperationalError: str | None = None
    version: str = VERSION

    @classmethod
    def create(cls, *, provider: str, investigation_id: str,
               trigger_stage_id: str, dependency_id: str, role: str) -> "ExternalJob":
        return cls(stable_job_id(provider, investigation_id, trigger_stage_id,
                                 dependency_id, role), provider, investigation_id,
                   trigger_stage_id, dependency_id, role, "SUBMITTED")


@dataclass(frozen=True)
class PollResult:
    state: str
    remote_result_url: str | None = None


class ExternalJobProvider(Protocol):
    def poll(self, job: ExternalJob) -> PollResult: ...


class ExternalJobPollUnavailable(RuntimeError):
    """A narrow retryable transport or remote-service polling failure."""


class ExternalJobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        if not job_id or "/" in job_id or "\\" in job_id:
            raise ValueError("Invalid external job ID")
        return self.root / f"{job_id}.json"

    def save(self, job: ExternalJob) -> ExternalJob:
        if job.state not in PENDING_STATES | TERMINAL_STATES:
            raise ValueError(f"Invalid external job state: {job.state}")
        path = self.path_for(job.id)
        if path.exists():
            old = self.load(job.id)
            if old.state == "COMPLETE" and job != old:
                raise ValueError("A COMPLETE external job record is immutable")
            if old.remoteTaskURL and old.remoteTaskURL != job.remoteTaskURL:
                raise ValueError("A persisted remote task URL cannot be replaced")
        self._atomic_write(path, asdict(job))
        return job

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path); temporary = ""
        finally:
            if temporary and os.path.exists(temporary): os.unlink(temporary)

    def load(self, job_id: str) -> ExternalJob:
        with self.path_for(job_id).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("version") != VERSION:
            raise ValueError(f"Unsupported external job version: {raw.get('version')}")
        return ExternalJob(**raw)

    def get(self, job_id: str) -> ExternalJob | None:
        return self.load(job_id) if self.path_for(job_id).exists() else None

    def list(self) -> tuple[ExternalJob, ...]:
        return tuple(self.load(path.stem) for path in sorted(self.root.glob("*.json")))

    def pending(self) -> tuple[ExternalJob, ...]:
        return tuple(job for job in self.list() if job.state in PENDING_STATES)

    def _dependency_groups(self) -> dict[tuple[str, str], list[ExternalJob]]:
        groups: dict[tuple[str, str], list[ExternalJob]] = {}
        for job in self.list():
            groups.setdefault((job.investigationID, job.dependencyID), []).append(job)
        return groups

    def ready_dependencies(self) -> tuple[tuple[str, str], ...]:
        """Reconstruct readiness solely from durable COMPLETE records."""
        return tuple(sorted(
            identity for identity, jobs in self._dependency_groups().items()
            if jobs and all(job.state == "COMPLETE" for job in jobs)
        ))

    def failed_dependencies(self) -> tuple[tuple[str, str], ...]:
        """Return durable dependency groups containing a terminal remote failure."""
        return tuple(sorted(
            identity for identity, jobs in self._dependency_groups().items()
            if jobs and any(job.state == "REMOTE_FAILED" for job in jobs)
        ))


class ExternalJobMonitor:
    """Perform at most one poll for each due job; never sleeps or submits."""
    def __init__(self, store: ExternalJobStore,
                 providers: dict[str, ExternalJobProvider], *, interval_seconds: int = 300):
        self.store, self.providers = store, providers
        self.interval_seconds = interval_seconds

    def poll_due(self, *, now: datetime | None = None) -> tuple[tuple[str, str], ...]:
        now = now or datetime.now(timezone.utc)
        touched: set[tuple[str, str]] = set()
        for job in self.store.pending():
            if job.nextCheckAt and datetime.fromisoformat(job.nextCheckAt) > now:
                continue
            checked = now.isoformat()
            next_check = (now + timedelta(seconds=self.interval_seconds)).isoformat()
            try:
                result = self.providers[job.provider].poll(job)
                completed = checked if result.state == "COMPLETE" else None
                updated = replace(job, state=result.state,
                    remoteResultURL=result.remote_result_url or job.remoteResultURL,
                    lastCheckedAt=checked, nextCheckAt=next_check,
                    completedAt=completed, lastOperationalError=None)
            except ExternalJobPollUnavailable as error:
                updated = replace(job, lastCheckedAt=checked, nextCheckAt=next_check,
                                  lastOperationalError=f"{type(error).__name__}: {error}")
            self.store.save(updated)
            touched.add((job.investigationID, job.dependencyID))
        ready = set(self.store.ready_dependencies())
        return tuple(sorted(identity for identity in touched if identity in ready))


def apply_external_job_wakeups(investigations: InvestigationStore,
                               ready: tuple[tuple[str, str], ...]) -> None:
    """Durably expose availability and an explicit portfolio wake marker."""
    for investigation_id, dependency_id in ready:
        investigation = investigations.load(investigation_id)
        metadata = dict(investigation.metadata)
        availability = dict(metadata.get("externalDataAvailability") or {})
        if availability.get(dependency_id) is True:
            continue
        availability[dependency_id] = True
        wake = set(metadata.get("externalJobWakeDependencies") or [])
        wake.add(dependency_id)
        metadata["externalDataAvailability"] = availability
        metadata["externalJobWakeDependencies"] = sorted(wake)
        metadata.pop("controlState", None)
        investigations.save(replace(investigation, metadata=metadata,
                                    updated_at=utc_now()))
