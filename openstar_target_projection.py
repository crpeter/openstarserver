"""Read-only dashboard projection of cataloged investigation history.

This module deliberately has no dependency on workflow execution.  It reads only
catalog-approved roots and emits browser-safe summaries without filesystem paths.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from openstar_science_runs import ScienceRunCatalog

EXCLUDED_KINDS = {"tess-sector-sweep", "tess-sector-reconstruction"}
MEANINGFUL = ("classification", "claim", "conclusion", "recommendation",
              "sourceAttribution", "physicalMechanism", "companionNature")


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


def _first(objects: Iterable[dict], *names: str) -> Any:
    lowered = {name.lower() for name in names}
    for obj in objects:
        for key, value in obj.items():
            if key.lower() in lowered and value not in (None, "", [], {}):
                return value
    return None


def _epoch(value: Any, fallback: float | None = None) -> float | None:
    """Normalize canonical ISO-8601 or legacy numeric timestamps for the browser."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            try:
                from datetime import datetime, timezone
                text = value.strip()
                parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except ValueError:
                pass
    return fallback


def _numbers(value: Any) -> list[int]:
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        try:
            number = int(item.get("sector") if isinstance(item, dict) else item)
            if number > 0 and number not in result:
                result.append(number)
        except (TypeError, ValueError):
            pass
    return result


def _stage_rows(snapshot: dict) -> list[dict[str, Any]]:
    raw = snapshot.get("stages") or snapshot.get("stageStates") or []
    if isinstance(raw, dict):
        raw = [{"id": key, **(value if isinstance(value, dict) else {"result": value})}
               for key, value in raw.items()]
    rows = []
    for index, stage in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(stage, dict):
            continue
        result = stage.get("result") if isinstance(stage.get("result"), dict) else {}
        started = _epoch(stage.get("started_at") or stage.get("startedAt"))
        completed = _epoch(stage.get("completed_at") or stage.get("completedAt"))
        rows.append({
            "id": str(stage.get("id") or stage.get("stageID") or index + 1),
            "handler": stage.get("handler_id") or stage.get("handlerID") or stage.get("handler") or "unknown",
            "status": str(stage.get("status") or "UNKNOWN").upper(),
            "startedAt": started, "completedAt": completed,
            "updatedAt": completed or started,
            "failureClassification": stage.get("failure_classification") or stage.get("failureClassification"),
            "result": result,
            "artifacts": stage.get("artifacts") if isinstance(stage.get("artifacts"), (list, tuple)) else [],
            "provenance": stage.get("provenance") if isinstance(stage.get("provenance"), dict) else None,
        })
    return rows


def _metadata_identity(metadata: dict[str, Any], *names: str) -> Any:
    """Read only target-level metadata; never adopt identifiers from evidence candidates."""
    value = _first([metadata], *names)
    target = metadata.get("target")
    if value is None and isinstance(target, dict):
        value = _first([target], *names)
    return value


def _latest_fields(stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply authoritative result fields in stage order, including explicit false."""
    aliases = {
        "classification": ("classification", "latestClassification"),
        "recommendedNextTest": ("recommendedNextTest", "recommendation", "nextTest"),
        "currentClaim": ("claim", "currentClaim", "conclusion"),
        "sourceAttribution": ("sourceAttribution", "sourceLocalization", "localizedSource"),
        "companionNature": ("companionNature", "companionEvidence", "companionStatus"),
        "physicalMechanism": ("physicalMechanism", "mechanismStatus", "mechanism"),
        "sourceAttributionResolved": ("sourceAttributionResolved",),
        "companionNatureResolved": ("companionNatureResolved",),
        "physicalMechanismResolved": ("physicalMechanismResolved",),
        "physicalCycleResolved": ("physicalCycleResolved",),
        "resolvedPhysicalPeriodDays": ("resolvedPhysicalPeriodDays",),
        "detectedPeriodDays": ("detectedPeriodDays", "detectedPeriod", "recurrentPhotometricPeriodDays"),
    }
    latest: dict[str, Any] = {}
    for stage in stages:
        if stage["status"] not in {"COMPLETE", "COMPLETED", "SUCCEEDED"}:
            continue
        objects = list(_authoritative_objects(stage["result"]))
        for output, names in aliases.items():
            value = _first(objects, *names)
            # _first deliberately preserves False because it is authoritative.
            if value is not None:
                latest[output] = value
    return latest


def _authoritative_objects(value: Any):
    """Walk a result while excluding catalog/model alternatives from decisions."""
    if not isinstance(value, dict):
        return
    yield value
    for key, child in value.items():
        lowered = key.lower()
        if any(term in lowered for term in ("candidate", "catalogsource", "neighbor",
                                             "modelcomponent", "comparisonobject")):
            continue
        if isinstance(child, dict):
            yield from _authoritative_objects(child)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, dict):
                    yield from _authoritative_objects(item)


def project_investigation(snapshot: dict[str, Any], *, catalog_run: Any = None,
                          record_mtime: float | None = None) -> dict[str, Any]:
    """Normalize the canonical InvestigationStore snapshot plus useful legacy forms."""
    if not isinstance(snapshot, dict):
        raise ValueError("investigation snapshot must be an object")
    stages = _stage_rows(snapshot)
    metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    tic = _metadata_identity(metadata, "ticID", "ticId", "tic_id", "tic")
    gaia = _metadata_identity(metadata, "gaiaID", "gaiaId", "gaia_id", "gaiaSourceID",
                              "gaiaDR3SourceID")
    investigation_id = str(snapshot.get("id") or snapshot.get("investigationID") or
                           metadata.get("investigationID") or "unknown")
    identity = (f"tic:{tic}" if tic is not None else f"gaia:{gaia}"
                if gaia is not None else f"investigation:{investigation_id}")
    target_id = "target_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    latest = _latest_fields(stages)
    objects = list(_walk(snapshot))
    completed = sum(row["status"] in {"COMPLETE", "COMPLETED", "SUCCEEDED"} for row in stages)
    failed = sum(row["status"] in {"FAILED", "ERROR"} for row in stages)
    status = str(snapshot.get("status") or getattr(catalog_run, "status", "UNKNOWN")).upper()
    primary = _numbers(_metadata_identity(metadata, "primarySectors", "primarySector", "sectors", "sector"))
    independent = _numbers(_metadata_identity(metadata, "independentSectors", "independentSector"))
    # Sector evidence may be stage-produced, but target identity never is.
    if not independent:
        for stage in stages:
            value = _first(list(_walk(stage["result"])), "independentSectors", "supportingSectors")
            if value is not None: independent = _numbers(value)
    artifacts = [artifact for stage in stages for artifact in stage["artifacts"]]
    report = _first(list(_walk(metadata)), "report", "reportPath")
    if report is not None: artifacts.append({"path": report})
    hashes = []
    for obj in objects:
        for key, value in obj.items():
            if ("hash" in key.lower() or key.lower() == "sha256") and isinstance(value, str):
                hashes.append({"name": key, "value": value})
    name = _metadata_identity(metadata, "targetDisplayName", "targetName", "displayName", "name")
    name = name or (f"TIC {tic}" if tic is not None else f"Gaia {gaia}" if gaia is not None else investigation_id)
    created = _epoch(snapshot.get("created_at") or snapshot.get("createdAt"), getattr(catalog_run, "created_at", None))
    updated = _epoch(snapshot.get("updated_at") or snapshot.get("updatedAt"), record_mtime or getattr(catalog_run, "updated_at", None))
    workflow_id = snapshot.get("workflow_id") or snapshot.get("workflowID")
    workflow_version = snapshot.get("workflow_version") or snapshot.get("workflowVersion")
    claim = latest.get("currentClaim") or latest.get("classification")
    return {
        "targetID": target_id, "identityKey": identity, "targetName": str(name),
        "ticID": str(tic) if tic is not None else None, "gaiaID": str(gaia) if gaia is not None else None,
        "coordinates": {"ra": _metadata_identity(metadata, "ra", "raDeg", "raDegrees"),
                        "dec": _metadata_identity(metadata, "dec", "decDeg", "declination", "decDegrees")},
        "datasetID": _metadata_identity(metadata, "datasetID", "datasetId", "dataset_id"),
        "projectID": _metadata_identity(metadata, "projectID", "projectId", "project_id"),
        "investigationID": investigation_id, "status": status,
        "workflow": workflow_id, "workflowVersion": workflow_version,
        "runID": getattr(catalog_run, "run_id", None), "runKind": getattr(catalog_run, "kind", None),
        "createdAt": created, "updatedAt": updated, "stages": stages,
        "stageCounts": {"completed": completed, "failed": failed, "total": len(stages)},
        "primarySectors": primary, "independentSectors": independent,
        "detectedPeriod": latest.get("detectedPeriodDays"),
        "resolvedPhysicalPeriod": latest.get("resolvedPhysicalPeriodDays"),
        "physicalCycleResolved": latest.get("physicalCycleResolved"),
        "currentClaim": claim, "classification": latest.get("classification"),
        "sourceAttribution": latest.get("sourceAttribution"),
        "sourceAttributionResolved": latest.get("sourceAttributionResolved"),
        "physicalMechanism": latest.get("physicalMechanism"),
        "physicalMechanismResolved": latest.get("physicalMechanismResolved"),
        "companionNature": latest.get("companionNature"),
        "companionNatureResolved": latest.get("companionNatureResolved"),
        "recommendedNextTest": latest.get("recommendedNextTest"),
        "reportAvailable": bool(artifacts), "artifacts": artifacts,
        "degraded": failed > 0 or status in {"FAILED", "BLOCKED", "RECOVERY_REQUIRED", "DEGRADED"},
        "recoveryRequired": status == "RECOVERY_REQUIRED",
        "answerKeyUsed": bool(_first([metadata], "answerKeyUsed", "usedAnswerKey", "answerKeyUse")),
        "provenanceHashes": hashes,
    }


class TargetProjectionStore:
    """Bounded-TTL, failure-isolating view over catalog-approved roots."""
    def __init__(self, catalog: str | Path, ttl: float = 3.0):
        self.catalog, self.ttl = catalog, ttl
        self._lock = threading.Lock()
        self._until = 0.0
        self._targets: dict[str, dict] = {}

    def _refresh(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now < self._until:
                return
            grouped: dict[str, list[dict]] = {}
            try:
                runs = ScienceRunCatalog(self.catalog).list_runs()
            except Exception:
                runs = []
            for run in runs:
                if run.kind in EXCLUDED_KINDS:
                    continue
                directory = Path(run.state_root) / "investigations"
                try:
                    records = list(directory.glob("*/investigation.json"))[:2000]
                except OSError:
                    continue
                for record in records:
                    try:
                        stat = record.stat()
                        projected = project_investigation(json.loads(record.read_text()),
                                                          catalog_run=run, record_mtime=stat.st_mtime)
                        grouped.setdefault(projected["targetID"], []).append(projected)
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        continue
            targets = {}
            for target_id, history in grouped.items():
                history.sort(key=lambda row: (row.get("updatedAt") or 0, row["investigationID"]), reverse=True)
                latest = history[0]
                targets[target_id] = {**latest, "runCount": len(history), "runs": history}
            self._targets, self._until = targets, now + self.ttl

    def list(self, params: dict[str, list[str]]) -> dict[str, Any]:
        self._refresh()
        rows = list(self._targets.values())
        get = lambda key, default="": (params.get(key) or [default])[0]
        query = get("q").casefold().strip()
        if query:
            rows = [r for r in rows if query in " ".join(str(r.get(k) or "") for k in
                    ("targetName", "ticID", "gaiaID", "investigationID")).casefold()]
        for key, field in (("status", "status"), ("classification", "classification"),
                           ("nextTest", "recommendedNextTest")):
            value = get(key).casefold()
            if value:
                rows = [r for r in rows if value == str(r.get(field) or "").casefold()]
        sector = get("sector")
        if sector:
            rows = [r for r in rows if sector in map(str, r["primarySectors"] + r["independentSectors"])]
        resolution = get("resolution")
        if resolution:
            rows = [r for r in rows if bool(r.get("sourceAttributionResolved") is True
                    or r.get("companionNatureResolved") is True
                    or r.get("physicalMechanismResolved") is True
                    or r.get("physicalCycleResolved") is True) == (resolution == "resolved")]
        health = get("health")
        if health:
            rows = [r for r in rows if r["degraded"] == (health == "degraded")]
        sort = get("sort", "updated")
        if sort == "depth": rows.sort(key=lambda r: (-r["stageCounts"]["total"], r["targetID"]))
        elif sort == "status": rows.sort(key=lambda r: (r["status"], r["targetID"]))
        elif sort == "identity": rows.sort(key=lambda r: (r["identityKey"], r["targetID"]))
        else: rows.sort(key=lambda r: (-(r.get("updatedAt") or 0), r["targetID"]))
        try: page, size = max(1, int(get("page", "1"))), min(100, max(1, int(get("pageSize", "24"))))
        except ValueError: page, size = 1, 24
        total = len(rows); start = (page - 1) * size
        compact = [_browser_safe({k: v for k, v in row.items() if k not in {"runs", "stages", "artifacts", "provenanceHashes"}})
                   for row in rows[start:start + size]]
        all_rows = list(self._targets.values())
        stats = {"totalTargets": len(all_rows),
                 "activeInvestigations": sum(r["status"] in {"RUNNING", "ACTIVE"} for r in all_rows),
                 "completedInvestigations": sum(r["status"] in {"COMPLETE", "COMPLETED", "FINISHED"} for r in all_rows),
                 "unresolvedTargets": sum(not (r["sourceAttributionResolved"] is True
                     or r["companionNatureResolved"] is True
                     or r["physicalMechanismResolved"] is True
                     or r["physicalCycleResolved"] is True) for r in all_rows),
                 "sourceLocalizedTargets": sum(r["sourceAttributionResolved"] is True for r in all_rows),
                 "companionNatureResolvedTargets": sum(r["companionNatureResolved"] is True for r in all_rows),
                 "physicalMechanismResolvedTargets": sum(r["physicalMechanismResolved"] is True for r in all_rows),
                 "degradedTargets": sum(r["degraded"] for r in all_rows)}
        return {"targets": compact, "page": page, "pageSize": size, "total": total, "stats": stats}

    def detail(self, target_id: str) -> dict[str, Any] | None:
        self._refresh()
        value = self._targets.get(target_id) if target_id.startswith("target_") else None
        return _browser_safe(value) if value else None

    def visuals(self, target_id: str) -> dict[str, Any] | None:
        detail = self.detail(target_id)
        if not detail: return None
        latest = detail["runs"][0]
        result_objects = [obj for stage in latest["stages"] for obj in _walk(stage["result"])]
        def evidence(*names: str) -> dict[str, Any]:
            value = _first(reversed(result_objects), *names)
            return ({"status": "available", "data": _browser_safe(value)} if value is not None
                    else {"status": "unavailable", "reason": "not_recorded_for_run"})
        return {"targetID": target_id,
                "sectorSupport": {"status": "available" if latest["primarySectors"] or latest["independentSectors"] else "unavailable",
                    "primary": latest["primarySectors"], "independent": latest["independentSectors"],
                    "reason": None if latest["primarySectors"] or latest["independentSectors"] else "not_recorded_for_run"},
                "periods": {"status": "available" if latest["detectedPeriod"] is not None or latest["resolvedPhysicalPeriod"] is not None else "unavailable",
                    "detected": latest["detectedPeriod"], "physical": latest["resolvedPhysicalPeriod"],
                    "physicalCycleResolved": latest["physicalCycleResolved"],
                    "reason": None if latest["detectedPeriod"] is not None or latest["resolvedPhysicalPeriod"] is not None else "not_recorded_for_run"},
                "differenceImage": evidence("differenceImage", "differenceImageSummary", "differenceImageEvidence"),
                "centroid": evidence("measuredCentroid", "centroidMeasurement", "centroidEvidence"),
                "sourceDistances": evidence("sourceDistances", "sourceDistanceComparison", "catalogSourceDistances"),
                "independentSectorAgreement": evidence("independentSectorAgreement", "sectorAgreement")}


def _browser_safe(value: Any, key: str = "") -> Any:
    """Remove local locations while retaining artifact names and science metadata."""
    if isinstance(value, dict):
        return {k: _browser_safe(v, k) for k, v in value.items()
                if k.lower() != "stateroot"}
    if isinstance(value, list):
        return [_browser_safe(item, key) for item in value]
    if isinstance(value, str) and (key.lower().endswith("path") or Path(value).is_absolute()):
        return Path(value).name
    return value
