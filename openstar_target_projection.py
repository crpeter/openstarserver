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
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _first(objects: Iterable[dict], *names: str) -> Any:
    lowered = {name.lower() for name in names}
    for obj in objects:
        for key, value in obj.items():
            if key.lower() in lowered and value not in (None, "", [], {}):
                return value
    return None


def _numbers(value: Any) -> list[int]:
    values = value if isinstance(value, list) else [value]
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
        rows.append({
            "id": str(stage.get("stageID") or stage.get("id") or index + 1),
            "handler": stage.get("handlerID") or stage.get("handler") or "unknown",
            "status": str(stage.get("status") or "UNKNOWN").upper(),
            "updatedAt": stage.get("updatedAt") or stage.get("completedAt"),
            "result": result,
        })
    return rows


def project_investigation(snapshot: dict[str, Any], *, catalog_run: Any = None,
                          record_mtime: float | None = None) -> dict[str, Any]:
    """Normalize one heterogeneous snapshot without assuming a fixed workflow."""
    if not isinstance(snapshot, dict):
        raise ValueError("investigation snapshot must be an object")
    stages = _stage_rows(snapshot)
    objects = list(_walk(snapshot))
    metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    tic = _first(objects, "ticID", "ticId", "tic_id", "tic")
    gaia = _first(objects, "gaiaID", "gaiaId", "gaia_id", "gaiaSourceID", "source_id")
    identity = f"tic:{tic}" if tic is not None else f"gaia:{gaia}" if gaia is not None else None
    investigation_id = str(snapshot.get("investigationID") or snapshot.get("id") or
                           metadata.get("investigationID") or "unknown")
    if identity is None:
        identity = f"investigation:{investigation_id}"
    target_id = "target_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    meaningful = {}
    for stage in stages:
        result = stage["result"]
        if any(_first([result], key) is not None for key in MEANINGFUL):
            meaningful = result
    completed = sum(row["status"] in {"COMPLETE", "COMPLETED", "SUCCEEDED"} for row in stages)
    failed = sum(row["status"] in {"FAILED", "ERROR"} for row in stages)
    status = str(snapshot.get("status") or getattr(catalog_run, "status", "UNKNOWN")).upper()
    detected = _first(objects, "detectedPeriod", "detectedPeriodDays", "periodDays", "period")
    physical = _first(objects, "resolvedPhysicalPeriod", "physicalPeriodDays", "orbitalPeriodDays")
    primary = _numbers(_first(objects, "primarySectors", "primarySector", "sectors", "sector"))
    independent = _numbers(_first(objects, "independentSectors", "independentSector"))
    classification = _first(list(_walk(meaningful)), "classification", "latestClassification")
    meaningful_objects = list(_walk(meaningful))
    recommendation = _first(meaningful_objects, "recommendedNextTest", "recommendation", "nextTest") or _first(objects, "recommendedNextTest", "recommendation", "nextTest")
    source = _first(meaningful_objects, "sourceAttribution", "sourceLocalization", "localizedSource") or _first(objects, "sourceAttribution", "sourceLocalization", "localizedSource")
    companion = _first(meaningful_objects, "companionNature", "companionEvidence", "companionStatus") or _first(objects, "companionNature", "companionEvidence", "companionStatus")
    mechanism = _first(meaningful_objects, "physicalMechanism", "mechanismStatus", "mechanism") or _first(objects, "physicalMechanism", "mechanismStatus", "mechanism")
    claim = _first(list(_walk(meaningful)), "claim", "currentClaim", "conclusion") or classification
    answer_key = bool(_first(objects, "answerKeyUsed", "usedAnswerKey", "answerKeyUse"))
    degraded = failed > 0 or status in {"FAILED", "BLOCKED", "RECOVERY_REQUIRED", "DEGRADED"}
    created = snapshot.get("createdAt") or getattr(catalog_run, "created_at", None)
    updated = snapshot.get("updatedAt") or record_mtime or getattr(catalog_run, "updated_at", None)
    artifacts = _first(objects, "artifacts", "artifactReferences", "reports") or []
    hashes = []
    for obj in objects:
        for key, value in obj.items():
            if "hash" in key.lower() and isinstance(value, str):
                hashes.append({"name": key, "value": value})
    name = _first(objects, "targetDisplayName", "targetName", "displayName", "name")
    if not name:
        name = f"TIC {tic}" if tic is not None else f"Gaia {gaia}" if gaia is not None else investigation_id
    return {
        "targetID": target_id, "identityKey": identity, "targetName": str(name),
        "ticID": str(tic) if tic is not None else None,
        "gaiaID": str(gaia) if gaia is not None else None,
        "coordinates": {"ra": _first(objects, "ra", "raDegrees"),
                        "dec": _first(objects, "dec", "declination", "decDegrees")},
        "datasetID": _first(objects, "datasetID", "datasetId"),
        "projectID": _first(objects, "projectID", "projectId"),
        "investigationID": investigation_id, "status": status,
        "workflow": _first(objects, "workflowID", "workflow", "workflowVersion"),
        "runID": getattr(catalog_run, "run_id", None),
        "runKind": getattr(catalog_run, "kind", None), "createdAt": created, "updatedAt": updated,
        "stages": stages, "stageCounts": {"completed": completed, "failed": failed, "total": len(stages)},
        "primarySectors": primary, "independentSectors": independent,
        "detectedPeriod": detected, "resolvedPhysicalPeriod": physical,
        "currentClaim": claim, "classification": classification,
        "sourceAttribution": source, "physicalMechanism": mechanism,
        "companionNature": companion, "recommendedNextTest": recommendation,
        "reportAvailable": bool(artifacts), "artifacts": artifacts,
        "degraded": degraded, "recoveryRequired": status == "RECOVERY_REQUIRED",
        "answerKeyUsed": answer_key, "provenanceHashes": hashes,
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
            rows = [r for r in rows if bool(r.get("companionNature") or r.get("sourceAttribution")) == (resolution == "resolved")]
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
        compact = [{k: v for k, v in row.items() if k not in {"runs", "stages", "artifacts", "provenanceHashes"}}
                   for row in rows[start:start + size]]
        all_rows = list(self._targets.values())
        stats = {"totalTargets": len(all_rows),
                 "activeInvestigations": sum(r["status"] in {"RUNNING", "ACTIVE"} for r in all_rows),
                 "completedInvestigations": sum(r["status"] in {"COMPLETE", "COMPLETED", "FINISHED"} for r in all_rows),
                 "unresolvedTargets": sum(not (r["sourceAttribution"] or r["companionNature"]) for r in all_rows),
                 "sourceLocalizedTargets": sum(bool(r["sourceAttribution"]) for r in all_rows),
                 "companionNatureResolvedTargets": sum(bool(r["companionNature"]) for r in all_rows),
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
        return {"targetID": target_id, "stageTimeline": [{k: row[k] for k in
                ("id", "handler", "status", "updatedAt")} for row in latest["stages"]],
                "sectorSupport": {"primary": latest["primarySectors"], "independent": latest["independentSectors"]},
                "periods": {"detected": latest["detectedPeriod"], "physical": latest["resolvedPhysicalPeriod"]},
                "differenceImage": None, "centroid": None,
                "message": "Additional visualization evidence was not recorded for this run."}


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
