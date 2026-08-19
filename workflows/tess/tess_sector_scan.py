"""Shallow TESS sector scan workflow: materialize, distribute, persist evidence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from openstar_autonomy import ScientificBranch
from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import ArtifactReference, Investigation, InvestigationStore, sha256_file, sha256_json
from openstar_targets import InvestigationTarget
from openstar_workflow import RetryableExecutionError, StageOutcome, StageRequest, WorkflowEngine

from .tess_preprocessing import broad_tess_frequency_search, read_and_prepare_tess_light_curve
from .tess_sector_archive import (
    TessArchiveProduct,
    TessArchiveTransientError,
    TessSectorArchiveProvider,
    TessSectorInventory,
)

WORKFLOW_ID = "openstar.workflow.tess-sector-scan.v1"
WORKFLOW_VERSION = "1"
MATERIALIZE_HANDLER = "openstar.tess-sector-scan.materialize-light-curve"
SCAN_HANDLER = "openstar.tess-sector-scan.broad-distributed-scan"
EVIDENCE_HANDLER = "openstar.tess-sector-scan.persist-scan-evidence"


def _atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); temporary = ""
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)


def _artifact(path: Path) -> ArtifactReference:
    return ArtifactReference(str(path.resolve()), sha256_file(path), "application/json")


class TessSectorScanTargetSource:
    id = "openstar.tess-sector-inventory-targets"
    version = "1"
    def __init__(self, inventory: TessSectorInventory, max_targets: int | None = None):
        if max_targets is not None and max_targets < 1: raise ValueError("max_targets must be positive")
        self.inventory, self.max_targets = inventory, max_targets

    def enumerate_targets(self) -> Sequence[InvestigationTarget]:
        entries = self.inventory.entries[:self.max_targets]
        return tuple(InvestigationTarget(
            id=f"tess-sector-{self.inventory.sector}-tic-{entry.product.tic_id}",
            investigation_id=f"tess-sector-scan-{self.inventory.sector}-tic-{entry.product.tic_id}",
            workflow_id=WORKFLOW_ID, workflow_version=WORKFLOW_VERSION, priority=position,
            metadata={"sector": self.inventory.sector, "ticID": entry.product.tic_id,
                      "targetName": entry.product.target_name, "archiveProduct": asdict(entry.product),
                      "archiveProvider": {"id": self.inventory.provider_id, "version": self.inventory.provider_version},
                      "selectionAlgorithmVersion": self.inventory.selection_algorithm_version},
        ) for position, entry in enumerate(entries))


def plan_tess_sector_scan(investigation: Investigation, target: InvestigationTarget):
    if not investigation.stages:
        return (ScientificBranch("materialize-selected-archive-product", StageRequest(
            "001-materialize-light-curve", MATERIALIZE_HANDLER, {}, None)),)

    latest = investigation.stages[-1]
    if latest.status != "COMPLETE":
        # RUNNING and FAILED stages are deliberately classified and recovered
        # by InvestigationLifecycleDriver, not by this scientific planner.
        return ()
    if latest.stop:
        return ()

    continuation = latest.next_stage
    if not isinstance(continuation, dict):
        raise RuntimeError(
            f"Completed nonterminal scan stage has no continuation: {latest.id}"
        )
    request = StageRequest(
        id=str(continuation["id"]),
        handler_id=str(continuation["handler_id"]),
        parameters=dict(continuation.get("parameters") or {}),
        triggered_by_stage_id=continuation.get("triggered_by_stage_id"),
    )
    if any(stage.id == request.id for stage in investigation.stages):
        # This is reachable only for malformed/manual state; never manufacture
        # a duplicate stage attempt. Generic lifecycle recovery owns real
        # RUNNING/FAILED attempts and encounters those as the latest stage.
        raise RuntimeError(f"Persisted scan continuation was already attempted: {request.id}")
    return (ScientificBranch(
        f"resume-persisted-continuation-{request.id}", request
    ),)


def _result(investigation: Investigation, handler: str) -> dict[str, Any]:
    for stage in reversed(investigation.stages):
        if stage.handler_id == handler and stage.status == "COMPLETE" and stage.result is not None:
            return stage.result
    raise RuntimeError(f"Missing completed stage: {handler}")


def _dataset_status(status: dict[str, Any]) -> dict[str, Any]:
    datasets = status.get("datasets")
    if isinstance(datasets, list) and datasets: return dict(datasets[0])
    return status


def _next_stage(request: StageRequest, label: str, handler: str, parameters: dict[str, Any]):
    prefix, separator, _ = request.id.partition("-")
    if not separator or not prefix.isdigit():
        raise ValueError(f"Stage id must begin with an integer prefix: {request.id}")
    return StageRequest(
        f"{int(prefix) + 1:03d}-{label}", handler, parameters, request.id
    )


def register_tess_sector_scan_handlers(
    store: InvestigationStore, coordinator: OpenStarCoordinatorClient,
    provider: TessSectorArchiveProvider, *, poll_interval: float = 1.0,
    timeout: float | None = None,
    preprocessing: Callable[[Path], Any] | None = None,
    scan_profile: dict[str, int | float] | None = None,
) -> WorkflowEngine:
    engine = WorkflowEngine(store)
    profile = dict(scan_profile or broad_tess_frequency_search())
    preprocessing = preprocessing or read_and_prepare_tess_light_curve

    def materialize(investigation, request):
        raw = investigation.metadata.get("archiveProduct")
        if not isinstance(raw, dict): raise RuntimeError("Investigation has no archive inventory product.")
        product = TessArchiveProduct(**raw)
        sector = int(investigation.metadata["sector"])
        if product.sector != sector or product.tic_id != investigation.metadata.get("ticID"):
            raise RuntimeError("Archive product identity does not match the investigation.")
        artifact_dir = store.directory_for(investigation.id) / "artifacts" / "scan-input"
        try:
            downloaded = provider.download_light_curve(product, artifact_dir / "source")
        except TessArchiveTransientError as error:
            archive_provider = investigation.metadata.get("archiveProvider")
            raise RetryableExecutionError(
                str(error),
                result={
                    "sector": sector,
                    "ticID": product.tic_id,
                    "productURI": product.product_uri or product.data_uri,
                    "productFilename": product.product_filename,
                    "archiveProvider": dict(archive_provider)
                    if isinstance(archive_provider, dict) else archive_provider,
                },
                input_hashes={"archiveProduct": sha256_json(asdict(product))},
            ) from error
        prepared = preprocessing(downloaded)
        dataset_path, project_path = artifact_dir / "dataset.json", artifact_dir / "project.json"
        dataset_id = f"tess-sector-{sector}-tic-{product.tic_id}"
        dataset = {"id": dataset_id, "targetName": product.target_name, "mission": "TESS",
                   "coordinates": list(prepared.coordinates), "values": list(prepared.values),
                   "times": list(prepared.coordinates), "flux": list(prepared.values),
                   "timeUnit": "days", "timeReference": "relative-to-first-distributed-sample",
                   "numericRepresentation": "Float32", "fluxUnit": "normalized",
                   "frequencySearch": profile,
                   "metadata": {"sector": sector, "ticID": product.tic_id,
                                "cadenceSeconds": product.cadence_seconds,
                                "sourceSampleCount": prepared.source_sample_count,
                                "finiteSampleCount": prepared.finite_sample_count,
                                "sampleCount": prepared.sample_count, "baselineDays": prepared.baseline_days,
                                "originalTimeOriginDays": prepared.time_origin_days,
                                "archiveProduct": asdict(product)}}
        _atomic_json(dataset_path, dataset)
        manifest = {"id": f"tess-sector-scan-{sector}-tic-{product.tic_id}",
                    "name": f"TESS sector {sector} broad scan TIC {product.tic_id}",
                    "workloadID": "openstar.lomb-scargle.v1",
                    "datasets": [{"id": dataset_id, "path": str(dataset_path.resolve()),
                                  "targetName": product.target_name, "ticID": product.tic_id, "sector": sector}]}
        _atomic_json(project_path, manifest)
        result = {"datasetID": dataset_id, "datasetPath": str(dataset_path.resolve()),
                  "datasetSha256": sha256_file(dataset_path), "projectPath": str(project_path.resolve()),
                  "projectManifestSha256": sha256_file(project_path), "downloadedProductPath": str(downloaded.resolve()),
                  "downloadedProductSha256": sha256_file(downloaded), "archiveProduct": asdict(product),
                  "sampleCount": prepared.sample_count, "sourceSampleCount": prepared.source_sample_count,
                  "baselineDays": prepared.baseline_days, "cadenceSeconds": product.cadence_seconds}
        return StageOutcome(result, _next_stage(request, "broad-distributed-scan", SCAN_HANDLER,
                            {"projectPath": str(project_path.resolve())}),
                            input_hashes={"archiveProduct": sha256_json(asdict(product)),
                                          "downloadedProduct": result["downloadedProductSha256"]},
                            artifacts=(_artifact(dataset_path), _artifact(project_path)))

    def scan(investigation, request):
        run = coordinator.run_project(request.parameters["projectPath"], poll_interval=poll_interval, timeout=timeout)
        return StageOutcome(run.status, _next_stage(request, "persist-scan-evidence", EVIDENCE_HANDLER, {}),
                            node_contributions=run.node_contributions, project_ids=(run.project_id,))

    def persist(investigation, request):
        prepared = _result(investigation, MATERIALIZE_HANDLER)
        run_status = _result(investigation, SCAN_HANDLER)
        target = _dataset_status(run_status)
        scan_stage = next(stage for stage in reversed(investigation.stages) if stage.handler_id == SCAN_HANDLER)
        frequency = target.get("bestFrequency") if target.get("bestFrequency") is not None else target.get("candidateFrequency")
        period = target.get("bestPeriodDays") if target.get("bestPeriodDays") is not None else target.get("candidatePeriodDays")
        evidence = {"sector": investigation.metadata["sector"], "ticID": investigation.metadata["ticID"],
                    "targetName": investigation.metadata.get("targetName"), "archiveProduct": prepared["archiveProduct"],
                    "datasetArtifact": prepared["datasetPath"], "datasetSha256": prepared["datasetSha256"],
                    "computeProjectIDs": list(scan_stage.provenance.project_ids if scan_stage.provenance else ()),
                    "nodeContributions": dict(scan_stage.provenance.node_contributions if scan_stage.provenance else {}),
                    "sampleCount": prepared["sampleCount"], "sourceSampleCount": prepared["sourceSampleCount"],
                    "baselineDays": prepared["baselineDays"], "cadenceSeconds": prepared["cadenceSeconds"],
                    "bestFrequency": frequency, "bestPeriodDays": period,
                    "bestPower": target.get("bestPower") if target.get("bestPower") is not None else target.get("candidatePower"),
                    "periodStatus": target.get("periodStatus"), "periodConfidence": target.get("periodConfidence"),
                    "foldCoherence": target.get("candidateFoldCoherence"),
                    "coverageComplete": target.get("coverageComplete")}
        evidence_path = store.directory_for(investigation.id) / "artifacts" / "scan-evidence.json"
        _atomic_json(evidence_path, evidence)
        return StageOutcome(evidence, stop=True, input_hashes={"dataset": prepared["datasetSha256"],
                            "coordinatorResult": sha256_json(run_status)}, artifacts=(_artifact(evidence_path),))

    engine.register_handler(MATERIALIZE_HANDLER, materialize)
    engine.register_handler(SCAN_HANDLER, scan)
    engine.register_handler(EVIDENCE_HANDLER, persist)
    return engine
