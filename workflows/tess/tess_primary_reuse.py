"""Exact-equivalence reuse gate for a TESS deep primary coordinator run."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from openstar_investigation import sha256_file, sha256_json
from openstar_workflow import StageOutcome, StageRequest


def coordinator_dataset_identity_matches(result, expected_dataset_id) -> bool:
    """Match a coordinator dataset result using ``id`` as its canonical identity."""
    if not isinstance(result, dict):
        return False
    if "id" in result:
        identity = result["id"]
        if "datasetID" in result and str(result["datasetID"]) != str(identity):
            return False
    elif "datasetID" in result:
        identity = result["datasetID"]
    else:
        return False
    return identity is not None and str(identity) == str(expected_dataset_id)


def run_primary(investigation, request, coordinator, *, poll_interval, timeout):
    prepared_stages = [stage for stage in investigation.stages
                       if stage.id == "001-prepare-target" and stage.status == "COMPLETE"]
    if len(prepared_stages) != 1 or not isinstance(prepared_stages[0].result, dict):
        raise RuntimeError("Missing completed stage: 001-prepare-target")
    prepared = prepared_stages[0].result
    reusable = investigation.metadata.get("reusablePrimary")
    if isinstance(reusable, dict):
        try:
            dataset_path = Path(prepared["datasetPath"]).resolve()
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            primary = json.loads(Path(request.parameters["projectPath"]).read_text(encoding="utf-8"))
            source_results = reusable.get("coordinatorResult", {}).get("datasets")
            exact = (
                reusable.get("verification") == "EXACT_FROZEN_SHALLOW_PRIMARY"
                and reusable.get("sourceProjectID") == prepared.get("sourceProjectID")
                and reusable.get("datasetID") == prepared.get("datasetID")
                and Path(str(reusable.get("datasetArtifact"))).resolve() == dataset_path
                and reusable.get("datasetSha256") == sha256_file(dataset_path)
                and reusable.get("datasetSha256") == investigation.metadata.get("datasetSha256")
                and reusable.get("sourceProjectManifestSha256") == investigation.metadata.get("sourceProjectManifestSha256")
                and reusable.get("sourceProjectManifestSha256") == sha256_file(prepared["sourceProjectPath"])
                and reusable.get("sourceEvidenceSha256") == investigation.metadata.get("sourceEvidenceSha256")
                and reusable.get("frequencySearchSha256") == sha256_json(dataset.get("frequencySearch"))
                and primary.get("workloadID") == "openstar.lomb-scargle.v1"
                and len(primary.get("datasets", [])) == 1
                and primary["datasets"][0] == prepared.get("sourceDatasetEntry")
                and sha256_json(reusable.get("coordinatorResult")) == reusable.get("coordinatorResultSha256")
                and isinstance(source_results, list) and len(source_results) == 1
                and isinstance(source_results[0], dict)
                and coordinator_dataset_identity_matches(
                    source_results[0], prepared.get("datasetID"))
                and (source_results[0].get("ticID") is None
                     or source_results[0].get("ticID") == prepared.get("ticID"))
                and (source_results[0].get("sector") is None
                     or source_results[0].get("sector") == prepared.get("sector"))
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            exact = False
        if exact:
            print("♻️ Reusing verified shallow distributed period search")
            reused_result = copy.deepcopy(reusable["coordinatorResult"])
            reused_result["reuseProvenance"] = {
                "mode": "VERIFIED_SHALLOW_PRIMARY_REUSE",
                "computeDisposition": "REUSED_SHALLOW_COMPUTE",
                "sourceScanInvestigationID": reusable["sourceScanInvestigationID"],
                "sourceWorkflowID": reusable["sourceWorkflowID"],
                "sourceWorkflowVersion": reusable["sourceWorkflowVersion"],
                "sourceCoordinatorResultSha256": reusable["coordinatorResultSha256"],
                "sourceEvidenceSha256": reusable["sourceEvidenceSha256"],
                "datasetSha256": reusable["datasetSha256"],
                "frequencySearchSha256": reusable["frequencySearchSha256"],
            }
            return StageOutcome(
                result=reused_result,
                next_stage=StageRequest("003-catalog-identity", "openstar.tess.catalog-identity", {}, request.id),
                input_hashes={"sourceProjectManifest": reusable["sourceProjectManifestSha256"],
                              "sourceDataset": reusable["datasetSha256"],
                              "sourceEvidence": reusable["sourceEvidenceSha256"],
                              "sourceCoordinatorResult": reusable["coordinatorResultSha256"],
                              "frequencySearch": reusable["frequencySearchSha256"]},
                node_contributions=dict(reusable.get("nodeContributions") or {}),
                project_ids=tuple(reusable.get("computeProjectIDs") or ()),
            )

    print("⚙️ Activating primary distributed period search")
    run = coordinator.run_project(request.parameters["projectPath"],
                                  poll_interval=poll_interval, timeout=timeout)
    result = copy.deepcopy(run.status)
    result["reuseProvenance"] = {"mode": "FRESH_DEEP_COMPUTE", "computeDisposition": "EXECUTED_DEEP_COMPUTE"}
    return StageOutcome(
        result=result,
        next_stage=StageRequest("003-catalog-identity", "openstar.tess.catalog-identity", {}, request.id),
        node_contributions=run.node_contributions, project_ids=(run.project_id,),
    )
