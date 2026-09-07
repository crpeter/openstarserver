"""Publish the verified bounded coarse domain under support-aware identities.

Only the four saved v1 winners are reproduced by the existing source verifier.
This builder never searches the grid or promotes a persisted v1 result.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from openstar_workloads.plugins import supported_morphology_grid as workload
from workflows.microlensing import build_anomaly_morphology_coarse_grid as coarse_builder
from workflows.microlensing import validate_anomaly_morphology_coarse_grid as validation
from workflows.microlensing.coarse_grid import _assert_identity_free, _atomic_write_bytes

BUILD_MANIFEST_SCHEMA_ID = "openstar.microlensing-supported-anomaly-morphology-grid-build.v1"
BUILD_MANIFEST_VERSION = "1.0"
ALGORITHM_ID = "openstar.microlensing-supported-anomaly-morphology-grid-build.v1"
PROJECT_RELATIVE_PATH = "project.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"


class SupportedAnomalyMorphologyGridBuildError(RuntimeError):
    """Source verification or atomic publication failed."""


def _verify_validation_root(root, verified, grid, investigation):
    root = coarse_builder._regular_directory(root, "coarse validation root")
    if {entry.name for entry in root.iterdir()} != {
        validation.RESULT_RELATIVE_PATH, validation.MANIFEST_RELATIVE_PATH,
    }:
        raise ValueError("coarse validation artifact set is incomplete or unexpected")
    report_bytes, report = validation._read_json_file(root / validation.RESULT_RELATIVE_PATH, "validation report")
    manifest_bytes, manifest = validation._read_json_file(root / validation.MANIFEST_RELATIVE_PATH, "validation manifest")
    # Reconstruct every report field, including accepted winners, metrics,
    # support, boundaries and comparisons. Classification alone is insufficient.
    validation._require_equal(report, validation._report(verified, grid, investigation), "validation report")
    expected_manifest = {
        "artifactManifestSchemaID": validation.MANIFEST_SCHEMA_ID,
        "artifactManifestVersion": validation.RESULT_VERSION,
        "resultSchemaID": validation.RESULT_SCHEMA_ID_REPORT,
        "resultVersion": validation.RESULT_VERSION,
        "relativeArtifactPaths": {
            "result": validation.RESULT_RELATIVE_PATH,
            "artifactManifest": validation.MANIFEST_RELATIVE_PATH,
        },
        "outputSHA256s": {validation.RESULT_RELATIVE_PATH: coarse_builder._sha256_bytes(report_bytes)},
        "inputHashes": {
            "morphologyPreparation": verified.preparation_file_sha256,
            "morphologyContractFile": verified.contract_file_sha256,
            "morphologyContractCanonical": verified.contract_canonical_sha256,
            "morphologyArtifactManifest": verified.manifest_file_sha256,
            "morphologyDatasets": dict(verified.series_file_sha256s),
            "coarseProjectArtifacts": dict(grid.artifact_sha256s),
            "coarseInvestigationRecord": investigation.investigation_sha256,
            "coarseStageLedgers": dict(investigation.stage_ledger_sha256s),
        },
        "parentHashes": verified.preparation["parentHashes"],
        "parentIDs": verified.preparation["parentIDs"],
        "projectID": grid.project["id"], "investigationID": investigation.investigation_id,
        "runStageID": investigation.run_stage_id,
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    validation._require_equal(manifest, expected_manifest, "validation manifest")
    _assert_identity_free((report_bytes, manifest_bytes))
    coarse_builder._assert_identity_isolated((report, manifest))
    return manifest, {
        validation.RESULT_RELATIVE_PATH: coarse_builder._sha256_bytes(report_bytes),
        validation.MANIFEST_RELATIVE_PATH: coarse_builder._sha256_bytes(manifest_bytes),
    }


def _build_impl(morphology_root, *, coarse_project_root, coarse_investigation_record,
                coarse_validation_root, project_id, output_root):
    if not isinstance(project_id, str) or coarse_builder._SAFE_PROJECT_ID.fullmatch(project_id) is None:
        raise ValueError("project ID is malformed or unsafe")
    morphology, coarse, record, report_root, output = (
        Path(value).expanduser().absolute() for value in (
            morphology_root, coarse_project_root, coarse_investigation_record,
            coarse_validation_root, output_root,
        )
    )
    if output.exists() or output.is_symlink():
        raise ValueError("output root already exists")
    coarse_builder._reject_symlink_components(output.parent, "output root")
    for path in (morphology, coarse, record, report_root):
        coarse_builder._reject_symlink_components(path, "input")
    for source in (morphology, coarse, record.parent, report_root):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("output root must not be inside an input artifact directory")

    verified = coarse_builder._verify_preparation(morphology)
    for document in (verified.contract, verified.preparation, verified.manifest, *verified.series):
        validation._finite_tree(document)
    validation._verify_interpretation_contract(verified.contract)
    grid = validation._verify_grid_root(coarse, verified)
    investigation = validation._verify_investigation(record, grid, coarse / PROJECT_RELATIVE_PATH)
    source_manifest, report_hashes = _verify_validation_root(report_root, verified, grid, investigation)
    if project_id == grid.project["id"]:
        raise ValueError("new project ID must differ from the coarse project ID")

    identities = {
        "workloadID": workload.WORKLOAD_ID, "datasetSchemaID": workload.DATASET_SCHEMA_ID,
        "payloadSchemaID": workload.PAYLOAD_SCHEMA_ID, "resultSchemaID": workload.RESULT_SCHEMA_ID,
        "executionContractID": workload.EXECUTION_CONTRACT_ID,
        "executionContractVersion": workload.EXECUTION_CONTRACT_VERSION,
        "supportPolicyID": workload.SUPPORT_POLICY_ID,
    }
    provenance = {
        "sourceProjectID": grid.project["id"],
        "sourceInvestigationID": investigation.investigation_id,
        "sourceRunStageID": investigation.run_stage_id,
        "inputHashes": copy.deepcopy(source_manifest["inputHashes"]),
        "coarseValidationArtifacts": report_hashes,
        "parentHashes": copy.deepcopy(verified.preparation["parentHashes"]),
        "parentIDs": copy.deepcopy(verified.preparation["parentIDs"]),
    }
    documents = []
    dataset_records = []
    references = []
    source_ids = {dataset["id"] for dataset in grid.datasets}
    for ordinal, (source, source_record) in enumerate(zip(grid.datasets, grid.build_manifest["datasets"]), 1):
        dataset = copy.deepcopy(source)
        dataset_id = f"{project_id}.{ordinal:03d}"
        if dataset_id in source_ids:
            raise ValueError("new dataset ID collides with an original dataset ID")
        path = source_record["outputFile"]
        dataset.update(identities)
        dataset["id"] = dataset_id
        dataset["supportedMorphologyProvenance"] = {
            "sourceDatasetID": source["id"],
            "sourceDatasetSHA256": grid.artifact_sha256s[path],
            **copy.deepcopy(provenance),
        }
        # Validation only: no payload generation or numerical candidate search.
        workload.PLUGIN.validate_dataset(dataset)
        payload = coarse_builder._stable_json_bytes(dataset)
        documents.append((path, dataset, payload))
        references.append({"id": dataset_id, "path": path})
        dataset_records.append({
            "datasetID": dataset_id, "sourceDatasetID": source["id"],
            "sourceDatasetSHA256": grid.artifact_sha256s[path],
            "outputFile": path, "outputSHA256": coarse_builder._sha256_bytes(payload),
            "modelClassID": source["modelClassID"],
            "candidateCount": source_record["coarseCandidateCount"],
            "candidatesPerWorkUnit": source["candidatesPerWorkUnit"],
            "expectedWorkUnitCount": source_record["expectedWorkUnitCount"],
            "expectedSampleCandidateEvaluationCount": source_record["expectedSampleCandidateEvaluationCount"],
        })
    project = {"id": project_id, "datasets": references, **identities, "provenance": provenance}
    project_bytes = coarse_builder._stable_json_bytes(project)
    manifest = {
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID, "buildManifestVersion": BUILD_MANIFEST_VERSION,
        "algorithmID": ALGORITHM_ID, "algorithmVersion": "1.0",
        "projectID": project_id, "workloadIdentities": identities,
        "supportPolicyID": workload.SUPPORT_POLICY_ID, "provenance": provenance,
        "datasets": dataset_records,
        "orderedDatasetIDs": [item["datasetID"] for item in dataset_records],
        "totalCandidateCount": grid.build_manifest["totalCoarseCandidateCount"],
        "totalExpectedWorkUnitCount": grid.build_manifest["totalExpectedWorkUnitCount"],
        "totalExpectedSampleCandidateEvaluationCount": grid.build_manifest["totalExpectedSampleCandidateEvaluationCount"],
        "relativeArtifactPaths": {
            "project": PROJECT_RELATIVE_PATH, "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
            "datasets": [item[0] for item in documents],
        },
        "outputSHA256s": {
            PROJECT_RELATIVE_PATH: coarse_builder._sha256_bytes(project_bytes),
            **{path: coarse_builder._sha256_bytes(data) for path, _, data in documents},
        },
        "domainPreservationStatement": "Original series, arrays, axes, indexing and shard sizes are unchanged.",
        "noCandidateSearchStatement": "Only the four saved source winners were reproduced during verification.",
        "interpretationStatement": "New results establish eligibility and winners only for the searched grid.",
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    # Hash the manifest body canonically without a self-referential digest.
    manifest["buildManifestSHA256"] = coarse_builder._sha256_bytes(
        coarse_builder._canonical_compact_json_bytes(manifest)
    )
    manifest_bytes = coarse_builder._stable_json_bytes(manifest)
    coarse_builder._assert_identity_isolated((project, manifest, *(item[1] for item in documents)))
    _assert_identity_free((project_bytes, manifest_bytes, *(item[2] for item in documents)))
    output.parent.mkdir(parents=True, exist_ok=True)
    coarse_builder._reject_symlink_components(output.parent, "output root")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for path, _, data in documents:
            _atomic_write_bytes(staging / path, data)
        _atomic_write_bytes(staging / PROJECT_RELATIVE_PATH, project_bytes)
        _atomic_write_bytes(staging / BUILD_MANIFEST_RELATIVE_PATH, manifest_bytes)
        if output.exists() or output.is_symlink():
            raise ValueError("output root already exists")
        staging.rename(output)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {"project": project, "datasets": [item[1] for item in documents], "buildManifest": manifest}


def build_supported_anomaly_morphology_grid(
    morphology_root: str | Path, *, coarse_project_root: str | Path,
    coarse_investigation_record: str | Path, coarse_validation_root: str | Path,
    project_id: str, output_root: str | Path,
) -> dict[str, Any]:
    """Verify source artifacts and atomically publish a separately identified project."""
    try:
        return _build_impl(
            morphology_root, coarse_project_root=coarse_project_root,
            coarse_investigation_record=coarse_investigation_record,
            coarse_validation_root=coarse_validation_root, project_id=project_id, output_root=output_root,
        )
    except SupportedAnomalyMorphologyGridBuildError:
        raise
    except (KeyError, IndexError, OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise SupportedAnomalyMorphologyGridBuildError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("morphology-root", "coarse-project-root", "coarse-investigation-record",
                 "coarse-validation-root", "project-id", "output-root"):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        build_supported_anomaly_morphology_grid(**vars(arguments))
    except SupportedAnomalyMorphologyGridBuildError as error:
        print(f"Supported morphology-grid build failed: {error}")
        return 1
    print(f"Published supported morphology-grid project: {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
