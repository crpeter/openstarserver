"""Build a local ordered-doublet diagnostic after verifying saved PR190 artifacts.

Only saved winners are reproduced by the existing ancestry verification chain.
No candidate grid or shard is searched and no new model preference is computed.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from openstar_workloads.plugins import supported_morphology_grid as workload
from openstar_workloads.plugins.supported_morphology_grid import _adapter as numerical
from workflows.microlensing import build_anomaly_morphology_coarse_grid as coarse
from workflows.microlensing import build_supported_anomaly_morphology_grid as supported_builder
from workflows.microlensing import validate_anomaly_morphology_coarse_grid as legacy
from workflows.microlensing import validate_supported_anomaly_morphology_grid as validation
from workflows.microlensing.coarse_grid import _assert_identity_free, _atomic_write_bytes

BUILD_MANIFEST_SCHEMA_ID = "openstar.microlensing-supported-doublet-refinement-build.v1"
BUILD_MANIFEST_VERSION = "1.0"
ALGORITHM_ID = "openstar.microlensing-supported-doublet-local-refinement.v1"
PROJECT_RELATIVE_PATH = "project.json"
DATASET_RELATIVE_PATH = "datasets/ordered-doublet-refinement.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"
_AXIS_ORDER = (
    "negativeCenter", "separation", "negativeLogScale", "negativeLogShape", "positiveLogScale", "positiveLogShape",
)
_RETENTION_INDICES = (4, 3, 2, 0, 2, 0)
_LIMITS = {
    "searchScope": "LOCAL_ORDERED_DOUBLET_DIAGNOSTIC",
    "purpose": "Diagnose coarse spacing and the lower separation boundary around the accepted ordered-doublet winner.",
    "additionalSearchEffortAllocatedTo": workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    "balancedModelComparisonEstablished": False,
    "globalOptimumEstablished": False, "convergenceEstablished": False,
    "measuredDurationEstablished": False, "crossSeriesReplicationEstablished": False,
    "planetaryInterpretationResolved": False, "discoveryClaim": False,
    "newModelPreferenceComputed": False,
    "statement": (
        "Additional search effort is allocated only to the ordered model. This local diagnostic does not establish "
        "a global optimum or a newly balanced comparison across model classes. Signs, per-series improvement "
        "thresholds and geometric support rules remain unchanged. Positive and independent searches are not rerun."
    ),
}


class SupportedDoubletRefinementBuildError(RuntimeError):
    """Ancestry, compatibility, axis derivation or atomic publication failed."""


def _verify_report(root, expected, input_hashes):
    validation._exact_artifacts(root, (
        validation.RESULT_RELATIVE_PATH, validation.MARKDOWN_RELATIVE_PATH, validation.MANIFEST_RELATIVE_PATH,
    ), "supported validation")
    report_bytes, report = validation._read(root / validation.RESULT_RELATIVE_PATH, "supported validation report")
    manifest_bytes, manifest = validation._read(root / validation.MANIFEST_RELATIVE_PATH, "supported validation manifest")
    markdown_bytes = coarse._read_bytes(root / validation.MARKDOWN_RELATIVE_PATH, "supported validation Markdown")
    validation._equal(report, expected, "supported validation report")
    if markdown_bytes != validation._markdown(expected):
        raise ValueError("supported validation Markdown does not reconstruct exactly")
    expected_manifest = {
        "artifactManifestSchemaID": validation.MANIFEST_SCHEMA_ID, "artifactManifestVersion": validation.REPORT_VERSION,
        "resultSchemaID": validation.REPORT_SCHEMA_ID, "resultVersion": validation.REPORT_VERSION,
        "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "relativeArtifactPaths": {
            "result": validation.RESULT_RELATIVE_PATH, "markdown": validation.MARKDOWN_RELATIVE_PATH,
            "artifactManifest": validation.MANIFEST_RELATIVE_PATH,
        },
        "outputSHA256s": {
            validation.RESULT_RELATIVE_PATH: coarse._sha256_bytes(report_bytes),
            validation.MARKDOWN_RELATIVE_PATH: coarse._sha256_bytes(markdown_bytes),
        },
        "inputHashes": input_hashes, "provenance": expected["provenance"],
        "projectID": expected["projectID"], "investigationID": expected["investigationID"],
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    validation._equal(manifest, expected_manifest, "supported validation manifest")
    coarse._assert_identity_isolated((report, manifest))
    _assert_identity_free((report_bytes, markdown_bytes, manifest_bytes))
    return report, {
        validation.RESULT_RELATIVE_PATH: coarse._sha256_bytes(report_bytes),
        validation.MARKDOWN_RELATIVE_PATH: coarse._sha256_bytes(markdown_bytes),
        validation.MANIFEST_RELATIVE_PATH: coarse._sha256_bytes(manifest_bytes),
    }


def _verify_sources(morphology, coarse_root, coarse_record, coarse_report, supported_root, supported_record, report_root):
    prepared = coarse._verify_preparation(morphology)
    for document in (prepared.contract, prepared.preparation, prepared.manifest, *prepared.series):
        legacy._finite_tree(document)
    legacy._verify_interpretation_contract(prepared.contract)
    grid = legacy._verify_grid_root(coarse_root, prepared)
    original = legacy._verify_investigation(coarse_record, grid, coarse_root / "project.json")
    source_manifest, coarse_report_hashes = supported_builder._verify_validation_root(coarse_report, prepared, grid, original)
    supported = validation._verify_supported_project(supported_root, prepared, grid, original, source_manifest, coarse_report_hashes)
    investigation, run, investigation_hashes = validation._verify_stages(supported_record, supported, supported_root / "project.json")
    input_hashes = {
        **copy.deepcopy(source_manifest["inputHashes"]), "coarseValidationArtifacts": coarse_report_hashes,
        "supportedProjectArtifacts": dict(supported.hashes), **investigation_hashes,
    }
    # Reconstruct the already published report, not a preference for new results.
    expected = validation._report(supported, investigation, run, prepared, input_hashes)
    report, report_hashes = _verify_report(report_root, expected, input_hashes)
    return supported, report, {**input_hashes, "supportedValidationArtifacts": report_hashes}


def _require_parent(dataset, search):
    workload.PLUGIN.validate_dataset(dataset)
    if dataset["modelClassID"] != workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET or search["modelClassID"] != dataset["modelClassID"]:
        raise ValueError("parent must be the canonical ordered-doublet search")
    if search["datasetID"] != dataset["id"]:
        raise ValueError("ordered-doublet parent dataset mapping is inconsistent")
    if search["coverageComplete"] is not True:
        raise ValueError("ordered-doublet parent requires complete coverage")
    winner = search["acceptedWinner"]
    if winner is None:
        raise ValueError("ordered-doublet parent requires a non-null accepted winner")
    if search["supportRequirementMet"] is not True:
        raise ValueError("ordered-doublet parent requires verified observational support")
    # Re-derive axis indices directly from the verified grid, without enumeration.
    axes = legacy._axis_boundaries(dataset, winner["gridIndex"])
    validation._equal(search["axes"], axes, "ordered-doublet parent axes")
    by_name = {axis["axis"]: axis for axis in axes}
    if by_name["separation"]["position"] != "LOWER_BOUNDARY" or not by_name["separation"]["searchedBoundary"]:
        raise ValueError("ordered-doublet separation must be at the lower searched boundary")
    for name in ("negativeCenter", "negativeLogScale", "positiveLogScale"):
        if by_name[name]["position"] != "INTERIOR":
            raise ValueError(f"ordered-doublet {name} must be interior")
    for name in ("negativeLogShape", "positiveLogShape"):
        if not by_name[name]["fixed"]:
            raise ValueError(f"ordered-doublet {name} must be fixed")
    return winner


def _linear_axis(start, step, count, name, *, positive=False):
    start = coarse._finite_number(start, f"{name} start")
    step = coarse._finite_number(step, f"{name} step")
    if step <= 0.0:
        raise ValueError(f"{name} derived step must be strictly positive")
    # At most 41 axis values, not a candidate grid or a fit search. Reject binary64
    # collapse as well as overflow; a positive step alone does not ensure distinct values.
    previous = None
    for index in range(count):
        value = coarse._finite_number(start + index * step, f"{name} derived axis value")
        if positive and value <= 0.0:
            raise ValueError(f"{name} derived values must be strictly positive")
        if previous is not None and value <= previous:
            raise ValueError(f"{name} derived axis values must be strictly increasing")
        previous = value
    return {"start": start, "step": step, "count": count}


def _derive_grid(dataset, winner):
    parent = dataset["morphologyGrid"]
    parameters = winner["parameters"]
    grid, derivation = {}, {}
    for name, count, divisor, half in (("negativeCenter", 9, 8, 4),
                                       ("negativeLogScale", 5, 4, 2), ("positiveLogScale", 5, 4, 2)):
        parent_axis = parent[f"{name}Axis"]
        parent_step = coarse._finite_number(parent_axis["step"], f"parent {name} step")
        if parent_step <= 0.0:
            raise ValueError(f"parent {name} step must be strictly positive")
        step = parent_step / divisor
        accepted = coarse._finite_number(parameters[name], f"accepted {name}")
        grid[f"{name}Axis"] = _linear_axis(accepted - half * step, step, count, name)
        derivation[name] = {
            "parentAxis": copy.deepcopy(parent_axis), "acceptedValue": accepted,
            "stepDivisor": divisor, "startOffsetInNewSteps": -half,
            "retainedParentIndex": half, "newAxis": copy.deepcopy(grid[f"{name}Axis"]),
        }
    separation = coarse._finite_number(parameters["separation"], "accepted separation")
    step = separation / 4.0
    grid["separationAxis"] = _linear_axis(step, step, 41, "separation", positive=True)
    derivation["separation"] = {
        "parentAxis": copy.deepcopy(parent["separationAxis"]), "acceptedValue": separation,
        "stepFormula": "accepted separation / 4", "startFormula": "new step",
        "retainedParentIndex": 3, "newAxis": copy.deepcopy(grid["separationAxis"]),
    }
    for name in ("negativeLogShape", "positiveLogShape"):
        grid[f"{name}Axis"] = copy.deepcopy(parent[f"{name}Axis"])
        derivation[name] = {"preservedExactly": True, "retainedParentIndex": 0,
                            "parentAxis": copy.deepcopy(parent[f"{name}Axis"]), "newAxis": copy.deepcopy(grid[f"{name}Axis"])}
    # Schema validation also checks exponentiated log axes and all safe count products.
    numerical_dataset = copy.deepcopy(dataset)
    numerical_dataset["morphologyGrid"] = grid
    _, validated = numerical.numerical_view(numerical_dataset)
    counts = tuple(axis.count for axis in validated.grid.axes)
    if counts != (9, 41, 5, 1, 5, 1):
        raise ValueError("refinement requires exactly two fixed shape axes and the frozen derived counts")
    retained_index = numerical.candidate_index(_RETENTION_INDICES, counts)
    retained_parameters = {name: axis.value(index) for name, axis, index in zip(_AXIS_ORDER, validated.grid.axes, _RETENTION_INDICES)}
    tolerance = workload.RESULT_RELATIVE_TOLERANCE
    for name in _AXIS_ORDER:
        expected = coarse._finite_number(parameters[name], f"accepted {name}")
        actual = retained_parameters[name]
        if abs(actual - expected) > tolerance * max(1.0, abs(actual), abs(expected)):
            raise ValueError(f"parent winner {name} is not retained within the published tolerance")
    return grid, derivation, {
        "gridIndex": retained_index, "axisIndices": dict(zip(_AXIS_ORDER, _RETENTION_INDICES)),
        "parameters": retained_parameters, "relativeTolerance": tolerance,
        "comparisonLimit": "relativeTolerance * max(1.0, abs(parent), abs(retained))",
        "verifiedWithinTolerance": True, "numericallyEvaluated": False,
    }, validated.grid.total_candidates


def _build_impl(morphology_root, *, coarse_project_root, coarse_investigation_record, coarse_validation_root,
                supported_project_root, supported_investigation_record, supported_validation_root, project_id, output_root):
    if not isinstance(project_id, str) or coarse._SAFE_PROJECT_ID.fullmatch(project_id) is None:
        raise ValueError("project ID is malformed or unsafe")
    paths = [Path(value).expanduser().absolute() for value in (
        morphology_root, coarse_project_root, coarse_investigation_record, coarse_validation_root,
        supported_project_root, supported_investigation_record, supported_validation_root,
    )]
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output root already exists")
    coarse._reject_symlink_components(output.parent, "output root")
    for index, path in enumerate(paths):
        coarse._reject_symlink_components(path, "input")
        source = path.parent if index in (2, 5) else path
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("output root must not be inside an input artifact directory")
    supported, report, input_hashes = _verify_sources(*paths)
    if project_id in (supported.project["id"], supported.manifest["provenance"]["sourceProjectID"]):
        raise ValueError("refinement project ID must differ from its parent project IDs")
    parent, search = supported.datasets[1], report["searches"][1]
    winner = _require_parent(parent, search)
    grid, derivation, inclusion, candidate_count = _derive_grid(parent, winner)
    dataset = copy.deepcopy(parent)
    dataset["id"] = f"{project_id}.ordered-doublet"
    if dataset["id"] in {item["id"] for item in supported.datasets}:
        raise ValueError("refinement dataset ID must differ from its parent dataset IDs")
    dataset["morphologyGrid"] = grid
    provenance = {
        "parentProjectID": supported.project["id"], "parentDatasetID": parent["id"],
        "parentInvestigationID": report["investigationID"],
        "parentValidationSchemaID": validation.REPORT_SCHEMA_ID,
        "parentValidationVersion": validation.REPORT_VERSION,
        "sourceAcceptedWinner": copy.deepcopy(winner), "sourceGridIndex": winner["gridIndex"],
        "inputHashes": input_hashes, "parentLineage": copy.deepcopy(supported.manifest["provenance"]),
        "inheritedContractReferences": "Inherited coarse and supported provenance fields identify the parent artifacts; axisDerivation defines the new search domain.",
    }
    dataset["doubletRefinementProvenance"] = copy.deepcopy(provenance)
    dataset["interpretationLimits"] = copy.deepcopy(_LIMITS)
    workload.PLUGIN.validate_dataset(dataset)
    size = coarse._exact_count(dataset["candidatesPerWorkUnit"], "candidatesPerWorkUnit", positive=True)
    units = (candidate_count + size - 1) // size
    sample_count = coarse._safe_sum([len(series["coordinates"]) for series in dataset["series"]], "sample count")
    evaluations = coarse._safe_product([candidate_count, sample_count], "sample-candidate evaluation budget")
    project = {"id": project_id, "datasets": [{"id": dataset["id"], "path": DATASET_RELATIVE_PATH}],
               **copy.deepcopy(validation._IDENTITIES), "provenance": provenance, "interpretationLimits": copy.deepcopy(_LIMITS)}
    project_bytes, dataset_bytes = coarse._stable_json_bytes(project), coarse._stable_json_bytes(dataset)
    retained_references = [{
        "datasetID": item["datasetID"], "modelClassID": item["modelClassID"],
        "parentProjectID": supported.project["id"], "parentInvestigationID": report["investigationID"],
        "validationReportSHA256": input_hashes["supportedValidationArtifacts"][validation.RESULT_RELATIVE_PATH],
        "validationSearchIndex": index, "acceptedWinner": copy.deepcopy(item["acceptedWinner"]),
        "rerunInThisProject": False,
    } for index, item in enumerate(report["searches"]) if index != 1]
    manifest = {
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID, "buildManifestVersion": BUILD_MANIFEST_VERSION,
        "algorithmID": ALGORITHM_ID, "algorithmVersion": "1.0", "projectID": project_id,
        "datasetID": dataset["id"], "workloadIdentities": copy.deepcopy(validation._IDENTITIES),
        "modelClassID": workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET, "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "provenance": provenance, "axisDerivation": derivation, "parentWinnerInclusion": inclusion,
        "parentBoundaryAssessment": copy.deepcopy(search["axes"]),
        "totalCandidateCount": candidate_count, "candidatesPerWorkUnit": size,
        "totalExpectedWorkUnitCount": units, "finalWorkUnitGridStartIndex": (units - 1) * size,
        "finalWorkUnitGridCount": candidate_count - (units - 1) * size,
        "sampleCount": sample_count, "totalExpectedSampleCandidateEvaluationCount": evaluations,
        "retainedModelResultReferences": retained_references, "interpretationLimits": copy.deepcopy(_LIMITS),
        "sourceVerificationScope": copy.deepcopy(report["verificationScope"]),
        "noCandidateSearchStatement": "Only bounded saved winners in the ancestry chain were reproduced; no refinement candidate was evaluated.",
        "relativeArtifactPaths": {"project": PROJECT_RELATIVE_PATH, "dataset": DATASET_RELATIVE_PATH, "buildManifest": BUILD_MANIFEST_RELATIVE_PATH},
        "outputSHA256s": {PROJECT_RELATIVE_PATH: coarse._sha256_bytes(project_bytes), DATASET_RELATIVE_PATH: coarse._sha256_bytes(dataset_bytes)},
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    manifest["buildManifestSHA256"] = coarse._sha256_bytes(coarse._canonical_compact_json_bytes(manifest))
    manifest_bytes = coarse._stable_json_bytes(manifest)
    coarse._assert_identity_isolated((project, dataset, manifest))
    _assert_identity_free((project_bytes, dataset_bytes, manifest_bytes))
    output.parent.mkdir(parents=True, exist_ok=True)
    coarse._reject_symlink_components(output.parent, "output root")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, data in ((PROJECT_RELATIVE_PATH, project_bytes), (DATASET_RELATIVE_PATH, dataset_bytes), (BUILD_MANIFEST_RELATIVE_PATH, manifest_bytes)):
            _atomic_write_bytes(staging / relative, data)
        if output.exists() or output.is_symlink():
            raise ValueError("output root already exists")
        staging.rename(output)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {"project": project, "dataset": dataset, "buildManifest": manifest}


def build_supported_doublet_refinement(
    morphology_root: str | Path, *, coarse_project_root: str | Path, coarse_investigation_record: str | Path,
    coarse_validation_root: str | Path, supported_project_root: str | Path, supported_investigation_record: str | Path,
    supported_validation_root: str | Path, project_id: str, output_root: str | Path,
) -> dict[str, Any]:
    """Verify compatible ancestry and atomically publish one diagnostic dataset."""
    try:
        return _build_impl(
            morphology_root, coarse_project_root=coarse_project_root, coarse_investigation_record=coarse_investigation_record,
            coarse_validation_root=coarse_validation_root, supported_project_root=supported_project_root,
            supported_investigation_record=supported_investigation_record, supported_validation_root=supported_validation_root,
            project_id=project_id, output_root=output_root,
        )
    except SupportedDoubletRefinementBuildError:
        raise
    except (KeyError, IndexError, OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise SupportedDoubletRefinementBuildError(str(error)) from error


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("morphology-root", "coarse-project-root", "coarse-investigation-record", "coarse-validation-root",
                 "supported-project-root", "supported-investigation-record", "supported-validation-root", "project-id", "output-root"):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = build_supported_doublet_refinement(**vars(arguments))
    except SupportedDoubletRefinementBuildError as error:
        print(f"Supported doublet refinement failed: {error}")
        return 1
    manifest = result["buildManifest"]
    print(f"Published {manifest['totalCandidateCount']} candidates / {manifest['totalExpectedWorkUnitCount']} work units: {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
