"""Offline PR192 artifact and accepted-winner audit with historical diagnostics.

No builders, grid enumeration, shard recomputation or new searches are invoked.
Persisted rejection decisions are checked for accounting consistency only.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from openstar_workloads.plugins import supported_morphology_grid as workload
from workflows.microlensing import build_anomaly_morphology_coarse_grid as coarse
from workflows.microlensing import build_supported_doublet_refinement as builder
from workflows.microlensing import validate_anomaly_morphology_coarse_grid as legacy
from workflows.microlensing import validate_supported_anomaly_morphology_grid as supported
from workflows.microlensing.coarse_grid import _assert_identity_free, _atomic_write_bytes

REPORT_SCHEMA_ID = "openstar.microlensing-supported-doublet-refinement-validation.v1"
MANIFEST_SCHEMA_ID = "openstar.microlensing-supported-doublet-refinement-validation-artifacts.v1"
REPORT_VERSION = "1.0"
RESULT_RELATIVE_PATH = "supported-doublet-refinement-validation.json"
MARKDOWN_RELATIVE_PATH = "supported-doublet-refinement-validation.md"
MANIFEST_RELATIVE_PATH = "artifact-manifest.json"
_COUNT_KEYS = (
    "completedWorkUnits", "totalWorkUnits", "totalCandidateCount", "completedCandidateCount",
    "totalInvalidCandidateCount", "totalSupportRejectedCandidateCount", "totalEligibleCandidateCount",
)
_CLAIMS = {
    "modelPreference": None, "modelPreferenceResolved": False, "balancedModelComparisonEstablished": False,
    "planetaryInterpretationResolved": False, "discoveryClaim": False, "globalOptimumEstablished": False,
    "convergenceEstablished": False, "measuredDurationEstablished": False, "crossSeriesReplicationEstablished": False,
}


class SupportedDoubletRefinementValidationError(RuntimeError):
    """Malformed, inconsistent or unverifiable diagnostic artifacts."""


def _reconstruct(project_id, parent_project, baseline, input_hashes):
    """Reconstruct the exact PR192 format in memory using its pure derivation.

    PR192 has no pure document-construction entry point. Keep this explicit
    versioned reconstruction separate from its publishing builder; every field
    is compared, including retained references and the canonical manifest hash.
    """
    if not isinstance(project_id, str) or coarse._SAFE_PROJECT_ID.fullmatch(project_id) is None:
        raise ValueError("refinement project ID is malformed or unsafe")
    if project_id in (parent_project.project["id"], parent_project.manifest["provenance"]["sourceProjectID"]):
        raise ValueError("refinement project ID must differ from its parent project IDs")
    parent, search = parent_project.datasets[1], baseline["searches"][1]
    winner = builder._require_parent(parent, search)
    grid, derivation, inclusion, count = builder._derive_grid(parent, winner)
    dataset = copy.deepcopy(parent)
    dataset["id"] = f"{project_id}.ordered-doublet"
    if dataset["id"] in {item["id"] for item in parent_project.datasets}:
        raise ValueError("refinement dataset ID must differ from its parent dataset IDs")
    dataset["morphologyGrid"] = grid
    provenance = {
        "parentProjectID": parent_project.project["id"], "parentDatasetID": parent["id"],
        "parentInvestigationID": baseline["investigationID"],
        "parentValidationSchemaID": supported.REPORT_SCHEMA_ID, "parentValidationVersion": supported.REPORT_VERSION,
        "sourceAcceptedWinner": copy.deepcopy(winner), "sourceGridIndex": winner["gridIndex"],
        "inputHashes": copy.deepcopy(input_hashes), "parentLineage": copy.deepcopy(parent_project.manifest["provenance"]),
        "inheritedContractReferences": "Inherited coarse and supported provenance fields identify the parent artifacts; axisDerivation defines the new search domain.",
    }
    dataset["doubletRefinementProvenance"] = copy.deepcopy(provenance)
    dataset["interpretationLimits"] = copy.deepcopy(builder._LIMITS)
    workload.PLUGIN.validate_dataset(dataset)
    size = coarse._exact_count(dataset["candidatesPerWorkUnit"], "candidatesPerWorkUnit", positive=True)
    units = (count + size - 1) // size
    samples = coarse._safe_sum([len(series["coordinates"]) for series in dataset["series"]], "sample count")
    evaluations = coarse._safe_product([count, samples], "sample-candidate evaluation budget")
    project = {
        "id": project_id, "datasets": [{"id": dataset["id"], "path": builder.DATASET_RELATIVE_PATH}],
        **copy.deepcopy(supported._IDENTITIES), "provenance": provenance, "interpretationLimits": copy.deepcopy(builder._LIMITS),
    }
    references = [{
        "datasetID": item["datasetID"], "modelClassID": item["modelClassID"],
        "parentProjectID": parent_project.project["id"], "parentInvestigationID": baseline["investigationID"],
        "validationReportSHA256": input_hashes["supportedValidationArtifacts"][supported.RESULT_RELATIVE_PATH],
        "validationSearchIndex": index, "acceptedWinner": copy.deepcopy(item["acceptedWinner"]), "rerunInThisProject": False,
    } for index, item in enumerate(baseline["searches"]) if index != 1]
    manifest = {
        "buildManifestSchemaID": builder.BUILD_MANIFEST_SCHEMA_ID, "buildManifestVersion": builder.BUILD_MANIFEST_VERSION,
        "algorithmID": builder.ALGORITHM_ID, "algorithmVersion": "1.0", "projectID": project_id,
        "datasetID": dataset["id"], "workloadIdentities": copy.deepcopy(supported._IDENTITIES),
        "modelClassID": workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET, "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "provenance": provenance, "axisDerivation": derivation, "parentWinnerInclusion": inclusion,
        "parentBoundaryAssessment": copy.deepcopy(search["axes"]), "totalCandidateCount": count, "candidatesPerWorkUnit": size,
        "totalExpectedWorkUnitCount": units, "finalWorkUnitGridStartIndex": (units - 1) * size,
        "finalWorkUnitGridCount": count - (units - 1) * size, "sampleCount": samples,
        "totalExpectedSampleCandidateEvaluationCount": evaluations, "retainedModelResultReferences": references,
        "interpretationLimits": copy.deepcopy(builder._LIMITS), "sourceVerificationScope": copy.deepcopy(baseline["verificationScope"]),
        "noCandidateSearchStatement": "Only bounded saved winners in the ancestry chain were reproduced; no refinement candidate was evaluated.",
        "relativeArtifactPaths": {"project": builder.PROJECT_RELATIVE_PATH, "dataset": builder.DATASET_RELATIVE_PATH,
                                  "buildManifest": builder.BUILD_MANIFEST_RELATIVE_PATH},
        "outputSHA256s": {builder.PROJECT_RELATIVE_PATH: supported._hash(supported._json(project)),
                          builder.DATASET_RELATIVE_PATH: supported._hash(supported._json(dataset))},
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    manifest["buildManifestSHA256"] = supported._hash(coarse._canonical_compact_json_bytes(manifest))
    return project, dataset, manifest


def _verify_project(root, parent, baseline, input_hashes):
    supported._exact_artifacts(root, (builder.PROJECT_RELATIVE_PATH, builder.BUILD_MANIFEST_RELATIVE_PATH, "datasets"), "refinement project")
    supported._exact_artifacts(root / "datasets", (Path(builder.DATASET_RELATIVE_PATH).name,), "refinement datasets")
    project_bytes, project = supported._read(root / builder.PROJECT_RELATIVE_PATH, "refinement project")
    dataset_bytes, dataset = supported._read(root / builder.DATASET_RELATIVE_PATH, "refinement dataset")
    manifest_bytes, manifest = supported._read(root / builder.BUILD_MANIFEST_RELATIVE_PATH, "refinement build manifest")
    # Check actual supported routing before any internal numerical view.
    supported.numerical.check_identities(project)
    workload.PLUGIN.validate_dataset(dataset)
    expected = _reconstruct(project.get("id"), parent, baseline, input_hashes)
    for actual, canonical, label in zip((project, dataset, manifest), expected,
                                      ("refinement project", "refinement dataset domain and provenance", "refinement build manifest")):
        supported._equal(actual, canonical, label)
    coarse._assert_identity_isolated((project, dataset, manifest))
    _assert_identity_free((project_bytes, dataset_bytes, manifest_bytes))
    hashes = dict(zip((builder.PROJECT_RELATIVE_PATH, builder.DATASET_RELATIVE_PATH, builder.BUILD_MANIFEST_RELATIVE_PATH),
                      map(supported._hash, (project_bytes, dataset_bytes, manifest_bytes))))
    # This is the actual single-dataset supported-workload project, with no ID
    # substitution or persisted-record translation. The stage helper is count agnostic.
    return supported._SupportedProject(project, manifest, (dataset,), hashes)


def _verify_search(project, run, contract):
    statuses = run.get("datasets")
    if not isinstance(statuses, list) or len(statuses) != 1 or not isinstance(statuses[0], Mapping):
        raise ValueError("refinement run requires exactly one canonical dataset summary")
    dataset, status, manifest = project.datasets[0], statuses[0], project.manifest
    supported._equal(status.get("id"), dataset["id"], "refinement dataset mapping")
    search = supported._search(status, dataset, {
        "candidateCount": manifest["totalCandidateCount"], "expectedWorkUnitCount": manifest["totalExpectedWorkUnitCount"],
    }, contract)
    supported._verify_run_counter_scopes(run, status, expected_project_work_units=manifest["totalExpectedWorkUnitCount"])
    if "projectProgress" in run and (isinstance(run["projectProgress"], bool) or coarse._finite_number(run["projectProgress"], "project progress") != 1.0):
        raise ValueError("refinement project progress is incomplete")
    supported._contributions(run.get("nodeContributions"), manifest["totalExpectedWorkUnitCount"], "refinement project contributions")
    supported._equal(run["nodeContributions"], status["nodeContributions"], "refinement dataset/project contributions")
    # The producer currently duplicates work counters, not candidate counters,
    # at run scope. Check candidate duplicates as well whenever present.
    for key in _COUNT_KEYS:
        if key in run:
            supported._equal(coarse._exact_count(run[key], key), search[key], f"refinement run {key}")
    return search


def _change(previous, refined):
    left = previous["weightedResidualSumSquares"] if previous else None
    right = refined["weightedResidualSumSquares"] if refined else None
    delta = coarse._finite_number(left - right, "WRSS change") if left is not None and right is not None else None
    tolerance = workload.RESULT_RELATIVE_TOLERANCE * max(1.0, abs(left), abs(right)) if delta is not None else None
    return {
        "previousWRSS": left, "refinedWRSS": right, "deltaWRSS": delta,
        "notWorseWithinNumericalTolerance": supported._gate(
            {"previousWRSS": left, "refinedWRSS": right, "delta": delta},
            delta >= -tolerance if delta is not None else None, threshold=-tolerance if tolerance is not None else None, operator=">=",
        ),
        "improvedBeyondNumericalTolerance": supported._gate(
            {"previousWRSS": left, "refinedWRSS": right, "delta": delta},
            delta > tolerance if delta is not None else None, threshold=tolerance, operator=">",
        ),
    }


def _unevaluated(value):
    if isinstance(value, dict):
        result = {key: _unevaluated(item) for key, item in value.items()}
        if "evaluated" in result and "passed" in result:
            result.update(evaluated=False, passed=False)
        return result
    if isinstance(value, list):
        return [_unevaluated(item) for item in value]
    return value


def _comparisons(search, baseline, contract):
    historical = copy.deepcopy(baseline["searches"])
    historical[1] = copy.deepcopy(search)
    independent_winners = [item["acceptedWinner"] for item in historical[2:]]
    independent = legacy._independent_aggregate(independent_winners, contract) if all(w is not None for w in independent_winners) else None
    supported._equal(independent, baseline["independentAggregate"], "preserved joint independent statistics")
    raw = supported._model_comparisons(historical, independent, contract)
    # Timing dispersion can be computed from historical independent results alone;
    # that does not make any comparison with a null refinement winner evaluated.
    if search["acceptedWinner"] is None:
        raw = _unevaluated(raw)
    for name in ("preferOrderedDoubletOverPositivePulse", "rejectOrderedDoubletForIndependentPulses"):
        comparison = raw[name]
        comparison["gates"] = {key: gate["passed"] for key, gate in comparison["gateDetails"].items()}
        comparison["evaluated"] = all(gate["evaluated"] for gate in comparison["gateDetails"].values())
        comparison["passed"] = comparison["evaluated"] and all(comparison["gates"].values())
        comparison["comparisonScope"] = "HISTORICAL_BASELINE_DIAGNOSTIC"
        comparison["balancedModelComparisonEstablished"] = False
        comparison["establishesModelPreference"] = False
    previous, refined = baseline["searches"][1]["acceptedWinner"], search["acceptedWinner"]
    previous_comparison = {
        "comparisonScope": "SAME_MODEL_NUMERICAL_DIAGNOSTIC", "evaluated": refined is not None,
        "relativeTolerance": workload.RESULT_RELATIVE_TOLERANCE,
        "toleranceFormula": "relativeTolerance * max(1.0, abs(previousWRSS), abs(refinedWRSS))",
        "thresholdScope": "Published numerical comparison tolerance only; no new scientific acceptance threshold.",
        "global": _change(previous, refined),
        "perSeries": [{"genericSeriesID": series_id, **_change(previous["seriesFits"][index], refined["seriesFits"][index] if refined else None)}
                      for index, series_id in enumerate(contract["admittedGenericSeriesIDs"])],
    }
    return previous_comparison, raw, independent


def _outcome(search, historical):
    rules = ("preferOrderedDoubletOverPositivePulse", "rejectOrderedDoubletForIndependentPulses")
    failed = [f"{name}.{key}" for name in rules
              for key, gate in historical[name]["gateDetails"].items() if gate["evaluated"] and not gate["passed"]]
    # These are alternative model-comparison conditions, not two requirements
    # that must both pass. Preserve all unmet gates without treating an unmet
    # rejection condition as a problem when another complete condition is met.
    comparison_condition_met = any(historical[name]["passed"] for name in rules)
    boundaries = search["searchedBoundaryAxes"]
    if search["acceptedWinner"] is None:
        classification, next_test = "UNRESOLVED_NO_ELIGIBLE_CANDIDATES", "REVIEW_SUPPORT_AND_ACCOUNTING_BEFORE_FURTHER_SEARCH"
    elif not all(historical[name]["evaluated"] for name in rules):
        classification, next_test = "UNRESOLVED_MISSING_HISTORICAL_WINNER", "REVIEW_MISSING_HISTORICAL_BASELINE_BEFORE_COMPARISON"
    elif boundaries:
        classification = "UNRESOLVED_SEARCHED_BOUNDARY"
        next_test = "PREDECLARE_BOUNDARY_AND_BALANCED_MODEL_FOLLOWUP" if comparison_condition_met else "REVIEW_FAILED_HISTORICAL_GATES_AND_SEARCHED_BOUNDARIES"
    elif not comparison_condition_met:
        classification, next_test = "DIAGNOSTIC_GATES_NOT_PASSED", "REVIEW_FAILED_HISTORICAL_GATES_BEFORE_BALANCED_FOLLOWUP"
    else:
        classification, next_test = "HISTORICAL_BASELINE_DIAGNOSTIC_COMPLETE", "PREDECLARE_BALANCED_MODEL_AND_STABILITY_CHECK"
    return classification, next_test, failed


def _report(project, investigation, run, search, baseline, contract, input_hashes):
    previous, historical, independent = _comparisons(search, baseline, contract)
    classification, next_test, failed = _outcome(search, historical)
    boundary_statement = (
        "No accepted winner: convergence remains unresolved." if search["acceptedWinner"] is None else
        "A searched boundary is present; convergence remains unresolved." if search["searchedBoundaryAxes"] else
        "The winner is interior on searched axes; this alone does not prove convergence. Convergence remains unresolved."
    )
    return {
        "resultSchemaID": REPORT_SCHEMA_ID, "resultVersion": REPORT_VERSION,
        "projectID": project.project["id"], "investigationID": investigation["id"],
        "workloadID": workload.WORKLOAD_ID, "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "coverageComplete": True, "projectCompletedWorkUnits": project.manifest["totalExpectedWorkUnitCount"],
        "projectTotalWorkUnits": project.manifest["totalExpectedWorkUnitCount"],
        "accountingTotals": {key: search[key] for key in _COUNT_KEYS}, "nodeContributions": copy.deepcopy(run["nodeContributions"]),
        "search": search, "previousOrderedComparison": previous, "historicalBaselineComparisons": historical,
        "independentAggregate": independent, "preservedPR190Report": copy.deepcopy(baseline),
        "comparisonScope": "HISTORICAL_BASELINE_DIAGNOSTIC", "additionalSearchEffortAllocatedTo": workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET,
        "overallClassification": classification, "recommendedNextTest": next_test, "failedHistoricalGatePaths": failed,
        "boundaryStatement": boundary_statement, "convergenceStatus": "UNRESOLVED", **copy.deepcopy(_CLAIMS),
        "supportRule": copy.deepcopy(baseline["supportRule"]),
        "frozenComparisonRules": copy.deepcopy(contract["decisionRules"]),
        "frozenCrossSeriesRequirements": copy.deepcopy(contract["crossSeriesRequirements"]),
        "inputHashes": input_hashes, "provenance": copy.deepcopy(project.manifest["provenance"]),
        "refinementBuildManifest": copy.deepcopy(project.manifest),
        "verificationScope": {
            "sourceArtifactsAndLedgersVerified": True, "refinementArtifactsReconstructed": True,
            "counterConsistencyVerified": True, "acceptedOriginalCoarseCandidatesReproduced": len(baseline["searches"]),
            "acceptedPreviousSupportedCandidatesReproduced": sum(item["acceptedWinner"] is not None for item in baseline["searches"]),
            "acceptedRefinementCandidatesReproduced": int(search["acceptedWinner"] is not None),
            "acceptedWinnerSupportIndependentlyVerified": search["acceptedWinner"] is not None,
            "fullGridEnumerated": False, "shardsRecomputed": False, "rejectionCountsIndependentlyReproduced": False,
            "globalOptimalityIndependentlyProven": False,
        },
        "limitations": [
            "Accepted winners were reproduced and rejection counts were checked for consistency; rejection decisions across the entire grid were not independently recomputed.",
            "Only saved winners in the verified ancestry and the accepted refinement winner are evaluated; no grid or shard search is performed.",
            "Additional search effort was allocated only to the doublet. Comparisons with unrefined alternatives are historical-baseline diagnostics, not a newly balanced model comparison.",
            "Raw frozen gate outcomes are retained separately; passing them does not establish a formal model preference in this diagnostic report.",
            boundary_statement,
            "Geometric eligibility, including zero-amplitude components, is not measured duration, convergence, cross-series replication or planetary evidence.",
            "A complete null-winner grid remains unresolved and does not exclude supported solutions outside the searched domain.",
        ],
    }


def _markdown(report):
    search, totals = report["search"], report["accountingTotals"]
    lines = ["# Supported doublet refinement validation", "", f"Classification: **{report['overallClassification']}**.",
             f"Work: {report['projectCompletedWorkUnits']}/{report['projectTotalWorkUnits']}; candidates: "
             f"{totals['completedCandidateCount']} evaluated, {totals['totalInvalidCandidateCount']} invalid, "
             f"{totals['totalSupportRejectedCandidateCount']} support-rejected, {totals['totalEligibleCandidateCount']} eligible.", "",
             f"Accepted grid index: {search['acceptedWinner']['gridIndex'] if search['acceptedWinner'] else 'null'}.",
             report["boundaryStatement"], f"Searched boundaries: {', '.join(search['searchedBoundaryAxes']) or 'none'}; "
             f"fixed axes: {', '.join(search['fixedAxes']) or 'none'}.", "",
             "WRSS change is previous ordered WRSS minus refined WRSS (positive means improvement):", "",
             "| Scope | Previous | Refined | Change |", "| --- | ---: | ---: | ---: |"]
    previous = report["previousOrderedComparison"]
    for label, row in [("Combined", previous["global"]), *((item["genericSeriesID"], item) for item in previous["perSeries"])]:
        lines.append(f"| {label} | {row['previousWRSS']} | {row['refinedWRSS']} | {row['deltaWRSS']} |")
    lines.extend(["", "Historical-baseline gates (additional effort went only to the doublet):", "",
                  "| Comparison | Gate | Outcome |", "| --- | --- | --- |"])
    for name, label in (("preferOrderedDoubletOverPositivePulse", "Doublet vs positive"),
                        ("rejectOrderedDoubletForIndependentPulses", "Independent vs doublet")):
        for gate_name, gate in report["historicalBaselineComparisons"][name]["gateDetails"].items():
            state = "unevaluated" if not gate["evaluated"] else "pass" if gate["passed"] else "fail"
            lines.append(f"| {label} | {gate_name} | {state} |")
    lines.extend(["", "Every gate's inputs and thresholds, component support counts/distances, amplitude signs and axis bounds are in the JSON report.",
                  "Model preference remains null; balanced comparison, global optimum, convergence, measured duration, planetary interpretation and discovery claims remain false.",
                  "Accepted winners were reproduced. Rejection counts were checked for consistency; individual grid rejection decisions were not independently recomputed.",
                  "", f"Recommended next test: `{report['recommendedNextTest']}`. No follow-up project was built or launched.", ""])
    return "\n".join(lines).encode("utf-8")


def _validate_impl(morphology_root, *, coarse_project_root, coarse_investigation_record, coarse_validation_root,
                   supported_project_root, supported_investigation_record, supported_validation_root,
                   refinement_project_root, refinement_investigation_record, output_root):
    paths = [Path(value).expanduser().absolute() for value in (
        morphology_root, coarse_project_root, coarse_investigation_record, coarse_validation_root,
        supported_project_root, supported_investigation_record, supported_validation_root,
        refinement_project_root, refinement_investigation_record,
    )]
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output root already exists")
    coarse._reject_symlink_components(output.parent, "output root")
    for index, path in enumerate(paths):
        coarse._reject_symlink_components(path, "input")
        source = path.parent if index in (2, 5, 8) else path
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("output root must not be inside an input artifact directory")
    parent, baseline, source_hashes = builder._verify_sources(*paths[:7])
    # Read the already verified preparation contract, checking its file digest
    # again to bind the diagnostic gates to the same frozen source artifact.
    contract_bytes, contract = supported._read(paths[0] / coarse.PREPARATION_CONTRACT_RELATIVE_PATH, "morphology contract")
    supported._equal(supported._hash(contract_bytes), source_hashes["morphologyContractFile"], "diagnostic morphology contract hash")
    project = _verify_project(paths[7], parent, baseline, source_hashes)
    investigation, run, stage_hashes = supported._verify_stages(paths[8], project, paths[7] / builder.PROJECT_RELATIVE_PATH)
    search = _verify_search(project, run, contract)
    input_hashes = {
        **source_hashes, "refinementProjectArtifacts": dict(project.hashes),
        "refinementInvestigationRecord": stage_hashes["supportedInvestigationRecord"],
        "refinementStageLedgers": stage_hashes["supportedStageLedgers"],
    }
    report = _report(project, investigation, run, search, baseline, contract, input_hashes)
    legacy._finite_tree(report)
    report_bytes, markdown_bytes = supported._json(report), _markdown(report)
    manifest = {
        "artifactManifestSchemaID": MANIFEST_SCHEMA_ID, "artifactManifestVersion": REPORT_VERSION,
        "resultSchemaID": REPORT_SCHEMA_ID, "resultVersion": REPORT_VERSION, "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "relativeArtifactPaths": {"result": RESULT_RELATIVE_PATH, "markdown": MARKDOWN_RELATIVE_PATH, "artifactManifest": MANIFEST_RELATIVE_PATH},
        "outputSHA256s": {RESULT_RELATIVE_PATH: supported._hash(report_bytes), MARKDOWN_RELATIVE_PATH: supported._hash(markdown_bytes)},
        "inputHashes": input_hashes, "provenance": report["provenance"], "projectID": project.project["id"],
        "investigationID": investigation["id"], **copy.deepcopy(_CLAIMS),
    }
    manifest_bytes = supported._json(manifest)
    coarse._assert_identity_isolated((report, manifest))
    _assert_identity_free((report_bytes, markdown_bytes, manifest_bytes))
    output.parent.mkdir(parents=True, exist_ok=True)
    coarse._reject_symlink_components(output.parent, "output root")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, data in ((RESULT_RELATIVE_PATH, report_bytes), (MARKDOWN_RELATIVE_PATH, markdown_bytes), (MANIFEST_RELATIVE_PATH, manifest_bytes)):
            _atomic_write_bytes(staging / relative, data)
        if output.exists() or output.is_symlink():
            raise ValueError("output root already exists")
        staging.rename(output)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {"result": report, "artifactManifest": manifest}


def validate_supported_doublet_refinement(
    morphology_root: str | Path, *, coarse_project_root: str | Path, coarse_investigation_record: str | Path,
    coarse_validation_root: str | Path, supported_project_root: str | Path, supported_investigation_record: str | Path,
    supported_validation_root: str | Path, refinement_project_root: str | Path,
    refinement_investigation_record: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    """Verify immutable artifacts and atomically publish the diagnostic report."""
    try:
        return _validate_impl(
            morphology_root, coarse_project_root=coarse_project_root, coarse_investigation_record=coarse_investigation_record,
            coarse_validation_root=coarse_validation_root, supported_project_root=supported_project_root,
            supported_investigation_record=supported_investigation_record, supported_validation_root=supported_validation_root,
            refinement_project_root=refinement_project_root, refinement_investigation_record=refinement_investigation_record,
            output_root=output_root,
        )
    except SupportedDoubletRefinementValidationError:
        raise
    except (KeyError, IndexError, OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise SupportedDoubletRefinementValidationError(str(error)) from error


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("morphology-root", "coarse-project-root", "coarse-investigation-record", "coarse-validation-root",
                 "supported-project-root", "supported-investigation-record", "supported-validation-root", "refinement-project-root",
                 "refinement-investigation-record", "output-root"):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = validate_supported_doublet_refinement(**vars(arguments))
    except SupportedDoubletRefinementValidationError as error:
        print(f"Supported doublet refinement validation failed: {error}")
        return 1
    print(f"{result['result']['overallClassification']}: {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
