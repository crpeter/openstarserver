"""Offline verification of saved supported-grid winners, accounting and interpretation.

No grid enumeration, shard recomputation, coordinator access or input mutation.
Persisted rejection counts are checked for consistency, not independently reproduced.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from openstar_workloads.plugins import supported_morphology_grid as workload
from openstar_workloads.plugins.supported_morphology_grid import _adapter as numerical
from workflows.microlensing import build_anomaly_morphology_coarse_grid as coarse
from workflows.microlensing import build_supported_anomaly_morphology_grid as builder
from workflows.microlensing import validate_anomaly_morphology_coarse_grid as legacy
from workflows.microlensing.coarse_grid import _assert_identity_free, _atomic_write_bytes
from workflows.microlensing.refine_grid import (
    PREPARE_HANDLER_ID, PROJECT_RUN_HANDLER_ID, SMOKE_WORKFLOW_ID,
    TERMINAL_CHECK_HANDLER_ID, _INVESTIGATION_FIELDS, _stage_ledgers, _validate_stage_shape,
)
from workflows.microlensing.validate_residual_grid import _verify_run_counter_scopes

REPORT_SCHEMA_ID = "openstar.microlensing-supported-anomaly-morphology-validation.v1"
MANIFEST_SCHEMA_ID = "openstar.microlensing-supported-anomaly-morphology-validation-artifacts.v1"
REPORT_VERSION = "1.0"
RESULT_RELATIVE_PATH = "supported-anomaly-morphology-validation.json"
MARKDOWN_RELATIVE_PATH = "supported-anomaly-morphology-validation.md"
MANIFEST_RELATIVE_PATH = "artifact-manifest.json"
_IDENTITIES = {
    "workloadID": workload.WORKLOAD_ID, "datasetSchemaID": workload.DATASET_SCHEMA_ID,
    "payloadSchemaID": workload.PAYLOAD_SCHEMA_ID, "resultSchemaID": workload.RESULT_SCHEMA_ID,
    "executionContractID": workload.EXECUTION_CONTRACT_ID,
    "executionContractVersion": workload.EXECUTION_CONTRACT_VERSION,
    "supportPolicyID": workload.SUPPORT_POLICY_ID,
}
_BEST_KEYS = (
    "gridIndex", "parameters", "seriesFits", "positiveWeightSampleCount",
    "weightedResidualSumSquares", "nominalParameterCount", "bayesianInformationCriterion",
    "correctedAkaikeInformationCriterion", "correctedAkaikeInformationCriterionDefined",
)
_equal = legacy._require_equal
_read = legacy._read_json_file
_hash = coarse._sha256_bytes
_json = coarse._stable_json_bytes
_count = coarse._exact_count


class SupportedAnomalyMorphologyGridValidationError(RuntimeError):
    """Malformed, inconsistent or unverifiable supported-grid artifacts."""


@dataclass(frozen=True)
class _SupportedProject:
    project: Mapping[str, Any]
    manifest: Mapping[str, Any]
    datasets: tuple[Mapping[str, Any], ...]
    hashes: Mapping[str, str]


def _exact_artifacts(root, names, label):
    root = coarse._regular_directory(root, label)
    if {entry.name for entry in root.iterdir()} != set(names):
        raise ValueError(f"{label} artifact set is incomplete or unexpected")
    return root


def _verify_supported_project(root, prepared, grid, investigation, source_manifest, report_hashes):
    _exact_artifacts(root, ("project.json", "build-manifest.json", "datasets"), "supported project")
    project_bytes, project = _read(root / "project.json", "supported project")
    manifest_bytes, manifest = _read(root / "build-manifest.json", "supported build manifest")
    project_id = coarse._nonempty_string(project.get("id"), "supported project ID")
    if coarse._SAFE_PROJECT_ID.fullmatch(project_id) is None or project_id == grid.project["id"]:
        raise ValueError("supported project ID is unsafe or reuses the coarse ID")
    provenance = {
        "sourceProjectID": grid.project["id"], "sourceInvestigationID": investigation.investigation_id,
        "sourceRunStageID": investigation.run_stage_id,
        "inputHashes": copy.deepcopy(source_manifest["inputHashes"]),
        "coarseValidationArtifacts": dict(report_hashes),
        "parentHashes": copy.deepcopy(prepared.preparation["parentHashes"]),
        "parentIDs": copy.deepcopy(prepared.preparation["parentIDs"]),
    }
    references, records, datasets = [], [], []
    hashes = {"project.json": _hash(project_bytes), "build-manifest.json": _hash(manifest_bytes)}
    _exact_artifacts(root / "datasets", [Path(r["outputFile"]).name for r in grid.build_manifest["datasets"]], "supported datasets")
    source_ids = {item["id"] for item in grid.datasets}
    for ordinal, (source, record) in enumerate(zip(grid.datasets, grid.build_manifest["datasets"]), 1):
        dataset_id = f"{project_id}.{ordinal:03d}"
        if dataset_id in source_ids:
            raise ValueError("supported dataset ID reuses a coarse ID")
        relative = record["outputFile"]
        payload, dataset = _read(root / relative, "supported dataset")
        # Validate the actual public identities before any internal numerical view.
        workload.PLUGIN.validate_dataset(dataset)
        expected = copy.deepcopy(source)
        expected.update(_IDENTITIES)
        expected["id"] = dataset_id
        expected["supportedMorphologyProvenance"] = {
            "sourceDatasetID": source["id"], "sourceDatasetSHA256": grid.artifact_sha256s[relative],
            **copy.deepcopy(provenance),
        }
        _equal(dataset, expected, "supported dataset domain and provenance")
        hashes[relative] = _hash(payload)
        datasets.append(dataset)
        references.append({"id": dataset_id, "path": relative})
        records.append({
            "datasetID": dataset_id, "sourceDatasetID": source["id"],
            "sourceDatasetSHA256": grid.artifact_sha256s[relative],
            "outputFile": relative, "outputSHA256": _hash(payload),
            "modelClassID": source["modelClassID"], "candidateCount": record["coarseCandidateCount"],
            "candidatesPerWorkUnit": source["candidatesPerWorkUnit"],
            "expectedWorkUnitCount": record["expectedWorkUnitCount"],
            "expectedSampleCandidateEvaluationCount": record["expectedSampleCandidateEvaluationCount"],
        })
    _equal(project, {"id": project_id, "datasets": references, **_IDENTITIES, "provenance": provenance}, "supported project")
    expected_manifest = {
        "buildManifestSchemaID": builder.BUILD_MANIFEST_SCHEMA_ID,
        "buildManifestVersion": builder.BUILD_MANIFEST_VERSION,
        "algorithmID": builder.ALGORITHM_ID, "algorithmVersion": "1.0",
        "projectID": project_id, "workloadIdentities": dict(_IDENTITIES),
        "supportPolicyID": workload.SUPPORT_POLICY_ID, "provenance": provenance,
        "datasets": records, "orderedDatasetIDs": [r["datasetID"] for r in records],
        "totalCandidateCount": grid.build_manifest["totalCoarseCandidateCount"],
        "totalExpectedWorkUnitCount": grid.build_manifest["totalExpectedWorkUnitCount"],
        "totalExpectedSampleCandidateEvaluationCount": grid.build_manifest["totalExpectedSampleCandidateEvaluationCount"],
        "relativeArtifactPaths": {
            "project": "project.json", "buildManifest": "build-manifest.json",
            "datasets": [r["outputFile"] for r in records],
        },
        "outputSHA256s": {path: digest for path, digest in hashes.items() if path != "build-manifest.json"},
        "domainPreservationStatement": "Original series, arrays, axes, indexing and shard sizes are unchanged.",
        "noCandidateSearchStatement": "Only the four saved source winners were reproduced during verification.",
        "interpretationStatement": "New results establish eligibility and winners only for the searched grid.",
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    expected_manifest["buildManifestSHA256"] = _hash(coarse._canonical_compact_json_bytes(expected_manifest))
    _equal(manifest, expected_manifest, "supported build manifest")
    coarse._assert_identity_isolated((project, manifest, *datasets))
    _assert_identity_free((project_bytes, manifest_bytes, *(_json(d) for d in datasets)))
    return _SupportedProject(project, manifest, tuple(datasets), hashes)


def _verify_stages(path, supported, project_path):
    if path.name != "investigation.json":
        raise ValueError("supported investigation record must be named investigation.json")
    directory = _exact_artifacts(path.parent, ("investigation.json", "stages"), "supported investigation")
    data, record = _read(path, "supported investigation record")
    if set(record) != _INVESTIGATION_FIELDS:
        raise ValueError("supported investigation field set is invalid")
    identity = coarse._nonempty_string(record.get("id"), "supported investigation ID")
    if coarse._SAFE_PROJECT_ID.fullmatch(identity) is None or directory.name != identity:
        raise ValueError("supported investigation ID is unsafe or mismatches its directory")
    _equal(record["workflow_id"], SMOKE_WORKFLOW_ID, "supported investigation workflow")
    _equal(record["workflow_version"], "20.0", "supported investigation workflow version")
    _equal(record["status"], "COMPLETE", "supported investigation status")
    for key in ("created_at", "updated_at"):
        coarse._nonempty_string(record[key], key)
    metadata = record["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {"coordinator", "projectPath"}:
        raise ValueError("supported investigation metadata is invalid")
    coarse._nonempty_string(metadata["coordinator"], "coordinator")
    metadata_path = Path(coarse._nonempty_string(metadata["projectPath"], "metadata project path")).expanduser()
    coarse._reject_symlink_components(metadata_path.absolute(), "metadata project path")
    if metadata_path.resolve() != project_path.resolve():
        raise ValueError("supported investigation refers to a different project")
    stages = record["stages"]
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("supported investigation requires exactly three stages")
    ids = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("supported stage must be a mapping")
        stage_id, _ = _validate_stage_shape(stage)
        if stage_id in ids or stage["artifacts"] != []:
            raise ValueError("supported stage IDs or artifacts are invalid")
        ids.append(stage_id)
        _read(directory / "stages" / f"{stage_id}.json", "supported stage ledger")
    ledgers = _stage_ledgers(path, stages)
    _assert_identity_free((data,))
    coarse._assert_identity_isolated(record)
    handlers = (PREPARE_HANDLER_ID, PROJECT_RUN_HANDLER_ID, TERMINAL_CHECK_HANDLER_ID)
    _equal([stage["handler_id"] for stage in stages], list(handlers), "supported stage order")
    project_id = supported.project["id"]
    project_hash = supported.hashes["project.json"]
    project_name = str(project_path.resolve())
    parameters = (
        {"projectPath": project_name},
        {"projectPath": project_name, "projectManifestSha256": project_hash},
        {"expectedProjectID": project_id},
    )
    for index, stage in enumerate(stages):
        _equal(stage["parameters"], parameters[index], "supported stage parameters")
        _equal(stage["triggered_by_stage_id"], ids[index - 1] if index else None, "supported stage causality")
        next_stage = {
            "handler_id": handlers[index + 1], "id": ids[index + 1],
            "parameters": parameters[index + 1], "triggered_by_stage_id": ids[index],
        } if index < 2 else None
        _equal(stage["next_stage"], next_stage, "supported stage continuation")
        _equal(stage["stop"], index == 2, "supported stage stop flag")
        provenance = stage["provenance"]
        _equal(provenance["input_hashes"], {"projectManifest": project_hash} if index < 2 else {}, "supported stage input hashes")
        _equal(provenance["project_ids"], [project_id] if index else [], "supported stage project IDs")
        if index != 1:
            _equal(provenance["node_contributions"], {}, "supported non-run contributions")
    _equal(stages[0]["result"], parameters[1], "supported prepare result")
    total = supported.manifest["totalExpectedWorkUnitCount"]
    _equal(stages[2]["result"], {
        "completedWorkUnits": total, "failedWorkUnits": 0, "passed": True,
        "projectID": project_id, "totalWorkUnits": total,
        "rule": "projectID matches and completed+failed == total",
    }, "supported terminal result")
    run = stages[1]["result"]
    for key, expected in {"projectID": project_id, "projectPath": project_name, "status": "COMPLETE",
                          "workloadID": workload.WORKLOAD_ID}.items():
        _equal(run.get(key), expected, f"supported run {key}")
    numerical.check_identities(run)
    _equal(run.get("nodeContributions"), stages[1]["provenance"]["node_contributions"], "supported run contributions")
    return record, run, {"supportedInvestigationRecord": _hash(data), "supportedStageLedgers": ledgers}


def _contributions(value, expected, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    for key in value:
        coarse._nonempty_string(key, "node ID")
    if coarse._safe_sum(list(value.values()), label) != expected:
        raise ValueError(f"{label} does not sum to completed work")
    return value


def _search(status, dataset, record, contract):
    if not isinstance(status, Mapping):
        raise ValueError("supported dataset summary must be a mapping")
    expected = {
        **{key: _IDENTITIES[key] for key in ("workloadID", "datasetSchemaID", "payloadSchemaID", "resultSchemaID", "supportPolicyID")},
        "id": dataset["id"], "modelClassID": dataset["modelClassID"],
        "morphologyFamilyID": workload.MORPHOLOGY_FAMILY_ID,
        "componentTemplateFamilyID": workload.COMPONENT_TEMPLATE_FAMILY_ID,
        "workloadStatus": "SUPPORTED_MORPHOLOGY_GRID_COMPLETE",
        "supportedMorphologyGridStatus": "SUPPORTED_MORPHOLOGY_GRID_COMPLETE", "coverageComplete": True,
        "assignedWorkUnits": 0, "pendingWorkUnits": 0, "failedWorkUnits": 0,
        "completedWorkUnits": record["expectedWorkUnitCount"], "totalWorkUnits": record["expectedWorkUnitCount"],
        "totalCandidateCount": record["candidateCount"], "completedCandidateCount": record["candidateCount"],
    }
    numerical.check_identities(status)
    for key, value in expected.items():
        _equal(status.get(key), value, f"supported dataset {key}")
    for key in ("progress",):
        if key in status and (isinstance(status[key], bool) or coarse._finite_number(status[key], key) != 1.0):
            raise ValueError("supported dataset progress is incomplete")
    _contributions(status.get("nodeContributions"), record["expectedWorkUnitCount"], "dataset contributions")
    completed = record["candidateCount"]
    invalid = _count(status.get("totalInvalidCandidateCount"), "invalid candidate count")
    rejected = _count(status.get("totalSupportRejectedCandidateCount"), "support-rejected candidate count")
    eligible = _count(status.get("totalEligibleCandidateCount"), "eligible candidate count")
    if invalid + rejected > completed or eligible != completed - invalid - rejected:
        raise ValueError("supported candidate accounting is inconsistent")
    payload = status.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"bestCandidate"}:
        raise ValueError("supported dataset result payload is invalid")
    winner = payload["bestCandidate"]
    if (winner is None) != (eligible == 0):
        raise ValueError("supported null winner disagrees with eligible count")
    components, axes = [], []
    if winner is not None:
        candidate = numerical.strict_candidate_payload(winner, dataset["modelClassID"])
        index = _count(candidate["gridIndex"], "accepted grid index")
        if index >= completed:
            raise ValueError("accepted grid index is outside the original grid")
        # The supported adapter verifies real public IDs; no investigation is relabeled.
        view, _ = numerical.numerical_view(dataset)
        axes = legacy._axis_boundaries(dataset, index)
        _equal(candidate["parameters"], {axis["axis"]: axis["value"] for axis in axes}, "accepted supported winner parameter mapping")
        evaluation = numerical.evaluate_candidate(view, index)
        if evaluation is None or not numerical.candidate_payload_matches(candidate, evaluation, dataset["modelClassID"]):
            raise ValueError("accepted supported winner does not reproduce")
        minimum = contract["crossSeriesRequirements"]["positiveWeightSupportPerComponentPerSeries"]
        for name, center, log_scale, log_shape in legacy._component_geometries(dataset["modelClassID"], candidate["parameters"]):
            support = [legacy._component_support(series, center, log_scale, log_shape, minimum) for series in dataset["series"]]
            if not all(item["supportRequirementMet"] for item in support):
                raise ValueError("accepted supported winner lacks observational support")
            components.append({
                "component": name, "center": center, "logScale": log_scale, "logShape": log_shape,
                "effectiveWidth": support[0]["effectiveWidth"], "seriesSupport": support, "supportRequirementMet": True,
                "amplitudeSigns": [{"genericSeriesID": fit["genericSeriesID"], "sign": fit[f"{name}AmplitudeSign"]}
                                   for fit in candidate["seriesFits"]],
            })
    for key in _BEST_KEYS:
        summary_key = "best" + key[0].upper() + key[1:]
        if summary_key not in status:
            raise ValueError(f"missing duplicated winner field {summary_key}")
        _equal(status[summary_key], winner[key] if winner else None, f"duplicated winner {summary_key}")
    return {
        "datasetID": dataset["id"], "modelClassID": dataset["modelClassID"],
        "genericSeriesIDs": dataset["sourceGenericSeriesIDs"], "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "coverageComplete": True, "completedWorkUnits": record["expectedWorkUnitCount"],
        "totalWorkUnits": record["expectedWorkUnitCount"], "totalCandidateCount": completed,
        "completedCandidateCount": completed, "totalInvalidCandidateCount": invalid,
        "totalSupportRejectedCandidateCount": rejected, "totalEligibleCandidateCount": eligible,
        "acceptedWinner": copy.deepcopy(winner), "acceptedWinnerReproduced": winner is not None,
        "boundaryAssessmentAvailable": winner is not None,
        "supportRequirementMet": True if winner else None, "components": components, "axes": axes,
        "searchedBoundaryAxes": [a["axis"] for a in axes if a["searchedBoundary"]],
        "fixedAxes": [a["axis"] for a in axes if a["fixed"]],
        "widthInterpretationLimitedByBoundary": any(a["searchedBoundary"] and ("Scale" in a["axis"] or "Shape" in a["axis"]) for a in axes),
    }


def _gate(inputs, passed, *, threshold=None, operator=None):
    return {"inputs": inputs, "threshold": threshold, "operator": operator,
            "evaluated": passed is not None, "passed": passed is True}


def _model_comparisons(searches, independent, contract):
    positive, ordered = (item["acceptedWinner"] for item in searches[:2])
    independent_winners = [item["acceptedWinner"] for item in searches[2:]]
    rules = contract["decisionRules"]
    ordered_rule = rules["preferOrderedDoubletOverPositivePulse"]
    independent_rule = rules["rejectOrderedDoubletForIndependentPulses"]
    per_series = []
    for index, series_id in enumerate(contract["admittedGenericSeriesIDs"]):
        p = positive["seriesFits"][index] if positive else None
        o = ordered["seriesFits"][index] if ordered else None
        i = independent_winners[index]["seriesFits"][0] if independent_winners[index] else None
        per_series.append({
            "genericSeriesID": series_id,
            "positiveWRSS": p["weightedResidualSumSquares"] if p else None,
            "orderedWRSS": o["weightedResidualSumSquares"] if o else None,
            "independentWRSS": i["weightedResidualSumSquares"] if i else None,
            "orderedOverPositiveDeltaWRSS": p["weightedResidualSumSquares"] - o["weightedResidualSumSquares"] if p and o else None,
            "independentOverOrderedDeltaWRSS": o["weightedResidualSumSquares"] - i["weightedResidualSumSquares"] if o and i else None,
            "positiveSign": p["positiveAmplitudeSign"] if p else None,
            "orderedSigns": [o["negativeAmplitudeSign"], o["positiveAmplitudeSign"]] if o else None,
            "independentSigns": [i["negativeAmplitudeSign"], i["positiveAmplitudeSign"]] if i else None,
        })
    comparisons = {"perSeries": per_series}
    for name, left, right, rule in (
        ("preferOrderedDoubletOverPositivePulse", positive, ordered, ordered_rule),
        ("rejectOrderedDoubletForIndependentPulses", ordered, independent, independent_rule),
    ):
        details = {}
        deltas = {}
        for metric, suffix in (("weightedResidualSumSquares", "WRSS"), ("bayesianInformationCriterion", "BIC")):
            delta = left[metric] - right[metric] if left and right else None
            threshold = rule[f"globalDelta{suffix}AtLeast"]
            deltas[f"delta{suffix}"] = delta
            details[f"globalDelta{suffix}Passed"] = _gate(
                {"left": left[metric] if left else None, "right": right[metric] if right else None, "delta": delta},
                delta >= threshold if delta is not None else None, threshold=threshold, operator=">=",
            )
        if name == "preferOrderedDoubletOverPositivePulse":
            threshold = rule["allPerSeriesDeltaWRSSAtLeast"]
            values = [item["orderedOverPositiveDeltaWRSS"] for item in per_series]
            constituent_gates = [{"genericSeriesID": item["genericSeriesID"],
                                  **_gate({"positiveWRSS": item["positiveWRSS"], "orderedWRSS": item["orderedWRSS"], "delta": value},
                                          value >= threshold if value is not None else None, threshold=threshold, operator=">=")}
                                 for item, value in zip(per_series, values)]
            details["allPerSeriesDeltaWRSSPassed"] = _gate(
                constituent_gates, all(value >= threshold for value in values) if all(value is not None for value in values) else None,
                threshold=threshold, operator="every series >=",
            )
            signs = [{"genericSeriesID": item["genericSeriesID"], "positive": item["positiveSign"], "ordered": item["orderedSigns"]}
                     for item in per_series]
            details["signRequirementsPassed"] = _gate(
                signs, all(item["positiveSign"] == "positive" and item["orderedSigns"] == ["negative", "positive"] for item in per_series)
                if positive and ordered else None, threshold={"positive": "positive", "ordered": ["negative", "positive"]}, operator="every series exact signs",
            )
            details["sharedTimingRequirementsPassed"] = _gate(
                {"modelClassID": workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET, "sharedDatasetGeometryVerified": ordered is not None},
                True if ordered else None, operator="one shared geometry for all series",
            )
            scope = searches[:2]
        else:
            details["centerDispersionExceedsTolerance"] = _gate(
                {"negativeCenterDispersion": independent["negativeCenterDispersion"] if independent else None,
                 "positiveCenterDispersion": independent["positiveCenterDispersion"] if independent else None,
                 "timingDispersion": independent["timingDispersion"] if independent else None},
                not independent["timingConsistent"] if independent else None,
                threshold=contract["crossSeriesRequirements"]["independentTimingConsistencyTolerance"] + contract["crossSeriesRequirements"]["timingComparisonTolerance"],
                operator=">",
            )
            details["signRequirementsPassed"] = _gate(
                [{"genericSeriesID": item["genericSeriesID"], "ordered": item["orderedSigns"], "independent": item["independentSigns"]} for item in per_series],
                all(item["orderedSigns"] == ["negative", "positive"] and item["independentSigns"] == ["negative", "positive"] for item in per_series)
                if ordered and independent else None, threshold=["negative", "positive"], operator="both models, every series exact signs",
            )
            scope = searches[1:]
        details["supportRequirementsPassed"] = _gate(
            [{"datasetID": item["datasetID"], "supportRequirementMet": item["supportRequirementMet"]} for item in scope],
            all(item["supportRequirementMet"] for item in scope) if all(item["acceptedWinner"] is not None for item in scope) else None,
            threshold=contract["crossSeriesRequirements"]["positiveWeightSupportPerComponentPerSeries"], operator="every component in every applicable series",
        )
        gates = {key: value["passed"] for key, value in details.items()}
        comparisons[name] = {**deltas, "thresholds": dict(rule), "gates": gates,
                             "gateDetails": details, "passed": all(gates.values())}
    # Retain the exact frozen PR186 outcomes when all inputs are available.
    if all(item["acceptedWinner"] is not None for item in searches):
        frozen = legacy._comparisons(searches, independent, contract)
        for name in ("preferOrderedDoubletOverPositivePulse", "rejectOrderedDoubletForIndependentPulses"):
            _equal(comparisons[name]["gates"], frozen[name]["gates"], "frozen interpretation gates")
    return comparisons


def _numerical_ranking(searches, independent, contract):
    metrics = [searches[0]["acceptedWinner"], searches[1]["acceptedWinner"], independent]
    ranked = []
    for ordinal, (model, candidate) in enumerate(zip(contract["modelClassOrder"], metrics)):
        if candidate is None:
            continue
        key = SimpleNamespace(
            weighted_residual_sum_squares=candidate["weightedResidualSumSquares"],
            bayesian_information_criterion=candidate["bayesianInformationCriterion"],
            corrected_akaike_information_criterion=candidate["correctedAkaikeInformationCriterion"],
            corrected_akaike_information_criterion_defined=candidate["correctedAkaikeInformationCriterionDefined"],
            grid_index=ordinal,
        )
        row = {"modelClassID": model, **{name: candidate[name] for name in _BEST_KEYS[3:]}}
        position = len(ranked)
        for index, (_, other) in enumerate(ranked):
            if numerical.candidate_precedes(key, other):
                position = index
                break
        ranked.insert(position, (row, key))
    return [row for row, _ in ranked]


def _report(supported, investigation, run, prepared, input_hashes):
    statuses = run.get("datasets")
    if not isinstance(statuses, list) or len(statuses) != 4 or any(not isinstance(item, Mapping) for item in statuses):
        raise ValueError("supported run requires exactly four canonical dataset summaries")
    _equal([item.get("id") for item in statuses], [item["id"] for item in supported.datasets], "supported dataset mapping and order")
    searches = [_search(status, dataset, record, prepared.contract)
                for status, dataset, record in zip(statuses, supported.datasets, supported.manifest["datasets"])]
    total_work = supported.manifest["totalExpectedWorkUnitCount"]
    _verify_run_counter_scopes(run, statuses[-1], expected_project_work_units=total_work)
    if "projectProgress" in run and (isinstance(run["projectProgress"], bool) or coarse._finite_number(run["projectProgress"], "projectProgress") != 1.0):
        raise ValueError("supported project progress is incomplete")
    totals = {key: coarse._safe_sum([item[key] for item in searches], key) for key in (
        "completedWorkUnits", "totalWorkUnits", "totalCandidateCount", "completedCandidateCount",
        "totalInvalidCandidateCount", "totalSupportRejectedCandidateCount", "totalEligibleCandidateCount",
    )}
    _equal(totals["completedWorkUnits"], total_work, "summed dataset work counts")
    _equal(totals["totalWorkUnits"], total_work, "summed dataset total work counts")
    _equal(totals["totalCandidateCount"], supported.manifest["totalCandidateCount"], "summed candidate counts")
    contributions = {}
    for status in statuses:
        for node, count in status["nodeContributions"].items():
            contributions[node] = contributions.get(node, 0) + count
    _contributions(run.get("nodeContributions"), total_work, "project contributions")
    _equal(contributions, run["nodeContributions"], "summed node contributions")
    independent_winners = [item["acceptedWinner"] for item in searches[2:]]
    independent = legacy._independent_aggregate(independent_winners, prepared.contract) if all(w is not None for w in independent_winners) else None
    comparisons = _model_comparisons(searches, independent, prepared.contract)
    preference = None
    null_count = sum(item["acceptedWinner"] is None for item in searches)
    classification = "MODEL_COMPARISON_UNRESOLVED"
    if null_count:
        classification = "UNRESOLVED_NO_ELIGIBLE_CANDIDATES"
    elif comparisons["rejectOrderedDoubletForIndependentPulses"]["passed"]:
        preference, classification = workload.INDEPENDENT_PULSES, "INDEPENDENT_PULSES_PREFERRED"
    elif comparisons["preferOrderedDoubletOverPositivePulse"]["passed"]:
        preference, classification = workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET, "ORDERED_DOUBLET_PREFERRED"
    limited = any(item["widthInterpretationLimitedByBoundary"] for item in searches)
    next_test = (
        "PREDECLARE_SUPPORTED_MORPHOLOGY_DOMAIN_REVIEW" if null_count else
        "BLIND_ANOMALY_MORPHOLOGY_REFINEMENT_AND_SUPPORT_REVIEW" if preference else
        "BLIND_ANOMALY_MORPHOLOGY_FAILED_GATE_AND_BOUNDARY_REVIEW"
    )
    return {
        "resultSchemaID": REPORT_SCHEMA_ID, "resultVersion": REPORT_VERSION,
        "supportPolicyID": workload.SUPPORT_POLICY_ID, "workloadID": workload.WORKLOAD_ID,
        "projectID": supported.project["id"], "investigationID": investigation["id"],
        "coverageComplete": True, "projectCompletedWorkUnits": total_work, "projectTotalWorkUnits": total_work,
        "accountingTotals": totals, "nodeContributions": contributions, "searches": searches,
        "completeNullWinnerSearchCount": null_count, "independentAggregate": independent,
        "numericalRanking": _numerical_ranking(searches, independent, prepared.contract),
        "modelComparisons": comparisons, "overallClassification": classification,
        "modelPreference": preference, "modelPreferenceResolved": preference is not None,
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
        "widthInterpretationResolved": False, "widthInterpretationLimitedByBoundary": limited,
        "recommendedNextTest": next_test, "inputHashes": input_hashes,
        "provenance": copy.deepcopy(supported.manifest["provenance"]),
        "morphologyContractID": prepared.contract["contractID"],
        "morphologyContractSHA256": prepared.contract_canonical_sha256,
        "supportRule": {"effectiveWidthFormula": "exp(logScale) * exp(logShape)", "inclusiveWidthMultiplier": 2.0,
                        "minimumPositiveWeightSamples": prepared.contract["crossSeriesRequirements"]["positiveWeightSupportPerComponentPerSeries"],
                        "zeroWeightsExcluded": True, "zeroAmplitudeStillRequiresSupport": True},
        "verificationScope": {
            "sourceArtifactsAndLedgersVerified": True, "counterConsistencyVerified": True,
            "acceptedSupportedCandidatesReproduced": 4 - null_count,
            "fullGridEnumerated": False, "shardsRecomputed": False,
            "rejectionCountsIndependentlyReproduced": False, "globalOptimalityIndependentlyProven": False,
        },
        "limitations": [
            "Coverage and rejection totals are checked against persisted producer accounting; individual rejection decisions and global optimality are not independently reproduced.",
            "Only saved accepted winners are numerically reproduced; no runner-up or new search is evaluated.",
            "Geometric support is eligibility, not measured duration, convergence, cross-series replication, or planetary evidence.",
            "Searched boundary winners retain bounded-domain limitations; fixed axes are not searched boundaries.",
            "Numerical ranking is separate from model preference; every frozen improvement, sign, timing and support gate must pass.",
            "A complete null-winner search leaves interpretation unresolved for this grid; it does not exclude a model outside this domain.",
        ],
    }


def _markdown(report):
    totals = report["accountingTotals"]
    model_names = {workload.POSITIVE_PULSE_ONLY: "Positive pulse", workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET: "Ordered doublet",
                   workload.INDEPENDENT_PULSES: "Independent pulses"}
    comparison_names = {"preferOrderedDoubletOverPositivePulse": "Ordered vs positive",
                        "rejectOrderedDoubletForIndependentPulses": "Independent vs ordered"}
    gate_names = {"globalDeltaWRSSPassed": "Combined WRSS improvement", "globalDeltaBICPassed": "BIC improvement",
                  "allPerSeriesDeltaWRSSPassed": "Improvement in every series", "signRequirementsPassed": "Strict amplitude signs",
                  "sharedTimingRequirementsPassed": "Shared timing", "supportRequirementsPassed": "Geometric support",
                  "centerDispersionExceedsTolerance": "Timing disagreement exceeds tolerance"}
    lines = ["# Supported anomaly morphology validation", "",
             f"Classification: **{report['overallClassification']}**.",
             f"Work: {report['projectCompletedWorkUnits']}/{report['projectTotalWorkUnits']}; candidates: "
             f"{totals['completedCandidateCount']} evaluated, {totals['totalInvalidCandidateCount']} invalid, "
             f"{totals['totalSupportRejectedCandidateCount']} support-rejected, {totals['totalEligibleCandidateCount']} eligible.", "",
             "| Search | Eligible | Winner index | Searched boundaries |", "| --- | ---: | ---: | --- |"]
    for search in report["searches"]:
        winner = search["acceptedWinner"]
        name = model_names[search["modelClassID"]]
        if search["modelClassID"] == workload.INDEPENDENT_PULSES:
            name += f" ({search['genericSeriesIDs'][0]})"
        lines.append(f"| {name} | {search['totalEligibleCandidateCount']} | "
                     f"{winner['gridIndex'] if winner else 'none'} | {', '.join(search['searchedBoundaryAxes']) or 'none'} |")
    lines += ["", "Numerical ranking (separate from gated preference): " +
              (", ".join(model_names[item["modelClassID"]] for item in report["numericalRanking"]) or "unavailable") + ".", "",
              "| Comparison / gate | Result |", "| --- | --- |"]
    for name, comparison in report["modelComparisons"].items():
        if name == "perSeries":
            continue
        for gate, detail in comparison["gateDetails"].items():
            verdict = "not evaluable" if not detail["evaluated"] else "pass" if detail["passed"] else "FAIL"
            lines.append(f"| {comparison_names[name]}: {gate_names[gate]} | {verdict} |")
    lines += ["", "The JSON report contains every gate input and threshold, per-series signs and improvements, support counts, distances, and axis indices.",
              "", "Only accepted winners were reproduced. Rejection counts were checked for consistency; this report does not independently prove global optimality.",
              "Geometric support does not establish measured duration, convergence, replication, or planetary evidence. Boundary warnings remain in force.",
              "Planetary interpretation: unresolved. Discovery claim: false.", "",
              f"Next test: `{report['recommendedNextTest']}`.", ""]
    return "\n".join(lines).encode("utf-8")


def _validate_impl(morphology_root, *, coarse_project_root, coarse_investigation_record, coarse_validation_root,
                   supported_project_root, supported_investigation_record, output_root):
    morphology, coarse_root, coarse_record, coarse_report, supported_root, supported_record, output = (
        Path(value).expanduser().absolute() for value in (
            morphology_root, coarse_project_root, coarse_investigation_record, coarse_validation_root,
            supported_project_root, supported_investigation_record, output_root,
        )
    )
    if output.exists() or output.is_symlink():
        raise ValueError("output root already exists")
    coarse._reject_symlink_components(output.parent, "output root")
    for path in (morphology, coarse_root, coarse_record, coarse_report, supported_root, supported_record):
        coarse._reject_symlink_components(path, "input")
    for source in (morphology, coarse_root, coarse_record.parent, coarse_report, supported_root, supported_record.parent):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("output root must not be inside an input artifact directory")
    prepared = coarse._verify_preparation(morphology)
    for document in (prepared.contract, prepared.preparation, prepared.manifest, *prepared.series):
        legacy._finite_tree(document)
    legacy._verify_interpretation_contract(prepared.contract)
    grid = legacy._verify_grid_root(coarse_root, prepared)
    original = legacy._verify_investigation(coarse_record, grid, coarse_root / "project.json")
    source_manifest, report_hashes = builder._verify_validation_root(coarse_report, prepared, grid, original)
    supported = _verify_supported_project(supported_root, prepared, grid, original, source_manifest, report_hashes)
    investigation, run, investigation_hashes = _verify_stages(supported_record, supported, supported_root / "project.json")
    input_hashes = {**copy.deepcopy(source_manifest["inputHashes"]), "coarseValidationArtifacts": report_hashes,
                    "supportedProjectArtifacts": dict(supported.hashes), **investigation_hashes}
    report = _report(supported, investigation, run, prepared, input_hashes)
    report_bytes, markdown_bytes = _json(report), _markdown(report)
    manifest = {
        "artifactManifestSchemaID": MANIFEST_SCHEMA_ID, "artifactManifestVersion": REPORT_VERSION,
        "resultSchemaID": REPORT_SCHEMA_ID, "resultVersion": REPORT_VERSION,
        "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "relativeArtifactPaths": {"result": RESULT_RELATIVE_PATH, "markdown": MARKDOWN_RELATIVE_PATH, "artifactManifest": MANIFEST_RELATIVE_PATH},
        "outputSHA256s": {RESULT_RELATIVE_PATH: _hash(report_bytes), MARKDOWN_RELATIVE_PATH: _hash(markdown_bytes)},
        "inputHashes": input_hashes, "provenance": report["provenance"],
        "projectID": supported.project["id"], "investigationID": investigation["id"],
        "planetaryInterpretationResolved": False, "discoveryClaim": False,
    }
    manifest_bytes = _json(manifest)
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


def validate_supported_anomaly_morphology_grid(
    morphology_root: str | Path, *, coarse_project_root: str | Path, coarse_investigation_record: str | Path,
    coarse_validation_root: str | Path, supported_project_root: str | Path,
    supported_investigation_record: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    """Verify immutable artifacts and atomically publish the offline report."""
    try:
        return _validate_impl(
            morphology_root, coarse_project_root=coarse_project_root, coarse_investigation_record=coarse_investigation_record,
            coarse_validation_root=coarse_validation_root, supported_project_root=supported_project_root,
            supported_investigation_record=supported_investigation_record, output_root=output_root,
        )
    except SupportedAnomalyMorphologyGridValidationError:
        raise
    except (KeyError, IndexError, OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise SupportedAnomalyMorphologyGridValidationError(str(error)) from error


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("morphology-root", "coarse-project-root", "coarse-investigation-record", "coarse-validation-root",
                 "supported-project-root", "supported-investigation-record", "output-root"):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        validate_supported_anomaly_morphology_grid(**vars(arguments))
    except SupportedAnomalyMorphologyGridValidationError as error:
        print(f"Supported morphology validation failed: {error}")
        return 1
    print(f"Published supported morphology validation: {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
