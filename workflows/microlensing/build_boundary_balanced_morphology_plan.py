"""Predeclare a comparable local morphology design and its budget, without execution."""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from openstar_workloads.plugins import supported_morphology_grid as workload
from openstar_workloads.plugins.supported_morphology_grid import _adapter as numerical
from workflows.microlensing import build_anomaly_morphology_coarse_grid as coarse
from workflows.microlensing import build_supported_doublet_refinement as parent_builder
from workflows.microlensing import validate_supported_anomaly_morphology_grid as supported
from workflows.microlensing import validate_supported_doublet_refinement as refinement
from workflows.microlensing.coarse_grid import _assert_identity_free, _atomic_write_bytes

PLAN_SCHEMA_ID = "openstar.microlensing-boundary-balanced-morphology-plan.v1"
MANIFEST_SCHEMA_ID = "openstar.microlensing-boundary-balanced-morphology-plan-artifacts.v1"
VERSION = "1.0"
CANDIDATE_LIMIT = 30_000
MAX_LATTICE_SUBDIVISIONS = 64
PLAN_PATH = "boundary-balanced-morphology-plan.json"
MARKDOWN_PATH = "boundary-balanced-morphology-plan.md"
MANIFEST_PATH = "artifact-manifest.json"
_REACHED = ("negativeCenter", "negativeLogScale", "positiveLogScale")


class BoundaryBalancedMorphologyPlanError(RuntimeError):
    """The source artifacts or proposed geometry cannot be verified safely."""


def _verify_inputs(paths):
    parent, baseline, hashes = parent_builder._verify_sources(*paths[:7])
    contract_bytes, contract = supported._read(paths[0] / coarse.PREPARATION_CONTRACT_RELATIVE_PATH, "morphology contract")
    supported._equal(supported._hash(contract_bytes), hashes["morphologyContractFile"], "plan morphology contract hash")
    project = refinement._verify_project(paths[7], parent, baseline, hashes)
    investigation, run, stages = supported._verify_stages(paths[8], project, paths[7] / parent_builder.PROJECT_RELATIVE_PATH)
    search = refinement._verify_search(project, run, contract)
    input_hashes = {**hashes, "refinementProjectArtifacts": dict(project.hashes),
                    "refinementInvestigationRecord": stages["supportedInvestigationRecord"],
                    "refinementStageLedgers": stages["supportedStageLedgers"]}
    expected = refinement._report(project, investigation, run, search, baseline, contract, input_hashes)
    root = paths[9]
    names = (refinement.RESULT_RELATIVE_PATH, refinement.MARKDOWN_RELATIVE_PATH, refinement.MANIFEST_RELATIVE_PATH)
    supported._exact_artifacts(root, names, "refinement validation")
    report_bytes, report = supported._read(root / names[0], "refinement validation report")
    markdown_bytes = coarse._read_bytes(root / names[1], "refinement validation Markdown")
    manifest_bytes, manifest = supported._read(root / names[2], "refinement validation manifest")
    supported._equal(report, expected, "refinement validation report")
    if markdown_bytes != refinement._markdown(expected):
        raise ValueError("refinement validation Markdown does not reconstruct exactly")
    expected_manifest = {
        "artifactManifestSchemaID": refinement.MANIFEST_SCHEMA_ID, "artifactManifestVersion": refinement.REPORT_VERSION,
        "resultSchemaID": refinement.REPORT_SCHEMA_ID, "resultVersion": refinement.REPORT_VERSION,
        "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "relativeArtifactPaths": {"result": names[0], "markdown": names[1], "artifactManifest": names[2]},
        "outputSHA256s": {names[0]: supported._hash(report_bytes), names[1]: supported._hash(markdown_bytes)},
        "inputHashes": input_hashes, "provenance": expected["provenance"], "projectID": project.project["id"],
        "investigationID": investigation["id"], **copy.deepcopy(refinement._CLAIMS),
    }
    supported._equal(manifest, expected_manifest, "refinement validation manifest")
    coarse._assert_identity_isolated((report, manifest))
    _assert_identity_free((report_bytes, markdown_bytes, manifest_bytes))
    input_hashes["refinementValidationArtifacts"] = dict(zip(names, map(supported._hash, (report_bytes, markdown_bytes, manifest_bytes))))
    return project.datasets[0], report, contract, input_hashes


def _require_followup(report):
    supported._equal(report["recommendedNextTest"], "PREDECLARE_BOUNDARY_AND_BALANCED_MODEL_FOLLOWUP", "source next test")
    supported._equal(report["overallClassification"], "UNRESOLVED_SEARCHED_BOUNDARY", "source classification")
    search = report["search"]
    if search["coverageComplete"] is not True or search["acceptedWinner"] is None or search["supportRequirementMet"] is not True:
        raise ValueError("follow-up requires a complete, reproduced and supported winner")
    gate = report["historicalBaselineComparisons"]["preferOrderedDoubletOverPositivePulse"]
    if not gate["evaluated"] or not gate["passed"]:
        raise ValueError("follow-up requires all historical ordered-over-positive gates to pass")
    axes = {axis["axis"]: axis for axis in search["axes"]}
    if set(search["searchedBoundaryAxes"]) != set(_REACHED) or any(axes[name]["position"] != "LOWER_BOUNDARY" for name in _REACHED):
        raise ValueError("follow-up requires exactly the three lower center/log-scale boundaries")
    if axes["separation"]["position"] != "INTERIOR" or not all(axes[name]["fixed"] for name in ("negativeLogShape", "positiveLogShape")):
        raise ValueError("follow-up requires interior separation and both fixed shapes")


def _end(axis):
    return coarse._finite_number(axis["start"] + (axis["count"] - 1) * axis["step"], "axis endpoint")


def _require_distinct_axis(axis):
    end = _end(axis)
    # Check representable spacing across the whole interval without enumerating
    # even a one-dimensional axis. Smaller steps can collapse interior values.
    if axis["count"] > 1 and axis["step"] < max(math.ulp(axis["start"]), math.ulp(end)):
        raise ValueError("proposed axis collapses in binary64")


def _common_step(steps, offsets=()):
    steps = [coarse._finite_number(value, "source step") for value in steps]
    if min(steps) <= 0.0:
        raise ValueError("source steps must be positive")
    # This bounded lattice-alignment calculation does not visit candidates.
    # No coarsening is allowed: every common step is <= each source step.
    for subdivision in range(1, MAX_LATTICE_SUBDIVISIONS + 1):
        tick = min(steps) / subdivision
        if tick <= 0.0:
            break
        ratios = [coarse._finite_number(value / tick, "lattice ratio") for value in (*steps, *offsets)]
        if all(abs(value - round(value)) <= workload.RESULT_RELATIVE_TOLERANCE for value in ratios):
            return tick, subdivision
    raise ValueError("source axes do not admit the declared common lattice within 64 subdivisions")


def _cover_axis(lower, upper, step):
    lower, upper, step = (coarse._finite_number(value, "proposed axis geometry") for value in (lower, upper, step))
    if step <= 0.0 or upper < lower:
        raise ValueError("proposed axis bounds or step are invalid")
    count = coarse._exact_count(math.ceil((upper - lower) / step) + 1, "proposed axis count", positive=True)
    axis = {"start": lower, "step": step, "count": count}
    if _end(axis) < upper:
        axis["count"] = coarse._exact_count(count + 1, "outward-rounded axis count", positive=True)
    _require_distinct_axis(axis)
    return axis


def _design(dataset):
    workload.PLUGIN.validate_dataset(dataset)
    source = dataset["morphologyGrid"]
    expanded, derivation = {}, {}
    for name in _REACHED:
        axis = source[name + "Axis"]
        span = (axis["count"] - 1) * axis["step"]
        if span <= 0.0:
            raise ValueError("reached axes must have a searched span")
        expanded[name] = (coarse._finite_number(axis["start"] - span, "extended lower bound"), _end(axis))
        derivation[name] = {"sourceAxis": copy.deepcopy(axis), "lowerExtensionInSourceSteps": axis["count"] - 1,
                            "requestedMinimum": expanded[name][0], "requestedMaximum": expanded[name][1]}
    sep = source["separationAxis"]
    timing_step, timing_subdivisions = _common_step((source["negativeCenterAxis"]["step"], sep["step"]), (sep["start"],))
    negative = _cover_axis(*expanded["negativeCenter"], timing_step)
    separation_start_index = round(sep["start"] / timing_step)
    if separation_start_index < 1:
        raise ValueError("separations must remain strictly positive")
    separation = _cover_axis(separation_start_index * timing_step, _end(sep), timing_step)
    maximum_separation_index = separation_start_index + separation["count"] - 1
    center_count = coarse._exact_count(negative["count"] + maximum_separation_index, "common center count", positive=True)
    centers = {"start": negative["start"], "step": timing_step, "count": center_count}
    _require_distinct_axis(centers)
    width_lower = min(expanded[name][0] for name in ("negativeLogScale", "positiveLogScale"))
    width_upper = max(expanded[name][1] for name in ("negativeLogScale", "positiveLogScale"))
    width_step, width_subdivisions = _common_step(
        (source["negativeLogScaleAxis"]["step"], source["positiveLogScaleAxis"]["step"]),
        (expanded["negativeLogScale"][0] - width_lower, expanded["positiveLogScale"][0] - width_lower),
    )
    widths = _cover_axis(width_lower, width_upper, width_step)
    negative_shape, positive_shape = (copy.deepcopy(source[name + "Axis"]) for name in ("negativeLogShape", "positiveLogShape"))
    width_axes = {"negativeLogScaleAxis": widths, "positiveLogScaleAxis": widths,
                  "negativeLogShapeAxis": negative_shape, "positiveLogShapeAxis": positive_shape}
    specifications = (
        (workload.POSITIVE_PULSE_ONLY, dataset["series"], {"centerAxis": centers, "logScaleAxis": widths, "logShapeAxis": positive_shape}),
        (workload.ORDERED_NEGATIVE_POSITIVE_DOUBLET, dataset["series"], {"negativeCenterAxis": negative, "separationAxis": separation, **width_axes}),
        *((workload.INDEPENDENT_PULSES, [series], {"centerAxis": centers, **width_axes}) for series in dataset["series"]),
    )
    searches = []
    for ordinal, (model, series, grid) in enumerate(specifications, 1):
        specification = {
            "id": f"planned-supported-morphology.{ordinal:03d}", **supported._IDENTITIES,
            "morphologyFamilyID": workload.MORPHOLOGY_FAMILY_ID, "componentTemplateFamilyID": workload.COMPONENT_TEMPLATE_FAMILY_ID,
            "modelClassID": model, "series": copy.deepcopy(series), "morphologyGrid": copy.deepcopy(grid),
            "candidatesPerWorkUnit": dataset["candidatesPerWorkUnit"],
        }
        _, validated = numerical.numerical_view(specification)
        count, size = validated.grid.total_candidates, specification["candidatesPerWorkUnit"]
        samples = coarse._safe_sum([len(item["coordinates"]) for item in series], "search sample count")
        units = (count + size - 1) // size
        searches.append({
            "searchID": specification["id"], "modelClassID": model, "genericSeriesIDs": [item["genericSeriesID"] for item in series],
            "datasetSpecification": specification, "candidateCount": count, "candidatesPerWorkUnit": size,
            "workUnitCount": units, "lastShardGridStartIndex": (units - 1) * size, "lastShardGridCount": count - (units - 1) * size,
            "sampleCount": samples, "sampleCandidateEvaluationCount": coarse._safe_product([samples, count], "sample-candidate budget"),
        })
    pair_count = centers["count"] * (centers["count"] - 1) // 2
    comparison = {
        "scope": "LOCAL_SHARED_DOMAIN_AND_RESOLUTION_WITH_DECLARED_INDEPENDENT_SUPERSET",
        "commonCenterAxis": centers, "commonLogScaleAxis": widths, "sharedNegativeCenterAxis": negative, "sharedSeparationAxis": separation,
        "timingLatticeSubdivisions": timing_subdivisions, "widthLatticeSubdivisions": width_subdivisions,
        "resolutionPolicy": "No coarser than either current component axis; use the largest aligned subdivision of the smallest source step (at most 64 subdivisions).",
        "roundingPolicy": "Upper endpoints round outward; separation origin aligns within the published numerical tolerance.",
        "orderedPairEmbedding": {
            "negativeIndex": "i", "positiveIndex": f"i + {separation_start_index} + j",
            "negativeIndexRange": [0, negative["count"] - 1], "separationIndexRange": [0, separation["count"] - 1],
            "independentCommonCenterCount": centers["count"], "minimumSeparationInCommonSteps": separation_start_index,
            "independentPairOrdering": "lexicographic 0 <= negativeIndex < positiveIndex < commonCenterCount",
            "independentPairIndexFormula": "negativeIndex*(2*commonCenterCount-negativeIndex-1)//2 + positiveIndex-negativeIndex-1",
            "sharedTimingPairCount": negative["count"] * separation["count"], "independentTimingPairCount": pair_count,
            "extraIndependentTimingPairCount": pair_count - negative["count"] * separation["count"],
            "parameterComparisonTolerance": workload.RESULT_RELATIVE_TOLERANCE,
        },
        "coverageStatement": "Every shared ordered timing pair is represented on each independent common axis within published parameter tolerance. Positive-only has the entire common timing axis. Every component gets the same union of extended log-scale ranges at the same resolution, with its shape fixed.",
        "independentSupersetStatement": "The worker's common center axis necessarily permits additional independent pairs outside the shared negative-center/separation rectangle; they are counted, not filtered away. Full parameter domains are not identical.",
        "equalCandidateCountsDefineBalance": False, "balancedModelComparisonEstablished": False,
    }
    return searches, derivation, comparison


def _plan(dataset, report, contract, hashes):
    _require_followup(report)
    searches, derivation, comparison = _design(dataset)
    totals = {key: coarse._safe_sum([search[key] for search in searches], key) for key in
              ("candidateCount", "workUnitCount", "sampleCandidateEvaluationCount")}
    fits_budget = totals["candidateCount"] <= CANDIDATE_LIMIT
    return {
        "planSchemaID": PLAN_SCHEMA_ID, "planVersion": VERSION, "supportPolicyID": workload.SUPPORT_POLICY_ID,
        "planStatus": "PREDECLARED_FOR_REVIEW" if fits_budget else "BUDGET_CONFLICT",
        "requiredSearches": searches, "proposedSearchIDs": [search["searchID"] for search in searches] if fits_budget else [],
        "budget": {
            "candidateLimit": CANDIDATE_LIMIT, "requiredCandidateCount": totals["candidateCount"],
            "requiredWorkUnitCount": totals["workUnitCount"], "requiredSampleCandidateEvaluationCount": totals["sampleCandidateEvaluationCount"],
            "proposedCandidateCount": totals["candidateCount"] if fits_budget else 0,
            "proposedWorkUnitCount": totals["workUnitCount"] if fits_budget else 0,
            "proposedSampleCandidateEvaluationCount": totals["sampleCandidateEvaluationCount"] if fits_budget else 0,
            "fitsDeclaredResolutionBudget": fits_budget, "candidateShortfall": max(0, totals["candidateCount"] - CANDIDATE_LIMIT),
            "conflict": None if fits_budget else "The complete required design exceeds 30,000 candidates at the predeclared resolution. No searches are allocated. A changed budget or resolution requires a separately reviewed predeclaration; no domain is silently narrowed and no claim of infeasibility at all possible resolutions is made.",
        },
        "axisDerivation": derivation, "comparability": comparison,
        "sourceProjectID": report["projectID"], "sourceInvestigationID": report["investigationID"],
        "sourceValidationSchemaID": report["resultSchemaID"], "sourceAcceptedWinner": copy.deepcopy(report["search"]["acceptedWinner"]),
        "sourceBoundaryAssessment": copy.deepcopy(report["search"]["axes"]), "inputHashes": hashes,
        "parentLineage": copy.deepcopy(report["provenance"]),
        "preservedHistoricalResults": copy.deepcopy(report["preservedPR190Report"]),
        "frozenNumericalExecution": copy.deepcopy(contract["deterministicExecution"]),
        "frozenComparisonRules": copy.deepcopy(contract["decisionRules"]),
        "frozenCrossSeriesRequirements": copy.deepcopy(contract["crossSeriesRequirements"]),
        "supportRule": copy.deepcopy(report["supportRule"]),
        "stoppingRules": [
            {"condition": "required design exceeds budget", "outcome": "UNRESOLVED_BUDGET_CONFLICT", "action": "Do not execute a partial design or reallocate only one model's opportunities."},
            {"condition": "any accepted winner remains on a searched boundary", "outcome": "UNRESOLVED_BOUNDARY", "action": "Stop after the declared searches; require a separately reviewed plan."},
            {"condition": "sampling or timing/width resolution remains unresolved, including interior winners", "outcome": "UNRESOLVED_SAMPLING", "action": "Do not infer convergence or measured duration from an interior grid point."},
            {"condition": "budget is exhausted, coverage incomplete, or no eligible winner", "outcome": "UNRESOLVED", "action": "Do not automatically repeat or refine searches."},
            {"condition": "all searches complete", "outcome": "REQUIRES_OFFLINE_REVIEW", "action": "Reproduce saved winners, apply unchanged signs/support/per-series gates and review independent extra pairs; planning alone establishes no model preference."},
        ],
        "automaticRepeatedRefinement": False, "executionAuthorized": False, "projectLaunched": False,
        **copy.deepcopy(refinement._CLAIMS), "verificationScope": copy.deepcopy(report["verificationScope"]),
        "limitations": [
            "This is a local extension, not a rerun of every historical coarse domain or evidence for a global optimum.",
            "Comparable access is defined by interval coverage and common resolution, not equal candidate counts. Independent searches necessarily have extra timing pairs.",
            "Binary64 addition order remains unchanged; a within-tolerance timing embedding is not proof of identical support decisions at an inclusive floating-point boundary.",
            "No proposed candidate was evaluated. Only saved winners required for ancestry verification were reproduced; rejection counts were checked for consistency, not independently recomputed.",
            "Geometric support is not measured duration, convergence, replication or planetary evidence. Historical alternatives received different earlier search effort.",
        ],
    }


def _markdown(plan):
    budget = plan["budget"]
    lines = ["# Boundary and comparable-model follow-up plan", "", f"Status: **{plan['planStatus']}**. No project is launched or authorized.",
             f"Required design: {budget['requiredCandidateCount']:,} candidates, {budget['requiredWorkUnitCount']:,} work units, "
             f"{budget['requiredSampleCandidateEvaluationCount']:,} sample-candidate evaluations.",
             f"Budget: {budget['candidateLimit']:,} candidates; proposed allocation: {budget['proposedCandidateCount']:,}.", ""]
    if budget["conflict"]:
        lines.extend([budget["conflict"], ""])
    lines.extend(["Extend each reached lower bound by one full current axis span and retain its upper bound. Keep the separation interval and both shapes fixed.",
                  "Use a common timing lattice and a union log-scale axis, without coarsening current resolution. The independent common center axis covers every shared ordered pair and includes the extra pairs required by its worker contract.", "",
                  "| Search | Series | Candidates | Work units | Sample-candidate evaluations |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for search in plan["requiredSearches"]:
        lines.append(f"| {search['modelClassID']} | {', '.join(search['genericSeriesIDs'])} | {search['candidateCount']:,} | {search['workUnitCount']:,} | {search['sampleCandidateEvaluationCount']:,} |")
    lines.extend(["", "Exact required axes (including unfunded designs):", ""])
    for search in plan["requiredSearches"]:
        lines.extend([f"**{search['searchID']}**", ""])
        for name, axis in search["datasetSpecification"]["morphologyGrid"].items():
            description = f"values={axis['values']}" if "values" in axis else f"start={axis['start']!r}, step={axis['step']!r}, count={axis['count']}, end={_end(axis)!r}"
            lines.append(f"- {name}: {description}")
        lines.append("")
    lines.extend(["Stop after the declared work. Remaining boundaries, unresolved sampling, incomplete coverage or exhausted budget remain unresolved; no repeated refinement is scheduled.",
                  "Equal counts do not define balance. Independent extra pairs and floating-point support boundaries require review. Model preference and all scientific claims remain unresolved.", ""])
    return "\n".join(lines).encode("utf-8")


def _build_impl(morphology_root, *, coarse_project_root, coarse_investigation_record, coarse_validation_root,
                supported_project_root, supported_investigation_record, supported_validation_root,
                refinement_project_root, refinement_investigation_record, refinement_validation_root, output_root):
    paths = [Path(value).expanduser().absolute() for value in (
        morphology_root, coarse_project_root, coarse_investigation_record, coarse_validation_root,
        supported_project_root, supported_investigation_record, supported_validation_root,
        refinement_project_root, refinement_investigation_record, refinement_validation_root,
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
    plan = _plan(*_verify_inputs(paths))
    plan_bytes, markdown_bytes = supported._json(plan), _markdown(plan)
    manifest = {"artifactManifestSchemaID": MANIFEST_SCHEMA_ID, "artifactManifestVersion": VERSION,
                "planSchemaID": PLAN_SCHEMA_ID, "planVersion": VERSION, "inputHashes": plan["inputHashes"],
                "supportPolicyID": workload.SUPPORT_POLICY_ID, "provenance": plan["parentLineage"],
                "relativeArtifactPaths": {"plan": PLAN_PATH, "markdown": MARKDOWN_PATH, "artifactManifest": MANIFEST_PATH},
                "outputSHA256s": {PLAN_PATH: supported._hash(plan_bytes), MARKDOWN_PATH: supported._hash(markdown_bytes)}}
    manifest_bytes = supported._json(manifest)
    coarse._assert_identity_isolated((plan, manifest))
    _assert_identity_free((plan_bytes, markdown_bytes, manifest_bytes))
    output.parent.mkdir(parents=True, exist_ok=True)
    coarse._reject_symlink_components(output.parent, "output root")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, data in ((PLAN_PATH, plan_bytes), (MARKDOWN_PATH, markdown_bytes), (MANIFEST_PATH, manifest_bytes)):
            _atomic_write_bytes(staging / relative, data)
        if output.exists() or output.is_symlink():
            raise ValueError("output root already exists")
        staging.rename(output)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return {"plan": plan, "artifactManifest": manifest}


def build_boundary_balanced_morphology_plan(morphology_root: str | Path, **source_arguments: Any) -> dict[str, Any]:
    """Verify inputs and publish only a plan; a budget conflict is a valid plan outcome."""
    try:
        return _build_impl(morphology_root, **source_arguments)
    except BoundaryBalancedMorphologyPlanError:
        raise
    except (KeyError, IndexError, OSError, OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise BoundaryBalancedMorphologyPlanError(str(error)) from error


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("morphology-root", "coarse-project-root", "coarse-investigation-record", "coarse-validation-root",
                 "supported-project-root", "supported-investigation-record", "supported-validation-root", "refinement-project-root",
                 "refinement-investigation-record", "refinement-validation-root", "output-root"):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = build_boundary_balanced_morphology_plan(**vars(_parser().parse_args(argv)))
    except BoundaryBalancedMorphologyPlanError as error:
        print(f"Boundary/comparable-model planning failed: {error}")
        return 1
    budget = result["plan"]["budget"]
    print(f"{result['plan']['planStatus']}: {budget['requiredCandidateCount']} required; {budget['proposedCandidateCount']}/{CANDIDATE_LIMIT} proposed candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
