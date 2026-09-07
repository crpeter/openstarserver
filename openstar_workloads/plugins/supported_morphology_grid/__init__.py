"""Generic morphology search with frozen observational-support eligibility."""

from __future__ import annotations

import math
from typing import Any, Iterator, Mapping, Sequence

from openstar_workloads.contract import DatasetReduction, ResultValidation, WorkloadDefinition
from . import _adapter as _numeric
from ._adapter import (
    COMPONENT_TEMPLATE_FAMILY_ID, DATASET_SCHEMA_ID, EXECUTION_CONTRACT_ID,
    EXECUTION_CONTRACT_VERSION, INDEPENDENT_PULSES, MAX_SAFE_INTEGER,
    MODEL_CLASS_IDS, MORPHOLOGY_FAMILY_ID, ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    PAYLOAD_SCHEMA_ID, POSITIVE_PULSE_ONLY, RESULT_RELATIVE_TOLERANCE,
    RESULT_SCHEMA_ID, SUPPORT_POLICY_ID, VALIDATOR_ID, WORKLOAD_ID,
    candidate_index, candidate_indices, independent_center_pair_index,
    independent_center_pair_indices,
)

_WORK_FIELDS = frozenset((
    "morphologyFamilyID", "modelClassID", "supportPolicyID", "gridStartIndex", "gridCount",
))
_RESULT_FIELDS = _WORK_FIELDS | frozenset((
    "bestCandidate", "evaluatedCandidateCount", "invalidCandidateCount",
    "supportRejectedCandidateCount",
))
_FLATTENABLE_FIELDS = _numeric.flattenable_result_fields | _RESULT_FIELDS
_ERRORS = (KeyError, RuntimeError, TypeError, ValueError, OverflowError)


def _strict_payload(envelope, fields):
    if not isinstance(envelope, Mapping):
        raise ValueError("envelope must be a mapping")
    _numeric.check_identities(envelope)
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("payload does not match the published field set")
    if _FLATTENABLE_FIELDS.intersection(key for key in envelope if key != "payload"):
        raise ValueError("workload fields must not be flattened")
    if payload["supportPolicyID"] != SUPPORT_POLICY_ID:
        raise ValueError("supportPolicyID is invalid")
    if payload["morphologyFamilyID"] != MORPHOLOGY_FAMILY_ID:
        raise ValueError("morphologyFamilyID is invalid")
    if payload["modelClassID"] not in MODEL_CLASS_IDS:
        raise ValueError("modelClassID is invalid")
    return payload


def _range(payload, grid):
    if payload["modelClassID"] != grid.model_class_id:
        raise ValueError("modelClassID does not match dataset")
    start = _numeric.nonnegative_integer(payload["gridStartIndex"], "gridStartIndex")
    count = _numeric.positive_integer(payload["gridCount"], "gridCount")
    if start > MAX_SAFE_INTEGER - count or start + count > grid.total_candidates:
        raise ValueError("shard exceeds the configured grid")
    return start, count


def _has_support(validated, parameters):
    if validated.grid.model_class_id == POSITIVE_PULSE_ONLY:
        components = ((parameters["center"], parameters["logScale"], parameters["logShape"]),)
    else:
        positive_center = (
            parameters["negativeCenter"] + parameters["separation"]
            if validated.grid.model_class_id == ORDERED_NEGATIVE_POSITIVE_DOUBLET
            else parameters["positiveCenter"]
        )
        components = (
            (parameters["negativeCenter"], parameters["negativeLogScale"], parameters["negativeLogShape"]),
            (positive_center, parameters["positiveLogScale"], parameters["positiveLogShape"]),
        )
    for center, log_scale, log_shape in components:
        try:
            scale = math.exp(log_scale)
            shape = math.exp(log_shape)
            effective_width = scale * shape
            radius = 2.0 * effective_width
        except OverflowError:
            return False
        if not math.isfinite(center) or any(
            not math.isfinite(value) or value <= 0.0
            for value in (scale, shape, effective_width, radius)
        ):
            return False
        for series in validated.series:
            if not any(
                weight > 0.0 and abs(coordinate - center) <= radius
                for coordinate, weight in zip(series.coordinates, series.inverse_variances)
            ):
                return False
    return True


def _recompute(view, validated, start, count):
    invalid = rejected = 0
    best = None
    for index in range(start, start + count):
        evaluation = _numeric.evaluate_candidate(view, index)
        if evaluation is None:
            invalid += 1
        elif not _has_support(validated, evaluation.parameters):
            rejected += 1
        elif best is None or _numeric.candidate_precedes(evaluation, best):
            best = evaluation
    return invalid, rejected, best


def _verify_result(work_unit, result, dataset):
    """Return trusted counts and winner, sharing no worker-supplied arithmetic."""
    work = _strict_payload(work_unit, _WORK_FIELDS)
    returned = _strict_payload(result, _RESULT_FIELDS)
    if result.get("status") != "completed":
        raise ValueError("work unit did not complete")
    if any(returned[key] != work[key] for key in _WORK_FIELDS):
        raise ValueError("result shard identity or range differs from work")
    view, validated = _numeric.numerical_view(dataset)
    start, count = _range(work, validated.grid)
    _range(returned, validated.grid)
    evaluated = _numeric.positive_integer(returned["evaluatedCandidateCount"], "evaluatedCandidateCount")
    invalid = _numeric.nonnegative_integer(returned["invalidCandidateCount"], "invalidCandidateCount")
    rejected = _numeric.nonnegative_integer(returned["supportRejectedCandidateCount"], "supportRejectedCandidateCount")
    if evaluated != count or invalid + rejected > count:
        raise ValueError("candidate counts are inconsistent")
    expected_invalid, expected_rejected, best = _recompute(view, validated, start, count)
    if (invalid, rejected) != (expected_invalid, expected_rejected):
        raise ValueError("rejection counts failed full-shard recomputation")
    raw = returned["bestCandidate"]
    if best is None:
        if raw is not None:
            raise ValueError("a shard with zero eligible candidates requires a null winner")
    elif raw is None:
        raise ValueError("result omitted the canonical eligible winner")
    else:
        candidate = _numeric.strict_candidate_payload(raw, validated.grid.model_class_id)
        if not _numeric.candidate_payload_matches(candidate, best, validated.grid.model_class_id):
            raise ValueError("winner failed full-shard recomputation")
    return count, invalid, rejected, best


class SupportedMorphologyGridPlugin:
    uses_legacy_coordinator_diagnostics = False
    uses_legacy_science_metadata_validation = False
    definition = WorkloadDefinition(
        workload_id=WORKLOAD_ID, dataset_schema_id=DATASET_SCHEMA_ID,
        payload_schema_id=PAYLOAD_SCHEMA_ID, result_schema_id=RESULT_SCHEMA_ID,
        allows_legacy_schemaless_workers=False,
    )

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None:
        _numeric.numerical_view(dataset)

    def build_work_payloads(self, dataset: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
        _, validated = _numeric.numerical_view(dataset)
        grid = validated.grid
        for start in range(0, grid.total_candidates, grid.candidates_per_work_unit):
            yield {
                "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
                "modelClassID": grid.model_class_id,
                "supportPolicyID": SUPPORT_POLICY_ID,
                "gridStartIndex": start,
                "gridCount": min(grid.candidates_per_work_unit, grid.total_candidates - start),
            }

    def legacy_work_unit_fields(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return {}

    def canonicalize_result(self, work_unit, result):
        # Preserve public identities verbatim; validation rejects old results.
        return dict(result)

    def validate_result(self, work_unit, result, dataset) -> ResultValidation:
        try:
            _, invalid, rejected, best = _verify_result(work_unit, result, dataset)
        except _ERRORS as error:
            return ResultValidation(False, str(error), {"method": "supported-morphology-grid-invalid"})
        return ResultValidation(True, "Supported morphology-grid result accepted.", {
            "method": "supported-morphology-grid-full-shard-recomputation",
            "invalidCandidateCount": invalid,
            "supportRejectedCandidateCount": rejected,
            "bestGridIndex": best.grid_index if best else None,
        })

    def reduce_dataset(self, dataset, work_units: Sequence[Mapping[str, Any]],
                       results: Sequence[Mapping[str, Any] | None], terminal: bool) -> DatasetReduction:
        if len(work_units) != len(results):
            raise RuntimeError("reduction requires aligned work and result sequences")
        _, validated = _numeric.numerical_view(dataset)
        expected = list(self.build_work_payloads(dataset))
        completed = invalid = rejected = 0
        accepted = []
        winners = []
        for ordinal, (work, result) in enumerate(zip(work_units, results)):
            try:
                if ordinal >= len(expected) or dict(_strict_payload(work, _WORK_FIELDS)) != expected[ordinal]:
                    raise ValueError("work does not match exact ordered coverage")
                count, shard_invalid, shard_rejected, winner = _verify_result(work, result, dataset)
            except _ERRORS:
                accepted.append(False)
                continue
            accepted.append(True)
            completed += count
            invalid += shard_invalid
            rejected += shard_rejected
            if winner is not None:
                winners.append(winner)
        complete = bool(terminal and len(work_units) == len(expected) and work_units
                        and all(accepted) and completed == validated.grid.total_candidates)
        status = "SUPPORTED_MORPHOLOGY_GRID_COMPLETE" if complete else "SUPPORTED_MORPHOLOGY_GRID_INCOMPLETE"
        best = None
        for winner in sorted(winners, key=lambda item: item.grid_index):
            if best is None or _numeric.candidate_precedes(winner, best):
                best = winner
        payload = _numeric.candidate_payload(best, validated.grid.model_class_id) if best else None
        summary_keys = (
            "gridIndex", "parameters", "seriesFits", "positiveWeightSampleCount",
            "weightedResidualSumSquares", "nominalParameterCount", "bayesianInformationCriterion",
            "correctedAkaikeInformationCriterion", "correctedAkaikeInformationCriterionDefined",
        )
        return DatasetReduction(payload={"bestCandidate": payload}, status_fields={
            "workloadStatus": status, "supportedMorphologyGridStatus": status,
            "coverageComplete": complete, "supportPolicyID": SUPPORT_POLICY_ID,
            "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
            "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
            "modelClassID": validated.grid.model_class_id,
            "totalCandidateCount": validated.grid.total_candidates,
            "completedCandidateCount": completed,
            "totalInvalidCandidateCount": invalid,
            "totalSupportRejectedCandidateCount": rejected,
            "totalEligibleCandidateCount": completed - invalid - rejected,
            **{"best" + key[0].upper() + key[1:]: payload[key] if payload else None for key in summary_keys},
        })

    def contribution_metrics(self, work_unit, dataset) -> Mapping[str, Any]:
        work = _strict_payload(work_unit, _WORK_FIELDS)
        _, validated = _numeric.numerical_view(dataset)
        _, count = _range(work, validated.grid)
        samples = sum(len(series.coordinates) for series in validated.series)
        return {
            "workloadID": WORKLOAD_ID, "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
            "modelClassID": validated.grid.model_class_id, "supportPolicyID": SUPPORT_POLICY_ID,
            "seriesCount": len(validated.series), "sampleCount": samples, "candidateCount": count,
            "sampleCandidateEvaluations": _numeric.safe_product("sample-candidate evaluations", samples, count),
        }


PLUGIN = SupportedMorphologyGridPlugin()
