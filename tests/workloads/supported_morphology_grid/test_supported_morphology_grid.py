"""Small deterministic wire conformance cases; no coordinator or external data."""

import copy
import math
import unittest
from unittest.mock import patch

from openstar_workloads.plugins import morphology_grid as v1
from openstar_workloads.plugins import supported_morphology_grid as supported
from openstar_workloads.plugins.supported_morphology_grid import _adapter


def axis(start, step=1.0, count=1):
    return {"start": start, "step": step, "count": count}


def basis(x, center, log_scale=0.0, log_shape=0.0):
    scale = math.exp(log_scale)
    shape = math.exp(log_shape)
    z = (x - center) / scale
    u2 = shape * shape + z * z
    return (u2 + 2.0) / (math.sqrt(u2) * math.sqrt(u2 + 4.0))


def dataset(model=supported.POSITIVE_PULSE_ONLY):
    coordinates = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    values = [100.0 * basis(x, 0.0, math.log(0.25)) + 0.001 * x for x in coordinates]
    grid = {
        "centerAxis": axis(0.0, count=3),
        "logScaleAxis": axis(math.log(0.25)), "logShapeAxis": {"values": [0.0]},
    }
    if model != supported.POSITIVE_PULSE_ONLY:
        values = [0.5 - 1.5 * basis(x, -1.0) + 2.5 * basis(x, 1.0) + 0.001 * x for x in coordinates]
        grid = {f"{sign}Log{kind}Axis": axis(0.0) for sign in ("negative", "positive") for kind in ("Scale", "Shape")}
        if model == supported.ORDERED_NEGATIVE_POSITIVE_DOUBLET:
            grid.update({"negativeCenterAxis": axis(-1.0), "separationAxis": axis(2.0)})
        else:
            grid["centerAxis"] = axis(-1.0, 1.0, 3)
    return {
        "id": "supported-fixture", "datasetSchemaID": supported.DATASET_SCHEMA_ID,
        "morphologyFamilyID": supported.MORPHOLOGY_FAMILY_ID,
        "componentTemplateFamilyID": supported.COMPONENT_TEMPLATE_FAMILY_ID,
        "executionContractID": supported.EXECUTION_CONTRACT_ID,
        "executionContractVersion": supported.EXECUTION_CONTRACT_VERSION,
        "supportPolicyID": supported.SUPPORT_POLICY_ID,
        "modelClassID": model, "morphologyGrid": grid, "candidatesPerWorkUnit": 2,
        "series": [{"genericSeriesID": "series-001", "coordinates": coordinates,
                    "values": values, "inverseVariances": [1.0] * len(coordinates)}],
    }


def reference_supported(document, parameters):
    """Independent direct wire-formula oracle, without the production support helper."""
    model = document["modelClassID"]
    if model == supported.POSITIVE_PULSE_ONLY:
        components = [(parameters["center"], parameters["logScale"], parameters["logShape"])]
    else:
        positive = parameters.get("positiveCenter")
        if model == supported.ORDERED_NEGATIVE_POSITIVE_DOUBLET:
            positive = parameters["negativeCenter"] + parameters["separation"]
        components = [(parameters["negativeCenter"], parameters["negativeLogScale"], parameters["negativeLogShape"]),
                      (positive, parameters["positiveLogScale"], parameters["positiveLogShape"])]
    return all(any(w > 0 and abs(x - center) <= 2.0 * (math.exp(ls) * math.exp(lh))
                   for x, w in zip(series["coordinates"], series["inverseVariances"]))
               for center, ls, lh in components for series in document["series"])


def result_for(document, work):
    numerical = copy.deepcopy(document)
    numerical.update({"datasetSchemaID": v1.DATASET_SCHEMA_ID, "executionContractID": v1.EXECUTION_CONTRACT_ID})
    payload = work["payload"]
    invalid = rejected = 0
    winner = None
    for index in range(payload["gridStartIndex"], payload["gridStartIndex"] + payload["gridCount"]):
        candidate = v1._evaluate_candidate(numerical, index)
        if candidate is None:
            invalid += 1
        elif not reference_supported(document, candidate.parameters):
            rejected += 1
        elif winner is None or v1._candidate_precedes(candidate, winner):
            winner = candidate
    return {"status": "completed", "workloadID": supported.WORKLOAD_ID,
            "resultSchemaID": supported.RESULT_SCHEMA_ID, "payload": {
                **payload, "evaluatedCandidateCount": payload["gridCount"],
                "invalidCandidateCount": invalid, "supportRejectedCandidateCount": rejected,
                "bestCandidate": v1._candidate_payload(winner, document["modelClassID"]) if winner else None,
            }}


def work_for(document):
    return [{"workloadID": supported.WORKLOAD_ID, "payloadSchemaID": supported.PAYLOAD_SCHEMA_ID,
             "payload": payload} for payload in supported.PLUGIN.build_work_payloads(document)]


class SupportedMorphologyGridTests(unittest.TestCase):
    def assert_accepted(self, document, work, result):
        validation = supported.PLUGIN.validate_result(work, result, document)
        self.assertTrue(validation.accepted, validation.message)

    def test_unsupported_numerical_winner_loses_to_supported_candidate(self):
        document = dataset()
        document["morphologyGrid"]["centerAxis"]["count"] = 2
        work = work_for(document)[0]
        view, _ = _adapter.numerical_view(document)
        invalid, original = v1._recompute_shard(view, 0, 2)
        self.assertEqual(invalid, 0)
        self.assertEqual(original.grid_index, 0)
        result = result_for(document, work)
        self.assertEqual(result["payload"]["bestCandidate"]["gridIndex"], 1)
        self.assertEqual(result["payload"]["supportRejectedCandidateCount"], 1)
        self.assert_accepted(document, work, result)
        result["payload"]["bestCandidate"] = v1._candidate_payload(original, document["modelClassID"])
        self.assertFalse(supported.PLUGIN.validate_result(work, result, document).accepted)

    def test_all_partial_and_zero_support_shards(self):
        for start, count, rejected in ((1.0, 2, 0), (0.0, 2, 1), (0.0, 1, 1)):
            with self.subTest(start=start, count=count):
                document = dataset()
                document["morphologyGrid"]["centerAxis"] = axis(start, count=count)
                work = work_for(document)[0]
                result = result_for(document, work)
                self.assertEqual(result["payload"]["supportRejectedCandidateCount"], rejected)
                self.assertEqual(result["payload"]["invalidCandidateCount"], 0)
                self.assertEqual(result["payload"]["evaluatedCandidateCount"], count)
                self.assertEqual(result["payload"]["bestCandidate"] is None, rejected == count)
                self.assert_accepted(document, work, result)

    def test_mixed_numerical_invalid_and_support_rejected_are_exclusive(self):
        document = dataset()
        document["morphologyGrid"].update({"logScaleAxis": axis(0.0), "logShapeAxis": {"values": [-400.0, math.log(0.25)]}})
        document["candidatesPerWorkUnit"] = 6
        work = work_for(document)[0]
        result = result_for(document, work)
        self.assertEqual((result["payload"]["evaluatedCandidateCount"], result["payload"]["invalidCandidateCount"],
                          result["payload"]["supportRejectedCandidateCount"]), (6, 2, 2))
        self.assert_accepted(document, work, result)
        reduction = supported.PLUGIN.reduce_dataset(document, [work], [result], True)
        self.assertEqual(reduction.status_fields["totalEligibleCandidateCount"], 2)
        self.assertEqual(reduction.status_fields["totalInvalidCandidateCount"], 2)
        self.assertEqual(reduction.status_fields["totalSupportRejectedCandidateCount"], 2)
        self.assertTrue(reduction.status_fields["coverageComplete"])

    def test_inclusive_boundary_next_float_and_zero_weight(self):
        document = dataset()
        document["morphologyGrid"].update({"centerAxis": axis(0.0), "logScaleAxis": axis(0.0)})
        series = document["series"][0]
        series.update({"coordinates": [2.0, 3.0, 4.0], "values": [2.0, 1.0, 0.5], "inverseVariances": [1.0, 1.0, 1.0]})
        for boundary, weight, expected in ((2.0, 1.0, 0), (math.nextafter(2.0, math.inf), 1.0, 1), (2.0, 0.0, 1)):
            with self.subTest(boundary=boundary, weight=weight):
                series["coordinates"][0] = boundary
                series["inverseVariances"][0] = weight
                work = work_for(document)[0]
                result = result_for(document, work)
                self.assertEqual(result["payload"]["invalidCandidateCount"], 0)
                self.assertEqual(result["payload"]["supportRejectedCandidateCount"], expected)
                self.assert_accepted(document, work, result)

    def test_all_models_and_independent_center_order(self):
        for model in supported.MODEL_CLASS_IDS:
            with self.subTest(model=model):
                document = dataset(model)
                units = work_for(document)
                results = [result_for(document, work) for work in units]
                for work, result in zip(units, results):
                    self.assert_accepted(document, work, result)
                    best = result["payload"]["bestCandidate"]
                    if model == supported.INDEPENDENT_PULSES and best:
                        self.assertLess(best["parameters"]["negativeCenter"], best["parameters"]["positiveCenter"])
                reduced = supported.PLUGIN.reduce_dataset(document, units, results, True)
                self.assertTrue(reduced.status_fields["coverageComplete"])
                self.assertIsNotNone(reduced.payload["bestCandidate"])
        self.assertEqual(supported.independent_center_pair_indices(3, 1), (0, 2))
        self.assertEqual(supported.candidate_index((1, 2), (3, 4)), 6)
        self.assertEqual(supported.candidate_indices(6, (3, 4)), (1, 2))

    def test_ordered_positive_center_and_every_series_every_component(self):
        document = dataset(supported.ORDERED_NEGATIVE_POSITIVE_DOUBLET)
        document["morphologyGrid"].update({
            "negativeLogScaleAxis": axis(math.log(0.25)),
            "positiveLogScaleAxis": axis(math.log(0.25)),
        })
        second = copy.deepcopy(document["series"][0])
        second["genericSeriesID"] = "series-002"
        # Only the derived positive center (+1) loses support in the second series.
        second["inverseVariances"][3] = 0.0
        document["series"].append(second)
        work = work_for(document)[0]
        result = result_for(document, work)
        self.assertEqual(result["payload"]["invalidCandidateCount"], 0)
        self.assertEqual(result["payload"]["supportRejectedCandidateCount"], 1)
        self.assert_accepted(document, work, result)
        second["inverseVariances"][3] = 1.0
        result = result_for(document, work)
        self.assertEqual(result["payload"]["supportRejectedCandidateCount"], 0)
        self.assert_accepted(document, work, result)
        # Independent searches contain only their own series; no external scope.
        independent = dataset(supported.INDEPENDENT_PULSES)
        independent["series"].append(second)
        with self.assertRaisesRegex(RuntimeError, "exactly one series"):
            supported.PLUGIN.validate_dataset(independent)

    def test_zero_amplitude_still_requires_support_and_stable_ties(self):
        document = dataset()
        document["series"][0]["values"] = [0.0] * 6
        units = work_for(document)
        results = [result_for(document, work) for work in units]
        reduced = supported.PLUGIN.reduce_dataset(document, units, results, True)
        self.assertEqual(reduced.status_fields["bestGridIndex"], 1)
        self.assertEqual(reduced.status_fields["totalSupportRejectedCandidateCount"], 1)
        self.assertEqual(reduced.payload["bestCandidate"]["seriesFits"][0]["positiveAmplitude"], 0.0)

    def test_partial_shard_coverage_and_no_eligible_complete(self):
        document = dataset()
        units = work_for(document)
        self.assertEqual([work["payload"]["gridCount"] for work in units], [2, 1])
        results = [result_for(document, work) for work in units]
        for work, result in zip(units, results):
            self.assert_accepted(document, work, result)
        for work_set, result_set, terminal in (
            (units, results, False), (units, [results[0], None], True),
            (units[:1], results[:1], True), (units[::-1], results[::-1], True),
            (units + units[:1], results + results[:1], True),
        ):
            self.assertFalse(supported.PLUGIN.reduce_dataset(document, work_set, result_set, terminal).status_fields["coverageComplete"])
        document["morphologyGrid"]["centerAxis"] = axis(0.0)
        units = work_for(document)
        results = [result_for(document, work) for work in units]
        reduced = supported.PLUGIN.reduce_dataset(document, units, results, True)
        self.assertEqual(reduced.status_fields["workloadStatus"], "SUPPORTED_MORPHOLOGY_GRID_COMPLETE")
        self.assertEqual(reduced.status_fields["supportedMorphologyGridStatus"], "SUPPORTED_MORPHOLOGY_GRID_COMPLETE")
        self.assertTrue(reduced.status_fields["coverageComplete"])
        self.assertEqual(reduced.status_fields["totalEligibleCandidateCount"], 0)
        self.assertIsNone(reduced.payload["bestCandidate"])
        self.assertTrue(all(value is None for key, value in reduced.status_fields.items() if key.startswith("best")))

    def test_all_numerically_invalid_shard_is_successful_with_null_winner(self):
        document = dataset()
        document["morphologyGrid"].update({
            "centerAxis": axis(1.0), "logScaleAxis": axis(0.0),
            "logShapeAxis": {"values": [-400.0]},
        })
        work = work_for(document)[0]
        result = result_for(document, work)
        self.assertEqual(result["payload"]["invalidCandidateCount"], 1)
        self.assertEqual(result["payload"]["supportRejectedCandidateCount"], 0)
        self.assertIsNone(result["payload"]["bestCandidate"])
        self.assert_accepted(document, work, result)
        reduced = supported.PLUGIN.reduce_dataset(document, [work], [result], True)
        self.assertTrue(reduced.status_fields["coverageComplete"])
        self.assertEqual(reduced.status_fields["totalEligibleCandidateCount"], 0)

    def test_support_geometry_is_finite_positive_and_uses_separate_exponentials(self):
        document = dataset()
        _, validated = _adapter.numerical_view(document)
        parameters = {"center": 1.0, "logScale": 0.0, "logShape": 0.0}
        exp = math.exp
        with patch.object(supported.math, "exp", wraps=exp) as exponentials:
            self.assertTrue(supported._has_support(validated, parameters))
        self.assertEqual([call.args for call in exponentials.call_args_list], [(0.0,), (0.0,)])
        for log_scale, log_shape in ((800.0, -800.0), (-400.0, -400.0), (400.0, 400.0)):
            with self.subTest(log_scale=log_scale, log_shape=log_shape):
                parameters.update({"logScale": log_scale, "logShape": log_shape})
                self.assertFalse(supported._has_support(validated, parameters))

    def test_fabricated_counts_ranges_candidates_and_nulls(self):
        document = dataset()
        work = work_for(document)[0]
        original = result_for(document, work)
        mutations = {
            "evaluatedCandidateCount": 1, "invalidCandidateCount": 1,
            "supportRejectedCandidateCount": 0, "gridStartIndex": 1,
            "gridCount": 1, "bestCandidate": None,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                result = copy.deepcopy(original)
                result["payload"][key] = value
                self.assertFalse(supported.PLUGIN.validate_result(work, result, document).accepted)
        for key in ("evaluatedCandidateCount", "invalidCandidateCount", "supportRejectedCandidateCount", "gridStartIndex", "gridCount"):
            result = copy.deepcopy(original)
            result["payload"][key] = True
            self.assertFalse(supported.PLUGIN.validate_result(work, result, document).accepted)
        result = copy.deepcopy(original)
        result["payload"]["bestCandidate"]["gridIndex"] = 2
        self.assertFalse(supported.PLUGIN.validate_result(work, result, document).accepted)
        result = copy.deepcopy(original)
        result["payload"]["bestCandidate"]["parameters"]["extra"] = 1
        self.assertFalse(supported.PLUGIN.validate_result(work, result, document).accepted)
        no_support = dataset()
        no_support["morphologyGrid"]["centerAxis"] = axis(0.0)
        empty_work = work_for(no_support)[0]
        empty_result = result_for(no_support, empty_work)
        empty_result["payload"]["bestCandidate"] = original["payload"]["bestCandidate"]
        self.assertFalse(supported.PLUGIN.validate_result(empty_work, empty_result, no_support).accepted)

    def test_identity_rejection_precedes_internal_view_and_never_relabels_v1(self):
        document = dataset()
        work = work_for(document)[0]
        result = result_for(document, work)
        for field in ("datasetSchemaID", "executionContractID", "executionContractVersion", "supportPolicyID",
                      "morphologyFamilyID", "componentTemplateFamilyID", "workloadID", "payloadSchemaID", "resultSchemaID"):
            bad = copy.deepcopy(document)
            bad[field] = "wrong"
            with self.subTest(field=field), patch.object(_adapter._v1, "_validated_dataset", side_effect=AssertionError("must validate identities first")):
                with self.assertRaises(RuntimeError):
                    supported.PLUGIN.validate_dataset(bad)
        for policy in (None, 1, "unknown", ""):
            bad = copy.deepcopy(document)
            bad["supportPolicyID"] = policy
            with self.assertRaises(RuntimeError):
                supported.PLUGIN.validate_dataset(bad)
        del document["supportPolicyID"]
        with self.assertRaises(RuntimeError):
            supported.PLUGIN.validate_dataset(document)
        document = dataset()
        for target, field, value in (("work", "workloadID", v1.WORKLOAD_ID),
                                     ("result", "resultSchemaID", v1.RESULT_SCHEMA_ID),
                                     ("payload", "supportPolicyID", "wrong")):
            bad_work, bad_result = copy.deepcopy(work), copy.deepcopy(result)
            mapping = bad_work if target == "work" else bad_result if target == "result" else bad_result["payload"]
            mapping[field] = value
            with patch.object(_adapter, "numerical_view", side_effect=AssertionError("identity first")):
                self.assertFalse(supported.PLUGIN.validate_result(bad_work, bad_result, document).accepted)
            self.assertEqual(supported.PLUGIN.canonicalize_result(bad_work, bad_result), bad_result)
        # An actual old payload lacks both required support fields.
        old = copy.deepcopy(result)
        del old["payload"]["supportPolicyID"]
        del old["payload"]["supportRejectedCandidateCount"]
        self.assertFalse(supported.PLUGIN.validate_result(work, old, document).accepted)

    def test_top_level_extensibility_strict_nested_fields_and_accounting(self):
        document = dataset()
        document["opaqueMetadata"] = {"arbitrary": True}
        supported.PLUGIN.validate_dataset(document)
        units = work_for(document)
        metrics = supported.PLUGIN.contribution_metrics(units[-1], document)
        self.assertEqual((metrics["candidateCount"], metrics["sampleCandidateEvaluations"]), (1, 6))
        self.assertEqual(metrics["supportPolicyID"], supported.SUPPORT_POLICY_ID)
        document["series"][0]["extra"] = 1
        with self.assertRaises(RuntimeError):
            supported.PLUGIN.validate_dataset(document)


if __name__ == "__main__":
    unittest.main()
