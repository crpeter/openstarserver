"""Small producer-shaped offline artifacts; never a coordinator or grid search."""

import copy
import math
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from openstar_workloads.plugins import morphology_grid as v1
from openstar_workloads.plugins import supported_morphology_grid as workload
from openstar_workloads.plugins.supported_morphology_grid import _adapter as numerical
from tests.workflows.microlensing import test_validate_anomaly_morphology_coarse_grid as fixtures
from tests.workflows.microlensing.test_refine_grid import stage_record
from workflows.microlensing import build_supported_anomaly_morphology_grid as builder
from workflows.microlensing import validate_supported_anomaly_morphology_grid as validation


def set_winner(status, winner):
    status["payload"] = {"bestCandidate": winner}
    for key in validation._BEST_KEYS:
        status["best" + key[0].upper() + key[1:]] = winner[key] if winner else None


def supported_investigation(root, project_root):
    project_path = project_root / "project.json"
    project = fixtures.read_json(project_path)
    manifest = fixtures.read_json(project_root / "build-manifest.json")
    project_hash = fixtures.sha256_bytes(project_path.read_bytes())
    total = manifest["totalExpectedWorkUnitCount"]
    statuses = []
    for reference, record in zip(project["datasets"], manifest["datasets"]):
        dataset = fixtures.read_json(project_root / reference["path"])
        view, _ = numerical.numerical_view(dataset)
        evaluation = numerical.evaluate_candidate(view, 0)
        if evaluation is None:
            raise AssertionError("accepted fixture candidate must be numerically valid")
        best = numerical.candidate_payload(evaluation, dataset["modelClassID"])
        count, work_count = record["candidateCount"], record["expectedWorkUnitCount"]
        status = {
            "id": dataset["id"], "workloadID": workload.WORKLOAD_ID,
            "datasetSchemaID": workload.DATASET_SCHEMA_ID, "payloadSchemaID": workload.PAYLOAD_SCHEMA_ID,
            "resultSchemaID": workload.RESULT_SCHEMA_ID, "supportPolicyID": workload.SUPPORT_POLICY_ID,
            "morphologyFamilyID": workload.MORPHOLOGY_FAMILY_ID,
            "componentTemplateFamilyID": workload.COMPONENT_TEMPLATE_FAMILY_ID,
            "modelClassID": dataset["modelClassID"],
            "workloadStatus": "SUPPORTED_MORPHOLOGY_GRID_COMPLETE",
            "supportedMorphologyGridStatus": "SUPPORTED_MORPHOLOGY_GRID_COMPLETE", "coverageComplete": True,
            "completedCandidateCount": count, "totalCandidateCount": count,
            "totalInvalidCandidateCount": 0, "totalSupportRejectedCandidateCount": count - 1,
            "totalEligibleCandidateCount": 1,
            "assignedWorkUnits": 0, "pendingWorkUnits": 0, "failedWorkUnits": 0,
            "completedWorkUnits": work_count, "totalWorkUnits": work_count,
            "nodeContributions": {"generic-node": work_count}, "progress": 1.0,
        }
        # Saved summary counters are internally consistent fixture observations.
        # They intentionally make no independently reproduced full-search claim.
        set_winner(status, best)
        statuses.append(status)
    counters = ("assignedWorkUnits", "pendingWorkUnits", "completedWorkUnits", "failedWorkUnits", "totalWorkUnits")
    run = {
        **{key: statuses[-1][key] for key in counters},
        "projectAssignedWorkUnits": 0, "projectPendingWorkUnits": 0, "projectFailedWorkUnits": 0,
        "projectCompletedWorkUnits": total, "projectTotalWorkUnits": total, "projectProgress": 1.0,
        "projectID": project["id"], "projectPath": str(project_path), "status": "COMPLETE",
        "workloadID": workload.WORKLOAD_ID, "nodeContributions": {"generic-node": total}, "datasets": statuses,
    }
    ids = ("001-prepare-project", "002-distributed-project", "003-terminal-check")
    handlers = ("local.project.prepare", "openstar.project.run", "generic.project.terminal-check")
    parameters = ({"projectPath": str(project_path)}, {"projectPath": str(project_path), "projectManifestSha256": project_hash}, {"expectedProjectID": project["id"]})
    results = (parameters[1], run, {"completedWorkUnits": total, "failedWorkUnits": 0, "passed": True,
                                  "projectID": project["id"], "totalWorkUnits": total,
                                  "rule": "projectID matches and completed+failed == total"})
    stages = [stage_record(
        stage_id=ids[index], handler_id=handlers[index], triggered_by=ids[index - 1] if index else None,
        parameters=parameters[index], result=results[index],
        input_hashes={"projectManifest": project_hash} if index < 2 else {}, project_ids=[project["id"]] if index else [],
        next_stage={"handler_id": handlers[index + 1], "id": ids[index + 1], "parameters": parameters[index + 1],
                    "triggered_by_stage_id": ids[index]} if index < 2 else None,
        stop=index == 2, node_contributions={"generic-node": total} if index == 1 else {},
    ) for index in range(3)]
    identity = "generic-supported-investigation"
    path = root / identity / "investigation.json"
    fixtures.write_record(path, {
        "id": identity, "created_at": "2026-09-07T00:00:00+00:00", "updated_at": "2026-09-07T00:00:03+00:00",
        "stages": stages, "status": "COMPLETE", "workflow_id": "openstar.workflow.project-smoke.v1",
        "workflow_version": "20.0", "metadata": {"coordinator": "http://127.0.0.1:8080", "projectPath": str(project_path)},
    })
    return path


class SupportedFixture(fixtures.MorphologyValidationFixture):
    supported_series = (0, 1)

    def _axes(self, **kwargs):
        axes = super()._axes(center_count=3, log_scale_count=1)
        axes["SEPARATION"]["count"] = 2
        return axes

    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        document["residualValues"] = [0.01 * x * x + 0.003 * x * x * x for x in document["coordinates"]]
        return document

    def setUp(self):
        super().setUp()
        self.report()
        self.coarse_validation = self.root / "report"
        self.supported_root = self.root / "supported-project"
        builder.build_supported_anomaly_morphology_grid(
            self.morphology, coarse_project_root=self.coarse, coarse_investigation_record=self.investigation,
            coarse_validation_root=self.coarse_validation, project_id="generic-supported-project", output_root=self.supported_root,
        )
        self.supported_record = supported_investigation(self.root, self.supported_root)

    def validate(self, name="validated", **overrides):
        arguments = {
            "coarse_project_root": self.coarse, "coarse_investigation_record": self.investigation,
            "coarse_validation_root": self.coarse_validation, "supported_project_root": self.supported_root,
            "supported_investigation_record": self.supported_record, "output_root": self.root / name,
        }
        arguments.update(overrides)
        return validation.validate_supported_anomaly_morphology_grid(self.morphology, **arguments)

    def reject_supported(self, pattern, **overrides):
        with self.assertRaisesRegex(validation.SupportedAnomalyMorphologyGridValidationError, pattern):
            self.validate("rejected", **overrides)
        self.assertFalse((self.root / "rejected").exists())

    def rewrite(self, mutate):
        record = fixtures.read_json(self.supported_record)
        mutate(record)
        fixtures.write_record(self.supported_record, record)


class SupportedValidationTests(SupportedFixture):
    def test_deterministic_success_untouched_inputs_and_no_search(self):
        roots = (self.morphology, self.coarse, self.investigation.parent, self.coarse_validation,
                 self.supported_root, self.supported_record.parent)
        before = {root: fixtures.serialized_tree(root) for root in roots}
        original_evaluate, supported_evaluate = v1._evaluate_candidate, numerical.evaluate_candidate
        with ExitStack() as stack:
            coarse_calls = stack.enter_context(patch.object(v1, "_evaluate_candidate", wraps=original_evaluate))
            supported_calls = stack.enter_context(patch.object(numerical, "evaluate_candidate", wraps=supported_evaluate))
            for owner, name in ((v1, "_recompute_shard"), (workload, "_recompute"),
                                (v1.PLUGIN, "build_work_payloads"), (workload.PLUGIN, "build_work_payloads"),
                                (workload.PLUGIN, "validate_result"), (workload.PLUGIN, "reduce_dataset"),
                                (builder, "build_supported_anomaly_morphology_grid")):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("no searches or builders during validation")))
            result = self.validate()
        self.assertEqual(coarse_calls.call_count, 4)
        self.assertEqual(supported_calls.call_count, 4)
        self.assertEqual({root: fixtures.serialized_tree(root) for root in roots}, before)
        self.validate("repeat")
        self.assertEqual(fixtures.serialized_tree(self.root / "validated"), fixtures.serialized_tree(self.root / "repeat"))
        report = result["result"]
        self.assertEqual(len(report["searches"]), 4)
        self.assertTrue(report["coverageComplete"])
        self.assertFalse(report["planetaryInterpretationResolved"])
        self.assertFalse(report["discoveryClaim"])
        self.assertFalse(report["verificationScope"]["globalOptimalityIndependentlyProven"])
        self.assertFalse(report["verificationScope"]["rejectionCountsIndependentlyReproduced"])
        for search in report["searches"]:
            self.assertTrue(search["supportRequirementMet"])
            self.assertGreater(search["acceptedWinner"]["weightedResidualSumSquares"], 0.0)
        manifest = result["artifactManifest"]
        for path, digest in manifest["outputSHA256s"].items():
            self.assertEqual(digest, fixtures.sha256_bytes((self.root / "validated" / path).read_bytes()))
        self.assertEqual(set(fixtures.serialized_tree(self.root / "validated")), {
            validation.RESULT_RELATIVE_PATH, validation.MARKDOWN_RELATIVE_PATH, validation.MANIFEST_RELATIVE_PATH,
        })

    def test_only_canonical_dataset_summaries_are_collected(self):
        self.rewrite(lambda r: r["stages"][1]["result"].update({"payload": {"best": {"id": "not-a-dataset", "payload": {"bestCandidate": None}}}}))
        report = self.validate()["result"]
        self.assertEqual(len(report["searches"]), 4)
        run = fixtures.read_json(self.supported_record)["stages"][1]["result"]
        self.assertNotEqual(run["completedWorkUnits"], run["projectCompletedWorkUnits"])
        self.assertEqual(report["projectCompletedWorkUnits"], run["projectCompletedWorkUnits"])

    def test_complete_null_search_is_valid_and_unresolved(self):
        def nullify(record):
            status = record["stages"][1]["result"]["datasets"][2]
            status["totalSupportRejectedCandidateCount"] = status["totalCandidateCount"]
            status["totalEligibleCandidateCount"] = 0
            set_winner(status, None)
        self.rewrite(nullify)
        report = self.validate()["result"]
        self.assertEqual(report["completeNullWinnerSearchCount"], 1)
        self.assertIsNone(report["independentAggregate"])
        self.assertIsNone(report["modelPreference"])
        self.assertEqual(report["overallClassification"], "UNRESOLVED_NO_ELIGIBLE_CANDIDATES")
        self.assertTrue(report["coverageComplete"])
        self.assertFalse(report["modelComparisons"]["rejectOrderedDoubletForIndependentPulses"]["gateDetails"]["globalDeltaWRSSPassed"]["evaluated"])

    def test_counter_types_bounds_coverage_and_duplicate_winner_fields(self):
        original = fixtures.read_json(self.supported_record)
        mutations = (
            ("totalInvalidCandidateCount", True), ("totalInvalidCandidateCount", -1),
            ("totalSupportRejectedCandidateCount", 10**20), ("totalEligibleCandidateCount", 0),
            ("completedCandidateCount", 0), ("completedCandidateCount", 9.0),
            ("completedWorkUnits", 0), ("failedWorkUnits", 1), ("pendingWorkUnits", 1), ("assignedWorkUnits", 1),
            ("bestGridIndex", 1), ("coverageComplete", 1),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                record = copy.deepcopy(original)
                record["stages"][1]["result"]["datasets"][0][key] = value
                fixtures.write_record(self.supported_record, record)
                self.reject_supported("candidate|supported dataset|duplicated winner")
        fixtures.write_record(self.supported_record, original)
        self.rewrite(lambda r: r["stages"][1]["result"].update({"projectCompletedWorkUnits": 1}))
        self.reject_supported("project counters")

    def test_supported_identities_order_duplicate_and_unexpected_records(self):
        original = fixtures.read_json(self.supported_record)
        for field, value in (("workloadID", v1.WORKLOAD_ID), ("supportPolicyID", "unknown"),
                             ("datasetSchemaID", None), ("resultSchemaID", v1.RESULT_SCHEMA_ID)):
            record = copy.deepcopy(original)
            record["stages"][1]["result"]["datasets"][0][field] = value
            fixtures.write_record(self.supported_record, record)
            self.reject_supported(f"{field} is invalid")
        for vector in ([0, 1, 2], [0, 1, 2, 3, 3], [1, 0, 2, 3], [0, 0, 2, 3]):
            record = copy.deepcopy(original)
            statuses = record["stages"][1]["result"]["datasets"]
            record["stages"][1]["result"]["datasets"] = [statuses[i] for i in vector]
            fixtures.write_record(self.supported_record, record)
            self.reject_supported("four canonical dataset|dataset mapping and order")

    def test_grid_index_parameter_fit_and_null_mismatches(self):
        original = fixtures.read_json(self.supported_record)
        for kind in ("index", "parameters", "fit", "bic", "aicc", "null"):
            record = copy.deepcopy(original)
            status = record["stages"][1]["result"]["datasets"][0]
            winner = status["payload"]["bestCandidate"]
            if kind == "index":
                winner["gridIndex"] = status["totalCandidateCount"]
            elif kind == "parameters":
                winner["parameters"]["center"] += 0.1
            elif kind == "fit":
                winner["seriesFits"][0]["offset"] += 0.1
            elif kind == "bic":
                winner["bayesianInformationCriterion"] += 1.0
            elif kind == "aicc":
                winner["correctedAkaikeInformationCriterionDefined"] = not winner["correctedAkaikeInformationCriterionDefined"]
            else:
                winner = None
            set_winner(status, winner)
            fixtures.write_record(self.supported_record, record)
            self.reject_supported("accepted grid index|winner parameter mapping|winner does not reproduce|null winner")

    def test_stage_hash_ledger_order_terminal_and_contributions(self):
        original = fixtures.read_json(self.supported_record)
        stale = copy.deepcopy(original)
        stale["stages"][1]["provenance"]["result_hash"] = "0" * 64
        fixtures.write_json(self.supported_record, stale)
        self.reject_supported("stage result hash does not match")
        fixtures.write_record(self.supported_record, original)
        ledger = self.supported_record.parent / "stages" / f"{original['stages'][1]['id']}.json"
        data = fixtures.read_json(ledger)
        data["result"]["projectTotalWorkUnits"] += 1
        fixtures.write_json(ledger, data)
        self.reject_supported("stage ledger .* does not match")
        fixtures.write_record(self.supported_record, original)
        self.rewrite(lambda r: r["stages"].reverse())
        self.reject_supported("stage order")
        fixtures.write_record(self.supported_record, original)
        self.rewrite(lambda r: r["stages"][2]["result"].update({"passed": False}))
        self.reject_supported("supported terminal result")
        fixtures.write_record(self.supported_record, original)
        self.rewrite(lambda r: r["stages"][1]["result"]["datasets"][0].update({"nodeContributions": {"generic-node": 0}}))
        self.reject_supported("dataset contributions")

    def test_canonical_json_rejection_separate_from_manifest_hash_mismatch(self):
        report_path = self.coarse_validation / fixtures.validation.RESULT_RELATIVE_PATH
        original_report = report_path.read_bytes()
        report_path.write_bytes(original_report + b"\n")
        self.reject_supported("^validation report is not canonical stable JSON$")
        report_path.write_bytes(original_report)
        manifest_path = self.coarse_validation / fixtures.validation.MANIFEST_RELATIVE_PATH
        manifest = fixtures.read_json(manifest_path)
        digest = manifest["outputSHA256s"][fixtures.validation.RESULT_RELATIVE_PATH]
        manifest["outputSHA256s"][fixtures.validation.RESULT_RELATIVE_PATH] = ("1" if digest[0] == "0" else "0") + digest[1:]
        fixtures.write_json(manifest_path, manifest)
        self.reject_supported("^validation manifest does not reconstruct exactly$")

    def test_supported_hash_provenance_arrays_and_schema_tampering(self):
        manifest_path = self.supported_root / "build-manifest.json"
        original = fixtures.read_json(manifest_path)
        for key in ("buildManifestSHA256", "supportPolicyID", "totalCandidateCount"):
            manifest = copy.deepcopy(original)
            manifest[key] = "0" * 64 if key != "totalCandidateCount" else 0
            fixtures.write_json(manifest_path, manifest)
            self.reject_supported("supported build manifest")
        fixtures.write_json(manifest_path, original)
        project = fixtures.read_json(self.supported_root / "project.json")
        dataset_path = self.supported_root / project["datasets"][0]["path"]
        dataset = fixtures.read_json(dataset_path)
        dataset["series"][0]["values"][0] += 0.1
        fixtures.write_json(dataset_path, dataset)
        self.reject_supported("supported dataset domain and provenance")

    def test_boundary_reporting_distinguishes_fixed_axes(self):
        report = self.validate()["result"]
        axes = {axis["axis"]: axis for axis in report["searches"][0]["axes"]}
        self.assertTrue(axes["center"]["searchedBoundary"])
        self.assertTrue(axes["logScale"]["fixed"])
        self.assertFalse(axes["logScale"]["searchedBoundary"])
        self.assertEqual(axes["logScale"]["position"], "FIXED")
        self.assertTrue(report["widthInterpretationLimitedByBoundary"])

    def test_transactional_failure_paths_and_existing_output(self):
        before = fixtures.serialized_tree(self.supported_root)
        with patch.object(validation, "_atomic_write_bytes", side_effect=OSError("injected publication failure")):
            self.reject_supported("injected publication failure")
        self.assertEqual(list(self.root.glob(".rejected.*")), [])
        self.assertEqual(fixtures.serialized_tree(self.supported_root), before)
        self.reject_supported("inside an input", output_root=self.supported_root / "child")
        linked = self.root / "linked-supported"
        linked.symlink_to(self.supported_root, target_is_directory=True)
        self.reject_supported("symlink", supported_project_root=linked)
        self.validate()
        outputs = fixtures.serialized_tree(self.root / "validated")
        with self.assertRaisesRegex(validation.SupportedAnomalyMorphologyGridValidationError, "output root already exists"):
            self.validate()
        self.assertEqual(fixtures.serialized_tree(self.root / "validated"), outputs)


class UnsupportedWinnerTests(SupportedFixture):
    supported_series = ()

    def test_numerically_valid_but_unsupported_accepted_winner_is_rejected(self):
        self.reject_supported("accepted supported winner lacks observational support")


class ZeroAmplitudeTests(SupportedFixture):
    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        document["residualValues"] = [0.0] * len(document["coordinates"])
        return document

    def test_zero_negative_amplitude_fails_sign_gate_despite_support(self):
        report = self.validate()["result"]
        ordered = report["searches"][1]
        self.assertTrue(ordered["supportRequirementMet"])
        self.assertTrue(all(fit["negativeAmplitude"] == 0.0 for fit in ordered["acceptedWinner"]["seriesFits"]))
        details = report["modelComparisons"]["preferOrderedDoubletOverPositivePulse"]["gateDetails"]
        self.assertTrue(details["supportRequirementsPassed"]["passed"])
        self.assertFalse(details["signRequirementsPassed"]["passed"])
        self.assertTrue(details["signRequirementsPassed"]["evaluated"])
        self.assertIsNone(report["modelPreference"])


class InterpretationTests(unittest.TestCase):
    def contract(self):
        preparation = fixtures.preparation
        return {
            "admittedGenericSeriesIDs": ["series-001", "series-002"],
            "modelClassOrder": list(workload.MODEL_CLASS_IDS),
            "independentAggregation": {"parameterCount": 18},
            "crossSeriesRequirements": {
                "independentTimingConsistencyTolerance": 0.5,
                "timingComparisonTolerance": preparation.TIMING_COMPARISON_TOLERANCE,
                "positiveWeightSupportPerComponentPerSeries": preparation.MINIMUM_COMPONENT_POSITIVE_WEIGHT_SUPPORT,
            },
            "decisionRules": {
                "preferOrderedDoubletOverPositivePulse": {
                    "allPerSeriesDeltaWRSSAtLeast": preparation.ORDERED_OVER_POSITIVE_MINIMUM_PER_SERIES_DELTA_WRSS,
                    "globalDeltaBICAtLeast": preparation.ORDERED_OVER_POSITIVE_MINIMUM_DELTA_BIC,
                    "globalDeltaWRSSAtLeast": preparation.ORDERED_OVER_POSITIVE_MINIMUM_DELTA_WRSS,
                    "signAndSharedTimingRequirementsMustPass": True,
                },
                "rejectOrderedDoubletForIndependentPulses": {
                    "globalDeltaBICAtLeast": preparation.INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_BIC,
                    "globalDeltaWRSSAtLeast": preparation.INDEPENDENT_OVER_ORDERED_MINIMUM_DELTA_WRSS,
                    "independentCenterDispersionMustExceedTolerance": True, "signRequirementsMustPass": True,
                },
            },
        }

    def winner(self, wrss_by_series, k, start=0):
        fits = [{"genericSeriesID": f"series-{index + start + 1:03d}", "weightedResidualSumSquares": value,
                 "positiveAmplitudeSign": "positive", "negativeAmplitudeSign": "negative"}
                for index, value in enumerate(wrss_by_series)]
        return {**fixtures.validation._information_criteria(sum(wrss_by_series), 40 * len(fits), k),
                "gridIndex": start, "seriesFits": fits,
                "parameters": {"negativeCenter": -1.0 + start, "positiveCenter": 1.0 + start}}

    def test_combined_improvement_cannot_hide_a_worsening_series(self):
        winners = [self.winner([200.0, 200.0], 7), self.winner([100.0, 201.0], 12),
                   self.winner([80.0], 9), self.winner([180.0], 9, 1)]
        searches = [{"datasetID": f"model-{index}", "acceptedWinner": winner, "supportRequirementMet": True}
                    for index, winner in enumerate(winners)]
        independent = fixtures.validation._independent_aggregate(winners[2:], self.contract())
        comparisons = validation._model_comparisons(searches, independent, self.contract())
        ordered = comparisons["preferOrderedDoubletOverPositivePulse"]
        self.assertTrue(ordered["gates"]["globalDeltaWRSSPassed"])
        self.assertTrue(ordered["gates"]["globalDeltaBICPassed"])
        self.assertFalse(ordered["gates"]["allPerSeriesDeltaWRSSPassed"])
        self.assertFalse(ordered["passed"])
        constituent = ordered["gateDetails"]["allPerSeriesDeltaWRSSPassed"]["inputs"][1]
        self.assertEqual(constituent["inputs"]["delta"], -1.0)
        self.assertFalse(constituent["passed"])
        self.assertEqual(constituent["threshold"], fixtures.preparation.ORDERED_OVER_POSITIVE_MINIMUM_PER_SERIES_DELTA_WRSS)
        ranking = validation._numerical_ranking(searches, independent, self.contract())
        self.assertNotEqual(ranking[0]["modelClassID"], workload.POSITIVE_PULSE_ONLY)

    def test_joint_independent_bic_and_aicc_use_combined_n_and_k(self):
        winners = [self.winner([10.0], 9), self.winner([30.0], 9, 1)]
        aggregate = validation.legacy._independent_aggregate(winners, self.contract())
        self.assertEqual(aggregate["weightedResidualSumSquares"], 40.0)
        self.assertEqual(aggregate["positiveWeightSampleCount"], 80)
        self.assertEqual(aggregate["nominalParameterCount"], 18)
        self.assertAlmostEqual(aggregate["bayesianInformationCriterion"], 40.0 + 18.0 * math.log(80))
        self.assertAlmostEqual(aggregate["correctedAkaikeInformationCriterion"], 40.0 + 36.0 + 2.0 * 18.0 * 19.0 / 61.0)
        self.assertNotEqual(aggregate["bayesianInformationCriterion"], sum(w["bayesianInformationCriterion"] for w in winners))
        self.assertNotEqual(aggregate["correctedAkaikeInformationCriterion"], sum(w["correctedAkaikeInformationCriterion"] for w in winners))

    def test_support_inclusive_boundary_zero_weight_and_separate_exponentials(self):
        series = {"genericSeriesID": "series-001", "coordinates": [-2.0, 0.0, math.nextafter(2.0, math.inf)],
                  "inverseVariances": [1.0, 0.0, 1.0]}
        support = validation.legacy._component_support(series, 0.0, 0.0, 0.0, 1)
        self.assertEqual(support["positiveWeightSamplesWithinTwoEffectiveWidths"], 1)
        self.assertEqual(support["nearestPositiveWeightDistanceInEffectiveWidths"], 2.0)
        series["inverseVariances"][0] = 0.0
        self.assertFalse(validation.legacy._component_support(series, 0.0, 0.0, 0.0, 1)["supportRequirementMet"])
        series = {"genericSeriesID": "series-001", "coordinates": [0.0, 1.0], "inverseVariances": [1.0, 1.0]}
        support = validation.legacy._component_support(series, 0.0, 1.25, -4.125, 1)
        self.assertEqual(support["effectiveWidth"], math.exp(1.25) * math.exp(-4.125))

    def test_cli_has_exact_required_arguments(self):
        parser = validation._parser()
        required = [action for action in parser._actions if action.required]
        self.assertEqual({action.dest for action in required}, {
            "morphology_root", "coarse_project_root", "coarse_investigation_record", "coarse_validation_root",
            "supported_project_root", "supported_investigation_record", "output_root",
        })


if __name__ == "__main__":
    unittest.main()
