"""Producer-shaped PR192 artifacts and single-dataset saved smoke investigations."""

import copy
import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from tests.workflows.microlensing import test_build_supported_doublet_refinement as fixtures
from tests.workflows.microlensing.test_refine_grid import stage_record
from workflows.microlensing import validate_supported_doublet_refinement as validation

artifacts, numerical, workload = fixtures.artifacts, fixtures.numerical, fixtures.workload
supported, builder = fixtures.validation, fixtures.refinement


def write_refinement_record(root, project_root, dataset, manifest, winner):
    project_path = project_root / "project.json"
    project = artifacts.read_json(project_path)
    project_hash = artifacts.sha256_bytes(project_path.read_bytes())
    count, total = manifest["totalCandidateCount"], manifest["totalExpectedWorkUnitCount"]
    status = {
        "id": dataset["id"], **supported._IDENTITIES, "modelClassID": dataset["modelClassID"],
        "morphologyFamilyID": workload.MORPHOLOGY_FAMILY_ID, "componentTemplateFamilyID": workload.COMPONENT_TEMPLATE_FAMILY_ID,
        "workloadStatus": "SUPPORTED_MORPHOLOGY_GRID_COMPLETE", "supportedMorphologyGridStatus": "SUPPORTED_MORPHOLOGY_GRID_COMPLETE",
        "coverageComplete": True, "totalCandidateCount": count, "completedCandidateCount": count,
        "totalInvalidCandidateCount": 0, "totalSupportRejectedCandidateCount": count - 1, "totalEligibleCandidateCount": 1,
        "assignedWorkUnits": 0, "pendingWorkUnits": 0, "failedWorkUnits": 0,
        "completedWorkUnits": total, "totalWorkUnits": total, "nodeContributions": {"generic-node": total}, "progress": 1.0,
    }
    # Counts are saved producer observations; fixture setup reproduces only its
    # one chosen candidate and makes no assertion about the uncomputed grid.
    fixtures.fixtures.set_winner(status, winner)
    run = {
        **{key: status[key] for key in ("assignedWorkUnits", "pendingWorkUnits", "failedWorkUnits", "completedWorkUnits", "totalWorkUnits")},
        "projectAssignedWorkUnits": 0, "projectPendingWorkUnits": 0, "projectFailedWorkUnits": 0,
        "projectCompletedWorkUnits": total, "projectTotalWorkUnits": total, "projectProgress": 1.0,
        "projectID": project["id"], "projectPath": str(project_path), "status": "COMPLETE", "workloadID": workload.WORKLOAD_ID,
        "nodeContributions": {"generic-node": total}, "datasets": [status],
    }
    ids = ("001-prepare-project", "002-distributed-project", "003-terminal-check")
    handlers = ("local.project.prepare", "openstar.project.run", "generic.project.terminal-check")
    parameters = ({"projectPath": str(project_path)}, {"projectPath": str(project_path), "projectManifestSha256": project_hash},
                  {"expectedProjectID": project["id"]})
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
    identity = "generic-refinement-investigation"
    path = root / identity / "investigation.json"
    artifacts.write_record(path, {
        "id": identity, "created_at": "2026-09-08T00:00:00+00:00", "updated_at": "2026-09-08T00:00:03+00:00",
        "stages": stages, "status": "COMPLETE", "workflow_id": "openstar.workflow.project-smoke.v1", "workflow_version": "20.0",
        "metadata": {"coordinator": "http://127.0.0.1:8080", "projectPath": str(project_path)},
    })
    return path


class DiagnosticFixture(fixtures.RefinementFixture):
    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        # Enough positive-weight samples for defined joint independent AICc.
        coordinates = sorted(document["coordinates"] + [-2.4, -2.3, 2.3, 2.4])
        document.update({
            "coordinates": coordinates, "residualValues": [0.01 * x * x + 0.003 * x * x * x for x in coordinates],
            "inverseVariances": [0.0 if x == -0.5 else 1.0 for x in coordinates],
            "sampleCount": len(coordinates), "sourceSampleIndices": list(range(len(coordinates))),
            "inclusionReasons": [["DETERMINISTIC_WINDOW"] for _ in coordinates],
        })
        return document

    def setUp(self):
        super().setUp()
        built = self.build()
        self.refinement_root = self.root / "refined"
        self.refinement_dataset, self.refinement_manifest = built["dataset"], built["buildManifest"]
        self.refined_winner = self.candidate((4, 3, 2, 0, 2, 0))
        self.refinement_record = write_refinement_record(
            self.root, self.refinement_root, self.refinement_dataset, self.refinement_manifest, self.refined_winner,
        )

    def candidate(self, indices):
        view, validated = numerical.numerical_view(self.refinement_dataset)
        index = numerical.candidate_index(indices, tuple(axis.count for axis in validated.grid.axes))
        evaluation = numerical.evaluate_candidate(view, index)
        self.assertIsNotNone(evaluation)
        return numerical.candidate_payload(evaluation, self.refinement_dataset["modelClassID"])

    def rewrite_refinement(self, mutate):
        record = artifacts.read_json(self.refinement_record)
        mutate(record)
        artifacts.write_record(self.refinement_record, record)

    def set_refined_winner(self, winner):
        self.rewrite_refinement(lambda r: fixtures.fixtures.set_winner(r["stages"][1]["result"]["datasets"][0], winner))

    def audit(self, name="audit", **overrides):
        arguments = {
            "coarse_project_root": self.coarse, "coarse_investigation_record": self.investigation,
            "coarse_validation_root": self.coarse_validation, "supported_project_root": self.supported_root,
            "supported_investigation_record": self.supported_record, "supported_validation_root": self.supported_validation,
            "refinement_project_root": self.refinement_root, "refinement_investigation_record": self.refinement_record,
            "output_root": self.root / name,
        }
        arguments.update(overrides)
        return validation.validate_supported_doublet_refinement(self.morphology, **arguments)

    def reject_audit(self, message, **overrides):
        with self.assertRaisesRegex(validation.SupportedDoubletRefinementValidationError, message):
            self.audit("rejected", **overrides)
        self.assertFalse((self.root / "rejected").exists())


class SupportedDoubletValidationTests(DiagnosticFixture):
    def test_actual_pr192_reconstruction_determinism_inputs_and_no_search(self):
        roots = (self.morphology, self.coarse, self.investigation.parent, self.coarse_validation, self.supported_root,
                 self.supported_record.parent, self.supported_validation, self.refinement_root, self.refinement_record.parent)
        before = {root: artifacts.serialized_tree(root) for root in roots}
        with ExitStack() as stack:
            old = fixtures.fixtures.v1
            old_calls = stack.enter_context(patch.object(old, "_evaluate_candidate", wraps=old._evaluate_candidate))
            new_calls = stack.enter_context(patch.object(numerical, "evaluate_candidate", wraps=numerical.evaluate_candidate))
            for owner, name in (
                (old, "_recompute_shard"), (workload, "_recompute"),
                (old.PLUGIN, "build_work_payloads"), (workload.PLUGIN, "build_work_payloads"),
                (old.PLUGIN, "validate_result"), (workload.PLUGIN, "validate_result"),
                (old.PLUGIN, "reduce_dataset"), (workload.PLUGIN, "reduce_dataset"),
                (builder, "build_supported_doublet_refinement"), (builder, "_build_impl"),
                (builder.supported_builder, "build_supported_anomaly_morphology_grid"), (builder.supported_builder, "_build_impl"),
                (builder.coarse, "build_anomaly_morphology_coarse_grid"), (builder.coarse, "_build_anomaly_morphology_coarse_grid_impl"),
                (supported, "validate_supported_anomaly_morphology_grid"), (supported, "_validate_impl"),
            ):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("no builders, grids, shards or parent publication")))
            result = self.audit()
        self.assertEqual(old_calls.call_count, 4)
        self.assertEqual(new_calls.call_count, 5)
        self.assertEqual([call.args[1] for call in new_calls.call_args_list], [0, self.winner["gridIndex"], 0, 0, self.refined_winner["gridIndex"]])
        self.assertEqual({root: artifacts.serialized_tree(root) for root in roots}, before)
        self.audit("repeat")
        self.assertEqual(artifacts.serialized_tree(self.root / "audit"), artifacts.serialized_tree(self.root / "repeat"))
        report = result["result"]
        self.assertEqual(report["refinementBuildManifest"], self.refinement_manifest)
        self.assertEqual(report["preservedPR190Report"], self.saved_report)
        self.assertEqual(report["search"]["acceptedWinner"], self.refined_winner)
        self.assertGreater(self.refined_winner["weightedResidualSumSquares"], 0.0)
        self.assertEqual(report["projectCompletedWorkUnits"], 145)
        self.assertEqual(report["accountingTotals"]["completedCandidateCount"], 9225)
        self.assertEqual(report["accountingTotals"]["totalEligibleCandidateCount"], 1)
        self.assertEqual(report["verificationScope"]["acceptedRefinementCandidatesReproduced"], 1)
        for field in ("fullGridEnumerated", "shardsRecomputed", "rejectionCountsIndependentlyReproduced", "globalOptimalityIndependentlyProven"):
            self.assertFalse(report["verificationScope"][field])
        for key, expected in validation._CLAIMS.items():
            self.assertEqual(report[key], expected)
        self.assertEqual(report["search"]["searchedBoundaryAxes"], [])
        self.assertEqual(report["search"]["fixedAxes"], ["negativeLogShape", "positiveLogShape"])
        self.assertIn("does not prove convergence", report["boundaryStatement"])
        for component in report["search"]["components"]:
            self.assertEqual(component["effectiveWidth"], math.exp(component["logScale"]) * math.exp(component["logShape"]))
            self.assertEqual(len(component["seriesSupport"]), 2)
            for row in component["seriesSupport"]:
                self.assertGreaterEqual(row["positiveWeightSamplesWithinTwoEffectiveWidths"], 1)
                self.assertLessEqual(row["nearestPositiveWeightDistanceInEffectiveWidths"], 2.0)
        positive = next(c for c in report["search"]["components"] if c["component"] == "positive")
        self.assertEqual(positive["center"], self.refined_winner["parameters"]["negativeCenter"] + self.refined_winner["parameters"]["separation"])
        for row in [report["previousOrderedComparison"]["global"], *report["previousOrderedComparison"]["perSeries"]]:
            self.assertAlmostEqual(row["deltaWRSS"], 0.0, delta=workload.RESULT_RELATIVE_TOLERANCE)
        for key in ("preferOrderedDoubletOverPositivePulse", "rejectOrderedDoubletForIndependentPulses"):
            comparison = report["historicalBaselineComparisons"][key]
            self.assertEqual(comparison["comparisonScope"], "HISTORICAL_BASELINE_DIAGNOSTIC")
            self.assertFalse(comparison["establishesModelPreference"])
            self.assertFalse(comparison["balancedModelComparisonEstablished"])
        manifest = result["artifactManifest"]
        for relative, digest in manifest["outputSHA256s"].items():
            self.assertEqual(digest, artifacts.sha256_bytes((self.root / "audit" / relative).read_bytes()))
        for relative, digest in report["inputHashes"]["refinementProjectArtifacts"].items():
            self.assertEqual(digest, artifacts.sha256_bytes((self.refinement_root / relative).read_bytes()))
        self.assertEqual(set(artifacts.serialized_tree(self.root / "audit")), {
            validation.RESULT_RELATIVE_PATH, validation.MARKDOWN_RELATIVE_PATH, validation.MANIFEST_RELATIVE_PATH,
        })

    def test_joint_independent_statistics_use_combined_counts(self):
        report = self.audit()["result"]
        winners = [item["acceptedWinner"] for item in self.saved_report["searches"][2:]]
        wrss = sum(item["weightedResidualSumSquares"] for item in winners)
        n = sum(item["positiveWeightSampleCount"] for item in winners)
        k = sum(item["nominalParameterCount"] for item in winners)
        joint = report["independentAggregate"]
        self.assertAlmostEqual(joint["bayesianInformationCriterion"], wrss + k * math.log(n), delta=workload.RESULT_RELATIVE_TOLERANCE)
        self.assertNotAlmostEqual(joint["bayesianInformationCriterion"], sum(w["bayesianInformationCriterion"] for w in winners))
        self.assertEqual(joint["correctedAkaikeInformationCriterionDefined"], n > k + 1)
        expected = wrss + 2 * k + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else None
        self.assertIsNotNone(expected)
        self.assertAlmostEqual(joint["correctedAkaikeInformationCriterion"], expected,
                               delta=workload.RESULT_RELATIVE_TOLERANCE * max(1.0, abs(expected)))
        self.assertNotAlmostEqual(joint["correctedAkaikeInformationCriterion"], sum(w["correctedAkaikeInformationCriterion"] for w in winners))

    def test_null_winner_is_complete_unresolved_with_all_comparisons_unevaluated(self):
        def nullify(record):
            status = record["stages"][1]["result"]["datasets"][0]
            status.update(totalEligibleCandidateCount=0, totalInvalidCandidateCount=2,
                          totalSupportRejectedCandidateCount=status["totalCandidateCount"] - 2)
            fixtures.fixtures.set_winner(status, None)
        self.rewrite_refinement(nullify)
        report = self.audit()["result"]
        self.assertTrue(report["coverageComplete"])
        self.assertIsNone(report["search"]["acceptedWinner"])
        self.assertEqual(report["overallClassification"], "UNRESOLVED_NO_ELIGIBLE_CANDIDATES")
        self.assertEqual(report["recommendedNextTest"], "REVIEW_SUPPORT_AND_ACCOUNTING_BEFORE_FURTHER_SEARCH")
        self.assertFalse(report["previousOrderedComparison"]["evaluated"])
        for name in ("preferOrderedDoubletOverPositivePulse", "rejectOrderedDoubletForIndependentPulses"):
            comparison = report["historicalBaselineComparisons"][name]
            self.assertFalse(comparison["evaluated"])
            self.assertFalse(comparison["passed"])
            self.assertTrue(all(not gate["evaluated"] and not gate["passed"] for gate in comparison["gateDetails"].values()))

    def test_single_dataset_counters_strict_accounting_and_duplicates(self):
        original = artifacts.read_json(self.refinement_record)
        mutations = (
            ("totalInvalidCandidateCount", True, "nonnegative integer"),
            ("totalSupportRejectedCandidateCount", -1, "nonnegative integer"),
            ("totalEligibleCandidateCount", 2**53, "safe integer"),
            ("totalEligibleCandidateCount", 2, "candidate accounting is inconsistent"),
            ("totalInvalidCandidateCount", 9226, "candidate accounting is inconsistent"),
            ("completedCandidateCount", 9225.0, "completedCandidateCount does not reconstruct exactly"),
            ("coverageComplete", False, "coverageComplete does not reconstruct exactly"),
            ("completedWorkUnits", 144, "completedWorkUnits does not reconstruct exactly"),
            ("failedWorkUnits", 1, "failedWorkUnits does not reconstruct exactly"),
            ("pendingWorkUnits", 1, "pendingWorkUnits does not reconstruct exactly"),
            ("assignedWorkUnits", 1, "assignedWorkUnits does not reconstruct exactly"),
            ("bestGridIndex", 0, "duplicated winner bestGridIndex"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key, value=value):
                record = copy.deepcopy(original)
                record["stages"][1]["result"]["datasets"][0][key] = value
                artifacts.write_record(self.refinement_record, record)
                self.reject_audit(message)
        artifacts.write_record(self.refinement_record, original)
        self.rewrite_refinement(lambda r: r["stages"][1]["result"].update(projectCompletedWorkUnits=144))
        self.reject_audit("project counters lack exact complete coverage")
        artifacts.write_record(self.refinement_record, original)
        self.rewrite_refinement(lambda r: r["stages"][1]["result"].update(completedWorkUnits=144))
        self.reject_audit("current-dataset counters disagree")
        artifacts.write_record(self.refinement_record, original)
        self.rewrite_refinement(lambda r: r["stages"][1]["result"].update(totalEligibleCandidateCount=True))
        self.reject_audit("totalEligibleCandidateCount must be a nonnegative integer")

    def test_actual_identities_mapping_and_nonrecursive_winner_selection(self):
        original = artifacts.read_json(self.refinement_record)
        for field, value in (("workloadID", fixtures.fixtures.v1.WORKLOAD_ID), ("supportPolicyID", "unknown"),
                             ("datasetSchemaID", None), ("executionContractVersion", 1.0)):
            record = copy.deepcopy(original)
            record["stages"][1]["result"]["datasets"][0][field] = value
            artifacts.write_record(self.refinement_record, record)
            self.reject_audit(f"{field} is invalid")
        for shape in ([], [0, 0]):
            record = copy.deepcopy(original)
            statuses = record["stages"][1]["result"]["datasets"]
            record["stages"][1]["result"]["datasets"] = [statuses[i] for i in shape]
            artifacts.write_record(self.refinement_record, record)
            self.reject_audit("exactly one canonical dataset summary")
        artifacts.write_record(self.refinement_record, original)
        self.rewrite_refinement(lambda r: r["stages"][1]["result"]["datasets"][0].update(id="wrong-dataset"))
        self.reject_audit("refinement dataset mapping does not reconstruct exactly")
        artifacts.write_record(self.refinement_record, original)
        self.rewrite_refinement(lambda r: r["stages"][1]["result"].update(payload={"best": {"payload": {"bestCandidate": None}}}))
        self.assertEqual(self.audit()["result"]["search"]["acceptedWinner"], self.refined_winner)

    def test_grid_parameters_numerical_fits_and_summary_fields(self):
        original = artifacts.read_json(self.refinement_record)
        for field, message in (("index", "accepted grid index is outside"), ("parameters", "winner parameter mapping"),
                               ("fit", "winner does not reproduce"), ("bic", "winner does not reproduce"),
                               ("null", "null winner disagrees")):
            record = copy.deepcopy(original)
            status = record["stages"][1]["result"]["datasets"][0]
            winner = status["payload"]["bestCandidate"]
            if field == "index":
                winner["gridIndex"] = status["totalCandidateCount"]
            elif field == "parameters":
                winner["parameters"]["negativeCenter"] += 0.01
            elif field == "fit":
                winner["seriesFits"][0]["offset"] += 0.1
            elif field == "bic":
                winner["bayesianInformationCriterion"] += 1.0
            else:
                winner = None
            fixtures.fixtures.set_winner(status, winner)
            artifacts.write_record(self.refinement_record, record)
            self.reject_audit(message)
        artifacts.write_record(self.refinement_record, original)
        self.rewrite_refinement(lambda r: r["stages"][1]["result"]["datasets"][0].pop("bestSeriesFits"))
        self.reject_audit("missing duplicated winner field bestSeriesFits")

    def test_unsupported_candidate_and_searched_versus_fixed_boundaries(self):
        self.set_refined_winner(self.candidate((4, 0, 2, 0, 2, 0)))
        self.reject_audit("accepted supported winner lacks observational support")
        self.set_refined_winner(self.candidate((0, 3, 2, 0, 2, 0)))
        report = self.audit()["result"]
        self.assertEqual(report["search"]["searchedBoundaryAxes"], ["negativeCenter"])
        self.assertEqual(report["search"]["fixedAxes"], ["negativeLogShape", "positiveLogShape"])
        self.assertEqual(report["overallClassification"], "UNRESOLVED_SEARCHED_BOUNDARY")
        self.assertIn("searched boundary is present", report["boundaryStatement"])
        self.assertFalse(report["convergenceEstablished"])

    def test_canonical_json_separate_from_manifest_hash_mismatch(self):
        report_path = self.supported_validation / supported.RESULT_RELATIVE_PATH
        original = report_path.read_bytes()
        report_path.write_bytes(original + b"\n")
        self.reject_audit("^supported validation report is not canonical stable JSON$")
        report_path.write_bytes(original)
        manifest_path = self.supported_validation / supported.MANIFEST_RELATIVE_PATH
        manifest = artifacts.read_json(manifest_path)
        digest = manifest["outputSHA256s"][supported.RESULT_RELATIVE_PATH]
        manifest["outputSHA256s"][supported.RESULT_RELATIVE_PATH] = ("1" if digest[0] == "0" else "0") + digest[1:]
        artifacts.write_json(manifest_path, manifest)
        self.reject_audit("^supported validation manifest does not reconstruct exactly$")
        self.assertEqual(report_path.read_bytes(), original)

    def test_pr192_hashes_derivation_samples_and_retained_references(self):
        manifest_path = self.refinement_root / builder.BUILD_MANIFEST_RELATIVE_PATH
        original_manifest = artifacts.read_json(manifest_path)
        for field in ("outputSHA256s", "axisDerivation", "parentWinnerInclusion", "totalCandidateCount", "finalWorkUnitGridCount",
                      "totalExpectedSampleCandidateEvaluationCount", "retainedModelResultReferences", "buildManifestSHA256"):
            with self.subTest(field=field):
                manifest = copy.deepcopy(original_manifest)
                manifest[field] = None
                artifacts.write_json(manifest_path, manifest)
                self.reject_audit("^refinement build manifest does not reconstruct exactly$")
        artifacts.write_json(manifest_path, original_manifest)
        manifest = copy.deepcopy(original_manifest)
        digest = manifest["outputSHA256s"][builder.DATASET_RELATIVE_PATH]
        manifest["outputSHA256s"][builder.DATASET_RELATIVE_PATH] = ("1" if digest[0] == "0" else "0") + digest[1:]
        manifest.pop("buildManifestSHA256")
        manifest["buildManifestSHA256"] = artifacts.sha256_bytes(builder.coarse._canonical_compact_json_bytes(manifest))
        artifacts.write_json(manifest_path, manifest)
        self.reject_audit("^refinement build manifest does not reconstruct exactly$")
        artifacts.write_json(manifest_path, original_manifest)
        dataset_path = self.refinement_root / builder.DATASET_RELATIVE_PATH
        original_dataset = artifacts.read_json(dataset_path)
        for field in ("axis", "sample", "weights", "series order", "fixed shape"):
            dataset = copy.deepcopy(original_dataset)
            if field == "axis":
                dataset["morphologyGrid"]["negativeCenterAxis"]["step"] *= 2
            elif field == "sample":
                dataset["series"][0]["values"][0] += 0.1
            elif field == "weights":
                dataset["series"][0]["inverseVariances"][0] *= 2
            elif field == "series order":
                dataset["series"].reverse()
            else:
                dataset["morphologyGrid"]["positiveLogShapeAxis"]["values"][0] += 0.1
            artifacts.write_json(dataset_path, dataset)
            self.reject_audit("series must be in canonical generic-series-ID order" if field == "series order"
                              else "refinement dataset domain and provenance does not reconstruct exactly")
        artifacts.write_json(dataset_path, original_dataset)
        project_path = self.refinement_root / builder.PROJECT_RELATIVE_PATH
        project = artifacts.read_json(project_path)
        project["datasets"][0]["path"] = "../outside.json"
        artifacts.write_json(project_path, project)
        self.reject_audit("refinement project does not reconstruct exactly")

    def test_original_ancestry_and_refinement_canonical_bytes_are_checked(self):
        manifest_path = self.coarse_validation / artifacts.validation.MANIFEST_RELATIVE_PATH
        original = manifest_path.read_bytes()
        manifest = artifacts.read_json(manifest_path)
        manifest["inputHashes"]["coarseInvestigationRecord"] = "0" * 64
        artifacts.write_json(manifest_path, manifest)
        self.reject_audit("^validation manifest does not reconstruct exactly$")
        manifest_path.write_bytes(original)
        project_path = self.refinement_root / builder.PROJECT_RELATIVE_PATH
        project_path.write_bytes(project_path.read_bytes() + b"\n")
        self.reject_audit("^refinement project is not canonical stable JSON$")

    def test_stage_hashes_ledgers_order_terminal_paths_and_contributions(self):
        original = artifacts.read_json(self.refinement_record)
        stale = copy.deepcopy(original)
        stale["stages"][1]["provenance"]["result_hash"] = "0" * 64
        artifacts.write_json(self.refinement_record, stale)
        self.reject_audit("result hash")
        artifacts.write_record(self.refinement_record, original)
        ledger = self.refinement_record.parent / "stages" / "002-distributed-project.json"
        changed = artifacts.read_json(ledger)
        changed["result"]["projectCompletedWorkUnits"] -= 1
        artifacts.write_json(ledger, changed)
        self.reject_audit("stage ledger .* does not match investigation")
        mutations = (
            (lambda r: r["stages"].reverse(), "supported stage order"),
            (lambda r: r.update(status="INCOMPLETE"), "supported investigation status"),
            (lambda r: r["stages"][2]["result"].update(passed=False), "supported terminal result"),
            (lambda r: r["metadata"].update(projectPath=str(self.supported_root / "project.json")), "different project"),
            (lambda r: r["stages"][1]["provenance"]["input_hashes"].update(projectManifest="0" * 64), "supported stage input hashes"),
            (lambda r: r["stages"][1]["result"]["datasets"][0].update(nodeContributions={"generic-node": True}), "nonnegative integer"),
            (lambda r: r["stages"][1]["result"]["datasets"][0].update(nodeContributions={"other-node": 145}), "dataset/project contributions"),
            (lambda r: r["metadata"].update(coordinator="http://OGLE-generic.invalid"), "identity"),
        )
        for mutate, message in mutations:
            record = copy.deepcopy(original)
            mutate(record)
            artifacts.write_record(self.refinement_record, record)
            self.reject_audit(message)

    def test_existing_output_symlinks_and_transactional_cleanup(self):
        self.audit()
        before = artifacts.serialized_tree(self.root / "audit")
        self.reject_audit("output root already exists", output_root=self.root / "audit")
        self.assertEqual(artifacts.serialized_tree(self.root / "audit"), before)
        self.reject_audit("inside an input artifact directory", output_root=self.refinement_root / "child")
        linked = self.root / "linked"
        linked.symlink_to(self.refinement_root, target_is_directory=True)
        self.reject_audit("symlink", refinement_project_root=linked)
        entries = set(self.root.iterdir())
        original_write = validation._atomic_write_bytes
        def fail_markdown(path, payload):
            if path.name == validation.MARKDOWN_RELATIVE_PATH:
                raise OSError("injected Markdown publication failure")
            original_write(path, payload)
        with patch.object(validation, "_atomic_write_bytes", side_effect=fail_markdown):
            self.reject_audit("^injected Markdown publication failure$")
        self.assertEqual(set(self.root.iterdir()), entries)
        with patch.object(validation.Path, "rename", side_effect=OSError("injected rename failure")):
            self.reject_audit("^injected rename failure$")
        self.assertEqual(set(self.root.iterdir()), entries)

    def test_frozen_gates_cannot_hide_a_worsening_series_or_imply_preference(self):
        # Pure gate fixture: these statistics are not submitted as fabricated
        # accepted results. Public reproduction rejects such substitutions above.
        report = self.audit()["result"]
        baseline = copy.deepcopy(self.saved_report)
        search = copy.deepcopy(report["search"])
        contract = artifacts.read_json(self.morphology / builder.coarse.PREPARATION_CONTRACT_RELATIVE_PATH)
        positive, ordered = baseline["searches"][0]["acceptedWinner"], search["acceptedWinner"]
        for candidate, values in ((positive, (100.0, 100.0)), (ordered, (110.0, 1.0))):
            for fit, value in zip(candidate["seriesFits"], values):
                fit["weightedResidualSumSquares"] = value
                fit["positiveAmplitudeSign"] = "positive"
                if candidate is ordered:
                    fit["negativeAmplitudeSign"] = "negative"
            candidate["weightedResidualSumSquares"] = sum(values)
            candidate["bayesianInformationCriterion"] = sum(values) + candidate["nominalParameterCount"] * math.log(candidate["positiveWeightSampleCount"])
        _, historical, _ = validation._comparisons(search, baseline, contract)
        details = historical["preferOrderedDoubletOverPositivePulse"]["gateDetails"]
        self.assertTrue(details["globalDeltaWRSSPassed"]["passed"])
        self.assertFalse(details["allPerSeriesDeltaWRSSPassed"]["passed"])
        self.assertFalse(details["allPerSeriesDeltaWRSSPassed"]["inputs"][0]["passed"])
        self.assertEqual(details["allPerSeriesDeltaWRSSPassed"]["threshold"], contract["decisionRules"]["preferOrderedDoubletOverPositivePulse"]["allPerSeriesDeltaWRSSAtLeast"])
        self.assertEqual(validation._outcome(search, historical)[:2], (
            "DIAGNOSTIC_GATES_NOT_PASSED", "REVIEW_FAILED_HISTORICAL_GATES_BEFORE_BALANCED_FOLLOWUP",
        ))
        self.assertIsNone(report["modelPreference"])
        self.assertFalse(report["balancedModelComparisonEstablished"])
        # Even a passing historical ordered-over-positive rule is never promoted
        # to a new preference by this report constructor.
        ordered["seriesFits"][0]["weightedResidualSumSquares"] = 1.0
        ordered.update(validation.legacy._information_criteria(2.0, ordered["positiveWeightSampleCount"], ordered["nominalParameterCount"]))
        record = artifacts.read_json(self.refinement_record)
        diagnostic = validation._report(
            SimpleNamespace(project=artifacts.read_json(self.refinement_root / "project.json"), manifest=self.refinement_manifest),
            record, record["stages"][1]["result"], search, baseline, contract, report["inputHashes"],
        )
        self.assertTrue(diagnostic["historicalBaselineComparisons"]["preferOrderedDoubletOverPositivePulse"]["passed"])
        self.assertIsNone(diagnostic["modelPreference"])
        self.assertFalse(diagnostic["balancedModelComparisonEstablished"])

    def test_passing_ordered_condition_does_not_require_independent_rejection(self):
        contract = artifacts.read_json(self.morphology / builder.coarse.PREPARATION_CONTRACT_RELATIVE_PATH)
        cases = (
            ((4, 3, 2, 0, 2, 0), "HISTORICAL_BASELINE_DIAGNOSTIC_COMPLETE", "PREDECLARE_BALANCED_MODEL_AND_STABILITY_CHECK"),
            ((0, 3, 2, 0, 2, 0), "UNRESOLVED_SEARCHED_BOUNDARY", "PREDECLARE_BOUNDARY_AND_BALANCED_MODEL_FOLLOWUP"),
        )
        for ordinal, (indices, classification, next_test) in enumerate(cases):
            with self.subTest(boundary=ordinal == 1):
                self.set_refined_winner(self.candidate(indices))
                report = self.audit(f"condition-{ordinal}")["result"]
                baseline, search = copy.deepcopy(self.saved_report), copy.deepcopy(report["search"])
                self.assertEqual(search["searchedBoundaryAxes"], ["negativeCenter"] if ordinal else [])
                # Pure gate/report fixtures, not fabricated persisted winners.
                # Keep the verified geometry and frozen thresholds, with clear
                # ordered improvement and insufficient independent improvement.
                for candidate, values in ((baseline["searches"][0]["acceptedWinner"], (100.0, 100.0)),
                                          (search["acceptedWinner"], (1.0, 1.0))):
                    for fit, wrss in zip(candidate["seriesFits"], values):
                        fit["weightedResidualSumSquares"] = wrss
                        fit["positiveAmplitude"], fit["positiveAmplitudeSign"] = 1.0, "positive"
                        if "negativeAmplitude" in fit:
                            fit["negativeAmplitude"], fit["negativeAmplitudeSign"] = -1.0, "negative"
                    candidate.update(validation.legacy._information_criteria(
                        sum(values), candidate["positiveWeightSampleCount"], candidate["nominalParameterCount"],
                    ))
                _, historical, _ = validation._comparisons(search, baseline, contract)
                ordered = historical["preferOrderedDoubletOverPositivePulse"]
                self.assertTrue(all(gate["evaluated"] and gate["passed"] for gate in ordered["gateDetails"].values()))
                rejection = historical["rejectOrderedDoubletForIndependentPulses"]
                for key in ("globalDeltaWRSSPassed", "globalDeltaBICPassed"):
                    self.assertTrue(rejection["gateDetails"][key]["evaluated"])
                    self.assertFalse(rejection["gateDetails"][key]["passed"])
                original_gates = copy.deepcopy(historical)
                record = artifacts.read_json(self.refinement_record)
                diagnostic = validation._report(
                    SimpleNamespace(project=artifacts.read_json(self.refinement_root / "project.json"), manifest=self.refinement_manifest),
                    record, record["stages"][1]["result"], search, baseline, contract, report["inputHashes"],
                )
                self.assertEqual(diagnostic["overallClassification"], classification)
                self.assertEqual(diagnostic["recommendedNextTest"], next_test)
                self.assertEqual(diagnostic["historicalBaselineComparisons"], original_gates)
                for key in ("globalDeltaWRSSPassed", "globalDeltaBICPassed"):
                    self.assertIn(f"rejectOrderedDoubletForIndependentPulses.{key}", diagnostic["failedHistoricalGatePaths"])
                for key, value in validation._CLAIMS.items():
                    self.assertEqual(diagnostic[key], value)
                self.assertEqual(diagnostic["convergenceStatus"], "UNRESOLVED")
                self.assertIn("convergence remains unresolved" if ordinal else "does not prove convergence", diagnostic["boundaryStatement"])


class ZeroAmplitudeTests(DiagnosticFixture):
    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        document["residualValues"] = [0.0] * len(document["coordinates"])
        return document

    def test_zero_amplitude_remains_supported_but_fails_strict_sign_gate(self):
        report = self.audit()["result"]
        self.assertTrue(report["search"]["supportRequirementMet"])
        self.assertTrue(all(fit["negativeAmplitude"] == 0.0 for fit in report["search"]["acceptedWinner"]["seriesFits"]))
        gate = report["historicalBaselineComparisons"]["preferOrderedDoubletOverPositivePulse"]["gateDetails"]["signRequirementsPassed"]
        self.assertTrue(gate["evaluated"])
        self.assertFalse(gate["passed"])
        self.assertIsNone(report["modelPreference"])
        self.set_refined_winner(self.candidate((4, 0, 2, 0, 2, 0)))
        self.reject_audit("accepted supported winner lacks observational support")


if __name__ == "__main__":
    unittest.main()
