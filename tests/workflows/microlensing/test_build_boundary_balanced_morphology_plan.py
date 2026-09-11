"""Small producer-shaped ancestry; planning never evaluates proposed candidates."""

import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from tests.workflows.microlensing import test_validate_supported_doublet_refinement as fixtures
from workflows.microlensing import build_boundary_balanced_morphology_plan as planning

artifacts, numerical, workload = fixtures.artifacts, fixtures.numerical, fixtures.workload


class PlanFixture(fixtures.DiagnosticFixture):
    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        basis = fixtures.fixtures.fixtures.v1._component_basis
        # Known miniature geometry at three lower refinement boundaries. The
        # source preparation/investigation producers still emit their real formats.
        document["residualValues"] = [
            -5.0 * basis(x, -1.75, -2.875, -4.0) + 4.0 * basis(x, -1.5, -2.875, -4.0)
            + 0.01 * x * x + 0.003 * x * x * x for x in document["coordinates"]
        ]
        return document

    def setUp(self):
        super().setUp()
        self.set_refined_winner(self.candidate((0, 3, 0, 0, 0, 0)))
        self.followup_report = self.audit("refinement-validation")["result"]
        self.refinement_validation = self.root / "refinement-validation"
        self.assertEqual(self.followup_report["recommendedNextTest"], "PREDECLARE_BOUNDARY_AND_BALANCED_MODEL_FOLLOWUP")

    def plan(self, name="plan", **overrides):
        arguments = {
            "coarse_project_root": self.coarse, "coarse_investigation_record": self.investigation,
            "coarse_validation_root": self.coarse_validation, "supported_project_root": self.supported_root,
            "supported_investigation_record": self.supported_record, "supported_validation_root": self.supported_validation,
            "refinement_project_root": self.refinement_root, "refinement_investigation_record": self.refinement_record,
            "refinement_validation_root": self.refinement_validation, "output_root": self.root / name,
        }
        arguments.update(overrides)
        return planning.build_boundary_balanced_morphology_plan(self.morphology, **arguments)

    def reject_plan(self, message, **overrides):
        with self.assertRaisesRegex(planning.BoundaryBalancedMorphologyPlanError, message):
            self.plan("rejected", **overrides)
        self.assertFalse((self.root / "rejected").exists())


class BoundaryBalancedPlanTests(PlanFixture):
    def test_deterministic_verified_plan_budget_conflict_and_no_search(self):
        roots = (self.morphology, self.coarse, self.investigation.parent, self.coarse_validation, self.supported_root,
                 self.supported_record.parent, self.supported_validation, self.refinement_root, self.refinement_record.parent,
                 self.refinement_validation)
        before = {root: artifacts.serialized_tree(root) for root in roots}
        old = fixtures.fixtures.fixtures.v1
        with ExitStack() as stack:
            old_calls = stack.enter_context(patch.object(old, "_evaluate_candidate", wraps=old._evaluate_candidate))
            supported_calls = stack.enter_context(patch.object(numerical, "evaluate_candidate", wraps=numerical.evaluate_candidate))
            for owner, name in (
                (old, "_recompute_shard"), (workload, "_recompute"),
                (old.PLUGIN, "build_work_payloads"), (workload.PLUGIN, "build_work_payloads"),
                (workload.PLUGIN, "validate_result"), (workload.PLUGIN, "reduce_dataset"),
                (planning.parent_builder, "build_supported_doublet_refinement"), (planning.parent_builder, "_build_impl"),
                (planning.parent_builder.supported_builder, "_build_impl"),
                (planning.coarse, "_build_anomaly_morphology_coarse_grid_impl"),
                (planning.refinement, "validate_supported_doublet_refinement"), (planning.refinement, "_validate_impl"),
                (planning.supported, "_validate_impl"),
            ):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("no builders, searches, shards or parent publication")))
            result = self.plan()
        self.assertEqual(old_calls.call_count, 4)
        self.assertEqual(supported_calls.call_count, 5)
        self.assertEqual(supported_calls.call_args_list[-1].args[1], self.followup_report["search"]["acceptedWinner"]["gridIndex"])
        self.assertEqual({root: artifacts.serialized_tree(root) for root in roots}, before)
        self.plan("repeat")
        self.assertEqual(artifacts.serialized_tree(self.root / "plan"), artifacts.serialized_tree(self.root / "repeat"))
        plan = result["plan"]
        self.assertEqual(plan["planStatus"], "BUDGET_CONFLICT")
        self.assertEqual(plan["proposedSearchIDs"], [])
        self.assertEqual(plan["budget"]["candidateLimit"], 30000)
        self.assertEqual(plan["budget"]["proposedCandidateCount"], 0)
        self.assertEqual(plan["budget"]["requiredCandidateCount"], 324765)
        self.assertEqual(plan["budget"]["requiredWorkUnitCount"], 5078)
        self.assertEqual(plan["budget"]["requiredSampleCandidateEvaluationCount"], 5344416)
        self.assertIn("no domain is silently narrowed", plan["budget"]["conflict"])
        self.assertEqual([s["candidateCount"] for s in plan["requiredSearches"]], [522, 56457, 133893, 133893])
        self.assertEqual([s["workUnitCount"] for s in plan["requiredSearches"]], [9, 883, 2093, 2093])
        self.assertEqual([s["lastShardGridCount"] for s in plan["requiredSearches"]], [10, 9, 5, 5])
        for key, value in planning.refinement._CLAIMS.items():
            self.assertEqual(plan[key], value)
        self.assertFalse(plan["executionAuthorized"])
        self.assertFalse(plan["automaticRepeatedRefinement"])
        self.assertFalse(plan["projectLaunched"])
        self.assertEqual(plan["preservedHistoricalResults"], self.followup_report["preservedPR190Report"])
        self.assertEqual(plan["sourceAcceptedWinner"], self.followup_report["search"]["acceptedWinner"])
        self.assertEqual(plan["supportRule"], self.followup_report["supportRule"])
        contract = artifacts.read_json(self.morphology / planning.coarse.PREPARATION_CONTRACT_RELATIVE_PATH)
        self.assertEqual(plan["frozenNumericalExecution"], contract["deterministicExecution"])
        self.assertEqual(plan["frozenComparisonRules"], contract["decisionRules"])
        outcomes = {item["outcome"] for item in plan["stoppingRules"]}
        self.assertTrue({"UNRESOLVED_BUDGET_CONFLICT", "UNRESOLVED_BOUNDARY", "UNRESOLVED_SAMPLING"} <= outcomes)
        for relative, digest in result["artifactManifest"]["outputSHA256s"].items():
            self.assertEqual(digest, artifacts.sha256_bytes((self.root / "plan" / relative).read_bytes()))
        for relative, digest in plan["inputHashes"]["refinementValidationArtifacts"].items():
            self.assertEqual(digest, artifacts.sha256_bytes((self.refinement_validation / relative).read_bytes()))
        self.assertEqual(set(artifacts.serialized_tree(self.root / "plan")), {planning.PLAN_PATH, planning.MARKDOWN_PATH, planning.MANIFEST_PATH})

    def test_axes_samples_shapes_and_actual_common_center_pair_contract(self):
        plan = self.plan()["plan"]
        searches, comparison = plan["requiredSearches"], plan["comparability"]
        center, width = comparison["commonCenterAxis"], comparison["commonLogScaleAxis"]
        self.assertEqual(center, {"start": -2.25, "step": 0.0625, "count": 58})
        self.assertEqual(width, {"start": -3.125, "step": 0.0625, "count": 9})
        source = self.refinement_dataset
        for ordinal, search in enumerate(searches):
            dataset = search["datasetSpecification"]
            self.assertEqual(dataset["series"], source["series"] if ordinal < 2 else [source["series"][ordinal - 2]])
            self.assertEqual(dataset["candidatesPerWorkUnit"], source["candidatesPerWorkUnit"])
            for key, value in planning.supported._IDENTITIES.items():
                self.assertEqual(dataset[key], value)
            grid = dataset["morphologyGrid"]
            if ordinal == 0:
                self.assertEqual(grid["centerAxis"], center)
                self.assertEqual(grid["logScaleAxis"], width)
                self.assertEqual(grid["logShapeAxis"], source["morphologyGrid"]["positiveLogShapeAxis"])
            else:
                self.assertEqual(grid["negativeLogScaleAxis"], width)
                self.assertEqual(grid["positiveLogScaleAxis"], width)
                for name in ("negativeLogShapeAxis", "positiveLogShapeAxis"):
                    self.assertEqual(grid[name], source["morphologyGrid"][name])
                if ordinal > 1:
                    self.assertEqual(grid["centerAxis"], center)
                    self.assertNotIn("negativeCenterAxis", grid)
            _, view = numerical.numerical_view(dataset)
            self.assertEqual(view.grid.total_candidates, search["candidateCount"])
        ordered = searches[1]["datasetSpecification"]["morphologyGrid"]
        proof = comparison["orderedPairEmbedding"]
        for i in (0, ordered["negativeCenterAxis"]["count"] - 1):
            for j in (0, ordered["separationAxis"]["count"] - 1):
                positive_index = i + proof["minimumSeparationInCommonSteps"] + j
                pair = numerical.independent_center_pair_index(center["count"], i, positive_index)
                self.assertEqual(numerical.independent_center_pair_indices(center["count"], pair), (i, positive_index))
                negative = center["start"] + i * center["step"]
                positive = negative + (ordered["separationAxis"]["start"] + j * ordered["separationAxis"]["step"])
                expected = center["start"] + positive_index * center["step"]
                self.assertAlmostEqual(positive, expected, delta=workload.RESULT_RELATIVE_TOLERANCE * max(1.0, abs(expected)))
        self.assertGreater(proof["extraIndependentTimingPairCount"], 0)
        self.assertFalse(comparison["equalCandidateCountsDefineBalance"])
        self.assertFalse(comparison["balancedModelComparisonEstablished"])

    def test_union_width_domain_lattice_alignment_and_small_affordable_design(self):
        # Pure axis/cost fixtures; the public API still verifies actual PR192/194
        # artifacts before reaching this helper. No candidate search is used.
        dataset = copy.deepcopy(self.refinement_dataset)
        grid = dataset["morphologyGrid"]
        grid["negativeCenterAxis"] = {"start": -1.5, "step": 0.125, "count": 3}
        grid["separationAxis"] = {"start": 0.125, "step": 0.125, "count": 3}
        grid["negativeLogScaleAxis"] = {"start": -3.0, "step": 0.125, "count": 2}
        grid["positiveLogScaleAxis"] = {"start": -2.875, "step": 0.125, "count": 2}
        plan = planning._plan(dataset, self.followup_report, artifacts.read_json(self.morphology / planning.coarse.PREPARATION_CONTRACT_RELATIVE_PATH), {})
        self.assertEqual(plan["comparability"]["commonLogScaleAxis"], {"start": -3.125, "step": 0.125, "count": 4})
        self.assertEqual(plan["budget"]["requiredCandidateCount"], 1168)
        self.assertEqual(plan["planStatus"], "PREDECLARED_FOR_REVIEW")
        self.assertEqual(len(plan["proposedSearchIDs"]), 4)
        self.assertEqual(plan["budget"]["proposedCandidateCount"], 1168)
        self.assertFalse(plan["executionAuthorized"])
        self.assertEqual(planning._common_step((0.1875, 0.125), (0.125,)), (0.0625, 2))
        for lower, upper, step in ((0.0, 1.0, 0.0), (float("nan"), 1.0, 0.1), (1e16, 1e16 + 4, 0.01)):
            with self.assertRaises((ValueError, RuntimeError)):
                planning._cover_axis(lower, upper, step)
        with self.assertRaisesRegex(ValueError, "common lattice"):
            planning._common_step((1.0, 2 ** 0.5))

    def test_report_canonical_bytes_hashes_and_recomputed_outcomes(self):
        path = self.refinement_validation / planning.refinement.RESULT_RELATIVE_PATH
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        self.reject_plan("^refinement validation report is not canonical stable JSON$")
        path.write_bytes(original)
        manifest_path = self.refinement_validation / planning.refinement.MANIFEST_RELATIVE_PATH
        original_manifest = manifest_path.read_bytes()
        manifest = artifacts.read_json(manifest_path)
        digest = manifest["outputSHA256s"][planning.refinement.RESULT_RELATIVE_PATH]
        manifest["outputSHA256s"][planning.refinement.RESULT_RELATIVE_PATH] = ("1" if digest[0] == "0" else "0") + digest[1:]
        artifacts.write_json(manifest_path, manifest)
        self.reject_plan("^refinement validation manifest does not reconstruct exactly$")
        manifest_path.write_bytes(original_manifest)
        report = artifacts.read_json(path)
        report["recommendedNextTest"] = "FABRICATED"
        artifacts.write_json(path, report)
        self.reject_plan("^refinement validation report does not reconstruct exactly$")

    def test_source_domains_and_ledgers_are_verified(self):
        path = self.refinement_root / planning.parent_builder.DATASET_RELATIVE_PATH
        original = path.read_bytes()
        dataset = artifacts.read_json(path)
        dataset["series"][0]["values"][0] += 0.1
        artifacts.write_json(path, dataset)
        self.reject_plan("refinement dataset domain and provenance does not reconstruct exactly")
        path.write_bytes(original)
        ledger = self.refinement_record.parent / "stages" / "002-distributed-project.json"
        record = artifacts.read_json(ledger)
        record["result"]["projectCompletedWorkUnits"] += 1
        artifacts.write_json(ledger, record)
        self.reject_plan("stage ledger .* does not match investigation")

    def test_incompatible_followup_conditions_remain_explicit(self):
        for mutate, reason in (
            (lambda r: r["search"].update(acceptedWinner=None), "complete, reproduced and supported winner"),
            (lambda r: r["search"]["axes"][0].update(position="UPPER_BOUNDARY"), "three lower"),
            (lambda r: r["search"]["axes"][1].update(position="LOWER_BOUNDARY"), "interior separation"),
            (lambda r: r["search"]["axes"][3].update(fixed=False), "both fixed shapes"),
            (lambda r: r["historicalBaselineComparisons"]["preferOrderedDoubletOverPositivePulse"].update(passed=False), "all historical"),
        ):
            report = copy.deepcopy(self.followup_report)
            mutate(report)
            with self.assertRaisesRegex(ValueError, reason):
                planning._require_followup(report)

    def test_existing_output_input_paths_identity_and_transactional_cleanup(self):
        self.plan()
        before = artifacts.serialized_tree(self.root / "plan")
        self.reject_plan("output root already exists", output_root=self.root / "plan")
        self.assertEqual(artifacts.serialized_tree(self.root / "plan"), before)
        self.reject_plan("inside an input", output_root=self.refinement_validation / "child")
        link = self.root / "linked"
        link.symlink_to(self.refinement_validation, target_is_directory=True)
        self.reject_plan("symlink", refinement_validation_root=link)
        entries = set(self.root.iterdir())
        original_write = planning._atomic_write_bytes
        def fail_markdown(path, data):
            if path.name == planning.MARKDOWN_PATH:
                raise OSError("injected publication failure")
            original_write(path, data)
        with patch.object(planning, "_atomic_write_bytes", side_effect=fail_markdown):
            self.reject_plan("^injected publication failure$")
        self.assertEqual(set(self.root.iterdir()), entries)
        with patch.object(planning.Path, "rename", side_effect=OSError("injected rename failure")):
            self.reject_plan("^injected rename failure$")
        self.assertEqual(set(self.root.iterdir()), entries)
        self.rewrite_refinement(lambda r: r["metadata"].update(coordinator="http://OGLE-generic.invalid"))
        self.reject_plan("identity")


if __name__ == "__main__":
    unittest.main()
