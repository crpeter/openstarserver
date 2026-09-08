"""Miniature producer artifacts; the builder must never search new candidates."""

import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from tests.workflows.microlensing import test_validate_supported_anomaly_morphology_grid as fixtures
from workflows.microlensing import build_supported_doublet_refinement as refinement

artifacts = fixtures.fixtures
numerical, workload, validation = fixtures.numerical, fixtures.workload, fixtures.validation


class RefinementFixture(fixtures.SupportedFixture):
    def _axes(self, **kwargs):
        axes = artifacts.CoarseMorphologyFixture._axes(self, center_count=3, log_scale_count=3)
        axes["SEPARATION"]["count"] = 2
        axes["LOG_SHAPE"].update(count=1, values=[-4.0])
        return axes

    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        coordinates = sorted(set(document["coordinates"] + [-1.25]))
        document.update({
            "coordinates": coordinates, "residualValues": [0.01 * x * x + 0.003 * x * x * x for x in coordinates],
            "inverseVariances": [0.0 if x == -0.5 else 1.0 for x in coordinates],
            "sampleCount": len(coordinates), "sourceSampleIndices": list(range(len(coordinates))),
            "inclusionReasons": [["DETERMINISTIC_WINDOW"] for _ in coordinates],
        })
        return document

    def setUp(self):
        super().setUp()
        project = artifacts.read_json(self.supported_root / "project.json")
        self.parent = artifacts.read_json(self.supported_root / project["datasets"][1]["path"])
        self.save_parent((1, 0, 1, 0, 1, 0))
        self.saved_report = self.validate("supported-validation")["result"]
        self.supported_validation = self.root / "supported-validation"

    def save_parent(self, indices):
        view, validated = numerical.numerical_view(self.parent)
        index = numerical.candidate_index(indices, tuple(axis.count for axis in validated.grid.axes))
        evaluation = numerical.evaluate_candidate(view, index)
        self.assertIsNotNone(evaluation)
        self.winner = numerical.candidate_payload(evaluation, self.parent["modelClassID"])
        self.rewrite(lambda record: fixtures.set_winner(record["stages"][1]["result"]["datasets"][1], self.winner))

    def build(self, name="refined", **overrides):
        arguments = {
            "coarse_project_root": self.coarse, "coarse_investigation_record": self.investigation,
            "coarse_validation_root": self.coarse_validation, "supported_project_root": self.supported_root,
            "supported_investigation_record": self.supported_record, "supported_validation_root": self.supported_validation,
            "project_id": "generic-doublet-refinement", "output_root": self.root / name,
        }
        arguments.update(overrides)
        return refinement.build_supported_doublet_refinement(self.morphology, **arguments)

    def reject_refinement(self, pattern, **overrides):
        with self.assertRaisesRegex(refinement.SupportedDoubletRefinementBuildError, pattern):
            self.build("rejected", **overrides)
        self.assertFalse((self.root / "rejected").exists())


class SupportedDoubletRefinementTests(RefinementFixture):
    def test_deterministic_preservation_budget_inclusion_and_no_search(self):
        roots = (self.morphology, self.coarse, self.investigation.parent, self.coarse_validation,
                 self.supported_root, self.supported_record.parent, self.supported_validation)
        before = {root: artifacts.serialized_tree(root) for root in roots}
        with ExitStack() as stack:
            old_calls = stack.enter_context(patch.object(fixtures.v1, "_evaluate_candidate", wraps=fixtures.v1._evaluate_candidate))
            supported_calls = stack.enter_context(patch.object(numerical, "evaluate_candidate", wraps=numerical.evaluate_candidate))
            for owner, name in (
                (fixtures.v1, "_recompute_shard"), (workload, "_recompute"),
                (fixtures.v1.PLUGIN, "build_work_payloads"), (workload.PLUGIN, "build_work_payloads"),
                (fixtures.v1.PLUGIN, "validate_result"), (workload.PLUGIN, "validate_result"),
                (fixtures.v1.PLUGIN, "reduce_dataset"), (workload.PLUGIN, "reduce_dataset"),
                (refinement.coarse, "build_anomaly_morphology_coarse_grid"),
                (refinement.coarse, "_build_anomaly_morphology_coarse_grid_impl"),
                (fixtures.builder, "build_supported_anomaly_morphology_grid"),
                (fixtures.builder, "_build_impl"),
                (validation, "validate_supported_anomaly_morphology_grid"), (validation, "_validate_impl"),
            ):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("no enumeration, shards or producer invocation")))
            result = self.build()
        self.assertEqual(old_calls.call_count, 4)
        self.assertEqual(supported_calls.call_count, 4)
        self.assertEqual([call.args[1] for call in supported_calls.call_args_list], [0, self.winner["gridIndex"], 0, 0])
        self.assertEqual({root: artifacts.serialized_tree(root) for root in roots}, before)
        self.build("repeat")
        self.assertEqual(artifacts.serialized_tree(self.root / "refined"), artifacts.serialized_tree(self.root / "repeat"))
        self.assertEqual(set(artifacts.serialized_tree(self.root / "refined")), {
            refinement.PROJECT_RELATIVE_PATH, refinement.DATASET_RELATIVE_PATH, refinement.BUILD_MANIFEST_RELATIVE_PATH,
        })
        dataset, manifest = result["dataset"], result["buildManifest"]
        preserved = copy.deepcopy(dataset)
        for key in ("doubletRefinementProvenance", "interpretationLimits"):
            preserved.pop(key)
        preserved["id"], preserved["morphologyGrid"] = self.parent["id"], self.parent["morphologyGrid"]
        self.assertEqual(preserved, self.parent)
        self.assertEqual(dataset["series"], self.parent["series"])
        for key, value in validation._IDENTITIES.items():
            self.assertEqual(dataset[key], value)
            self.assertEqual(result["project"][key], value)
        self.assertEqual(len(result["project"]["datasets"]), 1)
        self.assertEqual(manifest["totalCandidateCount"], 9225)
        self.assertEqual(manifest["candidatesPerWorkUnit"], 64)
        self.assertEqual(manifest["totalExpectedWorkUnitCount"], 145)
        self.assertEqual(manifest["finalWorkUnitGridStartIndex"], 9216)
        self.assertEqual(manifest["finalWorkUnitGridCount"], 9)
        samples = sum(len(series["coordinates"]) for series in self.parent["series"])
        self.assertEqual(manifest["sampleCount"], samples)
        self.assertEqual(manifest["totalExpectedSampleCandidateEvaluationCount"], 9225 * samples)
        inclusion = manifest["parentWinnerInclusion"]
        self.assertEqual(inclusion["gridIndex"], 4187)
        self.assertEqual(inclusion["axisIndices"], dict(zip(refinement._AXIS_ORDER, (4, 3, 2, 0, 2, 0))))
        self.assertFalse(inclusion["numericallyEvaluated"])
        for name, actual in inclusion["parameters"].items():
            expected = self.winner["parameters"][name]
            self.assertAlmostEqual(actual, expected, delta=workload.RESULT_RELATIVE_TOLERANCE * max(1.0, abs(actual), abs(expected)))
        axes = dataset["morphologyGrid"]
        for name, count, divisor, offset in (("negativeCenter", 9, 8, 4), ("negativeLogScale", 5, 4, 2), ("positiveLogScale", 5, 4, 2)):
            step = self.parent["morphologyGrid"][name + "Axis"]["step"] / divisor
            self.assertEqual(axes[name + "Axis"], {"start": self.winner["parameters"][name] - offset * step, "step": step, "count": count})
        separation_step = self.winner["parameters"]["separation"] / 4.0
        self.assertEqual(axes["separationAxis"], {"start": separation_step, "step": separation_step, "count": 41})
        self.assertGreater(separation_step, 0.0)
        for name in ("negativeLogShapeAxis", "positiveLogShapeAxis"):
            self.assertEqual(axes[name], self.parent["morphologyGrid"][name])
        self.assertEqual(manifest["provenance"]["sourceAcceptedWinner"], self.winner)
        self.assertEqual(manifest["provenance"]["sourceGridIndex"], self.winner["gridIndex"])
        for relative, digest in manifest["provenance"]["inputHashes"]["supportedValidationArtifacts"].items():
            self.assertEqual(digest, artifacts.sha256_bytes((self.supported_validation / relative).read_bytes()))
        for relative, digest in manifest["outputSHA256s"].items():
            self.assertEqual(digest, artifacts.sha256_bytes((self.root / "refined" / relative).read_bytes()))
        unsigned = copy.deepcopy(manifest)
        digest = unsigned.pop("buildManifestSHA256")
        self.assertEqual(digest, artifacts.sha256_bytes(refinement.coarse._canonical_compact_json_bytes(unsigned)))
        self.assertEqual([item["validationSearchIndex"] for item in manifest["retainedModelResultReferences"]], [0, 2, 3])
        for reference in manifest["retainedModelResultReferences"]:
            self.assertFalse(reference["rerunInThisProject"])
            self.assertEqual(reference["acceptedWinner"], self.saved_report["searches"][reference["validationSearchIndex"]]["acceptedWinner"])
        for key in ("newModelPreferenceComputed", "balancedModelComparisonEstablished", "globalOptimumEstablished",
                    "convergenceEstablished", "measuredDurationEstablished", "crossSeriesReplicationEstablished",
                    "planetaryInterpretationResolved", "discoveryClaim"):
            self.assertFalse(manifest["interpretationLimits"][key])

    def test_verified_but_incompatible_parent_boundaries(self):
        for ordinal, (indices, message) in enumerate((
            ((0, 0, 1, 0, 1, 0), "negativeCenter must be interior"),
            ((1, 1, 1, 0, 1, 0), "separation must be at the lower searched boundary"),
            ((1, 0, 0, 0, 1, 0), "negativeLogScale must be interior"),
            ((1, 0, 1, 0, 2, 0), "positiveLogScale must be interior"),
        )):
            with self.subTest(indices=indices):
                self.save_parent(indices)
                report_name = f"boundary-report-{ordinal}"
                self.validate(report_name)
                self.reject_refinement(message, supported_validation_root=self.root / report_name)

    def test_complete_null_ordered_winner_is_incompatible(self):
        def nullify(record):
            status = record["stages"][1]["result"]["datasets"][1]
            status["totalEligibleCandidateCount"] = 0
            status["totalSupportRejectedCandidateCount"] = status["totalCandidateCount"]
            fixtures.set_winner(status, None)
        self.rewrite(nullify)
        self.validate("null-report")
        self.reject_refinement("ordered-doublet parent requires a non-null accepted winner", supported_validation_root=self.root / "null-report")

    def test_parent_incomplete_or_unsupported_is_rejected(self):
        original = artifacts.read_json(self.supported_record)
        self.rewrite(lambda r: r["stages"][1]["result"]["datasets"][1].update(coverageComplete=False))
        self.reject_refinement("^supported dataset coverageComplete does not reconstruct exactly$")
        artifacts.write_record(self.supported_record, original)
        # An actual unsupported saved candidate at positive center -0.75.
        self.save_parent((2, 0, 1, 0, 1, 0))
        self.reject_refinement("accepted supported winner lacks observational support")

    def test_canonical_report_rejection_and_separate_manifest_hash_check(self):
        path = self.supported_validation / validation.RESULT_RELATIVE_PATH
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        self.reject_refinement("^supported validation report is not canonical stable JSON$")
        path.write_bytes(original)
        manifest_path = self.supported_validation / validation.MANIFEST_RELATIVE_PATH
        manifest = artifacts.read_json(manifest_path)
        digest = manifest["outputSHA256s"][validation.RESULT_RELATIVE_PATH]
        manifest["outputSHA256s"][validation.RESULT_RELATIVE_PATH] = ("1" if digest[0] == "0" else "0") + digest[1:]
        artifacts.write_json(manifest_path, manifest)
        self.reject_refinement("^supported validation manifest does not reconstruct exactly$")
        self.assertEqual(path.read_bytes(), original)

    def test_rehashed_report_cannot_change_interpretation_or_ancestry(self):
        path = self.supported_validation / validation.RESULT_RELATIVE_PATH
        report = artifacts.read_json(path)
        report["overallClassification"] = "FABRICATED"
        artifacts.write_json(path, report)
        manifest_path = self.supported_validation / validation.MANIFEST_RELATIVE_PATH
        manifest = artifacts.read_json(manifest_path)
        manifest["outputSHA256s"][validation.RESULT_RELATIVE_PATH] = artifacts.sha256_bytes(path.read_bytes())
        artifacts.write_json(manifest_path, manifest)
        self.reject_refinement("^supported validation report does not reconstruct exactly$")

    def test_original_ancestry_and_immutable_supported_ledger_are_checked(self):
        manifest_path = self.coarse_validation / artifacts.validation.MANIFEST_RELATIVE_PATH
        original = manifest_path.read_bytes()
        manifest = artifacts.read_json(manifest_path)
        manifest["inputHashes"]["coarseInvestigationRecord"] = "0" * 64
        artifacts.write_json(manifest_path, manifest)
        self.reject_refinement("^validation manifest does not reconstruct exactly$")
        manifest_path.write_bytes(original)
        ledger = self.supported_record.parent / "stages" / "002-distributed-project.json"
        document = artifacts.read_json(ledger)
        document["result"]["projectCompletedWorkUnits"] += 1
        artifacts.write_json(ledger, document)
        self.reject_refinement("stage ledger")

    def test_supported_identities_and_domain_tampering_are_rejected(self):
        project = artifacts.read_json(self.supported_root / "project.json")
        path = self.supported_root / project["datasets"][1]["path"]
        original = path.read_bytes()
        dataset = artifacts.read_json(path)
        dataset["supportPolicyID"] = "unknown"
        artifacts.write_json(path, dataset)
        self.reject_refinement("supportPolicyID is invalid")
        path.write_bytes(original)
        dataset = artifacts.read_json(path)
        dataset["series"][0]["values"][0] += 0.1
        artifacts.write_json(path, dataset)
        self.reject_refinement("supported dataset domain and provenance does not reconstruct exactly")

    def test_report_markdown_manifest_and_artifact_set_are_verified(self):
        markdown = self.supported_validation / validation.MARKDOWN_RELATIVE_PATH
        original = markdown.read_bytes()
        markdown.write_bytes(original + b"tampered\n")
        self.reject_refinement("supported validation Markdown does not reconstruct exactly")
        markdown.write_bytes(original)
        extra = self.supported_validation / "unexpected.json"
        extra.write_text("{}")
        self.reject_refinement("supported validation artifact set is incomplete or unexpected")
        extra.unlink()
        (self.supported_validation / validation.MANIFEST_RELATIVE_PATH).unlink()
        self.reject_refinement("supported validation artifact set is incomplete or unexpected")

    def test_unsafe_nonfinite_overflow_and_collapsed_derived_axes(self):
        for start, step, positive, message in (
            (0.0, 0.0, False, "step must be strictly positive"),
            (0.0, -1.0, False, "step must be strictly positive"),
            (float("inf"), 1.0, False, "start must be finite"),
            (0.0, float("nan"), False, "step must be finite"),
            (1e16, 0.01, False, "values must be strictly increasing"),
            (1e308, 1e308, False, "axis value must be finite"),
            (0.0, 1.0, True, "values must be strictly positive"),
        ):
            with self.subTest(start=start, step=step), self.assertRaisesRegex((ValueError, RuntimeError), message):
                refinement._linear_axis(start, step, 41, "separation", positive=positive)
        parent = copy.deepcopy(self.parent)
        parent["morphologyGrid"]["negativeCenterAxis"]["step"] = 5e-324
        with self.assertRaisesRegex(ValueError, "negativeCenter derived step must be strictly positive"):
            refinement._derive_grid(parent, self.winner)
        winner = copy.deepcopy(self.winner)
        winner["parameters"]["positiveLogScale"] = 710.0
        with self.assertRaisesRegex(RuntimeError, "invalid exponentiated value"):
            refinement._derive_grid(self.parent, winner)

    def test_existing_output_paths_blind_identity_and_transactional_cleanup(self):
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "sentinel").write_bytes(b"untouched")
        self.reject_refinement("output root already exists", output_root=existing)
        self.assertEqual((existing / "sentinel").read_bytes(), b"untouched")
        self.reject_refinement("inside an input artifact directory", output_root=self.supported_validation / "new")
        linked = self.root / "linked"
        linked.symlink_to(self.supported_validation, target_is_directory=True)
        self.reject_refinement("symlink", supported_validation_root=linked)
        for identity in ("../unsafe", "generic-supported-project", "generic-morphology-coarse"):
            self.reject_refinement("project ID", project_id=identity)
        self.reject_refinement("identity", project_id="OGLE-generic")
        before = set(self.root.iterdir())
        original_write = refinement._atomic_write_bytes
        def fail_dataset(path, payload):
            if path.name == "ordered-doublet-refinement.json":
                raise OSError("injected dataset publication failure")
            original_write(path, payload)
        with patch.object(refinement, "_atomic_write_bytes", side_effect=fail_dataset):
            self.reject_refinement("^injected dataset publication failure$")
        self.assertEqual(set(self.root.iterdir()), before)
        with patch.object(refinement.Path, "rename", side_effect=OSError("injected rename publication failure")):
            self.reject_refinement("^injected rename publication failure$")
        self.assertEqual(set(self.root.iterdir()), before)


class NonfixedShapeTests(RefinementFixture):
    def _axes(self, **kwargs):
        axes = super()._axes(**kwargs)
        axes["LOG_SHAPE"].update(count=2, values=[-4.0, -3.0])
        return axes

    def test_verified_nonfixed_shapes_are_incompatible(self):
        self.reject_refinement("negativeLogShape must be fixed")


class FixedSeparationTests(RefinementFixture):
    def _axes(self, **kwargs):
        axes = super()._axes(**kwargs)
        axes["SEPARATION"]["count"] = 1
        return axes

    def test_fixed_separation_is_not_a_lower_searched_boundary(self):
        self.reject_refinement("separation must be at the lower searched boundary")


if __name__ == "__main__":
    unittest.main()
