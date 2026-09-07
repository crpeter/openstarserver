"""Producer-shaped miniature artifacts; no grid search, coordinator, or network."""

import copy
from unittest.mock import patch

from openstar_workloads.plugins import morphology_grid as v1
from openstar_workloads.plugins import supported_morphology_grid as workload
from tests.workflows.microlensing import test_validate_anomaly_morphology_coarse_grid as fixtures
from workflows.microlensing import build_anomaly_morphology_coarse_grid as coarse_builder
from workflows.microlensing import build_supported_anomaly_morphology_grid as builder
from workflows.microlensing import validate_anomaly_morphology_coarse_grid as validation


class SupportedBuilderTests(fixtures.MorphologyValidationFixture):
    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        # Nonconstant data keep accepted candidate fits and WRSS nondegenerate.
        # The fixture records accepted index-zero results, without a grid search.
        document["residualValues"] = [0.01 * x * x + 0.003 * x * x * x for x in document["coordinates"]]
        return document

    def setUp(self):
        super().setUp()
        self.report()
        self.validation_root = self.root / "report"

    def supported_build(self, name="supported", **overrides):
        arguments = {
            "coarse_project_root": self.coarse,
            "coarse_investigation_record": self.investigation,
            "coarse_validation_root": self.validation_root,
            "project_id": "generic-supported-morphology", "output_root": self.root / name,
        }
        arguments.update(overrides)
        return builder.build_supported_anomaly_morphology_grid(self.morphology, **arguments)

    def reject_supported(self, pattern, **overrides):
        with self.assertRaisesRegex(builder.SupportedAnomalyMorphologyGridBuildError, pattern):
            self.supported_build("rejected", **overrides)
        self.assertFalse((self.root / "rejected").exists())

    def test_exact_domain_provenance_hashes_and_no_search(self):
        inputs = (self.morphology, self.coarse, self.investigation.parent, self.validation_root)
        before = {path: fixtures.serialized_tree(path) for path in inputs}
        evaluate = v1._evaluate_candidate
        with patch.object(v1, "_evaluate_candidate", wraps=evaluate) as reproduction, patch.object(
            v1, "_recompute_shard", side_effect=AssertionError("builder must not search")
        ), patch.object(workload, "_recompute", side_effect=AssertionError("builder must not search")), patch.object(
            workload.PLUGIN, "build_work_payloads", side_effect=AssertionError("copy existing sharding")
        ):
            built = self.supported_build()
        self.assertEqual(reproduction.call_count, 4)
        self.assertEqual({path: fixtures.serialized_tree(path) for path in inputs}, before)
        source = fixtures.read_json(self.coarse / "build-manifest.json")
        manifest = built["buildManifest"]
        self.assertEqual(len(built["datasets"]), 4)
        self.assertEqual(manifest["totalCandidateCount"], source["totalCoarseCandidateCount"])
        self.assertEqual(manifest["totalExpectedWorkUnitCount"], source["totalExpectedWorkUnitCount"])
        self.assertEqual(manifest["totalExpectedSampleCandidateEvaluationCount"], source["totalExpectedSampleCandidateEvaluationCount"])
        self.assertNotEqual(built["project"]["id"], source["projectID"])
        for new, reference, record, old_record in zip(built["datasets"], built["project"]["datasets"], manifest["datasets"], source["datasets"]):
            old = fixtures.read_json(self.coarse / old_record["outputFile"])
            self.assertNotEqual(new["id"], old["id"])
            preserved = copy.deepcopy(new)
            del preserved["supportedMorphologyProvenance"]
            del preserved["supportPolicyID"]
            for key in ("id", "workloadID", "datasetSchemaID", "payloadSchemaID", "resultSchemaID",
                        "executionContractID", "executionContractVersion"):
                preserved[key] = old[key]
            self.assertEqual(preserved, old)
            # Every array, explicit/linear axis and source sample-index list is
            # covered by full document equality, not just a count comparison.
            self.assertEqual(new["sourceSampleIndicesBySeries"], old["sourceSampleIndicesBySeries"])
            self.assertEqual(record["candidateCount"], old_record["coarseCandidateCount"])
            self.assertEqual(record["expectedWorkUnitCount"], old_record["expectedWorkUnitCount"])
            self.assertEqual(record["expectedWorkUnitCount"], (record["candidateCount"] + new["candidatesPerWorkUnit"] - 1) // new["candidatesPerWorkUnit"])
            old_grid = v1._validated_dataset(old).grid
            _, new_validated = workload._numeric.numerical_view(new)
            for index in (0, old_grid.total_candidates // 2, old_grid.total_candidates - 1):
                self.assertEqual(v1._candidate_parameters(old_grid, index), v1._candidate_parameters(new_validated.grid, index))
            self.assertEqual(new["supportPolicyID"], workload.SUPPORT_POLICY_ID)
            self.assertEqual(new["supportedMorphologyProvenance"]["sourceDatasetSHA256"], fixtures.sha256_bytes((self.coarse / reference["path"]).read_bytes()))
        provenance = manifest["provenance"]
        original_report_manifest = fixtures.read_json(self.validation_root / validation.MANIFEST_RELATIVE_PATH)
        self.assertEqual(provenance["inputHashes"], original_report_manifest["inputHashes"])
        self.assertEqual(provenance["parentHashes"], source["parentHashes"])
        for path, digest in provenance["coarseValidationArtifacts"].items():
            self.assertEqual(digest, fixtures.sha256_bytes((self.validation_root / path).read_bytes()))
        for path, digest in manifest["outputSHA256s"].items():
            self.assertEqual(digest, fixtures.sha256_bytes((self.root / "supported" / path).read_bytes()))
        body = copy.deepcopy(manifest)
        digest = body.pop("buildManifestSHA256")
        self.assertEqual(digest, fixtures.sha256_bytes(coarse_builder._canonical_compact_json_bytes(body)))
        self.assertEqual(set(fixtures.serialized_tree(self.root / "supported")), {
            "project.json", "build-manifest.json", *(item["path"] for item in built["project"]["datasets"]),
        })
        self.supported_build("same-output")
        self.assertEqual(fixtures.serialized_tree(self.root / "supported"), fixtures.serialized_tree(self.root / "same-output"))

    def test_report_tampering_even_with_rehashed_manifest_is_rejected(self):
        report_path = self.validation_root / validation.RESULT_RELATIVE_PATH
        report = fixtures.read_json(report_path)
        # Preserve the classification; alter a verified accepted winner instead.
        report["searches"][0]["acceptedWinner"]["weightedResidualSumSquares"] += 0.5
        fixtures.write_json(report_path, report)
        manifest_path = self.validation_root / validation.MANIFEST_RELATIVE_PATH
        manifest = fixtures.read_json(manifest_path)
        manifest["outputSHA256s"][validation.RESULT_RELATIVE_PATH] = fixtures.sha256_bytes(report_path.read_bytes())
        fixtures.write_json(manifest_path, manifest)
        self.reject_supported("validation report")

    def test_validation_manifest_hash_ancestry_and_ledger_links_are_verified(self):
        path = self.validation_root / validation.MANIFEST_RELATIVE_PATH
        original = fixtures.read_json(path)
        mutations = (
            ("coarseInvestigationRecord", "0" * 64),
            ("coarseStageLedgers", {}),
            ("coarseProjectArtifacts", {}),
            ("morphologyDatasets", {}),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                manifest = copy.deepcopy(original)
                manifest["inputHashes"][key] = value
                fixtures.write_json(path, manifest)
                self.reject_supported("validation manifest")
        manifest = copy.deepcopy(original)
        report_hash = manifest["outputSHA256s"][validation.RESULT_RELATIVE_PATH]
        manifest["outputSHA256s"][validation.RESULT_RELATIVE_PATH] = (
            ("1" if report_hash[0] == "0" else "0") + report_hash[1:]
        )
        fixtures.write_json(path, manifest)
        self.reject_supported("^validation manifest does not reconstruct exactly$")

        fixtures.write_json(path, original)
        report_path = self.validation_root / validation.RESULT_RELATIVE_PATH
        report_path.write_bytes(report_path.read_bytes() + b"\n")
        self.reject_supported("^validation report is not canonical stable JSON$")

    def test_source_arrays_are_verified_before_publication(self):
        project = fixtures.read_json(self.coarse / "project.json")
        path = self.coarse / project["datasets"][0]["path"]
        document = fixtures.read_json(path)
        document["series"][0]["values"][0] += 1.0
        fixtures.write_json(path, document)
        self.reject_supported(".")

    def test_investigation_coverage_and_ledger_tampering_are_rejected(self):
        original = fixtures.read_json(self.investigation)
        record = copy.deepcopy(original)
        record["stages"][1]["result"]["projectCompletedWorkUnits"] -= 1
        fixtures.write_record(self.investigation, record)
        self.reject_supported(".")
        fixtures.write_record(self.investigation, original)
        ledger = self.investigation.parent / "stages" / f"{original['stages'][1]['id']}.json"
        record = fixtures.read_json(ledger)
        record["result"]["projectTotalWorkUnits"] += 1
        fixtures.write_json(ledger, record)
        self.reject_supported(".")

    def test_existing_output_symlinks_input_nesting_and_project_identity(self):
        self.supported_build()
        original = fixtures.serialized_tree(self.root / "supported")
        with self.assertRaisesRegex(builder.SupportedAnomalyMorphologyGridBuildError, "already exists"):
            self.supported_build()
        self.assertEqual(fixtures.serialized_tree(self.root / "supported"), original)
        self.reject_supported("inside an input", output_root=self.coarse / "child")
        self.reject_supported("must differ", project_id="generic-morphology-coarse")
        self.reject_supported("malformed", project_id="../unsafe")
        linked = self.root / "linked-report"
        linked.symlink_to(self.validation_root, target_is_directory=True)
        self.reject_supported("symlink", coarse_validation_root=linked)

    def test_publication_failure_cleans_staging_and_preserves_inputs(self):
        before = fixtures.serialized_tree(self.coarse)
        with patch.object(builder, "_atomic_write_bytes", side_effect=OSError("injected failure")):
            self.reject_supported("injected failure")
        self.assertEqual(fixtures.serialized_tree(self.coarse), before)
        self.assertEqual(list(self.root.glob(".rejected.*")), [])

    def test_cli_requires_all_six_arguments(self):
        values = ["--morphology-root", str(self.morphology), "--coarse-project-root", str(self.coarse),
                  "--coarse-investigation-record", str(self.investigation), "--coarse-validation-root", str(self.validation_root),
                  "--project-id", "generic-supported-morphology", "--output-root", str(self.root / "supported")]
        self.assertEqual(set(vars(builder._parser().parse_args(values))), {
            "morphology_root", "coarse_project_root", "coarse_investigation_record", "coarse_validation_root", "project_id", "output_root",
        })
