import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_workloads.plugins.curve_grid import (
    FAMILY_ID,
    PLUGIN as CURVE_GRID_PLUGIN,
    _evaluate_candidate,
)
from tests.workflows.microlensing.test_prepare_residuals import (
    ResidualPreparationFixture,
    second_recenter_winning_result,
    write_residual_prepared_root,
)
from tests.workflows.microlensing.test_refine_grid import (
    FORBIDDEN_OUTPUT_TOKENS,
    read_json,
    rewrite_investigation_stage,
    serialized_tree,
    sha256_bytes,
    write_json,
)
from workflows.microlensing.prepare_residuals import (
    MANIFEST_RELATIVE_PATH as RESIDUAL_MANIFEST_RELATIVE_PATH,
    ResidualPreparationError,
    prepare_blind_microlensing_residuals,
)
from workflows.microlensing.residual_grid import (
    BUILD_MANIFEST_RELATIVE_PATH,
    CANDIDATES_PER_DATASET,
    CENTER_COUNT,
    CONTRACT_RELATIVE_PATH,
    DATASET_DIRECTORY,
    LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT,
    MINIMUM_IN_WINDOW_POSITIVE_WEIGHT_SAMPLES,
    PROJECT_RELATIVE_PATH,
    WORK_UNITS_PER_DATASET,
    ResidualGridBuildError,
    _admission_record,
    build_residual_grid_project,
)
from workflows.microlensing.second_recenter_grid import (
    DATASET_RELATIVE_PATH as SECOND_RECENTER_DATASET_RELATIVE_PATH,
)


PROJECT_ID = "openstar.generic-recovery-a.residual-grid.v1"
SYNTHETIC_CENTER = 2247.03


def write_residual_grid_prepared_root(root):
    write_residual_prepared_root(root)
    specifications = {
        "series-001": [
            SYNTHETIC_CENTER + (index - 4.5) * 0.01 for index in range(10)
        ],
        "series-002": [
            SYNTHETIC_CENTER - 1.0,
            *[
                SYNTHETIC_CENTER + (index - 3.0) * 0.01
                for index in range(7)
            ],
            SYNTHETIC_CENTER + 1.0,
        ],
        "series-003": [
            SYNTHETIC_CENTER - 1.0,
            *[
                SYNTHETIC_CENTER + (index - 3.5) * 0.01
                for index in range(8)
            ],
            SYNTHETIC_CENTER + 1.0,
        ],
    }
    manifest_path = root / "blind" / "preparation-manifest.json"
    manifest = read_json(manifest_path)
    for ordinal, record in enumerate(manifest["series"], start=1):
        series_path = root / "blind" / record["seriesFile"]
        series = read_json(series_path)
        coordinates = specifications[record["seriesID"]]
        series["coordinates"] = coordinates
        series["inverseVariances"] = [
            20.0 + ordinal + index for index in range(len(coordinates))
        ]
        series["values"] = [
            0.7
            + 0.04 * ordinal
            + 0.002 * index
            + (0.08 if index % 2 == 0 else -0.08)
            for index in range(len(coordinates))
        ]
        write_json(series_path, series)
        record["coordinateRange"] = {
            "maximum": max(coordinates),
            "minimum": min(coordinates),
        }
        record["sampleCount"] = len(coordinates)
        record["sha256"] = sha256_bytes(series_path.read_bytes())
    manifest["totalSampleCount"] = sum(
        record["sampleCount"] for record in manifest["series"]
    )
    write_json(manifest_path, manifest)


def replace_second_recenter_winner(record_path, best):
    def replace(stage):
        dataset_status = stage["result"]["datasets"][0]
        dataset_status["payload"]["best"] = dict(best)
        for key in (
            "bestAmplitude",
            "bestCenter",
            "bestGridIndex",
            "bestLogScale",
            "bestLogShape",
            "bestOffset",
            "bestWeightedResidualSumSquares",
        ):
            dataset_status[key] = best[key]

    rewrite_investigation_stage(record_path, 1, replace)


class ResidualGridFixture(ResidualPreparationFixture):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(strict=True)
        with patch(
            "tests.workflows.microlensing.test_prepare_residuals."
            "write_residual_prepared_root",
            side_effect=write_residual_grid_prepared_root,
        ):
            self.make_chain(self.root / "chain")
        self.residual = self.root / "residuals"
        prepare_blind_microlensing_residuals(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            refinement_project_root=self.refinement,
            refinement_investigation_record=self.refinement_investigation,
            first_recenter_project_root=self.first_recenter,
            first_recenter_investigation_record=(
                self.first_recenter_investigation
            ),
            second_recenter_project_root=self.second_recenter,
            second_recenter_investigation_record=(
                self.second_recenter_investigation
            ),
            output_root=self.residual,
        )

    def build(self, output_name="residual-grid"):
        output = self.root / output_name
        result = build_residual_grid_project(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            refinement_project_root=self.refinement,
            refinement_investigation_record=self.refinement_investigation,
            first_recenter_project_root=self.first_recenter,
            first_recenter_investigation_record=(
                self.first_recenter_investigation
            ),
            second_recenter_project_root=self.second_recenter,
            second_recenter_investigation_record=(
                self.second_recenter_investigation
            ),
            residual_root=self.residual,
            project_id=PROJECT_ID,
            output_root=output,
        )
        return output, result

    def assert_rejected(self, pattern=None, *, output_name="rejected"):
        context = (
            self.assertRaisesRegex(ResidualGridBuildError, pattern)
            if pattern
            else self.assertRaises(ResidualGridBuildError)
        )
        with context:
            self.build(output_name)


class ResidualGridSuccessTests(ResidualGridFixture):
    def test_success_is_deterministic_complete_and_identity_free(self):
        first, first_result = self.build("first-output")
        second, _ = self.build("second-output")
        self.assertEqual(serialized_tree(first), serialized_tree(second))

        manifest = read_json(first / BUILD_MANIFEST_RELATIVE_PATH)
        contract = read_json(first / CONTRACT_RELATIVE_PATH)
        project = read_json(first / PROJECT_RELATIVE_PATH)
        residual_manifest = read_json(
            self.residual / RESIDUAL_MANIFEST_RELATIVE_PATH
        )
        self.assertEqual(first_result["buildManifest"], manifest)
        self.assertEqual(FAMILY_ID, manifest["canonicalCurveFamilyID"])
        self.assertEqual(
            residual_manifest["parentProjectIDs"],
            manifest["parentProjectIDs"],
        )
        self.assertEqual(
            residual_manifest["parentInvestigationIDs"],
            manifest["parentInvestigationIDs"],
        )
        self.assertEqual(
            residual_manifest["parentArtifactHashes"],
            manifest["parentArtifactHashes"],
        )
        self.assertEqual(
            sha256_bytes(
                (self.residual / RESIDUAL_MANIFEST_RELATIVE_PATH).read_bytes()
            ),
            manifest["residualPreparationManifestFileSHA256"],
        )
        self.assertEqual(2, manifest["totalAdmittedDatasetCount"])
        self.assertEqual(
            ["series-001", "series-002", "series-003"],
            [record["genericSeriesID"] for record in manifest["admissionRecords"]],
        )
        self.assertEqual(
            ["ADMITTED", "EXCLUDED", "ADMITTED"],
            [
                record["admissionDecision"]
                for record in manifest["admissionRecords"]
            ],
        )
        self.assertEqual(
            [10, 7, 8],
            [
                record["inWindowPositiveWeightSampleCount"]
                for record in manifest["admissionRecords"]
            ],
        )
        self.assertEqual(2 * CANDIDATES_PER_DATASET, manifest["totalCandidateCount"])
        self.assertEqual(
            2 * WORK_UNITS_PER_DATASET,
            manifest["totalExpectedWorkUnitCount"],
        )
        expected_evaluations = (10 + 10) * CANDIDATES_PER_DATASET
        self.assertEqual(
            expected_evaluations,
            manifest["totalExpectedSampleCandidateEvaluationCount"],
        )
        self.assertEqual(
            [entry["id"] for entry in project["datasets"]],
            manifest["orderedAdmittedDatasetIDs"],
        )
        self.assertFalse(contract["admissionRule"]["residualValuesUsed"])

        serialized = b"".join(serialized_tree(first).values())
        for token in FORBIDDEN_OUTPUT_TOKENS:
            self.assertNotIn(token.encode("utf-8"), serialized)

    def test_exact_derived_axes_and_accounting(self):
        output, _ = self.build()
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        geometry = manifest["frozenGeometry"]
        core_width = math.exp(geometry["logScale"]) * math.exp(
            geometry["logShape"]
        )
        self.assertEqual(core_width, manifest["derivedCoreWidth"])
        axes = manifest["publishedAxes"]
        center_axis = axes["centerAxis"]
        self.assertEqual(CENTER_COUNT, center_axis["count"])
        self.assertEqual(
            geometry["center"] - 4.0 * core_width,
            center_axis["start"],
        )
        self.assertEqual(core_width / 16.0, center_axis["step"])
        self.assertEqual(
            geometry["center"],
            center_axis["start"] + 64 * center_axis["step"],
        )
        self.assertEqual(
            geometry["center"] + 4.0 * core_width,
            center_axis["start"]
            + (center_axis["count"] - 1) * center_axis["step"],
        )
        log_scale_axis = axes["logScaleAxis"]
        self.assertEqual(LOG_SCALE_COUNT, log_scale_axis["count"])
        self.assertEqual(
            geometry["logScale"] - math.log(16.0),
            log_scale_axis["start"],
        )
        self.assertEqual(math.log(16.0) / 16.0, log_scale_axis["step"])
        self.assertEqual(
            geometry["logScale"],
            log_scale_axis["start"]
            + (log_scale_axis["count"] - 1) * log_scale_axis["step"],
        )
        log_shape_axis = axes["logShapeAxis"]
        self.assertEqual(LOG_SHAPE_COUNT, log_shape_axis["count"])
        self.assertEqual(geometry["logShape"], log_shape_axis["start"])
        self.assertGreater(log_shape_axis["step"], 0.0)
        self.assertTrue(math.isfinite(log_shape_axis["step"]))

        for record in manifest["datasets"]:
            self.assertEqual(CANDIDATES_PER_DATASET, record["candidateCount"])
            self.assertEqual(
                WORK_UNITS_PER_DATASET, record["expectedWorkUnitCount"]
            )
            self.assertEqual(
                record["sampleCount"] * CANDIDATES_PER_DATASET,
                record["expectedSampleCandidateEvaluationCount"],
            )

    def test_admitted_datasets_retain_complete_signed_residual_series(self):
        output, _ = self.build()
        residual_manifest = read_json(
            self.residual / RESIDUAL_MANIFEST_RELATIVE_PATH
        )
        admitted_ids = ("series-001", "series-003")
        for ordinal, generic_series_id in enumerate(admitted_ids, start=1):
            source_record = next(
                record
                for record in residual_manifest["series"]
                if record["genericSeriesID"] == generic_series_id
            )
            source = read_json(self.residual / source_record["outputFile"])
            dataset = read_json(
                output
                / DATASET_DIRECTORY
                / f"residual-series-{ordinal:03d}.json"
            )
            self.assertEqual(source["coordinates"], dataset["coordinates"])
            self.assertEqual(
                source["inverseVariances"], dataset["inverseVariances"]
            )
            self.assertEqual(source["residualValues"], dataset["values"])
            self.assertEqual(source["sampleCount"], len(dataset["values"]))
            self.assertTrue(any(value < 0.0 for value in dataset["values"]))
            self.assertTrue(any(value > 0.0 for value in dataset["values"]))
            CURVE_GRID_PLUGIN.validate_dataset(dataset)

    def test_project_is_worker_compatible_multi_dataset_curve_grid(self):
        output, _ = self.build()
        project = read_json(output / PROJECT_RELATIVE_PATH)
        self.assertEqual(2, len(project["datasets"]))
        for reference in project["datasets"]:
            dataset = read_json(output / reference["path"])
            self.assertEqual(reference["id"], dataset["id"])
            self.assertEqual(FAMILY_ID, dataset["curveGrid"]["familyID"])
            CURVE_GRID_PLUGIN.validate_dataset(dataset)
            self.assertIsNotNone(_evaluate_candidate(dataset, 0))


class AdmissionRuleTests(unittest.TestCase):
    def test_exactly_eight_admit_and_seven_exclude(self):
        eight = _admission_record(
            "series-001",
            list(range(8)),
            [1.0] * 8,
            interval_minimum=0.0,
            interval_maximum=7.0,
        )
        seven = _admission_record(
            "series-002",
            list(range(7)),
            [1.0] * 7,
            interval_minimum=0.0,
            interval_maximum=7.0,
        )
        self.assertEqual(
            MINIMUM_IN_WINDOW_POSITIVE_WEIGHT_SAMPLES,
            eight["inWindowPositiveWeightSampleCount"],
        )
        self.assertEqual("ADMITTED", eight["admissionDecision"])
        self.assertEqual("EXCLUDED", seven["admissionDecision"])

    def test_zero_weights_do_not_count_and_endpoints_are_inclusive(self):
        coordinates = [-1.0, *[index / 10.0 for index in range(7)], 1.0]
        weights = [1.0, 0.0, *([1.0] * 6), 1.0]
        record = _admission_record(
            "series-001",
            coordinates,
            weights,
            interval_minimum=-1.0,
            interval_maximum=1.0,
        )
        self.assertEqual(8, record["inWindowPositiveWeightSampleCount"])
        self.assertEqual("ADMITTED", record["admissionDecision"])

    def test_admission_does_not_accept_or_depend_on_residual_values(self):
        coordinates = [float(index) for index in range(8)]
        weights = [1.0] * 8
        first_residuals = [-1000.0] * 8
        second_residuals = [1000.0] * 8
        self.assertNotEqual(first_residuals, second_residuals)
        first = _admission_record(
            "series-001",
            coordinates,
            weights,
            interval_minimum=0.0,
            interval_maximum=7.0,
        )
        second = _admission_record(
            "series-001",
            coordinates,
            weights,
            interval_minimum=0.0,
            interval_maximum=7.0,
        )
        self.assertEqual(first, second)

    def test_malformed_nonfinite_and_negative_inputs_are_rejected(self):
        cases = (
            ([], [], 0.0, 1.0),
            ([0.0], [1.0, 2.0], 0.0, 1.0),
            ([math.nan] * 8, [1.0] * 8, 0.0, 1.0),
            ([0.0] * 8, [math.inf] * 8, 0.0, 1.0),
            ([0.0] * 8, [-1.0] * 8, 0.0, 1.0),
        )
        for coordinates, weights, lower, upper in cases:
            with self.subTest(coordinates=coordinates, weights=weights):
                with self.assertRaises(ResidualGridBuildError):
                    _admission_record(
                        "series-001",
                        coordinates,
                        weights,
                        interval_minimum=lower,
                        interval_maximum=upper,
                    )


class ResidualGridRejectionTests(ResidualGridFixture):
    def _evaluated_second_winner(self, *, center_index, scale_index=8, shape_index=8):
        best = second_recenter_winning_result(
            self.second_recenter,
            center_index=center_index,
            scale_index=scale_index,
            shape_index=shape_index,
        )
        dataset = read_json(
            self.second_recenter / SECOND_RECENTER_DATASET_RELATIVE_PATH
        )
        evaluated = _evaluate_candidate(dataset, best["bestGridIndex"])
        if evaluated is None:
            raise AssertionError("synthetic second-recenter winner is invalid")
        best["bestOffset"] = evaluated.offset
        best["bestAmplitude"] = evaluated.amplitude
        best["bestWeightedResidualSumSquares"] = (
            evaluated.weighted_residual_sum_squares
        )
        return best

    def test_changed_interior_winner_is_rejected_by_convergence_gate(self):
        replace_second_recenter_winner(
            self.second_recenter_investigation,
            self._evaluated_second_winner(center_index=11),
        )
        self.assert_rejected("winners or objectives differ")

    def test_boundary_second_winner_is_rejected(self):
        replace_second_recenter_winner(
            self.second_recenter_investigation,
            self._evaluated_second_winner(center_index=0),
        )
        self.assert_rejected("grid boundary")

    def test_mutated_residual_and_manifest_are_rejected(self):
        targets = (
            self.residual / "series" / "residual-series-001.json",
            self.residual / RESIDUAL_MANIFEST_RELATIVE_PATH,
        )
        for ordinal, target in enumerate(targets, start=1):
            original = target.read_bytes()
            try:
                target.write_bytes(original + b" ")
                self.assert_rejected(output_name=f"mutated-residual-{ordinal}")
            finally:
                target.write_bytes(original)

    def test_mutated_parent_artifacts_across_ancestry_are_rejected(self):
        targets = (
            self.prepared / "blind" / "preparation-manifest.json",
            self.coarse / "build-manifest.json",
            self.refinement / "build-manifest.json",
            self.first_recenter / "build-manifest.json",
            self.second_recenter / "build-manifest.json",
        )
        for ordinal, target in enumerate(targets, start=1):
            original = target.read_bytes()
            try:
                target.write_bytes(original + b" ")
                self.assert_rejected(output_name=f"mutated-parent-{ordinal}")
            finally:
                target.write_bytes(original)

    def test_mutated_investigation_and_ledger_are_rejected(self):
        investigation = read_json(self.second_recenter_investigation)
        ledger = (
            self.second_recenter_investigation.parent
            / "stages"
            / f"{investigation['stages'][1]['id']}.json"
        )
        targets = (self.second_recenter_investigation, ledger)
        for ordinal, target in enumerate(targets, start=1):
            original = target.read_bytes()
            document = read_json(target)
            try:
                if target == self.second_recenter_investigation:
                    document["workflow_id"] = "wrong.workflow"
                else:
                    document["result"]["failedWorkUnits"] = 1
                write_json(target, document)
                self.assert_rejected(output_name=f"mutated-ledger-{ordinal}")
            finally:
                target.write_bytes(original)

    def test_malformed_and_nonfinite_residual_inputs_are_rejected(self):
        target = self.residual / "series" / "residual-series-001.json"
        original = target.read_bytes()
        cases = (
            lambda document: document.pop("residualValues"),
            lambda document: document["residualValues"].__setitem__(0, math.nan),
            lambda document: document["inverseVariances"].__setitem__(0, -1.0),
        )
        for ordinal, mutate in enumerate(cases, start=1):
            try:
                document = read_json(target)
                mutate(document)
                write_json(target, document, allow_nan=True)
                self.assert_rejected(output_name=f"malformed-residual-{ordinal}")
            finally:
                target.write_bytes(original)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_traversal_is_rejected(self):
        link = self.root / "residual-link"
        os.symlink(self.residual, link, target_is_directory=True)
        with self.assertRaises(ResidualGridBuildError):
            build_residual_grid_project(
                self.prepared,
                coarse_project_root=self.coarse,
                coarse_investigation_record=self.coarse_investigation,
                refinement_project_root=self.refinement,
                refinement_investigation_record=self.refinement_investigation,
                first_recenter_project_root=self.first_recenter,
                first_recenter_investigation_record=(
                    self.first_recenter_investigation
                ),
                second_recenter_project_root=self.second_recenter,
                second_recenter_investigation_record=(
                    self.second_recenter_investigation
                ),
                residual_root=link,
                project_id=PROJECT_ID,
                output_root=self.root / "symlink-output",
            )

    def test_existing_output_is_rejected_without_modification(self):
        output = self.root / "existing-output"
        output.mkdir()
        marker = output / "marker"
        marker.write_text("preserve", encoding="utf-8")
        self.assert_rejected("already exists", output_name="existing-output")
        self.assertEqual("preserve", marker.read_text(encoding="utf-8"))

    def test_identity_scan_rejects_contamination(self):
        from workflows.microlensing import residual_grid

        original_contract = residual_grid._contract

        def contaminated_contract(*args, **kwargs):
            contract = original_contract(*args, **kwargs)
            contract["leak"] = "OGLE"
            return contract

        with patch.object(residual_grid, "_contract", contaminated_contract):
            self.assert_rejected("identity", output_name="identity-output")

    def test_public_exception_translates_imported_builder_failures(self):
        with patch(
            "workflows.microlensing.residual_grid."
            "prepare_blind_microlensing_residuals",
            side_effect=ResidualPreparationError("synthetic parent failure"),
        ):
            with self.assertRaises(ResidualGridBuildError) as raised:
                self.build("translated-error")
        self.assertNotIsInstance(raised.exception, ResidualPreparationError)

    def test_public_exception_translates_curve_grid_validation_failures(self):
        with patch.object(
            CURVE_GRID_PLUGIN,
            "validate_dataset",
            side_effect=ValueError("synthetic dataset failure"),
        ):
            with self.assertRaises(ResidualGridBuildError):
                self.build("translated-plugin-error")

    def test_transactional_cleanup_after_publication_failure(self):
        from workflows.microlensing import residual_grid

        original_write = residual_grid._atomic_write_bytes
        call_count = 0

        def fail_after_first_write(path, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("synthetic publication failure")
            return original_write(path, payload)

        output_name = "transaction-output"
        with patch.object(
            residual_grid,
            "_atomic_write_bytes",
            side_effect=fail_after_first_write,
        ):
            self.assert_rejected(
                "publication failed", output_name=output_name
            )
        self.assertFalse((self.root / output_name).exists())
        self.assertEqual(
            [],
            list(self.root.glob(f".{output_name}.*")),
        )


if __name__ == "__main__":
    unittest.main()
