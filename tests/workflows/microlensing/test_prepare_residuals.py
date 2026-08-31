import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_workloads.plugins.curve_grid import (
    DATASET_SCHEMA_ID,
    FAMILY_ID,
    PAYLOAD_SCHEMA_ID,
    RESULT_SCHEMA_ID,
    WORKLOAD_ID,
    _evaluate_candidate,
)
from tests.workflows.microlensing.test_recenter_grid import (
    RECENTERED_PROJECT_ID,
    write_refinement_investigation,
)
from tests.workflows.microlensing.test_refine_grid import (
    COARSE_PROJECT_ID,
    FORBIDDEN_OUTPUT_TOKENS,
    REFINEMENT_PROJECT_ID,
    read_json,
    rewrite_investigation_stage,
    serialized_tree,
    sha256_bytes,
    stage_record,
    write_investigation as write_coarse_investigation,
    write_json,
    write_prepared_root,
)
from tests.workflows.microlensing.test_second_recenter_grid import (
    SECOND_RECENTER_PROJECT_ID,
    first_recenter_winning_result,
    replace_first_recenter_winner,
    write_first_recenter_investigation,
)
from workflows.microlensing.coarse_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as COARSE_BUILD_MANIFEST_RELATIVE_PATH,
    build_coarse_grid_project,
)
from workflows.microlensing.prepare_residuals import (
    MANIFEST_RELATIVE_PATH,
    RESIDUAL_PREPARATION_CONTRACT_ID,
    RESIDUAL_SERIES_SCHEMA_ID,
    ResidualPreparationError,
    _canonical_curve_basis,
    _fit_series,
    prepare_blind_microlensing_residuals,
)
from workflows.microlensing.recenter_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
    build_recentered_grid_project,
)
from workflows.microlensing.refine_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
    build_refinement_grid_project,
)
from workflows.microlensing.second_recenter_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as SECOND_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
    CANDIDATES_PER_WORK_UNIT,
    DATASET_RELATIVE_PATH as SECOND_RECENTER_DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT,
    PROJECT_RELATIVE_PATH as SECOND_RECENTER_PROJECT_RELATIVE_PATH,
    TOTAL_CANDIDATE_COUNT,
    build_second_recenter_grid_project,
)


SECOND_RECENTER_INVESTIGATION_ID = "generic-second-recenter-investigation"


def rewrite_prepared_series(prepared, series_id, mutate, *, allow_nan=False):
    manifest_path = prepared / "blind" / "preparation-manifest.json"
    manifest = read_json(manifest_path)
    record = next(item for item in manifest["series"] if item["seriesID"] == series_id)
    series_path = prepared / "blind" / record["seriesFile"]
    series = read_json(series_path)
    mutate(series)
    write_json(series_path, series, allow_nan=allow_nan)
    record["sha256"] = sha256_bytes(series_path.read_bytes())
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in series["coordinates"]
    ):
        record["coordinateRange"] = {
            "maximum": max(series["coordinates"]),
            "minimum": min(series["coordinates"]),
        }
    write_json(manifest_path, manifest, allow_nan=allow_nan)


def write_residual_prepared_root(root):
    write_prepared_root(root)
    for ordinal, series_id in enumerate(
        ("series-001", "series-002", "series-003"), start=1
    ):
        def mutate(series, ordinal=ordinal):
            count = len(series["coordinates"])
            series["coordinates"] = [
                2237.0 + ordinal + 2.5 * index for index in range(count)
            ]
            series["values"] = [
                0.75 + 0.03 * ordinal + 0.04 * index for index in range(count)
            ]

        rewrite_prepared_series(root, series_id, mutate)


def second_recenter_winning_result(
    second_recenter_root,
    *,
    center_index=10,
    scale_index=8,
    shape_index=8,
    objective=9.5,
):
    dataset = read_json(
        second_recenter_root / SECOND_RECENTER_DATASET_RELATIVE_PATH
    )
    grid = dataset["curveGrid"]
    grid_index = (
        (center_index * LOG_SCALE_COUNT + scale_index) * LOG_SHAPE_COUNT
        + shape_index
    )
    shard_start = (grid_index // CANDIDATES_PER_WORK_UNIT) * (
        CANDIDATES_PER_WORK_UNIT
    )
    shard_count = min(
        CANDIDATES_PER_WORK_UNIT,
        TOTAL_CANDIDATE_COUNT - shard_start,
    )
    return {
        "bestAmplitude": 1.25,
        "bestCenter": (
            grid["centerAxis"]["start"]
            + center_index * grid["centerAxis"]["step"]
        ),
        "bestGridIndex": grid_index,
        "bestLogScale": (
            grid["logScaleAxis"]["start"]
            + scale_index * grid["logScaleAxis"]["step"]
        ),
        "bestLogShape": (
            grid["logShapeAxis"]["start"]
            + shape_index * grid["logShapeAxis"]["step"]
        ),
        "bestOffset": -0.25,
        "bestWeightedResidualSumSquares": objective,
        "evaluatedCandidateCount": shard_count,
        "familyID": FAMILY_ID,
        "gridCount": shard_count,
        "gridStartIndex": shard_start,
        "invalidCandidateCount": 0,
    }


def write_second_recenter_investigation(root, second_recenter_root, *, best=None):
    investigation_root = root / SECOND_RECENTER_INVESTIGATION_ID
    stages_root = investigation_root / "stages"
    stages_root.mkdir(parents=True)
    project_path = (
        second_recenter_root / SECOND_RECENTER_PROJECT_RELATIVE_PATH
    ).resolve()
    project_hash = sha256_bytes(project_path.read_bytes())
    dataset = read_json(
        second_recenter_root / SECOND_RECENTER_DATASET_RELATIVE_PATH
    )
    if best is None:
        best = second_recenter_winning_result(second_recenter_root)
        evaluated = _evaluate_candidate(dataset, best["bestGridIndex"])
        if evaluated is None:
            raise AssertionError("synthetic second-recenter winner is invalid")
        best["bestOffset"] = evaluated.offset
        best["bestAmplitude"] = evaluated.amplitude
        best["bestWeightedResidualSumSquares"] = (
            evaluated.weighted_residual_sum_squares
        )
    best = dict(best)
    contributions = {"generic-node-a": 48, "generic-node-b": 47}
    dataset_status = {
        "assignedWorkUnits": 0,
        "bestAmplitude": best["bestAmplitude"],
        "bestCenter": best["bestCenter"],
        "bestGridIndex": best["bestGridIndex"],
        "bestLogScale": best["bestLogScale"],
        "bestLogShape": best["bestLogShape"],
        "bestOffset": best["bestOffset"],
        "bestWeightedResidualSumSquares": best[
            "bestWeightedResidualSumSquares"
        ],
        "completedCandidateCount": TOTAL_CANDIDATE_COUNT,
        "completedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "coverageComplete": True,
        "curveGridStatus": "CURVE_GRID_COMPLETE",
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "failedWorkUnits": 0,
        "familyID": FAMILY_ID,
        "id": dataset["id"],
        "nodeContributions": dict(contributions),
        "pendingWorkUnits": 0,
        "payload": {"best": best},
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "progress": 1.0,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "totalCandidateCount": TOTAL_CANDIDATE_COUNT,
        "totalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "workloadID": WORKLOAD_ID,
        "workloadStatus": "CURVE_GRID_COMPLETE",
    }
    run_result = {
        "assignedWorkUnits": 0,
        "completedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "datasets": [dataset_status],
        "failedWorkUnits": 0,
        "nodeContributions": dict(contributions),
        "pendingWorkUnits": 0,
        "projectAssignedWorkUnits": 0,
        "projectCompletedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "projectFailedWorkUnits": 0,
        "projectID": SECOND_RECENTER_PROJECT_ID,
        "projectPath": str(project_path),
        "projectPendingWorkUnits": 0,
        "projectProgress": 1.0,
        "projectTotalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "status": "COMPLETE",
        "totalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "workloadID": WORKLOAD_ID,
    }
    prepare_parameters = {"projectPath": str(project_path)}
    run_parameters = {
        "projectManifestSha256": project_hash,
        "projectPath": str(project_path),
    }
    terminal_parameters = {"expectedProjectID": SECOND_RECENTER_PROJECT_ID}
    prepare = stage_record(
        stage_id="001-prepare-second-recenter",
        handler_id="local.project.prepare",
        triggered_by=None,
        parameters=prepare_parameters,
        result={
            "projectManifestSha256": project_hash,
            "projectPath": str(project_path),
        },
        input_hashes={"projectManifest": project_hash},
        project_ids=(),
        next_stage={
            "handler_id": "openstar.project.run",
            "id": "002-run-second-recenter",
            "parameters": run_parameters,
            "triggered_by_stage_id": "001-prepare-second-recenter",
        },
        stop=False,
    )
    run = stage_record(
        stage_id="002-run-second-recenter",
        handler_id="openstar.project.run",
        triggered_by="001-prepare-second-recenter",
        parameters=run_parameters,
        result=run_result,
        input_hashes={"projectManifest": project_hash},
        project_ids=(SECOND_RECENTER_PROJECT_ID,),
        next_stage={
            "handler_id": "generic.project.terminal-check",
            "id": "003-terminal-second-recenter",
            "parameters": terminal_parameters,
            "triggered_by_stage_id": "002-run-second-recenter",
        },
        stop=False,
        node_contributions=contributions,
    )
    terminal = stage_record(
        stage_id="003-terminal-second-recenter",
        handler_id="generic.project.terminal-check",
        triggered_by="002-run-second-recenter",
        parameters=terminal_parameters,
        result={
            "completedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
            "failedWorkUnits": 0,
            "passed": True,
            "projectID": SECOND_RECENTER_PROJECT_ID,
            "rule": "projectID matches and completed+failed == total",
            "totalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        },
        input_hashes={},
        project_ids=(SECOND_RECENTER_PROJECT_ID,),
        next_stage=None,
        stop=True,
    )
    stages = [prepare, run, terminal]
    investigation = {
        "created_at": "2026-08-31T00:00:00+00:00",
        "id": SECOND_RECENTER_INVESTIGATION_ID,
        "metadata": {
            "coordinator": "http://127.0.0.1:8080",
            "projectPath": str(project_path),
        },
        "stages": stages,
        "status": "COMPLETE",
        "updated_at": "2026-08-31T00:00:02+00:00",
        "workflow_id": "openstar.workflow.project-smoke.v1",
        "workflow_version": "20.0",
    }
    record_path = investigation_root / "investigation.json"
    write_json(record_path, investigation)
    for stage in stages:
        write_json(stages_root / f"{stage['id']}.json", stage)
    return record_path


class ResidualPreparationFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.make_chain(self.root / "chain")

    def make_chain(self, root):
        self.prepared = root / "prepared"
        self.prepared.mkdir(parents=True)
        write_residual_prepared_root(self.prepared)
        self.coarse = root / "coarse"
        build_coarse_grid_project(
            self.prepared,
            project_id=COARSE_PROJECT_ID,
            output_root=self.coarse,
        )
        self.coarse_investigation = write_coarse_investigation(
            root / "coarse-investigations", self.coarse
        )
        self.refinement = root / "refinement"
        build_refinement_grid_project(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            project_id=REFINEMENT_PROJECT_ID,
            output_root=self.refinement,
        )
        self.refinement_investigation = write_refinement_investigation(
            root / "refinement-investigations", self.refinement
        )
        self.first_recenter = root / "first-recenter"
        build_recentered_grid_project(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            refinement_project_root=self.refinement,
            refinement_investigation_record=self.refinement_investigation,
            project_id=RECENTERED_PROJECT_ID,
            output_root=self.first_recenter,
        )
        first_best = first_recenter_winning_result(
            self.first_recenter,
            center_index=10,
            scale_index=8,
            shape_index=0,
            objective=9.5,
        )
        first_dataset = read_json(
            self.first_recenter / "datasets" / "primary-series.json"
        )
        first_evaluated = _evaluate_candidate(
            first_dataset, first_best["bestGridIndex"]
        )
        if first_evaluated is None:
            raise AssertionError("synthetic first-recenter winner is invalid")
        first_best["bestOffset"] = first_evaluated.offset
        first_best["bestAmplitude"] = first_evaluated.amplitude
        first_best["bestWeightedResidualSumSquares"] = (
            first_evaluated.weighted_residual_sum_squares
        )
        self.first_recenter_investigation = write_first_recenter_investigation(
            root / "first-recenter-investigations",
            self.first_recenter,
            best=first_best,
        )
        self.second_recenter = root / "second-recenter"
        build_second_recenter_grid_project(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            refinement_project_root=self.refinement,
            refinement_investigation_record=self.refinement_investigation,
            first_recenter_project_root=self.first_recenter,
            first_recenter_investigation_record=self.first_recenter_investigation,
            project_id=SECOND_RECENTER_PROJECT_ID,
            output_root=self.second_recenter,
        )
        self.second_recenter_investigation = write_second_recenter_investigation(
            root / "second-recenter-investigations", self.second_recenter
        )

    def build(self, output_name="residuals"):
        output = self.root / output_name
        result = prepare_blind_microlensing_residuals(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            refinement_project_root=self.refinement,
            refinement_investigation_record=self.refinement_investigation,
            first_recenter_project_root=self.first_recenter,
            first_recenter_investigation_record=self.first_recenter_investigation,
            second_recenter_project_root=self.second_recenter,
            second_recenter_investigation_record=(
                self.second_recenter_investigation
            ),
            output_root=output,
        )
        return output, result

    def assert_rejected(self, pattern=None, *, output_name="rejected"):
        context = (
            self.assertRaisesRegex(ResidualPreparationError, pattern)
            if pattern
            else self.assertRaises(ResidualPreparationError)
        )
        with context:
            self.build(output_name)


class ResidualPreparationSuccessTests(ResidualPreparationFixture):
    def test_success_is_deterministic_complete_and_identity_free(self):
        first, result = self.build("first-output")
        second, _ = self.build("second-output")
        self.assertEqual(serialized_tree(first), serialized_tree(second))
        manifest = read_json(first / MANIFEST_RELATIVE_PATH)
        self.assertEqual(result["manifest"], manifest)
        self.assertEqual(
            ["series-001", "series-002", "series-003"],
            manifest["orderedGenericSeriesIDs"],
        )
        self.assertEqual(3, manifest["totalSeriesCount"])
        self.assertEqual(19, manifest["totalSampleCount"])
        self.assertEqual(
            RESIDUAL_PREPARATION_CONTRACT_ID, manifest["contractID"]
        )
        self.assertTrue(
            manifest["convergenceEvidence"]["secondRecenterInteriorOnEveryAxis"]
        )
        self.assertTrue(manifest["convergenceEvidence"]["exactEquality"])
        self.assertEqual(
            [
                "series/residual-series-001.json",
                "series/residual-series-002.json",
                "series/residual-series-003.json",
            ],
            [record["outputFile"] for record in manifest["series"]],
        )
        selected_residual = read_json(
            first / "series" / "residual-series-002.json"
        )
        accepted = manifest["verifiedSecondRecenterWinner"]
        self.assertEqual(accepted["bestOffset"], selected_residual["fittedOffset"])
        self.assertEqual(
            accepted["bestAmplitude"],
            selected_residual["fittedAmplitude"],
        )
        self.assertEqual(
            accepted["bestWeightedResidualSumSquares"],
            selected_residual["fitDiagnostics"][
                "weightedResidualSumSquares"
            ],
        )
        rendered = b"\n".join(
            path.read_bytes() for path in sorted(first.rglob("*.json"))
        ).decode("utf-8")
        for token in FORBIDDEN_OUTPUT_TOKENS:
            self.assertNotIn(token.casefold(), rendered.casefold())

    def test_series_preserve_inputs_and_residuals_use_observed_minus_model(self):
        output, _ = self.build()
        for ordinal, series_id in enumerate(
            ("series-001", "series-002", "series-003"), start=1
        ):
            source = read_json(
                self.prepared / "blind" / "series" / f"{series_id}.json"
            )
            residual = read_json(
                output / "series" / f"residual-series-{ordinal:03d}.json"
            )
            self.assertEqual(
                RESIDUAL_SERIES_SCHEMA_ID,
                residual["residualSeriesSchemaID"],
            )
            self.assertEqual(source["coordinates"], residual["coordinates"])
            self.assertEqual(
                source["inverseVariances"], residual["inverseVariances"]
            )
            self.assertEqual(source["values"], residual["observedValues"])
            for observed, model, difference in zip(
                residual["observedValues"],
                residual["modelValues"],
                residual["residualValues"],
            ):
                self.assertEqual(observed - model, difference)

    def test_canonical_basis_and_fit_match_curve_grid_evaluator(self):
        center = 3.0
        requested_scale = 1.7
        requested_shape = 0.2
        log_scale = math.log(requested_scale)
        log_shape = math.log(requested_shape)
        scale = math.exp(log_scale)
        shape = math.exp(log_shape)
        coordinates = [-1.0, 0.5, 2.0, 3.0, 4.5, 7.0]
        weights = [1.0, 2.0, 3.0, 4.0, 2.5, 1.5]
        basis = _canonical_curve_basis(
            coordinates, center=center, scale=scale, shape=shape
        )
        independently_calculated = []
        for coordinate in coordinates:
            u_squared = shape * shape + ((coordinate - center) / scale) ** 2
            independently_calculated.append(
                (u_squared + 2.0)
                / (math.sqrt(u_squared) * math.sqrt(u_squared + 4.0))
            )
        self.assertEqual(tuple(independently_calculated), basis)
        known_offset = -0.4
        known_amplitude = 1.75
        exact_values = [
            known_offset + known_amplitude * item for item in basis
        ]
        exact_fit = _fit_series(
            coordinates,
            exact_values,
            weights,
            center=center,
            scale=scale,
            shape=shape,
        )
        self.assertAlmostEqual(known_offset, exact_fit.offset, places=12)
        self.assertAlmostEqual(known_amplitude, exact_fit.amplitude, places=12)
        noise = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01]
        values = [
            known_offset + known_amplitude * item + noise[index]
            for index, item in enumerate(basis)
        ]
        fit = _fit_series(
            coordinates,
            values,
            weights,
            center=center,
            scale=scale,
            shape=shape,
        )
        dataset = {
            "coordinates": coordinates,
            "curveGrid": {
                "candidatesPerWorkUnit": 1,
                "centerAxis": {"count": 1, "start": center, "step": 1.0},
                "familyID": FAMILY_ID,
                "logScaleAxis": {
                    "count": 1,
                    "start": log_scale,
                    "step": 1.0,
                },
                "logShapeAxis": {
                    "count": 1,
                    "start": log_shape,
                    "step": 1.0,
                },
            },
            "datasetSchemaID": DATASET_SCHEMA_ID,
            "id": "synthetic.dataset",
            "inverseVariances": weights,
            "values": values,
        }
        plugin = _evaluate_candidate(dataset, 0)
        self.assertIsNotNone(plugin)
        self.assertAlmostEqual(plugin.offset, fit.offset, places=13)
        self.assertAlmostEqual(plugin.amplitude, fit.amplitude, places=13)
        self.assertAlmostEqual(
            plugin.weighted_residual_sum_squares,
            fit.weighted_residual_sum_squares,
            places=13,
        )

    def test_weighted_fit_wrss_and_maximum_tie_are_independent(self):
        coordinates = [-2.0, -1.0, 1.0, 2.0]
        basis = _canonical_curve_basis(
            coordinates, center=0.0, scale=1.0, shape=0.25
        )
        offset = 0.5
        amplitude = -2.0
        values = [offset + amplitude * item for item in basis]
        values[0] += 0.25
        values[3] -= 0.25
        weights = [4.0, 1.0, 1.0, 4.0]
        fit = _fit_series(
            coordinates,
            values,
            weights,
            center=0.0,
            scale=1.0,
            shape=0.25,
        )
        independent_wrss = sum(
            weight * residual * residual
            for weight, residual in zip(weights, fit.residual_values)
        )
        self.assertEqual(independent_wrss, fit.weighted_residual_sum_squares)
        maximum = max(
            abs(residual) * math.sqrt(weight)
            for residual, weight in zip(fit.residual_values, weights)
        )
        expected_index = next(
            index
            for index, (residual, weight) in enumerate(
                zip(fit.residual_values, weights)
            )
            if abs(residual) * math.sqrt(weight) == maximum
        )
        self.assertEqual(
            expected_index,
            fit.maximum_absolute_standardized_residual_index,
        )
        self.assertEqual(maximum, fit.maximum_absolute_standardized_residual)


class ResidualPreparationConvergenceRejectionTests(ResidualPreparationFixture):
    def test_changed_geometry_changed_wrss_and_boundary_are_rejected(self):
        cases = (
            ("geometry", {"center_index": 11}, "winners or objectives differ"),
            ("objective", {"objective": 9.5001}, "winners or objectives differ"),
            ("boundary", {"shape_index": 0}, "grid boundary"),
        )
        for name, changes, pattern in cases:
            with self.subTest(case=name):
                self.make_chain(self.root / f"convergence-{name}")
                best = second_recenter_winning_result(
                    self.second_recenter, **changes
                )
                replace_first_recenter_winner(
                    self.second_recenter_investigation, best
                )
                self.assert_rejected(pattern, output_name=f"rejected-{name}")

    def test_aggregate_disagreement_incomplete_coverage_and_failure_are_rejected(self):
        def disagreement(stage):
            stage["result"]["datasets"][0]["bestCenter"] += 0.001

        def incomplete(stage):
            stage["result"]["datasets"][0]["coverageComplete"] = False

        def failed(stage):
            stage["result"]["projectFailedWorkUnits"] = 1

        for name, mutate in (
            ("disagreement", disagreement),
            ("incomplete", incomplete),
            ("failed", failed),
        ):
            with self.subTest(case=name):
                self.make_chain(self.root / f"coverage-{name}")
                rewrite_investigation_stage(
                    self.second_recenter_investigation, 1, mutate
                )
                self.assert_rejected(output_name=f"coverage-output-{name}")


class ResidualPreparationIntegrityRejectionTests(ResidualPreparationFixture):
    def test_wrong_stage_order_workflow_project_id_and_hash_are_rejected(self):
        def reorder(record):
            record["stages"][0], record["stages"][1] = (
                record["stages"][1], record["stages"][0]
            )

        def workflow(record):
            record["workflow_id"] = "openstar.workflow.other.v1"

        cases = (("order", reorder), ("workflow", workflow))
        for name, mutate in cases:
            with self.subTest(case=name):
                self.make_chain(self.root / f"record-{name}")
                record = read_json(self.second_recenter_investigation)
                mutate(record)
                write_json(self.second_recenter_investigation, record)
                self.assert_rejected(output_name=f"record-output-{name}")

        self.make_chain(self.root / "wrong-project")
        project_path = self.second_recenter / SECOND_RECENTER_PROJECT_RELATIVE_PATH
        project = read_json(project_path)
        project["id"] = "openstar.generic.wrong-project.v1"
        write_json(project_path, project)
        self.assert_rejected(output_name="wrong-project-output")

        self.make_chain(self.root / "wrong-hash")
        manifest_path = (
            self.second_recenter / SECOND_RECENTER_BUILD_MANIFEST_RELATIVE_PATH
        )
        manifest = read_json(manifest_path)
        manifest["outputProjectSHA256"] = "0" * 64
        write_json(manifest_path, manifest)
        self.assert_rejected(output_name="wrong-hash-output")

    def test_mutated_artifact_at_every_ancestry_level_is_rejected(self):
        cases = (
            ("coarse", "coarse", COARSE_BUILD_MANIFEST_RELATIVE_PATH),
            ("refinement", "refinement", REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH),
            (
                "first-recenter",
                "first_recenter",
                FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
            ),
            (
                "second-recenter",
                "second_recenter",
                SECOND_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
            ),
        )
        for name, root_name, relative_path in cases:
            with self.subTest(parent=name):
                self.make_chain(self.root / f"artifact-{name}")
                path = getattr(self, root_name) / relative_path
                value = read_json(path)
                value[next(iter(value))] = "mutated"
                write_json(path, value)
                self.assert_rejected(output_name=f"artifact-output-{name}")

    def test_mutated_ledger_at_every_ancestry_level_is_rejected(self):
        cases = (
            ("coarse", "coarse_investigation"),
            ("refinement", "refinement_investigation"),
            ("first-recenter", "first_recenter_investigation"),
            ("second-recenter", "second_recenter_investigation"),
        )
        for name, record_name in cases:
            with self.subTest(parent=name):
                self.make_chain(self.root / f"ledger-{name}")
                record_path = getattr(self, record_name)
                record = read_json(record_path)
                stage_id = record["stages"][1]["id"]
                ledger_path = record_path.parent / "stages" / f"{stage_id}.json"
                ledger = read_json(ledger_path)
                ledger["result"]["projectCompletedWorkUnits"] -= 1
                write_json(ledger_path, ledger)
                self.assert_rejected(output_name=f"ledger-output-{name}")

    def test_missing_and_malformed_prepared_series_are_rejected(self):
        self.make_chain(self.root / "missing-series")
        (self.prepared / "blind" / "series" / "series-001.json").unlink()
        self.assert_rejected(output_name="missing-series-output")

        self.make_chain(self.root / "malformed-series")
        path = self.prepared / "blind" / "series" / "series-001.json"
        path.write_bytes(b"{")
        self.assert_rejected(output_name="malformed-series-output")


class ResidualPreparationNumericalAndPublicationTests(ResidualPreparationFixture):
    def test_nonfinite_negative_insufficient_and_singular_inputs_are_rejected(self):
        with self.assertRaises(ResidualPreparationError):
            _fit_series(
                [0.0, 1.0],
                [1.0, math.inf],
                [1.0, 1.0],
                center=0.0,
                scale=1.0,
                shape=0.1,
            )
        with self.assertRaisesRegex(ResidualPreparationError, "nonnegative"):
            _fit_series(
                [0.0, 1.0],
                [1.0, 2.0],
                [1.0, -1.0],
                center=0.0,
                scale=1.0,
                shape=0.1,
            )
        with self.assertRaisesRegex(ResidualPreparationError, "insufficient"):
            _fit_series(
                [0.0, 1.0],
                [1.0, 2.0],
                [1.0, 0.0],
                center=0.0,
                scale=1.0,
                shape=0.1,
            )
        with self.assertRaisesRegex(ResidualPreparationError, "singular"):
            _fit_series(
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [1.0, 1.0, 1.0],
                center=1.0,
                scale=1.0,
                shape=0.1,
            )

    def test_existing_output_and_symlink_traversal_are_rejected(self):
        (self.root / "existing").mkdir()
        self.assert_rejected("already exists", output_name="existing")
        target = self.root / "real-parent"
        target.mkdir()
        link = self.root / "linked-parent"
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        with self.assertRaisesRegex(ResidualPreparationError, "traverses a symlink"):
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
                output_root=link / "output",
            )

    def test_identity_leakage_and_transaction_failure_clean_staging(self):
        original_contract = {
            "contractID": RESIDUAL_PREPARATION_CONTRACT_ID,
            "identityIsolationStatement": "OGLE",
            "maximumResidualTieRule": "earliest",
            "modelScopeStatement": "none",
        }
        with patch(
            "workflows.microlensing.prepare_residuals._residual_contract",
            return_value=original_contract,
        ):
            self.assert_rejected(output_name="identity-output")

        output_name = "transaction-output"
        with patch(
            "workflows.microlensing.prepare_residuals._atomic_write_bytes",
            side_effect=OSError("synthetic write failure"),
        ):
            self.assert_rejected(
                "atomic output publication failed", output_name=output_name
            )
        self.assertFalse((self.root / output_name).exists())
        self.assertEqual([], list(self.root.glob(f".{output_name}.*")))

    def test_imported_builder_exceptions_never_escape(self):
        from workflows.microlensing.coarse_grid import CoarseGridBuildError

        with patch(
            "workflows.microlensing.prepare_residuals._verify_blind_preparation",
            side_effect=CoarseGridBuildError("synthetic inherited failure"),
        ):
            with self.assertRaises(ResidualPreparationError) as context:
                self.build("translated-output")
        self.assertNotIsInstance(context.exception, CoarseGridBuildError)


if __name__ == "__main__":
    unittest.main()
