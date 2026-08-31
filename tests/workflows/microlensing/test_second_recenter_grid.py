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
    PLUGIN as CURVE_GRID_PLUGIN,
    RESULT_SCHEMA_ID,
    WORKLOAD_ID,
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
from workflows.microlensing.coarse_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as COARSE_BUILD_MANIFEST_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as COARSE_DATASET_RELATIVE_PATH,
    build_coarse_grid_project,
)
from workflows.microlensing.recenter_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
    CONTRACT_RELATIVE_PATH as FIRST_RECENTER_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as FIRST_RECENTER_DATASET_RELATIVE_PATH,
    PROJECT_RELATIVE_PATH as FIRST_RECENTER_PROJECT_RELATIVE_PATH,
    build_recentered_grid_project,
)
from workflows.microlensing.refine_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
    build_refinement_grid_project,
)
from workflows.microlensing.second_recenter_grid import (
    BUILD_MANIFEST_RELATIVE_PATH,
    CANDIDATES_PER_WORK_UNIT,
    CENTER_COUNT,
    CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT,
    PROJECT_RELATIVE_PATH,
    SECOND_RECENTER_GRID_CONTRACT_ID,
    TOTAL_CANDIDATE_COUNT,
    SecondRecenterGridBuildError,
    build_second_recenter_grid_project,
)


SECOND_RECENTER_PROJECT_ID = (
    "openstar.generic-recovery-a.second-recentered-grid.v1"
)
FIRST_RECENTER_INVESTIGATION_ID = "generic-first-recenter-investigation"


def first_recenter_winning_result(
    first_recenter_root,
    *,
    center_index=10,
    scale_index=8,
    shape_index=8,
    objective=9.5,
):
    dataset = read_json(
        first_recenter_root / FIRST_RECENTER_DATASET_RELATIVE_PATH
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


def write_first_recenter_investigation(
    root,
    first_recenter_root,
    *,
    best=None,
):
    investigation_root = root / FIRST_RECENTER_INVESTIGATION_ID
    stages_root = investigation_root / "stages"
    stages_root.mkdir(parents=True)
    project_path = (
        first_recenter_root / FIRST_RECENTER_PROJECT_RELATIVE_PATH
    ).resolve()
    project_hash = sha256_bytes(project_path.read_bytes())
    dataset = read_json(
        first_recenter_root / FIRST_RECENTER_DATASET_RELATIVE_PATH
    )
    best = dict(best or first_recenter_winning_result(first_recenter_root))
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
        "projectID": RECENTERED_PROJECT_ID,
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
    terminal_parameters = {"expectedProjectID": RECENTERED_PROJECT_ID}
    terminal_result = {
        "completedWorkUnits": EXPECTED_WORK_UNIT_COUNT,
        "failedWorkUnits": 0,
        "passed": True,
        "projectID": RECENTERED_PROJECT_ID,
        "rule": "projectID matches and completed+failed == total",
        "totalWorkUnits": EXPECTED_WORK_UNIT_COUNT,
    }
    prepare = stage_record(
        stage_id="001-prepare-first-recenter",
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
            "id": "002-run-first-recenter",
            "parameters": run_parameters,
            "triggered_by_stage_id": "001-prepare-first-recenter",
        },
        stop=False,
    )
    run = stage_record(
        stage_id="002-run-first-recenter",
        handler_id="openstar.project.run",
        triggered_by="001-prepare-first-recenter",
        parameters=run_parameters,
        result=run_result,
        input_hashes={"projectManifest": project_hash},
        project_ids=(RECENTERED_PROJECT_ID,),
        next_stage={
            "handler_id": "generic.project.terminal-check",
            "id": "003-terminal-first-recenter",
            "parameters": terminal_parameters,
            "triggered_by_stage_id": "002-run-first-recenter",
        },
        stop=False,
        node_contributions=contributions,
    )
    terminal = stage_record(
        stage_id="003-terminal-first-recenter",
        handler_id="generic.project.terminal-check",
        triggered_by="002-run-first-recenter",
        parameters=terminal_parameters,
        result=terminal_result,
        input_hashes={},
        project_ids=(RECENTERED_PROJECT_ID,),
        next_stage=None,
        stop=True,
    )
    stages = [prepare, run, terminal]
    investigation = {
        "created_at": "2026-08-31T00:00:00+00:00",
        "id": FIRST_RECENTER_INVESTIGATION_ID,
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


def replace_first_recenter_winner(record_path, best):
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


class SecondRecenterGridFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.set_chain(self.make_chain(self.root / "chain"))

    def set_chain(self, chain):
        (
            self.prepared,
            self.coarse,
            self.coarse_investigation,
            self.refinement,
            self.refinement_investigation,
            self.first_recenter,
            self.first_recenter_investigation,
        ) = chain

    def make_chain(self, root, *, first_winner_indices=(10, 8, 8)):
        prepared = root / "prepared"
        prepared.mkdir(parents=True)
        write_prepared_root(prepared)
        coarse = root / "coarse"
        build_coarse_grid_project(
            prepared,
            project_id=COARSE_PROJECT_ID,
            output_root=coarse,
        )
        coarse_investigation = write_coarse_investigation(
            root / "coarse-investigations",
            coarse,
        )
        refinement = root / "refinement"
        build_refinement_grid_project(
            prepared,
            coarse_project_root=coarse,
            coarse_investigation_record=coarse_investigation,
            project_id=REFINEMENT_PROJECT_ID,
            output_root=refinement,
        )
        refinement_investigation = write_refinement_investigation(
            root / "refinement-investigations",
            refinement,
        )
        first_recenter = root / "first-recenter"
        build_recentered_grid_project(
            prepared,
            coarse_project_root=coarse,
            coarse_investigation_record=coarse_investigation,
            refinement_project_root=refinement,
            refinement_investigation_record=refinement_investigation,
            project_id=RECENTERED_PROJECT_ID,
            output_root=first_recenter,
        )
        best = first_recenter_winning_result(
            first_recenter,
            center_index=first_winner_indices[0],
            scale_index=first_winner_indices[1],
            shape_index=first_winner_indices[2],
        )
        first_recenter_investigation = write_first_recenter_investigation(
            root / "first-recenter-investigations",
            first_recenter,
            best=best,
        )
        return (
            prepared,
            coarse,
            coarse_investigation,
            refinement,
            refinement_investigation,
            first_recenter,
            first_recenter_investigation,
        )

    def build(self, output_name="second-recenter"):
        output = self.root / output_name
        result = build_second_recenter_grid_project(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            refinement_project_root=self.refinement,
            refinement_investigation_record=self.refinement_investigation,
            first_recenter_project_root=self.first_recenter,
            first_recenter_investigation_record=(
                self.first_recenter_investigation
            ),
            project_id=SECOND_RECENTER_PROJECT_ID,
            output_root=output,
        )
        return output, result

    def assert_rejected(self, pattern=None, *, output_name="rejected"):
        context = (
            self.assertRaisesRegex(SecondRecenterGridBuildError, pattern)
            if pattern is not None
            else self.assertRaises(SecondRecenterGridBuildError)
        )
        with context:
            self.build(output_name)


class SecondRecenterGridBuildTests(SecondRecenterGridFixture):
    def test_success_is_deterministic_and_worker_compatible(self):
        first, result = self.build("first-output")
        second, _ = self.build("second-output")

        self.assertEqual(serialized_tree(first), serialized_tree(second))
        self.assertEqual(
            {
                BUILD_MANIFEST_RELATIVE_PATH,
                CONTRACT_RELATIVE_PATH,
                DATASET_RELATIVE_PATH,
                PROJECT_RELATIVE_PATH,
            },
            set(serialized_tree(first)),
        )
        project = read_json(first / PROJECT_RELATIVE_PATH)
        dataset = read_json(first / DATASET_RELATIVE_PATH)
        self.assertEqual(result["project"], project)
        self.assertEqual(SECOND_RECENTER_PROJECT_ID, project["id"])
        self.assertEqual(WORKLOAD_ID, project["workloadID"])
        self.assertEqual(DATASET_SCHEMA_ID, project["datasetSchemaID"])
        self.assertEqual(PAYLOAD_SCHEMA_ID, project["payloadSchemaID"])
        self.assertEqual(RESULT_SCHEMA_ID, project["resultSchemaID"])
        self.assertEqual(dataset["id"], project["datasets"][0]["id"])
        self.assertEqual(DATASET_RELATIVE_PATH, project["datasets"][0]["path"])
        CURVE_GRID_PLUGIN.validate_dataset(dataset)

    def test_interior_winner_is_centered_with_parent_counts_and_steps(self):
        output, _ = self.build()
        parent = read_json(
            self.first_recenter / FIRST_RECENTER_DATASET_RELATIVE_PATH
        )["curveGrid"]
        winner = first_recenter_winning_result(self.first_recenter)
        grid = read_json(output / DATASET_RELATIVE_PATH)["curveGrid"]

        self.assertEqual(CENTER_COUNT, grid["centerAxis"]["count"])
        self.assertEqual(LOG_SCALE_COUNT, grid["logScaleAxis"]["count"])
        self.assertEqual(LOG_SHAPE_COUNT, grid["logShapeAxis"]["count"])
        for axis_name in ("centerAxis", "logScaleAxis", "logShapeAxis"):
            self.assertEqual(parent[axis_name]["step"], grid[axis_name]["step"])
        self.assertEqual(
            winner["bestCenter"],
            grid["centerAxis"]["start"] + 10 * grid["centerAxis"]["step"],
        )
        self.assertEqual(
            winner["bestLogScale"],
            grid["logScaleAxis"]["start"] + 8 * grid["logScaleAxis"]["step"],
        )
        self.assertEqual(
            winner["bestLogShape"],
            grid["logShapeAxis"]["start"] + 8 * grid["logShapeAxis"]["step"],
        )

    def test_boundary_winner_is_accepted_and_centered(self):
        self.set_chain(
            self.make_chain(
                self.root / "boundary-chain",
                first_winner_indices=(0, 16, 0),
            )
        )
        output, _ = self.build("boundary-output")
        winner = first_recenter_winning_result(
            self.first_recenter,
            center_index=0,
            scale_index=16,
            shape_index=0,
        )
        grid = read_json(output / DATASET_RELATIVE_PATH)["curveGrid"]
        self.assertEqual(
            winner["bestCenter"],
            grid["centerAxis"]["start"] + 10 * grid["centerAxis"]["step"],
        )
        self.assertEqual(
            winner["bestLogScale"],
            grid["logScaleAxis"]["start"] + 8 * grid["logScaleAxis"]["step"],
        )
        self.assertEqual(
            winner["bestLogShape"],
            grid["logShapeAxis"]["start"] + 8 * grid["logShapeAxis"]["step"],
        )

    def test_counts_samples_hashes_and_complete_provenance_are_exact(self):
        output, _ = self.build()
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        dataset = read_json(output / DATASET_RELATIVE_PATH)
        source = read_json(
            self.prepared / "blind" / "series" / "series-002.json"
        )

        self.assertEqual(6069, TOTAL_CANDIDATE_COUNT)
        self.assertEqual(95, EXPECTED_WORK_UNIT_COUNT)
        self.assertEqual(64, CANDIDATES_PER_WORK_UNIT)
        self.assertEqual(TOTAL_CANDIDATE_COUNT, manifest["candidateCount"])
        self.assertEqual(EXPECTED_WORK_UNIT_COUNT, manifest["expectedWorkUnitCount"])
        self.assertEqual(
            len(source["coordinates"]) * TOTAL_CANDIDATE_COUNT,
            manifest["expectedSampleCandidateEvaluationCount"],
        )
        self.assertEqual(source["coordinates"], dataset["coordinates"])
        self.assertEqual(source["values"], dataset["values"])
        self.assertEqual(source["inverseVariances"], dataset["inverseVariances"])
        self.assertEqual(
            SECOND_RECENTER_GRID_CONTRACT_ID,
            manifest["contractSchemaID"],
        )
        self.assertEqual(
            manifest["parentArtifactHashes"],
            contract["parentArtifactHashes"],
        )
        self.assertEqual(
            {
                "coarse": COARSE_PROJECT_ID,
                "firstRecenter": RECENTERED_PROJECT_ID,
                "firstRefinement": REFINEMENT_PROJECT_ID,
            },
            manifest["parentProjectIDs"],
        )
        self.assertEqual(
            {
                "coarse": read_json(self.coarse_investigation)["id"],
                "firstRecenter": read_json(
                    self.first_recenter_investigation
                )["id"],
                "firstRefinement": read_json(
                    self.refinement_investigation
                )["id"],
            },
            manifest["parentInvestigationIDs"],
        )
        self.assertEqual(
            sha256_bytes((output / DATASET_RELATIVE_PATH).read_bytes()),
            manifest["outputDatasetSHA256"],
        )
        self.assertEqual(
            sha256_bytes((output / PROJECT_RELATIVE_PATH).read_bytes()),
            manifest["outputProjectSHA256"],
        )

    def test_outputs_are_identity_free(self):
        output, _ = self.build()
        rendered = b"\n".join(
            path.read_bytes() for path in sorted(output.rglob("*.json"))
        ).decode("utf-8")
        for token in FORBIDDEN_OUTPUT_TOKENS:
            self.assertNotIn(token, rendered)
        self.assertIn("were not consulted", rendered)


class SecondRecenterGridRejectionTests(SecondRecenterGridFixture):
    def test_existing_output_is_rejected(self):
        output = self.root / "existing-output"
        output.mkdir()
        self.assert_rejected("already exists", output_name="existing-output")

    def test_symlink_traversal_is_rejected(self):
        target = self.root / "real-output-parent"
        target.mkdir()
        link = self.root / "linked-output-parent"
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are unavailable")
        with self.assertRaisesRegex(
            SecondRecenterGridBuildError,
            "traverses a symlink",
        ):
            build_second_recenter_grid_project(
                self.prepared,
                coarse_project_root=self.coarse,
                coarse_investigation_record=self.coarse_investigation,
                refinement_project_root=self.refinement,
                refinement_investigation_record=self.refinement_investigation,
                first_recenter_project_root=self.first_recenter,
                first_recenter_investigation_record=(
                    self.first_recenter_investigation
                ),
                project_id=SECOND_RECENTER_PROJECT_ID,
                output_root=link / "output",
            )

        prepared_link = self.root / "linked-prepared"
        os.symlink(self.prepared, prepared_link, target_is_directory=True)
        with self.assertRaisesRegex(
            SecondRecenterGridBuildError,
            "traverses a symlink",
        ):
            build_second_recenter_grid_project(
                prepared_link,
                coarse_project_root=self.coarse,
                coarse_investigation_record=self.coarse_investigation,
                refinement_project_root=self.refinement,
                refinement_investigation_record=self.refinement_investigation,
                first_recenter_project_root=self.first_recenter,
                first_recenter_investigation_record=(
                    self.first_recenter_investigation
                ),
                project_id=SECOND_RECENTER_PROJECT_ID,
                output_root=self.root / "input-symlink-output",
            )

    def test_missing_or_malformed_parent_artifacts_are_rejected(self):
        cases = (
            ("missing", FIRST_RECENTER_CONTRACT_RELATIVE_PATH, None),
            ("malformed", FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH, b"{"),
        )
        for name, relative_path, replacement in cases:
            with self.subTest(case=name):
                self.set_chain(self.make_chain(self.root / f"artifact-{name}"))
                path = self.first_recenter / relative_path
                if replacement is None:
                    path.unlink()
                else:
                    path.write_bytes(replacement)
                self.assert_rejected(output_name=f"artifact-{name}-output")

    def test_wrong_project_and_workflow_identity_are_rejected(self):
        def wrong_project(project):
            project["id"] = "openstar.generic.wrong-project.v1"

        cases = (
            ("project", wrong_project),
            (
                "workflow",
                lambda record: record.__setitem__(
                    "workflow_id",
                    "openstar.workflow.other.v1",
                ),
            ),
        )
        for name, mutate in cases:
            with self.subTest(case=name):
                self.set_chain(self.make_chain(self.root / f"identity-{name}"))
                if name == "project":
                    path = self.first_recenter / FIRST_RECENTER_PROJECT_RELATIVE_PATH
                    value = read_json(path)
                    mutate(value)
                    write_json(path, value)
                else:
                    value = read_json(self.first_recenter_investigation)
                    mutate(value)
                    write_json(self.first_recenter_investigation, value)
                self.assert_rejected(output_name=f"identity-{name}-output")

    def test_stage_count_order_and_terminal_status_are_rejected(self):
        def remove_stage(record):
            record["stages"].pop()

        def reorder(record):
            record["stages"][0], record["stages"][1] = (
                record["stages"][1],
                record["stages"][0],
            )

        def nonterminal(record):
            record["status"] = "RUNNING"

        for name, mutate in (
            ("count", remove_stage),
            ("order", reorder),
            ("nonterminal", nonterminal),
        ):
            with self.subTest(case=name):
                self.set_chain(self.make_chain(self.root / f"stages-{name}"))
                record = read_json(self.first_recenter_investigation)
                mutate(record)
                write_json(self.first_recenter_investigation, record)
                self.assert_rejected(output_name=f"stages-{name}-output")

    def test_failed_incomplete_duplicated_or_reordered_coverage_is_rejected(self):
        def failed(stage):
            stage["result"]["projectFailedWorkUnits"] = 1

        def incomplete(stage):
            stage["result"]["datasets"][0]["coverageComplete"] = False

        def duplicated(stage):
            stage["result"]["datasets"][0]["completedCandidateCount"] += 1

        def reordered(stage):
            stage["result"]["datasets"][0]["curveGridStatus"] = (
                "CURVE_GRID_INCOMPLETE"
            )

        for name, mutate in (
            ("failed", failed),
            ("incomplete", incomplete),
            ("duplicated", duplicated),
            ("reordered", reordered),
        ):
            with self.subTest(case=name):
                self.set_chain(self.make_chain(self.root / f"coverage-{name}"))
                rewrite_investigation_stage(
                    self.first_recenter_investigation,
                    1,
                    mutate,
                )
                self.assert_rejected(output_name=f"coverage-{name}-output")

    def test_winner_mapping_aggregate_disagreement_and_nonfinite_are_rejected(self):
        def outside_grid(best):
            best["bestCenter"] += 0.001

        def aggregate_disagreement(stage):
            stage["result"]["datasets"][0]["bestLogScale"] += 0.001

        def nonfinite(best):
            best["bestWeightedResidualSumSquares"] = math.inf

        for name, kind, mutate in (
            ("outside", "best", outside_grid),
            ("disagreement", "stage", aggregate_disagreement),
            ("nonfinite", "best", nonfinite),
        ):
            with self.subTest(case=name):
                self.set_chain(self.make_chain(self.root / f"winner-{name}"))
                if kind == "best":
                    best = first_recenter_winning_result(self.first_recenter)
                    mutate(best)
                    if name == "nonfinite":
                        with self.assertRaises(ValueError):
                            replace_first_recenter_winner(
                                self.first_recenter_investigation,
                                best,
                            )
                        record = read_json(self.first_recenter_investigation)
                        stage = record["stages"][1]
                        stage["result"]["datasets"][0]["payload"]["best"][
                            "bestWeightedResidualSumSquares"
                        ] = math.inf
                        write_json(
                            self.first_recenter_investigation,
                            record,
                            allow_nan=True,
                        )
                        write_json(
                            self.first_recenter_investigation.parent
                            / "stages"
                            / f"{stage['id']}.json",
                            stage,
                            allow_nan=True,
                        )
                    else:
                        replace_first_recenter_winner(
                            self.first_recenter_investigation,
                            best,
                        )
                else:
                    rewrite_investigation_stage(
                        self.first_recenter_investigation,
                        1,
                        mutate,
                    )
                self.assert_rejected(output_name=f"winner-{name}-output")

    def test_mutated_parent_artifacts_ledgers_and_hashes_are_rejected(self):
        def mutate_dataset(value):
            value["coordinates"][0] += 0.01

        def mutate_manifest(value):
            value["selectedSampleCount"] += 1

        cases = (
            (
                "coarse-dataset",
                "coarse",
                COARSE_DATASET_RELATIVE_PATH,
                mutate_dataset,
            ),
            (
                "refinement-manifest",
                "refinement",
                REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
                mutate_manifest,
            ),
            (
                "first-recenter-dataset",
                "first_recenter",
                FIRST_RECENTER_DATASET_RELATIVE_PATH,
                mutate_dataset,
            ),
            (
                "first-recenter-hash",
                "first_recenter",
                FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
                lambda value: value.__setitem__(
                    "outputDatasetSHA256",
                    "0" * 64,
                ),
            ),
        )
        for name, root_name, relative_path, mutate in cases:
            with self.subTest(case=name):
                self.set_chain(self.make_chain(self.root / f"mutated-{name}"))
                path = getattr(self, root_name) / relative_path
                value = read_json(path)
                mutate(value)
                write_json(path, value)
                self.assert_rejected(output_name=f"mutated-{name}-output")

        self.set_chain(self.make_chain(self.root / "mutated-ledger"))
        ledger = (
            self.first_recenter_investigation.parent
            / "stages"
            / "002-run-first-recenter.json"
        )
        value = read_json(ledger)
        value["result"]["projectCompletedWorkUnits"] -= 1
        write_json(ledger, value)
        self.assert_rejected(output_name="mutated-ledger-output")

    def test_ancestry_mismatch_at_each_parent_level_is_rejected(self):
        cases = (
            ("coarse", COARSE_BUILD_MANIFEST_RELATIVE_PATH, ("projectID",)),
            (
                "refinement",
                REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
                ("coarseProjectID",),
            ),
            (
                "first-recenter",
                FIRST_RECENTER_BUILD_MANIFEST_RELATIVE_PATH,
                ("firstRefinementProvenance", "projectID"),
            ),
        )
        for name, relative_path, field_path in cases:
            with self.subTest(parent=name):
                self.set_chain(self.make_chain(self.root / f"ancestry-{name}"))
                root = {
                    "coarse": self.coarse,
                    "refinement": self.refinement,
                    "first-recenter": self.first_recenter,
                }[name]
                path = root / relative_path
                value = read_json(path)
                target = value
                for field_name in field_path[:-1]:
                    target = target[field_name]
                target[field_path[-1]] = "openstar.generic.wrong-ancestry.v1"
                write_json(path, value)
                self.assert_rejected(output_name=f"ancestry-{name}-output")

    def test_transactional_cleanup_after_publication_failure(self):
        output_name = "transaction-failure"
        output = self.root / output_name
        with patch(
            "workflows.microlensing.second_recenter_grid._atomic_write_bytes",
            side_effect=OSError("synthetic write failure"),
        ):
            self.assert_rejected(
                "atomic output publication failed",
                output_name=output_name,
            )
        self.assertFalse(output.exists())
        self.assertEqual([], list(self.root.glob(f".{output_name}.*")))


if __name__ == "__main__":
    unittest.main()
