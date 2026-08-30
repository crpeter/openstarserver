import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_workloads.plugins.curve_grid import (
    DATASET_SCHEMA_ID,
    FAMILY_ID,
    MAX_SAFE_INTEGER,
    PAYLOAD_SCHEMA_ID,
    PLUGIN as CURVE_GRID_PLUGIN,
    RESULT_SCHEMA_ID,
    WORKLOAD_ID,
)
from tests.workflows.microlensing.test_refine_grid import (
    COARSE_PROJECT_ID,
    FORBIDDEN_OUTPUT_TOKENS,
    REFINEMENT_PROJECT_ID,
    canonical_json_bytes,
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
    CONTRACT_RELATIVE_PATH as COARSE_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as COARSE_DATASET_RELATIVE_PATH,
    PROJECT_RELATIVE_PATH as COARSE_PROJECT_RELATIVE_PATH,
    build_coarse_grid_project,
)
from workflows.microlensing.recenter_grid import (
    BUILD_MANIFEST_RELATIVE_PATH,
    CANDIDATES_PER_WORK_UNIT,
    CENTER_COUNT,
    CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT,
    LOG_SCALE_COUNT,
    LOG_SHAPE_COUNT,
    PROJECT_RELATIVE_PATH,
    RECENTERED_GRID_CONTRACT_ID,
    TOTAL_CANDIDATE_COUNT,
    RecenterGridBuildError,
    _REQUIRED_DATASET_STATUS_FIELDS,
    _REQUIRED_RUN_RESULT_FIELDS,
    _safe_product,
    build_recentered_grid_project,
)
from workflows.microlensing.refine_grid import (
    BUILD_MANIFEST_RELATIVE_PATH as REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
    CONTRACT_RELATIVE_PATH as REFINEMENT_CONTRACT_RELATIVE_PATH,
    DATASET_RELATIVE_PATH as REFINEMENT_DATASET_RELATIVE_PATH,
    EXPECTED_WORK_UNIT_COUNT as REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
    PROJECT_RELATIVE_PATH as REFINEMENT_PROJECT_RELATIVE_PATH,
    TOTAL_CANDIDATE_COUNT as REFINEMENT_TOTAL_CANDIDATE_COUNT,
    build_refinement_grid_project,
)


RECENTERED_PROJECT_ID = "openstar.generic-recovery-a.recentered-grid.v1"
REFINEMENT_INVESTIGATION_ID = "generic-refinement-investigation"
BOUNDARY_CENTER_INDEX = 10
BOUNDARY_SCALE_INDEX = 8
BOUNDARY_SHAPE_INDEX = 0


def refinement_winning_result(
    refinement_root,
    *,
    center_index=BOUNDARY_CENTER_INDEX,
    scale_index=BOUNDARY_SCALE_INDEX,
    shape_index=BOUNDARY_SHAPE_INDEX,
    objective=10.25,
):
    dataset = read_json(refinement_root / REFINEMENT_DATASET_RELATIVE_PATH)
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
        REFINEMENT_TOTAL_CANDIDATE_COUNT - shard_start,
    )
    return {
        "bestAmplitude": 1.5,
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
        "bestOffset": -0.5,
        "bestWeightedResidualSumSquares": objective,
        "evaluatedCandidateCount": shard_count,
        "familyID": FAMILY_ID,
        "gridCount": shard_count,
        "gridStartIndex": shard_start,
        "invalidCandidateCount": 0,
    }


def write_refinement_investigation(root, refinement_root, *, best=None):
    investigation_root = root / REFINEMENT_INVESTIGATION_ID
    stages_root = investigation_root / "stages"
    stages_root.mkdir(parents=True)
    project_path = (refinement_root / REFINEMENT_PROJECT_RELATIVE_PATH).resolve()
    project_hash = sha256_bytes(project_path.read_bytes())
    dataset = read_json(refinement_root / REFINEMENT_DATASET_RELATIVE_PATH)
    best = dict(best or refinement_winning_result(refinement_root))

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
        "completedCandidateCount": REFINEMENT_TOTAL_CANDIDATE_COUNT,
        "completedWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "classification": None,
        "coverageComplete": True,
        "curveGridStatus": "CURVE_GRID_COMPLETE",
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "environmentUnavailableCount": 0,
        "executionFailureCount": 0,
        "executionFailureKinds": {},
        "failedWorkUnits": 0,
        "familyID": FAMILY_ID,
        "id": dataset["id"],
        "iPhoneContribution": 48,
        "macContribution": REFINEMENT_EXPECTED_WORK_UNIT_COUNT - 48,
        "mission": "",
        "nodeContributions": {
            "generic-node-a": 48,
            "generic-node-b": REFINEMENT_EXPECTED_WORK_UNIT_COUNT - 48,
        },
        "otherContribution": 0,
        "pendingWorkUnits": 0,
        "payload": {"best": best},
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "progress": 1.0,
        "publishedPeriodDays": None,
        "referenceMismatchCount": 0,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "retryCount": 0,
        "role": None,
        "sector": None,
        "targetName": dataset["id"],
        "ticID": None,
        "totalCandidateCount": REFINEMENT_TOTAL_CANDIDATE_COUNT,
        "totalWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "transportUnavailableCount": 0,
        "verificationFailureCount": 0,
        "workloadID": WORKLOAD_ID,
        "workloadStatus": "CURVE_GRID_COMPLETE",
    }
    run_result = {
        "activeNodes": 2,
        "assigned": 0,
        "assignedWorkUnits": 0,
        "bestFrequency": None,
        "bestPeriodDays": None,
        "bestPower": None,
        "candidateFrequency": None,
        "candidatePeriodDays": None,
        "candidatePower": None,
        "completed": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "completedWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "coverageComplete": True,
        "datasetID": dataset["id"],
        "datasets": [dataset_status],
        "environmentUnavailableCount": 0,
        "executionFailureCount": 0,
        "executionFailureKinds": {},
        "failedWorkUnits": 0,
        "harmonicCandidates": [],
        "mission": "",
        "nodeContributions": {
            "generic-node-a": 48,
            "generic-node-b": REFINEMENT_EXPECTED_WORK_UNIT_COUNT - 48,
        },
        "pending": 0,
        "pendingWorkUnits": 0,
        "periodConfidence": None,
        "periodStatus": None,
        "preferredPhysicalPeriodDays": None,
        "preferredPhysicalPeriodRelation": None,
        "progress": 1.0,
        "projectAssignedWorkUnits": 0,
        "projectCompletedWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "projectEnvironmentUnavailableCount": 0,
        "projectExecutionFailureCount": 0,
        "projectFailedWorkUnits": 0,
        "projectID": REFINEMENT_PROJECT_ID,
        "projectPath": str(project_path),
        "projectPendingWorkUnits": 0,
        "projectProgress": 1.0,
        "projectTotalWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "projectTransportUnavailableCount": 0,
        "retryCount": 0,
        "sampleCount": len(dataset["coordinates"]),
        "samples": len(dataset["coordinates"]),
        "status": "COMPLETE",
        "targetName": dataset["id"],
        "total": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "totalWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "transportUnavailableCount": 0,
        "verificationFailureCount": 0,
        "workloadID": WORKLOAD_ID,
    }
    prepare_parameters = {"projectPath": str(project_path)}
    run_parameters = {
        "projectManifestSha256": project_hash,
        "projectPath": str(project_path),
    }
    terminal_parameters = {"expectedProjectID": REFINEMENT_PROJECT_ID}
    terminal_result = {
        "completedWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
        "failedWorkUnits": 0,
        "passed": True,
        "projectID": REFINEMENT_PROJECT_ID,
        "rule": "projectID matches and completed+failed == total",
        "totalWorkUnits": REFINEMENT_EXPECTED_WORK_UNIT_COUNT,
    }
    prepare = stage_record(
        stage_id="001-prepare-refinement",
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
            "id": "002-run-refinement",
            "parameters": run_parameters,
            "triggered_by_stage_id": "001-prepare-refinement",
        },
        stop=False,
    )
    run = stage_record(
        stage_id="002-run-refinement",
        handler_id="openstar.project.run",
        triggered_by="001-prepare-refinement",
        parameters=run_parameters,
        result=run_result,
        input_hashes={"projectManifest": project_hash},
        project_ids=(REFINEMENT_PROJECT_ID,),
        next_stage={
            "handler_id": "generic.project.terminal-check",
            "id": "003-terminal-refinement",
            "parameters": terminal_parameters,
            "triggered_by_stage_id": "002-run-refinement",
        },
        stop=False,
        node_contributions=run_result["nodeContributions"],
    )
    terminal = stage_record(
        stage_id="003-terminal-refinement",
        handler_id="generic.project.terminal-check",
        triggered_by="002-run-refinement",
        parameters=terminal_parameters,
        result=terminal_result,
        input_hashes={},
        project_ids=(REFINEMENT_PROJECT_ID,),
        next_stage=None,
        stop=True,
    )
    stages = [prepare, run, terminal]
    investigation = {
        "created_at": "2026-08-30T01:00:00+00:00",
        "id": REFINEMENT_INVESTIGATION_ID,
        "metadata": {
            "coordinator": "http://127.0.0.1:8080",
            "projectPath": str(project_path),
        },
        "stages": stages,
        "status": "COMPLETE",
        "updated_at": "2026-08-30T01:00:02+00:00",
        "workflow_id": "openstar.workflow.project-smoke.v1",
        "workflow_version": "20.0",
    }
    record_path = investigation_root / "investigation.json"
    write_json(record_path, investigation)
    for stage in stages:
        write_json(stages_root / f"{stage['id']}.json", stage)
    return record_path


def replace_winner(record_path, best):
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


class RecenterGridFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (
            self.prepared,
            self.coarse,
            self.coarse_investigation,
            self.refinement,
            self.refinement_investigation,
        ) = self.make_chain(self.root / "chain")

    def make_chain(self, root, *, winner_indices=None):
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
        best = None
        if winner_indices is not None:
            best = refinement_winning_result(
                refinement,
                center_index=winner_indices[0],
                scale_index=winner_indices[1],
                shape_index=winner_indices[2],
            )
        refinement_investigation = write_refinement_investigation(
            root / "refinement-investigations",
            refinement,
            best=best,
        )
        return (
            prepared,
            coarse,
            coarse_investigation,
            refinement,
            refinement_investigation,
        )

    def build(self, name="recentered"):
        output = self.root / name
        result = build_recentered_grid_project(
            self.prepared,
            coarse_project_root=self.coarse,
            coarse_investigation_record=self.coarse_investigation,
            refinement_project_root=self.refinement,
            refinement_investigation_record=self.refinement_investigation,
            project_id=RECENTERED_PROJECT_ID,
            output_root=output,
        )
        return output, result

    def assert_rejected(self, pattern=None, *, output_name="rejected"):
        context = (
            self.assertRaisesRegex(RecenterGridBuildError, pattern)
            if pattern is not None
            else self.assertRaises(RecenterGridBuildError)
        )
        with context:
            build_recentered_grid_project(
                self.prepared,
                coarse_project_root=self.coarse,
                coarse_investigation_record=self.coarse_investigation,
                refinement_project_root=self.refinement,
                refinement_investigation_record=self.refinement_investigation,
                project_id=RECENTERED_PROJECT_ID,
                output_root=self.root / output_name,
            )


class RecenterGridBuildTests(RecenterGridFixture):
    def test_success_is_deterministic_and_directly_activatable(self):
        first, result = self.build("first")
        second, _ = self.build("second")

        self.assertEqual(serialized_tree(first), serialized_tree(second))
        project = read_json(first / PROJECT_RELATIVE_PATH)
        dataset = read_json(first / DATASET_RELATIVE_PATH)
        self.assertEqual(result["project"], project)
        self.assertEqual(RECENTERED_PROJECT_ID, project["id"])
        self.assertEqual(WORKLOAD_ID, project["workloadID"])
        self.assertEqual(DATASET_SCHEMA_ID, project["datasetSchemaID"])
        self.assertEqual(PAYLOAD_SCHEMA_ID, project["payloadSchemaID"])
        self.assertEqual(RESULT_SCHEMA_ID, project["resultSchemaID"])
        self.assertEqual(dataset["id"], project["datasets"][0]["id"])
        self.assertEqual(DATASET_RELATIVE_PATH, project["datasets"][0]["path"])
        CURVE_GRID_PLUGIN.validate_dataset(dataset)

    def test_realistic_coordinator_envelope_fields_are_accepted(self):
        investigation = read_json(self.refinement_investigation)
        run_result = investigation["stages"][1]["result"]
        dataset_status = run_result["datasets"][0]

        self.assertEqual(1.0, run_result["projectProgress"])
        self.assertEqual(0, run_result["projectExecutionFailureCount"])
        self.assertEqual(2, run_result["activeNodes"])
        self.assertEqual(1.0, dataset_status["progress"])
        self.assertEqual(0, dataset_status["retryCount"])
        self.assertEqual(
            run_result["nodeContributions"],
            dataset_status["nodeContributions"],
        )
        self.build()

    def test_arbitrary_additional_coordinator_envelope_fields_are_accepted(self):
        def add_future_fields(stage):
            stage["result"]["futureProjectDiagnostic"] = {
                "opaque": "project-value"
            }
            stage["result"]["datasets"][0]["futureDatasetDiagnostic"] = {
                "opaque": "dataset-value"
            }

        rewrite_investigation_stage(
            self.refinement_investigation,
            1,
            add_future_fields,
        )
        self.build()

    def test_axes_retain_steps_and_center_the_verified_winner(self):
        output, _ = self.build()
        prior_dataset = read_json(
            self.refinement / REFINEMENT_DATASET_RELATIVE_PATH
        )
        best = refinement_winning_result(self.refinement)
        dataset = read_json(output / DATASET_RELATIVE_PATH)
        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        grid = dataset["curveGrid"]

        self.assertEqual(CENTER_COUNT, grid["centerAxis"]["count"])
        self.assertEqual(LOG_SCALE_COUNT, grid["logScaleAxis"]["count"])
        self.assertEqual(LOG_SHAPE_COUNT, grid["logShapeAxis"]["count"])
        self.assertEqual(
            prior_dataset["curveGrid"]["centerAxis"]["step"],
            grid["centerAxis"]["step"],
        )
        self.assertEqual(
            prior_dataset["curveGrid"]["logScaleAxis"]["step"],
            grid["logScaleAxis"]["step"],
        )
        self.assertEqual(
            prior_dataset["curveGrid"]["logShapeAxis"]["step"],
            grid["logShapeAxis"]["step"],
        )
        self.assertEqual(
            best["bestCenter"] - 10 * grid["centerAxis"]["step"],
            grid["centerAxis"]["start"],
        )
        self.assertEqual(
            best["bestLogScale"] - 8 * grid["logScaleAxis"]["step"],
            grid["logScaleAxis"]["start"],
        )
        self.assertEqual(
            best["bestLogShape"] - 8 * grid["logShapeAxis"]["step"],
            grid["logShapeAxis"]["start"],
        )
        self.assertAlmostEqual(
            best["bestCenter"],
            grid["centerAxis"]["start"] + 10 * grid["centerAxis"]["step"],
        )
        self.assertAlmostEqual(
            best["bestLogScale"],
            grid["logScaleAxis"]["start"] + 8 * grid["logScaleAxis"]["step"],
        )
        self.assertAlmostEqual(
            best["bestLogShape"],
            grid["logShapeAxis"]["start"] + 8 * grid["logShapeAxis"]["step"],
        )
        self.assertEqual(grid, contract["curveGrid"])
        self.assertEqual(["logShape"], contract["boundaryTrigger"]["boundaryAxes"])
        self.assertEqual(10, contract["derivationRules"]["center"]["winnerIndex"])
        self.assertEqual(8, contract["derivationRules"]["logScale"]["winnerIndex"])
        self.assertEqual(8, contract["derivationRules"]["logShape"]["winnerIndex"])

    def test_counts_arrays_and_safe_evaluations_are_exact(self):
        output, _ = self.build()
        source = read_json(
            self.prepared / "blind" / "series" / "series-002.json"
        )
        dataset = read_json(output / DATASET_RELATIVE_PATH)
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)

        self.assertEqual(source["coordinates"], dataset["coordinates"])
        self.assertEqual(source["values"], dataset["values"])
        self.assertEqual(source["inverseVariances"], dataset["inverseVariances"])
        self.assertEqual(6069, TOTAL_CANDIDATE_COUNT)
        self.assertEqual(TOTAL_CANDIDATE_COUNT, manifest["totalCandidateCount"])
        self.assertEqual(95, EXPECTED_WORK_UNIT_COUNT)
        self.assertEqual(EXPECTED_WORK_UNIT_COUNT, manifest["expectedWorkUnitCount"])
        self.assertEqual(
            len(source["coordinates"]) * TOTAL_CANDIDATE_COUNT,
            manifest["expectedSampleCandidateEvaluationCount"],
        )
        self.assertEqual(
            169713516,
            _safe_product(27964, TOTAL_CANDIDATE_COUNT, "evaluations"),
        )
        with self.assertRaisesRegex(RecenterGridBuildError, "safe integer"):
            _safe_product(MAX_SAFE_INTEGER, 2, "evaluations")

    def test_complete_provenance_chain_and_hashes_are_bound(self):
        output, _ = self.build()
        manifest = read_json(output / BUILD_MANIFEST_RELATIVE_PATH)
        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        coarse = manifest["coarseProvenance"]
        refinement = manifest["firstRefinementProvenance"]

        self.assertEqual(coarse, contract["provenanceChain"]["coarse"])
        self.assertEqual(
            refinement,
            contract["provenanceChain"]["firstRefinement"],
        )
        self.assertEqual(
            manifest["preparationManifestSHA256"],
            contract["provenanceChain"]["preparationManifestSHA256"],
        )
        self.assertEqual(
            sha256_bytes(
                (
                    self.prepared
                    / "blind"
                    / "preparation-manifest.json"
                ).read_bytes()
            ),
            manifest["preparationManifestSHA256"],
        )
        for field_name, relative_path in (
            ("buildManifestSHA256", COARSE_BUILD_MANIFEST_RELATIVE_PATH),
            ("contractFileSHA256", COARSE_CONTRACT_RELATIVE_PATH),
            ("datasetSHA256", COARSE_DATASET_RELATIVE_PATH),
            ("projectSHA256", COARSE_PROJECT_RELATIVE_PATH),
        ):
            self.assertEqual(
                sha256_bytes((self.coarse / relative_path).read_bytes()),
                coarse[field_name],
            )
        self.assertEqual(
            sha256_bytes(self.coarse_investigation.read_bytes()),
            coarse["investigationRecordSHA256"],
        )
        for stage_id, digest in coarse["stageLedgerSHA256s"].items():
            self.assertEqual(
                sha256_bytes(
                    (
                        self.coarse_investigation.parent
                        / "stages"
                        / f"{stage_id}.json"
                    ).read_bytes()
                ),
                digest,
            )
        self.assertEqual(
            sha256_bytes(
                (self.refinement / REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH)
                .read_bytes()
            ),
            refinement["buildManifestSHA256"],
        )
        self.assertEqual(
            sha256_bytes(
                (self.refinement / REFINEMENT_CONTRACT_RELATIVE_PATH).read_bytes()
            ),
            refinement["contractFileSHA256"],
        )
        self.assertEqual(
            sha256_bytes(
                (self.refinement / REFINEMENT_DATASET_RELATIVE_PATH).read_bytes()
            ),
            refinement["datasetSHA256"],
        )
        self.assertEqual(
            sha256_bytes(
                (self.refinement / REFINEMENT_PROJECT_RELATIVE_PATH).read_bytes()
            ),
            refinement["projectSHA256"],
        )
        self.assertEqual(
            sha256_bytes(self.refinement_investigation.read_bytes()),
            refinement["investigationRecordSHA256"],
        )
        for stage_id, digest in refinement["stageLedgerSHA256s"].items():
            self.assertEqual(
                sha256_bytes(
                    (
                        self.refinement_investigation.parent
                        / "stages"
                        / f"{stage_id}.json"
                    ).read_bytes()
                ),
                digest,
            )
        self.assertEqual(
            refinement["stageLedgerSHA256s"]["002-run-refinement"],
            refinement["runStageLedgerSHA256"],
        )
        self.assertEqual(
            RECENTERED_GRID_CONTRACT_ID,
            contract["contractID"],
        )
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(contract)),
            manifest["recenteredSearchContractSHA256"],
        )
        self.assertEqual(
            sha256_bytes((output / CONTRACT_RELATIVE_PATH).read_bytes()),
            manifest["outputContractFileSHA256"],
        )
        self.assertEqual(
            sha256_bytes((output / DATASET_RELATIVE_PATH).read_bytes()),
            manifest["outputDatasetSHA256"],
        )
        self.assertEqual(
            sha256_bytes((output / PROJECT_RELATIVE_PATH).read_bytes()),
            manifest["outputProjectSHA256"],
        )

    def test_any_single_boundary_axis_justifies_recentering(self):
        cases = (
            ("center", (0, 8, 8)),
            ("log-scale", (10, 16, 8)),
            ("log-shape", (10, 8, 0)),
        )
        for name, indices in cases:
            with self.subTest(axis=name):
                chain = self.make_chain(self.root / name, winner_indices=indices)
                output = self.root / f"{name}-output"
                result = build_recentered_grid_project(
                    chain[0],
                    coarse_project_root=chain[1],
                    coarse_investigation_record=chain[2],
                    refinement_project_root=chain[3],
                    refinement_investigation_record=chain[4],
                    project_id=RECENTERED_PROJECT_ID,
                    output_root=output,
                )
                self.assertTrue(result["contract"]["boundaryTrigger"]["boundaryAxes"])

    def test_existing_output_roots_are_never_overwritten(self):
        target = self.root / "target"
        target.mkdir()
        directory = self.root / "existing-directory"
        directory.mkdir()
        file_path = self.root / "existing-file"
        file_path.write_text("keep", encoding="utf-8")
        link = self.root / "existing-link"
        os.symlink(target, link)

        for output in (directory, file_path, link):
            with self.subTest(output=output.name):
                with self.assertRaisesRegex(RecenterGridBuildError, "already exists"):
                    build_recentered_grid_project(
                        self.prepared,
                        coarse_project_root=self.coarse,
                        coarse_investigation_record=self.coarse_investigation,
                        refinement_project_root=self.refinement,
                        refinement_investigation_record=self.refinement_investigation,
                        project_id=RECENTERED_PROJECT_ID,
                        output_root=output,
                    )
        self.assertEqual("keep", file_path.read_text(encoding="utf-8"))

        linked_parent = self.root / "linked-parent"
        os.symlink(target, linked_parent)
        with self.assertRaisesRegex(
            RecenterGridBuildError,
            "traverses a symlink",
        ):
            build_recentered_grid_project(
                self.prepared,
                coarse_project_root=self.coarse,
                coarse_investigation_record=self.coarse_investigation,
                refinement_project_root=self.refinement,
                refinement_investigation_record=self.refinement_investigation,
                project_id=RECENTERED_PROJECT_ID,
                output_root=linked_parent / "new-output",
            )

    def test_sealed_state_is_not_read_and_outputs_are_identity_free(self):
        sealed = self.prepared / "sealed"
        sealed.mkdir()
        (sealed / "identity-seal.json").write_text(
            "must remain unread",
            encoding="utf-8",
        )
        original_read_bytes = Path.read_bytes

        def guarded_read_bytes(path):
            if "sealed" in path.parts:
                raise AssertionError("sealed input was read")
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", guarded_read_bytes):
            output, _ = self.build()

        serialized = b"\n".join(
            path.read_bytes() for path in sorted(output.rglob("*.json"))
        ).decode("utf-8").casefold()
        for token in FORBIDDEN_OUTPUT_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token.casefold(), serialized)


class RecenterGridRejectionTests(RecenterGridFixture):
    def test_interior_winner_is_rejected(self):
        best = refinement_winning_result(
            self.refinement,
            center_index=10,
            scale_index=8,
            shape_index=8,
        )
        replace_winner(self.refinement_investigation, best)
        self.assert_rejected("interior on every axis")

    def test_incorrect_winner_index_mapping_is_rejected(self):
        best = refinement_winning_result(self.refinement)
        best["bestLogShape"] += 0.001
        replace_winner(self.refinement_investigation, best)
        self.assert_rejected("map exactly")

    def test_removing_any_required_coordinator_field_is_rejected(self):
        cases = (
            ("project", _REQUIRED_RUN_RESULT_FIELDS),
            ("dataset", _REQUIRED_DATASET_STATUS_FIELDS),
        )
        for envelope_name, required_fields in cases:
            for field_name in sorted(required_fields):
                with self.subTest(envelope=envelope_name, field=field_name):
                    chain = self.make_chain(
                        self.root
                        / f"missing-{envelope_name}-{field_name}"
                    )
                    (
                        self.prepared,
                        self.coarse,
                        self.coarse_investigation,
                        self.refinement,
                        self.refinement_investigation,
                    ) = chain

                    def remove_required_field(
                        stage,
                        *,
                        envelope=envelope_name,
                        field=field_name,
                    ):
                        target = stage["result"]
                        if envelope == "dataset":
                            target = target["datasets"][0]
                        target.pop(field)

                    rewrite_investigation_stage(
                        self.refinement_investigation,
                        1,
                        remove_required_field,
                    )
                    self.assert_rejected(
                        "missing required fields",
                        output_name=(
                            f"missing-{envelope_name}-{field_name}-output"
                        ),
                    )

    def test_altering_any_required_coordinator_field_is_rejected(self):
        cases = (
            ("project", _REQUIRED_RUN_RESULT_FIELDS),
            ("dataset", _REQUIRED_DATASET_STATUS_FIELDS),
        )
        for envelope_name, required_fields in cases:
            for field_name in sorted(required_fields):
                with self.subTest(envelope=envelope_name, field=field_name):
                    chain = self.make_chain(
                        self.root
                        / f"altered-{envelope_name}-{field_name}"
                    )
                    (
                        self.prepared,
                        self.coarse,
                        self.coarse_investigation,
                        self.refinement,
                        self.refinement_investigation,
                    ) = chain

                    def alter_required_field(
                        stage,
                        *,
                        envelope=envelope_name,
                        field=field_name,
                    ):
                        target = stage["result"]
                        if envelope == "dataset":
                            target = target["datasets"][0]
                        target[field] = None

                    rewrite_investigation_stage(
                        self.refinement_investigation,
                        1,
                        alter_required_field,
                    )
                    self.assert_rejected(
                        output_name=(
                            f"altered-{envelope_name}-{field_name}-output"
                        )
                    )

    def test_additional_workload_payload_field_is_rejected(self):
        def add_payload_field(stage):
            stage["result"]["datasets"][0]["payload"][
                "futureWorkloadField"
            ] = "not-allowed"

        rewrite_investigation_stage(
            self.refinement_investigation,
            1,
            add_payload_field,
        )
        self.assert_rejected("result payload is invalid")

    def test_additional_winning_payload_field_is_rejected(self):
        def add_winner_field(stage):
            stage["result"]["datasets"][0]["payload"]["best"][
                "futureWinnerField"
            ] = "not-allowed"

        rewrite_investigation_stage(
            self.refinement_investigation,
            1,
            add_winner_field,
        )
        self.assert_rejected("winning result field set is invalid")

    def test_incomplete_coverage_and_failed_work_are_rejected(self):
        cases = (
            (
                "incomplete-project",
                lambda stage: stage["result"].__setitem__(
                    "projectCompletedWorkUnits",
                    REFINEMENT_EXPECTED_WORK_UNIT_COUNT - 1,
                ),
            ),
            (
                "failed-project",
                lambda stage: stage["result"].__setitem__(
                    "projectFailedWorkUnits",
                    1,
                ),
            ),
            (
                "incomplete-dataset",
                lambda stage: stage["result"]["datasets"][0].__setitem__(
                    "completedCandidateCount",
                    REFINEMENT_TOTAL_CANDIDATE_COUNT - 1,
                ),
            ),
            (
                "failed-dataset",
                lambda stage: stage["result"]["datasets"][0].__setitem__(
                    "failedWorkUnits",
                    1,
                ),
            ),
            (
                "noninteger-project-count",
                lambda stage: stage["result"].__setitem__(
                    "projectCompletedWorkUnits",
                    float(REFINEMENT_EXPECTED_WORK_UNIT_COUNT),
                ),
            ),
            (
                "nonboolean-coverage",
                lambda stage: stage["result"]["datasets"][0].__setitem__(
                    "coverageComplete",
                    1,
                ),
            ),
        )
        for name, mutator in cases:
            with self.subTest(name=name):
                chain = self.make_chain(self.root / name)
                self.prepared, self.coarse, self.coarse_investigation = chain[:3]
                self.refinement, self.refinement_investigation = chain[3:]
                rewrite_investigation_stage(
                    self.refinement_investigation,
                    1,
                    mutator,
                )
                self.assert_rejected(output_name=f"{name}-output")

    def test_altered_coarse_or_first_refinement_artifacts_are_rejected(self):
        cases = (
            (
                "coarse-dataset",
                "coarse",
                COARSE_DATASET_RELATIVE_PATH,
                lambda value: value["values"].__setitem__(0, 99.0),
            ),
            (
                "refinement-contract",
                "refinement",
                REFINEMENT_CONTRACT_RELATIVE_PATH,
                lambda value: value.__setitem__("candidateCount", 1),
            ),
            (
                "refinement-dataset",
                "refinement",
                REFINEMENT_DATASET_RELATIVE_PATH,
                lambda value: value["values"].__setitem__(0, 99.0),
            ),
            (
                "refinement-project",
                "refinement",
                REFINEMENT_PROJECT_RELATIVE_PATH,
                lambda value: value["datasets"][0].__setitem__(
                    "path",
                    "../outside.json",
                ),
            ),
            (
                "refinement-manifest",
                "refinement",
                REFINEMENT_BUILD_MANIFEST_RELATIVE_PATH,
                lambda value: value.__setitem__(
                    "outputDatasetSHA256",
                    "not-a-sha256",
                ),
            ),
        )
        for name, root_name, relative_path, mutator in cases:
            with self.subTest(name=name):
                chain = self.make_chain(self.root / name)
                self.prepared, self.coarse, self.coarse_investigation = chain[:3]
                self.refinement, self.refinement_investigation = chain[3:]
                artifact_root = (
                    self.coarse if root_name == "coarse" else self.refinement
                )
                path = artifact_root / relative_path
                value = read_json(path)
                mutator(value)
                write_json(path, value)
                self.assert_rejected(output_name=f"{name}-output")

    def test_mismatched_recorded_prepare_path_is_rejected(self):
        def alter_prepare(stage):
            stage["parameters"]["projectPath"] = "../different/project.json"

        rewrite_investigation_stage(
            self.refinement_investigation,
            0,
            alter_prepare,
        )
        self.assert_rejected("different project")

    def test_altered_investigation_records_are_rejected(self):
        cases = (
            (
                "coarse-record",
                "coarse",
                lambda value: value.__setitem__("workflow_id", "other.workflow"),
            ),
            (
                "refinement-record",
                "refinement",
                lambda value: value.__setitem__("status", "RUNNING"),
            ),
        )
        for name, record_name, mutator in cases:
            with self.subTest(name=name):
                chain = self.make_chain(self.root / name)
                self.prepared, self.coarse, self.coarse_investigation = chain[:3]
                self.refinement, self.refinement_investigation = chain[3:]
                record_path = (
                    self.coarse_investigation
                    if record_name == "coarse"
                    else self.refinement_investigation
                )
                value = read_json(record_path)
                mutator(value)
                write_json(record_path, value)
                self.assert_rejected(output_name=f"{name}-output")

    def test_altered_missing_and_symlinked_stage_ledgers_are_rejected(self):
        ledger_root = self.refinement_investigation.parent / "stages"
        altered = ledger_root / "002-run-refinement.json"
        value = read_json(altered)
        value["completed_at"] = "tampered"
        write_json(altered, value)
        self.assert_rejected("does not match", output_name="altered-output")

        chain = self.make_chain(self.root / "missing-ledger")
        self.prepared, self.coarse, self.coarse_investigation = chain[:3]
        self.refinement, self.refinement_investigation = chain[3:]
        missing = (
            self.refinement_investigation.parent
            / "stages"
            / "001-prepare-refinement.json"
        )
        missing.unlink()
        self.assert_rejected("ledger set", output_name="missing-output")

        chain = self.make_chain(self.root / "symlink-ledger")
        self.prepared, self.coarse, self.coarse_investigation = chain[:3]
        self.refinement, self.refinement_investigation = chain[3:]
        ledger = (
            self.refinement_investigation.parent
            / "stages"
            / "002-run-refinement.json"
        )
        target = self.root / "ledger-target.json"
        target.write_bytes(ledger.read_bytes())
        ledger.unlink()
        os.symlink(target, ledger)
        self.assert_rejected("non-symlink", output_name="symlink-ledger-output")

    def test_symlinks_unsafe_paths_and_unexpected_files_are_rejected(self):
        dataset = self.refinement / REFINEMENT_DATASET_RELATIVE_PATH
        target = self.root / "dataset-target.json"
        target.write_bytes(dataset.read_bytes())
        dataset.unlink()
        os.symlink(target, dataset)
        self.assert_rejected("non-symlink", output_name="dataset-link-output")

        chain = self.make_chain(self.root / "unsafe-path")
        self.prepared, self.coarse, self.coarse_investigation = chain[:3]
        self.refinement, self.refinement_investigation = chain[3:]
        project = self.refinement / REFINEMENT_PROJECT_RELATIVE_PATH
        value = read_json(project)
        value["datasets"][0]["path"] = "../outside.json"
        write_json(project, value)
        self.assert_rejected(output_name="unsafe-path-output")

        chain = self.make_chain(self.root / "unexpected")
        self.prepared, self.coarse, self.coarse_investigation = chain[:3]
        self.refinement, self.refinement_investigation = chain[3:]
        write_json(self.refinement / "unexpected.json", {})
        self.assert_rejected("unexpected", output_name="unexpected-output")


if __name__ == "__main__":
    unittest.main()
