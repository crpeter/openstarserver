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
from workflows.microlensing.coarse_grid import CoarseGridBuildError
from tests.workflows.microlensing.test_prepare_residuals import (
    ResidualPreparationFixture,
    write_residual_prepared_root,
)
from tests.workflows.microlensing.test_refine_grid import (
    FORBIDDEN_OUTPUT_TOKENS,
    read_json,
    rewrite_investigation_stage,
    serialized_tree,
    sha256_bytes,
    stage_record,
    write_json,
)
from workflows.microlensing.prepare_residuals import (
    ResidualPreparationError,
    _canonical_curve_basis,
    prepare_blind_microlensing_residuals,
)
from workflows.microlensing.residual_grid import (
    CANDIDATES_PER_DATASET,
    CANDIDATES_PER_WORK_UNIT,
    PROJECT_RELATIVE_PATH as GRID_PROJECT_RELATIVE_PATH,
    ResidualGridBuildError,
    WORK_UNITS_PER_DATASET,
    build_residual_grid_project,
)
from workflows.microlensing.refine_grid import RefinementGridBuildError
from workflows.microlensing.validate_residual_grid import (
    CONFIRMED_NEXT_TEST,
    CONTRACT_RELATIVE_PATH,
    CROSS_VALIDATION_CONTRACT_ID,
    CROSS_VALIDATION_RESULT_SCHEMA_ID,
    DISCOVERY_DELTA_WRSS_THRESHOLD,
    NEGATIVE_CLASSIFICATION,
    POSITIVE_CLASSIFICATION,
    RESULT_RELATIVE_PATH,
    UNCONFIRMED_NEXT_TEST,
    VALIDATION_DELTA_WRSS_THRESHOLD,
    ResidualGridValidationError,
    _VerifiedWinner,
    _discovery_gate,
    _held_out_result,
    _overall_classification,
    _parser,
    _verify_run_counter_scopes,
    validate_residual_grid,
)


PROJECT_ID = "openstar.generic-recovery-a.residual-grid-validation.v1"
INVESTIGATION_ID = "generic-residual-grid-investigation"
FROZEN_CENTER = 2247.0
LOCALIZED_SCALE = 0.5
LOCALIZED_SHAPE = math.sqrt(0.001)


def write_cross_validation_prepared_root(root):
    write_residual_prepared_root(root)
    manifest_path = root / "blind" / "preparation-manifest.json"
    manifest = read_json(manifest_path)
    coordinates = [FROZEN_CENTER + (index - 6) * 0.01 for index in range(13)]
    localized_basis = _canonical_curve_basis(
        coordinates,
        center=FROZEN_CENTER,
        scale=LOCALIZED_SCALE,
        shape=LOCALIZED_SHAPE,
    )
    for ordinal, record in enumerate(manifest["series"], start=1):
        series_path = root / "blind" / record["seriesFile"]
        series = read_json(series_path)
        amplitude = 12.0 - 2.0 * ordinal
        series["coordinates"] = list(coordinates)
        series["inverseVariances"] = [8.0 + ordinal + index for index in range(13)]
        series["values"] = [
            0.5 + 0.03 * ordinal + amplitude * basis
            for basis in localized_basis
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


def accepted_winner(dataset, *, scale_index):
    grid = dataset["curveGrid"]
    center_index = 64
    shape_index = 0
    grid_index = (
        (center_index * grid["logScaleAxis"]["count"] + scale_index)
        * grid["logShapeAxis"]["count"]
        + shape_index
    )
    evaluated = _evaluate_candidate(dataset, grid_index)
    if evaluated is None:
        raise AssertionError("synthetic residual-grid winner is invalid")
    shard_start = (
        grid_index // CANDIDATES_PER_WORK_UNIT
    ) * CANDIDATES_PER_WORK_UNIT
    shard_count = min(
        CANDIDATES_PER_WORK_UNIT,
        CANDIDATES_PER_DATASET - shard_start,
    )
    return {
        "bestAmplitude": evaluated.amplitude,
        "bestCenter": evaluated.center,
        "bestGridIndex": grid_index,
        "bestLogScale": evaluated.log_scale,
        "bestLogShape": evaluated.log_shape,
        "bestOffset": evaluated.offset,
        "bestWeightedResidualSumSquares": (
            evaluated.weighted_residual_sum_squares
        ),
        "evaluatedCandidateCount": shard_count,
        "familyID": FAMILY_ID,
        "gridCount": shard_count,
        "gridStartIndex": shard_start,
        "invalidCandidateCount": 0,
    }


def write_residual_grid_investigation(root, grid_root):
    investigation_root = root / INVESTIGATION_ID
    stages_root = investigation_root / "stages"
    stages_root.mkdir(parents=True)
    project_path = (grid_root / GRID_PROJECT_RELATIVE_PATH).resolve()
    project = read_json(project_path)
    project_hash = sha256_bytes(project_path.read_bytes())
    contributions = {"generic-node-a": 53, "generic-node-b": 52}
    statuses = []
    for ordinal, reference in enumerate(project["datasets"], start=1):
        dataset = read_json(grid_root / reference["path"])
        best = accepted_winner(
            dataset,
            scale_index=0 if ordinal == 1 else 4,
        )
        statuses.append(
            {
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
                "completedCandidateCount": CANDIDATES_PER_DATASET,
                "completedWorkUnits": WORK_UNITS_PER_DATASET,
                "coverageComplete": True,
                "curveGridStatus": "CURVE_GRID_COMPLETE",
                "datasetSchemaID": DATASET_SCHEMA_ID,
                "failedWorkUnits": 0,
                "familyID": FAMILY_ID,
                "id": dataset["id"],
                "nodeContributions": (
                    {"generic-node-a": 35}
                    if ordinal == 1
                    else {"generic-node-a": 18, "generic-node-b": 17}
                    if ordinal == 2
                    else {"generic-node-b": 35}
                ),
                "payload": {"best": best},
                "payloadSchemaID": PAYLOAD_SCHEMA_ID,
                "pendingWorkUnits": 0,
                "progress": 1.0,
                "resultSchemaID": RESULT_SCHEMA_ID,
                "totalCandidateCount": CANDIDATES_PER_DATASET,
                "totalWorkUnits": WORK_UNITS_PER_DATASET,
                "workloadID": WORKLOAD_ID,
                "workloadStatus": "CURVE_GRID_COMPLETE",
            }
        )
    expected_work_units = len(statuses) * WORK_UNITS_PER_DATASET
    if expected_work_units != 105:
        raise AssertionError("synthetic fixture must exercise 105 work units")
    final_status = statuses[-1]
    run_result = {
        "assignedWorkUnits": final_status["assignedWorkUnits"],
        "completedWorkUnits": final_status["completedWorkUnits"],
        "datasets": statuses,
        "failedWorkUnits": final_status["failedWorkUnits"],
        "nodeContributions": dict(contributions),
        "pendingWorkUnits": final_status["pendingWorkUnits"],
        "projectAssignedWorkUnits": 0,
        "projectCompletedWorkUnits": expected_work_units,
        "projectFailedWorkUnits": 0,
        "projectID": project["id"],
        "projectPath": str(project_path),
        "projectPendingWorkUnits": 0,
        "projectProgress": 1.0,
        "projectTotalWorkUnits": expected_work_units,
        "status": "COMPLETE",
        "totalWorkUnits": final_status["totalWorkUnits"],
        "workloadID": WORKLOAD_ID,
    }
    prepare_parameters = {"projectPath": str(project_path)}
    run_parameters = {
        "projectManifestSha256": project_hash,
        "projectPath": str(project_path),
    }
    terminal_parameters = {"expectedProjectID": project["id"]}
    prepare = stage_record(
        stage_id="001-prepare-residual-grid",
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
            "id": "002-run-residual-grid",
            "parameters": run_parameters,
            "triggered_by_stage_id": "001-prepare-residual-grid",
        },
        stop=False,
    )
    run = stage_record(
        stage_id="002-run-residual-grid",
        handler_id="openstar.project.run",
        triggered_by="001-prepare-residual-grid",
        parameters=run_parameters,
        result=run_result,
        input_hashes={"projectManifest": project_hash},
        project_ids=(project["id"],),
        next_stage={
            "handler_id": "generic.project.terminal-check",
            "id": "003-terminal-residual-grid",
            "parameters": terminal_parameters,
            "triggered_by_stage_id": "002-run-residual-grid",
        },
        stop=False,
        node_contributions=contributions,
    )
    terminal = stage_record(
        stage_id="003-terminal-residual-grid",
        handler_id="generic.project.terminal-check",
        triggered_by="002-run-residual-grid",
        parameters=terminal_parameters,
        result={
            "completedWorkUnits": expected_work_units,
            "failedWorkUnits": 0,
            "passed": True,
            "projectID": project["id"],
            "rule": "projectID matches and completed+failed == total",
            "totalWorkUnits": expected_work_units,
        },
        input_hashes={},
        project_ids=(project["id"],),
        next_stage=None,
        stop=True,
    )
    stages = [prepare, run, terminal]
    investigation = {
        "created_at": "2026-08-31T00:00:00+00:00",
        "id": INVESTIGATION_ID,
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


def synthetic_winner(*, amplitude=2.0):
    best = {
        "bestAmplitude": amplitude,
        "bestCenter": 0.0,
        "bestGridIndex": 0,
        "bestLogScale": 0.0,
        "bestLogShape": math.log(0.1),
        "bestOffset": 0.0,
        "bestWeightedResidualSumSquares": 0.0,
        "evaluatedCandidateCount": 1,
        "familyID": FAMILY_ID,
        "gridCount": 1,
        "gridStartIndex": 0,
        "invalidCandidateCount": 0,
    }
    return _VerifiedWinner(
        dataset_id="dataset-discovery",
        generic_series_id="series-001",
        grid_index=0,
        center_index=0,
        log_scale_index=0,
        log_shape_index=0,
        center=0.0,
        log_scale=0.0,
        log_shape=math.log(0.1),
        offset=0.0,
        amplitude=amplitude,
        objective=0.0,
        boundary_axes=(),
        result_payload=best,
    )


def synthetic_validation_dataset(*, amplitude, coordinates=None, weight=20.0):
    coordinates = list(
        coordinates
        or [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
    )
    basis = _canonical_curve_basis(
        coordinates,
        center=0.0,
        scale=1.0,
        shape=0.1,
    )
    return {
        "coordinates": coordinates,
        "id": "dataset-validation",
        "inverseVariances": [weight] * len(coordinates),
        "sourceGenericSeriesID": "series-002",
        "values": [0.75 + amplitude * value for value in basis],
    }


class ResidualCrossValidationFixture(ResidualPreparationFixture):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(strict=True)
        with patch(
            "tests.workflows.microlensing.test_prepare_residuals."
            "write_residual_prepared_root",
            side_effect=write_cross_validation_prepared_root,
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
        self.grid = self.root / "residual-grid"
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
            residual_root=self.residual,
            project_id=PROJECT_ID,
            output_root=self.grid,
        )
        self.investigation = write_residual_grid_investigation(
            self.root / "investigations", self.grid
        )

    def validate(self, output_name="cross-validation"):
        output = self.root / output_name
        result = validate_residual_grid(
            self.residual,
            residual_grid_root=self.grid,
            residual_grid_investigation_record=self.investigation,
            output_root=output,
        )
        return output, result

    def assert_rejected(self, pattern=None, *, output_name="rejected"):
        context = (
            self.assertRaisesRegex(ResidualGridValidationError, pattern)
            if pattern
            else self.assertRaises(ResidualGridValidationError)
        )
        with context:
            self.validate(output_name)


class ResidualCrossValidationSuccessTests(ResidualCrossValidationFixture):
    def test_multidataset_run_uses_distinct_counter_scopes(self):
        run_result = read_json(self.investigation)["stages"][1]["result"]
        final_status = run_result["datasets"][-1]
        for field_name in (
            "assignedWorkUnits",
            "completedWorkUnits",
            "failedWorkUnits",
            "pendingWorkUnits",
            "totalWorkUnits",
        ):
            self.assertEqual(final_status[field_name], run_result[field_name])
        self.assertEqual(35, run_result["completedWorkUnits"])
        self.assertEqual(35, run_result["totalWorkUnits"])
        self.assertEqual(105, run_result["projectCompletedWorkUnits"])
        self.assertEqual(105, run_result["projectTotalWorkUnits"])
        self.validate("multidataset-counter-output")

    def test_success_is_deterministic_complete_and_identity_free(self):
        first, first_result = self.validate("first-output")
        second, _ = self.validate("second-output")
        self.assertEqual(serialized_tree(first), serialized_tree(second))
        contract = read_json(first / CONTRACT_RELATIVE_PATH)
        result = read_json(first / RESULT_RELATIVE_PATH)
        self.assertEqual(first_result["result"], result)
        self.assertEqual(CROSS_VALIDATION_CONTRACT_ID, contract["contractID"])
        self.assertEqual(
            CROSS_VALIDATION_RESULT_SCHEMA_ID, result["resultSchemaID"]
        )
        self.assertFalse(result["planetaryInterpretationResolved"])
        self.assertFalse(result["discoveryClaim"])
        self.assertEqual(3, result["admittedSeriesCount"])
        self.assertEqual(6, result["heldOutValidationPairCount"])
        self.assertEqual(POSITIVE_CLASSIFICATION, result["overallClassification"])
        self.assertGreaterEqual(result["confirmedComponentCount"], 1)
        self.assertEqual(CONFIRMED_NEXT_TEST, result["recommendedNextTest"])
        serialized = b"".join(serialized_tree(first).values())
        for token in FORBIDDEN_OUTPUT_TOKENS:
            self.assertNotIn(token.encode("utf-8"), serialized)

    def test_exact_parent_reconstruction_and_canonical_winners(self):
        output, _ = self.validate()
        result = read_json(output / RESULT_RELATIVE_PATH)
        self.assertEqual(
            sha256_bytes((self.grid / "project.json").read_bytes()),
            result["parentHashes"]["residualGridProjectSHA256"],
        )
        self.assertEqual(
            read_json(self.residual / "residual-manifest.json")[
                "parentArtifactHashes"
            ],
            result["parentHashes"]["ancestryArtifactHashes"],
        )
        self.assertEqual(PROJECT_ID, result["parentIDs"]["residualGridProjectID"])
        project = read_json(self.grid / "project.json")
        for component, reference in zip(
            result["validatedComponents"], project["datasets"]
        ):
            dataset = read_json(self.grid / reference["path"])
            winner = component["discoveryWinner"]
            evaluated = _evaluate_candidate(dataset, winner["bestGridIndex"])
            self.assertIsNotNone(evaluated)
            self.assertAlmostEqual(
                winner["bestWeightedResidualSumSquares"],
                evaluated.weighted_residual_sum_squares,
                delta=1.0e-9 * max(
                    1.0, abs(evaluated.weighted_residual_sum_squares)
                ),
            )

    def test_all_ordered_pairs_exclude_self_and_freeze_geometry(self):
        output, _ = self.validate()
        result = read_json(output / RESULT_RELATIVE_PATH)
        ordered_ids = result["orderedAdmittedGenericSeriesIDs"]
        self.assertEqual(["series-001", "series-002", "series-003"], ordered_ids)
        observed_pairs = []
        for component in result["validatedComponents"]:
            discovery_id = component["discoveryGenericSeriesID"]
            winner = component["discoveryWinner"]
            for validation in component["heldOutValidations"]:
                validation_id = validation["validationGenericSeriesID"]
                self.assertNotEqual(discovery_id, validation_id)
                observed_pairs.append((discovery_id, validation_id))
                self.assertEqual(winner["bestCenter"], validation["frozenCenter"])
                self.assertEqual(
                    winner["bestLogScale"], validation["frozenLogScale"]
                )
                self.assertEqual(
                    winner["bestLogShape"], validation["frozenLogShape"]
                )
        expected_pairs = [
            (discovery, validation)
            for discovery in ordered_ids
            for validation in ordered_ids
            if validation != discovery
        ]
        self.assertEqual(expected_pairs, observed_pairs)

    def test_boundary_width_is_reported_without_changing_winner(self):
        output, _ = self.validate()
        component = read_json(output / RESULT_RELATIVE_PATH)[
            "validatedComponents"
        ][0]
        self.assertTrue(component["searchedAxisBoundaryReported"])
        self.assertIn("logScale", component["searchedBoundaryAxes"])
        self.assertTrue(component["widthInterpretationLimitedByBoundary"])
        for validation in component["heldOutValidations"]:
            self.assertEqual(
                component["discoveryWinner"]["bestLogScale"],
                validation["frozenLogScale"],
            )


class FrozenValidationDecisionTests(unittest.TestCase):
    def test_single_dataset_counter_scopes_naturally_coincide(self):
        final_status = {
            "assignedWorkUnits": 0,
            "completedWorkUnits": WORK_UNITS_PER_DATASET,
            "failedWorkUnits": 0,
            "pendingWorkUnits": 0,
            "totalWorkUnits": WORK_UNITS_PER_DATASET,
        }
        run_result = {
            **final_status,
            "projectAssignedWorkUnits": 0,
            "projectCompletedWorkUnits": WORK_UNITS_PER_DATASET,
            "projectFailedWorkUnits": 0,
            "projectPendingWorkUnits": 0,
            "projectTotalWorkUnits": WORK_UNITS_PER_DATASET,
        }
        _verify_run_counter_scopes(
            run_result,
            final_status,
            expected_project_work_units=WORK_UNITS_PER_DATASET,
        )

    def test_same_sign_held_out_component_passes(self):
        result = _held_out_result(
            synthetic_winner(amplitude=2.0),
            synthetic_validation_dataset(amplitude=2.0),
        )
        self.assertTrue(result["amplitudeSignMatchesDiscovery"])
        self.assertGreaterEqual(result["deltaWRSS"], VALIDATION_DELTA_WRSS_THRESHOLD)
        self.assertTrue(result["heldOutValidationGatePassed"])

    def test_opposite_sign_is_rejected(self):
        result = _held_out_result(
            synthetic_winner(amplitude=2.0),
            synthetic_validation_dataset(amplitude=-2.0),
        )
        self.assertEqual("negative", result["fittedAmplitudeSign"])
        self.assertFalse(result["amplitudeSignMatchesDiscovery"])
        self.assertFalse(result["heldOutValidationGatePassed"])

    def test_insufficient_support_is_rejected(self):
        result = _held_out_result(
            synthetic_winner(amplitude=2.0),
            synthetic_validation_dataset(
                amplitude=2.0,
                coordinates=[1.0 + 0.1 * index for index in range(9)],
            ),
        )
        self.assertEqual(
            0, result["positiveWeightSamplesWithinTwoEffectiveWidths"]
        )
        self.assertFalse(result["heldOutValidationGatePassed"])

    def test_failed_fit_is_preserved_as_a_held_out_result(self):
        result = _held_out_result(
            synthetic_winner(amplitude=2.0),
            synthetic_validation_dataset(
                amplitude=2.0,
                coordinates=[0.0] * 9,
            ),
        )
        self.assertEqual("FIT_FAILED", result["status"])
        self.assertFalse(result["heldOutValidationGatePassed"])
        self.assertIsNone(result["deltaWRSS"])
        self.assertTrue(result["decisionReasons"])

    def test_held_out_delta_below_nine_is_rejected(self):
        result = _held_out_result(
            synthetic_winner(amplitude=2.0),
            synthetic_validation_dataset(amplitude=0.001, weight=1.0),
        )
        self.assertLess(result["deltaWRSS"], VALIDATION_DELTA_WRSS_THRESHOLD)
        self.assertFalse(result["heldOutValidationGatePassed"])

    def test_discovery_delta_below_thirty_is_rejected(self):
        self.assertFalse(_discovery_gate(DISCOVERY_DELTA_WRSS_THRESHOLD - 0.001))
        self.assertTrue(_discovery_gate(DISCOVERY_DELTA_WRSS_THRESHOLD))

    def test_overall_classification_depends_only_on_confirmed_count(self):
        self.assertEqual(POSITIVE_CLASSIFICATION, _overall_classification(1))
        self.assertEqual(NEGATIVE_CLASSIFICATION, _overall_classification(0))

    def test_thresholds_are_not_cli_arguments(self):
        option_strings = {
            option
            for action in _parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--discovery-delta-wrss", option_strings)
        self.assertNotIn("--validation-delta-wrss", option_strings)
        self.assertNotIn("--minimum-support", option_strings)

    def test_negative_classification_selects_systematics_review(self):
        self.assertEqual(NEGATIVE_CLASSIFICATION, _overall_classification(0))
        self.assertEqual(
            UNCONFIRMED_NEXT_TEST,
            "RESIDUAL_SYSTEMATICS_AND_ERROR_MODEL_REVIEW",
        )


class ResidualCrossValidationRejectionTests(ResidualCrossValidationFixture):
    def test_mutated_project_dataset_manifest_residual_and_hash_are_rejected(self):
        targets = (
            self.grid / "project.json",
            self.grid / "datasets" / "residual-series-001.json",
            self.grid / "build-manifest.json",
            self.residual / "residual-manifest.json",
            self.residual / "series" / "residual-series-001.json",
        )
        for ordinal, target in enumerate(targets, start=1):
            original = target.read_bytes()
            try:
                target.write_bytes(original + b" ")
                self.assert_rejected(output_name=f"mutated-{ordinal}")
            finally:
                target.write_bytes(original)

        target = self.grid / "build-manifest.json"
        original = target.read_bytes()
        try:
            manifest = read_json(target)
            manifest["outputHashes"]["project"] = "0" * 64
            write_json(target, manifest)
            self.assert_rejected(output_name="mutated-hash")
        finally:
            target.write_bytes(original)

    def test_project_counter_and_mutated_ledger_are_rejected(self):
        original_record = self.investigation.read_bytes()
        stage_document = read_json(self.investigation)["stages"][1]
        ledger = (
            self.investigation.parent
            / "stages"
            / f"{stage_document['id']}.json"
        )
        original_ledger = ledger.read_bytes()
        try:

            def mutate(stage):
                stage["result"]["projectCompletedWorkUnits"] = 104

            rewrite_investigation_stage(self.investigation, 1, mutate)
            self.assert_rejected(
                "project counters",
                output_name="wrong-project-counter",
            )
        finally:
            self.investigation.write_bytes(original_record)
            ledger.write_bytes(original_ledger)

    def test_mutated_stage_ledger_is_rejected(self):
        stage_document = read_json(self.investigation)["stages"][1]
        ledger = (
            self.investigation.parent
            / "stages"
            / f"{stage_document['id']}.json"
        )
        original_ledger = ledger.read_bytes()
        try:
            document = read_json(ledger)
            document["result"]["failedWorkUnits"] = 1
            write_json(ledger, document)
            self.assert_rejected(output_name="mutated-ledger")
        finally:
            ledger.write_bytes(original_ledger)

    def test_unprefixed_counters_must_match_final_dataset(self):
        original_record = self.investigation.read_bytes()
        stage_document = read_json(self.investigation)["stages"][1]
        ledger = (
            self.investigation.parent
            / "stages"
            / f"{stage_document['id']}.json"
        )
        original_ledger = ledger.read_bytes()
        try:

            def mutate(stage):
                stage["result"]["completedWorkUnits"] = 34

            rewrite_investigation_stage(self.investigation, 1, mutate)
            self.assert_rejected(
                "current-dataset counters",
                output_name="wrong-current-dataset-counter",
            )
        finally:
            self.investigation.write_bytes(original_record)
            ledger.write_bytes(original_ledger)

    def test_terminal_counters_must_match_project_totals(self):
        original_record = self.investigation.read_bytes()
        stage_document = read_json(self.investigation)["stages"][2]
        ledger = (
            self.investigation.parent
            / "stages"
            / f"{stage_document['id']}.json"
        )
        original_ledger = ledger.read_bytes()
        try:

            def mutate(stage):
                stage["result"]["completedWorkUnits"] = 104

            rewrite_investigation_stage(self.investigation, 2, mutate)
            self.assert_rejected(
                "terminal check",
                output_name="wrong-terminal-counter",
            )
        finally:
            self.investigation.write_bytes(original_record)
            ledger.write_bytes(original_ledger)

    def test_per_dataset_incomplete_coverage_remains_rejected(self):
        original_record = self.investigation.read_bytes()
        stage_document = read_json(self.investigation)["stages"][1]
        ledger = (
            self.investigation.parent
            / "stages"
            / f"{stage_document['id']}.json"
        )
        original_ledger = ledger.read_bytes()
        try:

            def mutate(stage):
                stage["result"]["datasets"][1]["completedWorkUnits"] = 34

            rewrite_investigation_stage(self.investigation, 1, mutate)
            self.assert_rejected(
                "dataset coverage",
                output_name="incomplete-dataset-coverage",
            )
        finally:
            self.investigation.write_bytes(original_record)
            ledger.write_bytes(original_ledger)

    def test_aggregate_disagreement_and_changed_winner_are_rejected(self):
        original_record = self.investigation.read_bytes()
        run_stage = read_json(self.investigation)["stages"][1]
        ledger = (
            self.investigation.parent
            / "stages"
            / f"{run_stage['id']}.json"
        )
        original_ledger = ledger.read_bytes()
        try:

            def disagree(stage):
                stage["result"]["datasets"][0]["bestAmplitude"] += 1.0

            rewrite_investigation_stage(self.investigation, 1, disagree)
            self.assert_rejected(output_name="aggregate-disagreement")
        finally:
            self.investigation.write_bytes(original_record)
            ledger.write_bytes(original_ledger)

        original_record = self.investigation.read_bytes()
        original_ledger = ledger.read_bytes()
        try:

            def change_winner(stage):
                dataset = stage["result"]["datasets"][0]
                dataset["bestWeightedResidualSumSquares"] += 1.0
                dataset["payload"]["best"][
                    "bestWeightedResidualSumSquares"
                ] += 1.0

            rewrite_investigation_stage(self.investigation, 1, change_winner)
            self.assert_rejected(output_name="changed-winner")
        finally:
            self.investigation.write_bytes(original_record)
            ledger.write_bytes(original_ledger)

    def test_nonfinite_and_malformed_values_are_rejected(self):
        target = self.grid / "datasets" / "residual-series-001.json"
        original = target.read_bytes()
        cases = (
            lambda document: document.pop("values"),
            lambda document: document["values"].__setitem__(0, math.nan),
        )
        for ordinal, mutate in enumerate(cases, start=1):
            try:
                document = read_json(target)
                mutate(document)
                write_json(target, document, allow_nan=True)
                self.assert_rejected(output_name=f"malformed-{ordinal}")
            finally:
                target.write_bytes(original)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_traversal_is_rejected(self):
        link = self.root / "grid-link"
        os.symlink(self.grid, link, target_is_directory=True)
        with self.assertRaises(ResidualGridValidationError):
            validate_residual_grid(
                self.residual,
                residual_grid_root=link,
                residual_grid_investigation_record=self.investigation,
                output_root=self.root / "symlink-output",
            )

    def test_existing_output_is_rejected_without_modification(self):
        output = self.root / "existing-output"
        output.mkdir()
        marker = output / "marker"
        marker.write_text("preserve", encoding="utf-8")
        self.assert_rejected("already exists", output_name="existing-output")
        self.assertEqual("preserve", marker.read_text(encoding="utf-8"))

    def test_transactional_cleanup_after_result_publication_failure(self):
        from workflows.microlensing import validate_residual_grid

        original_write = validate_residual_grid._atomic_write_bytes
        call_count = 0

        def fail_result_write(path, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("synthetic result publication failure")
            return original_write(path, payload)

        output_name = "transaction-output"
        with patch.object(
            validate_residual_grid,
            "_atomic_write_bytes",
            side_effect=fail_result_write,
        ):
            self.assert_rejected("publication failed", output_name=output_name)
        self.assertFalse((self.root / output_name).exists())
        self.assertEqual([], list(self.root.glob(f".{output_name}.*")))

    def test_identity_leakage_is_rejected(self):
        from workflows.microlensing import validate_residual_grid

        original_contract = validate_residual_grid._contract

        def contaminated_contract():
            contract = original_contract()
            contract["leak"] = "OGLE"
            return contract

        with patch.object(
            validate_residual_grid,
            "_contract",
            side_effect=contaminated_contract,
        ):
            self.assert_rejected("identity", output_name="identity-output")

    def test_imported_exception_types_do_not_leak(self):
        imported_errors = (
            CoarseGridBuildError("synthetic coarse failure"),
            RefinementGridBuildError("synthetic refinement failure"),
            ResidualPreparationError("synthetic residual failure"),
            ResidualGridBuildError("synthetic grid failure"),
        )
        for ordinal, imported_error in enumerate(imported_errors, start=1):
            with self.subTest(imported_error=type(imported_error).__name__):
                with patch(
                    "workflows.microlensing.validate_residual_grid."
                    "_validate_residual_grid_impl",
                    side_effect=imported_error,
                ):
                    with self.assertRaises(
                        ResidualGridValidationError
                    ) as raised:
                        self.validate(f"translated-error-{ordinal}")
                self.assertNotIsInstance(raised.exception, type(imported_error))


if __name__ == "__main__":
    unittest.main()
