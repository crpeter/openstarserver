"""Local producer-shaped artifacts; no coordinator, network, or grid search."""

import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openstar_investigation import sha256_json
from openstar_workloads.plugins import morphology_grid as workload
from tests.workflows.microlensing.test_build_anomaly_morphology_coarse_grid import (
    CoarseMorphologyFixture, read_json, serialized_tree, sha256_bytes, write_json,
)
from tests.workflows.microlensing.test_refine_grid import stage_record
from workflows.microlensing import build_anomaly_morphology_coarse_grid as builder
from workflows.microlensing import prepare_anomaly_morphology as preparation
from workflows.microlensing import validate_anomaly_morphology_coarse_grid as validation


def write_investigation(root, coarse):
    """Use the project-smoke ledger and MorphologyGrid reduction field layouts."""
    project_path = coarse / "project.json"
    project = read_json(project_path)
    manifest = read_json(coarse / "build-manifest.json")
    project_hash = sha256_bytes(project_path.read_bytes())
    total = manifest["totalExpectedWorkUnitCount"]
    statuses = []
    for reference, record in zip(project["datasets"], manifest["datasets"]):
        dataset = read_json(coarse / reference["path"])
        # Constant-zero fixtures make every valid candidate tie at zero WRSS;
        # index zero is the canonical first winner, without searching the grid.
        evaluated = workload._evaluate_candidate(dataset, 0)
        if evaluated is None:
            raise AssertionError("fixture accepted index zero must be valid")
        best = workload._candidate_payload(evaluated, dataset["modelClassID"])
        count = record["expectedWorkUnitCount"]
        statuses.append({
            "id": dataset["id"], "workloadID": workload.WORKLOAD_ID,
            "datasetSchemaID": workload.DATASET_SCHEMA_ID,
            "payloadSchemaID": workload.PAYLOAD_SCHEMA_ID,
            "resultSchemaID": workload.RESULT_SCHEMA_ID,
            "morphologyFamilyID": workload.MORPHOLOGY_FAMILY_ID,
            "componentTemplateFamilyID": workload.COMPONENT_TEMPLATE_FAMILY_ID,
            "modelClassID": dataset["modelClassID"],
            "workloadStatus": "MORPHOLOGY_GRID_COMPLETE",
            "morphologyGridStatus": "MORPHOLOGY_GRID_COMPLETE", "coverageComplete": True,
            "completedCandidateCount": record["coarseCandidateCount"],
            "totalCandidateCount": record["coarseCandidateCount"],
            "totalInvalidCandidateCount": 0,
            "assignedWorkUnits": 0, "pendingWorkUnits": 0, "failedWorkUnits": 0,
            "completedWorkUnits": count, "totalWorkUnits": count,
            "nodeContributions": {"generic-node": count}, "progress": 1.0,
            "payload": {"bestCandidate": best},
            **{"best" + key[0].upper() + key[1:]: value for key, value in best.items()},
        })
    final = statuses[-1]
    counters = ("assignedWorkUnits", "pendingWorkUnits", "completedWorkUnits", "failedWorkUnits", "totalWorkUnits")
    result = {
        **{key: final[key] for key in counters},
        "projectAssignedWorkUnits": 0, "projectPendingWorkUnits": 0,
        "projectCompletedWorkUnits": total, "projectFailedWorkUnits": 0,
        "projectTotalWorkUnits": total, "projectProgress": 1.0,
        "nodeContributions": {"generic-node": total}, "datasets": statuses,
        "projectID": project["id"], "projectPath": str(project_path),
        "status": "COMPLETE", "workloadID": workload.WORKLOAD_ID,
    }
    prepare_params = {"projectPath": str(project_path)}
    run_params = {**prepare_params, "projectManifestSha256": project_hash}
    terminal_params = {"expectedProjectID": project["id"]}
    ids = ("001-prepare-project", "002-distributed-project", "003-terminal-check")
    handlers = ("local.project.prepare", "openstar.project.run", "generic.project.terminal-check")
    parameters = (prepare_params, run_params, terminal_params)
    results = (run_params, result, {
        "completedWorkUnits": total, "failedWorkUnits": 0, "passed": True,
        "projectID": project["id"], "totalWorkUnits": total,
        "rule": "projectID matches and completed+failed == total",
    })
    stages = []
    for index in range(3):
        stages.append(stage_record(
            stage_id=ids[index], handler_id=handlers[index],
            triggered_by=ids[index - 1] if index else None,
            parameters=parameters[index], result=results[index],
            input_hashes={"projectManifest": project_hash} if index < 2 else {},
            project_ids=[project["id"]] if index else [],
            next_stage={
                "handler_id": handlers[index + 1], "id": ids[index + 1],
                "parameters": parameters[index + 1], "triggered_by_stage_id": ids[index],
            } if index < 2 else None,
            stop=index == 2, node_contributions={"generic-node": total} if index == 1 else {},
        ))
    investigation_id = "generic-morphology-coarse-investigation"
    path = root / investigation_id / "investigation.json"
    record = {
        "id": investigation_id, "created_at": "2026-09-07T00:00:00+00:00",
        "updated_at": "2026-09-07T00:00:03+00:00", "stages": stages,
        "status": "COMPLETE", "workflow_id": "openstar.workflow.project-smoke.v1",
        "workflow_version": "20.0",
        "metadata": {"coordinator": "http://127.0.0.1:8080", "projectPath": str(project_path)},
    }
    write_record(path, record)
    return path


def write_record(path, record):
    for stage in record["stages"]:
        stage["provenance"]["parameters_hash"] = sha256_json(stage["parameters"])
        stage["provenance"]["result_hash"] = sha256_json(stage["result"])
        write_json(path.parent / "stages" / f"{stage['id']}.json", stage)
    write_json(path, record)


class MorphologyValidationFixture(CoarseMorphologyFixture):
    supported_series = ()

    def _contract(self, axes):
        # Use the real producer's complete contract; only the miniature axis
        # derivation is injected. The inherited fixture retains typed lineage,
        # preparation records, source arrays and all linked file/canonical hashes.
        window = SimpleNamespace(
            minimum=-2.5, maximum=2.5,
            negative=SimpleNamespace(effective_width=math.exp(-3.0) * math.exp(-4.0)),
            positive=SimpleNamespace(effective_width=math.exp(-3.0) * math.exp(-4.0)),
        )
        with patch.object(preparation, "_model_axes", return_value=(axes, self._candidate_mapping(axes))):
            return preparation._morphology_contract(window, None, self.series_ids)

    def _publish_preparation(self, **kwargs):
        super()._publish_preparation(**kwargs)
        # Emit the actual preparation and artifact-manifest producers too,
        # including confirmed component metadata and nested typed ancestry.
        geometry = lambda series_id, center: preparation._ComponentGeometry(
            series_id, center, -3.0, -4.0, math.exp(-3.0) * math.exp(-4.0), ("logScale",),
        )
        window = preparation._PreparedWindow(
            -2.5, 2.5, -1.0, 1.0, geometry(self.series_ids[1], -1.5),
            geometry(self.series_ids[0], 0.5), tuple(self.source_datasets),
        )
        evidence = {
            "discoveryDeltaWRSS": 40.0, "widthInterpretationLimitedByBoundary": True,
            "heldOutValidations": [{"validationGenericSeriesID": self.series_ids[1], "heldOutValidationGatePassed": True}],
        }
        cross = preparation._VerifiedCrossValidation(
            {}, {"parentHashes": self.source_preparation["parentHashes"], "parentIDs": self.source_preparation["parentIDs"]},
            sha256_bytes(b"generic-cross-contract"), sha256_bytes(b"generic-cross-result"),
        )
        records = self.source_preparation["preparedDatasets"]
        contract_hash = self.source_preparation["morphologyContractSHA256"]
        self.source_preparation = preparation._prepare_result(
            contract_sha256=contract_hash, cross=cross, window=window,
            positive_component=evidence, negative_component=evidence, dataset_records=records,
        )
        preparation_path = self.morphology / preparation.PREPARATION_RELATIVE_PATH
        write_json(preparation_path, self.source_preparation)
        self.source_manifest = preparation._artifact_manifest(
            contract_sha256=contract_hash,
            contract_file_sha256=sha256_bytes((self.morphology / preparation.CONTRACT_RELATIVE_PATH).read_bytes()),
            preparation_file_sha256=sha256_bytes(preparation_path.read_bytes()),
            preparation=self.source_preparation, dataset_records=records,
        )
        write_json(self.morphology / preparation.MANIFEST_RELATIVE_PATH, self.source_manifest)

    def _series_document(self, series_id, ordinal, contract_sha256):
        document = super()._series_document(series_id, ordinal, contract_sha256)
        coordinates = list(document["coordinates"])
        # A positive-only search at stride one and the other searches at stride
        # two use centers -2, -1 and ordered positive center -1.75 at index zero.
        if ordinal in self.supported_series:
            coordinates = sorted(set(coordinates + [-2.0, -1.75, -1.0]))
        document.update({
            "coordinates": coordinates, "residualValues": [0.0] * len(coordinates),
            "inverseVariances": [0.0 if x == -0.5 else 1.0 for x in coordinates],
            "sampleCount": len(coordinates), "sourceSampleIndices": list(range(len(coordinates))),
            "inclusionReasons": [["DETERMINISTIC_WINDOW"] for _ in coordinates],
        })
        return document

    def setUp(self):
        super().setUp()
        self.coarse = self.root / "coarse"
        builder.build_anomaly_morphology_coarse_grid(
            self.morphology, project_id="generic-morphology-coarse",
            output_root=self.coarse, maximum_candidates_per_search=8192,
        )
        self.investigation = write_investigation(self.root, self.coarse)

    def report(self, name="report"):
        return validation.validate_anomaly_morphology_coarse_grid(
            self.morphology, coarse_project_root=self.coarse,
            coarse_investigation_record=self.investigation, output_root=self.root / name,
        )

    def reject(self, pattern=None):
        context = self.assertRaisesRegex(validation.AnomalyMorphologyCoarseGridValidationError, pattern) if pattern else self.assertRaises(validation.AnomalyMorphologyCoarseGridValidationError)
        with context:
            self.report("rejected")
        self.assertFalse((self.root / "rejected").exists())


class UnsupportedWinnerTests(MorphologyValidationFixture):
    def test_reports_all_four_unsupported_winners_without_search_or_input_changes(self):
        before = {p: serialized_tree(p) for p in (self.morphology, self.coarse, self.investigation.parent)}
        original = workload._evaluate_candidate
        with patch.object(workload, "_evaluate_candidate", wraps=original) as evaluate, patch.object(
            workload, "_recompute_shard", side_effect=AssertionError("must not search shards")
        ):
            result = self.report()["result"]
        self.assertEqual(4, evaluate.call_count)
        self.assertEqual([0, 0, 0, 0], [call.args[1] for call in evaluate.call_args_list])
        self.assertEqual(validation.UNRESOLVED_SUPPORT, result["overallClassification"])
        self.assertEqual(4, result["unsupportedSearchCount"])
        self.assertIsNone(result["modelPreference"])
        self.assertFalse(result["modelPreferenceResolved"])
        self.assertFalse(result["planetaryInterpretationResolved"])
        self.assertFalse(result["discoveryClaim"])
        self.assertEqual(validation.SUPPORT_FOLLOW_UP, result["recommendedNextTest"])
        self.assertEqual(2, len(result["modelComparisons"]["perSeries"]))
        self.assertIn("deltaBIC", result["modelComparisons"]["rejectOrderedDoubletForIndependentPulses"])
        for path, contents in before.items():
            self.assertEqual(contents, serialized_tree(path))

    def test_deterministic_report_and_provenance_hashes_with_typed_ancestry(self):
        first = self.report("first")
        self.report("second")
        self.assertEqual(serialized_tree(self.root / "first"), serialized_tree(self.root / "second"))
        manifest = first["artifactManifest"]
        self.assertEqual(self.source_preparation["parentHashes"], manifest["parentHashes"])
        self.assertEqual(builder.SOURCE_COARSE_GRID_CONTRACT_ID,
                         manifest["parentHashes"]["ancestryArtifactHashes"]["coarse"]["contractID"])
        self.assertEqual(3, len(manifest["inputHashes"]["coarseStageLedgers"]))
        self.assertEqual(sha256_bytes((self.root / "first" / validation.RESULT_RELATIVE_PATH).read_bytes()),
                         manifest["outputSHA256s"][validation.RESULT_RELATIVE_PATH])

    def test_comparison_gates_preserve_numeric_passes_when_support_fails(self):
        result = self.report()["result"]
        searches = result["searches"]
        positive, ordered = (item["acceptedWinner"] for item in searches[:2])
        positive.update(weightedResidualSumSquares=100.0, bayesianInformationCriterion=140.0)
        ordered.update(weightedResidualSumSquares=70.0, bayesianInformationCriterion=130.0)
        for index, (p, o) in enumerate(zip(positive["seriesFits"], ordered["seriesFits"])):
            p.update(weightedResidualSumSquares=50.0, positiveAmplitudeSign="positive")
            o.update(weightedResidualSumSquares=41.0 if index == 0 else 29.0,
                     negativeAmplitudeSign="negative", positiveAmplitudeSign="positive")
            searches[index + 2]["acceptedWinner"]["seriesFits"][0].update(
                negativeAmplitudeSign="negative", positiveAmplitudeSign="positive",
            )
        independent = {**result["independentAggregate"], "weightedResidualSumSquares": 52.0,
                       "bayesianInformationCriterion": 120.0, "timingConsistent": False}
        comparisons = validation._comparisons(searches, independent, self.source_contract)
        for name in ("preferOrderedDoubletOverPositivePulse", "rejectOrderedDoubletForIndependentPulses"):
            comparison = comparisons[name]
            self.assertTrue(comparison["gates"]["globalDeltaWRSSPassed"])
            self.assertTrue(comparison["gates"]["globalDeltaBICPassed"])
            self.assertTrue(comparison["gates"]["signRequirementsPassed"])
            self.assertFalse(comparison["gates"]["supportRequirementsPassed"])
            self.assertFalse(comparison["passed"])
        for search in searches:
            search["supportRequirementMet"] = True
        comparisons = validation._comparisons(searches, independent, self.source_contract)
        self.assertTrue(comparisons["preferOrderedDoubletOverPositivePulse"]["passed"])
        self.assertTrue(comparisons["rejectOrderedDoubletForIndependentPulses"]["passed"])
        # Thresholds use exact >=, not objective tie tolerances.
        positive["weightedResidualSumSquares"] = math.nextafter(100.0, 0.0)
        comparisons = validation._comparisons(searches, independent, self.source_contract)
        self.assertFalse(comparisons["preferOrderedDoubletOverPositivePulse"]["gates"]["globalDeltaWRSSPassed"])
        independent["timingConsistent"] = True
        comparisons = validation._comparisons(searches, independent, self.source_contract)
        self.assertFalse(comparisons["rejectOrderedDoubletForIndependentPulses"]["passed"])

    def test_complete_multidataset_project_counters_differ_from_final_dataset(self):
        run = read_json(self.investigation)["stages"][1]["result"]
        self.assertGreater(run["projectTotalWorkUnits"], run["totalWorkUnits"])
        self.assertEqual(run["totalWorkUnits"], run["datasets"][-1]["totalWorkUnits"])
        result = self.report()["result"]
        self.assertEqual(run["projectTotalWorkUnits"], result["projectCompletedWorkUnits"])
        self.assertEqual(run["projectTotalWorkUnits"], sum(s["completedWorkUnits"] for s in result["searches"]))

    def test_ordered_positive_center_is_derived_without_payload_field(self):
        ordered = self.report()["result"]["searches"][1]
        parameters = ordered["acceptedWinner"]["parameters"]
        self.assertNotIn("positiveCenter", parameters)
        self.assertEqual(parameters["negativeCenter"] + parameters["separation"], ordered["components"][1]["center"])


class SupportedWinnerTests(MorphologyValidationFixture):
    supported_series = (0, 1)

    def test_supported_winners_keep_sign_gate_and_model_preference_separate(self):
        result = self.report()["result"]
        self.assertEqual(0, result["unsupportedSearchCount"])
        self.assertTrue(result["allAcceptedWinnersMeetSupportRequirement"])
        self.assertEqual("MODEL_COMPARISON_UNRESOLVED", result["overallClassification"])
        self.assertIsNone(result["modelPreference"])
        for comparison in ("preferOrderedDoubletOverPositivePulse", "rejectOrderedDoubletForIndependentPulses"):
            gates = result["modelComparisons"][comparison]["gates"]
            self.assertTrue(gates["supportRequirementsPassed"])
            self.assertFalse(gates["signRequirementsPassed"])


class PartialSupportTests(MorphologyValidationFixture):
    supported_series = (0,)

    def test_shared_component_reports_series_support_individually(self):
        result = self.report()["result"]
        self.assertEqual(3, result["unsupportedSearchCount"])
        support = result["searches"][0]["components"][0]["seriesSupport"]
        self.assertEqual([True, False], [s["supportRequirementMet"] for s in support])
        self.assertTrue(result["searches"][2]["supportRequirementMet"])
        self.assertIsNone(result["modelPreference"])
        self.assertEqual(validation.UNRESOLVED_SUPPORT, result["overallClassification"])


class SupportAndMetricTests(unittest.TestCase):
    def test_inclusive_two_width_threshold_zero_weight_exclusion_and_nearest_distance(self):
        series = {"genericSeriesID": "series-001", "coordinates": [-2.0, 0.0, 2.0, math.nextafter(2.0, math.inf)], "inverseVariances": [1.0, 0.0, 1.0, 1.0]}
        support = validation._component_support(series, 0.0, 0.0, 0.0, 2)
        self.assertEqual(2, support["positiveWeightSamplesWithinTwoEffectiveWidths"])
        self.assertEqual(2.0, support["nearestPositiveWeightDistanceInEffectiveWidths"])
        self.assertTrue(support["supportRequirementMet"])
        series["inverseVariances"] = [0.0, 0.0, 0.0, 1.0]
        support = validation._component_support(series, 0.0, 0.0, 0.0, 1)
        self.assertFalse(support["supportRequirementMet"])
        self.assertGreater(support["nearestPositiveWeightDistanceInEffectiveWidths"], 2.0)

    def test_effective_width_uses_separate_exponentials(self):
        series = {"genericSeriesID": "series-001", "coordinates": [0.0, 1.0], "inverseVariances": [1.0, 1.0]}
        result = validation._component_support(series, 0.0, 1.25, -4.125, 1)
        self.assertEqual(math.exp(1.25) * math.exp(-4.125), result["effectiveWidth"])

    def test_fixed_axes_are_not_searched_boundaries(self):
        dataset = {"modelClassID": workload.POSITIVE_PULSE_ONLY, "candidatesPerWorkUnit": 64, "morphologyGrid": {
            "centerAxis": {"start": 0.0, "step": 1.0, "count": 3},
            "logScaleAxis": {"start": -3.0, "step": 0.25, "count": 2},
            "logShapeAxis": {"values": [-4.0]},
        }}
        axes = validation._axis_boundaries(dataset, 3)  # center 1, scale 1, fixed shape
        self.assertEqual(["INTERIOR", "UPPER_BOUNDARY", "FIXED"], [a["position"] for a in axes])
        self.assertFalse(axes[2]["searchedBoundary"])
        self.assertTrue(axes[2]["fixed"])

    def test_independent_pair_boundaries_use_component_indices(self):
        dataset = {"modelClassID": workload.INDEPENDENT_PULSES, "candidatesPerWorkUnit": 64, "morphologyGrid": {
            "centerAxis": {"start": 0.0, "step": 1.0, "count": 4},
            "negativeLogScaleAxis": {"start": -3.0, "step": 0.25, "count": 1},
            "negativeLogShapeAxis": {"values": [-4.0]},
            "positiveLogScaleAxis": {"start": -3.0, "step": 0.25, "count": 1},
            "positiveLogShapeAxis": {"values": [-4.0]},
        }}
        # Pair (1, 2) has two interior centers despite being pair index three.
        axes = validation._axis_boundaries(dataset, 3)
        self.assertEqual([1, 2], [a["index"] for a in axes[:2]])
        self.assertFalse(any(a["searchedBoundary"] for a in axes))

    def test_global_independent_information_criteria_are_not_summed(self):
        winners = []
        for index, (wrss, n) in enumerate(((10.0, 15), (30.0, 25))):
            winners.append({
                **validation._information_criteria(wrss, n, 9), "gridIndex": index,
                "seriesFits": [{"genericSeriesID": f"series-{index + 1:03d}"}],
                "parameters": {"negativeCenter": float(index), "positiveCenter": float(index + 2)},
            })
        contract = {
            "admittedGenericSeriesIDs": ["series-001", "series-002"],
            "independentAggregation": {"parameterCount": 18},
            "crossSeriesRequirements": {"independentTimingConsistencyTolerance": 1.0, "timingComparisonTolerance": 0.0},
        }
        result = validation._independent_aggregate(winners, contract)
        self.assertEqual((40.0, 40, 18), (result["weightedResidualSumSquares"], result["positiveWeightSampleCount"], result["nominalParameterCount"]))
        self.assertEqual(40.0 + 18 * math.log(40), result["bayesianInformationCriterion"])
        self.assertEqual(40.0 + 36.0 + 2.0 * 18 * 19 / 21, result["correctedAkaikeInformationCriterion"])
        self.assertNotEqual(sum(w["bayesianInformationCriterion"] for w in winners), result["bayesianInformationCriterion"])
        self.assertNotEqual(sum(w["correctedAkaikeInformationCriterion"] for w in winners), result["correctedAkaikeInformationCriterion"])
        self.assertEqual(1.0, result["timingDispersion"])
        self.assertTrue(result["timingConsistent"])
        contract["crossSeriesRequirements"]["independentTimingConsistencyTolerance"] = math.nextafter(1.0, 0.0)
        self.assertFalse(validation._independent_aggregate(winners, contract)["timingConsistent"])

    def test_aicc_is_null_at_and_below_nominal_parameter_threshold(self):
        for n in (18, 19):
            result = validation._information_criteria(3.0, n, 18)
            self.assertIsNone(result["correctedAkaikeInformationCriterion"])
            self.assertFalse(result["correctedAkaikeInformationCriterionDefined"])
        self.assertTrue(validation._information_criteria(3.0, 20, 18)["correctedAkaikeInformationCriterionDefined"])


class MalformedInputTests(MorphologyValidationFixture):
    def test_bad_coverage_and_counter_scopes_with_rehashed_ledgers(self):
        original = read_json(self.investigation)
        mutations = (
            lambda r: r["stages"][1]["result"].update(projectTotalWorkUnits=1),
            lambda r: r["stages"][1]["result"].update(totalWorkUnits=r["stages"][1]["result"]["projectTotalWorkUnits"]),
            lambda r: r["stages"][1]["result"]["datasets"][0].update(coverageComplete=False),
            lambda r: r["stages"][1]["result"]["datasets"][0].update(completedCandidateCount=1),
            lambda r: r["stages"][1]["result"]["datasets"][0].update(completedWorkUnits=True),
            lambda r: r["stages"][1]["result"]["datasets"][0].update(totalInvalidCandidateCount=10**8),
            lambda r: r["stages"][1]["result"]["datasets"].reverse(),
            lambda r: r["stages"][2]["result"].update(completedWorkUnits=1),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = copy.deepcopy(original)
                mutate(record)
                write_record(self.investigation, record)
                self.reject()

    def test_wrong_winner_mapping_summary_and_noncanonical_fit(self):
        original = read_json(self.investigation)
        for field, value in (("gridIndex", 1), ("gridIndex", -1), ("gridIndex", True), ("gridIndex", 10**8), ("weightedResidualSumSquares", 123.0), ("nominalParameterCount", 1)):
            with self.subTest(field=field, value=value):
                record = copy.deepcopy(original)
                status = record["stages"][1]["result"]["datasets"][0]
                status["payload"]["bestCandidate"][field] = value
                status["best" + field[0].upper() + field[1:]] = value
                write_record(self.investigation, record)
                self.reject()
        record = copy.deepcopy(original)
        record["stages"][1]["result"]["datasets"][0]["bestParameters"]["center"] += 0.1
        write_record(self.investigation, record)
        self.reject()

    def test_each_duplicated_winner_field_must_agree(self):
        original = read_json(self.investigation)
        fields = original["stages"][1]["result"]["datasets"][0]["payload"]["bestCandidate"]
        for field in fields:
            with self.subTest(field=field):
                record = copy.deepcopy(original)
                del record["stages"][1]["result"]["datasets"][0]["best" + field[0].upper() + field[1:]]
                write_record(self.investigation, record)
                self.reject("duplicated winner")

    def test_mutated_stage_ledger_and_provenance(self):
        record = read_json(self.investigation)
        stage = record["stages"][1]
        ledger_path = self.investigation.parent / "stages" / f"{stage['id']}.json"
        ledger = read_json(ledger_path)
        ledger["stop"] = True
        write_json(ledger_path, ledger)
        self.reject("ledger")
        write_record(self.investigation, record)
        record["stages"][1]["provenance"]["input_hashes"]["projectManifest"] = "0" * 64
        write_record(self.investigation, record)
        self.reject("provenance")

    def test_malformed_identity_terminal_and_stage_causality(self):
        original = read_json(self.investigation)
        mutations = (
            lambda r: r.update(id="different-investigation"),
            lambda r: r.update(status="RUNNING"),
            lambda r: r["metadata"].update(projectPath=str(self.root / "other.json")),
            lambda r: r["stages"][1].update(triggered_by_stage_id="003-terminal-check"),
            lambda r: r["stages"][2].update(stop=False),
            lambda r: r["stages"][1]["result"]["datasets"][0]["nodeContributions"].update(unknown=1),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = copy.deepcopy(original)
                mutate(record)
                write_record(self.investigation, record)
                self.reject()

    def test_mutated_coarse_source_artifacts_and_nested_typed_ancestry(self):
        paths = [self.coarse / "project.json", self.coarse / "coarse-grid-contract.json", self.coarse / "build-manifest.json", self.coarse / builder.POSITIVE_DATASET_RELATIVE_PATH, self.morphology / "artifact-manifest.json"]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                value = read_json(path)
                value["unexpected"] = True
                write_json(path, value)
                self.reject()
                path.write_bytes(original)
        path = self.coarse / "build-manifest.json"
        value = read_json(path)
        value["parentHashes"]["ancestryArtifactHashes"]["coarse"]["contractID"] = "wrong-contract"
        write_json(path, value)
        self.reject("manifest")

    def test_nonfinite_and_duplicate_json_inputs(self):
        original = self.investigation.read_bytes()
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(token=token):
                text = original.decode().replace('"projectProgress": 1.0', f'"projectProgress": {token}')
                self.assertNotEqual(original.decode(), text)
                self.investigation.write_text(text)
                self.reject()
        self.investigation.write_bytes(original.replace(b'"status": "COMPLETE"', b'"status": "COMPLETE", "status": "COMPLETE"', 1))
        self.reject()

    def test_existing_output_and_input_nested_output_are_preserved(self):
        self.report()
        before = serialized_tree(self.root / "report")
        with self.assertRaisesRegex(validation.AnomalyMorphologyCoarseGridValidationError, "already exists"):
            self.report()
        self.assertEqual(before, serialized_tree(self.root / "report"))
        with self.assertRaisesRegex(validation.AnomalyMorphologyCoarseGridValidationError, "inside an input"):
            validation.validate_anomaly_morphology_coarse_grid(
                self.morphology, coarse_project_root=self.coarse,
                coarse_investigation_record=self.investigation, output_root=self.morphology / "report",
            )

    def test_symlinked_input_and_ledger_are_rejected(self):
        path = self.coarse / builder.POSITIVE_DATASET_RELATIVE_PATH
        original = path.read_bytes()
        target = self.root / "saved-dataset.json"
        target.write_bytes(original)
        path.unlink()
        path.symlink_to(target)
        self.reject("symlink")
        path.unlink()
        path.write_bytes(original)
        ledger = next((self.investigation.parent / "stages").iterdir())
        target = self.root / "saved-ledger.json"
        target.write_bytes(ledger.read_bytes())
        ledger.unlink()
        ledger.symlink_to(target)
        self.reject("symlink")

    def test_publication_failure_leaves_no_partial_report(self):
        with patch.object(validation, "_atomic_write_bytes", side_effect=OSError("injected publication failure")):
            self.reject("publication failure")
        self.assertFalse(list(self.root.glob(".rejected.*")))

    def test_rehashed_unsupported_decision_contract_is_rejected(self):
        self.source_contract["decisionRules"]["preferOrderedDoubletOverPositivePulse"]["globalDeltaWRSSAtLeast"] = 0.0
        self._refresh_contract_family()
        self.reject("decision rules")


if __name__ == "__main__":
    unittest.main()
