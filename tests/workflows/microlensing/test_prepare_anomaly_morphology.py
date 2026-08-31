import copy
import os
import unittest
from unittest.mock import patch

from tests.workflows.microlensing.test_refine_grid import (
    read_json,
    serialized_tree,
    sha256_bytes,
    write_json,
)
from tests.workflows.microlensing.test_validate_residual_grid import (
    ResidualCrossValidationFixture,
)
from workflows.microlensing.prepare_anomaly_morphology import (
    ARTIFACT_MANIFEST_SCHEMA_ID,
    CONTRACT_RELATIVE_PATH,
    DATASET_DIRECTORY,
    INDEPENDENT_PULSES,
    MANIFEST_RELATIVE_PATH,
    MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE,
    MORPHOLOGY_CONTRACT_ID,
    MORPHOLOGY_DATASET_SCHEMA_ID,
    MORPHOLOGY_PREPARATION_SCHEMA_ID,
    NEXT_TEST,
    ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    POSITIVE_PULSE_ONLY,
    PREPARATION_RELATIVE_PATH,
    AnomalyMorphologyPreparationError,
    _assert_identity_isolated,
    _parser,
    _source_arrays,
    prepare_anomaly_morphology,
)
from workflows.microlensing.validate_residual_grid import (
    CONFIRMED_STATUS,
    POSITIVE_CLASSIFICATION,
    UNCONFIRMED_STATUS,
)


class AnomalyMorphologyFixture(ResidualCrossValidationFixture):
    def setUp(self):
        super().setUp()
        self.cross_root, published = self.validate("cross-validation-parent")
        self.expected_cross_result = copy.deepcopy(published["result"])
        self._make_exactly_one_positive_confirmation()

    def _write_expected_cross_result(self):
        write_json(
            self.cross_root / "residual-cross-validation.json",
            self.expected_cross_result,
        )

    def _make_exactly_one_positive_confirmation(self):
        result = self.expected_cross_result
        components = result["validatedComponents"]
        if len(components) < 2:
            raise AssertionError("fixture needs two admitted discovery series")
        positive = components[0]
        negative = components[1]
        positive_id = positive["discoveryGenericSeriesID"]
        validation_id = negative["discoveryGenericSeriesID"]
        center_step = read_json(
            self.grid / "residual-search-contract.json"
        )["curveGrid"]["centerAxis"]["step"]

        for component in components:
            component["componentStatus"] = UNCONFIRMED_STATUS
            component["crossSeriesConfirmed"] = False
            component["heldOutPassingSeriesCount"] = 0
            for validation in component["heldOutValidations"]:
                validation["heldOutValidationGatePassed"] = False

        positive["componentStatus"] = CONFIRMED_STATUS
        positive["crossSeriesConfirmed"] = True
        positive["discoveryAmplitudeSign"] = "positive"
        positive["discoveryCoverageComplete"] = True
        positive["discoveryGatePassed"] = True
        positive["discoveryDeltaWRSS"] = 40.0
        positive["discoveryWinner"]["bestAmplitude"] = abs(
            positive["discoveryWinner"]["bestAmplitude"]
        ) or 1.0
        positive["searchedAxisBoundaryReported"] = True
        positive["searchedBoundaryAxes"] = ["logScale"]
        positive["widthInterpretationLimitedByBoundary"] = True
        passed = next(
            item
            for item in positive["heldOutValidations"]
            if item["validationGenericSeriesID"] == validation_id
        )
        passed.update(
            {
                "amplitudeSignMatchesDiscovery": True,
                "deltaWRSS": 9.25,
                "discoveryGenericSeriesID": positive_id,
                "fittedAmplitude": 1.5,
                "fittedAmplitudeSign": "positive",
                "heldOutValidationGatePassed": True,
                "positiveWeightSamplesWithinTwoEffectiveWidths": 2,
                "status": "EVALUATED",
            }
        )
        positive["heldOutPassingSeriesCount"] = 1

        negative["componentStatus"] = UNCONFIRMED_STATUS
        negative["crossSeriesConfirmed"] = False
        negative["discoveryAmplitudeSign"] = "negative"
        negative["discoveryCoverageComplete"] = True
        negative["discoveryGatePassed"] = True
        negative["discoveryDeltaWRSS"] = 38.0
        negative["discoveryWinner"]["bestAmplitude"] = -abs(
            negative["discoveryWinner"]["bestAmplitude"] or 1.0
        )
        negative["discoveryWinner"]["bestCenter"] = (
            positive["discoveryWinner"]["bestCenter"] - center_step
        )
        negative["searchedAxisBoundaryReported"] = True
        negative["searchedBoundaryAxes"] = ["logScale"]
        negative["widthInterpretationLimitedByBoundary"] = True

        result["confirmedComponentCount"] = 1
        result["overallClassification"] = POSITIVE_CLASSIFICATION
        result["planetaryInterpretationResolved"] = False
        result["discoveryClaim"] = False
        result["recommendedNextTest"] = (
            "BLIND_MICROLENSING_ANOMALY_MORPHOLOGY_MODELING"
        )
        self._write_expected_cross_result()

    def prepare(self, output_name="morphology-output"):
        output = self.root / output_name
        with patch(
            "workflows.microlensing.prepare_anomaly_morphology."
            "_expected_cross_validation_result",
            return_value=copy.deepcopy(self.expected_cross_result),
        ):
            published = prepare_anomaly_morphology(
                self.residual,
                residual_grid_root=self.grid,
                residual_grid_investigation_record=self.investigation,
                cross_validation_root=self.cross_root,
                output_root=output,
            )
        return output, published

    def assert_rejected(self, pattern=None, *, output_name="rejected"):
        context = (
            self.assertRaisesRegex(AnomalyMorphologyPreparationError, pattern)
            if pattern
            else self.assertRaises(AnomalyMorphologyPreparationError)
        )
        with context:
            self.prepare(output_name)


class AnomalyMorphologySuccessTests(AnomalyMorphologyFixture):
    def test_success_is_deterministic_complete_and_does_not_fit_models(self):
        first, first_published = self.prepare("first")
        second, _ = self.prepare("second")
        self.assertEqual(serialized_tree(first), serialized_tree(second))
        preparation = read_json(first / PREPARATION_RELATIVE_PATH)
        contract = read_json(first / CONTRACT_RELATIVE_PATH)
        manifest = read_json(first / MANIFEST_RELATIVE_PATH)
        self.assertEqual(first_published["preparation"], preparation)
        self.assertEqual(
            MORPHOLOGY_PREPARATION_SCHEMA_ID, preparation["resultSchemaID"]
        )
        self.assertEqual(MORPHOLOGY_CONTRACT_ID, contract["contractID"])
        self.assertEqual(
            ARTIFACT_MANIFEST_SCHEMA_ID,
            manifest["artifactManifestSchemaID"],
        )
        self.assertFalse(preparation["planetaryInterpretationResolved"])
        self.assertFalse(preparation["discoveryClaim"])
        self.assertEqual(NEXT_TEST, preparation["recommendedNextTest"])
        self.assertNotIn("fitResults", preparation)
        self.assertEqual(
            {
                CONTRACT_RELATIVE_PATH,
                DATASET_DIRECTORY,
                MANIFEST_RELATIVE_PATH,
                PREPARATION_RELATIVE_PATH,
            },
            {entry.name for entry in first.iterdir()},
        )
        self.assertEqual(
            len(preparation["admittedGenericSeriesIDs"]),
            len(list((first / DATASET_DIRECTORY).iterdir())),
        )
        self.assertEqual(
            [
                POSITIVE_PULSE_ONLY,
                ORDERED_NEGATIVE_POSITIVE_DOUBLET,
                INDEPENDENT_PULSES,
            ],
            preparation["modelClassIDs"],
        )

    def test_exact_parent_and_source_hashes_are_persisted(self):
        output, _ = self.prepare()
        preparation = read_json(output / PREPARATION_RELATIVE_PATH)
        self.assertEqual(
            sha256_bytes((self.grid / "project.json").read_bytes()),
            preparation["parentHashes"]["residualGridProjectSHA256"],
        )
        self.assertEqual(
            sha256_bytes(self.investigation.read_bytes()),
            preparation["parentHashes"][
                "residualGridInvestigationRecordSHA256"
            ],
        )
        self.assertEqual(
            sha256_bytes(
                (self.cross_root / "residual-cross-validation.json").read_bytes()
            ),
            preparation["parentHashes"]["crossValidationResultFileSHA256"],
        )
        manifest = read_json(output / MANIFEST_RELATIVE_PATH)
        for record in preparation["preparedDatasets"]:
            dataset_path = output / record["outputFile"]
            dataset = read_json(dataset_path)
            self.assertEqual(
                sha256_bytes(dataset_path.read_bytes()), record["outputSHA256"]
            )
            self.assertEqual(
                dataset["sourceResidualSeriesSHA256"],
                record["sourceResidualSeriesSHA256"],
            )
            self.assertEqual(
                record["outputSHA256"],
                manifest["outputSHA256s"][record["outputFile"]],
            )

    def test_window_is_mechanical_and_preserves_exact_source_samples(self):
        output, _ = self.prepare()
        preparation = read_json(output / PREPARATION_RELATIVE_PATH)
        bounds = preparation["preparedCoordinateBounds"]
        self.assertLess(bounds["minimum"], bounds["anomalyCoreMinimum"])
        self.assertLess(bounds["anomalyCoreMinimum"], bounds["anomalyCoreMaximum"])
        self.assertLess(bounds["anomalyCoreMaximum"], bounds["maximum"])
        for record in preparation["preparedDatasets"]:
            dataset = read_json(output / record["outputFile"])
            source = read_json(
                self.residual
                / "series"
                / f"residual-{dataset['genericSeriesID']}.json"
            )
            indices = dataset["sourceSampleIndices"]
            self.assertEqual(
                [source["coordinates"][index] for index in indices],
                dataset["coordinates"],
            )
            self.assertEqual(
                [source["residualValues"][index] for index in indices],
                dataset["residualValues"],
            )
            self.assertEqual(
                [source["inverseVariances"][index] for index in indices],
                dataset["inverseVariances"],
            )
            support = dataset["positiveWeightSupport"]
            self.assertGreaterEqual(
                support["leftBaseline"],
                MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE,
            )
            self.assertGreaterEqual(
                support["rightBaseline"],
                MINIMUM_BASELINE_POSITIVE_WEIGHT_SAMPLES_PER_SIDE,
            )
            self.assertGreaterEqual(
                support["precedingNegativeWithinTwoEffectiveWidths"], 1
            )
            self.assertGreaterEqual(
                support["confirmedPositiveWithinTwoEffectiveWidths"], 1
            )
            self.assertEqual(
                MORPHOLOGY_DATASET_SCHEMA_ID,
                dataset["morphologyDatasetSchemaID"],
            )
            self.assertTrue(all(dataset["inclusionReasons"]))

    def test_contract_predeclares_axes_models_mapping_and_decisions(self):
        output, _ = self.prepare()
        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        axes = contract["parameterAxes"]
        self.assertGreater(axes["CENTER"]["count"], 1)
        self.assertGreater(axes["CENTER"]["step"], 0.0)
        self.assertGreater(axes["SEPARATION"]["start"], 0.0)
        self.assertGreater(axes["LOG_SCALE"]["count"], 16)
        self.assertTrue(
            contract["axisRules"]["strictlyPositiveFiniteWidthsRequired"]
        )
        mapping = contract["candidateIndexMapping"]
        offsets = mapping["globalCandidateOffsets"]
        counts = mapping["candidateCounts"]
        self.assertEqual(0, offsets[POSITIVE_PULSE_ONLY])
        self.assertEqual(
            counts[POSITIVE_PULSE_ONLY],
            offsets[ORDERED_NEGATIVE_POSITIVE_DOUBLET],
        )
        self.assertEqual(
            counts[POSITIVE_PULSE_ONLY]
            + counts[ORDERED_NEGATIVE_POSITIVE_DOUBLET],
            offsets[INDEPENDENT_PULSES],
        )
        self.assertEqual(
            sum(counts.values()), mapping["globalCandidateCount"]
        )
        for model_id, ordered_axes in mapping[
            "axisOrderingByModelClass"
        ].items():
            sources = mapping["axisSourceByModelClass"][model_id]
            self.assertTrue(all(axis in sources for axis in ordered_axes))
            self.assertTrue(
                all(source in axes for source in sources.values())
            )
        self.assertEqual(
            ["negative", "positive"],
            contract["modelClasses"][ORDERED_NEGATIVE_POSITIVE_DOUBLET][
                "amplitudeConstraints"
            ],
        )
        self.assertTrue(
            contract["modelClasses"][ORDERED_NEGATIVE_POSITIVE_DOUBLET][
                "strictTemporalOrdering"
            ]
        )
        self.assertFalse(
            contract["modelClasses"][INDEPENDENT_PULSES][
                "centersSharedAcrossSeries"
            ]
        )
        self.assertIn("AICc", contract["comparisonMetrics"])
        self.assertIn("BIC", contract["comparisonMetrics"])
        self.assertEqual(
            30.0,
            contract["decisionRules"]["preferOrderedDoubletOverPositivePulse"][
                "globalDeltaWRSSAtLeast"
            ],
        )
        self.assertEqual(
            18.0,
            contract["decisionRules"][
                "rejectOrderedDoubletForIndependentPulses"
            ]["globalDeltaWRSSAtLeast"],
        )

    def test_previous_minimum_width_boundary_is_extended_and_unresolved(self):
        output, _ = self.prepare()
        contract = read_json(output / CONTRACT_RELATIVE_PATH)
        preparation = read_json(output / PREPARATION_RELATIVE_PATH)
        previous_log_scale = preparation["confirmedComponentProvenance"][
            "positive"
        ]["frozenLogScale"]
        self.assertLess(
            contract["parameterAxes"]["LOG_SCALE"]["start"],
            previous_log_scale,
        )
        self.assertFalse(preparation["widthInterpretationResolved"])
        self.assertGreater(
            contract["effectiveWidthBounds"]["minimum"], 0.0
        )
        self.assertLessEqual(
            contract["effectiveWidthBounds"]["maximum"],
            contract["effectiveWidthBounds"]["preparedWindowMaximum"],
        )
        self.assertIn(
            "unresolved",
            contract["interpretationLimits"]["boundaryWinnerRule"],
        )

    def test_cli_exposes_only_input_and_output_roots(self):
        option_strings = {
            option
            for action in _parser()._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {
                "--residual-root",
                "--residual-grid-root",
                "--residual-grid-investigation-record",
                "--cross-validation-root",
                "--output-root",
            }.issubset(option_strings)
        )
        self.assertNotIn("--minimum-width", option_strings)
        self.assertNotIn("--doublet-threshold", option_strings)


class AnomalyMorphologyRejectionTests(AnomalyMorphologyFixture):
    def test_zero_or_multiple_confirmed_components_are_rejected(self):
        for count in (0, 2):
            with self.subTest(count=count):
                original = copy.deepcopy(self.expected_cross_result)
                components = self.expected_cross_result["validatedComponents"]
                for index, component in enumerate(components):
                    confirmed = index < count
                    component["crossSeriesConfirmed"] = confirmed
                    component["componentStatus"] = (
                        CONFIRMED_STATUS if confirmed else UNCONFIRMED_STATUS
                    )
                self.expected_cross_result["confirmedComponentCount"] = count
                self._write_expected_cross_result()
                self.assert_rejected("exactly one", output_name=f"count-{count}")
                self.expected_cross_result = original
                self._write_expected_cross_result()

    def test_wrong_classification_or_next_test_is_rejected(self):
        for field_name, value in (
            ("overallClassification", "NO_REPRODUCIBLE_LOCALIZED_RESIDUAL_STRUCTURE"),
            ("recommendedNextTest", "RESIDUAL_SYSTEMATICS_AND_ERROR_MODEL_REVIEW"),
        ):
            with self.subTest(field_name=field_name):
                original = self.expected_cross_result[field_name]
                self.expected_cross_result[field_name] = value
                self._write_expected_cross_result()
                self.assert_rejected(output_name=f"wrong-{field_name}")
                self.expected_cross_result[field_name] = original
                self._write_expected_cross_result()

    def test_malformed_or_contradictory_held_out_evidence_is_rejected(self):
        positive = self.expected_cross_result["validatedComponents"][0]
        passed = next(
            item
            for item in positive["heldOutValidations"]
            if item["heldOutValidationGatePassed"]
        )
        mutations = (
            ("validationGenericSeriesID", positive["discoveryGenericSeriesID"]),
            ("amplitudeSignMatchesDiscovery", False),
            ("fittedAmplitudeSign", "negative"),
            ("deltaWRSS", 8.99),
            ("positiveWeightSamplesWithinTwoEffectiveWidths", 0),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name):
                original = passed[field_name]
                passed[field_name] = value
                self._write_expected_cross_result()
                self.assert_rejected(output_name=f"held-out-{field_name}")
                passed[field_name] = original
                self._write_expected_cross_result()

    def test_mutated_parent_artifacts_hashes_and_ledgers_are_rejected(self):
        targets = (
            self.residual / "series" / "residual-series-001.json",
            self.grid / "datasets" / "residual-series-001.json",
            self.cross_root / "residual-cross-validation-contract.json",
            self.cross_root / "residual-cross-validation.json",
        )
        for ordinal, target in enumerate(targets):
            with self.subTest(target=target.name):
                original = target.read_bytes()
                document = read_json(target)
                first_key = next(iter(document))
                document[first_key] = "mutated"
                write_json(target, document)
                self.assert_rejected(output_name=f"mutated-{ordinal}")
                target.write_bytes(original)

        original = self.investigation.read_bytes()
        investigation = read_json(self.investigation)
        investigation["workflow_id"] = "invalid.workflow"
        write_json(self.investigation, investigation)
        self.assert_rejected(output_name="mutated-investigation")
        self.investigation.write_bytes(original)

        ledger = self.investigation.parent / "stages" / "002-run-residual-grid.json"
        original = ledger.read_bytes()
        stage = read_json(ledger)
        stage["result"]["projectCompletedWorkUnits"] -= 1
        write_json(ledger, stage)
        self.assert_rejected("ledger", output_name="mutated-ledger")
        ledger.write_bytes(original)

    def test_nonfinite_and_malformed_source_values_are_rejected(self):
        source_path = self.residual / "series" / "residual-series-001.json"
        original = source_path.read_bytes()
        source = read_json(source_path)
        source["residualValues"][0] = float("inf")
        write_json(source_path, source, allow_nan=True)
        self.assert_rejected(output_name="nonfinite")
        source_path.write_bytes(original)

        source = read_json(source_path)
        source["inverseVariances"][0] = -1.0
        write_json(source_path, source)
        self.assert_rejected(output_name="negative-weight")
        source_path.write_bytes(original)

    def test_insufficient_baseline_or_component_support_is_rejected(self):
        source_path = self.residual / "series" / "residual-series-001.json"
        original = source_path.read_bytes()
        source = read_json(source_path)
        source["inverseVariances"] = [0.0] * len(source["inverseVariances"])
        write_json(source_path, source)
        self.assert_rejected(output_name="support")
        source_path.write_bytes(original)

    def test_symlink_traversal_is_rejected(self):
        alias = self.root / "cross-validation-alias"
        try:
            os.symlink(self.cross_root, alias, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks are unavailable")
        output = self.root / "symlink-output"
        with patch(
            "workflows.microlensing.prepare_anomaly_morphology."
            "_expected_cross_validation_result",
            return_value=copy.deepcopy(self.expected_cross_result),
        ):
            with self.assertRaises(AnomalyMorphologyPreparationError):
                prepare_anomaly_morphology(
                    self.residual,
                    residual_grid_root=self.grid,
                    residual_grid_investigation_record=self.investigation,
                    cross_validation_root=alias,
                    output_root=output,
                )

    def test_existing_output_is_rejected_without_modification(self):
        output = self.root / "existing-output"
        output.mkdir()
        marker = output / "marker"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(
            AnomalyMorphologyPreparationError, "already exists"
        ):
            self.prepare("existing-output")
        self.assertEqual("preserve", marker.read_text(encoding="utf-8"))

    def test_transactional_cleanup_after_publication_failure(self):
        from workflows.microlensing import prepare_anomaly_morphology as module

        original_write = module._atomic_write_bytes
        call_count = 0

        def fail_after_contract(path, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("synthetic publication failure")
            original_write(path, payload)

        with patch.object(module, "_atomic_write_bytes", fail_after_contract):
            self.assert_rejected("publication", output_name="atomic-failure")
        self.assertFalse((self.root / "atomic-failure").exists())
        self.assertEqual(
            [], list(self.root.glob(".atomic-failure.*"))
        )

    def test_public_exception_type_wraps_imported_builder_failures(self):
        from workflows.microlensing.prepare_residuals import (
            ResidualPreparationError,
        )

        with patch(
            "workflows.microlensing.prepare_anomaly_morphology."
            "_verify_residual_root",
            side_effect=ResidualPreparationError("synthetic imported failure"),
        ):
            with self.assertRaises(AnomalyMorphologyPreparationError):
                self.prepare("wrapped-error")


class IdentityIsolationTests(unittest.TestCase):
    def test_nonfinite_negative_weight_and_malformed_arrays_fail_closed(self):
        base = {
            "coordinates": [0.0, 1.0],
            "inverseVariances": [1.0, 1.0],
            "residualValues": [0.0, 1.0],
        }
        mutations = (
            ("coordinates", [0.0, float("inf")]),
            ("inverseVariances", [1.0, -1.0]),
            ("residualValues", [0.0]),
            ("coordinates", [0.0, 0.0]),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name, value=value):
                source = copy.deepcopy(base)
                source[field_name] = value
                with self.assertRaises(AnomalyMorphologyPreparationError):
                    _source_arrays(source)

    def test_exact_identity_fields_are_rejected_without_short_value_scans(self):
        _assert_identity_isolated(
            {
                "genericSeriesID": "id",
                "residualValues": [0.0, 1.0],
                "note": "ordinary identity-free generic evidence",
            }
        )
        for forbidden_key in (
            "eventName",
            "star_id",
            "archiveURL",
            "sourceFilename",
            "observatoryID",
            "citation",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaises(AnomalyMorphologyPreparationError):
                    _assert_identity_isolated({forbidden_key: "generic-value"})


if __name__ == "__main__":
    unittest.main()
