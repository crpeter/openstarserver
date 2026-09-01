import copy
import math
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from openstar_workloads.discovery import discover_workloads
from openstar_workloads.plugins import morphology_grid


def linear_axis(start=0.0, step=1.0, count=1):
    return {"start": start, "step": step, "count": count}


def explicit_axis(*values):
    return {"values": list(values)}


def canonical_basis(coordinate, center, log_scale, log_shape):
    scale = math.exp(log_scale)
    shape = math.exp(log_shape)
    difference = coordinate - center
    z = difference / scale
    shape_squared = shape * shape
    z_squared = z * z
    u_squared = shape_squared + z_squared
    u = math.sqrt(u_squared)
    numerator = u_squared + 2.0
    rooted = math.sqrt(u_squared + 4.0)
    denominator = u * rooted
    return numerator / denominator


def generic_series(
    series_id,
    *,
    offset=0.5,
    positive_amplitude=2.0,
    negative_amplitude=None,
    negative_center=-0.5,
    positive_center=0.5,
    weights=None,
):
    coordinates = [-2.0, -1.0, 0.0, 1.0, 2.0]
    if weights is None:
        weights = [1.0, 2.0, 0.0, 3.0, 1.0]
    positive = [
        canonical_basis(coordinate, positive_center, 0.0, 0.0)
        for coordinate in coordinates
    ]
    values = [
        offset + positive_amplitude * basis
        for basis in positive
    ]
    if negative_amplitude is not None:
        negative = [
            canonical_basis(coordinate, negative_center, 0.0, 0.0)
            for coordinate in coordinates
        ]
        values = [
            value + negative_amplitude * basis
            for value, basis in zip(values, negative)
        ]
    return {
        "genericSeriesID": series_id,
        "coordinates": coordinates,
        "values": values,
        "inverseVariances": list(weights),
    }


def positive_dataset(series_count=2):
    series = [
        generic_series(
            f"series-{index + 1:03d}",
            offset=0.5 + index,
            positive_amplitude=2.0 + index,
            positive_center=0.0,
        )
        for index in range(series_count)
    ]
    return {
        "id": "generic-positive-grid",
        "datasetSchemaID": morphology_grid.DATASET_SCHEMA_ID,
        "morphologyFamilyID": morphology_grid.MORPHOLOGY_FAMILY_ID,
        "componentTemplateFamilyID": (
            morphology_grid.COMPONENT_TEMPLATE_FAMILY_ID
        ),
        "modelClassID": morphology_grid.POSITIVE_PULSE_ONLY,
        "series": series,
        "morphologyGrid": {
            "centerAxis": linear_axis(-0.5, 0.5, 3),
            "logScaleAxis": linear_axis(-0.5, 0.5, 3),
            "logShapeAxis": explicit_axis(-0.5, 0.0, 0.5),
        },
        "candidatesPerWorkUnit": 5,
        "executionContractID": morphology_grid.EXECUTION_CONTRACT_ID,
        "executionContractVersion": (
            morphology_grid.EXECUTION_CONTRACT_VERSION
        ),
    }


def ordered_dataset():
    dataset = positive_dataset()
    dataset["id"] = "generic-ordered-grid"
    dataset["modelClassID"] = (
        morphology_grid.ORDERED_NEGATIVE_POSITIVE_DOUBLET
    )
    dataset["series"] = [
        generic_series(
            "series-001",
            offset=0.75,
            negative_amplitude=-1.25,
            positive_amplitude=2.5,
        ),
        generic_series(
            "series-002",
            offset=-0.25,
            negative_amplitude=-2.0,
            positive_amplitude=1.5,
        ),
    ]
    dataset["morphologyGrid"] = {
        "negativeCenterAxis": linear_axis(-0.5, 0.5, 2),
        "separationAxis": linear_axis(1.0, 0.5, 1),
        "negativeLogScaleAxis": linear_axis(0.0, 0.5, 1),
        "negativeLogShapeAxis": explicit_axis(0.0),
        "positiveLogScaleAxis": linear_axis(0.0, 0.5, 1),
        "positiveLogShapeAxis": explicit_axis(0.0),
    }
    dataset["candidatesPerWorkUnit"] = 1
    return dataset


def independent_dataset():
    dataset = positive_dataset(series_count=1)
    dataset["id"] = "generic-independent-grid"
    dataset["modelClassID"] = morphology_grid.INDEPENDENT_PULSES
    dataset["series"] = [
        generic_series(
            "series-001",
            offset=0.75,
            negative_amplitude=-1.25,
            positive_amplitude=2.5,
        )
    ]
    dataset["morphologyGrid"] = {
        "centerAxis": linear_axis(-1.0, 0.5, 5),
        "negativeLogScaleAxis": linear_axis(0.0, 0.5, 1),
        "negativeLogShapeAxis": explicit_axis(0.0),
        "positiveLogScaleAxis": linear_axis(0.0, 0.5, 1),
        "positiveLogShapeAxis": explicit_axis(0.0),
    }
    dataset["candidatesPerWorkUnit"] = 4
    return dataset


def worker_payload(dataset, shard):
    evaluations = []
    invalid_count = 0
    start = shard["gridStartIndex"]
    for grid_index in range(start, start + shard["gridCount"]):
        evaluation = morphology_grid._evaluate_candidate(dataset, grid_index)
        if evaluation is None:
            invalid_count += 1
        else:
            evaluations.append(evaluation)
    if not evaluations:
        raise AssertionError("synthetic shard unexpectedly has no valid candidate")
    best = evaluations[0]
    for evaluation in evaluations[1:]:
        limit = morphology_grid._objective_limit(
            evaluation.weighted_residual_sum_squares,
            best.weighted_residual_sum_squares,
        )
        if (
            evaluation.weighted_residual_sum_squares
            < best.weighted_residual_sum_squares - limit
        ):
            best = evaluation
    return {
        "morphologyFamilyID": morphology_grid.MORPHOLOGY_FAMILY_ID,
        "modelClassID": dataset["modelClassID"],
        "gridStartIndex": start,
        "gridCount": shard["gridCount"],
        "bestCandidate": morphology_grid._candidate_payload(
            best,
            dataset["modelClassID"],
        ),
        "evaluatedCandidateCount": shard["gridCount"],
        "invalidCandidateCount": invalid_count,
    }


def result_for(dataset, shard):
    return {"status": "completed", "payload": worker_payload(dataset, shard)}


class MorphologyGridIdentityTests(unittest.TestCase):
    def test_published_ids_and_model_classes_are_exact(self):
        plugin = morphology_grid.PLUGIN
        self.assertEqual(
            "openstar.morphology-grid.v1",
            plugin.definition.workload_id,
        )
        self.assertEqual(
            "openstar.dataset.morphology-grid.v1",
            plugin.definition.dataset_schema_id,
        )
        self.assertEqual(
            "openstar.payload.morphology-grid-shard.v1",
            plugin.definition.payload_schema_id,
        )
        self.assertEqual(
            "openstar.result.morphology-grid-shard.v1",
            plugin.definition.result_schema_id,
        )
        self.assertEqual(
            "openstar.microlensing-residual-morphology.v1",
            morphology_grid.MORPHOLOGY_FAMILY_ID,
        )
        self.assertEqual(
            "openstar.curve-family.symmetric-radial-amplification.v1",
            morphology_grid.COMPONENT_TEMPLATE_FAMILY_ID,
        )
        self.assertEqual(
            (
                "POSITIVE_PULSE_ONLY",
                "ORDERED_NEGATIVE_POSITIVE_DOUBLET",
                "INDEPENDENT_PULSES",
            ),
            morphology_grid.MODEL_CLASS_IDS,
        )
        self.assertNotEqual(
            morphology_grid.WORKLOAD_ID,
            "openstar.curve-grid.v1",
        )
        self.assertEqual(
            "openstar.morphology-grid-execution.v1",
            morphology_grid.EXECUTION_CONTRACT_ID,
        )
        self.assertEqual(
            "1.0",
            morphology_grid.EXECUTION_CONTRACT_VERSION,
        )
        self.assertFalse(plugin.definition.allows_legacy_schemaless_workers)
        with self.assertRaises(FrozenInstanceError):
            plugin.definition.workload_id = "changed"

    def test_discovery_is_deterministic_and_includes_morphology_grid(self):
        first = discover_workloads().workload_ids
        second = discover_workloads().workload_ids
        self.assertEqual(first, second)
        self.assertEqual(tuple(sorted(first)), first)
        self.assertIn(morphology_grid.WORKLOAD_ID, first)
        self.assertIs(
            morphology_grid.PLUGIN,
            discover_workloads().require(morphology_grid.WORKLOAD_ID),
        )

    def test_strict_payloads_have_no_legacy_projection(self):
        plugin = morphology_grid.PLUGIN
        payload = next(plugin.build_work_payloads(positive_dataset()))
        self.assertEqual(
            {
                "morphologyFamilyID",
                "modelClassID",
                "gridStartIndex",
                "gridCount",
            },
            set(payload),
        )
        self.assertEqual({}, plugin.legacy_work_unit_fields(payload))
        submitted = {
            "status": "completed",
            "resultSchemaID": morphology_grid.RESULT_SCHEMA_ID,
            "payload": worker_payload(positive_dataset(), payload),
        }
        self.assertEqual(
            submitted,
            plugin.canonicalize_result({"payload": payload}, submitted),
        )


class MorphologyGridDatasetTests(unittest.TestCase):
    def test_all_three_model_datasets_are_valid(self):
        for dataset in (
            positive_dataset(),
            ordered_dataset(),
            independent_dataset(),
        ):
            with self.subTest(model=dataset["modelClassID"]):
                morphology_grid.PLUGIN.validate_dataset(dataset)

    def test_top_level_metadata_is_extensible_but_nested_structures_are_strict(self):
        extensible = positive_dataset()
        extensible["opaqueMetadata"] = {"generic": "ignored"}
        morphology_grid.PLUGIN.validate_dataset(extensible)

        cases = []
        extra_series = positive_dataset()
        extra_series["series"][0]["legacyValue"] = 1
        cases.append(extra_series)
        extra_grid = positive_dataset()
        extra_grid["morphologyGrid"]["legacyAxis"] = linear_axis()
        cases.append(extra_grid)
        extra_axis = positive_dataset()
        extra_axis["morphologyGrid"]["centerAxis"]["end"] = 2.0
        cases.append(extra_axis)
        for case in cases:
            with self.assertRaises(RuntimeError):
                morphology_grid.PLUGIN.validate_dataset(case)

    def test_series_arrays_ids_coordinates_and_weights_fail_closed(self):
        cases = []
        duplicate = positive_dataset()
        duplicate["series"][1]["genericSeriesID"] = "series-001"
        cases.append(duplicate)
        unordered = positive_dataset()
        unordered["series"].reverse()
        cases.append(unordered)
        empty_id = positive_dataset()
        empty_id["series"][0]["genericSeriesID"] = ""
        cases.append(empty_id)
        unequal = positive_dataset()
        unequal["series"][0]["values"].pop()
        cases.append(unequal)
        repeated_coordinate = positive_dataset()
        repeated_coordinate["series"][0]["coordinates"][1] = -2.0
        cases.append(repeated_coordinate)
        nonfinite = positive_dataset()
        nonfinite["series"][0]["values"][0] = math.inf
        cases.append(nonfinite)
        negative_weight = positive_dataset()
        negative_weight["series"][0]["inverseVariances"][0] = -1.0
        cases.append(negative_weight)
        for index, case in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                morphology_grid.PLUGIN.validate_dataset(case)

    def test_zero_weights_remain_but_do_not_contribute(self):
        original = positive_dataset(series_count=1)
        changed = copy.deepcopy(original)
        zero_index = changed["series"][0]["inverseVariances"].index(0.0)
        changed["series"][0]["values"][zero_index] = 1.0e100
        morphology_grid.PLUGIN.validate_dataset(changed)
        original_evaluation = morphology_grid._evaluate_candidate(original, 13)
        changed_evaluation = morphology_grid._evaluate_candidate(changed, 13)
        self.assertIsNotNone(original_evaluation)
        self.assertIsNotNone(changed_evaluation)
        self.assertEqual(
            original_evaluation.weighted_residual_sum_squares,
            changed_evaluation.weighted_residual_sum_squares,
        )
        self.assertEqual(
            original_evaluation.series_fits[0].offset,
            changed_evaluation.series_fits[0].offset,
        )

    def test_positive_weight_rank_support_is_model_specific(self):
        positive = positive_dataset(series_count=1)
        positive["series"][0]["inverseVariances"] = [1.0, 0.0, 0.0, 0.0, 1.0]
        morphology_grid.PLUGIN.validate_dataset(positive)

        ordered = ordered_dataset()
        ordered["series"][0]["inverseVariances"] = [1.0, 0.0, 0.0, 0.0, 1.0]
        with self.assertRaises(RuntimeError):
            morphology_grid.PLUGIN.validate_dataset(ordered)

    def test_independent_model_requires_exactly_one_series(self):
        dataset = independent_dataset()
        second = copy.deepcopy(dataset["series"][0])
        second["genericSeriesID"] = "series-002"
        dataset["series"].append(second)
        with self.assertRaises(RuntimeError):
            morphology_grid.PLUGIN.validate_dataset(dataset)

    def test_wrong_identities_axes_and_unsafe_counts_are_rejected(self):
        identity_fields = (
            "datasetSchemaID",
            "morphologyFamilyID",
            "componentTemplateFamilyID",
            "executionContractID",
            "executionContractVersion",
        )
        for field_name in identity_fields:
            with self.subTest(field_name=field_name):
                dataset = positive_dataset()
                dataset[field_name] = "wrong"
                with self.assertRaises(RuntimeError):
                    morphology_grid.PLUGIN.validate_dataset(dataset)

        malformed = positive_dataset()
        malformed["morphologyGrid"]["logShapeAxis"] = {
            "values": [0.0, 0.0]
        }
        with self.assertRaises(RuntimeError):
            morphology_grid.PLUGIN.validate_dataset(malformed)

        zero_separation = ordered_dataset()
        zero_separation["morphologyGrid"]["separationAxis"]["start"] = 0.0
        with self.assertRaises(RuntimeError):
            morphology_grid.PLUGIN.validate_dataset(zero_separation)

        unsafe = positive_dataset()
        unsafe["morphologyGrid"]["centerAxis"]["count"] = (
            morphology_grid.MAX_SAFE_INTEGER
        )
        with self.assertRaises(RuntimeError):
            morphology_grid.PLUGIN.validate_dataset(unsafe)


class MorphologyGridMappingTests(unittest.TestCase):
    def test_center_pair_forward_inverse_golden_vectors_are_exact(self):
        expected = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ]
        for pair_index, indices in enumerate(expected):
            with self.subTest(pair_index=pair_index):
                self.assertEqual(
                    pair_index,
                    morphology_grid.independent_center_pair_index(
                        5,
                        *indices,
                    ),
                )
                self.assertEqual(
                    indices,
                    morphology_grid.independent_center_pair_indices(
                        5,
                        pair_index,
                    ),
                )
        for invalid in ((0, 0), (1, 0), (-1, 2), (0, 5)):
            with self.assertRaises(ValueError):
                morphology_grid.independent_center_pair_index(5, *invalid)

    def test_mixed_radix_forward_inverse_is_rightmost_fastest(self):
        counts = (3, 2, 4)
        expected = [
            (first, second, third)
            for first in range(3)
            for second in range(2)
            for third in range(4)
        ]
        for index, indices in enumerate(expected):
            self.assertEqual(
                index,
                morphology_grid.candidate_index(indices, counts),
            )
            self.assertEqual(
                indices,
                morphology_grid.candidate_indices(index, counts),
            )
        self.assertEqual((0, 0, 1), expected[1])
        self.assertEqual((0, 1, 0), expected[4])
        self.assertEqual((1, 0, 0), expected[8])

    def test_model_candidate_counts_and_independent_pairs_are_exact(self):
        positive_grid = morphology_grid._grid(positive_dataset())
        ordered_grid = morphology_grid._grid(ordered_dataset())
        independent_grid = morphology_grid._grid(independent_dataset())
        self.assertEqual(27, positive_grid.total_candidates)
        self.assertEqual(2, ordered_grid.total_candidates)
        self.assertEqual(10, independent_grid.total_candidates)
        self.assertEqual(
            (0, 1),
            morphology_grid.independent_center_pair_indices(5, 0),
        )
        self.assertEqual(
            (3, 4),
            morphology_grid.independent_center_pair_indices(5, 9),
        )
        self.assertEqual(
            9,
            morphology_grid._candidate_index(independent_grid, 9, 0, 0, 0, 0),
        )

    def test_mapping_helpers_reject_unsafe_or_out_of_range_values(self):
        with self.assertRaises(ValueError):
            morphology_grid.independent_center_pair_indices(5, 10)
        with self.assertRaises(ValueError):
            morphology_grid.independent_center_pair_index(1 << 53, 0, 1)
        with self.assertRaises(ValueError):
            morphology_grid.mixed_radix_index((0,), (1 << 53,))
        with self.assertRaises(ValueError):
            morphology_grid.mixed_radix_index(
                (0, 0),
                (morphology_grid.MAX_SAFE_INTEGER, 2),
            )
        with self.assertRaises(ValueError):
            morphology_grid.mixed_radix_indices(0, (1 << 53,))

    def test_shards_are_contiguous_complete_and_have_partial_tail(self):
        payloads = list(
            morphology_grid.PLUGIN.build_work_payloads(independent_dataset())
        )
        self.assertEqual([(0, 4), (4, 4), (8, 2)], [
            (payload["gridStartIndex"], payload["gridCount"])
            for payload in payloads
        ])
        covered = [
            index
            for payload in payloads
            for index in range(
                payload["gridStartIndex"],
                payload["gridStartIndex"] + payload["gridCount"],
            )
        ]
        self.assertEqual(list(range(10)), covered)
        self.assertTrue(all(set(payload) == {
            "morphologyFamilyID",
            "modelClassID",
            "gridStartIndex",
            "gridCount",
        } for payload in payloads))


class MorphologyGridNumericalTests(unittest.TestCase):
    def test_component_basis_matches_frozen_equation(self):
        vectors = (
            (-1.25, -0.5, -0.2, -1.0),
            (0.0, 0.25, 0.0, 0.0),
            (2.5, 1.0, 0.5, -0.5),
        )
        for vector in vectors:
            self.assertEqual(
                canonical_basis(*vector),
                morphology_grid._component_basis(*vector),
            )

    def test_positive_model_recovers_per_series_offsets_and_amplitudes(self):
        dataset = positive_dataset()
        evaluation = morphology_grid._evaluate_candidate(dataset, 13)
        self.assertIsNotNone(evaluation)
        self.assertEqual(
            {"center": 0.0, "logScale": 0.0, "logShape": 0.0},
            dict(evaluation.parameters),
        )
        for index, fit in enumerate(evaluation.series_fits):
            self.assertAlmostEqual(0.5 + index, fit.offset, places=12)
            self.assertAlmostEqual(
                2.0 + index,
                fit.positive_amplitude,
                places=12,
            )
            self.assertIsNone(fit.negative_amplitude)

    def test_ordered_and_independent_signed_fits_recover_known_values(self):
        for dataset, candidate_index in (
            (ordered_dataset(), 0),
            (independent_dataset(), 5),
        ):
            with self.subTest(model=dataset["modelClassID"]):
                evaluation = morphology_grid._evaluate_candidate(
                    dataset,
                    candidate_index,
                )
                self.assertIsNotNone(evaluation)
                for index, fit in enumerate(evaluation.series_fits):
                    self.assertAlmostEqual(
                        -1.25 if index == 0 else -2.0,
                        fit.negative_amplitude,
                        places=11,
                    )
                    self.assertAlmostEqual(
                        2.5 if index == 0 else 1.5,
                        fit.positive_amplitude,
                        places=11,
                    )
                    self.assertLess(
                        fit.weighted_residual_sum_squares,
                        1.0e-20,
                    )

    def test_amplitude_constraints_choose_zero_active_sets(self):
        dataset = positive_dataset(series_count=1)
        series = dataset["series"][0]
        series["values"] = [1.5] * len(series["values"])
        evaluation = morphology_grid._evaluate_candidate(dataset, 13)
        self.assertIsNotNone(evaluation)
        self.assertEqual(0.0, evaluation.series_fits[0].positive_amplitude)
        self.assertAlmostEqual(1.5, evaluation.series_fits[0].offset, places=12)

    def test_rank_deficient_candidate_is_invalid(self):
        dataset = positive_dataset(series_count=1)
        dataset["morphologyGrid"]["logScaleAxis"] = linear_axis(
            -700.0,
            1.0,
            1,
        )
        self.assertIsNone(morphology_grid._evaluate_candidate(dataset, 0))


class MorphologyGridResultTests(unittest.TestCase):
    def setUp(self):
        self.dataset = positive_dataset()
        self.shards = list(
            morphology_grid.PLUGIN.build_work_payloads(self.dataset)
        )
        self.work_units = [
            {"id": f"work-{index}", "payload": payload}
            for index, payload in enumerate(self.shards)
        ]
        self.results = [
            result_for(self.dataset, payload)
            for payload in self.shards
        ]

    def test_valid_result_recomputes_only_reported_winner(self):
        original = morphology_grid._evaluate_candidate
        calls = []

        def recording_evaluator(dataset, grid_index):
            calls.append(grid_index)
            return original(dataset, grid_index)

        with patch.object(
            morphology_grid,
            "_evaluate_candidate",
            side_effect=recording_evaluator,
        ):
            validation = morphology_grid.PLUGIN.validate_result(
                self.work_units[0],
                self.results[0],
                self.dataset,
            )
        self.assertTrue(validation.accepted, validation.message)
        self.assertEqual(
            [self.results[0]["payload"]["bestCandidate"]["gridIndex"]],
            calls,
        )
        self.assertEqual(
            "morphology-grid-recomputation",
            validation.details["method"],
        )

    def test_result_identity_counts_parameters_and_numerics_fail_closed(self):
        mutations = (
            ("morphologyFamilyID", "wrong"),
            ("modelClassID", morphology_grid.INDEPENDENT_PULSES),
            ("gridStartIndex", 1),
            ("gridCount", 4),
            ("evaluatedCandidateCount", 4),
            ("invalidCandidateCount", 5),
        )
        for field_name, value in mutations:
            result = copy.deepcopy(self.results[0])
            result["payload"][field_name] = value
            self.assertFalse(
                morphology_grid.PLUGIN.validate_result(
                    self.work_units[0],
                    result,
                    self.dataset,
                ).accepted
            )

        wrong_parameter = copy.deepcopy(self.results[0])
        wrong_parameter["payload"]["bestCandidate"]["parameters"]["center"] += 0.5
        self.assertFalse(
            morphology_grid.PLUGIN.validate_result(
                self.work_units[0],
                wrong_parameter,
                self.dataset,
            ).accepted
        )

        wrong_sign = copy.deepcopy(self.results[0])
        wrong_sign["payload"]["bestCandidate"]["seriesFits"][0][
            "positiveAmplitude"
        ] = -1.0e-12
        self.assertFalse(
            morphology_grid.PLUGIN.validate_result(
                self.work_units[0],
                wrong_sign,
                self.dataset,
            ).accepted
        )

        outside_tolerance = copy.deepcopy(self.results[0])
        fit = outside_tolerance["payload"]["bestCandidate"]["seriesFits"][0]
        fit["offset"] += 2.0e-9 * max(1.0, abs(fit["offset"]))
        self.assertFalse(
            morphology_grid.PLUGIN.validate_result(
                self.work_units[0],
                outside_tolerance,
                self.dataset,
            ).accepted
        )

    def test_all_invalid_shard_requires_exact_server_recomputation(self):
        dataset = positive_dataset(series_count=1)
        dataset["morphologyGrid"]["logScaleAxis"] = linear_axis(
            -700.0,
            1.0,
            1,
        )
        dataset["candidatesPerWorkUnit"] = 9
        shard = next(morphology_grid.PLUGIN.build_work_payloads(dataset))
        work_unit = {"id": "all-invalid", "payload": shard}
        result = {
            "status": "completed",
            "payload": {
                "morphologyFamilyID": morphology_grid.MORPHOLOGY_FAMILY_ID,
                "modelClassID": morphology_grid.POSITIVE_PULSE_ONLY,
                "gridStartIndex": 0,
                "gridCount": 9,
                "bestCandidate": None,
                "evaluatedCandidateCount": 9,
                "invalidCandidateCount": 9,
            },
        }
        validation = morphology_grid.PLUGIN.validate_result(
            work_unit,
            result,
            dataset,
        )
        self.assertTrue(validation.accepted, validation.message)
        self.assertEqual(
            "morphology-grid-all-invalid-recomputation",
            validation.details["method"],
        )

        dishonest = copy.deepcopy(result)
        dishonest["payload"]["invalidCandidateCount"] = 8
        self.assertFalse(
            morphology_grid.PLUGIN.validate_result(
                work_unit,
                dishonest,
                dataset,
            ).accepted
        )
        nonfinite = copy.deepcopy(self.results[0])
        nonfinite["payload"]["bestCandidate"][
            "weightedResidualSumSquares"
        ] = math.nan
        self.assertFalse(
            morphology_grid.PLUGIN.validate_result(
                self.work_units[0],
                nonfinite,
                self.dataset,
            ).accepted
        )

    def test_strict_result_payload_rejects_unknown_nested_and_flattened_fields(self):
        cases = []
        outer = copy.deepcopy(self.results[0])
        outer["payload"]["legacyScore"] = 1.0
        cases.append(outer)
        candidate = copy.deepcopy(self.results[0])
        candidate["payload"]["bestCandidate"]["legacyScore"] = 1.0
        cases.append(candidate)
        parameters = copy.deepcopy(self.results[0])
        parameters["payload"]["bestCandidate"]["parameters"]["legacy"] = 1.0
        cases.append(parameters)
        fit = copy.deepcopy(self.results[0])
        fit["payload"]["bestCandidate"]["seriesFits"][0]["legacy"] = 1.0
        cases.append(fit)
        flattened = copy.deepcopy(self.results[0])
        flattened["gridStartIndex"] = 0
        cases.append(flattened)
        for case in cases:
            self.assertFalse(
                morphology_grid.PLUGIN.validate_result(
                    self.work_units[0],
                    case,
                    self.dataset,
                ).accepted
            )

    def test_partial_terminal_reduction_and_ties_are_deterministic(self):
        partial = morphology_grid.PLUGIN.reduce_dataset(
            self.dataset,
            self.work_units,
            [self.results[0], *([None] * (len(self.results) - 1))],
            terminal=False,
        )
        self.assertEqual(
            "MORPHOLOGY_GRID_INCOMPLETE",
            partial.status_fields["morphologyGridStatus"],
        )
        terminal = morphology_grid.PLUGIN.reduce_dataset(
            self.dataset,
            self.work_units,
            self.results,
            terminal=True,
        )
        self.assertEqual(
            "MORPHOLOGY_GRID_COMPLETE",
            terminal.status_fields["morphologyGridStatus"],
        )
        self.assertTrue(terminal.status_fields["coverageComplete"])
        self.assertEqual(27, terminal.status_fields["completedCandidateCount"])

        tied = copy.deepcopy(self.results)
        first = tied[0]["payload"]["bestCandidate"]
        second = tied[1]["payload"]["bestCandidate"]
        first["weightedResidualSumSquares"] = 1.0
        second["weightedResidualSumSquares"] = 1.0
        reduced = morphology_grid.PLUGIN.reduce_dataset(
            self.dataset,
            self.work_units[:2],
            tied[:2],
            terminal=False,
        )
        self.assertEqual(
            min(first["gridIndex"], second["gridIndex"]),
            reduced.status_fields["bestGridIndex"],
        )

    def test_accounting_uses_server_owned_dataset_and_payload(self):
        work_unit = {
            "payload": self.shards[-1],
            "candidateCount": 999999,
            "sampleCount": 999999,
        }
        metrics = morphology_grid.PLUGIN.contribution_metrics(
            work_unit,
            self.dataset,
        )
        self.assertEqual(morphology_grid.WORKLOAD_ID, metrics["workloadID"])
        self.assertEqual(2, metrics["seriesCount"])
        self.assertEqual(10, metrics["sampleCount"])
        self.assertEqual(2, metrics["candidateCount"])
        self.assertEqual(20, metrics["sampleCandidateEvaluations"])

        invalid_range = copy.deepcopy(work_unit)
        invalid_range["payload"]["gridStartIndex"] = 27
        with self.assertRaises(RuntimeError):
            morphology_grid.PLUGIN.contribution_metrics(
                invalid_range,
                self.dataset,
            )


if __name__ == "__main__":
    unittest.main()
