import copy
import math
import unittest
from dataclasses import FrozenInstanceError

from openstar_workloads.discovery import discover_workloads
from openstar_workloads.plugins import curve_grid


GOLDEN_VECTOR = {
    "coordinates": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "values": [
        2.5869967789998038,
        2.8094010767585034,
        3.1832815729997477,
        2.8094010767585034,
        2.5869967789998038,
    ],
    "inverseVariances": [1.0, 1.0, 1.0, 1.0, 1.0],
    "centerAxis": {
        "start": -0.5,
        "step": 0.5,
        "count": 3,
    },
    "logScaleAxis": {
        "start": -0.6931471805599453,
        "step": 0.6931471805599453,
        "count": 3,
    },
    "logShapeAxis": {
        "start": -0.6931471805599453,
        "step": 0.6931471805599453,
        "count": 3,
    },
    "candidatesPerWorkUnit": 5,
}


def golden_dataset():
    return {
        "id": "golden-curve-grid",
        "datasetSchemaID": curve_grid.DATASET_SCHEMA_ID,
        "coordinates": list(GOLDEN_VECTOR["coordinates"]),
        "values": list(GOLDEN_VECTOR["values"]),
        "inverseVariances": list(GOLDEN_VECTOR["inverseVariances"]),
        "curveGrid": {
            "familyID": curve_grid.FAMILY_ID,
            "centerAxis": dict(GOLDEN_VECTOR["centerAxis"]),
            "logScaleAxis": dict(GOLDEN_VECTOR["logScaleAxis"]),
            "logShapeAxis": dict(GOLDEN_VECTOR["logShapeAxis"]),
            "candidatesPerWorkUnit": GOLDEN_VECTOR[
                "candidatesPerWorkUnit"
            ],
        },
    }


def worker_payload(dataset, shard):
    evaluations = []
    invalid_count = 0
    start = shard["gridStartIndex"]
    for grid_index in range(start, start + shard["gridCount"]):
        evaluation = curve_grid._evaluate_candidate(dataset, grid_index)
        if evaluation is None:
            invalid_count += 1
        else:
            evaluations.append(evaluation)
    if not evaluations:
        raise AssertionError("test shard unexpectedly has no valid candidate")
    best = min(
        evaluations,
        key=lambda item: (
            item.weighted_residual_sum_squares,
            item.grid_index,
        ),
    )
    return {
        "familyID": curve_grid.FAMILY_ID,
        "gridStartIndex": start,
        "gridCount": shard["gridCount"],
        "bestGridIndex": best.grid_index,
        "bestCenter": best.center,
        "bestLogScale": best.log_scale,
        "bestLogShape": best.log_shape,
        "bestOffset": best.offset,
        "bestAmplitude": best.amplitude,
        "bestWeightedResidualSumSquares": (
            best.weighted_residual_sum_squares
        ),
        "evaluatedCandidateCount": shard["gridCount"],
        "invalidCandidateCount": invalid_count,
    }


def result_for(dataset, shard):
    return {
        "status": "completed",
        "payload": worker_payload(dataset, shard),
    }


class CurveGridIdentityAndDiscoveryTests(unittest.TestCase):
    def test_published_ids_are_exact_and_definition_is_immutable(self):
        plugin = curve_grid.PLUGIN
        self.assertEqual("openstar.curve-grid.v1", plugin.definition.workload_id)
        self.assertEqual(
            "openstar.dataset.curve-grid.v1",
            plugin.definition.dataset_schema_id,
        )
        self.assertEqual(
            "openstar.payload.curve-grid-shard.v1",
            plugin.definition.payload_schema_id,
        )
        self.assertEqual(
            "openstar.result.curve-grid-shard.v1",
            plugin.definition.result_schema_id,
        )
        self.assertEqual(
            "openstar.curve-family.symmetric-radial-amplification.v1",
            curve_grid.FAMILY_ID,
        )
        self.assertFalse(plugin.definition.allows_legacy_schemaless_workers)
        self.assertFalse(plugin.uses_legacy_coordinator_diagnostics)
        self.assertFalse(plugin.uses_legacy_science_metadata_validation)
        with self.assertRaises(FrozenInstanceError):
            plugin.definition.workload_id = "changed"

    def test_discovery_is_deterministic_and_includes_curve_grid(self):
        first = discover_workloads().workload_ids
        second = discover_workloads().workload_ids
        self.assertEqual(first, second)
        self.assertEqual(tuple(sorted(first)), first)
        self.assertIn(curve_grid.WORKLOAD_ID, first)
        self.assertIs(
            discover_workloads().require(curve_grid.WORKLOAD_ID),
            curve_grid.PLUGIN,
        )

    def test_strict_payloads_have_no_legacy_projection(self):
        plugin = curve_grid.PLUGIN
        payload = next(iter(plugin.build_work_payloads(golden_dataset())))
        self.assertEqual(
            {"familyID", "gridStartIndex", "gridCount"},
            set(payload),
        )
        self.assertEqual({}, plugin.legacy_work_unit_fields(payload))

        submitted = {
            "status": "completed",
            "resultSchemaID": curve_grid.RESULT_SCHEMA_ID,
            "workUnitID": "work",
            "nodeID": "node",
            "payload": worker_payload(golden_dataset(), payload),
        }
        self.assertEqual(
            submitted,
            plugin.canonicalize_result({"payload": payload}, submitted),
        )


class CurveGridDatasetValidationTests(unittest.TestCase):
    def test_golden_dataset_is_valid(self):
        curve_grid.PLUGIN.validate_dataset(golden_dataset())

    def test_missing_or_legacy_schema_identity_is_rejected(self):
        for supplied in (None, "openstar.dataset.curve-grid"):
            with self.subTest(supplied=supplied):
                dataset = golden_dataset()
                if supplied is None:
                    del dataset["datasetSchemaID"]
                else:
                    dataset["datasetSchemaID"] = supplied
                with self.assertRaises(RuntimeError):
                    curve_grid.PLUGIN.validate_dataset(dataset)

    def test_malformed_arrays_are_rejected(self):
        cases = []

        too_short = golden_dataset()
        for key in ("coordinates", "values", "inverseVariances"):
            too_short[key] = too_short[key][:2]
        cases.append(too_short)

        unequal = golden_dataset()
        unequal["values"] = unequal["values"][:-1]
        cases.append(unequal)

        non_array = golden_dataset()
        non_array["coordinates"] = tuple(non_array["coordinates"])
        cases.append(non_array)

        boolean_sample = golden_dataset()
        boolean_sample["values"][0] = True
        cases.append(boolean_sample)

        nonfinite = golden_dataset()
        nonfinite["coordinates"][0] = math.inf
        cases.append(nonfinite)

        zero_weight = golden_dataset()
        zero_weight["inverseVariances"][0] = 0.0
        cases.append(zero_weight)

        negative_weight = golden_dataset()
        negative_weight["inverseVariances"][0] = -1.0
        cases.append(negative_weight)

        for index, dataset in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                curve_grid.PLUGIN.validate_dataset(dataset)

    def test_malformed_axes_and_counts_are_rejected(self):
        mutations = (
            ("centerAxis", "start", math.nan),
            ("centerAxis", "step", True),
            ("centerAxis", "count", True),
            ("centerAxis", "count", 1.0),
            ("centerAxis", "count", 0),
            ("centerAxis", "step", 0.0),
            ("logScaleAxis", "start", 1000.0),
            ("logShapeAxis", "start", -1000.0),
        )
        for axis, key, value in mutations:
            with self.subTest(axis=axis, key=key, value=value):
                dataset = golden_dataset()
                dataset["curveGrid"][axis][key] = value
                with self.assertRaises(RuntimeError):
                    curve_grid.PLUGIN.validate_dataset(dataset)

        zero_step_singleton = golden_dataset()
        zero_step_singleton["curveGrid"]["centerAxis"] = {
            "start": 0.0,
            "step": 0.0,
            "count": 1,
        }
        curve_grid.PLUGIN.validate_dataset(zero_step_singleton)

    def test_unknown_grid_fields_and_unsafe_products_are_rejected(self):
        unexpected = golden_dataset()
        unexpected["curveGrid"]["legacyCount"] = 27
        with self.assertRaises(RuntimeError):
            curve_grid.PLUGIN.validate_dataset(unexpected)

        unexpected_axis = golden_dataset()
        unexpected_axis["curveGrid"]["centerAxis"]["end"] = 0.5
        with self.assertRaises(RuntimeError):
            curve_grid.PLUGIN.validate_dataset(unexpected_axis)

        unsafe = golden_dataset()
        unsafe["curveGrid"]["centerAxis"]["count"] = (
            curve_grid.MAX_SAFE_INTEGER
        )
        unsafe["curveGrid"]["logScaleAxis"] = {
            "start": 0.0,
            "step": 0.0,
            "count": 1,
        }
        unsafe["curveGrid"]["logShapeAxis"] = {
            "start": 0.0,
            "step": 0.0,
            "count": 1,
        }
        with self.assertRaises(RuntimeError):
            curve_grid.PLUGIN.validate_dataset(unsafe)

        unsafe_count = golden_dataset()
        unsafe_count["curveGrid"]["candidatesPerWorkUnit"] = (
            curve_grid.MAX_SAFE_INTEGER + 1
        )
        with self.assertRaises(RuntimeError):
            curve_grid.PLUGIN.validate_dataset(unsafe_count)


class CurveGridOrderingAndNumericalTests(unittest.TestCase):
    def test_flattening_and_inverse_mapping_are_exact(self):
        grid = curve_grid._grid(golden_dataset())
        expected = []
        for center_index in range(3):
            for scale_index in range(3):
                for shape_index in range(3):
                    expected.append(
                        (center_index, scale_index, shape_index)
                    )

        self.assertEqual(27, grid.total_candidates)
        for grid_index, indices in enumerate(expected):
            with self.subTest(grid_index=grid_index):
                self.assertEqual(
                    indices,
                    curve_grid._grid_indices(grid, grid_index),
                )
                self.assertEqual(
                    grid_index,
                    curve_grid._grid_index(grid, *indices),
                )
        self.assertEqual((0, 0, 0), expected[0])
        self.assertEqual((0, 0, 1), expected[1])
        self.assertEqual((0, 1, 0), expected[3])
        self.assertEqual((1, 0, 0), expected[9])

    def test_shards_are_contiguous_complete_and_include_partial_tail(self):
        payloads = list(
            curve_grid.PLUGIN.build_work_payloads(golden_dataset())
        )
        self.assertEqual(6, len(payloads))
        self.assertEqual(
            [(0, 5), (5, 5), (10, 5), (15, 5), (20, 5), (25, 2)],
            [
                (payload["gridStartIndex"], payload["gridCount"])
                for payload in payloads
            ],
        )
        covered = [
            grid_index
            for payload in payloads
            for grid_index in range(
                payload["gridStartIndex"],
                payload["gridStartIndex"] + payload["gridCount"],
            )
        ]
        self.assertEqual(list(range(27)), covered)
        self.assertTrue(
            all(payload["familyID"] == curve_grid.FAMILY_ID for payload in payloads)
        )

    def test_golden_vector_winner_and_fit(self):
        dataset = golden_dataset()
        evaluations = [
            curve_grid._evaluate_candidate(dataset, grid_index)
            for grid_index in range(27)
        ]
        valid = [item for item in evaluations if item is not None]
        best = min(
            valid,
            key=lambda item: (
                item.weighted_residual_sum_squares,
                item.grid_index,
            ),
        )
        self.assertEqual(13, best.grid_index)
        self.assertEqual(0.0, best.center)
        self.assertEqual(0.0, best.log_scale)
        self.assertEqual(0.0, best.log_shape)
        self.assertAlmostEqual(0.5, best.offset, places=12)
        self.assertAlmostEqual(2.0, best.amplitude, places=12)
        self.assertLess(best.weighted_residual_sum_squares, 1.0e-20)


class CurveGridResultAndReductionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = golden_dataset()
        self.shards = list(
            curve_grid.PLUGIN.build_work_payloads(self.dataset)
        )
        self.work_units = [
            {"id": f"work-{index}", "payload": shard}
            for index, shard in enumerate(self.shards)
        ]
        self.results = [
            result_for(self.dataset, shard) for shard in self.shards
        ]

    def test_valid_result_is_recomputed_and_accepted(self):
        validation = curve_grid.PLUGIN.validate_result(
            self.work_units[0],
            self.results[0],
            self.dataset,
        )
        self.assertTrue(validation.accepted, validation.message)
        self.assertEqual(
            "curve-grid-recomputation",
            validation.details["method"],
        )

    def test_result_identity_boundaries_counts_and_parameters_fail_closed(self):
        base = self.results[0]
        mutations = (
            ("familyID", "wrong-family"),
            ("gridStartIndex", 1),
            ("gridCount", 4),
            ("bestGridIndex", -1),
            ("bestGridIndex", 5),
            ("evaluatedCandidateCount", 4),
            ("evaluatedCandidateCount", True),
            ("invalidCandidateCount", 5),
            ("invalidCandidateCount", -1),
            ("bestCenter", 0.25),
            ("bestLogScale", 0.25),
            ("bestLogShape", 0.25),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name, value=value):
                result = copy.deepcopy(base)
                result["payload"][field_name] = value
                validation = curve_grid.PLUGIN.validate_result(
                    self.work_units[0],
                    result,
                    self.dataset,
                )
                self.assertFalse(validation.accepted)

        wrong_work = copy.deepcopy(self.work_units[0])
        wrong_work["payload"]["familyID"] = "wrong-family"
        self.assertFalse(
            curve_grid.PLUGIN.validate_result(
                wrong_work,
                base,
                self.dataset,
            ).accepted
        )

    def test_result_numerical_validation_uses_published_tolerance(self):
        for field_name in (
            "bestOffset",
            "bestAmplitude",
            "bestWeightedResidualSumSquares",
        ):
            with self.subTest(field_name=field_name):
                result = copy.deepcopy(self.results[0])
                expected = result["payload"][field_name]
                result["payload"][field_name] = expected + 2.0e-9 * max(
                    1.0,
                    abs(expected),
                )
                self.assertFalse(
                    curve_grid.PLUGIN.validate_result(
                        self.work_units[0],
                        result,
                        self.dataset,
                    ).accepted
                )

        nonfinite = copy.deepcopy(self.results[0])
        nonfinite["payload"]["bestOffset"] = math.nan
        self.assertFalse(
            curve_grid.PLUGIN.validate_result(
                self.work_units[0],
                nonfinite,
                self.dataset,
            ).accepted
        )

        negative = copy.deepcopy(self.results[0])
        negative["payload"]["bestWeightedResidualSumSquares"] = -1.0
        self.assertFalse(
            curve_grid.PLUGIN.validate_result(
                self.work_units[0],
                negative,
                self.dataset,
            ).accepted
        )

    def test_result_payload_is_exact_and_rejects_flattened_fields(self):
        extra = copy.deepcopy(self.results[0])
        extra["payload"]["legacyScore"] = 1.0
        self.assertFalse(
            curve_grid.PLUGIN.validate_result(
                self.work_units[0],
                extra,
                self.dataset,
            ).accepted
        )

        flattened = copy.deepcopy(self.results[0])
        flattened["bestGridIndex"] = flattened["payload"]["bestGridIndex"]
        self.assertFalse(
            curve_grid.PLUGIN.validate_result(
                self.work_units[0],
                flattened,
                self.dataset,
            ).accepted
        )

        missing = copy.deepcopy(self.results[0])
        del missing["payload"]["bestAmplitude"]
        self.assertFalse(
            curve_grid.PLUGIN.validate_result(
                self.work_units[0],
                missing,
                self.dataset,
            ).accepted
        )

    def test_partial_and_terminal_reduction(self):
        partial_results = [self.results[0], *([None] * 5)]
        partial = curve_grid.PLUGIN.reduce_dataset(
            self.dataset,
            self.work_units,
            partial_results,
            terminal=False,
        )
        self.assertEqual(
            "CURVE_GRID_INCOMPLETE",
            partial.status_fields["curveGridStatus"],
        )
        self.assertFalse(partial.status_fields["coverageComplete"])
        self.assertEqual(5, partial.status_fields["completedCandidateCount"])
        self.assertEqual(27, partial.status_fields["totalCandidateCount"])

        terminal = curve_grid.PLUGIN.reduce_dataset(
            self.dataset,
            self.work_units,
            self.results,
            terminal=True,
        )
        self.assertEqual(
            "CURVE_GRID_COMPLETE",
            terminal.status_fields["curveGridStatus"],
        )
        self.assertTrue(terminal.status_fields["coverageComplete"])
        self.assertEqual(27, terminal.status_fields["completedCandidateCount"])
        self.assertEqual(13, terminal.status_fields["bestGridIndex"])
        self.assertAlmostEqual(0.5, terminal.status_fields["bestOffset"], places=12)
        self.assertAlmostEqual(
            2.0,
            terminal.status_fields["bestAmplitude"],
            places=12,
        )

        missing_terminal = curve_grid.PLUGIN.reduce_dataset(
            self.dataset,
            self.work_units,
            partial_results,
            terminal=True,
        )
        self.assertEqual(
            "CURVE_GRID_INCOMPLETE",
            missing_terminal.status_fields["curveGridStatus"],
        )

    def test_reduction_ties_select_smaller_global_grid_index(self):
        first = copy.deepcopy(self.results[0])
        second = copy.deepcopy(self.results[1])
        first["payload"]["bestWeightedResidualSumSquares"] = 1.0
        second["payload"]["bestWeightedResidualSumSquares"] = 1.0
        first["payload"]["bestGridIndex"] = 4
        second["payload"]["bestGridIndex"] = 5

        reduced = curve_grid.PLUGIN.reduce_dataset(
            self.dataset,
            self.work_units,
            [first, second, *([None] * 4)],
            terminal=False,
        )
        self.assertEqual(4, reduced.status_fields["bestGridIndex"])

        reversed_reduced = curve_grid.PLUGIN.reduce_dataset(
            self.dataset,
            list(reversed(self.work_units)),
            [*reversed([first, second, *([None] * 4)])],
            terminal=False,
        )
        self.assertEqual(4, reversed_reduced.status_fields["bestGridIndex"])

    def test_accounting_uses_only_server_owned_dataset_and_work_payload(self):
        work_unit = {
            "payload": self.shards[-1],
            "candidateCount": 999999,
            "sampleCount": 999999,
        }
        self.assertEqual(
            {
                "workloadID": curve_grid.WORKLOAD_ID,
                "familyID": curve_grid.FAMILY_ID,
                "sampleCount": 5,
                "candidateCount": 2,
                "sampleCandidateEvaluations": 10,
            },
            dict(
                curve_grid.PLUGIN.contribution_metrics(
                    work_unit,
                    self.dataset,
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
