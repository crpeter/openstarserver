import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_workloads.plugins.morphology_grid import (
    COMPONENT_TEMPLATE_FAMILY_ID,
    DATASET_SCHEMA_ID,
    EXECUTION_CONTRACT_ID,
    EXECUTION_CONTRACT_VERSION,
    INDEPENDENT_PULSES,
    MAX_SAFE_INTEGER,
    MORPHOLOGY_FAMILY_ID,
    ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    PAYLOAD_SCHEMA_ID,
    POSITIVE_PULSE_ONLY,
    RESULT_SCHEMA_ID,
    WORKLOAD_ID,
)
from workflows.microlensing.build_anomaly_morphology_coarse_grid import (
    BUILD_MANIFEST_RELATIVE_PATH,
    CANDIDATES_PER_WORK_UNIT,
    CONTRACT_RELATIVE_PATH,
    DEFAULT_MAXIMUM_CANDIDATES_PER_SEARCH,
    INDEPENDENT_DATASET_RELATIVE_PATHS,
    ORDERED_DATASET_RELATIVE_PATH,
    POSITIVE_DATASET_RELATIVE_PATH,
    PROJECT_RELATIVE_PATH,
    AnomalyMorphologyCoarseGridBuildError,
    _canonical_compact_json_bytes,
    _parser,
    _stable_json_bytes,
    build_anomaly_morphology_coarse_grid,
)
from workflows.microlensing.prepare_anomaly_morphology import (
    ARTIFACT_MANIFEST_SCHEMA_ID,
    ARTIFACT_MANIFEST_VERSION,
    CONTRACT_RELATIVE_PATH as SOURCE_CONTRACT_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH as SOURCE_MANIFEST_RELATIVE_PATH,
    MORPHOLOGY_CONTRACT_ID,
    MORPHOLOGY_CONTRACT_VERSION,
    MORPHOLOGY_DATASET_SCHEMA_ID,
    MORPHOLOGY_DATASET_VERSION,
    MORPHOLOGY_PREPARATION_SCHEMA_ID,
    MORPHOLOGY_PREPARATION_VERSION,
    NEXT_TEST,
    PREPARATION_RELATIVE_PATH,
)


SOURCE_DATASET_PATHS = (
    "datasets/morphology-series-001.json",
    "datasets/morphology-series-002.json",
)
MODEL_ORDER = (
    POSITIVE_PULSE_ONLY,
    ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    INDEPENDENT_PULSES,
)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_stable_json_bytes(value))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def serialized_tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class CoarseMorphologyFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.morphology = self.root / "morphology"
        self.series_ids = ("series-001", "series-002")
        self._publish_preparation()

    def _axes(self, *, center_count=9, log_scale_count=7):
        return {
            "CENTER": {"count": center_count, "start": -2.0, "step": 0.5},
            "LOG_SCALE": {
                "count": log_scale_count,
                "start": -3.0,
                "step": 0.25,
            },
            "LOG_SHAPE": {
                "count": 3,
                "ordering": "strictly ascending explicit values",
                "values": [-4.0, -3.0, -2.0],
            },
            "SEPARATION": {"count": 8, "start": 0.25, "step": 0.25},
        }

    def _candidate_mapping(self, axes):
        center = axes["CENTER"]["count"]
        log_scale = axes["LOG_SCALE"]["count"]
        log_shape = axes["LOG_SHAPE"]["count"]
        separation = axes["SEPARATION"]["count"]
        center_pairs = center * (center - 1) // 2
        positive = center * log_scale * log_shape
        ordered = (
            center
            * separation
            * log_scale
            * log_shape
            * log_scale
            * log_shape
        )
        independent_one = (
            center_pairs * log_scale * log_shape * log_scale * log_shape
        )
        independent_total = independent_one * len(self.series_ids)
        independent_start = positive + ordered
        searches = []
        for ordinal, series_id in enumerate(self.series_ids):
            start = independent_start + ordinal * independent_one
            searches.append(
                {
                    "candidateCount": independent_one,
                    "canonicalSeriesIndex": ordinal,
                    "genericSeriesID": series_id,
                    "globalEndExclusive": start + independent_one,
                    "globalStartIndex": start,
                }
            )
        return {
            "axisOrderingByModelClass": {
                POSITIVE_PULSE_ONLY: ["CENTER", "LOG_SCALE", "LOG_SHAPE"],
                ORDERED_NEGATIVE_POSITIVE_DOUBLET: [
                    "NEGATIVE_CENTER",
                    "SEPARATION",
                    "NEGATIVE_LOG_SCALE",
                    "NEGATIVE_LOG_SHAPE",
                    "POSITIVE_LOG_SCALE",
                    "POSITIVE_LOG_SHAPE",
                ],
                INDEPENDENT_PULSES: [
                    "NEGATIVE_CENTER_PAIR_POSITIVE_CENTER",
                    "NEGATIVE_LOG_SCALE",
                    "NEGATIVE_LOG_SHAPE",
                    "POSITIVE_LOG_SCALE",
                    "POSITIVE_LOG_SHAPE",
                ],
            },
            "axisSourceByModelClass": {
                POSITIVE_PULSE_ONLY: {
                    "CENTER": "CENTER",
                    "LOG_SCALE": "LOG_SCALE",
                    "LOG_SHAPE": "LOG_SHAPE",
                },
                ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
                    "NEGATIVE_CENTER": "CENTER",
                    "NEGATIVE_LOG_SCALE": "LOG_SCALE",
                    "NEGATIVE_LOG_SHAPE": "LOG_SHAPE",
                    "POSITIVE_LOG_SCALE": "LOG_SCALE",
                    "POSITIVE_LOG_SHAPE": "LOG_SHAPE",
                    "SEPARATION": "SEPARATION",
                },
                INDEPENDENT_PULSES: {
                    "NEGATIVE_CENTER_PAIR_POSITIVE_CENTER": "CENTER",
                    "NEGATIVE_LOG_SCALE": "LOG_SCALE",
                    "NEGATIVE_LOG_SHAPE": "LOG_SHAPE",
                    "POSITIVE_LOG_SCALE": "LOG_SCALE",
                    "POSITIVE_LOG_SHAPE": "LOG_SHAPE",
                },
            },
            "candidateCounts": {
                POSITIVE_PULSE_ONLY: positive,
                ORDERED_NEGATIVE_POSITIVE_DOUBLET: ordered,
                INDEPENDENT_PULSES: independent_total,
            },
            "globalCandidateCount": positive + ordered + independent_total,
            "globalCandidateOffsets": {
                POSITIVE_PULSE_ONLY: 0,
                ORDERED_NEGATIVE_POSITIVE_DOUBLET: positive,
                INDEPENDENT_PULSES: independent_start,
            },
            "independentPerSeriesMapping": {
                "centerPairIndexFormula": (
                    "pairIndex = negativeCenterIndex * "
                    "(2 * centerCount - negativeCenterIndex - 1) // 2 + "
                    "(positiveCenterIndex - negativeCenterIndex - 1)"
                ),
                "independentSeriesSearches": searches,
                "localMixedRadixFormula": (
                    "localIndex = ((((pairIndex * logScaleCount + "
                    "negativeLogScaleIndex) * logShapeCount + "
                    "negativeLogShapeIndex) * logScaleCount + "
                    "positiveLogScaleIndex) * logShapeCount + "
                    "positiveLogShapeIndex)"
                ),
                "orderedCenterPairCount": center_pairs,
                "orderedCenterPairRule": (
                    "Enumerate negativeCenterIndex ascending, then "
                    "positiveCenterIndex ascending, retaining exactly 0 <= "
                    "negativeCenterIndex < positiveCenterIndex < centerCount."
                ),
                "perSeriesCandidateCount": independent_one,
                "totalCandidateCount": independent_total,
            },
            "linearizationRule": (
                "Shared model classes use the declared class order and "
                "rightmost-fastest mixed radix. INDEPENDENT_PULSES "
                "concatenates canonical per-series searches; it never forms a "
                "product across series."
            ),
            "maximumSafeInteger": MAX_SAFE_INTEGER,
        }

    def _execution(self):
        return {
            "amplitudeConstraints": {},
            "arithmetic": {
                "format": "IEEE-754 binary64",
                "roundingMode": "roundTiesToEven",
            },
            "candidateWinnerOrdering": [
                "finite WRSS ascending within relative tolerance",
                "finite BIC ascending within relative tolerance",
                "finite AICc ascending within relative tolerance; null after finite",
                "global candidate index ascending",
            ],
            "comparisonTolerances": {
                "constraintTolerance": 0.0,
                "objectiveRelativeTolerance": 1.0e-9,
                "rankRelativeTolerance": 1.0e-12,
                "timingComparisonTolerance": 0.0,
            },
            "componentBasis": {
                "equation": (
                    "scale=exp(logScale); shape=exp(logShape); "
                    "z=(coordinate-center)/scale; uSquared=shape*shape+z*z; "
                    "basis=(uSquared+2)/(sqrt(uSquared)*sqrt(uSquared+4))"
                )
            },
            "constrainedLinearFit": {},
            "decisionThresholdComparison": "exact",
            "designMatrices": {},
            "linearSolve": {},
            "modelWinnerOrdering": "declared order",
            "normalEquations": {},
            "objective": {},
            "weightRules": {
                "negative": "INVALID_NEGATIVE_WEIGHT",
                "positive": "contributes",
                "zero": "retained but excluded",
            },
        }

    def _contract(self, axes):
        mapping = self._candidate_mapping(axes)
        return {
            "admittedGenericSeriesIDs": list(self.series_ids),
            "axisRules": {},
            "benchmarkKind": "known-event-recovery",
            "candidateIndexMapping": mapping,
            "comparisonMetrics": {
                "parameterCounts": {
                    POSITIVE_PULSE_ONLY: {
                        "linear": 4,
                        "nonlinear": 3,
                        "total": 7,
                    },
                    ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
                        "linear": 6,
                        "nonlinear": 6,
                        "total": 12,
                    },
                    INDEPENDENT_PULSES: {
                        "linear": 6,
                        "nonlinear": 12,
                        "total": 18,
                    },
                }
            },
            "contractHashRule": "canonical compact JSON SHA-256",
            "contractID": MORPHOLOGY_CONTRACT_ID,
            "contractVersion": MORPHOLOGY_CONTRACT_VERSION,
            "crossSeriesRequirements": {},
            "deterministicExecution": self._execution(),
            "decisionRules": {},
            "effectiveWidthBounds": {},
            "familyIdentities": {
                "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
                "componentTemplateScope": (
                    "Identifies only one unit symmetric radial component, not "
                    "any compound morphology model class."
                ),
                "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
                "morphologyFamilyScope": (
                    "Identifies the three compound residual-morphology classes "
                    "and their deterministic execution contract."
                ),
            },
            "finiteValueRules": "finite only",
            "identityIsolationStatement": (
                "Only generic series identifiers and verified identity-free "
                "numerical residual evidence are admitted."
            ),
            "independentAggregation": {},
            "interpretationLimits": {},
            "invalidCandidateBehavior": {},
            "modelClassOrder": list(MODEL_ORDER),
            "modelClasses": {},
            "parameterAxes": axes,
            "preparedCoordinateBounds": {"maximum": 2.5, "minimum": -2.5},
            "separationRules": {},
        }

    def _series_document(self, series_id, ordinal, contract_sha256):
        coordinates = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        residuals = [
            float(ordinal) + value
            for value in (0.25, -0.5, 1.0, -1.5, 0.75, 0.0)
        ]
        return {
            "coordinates": coordinates,
            "genericSeriesID": series_id,
            "inclusionReasons": [["DETERMINISTIC_WINDOW"] for _ in coordinates],
            "inverseVariances": [1.0, 2.0, 0.0, 3.0, 1.5, 1.0],
            "morphologyContractID": MORPHOLOGY_CONTRACT_ID,
            "morphologyContractSHA256": contract_sha256,
            "morphologyContractVersion": MORPHOLOGY_CONTRACT_VERSION,
            "morphologyDatasetSchemaID": MORPHOLOGY_DATASET_SCHEMA_ID,
            "morphologyDatasetVersion": MORPHOLOGY_DATASET_VERSION,
            "positiveWeightSupport": {
                "confirmedPositiveWithinTwoEffectiveWidths": 1,
                "leftBaseline": 2,
                "precedingNegativeWithinTwoEffectiveWidths": 1,
                "rightBaseline": 2,
            },
            "preparedCoordinateBounds": {"maximum": 2.5, "minimum": -2.5},
            "residualValues": residuals,
            "sampleCount": len(coordinates),
            "sourceResidualSeriesSHA256": sha256_bytes(
                f"source-{series_id}".encode("ascii")
            ),
            "sourceSampleIndices": list(range(len(coordinates))),
        }

    def _publish_preparation(self, *, center_count=9, log_scale_count=7):
        if self.morphology.exists() or self.morphology.is_symlink():
            if self.morphology.is_symlink():
                self.morphology.unlink()
            else:
                shutil.rmtree(self.morphology)
        axes = self._axes(
            center_count=center_count,
            log_scale_count=log_scale_count,
        )
        contract = self._contract(axes)
        contract_sha256 = sha256_bytes(_canonical_compact_json_bytes(contract))
        write_json(self.morphology / SOURCE_CONTRACT_RELATIVE_PATH, contract)

        datasets = []
        records = []
        output_hashes = {}
        per_series = {}
        for ordinal, (series_id, relative_path) in enumerate(
            zip(self.series_ids, SOURCE_DATASET_PATHS)
        ):
            document = self._series_document(
                series_id,
                ordinal,
                contract_sha256,
            )
            path = self.morphology / relative_path
            write_json(path, document)
            file_sha256 = sha256_bytes(path.read_bytes())
            datasets.append(document)
            per_series[series_id] = document["sampleCount"]
            output_hashes[relative_path] = file_sha256
            records.append(
                {
                    "genericSeriesID": series_id,
                    "outputFile": relative_path,
                    "outputSHA256": file_sha256,
                    "sampleCount": document["sampleCount"],
                    "sourceResidualSeriesSHA256": document[
                        "sourceResidualSeriesSHA256"
                    ],
                }
            )
        parent_hashes = {
            "ancestryArtifactHashes": {
                "coarseInvestigationSHA256": sha256_bytes(
                    b"generic-coarse-investigation"
                ),
                "projectArtifacts": {
                    "coarseProjectSHA256": sha256_bytes(
                        b"generic-coarse-project"
                    ),
                    "residualPreparationSHA256": sha256_bytes(
                        b"generic-residual-preparation"
                    ),
                },
            },
            "residualGridProjectSHA256": sha256_bytes(b"generic-grid-project"),
        }
        parent_ids = {
            "ancestryProjectIDs": {
                "coarseInvestigationID": "generic-coarse-investigation",
                "projects": {
                    "coarseProjectID": "generic-coarse-project",
                    "residualPreparationID": "generic-residual-preparation",
                },
            },
            "residualGridProjectID": "generic-grid-project",
        }
        preparation = {
            "admittedGenericSeriesIDs": list(self.series_ids),
            "confirmedComponentProvenance": {
                "discoveryGenericSeriesID": self.series_ids[0],
                "validationGenericSeriesID": self.series_ids[1],
            },
            "discoveryClaim": False,
            "modelClassIDs": list(MODEL_ORDER),
            "morphologyContractID": MORPHOLOGY_CONTRACT_ID,
            "morphologyContractSHA256": contract_sha256,
            "morphologyContractVersion": MORPHOLOGY_CONTRACT_VERSION,
            "parentHashes": parent_hashes,
            "parentIDs": parent_ids,
            "planetaryInterpretationResolved": False,
            "preparedCoordinateBounds": {
                "anomalyCoreMaximum": 1.0,
                "anomalyCoreMinimum": -1.0,
                "maximum": contract["preparedCoordinateBounds"]["maximum"],
                "minimum": contract["preparedCoordinateBounds"]["minimum"],
            },
            "preparedDatasets": records,
            "recommendedNextTest": NEXT_TEST,
            "resultSchemaID": MORPHOLOGY_PREPARATION_SCHEMA_ID,
            "resultVersion": MORPHOLOGY_PREPARATION_VERSION,
            "sampleCounts": {
                "perSeries": per_series,
                "total": sum(per_series.values()),
            },
            "widthInterpretationResolved": False,
        }
        write_json(self.morphology / PREPARATION_RELATIVE_PATH, preparation)
        manifest = {
            "artifactManifestSchemaID": ARTIFACT_MANIFEST_SCHEMA_ID,
            "artifactManifestVersion": ARTIFACT_MANIFEST_VERSION,
            "identityIsolationStatement": (
                "Artifacts contain generic identifiers and identity-free "
                "numerical evidence only."
            ),
            "modelScopeStatement": "generic residual morphology only",
            "morphologyContractFileSHA256": sha256_bytes(
                (self.morphology / SOURCE_CONTRACT_RELATIVE_PATH).read_bytes()
            ),
            "morphologyContractID": MORPHOLOGY_CONTRACT_ID,
            "morphologyContractSHA256": contract_sha256,
            "morphologyPreparationFileSHA256": sha256_bytes(
                (self.morphology / PREPARATION_RELATIVE_PATH).read_bytes()
            ),
            "orderedDatasetFiles": list(SOURCE_DATASET_PATHS),
            "orderedGenericSeriesIDs": list(self.series_ids),
            "outputSHA256s": output_hashes,
            "parentHashes": parent_hashes,
            "parentIDs": parent_ids,
            "totalSampleCount": sum(per_series.values()),
        }
        write_json(self.morphology / SOURCE_MANIFEST_RELATIVE_PATH, manifest)
        self.source_contract = contract
        self.source_preparation = preparation
        self.source_manifest = manifest
        self.source_datasets = datasets

    def _refresh_preparation_manifest(self):
        preparation_path = self.morphology / PREPARATION_RELATIVE_PATH
        write_json(preparation_path, self.source_preparation)
        self.source_manifest["morphologyPreparationFileSHA256"] = sha256_bytes(
            preparation_path.read_bytes()
        )
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )

    def _refresh_contract_family(self):
        contract_path = self.morphology / SOURCE_CONTRACT_RELATIVE_PATH
        write_json(contract_path, self.source_contract)
        contract_sha256 = sha256_bytes(
            _canonical_compact_json_bytes(self.source_contract)
        )
        for ordinal, relative_path in enumerate(SOURCE_DATASET_PATHS):
            document = self.source_datasets[ordinal]
            document["morphologyContractSHA256"] = contract_sha256
            path = self.morphology / relative_path
            write_json(path, document)
            output_sha256 = sha256_bytes(path.read_bytes())
            self.source_preparation["preparedDatasets"][ordinal][
                "outputSHA256"
            ] = output_sha256
            self.source_manifest["outputSHA256s"][relative_path] = output_sha256
        self.source_preparation["morphologyContractSHA256"] = contract_sha256
        write_json(
            self.morphology / PREPARATION_RELATIVE_PATH,
            self.source_preparation,
        )
        self.source_manifest["morphologyContractSHA256"] = contract_sha256
        self.source_manifest["morphologyContractFileSHA256"] = sha256_bytes(
            contract_path.read_bytes()
        )
        self.source_manifest["morphologyPreparationFileSHA256"] = sha256_bytes(
            (self.morphology / PREPARATION_RELATIVE_PATH).read_bytes()
        )
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )

    def _refresh_dataset(self, ordinal):
        relative_path = SOURCE_DATASET_PATHS[ordinal]
        path = self.morphology / relative_path
        document = self.source_datasets[ordinal]
        write_json(path, document)
        output_sha256 = sha256_bytes(path.read_bytes())
        record = self.source_preparation["preparedDatasets"][ordinal]
        record.update(
            {
                "genericSeriesID": document["genericSeriesID"],
                "outputSHA256": output_sha256,
                "sampleCount": document["sampleCount"],
                "sourceResidualSeriesSHA256": document[
                    "sourceResidualSeriesSHA256"
                ],
            }
        )
        self.source_manifest["outputSHA256s"][relative_path] = output_sha256
        write_json(
            self.morphology / PREPARATION_RELATIVE_PATH,
            self.source_preparation,
        )
        self.source_manifest["morphologyPreparationFileSHA256"] = sha256_bytes(
            (self.morphology / PREPARATION_RELATIVE_PATH).read_bytes()
        )
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )

    def build(self, name="output", *, limit=DEFAULT_MAXIMUM_CANDIDATES_PER_SEARCH):
        return build_anomaly_morphology_coarse_grid(
            self.morphology,
            project_id="generic-morphology-coarse",
            output_root=self.root / name,
            maximum_candidates_per_search=limit,
        )

    def assert_rejected(self, pattern=None, *, name="rejected", limit=8192):
        context = (
            self.assertRaisesRegex(AnomalyMorphologyCoarseGridBuildError, pattern)
            if pattern
            else self.assertRaises(AnomalyMorphologyCoarseGridBuildError)
        )
        with context:
            self.build(name, limit=limit)


class CoarseMorphologySuccessTests(CoarseMorphologyFixture):
    def test_deterministic_four_search_project_and_default_limit(self):
        first = self.build("first")
        second = self.build("second")
        self.assertEqual(
            serialized_tree(self.root / "first"),
            serialized_tree(self.root / "second"),
        )
        self.assertEqual(8192, DEFAULT_MAXIMUM_CANDIDATES_PER_SEARCH)
        self.assertEqual(8192, first["contract"]["maximumCandidatesPerSearch"])
        project = first["project"]
        self.assertEqual(WORKLOAD_ID, project["workloadID"])
        self.assertEqual(
            [
                POSITIVE_DATASET_RELATIVE_PATH,
                ORDERED_DATASET_RELATIVE_PATH,
                *INDEPENDENT_DATASET_RELATIVE_PATHS,
            ],
            [item["path"] for item in project["datasets"]],
        )
        self.assertEqual(
            [
                POSITIVE_PULSE_ONLY,
                ORDERED_NEGATIVE_POSITIVE_DOUBLET,
                INDEPENDENT_PULSES,
                INDEPENDENT_PULSES,
            ],
            [item["modelClassID"] for item in first["datasets"]],
        )
        self.assertEqual(
            {
                BUILD_MANIFEST_RELATIVE_PATH,
                CONTRACT_RELATIVE_PATH,
                PROJECT_RELATIVE_PATH,
                POSITIVE_DATASET_RELATIVE_PATH,
                ORDERED_DATASET_RELATIVE_PATH,
                *INDEPENDENT_DATASET_RELATIVE_PATHS,
            },
            set(serialized_tree(self.root / "first")),
        )

    def test_independent_searches_are_single_series_without_cross_product(self):
        published = self.build()
        independent = published["datasets"][2:]
        self.assertEqual([[self.series_ids[0]], [self.series_ids[1]]], [
            [series["genericSeriesID"] for series in dataset["series"]]
            for dataset in independent
        ])
        records = published["buildManifest"]["datasets"][2:]
        self.assertEqual(
            [record["fullCandidateCount"] for record in records],
            [
                self.source_contract["candidateIndexMapping"]
                ["independentPerSeriesMapping"]["perSeriesCandidateCount"]
            ] * 2,
        )
        self.assertNotIn("crossSeriesCandidateCount", published["contract"])

    def test_first_admissible_stride_and_exact_axis_transformations(self):
        published = self.build(limit=500)
        records = published["buildManifest"]["datasets"]
        self.assertEqual([1, 3, 3, 3], [item["selectedStride"] for item in records])
        ordered_grid = published["datasets"][1]["morphologyGrid"]
        self.assertEqual(
            {"count": 3, "start": -2.0, "step": 1.5},
            ordered_grid["negativeCenterAxis"],
        )
        self.assertEqual(
            {"values": [-4.0]},
            ordered_grid["negativeLogShapeAxis"],
        )
        self.assertGreater(
            9 * 8 * 7 * 3 * 7 * 3,
            500,
        )
        self.assertLessEqual(records[1]["coarseCandidateCount"], 500)

    def test_positive_stride_one_and_strict_pair_candidate_count(self):
        published = self.build()
        records = published["buildManifest"]["datasets"]
        self.assertEqual(1, records[0]["selectedStride"])
        self.assertEqual(189, records[0]["coarseCandidateCount"])
        self.assertEqual(2, records[2]["selectedStride"])
        retained_center_count = 5
        expected = (retained_center_count * (retained_center_count - 1) // 2)
        expected *= 4 * 2 * 4 * 2
        self.assertEqual(expected, records[2]["coarseCandidateCount"])
        self.assertLessEqual(expected, 8192)

    def test_exact_work_unit_and_sample_evaluation_accounting(self):
        published = self.build()
        manifest = published["buildManifest"]
        total_work = 0
        total_evaluations = 0
        for record in manifest["datasets"]:
            expected_work = math.ceil(
                record["coarseCandidateCount"] / CANDIDATES_PER_WORK_UNIT
            )
            self.assertEqual(expected_work, record["expectedWorkUnitCount"])
            self.assertEqual(
                record["sampleCount"] * record["coarseCandidateCount"],
                record["expectedSampleCandidateEvaluationCount"],
            )
            total_work += expected_work
            total_evaluations += record[
                "expectedSampleCandidateEvaluationCount"
            ]
        self.assertEqual(total_work, manifest["totalExpectedWorkUnitCount"])
        self.assertEqual(
            total_evaluations,
            manifest["totalExpectedSampleCandidateEvaluationCount"],
        )

    def test_samples_provenance_hashes_and_workload_contract_are_exact(self):
        published = self.build()
        for dataset in published["datasets"]:
            self.assertEqual(DATASET_SCHEMA_ID, dataset["datasetSchemaID"])
            self.assertEqual(MORPHOLOGY_FAMILY_ID, dataset["morphologyFamilyID"])
            self.assertEqual(
                COMPONENT_TEMPLATE_FAMILY_ID,
                dataset["componentTemplateFamilyID"],
            )
            self.assertEqual(EXECUTION_CONTRACT_ID, dataset["executionContractID"])
            self.assertEqual(
                EXECUTION_CONTRACT_VERSION,
                dataset["executionContractVersion"],
            )
            self.assertEqual(PAYLOAD_SCHEMA_ID, dataset["payloadSchemaID"])
            self.assertEqual(RESULT_SCHEMA_ID, dataset["resultSchemaID"])
            for series in dataset["series"]:
                source = self.source_datasets[self.series_ids.index(
                    series["genericSeriesID"]
                )]
                self.assertEqual(source["coordinates"], series["coordinates"])
                self.assertEqual(source["residualValues"], series["values"])
                self.assertEqual(
                    source["inverseVariances"], series["inverseVariances"]
                )
        manifest = published["buildManifest"]
        self.assertEqual(
            self.source_preparation["parentIDs"], manifest["parentIDs"]
        )
        self.assertEqual(
            self.source_preparation["parentHashes"], manifest["parentHashes"]
        )
        for record in manifest["datasets"]:
            self.assertEqual(
                sha256_bytes(
                    (self.root / "output" / record["outputFile"]).read_bytes()
                ),
                record["outputSHA256"],
            )

    def test_nested_parent_lineage_is_preserved_in_sorted_structure(self):
        published = self.build()
        manifest = published["buildManifest"]
        self.assertEqual(
            self.source_preparation["parentHashes"],
            manifest["parentHashes"],
        )
        self.assertEqual(
            self.source_preparation["parentIDs"],
            manifest["parentIDs"],
        )

        def assert_sorted_mappings(value):
            if not isinstance(value, dict):
                return
            self.assertEqual(sorted(value), list(value))
            for item in value.values():
                assert_sorted_mappings(item)

        assert_sorted_mappings(manifest["parentHashes"])
        assert_sorted_mappings(manifest["parentIDs"])
        self.assertIn(
            "ancestryArtifactHashes",
            manifest["parentHashes"],
        )
        self.assertIn("ancestryProjectIDs", manifest["parentIDs"])

    def test_outputs_are_identity_free_and_contain_no_evaluation(self):
        published = self.build()
        forbidden_keys = {
            "archivefilename",
            "archiveurl",
            "catalogidentifier",
            "citation",
            "eventidentity",
            "eventname",
            "observatoryid",
            "sealedmetadata",
            "skycoordinates",
            "sourcefilename",
            "starid",
            "uid",
        }

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    normalized = "".join(
                        character for character in key.casefold()
                        if character.isalnum()
                    )
                    self.assertNotIn(normalized, forbidden_keys)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(published)
        self.assertIn(
            "no basis evaluation",
            published["contract"]["noCandidateEvaluationStatement"],
        )
        for artifact in serialized_tree(self.root / "output"):
            self.assertNotIn("result", Path(artifact).name.casefold())
        self.assertNotIn("bestCandidate", json.dumps(published))

    def test_cli_default_is_frozen(self):
        arguments = _parser().parse_args(
            [
                "--morphology-root",
                str(self.morphology),
                "--project-id",
                "generic-project",
                "--output-root",
                str(self.root / "cli-output"),
            ]
        )
        self.assertEqual(
            DEFAULT_MAXIMUM_CANDIDATES_PER_SEARCH,
            arguments.maximum_candidates_per_search,
        )


class CoarseMorphologyRejectionTests(CoarseMorphologyFixture):
    def test_invalid_nested_parent_lineage_values_are_rejected(self):
        cases = (
            (
                "nested-hash",
                "parentHashes",
                (
                    "ancestryArtifactHashes",
                    "projectArtifacts",
                    "coarseProjectSHA256",
                ),
                "not-a-sha256",
                "lowercase SHA-256",
            ),
            (
                "nested-id",
                "parentIDs",
                ("ancestryProjectIDs", "projects", "coarseProjectID"),
                17,
                "nonempty string",
            ),
            (
                "empty-mapping",
                "parentHashes",
                ("ancestryArtifactHashes", "projectArtifacts"),
                {},
                "nonempty mapping",
            ),
            (
                "list-value",
                "parentIDs",
                ("ancestryProjectIDs",),
                ["generic-project"],
                "nonempty string",
            ),
        )
        for name, root_key, path, invalid_value, pattern in cases:
            with self.subTest(name=name):
                self._publish_preparation()
                target = self.source_preparation[root_key]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = invalid_value
                self._refresh_preparation_manifest()
                self.assert_rejected(pattern, name=name)

    def test_preparation_manifest_nested_lineage_disagreement_is_rejected(self):
        manifest_hashes = json.loads(
            json.dumps(self.source_manifest["parentHashes"])
        )
        manifest_hashes["ancestryArtifactHashes"]["projectArtifacts"][
            "coarseProjectSHA256"
        ] = "f" * 64
        self.source_manifest["parentHashes"] = manifest_hashes
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )
        self.assert_rejected("manifest", name="nested-disagreement")

    def test_mutated_dataset_and_parent_hash_mismatch_are_rejected(self):
        path = self.morphology / SOURCE_DATASET_PATHS[0]
        path.write_bytes(path.read_bytes() + b" ")
        self.assert_rejected("canonical|hash")

        self._publish_preparation()
        self.source_manifest["parentHashes"]["residualGridProjectSHA256"] = "f" * 64
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )
        self.assert_rejected("manifest|parent", name="parent-hash")

    def test_incorrect_full_counts_duplicate_and_reordered_series_are_rejected(self):
        self.source_contract["candidateIndexMapping"]["candidateCounts"][
            POSITIVE_PULSE_ONLY
        ] += 1
        self._refresh_contract_family()
        self.assert_rejected("candidate counts")

        self._publish_preparation()
        self.source_datasets[1]["genericSeriesID"] = self.series_ids[0]
        self._refresh_dataset(1)
        self.assert_rejected("series ID|record", name="duplicate")

        self._publish_preparation()
        self.source_preparation["admittedGenericSeriesIDs"].reverse()
        write_json(
            self.morphology / PREPARATION_RELATIVE_PATH,
            self.source_preparation,
        )
        self.source_manifest["morphologyPreparationFileSHA256"] = sha256_bytes(
            (self.morphology / PREPARATION_RELATIVE_PATH).read_bytes()
        )
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )
        self.assert_rejected("preparation", name="reordered")

    def test_nonfinite_negative_weight_and_unsafe_arithmetic_are_rejected(self):
        path = self.morphology / SOURCE_DATASET_PATHS[0]
        path.write_text('{"value":NaN}\n', encoding="utf-8")
        self.assert_rejected("non-finite|JSON")

        self._publish_preparation()
        self.source_datasets[0]["inverseVariances"][0] = -1.0
        self._refresh_dataset(0)
        self.assert_rejected("negative weight", name="negative")

        self._publish_preparation()
        self.source_contract["parameterAxes"]["CENTER"]["count"] = (
            MAX_SAFE_INTEGER
        )
        self._refresh_contract_family()
        self.assert_rejected("safe integer", name="unsafe")

    def test_missing_malformed_or_unsupported_preparation_is_rejected(self):
        (self.morphology / SOURCE_MANIFEST_RELATIVE_PATH).unlink()
        self.assert_rejected("artifact set")

        self._publish_preparation()
        self.source_contract["familyIdentities"]["morphologyFamilyID"] = (
            "unsupported.family"
        )
        self._refresh_contract_family()
        self.assert_rejected("family", name="family")

        self._publish_preparation()
        self.source_preparation["recommendedNextTest"] = "ALTERED"
        write_json(
            self.morphology / PREPARATION_RELATIVE_PATH,
            self.source_preparation,
        )
        self.source_manifest["morphologyPreparationFileSHA256"] = sha256_bytes(
            (self.morphology / PREPARATION_RELATIVE_PATH).read_bytes()
        )
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )
        self.assert_rejected("unsupported", name="next-test")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_and_path_traversal_are_rejected(self):
        dataset = self.morphology / SOURCE_DATASET_PATHS[0]
        target = self.root / "outside.json"
        target.write_bytes(dataset.read_bytes())
        dataset.unlink()
        dataset.symlink_to(target)
        self.assert_rejected("symlink|regular")

        self._publish_preparation()
        self.source_manifest["orderedDatasetFiles"][0] = "../outside.json"
        write_json(
            self.morphology / SOURCE_MANIFEST_RELATIVE_PATH,
            self.source_manifest,
        )
        self.assert_rejected("path order|unsupported", name="traversal")

    def test_existing_output_invalid_limits_and_impossible_limit_are_rejected(self):
        (self.root / "existing").mkdir()
        self.assert_rejected("already exists", name="existing")
        for ordinal, limit in enumerate((0, -1, True, 1.5, 1_000_001)):
            self.assert_rejected(name=f"limit-{ordinal}", limit=limit)

        self._publish_preparation(center_count=2, log_scale_count=7)
        self.assert_rejected(
            "no admissible",
            name="impossible",
            limit=1,
        )

    def test_transactional_cleanup_after_publication_failure(self):
        original = __import__(
            "workflows.microlensing.build_anomaly_morphology_coarse_grid",
            fromlist=["_atomic_write_bytes"],
        )._atomic_write_bytes
        calls = 0

        def fail_during_publication(path, payload):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("synthetic publication failure")
            return original(path, payload)

        with patch(
            "workflows.microlensing.build_anomaly_morphology_coarse_grid."
            "_atomic_write_bytes",
            side_effect=fail_during_publication,
        ):
            self.assert_rejected("publication", name="atomic-failure")
        self.assertFalse((self.root / "atomic-failure").exists())
        self.assertEqual([], list(self.root.glob(".atomic-failure.*")))


if __name__ == "__main__":
    unittest.main()
