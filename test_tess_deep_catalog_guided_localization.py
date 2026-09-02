import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import InvestigationStage, InvestigationStore
from workflows.tess.tess_autonomy import repair_obsolete_terminal_wait
from workflows.tess.tess_deep_catalog_counterpart import HANDLER_ID as DEEP_HANDLER_ID
from workflows.tess.tess_deep_catalog_guided_localization import (
    INTERPRET_HANDLER_ID,
    METHOD_VERSION,
    PREPARE_HANDLER_ID,
    interpret_deep_catalog_guided_localization,
    prepare_deep_catalog_guided_localization,
    run_deep_catalog_guided_localization,
    validate_deep_catalog_boundary,
)
from workflows.tess.tess_investigation import deep_catalog_counterpart_continuation


class DeepCatalogGuidedLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {
                "raDeg": 100.0 + index * 0.001,
                "decDeg": -30.0 + index * 0.001,
                "separationArcsec": float(index),
                "targetSeparationArcsec": float(index + 5),
                "isTarget": False,
                "catalogIDs": {"nscDR2ObjectID": f"nsc-{index}"},
                "catalogRecords": {"NSCDR2": {"objectID": f"nsc-{index}"}},
                "motivatingComponentID": "offset-1",
                "variabilityConfirmed": False,
                "rankingEvidence": {"catalogCount": 1},
            }
            for index in range(1, 6)
        ]
        self.deep = {
            "version": "openstar.tess-deep-catalog-counterpart-identification.v1",
            "classification": "AMBIGUOUS_DEEP_CATALOG_COUNTERPARTS",
            "counterpartIdentified": False,
            "preferredCandidate": None,
            "plausibleCatalogCandidates": self.candidates,
            "variabilityConfirmed": False,
            "physicalMechanismResolved": False,
            "claimLevelChanged": False,
            "externalDataState": "AVAILABLE",
            "queryErrors": [],
            "recommendedNextTest": "HIGH_RESOLUTION_RESIDUAL_SOURCE_LOCALIZATION",
        }
        self.prf = {
            "version": "openstar.tess-prf-deblending.v1",
            "artifactRoot": "/frozen/prf-deblending",
            "modelSource": "official-public-SPOC-TESS-PRF-FITS",
            "ticID": 52244725,
            "target": {"componentID": "target"},
            "targetSky": {"raDeg": 100.0, "decDeg": -30.0},
            "sectors": [1, 2, 28, 68, 69],
            "referenceFamilyPeriodDays": 13.259005075877733,
            "residualReferenceFrequency": 1.0 / 3.259357526415564,
            "residualTimeReferenceDays": 1325.0,
            "fractionalFrequencyDriftPerDay": 0.0,
            "subtractedHarmonicOrders": [1, 2, 3, 4],
        }

    def _prepare(self, directory):
        return prepare_deep_catalog_guided_localization(
            deep_catalog_summary=self.deep, prf_preparation=self.prf,
            output_dir=Path(directory), investigation_id="tic-52244725")

    def test_exact_boundary_freezes_all_five_candidates_without_query(self):
        self.assertEqual(self.candidates, validate_deep_catalog_boundary(self.deep))
        with tempfile.TemporaryDirectory() as directory:
            preparation = self._prepare(directory)
            self.assertTrue(Path(preparation["preparationPath"]).is_file())
        self.assertEqual(METHOD_VERSION, preparation["version"])
        self.assertEqual(self.candidates, preparation["catalogCandidates"])
        self.assertEqual(
            ["target", "candidate-1", "candidate-2", "candidate-3", "candidate-4", "candidate-5"],
            preparation["componentIDs"],
        )
        self.assertEqual(63, preparation["modelHypothesisCount"])
        self.assertFalse(preparation["catalogQueriesRepeated"])

    def test_boundary_rejects_truncation_or_query_failure(self):
        with self.assertRaises(RuntimeError):
            validate_deep_catalog_boundary(dict(
                self.deep, plausibleCatalogCandidates=self.candidates + [dict(
                    self.candidates[-1], raDeg=101.0, decDeg=-29.0)]))
        with self.assertRaises(RuntimeError):
            validate_deep_catalog_boundary(dict(
                self.deep, externalDataState="BLOCKED_EXTERNAL_DATA",
                queryErrors=[{"catalog": "NSCDR2", "error": "timeout"}]))

    def test_run_passes_all_frozen_sources_and_explicit_prewhitening(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation = self._prepare(directory)
            sector_input = {
                "sector": 1, "times": np.arange(40.0),
                "prewhitened": np.ones((40, 2, 1)),
                "valid": np.ones((2, 1), dtype=bool),
                "calibrationImage": np.ones(2), "backgroundColumns": [],
                "renderTemplates": lambda _dx, _dy: np.ones((2, 6)),
                "blockCount": 4, "acquisitionProvenance": {"test": True},
            }
            with mock.patch(
                "workflows.tess.tess_deep_catalog_guided_localization."
                "analyze_generalized_catalog_guided_sector",
                autospec=True, return_value={"sector": 1},
            ) as analyze:
                result = run_deep_catalog_guided_localization(
                    preparation, sector_inputs=[sector_input])
        kwargs = analyze.call_args.kwargs
        self.assertEqual(preparation["componentIDs"], kwargs["component_ids"])
        self.assertEqual((1, 2, 3, 4), kwargs["harmonic_orders"])
        self.assertEqual(1.0 / 3.259357526415564, kwargs["candidate_frequency"])
        self.assertFalse(result["catalogQueriesRepeated"])

    @staticmethod
    def _sector(sector, source_ids):
        model = "SOURCE_SUBSET_" + "__".join(source_ids)
        return {
            "sector": sector,
            "scientificallyValid": True,
            "fullDataComparison": {
                "bestModel": model,
                "bestModelSourceIDs": source_ids,
                "bestModelIdentifiable": True,
                "completeModelFullRank": True,
            },
            "temporalPredictiveValidation": {
                "predictiveModel": model,
                "predictiveModelSourceIDs": source_ids,
                "sourceVectorTemporalCompatibility": {"compatible": True},
            },
        }

    def test_stable_single_candidate_preserves_verbatim_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation = self._prepare(directory)
        sectors = [self._sector(sector, ["target", "candidate-3"])
                   for sector in self.prf["sectors"]]
        result = interpret_deep_catalog_guided_localization(
            preparation, {"sectorResults": sectors})
        self.assertEqual("DEEP_CATALOG_RESIDUAL_SOURCE_LOCALIZED", result["classification"])
        self.assertEqual("candidate-3", result["stableComponentID"])
        self.assertEqual(self.candidates[2], result["preferredCandidate"])
        self.assertTrue(result["sourceAttributionResolved"])
        self.assertEqual("INDEPENDENT_DEEP_COUNTERPART_VARIABILITY_VALIDATION",
                         result["recommendedNextTest"])

    def test_sector_switching_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation = self._prepare(directory)
        sectors = [
            self._sector(1, ["candidate-1"]),
            self._sector(2, ["candidate-1"]),
            self._sector(28, ["candidate-2"]),
        ]
        result = interpret_deep_catalog_guided_localization(
            preparation, {"sectorResults": sectors})
        self.assertEqual("DEEP_CATALOG_RESIDUAL_SOURCE_SWITCHING_OR_BLEND",
                         result["classification"])
        self.assertIsNone(result["preferredCandidate"])
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertEqual("DEDICATED_HIGH_RESOLUTION_TIME_SERIES_IMAGING",
                         result["recommendedNextTest"])

    def test_ambiguous_continuation_routes_to_prepare(self):
        request = deep_catalog_counterpart_continuation(
            self.deep, request_id="022-deep-catalog-counterpart")
        self.assertEqual(PREPARE_HANDLER_ID, request.handler_id)
        self.assertEqual("023-prepare-deep-catalog-guided-prf-localization", request.id)
        self.assertEqual({}, request.parameters)

    def test_exact_finalized_pr182_boundary_is_reopened_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create(
                "deep-prf-repair", "openstar.workflow.tess-investigation.v1", "20.2")
            deep = InvestigationStage(
                "022-deep-catalog-counterpart", DEEP_HANDLER_ID, "COMPLETE", "020-catalog", {},
                result=self.deep,
                next_stage={
                    "id": "023-finalize", "handler_id": "openstar.tess.finalize",
                    "parameters": {"outputSuffix": "deep-catalog-counterpart"},
                    "triggered_by_stage_id": "022-deep-catalog-counterpart",
                },
            )
            conclusion = {
                "deepCatalogCounterpartIdentification": self.deep,
                "recommendedNextTest": "HIGH_RESOLUTION_RESIDUAL_SOURCE_LOCALIZATION",
            }
            final = InvestigationStage(
                "023-finalize", "openstar.tess.finalize", "COMPLETE", deep.id,
                {"outputSuffix": "deep-catalog-counterpart"}, result=conclusion, stop=True)
            metadata = dict(investigation.metadata)
            metadata["controlState"] = {
                "schedulerAction": "INVESTIGATION_COMPLETE", "selectedExperiment": None}
            investigation = replace(
                investigation, status="COMPLETE", stages=(deep, final), metadata=metadata)
            repaired = repair_obsolete_terminal_wait(store, investigation)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("RUNNING", repaired.status)
            self.assertEqual(PREPARE_HANDLER_ID, selected["handler_id"])
            self.assertEqual("024-prepare-deep-catalog-guided-prf-localization", selected["id"])
            self.assertEqual(deep.id, selected["triggered_by_stage_id"])
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))


if __name__ == "__main__":
    unittest.main()
