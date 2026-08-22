import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from workflows.tess.tess_catalog_guided_localization import (
    COMPONENT_IDS,
    _fit_shared_astrometric_shift,
    _temporal_predictive_validation,
    interpret_catalog_guided_localization,
    prepare_catalog_guided_localization,
)
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_offset_variability import _nuisance_catalog_sources


class CatalogGuidedLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0.0, 24.0, 240, endpoint=False)
        phase = 2 * np.pi * 0.37 * self.times
        self.basis = np.column_stack((np.sin(phase), np.cos(phase)))
        self.templates = np.asarray([
            [1.0, 0.18, 0.04], [0.6, 0.65, 0.10], [0.12, 0.72, 0.38],
            [0.04, 0.22, 0.90], [0.24, 0.08, 0.55],
        ])

    def _cube(self, vectors):
        cube = np.zeros((len(self.times), len(self.templates)))
        blocks = np.array_split(np.arange(len(self.times)), 4)
        for block, source_vectors in zip(blocks, vectors):
            for source, vector in enumerate(source_vectors):
                cube[block] += ((self.basis[block] @ np.asarray(vector))[:, None]
                                * self.templates[:, source][None, :])
        # deterministic nonzero residual variance makes coherent covariances proper.
        cube += 0.015 * np.sin(np.arange(len(cube))[:, None] * 1.731
                              + np.arange(cube.shape[1])[None, :] * 0.43)
        return cube[:, :, None]

    def _validate(self, vectors):
        return _temporal_predictive_validation(
            times=self.times, prewhitened=self._cube(vectors),
            valid=np.ones((len(self.templates), 1), dtype=bool), templates=self.templates,
            residual_frequency=0.37, time_reference=0.0, drift=0.0,
            coherent_basis=self.basis)

    def test_refit_trap_is_rejected_by_frozen_aggregate_prediction(self):
        # Candidate 1 is strong in each separately refitted block, but its vector flips.
        vectors = [[(1.0, 0.35), (0.9 * (-1) ** block, 0.6 * (-1) ** block), (0, 0)]
                   for block in range(4)]
        result = self._validate(vectors)
        self.assertEqual("TARGET_ONLY", result["predictiveModel"])
        self.assertEqual(set(result["totalHeldOutLogLikelihoodByModel"]), {
            "TARGET_ONLY", "CANDIDATE_1_ONLY", "CANDIDATE_2_ONLY",
            "TARGET_PLUS_CANDIDATE_1", "TARGET_PLUS_CANDIDATE_2",
            "CANDIDATE_1_PLUS_CANDIDATE_2", "TARGET_PLUS_BOTH"})
        for fold in result["folds"]:
            for model in fold["models"].values():
                self.assertIn("heldOutChiSquare", model)
                self.assertIn("heldOutLogLikelihood", model)

    def test_stable_candidate_survives_aggregate_prediction(self):
        vectors = [[(0, 0), (1.2, -0.45), (0, 0)] for _ in range(4)]
        result = self._validate(vectors)
        self.assertEqual("CANDIDATE_1_ONLY", result["predictiveModel"])

    def test_one_shared_astrometric_shift_applies_to_all_sources(self):
        expected = (0.2, -0.2)
        base = self.templates
        def render(dx, dy):
            penalty = (dx - expected[0]) ** 2 + (dy - expected[1]) ** 2
            return base + penalty * np.asarray([1, -1, 1, -1, 1])[:, None]
        truth = render(*expected)
        observations = truth * np.asarray([1.0, 0.5, -0.3])[None, :]
        observations = np.column_stack((observations.sum(axis=1),
                                        0.3 * observations.sum(axis=1)))
        result = _fit_shared_astrometric_shift(
            observations=observations,
            covariances=np.repeat(np.eye(2)[None, :, :] * 1e-3, len(base), axis=0),
            render_templates=render)
        calibration = result["sharedAstrometricCalibration"]
        self.assertEqual(expected, (calibration["dxPixels"], calibration["dyPixels"]))
        self.assertEqual(list(COMPONENT_IDS), calibration["appliedToComponents"])
        self.assertFalse(calibration["independentSourceMotion"])

    def test_tic_path_preserves_ambiguous_preparation_contract(self):
        catalog = {"recommendedNextTest": "CATALOG_GUIDED_SOURCE_LOCALIZATION",
                   "physicalMechanismResolved": False,
                   "plausibleCatalogCandidates": [{"id": 1}, {"id": 2}]}
        prf = {"target": {"componentID": "target"}, "residualReferenceFrequency": 0.3,
               "residualTimeReferenceDays": 1.0, "fractionalFrequencyDriftPerDay": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_catalog_guided_localization(
                catalog_summary=catalog, prf_preparation=prf,
                output_dir=Path(directory), investigation_id="TIC-277940827")
        self.assertEqual([{"id": 1}, {"id": 2}], result["catalogCandidates"])
        self.assertIsNone(result["preferredCandidate"])
        self.assertEqual([1, 2, 3, 4], result["subtractedHarmonicOrders"])
        self.assertFalse(result["physicalCycleResolved"])

    def test_interpretation_preserves_verbatim_justified_candidate(self):
        candidates = [
            {"raDeg": 10.1, "decDeg": -20.1,
             "catalogIDs": {"ticID": 111, "gaiaDR3SourceID": 222}, "frozen": "one"},
            {"raDeg": 10.2, "decDeg": -20.2,
             "catalogIDs": {"ticID": 333}, "frozen": "two"},
        ]
        sector = {
            "fullDataComparison": {"bestModel": "TARGET_PLUS_CANDIDATE_1",
                                   "bestModelIdentifiable": True},
            "temporalPredictiveValidation": {
                "predictiveModel": "TARGET_PLUS_CANDIDATE_1",
                "sourceVectorTemporalCompatibility": {"compatible": True}},
        }
        result = interpret_catalog_guided_localization(
            {"catalogCandidates": candidates}, {"sectorResults": [sector]})
        self.assertTrue(result["sourceAttributionResolved"])
        self.assertEqual(candidates[0], result["preferredCandidate"])
        self.assertEqual(candidates, result["catalogCandidates"])
        self.assertEqual("INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
                         result["recommendedNextTest"])

    def test_workflow_045_through_047_needs_no_injected_sector_inputs(self):
        catalog = {
            "recommendedNextTest": "CATALOG_GUIDED_SOURCE_LOCALIZATION",
            "physicalMechanismResolved": False,
            "plausibleCatalogCandidates": [
                {"raDeg": 10.1, "decDeg": -20.1, "catalogIDs": {"ticID": 111}},
                {"raDeg": 10.2, "decDeg": -20.2, "catalogIDs": {"ticID": 222}},
            ],
        }
        prf_prepare = {
            "ticID": 277940827, "target": {"componentID": "target"},
            "targetSky": {"raDeg": 10.0, "decDeg": -20.0}, "sectors": [1],
            "referenceFamilyPeriodDays": 13.72, "residualReferenceFrequency": 0.37,
            "residualTimeReferenceDays": 100.0, "fractionalFrequencyDriftPerDay": 0.0,
        }
        unresolved = {
            "sectorResults": [{
                "fullDataComparison": {"bestModel": "TARGET_ONLY",
                                       "bestModelIdentifiable": True},
                "temporalPredictiveValidation": {
                    "predictiveModel": "TARGET_ONLY",
                    "sourceVectorTemporalCompatibility": {"compatible": True}},
            }], "physicalCycleResolved": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create("tic-277940827", "test", "1")
            investigation = replace(investigation, stages=(
                InvestigationStage("038-family", "openstar.tess.dynamic-harmonic.analyze",
                                   "COMPLETE", None, {}, result={"physicalCycleResolved": False}),
                InvestigationStage("041-prf-prepare",
                                   "openstar.tess.official-spoc-prf-forward-modeling.prepare",
                                   "COMPLETE", None, {}, result=prf_prepare),
                InvestigationStage("043-prf-interpret",
                                   "openstar.tess.official-spoc-prf-forward-modeling.interpret",
                                   "COMPLETE", None, {}, result={"physicalMechanismResolved": False}),
                InvestigationStage("044-catalog",
                                   "openstar.tess.catalog-counterpart-identification.analyze",
                                   "COMPLETE", None, {}, result=catalog),
            ))
            store.save(investigation)
            engine = build_engine(store, SimpleNamespace(), poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            investigation, run_request = engine.run_stage(
                investigation, StageRequest(
                    "045-prepare-catalog-guided-source-localization",
                    "openstar.tess.catalog-guided-source-localization.prepare", {}),
                software_id="test", software_version="1")
            self.assertEqual({}, run_request.parameters)
            with mock.patch(
                "workflows.tess.tess_investigation.run_catalog_guided_localization",
                autospec=True, return_value=unresolved) as production_run:
                investigation, interpret_request = engine.run_stage(
                    investigation, run_request,
                    software_id="test", software_version="1")
            self.assertIsNone(production_run.call_args.kwargs["sector_inputs"])
            self.assertEqual("047-interpret-catalog-guided-source-localization",
                             interpret_request.id)
            investigation, next_stage = engine.run_stage(
                investigation, interpret_request, software_id="test", software_version="1")
            self.assertIsNone(next_stage)
            self.assertEqual("UNRESOLVED", investigation.stages[-1].result["classification"])
            self.assertEqual("QUIESCENT_AWAITING_DATA", investigation.status)

    def test_stage_048_uses_unresolved_family_bridge_without_obsolete_evidence(self):
        candidate = {"raDeg": 10.1, "decDeg": -20.1,
                     "catalogIDs": {"ticID": 111, "gaiaDR3SourceID": 222}}
        alternate = {"raDeg": 10.2, "decDeg": -20.2,
                     "catalogIDs": {"ticID": 333, "gaiaDR3SourceID": 444}}
        bridge = {
            "version": "bridge", "preparationPath": "/frozen/045.json",
            "referenceFamilyPeriodDays": 10.30084080080649,
            "subtractedHarmonicOrders": [1, 2, 3, 4],
            "physicalCycleResolved": False, "residualReferenceFrequency": 0.37,
            "residualTimeReferenceDays": 100.0,
            "fractionalFrequencyDriftPerDay": 0.002, "sectors": [1, 27],
            "priorEvidence": {"stage038": "frozen", "stage041": "frozen"},
        }
        localized = {
            "classification": "SINGLE_CATALOG_CANDIDATE_ATTRIBUTED",
            "sourceAttributionResolved": True, "preferredCandidate": candidate,
            "recommendedNextTest": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
            "physicalMechanismResolved": False,
            "catalogCandidates": [candidate, alternate],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_project = root / "source.json"
            source_project.write_text('{"id":"source","workloadID":"openstar.lomb-scargle.v1"}')
            project = root / "prepared.json"
            project.write_text("{}")
            store = InvestigationStore(root / "store")
            investigation = store.create("tic-277940827-stage-048", "test", "1")
            stages = (
                InvestigationStage("001-target", "openstar.tess.prepare-target", "COMPLETE",
                    None, {}, result={"ticID": 277940827, "sourceProjectPath": str(source_project),
                                      "sourceDatasetEntry": {"id": "tic"}, "sector": 1}),
                InvestigationStage("002-identity", "openstar.tess.catalog-identity", "COMPLETE",
                    None, {}, result={"tic": {"metadata": {"raDeg": 10.0, "decDeg": -20.0}}}),
                InvestigationStage("003-independent", "openstar.tess.independent.prepare", "COMPLETE",
                    None, {}, result={"preparedSectors": [{"sector": 1}, {"sector": 27}]}),
                InvestigationStage("038-family", "openstar.tess.dynamic-harmonic.analyze", "COMPLETE",
                    None, {}, result={"physicalCycleResolved": False}),
                InvestigationStage("040-multisource", "openstar.tess.multi-source-residual.interpret",
                    "COMPLETE", None, {}, result={"bestOffsetComponentID": "offset-1",
                        "componentSummaries": [{"componentID": "offset-1"}]}),
                InvestigationStage("041-prf", "openstar.tess.official-spoc-prf-forward-modeling.prepare",
                    "COMPLETE", None, {}, result=bridge),
                InvestigationStage("043-prf-interpret",
                    "openstar.tess.official-spoc-prf-forward-modeling.interpret", "COMPLETE",
                    None, {}, result={"physicalMechanismResolved": False}),
                InvestigationStage("044-catalog",
                    "openstar.tess.catalog-counterpart-identification.analyze", "COMPLETE",
                    None, {}, result={"preferredCandidate": None}),
                InvestigationStage("045-localize-prepare",
                    "openstar.tess.catalog-guided-source-localization.prepare", "COMPLETE",
                    None, {}, result=bridge),
                InvestigationStage("046-localize-run",
                    "openstar.tess.catalog-guided-source-localization.run", "COMPLETE",
                    None, {}, result={}),
                InvestigationStage("047-localize-interpret",
                    "openstar.tess.catalog-guided-source-localization.interpret", "COMPLETE",
                    None, {}, result=localized),
            )
            investigation = replace(investigation, stages=stages)
            store.save(investigation)
            engine = build_engine(store, SimpleNamespace(), poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            returned = {"projectPath": str(project), "preparedSeries": [],
                        "workloadID": "openstar.lomb-scargle.v1", "referencePeriodDays": 1 / 0.37,
                        "totalWorkUnits": 0}
            with mock.patch(
                "workflows.tess.tess_investigation.build_offset_source_variability_project",
                autospec=True, return_value=returned) as builder:
                completed, next_request = engine.run_stage(
                    investigation, StageRequest(
                        "048-prepare-offset-source-variability",
                        "openstar.tess.offset-source-variability.prepare", {}),
                    software_id="test", software_version="1")
            kwargs = builder.call_args.kwargs
            self.assertEqual(10.30084080080649, kwargs["reference_family_period_days"])
            self.assertEqual([1, 2, 3, 4], kwargs["harmonic_orders"])
            self.assertFalse(kwargs["physical_cycle_resolved"])
            self.assertEqual((0.37, 100.0, 0.002), (
                kwargs["residual_reference_frequency"], kwargs["residual_time_reference_days"],
                kwargs["fractional_frequency_drift_per_day"]))
            self.assertEqual([1, 27], kwargs["frozen_sectors"])
            self.assertIsNone(kwargs["nonstationary_summary"])
            self.assertIsNone(kwargs["physical_period_days"])
            self.assertEqual(localized, kwargs["offset_source_identification"])
            self.assertEqual([alternate], _nuisance_catalog_sources(
                offset_source_identification=kwargs["offset_source_identification"],
                best_candidate=kwargs["offset_source_identification"]["preferredCandidate"]))
            self.assertEqual("049-run-offset-source-variability", next_request.id)
            self.assertEqual("COMPLETE", completed.stages[-1].status)


if __name__ == "__main__":
    unittest.main()
