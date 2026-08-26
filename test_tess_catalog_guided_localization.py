import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from workflows.tess.tess_catalog_guided_localization import (
    COMPONENT_IDS,
    _compare_hypotheses,
    _fit_shared_astrometric_shift,
    _prewhiten_production_cube,
    _temporal_predictive_validation,
    interpret_catalog_guided_localization,
    prepare_catalog_guided_localization,
    analyze_generalized_catalog_guided_sector,
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
        coefficients = self.templates[:, 1, None] * np.asarray([[1.2, -0.45]])
        comparison = _compare_hypotheses(
            coefficients, np.repeat(np.eye(2)[None] * 1e-4, len(self.templates), axis=0),
            self.templates)
        self.assertEqual("CANDIDATE_1_ONLY", comparison["bestModel"])
        self.assertTrue(comparison["bestModelIdentifiable"])

    def test_overlapping_candidates_fail_conditional_identifiability(self):
        templates = self.templates.copy()
        templates[:, 2] = templates[:, 1]
        coefficients = templates[:, 1, None] * np.asarray([[1.2, -0.45]])
        comparison = _compare_hypotheses(
            coefficients, np.repeat(np.eye(2)[None] * 1e-4, len(templates), axis=0),
            templates)
        self.assertEqual("CANDIDATE_1_ONLY", comparison["bestModel"])
        self.assertFalse(comparison["bestModelIdentifiable"])

    def test_one_shared_astrometric_shift_applies_to_all_sources(self):
        expected = (0.2, -0.2)
        yy, xx = np.mgrid[0:4, 0:4]
        centers = [(0.8, 0.9), (2.0, 1.5), (2.8, 2.7)]
        def render(dx, dy):
            columns = []
            for cx, cy in centers:
                value = np.exp(-0.5 * (((xx - cx - dx) / 0.55) ** 2
                                       + ((yy - cy - dy) / 0.55) ** 2)).reshape(-1)
                columns.append(value / value.sum())
            return np.column_stack(columns)
        background = [np.ones(16), np.tile(np.linspace(-1, 1, 4), 4),
                      np.repeat(np.linspace(-1, 1, 4), 4)]
        calibration_image = np.column_stack(
            [render(*expected), *background]) @ np.asarray([8, 5, 3, 2, .2, -.1])
        result = _fit_shared_astrometric_shift(
            calibration_image=calibration_image, background_columns=background,
            render_templates=render)
        self.assertTrue(result["available"])
        calibration = result["sharedAstrometricCalibration"]
        self.assertEqual(expected, (calibration["dxPixels"], calibration["dyPixels"]))
        self.assertEqual(list(COMPONENT_IDS), calibration["appliedToComponents"])
        self.assertFalse(calibration["independentSourceMotion"])

    def test_flat_astrometric_objective_is_rejected(self):
        with mock.patch(
            "workflows.tess.tess_catalog_guided_localization._fit_static_image",
            return_value=(1.0, np.ones(6), 0.9)):
            result = _fit_shared_astrometric_shift(
                calibration_image=np.ones(16), background_columns=[np.ones(16)] * 3,
                render_templates=lambda _dx, _dy: np.ones((16, 3)))
        self.assertFalse(result["available"])
        self.assertIn("unique minimum", result["reason"])

    def test_production_prewhitening_preserves_persisted_order_and_sequence(self):
        times=np.arange(8,dtype=float); cube=np.ones((8,2,2)); expected=(4,2)
        returned=(np.zeros_like(cube),np.ones((2,2),bool))
        with mock.patch(
            "workflows.tess.tess_catalog_guided_localization._prewhiten_cube_raw",
            return_value=returned) as prewhiten:
            actual=_prewhiten_production_cube(
                times=times,corrected=cube,reference_family_period_days=10.0,
                harmonic_orders=expected)
        self.assertIs(returned,actual)
        self.assertEqual(expected,prewhiten.call_args.kwargs["harmonic_orders"])

    def test_inadequate_astrometric_explained_variance_is_rejected(self):
        calls = iter(range(9))
        def inadequate(*_args):
            index = next(calls)
            return (float(index), np.ones(6), 0.1)
        with mock.patch(
            "workflows.tess.tess_catalog_guided_localization._fit_static_image",
            side_effect=inadequate):
            result = _fit_shared_astrometric_shift(
                calibration_image=np.ones(16), background_columns=[np.ones(16)] * 3,
                render_templates=lambda _dx, _dy: np.ones((16, 3)))
        self.assertFalse(result["available"])
        self.assertIn("inadequate", result["reason"])

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
            self.assertEqual("openstar.tess.residual-phase-difference-imaging.prepare",
                             next_stage.handler_id)
            self.assertEqual("UNRESOLVED", investigation.stages[-1].result["classification"])
            self.assertEqual("RUNNING", investigation.status)

    def test_generalized_fitter_explicit_orders_and_legacy_default(self):
        common={"sector":1,"times":np.arange(20.),"prewhitened":np.ones((20,2,1)),
            "valid":np.ones((2,1),bool),"calibration_image":np.ones(2),
            "background_columns":[],"render_templates":lambda _x,_y:np.ones((2,3)),
            "candidate_frequency":.45,"original_time_origin":2500.,
            "physical_frequency":1/10.30084080080649,"component_ids":COMPONENT_IDS}
        with mock.patch("workflows.tess.tess_catalog_guided_localization._prewhitened_coherent_basis",
                return_value=np.ones((20,4))) as basis, mock.patch(
                "workflows.tess.tess_catalog_guided_localization._coherent_pixel_fit",
                return_value={}), mock.patch(
                "workflows.tess.tess_catalog_guided_localization._fit_shared_astrometric_shift",
                return_value={"available":False,"reason":"test"}):
            analyze_generalized_catalog_guided_sector(**common,harmonic_orders=(1,2,3,4))
            self.assertEqual((1,2,3,4),basis.call_args.kwargs["harmonic_orders"])
            analyze_generalized_catalog_guided_sector(**common)
            self.assertEqual((1,2),basis.call_args.kwargs["harmonic_orders"])

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
