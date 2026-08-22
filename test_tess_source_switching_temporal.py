import tempfile
import unittest
from pathlib import Path

import numpy as np
from unittest import mock

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import build_engine

from workflows.tess.tess_source_switching_temporal import (
    interpret_source_switching_temporal_model,
    prepare_source_switching_temporal_model,
    run_source_switching_temporal_model,
)


class SourceSwitchingTemporalTests(unittest.TestCase):
    def setUp(self):
        self.preparation = {
            "sectors": [94, 95, 102, 103], "referenceFamilyPeriodDays": 10.30084080080649,
            "subtractedHarmonicOrders": [1, 2, 3, 4], "physicalCycleResolved": False,
            "residualReferenceFrequency": 1 / 2.207, "residualTimeReferenceDays": 2500.,
            "fractionalFrequencyDriftPerDay": 0.,
        }

    def _inputs(self, vectors, templates=None, noise=.03):
        if templates is None:
            templates = np.eye(3).reshape(3, 1, 3)
        result=[]
        rng=np.random.default_rng(277940827)
        for sector, (sinv, cosv) in zip(self.preparation["sectors"], vectors):
            times=np.linspace(2500, 2527, 240)
            angle=2*np.pi*self.preparation["residualReferenceFrequency"]*(times-2500)
            flat=np.asarray(templates).reshape(3, -1)
            values=np.sin(angle)[:,None]@(np.asarray(sinv)[None,:]@flat) + np.cos(angle)[:,None]@(np.asarray(cosv)[None,:]@flat)
            cube=values.reshape(len(times), *np.asarray(templates).shape[1:])
            cube += rng.normal(0, noise, cube.shape)
            result.append({"sector": sector, "times": times, "prewhitened": cube,
                           "valid": np.ones(cube.shape[1:], bool), "sourceTemplates": templates})
        return result

    def _classify(self, vectors, **kwargs):
        run=run_source_switching_temporal_model(self.preparation, sector_inputs=self._inputs(vectors, **kwargs))
        return interpret_source_switching_temporal_model(self.preparation, run), run

    def test_stationary_target_wins(self):
        result, _ = self._classify([([2,0,0],[.4,0,0])]*4)
        self.assertEqual("STATIONARY_TARGET_SOURCE", result["classification"])

    def test_stationary_candidate_one_wins(self):
        candidate1={"raDeg":1.1,"decDeg":2.1,"catalogIDs":{"ticID":1}}
        candidate2={"raDeg":1.2,"decDeg":2.2,"catalogIDs":{"ticID":2}}
        self.preparation["catalogCandidates"]=[candidate1,candidate2]
        result, _ = self._classify([([0,2,0],[0,-.3,0])]*4)
        self.assertEqual("STATIONARY_CANDIDATE_1_SOURCE", result["classification"])
        self.assertIs(candidate1, result["preferredCandidate"])
        self.assertEqual([candidate1,candidate2], result["catalogCandidates"])
        self.assertEqual("INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
                         result["recommendedNextTest"])

    def test_stationary_candidate_two_preserves_direct_validation(self):
        candidates=[{"catalogIDs":{"ticID":1}}, {"catalogIDs":{"ticID":2}}]
        self.preparation["catalogCandidates"]=candidates
        result, _ = self._classify([([0,0,2],[0,0,-.3])]*4)
        self.assertEqual("STATIONARY_CANDIDATE_2_SOURCE", result["classification"])
        self.assertIs(candidates[1], result["preferredCandidate"])
        self.assertEqual("INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
                         result["recommendedNextTest"])

    def test_switching_wins_bic_and_held_out_prediction(self):
        vectors=[([2,0,0],[.2,0,0]), ([2,0,0],[.2,0,0]),
                 ([0,2,0],[0,-.2,0]), ([0,2,0],[0,-.2,0])]
        result, run = self._classify(vectors)
        self.assertEqual("SOURCE_SWITCHING_CONFIRMED", result["classification"])
        self.assertEqual("SECTOR_VARYING_SOURCE_AMPLITUDES", result["bicWinningModel"])
        self.assertEqual("SECTOR_VARYING_SOURCE_AMPLITUDES", result["heldOutWinningModel"])
        self.assertEqual(4, len(run["perSectorSourceCoherentVectors"]))
        folds=run["heldOutTemporalValidation"]["folds"]
        self.assertIn("contiguous temporal blocks",
                      run["heldOutTemporalValidation"]["method"])
        self.assertEqual(4 * len(run["models"]), len(folds))
        self.assertTrue(all(fold["sectorEvidence"][0]["heldOutTimeRange"][0]
                            <= fold["sectorEvidence"][0]["heldOutTimeRange"][1]
                            for fold in folds))

    def test_stable_two_source_blend_is_not_switching(self):
        result, _ = self._classify([([1.5,1.1,0],[.3,-.2,0])]*4)
        self.assertEqual("MULTI_SOURCE_STATIONARY_BLEND", result["classification"])

    def test_stable_target_candidate_two_blend(self):
        result, _ = self._classify([([1.5,0,1.1],[.3,0,-.2])]*4)
        self.assertEqual("MULTI_SOURCE_STATIONARY_BLEND", result["classification"])

    def test_stable_candidate_one_candidate_two_blend(self):
        result, _ = self._classify([([0,1.5,1.1],[0,.3,-.2])]*4)
        self.assertEqual("MULTI_SOURCE_STATIONARY_BLEND", result["classification"])

    def test_varying_near_tied_vectors_are_not_called_switching(self):
        vectors=[([1.00,.99,0],[.1,.1,0]), ([1.25,1.24,0],[.1,.1,0]),
                 ([.78,.79,0],[-.1,-.1,0]), ([1.45,1.44,0],[-.1,-.1,0])]
        result, _ = self._classify(vectors, noise=.08)
        self.assertEqual("SECTOR_VARIABLE_MULTI_SOURCE", result["classification"])

    def test_collinear_sources_are_unresolved(self):
        templates=np.ones((3,1,3))
        result, run = self._classify([([1,0,0],[0,0,0])]*4, templates=templates)
        self.assertFalse(run["sourceIdentifiable"])
        self.assertEqual("UNRESOLVED", result["classification"])

    def test_prepare_preserves_authoritative_bridge_and_candidates(self):
        bridge={**self.preparation, "ticID":277940827, "targetSky":{"raDeg":1,"decDeg":2},
                "catalogCandidates":[{"catalogIDs":{"ticID":1}}, {"catalogIDs":{"ticID":2}}],
                "spatialHypotheses":[]}
        stage050={"classification":"SOURCE_SWITCHING_BY_SECTOR",
                  "recommendedNextTest":"SOURCE_SWITCHING_TEMPORAL_MODEL"}
        with tempfile.TemporaryDirectory() as root:
            result=prepare_source_switching_temporal_model(
                difference_interpretation=stage050, difference_preparation=bridge,
                output_dir=Path(root), investigation_id="tic-277940827")
        self.assertEqual(bridge["catalogCandidates"], result["catalogCandidates"])
        self.assertEqual([1,2,3,4], result["subtractedHarmonicOrders"])
        self.assertFalse(result["physicalCycleResolved"])

    def test_production_path_requires_calibrated_official_prf_templates(self):
        times=np.linspace(2500,2527,120); cube=np.zeros((120,1,3)); valid=np.ones((1,3),bool)
        production={"sector":94,"times":times,"prewhitened":cube,"valid":valid,
                    "renderTemplates":lambda dx,dy: np.eye(3),
                    "calibrationImage":np.ones(3),"backgroundColumns":[]}
        inputs=[{**production,"sector":sector} for sector in self.preparation["sectors"]]
        calibrated={"available":True,"templates":np.eye(3),
                    "sharedAstrometricCalibration":{"dxPixels":0.,"dyPixels":0.,
                                                     "independentSourceMotion":False}}
        with mock.patch("workflows.tess.tess_source_switching_temporal._production_sector_inputs",
                        return_value=inputs) as acquire, mock.patch(
            "workflows.tess.tess_source_switching_temporal._fit_shared_astrometric_shift",
            return_value=calibrated) as calibrate:
            run_source_switching_temporal_model(self.preparation)
        acquire.assert_called_once_with(self.preparation)
        self.assertEqual(4, calibrate.call_count)

    def test_production_path_has_no_synthetic_template_fallback(self):
        item=self._inputs([([1,0,0],[0,0,0])]*4)[0]
        item.pop("sourceTemplates")
        with self.assertRaisesRegex(RuntimeError, "synthetic spatial-template fallback is forbidden"):
            run_source_switching_temporal_model(
                self.preparation, sector_inputs=[{**item,"sector":sector}
                                                 for sector in self.preparation["sectors"]])

    def test_stage_053_candidate_results_schedule_existing_validation(self):
        for model, index in (("CANDIDATE_1_STATIONARY", 0),
                             ("CANDIDATE_2_STATIONARY", 1)):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as root:
                root=Path(root); store=InvestigationStore(root/"store")
                investigation=store.create("tic-277940827","workflow","1")
                candidates=[{"raDeg":1.1,"decDeg":2.1,"catalogIDs":{"ticID":1}},
                            {"raDeg":1.2,"decDeg":2.2,"catalogIDs":{"ticID":2}}]
                preparation={**self.preparation,"artifactRoot":str(root/"artifacts"),
                             "catalogCandidates":candidates}
                (root/"artifacts").mkdir()
                other="CANDIDATE_2_STATIONARY" if index == 0 else "CANDIDATE_1_STATIONARY"
                run={"models":{model:{"bic":0.,"heldOutRSS":0.},
                               other:{"bic":10.,"heldOutRSS":10.}},
                     "sourceIdentifiable":True,"perSectorSourceCoherentVectors":[]}
                for stage_id, handler, result in (
                    ("051-prepare-source-switching-temporal-model",
                     "openstar.tess.source-switching-temporal-model.prepare",preparation),
                    ("052-run-source-switching-temporal-model",
                     "openstar.tess.source-switching-temporal-model.run",run)):
                    running=InvestigationStage(stage_id,handler,"RUNNING",None,{})
                    investigation=store.append_running_stage(investigation,running)
                    terminal=store.build_terminal_stage(
                        stage_id=stage_id,handler_id=handler,status="COMPLETE",
                        triggered_by_stage_id=None,parameters={},result=result,error=None,
                        software_id="test",software_version="1",started_at=running.started_at)
                    investigation=store.complete_current_stage(investigation,terminal)
                completed,next_stage=build_engine(store,object(),poll_interval=0,timeout=0).run_stage(
                    investigation,StageRequest("053-interpret-source-switching-temporal-model",
                    "openstar.tess.source-switching-temporal-model.interpret",{},
                    "052-run-source-switching-temporal-model"),software_id="test",software_version="1")
                self.assertEqual(candidates[index],completed.stages[-1].result["preferredCandidate"])
                self.assertEqual("054-prepare-offset-source-variability",next_stage.id)
                self.assertEqual("openstar.tess.offset-source-variability.prepare",next_stage.handler_id)
