import tempfile
import unittest
from pathlib import Path

import numpy as np

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
        result, _ = self._classify([([0,2,0],[0,-.3,0])]*4)
        self.assertEqual("STATIONARY_CANDIDATE_1_SOURCE", result["classification"])

    def test_switching_wins_bic_and_held_out_prediction(self):
        vectors=[([2,0,0],[.2,0,0]), ([2,0,0],[.2,0,0]),
                 ([0,2,0],[0,-.2,0]), ([0,2,0],[0,-.2,0])]
        result, run = self._classify(vectors)
        self.assertEqual("SOURCE_SWITCHING_CONFIRMED", result["classification"])
        self.assertEqual("SECTOR_VARYING_SOURCE_AMPLITUDES", result["bicWinningModel"])
        self.assertEqual("SECTOR_VARYING_SOURCE_AMPLITUDES", result["heldOutWinningModel"])
        self.assertEqual(4, len(run["perSectorSourceCoherentVectors"]))

    def test_stable_two_source_blend_is_not_switching(self):
        result, _ = self._classify([([1.5,1.1,0],[.3,-.2,0])]*4)
        self.assertEqual("MULTI_SOURCE_STATIONARY_BLEND", result["classification"])

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
