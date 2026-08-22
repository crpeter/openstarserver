import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.tess.tess_frequency_localized_pixel import _response_map
from workflows.tess.tess_time_resolved_frequency_localization import (
    interpret_time_resolved_frequency_localization,
    prepare_time_resolved_frequency_localization,
    run_time_resolved_frequency_localization,
)
from workflows.tess.tess_time_resolved_residual_phase_localization import (
    run_time_resolved_residual_phase_localization,
)


class TimeResolvedFrequencyLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.base = {"ticID": 277940827, "sectors": [94, 95, 102, 103],
            "targetSky": {"raDeg": 10., "decDeg": 20.},
            "catalogCandidates": [{"raDeg": 10.1, "decDeg": 20.1, "catalogIDs": {"ticID": 111}},
                                  {"raDeg": 10.2, "decDeg": 20.2, "catalogIDs": {"ticID": 222}}],
            "referenceFamilyPeriodDays": 10.3, "subtractedHarmonicOrders": [7, 3, 1],
            "physicalCycleResolved": False, "residualReferenceFrequency": .45,
            "residualTimeReferenceDays": 2500., "fractionalFrequencyDriftPerDay": .0002,
            "spatialHypotheses": ["TARGET", "CANDIDATE_1", "CANDIDATE_2"]}
        self.centers = [{"componentID":"target", "x":1., "y":1.},
                        {"componentID":"candidate-1", "x":3., "y":1.},
                        {"componentID":"candidate-2", "x":2., "y":3.}]

    def inputs(self, sources):
        rng=np.random.default_rng(56); yy,xx=np.mgrid[:5,:5]; out=[]
        templates={c["componentID"]:np.exp(-((xx-c["x"])**2+(yy-c["y"])**2)/.45) for c in self.centers}
        for sector, pair in zip(self.base["sectors"], sources):
            t=np.linspace(2500,2520,640); cube=rng.normal(0,.03,(640,5,5))
            for idx,source in zip(np.array_split(np.arange(640),2),pair):
                warped=(t[idx]-2500)*(1+.5*self.base["fractionalFrequencyDriftPerDay"]*(t[idx]-2500))
                cube[idx]+=2*np.sin(2*np.pi*.45*warped)[:,None,None]*templates[source]
            out.append({"sector":sector,"times":t,"prewhitened":cube,"valid":np.ones((5,5),bool),
                        "componentPixelCenters":self.centers})
        return out

    def experiment(self, sources, prior=None):
        inputs=self.inputs(sources)
        old=run_time_resolved_residual_phase_localization(self.base,sector_inputs=inputs)
        if prior:
            for s, labels in zip(old["sectorResults"],prior):
                for w,label in zip(s["windowResults"],labels): w["classification"]=label
        stage056={"classification":"TIME_VARIABLE_LOCALIZATION","sourceAttributionResolved":False,
                  "physicalMechanismResolved":False,
                  "recommendedNextTest":"TIME_VARIABLE_SOURCE_LOCALIZATION_FOLLOWUP"}
        with tempfile.TemporaryDirectory() as root:
            prep=prepare_time_resolved_frequency_localization(stage054=self.base,stage055=old,
                stage056=stage056,output_dir=Path(root),investigation_id="tic-277940827")
            run=run_time_resolved_frequency_localization(prep,sector_inputs=inputs)
            result=interpret_time_resolved_frequency_localization(prep,run)
        return prep,run,result

    def test_stable_target_in_both_observables(self):
        _,run,result=self.experiment([["target","target"]]*4)
        self.assertEqual("STABLE_TARGET_LOCALIZATION",result["classification"])
        self.assertTrue(all(c["assessment"]=="REINFORCED" for s in result["sectorEvidence"] for c in s["stage056Comparison"]))
        self.assertTrue(all("coherentResponseMap" in w["response"] for s in run["sectorResults"] for w in s["windowResults"]))

    def test_stable_candidate_one_and_nondefault_harmonics_survive(self):
        prep,_,result=self.experiment([["candidate-1","candidate-1"]]*4)
        self.assertEqual("STABLE_CANDIDATE_1_LOCALIZATION",result["classification"])
        self.assertEqual([7,3,1],prep["subtractedHarmonicOrders"])
        self.assertEqual(self.base["catalogCandidates"][0],result["preferredCandidate"])

    def test_target_early_candidate_late_switches(self):
        _,_,result=self.experiment([["target","candidate-1"]]*4)
        self.assertEqual("WITHIN_SECTOR_SOURCE_SWITCHING_CONFIRMED",result["classification"])

    def test_moving_unmatched_centroid_is_confirmed(self):
        inputs=self.inputs([["target","target"]]*4)
        # Move the frozen hypotheses away while retaining two distinct coherent sources.
        for item,pair in zip(inputs, [[(0,4),(4,4)]]*4):
            t=item["times"]; yy,xx=np.mgrid[:5,:5]; item["prewhitened"]*=.02
            for idx,(x,y) in zip(np.array_split(np.arange(640),2),pair):
                item["prewhitened"][idx]+=2*np.sin(2*np.pi*.45*(t[idx]-2500))[:,None,None]*np.exp(-((xx-x)**2+(yy-y)**2)/.3)
        old=run_time_resolved_residual_phase_localization(self.base,sector_inputs=inputs)
        stage056={"classification":"TIME_VARIABLE_LOCALIZATION","sourceAttributionResolved":False,"physicalMechanismResolved":False,"recommendedNextTest":"TIME_VARIABLE_SOURCE_LOCALIZATION_FOLLOWUP"}
        with tempfile.TemporaryDirectory() as root:
            prep=prepare_time_resolved_frequency_localization(stage054=self.base,stage055=old,stage056=stage056,output_dir=Path(root),investigation_id="x")
            result=interpret_time_resolved_frequency_localization(prep,run_time_resolved_frequency_localization(prep,sector_inputs=inputs))
        self.assertEqual("TIME_VARIABLE_LOCALIZATION_CONFIRMED",result["classification"])

    def test_strong_power_poor_phase_concentration_fails_quality(self):
        t=np.linspace(0,12,500); rng=np.random.default_rng(9); cube=np.empty((500,4,4))
        for y in range(4):
            for x in range(4): cube[:,y,x]=np.sin(2*np.pi*.45*t+rng.uniform(0,2*np.pi))
        response=_response_map(times=t,residual_cube=cube,valid_pixels=np.ones((4,4),bool),frequency=.45,power_map=np.ones((4,4)))
        self.assertGreater(response["peakPower"],.05)
        self.assertLess(response["phaseConcentration"], .35)
        self.assertFalse(response["mapUsable"])

    def test_overlapping_candidates_do_not_force_switch(self):
        self.centers[2].update(x=3.05,y=1.)
        _,run,result=self.experiment([["candidate-1","candidate-2"]]*4)
        self.assertNotIn("SWITCHING",result["classification"])
        self.assertTrue(all(w["classification"] in {"MULTIPLE_OR_BLENDED","UNRESOLVED","NO_QUALITY_LOCALIZATION"} for s in run["sectorResults"] for w in s["windowResults"]))

    def test_independent_observable_disagreement_is_explicit(self):
        _,_,result=self.experiment([["target","target"]]*4,prior=[["CANDIDATE_1_SUPPORTED"]*2]*4)
        self.assertEqual("UNRESOLVED",result["classification"])
        self.assertTrue(any(c["assessment"]=="CONFLICTING_UNRESOLVED" for s in result["sectorEvidence"] for c in s["stage056Comparison"]))


if __name__ == "__main__": unittest.main()
