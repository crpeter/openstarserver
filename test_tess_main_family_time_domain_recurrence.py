import math
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from workflows.tess.tess_main_family_time_domain_recurrence import (
    METHOD, ROTATION_MULTICYCLE, UNRESOLVED, analyze_sector,
    analyze_time_domain_recurrence, combine_sector_results,
)
from workflows.tess.tess_investigation import astrophysical_interpretation_continuation
from workflows.tess.tess_autonomy import repair_obsolete_terminal_wait
from openstar_investigation import ArtifactReference, Investigation, InvestigationStage, InvestigationStore, sha256_file
from openstar_workflow import StageRequest


class MainFamilyTimeDomainRecurrenceTests(unittest.TestCase):
    P = 3.600708338567666
    FAMILY = {"available": True, "representativeRawPeriodDays": 7.546257528330875,
              "possibleDoubleCycleDays": 15.09251505666175,
              "physicalCycleResolved": False, "supportingSectorIDs": [94, 95]}

    def sector(self, sector, alternating=False, gap=False, duration=40):
        time = np.arange(0, duration, .04)
        cycle = np.floor(time/self.P).astype(int)
        flux = np.sin(2*math.pi*time/self.P)
        if alternating:
            flux += .7*((-1.0)**cycle)*np.cos(4*math.pi*time/self.P)
        if gap:
            keep = ~(((time > 8)&(time < 11)) | ((time > 24)&(time < 27)))
            time, flux = time[keep], flux[keep]
        return {"sectorID":sector,"time":time.tolist(),"flux":flux.tolist()}

    def test_gap_aware_acf_is_deterministic_and_recovers_rotation(self):
        a = analyze_sector(**{"time":self.sector(1, gap=True)["time"],
            "flux":self.sector(1, gap=True)["flux"]}, sector_id=1,
            rotation_period_days=self.P, possible_double_days=self.FAMILY["possibleDoubleCycleDays"])
        b = analyze_sector(self.sector(1,gap=True)["time"],self.sector(1,gap=True)["flux"],
            sector_id=1,rotation_period_days=self.P,possible_double_days=self.FAMILY["possibleDoubleCycleDays"])
        self.assertEqual(a,b)
        self.assertAlmostEqual(self.P,a["rotationRecurrencePeak"]["lagDays"],delta=.2)
        self.assertIn("perResamplePeakLocationsDays",a["rotationJackknife"])

    def test_alternating_morphology_preserves_pairs(self):
        synthetic_family={"available":True,"representativeRawPeriodDays":2*self.P,
            "possibleDoubleCycleDays":4*self.P,"physicalCycleResolved":False,
            "supportingSectorIDs":[94,95]}
        result=analyze_time_domain_recurrence([self.sector(94,True,duration=19.2),
            self.sector(95,True,duration=19.2)],
            rotation_period_days=self.P,rotation_classification="ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED",
            main_photometric_family=synthetic_family)
        pairs=result["cycleRecurrenceSectorResults"][0]["cyclePairMeasurements"]
        self.assertGreaterEqual(len(pairs["2"]),3)
        self.assertGreater(np.median([x["correlation"] for x in pairs["2"]]),
                           np.median([x["correlation"] for x in pairs["1"]]))
        self.assertEqual(ROTATION_MULTICYCLE,result["classification"])
        self.assertTrue(result["decisionGates"]["cycleMorphologySupportsSameRelation"])
        self.assertFalse(result["physicalCycleResolved"])

    def test_materially_offset_persisted_family_is_not_falsely_rotation_related(self):
        offset_family={"available":True,"representativeRawPeriodDays":8.5,
            "possibleDoubleCycleDays":17.0,"physicalCycleResolved":False,
            "supportingSectorIDs":[94,95]}
        result=analyze_time_domain_recurrence([self.sector(94),self.sector(95)],
            rotation_period_days=self.P,
            rotation_classification="ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED",
            main_photometric_family=offset_family)
        self.assertEqual("FREQUENCY_FAMILY_NOT_TIME_DOMAIN_REPLICATED",
            result["classification"])
        self.assertFalse(result["mainFamilyRelationshipToRotationResolved"])

    def test_replication_required(self):
        synthetic={"sectorID":94,"rotationRecurrencePeak":{"lagDays":3.6,"localPeakWidthDays":.2,
            "uncertaintyEstimate":{"intervalDays":[3.55,3.65]}},
            "acfPeaks":[{"lagDays":7.2,"peakCorrelation":.8,"localPeakWidthDays":.2}],
            "cycleSeparationSummaries":{"1":{"pairCount":3,"medianCorrelation":0},
                "2":{"pairCount":3,"medianCorrelation":.9},"4":{"pairCount":0,"medianCorrelation":None}}}
        combined=combine_sector_results([synthetic],rotation_period_days=self.P,
            family_period_days=7.2,possible_double_days=14.4)
        self.assertEqual(UNRESOLVED,combined["classification"])

    def combined_fixture(self, sector, relation="related", order=2,
                         raw_coverage=True, double_coverage=True, morphology=None):
        lag=7.2 if order==2 else 14.4
        multiple={"order":order,"consistent":relation=="related"}
        evidence=lambda coverage,detected: {"coverageSufficient":coverage,
            "candidateWithinEmpiricalPeakUncertainty":detected,
            "sectorLocalRotationMultipleConsistency":multiple}
        summaries={str(k):{"pairCount":3,"medianCorrelation":
            (0.9 if k==(morphology or order) else 0.1)} for k in (1,2,4)}
        return {"sectorID":sector,"rotationRecurrencePeak":{"lagDays":3.6,
            "localPeakWidthDays":.2,"uncertaintyEstimate":{"intervalDays":[3.55,3.65]}},
            "acfPeaks":[{"lagDays":lag,"peakCorrelation":.8,"localPeakWidthDays":.2}],
            "rawFamilyCandidateEvidence":evidence(raw_coverage,raw_coverage),
            "possibleDoubleCandidateEvidence":evidence(double_coverage,False),
            "cycleSeparationSummaries":summaries}

    def combine(self, rows):
        return combine_sector_results(rows,rotation_period_days=self.P,
            family_period_days=7.2,possible_double_days=14.4)

    def test_inadequate_candidate_coverage_never_means_not_replicated(self):
        rows=[self.combined_fixture(i,raw_coverage=False,double_coverage=False) for i in (1,2,3)]
        for row in rows:
            row["rawFamilyCandidateEvidence"]["candidateWithinEmpiricalPeakUncertainty"]=False
        result=self.combine(rows)
        self.assertEqual(UNRESOLVED,result["classification"])
        self.assertEqual([],result["rawFamilyCoverageSectorIDs"])
        self.assertEqual("LONG_BASELINE_TIME_DOMAIN_RECURRENCE_DATA",result["recommendedNextTest"])

    def test_related_and_independent_evidence_fail_closed(self):
        for rows in ([self.combined_fixture(1),self.combined_fixture(2),
                      self.combined_fixture(3,"independent")],
                     [self.combined_fixture(1,"independent"),self.combined_fixture(2,"independent"),
                      self.combined_fixture(3)]):
            result=self.combine(rows)
            self.assertEqual(UNRESOLVED,result["classification"])
            self.assertFalse(result["decisionGates"]["noStrongerContradiction"])

    def test_pure_replication_and_same_morphology_order_resolve(self):
        related=self.combine([self.combined_fixture(1),self.combined_fixture(2)])
        self.assertEqual(ROTATION_MULTICYCLE,related["classification"])
        independent=self.combine([self.combined_fixture(1,"independent"),
            self.combined_fixture(2,"independent")])
        self.assertEqual("INDEPENDENT_LONGER_PERIOD_RECURRENCE_SUPPORTED",independent["classification"])

    def test_morphology_must_support_same_acf_multiple(self):
        matching=self.combine([self.combined_fixture(1,order=2,morphology=2),
            self.combined_fixture(2,order=2,morphology=2)])
        self.assertTrue(matching["decisionGates"]["cycleMorphologySupportsSameRelation"])
        mismatched=self.combine([self.combined_fixture(1,order=2,morphology=4),
            self.combined_fixture(2,order=2,morphology=4)])
        self.assertEqual(UNRESOLVED,mismatched["classification"])
        four=self.combine([self.combined_fixture(1,order=4,morphology=4),
            self.combined_fixture(2,order=4,morphology=4)])
        self.assertEqual(ROTATION_MULTICYCLE,four["classification"])
        mixed=self.combine([self.combined_fixture(1,order=2,morphology=2),
            self.combined_fixture(2,order=4,morphology=4)])
        self.assertEqual(UNRESOLVED,mixed["classification"])

    def test_short_sector_fails_closed_for_double(self):
        s=self.sector(94); keep=np.asarray(s["time"])<18
        result=analyze_sector(np.asarray(s["time"])[keep],np.asarray(s["flux"])[keep],
            sector_id=94,rotation_period_days=self.P,possible_double_days=15.09)
        self.assertFalse(result["canConstrainPossibleDouble"])

    def test_historical_values_are_dynamic_and_future_route_precedes_finalizer(self):
        summary={"targetResidualMechanismResolved":True,
            "classification":"ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED",
            "mainPhotometricFamily":dict(self.FAMILY)}
        request=astrophysical_interpretation_continuation(summary,request_id="031-target-residual-astrophysical-interpretation")
        self.assertEqual("032-main-family-time-domain-recurrence",request.id)
        self.assertNotEqual("openstar.tess.finalize",request.handler_id)
        self.assertEqual([1.0,18.0],METHOD["lagSearchDays"])

    def test_real_shaped_historical_auto_discovery_returns_unexecuted_034(self):
        from workflows.tess.tess_investigation import build_engine
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); store=InvestigationStore(root/"state")
            def complete(stage_id,handler,result,name,trigger=None,parameters=None,stop=False):
                path=root/name; path.write_text(json.dumps(result),encoding="utf-8")
                return InvestigationStage(stage_id,handler,"COMPLETE",trigger,parameters or {},
                    result=result,artifacts=(ArtifactReference(str(path),sha256_file(path),"application/json"),),stop=stop)
            def ledger_only(stage_id,handler,result):
                return InvestigationStage(stage_id,handler,"COMPLETE",None,{},
                    result=result,artifacts=())
            prepared=[]
            for sector in (94,95):
                times=np.arange(0,18,.04)
                flux=np.sin(2*np.pi*times/self.P)
                path=root/f"sector-{sector}.json"
                path.write_text(json.dumps({"times":times.tolist(),"flux":flux.tolist(),
                    "source":{"sector":sector,"originalTimeOriginDays":2459000.0}}))
                prepared.append({"sector":sector,"datasetPath":str(path)})
            family_value={"representativeRawPeriodDays":self.FAMILY["representativeRawPeriodDays"],
                "possibleDoubleCycleDays":self.FAMILY["possibleDoubleCycleDays"],
                "physicalCycleResolved":False,"supportingSectors":[94,95]}
            source={"classification":"ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED",
                "physicalMechanismResolved":True,"targetResidualMechanismResolved":True,
                "targetResidualPeriodDays":self.P,"smoothAmplitudeSupportingSectorIDs":[68,95],
                "mainPhotometricFamily":dict(self.FAMILY)}
            stages=(
                complete("001-prepare-target","openstar.tess.prepare-target",
                    {"preparedSectors":prepared},"prepare.json"),
                ledger_only("011-interpret-broad-independent-search",
                    "openstar.tess.independent.broad.interpret",{"harmonicFamily":family_value}),
                complete("018-mode-identification","openstar.tess.mode-identification.analyze",
                    {"modeCandidate":{"periodDays":self.P}},"mode.json"),
                complete("031-target-residual-astrophysical-interpretation",
                    "openstar.tess.target-residual-astrophysical-interpretation.analyze",source,
                    "target-residual-astrophysical-interpretation-v20.14.1.json"),
                complete("032-finalize","openstar.tess.finalize",
                    {"targetResidualAstrophysicalInterpretation":source},
                    "conclusion-v20.14.1-astrophysical-interpretation.json",
                    "031-target-residual-astrophysical-interpretation",
                    {"outputSuffix":"v20.14.1-astrophysical-interpretation"},True))
            terminal={"branchAssessments":[],"selectedExperiment":None,
                "schedulerAction":"INVESTIGATION_COMPLETE"}
            investigation=Investigation("historical-recurrence","openstar.workflow.tess-investigation.v1",
                "20.2","COMPLETE","now","now",{"controlState":terminal},stages)
            store.save(investigation)
            for item in stages:
                store._atomic_write_json(store.stage_path_for(investigation.id,item.id),asdict(item),replace=False)
            immutable={store.stage_path_for(investigation.id,s.id):store.stage_path_for(investigation.id,s.id).read_bytes() for s in stages}
            admitted=repair_obsolete_terminal_wait(store,investigation)
            selected=admitted.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("033-main-family-time-domain-recurrence",selected["id"])
            self.assertEqual((),stages[1].artifacts)
            engine=build_engine(store,SimpleNamespace(),poll_interval=0,timeout=None); engine.chain_stages=False
            completed,next_request=engine.run_stage(admitted,StageRequest(**selected),software_id="test",software_version="1")
            result=completed.stages[-1].result
            self.assertEqual([94,95],result["sectorsEvaluated"])
            self.assertTrue(all(x["rotationRecurrencePeak"] for x in result["acfSectorResults"]))
            self.assertTrue(all("rawFamilyCandidateEvidence" in x and "possibleDoubleCandidateEvidence" in x for x in result["acfSectorResults"]))
            self.assertTrue(all(not x["possibleDoubleCandidateEvidence"]["coverageSufficient"] for x in result["acfSectorResults"]))
            self.assertFalse(result["physicalCycleResolved"])
            self.assertEqual("034-finalize",next_request.id)
            self.assertFalse(any(s.id=="034-finalize" for s in completed.stages))
            self.assertTrue(all(path.read_bytes()==value for path,value in immutable.items()))


if __name__ == "__main__": unittest.main()
