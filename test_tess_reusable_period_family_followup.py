import json
import math
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import Investigation, InvestigationStage, InvestigationStore, sha256_json
from openstar_targets import InvestigationTarget
from openstar_workflow import StageRequest
from workflows.tess.period_family_followup import (
    build_period_family_followup_recommendation,
    extract_gaia_context,
    freeze_period_family_contract,
    select_untouched_sectors,
)
from workflows.tess.tess_autonomy import plan_tess_branches
from workflows.tess.tess_investigation import build_engine
from workflows.tess.external_long_baseline import (
    ASASSNSkyPatrolProvider, MalformedProviderData, OfficialASASSNTransport,
    ProviderConfigurationUnavailable, ProviderUnavailable,
    analyze_seasonal_coherence, run_external_experiment,
)


class Transport:
    def __init__(self, rows, available=True): self.rows=rows; self.available=available
    def coverage(self, target): return {"available": self.available, "product":"asas-sn-lightcurve-v1"}
    def acquire(self, target, request): return json.dumps(self.rows).encode()


def rows(evolving=False):
    out=[]; period=4.55
    for season in range(4):
        for i in range(35):
            t=season*365.25+i*0.71
            phase=(0.35*season if evolving else 0)
            out.append({"time":t,"flux":math.sin(2*math.pi*t/period+phase),
                        "uncertainty":0.15,"band":"g","quality":"GOOD"})
    return out


class ReusableFollowupTests(unittest.TestCase):
    @staticmethod
    def _append_complete(store, investigation, stage_id, handler_id, result,
                         triggered_by=None, stop=False):
        running=InvestigationStage(stage_id,handler_id,"RUNNING",triggered_by,{})
        investigation=store.append_running_stage(investigation,running)
        terminal=store.build_terminal_stage(stage_id=stage_id,handler_id=handler_id,
            status="COMPLETE",triggered_by_stage_id=triggered_by,parameters={},
            result=result,error=None,software_id="test",software_version="1",
            started_at=running.started_at,stop=stop)
        return store.complete_current_stage(investigation,terminal)

    def test_production_evidence_builds_frozen_followup_recommendation(self):
        periods={93:4.5368,95:4.5136,96:4.5195,98:4.5558}
        datasets=[{"datasetID":f"d{sector}","candidatePeriodDays":period,
                   "candidateFrequency":1/period,"candidatePower":0.8,
                   "candidatePeakProminenceRatio":2.0,"candidateFoldCoherence":0.7,
                   "periodStatus":"RELIABLE","periodConfidence":"high"}
                  for sector,period in periods.items()]
        sector_results=[{"sector":sector,"datasetID":f"d{sector}",
                         "candidatePeriodDays":period,"candidateFrequency":1/period,
                         "recurrenceClassification":"RESOLUTION_LIMITED",
                         "resolutionLimited":True,"supportsTarget":False,
                         "eligibleForRecurrence":True,"boundaryHit":False}
                        for sector,period in periods.items()]
        evidence=[
            ("openstar.tess.prepare-target",{"ticID":238919539,"sector":1}),
            ("openstar.tess.primary-project.run",{"candidatePeriodDays":4.550327172,
                "candidateFrequency":1/4.550327172,"candidatePower":0.9,
                "periodStatus":"RELIABLE","periodConfidence":"high"}),
            ("openstar.tess.catalog-identity",{"ticID":238919539,
                "tic":{"metadata":{"raDeg":10.0,"decDeg":-20.0}},
                "tess":{"officialSectors":[1,93,95,96,98]}}),
            ("openstar.tess.independent.prepare",{"preparedSectors":[
                {"sector":sector} for sector in periods]}),
            ("openstar.tess.independent.run",{"datasets":datasets}),
            ("openstar.tess.independent.interpret",{"sectorResults":sector_results,
                "eligibleSectorCount":4,"supportingSectorCount":0,
                "resolutionLimitedSectorCount":4,
                "claimDecision":{"claim":"HUMAN_REVIEW_REQUIRED"},
                "contradictionPlan":{"action":"BROAD_INDEPENDENT_SEARCH",
                    "reliableSectorCount":4,
                    "reason":"targeted-candidate-not-recurrent-independent-sectors-contain-alternate-reliable-structure"}}),
        ]
        stages=tuple(InvestigationStage(f"{index:03d}-stage",handler,"COMPLETE",
            (f"{index-1:03d}-stage" if index else None),{},result=result)
            for index,(handler,result) in enumerate(evidence))
        investigation=Investigation("inv","workflow","1","RUNNING","now","now",{},stages)
        recommendation=build_period_family_followup_recommendation(investigation,{
            "claimDecision":{"claim":"HUMAN_REVIEW_REQUIRED"},
            "promotionEligible":False,"selectedPeriodDays":None},origin_stage_id="012-broad")
        self.assertEqual(recommendation["recommendedNextTest"],
                         "PERIOD_FAMILY_DIFFERENCE_IMAGE_LOCALIZATION")
        self.assertEqual(recommendation["periodFamilyContract"]["consumedSectors"],
                         [1,93,95,96,98])
        self.assertFalse(recommendation["frozenPeriodFamily"]["periodDetectionRecomputed"])

    def test_sector_selection_is_deterministic_and_excludes_consumed(self):
        years={3:2018,4:2018,20:2019,40:2021}
        catalog=[{"sector":s,"author":"SPOC","mission":f"TESS Sector {s}",
                  "exptimeSeconds":120,"observationYear":years[s]}
                 for s in [20,3,40,4]]
        first=select_untouched_sectors(catalog,[3])
        self.assertEqual(first["selectedSectors"],[4,20,40])
        self.assertEqual(first,select_untouched_sectors(reversed(catalog),[3]))
        self.assertFalse(first["fluxInspectedDuringSelection"])
        self.assertEqual(first["rejectedSectors"][0]["reason"],"already-consumed-by-time-domain-observable")
        self.assertEqual(select_untouched_sectors(catalog,[3,40])["status"],"INSUFFICIENT_EPOCH_COVERAGE")

    def test_three_semantic_autonomy_routes_and_quiescence(self):
        target=InvestigationTarget("x","inv","openstar.workflow.tess-investigation.v1","20.2",{})
        routes={
          "PERIOD_FAMILY_DIFFERENCE_IMAGE_LOCALIZATION":"generic-period-family-difference-imaging.prepare",
          "UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION":"generic-period-family-time-domain-evolution.prepare",
          "ADDITIONAL_LONG_BASELINE_TIME_DOMAIN_DATA":"external-long-baseline.prepare"}
        for n,(trigger,suffix) in enumerate(routes.items()):
            result={"recommendedNextTest":trigger,"periodFamilyResolved":False,
                    "claimDecision":{"claim":"HUMAN_REVIEW_REQUIRED"}}
            stage=InvestigationStage(f"00{n+1}-e","science","COMPLETE",None,{},result=result,stop=True)
            inv=Investigation(f"different-{n}",target.workflow_id,"20.2","RUNNING","now","now",{},(stage,))
            branch=plan_tess_branches(inv,target)[0]
            self.assertTrue(branch.experiment.handler_id.endswith(suffix))
            quiet=replace(inv,status="QUIESCENT_AWAITING_DATA")
            # Upgrade does not choose any of the new routes.
            self.assertFalse(any(b.experiment.handler_id.endswith(suffix) for b in plan_tess_branches(quiet,target)))

    def test_generic_handler_chain_preserves_contract_and_ledgers_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=InvestigationStore(Path(temporary)/"state")
            investigation=store.create("generic-chain","workflow","1",{})
            identity={"ticID":238919539,
                "tic":{"metadata":{"raDeg":10.0,"decDeg":-20.0}},
                "gaiaDR3":{"nearest":{"sourceID":"target"},
                    "queryProvenance":{"radiusArcsec":16.0},
                    "sources":[{"sourceID":"target","raDeg":10.0,"decDeg":-20.0,
                                "gMag":10.0,"separationArcsec":0.1}]},
                "tess":{"officialProducts":[
                    {"sector":5,"author":"SPOC","mission":"TESS Sector 5",
                     "exptimeSeconds":120,"observationYear":2018},
                    {"sector":20,"author":"SPOC","mission":"TESS Sector 20",
                     "exptimeSeconds":120,"observationYear":2019},
                    {"sector":40,"author":"SPOC","mission":"TESS Sector 40",
                     "exptimeSeconds":120,"observationYear":2021}]}}
            investigation=self._append_complete(store,investigation,"001-identity",
                "openstar.tess.catalog-identity",identity)
            family={"originStageID":"002-semantic","ticID":238919539,
                "targetSky":{"raDeg":10.0,"decDeg":-20.0},
                "primaryDetection":{"sector":1,"periodDays":4.55,
                    "frequencyCyclesPerDay":1/4.55},
                "independentSectorDetections":[
                    {"sector":sector,"periodDays":period,
                     "frequencyCyclesPerDay":1/period}
                    for sector,period in ((93,4.53),(95,4.51),(96,4.52),(98,4.56))],
                "primaryPeriodDays":4.55,"familyCenterDays":4.53,
                "periodFamilyMembersDays":[4.51,4.52,4.53,4.55,4.56],
                "familyAcceptanceWindowDays":[4.4,4.67],
                "consumedSectors":[1,93,95,96,98],
                "observableDefinition":"persisted-sector-period-phase-reference"}
            contract=freeze_period_family_contract({"frozenPeriodFamily":family},"002-semantic")
            semantic={"recommendedNextTest":"PERIOD_FAMILY_DIFFERENCE_IMAGE_LOCALIZATION",
                "periodFamilyResolved":False,"claimDecision":{"claim":"HUMAN_REVIEW_REQUIRED"},
                "autonomousContinuationEligible":True,"frozenPeriodFamily":family,
                "periodFamilyContract":contract,
                "periodFamilyContractSHA256":sha256_json(contract)}
            investigation=self._append_complete(store,investigation,"002-semantic","science",
                semantic,triggered_by="001-identity",stop=True)
            engine=build_engine(store,mock.Mock(),poll_interval=0.0,timeout=0.0)
            request=StageRequest("003-difference-prepare",
                "openstar.tess.generic-period-family-difference-imaging.prepare",{},"002-semantic")
            investigation,next_request=engine.run_stage(investigation,request,
                software_id="test",software_version="1")
            self.assertEqual(len(investigation.stages[-1].artifacts),2)
            with mock.patch("workflows.tess.tess_investigation.run_period_family_difference_imaging",
                            return_value={"sectorResults":[]}), \
                 mock.patch("workflows.tess.tess_investigation.interpret_period_family_difference_imaging",
                            return_value={"recommendedNextTest":"UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION",
                                "periodFamilyResolved":False,
                                "claimDecision":{"claim":"HUMAN_REVIEW_REQUIRED"}}):
                investigation,next_request=engine.run_stage(investigation,next_request,
                    software_id="test",software_version="1")
                investigation,_=engine.run_stage(investigation,next_request,
                    software_id="test",software_version="1")
            self.assertEqual(investigation.status,"QUIESCENT_AWAITING_DATA")
            self.assertEqual(investigation.stages[-1].result["periodFamilyContractSHA256"],
                             sha256_json(contract))
            investigation=store.set_status(investigation,"RUNNING")
            request=StageRequest("006-time-prepare",
                "openstar.tess.generic-period-family-time-domain-evolution.prepare",{},
                investigation.stages[-1].id)
            investigation,next_request=engine.run_stage(investigation,request,
                software_id="test",software_version="1")
            self.assertEqual(investigation.stages[-1].result["untouchedSectors"],[5,20,40])
            with mock.patch("workflows.tess.tess_investigation.run_period_family_time_domain_evolution",
                            return_value={"frozenDatasets":[],"sectorResults":[],"errors":[]}), \
                 mock.patch("workflows.tess.tess_investigation.interpret_period_family_time_domain_evolution",
                            return_value={"recommendedNextTest":"ADDITIONAL_LONG_BASELINE_TIME_DOMAIN_DATA",
                                "periodFamilyResolved":False,
                                "claimDecision":{"claim":"HUMAN_REVIEW_REQUIRED"}}):
                investigation,next_request=engine.run_stage(investigation,next_request,
                    software_id="test",software_version="1")
                investigation,_=engine.run_stage(investigation,next_request,
                    software_id="test",software_version="1")
            investigation=store.set_status(investigation,"RUNNING")
            request=StageRequest("009-external-prepare",
                "openstar.tess.external-long-baseline.prepare",{},investigation.stages[-1].id)
            investigation,next_request=engine.run_stage(investigation,request,
                software_id="test",software_version="1")
            with mock.patch("workflows.tess.tess_investigation.ASASSNSkyPatrolProvider.from_environment",
                            side_effect=ProviderConfigurationUnavailable("not installed")):
                investigation,next_request=engine.run_stage(investigation,next_request,
                    software_id="test",software_version="1")
            investigation,_=engine.run_stage(investigation,next_request,
                software_id="test",software_version="1")
            self.assertEqual(investigation.status,"QUIESCENT_AWAITING_DATA")
            self.assertEqual(investigation.stages[-2].result["operationalOutcome"],
                             "PROVIDER_CONFIGURATION_UNAVAILABLE")
            for stage in investigation.stages:
                if stage.status == "COMPLETE":
                    self.assertIsNotNone(store.verified_terminal_stage_ledger_hash(
                        investigation.id,stage))

    def test_asassn_success_unavailable_malformed_and_contamination(self):
        with tempfile.TemporaryDirectory() as d:
            provider=ASASSNSkyPatrolProvider(Transport(rows()))
            result=run_external_experiment(target={"raDeg":1,"decDeg":2}, family_window=[4.50,4.60],
                neighbors=[{"separationArcsec":30,"fluxFraction":0.0,"providerRadiusArcsec":16}],providers=[provider],artifact_root=Path(d))
            self.assertEqual(result["classification"],"EXTERNAL_STABLE_CLOCK_SUPPORTED")
            self.assertFalse(result["lombScarglePerformed"])
            self.assertTrue(Path(result["rawResponsePath"]).exists())
            self.assertEqual({item["role"] for item in result["artifactManifest"]},{
                "EXECUTION_PREREGISTRATION","PROVIDER_COVERAGE","RAW_PROVIDER_RESPONSE",
                "CLEANED_MEASUREMENTS","OBJECTIVE_QUALITY_GATE","ACQUISITION_METADATA"})
            recovered=run_external_experiment(target={"raDeg":1,"decDeg":2},
                family_window=[4.50,4.60],neighbors=[{"separationArcsec":30,
                    "fluxFraction":0.0,"providerRadiusArcsec":16}],providers=[provider],
                artifact_root=Path(d))
            self.assertEqual(result["acquiredAt"],recovered["acquiredAt"])
            changed=ASASSNSkyPatrolProvider(Transport(rows()+[{"time":3000,"flux":0,
                "uncertainty":.1,"band":"g","quality":"GOOD"}]))
            with self.assertRaises(RuntimeError):
                run_external_experiment(target={"raDeg":1,"decDeg":2},
                    family_window=[4.50,4.60],neighbors=[{"separationArcsec":30,
                        "fluxFraction":0.0,"providerRadiusArcsec":16}],providers=[changed],
                    artifact_root=Path(d))
        with self.assertRaises(ProviderUnavailable): ASASSNSkyPatrolProvider().coverage({})
        with self.assertRaises(MalformedProviderData): ASASSNSkyPatrolProvider(Transport([])).parse(b"bad")
        with tempfile.TemporaryDirectory() as d:
            result=run_external_experiment(target={},family_window=[4.5,4.6],
                neighbors=[{"separationArcsec":2,"fluxFraction":.5}],providers=[provider],artifact_root=Path(d))
            self.assertEqual(result["classification"],"EXTERNAL_CONTAMINATION_AMBIGUOUS")

    def test_real_gaia_schema_isolated_crowded_and_ambiguous(self):
        identity={"gaiaDR3":{"nearest":{"sourceID":"target"},
            "queryProvenance":{"radiusArcsec":16.0},"sources":[
            {"sourceID":"target","gMag":10.0,"separationArcsec":0.1},
            {"sourceID":"neighbor","gMag":12.5,"separationArcsec":5.0}]}}
        context=extract_gaia_context(identity)
        self.assertFalse(context["identityAmbiguous"])
        self.assertAlmostEqual(context["neighbors"][0]["fluxFraction"],0.1)
        ambiguous=extract_gaia_context({"gaiaDR3":{"sources":identity["gaiaDR3"]["sources"]}})
        self.assertTrue(ambiguous["identityAmbiguous"])
        self.assertIsNone(ambiguous["neighbors"])

    def test_gaia_context_fails_closed_when_catalog_cone_is_smaller_than_aperture(self):
        identity={"gaiaDR3":{"nearest":{"sourceID":"target"},
            "queryProvenance":{"radiusArcsec":5.0},"sources":[
            {"sourceID":"target","gMag":10.0,"separationArcsec":0.1}]}}
        context=extract_gaia_context(identity,aperture_arcsec=16.0)
        self.assertTrue(context["identityAmbiguous"])
        self.assertFalse(context["catalogCoverageCompleteForAperture"])
        self.assertIsNone(context["neighbors"])

    def test_official_skypatrol_adapter_uses_public_client_and_actual_schema(self):
        class Frame:
            def __init__(self, records): self.records=records
            def __len__(self): return len(self.records)
            def to_dict(self, orient):
                if orient != "records": raise AssertionError(orient)
                return self.records
        calls=[]
        class Client:
            def __init__(self): calls.append(("init",))
            def query_list(self, ids, **kwargs):
                calls.append(("query_list",ids,kwargs))
                if kwargs["download"]:
                    return types.SimpleNamespace(data=Frame([{"jd":2459000.5,"flux":1.1,
                        "flux_err":0.02,"phot_filter":"g","quality":"G",
                        "asas_sn_id":321}]))
                return Frame([{"asas_sn_id":321,"ra_deg":10.0,"dec_deg":-20.0}])
        package=types.ModuleType("pyasassn")
        client_module=types.ModuleType("pyasassn.client")
        client_module.SkyPatrolClient=Client
        with mock.patch.dict(sys.modules,{"pyasassn":package,"pyasassn.client":client_module}), \
             mock.patch("importlib.metadata.version",return_value="0.6.21"):
            transport=OfficialASASSNTransport()
            coverage=transport.coverage({"ticID":123,"raDeg":10.0,"decDeg":-20.0})
            raw=transport.acquire({"ticID":123},{})
        self.assertEqual(coverage["selectedSource"],"321")
        self.assertLess(coverage["matchSeparationArcsec"],1e-6)
        self.assertEqual(json.loads(raw)[0]["quality"],"G")
        self.assertEqual(calls[0],("init",))
        self.assertEqual(calls[1][2],{"catalog":"stellar_main","id_col":"tic_id","download":False})
        self.assertEqual(calls[2][2],{"catalog":"stellar_main","id_col":"tic_id","download":True})

    def test_stable_evolving_and_insufficient_science(self):
        stable=analyze_seasonal_coherence(rows(),[4.5,4.6])
        evolving=analyze_seasonal_coherence(rows(True),[4.5,4.6])
        insufficient=analyze_seasonal_coherence(rows()[:20],[4.5,4.6])
        self.assertEqual(stable["classification"],"EXTERNAL_STABLE_CLOCK_SUPPORTED")
        self.assertEqual(evolving["classification"],"EXTERNAL_EVOLVING_RECURRENCE_SUPPORTED")
        self.assertEqual(insufficient["classification"],"EXTERNAL_DATA_INSUFFICIENT")
        self.assertGreater(stable["periodUncertaintyDays"],0)
        self.assertGreater(stable["periodUncertainty"]["gridResolutionFloorDays"],0)
        self.assertTrue(stable["allSeasonsHeldOut"])
        self.assertEqual(len(stable["blockedSeasonFolds"]),stable["seasonCount"])
        self.assertGreater(stable["nullModel"]["fractionalImprovement"],0.1)
        self.assertTrue(stable["aliasAssessment"]["dailyAndSeasonalAliasesTested"])

    def test_null_and_band_disagreement_fail_closed(self):
        constant=[{**item,"flux":1.0} for item in rows()]
        null=analyze_seasonal_coherence(constant,[4.5,4.6])
        self.assertEqual(null["classification"],"EXTERNAL_RECURRENCE_NOT_REPLICATED")
        self.assertFalse(null["externalRecurrenceReplicated"])
        other=[]
        for item in rows():
            other.append({**item,"band":"V",
                          "flux":math.sin(2*math.pi*item["time"]/4.59)})
        disagreement=analyze_seasonal_coherence(rows()+other,[4.5,4.6])
        self.assertEqual(disagreement["classification"],
                         "EXTERNAL_PIPELINE_OR_BAND_DEPENDENT")
        self.assertFalse(disagreement["stableClockSupported"])

if __name__ == '__main__': unittest.main()
