import unittest
from unittest import mock
import copy, json, tempfile, shutil, uuid
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from openstar_investigation import (ArtifactReference, InvestigationStage, InvestigationStore,
    StageProvenance, sha256_file, sha256_json)
from workflows.tess.tess_autonomy import repair_obsolete_terminal_wait, WORKFLOW_ID, WORKFLOW_VERSION
from openstar_path_relocation import HistoricalPathResolver
from openstar_workflow import StageRequest
import test_tess_target_residual_archival_baseline as archival_tests
try:
    import numpy as np
    NUMPY_AVAILABLE=hasattr(np,"ma") and hasattr(np,"sin") and hasattr(np,"linalg")
except ModuleNotFoundError:
    np=None; NUMPY_AVAILABLE=False

try:
    from workflows.tess.tess_target_residual_pixel_recurrence import (
        classify_centroid, freeze_catalog_hypotheses, interpret_sectors,
        acquire_selected_sector, CatalogInfrastructureError, NoPixelCoverageError,
        tpf_flux_cube, measure_sector,
    )
except ModuleNotFoundError as error:
    classify_centroid = None
    IMPORT_ERROR = error


@unittest.skipIf(classify_centroid is None, "optional numerical dependencies unavailable")
class TargetResidualPixelRecurrenceTests(unittest.TestCase):
    def _boundary(self,directory,classification="ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED"):
        base=archival_tests.TessTargetResidualArchivalBaselineTests(methodName="test_constants_preserve_generic_boundary")
        stages=base._lineage(directory)
        target=stages[0]; stages[0]=replace(target,result={**target.result,"datasetID":"d","targetName":"TIC 1","sector":1})
        family_index=next(i for i,s in enumerate(stages) if s.handler_id.endswith("harmonic-family.interpret"))
        family=stages[family_index]; family_result={**family.result,"claimDecision":{"claim":"CANDIDATE_PERIOD","rationale":["test"]},"selectedPeriodDays":10.,"selectedSource":"test"}
        family_path=Path(family.artifacts[0].path); family_path.write_text(json.dumps(family_result)+"\n")
        stages[family_index]=replace(family,result=family_result,artifacts=(replace(family.artifacts[0],sha256=sha256_file(family_path)),))
        stages.extend([
            InvestigationStage("003-catalog-identity","openstar.tess.catalog-identity","COMPLETE","002-primary",{},result={"tic":{"metadata":{"raDeg":10.,"decDeg":20.}}}),
            InvestigationStage("003-hypotheses","openstar.tess.hypotheses","COMPLETE","002-primary",{},result={"observedPeriodDays":5.}),
            InvestigationStage("004-planner","openstar.tess.planner","COMPLETE","003-hypotheses",{},result={"claimDecision":{"claim":"CANDIDATE_PERIOD","rationale":["test"]}}),
        ])
        def artifact(name,value):
            path=Path(directory)/name; path.write_text(json.dumps(value)+"\n")
            return ArtifactReference(str(path),sha256_file(path),"application/json")
        selected={"sector":2,"originalTimeOriginDays":100.,"selectionReason":"earliest supporting observation epoch"}
        prep_result={"preparedSectors":[{"sector":2}]}
        run_result={"datasets":[{"sector":2}]}
        evidence={"sector":2,"candidateFrequency":.1,"candidateFrequencyConfidenceInterval":[.09,.11],
            "originalTimeOriginDays":100.,"selectionReason":"earliest supporting observation epoch",
            "supportsHistoricalResidualFamily":True,"recurrenceClassification":"SUPPORTING_HISTORICAL_RESIDUAL_FAMILY"}
        science={"classification":classification,"recommendedNextTest":"PIXEL_LEVEL_SOURCE_RESOLVED_RESIDUAL_RECURRENCE_VALIDATION",
            "sourceAttributionResolved":False,"physicalMechanismResolved":False,"crossSectorPhaseUsed":False,
            "historicalResidualDriftExtrapolated":False,"selectedFuturePixelFollowupSectors":[selected],"sectorEvidence":[evidence]}
        s32=InvestigationStage("032-target-residual-archival-baseline-prepare","openstar.tess.target-residual-archival-baseline.prepare","COMPLETE",stages[-1].id,{},result=prep_result,artifacts=(artifact("target-residual-archival-baseline-prepare-v20.17.json",prep_result),))
        s33=InvestigationStage("033-target-residual-archival-baseline-run","openstar.tess.target-residual-archival-baseline.run","COMPLETE",s32.id,{},result=run_result)
        s34=InvestigationStage("034-target-residual-archival-baseline-interpret","openstar.tess.target-residual-archival-baseline.interpret","COMPLETE",s33.id,{},result=science,
            artifacts=(artifact("target-residual-archival-baseline-v20.17.json",science),),provenance=StageProvenance("test","1",{"preparation":sha256_json(prep_result),"distributedResult":sha256_json(run_result)}))
        conclusion={"targetResidualArchivalBaselineExtension":science,"recommendedNextTest":science["recommendedNextTest"]}
        s35=InvestigationStage("035-finalize","openstar.tess.finalize","COMPLETE",s34.id,{"outputSuffix":"v20.17-target-residual-archival-baseline"},result=conclusion,
            artifacts=(artifact("conclusion-v20.17-target-residual-archival-baseline.json",conclusion),),stop=True)
        store=InvestigationStore(Path(directory)/"store"); inv=store.create("v18",WORKFLOW_ID,WORKFLOW_VERSION)
        inv=replace(inv,stages=tuple(stages+[s32,s33,s34,s35]))
        inv=store.set_control_state(inv,status="COMPLETE",control_state={"branchAssessments":[],"selectedExperiment":None,"schedulerAction":"INVESTIGATION_COMPLETE"})
        return store,inv

    def _replace_science_consistently(self,inv,result):
        stages=list(inv.stages); science=stages[-2]
        science_path=Path(science.artifacts[0].path); science_path.write_text(json.dumps(result)+"\n")
        science=replace(science,result=result,artifacts=(replace(science.artifacts[0],sha256=sha256_file(science_path)),))
        final=stages[-1]; conclusion={**final.result,"targetResidualArchivalBaselineExtension":result,
            "recommendedNextTest":result.get("recommendedNextTest")}
        final_path=Path(final.artifacts[0].path); final_path.write_text(json.dumps(conclusion)+"\n")
        final=replace(final,result=conclusion,artifacts=(replace(final.artifacts[0],sha256=sha256_file(final_path)),))
        stages[-2:]=[science,final]; return replace(inv,stages=tuple(stages))

    def test_supported_and_suggestive_boundaries_admit_append_only_idempotently(self):
        for classification in ("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED","ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUGGESTIVE"):
            with self.subTest(classification=classification),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory,classification); before=copy.deepcopy(inv.stages)
                admitted=repair_obsolete_terminal_wait(store,inv)
                self.assertEqual("036-target-residual-pixel-recurrence-prepare",admitted.metadata["controlState"]["selectedExperiment"]["id"])
                self.assertEqual(before,admitted.stages)
                self.assertEqual(admitted,repair_obsolete_terminal_wait(store,admitted))

    def test_complete_v2017_history_relocates_without_rewriting_immutable_paths(self):
        root=Path.cwd()/f".v2018-relocation-{uuid.uuid4().hex}"; self.addCleanup(shutil.rmtree,root,True)
        old=root/"OLD"; new=root/"NEW"; active=root/"ACTIVE"; old.mkdir(parents=True)
        _,inv=self._boundary(old); before=copy.deepcopy(inv.stages)
        paths=tuple(ref.path for stage in inv.stages for ref in stage.artifacts)
        hashes=tuple(ref.sha256 for stage in inv.stages for ref in stage.artifacts)
        shutil.copytree(old,new); shutil.rmtree(old)
        store=InvestigationStore(active); admitted=repair_obsolete_terminal_wait(store,inv,
            historical_path_resolver=HistoricalPathResolver({old:new}))
        self.assertEqual("036-target-residual-pixel-recurrence-prepare",admitted.metadata["controlState"]["selectedExperiment"]["id"])
        self.assertEqual(before,admitted.stages)
        self.assertEqual(paths,tuple(ref.path for stage in admitted.stages for ref in stage.artifacts))
        self.assertEqual(hashes,tuple(ref.sha256 for stage in admitted.stages for ref in stage.artifacts))
        self.assertFalse(old.exists())

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for full mocked workflow lifecycle")
    def test_full_mocked_v2018_lifecycle_is_append_only_durable_and_ordered(self):
        from workflows.tess.tess_investigation import build_engine
        root=Path.cwd()/f".v2018-lifecycle-{uuid.uuid4().hex}"; self.addCleanup(shutil.rmtree,root,True); root.mkdir()
        store,inv=self._boundary(root); historical=copy.deepcopy(inv.stages); historical_paths=tuple(
            ref.path for stage in inv.stages for ref in stage.artifacts)
        admitted=repair_obsolete_terminal_wait(store,inv); selected=admitted.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("036-target-residual-pixel-recurrence-prepare",selected["id"])
        calls=[]; frozen=[{"sourceID":"TIC-1","isTarget":True,"ticID":1,"gaiaDR3SourceID":11,"raDeg":10.,"decDeg":20.},
            {"sourceID":"TIC-2","isTarget":False,"ticID":2,"gaiaDR3SourceID":22,"raDeg":10.01,"decDeg":20.}]
        def freeze(**kwargs):
            calls.append("catalog-freeze"); return {"catalogHypotheses":copy.deepcopy(frozen),"catalogQueries":{"tic":{"sources":[]},"gaiaDR3":{"sources":[]}},"queryProvenance":{"mocked":True}}
        class WCS:
            def world_to_pixel(self,coord): return (1.,1.) if coord[0]==10. else (4.,4.)
            def pixel_to_world(self,x,y): return SimpleNamespace(ra=SimpleNamespace(deg=10.),dec=SimpleNamespace(deg=20.))
        tpf=SimpleNamespace(time=SimpleNamespace(value=np.linspace(0.,27.,400)),flux=SimpleNamespace(value=np.ones((400,5,5))),wcs=WCS())
        def download(**kwargs): calls.append("pixel-download"); return tpf,{"sourceType":"OFFICIAL_TPF","author":"SPOC","cadenceSeconds":120.}
        measured={"candidateFrequencyUsed":.1,"establishedFamilyPrewhitening":{"harmonicOrders":[1,2]},"highCadenceCount":100,"lowCadenceCount":100,
            "differenceImage":[],"snrImage":[],"peakSNR":10.,"centroidX":1.,"centroidY":1.,"centroidUncertaintyPixels":.1,
            "jackknifeCentroids":[],"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":"TIC-1","distancesPixels":{"TIC-1":0.,"TIC-2":3.},
            "crossSectorPhaseUsed":False,"historicalResidualDriftExtrapolated":False}
        engine=build_engine(store,SimpleNamespace(),poll_interval=0,timeout=None); engine.chain_stages=False
        request=StageRequest(**selected)
        with mock.patch("workflows.tess.tess_investigation.freeze_catalog_hypotheses",side_effect=freeze),mock.patch(
                "workflows.tess.tess_residual_localization._download_tpf",side_effect=download),mock.patch(
                "workflows.tess.tess_investigation.measure_sector",return_value=measured),mock.patch(
                "workflows.tess.tess_offset_variability._skycoord",side_effect=lambda ra,dec:(ra,dec)):
            inv,request=engine.run_stage(admitted,request,software_id="test",software_version="20.35")
            self.assertEqual(["catalog-freeze"],calls); persisted_hypotheses=copy.deepcopy(inv.stages[-1].result["catalogHypotheses"])
            inv,request=engine.run_stage(inv,request,software_id="test",software_version="20.35")
            self.assertEqual(["catalog-freeze","pixel-download"],calls)
            projected=inv.stages[-1].result["sectorResults"][0]["catalogPixelPositions"]
            self.assertEqual([x["sourceID"] for x in persisted_hypotheses],[x["sourceID"] for x in projected])
            self.assertEqual(persisted_hypotheses,[{k:v for k,v in x.items() if k not in {"x","y"}} for x in projected])
            inv,request=engine.run_stage(inv,request,software_id="test",software_version="20.35")
            stage038=inv.stages[-1]
            inv,request=engine.run_stage(inv,request,software_id="test",software_version="20.35")
        appended=inv.stages[len(historical):]
        self.assertEqual(["036-target-residual-pixel-recurrence-prepare","037-target-residual-pixel-recurrence-run",
            "038-target-residual-pixel-recurrence-interpret","039-finalize"],[x.id for x in appended])
        self.assertTrue(all(x.status=="COMPLETE" for x in appended)); self.assertEqual(historical,inv.stages[:len(historical)])
        self.assertEqual(historical_paths,tuple(ref.path for stage in inv.stages[:len(historical)] for ref in stage.artifacts))
        active=store.directory_for(inv.id).resolve()
        for stage in appended:
            for ref in stage.artifacts:
                self.assertTrue(Path(ref.path).resolve().is_relative_to(active)); self.assertFalse(str(ref.path).startswith(("/tmp/","/private/tmp/","/var/tmp/")))
        final=appended[-1]
        self.assertEqual({"outputSuffix":"v20.18-target-residual-pixel-recurrence-validation"},final.parameters)
        self.assertEqual(stage038.result,final.result["targetResidualPixelRecurrenceValidation"])
        self.assertEqual(stage038.result["recommendedNextTest"],final.result["recommendedNextTest"])
        self.assertNotEqual("PIXEL_LEVEL_SOURCE_RESOLVED_RESIDUAL_RECURRENCE_VALIDATION",final.result["recommendedNextTest"])
        self.assertTrue(final.stop); self.assertIsNone(request); self.assertFalse(any(x.id.startswith("040-") for x in inv.stages))

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for workflow handler integration")
    def test_stage036_catalog_failure_is_persisted_retryable_without_stage037(self):
        from workflows.tess.tess_investigation import build_engine
        from openstar_workflow import RetryableExecutionError
        root=Path.cwd()/f".v2018-catalog-retry-{uuid.uuid4().hex}"; self.addCleanup(shutil.rmtree,root,True); root.mkdir()
        store,inv=self._boundary(root); admitted=repair_obsolete_terminal_wait(store,inv)
        request=StageRequest(**admitted.metadata["controlState"]["selectedExperiment"])
        engine=build_engine(store,SimpleNamespace(),poll_interval=0,timeout=None); engine.chain_stages=False
        with mock.patch("workflows.tess.tess_investigation.freeze_catalog_hypotheses",side_effect=CatalogInfrastructureError("outage")),self.assertRaises(RetryableExecutionError):
            engine.run_stage(admitted,request,software_id="test",software_version="20.35")
        failed=store.load(inv.id).stages[-1]
        self.assertEqual("FAILED",failed.status); self.assertEqual("TRANSIENT_INFRASTRUCTURE",failed.failure_classification)
        self.assertFalse(any(x.id=="037-target-residual-pixel-recurrence-run" for x in store.load(inv.id).stages))

    def test_nonrecurrence_wrong_recommendation_and_existing_attempt_refuse(self):
        for change in ("classification","baseline","recommendation","existing"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory)
                if change=="existing":
                    inv=replace(inv,stages=inv.stages+(InvestigationStage("036-x","openstar.tess.target-residual-pixel-recurrence.prepare","FAILED",inv.stages[-1].id,{}),))
                else:
                    result=copy.deepcopy(inv.stages[-2].result)
                    result["classification"]=("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_NOT_ESTABLISHED" if change=="classification" else
                        "ARCHIVAL_TARGET_RESIDUAL_BASELINE_INSUFFICIENT" if change=="baseline" else result["classification"])
                    if change=="recommendation": result["recommendedNextTest"]="WRONG"
                    inv=self._replace_science_consistently(inv,result)
                self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_missing_v2017_science_or_finalizer_artifact_refuses(self):
        for index in (-2,-1):
            with self.subTest(stage=index),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory); Path(inv.stages[index].artifacts[0].path).unlink()
                self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_malformed_complete_finalizer_refuses_without_attribute_error(self):
        with tempfile.TemporaryDirectory() as directory:
            store,inv=self._boundary(directory); stages=list(inv.stages)
            path=Path(stages[-1].artifacts[0].path); path.write_text("null\n")
            stages[-1]=replace(stages[-1],result=None,artifacts=(replace(stages[-1].artifacts[0],sha256=sha256_file(path)),)); inv=replace(inv,stages=tuple(stages))
            self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_selected_metadata_disagreement_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            store,inv=self._boundary(directory); stages=list(inv.stages); result=copy.deepcopy(stages[-2].result)
            result["selectedFuturePixelFollowupSectors"][0]["originalTimeOriginDays"]=101.
            inv=self._replace_science_consistently(inv,result)
            self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_altered_run_binding_and_non_supporting_selection_refuse(self):
        for change in ("run","support"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory); stages=list(inv.stages)
                if change=="run": stages[-3]=replace(stages[-3],result={"datasets":[{"altered":True}]})
                else:
                    result=copy.deepcopy(stages[-2].result); result["sectorEvidence"][0]["supportsHistoricalResidualFamily"]=False
                    inv=self._replace_science_consistently(inv,result); stages=list(inv.stages)
                if change=="run": inv=replace(inv,stages=tuple(stages))
                self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_invalid_frozen_frequency_and_duplicate_selected_sector_semantically_refuse(self):
        for value in (0.,-1.,float("inf"),"duplicate"):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory); result=copy.deepcopy(inv.stages[-2].result)
                if value=="duplicate": result["selectedFuturePixelFollowupSectors"].append(copy.deepcopy(result["selectedFuturePixelFollowupSectors"][0]))
                else: result["sectorEvidence"][0]["candidateFrequency"]=value
                inv=self._replace_science_consistently(inv,result)
                self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_unique_target_and_catalog_localizations(self):
        hypotheses=[{"sourceID":"target","x":1.0,"y":1.0},
                    {"sourceID":"catalog","x":4.0,"y":4.0}]
        self.assertEqual("target",classify_centroid((1.0,1.0),hypotheses,.1,100)["preferredSource"])
        self.assertEqual("catalog",classify_centroid((4.0,4.0),hypotheses,.1,100)["preferredSource"])

    def test_ambiguity_uncertainty_and_snr_gates(self):
        hypotheses=[{"sourceID":"a","x":1.0,"y":1.0},{"sourceID":"b","x":1.2,"y":1.0}]
        self.assertIsNone(classify_centroid((1.1,1),hypotheses,.01,100)["preferredSource"])
        self.assertIsNone(classify_centroid((1,1),hypotheses,1.0,100)["preferredSource"])
        self.assertIsNone(classify_centroid((1,1),hypotheses,.01,0)["preferredSource"])

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for synthetic pixel-cube science")
    def test_masked_tpf_flux_is_filled_with_nan(self):
        masked=np.ma.array(np.ones((3,2,2)),mask=False); masked.mask[1,:,:]=True
        cube=tpf_flux_cube(type("TPF",(),{"flux":type("Flux",(),{"value":masked})()})())
        self.assertTrue(np.isnan(cube[1]).all()); self.assertEqual(np.float64,cube.dtype)

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for synthetic pixel-cube science")
    def test_measure_sector_localizes_frozen_sector_frequency_without_warp(self):
        times=np.linspace(100.,127.,1200); established=.2; candidate=.113
        rng=np.random.default_rng(1818); cube=rng.normal(0,.03,(len(times),7,7))
        main=np.sin(2*np.pi*established*times)
        cube += main[:,None,None]
        residual=np.sin(2*np.pi*candidate*(times-times[0]))
        yy,xx=np.mgrid[:7,:7]; profile=np.exp(-((xx-3.)**2+(yy-3.)**2)/.7)
        cube += 1.5*residual[:,None,None]*profile[None,:,:]
        hypotheses=[{"sourceID":"target","x":3.,"y":3.},{"sourceID":"other","x":6.,"y":6.}]
        result=measure_sector(times,cube,np.ones((7,7),bool),established_frequency=established,
            candidate_frequency=candidate,hypotheses=hypotheses)
        self.assertEqual(candidate,result["candidateFrequencyUsed"])
        self.assertEqual("target",result["preferredSource"])
        self.assertEqual([1,2],result["establishedFamilyPrewhitening"]["harmonicOrders"])
        self.assertFalse(result["crossSectorPhaseUsed"]); self.assertFalse(result["historicalResidualDriftExtrapolated"])

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for synthetic pixel-cube science")
    def test_measure_sector_catalog_signal_and_close_blend(self):
        times=np.linspace(0.,27.,1200); rng=np.random.default_rng(1819)
        cube=rng.normal(0,.02,(len(times),7,7)); signal=np.sin(2*np.pi*.117*times)
        yy,xx=np.mgrid[:7,:7]; cube += signal[:,None,None]*np.exp(-((xx-5.)**2+(yy-3.)**2)/.6)[None,:,:]
        catalog=[{"sourceID":"target","x":1.,"y":1.},{"sourceID":"catalog","x":5.,"y":3.}]
        result=measure_sector(times,cube,np.ones((7,7),bool),established_frequency=.2,candidate_frequency=.117,hypotheses=catalog)
        self.assertEqual("catalog",result["preferredSource"])
        close=[{"sourceID":"a","x":result["centroidX"]-.1,"y":result["centroidY"]},
               {"sourceID":"b","x":result["centroidX"]+.1,"y":result["centroidY"]}]
        blended=measure_sector(times,cube,np.ones((7,7),bool),established_frequency=.2,candidate_frequency=.117,hypotheses=close)
        self.assertIsNone(blended["preferredSource"])

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for synthetic pixel-cube science")
    def test_measure_sector_snr_and_jackknife_uncertainty_block_attribution(self):
        times=np.linspace(0.,27.,500); cube=np.random.default_rng(1820).normal(0,.1,(len(times),4,4))
        hypotheses=[{"sourceID":"target","x":1.,"y":1.},{"sourceID":"other","x":3.,"y":3.}]
        image={"centroidX":1.,"centroidY":1.,"peakSNR":100.,"differenceImage":[],"snrImage":[]}
        with mock.patch("workflows.tess.tess_difference_image._centroid_from_frames",return_value=image),mock.patch(
                "workflows.tess.tess_difference_image._jackknife_uncertainty",return_value=(10.,[])):
            uncertain=measure_sector(times,cube,np.ones((4,4),bool),established_frequency=.2,candidate_frequency=.11,hypotheses=hypotheses)
        self.assertIsNone(uncertain["preferredSource"]); self.assertEqual(10.,uncertain["centroidUncertaintyPixels"])
        low={**image,"peakSNR":0.}
        with mock.patch("workflows.tess.tess_difference_image._centroid_from_frames",return_value=low),mock.patch(
                "workflows.tess.tess_difference_image._jackknife_uncertainty",return_value=(.1,[])):
            weak=measure_sector(times,cube,np.ones((4,4),bool),established_frequency=.2,candidate_frequency=.11,hypotheses=hypotheses)
        self.assertIsNone(weak["preferredSource"]); self.assertEqual(0.,weak["peakSNR"])

    def test_cross_sector_resolution_threshold_and_switching(self):
        rows=[{"sector":n,"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":"target"} for n in range(3)]
        result=interpret_sectors(rows,"target")
        self.assertTrue(result["sourceAttributionResolved"])
        self.assertFalse(result["crossSectorPhaseUsed"])
        self.assertFalse(result["historicalResidualDriftExtrapolated"])
        self.assertFalse(interpret_sectors(rows[:2],"target")["sourceAttributionResolved"])
        switched=rows[:2]+[{"sector":3+n,"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":"other"} for n in range(2)]
        self.assertEqual("PIXEL_RECURRENCE_SOURCE_SWITCHING_OR_BLEND",interpret_sectors(switched,"target")["classification"])
        self.assertNotIn("target",result["supportByCatalogSource"])

    def _catalog(self,tic_sources,gaia_sources):
        return freeze_catalog_hypotheses(tic_id=1,ra_deg=10.,dec_deg=20.,
            coordinate_factory=lambda *x:x,query_tic=lambda *_:{"found":True,"sources":tic_sources},
            query_gaia=lambda *_:{"found":True,"sources":gaia_sources})

    def test_catalog_target_explicit_duplicates_merge_and_raw_provenance_persist(self):
        tic=[{"catalog":"TIC","ticID":1,"isTargetTIC":True,"gaiaSourceID":11,"raDeg":10.,"decDeg":20.,"separationArcsec":0.},
             {"catalog":"TIC","ticID":2,"isTargetTIC":False,"gaiaSourceID":22,"raDeg":10.01,"decDeg":20.,"separationArcsec":34.}]
        gaia=[{"catalog":"GaiaDR3","gaiaSourceID":22,"raDeg":10.01001,"decDeg":20.,"separationArcsec":34.}]
        frozen=self._catalog(tic,gaia)
        self.assertEqual(["TIC-1","TIC-2"],[x["sourceID"] for x in frozen["catalogHypotheses"]])
        self.assertEqual(11,frozen["catalogHypotheses"][0]["gaiaDR3SourceID"])
        self.assertEqual(tic,frozen["catalogQueries"]["tic"]["sources"])
        self.assertTrue(frozen["queryProvenance"]["responsesPersistedVerbatim"])

    def test_distinct_catalog_sources_remain_distinct(self):
        gaia=[{"catalog":"GaiaDR3","gaiaSourceID":22,"raDeg":10.01,"decDeg":20.,"separationArcsec":34.},
              {"catalog":"GaiaDR3","gaiaSourceID":33,"raDeg":10.02,"decDeg":20.,"separationArcsec":68.}]
        self.assertEqual(3,len(self._catalog([],gaia)["catalogHypotheses"]))

    def test_catalog_transient_failure_does_not_freeze_incomplete_set(self):
        with self.assertRaises(CatalogInfrastructureError):
            freeze_catalog_hypotheses(tic_id=1,ra_deg=10.,dec_deg=20.,coordinate_factory=lambda *x:x,
                query_tic=lambda *_:{"sources":[],"queryError":"timeout"},query_gaia=lambda *_:{"sources":[]})

    def test_acquisition_boundary_only_accepts_exact_no_coverage(self):
        def absent(**kwargs): raise RuntimeError(f"No official TPF or TESScut coverage available for Sector {kwargs['sector']}.")
        with self.assertRaises(NoPixelCoverageError): acquire_selected_sector(absent,sector=2)
        for message in ("corrupt pixels","TESScut download returned no data for Sector 2."):
            with self.subTest(message=message),self.assertRaises(RuntimeError):
                acquire_selected_sector(lambda **_:(_ for _ in ()).throw(RuntimeError(message)),sector=2)

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for workflow handler integration")
    def test_stage037_transient_archive_failure_is_persisted_retryable(self):
        from workflows.tess.tess_investigation import build_engine
        from workflows.tess.tess_sector_archive import TessArchiveTransientError
        from openstar_workflow import StageRequest, RetryableExecutionError
        root=Path.cwd()/f".v2018-handler-{uuid.uuid4().hex}"; self.addCleanup(shutil.rmtree,root,True)
        store=InvestigationStore(root); inv=store.create("transient",WORKFLOW_ID,WORKFLOW_VERSION)
        prep={"ticID":1,"targetSky":{"raDeg":1.,"decDeg":2.},"catalogHypotheses":[],
            "selectedSectorEvidence":[{"sector":2,"candidateFrequency":.1}],"frozenEstablishedPhysicalFrequency":.2}
        stage=InvestigationStage("036-target-residual-pixel-recurrence-prepare","openstar.tess.target-residual-pixel-recurrence.prepare","COMPLETE",None,{},result=prep)
        inv=replace(inv,stages=(stage,)); engine=build_engine(store,SimpleNamespace(),poll_interval=0,timeout=None); engine.chain_stages=False
        with mock.patch("workflows.tess.tess_residual_localization._download_tpf",side_effect=TessArchiveTransientError("outage")),self.assertRaises(RetryableExecutionError):
            engine.run_stage(inv,StageRequest("037-target-residual-pixel-recurrence-run","openstar.tess.target-residual-pixel-recurrence.run",{},stage.id),software_id="test",software_version="1")
        failed=store.load(inv.id).stages[-1]
        self.assertEqual("FAILED",failed.status); self.assertEqual("TRANSIENT_INFRASTRUCTURE",failed.failure_classification)

    @unittest.skipUnless(NUMPY_AVAILABLE,"NumPy required for workflow handler integration")
    def test_stage037_exact_no_coverage_continues_without_substitution(self):
        from workflows.tess.tess_investigation import build_engine
        from openstar_workflow import StageRequest
        root=Path.cwd()/f".v2018-handler-{uuid.uuid4().hex}"; self.addCleanup(shutil.rmtree,root,True)
        store=InvestigationStore(root); inv=store.create("unavailable",WORKFLOW_ID,WORKFLOW_VERSION)
        selected=[{"sector":2,"candidateFrequency":.1},{"sector":65,"candidateFrequency":.11}]
        prep={"ticID":1,"targetSky":{"raDeg":1.,"decDeg":2.},"catalogHypotheses":[],
            "selectedSectorEvidence":selected,"frozenEstablishedPhysicalFrequency":.2}
        stage=InvestigationStage("036-target-residual-pixel-recurrence-prepare","openstar.tess.target-residual-pixel-recurrence.prepare","COMPLETE",None,{},result=prep)
        inv=replace(inv,stages=(stage,)); engine=build_engine(store,SimpleNamespace(),poll_interval=0,timeout=None); engine.chain_stages=False
        def absent(**kwargs): raise RuntimeError(f"No official TPF or TESScut coverage available for Sector {kwargs['sector']}.")
        with mock.patch("workflows.tess.tess_residual_localization._download_tpf",side_effect=absent):
            completed,next_request=engine.run_stage(inv,StageRequest("037-target-residual-pixel-recurrence-run","openstar.tess.target-residual-pixel-recurrence.run",{},stage.id),software_id="test",software_version="1")
        rows=completed.stages[-1].result["sectorResults"]
        self.assertEqual([2,65],[x["sector"] for x in rows]); self.assertTrue(all(x["classification"]=="UNAVAILABLE" for x in rows))
        self.assertEqual("038-target-residual-pixel-recurrence-interpret",next_request.id)

    def test_unavailable_is_preserved_and_does_not_affect_other_sector_order(self):
        rows=[{"sector":2,"classification":"UNAVAILABLE"},{"sector":65,"classification":"AMBIGUOUS_OR_BLENDED"}]
        result=interpret_sectors(rows,"target")
        self.assertEqual([2],result["unavailableSectors"])
        self.assertEqual([2,65],[x["sector"] for x in result["sectorResults"]])

    def test_forward_recommendations(self):
        def rows(source,count=3): return [{"sector":i,"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":source} for i in range(count)]
        self.assertEqual("ARCHIVAL_RECURRENCE_INFORMED_TARGET_MECHANISM_MODELING",interpret_sectors(rows("target"),"target")["recommendedNextTest"])
        self.assertEqual("INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",interpret_sectors(rows("other"),"target")["recommendedNextTest"])
        self.assertEqual("ADDITIONAL_SOURCE_LOCALIZATION_DATA",interpret_sectors(rows("target",2),"target")["recommendedNextTest"])


if __name__ == "__main__": unittest.main()
