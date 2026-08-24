import unittest
from unittest import mock
import copy, json, tempfile
from pathlib import Path
from dataclasses import replace
from openstar_investigation import (ArtifactReference, InvestigationStage, InvestigationStore,
    StageProvenance, sha256_file, sha256_json)
from workflows.tess.tess_autonomy import repair_obsolete_terminal_wait, WORKFLOW_ID, WORKFLOW_VERSION
import test_tess_target_residual_archival_baseline as archival_tests

try:
    from workflows.tess.tess_target_residual_pixel_recurrence import (
        classify_centroid, freeze_catalog_hypotheses, interpret_sectors,
        acquire_selected_sector, CatalogInfrastructureError, NoPixelCoverageError,
    )
except ModuleNotFoundError as error:
    classify_centroid = None
    IMPORT_ERROR = error


@unittest.skipIf(classify_centroid is None, "optional numerical dependencies unavailable")
class TargetResidualPixelRecurrenceTests(unittest.TestCase):
    def _boundary(self,directory,classification="ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED"):
        base=archival_tests.TessTargetResidualArchivalBaselineTests(methodName="test_constants_preserve_generic_boundary")
        stages=base._lineage(directory)
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

    def test_supported_and_suggestive_boundaries_admit_append_only_idempotently(self):
        for classification in ("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED","ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUGGESTIVE"):
            with self.subTest(classification=classification),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory,classification); before=copy.deepcopy(inv.stages)
                admitted=repair_obsolete_terminal_wait(store,inv)
                self.assertEqual("036-target-residual-pixel-recurrence-prepare",admitted.metadata["controlState"]["selectedExperiment"]["id"])
                self.assertEqual(before,admitted.stages)
                self.assertEqual(admitted,repair_obsolete_terminal_wait(store,admitted))

    def test_nonrecurrence_wrong_recommendation_and_existing_attempt_refuse(self):
        for change in ("classification","baseline","recommendation","existing"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory)
                stages=list(inv.stages); science=stages[-2]
                if change=="classification": science=replace(science,result={**science.result,"classification":"ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_NOT_ESTABLISHED"})
                elif change=="baseline": science=replace(science,result={**science.result,"classification":"ARCHIVAL_TARGET_RESIDUAL_BASELINE_INSUFFICIENT"})
                elif change=="recommendation": science=replace(science,result={**science.result,"recommendedNextTest":"WRONG"})
                else: stages.append(InvestigationStage("036-x","openstar.tess.target-residual-pixel-recurrence.prepare","FAILED",stages[-1].id,{}))
                if change!="existing": stages[-2]=science
                inv=replace(inv,stages=tuple(stages))
                self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_missing_v2017_science_or_finalizer_artifact_refuses(self):
        for index in (-2,-1):
            with self.subTest(stage=index),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory); Path(inv.stages[index].artifacts[0].path).unlink()
                self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_selected_metadata_disagreement_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            store,inv=self._boundary(directory); stages=list(inv.stages); result=copy.deepcopy(stages[-2].result)
            result["selectedFuturePixelFollowupSectors"][0]["originalTimeOriginDays"]=101.
            stages[-2]=replace(stages[-2],result=result); inv=replace(inv,stages=tuple(stages))
            self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def test_altered_run_binding_and_non_supporting_selection_refuse(self):
        for change in ("run","support"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                store,inv=self._boundary(directory); stages=list(inv.stages)
                if change=="run": stages[-3]=replace(stages[-3],result={"datasets":[{"altered":True}]})
                else:
                    result=copy.deepcopy(stages[-2].result); result["sectorEvidence"][0]["supportsHistoricalResidualFamily"]=False
                    stages[-2]=replace(stages[-2],result=result)
                inv=replace(inv,stages=tuple(stages)); self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

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

    def test_cross_sector_resolution_threshold_and_switching(self):
        rows=[{"sector":n,"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":"target"} for n in range(3)]
        result=interpret_sectors(rows,"target")
        self.assertTrue(result["sourceAttributionResolved"])
        self.assertFalse(result["crossSectorPhaseUsed"])
        self.assertFalse(result["historicalResidualDriftExtrapolated"])
        self.assertFalse(interpret_sectors(rows[:2],"target")["sourceAttributionResolved"])
        switched=rows[:2]+[{"sector":3+n,"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":"other"} for n in range(2)]
        self.assertEqual("PIXEL_RECURRENCE_SOURCE_SWITCHING_OR_BLEND",interpret_sectors(switched,"target")["classification"])

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
