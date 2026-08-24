import math
import unittest
import json
import tempfile
from pathlib import Path
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
from types import SimpleNamespace
from dataclasses import replace
from openstar_investigation import (ArtifactReference, InvestigationStage, InvestigationStore, StageProvenance, sha256_file, sha256_json)
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION, repair_obsolete_terminal_wait
from workflows.tess.tess_target_residual_archival_baseline import (
    FREQUENCIES_PER_WORK_UNIT, HARMONIC_ORDERS, TOTAL_FREQUENCIES,
    adjudicate_sector, adjudicate_target, frozen_search_grid,
    prewhiten_established_family, previously_consumed_tess_sectors,
    MAX_ARCHIVAL_BASELINE_SECTORS, WORKLOAD_ID, CONSUMED_SECTOR_SCHEMAS, verify_frozen_science_lineage,
)

class TessTargetResidualArchivalBaselineTests(unittest.TestCase):
    def test_explicit_consumption_excludes_primary_and_known_materializations(self):
        stages=[SimpleNamespace(id="000-prepare",handler_id="openstar.tess.prepare-target",status="COMPLETE",result={"sector":1,"unrelatedNumber":99}),
            SimpleNamespace(id="010-independent",handler_id="openstar.tess.independent.prepare",status="COMPLETE",result={"preparedSectors":[{"sector":4},{"sector":7}]}),
            SimpleNamespace(id="noise",handler_id="unknown",status="COMPLETE",result={"sector":88})]
        consumed=previously_consumed_tess_sectors(stages)
        self.assertEqual([1,4,7],list(consumed)); self.assertNotIn(99,consumed); self.assertNotIn(88,consumed)
        self.assertEqual("000-prepare",consumed[1][0]["stageID"])

    def test_every_explicit_consumption_schema_is_honored(self):
        stages=[]
        for sector,(handler,paths) in enumerate(CONSUMED_SECTOR_SCHEMAS.items(),start=1):
            result={}; cursor=result
            for key in paths[0][:-1]: cursor=cursor.setdefault(key,{})
            cursor[paths[0][-1]] = [{"sector":sector}] if paths[0][-1].endswith("Sectors") or paths[0][-1]=="preparedSeries" else sector
            stages.append(SimpleNamespace(id=f"s{sector}",handler_id=handler,status="COMPLETE",result=result))
        self.assertEqual(list(range(1,len(stages)+1)),list(previously_consumed_tess_sectors(stages)))

    def test_frozen_grid_is_exact_and_targeted(self):
        grid=frozen_search_grid(.1)
        self.assertAlmostEqual(.08,grid["minimumFrequency"]); self.assertAlmostEqual(.12,grid["maximumFrequency"])
        self.assertEqual(TOTAL_FREQUENCIES,8192); self.assertEqual(FREQUENCIES_PER_WORK_UNIT,2048)

    def _candidate(self, sector=1, frequency=.1, origin=0, **updates):
        value={"sector":sector,"candidateFrequency":frequency,"candidateFrequencyConfidenceInterval":{"lower":frequency-.002,"upper":frequency+.002},
            "periodStatus":"RELIABLE","periodConfidence":"high","baselineDays":27,"cycleCoverage":2.7,"boundaryHit":False,"originalTimeOriginDays":origin}
        value.update(updates); return value

    def test_recurrence_requires_all_preregistered_rules(self):
        supported=adjudicate_sector(self._candidate(),(.09,.11)); self.assertTrue(supported["supportsHistoricalResidualFamily"])
        for changed in ({"boundaryHit":True},{"periodStatus":"UNRELIABLE"},{"cycleCoverage":1.49},{"candidateFrequencyConfidenceInterval":None},{"candidateFrequency":.12}):
            self.assertFalse(adjudicate_sector(self._candidate(**changed),(.09,.11))["supportsHistoricalResidualFamily"])
        limited=adjudicate_sector(self._candidate(candidateFrequencyConfidenceInterval={"lower":.05,"upper":.15}),(.09,.11))
        self.assertEqual("RESOLUTION_LIMITED",limited["recurrenceClassification"])

    def test_target_classifications_and_invariants(self):
        rows=[adjudicate_sector(self._candidate(i,.1,(i-1)*400),(.09,.11)) for i in range(1,4)]
        result=adjudicate_target(rows); self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED",result["classification"])
        for key in ("physicalMechanismResolved","sourceAttributionResolved","crossSectorPhaseUsed","historicalResidualDriftExtrapolated"): self.assertFalse(result[key])
        self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_BASELINE_INSUFFICIENT",adjudicate_target(rows[:2])["classification"])
        with self.assertRaises(ValueError): adjudicate_target(rows+[rows[0]])

    def test_boundary_and_low_coverage_are_ineligible_and_do_not_change_denominator(self):
        support = adjudicate_sector(self._candidate(1), (.09, .11))
        boundary = adjudicate_sector(self._candidate(2, boundaryHit=True), (.09, .11))
        short = adjudicate_sector(self._candidate(3, cycleCoverage=1.49), (.09, .11))
        self.assertEqual("INELIGIBLE", boundary["recurrenceClassification"])
        self.assertEqual("INELIGIBLE", short["recurrenceClassification"])
        result = adjudicate_target([support, boundary, short])
        self.assertEqual(1, result["eligibleSectorCount"])
        self.assertEqual(1, result["supportingSectorCount"])
        self.assertEqual(adjudicate_target([support])["classification"], result["classification"])

    def test_counts_are_mutually_exclusive(self):
        support = adjudicate_sector(self._candidate(1), (.09, .11))
        limited = adjudicate_sector(self._candidate(2,
            candidateFrequencyConfidenceInterval={"lower": .05, "upper": .15}), (.09, .11))
        nonsupport = adjudicate_sector(self._candidate(3, periodStatus="UNRELIABLE"), (.09, .11))
        ineligible = adjudicate_sector(self._candidate(4, boundaryHit=True), (.09, .11))
        result = adjudicate_target([support, limited, nonsupport, ineligible])
        self.assertEqual((3, 1, 1, 1), (result["eligibleSectorCount"],
            result["supportingSectorCount"], result["resolutionLimitedSectorCount"],
            result["nonSupportingSectorCount"]))

    def test_target_outcomes_use_actual_epoch_span(self):
        support = [adjudicate_sector(self._candidate(i, origin=(i-1)*100), (.09,.11)) for i in range(1,4)]
        self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUGGESTIVE",
            adjudicate_target(support)["classification"])
        support[-1]["originalTimeOriginDays"] = 301
        selected = adjudicate_target(support)
        self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED", selected["classification"])
        self.assertEqual([1, 3, 2], [x["sector"] for x in selected["selectedFuturePixelFollowupSectors"]])

    def test_not_established_and_suggestive(self):
        rows = [adjudicate_sector(self._candidate(i, periodStatus="UNRELIABLE"), (.09,.11)) for i in range(1,4)]
        self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_NOT_ESTABLISHED", adjudicate_target(rows)["classification"])
        rows[:2] = [adjudicate_sector(self._candidate(i, origin=i*400), (.09,.11)) for i in range(1,3)]
        self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUGGESTIVE", adjudicate_target(rows)["classification"])

    def test_constants_preserve_generic_boundary(self):
        self.assertEqual("openstar.lomb-scargle.v1", WORKLOAD_ID)
        self.assertEqual(64, MAX_ARCHIVAL_BASELINE_SECTORS)
        self.assertEqual((1, 2), HARMONIC_ORDERS)

    @unittest.skipIf(np is None or not hasattr(np, "linspace") or not hasattr(np, "linalg"), "NumPy is required for Float64 preprocessing regression")
    def test_prewhitening_float64_order_and_signal_survival(self):
        times = np.linspace(1000., 1027., 4096, dtype=np.float64)
        main = 8*np.sin(2*np.pi*.2*times)
        residual_signal = .4*np.sin(2*np.pi*.1*times)
        output, provenance = prewhiten_established_family(times, main+residual_signal, .2)
        self.assertEqual(np.float32, output.dtype)
        self.assertEqual(.2, provenance["frozenEstablishedPhysicalFrequency"])
        self.assertEqual([1,2], provenance["harmonicOrders"])
        self.assertTrue(provenance["linearTrendIncluded"])
        self.assertFalse(provenance["historicalResidualDriftExtrapolated"])
        local=times-times[0]
        self.assertGreater(abs(np.dot(output,np.sin(2*np.pi*.1*local))),
                           20*abs(np.dot(output,np.sin(2*np.pi*.2*local))))

    def _finalized_boundary(self, directory):
        store = InvestigationStore(Path(directory)/"investigations")
        inv = store.create("archival-boundary", WORKFLOW_ID, WORKFLOW_VERSION)
        predictive = {"classification":"TARGET_RESIDUAL_MECHANISM_PREDICTIVE_VALIDATION_UNRESOLVED",
            "recommendedNextTest":"ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION",
            "physicalMechanismResolved":False}
        conclusion = {"targetResidualMechanismPredictiveValidation":predictive,
            "recommendedNextTest":"ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION"}
        def artifact(name, value):
            path=Path(directory)/name; path.write_text(json.dumps(value)+"\n",encoding="utf-8")
            return ArtifactReference(str(path),sha256_file(path),"application/json")
        science=InvestigationStage("030-target-residual-mechanism-predictive-validation",
            "openstar.tess.target-residual-mechanism-predictive-validation.analyze","COMPLETE",None,{},
            result=predictive,artifacts=(artifact("target-residual-mechanism-predictive-validation-v20.16.json",predictive),))
        final=InvestigationStage("031-finalize","openstar.tess.finalize","COMPLETE",science.id,
            {"outputSuffix":"v20.16-target-residual-predictive-validation"},result=conclusion,
            artifacts=(artifact("conclusion-v20.16-target-residual-predictive-validation.json",conclusion),),stop=True)
        inv=replace(inv,stages=(science,final))
        inv=store.set_control_state(inv,status="COMPLETE",control_state={"branchAssessments":[],"selectedExperiment":None,"schedulerAction":"INVESTIGATION_COMPLETE"})
        return store,inv

    def test_exact_post_031_boundary_reopens_once_and_altered_control_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            store,inv=self._finalized_boundary(directory)
            before=tuple(inv.stages)
            reopened=repair_obsolete_terminal_wait(store,inv)
            self.assertEqual("032-target-residual-archival-baseline-prepare",
                reopened.metadata["controlState"]["selectedExperiment"]["id"])
            self.assertEqual(before,reopened.stages)
            self.assertEqual(reopened,repair_obsolete_terminal_wait(store,reopened))
        with tempfile.TemporaryDirectory() as directory:
            store,inv=self._finalized_boundary(directory)
            changed=store.set_control_state(inv,status="COMPLETE",control_state={"branchAssessments":[],"selectedExperiment":{},"schedulerAction":"INVESTIGATION_COMPLETE"})
            self.assertEqual(changed,repair_obsolete_terminal_wait(store,changed))

    def test_altered_finalizer_parameters_leave_history_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            store,inv=self._finalized_boundary(directory)
            bad=inv.stages[-1]
            bad=InvestigationStage(**{**bad.__dict__,"parameters":{"outputSuffix":"wrong"}})
            inv=replace(inv,stages=inv.stages[:-1]+(bad,))
            self.assertEqual(inv,repair_obsolete_terminal_wait(store,inv))

    def _lineage(self, directory, direct=False):
        root=Path(directory)
        def artifact(name,value):
            path=root/name; path.write_text(json.dumps(value)+"\n",encoding="utf-8")
            return ArtifactReference(str(path),sha256_file(path),"application/json")
        def stage(sid,handler,result,name,hashes=None,trigger=None,parameters=None):
            return InvestigationStage(sid,handler,"COMPLETE",trigger,parameters or {},result=result,
                artifacts=(artifact(name,result),),provenance=StageProvenance("test","1",hashes or {}))
        source_project=root/"source-project.json"; source_project.write_text('{}\n')
        source_dataset=root/"source-dataset.json"; source_dataset.write_text('{}\n')
        primary_manifest=artifact("primary.json",{"id":"primary"})
        target_result={"sourceProjectPath":str(source_project),"datasetPath":str(source_dataset),"projectPath":primary_manifest.path,"ticID":1,"sourceDatasetEntry":{"id":"d"}}
        target=InvestigationStage("001-prepare-target","openstar.tess.prepare-target","COMPLETE",None,{},result=target_result,
            artifacts=(primary_manifest,),provenance=StageProvenance("test","1",{"sourceProjectManifest":sha256_file(source_project),"sourceDataset":sha256_file(source_dataset)}))
        morphology_result={"resolvedPhysicalPeriodDays":10.0}
        morphology=stage("010-morphology","openstar.tess.morphology.analyze",morphology_result,"morphology-v20.4.json")
        prep_result={"preparedSeries":[]}; prep=stage("020-prep","openstar.tess.multi-source-residual.prepare",prep_result,"prepared-dataset.json")
        interpretation_result={"componentSummaries":[]}; interpretation=stage("021-interpret","openstar.tess.multi-source-residual.interpret",interpretation_result,"multi-source-residual-v20.12.json",{"preparation":sha256_json(prep_result)})
        v13_result={"inputProvenance":{"v20.12PreparationResultHash":sha256_json(prep_result),"v20.12InterpretationResultHash":sha256_json(interpretation_result)}}
        v13=stage("022-v13","openstar.tess.intrinsic-nonstationary.analyze",v13_result,"intrinsic-nonstationary-v20.31.json",{"v20.12Preparation":sha256_json(prep_result),"v20.12Interpretation":sha256_json(interpretation_result)})
        v14_result={"adjudicationVersion":"route-independent-all-models-v1"}
        v14=stage("028-v14","openstar.tess.target-residual-mechanism.analyze",v14_result,"target-residual-mechanism-v20.14.json",{"v20.12Preparation":sha256_json(prep_result),"v20.12Interpretation":sha256_json(interpretation_result),"v20.13Result":sha256_json(v13_result)})
        v14_sha=v14.artifacts[0].sha256
        v15_result={"inputProvenance":{"frozenV20.14ResultHash":sha256_json(v14_result),"frozenV20.14ArtifactSHA256":v14_sha}}
        v15=stage("029-v15","openstar.tess.target-residual-mechanism-adjudication.analyze",v15_result,"target-residual-mechanism-adjudication-v20.15.json",{"v20.14Result":sha256_json(v14_result),"v20.14Artifact":v14_sha})
        adjudication=v14 if direct else v15
        v16_result={"adjudicationSource":{"stageID":adjudication.id,"handlerID":adjudication.handler_id,"resultHash":sha256_json(adjudication.result),"artifactSHA256":adjudication.artifacts[0].sha256},
            "frozenV20.14ResultHash":sha256_json(v14_result),"frozenV20.14ArtifactSHA256":v14_sha,
            "frozenV20.13ResultHash":sha256_json(v13_result),"frozenV20.13ArtifactSHA256":v13.artifacts[0].sha256,
            "classification":"TARGET_RESIDUAL_MECHANISM_PREDICTIVE_VALIDATION_UNRESOLVED","physicalMechanismResolved":False,"recommendedNextTest":"ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION"}
        v16=stage("030-target-residual-mechanism-predictive-validation","openstar.tess.target-residual-mechanism-predictive-validation.analyze",v16_result,"target-residual-mechanism-predictive-validation-v20.16.json",{"adjudicationResult":sha256_json(adjudication.result),"adjudicationArtifact":adjudication.artifacts[0].sha256,"v20.14Result":sha256_json(v14_result),"v20.14Artifact":v14_sha,"v20.13Result":sha256_json(v13_result),"v20.13Artifact":v13.artifacts[0].sha256})
        conclusion={"targetResidualMechanismPredictiveValidation":v16_result,"recommendedNextTest":"ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION"}
        final=stage("031-finalize","openstar.tess.finalize",conclusion,"conclusion-v20.16-target-residual-predictive-validation.json",trigger=v16.id,parameters={"outputSuffix":"v20.16-target-residual-predictive-validation"})
        return [target,morphology,prep,interpretation,v13,v14]+([] if direct else [v15])+[v16,final]

    def test_connected_historical_v2015_and_direct_v2014_lineages_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual("029-v15",verify_frozen_science_lineage(self._lineage(directory))["adjudication"].id)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual("028-v14",verify_frozen_science_lineage(self._lineage(directory,True))["adjudication"].id)

    def test_each_broken_lineage_link_fails_closed(self):
        for index,key in ((4,"v20.12Preparation"),(5,"v20.13Result"),(6,"v20.14Result")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                stages=self._lineage(directory)
                target=stages[index]; hashes=dict(target.provenance.input_hashes); hashes[key]="broken"
                stages[index]=replace(target,provenance=replace(target.provenance,input_hashes=hashes))
                with self.assertRaises(RuntimeError): verify_frozen_science_lineage(stages)

    def test_modified_artifact_or_final_conclusion_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            stages=self._lineage(directory); Path(stages[4].artifacts[0].path).write_text('{}')
            with self.assertRaises(RuntimeError): verify_frozen_science_lineage(stages)
        with tempfile.TemporaryDirectory() as directory:
            stages=self._lineage(directory); stages[-1]=replace(stages[-1],result={"recommendedNextTest":"wrong"})
            with self.assertRaises(RuntimeError): verify_frozen_science_lineage(stages)

if __name__ == "__main__": unittest.main()
