import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_investigation import (ArtifactReference, Investigation,
    InvestigationStage, InvestigationStore, sha256_file)
from workflows.tess.tess_autonomy import repair_obsolete_terminal_wait
from workflows.tess.tess_target_residual_astrophysical_interpretation import (
    ROTATION, UNRESOLVED, interpret_target_residual_astrophysics,
    newest_authoritative_recommendation)


def stage(stage_id, handler, result, artifact_path, *, trigger=None, parameters=None,
          stop=False):
    artifact_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return InvestigationStage(stage_id, handler, "COMPLETE", trigger,
        parameters or {}, result=result, stop=stop,
        artifacts=(ArtifactReference(str(artifact_path), sha256_file(artifact_path),
                                     "application/json"),))


class AstrophysicalDecisionTests(unittest.TestCase):
    def setUp(self):
        self.mechanism = {"classification": "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION",
            "targetResidualPeriodDays": 3.604,
            "replicatedSupportingSectors": [68, 95]}
        self.attribution = {"classification": "TARGET_RESIDUAL_COMPONENT_DOMINANT",
                            "residualModeOrigin": "TARGET_DOMINANT"}
        self.stellar = {"rotationPhysicallyAllowed": True}
        self.record = {"provider": "literature-fixture", "stableObjectID": "HIP 1113",
            "queryParameters": {"object": "HIP 1113"},
            "citation": {"doi": "10.1051/0004-6361/200913644"},
            "mechanism": "STARSPOT_ROTATION", "periodRangeDays": [3.58, 3.75]}

    def decide(self, external):
        return interpret_target_residual_astrophysics(mechanism=self.mechanism,
            target_attribution=self.attribution, stellar_context=self.stellar,
            external_evidence=external, retrieved_at="2026-01-01T00:00:00+00:00")

    def test_historical_rotation_range_promotes_without_resolving_main_family(self):
        result = self.decide({"available": True, "records": [self.record]})
        self.assertEqual(ROTATION, result["classification"])
        self.assertFalse(result["physicalCycleResolved"])
        self.assertFalse(result["mainPhotometricFamily"]["physicalCycleResolved"])

    def test_unavailable_or_rot_label_alone_fails_closed(self):
        self.assertEqual(UNRESOLVED, self.decide({"available": False,
                                                "records": []})["classification"])
        label = dict(self.record); label.pop("periodRangeDays")
        self.assertEqual(UNRESOLVED, self.decide({"available": True,
                                                "records": [label]})["classification"])

    def test_exact_catalog_mismatch_does_not_veto_historical_range(self):
        external = {"available": True, "exactCatalogPeriodMatch": False,
                    "records": [self.record]}
        self.assertEqual(ROTATION, self.decide(external)["classification"])

    def test_provider_output_is_deterministic_and_provenance_required(self):
        external = {"available": True, "records": [self.record]}
        self.assertEqual(self.decide(external), self.decide(external))
        incomplete = dict(self.record); incomplete.pop("citation")
        self.assertEqual(UNRESOLVED, self.decide({"available": True,
            "records": [incomplete]})["classification"])


class V2014AdmissionTests(unittest.TestCase):
    def test_future_finalizer_uses_newest_science_recommendation(self):
        stages = tuple(InvestigationStage(f"{number:03d}-science", handler,
            "COMPLETE", None, {}, result={"recommendedNextTest": recommendation})
            for number, handler, recommendation in (
                (26, "openstar.tess.multi-source-residual.interpret",
                 "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"),
                (27, "openstar.tess.intrinsic-nonstationary.analyze",
                 "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"),
                (28, "openstar.tess.target-residual-mechanism.analyze",
                 "ASTROPHYSICAL_MECHANISM_INTERPRETATION")))
        self.assertEqual("ASTROPHYSICAL_MECHANISM_INTERPRETATION",
                         newest_authoritative_recommendation(stages))

    def make_boundary(self, root):
        science_result = {"classification": "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION",
            "physicalMechanismResolved": False,
            "recommendedNextTest": "ASTROPHYSICAL_MECHANISM_INTERPRETATION",
            "adjudicationVersion": "route-independent-all-models-v1",
            "crossSectorPhaseUsed": False, "failClosedReasons": []}
        science = stage("028-target-residual-mechanism",
            "openstar.tess.target-residual-mechanism.analyze", science_result,
            root / "target-residual-mechanism-v20.14.json")
        conclusion = {"recommendedNextTest":
                      "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"}
        final = stage("029-finalize", "openstar.tess.finalize", conclusion,
            root / "conclusion-v20.14-intrinsic.json", trigger=science.id,
            parameters={"outputSuffix": "v20.14-intrinsic"}, stop=True)
        control = {"branchAssessments": [], "selectedExperiment": None,
                   "schedulerAction": "INVESTIGATION_COMPLETE"}
        return Investigation("i", "openstar.workflow.tess-investigation.v1", "20.2",
            "COMPLETE", "now", "now", {"controlState": control}, (science, final))

    def test_exact_boundary_admits_once_and_tolerates_stale_finalizer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); store = InvestigationStore(root / "state")
            investigation = self.make_boundary(root); store.save(investigation)
            admitted = repair_obsolete_terminal_wait(store, investigation)
            selected = admitted.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("030-target-residual-astrophysical-interpretation",
                             selected["id"])
            self.assertEqual(admitted, repair_obsolete_terminal_wait(store, admitted))

    def test_malformed_sha_classification_resolution_or_existing_attempt_rejected(self):
        mutations = [
            lambda i: replace(i, stages=(replace(i.stages[0], result={
                **i.stages[0].result, "classification": "WRONG"}), i.stages[1])),
            lambda i: replace(i, stages=(replace(i.stages[0], result={
                **i.stages[0].result, "physicalMechanismResolved": True}), i.stages[1])),
            lambda i: replace(i, stages=(replace(i.stages[0], artifacts=(
                replace(i.stages[0].artifacts[0], sha256="0" * 64),)), i.stages[1])),
            lambda i: replace(i, stages=i.stages + (InvestigationStage("030-x",
                "openstar.tess.target-residual-astrophysical-interpretation.analyze",
                "FAILED", "029-finalize", {}),)),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as td:
                root = Path(td); store = InvestigationStore(root / "state")
                investigation = mutate(self.make_boundary(root)); store.save(investigation)
                self.assertEqual(investigation,
                    repair_obsolete_terminal_wait(store, investigation))


if __name__ == "__main__":
    unittest.main()
