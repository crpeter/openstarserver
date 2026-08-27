import json
import tempfile
import unittest
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from openstar_investigation import (ArtifactReference, Investigation,
    InvestigationStage, InvestigationStore, sha256_file)
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import repair_obsolete_terminal_wait
from workflows.tess.tess_target_residual_astrophysical_interpretation import (
    FrozenCatalogAstrophysicalEvidenceProvider, ROTATION, UNRESOLVED,
    interpret_target_residual_astrophysics,
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
            "replicatedMechanisms": ["SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"],
            "replicatedMechanismSupportingSectorIDs": {
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION": [68, 95]}}
        self.attribution = {"classification": "TARGET_RESIDUAL_COMPONENT_DOMINANT",
                            "residualModeOrigin": "TARGET_DOMINANT"}
        self.stellar = {"evaluated": True, "periodDays": 3.604,
            "massMsun": .94, "radiusRsun": .864, "criticalSpeedKmS": 455.0,
            "equatorialSpeedKmS": 12.1, "equatorialToCriticalRatio": .027,
            "status": "not-ruled-out"}
        self.family = {"available": True,
            "representativeRawPeriodDays": 7.546428731,
            "possibleDoubleCycleDays": 15.092857462,
            "physicalCycleResolved": False}
        self.record = {"provider": "literature-fixture", "stableObjectID": "HIP 1113",
            "queryParameters": {"object": "HIP 1113"},
            "citation": {"doi": "10.1051/0004-6361/200913644"},
            "retrievalTimestamp": "2026-01-01T00:00:00+00:00",
            "mechanism": "STARSPOT_ROTATION", "periodRangeDays": [3.58, 3.75],
            "targetAssociation": {"method": "SIMBAD_IDENTIFIER_CROSSMATCH",
                "associated": True, "matchedIdentifier": "HIP 1113"}}

    def decide(self, external):
        return interpret_target_residual_astrophysics(mechanism=self.mechanism,
            target_attribution=self.attribution, stellar_context=self.stellar,
            external_evidence=external, residual_period_days=3.604,
            main_photometric_family=self.family,
            retrieved_at="2026-01-01T00:00:00+00:00")

    def test_historical_rotation_range_promotes_without_resolving_main_family(self):
        result = self.decide({"available": True, "records": [self.record]})
        self.assertEqual(ROTATION, result["classification"])
        self.assertFalse(result["physicalCycleResolved"])
        self.assertFalse(result["mainPhotometricFamily"]["physicalCycleResolved"])
        self.assertEqual(7.546428731,
                         result["mainPhotometricFamily"]["representativeRawPeriodDays"])
        self.assertEqual([68, 95], result["smoothAmplitudeSupportingSectorIDs"])

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

    def test_default_frozen_catalog_provider_is_available_by_construction(self):
        identity = {"retrievedAt": "2025-01-01T00:00:00Z", "vsx": {
            "found": True, "nearest": {"name": "NSV 15055", "type": "TTS/ROT",
                "periodDays": 3.721, "separationArcsec": .15},
            "queryProvenance": {"service": "VizieR", "catalog": "B/vsx/vsx",
                "ticID": 266997586, "radiusArcsec": 10.0}}}
        evidence = FrozenCatalogAstrophysicalEvidenceProvider().fetch({
            "ticID": 266997586, "frozenCatalogIdentity": identity})
        self.assertTrue(evidence["available"])
        self.assertEqual("ROTATIONAL_MODULATION", evidence["records"][0]["mechanism"])
        self.assertTrue(evidence["records"][0]["targetAssociation"]["associated"])
        self.assertEqual(1.0, evidence["records"][0]["targetAssociation"]["thresholdArcsec"])
        self.assertIn("active-region evolution", evidence["periodTolerance"]["basis"])

    def _catalog_decision(self, *, nearest, matches=None, identifiers=()):
        identity = {"retrievedAt": "2025-01-01T00:00:00Z",
            "simbad": {"identifiers": list(identifiers)},
            "vsx": {"found": True, "nearest": nearest,
                "matches": list(matches or [nearest]), "queryProvenance": {
                    "service": "VizieR", "catalog": "B/vsx/vsx",
                    "ticID": 266997586, "radiusArcsec": 10.0}}}
        evidence = FrozenCatalogAstrophysicalEvidenceProvider().fetch({
            "ticID": 266997586, "frozenCatalogIdentity": identity})
        return self.decide(evidence), evidence

    def test_unrelated_rot_neighbor_inside_query_cone_fails_closed(self):
        result, evidence = self._catalog_decision(nearest={"name": "UNRELATED",
            "type": "ROT", "periodDays": 3.61, "separationArcsec": 4.2})
        self.assertFalse(evidence["records"][0]["targetAssociation"]["associated"])
        self.assertEqual(UNRESOLVED, result["classification"])
        self.assertFalse(result["decisionGates"]
                         ["independentRotationEvidenceAssociatedWithTarget"])

    def test_identifier_confirmed_vsx_target_passes(self):
        result, evidence = self._catalog_decision(nearest={"name": "NSV 15055",
            "type": "TTS/ROT", "periodDays": 3.721, "separationArcsec": 2.0},
            identifiers=["HD 987", "NSV 15055"])
        association = evidence["records"][0]["targetAssociation"]
        self.assertEqual("SIMBAD_IDENTIFIER_CROSSMATCH", association["method"])
        self.assertEqual("NSV 15055", association["matchedIdentifier"])
        self.assertEqual(ROTATION, result["classification"])

    def test_nearest_without_association_and_non_rot_nearest_cannot_promote(self):
        unassociated, _ = self._catalog_decision(nearest={"name": "NEARBY",
            "type": "ROT", "periodDays": 3.604, "separationArcsec": None})
        non_rot, _ = self._catalog_decision(nearest={"name": "TARGET",
            "type": "EA", "periodDays": 3.604, "separationArcsec": .1})
        self.assertEqual(UNRESOLVED, unassociated["classification"])
        self.assertEqual(UNRESOLVED, non_rot["classification"])

    def test_multiple_matches_do_not_select_unrelated_rot_source(self):
        nearest = {"name": "TARGET", "type": "EA", "periodDays": 3.604,
                   "separationArcsec": .1}
        unrelated = {"name": "NEIGHBOR", "type": "ROT", "periodDays": 3.604,
                     "separationArcsec": 3.0}
        result, evidence = self._catalog_decision(nearest=nearest,
            matches=[nearest, unrelated], identifiers=["TARGET"])
        self.assertEqual("TARGET", evidence["records"][0]["stableObjectID"])
        self.assertEqual(UNRESOLVED, result["classification"])


class V2014AdmissionTests(unittest.TestCase):
    def test_normal_forward_workflow_routes_smooth_result_directly(self):
        sys.modules.setdefault("numpy", MagicMock())
        from workflows.tess.tess_investigation import (
            target_residual_mechanism_continuation)
        summary = {"classification": "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION",
            "physicalMechanismResolved": False,
            "recommendedNextTest": "ASTROPHYSICAL_MECHANISM_INTERPRETATION",
            "failClosedReasons": [], "replicatedMechanisms": [
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"],
            "replicatedMechanismSupportingSectorIDs": {
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION": [12, 44]}}
        request = target_residual_mechanism_continuation(
            summary, request_id="041-target-residual-mechanism")
        self.assertEqual("042-target-residual-astrophysical-interpretation", request.id)
        self.assertEqual(
            "openstar.tess.target-residual-astrophysical-interpretation.analyze",
            request.handler_id)
        self.assertNotEqual("openstar.tess.finalize", request.handler_id)

    def test_normal_forward_workflow_preserves_other_routes(self):
        sys.modules.setdefault("numpy", MagicMock())
        from workflows.tess.tess_investigation import (
            target_residual_mechanism_continuation)
        corrected = {"adjudicationVersion": "route-independent-all-models-v1",
            "classification": "TARGET_RESIDUAL_MECHANISM_UNRESOLVED",
            "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP",
            "physicalMechanismResolved": False, "failClosedReasons": []}
        predictive = target_residual_mechanism_continuation(
            corrected, request_id="028-target-residual-mechanism")
        self.assertEqual(
            "openstar.tess.target-residual-mechanism-predictive-validation.analyze",
            predictive.handler_id)
        terminal = target_residual_mechanism_continuation({
            "classification": "EPISODIC_TARGET_MODE_ACTIVATION",
            "physicalMechanismResolved": False,
            "recommendedNextTest": "ADDITIONAL_TEMPORAL_BASELINE",
            "failClosedReasons": []}, request_id="028-target-residual-mechanism")
        self.assertEqual("openstar.tess.finalize", terminal.handler_id)
        self.assertEqual({"outputSuffix": "v20.14-intrinsic"}, terminal.parameters)

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
            "crossSectorPhaseUsed": False, "failClosedReasons": [],
            "replicatedMechanisms": ["SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"],
            "replicatedMechanismSupportingSectorIDs": {
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION": [68, 95]}}
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

    def test_exact_failed_stage030_admits_append_only_stage031_idempotently(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); store = InvestigationStore(root / "state")
            boundary = self.make_boundary(root)
            family_value = {"interpretation": "recurrent-raw-period-family",
                "representativeRawPeriodDays": 7.546257528330875,
                "possibleDoubleCycleDays": 15.09251505666175,
                "physicalCycleResolved": False,
                "supportingSectorCount": 2, "supportingSectors": [94, 95]}
            family = InvestigationStage("011-interpret-broad-independent-search",
                "openstar.tess.independent.broad.interpret", "COMPLETE", "010-search",
                {}, result={"harmonicFamily": family_value})
            final_result = {**boundary.stages[1].result,
                "independentBroadVerification": {"harmonicFamily": family_value}}
            final_path = Path(boundary.stages[1].artifacts[0].path)
            final_path.write_text(json.dumps(final_result, sort_keys=True))
            final = replace(boundary.stages[1], result=final_result, artifacts=(
                replace(boundary.stages[1].artifacts[0], sha256=sha256_file(final_path)),))
            failed = InvestigationStage(
                "030-target-residual-astrophysical-interpretation",
                "openstar.tess.target-residual-astrophysical-interpretation.analyze",
                "FAILED", "029-finalize", {}, result=None,
                error="RuntimeError: main recurrent-family artifact verification failed",
                failure_classification="NON_RETRYABLE")
            investigation = replace(boundary, status="FAILED",
                stages=(family, boundary.stages[0], final, failed))
            store.save(investigation)
            family_ledger = store.stage_path_for(investigation.id, family.id)
            family_ledger.parent.mkdir(parents=True)
            family_ledger.write_text(json.dumps(asdict(family)))
            failed_ledger = store.stage_path_for(investigation.id, failed.id)
            failed_ledger.write_text(json.dumps(asdict(failed)))
            failed_bytes = failed_ledger.read_bytes()

            admitted = repair_obsolete_terminal_wait(store, investigation)
            selected = admitted.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("RUNNING", admitted.status)
            self.assertEqual("031-target-residual-astrophysical-interpretation",
                             selected["id"])
            self.assertEqual(failed.id, selected["triggered_by_stage_id"])
            self.assertEqual("FAILED", admitted.stages[-1].status)
            self.assertEqual(failed_bytes, failed_ledger.read_bytes())
            self.assertEqual(admitted, repair_obsolete_terminal_wait(store, admitted))

    def test_failed030_and_family_provenance_mutations_fail_closed(self):
        mutations = {
            "snapshot error": lambda i: replace(i, stages=i.stages[:-1] +
                (replace(i.stages[-1], error="RuntimeError: different"),)),
            "classification": lambda i: replace(i, stages=i.stages[:-1] +
                (replace(i.stages[-1], failure_classification="RETRYABLE"),)),
            "trigger": lambda i: replace(i, stages=i.stages[:-1] +
                (replace(i.stages[-1], triggered_by_stage_id="028-target-residual-mechanism"),)),
            "result": lambda i: replace(i, stages=i.stages[:-1] +
                (replace(i.stages[-1], result={"unexpected": True}),)),
            "handler": lambda i: replace(i, stages=i.stages[:-1] +
                (replace(i.stages[-1], handler_id="openstar.tess.wrong"),)),
            "prior successful interpretation": lambda i: replace(i,
                stages=i.stages[:-1] + (InvestigationStage("030-prior-success",
                    "openstar.tess.target-residual-astrophysical-interpretation.analyze",
                    "COMPLETE", "029-finalize", {}, result={"science": True}),
                    i.stages[-1])),
            "raw period": lambda i: replace(i, stages=(replace(i.stages[0],
                result={"harmonicFamily": {**i.stages[0].result["harmonicFamily"],
                    "representativeRawPeriodDays": 7.0}}),) + i.stages[1:]),
            "double period": lambda i: replace(i, stages=(replace(i.stages[0],
                result={"harmonicFamily": {**i.stages[0].result["harmonicFamily"],
                    "possibleDoubleCycleDays": 14.0}}),) + i.stages[1:]),
            "resolved family": lambda i: replace(i, stages=(replace(i.stages[0],
                result={"harmonicFamily": {**i.stages[0].result["harmonicFamily"],
                    "physicalCycleResolved": True}}),) + i.stages[1:]),
            "stage029 raw period": lambda i: replace(i, stages=i.stages[:2] +
                (replace(i.stages[2], result={**i.stages[2].result,
                    "independentBroadVerification": {"harmonicFamily": {
                        **i.stages[2].result["independentBroadVerification"]["harmonicFamily"],
                        "representativeRawPeriodDays": 7.0}}}),) + i.stages[3:]),
            "stage029 double period": lambda i: replace(i, stages=i.stages[:2] +
                (replace(i.stages[2], result={**i.stages[2].result,
                    "independentBroadVerification": {"harmonicFamily": {
                        **i.stages[2].result["independentBroadVerification"]["harmonicFamily"],
                        "possibleDoubleCycleDays": 14.0}}}),) + i.stages[3:]),
            "stage029 resolved family": lambda i: replace(i, stages=i.stages[:2] +
                (replace(i.stages[2], result={**i.stages[2].result,
                    "independentBroadVerification": {"harmonicFamily": {
                        **i.stages[2].result["independentBroadVerification"]["harmonicFamily"],
                        "physicalCycleResolved": True}}}),) + i.stages[3:]),
            "final sha": lambda i: replace(i, stages=i.stages[:2] +
                (replace(i.stages[2], artifacts=(replace(i.stages[2].artifacts[0],
                    sha256="0" * 64),)),) + i.stages[3:]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = Path(td); store = InvestigationStore(root / "state")
                boundary = self.make_boundary(root)
                family_value = {"representativeRawPeriodDays": 7.546257528330875,
                    "possibleDoubleCycleDays": 15.09251505666175,
                    "physicalCycleResolved": False, "supportingSectors": [94, 95]}
                family = InvestigationStage("011-interpret-broad-independent-search",
                    "openstar.tess.independent.broad.interpret", "COMPLETE",
                    "010-search", {}, result={"harmonicFamily": family_value})
                final_result = {**boundary.stages[1].result,
                    "independentBroadVerification": {"harmonicFamily": family_value}}
                final_path = Path(boundary.stages[1].artifacts[0].path)
                final_path.write_text(json.dumps(final_result, sort_keys=True))
                final = replace(boundary.stages[1], result=final_result, artifacts=(
                    replace(boundary.stages[1].artifacts[0],
                        sha256=sha256_file(final_path)),))
                failed = InvestigationStage(
                    "030-target-residual-astrophysical-interpretation",
                    "openstar.tess.target-residual-astrophysical-interpretation.analyze",
                    "FAILED", "029-finalize", {}, result=None,
                    error="RuntimeError: main recurrent-family artifact verification failed",
                    failure_classification="NON_RETRYABLE")
                original = replace(boundary, status="FAILED",
                    stages=(family, boundary.stages[0], final, failed))
                store.save(original)
                for item in (family, failed):
                    path = store.stage_path_for(original.id, item.id)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(asdict(item)))
                changed = mutate(original)
                if name.startswith("stage029"):
                    changed_final = changed.stages[2]
                    final_path.write_text(json.dumps(changed_final.result, sort_keys=True))
                    changed = replace(changed, stages=changed.stages[:2] +
                        (replace(changed_final, artifacts=(replace(
                            changed_final.artifacts[0], sha256=sha256_file(final_path)),)),)
                        + changed.stages[3:])
                self.assertEqual(changed,
                    repair_obsolete_terminal_wait(store, changed))

    def test_missing_or_tampered_failed030_ledger_fails_closed(self):
        for tamper in (False, True):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as td:
                root = Path(td); store = InvestigationStore(root / "state")
                # Start from the admitted fixture produced by the happy-path test shape.
                boundary = self.make_boundary(root)
                family_value = {"representativeRawPeriodDays": 7.546257528330875,
                    "possibleDoubleCycleDays": 15.09251505666175,
                    "physicalCycleResolved": False, "supportingSectors": [94, 95]}
                family = InvestigationStage("011-interpret-broad-independent-search",
                    "openstar.tess.independent.broad.interpret", "COMPLETE", None, {},
                    result={"harmonicFamily": family_value})
                final_result = {**boundary.stages[1].result,
                    "independentBroadVerification": {"harmonicFamily": family_value}}
                final_path = Path(boundary.stages[1].artifacts[0].path)
                final_path.write_text(json.dumps(final_result))
                final = replace(boundary.stages[1], result=final_result, artifacts=(
                    replace(boundary.stages[1].artifacts[0], sha256=sha256_file(final_path)),))
                failed = InvestigationStage("030-target-residual-astrophysical-interpretation",
                    "openstar.tess.target-residual-astrophysical-interpretation.analyze",
                    "FAILED", "029-finalize", {}, result=None,
                    error="RuntimeError: main recurrent-family artifact verification failed",
                    failure_classification="NON_RETRYABLE")
                investigation = replace(boundary, status="FAILED",
                    stages=(family, boundary.stages[0], final, failed))
                store.save(investigation)
                family_path = store.stage_path_for(investigation.id, family.id)
                family_path.parent.mkdir(parents=True)
                family_path.write_text(json.dumps(asdict(family)))
                if tamper:
                    failed_path = store.stage_path_for(investigation.id, failed.id)
                    failed_path.write_text(json.dumps({**asdict(failed), "stop": True}))
                self.assertEqual(investigation,
                    repair_obsolete_terminal_wait(store, investigation))


class RealBoundaryHandlerIntegrationTests(unittest.TestCase):
    def test_failed030_recovery_executes_031_and_returns_unexecuted_032(self):
        sys.modules.setdefault("numpy", MagicMock())
        from workflows.tess.tess_investigation import build_engine

        with tempfile.TemporaryDirectory() as td:
            root = Path(td); store = InvestigationStore(root / "state")
            family_value = {"interpretation": "recurrent-raw-period-family",
                "representativeRawPeriodDays": 7.546257528330875,
                "possibleDoubleCycleDays": 15.09251505666175,
                "physicalCycleResolved": False, "supportingSectorCount": 2,
                "supportingSectors": [94, 95]}
            identity = {"ticID": 266997586, "retrievedAt": "2025-01-01T00:00:00Z",
                "tic": {"metadata": {"massMsun": .94, "radiusRsun": .864}},
                "vsx": {"found": True, "nearest": {"name": "NSV 15055",
                    "type": "TTS/ROT", "periodDays": 3.721,
                    "separationArcsec": .15}, "queryProvenance": {
                        "service": "VizieR", "catalog": "B/vsx/vsx",
                        "ticID": 266997586, "radiusArcsec": 10.0}}}
            mode = {"classification": "INDEPENDENT_STABLE_MODE",
                "modeCandidate": {"periodDays": 3.604,
                    "frequencyCyclesPerDay": 1 / 3.604,
                    "supportingSectors": [1, 28, 67, 68, 94, 95]},
                "physicalMechanismResolved": False}
            attribution = {"classification": "TARGET_RESIDUAL_COMPONENT_DOMINANT",
                "residualModeOrigin": "TARGET_DOMINANT",
                "physicalMechanismResolved": False}
            intrinsic = {"classification": "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL",
                "physicalMechanismResolved": False,
                "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"}
            mechanism = {"classification": "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION",
                "physicalMechanismResolved": False,
                "recommendedNextTest": "ASTROPHYSICAL_MECHANISM_INTERPRETATION",
                "adjudicationVersion": "route-independent-all-models-v1",
                "crossSectorPhaseUsed": False, "failClosedReasons": [],
                "replicatedMechanisms": ["SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"],
                "replicatedMechanismSupportingSectorIDs": {
                    "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION": [68, 95]}}
            family = InvestigationStage("011-interpret-broad-independent-search",
                "openstar.tess.independent.broad.interpret", "COMPLETE", "010-search",
                {}, result={"harmonicFamily": family_value})
            stages = [
                stage("003-catalog-identity", "openstar.tess.catalog-identity", identity,
                    root / "003-catalog-identity.json"), family,
                stage("018-mode-identification", "openstar.tess.mode-identification.analyze",
                    mode, root / "mode-identification-v20.9.json"),
                stage("026-interpret-multi-source-residual",
                    "openstar.tess.multi-source-residual.interpret", attribution,
                    root / "multi-source-residual-v20.12.json"),
                stage("027-classify-intrinsic-target-residual",
                    "openstar.tess.intrinsic-nonstationary.analyze", intrinsic,
                    root / "intrinsic-nonstationary-v20.13.json"),
                stage("028-target-residual-mechanism",
                    "openstar.tess.target-residual-mechanism.analyze", mechanism,
                    root / "target-residual-mechanism-v20.14.json")]
            conclusion = {"recommendedNextTest":
                "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION",
                "independentBroadVerification": {"harmonicFamily": family_value}}
            stages.append(stage("029-finalize", "openstar.tess.finalize", conclusion,
                root / "conclusion-v20.14-intrinsic.json",
                trigger="028-target-residual-mechanism",
                parameters={"outputSuffix": "v20.14-intrinsic"}, stop=True))
            failed = InvestigationStage(
                "030-target-residual-astrophysical-interpretation",
                "openstar.tess.target-residual-astrophysical-interpretation.analyze",
                "FAILED", "029-finalize", {}, result=None,
                error="RuntimeError: main recurrent-family artifact verification failed",
                failure_classification="NON_RETRYABLE")
            stages.append(failed)
            control = {"branchAssessments": [], "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE"}
            investigation = Investigation("real-shaped-recovery",
                "openstar.workflow.tess-investigation.v1", "20.2", "FAILED",
                "now", "now", {"ticID": 266997586, "controlState": control},
                tuple(stages))
            store.save(investigation)
            for old_stage in stages:
                ledger = store.stage_path_for(investigation.id, old_stage.id)
                ledger.parent.mkdir(parents=True, exist_ok=True)
                ledger.write_text(json.dumps(asdict(old_stage), sort_keys=True))
            prior_files = {store.stage_path_for(investigation.id, item.id):
                store.stage_path_for(investigation.id, item.id).read_bytes()
                for item in stages}
            prior_files.update({Path(ref.path): Path(ref.path).read_bytes()
                for item in stages for ref in item.artifacts})

            admitted = repair_obsolete_terminal_wait(store, investigation)
            selected = admitted.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("031-target-residual-astrophysical-interpretation",
                selected["id"])
            self.assertEqual(
                "openstar.tess.target-residual-astrophysical-interpretation.analyze",
                selected["handler_id"])
            self.assertEqual(failed.id, selected["triggered_by_stage_id"])
            engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            completed, next_request = engine.run_stage(admitted,
                StageRequest(**selected), software_id="test", software_version="1")
            result = completed.stages[-1].result
            self.assertEqual("COMPLETE", completed.stages[-1].status)
            self.assertEqual(ROTATION, result["classification"])
            self.assertEqual(3.604, result["targetResidualPeriodDays"])
            self.assertEqual(7.546257528330875,
                result["mainPhotometricFamily"]["representativeRawPeriodDays"])
            self.assertEqual(15.09251505666175,
                result["mainPhotometricFamily"]["possibleDoubleCycleDays"])
            self.assertFalse(result["mainPhotometricFamily"]["physicalCycleResolved"])
            self.assertEqual("032-main-family-time-domain-recurrence", next_request.id)
            self.assertFalse(any(item.id == "032-finalize" for item in completed.stages))
            self.assertEqual("FAILED", next(item for item in completed.stages
                if item.id == failed.id).status)
            self.assertEqual(prior_files,
                {path: path.read_bytes() for path in prior_files})

    def test_real_schema_boundary_executes_stage030_append_only(self):
        # Numerical modules are not exercised by this coordinator-local stage;
        # isolate their optional import in this minimal test environment.
        sys.modules.setdefault("numpy", MagicMock())
        from workflows.tess.tess_investigation import build_engine

        with tempfile.TemporaryDirectory() as td:
            root = Path(td); store = InvestigationStore(root / "state")
            mode_result = {"classification": "INDEPENDENT_STABLE_MODE",
                "modeCandidate": {"periodDays": 3.604,
                    "frequencyCyclesPerDay": 1 / 3.604,
                    "supportingSectors": [1, 28, 67, 68, 94, 95]},
                "physicalMechanismResolved": False}
            rotation = {"evaluated": True, "periodDays": 3.604,
                "massMsun": .94, "radiusRsun": .864,
                "criticalSpeedKmS": 455.0, "equatorialSpeedKmS": 12.1,
                "equatorialToCriticalRatio": .027, "status": "not-ruled-out"}
            identity = {"ticID": 266997586, "retrievedAt": "2025-01-01T00:00:00Z",
                "tic": {"metadata": {"massMsun": .94, "radiusRsun": .864}},
                "vsx": {"found": True, "nearest": {"name": "NSV 15055",
                    "type": "TTS/ROT", "periodDays": 3.721,
                    "separationArcsec": .15}, "queryProvenance": {
                        "service": "VizieR", "catalog": "B/vsx/vsx",
                        "ticID": 266997586, "radiusArcsec": 10.0}}}
            family = {"harmonicFamily": {
                "representativeRawPeriodDays": 7.546428731,
                "possibleDoubleCycleDays": 15.092857462,
                "physicalCycleResolved": False}}
            attribution = {"classification": "TARGET_RESIDUAL_COMPONENT_DOMINANT",
                "residualModeOrigin": "TARGET_DOMINANT",
                "physicalMechanismResolved": False,
                "recommendedNextTest": "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"}
            intrinsic = {"classification": "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL",
                "physicalMechanismResolved": False,
                "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"}
            mechanism = {"classification": "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION",
                "physicalMechanismResolved": False,
                "recommendedNextTest": "ASTROPHYSICAL_MECHANISM_INTERPRETATION",
                "adjudicationVersion": "route-independent-all-models-v1",
                "crossSectorPhaseUsed": False, "failClosedReasons": [],
                "replicatedMechanisms": ["SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"],
                "replicatedMechanismSupportingSectorIDs": {
                    "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION": [68, 95]}}
            stages = (
                stage("003-catalog-identity", "openstar.tess.catalog-identity", identity,
                      root / "003-catalog-identity.json"),
                stage("004-hypotheses", "openstar.tess.hypotheses",
                      {"rotationSanity": rotation}, root / "004-hypotheses.json"),
                stage("009-harmonic", "openstar.tess.independent.harmonic-family.interpret",
                      family, root / "009-harmonic.json"),
                stage("018-mode-identification", "openstar.tess.mode-identification.analyze",
                      mode_result, root / "mode-identification-v20.9.json"),
                stage("026-interpret-multi-source-residual",
                      "openstar.tess.multi-source-residual.interpret", attribution,
                      root / "multi-source-residual-v20.12.json"),
                stage("027-classify-intrinsic-target-residual",
                      "openstar.tess.intrinsic-nonstationary.analyze", intrinsic,
                      root / "intrinsic-nonstationary-v20.13.json"),
                stage("028-target-residual-mechanism",
                      "openstar.tess.target-residual-mechanism.analyze", mechanism,
                      root / "target-residual-mechanism-v20.14.json"),
                stage("029-finalize", "openstar.tess.finalize", {
                    "recommendedNextTest": "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"},
                    root / "conclusion-v20.14-intrinsic.json",
                    trigger="028-target-residual-mechanism",
                    parameters={"outputSuffix": "v20.14-intrinsic"}, stop=True))
            investigation = Investigation("real-shaped", "openstar.workflow.tess-investigation.v1",
                "20.2", "RUNNING", "now", "now", {"ticID": 266997586}, stages)
            store.save(investigation)
            immutable_bytes = {ref.path: Path(ref.path).read_bytes()
                for old_stage in stages for ref in old_stage.artifacts}

            engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            completed, next_request = engine.run_stage(investigation, StageRequest(
                "030-target-residual-astrophysical-interpretation",
                "openstar.tess.target-residual-astrophysical-interpretation.analyze", {},
                "029-finalize"), software_id="test", software_version="1")
            result = completed.stages[-1].result
            self.assertEqual(ROTATION, result["classification"])
            self.assertEqual(3.604, result["targetResidualPeriodDays"])
            self.assertEqual([68, 95], result["smoothAmplitudeSupportingSectorIDs"])
            self.assertTrue(result["decisionGates"]["rotationPhysicallyAllowed"])
            self.assertTrue(result["decisionGates"]
                            ["independentRotationEvidenceAssociatedWithTarget"])
            self.assertFalse(result["mainPhotometricFamily"]["physicalCycleResolved"])
            self.assertEqual(7.546428731,
                result["mainPhotometricFamily"]["representativeRawPeriodDays"])
            self.assertEqual("031-main-family-time-domain-recurrence", next_request.id)
            self.assertFalse(engine.chain_stages)
            self.assertEqual(immutable_bytes, {path: Path(path).read_bytes()
                                              for path in immutable_bytes})


if __name__ == "__main__":
    unittest.main()
