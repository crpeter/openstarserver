import copy
import importlib.util
import math
import unittest
from types import SimpleNamespace

from openstar_investigation import sha256_json
from workflows.tess.tess_event_depth_accuracy import freeze_photometry
from workflows.tess.tess_external_companion_evidence import (
    FREEZE_HANDLER_ID as EXTERNAL_FREEZE_HANDLER_ID,
    REVIEW_HANDLER_ID,
)
from workflows.tess.tess_event_depth_accuracy import (
    AUDIT_HANDLER_ID, FREEZE_HANDLER_ID as PHOTOMETRY_FREEZE_HANDLER_ID,
)
from workflows.tess.tess_joint_event_phase_model import chronology_from_completed_stages
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
from workflows.tess.tess_joint_event_phase_model import (
    HANDLER_ID, RESULT_VERSION, _select, _sha, fit_joint_event_phase_model,
    validate_model_hash,
)


@unittest.skipUnless(HAS_NUMPY, "NumPy required by joint event/phase model")
class JointEventPhaseModelTests(unittest.TestCase):
    def binary(self, sectors=(1, 2, 3)):
        return {"linearEphemeris": {"coherent": True, "referenceEpoch": 0.,
                    "refinedPeriodDays": 2., "timingSectors": list(sectors)},
                "independentEvidence": {"supportingSectors": list(sectors),
                    "supportingIndependentSectorCount": len(sectors),
                    "classification": "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
                    "independentLinearEphemeris": {"coherent": True}},
                "sectorResults": [{"sector": sector, "role": "INDEPENDENT", "usable": True}
                                  for sector in sectors],
                "catalogAnswerKeyUsed": False}

    def products(self, sectors=(1, 2, 3), depth=.01, eclipse=.002, phase_terms=(.001, -.0005, .0003, .0002),
                 cadences=None, sector_depths=None, outlier=False, integrate=False,
                 secondary_offset=0., orbit_depth_variation=0.):
        answer = []
        for index, sector in enumerate(sectors):
            cadence = (cadences or [.01]*len(sectors))[index]
            times = [sector*20+i*cadence for i in range(int(12/cadence))]
            actual_depth = (sector_depths or [depth]*len(sectors))[index]
            flux = []
            def instantaneous(time):
                phase = (time/2) % 1; distance = abs((phase+.5)%1-.5)
                opposite = abs((phase-.5-secondary_offset+.5)%1-.5)
                primary_shape = max(0., min(1., (.03-distance)/.006))
                secondary_shape = max(0., min(1., (.03-opposite)/.006)); angle = 2*math.pi*phase
                cycle=int(math.floor(time/2+.5))
                event_depth=actual_depth+orbit_depth_variation*((cycle%3)-1)
                return (1 + .0002*index + 2e-6*(time-sum(times)/len(times))
                        - event_depth*primary_shape-eclipse*secondary_shape
                        + phase_terms[0]*math.sin(angle)+phase_terms[1]*math.cos(angle)
                        + phase_terms[2]*math.sin(2*angle)+phase_terms[3]*math.cos(2*angle))
            for i, time in enumerate(times):
                value = (sum(instantaneous(time+cadence*((j+.5)/41-.5)) for j in range(41))/41
                         if integrate else instantaneous(time)) + .00008*math.sin(i*1.618)
                if outlier and i == 177: value += .04
                flux.append(value)
            answer.append({"sector": sector, "time": times, "flux": flux, "cadenceSeconds": cadence*86400,
                "author": "GENERIC", "productIdentity": {"dataURI": f"frozen:{sector}"},
                "sourceProductProvenance": {"selectionRule": "fixture"}, "fluxColumn": "RELATIVE_FLUX",
                "fluxUnits": "dimensionless", "qualityMaskPolicy": "frozen-fixture"})
        return answer

    def inputs(self, **kwargs):
        sectors = kwargs.pop("sectors", (1, 2, 3)); binary = self.binary(sectors); digest = sha256_json(binary)
        chronology = {"verifiedFromCompletedStages": True, "externalEvidenceStageAlreadyCompleted": False,
                      "completedStageHandlerIDs": ["source-review", "photometry-freeze", "depth-audit"]}
        freeze = freeze_photometry(self.products(sectors, **kwargs), list(sectors),
            binary_confirmation_sha256=digest, chronology_proof=copy.deepcopy(chronology))
        audit = {"resultVersion": "openstar.tess-event-depth-attenuation-audit.v1", "status": "COMPLETE",
            "suitableForLaterPrecisionModeling": True, "recommendedNextTest": "JOINT_TRANSIT_ECLIPSE_PHASE_CURVE_MODELING",
            "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False,
            "binaryConfirmationSHA256": digest, "eventDurationDays": .12}
        audit["auditSHA256"] = _sha(audit)
        return freeze, binary, audit, digest, chronology

    def rebind(self, freeze, binary, audit):
        digest=sha256_json(binary)
        freeze["binaryConfirmationSHA256"]=digest
        freeze["freezeSHA256"]=_sha({key:value for key,value in freeze.items() if key!="freezeSHA256"})
        audit["binaryConfirmationSHA256"]=digest
        audit["auditSHA256"]=_sha({key:value for key,value in audit.items() if key!="auditSHA256"})
        return digest

    def fit(self, **kwargs):
        freeze, binary, audit, digest, chronology = self.inputs(**kwargs)
        return fit_joint_event_phase_model(freeze, binary, audit,
            binary_confirmation_sha256=digest, chronology_proof=chronology)

    def test_contract_recovery_exposure_and_boundaries(self):
        freeze, binary, audit, digest, chronology = self.inputs(cadences=[.002, .01, .06], integrate=True)
        result = fit_joint_event_phase_model(freeze, binary, audit,
            binary_confirmation_sha256=digest, chronology_proof=chronology)
        self.assertEqual(RESULT_VERSION, result["resultVersion"]); self.assertEqual(64, len(validate_model_hash(result)))
        self.assertAlmostEqual(.01, result["globalFit"]["midTransitFractionalFluxDeficit"], delta=.002)
        self.assertEqual([.7, .85, 1., 1.15, 1.3], result["modelSpecification"]["durationMultipliers"])
        self.assertIn("EXPOSURE", result["modelSpecification"]["primaryTemplate"])
        self.assertLessEqual(abs(result["globalFit"]["midTransitFractionalFluxDeficit"]-.01),
                             result["globalFit"]["conservativeTransitDepthUncertainty"])
        _, _, _, _, nonintegrated = _select(freeze["sectors"], 2., 0., .12, integrate=False)
        self.assertGreater(abs(-nonintegrated["coefficients"]["transit"]-.01),
                           abs(result["globalFit"]["midTransitFractionalFluxDeficit"]-.01))

    def test_phase_only_does_not_resolve_transit(self):
        result = self.fit(depth=0, eclipse=0, phase_terms=(.004, -.003, .002, .001))
        self.assertFalse(result["precisionEmpiricalTransitDepthResolved"])

    def test_transit_without_eclipse_keeps_component_independent(self):
        result = self.fit(eclipse=0)
        self.assertAlmostEqual(.01, result["globalFit"]["midTransitFractionalFluxDeficit"], delta=.002)
        self.assertEqual("UNRESOLVED", result["globalFit"]["oppositeConjunctionEclipseStatus"])

    def test_offsets_slopes_outliers_and_phase_terms_do_not_bias_depth(self):
        plain = self.fit(phase_terms=(0, 0, 0, 0))["globalFit"]["midTransitFractionalFluxDeficit"]
        complex_result = self.fit(outlier=True)
        self.assertAlmostEqual(plain, complex_result["globalFit"]["midTransitFractionalFluxDeficit"], delta=.001)
        self.assertTrue(complex_result["fitDiagnostics"]["eventCadencesProtectedFromClipping"])

    def test_cross_sector_inconsistency_fails_closed(self):
        result = self.fit(sector_depths=[.003, .01, .025])
        self.assertFalse(result["resolutionGates"]["crossSectorDepthConsistency"])
        self.assertEqual("UNRESOLVED", result["status"])
        self.assertIn("CROSS_SECTOR_DEPTH_INCONSISTENCY", result["unresolvedReasons"])

    def test_within_sector_event_variation_informs_non_circular_consistency(self):
        result=self.fit(orbit_depth_variation=.0005)
        self.assertTrue(result["resolutionGates"]["crossSectorDepthConsistency"])
        for sector in result["perSectorDiagnostics"]:
            self.assertGreater(sector["eventBlockTransitDepthUncertainty"],0)
            self.assertEqual(max(sector["formalTransitDepthUncertainty"],
                                 sector["eventBlockTransitDepthUncertainty"]),
                             sector["consistencyTransitDepthUncertainty"])

    def test_secondary_offset_boundary_does_not_invalidate_transit(self):
        result = self.fit(secondary_offset=.02, eclipse=.004)
        self.assertTrue(result["componentResolutionGates"]["eclipseEvidenceIndependentOfTransit"])
        self.assertFalse(result["componentResolutionGates"]["secondaryPhaseOffsetNotBoundaryPinned"])
        self.assertNotIn("secondaryPhaseOffsetNotBoundaryPinned", result["resolutionGates"])
        self.assertEqual("UNRESOLVED", result["globalFit"]["oppositeConjunctionEclipseStatus"])
        self.assertTrue(result["precisionEmpiricalTransitDepthResolved"])
        self.assertEqual("COMPLETE", result["status"])

    def test_primary_cannot_rescue_two_independent_sectors(self):
        freeze, binary, audit, digest, chronology = self.inputs(sectors=(1, 2, 3))
        binary["independentEvidence"]["supportingSectors"] = [1, 2]
        # Rebind the immutable upstream objects to the deliberately insufficient result.
        digest = sha256_json(binary); freeze["binaryConfirmationSHA256"] = digest
        for row in freeze["sectors"]: row["frozenInputSHA256"] = _sha({k:v for k,v in row.items() if k != "frozenInputSHA256"})
        freeze["freezeSHA256"] = _sha({k:v for k,v in freeze.items() if k != "freezeSHA256"})
        audit["binaryConfirmationSHA256"] = digest; audit["auditSHA256"] = _sha({k:v for k,v in audit.items() if k != "auditSHA256"})
        result = fit_joint_event_phase_model(freeze, binary, audit, binary_confirmation_sha256=digest, chronology_proof=chronology)
        self.assertIn("INDEPENDENT_SUPPORT_COUNT_OR_SECTOR_LIST_INVALID", result["unresolvedReasons"])

    def test_all_upstream_mutations_and_result_mutation_rejected(self):
        freeze, binary, audit, digest, chronology = self.inputs()
        result = fit_joint_event_phase_model(freeze, binary, audit, binary_confirmation_sha256=digest, chronology_proof=chronology)
        result["catalogAnswerKeyUsed"] = True
        with self.assertRaises(ValueError): validate_model_hash(result)
        changed = copy.deepcopy(freeze); changed["sectors"][0]["relativeFluxFloat64"][0] += .1
        unresolved = fit_joint_event_phase_model(changed, binary, audit, binary_confirmation_sha256=digest, chronology_proof=chronology)
        self.assertEqual("UNRESOLVED", unresolved["status"]); validate_model_hash(unresolved)

    def test_every_exact_upstream_gate_fails_closed(self):
        mutations = [
            lambda f,b,a: a.update(resultVersion="wrong"),
            lambda f,b,a: a.update(catalogAnswerKeyUsed=True),
            lambda f,b,a: b.update(catalogAnswerKeyUsed=True),
            lambda f,b,a: b.pop("catalogAnswerKeyUsed"),
            lambda f,b,a: b.update(catalogAnswerKeyUsed=0),
            lambda f,b,a: b["independentEvidence"].update(classification="UNRESOLVED"),
            lambda f,b,a: b["independentEvidence"].update(supportingIndependentSectorCount=True),
            lambda f,b,a: b["independentEvidence"].update(supportingIndependentSectorCount=4),
            lambda f,b,a: b["independentEvidence"]["supportingSectors"].append(3),
            lambda f,b,a: b["independentEvidence"].update(supportingSectors=[True,2,3]),
            lambda f,b,a: b["independentEvidence"].update(supportingSectors=["1",2,3]),
            lambda f,b,a: b["independentEvidence"].update(supportingSectors=[0,2,3]),
            lambda f,b,a: b["independentEvidence"]["independentLinearEphemeris"].update(coherent=False),
            lambda f,b,a: b["linearEphemeris"].update(coherent=False),
            lambda f,b,a: b["sectorResults"][0].update(role="PRIMARY"),
            lambda f,b,a: b["sectorResults"][0].update(usable=False),
            lambda f,b,a: b["sectorResults"].append(copy.deepcopy(b["sectorResults"][0])),
            lambda f,b,a: b["sectorResults"].pop(),
            lambda f,b,a: b["sectorResults"].append({"sector":4,"role":"INDEPENDENT","usable":True}),
            lambda f,b,a: b["sectorResults"][0].update(sector=True),
            lambda f,b,a: f["sectors"][0].update(sector=9),
        ]
        for mutate in mutations:
            freeze, binary, audit, digest, chronology = self.inputs(); mutate(freeze, binary, audit)
            digest=self.rebind(freeze,binary,audit)
            result = fit_joint_event_phase_model(freeze, binary, audit,
                binary_confirmation_sha256=digest, chronology_proof=chronology)
            self.assertEqual("UNRESOLVED", result["status"])
            validate_model_hash(result)

    def test_supporting_sectors_outside_frozen_timing_set_fail_closed(self):
        freeze,binary,audit,digest,chronology=self.inputs()
        binary["independentEvidence"]["supportingSectors"]=[7,8,9]
        binary["sectorResults"]=[{"sector":sector,"role":"INDEPENDENT","usable":True}
                                 for sector in (7,8,9)]
        digest=self.rebind(freeze,binary,audit)
        result=fit_joint_event_phase_model(freeze,binary,audit,
            binary_confirmation_sha256=digest,chronology_proof=chronology)
        self.assertEqual("UNRESOLVED",result["status"])
        self.assertIn("INDEPENDENT_SUPPORT_SECTORS_NOT_IN_FROZEN_TIMING_SET",
                      result["unresolvedReasons"])
        self.assertEqual(result["modelSHA256"],validate_model_hash(result))

    def test_bad_chronology_histories_fail_closed(self):
        for handlers in (["photometry-freeze", "depth-audit"],
                         ["depth-audit", "photometry-freeze", "source-review"],
                         ["source-review", "source-review", "photometry-freeze", "depth-audit"],
                         ["source-review", "photometry-freeze", "depth-audit", "external-companion-evidence-freeze"]):
            freeze,binary,audit,digest,chronology=self.inputs()
            chronology.update(verifiedFromCompletedStages=False,
                              externalEvidenceStageAlreadyCompleted="external-companion-evidence-freeze" in handlers,
                              completedStageHandlerIDs=handlers)
            result=fit_joint_event_phase_model(freeze,binary,audit,binary_confirmation_sha256=digest,
                                               chronology_proof=chronology)
            self.assertIn("MODEL_BEFORE_EXTERNAL_QUERY_CHRONOLOGY_UNPROVEN", result["unresolvedReasons"])

    def test_blindness_chronology_and_claim_boundaries(self):
        result = self.fit()
        self.assertEqual("openstar.tess.joint-event-phase-model.fit", HANDLER_ID)
        for field in ("externalCatalogInformationUsed", "catalogAnswerKeyUsed", "companionRadiusInferred",
                      "fullPhysicalTransitSolutionClaimed", "automaticDiscoveryClaimed"):
            self.assertIs(result[field], False)
        self.assertTrue(result["modelRanBeforeExternalKnownObjectQuery"])


class RegisteredChronologyCalculationTests(unittest.TestCase):
    def stage(self, identifier, handler):
        return SimpleNamespace(id=identifier, handler_id=handler)

    def test_registered_stage_chronology_rejects_malformed_ledgers(self):
        valid=[self.stage("review",REVIEW_HANDLER_ID),
               self.stage("freeze",PHOTOMETRY_FREEZE_HANDLER_ID),
               self.stage("audit",AUDIT_HANDLER_ID)]
        required=[REVIEW_HANDLER_ID,PHOTOMETRY_FREEZE_HANDLER_ID,AUDIT_HANDLER_ID]
        external={EXTERNAL_FREEZE_HANDLER_ID,
                  "openstar.tess.external-companion-evidence.interpret"}
        proof=chronology_from_completed_stages(valid,required,external)
        self.assertTrue(proof["verifiedFromCompletedStages"])
        cases=(valid[1:], [valid[2],valid[1],valid[0]],
               [valid[0],copy.copy(valid[0]),valid[1],valid[2]],
               valid+[self.stage("external",EXTERNAL_FREEZE_HANDLER_ID)])
        for stages in cases:
            proof=chronology_from_completed_stages(stages,required,external)
            self.assertFalse(proof["verifiedFromCompletedStages"])


class PreFitFailClosedRegressionTests(unittest.TestCase):
    def test_out_of_freeze_support_never_reaches_numeric_fit(self):
        fixture=JointEventPhaseModelTests()
        freeze,binary,audit,digest,chronology=fixture.inputs()
        binary["independentEvidence"]["supportingSectors"]=[7,8,9]
        binary["sectorResults"]=[{"sector":sector,"role":"INDEPENDENT","usable":True}
                                 for sector in (7,8,9)]
        digest=fixture.rebind(freeze,binary,audit)
        result=fit_joint_event_phase_model(freeze,binary,audit,
            binary_confirmation_sha256=digest,chronology_proof=chronology)
        self.assertEqual("UNRESOLVED",result["status"])
        self.assertIn("INDEPENDENT_SUPPORT_SECTORS_NOT_IN_FROZEN_TIMING_SET",
                      result["unresolvedReasons"])
        validate_model_hash(result)


if __name__ == "__main__": unittest.main()
