import copy
import importlib.util
import math
import unittest

from openstar_investigation import sha256_json
from workflows.tess.tess_event_depth_accuracy import freeze_photometry
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    from workflows.tess.tess_joint_event_phase_model import (
        HANDLER_ID, RESULT_VERSION, _sha, fit_joint_event_phase_model,
        validate_model_hash,
    )


@unittest.skipUnless(HAS_NUMPY, "NumPy required by joint event/phase model")
class JointEventPhaseModelTests(unittest.TestCase):
    def binary(self, sectors=(1, 2, 3)):
        return {"linearEphemeris": {"coherent": True, "referenceEpoch": 0.,
                    "refinedPeriodDays": 2., "timingSectors": list(sectors)},
                "independentEvidence": {"supportingSectors": list(sectors),
                    "supportingIndependentSectorCount": len(sectors)},
                "catalogAnswerKeyUsed": False}

    def products(self, sectors=(1, 2, 3), depth=.01, eclipse=.002, phase_terms=(.001, -.0005, .0003, .0002),
                 cadences=None, sector_depths=None, outlier=False):
        answer = []
        for index, sector in enumerate(sectors):
            cadence = (cadences or [.01]*len(sectors))[index]
            times = [sector*20+i*cadence for i in range(int(12/cadence))]
            actual_depth = (sector_depths or [depth]*len(sectors))[index]
            flux = []
            for i, time in enumerate(times):
                phase = (time/2) % 1; distance = abs((phase+.5)%1-.5); opposite = abs((phase-.5+.5)%1-.5)
                primary_shape = max(0., min(1., (.03-distance)/.006))
                secondary_shape = max(0., min(1., (.03-opposite)/.006))
                angle = 2*math.pi*phase
                value = (1 + .0002*index + 2e-6*(time-sum(times)/len(times))
                    - actual_depth*primary_shape-eclipse*secondary_shape
                    + phase_terms[0]*math.sin(angle)+phase_terms[1]*math.cos(angle)
                    + phase_terms[2]*math.sin(2*angle)+phase_terms[3]*math.cos(2*angle)
                    + .00008*math.sin(i*1.618))
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
            binary_confirmation_sha256=digest, chronology_proof=chronology)
        audit = {"resultVersion": "openstar.tess-event-depth-attenuation-audit.v1", "status": "COMPLETE",
            "suitableForLaterPrecisionModeling": True, "recommendedNextTest": "JOINT_TRANSIT_ECLIPSE_PHASE_CURVE_MODELING",
            "externalCatalogInformationUsed": False, "catalogAnswerKeyUsed": False,
            "binaryConfirmationSHA256": digest, "eventDurationDays": .12}
        audit["auditSHA256"] = _sha(audit)
        return freeze, binary, audit, digest, chronology

    def fit(self, **kwargs):
        freeze, binary, audit, digest, chronology = self.inputs(**kwargs)
        return fit_joint_event_phase_model(freeze, binary, audit,
            binary_confirmation_sha256=digest, chronology_proof=chronology)

    def test_contract_recovery_exposure_and_boundaries(self):
        result = self.fit(cadences=[.005, .01, .02])
        self.assertEqual(RESULT_VERSION, result["resultVersion"]); self.assertEqual(64, len(validate_model_hash(result)))
        self.assertAlmostEqual(.01, result["globalFit"]["midTransitFractionalFluxDeficit"], delta=.002)
        self.assertEqual([.7, .85, 1., 1.15, 1.3], result["modelSpecification"]["durationMultipliers"])
        self.assertIn("EXPOSURE", result["modelSpecification"]["primaryTemplate"])

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

    def test_primary_cannot_rescue_two_independent_sectors(self):
        freeze, binary, audit, digest, chronology = self.inputs(sectors=(1, 2, 3))
        binary["independentEvidence"]["supportingSectors"] = [1, 2]
        # Rebind the immutable upstream objects to the deliberately insufficient result.
        digest = sha256_json(binary); freeze["binaryConfirmationSHA256"] = digest
        for row in freeze["sectors"]: row["frozenInputSHA256"] = _sha({k:v for k,v in row.items() if k != "frozenInputSHA256"})
        freeze["freezeSHA256"] = _sha({k:v for k,v in freeze.items() if k != "freezeSHA256"})
        audit["binaryConfirmationSHA256"] = digest; audit["auditSHA256"] = _sha({k:v for k,v in audit.items() if k != "auditSHA256"})
        result = fit_joint_event_phase_model(freeze, binary, audit, binary_confirmation_sha256=digest, chronology_proof=chronology)
        self.assertIn("INSUFFICIENT_INDEPENDENT_SUPPORTING_SECTORS", result["unresolvedReasons"])

    def test_all_upstream_mutations_and_result_mutation_rejected(self):
        freeze, binary, audit, digest, chronology = self.inputs()
        result = fit_joint_event_phase_model(freeze, binary, audit, binary_confirmation_sha256=digest, chronology_proof=chronology)
        result["catalogAnswerKeyUsed"] = True
        with self.assertRaises(ValueError): validate_model_hash(result)
        changed = copy.deepcopy(freeze); changed["sectors"][0]["relativeFluxFloat64"][0] += .1
        unresolved = fit_joint_event_phase_model(changed, binary, audit, binary_confirmation_sha256=digest, chronology_proof=chronology)
        self.assertEqual("UNRESOLVED", unresolved["status"]); validate_model_hash(unresolved)

    def test_blindness_chronology_and_claim_boundaries(self):
        result = self.fit()
        self.assertEqual("openstar.tess.joint-event-phase-model.fit", HANDLER_ID)
        for field in ("externalCatalogInformationUsed", "catalogAnswerKeyUsed", "companionRadiusInferred",
                      "fullPhysicalTransitSolutionClaimed", "automaticDiscoveryClaimed"):
            self.assertIs(result[field], False)
        self.assertTrue(result["modelRanBeforeExternalKnownObjectQuery"])


if __name__ == "__main__": unittest.main()
