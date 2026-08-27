import json
import importlib.util
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openstar_investigation import (ArtifactReference, InvestigationStage,
                                    InvestigationStore, sha256_file, sha256_json)
from openstar_workflow import RetryableExecutionError, StageRequest

from workflows.tess.tess_external_companion_evidence import (
    FIELDS, ExternalEvidenceTransientError, acquire_external_evidence,
    canonical_gaia_dr3_id, canonical_tic_id, interpret_external_evidence,
    localization_gate, review_source_attribution,
)


class Response:
    status = 200
    headers = {"Content-Type": "application/json"}
    def __init__(self, value): self.value = value
    def read(self): return self.value
    def __enter__(self): return self
    def __exit__(self, *args): pass


class ExternalCompanionEvidenceTests(unittest.TestCase):
    def setUp(self):
        sectors = [{"sector": 1, "role": "PRIMARY", "usable": True,
                    "classification": "TARGET_CONSISTENT", "matchedCatalogHypothesis": "TIC-42"}]
        sectors += [{"sector": number, "role": "INDEPENDENT", "usable": True,
                     "classification": "TARGET_CONSISTENT", "matchedCatalogHypothesis": "TIC-42"}
                    for number in (2, 3, 4)]
        self.localization = {
            "resultVersion": "openstar.tess-eclipse-event-source-localization.v1",
            "sourceAttributionResolved": True, "classification": "TARGET_CONSISTENT_ECLIPSE_SOURCE",
            "recommendedNextTest": "SOURCE_ATTRIBUTION_REVIEW",
            "pixelDataChangedFrozenEventDefinition": False, "catalogAnswerKeyUsed": False,
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "attributedCatalogHypothesis": "TIC-42", "sectorResults": sectors,
            "frozenCatalog": {"catalogHypotheses": [{"sourceID": "TIC-42", "isTarget": True,
                                                       "ticID": 42, "gaiaDR3SourceID": "Gaia DR3 7"}]},
            "frozenEphemeris": {"refinedPeriodDays": 2.0}, "binaryConfirmationSHA256": "b" * 64,
        }
        self.review = review_source_attribution(self.localization)

    def row(self, mass=10.0, up=1.0, down=-1.0, period=2.0, **changes):
        row = {field: None for field in FIELDS}
        row.update({"pl_name": "published object", "hostname": "published host", "tic_id": "TIC 42",
                    "gaia_dr3_id": "Gaia DR3 7", "default_flag": 1, "soltype": "Published Confirmed",
                    "pl_controv_flag": 0, "rv_flag": 1, "tran_flag": 1, "obm_flag": 0,
                    "pl_orbper": period, "pl_orbpererr1": 0.001, "pl_orbpererr2": -0.001,
                    "pl_bmassj": mass, "pl_bmassjerr1": up, "pl_bmassjerr2": down,
                    "pl_bmassjlim": 0, "pl_bmassprov": "Mass"})
        row.update(changes)
        return row

    def freeze(self, rows):
        raw = json.dumps(rows, separators=(",", ":")).encode()
        return acquire_external_evidence(self.review, opener=lambda *a, **k: Response(raw),
                                         retrieved_at="2026-01-01T00:00:00+00:00")

    def test_every_gate_field_is_exact(self):
        self.assertTrue(localization_gate(self.localization))
        changes = {"resultVersion": "old", "sourceAttributionResolved": False,
                   "classification": "OTHER", "recommendedNextTest": "OTHER",
                   "pixelDataChangedFrozenEventDefinition": True, "catalogAnswerKeyUsed": True,
                   "physicalMechanismResolved": True, "companionNatureResolved": True}
        for key, bad in changes.items():
            value = dict(self.localization); value[key] = bad
            self.assertFalse(localization_gate(value), key)

    def test_review_recomputes_independent_support_and_ignores_primary_and_ambiguous(self):
        self.assertEqual(3, self.review["supportingIndependentSectorCount"])
        value = dict(self.localization); value["sectorResults"] = list(self.localization["sectorResults"])
        value["sectorResults"][3] = {"sector": 4, "role": "INDEPENDENT", "usable": False,
                                     "classification": "AMBIGUOUS"}
        result = review_source_attribution(value)
        self.assertEqual("SOURCE_ATTRIBUTION_REVIEW_FAILED", result["classification"])
        self.assertEqual([4], result["ambiguousIndependentSectors"])

    def test_conflicting_source_fails_closed(self):
        value = json.loads(json.dumps(self.localization))
        value["sectorResults"].append({"sector": 5, "role": "INDEPENDENT", "usable": True,
                                       "classification": "CATALOG_CANDIDATE_CONSISTENT",
                                       "matchedCatalogHypothesis": "OTHER"})
        self.assertEqual("SOURCE_ATTRIBUTION_REVIEW_FAILED", review_source_attribution(value)["classification"])

    def test_off_target_catalog_and_off_catalog_are_distinct(self):
        value = json.loads(json.dumps(self.localization))
        value["classification"] = "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE"
        value["attributedCatalogHypothesis"] = "NEIGHBOR"
        value["frozenCatalog"]["catalogHypotheses"].append(
            {"sourceID": "NEIGHBOR", "isTarget": False, "ticID": 84,
             "gaiaDR3SourceID": 8})
        for sector in value["sectorResults"]:
            sector.update({"classification": "CATALOG_CANDIDATE_CONSISTENT",
                           "matchedCatalogHypothesis": "NEIGHBOR"})
        self.assertEqual("OFF_TARGET_CATALOG_ATTRIBUTION_REVIEW_PASSED", review_source_attribution(value)["classification"])
        value["classification"] = "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE"
        value["attributedCatalogHypothesis"] = "OFF_CATALOG_SKY_CLUSTER"
        for sector in value["sectorResults"]: sector.update({"classification": "OFF_CATALOG", "matchedCatalogHypothesis": None})
        value["frozenCatalog"]["catalogHypotheses"] = [
            {"sourceID": "TIC-42", "isTarget": True, "ticID": 42}]
        result = review_source_attribution(value)
        self.assertEqual("UNCATALOGUED_SOURCE_REQUIRES_FOLLOWUP", result["classification"])
        self.assertIsNone(result["attributedSource"])

    def test_review_role_sector_classification_and_duplicates_fail_closed(self):
        for mutate in ("role", "sector-class", "duplicate"):
            value = json.loads(json.dumps(self.localization))
            if mutate == "role":
                value["frozenCatalog"]["catalogHypotheses"][0]["isTarget"] = False
            elif mutate == "sector-class":
                value["sectorResults"][1]["classification"] = "CATALOG_CANDIDATE_CONSISTENT"
            else:
                value["sectorResults"].append(dict(value["sectorResults"][1]))
            self.assertEqual("SOURCE_ATTRIBUTION_REVIEW_FAILED",
                             review_source_attribution(value)["classification"], mutate)

    def test_identifier_canonicalization_and_malformed_values(self):
        self.assertEqual("TIC 42", canonical_tic_id(42))
        self.assertEqual("TIC 42", canonical_tic_id("TIC 42"))
        self.assertEqual("Gaia DR3 7", canonical_gaia_dr3_id(7))
        self.assertEqual("Gaia DR3 7", canonical_gaia_dr3_id("Gaia DR3 7"))
        for function, values in ((canonical_tic_id, (None, "", -1, "TIC-1", "Gaia DR3 1")),
                                 (canonical_gaia_dr3_id, (None, "", -1, "Gaia DR2 1", "DR3 1"))):
            for value in values:
                with self.assertRaises(ValueError): function(value)

    def test_numeric_frozen_gaia_matches_archive_canonical_form(self):
        value = json.loads(json.dumps(self.localization))
        value["frozenCatalog"]["catalogHypotheses"][0]["gaiaDR3SourceID"] = 7
        frozen = acquire_external_evidence(review_source_attribution(value),
            opener=lambda *a, **k: Response(json.dumps([self.row()]).encode()),
            retrieved_at="2026-01-01T00:00:00+00:00")
        result = interpret_external_evidence(frozen)
        self.assertEqual("PERIOD_MATCHED_PLANETARY_MASS_COMPANION_SUPPORTED",
                         result["classification"])
        self.assertEqual(7, frozen["attributedSourceIdentifiers"]["rawGaiaDR3SourceID"])

    def test_different_and_malformed_archive_gaia_conflict(self):
        for gaia in ("Gaia DR3 8", "Gaia DR2 7", ""):
            result = interpret_external_evidence(self.freeze([self.row(gaia_dr3_id=gaia)]))
            self.assertEqual("CONFLICTING_EXTERNAL_COMPANION_EVIDENCE", result["classification"])

    def test_freeze_exact_identity_hash_and_empty_success(self):
        frozen = self.freeze([])
        self.assertEqual([], frozen["returnedRows"])
        self.assertEqual("TIC 42", frozen["attributedSourceIdentifiers"]["ticID"])
        self.assertEqual(64, len(frozen["rawResponseSHA256"]))
        self.assertIn("default_flag+%3D+1", frozen["exactEncodedQuery"])
        self.assertEqual("NO_MATCHING_EXTERNAL_COMPANION_EVIDENCE",
                         interpret_external_evidence(frozen)["classification"])

    def test_malformed_and_schema_responses_fail_closed(self):
        for raw in (b"not json", b"{}", b"[{}]"):
            with self.assertRaises(ValueError):
                acquire_external_evidence(self.review, opener=lambda *a, raw=raw, **k: Response(raw))

    def test_identity_conflict_and_multiple_period_matches(self):
        conflict = self.row(tic_id="TIC 99")
        self.assertEqual("CONFLICTING_EXTERNAL_COMPANION_EVIDENCE",
                         interpret_external_evidence(self.freeze([conflict]))["classification"])
        self.assertEqual("MULTIPLE_MATCHING_EXTERNAL_COMPANIONS",
                         interpret_external_evidence(self.freeze([self.row(), self.row(mass=11)]))["classification"])

    def test_period_tolerance_boundary_and_mismatch(self):
        self.assertEqual("PERIOD_MATCHED_PLANETARY_MASS_COMPANION_SUPPORTED",
                         interpret_external_evidence(self.freeze([self.row(period=2.011)]))["classification"])
        self.assertEqual("NO_MATCHING_EXTERNAL_COMPANION_EVIDENCE",
                         interpret_external_evidence(self.freeze([self.row(period=2.01101)]))["classification"])

    def test_required_published_mass_evidence(self):
        for changes in ({"rv_flag": 0}, {"pl_controv_flag": 1}, {"pl_bmassj": None},
                        {"pl_bmassjerr1": None}, {"pl_bmassjerr2": None}, {"pl_bmassjlim": 1}):
            result = interpret_external_evidence(self.freeze([self.row(**changes)]))
            self.assertEqual("PERIOD_MATCHED_COMPANION_MASS_UNRESOLVED", result["classification"])

    def test_nonphysical_mass_and_lower_bound_are_unresolved(self):
        for row in (self.row(mass=0), self.row(mass=-1), self.row(mass=1, down=-1),
                    self.row(mass=1, down=-2), self.row(pl_bmassprov="  ")):
            self.assertEqual("PERIOD_MATCHED_COMPANION_MASS_UNRESOLVED",
                             interpret_external_evidence(self.freeze([row]))["classification"])

    def test_mass_regimes_and_boundaries(self):
        cases = [(10, 2, -1, "PLANETARY"), (13, 1, -1, None), (30, 2, -2, "BROWN_DWARF"),
                 (80, 2, -1, None), (82, 2, -2, "STELLAR")]
        for mass, up, down, expected in cases:
            result = interpret_external_evidence(self.freeze([self.row(mass, up, down)]))
            self.assertEqual(expected, result["supportedCompanionMassRegime"])
            self.assertFalse(result["physicalMechanismResolved"])
            self.assertFalse(result["companionNatureResolved"])
            self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_network_failure_is_retryable(self):
        def fail(*args, **kwargs): raise urllib.error.URLError("offline")
        with self.assertRaises(ExternalEvidenceTransientError):
            acquire_external_evidence(self.review, opener=fail)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required by TESS workflow engine")
    def test_workflow_transient_preserves_review_artifact_and_hashes(self):
        from workflows.tess.tess_external_companion_evidence import FREEZE_HANDLER_ID, REVIEW_HANDLER_ID
        from workflows.tess.tess_investigation import build_engine
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            store = InvestigationStore(directory)
            investigation = store.create("external-retry", "test", "1")
            artifact_path = Path(directory) / "source-attribution-review-v1.json"
            artifact_path.write_text("{}\n", encoding="utf-8")
            artifact = ArtifactReference(str(artifact_path.resolve()), sha256_file(artifact_path),
                                         "application/json")
            review = dict(self.review)
            stage = InvestigationStage("023-review", REVIEW_HANDLER_ID, "COMPLETE", "022-localize", {},
                                       result=review, artifacts=(artifact,))
            audit = {"resultVersion": "openstar.tess-event-depth-attenuation-audit.v1",
                     "status": "COMPLETE"}
            audit["auditSHA256"] = sha256_json(audit)
            audit_stage = InvestigationStage("025-audit", "openstar.tess.event-depth-attenuation.audit",
                                              "COMPLETE", "024-depth-freeze", {}, result=audit)
            investigation = type(investigation)(**{**investigation.__dict__, "stages": (stage, audit_stage)})
            store.save(investigation)
            engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            request = StageRequest("024-freeze", FREEZE_HANDLER_ID, {}, "023-review")
            with mock.patch("workflows.tess.tess_investigation.acquire_external_evidence",
                            side_effect=ExternalEvidenceTransientError("rate limited")), \
                    self.assertRaises(RetryableExecutionError):
                engine.run_stage(investigation, request, software_id="test", software_version="1")
            failed = store.load(investigation.id).stages[-1]
            self.assertEqual("TRANSIENT_INFRASTRUCTURE", failed.failure_classification)
            self.assertEqual(sha256_json(review), failed.provenance.input_hashes["sourceAttributionReview"])
            self.assertEqual(review["sourceLocalizationSHA256"],
                             failed.provenance.input_hashes["sourceLocalization"])
            self.assertEqual((artifact,), failed.artifacts)
            self.assertEqual("external-companion-evidence-freeze", failed.result["operation"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required by TESS workflow engine")
    def test_registered_lifecycle_queries_once_and_restart_interprets_frozen_response(self):
        from workflows.tess.tess_external_companion_evidence import (
            FREEZE_HANDLER_ID, INTERPRET_HANDLER_ID, REVIEW_HANDLER_ID)
        from workflows.tess.tess_eclipse_event_localization import HANDLER_ID
        from workflows.tess.tess_event_depth_accuracy import (
            AUDIT_HANDLER_ID as DEPTH_AUDIT_HANDLER_ID,
            FREEZE_HANDLER_ID as DEPTH_FREEZE_HANDLER_ID,
        )
        from workflows.tess.tess_investigation import build_engine
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            store = InvestigationStore(directory)
            investigation = store.create("external-lifecycle", "test", "1")
            binary = {"linearEphemeris": {"coherent": True, "referenceEpoch": 1.0,
                                           "refinedPeriodDays": 2.0, "timingSectors": [1, 2, 3]},
                      "sectorResults": [{"usable": True, "dutyCycle": .05} for _ in range(3)],
                      "catalogAnswerKeyUsed": False}
            prepared_stage = InvestigationStage("001-prepare-target", "openstar.tess.prepare-target",
                "COMPLETE", None, {}, result={"ticID": 42, "datasetID": "synthetic"})
            binary_stage = InvestigationStage("021-binary", "openstar.tess.binary-confirmation.analyze",
                                                "COMPLETE", None, {}, result=binary)
            localization_stage = InvestigationStage("022-localize", HANDLER_ID, "COMPLETE", None, {},
                                                    result=self.localization)
            investigation = type(investigation)(**{**investigation.__dict__,
                                                    "stages": (prepared_stage, binary_stage, localization_stage)})
            store.save(investigation)
            engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            investigation, next_request = engine.run_stage(
                investigation, StageRequest("023-review", REVIEW_HANDLER_ID, {}, "022-localize"),
                software_id="test", software_version="1")
            self.assertEqual(DEPTH_FREEZE_HANDLER_ID, next_request.handler_id)
            photometry = {"resultVersion": "openstar.tess-event-depth-photometry-freeze.v1",
                          "status": "FROZEN", "sectors": [], "freezeSHA256": "f" * 64}
            with mock.patch("workflows.tess.tess_investigation.acquire_full_precision_photometry",
                            return_value=photometry) as mast:
                investigation, next_request = engine.run_stage(
                    investigation, next_request, software_id="test", software_version="1")
            mast.assert_called_once()
            self.assertEqual(DEPTH_AUDIT_HANDLER_ID, next_request.handler_id)
            audit = {"resultVersion": "openstar.tess-event-depth-attenuation-audit.v1",
                     "status": "COMPLETE", "externalCatalogInformationUsed": False,
                     "catalogAnswerKeyUsed": False}
            audit["auditSHA256"] = sha256_json(audit)
            with mock.patch("workflows.tess.tess_investigation.audit_depth_attenuation",
                            return_value=audit) as auditor:
                investigation, next_request = engine.run_stage(
                    investigation, next_request, software_id="test", software_version="1")
            self.assertEqual(photometry, auditor.call_args.args[0])
            self.assertEqual(FREEZE_HANDLER_ID, next_request.handler_id)
            frozen = self.freeze([self.row()])
            with mock.patch("workflows.tess.tess_investigation.acquire_external_evidence",
                            return_value=frozen) as archive:
                investigation, next_request = engine.run_stage(
                    investigation, next_request, software_id="test", software_version="1")
            archive.assert_called_once()
            external_freeze_stage = investigation.stages[-1]
            self.assertEqual(audit["auditSHA256"],
                             external_freeze_stage.provenance.input_hashes["eventDepthAttenuationAudit"])
            self.assertEqual(INTERPRET_HANDLER_ID, next_request.handler_id)
            # Restart at the durable completed freeze boundary interprets the persisted
            # response and cannot call acquisition again.
            with mock.patch("workflows.tess.tess_investigation.acquire_external_evidence") as archive:
                investigation, next_request = engine.run_stage(
                    investigation, next_request, software_id="test", software_version="1")
            archive.assert_not_called()
            self.assertEqual("openstar.tess.final-companion-evidence-synthesis",
                             next_request.handler_id)
            result = investigation.stages[-1].result
            self.assertEqual("PERIOD_MATCHED_PLANETARY_MASS_COMPANION_SUPPORTED",
                             result["classification"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required by TESS workflow engine")
    def test_registered_review_stops_off_catalog_and_failed_attribution_before_query(self):
        from workflows.tess.tess_external_companion_evidence import REVIEW_HANDLER_ID
        from workflows.tess.tess_eclipse_event_localization import HANDLER_ID
        from workflows.tess.tess_investigation import build_engine
        cases = []
        off_catalog = json.loads(json.dumps(self.localization))
        off_catalog.update({"classification": "CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE",
                            "attributedCatalogHypothesis": "OFF_CATALOG_SKY_CLUSTER"})
        off_catalog["frozenCatalog"]["catalogHypotheses"][0].pop("gaiaDR3SourceID", None)
        for sector in off_catalog["sectorResults"]:
            sector.update({"classification": "OFF_CATALOG", "matchedCatalogHypothesis": None})
        cases.append(off_catalog)
        failed = json.loads(json.dumps(self.localization))
        failed["sectorResults"][1]["classification"] = "CATALOG_CANDIDATE_CONSISTENT"
        cases.append(failed)
        for index, localization in enumerate(cases):
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                store = InvestigationStore(directory)
                investigation = store.create(f"review-stop-{index}", "test", "1")
                stage = InvestigationStage("022-localize", HANDLER_ID, "COMPLETE", None, {},
                                           result=localization)
                investigation = type(investigation)(**{**investigation.__dict__, "stages": (stage,)})
                store.save(investigation)
                engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None)
                engine.chain_stages = False
                with mock.patch("workflows.tess.tess_investigation.acquire_external_evidence") as archive:
                    _, next_request = engine.run_stage(investigation,
                        StageRequest("023-review", REVIEW_HANDLER_ID, {}, "022-localize"),
                        software_id="test", software_version="1")
                archive.assert_not_called()
                self.assertEqual("openstar.tess.finalize", next_request.handler_id)


if __name__ == "__main__": unittest.main()
