import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from openstar_dashboard import DashboardApplication, make_server
from openstar_investigation import ArtifactReference, InvestigationStage, InvestigationStore
from openstar_science_runs import ScienceRunCatalog
from openstar_target_projection import TargetProjectionStore


class QuietCoordinator:
    def observation(self):
        return {"health": {"ok": True}, "nodes": [], "projects": [],
                "contributions": {"currentSession": {}, "allTime": {}}}


class ProductionTargetProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.catalog = self.base / "science-runs.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def persist(self, root_name, investigation_id, metadata, results, *, status="COMPLETE", artifacts=()):
        root = self.base / root_name
        store = InvestigationStore(root / "investigations")
        investigation = store.create(investigation_id, "openstar.tess.autonomous",
                                     "20.31", metadata)
        for index, result in enumerate(results):
            stage_id = f"{index + 1:03d}-science"
            running = InvestigationStage(stage_id, f"openstar.tess.handler.{index + 1}",
                "RUNNING", None, {}, started_at=f"2026-01-0{index + 1}T00:00:00+00:00")
            investigation = store.append_running_stage(investigation, running)
            terminal = replace(running, status="COMPLETE",
                completed_at=f"2026-01-0{index + 1}T01:00:00+00:00", result=result,
                artifacts=artifacts if index == len(results) - 1 else ())
            investigation = store.complete_current_stage(investigation, terminal)
        investigation = replace(investigation, status=status)
        store.save(investigation)
        ScienceRunCatalog(self.catalog).record("tess-autonomous", root, status=status,
                                               logical_identity=investigation_id)
        return store, investigation

    def test_canonical_resolved_record_latest_stage_and_stats(self):
        old = {"classification": "OLD_CLASS", "recommendedNextTest": "OLD_TEST",
               "sourceAttributionResolved": False, "companionNatureResolved": False,
               "detectedPeriodDays": 2.4}
        latest = {"classification": "ECLIPSING_BINARY", "recommendedNextTest": "SPECTROSCOPY",
            "claim": "SOURCE_LOCALIZED_BINARY", "sourceAttribution": "TARGET",
            "sourceAttributionResolved": True, "companionNature": "STELLAR",
            "companionNatureResolved": True, "physicalMechanism": "ECLIPSE",
            "physicalMechanismResolved": True, "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": 4.8, "independentSectors": [28],
            "differenceImageSummary": {"sector": 28, "pixels": [[1, 2], [3, 4]]},
            "measuredCentroid": {"row": 4.2, "column": 7.1}}
        store, investigation = self.persist("resolved", "resolved-1", {
            "ticID": 1001, "gaiaDR3SourceID": 9001, "targetName": "Resolved target",
            "sector": 1, "raDeg": 12.5, "decDeg": -4.5}, [old, latest],
            artifacts=(ArtifactReference(str(self.base / "private" / "report.pdf"), "a" * 64, "application/pdf"),))
        before_snapshot = store.path_for(investigation.id).read_bytes()
        before_ledgers = {path: path.read_bytes() for path in
                          store.directory_for(investigation.id).glob("stages/*.json")}
        projection = TargetProjectionStore(self.catalog, ttl=60)
        listing = projection.list({})
        self.assertEqual(1, listing["total"])
        self.assertEqual(0, listing["stats"]["unresolvedTargets"])
        self.assertEqual(1, listing["stats"]["sourceLocalizedTargets"])
        self.assertEqual(1, listing["stats"]["companionNatureResolvedTargets"])
        self.assertEqual(1, listing["stats"]["physicalMechanismResolvedTargets"])
        row = projection.detail(listing["targets"][0]["targetID"])
        self.assertEqual("openstar.tess.autonomous", row["workflow"])
        self.assertEqual("20.31", row["workflowVersion"])
        self.assertEqual("openstar.tess.handler.2", row["stages"][1]["handler"])
        self.assertIsInstance(row["createdAt"], float)
        self.assertIsInstance(row["stages"][1]["completedAt"], float)
        self.assertEqual("ECLIPSING_BINARY", row["classification"])
        self.assertEqual("SPECTROSCOPY", row["recommendedNextTest"])
        self.assertEqual(4.8, row["resolvedPhysicalPeriod"])
        self.assertEqual("report.pdf", row["artifacts"][0]["path"])
        self.assertNotIn(str(self.base), json.dumps(row))
        visuals = projection.visuals(row["targetID"])
        self.assertEqual("available", visuals["differenceImage"]["status"])
        self.assertEqual("available", visuals["centroid"]["status"])
        self.assertEqual(before_snapshot, store.path_for(investigation.id).read_bytes())
        self.assertEqual(before_ledgers, {path: path.read_bytes() for path in
                         store.directory_for(investigation.id).glob("stages/*.json")})

    def test_explicit_unresolved_and_neighbor_identity_is_not_adopted(self):
        self.persist("unresolved", "unresolved-1", {"targetName": "Anonymous target"}, [{
            "classification": "UNRESOLVED", "sourceAttribution": "candidate nearby",
            "sourceAttributionResolved": False, "companionNatureResolved": False,
            "physicalMechanismResolved": False, "physicalCycleResolved": False,
            "catalogCandidates": [{"ticID": 777, "gaiaDR3SourceID": 888,
                                   "periodDays": 99.0}]}])
        listing = TargetProjectionStore(self.catalog).list({})
        row = listing["targets"][0]
        self.assertEqual("investigation:unresolved-1", row["identityKey"])
        self.assertIsNone(row["ticID"]); self.assertIsNone(row["detectedPeriod"])
        self.assertEqual(1, listing["stats"]["unresolvedTargets"])
        self.assertEqual(0, listing["stats"]["sourceLocalizedTargets"])

    def test_nested_measurement_does_not_override_authoritative_interpretation(self):
        interpretation = {"classification": "UNRESOLVED",
            "sourceAttributionResolved": False,
            "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
            "currentClaim": {"claim": "CANDIDATE_PERIOD", "rationale": ["Needs another sector."]}}
        raw_run = {"sectorResults": [{"sector": 28, "classification": "TARGET_SUPPORTED",
            "differenceImage": [[float(x + y) for x in range(40)] for y in range(40)],
            "measuredPixelCentroid": {"x": 4.2, "y": 7.1},
            "centroidX": 4.2, "centroidY": 7.1,
            "centroidSky": {"ra": 11.1, "dec": -2.2},
            "centroidUncertaintyPixels": 0.3,
            "differenceImagePeakSNR": 9.5,
            "catalogDistances": {"target": 0.2, "neighbor": 3.8},
            "distancesPixels": [0.2, 3.8],
            "matchedCatalogHypothesis": "TARGET"}]}
        self.persist("nested", "nested", {"ticID": 71}, [interpretation, raw_run])
        projection = TargetProjectionStore(self.catalog)
        row = projection.detail(projection.list({})["targets"][0]["targetID"])
        self.assertEqual("UNRESOLVED", row["classification"])
        self.assertEqual("ADDITIONAL_SOURCE_LOCALIZATION_DATA", row["recommendedNextTest"])
        self.assertEqual("CANDIDATE_PERIOD", row["currentClaim"])
        self.assertEqual(["Needs another sector."], row["claimRationale"])
        self.assertIsInstance(row["currentClaim"], str)
        self.assertNotIn("[object Object]", json.dumps(row))
        visual = projection.visuals(row["targetID"])
        self.assertEqual("available", visual["differenceImage"]["status"])
        self.assertEqual(32, len(visual["differenceImage"]["data"]["values"]))
        self.assertTrue(visual["differenceImage"]["data"]["truncated"])
        self.assertEqual(9.5, visual["differenceImage"]["data"]["peakSNR"])
        self.assertEqual("available", visual["centroid"]["status"])
        self.assertEqual(0.3, visual["centroid"]["data"]["centroidUncertaintyPixels"])
        self.assertEqual("available", visual["sourceDistances"]["status"])
        self.assertIn("catalogDistances", visual["sourceDistances"]["data"])
        self.assertIn("distancesPixels", visual["sourceDistances"]["data"])

    def test_later_top_level_decision_updates_classification(self):
        self.persist("updated", "updated", {"ticID": 72}, [
            {"classification": "UNRESOLVED"},
            {"sectorResults": [{"classification": "TARGET_SUPPORTED"}]},
            {"classification": "SOURCE_LOCALIZED", "sourceAttributionResolved": True}])
        row = TargetProjectionStore(self.catalog).list({})["targets"][0]
        self.assertEqual("SOURCE_LOCALIZED", row["classification"])

    def test_answer_key_true_is_aggregated_across_preserved_history(self):
        self.persist("answer-old", "answer-old", {"ticID": 73}, [
            {"classification": "KNOWN", "catalogAnswerKeyUsed": True}])
        self.persist("answer-new", "answer-new", {"ticID": 73}, [
            {"classification": "REPLAY", "catalogAnswerKeyUsed": False,
             "answerKeyUsed": False}])
        projection = TargetProjectionStore(self.catalog)
        row = projection.detail(projection.list({})["targets"][0]["targetID"])
        self.assertTrue(row["answerKeyUsed"])
        self.assertTrue(any(run["answerKeyUsed"] for run in row["runs"]))

    def test_partially_resolved_target_remains_unresolved(self):
        self.persist("partial", "partial", {"ticID": 74}, [{
            "sourceAttributionResolved": True, "companionNatureResolved": False,
            "physicalMechanismResolved": False, "classification": "SOURCE_LOCALIZED"}])
        projection = TargetProjectionStore(self.catalog)
        stats = projection.list({})["stats"]
        self.assertEqual(1, stats["sourceLocalizedTargets"])
        self.assertEqual(1, stats["unresolvedTargets"])
        self.assertEqual(1, projection.list({"resolution": ["unresolved"]})["total"])
        self.assertEqual(0, projection.list({"resolution": ["resolved"]})["total"])

    def test_exact_tic_and_gaia_grouping_with_mixed_timestamps(self):
        self.persist("tic-a", "tic-a", {"ticID": 42}, [{"classification": "A"}])
        store, investigation = self.persist("tic-b", "tic-b", {"ticID": 42}, [{"classification": "B"}])
        payload = json.loads(store.path_for(investigation.id).read_text())
        payload["updated_at"] = 1893456000
        store.path_for(investigation.id).write_text(json.dumps(payload))
        self.persist("gaia-a", "gaia-a", {"gaiaDR3SourceID": 55}, [{"classification": "G"}])
        listing = TargetProjectionStore(self.catalog).list({"sort": ["updated"]})
        self.assertEqual(2, listing["total"])
        tic = next(row for row in listing["targets"] if row["ticID"] == "42")
        self.assertEqual(2, tic["runCount"])

    def test_corrupt_isolation_browser_safe_paths_and_unavailable_visuals(self):
        self.persist("healthy", "healthy", {"ticID": 9}, [{"classification": "PERIODIC"}])
        root = self.base / "corrupt"
        record = root / "investigations" / "bad" / "investigation.json"
        record.parent.mkdir(parents=True); record.write_text("{")
        ScienceRunCatalog(self.catalog).record("tess-autonomous", root, logical_identity="bad")
        projection = TargetProjectionStore(self.catalog)
        listing = projection.list({}); self.assertEqual(1, listing["total"])
        detail = projection.detail(listing["targets"][0]["targetID"])
        self.assertNotIn(str(self.base), json.dumps(detail))
        visuals = projection.visuals(detail["targetID"])
        self.assertEqual({"status": "unavailable", "reason": "not_recorded_for_run"},
                         visuals["differenceImage"])


class TargetProjectionHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        base = Path(cls.temporary.name); catalog = base / "catalog.sqlite3"
        store = InvestigationStore(base / "run" / "investigations")
        investigation = store.create("http-target", "workflow.real", "3", {"ticID": 321})
        running = InvestigationStage("science", "handler.real", "RUNNING", None, {},
                                     started_at="2026-01-01T00:00:00Z")
        investigation = store.append_running_stage(investigation, running)
        investigation = store.complete_current_stage(investigation, replace(running,
            status="COMPLETE", completed_at="2026-01-01T01:00:00Z",
            result={"classification": "REAL", "sourceAttributionResolved": True,
                    "recommendedNextTest": "FOLLOW_UP", "sectorAgreement": {"sectors": [1, 2]}}))
        ScienceRunCatalog(catalog).record("tess-autonomous", base / "run",
                                         status="COMPLETE", logical_identity="http-target")
        cls.server = make_server("127.0.0.1", 0, DashboardApplication(
            QuietCoordinator(), science_run_catalog=catalog, contribution_ledger=None))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.temporary.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as response:
            self.assertEqual("no-store", response.headers["Cache-Control"])
            return response.status, json.loads(response.read())

    def test_list_detail_and_visuals(self):
        status, listing = self.get("/api/dashboard/targets?pageSize=1&sort=updated")
        self.assertEqual(200, status); self.assertEqual(1, listing["total"])
        target_id = listing["targets"][0]["targetID"]
        self.assertEqual("handler.real", self.get(f"/api/dashboard/targets/{target_id}")[1]["stages"][0]["handler"])
        self.assertEqual("available", self.get(f"/api/dashboard/targets/{target_id}/visuals")[1]["independentSectorAgreement"]["status"])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/api/dashboard/targets/%2e%2e%2fetc%2fpasswd")
        self.assertEqual(404, caught.exception.code)


if __name__ == "__main__": unittest.main()
