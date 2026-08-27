import json
import tempfile
import unittest
from pathlib import Path

from openstar_science_runs import ScienceRunCatalog
from openstar_target_projection import TargetProjectionStore, project_investigation


def snapshot(investigation="inv-1", tic="123", status="COMPLETE", stages=None, **extra):
    value = {"investigationID": investigation, "status": status,
             "metadata": {"targetName": "Archive target", "ticID": tic,
                          "gaiaID": "987", "sectors": [1, 28]},
             "stages": stages or [
                 {"id": "detect", "handlerID": "tess.detect.v1", "status": "COMPLETE",
                  "result": {"detectedPeriodDays": 2.5, "classification": "periodic"}},
                 {"id": "final", "handlerID": "generic.finalizer", "status": "COMPLETE",
                  "result": {"summary": "done"}},
             ]}
    value.update(extra)
    return value


class InvestigationProjectionTests(unittest.TestCase):
    def test_projects_identity_period_sectors_and_latest_meaningful_result(self):
        row = project_investigation(snapshot())
        self.assertEqual("tic:123", row["identityKey"])
        self.assertEqual("periodic", row["classification"])
        self.assertEqual(2.5, row["detectedPeriod"])
        self.assertEqual([1, 28], row["primarySectors"])
        self.assertEqual({"completed": 2, "failed": 0, "total": 2}, row["stageCounts"])

    def test_future_stage_and_missing_identity_degrade_gracefully(self):
        row = project_investigation({"investigationID": "future", "status": "ACTIVE",
            "stages": {"new-stage": {"handler": "future.handler", "status": "WAITING"}}})
        self.assertEqual("investigation:future", row["identityKey"])
        self.assertEqual("future.handler", row["stages"][0]["handler"])

    def test_failure_recovery_source_and_companion_evidence(self):
        stages = [{"id": "source", "status": "FAILED", "result": {
            "sourceAttribution": {"source": "catalog-7"},
            "companionNature": "stellar", "recommendedNextTest": "spectroscopy",
            "resolvedPhysicalPeriod": 5.0, "independentSectors": [28]}}]
        row = project_investigation(snapshot(status="RECOVERY_REQUIRED", stages=stages))
        self.assertTrue(row["degraded"]); self.assertTrue(row["recoveryRequired"])
        self.assertEqual({"source": "catalog-7"}, row["sourceAttribution"])
        self.assertEqual("stellar", row["companionNature"])
        self.assertEqual([28], row["independentSectors"])


class TargetProjectionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog.sqlite3"

    def tearDown(self): self.temp.cleanup()

    def add(self, run_root, investigation, payload, kind="tess-autonomous"):
        record = run_root / "investigations" / investigation / "investigation.json"
        record.parent.mkdir(parents=True); record.write_text(json.dumps(payload))
        ScienceRunCatalog(self.catalog).record(kind, run_root, status=payload.get("status", "FINISHED"),
                                                logical_identity=investigation)
        return record

    def test_groups_exact_tic_across_preserved_runs_and_filters_sorts_pages(self):
        first = self.add(self.root / "one", "one", snapshot("one", "123"))
        self.add(self.root / "two", "two", snapshot("two", "123", status="RUNNING"))
        self.add(self.root / "three", "three", snapshot("three", "124"))
        before = first.read_bytes(); store = TargetProjectionStore(self.catalog, ttl=60)
        result = store.list({"q": ["123"], "sort": ["depth"], "pageSize": ["1"]})
        self.assertEqual(1, result["total"]); self.assertEqual(2, result["targets"][0]["runCount"])
        self.assertEqual(before, first.read_bytes())
        detail = store.detail(result["targets"][0]["targetID"])
        self.assertEqual(2, len(detail["runs"]))
        self.assertIsNotNone(store.visuals(detail["targetID"])["stageTimeline"])

    def test_gaia_grouping_only_when_tic_missing_and_missing_identity_is_distinct(self):
        a = snapshot("a", None); b = snapshot("b", None)
        self.add(self.root / "a", "a", a); self.add(self.root / "b", "b", b)
        c = snapshot("c", None); c["metadata"].pop("gaiaID")
        self.add(self.root / "c", "c", c)
        rows = TargetProjectionStore(self.catalog).list({})
        self.assertEqual(2, rows["total"])

    def test_corrupt_record_missing_artifact_and_bulk_sweep_are_isolated(self):
        good = snapshot(); good["artifacts"] = [{"path": "/private/missing/report.pdf"}]
        self.add(self.root / "good", "good", good)
        corrupt = self.root / "bad" / "investigations" / "bad" / "investigation.json"
        corrupt.parent.mkdir(parents=True); corrupt.write_text("{")
        ScienceRunCatalog(self.catalog).record("tess-autonomous", self.root / "bad", logical_identity="bad")
        self.add(self.root / "bulk", "bulk", snapshot("bulk", "999"), kind="tess-sector-sweep")
        store = TargetProjectionStore(self.catalog); result = store.list({})
        self.assertEqual(1, result["total"])
        detail = store.detail(result["targets"][0]["targetID"])
        encoded = json.dumps(detail)
        self.assertNotIn(str(self.root), encoded); self.assertIn("report.pdf", encoded)
        self.assertIsNone(store.detail("../etc/passwd")); self.assertIsNone(store.detail("target_deadbeef"))


if __name__ == "__main__": unittest.main()
