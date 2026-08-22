import tempfile
import unittest
from pathlib import Path

from workflows.tess.tess_hypotheses import analyze, catalog_period_evidence
from workflows.tess.tess_ranked_followup import (
    INCOMPLETE, KNOWN, NOVEL, TessNoveltyScreenStore,
)
from workflows.tess.tess_sector_ranking import TessSectorRanking


def identity(period=None, complete=True):
    return {"catalogCoverageComplete": complete, "tic": {"found": True},
            "vsx": {"matches": ([{"name": "control", "periodDays": period}]
                                  if period is not None else [])},
            "gaiaDR3": {}, "gaiaVariability": {"periodCandidates": []}}


def ranking(count=10):
    return TessSectorRanking(1, {"sector": 1, "rankingPolicyID": "policy",
        "rankingPolicyVersion": "1", "rankedEntries": [
            {"ticID": tic, "rank": tic, "scanInvestigationID": f"scan-{tic}",
             "sourceEvidenceSha256": f"evidence-{tic}", "datasetID": f"dataset-{tic}"}
            for tic in range(1, count + 1)]})


class NoveltyScreeningTests(unittest.TestCase):
    def test_shared_matcher_direct_harmonics_and_tolerance_boundary(self):
        # Stay one representable margin inside the inclusive 3% boundary;
        # binary rounding of the algebraically exact value can land outside.
        boundary = 10.0 / 0.97 - 1e-12
        for published in (5.0, 10.0, 20.0, boundary):
            with self.subTest(published=published):
                catalog = identity(published)
                screened = catalog_period_evidence(catalog, 10.0)
                planned = analyze({"preferredPhysicalPeriodDays": 10.0,
                                   "periodStatus": "RELIABLE", "periodConfidence": "high"}, catalog)
                self.assertEqual(any(item["matches"] for item in screened),
                                 planned["bestCatalogMatch"] is not None)
                self.assertTrue(screened[0]["matches"])
        self.assertFalse(catalog_period_evidence(identity(boundary + .001), 10.0)[0]["matches"])

    def test_known_sequence_does_not_crowd_out_lower_novel_and_query_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def collect(tic):
                calls.append(tic)
                return identity(10.0 if tic <= 6 else None)
            selected, stats = TessNoveltyScreenStore(Path(root) / "screens.json", 1).select(
                ranking(100), 1, 2, set(), collect,
                lambda entry: {"preferredPhysicalPeriodDays": 10.0})
            self.assertEqual([7, 1, 2], [entry["ticID"] for entry, _, _ in selected])
            self.assertEqual(["NOVEL_PRIORITY", "KNOWN_PERIOD_VALIDATION",
                              "KNOWN_PERIOD_VALIDATION"], [basis for _, basis, _ in selected])
            self.assertEqual(list(range(1, 8)), calls)
            self.assertEqual(1, stats["novel_candidates_found"])

    def test_complete_reuse_and_incomplete_retry(self):
        with tempfile.TemporaryDirectory() as root:
            store = TessNoveltyScreenStore(Path(root) / "screens.json", 1)
            calls = []
            responses = [identity(None, False), identity(None, True)]
            def collect(tic): calls.append(tic); return responses.pop(0)
            first, _ = store.select(ranking(1), 1, 0, set(), collect,
                                    lambda entry: {"candidatePeriodDays": 10})
            self.assertEqual("CATALOG_COVERAGE_INCOMPLETE", first[0][1])
            second, _ = store.select(ranking(1), 1, 0, set(), collect,
                                     lambda entry: {"candidatePeriodDays": 10})
            self.assertEqual("NOVEL_PRIORITY", second[0][1])
            third, _ = store.select(ranking(1), 1, 0, set(), collect,
                                    lambda entry: {"candidatePeriodDays": 10})
            self.assertEqual("NOVEL_PRIORITY", third[0][1])
            self.assertEqual([1, 1], calls)

    def test_known_quota_never_extends_screening_past_novel_tranche(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            selected, _ = TessNoveltyScreenStore(Path(root) / "screens.json", 1).select(
                ranking(5000), 4, 2, set(),
                lambda tic: calls.append(tic) or identity(None),
                lambda entry: {"preferredPhysicalPeriodDays": 10})
            self.assertEqual([1, 2, 3, 4], calls)
            self.assertEqual([1, 2, 3, 4], [entry["ticID"] for entry, _, _ in selected])

    def test_only_known_encountered_before_final_novel_is_admitted(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            selected, _ = TessNoveltyScreenStore(Path(root) / "screens.json", 1).select(
                ranking(5000), 3, 2, set(),
                lambda tic: calls.append(tic) or identity(10 if tic == 2 else None),
                lambda entry: {"preferredPhysicalPeriodDays": 10})
            self.assertEqual([1, 2, 3, 4], calls)
            self.assertEqual([1, 3, 4, 2], [entry["ticID"] for entry, _, _ in selected])
            self.assertEqual(1, sum(basis == "KNOWN_PERIOD_VALIDATION"
                                    for _, basis, _ in selected))

    def test_incomplete_is_never_novel_and_admitted_tics_are_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            selected, _ = TessNoveltyScreenStore(Path(root) / "screens.json", 1).select(
                ranking(3), 1, 0, {1},
                lambda tic: calls.append(tic) or identity(None, tic == 3),
                lambda entry: {"preferredPhysicalPeriodDays": 2})
            self.assertEqual([2, 3], calls)
            self.assertEqual("NOVEL_PRIORITY", selected[0][1])
            persisted = (Path(root) / "screens.json").read_text()
            self.assertIn(INCOMPLETE, persisted)
            self.assertIn(NOVEL, persisted)
            self.assertNotIn(KNOWN, persisted)


if __name__ == "__main__":
    unittest.main()
