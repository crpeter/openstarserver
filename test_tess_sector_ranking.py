import json
import tempfile
import unittest
from pathlib import Path

from openstar_investigation import InvestigationStore, sha256_file
from run_openstar_tess_sector_ranking import run_tess_sector_ranking
from test_tess_sector_sweep import FakeCoordinator, FakeProvider, Prepared, product
from workflows.tess import tess_sector_scan
from workflows.tess.tess_autonomy import TessInvestigationTargetSource
from workflows.tess.tess_sector_archive import TessSectorInventoryStore
from workflows.tess.tess_sector_ranking import aggregate_tess_sector_ranking
from run_openstar_tess_sector_sweep import run_tess_sector_sweep


class SectorRankingTests(unittest.TestCase):
    def setUp(self):
        self.original_preprocessing = tess_sector_scan.read_and_prepare_tess_light_curve
        tess_sector_scan.read_and_prepare_tess_light_curve = lambda path: Prepared()

    def tearDown(self):
        tess_sector_scan.read_and_prepare_tess_light_curve = self.original_preprocessing

    def _sweep(self, root, tics=(1, 2, 3)):
        run_tess_sector_sweep(7, "unused", root,
            provider=FakeProvider([product(tic) for tic in tics]), coordinator=FakeCoordinator())

    def _ranking(self, root):
        inventory = TessSectorInventoryStore(Path(root) / "tess-sector-7-inventory.json").load()
        return aggregate_tess_sector_ranking(inventory, InvestigationStore(Path(root) / "investigations"))

    def _change_evidence(self, root, tic, **updates):
        store = InvestigationStore(Path(root) / "investigations")
        investigation = store.load(f"tess-sector-scan-7-tic-{tic}")
        stage = investigation.stages[-1]
        result = {**stage.result, **updates}
        evidence_path = Path(stage.artifacts[0].path)
        evidence_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        from dataclasses import replace
        artifact = replace(stage.artifacts[0], sha256=sha256_file(evidence_path))
        stage = replace(stage, result=result, artifacts=(artifact,))
        store.save(replace(investigation, stages=investigation.stages[:-1] + (stage,)))

    @staticmethod
    def _tree_bytes(root):
        root = Path(root)
        return {path.relative_to(root): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    def test_policy_tie_breakers_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp)
            self._change_evidence(tmp, 1, periodConfidence="medium", foldCoherence=.99, bestPower=.99)
            self._change_evidence(tmp, 2, periodConfidence="high", foldCoherence=.5, bestPower=.7)
            self._change_evidence(tmp, 3, periodConfidence="high", foldCoherence=.5, bestPower=.8)
            first = self._ranking(tmp)
            second = self._ranking(tmp)
            self.assertEqual(first.content, second.content)
            self.assertEqual([3, 2, 1], [x["ticID"] for x in first.content["rankedEntries"]])
            self.assertEqual(["high", .5, .8, 3], first.content["rankedEntries"][0]["rankingKey"])

    def test_incomplete_coverage_and_missing_are_durable_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp, (1, 2))
            self._change_evidence(tmp, 1, coverageComplete=False)
            investigation = Path(tmp) / "investigations" / "tess-sector-scan-7-tic-2"
            import shutil; shutil.rmtree(investigation)
            ranking = self._ranking(tmp).content
            self.assertFalse(ranking["rankingComplete"])
            self.assertEqual(1, ranking["missingCount"])
            self.assertEqual({"COMPLETE_NO_RELIABLE_PERIOD", "MISSING"},
                             {x["state"] for x in ranking["excludedEntries"]})

    def test_nonfinite_and_bad_hash_exclude_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp, (1,))
            self._change_evidence(tmp, 1, bestPower=float("nan"))
            ranking = self._ranking(tmp).content
            self.assertEqual(0, ranking["eligibleRankedCount"])
            self.assertIn("INVALID_BEST_POWER", ranking["excludedEntries"][0]["exclusionReasons"])

    def test_promotion_is_explicit_reusable_and_target_source_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sweep(tmp, (1, 2))
            shallow_before = {p: p.read_bytes() for p in Path(tmp).glob("investigations/*/investigation.json")}
            self.assertEqual(0, run_tess_sector_ranking(7, tmp, promote_top=1))
            promotion = Path(tmp) / "tess-sector-7-promoted-top-1.json"
            raw = json.loads(promotion.read_text())
            self.assertEqual(1, len(raw["datasets"]))
            dataset = raw["datasets"][0]
            self.assertEqual(1, dataset["autonomousPriority"])
            self.assertNotEqual(dataset["investigationID"], dataset["sourceScanInvestigationID"])
            self.assertEqual(1, len(TessInvestigationTargetSource([promotion]).enumerate_targets()))
            self.assertEqual(shallow_before, {p: p.read_bytes() for p in shallow_before})

    def test_default_and_custom_outputs_remain_inside_state_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._sweep(root, (1, 2))
            self.assertEqual(0, run_tess_sector_ranking(7, root))
            self.assertTrue((root / "tess-sector-7-ranking.json").is_file())
            self.assertEqual(0, run_tess_sector_ranking(7, root, promote_top=1))
            self.assertTrue((root / "tess-sector-7-promoted-top-1.json").is_file())

            self.assertEqual(0, run_tess_sector_ranking(
                7, root, output="derived/custom-ranking.json"
            ))
            self.assertTrue((root / "derived/custom-ranking.json").is_file())
            self.assertEqual(0, run_tess_sector_ranking(
                7, root, promote_top=1,
                promotion_output="derived/custom-promotion.json"
            ))
            self.assertTrue((root / "derived/custom-promotion.json").is_file())

    def test_outside_ranking_output_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent / "sweep"; root.mkdir()
            self._sweep(root, (1,))
            before = self._tree_bytes(root)
            outside = parent / "outside-ranking.json"
            with self.assertRaisesRegex(RuntimeError, "escapes sector-sweep"):
                run_tess_sector_ranking(7, root, output=outside)
            self.assertFalse(outside.exists())
            self.assertEqual(before, self._tree_bytes(root))

    def test_outside_promotion_output_rejects_before_ranking_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent / "sweep"; root.mkdir()
            self._sweep(root, (1,))
            before = self._tree_bytes(root)
            outside = parent / "outside-promotion.json"
            with self.assertRaisesRegex(RuntimeError, "escapes sector-sweep"):
                run_tess_sector_ranking(7, root, promote_top=1,
                                        promotion_output=outside)
            self.assertFalse(outside.exists())
            self.assertFalse((root / "tess-sector-7-ranking.json").exists())
            self.assertEqual(before, self._tree_bytes(root))

    def test_parent_traversal_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent / "sweep"; root.mkdir()
            outside = parent / "outside"; outside.mkdir()
            self._sweep(root, (1,))
            before = self._tree_bytes(root)
            with self.assertRaisesRegex(RuntimeError, "escapes sector-sweep"):
                run_tess_sector_ranking(7, root, output="../traversal.json")
            (root / "escape-link").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "escapes sector-sweep"):
                run_tess_sector_ranking(7, root, output="escape-link/ranking.json")
            self.assertFalse((parent / "traversal.json").exists())
            self.assertFalse((outside / "ranking.json").exists())
            self.assertEqual(before, {key: value for key, value in self._tree_bytes(root).items()
                                      if key != Path("escape-link")})

    def test_legacy_blind_c_like_destination_remains_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root = parent / "sweep"; root.mkdir()
            blind_c = parent / "blind-c"; blind_c.mkdir()
            (blind_c / "lifecycle.json").write_bytes(b"legacy lifecycle\x00")
            (blind_c / "portfolio.json").write_bytes(b"legacy portfolio\x00")
            self._sweep(root, (1,))
            sweep_before = self._tree_bytes(root)
            blind_before = self._tree_bytes(blind_c)
            with self.assertRaisesRegex(RuntimeError, "escapes sector-sweep"):
                run_tess_sector_ranking(
                    7, root, promote_top=1,
                    promotion_output=blind_c / "promotion.json"
                )
            self.assertEqual(blind_before, self._tree_bytes(blind_c))
            self.assertEqual(sweep_before, self._tree_bytes(root))

    def test_legacy_state_rejected_before_writes(self):
        for legacy_name in ("lifecycle.json", "portfolio.json"):
            with self.subTest(legacy_name=legacy_name), tempfile.TemporaryDirectory() as tmp:
                legacy = Path(tmp) / legacy_name; legacy.write_bytes(b"do not touch")
                with self.assertRaisesRegex(RuntimeError, "refuses legacy"):
                    run_tess_sector_ranking(7, tmp)
                self.assertEqual(b"do not touch", legacy.read_bytes())
                self.assertEqual([legacy_name], [x.name for x in Path(tmp).iterdir()])


if __name__ == "__main__": unittest.main()
