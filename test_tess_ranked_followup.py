import tempfile
import unittest
from pathlib import Path

from openstar_investigation import InvestigationStore
from run_openstar_tess_sector_sweep import run_tess_sector_sweep
from test_tess_sector_sweep import FakeCoordinator, FakeProvider, Prepared, product
from workflows.tess import tess_sector_scan
from workflows.tess.tess_ranked_followup import TessDeepAdmissionStore, TessRankedFollowupTargetSource
from workflows.tess.tess_sector_archive import TessSectorInventoryStore
from workflows.tess.tess_sector_ranking import aggregate_tess_sector_ranking


class RankedFollowupTests(unittest.TestCase):
    def setUp(self):
        self.original = tess_sector_scan.read_and_prepare_tess_light_curve
        tess_sector_scan.read_and_prepare_tess_light_curve = lambda path: Prepared()

    def tearDown(self):
        tess_sector_scan.read_and_prepare_tess_light_curve = self.original

    def _ranking(self, root):
        run_tess_sector_sweep(7, "unused", root,
            provider=FakeProvider([product(1), product(2)]), coordinator=FakeCoordinator())
        inventory = TessSectorInventoryStore(Path(root) / "tess-sector-7-inventory.json").load()
        return aggregate_tess_sector_ranking(inventory, InvestigationStore(Path(root) / "investigations"))

    def test_stable_targets_and_append_only_admissions(self):
        with tempfile.TemporaryDirectory() as shallow, tempfile.TemporaryDirectory() as deep:
            ranking = self._ranking(shallow)
            ledger = TessDeepAdmissionStore(Path(deep) / "tess-sector-7-deep-admissions.json", 7)
            admitted, new, excluded = ledger.admit(ranking, 1)
            self.assertEqual((1, 1, 0), (len(admitted), len(new), len(excluded)))
            first_bytes = ledger.path.read_bytes()
            repeated, new, _ = ledger.admit(ranking, 1)
            self.assertEqual((), new)
            self.assertEqual(first_bytes, ledger.path.read_bytes())
            targets = TessRankedFollowupTargetSource(repeated).enumerate_targets()
            self.assertEqual("tess-sector-7-ranked-followup-tic-1", targets[0].id)
            self.assertEqual("tess-discovery-sector-7-tic-1", targets[0].investigation_id)
            admitted, new, _ = ledger.admit(ranking, 2)
            self.assertEqual([2], [item.ticID for item in new])
            self.assertEqual([1, 2], [item.ticID for item in admitted])

    def test_mutated_source_project_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as shallow, tempfile.TemporaryDirectory() as deep:
            ranking = self._ranking(shallow)
            Path(ranking.content["rankedEntries"][0]["sourceProjectPath"]).write_text("{}\n")
            admitted, new, excluded = TessDeepAdmissionStore(
                Path(deep) / "ledger.json", 7).admit(ranking, 1)
            self.assertEqual(((), ()), (admitted, new))
            self.assertIn("SOURCE_PROJECT_MANIFEST_SHA256_MISMATCH", excluded[0]["reason"])


if __name__ == "__main__": unittest.main()
