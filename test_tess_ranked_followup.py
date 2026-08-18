import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_investigation import InvestigationStore
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine
from run_openstar_tess_sector_sweep import run_tess_sector_sweep
from run_openstar_tess_ranked_followup import run_tess_ranked_followup, validate_state_roots
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

    def _sweep(self, root, tics=(1, 2)):
        run_tess_sector_sweep(7, "unused", root,
            provider=FakeProvider([product(tic) for tic in tics]), coordinator=FakeCoordinator())

    def _run(self, shallow, deep, top, executions, *, planner=None, execute_hook=None):
        class Coordinator:
            calls = []
            def run_project(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                raise AssertionError("test workflow must not invoke coordinator")

        coordinator = Coordinator()

        def register(store, unused_coordinator, **kwargs):
            workflow = WorkflowEngine(store)
            def execute(investigation, request):
                ledger = Path(deep) / "tess-sector-7-deep-admissions.json"
                self.assertTrue(ledger.is_file(), "admission must precede dispatch")
                executions.append(investigation.metadata["ticID"])
                if execute_hook is not None: execute_hook(investigation)
                if request.parameters.get("fail"):
                    raise ValueError("isolated failure")
                return StageOutcome({"claim": "KNOWN_PERIOD_RECOVERED"}, stop=True)
            workflow.register_handler("test.execute", execute)
            return workflow

        def default_planner(investigation, target):
            if investigation.stages: return ()
            return (ScientificBranch("run", StageRequest(
                "001-run", "test.execute", {})),)

        with patch("run_openstar_tess_ranked_followup.register_tess_workflow_handlers", register), \
             patch("run_openstar_tess_ranked_followup.plan_tess_branches", planner or default_planner):
            code = run_tess_ranked_followup(7, shallow, deep, "unused", top,
                max_concurrent_investigations=2, coordinator=coordinator)
        return code, coordinator

    @staticmethod
    def _tree_bytes(root):
        root = Path(root)
        return {path.relative_to(root): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

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

    def test_mutated_dataset_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as shallow, tempfile.TemporaryDirectory() as deep:
            ranking = self._ranking(shallow)
            Path(ranking.content["rankedEntries"][0]["datasetArtifact"]).write_bytes(b"changed")
            admitted, new, excluded = TessDeepAdmissionStore(
                Path(deep) / "ledger.json", 7).admit(ranking, 1)
            self.assertEqual(((), ()), (admitted, new))
            self.assertIn("DATASET_SHA256_MISMATCH", excluded[0]["reason"])

    def test_overlap_validation_same_nested_both_directions_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            child = root / "child"
            child.mkdir()
            for sector, deep in ((root, root), (root, child), (child, root)):
                with self.subTest(sector=sector, deep=deep), self.assertRaisesRegex(
                        RuntimeError, "non-overlapping"):
                    validate_state_roots(sector, deep)
            sibling = root / "sibling"
            sibling.mkdir()
            link = sibling / "linked-child"
            link.symlink_to(child, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "non-overlapping"):
                validate_state_roots(root, link)

    def test_direct_and_nested_legacy_roots_are_rejected_without_writes(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                legacy = base / "blind-c"
                legacy.mkdir()
                (legacy / "lifecycle.json").write_bytes(b"legacy lifecycle\x00")
                (legacy / "portfolio.json").write_bytes(b"legacy portfolio\x00")
                requested = legacy / "ranked-followup" if nested else legacy
                if nested: requested.mkdir()
                sector = base / "sector"
                sector.mkdir()
                before = self._tree_bytes(legacy)
                with patch("run_openstar_tess_ranked_followup.OpenStarCoordinatorClient") as coordinator:
                    with self.assertRaisesRegex(RuntimeError, "legacy state"):
                        run_tess_ranked_followup(7, sector, requested, "unused", 1)
                    coordinator.assert_not_called()
                self.assertEqual(before, self._tree_bytes(legacy))
                self.assertEqual({}, self._tree_bytes(sector))

    def test_symlink_resolved_child_of_legacy_root_is_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); legacy = base / "blind-c"; legacy.mkdir()
            (legacy / "lifecycle.json").write_bytes(b"do not touch")
            child = legacy / "child"; child.mkdir()
            link = base / "deep-link"; link.symlink_to(child, target_is_directory=True)
            sector = base / "sector"; sector.mkdir()
            before = self._tree_bytes(legacy)
            with self.assertRaisesRegex(RuntimeError, "legacy state"):
                validate_state_roots(sector, link)
            self.assertEqual(before, self._tree_bytes(legacy))

    def test_runner_persists_before_dispatch_completed_rerun_and_top_growth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow)
            executions = []
            code, coordinator = self._run(shallow, deep, 1, executions)
            self.assertEqual((0, [1], []), (code, executions, coordinator.calls))
            investigation_path = deep / "investigations/tess-discovery-sector-7-tic-1/investigation.json"
            completed_bytes = investigation_path.read_bytes()
            code, coordinator = self._run(shallow, deep, 1, executions)
            self.assertEqual((0, [1], []), (code, executions, coordinator.calls))
            self.assertEqual(completed_bytes, investigation_path.read_bytes())
            code, coordinator = self._run(shallow, deep, 2, executions)
            self.assertEqual((0, [1, 2], []), (code, executions, coordinator.calls))
            self.assertEqual(completed_bytes, investigation_path.read_bytes())

    def test_new_deep_target_receives_verified_full_shallow_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow, (1,))
            captured = {}
            def inspect(investigation):
                captured.update(investigation.metadata["reusablePrimary"])
            code, coordinator = self._run(shallow, deep, 1, [], execute_hook=inspect)
            self.assertEqual((0, []), (code, coordinator.calls))
            self.assertEqual("EXACT_FROZEN_SHALLOW_PRIMARY", captured["verification"])
            self.assertEqual("openstar.workflow.tess-sector-scan.v1", captured["sourceWorkflowID"])
            self.assertEqual("COMPLETE", captured["coordinatorResult"]["status"])
            self.assertEqual(1, len(captured["coordinatorResult"]["datasets"]))
            self.assertEqual(["project-1"], captured["computeProjectIDs"])

    def test_unverified_shallow_result_is_not_offered_for_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow, (1,))
            store = InvestigationStore(shallow / "investigations")
            investigation = store.load("tess-sector-scan-7-tic-1")
            evidence_path = Path(investigation.stages[-1].artifacts[0].path)
            evidence_path.write_text("{}\n", encoding="utf-8")
            def inspect(investigation):
                self.assertNotIn("reusablePrimary", investigation.metadata)
            code, coordinator = self._run(shallow, deep, 1, [], execute_hook=inspect)
            self.assertEqual((0, []), (code, coordinator.calls))

    def test_ranked_rerun_applies_tess_repair_before_scheduler_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow)
            executions = []
            self._run(shallow, deep, 1, executions)
            repaired = []

            def repair(store, investigation):
                repaired.append(investigation.id)
                return investigation

            with patch(
                "run_openstar_tess_ranked_followup.repair_obsolete_terminal_wait",
                side_effect=repair,
            ):
                code, _ = self._run(shallow, deep, 1, executions)
            self.assertEqual(0, code)
            self.assertEqual(["tess-discovery-sector-7-tic-1"], repaired)
            self.assertEqual([1], executions)

    def test_runner_refresh_is_local_and_shallow_science_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow)
            before = self._tree_bytes(shallow)
            with patch("workflows.tess.tess_sector_archive.MastTessSectorArchiveProvider.inventory_sector",
                       side_effect=AssertionError("MAST must not be called")) as mast:
                _, coordinator = self._run(shallow, deep, 1, [])
            mast.assert_not_called(); self.assertEqual([], coordinator.calls)
            after = self._tree_bytes(shallow)
            ranking_path = Path("tess-sector-7-ranking.json")
            self.assertEqual({k: v for k, v in before.items() if k != ranking_path},
                             {k: v for k, v in after.items() if k != ranking_path})
            self.assertIn(ranking_path, after)

    def test_waiting_and_failed_candidates_do_not_block_runnable_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow, (1, 2, 3))
            executions = []
            def planner(investigation, target):
                tic = target.metadata["ticID"]
                if tic == 1:
                    return (ScientificBranch("wait", StageRequest("001-wait", "test.execute", {}),
                        external_data=(ExternalDataDependency("remote", False),)),)
                if investigation.stages: return ()
                return (ScientificBranch("run", StageRequest("001-run", "test.execute",
                    {"fail": tic == 2})),)
            code, _ = self._run(shallow, deep, 3, executions, planner=planner)
            self.assertEqual(1, code)
            self.assertCountEqual([2, 3], executions)
            store = InvestigationStore(deep / "investigations")
            self.assertEqual("FAILED", store.load("tess-discovery-sector-7-tic-2").status)
            self.assertEqual("COMPLETE", store.load("tess-discovery-sector-7-tic-3").status)

    def test_runner_uses_existing_scheduler_concurrency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow)
            both_started = threading.Event(); release = threading.Event(); lock = threading.Lock()
            starts = []
            def hook(investigation):
                with lock:
                    starts.append(investigation.id)
                    if len(starts) == 2: both_started.set()
                self.assertTrue(both_started.wait(2))
                if len(starts) == 2: release.set()
                self.assertTrue(release.wait(2))
            code, _ = self._run(shallow, deep, 2, [], execute_hook=hook)
            self.assertEqual(0, code)
            self.assertEqual(2, len(starts))


if __name__ == "__main__": unittest.main()
