import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_coordinator_client import ProjectRunResult
from openstar_investigation import InvestigationStage, InvestigationStore, sha256_file, sha256_json
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine
from run_openstar_tess_sector_sweep import run_tess_sector_sweep
from run_openstar_tess_ranked_followup import run_tess_ranked_followup, validate_state_roots
from test_tess_sector_sweep import FakeCoordinator, FakeProvider, Prepared, product
from workflows.tess import tess_sector_scan
from workflows.tess.tess_followup import build_single_target_primary
from workflows.tess.tess_primary_reuse import run_primary as run_primary_with_reuse
from workflows.tess.tess_ranked_followup import (
    TessDeepAdmissionStore, TessRankedFollowupTargetSource, verified_reusable_primary,
)
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

    def _direct_primary(self, root, *, mutate_shallow=None, mutate_after_prepare=None):
        shallow, deep = Path(root) / "shallow", Path(root) / "deep"
        self._sweep(shallow, (1,))
        ranking = aggregate_tess_sector_ranking(
            TessSectorInventoryStore(shallow / "tess-sector-7-inventory.json").load(),
            InvestigationStore(shallow / "investigations"))
        admission = TessDeepAdmissionStore(deep / "ledger.json", 7).admit(ranking, 1)[0][0]
        shallow_store = InvestigationStore(shallow / "investigations")
        before = {p.relative_to(shallow): p.read_bytes() for p in shallow.rglob("*") if p.is_file()}
        if mutate_shallow is not None:
            mutate_shallow(shallow_store, admission)
        reusable = verified_reusable_primary(shallow_store, admission)
        target = TessRankedFollowupTargetSource(
            (admission,), {admission.deepInvestigationID: reusable} if reusable else {}).enumerate_targets()[0]
        deep_store = InvestigationStore(deep / "investigations")
        investigation = deep_store.create(target.investigation_id, target.workflow_id,
                                          target.workflow_version, target.metadata)

        class Coordinator:
            calls = 0
            def run_project(self, project_path, **kwargs):
                self.calls += 1
                manifest = json.loads(Path(project_path).read_text())
                entry = manifest["datasets"][0]
                status = {"projectID": "fresh-deep-project", "status": "COMPLETE",
                          "datasets": [{"id": entry["id"], "ticID": entry["ticID"],
                                        "sector": entry["sector"], "bestFrequency": .5,
                                        "bestPeriodDays": 2.0, "bestPower": .8}],
                          "nodeContributions": {"fresh": 1}}
                return ProjectRunResult("fresh-deep-project", status)

        coordinator = Coordinator(); engine = WorkflowEngine(deep_store)
        def prepare(investigation, request):
            prepared = build_single_target_primary(
                source_project_path=request.parameters["projectPath"],
                output_dir=deep_store.directory_for(investigation.id) / "artifacts",
                investigation_id=investigation.id, dataset_id=request.parameters["datasetID"],
                tic_id=request.parameters["ticID"])
            return StageOutcome(prepared, StageRequest("002-primary-distributed-search",
                "openstar.tess.primary-project.run", {"projectPath": prepared["projectPath"]}, request.id),
                input_hashes={"sourceProjectManifest": sha256_file(request.parameters["projectPath"]),
                              "sourceDataset": sha256_file(prepared["datasetPath"])})
        engine.register_handler("openstar.tess.prepare-target", prepare)
        engine.register_handler("openstar.tess.primary-project.run", lambda investigation, request:
            run_primary_with_reuse(investigation, request, coordinator, poll_interval=0, timeout=None))
        investigation, primary_request = engine.run_stage(investigation, StageRequest(
            "001-prepare-target", "openstar.tess.prepare-target",
            {"projectPath": admission.sourceProjectPath, "datasetID": admission.datasetID,
             "ticID": admission.ticID}), software_id="test", software_version="1")
        if mutate_after_prepare is not None:
            mutate_after_prepare(investigation, admission)
        output = io.StringIO()
        with redirect_stdout(output):
            investigation, downstream = engine.run_stage(investigation, primary_request,
                software_id="test", software_version="1")
        unchanged = {p.relative_to(shallow): p.read_bytes() for p in shallow.rglob("*") if p.is_file()}
        return investigation, downstream, coordinator.calls, before, unchanged, output.getvalue()

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
            workflow.register_handler(
                "openstar.tess.residual-mode-localization-review.prepare", execute
            )
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

    def test_novel_admission_does_not_rewrite_legacy_record_shape(self):
        with tempfile.TemporaryDirectory() as shallow, tempfile.TemporaryDirectory() as deep:
            ranking = self._ranking(shallow)
            ledger = TessDeepAdmissionStore(Path(deep) / "ledger.json", 7)
            ledger.admit(ranking, 1)
            original = json.loads(ledger.path.read_text())["admissions"][0]
            self.assertNotIn("admissionBasis", original)
            self.assertNotIn("noveltyScreeningSha256", original)

            entry = ranking.content["rankedEntries"][1]
            admitted, new, excluded = ledger.admit_selected(
                ranking, ((entry, "NOVEL_PRIORITY", "screen-hash"),))
            raw = json.loads(ledger.path.read_text())["admissions"]
            self.assertEqual(original, raw[0])
            self.assertEqual("NOVEL_PRIORITY", raw[1]["admissionBasis"])
            self.assertEqual("screen-hash", raw[1]["noveltyScreeningSha256"])
            self.assertEqual((2, 1, 0), (len(admitted), len(new), len(excluded)))

            admitted, duplicate, excluded = ledger.admit_selected(
                ranking, ((entry, "NOVEL_PRIORITY", "other-hash"),))
            self.assertEqual((2, 0, 0), (len(admitted), len(duplicate), len(excluded)))
            self.assertEqual(raw, json.loads(ledger.path.read_text())["admissions"])

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

    def test_deep_primary_reuses_full_result_and_continues_without_compute(self):
        with tempfile.TemporaryDirectory() as temporary:
            investigation, downstream, calls, before, after, output = self._direct_primary(temporary)
            primary = investigation.stages[-1]
            self.assertEqual(0, calls)
            self.assertIn("♻️ Reusing verified shallow distributed period search", output)
            self.assertEqual("003-catalog-identity", downstream.id)
            self.assertEqual("openstar.tess.catalog-identity", downstream.handler_id)
            self.assertEqual("COMPLETE", primary.result["status"])
            self.assertEqual("project-1", primary.result["projectID"])
            self.assertEqual({"mac": 2, "iphone": 1}, primary.result["nodeContributions"])
            self.assertEqual("tess-sector-7-tic-1", primary.result["datasets"][0]["id"])
            self.assertEqual("REUSED_SHALLOW_COMPUTE",
                             primary.result["reuseProvenance"]["computeDisposition"])
            self.assertEqual(("project-1",), primary.provenance.project_ids)
            self.assertEqual(before, after, "deep reuse must not rewrite shallow state")

    def test_legacy_dataset_id_only_result_remains_reusable(self):
        def use_legacy_identity(store, admission):
            investigation = store.load(admission.sourceScanInvestigationID)
            stages = list(investigation.stages)
            index = next(i for i, stage in enumerate(stages)
                         if stage.id == "002-broad-distributed-scan")
            result = dict(stages[index].result)
            dataset = dict(result["datasets"][0])
            dataset["datasetID"] = dataset.pop("id")
            result["datasets"] = [dataset]
            stages[index] = replace(stages[index], result=result)
            store.save(replace(investigation, stages=tuple(stages)))

        with tempfile.TemporaryDirectory() as temporary:
            investigation, _, calls, _, _, output = self._direct_primary(
                temporary, mutate_shallow=use_legacy_identity)
            self.assertEqual(0, calls)
            self.assertIn("♻️ Reusing verified shallow distributed period search", output)
            self.assertEqual("REUSED_SHALLOW_COMPUTE",
                             investigation.stages[-1].result["reuseProvenance"]["computeDisposition"])

    def test_primary_gate_rejects_disagreeing_canonical_and_legacy_ids(self):
        def contradict_reusable_identity(investigation, admission):
            reusable = investigation.metadata["reusablePrimary"]
            reusable["coordinatorResult"]["datasets"][0]["datasetID"] = "wrong"
            reusable["coordinatorResultSha256"] = sha256_json(reusable["coordinatorResult"])

        with tempfile.TemporaryDirectory() as temporary:
            investigation, _, calls, _, _, output = self._direct_primary(
                temporary, mutate_after_prepare=contradict_reusable_identity)
            self.assertEqual(1, calls)
            self.assertIn("⚙️ Activating primary distributed period search", output)
            self.assertEqual("FRESH_DEEP_COMPUTE",
                             investigation.stages[-1].result["reuseProvenance"]["mode"])

    def test_deep_primary_falls_back_when_frozen_input_changes(self):
        def mutate_dataset(investigation, admission):
            Path(admission.datasetArtifact).write_text("{}\n", encoding="utf-8")
        def mutate_project(investigation, admission):
            Path(admission.sourceProjectPath).write_text("{}\n", encoding="utf-8")
        def mutate_workload(investigation, admission):
            primary = Path(investigation.stages[-1].result["projectPath"])
            value = json.loads(primary.read_text()); value["workloadID"] = "wrong"
            primary.write_text(json.dumps(value), encoding="utf-8")
        def mutate_frequency(investigation, admission):
            dataset = Path(admission.datasetArtifact)
            value = json.loads(dataset.read_text()); value["frequencySearch"]["minimumFrequency"] = 999
            dataset.write_text(json.dumps(value), encoding="utf-8")
        for label, mutation in (("dataset-hash", mutate_dataset), ("project-hash", mutate_project),
                                ("workload", mutate_workload), ("frequency-search", mutate_frequency)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                investigation, _, calls, _, _, _ = self._direct_primary(
                    temporary, mutate_after_prepare=mutation)
                self.assertEqual(1, calls)
                self.assertEqual("FRESH_DEEP_COMPUTE",
                                 investigation.stages[-1].result["reuseProvenance"]["mode"])
                self.assertEqual(("fresh-deep-project",), investigation.stages[-1].provenance.project_ids)

    def test_shallow_verification_doubt_forces_fresh_compute(self):
        def change_scan(store, admission, transform):
            investigation = store.load(admission.sourceScanInvestigationID)
            stages = list(investigation.stages)
            index = next(i for i, stage in enumerate(stages) if stage.id == "002-broad-distributed-scan")
            stages[index] = transform(stages[index])
            store.save(replace(investigation, stages=tuple(stages)))

        mutators = {
            "scan-project-path": lambda store, admission: change_scan(
                store, admission, lambda stage: replace(stage, parameters={"projectPath": "/wrong/project.json"})),
            "result-disagreeing-id-and-dataset-id": lambda store, admission: change_scan(
                store, admission, lambda stage: replace(stage, result={**stage.result,
                    "datasets": [{**stage.result["datasets"][0], "datasetID": "wrong"}]})),
            "result-tic": lambda store, admission: change_scan(
                store, admission, lambda stage: replace(stage, result={**stage.result,
                    "datasets": [{**stage.result["datasets"][0], "ticID": 999}]})),
            "result-sector": lambda store, admission: change_scan(
                store, admission, lambda stage: replace(stage, result={**stage.result,
                    "datasets": [{**stage.result["datasets"][0], "sector": 99}]})),
            "incomplete": lambda store, admission: store.set_status(
                store.load(admission.sourceScanInvestigationID), "RUNNING"),
            "failed": lambda store, admission: store.set_status(
                store.load(admission.sourceScanInvestigationID), "FAILED"),
            "malformed-result": lambda store, admission: change_scan(
                store, admission, lambda stage: replace(stage, result={"datasets": "bad"})),
            "unsuccessful-result": lambda store, admission: change_scan(
                store, admission, lambda stage: replace(stage, result={**stage.result, "status": "FAILED"})),
            "evidence-hash": lambda store, admission: Path(
                store.load(admission.sourceScanInvestigationID).stages[-1].artifacts[0].path
            ).write_text("{}\n", encoding="utf-8"),
        }
        for label, mutation in mutators.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                investigation, _, calls, _, _, _ = self._direct_primary(
                    temporary, mutate_shallow=mutation)
                self.assertEqual(1, calls)
                self.assertEqual("FRESH_DEEP_COMPUTE",
                                 investigation.stages[-1].result["reuseProvenance"]["mode"])

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

    def test_ranked_rerun_recovers_admitted_chained_stage_022_failure_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); shallow = root / "shallow"; deep = root / "deep"
            self._sweep(shallow, (1,))
            executions = []
            self._run(shallow, deep, 1, executions)
            store = InvestigationStore(deep / "investigations")
            investigation = store.load("tess-discovery-sector-7-tic-1")
            stages = (
                InvestigationStage("010-morphology", "openstar.tess.morphology.analyze", "COMPLETE", None, {}, result={"physicalCycleResolved": False}),
                InvestigationStage("011-dynamic", "openstar.tess.dynamic-harmonic.analyze", "COMPLETE", "010-morphology", {}, result={"referenceFamilyPeriodDays": 10.3, "supportedHarmonicOrders": [1, 2, 3, 4]}),
                InvestigationStage("012-time-frequency-prepare", "openstar.tess.time-frequency.prepare", "COMPLETE", "011-dynamic", {}, result={"absoluteTimeReferenceDays": 2500.0}),
                InvestigationStage("013-time-frequency-summary", "openstar.tess.time-frequency.summarize", "COMPLETE", "012-time-frequency-prepare", {}, result={"residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"}}),
                InvestigationStage("018-mode-identification", "openstar.tess.mode-identification.analyze", "COMPLETE", "013-time-frequency-summary", {}, result={"independentModeEvidenceSurvived": True, "physicalMechanismResolved": False, "establishedPeriodFamily": {"referencePeriodDays": 10.3, "modeledHarmonicOrders": [1, 2, 3, 4]}, "modeCandidate": {"frequencyCyclesPerDay": 0.25, "periodDays": 4.0, "supportingSectors": [2, 29, 68, 69]}}),
                InvestigationStage("019-prepare-residual-mode-localization", "openstar.tess.residual-mode-localization.prepare", "COMPLETE", "018-mode-identification", {}, result={"subtractedHarmonicOrders": [1, 2, 3, 4]}),
                InvestigationStage("020-run-residual-mode-localization", "openstar.tess.residual-mode-localization.run", "COMPLETE", "019-prepare-residual-mode-localization", {}, result={}),
                InvestigationStage("021-interpret-residual-mode-localization", "openstar.tess.residual-mode-localization.interpret", "COMPLETE", "020-run-residual-mode-localization", {}, result={"recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"}),
                InvestigationStage("022-prepare-residual-mode-localization-review", "openstar.tess.residual-mode-localization-review.prepare", "FAILED", "021-interpret-residual-mode-localization", {}, error="RuntimeError: v20.11 requires the completed v20.9 nonstationary model.", failure_classification="NON_RETRYABLE"),
            )
            selected = stages[-1]
            failed = replace(investigation, status="FAILED", stages=stages, metadata={
                **investigation.metadata,
                "controlState": {"schedulerAction": "RUN_EXPERIMENT", "selectedExperiment": {
                    "id": selected.id, "handler_id": selected.handler_id,
                    "parameters": {}, "triggered_by_stage_id": selected.triggered_by_stage_id,
                }},
            })
            store.save(failed)
            ledger = deep / "tess-sector-7-deep-admissions.json"
            ledger_before = ledger.read_bytes()
            executions.clear()

            code, _ = self._run(shallow, deep, 1, executions)

            self.assertEqual(0, code)
            recovered = store.load(failed.id)
            self.assertEqual(stages, recovered.stages[:-1])
            self.assertEqual("023-prepare-residual-mode-localization-review", recovered.stages[-1].id)
            self.assertEqual([1], executions)
            self.assertEqual(ledger_before, ledger.read_bytes())
            code, _ = self._run(shallow, deep, 1, executions)
            self.assertEqual(0, code)
            self.assertEqual([1], executions)
            self.assertEqual(len(stages) + 1, len(store.load(failed.id).stages))

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
