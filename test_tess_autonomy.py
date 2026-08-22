import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from openstar_autonomy import AutonomousInvestigationEngine
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_lifecycle import InvestigationLifecycleLoop
from openstar_targets import InvestigationTargetPortfolio
from openstar_workflow import StageOutcome, WorkflowEngine
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import (
    TessInvestigationTargetSource,
    plan_tess_branches,
    repair_obsolete_terminal_wait,
)
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION


class TessAutonomyIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = InvestigationStore(self.root / "investigations")
        self.project = self.root / "tess-project.json"
        self.project.write_text(
            json.dumps(
                {
                    "id": "real-tess-project",
                    "workloadID": "openstar.tess-period-search.v1",
                    "datasets": [
                        {
                            "id": "blind-c",
                            "targetName": "Blind C",
                            "path": "c.json",
                            "ticID": 1,
                        },
                        {
                            "id": "next-real-target",
                            "targetName": "Next",
                            "path": "next.json",
                            "ticID": 2,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.source = TessInvestigationTargetSource((self.project,))

    def _complete(
        self, investigation, stage_id, handler_id, result, next_stage=None, stop=False
    ):
        running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
        investigation = self.store.append_running_stage(investigation, running)
        terminal = self.store.build_terminal_stage(
            stage_id=stage_id,
            handler_id=handler_id,
            status="COMPLETE",
            triggered_by_stage_id=None,
            parameters={},
            result=result,
            error=None,
            software_id="replay",
            software_version="20.28",
            started_at=running.started_at,
            next_stage=(
                {
                    "id": next_stage.id,
                    "handler_id": next_stage.handler_id,
                    "parameters": next_stage.parameters,
                    "triggered_by_stage_id": next_stage.triggered_by_stage_id,
                }
                if next_stage is not None
                else None
            ),
            stop=stop,
        )
        return self.store.complete_current_stage(investigation, terminal)

    def test_real_project_targets_have_stable_ids_priority_and_primary_handler(self):
        first = self.source.enumerate_targets()
        second = self.source.enumerate_targets()
        self.assertEqual(first, second)
        self.assertEqual([0, 1], [target.priority for target in first])
        investigation = self.store.create(
            first[0].investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=first[0].metadata,
        )
        branches = plan_tess_branches(investigation, first[0])
        self.assertEqual(
            "openstar.tess.prepare-target", branches[0].experiment.handler_id
        )
        self.assertEqual("blind-c", branches[0].experiment.parameters["datasetID"])

    def test_tic_277940827_stage_047_schedules_residual_phase_difference_image_append_only(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
                                          metadata={**target.metadata, "ticID": 277940827})
        for number in range(1, 47):
            investigation = self._complete(
                investigation, f"{number:03d}-persisted-science", "persisted.science", {"stage": number})
        investigation = self._complete(
            investigation, "047-interpret-catalog-guided-source-localization",
            "openstar.tess.catalog-guided-source-localization.interpret",
            {"recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
             "classification": "UNRESOLVED", "sourceAttributionResolved": False,
             "physicalCycleResolved": False})
        old_stages = investigation.stages
        branches = plan_tess_branches(investigation, target)
        request = branches[0].experiment
        self.assertEqual("048-prepare-residual-phase-difference-imaging", request.id)
        self.assertEqual("openstar.tess.residual-phase-difference-imaging.prepare",
                         request.handler_id)
        engine = WorkflowEngine(self.store)
        engine.register_handler(request.handler_id, lambda _investigation, _request: StageOutcome(
            result={"scheduledFrom": "ADDITIONAL_SOURCE_LOCALIZATION_DATA"}, stop=True))
        completed, _ = engine.run_stage(
            investigation, request, software_id="test", software_version="stage-048")
        self.assertEqual(old_stages, completed.stages[:47])
        self.assertEqual("048-prepare-residual-phase-difference-imaging", completed.stages[47].id)

    def test_stage_050_candidate_recovery_is_authoritative_and_idempotent(self):
        target = self.source.enumerate_targets()[0]
        candidate_1 = {"raDeg": 10.1, "decDeg": -20.1,
                       "catalogIDs": {"ticID": 111}}
        candidate_2 = {"raDeg": 10.2, "decDeg": -20.2,
                       "catalogIDs": {"gaiaDR3SourceID": 222}}

        def history():
            investigation = self.store.create(
                target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
                metadata={**target.metadata, "ticID": 277940827})
            for number in range(1, 47):
                investigation = self._complete(
                    investigation, f"{number:03d}-persisted-science",
                    "persisted.science", {"stage": number})
            evidence = (
                ("047-interpret-catalog-guided-source-localization",
                 "openstar.tess.catalog-guided-source-localization.interpret",
                 {"classification": "UNRESOLVED", "sourceAttributionResolved": False,
                  "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA"}),
                ("048-prepare-residual-phase-difference-imaging",
                 "openstar.tess.residual-phase-difference-imaging.prepare", {}),
                ("049-run-residual-phase-difference-imaging",
                 "openstar.tess.residual-phase-difference-imaging.run", {}),
                ("050-interpret-residual-phase-difference-imaging",
                 "openstar.tess.residual-phase-difference-imaging.interpret",
                 {"classification": "CANDIDATE_2_SUPPORTED",
                  "sourceAttributionResolved": True,
                  "preferredCandidate": candidate_2,
                  "catalogCandidates": [candidate_1, candidate_2],
                  "recommendedNextTest":
                  "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"}),
            )
            for stage_id, handler, result in evidence:
                investigation = self._complete(investigation, stage_id, handler, result)
            return investigation

        investigation = history()
        branches = plan_tess_branches(investigation, target)
        self.assertEqual(1, len(branches))
        request = branches[0].experiment
        self.assertEqual("051-prepare-offset-source-variability", request.id)
        self.assertEqual("openstar.tess.offset-source-variability.prepare", request.handler_id)
        self.assertEqual("050-interpret-residual-phase-difference-imaging",
                         request.triggered_by_stage_id)

        running = InvestigationStage(
            request.id, request.handler_id, "RUNNING", request.triggered_by_stage_id, {})
        with_running = self.store.append_running_stage(investigation, running)
        self.assertEqual((), plan_tess_branches(with_running, target))

        terminal = self.store.build_terminal_stage(
            stage_id=request.id, handler_id=request.handler_id, status="COMPLETE",
            triggered_by_stage_id=request.triggered_by_stage_id, parameters={}, result={},
            error=None, software_id="test", software_version="1", started_at=running.started_at)
        with_complete = self.store.complete_current_stage(with_running, terminal)
        self.assertEqual((), plan_tess_branches(with_complete, target))

    def test_stage_053_candidate_two_recovery_is_idempotent(self):
        target = self.source.enumerate_targets()[0]
        candidate_1 = {"raDeg": 10.1, "decDeg": -20.1,
                       "catalogIDs": {"ticID": 111}}
        candidate_2 = {"raDeg": 10.2, "decDeg": -20.2,
                       "catalogIDs": {"gaiaDR3SourceID": 222}}
        investigation = self.store.create(
            target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
            metadata={**target.metadata, "ticID": 277940827})
        evidence = (
            ("047-interpret-catalog-guided-source-localization",
             "openstar.tess.catalog-guided-source-localization.interpret",
             {"classification": "UNRESOLVED", "sourceAttributionResolved": False,
              "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA"}),
            ("050-interpret-residual-phase-difference-imaging",
             "openstar.tess.residual-phase-difference-imaging.interpret",
             {"classification": "SOURCE_SWITCHING_BY_SECTOR",
              "recommendedNextTest": "SOURCE_SWITCHING_TEMPORAL_MODEL"}),
            ("051-prepare-source-switching-temporal-model",
             "openstar.tess.source-switching-temporal-model.prepare", {}),
            ("052-run-source-switching-temporal-model",
             "openstar.tess.source-switching-temporal-model.run", {}),
            ("053-interpret-source-switching-temporal-model",
             "openstar.tess.source-switching-temporal-model.interpret",
             {"classification": "STATIONARY_CANDIDATE_2_SOURCE",
              "sourceAttributionResolved": True, "preferredCandidate": candidate_2,
              "catalogCandidates": [candidate_1, candidate_2],
              "recommendedNextTest":
              "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"}),
        )
        for stage_id, handler, result in evidence:
            investigation = self._complete(investigation, stage_id, handler, result)
        branches = plan_tess_branches(investigation, target)
        self.assertEqual(1, len(branches))
        request = branches[0].experiment
        self.assertEqual("054-prepare-offset-source-variability", request.id)
        self.assertEqual("openstar.tess.offset-source-variability.prepare", request.handler_id)
        running = InvestigationStage(
            request.id, request.handler_id, "RUNNING", request.triggered_by_stage_id, {})
        with_running = self.store.append_running_stage(investigation, running)
        self.assertEqual((), plan_tess_branches(with_running, target))
        terminal = self.store.build_terminal_stage(
            stage_id=request.id, handler_id=request.handler_id, status="COMPLETE",
            triggered_by_stage_id=request.triggered_by_stage_id, parameters={}, result={},
            error=None, software_id="test", software_version="1", started_at=running.started_at)
        with_complete = self.store.complete_current_stage(with_running, terminal)
        self.assertEqual((), plan_tess_branches(with_complete, target))

    def test_completed_mode_identification_boundary_is_append_only_and_idempotent(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
                                          metadata=target.metadata)
        investigation = self._complete(
            investigation, "019-summarize-time-frequency", "openstar.tess.time-frequency.summarize",
            {"recommendedNextTest": "MODE_IDENTIFICATION_OR_PULSATION_MODELING",
             "physicalMechanismResolved": False})
        investigation = self._complete(investigation, "020-finalize", "openstar.tess.finalize", {}, stop=True)
        investigation = self.store.set_control_state(
            investigation, status="COMPLETE",
            control_state={"schedulerAction": "INVESTIGATION_COMPLETE"})
        old_stages = investigation.stages
        repaired = repair_obsolete_terminal_wait(self.store, investigation)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(old_stages, repaired.stages)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("openstar.tess.mode-identification.analyze", selected["handler_id"])
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

    def test_unrelated_complete_investigation_is_unchanged(self):
        target = self.source.enumerate_targets()[1]
        investigation = self.store.create(target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
                                          metadata=target.metadata)
        investigation = self._complete(investigation, "020-finalize", "openstar.tess.finalize", {}, stop=True)
        investigation = self.store.set_control_state(
            investigation, status="COMPLETE",
            control_state={"schedulerAction": "INVESTIGATION_COMPLETE"})
        before = self.store.path_for(investigation.id).read_bytes()
        self.assertEqual(investigation, repair_obsolete_terminal_wait(self.store, investigation))
        self.assertEqual(before, self.store.path_for(investigation.id).read_bytes())

    def test_completed_dynamic_harmonic_boundary_is_append_only_and_idempotent(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
                                          metadata=target.metadata)
        investigation = self._complete(
            investigation, "020-mode-identification", "openstar.tess.mode-identification.analyze",
            {"recommendedNextTest": "DYNAMIC_HARMONIC_MODELING",
             "physicalMechanismResolved": False})
        investigation = self._complete(investigation, "021-finalize", "openstar.tess.finalize", {}, stop=True)
        investigation = self.store.set_control_state(
            investigation, status="COMPLETE",
            control_state={"schedulerAction": "INVESTIGATION_COMPLETE"})
        old_stages = investigation.stages
        repaired = repair_obsolete_terminal_wait(self.store, investigation)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(old_stages, repaired.stages)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("openstar.tess.dynamic-harmonic.analyze", selected["handler_id"])
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

    def test_failed_unresolved_dynamic_localization_review_reruns_full_family_once(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
                                          metadata=target.metadata)
        evidence = (
            ("010-morphology", "openstar.tess.morphology.analyze",
             {"physicalCycleResolved": False, "physicalMechanismResolved": False}),
            ("011-dynamic", "openstar.tess.dynamic-harmonic.analyze",
             {"referenceFamilyPeriodDays": 10.30084080080649,
              "supportedHarmonicOrders": [1, 2, 3, 4],
              "classification": "ADDITIONAL_VARIABILITY_REMAINS",
              "physicalMechanismResolved": False}),
            ("012-time-frequency-prepare", "openstar.tess.time-frequency.prepare",
             {"absoluteTimeReferenceDays": 2500.0,
              "subtractedHarmonicOrders": [1, 2, 3, 4],
              "workloadID": "openstar.lomb-scargle.v1"}),
            ("013-time-frequency-summary", "openstar.tess.time-frequency.summarize",
             {"residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"},
              "physicalMechanismResolved": False}),
            ("014-mode", "openstar.tess.mode-identification.analyze",
             {"independentModeEvidenceSurvived": True, "physicalMechanismResolved": False,
              "modeCandidate": {"frequencyCyclesPerDay": 1 / 2.206,
                                "supportingSectors": [94, 95, 102, 103]}}),
            ("015-localization-prepare", "openstar.tess.residual-mode-localization.prepare",
             {"subtractedHarmonicOrders": [1, 2],
              "workloadID": "openstar.lomb-scargle.v1"}),
            ("017-localization-interpret", "openstar.tess.residual-mode-localization.interpret",
             {"classification": "RESIDUAL_MODE_LOCALIZATION_UNRESOLVED",
              "recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"}),
        )
        for stage_id, handler, result in evidence:
            investigation = self._complete(investigation, stage_id, handler, result)
        request = StageRequest(
            "018-review-prepare", "openstar.tess.residual-mode-localization-review.prepare",
            {}, "017-localization-interpret",
        )
        investigation = self.store.set_control_state(
            investigation, status="RUNNING",
            control_state={"schedulerAction": "RUN_EXPERIMENT",
                           "selectedExperiment": asdict(request)},
        )
        engine = WorkflowEngine(self.store)

        def fail_review_prepare(_investigation, _request):
            raise RuntimeError(
                "v20.11 requires the morphology-resolved physical period."
            )

        engine.register_handler(request.handler_id, fail_review_prepare)
        with self.assertRaisesRegex(RuntimeError, "morphology-resolved physical period"):
            engine.run_stage(
                investigation, request, software_id="legacy", software_version="20.9"
            )
        investigation = self.store.load(investigation.id)
        self.assertEqual("FAILED", investigation.status)
        self.assertEqual("RUN_EXPERIMENT",
                         investigation.metadata["controlState"]["schedulerAction"])
        old_stages = investigation.stages
        old_stage_files = {
            stage.id: self.store.stage_path_for(investigation.id, stage.id).read_bytes()
            for stage in old_stages
        }

        repaired = repair_obsolete_terminal_wait(self.store, investigation)

        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(old_stages, repaired.stages)
        self.assertEqual(
            "TESS_UNRESOLVED_DYNAMIC_LOCALIZATION_REVIEW_COMPATIBILITY_RETRY",
            repaired.metadata["controlState"]["recovery"],
        )
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("openstar.tess.residual-mode-localization.prepare",
                         selected["handler_id"])
        self.assertEqual(
            old_stage_files,
            {stage.id: self.store.stage_path_for(repaired.id, stage.id).read_bytes()
             for stage in repaired.stages},
        )
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

        unrelated = replace(
            investigation,
            metadata={**investigation.metadata,
                      "controlState": {"schedulerAction": "RUN_EXPERIMENT",
                                       "selectedExperiment": {
                                           **asdict(request), "id": "another-stage"}}},
        )
        self.assertEqual(unrelated, repair_obsolete_terminal_wait(self.store, unrelated))
        unrelated_failure = replace(
            investigation,
            stages=investigation.stages[:-1] + (
                replace(investigation.stages[-1], error="RuntimeError: unrelated failure"),
            ),
        )
        self.assertEqual(
            unrelated_failure,
            repair_obsolete_terminal_wait(self.store, unrelated_failure),
        )

    def test_real_failed_multisource_prepare_appends_corrected_retry(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        evidence = (
            ("001-prepared", "openstar.tess.prepare-target",
             {"sourceProjectPath": str(self.project),
              "sourceDatasetEntry": {"id": "blind-c"},
              "ticID": 277940827, "sector": 1}),
            ("002-identity", "openstar.tess.catalog-identity", {}),
            ("003-independent", "openstar.tess.independent.prepare", {}),
            ("010-morphology", "openstar.tess.morphology.analyze",
             {"physicalCycleResolved": False}),
            ("021-dynamic", "openstar.tess.dynamic-harmonic.analyze",
             {"referenceFamilyPeriodDays": 10.30084080080649,
              "supportedHarmonicOrders": [1, 2, 3, 4],
              "classification": "ADDITIONAL_VARIABILITY_REMAINS"}),
            ("035-review", "openstar.tess.residual-mode-localization-review.interpret",
             {"residualFrequencyAtReference": 1 / 2.2071724078510457,
              "residualPeriodAtReferenceDays": 2.2071724078510457,
              "fractionalFrequencyDriftPerDay": 0.0,
              "timeReferenceDays": 2500.0,
              "signalSectors": [94, 95, 102, 103],
              "crossTime": {
                  "classification": "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND",
                  "residualModeOrigin": "TIME_VARIABLE_OR_BLENDED",
              },
              "recommendedNextTest": "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"}),
        )
        for stage_id, handler, result in evidence:
            investigation = self._complete(
                investigation, stage_id, handler, result
            )
        request = StageRequest(
            "036-prepare-multi-source-residual",
            "openstar.tess.multi-source-residual.prepare", {}, "035-review",
        )
        investigation = self.store.set_control_state(
            investigation, status="RUNNING",
            control_state={"schedulerAction": "RUN_EXPERIMENT",
                           "selectedExperiment": asdict(request)},
        )
        engine = WorkflowEngine(self.store)

        def obsolete_gate(_investigation, _request):
            raise RuntimeError("v20.12 requires the morphology-resolved physical period.")

        engine.register_handler(request.handler_id, obsolete_gate)
        with self.assertRaisesRegex(RuntimeError, "morphology-resolved"):
            engine.run_stage(
                investigation, request, software_id="legacy", software_version="20.11"
            )
        failed = self.store.load(investigation.id)
        repaired_once = repair_obsolete_terminal_wait(self.store, failed)
        retry_037 = StageRequest(**repaired_once.metadata["controlState"]["selectedExperiment"])

        def obsolete_nonstationary_gate(_investigation, _request):
            raise RuntimeError(
                "v20.12 requires the completed v20.9 nonstationary model."
            )

        engine_037 = WorkflowEngine(self.store)
        engine_037.register_handler(request.handler_id, obsolete_nonstationary_gate)
        with self.assertRaisesRegex(RuntimeError, "v20.9 nonstationary"):
            engine_037.run_stage(
                repaired_once, retry_037, software_id="legacy", software_version="20.11"
            )
        failed = self.store.load(investigation.id)
        historical_stages = failed.stages
        historical_files = {
            stage.id: self.store.stage_path_for(failed.id, stage.id).read_bytes()
            for stage in historical_stages
        }

        repaired = repair_obsolete_terminal_wait(self.store, failed)

        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(historical_stages, repaired.stages)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("038-prepare-multi-source-residual", selected["id"])
        self.assertEqual(request.handler_id, selected["handler_id"])
        self.assertEqual(retry_037.id, selected["triggered_by_stage_id"])
        self.assertEqual(
            historical_files,
            {stage.id: self.store.stage_path_for(repaired.id, stage.id).read_bytes()
             for stage in repaired.stages},
        )
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

        def legitimate_new_failure(_investigation, _request):
            raise RuntimeError("v20.12 could not prepare a spatial component dataset.")

        engine_038 = WorkflowEngine(self.store)
        engine_038.register_handler(request.handler_id, legitimate_new_failure)
        retry_038 = StageRequest(**selected)
        with self.assertRaisesRegex(RuntimeError, "spatial component"):
            engine_038.run_stage(
                repaired, retry_038, software_id="current", software_version="20.12"
            )
        new_failure = self.store.load(investigation.id)
        self.assertEqual("NON_RETRYABLE", new_failure.stages[-1].failure_classification)
        self.assertEqual(new_failure, repair_obsolete_terminal_wait(self.store, new_failure))

    def test_legacy_invalid_low_frequency_failure_resumes_independent_branch(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        running = InvestigationStage(
            "006-prepare-followup",
            "openstar.tess.followup.prepare-low-frequency",
            "RUNNING",
            "005-planner",
            {},
        )
        investigation = self.store.append_running_stage(investigation, running)
        failed = self.store.build_terminal_stage(
            stage_id=running.id,
            handler_id=running.handler_id,
            status="FAILED",
            triggered_by_stage_id=running.triggered_by_stage_id,
            parameters={},
            result=None,
            error=(
                "ValueError: Follow-up frequency window is invalid: "
                "0.8123914343264318..0.098."
            ),
            software_id="legacy",
            software_version="20.2",
            started_at=running.started_at,
        )
        investigation = self.store.complete_current_stage(investigation, failed)

        branches = plan_tess_branches(investigation, target)

        self.assertEqual(1, len(branches))
        self.assertEqual(
            "openstar.tess.independent.prepare", branches[0].experiment.handler_id
        )
        self.assertEqual(running.id, branches[0].experiment.triggered_by_stage_id)

    def test_blind_c_v20_28_replay_is_quiescent_and_next_target_is_eligible(self):
        blind_c, next_target = self.source.enumerate_targets()
        investigation = self.store.create(
            blind_c.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=blind_c.metadata,
        )
        # Replay the persisted scientific milestones required by the real
        # v20.28 handler rather than manufacturing only its final stage.
        replay = (
            (
                "001-prepare-target",
                "openstar.tess.prepare-target",
                {"datasetID": "blind-c"},
            ),
            (
                "087-external-high-resolution",
                "openstar.tess.external-high-resolution-variability-validation.interpret",
                {
                    "recommendedNextTest": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
                },
            ),
            (
                "097-atlas-fixed-window",
                "openstar.tess.atlas-fixed-window.interpret",
                {
                    "recommendedNextTest": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
                },
            ),
            (
                "099-targeted-observation-planning",
                "openstar.tess.targeted-observation-planning.generate",
                {
                    "status": "PREREGISTERED",
                    "recommendedNextTest": "INGEST_TARGETED_OBSERVATIONS",
                },
            ),
        )
        for stage_id, handler_id, result in replay:
            investigation = self._complete(investigation, stage_id, handler_id, result)

        branches = plan_tess_branches(investigation, blind_c)
        autonomy = AutonomousInvestigationEngine(self.store)
        decision = autonomy.inspect(investigation, branches)
        self.assertEqual("BLOCKED_EXTERNAL_DATA", decision.branch_assessments[0].state)
        self.assertEqual("QUIESCENT_AWAITING_DATA", decision.investigation_status)
        self.assertEqual("ADVANCE_TO_NEXT_TARGET", decision.scheduler_action)
        self.assertNotEqual(blind_c.investigation_id, next_target.investigation_id)
        self.assertTrue(next_target.eligible)

        investigation, _ = autonomy.decide(investigation, branches)
        workflow = WorkflowEngine(self.store)
        executions = []

        def existing_prepare_handler(current, request):
            executions.append(request.handler_id)
            return StageOutcome(
                result={"datasetID": request.parameters["datasetID"]}, stop=True
            )

        workflow.register_handler(
            "openstar.tess.prepare-target", existing_prepare_handler
        )
        dispatcher = InvestigationDispatcher(self.store, workflow)
        trigger = dispatcher.dispatch(
            investigation.id, software_id="replay", software_version="20.28"
        )
        advanced = InvestigationTargetPortfolio(
            self.root / "portfolio.json", self.store, dispatcher
        ).advance(
            trigger,
            self.source,
            {WORKFLOW_ID: plan_tess_branches},
            software_id="replay",
            software_version="20.28",
        )
        self.assertEqual(next_target.id, advanced.target.id)
        self.assertEqual(["openstar.tess.prepare-target"], executions)

    def test_persisted_v20_27_evidence_selects_registered_v20_28_handler(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        investigation = self._complete(
            investigation,
            "098-atlas-fixed-window",
            "openstar.tess.atlas-fixed-window.interpret",
            {"recommendedNextTest": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"},
        )
        request = plan_tess_branches(investigation, target)[0].experiment
        self.assertEqual(
            "openstar.tess.targeted-observation-planning.generate", request.handler_id
        )

    def test_fresh_target_progresses_stage_by_stage_using_handler_continuations(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        workflow = WorkflowEngine(self.store)
        workflow.chain_stages = False
        calls = []

        def prepare(current, request):
            calls.append(request.handler_id)
            return StageOutcome(
                result={"projectPath": "prepared-primary.json"},
                next_stage=StageRequest(
                    "002-primary-distributed-search",
                    "openstar.tess.primary-project.run",
                    {"projectPath": "prepared-primary.json"},
                    request.id,
                ),
            )

        def primary(current, request):
            calls.append(request.handler_id)
            return StageOutcome(
                result={"status": "COMPLETE"},
                next_stage=StageRequest(
                    "003-catalog-identity",
                    "openstar.tess.catalog-identity",
                    {},
                    request.id,
                ),
            )

        workflow.register_handler("openstar.tess.prepare-target", prepare)
        workflow.register_handler("openstar.tess.primary-project.run", primary)
        dispatcher = InvestigationDispatcher(self.store, workflow)
        autonomy = AutonomousInvestigationEngine(self.store)

        for expected in (
            "openstar.tess.prepare-target",
            "openstar.tess.primary-project.run",
        ):
            investigation = self.store.load(investigation.id)
            investigation, decision = autonomy.decide(
                investigation, plan_tess_branches(investigation, target)
            )
            self.assertEqual(expected, decision.selected_experiment.handler_id)
            dispatcher.dispatch(
                investigation.id, software_id="test", software_version="20.28"
            )

        investigation = self.store.load(investigation.id)
        continuation = plan_tess_branches(investigation, target)[0].experiment
        self.assertEqual("openstar.tess.catalog-identity", continuation.handler_id)
        self.assertEqual(
            ["openstar.tess.prepare-target", "openstar.tess.primary-project.run"],
            calls,
        )

    def test_terminal_finalize_is_complete_but_unknown_state_still_waits(self):
        target = self.source.enumerate_targets()[0]
        terminal = self.store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        terminal = self._complete(
            terminal,
            "006-finalize",
            "openstar.tess.finalize",
            {"scientificConclusion": "ANY_VALID_FINAL_CONCLUSION"},
            stop=True,
        )
        terminal = self.store.load(terminal.id)
        self.assertTrue(terminal.stages[-1].stop)
        terminal_record = json.loads(
            self.store.stage_path_for(terminal.id, "006-finalize").read_text(
                encoding="utf-8"
            )
        )
        self.assertIs(True, terminal_record["stop"])
        autonomy = AutonomousInvestigationEngine(self.store)
        decision = autonomy.inspect(terminal, plan_tess_branches(terminal, target))
        self.assertEqual((), plan_tess_branches(terminal, target))
        self.assertEqual("INVESTIGATION_COMPLETE", decision.scheduler_action)

        unknown_target = self.source.enumerate_targets()[1]
        unknown = self.store.create(
            unknown_target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=unknown_target.metadata,
        )
        unknown = self._complete(
            unknown,
            "006-unknown",
            "openstar.tess.unrecognized-nonterminal",
            {"status": "COMPLETE"},
        )
        decision = autonomy.inspect(
            unknown, plan_tess_branches(unknown, unknown_target)
        )
        self.assertEqual("WAIT_FOR_PREREQUISITES", decision.scheduler_action)
        unknown, _ = autonomy.decide(
            unknown, plan_tess_branches(unknown, unknown_target)
        )
        before_repair = self.store.path_for(unknown.id).read_bytes()
        self.assertEqual(unknown, repair_obsolete_terminal_wait(self.store, unknown))
        self.assertEqual(before_repair, self.store.path_for(unknown.id).read_bytes())
        self.assertEqual("BLOCKED", self.store.load(unknown.id).status)

    def test_legacy_completed_finalize_without_stop_is_terminal(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        investigation = self._complete(
            investigation,
            "006-finalize",
            "openstar.tess.finalize",
            {"scientificConclusion": "LEGACY_FINAL_CONCLUSION"},
        )
        snapshot_path = self.store.path_for(investigation.id)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        del snapshot["stages"][-1]["stop"]
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

        reloaded = self.store.load(investigation.id)
        self.assertFalse(reloaded.stages[-1].stop)
        self.assertEqual((), plan_tess_branches(reloaded, target))

    def test_restart_after_terminal_finalize_advances_once_without_rerun(self):
        finished_target, next_target = self.source.enumerate_targets()
        investigation = self.store.create(
            finished_target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=finished_target.metadata,
        )
        investigation = self._complete(
            investigation,
            "006-finalize",
            "openstar.tess.finalize",
            {"scientificConclusion": "FINAL"},
            stop=True,
        )
        snapshot_path = self.store.path_for(investigation.id)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        del snapshot["stages"][-1]["stop"]
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
        investigation = self.store.load(investigation.id)
        investigation = self.store.set_control_state(
            investigation,
            status="BLOCKED",
            control_state={
                "branchAssessments": [
                    {
                        "branch_id": "unresolved-tess-continuation",
                        "state": "NOT_READY",
                        "missing_stage_ids": ["tess-continuation-decision"],
                        "unavailable_external_data": [],
                    }
                ],
                "selectedExperiment": None,
                "schedulerAction": "WAIT_FOR_PREREQUISITES",
            },
        )

        workflow = WorkflowEngine(self.store)
        workflow.chain_stages = False
        executions = []

        def prepare(current, request):
            executions.append((current.id, request.handler_id))
            return StageOutcome(
                result={"projectPath": "next-primary.json"},
                next_stage=StageRequest(
                    "002-primary-distributed-search",
                    "openstar.tess.primary-project.run",
                    {"projectPath": "next-primary.json"},
                    request.id,
                ),
            )

        workflow.register_handler("openstar.tess.prepare-target", prepare)
        dispatcher = InvestigationDispatcher(self.store, workflow)

        def loop():
            return InvestigationLifecycleLoop(
                self.root / "lifecycle.json",
                self.store,
                dispatcher,
                InvestigationTargetPortfolio(
                    self.root / "portfolio.json", self.store, dispatcher
                ),
                self.source,
                {WORKFLOW_ID: plan_tess_branches},
                software_id="test",
                software_version="20.28",
            )

        loop().start(finished_target)
        # Process restart runs the narrow TESS compatibility repair before the
        # generic lifecycle consumes its durable action.
        repaired = repair_obsolete_terminal_wait(
            self.store, self.store.load(finished_target.investigation_id)
        )
        self.assertEqual("COMPLETE", repaired.status)
        self.assertEqual(
            "INVESTIGATION_COMPLETE",
            self.store.load(finished_target.investigation_id)
            .metadata["controlState"]["schedulerAction"],
        )

        # Reapplying the repair is a no-op, including across another restart.
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

        advanced = loop().run(max_transitions=2)
        self.assertEqual("LIFECYCLE_CHECKPOINT", advanced.disposition)
        state = json.loads(
            (self.root / "lifecycle.json").read_text(encoding="utf-8")
        )
        self.assertEqual(next_target.id, state["currentTarget"]["id"])
        self.assertEqual(
            [(next_target.investigation_id, "openstar.tess.prepare-target")],
            executions,
        )
        portfolio = json.loads(
            (self.root / "portfolio.json").read_text(encoding="utf-8")
        )
        self.assertEqual(1, len(portfolio["selections"]))
        self.assertEqual(
            ["006-finalize"],
            [
                stage.id
                for stage in self.store.load(
                    finished_target.investigation_id
                ).stages
            ],
        )


class TessShiftedStageLookupCompatibilityTests(unittest.TestCase):
    def _failed(self, root, *, error=None):
        store = InvestigationStore(root)
        investigation = store.create(
            "tic-8196173", WORKFLOW_ID, "20.2",
            metadata={"controlState": {"schedulerAction": "INVESTIGATION_FAILED"}},
        )
        stages = (
            InvestigationStage("001-prepare-target", "openstar.tess.prepare-target", "COMPLETE", None, {}, result={"ticID": 8196173}),
            InvestigationStage("002-primary-distributed-search", "openstar.tess.primary-project.run", "COMPLETE", "001-prepare-target", {}, result={"primary": True}),
            InvestigationStage("003-catalog-identity", "openstar.tess.catalog-identity", "FAILED", "002-primary-distributed-search", {}, error="TimeoutError: VSX timeout", failure_classification="TRANSIENT_INFRASTRUCTURE"),
            InvestigationStage("004-catalog-identity", "openstar.tess.catalog-identity", "COMPLETE", "003-catalog-identity", {}, result={"identityResolved": True}),
            InvestigationStage("005-hypotheses", "openstar.tess.hypotheses", "COMPLETE", "004-catalog-identity", {}, result={"observedPeriodDays": 2.0}),
            InvestigationStage("006-planner", "openstar.tess.planner", "COMPLETE", "005-hypotheses", {}, result={"action": "INDEPENDENT_SECTOR_FOLLOWUP"}),
            InvestigationStage(
                "007-prepare-independent-sectors", "openstar.tess.independent.prepare",
                "FAILED", "006-planner", {"preserve": True},
                error=error or "RuntimeError: Stage is not COMPLETE with a result: 003-catalog-identity",
                failure_classification="NON_RETRYABLE",
            ),
        )
        investigation = replace(investigation, status="FAILED", stages=stages)
        store.save(investigation)
        return store, investigation

    def test_repairs_exact_shifted_lookup_failure_without_mutating_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._failed(temporary)
            failed_bytes = json.dumps(
                asdict(investigation.stages[-1]), sort_keys=True
            ).encode()

            repaired = repair_obsolete_terminal_wait(store, investigation)

            self.assertEqual("RUNNING", repaired.status)
            self.assertEqual(investigation.stages, repaired.stages)
            self.assertEqual(
                failed_bytes,
                json.dumps(asdict(repaired.stages[-1]), sort_keys=True).encode(),
            )
            control = repaired.metadata["controlState"]
            self.assertEqual("RUN_EXPERIMENT", control["schedulerAction"])
            self.assertEqual(
                "TESS_RETRY_SHIFTED_STAGE_LOOKUP_COMPATIBILITY_RETRY",
                control["recovery"],
            )
            self.assertEqual(
                {
                    "id": "008-prepare-independent-sectors",
                    "handler_id": "openstar.tess.independent.prepare",
                    "parameters": {"preserve": True},
                    "triggered_by_stage_id": "007-prepare-independent-sectors",
                },
                control["selectedExperiment"],
            )
            self.assertEqual(1, sum(
                stage.handler_id == "openstar.tess.primary-project.run"
                for stage in repaired.stages
            ))
            self.assertEqual(2, sum(
                stage.handler_id == "openstar.tess.catalog-identity"
                for stage in repaired.stages
            ))
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))

    def test_does_not_repair_unrelated_non_retryable_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._failed(temporary, error="RuntimeError: unrelated")
            self.assertEqual(
                investigation, repair_obsolete_terminal_wait(store, investigation)
            )


class TessIndependentPeriodCharacterizationCompatibilityTests(unittest.TestCase):
    def _completed(self, root, *, family=True, characterized=False):
        store = InvestigationStore(root)
        investigation = store.create(
            "tic-25132999", WORKFLOW_ID, "20.3.1",
            metadata={"controlState": {
                "schedulerAction": "INVESTIGATION_COMPLETE",
            }},
        )
        harmonic_family = ({
            "representativeRawPeriodDays": 5.160480186046465,
            "possibleDoubleCycleDays": 10.32096037209293,
            "physicalCycleResolved": False,
        } if family else None)
        stages = [
            InvestigationStage("001-prepare-target", "openstar.tess.prepare-target", "COMPLETE", None, {}, result={"ticID": 25132999}),
            InvestigationStage("002-primary", "openstar.tess.primary-project.run", "COMPLETE", "001-prepare-target", {}, result={"primary": True}),
            InvestigationStage("003-identity", "openstar.tess.catalog-identity", "COMPLETE", "002-primary", {}, result={"catalogued": False}),
            InvestigationStage("004-hypotheses", "openstar.tess.hypotheses", "COMPLETE", "003-identity", {}, result={"rawCandidatePeriodDays": 5.16}),
            InvestigationStage("005-independent-prepare", "openstar.tess.independent.prepare", "COMPLETE", "004-hypotheses", {}, result={"preparedSectors": [94, 96, 97, 98]}),
            InvestigationStage("006-independent-run", "openstar.tess.independent.run", "COMPLETE", "005-independent-prepare", {}, result={"status": "COMPLETE"}),
            InvestigationStage("007-broad-prepare", "openstar.tess.independent.broad.prepare", "COMPLETE", "006-independent-run", {}, result={"prepared": True}),
            InvestigationStage("008-broad-run", "openstar.tess.independent.broad.run", "COMPLETE", "007-broad-prepare", {}, result={"status": "COMPLETE"}),
            InvestigationStage("009-broad-interpret", "openstar.tess.independent.broad.interpret", "COMPLETE", "008-broad-run", {}, result={
                "claimDecision": {"claim": "INDEPENDENT_PERIOD_ESTIMATE"},
                "promotionEligible": True,
                "harmonicFamily": harmonic_family,
            }),
        ]
        if characterized:
            stages.append(InvestigationStage(
                "010-morphology", "openstar.tess.morphology.analyze", "COMPLETE",
                "009-broad-interpret", {}, result={"physicalCycleResolved": False},
            ))
        stages.append(InvestigationStage(
            "011-finalize" if characterized else "010-finalize",
            "openstar.tess.finalize", "COMPLETE",
            "010-morphology" if characterized else "009-broad-interpret", {},
            result={"claim": {"claim": "INDEPENDENT_PERIOD_ESTIMATE"}}, stop=True,
        ))
        investigation = replace(investigation, status="COMPLETE", stages=tuple(stages))
        store.save(investigation)
        return store, investigation

    def test_reopens_real_shaped_terminal_directly_to_morphology(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._completed(temporary)
            historical_stages = investigation.stages

            repaired = repair_obsolete_terminal_wait(store, investigation)

            self.assertEqual("RUNNING", repaired.status)
            self.assertEqual(historical_stages, repaired.stages)
            control = repaired.metadata["controlState"]
            self.assertEqual("RUN_EXPERIMENT", control["schedulerAction"])
            self.assertEqual(
                "TESS_INDEPENDENT_PERIOD_CHARACTERIZATION_COMPATIBILITY_CONTINUATION",
                control["recovery"],
            )
            self.assertEqual({
                "id": "011-morphology",
                "handler_id": "openstar.tess.morphology.analyze",
                "parameters": {},
                "triggered_by_stage_id": "009-broad-interpret",
            }, control["selectedExperiment"])
            for handler in (
                "openstar.tess.prepare-target",
                "openstar.tess.primary-project.run",
                "openstar.tess.independent.prepare",
                "openstar.tess.independent.run",
                "openstar.tess.independent.broad.run",
            ):
                self.assertEqual(1, sum(
                    stage.handler_id == handler for stage in repaired.stages
                ))
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))

    def test_does_not_reopen_already_characterized_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._completed(temporary, characterized=True)
            self.assertEqual(
                investigation, repair_obsolete_terminal_wait(store, investigation)
            )

    def test_does_not_reopen_without_valid_harmonic_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._completed(temporary, family=False)
            self.assertEqual(
                investigation, repair_obsolete_terminal_wait(store, investigation)
            )


if __name__ == "__main__":
    unittest.main()
