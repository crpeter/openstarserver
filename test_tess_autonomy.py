import json
import shutil
import tempfile
import unittest
from unittest import mock
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
    _repair_official_prf_transport_terminal,
    _repair_resolved_family_multisource_failure,
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

    def _historical_blocked_prf(self, errors=None, **interpret_overrides):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        evidence = (
            ("071-prf-prepare", "openstar.tess.official-spoc-prf-forward-modeling.prepare",
             {"version": "openstar.tess-prf-deblending.v1"}, None),
            ("072-prf-run", "openstar.tess.official-spoc-prf-forward-modeling.run",
             {"sectorResults": [], "errors": errors if errors is not None else [
                 {"sector": 2, "error": "TimeoutError: The read operation timed out"},
                 {"sector": 29, "error": "URLError: <urlopen error [Errno 60] Operation timed out>"},
                 {"sector": 68, "error": "URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred>"},
                 {"sector": 69, "error": "URLError: <urlopen error timed out>"},
             ]}, "071-prf-prepare"),
            ("073-prf-interpret", "openstar.tess.official-spoc-prf-forward-modeling.interpret",
             {"classification": "BLOCKED_EXTERNAL_DATA",
              "recommendedNextTest": "RETRY_PIXEL_RESPONSE_FUNCTION_DEBLENDING",
              "physicalMechanismResolved": False, **interpret_overrides}, "072-prf-run"),
            ("074-finalize", "openstar.tess.finalize", {"claim": {}}, "073-prf-interpret"),
        )
        for stage_id, handler, result, trigger in evidence:
            running = InvestigationStage(stage_id, handler, "RUNNING", trigger, {})
            investigation = self.store.append_running_stage(investigation, running)
            terminal = self.store.build_terminal_stage(
                stage_id=stage_id, handler_id=handler, status="COMPLETE",
                triggered_by_stage_id=trigger, parameters={}, result=result, error=None,
                software_id="historical", software_version="20.18",
                started_at=running.started_at, stop=handler == "openstar.tess.finalize",
            )
            investigation = self.store.complete_current_stage(investigation, terminal)
        return self.store.set_control_state(
            investigation, status="COMPLETE", control_state={
                "branchAssessments": [], "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            },
        )

    def test_historical_prf_transport_recovery_is_append_only_and_idempotent(self):
        investigation = self._historical_blocked_prf()
        stage_files = {
            path: path.read_bytes()
            for path in self.store.directory_for(investigation.id).joinpath("stages").iterdir()
        }
        repaired = repair_obsolete_terminal_wait(self.store, investigation)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual("openstar.tess.official-spoc-prf-forward-modeling.run",
                         selected["handler_id"])
        self.assertEqual({}, selected["parameters"])
        self.assertEqual("073-prf-interpret", selected["triggered_by_stage_id"])
        self.assertEqual(investigation.stages, repaired.stages)
        self.assertEqual(stage_files, {path: path.read_bytes() for path in stage_files})
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

    def test_historical_prf_recovery_fails_closed(self):
        cases = (
            ({"errors": [{"sector": 2, "error": "RuntimeError: invalid PRF grid"}]}, {}),
            ({"errors": []}, {}),
            ({}, {"classification": "UNRESOLVED"}),
            ({}, {"recommendedNextTest": "OTHER"}),
            ({}, {"physicalMechanismResolved": True}),
        )
        for position, (history_kwargs, overrides) in enumerate(cases):
            with self.subTest(position=position):
                # Each history needs a distinct durable investigation id.
                investigation = self._historical_blocked_prf(
                    **history_kwargs, **overrides
                )
                control = investigation.metadata["controlState"]
                self.assertIsNone(_repair_official_prf_transport_terminal(
                    self.store, investigation, control
                ))
                # Remove this case before creating the next identical test id.
                shutil.rmtree(self.store.directory_for(investigation.id))

        investigation = self._historical_blocked_prf()
        control = investigation.metadata["controlState"]
        run = investigation.stages[1]
        with_sector_result = replace(
            investigation,
            stages=(investigation.stages[0], replace(
                run, result={**run.result, "sectorResults": [{"sector": 2}]}
            ), *investigation.stages[2:]),
        )
        broken_lineage = replace(
            investigation,
            stages=(*investigation.stages[:2], replace(
                investigation.stages[2], triggered_by_stage_id="unrelated-run"
            ), investigation.stages[3]),
        )
        later_attempt = replace(
            investigation,
            stages=investigation.stages + (InvestigationStage(
                "075-prf-run", "openstar.tess.official-spoc-prf-forward-modeling.run",
                "COMPLETE", investigation.stages[2].id, {}, result={}
            ),),
        )
        inconsistent_control = {**control, "selectedExperiment": {"id": "other"}}
        for snapshot, snapshot_control in (
            (with_sector_result, control),
            (broken_lineage, control),
            (later_attempt, control),
            (replace(investigation, status="RUNNING"), control),
            (investigation, inconsistent_control),
        ):
            self.assertIsNone(_repair_official_prf_transport_terminal(
                self.store, snapshot, snapshot_control
            ))

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

    def test_stage_053_variable_multisource_schedules_time_resolved_once_append_only(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(
            target.investigation_id, WORKFLOW_ID, WORKFLOW_VERSION,
            metadata={**target.metadata, "ticID": 277940827})
        for number in range(1, 51):
            investigation = self._complete(
                investigation, f"{number:03d}-persisted-science", "persisted.science",
                {"stage": number})
        investigation = self._complete(
            investigation, "051-prepare-source-switching-temporal-model",
            "openstar.tess.source-switching-temporal-model.prepare", {})
        investigation = self._complete(
            investigation, "052-run-source-switching-temporal-model",
            "openstar.tess.source-switching-temporal-model.run", {})
        investigation = self._complete(
            investigation, "053-interpret-source-switching-temporal-model",
            "openstar.tess.source-switching-temporal-model.interpret",
            {"classification": "SECTOR_VARIABLE_MULTI_SOURCE",
             "sourceIdentifiable": True, "sourceAttributionResolved": False,
             "physicalMechanismResolved": False,
             "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA"})
        original = investigation.stages
        branches = plan_tess_branches(investigation, target)
        self.assertEqual(1, len(branches))
        request = branches[0].experiment
        self.assertEqual("054-prepare-time-resolved-residual-phase-localization", request.id)
        self.assertEqual("openstar.tess.time-resolved-residual-phase-localization.prepare",
                         request.handler_id)
        self.assertEqual(original, investigation.stages)
        running = InvestigationStage(request.id, request.handler_id, "RUNNING",
                                     request.triggered_by_stage_id, {})
        with_running = self.store.append_running_stage(investigation, running)
        self.assertEqual((), plan_tess_branches(with_running, target))

    def test_stage_056_candidate_recovery_schedules_validation_exactly_once(self):
        target = self.source.enumerate_targets()[0]
        for classification, tic_id in (("STABLE_CANDIDATE_1_LOCALIZATION", 111),
                                       ("STABLE_CANDIDATE_2_LOCALIZATION", 222)):
            with self.subTest(classification=classification):
                candidate = {"raDeg": 10.1, "decDeg": -20.1,
                             "catalogIDs": {"ticID": tic_id}}
                investigation = self.store.create(
                    f"{target.investigation_id}-{tic_id}", WORKFLOW_ID, WORKFLOW_VERSION,
                    metadata={**target.metadata, "ticID": 277940827})
                investigation = self._complete(
                    investigation, "053-interpret-source-switching-temporal-model",
                    "openstar.tess.source-switching-temporal-model.interpret",
                    {"classification": "SECTOR_VARIABLE_MULTI_SOURCE",
                     "sourceIdentifiable": True, "sourceAttributionResolved": False,
                     "physicalMechanismResolved": False,
                     "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA"})
                investigation = self._complete(
                    investigation, "056-interpret-time-resolved-residual-phase-localization",
                    "openstar.tess.time-resolved-residual-phase-localization.interpret",
                    {"classification": classification, "sourceAttributionResolved": True,
                     "physicalMechanismResolved": False, "preferredCandidate": candidate,
                     "catalogCandidates": [candidate, {"raDeg": 11., "decDeg": -21.}],
                     "recommendedNextTest":
                     "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"})
                branches = plan_tess_branches(investigation, target)
                self.assertEqual(1, len(branches))
                request = branches[0].experiment
                self.assertEqual("057-prepare-offset-source-variability", request.id)
                self.assertEqual("openstar.tess.offset-source-variability.prepare",
                                 request.handler_id)
                running = InvestigationStage(request.id, request.handler_id, "RUNNING",
                                             request.triggered_by_stage_id, {})
                with_running = self.store.append_running_stage(investigation, running)
                self.assertEqual((), plan_tess_branches(with_running, target))
                terminal = self.store.build_terminal_stage(
                    stage_id=request.id, handler_id=request.handler_id, status="COMPLETE",
                    triggered_by_stage_id=request.triggered_by_stage_id, parameters={}, result={},
                    error=None, software_id="test", software_version="1",
                    started_at=running.started_at)
                with_complete = self.store.complete_current_stage(with_running, terminal)
                self.assertEqual((), plan_tess_branches(with_complete, target))

    def test_real_stage_056_time_variable_result_schedules_independent_followup_once(self):
        target = self.source.enumerate_targets()[0]
        investigation = self.store.create(target.investigation_id, WORKFLOW_ID,
            WORKFLOW_VERSION, metadata={**target.metadata, "ticID": 277940827})
        for number in range(1, 57):
            result = ({"classification": "TIME_VARIABLE_LOCALIZATION",
                "sourceAttributionResolved": False, "physicalMechanismResolved": False,
                "recommendedNextTest": "TIME_VARIABLE_SOURCE_LOCALIZATION_FOLLOWUP"}
                if number == 56 else {"persistedStage": number})
            handler = ("openstar.tess.time-resolved-residual-phase-localization.interpret"
                if number == 56 else "persisted.science")
            investigation = self._complete(investigation, f"{number:03d}-persisted", handler, result)
        original = investigation.stages
        branches = plan_tess_branches(investigation, target)
        self.assertEqual(1, len(branches))
        request = branches[0].experiment
        self.assertEqual("057-prepare-time-resolved-frequency-localization", request.id)
        self.assertEqual("openstar.tess.time-resolved-frequency-localization.prepare", request.handler_id)
        self.assertEqual(original, investigation.stages)
        running = InvestigationStage(request.id, request.handler_id, "RUNNING",
                                     request.triggered_by_stage_id, {})
        with_running = self.store.append_running_stage(investigation, running)
        self.assertEqual((), plan_tess_branches(with_running, target))
        self.assertEqual(original, with_running.stages[:56])

    def test_stage_059_candidate_recovery_schedules_stage_060_exactly_once(self):
        target = self.source.enumerate_targets()[0]
        for number, catalog_id in ((1,111),(2,222)):
            candidate={"raDeg":10.+number/10,"decDeg":-20.-number/10,
                       "catalogIDs":{"ticID":catalog_id}}
            investigation=self.store.create(f"{target.investigation_id}-stage59-{number}",
                WORKFLOW_ID,WORKFLOW_VERSION,metadata={**target.metadata,"ticID":277940827})
            investigation=self._complete(investigation,
                "056-interpret-time-resolved-residual-phase-localization",
                "openstar.tess.time-resolved-residual-phase-localization.interpret",
                {"classification":"TIME_VARIABLE_LOCALIZATION","sourceAttributionResolved":False,
                 "physicalMechanismResolved":False,
                 "recommendedNextTest":"TIME_VARIABLE_SOURCE_LOCALIZATION_FOLLOWUP"})
            investigation=self._complete(investigation,
                "059-interpret-time-resolved-frequency-localization",
                "openstar.tess.time-resolved-frequency-localization.interpret",
                {"classification":f"STABLE_CANDIDATE_{number}_LOCALIZATION",
                 "sourceAttributionResolved":True,"physicalMechanismResolved":False,
                 "preferredCandidate":candidate,
                 "recommendedNextTest":"INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"})
            branches=plan_tess_branches(investigation,target)
            self.assertEqual(1,len(branches)); request=branches[0].experiment
            self.assertEqual("060-prepare-offset-source-variability",request.id)
            running=InvestigationStage(request.id,request.handler_id,"RUNNING",
                                       request.triggered_by_stage_id,{})
            with_running=self.store.append_running_stage(investigation,running)
            self.assertEqual((),plan_tess_branches(with_running,target))
            terminal=self.store.build_terminal_stage(stage_id=request.id,
                handler_id=request.handler_id,status="COMPLETE",
                triggered_by_stage_id=request.triggered_by_stage_id,parameters={},result={},
                error=None,software_id="test",software_version="1",started_at=running.started_at)
            with_complete=self.store.complete_current_stage(with_running,terminal)
            self.assertEqual((),plan_tess_branches(with_complete,target))

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
              "establishedPeriodFamily": {"referencePeriodDays": 10.30084080080649,
                                            "modeledHarmonicOrders": [1, 2, 3, 4]},
              "modeCandidate": {"frequencyCyclesPerDay": 1 / 2.206,
                                "periodDays": 2.206,
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
        failed_request = investigation.stages[-1]
        investigation = self.store.set_control_state(
            investigation, status="FAILED",
            control_state={"schedulerAction": "RUN_EXPERIMENT", "selectedExperiment": {
                "id": failed_request.id, "handler_id": failed_request.handler_id,
                "parameters": failed_request.parameters,
                "triggered_by_stage_id": failed_request.triggered_by_stage_id,
            }},
        )
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

    def test_real_scheduler_chained_stage_022_failures_recover_append_only(self):
        errors = (
            "RuntimeError: v20.11 requires the morphology-resolved physical period.",
            "RuntimeError: v20.11 requires the completed v20.9 nonstationary model.",
        )
        for offset, error in enumerate(errors):
            with self.subTest(error=error):
                investigation = self.store.create(
                    f"real-stage-022-{offset}", WORKFLOW_ID, WORKFLOW_VERSION,
                    metadata={},
                )
                resolved = offset == 1
                family_period = 14.636494965204527 if not resolved else 10.510316195053623
                orders = [1, 2, 4] if not resolved else [1, 2, 3]
                frequency = 0.27628980191811653 if not resolved else 0.27101611598985065
                stages = (
                    InvestigationStage("010-morphology", "openstar.tess.morphology.analyze", "COMPLETE", None, {}, result={"physicalCycleResolved": resolved, "resolvedPhysicalPeriodDays": family_period if resolved else None}),
                    InvestigationStage("012-time-frequency-prepare", "openstar.tess.time-frequency.prepare", "COMPLETE", "010-morphology", {}, result={"subtractedHarmonicOrders": orders, "absoluteTimeReferenceDays": 2500.0}),
                    InvestigationStage("013-time-frequency-summary", "openstar.tess.time-frequency.summarize", "COMPLETE", "012-time-frequency-prepare", {}, result={"residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"}}),
                    InvestigationStage("018-mode-identification", "openstar.tess.mode-identification.analyze", "COMPLETE", "013-time-frequency-summary", {}, result={"classification": "INDEPENDENT_STABLE_MODE", "independentModeEvidenceSurvived": True, "physicalMechanismResolved": False, "recommendedNextTest": "RESIDUAL_MODE_PIXEL_LOCALIZATION", "establishedPeriodFamily": {"referencePeriodDays": family_period, "modeledHarmonicOrders": orders}, "modeCandidate": {"frequencyCyclesPerDay": frequency, "periodDays": 1 / frequency, "supportingSectors": [2, 29, 68, 69]}}),
                    InvestigationStage("019-prepare-residual-mode-localization", "openstar.tess.residual-mode-localization.prepare", "COMPLETE", "018-mode-identification", {}, result={"subtractedHarmonicOrders": orders}),
                    InvestigationStage("020-run-residual-mode-localization", "openstar.tess.residual-mode-localization.run", "COMPLETE", "019-prepare-residual-mode-localization", {}, result={"status": "COMPLETE"}),
                    InvestigationStage("021-interpret-residual-mode-localization", "openstar.tess.residual-mode-localization.interpret", "COMPLETE", "020-run-residual-mode-localization", {}, result={"recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"}),
                    InvestigationStage("022-prepare-residual-mode-localization-review", "openstar.tess.residual-mode-localization-review.prepare", "FAILED", "021-interpret-residual-mode-localization", {}, error=error, failure_classification="NON_RETRYABLE"),
                )
                selected = stages[-1]
                investigation = replace(
                    investigation, status="FAILED", stages=stages,
                    metadata={"controlState": {
                        "schedulerAction": "RUN_EXPERIMENT",
                        "selectedExperiment": {
                            "id": selected.id, "handler_id": selected.handler_id,
                            "parameters": selected.parameters,
                            "triggered_by_stage_id": selected.triggered_by_stage_id,
                        },
                    }},
                )
                self.store.save(investigation)
                historical = investigation.stages

                if resolved:
                    inconsistent = replace(
                        investigation,
                        stages=(replace(
                            investigation.stages[0],
                            result={"physicalCycleResolved": True,
                                    "resolvedPhysicalPeriodDays": family_period + 1.0},
                        ),) + investigation.stages[1:],
                    )
                    self.assertEqual(
                        inconsistent,
                        repair_obsolete_terminal_wait(self.store, inconsistent),
                    )

                repaired = repair_obsolete_terminal_wait(self.store, investigation)
                self.assertEqual("RUNNING", repaired.status)
                self.assertEqual(historical, repaired.stages)
                request = repaired.metadata["controlState"]["selectedExperiment"]
                self.assertEqual("023-prepare-residual-mode-localization-review", request["id"])

                workflow = WorkflowEngine(self.store)
                workflow.register_handler(
                    request["handler_id"],
                    lambda _investigation, _request: StageOutcome(
                        {"reviewPreparedFromPersistedEvidence": True}, stop=True
                    ),
                )
                dispatcher = InvestigationDispatcher(self.store, workflow)
                result = dispatcher.dispatch(
                    repaired.id, software_id="test", software_version="current"
                )
                self.assertEqual("EXPERIMENT_DISPATCHED", result.disposition)
                completed = self.store.load(repaired.id)
                self.assertEqual(historical, completed.stages[:-1])
                self.assertEqual(request["id"], completed.stages[-1].id)
                self.assertEqual(completed, repair_obsolete_terminal_wait(self.store, completed))
                restarted = dispatcher.dispatch(
                    repaired.id, software_id="test", software_version="current"
                )
                self.assertEqual("EXPERIMENT_ALREADY_DISPATCHED", restarted.disposition)
                self.assertEqual(len(historical) + 1, len(self.store.load(repaired.id).stages))

    def test_archive_timeout_stage_023_lineages_recover_append_only_and_fail_closed(self):
        for resolved in (False, True):
            with self.subTest(resolved=resolved):
                family_period = 10.510316195053623 if resolved else 14.636494965204527
                orders = [1, 2, 3] if resolved else [1, 2, 4]
                frequency = 0.27101611598985065 if resolved else 0.27628980191811653
                investigation = self.store.create(
                    f"archive-timeout-{'resolved' if resolved else 'unresolved'}",
                    WORKFLOW_ID, WORKFLOW_VERSION,
                )
                stages = (
                    InvestigationStage("010-morphology", "openstar.tess.morphology.analyze", "COMPLETE", None, {}, result={"physicalCycleResolved": resolved, "resolvedPhysicalPeriodDays": family_period if resolved else None}),
                    InvestigationStage("012-time-frequency-prepare", "openstar.tess.time-frequency.prepare", "COMPLETE", "010-morphology", {}, result={"absoluteTimeReferenceDays": 2500.0}),
                    InvestigationStage("013-time-frequency-summary", "openstar.tess.time-frequency.summarize", "COMPLETE", "012-time-frequency-prepare", {}, result={"residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"}}),
                    InvestigationStage("018-mode-identification", "openstar.tess.mode-identification.analyze", "COMPLETE", "013-time-frequency-summary", {}, result={"classification": "INDEPENDENT_STABLE_MODE", "independentModeEvidenceSurvived": True, "physicalMechanismResolved": False, "establishedPeriodFamily": {"referencePeriodDays": family_period, "modeledHarmonicOrders": orders}, "modeCandidate": {"frequencyCyclesPerDay": frequency, "periodDays": 1 / frequency, "supportingSectors": [1, 28]}}),
                    InvestigationStage("019-prepare-residual-mode-localization", "openstar.tess.residual-mode-localization.prepare", "COMPLETE", "018-mode-identification", {}, result={"subtractedHarmonicOrders": [1, 2]}),
                    InvestigationStage("020-run-residual-mode-localization", "openstar.tess.residual-mode-localization.run", "COMPLETE", "019-prepare-residual-mode-localization", {}, result={}),
                    InvestigationStage("021-interpret-residual-mode-localization", "openstar.tess.residual-mode-localization.interpret", "COMPLETE", "020-run-residual-mode-localization", {}, result={"recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"}),
                    InvestigationStage("022-prepare-residual-mode-localization-review", "openstar.tess.residual-mode-localization-review.prepare", "FAILED", "021-interpret-residual-mode-localization", {}, error=("RuntimeError: v20.11 requires the completed v20.9 nonstationary model." if resolved else "RuntimeError: v20.11 requires the morphology-resolved physical period."), failure_classification="NON_RETRYABLE"),
                    InvestigationStage("023-prepare-residual-mode-localization", "openstar.tess.residual-mode-localization.prepare", "FAILED", "022-prepare-residual-mode-localization-review", {}, error="RuntimeError: v20.10 could not prepare any residual-mode pixel datasets.", failure_classification="NON_RETRYABLE"),
                )
                selected = stages[-1]
                failed = replace(investigation, status="FAILED", stages=stages, metadata={
                    "controlState": {"schedulerAction": "RUN_EXPERIMENT",
                                     "selectedExperiment": asdict(StageRequest(
                                         selected.id, selected.handler_id,
                                         selected.parameters,
                                         selected.triggered_by_stage_id))}
                })
                self.store.save(failed)
                # Materialize the two immutable ledger records represented by
                # this production-history fixture.
                for stage in stages[-2:]:
                    self.store._atomic_write_json(
                        self.store.stage_path_for(failed.id, stage.id),
                        asdict(stage), replace=False,
                    )
                bytes_022 = self.store.stage_path_for(failed.id, stages[-2].id).read_bytes()
                bytes_023 = self.store.stage_path_for(failed.id, stages[-1].id).read_bytes()

                repaired = repair_obsolete_terminal_wait(self.store, failed)
                request = repaired.metadata["controlState"]["selectedExperiment"]
                self.assertEqual("RUNNING", repaired.status)
                self.assertEqual("024-prepare-residual-mode-localization", request["id"])
                self.assertEqual(stages[-1].id, request["triggered_by_stage_id"])
                self.assertEqual(stages, repaired.stages)
                self.assertEqual(bytes_022, self.store.stage_path_for(failed.id, stages[-2].id).read_bytes())
                self.assertEqual(bytes_023, self.store.stage_path_for(failed.id, stages[-1].id).read_bytes())
                self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

                # The registered production workflow recognizes the selected handler.
                from workflows.tess.tess_investigation import build_engine
                engine = build_engine(
                    self.store, mock.Mock(), poll_interval=0.0, timeout=1.0
                )
                self.assertIn(request["handler_id"], engine.handlers)

                for changed in (
                    replace(failed, stages=stages[:-1] + (replace(stages[-1], triggered_by_stage_id="unrelated"),)),
                    replace(failed, stages=(replace(stages[0], result={"physicalCycleResolved": True, "resolvedPhysicalPeriodDays": family_period + 1}),) + stages[1:]),
                    replace(failed, stages=stages[:-2] + (replace(stages[-2], handler_id="openstar.tess.other.prepare"), stages[-1])),
                ):
                    self.assertEqual(changed, repair_obsolete_terminal_wait(self.store, changed))

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
             {"classification": "ADDITIONAL_VARIABILITY_REMAINS"}),
            ("022-time-frequency-prepare", "openstar.tess.time-frequency.prepare",
             {"absoluteTimeReferenceDays": 2500.0}),
            ("023-time-frequency-summary", "openstar.tess.time-frequency.summarize",
             {"residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"}}),
            ("024-mode", "openstar.tess.mode-identification.analyze", {
                "independentModeEvidenceSurvived": True,
                "physicalMechanismResolved": False,
                "establishedPeriodFamily": {
                    "referencePeriodDays": 10.30084080080649,
                    "modeledHarmonicOrders": [1, 2, 3, 4]},
                "modeCandidate": {
                    "frequencyCyclesPerDay": 1 / 2.2071724078510457,
                    "periodDays": 2.2071724078510457,
                    "supportingSectors": [94, 95, 102, 103]},
             }),
            ("024-prepare-localization", "openstar.tess.residual-mode-localization.prepare", {}),
            ("025-run-localization", "openstar.tess.residual-mode-localization.run", {}),
            ("026-localization", "openstar.tess.residual-mode-localization.interpret",
             {"recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"}),
            ("027-review-prepare", "openstar.tess.residual-mode-localization-review.prepare", {}),
            ("028-review-run", "openstar.tess.residual-mode-localization-review.run", {}),
            ("029-review", "openstar.tess.residual-mode-localization-review.interpret",
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
        lineage = {
            "025-run-localization": "024-prepare-localization",
            "026-localization": "025-run-localization",
            "027-review-prepare": "026-localization",
            "028-review-run": "027-review-prepare",
            "029-review": "028-review-run",
        }
        investigation = replace(
            investigation,
            stages=tuple(replace(stage, triggered_by_stage_id=lineage.get(stage.id))
                         for stage in investigation.stages),
        )
        self.store.save(investigation)
        request = StageRequest(
            "030-prepare-multi-source-residual",
            "openstar.tess.multi-source-residual.prepare", {}, "029-review",
        )
        investigation = self.store.set_control_state(
            investigation, status="RUNNING",
            control_state={"schedulerAction": "RUN_EXPERIMENT",
                           "selectedExperiment": asdict(request)},
        )
        engine = WorkflowEngine(self.store)

        def obsolete_gate(_investigation, _request):
            raise RuntimeError(
                "v20.12 requires either a morphology-resolved physical period or "
                "an established unresolved dynamic harmonic family."
            )

        engine.register_handler(request.handler_id, obsolete_gate)
        with self.assertRaisesRegex(RuntimeError, "morphology-resolved"):
            engine.run_stage(
                investigation, request, software_id="legacy", software_version="20.11"
            )
        failed = self.store.load(investigation.id)
        historical_stages = failed.stages
        historical_files = {
            stage.id: self.store.stage_path_for(failed.id, stage.id).read_bytes()
            for stage in historical_stages
        }
        unrelated_failed_stage = replace(
            failed.stages[-1], triggered_by_stage_id="unrelated-completed-stage"
        )
        unrelated_lineage = replace(
            failed,
            stages=failed.stages[:-1] + (unrelated_failed_stage,),
            metadata={
                **failed.metadata,
                "controlState": {
                    **failed.metadata["controlState"],
                    "selectedExperiment": asdict(StageRequest(
                        unrelated_failed_stage.id,
                        unrelated_failed_stage.handler_id,
                        unrelated_failed_stage.parameters,
                        unrelated_failed_stage.triggered_by_stage_id,
                    )),
                },
            },
        )
        self.assertEqual(
            unrelated_lineage,
            repair_obsolete_terminal_wait(self.store, unrelated_lineage),
        )

        repaired = repair_obsolete_terminal_wait(self.store, failed)

        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(historical_stages, repaired.stages)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("031-prepare-multi-source-residual", selected["id"])
        self.assertEqual(request.handler_id, selected["handler_id"])
        self.assertEqual(request.id, selected["triggered_by_stage_id"])
        self.assertEqual(
            historical_files,
            {stage.id: self.store.stage_path_for(repaired.id, stage.id).read_bytes()
             for stage in repaired.stages},
        )
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

        def legitimate_new_failure(_investigation, _request):
            raise RuntimeError("v20.12 could not prepare a spatial component dataset.")

        engine_031 = WorkflowEngine(self.store)
        engine_031.register_handler(request.handler_id, legitimate_new_failure)
        retry_031 = StageRequest(**selected)
        with self.assertRaisesRegex(RuntimeError, "spatial component"):
            engine_031.run_stage(
                repaired, retry_031, software_id="current", software_version="20.12"
            )
        new_failure = self.store.load(investigation.id)
        self.assertEqual("NON_RETRYABLE", new_failure.stages[-1].failure_classification)
        self.assertEqual(new_failure, repair_obsolete_terminal_wait(self.store, new_failure))

    def test_resolved_family_v2012_failure_repairs_append_only_and_fails_closed(self):
        investigation = self.store.create(
            "tess-discovery-sector-1-tic-29495621", WORKFLOW_ID, WORKFLOW_VERSION,
        )
        family_period = 10.510316195053623
        residual_frequency = 0.27101611598985065
        evidence = (
            ("010-morphology", "openstar.tess.morphology.analyze", {
                "physicalCycleResolved": True,
                "resolvedPhysicalPeriodDays": family_period,
            }),
            ("021-time-frequency-prepare", "openstar.tess.time-frequency.prepare", {
                "absoluteTimeReferenceDays": 2500.0,
            }),
            ("022-time-frequency-summary", "openstar.tess.time-frequency.summarize", {
                "residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"},
            }),
            ("023-mode", "openstar.tess.mode-identification.analyze", {
                "independentModeEvidenceSurvived": True,
                "physicalMechanismResolved": False,
                "establishedPeriodFamily": {
                    "referencePeriodDays": family_period,
                    "modeledHarmonicOrders": [1, 2, 3],
                },
                "modeCandidate": {
                    "frequencyCyclesPerDay": residual_frequency,
                    "periodDays": 1 / residual_frequency,
                    "supportingSectors": [28, 68, 92, 95],
                },
            }),
            ("025-prepare-residual-mode-localization",
             "openstar.tess.residual-mode-localization.prepare", {}),
            ("026-run-residual-mode-localization",
             "openstar.tess.residual-mode-localization.run", {}),
            ("027-interpret-residual-mode-localization",
             "openstar.tess.residual-mode-localization.interpret", {
                 "recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW",
             }),
            ("028-prepare-residual-mode-localization-review",
             "openstar.tess.residual-mode-localization-review.prepare", {}),
            ("029-run-residual-mode-localization-review",
             "openstar.tess.residual-mode-localization-review.run", {}),
            ("030-interpret-residual-mode-localization-review",
             "openstar.tess.residual-mode-localization-review.interpret", {
                 "crossTime": {
                     "classification": "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND",
                     "residualModeOrigin": "TIME_VARIABLE_OR_BLENDED",
                 },
                 "recommendedNextTest": "MULTI_SOURCE_RESIDUAL_DECOMPOSITION",
             }),
        )
        for stage_id, handler, result in evidence:
            investigation = self._complete(investigation, stage_id, handler, result)
        lineage = {
            "026-run-residual-mode-localization":
                "025-prepare-residual-mode-localization",
            "027-interpret-residual-mode-localization":
                "026-run-residual-mode-localization",
            "028-prepare-residual-mode-localization-review":
                "027-interpret-residual-mode-localization",
            "029-run-residual-mode-localization-review":
                "028-prepare-residual-mode-localization-review",
            "030-interpret-residual-mode-localization-review":
                "029-run-residual-mode-localization-review",
        }
        investigation = replace(
            investigation,
            stages=tuple(replace(stage, triggered_by_stage_id=lineage.get(stage.id))
                         for stage in investigation.stages),
        )
        self.store.save(investigation)
        request = StageRequest(
            "031-prepare-multi-source-residual",
            "openstar.tess.multi-source-residual.prepare", {},
            "030-interpret-residual-mode-localization-review",
        )
        investigation = self.store.set_control_state(
            investigation, status="RUNNING",
            control_state={"schedulerAction": "RUN_EXPERIMENT",
                           "selectedExperiment": asdict(request)},
        )
        engine = WorkflowEngine(self.store)

        def obsolete_gate(_investigation, _request):
            raise RuntimeError(
                "v20.12 requires the completed v20.9 nonstationary model."
            )

        engine.register_handler(request.handler_id, obsolete_gate)
        with self.assertRaisesRegex(RuntimeError, "completed v20.9"):
            engine.run_stage(
                investigation, request, software_id="legacy", software_version="20.12"
            )
        failed = self.store.load(investigation.id)
        historical_stages = failed.stages
        historical_files = {
            stage.id: self.store.stage_path_for(failed.id, stage.id).read_bytes()
            for stage in historical_stages
        }

        def changed(*, stage=None, stages=None, selected=None):
            updated_stages = stages or (
                failed.stages[:-1] + (stage or failed.stages[-1],)
            )
            updated_selected = selected
            if updated_selected is None:
                tail = updated_stages[-1]
                updated_selected = asdict(StageRequest(
                    tail.id, tail.handler_id, tail.parameters,
                    tail.triggered_by_stage_id,
                ))
            return replace(
                failed, stages=updated_stages,
                metadata={**failed.metadata, "controlState": {
                    **failed.metadata["controlState"],
                    "selectedExperiment": updated_selected,
                }},
            )

        tail = failed.stages[-1]
        review_index = next(i for i, stage in enumerate(failed.stages)
                            if stage.id.startswith("030-"))
        morphology_index = next(i for i, stage in enumerate(failed.stages)
                                if stage.handler_id == "openstar.tess.morphology.analyze")
        mode_index = next(i for i, stage in enumerate(failed.stages)
                          if stage.handler_id == "openstar.tess.mode-identification.analyze")
        cases = {
            "wrong error": changed(stage=replace(tail, error="RuntimeError: other")),
            "wrong handler": changed(stage=replace(tail, handler_id="other.handler")),
            "wrong classification": changed(
                stage=replace(tail, failure_classification="TRANSIENT_INFRASTRUCTURE")),
            "mismatched selection": changed(selected={**asdict(request), "parameters": {"x": 1}}),
            "unrelated trigger": changed(
                stage=replace(tail, triggered_by_stage_id="unrelated-stage")),
            "broken chain": changed(stages=tuple(
                replace(stage, triggered_by_stage_id="broken")
                if stage.id == "029-run-residual-mode-localization-review" else stage
                for stage in failed.stages
            )),
            "missing review": changed(stages=failed.stages[:review_index]
                                      + failed.stages[review_index + 1:]),
            "wrong recommendation": changed(stages=tuple(
                replace(stage, result={**stage.result,
                    "recommendedNextTest": "OTHER"}) if i == review_index else stage
                for i, stage in enumerate(failed.stages)
            )),
            "wrong classification science": changed(stages=tuple(
                replace(stage, result={**stage.result, "crossTime": {
                    **stage.result["crossTime"], "classification": "ON_TARGET"}})
                if i == review_index else stage
                for i, stage in enumerate(failed.stages)
            )),
            "unresolved morphology": changed(stages=tuple(
                replace(stage, result={"physicalCycleResolved": False})
                if i == morphology_index else stage
                for i, stage in enumerate(failed.stages)
            )),
            "invalid frozen family": changed(stages=tuple(
                replace(stage, result={}) if i == mode_index else stage
                for i, stage in enumerate(failed.stages)
            )),
        }
        nonstationary = InvestigationStage(
            "024-nonstationary", "openstar.tess.nonstationary.summarize", "COMPLETE",
            None, {}, result={"preferredFrequencyAtReference": residual_frequency},
        )
        cases["real v20.9"] = changed(
            stages=failed.stages[:-1] + (nonstationary, failed.stages[-1],)
        )
        for label, candidate in cases.items():
            with self.subTest(label=label):
                self.assertIsNone(_repair_resolved_family_multisource_failure(
                    self.store, candidate, candidate.metadata["controlState"]
                ))
        with mock.patch(
            "workflows.tess.tess_autonomy.frozen_residual_localization_family",
            return_value=(family_period, (1, 2, 3), {},
                          "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE"),
        ):
            self.assertIsNone(_repair_resolved_family_multisource_failure(
                self.store, failed, failed.metadata["controlState"]
            ))

        repaired = repair_obsolete_terminal_wait(self.store, failed)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("032-prepare-multi-source-residual", selected["id"])
        self.assertEqual("031-prepare-multi-source-residual",
                         selected["triggered_by_stage_id"])
        self.assertEqual(historical_stages, repaired.stages)
        self.assertEqual(
            historical_files,
            {stage.id: self.store.stage_path_for(repaired.id, stage.id).read_bytes()
             for stage in repaired.stages},
        )
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

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
