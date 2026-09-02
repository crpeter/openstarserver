import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_investigation import (
    ArtifactReference,
    Investigation,
    InvestigationStage,
    InvestigationStore,
    sha256_file,
    sha256_json,
)
from openstar_targets import InvestigationTarget
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    _repair_binary_confirmation_time_origin_failure,
    _repair_residual_phase_difference_imaging_terminal_handoff,
    plan_tess_branches,
    repair_obsolete_terminal_wait,
)
from workflows.tess.tess_binary_confirmation import (
    MORPHOLOGY_EVENT_SCREEN_ENTRY,
    _dataset_sector,
    _original_time_origin,
    _sector,
)


class BinaryConfirmationTimeOriginTests(unittest.TestCase):
    def test_source_and_primary_metadata_schemas_resolve_exactly(self):
        source = {
            "source": {"sector": 28, "originalTimeOriginDays": 2500.25}
        }
        primary = {
            "metadata": {"sector": 1, "originalTimeOriginDays": 1325.5}
        }

        self.assertEqual(2500.25, _original_time_origin(source))
        self.assertEqual(1325.5, _original_time_origin(primary))
        self.assertEqual(28, _dataset_sector(source))
        self.assertEqual(1, _dataset_sector(primary))

        measured = _sector(
            {
                **primary,
                "id": "primary",
                "times": [0.0, 1.0],
                "flux": [0.0, 0.0],
            },
            2.0,
            "PRIMARY",
        )
        self.assertEqual(1325.5, measured["originalTimeOriginDays"])
        self.assertEqual(1, measured["sector"])
        self.assertFalse(measured["usable"])
        self.assertEqual("INSUFFICIENT_SAMPLES", measured["reason"])

    def test_duplicate_lineage_must_agree(self):
        matching = {
            "source": {"sector": 1, "originalTimeOriginDays": 1325.5},
            "metadata": {"sector": 1, "originalTimeOriginDays": 1325.5},
        }
        self.assertEqual(1325.5, _original_time_origin(matching))
        self.assertEqual(1, _dataset_sector(matching))

        conflicting_origin = json.loads(json.dumps(matching))
        conflicting_origin["metadata"]["originalTimeOriginDays"] = 1325.75
        with self.assertRaisesRegex(ValueError, "conflicting.*originalTimeOriginDays"):
            _original_time_origin(conflicting_origin)

        conflicting_sector = json.loads(json.dumps(matching))
        conflicting_sector["metadata"]["sector"] = 2
        with self.assertRaisesRegex(ValueError, "conflicting.*sector"):
            _dataset_sector(conflicting_sector)

    def test_missing_invalid_and_nonfinite_origins_fail_closed(self):
        invalid = (
            {},
            {"metadata": {"originalTimeOriginDays": True}},
            {"metadata": {"originalTimeOriginDays": "not-a-number"}},
            {"metadata": {"originalTimeOriginDays": float("nan")}},
            {"source": {"originalTimeOriginDays": float("inf")}},
        )
        for dataset in invalid:
            with self.subTest(dataset=dataset):
                with self.assertRaisesRegex(ValueError, "originalTimeOriginDays"):
                    _original_time_origin(dataset)


class BinaryConfirmationTimeOriginRecoveryTests(unittest.TestCase):
    HANDLERS = (
        "openstar.tess.prepare-target",
        "openstar.tess.primary-project.run",
        "openstar.tess.catalog-identity",
        "openstar.tess.hypotheses",
        "openstar.tess.planner",
        "openstar.tess.independent.prepare",
        "openstar.tess.independent.run",
        "openstar.tess.independent.interpret",
        "openstar.tess.independent.broad.prepare",
        "openstar.tess.independent.broad.run",
        "openstar.tess.independent.broad.interpret",
        "openstar.tess.morphology.analyze",
        "openstar.tess.binary-confirmation.analyze",
    )
    IDS = (
        "001-prepare-target",
        "002-primary-distributed-search",
        "003-catalog-identity",
        "004-hypotheses",
        "005-planner",
        "006-prepare-independent-sectors",
        "007-run-independent-sectors",
        "008-interpret-independent-sectors",
        "009-prepare-broad-independent-search",
        "010-run-broad-independent-search",
        "011-interpret-broad-independent-search",
        "012-characterize-variability",
        "013-periodic-event-screen",
    )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = InvestigationStore(self.root / "investigations")

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def _append(self, investigation, index, result, input_hashes, artifacts=()):
        stage_id = self.IDS[index]
        handler_id = self.HANDLERS[index]
        triggered_by = self.IDS[index - 1] if index else None
        failed = index == len(self.IDS) - 1
        parameters = (
            {"entryMode": MORPHOLOGY_EVENT_SCREEN_ENTRY} if failed else {}
        )
        running = InvestigationStage(
            id=stage_id,
            handler_id=handler_id,
            status="RUNNING",
            triggered_by_stage_id=triggered_by,
            parameters=parameters,
        )
        investigation = self.store.append_running_stage(investigation, running)
        terminal = self.store.build_terminal_stage(
            stage_id=stage_id,
            handler_id=handler_id,
            status="FAILED" if failed else "COMPLETE",
            triggered_by_stage_id=triggered_by,
            parameters=parameters,
            result=None if failed else result,
            error=(
                "ValueError: frozen dataset lacks originalTimeOriginDays"
                if failed
                else None
            ),
            failure_classification="NON_RETRYABLE" if failed else None,
            software_id="openstar.tess-ranked-followup-runner",
            software_version="1",
            input_hashes=input_hashes,
            artifacts=artifacts,
            started_at=running.started_at,
        )
        return self.store.complete_current_stage(investigation, terminal)

    def _failed_boundary(self):
        primary_path = self.root / "primary.json"
        self._write(
            primary_path,
            {
                "id": "tess-sector-1-tic-29495621",
                "metadata": {
                    "sector": 1,
                    "ticID": 29495621,
                    "originalTimeOriginDays": 1325.25,
                },
                "times": [0.0, 1.0],
                "flux": [0.0, 0.0],
            },
        )
        primary_hash = sha256_file(primary_path)

        prepared_sectors = []
        independent_hashes = {}
        for sector in (28, 68, 92, 95):
            path = self.root / f"sector-{sector}.json"
            self._write(
                path,
                {
                    "id": f"sector-{sector}",
                    "source": {
                        "sector": sector,
                        "originalTimeOriginDays": 2000.0 + sector,
                    },
                    "times": [0.0, 1.0],
                    "flux": [0.0, 0.0],
                },
            )
            prepared_sectors.append({"sector": sector, "datasetPath": str(path)})
            independent_hashes[f"independentSector{sector}"] = sha256_file(path)

        stage_results = [{} for _ in self.IDS]
        stage_results[0] = {
            "datasetPath": str(primary_path),
            "sector": 1,
            "ticID": 29495621,
        }
        stage_results[5] = {"preparedSectors": prepared_sectors}
        stage_results[11] = {
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": 10.510316195053623,
            "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED",
        }
        morphology_path = self.root / "morphology-v20.4.json"
        self._write(morphology_path, stage_results[11])
        morphology_artifact = ArtifactReference(
            path=str(morphology_path),
            sha256=sha256_file(morphology_path),
            media_type="application/json",
        )

        investigation = self.store.create(
            "tess-discovery-sector-1-tic-29495621",
            WORKFLOW_ID,
            "20.2",
            metadata={"sector": 1, "ticID": 29495621},
        )
        for index in range(len(self.IDS)):
            input_hashes = {}
            if index == 0:
                input_hashes["sourceDataset"] = primary_hash
            if index == 11:
                input_hashes = {
                    "primaryDataset": primary_hash,
                    **independent_hashes,
                }
            investigation = self._append(
                investigation,
                index,
                stage_results[index],
                input_hashes,
                (morphology_artifact,) if index == 11 else (),
            )

        failed = investigation.stages[-1]
        control = {
            "branchAssessments": [],
            "selectedExperiment": {
                "id": failed.id,
                "handler_id": failed.handler_id,
                "parameters": failed.parameters,
                "triggered_by_stage_id": failed.triggered_by_stage_id,
            },
            "schedulerAction": "INVESTIGATION_FAILED",
        }
        investigation = self.store.set_control_state(
            investigation,
            status="FAILED",
            control_state=control,
        )
        return investigation, primary_path

    def test_exact_failure_reopens_append_only_and_is_idempotent(self):
        investigation, _ = self._failed_boundary()
        original_stages = investigation.stages

        repaired = repair_obsolete_terminal_wait(self.store, investigation)

        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(original_stages, repaired.stages)
        self.assertEqual(
            {
                "id": "014-periodic-event-screen-recovery",
                "handler_id": "openstar.tess.binary-confirmation.analyze",
                "parameters": {"entryMode": MORPHOLOGY_EVENT_SCREEN_ENTRY},
                "triggered_by_stage_id": "013-periodic-event-screen",
            },
            repaired.metadata["controlState"]["selectedExperiment"],
        )
        self.assertEqual(
            "TESS_BINARY_CONFIRMATION_TIME_ORIGIN_COMPATIBILITY_RETRY",
            repaired.metadata["controlState"]["recovery"],
        )
        self.assertEqual(repaired, repair_obsolete_terminal_wait(self.store, repaired))

    def test_recovery_rejects_changed_failure_and_corrupted_dataset(self):
        investigation, primary_path = self._failed_boundary()
        control = investigation.metadata["controlState"]

        altered_failure = replace(
            investigation,
            stages=investigation.stages[:-1] + (
                replace(investigation.stages[-1], error="ValueError: unrelated"),
            ),
        )
        self.assertIsNone(
            _repair_binary_confirmation_time_origin_failure(
                self.store, altered_failure, control
            )
        )

        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        primary["metadata"]["originalTimeOriginDays"] += 1.0
        self._write(primary_path, primary)
        self.assertIsNone(
            _repair_binary_confirmation_time_origin_failure(
                self.store, investigation, control
            )
        )

    def test_phase_difference_prepare_preserves_its_persisted_run(self):
        localization = InvestigationStage(
            id="035-interpret-catalog-guided-source-localization",
            handler_id="openstar.tess.catalog-guided-source-localization.interpret",
            status="COMPLETE",
            triggered_by_stage_id="034-run-catalog-guided-source-localization",
            parameters={},
            result={
                "classification": "UNRESOLVED",
                "sourceAttributionResolved": False,
                "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
            },
        )
        persisted_run = {
            "id": "037-run-residual-phase-difference-imaging",
            "handler_id": "openstar.tess.residual-phase-difference-imaging.run",
            "parameters": {},
            "triggered_by_stage_id": "036-prepare-residual-phase-difference-imaging",
        }
        preparation = InvestigationStage(
            id="036-prepare-residual-phase-difference-imaging",
            handler_id="openstar.tess.residual-phase-difference-imaging.prepare",
            status="COMPLETE",
            triggered_by_stage_id=localization.id,
            parameters={},
            result={},
            next_stage=persisted_run,
        )
        investigation = Investigation(
            id="tess-discovery-sector-1-tic-29495621",
            workflow_id=WORKFLOW_ID,
            workflow_version="20.2",
            status="COMPLETE",
            created_at="2026-09-01T00:00:00+00:00",
            updated_at="2026-09-01T00:00:00+00:00",
            metadata={},
            stages=(localization, preparation),
        )
        target = InvestigationTarget(
            id="tess-sector-1-tic-29495621",
            investigation_id=investigation.id,
            workflow_id=WORKFLOW_ID,
            workflow_version="20.2",
        )

        branches = plan_tess_branches(investigation, target)

        self.assertEqual(1, len(branches))
        self.assertEqual(
            persisted_run["id"], branches[0].experiment.id
        )
        self.assertEqual(
            persisted_run["handler_id"], branches[0].experiment.handler_id
        )

    def test_terminal_phase_difference_prepare_reopens_append_only(self):
        investigation = self.store.create(
            "tess-discovery-sector-1-tic-29495621",
            WORKFLOW_ID,
            "20.2",
        )

        def append_complete(stage_id, handler_id, result, *, triggered_by,
                            input_hashes=None, artifacts=(), next_stage=None):
            nonlocal investigation
            running = InvestigationStage(
                id=stage_id,
                handler_id=handler_id,
                status="RUNNING",
                triggered_by_stage_id=triggered_by,
                parameters={},
            )
            investigation = self.store.append_running_stage(
                investigation, running
            )
            terminal = self.store.build_terminal_stage(
                stage_id=stage_id,
                handler_id=handler_id,
                status="COMPLETE",
                triggered_by_stage_id=triggered_by,
                parameters={},
                result=result,
                error=None,
                software_id="openstar.tess-ranked-followup-runner",
                software_version="1",
                input_hashes=input_hashes or {},
                artifacts=artifacts,
                started_at=running.started_at,
                next_stage=next_stage,
            )
            investigation = self.store.complete_current_stage(
                investigation, terminal
            )

        bridge = {"version": "catalog-guided-preparation-v1"}
        localization = {
            "classification": "UNRESOLVED",
            "sourceAttributionResolved": False,
            "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
        }
        append_complete(
            "033-prepare-catalog-guided-source-localization",
            "openstar.tess.catalog-guided-source-localization.prepare",
            bridge,
            triggered_by="032-catalog-counterpart",
        )
        append_complete(
            "035-interpret-catalog-guided-source-localization",
            "openstar.tess.catalog-guided-source-localization.interpret",
            localization,
            triggered_by="034-run-catalog-guided-source-localization",
        )

        preparation_path = self.root / "preparation.json"
        preparation = {
            "version": (
                "openstar.tess-residual-phase-difference-imaging-"
                "preparation.v1"
            ),
            "execution": "coordinator-local-difference-image-centroiding",
            "physicalCycleResolved": False,
            "preparationPath": str(preparation_path),
        }
        self._write(preparation_path, preparation)
        artifact = ArtifactReference(
            path=str(preparation_path),
            sha256=sha256_file(preparation_path),
            media_type="application/json",
        )
        persisted_run = {
            "id": "037-run-residual-phase-difference-imaging",
            "handler_id": "openstar.tess.residual-phase-difference-imaging.run",
            "parameters": {},
            "triggered_by_stage_id": "036-prepare-residual-phase-difference-imaging",
        }
        append_complete(
            "036-prepare-residual-phase-difference-imaging",
            "openstar.tess.residual-phase-difference-imaging.prepare",
            preparation,
            triggered_by="035-interpret-catalog-guided-source-localization",
            input_hashes={
                "catalogGuidedPreparation": sha256_json(bridge),
                "catalogGuidedInterpretation": sha256_json(localization),
            },
            artifacts=(artifact,),
            next_stage=persisted_run,
        )
        investigation = self.store.set_control_state(
            investigation,
            status="COMPLETE",
            control_state={
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            },
        )
        original_stages = investigation.stages

        repaired = _repair_residual_phase_difference_imaging_terminal_handoff(
            self.store,
            investigation,
            investigation.metadata["controlState"],
        )

        self.assertIsNotNone(repaired)
        self.assertEqual(original_stages, repaired.stages)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(
            persisted_run,
            repaired.metadata["controlState"]["selectedExperiment"],
        )
        self.assertEqual(
            repaired,
            repair_obsolete_terminal_wait(self.store, repaired),
        )

    def test_recovery_rejects_wrong_morphology_and_missing_ledger(self):
        investigation, _ = self._failed_boundary()
        control = investigation.metadata["controlState"]

        wrong_morphology = replace(
            investigation,
            stages=investigation.stages[:11] + (
                replace(
                    investigation.stages[11],
                    result={
                        "physicalCycleResolved": False,
                        "resolvedPhysicalPeriodDays": None,
                        "morphologyClass": "UNRESOLVED",
                    },
                ),
            ) + investigation.stages[12:],
        )
        self.assertIsNone(
            _repair_binary_confirmation_time_origin_failure(
                self.store, wrong_morphology, control
            )
        )

        self.store.stage_path_for(
            investigation.id, "012-characterize-variability"
        ).unlink()
        self.assertIsNone(
            _repair_binary_confirmation_time_origin_failure(
                self.store, investigation, control
            )
        )


if __name__ == "__main__":
    unittest.main()
