import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import test_tess_recurrent_residual_long_baseline_confirmation as confirmation_fixture
from openstar_investigation import InvestigationStage
from openstar_workflow import StageRequest
from run_tess_investigation import (
    _can_continue_recurrent_residual_nonstationary_mode_modeling,
)
from workflows.tess.tess_autonomy import (
    _repair_recurrent_residual_nonstationary_terminal,
)
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_nonstationary import (
    RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
    RECURRENT_RESIDUAL_NONSTATIONARY_METHOD_CONTRACT_ID,
    _recurrent_window_series,
    build_nonstationary_project,
    build_recurrent_residual_nonstationary_method_contract,
    recurrent_residual_nonstationary_method_contract_hash,
    validate_recurrent_residual_nonstationary_boundary,
)
from workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation import (
    HANDLER_ID as CONFIRMATION_HANDLER_ID,
    build_dataset_specs,
)


class RecurrentResidualNonstationaryContinuationTests(
    unittest.TestCase
):
    def _confirmation_terminal(self, root):
        fixture = (
            confirmation_fixture
            .RecurrentResidualLongBaselineContinuationTests()
        )
        store, investigation = fixture._recurrent_terminal(root)
        investigation = store.set_status(investigation, "RUNNING")
        coordinator = mock.Mock()
        engine = build_engine(
            store, coordinator, poll_interval=0.0, timeout=None
        )
        engine.chain_stages = False
        completed, finalize = engine.run_stage(
            investigation,
            StageRequest(
                "013-long-baseline-time-frequency-confirmation",
                CONFIRMATION_HANDLER_ID,
                {},
                "011-transient-mode-validation",
            ),
            software_id="integration",
            software_version="1",
        )
        terminal, next_request = engine.run_stage(
            completed,
            finalize,
            software_id="integration",
            software_version="1",
        )
        self.assertIsNone(next_request)
        self.assertEqual(
            "NONSTATIONARY_RESIDUAL_STRUCTURE_CONFIRMED",
            terminal.stages[-2].result["classification"],
        )
        self.assertEqual(
            "LONG_BASELINE_NONSTATIONARY_MODE_MODELING",
            terminal.stages[-2].result["recommendedNextTest"],
        )
        coordinator.assert_not_called()
        return store, terminal

    def _window_specs(self, investigation):
        prepared = investigation.stages[0].result
        preparation = next(
            stage.result
            for stage in investigation.stages
            if stage.handler_id
            == "openstar.tess.time-frequency.prepare"
        )
        return build_dataset_specs(
            expected_tic_id=int(prepared["ticID"]),
            preparation=preparation,
        )

    def test_exact_boundary_builds_distinct_deterministic_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._confirmation_terminal(temporary)
            confirmation = investigation.stages[-2].result
            evidence = (
                validate_recurrent_residual_nonstationary_boundary(
                    confirmation
                )
            )
            first = (
                build_recurrent_residual_nonstationary_method_contract(
                    confirmation
                )
            )
            second = (
                build_recurrent_residual_nonstationary_method_contract(
                    copy.deepcopy(confirmation)
                )
            )

        self.assertEqual(
            RECURRENT_RESIDUAL_NONSTATIONARY_METHOD_CONTRACT_ID,
            first["methodContractID"],
        )
        self.assertEqual(
            RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
            first["evidenceLineage"],
        )
        self.assertEqual(first, second)
        self.assertEqual(
            recurrent_residual_nonstationary_method_contract_hash(
                first
            ),
            recurrent_residual_nonstationary_method_contract_hash(
                second
            ),
        )
        self.assertEqual(
            confirmation["methodContractHash"],
            evidence["sourceMethodContractHash"],
        )
        self.assertFalse(first["dataPolicy"]["downloadNewData"])
        self.assertFalse(first["dataPolicy"]["readOriginalSectorFlux"])
        self.assertFalse(
            first["claimPolicy"]["physicalMechanismResolved"]
        )

    def test_frozen_windows_form_nonoverlapping_sector_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._confirmation_terminal(temporary)
            confirmation = investigation.stages[-2].result
            contract = (
                build_recurrent_residual_nonstationary_method_contract(
                    confirmation
                )
            )
            loaded = _recurrent_window_series(
                method_contract=contract,
                dataset_specs=self._window_specs(investigation),
            )

        self.assertEqual(
            [1, 2, 28, 68, 69],
            [item["item"]["sector"] for item in loaded],
        )
        for item in loaded:
            times = item["absoluteTimes"]
            self.assertEqual(len(times), len(set(times.tolist())))
            self.assertGreaterEqual(len(times), 256)
            self.assertTrue(
                item["residualMeta"]["familySubtractionReused"]
            )
            self.assertEqual(
                "NEAREST_WINDOW_CENTER_PER_SECTOR_TIME_TIE_LOWEST_INDEX",
                item["residualMeta"]["overlapResolution"],
            )

    def test_project_builder_never_reads_original_sector_flux(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, investigation = self._confirmation_terminal(temporary)
            confirmation = investigation.stages[-2].result
            evidence = (
                validate_recurrent_residual_nonstationary_boundary(
                    confirmation
                )
            )
            contract = (
                build_recurrent_residual_nonstationary_method_contract(
                    confirmation
                )
            )
            source_project = root / "source-project.json"
            source_project.write_text(json.dumps({
                "id": "synthetic-source",
                "name": "Synthetic source",
                "workloadID": "openstar.lomb-scargle.v1",
                "datasets": [],
            }), encoding="utf-8")
            spec = build_nonstationary_project(
                source_project_path=source_project,
                source_dataset_entry={
                    "id": "primary-sector-1",
                    "targetName": "Synthetic recurrent residual",
                },
                primary_dataset_path=(
                    root / "original-flux-must-not-be-read.json"
                ),
                primary_sector=1,
                independent_spec={"preparedSectors": []},
                physical_period_days=evidence[
                    "establishedPeriodDays"
                ],
                time_frequency_summary=None,
                output_dir=root / "artifacts",
                investigation_id="recurrent-v20-9-3",
                recurrent_method_contract=contract,
                recurrent_dataset_specs=self._window_specs(
                    investigation
                ),
            )

        self.assertFalse(spec["originalSectorFluxRead"])
        self.assertEqual(
            RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
            spec["evidenceLineage"],
        )
        self.assertEqual(
            RECURRENT_RESIDUAL_NONSTATIONARY_METHOD_CONTRACT_ID,
            spec["methodContractID"],
        )
        self.assertEqual(
            recurrent_residual_nonstationary_method_contract_hash(
                contract
            ),
            spec["methodContractHash"],
        )
        self.assertEqual(66, len(spec["preparedDatasets"]))
        self.assertEqual(66, len(spec["groups"]) * 33)
        self.assertTrue(Path(spec["projectPath"]).is_file())
        self.assertTrue(Path(spec["analysisSeriesPath"]).is_file())

    def test_manual_boundary_rejects_mutated_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._confirmation_terminal(temporary)
            _can_continue_recurrent_residual_nonstationary_mode_modeling(
                investigation
            )

            stages = list(investigation.stages)
            altered = copy.deepcopy(stages[-2].result)
            altered["classification"] = "OTHER"
            stages[-2] = replace(stages[-2], result=altered)
            conclusion = copy.deepcopy(stages[-1].result)
            conclusion[
                "longBaselineTimeFrequencyConfirmation"
            ] = altered
            stages[-1] = replace(stages[-1], result=conclusion)
            changed = replace(
                investigation, stages=tuple(stages)
            )

            with self.assertRaisesRegex(
                RuntimeError, "exact conservative"
            ):
                _can_continue_recurrent_residual_nonstationary_mode_modeling(
                    changed
                )

    def test_automatic_repair_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._confirmation_terminal(
                temporary
            )
            immutable = tuple(
                json.dumps(asdict(stage), sort_keys=True)
                for stage in investigation.stages
            )
            repaired = (
                _repair_recurrent_residual_nonstationary_terminal(
                    store,
                    investigation,
                    investigation.metadata["controlState"],
                )
            )
            repeated = (
                _repair_recurrent_residual_nonstationary_terminal(
                    store,
                    repaired,
                    repaired.metadata["controlState"],
                )
            )

        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(
            immutable,
            tuple(
                json.dumps(asdict(stage), sort_keys=True)
                for stage in repaired.stages
            ),
        )
        selected = repaired.metadata[
            "controlState"
        ]["selectedExperiment"]
        self.assertEqual(
            "015-prepare-recurrent-residual-nonstationary",
            selected["id"],
        )
        self.assertEqual(
            "openstar.tess.nonstationary.prepare",
            selected["handler_id"],
        )
        self.assertEqual(
            "013-long-baseline-time-frequency-confirmation",
            selected["triggered_by_stage_id"],
        )
        self.assertEqual(
            RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
            selected["parameters"]["evidenceLineage"],
        )
        self.assertIsNone(repeated)

    def test_prepare_handler_uses_recurrent_contract_and_window_specs(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._confirmation_terminal(
                temporary
            )
            root = Path(temporary)
            project_path = root / "project.json"
            series_path = root / "series.json"
            project_path.write_text("{}", encoding="utf-8")
            series_path.write_text("{}", encoding="utf-8")

            stages = list(investigation.stages)
            prepared = copy.deepcopy(stages[0].result)
            prepared.update({
                "sourceProjectPath": str(root / "source-project.json"),
                "sourceDatasetEntry": {
                    "id": "primary-sector-1",
                    "targetName": "Synthetic recurrent residual",
                },
            })
            stages[0] = replace(stages[0], result=prepared)
            confirmation_index = next(
                index for index, stage in enumerate(stages)
                if stage.handler_id == CONFIRMATION_HANDLER_ID
            )
            stages.insert(
                confirmation_index,
                InvestigationStage(
                    "012b-independent-prepare",
                    "openstar.tess.independent.prepare",
                    "COMPLETE",
                    "012-finalize",
                    {},
                    result={"preparedSectors": []},
                ),
            )
            investigation = replace(
                investigation, stages=tuple(stages)
            )
            investigation = store.set_status(
                investigation, "RUNNING"
            )
            spec = {
                "available": True,
                "projectID": "recurrent-test",
                "projectPath": str(project_path),
                "analysisSeriesPath": str(series_path),
                "workloadID": "openstar.lomb-scargle.v1",
                "residualCenterPeriodDays": 3.3,
                "frequencySearch": {
                    "minimumFrequency": 0.2,
                    "maximumFrequency": 0.4,
                },
                "driftGrid": {
                    "minimumFractionalFrequencyDriftPerDay": -0.001,
                    "maximumFractionalFrequencyDriftPerDay": 0.001,
                    "count": 33,
                },
                "groups": [],
                "preparedDatasets": [],
                "totalWorkUnits": 0,
            }
            coordinator = mock.Mock()
            engine = build_engine(
                store, coordinator, poll_interval=0.0, timeout=None
            )
            engine.chain_stages = False
            with mock.patch(
                "workflows.tess.tess_investigation."
                "build_nonstationary_project",
                return_value=spec,
            ) as builder:
                completed, next_request = engine.run_stage(
                    investigation,
                    StageRequest(
                        "015-prepare-recurrent-residual-nonstationary",
                        "openstar.tess.nonstationary.prepare",
                        {
                            "evidenceLineage": (
                                RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE
                            )
                        },
                        "013-long-baseline-time-frequency-confirmation",
                    ),
                    software_id="integration",
                    software_version="1",
                )

        kwargs = builder.call_args.kwargs
        self.assertIsNone(kwargs["time_frequency_summary"])
        self.assertIsNotNone(kwargs["recurrent_method_contract"])
        self.assertGreater(
            len(kwargs["recurrent_dataset_specs"]), 0
        )
        self.assertFalse(
            completed.stages[-1].result.get(
                "originalSectorFluxRead", False
            )
        )
        self.assertEqual(
            "openstar.tess.nonstationary.run",
            next_request.handler_id,
        )
        coordinator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
