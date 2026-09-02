import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from openstar_investigation import InvestigationStage
from openstar_workflow import StageRequest
from run_tess_investigation import (
    _can_continue_residual_mode_localization,
)
from test_tess_recurrent_residual_nonstationary_mode_modeling import (
    RecurrentResidualNonstationaryContinuationTests,
)
from workflows.tess.tess_autonomy import (
    _repair_recurrent_residual_localization_terminal,
)
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_nonstationary import (
    RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
    build_recurrent_residual_nonstationary_method_contract,
    recurrent_residual_nonstationary_method_contract_hash,
    validate_recurrent_residual_nonstationary_localization_boundary,
)
from workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation import (
    HANDLER_ID as CONFIRMATION_HANDLER_ID,
)


class RecurrentResidualLocalizationContinuationTests(unittest.TestCase):
    def _terminal(self, root):
        fixture = RecurrentResidualNonstationaryContinuationTests()
        store, investigation = fixture._confirmation_terminal(root)
        confirmation = next(
            stage.result
            for stage in reversed(investigation.stages)
            if stage.handler_id == CONFIRMATION_HANDLER_ID
        )
        contract = (
            build_recurrent_residual_nonstationary_method_contract(
                confirmation
            )
        )
        frequency = 0.306808947662884
        summary = {
            "evidenceLineage": (
                RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE
            ),
            "methodContractID": contract["methodContractID"],
            "methodContractHash": (
                recurrent_residual_nonstationary_method_contract_hash(
                    contract
                )
            ),
            "methodContract": contract,
            "classification": "AMPLITUDE_PHASE_EVOLVING_MODE",
            "preferredModel": {
                "modelID": "STATIONARY_SECTOR_EVOLVING_MODE",
                "signalSectors": [1, 2, 28, 68, 69],
            },
            "preferredFrequencyAtReference": frequency,
            "preferredPeriodAtReferenceDays": 1.0 / frequency,
            "fractionalFrequencyDriftPerDay": 0.0,
            "timeReferenceDays": 5000.0,
            "physicalMechanismResolved": False,
            "claimLevelChanged": False,
            "recommendedNextTest": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
        }
        stages = list(investigation.stages)
        stages.extend((
            InvestigationStage(
                "015-prepare-recurrent-residual-nonstationary",
                "openstar.tess.nonstationary.prepare",
                "COMPLETE",
                "013-long-baseline-time-frequency-confirmation",
                {
                    "evidenceLineage": (
                        RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE
                    )
                },
                result={
                    "evidenceLineage": (
                        RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE
                    ),
                    "methodContract": contract,
                },
            ),
            InvestigationStage(
                "016-run-nonstationary",
                "openstar.tess.nonstationary.run",
                "COMPLETE",
                "015-prepare-recurrent-residual-nonstationary",
                {},
                result={"datasets": []},
            ),
            InvestigationStage(
                "017-interpret-nonstationary",
                "openstar.tess.nonstationary.interpret",
                "COMPLETE",
                "016-run-nonstationary",
                {},
                result={"groups": {}},
            ),
            InvestigationStage(
                "018-summarize-nonstationary",
                "openstar.tess.nonstationary.summarize",
                "COMPLETE",
                "017-interpret-nonstationary",
                {},
                result=summary,
            ),
            InvestigationStage(
                "019-finalize",
                "openstar.tess.finalize",
                "COMPLETE",
                "018-summarize-nonstationary",
                {
                    "outputSuffix": (
                        "v20.9.3-recurrent-residual-nonstationary"
                    )
                },
                result={
                    "nonstationaryModeling": summary,
                    "recommendedNextTest": (
                        "RESIDUAL_MODE_PIXEL_LOCALIZATION"
                    ),
                },
                stop=True,
            ),
        ))
        investigation = replace(
            investigation,
            status="COMPLETE",
            stages=tuple(stages),
        )
        store.save(investigation)
        return store, investigation, confirmation, summary

    def test_exact_boundary_validates_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, confirmation, summary = self._terminal(temporary)
            period = (
                validate_recurrent_residual_nonstationary_localization_boundary(
                    summary, confirmation
                )
            )
            altered = copy.deepcopy(summary)
            altered["preferredPeriodAtReferenceDays"] += 0.01
            with self.assertRaisesRegex(
                RuntimeError, "exact persisted"
            ):
                validate_recurrent_residual_nonstationary_localization_boundary(
                    altered, confirmation
                )

        self.assertEqual(
            confirmation["methodContract"]["evidenceBoundary"][
                "periodReference"
            ]["periodDays"],
            period,
        )

    def test_manual_gate_requires_exact_append_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation, _, _ = self._terminal(temporary)
            _can_continue_residual_mode_localization(investigation)
            stages = list(investigation.stages)
            stages[-1] = replace(
                stages[-1],
                parameters={"outputSuffix": "OTHER"},
            )
            with self.assertRaisesRegex(
                RuntimeError, "exact finalized v20.9.3"
            ):
                _can_continue_residual_mode_localization(
                    replace(investigation, stages=tuple(stages))
                )

    def test_automatic_repair_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation, _, _ = self._terminal(temporary)
            immutable = tuple(
                json.dumps(asdict(stage), sort_keys=True)
                for stage in investigation.stages
            )
            repaired = _repair_recurrent_residual_localization_terminal(
                store,
                investigation,
                investigation.metadata["controlState"],
            )
            repeated = _repair_recurrent_residual_localization_terminal(
                store,
                repaired,
                repaired.metadata["controlState"],
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
            "020-prepare-residual-mode-localization",
            selected["id"],
        )
        self.assertEqual(
            "openstar.tess.residual-mode-localization.prepare",
            selected["handler_id"],
        )
        self.assertEqual(
            "018-summarize-nonstationary",
            selected["triggered_by_stage_id"],
        )
        self.assertEqual(
            RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
            selected["parameters"]["evidenceLineage"],
        )
        self.assertIsNone(repeated)

    def test_prepare_handler_uses_authenticated_recurrent_period(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, investigation, confirmation, summary = self._terminal(
                temporary
            )
            project_path = root / "pixel-project.json"
            project_path.write_text("{}", encoding="utf-8")
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
            continuation_index = next(
                index for index, stage in enumerate(stages)
                if stage.id
                == "015-prepare-recurrent-residual-nonstationary"
            )
            stages[continuation_index:continuation_index] = [
                InvestigationStage(
                    "014a-catalog-identity",
                    "openstar.tess.catalog-identity",
                    "COMPLETE",
                    "014-finalize",
                    {},
                    result={
                        "tic": {
                            "metadata": {
                                "raDeg": 10.0,
                                "decDeg": -20.0,
                            }
                        }
                    },
                ),
                InvestigationStage(
                    "014b-independent-prepare",
                    "openstar.tess.independent.prepare",
                    "COMPLETE",
                    "014a-catalog-identity",
                    {},
                    result={"preparedSectors": []},
                ),
            ]
            investigation = replace(
                investigation,
                status="RUNNING",
                stages=tuple(stages),
            )
            spec = {
                "available": True,
                "projectID": "recurrent-pixel-test",
                "projectPath": str(project_path),
                "workloadID": "openstar.lomb-scargle.v1",
                "signalSectors": [1, 2, 28, 68, 69],
                "preparedPixels": [],
                "totalWorkUnits": 0,
            }
            coordinator = mock.Mock()
            engine = build_engine(
                store, coordinator, poll_interval=0.0, timeout=None
            )
            engine.chain_stages = False
            with mock.patch(
                "workflows.tess.tess_investigation."
                "build_residual_mode_pixel_project",
                return_value=spec,
            ) as builder:
                completed, next_request = engine.run_stage(
                    investigation,
                    StageRequest(
                        "020-prepare-residual-mode-localization",
                        "openstar.tess.residual-mode-localization.prepare",
                        {
                            "evidenceLineage": (
                                RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE
                            )
                        },
                        "018-summarize-nonstationary",
                    ),
                    software_id="integration",
                    software_version="1",
                )

        expected_contract = (
            build_recurrent_residual_nonstationary_method_contract(
                confirmation
            )
        )
        self.assertEqual(
            expected_contract["evidenceBoundary"][
                "establishedPeriodDays"
            ],
            builder.call_args.kwargs["physical_period_days"],
        )
        self.assertEqual(
            summary,
            builder.call_args.kwargs["nonstationary_summary"],
        )
        self.assertEqual(
            "RECURRENT_RESIDUAL_RESOLVED_PHYSICAL_CYCLE",
            completed.stages[-1].result["periodReference"]["kind"],
        )
        self.assertEqual(
            "openstar.tess.residual-mode-localization.run",
            next_request.handler_id,
        )
        coordinator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
