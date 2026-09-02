import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import test_tess_transient_mode_validation as transient_fixture
from openstar_investigation import InvestigationStage
from openstar_workflow import StageRequest
from run_tess_investigation import (
    _can_continue_v20_8_long_baseline_time_frequency_confirmation,
)
from workflows.tess.tess_autonomy import (
    _repair_v20_8_long_baseline_time_frequency_confirmation_terminal,
)
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_recurrent_residual_long_baseline_confirmation import (
    METHOD_CONTRACT_ID,
    RECURRENT_CLASSIFICATION,
    RESULT_VERSION,
    build_method_contract,
    method_contract_hash,
    validate_recurrent_residual_boundary,
)
from workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation import (
    HANDLER_ID,
)


class RecurrentResidualLongBaselineContinuationTests(unittest.TestCase):
    def _recurrent_terminal(self, root):
        fixture = transient_fixture.TransientModeContinuationTests()
        store, investigation = fixture._history(root)

        for window_index in (1, 2, 3):
            dataset_id = f"sector-68-window-{window_index}"
            transient_fixture._write_window(
                Path(root) / f"{dataset_id}.json",
                dataset_id=dataset_id,
                sector=68,
                role="independent-time-frequency-window",
                window_index=window_index,
                frequency=transient_fixture.TRANSIENT_FREQUENCY,
            )

        investigation = store.set_status(investigation, "RUNNING")
        coordinator = mock.Mock()
        engine = build_engine(
            store, coordinator, poll_interval=0.0, timeout=None
        )
        engine.chain_stages = False
        completed, finalize = engine.run_stage(
            investigation,
            StageRequest(
                "011-transient-mode-validation",
                transient_fixture.HANDLER_ID,
                {},
                "009-summarize-time-frequency",
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
        transient = terminal.stages[-2]
        self.assertIsNone(next_request)
        self.assertEqual(
            RECURRENT_CLASSIFICATION, transient.result["classification"]
        )
        self.assertEqual(
            "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION",
            transient.result["recommendedNextTest"],
        )
        coordinator.assert_not_called()
        return store, terminal

    def test_exact_recurrent_boundary_builds_distinct_frozen_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._recurrent_terminal(temporary)
            transient = investigation.stages[-2]
            evidence = validate_recurrent_residual_boundary(transient.result)
            contract = build_method_contract(
                transient_validation=transient.result
            )

        self.assertEqual(METHOD_CONTRACT_ID, contract["methodContractID"])
        self.assertEqual(RESULT_VERSION, contract["resultVersion"])
        self.assertEqual(
            transient.result["methodContractHash"],
            contract["evidenceBoundary"][
                "sourceTransientMethodContractHash"
            ],
        )
        self.assertEqual(
            [2, 28, 68, 69],
            evidence["acceptedIndependentSectors"],
        )
        self.assertGreaterEqual(
            evidence["acceptedIndependentWindowCount"], 5
        )
        self.assertFalse(contract["networkAccess"])
        self.assertFalse(contract["dataPolicy"]["downloadNewData"])

    def test_manual_boundary_rejects_altered_recurrent_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._recurrent_terminal(temporary)
            _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                investigation
            )

            stages = list(investigation.stages)
            altered = copy.deepcopy(stages[-2].result)
            altered["classification"] = (
                "TRANSIENT_INDEPENDENT_FREQUENCY_SUPPORTED"
            )
            stages[-2] = replace(stages[-2], result=altered)
            conclusion = copy.deepcopy(stages[-1].result)
            conclusion["transientModeValidation"] = altered
            stages[-1] = replace(stages[-1], result=conclusion)
            changed = replace(investigation, stages=tuple(stages))

            with self.assertRaisesRegex(
                RuntimeError, "recompute exactly"
            ):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    changed
                )

    def test_automatic_repair_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._recurrent_terminal(temporary)
            immutable = tuple(
                json.dumps(asdict(stage), sort_keys=True)
                for stage in investigation.stages
            )
            control = investigation.metadata["controlState"]
            with mock.patch.object(
                store,
                "verified_terminal_stage_ledger_hash",
                return_value=True,
            ), mock.patch(
                "workflows.tess.tess_autonomy._verified_stage_json",
                return_value=True,
            ):
                repaired = (
                    _repair_v20_8_long_baseline_time_frequency_confirmation_terminal(
                        store, investigation, control
                    )
                )
                repeated = (
                    _repair_v20_8_long_baseline_time_frequency_confirmation_terminal(
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
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(
            "013-long-baseline-time-frequency-confirmation",
            selected["id"],
        )
        self.assertEqual(HANDLER_ID, selected["handler_id"])
        self.assertEqual(
            "011-transient-mode-validation",
            selected["triggered_by_stage_id"],
        )
        self.assertEqual(
            "TESS_RECURRENT_RESIDUAL_LONG_BASELINE_CONFIRMATION",
            repaired.metadata["controlState"]["recovery"],
        )
        self.assertIsNone(repeated)

    def test_handler_persists_v20_8_2_without_claim_or_mechanism_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._recurrent_terminal(temporary)
            frozen = copy.deepcopy(investigation.stages)
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
                    HANDLER_ID,
                    {},
                    "011-transient-mode-validation",
                ),
                software_id="integration",
                software_version="1",
            )
            self.assertEqual(frozen, completed.stages[:-1])
            result = completed.stages[-1].result
            self.assertEqual(METHOD_CONTRACT_ID, result["methodContractID"])
            self.assertEqual(RESULT_VERSION, result["version"])
            self.assertEqual(
                method_contract_hash(result["methodContract"]),
                result["methodContractHash"],
            )
            self.assertFalse(result["physicalMechanismResolved"])
            self.assertFalse(result["claimLevelChanged"])
            self.assertFalse(result["automaticDiscoveryClaim"])
            self.assertEqual(
                "v20.8.2-recurrent-residual-long-baseline-confirmation",
                finalize.parameters["outputSuffix"],
            )

            final, next_request = engine.run_stage(
                completed,
                finalize,
                software_id="integration",
                software_version="1",
            )
            conclusion = final.stages[-1].result

        self.assertIsNone(next_request)
        self.assertEqual(
            "CANDIDATE_PERIOD", conclusion["claim"]["claim"]
        )
        self.assertEqual(
            result,
            conclusion["longBaselineTimeFrequencyConfirmation"],
        )
        coordinator.assert_not_called()

    def test_existing_stage_and_nonterminal_state_remain_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._recurrent_terminal(temporary)
            existing = InvestigationStage(
                "013-confirmation",
                HANDLER_ID,
                "COMPLETE",
                "011-transient-mode-validation",
                {},
                result={},
            )
            with self.assertRaisesRegex(RuntimeError, "already contains"):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    replace(
                        investigation,
                        stages=investigation.stages + (existing,),
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "terminal investigation"):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    replace(investigation, status="FAILED")
                )


if __name__ == "__main__":
    unittest.main()
