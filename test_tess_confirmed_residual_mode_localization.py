import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_investigation import InvestigationStage, InvestigationStore
from test_tess_confirmed_nonstationary_continuation import _confirmation
from test_tess_long_baseline_frequency_confirmation import INDEPENDENT_SECTORS
from workflows.tess.tess_nonstationary import (
    CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
    build_confirmed_nonstationary_method_contract,
    confirmed_nonstationary_method_contract_hash,
    validate_confirmed_nonstationary_localization_boundary,
)
from workflows.tess.tess_resolved_cycle import authoritative_resolved_cycle
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    _repair_confirmed_residual_localization_terminal,
)


def _boundary(paths):
    confirmation = _confirmation(paths)
    contract = build_confirmed_nonstationary_method_contract(confirmation)
    summary = {
        "evidenceLineage": CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
        "methodContractID": contract["methodContractID"],
        "methodContractHash": confirmed_nonstationary_method_contract_hash(
            contract
        ),
        "methodContract": contract,
        "classification": "AMPLITUDE_PHASE_EVOLVING_MODE",
        "preferredFrequencyAtReference": 0.304082882,
        "preferredPeriodAtReferenceDays": 3.288577149,
        "fractionalFrequencyDriftPerDay": 0.0,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
    }
    cycle = authoritative_resolved_cycle(morphology={
        "physicalCycleResolved": True,
        "resolvedPhysicalPeriodDays": 10.0,
        "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED",
    })
    return confirmation, summary, cycle


class ConfirmedResidualLocalizationBoundaryTests(unittest.TestCase):
    def test_accepts_authoritative_cycle_without_resolved_morphology_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation, summary, cycle = _boundary(paths)
        self.assertEqual(
            10.0,
            validate_confirmed_nonstationary_localization_boundary(
                summary, confirmation, cycle
            ),
        )

    def test_rejects_tampered_modeling_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation, summary, cycle = _boundary(paths)
        altered = copy.deepcopy(summary)
        altered["methodContract"]["modeling"]["driftGridCount"] += 1
        with self.assertRaisesRegex(RuntimeError, "exact persisted"):
            validate_confirmed_nonstationary_localization_boundary(
                altered, confirmation, cycle
            )

    def test_rejects_wrong_recommendation_or_claim_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation, summary, cycle = _boundary(paths)
        for key, value in (("recommendedNextTest", "OTHER"),
                           ("physicalMechanismResolved", True),
                           ("claimLevelChanged", True)):
            altered = copy.deepcopy(summary)
            altered[key] = value
            with self.assertRaisesRegex(RuntimeError, "exact persisted"):
                validate_confirmed_nonstationary_localization_boundary(
                    altered, confirmation, cycle
                )

    def test_rejects_missing_or_mismatched_authoritative_cycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation, summary, cycle = _boundary(paths)
        with self.assertRaisesRegex(RuntimeError, "authoritative"):
            validate_confirmed_nonstationary_localization_boundary(
                summary, confirmation, None
            )
        changed = copy.deepcopy(cycle)
        changed["periodDays"] = 11.0
        with self.assertRaises(RuntimeError):
            validate_confirmed_nonstationary_localization_boundary(
                summary, confirmation, changed
            )

    def test_terminal_repair_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation, summary, cycle = _boundary(paths)
            store = InvestigationStore(root / "investigations")
            investigation = store.create(
                "confirmed-localization", WORKFLOW_ID, WORKFLOW_VERSION,
            )
            stages = (
                InvestigationStage(
                    "006-localization",
                    "openstar.tess.source-localization.analyze",
                    "COMPLETE", None, {},
                    result={"physicalCycleEvidence": cycle},
                ),
                InvestigationStage(
                    "036-confirmation",
                    "openstar.tess.long-baseline-frequency-confirmation.analyze",
                    "COMPLETE", "006-localization", {}, result=confirmation,
                ),
                InvestigationStage(
                    "041-summarize-confirmed-nonstationary",
                    "openstar.tess.nonstationary.summarize",
                    "COMPLETE", "036-confirmation", {}, result=summary,
                ),
                InvestigationStage(
                    "042-finalize",
                    "openstar.tess.finalize",
                    "COMPLETE", "041-summarize-confirmed-nonstationary",
                    {"outputSuffix": "v20.9.2-confirmed-nonstationary"},
                    result={"nonstationaryModeling": summary}, stop=True,
                ),
            )
            investigation = replace(
                investigation, status="COMPLETE", stages=stages
            )
            store.save(investigation)
            control = {
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            }
            repaired = _repair_confirmed_residual_localization_terminal(
                store, investigation, control
            )
            repeated = _repair_confirmed_residual_localization_terminal(
                store,
                repaired,
                repaired.metadata["controlState"],
            )
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(stages, repaired.stages)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(
            "openstar.tess.residual-mode-localization.prepare",
            selected["handler_id"],
        )
        self.assertEqual(
            "041-summarize-confirmed-nonstationary",
            selected["triggered_by_stage_id"],
        )
        self.assertIsNone(repeated)


if __name__ == "__main__":
    unittest.main()
