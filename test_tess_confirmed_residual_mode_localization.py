import copy
import tempfile
import unittest
from pathlib import Path

from test_tess_confirmed_nonstationary_continuation import _confirmation
from test_tess_long_baseline_frequency_confirmation import INDEPENDENT_SECTORS
from workflows.tess.tess_nonstationary import (
    CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
    build_confirmed_nonstationary_method_contract,
    confirmed_nonstationary_method_contract_hash,
    validate_confirmed_nonstationary_localization_boundary,
)
from workflows.tess.tess_resolved_cycle import authoritative_resolved_cycle


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


if __name__ == "__main__":
    unittest.main()
