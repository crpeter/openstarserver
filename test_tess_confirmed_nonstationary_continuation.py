import copy
import tempfile
import unittest
from pathlib import Path

from test_tess_long_baseline_frequency_confirmation import (
    INDEPENDENT_SECTORS,
    _mode_result,
)
from workflows.tess.tess_long_baseline_frequency_confirmation import (
    build_method_contract,
    classify_long_baseline_confirmation,
    method_contract_hash,
)
from workflows.tess.tess_nonstationary import (
    CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
    CONFIRMED_NONSTATIONARY_METHOD_CONTRACT_ID,
    build_confirmed_nonstationary_method_contract,
    confirmed_nonstationary_physical_period,
    confirmed_nonstationary_method_contract_hash,
    validate_confirmed_nonstationary_boundary,
)
from workflows.tess.tess_resolved_cycle import authoritative_resolved_cycle


def _confirmation(paths):
    source = build_method_contract(_mode_result(paths))
    folds = []
    supports = ("INDEPENDENT_MODE", "HARMONIC", "NEITHER", "NEITHER")
    frequencies = (0.32, 0.34, 0.36, 0.38)
    for sector, support, frequency in zip(
        INDEPENDENT_SECTORS, supports, frequencies, strict=True
    ):
        folds.append({
            "heldOutSector": sector,
            "trainingSectors": [item for item in INDEPENDENT_SECTORS
                                if item != sector],
            "support": support,
            "learnedIndependentFrequencyCyclesPerDay": frequency,
            "predictiveBIC": {"A": 90.0, "B": 88.0, "C": 100.0},
            "failureOrInsufficiencyReasons": [],
        })
    aggregate = classify_long_baseline_confirmation(
        folds, long_baseline_frequency_resolution=0.001)
    return {
        **aggregate,
        "methodContract": source,
        "methodContractHash": method_contract_hash(source),
        "perSectorEvidence": folds,
        "leaveOneIndependentSectorOut": True,
        "longBaselineDays": 400.0,
        "longBaselineFrequencyResolutionCyclesPerDay": 0.001,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "automaticDiscoveryClaim": False,
        "dataReuse": {
            "frozenDatasetPaths": [str(path.resolve()) for path in paths],
            "downloadPerformed": False,
        },
    }


class ConfirmedNonstationaryContractTests(unittest.TestCase):
    def test_uses_authoritative_cycle_when_morphology_stage_is_unresolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation = _confirmation(paths)
        cycle = authoritative_resolved_cycle(morphology={
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": 10.0,
            "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED",
        })
        self.assertEqual(
            10.0,
            confirmed_nonstationary_physical_period(confirmation, cycle),
        )
        with self.assertRaisesRegex(RuntimeError, "authoritative"):
            confirmed_nonstationary_physical_period(confirmation, None)

    def test_exact_boundary_builds_deterministic_versioned_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation = _confirmation(paths)
            first = build_confirmed_nonstationary_method_contract(confirmation)
            second = build_confirmed_nonstationary_method_contract(
                copy.deepcopy(confirmation))
        self.assertEqual(CONFIRMED_NONSTATIONARY_METHOD_CONTRACT_ID,
                         first["methodContractID"])
        self.assertEqual(CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
                         first["evidenceLineage"])
        self.assertEqual(first, second)
        self.assertEqual(confirmed_nonstationary_method_contract_hash(first),
                         confirmed_nonstationary_method_contract_hash(second))
        self.assertFalse(first["dataPolicy"]["downloadNewData"])
        self.assertFalse(first["claimPolicy"]["physicalMechanismResolved"])

    def test_rejects_altered_recommendation_or_claim_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation = _confirmation(paths)
        for key, value in (("recommendedNextTest", "OTHER"),
                           ("physicalMechanismResolved", True)):
            altered = copy.deepcopy(confirmation)
            altered[key] = value
            with self.assertRaises(RuntimeError):
                validate_confirmed_nonstationary_boundary(altered)

    def test_rejects_leave_one_sector_out_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation = _confirmation(paths)
        confirmation["perSectorEvidence"][0]["trainingSectors"].append(
            confirmation["perSectorEvidence"][0]["heldOutSector"])
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            validate_confirmed_nonstationary_boundary(confirmation)

    def test_rejects_insufficient_independent_sectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation = _confirmation(paths)
        confirmation["perSectorEvidence"][1]["support"] = "INSUFFICIENT"
        confirmation["perSectorEvidence"][2]["support"] = "INSUFFICIENT"
        with self.assertRaisesRegex(RuntimeError, "at least three"):
            validate_confirmed_nonstationary_boundary(confirmation)

    def test_rejects_tampered_source_contract_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"sector-{sector}.json"
                     for sector in (1, *INDEPENDENT_SECTORS)]
            confirmation = _confirmation(paths)
        confirmation["methodContract"]["crossValidation"][
            "heldOutFrequencySelection"] = True
        with self.assertRaisesRegex(RuntimeError, "method-contract"):
            validate_confirmed_nonstationary_boundary(confirmation)


if __name__ == "__main__":
    unittest.main()
