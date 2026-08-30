import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from workflows.tess.tess_physical import (
    analyze_physical_interpretation,
    physical_source_localization_continuation,
)
from workflows.tess.tess_resolved_cycle import (
    MORPHOLOGY_SOURCE,
    NESTED_ALIAS_SOURCE,
    authoritative_resolved_cycle,
    validated_cycle_period,
)


def nested_result(raw_period=6.5):
    resolved_period = 2.0 * raw_period
    comparisons = [
        {
            "sector": sector,
            "role": "PRIMARY" if sector == 1 else "INDEPENDENT",
            "oddHarmonicStructureSupported": sector != 1,
        }
        for sector in (1, 2, 4, 98)
    ]
    return {
        "evidenceLineage":
        "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT",
        "classification":
        "DOUBLE_CYCLE_ODD_HARMONICS_PREDICTIVELY_SUPPORTED",
        "rawFamilyPeriodDays": raw_period,
        "possibleDoubleCycleDays": resolved_period,
        "resolvedPhysicalPeriodDays": resolved_period,
        "referenceFamilyPeriodDays": resolved_period,
        "referencePeriodRole": "PREDICTIVELY_RESOLVED_PHOTOMETRIC_CYCLE",
        "physicalCycleResolved": True,
        "physicalMechanismResolved": False,
        "recommendedNextTest": "BINARY_ROTATION_EXTERNAL_EVIDENCE",
        "periodAliasResolution": {
            "method": (
                "NESTED_EVEN_ONLY_VS_EVEN_PLUS_ODD_LEAVE_ONE_SECTOR_OUT_"
                "PREDICTION"
            ),
            "criterion": "BIC",
            "conservativeThreshold": 10.0,
            "equalHalfEvenHarmonicOrders": [2, 4, 6, 8],
            "discriminatingOddHarmonicOrders": [1, 3, 5, 7],
            "fullDoubleCycleHarmonicOrders": list(range(1, 9)),
            "maximumAbsoluteFrequencyMatched": True,
            "primarySector": 1,
            "minimumSupportingIndependentHeldOutSectors": 3,
            "aggregateIndependentDeltaBicFullMinusEvenOnly": 125.0,
            "oddHarmonicSupportingIndependentHeldOutSectors": [2, 4, 98],
            "selectedPeriodRelation": "DOUBLE_CYCLE",
            "selectedPeriodDays": resolved_period,
            "physicalCycleResolved": True,
            "comparisons": comparisons,
        },
    }


class AuthoritativeResolvedCycleTests(unittest.TestCase):
    def test_exact_nested_prediction_becomes_authoritative_contract(self):
        cycle = authoritative_resolved_cycle(
            morphology=None, dynamic_harmonic=nested_result())
        self.assertEqual(NESTED_ALIAS_SOURCE, cycle["sourceKind"])
        self.assertEqual([2, 4, 98], cycle["supportingIndependentSectors"])
        self.assertEqual(13.0, validated_cycle_period(cycle))

    def test_legacy_morphology_resolution_remains_supported(self):
        cycle = authoritative_resolved_cycle(morphology={
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": 2.0,
            "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED",
        })
        self.assertEqual(MORPHOLOGY_SOURCE, cycle["sourceKind"])
        self.assertEqual(2.0, validated_cycle_period(cycle))

    def test_nested_contract_rejects_science_gate_tampering(self):
        for path, value in (
            (("periodAliasResolution", "conservativeThreshold"), 9.0),
            (("periodAliasResolution", "minimumSupportingIndependentHeldOutSectors"), 2),
            (("periodAliasResolution", "maximumAbsoluteFrequencyMatched"), False),
            (("periodAliasResolution", "equalHalfEvenHarmonicOrders"), [2, 4]),
            (("periodAliasResolution", "method"), "OTHER"),
        ):
            with self.subTest(path=path):
                result = copy.deepcopy(nested_result())
                result[path[0]][path[1]] = value
                self.assertIsNone(authoritative_resolved_cycle(
                    morphology=None, dynamic_harmonic=result))

    def test_primary_cannot_count_as_independent_support(self):
        result = nested_result()
        result["periodAliasResolution"][
            "oddHarmonicSupportingIndependentHeldOutSectors"] = [1, 2, 4]
        self.assertIsNone(authoritative_resolved_cycle(
            morphology=None, dynamic_harmonic=result))

    def test_malformed_comparison_sector_fails_closed(self):
        result = nested_result()
        result["periodAliasResolution"]["comparisons"][1]["sector"] = "bad"
        self.assertIsNone(authoritative_resolved_cycle(
            morphology=None, dynamic_harmonic=result))

    def test_inconsistent_morphology_and_nested_periods_fail_closed(self):
        self.assertIsNone(authoritative_resolved_cycle(
            morphology={
                "physicalCycleResolved": True,
                "resolvedPhysicalPeriodDays": 12.0,
            },
            dynamic_harmonic=nested_result(),
        ))

    def test_physical_interpreter_uses_nested_contract_with_unresolved_morphology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for sector in (1, 2, 4, 98):
                times = [index * 0.02 for index in range(400)]
                flux = [
                    math.sin(math.pi * time)
                    + 0.4 * math.cos(2.0 * math.pi * time)
                    for time in times
                ]
                path = root / f"sector-{sector}.json"
                path.write_text(json.dumps({
                    "id": f"sector-{sector}",
                    "source": {"sector": sector},
                    "times": times,
                    "flux": flux,
                }), encoding="utf-8")
                paths.append(path)
            cycle = authoritative_resolved_cycle(
                morphology=None,
                dynamic_harmonic=nested_result(raw_period=1.0),
            )
            morphology = {
                "physicalCycleResolved": False,
                "resolvedPhysicalPeriodDays": None,
            }
            result = analyze_physical_interpretation(
                primary_dataset_path=paths[0],
                independent_spec={"preparedSectors": [
                    {"sector": sector, "datasetPath": str(path)}
                    for sector, path in zip((2, 4, 98), paths[1:])
                ]},
                identity={},
                morphology=morphology,
                broad_interpretation=None,
                resolved_cycle=cycle,
            )
            self.assertEqual(2.0, result["physicalPeriodDays"])
            self.assertEqual(cycle, result["physicalCycleEvidence"])
            self.assertFalse(morphology["physicalCycleResolved"])

    def test_exact_physical_contamination_boundary_routes_localization(self):
        cycle = authoritative_resolved_cycle(
            morphology=None, dynamic_harmonic=nested_result())
        physical = {
            "version": "openstar.tess-physical-interpretation.v2",
            "physicalPeriodDays": 13.0,
            "photometricFirstHarmonicPeriodDays": 6.5,
            "physicalCycleEvidence": cycle,
            "physicalMechanismResolved": False,
            "contaminationScreen": {"flaggedByExistingMetadata": True},
            "recommendedNextTest": "PIXEL_LEVEL_SOURCE_LOCALIZATION",
        }
        self.assertTrue(physical_source_localization_continuation(
            physical, cycle))
        for key, value in (
            ("physicalPeriodDays", 12.0),
            ("physicalCycleEvidence", None),
            ("physicalMechanismResolved", True),
            ("recommendedNextTest", "OTHER"),
        ):
            with self.subTest(key=key):
                changed = {**physical, key: value}
                self.assertFalse(physical_source_localization_continuation(
                    changed, cycle))
        changed = copy.deepcopy(physical)
        changed["contaminationScreen"]["flaggedByExistingMetadata"] = False
        self.assertFalse(physical_source_localization_continuation(
            changed, cycle))


if __name__ == "__main__":
    unittest.main()
