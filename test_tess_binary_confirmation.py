import json
import math
import tempfile
import unittest
from pathlib import Path

from workflows.tess.tess_binary_confirmation import (
    _sector,
    analyze_binary_confirmation,
    physical_interpretation_continuation,
)
from workflows.tess.tess_target_residual_astrophysical_interpretation import newest_authoritative_recommendation
from openstar_investigation import InvestigationStage


PERIOD = 2.0


def dataset(sector, start, *, primary=True, secondary=False, phase_shift=0.0,
            broad=False):
    times = [start + i * 0.02 for i in range(700)]
    flux = []
    for i, time in enumerate(times):
        phase = ((time / PERIOD) - phase_shift) % 1
        value = 0.35 * math.sin(2 * math.pi * phase) + 0.7 * math.cos(4 * math.pi * phase)
        value += 0.025 * math.sin(i * 1.731)  # deterministic, non-orbital noise
        distance = abs((phase + 0.5) % 1 - 0.5)
        if primary and distance < (0.20 if broad else 0.025) / 2:
            value -= 0.8
        if secondary and abs((phase - 0.5 + 0.5) % 1 - 0.5) < 0.02 / 2:
            value -= 0.35
        flux.append(value)
    return {"id": f"sector-{sector}", "source": {"sector": sector},
            "times": times, "flux": flux}


class BinaryConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.morphology = {"physicalCycleResolved": True,
                           "resolvedPhysicalPeriodDays": PERIOD}
        self.physical = {"recommendedNextTest": "INDEPENDENT_BINARY_CONFIRMATION",
                         "physicalMechanismResolved": False,
                         "preferredPhotometricHypothesis": "BINARY_LIKE_DOUBLE_WAVE"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, makers):
        primary = self.root / "primary.json"
        primary.write_text(json.dumps(dataset(1, 0)), encoding="utf-8")
        sectors = []
        for sector, maker in enumerate(makers, 2):
            path = self.root / f"s{sector}.json"
            path.write_text(json.dumps(maker(sector, (sector - 1) * 30.0)), encoding="utf-8")
            sectors.append({"sector": sector, "datasetPath": str(path)})
        return analyze_binary_confirmation(primary_dataset_path=primary,
            independent_spec={"preparedSectors": sectors}, morphology=self.morphology,
            physical_interpretation=self.physical)

    def test_replicated_eclipse_and_no_secondary(self):
        result = self.run_case([lambda s, t: dataset(s, t) for _ in range(4)])
        self.assertEqual("REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
                         result["independentEvidence"]["classification"])
        self.assertGreaterEqual(result["independentEvidence"]["supportingIndependentSectorCount"], 3)
        self.assertTrue(result["linearEphemeris"]["coherent"])
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertFalse(result["companionNatureResolved"])
        self.assertFalse(result["catalogAnswerKeyUsed"])
        self.assertEqual("OPPOSITE_CONJUNCTION_EVENT_UNRESOLVED",
                         result["oppositeConjunctionEvidence"]["classification"])
        self.assertNotIn("radius", json.dumps(result).lower())

    def test_pure_double_wave_fails_closed(self):
        result = self.run_case([lambda s, t: dataset(s, t, primary=False) for _ in range(4)])
        self.assertEqual("ECLIPSE_LIKE_EVENT_UNRESOLVED",
                         result["independentEvidence"]["classification"])

    def test_evolving_rotation_does_not_manufacture_event(self):
        def evolving(s, start):
            value = dataset(s, start, primary=False)
            value["flux"] = [f + 0.08 * math.sin(i / 180 + s) for i, f in enumerate(value["flux"])]
            return value
        result = self.run_case([evolving] * 4)
        self.assertEqual("ECLIPSE_LIKE_EVENT_UNRESOLVED",
                         result["independentEvidence"]["classification"])

    def test_single_sector_is_insufficient(self):
        makers = [lambda s, t: dataset(s, t)] + [lambda s, t: dataset(s, t, primary=False)] * 3
        result = self.run_case(makers)
        self.assertEqual("ECLIPSE_LIKE_EVENT_UNRESOLVED",
                         result["independentEvidence"]["classification"])

    def test_incoherent_timings_fail_ephemeris(self):
        shifts = [0.0, 0.08, -0.09, 0.12]
        result = self.run_case([lambda s, t, shift=shift: dataset(s, t, phase_shift=shift)
                                for shift in shifts])
        self.assertFalse(result["linearEphemeris"]["coherent"])
        self.assertEqual("ECLIPSE_LIKE_EVENT_UNRESOLVED",
                         result["independentEvidence"]["classification"])

    def test_maximum_duration_boundary_rejected(self):
        measured = _sector(dataset(2, 30, broad=True), PERIOD, "INDEPENDENT")
        self.assertTrue(measured["durationBoundaryHit"])
        self.assertEqual("MAXIMUM", measured["durationBoundary"])
        self.assertFalse(measured["usable"])

    def test_secondary_is_separate_supporting_evidence(self):
        result = self.run_case([lambda s, t: dataset(s, t, secondary=True) for _ in range(4)])
        self.assertEqual("REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
                         result["independentEvidence"]["classification"])
        self.assertEqual("OPPOSITE_CONJUNCTION_EVENT_SUPPORTED",
                         result["oppositeConjunctionEvidence"]["classification"])
        self.assertEqual("SAME_STANDARDIZED_LIGHT_CURVE_NONPHYSICAL",
                         result["oppositeConjunctionEvidence"]["depthNormalization"])

    def test_masked_refit_preserves_narrow_event(self):
        measured = _sector(dataset(2, 30), PERIOD, "INDEPENDENT")
        self.assertTrue(measured["usable"])
        self.assertGreater(measured["depthStandardized"], 0.65)
        self.assertTrue(measured["smoothModel"]["candidateEventMaskedDuringRefit"])

    def test_exact_authoritative_routing_gate(self):
        self.assertTrue(physical_interpretation_continuation(self.physical, self.morphology))
        for key, value in (("recommendedNextTest", "OTHER"),
                           ("physicalMechanismResolved", True),
                           ("preferredPhotometricHypothesis", "ROTATION_LIKE")):
            changed = dict(self.physical); changed[key] = value
            self.assertFalse(physical_interpretation_continuation(changed, self.morphology))

    def test_newest_science_recommendation_wins(self):
        stages = [
            InvestigationStage("1", "openstar.tess.time-frequency.summarize", "COMPLETE", None, {},
                               result={"recommendedNextTest": "BINARY_ROTATION_EXTERNAL_EVIDENCE"}),
            InvestigationStage("2", "openstar.tess.physical.interpret", "COMPLETE", "1", {},
                               result={"recommendedNextTest": "INDEPENDENT_BINARY_CONFIRMATION"}),
        ]
        self.assertEqual("INDEPENDENT_BINARY_CONFIRMATION",
                         newest_authoritative_recommendation(stages))
        stages.append(InvestigationStage("3", "openstar.tess.binary-confirmation.analyze",
            "COMPLETE", "2", {}, result={"recommendedNextTest": "ECLIPSE_EVENT_SOURCE_LOCALIZATION"}))
        self.assertEqual("ECLIPSE_EVENT_SOURCE_LOCALIZATION",
                         newest_authoritative_recommendation(stages))

    def test_provenance_names_are_wired_without_answer_key(self):
        source = Path("workflows/tess/tess_investigation.py").read_text(encoding="utf-8")
        block = source[source.index("def binary_confirmation_stage"):source.index("def source_localization_stage")]
        for name in ("physicalInterpretation", "morphology", "primaryDataset",
                     "independentPreparation", "independentSector"):
            self.assertIn(name, block)
        self.assertNotIn("catalogAnswerKey", block)


if __name__ == "__main__":
    unittest.main()
