import json
import math
import tempfile
import unittest
from pathlib import Path

from workflows.tess.tess_binary_confirmation import (
    MORPHOLOGY_EVENT_SCREEN_ENTRY,
    _sector,
    analyze_binary_confirmation,
    morphology_event_screening_continuation,
    physical_interpretation_continuation,
)
from workflows.tess.tess_target_residual_astrophysical_interpretation import newest_authoritative_recommendation
from openstar_investigation import InvestigationStage


PERIOD = 2.0


def dataset(sector, origin, *, primary=True, secondary=False, phase_shift=0.0,
            broad=False, samples=700, cadence=0.02, local_shift=0.0):
    times = [local_shift + i * cadence for i in range(samples)]
    flux = []
    for i, time in enumerate(times):
        absolute_time = origin - local_shift + time
        phase = ((absolute_time / PERIOD) - phase_shift) % 1
        value = 0.35 * math.sin(2 * math.pi * phase) + 0.7 * math.cos(4 * math.pi * phase)
        value += 0.025 * math.sin(i * 1.731)  # deterministic, non-orbital noise
        distance = abs((phase + 0.5) % 1 - 0.5)
        if primary and distance < (0.20 if broad else 0.025) / 2:
            value -= 0.8
        if secondary and abs((phase - 0.5 + 0.5) % 1 - 0.5) < 0.02 / 2:
            value -= 0.35
        flux.append(value)
    return {"id": f"sector-{sector}", "source": {"sector": sector,
            "originalTimeOriginDays": origin - local_shift},
            "times": times, "flux": flux}


class BinaryConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.morphology = {"physicalCycleResolved": True,
                           "resolvedPhysicalPeriodDays": PERIOD,
                           "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED"}
        self.physical = {"recommendedNextTest": "INDEPENDENT_BINARY_CONFIRMATION",
                         "physicalMechanismResolved": False,
                         "preferredPhotometricHypothesis": "BINARY_LIKE_DOUBLE_WAVE"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, makers):
        primary = self.root / "primary.json"
        primary.write_text(json.dumps(dataset(1, 1000.0)), encoding="utf-8")
        sectors = []
        for sector, maker in enumerate(makers, 2):
            path = self.root / f"s{sector}.json"
            path.write_text(json.dumps(maker(sector, 1000.0 + (sector - 1) * 30.0)), encoding="utf-8")
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
        self.assertAlmostEqual(PERIOD, result["linearEphemeris"]["refinedPeriodDays"], places=6)
        self.assertTrue(all(item["timeReference"] ==
            "BTJD_RECONSTRUCTED_FROM_FROZEN_RELATIVE_TIME"
            for item in result["sectorResults"]))
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertFalse(result["companionNatureResolved"])
        self.assertFalse(result["catalogAnswerKeyUsed"])
        self.assertEqual("OPPOSITE_CONJUNCTION_EVENT_UNRESOLVED",
                         result["oppositeConjunctionEvidence"]["classification"])
        self.assertNotIn("radius", json.dumps(result).lower())

    def test_primary_epoch_expands_timing_baseline_only_after_replication(self):
        result = self.run_case([lambda s, t: dataset(s, t) for _ in range(4)])
        evidence = result["independentEvidence"]
        final = result["linearEphemeris"]
        independent = evidence["independentLinearEphemeris"]
        self.assertEqual("REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED", evidence["classification"])
        self.assertEqual(4, evidence["supportingIndependentSectorCount"])
        self.assertEqual(5, final["timingSectorCount"])
        self.assertTrue(final["primarySectorIncluded"])
        self.assertEqual("PRIMARY_PLUS_INDEPENDENT_AFTER_INDEPENDENT_REPLICATION",
                         final["timingEvidenceBasis"])
        final_span = (final["cycleAssignments"][-1]["cycleNumber"] -
                      final["cycleAssignments"][0]["cycleNumber"])
        independent_span = (independent["cycleAssignments"][-1]["cycleNumber"] -
                            independent["cycleAssignments"][0]["cycleNumber"])
        self.assertGreater(final_span, independent_span)
        self.assertAlmostEqual(PERIOD, final["refinedPeriodDays"], places=6)

    def test_primary_cannot_rescue_weak_independent_evidence(self):
        makers = ([lambda s, t: dataset(s, t)] * 2 +
                  [lambda s, t: dataset(s, t, primary=False)] * 2)
        result = self.run_case(makers)
        self.assertEqual("ECLIPSE_LIKE_EVENT_UNRESOLVED",
                         result["independentEvidence"]["classification"])
        self.assertEqual(2, result["independentEvidence"]["supportingIndependentSectorCount"])
        self.assertFalse(result["linearEphemeris"]["coherent"])
        self.assertNotIn("timingSectorCount", result["linearEphemeris"])

    def test_unusable_primary_leaves_independent_ephemeris_authoritative(self):
        primary = self.root / "primary.json"
        primary.write_text(json.dumps(dataset(1, 1000.0, primary=False)), encoding="utf-8")
        sectors = []
        for sector in range(2, 6):
            path = self.root / f"s{sector}.json"
            path.write_text(json.dumps(dataset(sector, 1000.0 + (sector - 1) * 30.0)),
                            encoding="utf-8")
            sectors.append({"sector": sector, "datasetPath": str(path)})
        result = analyze_binary_confirmation(primary_dataset_path=primary,
            independent_spec={"preparedSectors": sectors}, morphology=self.morphology,
            physical_interpretation=self.physical)
        self.assertEqual("REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
                         result["independentEvidence"]["classification"])
        self.assertFalse(result["linearEphemeris"]["primarySectorIncluded"])
        self.assertEqual(4, result["linearEphemeris"]["timingSectorCount"])
        self.assertEqual("INDEPENDENT_ONLY_PRIMARY_UNUSABLE",
                         result["linearEphemeris"]["timingEvidenceBasis"])

    def test_primary_timing_outlier_falls_back_without_erasing_replication(self):
        primary = self.root / "primary.json"
        primary.write_text(json.dumps(dataset(1, 1000.0, phase_shift=0.20)), encoding="utf-8")
        sectors = []
        for sector in range(2, 6):
            path = self.root / f"s{sector}.json"
            path.write_text(json.dumps(dataset(sector, 1000.0 + (sector - 1) * 30.0)),
                            encoding="utf-8")
            sectors.append({"sector": sector, "datasetPath": str(path)})
        result = analyze_binary_confirmation(primary_dataset_path=primary,
            independent_spec={"preparedSectors": sectors}, morphology=self.morphology,
            physical_interpretation=self.physical)
        self.assertEqual("REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
                         result["independentEvidence"]["classification"])
        final = result["linearEphemeris"]
        self.assertTrue(final["coherent"])
        self.assertFalse(final["primarySectorIncluded"])
        self.assertFalse(final["primaryTimingConsistent"])
        self.assertFalse(final["expandedTimingAttempt"]["coherent"])
        self.assertEqual("ECLIPSE_TIMING_REFINEMENT_REVIEW", result["recommendedNextTest"])

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

    def test_relative_time_translation_leaves_ephemeris_unchanged(self):
        baseline = self.run_case([lambda s, origin: dataset(s, origin) for _ in range(4)])
        shifted = self.run_case([lambda s, origin: dataset(s, origin, local_shift=17.25 + s)
                                 for _ in range(4)])
        for key in ("referenceEpoch", "refinedPeriodDays", "rmsOMinusCDays"):
            self.assertAlmostEqual(baseline["linearEphemeris"][key],
                                   shifted["linearEphemeris"][key], places=9)

    def test_missing_or_nonfinite_time_origin_fails_closed(self):
        for invalid in (None, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                frozen = dataset(2, 1030.0)
                if invalid is None:
                    frozen["source"].pop("originalTimeOriginDays")
                else:
                    frozen["source"]["originalTimeOriginDays"] = invalid
                with self.assertRaisesRegex(ValueError, "originalTimeOriginDays"):
                    _sector(frozen, PERIOD, "INDEPENDENT")

    def test_realistic_18000_sample_fixed_period_search(self):
        measured = _sector(dataset(2, 1030.0, samples=18_000, cadence=0.001),
                           PERIOD, "INDEPENDENT")
        self.assertTrue(measured["usable"])
        self.assertEqual(18_000, measured["sampleCount"])

    def test_exact_authoritative_routing_gate(self):
        self.assertTrue(physical_interpretation_continuation(self.physical, self.morphology))
        for key, value in (("recommendedNextTest", "OTHER"),
                           ("physicalMechanismResolved", True),
                           ("preferredPhotometricHypothesis", "ROTATION_LIKE")):
            changed = dict(self.physical); changed[key] = value
            self.assertFalse(physical_interpretation_continuation(changed, self.morphology))

    def test_resolved_double_wave_enters_blind_event_screen_without_physical_label(self):
        independent = {
            "preparedSectors": [
                {"datasetPath": f"sector-{sector}.json"}
                for sector in range(2, 5)
            ]
        }
        self.assertTrue(
            morphology_event_screening_continuation(self.morphology, independent)
        )
        for changed in (
            {**self.morphology, "physicalCycleResolved": False},
            {**self.morphology, "resolvedPhysicalPeriodDays": float("nan")},
            {**self.morphology, "morphologyClass": "RAW_CYCLE_SUPPORTED"},
        ):
            self.assertFalse(
                morphology_event_screening_continuation(changed, independent)
            )
        self.assertFalse(
            morphology_event_screening_continuation(
                self.morphology,
                {"preparedSectors": independent["preparedSectors"][:2]},
            )
        )

        primary = self.root / "screen-primary.json"
        primary.write_text(json.dumps(dataset(1, 1000.0)), encoding="utf-8")
        sectors = []
        for sector in range(2, 6):
            path = self.root / f"screen-{sector}.json"
            path.write_text(
                json.dumps(dataset(sector, 1000.0 + (sector - 1) * 30.0)),
                encoding="utf-8",
            )
            sectors.append({"sector": sector, "datasetPath": str(path)})
        result = analyze_binary_confirmation(
            primary_dataset_path=primary,
            independent_spec={"preparedSectors": sectors},
            morphology=self.morphology,
            physical_interpretation=None,
            entry_mode=MORPHOLOGY_EVENT_SCREEN_ENTRY,
        )
        self.assertEqual(MORPHOLOGY_EVENT_SCREEN_ENTRY, result["entryBoundary"])
        self.assertEqual(
            "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
            result["independentEvidence"]["classification"],
        )
        self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_long_baseline_event_epochs_correct_a_biased_morphology_period(self):
        primary = self.root / "accuracy-primary.json"
        primary.write_text(json.dumps(dataset(1, 1000.0)), encoding="utf-8")
        sectors = []
        for sector, origin in zip(range(2, 6), (1200.0, 1400.0, 1600.0, 1800.0)):
            path = self.root / f"accuracy-{sector}.json"
            path.write_text(json.dumps(dataset(sector, origin)), encoding="utf-8")
            sectors.append({"sector": sector, "datasetPath": str(path)})
        biased_period = PERIOD * 1.00036
        result = analyze_binary_confirmation(
            primary_dataset_path=primary,
            independent_spec={"preparedSectors": sectors},
            morphology={
                **self.morphology,
                "resolvedPhysicalPeriodDays": biased_period,
            },
            physical_interpretation=None,
            entry_mode=MORPHOLOGY_EVENT_SCREEN_ENTRY,
        )
        refined = result["linearEphemeris"]["refinedPeriodDays"]
        self.assertGreater(abs(biased_period - PERIOD), 0.0007)
        self.assertAlmostEqual(PERIOD, refined, places=8)
        self.assertLess(abs(refined - PERIOD), abs(biased_period - PERIOD) / 1000.0)

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
