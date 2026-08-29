import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION
from workflows.tess.tess_investigation import build_engine
from workflows.tess import tess_blind_transit_search

from workflows.tess.tess_blind_transit_search import (
    HANDLER_ID,
    UNRELIABLE_PRIMARY_ENTRY,
    analyze_blind_transit_search,
    analyze_iterative_blind_transit_search,
    blind_transit_search_continuation,
)


class BlindTransitSearchTests(unittest.TestCase):
    def _complete(self, store, investigation, stage_id, handler_id, result):
        running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
        investigation = store.append_running_stage(investigation, running)
        terminal = store.build_terminal_stage(
            stage_id=stage_id, handler_id=handler_id, status="COMPLETE",
            triggered_by_stage_id=None, parameters={}, result=result, error=None,
            software_id="test", software_version="1", started_at=running.started_at,
        )
        return store.complete_current_stage(investigation, terminal)

    def _dataset(self, root: Path, *, sector: int, origin: float,
                 transit: bool = True, period: float = 2.21857567,
                 additional_transits=()) -> Path:
        epoch = 1000.37
        duration = 0.08
        relative_times = [0.01 * index for index in range(2601)]
        flux = []
        for relative in relative_times:
            absolute = origin + relative
            value = (
                0.05 * math.sin(2.0 * math.pi * absolute / 6.36 + 0.1 * sector)
                + 0.012 * math.sin(2.0 * math.pi * absolute / 10.7 - 0.2 * sector)
                + 0.002 * math.sin(2.0 * math.pi * absolute / 0.173 + sector)
            )
            phase_days = abs((absolute - epoch + period / 2.0) % period - period / 2.0)
            if transit and phase_days <= duration / 2.0:
                value -= 0.04
            for extra_period, extra_epoch, extra_duration, extra_depth in (
                additional_transits
            ):
                extra_phase_days = abs(
                    (absolute - extra_epoch + extra_period / 2.0)
                    % extra_period - extra_period / 2.0
                )
                if extra_phase_days <= extra_duration / 2.0:
                    value -= extra_depth
            flux.append(value)
        path = root / f"sector-{sector}.json"
        path.write_text(json.dumps({
            "id": f"sector-{sector}",
            "source": {
                "sector": sector,
                "baselineDays": relative_times[-1],
                "originalTimeOriginDays": origin,
            },
            "frequencySearch": {
                "minimumFrequency": 0.1,
                "maximumFrequency": 5.0,
            },
            "times": relative_times,
            "flux": flux,
        }), encoding="utf-8")
        return path

    def _inputs(self, root: Path, *, transit: bool = True):
        paths = [
            self._dataset(root, sector=sector, origin=origin, transit=transit)
            for sector, origin in ((41, 1000.0), (54, 1700.0), (81, 2400.0))
        ]
        independent = {
            "investigationGoal": "FULL_CHARACTERIZATION",
            "preparedSectors": [
                {"sector": sector, "datasetPath": str(path)}
                for sector, path in zip((54, 81), paths[1:])
            ],
        }
        morphology = {"physicalCycleResolved": False}
        broad = {"claimDecision": {"claim": "CANDIDATE_PERIOD"}}
        return paths[0], independent, morphology, broad

    def _prepared_pooled_sectors(
        self, *, period: float = 5.0, late_transits: bool = True
    ):
        import numpy as np

        sectors = []
        epoch = 0.37
        specifications = [("PRIMARY", 1, 0.0, 1.8)] + [
            (
                "INDEPENDENT",
                index + 2,
                100.0 * (index + 1),
                1.0 if late_transits or index < 4 else 0.0,
            )
            for index in range(8)
        ]
        for role, sector, origin, depth in specifications:
            times = np.arange(origin, origin + 20.0, 0.01)
            residual = np.zeros_like(times)
            distance = np.abs(
                np.remainder(times - epoch + period / 2.0, period)
                - period / 2.0
            )
            residual[distance <= 0.025] -= depth
            sectors.append({
                "role": role,
                "sector": sector,
                "datasetID": f"sector-{sector}",
                "times": times,
                "residual": residual,
                "sigma": 1.0,
                "origin": origin,
                "cadence": 0.01,
                "detrendWindowSamples": 75,
            })
        return sectors

    def _prepared_joint_outlier_sectors(self):
        import numpy as np

        period = 6.2713
        epoch = 0.41
        specifications = (
            ("PRIMARY", 1, 0.0),
            ("INDEPENDENT", 4, 100.0),
            ("INDEPENDENT", 8, 700.0),
            ("INDEPENDENT", 11, 1400.0),
            ("INDEPENDENT", 12, 2100.0),
        )
        sectors = []
        for role, sector, origin in specifications:
            times = np.arange(origin, origin + 25.0, 0.02)
            residual = np.zeros_like(times)
            distance = np.abs(
                np.remainder(times - epoch + period / 2.0, period)
                - period / 2.0
            )
            residual[distance <= 0.05] -= 2.0
            if sector in {4, 8}:
                wrong_phase = 0.25 if sector == 4 else 0.40
                wrong_epoch = epoch + wrong_phase * period
                wrong_distance = np.abs(
                    np.remainder(
                        times - wrong_epoch + period / 2.0, period
                    ) - period / 2.0
                )
                residual[wrong_distance <= 0.05] -= 6.0 if sector == 4 else 5.0
            sectors.append({
                "role": role,
                "sector": sector,
                "datasetID": f"sector-{sector}",
                "times": times,
                "residual": residual,
                "sigma": 1.0,
                "origin": origin,
                "cadence": 0.02,
                "detrendWindowSamples": 37,
            })
        return sectors, period

    def test_entry_gate_is_exact_and_requires_two_independent_sectors(self):
        morphology = {"physicalCycleResolved": False}
        independent = {
            "investigationGoal": "FULL_CHARACTERIZATION",
            "preparedSectors": [
                {"datasetPath": "/one"},
                {"datasetPath": "/two"},
            ],
        }
        broad = {"claimDecision": {"claim": "CANDIDATE_PERIOD"}}
        self.assertTrue(blind_transit_search_continuation(morphology, independent, broad))
        self.assertFalse(blind_transit_search_continuation(
            morphology, {**independent, "investigationGoal": None}, broad))
        self.assertFalse(blind_transit_search_continuation(
            {"physicalCycleResolved": True}, independent, broad))
        self.assertFalse(blind_transit_search_continuation(
            morphology, {**independent, "preparedSectors": independent["preparedSectors"][:1]}, broad))
        self.assertFalse(blind_transit_search_continuation(morphology, independent, None))

        targeted = {
            "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
            "primaryBoundaryHit": True,
            "supportingSectorCount": 0,
            "contradictionPlan": {
                "action": "STOP",
                "reason": "insufficient-independent-evidence-for-broad-contradiction-search",
            },
        }
        self.assertTrue(blind_transit_search_continuation(
            None, independent, None, targeted
        ))
        self.assertFalse(blind_transit_search_continuation(
            None, independent, None, {**targeted, "primaryBoundaryHit": False}
        ))
        self.assertFalse(blind_transit_search_continuation(
            None, independent, None, {**targeted, "supportingSectorCount": 1}
        ))

        unreliable = {
            **targeted,
            "primaryBoundaryHit": False,
            "primaryReliable": False,
        }
        self.assertTrue(blind_transit_search_continuation(
            None, independent, None, unreliable
        ))
        self.assertTrue(blind_transit_search_continuation(
            None,
            independent,
            None,
            {
                **unreliable,
                "contradictionPlan": {
                    "action": "BROAD_INDEPENDENT_SEARCH",
                    "reason": "alternate-reliable-structure",
                },
            },
        ))
        self.assertFalse(blind_transit_search_continuation(
            None, independent, None, {**unreliable, "primaryReliable": None}
        ))
        self.assertTrue(blind_transit_search_continuation(
            None,
            independent,
            None,
            {
                **unreliable,
                "claimDecision": {"claim": "CANDIDATE_PERIOD"},
                "supportingSectorCount": 1,
                "requiredSupportingSectorCount": 2,
            },
        ))

    def test_recovers_repeated_narrow_period_beneath_stronger_smooth_variability(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, independent, morphology, broad = self._inputs(Path(temporary))
            result = analyze_blind_transit_search(
                primary_dataset_path=primary,
                independent_spec=independent,
                morphology=morphology,
                broad_interpretation=broad,
            )

        self.assertEqual("REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                         result["classification"])
        self.assertAlmostEqual(2.21857567, result["candidatePeriodDays"], delta=0.003)
        self.assertTrue(result["primarySectorSupported"])
        self.assertEqual(2, result["supportingIndependentSectorCount"])
        self.assertTrue(result["linearEphemeris"]["coherent"])
        self.assertEqual("CANDIDATE_PERIOD", result["claimDecision"]["claim"])
        self.assertEqual("ADDITIONAL_INDEPENDENT_SECTOR_TRANSIT_CONFIRMATION",
                         result["recommendedNextTest"])
        self.assertFalse(result["catalogAnswerKeyUsed"])
        self.assertFalse(result["physicalCycleResolved"])
        self.assertFalse(result["companionNatureResolved"])
        self.assertEqual(
            "SHARED_PERIOD_EPOCH_DURATION_BOX_SEARCH",
            result["jointTransitSearch"]["method"],
        )
        self.assertEqual(
            "MINIMUM_TRANSIT_DUTY_CYCLE_OVER_FULL_OBSERVATION_SPAN",
            result["searchGrid"]["frequencyResolutionBasis"],
        )
        self.assertEqual(1, len({
            item["eventPhase"] for item in result["sectorResults"]
        }))
        self.assertEqual(
            "RETAIN_BASE_PERIOD",
            result["alternatingCycleAliasResolution"]["decision"],
        )

    def test_iterative_search_masks_first_clock_and_recovers_second(self):
        first_period = 2.21857567
        second_period = 3.133
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                self._dataset(
                    root,
                    sector=sector,
                    origin=origin,
                    period=first_period,
                    additional_transits=((second_period, 1001.11, 0.06, 0.05),),
                )
                for sector, origin in ((41, 1000.0), (54, 1700.0), (81, 2400.0))
            ]
            independent = {
                "investigationGoal": "FULL_CHARACTERIZATION",
                "preparedSectors": [
                    {"sector": sector, "datasetPath": str(path)}
                    for sector, path in zip((54, 81), paths[1:])
                ],
            }
            result = analyze_iterative_blind_transit_search(
                primary_dataset_path=paths[0],
                independent_spec=independent,
                morphology={"physicalCycleResolved": False},
                broad_interpretation={
                    "claimDecision": {"claim": "CANDIDATE_PERIOD"}
                },
                maximum_candidates=2,
            )

        candidates = result["candidateSignals"]
        self.assertEqual(2, len(candidates))
        recovered = sorted(item["candidatePeriodDays"] for item in candidates)
        expected = sorted((first_period, second_period))
        for measured, target in zip(recovered, expected):
            self.assertAlmostEqual(target, measured, delta=0.004)
        self.assertAlmostEqual(
            candidates[0]["candidatePeriodDays"],
            result["candidatePeriodDays"],
            delta=1e-12,
        )
        self.assertGreater(
            candidates[0]["residualSearchMask"]["totalRemovedSampleCount"], 0
        )
        self.assertTrue(result["iterativeSearch"]["iterations"][1][
            "frequencyFamilySeparation"
        ]["distinct"])
        self.assertEqual(
            "MAXIMUM_CANDIDATE_COUNT_REACHED",
            result["iterativeSearch"]["terminationReason"],
        )
        self.assertTrue(all(
            item["linearEphemeris"]["coherent"] for item in candidates
        ))
        self.assertTrue(all(
            item["catalogAnswerKeyUsed"] is False for item in candidates
        ))
        self.assertFalse(result["iterativeSearch"]["catalogAnswerKeyUsed"])

    def test_iterative_search_recovers_three_distinct_shared_clocks(self):
        periods = (2.21857567, 3.133, 7.17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                self._dataset(
                    root,
                    sector=sector,
                    origin=origin,
                    period=periods[0],
                    additional_transits=(
                        (periods[1], 1001.11, 0.06, 0.05),
                        (periods[2], 1002.03, 0.10, 0.045),
                    ),
                )
                for sector, origin in ((41, 1000.0), (54, 1700.0), (81, 2400.0))
            ]
            independent = {
                "investigationGoal": "FULL_CHARACTERIZATION",
                "preparedSectors": [
                    {"sector": sector, "datasetPath": str(path)}
                    for sector, path in zip((54, 81), paths[1:])
                ],
            }
            result = analyze_iterative_blind_transit_search(
                primary_dataset_path=paths[0],
                independent_spec=independent,
                morphology={"physicalCycleResolved": False},
                broad_interpretation={
                    "claimDecision": {"claim": "CANDIDATE_PERIOD"}
                },
                maximum_candidates=4,
            )

        candidates = result["candidateSignals"]
        self.assertEqual(3, len(candidates))
        recovered = sorted(item["candidatePeriodDays"] for item in candidates)
        for measured, target in zip(recovered, sorted(periods)):
            self.assertAlmostEqual(target, measured, delta=0.004)
        longest = max(candidates, key=lambda item: item["candidatePeriodDays"])
        alias = longest["alternatingCycleAliasResolution"]
        self.assertEqual("PROMOTE_INTEGER_MULTIPLE_PERIOD", alias["decision"])
        self.assertEqual(4, alias["selectedCycleMultiplier"])
        self.assertEqual(
            "NEXT_RESIDUAL_SIGNAL_UNRESOLVED",
            result["iterativeSearch"]["terminationReason"],
        )
        self.assertTrue(all(
            item["linearEphemeris"]["coherent"] for item in candidates
        ))
        self.assertFalse(result["iterativeSearch"]["catalogAnswerKeyUsed"])

    def test_distinct_frequency_gate_rejects_exact_alias_but_allows_near_resonance(self):
        prior = {
            "candidateIndex": 1,
            "jointTransitSearch": {"frequencyPerDay": 0.5},
            "searchGrid": {"fineFrequencyStepPerDay": 0.00001},
        }
        exact_harmonic = {
            "jointTransitSearch": {"frequencyPerDay": 1.0},
            "searchGrid": {"fineFrequencyStepPerDay": 0.00001},
        }
        near_resonance = {
            "jointTransitSearch": {"frequencyPerDay": 0.99},
            "searchGrid": {"fineFrequencyStepPerDay": 0.00001},
        }

        distinct, audit = tess_blind_transit_search._distinct_frequency_family(
            exact_harmonic, [prior]
        )
        self.assertFalse(distinct)
        self.assertEqual(
            "DUPLICATE_OR_EXACT_HARMONIC_FREQUENCY_FAMILY", audit["reason"]
        )
        distinct, audit = tess_blind_transit_search._distinct_frequency_family(
            near_resonance, [prior]
        )
        self.assertTrue(distinct)
        self.assertEqual("SEPARATE_FREQUENCY_FAMILY", audit["reason"])

    def test_iterative_search_stops_after_masking_a_single_real_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, independent, morphology, broad = self._inputs(
                Path(temporary)
            )
            result = analyze_iterative_blind_transit_search(
                primary_dataset_path=primary,
                independent_spec=independent,
                morphology=morphology,
                broad_interpretation=broad,
                maximum_candidates=2,
            )

        self.assertEqual(1, len(result["candidateSignals"]))
        self.assertEqual(
            "NEXT_RESIDUAL_SIGNAL_UNRESOLVED",
            result["iterativeSearch"]["terminationReason"],
        )
        stopping_iteration = result["iterativeSearch"]["iterations"][-1]
        self.assertFalse(stopping_iteration["accepted"])
        self.assertEqual(
            "BLIND_TRANSIT_PERIOD_UNRESOLVED",
            stopping_iteration["classification"],
        )
        self.assertIn("sectorResults", stopping_iteration["candidateEvidence"])
        self.assertIn(
            "recurrenceSupportGate", stopping_iteration["candidateEvidence"]
        )
        self.assertFalse(
            stopping_iteration["candidateEvidence"]["catalogAnswerKeyUsed"]
        )
        self.assertGreater(
            result["candidateSignals"][0]["residualSearchMask"][
                "totalRemovedSampleCount"
            ],
            0,
        )

    def test_promotes_half_period_alias_when_only_alternating_cycles_transit(self):
        true_period = 3.731
        half_frequency = 2.0 / true_period
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                self._dataset(
                    root, sector=sector, origin=origin, period=true_period
                )
                for sector, origin in ((8, 1000.0), (35, 1700.0), (62, 2400.0))
            ]
            independent = {
                "investigationGoal": "FULL_CHARACTERIZATION",
                "preparedSectors": [
                    {"sector": sector, "datasetPath": str(path)}
                    for sector, path in zip((35, 62), paths[1:])
                ],
            }

            def forced_half_period(sectors, minimum, maximum):
                measurements = [
                    tess_blind_transit_search._box_score(
                        item["times"], item["residual"], item["sigma"],
                        half_frequency,
                    )
                    for item in sectors
                ]
                full_span = max(item["times"][-1] for item in sectors) - min(
                    item["times"][0] for item in sectors
                )
                return (
                    half_frequency, 10.0, measurements, 0.001, 0.000001,
                    float(full_span),
                )

            with mock.patch.object(
                tess_blind_transit_search, "_search_grid",
                side_effect=forced_half_period,
            ):
                result = analyze_blind_transit_search(
                    primary_dataset_path=paths[0],
                    independent_spec=independent,
                    morphology=None,
                    broad_interpretation=None,
                    targeted_interpretation={
                        "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                        "primaryReliable": False,
                    },
                )

        self.assertEqual("REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                         result["classification"])
        self.assertAlmostEqual(true_period, result["candidatePeriodDays"], delta=0.003)
        self.assertAlmostEqual(true_period / 2.0,
                               result["coarseCandidatePeriodDays"], delta=1e-9)
        alias = result["alternatingCycleAliasResolution"]
        self.assertEqual("PROMOTE_DOUBLE_PERIOD", alias["decision"])
        self.assertEqual(
            "TRANSITS_OCCUR_ON_ONLY_ONE_ALTERNATING_CYCLE_PARITY",
            alias["reason"],
        )
        self.assertTrue(alias["doublePeriodValidation"]["supported"])
        self.assertTrue(all(
            item["decisiveAlternatingEvents"] for item in alias["sectorEvidence"]
        ))
        self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_promotes_quarter_period_alias_when_one_cycle_residue_transits(self):
        true_period = 7.17
        quarter_frequency = 4.0 / true_period
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                self._dataset(
                    root, sector=sector, origin=origin, period=true_period
                )
                for sector, origin in ((8, 1000.0), (35, 1700.0), (62, 2400.0))
            ]
            independent = {
                "investigationGoal": "FULL_CHARACTERIZATION",
                "preparedSectors": [
                    {"sector": sector, "datasetPath": str(path)}
                    for sector, path in zip((35, 62), paths[1:])
                ],
            }

            def forced_quarter_period(sectors, minimum, maximum):
                measurements = [
                    tess_blind_transit_search._box_score(
                        item["times"], item["residual"], item["sigma"],
                        quarter_frequency,
                    )
                    for item in sectors
                ]
                full_span = max(item["times"][-1] for item in sectors) - min(
                    item["times"][0] for item in sectors
                )
                return (
                    quarter_frequency, 10.0, measurements, 0.001, 0.000001,
                    float(full_span),
                )

            with mock.patch.object(
                tess_blind_transit_search, "_search_grid",
                side_effect=forced_quarter_period,
            ):
                result = analyze_blind_transit_search(
                    primary_dataset_path=paths[0],
                    independent_spec=independent,
                    morphology=None,
                    broad_interpretation=None,
                    targeted_interpretation={
                        "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                        "primaryReliable": False,
                    },
                )

        self.assertEqual(
            "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
            result["classification"],
        )
        self.assertAlmostEqual(
            true_period, result["candidatePeriodDays"], delta=0.003
        )
        self.assertAlmostEqual(
            true_period / 4.0,
            result["coarseCandidatePeriodDays"],
            delta=1e-9,
        )
        alias = result["alternatingCycleAliasResolution"]
        self.assertEqual("PROMOTE_INTEGER_MULTIPLE_PERIOD", alias["decision"])
        self.assertEqual(4, alias["selectedCycleMultiplier"])
        selected = next(
            item for item in alias["integerMultipleTests"]
            if item["multiplier"] == 4
        )
        self.assertTrue(selected["recurrenceValidation"]["supported"])
        self.assertTrue(all(
            item["decisiveSingleResidueEvents"]
            for item in selected["sectorEvidence"]
        ))
        self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_recovers_from_nonrecurrent_primary_boundary_without_broad_morphology(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, independent, _, _ = self._inputs(Path(temporary))
            targeted = {
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "primaryBoundaryHit": True,
                "supportingSectorCount": 0,
                "contradictionPlan": {
                    "action": "STOP",
                    "reason": "insufficient-independent-evidence-for-broad-contradiction-search",
                },
            }
            result = analyze_blind_transit_search(
                primary_dataset_path=primary,
                independent_spec=independent,
                morphology=None,
                broad_interpretation=None,
                targeted_interpretation=targeted,
            )

        self.assertEqual("FULL_CHARACTERIZATION_NONRECURRENT_BOUNDARY_PERIOD",
                         result["entryBoundary"])
        self.assertEqual("REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                         result["classification"])
        self.assertAlmostEqual(2.21857567, result["candidatePeriodDays"], delta=0.003)
        self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_recovers_from_nonrecurrent_unreliable_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, independent, _, _ = self._inputs(Path(temporary))
            targeted = {
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "primaryBoundaryHit": False,
                "primaryReliable": False,
                "supportingSectorCount": 0,
                "contradictionPlan": {
                    "action": "STOP",
                    "reason": "insufficient-independent-evidence-for-broad-contradiction-search",
                },
            }
            result = analyze_blind_transit_search(
                primary_dataset_path=primary,
                independent_spec=independent,
                morphology=None,
                broad_interpretation=None,
                targeted_interpretation=targeted,
            )

        self.assertEqual(UNRELIABLE_PRIMARY_ENTRY, result["entryBoundary"])
        self.assertEqual("REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                         result["classification"])
        self.assertAlmostEqual(2.21857567, result["candidatePeriodDays"], delta=0.003)
        self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_ignores_nonrecurring_independent_sectors_when_required_support_recurs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specifications = (
                (8, 1000.0, True),
                (35, 1700.0, True),
                (62, 2400.0, True),
                (89, 3100.0, False),
                (99, 3400.0, False),
            )
            paths = [
                self._dataset(root, sector=sector, origin=origin, transit=transit)
                for sector, origin, transit in specifications
            ]
            independent = {
                "investigationGoal": "FULL_CHARACTERIZATION",
                "preparedSectors": [
                    {"sector": sector, "datasetPath": str(path)}
                    for (sector, _, _), path in zip(specifications[1:], paths[1:])
                ],
            }
            targeted = {
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "primaryReliable": False,
            }
            result = analyze_blind_transit_search(
                primary_dataset_path=paths[0],
                independent_spec=independent,
                morphology=None,
                broad_interpretation=None,
                targeted_interpretation=targeted,
            )

        self.assertEqual("REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                         result["classification"])
        self.assertAlmostEqual(2.21857567, result["candidatePeriodDays"], delta=0.003)
        self.assertTrue(result["primarySectorSupported"])
        self.assertEqual([35, 62], result["supportingIndependentSectors"])
        self.assertTrue(result["linearEphemeris"]["coherent"])
        self.assertGreater(result["searchGrid"]["fullObservationSpanDays"], 2000.0)
        self.assertLess(
            result["searchGrid"]["fineFrequencyStepPerDay"],
            result["searchGrid"]["coarseFrequencyStepPerDay"],
        )
        self.assertEqual(
            "SHARED_PERIOD_EPOCH_DURATION_PRIMARY_PLUS_TWO_INDEPENDENT",
            result["searchGrid"]["selectionSupportRule"],
        )
        self.assertFalse(result["catalogAnswerKeyUsed"])

    def test_boundary_dominated_independent_failure_routes_directly_to_blind_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InvestigationStore(root / "investigations")
            investigation = store.create("boundary-transit-route", WORKFLOW_ID,
                                         WORKFLOW_VERSION)
            primary = root / "primary.json"
            primary.write_text("{}", encoding="utf-8")
            prepared_sectors = []
            datasets = []
            for index, sector in enumerate((83, 82, 76, 75)):
                path = root / f"sector-{sector}.json"
                path.write_text("{}", encoding="utf-8")
                prepared_sectors.append({
                    "datasetID": f"sector-{sector}",
                    "sector": sector,
                    "baselineDays": 26.0,
                    "datasetPath": str(path),
                })
                frequency = 0.2 if index == 0 else 0.1
                datasets.append({
                    "datasetID": f"sector-{sector}",
                    "periodStatus": "RELIABLE" if index == 0 else "LOW_CONFIDENCE",
                    "periodConfidence": "high" if index == 0 else "low",
                    "candidatePeriodDays": 1.0 / frequency,
                    "candidateFrequency": frequency,
                    "candidateFrequencyConfidenceInterval": {
                        "lower": frequency - 0.001,
                        "upper": frequency + 0.001,
                    },
                })
            for stage in (
                ("001-prepare-target", "openstar.tess.prepare-target",
                 {"datasetPath": str(primary)}),
                ("004-hypotheses", "openstar.tess.hypotheses", {
                    "observedPeriodDays": 10.0,
                    "primaryBoundaryHit": True,
                }),
                ("005-independent", "openstar.tess.independent.prepare", {
                    "investigationGoal": "FULL_CHARACTERIZATION",
                    "targetPeriodDays": 10.0,
                    "preparedSectors": prepared_sectors,
                    "frequencySearch": {
                        "minimumFrequency": 0.1,
                        "maximumFrequency": 5.0,
                        "frequencyStep": 0.000001,
                    },
                }),
                ("006-independent-run", "openstar.tess.independent.run",
                 {"datasets": datasets}),
            ):
                investigation = self._complete(store, investigation, *stage)
            engine = build_engine(store, types.SimpleNamespace(), poll_interval=0,
                                  timeout=None)
            engine.chain_stages = False
            completed, request = engine.run_stage(
                investigation,
                StageRequest("007-independent-interpret",
                             "openstar.tess.independent.interpret", {},
                             "006-independent-run"),
                software_id="test", software_version="boundary-route",
            )

        result = completed.stages[-1].result
        self.assertEqual("HUMAN_REVIEW_REQUIRED",
                         result["claimDecision"]["claim"])
        self.assertEqual(0, result["supportingSectorCount"])
        self.assertTrue(result["primaryBoundaryHit"])
        self.assertEqual("STOP", result["contradictionPlan"]["action"])
        self.assertEqual(HANDLER_ID, request.handler_id)

    def test_unreliable_primary_routes_to_blind_search_before_broad_variability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InvestigationStore(root / "investigations")
            investigation = store.create("unreliable-transit-route", WORKFLOW_ID,
                                         WORKFLOW_VERSION)
            primary = root / "primary.json"
            primary.write_text("{}", encoding="utf-8")
            prepared_sectors = []
            datasets = []
            for sector, frequency in ((35, 0.5), (62, 1.0 / 3.0)):
                path = root / f"sector-{sector}.json"
                path.write_text("{}", encoding="utf-8")
                prepared_sectors.append({
                    "datasetID": f"sector-{sector}",
                    "sector": sector,
                    "baselineDays": 24.0,
                    "datasetPath": str(path),
                })
                datasets.append({
                    "datasetID": f"sector-{sector}",
                    "periodStatus": "RELIABLE",
                    "periodConfidence": "high",
                    "candidatePeriodDays": 1.0 / frequency,
                    "candidateFrequency": frequency,
                    "candidateFrequencyConfidenceInterval": {
                        "lower": frequency - 0.001,
                        "upper": frequency + 0.001,
                    },
                })
            for stage in (
                ("001-prepare-target", "openstar.tess.prepare-target",
                 {"datasetPath": str(primary)}),
                ("004-hypotheses", "openstar.tess.hypotheses", {
                    "observedPeriodDays": 6.08,
                    "primaryBoundaryHit": False,
                    "primaryReliable": False,
                }),
                ("005-independent", "openstar.tess.independent.prepare", {
                    "investigationGoal": "FULL_CHARACTERIZATION",
                    "targetPeriodDays": 6.08,
                    "preparedSectors": prepared_sectors,
                    "frequencySearch": {
                        "minimumFrequency": 0.1,
                        "maximumFrequency": 5.0,
                        "frequencyStep": 0.000001,
                    },
                }),
                ("006-independent-run", "openstar.tess.independent.run",
                 {"datasets": datasets}),
            ):
                investigation = self._complete(store, investigation, *stage)
            engine = build_engine(store, types.SimpleNamespace(), poll_interval=0,
                                  timeout=None)
            engine.chain_stages = False
            completed, request = engine.run_stage(
                investigation,
                StageRequest("007-independent-interpret",
                             "openstar.tess.independent.interpret", {},
                             "006-independent-run"),
                software_id="test", software_version="unreliable-route",
            )

        result = completed.stages[-1].result
        self.assertEqual("HUMAN_REVIEW_REQUIRED",
                         result["claimDecision"]["claim"])
        self.assertEqual(0, result["supportingSectorCount"])
        self.assertFalse(result["primaryReliable"])
        self.assertEqual("BROAD_INDEPENDENT_SEARCH",
                         result["contradictionPlan"]["action"])
        self.assertEqual(HANDLER_ID, request.handler_id)

    def test_smooth_variability_without_dips_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            primary, independent, morphology, broad = self._inputs(
                Path(temporary), transit=False
            )
            result = analyze_blind_transit_search(
                primary_dataset_path=primary,
                independent_spec=independent,
                morphology=morphology,
                broad_interpretation=broad,
            )

        self.assertEqual("BLIND_TRANSIT_PERIOD_UNRESOLVED", result["classification"])
        self.assertEqual("HUMAN_REVIEW_REQUIRED", result["claimDecision"]["claim"])
        self.assertEqual("HUMAN_SCIENTIFIC_REVIEW", result["recommendedNextTest"])

    def test_joint_gate_accepts_coherent_near_threshold_independent_sectors(self):
        results = [
            {
                "role": "PRIMARY", "sector": 9, "snr": 8.72, "usable": True,
                "cycleCoverage": 20.0, "eventEpoch": 1000.2,
                "durationDays": 0.05,
            },
            {
                "role": "INDEPENDENT", "sector": 10, "snr": 6.60,
                "usable": False, "cycleCoverage": 20.0, "eventEpoch": 1100.2,
                "durationDays": 0.05,
            },
            {
                "role": "INDEPENDENT", "sector": 11, "snr": 6.62,
                "usable": False, "cycleCoverage": 20.0, "eventEpoch": 1200.2,
                "durationDays": 0.05,
            },
        ]

        primary, supporters, ephemeris, supported, gate = (
            tess_blind_transit_search._evaluate_recurrence_support(results, 1.0)
        )

        self.assertTrue(primary)
        self.assertTrue(supported)
        self.assertEqual([11, 10], [item["sector"] for item in supporters])
        self.assertTrue(ephemeris["coherent"])
        self.assertEqual("JOINT_NEAR_THRESHOLD_SECTORS", gate["mode"])
        self.assertGreaterEqual(
            gate["jointRecurrenceSnr"], gate["minimumJointRecurrenceSnr"]
        )

    def test_joint_gate_fails_closed_without_two_independent_six_sigma_events(self):
        results = [
            {
                "role": "PRIMARY", "sector": 9, "snr": 8.72, "usable": True,
                "cycleCoverage": 20.0, "eventEpoch": 1000.2,
                "durationDays": 0.05,
            },
            {
                "role": "INDEPENDENT", "sector": 10, "snr": 6.60,
                "usable": False, "cycleCoverage": 20.0, "eventEpoch": 1100.2,
                "durationDays": 0.05,
            },
            {
                "role": "INDEPENDENT", "sector": 11, "snr": 5.99,
                "usable": False, "cycleCoverage": 20.0, "eventEpoch": 1200.2,
                "durationDays": 0.05,
            },
        ]

        _, supporters, ephemeris, supported, gate = (
            tess_blind_transit_search._evaluate_recurrence_support(results, 1.0)
        )

        self.assertFalse(supported)
        self.assertEqual([], supporters)
        self.assertFalse(ephemeris["coherent"])
        self.assertEqual("NOT_SATISFIED", gate["mode"])

    def test_joint_gate_fails_closed_when_event_times_are_incoherent(self):
        results = [
            {
                "role": "PRIMARY", "sector": 9, "snr": 8.72, "usable": True,
                "cycleCoverage": 20.0, "eventEpoch": 1000.2,
                "durationDays": 0.05,
            },
            {
                "role": "INDEPENDENT", "sector": 10, "snr": 6.60,
                "usable": False, "cycleCoverage": 20.0, "eventEpoch": 1100.2,
                "durationDays": 0.05,
            },
            {
                "role": "INDEPENDENT", "sector": 11, "snr": 6.62,
                "usable": False, "cycleCoverage": 20.0, "eventEpoch": 1200.7,
                "durationDays": 0.05,
            },
        ]

        _, supporters, ephemeris, supported, gate = (
            tess_blind_transit_search._evaluate_recurrence_support(results, 1.0)
        )

        self.assertFalse(supported)
        self.assertEqual([], supporters)
        self.assertFalse(ephemeris["coherent"])
        self.assertEqual("NOT_SATISFIED", gate["mode"])

    def test_pooled_gate_accepts_weak_events_repeated_across_time(self):
        sectors = self._prepared_pooled_sectors()

        results, primary, supporters, ephemeris, supported, gate = (
            tess_blind_transit_search._candidate_evidence(sectors, 0.2)
        )

        self.assertTrue(primary)
        self.assertTrue(supported)
        self.assertTrue(ephemeris["coherent"])
        self.assertEqual("POOLED_SPLIT_RECURRENCE", gate["mode"])
        self.assertGreaterEqual(
            gate["pooledIndependentSnr"], gate["minimumPooledIndependentSnr"]
        )
        self.assertGreaterEqual(
            gate["minimumObservedLeaveOneOutSnr"],
            gate["minimumLeaveOneOutSnr"],
        )
        self.assertGreaterEqual(len(supporters), 4)
        self.assertTrue(all(
            not item["usable"] for item in results
            if item["role"] == "INDEPENDENT"
        ))
        self.assertFalse(gate["catalogAnswerKeyUsed"])

    def test_pooled_gate_rejects_events_missing_from_late_sectors(self):
        sectors = self._prepared_pooled_sectors(late_transits=False)

        _, _, supporters, ephemeris, supported, gate = (
            tess_blind_transit_search._candidate_evidence(sectors, 0.2)
        )

        self.assertFalse(supported)
        self.assertEqual([], supporters)
        self.assertFalse(ephemeris["coherent"])
        pooled = gate["pooledRecurrence"]
        self.assertEqual("NOT_SATISFIED", pooled["mode"])
        self.assertLess(pooled["lateIndependentSnr"], pooled["minimumSplitSnr"])

    def test_pooled_search_grid_selects_shared_weak_clock(self):
        sectors = self._prepared_pooled_sectors(period=5.0)

        frequency, _, _, _, _, _ = tess_blind_transit_search._search_grid(
            sectors, 0.15, 0.3
        )

        self.assertAlmostEqual(0.2, frequency, delta=0.001)

    def test_joint_search_ignores_louder_wrong_phase_sector_events(self):
        sectors, true_period = self._prepared_joint_outlier_sectors()
        true_frequency = 1.0 / true_period
        local = [
            tess_blind_transit_search._box_score(
                item["times"], item["residual"], item["sigma"], true_frequency
            )
            for item in sectors
        ]
        self.assertGreater(
            tess_blind_transit_search._phase_distance(
                local[0]["eventPhase"], local[1]["eventPhase"]
            ),
            0.20,
        )
        self.assertGreater(
            tess_blind_transit_search._phase_distance(
                local[0]["eventPhase"], local[2]["eventPhase"]
            ),
            0.35,
        )

        frequency, _, _, _, fine_step, full_span = (
            tess_blind_transit_search._search_grid(sectors, 0.14, 0.18)
        )
        results, primary, supporters, ephemeris, supported, _ = (
            tess_blind_transit_search._candidate_evidence(sectors, frequency)
        )

        self.assertAlmostEqual(true_period, 1.0 / frequency, delta=0.001)
        self.assertTrue(primary)
        self.assertTrue(supported)
        self.assertTrue(ephemeris["coherent"])
        self.assertGreaterEqual(len(supporters), 2)
        self.assertEqual(1, len({item["eventPhase"] for item in results}))
        self.assertAlmostEqual(
            min(tess_blind_transit_search.DUTY_CYCLES)
            / (full_span * tess_blind_transit_search.OVERSAMPLING),
            fine_step,
        )

    def test_short_long_period_transit_is_not_diluted_by_one_percent_floor(self):
        import numpy as np

        period = 10.0
        frequency = 1.0 / period
        times = np.arange(0.0, 30.0, 0.005)
        residual = np.zeros_like(times)
        distance = np.abs(
            np.remainder(times - 0.37 + period / 2.0, period)
            - period / 2.0
        )
        residual[distance <= 0.025] -= 1.0

        measurement = tess_blind_transit_search._box_score(
            times, residual, 1.0, frequency
        )
        with mock.patch.object(
            tess_blind_transit_search,
            "DUTY_CYCLES",
            (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10),
        ):
            old_floor = tess_blind_transit_search._box_score(
                times, residual, 1.0, frequency
            )

        self.assertLessEqual(measurement["dutyCycle"], 0.005)
        self.assertLessEqual(measurement["durationDays"], 0.05)
        self.assertGreater(measurement["snr"], 1.25 * old_floor["snr"])

    def test_handler_id_is_stable(self):
        self.assertEqual("openstar.tess.blind-transit-search.analyze", HANDLER_ID)

    def test_unresolved_blind_search_freezes_additional_sectors_and_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InvestigationStore(root / "investigations")
            investigation = store.create("adaptive-blind-search", WORKFLOW_ID,
                                         WORKFLOW_VERSION)
            source_project = root / "source-project.json"
            source_project.write_text(json.dumps({
                "id": "source", "name": "source", "workloadID": "ls",
                "datasets": [],
            }), encoding="utf-8")
            primary = root / "primary.json"
            primary.write_text("{}", encoding="utf-8")
            existing = []
            for sector in (2, 3):
                path = root / f"sector-{sector}.json"
                path.write_text("{}", encoding="utf-8")
                existing.append({"sector": sector, "datasetPath": str(path)})
            added_path = root / "sector-4.json"
            added_path.write_text("{}", encoding="utf-8")
            extension_project = root / "extension-project.json"
            extension_project.write_text("{}", encoding="utf-8")

            for stage in (
                ("001-prepare-target", "openstar.tess.prepare-target", {
                    "datasetPath": str(primary),
                    "sourceProjectPath": str(source_project),
                    "sourceDatasetEntry": {"id": "target", "targetName": "Blind"},
                    "ticID": 1,
                    "sector": 1,
                }),
                ("002-independent", "openstar.tess.independent.prepare", {
                    "investigationGoal": "FULL_CHARACTERIZATION",
                    "targetPeriodDays": 5.5,
                    "candidateSectors": [2, 3, 4, 5, 6],
                    "preparedSectors": existing,
                }),
                ("003-interpret", "openstar.tess.independent.interpret", {
                    "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                    "primaryReliable": False,
                }),
            ):
                investigation = self._complete(store, investigation, *stage)

            unresolved = {
                "classification": "BLIND_TRANSIT_PERIOD_UNRESOLVED",
                "coarseCandidatePeriodDays": 4.2,
                "candidatePeriodDays": None,
                "supportingIndependentSectors": [],
                "linearEphemeris": {"coherent": False},
                "recommendedNextTest": "HUMAN_SCIENTIFIC_REVIEW",
            }
            recovered = {
                "classification": "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                "candidatePeriodDays": 9.9,
                "supportingIndependentSectors": [2, 4],
                "linearEphemeris": {"coherent": True},
                "recommendedNextTest": (
                    "ADDITIONAL_INDEPENDENT_SECTOR_TRANSIT_CONFIRMATION"
                ),
            }
            extension = {
                "preparedSectors": [
                    {"sector": 4, "datasetPath": str(added_path)}
                ],
                "projectPath": str(extension_project),
            }
            engine = build_engine(store, types.SimpleNamespace(), poll_interval=0,
                                  timeout=None)
            engine.chain_stages = False
            with mock.patch(
                "workflows.tess.tess_investigation.analyze_blind_transit_search",
                side_effect=[unresolved, recovered],
            ) as analyze_search, mock.patch(
                "workflows.tess.tess_investigation.build_independent_sector_project",
                return_value=extension,
            ) as build_extension, mock.patch(
                "workflows.tess.tess_investigation."
                "analyze_iterative_blind_transit_search",
                side_effect=lambda **kwargs: {
                    **kwargs["initial_result"],
                    "candidateSignals": [{
                        "candidateIndex": 1,
                        "candidatePeriodDays": 9.9,
                        "supportingIndependentSectors": [2, 4],
                        "linearEphemeris": {"coherent": True},
                    }],
                    "iterativeSearch": {
                        "acceptedCandidateCount": 1,
                        "terminationReason": "NEXT_RESIDUAL_SIGNAL_UNRESOLVED",
                        "catalogAnswerKeyUsed": False,
                    },
                },
            ) as iterative_search:
                completed, next_request = engine.run_stage(
                    investigation,
                    StageRequest("004-blind", HANDLER_ID, {}, "003-interpret"),
                    software_id="test", software_version="adaptive-extension",
                )

        result = completed.stages[-1].result
        self.assertEqual("REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                         result["classification"])
        self.assertEqual([4], result["adaptiveSectorExtension"][
            "additionalPreparedSectors"])
        self.assertFalse(result["adaptiveSectorExtension"]["catalogAnswerKeyUsed"])
        self.assertEqual(2, analyze_search.call_count)
        self.assertEqual(1, iterative_search.call_count)
        second_spec = analyze_search.call_args_list[1].kwargs["independent_spec"]
        self.assertEqual([2, 3, 4], [
            item["sector"] for item in second_spec["preparedSectors"]
        ])
        self.assertEqual(8, build_extension.call_args.kwargs["maximum_sectors"])
        self.assertEqual([2, 3], sorted(
            build_extension.call_args.kwargs["excluded_sectors"]
        ))
        self.assertEqual([2, 3, 4], [
            item["sector"] for item in
            iterative_search.call_args.kwargs["independent_spec"]["preparedSectors"]
        ])
        self.assertEqual("openstar.tess.finalize", next_request.handler_id)

    def test_unresolved_full_characterization_morphology_routes_to_blind_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InvestigationStore(root / "investigations")
            investigation = store.create("transit-route", WORKFLOW_ID, WORKFLOW_VERSION)
            paths = []
            for name in ("primary", "sector-2", "sector-3"):
                path = root / f"{name}.json"
                path.write_text("{}", encoding="utf-8")
                paths.append(path)
            prepared = {"datasetPath": str(paths[0])}
            independent = {
                "investigationGoal": "FULL_CHARACTERIZATION",
                "preparedSectors": [
                    {"sector": 2, "datasetPath": str(paths[1])},
                    {"sector": 3, "datasetPath": str(paths[2])},
                ],
            }
            broad = {"harmonicFamily": {
                "representativeRawPeriodDays": 6.0,
                "possibleDoubleCycleDays": 12.0,
            }}
            for stage in (
                ("001-prepare-target", "openstar.tess.prepare-target", prepared),
                ("005-independent", "openstar.tess.independent.prepare", independent),
                ("009-broad", "openstar.tess.independent.broad.interpret", broad),
            ):
                investigation = self._complete(store, investigation, *stage)
            engine = build_engine(store, types.SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            morphology = {
                "physicalCycleResolved": False,
                "continuationEvidence": {"timeFrequencyEvolutionWarranted": False},
                "sectorResults": [],
            }
            with mock.patch(
                "workflows.tess.tess_investigation.analyze_morphology",
                return_value=morphology,
            ):
                _, next_request = engine.run_stage(
                    investigation,
                    StageRequest("010-morphology", "openstar.tess.morphology.analyze", {}, "009-broad"),
                    software_id="test", software_version="1",
                )
        self.assertEqual(HANDLER_ID, next_request.handler_id)

    def test_finalizer_promotes_candidate_period_but_not_companion_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InvestigationStore(root / "investigations")
            investigation = store.create("transit-finalizer", WORKFLOW_ID, WORKFLOW_VERSION)
            transit = {
                "classification": "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE",
                "candidatePeriodDays": 2.21858,
                "primarySectorSupported": True,
                "supportingIndependentSectors": [54, 81],
                "supportingIndependentSectorCount": 2,
                "linearEphemeris": {
                    "coherent": True, "referenceEpoch": 1000.37,
                    "rmsOMinusCDays": 0.001,
                },
                "sectorResults": [],
                "claimDecision": {"claim": "CANDIDATE_PERIOD", "rationale": ["blind box recurrence"]},
                "physicalCycleResolved": False,
                "companionNatureResolved": False,
                "catalogAnswerKeyUsed": False,
                "recommendedNextTest": "ADDITIONAL_INDEPENDENT_SECTOR_TRANSIT_CONFIRMATION",
            }
            stages = (
                ("001-prepare-target", "openstar.tess.prepare-target", {
                    "datasetID": "primary", "ticID": 1, "targetName": "Blind C", "sector": 41,
                }),
                ("002-hypotheses", "openstar.tess.hypotheses", {
                    "rawCandidatePeriodDays": 6.36, "observedPeriodDays": 12.72,
                }),
                ("003-planner", "openstar.tess.planner", {
                    "claimDecision": {"claim": "CANDIDATE_PERIOD", "rationale": []},
                }),
                ("008-broad", "openstar.tess.independent.broad.interpret", {
                    "claimDecision": {"claim": "CANDIDATE_PERIOD", "rationale": []},
                    "selectedPeriodDays": 10.6, "selectedSource": "broad",
                    "harmonicFamily": {"representativeRawPeriodDays": 10.6,
                                       "possibleDoubleCycleDays": 21.2},
                }),
                ("009-morphology", "openstar.tess.morphology.analyze", {
                    "physicalCycleResolved": False,
                }),
                ("010-transit", HANDLER_ID, transit),
            )
            for stage in stages:
                investigation = self._complete(store, investigation, *stage)
            engine = build_engine(store, types.SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            completed, next_request = engine.run_stage(
                investigation,
                StageRequest("011-finalize", "openstar.tess.finalize", {}, "010-transit"),
                software_id="test", software_version="1",
            )
            result = completed.stages[-1].result
            report = Path(result["reportPath"]).read_text(encoding="utf-8")
        self.assertIsNone(next_request)
        self.assertEqual("CANDIDATE_PERIOD", result["claim"]["claim"])
        self.assertAlmostEqual(2.21858, result["periodEvidence"]["recurrentPhotometricPeriodDays"])
        self.assertFalse(result["periodEvidence"]["physicalCycleResolved"])
        self.assertIsNone(result["selectedPeriodDays"])
        self.assertEqual("ADDITIONAL_INDEPENDENT_SECTOR_TRANSIT_CONFIRMATION",
                         result["recommendedNextTest"])
        self.assertEqual(transit, result["blindTransitSearch"])
        self.assertIn("Software-blind transit-period search", report)


if __name__ == "__main__":
    unittest.main()
