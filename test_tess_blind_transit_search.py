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

from workflows.tess.tess_blind_transit_search import (
    HANDLER_ID,
    analyze_blind_transit_search,
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
                 transit: bool = True) -> Path:
        period = 2.21857567
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

    def test_handler_id_is_stable(self):
        self.assertEqual("openstar.tess.blind-transit-search.analyze", HANDLER_ID)

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
