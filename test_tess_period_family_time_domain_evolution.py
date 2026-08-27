import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_period_family_time_domain_evolution import (
    DEFAULT_UNTOUCHED_SECTORS,
    HANDLER_PREFIX,
    PREPARE_HANDLER,
    admit_period_family_time_domain_evolution,
    analyze_sector_flux_pair,
    freeze_time_domain_evolution_boundary,
    interpret_period_family_time_domain_evolution,
    _select_spoc_120s,
    verified_time_domain_evolution_boundary,
)


INDEPENDENT_SECTORS = (98, 96, 95, 93)
FREQUENCIES = (0.2195011427899658, 0.22126030428069043,
               0.22155137519848722, 0.22041593035095275)


class PeriodFamilyTimeDomainEvolutionTests(unittest.TestCase):
    def _boundary(self, root: Path):
        store = InvestigationStore(root / "investigations")
        investigation = store.create(
            "manual-period-family", "openstar.workflow.tess-investigation.v1", "20.2",
            {"ticID": 238919539},
        )
        datasets = [{
            "sector": sector,
            "datasetID": f"sector-{sector}",
            "candidateFrequency": frequency,
            "candidatePeriodDays": 1.0 / frequency,
            "candidatePower": 0.9,
            "candidatePeakProminenceRatio": 5.0,
            "candidateFoldCoherence": 0.95,
            "periodStatus": "RELIABLE",
            "periodConfidence": "high",
        } for sector, frequency in zip(INDEPENDENT_SECTORS, FREQUENCIES)]
        independent = [{
            "sector": sector,
            "datasetID": f"sector-{sector}",
            "candidateFrequency": frequency,
            "candidatePeriodDays": 1.0 / frequency,
            "recurrenceClassification": "RESOLUTION_LIMITED",
            "resolutionLimited": True,
            "supportsTarget": False,
            "eligibleForRecurrence": True,
            "boundaryHit": False,
        } for sector, frequency in zip(INDEPENDENT_SECTORS, FREQUENCIES)]
        prepared = [{"sector": sector} for sector in INDEPENDENT_SECTORS]
        primary = {
            "candidateFrequency": 0.21976441741253439,
            "candidatePeriodDays": 4.550327172040929,
            "candidatePower": 0.9,
            "periodStatus": "RELIABLE",
            "periodConfidence": "high",
        }
        period_detections = [{
            "sector": sector,
            "datasetID": f"sector-{sector}",
            "frequencyCyclesPerDay": frequency,
            "periodDays": 1.0 / frequency,
            "power": 0.9,
            "peakProminenceRatio": 5.0,
            "foldCoherence": 0.95,
            "recurrenceClassification": "RESOLUTION_LIMITED",
            "supportsOriginalCandidate": False,
        } for sector, frequency in zip(INDEPENDENT_SECTORS, FREQUENCIES)]
        localization_preparation = {
            "version": "test-preparation",
            "ticID": 238919539,
            "targetSky": {"raDeg": 94.229274, "decDeg": -52.873880},
            "primaryDetection": {
                "sector": 1,
                "frequencyCyclesPerDay": primary["candidateFrequency"],
                "periodDays": primary["candidatePeriodDays"],
                "power": primary["candidatePower"],
            },
            "sectorDetections": period_detections,
        }
        localized = [{"sector": sector, "classification": "TARGET_CONSISTENT"}
                     for sector in INDEPENDENT_SECTORS]
        from openstar_investigation import sha256_json
        stage_results = {
            "001-prepare-target": {"ticID": 238919539, "sector": 1},
            "002-primary-distributed-search": primary,
            "003-catalog-identity": {
                "ticID": 238919539,
                "tic": {"metadata": {"raDeg": 94.229274, "decDeg": -52.873880}},
                "tess": {"officialSectors": [1, *INDEPENDENT_SECTORS,
                                               *DEFAULT_UNTOUCHED_SECTORS]},
            },
            "004-hypotheses": {"observedPeriodDays": primary["candidatePeriodDays"]},
            "005-planner": {"action": "INDEPENDENT_SECTOR_VERIFICATION"},
            "006-prepare-independent-sectors": {
                "targetPeriodDays": primary["candidatePeriodDays"],
                "preparedSectors": prepared,
            },
            "007-run-independent-sectors": {"datasets": datasets},
            "008-interpret-independent-sectors": {
                "eligibleSectorCount": 4,
                "supportingSectorCount": 0,
                "resolutionLimitedSectorCount": 4,
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "contradictionPlan": {
                    "action": "BROAD_INDEPENDENT_SEARCH",
                    "reason": "targeted-candidate-not-recurrent-independent-sectors-contain-alternate-reliable-structure",
                    "reliableSectorCount": 4,
                },
                "sectorResults": independent,
            },
            "009-prepare-broad-independent-search": {"preparedSectors": prepared},
            "010-run-broad-independent-search": {"datasets": []},
            "011-interpret-broad-independent-search": {
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "sectorResults": [{"sector": sector, "boundaryHit": index > 0}
                                  for index, sector in enumerate(INDEPENDENT_SECTORS)],
                "eligibleSectorCount": 1,
                "bestCluster": {"count": 1},
                "promotionEligible": False,
                "promotionBlockers": ["insufficient-independent-sector-support"],
                "selectedPeriodDays": None,
            },
            "012-finalize": {
                "claim": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "automaticDiscoveryClaim": False,
                "selectedPeriodDays": None,
                "recommendedNextTest": None,
            },
            "013-prepare-period-family-difference-imaging": localization_preparation,
            "014-run-period-family-difference-imaging": {
                "periodDetectionRecomputed": False,
                "sectorResults": localized,
                "errors": [],
            },
            "015-interpret-period-family-difference-imaging": {
                "classification": "TARGET_PERIOD_FAMILY_SUPPORTED",
                "claimDecision": {"claim": "HUMAN_REVIEW_REQUIRED"},
                "sourceAttributionResolved": True,
                "variableSignalOrigin": "TARGET",
                "qualitySectorCount": 4,
                "requiredSupportCount": 3,
                "targetSupportingSectors": sorted(INDEPENDENT_SECTORS),
                "offTargetSectors": [],
                "ambiguousSectors": [],
                "noQualitySectors": [],
                "sectorResults": localized,
                "errors": [],
                "periodDetectionRecomputed": False,
                "periodFamilyResolved": False,
                "physicalCycleResolved": False,
                "physicalMechanismResolved": False,
                "recommendedNextTest": "UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION",
                "preparationSHA256": sha256_json(localization_preparation),
            },
        }
        handlers = [
            ("001-prepare-target", "openstar.tess.prepare-target"),
            ("002-primary-distributed-search", "openstar.tess.primary-project.run"),
            ("003-catalog-identity", "openstar.tess.catalog-identity"),
            ("004-hypotheses", "openstar.tess.hypotheses"),
            ("005-planner", "openstar.tess.planner"),
            ("006-prepare-independent-sectors", "openstar.tess.independent.prepare"),
            ("007-run-independent-sectors", "openstar.tess.independent.run"),
            ("008-interpret-independent-sectors", "openstar.tess.independent.interpret"),
            ("009-prepare-broad-independent-search", "openstar.tess.independent.broad.prepare"),
            ("010-run-broad-independent-search", "openstar.tess.independent.broad.run"),
            ("011-interpret-broad-independent-search", "openstar.tess.independent.broad.interpret"),
            ("012-finalize", "openstar.tess.finalize"),
            ("013-prepare-period-family-difference-imaging",
             "openstar.tess.period-family-difference-imaging.prepare"),
            ("014-run-period-family-difference-imaging",
             "openstar.tess.period-family-difference-imaging.run"),
            ("015-interpret-period-family-difference-imaging",
             "openstar.tess.period-family-difference-imaging.interpret"),
        ]
        previous = None
        for stage_id, handler in handlers:
            running = InvestigationStage(stage_id, handler, "RUNNING", previous, {})
            investigation = store.append_running_stage(investigation, running)
            terminal = store.build_terminal_stage(
                stage_id=stage_id,
                handler_id=handler,
                status="COMPLETE",
                triggered_by_stage_id=previous,
                parameters={},
                result=stage_results[stage_id],
                error=None,
                software_id="test",
                software_version="1",
                started_at=running.started_at,
                stop=stage_id in {"012-finalize", "015-interpret-period-family-difference-imaging"},
            )
            investigation = store.complete_current_stage(investigation, terminal)
            previous = stage_id
        investigation = store.set_control_state(
            investigation,
            status="QUIESCENT_AWAITING_DATA",
            control_state={
                "branchAssessments": [],
                "selectedExperiment": {
                    "id": "013-prepare-period-family-difference-imaging",
                    "handler_id": "openstar.tess.period-family-difference-imaging.prepare",
                    "parameters": {},
                    "triggered_by_stage_id": "012-finalize",
                },
                "schedulerAction": "RUN_EXPERIMENT",
                "recovery": "TESS_MANUAL_PERIOD_FAMILY_DIFFERENCE_IMAGING_V1",
            },
        )
        return store, investigation

    @staticmethod
    def _preparation(root: Path):
        periods = [4.550327172040929, *(1.0 / value for value in FREQUENCIES)]
        return {
            "artifactRoot": str(root),
            "ticID": 238919539,
            "untouchedSectors": list(DEFAULT_UNTOUCHED_SECTORS),
            "familyCenterDays": float(np.median(periods)),
            "familyAcceptanceWindowDays": [4.40, 4.67],
        }

    @staticmethod
    def _signal_input(sector: int, *, evolving: bool = False, pdcsap_signal: bool = True):
        period = float(np.median([4.550327172040929, *(1.0 / value for value in FREQUENCIES)]))
        time = np.linspace(0.0, 27.0, 2400)
        amplitude = (0.2 + 1.8 * time / time[-1]) if evolving else np.ones(len(time))
        sap = amplitude * np.sin(2.0 * math.pi * time / period)
        pdcsap = (amplitude * np.sin(2.0 * math.pi * time / period)
                  if pdcsap_signal else np.random.default_rng(sector).normal(0, 1, len(time)))
        return {
            "sector": sector,
            "time": time,
            "sap": sap,
            "pdcsap": pdcsap,
            "originalSamples": len(time),
            "commonFiniteSamples": len(time),
            "analysisSamples": len(time),
            "baselineDays": 27.0,
            "provenance": {"author": "SPOC", "cadenceSeconds": 120.0},
        }

    def test_stage_015_boundary_freezes_only_untouched_official_sectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._boundary(Path(temporary))
            frozen, hashes = verified_time_domain_evolution_boundary(store, investigation)
            self.assertEqual(15, len(hashes))
            self.assertEqual(list(DEFAULT_UNTOUCHED_SECTORS), frozen["untouchedSectors"])
            self.assertFalse(set(frozen["untouchedSectors"]) &
                             set(frozen["previouslyConsumedSectors"]))
            self.assertEqual("TARGET", frozen["sourceAttribution"])
            self.assertFalse(frozen["periodSearchPerformed"])

    def test_admission_changes_control_state_without_changing_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._boundary(Path(temporary))
            history = copy.deepcopy(investigation.stages)
            ledger_bytes = {stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                            for stage in history}
            admitted = admit_period_family_time_domain_evolution(store, investigation)
            self.assertEqual(history, admitted.stages)
            self.assertEqual("RUNNING", admitted.status)
            selected = admitted.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("016-prepare-period-family-time-domain-evolution", selected["id"])
            self.assertEqual(PREPARE_HANDLER, selected["handler_id"])
            self.assertEqual(ledger_bytes, {
                stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                for stage in history
            })

    def test_ledger_or_localization_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._boundary(Path(temporary))
            path = store.stage_path_for(investigation.id,
                                        "015-interpret-period-family-difference-imaging")
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ledger verification failed"):
                verified_time_domain_evolution_boundary(store, investigation)

    def test_real_acf_and_waveform_observables_support_stable_recurrence(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation = self._preparation(Path(temporary))
            results = [analyze_sector_flux_pair(self._signal_input(sector), preparation)
                       for sector in (5, 61, 87)]
            interpretation = interpret_period_family_time_domain_evolution(
                preparation, {"sectorResults": results, "errors": []}
            )
            self.assertEqual("PERSISTENT_STABLE_TIME_DOMAIN_RECURRENCE",
                             interpretation["classification"])
            self.assertEqual([5, 61, 87], interpretation["supportingSectors"])
            self.assertFalse(interpretation["periodSearchPerformed"])
            self.assertFalse(interpretation["physicalCycleResolved"])
            self.assertFalse(interpretation["physicalMechanismResolved"])

    def test_amplitude_evolution_is_detection_not_physical_interpretation(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation = self._preparation(Path(temporary))
            results = [analyze_sector_flux_pair(
                self._signal_input(sector, evolving=True), preparation
            ) for sector in (5, 61, 87)]
            interpretation = interpret_period_family_time_domain_evolution(
                preparation, {"sectorResults": results, "errors": []}
            )
            self.assertEqual("PERSISTENT_EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE",
                             interpretation["classification"])
            self.assertTrue(interpretation["waveformEvolutionOrComplexitySupported"])
            self.assertEqual("HUMAN_REVIEW_REQUIRED",
                             interpretation["claimDecision"]["claim"])
            self.assertFalse(interpretation["physicalMechanismResolved"])

    def test_sap_pdcsap_disagreement_fails_closed(self):
        sectors = [{"sector": sector,
                    "campaign": campaign,
                    "classification": "SAP_PDCSAP_DISAGREEMENT"}
                   for sector, campaign in ((5, "EARLY_TESS"),
                                            (61, "MIDDLE_EPOCH"),
                                            (87, "RECENT_EPOCH"))]
        result = interpret_period_family_time_domain_evolution(
            {"version": "test"}, {"sectorResults": sectors, "errors": []}
        )
        self.assertEqual("PIPELINE_DEPENDENT_TIME_DOMAIN_RESULT", result["classification"])
        self.assertFalse(result["timeDomainFamilyReplicated"])
        self.assertEqual("HUMAN_REVIEW_REQUIRED", result["claimDecision"]["claim"])

    def test_cross_epoch_lag_spread_is_complex_not_a_stable_clock(self):
        sectors = [{
            "sector": sector,
            "campaign": campaign,
            "classification": "STABLE_TIME_DOMAIN_RECURRENCE",
            "consensusFamilyLagDays": lag,
        } for sector, campaign, lag in (
            (5, "EARLY_TESS", 4.45),
            (61, "MIDDLE_EPOCH", 4.55),
            (87, "RECENT_EPOCH", 4.65),
        )]
        result = interpret_period_family_time_domain_evolution(
            {"version": "test"}, {"sectorResults": sectors, "errors": []}
        )
        self.assertEqual("PERSISTENT_EVOLVING_OR_COMPLEX_TIME_DOMAIN_RECURRENCE",
                         result["classification"])
        self.assertFalse(result["stableClockSupported"])
        self.assertFalse(result["periodFamilyResolved"])

    def test_archive_selection_requires_spoc_120s_without_fallback(self):
        class Table:
            colnames = ["sequence_number", "author", "exptime"]

            def __init__(self, authors, cadences):
                self.values = {
                    "sequence_number": [5] * len(authors),
                    "author": authors,
                    "exptime": cadences,
                }

            def __len__(self):
                return len(self.values["author"])

            def __getitem__(self, name):
                return self.values[name]

        class Search:
            def __init__(self, authors, cadences):
                self.table = Table(authors, cadences)

            def __getitem__(self, value):
                return (value.start, value.stop)

        selected, author, cadence = _select_spoc_120s(
            Search(["TESS-SPOC", "SPOC", "SPOC"], [120.0, 600.0, 120.0]), 5
        )
        self.assertEqual((2, 3), selected)
        self.assertEqual("SPOC", author)
        self.assertEqual(120.0, cadence)
        with self.assertRaisesRegex(RuntimeError, "No official SPOC 120-second"):
            _select_spoc_120s(Search(["TESS-SPOC", "SPOC"], [120.0, 600.0]), 5)

    def test_full_mocked_lifecycle_appends_016_to_018_without_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, investigation = self._boundary(root)
            history = copy.deepcopy(investigation.stages)
            ledger_bytes = {stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                            for stage in history}
            admitted = admit_period_family_time_domain_evolution(store, investigation)
            request = StageRequest(**admitted.metadata["controlState"]["selectedExperiment"])
            engine = build_engine(store, object(), poll_interval=0, timeout=None)
            inputs = [self._signal_input(sector) for sector in (5, 61, 87)]
            with mock.patch(
                "workflows.tess.tess_period_family_time_domain_evolution._production_sector_inputs",
                return_value=(inputs, []),
            ):
                completed = engine.run(
                    admitted, request, software_id="test", software_version="1", max_stages=3
                )
            self.assertEqual(
                ["016-prepare-period-family-time-domain-evolution",
                 "017-run-period-family-time-domain-evolution",
                 "018-interpret-period-family-time-domain-evolution"],
                [stage.id for stage in completed.stages[len(history):]],
            )
            self.assertEqual(history, completed.stages[:len(history)])
            self.assertEqual("QUIESCENT_AWAITING_DATA", completed.status)
            self.assertEqual("PERSISTENT_STABLE_TIME_DOMAIN_RECURRENCE",
                             completed.stages[-1].result["classification"])
            self.assertEqual(ledger_bytes, {
                stage.id: store.stage_path_for(investigation.id, stage.id).read_bytes()
                for stage in history
            })
            self.assertTrue(all(
                Path(reference.path).resolve().is_relative_to(
                    store.directory_for(investigation.id).resolve())
                for stage in completed.stages[len(history):]
                for reference in stage.artifacts
            ))

    def test_nonofficial_or_consumed_sector_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._boundary(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "previously consumed"):
                freeze_time_domain_evolution_boundary(
                    investigation, sectors=(1, 5, 61, 87)
                )


if __name__ == "__main__":
    unittest.main()
