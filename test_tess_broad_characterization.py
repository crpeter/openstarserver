import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from openstar_investigation import InvestigationStage, InvestigationStore, sha256_json
from openstar_targets import InvestigationTarget
from openstar_workflow import StageRequest
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION, plan_tess_branches

from workflows.tess.tess_hypotheses import (
    broad_independent_next_handler,
    interpret_broad_independent_sectors,
)
from workflows.tess.tess_morphology import analyze_morphology
from workflows.tess.tess_morphology import _stationarity_evidence

# The integration exercises handlers whose selected path is pure Python.  The
# full adapter also registers optional NumPy-backed archival handlers; keep
# those unavailable branches importable in the dependency-minimal server test
# environment without pretending to execute them.
try:
    import numpy as _real_numpy
except ModuleNotFoundError:
    _real_numpy = None
if _real_numpy is None:
    sys.modules["numpy"] = types.ModuleType("numpy")

from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_investigation import time_frequency_continuation
from workflows.tess.tess_investigation import nonstationary_continuation
from workflows.tess.tess_investigation import residual_mode_localization_continuation
from workflows.tess.tess_time_frequency import _fit_family_python


class BroadIndependentCharacterizationTests(unittest.TestCase):
    def _complete(self, store, investigation, stage_id, handler_id, result):
        running = InvestigationStage(stage_id, handler_id, "RUNNING", None, {})
        investigation = store.append_running_stage(investigation, running)
        terminal = store.build_terminal_stage(
            stage_id=stage_id,
            handler_id=handler_id,
            status="COMPLETE",
            triggered_by_stage_id=None,
            parameters={},
            result=result,
            error=None,
            software_id="integration-replay",
            software_version="20.28",
            started_at=running.started_at,
        )
        return store.complete_current_stage(investigation, terminal)

    def _write_light_curve(
        self, root, name, sector, phase_offset, shape="mixed", *,
        noise=0.0, gap_modulus=None, amplitude_scale=1.0,
        double_phase=0.0, raw_amplitude=None, double_amplitude=None,
    ):
        path = root / f"{name}.json"
        times = [index * 0.01 for index in range(3001)
                 if gap_modulus is None or index % gap_modulus not in (0, 1)]
        double_amplitude = (
            {"raw": 0.0, "double": 0.03, "mixed": 0.008}[shape]
            if double_amplitude is None else double_amplitude
        ) * amplitude_scale
        raw_amplitude = (
            {"raw": 0.025, "double": 0.008, "mixed": 0.02}[shape]
            if raw_amplitude is None else raw_amplitude
        ) * amplitude_scale
        flux = [1.0
            + raw_amplitude * math.sin(2.0 * math.pi * time / 6.86 + phase_offset)
            + double_amplitude * math.sin(2.0 * math.pi * time / 13.72 + double_phase)
            + noise * math.sin(2.0 * math.pi * time / 0.371 + sector * 0.17)
            for time in times]
        path.write_text(
            json.dumps(
                {
                    "id": name,
                    "targetName": "Synthetic recurrent source",
                    "source": {"sector": sector, "baselineDays": 30.0},
                    "times": times,
                    "flux": flux,
                }
            ),
            encoding="utf-8",
        )
        return path

    def _interpret(self, periods, prominences, reliable=None):
        reliable = reliable or [True] * len(periods)
        prepared = []
        datasets = []
        for index, (period, prominence, is_reliable) in enumerate(
            zip(periods, prominences, reliable), start=1
        ):
            dataset_id = f"sector-{index}"
            prepared.append(
                {"datasetID": dataset_id, "sector": index, "baselineDays": 27.0}
            )
            datasets.append(
                {
                    "datasetID": dataset_id,
                    "periodStatus": "RELIABLE" if is_reliable else "UNRELIABLE",
                    "periodConfidence": "high" if is_reliable else "low",
                    "candidatePeriodDays": period,
                    "candidateFrequency": 1.0 / period,
                    "candidatePeakProminenceRatio": prominence,
                }
            )
        return interpret_broad_independent_sectors(
            project_status={"datasets": datasets},
            broad_spec={
                "frequencySearch": {
                    "minimumFrequency": 0.01,
                    "maximumFrequency": 0.5,
                    "frequencyStep": 0.0001,
                },
                "preparedSectors": prepared,
            },
            primary_raw_period_days=7.0,
            primary_preferred_period_days=14.0,
            same_sector_candidate_days=13.8,
        )

    def test_stable_promoted_recurrence_finalizes_normally(self):
        evidence = self._interpret([6.98, 7.0, 7.02], [2.0, 2.1, 1.9])

        self.assertTrue(evidence["promotionEligible"])
        self.assertFalse(evidence["variabilityCharacterization"]["warranted"])
        self.assertEqual(
            "openstar.tess.finalize", broad_independent_next_handler(evidence)
        )

    def test_no_meaningful_recurrent_family_finalizes_conservatively(self):
        evidence = self._interpret(
            [4.0, 7.0, 11.0], [2.0, 2.0, 2.0], [True, False, False]
        )

        self.assertEqual("HUMAN_REVIEW_REQUIRED", evidence["claimDecision"]["claim"])
        self.assertIsNone(evidence["harmonicFamily"])
        self.assertEqual(
            "openstar.tess.finalize", broad_independent_next_handler(evidence)
        )

    def test_recurrent_inconsistent_family_routes_to_morphology(self):
        evidence = self._interpret([6.52, 7.20], [1.2, 1.3])

        characterization = evidence["variabilityCharacterization"]
        self.assertEqual(
            "RECURRENT_BUT_UNRESOLVED_CROSS_SECTOR_VARIABILITY",
            characterization["state"],
        )
        self.assertTrue(characterization["warranted"])
        self.assertIn("cluster-spread-too-wide", characterization["reasons"])
        self.assertEqual(
            "openstar.tess.morphology.analyze",
            broad_independent_next_handler(evidence),
        )

    def test_restart_selection_is_deterministic_from_persisted_evidence(self):
        evidence = self._interpret([6.52, 7.20], [1.2, 1.3])

        first = broad_independent_next_handler(evidence)
        second = broad_independent_next_handler(evidence)
        self.assertEqual(first, second)

    def test_broad_continuation_executes_existing_morphology_handler(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = InvestigationStore(root / "investigations")
        target = InvestigationTarget(
            id="synthetic:unresolved-family",
            investigation_id="synthetic-unresolved-family",
            workflow_id=WORKFLOW_ID,
            workflow_version=WORKFLOW_VERSION,
            priority=0,
            eligible=True,
            metadata={},
        )
        investigation = store.create(
            target.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata=target.metadata,
        )
        primary_path = self._write_light_curve(root, "primary", 62, 0.0, "mixed")
        sector_66_path = self._write_light_curve(root, "sector-66", 66, 0.2, "raw")
        sector_67_path = self._write_light_curve(root, "sector-67", 67, 0.5, "double")
        source_project_path = root / "source-project.json"
        source_project_path.write_text(json.dumps({
            "id": "synthetic-source", "name": "Synthetic source",
            "workloadID": "openstar.lomb-scargle.v1", "datasets": []
        }), encoding="utf-8")
        independent = {
            "preparedSectors": [
                {
                    "datasetID": "sector-66",
                    "sector": 66,
                    "baselineDays": 30.0,
                    "datasetPath": str(sector_66_path),
                },
                {
                    "datasetID": "sector-67",
                    "sector": 67,
                    "baselineDays": 30.0,
                    "datasetPath": str(sector_67_path),
                },
            ]
        }
        broad_spec = {
            **independent,
            "frequencySearch": {
                "minimumFrequency": 0.01,
                "maximumFrequency": 0.5,
                "frequencyStep": 0.0001,
            },
        }
        broad_run = {
            "datasets": [
                {
                    "datasetID": "sector-66",
                    "periodStatus": "RELIABLE",
                    "periodConfidence": "high",
                    "candidatePeriodDays": 6.52,
                    "candidateFrequency": 1.0 / 6.52,
                    "candidatePeakProminenceRatio": 1.2,
                },
                {
                    "datasetID": "sector-67",
                    "periodStatus": "RELIABLE",
                    "periodConfidence": "high",
                    "candidatePeriodDays": 7.20,
                    "candidateFrequency": 1.0 / 7.20,
                    "candidatePeakProminenceRatio": 1.3,
                },
            ]
        }
        replay = (
            (
                "001-prepare-target",
                "openstar.tess.prepare-target",
                {
                    "datasetPath": str(primary_path),
                    "sector": 62,
                    "sourceProjectPath": str(source_project_path),
                    "sourceDatasetEntry": {"id": "primary", "targetName": "Synthetic recurrent source"},
                },
            ),
            (
                "002-primary-distributed-search",
                "openstar.tess.primary-project.run",
                {"datasets": []},
            ),
            (
                "004-hypotheses",
                "openstar.tess.hypotheses",
                {"rawCandidatePeriodDays": 7.0, "observedPeriodDays": 14.0},
            ),
            (
                "006-prepare-independent",
                "openstar.tess.independent.prepare",
                independent,
            ),
            (
                "007-run-independent",
                "openstar.tess.independent.run",
                {"status": "COMPLETE"},
            ),
            (
                "008-prepare-broad",
                "openstar.tess.independent.broad.prepare",
                broad_spec,
            ),
            ("009-run-broad", "openstar.tess.independent.broad.run", broad_run),
        )
        for stage_id, handler_id, result in replay:
            investigation = self._complete(
                store, investigation, stage_id, handler_id, result
            )

        class DeterministicCoordinator:
            def run_project(self, project_path, **_kwargs):
                manifest = json.loads(Path(project_path).read_text(encoding="utf-8"))
                datasets = []
                for index, entry in enumerate(manifest["datasets"]):
                    frequency = 0.31 + 0.001 * (index % 3)
                    datasets.append({
                        "datasetID": entry["id"],
                        "periodStatus": "RELIABLE",
                        "periodConfidence": "high",
                        "candidateFrequency": frequency,
                        "candidatePeriodDays": 1.0 / frequency,
                        "candidatePower": 0.8,
                        "candidatePeakProminenceRatio": 2.5,
                    })
                return types.SimpleNamespace(
                    status={"datasets": datasets}, node_contributions={},
                    project_id=manifest["id"],
                )

        engine = build_engine(
            store, coordinator=DeterministicCoordinator(), poll_interval=0.0, timeout=None
        )
        engine.chain_stages = False
        broad_request = StageRequest(
            "010-interpret-broad",
            "openstar.tess.independent.broad.interpret",
            {},
            "009-run-broad",
        )
        investigation, morphology_request = engine.run_stage(
            investigation,
            broad_request,
            software_id="integration",
            software_version="20.28",
        )

        self.assertEqual(
            "openstar.tess.morphology.analyze", morphology_request.handler_id
        )
        persisted_broad = investigation.stages[-1]
        self.assertEqual(
            "RECURRENT_BUT_UNRESOLVED_CROSS_SECTOR_VARIABILITY",
            persisted_broad.result["variabilityCharacterization"]["state"],
        )

        # A process restart reconstructs exactly the persisted continuation and
        # does not replay any of the seven completed prerequisite stages.
        restarted = store.load(investigation.id)
        restarted_request = plan_tess_branches(restarted, target)[0].experiment
        self.assertEqual(morphology_request, restarted_request)
        self.assertEqual(8, len(restarted.stages))

        completed, next_request = engine.run_stage(
            restarted,
            restarted_request,
            software_id="integration",
            software_version="20.28",
        )

        morphology_stage = completed.stages[-1]
        self.assertEqual("COMPLETE", morphology_stage.status)
        self.assertEqual(
            "openstar.tess.morphology.analyze", morphology_stage.handler_id
        )
        self.assertEqual(3, len(morphology_stage.result["sectorResults"]))
        self.assertEqual(
            {
                "periodFamily",
                "primaryDataset",
                "independentSector66",
                "independentSector67",
            },
            set(morphology_stage.provenance.input_hashes),
        )
        self.assertEqual(1, len(morphology_stage.artifacts))
        self.assertEqual("openstar.tess.time-frequency.prepare", next_request.handler_id)
        self.assertEqual(9, len(completed.stages))

        # Restarting after morphology reconstructs its persisted continuation,
        # without duplicating morphology or any prerequisite stage.
        restarted_after_morphology = store.load(completed.id)
        replayed_next = plan_tess_branches(restarted_after_morphology, target)[0].experiment
        self.assertEqual(next_request, replayed_next)
        self.assertEqual(9, len(restarted_after_morphology.stages))

        prepared, run_request = engine.run_stage(
            restarted_after_morphology, replayed_next,
            software_id="integration", software_version="20.28",
        )
        time_frequency_stage = prepared.stages[-1]
        self.assertEqual("COMPLETE", time_frequency_stage.status)
        self.assertEqual("openstar.tess.time-frequency.prepare", time_frequency_stage.handler_id)
        self.assertGreater(len(time_frequency_stage.result["preparedWindows"]), 0)
        self.assertAlmostEqual(13.72, time_frequency_stage.result["physicalPeriodDays"])
        self.assertTrue(Path(time_frequency_stage.result["projectPath"]).is_file())
        self.assertEqual("openstar.tess.time-frequency.run", run_request.handler_id)
        self.assertEqual(10, len(prepared.stages))

        # Every persisted boundary reconstructs the exact continuation. Execute
        # the real registered run, interpret, and summarize handlers.
        current = prepared
        request = run_request
        expected_handlers = [
            "openstar.tess.time-frequency.run",
            "openstar.tess.time-frequency.interpret",
            "openstar.tess.time-frequency.summarize",
        ]
        for expected_handler in expected_handlers:
            restarted_boundary = store.load(current.id)
            planned = plan_tess_branches(restarted_boundary, target)[0].experiment
            self.assertEqual(request, planned)
            stage_count = len(restarted_boundary.stages)
            current, request = engine.run_stage(
                restarted_boundary, planned,
                software_id="integration", software_version="20.28",
            )
            self.assertEqual(stage_count + 1, len(current.stages))
            self.assertEqual(expected_handler, current.stages[-1].handler_id)
            self.assertEqual("COMPLETE", current.stages[-1].status)

        summary = current.stages[-1].result
        self.assertFalse(morphology_stage.result["physicalCycleResolved"])
        self.assertAlmostEqual(13.72, summary["periodReference"]["periodDays"])
        self.assertEqual("UNRESOLVED_FAMILY_ANALYSIS_REFERENCE", summary["periodReference"]["kind"])
        self.assertFalse(summary["periodReference"]["physicalCycleResolved"])
        self.assertFalse(summary["physicalMechanismResolved"])
        self.assertEqual("TRANSIENT_MODE_VALIDATION", summary["recommendedNextTest"])
        self.assertEqual("openstar.tess.finalize", request.handler_id)

    def test_coupled_python_harmonic_fit_on_irregular_gapped_samples(self):
        frequency = 1.0 / 13.72
        coefficients = [1.13, 0.027, -0.014, 0.019, 0.011]
        times = [
            index * 0.071 + 0.013 * (index % 5)
            for index in range(420)
            if index % 7 not in (0, 3) and not 9.0 < index * 0.071 < 13.0
        ]
        flux = []
        for time in times:
            angle = 2.0 * math.pi * frequency * time
            basis = [1.0, math.sin(angle), math.cos(angle), math.sin(2 * angle), math.cos(2 * angle)]
            flux.append(sum(value * component for value, component in zip(coefficients, basis)))

        residual, fit = _fit_family_python(times, flux, frequency)
        for expected, actual in zip(coefficients, fit["coefficients"]):
            self.assertAlmostEqual(expected, actual, places=11)
        self.assertAlmostEqual(math.hypot(coefficients[1], coefficients[2]), fit["fundamentalAmplitude"], places=11)
        self.assertAlmostEqual(math.hypot(coefficients[3], coefficients[4]), fit["firstHarmonicAmplitude"], places=11)
        self.assertLess(fit["residualStdDevBeforeNormalization"], 1e-11)
        self.assertLess(max(abs(value) for value in residual), 1e-10)

        # Deliberately prove that this sampling is not an orthogonal special
        # case, so independent projections would not be a valid substitute.
        sin_f = [math.sin(2.0 * math.pi * frequency * time) for time in times]
        cos_2f = [math.cos(4.0 * math.pi * frequency * time) for time in times]
        cross_term = sum(left * right for left, right in zip(sin_f, cos_2f))
        self.assertGreater(abs(cross_term), 1.0)

    def test_stable_morphology_still_finalizes(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        primary = self._write_light_curve(root, "primary-stable", 1, 0.0, "raw")
        paths = [self._write_light_curve(root, f"stable-{sector}", sector, 0.1, "raw")
                 for sector in (2, 3, 4)]
        result = analyze_morphology(
            primary_dataset_path=primary,
            independent_spec={"preparedSectors": [
                {"sector": sector, "datasetPath": str(path)}
                for sector, path in zip((2, 3, 4), paths)
            ]},
            raw_period_days=6.86,
            possible_double_cycle_days=13.72,
        )
        self.assertTrue(result["physicalCycleResolved"])
        self.assertFalse(result["continuationEvidence"]["timeFrequencyEvolutionWarranted"])
        self.assertEqual(
            "NO_TIME_FREQUENCY_EVOLUTION_FOLLOWUP_WARRANTED",
            result["continuationEvidence"]["stationarityEvidence"]["classification"],
        )

    def test_stable_double_wave_morphology_finalizes(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        primary = self._write_light_curve(root, "primary-double", 1, 0.0, "double")
        paths = [self._write_light_curve(root, f"double-{sector}", sector, 0.0, "double")
                 for sector in (2, 3, 4)]
        result = analyze_morphology(
            primary_dataset_path=primary,
            independent_spec={"preparedSectors": [
                {"sector": sector, "datasetPath": str(path)}
                for sector, path in zip((2, 3, 4), paths)
            ]}, raw_period_days=6.86, possible_double_cycle_days=13.72,
        )
        self.assertEqual("DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED", result["morphologyClass"])
        self.assertFalse(result["continuationEvidence"]["timeFrequencyEvolutionWarranted"])

    def test_resolved_cycle_with_generic_morphology_evolution_continues(self):
        # Blind-C-like values are used only as a regression input.  The rule is
        # based on robust cross-sector IQRs and requires multiple diagnostics.
        gains = [0.1547, 0.0951, 0.0920, 0.2383, 0.4700]
        halves = [0.3876, 0.2966, 0.3599, 0.6052, 0.5695]
        evidence = _stationarity_evidence([
            {
                "doubleExplainedVarianceImprovement": gain,
                "doubleWaveMetrics": {"halfCycleDifferenceRatio": half},
                "doubleProfile": {
                    "profileAmplitude": 0.03, "minimumPhase": 0.2,
                    "minimumDutyCycle": 0.25, "profileRoughness": 0.05,
                },
            }
            for gain, half in zip(gains, halves)
        ])
        self.assertTrue(evidence["followupWarranted"])
        self.assertEqual(
            ["doubleExplainedVarianceGainIqr", "halfCycleDifferenceRatioIqr"],
            evidence["triggeredMetrics"],
        )
        self.assertEqual(["SHAPE_RAW_DOUBLE"], evidence["triggeredEvidenceFamilies"])
        self.assertIn("does not establish nonstationarity", evidence["interpretation"])

    def test_stable_noisy_gapped_controls_do_not_trigger_followup(self):
        for shape in ("raw", "double"):
            with self.subTest(shape=shape):
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                root = Path(temporary.name)
                primary = self._write_light_curve(
                    root, f"{shape}-primary", 10, 0.0, shape,
                    noise=0.003, gap_modulus=17,
                )
                paths = [self._write_light_curve(
                    root, f"{shape}-{sector}", sector, 0.0, shape,
                    noise=0.003, gap_modulus=modulus,
                ) for sector, modulus in zip((11, 12, 13, 14), (13, 19, 23, 29))]
                result = analyze_morphology(
                    primary_dataset_path=primary,
                    independent_spec={"preparedSectors": [
                        {"sector": sector, "datasetPath": str(path)}
                        for sector, path in zip((11, 12, 13, 14), paths)
                    ]}, raw_period_days=6.86, possible_double_cycle_days=13.72,
                )
                stationarity = result["continuationEvidence"]["stationarityEvidence"]
                self.assertFalse(stationarity["followupWarranted"], stationarity)
                self.assertEqual([], stationarity["triggeredMetrics"])

    def test_deliberate_amplitude_phase_and_shape_changes_trigger_expected_families(self):
        cases = {
            "AMPLITUDE": [
                {"doubleProfile": {"profileAmplitude": value, "minimumPhase": .2,
                 "minimumDutyCycle": .2, "profileRoughness": .05},
                 "doubleExplainedVarianceImprovement": .2,
                 "doubleWaveMetrics": {"halfCycleDifferenceRatio": .4}}
                for value in (.01, .02, .04, .08, .12)
            ],
            "PHASE": [
                {"doubleProfile": {"profileAmplitude": .03, "minimumPhase": value,
                 "minimumDutyCycle": .2, "profileRoughness": .05},
                 "doubleExplainedVarianceImprovement": .2,
                 "doubleWaveMetrics": {"halfCycleDifferenceRatio": .4}}
                for value in (.05, .15, .30, .45, .60)
            ],
            "SHAPE_RAW_DOUBLE": [
                {"doubleProfile": {"profileAmplitude": .03, "minimumPhase": .2,
                 "minimumDutyCycle": .2, "profileRoughness": .05},
                 "doubleExplainedVarianceImprovement": gain,
                 "doubleWaveMetrics": {"halfCycleDifferenceRatio": half}}
                for gain, half in zip((.04, .08, .18, .32, .48), (.12, .20, .35, .55, .75))
            ],
        }
        for expected_family, sectors in cases.items():
            with self.subTest(expected_family=expected_family):
                evidence = _stationarity_evidence(sectors)
                self.assertTrue(evidence["followupWarranted"])
                self.assertIn(expected_family, evidence["triggeredEvidenceFamilies"])

    def test_resolved_evolving_real_time_frequency_preparation_uses_resolved_period(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = InvestigationStore(root / "investigations")
        investigation = store.create("resolved-evolving", WORKFLOW_ID, WORKFLOW_VERSION)
        primary = self._write_light_curve(root, "resolved-primary", 62, 0.0, "double")
        independent_paths = [
            self._write_light_curve(root, f"resolved-{sector}", sector, 0.1 * index, "double")
            for index, sector in enumerate((64, 65, 66), start=1)
        ]
        source_project = root / "source-project.json"
        source_project.write_text(json.dumps({
            "id": "resolved-source", "name": "Resolved evolving source",
            "workloadID": "openstar.lomb-scargle.v1", "datasets": [],
        }), encoding="utf-8")
        independent = {"preparedSectors": [
            {"datasetID": f"sector-{sector}", "sector": sector,
             "baselineDays": 30.0, "datasetPath": str(path)}
            for sector, path in zip((64, 65, 66), independent_paths)
        ]}
        morphology = {
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": 13.717836472675433,
            "continuationEvidence": {
                "timeFrequencyEvolutionWarranted": True,
                "entryReason": "RESOLVED_MORPHOLOGY_EVOLUTION_FOLLOWUP",
                "analysisReferencePeriodDays": 13.717836472675433,
                "periodReferenceKind": "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD",
            },
        }
        for stage_id, handler, result in (
            ("001-prepare-target", "openstar.tess.prepare-target", {
                "datasetPath": str(primary), "sector": 62,
                "sourceProjectPath": str(source_project),
                "sourceDatasetEntry": {"id": "primary", "targetName": "Resolved source"},
            }),
            ("006-prepare-independent", "openstar.tess.independent.prepare", independent),
            ("010-morphology", "openstar.tess.morphology.analyze", morphology),
        ):
            investigation = self._complete(store, investigation, stage_id, handler, result)
        engine = build_engine(store, coordinator=types.SimpleNamespace(), poll_interval=0.0, timeout=None)
        engine.chain_stages = False
        completed, next_request = engine.run_stage(
            investigation,
            StageRequest("011-prepare-time-frequency", "openstar.tess.time-frequency.prepare",
                         {"entryReason": "RESOLVED_MORPHOLOGY_EVOLUTION_FOLLOWUP"}, "010-morphology"),
            software_id="integration", software_version="20.29",
        )
        prepared = completed.stages[-1].result
        self.assertAlmostEqual(13.717836472675433, prepared["physicalPeriodDays"])
        self.assertEqual("MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD", prepared["periodReference"]["kind"])
        self.assertNotEqual("UNRESOLVED_FAMILY_ANALYSIS_REFERENCE", prepared["periodReference"]["kind"])
        self.assertEqual("openstar.tess.time-frequency.run", next_request.handler_id)

    @unittest.skipUnless(_real_numpy is not None, "real nonstationary integration requires NumPy")
    def test_blind_c_shaped_summary_routes_and_prepares_nonstationary_model(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = InvestigationStore(root / "investigations")
        investigation = store.create("blind-c-shaped", WORKFLOW_ID, WORKFLOW_VERSION)
        physical_period = 13.717836472675433

        paths = []
        origins = (1000.0, 1700.0, 2400.0)
        residual_frequency = 1.0 / 4.260988860233214
        injected_drift = 0.0004
        for index, (sector, origin) in enumerate(zip((62, 64, 65), origins)):
            path = root / f"frozen-{sector}.json"
            times = [sample * 0.02 for sample in range(1501)]
            absolute_times = [origin + value for value in times]
            residual_amplitude = (0.008, 0.020, 0.035)[index]
            flux = [
                1.0
                + 0.025 * math.sin(2.0 * math.pi * absolute / physical_period + 0.2 * index)
                + 0.012 * math.sin(4.0 * math.pi * absolute / physical_period - 0.1 * index)
                + residual_amplitude * math.sin(
                    2.0 * math.pi * residual_frequency
                    * (absolute + 0.5 * injected_drift * (absolute - 1715.0) ** 2)
                    + 0.7 * index
                )
                for absolute in absolute_times
            ]
            path.write_text(json.dumps({
                "id": f"frozen-{sector}", "targetName": "Frozen source",
                "source": {"sector": sector, "baselineDays": 30.0,
                           "originalTimeOriginDays": origin},
                "times": times, "flux": flux,
            }), encoding="utf-8")
            paths.append(path)
        source_project = root / "source-project.json"
        source_project.write_text(json.dumps({
            "id": "frozen-source", "name": "Frozen source",
            "workloadID": "openstar.lomb-scargle.v1", "datasets": [],
        }), encoding="utf-8")
        independent = {"preparedSectors": [
            {"datasetID": f"sector-{sector}", "sector": sector,
             "baselineDays": 30.0, "datasetPath": str(path)}
            for sector, path in zip((64, 65), paths[1:])
        ]}
        morphology = {
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": physical_period,
        }
        drift_indices = list(range(8)) + [25, 26]
        frequencies = [0.224687793443882 + 0.003 * index for index in drift_indices]
        interpretation = {
            "windowResults": [
                {
                    "sector": 64 if position < 4 else (65 if position < 8 else 66),
                    "role": "independent-time-frequency-window",
                    "windowIndex": position % 5,
                    "absoluteWindowCenterDays": 1100.0 + 100.0 * index,
                    "candidateFrequency": frequency,
                    "candidatePeriodDays": 1.0 / frequency,
                    "candidatePeakProminenceRatio": 2.0,
                    "acceptedTimeFrequencyFeature": True,
                    "nearEstablishedFamily": False,
                }
                for position, (index, frequency) in enumerate(zip(drift_indices, frequencies))
            ] + [
                {"sector": 62, "role": "primary-time-frequency-window",
                 "windowIndex": index, "absoluteWindowCenterDays": 1000.0 + 20.0 * index,
                 "candidateFrequency": 0.18, "candidatePeriodDays": 1.0 / 0.18,
                 "candidatePeakProminenceRatio": 1.0,
                 "acceptedTimeFrequencyFeature": False, "nearEstablishedFamily": False}
                for index in range(5)
            ],
            "familyTrack": [
                {"absoluteWindowCenterDays": 1100.0 + 350.0 * index,
                 "familyFit": {"fundamentalAmplitude": 0.01 * (index + 1),
                               "firstHarmonicAmplitude": 0.02,
                               "fundamentalPhaseRad": 0.8 * index,
                               "firstHarmonicPhaseRad": 0.4 * index}}
                for index in range(5)
            ],
        }
        for stage_id, handler, result in (
            ("001-prepare-target", "openstar.tess.prepare-target", {
                "datasetPath": str(paths[0]), "sector": 62,
                "ticID": 123456789,
                "sourceProjectPath": str(source_project),
                "sourceDatasetEntry": {"id": "primary", "targetName": "Frozen source"},
            }),
            ("002-catalog-identity", "openstar.tess.catalog-identity", {
                "tic": {"metadata": {"raDeg": 100.0, "decDeg": -30.0}},
            }),
            ("006-prepare-independent", "openstar.tess.independent.prepare", independent),
            ("010-morphology", "openstar.tess.morphology.analyze", morphology),
            ("013-interpret-time-frequency", "openstar.tess.time-frequency.interpret", interpretation),
        ):
            investigation = self._complete(store, investigation, stage_id, handler, result)

        class DeterministicDriftCoordinator:
            def run_project(self, project_path, **_kwargs):
                manifest = json.loads(Path(project_path).read_text(encoding="utf-8"))
                datasets = []
                for entry in manifest["datasets"]:
                    dataset = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
                    q = float(dataset["science"]["fractionalFrequencyDriftPerDay"])
                    if dataset["science"].get("role") in (
                        "residual-mode-pixel-localization",
                        "residual-mode-time-resolved-pixel-localization",
                    ):
                        sector = int(dataset["science"]["sector"])
                        row = int(dataset["science"]["pixelRow"])
                        column = int(dataset["science"]["pixelColumn"])
                        source_pixel = (1, 1) if sector == 64 else (3, 3)
                        power = 1.0 if (row, column) == source_pixel else 0.01
                    else:
                        power = 1.0 - abs(q - injected_drift) * 1000.0
                    datasets.append({
                        "datasetID": entry["id"], "periodStatus": "RELIABLE",
                        "periodConfidence": "high", "candidateFrequency": residual_frequency,
                        "candidatePeriodDays": 1.0 / residual_frequency,
                        "candidatePower": power,
                        "candidatePeakProminenceRatio": 3.0,
                    })
                return types.SimpleNamespace(
                    status={"datasets": datasets}, node_contributions={"synthetic-worker": len(datasets)},
                    project_id=manifest["id"],
                )

        engine = build_engine(store, coordinator=DeterministicDriftCoordinator(),
                              poll_interval=0.0, timeout=None)
        engine.chain_stages = False
        summarized, prepare_request = engine.run_stage(
            investigation,
            StageRequest("014-summarize-time-frequency", "openstar.tess.time-frequency.summarize", {},
                         "013-interpret-time-frequency"),
            software_id="integration", software_version="20.30",
        )
        summary_stage = summarized.stages[-1]
        self.assertEqual("DRIFTING_RESIDUAL_MODE", summary_stage.result["classification"])
        self.assertEqual(15, summary_stage.result["windowCount"])
        self.assertEqual(10, summary_stage.result["acceptedFeatureCount"])
        self.assertEqual("LONG_BASELINE_NONSTATIONARY_MODE_MODELING",
                         summary_stage.result["recommendedNextTest"])
        self.assertFalse(summary_stage.result["physicalMechanismResolved"])
        self.assertEqual("MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD",
                         summary_stage.result["periodReference"]["kind"])
        self.assertEqual("openstar.tess.nonstationary.prepare", prepare_request.handler_id)

        restarted = store.load(summarized.id)
        target = InvestigationTarget("synthetic:blind-c-shaped", restarted.id, WORKFLOW_ID,
                                     WORKFLOW_VERSION, 0, True, {})
        replayed = plan_tess_branches(restarted, target)[0].experiment
        self.assertEqual(prepare_request, replayed)
        self.assertEqual(len(summarized.stages), len(restarted.stages))
        prepared, run_request = engine.run_stage(
            restarted, replayed, software_id="integration", software_version="20.30"
        )
        preparation = prepared.stages[-1]
        self.assertEqual("openstar.tess.nonstationary.prepare", preparation.handler_id)
        self.assertAlmostEqual(physical_period, preparation.result["physicalPeriodDays"])
        self.assertEqual([64, 65], preparation.result["supportingSectors"])
        self.assertEqual({"morphology", "timeFrequency", "independentPreparation"},
                         set(preparation.provenance.input_hashes))
        self.assertEqual(33 * 2, len(preparation.result["preparedDatasets"]))
        self.assertEqual(33, preparation.result["driftGrid"]["count"])
        self.assertEqual(66 * 64, preparation.result["totalWorkUnits"])
        self.assertTrue(Path(preparation.result["projectPath"]).is_file())
        self.assertTrue(Path(preparation.result["analysisSeriesPath"]).is_file())
        self.assertTrue(all(Path(item["datasetPath"]).is_file()
                            for item in preparation.result["preparedDatasets"]))
        series = json.loads(Path(preparation.result["analysisSeriesPath"]).read_text(encoding="utf-8"))
        self.assertEqual(4503, len(series["absoluteTimes"]))
        self.assertEqual({62, 64, 65}, set(series["sectorIDs"]))
        self.assertTrue(all(item["residualStdDevBeforeNormalization"] > 0
                            for item in preparation.result["sectorResiduals"]))
        self.assertEqual("openstar.tess.nonstationary.run", run_request.handler_id)
        restarted_prepared = store.load(prepared.id)
        self.assertEqual(run_request, plan_tess_branches(restarted_prepared, target)[0].experiment)
        self.assertEqual(len(prepared.stages), len(restarted_prepared.stages))

        current = restarted_prepared
        request = run_request
        for expected_handler in (
            "openstar.tess.nonstationary.run",
            "openstar.tess.nonstationary.interpret",
            "openstar.tess.nonstationary.summarize",
        ):
            before = len(current.stages)
            current, request = engine.run_stage(
                current, request, software_id="integration", software_version="20.30"
            )
            self.assertEqual(expected_handler, current.stages[-1].handler_id)
            self.assertEqual(before + 1, len(current.stages))
            if expected_handler == "openstar.tess.nonstationary.interpret":
                interpreted = current.stages[-1].result
                self.assertEqual(len(preparation.result["preparedDatasets"]),
                                 len(interpreted["candidateResults"]))
                self.assertEqual(
                    {item["datasetID"] for item in preparation.result["preparedDatasets"]},
                    {item["datasetID"] for item in interpreted["candidateResults"]},
                )
                self.assertTrue(all(group["candidateCount"] == 33
                                    for group in interpreted["groups"].values()))
            restarted_boundary = store.load(current.id)
            self.assertEqual(request, plan_tess_branches(restarted_boundary, target)[0].experiment)
            self.assertEqual(len(current.stages), len(restarted_boundary.stages))
            current = restarted_boundary

        nonstationary = current.stages[-1].result
        self.assertEqual("NONSTATIONARY_DRIFT_WITH_SECTOR_EVOLUTION",
                         nonstationary["classification"])
        self.assertEqual("DRIFT_SECTOR_EVOLVING_MODE",
                         nonstationary["modelComparison"]["bestModelID"])
        self.assertEqual("RESIDUAL_MODE_PIXEL_LOCALIZATION",
                         nonstationary["recommendedNextTest"])
        self.assertFalse(nonstationary["physicalMechanismResolved"])
        self.assertNotIn("physicalPeriodDays", nonstationary["preferredModel"])
        self.assertAlmostEqual(physical_period, preparation.result["physicalPeriodDays"])
        self.assertAlmostEqual(residual_frequency,
                               nonstationary["preferredFrequencyAtReference"])
        self.assertEqual("openstar.tess.residual-mode-localization.prepare",
                         request.handler_id)

        class FrozenWCS:
            def world_to_pixel(self, _target):
                return 1.0, 1.0

            def pixel_to_world(self, x, y):
                return types.SimpleNamespace(x=x, y=y)

        class FrozenTPF:
            def __init__(self, sector):
                absolute = _real_numpy.arange(360, dtype=float) * 0.08 + {
                    64: 1700.0, 65: 2400.0,
                }[sector]
                cube = _real_numpy.ones((len(absolute), 5, 5), dtype=float)
                for row in range(5):
                    for column in range(5):
                        cube[:, row, column] += 0.001 * _real_numpy.sin(
                            2.0 * math.pi * absolute / (2.3 + 0.07 * row + 0.03 * column)
                            + 0.2 * row - 0.1 * column
                        )
                center = (1, 1) if sector == 64 else (3, 3)
                phase = 2.0 * math.pi * residual_frequency * (
                    absolute + 0.5 * injected_drift * (absolute - 1715.0) ** 2
                )
                cube[:, center[0], center[1]] += 0.04 * _real_numpy.sin(phase)
                physical_phase = 2.0 * math.pi * absolute / physical_period
                cube += 0.01 * _real_numpy.sin(physical_phase)[:, None, None]
                self.time = types.SimpleNamespace(value=absolute)
                self.flux = types.SimpleNamespace(value=cube)
                self.wcs = FrozenWCS()

        def frozen_tpf(**kwargs):
            return FrozenTPF(kwargs["sector"]), {
                "sourceType": "frozen-synthetic-tpf", "author": "regression",
                "cadenceSeconds": 6912.0,
            }

        with mock.patch(
            "workflows.tess.tess_residual_localization._target_coordinate",
            return_value=types.SimpleNamespace(),
        ), mock.patch(
            "workflows.tess.tess_residual_localization._download_tpf",
            side_effect=frozen_tpf,
        ):
            current, request = engine.run_stage(
                current, request, software_id="integration", software_version="20.30"
            )
        localization_preparation = current.stages[-1]
        self.assertEqual("openstar.tess.residual-mode-localization.prepare",
                         localization_preparation.handler_id)
        self.assertAlmostEqual(physical_period,
                               localization_preparation.result["physicalPeriodDays"])
        self.assertAlmostEqual(residual_frequency,
                               localization_preparation.result["residualFrequencyAtReference"])
        self.assertNotAlmostEqual(1.0 / physical_period,
                                  localization_preparation.result["residualFrequencyAtReference"])
        self.assertEqual("openstar.tess.residual-mode-localization.run", request.handler_id)
        self.assertEqual("frozen-synthetic-tpf",
                         localization_preparation.result["sectorMetadata"][0]["source"]["sourceType"])
        self.assertEqual(sha256_json(nonstationary),
                         localization_preparation.provenance.input_hashes["nonstationaryModeling"])

        restarted = store.load(current.id)
        self.assertEqual(request, plan_tess_branches(restarted, target)[0].experiment)
        current = restarted
        for expected_handler in (
            "openstar.tess.residual-mode-localization.run",
            "openstar.tess.residual-mode-localization.interpret",
        ):
            before = len(current.stages)
            current, request = engine.run_stage(
                current, request, software_id="integration", software_version="20.30"
            )
            self.assertEqual(expected_handler, current.stages[-1].handler_id)
            self.assertEqual(before + 1, len(current.stages))
            restarted = store.load(current.id)
            self.assertEqual(request, plan_tess_branches(restarted, target)[0].experiment)
            self.assertEqual(len(current.stages), len(restarted.stages))
            current = restarted

        localization = current.stages[-1]
        self.assertEqual("RESIDUAL_MODE_LOCALIZATION_UNRESOLVED",
                         localization.result["crossSector"]["classification"])
        self.assertEqual("UNRESOLVED",
                         localization.result["crossSector"]["residualModeOrigin"])
        self.assertEqual([64], localization.result["crossSector"]["targetSupportingSectors"])
        self.assertEqual([65], localization.result["crossSector"]["offTargetSectors"])
        self.assertEqual("RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW",
                         localization.result["recommendedNextTest"])
        self.assertFalse(localization.result["physicalMechanismResolved"])
        self.assertEqual("openstar.tess.residual-mode-localization-review.prepare",
                         request.handler_id)
        self.assertEqual(sha256_json(nonstationary),
                         localization.provenance.input_hashes["nonstationaryModeling"])

        with mock.patch(
            "workflows.tess.tess_residual_localization_review._download_tpf",
            side_effect=frozen_tpf,
        ), mock.patch(
            "workflows.tess.tess_residual_localization_review._pixel_scale_arcsec",
            return_value=21.0,
        ), mock.patch(
            "workflows.tess.tess_residual_localization_review._local_sky_jacobian",
            return_value={"eastArcsecPerPixelX": 21.0,
                          "eastArcsecPerPixelY": 0.0,
                          "northArcsecPerPixelX": 0.0,
                          "northArcsecPerPixelY": 21.0},
        ):
            current, request = engine.run_stage(
                current, request, software_id="integration", software_version="20.30"
            )
        review_preparation = current.stages[-1]
        self.assertEqual("openstar.tess.residual-mode-localization-review.prepare",
                         review_preparation.handler_id)
        self.assertEqual(sha256_json(localization.result),
                         review_preparation.provenance.input_hashes["residualModeLocalization"])
        self.assertEqual(sha256_json(nonstationary),
                         review_preparation.provenance.input_hashes["nonstationaryModeling"])
        self.assertAlmostEqual(physical_period,
                               review_preparation.result["physicalPeriodDays"])
        self.assertAlmostEqual(residual_frequency,
                               review_preparation.result["residualFrequencyAtReference"])
        self.assertNotAlmostEqual(1.0 / physical_period,
                                  review_preparation.result["residualFrequencyAtReference"])
        self.assertEqual({"TARGET_CONSISTENT", "OFF_TARGET"}, {
            item["v20_10StaticClassification"]
            for item in review_preparation.result["windowMetadata"]
        })
        self.assertTrue(all(item.get("skyJacobian")
                            for item in review_preparation.result["windowMetadata"]))

        restarted = store.load(current.id)
        self.assertEqual(request, plan_tess_branches(restarted, target)[0].experiment)
        current = restarted
        for expected_handler in (
            "openstar.tess.residual-mode-localization-review.run",
            "openstar.tess.residual-mode-localization-review.interpret",
        ):
            before = len(current.stages)
            current, request = engine.run_stage(
                current, request, software_id="integration", software_version="20.30"
            )
            self.assertEqual(expected_handler, current.stages[-1].handler_id)
            self.assertEqual(before + 1, len(current.stages))
            restarted = store.load(current.id)
            self.assertEqual(request, plan_tess_branches(restarted, target)[0].experiment)
            self.assertEqual(len(current.stages), len(restarted.stages))
            current = restarted

        review = current.stages[-1].result
        self.assertEqual("RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND",
                         review["crossTime"]["classification"])
        self.assertEqual("TIME_VARIABLE_OR_BLENDED",
                         review["crossTime"]["residualModeOrigin"])
        self.assertEqual("MULTI_SOURCE_RESIDUAL_DECOMPOSITION",
                         review["recommendedNextTest"])
        self.assertFalse(review["physicalMechanismResolved"])
        self.assertEqual("openstar.tess.finalize", request.handler_id)
        self.assertTrue(any(item["skySeparationArcsec"] > 0
                            for item in review["windowResults"]))
        self.assertTrue(any(item["offsetPixels"] > 0
                            for item in review["windowResults"]))

    def test_other_nonstationary_recommendations_do_not_enter_localization(self):
        localization = nonstationary_continuation(
            {"recommendedNextTest": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
             "physicalMechanismResolved": False},
            request_id="017-summarize-nonstationary",
        )
        self.assertEqual("openstar.tess.residual-mode-localization.prepare",
                         localization.handler_id)
        for recommendation, resolved in (
            ("RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW", False),
            ("MULTI_SOURCE_RESIDUAL_DECOMPOSITION", False),
            ("RESIDUAL_MODE_PIXEL_LOCALIZATION", True),
        ):
            with self.subTest(recommendation=recommendation, resolved=resolved):
                request = nonstationary_continuation(
                    {"recommendedNextTest": recommendation,
                     "physicalMechanismResolved": resolved},
                    request_id="017-summarize-nonstationary",
                )
                self.assertEqual("openstar.tess.finalize", request.handler_id)

    def test_only_unresolved_source_review_recommendation_enters_review(self):
        review = residual_mode_localization_continuation(
            {"recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW",
             "physicalMechanismResolved": False},
            request_id="020-interpret-residual-mode-localization",
        )
        self.assertEqual("openstar.tess.residual-mode-localization-review.prepare",
                         review.handler_id)
        for recommendation, resolved in (
            ("MULTI_SOURCE_RESIDUAL_DECOMPOSITION", False),
            ("RESIDUAL_MODE_PIXEL_LOCALIZATION", False),
            ("RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW", True),
        ):
            with self.subTest(recommendation=recommendation, resolved=resolved):
                request = residual_mode_localization_continuation(
                    {"recommendedNextTest": recommendation,
                     "physicalMechanismResolved": resolved},
                    request_id="020-interpret-residual-mode-localization",
                )
                self.assertEqual("openstar.tess.finalize", request.handler_id)

    def test_transient_recommendation_does_not_route_to_nonstationary(self):
        request = time_frequency_continuation(
            {"recommendedNextTest": "TRANSIENT_MODE_VALIDATION",
             "physicalMechanismResolved": False},
            request_id="014-summarize-time-frequency",
        )
        self.assertEqual("openstar.tess.finalize", request.handler_id)

    def test_weak_uninformative_morphology_does_not_continue(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        primary = self._write_light_curve(root, "primary-weak", 1, 0.0, "mixed")
        result = analyze_morphology(
            primary_dataset_path=primary,
            independent_spec={"preparedSectors": []},
            raw_period_days=6.86,
            possible_double_cycle_days=13.72,
        )
        self.assertFalse(result["physicalCycleResolved"])
        self.assertFalse(result["continuationEvidence"]["timeFrequencyEvolutionWarranted"])


if __name__ == "__main__":
    unittest.main()
