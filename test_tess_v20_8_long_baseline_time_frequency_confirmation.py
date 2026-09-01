import copy
import json
import math
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from openstar_investigation import (
    InvestigationStage,
    InvestigationStore,
    StageProvenance,
    sha256_json,
)
from openstar_workflow import StageRequest
from run_tess_investigation import (
    _can_continue_v20_8_long_baseline_time_frequency_confirmation,
)
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    _repair_v20_8_long_baseline_time_frequency_confirmation_terminal,
)
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation import (
    COHERENT,
    HANDLER_ID,
    HARMONIC,
    INCONCLUSIVE,
    INTERMITTENT,
    METHOD_CONTRACT_ID,
    NONSTATIONARY,
    analyze_long_baseline_time_frequency_confirmation,
    build_dataset_specs,
    build_method_contract,
    classify_confirmation,
    method_contract_hash,
    validate_frozen_window_lineage,
)


TIC_ID = 52365081
PRIMARY_SECTOR = 1
INDEPENDENT_SECTORS = (28, 68, 69)
FAMILY_PERIOD_DAYS = 14.102965067314074
FAMILY_FREQUENCY = 1.0 / FAMILY_PERIOD_DAYS
HARMONIC_FREQUENCY = 4.0 * FAMILY_FREQUENCY


def _write_window(
    path,
    *,
    dataset_id,
    sector,
    role,
    window_index,
    frequency=HARMONIC_FREQUENCY,
    phase=0.25,
):
    origin = float(sector * 100.0)
    window_start = float((window_index - 1) * 12.0)
    local_times = [10.0 * index / 127.0 for index in range(128)]
    absolute_times = [
        origin + window_start + value for value in local_times
    ]
    flux = [
        0.55 * math.sin(2.0 * math.pi * frequency * time + phase)
        for time in absolute_times
    ]
    center_dataset_days = window_start + 5.0
    path.write_text(json.dumps({
        "id": dataset_id,
        "source": {
            "ticID": TIC_ID,
            "sector": sector,
            "originalTimeOriginDays": origin,
            "timeFrequencyWindowIndex": window_index,
            "windowStartDatasetDays": window_start,
            "windowCenterDatasetDays": center_dataset_days,
            "absoluteWindowCenterDays": origin + center_dataset_days,
        },
        "metadata": {},
        "science": {
            "purpose": "sliding-window-time-frequency-evolution",
            "role": role,
            "windowIndex": window_index,
        },
        "times": local_times,
        "flux": flux,
    }), encoding="utf-8")


def _evidence(root, *, sectors=INDEPENDENT_SECTORS):
    prepared_windows = []
    window_results = []
    entries = ((PRIMARY_SECTOR, "primary-time-frequency-window"), *(
        (sector, "independent-time-frequency-window")
        for sector in sectors
    ))
    for window_index, (sector, role) in enumerate(entries, 1):
        dataset_id = f"sector-{sector}-window-{window_index}"
        path = Path(root) / f"{dataset_id}.json"
        candidate_frequency = HARMONIC_FREQUENCY + window_index * 0.0002
        _write_window(
            path,
            dataset_id=dataset_id,
            sector=sector,
            role=role,
            window_index=window_index,
            frequency=candidate_frequency,
        )
        prepared_windows.append({
            "datasetID": dataset_id,
            "datasetPath": str(path.resolve()),
            "sectorKey": str(sector),
            "sector": sector,
            "role": role,
            "windowIndex": window_index,
            "sampleCount": 128,
            "baselineDays": 10.0,
        })
        window_results.append({
            "datasetID": dataset_id,
            "sectorKey": str(sector),
            "sector": sector,
            "role": role,
            "windowIndex": window_index,
            "candidateFrequency": candidate_frequency,
            "candidatePeriodDays": 1.0 / candidate_frequency,
            "acceptedTimeFrequencyFeature": True,
            "nearEstablishedFamily": False,
        })
    accepted_independent = [
        item for item in window_results
        if item["role"] == "independent-time-frequency-window"
    ]
    preparation = {
        "available": True,
        "physicalPeriodDays": FAMILY_PERIOD_DAYS,
        "physicalFrequency": FAMILY_FREQUENCY,
        "firstHarmonicFrequency": 2.0 * FAMILY_FREQUENCY,
        "subtractedHarmonicOrders": [1, 2],
        "preparedWindows": prepared_windows,
    }
    interpretation = {
        "physicalFrequency": FAMILY_FREQUENCY,
        "firstHarmonicFrequency": 2.0 * FAMILY_FREQUENCY,
        "windowResults": copy.deepcopy(window_results),
        "acceptedFeatureCount": len(window_results),
    }
    summary = {
        "classification": "NONSTATIONARY_VARIABILITY",
        "physicalPeriodDays": FAMILY_PERIOD_DAYS,
        "physicalFrequency": FAMILY_FREQUENCY,
        "windowCount": len(window_results),
        "acceptedFeatureCount": len(window_results),
        "acceptedIndependentFeatureCount": len(accepted_independent),
        "acceptedIndependentSectors": sorted(sectors),
        "windowResults": copy.deepcopy(window_results),
        "residualEvolution": {
            "classification": "NONSTATIONARY_RESIDUAL_VARIABILITY",
        },
        "familyEvolution": {"classification": "FAMILY_PHASE_EVOLUTION"},
        "periodReference": {
            "periodDays": FAMILY_PERIOD_DAYS,
            "kind": "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE",
            "physicalCycleResolved": False,
        },
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": (
            "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
        ),
    }
    return preparation, interpretation, summary


def _fold(support, h_bic, s_bic, n_bic, frequency):
    return {
        "support": support,
        "learnedCoherentFrequencyCyclesPerDay": frequency,
        "predictiveBIC": {"H": h_bic, "S": s_bic, "N": n_bic},
        "failureOrInsufficiencyReasons": [],
    }


class V208LongBaselineAnalysisTests(unittest.TestCase):
    def test_method_contract_hash_is_deterministic_and_precedes_flux(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation, interpretation, summary = _evidence(temporary)
            reordered = json.loads(json.dumps(summary, sort_keys=True))
            with mock.patch(
                "workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation._load_frozen_dataset"
            ) as loader:
                first = build_method_contract(
                    preparation=preparation,
                    interpretation=interpretation,
                    summary=summary,
                )
                second = build_method_contract(
                    preparation=json.loads(json.dumps(preparation, sort_keys=True)),
                    interpretation=json.loads(json.dumps(
                        interpretation, sort_keys=True
                    )),
                    summary=reordered,
                )
            loader.assert_not_called()
        self.assertEqual(METHOD_CONTRACT_ID, first["methodContractID"])
        self.assertEqual(first, second)
        self.assertEqual(method_contract_hash(first), method_contract_hash(second))
        self.assertFalse(first["crossValidation"]["heldOutFrequencySelection"])
        self.assertFalse(first["crossValidation"]["heldOutPhaseSelection"])

    def test_leave_one_sector_out_selection_cannot_read_held_out_flux(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation, interpretation, summary = _evidence(temporary)
            contract = build_method_contract(
                preparation=preparation,
                interpretation=interpretation,
                summary=summary,
            )
            specs = build_dataset_specs(
                expected_tic_id=TIC_ID, preparation=preparation
            )
            first = analyze_long_baseline_time_frequency_confirmation(
                method_contract=contract, dataset_specs=specs
            )
            held_out_spec = next(
                item for item in specs
                if item["sector"] == INDEPENDENT_SECTORS[0]
            )
            path = Path(held_out_spec["datasetPath"])
            dataset = json.loads(path.read_text(encoding="utf-8"))
            dataset["flux"] = [
                40.0 * math.sin(index * 0.73)
                for index in range(len(dataset["flux"]))
            ]
            path.write_text(json.dumps(dataset), encoding="utf-8")
            second = analyze_long_baseline_time_frequency_confirmation(
                method_contract=contract, dataset_specs=specs
            )
        first_fold = next(
            fold for fold in first["perSectorEvidence"]
            if fold["heldOutSector"] == INDEPENDENT_SECTORS[0]
        )
        second_fold = next(
            fold for fold in second["perSectorEvidence"]
            if fold["heldOutSector"] == INDEPENDENT_SECTORS[0]
        )
        self.assertEqual(
            first_fold["learnedCoherentFrequencyCyclesPerDay"],
            second_fold["learnedCoherentFrequencyCyclesPerDay"],
        )

    def test_corrupted_or_mismatched_frozen_windows_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            preparation, interpretation, summary = _evidence(temporary)
            contract = build_method_contract(
                preparation=preparation,
                interpretation=interpretation,
                summary=summary,
            )
            specs = build_dataset_specs(
                expected_tic_id=TIC_ID, preparation=preparation
            )
            path = Path(specs[1]["datasetPath"])
            dataset = json.loads(path.read_text(encoding="utf-8"))
            dataset["source"]["sector"] = 999
            path.write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                validate_frozen_window_lineage(
                    method_contract=contract, dataset_specs=specs
                )

    def test_coherent_harmonic_nonstationary_intermittent_and_inconclusive(self):
        coherent = classify_confirmation([
            _fold("COHERENT", 100, 70, 100, 0.31 + index * 0.0001)
            for index in range(3)
        ], long_baseline_frequency_resolution=0.001,
            accepted_independent_window_count=3,
            independent_window_count=3)
        harmonic = classify_confirmation([
            _fold("HARMONIC", 70, 100, 100, HARMONIC_FREQUENCY)
            for _ in range(3)
        ], long_baseline_frequency_resolution=0.001,
            accepted_independent_window_count=3,
            independent_window_count=3)
        nonstationary = classify_confirmation([
            _fold("NEITHER", 100, 100, 100, 0.28 + index * 0.02)
            for index in range(3)
        ], long_baseline_frequency_resolution=0.001,
            accepted_independent_window_count=4,
            independent_window_count=5)
        intermittent = classify_confirmation([
            _fold("NEITHER", 100, 100, 100, 0.28 + index * 0.02)
            for index in range(3)
        ], long_baseline_frequency_resolution=0.001,
            accepted_independent_window_count=3,
            independent_window_count=6)
        inconclusive = classify_confirmation([
            _fold("NEITHER", 100, 100, 100, 0.30)
            for _ in range(3)
        ], long_baseline_frequency_resolution=0.001,
            accepted_independent_window_count=2,
            independent_window_count=6)
        self.assertEqual(COHERENT, coherent["classification"])
        self.assertEqual(HARMONIC, harmonic["classification"])
        self.assertEqual(NONSTATIONARY, nonstationary["classification"])
        self.assertEqual(INTERMITTENT, intermittent["classification"])
        self.assertEqual(INCONCLUSIVE, inconclusive["classification"])

    def test_bic_threshold_edge_is_inclusive(self):
        folds = [
            _fold("COHERENT", 100, 90, 100, 0.31),
            _fold("COHERENT", 100, 100, 100, 0.31),
            _fold("COHERENT", 100, 100, 100, 0.31),
        ]
        result = classify_confirmation(
            folds,
            long_baseline_frequency_resolution=0.001,
            accepted_independent_window_count=3,
            independent_window_count=3,
        )
        self.assertEqual(COHERENT, result["classification"])


class V208LongBaselineContinuationTests(unittest.TestCase):
    def _history(self, root, *, status="COMPLETE"):
        root = Path(root)
        store = InvestigationStore(root / "investigations")
        preparation, interpretation, summary = _evidence(root)
        run_result = {"datasets": []}
        morphology = {
            "physicalCycleResolved": False,
            "resolvedPhysicalPeriodDays": None,
            "continuationEvidence": {
                "timeFrequencyEvolutionWarranted": True,
                "analysisReferencePeriodDays": FAMILY_PERIOD_DAYS,
            },
        }
        prepared = {
            "datasetID": "primary-sector-1",
            "datasetPath": preparation["preparedWindows"][0]["datasetPath"],
            "ticID": TIC_ID,
            "targetName": "Synthetic v20.8 boundary",
            "sector": PRIMARY_SECTOR,
        }
        preparation_provenance = StageProvenance(
            software_id="fixture",
            software_version="1",
            input_hashes={"morphology": sha256_json(morphology)},
        )
        interpretation_provenance = StageProvenance(
            software_id="fixture",
            software_version="1",
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run_result),
            },
        )
        summary_provenance = StageProvenance(
            software_id="fixture",
            software_version="1",
            input_hashes={
                "morphology": sha256_json(morphology),
                "timeFrequencyInterpretation": sha256_json(interpretation),
            },
        )
        stages = [
            InvestigationStage(
                "001-prepare-target", "openstar.tess.prepare-target",
                "COMPLETE", None, {}, result=prepared,
            ),
            InvestigationStage(
                "002-hypotheses", "openstar.tess.hypotheses", "COMPLETE",
                "001-prepare-target", {}, result={
                    "rawCandidatePeriodDays": FAMILY_PERIOD_DAYS / 2.0,
                    "observedPeriodDays": FAMILY_PERIOD_DAYS / 2.0,
                    "possibleDoubleCycleDays": FAMILY_PERIOD_DAYS,
                },
            ),
            InvestigationStage(
                "003-planner", "openstar.tess.planner", "COMPLETE",
                "002-hypotheses", {}, result={
                    "claimDecision": {
                        "claim": "CANDIDATE_PERIOD",
                        "rationale": ["Frozen pre-continuation claim."],
                    },
                },
            ),
            InvestigationStage(
                "004-morphology", "openstar.tess.morphology.analyze",
                "COMPLETE", "003-planner", {}, result=morphology,
            ),
            InvestigationStage(
                "005-prepare-time-frequency",
                "openstar.tess.time-frequency.prepare", "COMPLETE",
                "004-morphology", {
                    "entryReason": "UNRESOLVED_EVOLVING_MORPHOLOGY"
                }, result=preparation, provenance=preparation_provenance,
            ),
            InvestigationStage(
                "006-run-time-frequency", "openstar.tess.time-frequency.run",
                "COMPLETE", "005-prepare-time-frequency", {},
                result=run_result,
            ),
            InvestigationStage(
                "007-interpret-time-frequency",
                "openstar.tess.time-frequency.interpret", "COMPLETE",
                "006-run-time-frequency", {}, result=interpretation,
                provenance=interpretation_provenance,
            ),
            InvestigationStage(
                "008-summarize-time-frequency",
                "openstar.tess.time-frequency.summarize", "COMPLETE",
                "007-interpret-time-frequency", {}, result=summary,
                provenance=summary_provenance,
            ),
        ]
        conclusion = {
            "claim": {
                "claim": "CANDIDATE_PERIOD",
                "rationale": ["Frozen pre-continuation claim."],
            },
            "timeFrequencyEvolution": summary,
            "recommendedNextTest": (
                "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
            ),
        }
        stages.append(InvestigationStage(
            "009-finalize", "openstar.tess.finalize", "COMPLETE",
            "008-summarize-time-frequency", {"outputSuffix": "v20.8"},
            result=conclusion, stop=True,
        ))
        investigation = store.create(
            "v20-8-unresolved", WORKFLOW_ID, WORKFLOW_VERSION,
            metadata={"controlState": {
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            }},
        )
        investigation = replace(
            investigation, status=status, stages=tuple(stages)
        )
        store.save(investigation)
        return store, investigation

    def test_exact_terminal_manual_continuation_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                investigation
            )

    def test_manual_validation_rejects_running_and_nonterminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary, status="RUNNING")
            with self.assertRaisesRegex(RuntimeError, "terminal investigation"):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    investigation
                )
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            running = InvestigationStage(
                "010-running", "openstar.tess.other", "RUNNING",
                "009-finalize", {},
            )
            with self.assertRaisesRegex(RuntimeError, "RUNNING stage"):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    replace(
                        investigation,
                        stages=investigation.stages + (running,),
                    )
                )

    def test_rejects_altered_evidence_wrong_recommendation_and_existing_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            stages = list(investigation.stages)
            changed_summary = copy.deepcopy(stages[-2].result)
            changed_summary["residualEvolution"]["classification"] = "OTHER"
            stages[-2] = replace(stages[-2], result=changed_summary)
            changed_final = copy.deepcopy(stages[-1].result)
            changed_final["timeFrequencyEvolution"] = changed_summary
            stages[-1] = replace(stages[-1], result=changed_final)
            with self.assertRaises(RuntimeError):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    replace(investigation, stages=tuple(stages))
                )
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            stages = list(investigation.stages)
            wrong_final = copy.deepcopy(stages[-1].result)
            wrong_final["recommendedNextTest"] = "OTHER"
            stages[-1] = replace(stages[-1], result=wrong_final)
            with self.assertRaisesRegex(RuntimeError, "exact finalized"):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    replace(investigation, stages=tuple(stages))
                )
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            existing = InvestigationStage(
                "010-confirm", HANDLER_ID, "COMPLETE",
                "008-summarize-time-frequency", {}, result={},
            )
            with self.assertRaisesRegex(RuntimeError, "already contains"):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    replace(
                        investigation,
                        stages=investigation.stages + (existing,),
                    )
                )

    def test_rejects_insufficient_independent_sectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            stages = list(investigation.stages)
            summary = copy.deepcopy(stages[-2].result)
            summary["acceptedIndependentSectors"] = [28, 68]
            summary["acceptedIndependentFeatureCount"] = 2
            summary["acceptedFeatureCount"] = 3
            summary["windowCount"] = 3
            summary["windowResults"] = summary["windowResults"][:3]
            stages[-2] = replace(stages[-2], result=summary)
            final = copy.deepcopy(stages[-1].result)
            final["timeFrequencyEvolution"] = summary
            stages[-1] = replace(stages[-1], result=final)
            with self.assertRaises(RuntimeError):
                _can_continue_v20_8_long_baseline_time_frequency_confirmation(
                    replace(investigation, stages=tuple(stages))
                )

    def test_automatic_repair_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._history(temporary)
            immutable = tuple(
                json.dumps(asdict(stage), sort_keys=True)
                for stage in investigation.stages
            )
            control = investigation.metadata["controlState"]
            with mock.patch.object(
                store, "verified_terminal_stage_ledger_hash",
                return_value=True,
            ), mock.patch(
                "workflows.tess.tess_autonomy._verified_stage_json",
                return_value=True,
            ):
                repaired = (
                    _repair_v20_8_long_baseline_time_frequency_confirmation_terminal(
                        store, investigation, control
                    )
                )
                repeated = (
                    _repair_v20_8_long_baseline_time_frequency_confirmation_terminal(
                        store, repaired, repaired.metadata["controlState"]
                    )
                )
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(immutable, tuple(
            json.dumps(asdict(stage), sort_keys=True)
            for stage in repaired.stages
        ))
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(
            "010-long-baseline-time-frequency-confirmation", selected["id"]
        )
        self.assertEqual(HANDLER_ID, selected["handler_id"])
        self.assertEqual(
            "008-summarize-time-frequency",
            selected["triggered_by_stage_id"],
        )
        self.assertIsNone(repeated)

    def test_handler_persists_report_without_claim_or_mechanism_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._history(temporary)
            frozen_stages = copy.deepcopy(investigation.stages)
            investigation = store.set_status(investigation, "RUNNING")
            coordinator = mock.Mock()
            engine = build_engine(
                store, coordinator, poll_interval=0.0, timeout=None
            )
            engine.chain_stages = False
            completed, finalize = engine.run_stage(
                investigation,
                StageRequest(
                    "010-long-baseline-time-frequency-confirmation",
                    HANDLER_ID,
                    {},
                    "008-summarize-time-frequency",
                ),
                software_id="integration",
                software_version="1",
            )
            self.assertEqual(frozen_stages, completed.stages[:-1])
            self.assertEqual("openstar.tess.finalize", finalize.handler_id)
            self.assertEqual(
                "v20.8.1-long-baseline-time-frequency-confirmation",
                finalize.parameters["outputSuffix"],
            )
            final, next_request = engine.run_stage(
                completed,
                finalize,
                software_id="integration",
                software_version="1",
            )
            conclusion = final.stages[-1].result
            persisted = conclusion[
                "longBaselineTimeFrequencyConfirmation"
            ]
            report = Path(conclusion["reportPath"]).read_text(encoding="utf-8")
            conclusion_file = json.loads(
                Path(conclusion["conclusionPath"]).read_text(encoding="utf-8")
            )
        self.assertIsNone(next_request)
        self.assertEqual("CANDIDATE_PERIOD", conclusion["claim"]["claim"])
        self.assertFalse(persisted["physicalMechanismResolved"])
        self.assertFalse(persisted["claimLevelChanged"])
        self.assertFalse(persisted["automaticDiscoveryClaim"])
        self.assertEqual(METHOD_CONTRACT_ID, persisted["methodContractID"])
        self.assertEqual(
            persisted,
            conclusion_file["longBaselineTimeFrequencyConfirmation"],
        )
        self.assertIn(
            "v20.8 long-baseline time-frequency confirmation", report
        )
        self.assertIn("Physical mechanism resolved: False", report)
        coordinator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
