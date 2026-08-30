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
    _can_continue_long_baseline_frequency_confirmation,
)
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    repair_obsolete_terminal_wait,
)
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_long_baseline_frequency_confirmation import (
    HANDLER_ID,
    METHOD_CONTRACT_ID,
    analyze_long_baseline_frequency_confirmation,
    build_method_contract,
    classify_long_baseline_confirmation,
    method_contract_hash,
    validate_ambiguous_mode_identification,
    validate_frozen_dataset_lineage,
)
from workflows.tess.tess_resolved_cycle import authoritative_resolved_cycle


TIC_ID = 277940827
PRIMARY_SECTOR = 1
INDEPENDENT_SECTORS = (2, 3, 4, 5)
PHYSICAL_PERIOD_DAYS = 10.0
FAMILY_FREQUENCY = 1.0 / PHYSICAL_PERIOD_DAYS
TESTED_HARMONIC_ORDER = 4
TESTED_HARMONIC_FREQUENCY = (
    TESTED_HARMONIC_ORDER * FAMILY_FREQUENCY
)


def _write_dataset(path, *, dataset_id, sector, frequency, phase=0.2):
    origin = (sector - 1) * 100.0
    times = [20.0 * index / 159.0 for index in range(160)]
    absolute = [origin + value for value in times]
    flux = [
        0.7 * math.sin(2.0 * math.pi * FAMILY_FREQUENCY * time + 0.1)
        + 0.25 * math.cos(4.0 * math.pi * FAMILY_FREQUENCY * time - 0.3)
        + 0.5 * math.sin(2.0 * math.pi * frequency * time + phase)
        for time in absolute
    ]
    path.write_text(json.dumps({
        "id": dataset_id,
        "source": {
            "ticID": TIC_ID,
            "sector": sector,
            "originalTimeOriginDays": origin,
        },
        "times": times,
        "flux": flux,
    }), encoding="utf-8")


def _mode_result(paths, *, sectors=INDEPENDENT_SECTORS,
                 refined_frequency=0.34):
    family_bic = 100.0
    harmonic_bic = 80.0
    independent_bic = 65.0
    return {
        "classification": "AMBIGUOUS_HARMONIC_OR_MODE",
        "establishedPeriodFamily": {
            "referencePeriodDays": PHYSICAL_PERIOD_DAYS,
            "referenceFrequencyCyclesPerDay": FAMILY_FREQUENCY,
            "modeledHarmonicOrders": [1, 2, TESTED_HARMONIC_ORDER],
        },
        "residualCandidate": {
            "measuredPeriodDays": 1.0 / TESTED_HARMONIC_FREQUENCY,
            "measuredFrequencyCyclesPerDay": TESTED_HARMONIC_FREQUENCY,
            "refinedPeriodDays": 1.0 / refined_frequency,
            "refinedFrequencyCyclesPerDay": refined_frequency,
        },
        "harmonicRelation": {
            "testedOrder": TESTED_HARMONIC_ORDER,
            "harmonicFrequencyCyclesPerDay": TESTED_HARMONIC_FREQUENCY,
            "absoluteFrequencySeparation": 0.0,
            "frequencyResolutionCyclesPerDay": 0.01,
            "baselineDays": 100.0,
            "commensurateWithinResolution": True,
        },
        "modelComparison": {
            "criterion": "BIC",
            "conservativeThreshold": 10.0,
            "models": {
                "establishedFamily": {"bic": family_bic},
                "extendedHigherHarmonics": {"bic": harmonic_bic},
                "familyPlusIndependentFreeFrequency": {
                    "bic": independent_bic,
                },
            },
            "bicImprovementExtendedOverFamily": (
                family_bic - harmonic_bic
            ),
            "bicImprovementIndependentOverFamily": (
                family_bic - independent_bic
            ),
            "bicImprovementIndependentOverExtended": (
                harmonic_bic - independent_bic
            ),
        },
        "independentSectorSupport": {
            "sectors": list(sectors),
            "count": len(sectors),
            "requiredCount": 3,
            "sufficient": len(sectors) >= 3,
        },
        "independentModeEvidenceSurvived": False,
        "modeCandidate": None,
        "physicalMechanismResolved": False,
        "recommendedNextTest": (
            "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
        ),
        "dataReuse": {
            "frozenDatasetPaths": [str(path.resolve()) for path in paths],
            "downloadPerformed": False,
        },
    }


def _multimode_summary():
    points = [{
        "iteration": 1,
        "sector": sector,
        "role": "independent-residual-multimode",
        "candidateFrequency": TESTED_HARMONIC_FREQUENCY,
        "candidatePeriodDays": 1.0 / TESTED_HARMONIC_FREQUENCY,
        "candidatePeakProminenceRatio": 4.0,
        "acceptedDistinctMode": True,
    } for sector in INDEPENDENT_SECTORS]
    members = [{
        "iteration": item["iteration"],
        "sector": item["sector"],
        "role": item["role"],
        "frequency": item["candidateFrequency"],
        "periodDays": item["candidatePeriodDays"],
        "prominence": item["candidatePeakProminenceRatio"],
    } for item in points]
    recurrent = {
        "medianFrequency": TESTED_HARMONIC_FREQUENCY,
        "medianPeriodDays": 1.0 / TESTED_HARMONIC_FREQUENCY,
        "independentSectors": list(INDEPENDENT_SECTORS),
        "independentSectorCount": len(INDEPENDENT_SECTORS),
        "combinedSupport": False,
        "members": members,
    }
    return {
        "classification": "MULTI_MODE_RECURRENT",
        "physicalPeriodDays": PHYSICAL_PERIOD_DAYS,
        "physicalFrequency": FAMILY_FREQUENCY,
        "firstHarmonicFrequency": 2.0 * FAMILY_FREQUENCY,
        "iterationsCompleted": 2,
        "acceptedResidualModes": points,
        "frequencyClusters": [recurrent],
        "bestRecurrentSecondaryMode": recurrent,
        "independentSectorsWithAcceptedResidualModes": list(
            INDEPENDENT_SECTORS
        ),
        "minimumRecurrentIndependentSectorCount": 3,
        "clusterRelativeTolerance": 0.05,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": (
            "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
        ),
    }


def _fold(*, support, a, b, c, frequency):
    return {
        "support": support,
        "learnedIndependentFrequencyCyclesPerDay": frequency,
        "predictiveBIC": {"A": a, "B": b, "C": c},
        "failureOrInsufficiencyReasons": [],
    }


class LongBaselineAnalysisTests(unittest.TestCase):
    def _datasets(self, root, *, frequency=TESTED_HARMONIC_FREQUENCY):
        paths = []
        specs = []
        for role, sector in (
            ("PRIMARY", PRIMARY_SECTOR),
            *(("INDEPENDENT", sector) for sector in INDEPENDENT_SECTORS),
        ):
            dataset_id = f"sector-{sector}"
            path = Path(root) / f"{dataset_id}.json"
            _write_dataset(
                path,
                dataset_id=dataset_id,
                sector=sector,
                frequency=frequency,
            )
            paths.append(path)
            specs.append({
                "datasetID": dataset_id,
                "datasetPath": str(path.resolve()),
                "ticID": TIC_ID,
                "sector": sector,
                "role": role,
            })
        return paths, specs

    def test_method_contract_hash_is_deterministic_and_versioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = self._datasets(temporary)
            mode = _mode_result(paths)
            first = build_method_contract(mode)
            reordered = json.loads(json.dumps(mode, sort_keys=True))
            second = build_method_contract(reordered)
        self.assertEqual(METHOD_CONTRACT_ID, first["methodContractID"])
        self.assertEqual(first, second)
        self.assertEqual(method_contract_hash(first), method_contract_hash(second))
        self.assertTrue(
            first["crossValidation"]["frequencySelectionUsesTrainingSectorsOnly"]
        )
        self.assertFalse(first["crossValidation"]["heldOutFrequencySelection"])

    def test_leave_one_sector_out_frequency_selection_cannot_read_held_out_flux(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, specs = self._datasets(temporary, frequency=0.34)
            contract = build_method_contract(_mode_result(paths))
            first = analyze_long_baseline_frequency_confirmation(
                method_contract=contract,
                dataset_specs=specs,
            )
            held_out_path = paths[1]
            held_out = json.loads(held_out_path.read_text(encoding="utf-8"))
            held_out["flux"] = [
                25.0 * math.sin(index * 0.71)
                for index in range(len(held_out["flux"]))
            ]
            held_out_path.write_text(json.dumps(held_out), encoding="utf-8")
            second = analyze_long_baseline_frequency_confirmation(
                method_contract=contract,
                dataset_specs=specs,
            )
        first_fold = next(
            item for item in first["perSectorEvidence"]
            if item["heldOutSector"] == INDEPENDENT_SECTORS[0]
        )
        second_fold = next(
            item for item in second["perSectorEvidence"]
            if item["heldOutSector"] == INDEPENDENT_SECTORS[0]
        )
        self.assertEqual(
            first_fold["learnedIndependentFrequencyCyclesPerDay"],
            second_fold["learnedIndependentFrequencyCyclesPerDay"],
        )
        self.assertEqual(first["methodContractHash"], second["methodContractHash"])

    def test_classifies_harmonic_locked(self):
        folds = [
            _fold(
                support="HARMONIC", a=70.0, b=76.0, c=100.0,
                frequency=TESTED_HARMONIC_FREQUENCY,
            )
            for _ in range(4)
        ]
        result = classify_long_baseline_confirmation(
            folds, long_baseline_frequency_resolution=0.001
        )
        self.assertEqual(
            "HARMONIC_LOCKED_ACROSS_BASELINE", result["classification"]
        )
        self.assertEqual(
            "BINARY_ROTATION_EXTERNAL_EVIDENCE",
            result["recommendedNextTest"],
        )

    def test_classifies_independent_stable_mode(self):
        folds = [
            _fold(
                support="INDEPENDENT_MODE", a=95.0, b=65.0, c=100.0,
                frequency=0.34 + index * 0.0001,
            )
            for index in range(4)
        ]
        result = classify_long_baseline_confirmation(
            folds, long_baseline_frequency_resolution=0.001
        )
        self.assertEqual(
            "INDEPENDENT_STABLE_MODE_CONFIRMED", result["classification"]
        )
        self.assertEqual(
            "RESIDUAL_MODE_PIXEL_LOCALIZATION",
            result["recommendedNextTest"],
        )

    def test_classifies_nonstationary_or_intermittent_structure(self):
        folds = [
            _fold(support="INDEPENDENT_MODE", a=95, b=65, c=100,
                  frequency=0.32),
            _fold(support="INDEPENDENT_MODE", a=94, b=66, c=100,
                  frequency=0.37),
            _fold(support="HARMONIC", a=70, b=80, c=100,
                  frequency=0.40),
            _fold(support="NEITHER", a=98, b=97, c=100,
                  frequency=0.35),
        ]
        result = classify_long_baseline_confirmation(
            folds, long_baseline_frequency_resolution=0.001
        )
        self.assertEqual(
            "NONSTATIONARY_OR_INTERMITTENT_STRUCTURE",
            result["classification"],
        )
        self.assertEqual(
            "LONG_BASELINE_NONSTATIONARY_MODE_MODELING",
            result["recommendedNextTest"],
        )

    def test_classifies_inconclusive_without_predictive_structure(self):
        folds = [
            _fold(support="NEITHER", a=100, b=100, c=100,
                  frequency=0.34)
            for _ in range(4)
        ]
        result = classify_long_baseline_confirmation(
            folds, long_baseline_frequency_resolution=0.001
        )
        self.assertEqual(
            "LONG_BASELINE_CONFIRMATION_INCONCLUSIVE",
            result["classification"],
        )
        self.assertIsNone(result["recommendedNextTest"])

    def test_bic_threshold_edge_is_inclusive(self):
        folds = [
            _fold(support="INDEPENDENT_MODE", a=100, b=90, c=100,
                  frequency=0.34),
            _fold(support="INDEPENDENT_MODE", a=100, b=100, c=100,
                  frequency=0.34),
            _fold(support="INDEPENDENT_MODE", a=100, b=100, c=100,
                  frequency=0.34),
        ]
        result = classify_long_baseline_confirmation(
            folds, long_baseline_frequency_resolution=0.001
        )
        self.assertEqual(
            "INDEPENDENT_STABLE_MODE_CONFIRMED", result["classification"]
        )

    def test_corrupted_or_mismatched_frozen_dataset_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, specs = self._datasets(temporary)
            contract = build_method_contract(_mode_result(paths))
            corrupted = json.loads(paths[2].read_text(encoding="utf-8"))
            corrupted["source"]["sector"] = 999
            paths[2].write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                validate_frozen_dataset_lineage(
                    method_contract=contract,
                    dataset_specs=specs,
                )


class LongBaselineContinuationTests(unittest.TestCase):
    def _history(self, root, *, status="COMPLETE"):
        root = Path(root)
        store = InvestigationStore(root / "investigations")
        paths = []
        prepared_sectors = []
        for sector in (PRIMARY_SECTOR, *INDEPENDENT_SECTORS):
            dataset_id = f"sector-{sector}"
            path = root / f"{dataset_id}.json"
            _write_dataset(
                path,
                dataset_id=dataset_id,
                sector=sector,
                frequency=TESTED_HARMONIC_FREQUENCY,
            )
            paths.append(path)
            if sector != PRIMARY_SECTOR:
                prepared_sectors.append({
                    "sector": sector,
                    "datasetID": dataset_id,
                    "datasetPath": str(path.resolve()),
                })

        cycle = authoritative_resolved_cycle(morphology={
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": PHYSICAL_PERIOD_DAYS,
            "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED",
        })
        physical = {
            "version": "openstar.tess-physical-interpretation.v2",
            "physicalPeriodDays": PHYSICAL_PERIOD_DAYS,
            "photometricFirstHarmonicPeriodDays": PHYSICAL_PERIOD_DAYS / 2.0,
            "physicalCycleEvidence": copy.deepcopy(cycle),
            "physicalMechanismResolved": False,
            "recommendedNextTest": "PIXEL_LEVEL_SOURCE_LOCALIZATION",
        }
        localization = {
            "version": "openstar.tess-pixel-localization.v1",
            "physicalPeriodDays": PHYSICAL_PERIOD_DAYS,
            "photometricFirstHarmonicPeriodDays": PHYSICAL_PERIOD_DAYS / 2.0,
            "physicalCycleEvidence": copy.deepcopy(cycle),
            "crossSector": {
                "classification": "TARGET_SOURCE_SUPPORTED",
                "variableSignalOrigin": "TARGET_CONSISTENT",
                "targetSupportingSectors": list(INDEPENDENT_SECTORS),
                "recommendedNextTest": "MULTI_MODE_FREQUENCY_DECOMPOSITION",
            },
            "recommendedNextTest": "MULTI_MODE_FREQUENCY_DECOMPOSITION",
        }
        multimode = _multimode_summary()
        time_frequency = {
            "classification": "FAMILY_PHASE_EVOLUTION",
            "physicalMechanismResolved": False,
            "periodReference": {
                "periodDays": PHYSICAL_PERIOD_DAYS,
                "kind": "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD",
                "physicalCycleResolved": True,
            },
            "residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"},
            "familyEvolution": {"classification": "FAMILY_PHASE_EVOLUTION"},
            "recommendedNextTest": "MODE_IDENTIFICATION_OR_PULSATION_MODELING",
        }
        mode = _mode_result(paths)
        independent = {"preparedSectors": prepared_sectors}
        mode_hashes = {
            "multiModeDecomposition": sha256_json(multimode),
            "resolvedCycle": sha256_json(cycle),
            "physicalInterpretation": sha256_json(physical),
            "sourceLocalization": sha256_json(localization),
            "independentPreparation": sha256_json(independent),
        }
        mode_provenance = StageProvenance(
            software_id="fixture",
            software_version="20.39",
            input_hashes=mode_hashes,
        )
        prepared = {
            "datasetID": "sector-1",
            "datasetPath": str(paths[0].resolve()),
            "ticID": TIC_ID,
            "targetName": "Synthetic ambiguous mode",
            "sector": PRIMARY_SECTOR,
        }
        stages = [
            InvestigationStage(
                "001-prepare-target", "openstar.tess.prepare-target",
                "COMPLETE", None, {}, result=prepared,
            ),
            InvestigationStage(
                "002-hypotheses", "openstar.tess.hypotheses", "COMPLETE",
                "001-prepare-target", {}, result={
                    "rawCandidatePeriodDays": PHYSICAL_PERIOD_DAYS,
                    "observedPeriodDays": PHYSICAL_PERIOD_DAYS,
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
                "004-independent", "openstar.tess.independent.prepare",
                "COMPLETE", "003-planner", {}, result=independent,
            ),
            InvestigationStage(
                "005-physical", "openstar.tess.physical.interpret", "COMPLETE",
                "004-independent", {}, result=physical,
            ),
            InvestigationStage(
                "006-localization", "openstar.tess.source-localization.analyze",
                "COMPLETE", "005-physical", {}, result=localization,
            ),
            InvestigationStage(
                "007-multimode-interpret-1",
                "openstar.tess.multimode.interpret", "COMPLETE",
                "006-localization", {}, result={"iteration": 1},
            ),
            InvestigationStage(
                "008-multimode-interpret-2",
                "openstar.tess.multimode.interpret", "COMPLETE",
                "007-multimode-interpret-1", {}, result={"iteration": 2},
            ),
            InvestigationStage(
                "009-multimode", "openstar.tess.multimode.summarize", "COMPLETE",
                "008-multimode-interpret-2", {}, result=multimode,
            ),
            InvestigationStage(
                "010-time-frequency", "openstar.tess.time-frequency.summarize",
                "COMPLETE", "009-multimode", {}, result=time_frequency,
            ),
        ]
        for number in range(11, 34):
            stages.append(InvestigationStage(
                f"{number:03d}-preserved-evidence",
                "openstar.tess.preserved-evidence", "COMPLETE",
                stages[-1].id, {}, result={},
            ))
        stages.append(InvestigationStage(
            "034-mode-identification",
            "openstar.tess.mode-identification.analyze", "COMPLETE",
            "009-multimode", {
                "evidenceLineage":
                "MULTIMODE_RECURRENT_SECONDARY_FREQUENCY"
            }, result=mode, provenance=mode_provenance,
        ))
        conclusion = {
            "claim": {
                "claim": "CANDIDATE_PERIOD",
                "rationale": ["Frozen pre-continuation claim."],
            },
            "sourceLocalization": localization,
            "multiModeDecomposition": multimode,
            "timeFrequencyEvolution": time_frequency,
            "modeIdentification": mode,
            "recommendedNextTest": (
                "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
            ),
        }
        stages.append(InvestigationStage(
            "035-finalize", "openstar.tess.finalize", "COMPLETE",
            "034-mode-identification",
            {"outputSuffix": "v20.9-mode-identification"},
            result=conclusion, stop=True,
        ))
        investigation = store.create(
            "ambiguous-mode", WORKFLOW_ID, WORKFLOW_VERSION,
            metadata={"controlState": {
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            }},
        )
        investigation = replace(
            investigation,
            status=status,
            stages=tuple(stages),
        )
        store.save(investigation)
        return store, investigation

    def test_exact_terminal_continuation_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            _can_continue_long_baseline_frequency_confirmation(investigation)
        self.assertEqual(35, len(investigation.stages))

    def test_manual_validation_rejects_running_or_nonterminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary, status="RUNNING")
            with self.assertRaisesRegex(RuntimeError, "terminal investigation"):
                _can_continue_long_baseline_frequency_confirmation(investigation)
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            running = InvestigationStage(
                "036-running", "openstar.tess.other", "RUNNING",
                investigation.stages[-1].id, {},
            )
            investigation = replace(
                investigation, stages=investigation.stages + (running,)
            )
            with self.assertRaisesRegex(RuntimeError, "RUNNING stage"):
                _can_continue_long_baseline_frequency_confirmation(investigation)

    def test_rejects_altered_mode_evidence_or_wrong_recommendation(self):
        for mutation in (
            lambda result: result.update(classification="OTHER"),
            lambda result: result.update(recommendedNextTest="OTHER"),
            lambda result: result["modelComparison"].update(
                bicImprovementIndependentOverExtended=999.0
            ),
        ):
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as temporary:
                _, investigation = self._history(temporary)
                stages = list(investigation.stages)
                mode = copy.deepcopy(stages[-2].result)
                mutation(mode)
                stages[-2] = replace(stages[-2], result=mode)
                changed = replace(investigation, stages=tuple(stages))
                with self.assertRaises(RuntimeError):
                    _can_continue_long_baseline_frequency_confirmation(changed)

    def test_rejects_existing_stage_and_insufficient_independent_sectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            existing = InvestigationStage(
                "036-long-baseline-frequency-confirmation", HANDLER_ID,
                "COMPLETE", "034-mode-identification", {}, result={},
            )
            changed = replace(
                investigation, stages=investigation.stages + (existing,)
            )
            with self.assertRaisesRegex(RuntimeError, "already contains"):
                _can_continue_long_baseline_frequency_confirmation(changed)
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            stages = list(investigation.stages)
            mode = copy.deepcopy(stages[-2].result)
            mode["independentSectorSupport"] = {
                "sectors": [2, 3], "count": 2,
                "requiredCount": 3, "sufficient": False,
            }
            mode["dataReuse"]["frozenDatasetPaths"] = (
                mode["dataReuse"]["frozenDatasetPaths"][:3]
            )
            stages[-2] = replace(stages[-2], result=mode)
            with self.assertRaisesRegex(RuntimeError, "At least three"):
                _can_continue_long_baseline_frequency_confirmation(
                    replace(investigation, stages=tuple(stages))
                )

    def test_automatic_recovery_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._history(temporary)
            immutable = tuple(
                json.dumps(asdict(stage), sort_keys=True)
                for stage in investigation.stages
            )
            with mock.patch.object(
                store, "verified_terminal_stage_ledger_hash",
                return_value=True,
            ), mock.patch(
                "workflows.tess.tess_autonomy._verified_stage_json",
                return_value=True,
            ):
                repaired = repair_obsolete_terminal_wait(store, investigation)
                repeated = repair_obsolete_terminal_wait(store, repaired)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(immutable, tuple(
            json.dumps(asdict(stage), sort_keys=True)
            for stage in repaired.stages
        ))
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(
            "036-long-baseline-frequency-confirmation", selected["id"]
        )
        self.assertEqual(HANDLER_ID, selected["handler_id"])
        self.assertEqual(
            "034-mode-identification", selected["triggered_by_stage_id"]
        )
        self.assertEqual(repaired, repeated)

    def test_handler_finalizes_and_persists_without_claim_upgrade(self):
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
                    "036-long-baseline-frequency-confirmation",
                    HANDLER_ID,
                    {},
                    "034-mode-identification",
                ),
                software_id="integration",
                software_version="20.40",
            )
            self.assertEqual(frozen_stages, completed.stages[:-1])
            self.assertEqual("openstar.tess.finalize", finalize.handler_id)
            self.assertEqual(
                "v20.9.1-long-baseline-frequency-confirmation",
                finalize.parameters["outputSuffix"],
            )
            final, next_request = engine.run_stage(
                completed,
                finalize,
                software_id="integration",
                software_version="20.40",
            )
            conclusion = final.stages[-1].result
            persisted = conclusion["longBaselineFrequencyConfirmation"]
            report = Path(conclusion["reportPath"]).read_text(encoding="utf-8")
            conclusion_file = json.loads(
                Path(conclusion["conclusionPath"]).read_text(encoding="utf-8")
            )
        self.assertIsNone(next_request)
        self.assertEqual("CANDIDATE_PERIOD", conclusion["claim"]["claim"])
        self.assertFalse(persisted["physicalMechanismResolved"])
        self.assertFalse(persisted["claimLevelChanged"])
        self.assertEqual(METHOD_CONTRACT_ID, persisted["methodContractID"])
        self.assertEqual(persisted, conclusion_file[
            "longBaselineFrequencyConfirmation"
        ])
        self.assertIn("Long-baseline frequency confirmation", report)
        self.assertIn("Physical mechanism resolved: False", report)
        coordinator.assert_not_called()


class AmbiguousModeEvidenceValidationTests(unittest.TestCase):
    def test_exact_evidence_validates_and_wrong_recommendation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"dataset-{index}.json"
                     for index in range(5)]
            result = _mode_result(paths)
            validated = validate_ambiguous_mode_identification(result)
            self.assertEqual(
                list(INDEPENDENT_SECTORS), validated["independentSectors"]
            )
            changed = copy.deepcopy(result)
            changed["recommendedNextTest"] = "LONG_BASELINE_NONSTATIONARY_MODE_MODELING"
            with self.assertRaisesRegex(RuntimeError, "exact unresolved"):
                validate_ambiguous_mode_identification(changed)


if __name__ == "__main__":
    unittest.main()
