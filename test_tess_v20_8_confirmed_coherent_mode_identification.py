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
    sha256_file,
    sha256_json,
)
from openstar_workflow import StageRequest
from run_tess_investigation import (
    _can_continue_confirmed_coherent_mode_identification,
)
from test_tess_v20_8_long_baseline_time_frequency_confirmation import (
    FAMILY_PERIOD_DAYS,
    INDEPENDENT_SECTORS,
    PRIMARY_SECTOR,
    TIC_ID,
    _evidence,
)
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    _repair_v20_8_confirmed_coherent_mode_identification_terminal,
)
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_mode_identification import (
    CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID,
    V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE,
    analyze_confirmed_coherent_residual_mode,
    build_confirmed_coherent_mode_method_contract,
    confirmed_coherent_mode_method_contract_hash,
    validate_confirmed_coherent_mode_dataset_lineage,
    validate_v20_8_confirmed_coherent_residual,
)
from workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation import (
    COHERENT,
    HANDLER_ID as CONFIRMATION_HANDLER_ID,
    METHOD_CONTRACT_ID as CONFIRMATION_METHOD_CONTRACT_ID,
    build_dataset_specs as build_confirmation_dataset_specs,
    build_method_contract as build_confirmation_method_contract,
    method_contract_hash as confirmation_method_contract_hash,
)


RESIDUAL_FREQUENCY = 0.282117145741729


def _write_full_sector(path, *, dataset_id, sector):
    origin = float(sector * 100.0)
    times = [25.0 * index / 255.0 for index in range(256)]
    family_frequency = 1.0 / FAMILY_PERIOD_DAYS
    flux = [
        0.7 * math.sin(2.0 * math.pi * family_frequency * (origin + time))
        + 0.3 * math.cos(
            2.0 * math.pi * 2.0 * family_frequency * (origin + time)
        )
        + 0.25 * math.sin(
            2.0 * math.pi * RESIDUAL_FREQUENCY * (origin + time) + 0.2
        )
        for time in times
    ]
    path.write_text(json.dumps({
        "id": dataset_id,
        "source": {
            "ticID": TIC_ID,
            "sector": sector,
            "originalTimeOriginDays": origin,
        },
        "metadata": {"ticID": TIC_ID, "sector": sector},
        "times": times,
        "flux": flux,
    }), encoding="utf-8")


def _confirmed_result(contract):
    paths = list(contract["evidenceBoundary"]["frozenWindowDatasetPaths"])
    frequencies = [RESIDUAL_FREQUENCY for _ in INDEPENDENT_SECTORS]
    folds = [{
        "trainingSectors": [
            PRIMARY_SECTOR,
            *(item for item in INDEPENDENT_SECTORS if item != sector),
        ],
        "heldOutSector": sector,
        "learnedCoherentFrequencyCyclesPerDay": RESIDUAL_FREQUENCY,
        "support": "COHERENT",
        "predictiveBIC": {"H": 130.0, "S": 90.0, "N": 140.0},
        "failureOrInsufficiencyReasons": [],
    } for sector in INDEPENDENT_SECTORS]
    return {
        "version": "openstar.tess-v20.8-long-baseline-confirmation.v1",
        "methodContractID": CONFIRMATION_METHOD_CONTRACT_ID,
        "methodContractHash": confirmation_method_contract_hash(contract),
        "methodContract": copy.deepcopy(contract),
        "leaveOneIndependentSectorOut": True,
        "perSectorEvidence": folds,
        "longBaselineDays": 1000.0,
        "longBaselineFrequencyResolutionCyclesPerDay": 0.001,
        "classification": COHERENT,
        "recommendedNextTest": "MODE_IDENTIFICATION_OR_PULSATION_MODELING",
        "aggregateDecision": {
            "predictiveBIC": {"H": 390.0, "S": 270.0, "N": 420.0},
            "bicImprovementHarmonicOverNull": 30.0,
            "bicImprovementCoherentOverNull": 150.0,
            "bicImprovementCoherentOverHarmonic": 120.0,
            "sufficientHeldOutSectorCount": 3,
            "coherentSupportingSectorCount": 3,
            "harmonicSupportingSectorCount": 0,
        },
        "frequencyStability": {
            "learnedFrequenciesCyclesPerDay": frequencies,
            "medianFrequencyCyclesPerDay": RESIDUAL_FREQUENCY,
            "rangeCyclesPerDay": 0.0,
            "maximumAllowedRangeCyclesPerDay": 0.001,
            "stableWithinLongBaselineResolution": True,
        },
        "failureOrInsufficiencyReasons": [],
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "automaticDiscoveryClaim": False,
        "dataReuse": {
            "frozenWindowDatasetPaths": paths,
            "downloadPerformed": False,
            "originalSectorFluxRead": False,
        },
    }


def _mode_result(classification):
    recommended = {
        "INDEPENDENT_STABLE_MODE": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
        "HIGHER_ORDER_HARMONIC_STRUCTURE": "DYNAMIC_HARMONIC_MODELING",
        "NO_COMPELLING_RESIDUAL_MODE": "BINARY_ROTATION_EXTERNAL_EVIDENCE",
        "AMBIGUOUS_HARMONIC_OR_MODE": (
            "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
        ),
    }[classification]
    survived = classification == "INDEPENDENT_STABLE_MODE"
    return {
        "classification": classification,
        "establishedPeriodFamily": {
            "referencePeriodDays": FAMILY_PERIOD_DAYS,
        },
        "residualCandidate": {
            "refinedPeriodDays": 1.0 / RESIDUAL_FREQUENCY,
            "refinedFrequencyCyclesPerDay": RESIDUAL_FREQUENCY,
        },
        "harmonicRelation": {
            "testedOrder": 4,
            "commensurateWithinResolution": classification
            == "HIGHER_ORDER_HARMONIC_STRUCTURE",
        },
        "modelComparison": {
            "criterion": "BIC",
            "conservativeThreshold": 10.0,
        },
        "independentSectorSupport": {
            "sectors": list(INDEPENDENT_SECTORS),
            "count": 3,
            "requiredCount": 3,
            "sufficient": True,
        },
        "independentModeEvidenceSurvived": survived,
        "modeCandidate": ({
            "periodDays": 1.0 / RESIDUAL_FREQUENCY,
            "frequencyCyclesPerDay": RESIDUAL_FREQUENCY,
            "supportingSectors": list(INDEPENDENT_SECTORS),
        } if survived else None),
        "physicalMechanismResolved": False,
        "recommendedNextTest": recommended,
        "dataReuse": {"frozenDatasetPaths": [], "downloadPerformed": False},
        "frequencyRefinement": {"execution": "PYTHON_SERVER"},
    }


class ConfirmedCoherentModeFixture(unittest.TestCase):
    def _history(self, root, *, status="COMPLETE"):
        root = Path(root)
        store = InvestigationStore(root / "investigations")
        preparation, interpretation, summary = _evidence(root)
        confirmation_contract = build_confirmation_method_contract(
            preparation=preparation,
            interpretation=interpretation,
            summary=summary,
        )
        confirmation = _confirmed_result(confirmation_contract)
        window_specs = build_confirmation_dataset_specs(
            expected_tic_id=TIC_ID, preparation=preparation
        )

        full_paths = {}
        for sector in (PRIMARY_SECTOR, *INDEPENDENT_SECTORS):
            path = root / f"sector-{sector}-full.json"
            _write_full_sector(
                path, dataset_id=f"sector-{sector}-full", sector=sector
            )
            full_paths[sector] = path
        prepared = {
            "datasetID": "sector-1-full",
            "datasetPath": str(full_paths[PRIMARY_SECTOR].resolve()),
            "ticID": TIC_ID,
            "targetName": "Synthetic confirmed coherent residual",
            "sector": PRIMARY_SECTOR,
        }
        independent = {
            "preparedSectors": [{
                "datasetID": f"sector-{sector}-full",
                "datasetPath": str(full_paths[sector].resolve()),
                "sector": sector,
            } for sector in INDEPENDENT_SECTORS]
        }
        morphology = {
            "physicalCycleResolved": False,
            "resolvedPhysicalPeriodDays": None,
            "continuationEvidence": {
                "timeFrequencyEvolutionWarranted": True,
                "analysisReferencePeriodDays": FAMILY_PERIOD_DAYS,
            },
        }
        run_result = {"datasets": []}
        preparation_provenance = StageProvenance(
            software_id="fixture", software_version="1",
            input_hashes={"morphology": sha256_json(morphology)},
        )
        interpretation_provenance = StageProvenance(
            software_id="fixture", software_version="1",
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run_result),
            },
        )
        summary_provenance = StageProvenance(
            software_id="fixture", software_version="1",
            input_hashes={
                "morphology": sha256_json(morphology),
                "timeFrequencyInterpretation": sha256_json(interpretation),
            },
        )
        confirmation_hashes = {
            "methodContract": confirmation["methodContractHash"],
            "morphology": sha256_json(morphology),
            "timeFrequencyPreparation": sha256_json(preparation),
            "timeFrequencyProjectResult": sha256_json(run_result),
            "timeFrequencyInterpretation": sha256_json(interpretation),
            "timeFrequencySummary": sha256_json(summary),
        }
        confirmation_hashes.update({
            "frozenWindowDataset:"
            f"{spec['role']}:{spec['sector']}:{spec['windowIndex']}":
            sha256_file(spec["datasetPath"])
            for spec in window_specs
        })
        confirmation_provenance = StageProvenance(
            software_id="fixture", software_version="1",
            input_hashes=confirmation_hashes,
        )

        claim = {
            "claim": "CANDIDATE_PERIOD",
            "rationale": ["Frozen pre-continuation claim."],
        }
        stages = (
            InvestigationStage(
                "001-prepare-target", "openstar.tess.prepare-target",
                "COMPLETE", None, {}, result=prepared,
            ),
            InvestigationStage(
                "004-hypotheses", "openstar.tess.hypotheses", "COMPLETE",
                "001-prepare-target", {}, result={
                    "rawCandidatePeriodDays": FAMILY_PERIOD_DAYS / 2.0,
                    "observedPeriodDays": FAMILY_PERIOD_DAYS / 2.0,
                    "possibleDoubleCycleDays": FAMILY_PERIOD_DAYS,
                },
            ),
            InvestigationStage(
                "005-planner", "openstar.tess.planner", "COMPLETE",
                "004-hypotheses", {}, result={"claimDecision": claim},
            ),
            InvestigationStage(
                "006-prepare-independent-sectors",
                "openstar.tess.independent.prepare", "COMPLETE",
                "005-planner", {}, result=independent,
            ),
            InvestigationStage(
                "012-characterize-variability",
                "openstar.tess.morphology.analyze", "COMPLETE",
                "006-prepare-independent-sectors", {}, result=morphology,
            ),
            InvestigationStage(
                "013-prepare-time-frequency",
                "openstar.tess.time-frequency.prepare", "COMPLETE",
                "012-characterize-variability", {}, result=preparation,
                provenance=preparation_provenance,
            ),
            InvestigationStage(
                "014-run-time-frequency", "openstar.tess.time-frequency.run",
                "COMPLETE", "013-prepare-time-frequency", {},
                result=run_result,
            ),
            InvestigationStage(
                "015-interpret-time-frequency",
                "openstar.tess.time-frequency.interpret", "COMPLETE",
                "014-run-time-frequency", {}, result=interpretation,
                provenance=interpretation_provenance,
            ),
            InvestigationStage(
                "016-summarize-time-frequency",
                "openstar.tess.time-frequency.summarize", "COMPLETE",
                "015-interpret-time-frequency", {}, result=summary,
                provenance=summary_provenance,
            ),
            InvestigationStage(
                "017-finalize", "openstar.tess.finalize", "COMPLETE",
                "016-summarize-time-frequency", {"outputSuffix": "v20.8"},
                result={
                    "claim": claim,
                    "timeFrequencyEvolution": summary,
                    "recommendedNextTest": (
                        "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
                    ),
                }, stop=True,
            ),
            InvestigationStage(
                "018-long-baseline-time-frequency-confirmation",
                CONFIRMATION_HANDLER_ID, "COMPLETE",
                "016-summarize-time-frequency", {}, result=confirmation,
                provenance=confirmation_provenance,
            ),
            InvestigationStage(
                "019-finalize", "openstar.tess.finalize", "COMPLETE",
                "018-long-baseline-time-frequency-confirmation", {
                    "outputSuffix": (
                        "v20.8.1-long-baseline-time-frequency-confirmation"
                    )
                }, result={
                    "claim": claim,
                    "timeFrequencyEvolution": summary,
                    "longBaselineTimeFrequencyConfirmation": confirmation,
                    "recommendedNextTest": (
                        "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
                    ),
                }, stop=True,
            ),
        )
        investigation = store.create(
            "confirmed-coherent-v20-8-1", WORKFLOW_ID, WORKFLOW_VERSION,
            metadata={"controlState": {
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            }},
        )
        investigation = replace(
            investigation, status=status, stages=stages
        )
        store.save(investigation)
        specs = [{
            "datasetID": prepared["datasetID"],
            "datasetPath": prepared["datasetPath"],
            "ticID": TIC_ID,
            "sector": PRIMARY_SECTOR,
            "role": "PRIMARY",
        }, *({
            "datasetID": item["datasetID"],
            "datasetPath": item["datasetPath"],
            "ticID": TIC_ID,
            "sector": item["sector"],
            "role": "INDEPENDENT",
        } for item in independent["preparedSectors"])]
        contract = build_confirmed_coherent_mode_method_contract(
            confirmation=confirmation, dataset_specs=specs
        )
        return store, investigation, contract, specs


class ConfirmedCoherentModeContractTests(ConfirmedCoherentModeFixture):
    def test_method_contract_hash_is_deterministic_before_flux(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation, contract, specs = self._history(temporary)
            with mock.patch(
                "workflows.tess.tess_mode_identification._load_frozen_dataset"
            ) as loader:
                rebuilt = build_confirmed_coherent_mode_method_contract(
                    confirmation=copy.deepcopy(investigation.stages[-2].result),
                    dataset_specs=copy.deepcopy(specs),
                )
            loader.assert_not_called()
        self.assertEqual(
            CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID,
            contract["methodContractID"],
        )
        self.assertEqual(
            confirmed_coherent_mode_method_contract_hash(contract),
            confirmed_coherent_mode_method_contract_hash(copy.deepcopy(contract)),
        )
        self.assertEqual(
            contract["modelComparison"]["minimumImprovement"], 10.0
        )
        self.assertEqual(contract, rebuilt)

    def test_corrupted_or_mismatched_full_sector_data_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, contract, specs = self._history(temporary)
            path = Path(specs[1]["datasetPath"])
            dataset = json.loads(path.read_text(encoding="utf-8"))
            dataset["source"]["sector"] = 999
            path.write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                validate_confirmed_coherent_mode_dataset_lineage(
                    method_contract=contract, dataset_specs=specs
                )

    def test_insufficient_predictive_sector_support_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation, _, _ = self._history(temporary)
            confirmation = copy.deepcopy(investigation.stages[-2].result)
            confirmation["methodContract"]["evidenceBoundary"][
                "acceptedIndependentSectors"
            ] = list(INDEPENDENT_SECTORS[:2])
            confirmation["methodContractHash"] = (
                confirmation_method_contract_hash(
                    confirmation["methodContract"]
                )
            )
            confirmation["perSectorEvidence"] = confirmation[
                "perSectorEvidence"
            ][:2]
            confirmation["frequencyStability"][
                "learnedFrequenciesCyclesPerDay"
            ] = [RESIDUAL_FREQUENCY, RESIDUAL_FREQUENCY]
            confirmation["aggregateDecision"].update({
                "sufficientHeldOutSectorCount": 2,
                "coherentSupportingSectorCount": 2,
            })
            with self.assertRaisesRegex(RuntimeError, "exact confirmed"):
                validate_v20_8_confirmed_coherent_residual(confirmation)

    def test_classification_routing_is_conservative(self):
        expected = {
            "INDEPENDENT_STABLE_MODE": (
                "RESIDUAL_MODE_PIXEL_LOCALIZATION",
                "PHOTOMETRIC_MODE_SUPPORTED_PULSATION_MECHANISM_UNRESOLVED",
            ),
            "HIGHER_ORDER_HARMONIC_STRUCTURE": (
                "DYNAMIC_HARMONIC_MODELING",
                "PULSATION_NOT_ESTABLISHED_HARMONIC_STRUCTURE_SUPPORTED",
            ),
            "NO_COMPELLING_RESIDUAL_MODE": (
                "HUMAN_SCIENTIFIC_REVIEW",
                "PULSATION_OR_MODE_IDENTIFICATION_INCONCLUSIVE",
            ),
            "AMBIGUOUS_HARMONIC_OR_MODE": (
                "HUMAN_SCIENTIFIC_REVIEW",
                "PULSATION_OR_MODE_IDENTIFICATION_INCONCLUSIVE",
            ),
        }
        for classification, (recommendation, interpretation) in expected.items():
            with self.subTest(classification=classification), \
                    tempfile.TemporaryDirectory() as temporary:
                _, _, contract, specs = self._history(temporary)
                with mock.patch(
                    "workflows.tess.tess_mode_identification.identify_residual_mode",
                    return_value=_mode_result(classification),
                ):
                    result = analyze_confirmed_coherent_residual_mode(
                        method_contract=contract, dataset_specs=specs
                    )
                self.assertEqual(classification, result["classification"])
                self.assertEqual(recommendation, result["recommendedNextTest"])
                self.assertEqual(interpretation, result["pulsationInterpretation"])
                self.assertFalse(result["physicalMechanismResolved"])
                self.assertFalse(result["pulsationMechanismResolved"])
                self.assertFalse(result["claimLevelChanged"])
                self.assertFalse(result["automaticDiscoveryClaim"])


class ConfirmedCoherentModeContinuationTests(ConfirmedCoherentModeFixture):
    def test_exact_manual_boundary_and_append_only_automatic_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation, _, _ = self._history(temporary)
            _can_continue_confirmed_coherent_mode_identification(investigation)
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
                repaired = (
                    _repair_v20_8_confirmed_coherent_mode_identification_terminal(
                        store, investigation,
                        investigation.metadata["controlState"],
                    )
                )
                repeated = (
                    _repair_v20_8_confirmed_coherent_mode_identification_terminal(
                        store, repaired, repaired.metadata["controlState"],
                    )
                )
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(immutable, tuple(
            json.dumps(asdict(stage), sort_keys=True)
            for stage in repaired.stages
        ))
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("020-mode-identification", selected["id"])
        self.assertEqual(
            "openstar.tess.mode-identification.analyze",
            selected["handler_id"],
        )
        self.assertEqual(
            {"evidenceLineage": V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE},
            selected["parameters"],
        )
        self.assertEqual(
            "018-long-baseline-time-frequency-confirmation",
            selected["triggered_by_stage_id"],
        )
        self.assertIsNone(repeated)

    def test_manual_gate_rejects_running_nonterminal_and_existing_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation, _, _ = self._history(temporary, status="RUNNING")
            with self.assertRaisesRegex(RuntimeError, "terminal investigation"):
                _can_continue_confirmed_coherent_mode_identification(investigation)
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation, _, _ = self._history(temporary)
            running = InvestigationStage(
                "020-running", "openstar.tess.other", "RUNNING",
                "019-finalize", {},
            )
            with self.assertRaisesRegex(RuntimeError, "RUNNING stage"):
                _can_continue_confirmed_coherent_mode_identification(replace(
                    investigation,
                    stages=investigation.stages + (running,),
                ))
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation, _, _ = self._history(temporary)
            existing = InvestigationStage(
                "020-mode-identification",
                "openstar.tess.mode-identification.analyze", "COMPLETE",
                "018-long-baseline-time-frequency-confirmation", {},
                result={},
            )
            with self.assertRaisesRegex(RuntimeError, "already contains"):
                _can_continue_confirmed_coherent_mode_identification(replace(
                    investigation,
                    stages=investigation.stages + (existing,),
                ))

    def test_rejects_altered_confirmation_and_wrong_recommendation(self):
        for change in ("classification", "recommendedNextTest"):
            with self.subTest(change=change), \
                    tempfile.TemporaryDirectory() as temporary:
                _, investigation, _, _ = self._history(temporary)
                stages = list(investigation.stages)
                result = copy.deepcopy(stages[-2].result)
                result[change] = "WRONG"
                stages[-2] = replace(stages[-2], result=result)
                final = copy.deepcopy(stages[-1].result)
                final["longBaselineTimeFrequencyConfirmation"] = result
                if change == "recommendedNextTest":
                    final["recommendedNextTest"] = "WRONG"
                stages[-1] = replace(stages[-1], result=final)
                with self.assertRaises(RuntimeError):
                    _can_continue_confirmed_coherent_mode_identification(
                        replace(investigation, stages=tuple(stages))
                    )

    def test_handler_persists_v20_8_2_without_claim_or_mechanism_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation, _, _ = self._history(temporary)
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
                    "020-mode-identification",
                    "openstar.tess.mode-identification.analyze",
                    {"evidenceLineage": (
                        V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE
                    )},
                    "018-long-baseline-time-frequency-confirmation",
                ),
                software_id="integration", software_version="1",
            )
            self.assertEqual(frozen_stages, completed.stages[:-1])
            self.assertEqual("openstar.tess.finalize", finalize.handler_id)
            self.assertEqual(
                "v20.8.2-confirmed-coherent-mode-identification",
                finalize.parameters["outputSuffix"],
            )
            final, next_request = engine.run_stage(
                completed, finalize,
                software_id="integration", software_version="1",
            )
            conclusion = final.stages[-1].result
            persisted = conclusion["modeIdentification"]
            report = Path(conclusion["reportPath"]).read_text(encoding="utf-8")
            conclusion_file = json.loads(
                Path(conclusion["conclusionPath"]).read_text(encoding="utf-8")
            )
        self.assertIsNone(next_request)
        self.assertEqual("CANDIDATE_PERIOD", conclusion["claim"]["claim"])
        self.assertFalse(persisted["physicalMechanismResolved"])
        self.assertFalse(persisted["pulsationMechanismResolved"])
        self.assertFalse(persisted["claimLevelChanged"])
        self.assertFalse(persisted["automaticDiscoveryClaim"])
        self.assertEqual(
            CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID,
            persisted["methodContractID"],
        )
        self.assertEqual(persisted, conclusion_file["modeIdentification"])
        self.assertIn("Stable residual mode identification", report)
        self.assertIn("Pulsation mechanism resolved: False", report)
        coordinator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
