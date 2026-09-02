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
from run_tess_investigation import _can_continue_transient_mode_validation
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    _repair_transient_mode_validation_terminal,
)
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_transient_mode_validation import (
    HANDLER_ID,
    INCONCLUSIVE,
    METHOD_CONTRACT_ID,
    RECURRENT,
    TRANSIENT_HARMONIC,
    TRANSIENT_INDEPENDENT,
    analyze_transient_mode_validation,
    build_dataset_specs,
    build_method_contract,
    classify_transient_validation,
    method_contract_hash,
    validate_frozen_dataset_lineage,
)


TIC_ID = 52244725
PRIMARY_SECTOR = 1
INDEPENDENT_SECTORS = (2, 28, 68, 69)
PHYSICAL_PERIOD_DAYS = 13.259005075877733
FAMILY_FREQUENCY = 1.0 / PHYSICAL_PERIOD_DAYS
TRANSIENT_FREQUENCY = 1.0 / 3.3201443206629397
HARMONIC_FREQUENCY = 4.0 * FAMILY_FREQUENCY


def _write_window(
    path,
    *,
    dataset_id,
    sector,
    role,
    window_index,
    frequency=None,
    phase=0.35,
):
    origin = float(sector * 1000.0)
    window_start = float((window_index - 1) * 7.0)
    local_times = [10.0 * index / 127.0 for index in range(128)]
    absolute_times = [
        origin + window_start + value for value in local_times
    ]
    if frequency is None:
        flux = [0.02 * math.sin(index * 0.731) for index in range(128)]
    else:
        flux = [
            0.7 * math.sin(2.0 * math.pi * frequency * time + phase)
            for time in absolute_times
        ]
    source = {
        "ticID": TIC_ID,
        "sector": sector,
        "timeFrequencyWindowIndex": window_index,
        "windowStartDatasetDays": window_start,
        "windowCenterDatasetDays": window_start + 5.0,
    }
    metadata = {}
    if role == "primary-time-frequency-window":
        metadata = {
            "ticID": TIC_ID,
            "sector": sector,
            "originalTimeOriginDays": origin,
        }
        source["absoluteWindowCenterDays"] = None
    else:
        source["originalTimeOriginDays"] = origin + window_start
        source["absoluteWindowCenterDays"] = origin + window_start + 5.0
    path.write_text(json.dumps({
        "id": dataset_id,
        "source": source,
        "metadata": metadata,
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
    detection_keys = {(2, 1), (28, 1)}
    for sector, role in (
        (PRIMARY_SECTOR, "primary-time-frequency-window"),
        *((value, "independent-time-frequency-window") for value in sectors),
    ):
        for window_index in (1, 2, 3):
            dataset_id = f"sector-{sector}-window-{window_index}"
            path = Path(root) / f"{dataset_id}.json"
            signal_frequency = (
                TRANSIENT_FREQUENCY
                if (sector, window_index) in detection_keys else None
            )
            _write_window(
                path,
                dataset_id=dataset_id,
                sector=sector,
                role=role,
                window_index=window_index,
                frequency=signal_frequency,
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
            accepted = (sector, window_index) in detection_keys
            candidate_frequency = (
                TRANSIENT_FREQUENCY
                if accepted else 0.22 + 0.001 * window_index
            )
            # Preserve an accepted, unrelated control-sector feature so the
            # summary has the same three accepted independent sectors as the
            # persisted TIC 52244725 boundary.
            if (sector, window_index) == (69, 2):
                accepted = True
                candidate_frequency = 0.245
            window_results.append({
                "datasetID": dataset_id,
                "sectorKey": str(sector),
                "sector": sector,
                "role": role,
                "windowIndex": window_index,
                "candidateFrequency": candidate_frequency,
                "candidatePeriodDays": 1.0 / candidate_frequency,
                "candidatePeakProminenceRatio": 8.0 if accepted else 1.1,
                "acceptedTimeFrequencyFeature": accepted,
                "nearEstablishedFamily": False,
            })

    accepted = [
        item for item in window_results
        if item["acceptedTimeFrequencyFeature"]
    ]
    accepted_independent = [
        item for item in accepted
        if item["role"] == "independent-time-frequency-window"
    ]
    members = [
        {
            "sector": sector,
            "windowIndex": 1,
            "absoluteWindowCenterDays": sector * 1000.0 + 5.0,
            "frequency": TRANSIENT_FREQUENCY,
            "periodDays": 1.0 / TRANSIENT_FREQUENCY,
            "prominence": 8.0,
            "nearEstablishedFamily": False,
        }
        for sector in (2, 28)
        if sector in sectors
    ]
    preparation = {
        "available": True,
        "physicalPeriodDays": PHYSICAL_PERIOD_DAYS,
        "physicalFrequency": FAMILY_FREQUENCY,
        "firstHarmonicFrequency": 2.0 * FAMILY_FREQUENCY,
        "subtractedHarmonicOrders": [1, 2],
        "preparedWindows": prepared_windows,
    }
    interpretation = {
        "physicalFrequency": FAMILY_FREQUENCY,
        "firstHarmonicFrequency": 2.0 * FAMILY_FREQUENCY,
        "windowResults": copy.deepcopy(window_results),
    }
    summary = {
        "classification": "TRANSIENT_RESIDUAL_MODE",
        "physicalPeriodDays": PHYSICAL_PERIOD_DAYS,
        "physicalFrequency": FAMILY_FREQUENCY,
        "windowCount": len(window_results),
        "acceptedFeatureCount": len(accepted),
        "acceptedIndependentFeatureCount": len(accepted_independent),
        "acceptedIndependentSectors": sorted({
            item["sector"] for item in accepted_independent
        }),
        "windowResults": copy.deepcopy(window_results),
        "residualEvolution": {
            "classification": "TRANSIENT_RESIDUAL_MODE",
            "bestCluster": {
                "medianFrequency": TRANSIENT_FREQUENCY,
                "medianPeriodDays": 1.0 / TRANSIENT_FREQUENCY,
                "windowCount": len(members),
                "independentSectors": sorted(item["sector"] for item in members),
                "independentSectorCount": len(members),
                "relativeSpan": 0.0,
                "nearEstablishedFamilyFraction": 0.0,
                "members": members,
            },
        },
        "familyEvolution": {
            "classification": "FAMILY_AMPLITUDE_AND_PHASE_EVOLUTION",
        },
        "periodReference": {
            "periodDays": PHYSICAL_PERIOD_DAYS,
            "kind": "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD",
            "physicalCycleResolved": True,
        },
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": "TRANSIENT_MODE_VALIDATION",
    }
    morphology = {
        "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED",
        "physicalCycleResolved": True,
        "resolvedPhysicalPeriodDays": PHYSICAL_PERIOD_DAYS,
    }
    binary = {
        "independentEvidence": {
            "classification": "ECLIPSE_LIKE_EVENT_UNRESOLVED",
        },
        "linearEphemeris": {"coherent": False},
        "physicalMechanismResolved": False,
    }
    return morphology, binary, preparation, interpretation, summary


def _fold(support, h_bic, t_bic, n_bic, sector):
    return {
        "heldOutSector": sector,
        "support": support,
        "predictiveBIC": {"H": h_bic, "T": t_bic, "N": n_bic},
        "failureOrInsufficiencyReasons": [],
    }


def _control(support, sector, window_index=2):
    return {
        "role": "INDEPENDENT_WINDOW",
        "sector": sector,
        "windowIndex": window_index,
        "support": support,
        "failureOrInsufficiencyReasons": [],
    }


class TransientModeContractTests(unittest.TestCase):
    def test_method_contract_hash_is_deterministic_before_flux(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = _evidence(temporary)
            with mock.patch(
                "workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation._load_frozen_dataset"
            ) as loader:
                first = build_method_contract(
                    morphology=values[0],
                    binary_confirmation=values[1],
                    preparation=values[2],
                    interpretation=values[3],
                    summary=values[4],
                )
                second = build_method_contract(
                    morphology=json.loads(json.dumps(values[0], sort_keys=True)),
                    binary_confirmation=json.loads(json.dumps(values[1], sort_keys=True)),
                    preparation=json.loads(json.dumps(values[2], sort_keys=True)),
                    interpretation=json.loads(json.dumps(values[3], sort_keys=True)),
                    summary=json.loads(json.dumps(values[4], sort_keys=True)),
                )
            loader.assert_not_called()
        self.assertEqual(METHOD_CONTRACT_ID, first["methodContractID"])
        self.assertEqual(first, second)
        self.assertEqual(method_contract_hash(first), method_contract_hash(second))
        self.assertFalse(first["crossValidation"]["heldOutFrequencySelection"])
        self.assertFalse(first["crossValidation"]["heldOutPhaseSelection"])
        self.assertFalse(first["crossValidation"]["controlWindowsUsedForSelection"])

    def test_leave_one_detection_sector_out_prevents_flux_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = _evidence(temporary)
            contract = build_method_contract(
                morphology=values[0],
                binary_confirmation=values[1],
                preparation=values[2],
                interpretation=values[3],
                summary=values[4],
            )
            specs = build_dataset_specs(
                expected_tic_id=TIC_ID, preparation=values[2]
            )
            first = analyze_transient_mode_validation(
                method_contract=contract, dataset_specs=specs
            )
            held_out = next(
                spec for spec in specs
                if spec["sector"] == 2 and spec["windowIndex"] == 1
            )
            path = Path(held_out["datasetPath"])
            dataset = json.loads(path.read_text(encoding="utf-8"))
            dataset["flux"] = [
                50.0 * math.sin(index * 0.317)
                for index in range(len(dataset["flux"]))
            ]
            path.write_text(json.dumps(dataset), encoding="utf-8")
            second = analyze_transient_mode_validation(
                method_contract=contract, dataset_specs=specs
            )
        first_fold = next(
            item for item in first["perDetectionSectorEvidence"]
            if item["heldOutSector"] == 2
        )
        second_fold = next(
            item for item in second["perDetectionSectorEvidence"]
            if item["heldOutSector"] == 2
        )
        self.assertEqual(
            first_fold["learnedTransientFrequencyCyclesPerDay"],
            second_fold["learnedTransientFrequencyCyclesPerDay"],
        )

    def test_corrupted_or_mismatched_windows_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = _evidence(temporary)
            contract = build_method_contract(
                morphology=values[0],
                binary_confirmation=values[1],
                preparation=values[2],
                interpretation=values[3],
                summary=values[4],
            )
            specs = build_dataset_specs(
                expected_tic_id=TIC_ID, preparation=values[2]
            )
            path = Path(specs[4]["datasetPath"])
            dataset = json.loads(path.read_text(encoding="utf-8"))
            dataset["source"]["sector"] = 999
            path.write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                validate_frozen_dataset_lineage(
                    method_contract=contract, dataset_specs=specs
                )

    def test_altered_boundary_wrong_recommendation_and_controls_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = list(_evidence(temporary))
            for change in ("classification", "recommendation", "binary"):
                changed = copy.deepcopy(values)
                if change == "classification":
                    changed[4]["classification"] = "OTHER"
                elif change == "recommendation":
                    changed[4]["recommendedNextTest"] = "OTHER"
                else:
                    changed[1]["linearEphemeris"]["coherent"] = True
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    build_method_contract(
                        morphology=changed[0],
                        binary_confirmation=changed[1],
                        preparation=changed[2],
                        interpretation=changed[3],
                        summary=changed[4],
                    )
        with tempfile.TemporaryDirectory() as temporary:
            values = _evidence(temporary, sectors=(2, 28))
            with self.assertRaises(RuntimeError):
                build_method_contract(
                    morphology=values[0],
                    binary_confirmation=values[1],
                    preparation=values[2],
                    interpretation=values[3],
                    summary=values[4],
                )

    def test_all_classifications_and_threshold_edge(self):
        independent = classify_transient_validation([
            _fold("TRANSIENT_FREQUENCY", 100, 80, 100, 2),
            _fold("TRANSIENT_FREQUENCY", 100, 80, 100, 28),
        ], [_control("NEITHER", 68), _control("NEITHER", 69)])
        harmonic = classify_transient_validation([
            _fold("HARMONIC", 80, 100, 100, 2),
            _fold("HARMONIC", 80, 100, 100, 28),
        ], [_control("NEITHER", 68), _control("NEITHER", 69)])
        recurrent = classify_transient_validation([
            _fold("TRANSIENT_FREQUENCY", 100, 80, 100, 2),
            _fold("TRANSIENT_FREQUENCY", 100, 80, 100, 28),
        ], [
            _control("TRANSIENT_FREQUENCY", 68, 1),
            _control("TRANSIENT_FREQUENCY", 68, 2),
            _control("STRUCTURED_UNRESOLVED", 69, 1),
        ])
        inconclusive = classify_transient_validation([
            _fold("STRUCTURED_UNRESOLVED", 90, 90, 100, 2),
            _fold("NEITHER", 100, 100, 100, 28),
        ], [_control("NEITHER", 68)])
        threshold = classify_transient_validation([
            _fold("TRANSIENT_FREQUENCY", 100, 90, 100, 2),
            _fold("TRANSIENT_FREQUENCY", 100, 100, 100, 28),
        ], [_control("NEITHER", 68)])
        self.assertEqual(TRANSIENT_INDEPENDENT, independent["classification"])
        self.assertEqual(TRANSIENT_HARMONIC, harmonic["classification"])
        self.assertEqual(RECURRENT, recurrent["classification"])
        self.assertEqual(INCONCLUSIVE, inconclusive["classification"])
        self.assertEqual(TRANSIENT_INDEPENDENT, threshold["classification"])


class TransientModeContinuationTests(unittest.TestCase):
    def _history(self, root, *, status="COMPLETE"):
        root = Path(root)
        store = InvestigationStore(root / "investigations")
        morphology, binary, preparation, interpretation, summary = _evidence(root)
        run_result = {"datasets": []}
        prepared = {
            "datasetID": "primary-sector-1",
            "datasetPath": preparation["preparedWindows"][0]["datasetPath"],
            "ticID": TIC_ID,
            "targetName": "Synthetic transient boundary",
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
                    "rawCandidatePeriodDays": PHYSICAL_PERIOD_DAYS / 2.0,
                    "observedPeriodDays": PHYSICAL_PERIOD_DAYS / 2.0,
                    "possibleDoubleCycleDays": PHYSICAL_PERIOD_DAYS,
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
                "005-binary", "openstar.tess.binary-confirmation.analyze",
                "COMPLETE", "004-morphology", {}, result=binary,
            ),
            InvestigationStage(
                "006-prepare-time-frequency",
                "openstar.tess.time-frequency.prepare", "COMPLETE",
                "005-binary", {}, result=preparation,
                provenance=StageProvenance(
                    software_id="fixture", software_version="1",
                    input_hashes={"morphology": sha256_json(morphology)},
                ),
            ),
            InvestigationStage(
                "007-run-time-frequency", "openstar.tess.time-frequency.run",
                "COMPLETE", "006-prepare-time-frequency", {}, result=run_result,
            ),
            InvestigationStage(
                "008-interpret-time-frequency",
                "openstar.tess.time-frequency.interpret", "COMPLETE",
                "007-run-time-frequency", {}, result=interpretation,
                provenance=StageProvenance(
                    software_id="fixture", software_version="1",
                    input_hashes={
                        "preparation": sha256_json(preparation),
                        "projectResult": sha256_json(run_result),
                    },
                ),
            ),
            InvestigationStage(
                "009-summarize-time-frequency",
                "openstar.tess.time-frequency.summarize", "COMPLETE",
                "008-interpret-time-frequency", {}, result=summary,
                provenance=StageProvenance(
                    software_id="fixture", software_version="1",
                    input_hashes={
                        "morphology": sha256_json(morphology),
                        "timeFrequencyInterpretation": sha256_json(interpretation),
                    },
                ),
            ),
        ]
        conclusion = {
            "claim": {
                "claim": "CANDIDATE_PERIOD",
                "rationale": ["Frozen pre-continuation claim."],
            },
            "binaryConfirmation": binary,
            "timeFrequencyEvolution": summary,
            "recommendedNextTest": "TRANSIENT_MODE_VALIDATION",
        }
        stages.append(InvestigationStage(
            "010-finalize", "openstar.tess.finalize", "COMPLETE",
            "009-summarize-time-frequency", {"outputSuffix": "v20.8"},
            result=conclusion, stop=True,
        ))
        investigation = store.create(
            "transient-v20-8", WORKFLOW_ID, WORKFLOW_VERSION,
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

    def test_exact_manual_boundary_and_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary)
            _can_continue_transient_mode_validation(investigation)
            running = InvestigationStage(
                "011-running", "openstar.tess.other", "RUNNING",
                "010-finalize", {},
            )
            with self.assertRaisesRegex(RuntimeError, "RUNNING stage"):
                _can_continue_transient_mode_validation(replace(
                    investigation,
                    stages=investigation.stages + (running,),
                ))
            existing = InvestigationStage(
                "011-transient", HANDLER_ID, "COMPLETE",
                "009-summarize-time-frequency", {}, result={},
            )
            with self.assertRaisesRegex(RuntimeError, "already contains"):
                _can_continue_transient_mode_validation(replace(
                    investigation,
                    stages=investigation.stages + (existing,),
                ))
        with tempfile.TemporaryDirectory() as temporary:
            _, investigation = self._history(temporary, status="FAILED")
            with self.assertRaisesRegex(RuntimeError, "terminal investigation"):
                _can_continue_transient_mode_validation(investigation)

    def test_automatic_repair_is_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._history(temporary)
            immutable = tuple(
                json.dumps(asdict(stage), sort_keys=True)
                for stage in investigation.stages
            )
            with mock.patch.object(
                store, "verified_terminal_stage_ledger_hash", return_value=True,
            ), mock.patch(
                "workflows.tess.tess_autonomy._verified_stage_json",
                return_value=True,
            ):
                repaired = _repair_transient_mode_validation_terminal(
                    store,
                    investigation,
                    investigation.metadata["controlState"],
                )
                repeated = _repair_transient_mode_validation_terminal(
                    store,
                    repaired,
                    repaired.metadata["controlState"],
                )
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(immutable, tuple(
            json.dumps(asdict(stage), sort_keys=True)
            for stage in repaired.stages
        ))
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual("011-transient-mode-validation", selected["id"])
        self.assertEqual(HANDLER_ID, selected["handler_id"])
        self.assertEqual(
            "009-summarize-time-frequency", selected["triggered_by_stage_id"]
        )
        self.assertIsNone(repeated)

    def test_handler_persists_conclusion_without_claim_or_mechanism_upgrade(self):
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
                    "011-transient-mode-validation",
                    HANDLER_ID,
                    {},
                    "009-summarize-time-frequency",
                ),
                software_id="integration",
                software_version="1",
            )
            self.assertEqual(frozen_stages, completed.stages[:-1])
            self.assertEqual("openstar.tess.finalize", finalize.handler_id)
            self.assertEqual(
                "v20.8.1-transient-mode-validation",
                finalize.parameters["outputSuffix"],
            )
            final, next_request = engine.run_stage(
                completed,
                finalize,
                software_id="integration",
                software_version="1",
            )
            conclusion = final.stages[-1].result
            persisted = conclusion["transientModeValidation"]
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
        self.assertEqual(persisted, conclusion_file["transientModeValidation"])
        self.assertIn("Transient residual-mode validation", report)
        self.assertIn("Physical mechanism resolved: False", report)
        coordinator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
