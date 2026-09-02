from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from openstar_autonomy import (
    AutonomousInvestigationEngine,
    ExternalDataDependency,
    ScientificBranch,
)
from openstar_investigation import Investigation, InvestigationStore, sha256_file, sha256_json
from openstar_path_relocation import (
    HistoricalPathResolver,
    NO_HISTORICAL_PATH_RELOCATION,
)
from openstar_targets import InvestigationTarget
from openstar_workflow import StageRequest, WorkflowEngine
from .tess_localization_evidence import frozen_residual_localization_family
from .tess_mode_identification import (
    MULTIMODE_MODE_EVIDENCE_LINEAGE,
    V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE,
    build_confirmed_coherent_mode_method_contract,
    validate_confirmed_coherent_mode_dataset_lineage,
    validate_v20_8_confirmed_coherent_residual,
    validated_multimode_mode_evidence,
)
from .tess_long_baseline_frequency_confirmation import (
    HANDLER_ID as LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID,
    build_method_contract as build_long_baseline_frequency_confirmation_contract,
    validate_ambiguous_mode_identification,
    validate_frozen_dataset_lineage,
)
from .tess_v20_8_long_baseline_time_frequency_confirmation import (
    HANDLER_ID as V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID,
    build_dataset_specs as build_v20_8_long_baseline_dataset_specs,
    build_method_contract as build_v20_8_long_baseline_method_contract,
    method_contract_hash as v20_8_confirmation_method_contract_hash,
    validate_frozen_window_lineage as validate_v20_8_frozen_window_lineage,
)
from .tess_transient_mode_validation import (
    HANDLER_ID as TRANSIENT_MODE_VALIDATION_HANDLER_ID,
    build_dataset_specs as build_transient_mode_dataset_specs,
    build_method_contract as build_transient_mode_method_contract,
    validate_frozen_dataset_lineage as validate_transient_mode_frozen_lineage,
)
from .tess_nonstationary import (
    CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
    build_confirmed_nonstationary_method_contract,
    validate_confirmed_nonstationary_localization_boundary,
)
from .tess_residual_external_evidence import (
    HANDLER_ID as RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID,
    validate_target_supported_boundary,
)
from .tess_target_residual_astrophysical_mechanism import (
    HANDLER_ID as TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID,
    validate_mechanism_followup_boundary,
)
from .tess_neighbor_catalog_pixel_response_review import (
    HANDLER_ID as NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID,
    validate_review_boundary as validate_neighbor_catalog_pixel_response_boundary,
)
from .tess_binary_confirmation import (
    MORPHOLOGY_EVENT_SCREEN_ENTRY,
    _dataset_sector as binary_confirmation_dataset_sector,
    _original_time_origin as binary_confirmation_time_origin,
    morphology_event_screening_continuation,
)
from .tess_resolved_cycle import validated_cycle_period
from .tess_source_pair_lineage import frozen_source_pair_evidence

# Kept identical to the public identifiers in tess_investigation.  Importing that
# module eagerly would also import optional numerical/astronomy dependencies,
# even when a server is only enumerating targets.
WORKFLOW_ID = "openstar.workflow.tess-investigation.v1"
WORKFLOW_VERSION = "20.2"

_V2017_HANDLER_PREFIX = "openstar.tess.target-residual-archival-baseline."
_V2018_HANDLER_PREFIX = "openstar.tess.target-residual-pixel-recurrence."
_V2019_HANDLER_PREFIX = "openstar.tess.target-residual-multisector-source."
_ADDITIONAL_SECTOR_PREFIX = "openstar.tess.additional-sector-source-localization."
_ASTROPHYSICAL_INTERPRETATION_HANDLER = (
    "openstar.tess.target-residual-astrophysical-interpretation.analyze")
_MAIN_FAMILY_RECURRENCE_HANDLER = "openstar.tess.main-family-time-domain-recurrence.analyze"
_MAIN_FAMILY_FREQUENCY_REASSESSMENT_HANDLER = (
    "openstar.tess.main-family-frequency-domain-reassessment.analyze")

_KNOWN_TARGET_BLIND_PREPARER = "openstar.tess-known-target-blind-benchmark-preparer"


def _repair_known_target_blind_full_characterization(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append follow-up only at the verified schema-v1 catalog-match boundary."""
    expected_handlers = (
        "openstar.tess.prepare-target",
        "openstar.tess.primary-project.run",
        "openstar.tess.catalog-identity",
        "openstar.tess.hypotheses",
        "openstar.tess.planner",
        "openstar.tess.finalize",
    )
    stages = investigation.stages
    if not (
        investigation.status == "COMPLETE"
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and control.get("selectedExperiment") in (None, {})
        and len(stages) == len(expected_handlers)
        and tuple(stage.id for stage in stages) == (
            "001-prepare-target", "002-primary-distributed-search",
            "003-catalog-identity", "004-hypotheses", "005-planner", "006-finalize",
        )
        and tuple(stage.handler_id for stage in stages) == expected_handlers
        and all(stage.status == "COMPLETE" for stage in stages)
        and stages[0].triggered_by_stage_id is None
        and all(stages[index].triggered_by_stage_id == stages[index - 1].id
                for index in range(1, len(stages)))
        and stages[-1].stop is True
        and all(store.verified_terminal_stage_ledger_hash(investigation.id, stage)
                for stage in stages)
    ):
        return None

    prepared = stages[0].result or {}
    planner = stages[4].result or {}
    final = stages[5].result or {}
    claim = final.get("claim")
    if isinstance(claim, dict):
        claim = claim.get("claim")
    if not (
        planner.get("action") == "STOP"
        and planner.get("reason") == "catalog-period-match"
        and claim == "KNOWN_PERIOD_RECOVERED"
        and (prepared.get("sourceDatasetEntry") or {}).get("role") == "blind"
    ):
        return None

    try:
        project_path = Path(prepared["sourceProjectPath"])
        dataset_path = Path(prepared["datasetPath"])
        project = json.loads(project_path.read_text(encoding="utf-8"))
        dataset_entry = prepared["sourceDatasetEntry"]
        marker = project["preparer"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    expected_marker = {
        "preparerID": _KNOWN_TARGET_BLIND_PREPARER,
        "schemaVersion": 1,
        "ownedFiles": ["dataset.json", "project.json"],
        "projectID": project.get("id"),
    }
    hashes = stages[0].provenance.input_hashes if stages[0].provenance else {}
    if not project_path.is_file() or not dataset_path.is_file():
        return None
    if not (
        marker == expected_marker
        and project.get("datasets") == [dataset_entry]
        and dataset_entry.get("role") == "blind"
        and Path(str(dataset_entry.get("path"))).resolve() == dataset_path.resolve()
        and dataset_entry.get("datasetSHA256") == sha256_file(dataset_path)
        and hashes.get("sourceProjectManifest") == sha256_file(project_path)
        and hashes.get("sourceDataset") == sha256_file(dataset_path)
        and all(Path(artifact.path).is_file()
                and sha256_file(artifact.path) == artifact.sha256
                for stage in stages for artifact in stage.artifacts)
        and (stages[2].provenance and
             stages[2].provenance.input_hashes.get("primaryTargetResult")
             == sha256_json(stages[1].result))
        and (stages[3].provenance and
             stages[3].provenance.input_hashes.get("identity")
             == sha256_json(stages[2].result))
        and (stages[4].provenance and
             stages[4].provenance.input_hashes.get("hypothesisAnalysis")
             == sha256_json(stages[3].result))
        and (stages[5].provenance and
             stages[5].provenance.input_hashes.get("planner")
             == sha256_json(stages[4].result))
    ):
        return None

    request = StageRequest(
        id="007-prepare-independent-sectors",
        handler_id="openstar.tess.independent.prepare",
        parameters={"investigationGoal": "FULL_CHARACTERIZATION"},
        triggered_by_stage_id=stages[-1].id,
    )
    return store.set_control_state(investigation, status="RUNNING", control_state={
        "branchAssessments": [], "selectedExperiment": asdict(request),
        "schedulerAction": "RUN_EXPERIMENT",
        "recovery": "KNOWN_TARGET_BLIND_V1_FULL_CHARACTERIZATION_CONTINUATION",
    })


def _continue_finalized_main_family_frequency_reassessment(store, investigation, control,
        *, historical_path_resolver):
    """Admit only the immutable real 031/033/034 terminal lineage."""
    canonical = {"branchAssessments": [], "selectedExperiment": None,
        "schedulerAction": "INVESTIGATION_COMPLETE"}
    # A terminal status may predate generic control-state canonicalization.  Only
    # this exact boundary is allowed to normalize such mutable metadata.
    if investigation.status != "COMPLETE" or any(s.handler_id ==
            _MAIN_FAMILY_FREQUENCY_REASSESSMENT_HANDLER for s in investigation.stages):
        return None
    science = next((s for s in investigation.stages if s.id ==
        "031-target-residual-astrophysical-interpretation"), None)
    recurrence = next((s for s in investigation.stages if s.id ==
        "033-main-family-time-domain-recurrence"), None)
    final = next((s for s in investigation.stages if s.id == "034-finalize"), None)
    family_stage = next((s for s in reversed(investigation.stages) if s.status == "COMPLETE"
        and s.handler_id in {"openstar.tess.independent.harmonic-family.interpret",
            "openstar.tess.independent.broad.interpret"}), None)
    prior = recurrence.result if recurrence and isinstance(recurrence.result, dict) else {}
    combined = prior.get("combinedEvidence") or {}
    if not (science and recurrence and final and family_stage and final is investigation.stages[-1]
            and science.status == recurrence.status == final.status == "COMPLETE"
            and science.handler_id == _ASTROPHYSICAL_INTERPRETATION_HANDLER
            and recurrence.handler_id == _MAIN_FAMILY_RECURRENCE_HANDLER
            and final.handler_id == "openstar.tess.finalize" and final.stop is True
            and recurrence.triggered_by_stage_id == "032-finalize"
            and final.triggered_by_stage_id == recurrence.id
            and prior.get("classification") == "FREQUENCY_FAMILY_NOT_TIME_DOMAIN_REPLICATED"
            and prior.get("recommendedNextTest") == "MAIN_FAMILY_FREQUENCY_DOMAIN_REASSESSMENT"
            and combined.get("rawFamilyRecurrenceSectorIDs") == []
            and combined.get("possibleDoubleRecurrenceSectorIDs") == []
            and store.verified_terminal_stage_ledger_hash(investigation.id, science)
            and store.verified_terminal_stage_ledger_hash(investigation.id, family_stage)
            and store.verified_terminal_stage_ledger_hash(investigation.id, recurrence)
            and store.verified_terminal_stage_ledger_hash(investigation.id, final)
            and _verified_stage_json(science,
                "target-residual-astrophysical-interpretation-v20.14.1.json",
                resolver=historical_path_resolver)
            and _verified_stage_json(recurrence,
                "main-family-time-domain-recurrence-v20.14.2.json",
                resolver=historical_path_resolver)
            and _verified_stage_json(final,
                "conclusion-v20.14.2-main-family-time-domain-recurrence.json",
                resolver=historical_path_resolver)):
        return None
    request = StageRequest("035-main-family-frequency-domain-reassessment",
        _MAIN_FAMILY_FREQUENCY_REASSESSMENT_HANDLER, {}, final.id)
    return store.set_control_state(investigation, status="RUNNING", control_state={
        **canonical, "selectedExperiment": asdict(request),
        "schedulerAction": "RUN_EXPERIMENT",
        "recovery": "TESS_STAGE_034_MAIN_FAMILY_FREQUENCY_DOMAIN_REASSESSMENT"})


def _continue_finalized_main_family_recurrence(store, investigation, control,
        *, historical_path_resolver):
    """Admit exactly one append-only continuation after the completed 031/032 boundary."""
    terminal={"branchAssessments":[],"selectedExperiment":None,
              "schedulerAction":"INVESTIGATION_COMPLETE"}
    if (investigation.status != "COMPLETE" or control != terminal
            or any(s.handler_id == _MAIN_FAMILY_RECURRENCE_HANDLER
                   for s in investigation.stages)):
        return None
    science=next((s for s in investigation.stages if s.id ==
        "031-target-residual-astrophysical-interpretation"),None)
    final=next((s for s in investigation.stages if s.id=="032-finalize"),None)
    result=science.result if science else {}; family=result.get("mainPhotometricFamily") or {}
    if not (science and final and final is investigation.stages[-1]
            and science.status=="COMPLETE" and science.handler_id==_ASTROPHYSICAL_INTERPRETATION_HANDLER
            and result.get("classification")=="ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED"
            and result.get("physicalMechanismResolved") is True
            and result.get("targetResidualMechanismResolved") is True
            and result.get("targetResidualPeriodDays") == 3.600708338567666
            and result.get("smoothAmplitudeSupportingSectorIDs") == [68,95]
            and family.get("available") is True
            and family.get("representativeRawPeriodDays") == 7.546257528330875
            and family.get("possibleDoubleCycleDays") == 15.09251505666175
            and family.get("physicalCycleResolved") is False
            and _verified_stage_json(science,
                "target-residual-astrophysical-interpretation-v20.14.1.json",
                resolver=historical_path_resolver)
            and final.status=="COMPLETE" and final.handler_id=="openstar.tess.finalize"
            and final.stop is True and final.triggered_by_stage_id==science.id
            and final.parameters=={"outputSuffix":"v20.14.1-astrophysical-interpretation"}
            and _verified_stage_json(final,
                "conclusion-v20.14.1-astrophysical-interpretation.json",
                resolver=historical_path_resolver)):
        return None
    request=StageRequest("033-main-family-time-domain-recurrence",
        _MAIN_FAMILY_RECURRENCE_HANDLER,{},final.id)
    return store.set_control_state(investigation,status="RUNNING",control_state={
        "branchAssessments":[],"selectedExperiment":asdict(request),
        "schedulerAction":"RUN_EXPERIMENT",
        "recovery":"TESS_STAGE_032_MAIN_FAMILY_TIME_DOMAIN_RECURRENCE"})
_FAMILY_LEDGER_COMPATIBILITY_ERROR = (
    "RuntimeError: main recurrent-family artifact verification failed")


def _recover_failed_v2014_family_ledger_compatibility(store, investigation,
        *, historical_path_resolver):
    """Append the sole allowed retry of the known historical stage-030 failure."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    attempts = [s for s in investigation.stages
        if s.handler_id == _ASTROPHYSICAL_INTERPRETATION_HANDLER]
    if not (failed.id == "030-target-residual-astrophysical-interpretation"
            and failed.handler_id == _ASTROPHYSICAL_INTERPRETATION_HANDLER
            and failed.status == "FAILED"
            and failed.failure_classification == "NON_RETRYABLE"
            and failed.error == _FAMILY_LEDGER_COMPATIBILITY_ERROR
            and failed.triggered_by_stage_id == "029-finalize"
            and failed.result is None and attempts == [failed]
            and store.verified_terminal_stage_ledger_hash(
                investigation.id, failed)):
        return None
    science = next((s for s in investigation.stages
        if s.id == "028-target-residual-mechanism"), None)
    final = next((s for s in investigation.stages if s.id == "029-finalize"), None)
    family_stage = next((s for s in reversed(investigation.stages)
        if s.status == "COMPLETE" and s.handler_id in {
            "openstar.tess.independent.harmonic-family.interpret",
            "openstar.tess.independent.broad.interpret"}), None)
    family = ((family_stage.result or {}).get("harmonicFamily")
              if family_stage else None)
    final_family = (((final.result or {}).get("independentBroadVerification")
        or {}).get("harmonicFamily") if final else None)
    science_result = science.result if science else {}
    if not (science and final and family_stage
            and final.id == "029-finalize"
            and len(investigation.stages) >= 2
            and investigation.stages[-2] is final
            and final.triggered_by_stage_id == science.id
            and science.status == "COMPLETE"
            and science.handler_id == "openstar.tess.target-residual-mechanism.analyze"
            and science_result.get("classification") ==
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"
            and science_result.get("physicalMechanismResolved") is False
            and science_result.get("recommendedNextTest") ==
                "ASTROPHYSICAL_MECHANISM_INTERPRETATION"
            and science_result.get("adjudicationVersion") ==
                "route-independent-all-models-v1"
            and science_result.get("crossSectorPhaseUsed") is False
            and science_result.get("failClosedReasons") == []
            and science_result.get("replicatedMechanisms") ==
                ["SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"]
            and (science_result.get("replicatedMechanismSupportingSectorIDs")
                or {}).get("SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION") == [68, 95]
            and _verified_stage_json(science, "target-residual-mechanism-v20.14.json",
                resolver=historical_path_resolver)
            and final.status == "COMPLETE" and final.handler_id == "openstar.tess.finalize"
            and final.stop is True
            and final.parameters == {"outputSuffix": "v20.14-intrinsic"}
            and _verified_stage_json(final, "conclusion-v20.14-intrinsic.json",
                resolver=historical_path_resolver)
            and store.verified_terminal_stage_ledger_hash(
                investigation.id, family_stage)
            and family and family == final_family):
        return None
    request = StageRequest("031-target-residual-astrophysical-interpretation",
        _ASTROPHYSICAL_INTERPRETATION_HANDLER, {}, failed.id)
    return store.set_control_state(investigation, status="RUNNING", control_state={
        "branchAssessments": [], "selectedExperiment": asdict(request),
        "schedulerAction": "RUN_EXPERIMENT", "recovery":
        "TESS_V20_14_ASTROPHYSICAL_FAMILY_LEDGER_COMPATIBILITY_RECOVERY"})


def _continue_finalized_v2014_astrophysical_interpretation(store, investigation,
        control, *, historical_path_resolver):
    """Admit only the exact immutable stage-028/029 v20.14 compatibility boundary."""
    terminal = {"branchAssessments": [], "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE"}
    if (investigation.status != "COMPLETE" or control != terminal
            or any(s.handler_id == _ASTROPHYSICAL_INTERPRETATION_HANDLER
                   for s in investigation.stages)):
        return None
    science = next((s for s in investigation.stages
        if s.id == "028-target-residual-mechanism"), None)
    final = next((s for s in investigation.stages if s.id == "029-finalize"), None)
    result = science.result if science else {}
    if not (science and science.status == "COMPLETE"
            and science.handler_id == "openstar.tess.target-residual-mechanism.analyze"
            and result.get("classification") == "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"
            and result.get("physicalMechanismResolved") is False
            and result.get("recommendedNextTest") == "ASTROPHYSICAL_MECHANISM_INTERPRETATION"
            and result.get("adjudicationVersion") == "route-independent-all-models-v1"
            and result.get("crossSectorPhaseUsed") is False
            and result.get("failClosedReasons") == []
            and result.get("replicatedMechanisms") ==
                ["SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"]
            and (result.get("replicatedMechanismSupportingSectorIDs") or {}).get(
                "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION") == [68, 95]
            and _verified_stage_json(science, "target-residual-mechanism-v20.14.json",
                                     resolver=historical_path_resolver)
            and final and final is investigation.stages[-1] and final.status == "COMPLETE"
            and final.stop is True and final.handler_id == "openstar.tess.finalize"
            and final.parameters == {"outputSuffix": "v20.14-intrinsic"}
            and final.triggered_by_stage_id == science.id
            and _verified_stage_json(final, "conclusion-v20.14-intrinsic.json",
                                     resolver=historical_path_resolver)):
        return None
    request = StageRequest("030-target-residual-astrophysical-interpretation",
        _ASTROPHYSICAL_INTERPRETATION_HANDLER, {}, final.id)
    return store.set_control_state(investigation, status="RUNNING", control_state={
        "branchAssessments": [], "selectedExperiment": asdict(request),
        "schedulerAction": "RUN_EXPERIMENT",
        "recovery": "TESS_V20_14_ASTROPHYSICAL_MECHANISM_INTERPRETATION"})


def _recover_stage052_additional_sectors(store, investigation, control):
    """Narrow append-only admission for the already-persisted real stage 052."""
    expected_control={"branchAssessments":[],"selectedExperiment":None,
                      "schedulerAction":"ADVANCE_TO_NEXT_TARGET"}
    if (investigation.status != "QUIESCENT_AWAITING_DATA"
            or control != expected_control
            or any(s.handler_id.startswith(_ADDITIONAL_SECTOR_PREFIX) for s in investigation.stages)):
        return None
    boundary = next((s for s in reversed(investigation.stages)
        if s.handler_id == "openstar.tess.time-resolved-frequency-localization.interpret"), None)
    bridge = next((s for s in reversed(investigation.stages)
        if s.handler_id == "openstar.tess.time-resolved-frequency-localization.prepare" and s.status == "COMPLETE"), None)
    identity = next((s for s in reversed(investigation.stages)
        if s.handler_id == "openstar.tess.catalog-identity" and s.status == "COMPLETE"), None)
    if not (boundary and boundary is investigation.stages[-1] and boundary.status == "COMPLETE"
            and boundary.id == "052-interpret-time-resolved-frequency-localization"
            and boundary.stop is True and boundary.next_stage is None
            and bridge and identity): return None
    from .tess_additional_sector_source_localization import boundary_authorized, bridge_is_complete, unused_official_sectors
    if (not boundary_authorized(boundary.result or {}) or not bridge_is_complete(bridge.result or {})
            or not unused_official_sectors(identity.result or {}, bridge.result or {})): return None
    request = StageRequest("053-prepare-additional-sector-source-localization",
        _ADDITIONAL_SECTOR_PREFIX + "prepare", {}, boundary.id)
    return store.set_control_state(investigation, status="RUNNING", control_state={
        "branchAssessments": [], "selectedExperiment": asdict(request),
        "schedulerAction": "RUN_EXPERIMENT",
        "recovery": "TESS_STAGE_052_ADDITIONAL_SECTOR_SOURCE_LOCALIZATION"})


def _continue_finalized_v2018_multisector_source(store, investigation, control, *, historical_path_resolver):
    """Admit only the exact, cryptographically verified unresolved v20.18 result."""
    terminal = {"branchAssessments": [], "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE"}
    stale_finalizer = {"branchAssessments": [],
        "recovery": "TESS_V20_17_PIXEL_RECURRENCE_LOCALIZATION_V20_18",
        "schedulerAction": "RUN_EXPERIMENT", "selectedExperiment": {
            "handler_id": "openstar.tess.target-residual-pixel-recurrence.prepare",
            "id": "036-target-residual-pixel-recurrence-prepare", "parameters": {},
            "triggered_by_stage_id": "035-finalize"}}
    if (investigation.status != "COMPLETE" or control not in (terminal, stale_finalizer)
            or any(stage.handler_id.startswith(_V2019_HANDLER_PREFIX) for stage in investigation.stages)):
        return None
    try:
        from .tess_target_residual_multisector_source import verify_v2018_lineage
        lineage = verify_v2018_lineage(investigation.stages, resolver=historical_path_resolver)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError):
        return None
    request = StageRequest("040-target-residual-multisector-source-prepare",
        _V2019_HANDLER_PREFIX + "prepare", {}, lineage["finalizer"].id)
    return store.set_control_state(investigation, status="RUNNING", control_state={
        "branchAssessments": [], "selectedExperiment": asdict(request),
        "schedulerAction": "RUN_EXPERIMENT",
        "recovery": "TESS_V20_18_MULTISECTOR_SOURCE_LOCALIZATION_V20_19"})


def _continue_finalized_v2017_pixel_recurrence(store, investigation, control, *, historical_path_resolver):
    """Admit only the cryptographically verified v20.17 terminal boundary."""
    expected_terminal_control={"branchAssessments":[],"selectedExperiment":None,
                               "schedulerAction":"INVESTIGATION_COMPLETE"}
    expected_stale_admission = {
        "branchAssessments": [],
        "recovery": "TESS_V20_16_ARCHIVAL_BASELINE_EXTENSION_V20_17",
        "schedulerAction": "RUN_EXPERIMENT",
        "selectedExperiment": {
            "handler_id": "openstar.tess.target-residual-archival-baseline.prepare",
            "id": "032-target-residual-archival-baseline-prepare",
            "parameters": {},
            "triggered_by_stage_id": "031-finalize",
        },
    }
    if (investigation.status != "COMPLETE"
            or control not in (expected_terminal_control, expected_stale_admission)
            or any(s.handler_id.startswith(_V2018_HANDLER_PREFIX) for s in investigation.stages)):
        return None
    try:
        from .tess_target_residual_pixel_recurrence import verify_v2017_lineage
        lineage = verify_v2017_lineage(investigation.stages, resolver=historical_path_resolver)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError):
        return None
    if lineage["finalizer"] is not investigation.stages[-1]: return None
    request=StageRequest("036-target-residual-pixel-recurrence-prepare",
        "openstar.tess.target-residual-pixel-recurrence.prepare",{},lineage["finalizer"].id)
    return store.set_control_state(investigation,status="RUNNING",control_state={
        "branchAssessments":[],"selectedExperiment":asdict(request),"schedulerAction":"RUN_EXPERIMENT",
        "recovery":"TESS_V20_17_PIXEL_RECURRENCE_LOCALIZATION_V20_18"})


def _verified_stage_json(
    stage,
    expected_filename: str,
    *,
    resolver: HistoricalPathResolver = NO_HISTORICAL_PATH_RELOCATION,
) -> bool:
    if not isinstance(stage.result, dict): return False
    for reference in stage.artifacts:
        try:
            path = resolver.resolve(reference.path)
            if path.name == expected_filename and path.is_file() and reference.sha256 == sha256_file(path):
                with path.open(encoding="utf-8") as handle:
                    if json.load(handle) == stage.result: return True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return False


def _continue_finalized_v2016_archival_baseline(store: InvestigationStore,
        investigation: Investigation, control: dict, *,
        historical_path_resolver: HistoricalPathResolver =
        NO_HISTORICAL_PATH_RELOCATION) -> Investigation | None:
    """Narrow, idempotent admission of the exact finalized v20.16 boundary."""
    expected_terminal_control = {"branchAssessments": [], "selectedExperiment": None,
                                 "schedulerAction": "INVESTIGATION_COMPLETE"}
    expected_stale_finalizer = {
        "branchAssessments": [],
        "recovery": "TESS_V20_16_AWAITING_REVIEW",
        "schedulerAction": "RUN_EXPERIMENT",
        "selectedExperiment": {
            "handler_id": "openstar.tess.finalize",
            "id": "031-finalize",
            "parameters": {
                "outputSuffix": "v20.16-target-residual-predictive-validation",
            },
            "triggered_by_stage_id":
                "030-target-residual-mechanism-predictive-validation",
        },
    }
    if (investigation.status != "COMPLETE"
            or control not in (expected_terminal_control, expected_stale_finalizer)
            or any(s.handler_id.startswith(_V2017_HANDLER_PREFIX) for s in investigation.stages)):
        return None
    science=next((s for s in investigation.stages if s.id=="030-target-residual-mechanism-predictive-validation"),None)
    final=next((s for s in investigation.stages if s.id=="031-finalize"),None)
    if not (science and science.status=="COMPLETE" and science.handler_id=="openstar.tess.target-residual-mechanism-predictive-validation.analyze"
            and science.result and science.result.get("classification")=="TARGET_RESIDUAL_MECHANISM_PREDICTIVE_VALIDATION_UNRESOLVED"
            and science.result.get("recommendedNextTest")=="ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION"
            and science.result.get("physicalMechanismResolved") is False and _verified_stage_json(
                science, "target-residual-mechanism-predictive-validation-v20.16.json",
                resolver=historical_path_resolver)
            and final and final.status=="COMPLETE" and final.handler_id=="openstar.tess.finalize"
            and final.parameters == {"outputSuffix":"v20.16-target-residual-predictive-validation"}
            and final.triggered_by_stage_id==science.id and final is investigation.stages[-1]
            and final.result and final.result.get("targetResidualMechanismPredictiveValidation")==science.result
            and final.result.get("recommendedNextTest")=="ADDITIONAL_TEMPORAL_BASELINE_OR_MECHANISM_DISCRIMINATION"
            and _verified_stage_json(final,
                "conclusion-v20.16-target-residual-predictive-validation.json",
                resolver=historical_path_resolver)):
        return None
    request=StageRequest("032-target-residual-archival-baseline-prepare",
        "openstar.tess.target-residual-archival-baseline.prepare",{},final.id)
    return store.set_control_state(investigation,status="RUNNING",control_state={
        "branchAssessments":[],"selectedExperiment":asdict(request),"schedulerAction":"RUN_EXPERIMENT",
        "recovery":"TESS_V20_16_ARCHIVAL_BASELINE_EXTENSION_V20_17"})


def _stable_id(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-")
    if not normalized:
        raise ValueError("TESS target identifiers must not be empty.")
    return normalized


class TessInvestigationTargetSource:
    """Expose datasets in persisted TESS project manifests to the portfolio."""

    id = "openstar.tess-project-targets"
    version = "1"

    def __init__(self, project_paths: Sequence[str | Path]):
        self.project_paths = tuple(
            Path(path).expanduser().resolve() for path in project_paths
        )

    def enumerate_targets(self) -> tuple[InvestigationTarget, ...]:
        targets: list[InvestigationTarget] = []
        for project_path in sorted(self.project_paths, key=str):
            with project_path.open("r", encoding="utf-8") as handle:
                project = json.load(handle)
            project_id = project.get("id")
            datasets = project.get("datasets")
            if not project_id or not isinstance(datasets, list):
                raise ValueError(f"Not an OpenStar project manifest: {project_path}")

            for position, dataset in enumerate(datasets):
                if not isinstance(dataset, dict) or not dataset.get("id"):
                    raise ValueError(f"Invalid dataset entry in {project_path}")
                dataset_id = str(dataset["id"])
                target_id = f"{_stable_id(project_id)}:{_stable_id(dataset_id)}"
                investigation_id = str(
                    dataset.get("investigationID")
                    or f"tess-{_stable_id(project_id)}-{_stable_id(dataset_id)}"
                )
                raw_priority = dataset.get("autonomousPriority", position)
                eligible = dataset.get("autonomousEligible", True) is True
                targets.append(
                    InvestigationTarget(
                        id=target_id,
                        investigation_id=investigation_id,
                        workflow_id=WORKFLOW_ID,
                        workflow_version=WORKFLOW_VERSION,
                        priority=int(raw_priority),
                        eligible=eligible,
                        metadata={
                            "sourceProjectPath": str(project_path),
                            "sourceProjectID": str(project_id),
                            "datasetID": dataset_id,
                            "ticID": dataset.get("ticID"),
                            "targetName": dataset.get("targetName"),
                        },
                    )
                )
        return tuple(targets)


def _latest_complete(investigation: Investigation, handler_id: str):
    return next(
        (
            stage
            for stage in reversed(investigation.stages)
            if stage.handler_id == handler_id and stage.status == "COMPLETE"
        ),
        None,
    )


def _continuation_stage_id(stage, label: str) -> str:
    """Reconstruct the workflow continuation ID from its triggering stage."""
    try:
        number = int(str(stage.id).split("-", 1)[0]) + 1
    except (TypeError, ValueError):
        raise ValueError(f"Stage id must begin with an integer prefix: {stage.id}")
    return f"{number:03d}-{label}"


def _has_terminal_tess_evidence(investigation: Investigation) -> bool:
    if not investigation.stages:
        return False
    latest = investigation.stages[-1]
    return latest.status == "COMPLETE" and (
        latest.stop
        # Backward compatibility for terminal TESS evidence persisted before
        # InvestigationStage retained StageOutcome.stop.
        or latest.handler_id == "openstar.tess.finalize"
    )


def _intrinsic_target_boundary(investigation: Investigation):
    """Return only the unconsumed exact v20.12 target-dominant boundary."""
    stage = _latest_complete(investigation, "openstar.tess.multi-source-residual.interpret")
    result = (stage.result or {}) if stage is not None else {}
    attempted = any(item.handler_id == "openstar.tess.intrinsic-nonstationary.analyze"
                    for item in investigation.stages)
    if (stage is not None and not attempted
            and result.get("recommendedNextTest") == "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"
            and result.get("classification") == "TARGET_RESIDUAL_COMPONENT_DOMINANT"
            and result.get("residualModeOrigin") == "TARGET_DOMINANT"
            and result.get("physicalMechanismResolved") is False):
        return stage
    return None


def _target_residual_mechanism_boundary(investigation: Investigation):
    """Return only an unconsumed v20.13 temporal-mechanism recommendation."""
    stage = _latest_complete(investigation, "openstar.tess.intrinsic-nonstationary.analyze")
    result = (stage.result or {}) if stage is not None else {}
    attempted = any(item.handler_id == "openstar.tess.target-residual-mechanism.analyze"
                    for item in investigation.stages)
    if (stage is not None and not attempted
            and result.get("classification") in {
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL",
                "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL",
            }
            and result.get("recommendedNextTest")
            == "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
            and result.get("physicalMechanismResolved") is False):
        return stage
    return None


def _old_target_residual_adjudication_boundary(investigation: Investigation):
    """Match only the known stage-028 old-semantics pending-finalizer boundary."""
    stage = next((item for item in investigation.stages
        if item.id == "028-target-residual-mechanism"
        and item.handler_id == "openstar.tess.target-residual-mechanism.analyze"
        and item.status == "COMPLETE" and item.result is not None), None)
    if (stage is None or stage.result.get("adjudicationVersion") is not None
            or not any(Path(reference.path).name == "target-residual-mechanism-v20.14.json"
                       and bool(reference.sha256) for reference in stage.artifacts)):
        return None
    if any(item.handler_id == "openstar.tess.target-residual-mechanism-adjudication.analyze"
           for item in investigation.stages):
        return None
    control = investigation.metadata.get("controlState") or {}
    selected = control.get("selectedExperiment") or {}
    expected_next_stage = {
        "id": "029-finalize",
        "handler_id": "openstar.tess.finalize",
        "parameters": {"outputSuffix": "v20.14-intrinsic"},
        "triggered_by_stage_id": "028-target-residual-mechanism",
    }
    if (investigation.status != "RUNNING"
            or control.get("schedulerAction") != "RUN_EXPERIMENT"
            or not isinstance(stage.next_stage, dict)
            or stage.next_stage != expected_next_stage
            or selected != stage.next_stage):
        return None
    return stage


def _v2015_predictive_validation_boundary(investigation: Investigation):
    """Match only the known stage-029 v20.15 pending-finalizer boundary."""
    stage = next((item for item in investigation.stages
        if item.id == "029-target-residual-mechanism-adjudication"
        and item.handler_id == "openstar.tess.target-residual-mechanism-adjudication.analyze"
        and item.status == "COMPLETE" and item.result is not None), None)
    if stage is None or any(item.handler_id ==
            "openstar.tess.target-residual-mechanism-predictive-validation.analyze"
            for item in investigation.stages):
        return None
    result = stage.result
    artifacts = [ref for ref in stage.artifacts
        if Path(ref.path).name == "target-residual-mechanism-adjudication-v20.15.json"]
    if len(artifacts) != 1:
        return None
    artifact = artifacts[0]
    try:
        if (not artifact.sha256 or not Path(artifact.path).is_file()
                or sha256_file(artifact.path) != artifact.sha256):
            return None
        with Path(artifact.path).open(encoding="utf-8") as handle:
            frozen_result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if frozen_result != result:
        return None
    v14_stage = next((item for item in reversed(investigation.stages)
        if item.handler_id == "openstar.tess.target-residual-mechanism.analyze"
        and item.status == "COMPLETE" and item.result is not None), None)
    if v14_stage is None:
        return None
    v14_artifacts = [ref for ref in v14_stage.artifacts
        if Path(ref.path).name == "target-residual-mechanism-v20.14.json"]
    if len(v14_artifacts) != 1:
        return None
    v14_artifact = v14_artifacts[0]
    provenance = result.get("inputProvenance") or {}
    if (not v14_artifact.sha256 or not Path(v14_artifact.path).is_file()
            or sha256_file(v14_artifact.path) != v14_artifact.sha256
            or provenance.get("frozenV20.14ResultHash") != sha256_json(v14_stage.result)
            or provenance.get("frozenV20.14ArtifactSHA256") != v14_artifact.sha256):
        return None
    if (result.get("classification") != "TARGET_RESIDUAL_MECHANISM_UNRESOLVED"
            or result.get("recommendedNextTest") != "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
            or result.get("physicalMechanismResolved") is not False
            or result.get("failClosedReasons")):
        return None
    expected = {"id": "030-finalize", "handler_id": "openstar.tess.finalize",
        "parameters": {"outputSuffix": "v20.15-intrinsic-corrective-adjudication"},
        "triggered_by_stage_id": stage.id}
    control = investigation.metadata.get("controlState") or {}
    if (investigation.status != "RUNNING" or control.get("schedulerAction") != "RUN_EXPERIMENT"
            or stage.next_stage != expected or control.get("selectedExperiment") != expected):
        return None
    return stage


def _awaiting_nsc_adapter(investigation: Investigation) -> bool:
    """Identify only the incremental SkyMapper-to-NSC implementation boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.skymapper-resolved-counterpart-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest") == "NSC_RESOLVED_COUNTERPART_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.nsc-resolved-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_noirlab_adapter(investigation: Investigation) -> bool:
    """Identify only the incremental NSC-to-image-level implementation boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.nsc-resolved-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest") == "NOIRLAB_IMAGE_LEVEL_FORCED_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.noirlab-image-forced-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_current_des_adapter(investigation: Investigation) -> bool:
    """Keep the intentional NOIRLab-to-current-DES boundary durable."""
    stage = _latest_complete(
        investigation, "openstar.tess.noirlab-image-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest")
        == "DES_DR2_SINGLE_EPOCH_LOCAL_FORCED_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.des-dr2-se-local-forced-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_current_atlas_adapter(investigation: Investigation) -> bool:
    """Identify only the DES-to-credential-aware ATLAS implementation boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.des-dr2-se-local-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest") == "ATLAS_FORCED_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.atlas-forced-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_atlas_signed_reanalysis_adapter(investigation: Investigation) -> bool:
    stage = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest")
        == "ATLAS_SIGNED_FORCED_PHOTOMETRY_REANALYSIS"
        and result.get("physicalMechanismResolved") is False
        and not any(item.handler_id.startswith(
            "openstar.tess.atlas-forced-photometry-reanalysis."
        ) for item in investigation.stages)
    )


def _awaiting_post_atlas_targeted_observation_adapter(
    investigation: Investigation,
) -> bool:
    """Recognize only the complete, unambiguous direct v20.24 boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    if not (
        stage is not None
        and result.get("classification") == "ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("residualModeOrigin")
        == "ARCHIVAL_ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("recommendedNextTest")
        == "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
    ):
        return False
    position = investigation.stages.index(stage)
    preceding = investigation.stages[:position]
    later = investigation.stages[position + 1:]
    if not stage.triggered_by_stage_id or not any(
        item.id == stage.triggered_by_stage_id
        and item.status == "COMPLETE"
        and item.handler_id in (
            "openstar.tess.atlas-forced-photometry.run",
            "openstar.tess.atlas-forced-photometry.prepare",
        )
        for item in preceding
    ):
        return False

    search = dict(
        (result.get("distributedValidation") or {}).get("frequencySearch")
        or result.get("frequencySearch")
        or {}
    )
    minimum = search.get(
        "minimumFrequency", search.get("minFrequency", search.get("startFrequency"))
    )
    maximum = search.get(
        "maximumFrequency", search.get("maxFrequency", search.get("endFrequency"))
    )
    try:
        if maximum is None:
            total = int(search.get("totalFrequencies", search.get("frequencyCount")))
            step = float(search.get("frequencyStep", search.get("step")))
            maximum = float(minimum) + max(total - 1, 0) * step
        valid_search = (
            math.isfinite(float(minimum))
            and math.isfinite(float(maximum))
            and 0 < float(minimum) < float(maximum)
        )
    except (TypeError, ValueError):
        valid_search = False
    if not valid_search:
        return False

    if frozen_source_pair_evidence(investigation, stage) is None:
        return False

    return not any(
        item.handler_id.startswith((
            "openstar.tess.atlas-forced-photometry-reanalysis.",
            "openstar.tess.atlas-time-resolved.",
            "openstar.tess.atlas-fixed-window.",
            "openstar.tess.targeted-observation-planning.",
        ))
        for item in (*preceding, *later)
    )


def _is_unresolved_atlas_targeted_observation_boundary(investigation) -> bool:
    """Identify the semantic boundary even when its adapter evidence is invalid."""
    stage = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("classification") == "ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("residualModeOrigin")
        == "ARCHIVAL_ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("recommendedNextTest")
        == "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(item.handler_id.startswith(
            "openstar.tess.targeted-observation-planning."
        ) for item in investigation.stages)
    )


def _persisted_archive_continuation(investigation: Investigation):
    """Return a valid unattempted continuation recorded by an archive stage."""
    attempted_ids = {stage.id for stage in investigation.stages}
    archive_handlers = {
        "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare",
        "openstar.tess.gaia-source-resolved-counterpart-photometry.run",
        "openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
        "openstar.tess.skymapper-resolved-counterpart-photometry.run",
        "openstar.tess.nsc-resolved-photometry.prepare",
        "openstar.tess.nsc-resolved-photometry.run",
        "openstar.tess.noirlab-image-forced-photometry.prepare",
        "openstar.tess.noirlab-image-forced-photometry.run",
        "openstar.tess.des-dr2-se-local-forced-photometry.prepare",
        "openstar.tess.des-dr2-se-local-forced-photometry.run",
        "openstar.tess.atlas-forced-photometry.prepare",
        "openstar.tess.atlas-forced-photometry.collect",
        "openstar.tess.atlas-forced-photometry.run",
    }
    for stage in reversed(investigation.stages):
        if stage.handler_id not in archive_handlers or stage.status != "COMPLETE":
            continue
        raw = stage.next_stage
        if not isinstance(raw, dict):
            continue
        continuation_id = raw.get("id")
        handler_id = raw.get("handler_id")
        parameters = raw.get("parameters")
        if (
            not isinstance(continuation_id, str)
            or not continuation_id
            or continuation_id in attempted_ids
            or not isinstance(handler_id, str)
            or not isinstance(parameters, dict)
            or raw.get("triggered_by_stage_id") != stage.id
        ):
            continue

        # Collection is an asynchronous archive boundary.  Resume only the exact
        # immutable continuation that the collector produced, and ensure its
        # parameters agree with that durable collection result.  In particular,
        # never infer a replacement run or repeat archive collection here.
        if stage.handler_id == "openstar.tess.atlas-forced-photometry.collect":
            result = stage.result if isinstance(stage.result, dict) else {}
            project_path = result.get("projectPath")
            if handler_id == "openstar.tess.atlas-forced-photometry.run":
                valid = (
                    isinstance(project_path, str)
                    and bool(project_path)
                    and parameters == {"projectPath": project_path}
                )
            elif handler_id == "openstar.tess.atlas-forced-photometry.interpret":
                valid = not project_path and parameters == {
                    "distributedRunExpected": False
                }
            else:
                valid = False
            if not valid or any(
                item.handler_id in {
                    "openstar.tess.atlas-forced-photometry.run",
                    "openstar.tess.atlas-forced-photometry.interpret",
                }
                for item in investigation.stages
            ):
                continue
        return stage, raw
    return None


def _request_from_persisted(raw: dict) -> StageRequest:
    return StageRequest(
        id=str(raw["id"]),
        handler_id=str(raw["handler_id"]),
        parameters=dict(raw.get("parameters") or {}),
        triggered_by_stage_id=raw.get("triggered_by_stage_id"),
    )


_LEGACY_TRANSIENT_QUERY_TYPES = {
    "Timeout", "ReadTimeout", "ConnectTimeout", "TimeoutError", "ConnectionError",
    "ConnectError", "NewConnectionError", "MaxRetryError", "NameResolutionError",
    "NetworkError",
}


def _transient_required_identity_failure(result: dict) -> bool:
    # Catalog enrichment outages do not invalidate an identity already anchored
    # by TIC.  This also makes historical snapshots (which predate aggregate
    # coverage fields) restart-safe without scheduling repeated VizieR calls.
    if result.get("identityResolved") is True or (result.get("tic") or {}).get("found"):
        return False
    for key in ("tic", "vsx", "gaiaDR3", "gaiaVariability"):
        query = result.get(key) or {}
        if query.get("queryErrorClassification") == "TRANSIENT_INFRASTRUCTURE":
            return True
        # Historical identity snapshots retained the exception type at the
        # beginning of queryError, before structured classification existed.
        error_type = query.get("queryErrorType")
        if not error_type and isinstance(query.get("queryError"), str):
            error_type = query["queryError"].partition(":")[0]
        if error_type in _LEGACY_TRANSIENT_QUERY_TYPES:
            return True
    return False


def _repair_catalog_timeout_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    if investigation.status != "COMPLETE" or control.get("schedulerAction") != "INVESTIGATION_COMPLETE":
        return None
    conclusion = next((stage.result for stage in reversed(investigation.stages)
                       if stage.status == "COMPLETE" and isinstance(stage.result, dict)
                       and (stage.result.get("claim") is not None)), None)
    claim = conclusion.get("claim") if conclusion else None
    if isinstance(claim, dict):
        claim = claim.get("claim")
    planner = _latest_complete(investigation, "openstar.tess.planner")
    if claim != "HUMAN_REVIEW_REQUIRED" or planner is None or (planner.result or {}).get("reason") != "catalog-coverage-incomplete":
        return None

    identities = [stage for stage in investigation.stages
                  if stage.handler_id == "openstar.tess.catalog-identity"
                  and stage.status == "COMPLETE" and isinstance(stage.result, dict)]
    failed_index = next((index for index, stage in enumerate(identities)
                         if _transient_required_identity_failure(stage.result or {})), None)
    if failed_index is None or any(
        not _transient_required_identity_failure(stage.result or {})
        for stage in identities[failed_index + 1:]
    ):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-catalog-identity",
        handler_id="openstar.tess.catalog-identity",
        parameters={},
        triggered_by_stage_id=identities[failed_index].id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TRANSIENT_CATALOG_IDENTITY_COMPATIBILITY_RETRY",
        },
    )


_OFFICIAL_PRF_PREPARE = "openstar.tess.official-spoc-prf-forward-modeling.prepare"
_OFFICIAL_PRF_RUN = "openstar.tess.official-spoc-prf-forward-modeling.run"
_OFFICIAL_PRF_INTERPRET = "openstar.tess.official-spoc-prf-forward-modeling.interpret"


def _legacy_prf_transport_error(error: object) -> bool:
    """Recognize only transport text emitted by the obsolete PRF runner."""
    if not isinstance(error, str):
        return False
    return bool(
        re.fullmatch(
            r"TimeoutError: (?:The read operation timed out|timed out|read timed out)",
            error,
        )
        or re.fullmatch(
            r"URLError: <urlopen error (?:timed out|\[Errno 60\] Operation timed out)>",
            error,
        )
        or (
            error.startswith("URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING]")
            and error.endswith(">")
        )
    )


def _repair_official_prf_transport_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Select one append-only retry for a legacy transport-only PRF result."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
        or control.get("selectedExperiment") not in (None, {})
        or not _has_terminal_tess_evidence(investigation)
    ):
        return None

    stages = investigation.stages
    interpret_index = next((
        index for index in range(len(stages) - 1, -1, -1)
        if stages[index].handler_id == _OFFICIAL_PRF_INTERPRET
        and stages[index].status == "COMPLETE"
    ), None)
    if interpret_index is None:
        return None
    interpretation = stages[interpret_index]
    result = interpretation.result or {}
    if not (
        result.get("classification") == "BLOCKED_EXTERNAL_DATA"
        and result.get("recommendedNextTest")
        == "RETRY_PIXEL_RESPONSE_FUNCTION_DEBLENDING"
        and result.get("physicalMechanismResolved") is False
    ):
        return None

    by_id = {stage.id: stage for stage in stages}
    run = by_id.get(interpretation.triggered_by_stage_id)
    prepare = by_id.get(run.triggered_by_stage_id) if run is not None else None
    if not (
        run is not None and run.handler_id == _OFFICIAL_PRF_RUN
        and run.status == "COMPLETE" and isinstance(run.result, dict)
        and prepare is not None and prepare.handler_id == _OFFICIAL_PRF_PREPARE
        and prepare.status == "COMPLETE"
    ):
        return None
    sector_results = run.result.get("sectorResults")
    errors = run.result.get("errors")
    if sector_results != [] or not isinstance(errors, list) or not errors:
        return None
    if not all(
        isinstance(item, dict) and _legacy_prf_transport_error(item.get("error"))
        for item in errors
    ):
        return None

    # The completed finalizer must be the current terminal evidence and must
    # descend directly from this interpretation. Any later PRF stage means a
    # retry has already been attempted or superseded.
    terminal = stages[-1]
    if not (
        terminal.handler_id == "openstar.tess.finalize"
        and terminal.status == "COMPLETE"
        and terminal.triggered_by_stage_id == interpretation.id
        and all(
            not stage.handler_id.startswith(
                "openstar.tess.official-spoc-prf-forward-modeling."
            )
            for stage in stages[interpret_index + 1:]
        )
    ):
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-run-official-spoc-prf-forward-modeling",
        handler_id=_OFFICIAL_PRF_RUN,
        parameters={},
        triggered_by_stage_id=interpretation.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_OFFICIAL_PRF_TRANSPORT_COMPATIBILITY_RETRY",
        },
    )


def _positive_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _repair_promoted_period_characterization_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Continue pre-v20.4 promoted period families directly to morphology."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
    ):
        return None

    broad_index = next((
        index for index in range(len(investigation.stages) - 1, -1, -1)
        if investigation.stages[index].status == "COMPLETE"
        and investigation.stages[index].handler_id
        == "openstar.tess.independent.broad.interpret"
    ), None)
    if broad_index is None:
        return None
    broad = investigation.stages[broad_index]
    result = broad.result or {}
    family = result.get("harmonicFamily") or {}
    if not (
        (result.get("claimDecision") or {}).get("claim")
        == "INDEPENDENT_PERIOD_ESTIMATE"
        and result.get("promotionEligible") is True
        and _positive_number(family.get("representativeRawPeriodDays"))
        and _positive_number(family.get("possibleDoubleCycleDays"))
        and family.get("physicalCycleResolved") is not True
    ):
        return None

    later_stages = investigation.stages[broad_index + 1:]
    if any(
        stage.status == "COMPLETE"
        and stage.handler_id != "openstar.tess.finalize"
        for stage in later_stages
    ):
        return None
    terminal = next((
        stage for stage in reversed(later_stages)
        if stage.status == "COMPLETE"
        and stage.handler_id == "openstar.tess.finalize"
    ), None)
    terminal_claim = (((terminal.result or {}).get("claim") or {}).get("claim")
                      if terminal is not None else None)
    if terminal_claim != "INDEPENDENT_PERIOD_ESTIMATE":
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-morphology",
        handler_id="openstar.tess.morphology.analyze",
        parameters={},
        triggered_by_stage_id=broad.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": (
                "TESS_INDEPENDENT_PERIOD_CHARACTERIZATION_"
                "COMPATIBILITY_CONTINUATION"
            ),
        },
    )


def _repair_mode_identification_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append the stage introduced after the stable-residual terminal boundary."""
    if (investigation.status != "COMPLETE"
            or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"):
        return None
    summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    result = (summary.result or {}) if summary is not None else {}
    if not (summary is not None
            and result.get("recommendedNextTest") == "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
            and result.get("physicalMechanismResolved") is False
            and not any(stage.handler_id == "openstar.tess.mode-identification.analyze"
                        for stage in investigation.stages)):
        return None
    # Only repair the old implementation boundary: no completed scientific
    # stage other than finalize may follow the recommendation.
    index = investigation.stages.index(summary)
    if any(stage.status == "COMPLETE" and stage.handler_id != "openstar.tess.finalize"
           for stage in investigation.stages[index + 1:]):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-mode-identification",
        handler_id="openstar.tess.mode-identification.analyze",
        parameters={},
        triggered_by_stage_id=summary.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(continuation),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_MODE_IDENTIFICATION_COMPATIBILITY_CONTINUATION"},
    )


def _repair_v20_8_long_baseline_time_frequency_confirmation_terminal(
    store: InvestigationStore,
    investigation: Investigation,
    control: dict,
) -> Investigation | None:
    """Append only at the exact finalized unresolved v20.8 boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(
            stage.handler_id
            == V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
            for stage in investigation.stages
        )
    ):
        return None

    summary = _latest_complete(
        investigation, "openstar.tess.time-frequency.summarize"
    )
    latest = investigation.stages[-1] if investigation.stages else None
    if summary is None or latest is None or not (
        isinstance(summary.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE"
        and latest.stop is True
        and latest.triggered_by_stage_id == summary.id
        and latest.parameters.get("outputSuffix") == "v20.8"
        and isinstance(latest.result, dict)
        and latest.result.get("timeFrequencyEvolution") == summary.result
        and latest.result.get("recommendedNextTest")
        == "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
    ):
        return None
    summary_index = investigation.stages.index(summary)
    if tuple(investigation.stages[summary_index + 1:]) != (latest,):
        return None

    interpretation = next((
        stage for stage in investigation.stages
        if stage.id == summary.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.interpret"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    run = next((
        stage for stage in investigation.stages
        if interpretation is not None
        and stage.id == interpretation.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.run"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    preparation = next((
        stage for stage in investigation.stages
        if run is not None
        and stage.id == run.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    morphology = next((
        stage for stage in reversed(investigation.stages[:summary_index])
        if stage.handler_id == "openstar.tess.morphology.analyze"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if any(stage is None for stage in (
        interpretation, run, preparation, prepared, morphology
    )):
        return None

    interpretation_hashes = (
        interpretation.provenance.input_hashes
        if interpretation.provenance else {}
    )
    preparation_hashes = (
        preparation.provenance.input_hashes if preparation.provenance else {}
    )
    summary_hashes = (
        summary.provenance.input_hashes if summary.provenance else {}
    )
    if not (
        interpretation_hashes.get("preparation")
        == sha256_json(preparation.result)
        and interpretation_hashes.get("projectResult")
        == sha256_json(run.result)
        and preparation_hashes.get("morphology")
        == sha256_json(morphology.result)
        and summary_hashes.get("morphology")
        == sha256_json(morphology.result)
        and summary_hashes.get("timeFrequencyInterpretation")
        == sha256_json(interpretation.result)
        and all(
            store.verified_terminal_stage_ledger_hash(investigation.id, stage)
            for stage in (
                prepared, morphology, preparation, run,
                interpretation, summary, latest,
            )
        )
        and _verified_stage_json(summary, "time-frequency-v20.8.json")
        and _verified_stage_json(latest, "conclusion-v20.8.json")
    ):
        return None

    try:
        # The method is frozen before the lineage validator opens residual
        # window files containing flux values.
        contract = build_v20_8_long_baseline_method_contract(
            preparation=preparation.result,
            interpretation=interpretation.result,
            summary=summary.result,
        )
        dataset_specs = build_v20_8_long_baseline_dataset_specs(
            expected_tic_id=int(prepared.result["ticID"]),
            preparation=preparation.result,
        )
        validate_v20_8_frozen_window_lineage(
            method_contract=contract,
            dataset_specs=dataset_specs,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None

    continuation = StageRequest(
        id=_continuation_stage_id(
            latest, "long-baseline-time-frequency-confirmation"
        ),
        handler_id=(
            V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
        ),
        parameters={},
        triggered_by_stage_id=summary.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": (
                "TESS_V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
            ),
        },
    )


def _repair_transient_mode_validation_terminal(
    store: InvestigationStore,
    investigation: Investigation,
    control: dict,
) -> Investigation | None:
    """Append only at the exact finalized resolved-cycle transient boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and control.get("selectedExperiment") in (None, {})
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(
            stage.handler_id == TRANSIENT_MODE_VALIDATION_HANDLER_ID
            for stage in investigation.stages
        )
    ):
        return None

    summary = _latest_complete(
        investigation, "openstar.tess.time-frequency.summarize"
    )
    binary = _latest_complete(
        investigation, "openstar.tess.binary-confirmation.analyze"
    )
    latest = investigation.stages[-1] if investigation.stages else None
    if summary is None or binary is None or latest is None or not (
        isinstance(summary.result, dict)
        and isinstance(binary.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE"
        and latest.stop is True
        and latest.triggered_by_stage_id == summary.id
        and latest.parameters.get("outputSuffix") == "v20.8"
        and isinstance(latest.result, dict)
        and latest.result.get("timeFrequencyEvolution") == summary.result
        and latest.result.get("binaryConfirmation") == binary.result
        and latest.result.get("recommendedNextTest")
        == "TRANSIENT_MODE_VALIDATION"
    ):
        return None
    summary_index = investigation.stages.index(summary)
    if tuple(investigation.stages[summary_index + 1:]) != (latest,):
        return None

    interpretation = next((
        stage for stage in investigation.stages
        if stage.id == summary.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.interpret"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    run = next((
        stage for stage in investigation.stages
        if interpretation is not None
        and stage.id == interpretation.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.run"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    preparation = next((
        stage for stage in investigation.stages
        if run is not None
        and stage.id == run.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    morphology = next((
        stage for stage in reversed(investigation.stages[:summary_index])
        if stage.handler_id == "openstar.tess.morphology.analyze"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if any(stage is None for stage in (
        interpretation,
        run,
        preparation,
        prepared,
        morphology,
    )):
        return None

    interpretation_hashes = (
        interpretation.provenance.input_hashes
        if interpretation.provenance else {}
    )
    preparation_hashes = (
        preparation.provenance.input_hashes if preparation.provenance else {}
    )
    summary_hashes = summary.provenance.input_hashes if summary.provenance else {}
    if not (
        interpretation_hashes.get("preparation")
        == sha256_json(preparation.result)
        and interpretation_hashes.get("projectResult")
        == sha256_json(run.result)
        and preparation_hashes.get("morphology")
        == sha256_json(morphology.result)
        and summary_hashes.get("morphology")
        == sha256_json(morphology.result)
        and summary_hashes.get("timeFrequencyInterpretation")
        == sha256_json(interpretation.result)
        and all(
            store.verified_terminal_stage_ledger_hash(investigation.id, stage)
            for stage in (
                prepared,
                morphology,
                binary,
                preparation,
                run,
                interpretation,
                summary,
                latest,
            )
        )
        and _verified_stage_json(summary, "time-frequency-v20.8.json")
        and _verified_stage_json(latest, "conclusion-v20.8.json")
    ):
        return None

    try:
        # Freeze all choices before the lineage validator opens any window
        # dataset containing family-subtracted flux values.
        contract = build_transient_mode_method_contract(
            morphology=morphology.result,
            binary_confirmation=binary.result,
            preparation=preparation.result,
            interpretation=interpretation.result,
            summary=summary.result,
        )
        dataset_specs = build_transient_mode_dataset_specs(
            expected_tic_id=int(prepared.result["ticID"]),
            preparation=preparation.result,
        )
        validate_transient_mode_frozen_lineage(
            method_contract=contract,
            dataset_specs=dataset_specs,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None

    continuation = StageRequest(
        id=_continuation_stage_id(latest, "transient-mode-validation"),
        handler_id=TRANSIENT_MODE_VALIDATION_HANDLER_ID,
        parameters={},
        triggered_by_stage_id=summary.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_TRANSIENT_MODE_VALIDATION",
        },
    )


def _repair_v20_8_confirmed_coherent_mode_identification_terminal(
    store: InvestigationStore,
    investigation: Investigation,
    control: dict,
) -> Investigation | None:
    """Append v20.8.2 only at the verified positive v20.8.1 boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and control.get("selectedExperiment") in (None, {})
        and not any(
            stage.status == "RUNNING" for stage in investigation.stages
        )
        and not any(
            stage.handler_id == "openstar.tess.mode-identification.analyze"
            for stage in investigation.stages
        )
    ):
        return None

    confirmation = _latest_complete(
        investigation,
        V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID,
    )
    latest = investigation.stages[-1] if investigation.stages else None
    if confirmation is None or latest is None or not (
        isinstance(confirmation.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE"
        and latest.stop is True
        and latest.triggered_by_stage_id == confirmation.id
        and latest.parameters == {
            "outputSuffix": (
                "v20.8.1-long-baseline-time-frequency-confirmation"
            )
        }
        and isinstance(latest.result, dict)
        and latest.result.get("longBaselineTimeFrequencyConfirmation")
        == confirmation.result
        and latest.result.get("recommendedNextTest")
        == "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
    ):
        return None
    confirmation_index = investigation.stages.index(confirmation)
    if tuple(investigation.stages[confirmation_index + 1:]) != (latest,):
        return None

    summary = next((
        stage for stage in investigation.stages
        if stage.id == confirmation.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.summarize"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    interpretation = next((
        stage for stage in investigation.stages
        if summary is not None
        and stage.id == summary.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.interpret"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    run = next((
        stage for stage in investigation.stages
        if interpretation is not None
        and stage.id == interpretation.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.run"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    preparation = next((
        stage for stage in investigation.stages
        if run is not None
        and stage.id == run.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    morphology = next((
        stage for stage in reversed(
            investigation.stages[:confirmation_index]
        )
        if stage.handler_id == "openstar.tess.morphology.analyze"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    independent = next((
        stage for stage in reversed(
            investigation.stages[:confirmation_index]
        )
        if stage.handler_id == "openstar.tess.independent.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if any(stage is None for stage in (
        summary, interpretation, run, preparation, prepared, morphology,
        independent,
    )):
        return None

    hashes = (
        confirmation.provenance.input_hashes
        if confirmation.provenance else {}
    )
    try:
        confirmation_contract = build_v20_8_long_baseline_method_contract(
            preparation=preparation.result,
            interpretation=interpretation.result,
            summary=summary.result,
        )
        confirmation_contract_hash = (
            v20_8_confirmation_method_contract_hash(confirmation_contract)
        )
        if not (
            confirmation.result.get("methodContract")
            == confirmation_contract
            and confirmation.result.get("methodContractHash")
            == confirmation_contract_hash
            and hashes.get("methodContract")
            == confirmation_contract_hash
            and hashes.get("morphology")
            == sha256_json(morphology.result)
            and hashes.get("timeFrequencyPreparation")
            == sha256_json(preparation.result)
            and hashes.get("timeFrequencyProjectResult")
            == sha256_json(run.result)
            and hashes.get("timeFrequencyInterpretation")
            == sha256_json(interpretation.result)
            and hashes.get("timeFrequencySummary")
            == sha256_json(summary.result)
        ):
            return None
        evidence = validate_v20_8_confirmed_coherent_residual(
            confirmation.result
        )

        prepared_by_sector = {}
        for item in independent.result.get("preparedSectors") or []:
            if not isinstance(item, dict) or item.get("sector") is None:
                continue
            sector = int(item["sector"])
            if sector in prepared_by_sector:
                return None
            prepared_by_sector[sector] = item
        support = evidence["independentSectors"]
        if not set(support).issubset(prepared_by_sector):
            return None
        dataset_specs = [{
            "datasetID": prepared.result["datasetID"],
            "datasetPath": prepared.result["datasetPath"],
            "ticID": prepared.result["ticID"],
            "sector": prepared.result["sector"],
            "role": "PRIMARY",
        }]
        dataset_specs.extend({
            "datasetID": prepared_by_sector[sector]["datasetID"],
            "datasetPath": prepared_by_sector[sector]["datasetPath"],
            "ticID": prepared.result["ticID"],
            "sector": sector,
            "role": "INDEPENDENT",
        } for sector in support)

        # Freeze the next method before either lineage validator reads flux.
        mode_contract = build_confirmed_coherent_mode_method_contract(
            confirmation=confirmation.result,
            dataset_specs=dataset_specs,
        )
        window_specs = build_v20_8_long_baseline_dataset_specs(
            expected_tic_id=int(prepared.result["ticID"]),
            preparation=preparation.result,
        )
        validate_v20_8_frozen_window_lineage(
            method_contract=confirmation_contract,
            dataset_specs=window_specs,
        )
        if any(
            hashes.get(
                "frozenWindowDataset:"
                f"{spec['role']}:{spec['sector']}:{spec['windowIndex']}"
            ) != sha256_file(spec["datasetPath"])
            for spec in window_specs
        ):
            return None
        validate_confirmed_coherent_mode_dataset_lineage(
            method_contract=mode_contract,
            dataset_specs=dataset_specs,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None

    if not (
        all(
            store.verified_terminal_stage_ledger_hash(
                investigation.id, stage
            )
            for stage in (
                prepared, independent, morphology, preparation, run,
                interpretation, summary, confirmation, latest,
            )
        )
        and _verified_stage_json(
            confirmation,
            "long-baseline-time-frequency-confirmation-v20.8.1.json",
        )
        and _verified_stage_json(
            latest,
            "conclusion-v20.8.1-long-baseline-time-frequency-confirmation.json",
        )
    ):
        return None

    continuation = StageRequest(
        id=_continuation_stage_id(latest, "mode-identification"),
        handler_id="openstar.tess.mode-identification.analyze",
        parameters={
            "evidenceLineage": (
                V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE
            )
        },
        triggered_by_stage_id=confirmation.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": (
                "TESS_V20_8_CONFIRMED_COHERENT_MODE_IDENTIFICATION"
            ),
        },
    )


def _repair_long_baseline_frequency_confirmation_terminal(
    store: InvestigationStore,
    investigation: Investigation,
    control: dict,
) -> Investigation | None:
    """Append only at the validated ambiguous mode-identification boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(
            stage.handler_id == LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID
            for stage in investigation.stages
        )
    ):
        return None

    mode = _latest_complete(
        investigation, "openstar.tess.mode-identification.analyze"
    )
    physical = _latest_complete(investigation, "openstar.tess.physical.interpret")
    localization = _latest_complete(
        investigation, "openstar.tess.source-localization.analyze"
    )
    multimode = _latest_complete(
        investigation, "openstar.tess.multimode.summarize"
    )
    time_frequency = _latest_complete(
        investigation, "openstar.tess.time-frequency.summarize"
    )
    independent = _latest_complete(
        investigation, "openstar.tess.independent.prepare"
    )
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    if any(stage is None for stage in (
        mode, physical, localization, multimode, time_frequency,
        independent, prepared, latest,
    )):
        return None
    if not (
        isinstance(mode.result, dict)
        and isinstance(physical.result, dict)
        and isinstance(localization.result, dict)
        and isinstance(multimode.result, dict)
        and isinstance(time_frequency.result, dict)
        and isinstance(independent.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE"
        and latest.stop is True
        and latest.triggered_by_stage_id == mode.id
        and latest.parameters.get("outputSuffix")
        == "v20.9-mode-identification"
        and latest.result
        and latest.result.get("modeIdentification") == mode.result
        and latest.result.get("multiModeDecomposition") == multimode.result
        and latest.result.get("timeFrequencyEvolution") == time_frequency.result
        and latest.result.get("sourceLocalization") == localization.result
        and latest.result.get("recommendedNextTest")
        == "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
        and mode.parameters == {
            "evidenceLineage": MULTIMODE_MODE_EVIDENCE_LINEAGE
        }
        and mode.triggered_by_stage_id == multimode.id
    ):
        return None
    mode_index = investigation.stages.index(mode)
    if tuple(investigation.stages[mode_index + 1:]) != (latest,):
        return None

    try:
        mode_evidence = validate_ambiguous_mode_identification(mode.result)
        contract = build_long_baseline_frequency_confirmation_contract(
            mode.result
        )
        cycle = localization.result.get("physicalCycleEvidence")
        period = validated_cycle_period(cycle)
        physical_period = float(physical.result["physicalPeriodDays"])
        localized_period = float(localization.result["physicalPeriodDays"])
        time_frequency_period = float(
            (time_frequency.result.get("periodReference") or {})["periodDays"]
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    cross = localization.result.get("crossSector") or {}
    if not (
        period is not None
        and physical.result.get("physicalCycleEvidence") == cycle
        and physical.result.get("physicalMechanismResolved") is False
        and localization.result.get("version")
        == "openstar.tess-pixel-localization.v1"
        and cross.get("classification") == "TARGET_SOURCE_SUPPORTED"
        and cross.get("variableSignalOrigin") == "TARGET_CONSISTENT"
        and math.isclose(physical_period, period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(localized_period, period, rel_tol=1e-9, abs_tol=1e-12)
        and time_frequency.result.get("physicalMechanismResolved") is False
        and math.isclose(
            time_frequency_period, period, rel_tol=1e-9, abs_tol=1e-12
        )
    ):
        return None

    iterations = [
        stage for stage in investigation.stages
        if stage.handler_id == "openstar.tess.multimode.interpret"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ]
    multimode_evidence = validated_multimode_mode_evidence(
        multimode.result,
        physical_period_days=period,
        target_supporting_sectors=cross.get("targetSupportingSectors") or [],
        iteration_count=len(iterations),
    )
    candidate = mode.result.get("residualCandidate") or {}
    recurrent = multimode.result.get("bestRecurrentSecondaryMode") or {}
    if not (
        multimode_evidence is not None
        and mode_evidence["independentSectors"]
        == multimode_evidence["independentSectors"]
        and math.isclose(
            float(candidate.get("measuredFrequencyCyclesPerDay")),
            float(recurrent.get("medianFrequency")),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(mode_evidence["establishedPeriodDays"]),
            period,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        return None

    mode_hashes = mode.provenance.input_hashes if mode.provenance else {}
    if not (
        mode_hashes.get("multiModeDecomposition") == sha256_json(multimode.result)
        and mode_hashes.get("resolvedCycle") == sha256_json(cycle)
        and mode_hashes.get("physicalInterpretation") == sha256_json(physical.result)
        and mode_hashes.get("sourceLocalization") == sha256_json(localization.result)
        and mode_hashes.get("independentPreparation") == sha256_json(independent.result)
        and all(
            store.verified_terminal_stage_ledger_hash(investigation.id, stage)
            for stage in (
                prepared, independent, physical, localization,
                multimode, time_frequency, mode, latest,
            )
        )
        and _verified_stage_json(
            mode, "mode-identification-v20.9.json"
        )
        and _verified_stage_json(
            latest, "conclusion-v20.9-mode-identification.json"
        )
    ):
        return None

    support = set(mode_evidence["independentSectors"])
    prepared_sectors = [
        item for item in independent.result.get("preparedSectors") or []
        if isinstance(item, dict)
        and item.get("sector") is not None
        and item.get("datasetPath")
    ]
    if not support.issubset({int(item["sector"]) for item in prepared_sectors}):
        return None
    try:
        primary = prepared.result
        dataset_specs = [{
            "datasetID": primary["datasetID"],
            "datasetPath": primary["datasetPath"],
            "ticID": primary["ticID"],
            "sector": primary["sector"],
            "role": "PRIMARY",
        }]
        dataset_specs.extend({
            "datasetID": item["datasetID"],
            "datasetPath": item["datasetPath"],
            "ticID": primary["ticID"],
            "sector": int(item["sector"]),
            "role": "INDEPENDENT",
        } for item in prepared_sectors)
        validate_frozen_dataset_lineage(
            method_contract=contract,
            dataset_specs=dataset_specs,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None

    continuation = StageRequest(
        id=_continuation_stage_id(
            latest, "long-baseline-frequency-confirmation"
        ),
        handler_id=LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID,
        parameters={},
        triggered_by_stage_id=mode.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": (
                "TESS_AMBIGUOUS_MODE_LONG_BASELINE_FREQUENCY_CONFIRMATION"
            ),
        },
    )


def _repair_confirmed_nonstationary_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append modeling only at the exact finalized v20.9.1 boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(stage.handler_id.startswith("openstar.tess.nonstationary.")
                    for stage in investigation.stages)
    ):
        return None
    confirmation = _latest_complete(
        investigation, LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID)
    latest = investigation.stages[-1] if investigation.stages else None
    if confirmation is None or latest is None or not (
        isinstance(confirmation.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE" and latest.stop is True
        and latest.triggered_by_stage_id == confirmation.id
        and latest.parameters.get("outputSuffix")
        == "v20.9.1-long-baseline-frequency-confirmation"
        and (latest.result or {}).get("longBaselineFrequencyConfirmation")
        == confirmation.result
    ):
        return None
    index = investigation.stages.index(confirmation)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        return None
    try:
        contract = build_confirmed_nonstationary_method_contract(
            confirmation.result)
        paths = (contract.get("evidenceBoundary") or {}).get(
            "frozenDatasetPaths") or []
        if not paths or any(not isinstance(path, str) or not Path(path).is_file()
                            for path in paths):
            return None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    continuation = StageRequest(
        id=_continuation_stage_id(latest, "prepare-confirmed-nonstationary"),
        handler_id="openstar.tess.nonstationary.prepare",
        parameters={"evidenceLineage": CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE},
        triggered_by_stage_id=confirmation.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_CONFIRMED_NONSTATIONARY_MODE_MODELING",
        },
    )


def _repair_confirmed_residual_localization_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append v20.10 only at the exact finalized confirmed-model boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(
            stage.handler_id.startswith("openstar.tess.residual-mode-localization.")
            for stage in investigation.stages
        )
    ):
        return None
    summary = _latest_complete(
        investigation, "openstar.tess.nonstationary.summarize")
    confirmation = _latest_complete(
        investigation, LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID)
    localization = _latest_complete(
        investigation, "openstar.tess.source-localization.analyze")
    latest = investigation.stages[-1] if investigation.stages else None
    if any(stage is None for stage in (summary, confirmation, localization, latest)):
        return None
    if not (
        isinstance(summary.result, dict)
        and isinstance(confirmation.result, dict)
        and isinstance(localization.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE" and latest.stop is True
        and latest.triggered_by_stage_id == summary.id
        and latest.parameters.get("outputSuffix")
        == "v20.9.2-confirmed-nonstationary"
        and (latest.result or {}).get("nonstationaryModeling") == summary.result
    ):
        return None
    summary_index = investigation.stages.index(summary)
    if tuple(investigation.stages[summary_index + 1:]) != (latest,):
        return None
    try:
        validate_confirmed_nonstationary_localization_boundary(
            summary.result,
            confirmation.result,
            localization.result.get("physicalCycleEvidence"),
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    continuation = StageRequest(
        id=_continuation_stage_id(latest, "prepare-residual-mode-localization"),
        handler_id="openstar.tess.residual-mode-localization.prepare",
        parameters={},
        triggered_by_stage_id=summary.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_CONFIRMED_RESIDUAL_MODE_PIXEL_LOCALIZATION",
        },
    )


def _repair_target_supported_residual_external_evidence_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append v20.10.1 only at the exact finalized target-supported boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and control.get("selectedExperiment") in (None, {})
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(
            stage.handler_id == RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID
            for stage in investigation.stages
        )
    ):
        return None
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    nonstationary = _latest_complete(
        investigation, "openstar.tess.nonstationary.summarize"
    )
    confirmation = _latest_complete(
        investigation, LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID
    )
    source_localization = _latest_complete(
        investigation, "openstar.tess.source-localization.analyze"
    )
    identity = _latest_complete(investigation, "openstar.tess.catalog-identity")
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target" and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    if any(stage is None for stage in (
        localization, nonstationary, confirmation, source_localization,
        identity, prepared, latest,
    )):
        return None
    if not (
        isinstance(localization.result, dict)
        and isinstance(nonstationary.result, dict)
        and isinstance(confirmation.result, dict)
        and isinstance(source_localization.result, dict)
        and isinstance(identity.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE" and latest.stop is True
        and latest.triggered_by_stage_id == localization.id
        and latest.parameters == {"outputSuffix": "v20.10"}
        and (latest.result or {}).get("residualModeLocalization")
        == localization.result
        and (latest.result or {}).get("recommendedNextTest")
        == "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
    ):
        return None
    index = investigation.stages.index(localization)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        return None
    try:
        validate_target_supported_boundary(
            localization=localization.result,
            nonstationary=nonstationary.result,
            confirmation=confirmation.result,
            physical_cycle=source_localization.result.get("physicalCycleEvidence"),
            identity=identity.result,
            expected_tic_id=int(prepared.result["ticID"]),
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    hashes = localization.provenance.input_hashes if localization.provenance else {}
    if not (
        hashes.get("nonstationaryModeling") == sha256_json(nonstationary.result)
        and all(
            store.verified_terminal_stage_ledger_hash(investigation.id, stage)
            for stage in (
                prepared, identity, confirmation, source_localization,
                nonstationary, localization, latest,
            )
        )
        and _verified_stage_json(
            localization, "residual-mode-localization-v20.10.json"
        )
        and _verified_stage_json(latest, "conclusion-v20.10.json")
    ):
        return None
    continuation = StageRequest(
        id=_continuation_stage_id(latest, "residual-external-evidence"),
        handler_id=RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID,
        parameters={},
        triggered_by_stage_id=localization.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_TARGET_SUPPORTED_RESIDUAL_EXTERNAL_EVIDENCE",
        },
    )


def _repair_target_residual_astrophysical_mechanism_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append v20.10.2 only at the verified finalized v20.10.1 boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and control.get("selectedExperiment") in (None, {})
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(
            stage.handler_id == TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID
            for stage in investigation.stages
        )
    ):
        return None
    external = _latest_complete(investigation, RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID)
    identity = _latest_complete(investigation, "openstar.tess.catalog-identity")
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target" and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    if any(stage is None for stage in (external, identity, prepared, latest)):
        return None
    if not (
        isinstance(external.result, dict)
        and isinstance(identity.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE" and latest.stop is True
        and latest.triggered_by_stage_id == external.id
        and latest.parameters == {
            "outputSuffix": "v20.10.1-residual-external-evidence"
        }
        and (latest.result or {}).get("residualExternalEvidence") == external.result
        and (latest.result or {}).get("recommendedNextTest")
        == "TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_FOLLOWUP"
    ):
        return None
    index = investigation.stages.index(external)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        return None
    try:
        validate_mechanism_followup_boundary(
            external_evidence=external.result,
            identity=identity.result,
            expected_tic_id=int(prepared.result["ticID"]),
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    hashes = external.provenance.input_hashes if external.provenance else {}
    if not (
        hashes.get("catalogIdentity") == sha256_json(identity.result)
        and all(
            store.verified_terminal_stage_ledger_hash(investigation.id, stage)
            for stage in (prepared, identity, external, latest)
        )
        and _verified_stage_json(
            external, "residual-external-evidence-v20.10.1.json"
        )
        and _verified_stage_json(
            latest, "conclusion-v20.10.1-residual-external-evidence.json"
        )
    ):
        return None
    continuation = StageRequest(
        id=_continuation_stage_id(
            latest, "target-residual-astrophysical-mechanism"
        ),
        handler_id=TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID,
        parameters={},
        triggered_by_stage_id=external.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM",
        },
    )


def _repair_neighbor_catalog_pixel_response_review_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append v20.11.1 only at the verified finalized unresolved v20.11 boundary."""
    if not (
        investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and control.get("selectedExperiment") in (None, {})
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
        and not any(
            stage.handler_id == NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID
            for stage in investigation.stages
        )
    ):
        return None
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target" and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    identity = _latest_complete(investigation, "openstar.tess.catalog-identity")
    mode = _latest_complete(
        investigation, "openstar.tess.mode-identification.analyze"
    )
    review_preparation = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.prepare"
    )
    review = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.interpret"
    )
    latest = investigation.stages[-1] if investigation.stages else None
    if any(stage is None for stage in (
        prepared, identity, mode, review_preparation, review, latest,
    )):
        return None
    if not (
        isinstance(identity.result, dict)
        and isinstance(mode.result, dict)
        and isinstance(review_preparation.result, dict)
        and isinstance(review.result, dict)
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE"
        and latest.stop is True
        and latest.triggered_by_stage_id == review.id
        and latest.parameters == {"outputSuffix": "v20.11"}
        and (latest.result or {}).get("residualModeLocalizationReview")
        == review.result
        and (latest.result or {}).get("recommendedNextTest")
        == "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW"
    ):
        return None
    index = investigation.stages.index(review)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        return None
    try:
        validate_neighbor_catalog_pixel_response_boundary(
            preparation=review_preparation.result,
            localization_review=review.result,
            mode_identification=mode.result,
            identity=identity.result,
            expected_tic_id=int(prepared.result["ticID"]),
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    review_hashes = review.provenance.input_hashes if review.provenance else {}
    preparation_hashes = (
        review_preparation.provenance.input_hashes
        if review_preparation.provenance else {}
    )
    if not (
        review_hashes.get("preparation") == sha256_json(review_preparation.result)
        and preparation_hashes.get("modeIdentification") == sha256_json(mode.result)
        and all(
            store.verified_terminal_stage_ledger_hash(investigation.id, stage)
            for stage in (
                prepared, identity, mode, review_preparation, review, latest,
            )
        )
        and _verified_stage_json(
            review, "residual-mode-localization-review-v20.11.json"
        )
        and _verified_stage_json(latest, "conclusion-v20.11.json")
    ):
        return None
    continuation = StageRequest(
        id=_continuation_stage_id(
            latest, "neighbor-catalog-pixel-response-review"
        ),
        handler_id=NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID,
        parameters={},
        triggered_by_stage_id=review.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW",
        },
    )


def _repair_dynamic_harmonic_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append only the stage introduced after the old harmonic boundary."""
    if (investigation.status != "COMPLETE"
            or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"):
        return None
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    result = (mode.result or {}) if mode is not None else {}
    if not (mode is not None
            and result.get("recommendedNextTest") == "DYNAMIC_HARMONIC_MODELING"
            and result.get("physicalMechanismResolved") is False
            and not any(stage.handler_id == "openstar.tess.dynamic-harmonic.analyze"
                        for stage in investigation.stages)):
        return None
    index = investigation.stages.index(mode)
    if any(stage.status == "COMPLETE" and stage.handler_id != "openstar.tess.finalize"
           for stage in investigation.stages[index + 1:]):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-dynamic-harmonic-modeling",
        handler_id="openstar.tess.dynamic-harmonic.analyze",
        parameters={}, triggered_by_stage_id=mode.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(continuation),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_DYNAMIC_HARMONIC_COMPATIBILITY_CONTINUATION"},
    )


def _repair_unresolved_family_dynamic_harmonic_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append the new unresolved-family alias test at its exact old boundary."""
    if (investigation.status != "COMPLETE"
            or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
            or not investigation.stages):
        return None
    summary = _latest_complete(
        investigation, "openstar.tess.time-frequency.summarize")
    morphology = _latest_complete(
        investigation, "openstar.tess.morphology.analyze")
    result = (summary.result or {}) if summary is not None else {}
    morphology_result = (morphology.result or {}) if morphology is not None else {}
    period_reference = result.get("periodReference") or {}
    continuation = morphology_result.get("continuationEvidence") or {}
    try:
        raw_period = float(morphology_result.get("rawPeriodDays"))
        double_period = float(morphology_result.get("possibleDoubleCycleDays"))
        analysis_period = float(continuation.get("analysisReferencePeriodDays"))
        summary_period = float(period_reference.get("periodDays"))
    except (TypeError, ValueError):
        return None
    latest = investigation.stages[-1]
    exact_boundary = (
        summary is not None
        and morphology is not None
        and result.get("recommendedNextTest") == "DYNAMIC_HARMONIC_MODELING"
        and result.get("physicalMechanismResolved") is False
        and period_reference.get("kind")
        == "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE"
        and period_reference.get("physicalCycleResolved") is False
        and morphology_result.get("physicalCycleResolved") is False
        and morphology_result.get("resolvedPhysicalPeriodDays") is None
        and continuation.get("timeFrequencyEvolutionWarranted") is True
        and continuation.get("periodReferenceKind")
        == "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE"
        and all(math.isfinite(value) and value > 0 for value in (
            raw_period, double_period, analysis_period, summary_period,
        ))
        and math.isclose(
            double_period, 2.0 * raw_period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            analysis_period, double_period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            summary_period, double_period, rel_tol=1e-9, abs_tol=1e-12)
        and latest.status == "COMPLETE"
        and latest.handler_id == "openstar.tess.finalize"
        and latest.stop is True
        and latest.triggered_by_stage_id == summary.id
        and not any(
            stage.handler_id == "openstar.tess.dynamic-harmonic.analyze"
            for stage in investigation.stages
        )
    )
    if not exact_boundary:
        return None
    summary_index = investigation.stages.index(summary)
    if tuple(investigation.stages[summary_index + 1:]) != (latest,):
        return None
    prefixes = [
        int(stage.id.partition("-")[0])
        for stage in investigation.stages
        if stage.id.partition("-")[0].isdigit()
    ]
    continuation_request = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-dynamic-harmonic-modeling",
        handler_id="openstar.tess.dynamic-harmonic.analyze",
        parameters={
            "evidenceLineage":
            "UNRESOLVED_FAMILY_TIME_FREQUENCY_RECOMMENDATION"
        },
        triggered_by_stage_id=summary.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation_request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": (
                "TESS_UNRESOLVED_FAMILY_DYNAMIC_HARMONIC_CONTINUATION"
            ),
        },
    )


def _repair_unmatched_alias_model_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append the matched-frequency alias test after the obsolete v20.10 result."""
    if (investigation.status != "COMPLETE"
            or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
            or not investigation.stages):
        return None
    dynamic = _latest_complete(
        investigation, "openstar.tess.dynamic-harmonic.analyze")
    if dynamic is None:
        return None
    result = dynamic.result or {}
    alias = result.get("periodAliasResolution") or {}
    models = result.get("periodHypothesisModels") or {}
    raw_model = models.get("rawFamily") or {}
    double_model = models.get("doubleCycle") or {}
    latest = investigation.stages[-1]
    exact_boundary = (
        result.get("evidenceLineage")
        == "UNRESOLVED_FAMILY_TIME_FREQUENCY_RECOMMENDATION"
        and result.get("classification")
        == "UNRESOLVED_FAMILY_DYNAMIC_HARMONIC_ALIAS_AMBIGUOUS"
        and result.get("physicalCycleResolved") is False
        and result.get("resolvedPhysicalPeriodDays") is None
        and alias.get("method")
        == "LEAVE_ONE_SECTOR_OUT_PHASE_PREDICTION_WITH_SECTOR_AMPLITUDES"
        and raw_model.get("harmonicOrdersTested") == [1, 2, 3, 4]
        and double_model.get("harmonicOrdersTested") == [1, 2, 3, 4]
        and latest.status == "COMPLETE"
        and latest.handler_id == "openstar.tess.finalize"
        and latest.stop is True
        and latest.triggered_by_stage_id == dynamic.id
        and not any(
            (stage.parameters or {}).get("evidenceLineage")
            == "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT"
            for stage in investigation.stages
        )
    )
    if not exact_boundary:
        return None
    dynamic_index = investigation.stages.index(dynamic)
    if tuple(investigation.stages[dynamic_index + 1:]) != (latest,):
        return None
    prefixes = [
        int(stage.id.partition("-")[0])
        for stage in investigation.stages
        if stage.id.partition("-")[0].isdigit()
    ]
    continuation_request = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-nested-cycle-alias-reassessment",
        handler_id="openstar.tess.dynamic-harmonic.analyze",
        parameters={
            "evidenceLineage":
            "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT"
        },
        triggered_by_stage_id=dynamic.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation_request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_NESTED_ODD_HARMONIC_ALIAS_REASSESSMENT",
        },
    )


def _continue_finalized_nested_cycle_physical_interpretation(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append mechanism interpretation after the exact resolved v20.10.1 edge."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
        or not investigation.stages
    ):
        return None
    dynamic = _latest_complete(
        investigation, "openstar.tess.dynamic-harmonic.analyze")
    if dynamic is None:
        return None
    result = dynamic.result or {}
    alias = result.get("periodAliasResolution") or {}
    latest = investigation.stages[-1]
    try:
        raw_period = float(result.get("rawFamilyPeriodDays"))
        resolved_period = float(result.get("resolvedPhysicalPeriodDays"))
        reference_period = float(result.get("referenceFamilyPeriodDays"))
        selected_period = float(alias.get("selectedPeriodDays"))
        possible_double = float(result.get("possibleDoubleCycleDays"))
        threshold = float(alias.get("conservativeThreshold"))
        aggregate = float(
            alias.get("aggregateIndependentDeltaBicFullMinusEvenOnly"))
        primary = int(alias.get("primarySector"))
        minimum = int(alias.get("minimumSupportingIndependentHeldOutSectors"))
        supporters = [int(value) for value in (
            alias.get("oddHarmonicSupportingIndependentHeldOutSectors") or [])]
    except (TypeError, ValueError):
        return None
    comparison_support = set()
    try:
        for comparison in alias.get("comparisons") or []:
            if (
                isinstance(comparison, dict)
                and comparison.get("role") == "INDEPENDENT"
                and comparison.get("oddHarmonicStructureSupported") is True
            ):
                comparison_support.add(int(comparison.get("sector")))
    except (TypeError, ValueError):
        return None
    exact_boundary = (
        result.get("evidenceLineage")
        in {
            "UNRESOLVED_FAMILY_TIME_FREQUENCY_RECOMMENDATION",
            "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT",
        }
        and result.get("classification")
        == "DOUBLE_CYCLE_ODD_HARMONICS_PREDICTIVELY_SUPPORTED"
        and result.get("physicalCycleResolved") is True
        and result.get("physicalMechanismResolved") is False
        and result.get("referencePeriodRole")
        == "PREDICTIVELY_RESOLVED_PHOTOMETRIC_CYCLE"
        and result.get("recommendedNextTest")
        == "BINARY_ROTATION_EXTERNAL_EVIDENCE"
        and alias.get("method")
        == "NESTED_EVEN_ONLY_VS_EVEN_PLUS_ODD_LEAVE_ONE_SECTOR_OUT_PREDICTION"
        and alias.get("criterion") == "BIC"
        and alias.get("physicalCycleResolved") is True
        and alias.get("selectedPeriodRelation") == "DOUBLE_CYCLE"
        and alias.get("equalHalfEvenHarmonicOrders") == [2, 4, 6, 8]
        and alias.get("discriminatingOddHarmonicOrders") == [1, 3, 5, 7]
        and alias.get("fullDoubleCycleHarmonicOrders") == list(range(1, 9))
        and alias.get("maximumAbsoluteFrequencyMatched") is True
        and all(math.isfinite(value) and value > 0 for value in (
            raw_period, resolved_period, reference_period, selected_period,
            possible_double, threshold, aggregate,
        ))
        and math.isclose(
            resolved_period, 2.0 * raw_period, rel_tol=1e-9, abs_tol=1e-12)
        and all(math.isclose(
            value, resolved_period, rel_tol=1e-9, abs_tol=1e-12)
            for value in (reference_period, selected_period, possible_double))
        and math.isclose(threshold, 10.0, rel_tol=0.0, abs_tol=1e-12)
        and aggregate >= threshold
        and minimum == 3
        and len(supporters) >= minimum
        and len(set(supporters)) == len(supporters)
        and primary not in supporters
        and set(supporters).issubset(comparison_support)
        and latest.status == "COMPLETE"
        and latest.handler_id == "openstar.tess.finalize"
        and latest.stop is True
        and latest.triggered_by_stage_id == dynamic.id
        and not any(
            stage.handler_id == "openstar.tess.physical.interpret"
            for stage in investigation.stages
        )
    )
    if not exact_boundary:
        return None
    dynamic_index = investigation.stages.index(dynamic)
    if tuple(investigation.stages[dynamic_index + 1:]) != (latest,):
        return None
    prefixes = [
        int(stage.id.partition("-")[0])
        for stage in investigation.stages
        if stage.id.partition("-")[0].isdigit()
    ]
    request = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-physical-interpretation",
        handler_id="openstar.tess.physical.interpret",
        parameters={
            "evidenceLineage": "NESTED_ODD_HARMONIC_RESOLVED_CYCLE",
        },
        triggered_by_stage_id=dynamic.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_NESTED_CYCLE_PHYSICAL_INTERPRETATION",
        },
    )


def _validated_persisted_nested_cycle_period(
    cycle: dict,
) -> float | None:
    """Validate the dependency-light persisted form of the resolved cycle."""
    try:
        period = float(cycle.get("periodDays"))
        raw_period = float(cycle.get("rawFamilyPeriodDays"))
        possible_double = float(cycle.get("possibleDoubleCycleDays"))
        reference_period = float(cycle.get("referenceFamilyPeriodDays"))
        selected_period = float(cycle.get("selectedPeriodDays"))
        threshold = float(cycle.get("conservativeThreshold"))
        aggregate = float(
            cycle.get("aggregateIndependentDeltaBicFullMinusEvenOnly"))
        minimum = int(cycle.get("minimumSupportingIndependentSectors"))
        supporters = [int(value) for value in (
            cycle.get("supportingIndependentSectors") or [])]
    except (TypeError, ValueError):
        return None
    valid = (
        cycle.get("contractVersion")
        == "openstar.tess-authoritative-resolved-cycle.v1"
        and cycle.get("sourceKind") in {
            "NESTED_ODD_HARMONIC_PREDICTIVE_RESOLUTION",
            "MORPHOLOGY_AND_NESTED_PREDICTION_CONSISTENT",
        }
        and cycle.get("sourceClassification")
        == "DOUBLE_CYCLE_ODD_HARMONICS_PREDICTIVELY_SUPPORTED"
        and cycle.get("sourceEvidenceLineage") in {
            "UNRESOLVED_FAMILY_TIME_FREQUENCY_RECOMMENDATION",
            "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT",
        }
        and cycle.get("physicalCycleResolved") is True
        and cycle.get("physicalMechanismResolved") is False
        and cycle.get("criterion") == "BIC"
        and all(math.isfinite(value) and value > 0 for value in (
            period, raw_period, possible_double, reference_period,
            selected_period, threshold, aggregate,
        ))
        and math.isclose(
            period, 2.0 * raw_period, rel_tol=1e-9, abs_tol=1e-12)
        and all(math.isclose(
            value, period, rel_tol=1e-9, abs_tol=1e-12)
            for value in (
                possible_double, reference_period, selected_period))
        and math.isclose(threshold, 10.0, rel_tol=0.0, abs_tol=1e-12)
        and aggregate >= threshold
        and minimum == 3
        and len(supporters) >= minimum
        and len(set(supporters)) == len(supporters)
        and cycle.get("primarySectorExcludedFromSupport") is True
        and cycle.get("maximumAbsoluteFrequencyMatched") is True
    )
    return period if valid else None


def _continue_finalized_physical_source_localization(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append v20.6 after the exact finalized v20.5.1 contamination edge."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
        or not investigation.stages
    ):
        return None
    physical = _latest_complete(
        investigation, "openstar.tess.physical.interpret")
    if physical is None:
        return None
    result = physical.result or {}
    cycle = result.get("physicalCycleEvidence") or {}
    period = _validated_persisted_nested_cycle_period(cycle)
    try:
        reported_period = float(result.get("physicalPeriodDays"))
        reported_harmonic = float(
            result.get("photometricFirstHarmonicPeriodDays"))
    except (TypeError, ValueError):
        return None
    latest = investigation.stages[-1]
    exact_boundary = (
        period is not None
        and result.get("version")
        == "openstar.tess-physical-interpretation.v2"
        and math.isfinite(reported_period)
        and math.isfinite(reported_harmonic)
        and math.isclose(
            reported_period, period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            reported_harmonic, period / 2.0,
            rel_tol=1e-9, abs_tol=1e-12)
        and result.get("physicalMechanismResolved") is False
        and (result.get("contaminationScreen") or {}).get(
            "flaggedByExistingMetadata") is True
        and result.get("recommendedNextTest")
        == "PIXEL_LEVEL_SOURCE_LOCALIZATION"
        and latest.status == "COMPLETE"
        and latest.handler_id == "openstar.tess.finalize"
        and latest.stop is True
        and latest.triggered_by_stage_id == physical.id
        and latest.parameters.get("outputSuffix")
        == "v20.5.1-dynamic-cycle"
        and not any(
            stage.handler_id == "openstar.tess.source-localization.analyze"
            for stage in investigation.stages
        )
    )
    if not exact_boundary:
        return None
    physical_index = investigation.stages.index(physical)
    if tuple(investigation.stages[physical_index + 1:]) != (latest,):
        return None
    prefixes = [
        int(stage.id.partition("-")[0])
        for stage in investigation.stages
        if stage.id.partition("-")[0].isdigit()
    ]
    request = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-source-localization",
        handler_id="openstar.tess.source-localization.analyze",
        parameters={
            "evidenceLineage":
            "PHYSICAL_INTERPRETATION_PIXEL_LOCALIZATION",
        },
        triggered_by_stage_id=physical.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_V20_5_1_PIXEL_SOURCE_LOCALIZATION",
        },
    )


def _continue_finalized_source_localization_multimode(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append v20.7 after the exact finalized target-source v20.6 edge."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
        or not investigation.stages
    ):
        return None
    localization = _latest_complete(
        investigation, "openstar.tess.source-localization.analyze")
    physical = _latest_complete(
        investigation, "openstar.tess.physical.interpret")
    if localization is None or physical is None:
        return None
    result = localization.result or {}
    cross = result.get("crossSector") or {}
    cycle = result.get("physicalCycleEvidence") or {}
    period = _validated_persisted_nested_cycle_period(cycle)
    try:
        reported_period = float(result.get("physicalPeriodDays"))
        reported_harmonic = float(
            result.get("photometricFirstHarmonicPeriodDays"))
        eligible = int(cross.get("independentEligibleSectorCount"))
        required = int(cross.get("requiredIndependentSupportCount"))
        target_support = sorted(
            int(value) for value in cross.get("targetSupportingSectors") or [])
        off_target = sorted(
            int(value) for value in cross.get("offTargetSectors") or [])
        ambiguous = sorted(
            int(value) for value in cross.get("ambiguousSectors") or [])
    except (TypeError, ValueError):
        return None

    independent = [
        item for item in result.get("sectorResults") or []
        if isinstance(item, dict)
        and item.get("role") == "independent"
        and item.get("available") is True
    ]
    try:
        independent_sectors = [int(item["sector"]) for item in independent]
        observed_target = sorted(
            int(item["sector"]) for item in independent
            if item.get("classification") == "TARGET_CONSISTENT"
        )
        observed_off_target = sorted(
            int(item["sector"]) for item in independent
            if item.get("classification") == "OFF_TARGET"
        )
        observed_ambiguous = sorted(
            int(item["sector"]) for item in independent
            if item.get("classification") == "AMBIGUOUS"
        )
    except (KeyError, TypeError, ValueError):
        return None

    latest = investigation.stages[-1]
    physical_result = physical.result or {}
    exact_boundary = (
        period is not None
        and result.get("version") == "openstar.tess-pixel-localization.v1"
        and result.get("physicalCycleEvidence")
        == physical_result.get("physicalCycleEvidence")
        and math.isfinite(reported_period)
        and math.isfinite(reported_harmonic)
        and math.isclose(
            reported_period, period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            reported_harmonic, period / 2.0,
            rel_tol=1e-9, abs_tol=1e-12)
        and localization.parameters.get("evidenceLineage")
        == "PHYSICAL_INTERPRETATION_PIXEL_LOCALIZATION"
        and localization.triggered_by_stage_id == physical.id
        and cross.get("classification") == "TARGET_SOURCE_SUPPORTED"
        and cross.get("variableSignalOrigin") == "TARGET_CONSISTENT"
        and cross.get("recommendedNextTest")
        == "MULTI_MODE_FREQUENCY_DECOMPOSITION"
        and result.get("recommendedNextTest")
        == "MULTI_MODE_FREQUENCY_DECOMPOSITION"
        and (result.get("contaminationInterpretation") or {}).get(
            "existingCatalogContaminationCanBeCleared") is False
        and eligible == len(independent)
        and len(set(independent_sectors)) == eligible
        and eligible >= 3
        and required == max(3, eligible // 2 + 1)
        and (
            len(observed_target)
            + len(observed_off_target)
            + len(observed_ambiguous)
            == eligible
        )
        and len(target_support) >= required
        and len(set(target_support)) == len(target_support)
        and target_support == observed_target
        and off_target == observed_off_target
        and ambiguous == observed_ambiguous
        and latest.status == "COMPLETE"
        and latest.handler_id == "openstar.tess.finalize"
        and latest.stop is True
        and latest.triggered_by_stage_id == localization.id
        and latest.parameters.get("outputSuffix") == "v20.6"
        and not any(
            stage.handler_id.startswith("openstar.tess.multimode.")
            for stage in investigation.stages
        )
    )
    if not exact_boundary:
        return None
    localization_index = investigation.stages.index(localization)
    if tuple(investigation.stages[localization_index + 1:]) != (latest,):
        return None
    prefixes = [
        int(stage.id.partition("-")[0])
        for stage in investigation.stages
        if stage.id.partition("-")[0].isdigit()
    ]
    request = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-multimode-iteration-1",
        handler_id="openstar.tess.multimode.prepare",
        parameters={"iteration": 1},
        triggered_by_stage_id=localization.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_V20_6_MULTI_MODE_FREQUENCY_DECOMPOSITION",
        },
    )


def _continue_finalized_multimode_mode_identification(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append v20.9 after the exact finalized recurrent-mode v20.7 edge."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
        or not investigation.stages
    ):
        return None
    summary = _latest_complete(
        investigation, "openstar.tess.multimode.summarize")
    localization = _latest_complete(
        investigation, "openstar.tess.source-localization.analyze")
    physical = _latest_complete(
        investigation, "openstar.tess.physical.interpret")
    if summary is None or localization is None or physical is None:
        return None
    localization_result = localization.result or {}
    physical_result = physical.result or {}
    cross = localization_result.get("crossSector") or {}
    cycle = localization_result.get("physicalCycleEvidence") or {}
    physical_period = _validated_persisted_nested_cycle_period(cycle)
    try:
        reported_physical_period = float(
            physical_result.get("physicalPeriodDays"))
        reported_physical_harmonic = float(
            physical_result.get("photometricFirstHarmonicPeriodDays"))
        localized_period = float(
            localization_result.get("physicalPeriodDays"))
        localized_harmonic = float(
            localization_result.get("photometricFirstHarmonicPeriodDays"))
        eligible = int(cross.get("independentEligibleSectorCount"))
        required = int(cross.get("requiredIndependentSupportCount"))
        target_support = sorted(int(value) for value in (
            cross.get("targetSupportingSectors") or []))
        off_target = sorted(int(value) for value in (
            cross.get("offTargetSectors") or []))
        ambiguous = sorted(int(value) for value in (
            cross.get("ambiguousSectors") or []))
    except (TypeError, ValueError):
        return None
    independent = [
        item for item in localization_result.get("sectorResults") or []
        if isinstance(item, dict)
        and item.get("role") == "independent"
        and item.get("available") is True
    ]
    try:
        independent_sectors = [int(item["sector"]) for item in independent]
        observed_target = sorted(
            int(item["sector"]) for item in independent
            if item.get("classification") == "TARGET_CONSISTENT"
        )
        observed_off_target = sorted(
            int(item["sector"]) for item in independent
            if item.get("classification") == "OFF_TARGET"
        )
        observed_ambiguous = sorted(
            int(item["sector"]) for item in independent
            if item.get("classification") == "AMBIGUOUS"
        )
    except (KeyError, TypeError, ValueError):
        return None
    preparations = [
        stage for stage in investigation.stages
        if stage.handler_id == "openstar.tess.multimode.prepare"
        and stage.status == "COMPLETE"
    ]
    runs = [
        stage for stage in investigation.stages
        if stage.handler_id == "openstar.tess.multimode.run"
        and stage.status == "COMPLETE"
    ]
    interpretations = [
        stage for stage in investigation.stages
        if stage.handler_id == "openstar.tess.multimode.interpret"
        and stage.status == "COMPLETE"
    ]
    if physical_period is None:
        return None
    mode_evidence = validated_multimode_mode_evidence(
        summary.result,
        physical_period_days=physical_period,
        target_supporting_sectors=target_support,
        iteration_count=len(interpretations),
    )
    if mode_evidence is None:
        return None

    iteration_count = len(interpretations)
    if not (
        iteration_count >= 1
        and iteration_count <= 3
        and len(preparations) == iteration_count
        and len(runs) == iteration_count
        and physical_result.get("version")
        == "openstar.tess-physical-interpretation.v2"
        and localization_result.get("version")
        == "openstar.tess-pixel-localization.v1"
        and localization_result.get("physicalCycleEvidence")
        == physical_result.get("physicalCycleEvidence")
        and all(math.isfinite(value) and value > 0 for value in (
            reported_physical_period,
            reported_physical_harmonic,
            localized_period,
            localized_harmonic,
        ))
        and math.isclose(
            reported_physical_period, physical_period,
            rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            reported_physical_harmonic, physical_period / 2.0,
            rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            localized_period, physical_period,
            rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            localized_harmonic, physical_period / 2.0,
            rel_tol=1e-9, abs_tol=1e-12)
        and physical_result.get("physicalMechanismResolved") is False
        and (physical_result.get("contaminationScreen") or {}).get(
            "flaggedByExistingMetadata") is True
        and physical_result.get("recommendedNextTest")
        == "PIXEL_LEVEL_SOURCE_LOCALIZATION"
        and localization.parameters.get("evidenceLineage")
        == "PHYSICAL_INTERPRETATION_PIXEL_LOCALIZATION"
        and localization.triggered_by_stage_id == physical.id
        and cross.get("classification") == "TARGET_SOURCE_SUPPORTED"
        and cross.get("variableSignalOrigin") == "TARGET_CONSISTENT"
        and cross.get("recommendedNextTest")
        == "MULTI_MODE_FREQUENCY_DECOMPOSITION"
        and localization_result.get("recommendedNextTest")
        == "MULTI_MODE_FREQUENCY_DECOMPOSITION"
        and (localization_result.get("contaminationInterpretation") or {}).get(
            "existingCatalogContaminationCanBeCleared") is False
        and eligible == len(independent)
        and len(set(independent_sectors)) == eligible
        and eligible >= 3
        and required == max(3, eligible // 2 + 1)
        and (
            len(observed_target)
            + len(observed_off_target)
            + len(observed_ambiguous)
            == eligible
        )
        and len(target_support) >= required
        and len(set(target_support)) == len(target_support)
        and target_support == observed_target
        and off_target == observed_off_target
        and ambiguous == observed_ambiguous
        and set(mode_evidence["independentSectors"]).issubset(
            target_support)
        and summary.status == "COMPLETE"
        and not summary.stop
        and not any(
            stage.handler_id == "openstar.tess.mode-identification.analyze"
            for stage in investigation.stages
        )
    ):
        return None

    expected_multimode = []
    previous_interpretation = None
    for iteration, (preparation, run, interpretation) in enumerate(
        zip(preparations, runs, interpretations), start=1
    ):
        preparation_result = preparation.result or {}
        interpretation_result = interpretation.result or {}
        expected_trigger = (
            localization.id
            if previous_interpretation is None
            else previous_interpretation.id
        )
        if not (
            preparation.parameters == {"iteration": iteration}
            and preparation_result.get("iteration") == iteration
            and preparation.triggered_by_stage_id == expected_trigger
            and run.parameters.get("iteration") == iteration
            and run.triggered_by_stage_id == preparation.id
            and interpretation.parameters == {"iteration": iteration}
            and interpretation_result.get("iteration") == iteration
            and interpretation.triggered_by_stage_id == run.id
            and (
                iteration == iteration_count
                or interpretation_result.get("continueRecommended") is True
            )
        ):
            return None
        expected_multimode.extend((preparation, run, interpretation))
        previous_interpretation = interpretation
    final_interpretation = interpretations[-1]
    if (
        iteration_count < 3
        and (final_interpretation.result or {}).get(
            "continueRecommended") is not False
    ):
        return None
    if summary.triggered_by_stage_id != final_interpretation.id:
        return None
    expected_multimode.append(summary)

    source_finalize = next((
        stage for stage in investigation.stages
        if stage.handler_id == "openstar.tess.finalize"
        and stage.status == "COMPLETE"
        and stage.stop is True
        and stage.triggered_by_stage_id == localization.id
        and stage.parameters.get("outputSuffix") == "v20.6"
    ), None)
    latest = investigation.stages[-1]
    if source_finalize is None or not (
        latest.status == "COMPLETE"
        and latest.handler_id == "openstar.tess.finalize"
        and latest.stop is True
        and latest.triggered_by_stage_id == summary.id
        and latest.parameters.get("outputSuffix") == "v20.7"
    ):
        return None
    localization_index = investigation.stages.index(localization)
    first_prepare_index = investigation.stages.index(preparations[0])
    summary_index = investigation.stages.index(summary)
    if not (
        tuple(investigation.stages[
            localization_index + 1:first_prepare_index
        ]) == (source_finalize,)
        and tuple(investigation.stages[
            first_prepare_index:summary_index + 1
        ]) == tuple(expected_multimode)
        and tuple(investigation.stages[summary_index + 1:]) == (latest,)
    ):
        return None

    prefixes = [
        int(stage.id.partition("-")[0])
        for stage in investigation.stages
        if stage.id.partition("-")[0].isdigit()
    ]
    request = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-mode-identification",
        handler_id="openstar.tess.mode-identification.analyze",
        parameters={
            "evidenceLineage": MULTIMODE_MODE_EVIDENCE_LINEAGE,
        },
        triggered_by_stage_id=summary.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_V20_7_MULTI_MODE_IDENTIFICATION",
        },
    )


def _repair_unresolved_dynamic_localization_review_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Resume only the obsolete v20.11 unresolved-family preparation failure."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.residual-mode-localization-review.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error not in {
                "RuntimeError: v20.11 requires the morphology-resolved physical period.",
                "RuntimeError: v20.11 requires the completed v20.9 nonstationary model.",
                "RuntimeError: v20.11 requires a complete frozen residual-mode family.",
            }):
        return None
    scheduler_action = control.get("schedulerAction")
    if scheduler_action not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED"):
        return None
    selected = control.get("selectedExperiment")
    if selected is not None:
        if not isinstance(selected, dict):
            return None
        # The real persisted failures retain the failed request itself.  Do not
        # broaden recovery to an earlier request merely because it is an ancestor.
        selected_stage = next(
            (stage for stage in investigation.stages if stage.id == selected.get("id")),
            None,
        )
        if selected_stage is None or (
            selected.get("handler_id") != selected_stage.handler_id
            or not isinstance(selected.get("parameters"), dict)
            or selected.get("parameters") != selected_stage.parameters
            or selected.get("triggered_by_stage_id")
            != selected_stage.triggered_by_stage_id
        ):
            return None
        if selected_stage.id != failed.id:
            return None
    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    mode_result = (mode.result or {}) if mode else {}
    localization_result = (localization.result or {}) if localization else {}
    family_context = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode_result if mode else None,
    )
    if not (family_context is not None
            and localization
            and localization_result.get("recommendedNextTest")
                 == "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"):
        return None
    orders = list(family_context[1])
    localization_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    subtracted = ((localization_prepare.result or {}).get("subtractedHarmonicOrders")
                  if localization_prepare else None)
    rerun_localization = list(subtracted or [1, 2]) != orders
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=(f"{max(prefixes, default=0) + 1:03d}-" +
            ("prepare-residual-mode-localization" if rerun_localization
             else "prepare-residual-mode-localization-review")),
        handler_id=("openstar.tess.residual-mode-localization.prepare" if rerun_localization
                    else "openstar.tess.residual-mode-localization-review.prepare"),
        parameters={}, triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(continuation),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_UNRESOLVED_DYNAMIC_LOCALIZATION_REVIEW_COMPATIBILITY_RETRY"},
    )


def _repair_archive_timeout_localization_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Retry only a localization spawned by the obsolete stage-022 repair."""
    if investigation.status != "FAILED" or len(investigation.stages) < 2:
        return None
    failed = investigation.stages[-1]
    review_failure = investigation.stages[-2]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.residual-mode-localization.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error != (
                "RuntimeError: v20.10 could not prepare any residual-mode pixel datasets."
            )
            or failed.triggered_by_stage_id != review_failure.id
            or review_failure.status != "FAILED"
            or review_failure.handler_id
               != "openstar.tess.residual-mode-localization-review.prepare"
            or review_failure.failure_classification != "NON_RETRYABLE"
            or review_failure.error not in {
                "RuntimeError: v20.11 requires the morphology-resolved physical period.",
                "RuntimeError: v20.11 requires the completed v20.9 nonstationary model.",
            }):
        return None
    if control.get("schedulerAction") not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED"):
        return None
    selected = control.get("selectedExperiment")
    if not isinstance(selected, dict) or any((
        selected.get("id") != failed.id,
        selected.get("handler_id") != failed.handler_id,
        selected.get("parameters") != failed.parameters,
        selected.get("triggered_by_stage_id") != failed.triggered_by_stage_id,
    )):
        return None

    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    family = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode.result if mode else None,
    )
    if (family is None or localization is None
            or (localization.result or {}).get("recommendedNextTest")
               != "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"
            or review_failure.triggered_by_stage_id != localization.id):
        return None
    prior_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    if (prior_prepare is None
            or list((prior_prepare.result or {}).get("subtractedHarmonicOrders") or [1, 2])
               == list(family[1])):
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-residual-mode-localization",
        handler_id="openstar.tess.residual-mode-localization.prepare",
        parameters={}, triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={
            "branchAssessments": [], "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_ARCHIVE_TIMEOUT_LOCALIZATION_COMPATIBILITY_RETRY",
        },
    )


def _repair_unresolved_dynamic_multisource_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Retry only obsolete v20.12 frozen-family compatibility gates append-only."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.multi-source-residual.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error not in {
                "RuntimeError: v20.12 requires the morphology-resolved physical period.",
                "RuntimeError: v20.12 requires the completed v20.9 nonstationary model.",
                "RuntimeError: v20.12 requires either a morphology-resolved physical "
                "period or an established unresolved dynamic harmonic family.",
            }):
        return None
    if control.get("schedulerAction") not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED"):
        return None
    selected = control.get("selectedExperiment")
    if selected is not None:
        if (not isinstance(selected, dict)
                or selected.get("id") != failed.id
                or selected.get("handler_id") != failed.handler_id
                or selected.get("parameters") != failed.parameters
                or selected.get("triggered_by_stage_id")
                   != failed.triggered_by_stage_id):
            return None
    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    localization_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    localization_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.run"
    )
    review_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.prepare"
    )
    review_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.run"
    )
    review = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.interpret"
    )
    review_result = (review.result or {}) if review else {}
    cross = review_result.get("crossTime") or {}
    family = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode.result if mode else None,
    )
    valid_review_lineage = bool(
        localization_prepare and localization_run and localization
        and review_prepare and review_run and review
        and localization_run.triggered_by_stage_id == localization_prepare.id
        and localization.triggered_by_stage_id == localization_run.id
        and review_prepare.triggered_by_stage_id == localization.id
        and review_run.triggered_by_stage_id == review_prepare.id
        and review.triggered_by_stage_id == review_run.id
    )
    if not (morphology and (morphology.result or {}).get("physicalCycleResolved") is False
            and family is not None
            and review
            and valid_review_lineage
            and failed.triggered_by_stage_id == review.id
            and cross.get("classification")
                == "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND"
            and cross.get("residualModeOrigin") == "TIME_VARIABLE_OR_BLENDED"
            and review_result.get("recommendedNextTest")
                == "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-multi-source-residual",
        handler_id=failed.handler_id,
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(retry),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_UNRESOLVED_DYNAMIC_MULTISOURCE_COMPATIBILITY_RETRY"},
    )


def _repair_resolved_family_multisource_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append a retry for the obsolete resolved-without-v20.9 v20.12 gate."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.multi-source-residual.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error != (
                "RuntimeError: v20.12 requires the completed v20.9 nonstationary model."
            )):
        return None
    selected = control.get("selectedExperiment")
    expected_selected = asdict(StageRequest(
        failed.id, failed.handler_id, dict(failed.parameters),
        failed.triggered_by_stage_id,
    ))
    if (control.get("schedulerAction") not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED")
            or not isinstance(selected, dict)
            or selected != expected_selected):
        return None

    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    nonstationary = _latest_complete(investigation, "openstar.tess.nonstationary.summarize")
    localization_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    localization_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.run"
    )
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    review_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.prepare"
    )
    review_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.run"
    )
    review = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.interpret"
    )
    family = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode.result if mode else None,
    )
    review_result = (review.result or {}) if review else {}
    cross = review_result.get("crossTime") or {}
    valid_lineage = bool(
        localization_prepare and localization_run and localization
        and review_prepare and review_run and review
        and localization_run.triggered_by_stage_id == localization_prepare.id
        and localization.triggered_by_stage_id == localization_run.id
        and review_prepare.triggered_by_stage_id == localization.id
        and review_run.triggered_by_stage_id == review_prepare.id
        and review.triggered_by_stage_id == review_run.id
        and failed.triggered_by_stage_id == review.id
    )
    if not (morphology
            and (morphology.result or {}).get("physicalCycleResolved") is True
            and nonstationary is None
            and family is not None
            and family[3] == "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
            and valid_lineage
            and cross.get("classification")
                == "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND"
            and cross.get("residualModeOrigin") == "TIME_VARIABLE_OR_BLENDED"
            and review_result.get("recommendedNextTest")
                == "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-multi-source-residual",
        handler_id=failed.handler_id,
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(retry),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_RESOLVED_FAMILY_MULTISOURCE_COMPATIBILITY_RETRY"},
    )


def _repair_closed_file_independent_prepare(
    store: InvestigationStore, investigation: Investigation
) -> Investigation | None:
    """Retry only the known pre-lock Lightkurve closed-file durable failure."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if not (
        failed.status == "FAILED"
        and failed.handler_id == "openstar.tess.independent.prepare"
        and failed.error == "ValueError: I/O operation on closed file."
        and failed.failure_classification == "NON_RETRYABLE"
    ):
        return None
    required = {
        "openstar.tess.prepare-target",
        "openstar.tess.primary-project.run",
        "openstar.tess.catalog-identity",
        "openstar.tess.planner",
    }
    completed = {stage.handler_id for stage in investigation.stages
                 if stage.status == "COMPLETE"}
    if not required.issubset(completed):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-independent-sectors",
        handler_id="openstar.tess.independent.prepare",
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_LIGHTKURVE_CLOSED_FILE_COMPATIBILITY_RETRY",
        },
    )


def _repair_shifted_stage_lookup_independent_prepare(
    store: InvestigationStore, investigation: Investigation
) -> Investigation | None:
    """Retry the one obsolete numeric catalog-stage lookup failure shape."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if not (
        failed.status == "FAILED"
        and failed.handler_id == "openstar.tess.independent.prepare"
        and failed.failure_classification == "NON_RETRYABLE"
        and failed.error
        == "RuntimeError: Stage is not COMPLETE with a result: 003-catalog-identity"
    ):
        return None

    identity_failures = [
        index for index, stage in enumerate(investigation.stages)
        if stage.handler_id == "openstar.tess.catalog-identity"
        and stage.status == "FAILED"
        and stage.failure_classification == "TRANSIENT_INFRASTRUCTURE"
    ]
    successful_identity = next((
        index for index, stage in enumerate(investigation.stages)
        if stage.handler_id == "openstar.tess.catalog-identity"
        and stage.status == "COMPLETE"
        and any(failed_index < index for failed_index in identity_failures)
    ), None)
    hypotheses = next((
        index for index, stage in enumerate(investigation.stages)
        if successful_identity is not None and index > successful_identity
        and stage.handler_id == "openstar.tess.hypotheses" and stage.status == "COMPLETE"
    ), None)
    planner = next((
        index for index, stage in enumerate(investigation.stages)
        if hypotheses is not None and index > hypotheses
        and stage.handler_id == "openstar.tess.planner" and stage.status == "COMPLETE"
    ), None)
    failed_index = len(investigation.stages) - 1
    if planner is None or any(
        index > failed_index
        and stage.handler_id == "openstar.tess.independent.prepare"
        and stage.status == "COMPLETE"
        for index, stage in enumerate(investigation.stages)
    ):
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-independent-sectors",
        handler_id="openstar.tess.independent.prepare",
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_RETRY_SHIFTED_STAGE_LOOKUP_COMPATIBILITY_RETRY",
        },
    )


def _repair_binary_confirmation_time_origin_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append one retry for the exact primary time-origin schema failure."""
    expected_ids = (
        "001-prepare-target",
        "002-primary-distributed-search",
        "003-catalog-identity",
        "004-hypotheses",
        "005-planner",
        "006-prepare-independent-sectors",
        "007-run-independent-sectors",
        "008-interpret-independent-sectors",
        "009-prepare-broad-independent-search",
        "010-run-broad-independent-search",
        "011-interpret-broad-independent-search",
        "012-characterize-variability",
        "013-periodic-event-screen",
    )
    expected_handlers = (
        "openstar.tess.prepare-target",
        "openstar.tess.primary-project.run",
        "openstar.tess.catalog-identity",
        "openstar.tess.hypotheses",
        "openstar.tess.planner",
        "openstar.tess.independent.prepare",
        "openstar.tess.independent.run",
        "openstar.tess.independent.interpret",
        "openstar.tess.independent.broad.prepare",
        "openstar.tess.independent.broad.run",
        "openstar.tess.independent.broad.interpret",
        "openstar.tess.morphology.analyze",
        "openstar.tess.binary-confirmation.analyze",
    )
    stages = investigation.stages
    if not (
        investigation.status == "FAILED"
        and len(stages) == len(expected_ids)
        and tuple(stage.id for stage in stages) == expected_ids
        and tuple(stage.handler_id for stage in stages) == expected_handlers
        and all(stage.status == "COMPLETE" for stage in stages[:-1])
        and stages[-1].status == "FAILED"
        and stages[0].triggered_by_stage_id is None
        and all(
            stages[index].triggered_by_stage_id == stages[index - 1].id
            for index in range(1, len(stages))
        )
        and all(
            store.verified_terminal_stage_ledger_hash(investigation.id, stage)
            for stage in stages
        )
    ):
        return None

    failed = stages[-1]
    if not (
        failed.parameters == {"entryMode": MORPHOLOGY_EVENT_SCREEN_ENTRY}
        and failed.result is None
        and not failed.artifacts
        and failed.error == "ValueError: frozen dataset lacks originalTimeOriginDays"
        and failed.failure_classification == "NON_RETRYABLE"
        and failed.stop is False
    ):
        return None

    if control.get("schedulerAction") not in {
        "RUN_EXPERIMENT",
        "INVESTIGATION_FAILED",
    }:
        return None
    selected = control.get("selectedExperiment")
    if selected is not None and selected != {
        "id": failed.id,
        "handler_id": failed.handler_id,
        "parameters": failed.parameters,
        "triggered_by_stage_id": failed.triggered_by_stage_id,
    }:
        return None

    prepared = stages[0].result or {}
    independent = stages[5].result or {}
    morphology = stages[11].result or {}
    if not (
        morphology_event_screening_continuation(morphology, independent)
        and _verified_stage_json(stages[11], "morphology-v20.4.json")
    ):
        return None

    try:
        primary_path = Path(str(prepared["datasetPath"])).expanduser().resolve()
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        primary_hash = sha256_file(primary_path)
        source = primary.get("source") or {}
        metadata = primary.get("metadata") or {}
        metadata["originalTimeOriginDays"]
        binary_confirmation_time_origin(primary)
        primary_sector = binary_confirmation_dataset_sector(primary)
        if (
            primary_sector != int(prepared["sector"])
            or int(metadata["ticID"]) != int(prepared["ticID"])
        ):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if "originalTimeOriginDays" in source:
        return None

    prepare_hashes = stages[0].provenance.input_hashes if stages[0].provenance else {}
    morphology_hashes = (
        stages[11].provenance.input_hashes if stages[11].provenance else {}
    )
    if not (
        prepare_hashes.get("sourceDataset") == primary_hash
        and morphology_hashes.get("primaryDataset") == primary_hash
    ):
        return None

    prepared_sectors = independent.get("preparedSectors") or []
    sector_ids: set[int] = set()
    if len(prepared_sectors) < 3:
        return None
    for item in prepared_sectors:
        try:
            sector = int(item["sector"])
            path = Path(str(item["datasetPath"])).expanduser().resolve()
            expected_hash = morphology_hashes[f"independentSector{sector}"]
            dataset = json.loads(path.read_text(encoding="utf-8"))
            binary_confirmation_time_origin(dataset)
            dataset_sector = binary_confirmation_dataset_sector(dataset)
            if (
                sector <= 0
                or sector in sector_ids
                or dataset_sector != sector
                or sha256_file(path) != expected_hash
            ):
                return None
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None
        sector_ids.add(sector)

    retry = StageRequest(
        id="014-periodic-event-screen-recovery",
        handler_id=failed.handler_id,
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_BINARY_CONFIRMATION_TIME_ORIGIN_COMPATIBILITY_RETRY",
        },
    )


def _repair_residual_phase_difference_imaging_terminal_handoff(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Reopen the exact prepare-to-run handoff hidden by the old planner."""
    if not (
        investigation.status == "COMPLETE"
        and control == {
            "branchAssessments": [],
            "selectedExperiment": None,
            "schedulerAction": "INVESTIGATION_COMPLETE",
        }
        and len(investigation.stages) >= 3
        and not any(stage.status == "RUNNING" for stage in investigation.stages)
    ):
        return None

    bridge = next((
        stage for stage in reversed(investigation.stages[:-1])
        if stage.id == "033-prepare-catalog-guided-source-localization"
        and stage.handler_id
        == "openstar.tess.catalog-guided-source-localization.prepare"
        and stage.status == "COMPLETE"
    ), None)
    localization = next((
        stage for stage in reversed(investigation.stages[:-1])
        if stage.id == "035-interpret-catalog-guided-source-localization"
        and stage.handler_id
        == "openstar.tess.catalog-guided-source-localization.interpret"
        and stage.status == "COMPLETE"
    ), None)
    preparation = investigation.stages[-1]
    expected_run = {
        "id": "037-run-residual-phase-difference-imaging",
        "handler_id": "openstar.tess.residual-phase-difference-imaging.run",
        "parameters": {},
        "triggered_by_stage_id": "036-prepare-residual-phase-difference-imaging",
    }
    if not (
        bridge is not None
        and localization is not None
        and preparation.id == "036-prepare-residual-phase-difference-imaging"
        and preparation.handler_id
        == "openstar.tess.residual-phase-difference-imaging.prepare"
        and preparation.status == "COMPLETE"
        and preparation.triggered_by_stage_id == localization.id
        and preparation.stop is False
        and preparation.next_stage == expected_run
        and not any(stage.id == expected_run["id"] for stage in investigation.stages)
        and store.verified_terminal_stage_ledger_hash(investigation.id, bridge)
        and store.verified_terminal_stage_ledger_hash(investigation.id, localization)
        and store.verified_terminal_stage_ledger_hash(investigation.id, preparation)
        and _verified_stage_json(preparation, "preparation.json")
    ):
        return None

    result = preparation.result or {}
    localization_result = localization.result or {}
    hashes = preparation.provenance.input_hashes if preparation.provenance else {}
    if not (
        result.get("version")
        == "openstar.tess-residual-phase-difference-imaging-preparation.v1"
        and result.get("execution")
        == "coordinator-local-difference-image-centroiding"
        and result.get("physicalCycleResolved") is False
        and localization_result.get("classification") == "UNRESOLVED"
        and localization_result.get("sourceAttributionResolved") is False
        and localization_result.get("recommendedNextTest")
        == "ADDITIONAL_SOURCE_LOCALIZATION_DATA"
        and hashes.get("catalogGuidedPreparation") == sha256_json(bridge.result)
        and hashes.get("catalogGuidedInterpretation")
        == sha256_json(localization.result)
    ):
        return None

    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": expected_run,
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_RESIDUAL_PHASE_DIFFERENCE_IMAGING_HANDOFF",
        },
    )


def repair_obsolete_terminal_wait(
    store: InvestigationStore,
    investigation: Investigation,
    *,
    historical_path_resolver: HistoricalPathResolver | None = None,
) -> Investigation:
    """Repair only known obsolete terminal TESS decisions from older code."""

    control = investigation.metadata.get("controlState")
    if investigation.workflow_id != WORKFLOW_ID or not isinstance(control, dict):
        return investigation

    resolver = historical_path_resolver or NO_HISTORICAL_PATH_RELOCATION
    benchmark_repair = _repair_known_target_blind_full_characterization(
        store, investigation, control
    )
    if benchmark_repair is not None:
        return benchmark_repair
    family_recovery = _recover_failed_v2014_family_ledger_compatibility(
        store, investigation, historical_path_resolver=resolver)
    if family_recovery is not None:
        return family_recovery
    recurrence = _continue_finalized_main_family_recurrence(
        store, investigation, control, historical_path_resolver=resolver)
    if recurrence is not None:
        return recurrence
    frequency_reassessment = _continue_finalized_main_family_frequency_reassessment(
        store, investigation, control, historical_path_resolver=resolver)
    if frequency_reassessment is not None:
        return frequency_reassessment
    astrophysical = _continue_finalized_v2014_astrophysical_interpretation(
        store, investigation, control, historical_path_resolver=resolver)
    if astrophysical is not None:
        return astrophysical
    recovered = _recover_stage052_additional_sectors(store, investigation, control)
    if recovered is not None:
        return recovered
    multisector = _continue_finalized_v2018_multisector_source(
        store, investigation, control, historical_path_resolver=resolver)
    if multisector is not None:
        return multisector
    pixel = _continue_finalized_v2017_pixel_recurrence(
        store, investigation, control, historical_path_resolver=resolver)
    if pixel is not None:
        return pixel
    archival = _continue_finalized_v2016_archival_baseline(
        store, investigation, control, historical_path_resolver=resolver)
    if archival is not None:
        return archival

    old_mechanism = _old_target_residual_adjudication_boundary(investigation)
    if old_mechanism is not None:
        request = StageRequest(
            id=_continuation_stage_id(old_mechanism, "target-residual-mechanism-adjudication"),
            handler_id="openstar.tess.target-residual-mechanism-adjudication.analyze",
            parameters={}, triggered_by_stage_id=old_mechanism.id,
        )
        return store.set_control_state(
            investigation, status="RUNNING",
            control_state={"branchAssessments": [], "selectedExperiment": asdict(request),
                           "schedulerAction": "RUN_EXPERIMENT",
                           "recovery": "TESS_V20_14_ROUTE_ADJUDICATION_V20_15"},
        )

    adjudication = _v2015_predictive_validation_boundary(investigation)
    if adjudication is not None:
        request = StageRequest(id="030-target-residual-mechanism-predictive-validation",
            handler_id="openstar.tess.target-residual-mechanism-predictive-validation.analyze",
            parameters={}, triggered_by_stage_id=adjudication.id)
        return store.set_control_state(investigation, status="RUNNING", control_state={
            "branchAssessments": [], "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_V20_15_PREDICTIVE_VALIDATION_V20_16"})

    mechanism = _target_residual_mechanism_boundary(investigation)
    if (mechanism is not None and investigation.status == "COMPLETE"
            and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"):
        request = StageRequest(
            id=_continuation_stage_id(mechanism, "target-residual-mechanism"),
            handler_id="openstar.tess.target-residual-mechanism.analyze",
            parameters={}, triggered_by_stage_id=mechanism.id,
        )
        return store.set_control_state(
            investigation, status="RUNNING",
            control_state={"branchAssessments": [], "selectedExperiment": asdict(request),
                           "schedulerAction": "RUN_EXPERIMENT",
                           "recovery": "TESS_V20_13_TARGET_MECHANISM_CONTINUATION"},
        )

    intrinsic = _intrinsic_target_boundary(investigation)
    if (intrinsic is not None and investigation.status == "COMPLETE"
            and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"):
        request = StageRequest(
            id=_continuation_stage_id(intrinsic, "classify-intrinsic-target-residual"),
            handler_id="openstar.tess.intrinsic-nonstationary.analyze",
            parameters={}, triggered_by_stage_id=intrinsic.id,
        )
        return store.set_control_state(
            investigation, status="RUNNING",
            control_state={"branchAssessments": [], "selectedExperiment": asdict(request),
                           "schedulerAction": "RUN_EXPERIMENT",
                           "recovery": "TESS_V20_12_TARGET_INTRINSIC_CONTINUATION"},
        )

    prf_transport_repair = _repair_official_prf_transport_terminal(
        store, investigation, control
    )
    if prf_transport_repair is not None:
        return prf_transport_repair

    archive_timeout_repair = _repair_archive_timeout_localization_failure(
        store, investigation, control
    )
    if archive_timeout_repair is not None:
        return archive_timeout_repair

    localization_review_repair = _repair_unresolved_dynamic_localization_review_failure(
        store, investigation, control
    )
    if localization_review_repair is not None:
        return localization_review_repair

    multisource_repair = _repair_resolved_family_multisource_failure(
        store, investigation, control
    )
    if multisource_repair is not None:
        return multisource_repair

    multisource_repair = _repair_unresolved_dynamic_multisource_failure(
        store, investigation, control
    )
    if multisource_repair is not None:
        return multisource_repair

    independent_repair = _repair_shifted_stage_lookup_independent_prepare(
        store, investigation
    )
    if independent_repair is not None:
        return independent_repair

    independent_repair = _repair_closed_file_independent_prepare(store, investigation)
    if independent_repair is not None:
        return independent_repair

    binary_confirmation_repair = _repair_binary_confirmation_time_origin_failure(
        store, investigation, control
    )
    if binary_confirmation_repair is not None:
        return binary_confirmation_repair

    phase_difference_handoff = (
        _repair_residual_phase_difference_imaging_terminal_handoff(
            store, investigation, control
        )
    )
    if phase_difference_handoff is not None:
        return phase_difference_handoff

    catalog_repair = _repair_catalog_timeout_terminal(store, investigation, control)
    if catalog_repair is not None:
        return catalog_repair

    period_repair = _repair_promoted_period_characterization_terminal(
        store, investigation, control
    )
    if period_repair is not None:
        return period_repair

    mode_continuation = _continue_finalized_multimode_mode_identification(
        store, investigation, control)
    if mode_continuation is not None:
        return mode_continuation

    multimode_continuation = _continue_finalized_source_localization_multimode(
        store, investigation, control)
    if multimode_continuation is not None:
        return multimode_continuation

    physical_localization = _continue_finalized_physical_source_localization(
        store, investigation, control)
    if physical_localization is not None:
        return physical_localization

    nested_cycle_continuation = \
        _continue_finalized_nested_cycle_physical_interpretation(
            store, investigation, control)
    if nested_cycle_continuation is not None:
        return nested_cycle_continuation

    nested_alias_repair = _repair_unmatched_alias_model_terminal(
        store, investigation, control)
    if nested_alias_repair is not None:
        return nested_alias_repair

    unresolved_dynamic_repair = \
        _repair_unresolved_family_dynamic_harmonic_terminal(
            store, investigation, control)
    if unresolved_dynamic_repair is not None:
        return unresolved_dynamic_repair

    mode_repair = _repair_mode_identification_terminal(store, investigation, control)
    if mode_repair is not None:
        return mode_repair

    v20_8_long_baseline_repair = (
        _repair_v20_8_long_baseline_time_frequency_confirmation_terminal(
            store, investigation, control
        )
    )
    if v20_8_long_baseline_repair is not None:
        return v20_8_long_baseline_repair

    transient_mode_repair = _repair_transient_mode_validation_terminal(
        store, investigation, control
    )
    if transient_mode_repair is not None:
        return transient_mode_repair

    v20_8_mode_repair = (
        _repair_v20_8_confirmed_coherent_mode_identification_terminal(
            store, investigation, control
        )
    )
    if v20_8_mode_repair is not None:
        return v20_8_mode_repair

    long_baseline_repair = \
        _repair_long_baseline_frequency_confirmation_terminal(
            store, investigation, control
        )
    if long_baseline_repair is not None:
        return long_baseline_repair

    confirmed_nonstationary_repair = _repair_confirmed_nonstationary_terminal(
        store, investigation, control
    )
    if confirmed_nonstationary_repair is not None:
        return confirmed_nonstationary_repair

    confirmed_localization_repair = \
        _repair_confirmed_residual_localization_terminal(
            store, investigation, control
        )
    if confirmed_localization_repair is not None:
        return confirmed_localization_repair

    residual_external_repair = \
        _repair_target_supported_residual_external_evidence_terminal(
            store, investigation, control
        )
    if residual_external_repair is not None:
        return residual_external_repair

    residual_mechanism_repair = \
        _repair_target_residual_astrophysical_mechanism_terminal(
            store, investigation, control
        )
    if residual_mechanism_repair is not None:
        return residual_mechanism_repair

    neighbor_catalog_review_repair = \
        _repair_neighbor_catalog_pixel_response_review_terminal(
            store, investigation, control
        )
    if neighbor_catalog_review_repair is not None:
        return neighbor_catalog_review_repair

    dynamic_repair = _repair_dynamic_harmonic_terminal(store, investigation, control)
    if dynamic_repair is not None:
        return dynamic_repair

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_noirlab_adapter(investigation)
    ):
        metadata = investigation.metadata or {}
        target = InvestigationTarget(
            id=str(metadata.get("datasetID") or investigation.id),
            investigation_id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            metadata=dict(metadata),
        )
        repaired, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target)
        )
        return repaired

    if (
        (
            investigation.status == "BLOCKED"
            and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
            and (
                _awaiting_atlas_signed_reanalysis_adapter(investigation)
                or _awaiting_post_atlas_targeted_observation_adapter(investigation)
            )
        )
        or (
            investigation.status == "COMPLETE"
            and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
            and _awaiting_post_atlas_targeted_observation_adapter(investigation)
        )
    ):
        metadata = investigation.metadata or {}
        target = InvestigationTarget(
            id=str(metadata.get("datasetID") or investigation.id),
            investigation_id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            metadata=dict(metadata),
        )
        repaired, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target)
        )
        return repaired

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_current_atlas_adapter(investigation)
    ):
        metadata = investigation.metadata or {}
        target = InvestigationTarget(
            id=str(metadata.get("datasetID") or investigation.id),
            investigation_id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            metadata=dict(metadata),
        )
        repaired, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target)
        )
        control_state = dict(repaired.metadata.get("controlState") or {})
        if control_state.get("schedulerAction") == "WAIT_FOR_PREREQUISITES":
            branch = plan_tess_branches(repaired, target)[0]
            control_state["selectedExperiment"] = asdict(branch.experiment)
            control_state["missingPrerequisites"] = list(branch.required_stage_ids)
            repaired = store.set_control_state(
                repaired, status="BLOCKED", control_state=control_state
            )
        return repaired

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_nsc_adapter(investigation)
    ):
        metadata = investigation.metadata or {}
        target = InvestigationTarget(
            id=str(metadata.get("datasetID") or investigation.id),
            investigation_id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            metadata=dict(metadata),
        )
        repaired, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target)
        )
        return repaired

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_current_des_adapter(investigation)
    ):
        metadata = investigation.metadata or {}
        target = InvestigationTarget(
            id=str(metadata.get("datasetID") or investigation.id),
            investigation_id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            metadata=dict(metadata),
        )
        repaired, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target)
        )
        return repaired

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _has_terminal_tess_evidence(investigation)
        and not _awaiting_nsc_adapter(investigation)
        and not _awaiting_noirlab_adapter(investigation)
        and not _awaiting_current_atlas_adapter(investigation)
        and not _awaiting_atlas_signed_reanalysis_adapter(investigation)
        and not _awaiting_post_atlas_targeted_observation_adapter(investigation)
        and not _is_unresolved_atlas_targeted_observation_boundary(investigation)
    ):
        repaired, _ = AutonomousInvestigationEngine(store).decide(investigation, ())
        return repaired

    gaia_continuation = _persisted_archive_continuation(investigation)
    if (
        investigation.status == "COMPLETE"
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and gaia_continuation is not None
    ):
        metadata = investigation.metadata or {}
        target = InvestigationTarget(
            id=str(metadata.get("datasetID") or investigation.id),
            investigation_id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            metadata=dict(metadata),
        )
        repaired, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target)
        )
        return repaired

    offset_variability = _latest_complete(
        investigation, "openstar.tess.offset-source-variability.interpret"
    )
    gaia_attempted = any(
        stage.handler_id.startswith(
            "openstar.tess.gaia-source-resolved-counterpart-photometry."
        )
        for stage in investigation.stages
    )
    offset_result = (offset_variability.result or {}) if offset_variability else {}
    gaia_interpretation = _latest_complete(
        investigation, "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret"
    )
    skymapper_attempted = any(stage.handler_id.startswith(
        "openstar.tess.skymapper-resolved-counterpart-photometry."
    ) for stage in investigation.stages)
    gaia_result = (gaia_interpretation.result or {}) if gaia_interpretation else {}
    obsolete_gaia_terminal = (
        gaia_interpretation is not None
        and gaia_result.get("recommendedNextTest") == "SKYMAPPER_RESOLVED_COUNTERPART_PHOTOMETRY"
        and gaia_result.get("physicalMechanismResolved") is False
        and not skymapper_attempted
    )
    obsolete_skymapper_terminal = _awaiting_nsc_adapter(investigation)
    if not (
        investigation.status == "COMPLETE"
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and (obsolete_gaia_terminal or obsolete_skymapper_terminal or (
            offset_variability is not None
            and offset_result.get("recommendedNextTest")
            == "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
            and offset_result.get("physicalMechanismResolved") is False
            and not gaia_attempted
        ))
    ):
        return investigation

    metadata = investigation.metadata or {}
    target = InvestigationTarget(
        id=str(metadata.get("datasetID") or investigation.id),
        investigation_id=investigation.id,
        workflow_id=investigation.workflow_id,
        workflow_version=investigation.workflow_version,
        metadata=dict(metadata),
    )
    repaired, _ = AutonomousInvestigationEngine(store).decide(
        investigation, plan_tess_branches(investigation, target)
    )
    return repaired


def plan_tess_branches(
    investigation: Investigation, target: InvestigationTarget
) -> tuple[ScientificBranch, ...]:
    """Translate persisted TESS evidence into domain-neutral branch declarations."""

    mechanism = _target_residual_mechanism_boundary(investigation)
    if mechanism is not None:
        return (ScientificBranch(
            id=f"target-residual-mechanism-after-{mechanism.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(mechanism, "target-residual-mechanism"),
                handler_id="openstar.tess.target-residual-mechanism.analyze",
                parameters={}, triggered_by_stage_id=mechanism.id,
            ),
        ),)

    intrinsic = _intrinsic_target_boundary(investigation)
    if intrinsic is not None:
        return (ScientificBranch(
            id=f"intrinsic-target-residual-after-{intrinsic.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(intrinsic, "classify-intrinsic-target-residual"),
                handler_id="openstar.tess.intrinsic-nonstationary.analyze",
                parameters={}, triggered_by_stage_id=intrinsic.id,
            ),
        ),)

    prf_interpretation = _latest_complete(
        investigation, "openstar.tess.official-spoc-prf-forward-modeling.interpret"
    )
    catalog_identification = _latest_complete(
        investigation, "openstar.tess.catalog-counterpart-identification.analyze"
    )
    catalog_guided_localization = _latest_complete(
        investigation, "openstar.tess.catalog-guided-source-localization.interpret"
    )
    residual_phase_difference_image = _latest_complete(
        investigation, "openstar.tess.residual-phase-difference-imaging.interpret"
    )
    source_switching_temporal = _latest_complete(
        investigation, "openstar.tess.source-switching-temporal-model.interpret"
    )
    time_resolved_localization = _latest_complete(
        investigation, "openstar.tess.time-resolved-residual-phase-localization.interpret"
    )
    time_resolved_frequency = _latest_complete(
        investigation, "openstar.tess.time-resolved-frequency-localization.interpret"
    )
    time_resolved_frequency_bridge = _latest_complete(
        investigation, "openstar.tess.time-resolved-frequency-localization.prepare"
    )
    catalog_identity = _latest_complete(investigation, "openstar.tess.catalog-identity")
    variability_interpretation = _latest_complete(
        investigation, "openstar.tess.offset-source-variability.interpret"
    )
    gaia_interpretation = _latest_complete(
        investigation,
        "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
    )
    nsc_interpretation = _latest_complete(
        investigation, "openstar.tess.nsc-resolved-photometry.interpret"
    )
    atlas_submission = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.prepare"
    )
    atlas_collection = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.collect"
    )
    if (atlas_submission is not None and atlas_collection is None
            and atlas_submission.next_stage is None):
        dependency_id = str((atlas_submission.result or {}).get("externalDependencyID") or "")
        available = (investigation.metadata.get("externalDataAvailability") or {}).get(dependency_id) is True
        return (ScientificBranch(
            id=f"collect-atlas-after-{atlas_submission.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(atlas_submission, "collect-atlas-forced-photometry"),
                handler_id="openstar.tess.atlas-forced-photometry.collect", parameters={},
                triggered_by_stage_id=atlas_submission.id),
            external_data=(ExternalDataDependency(dependency_id, available,
                None if available else "The two persisted ATLAS jobs are still pending."),),
        ),)
    if _awaiting_current_atlas_adapter(investigation):
        from .tess_atlas_forced_photometry import atlas_credentials_available
        des = _latest_complete(
            investigation, "openstar.tess.des-dr2-se-local-forced-photometry.interpret"
        )
        return (
            ScientificBranch(
                id=f"await-current-atlas-adapter-after-{des.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(des, "prepare-atlas-forced-photometry"),
                    handler_id="openstar.tess.atlas-forced-photometry.prepare",
                    parameters={},
                    triggered_by_stage_id=des.id,
                ),
                required_stage_ids=() if atlas_credentials_available() else (
                    "openstar.capability.atlas-forced-photometry-credentials",
                ),
            ),
        )
    if _awaiting_atlas_signed_reanalysis_adapter(investigation):
        atlas = _latest_complete(
            investigation, "openstar.tess.atlas-forced-photometry.interpret"
        )
        return (
            ScientificBranch(
                id=f"await-current-atlas-signed-reanalysis-adapter-after-{atlas.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        atlas, "prepare-atlas-forced-photometry-reanalysis"
                    ),
                    handler_id="openstar.tess.atlas-forced-photometry-reanalysis.prepare",
                    parameters={}, triggered_by_stage_id=atlas.id,
                ),
                required_stage_ids=(
                    "openstar.capability.current-atlas-signed-reanalysis-adapter",
                ),
            ),
        )
    if _awaiting_post_atlas_targeted_observation_adapter(investigation):
        atlas = _latest_complete(
            investigation, "openstar.tess.atlas-forced-photometry.interpret"
        )
        return (
            ScientificBranch(
                id=f"await-current-targeted-observation-adapter-after-{atlas.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        atlas, "generate-targeted-observation-plan"
                    ),
                    handler_id="openstar.tess.targeted-observation-planning.generate",
                    parameters={}, triggered_by_stage_id=atlas.id,
                ),
                required_stage_ids=(),
            ),
        )
    if _awaiting_current_des_adapter(investigation):
        noirlab = _latest_complete(
            investigation, "openstar.tess.noirlab-image-forced-photometry.interpret"
        )
        pair = ((noirlab.result or {}).get("sourcePair") or {})
        adapter_ready = pair.get("version") == "openstar.current-source-pair.v1"
        return (
            ScientificBranch(
                id=f"await-current-des-adapter-after-{noirlab.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        noirlab, "prepare-des-dr2-single-epoch-local-forced-photometry"
                    ),
                    handler_id="openstar.tess.des-dr2-se-local-forced-photometry.prepare",
                    parameters={}, triggered_by_stage_id=noirlab.id,
                ),
                required_stage_ids=() if adapter_ready else (
                    "openstar.capability.current-des-source-pair-adapter",
                ),
            ),
        )
    if _awaiting_noirlab_adapter(investigation):
        return (
            ScientificBranch(
                id=f"continue-noirlab-after-{nsc_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        nsc_interpretation, "prepare-noirlab-image-level-forced-photometry"
                    ),
                    handler_id="openstar.tess.noirlab-image-forced-photometry.prepare",
                    parameters={}, triggered_by_stage_id=nsc_interpretation.id,
                ),
                required_stage_ids=(),
            ),
        )
    archive_continuation = _persisted_archive_continuation(investigation)
    if archive_continuation is not None:
        completed, raw = archive_continuation
        return (
            ScientificBranch(
                id=f"continue-after-{completed.id}",
                experiment=_request_from_persisted(raw),
            ),
        )

    skymapper_interpretation = _latest_complete(
        investigation, "openstar.tess.skymapper-resolved-counterpart-photometry.interpret"
    )
    skymapper_started = any(stage.handler_id.startswith(
        "openstar.tess.skymapper-resolved-counterpart-photometry."
    ) for stage in investigation.stages)
    gaia_result = (gaia_interpretation.result or {}) if gaia_interpretation else {}
    if _awaiting_nsc_adapter(investigation):
        return (
            ScientificBranch(
                id=f"continue-nsc-after-{skymapper_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        skymapper_interpretation,
                        "prepare-nsc-resolved-counterpart-photometry",
                    ),
                    handler_id="openstar.tess.nsc-resolved-photometry.prepare",
                    parameters={},
                    triggered_by_stage_id=skymapper_interpretation.id,
                ),
            ),
        )
    if (gaia_interpretation is not None and skymapper_interpretation is None
            and not skymapper_started
            and gaia_result.get("recommendedNextTest") == "SKYMAPPER_RESOLVED_COUNTERPART_PHOTOMETRY"
            and gaia_result.get("physicalMechanismResolved") is False):
        return (ScientificBranch(
            id=f"continue-skymapper-after-{gaia_interpretation.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(
                    gaia_interpretation, "prepare-skymapper-resolved-counterpart-photometry"
                ),
                handler_id="openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
                parameters={}, triggered_by_stage_id=gaia_interpretation.id,
            ),
        ),)
    if gaia_interpretation is not None:
        return ()

    if (
        prf_interpretation is not None
        and catalog_identification is None
        and (prf_interpretation.result or {}).get("recommendedNextTest")
        == "CATALOG_COUNTERPART_IDENTIFICATION"
        and (prf_interpretation.result or {}).get("physicalMechanismResolved") is False
    ):
        return (
            ScientificBranch(
                id=f"continue-catalog-after-{prf_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        prf_interpretation, "catalog-counterpart"
                    ),
                    handler_id="openstar.tess.catalog-counterpart-identification.analyze",
                    parameters={},
                    triggered_by_stage_id=prf_interpretation.id,
                ),
            ),
        )

    catalog_result = (catalog_identification.result or {}) if catalog_identification else {}
    # A completed stage 056 is authoritative over the stage-053 continuation.
    # Recreate its direct candidate-validation request after restart, exactly once.
    if time_resolved_frequency is not None:
        result = time_resolved_frequency.result or {}
        candidate = result.get("preferredCandidate") or {}; ids = candidate.get("catalogIDs") or {}
        justified = candidate.get("raDeg") is not None and candidate.get("decDeg") is not None and (
            ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None)
        started = any(stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (not started and justified and result.get("classification") in {
                "STABLE_CANDIDATE_1_LOCALIZATION", "STABLE_CANDIDATE_2_LOCALIZATION"}
                and result.get("recommendedNextTest") ==
                "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"):
            return (ScientificBranch(id=f"continue-counterpart-variability-after-{time_resolved_frequency.id}",
                experiment=StageRequest(id=_continuation_stage_id(time_resolved_frequency,
                    "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare", parameters={},
                    triggered_by_stage_id=time_resolved_frequency.id)),)
        additional_started = any(stage.handler_id.startswith(_ADDITIONAL_SECTOR_PREFIX)
                                 for stage in investigation.stages)
        if (not additional_started and time_resolved_frequency_bridge is not None
                and catalog_identity is not None):
            from .tess_additional_sector_source_localization import boundary_authorized, bridge_is_complete, unused_official_sectors
            if (boundary_authorized(result)
                    and bridge_is_complete(time_resolved_frequency_bridge.result or {})
                    and unused_official_sectors(
                    catalog_identity.result or {}, time_resolved_frequency_bridge.result or {})):
                return (ScientificBranch(id=f"continue-additional-sectors-after-{time_resolved_frequency.id}",
                    experiment=StageRequest(id=_continuation_stage_id(time_resolved_frequency,
                        "prepare-additional-sector-source-localization"),
                        handler_id=_ADDITIONAL_SECTOR_PREFIX + "prepare", parameters={},
                        triggered_by_stage_id=time_resolved_frequency.id)),)
        return ()
    if time_resolved_localization is not None:
        localization_result = time_resolved_localization.result or {}
        followup_started = any(
            stage.handler_id.startswith("openstar.tess.time-resolved-frequency-localization.")
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (time_resolved_frequency is None and not followup_started
                and localization_result.get("classification") == "TIME_VARIABLE_LOCALIZATION"
                and localization_result.get("sourceAttributionResolved") is False
                and localization_result.get("physicalMechanismResolved") is False
                and localization_result.get("recommendedNextTest")
                == "TIME_VARIABLE_SOURCE_LOCALIZATION_FOLLOWUP"):
            return (ScientificBranch(
                id=f"continue-frequency-localization-after-{time_resolved_localization.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        time_resolved_localization, "prepare-time-resolved-frequency-localization"),
                    handler_id="openstar.tess.time-resolved-frequency-localization.prepare", parameters={},
                    triggered_by_stage_id=time_resolved_localization.id)),)
        candidate = localization_result.get("preferredCandidate") or {}; ids = candidate.get("catalogIDs") or {}
        validation_started = any(stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (not validation_started and candidate.get("raDeg") is not None
                and candidate.get("decDeg") is not None
                and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None)
                and localization_result.get("classification") in {
                    "STABLE_CANDIDATE_1_LOCALIZATION", "STABLE_CANDIDATE_2_LOCALIZATION"}
                and localization_result.get("sourceAttributionResolved") is True
                and localization_result.get("recommendedNextTest") ==
                "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"):
            return (ScientificBranch(id=f"continue-counterpart-variability-after-{time_resolved_localization.id}",
                experiment=StageRequest(id=_continuation_stage_id(time_resolved_localization,
                    "prepare-offset-source-variability"), handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={}, triggered_by_stage_id=time_resolved_localization.id)),)
        return ()
    # Stage 053 supersedes stage 050 for a resolved temporal candidate.  Check it
    # before the stage-050 terminal guard so restart recovery remains reachable.
    if source_switching_temporal is not None:
        temporal_result = source_switching_temporal.result or {}
        localization_started = any(
            stage.handler_id.startswith("openstar.tess.time-resolved-residual-phase-localization.")
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (time_resolved_localization is None and not localization_started
                and temporal_result.get("classification") == "SECTOR_VARIABLE_MULTI_SOURCE"
                and temporal_result.get("sourceIdentifiable") is True
                and temporal_result.get("sourceAttributionResolved") is False
                and temporal_result.get("physicalMechanismResolved") is False
                and temporal_result.get("recommendedNextTest")
                == "ADDITIONAL_SOURCE_LOCALIZATION_DATA"):
            return (ScientificBranch(
                id=f"continue-time-resolved-localization-after-{source_switching_temporal.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(source_switching_temporal,
                        "prepare-time-resolved-residual-phase-localization"),
                    handler_id="openstar.tess.time-resolved-residual-phase-localization.prepare",
                    parameters={}, triggered_by_stage_id=source_switching_temporal.id)),)
        if localization_started:
            return ()
        candidate = temporal_result.get("preferredCandidate") or {}
        ids = candidate.get("catalogIDs") or {}
        justified = (candidate.get("raDeg") is not None and candidate.get("decDeg") is not None
                     and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None))
        validation_started = any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (not validation_started and temporal_result.get("classification") in {
                "STATIONARY_CANDIDATE_1_SOURCE", "STATIONARY_CANDIDATE_2_SOURCE"}
                and temporal_result.get("recommendedNextTest")
                == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
                and justified):
            return (ScientificBranch(
                id=f"continue-counterpart-variability-after-{source_switching_temporal.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        source_switching_temporal, "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare", parameters={},
                    triggered_by_stage_id=source_switching_temporal.id),
            ),)
        if validation_started:
            return ()
    # Once independent residual-phase difference imaging has completed, it is
    # authoritative over the older unresolved catalog-guided interpretation.
    # Reproduce the handler's direct continuation after a process restart.
    if residual_phase_difference_image is not None:
        spatial_result = residual_phase_difference_image.result or {}
        candidate = spatial_result.get("preferredCandidate") or {}
        ids = candidate.get("catalogIDs") or {}
        justified = (
            candidate.get("raDeg") is not None
            and candidate.get("decDeg") is not None
            and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None)
        )
        validation_started = any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"}
            for stage in investigation.stages
        )
        if (
            not validation_started
            and spatial_result.get("classification")
            in {"CANDIDATE_1_SUPPORTED", "CANDIDATE_2_SUPPORTED"}
            and spatial_result.get("sourceAttributionResolved") is True
            and spatial_result.get("recommendedNextTest")
            == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
            and justified
        ):
            return (ScientificBranch(
                id=f"continue-counterpart-variability-after-{residual_phase_difference_image.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        residual_phase_difference_image, "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={}, triggered_by_stage_id=residual_phase_difference_image.id),
            ),)
        # Other spatial outcomes deliberately remain quiescent until their
        # recommended scientific continuation is implemented.
        return ()
    localization_started = any(
        stage.handler_id.startswith("openstar.tess.catalog-guided-source-localization.")
        for stage in investigation.stages
    )
    if (
        catalog_identification is not None
        and catalog_guided_localization is None
        and not localization_started
        and catalog_result.get("recommendedNextTest") == "CATALOG_GUIDED_SOURCE_LOCALIZATION"
        and catalog_result.get("physicalMechanismResolved") is False
        and len(catalog_result.get("plausibleCatalogCandidates") or []) >= 2
    ):
        return (ScientificBranch(
            id=f"continue-catalog-guided-localization-after-{catalog_identification.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(
                    catalog_identification, "prepare-catalog-guided-source-localization"),
                handler_id="openstar.tess.catalog-guided-source-localization.prepare",
                parameters={}, triggered_by_stage_id=catalog_identification.id),
        ),)
    if catalog_guided_localization is not None:
        localization_result = catalog_guided_localization.result or {}
        candidate = localization_result.get("preferredCandidate") or {}
        ids = candidate.get("catalogIDs") or {}
        justified = (candidate.get("raDeg") is not None and candidate.get("decDeg") is not None
                     and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None))
        validation_started = any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            for stage in investigation.stages)
        difference_image_started = any(
            stage.handler_id.startswith("openstar.tess.residual-phase-difference-imaging.")
            for stage in investigation.stages)
        temporal_started = any(
            stage.handler_id.startswith("openstar.tess.source-switching-temporal-model.")
            for stage in investigation.stages)
        if (residual_phase_difference_image is not None
                and source_switching_temporal is None and not temporal_started):
            difference_result = residual_phase_difference_image.result or {}
            if (difference_result.get("classification") == "SOURCE_SWITCHING_BY_SECTOR"
                    and difference_result.get("recommendedNextTest")
                    == "SOURCE_SWITCHING_TEMPORAL_MODEL"):
                return (ScientificBranch(
                    id=f"continue-source-switching-temporal-after-{residual_phase_difference_image.id}",
                    experiment=StageRequest(
                        id=_continuation_stage_id(
                            residual_phase_difference_image,
                            "prepare-source-switching-temporal-model"),
                        handler_id="openstar.tess.source-switching-temporal-model.prepare",
                        parameters={}, triggered_by_stage_id=residual_phase_difference_image.id),
                ),)
        if (residual_phase_difference_image is None and not difference_image_started
                and localization_result.get("recommendedNextTest")
                == "ADDITIONAL_SOURCE_LOCALIZATION_DATA"
                and localization_result.get("classification") == "UNRESOLVED"
                and localization_result.get("sourceAttributionResolved") is False):
            return (ScientificBranch(
                id=f"continue-residual-phase-difference-imaging-after-{catalog_guided_localization.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        catalog_guided_localization,
                        "prepare-residual-phase-difference-imaging"),
                    handler_id="openstar.tess.residual-phase-difference-imaging.prepare",
                    parameters={}, triggered_by_stage_id=catalog_guided_localization.id),
            ),)
        if (not validation_started
                and localization_result.get("sourceAttributionResolved") is True
                and justified
                and localization_result.get("recommendedNextTest")
                == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"):
            return (ScientificBranch(
                id=f"continue-counterpart-variability-after-{catalog_guided_localization.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        catalog_guided_localization, "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={}, triggered_by_stage_id=catalog_guided_localization.id),
            ),)
        # A completed prepare or run stage already owns the exact next step.
        # Let the persisted-continuation adapter below reproduce that handoff;
        # returning here would incorrectly mark the investigation complete
        # between the three residual phase-difference imaging stages.
        if not (
            difference_image_started
            and residual_phase_difference_image is None
            and investigation.stages[-1].next_stage is not None
        ):
            return ()
    preferred = catalog_result.get("preferredCandidate") or {}
    preferred_ids = preferred.get("catalogIDs") or {}
    justified_preferred = (
        preferred.get("raDeg") is not None
        and preferred.get("decDeg") is not None
        and (preferred_ids.get("ticID") is not None
             or preferred_ids.get("gaiaDR3SourceID") is not None)
    )
    if (
        catalog_identification is not None
        and variability_interpretation is None
        and catalog_result.get("recommendedNextTest")
        == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
        and catalog_result.get("physicalMechanismResolved") is False
        and justified_preferred
        and not any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"}
            for stage in investigation.stages
        )
    ):
        return (
            ScientificBranch(
                id=f"continue-counterpart-variability-after-{catalog_identification.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        catalog_identification, "prepare-offset-source-variability"
                    ),
                    handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={},
                    triggered_by_stage_id=catalog_identification.id,
                ),
            ),
        )

    variability_result = (
        variability_interpretation.result or {}
        if variability_interpretation is not None
        else {}
    )
    gaia_started = any(
        stage.handler_id.startswith(
            "openstar.tess.gaia-source-resolved-counterpart-photometry."
        )
        for stage in investigation.stages
    )
    if (
        variability_interpretation is not None
        and gaia_interpretation is None
        and not gaia_started
        and variability_result.get("recommendedNextTest")
        == "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
        and variability_result.get("physicalMechanismResolved") is False
    ):
        return (
            ScientificBranch(
                id=f"continue-gaia-counterpart-after-{variability_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        variability_interpretation,
                        "prepare-gaia-source-resolved-counterpart-photometry",
                    ),
                    handler_id=(
                        "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare"
                    ),
                    parameters={},
                    triggered_by_stage_id=variability_interpretation.id,
                ),
            ),
        )

    if variability_interpretation is not None:
        return ()

    if _has_terminal_tess_evidence(investigation):
        return ()

    # v20.2 could persist this preparation failure before preparation had a
    # structured non-executable result.  Preserve the immutable failed record,
    # but resume at the scientifically appropriate independent-sector branch.
    latest = investigation.stages[-1] if investigation.stages else None
    if (
        latest is not None
        and latest.status == "FAILED"
        and latest.handler_id == "openstar.tess.followup.prepare-low-frequency"
        and "Follow-up frequency window is invalid" in (latest.error or "")
    ):
        return (
            ScientificBranch(
                id="recover-unavailable-low-frequency-followup",
                experiment=StageRequest(
                    id=f"{len(investigation.stages) + 1:03d}-prepare-independent-sectors",
                    handler_id="openstar.tess.independent.prepare",
                    parameters={},
                    triggered_by_stage_id=latest.id,
                ),
            ),
        )

    targeted = _latest_complete(
        investigation, "openstar.tess.targeted-observation-planning.generate"
    )
    if targeted is not None:
        availability = investigation.metadata.get("externalDataAvailability") or {}
        dependency_id = "targeted-high-resolution-time-series-photometry"
        available = availability.get(dependency_id) is True
        return (
            ScientificBranch(
                id="targeted-observation-analysis",
                experiment=StageRequest(
                    id=f"{len(investigation.stages) + 1:03d}-targeted-observation-analysis",
                    handler_id="openstar.tess.targeted-observation-analysis.run",
                    parameters={"observationPlanStageID": targeted.id},
                    triggered_by_stage_id=targeted.id,
                ),
                external_data=(
                    ExternalDataDependency(
                        dependency_id,
                        available,
                        (
                            None
                            if available
                            else "The preregistered targeted observations have not been acquired and ingested."
                        ),
                    ),
                ),
            ),
        )

    # The existing TESS handlers already make the within-run scientific
    # continuation decision in StageOutcome.next_stage.  Autonomous dispatch
    # records that decision on the immutable stage and asks this adapter to
    # translate it, rather than reimplementing the decision tree here.
    if investigation.stages:
        latest = investigation.stages[-1]
        if latest.status == "COMPLETE" and latest.next_stage is not None:
            raw = latest.next_stage
            return (
                ScientificBranch(
                    id=f"continue-after-{latest.id}",
                    experiment=StageRequest(
                        id=str(raw["id"]),
                        handler_id=str(raw["handler_id"]),
                        parameters=dict(raw.get("parameters") or {}),
                        triggered_by_stage_id=raw.get("triggered_by_stage_id"),
                    ),
                ),
            )

    atlas_fixed = _latest_complete(
        investigation, "openstar.tess.atlas-fixed-window.interpret"
    )
    if (
        atlas_fixed is not None
        and (atlas_fixed.result or {}).get("recommendedNextTest")
        == "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    ):
        return (
            ScientificBranch(
                id="targeted-observation-planning",
                experiment=StageRequest(
                    id=f"{len(investigation.stages) + 1:03d}-targeted-observation-planning",
                    handler_id="openstar.tess.targeted-observation-planning.generate",
                    parameters={},
                    triggered_by_stage_id=atlas_fixed.id,
                ),
            ),
        )

    if not investigation.stages:
        metadata = investigation.metadata
        return (
            ScientificBranch(
                id="primary-tess-investigation",
                experiment=StageRequest(
                    id="001-prepare-target",
                    handler_id="openstar.tess.prepare-target",
                    parameters={
                        "projectPath": metadata["sourceProjectPath"],
                        "datasetID": metadata["datasetID"],
                        "ticID": metadata.get("ticID"),
                    },
                ),
            ),
        )

    # A nonempty state with no recognized scientific continuation is not proof
    # that the investigation is complete.
    return (
        ScientificBranch(
            id="unresolved-tess-continuation",
            experiment=StageRequest(
                id=f"{len(investigation.stages) + 1:03d}-unresolved-continuation",
                handler_id="openstar.tess.unresolved-continuation",
                parameters={},
            ),
            required_stage_ids=("tess-continuation-decision",),
        ),
    )


def register_tess_workflow_handlers(
    store: InvestigationStore,
    coordinator,
    *,
    poll_interval: float = 1.0,
    timeout: float | None = None,
    historical_path_resolver=None,
) -> WorkflowEngine:
    """Register the existing v20.28 TESS handlers for autonomous dispatch."""

    from .tess_investigation import build_engine

    engine = build_engine(
        store,
        coordinator,
        poll_interval=poll_interval,
        timeout=timeout,
        historical_path_resolver=historical_path_resolver,
    )
    engine.chain_stages = False
    return engine
