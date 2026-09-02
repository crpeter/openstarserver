Warning: truncated output (original token count: 199373)
Total output lines: 15293

from __future__ import annotations

import copy

import json
import math
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

from .tess_localization_evidence import frozen_residual_localization_family

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import (
    ArtifactReference,
    Investigation,
    InvestigationStore,
    sha256_file,
    sha256_json,
)
from openstar_path_relocation import HistoricalPathResolver
from openstar_workflow import (
    RetryableExecutionError,
    StageOutcome,
    StageRequest,
    WorkflowEngine,
)

from .tess_claims import validate_claim
from .tess_followup import (
    build_low_frequency_followup,
    build_single_target_primary,
)
from .tess_primary_reuse import run_primary as run_primary_with_reuse
from .tess_hypotheses import (
    analyze,
    broad_independent_next_handler,
    interpret_broad_independent_sectors,
    interpret_followup,
    interpret_independent_sectors,
    plan,
    plan_independent_contradiction_resolution,
    rotational_sanity,
)
from .tess_identity import collect_identity, transient_required_catalog_failures
from .tess_morphology import analyze_morphology
from .tess_physical import (
    analyze_physical_interpretation,
    physical_source_localization_continuation,
)
from .tess_binary_confirmation import (
    MORPHOLOGY_EVENT_SCREEN_ENTRY,
    analyze_binary_confirmation,
    morphology_event_screening_continuation,
    physical_interpretation_continuation,
)
from .tess_blind_transit_search import (
    HANDLER_ID as BLIND_TRANSIT_SEARCH_HANDLER_ID,
    analyze_blind_transit_search,
    analyze_exhausted_distributed_residual_candidates,
    analyze_iterative_blind_transit_search,
    blind_transit_search_continuation,
)
from .tess_exhausted_residual_candidates import (
    build_exhausted_residual_candidate_project,
    distributed_candidate_generation_warranted,
    interpret_exhausted_residual_candidate_project,
)
from .tess_eclipse_event_localization import (
    HANDLER_ID as ECLIPSE_LOCALIZATION_HANDLER_ID,
    PREPARE_HANDLER_ID as ECLIPSE_LOCALIZATION_PREPARE_HANDLER_ID,
    authoritative_binary_gate,
    localize_eclipse_events,
)
from .tess_external_companion_evidence import (
    ExternalEvidenceTransientError,
    FREEZE_HANDLER_ID as EXTERNAL_EVIDENCE_FREEZE_HANDLER_ID,
    INTERPRET_HANDLER_ID as EXTERNAL_EVIDENCE_INTERPRET_HANDLER_ID,
    REVIEW_HANDLER_ID as SOURCE_ATTRIBUTION_REVIEW_HANDLER_ID,
    acquire_external_evidence,
    interpret_external_evidence,
    review_source_attribution,
)
from .tess_companion_evidence_synthesis import (
    HANDLER_ID as COMPANION_SYNTHESIS_HANDLER_ID,
    synthesize_companion_evidence,
)
from .tess_event_depth_accuracy import (
    AUDIT_HANDLER_ID as EVENT_DEPTH_AUDIT_HANDLER_ID,
    FREEZE_HANDLER_ID as EVENT_DEPTH_FREEZE_HANDLER_ID,
    acquire_full_precision_photometry,
    audit_depth_attenuation,
    validate_audit_hash,
)
from .tess_joint_event_phase_model import (
    HANDLER_ID as JOINT_EVENT_PHASE_MODEL_HANDLER_ID,
    chronology_from_completed_stages,
    fit_joint_event_phase_model,
    model_required,
    validate_model_hash,
)
from .tess_localization import localize_periodic_source
from .tess_sector_archive import TessArchiveTransientError
from .tess_multimode import (
    MAX_RESIDUAL_ITERATIONS,
    build_residual_search_project,
    interpret_residual_iteration,
    summarize_multimode_decomposition,
)
from .tess_time_frequency import (
    build_time_frequency_project,
    interpret_time_frequency_project,
    summarize_time_frequency_evolution,
)
from .tess_nonstationary import (
    CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
    RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
    build_confirmed_nonstationary_method_contract,
    build_nonstationary_project,
    build_recurrent_residual_nonstationary_method_contract,
    confirmed_nonstationary_physical_period,
    confirmed_nonstationary_method_contract_hash,
    interpret_nonstationary_project,
    recurrent_residual_nonstationary_method_contract_hash,
    summarize_nonstationary_modeling,
    validate_confirmed_nonstationary_boundary,
    validate_confirmed_nonstationary_localization_boundary,
    validate_recurrent_residual_nonstationary_boundary,
    validate_recurrent_residual_nonstationary_localization_boundary,
)
from .tess_mode_identification import (
    CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID,
    V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE,
    MULTIMODE_MODE_EVIDENCE_LINEAGE,
    analyze_confirmed_coherent_residual_mode,
    build_confirmed_coherent_mode_method_contract,
    confirmed_coherent_mode_method_contract_hash,
    identify_residual_mode,
    validate_v20_8_confirmed_coherent_residual,
    validated_multimode_mode_evidence,
)
from .tess_long_baseline_frequency_confirmation import (
    HANDLER_ID as LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID,
    analyze_long_baseline_frequency_confirmation,
    build_method_contract as build_long_baseline_frequency_confirmation_contract,
    method_contract_hash as long_baseline_frequency_confirmation_contract_hash,
)
from .tess_v20_8_long_baseline_time_frequency_confirmation import (
    HANDLER_ID as V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID,
    METHOD_CONTRACT_ID as V20_8_LONG_BASELINE_METHOD_CONTRACT_ID,
    analyze_long_baseline_time_frequency_confirmation as analyze_v20_8_long_baseline_time_frequency_confirmation,
    build_dataset_specs as build_v20_8_long_baseline_dataset_specs,
    build_method_contract as build_v20_8_long_baseline_method_contract,
    method_contract_hash as v20_8_long_baseline_method_contract_hash,
)
from .tess_transient_mode_validation import (
    HANDLER_ID as TRANSIENT_MODE_VALIDATION_HANDLER_ID,
    analyze_transient_mode_validation,
    build_dataset_specs as build_transient_mode_dataset_specs,
    build_method_contract as build_transient_mode_method_contract,
    method_contract_hash as transient_mode_method_contract_hash,
)
from .tess_recurrent_residual_long_baseline_confirmation import (
    METHOD_CONTRACT_ID as RECURRENT_RESIDUAL_METHOD_CONTRACT_ID,
    analyze_recurrent_residual_long_baseline_confirmation,
    build_dataset_specs as build_recurrent_residual_dataset_specs,
    build_method_contract as build_recurrent_residual_method_contract,
)
from .tess_dynamic_harmonic import (
    compare_unresolved_family_dynamic_harmonics,
    model_dynamic_harmonics,
    refine_harmonic_family_frequency,
)
from .tess_resolved_cycle import (
    CORROBORATED_SOURCE as CORROBORATED_RESOLVED_CYCLE_SOURCE,
    NESTED_ALIAS_SOURCE as NESTED_ALIAS_RESOLVED_CYCLE_SOURCE,
    authoritative_resolved_cycle,
    validated_cycle_period,
)
from .tess_residual_localization import (
    build_residual_mode_pixel_project,
    interpret_residual_mode_pixel_project,
)
from .tess_residual_external_evidence import (
    HANDLER_ID as RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID,
    analyze_residual_external_evidence,
)
from .tess_target_residual_astrophysical_mechanism import (
    HANDLER_ID as TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID,
    analyze_target_residual_astrophysical_mechanism,
)
from .tess_neighbor_catalog_pixel_response_review import (
    HANDLER_ID as NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID,
    analyze_neighbor_catalog_pixel_response_review,
)
from .tess_residual_localization_review import (
    build_residual_mode_localization_review_project,
    interpret_residual_mode_localization_review_project,
)
from .tess_multisource_residual import (
    build_multisource_residual_project,
    interpret_multisource_residual_project,
)
from .tess_intrinsic_nonstationary import classify_target_component
from .tess_target_residual_mechanism import analyze_target_residual_mechanism
from .tess_target_residual_astrophysical_interpretation import (
    FrozenCatalogAstrophysicalEvidenceProvider,
    interpret_target_residual_astrophysics,
    newest_authoritative_recommendation,
)
from .tess_target_residual_mechanism_adjudication import (
    adjudicate_frozen_target_residual_mechanism,
)
from .tess_target_residual_mechanism_predictive_validation import (
    analyze_predictive_validation,
    v2013_lineage_matches,
)


def _joint_model_chronology_from_completed_stages(completed):
    """Derive, rather than assert, the strict pre-model completed-stage proof."""
    required_handlers = [SOURCE_ATTRIBUTION_REVIEW_HANDLER_ID, EVENT_DEPTH_FREEZE_HANDLER_ID,
                         EVENT_DEPTH_AUDIT_HANDLER_ID]
    return chronology_from_completed_stages(completed, required_handlers,
        {EXTERNAL_EVIDENCE_FREEZE_HANDLER_ID, EXTERNAL_EVIDENCE_INTERPRET_HANDLER_ID})
from .tess_target_residual_archival_baseline import (
    adjudicate_target, adjudicate_sector, build_archival_baseline_project,
    previously_consumed_tess_sectors, verify_frozen_science_lineage,
)
from .tess_target_residual_pixel_recurrence import (
    verify_v2017_lineage, interpret_sectors, freeze_catalog_hypotheses, measure_sector,
    CatalogInfrastructureError, NoPixelCoverageError, acquire_selected_sector,
    tpf_flux_cube,
)
from .tess_target_residual_multisector_source import (
    verify_v2018_lineage, derive_competing_sources, derive_additional_sectors,
    eligible_additional_sectors, interpret_multisector, run_multisector_source_localization,
)
from .tess_offset_source import identify_offset_residual_source
from .tess_offset_variability import (
    build_offset_source_variability_project,
    interpret_offset_source_variability_project,
)
from .tess_prf_deblend import (
    build_calibrated_prf_deblending_project,
    interpret_calibrated_prf_deblending_project,
)
from .tess_catalog_guided_localization import (
    prepare_catalog_guided_localization,
    run_catalog_guided_localization,
    interpret_catalog_guided_localization,
)
from .tess_deep_catalog_guided_localization import (
    PREPARE_HANDLER_ID as DEEP_CATALOG_PRF_PREPARE_HANDLER_ID,
    RUN_HANDLER_ID as DEEP_CATALOG_PRF_RUN_HANDLER_ID,
    INTERPRET_HANDLER_ID as DEEP_CATALOG_PRF_INTERPRET_HANDLER_ID,
    prepare_deep_catalog_guided_localization,
    run_deep_catalog_guided_localization,
    interpret_deep_catalog_guided_localization,
)
from .tess_difference_image import (
    build_difference_image_project,
    interpret_difference_image_project,
)
from .tess_residual_phase_difference_image import (
    prepare_residual_phase_difference_imaging,
    run_residual_phase_difference_imaging,
    interpret_residual_phase_difference_imaging,
)
from .tess_period_family_difference_image import (
    PREPARE_HANDLER as PERIOD_FAMILY_DIFFERENCE_PREPARE_HANDLER,
    RUN_HANDLER as PERIOD_FAMILY_DIFFERENCE_RUN_HANDLER,
    INTERPRET_HANDLER as PERIOD_FAMILY_DIFFERENCE_INTERPRET_HANDLER,
    verified_period_family_boundary,
    prepare_period_family_difference_imaging,
    run_period_family_difference_imaging,
    interpret_period_family_difference_imaging,
)
from .tess_period_family_time_domain_evolution import (
    PREPARE_HANDLER as PERIOD_FAMILY_TIME_DOMAIN_PREPARE_HANDLER,
    RUN_HANDLER as PERIOD_FAMILY_TIME_DOMAIN_RUN_HANDLER,
    INTERPRET_HANDLER as PERIOD_FAMILY_TIME_DOMAIN_INTERPRET_HANDLER,
    verified_time_domain_evolution_boundary,
    prepare_period_family_time_domain_evolution,
    run_period_family_time_domain_evolution,
    interpret_period_family_time_domain_evolution,
)
from .tess_source_switching_temporal import (
    prepare_source_switching_temporal_model,
    run_source_switching_temporal_model,
    interpret_source_switching_temporal_model,
)
from .tess_time_resolved_residual_phase_localization import (
    prepare_time_resolved_residual_phase_localization,
    run_time_resolved_residual_phase_localization,
    interpret_time_resolved_residual_phase_localization,
)
from .tess_time_resolved_frequency_localization import (
    prepare_time_resolved_frequency_localization,
    run_time_resolved_frequency_localization,
    interpret_time_resolved_frequency_localization,
)
from .tess_additional_sector_source_localization import (
    prepare_additional_sector_source_localization,
    run_additional_sector_source_localization,
    interpret_additional_sector_source_localization,
    unused_official_sectors,
    bridge_is_complete,
)
from .tess_frequency_localized_pixel import (
    build_frequency_localized_pixel_project,
    interpret_frequency_localized_pixel_project,
)
from .tess_spoc_prf import (
    build_official_spoc_prf_project,
    interpret_official_spoc_prf_project,
)
from .tess_prf_refinement import (
    prepare_prf_deblending,
    run_prf_deblending,
    interpret_prf_deblending,
)
from .tess_catalog_counterpart import identify_catalog_counterparts
from .tess_deep_catalog_counterpart import (
    HANDLER_ID as DEEP_CATALOG_COUNTERPART_HANDLER_ID,
    identify_deep_catalog_counterparts,
)
from .tess_external_highres import (
    build_external_high_resolution_project,
    interpret_external_high_resolution_project,
)
from .tess_gaia_counterpart import (
    GaiaArchiveUnavailable,
    build_current_gaia_counterpart_project,
    interpret_current_gaia_counterpart_project,
)
from .tess_skymapper_resolved import (
    SkyMapperArchiveUnavailable,
    build_skymapper_resolved_project,
    interpret_skymapper_resolved_project,
)
from .tess_nsc_resolved import (
    NSCArchiveUnavailable,
    build_nsc_resolved_project,
    interpret_nsc_resolved_project,
)
from .tess_noirlab_forced_photometry import (
    NOIRLabArchiveUnavailable,
    build_noirlab_image_forced_photometry_project,
    interpret_noirlab_image_forced_photometry_project,
)
from .tess_des_dr2_se_local_forced import (
    CURRENT_TRIGGER as CURRENT_DES_TRIGGER,
    DESArchiveUnavailable,
    build_des_dr2_se_local_forced_project,
    interpret_des_dr2_se_local_forced_project,
)
from .tess_atlas_forced_photometry import (
    ATLASArchiveUnavailable,
    CURRENT_TRIGGER as CURRENT_ATLAS_TRIGGER,
    SIGNED_REANALYSIS as ATLAS_SIGNED_REANALYSIS,
    build_atlas_forced_photometry_project,
    submit_atlas_forced_photometry_jobs,
    interpret_atlas_forced_photometry_project,
)
from .tess_atlas_forced_reanalysis import (
    build_atlas_forced_photometry_reanalysis_project,
    interpret_atlas_forced_photometry_reanalysis_project,
)
from .tess_atlas_time_resolved import (
    build_atlas_time_resolved_project,
    interpret_atlas_time_resolved_project,
)
from .tess_atlas_fixed_windows import (
    build_atlas_fixed_window_project,
    interpret_atlas_fixed_window_project,
)
from .tess_observation_planning import (
    build_targeted_observation_plan,
)
from .tess_source_pair_lineage import frozen_source_pair_evidence
from .tess_multisector import (
    TessArchiveInfrastructureError,
    build_broad_independent_sector_project,
    build_independent_sector_project,
)


WORKFLOW_ID = "openstar.workflow.tess-investigation.v1"
WORKFLOW_VERSION = "20.2"
SOFTWARE_ID = "openstar.tess-investigation-plugin"
SOFTWARE_VERSION = "20.46"
ADAPTIVE_BLIND_TRANSIT_ADDITIONAL_SECTORS = 8
EXHAUSTED_RESIDUAL_RUN_HANDLER_ID = (
    "openstar.tess.blind-transit-distributed.run"
)
EXHAUSTED_RESIDUAL_INTERPRET_HANDLER_ID = (
    "openstar.tess.blind-transit-distributed.interpret"
)
EXHAUSTED_RESIDUAL_TRANSIT_NEXT_TEST = (
    "GENERIC_DISTRIBUTED_RESIDUAL_TRANSIT_CANDIDATE_GENERATION"
)


def _blind_transit_sector_availability(
    independent_prepare: dict[str, Any], analysis_spec: dict[str, Any],
) -> dict[str, Any]:
    candidate_sectors = sorted({
        int(sector)
        for sector in independent_prepare.get("candidateSectors") or []
    })
    prepared_sectors = sorted({
        int(item["sector"])
        for item in analysis_spec.get("preparedSectors") or []
        if item.get("sector") is not None
    })
    prepared = set(prepared_sectors)
    remaining_sectors = [
        sector for sector in candidate_sectors if sector not in prepared
    ]
    return {
        "candidateSectors": candidate_sectors,
        "preparedSectors": prepared_sectors,
        "remainingSectors": remaining_sectors,
        "allCandidateSectorsPrepared": bool(
            candidate_sectors and not remaining_sectors
        ),
        "catalogAnswerKeyUsed": False,
    }


def _exhausted_residual_family_census(
    result: dict[str, Any], sector_availability: dict[str, Any],
) -> dict[str, Any] | None:
    iterative = result.get("iterativeSearch") or {}
    if not (
        result.get("classification")
        == "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE"
        and iterative.get("terminationReason")
        == "NEXT_RESIDUAL_SIGNAL_UNRESOLVED"
        and sector_availability.get("allCandidateSectorsPrepared") is True
    ):
        return None
    iterations = iterative.get("iterations") or []
    if not iterations:
        return None
    stopping = iterations[-1]
    fallback = stopping.get("boxModelSubtractionFallback") or {}
    evidence_sources = [
        (
            "CUMULATIVE_TRANSIT_WINDOW_MASKING",
            stopping.get("candidateEvidence") or {},
        ),
        (
            "BOX_MODEL_SUBTRACTION",
            fallback.get("subtractionCandidateEvidence") or {},
        ),
    ]
    methods = []
    for residual_method, evidence in evidence_sources:
        audit = evidence.get("candidateGenerationAudit") or {}
        if not audit.get("families"):
            continue
        methods.append({
            "residualSearchMethod": residual_method,
            "candidateGenerationAudit": copy.deepcopy(audit),
        })
    if not methods:
        return None
    return {
        "version": "1.0",
        "trigger": "ALL_OFFICIAL_INDEPENDENT_SECTORS_CONSUMED_WITH_UNRESOLVED_RESIDUAL",
        "sectorAvailability": copy.deepcopy(sector_availability),
        "methods": methods,
        "selectionEligible": False,
        "candidateSelectionAffected": False,
        "claimDecisionAffected": False,
        "recommendedNextTestAffectedByFamilyScores": False,
        "catalogAnswerKeyUsed": False,
    }


def _apply_blind_transit_sector_exhaustion(
    result: dict[str, Any], independent_prepare: dict[str, Any],
    analysis_spec: dict[str, Any],
) -> dict[str, Any]:
    sector_availability = _blind_transit_sector_availability(
        independent_prepare, analysis_spec
    )
    updated = dict(result)
    updated["independentSectorAvailability"] = sector_availability
    census = _exhausted_residual_family_census(updated, sector_availability)
    if census is not None:
        updated["exhaustedSectorResidualFamilyCensus"] = census
    iterative = updated.get("iterativeSearch") or {}
    previous = updated.get("recommendedNextTest")
    if (
        sector_availability.get("allCandidateSectorsPrepared") is True
        and iterative.get("terminationReason")
        == "NEXT_RESIDUAL_SIGNAL_UNRESOLVED"
        and previous
        == "ADDITIONAL_INDEPENDENT_SECTOR_TRANSIT_CONFIRMATION"
    ):
        updated["recommendedNextTest"] = EXHAUSTED_RESIDUAL_TRANSIT_NEXT_TEST
        updated["recommendedNextTestRevision"] = {
            "previousRecommendedNextTest": previous,
            "recommendedNextTest": EXHAUSTED_RESIDUAL_TRANSIT_NEXT_TEST,
            "reason": "ALL_OFFICIAL_INDEPENDENT_SECTORS_ALREADY_CONSUMED",
            "candidateSelectionAffected": False,
            "claimDecisionAffected": False,
            "catalogAnswerKeyUsed": False,
        }
    return updated


def _residual_trial_passes_gates_but_remains_conservatively_blocked(
    trial: dict[str, Any],
) -> bool:
    """Recognize evidence that permits more data, never candidate promotion."""
    return bool(
        trial.get("accepted") is False
        and trial.get("rejectionReasons")
        == ["RANKED_FALLBACK_REQUIRES_INTEGER_CYCLE_ALIAS_PROMOTION"]
        and trial.get("primarySectorSupported") is True
        and len(trial.get("supportingIndependentSectors") or []) >= 2
        and (trial.get("linearEphemeris") or {}).get("coherent") is True
        and (trial.get("frequencyFamilySeparation") or {}).get("distinct") is True
        and (trial.get("recurrenceSupportGate") or {}).get("mode")
        != "NOT_SATISFIED"
    )


def _iterative_blind_sector_extension_reason(
    result: dict[str, Any],
) -> str | None:
    """Return the exact evidence boundary that permits one sector extension."""
    iterative = result.get("iterativeSearch") or {}
    if not (
        result.get("classification")
        == "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE"
        and iterative.get("terminationReason")
        == "NEXT_RESIDUAL_SIGNAL_UNRESOLVED"
    ):
        return None
    accepted_count = len(result.get("candidateSignals") or [])
    if accepted_count >= 2:
        return "MULTI_CLOCK_ITERATIVE_RESIDUAL_SIGNAL_UNRESOLVED"
    if accepted_count != 1:
        return None
    iterations = iterative.get("iterations") or []
    if not iterations:
        return None
    stopping = iterations[-1]
    masked_selection = (
        (stopping.get("candidateEvidence") or {}).get(
            "rankedFrequencyFamilySelection"
        )
        or {}
    )
    subtraction_evidence = (
        stopping.get("boxModelSubtractionFallback") or {}
    ).get("subtractionCandidateEvidence") or {}
    subtraction_selection = (
        subtraction_evidence.get("rankedFrequencyFamilySelection") or {}
    )
    selections = [
        masked_selection,
        subtraction_selection,
    ]
    if any(
        _residual_trial_passes_gates_but_remains_conservatively_blocked(trial)
        for selection in selections
        for trial in selection.get("trials") or []
    ):
        return "SINGLE_CLOCK_DISTINCT_RESIDUAL_REQUIRES_MORE_SECTORS"
    return None


def _iterative_blind_sector_extension_warranted(result: dict[str, Any]) -> bool:
    return _iterative_blind_sector_extension_reason(result) is not None


def _stage(investigation: Investigation, stage_id: str):
    for stage in investigation.stages:
        if stage.id == stage_id:
            return stage
    raise KeyError(f"Investigation stage not found: {stage_id}")


def _result(investigation: Investigation, stage_id: str) -> dict[str, Any]:
    stage = _stage(investigation, stage_id)
    if stage.status != "COMPLETE" or stage.result is None:
        raise RuntimeError(f"Stage is not COMPLETE with a result: {stage_id}")
    return stage.result


def _latest_result_for_handler(
    investigation: Investigation,
    handler_id: str,
) -> dict[str, Any] | None:
    for stage in reversed(investigation.stages):
        if stage.handler_id == handler_id and stage.status == "COMPLETE":
            return stage.result
    return None


def _latest_blind_transit_result(
    investigation: Investigation,
) -> dict[str, Any] | None:
    distributed = _latest_result_for_handler(
        investigation, EXHAUSTED_RESIDUAL_INTERPRET_HANDLER_ID
    )
    if distributed is not None:
        return distributed
    return _latest_result_for_handler(
        investigation, BLIND_TRANSIT_SEARCH_HANDLER_ID
    )


def _required_latest_result_for_handler(
    investigation: Investigation,
    handler_id: str,
) -> dict[str, Any]:
    result = _latest_result_for_handler(investigation, handler_id)
    if result is None:
        raise RuntimeError(f"No COMPLETE stage result for handler: {handler_id}")
    return result


def _all_results_for_handler(
    investigation: Investigation,
    handler_id: str,
) -> list[dict[str, Any]]:
    return [
        stage.result
        for stage in investigation.stages
        if stage.handler_id == handler_id
        and stage.status == "COMPLETE"
        and stage.result is not None
    ]


def _validated_multimode_cycle(
    investigation: Investigation,
) -> tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the exact physical cycle carried through target localization."""
    physical = _latest_result_for_handler(
        investigation,
        "openstar.tess.physical.interpret",
    )
    localization = _latest_result_for_handler(
        investigation,
        "openstar.tess.source-localization.analyze",
    )
    if physical is None or localization is None:
        raise RuntimeError(
            "v20.7 requires completed physical interpretation and source localization."
        )

    cycle = localization.get("physicalCycleEvidence")
    period = validated_cycle_period(cycle)
    cross = localization.get("crossSector") or {}
    try:
        physical_period = float(physical.get("physicalPeriodDays"))
        physical_harmonic = float(
            physical.get("photometricFirstHarmonicPeriodDays")
        )
        localized_period = float(localization.get("physicalPeriodDays"))
        localized_harmonic = float(
            localization.get("photometricFirstHarmonicPeriodDays")
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            "v20.7 requires finite, consistent physical-cycle periods."
        ) from None

    exact_cycle = (
        period is not None
        and physical.get("physicalCycleEvidence") == cycle
        and physical.get("physicalMechanismResolved") is False
        and localization.get("version")
        == "openstar.tess-pixel-localization.v1"
        and cross.get("classification") == "TARGET_SOURCE_SUPPORTED"
        and cross.get("variableSignalOrigin") == "TARGET_CONSISTENT"
        and cross.get("recommendedNextTest")
        == "MULTI_MODE_FREQUENCY_DECOMPOSITION"
        and localization.get("recommendedNextTest")
        == "MULTI_MODE_FREQUENCY_DECOMPOSITION"
        and all(
            math.isfinite(value)
            for value in (
                physical_period,
                physical_harmonic,
                localized_period,
                localized_harmonic,
            )
        )
        and math.isclose(
            physical_period, period, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            localized_period, period, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            physical_harmonic, period / 2.0,
            rel_tol=1e-9, abs_tol=1e-12,
        )
        and math.isclose(
            localized_harmonic, period / 2.0,
            rel_tol=1e-9, abs_tol=1e-12,
        )
    )
    if not exact_cycle:
        raise RuntimeError(
            "v20.7 requires the exact authoritative physical-cycle evidence "
            "carried through TARGET_SOURCE_SUPPORTED localization."
        )
    return period, cycle, physical, localization


def _artifact(path: Path, media_type: str) -> ArtifactReference:
    return ArtifactReference(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        media_type=media_type,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_baseline_days(path: str | Path) -> float | None:
    dataset = _load_json(path)
    source = dataset.get("source") or {}
    if source.get("baselineDays") is not None:
        try:
            return float(source["baselineDays"])
        except (TypeError, ValueError):
            pass
    times = dataset.get("times") or []
    if len(times) > 1:
        return float(times[-1]) - float(times[0])
    return None


def _target_result(project_status: dict[str, Any]) -> dict[str, Any]:
    datasets = project_status.get("datasets") or []
    if len(datasets) != 1:
        raise RuntimeError(
            "A primary/same-dataset TESS investigation project must contain exactly one dataset; "
            f"got {len(datasets)}."
        )
    return dict(datasets[0])


def _next_stage_id(current_id: str, label: str) -> str:
    try:
        number = int(str(current_id).split("-", 1)[0]) + 1
    except (TypeError, ValueError):
        raise ValueError(f"Stage id must begin with an integer prefix: {current_id}")
    return f"{number:03d}-{label}"


def _retry_transient_tess_archive_failures(handler):
    """Adapt provider-neutral TESS archive failures at the workflow boundary."""
    @wraps(handler)
    def wrapped(investigation, request):
        try:
            return handler(investigation, request)
        except TessArchiveTransientError as error:
            raise RetryableExecutionError(str(error)) from error
    return wrapped


def broad_independent_continuation(
    interpreted: dict[str, Any],
    *,
    request_id: str,
    finalize_parameters: dict[str, Any] | None = None,
) -> StageRequest:
    """Choose the next scientific question from persisted broad-cluster evidence."""

    next_handler = broad_independent_next_handler(interpreted)
    warranted = next_handler != "openstar.tess.finalize"
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "characterize-variability" if warranted else "finalize",
        ),
        handler_id=next_handler,
        parameters={} if warranted else dict(finalize_parameters or {}),
        triggered_by_stage_id=request_id,
    )


def _primary_harmonic_morphology_family(
    analysis: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a persisted primary harmonic question, or fail closed.

    The relation label and the two periods must describe the same multiplicative
    family.  This prevents an unrelated preferred period from being promoted to
    morphology merely because both values happen to be present.
    """

    if analysis.get("primaryReliable") is not True:
        return None
    relation = str(analysis.get("preferredPhysicalPeriodRelation") or "1x")
    if relation == "1x" or not relation.endswith("x"):
        return None
    try:
        relation_multiplier = float(relation[:-1])
        raw_period = float(
            analysis.get("rawCandidatePeriodDays")
            if analysis.get("rawCandidatePeriodDays") is not None
            else analysis.get("candidatePeriodDays")
        )
        preferred_period = float(
            analysis.get("preferredPhysicalPeriodDays")
            if analysis.get("preferredPhysicalPeriodDays") is not None
            else analysis.get("observedPeriodDays")
        )
    except (TypeError, ValueError):
        return None
    values = (relation_multiplier, raw_period, preferred_period)
    if not all(math.isfinite(value) and value > 0 for value in values):
        return None
    observed_multiplier = preferred_period / raw_period
    if not math.isclose(observed_multiplier, relation_multiplier, rel_tol=0.05):
        return None
    return {
        "representativeRawPeriodDays": raw_period,
        "possibleDoubleCycleDays": preferred_period,
        "preferredPhysicalPeriodRelation": relation,
        "physicalCycleResolved": False,
        "evidenceSource": "authoritative-persisted-primary-analysis",
    }


def time_frequency_continuation(summary: dict[str, Any], *, request_id: str) -> StageRequest:
    """Continue only the explicitly recommended, still-unresolved experiment."""

    period_reference = summary.get("periodReference") or {}
    post_dynamic_harmonic_residual = (
        summary.get("evidenceLineage")
        == "POST_DYNAMIC_HARMONIC_RESIDUAL_TIME_FREQUENCY"
    )
    try:
        physical_period = float(period_reference.get("periodDays"))
    except (TypeError, ValueError):
        physical_period = math.nan
    run_physical_interpretation = (
        summary.get("recommendedNextTest") == "BINARY_ROTATION_EXTERNAL_EVIDENCE"
        and summary.get("physicalMechanismResolved") is False
        and period_reference.get("physicalCycleResolved") is True
        and period_reference.get("kind") == "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
        and math.isfinite(physical_period)
        and physical_period > 0
    )
    run_mode_identification = (
        summary.get("recommendedNextTest") == "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
        and summary.get("physicalMechanismResolved") is False
    )
    run_nonstationary = (
        summary.get("recommendedNextTest")
        == "LONG_BASELINE_NONSTATIONARY_MODE_MODELING"
        and summary.get("physicalMechanismResolved") is False
    )
    run_resolved_dynamic_harmonic = (
        summary.get("recommendedNextTest") == "DYNAMIC_HARMONIC_MODELING"
        and summary.get("physicalMechanismResolved") is False
        and not post_dynamic_harmonic_residual
        and period_reference.get("kind") == "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
        and period_reference.get("physicalCycleResolved") is True
        and math.isfinite(physical_period)
        and physical_period > 0
    )
    run_unresolved_dynamic_harmonic = (
        summary.get("recommendedNextTest") == "DYNAMIC_HARMONIC_MODELING"
        and summary.get("physicalMechanismResolved") is False
        and not post_dynamic_harmonic_residual
        and period_reference.get("kind") == "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE"
        and period_reference.get("physicalCycleResolved") is False
        and math.isfinite(physical_period)
        and physical_period > 0
    )
    run_dynamic_harmonic = (
        run_resolved_dynamic_harmonic or run_unresolved_dynamic_harmonic
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "dynamic-harmonic-modeling" if run_dynamic_harmonic else
            ("mode-identification" if run_mode_identification else
            ("prepare-nonstationary" if run_nonstationary else
             ("physical-interpretation" if run_physical_interpretation else "finalize"))),
        ),
        handler_id=(
            "openstar.tess.dynamic-harmonic.analyze" if run_dynamic_harmonic else
            ("openstar.tess.mode-identification.analyze" if run_mode_identification else
            ("openstar.tess.nonstationary.prepare"
            if run_nonstationary
            else ("openstar.tess.physical.interpret"
                  if run_physical_interpretation
                  else "openstar.tess.finalize")
            ))
        ),
        parameters=({"evidenceLineage": (
                        "MORPHOLOGY_RESOLVED_TIME_FREQUENCY_RECOMMENDATION"
                        if run_resolved_dynamic_harmonic else
                        "UNRESOLVED_FAMILY_TIME_FREQUENCY_RECOMMENDATION"
                    )}
                    if run_dynamic_harmonic else
                    ({} if (run_nonstationary or run_mode_identification
                            or run_physical_interpretation) else {"outputSuffix": "v20.8"})),
        triggered_by_stage_id=request_id,
    )


def mode_identification_continuation(summary: dict[str, Any], *, request_id: str) -> StageRequest:
    dynamic = (summary.get("recommendedNextTest") == "DYNAMIC_HARMONIC_MODELING"
               and summary.get("physicalMechanismResolved") is False)
    localize = (summary.get("recommendedNextTest") == "RESIDUAL_MODE_PIXEL_LOCALIZATION"
                and summary.get("independentModeEvidenceSurvived") is True
                and summary.get("physicalMechanismResolved") is False)
    return StageRequest(
        id=_next_stage_id(request_id, "dynamic-harmonic-modeling" if dynamic else
                          ("prepare-residual-mode-localization" if localize else "finalize")),
        handler_id=("openstar.tess.dynamic-harmonic.analyze" if dynamic else
                    ("openstar.tess.residual-mode-localization.prepare" if localize else "openstar.tess.finalize")),
        parameters={} if (localize or dynamic) else {"outputSuffix": "v20.9-mode-identification"},
        triggered_by_stage_id=request_id,
    )


def confirmed_coherent_mode_identification_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Finalize the local v20.8.2 adjudication before any new data work."""
    recommendations = {
        "INDEPENDENT_STABLE_MODE": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
        "HIGHER_ORDER_HARMONIC_STRUCTURE": "DYNAMIC_HARMONIC_MODELING",
        "NO_COMPELLING_RESIDUAL_MODE": "HUMAN_SCIENTIFIC_REVIEW",
        "AMBIGUOUS_HARMONIC_OR_MODE": "HUMAN_SCIENTIFIC_REVIEW",
    }
    classification = summary.get("classification")
    method_contract = summary.get("methodContract") or {}
    if not (
        summary.get("evidenceLineage")
        == V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE
        and classification in recommendations
        and summary.get("recommendedNextTest")
        == recommendations[classification]
        and summary.get("methodContractID")
        == CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID
        and summary.get("methodContractHash")
        == confirmed_coherent_mode_method_contract_hash(method_contract)
        and summary.get("physicalMechanismResolved") is False
        and summary.get("pulsationMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and summary.get("automaticDiscoveryClaim") is False
    ):
        raise RuntimeError(
            "Confirmed coherent mode identification violated its "
            "conservative result contract."
        )
    return StageRequest(
        id=_next_stage_id(request_id, "finalize"),
        handler_id="openstar.tess.finalize",
        parameters={
            "outputSuffix": "v20.8.2-confirmed-coherent-mode-identification"
        },
        triggered_by_stage_id=request_id,
    )


def long_baseline_frequency_confirmation_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Finalize the append-only result while preserving its next-test advice."""
    if summary.get("classification") not in {
        "INDEPENDENT_STABLE_MODE_CONFIRMED",
        "HARMONIC_LOCKED_ACROSS_BASELINE",
        "NONSTATIONARY_OR_INTERMITTENT_STRUCTURE",
        "LONG_BASELINE_CONFIRMATION_INCONCLUSIVE",
    }:
        raise RuntimeError("Long-baseline confirmation classification is invalid.")
    if summary.get("physicalMechanismResolved") is not False:
        raise RuntimeError(
            "Long-baseline confirmation cannot resolve the physical mechanism."
        )
    return StageRequest(
        id=_next_stage_id(request_id, "finalize"),
        handler_id="openstar.tess.finalize",
        parameters={
            "outputSuffix":
            "v20.9.1-long-baseline-frequency-confirmation"
        },
        triggered_by_stage_id=request_id,
    )


def v20_8_long_baseline_time_frequency_confirmation_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Finalize one exact v20.8 long-baseline confirmation lineage."""
    if summary.get("classification") not in {
        "COHERENT_RESIDUAL_FREQUENCY_CONFIRMED",
        "HARMONIC_LOCKED_RESIDUAL_CONFIRMED",
        "NONSTATIONARY_RESIDUAL_STRUCTURE_CONFIRMED",
        "INTERMITTENT_RESIDUAL_STRUCTURE_CONFIRMED",
        "LONG_BASELINE_TIME_FREQUENCY_INCONCLUSIVE",
    }:
        raise RuntimeError(
            "v20.8 long-baseline time-frequency classification is invalid."
        )
    method_contract_id = summary.get("methodContractID")
    if method_contract_id == V20_8_LONG_BASELINE_METHOD_CONTRACT_ID:
        output_suffix = (
            "v20.8.1-long-baseline-time-frequency-confirmation"
        )
    elif method_contract_id == RECURRENT_RESIDUAL_METHOD_CONTRACT_ID:
        output_suffix = (
            "v20.8.2-recurrent-residual-long-baseline-confirmation"
        )
    else:
        raise RuntimeError(
            "v20.8 long-baseline method-contract lineage is invalid."
        )
    if not (
        summary.get("methodContractHash")
        == v20_8_long_baseline_method_contract_hash(
            summary.get("methodContract") or {}
        )
        and summary.get("physicalMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and summary.get("automaticDiscoveryClaim") is False
    ):
        raise RuntimeError(
            "v20.8 long-baseline confirmation cannot resolve the mechanism "
            "or upgrade the claim."
        )
    return StageRequest(
        id=_next_stage_id(request_id, "finalize"),
        handler_id="openstar.tess.finalize",
        parameters={"outputSuffix": output_suffix},
        triggered_by_stage_id=request_id,
    )

def transient_mode_validation_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Finalize the append-only transient validation and preserve its advice."""
    if summary.get("classification") not in {
        "TRANSIENT_INDEPENDENT_FREQUENCY_SUPPORTED",
        "TRANSIENT_HARMONIC_STRUCTURE_SUPPORTED",
        "RESIDUAL_STRUCTURE_RECURRENT_ACROSS_BASELINE",
        "TRANSIENT_MODE_VALIDATION_INCONCLUSIVE",
    }:
        raise RuntimeError("Transient-mode validation classification is invalid.")
    if not (
        summary.get("methodContractID")
        == (
            "openstar.tess.transient-mode-validation."
            "leave-one-detection-sector-out.v1"
        )
        and summary.get("methodContractHash")
        == transient_mode_method_contract_hash(summary.get("methodContract") or {})
        and summary.get("physicalMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and summary.get("automaticDiscoveryClaim") is False
    ):
        raise RuntimeError(
            "Transient-mode validation cannot resolve the mechanism or "
            "upgrade the claim."
        )
    return StageRequest(
        id=_next_stage_id(request_id, "finalize"),
        handler_id="openstar.tess.finalize",
        parameters={"outputSuffix": "v20.8.1-transient-mode-validation"},
        triggered_by_stage_id=request_id,
    )


def target_residual_mechanism_continuation(summary: dict[str, Any], *,
        request_id: str) -> StageRequest:
    """Route a newly computed v20.14 result without target-specific evidence."""
    corrected_unresolved = (
        summary.get("adjudicationVersion") == "route-independent-all-models-v1"
        and summary.get("classification") == "TARGET_RESIDUAL_MECHANISM_UNRESOLVED"
        and summary.get("recommendedNextTest") ==
            "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
        and summary.get("physicalMechanismResolved") is False
        and not summary.get("failClosedReasons"))
    astrophysical = (
        summary.get("classification") == "SMOOTH_TARGET_MODE_AMPLITUDE_MODULATION"
        and summary.get("physicalMechanismResolved") is False
        and summary.get("recommendedNextTest") ==
            "ASTROPHYSICAL_MECHANISM_INTERPRETATION"
        and not summary.get("failClosedReasons"))
    if corrected_unresolved:
        label = "target-residual-mechanism-predictive-validation"
        handler = "openstar.tess.target-residual-mechanism-predictive-validation.analyze"
        parameters = {}
    elif astrophysical:
        label = "target-residual-astrophysical-interpretation"
        handler = "openstar.tess.target-residual-astrophysical-interpretation.analyze"
        parameters = {}
    else:
        label = "finalize"
        handler = "openstar.tess.finalize"
        parameters = {"outputSuffix": "v20.14-intrinsic"}
    return StageRequest(id=_next_stage_id(request_id, label), handler_id=handler,
        parameters=parameters, triggered_by_stage_id=request_id)


def astrophysical_interpretation_continuation(summary: dict[str, Any], *,
        request_id: str) -> StageRequest:
    """Schedule the independent recurrence test before finalization when warranted."""
    family = summary.get("mainPhotometricFamily") or {}
    warranted = (summary.get("targetResidualMechanismResolved") is True
        and summary.get("classification") ==
            "ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED"
        and family.get("available") is True
        and family.get("physicalCycleResolved") is False)
    label = "main-family-time-domain-recurrence" if warranted else "finalize"
    handler = ("openstar.tess.main-family-time-domain-recurrence.analyze" if warranted
               else "openstar.tess.finalize")
    return StageRequest(_next_stage_id(request_id, label), handler,
        {} if warranted else {"outputSuffix":"v20.14.1-astrophysical-interpretation"},
        request_id)


def _dynamic_mode_localization_evidence(
    dynamic: dict[str, Any] | None,
    time_frequency_prepare: dict[str, Any] | None,
    time_frequency: dict[str, Any] | None,
    mode: dict[str, Any] | None,
) -> tuple[float, tuple[int, ...], dict[str, Any]] | None:
    """Adapt persisted unresolved-family evidence to the localization interface."""
    if not all((dynamic, time_frequency_prepare, time_frequency, mode)):
        return None
    candidate = (mode or {}).get("modeCandidate") or {}
    family_period = (dynamic or {}).get("referenceFamilyPeriodDays")
    orders = tuple(int(value) for value in
                   ((dynamic or {}).get("supportedHarmonicOrders") or ()))
    reference_frequency = candidate.get("frequencyCyclesPerDay")
    sectors = candidate.get("supportingSectors") or []
    time_reference = (time_frequency_prepare or {}).get("absoluteTimeReferenceDays")
    stable = ((time_frequency or {}).get("residualEvolution") or {}).get("classification")
    if not (family_period and orders and reference_frequency and sectors
            and time_reference is not None
            and (mode or {}).get("independentModeEvidenceSurvived") is True
            and (mode or {}).get("physicalMechanismResolved") is False
            and stable == "STABLE_RESIDUAL_MODE"):
        return None
    evidence = {
        "preferredFrequencyAtReference": float(reference_frequency),
        "preferredPeriodAtReferenceDays": 1.0 / float(reference_frequency),
        # A stable time-frequency classification is the persisted zero-drift
        # model; this is evidence adaptation, not a fabricated measurement.
        "fractionalFrequencyDriftPerDay": 0.0,
        "timeReferenceDays": float(time_reference),
        "preferredModel": {"signalSectors": [int(value) for value in sectors]},
        "recommendedNextTest": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
        "evidenceSource": {
            "path": "UNRESOLVED_FAMILY_DYNAMIC_HARMONIC_MODE_IDENTIFICATION",
            "residualFrequency": "modeIdentification.modeCandidate.frequencyCyclesPerDay",
            "signalSectors": "modeIdentification.modeCandidate.supportingSectors",
            "timeReference": "timeFrequencyPreparation.absoluteTimeReferenceDays",
            "drift": "timeFrequencySummary.residualEvolution.classification=STABLE_RESIDUAL_MODE",
        },
    }
    return float(family_period), orders, evidence


def dynamic_harmonic_continuation(summary: dict[str, Any], *, request_id: str) -> StageRequest:
    """Route dynamic-family evidence without assigning a physical mechanism."""
    residual = summary.get("recommendedNextTest") == "RESIDUAL_MULTIMODE_LOCALIZATION"
    refine = summary.get("recommendedNextTest") == "LOMB_SCARGLE_FREQUENCY_REFINEMENT"
    resolved_cycle = authoritative_resolved_cycle(
        morphology=None,
        dynamic_harmonic=summary,
    )
    interpret_physical = (
        summary.get("recommendedNextTest")
        == "BINARY_ROTATION_EXTERNAL_EVIDENCE"
        and resolved_cycle is not None
        and resolved_cycle.get("sourceKind") == NESTED_ALIAS_RESOLVED_CYCLE_SOURCE
    )
    try:
        resolved_period = float(summary.get("resolvedPhysicalPeriodDays"))
        reference_period = float(summary.get("referenceFamilyPeriodDays"))
    except (TypeError, ValueError):
        resolved_period = reference_period = math.nan
    predictively_resolved = (
        summary.get("evidenceLineage") in {
            "UNRESOLVED_FAMILY_TIME_FREQUENCY_RECOMMENDATION",
            "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT",
        }
        and summary.get("physicalCycleResolved") is True
        and summary.get("referencePeriodRole")
        == "PREDICTIVELY_RESOLVED_PHOTOMETRIC_CYCLE"
        and math.isfinite(resolved_period) and resolved_period > 0
        and math.isfinite(reference_period) and reference_period > 0
        and math.isclose(
            resolved_period, reference_period, rel_tol=1e-9, abs_tol=1e-12)
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "refine-harmonic-frequency" if refine else
            ("prepare-time-frequency" if residual else
             ("physical-interpretation" if interpret_physical else "finalize")),
        ),
        handler_id=("openstar.tess.dynamic-harmonic.frequency-refinement" if refine else
                    ("openstar.tess.time-frequency.prepare" if residual else
                     ("openstar.tess.physical.interpret" if interpret_physical else
                      "openstar.tess.finalize"))),
        parameters=({} if refine else
                    ({"entryReason": (
                        "RESOLVED_DYNAMIC_HARMONIC_RESIDUAL"
                        if (summary.get("evidenceLineage") ==
                            "MORPHOLOGY_RESOLVED_TIME_FREQUENCY_RECOMMENDATION"
                            or predictively_resolved)
                        else "DYNAMIC_HARMONIC_RESIDUAL")}
                     if residual else
                    ({"evidenceLineage":
                      "NESTED_ODD_HARMONIC_RESOLVED_CYCLE"}
                     if interpret_physical else
                    {"outputSuffix": (
                        "v20.10.1-nested-cycle-alias"
                        if summary.get("evidenceLineage")
                        == "UNRESOLVED_FAMILY_NESTED_ODD_HARMONIC_REASSESSMENT"
                        else "v20.10-dynamic-harmonic"
                    )}))),
        triggered_by_stage_id=request_id,
    )


def nonstationary_continuation(summary: dict[str, Any], *, request_id: str) -> StageRequest:
    """Route only the persisted unresolved residual-localization recommendation."""

    run_localization = (
        summary.get("recommendedNextTest") == "RESIDUAL_MODE_PIXEL_LOCALIZATION"
        and summary.get("physicalMechanismResolved") is False
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "prepare-residual-mode-localization" if run_localization else "finalize",
        ),
        handler_id=(
            "openstar.tess.residual-mode-localization.prepare"
            if run_localization
            else "openstar.tess.finalize"
        ),
        parameters={} if run_localization else {"outputSuffix": "v20.9"},
        triggered_by_stage_id=request_id,
    )


def residual_mode_localization_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Route only the two persisted residual-localization recommendations."""

    run_review = (
        summary.get("recommendedNextTest")
        == "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"
        and summary.get("physicalMechanismResolved") is False
    )
    run_external = (
        summary.get("recommendedNextTest")
        == "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
        and summary.get("physicalMechanismResolved") is False
        and (summary.get("crossSector") or {}).get("classification")
        == "RESIDUAL_MODE_TARGET_SUPPORTED"
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "prepare-residual-mode-localization-review" if run_review else
            ("residual-external-evidence" if run_external else "finalize"),
        ),
        handler_id=(
            "openstar.tess.residual-mode-localization-review.prepare"
            if run_review else
            (RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID if run_external
            else "openstar.tess.finalize"
            )
        ),
        parameters={} if (run_review or run_external) else {"outputSuffix": "v20.10"},
        triggered_by_stage_id=request_id,
    )


def residual_mode_localization_review_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Route only a persisted, unresolved multi-source decomposition request."""

    run_decomposition = (
        summary.get("recommendedNextTest") == "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"
        and summary.get("physicalMechanismResolved") is False
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "prepare-multi-source-residual" if run_decomposition else "finalize",
        ),
        handler_id=(
            "openstar.tess.multi-source-residual.prepare"
            if run_decomposition
            else "openstar.tess.finalize"
        ),
        parameters={} if run_decomposition else {"outputSuffix": "v20.11"},
        triggered_by_stage_id=request_id,
    )


def multisource_residual_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Continue only exact v20.12 boundaries, preserving historical routes."""

    run_intrinsic = (
        summary.get("recommendedNextTest")
        == "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"
        and summary.get("classification") == "TARGET_RESIDUAL_COMPONENT_DOMINANT"
        and summary.get("residualModeOrigin") == "TARGET_DOMINANT"
        and summary.get("physicalMechanismResolved") is False
    )
    run_prf = (
        summary.get("recommendedNextTest") == "PIXEL_RESPONSE_FUNCTION_DEBLENDING"
        and summary.get("physicalMechanismResolved") is False
    )
    label = "classify-intrinsic-target-residual" if run_intrinsic else (
        "prepare-prf-deblending" if run_prf else "finalize"
    )
    return StageRequest(
        id=_next_stage_id(request_id, label),
        handler_id=(
            "openstar.tess.intrinsic-nonstationary.analyze" if run_intrinsic else
            ("openstar.tess.official-spoc-prf-forward-modeling.prepare"
             if run_prf else "openstar.tess.finalize")
        ),
        parameters={} if (run_prf or run_intrinsic) else {"outputSuffix": "v20.12"},
        triggered_by_stage_id=request_id,
    )


def intrinsic_nonstationary_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Append the narrow temporal-mechanism experiment after exact v20.13 results."""
    run_followup = (
        summary.get("classification") in {
            "AMPLITUDE_EVOLVING_TARGET_RESIDUAL",
            "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL",
        }
        and summary.get("recommendedNextTest")
        == "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
        and summary.get("physicalMechanismResolved") is False
    )
    return StageRequest(
        id=_next_stage_id(request_id, "target-residual-mechanism" if run_followup else "finalize"),
        handler_id=("openstar.tess.target-residual-mechanism.analyze" if run_followup
                    else "openstar.tess.finalize"),
        parameters={} if run_followup else {"outputSuffix": "v20.13-intrinsic"},
        triggered_by_stage_id=request_id,
    )


def prf_catalog_counterpart_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Continue exact unresolved official-PRF recommendations into catalog lookup."""
    run_catalog = (
        summary.get("recommendedNextTest") == "CATALOG_COUNTERPART_IDENTIFICATION"
        and summary.get("physicalMechanismResolved") is False
    )
    return StageRequest(
        id=_next_stage_id(request_id, "catalog-counterpart" if run_catalog else "finalize"),
        handler_id=("openstar.tess.catalog-counterpart-identification.analyze"
                    if run_catalog else "openstar.tess.finalize"),
        parameters={} if run_catalog else {"outputSuffix": "v20.13-prf"},
        triggered_by_stage_id=request_id,
    )


def catalog_counterpart_variability_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Enter catalog-guided variability validation only for a justified candidate."""
    candidate = summary.get("preferredCandidate") or {}
    ids = candidate.get("catalogIDs") or {}
    justified = (
        candidate.get("raDeg") is not None
        and candidate.get("decDeg") is not None
        and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None)
    )
    run_localization = (
        summary.get("recommendedNextTest") == "CATALOG_GUIDED_SOURCE_LOCALIZATION"
        and summary.get("physicalMechanismResolved") is False
        and len(summary.get("plausibleCatalogCandidates") or []) >= 2
    )
    run_validation = (
        summary.get("recommendedNextTest")
        == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
        and summary.get("physicalMechanismResolved") is False
        and justified
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "prepare-catalog-guided-source-localization" if run_localization
            else "prepare-offset-source-variability" if run_validation else "finalize",
        ),
        handler_id=(
            "openstar.tess.catalog-guided-source-localization.prepare" if run_localization
            else "openstar.tess.offset-source-variability.prepare" if run_validation
            else "openstar.tess.finalize"
        ),
        parameters={} if (run_validation or run_localization)
        else {"outputSuffix": "catalog-counterpart"},
        triggered_by_stage_id=request_id,
    )


def deep_catalog_counterpart_continuation(
    summary: dict[str, Any], *, request_id: str
) -> StageRequest:
    """Enter bounded multi-source PRF localization only at the ambiguous boundary."""
    localize = (
        summary.get("version")
        == "openstar.tess-deep-catalog-counterpart-identification.v1"
        and summary.get("classification") == "AMBIGUOUS_DEEP_CATALOG_COUNTERPARTS"
        and summary.get("counterpartIdentified") is False
        and summary.get("preferredCandidate") is None
        and 2 <= len(summary.get("plausibleCatalogCandidates") or []) <= 5
        and summary.get("variabilityConfirmed") is False
        and summary.get("physicalMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and summary.get("externalDataState") == "AVAILABLE"
        and not (summary.get("queryErrors") or [])
        and summary.get("recommendedNextTest")
        == "HIGH_RESOLUTION_RESIDUAL_SOURCE_LOCALIZATION"
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "prepare-deep-catalog-guided-prf-localization" if localize else "finalize",
        ),
        handler_id=(DEEP_CATALOG_PRF_PREPARE_HANDLER_ID if localize
                    else "openstar.tess.finalize"),
        parameters={} if localize else {"outputSuffix": "deep-catalog-counterpart"},
        triggered_by_stage_id=request_id,
    )


def deep_catalog_prf_localization_continuation(*, request_id: str) -> StageRequest:
    """Finalize the bounded localization result before another science step."""
    return StageRequest(
        id=_next_stage_id(request_id, "finalize"),
        handler_id="openstar.tess.finalize",
        parameters={"outputSuffix": "deep-catalog-guided-prf-localization"},
        triggered_by_stage_id=request_id,
    )


def _build_period_evidence(
    *,
    claim_decision: dict[str, Any],
    selected_period: float | None,
    selected_source: str | None,
    primary_analysis: dict[str, Any],
    followup_interpretation: dict[str, Any] | None,
    independent_interpretation: dict[str, Any] | None,
    broad_interpretation: dict[str, Any] | None,
    harmonic_family_interpretation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate recurrent photometric evidence from a resolved physical cycle."""
    harmonic_result = harmonic_family_interpretation or broad_interpretation or {}
    family = harmonic_result.get("harmonicFamily") or {}

    recurrent = family.get("representativeRawPeriodDays")
    possible_cycle = family.get("possibleDoubleCycleDays")
    if recurrent is None:
        recurrent = selected_period

    family_interpretation = str(family.get("interpretation") or "")
    physical_cycle_resolved = bool(family.get("physicalCycleResolved"))
    claim = str(claim_decision.get("claim") or "")

    if not possible_cycle and claim in {
        "KNOWN_PERIOD_RECOVERED",
        "KNOWN_PHENOMENON_EXPLAINED",
        "INDEPENDENT_PERIOD_ESTIMATE",
    }:
        physical_cycle_resolved = selected_period is not None

    if family_interpretation == "possible-double-wave-period-family":
        physical_cycle_resolved = False

    physical_period = selected_period if physical_cycle_resolved else None

    return {
        "candidatePeriodDays": selected_period,
        "candidateSource": selected_source,
        "recurrentPhotometricPeriodDays": recurrent,
        "possiblePhysicalCycleDays": possible_cycle,
        "physicalCycleResolved": physical_cycle_resolved,
        "physicalPeriodDays": physical_period,
        "primaryRawPeriodDays": primary_analysis.get("rawCandidatePeriodDays"),
        "primaryPreferredCycleDays": primary_analysis.get("observedPeriodDays"),
        "sameSectorCandidateDays": (followup_interpretation or {}).get("selectedPeriodDays"),
        "independentTargetedCandidateDays": (independent_interpretation or {}).get("selectedPeriodDays"),
        "interpretation": (
            "physical-cycle-resolved"
            if physical_cycle_resolved
            else "photometric-period-family-physical-cycle-unresolved"
        ),
    }

def _render_report(conclusion: dict[str, Any]) -> str:
    claim = conclusion["claim"]
    target = conclusion["target"]
    analysis = conclusion.get("primaryAnalysis") or {}
    rotation = analysis.get("rotationSanity") or {}
    preferred_coverage = analysis.get("preferredCycleCoverage") or {}

    lines = [
        "# OpenStar TESS Investigation",
        "",
        f"- Investigation: `{conclusion['investigationID']}`",
        f"- TIC: `{target['ticID']}`",
        f"- Target: {target.get('targetName') or target.get('datasetID')}",
        f"- Claim level: **{claim['claim']}**",
    ]
    period_evidence = conclusion.get("periodEvidence") or {}
    if period_evidence.get("recurrentPhotometricPeriodDays") is not None:
        lines.append(
            f"- Recurrent photometric periodicity: {period_evidence.get('recurrentPhotometricPeriodDays')} days"
        )
    transit_candidate_periods = (
        period_evidence.get("transitLikeCandidatePeriodsDays") or []
    )
    if len(transit_candidate_periods) > 1:
        lines.append(
            "- Accepted distinct transit-like periods: "
            f"{transit_candidate_periods} days"
        )
    if period_evidence.get("possiblePhysicalCycleDays") is not None:
        lines.append(
            f"- Possible double-wave / physical cycle: {period_evidence.get('possiblePhysicalCycleDays')} days"
        )
    if period_evidence.get("physicalCycleResolved"):
        lines.append(f"- Physical period: {period_evidence.get('physicalPeriodDays')} days")
    else:
        lines.append("- Physical period: **unresolved**")
    lines.extend([
        f"- Evidence source: {period_evidence.get('candidateSource') or '[none]'}",
        "",
        "## Rationale",
        "",
    ])
    for reason in claim.get("rationale") or []:
        lines.append(f"- {reason}")

    lines.extend([
        "",
        "## Primary distributed result",
        "",
        f"- Period status: {analysis.get('periodStatus')}",
        f"- Confidence: {analysis.get('periodConfidence')}",
        f"- Raw candidate period: {analysis.get('rawCandidatePeriodDays')} days",
        f"- Preferred period: {analysis.get('observedPeriodDays')} days",
        f"- Preferred relation: {analysis.get('preferredPhysicalPeriodRelation')}",
        f"- Observation baseline: {analysis.get('observationBaselineDays')} days",
        f"- Preferred-cycle coverage: {preferred_coverage.get('observedCycles')}",
        "",
        "## Catalog / physical checks",
        "",
        f"- Catalog period match: {'yes' if analysis.get('bestCatalogMatch') else 'no'}",
        f"- Rotation sanity: {rotation.get('status')}",
        f"- Equatorial speed: {rotation.get('equatorialSpeedKmS')} km/s",
        f"- Minimum mass for subcritical rotation: {rotation.get('minimumMassForSubcriticalRotationMsun')} M_sun",
    ])

    if conclusion.get("followup") is not None:
        followup = conclusion["followup"]
        diagnostics = followup.get("diagnostics") or {}
        coverage = diagnostics.get("cycleCoverage") or {}
        lines.extend([
            "",
            "## Same-sector hypothesis follow-up",
            "",
            f"- Trigger: {conclusion.get('planner', {}).get('reason')}",
            f"- Reliable: {followup.get('followupReliable')}",
            f"- Selected period: {followup.get('selectedPeriodDays')} days",
            f"- Boundary hit: {diagnostics.get('boundaryHit')}",
            f"- Observed cycles: {coverage.get('observedCycles')}",
        ])

    independent = conclusion.get("independentVerification")
    if independent is not None:
        lines.extend([
            "",
            "## Independent TESS-sector verification",
            "",
            f"- Eligible sectors: {independent.get('eligibleSectorCount')}",
            f"- Supporting sectors: {independent.get('supportingSectorCount')}",
            f"- Required supporting sectors: {independent.get('requiredSupportingSectorCount')}",
        ])
        for item in independent.get("sectorResults") or []:
            coverage = item.get("cycleCoverage") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: period={item.get('candidatePeriodDays')} d, "
                f"cycles={coverage.get('observedCycles')}, "
                f"support={item.get('supportsTarget')}"
            )

    broad = conclusion.get("independentBroadVerification")
    if broad is not None:
        lines.extend([
            "",
            "## Contradiction-resolution broad independent search",
            "",
            f"- Eligible sectors: {broad.get('eligibleSectorCount')}",
            f"- Required cluster support: {broad.get('requiredClusterSupportCount')}",
            f"- Best cluster: {broad.get('bestCluster')}",
        ])
        for item in broad.get("sectorResults") or []:
            coverage = item.get("cycleCoverage") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: period={item.get('candidatePeriodDays')} d, "
                f"cycles={coverage.get('observedCycles')}, "
                f"prominence={item.get('candidatePeakProminenceRatio')}, "
                f"eligible={item.get('eligibleForClustering')}"
            )

    harmonic = conclusion.get("independentHarmonicFamilyVerification")
    if harmonic is not None:
        family = harmonic.get("harmonicFamily") or {}
        cluster = harmonic.get("bestCluster") or {}
        lines.extend([
            "",
            "## Harmonic-family reinterpretation",
            "",
            f"- Promotion eligible: {harmonic.get('promotionEligible')}",
            f"- Promotion blockers: {harmonic.get('promotionBlockers')}",
            f"- Supporting sectors: {cluster.get('sectors')}",
            f"- Representative raw periodicity: {family.get('representativeRawPeriodDays')} days",
            f"- Possible 2x physical cycle: {family.get('possibleDoubleCycleDays')} days",
            f"- Physical cycle resolved: {family.get('physicalCycleResolved')}",
        ])

    morphology = conclusion.get("morphology")
    if morphology is not None:
        lines.extend([
            "",
            "## Morphology / physical-cycle discrimination",
            "",
            f"- Morphology class: {morphology.get('morphologyClass')}",
            f"- Phenomenology: {morphology.get('phenomenology')}",
            f"- Eligible sectors: {morphology.get('eligibleSectorCount')}",
            f"- Independent eligible sectors: {morphology.get('independentEligibleSectorCount')}",
            f"- Required independent support: {morphology.get('requiredIndependentSupportCount')}",
            f"- Raw-cycle supporters: {morphology.get('rawCycleSupportingSectors')}",
            f"- Double-cycle supporters: {morphology.get('doubleCycleSupportingSectors')}",
            f"- Physical cycle resolved: {morphology.get('physicalCycleResolved')}",
            f"- Resolved physical period: {morphology.get('resolvedPhysicalPeriodDays')} days",
        ])
        for item in morphology.get("sectorResults") or []:
            double_metrics = item.get("doubleWaveMetrics") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: rawEV={(item.get('rawProfile') or {}).get('explainedVariance')}, "
                f"doubleEV={(item.get('doubleProfile') or {}).get('explainedVariance')}, "
                f"doubleGain={item.get('doubleExplainedVarianceImprovement')}, "
                f"halfDiff={double_metrics.get('halfCycleDifferenceRatio')}, "
                f"rawSupport={item.get('supportsRawCycle')}, "
                f"doubleSupport={item.get('supportsDoubleCycle')}"
            )

    blind_transit = conclusion.get("blindTransitSearch")
    if blind_transit is not None:
        ephemeris = blind_transit.get("linearEphemeris") or {}
        candidate_signals = blind_transit.get("candidateSignals") or []
        sector_availability = (
            blind_transit.get("independentSectorAvailability") or {}
        )
        residual_census = (
            blind_transit.get("exhaustedSectorResidualFamilyCensus") or {}
        )
        lines.extend([
            "",
            "## Software-blind transit-period search",
            "",
            f"- Classification: {blind_transit.get('classification')}",
            f"- Candidate period: {blind_transit.get('candidatePeriodDays')} days",
            f"- Primary sector supported: {blind_transit.get('primarySectorSupported')}",
            f"- Supporting independent sectors: {blind_transit.get('supportingIndependentSectors')}",
            f"- Linear ephemeris coherent: {ephemeris.get('coherent')}",
            f"- Companion nature resolved: {blind_transit.get('companionNatureResolved')}",
            f"- Catalog answer key used: {blind_transit.get('catalogAnswerKeyUsed')}",
            "- Official independent-sector inventory exhausted: "
            f"{sector_availability.get('allCandidateSectorsPrepared')}",
            f"- Recommended next test: {blind_transit.get('recommendedNextTest')}",
        ])
        if residual_census:
            if blind_transit.get("distributedResidualCandidateGeneration"):
                lines.append(
                    "- Expanded residual-family census: frozen candidate "
                    "inventory only; no family score directly satisfied a "
                    "transit or claim gate"
                )
            else:
                lines.append(
                    "- Expanded residual-family census: audit only; candidate "
                    "selection and claim decision unchanged"
                )
            for method in residual_census.get("methods") or []:
                audit = method.get("candidateGenerationAudit") or {}
                lines.append(
                    "- Residual census "
                    f"{method.get('residualSearchMethod')}: "
                    f"{audit.get('recordedFamilyCount')} coarse families"
                )
        distributed_residual = blind_transit.get(
            "distributedResidualCandidateGeneration"
        ) or {}
        if distributed_residual:
            generic = distributed_residual.get(
                "genericCandidateInterpretation"
            ) or {}
            validation = distributed_residual.get(
                "serverTransitValidation"
            ) or {}
            lines.extend([
                "- Distributed residual worker semantics: "
                f"{generic.get('workerSemantics')}",
                "- Worker candidate selection authority: false",
                "- Worker claim-decision authority: false",
                "- Server transit thresholds changed: false",
                "- Dual-residual family groups corroborated: "
                f"{validation.get('corroboratedFamilyGroupCount')}",
                "- Additional server-validated candidate accepted: "
                f"{validation.get('accepted')}",
            ])
        if candidate_signals:
            lines.append(
                f"- Accepted distinct transit-like clocks: {len(candidate_signals)}"
            )
            for candidate in candidate_signals:
                lines.append(
                    "- Candidate "
                    f"{candidate.get('candidateIndex')}: "
                    f"period={candidate.get('candidatePeriodDays')} d, "
                    "supportingIndependentSectors="
                    f"{candidate.get('supportingIndependentSectors')}, "
                    "ephemerisCoherent="
                    f"{(candidate.get('linearEphemeris') or {}).get('coherent')}"
                )
        for item in blind_transit.get("sectorResults") or []:
            lines.append(
                "- Sector "
                f"{item.get('sector')} ({item.get('role')}): "
                f"SNR={item.get('snr')}, duration={item.get('durationDays')} d, "
                f"usable={item.get('usable')}"
            )

    physical = conclusion.get("physicalInterpretation")
    if physical is not None:
        fourier = physical.get("crossSectorFourierSummary") or {}
        rotation = physical.get("rotationConstraint") or {}
        contamination = physical.get("contaminationScreen") or {}
        lines.extend([
            "",
            "## Physical-mechanism discrimination",
            "",
            f"- Physical mechanism resolved: {physical.get('physicalMechanismResolved')}",
            f"- Preferred photometric hypothesis: {physical.get('preferredPhotometricHypothesis')}",
            f"- Recommended next test: {physical.get('recommendedNextTest')}",
            f"- Independent Fourier sectors: {fourier.get('independentEligibleSectorCount')}",
            f"- Harmonic-dominant sectors: {fourier.get('independentHarmonicDominantSectors')}",
            f"- Relative harmonic phase concentration: {fourier.get('relativeHarmonicPhaseConcentration')}",
            f"- Harmonic amplitude variation fraction: {fourier.get('firstHarmonicAmplitudeVariationFraction')}",
            f"- Rotation status at resolved physical period: {rotation.get('status')}",
            f"- Equatorial speed: {rotation.get('equatorialSpeedKmS')} km/s",
            f"- Minimum mass for subcritical rotation: {rotation.get('minimumMassForSubcriticalRotationMsun')} M_sun",
            f"- TIC contamination ratio: {contamination.get('ticContaminationRatio')}",
            f"- Existing metadata can exclude TESS aperture contamination: {contamination.get('canExcludeTessApertureContamination')}",
            "",
            "### Mechanism ranking",
            "",
        ])
        for item in physical.get("mechanismRankings") or []:
            lines.append(
                f"- {item.get('hypothesis')}: score={item.get('score')}, "
                f"level={item.get('evidenceLevel')}, reasons={item.get('reasons')}, "
                f"cautions={item.get('cautions')}"
            )
        lines.extend([
            "",
            "### Per-sector Fourier model",
            "",
        ])
        for item in physical.get("sectorResults") or []:
            fit = item.get("fourierFit") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: R2={fit.get('explainedVariance')}, "
                f"A1={fit.get('fundamentalAmplitude')}, "
                f"A2={fit.get('firstHarmonicAmplitude')}, "
                f"A2/A1={fit.get('firstHarmonicToFundamentalAmplitudeRatio')}, "
                f"component={fit.get('dominantFourierComponent')}, "
                f"relativePhase={fit.get('translationInvariantRelativeHarmonicPhaseRad')}"
            )

    localization = conclusion.get("sourceLocalization")
    if localization is not None:
        cross = localization.get("crossSector") or {}
        lines.extend([
            "",
            "## Pixel-level source localization",
            "",
            f"- Classification: {cross.get('classification')}",
            f"- Variable signal origin: {cross.get('variableSignalOrigin')}",
            f"- Independent eligible sectors: {cross.get('independentEligibleSectorCount')}",
            f"- Required independent support: {cross.get('requiredIndependentSupportCount')}",
            f"- Target-supporting sectors: {cross.get('targetSupportingSectors')}",
            f"- Off-target sectors: {cross.get('offTargetSectors')}",
            f"- Ambiguous sectors: {cross.get('ambiguousSectors')}",
            f"- Median sky separation: {cross.get('medianSkySeparationArcsec')} arcsec",
            f"- Off-target sky-offset scatter: {cross.get('offTargetSkyOffsetScatterArcsec')} arcsec",
            f"- Recommended next test: {localization.get('recommendedNextTest')}",
            "- Static aperture contamination cleared: False",
            "",
            "### Per-sector localization",
            "",
        ])
        for item in localization.get("sectorResults") or []:
            source = item.get("source") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: role={item.get('role')}, "
                f"source={source.get('sourceType')}, "
                f"offsetPixels={item.get('offsetPixels')}, "
                f"skySeparationArcsec={item.get('skySeparationArcsec')}, "
                f"classification={item.get('classification')}"
            )

    multimode = conclusion.get("multiModeDecomposition")
    if multimode is not None:
        recurrent = multimode.get("bestRecurrentSecondaryMode") or {}
        lines.extend([
            "",
            "## Residual multi-mode frequency decomposition",
            "",
            f"- Classification: {multimode.get('classification')}",
            f"- Iterations completed: {multimode.get('iterationsCompleted')}",
            f"- Independent sectors with residual modes: {multimode.get('independentSectorsWithAcceptedResidualModes')}",
            f"- Resolved-near-primary sectors: {multimode.get('resolvedNearPrimaryFamilySectors')}",
            f"- Recurrent secondary period: {recurrent.get('medianPeriodDays')} days",
            f"- Recurrent secondary supporting sectors: {recurrent.get('independentSectors')}",
            f"- Physical mechanism resolved: {multimode.get('physicalMechanismResolved')}",
            f"- Recommended next test: {multimode.get('recommendedNextTest')}",
        ])

    time_frequency = conclusion.get("timeFrequencyEvolution")
    if time_frequency is not None:
        residual = time_frequency.get("residualEvolution") or {}
        family = time_frequency.get("familyEvolution") or {}
        best_cluster = residual.get("bestCluster") or {}
        lines.extend([
            "",
            "## Time-frequency evolution",
            "",
            f"- Overall classification: {time_frequency.get('classification')}",
            f"- Sliding windows: {time_frequency.get('windowCount')}",
            f"- Accepted residual-window features: {time_frequency.get('acceptedFeatureCount')}",
            f"- Independent accepted features: {time_frequency.get('acceptedIndependentFeatureCount')}",
            f"- Independent sectors with accepted features: {time_frequency.get('acceptedIndependentSectors')}",
            f"- Residual evolution: {residual.get('classification')}",
            f"- Established-family evolution: {family.get('classification')}",
            f"- Best residual cluster period: {best_cluster.get('medianPeriodDays')} days",
            f"- Best residual cluster independent sectors: {best_cluster.get('independentSectors')}",
            f"- Physical mechanism resolved: {time_frequency.get('physicalMechanismResolved')}",
            f"- Recommended next test: {time_frequency.get('recommendedNextTest')}",
            "",
            "### Sliding-window results",
            "",
        ])
        for item in time_frequency.get("windowResults") or []:
            lines.append(
                "- Sector "
                f"{item.get('sector')} window {item.get('windowIndex')}: "
                f"period={item.get('candidatePeriodDays')} d, "
                f"prominence={item.get('candidatePeakProminenceRatio')}, "
                f"accepted={item.get('acceptedTimeFrequencyFeature')}, "
                f"nearFamily={item.get('nearEstablishedFamily')}"
            )

    v20_8_confirmation = conclusion.get(
        "longBaselineTimeFrequencyConfirmation"
    )
    if v20_8_confirmation is not None:
        lines.extend([
            "",
            "## v20.8 long-baseline time-frequency confirmation",
            "",
            f"- Method contract: {v20_8_confirmation.get('methodContractID')}",
            "- Method contract hash: "
            f"{v20_8_confirmation.get('methodContractHash')}",
            "- Leave-one-independent-sector-out validation: True",
            "- Long-baseline frequency resolution: "
            f"{v20_8_confirmation.get('longBaselineFrequencyResolutionCyclesPerDay')} "
            "cycles/day",
            "- Aggregate predictive decision: "
            f"{v20_8_confirmation.get('aggregateDecision')}",
            "- Frequency stability: "
            f"{v20_8_confirmation.get('frequencyStability')}",
            f"- Classification: {v20_8_confirmation.get('classification')}",
            "- Physical mechanism resolved: "
            f"{v20_8_confirmation.get('physicalMechanismResolved')}",
            f"- Claim level changed: {v20_8_confirmation.get('claimLevelChanged')}",
            "- Recommended next test: "
            f"{v20_8_confirmation.get('recommendedNextTest')}",
            "",
            "### Held-out sector evidence",
            "",
        ])
        for fold in v20_8_confirmation.get("perSectorEvidence") or []:
            lines.append(
                "- Held-out sector "
                f"{fold.get('heldOutSector')}: training={fold.get('trainingSectors')}, "
                "learnedCoherentFrequency="
                f"{fold.get('learnedCoherentFrequencyCyclesPerDay')}, "
                f"exactHarmonicFrequency={fold.get('exactHarmonicFrequencyCyclesPerDay')}, "
                f"separation={fold.get('frequencySeparationCyclesPerDay')}, "
                f"predictiveBIC={fold.get('predictiveBIC')}, "
                f"deltas={fold.get('predictiveBICDeltas')}, "
                f"support={fold.get('support')}, "
                "failureOrInsufficiencyReasons="
                f"{fold.get('failureOrInsufficiencyReasons')}"
            )

    transient_validation = conclusion.get("transientModeValidation")
    if transient_validation is not None:
        lines.extend([
            "",
            "## Transient residual-mode validation",
            "",
            f"- Method contract: {transient_validation.get('methodContractID')}",
            "- Method contract hash: "
            f"{transient_validation.get('methodContractHash')}",
            "- Leave-one-transient-detection-sector-out validation: True",
            "- Control windows used for selection: False",
            "- Exact tested harmonic frequency: "
            f"{transient_validation.get('exactHarmonicFrequencyCyclesPerDay')} "
            "cycles/day",
            "- Learned transient frequency from all detection windows: "
            f"{transient_validation.get('allDetectionWindowLearnedFrequencyCyclesPerDay')} "
            "cycles/day",
            "- Long-baseline frequency resolution: "
            f"{transient_validation.get('longBaselineFrequencyResolutionCyclesPerDay')} "
            "cycles/day",
            "- Aggregate predictive decision: "
            f"{transient_validation.get('aggregateDecision')}",
            f"- Classification: {transient_validation.get('classification')}",
            "- Physical mechanism resolved: "
            f"{transient_validation.get('physicalMechanismResolved')}",
            f"- Claim level changed: {transient_validation.get('claimLevelChanged')}",
            "- Recommended next test: "
            f"{transient_validation.get('recommendedNextTest')}",
            "",
            "### Held-out transient detection sectors",
            "",
        ])
        for fold in transient_validation.get("perDetectionSectorEvidence") or []:
            lines.append(
                "- Held-out sector "
                f"{fold.get('heldOutSector')}: "
                f"training={fold.get('trainingDetectionSectors')}, "
                "learnedTransientFrequency="
                f"{fold.get('learnedTransientFrequencyCyclesPerDay')}, "
                f"exactHarmonicFrequency={fold.get('exactHarmonicFrequencyCyclesPerDay')}, "
                f"separation={fold.get('frequencySeparationCyclesPerDay')}, "
                f"predictiveBIC={fold.get('predictiveBIC')}, "
                f"deltas={fold.get('predictiveBICDeltas')}, "
                f"support={fold.get('support')}, "
                "failureOrInsufficiencyReasons="
                f"{fold.get('failureOrInsufficiencyReasons')}"
            )
        lines.extend(["", "### Untouched control windows", ""])
        for item in transient_validation.get("perControlWindowEvidence") or []:
            lines.append(
                f"- Sector {item.get('sector')} window {item.get('windowIndex')}: "
                f"role={item.get('role')}, predictiveBIC={item.get('predictiveBIC')}, "
                f"deltas={item.get('predictiveBICDeltas')}, "
                f"support={item.get('support')}, "
                "failureOrInsufficiencyReasons="
                f"{item.get('failureOrInsufficiencyReasons')}"
            )

    nonstationary = conclusion.get("nonstationaryModeling")
    if nonstationary is not None:
        preferred = nonstationary.get("preferredModel") or {}
        comparison = nonstationary.get("modelComparison") or {}
        lines.extend([
            "",
            "## Long-baseline nonstationary mode modeling",
            "",
            f"- Classification: {nonstationary.get('classification')}",
            f"- Generic workload: {(nonstationary.get('distributedModeling') or {}).get('workloadID')}",
            f"- Distributed work units: {(nonstationary.get('distributedModeling') or {}).get('totalWorkUnits')}",
            f"- Preferred model: {comparison.get('bestModelID')}",
            f"- BIC improvement over null: {comparison.get('bicImprovementOverNull')}",
            f"- Preferred period at reference time: {nonstationary.get('preferredPeriodAtReferenceDays')} days",
            f"- Fractional frequency drift/day: {nonstationary.get('fractionalFrequencyDriftPerDay')}",
            f"- Frequency derivative: {nonstationary.get('frequencyDerivativeCyclesPerDaySquared')} cycles/day²",
            f"- Signal sectors: {preferred.get('signalSectors')}",
            f"- Physical mechanism resolved: {nonstationary.get('physicalMechanismResolved')}",
            f"- Recommended next test: {nonstationary.get('recommendedNextTest')}",
            "",
            "### Competing models",
            "",
        ])
        for model in comparison.get("models") or []:
            lines.append(
                f"- {model.get('modelID')}: BIC={model.get('bic')}, "
                f"period={model.get('periodDays')} d, q={model.get('fractionalFrequencyDriftPerDay')}, "
                f"sectors={model.get('signalSectors')}"
            )

    mode_identification = conclusion.get("modeIdentification")
    if mode_identification is not None:
        relation = mode_identification.get("harmonicRelation") or {}
        comparison = mode_identification.get("modelComparison") or {}
        candidate = mode_identification.get("residualCandidate") or {}
        lines.extend([
            "", "## Stable residual mode identification", "",
            f"- Method contract: {mode_identification.get('methodContractID')}",
            f"- Method contract hash: {mode_identification.get('methodContractHash')}",
            f"- Evidence lineage: {mode_identification.get('evidenceLineage')}",
            f"- Established period family: {mode_identification.get('establishedPeriodFamily')}",
            f"- Residual candidate period/frequency: {candidate.get('refinedPeriodDays')} days / {candidate.get('refinedFrequencyCyclesPerDay')} cycles/day",
            f"- Tested harmonic relation: order {relation.get('testedOrder')}, commensurate within measured resolution={relation.get('commensurateWithinResolution')}",
            f"- BIC model comparison: {comparison}",
            f"- Independent-sector support: {mode_identification.get('independentSectorSupport')}",
            f"- Classification: {mode_identification.get('classification')}",
            f"- Independent-mode evidence survived: {mode_identification.get('independentModeEvidenceSurvived')}",
            f"- Pulsation interpretation: {mode_identification.get('pulsationInterpretation')}",
            f"- Pulsation mechanism resolved: {mode_identification.get('pulsationMechanismResolved')}",
            f"- Physical mechanism resolved: {mode_identification.get('physicalMechanismResolved')}",
            f"- Claim level changed: {mode_identification.get('claimLevelChanged')}",
            f"- Recommended next test: {mode_identification.get('recommendedNextTest')}",
        ])

    long_baseline_confirmation = conclusion.get(
        "longBaselineFrequencyConfirmation"
    )
    if long_baseline_confirmation is not None:
        stability = long_baseline_confirmation.get("frequencyStability") or {}
        aggregate = long_baseline_confirmation.get("aggregateDecision") or {}
        lines.extend([
            "",
            "## Long-baseline frequency confirmation",
            "",
            "- Method contract: "
            f"{long_baseline_confirmation.get('methodContractID')}",
            "- Method contract hash: "
            f"{long_baseline_confirmation.get('methodContractHash')}",
            "- Leave-one-independent-sector-out validation: True",
            "- Long-baseline frequency resolution: "
            f"{long_baseline_confirmation.get('longBaselineFrequencyResolutionCyclesPerDay')} "
            "cycles/day",
            f"- Frequency stability: {stability}",
            f"- Aggregate predictive decision: {aggregate}",
            f"- Classification: {long_baseline_confirmation.get('classification')}",
            "- Physical mechanism resolved: "
            f"{long_baseline_confirmation.get('physicalMechanismResolved')}",
            "- Claim level changed: "
            f"{long_baseline_confirmation.get('claimLevelChanged')}",
            "- Recommended next test: "
            f"{long_baseline_confirmation.get('recommendedNextTest')}",
            "",
            "### Held-out sector evidence",
            "",
        ])
        for fold in long_baseline_confirmation.get("perSectorEvidence") or []:
            lines.append(
                "- Held-out sector "
                f"{fold.get('heldOutSector')}: training={fold.get('trainingSectors')}, "
                "learnedIndependentFrequency="
                f"{fold.get('learnedIndependentFrequencyCyclesPerDay')}, "
                f"exactHarmonicFrequency={fold.get('exactHarmonicFrequencyCyclesPerDay')}, "
                f"separation={fold.get('frequencySeparationCyclesPerDay')}, "
                f"predictiveBIC={fold.get('predictiveBIC')}, "
                f"deltas={fold.get('predictiveBICDeltas')}, "
                f"support={fold.get('support')}, "
                "failureOrInsufficiencyReasons="
                f"{fold.get('failureOrInsufficiencyReasons')}"
            )

    dynamic_harmonic = conclusion.get("dynamicHarmonicModeling")
    if dynamic_harmonic is not None:
        lines.extend([
            "", "## Dynamic harmonic modeling", "",
            f"- Reference physical/family period: {dynamic_harmonic.get('referenceFamilyPeriodDays')} days",
            f"- Harmonic orders tested: {dynamic_harmonic.get('harmonicOrdersTested')}",
            f"- Per-sector amplitudes/phases: {dynamic_harmonic.get('sectorFits')}",
            f"- Model comparison/BIC: {dynamic_harmonic.get('modelComparison')}",
            f"- Amplitude evolution: {dynamic_harmonic.get('amplitudeEvolution')}",
            f"- Phase evolution: {dynamic_harmonic.get('phaseEvolution')}",
            f"- Harmonic ratios: {dynamic_harmonic.get('harmonicAmplitudeRatios')}",
            f"- Coherence assessment: {dynamic_harmonic.get('coherenceAssessment')}",
            f"- Residual unexplained variance: {dynamic_harmonic.get('residualUnexplainedVarianceFraction')}",
            f"- Cycle-alias resolution: {dynamic_harmonic.get('periodAliasResolution')}",
            f"- Resolved photometric cycle: {dynamic_harmonic.get('resolvedPhysicalPeriodDays')} days",
            f"- Classification: {dynamic_harmonic.get('classification')}",
            f"- Physical mechanism resolved: {dynamic_harmonic.get('physicalMechanismResolved')}",
            f"- Recommended next test: {dynamic_harmonic.get('recommendedNextTest')}",
        ])
    frequency_refinement = conclusion.get("dynamicHarmonicFrequencyRefinement")
    if frequency_refinement is not None:
        lines.extend([
            "", "### Harmonic-family frequency refinement", "",
            f"- Original period: {frequency_refinement.get('originalPeriodDays')} days",
            f"- Refined period: {frequency_refinement.get('refinedPeriodDays')} days",
            f"- Evidence: {frequency_refinement.get('evidence')}",
            f"- Generic workload semantics: {frequency_refinement.get('distributedRefinement')}",
            f"- Physical period change claimed: {frequency_refinement.get('physicalPeriodChangeClaimed')}",
        ])

    residual_localization = conclusion.get("residualModeLocalization")
    if residual_localization is not None:
        cross = residual_localization.get("crossSector") or {}
        distributed = residual_localization.get("distributedLocalization") or {}
        lines.extend([
            "",
            "## Drifting residual-mode pixel localization",
            "",
            f"- Classification: {cross.get('classification')}",
            f"- Residual-mode origin: {cross.get('residualModeOrigin')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Residual period at reference: {residual_localization.get('residualPeriodAtReferenceDays')} days",
            f"- Fractional frequency drift/day: {residual_localization.get('fractionalFrequencyDriftPerDay')}",
            f"- Independent eligible sectors: {cross.get('independentEligibleSectorCount')}",
            f"- Target-supporting sectors: {cross.get('targetSupportingSectors')}",
            f"- Off-target sectors: {cross.get('offTargetSectors')}",
            f"- Ambiguous sectors: {cross.get('ambiguousSectors')}",
            f"- Median sky separation: {cross.get('medianSkySeparationArcsec')} arcsec",
            f"- Physical mechanism resolved: {residual_localization.get('physicalMechanismResolved')}",
            f"- Recommended next test: {residual_localization.get('recommendedNextTest')}",
            "",
            "### Sector localization",
            "",
        ])
        for item in residual_localization.get("sectorResults") or []:
            lines.append(
                f"- Sector {item.get('sector')}: role={item.get('role')}, "
                f"offsetPixels={item.get('offsetPixels')}, "
                f"skySeparationArcsec={item.get('skySeparationArcsec')}, "
                f"peakPower={item.get('peakPower')}, "
                f"powerContrast={item.get('powerContrast')}, "
                f"classification={item.get('classification')}"
            )

    residual_external = conclusion.get("residualExternalEvidence")
    if residual_external is not None:
        spatial = residual_external.get("spatialEvidence") or {}
        lines.extend([
            "",
            "## Frozen external variability and binary evidence",
            "",
            f"- Method contract: {residual_external.get('methodContractID')}",
            f"- Method contract hash: {residual_external.get('methodContractHash')}",
            f"- Classification: {residual_external.get('classification')}",
            f"- Catalog coverage complete: {residual_external.get('catalogCoverageComplete')}",
            f"- Residual period: {residual_external.get('residualPeriodAtReferenceDays')} days",
            f"- Established physical period: {residual_external.get('establishedPhysicalPeriodDays')} days",
            f"- Target-supporting residual sectors: {spatial.get('targetSupportingSectors')}",
            f"- Retained off-target residual sectors: {spatial.get('offTargetSectors')}",
            f"- Spatial cautions: {spatial.get('cautions')}",
            f"- Insufficiency reasons: {residual_external.get('insufficiencyReasons')}",
            f"- Physical mechanism resolved: {residual_external.get('physicalMechanismResolved')}",
            f"- Claim level changed: {residual_external.get('claimLevelChanged')}",
            f"- Recommended next test: {residual_external.get('recommendedNextTest')}",
            "",
            "### Catalog evidence",
            "",
        ])
        for item in residual_external.get("catalogEvidence") or []:
            lines.append(
                f"- {item.get('source')} {item.get('stableObjectID')}: "
                f"classification={item.get('classification')}, "
                f"family={item.get('classificationFamily')}, "
                f"targetAssociated={item.get('targetAssociated')}, "
                f"catalogPeriodDays={item.get('catalogPeriodDays')}, "
                f"periodComparisons={item.get('periodComparisons')}"
            )

    residual_mechanism = conclusion.get(
        "targetResidualAstrophysicalMechanismFollowup"
    )
    if residual_mechanism is not None:
        spatial = residual_mechanism.get("spatialEvidence") or {}
        rotation = residual_mechanism.get("rotationConstraintAtResidualPeriod") or {}
        lines.extend([
            "",
            "## Target-residual astrophysical mechanism follow-up",
            "",
            f"- Method contract: {residual_mechanism.get('methodContractID')}",
            f"- Method contract hash: {residual_mechanism.get('methodContractHash')}",
            f"- Classification: {residual_mechanism.get('classification')}",
            f"- Residual period: {residual_mechanism.get('residualPeriodAtReferenceDays')} days",
            f"- Established physical period: {residual_mechanism.get('establishedPhysicalPeriodDays')} days",
            f"- Rotation sanity at residual period: {rotation.get('status')}",
            f"- Retained off-target residual sectors: {spatial.get('offTargetSectors')}",
            f"- Spatial cautions: {spatial.get('cautions')}",
            f"- Insufficiency reasons: {residual_mechanism.get('insufficiencyReasons')}",
            f"- Physical mechanism resolved: {residual_mechanism.get('physicalMechanismResolved')}",
            f"- Claim level changed: {residual_mechanism.get('claimLevelChanged')}",
            f"- Recommended next test: {residual_mechanism.get('recommendedNextTest')}",
            "",
            "### Adjudicated frozen catalog evidence",
            "",
        ])
        for item in residual_mechanism.get("adjudicatedCatalogEvidence") or []:
            lines.append(
                f"- {item.get('source')} {item.get('stableObjectID')}: "
                f"classification={item.get('classification')}, "
                f"family={item.get('classificationFamily')}, "
                f"adjudication={item.get('adjudication')}, "
                f"supportsResidualPeriod={item.get('supportsResidualPeriod')}, "
                f"supportsEstablishedFamily={item.get('supportsEstablishedFamily')}"
            )


    residual_review = conclusion.get("residualModeLocalizationReview")
    if residual_review is not None:
        cross_time = residual_review.get("crossTime") or {}
        distributed = residual_review.get("distributedLocalizationReview") or {}
        lines.extend([
            "",
            "## Time-resolved residual-mode source localization review",
            "",
            f"- Classification: {cross_time.get('classification')}",
            f"- Residual-mode origin: {cross_time.get('residualModeOrigin')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target-dominant sectors: {cross_time.get('targetDominantSectors')}",
            f"- Off-target-dominant sectors: {cross_time.get('offTargetDominantSectors')}",
            f"- Source-switching sectors: {cross_time.get('sourceSwitchingSectors')}",
            f"- Off-target sky scatter: {cross_time.get('offTargetSkyOffsetScatterArcsec')} arcsec",
            f"- Physical mechanism resolved: {residual_review.get('physicalMechanismResolved')}",
            f"- Recommended next test: {residual_review.get('recommendedNextTest')}",
            "",
            "### Time-window localization",
            "",
        ])
        for item in residual_review.get("windowResults") or []:
            lines.append(
                f"- Sector {item.get('sector')} window {item.get('windowIndex')}: "
                f"offsetPixels={item.get('offsetPixels')}, "
                f"skySeparationArcsec={item.get('skySeparationArcsec')}, "
                f"peakPower={item.get('peakPower')}, "
                f"powerContrast={item.get('powerContrast')}, "
                f"classification={item.get('classification')}"
            )

    neighbor_review = conclusion.get("neighborCatalogPixelResponseReview")
    if neighbor_review is not None:
        decision = neighbor_review.get("aggregateDecision") or {}
        lines.extend([
            "",
            "## Neighbor catalog and frozen pixel-response review",
            "",
            f"- Method contract: {neighbor_review.get('methodContractID')}",
            f"- Method contract hash: {neighbor_review.get('methodContractHash')}",
            f"- Classification: {neighbor_review.get('classification')}",
            f"- Residual-mode origin: {neighbor_review.get('residualModeOrigin')}",
            f"- Catalog query complete: {neighbor_review.get('catalogQueryComplete')}",
            f"- Catalog query errors: {neighbor_review.get('catalogQueryErrors')}",
            f"- Catalog candidates: {len(neighbor_review.get('catalogCandidates') or [])}",
            f"- Target-supporting sectors: {decision.get('targetSupportingSectors')}",
            f"- Best neighbor: {decision.get('bestNeighborSourceID')}",
            f"- Best-neighbor supporting sectors: {decision.get('bestNeighborSupportingSectors')}",
            f"- Physical mechanism resolved: {neighbor_review.get('physicalMechanismResolved')}",
            f"- Claim level changed: {neighbor_review.get('claimLevelChanged')}",
            f"- Recommended next test: {neighbor_review.get('recommendedNextTest')}",
            "",
            "### Independent-sector pixel-response evidence",
            "",
        ])
        for item in neighbor_review.get("sectorEvidence") or []:
            lines.append(
                f"- Sector {item.get('sector')}: "
                f"qualityWindows={item.get('qualityWindowCount')}, "
                f"classification={item.get('classification')}, "
                f"supportedSource={item.get('supportedSourceID')}, "
                f"windowSupport={item.get('windowSupportCounts')}"
            )

    multisource = conclusion.get("multiSourceResidualDecomposition")
    if multisource is not None:
        distributed = multisource.get("distributedDecomposition") or {}
        lines.extend([
            "",
            "## Multi-source residual decomposition",
            "",
            f"- Classification: {multisource.get('classification')}",
            f"- Residual-mode origin: {multisource.get('residualModeOrigin')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target component: {multisource.get('targetComponentID')}",
            f"- Best offset component: {multisource.get('bestOffsetComponentID')}",
            f"- Physical mechanism resolved: {multisource.get('physicalMechanismResolved')}",
            f"- Recommended next test: {multisource.get('recommendedNextTest')}",
            "",
            "### Spatial component evidence",
            "",
        ])
        for item in multisource.get("componentSummaries") or []:
            lines.append(
                f"- {item.get('componentID')}: type={item.get('componentType')}, "
                f"independentSupport={item.get('independentSupportCount')}, "
                f"sectors={item.get('independentSupportingSectors')}, "
                f"combinedPower={item.get('combinedPower')}, "
                f"combinedPeriod={item.get('combinedPeriodDays')} d"
            )

    predictive_validation = conclusion.get(
        "targetResidualMechanismPredictiveValidation"
    )
    if predictive_validation is not None:
        lines.extend([
            "",
            "## Target-residual mechanism predictive validation (v20.16)",
            "",
            f"- Classification: {predictive_validation.get('classification')}",
            f"- Recommended next test: {predictive_validation.get('recommendedNextTest')}",
            "- Replicated predictive mechanisms: "
            f"{predictive_validation.get('replicatedPredictiveMechanisms')}",
            "- Supporting sectors by replicated mechanism: "
            f"{predictive_validation.get('replicatedPredictiveMechanismSupportingSectorIDs')}",
            "- Conservative validation limitations (fail-closed reasons): "
            f"{predictive_validation.get('failClosedReasons')}",
            "- Execution: deterministic held-out validation with local training-only refits; "
            "no distributed work or archive query was performed by v20.16.",
            "",
            "### Compact per-sector predictive evidence",
            "",
        ])
        for item in predictive_validation.get("sectorPredictiveEvidence") or []:
            lines.append(
                f"- Sector {item.get('sector')}: "
                f"classification={item.get('sectorClassification')}, "
                f"best={item.get('bestPredictiveModel')}, "
                f"secondBest={item.get('secondBestPredictiveModel')}, "
                f"deltaLogLikelihood={item.get('predictiveDeltaLogLikelihood')}, "
                f"foldWins={item.get('foldWinsByModel')}, "
                f"fairComparison={item.get('fairAllModelComparisonCompleted')}, "
                f"morphologyGateBlocked={item.get('morphologyGateBlockedPromotion')}"
            )

    offset_source = conclusion.get("offsetResidualSourceIdentification")
    if offset_source is not None:
        component = offset_source.get("component") or {}
        sky = component.get("componentSky") or {}
        best = offset_source.get("bestCandidate") or {}
        ids = best.get("catalogIDs") or {}
        vsx = best.get("vsxMatch") or {}
        simbad = best.get("simbadMatch") or {}
        gaia_var = best.get("gaiaVariability") or {}
        lines.extend([
            "",
            "## Offset residual source identification",
            "",
            f"- Classification: {offset_source.get('classification')}",
            f"- Offset component: {component.get('componentID')}",
            f"- Component sky position: RA={sky.get('raDeg')} deg, Dec={sky.get('decDeg')} deg",
            f"- Separation from TIC target: {component.get('targetSeparationArcsec')} arcsec",
            f"- Secure association radius: {(offset_source.get('search') or {}).get('secureAssociationRadiusArcsec')} arcsec",
            f"- Best candidate separation: {best.get('separationArcsec')} arcsec",
            f"- TIC counterpart: {ids.get('ticID')}",
            f"- Gaia DR3 counterpart: {ids.get('gaiaDR3SourceID')}",
            f"- VSX match: {vsx.get('name')}",
            f"- VSX type: {vsx.get('type')}",
            f"- VSX period: {vsx.get('periodDays')} days",
            f"- SIMBAD match: {simbad.get('mainID')}",
            f"- SIMBAD object type: {simbad.get('objectType')}",
            f"- Gaia variability classification: {gaia_var.get('classification')}",
            f"- Known-variable catalog evidence: {offset_source.get('knownVariableCatalogEvidence')}",
            f"- Catalog query errors: {offset_source.get('queryErrors')}",
            f"- Recommended next test: {offset_source.get('recommendedNextTest')}",
            "- Guard: positional catalog association is not treated as direct proof of the residual variability source.",
        ])

    offset_variability = conclusion.get("offsetSourceVariabilityValidation")
    if offset_variability is not None:
        counterpart = offset_variability.get("catalogCounterpart") or {}
        target_control = offset_variability.get("targetControl") or {}
        candidate_evidence = offset_variability.get("catalogCounterpartEvidence") or {}
        distributed = offset_variability.get("distributedValidation") or {}
        lines.extend([
            "",
            "## Offset catalog-counterpart variability validation",
            "",
            f"- Classification: {offset_variability.get('classification')}",
            f"- Residual-mode origin: {offset_variability.get('residualModeOrigin')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Reference residual period: {offset_variability.get('referencePeriodDays')} days",
            f"- Counterpart independent support: {candidate_evidence.get('independentSupportCount')}",
            f"- Counterpart supporting sectors: {candidate_evidence.get('independentSupportingSectors')}",
            f"- Counterpart combined power: {candidate_evidence.get('combinedPower')}",
            f"- Counterpart combined period: {candidate_evidence.get('combinedPeriodDays')} days",
            f"- Target-control independent support: {target_control.get('independentSupportCount')}",
            f"- Target-control combined power: {target_control.get('combinedPower')}",
            f"- Variability confirmed: {offset_variability.get('variabilityConfirmed')}",
            f"- Physical mechanism resolved: {offset_variability.get('physicalMechanismResolved')}",
            f"- Recommended next test: {offset_variability.get('recommendedNextTest')}",
            "- Guard: this is catalog-guided Gaussian deblending, not a calibrated TESS PRF solution.",
        ])

    calibrated_prf = conclusion.get("calibratedPrfSourceDeblending")
    if calibrated_prf is not None:
        counterpart = calibrated_prf.get("catalogCounterpart") or {}
        target_control = calibrated_prf.get("targetControl") or {}
        candidate_evidence = calibrated_prf.get("catalogCounterpartEvidence") or {}
        distributed = calibrated_prf.get("distributedValidation") or {}
        lines.extend([
            "",
            "## Sector-calibrated pixel-response source deblending",
            "",
            f"- Classification: {calibrated_prf.get('classification')}",
            f"- Residual-mode origin: {calibrated_prf.get('residualModeOrigin')}",
            f"- Deblend backend: {calibrated_prf.get('deblendBackend')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Reference residual period: {calibrated_prf.get('referencePeriodDays')} days",
            f"- Counterpart independent support: {candidate_evidence.get('independentSupportCount')}",
            f"- Counterpart supporting sectors: {candidate_evidence.get('independentSupportingSectors')}",
            f"- Counterpart combined power: {candidate_evidence.get('combinedPower')}",
            f"- Counterpart combined period: {candidate_evidence.get('combinedPeriodDays')} days",
            f"- Target-control independent support: {target_control.get('independentSupportCount')}",
            f"- Target-control supporting sectors: {target_control.get('independentSupportingSectors')}",
            f"- Target-control combined power: {target_control.get('combinedPower')}",
            f"- Physical mechanism resolved: {calibrated_prf.get('physicalMechanismResolved')}",
            f"- Recommended next test: {calibrated_prf.get('recommendedNextTest')}",
            "",
            "### Sector ePRF calibration diagnostics",
            "",
        ])
        for item in calibrated_prf.get("calibrationDiagnostics") or []:
            lines.append(
                f"- Sector {item.get('sector')}: backend={item.get('backend')}, "
                f"R2={item.get('explainedVariance')}, sigmaX={item.get('sigmaX')}, "
                f"axisRatio={item.get('axisRatio')}, wingFraction={item.get('wingFraction')}, "
                f"shift=({item.get('dx')},{item.get('dy')}), "
                f"templateCorrelation={item.get('targetCounterpartTemplateCorrelation')}, "
                f"condition={item.get('designConditionNumber')}"
            )
        lines.append(
            "- Guard: this is a sector-calibrated empirical TESS pixel-response model; it is not claimed to be identical to the SPOC engineering PRF calibration."
        )

    difference_image = conclusion.get("differenceImageSourceLocalization")
    if difference_image is not None:
        counterpart = difference_image.get("catalogCounterpart") or {}
        distributed = difference_image.get("distributedFrequencyRefinement") or {}
        lines.extend([
            "",
            "## Difference-image residual source localization",
            "",
            f"- Classification: {difference_image.get('classification')}",
            f"- Residual-mode origin: {difference_image.get('residualModeOrigin')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Reference residual period: {difference_image.get('referencePeriodDays')} days",
            f"- Quality sectors: {difference_image.get('qualitySectorCount')}",
            f"- Counterpart-supporting sectors: {difference_image.get('counterpartSupportingSectors')}",
            f"- Target-supporting sectors: {difference_image.get('targetSupportingSectors')}",
            f"- Ambiguous sectors: {difference_image.get('ambiguousSectors')}",
            f"- Median counterpart separation: {difference_image.get('medianCounterpartSeparationArcsec')} arcsec",
            f"- Median target separation: {difference_image.get('medianTargetSeparationArcsec')} arcsec",
            f"- Recommended next test: {difference_image.get('recommendedNextTest')}",
            "",
            "### Sector difference-image evidence",
            "",
        ])
        for item in difference_image.get("sectorResults") or []:
            image = item.get("differenceImage") or {}
            frequency = item.get("frequencyResult") or {}
            lines.append(
                f"- Sector {item.get('sector')}: classification={item.get('classification')}, "
                f"frequency={frequency.get('candidateFrequency')} c/d, "
                f"period={frequency.get('candidatePeriodDays')} d, "
                f"peakSNR={image.get('peakSNR')}, "
                f"targetDistance={item.get('targetDistancePixels')} px, "
                f"counterpartDistance={item.get('counterpartDistancePixels')} px, "
                f"centroidUncertainty={item.get('centroidUncertaintyPixels')} px"
            )
        lines.append(
            "- Guard: difference-image attribution requires recurring independent-sector localization and does not alter the v20.6 target association of the established main periodic family."
        )

    frequency_localized = conclusion.get("frequencyLocalizedPixelResponse")
    if frequency_localized is not None:
        counterpart = frequency_localized.get("catalogCounterpart") or {}
        distributed = frequency_localized.get("distributedPixelSearch") or {}
        lines.extend([
            "",
            "## Frequency-localized pixel-response confirmation",
            "",
            f"- Classification: {frequency_localized.get('classification')}",
            f"- Residual-mode origin: {frequency_localized.get('residualModeOrigin')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Pixel datasets: {distributed.get('datasetCount')}",
            f"- Reference residual period: {frequency_localized.get('referencePeriodDays')} days",
            f"- Quality sectors: {frequency_localized.get('qualitySectorCount')}",
            f"- Counterpart-supporting sectors: {frequency_localized.get('counterpartSupportingSectors')}",
            f"- Target-supporting sectors: {frequency_localized.get('targetSupportingSectors')}",
            f"- Ambiguous sectors: {frequency_localized.get('ambiguousSectors')}",
            f"- Recommended next test: {frequency_localized.get('recommendedNextTest')}",
            "",
            "### Sector frequency-localized evidence",
            "",
        ])
        for item in frequency_localized.get("sectorResults") or []:
            response = item.get("response") or {}
            lines.append(
                f"- Sector {item.get('sector')}: classification={item.get('classification')}, "
                f"targetFrequency={item.get('targetFrequency')} c/d, "
                f"source={item.get('frequencySource')}, peakPower={response.get('peakPower')}, "
                f"powerContrast={response.get('powerContrast')}, phaseConcentration={response.get('phaseConcentration')}, "
                f"clusterWeightFraction={response.get('clusterWeightFraction')}, "
                f"targetDistance={item.get('targetDistancePixels')} px, "
                f"counterpartDistance={item.get('counterpartDistancePixels')} px, "
                f"centroidUncertainty={item.get('centroidUncertaintyPixels')} px"
            )
        lines.append(
            "- Guard: the narrow-band per-pixel searches are conditioned on the existing residual family and are used only for source localization; they are not independent period discoveries."
        )

    official_spoc_prf = conclusion.get("officialSpocPrfForwardModeling")
    if official_spoc_prf is not None:
        counterpart = official_spoc_prf.get("catalogCounterpart") or {}
        distributed = official_spoc_prf.get("distributedValidation") or {}
        candidate_evidence = official_spoc_prf.get("catalogCounterpartEvidence") or {}
        target_control = official_spoc_prf.get("targetControl") or {}
        lines.extend([
            "",
            "## Official SPOC PRF forward modeling",
            "",
            f"- Classification: {official_spoc_prf.get('classification')}",
            f"- Residual-mode origin: {official_spoc_prf.get('residualModeOrigin')}",
            f"- Backend: {official_spoc_prf.get('deblendBackend')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Counterpart independent support: {candidate_evidence.get('independentSupportCount')}",
            f"- Counterpart supporting sectors: {candidate_evidence.get('independentSupportingSectors')}",
            f"- Counterpart combined power: {candidate_evidence.get('combinedPower')}",
            f"- Counterpart combined period: {candidate_evidence.get('combinedPeriodDays')} days",
            f"- Target-control independent support: {target_control.get('independentSupportCount')}",
            f"- Target-control supporting sectors: {target_control.get('independentSupportingSectors')}",
            f"- Target-control combined power: {target_control.get('combinedPower')}",
            f"- Recommended next test: {official_spoc_prf.get('recommendedNextTest')}",
            "",
            "### Sector official-PRF diagnostics",
            "",
        ])
        for item in official_spoc_prf.get("calibrationDiagnostics") or []:
            lines.append(
                f"- Sector {item.get('sector')}: start_s{item.get('officialPRFStartSector')}, "
                f"camera={item.get('camera')}, ccd={item.get('ccd')}, R2={item.get('explainedVariance')}, "
                f"shift=({item.get('dx')},{item.get('dy')}), "
                f"templateCorrelation={item.get('targetCounterpartTemplateCorrelation')}, "
                f"condition={item.get('designConditionNumber')}"
            )
        lines.append(
            "- Guard: official SPOC PRF source attribution still requires recurring independent-sector residual-frequency evidence; an unresolved result is retained as a TESS spatial-attribution limit."
        )

    catalog_counterpart = conclusion.get("catalogCounterpartIdentification")
    if catalog_counterpart is not None:
        preferred = catalog_counterpart.get("preferredCandidate") or {}
        catalog_ids = preferred.get("catalogIDs") or {}
        ranking = preferred.get("rankingEvidence") or {}
        lines.extend([
            "",
            "## Catalog counterpart identification",
            "",
            f"- Catalog counterpart classification: {catalog_counterpart.get('classification')}",
            f"- Counterpart TIC: {catalog_ids.get('ticID')}",
            f"- Counterpart Gaia DR3: {catalog_ids.get('gaiaDR3SourceID')}",
            f"- Residual-position separation: {ranking.get('residualPositionSeparationArcsec')} arcsec",
            f"- Target-to-counterpart separation: {ranking.get('targetSeparationArcsec')} arcsec",
            f"- Variability confirmed: {catalog_counterpart.get('variabilityConfirmed')}",
            f"- Recommended next test: {catalog_counterpart.get('recommendedNextTest')}",
        ])

    deep_catalog_counterpart = conclusion.get("deepCatalogCounterpartIdentification")
    if deep_catalog_counterpart is not None:
        preferred = deep_catalog_counterpart.get("preferredCandidate") or {}
        catalog_ids = preferred.get("catalogIDs") or {}
        ranking = preferred.get("rankingEvidence") or {}
        lines.extend([
            "",
            "## Deeper catalog counterpart identification",
            "",
            f"- Classification: {deep_catalog_counterpart.get('classification')}",
            f"- SkyMapper DR4 object: {catalog_ids.get('skyMapperDR4ObjectID')}",
            f"- NSC DR2 object: {catalog_ids.get('nscDR2ObjectID')}",
            f"- Residual-position separation: {ranking.get('residualPositionSeparationArcsec')} arcsec",
            f"- Target-to-counterpart separation: {ranking.get('targetSeparationArcsec')} arcsec",
            f"- Plausible deeper-catalog candidates: {len(deep_catalog_counterpart.get('plausibleCatalogCandidates') or [])}",
            f"- Catalog query errors: {deep_catalog_counterpart.get('queryErrors')}",
            f"- Variability confirmed: {deep_catalog_counterpart.get('variabilityConfirmed')}",
            f"- Recommended next test: {deep_catalog_counterpart.get('recommendedNextTest')}",
            f"- Guard: {deep_catalog_counterpart.get('interpretationGuard')}",
        ])

    deep_catalog_prf = conclusion.get("deepCatalogGuidedPrfLocalization")
    if deep_catalog_prf is not None:
        preferred = deep_catalog_prf.get("preferredCandidate") or {}
        catalog_ids = preferred.get("catalogIDs") or {}
        lines.extend([
            "",
            "## Deep-catalog-guided multi-source PRF localization",
            "",
            f"- Classification: {deep_catalog_prf.get('classification')}",
            f"- Method scope: {deep_catalog_prf.get('methodScope')}",
            f"- Decisive sectors: {deep_catalog_prf.get('decisiveSectorCount')} / {deep_catalog_prf.get('sectorCount')}",
            f"- Stable component: {deep_catalog_prf.get('stableComponentID')}",
            f"- SkyMapper DR4 object: {catalog_ids.get('skyMapperDR4ObjectID')}",
            f"- NSC DR2 object: {catalog_ids.get('nscDR2ObjectID')}",
            f"- Source attribution resolved: {deep_catalog_prf.get('sourceAttributionResolved')}",
            f"- Catalog queries repeated: {deep_catalog_prf.get('catalogQueriesRepeated')}",
            f"- Recommended next test: {deep_catalog_prf.get('recommendedNextTest')}",
            f"- Guard: {deep_catalog_prf.get('interpretationGuard')}",
        ])

    external_high_resolution = conclusion.get("externalHighResolutionVariabilityValidation")
    if external_high_resolution is not None:
        distributed = external_high_resolution.get("distributedValidation") or {}
        pair = external_high_resolution.get("sourcePair") or {}
        target = external_high_resolution.get("targetControl") or {}
        counterpart = external_high_resolution.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## External high-resolution variability validation",
            "",
            f"- Classification: {external_high_resolution.get('classification')}",
            f"- Residual-mode origin: {external_high_resolution.get('residualModeOrigin')}",
            f"- Archive: {external_high_resolution.get('archive')}",
            f"- Target Gaia DR3 source: {pair.get('targetGaiaDR3SourceID')}",
            f"- Counterpart Gaia DR3 source: {pair.get('counterpartGaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- TESS drift extrapolated into Gaia epoch: {external_high_resolution.get('driftExtrapolatedToGaiaEpoch')}",
            f"- Target residual-band accepted: {(target or {}).get('acceptedResidualBandVariability')}",
            f"- Target period: {(target or {}).get('candidatePeriodDays')} days",
            f"- Target power: {(target or {}).get('candidatePower')}",
            f"- Counterpart residual-band accepted: {(counterpart or {}).get('acceptedResidualBandVariability')}",
            f"- Counterpart period: {(counterpart or {}).get('candidatePeriodDays')} days",
            f"- Counterpart power: {(counterpart or {}).get('candidatePower')}",
            f"- Recommended next test: {external_high_resolution.get('recommendedNextTest')}",
            f"- Guard: {external_high_resolution.get('interpretationGuard')}",
        ])

    skymapper = conclusion.get("skyMapperResolvedPhotometryScreen")
    if skymapper is not None:
        distributed = skymapper.get("distributedValidation") or {}
        pair = skymapper.get("sourcePair") or {}
        target = skymapper.get("targetControl") or {}
        counterpart = skymapper.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## SkyMapper DR4 resolved-photometry screen",
            "",
            f"- Classification: {skymapper.get('classification')}",
            f"- Residual-mode origin: {skymapper.get('residualModeOrigin')}",
            f"- Archive: {skymapper.get('archive')}",
            f"- Target Gaia DR3 source: {pair.get('targetGaiaDR3SourceID')}",
            f"- Counterpart Gaia DR3 source: {pair.get('counterpartGaiaDR3SourceID')}",
            f"- Pair separation: {skymapper.get('pairSeparationArcsec')} arcsec",
            f"- Seeing limit: {skymapper.get('seeingLimitArcsec')} arcsec",
            f"- Pair separately resolved in SkyMapper master: {skymapper.get('pairSeparatelyResolvedInSkyMapperMaster')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target accepted bands: {target.get('acceptedBands')}",
            f"- Target cross-band supported: {target.get('sourceSupported')}",
            f"- Counterpart accepted bands: {counterpart.get('acceptedBands')}",
            f"- Counterpart cross-band supported: {counterpart.get('sourceSupported')}",
            f"- Recommended next test: {skymapper.get('recommendedNextTest')}",
            f"- Guard: {skymapper.get('interpretationGuard')}",
        ])

    nsc = conclusion.get("nscResolvedPhotometryScreen")
    if nsc is not None:
        distributed = nsc.get("distributedValidation") or {}
        pair = nsc.get("sourcePair") or {}
        target = nsc.get("targetControl") or {}
        counterpart = nsc.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## NOIRLab Source Catalog DR2 resolved-photometry screen",
            "",
            f"- Classification: {nsc.get('classification')}",
            f"- Residual-mode origin: {nsc.get('residualModeOrigin')}",
            f"- Archive: {nsc.get('archive')}",
            f"- Target Gaia DR3 source: {pair.get('targetGaiaDR3SourceID')}",
            f"- Counterpart Gaia DR3 source: {pair.get('counterpartGaiaDR3SourceID')}",
            f"- Pair separation: {nsc.get('pairSeparationArcsec')} arcsec",
            f"- Pair separately resolved in NSC: {nsc.get('pairSeparatelyResolvedInNSC')}",
            f"- Observed NSC object separation: {nsc.get('observedNSCObjectSeparationArcsec')} arcsec",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target accepted bands: {target.get('acceptedBands')}",
            f"- Target cross-band supported: {target.get('sourceSupported')}",
            f"- Counterpart accepted bands: {counterpart.get('acceptedBands')}",
            f"- Counterpart cross-band supported: {counterpart.get('sourceSupported')}",
            f"- Recommended next test: {nsc.get('recommendedNextTest')}",
            f"- Guard: {nsc.get('interpretationGuard')}",
        ])

    noirlab_forced = conclusion.get("noirlabImageForcedPhotometry")
    if noirlab_forced is not None:
        distributed = noirlab_forced.get("distributedValidation") or {}
        pair = noirlab_forced.get("sourcePair") or {}
        target = noirlab_forced.get("targetControl") or {}
        counterpart = noirlab_forced.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## NOIRLab image-level forced two-source photometry",
            "",
            f"- Classification: {noirlab_forced.get('classification')}",
            f"- Residual-mode origin: {noirlab_forced.get('residualModeOrigin')}",
            f"- Archive: {noirlab_forced.get('archive')}",
            f"- Target Gaia DR3 source: {pair.get('targetGaiaDR3SourceID')}",
            f"- Counterpart Gaia DR3 source: {pair.get('counterpartGaiaDR3SourceID')}",
            f"- Gaia target-counterpart separation: {noirlab_forced.get('pairSeparationArcsec')} arcsec",
            f"- Offset-component to catalog-match association separation carried from v20.19: {noirlab_forced.get('catalogAssociationSeparationArcsec')} arcsec",
            f"- SIA rows: {noirlab_forced.get('siaRows')}",
            f"- Candidate single-epoch images: {noirlab_forced.get('candidateExposures')}",
            f"- Successful forced-photometry images: {noirlab_forced.get('successfulForcedPhotometryExposures')}",
            f"- Rejection reasons: {noirlab_forced.get('failureReasons')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target accepted bands: {target.get('acceptedBands')}",
            f"- Target cross-band supported: {target.get('sourceSupported')}",
            f"- Counterpart accepted bands: {counterpart.get('acceptedBands')}",
            f"- Counterpart cross-band supported: {counterpart.get('sourceSupported')}",
            f"- Recommended next test: {noirlab_forced.get('recommendedNextTest')}",
            f"- Guard: {noirlab_forced.get('interpretationGuard')}",
        ])

    des_local = conclusion.get("desDr2SeLocalForcedPhotometry")
    if des_local is not None:
        distributed = des_local.get("distributedValidation") or {}
        target = des_local.get("targetControl") or {}
        counterpart = des_local.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## DES DR2 single-epoch source-local forced photometry",
            "",
            f"- Classification: {des_local.get('classification')}",
            f"- Residual-mode origin: {des_local.get('residualModeOrigin')}",
            f"- Archive: {des_local.get('archive')}",
            f"- Actual Gaia target-counterpart separation: {des_local.get('pairSeparationArcsec')} arcsec",
            f"- SIA rows: {des_local.get('siaRows')}",
            f"- Candidate single-epoch images: {des_local.get('candidateExposures')}",
            f"- Source attempts: {des_local.get('sourceAttempts')}",
            f"- Source successes: {des_local.get('sourceSuccesses')}",
            f"- Failure reasons: {des_local.get('failureReasons')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target accepted bands: {target.get('acceptedBands')}",
            f"- Target cross-band supported: {target.get('sourceSupported')}",
            f"- Counterpart accepted bands: {counterpart.get('acceptedBands')}",
            f"- Counterpart cross-band supported: {counterpart.get('sourceSupported')}",
            f"- Recommended next test: {des_local.get('recommendedNextTest')}",
            f"- Guard: {des_local.get('interpretationGuard')}",
        ])

    atlas = conclusion.get("atlasForcedPhotometry")
    if atlas is not None:
        distributed = atlas.get("distributedValidation") or {}
        target = atlas.get("targetControl") or {}
        counterpart = atlas.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## ATLAS source-resolved forced photometry",
            "",
            f"- Classification: {atlas.get('classification')}",
            f"- Residual-mode origin: {atlas.get('residualModeOrigin')}",
            f"- Archive: {atlas.get('archive')}",
            f"- Corrected Gaia target-counterpart separation: {atlas.get('gaiaPairSeparationArcsec')} arcsec",
            f"- Target-image forced photometry: {atlas.get('useReducedTargetImages')}",
            f"- Difference imaging used: {atlas.get('differenceImagingUsed')}",
            f"- Minimum MJD: {atlas.get('mjdMinimum')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target accepted bands: {target.get('acceptedBands')}",
            f"- Target cross-band supported: {target.get('sourceSupported')}",
            f"- Counterpart accepted bands: {counterpart.get('acceptedBands')}",
            f"- Counterpart cross-band supported: {counterpart.get('sourceSupported')}",
            f"- Recommended next test: {atlas.get('recommendedNextTest')}",
            f"- Guard: {atlas.get('interpretationGuard')}",
        ])

    atlas_reanalysis = conclusion.get("atlasForcedPhotometryReanalysis")
    if atlas_reanalysis is not None:
        distributed = atlas_reanalysis.get("distributedValidation") or {}
        target = atlas_reanalysis.get("targetControl") or {}
        counterpart = atlas_reanalysis.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## ATLAS signed forced-photometry reanalysis",
            "",
            f"- Classification: {atlas_reanalysis.get('classification')}",
            f"- Residual-mode origin: {atlas_reanalysis.get('residualModeOrigin')}",
            f"- Archive: {atlas_reanalysis.get('archive')}",
            f"- Raw v20.24 artifacts reused: {atlas_reanalysis.get('rawArtifactsReusedFromV20_24')}",
            f"- Individual detection threshold applied: {atlas_reanalysis.get('individualDetectionThresholdApplied')}",
            f"- Signed forced flux retained: {atlas_reanalysis.get('signedForcedFluxRetained')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target accepted bands: {target.get('acceptedBands')}",
            f"- Target cross-band supported: {target.get('sourceSupported')}",
            f"- Counterpart accepted bands: {counterpart.get('acceptedBands')}",
            f"- Counterpart cross-band supported: {counterpart.get('sourceSupported')}",
            f"- Recommended next test: {atlas_reanalysis.get('recommendedNextTest')}",
            f"- Guard: {atlas_reanalysis.get('interpretationGuard')}",
        ])

    atlas_time_resolved = conclusion.get("atlasTimeResolved")
    if atlas_time_resolved is not None:
        distributed = atlas_time_resolved.get("distributedValidation") or {}
        counterpart = atlas_time_resolved.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## ATLAS time-resolved counterpart recurrence",
            "",
            f"- Classification: {atlas_time_resolved.get('classification')}",
            f"- Residual-mode origin: {atlas_time_resolved.get('residualModeOrigin')}",
            f"- Counterpart Gaia DR3: {atlas_time_resolved.get('counterpartGaiaDR3SourceID')}",
            f"- Season gap: {atlas_time_resolved.get('seasonGapDays')} days",
            f"- Seasons: {len(atlas_time_resolved.get('seasons') or [])}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Accepted season-band results: {counterpart.get('acceptedSeasonBandCount')}",
            f"- Accepted seasons: {counterpart.get('acceptedSeasons')}",
            f"- Accepted bands: {counterpart.get('acceptedBands')}",
            f"- Cross-band-consistent seasons: {counterpart.get('crossBandConsistentSeasons')}",
            f"- Independent ATLAS frequency trend: {counterpart.get('independentFrequencyTrend')}",
            f"- Counterpart supported: {counterpart.get('sourceSupported')}",
            f"- Counterpart suggestive: {counterpart.get('sourceSuggestive')}",
            f"- Recommended next test: {atlas_time_resolved.get('recommendedNextTest')}",
            f"- Guard: {atlas_time_resolved.get('interpretationGuard')}",
        ])

    atlas_fixed = conclusion.get("atlasFixedWindowRecurrence")
    if atlas_fixed is not None:
        distributed = atlas_fixed.get("distributedValidation") or {}
        counterpart = atlas_fixed.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## ATLAS fixed-window counterpart recurrence",
            "",
            f"- Classification: {atlas_fixed.get('classification')}",
            f"- Residual-mode origin: {atlas_fixed.get('residualModeOrigin')}",
            f"- Counterpart Gaia DR3: {atlas_fixed.get('counterpartGaiaDR3SourceID')}",
            f"- Fixed window size: {atlas_fixed.get('windowDays')} days",
            f"- Window anchor: {atlas_fixed.get('windowAnchor')}",
            f"- Windows intersecting data: {len(atlas_fixed.get('windows') or [])}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Accepted window-band results: {counterpart.get('acceptedWindowBandCount')}",
            f"- Accepted windows: {counterpart.get('acceptedWindows')}",
            f"- Accepted bands: {counterpart.get('acceptedBands')}",
            f"- Cross-band-consistent windows: {counterpart.get('crossBandConsistentWindows')}",
            f"- Independent ATLAS frequency trend: {counterpart.get('independentATLASFrequencyTrend')}",
            f"- Counterpart supported: {counterpart.get('sourceSupported')}",
            f"- Counterpart suggestive: {counterpart.get('sourceSuggestive')}",
            f"- Recommended next test: {atlas_fixed.get('recommendedNextTest')}",
            f"- Guard: {atlas_fixed.get('interpretationGuard')}",
        ])

    observation_plan = conclusion.get("targetedObservationPlan")
    if observation_plan is not None:
        geometry = observation_plan.get("sourceGeometry") or {}
        cadence = observation_plan.get("cadence") or {}
        exposure = observation_plan.get("exposureStrategy") or {}
        filters = observation_plan.get("filterStrategy") or {}
        artifacts = observation_plan.get("artifacts") or {}
        lines.extend([
            "",
            "## Targeted high-resolution time-series observation plan",
            "",
            f"- Status: {observation_plan.get('status')}",
            f"- Scientific objective: {observation_plan.get('scientificObjective')}",
            f"- Gaia target-counterpart separation: {geometry.get('separationArcsec')} arcsec",
            f"- Preferred FWHM: {geometry.get('preferredFwhmArcsec')} arcsec",
            f"- Maximum FWHM: {geometry.get('maximumFwhmArcsec')} arcsec",
            f"- Preferred pixel scale: {geometry.get('preferredPixelScaleArcsec')} arcsec/pixel",
            f"- Maximum pixel scale: {geometry.get('maximumPixelScaleArcsec')} arcsec/pixel",
            f"- Frozen frequency range: {cadence.get('frozenFrequencyRangeCyclesPerDay')}",
            f"- Frozen period range: {cadence.get('frozenPeriodRangeDays')}",
            f"- Minimum baseline: {cadence.get('minimumBaselineDays')} days",
            f"- Preferred baseline: {cadence.get('preferredBaselineDays')} days",
            f"- Minimum distinct nights: {cadence.get('minimumDistinctNights')}",
            f"- Preferred distinct nights: {cadence.get('preferredDistinctNights')}",
            f"- Minimum visits per night: {cadence.get('minimumVisitsPerObservedNight')}",
            f"- Time-resolved analysis: {cadence.get('timeResolvedAnalysis')}",
            f"- Filters: {filters}",
            f"- Exposure strategy: {exposure}",
            f"- JSON plan: {artifacts.get('jsonPlanPath')}",
            f"- Markdown plan: {artifacts.get('markdownPlanPath')}",
            f"- CSV ingest template: {artifacts.get('csvIngestTemplatePath')}",
            f"- Recommended next test: {observation_plan.get('recommendedNextTest')}",
        ])

    lines.extend([
        "",
        "## Claim policy",
        "",
        "This report is produced by deterministic OpenStar rules. It never automatically emits `DISCOVERY`.",
        "",
    ])
    archival = conclusion.get("targetResidualArchivalBaselineExtension")
    if archival is not None:
        envelope = archival.get("historicalFrequencyEnvelope") or {}
        lines.extend(["", "## Archival target-residual baseline extension", "",
            f"- Classification: {archival.get('classification')}",
            f"- Recommended next test: {archival.get('recommendedNextTest')}",
            f"- Frozen residual reference: {archival.get('frozenResidualReferencePeriodDays')} days / {archival.get('frozenResidualReferenceFrequency')} cycles/day",
            f"- Historical frequency envelope: {envelope.get('minimum')}–{envelope.get('maximum')} cycles/day",
            f"- Eligible/supporting/resolution-limited/non-supporting sectors: {archival.get('eligibleSectorCount')}/{archival.get('supportingSectorCount')}/{archival.get('resolutionLimitedSectorCount')}/{archival.get('nonSupportingSectorCount')}",
            f"- Supporting temporal span: {archival.get('supportingTemporalSpanDays')} days",
            f"- Future pixel-followup sectors: {archival.get('selectedFuturePixelFollowupSectors')}",
            f"- Archive/materialization limitations: {archival.get('archiveMaterializationLimitations') or []}",
            "", "### Archival sector recurrence", ""])
        for item in archival.get("sectorEvidence") or []:
            lines.append(f"- Sector {item.get('sector')}: author={item.get('author')}, cadence={item.get('cadenceSeconds')} s, baseline={item.get('baselineDays')} d, candidatePeriod={item.get('candidatePeriodDays')} d, candidateFrequency={item.get('candidateFrequency')}, CI={item.get('candidateFrequencyConfidenceInterval')}, classification={item.get('recurrenceClassification')}, supports={item.get('supportsHistoricalResidualFamily')}")
    source_review = conclusion.get("sourceAttributionReview")
    depth_audit = conclusion.get("eventDepthAttenuationAudit")
    joint_model = conclusion.get("jointEventPhaseModel")
    external_companion = conclusion.get("externalCompanionEvidence")
    if source_review is not None:
        lines.extend([
            "", "## Eclipse source-attribution review", "",
            f"- Classification: {source_review.get('classification')}",
            f"- Recomputed independent support: {source_review.get('supportingIndependentSectorCount')}",
            f"- Supporting independent sectors: {source_review.get('supportingIndependentSectors')}",
        ])
    if depth_audit is not None:
        lines.extend([
            "", "## Software-blind event-depth attenuation audit", "",
            f"- Status: {depth_audit.get('status')}",
            "- Detection-only standardized box depth remains nonphysical.",
            "- Diagnostic transformations: full-precision local baseline, downsampling, Float32 standardization, protected harmonic subtraction, and discrete duration selection.",
            f"- Cross-sector attenuation summary: {depth_audit.get('crossSectorRobustSummary')}",
            f"- Suitable for later precision modeling: {depth_audit.get('suitableForLaterPrecisionModeling')}",
            f"- Recommendation: {depth_audit.get('recommendedNextTest')}",
            f"- Unresolved reasons: {depth_audit.get('unresolvedReasons') or []}",
            f"- External catalog information used: {depth_audit.get('externalCatalogInformationUsed')}",
            f"- Catalog answer key used: {depth_audit.get('catalogAnswerKeyUsed')}",
            "- No companion radius or precision physical transit solution is claimed.",
        ])
        for sector in depth_audit.get("sectorResults") or []:
            lines.append(f"- Sector {sector.get('sector')}: samples={sector.get('sampleCounts')}, events={sector.get('eventResults')}")
    if joint_model is not None:
        fitted = joint_model.get("globalFit") or {}
        lines.extend([
            "", "## Software-blind joint transit, eclipse, and orbital phase-curve model", "",
            f"- Status/classification: {joint_model.get('status')} / {joint_model.get('classification')}",
            f"- Mid-transit empirical deficit: {fitted.get('midTransitFractionalFluxDeficit')} ± {fitted.get('conservativeTransitDepthUncertainty')}",
            f"- Equivalent-box transit depth: {fitted.get('equivalentBoxTransitDepthFractionalFlux')}",
            f"- Opposite-conjunction eclipse: {fitted.get('oppositeConjunctionEclipseDepthFractionalFlux')} ({fitted.get('oppositeConjunctionEclipseStatus')})",
            f"- Fundamental / second-harmonic phase status: {fitted.get('fundamentalPhaseCurveStatus')} / {fitted.get('secondHarmonicPhaseCurveStatus')}",
            f"- Independent supporting sectors: {joint_model.get('independentSupportingSectorCount')}",
            f"- Leave-one-sector-out stable: {(joint_model.get('resolutionGates') or {}).get('leaveOneSectorOutStable')}",
            f"- Unresolved reasons: {joint_model.get('unresolvedReasons') or []}",
            f"- Model SHA-256: {joint_model.get('modelSHA256')}",
            "- This empirical model claims neither a companion radius nor a complete physical transit solution.",
            "- Fundamental phase terms are not uniquely interpreted as reflection or thermal emission.",
        ])
    if external_companion is not None:
        synthesis_complete = conclusion.get("finalCompanionEvidenceSynthesis") is not None
        lines.extend([
            "", "## Published external companion confirmation", "",
            f"- Classification: {external_companion.get('classification')}",
            f"- Matched external period: {external_companion.get('externalOrbitalPeriodDays')} days",
            f"- Period difference: {external_companion.get('externalOrbitalPeriodDifferenceDays')} days",
            f"- Published mass: {external_companion.get('externalMassJupiter')} Jupiter masses",
            f"- Published mass uncertainty interval: {external_companion.get('externalMassIntervalJupiter')}",
            f"- Supported mass regime: {external_companion.get('supportedCompanionMassRegime')}",
            f"- Known-object catalog used: {external_companion.get('externalKnownObjectCatalogUsed')}",
            f"- Preceding photometric/spatial evidence remained software-blind: {external_companion.get('softwareBlindPhotometricEvidencePreserved')}",
            f"- At external interpretation, physical mechanism resolved: {external_companion.get('physicalMechanismResolved')}",
            ("- At external interpretation, companion nature was pending final synthesis."
             if synthesis_complete else
             f"- At external interpretation, companion nature resolved: {external_companion.get('companionNatureResolved')}"),
            f"- Authoritative next test: {conclusion.get('recommendedNextTest')}",
        ])
    synthesis = conclusion.get("finalCompanionEvidenceSynthesis")
    if synthesis is not None:
        relationship = ("the investigated target" if synthesis.get("sourceRelationship") == "TARGET_ASSOCIATED"
                        else "an off-target source (not the investigated target)")
        lines.extend([
            "", "## Final companion-evidence synthesis", "",
            f"- Classification: {synthesis.get('classification')}",
            f"- Source attribution: {relationship}",
            "- Evidence separation: software-blind photometric and spatial evidence was frozen before published known-object confirmation evidence was consulted.",
            f"- Resolved companion mass regime: {synthesis.get('supportedCompanionMassRegime')}",
            "- Detailed photometric mechanism: **unresolved** (reflection, thermal emission, ellipsoidal variation, beaming, eclipses, and other components are not uniquely decomposed).",
            f"- Autonomous companion-evidence analysis complete: {synthesis.get('autonomousCompanionEvidenceComplete')}",
            "- Required next step: human scientific review.",
            f"- Automatic discovery claim: {synthesis.get('automaticDiscoveryClaim')}",
            f"- Catalog answer key used: {synthesis.get('catalogAnswerKeyUsed')}",
            "- Interpretation: OpenStar independently recovered evidence consistent with a previously known companion; it did not discover a new object.",
        ])
    return "\n".join(lines)


def build_engine(
    store: InvestigationStore,
    coordinator: OpenStarCoordinatorClient,
    *,
    poll_interval: float,
    timeout: float | None,
    historical_path_resolver: HistoricalPathResolver | None = None,
    astrophysical_evidence_provider=None,
) -> WorkflowEngine:
    engine = WorkflowEngine(store)
    astrophysical_evidence_provider = (astrophysical_evidence_provider or
        FrozenCatalogAstrophysicalEvidenceProvider())

    def prepare_target(investigation, request):
        source_project = Path(request.parameters["projectPath"]).expanduser().resolve()
        if not source_project.exists():
            raise FileNotFoundError(source_project)

        artifact_root = store.directory_for(investigation.id) / "artifacts"
        prepared = build_single_target_primary(
            source_project_path=source_project,
            output_dir=artifact_root,
            investigation_id=investigation.id,
            dataset_id=request.parameters.get("datasetID"),
            tic_id=request.parameters.get("ticID"),
        )
        source_dataset = Path(prepared["datasetPath"])
        primary_manifest = Path(prepared["projectPath"])
        prepared["observationBaselineDays"] = _dataset_baseline_days(source_dataset)

        print("🔒 TESS target frozen for investigation")
        print(f"   target: {prepared.get('targetName')}")
        print(f"   dataset: {prepared['datasetID']}")
        print(f"   TIC: {prepared['ticID']}")
        print(f"   sector: {prepared.get('sector')}")
        print(f"   baseline: {prepared.get('observationBaselineDays')} days")
        print(f"   primary project: {prepared['projectID']}")

        return StageOutcome(
            result=prepared,
            next_stage=StageRequest(
                id="002-primary-distributed-search",
                handler_id="openstar.tess.primary-project.run",
                parameters={"projectPath": prepared["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "sourceProjectManifest": sha256_file(source_project),
                "sourceDataset": sha256_file(source_dataset),
            },
            artifacts=(_artifact(primary_manifest, "application/json"),),
        )

    def run_primary(investigation, request):
        return run_primary_with_reuse(investigation, request, coordinator,
                                      poll_interval=poll_interval, timeout=timeout)

    def identity_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        primary = _target_result(_result(investigation, "002-primary-distributed-search"))
        tic_id = int(prepared["ticID"])

        print("🌐 Resolving TIC / SIMBAD / VSX / Gaia / TESS identity")
        identity = collect_identity(tic_id)
        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "identity"
            / f"{request.id}.json"
        )
        _write_json(artifact_path, identity)
        print(f"   identity resolved: {identity.get('identityResolved')}")
        print(f"   catalog query errors: {len(identity.get('queryErrors') or [])}")
        tess = identity.get("tess") or {}
        print(f"   official TESS sectors: {tess.get('officialSectors') or []}")
        transient_failures = transient_required_catalog_failures(identity)
        if transient_failures:
            raise RetryableExecutionError(
                "Required catalog infrastructure failed transiently: "
                + ", ".join(transient_failures),
                result=identity,
                input_hashes={"primaryTargetResult": sha256_json(primary)},
                artifacts=(_artifact(artifact_path, "application/json"),),
            )

        return StageOutcome(
            result=identity,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "hypotheses"),
                handler_id="openstar.tess.hypotheses",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"primaryTargetResult": sha256_json(primary)},
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def hypothesis_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        primary = _target_result(_result(investigation, "002-primary-distributed-search"))
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        if identity is None:
            raise RuntimeError("Hypotheses require completed catalog identity.")
        primary_search = (
            _load_json(Path(prepared["datasetPath"])).get("frequencySearch") or {}
        )
        analysis = analyze(
            primary,
            identity,
            observation_baseline_days=prepared.get("observationBaselineDays"),
            primary_minimum_frequency=primary_search.get("minimumFrequency"),
            primary_maximum_frequency=primary_search.get("maximumFrequency"),
            primary_frequency_step=primary_search.get("frequencyStep"),
        )
        print("🧠 Deterministic TESS hypotheses evaluated")
        print(f"   reliable primary: {analysis.get('primaryReliable')}")
        print(f"   catalog period match: {bool(analysis.get('bestCatalogMatch'))}")
        print(f"   rotation sanity: {(analysis.get('rotationSanity') or {}).get('status')}")
        coverage = analysis.get("preferredCycleCoverage") or {}
        print(f"   preferred-cycle coverage: {coverage.get('observedCycles')}")
        return StageOutcome(
            result=analysis,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "planner"),
                handler_id="openstar.tess.planner",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "primaryTargetResult": sha256_json(primary),
                "identity": sha256_json(identity),
            },
        )

    def planner_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        analysis = _latest_result_for_handler(investigation, "openstar.tess.hypotheses")
        if identity is None or analysis is None:
            raise RuntimeError("Planner requires completed identity and hypotheses.")
        planned = plan(analysis, identity, prepared.get("investigationGoal"))
        print("🧭 Deterministic planner")
        print(f"   action: {planned['action']}")
        print(f"   reason: {planned['reason']}")

        if planned["action"] == "LOW_FREQUENCY_FOLLOWUP":
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "prepare-followup"),
                handler_id="openstar.tess.followup.prepare-low-frequency",
                parameters={},
                triggered_by_stage_id=request.id,
            )
        elif planned["action"] == "INDEPENDENT_SECTOR_FOLLOWUP":
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "prepare-independent-sectors"),
                handler_id="openstar.tess.independent.prepare",
                parameters={},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=planned,
            next_stage=next_stage,
            input_hashes={"hypothesisAnalysis": sha256_json(analysis)},
        )

    def prepare_followup(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        analysis = _required_latest_result_for_handler(
            investigation, "openstar.tess.hypotheses"
        )
        planner = _required_latest_result_for_handler(
            investigation, "openstar.tess.planner"
        )
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        follow = build_low_frequency_followup(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_path=prepared["datasetPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            output_dir=artifact_root,
            investigation_id=investigation.id,
            trigger_reason=planner.get("reason"),
            primary_period_days=analysis.get("observedPeriodDays"),
        )
        if not follow.get("executable", True):
            print("🔬 Same-sector low-frequency follow-up is not executable")
            print(f"   reason: {follow.get('reason')}")
            return StageOutcome(
                result=follow,
                next_stage=StageRequest(
                    id=_next_stage_id(request.id, "prepare-independent-sectors"),
                    handler_id="openstar.tess.independent.prepare",
                    parameters={},
                    triggered_by_stage_id=request.id,
                ),
                input_hashes={"sourceDataset": sha256_file(prepared["datasetPath"])},
            )
        dataset_path = Path(follow["datasetPath"])
        manifest_path = Path(follow["projectPath"])
        print("🔬 Decisive same-sector frequency follow-up prepared")
        print(f"   mode: {follow.get('followupMode')}")
        print(f"   target period: {follow.get('targetPeriodDays')} days")
        print(
            "   frequency range: "
            f"{follow['frequencySearch']['minimumFrequency']:.6f} - "
            f"{follow['frequencySearch']['maximumFrequency']:.6f} cycles/day"
        )
        print(f"   source baseline: {follow.get('sourceBaselineDays')} days")
        print(
            "   work units: "
            f"{(follow['frequencySearch']['totalFrequencies'] + follow['frequencySearch']['frequenciesPerWorkUnit'] - 1) // follow['frequencySearch']['frequenciesPerWorkUnit']}"
        )
        return StageOutcome(
            result=follow,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-followup"),
                handler_id="openstar.tess.followup.run",
                parameters={"projectPath": follow["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"sourceDataset": sha256_file(prepared["datasetPath"])},
            artifacts=(
                _artifact(dataset_path, "application/json"),
                _artifact(manifest_path, "application/json"),
            ),
        )

    def run_followup(investigation, request):
        print("⚙️ Activating distributed same-sector follow-up")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        target = _target_result(run.status)
        print("✅ Same-sector follow-up complete")
        print(f"   period status: {target.get('periodStatus')}")
        print(f"   candidate period: {target.get('candidatePeriodDays')} days")
        print(f"   reducer preferred period: {target.get('preferredPhysicalPeriodDays')} days")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-followup"),
                handler_id="openstar.tess.followup.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def interpret_stage(investigation, request):
        analysis = _required_latest_result_for_handler(
            investigation, "openstar.tess.hypotheses"
        )
        identity = _required_latest_result_for_handler(
            investigation, "openstar.tess.catalog-identity"
        )
        followup_spec = _required_latest_result_for_handler(
            investigation, "openstar.tess.followup.prepare-low-frequency"
        )
        followup = _required_latest_result_for_handler(
            investigation, "openstar.tess.followup.run"
        )
        interpreted = interpret_followup(
            analysis,
            followup,
            followup_spec=followup_spec,
            identity=identity,
        )
        print("🔎 Same-sector follow-up interpretation")
        print(f"   claim: {interpreted['claimDecision']['claim']}")
        print(f"   selected period: {interpreted.get('selectedPeriodDays')} days")
        coverage = (interpreted.get("diagnostics") or {}).get("cycleCoverage") or {}
        print(f"   observed cycles: {coverage.get('observedCycles')}")

        if (
            interpreted["claimDecision"]["claim"] == "CANDIDATE_PERIOD"
            and interpreted.get("selectedPeriodDays") is not None
        ):
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "prepare-independent-sectors"),
                handler_id="openstar.tess.independent.prepare",
                parameters={},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=interpreted,
            next_stage=next_stage,
            input_hashes={"followupResult": sha256_json(followup)},
        )

    def prepare_independent(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        identity = _required_latest_result_for_handler(
            investigation, "openstar.tess.catalog-identity"
        )
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        planner = _required_latest_result_for_handler(
            investigation, "openstar.tess.planner"
        )
        analysis = _required_latest_result_for_handler(
            investigation, "openstar.tess.hypotheses"
        )

        if followup_interpretation is not None:
            target_period = followup_interpretation.get("selectedPeriodDays")
        else:
            target_period = analysis.get("observedPeriodDays")

        if target_period is None:
            raise RuntimeError("Independent-sector follow-up has no candidate period to test.")

        official_sectors = ((identity.get("tess") or {}).get("officialSectors") or [])
        artifact_root = store.directory_for(investigation.id) / "artifacts"

        print("🛰 Preparing independent TESS-sector verification")
        print(f"   primary sector: {prepared.get('sector')}")
        print(f"   target period: {target_period} days")
        print(f"   catalog official sectors: {official_sectors}")

        try:
            spec = build_independent_sector_project(
                source_project_path=prepared["sourceProjectPath"],
                source_dataset_entry=prepared["sourceDatasetEntry"],
                tic_id=int(prepared["ticID"]),
                primary_sector=prepared.get("sector"),
                target_period_days=float(target_period),
                candidate_sectors=list(official_sectors),
                output_dir=artifact_root,
                investigation_id=investigation.id,
            )
        except TessArchiveInfrastructureError as error:
            raise RetryableExecutionError(
                str(error), result=error.diagnostics,
                input_hashes={"identity": sha256_json(identity),
                              "planner": sha256_json(planner)},
            ) from error

        spec = dict(spec)
        investigation_goal = (
            prepared.get("investigationGoal")
            or request.parameters.get("investigationGoal")
        )
        if investigation_goal is not None:
            spec["investigationGoal"] = investigation_goal
        print(f"   prepared independent sectors: {[item.get('sector') for item in spec.get('preparedSectors') or []]}")
        if spec.get("errors"):
            print(f"   sector preparation errors: {len(spec['errors'])}")

        artifacts: list[ArtifactReference] = []
        for item in spec.get("preparedSectors") or []:
            artifacts.append(_artifact(Path(item["datasetPath"]), "application/json"))
        if spec.get("projectPath"):
            artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))

        if spec.get("available"):
            print(f"   independent work units: {spec.get('totalWorkUnits')}")
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "run-independent-sectors"),
                handler_id="openstar.tess.independent.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            )
        else:
            print("   no independent TESS sectors could be prepared")
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=spec,
            next_stage=next_stage,
            input_hashes={
                "identity": sha256_json(identity),
                "planner": sha256_json(planner),
            },
            artifacts=tuple(artifacts),
        )

    def run_independent(investigation, request):
        print("⚙️ Activating independent TESS-sector verification")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Independent-sector distributed search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-independent-sectors"),
                handler_id="openstar.tess.independent.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def interpret_independent(investigation, request):
        spec = _latest_result_for_handler(investigation, "openstar.tess.independent.prepare")
        run = _latest_result_for_handler(investigation, "openstar.tess.independent.run")
        if spec is None or run is None:
            raise RuntimeError("Independent-sector interpretation is missing its prepare/run stages.")

        target_period = float(spec["targetPeriodDays"])
        interpreted = interpret_independent_sectors(
            target_period_days=target_period,
            project_status=run,
            independent_spec=spec,
        )
        goal = spec.get("investigationGoal")
        if goal is not None:
            interpreted["investigationGoal"] = goal
        print("🔭 Independent-sector recurrence interpretation")
        print(f"   eligible sectors: {interpreted.get('eligibleSectorCount')}")
        print(f"   supporting sectors: {interpreted.get('supportingSectorCount')}")
        print(f"   required support: {interpreted.get('requiredSupportingSectorCount')}")
        print(f"   claim: {interpreted['claimDecision']['claim']}")
        print(f"   selected period: {interpreted.get('selectedPeriodDays')} days")

        contradiction_plan = plan_independent_contradiction_resolution(
            interpreted
        )
        primary_analysis = _required_latest_result_for_handler(
            investigation, "openstar.tess.hypotheses"
        )
        print("🧭 Independent contradiction planner")
        print(f"   action: {contradiction_plan['action']}")
        print(f"   reason: {contradiction_plan['reason']}")
        print(f"   reliable sectors: {contradiction_plan.get('reliableSectorCount')}")
        print(f"   boundary hits: {contradiction_plan.get('boundaryHitCount')}")

        interpreted = dict(interpreted)
        interpreted["primaryBoundaryHit"] = (
            primary_analysis.get("primaryBoundaryHit") is True
        )
        interpreted["primaryReliable"] = primary_analysis.get("primaryReliable")
        interpreted["contradictionPlan"] = contradiction_plan
        full_characterization_confirmed = (
            goal == "FULL_CHARACTERIZATION"
            and contradiction_plan["reason"] == "targeted-independent-recurrence-confirmed"
        )
        boundary_failure_transit_fallback = blind_transit_search_continuation(
            None, spec, None, interpreted
        )
        if boundary_failure_transit_fallback:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "blind-transit-period-search"),
                handler_id=BLIND_TRANSIT_SEARCH_HANDLER_ID,
                parameters={},
                triggered_by_stage_id=request.id,
            )
        elif contradiction_plan["action"] == "BROAD_INDEPENDENT_SEARCH" or full_characterization_confirmed:
            primary_family = _primary_harmonic_morphology_family(primary_analysis)
            morphology_already_completed = _latest_result_for_handler(
                investigation, "openstar.tess.morphology.analyze"
            ) is not None
            if primary_family is not None and not morphology_already_completed:
                next_stage = StageRequest(
                    id=_next_stage_id(request.id, "morphology"),
                    handler_id="openstar.tess.morphology.analyze",
                    parameters={
                        "evidenceSource": (
                            "full-characterization-independent-confirmation"
                            if full_characterization_confirmed
                            else "primary-harmonic-contradiction"
                        ),
                        "unresolvedFallback": "BROAD_INDEPENDENT_SEARCH",
                    },
                    triggered_by_stage_id=request.id,
                )
            else:
                next_stage = StageRequest(
                    id=_next_stage_id(request.id, "prepare-broad-independent-search"),
                    handler_id="openstar.tess.independent.broad.prepare",
                    parameters={"continuation": False},
                    triggered_by_stage_id=request.id,
                )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=interpreted,
            next_stage=next_stage,
            input_hashes={"independentProjectResult": sha256_json(run)},
        )

    def prepare_broad_independent(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        targeted_spec = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        if targeted_spec is None:
            raise RuntimeError(
                "Broad independent search requires the frozen independent-sector preparation."
            )

        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🌐 Preparing target-independent broad independent-sector search")
        print("   reusing frozen sectors; no MAST download")
        spec = build_broad_independent_sector_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            independent_spec=targeted_spec,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        spec["continuation"] = bool(request.parameters.get("continuation"))
        search = spec.get("frequencySearch") or {}
        print(
            "   broad frequency range: "
            f"{search.get('minimumFrequency'):.6f} - "
            f"{search.get('maximumFrequency'):.6f} cycles/day"
        )
        print(
            "   reused sectors: "
            f"{[item.get('sector') for item in spec.get('preparedSectors') or []]}"
        )
        print(f"   broad work units: {spec.get('totalWorkUnits')}")

        artifacts: list[ArtifactReference] = []
        for item in spec.get("preparedSectors") or []:
            artifacts.append(_artifact(Path(item["datasetPath"]), "application/json"))
        if spec.get("projectPath"):
            artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))

        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-broad-independent-search"),
                handler_id="openstar.tess.independent.broad.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"targetedIndependentSpec": sha256_json(targeted_spec)},
            artifacts=tuple(artifacts),
        )

    def run_broad_independent(investigation, request):
        print("⚙️ Activating target-independent broad sector search")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Broad independent-sector search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-broad-independent-search"),
                handler_id="openstar.tess.independent.broad.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def interpret_broad_independent(investigation, request):
        spec = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.run",
        )
        if spec is None or run is None:
            raise RuntimeError(
                "Broad independent interpretation is missing its prepare/run stages."
            )

        primary_analysis = _required_latest_result_for_handler(
            investigation, "openstar.tess.hypotheses"
        )
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        interpreted = interpret_broad_independent_sectors(
            project_status=run,
            broad_spec=spec,
            primary_raw_period_days=primary_analysis.get("rawCandidatePeriodDays"),
            primary_preferred_period_days=primary_analysis.get("observedPeriodDays"),
            same_sector_candidate_days=(
                (followup_interpretation or {}).get("selectedPeriodDays")
            ),
        )
        cluster = interpreted.get("bestCluster") or {}
        print("🧩 Independent-sector period clustering")
        print(f"   eligible sectors: {interpreted.get('eligibleSectorCount')}")
        print(f"   best cluster sectors: {cluster.get('sectors') or []}")
        print(f"   cluster median period: {cluster.get('medianPeriodDays')} days")
        print(f"   required support: {interpreted.get('requiredClusterSupportCount')}")
        print(f"   promotion eligible: {interpreted.get('promotionEligible')}")
        blockers = interpreted.get("promotionBlockers") or []
        if blockers:
            print(f"   promotion blockers: {blockers}")
        harmonic_family = interpreted.get("harmonicFamily") or {}
        if harmonic_family:
            print(
                "   recurrent raw family: "
                f"{harmonic_family.get('representativeRawPeriodDays')} days"
            )
            print(
                "   possible 2x cycle: "
                f"{harmonic_family.get('possibleDoubleCycleDays')} days"
            )
        print(f"   claim: {interpreted['claimDecision']['claim']}")

        finalize_parameters = {}
        if request.parameters.get("outputSuffix"):
            finalize_parameters["outputSuffix"] = request.parameters["outputSuffix"]
        elif spec.get("continuation"):
            finalize_parameters["outputSuffix"] = "v20.3.1"

        return StageOutcome(
            result=interpreted,
            next_stage=broad_independent_continuation(
                interpreted,
                request_id=request.id,
                finalize_parameters=finalize_parameters,
            ),
            input_hashes={"broadIndependentProjectResult": sha256_json(run)},
        )

    def reinterpret_harmonic_family(investigation, request):
        spec = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.run",
        )
        if spec is None or run is None:
            raise RuntimeError(
                "Harmonic-family reinterpretation requires completed broad independent prepare/run stages."
            )

        primary_analysis = _required_latest_result_for_handler(
            investigation, "openstar.tess.hypotheses"
        )
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        interpreted = interpret_broad_independent_sectors(
            project_status=run,
            broad_spec=spec,
            primary_raw_period_days=primary_analysis.get("rawCandidatePeriodDays"),
            primary_preferred_period_days=primary_analysis.get("observedPeriodDays"),
            same_sector_candidate_days=(
                (followup_interpretation or {}).get("selectedPeriodDays")
            ),
        )

        cluster = interpreted.get("bestCluster") or {}
        harmonic_family = interpreted.get("harmonicFamily") or {}
        print("🧬 Reinterpreting independent evidence as a harmonic family")
        print(f"   eligible sectors: {interpreted.get('eligibleSectorCount')}")
        print(f"   best raw cluster sectors: {cluster.get('sectors') or []}")
        print(f"   raw family median: {cluster.get('medianPeriodDays')} days")
        if harmonic_family:
            print(
                "   possible 2x physical cycle: "
                f"{harmonic_family.get('possibleDoubleCycleDays')} days"
            )
        print(f"   promotion eligible: {interpreted.get('promotionEligible')}")
        print(f"   promotion blockers: {interpreted.get('promotionBlockers') or []}")
        print(f"   claim: {interpreted['claimDecision']['claim']}")

        return StageOutcome(
            result=interpreted,
            next_stage=broad_independent_continuation(
                interpreted,
                request_id=request.id,
                finalize_parameters={"outputSuffix": "v20.3.1"},
            ),
            input_hashes={"broadIndependentProjectResult": sha256_json(run)},
        )

    def morphology_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        harmonic = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.harmonic-family.interpret",
        )
        broad = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.interpret",
        )
        family = ((harmonic or broad or {}).get("harmonicFamily") or {})
        primary_analysis = None
        direct_primary_family = False
        if not family and request.parameters.get("evidenceSource") in {
            "primary-harmonic-contradiction",
            "full-characterization-independent-confirmation",
        }:
            primary_analysis = _required_latest_result_for_handler(
                investigation, "openstar.tess.hypotheses"
            )
            family = _primary_harmonic_morphology_family(primary_analysis) or {}
            direct_primary_family = bool(family)
        raw_period = family.get("representativeRawPeriodDays")
        double_period = family.get("possibleDoubleCycleDays")
        if independent_prepare is None:
            raise RuntimeError(
                "Morphology analysis requires the frozen independent-sector preparation."
            )
        if raw_period is None or double_period is None:
            raise RuntimeError(
                "Morphology analysis requires a recurrent raw period and possible doubled cycle."
            )

        print("🧬 Analyzing folded light-curve morphology across frozen TESS sectors")
        print(f"   recurrent raw family: {raw_period} days")
        print(f"   possible 2x physical cycle: {double_period} days")
        print("   no MAST download; no distributed compute")

        morphology = analyze_morphology(
            primary_dataset_path=prepared["datasetPath"],
            independent_spec=independent_prepare,
            raw_period_days=float(raw_period),
            possible_double_cycle_days=float(double_period),
        )

        for item in morphology.get("sectorResults") or []:
            double_metrics = item.get("doubleWaveMetrics") or {}
            print(
                "   sector "
                f"{item.get('sector')}: "
                f"double gain={item.get('doubleExplainedVarianceImprovement'):.4f}, "
                f"half difference={double_metrics.get('halfCycleDifferenceRatio')}, "
                f"raw={item.get('supportsRawCycle')}, "
                f"double={item.get('supportsDoubleCycle')}"
            )
        print(f"   morphology class: {morphology.get('morphologyClass')}")
        print(f"   phenomenology: {morphology.get('phenomenology')}")
        print(f"   physical cycle resolved: {morphology.get('physicalCycleResolved')}")
        if morphology.get("resolvedPhysicalPeriodDays") is not None:
            print(
                "   resolved physical period: "
                f"{morphology.get('resolvedPhysicalPeriodDays')} days"
            )

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "morphology"
            / "morphology-v20.4.json"
        )
        _write_json(artifact_path, morphology)

        input_hashes = {
            "periodFamily": sha256_json(family),
            "primaryDataset": sha256_file(Path(prepared["datasetPath"])),
        }
        if direct_primary_family:
            input_hashes["primaryAnalysis"] = sha256_json(primary_analysis)
            input_hashes["independentPreparation"] = sha256_json(independent_prepare)
        for item in independent_prepare.get("preparedSectors") or []:
            sector = item.get("sector")
            path = item.get("datasetPath")
            if path:
                input_hashes[f"independentSector{sector}"] = sha256_file(Path(path))

        continuation = morphology.get("continuationEvidence") or {}
        if morphology_event_screening_continuation(morphology, independent_prepare):
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "periodic-event-screen"),
                handler_id="openstar.tess.binary-confirmation.analyze",
                parameters={"entryMode": MORPHOLOGY_EVENT_SCREEN_ENTRY},
                triggered_by_stage_id=request.id,
            )
        elif continuation.get("timeFrequencyEvolutionWarranted"):
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "prepare-time-frequency"),
                handler_id="openstar.tess.time-frequency.prepare",
                parameters={"entryReason": continuation.get("entryReason")},
                triggered_by_stage_id=request.id,
            )
        elif request.parameters.get("unresolvedFallback") == "BROAD_INDEPENDENT_SEARCH":
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "prepare-broad-independent-search"),
                handler_id="openstar.tess.independent.broad.prepare",
                parameters={"continuation": False},
                triggered_by_stage_id=request.id,
            )
        elif blind_transit_search_continuation(
            morphology, independent_prepare, broad
        ):
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "blind-transit-period-search"),
                handler_id=BLIND_TRANSIT_SEARCH_HANDLER_ID,
                parameters={},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.4"},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=morphology,
            next_stage=next_stage,
            input_hashes=input_hashes,
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def blind_transit_search_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        independent_prepare = _required_latest_result_for_handler(
            investigation, "openstar.tess.independent.prepare"
        )
        broad = _latest_result_for_handler(
            investigation, "openstar.tess.independent.broad.interpret"
        )
        morphology = _latest_result_for_handler(
            investigation, "openstar.tess.morphology.analyze"
        )
        targeted = _latest_result_for_handler(
            investigation, "openstar.tess.independent.interpret"
        )
        print("🕳 Searching frozen sectors for a software-blind transit period")
        print("   removing smooth local variability; no catalog period and no MAST download")
        first_pass = analyze_blind_transit_search(
            primary_dataset_path=prepared["datasetPath"],
            independent_spec=independent_prepare,
            morphology=morphology,
            broad_interpretation=broad,
            targeted_interpretation=targeted,
        )
        result = first_pass
        analysis_spec = independent_prepare
        extension_spec = None
        if first_pass.get("classification") == "BLIND_TRANSIT_PERIOD_UNRESOLVED":
            consumed = {
                int(item["sector"])
                for item in independent_prepare.get("preparedSectors") or []
                if item.get("sector") is not None
            }
            remaining = [
                int(sector)
                for sector in independent_prepare.get("candidateSectors") or []
                if int(sector) not in consumed
            ]
            if remaining:
                print("🔭 Extending unresolved blind transit evidence")
                print(
                    "   freezing up to "
                    f"{ADAPTIVE_BLIND_TRANSIT_ADDITIONAL_SECTORS} additional "
                    "balanced sectors"
                )
                artifact_root = store.directory_for(investigation.id) / "artifacts"
                try:
                    extension_spec = build_independent_sector_project(
                        source_project_path=prepared["sourceProjectPath"],
                        source_dataset_entry=prepared["sourceDatasetEntry"],
                        tic_id=int(prepared["ticID"]),
                        primary_sector=prepared.get("sector"),
                        target_period_days=float(independent_prepare["targetPeriodDays"]),
                        candidate_sectors=remaining,
                        output_dir=artifact_root,
                        investigation_id=investigation.id,
                        maximum_sectors=ADAPTIVE_BLIND_TRANSIT_ADDITIONAL_SECTORS,
                        excluded_sectors=list(consumed),
                        artifact_subdirectory="blind-transit-extension-sectors",
                        project_suffix="blind-transit-extension-v1",
                    )
                except TessArchiveInfrastructureError as error:
                    raise RetryableExecutionError(
                        str(error), result=error.diagnostics,
                        input_hashes={
                            "initialBlindTransitSearch": sha256_json(first_pass),
                            "independentPreparation": sha256_json(independent_prepare),
                        },
                    ) from error
                added = extension_spec.get("preparedSectors") or []
                print(
                    "   additional frozen sectors: "
                    f"{[item.get('sector') for item in added]}"
                )
                if added:
                    analysis_spec = dict(independent_prepare)
                    analysis_spec["preparedSectors"] = [
                        *(independent_prepare.get("preparedSectors") or []),
                        *added,
                    ]
                    result = analyze_blind_transit_search(
                        primary_dataset_path=prepared["datasetPath"],
                        independent_spec=analysis_spec,
                        morphology=morphology,
                        broad_interpretation=broad,
                        targeted_interpretation=targeted,
                    )
                    result = dict(result)
                    result["adaptiveSectorExtension"] = {
                        "attempted": True,
                        "reason": "INITIAL_BLIND_TRANSIT_PERIOD_UNRESOLVED",
                        "initialClassification": first_pass.get("classification"),
                        "initialCoarseCandidatePeriodDays": first_pass.get(
                            "coarseCandidatePeriodDays"
                        ),
                        "initialPreparedSectors": [
                            item.get("sector")
                            for item in independent_prepare.get("preparedSectors") or []
                        ],
                        "additionalPreparedSectors": [
                            item.get("sector") for item in added
                        ],
                        "catalogAnswerKeyUsed": False,
                    }
        if result.get("classification") == (
            "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE"
        ):
            result = analyze_iterative_blind_transit_search(
                primary_dataset_path=prepared["datasetPath"],
                independent_spec=analysis_spec,
                morphology=morphology,
                broad_interpretation=broad,
                targeted_interpretation=targeted,
                initial_result=result,
            )
        iterative_extension_reason = _iterative_blind_sector_extension_reason(
            result
        )
        if extension_spec is None and iterative_extension_reason is not None:
            consumed = {
                int(item["sector"])
                for item in analysis_spec.get("preparedSectors") or []
                if item.get("sector") is not None
            }
            remaining = [
                int(sector)
                for sector in independent_prepare.get("candidateSectors") or []
                if int(sector) not in consumed
            ]
            if remaining:
                if iterative_extension_reason == (
                    "MULTI_CLOCK_ITERATIVE_RESIDUAL_SIGNAL_UNRESOLVED"
                ):
                    print("🔭 Extending replicated multi-clock transit evidence")
                    print(
                        "   residual signal unresolved after multiple accepted "
                        "clocks; freezing up to "
                        f"{ADAPTIVE_BLIND_TRANSIT_ADDITIONAL_SECTORS} additional "
                        "balanced sectors"
                    )
                else:
                    print("🔭 Extending conservatively blocked residual evidence")
                    print(
                        "   one clock accepted and a distinct residual family "
                        "passes recurrence gates but remains unpromoted; freezing "
                        "up to "
                        f"{ADAPTIVE_BLIND_TRANSIT_ADDITIONAL_SECTORS} additional "
                        "balanced sectors"
                    )
                initial_iterative_result = result
                artifact_root = store.directory_for(investigation.id) / "artifacts"
                try:
                    extension_spec = build_independent_sector_project(
                        source_project_path=prepared["sourceProjectPath"],
                        source_dataset_entry=prepared["sourceDatasetEntry"],
                        tic_id=int(prepared["ticID"]),
                        primary_sector=prepared.get("sector"),
                        target_period_days=float(
                            independent_prepare["targetPeriodDays"]
                        ),
                        candidate_sectors=remaining,
                        output_dir=artifact_root,
                        investigation_id=investigation.id,
                        maximum_sectors=(
                            ADAPTIVE_BLIND_TRANSIT_ADDITIONAL_SECTORS
                        ),
                        excluded_sectors=list(consumed),
                        artifact_subdirectory=(
                            "blind-transit-extension-sectors"
                        ),
                        project_suffix="blind-transit-extension-v1",
                    )
                except TessArchiveInfrastructureError as error:
                    raise RetryableExecutionError(
                        str(error), result=error.diagnostics,
                        input_hashes={
                            "initialIterativeBlindTransitSearch": sha256_json(
                                initial_iterative_result
                            ),
                            "independentPreparation": sha256_json(
                                independent_prepare
                            ),
                        },
                    ) from error
                added = extension_spec.get("preparedSectors") or []
                print(
                    "   additional frozen sectors: "
                    f"{[item.get('sector') for item in added]}"
                )
                i…99373 tokens truncated…eturn StageOutcome(
            result=spec,
            stop=True,
            final_status="QUIESCENT_AWAITING_DATA",
            input_hashes={
                "sourcePairEvidence": sha256_json(source_evidence),
                "desDr2SeLocalForcedPhotometry": sha256_json(des),
            },
        )

    def atlas_forced_collect_stage(investigation, request):
        submission = _latest_result_for_handler(
            investigation, "openstar.tess.atlas-forced-photometry.prepare")
        if submission is None:
            raise RuntimeError("ATLAS collection requires its immutable submission stage")
        from openstar_external_jobs import ExternalJobStore
        jobs = ExternalJobStore(store.root.parent / "external-jobs")
        exact = [jobs.load(job_id) for job_id in submission["externalJobIDs"]]
        try:
            spec = build_atlas_forced_photometry_project(
                source_project_id=str(submission["sourceProjectID"]),
                source_dataset_id=str(submission["sourceDatasetID"]),
                external_high_resolution_summary={"sourcePair": submission["sourcePair"],
                    "frequencySearch": submission["frequencySearch"]},
                des_dr2_se_summary={"recommendedNextTest": CURRENT_ATLAS_TRIGGER,
                    "frequencySearch": submission["frequencySearch"]},
                output_dir=store.directory_for(investigation.id) / "artifacts",
                investigation_id=investigation.id, external_jobs=exact)
        except ATLASArchiveUnavailable as exc:
            raise RetryableExecutionError(str(exc)) from exc
        project_path = spec.get("projectPath")
        next_stage = StageRequest(
            id=_next_stage_id(request.id, "run-atlas-forced-photometry") if project_path
               else _next_stage_id(request.id, "interpret-atlas-forced-photometry"),
            handler_id="openstar.tess.atlas-forced-photometry.run" if project_path
               else "openstar.tess.atlas-forced-photometry.interpret",
            parameters={"projectPath": project_path} if project_path
               else {"distributedRunExpected": False}, triggered_by_stage_id=request.id)
        return StageOutcome(result=spec, next_stage=next_stage,
                            input_hashes={"submission": sha256_json(submission)})

    def atlas_forced_run_stage(investigation, request):
        print("⚙️ Activating generic ATLAS nightly source-resolved Lomb-Scargle work")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed ATLAS forced-photometry search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-atlas-forced-photometry"),
                handler_id="openstar.tess.atlas-forced-photometry.interpret",
                parameters={"distributedRunExpected": True},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def atlas_forced_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry.collect",
        )
        if preparation is None:  # compatibility with completed synchronous v20.24 stages
            preparation = _latest_result_for_handler(
                investigation, "openstar.tess.atlas-forced-photometry.prepare"
            )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry.run",
        )
        if preparation is None:
            raise RuntimeError("ATLAS forced photometry requires a prepare stage.")
        if bool(request.parameters.get("distributedRunExpected")) and run is None:
            raise RuntimeError(
                "ATLAS forced photometry expected a distributed run result."
            )

        summary = interpret_atlas_forced_photometry_project(
            project_status=run,
            preparation=preparation,
        )
        target = summary.get("targetControl") or {}
        counterpart = summary.get("catalogCounterpartEvidence") or {}
        print("🌐 ATLAS source-resolved forced photometry")
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   corrected Gaia source separation: {summary.get('gaiaPairSeparationArcsec')} arcsec")
        print(f"   target accepted bands: {target.get('acceptedBands')}")
        print(f"   target cross-band supported: {target.get('sourceSupported')}")
        print(f"   counterpart accepted bands: {counterpart.get('acceptedBands')}")
        print(f"   counterpart cross-band supported: {counterpart.get('sourceSupported')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "atlas-forced-photometry"
            / "atlas-forced-photometry-v20.24.json"
        )
        _write_json(artifact_path, summary)

        input_hashes = {"preparation": sha256_json(preparation)}
        if run is not None:
            input_hashes["projectResult"] = sha256_json(run)

        awaiting_followup = (
            summary.get("recommendedNextTest") in {
                ATLAS_SIGNED_REANALYSIS,
                "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY",
            }
            and summary.get("physicalMechanismResolved") is False
        )
        return StageOutcome(
            result=summary,
            next_stage=None if awaiting_followup else StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.24"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes=input_hashes,
            artifacts=(_artifact(artifact_path, "application/json"),),
            stop=awaiting_followup,
            final_status="BLOCKED" if awaiting_followup else None,
        )

    def atlas_reanalysis_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        atlas = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry.interpret",
        )
        if prepared is None or atlas is None:
            raise RuntimeError(
                "v20.25 requires the frozen target and completed v20.24 ATLAS interpretation."
            )

        artifact_root = store.directory_for(investigation.id) / "artifacts"

        print("♻️ Preparing ATLAS signed forced-photometry reanalysis")
        print(f"   target TIC: {prepared.get('ticID')}")
        print("   no new ATLAS query is performed")
        print("   immutable v20.24 raw target-image photometry files are reused")
        print("   low-S/N and negative signed forced fluxes are retained")
        print("   actual fit/error failures remain rejected")
        print("   nightly source-band series become ordinary openstar.lomb-scargle.v1 datasets")

        spec = build_atlas_forced_photometry_reanalysis_project(
            source_project_id=str(prepared["sourceProjectID"]),
            source_dataset_id=str(prepared["datasetID"]),
            atlas_v20_24_summary=atlas,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )

        print(f"   prepared source-band datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   distributed work units: {spec.get('totalWorkUnits')}")

        artifacts: list[ArtifactReference] = []
        for item in spec.get("preparedSeries") or []:
            dataset_path = item.get("datasetPath")
            if dataset_path and Path(dataset_path).exists():
                artifacts.append(_artifact(Path(dataset_path), "application/json"))

        project_path = spec.get("projectPath")
        if project_path and Path(project_path).exists():
            artifacts.append(_artifact(Path(project_path), "application/json"))

        if spec.get("available") and project_path:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "run-atlas-forced-photometry-reanalysis"),
                handler_id="openstar.tess.atlas-forced-photometry-reanalysis.run",
                parameters={"projectPath": project_path},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "interpret-atlas-forced-photometry-reanalysis"),
                handler_id="openstar.tess.atlas-forced-photometry-reanalysis.interpret",
                parameters={"distributedRunExpected": False},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=spec,
            next_stage=next_stage,
            input_hashes={
                "atlasV20_24Interpretation": sha256_json(atlas),
            },
            artifacts=tuple(artifacts),
        )

    def atlas_reanalysis_run_stage(investigation, request):
        print("⚙️ Activating generic ATLAS signed nightly Lomb-Scargle work")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed ATLAS reanalysis search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")

        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-atlas-forced-photometry-reanalysis"),
                handler_id="openstar.tess.atlas-forced-photometry-reanalysis.interpret",
                parameters={"distributedRunExpected": True},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def atlas_reanalysis_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry-reanalysis.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry-reanalysis.run",
        )

        if preparation is None:
            raise RuntimeError("ATLAS reanalysis requires a prepare stage.")
        if bool(request.parameters.get("distributedRunExpected")) and run is None:
            raise RuntimeError(
                "ATLAS reanalysis expected a distributed run result."
            )

        summary = interpret_atlas_forced_photometry_reanalysis_project(
            project_status=run,
            preparation=preparation,
        )

        target = summary.get("targetControl") or {}
        counterpart = summary.get("catalogCounterpartEvidence") or {}

        print("♻️ ATLAS signed forced-photometry reanalysis")
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   target accepted bands: {target.get('acceptedBands')}")
        print(f"   target cross-band supported: {target.get('sourceSupported')}")
        print(f"   counterpart accepted bands: {counterpart.get('acceptedBands')}")
        print(f"   counterpart cross-band supported: {counterpart.get('sourceSupported')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "atlas-forced-photometry-reanalysis"
            / "atlas-forced-photometry-reanalysis-v20.25.json"
        )
        _write_json(artifact_path, summary)

        input_hashes = {"preparation": sha256_json(preparation)}
        if run is not None:
            input_hashes["projectResult"] = sha256_json(run)

        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.25"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes=input_hashes,
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def atlas_time_resolved_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        atlas_v20_24 = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry.interpret",
        )
        atlas_v20_25_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry-reanalysis.prepare",
        )
        atlas_v20_25 = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry-reanalysis.interpret",
        )

        if (
            prepared is None
            or atlas_v20_24 is None
            or atlas_v20_25_prepare is None
            or atlas_v20_25 is None
        ):
            raise RuntimeError(
                "v20.26 requires the frozen target plus completed v20.24 and v20.25 ATLAS stages."
            )

        artifact_root = store.directory_for(investigation.id) / "artifacts"

        print("🕰️ Preparing ATLAS time-resolved counterpart recurrence")
        print(f"   target TIC: {prepared.get('ticID')}")
        print("   no new ATLAS query is performed")
        print("   immutable v20.24 counterpart photometry is reused")
        print("   the shared c/o cadence is split into independent observing seasons")
        print("   each season/filter keeps the same strict prominence >= 2.0 acceptance rule")
        print("   no TESS drift extrapolation is used to choose ATLAS season frequencies")

        spec = build_atlas_time_resolved_project(
            source_project_id=str(prepared["sourceProjectID"]),
            source_dataset_id=str(prepared["datasetID"]),
            atlas_v20_24_summary=atlas_v20_24,
            atlas_v20_25_preparation=atlas_v20_25_prepare,
            atlas_v20_25_summary=atlas_v20_25,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )

        print(f"   seasons: {len(spec.get('seasons') or [])}")
        print(f"   prepared season-band datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   distributed work units: {spec.get('totalWorkUnits')}")

        artifacts: list[ArtifactReference] = []
        for item in spec.get("preparedSeries") or []:
            dataset_path = item.get("datasetPath")
            if dataset_path and Path(dataset_path).exists():
                artifacts.append(_artifact(Path(dataset_path), "application/json"))

        project_path = spec.get("projectPath")
        if project_path and Path(project_path).exists():
            artifacts.append(_artifact(Path(project_path), "application/json"))

        if spec.get("available") and project_path:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "run-atlas-time-resolved"),
                handler_id="openstar.tess.atlas-time-resolved.run",
                parameters={"projectPath": project_path},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "interpret-atlas-time-resolved"),
                handler_id="openstar.tess.atlas-time-resolved.interpret",
                parameters={"distributedRunExpected": False},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=spec,
            next_stage=next_stage,
            input_hashes={
                "atlasV20_24Interpretation": sha256_json(atlas_v20_24),
                "atlasV20_25Preparation": sha256_json(atlas_v20_25_prepare),
                "atlasV20_25Interpretation": sha256_json(atlas_v20_25),
            },
            artifacts=tuple(artifacts),
        )

    def atlas_time_resolved_run_stage(investigation, request):
        print("⚙️ Activating generic ATLAS season/filter Lomb-Scargle work")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed ATLAS time-resolved search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")

        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-atlas-time-resolved"),
                handler_id="openstar.tess.atlas-time-resolved.interpret",
                parameters={"distributedRunExpected": True},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def atlas_time_resolved_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-time-resolved.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-time-resolved.run",
        )

        if preparation is None:
            raise RuntimeError("ATLAS time-resolved recurrence requires a prepare stage.")
        if bool(request.parameters.get("distributedRunExpected")) and run is None:
            raise RuntimeError(
                "ATLAS time-resolved recurrence expected a distributed run result."
            )

        summary = interpret_atlas_time_resolved_project(
            project_status=run,
            preparation=preparation,
        )
        counterpart = summary.get("catalogCounterpartEvidence") or {}

        print("🕰️ ATLAS time-resolved counterpart recurrence")
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   accepted season-band results: {counterpart.get('acceptedSeasonBandCount')}")
        print(f"   accepted seasons: {counterpart.get('acceptedSeasons')}")
        print(f"   accepted bands: {counterpart.get('acceptedBands')}")
        print(f"   cross-band-consistent seasons: {counterpart.get('crossBandConsistentSeasons')}")
        print(f"   independent ATLAS frequency trend: {counterpart.get('independentFrequencyTrend')}")
        print(f"   counterpart supported: {counterpart.get('sourceSupported')}")
        print(f"   counterpart suggestive: {counterpart.get('sourceSuggestive')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "atlas-time-resolved"
            / "atlas-time-resolved-v20.26.json"
        )
        _write_json(artifact_path, summary)

        input_hashes = {"preparation": sha256_json(preparation)}
        if run is not None:
            input_hashes["projectResult"] = sha256_json(run)

        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.26"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes=input_hashes,
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def atlas_fixed_window_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(
            investigation,
            "openstar.tess.prepare-target",
        )
        atlas_v20_24 = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry.interpret",
        )
        atlas_v20_26 = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-time-resolved.interpret",
        )

        if (
            prepared is None
            or atlas_v20_24 is None
            or atlas_v20_26 is None
        ):
            raise RuntimeError(
                "v20.27 requires the frozen target plus completed v20.24 and v20.26 ATLAS results."
            )

        artifact_root = (
            store.directory_for(investigation.id) / "artifacts"
        )

        print("🧱 Preparing ATLAS fixed-window counterpart recurrence")
        print(f"   target TIC: {prepared.get('ticID')}")
        print("   no new ATLAS query is performed")
        print("   immutable v20.24 counterpart photometry is reused")
        print("   v20.26 gap-based season splitting is replaced because it produced one 4.6-year season")
        print("   windows are fixed non-overlapping 180-day absolute-MJD bins")
        print("   the strict prominence >= 2.0 acceptance rule is unchanged")
        print("   boundary-hit candidates are rejected")
        print("   no TESS drift extrapolation is used to choose ATLAS window frequencies")

        spec = build_atlas_fixed_window_project(
            source_project_id=str(prepared["sourceProjectID"]),
            source_dataset_id=str(prepared["datasetID"]),
            atlas_v20_24_summary=atlas_v20_24,
            atlas_v20_26_summary=atlas_v20_26,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )

        print(f"   windows intersecting data: {len(spec.get('windows') or [])}")
        print(f"   prepared window-band datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   distributed work units: {spec.get('totalWorkUnits')}")

        artifacts: list[ArtifactReference] = []

        for item in spec.get("preparedSeries") or []:
            dataset_path = item.get("datasetPath")
            if dataset_path and Path(dataset_path).exists():
                artifacts.append(
                    _artifact(
                        Path(dataset_path),
                        "application/json",
                    )
                )

        project_path = spec.get("projectPath")
        if project_path and Path(project_path).exists():
            artifacts.append(
                _artifact(
                    Path(project_path),
                    "application/json",
                )
            )

        if spec.get("available") and project_path:
            next_stage = StageRequest(
                id=_next_stage_id(
                    request.id,
                    "run-atlas-fixed-window",
                ),
                handler_id=(
                    "openstar.tess.atlas-fixed-window.run"
                ),
                parameters={
                    "projectPath": project_path,
                },
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(
                    request.id,
                    "interpret-atlas-fixed-window",
                ),
                handler_id=(
                    "openstar.tess.atlas-fixed-window.interpret"
                ),
                parameters={
                    "distributedRunExpected": False,
                },
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=spec,
            next_stage=next_stage,
            input_hashes={
                "atlasV20_24Interpretation": sha256_json(
                    atlas_v20_24
                ),
                "atlasV20_26Interpretation": sha256_json(
                    atlas_v20_26
                ),
            },
            artifacts=tuple(artifacts),
        )

    def atlas_fixed_window_run_stage(investigation, request):
        print("⚙️ Activating generic ATLAS fixed-window Lomb-Scargle work")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed ATLAS fixed-window search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")

        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(
                    request.id,
                    "interpret-atlas-fixed-window",
                ),
                handler_id=(
                    "openstar.tess.atlas-fixed-window.interpret"
                ),
                parameters={
                    "distributedRunExpected": True,
                },
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def atlas_fixed_window_interpret_stage(
        investigation,
        request,
    ):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-fixed-window.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-fixed-window.run",
        )

        if preparation is None:
            raise RuntimeError(
                "ATLAS fixed-window recurrence requires a prepare stage."
            )

        if (
            bool(
                request.parameters.get(
                    "distributedRunExpected"
                )
            )
            and run is None
        ):
            raise RuntimeError(
                "ATLAS fixed-window recurrence expected a distributed run result."
            )

        summary = interpret_atlas_fixed_window_project(
            project_status=run,
            preparation=preparation,
        )
        counterpart = (
            summary.get("catalogCounterpartEvidence") or {}
        )

        print("🧱 ATLAS fixed-window counterpart recurrence")
        print(
            f"   classification: "
            f"{summary.get('classification')}"
        )
        print(
            f"   residual mode origin: "
            f"{summary.get('residualModeOrigin')}"
        )
        print(
            "   accepted window-band results: "
            f"{counterpart.get('acceptedWindowBandCount')}"
        )
        print(
            f"   accepted windows: "
            f"{counterpart.get('acceptedWindows')}"
        )
        print(
            f"   accepted bands: "
            f"{counterpart.get('acceptedBands')}"
        )
        print(
            "   cross-band-consistent windows: "
            f"{counterpart.get('crossBandConsistentWindows')}"
        )
        print(
            "   independent ATLAS frequency trend: "
            f"{counterpart.get('independentATLASFrequencyTrend')}"
        )
        print(
            f"   counterpart supported: "
            f"{counterpart.get('sourceSupported')}"
        )
        print(
            f"   counterpart suggestive: "
            f"{counterpart.get('sourceSuggestive')}"
        )
        print(
            f"   recommended next test: "
            f"{summary.get('recommendedNextTest')}"
        )

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "atlas-fixed-windows"
            / "atlas-fixed-window-v20.27.json"
        )
        _write_json(artifact_path, summary)

        input_hashes = {
            "preparation": sha256_json(preparation),
        }
        if run is not None:
            input_hashes["projectResult"] = sha256_json(run)

        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(
                    request.id,
                    "finalize",
                ),
                handler_id="openstar.tess.finalize",
                parameters={
                    "outputSuffix": "v20.27",
                },
                triggered_by_stage_id=request.id,
            ),
            input_hashes=input_hashes,
            artifacts=(
                _artifact(
                    artifact_path,
                    "application/json",
                ),
            ),
        )

    def targeted_observation_plan_stage(investigation, request):
        prepared = _latest_result_for_handler(
            investigation,
            "openstar.tess.prepare-target",
        )
        external = _latest_result_for_handler(
            investigation,
            "openstar.tess.external-high-resolution-variability-validation.interpret",
        )
        atlas_fixed = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-fixed-window.interpret",
        )
        atlas_forced_stage = next((
            stage for stage in reversed(investigation.stages)
            if stage.handler_id == "openstar.tess.atlas-forced-photometry.interpret"
            and stage.status == "COMPLETE"
            and stage.result is not None
        ), None)
        atlas_forced = atlas_forced_stage.result if atlas_forced_stage is not None else None

        if prepared is None or (atlas_fixed is None and atlas_forced is None):
            raise RuntimeError(
                "v20.28 requires the frozen target and a supported ATLAS result."
            )

        direct_atlas = atlas_fixed is None
        frozen_pair_evidence = None
        if direct_atlas:
            assert atlas_forced_stage is not None and atlas_forced is not None
            position = investigation.stages.index(atlas_forced_stage)
            preceding = investigation.stages[:position]
            later = investigation.stages[position + 1:]
            valid_lineage = bool(
                request.triggered_by_stage_id == atlas_forced_stage.id
                and atlas_forced_stage.triggered_by_stage_id
                and any(
                    item.id == atlas_forced_stage.triggered_by_stage_id
                    and item.status == "COMPLETE"
                    and item.handler_id in (
                        "openstar.tess.atlas-forced-photometry.run",
                        "openstar.tess.atlas-forced-photometry.prepare",
                    )
                    for item in preceding
                )
            )
            superseded = any(
                item.handler_id.startswith((
                    "openstar.tess.atlas-forced-photometry-reanalysis.",
                    "openstar.tess.atlas-time-resolved.",
                    "openstar.tess.atlas-fixed-window.",
                ))
                for item in later
            )
            prior_attempt = any(
                item.handler_id
                == "openstar.tess.targeted-observation-planning.generate"
                and item.id != request.id
                for item in investigation.stages
            )
            frozen_pair_evidence = frozen_source_pair_evidence(
                investigation, atlas_forced_stage
            )
            if not (
                valid_lineage
                and not superseded
                and not prior_attempt
                and atlas_forced.get("classification")
                == "ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
                and atlas_forced.get("residualModeOrigin")
                == "ARCHIVAL_ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
                and atlas_forced.get("recommendedNextTest")
                == "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
                and atlas_forced.get("physicalMechanismResolved") is False
                and frozen_pair_evidence is not None
            ):
                raise RuntimeError(
                    "Direct v20.24 targeted-observation planning evidence is incomplete, ambiguous, or superseded."
                )
        elif external is None:
            raise RuntimeError(
                "v20.28 fixed-window planning requires completed v20.19 evidence."
            )
        elif atlas_fixed.get("recommendedNextTest") != (
            "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        ):
            raise RuntimeError(
                "v20.27 did not leave the investigation at targeted high-resolution time-series photometry."
            )

        artifact_root = (
            store.directory_for(investigation.id) / "artifacts"
        )

        print("🔭 Building targeted high-resolution time-series observation plan")
        print(f"   target TIC: {prepared.get('ticID')}")
        print("   no archive query is performed")
        print("   no distributed science work is activated")
        print("   the existing residual-frequency search band is frozen into the plan")
        print("   the existing prominence >= 2.0 acceptance rule is retained")
        print("   campaign cadence, image-quality, exposure-tier, filter, and ingest rules are preregistered")
        print("   the established main-family target association remains unchanged")

        plan = build_targeted_observation_plan(
            source_project_id=str(prepared["sourceProjectID"]),
            source_dataset_id=str(prepared["datasetID"]),
            investigation_id=investigation.id,
            external_high_resolution_summary=(
                frozen_pair_evidence if direct_atlas else external
            ),
            output_dir=artifact_root,
            atlas_fixed_window_summary=atlas_fixed,
            atlas_forced_photometry_summary=(atlas_forced if direct_atlas else None),
        )

        geometry = plan.get("sourceGeometry") or {}
        cadence = plan.get("cadence") or {}
        artifacts_map = plan.get("artifacts") or {}

        print(f"   Gaia source separation: {geometry.get('separationArcsec')} arcsec")
        print(f"   minimum campaign baseline: {cadence.get('minimumBaselineDays')} days")
        print(f"   preferred campaign baseline: {cadence.get('preferredBaselineDays')} days")
        print(f"   minimum distinct nights: {cadence.get('minimumDistinctNights')}")
        print(f"   preferred distinct nights: {cadence.get('preferredDistinctNights')}")
        print(f"   JSON plan: {artifacts_map.get('jsonPlanPath')}")
        print(f"   markdown plan: {artifacts_map.get('markdownPlanPath')}")
        print(f"   CSV ingest template: {artifacts_map.get('csvIngestTemplatePath')}")
        print(f"   recommended next test: {plan.get('recommendedNextTest')}")

        artifacts: list[ArtifactReference] = []

        for key, media_type in (
            ("jsonPlanPath", "application/json"),
            ("markdownPlanPath", "text/markdown"),
            ("csvIngestTemplatePath", "text/csv"),
        ):
            path_text = artifacts_map.get(key)
            if path_text and Path(path_text).exists():
                artifacts.append(
                    _artifact(
                        Path(path_text),
                        media_type,
                    )
                )

        return StageOutcome(
            result=plan,
            next_stage=StageRequest(
                id=_next_stage_id(
                    request.id,
                    "finalize",
                ),
                handler_id="openstar.tess.finalize",
                parameters={
                    "outputSuffix": "v20.28",
                },
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                (
                    "frozenSourcePairEvidence"
                    if direct_atlas
                    else "externalHighResolutionValidation"
                ): sha256_json(
                    frozen_pair_evidence if direct_atlas else external
                ),
                (
                    "atlasForcedPhotometry"
                    if direct_atlas
                    else "atlasFixedWindowRecurrence"
                ): sha256_json(atlas_forced if direct_atlas else atlas_fixed),
            },
            artifacts=tuple(artifacts),
        )

    def finalize_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        primary_analysis = _latest_result_for_handler(investigation, "openstar.tess.hypotheses")
        planner = _latest_result_for_handler(investigation, "openstar.tess.planner")
        if primary_analysis is None or planner is None:
            raise RuntimeError("Finalization requires completed hypotheses and planner stages.")
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        independent_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.interpret",
        )
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        harmonic_family_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.harmonic-family.interpret",
        )
        broad_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.interpret",
        )
        broad_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.prepare",
        )
        morphology_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        blind_transit_search = _latest_blind_transit_result(investigation)
        physical_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.physical.interpret",
        )
        binary_confirmation = _latest_result_for_handler(
            investigation,
            "openstar.tess.binary-confirmation.analyze",
        )
        eclipse_event_localization = _latest_result_for_handler(
            investigation, ECLIPSE_LOCALIZATION_HANDLER_ID,
        )
        source_attribution_review = _latest_result_for_handler(
            investigation, SOURCE_ATTRIBUTION_REVIEW_HANDLER_ID,
        )
        event_depth_attenuation_audit = _latest_result_for_handler(
            investigation, EVENT_DEPTH_AUDIT_HANDLER_ID,
        )
        joint_event_phase_model = _latest_result_for_handler(
            investigation, JOINT_EVENT_PHASE_MODEL_HANDLER_ID,
        )
        external_companion_evidence = _latest_result_for_handler(
            investigation, EXTERNAL_EVIDENCE_INTERPRET_HANDLER_ID,
        )
        final_companion_evidence_synthesis = _latest_result_for_handler(
            investigation, COMPANION_SYNTHESIS_HANDLER_ID,
        )
        source_localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.source-localization.analyze",
        )
        multimode_decomposition = _latest_result_for_handler(
            investigation,
            "openstar.tess.multimode.summarize",
        )
        time_frequency_evolution = _latest_result_for_handler(
            investigation,
            "openstar.tess.time-frequency.summarize",
        )
        v20_8_long_baseline_time_frequency_confirmation = (
            _latest_result_for_handler(
                investigation,
                V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID,
            )
        )
        transient_mode_validation = _latest_result_for_handler(
            investigation,
            TRANSIENT_MODE_VALIDATION_HANDLER_ID,
        )
        nonstationary_modeling = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.summarize",
        )
        mode_identification = _latest_result_for_handler(
            investigation, "openstar.tess.mode-identification.analyze",
        )
        long_baseline_frequency_confirmation = _latest_result_for_handler(
            investigation, LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID,
        )
        dynamic_harmonic_modeling = _latest_result_for_handler(
            investigation, "openstar.tess.dynamic-harmonic.analyze",
        )
        dynamic_harmonic_frequency_refinement = _latest_result_for_handler(
            investigation, "openstar.tess.dynamic-harmonic.frequency-refinement",
        )
        residual_mode_localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization.interpret",
        )
        residual_external_evidence = _latest_result_for_handler(
            investigation, RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID,
        )
        target_residual_astrophysical_mechanism = _latest_result_for_handler(
            investigation, TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID,
        )
        residual_mode_localization_review = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization-review.interpret",
        )
        neighbor_catalog_pixel_response_review = _latest_result_for_handler(
            investigation,
            NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID,
        )
        multisource_residual = _latest_result_for_handler(
            investigation,
            "openstar.tess.multi-source-residual.interpret",
        )
        target_residual_multisector_source = _latest_result_for_handler(
            investigation, "openstar.tess.target-residual-multisector-source.interpret")
        offset_source_identification = _latest_result_for_handler(
            investigation,
            "openstar.tess.offset-source-identification.analyze",
        )
        offset_source_variability = _latest_result_for_handler(
            investigation,
            "openstar.tess.offset-source-variability.interpret",
        )
        calibrated_prf_deblending = _latest_result_for_handler(
            investigation,
            "openstar.tess.calibrated-prf-deblending.interpret",
        )
        difference_image_localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.difference-image-localization.interpret",
        )
        frequency_localized_pixel_response = _latest_result_for_handler(
            investigation,
            "openstar.tess.frequency-localized-pixel-response.interpret",
        )
        official_spoc_prf_forward_modeling = _latest_result_for_handler(
            investigation,
            "openstar.tess.official-spoc-prf-forward-modeling.interpret",
        )
        catalog_counterpart_identification = _latest_result_for_handler(
            investigation,
            "openstar.tess.catalog-counterpart-identification.analyze",
        )
        deep_catalog_counterpart_identification = _latest_result_for_handler(
            investigation,
            DEEP_CATALOG_COUNTERPART_HANDLER_ID,
        )
        deep_catalog_prf_localization = _latest_result_for_handler(
            investigation,
            DEEP_CATALOG_PRF_INTERPRET_HANDLER_ID,
        )
        external_high_resolution_validation = _latest_result_for_handler(
            investigation,
            "openstar.tess.external-high-resolution-variability-validation.interpret",
        )
        skymapper_resolved_photometry = _latest_result_for_handler(
            investigation,
            "openstar.tess.skymapper-resolved-photometry.interpret",
        )
        nsc_resolved_photometry = _latest_result_for_handler(
            investigation,
            "openstar.tess.nsc-resolved-photometry.interpret",
        )
        noirlab_image_forced_photometry = _latest_result_for_handler(
            investigation,
            "openstar.tess.noirlab-image-forced-photometry.interpret",
        )
        des_dr2_se_local_forced_photometry = _latest_result_for_handler(
            investigation,
            "openstar.tess.des-dr2-se-local-forced-photometry.interpret",
        )
        atlas_forced_photometry = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry.interpret",
        )
        atlas_forced_photometry_reanalysis = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-forced-photometry-reanalysis.interpret",
        )
        atlas_time_resolved = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-time-resolved.interpret",
        )
        atlas_fixed_window = _latest_result_for_handler(
            investigation,
            "openstar.tess.atlas-fixed-window.interpret",
        )
        targeted_observation_plan = _latest_result_for_handler(
            investigation,
            "openstar.tess.targeted-observation-planning.generate",
        )
        target_residual_mechanism_predictive_validation = _latest_result_for_handler(
            investigation,
            "openstar.tess.target-residual-mechanism-predictive-validation.analyze",
        )
        target_residual_astrophysical_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.target-residual-astrophysical-interpretation.analyze",
        )
        main_family_time_domain_recurrence = _latest_result_for_handler(
            investigation, "openstar.tess.main-family-time-domain-recurrence.analyze")
        main_family_frequency_domain_reassessment = _latest_result_for_handler(
            investigation, "openstar.tess.main-family-frequency-domain-reassessment.analyze")
        target_residual_archival_baseline_extension = _latest_result_for_handler(
            investigation, "openstar.tess.target-residual-archival-baseline.interpret",
        )
        target_residual_pixel_recurrence = _latest_result_for_handler(
            investigation, "openstar.tess.target-residual-pixel-recurrence.interpret",
        )

        if (blind_transit_search or {}).get("classification") == (
            "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE"
        ):
            claim_decision = blind_transit_search["claimDecision"]
            selected_period = blind_transit_search.get("candidatePeriodDays")
            selected_source = "software-blind-multi-sector-box-period-search"
        elif blind_transit_search is not None:
            claim_decision = blind_transit_search["claimDecision"]
            selected_period = None
            selected_source = None
        elif harmonic_family_interpretation is not None:
            claim_decision = harmonic_family_interpretation["claimDecision"]
            selected_period = harmonic_family_interpretation.get("selectedPeriodDays")
            selected_source = harmonic_family_interpretation.get("selectedSource")
        elif broad_interpretation is not None:
            claim_decision = broad_interpretation["claimDecision"]
            selected_period = broad_interpretation.get("selectedPeriodDays")
            selected_source = broad_interpretation.get("selectedSource")
        elif independent_interpretation is not None:
            claim_decision = independent_interpretation["claimDecision"]
            selected_period = independent_interpretation.get("selectedPeriodDays")
            selected_source = independent_interpretation.get("selectedSource")
        elif followup_interpretation is not None:
            claim_decision = followup_interpretation["claimDecision"]
            selected_period = followup_interpretation.get("selectedPeriodDays")
            selected_source = followup_interpretation.get("selectedSource")
        else:
            claim_decision = planner.get("claimDecision")
            selected_period = primary_analysis.get("observedPeriodDays")
            selected_source = "primary-distributed-search"

        if not claim_decision:
            raise RuntimeError("Finalization reached without a claim decision.")
        validate_claim(claim_decision["claim"])

        if morphology_interpretation is not None and morphology_interpretation.get(
            "physicalCycleResolved"
        ):
            resolved_period = morphology_interpretation.get("resolvedPhysicalPeriodDays")
            morphology_class = morphology_interpretation.get("morphologyClass")
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": [
                    (
                        "Multi-sector folded-light-curve morphology resolves the harmonic "
                        f"interpretation as {morphology_class} at approximately "
                        f"{resolved_period} days."
                    ),
                    (
                        "The claim level is not automatically upgraded by morphology alone; "
                        "independent recurrence promotion remains governed by the existing "
                        "sector-count, cluster-width, prominence, boundary, and coverage rules."
                    ),
                ],
            }

        if physical_interpretation is not None:
            preferred = physical_interpretation.get("preferredPhotometricHypothesis")
            next_test = physical_interpretation.get("recommendedNextTest")
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                (
                    "v20.5 physical-mechanism discrimination ranks the frozen multi-sector "
                    f"photometry with preferred photometric hypothesis {preferred or 'UNRESOLVED'}; "
                    "it does not treat morphology alone as a resolved physical mechanism."
                )
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {next_test}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if binary_confirmation is not None:
            evidence = binary_confirmation.get("independentEvidence") or {}
            ephemeris = binary_confirmation.get("linearEphemeris") or {}
            opposite = binary_confirmation.get("oppositeConjunctionEvidence") or {}
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "Fixed-clock independent narrow-event replication classified the "
                f"orbital-geometry evidence as {evidence.get('classification')}; "
                f"linear-ephemeris coherence={ephemeris.get('coherent')}, and the "
                "separate opposite-conjunction test classified the evidence as "
                f"{opposite.get('classification')}. This does not resolve companion nature."
            )
            if ephemeris.get("timingSectors") is not None:
                existing_rationale.append(
                    f"Ephemeris timing sectors: {ephemeris.get('timingSectors')}; "
                    "primary timing epoch included: "
                    f"{str(ephemeris.get('primarySectorIncluded')).lower()}."
                )
            existing_rationale.append(
                "Authoritative recommended next test: "
                f"{binary_confirmation.get('recommendedNextTest')}."
            )
            claim_decision = {"claim": claim_decision["claim"],
                              "rationale": existing_rationale}

        if eclipse_event_localization is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "Fixed-ephemeris eclipse difference imaging classified source attribution as "
                f"{eclipse_event_localization.get('classification')} using "
                f"{eclipse_event_localization.get('usableIndependentSectorCount')} usable independent sectors. "
                "This spatial result does not resolve companion nature or physical mechanism."
            )
            existing_rationale.append(
                "Authoritative recommended next spatial test: "
                f"{eclipse_event_localization.get('recommendedNextTest')}."
            )
            claim_decision = {"claim": claim_decision["claim"], "rationale": existing_rationale}

        if source_attribution_review is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "Durable source-attribution review independently recomputed "
                f"{source_attribution_review.get('supportingIndependentSectorCount')} supporting "
                f"sectors and classified the boundary as {source_attribution_review.get('classification')}."
            )
            claim_decision = {"claim": claim_decision["claim"], "rationale": existing_rationale}
        if external_companion_evidence is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            interpretation_boundary = (
                "At the external-evidence interpretation stage, companion nature and the "
                "detailed photometric mechanism were still pending final synthesis."
                if final_companion_evidence_synthesis is not None else
                "Companion nature and the detailed photometric mechanism remain pending final synthesis."
            )
            existing_rationale.append(
                "Published known-object confirmation evidence was kept separate from the "
                "software-blind photometric/spatial evidence and classified as "
                f"{external_companion_evidence.get('classification')}. {interpretation_boundary}"
            )
            claim_decision = {"claim": claim_decision["claim"], "rationale": existing_rationale}
        if final_companion_evidence_synthesis is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "Final hash-linked synthesis resolved the known companion mass regime and "
                f"source relationship as {final_companion_evidence_synthesis.get('classification')}. "
                "OpenStar independently recovered evidence consistent with a previously known "
                "companion; detailed photometric mechanism remains unresolved and no automatic "
                "discovery claim is made. Human scientific review is required."
            )
            claim_decision = {"claim": claim_decision["claim"], "rationale": existing_rationale}

        if source_localization is not None:
            cross = source_localization.get("crossSector") or {}
            localization_class = cross.get("classification")
            next_test = source_localization.get("recommendedNextTest")
            existing_rationale = list(claim_decision.get("rationale") or [])
            if localization_class == "TARGET_SOURCE_SUPPORTED":
                existing_rationale.append(
                    "v20.6 pixel-level localization supports the TIC target as the origin of the periodic variability across a strict majority of eligible independent sectors; static aperture-flux contamination is still not claimed absent."
                )
            elif localization_class == "OFF_TARGET_VARIABLE_SOURCE_SUPPORTED":
                existing_rationale.append(
                    "v20.6 pixel-level localization places the periodic variability away from the TIC target across a strict majority of eligible independent sectors with a consistent sky offset."
                )
                claim_decision = {
                    "claim": "HUMAN_REVIEW_REQUIRED",
                    "rationale": existing_rationale,
                }
            else:
                existing_rationale.append(
                    "v20.6 pixel-level localization did not resolve the source association strongly enough to attribute the periodic variability to the TIC target or to an offset source."
                )
            existing_rationale.append(f"Recommended next confirmation step: {next_test}.")
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }
            validate_claim(claim_decision["claim"])

        if multimode_decomposition is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.7 iterative residual prewhitening classifies the post-family frequency structure as "
                f"{multimode_decomposition.get('classification')}; this does not by itself change the claim level or resolve the physical mechanism."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {multimode_decomposition.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if time_frequency_evolution is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            residual = time_frequency_evolution.get("residualEvolution") or {}
            family = time_frequency_evolution.get("familyEvolution") or {}
            existing_rationale.append(
                "v20.8 sliding-window time-frequency analysis classifies the residual evolution as "
                f"{residual.get('classification')} and the established-family evolution as "
                f"{family.get('classification')}; it does not by itself change the claim level or resolve the physical mechanism."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {time_frequency_evolution.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if v20_8_long_baseline_time_frequency_confirmation is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.8.1 leave-one-independent-sector-out long-baseline "
                "time-frequency confirmation classifies the unresolved "
                "residual structure as "
                f"{v20_8_long_baseline_time_frequency_confirmation.get('classification')}; "
                "it does not upgrade the claim or resolve the physical mechanism."
            )
            existing_rationale.append(
                "Recommended next confirmation step: "
                f"{v20_8_long_baseline_time_frequency_confirmation.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if mode_identification is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            version = (
                "v20.8.2 confirmed-coherent full-sector"
                if mode_identification.get("evidenceLineage")
                == V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE
                else "stable-residual"
            )
            existing_rationale.append(
                f"The {version} mode-identification comparison classifies "
                "the residual as "
                f"{mode_identification.get('classification')}; it does not "
                "upgrade the claim or resolve the pulsation or physical "
                "mechanism."
            )
            existing_rationale.append(
                "Recommended next confirmation step: "
                f"{mode_identification.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if long_baseline_frequency_confirmation is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.9.1 leave-one-independent-sector-out long-baseline "
                "confirmation classifies the ambiguous residual as "
                f"{long_baseline_frequency_confirmation.get('classification')}; "
                "the held-out predictive analysis does not upgrade the claim "
                "or resolve the physical mechanism."
            )
            existing_rationale.append(
                "Recommended next confirmation step: "
                f"{long_baseline_frequency_confirmation.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if nonstationary_modeling is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.9 distributed long-baseline model comparison classifies the residual structure as "
                f"{nonstationary_modeling.get('classification')} using generic Lomb-Scargle work units over a deterministic frequency-drift grid; "
                "it does not by itself change the claim level or resolve the physical mechanism."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {nonstationary_modeling.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if residual_mode_localization is not None:
            cross = residual_mode_localization.get("crossSector") or {}
            localization_class = cross.get("classification")
            existing_rationale = list(claim_decision.get("rationale") or [])
            if localization_class == "RESIDUAL_MODE_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.10 distributed pixel localization supports the TIC target as the origin of the v20.9 drifting residual component across a strict majority of eligible independent sectors; this supports intrinsic multi-component variability but does not identify the physical mechanism."
                )
            elif localization_class == "RESIDUAL_MODE_OFF_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.10 distributed pixel localization places the drifting residual component away from the TIC target across a consistent strict majority of eligible independent sectors; this does not undo the v20.6 target localization of the main periodic family."
                )
            else:
                existing_rationale.append(
                    "v20.10 distributed pixel localization does not resolve whether the drifting residual component is target-centered or offset; the main periodic family's v20.6 target association remains unchanged."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {residual_mode_localization.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if residual_external_evidence is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.10.1 adjudicates only the external variability and binary "
                "classifications already frozen by the catalog-identity stage, "
                "classifying that context as "
                f"{residual_external_evidence.get('classification')}. It retains "
                "discordant off-target sector evidence and does not resolve the "
                "physical mechanism or change the claim level."
            )
            existing_rationale.append(
                "Recommended next confirmation step: "
                f"{residual_external_evidence.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if target_residual_astrophysical_mechanism is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.10.2 conservatively adjudicates whether the frozen, "
                "target-associated nonbinary catalog evidence is specifically "
                "consistent with the residual period, classifying the hypothesis "
                "as "
                f"{target_residual_astrophysical_mechanism.get('classification')}. "
                "It retains discordant off-target evidence and neither resolves "
                "the physical mechanism nor changes the claim level."
            )
            existing_rationale.append(
                "Recommended next confirmation step: "
                f"{target_residual_astrophysical_mechanism.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }


        if residual_mode_localization_review is not None:
            cross_time = residual_mode_localization_review.get("crossTime") or {}
            review_class = cross_time.get("classification")
            existing_rationale = list(claim_decision.get("rationale") or [])
            if review_class == "RESIDUAL_MODE_TARGET_SUPPORTED_TIME_RESOLVED":
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization supports the TIC target as the dominant origin of the drifting residual component across a strict majority of eligible independent sectors."
                )
            elif review_class == "RESIDUAL_MODE_OFF_TARGET_SUPPORTED_TIME_RESOLVED":
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization supports a consistent off-target origin for the drifting residual component; this remains separate from the v20.6 target association of the established main family."
                )
            elif review_class == "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND":
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization finds target-centered and offset residual behavior across time/sectors, supporting source switching or blended nonstationary variability rather than a single static residual centroid."
                )
            else:
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization still does not uniquely assign the drifting residual component to the TIC target or one consistent offset source."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {residual_mode_localization_review.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if neighbor_catalog_pixel_response_review is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.11.1 projects a fixed TIC/Gaia neighborhood through the "
                "persisted v20.11 per-window pixel-to-sky geometry and compares "
                "only the already-frozen residual power maps, classifying the "
                "source evidence as "
                f"{neighbor_catalog_pixel_response_review.get('classification')}. "
                "It neither rereads flux nor changes the claim or physical "
                "mechanism status."
            )
            existing_rationale.append(
                "Recommended next confirmation step: "
                f"{neighbor_catalog_pixel_response_review.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if multisource_residual is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = multisource_residual.get("classification")
            if classification == "MULTIPLE_RESIDUAL_SOURCES_SUPPORTED":
                existing_rationale.append(
                    "v20.12 spatial decomposition supports both target-centered and offset residual components, explaining the v20.11 source-switching/blend behavior without changing the established main-family target association."
                )
            elif classification == "TARGET_RESIDUAL_COMPONENT_DOMINANT":
                existing_rationale.append(
                    "v20.12 spatial decomposition finds the target-centered residual component dominant after separating the v20.11 blended/source-switching evidence."
                )
            elif classification == "OFF_TARGET_RESIDUAL_COMPONENT_DOMINANT":
                existing_rationale.append(
                    "v20.12 spatial decomposition finds an offset residual component dominant; this remains separate from the v20.6 target localization of the established main periodic family."
                )
            else:
                existing_rationale.append(
                    "v20.12 spatial decomposition does not yet separate the time-variable residual structure strongly enough to assign it to one or multiple sky components."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {multisource_residual.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if target_residual_mechanism_predictive_validation is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = target_residual_mechanism_predictive_validation.get(
                "classification"
            )
            existing_rationale.append(
                "v20.16 performed deterministic held-out validation of the four "
                "preregistered target-residual temporal model families using local "
                "training-only refits; it performed no distributed work or archive query."
            )
            if classification == (
                "TARGET_RESIDUAL_MECHANISM_PREDICTIVE_VALIDATION_UNRESOLVED"
            ):
                existing_rationale.append(
                    "No temporal mechanism replicated predictively across the required "
                    "independent sectors, so the target-residual mechanism remains unresolved."
                )
            else:
                existing_rationale.append(
                    f"The predictive-validation classification is {classification}."
                )
            limitations = (
                target_residual_mechanism_predictive_validation.get("failClosedReasons")
                or []
            )
            if limitations:
                existing_rationale.append(
                    "Conservative predictive-validation limitations: "
                    + "; ".join(str(reason) for reason in limitations)
                    + ". These limitations are not positive evidence for a mechanism."
                )
            branch_recommendation = (
                target_residual_mechanism_predictive_validation.get(
                    "recommendedNextTest"
                )
            )
            existing_rationale.append(
                "v20.16 recommends the next target-residual test: "
                f"{branch_recommendation}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if offset_source_identification is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = offset_source_identification.get("classification")
            best = offset_source_identification.get("bestCandidate") or {}
            ids = best.get("catalogIDs") or {}
            if classification == "KNOWN_VARIABLE_CATALOG_COUNTERPART_IDENTIFIED":
                existing_rationale.append(
                    "v20.13 catalog crossmatch identifies a spatially favored non-target counterpart to the dominant offset residual component and finds independent catalog evidence that the counterpart is variable; this still requires a direct variability-match test before attributing the TESS residual to that source."
                )
            elif classification == "CATALOG_COUNTERPART_IDENTIFIED":
                existing_rationale.append(
                    "v20.13 catalog crossmatch identifies a spatially favored non-target counterpart to the dominant offset residual component, but catalog position alone does not prove that source produces the TESS residual variability."
                )
            elif classification == "MULTIPLE_PLAUSIBLE_CATALOG_COUNTERPARTS":
                existing_rationale.append(
                    "v20.13 catalog crossmatch finds multiple plausible non-target counterparts within the offset-component localization uncertainty, so the residual source identity remains unresolved."
                )
            elif classification == "CATALOG_QUERY_INCOMPLETE":
                existing_rationale.append(
                    "v20.13 could not complete enough external catalog queries to identify the dominant offset residual component securely."
                )
            else:
                existing_rationale.append(
                    "v20.13 finds no secure catalog counterpart for the dominant offset residual component within the adopted localization uncertainty."
                )
            if ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Best offset counterpart identifiers: TIC={ids.get('ticID')}, GaiaDR3={ids.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {offset_source_identification.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if offset_source_variability is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = offset_source_variability.get("classification")
            counterpart = offset_source_variability.get("catalogCounterpart") or {}
            if classification == "OFFSET_COUNTERPART_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.14 catalog-guided target/counterpart deblending supports the v20.13 non-target catalog counterpart as the dominant carrier of residual variability matching the v20.12 offset component; this does not alter the target association of the established main periodic family."
                )
            elif classification == "TARGET_AND_OFFSET_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.14 finds matching residual variability in both the catalog counterpart and the target-control component, supporting a blended two-source residual interpretation rather than assigning all residual structure to one object."
                )
            elif classification == "OFFSET_COUNTERPART_VARIABILITY_NOT_SUPPORTED":
                existing_rationale.append(
                    "v20.14 does not validate the v20.13 catalog counterpart as the carrier of the offset residual; the target-control component is better supported and calibrated PRF deblending remains necessary."
                )
            elif classification == "OFFSET_COUNTERPART_VARIABILITY_SUGGESTIVE":
                existing_rationale.append(
                    "v20.14 finds suggestive but insufficient independent-sector variability support for the v20.13 catalog counterpart, so the residual source attribution remains provisional."
                )
            else:
                existing_rationale.append(
                    "v20.14 catalog-guided variability validation remains unresolved and does not securely attribute the residual variability to the v20.13 counterpart or the target-control component."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Validated counterpart identifiers under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {offset_source_variability.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if calibrated_prf_deblending is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = calibrated_prf_deblending.get("classification")
            counterpart = calibrated_prf_deblending.get("catalogCounterpart") or {}
            if classification == "PRF_OFFSET_COUNTERPART_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.15 sector-calibrated empirical pixel-response deblending supports the v20.13 non-target catalog counterpart as the dominant carrier of residual variability matching the v20.12 offset component; the established main periodic family's v20.6 target association remains unchanged."
                )
            elif classification == "PRF_TARGET_AND_OFFSET_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.15 sector-calibrated pixel-response deblending supports residual variability in both the target-control and catalog-counterpart components, strengthening the blended two-source interpretation."
                )
            elif classification == "PRF_TARGET_CONTROL_DOMINANT":
                existing_rationale.append(
                    "v20.15 sector-calibrated pixel-response deblending favors the Blind C target-control component over the proposed offset counterpart for the residual family, so the intrinsic target residual must be modeled directly."
                )
            elif classification == "PRF_OFFSET_COUNTERPART_VARIABILITY_SUGGESTIVE":
                existing_rationale.append(
                    "v20.15 improves the spatial model beyond fixed Gaussian templates and still finds only suggestive offset-counterpart support; difference-image localization is required before source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.15 sector-calibrated pixel-response deblending remains unresolved between the target and offset counterpart; it does not change the established main-family source association."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Calibrated deblend counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {calibrated_prf_deblending.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if difference_image_localization is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = difference_image_localization.get("classification")
            counterpart = difference_image_localization.get("catalogCounterpart") or {}
            if classification == "DIFFERENCE_IMAGE_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.16 drift-corrected high-minus-low TESS difference images localize the residual variability to the v20.13 catalog counterpart across at least three independent sectors; this is an image-domain confirmation separate from the v20.14/v20.15 source-amplitude deblends and does not alter the established main-family target association."
                )
            elif classification == "DIFFERENCE_IMAGE_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.16 difference images localize the residual variability back to the Blind C target across at least three independent sectors, arguing that the residual is intrinsic after the established-family subtraction."
                )
            elif classification == "DIFFERENCE_IMAGE_MIXED_OR_BLENDED":
                existing_rationale.append(
                    "v20.16 difference images support both target-centered and counterpart-centered residual behavior in independent sectors, preserving a blended/time-variable two-source interpretation."
                )
            elif classification == "DIFFERENCE_IMAGE_COUNTERPART_SUGGESTIVE":
                existing_rationale.append(
                    "v20.16 difference images favor the catalog counterpart in multiple sectors but do not reach the independent-sector threshold required for secure source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.16 difference-image localization remains unresolved and therefore does not assign the drifting residual securely to either Blind C or the catalog counterpart."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Difference-image counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {difference_image_localization.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if frequency_localized_pixel_response is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = frequency_localized_pixel_response.get("classification")
            counterpart = frequency_localized_pixel_response.get("catalogCounterpart") or {}
            if classification == "FREQUENCY_LOCALIZED_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.17 narrow-band per-pixel Lomb-Scargle plus fixed-frequency phase-coherent response mapping localizes the residual variability to the catalog counterpart across at least three independent sectors; this source attribution remains separate from the v20.6 target association of the established main periodic family."
                )
            elif classification == "FREQUENCY_LOCALIZED_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.17 frequency-localized phase-coherent pixel response maps localize the residual variability back to Blind C across at least three independent sectors, supporting an intrinsic target residual after the established-family subtraction."
                )
            elif classification == "FREQUENCY_LOCALIZED_MIXED_OR_BLENDED":
                existing_rationale.append(
                    "v20.17 frequency-localized pixel response maps support both Blind C and the catalog counterpart in independent sectors, preserving a mixed or blended residual-source interpretation."
                )
            elif classification in {
                "FREQUENCY_LOCALIZED_COUNTERPART_SUGGESTIVE",
                "FREQUENCY_LOCALIZED_TARGET_SUGGESTIVE",
            }:
                existing_rationale.append(
                    "v20.17 frequency-localized pixel response mapping favors one source in multiple sectors but does not reach the independent-sector threshold required for secure residual-source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.17 frequency-localized pixel response confirmation remains unresolved and therefore does not securely assign the residual to Blind C or the catalog counterpart."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Frequency-localized counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {frequency_localized_pixel_response.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if official_spoc_prf_forward_modeling is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = official_spoc_prf_forward_modeling.get("classification")
            counterpart = official_spoc_prf_forward_modeling.get("catalogCounterpart") or {}
            if classification == "SPOC_PRF_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling independently separates the target and catalog counterpart and supports the counterpart residual across the required independent sectors; this attribution applies only to the residual component and does not alter the v20.6 target association of the established main periodic family."
                )
            elif classification == "SPOC_PRF_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling supports the residual component on Blind C across the required independent sectors after separating the catalog neighbor, favoring an intrinsic target residual."
                )
            elif classification == "SPOC_PRF_TARGET_AND_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling supports residual variability in both Blind C and the catalog counterpart across independent sectors, preserving a blended/multi-source interpretation."
                )
            elif classification in {"SPOC_PRF_COUNTERPART_SUGGESTIVE", "SPOC_PRF_TARGET_SUGGESTIVE"}:
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling favors one source but still does not reach the independent-sector threshold required for secure residual-source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling remains unable to securely assign the residual to Blind C or the catalog counterpart; the TESS pixel-scale source attribution is therefore treated as unresolved rather than forcing a source."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Official-PRF counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {official_spoc_prf_forward_modeling.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if deep_catalog_counterpart_identification is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = deep_catalog_counterpart_identification.get("classification")
            if classification == "DEEP_CATALOG_COUNTERPART_IDENTIFIED":
                existing_rationale.append(
                    "A source in SkyMapper DR4 and/or NSC DR2 is positionally consistent "
                    "with the frozen PRF residual location, but this deeper-catalog match "
                    "does not establish which source carries the residual variability."
                )
            elif classification == "AMBIGUOUS_DEEP_CATALOG_COUNTERPARTS":
                existing_rationale.append(
                    "The deeper catalog search finds multiple plausible sources at the "
                    "frozen PRF residual location, so source identity remains ambiguous."
                )
            elif classification == "NO_DEEP_CATALOG_COUNTERPART":
                existing_rationale.append(
                    "Successful SkyMapper DR4 and NSC DR2 searches find no usable non-target "
                    "source at the frozen PRF residual location; dedicated high-resolution "
                    "imaging is therefore required."
                )
            else:
                existing_rationale.append(
                    "The deeper catalog search is incomplete because an external catalog "
                    "service was unavailable; absence of a counterpart is not inferred."
                )
            existing_rationale.append(
                "Recommended next confirmation step: "
                f"{deep_catalog_counterpart_identification.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if external_high_resolution_validation is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = external_high_resolution_validation.get("classification")
            pair = external_high_resolution_validation.get("sourcePair") or {}
            if classification == "GAIA_DR3_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry supports residual-band variability on the catalog counterpart while the target control does not meet the same acceptance guard. This external attribution applies only to the residual component and does not alter the established main periodic family's v20.6 target association."
                )
            elif classification == "GAIA_DR3_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry supports residual-band variability on Blind C while the catalog counterpart does not meet the same acceptance guard, favoring an intrinsic target residual."
                )
            elif classification == "GAIA_DR3_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry supports residual-band variability in both Blind C and the catalog counterpart, preserving a multi-source interpretation."
                )
            elif classification == "GAIA_DR3_EPOCH_PHOTOMETRY_UNAVAILABLE":
                existing_rationale.append(
                    "v20.19 could not obtain usable Gaia DR3 epoch photometry for either frozen source, so the external archive provides no source-attribution evidence."
                )
            elif classification == "GAIA_DR3_SOURCE_RESOLVED_COVERAGE_INCOMPLETE":
                existing_rationale.append(
                    "v20.19 has source-resolved Gaia DR3 epoch coverage for only part of the frozen target/counterpart pair, so it does not force a residual-source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry does not securely distinguish the residual-band variability between Blind C and the catalog counterpart."
                )
            existing_rationale.append(
                "The v20.9 TESS frequency-drift model was not extrapolated backward into the Gaia observing epoch; Gaia was searched independently inside the frozen residual-frequency band."
            )
            existing_rationale.append(
                f"External source pair: target GaiaDR3={pair.get('targetGaiaDR3SourceID')}, counterpart GaiaDR3={pair.get('counterpartGaiaDR3SourceID')}."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {external_high_resolution_validation.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if skymapper_resolved_photometry is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = skymapper_resolved_photometry.get("classification")
            if classification == "SKYMAPPER_DR4_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.20 SkyMapper DR4 clean good-seeing PSF photometry finds cross-band residual-frequency support on the catalog counterpart but not the target control, providing archival support for the offset residual source."
                )
            elif classification == "SKYMAPPER_DR4_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.20 SkyMapper DR4 clean good-seeing PSF photometry finds cross-band residual-frequency support on Blind C but not the catalog counterpart, favoring an intrinsic target residual."
                )
            elif classification == "SKYMAPPER_DR4_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.20 SkyMapper DR4 clean good-seeing PSF photometry finds cross-band residual-frequency support on both frozen sources, preserving a multi-source interpretation."
                )
            elif classification == "SKYMAPPER_DR4_PAIR_NOT_SEPARATELY_RESOLVED":
                existing_rationale.append(
                    "v20.20 finds that SkyMapper DR4 does not represent the frozen Gaia pair as two distinct usable survey objects, so the archive cannot provide source-resolved variability evidence at this separation."
                )
            elif classification == "SKYMAPPER_DR4_NO_QUALIFYING_RESOLVED_EPOCH_SERIES":
                existing_rationale.append(
                    "v20.20 finds no SkyMapper DR4 time series with enough clean, position-matched, good-seeing PSF epochs to test the residual-frequency band without relaxing the spatial-quality guard."
                )
            elif "SUGGESTIVE" in str(classification):
                existing_rationale.append(
                    "v20.20 SkyMapper DR4 provides only single-band/suggestive residual-frequency evidence and therefore does not promote the residual source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.20 SkyMapper DR4 resolved-photometry screening remains unable to securely distinguish the drifting residual source."
                )
            existing_rationale.append(
                "SkyMapper was treated only as an opportunistic archival screen: detections had to be clean, position-matched, and obtained in seeing tighter than the frozen pair separation; the TESS drift law was not extrapolated into the SkyMapper epoch."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {skymapper_resolved_photometry.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if nsc_resolved_photometry is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = nsc_resolved_photometry.get("classification")
            if classification == "NSC_DR2_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.21 NOIRLab Source Catalog DR2 co-detected resolved photometry finds cross-band residual-frequency support on the catalog counterpart but not the target control."
                )
            elif classification == "NSC_DR2_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.21 NOIRLab Source Catalog DR2 co-detected resolved photometry finds cross-band residual-frequency support on Blind C but not the catalog counterpart."
                )
            elif classification == "NSC_DR2_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.21 NOIRLab Source Catalog DR2 co-detected resolved photometry supports residual-band variability on both frozen sources."
                )
            elif classification == "NSC_DR2_PAIR_NOT_SEPARATELY_RESOLVED":
                existing_rationale.append(
                    "v20.21 finds that NSC DR2 does not safely represent the frozen Gaia pair as two distinct source matches, so the archive cannot provide source-resolved variability evidence."
                )
            elif classification == "NSC_DR2_NO_QUALIFYING_CODETECTED_RESOLVED_SERIES":
                existing_rationale.append(
                    "v20.21 finds no NSC DR2 time series with enough same-exposure, independently position-matched co-detections of both frozen sources to test the residual-frequency band."
                )
            elif "SUGGESTIVE" in str(classification):
                existing_rationale.append(
                    "v20.21 NSC DR2 provides only single-band/suggestive residual-frequency evidence and therefore does not promote the residual source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.21 NOIRLab Source Catalog DR2 resolved-photometry screening remains unable to securely distinguish the drifting residual source."
                )
            existing_rationale.append(
                "NSC measurements were admitted only when both distinct sources were co-detected in the same exposure/filter and independently position-matched to their frozen Gaia coordinates; the TESS drift law was not extrapolated into the NSC epochs."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {nsc_resolved_photometry.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if noirlab_image_forced_photometry is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = noirlab_image_forced_photometry.get("classification")
            if classification == "NOIRLAB_IMAGE_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.22 image-level two-source forced photometry on public calibrated NOIRLab images finds cross-band residual-frequency support on the catalog counterpart but not the target control."
                )
            elif classification == "NOIRLAB_IMAGE_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.22 image-level two-source forced photometry on public calibrated NOIRLab images finds cross-band residual-frequency support on Blind C but not the catalog counterpart."
                )
            elif classification == "NOIRLAB_IMAGE_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.22 image-level two-source forced photometry on public calibrated NOIRLab images supports residual-band variability on both frozen sources."
                )
            elif classification == "NOIRLAB_IMAGE_FORCED_PHOTOMETRY_SATURATION_LIMIT":
                existing_rationale.append(
                    "v20.22 reaches an archival image-level saturation limit: the public NOIRLab exposures do not provide enough unsaturated two-source fits for a source-resolved residual-frequency test."
                )
            elif classification == "NOIRLAB_IMAGE_FORCED_PHOTOMETRY_NO_QUALIFYING_EXPOSURES":
                existing_rationale.append(
                    "v20.22 finds public NOIRLab image coverage but no exposures that pass the preregistered two-source PSF, saturation, conditioning, fit-quality, and source-SNR guards."
                )
            elif classification == "NOIRLAB_IMAGE_FORCED_PHOTOMETRY_INSUFFICIENT_TIME_SERIES":
                existing_rationale.append(
                    "v20.22 obtains some acceptable image-level two-source fits but not enough calibrated epochs/baseline to form a defensible distributed residual-frequency test."
                )
            elif "SUGGESTIVE" in str(classification):
                existing_rationale.append(
                    "v20.22 image-level NOIRLab photometry provides only single-band/suggestive residual-frequency evidence and therefore does not promote the residual source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.22 image-level NOIRLab two-source forced photometry remains unable to securely distinguish the drifting residual source."
                )
            existing_rationale.append(
                "v20.22 corrects the archival spatial geometry by recomputing the Blind-C-to-counterpart separation directly from the two frozen Gaia coordinates. The earlier catalogSeparationArcsec field was the offset-component-to-catalog-match association distance, not the Gaia source-pair separation; no prior stage promoted a positive claim from that field."
            )
            existing_rationale.append(
                "v20.22 bypassed the NSC extracted-object catalog and fit the public calibrated image pixels directly at the frozen Gaia source positions; the TESS drift law was not extrapolated into the archival image epochs."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {noirlab_image_forced_photometry.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if des_dr2_se_local_forced_photometry is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = des_dr2_se_local_forced_photometry.get("classification")
            if classification == "DES_DR2_SE_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.23 DES DR2 single-epoch source-local forced photometry finds cross-band residual-frequency support on the frozen catalog counterpart without requiring Blind C to be unsaturated in the same exposure."
                )
            elif classification == "DES_DR2_SE_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.23 DES DR2 single-epoch source-local forced photometry finds cross-band residual-frequency support on Blind C."
                )
            elif classification == "DES_DR2_SE_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.23 DES DR2 single-epoch source-local forced photometry supports residual-band variability on both frozen Gaia sources."
                )
            elif classification == "DES_DR2_SE_NO_FIELD_COVERAGE":
                existing_rationale.append(
                    "v20.23 finds no usable DES DR2 single-epoch coverage for the frozen Gaia source pair."
                )
            elif classification == "DES_DR2_SE_NO_QUALIFYING_LOCAL_SOURCE_FITS":
                existing_rationale.append(
                    "v20.23 finds DES DR2 single-epoch coverage but no source-local image fits that pass the preregistered saturation, PSF, conditioning, fit-quality, and SNR guards."
                )
            elif classification == "DES_DR2_SE_LOCAL_PHOTOMETRY_INSUFFICIENT_TIME_SERIES":
                existing_rationale.append(
                    "v20.23 recovers some acceptable DES DR2 source-local measurements but not enough independent epochs and baseline to form a defensible residual-frequency time series."
                )
            elif "SUGGESTIVE" in str(classification):
                existing_rationale.append(
                    "v20.23 DES DR2 source-local photometry provides only single-band or otherwise suggestive residual-frequency evidence and therefore does not promote the source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.23 DES DR2 single-epoch source-local forced photometry remains unable to securely assign the drifting residual source."
                )
            existing_rationale.append(
                "v20.23 uses the corrected Gaia-to-Gaia separation and fits the two sources in independent local cutouts; saturation of Blind C is not used to veto the counterpart unless the counterpart's own pixels are contaminated."
            )
            existing_rationale.append(
                "The TESS frequency-drift law was not extrapolated into DES epochs; accepted DES light curves were searched only inside the frozen residual-frequency band."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {des_dr2_se_local_forced_photometry.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if atlas_forced_photometry is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = atlas_forced_photometry.get("classification")
            if classification == "ATLAS_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.24 ATLAS calibrated target-image forced photometry finds cross-filter residual-frequency support on the frozen catalog counterpart but not Blind C."
                )
            elif classification == "ATLAS_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.24 ATLAS calibrated target-image forced photometry finds cross-filter residual-frequency support on Blind C but not the frozen catalog counterpart."
                )
            elif classification == "ATLAS_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.24 ATLAS calibrated target-image forced photometry supports residual-band variability on both frozen Gaia sources."
                )
            elif classification == "ATLAS_NO_QUALIFYING_FORCED_PHOTOMETRY_TIME_SERIES":
                existing_rationale.append(
                    "v20.24 ATLAS forced photometry does not produce enough quality-controlled nightly source-resolved measurements to test the residual-frequency band under the preregistered guards."
                )
            elif "SUGGESTIVE" in str(classification):
                existing_rationale.append(
                    "v20.24 ATLAS source-resolved forced photometry provides only single-filter or otherwise suggestive residual-frequency evidence and therefore does not promote the residual source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.24 ATLAS source-resolved forced photometry remains unable to securely assign the drifting residual source."
                )
            existing_rationale.append(
                "ATLAS target-image forced photometry was used instead of southern difference imaging; each source was measured independently at its frozen Gaia coordinate using the corrected Gaia-to-Gaia separation."
            )
            existing_rationale.append(
                "The TESS frequency-drift law was not extrapolated into ATLAS epochs; accepted nightly light curves were searched independently inside the frozen residual-frequency band."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {atlas_forced_photometry.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if atlas_forced_photometry_reanalysis is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = atlas_forced_photometry_reanalysis.get("classification")
            if classification == "ATLAS_REANALYSIS_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.25 statistically valid reanalysis of the immutable ATLAS target-image forced-photometry files finds cross-filter residual-frequency support on the frozen catalog counterpart but not Blind C."
                )
            elif classification == "ATLAS_REANALYSIS_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.25 statistically valid reanalysis of the immutable ATLAS target-image forced-photometry files finds cross-filter residual-frequency support on Blind C."
                )
            elif classification == "ATLAS_REANALYSIS_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.25 statistically valid ATLAS forced-photometry reanalysis supports residual-band variability on both frozen Gaia sources."
                )
            elif classification == "ATLAS_REANALYSIS_NO_QUALIFYING_NIGHTLY_TIME_SERIES":
                existing_rationale.append(
                    "v20.25 corrects the v20.24 individual-SNR selection gate but still cannot form enough quality-controlled nightly ATLAS time series for a defensible residual-frequency test."
                )
            elif "SUGGESTIVE" in str(classification):
                existing_rationale.append(
                    "v20.25 ATLAS signed-flux reanalysis provides only single-filter or otherwise suggestive residual-frequency evidence and therefore does not promote the residual source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.25 ATLAS signed-flux reanalysis remains unable to securely assign the drifting residual source."
                )
            existing_rationale.append(
                "v20.25 is a methodological correction to v20.24: it reuses the same immutable ATLAS target-image files but removes the inappropriate requirement that each forced measurement be an individual positive >=3-sigma detection before nightly binning."
            )
            existing_rationale.append(
                "Actual tphot errors and excessive fit chi/N remain rejected; signed quality-valid fluxes are inverse-variance binned nightly before the frozen residual-frequency search."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {atlas_forced_photometry_reanalysis.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if atlas_time_resolved is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = atlas_time_resolved.get("classification")

            if classification == "ATLAS_TIME_RESOLVED_COUNTERPART_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.26 time-resolved ATLAS analysis finds strict residual-band recurrence on the frozen catalog counterpart across multiple independent observing seasons and filters."
                )
            elif classification == "ATLAS_TIME_RESOLVED_COUNTERPART_VARIABILITY_SUGGESTIVE":
                existing_rationale.append(
                    "v20.26 time-resolved ATLAS analysis finds recurrent strict residual-band evidence on the frozen catalog counterpart, but the preregistered multi-season/multi-filter support requirements are not fully satisfied."
                )
            elif classification == "ATLAS_TIME_RESOLVED_NO_QUALIFYING_SEASON_SERIES":
                existing_rationale.append(
                    "v20.26 cannot construct enough independent ATLAS season/filter series to perform the preregistered recurrence test."
                )
            else:
                existing_rationale.append(
                    "v20.26 splits the immutable ATLAS counterpart light curve into independent observing seasons but does not confirm strict recurrent residual-band variability."
                )

            existing_rationale.append(
                "v20.26 does not lower the prominence>=2.0 gate that blocked the v20.25 global peaks; every seasonal result must independently satisfy the same acceptance threshold."
            )
            existing_rationale.append(
                "The TESS frequency-drift law is not extrapolated into ATLAS epochs; any ATLAS frequency trend is fit only after independent seasonal frequencies are measured."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {atlas_time_resolved.get('recommendedNextTest')}."
            )

            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if atlas_fixed_window is not None:
            existing_rationale = list(
                claim_decision.get("rationale") or []
            )
            classification = atlas_fixed_window.get(
                "classification"
            )

            if classification == "ATLAS_FIXED_WINDOW_COUNTERPART_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.27 fixed-window ATLAS analysis finds strict residual-band recurrence on the frozen catalog counterpart across multiple deterministic independent 180-day windows and both ATLAS filters."
                )
            elif classification == "ATLAS_FIXED_WINDOW_COUNTERPART_VARIABILITY_SUGGESTIVE":
                existing_rationale.append(
                    "v20.27 fixed-window ATLAS analysis finds strict residual-band recurrence on the frozen catalog counterpart in multiple deterministic windows, but the preregistered strong-support criteria are not fully satisfied."
                )
            elif classification == "ATLAS_FIXED_WINDOW_NO_QUALIFYING_WINDOW_SERIES":
                existing_rationale.append(
                    "v20.27 deterministic 180-day ATLAS windows do not contain enough quality-controlled cadence to perform the preregistered recurrence test."
                )
            else:
                existing_rationale.append(
                    "v20.27 deterministic 180-day ATLAS windows do not confirm strict recurrent residual-band variability on the frozen catalog counterpart."
                )

            existing_rationale.append(
                "v20.27 corrects only the v20.26 segmentation rule: v20.26 produced one 4.6-year season because the ATLAS cadence contained no gap longer than 75 days. v20.27 uses non-overlapping 180-day bins anchored to absolute MJD zero, independent of the measured light curve and frequencies."
            )
            existing_rationale.append(
                "The prominence>=2.0 gate is unchanged, search-grid boundary hits are rejected, and the TESS frequency-drift law is not extrapolated into ATLAS epochs."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {atlas_fixed_window.get('recommendedNextTest')}."
            )

            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if targeted_observation_plan is not None:
            existing_rationale = list(
                claim_decision.get("rationale") or []
            )
            existing_rationale.append(
                "v20.28 treats the public archival branch as exhausted for the residual-source attribution question and freezes a targeted source-resolved observing experiment before new measurements are collected."
            )
            existing_rationale.append(
                "The observation plan preserves the existing residual-frequency search band, the RELIABLE + prominence>=2.0 acceptance rule, cross-filter consistency, and time-resolved recurrence requirements rather than tuning thresholds after future data are seen."
            )
            existing_rationale.append(
                "The campaign requires paired short/deep exposures so Blind C can remain unsaturated in the short tier while the fainter counterpart can reach useful precision in the deep tier; target saturation in deep frames is allowed only when it does not contaminate the counterpart measurement region."
            )
            existing_rationale.append(
                "The established main periodic-family target association is unchanged; the new campaign tests only the drifting residual component."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {targeted_observation_plan.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        period_evidence = _build_period_evidence(
            claim_decision=claim_decision,
            selected_period=selected_period,
            selected_source=selected_source,
            primary_analysis=primary_analysis,
            followup_interpretation=followup_interpretation,
            independent_interpretation=independent_interpretation,
            broad_interpretation=broad_interpretation,
            harmonic_family_interpretation=harmonic_family_interpretation,
        )
        if morphology_interpretation is not None:
            period_evidence["morphologyClass"] = morphology_interpretation.get("morphologyClass")
            period_evidence["phenomenology"] = morphology_interpretation.get("phenomenology")
            if morphology_interpretation.get("physicalCycleResolved"):
                period_evidence["physicalCycleResolved"] = True
                resolved_physical_period = morphology_interpretation.get(
                    "resolvedPhysicalPeriodDays"
                )
                period_evidence["physicalPeriodDays"] = resolved_physical_period
                period_evidence["resolvedPhysicalPeriodDays"] = (
                    resolved_physical_period
                )
                period_evidence["interpretation"] = "morphology-resolved-physical-cycle"
                period_evidence["candidateSource"] = "multi-sector-morphology-discrimination"
        dynamic_alias = (
            (dynamic_harmonic_modeling or {}).get("periodAliasResolution") or {})
        try:
            dynamic_resolved_period = float(
                (dynamic_harmonic_modeling or {}).get(
                    "resolvedPhysicalPeriodDays"))
        except (TypeError, ValueError):
            dynamic_resolved_period = math.nan
        predictive_dynamic_cycle = (
            (dynamic_harmonic_modeling or {}).get("physicalCycleResolved") is True
            and (dynamic_harmonic_modeling or {}).get("referencePeriodRole")
            == "PREDICTIVELY_RESOLVED_PHOTOMETRIC_CYCLE"
            and dynamic_alias.get("method")
            == "NESTED_EVEN_ONLY_VS_EVEN_PLUS_ODD_LEAVE_ONE_SECTOR_OUT_PREDICTION"
            and dynamic_alias.get("selectedPeriodRelation") == "DOUBLE_CYCLE"
            and math.isfinite(dynamic_resolved_period)
            and dynamic_resolved_period > 0
        )
        if predictive_dynamic_cycle:
            period_evidence["physicalCycleResolved"] = True
            period_evidence["physicalPeriodDays"] = dynamic_resolved_period
            period_evidence["resolvedPhysicalPeriodDays"] = (
                dynamic_resolved_period)
            period_evidence["interpretation"] = (
                "nested-predictive-odd-harmonic-resolved-photometric-cycle")
            period_evidence["candidateSource"] = (
                "leave-one-independent-sector-out-odd-harmonic-prediction")
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "A matched-frequency nested comparison resolves the doubled "
                "photometric cycle through predictive odd-harmonic support in "
                "at least three independent held-out sectors. The claim level "
                "and physical mechanism remain unchanged."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }
        if (blind_transit_search or {}).get("classification") == (
            "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE"
        ):
            transit_period = blind_transit_search.get("candidatePeriodDays")
            ephemeris = blind_transit_search.get("linearEphemeris") or {}
            candidate_periods = []
            for candidate in blind_transit_search.get("candidateSignals") or []:
                try:
                    candidate_period = float(candidate.get("candidatePeriodDays"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(candidate_period) and candidate_period > 0.0:
                    candidate_periods.append(candidate_period)
            if not candidate_periods and transit_period is not None:
                candidate_periods.append(float(transit_period))
            period_evidence.update({
                "candidatePeriodDays": transit_period,
                "candidateSource": "software-blind-multi-sector-box-period-search",
                "recurrentPhotometricPeriodDays": transit_period,
                "possiblePhysicalCycleDays": None,
                "physicalCycleResolved": False,
                "physicalPeriodDays": None,
                "transitLikeEventPeriodDays": transit_period,
                "transitLikeEventReferenceEpoch": ephemeris.get("referenceEpoch"),
                "transitLikeEventTimingRmsOMinusCDays": ephemeris.get("rmsOMinusCDays"),
                "transitLikeCandidateCount": len(candidate_periods),
                "transitLikeCandidatePeriodsDays": candidate_periods,
                "interpretation": "replicated-transit-like-event-period-candidate",
            })
        if binary_confirmation is not None and authoritative_binary_gate(binary_confirmation):
            ephemeris = binary_confirmation["linearEphemeris"]
            refined_period = ephemeris.get("refinedPeriodDays")
            reference_epoch = ephemeris.get("referenceEpoch")
            try:
                refined_period = float(refined_period)
                reference_epoch = float(reference_epoch)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Authoritative binary confirmation lacks a finite refined ephemeris."
                ) from error
            if not (math.isfinite(refined_period) and refined_period > 0
                    and math.isfinite(reference_epoch)):
                raise ValueError(
                    "Authoritative binary confirmation lacks a finite refined ephemeris."
                )
            period_evidence["morphologyResolvedPhysicalPeriodDays"] = (
                period_evidence.get("physicalPeriodDays")
            )
            period_evidence["physicalCycleResolved"] = True
            period_evidence["physicalPeriodDays"] = refined_period
            period_evidence["resolvedPhysicalPeriodDays"] = refined_period
            period_evidence["eventTimingReferenceEpoch"] = reference_epoch
            period_evidence["eventTimingRmsOMinusCDays"] = ephemeris.get(
                "rmsOMinusCDays"
            )
            period_evidence["eventTimingMaximumAbsoluteOMinusCDays"] = ephemeris.get(
                "maximumAbsoluteOMinusCDays"
            )
            period_evidence["eventTimingSectors"] = ephemeris.get("timingSectors")
            period_evidence["interpretation"] = (
                "replicated-event-linear-ephemeris-refined-physical-cycle"
            )
            period_evidence["candidateSource"] = (
                "multi-sector-replicated-event-timing-refinement"
            )
        if source_localization is not None:
            cross = source_localization.get("crossSector") or {}
            period_evidence["sourceAssociation"] = cross.get("variableSignalOrigin")
            period_evidence["sourceAssociationClassification"] = cross.get("classification")
            period_evidence["targetAssociationResolved"] = cross.get("classification") in {
                "TARGET_SOURCE_SUPPORTED",
                "OFF_TARGET_VARIABLE_SOURCE_SUPPORTED",
            }

        archival_stage_index = max((index for index, stage in enumerate(investigation.stages)
            if stage.status == "COMPLETE" and stage.handler_id ==
            "openstar.tess.target-residual-archival-baseline.interpret"), default=-1)
        later_unrelated_science = any(index > archival_stage_index and stage.status == "COMPLETE"
            and stage.handler_id != "openstar.tess.finalize" and not stage.handler_id.startswith(
                "openstar.tess.target-residual-archival-baseline.")
            for index, stage in enumerate(investigation.stages))
        if neighbor_catalog_pixel_response_review is not None:
            recommended_next_test = neighbor_catalog_pixel_response_review.get(
                "recommendedNextTest"
            )
        elif target_residual_astrophysical_mechanism is not None:
            recommended_next_test = target_residual_astrophysical_mechanism.get(
                "recommendedNextTest"
            )
        elif residual_external_evidence is not None:
            recommended_next_test = residual_external_evidence.get(
                "recommendedNextTest"
            )
        elif long_baseline_frequency_confirmation is not None:
            recommended_next_test = long_baseline_frequency_confirmation.get(
                "recommendedNextTest"
            )
        elif target_residual_astrophysical_interpretation is not None:
            recommended_next_test = target_residual_astrophysical_interpretation.get("recommendedNextTest")
        elif target_residual_multisector_source is not None:
            recommended_next_test = target_residual_multisector_source.get("recommendedNextTest")
        elif target_residual_pixel_recurrence is not None:
            recommended_next_test = target_residual_pixel_recurrence.get("recommendedNextTest")
        elif target_residual_archival_baseline_extension is not None and not later_unrelated_science:
            # This branch is appended after v20.16. Genuinely later unrelated
            # evidence remains above it when introduced in the precedence list.
            recommended_next_test = target_residual_archival_baseline_extension.get("recommendedNextTest")
        elif deep_catalog_prf_localization is not None:
            recommended_next_test = deep_catalog_prf_localization.get(
                "recommendedNextTest"
            )
        elif deep_catalog_counterpart_identification is not None:
            recommended_next_test = deep_catalog_counterpart_identification.get(
                "recommendedNextTest"
            )
        elif (catalog_counterpart_identification is not None
                and offset_source_variability is not None):
            # This validation is appended after historical finalization, so
            # its recommendation supersedes stale pre-catalog conclusions.
            recommended_next_test = offset_source_variability.get("recommendedNextTest")
        elif targeted_observation_plan is not None:
            recommended_next_test = targeted_observation_plan.get("recommendedNextTest")
        elif atlas_fixed_window is not None:
            recommended_next_test = atlas_fixed_window.get("recommendedNextTest")
        elif atlas_time_resolved is not None:
            recommended_next_test = atlas_time_resolved.get("recommendedNextTest")
        elif atlas_forced_photometry_reanalysis is not None:
            recommended_next_test = atlas_forced_photometry_reanalysis.get("recommendedNextTest")
        elif atlas_forced_photometry is not None:
            recommended_next_test = atlas_forced_photometry.get("recommendedNextTest")
        elif des_dr2_se_local_forced_photometry is not None:
            recommended_next_test = des_dr2_se_local_forced_photometry.get("recommendedNextTest")
        elif noirlab_image_forced_photometry is not None:
            recommended_next_test = noirlab_image_forced_photometry.get("recommendedNextTest")
        elif nsc_resolved_photometry is not None:
            recommended_next_test = nsc_resolved_photometry.get("recommendedNextTest")
        elif skymapper_resolved_photometry is not None:
            recommended_next_test = skymapper_resolved_photometry.get("recommendedNextTest")
        elif external_high_resolution_validation is not None:
            recommended_next_test = external_high_resolution_validation.get("recommendedNextTest")
        elif offset_source_variability is not None:
            recommended_next_test = offset_source_variability.get("recommendedNextTest")
        elif catalog_counterpart_identification is not None:
            recommended_next_test = catalog_counterpart_identification.get("recommendedNextTest")
        elif official_spoc_prf_forward_modeling is not None:
            recommended_next_test = official_spoc_prf_forward_modeling.get("recommendedNextTest")
        elif frequency_localized_pixel_response is not None:
            recommended_next_test = frequency_localized_pixel_response.get("recommendedNextTest")
        elif difference_image_localization is not None:
            recommended_next_test = difference_image_localization.get("recommendedNextTest")
        elif calibrated_prf_deblending is not None:
            recommended_next_test = calibrated_prf_deblending.get("recommendedNextTest")
        elif offset_source_identification is not None:
            recommended_next_test = offset_source_identification.get("recommendedNextTest")
        elif target_residual_mechanism_predictive_validation is not None:
            recommended_next_test = target_residual_mechanism_predictive_validation.get(
                "recommendedNextTest"
            )
        elif multisource_residual is not None:
            recommended_next_test = multisource_residual.get("recommendedNextTest")
        elif residual_mode_localization_review is not None:
            recommended_next_test = residual_mode_localization_review.get("recommendedNextTest")
        elif residual_mode_localization is not None:
            recommended_next_test = residual_mode_localization.get("recommendedNextTest")
        elif dynamic_harmonic_frequency_refinement is not None:
            recommended_next_test = dynamic_harmonic_frequency_refinement.get("recommendedNextTest")
        elif dynamic_harmonic_modeling is not None:
            recommended_next_test = dynamic_harmonic_modeling.get("recommendedNextTest")
        elif mode_identification is not None:
            recommended_next_test = mode_identification.get("recommendedNextTest")
        elif nonstationary_modeling is not None:
            recommended_next_test = nonstationary_modeling.get("recommendedNextTest")
        elif transient_mode_validation is not None:
            recommended_next_test = transient_mode_validation.get(
                "recommendedNextTest"
            )
        elif time_frequency_evolution is not None:
            recommended_next_test = time_frequency_evolution.get("recommendedNextTest")
        elif multimode_decomposition is not None:
            recommended_next_test = multimode_decomposition.get("recommendedNextTest")
        elif source_localization is not None:
            recommended_next_test = source_localization.get("recommendedNextTest")
        else:
            recommended_next_test = (physical_interpretation or {}).get("recommendedNextTest")

        if main_family_time_domain_recurrence is not None:
            recurrence = main_family_time_domain_recurrence
            combined = recurrence.get("combinedEvidence") or {}
            family = recurrence.get("mainPhotometricFamily") or {}
            coverage = sorted(set((combined.get("rawFamilyCoverageSectorIDs") or []) +
                (combined.get("possibleDoubleCoverageSectorIDs") or [])))
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "Superseding time-domain evidence: the previously persisted "
                f"~{family.get('representativeRawPeriodDays'):.3f}-day raw family and "
                f"~{family.get('possibleDoubleCycleDays'):.3f}-day possible double were not "
                f"reproduced ({recurrence.get('classification')}) despite adequate coverage "
                f"in sectors {coverage}; recurrence sectors were raw="
                f"{combined.get('rawFamilyRecurrenceSectorIDs') or []} and double="
                f"{combined.get('possibleDoubleRecurrenceSectorIDs') or []}. Nearby recurrence "
                f"instead tracks approximately 2x/4x the solved ~{recurrence.get('authoritativeRotationPeriodDays'):.4f}-day "
                "rotation. The historical frequency family remains a candidate requiring "
                "frequency-domain reassessment, is not by itself evidence for an independent "
                "longer physical clock, and its exact physical cycle remains unresolved.")
            claim_decision = {"claim": claim_decision["claim"], "rationale": existing_rationale}
        if main_family_frequency_domain_reassessment is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "Newest superseding frequency-domain reassessment classification: "
                f"{main_family_frequency_domain_reassessment.get('classification')}. This "
                "fixed-hypothesis diagnostic preserves the negative independent time-domain "
                "evidence and cannot establish an exact physical cycle; physicalCycleResolved "
                "remains false.")
            claim_decision = {"claim": claim_decision["claim"], "rationale": existing_rationale}

        # Finalizers summarize an append-only history.  The newest completed
        # science stage carrying an explicit recommendation is authoritative;
        # handler-specific variables above only preserve report compatibility.
        # This prevents an older branch (notably v20.12) from leaking through
        # after a newer v20.13/v20.14 result superseded it.
        recommended_next_test = newest_authoritative_recommendation(
            investigation.stages[:-1], recommended_next_test)

        conclusion = {
            "investigationID": investigation.id,
            "workflowID": WORKFLOW_ID,
            "workflowVersion": WORKFLOW_VERSION,
            "target": {
                "datasetID": prepared["datasetID"],
                "ticID": prepared["ticID"],
                "targetName": prepared.get("targetName"),
                "sector": prepared.get("sector"),
            },
            "claim": claim_decision,
            "periodEvidence": period_evidence,
            "selectedPeriodDays": period_evidence.get("physicalPeriodDays"),
            "selectedSource": (
                period_evidence.get("candidateSource")
                if period_evidence.get("physicalCycleResolved")
                else None
            ),
            "primaryAnalysis": primary_analysis,
            "planner": planner,
            "followup": followup_interpretation,
            "independentPreparation": independent_prepare,
            "independentVerification": independent_interpretation,
            "independentBroadPreparation": broad_prepare,
            "independentBroadVerification": broad_interpretation,
            "independentHarmonicFamilyVerification": harmonic_family_interpretation,
            "morphology": morphology_interpretation,
            "blindTransitSearch": blind_transit_search,
            "physicalInterpretation": physical_interpretation,
            "binaryConfirmation": binary_confirmation,
            "eclipseEventSourceLocalization": eclipse_event_localization,
            "sourceAttributionReview": source_attribution_review,
            "eventDepthAttenuationAudit": event_depth_attenuation_audit,
            "jointEventPhaseModel": joint_event_phase_model,
            "externalCompanionEvidence": external_companion_evidence,
            "finalCompanionEvidenceSynthesis": final_companion_evidence_synthesis,
            "sourceLocalization": source_localization,
            "multiModeDecomposition": multimode_decomposition,
            "timeFrequencyEvolution": time_frequency_evolution,
            "longBaselineTimeFrequencyConfirmation": (
                v20_8_long_baseline_time_frequency_confirmation
            ),
            "transientModeValidation": transient_mode_validation,
            "nonstationaryModeling": nonstationary_modeling,
            "modeIdentification": mode_identification,
            "longBaselineFrequencyConfirmation": (
                long_baseline_frequency_confirmation
            ),
            "dynamicHarmonicModeling": dynamic_harmonic_modeling,
            "dynamicHarmonicFrequencyRefinement": dynamic_harmonic_frequency_refinement,
            "residualModeLocalization": residual_mode_localization,
            "residualExternalEvidence": residual_external_evidence,
            "targetResidualAstrophysicalMechanismFollowup": (
                target_residual_astrophysical_mechanism
            ),
            "residualModeLocalizationReview": residual_mode_localization_review,
            "neighborCatalogPixelResponseReview": (
                neighbor_catalog_pixel_response_review
            ),
            "multiSourceResidualDecomposition": multisource_residual,
            "targetResidualMechanismPredictiveValidation": (
                target_residual_mechanism_predictive_validation
            ),
            "targetResidualAstrophysicalInterpretation": target_residual_astrophysical_interpretation,
            "mainFamilyTimeDomainRecurrence": main_family_time_domain_recurrence,
            "mainFamilyFrequencyDomainReassessment": main_family_frequency_domain_reassessment,
            "targetResidualArchivalBaselineExtension": target_residual_archival_baseline_extension,
            "targetResidualPixelRecurrenceValidation": target_residual_pixel_recurrence,
            "targetResidualMultisectorSourceLocalization": target_residual_multisector_source,
            "offsetResidualSourceIdentification": offset_source_identification,
            "offsetSourceVariabilityValidation": offset_source_variability,
            "calibratedPrfSourceDeblending": calibrated_prf_deblending,
            "differenceImageSourceLocalization": difference_image_localization,
            "frequencyLocalizedPixelResponse": frequency_localized_pixel_response,
            "officialSpocPrfForwardModeling": official_spoc_prf_forward_modeling,
            "catalogCounterpartIdentification": catalog_counterpart_identification,
            "deepCatalogCounterpartIdentification": deep_catalog_counterpart_identification,
            "deepCatalogGuidedPrfLocalization": deep_catalog_prf_localization,
            "externalHighResolutionVariabilityValidation": external_high_resolution_validation,
            "skyMapperResolvedPhotometryScreen": skymapper_resolved_photometry,
            "nscResolvedPhotometryScreen": nsc_resolved_photometry,
            "noirlabImageForcedPhotometry": noirlab_image_forced_photometry,
            "desDr2SeLocalForcedPhotometry": des_dr2_se_local_forced_photometry,
            "atlasForcedPhotometry": atlas_forced_photometry,
            "atlasForcedPhotometryReanalysis": atlas_forced_photometry_reanalysis,
            "atlasTimeResolved": atlas_time_resolved,
            "atlasFixedWindowRecurrence": atlas_fixed_window,
            "targetedObservationPlan": targeted_observation_plan,
            "recommendedNextTest": recommended_next_test,
            "automaticDiscoveryClaim": False,
        }

        output_dir = store.directory_for(investigation.id)
        suffix = str(request.parameters.get("outputSuffix") or "").strip()
        if suffix:
            conclusion_path = output_dir / f"conclusion-{suffix}.json"
            report_path = output_dir / f"report-{suffix}.md"
        else:
            conclusion_path = output_dir / "conclusion.json"
            report_path = output_dir / "report.md"
        conclusion["conclusionPath"] = str(conclusion_path)
        conclusion["reportPath"] = str(report_path)
        _write_json(conclusion_path, conclusion)
        report_path.write_text(_render_report(conclusion), encoding="utf-8")

        final_status = (
            "HUMAN_REVIEW_REQUIRED"
            if claim_decision["claim"] == "HUMAN_REVIEW_REQUIRED"
            else "COMPLETE"
        )
        print("🏁 TESS investigation conclusion")
        print(f"   claim: {claim_decision['claim']}")
        if period_evidence.get("recurrentPhotometricPeriodDays") is not None:
            print(
                "   recurrent photometric periodicity: "
                f"{period_evidence.get('recurrentPhotometricPeriodDays')} days"
            )
        transit_candidate_periods = (
            period_evidence.get("transitLikeCandidatePeriodsDays") or []
        )
        if len(transit_candidate_periods) > 1:
            print(
                "   accepted distinct transit-like periods: "
                f"{transit_candidate_periods} days"
            )
        if period_evidence.get("possiblePhysicalCycleDays") is not None:
            print(
                "   possible physical/full cycle: "
                f"{period_evidence.get('possiblePhysicalCycleDays')} days"
            )
        if period_evidence.get("physicalCycleResolved"):
            print(f"   physical period: {period_evidence.get('physicalPeriodDays')} days")
        else:
            print("   physical period: unresolved")
        if physical_interpretation is not None:
            print(
                "   preferred photometric hypothesis: "
                f"{physical_interpretation.get('preferredPhotometricHypothesis')}"
            )
            print(
                "   physical mechanism resolved: "
                f"{physical_interpretation.get('physicalMechanismResolved')}"
            )
        if catalog_counterpart_identification is not None:
            preferred = catalog_counterpart_identification.get("preferredCandidate") or {}
            catalog_ids = preferred.get("catalogIDs") or {}
            ranking = preferred.get("rankingEvidence") or {}
            print(
                "   catalog counterpart classification: "
                f"{catalog_counterpart_identification.get('classification')}"
            )
            print(f"   counterpart TIC: {catalog_ids.get('ticID')}")
            print(f"   counterpart Gaia DR3: {catalog_ids.get('gaiaDR3SourceID')}")
            print(
                "   residual-position separation: "
                f"{ranking.get('residualPositionSeparationArcsec')} arcsec"
            )
            print(
                "   target-to-counterpart separation: "
                f"{ranking.get('targetSeparationArcsec')} arcsec"
            )
            print(
                "   variability confirmed: "
                f"{catalog_counterpart_identification.get('variabilityConfirmed')}"
            )
            print(
                "   recommended next test: "
                f"{catalog_counterpart_identification.get('recommendedNextTest')}"
            )
        if deep_catalog_counterpart_identification is not None:
            preferred = deep_catalog_counterpart_identification.get("preferredCandidate") or {}
            catalog_ids = preferred.get("catalogIDs") or {}
            ranking = preferred.get("rankingEvidence") or {}
            print(
                "   deep catalog counterpart classification: "
                f"{deep_catalog_counterpart_identification.get('classification')}"
            )
            print(f"   SkyMapper DR4 object: {catalog_ids.get('skyMapperDR4ObjectID')}")
            print(f"   NSC DR2 object: {catalog_ids.get('nscDR2ObjectID')}")
            print(
                "   residual-position separation: "
                f"{ranking.get('residualPositionSeparationArcsec')} arcsec"
            )
            print(
                "   recommended next test: "
                f"{deep_catalog_counterpart_identification.get('recommendedNextTest')}"
            )
        if deep_catalog_prf_localization is not None:
            preferred = deep_catalog_prf_localization.get("preferredCandidate") or {}
            print(
                "   deep-catalog-guided PRF localization: "
                f"{deep_catalog_prf_localization.get('classification')}"
            )
            print(
                "   localized catalog component: "
                f"{deep_catalog_prf_localization.get('stableComponentID')}"
            )
            print(
                "   localized SkyMapper DR4 object: "
                f"{(preferred.get('catalogIDs') or {}).get('skyMapperDR4ObjectID')}"
            )
            print(
                "   localized NSC DR2 object: "
                f"{(preferred.get('catalogIDs') or {}).get('nscDR2ObjectID')}"
            )
            print(
                "   recommended next test: "
                f"{deep_catalog_prf_localization.get('recommendedNextTest')}"
            )
        if source_localization is not None:
            cross = source_localization.get("crossSector") or {}
            print(f"   source localization: {cross.get('classification')}")
            print(f"   variable signal origin: {cross.get('variableSignalOrigin')}")
        if v20_8_long_baseline_time_frequency_confirmation is not None:
            print(
                "   long-baseline time-frequency confirmation: "
                f"{v20_8_long_baseline_time_frequency_confirmation.get('classification')}"
            )
            print("   physical mechanism resolved: False")
            print(
                "   recommended next test: "
                f"{v20_8_long_baseline_time_frequency_confirmation.get('recommendedNextTest')}"
            )
        if transient_mode_validation is not None:
            print(
                "   transient residual-mode validation: "
                f"{transient_mode_validation.get('classification')}"
            )
            print("   physical mechanism resolved: False")
            print(
                "   recommended next test: "
                f"{transient_mode_validation.get('recommendedNextTest')}"
            )
        if mode_identification is not None:
            print(
                "   mode identification: "
                f"{mode_identification.get('classification')}"
            )
            if mode_identification.get("pulsationInterpretation") is not None:
                print(
                    "   pulsation interpretation: "
                    f"{mode_identification.get('pulsationInterpretation')}"
                )
                print("   pulsation mechanism resolved: False")
            print(
                "   recommended next test: "
                f"{mode_identification.get('recommendedNextTest')}"
            )
        if nonstationary_modeling is not None:
            comparison = nonstationary_modeling.get("modelComparison") or {}
            print(f"   long-baseline residual model: {nonstationary_modeling.get('classification')}")
            print(f"   preferred temporal model: {comparison.get('bestModelID')}")
            print(f"   residual period at reference: {nonstationary_modeling.get('preferredPeriodAtReferenceDays')} days")
            print(f"   fractional frequency drift/day: {nonstationary_modeling.get('fractionalFrequencyDriftPerDay')}")
            if residual_mode_localization is None:
                print(f"   recommended next test: {nonstationary_modeling.get('recommendedNextTest')}")
        elif (
            v20_8_long_baseline_time_frequency_confirmation is not None
            or transient_mode_validation is not None
        ):
            pass
        elif time_frequency_evolution is not None:
            residual = time_frequency_evolution.get("residualEvolution") or {}
            family = time_frequency_evolution.get("familyEvolution") or {}
            print(f"   time-frequency structure: {time_frequency_evolution.get('classification')}")
            print(f"   residual evolution: {residual.get('classification')}")
            print(f"   established-family evolution: {family.get('classification')}")
            if physical_interpretation is None and binary_confirmation is None:
                print(f"   recommended next test: {time_frequency_evolution.get('recommendedNextTest')}")
        elif multimode_decomposition is not None:
            print(f"   residual frequency structure: {multimode_decomposition.get('classification')}")
            recurrent = multimode_decomposition.get("bestRecurrentSecondaryMode") or {}
            if recurrent:
                print(f"   recurrent secondary period: {recurrent.get('medianPeriodDays')} days")
            print(f"   recommended next test: {multimode_decomposition.get('recommendedNextTest')}")
        elif source_localization is not None:
            print(f"   recommended next test: {source_localization.get('recommendedNextTest')}")
        elif physical_interpretation is not None:
            print(
                "   recommended next test: "
                f"{physical_interpretation.get('recommendedNextTest')}"
            )
        if binary_confirmation is not None:
            binary_evidence = binary_confirmation.get("independentEvidence") or {}
            binary_ephemeris = binary_confirmation.get("linearEphemeris") or {}
            binary_opposite = binary_confirmation.get("oppositeConjunctionEvidence") or {}
            print(f"   eclipse-like replication: {binary_evidence.get('classification')}")
            print(f"   binary-confirmation ephemeris coherent: {binary_ephemeris.get('coherent')}")
            print(f"   ephemeris timing sectors: {binary_ephemeris.get('timingSectors')}")
            print(f"   primary timing epoch included: {binary_ephemeris.get('primarySectorIncluded')}")
            if joint_event_phase_model is None:
                print(f"   opposite-conjunction evidence: {binary_opposite.get('classification')}")
            else:
                print("   pre-joint binary opposite-conjunction evidence: "
                      f"{binary_opposite.get('classification')}")
        if joint_event_phase_model is not None:
            joint_fit = joint_event_phase_model.get("globalFit") or {}
            joint_gates = joint_event_phase_model.get("resolutionGates") or {}
            print("   joint model status/classification: "
                  f"{joint_event_phase_model.get('status')} / "
                  f"{joint_event_phase_model.get('classification')}")
            print("   empirical mid-transit deficit and conservative uncertainty: "
                  f"{joint_fit.get('midTransitFractionalFluxDeficit')} ± "
                  f"{joint_fit.get('conservativeTransitDepthUncertainty')}")
            print("   equivalent-box transit depth: "
                  f"{joint_fit.get('equivalentBoxTransitDepthFractionalFlux')}")
            print("   opposite-conjunction eclipse depth/status: "
                  f"{joint_fit.get('oppositeConjunctionEclipseDepthFractionalFlux')} / "
                  f"{joint_fit.get('oppositeConjunctionEclipseStatus')}")
            print("   fundamental phase status: "
                  f"{joint_fit.get('fundamentalPhaseCurveStatus')}")
            print("   second-harmonic phase status: "
                  f"{joint_fit.get('secondHarmonicPhaseCurveStatus')}")
            print("   independent supporting-sector count: "
                  f"{joint_event_phase_model.get('independentSupportingSectorCount')}")
            print("   leave-one-sector-out stable: "
                  f"{joint_gates.get('leaveOneSectorOutStable')}")
            print("   joint model unresolved reasons: "
                  f"{joint_event_phase_model.get('unresolvedReasons') or []}")
        if source_attribution_review is not None:
            print(f"   source-attribution review: {source_attribution_review.get('classification')}")
            print("   recomputed independent source support: "
                  f"{source_attribution_review.get('supportingIndependentSectorCount')}")
        if external_companion_evidence is not None:
            print(f"   external companion evidence: {external_companion_evidence.get('classification')}")
            print(f"   matched external period: {external_companion_evidence.get('externalOrbitalPeriodDays')} days")
            print(f"   external period difference: {external_companion_evidence.get('externalOrbitalPeriodDifferenceDays')} days")
            print(f"   published mass and interval: {external_companion_evidence.get('externalMassJupiter')} / {external_companion_evidence.get('externalMassIntervalJupiter')} Jupiter masses")
            print(f"   supported mass regime: {external_companion_evidence.get('supportedCompanionMassRegime')}")
            print(f"   known-object catalog used: {external_companion_evidence.get('externalKnownObjectCatalogUsed')}")
            print(f"   software-blind photometric/spatial evidence preserved: {external_companion_evidence.get('softwareBlindPhotometricEvidencePreserved')}")
        if final_companion_evidence_synthesis is not None:
            print(f"   final companion synthesis: {final_companion_evidence_synthesis.get('classification')}")
            print(f"   source relationship: {final_companion_evidence_synthesis.get('sourceRelationship')}")
            print(f"   supported mass regime: {final_companion_evidence_synthesis.get('supportedCompanionMassRegime')}")
            print("   companion nature resolved: True")
            print("   detailed photometric mechanism unresolved: True")
            print("   automatic discovery claim: False")
            print("   authoritative next test: HUMAN_SCIENTIFIC_REVIEW")
        print(f"   authoritative recommended next test: {recommended_next_test}")
        if residual_mode_localization is not None:
            residual_cross = residual_mode_localization.get("crossSector") or {}
            print(f"   residual-mode localization: {residual_cross.get('classification')}")
            print(f"   residual-mode origin: {residual_cross.get('residualModeOrigin')}")
            if (residual_mode_localization_review is None
                    and residual_external_evidence is None):
                print(f"   recommended next test: {residual_mode_localization.get('recommendedNextTest')}")
        if residual_external_evidence is not None:
            print(
                "   frozen external variability/binary evidence: "
                f"{residual_external_evidence.get('classification')}"
            )
            print(
                "   retained off-target residual sectors: "
                f"{(residual_external_evidence.get('spatialEvidence') or {}).get('offTargetSectors')}"
            )
            if target_residual_astrophysical_mechanism is None:
                print(
                    "   recommended next test: "
                    f"{residual_external_evidence.get('recommendedNextTest')}"
                )
        if target_residual_astrophysical_mechanism is not None:
            print(
                "   target-residual astrophysical mechanism: "
                f"{target_residual_astrophysical_mechanism.get('classification')}"
            )
            print(
                "   retained off-target residual sectors: "
                f"{(target_residual_astrophysical_mechanism.get('spatialEvidence') or {}).get('offTargetSectors')}"
            )
            print(
                "   recommended next test: "
                f"{target_residual_astrophysical_mechanism.get('recommendedNextTest')}"
            )
        if residual_mode_localization_review is not None:
            review_cross = residual_mode_localization_review.get("crossTime") or {}
            print(f"   time-resolved residual localization: {review_cross.get('classification')}")
            print(f"   time-resolved residual origin: {review_cross.get('residualModeOrigin')}")
            print(f"   source-switching sectors: {review_cross.get('sourceSwitchingSectors')}")
            if multisource_residual is None:
                print(f"   recommended next test: {residual_mode_localization_review.get('recommendedNextTest')}")
        if neighbor_catalog_pixel_response_review is not None:
            neighbor_decision = (
                neighbor_catalog_pixel_response_review.get("aggregateDecision")
                or {}
            )
            print(
                "   neighbor catalog/pixel-response review: "
                f"{neighbor_catalog_pixel_response_review.get('classification')}"
            )
            print(
                "   catalog-guided residual origin: "
                f"{neighbor_catalog_pixel_response_review.get('residualModeOrigin')}"
            )
            print(
                "   target-supporting sectors: "
                f"{neighbor_decision.get('targetSupportingSectors')}"
            )
            print(
                "   best neighbor/supporting sectors: "
                f"{neighbor_decision.get('bestNeighborSourceID')} / "
                f"{neighbor_decision.get('bestNeighborSupportingSectors')}"
            )
            print(
                "   recommended next test: "
                f"{neighbor_catalog_pixel_response_review.get('recommendedNextTest')}"
            )
        if multisource_residual is not None:
            print(f"   multi-source residual decomposition: {multisource_residual.get('classification')}")
            print(f"   decomposed residual origin: {multisource_residual.get('residualModeOrigin')}")
            print(f"   target residual component: {multisource_residual.get('targetComponentID')}")
            print(f"   best offset residual component: {multisource_residual.get('bestOffsetComponentID')}")
            if offset_source_identification is None:
                print(f"   recommended next test: {multisource_residual.get('recommendedNextTest')}")
        if offset_source_identification is not None:
            best = offset_source_identification.get("bestCandidate") or {}
            ids = best.get("catalogIDs") or {}
            print(f"   offset source identification: {offset_source_identification.get('classification')}")
            print(f"   offset component: {(offset_source_identification.get('component') or {}).get('componentID')}")
            print(f"   matched TIC: {ids.get('ticID')}")
            print(f"   matched Gaia DR3: {ids.get('gaiaDR3SourceID')}")
            print(f"   counterpart separation: {best.get('separationArcsec')} arcsec")
            print(f"   known variable catalog evidence: {offset_source_identification.get('knownVariableCatalogEvidence')}")
            if offset_source_variability is None:
                print(f"   recommended next test: {offset_source_identification.get('recommendedNextTest')}")
        if offset_source_variability is not None:
            counterpart = offset_source_variability.get("catalogCounterpart") or {}
            candidate = offset_source_variability.get("catalogCounterpartEvidence") or {}
            target_control = offset_source_variability.get("targetControl") or {}
            print(f"   offset counterpart variability: {offset_source_variability.get('classification')}")
            print(f"   tested TIC: {counterpart.get('ticID')}")
            print(f"   tested Gaia DR3: {counterpart.get('gaiaDR3SourceID')}")
            print(f"   counterpart independent support: {candidate.get('independentSupportCount')}")
            print(f"   counterpart combined power: {candidate.get('combinedPower')}")
            print(f"   target-control independent support: {target_control.get('independentSupportCount')}")
            print(f"   target-control combined power: {target_control.get('combinedPower')}")
            print(f"   variability confirmed: {offset_source_variability.get('variabilityConfirmed')}")
            print(f"   physical mechanism resolved: {offset_source_variability.get('physicalMechanismResolved')}")
            if calibrated_prf_deblending is None:
                print(f"   recommended next test: {offset_source_variability.get('recommendedNextTest')}")
        if calibrated_prf_deblending is not None:
            counterpart = calibrated_prf_deblending.get("catalogCounterpart") or {}
            candidate = calibrated_prf_deblending.get("catalogCounterpartEvidence") or {}
            target_control = calibrated_prf_deblending.get("targetControl") or {}
            print(f"   calibrated ePRF deblending: {calibrated_prf_deblending.get('classification')}")
            print(f"   calibrated residual origin: {calibrated_prf_deblending.get('residualModeOrigin')}")
            print(f"   deblend backend: {calibrated_prf_deblending.get('deblendBackend')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart independent support: {candidate.get('independentSupportCount')}")
            print(f"   counterpart combined power: {candidate.get('combinedPower')}")
            print(f"   target-control independent support: {target_control.get('independentSupportCount')}")
            print(f"   target-control combined power: {target_control.get('combinedPower')}")
            if difference_image_localization is None:
                print(f"   recommended next test: {calibrated_prf_deblending.get('recommendedNextTest')}")
        if difference_image_localization is not None:
            counterpart = difference_image_localization.get("catalogCounterpart") or {}
            print(f"   difference-image localization: {difference_image_localization.get('classification')}")
            print(f"   difference-image residual origin: {difference_image_localization.get('residualModeOrigin')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart-supporting sectors: {difference_image_localization.get('counterpartSupportingSectors')}")
            print(f"   target-supporting sectors: {difference_image_localization.get('targetSupportingSectors')}")
            print(f"   ambiguous sectors: {difference_image_localization.get('ambiguousSectors')}")
            if frequency_localized_pixel_response is None:
                print(f"   recommended next test: {difference_image_localization.get('recommendedNextTest')}")
        if frequency_localized_pixel_response is not None:
            counterpart = frequency_localized_pixel_response.get("catalogCounterpart") or {}
            print(f"   frequency-localized pixel response: {frequency_localized_pixel_response.get('classification')}")
            print(f"   frequency-localized residual origin: {frequency_localized_pixel_response.get('residualModeOrigin')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart-supporting sectors: {frequency_localized_pixel_response.get('counterpartSupportingSectors')}")
            print(f"   target-supporting sectors: {frequency_localized_pixel_response.get('targetSupportingSectors')}")
            print(f"   ambiguous sectors: {frequency_localized_pixel_response.get('ambiguousSectors')}")
            if official_spoc_prf_forward_modeling is None:
                print(f"   recommended next test: {frequency_localized_pixel_response.get('recommendedNextTest')}")
        if official_spoc_prf_forward_modeling is not None:
            counterpart = official_spoc_prf_forward_modeling.get("catalogCounterpart") or {}
            candidate = official_spoc_prf_forward_modeling.get("catalogCounterpartEvidence") or {}
            target_control = official_spoc_prf_forward_modeling.get("targetControl") or {}
            print(f"   official SPOC PRF forward modeling: {official_spoc_prf_forward_modeling.get('classification')}")
            print(f"   official-PRF residual origin: {official_spoc_prf_forward_modeling.get('residualModeOrigin')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart independent support: {candidate.get('independentSupportCount')}")
            print(f"   counterpart combined power: {candidate.get('combinedPower')}")
            print(f"   target-control independent support: {target_control.get('independentSupportCount')}")
            print(f"   target-control combined power: {target_control.get('combinedPower')}")
            if external_high_resolution_validation is None:
                print(f"   recommended next test: {official_spoc_prf_forward_modeling.get('recommendedNextTest')}")
        if external_high_resolution_validation is not None:
            pair = external_high_resolution_validation.get("sourcePair") or {}
            target_external = external_high_resolution_validation.get("targetControl") or {}
            counterpart_external = external_high_resolution_validation.get("catalogCounterpartEvidence") or {}
            print(f"   external high-resolution validation: {external_high_resolution_validation.get('classification')}")
            print(f"   external residual origin: {external_high_resolution_validation.get('residualModeOrigin')}")
            print(f"   archive: {external_high_resolution_validation.get('archive')}")
            print(f"   target Gaia DR3: {pair.get('targetGaiaDR3SourceID')}")
            print(f"   counterpart Gaia DR3: {pair.get('counterpartGaiaDR3SourceID')}")
            print(f"   target residual-band accepted: {target_external.get('acceptedResidualBandVariability')}")
            print(f"   counterpart residual-band accepted: {counterpart_external.get('acceptedResidualBandVariability')}")
            if skymapper_resolved_photometry is None:
                print(f"   recommended next test: {external_high_resolution_validation.get('recommendedNextTest')}")
        if skymapper_resolved_photometry is not None:
            target_skymapper = skymapper_resolved_photometry.get("targetControl") or {}
            counterpart_skymapper = skymapper_resolved_photometry.get("catalogCounterpartEvidence") or {}
            print(f"   SkyMapper DR4 resolved photometry: {skymapper_resolved_photometry.get('classification')}")
            print(f"   SkyMapper residual origin: {skymapper_resolved_photometry.get('residualModeOrigin')}")
            print(f"   pair separately resolved: {skymapper_resolved_photometry.get('pairSeparatelyResolvedInSkyMapperMaster')}")
            print(f"   target accepted bands: {target_skymapper.get('acceptedBands')}")
            print(f"   counterpart accepted bands: {counterpart_skymapper.get('acceptedBands')}")
            print(f"   recommended next test: {skymapper_resolved_photometry.get('recommendedNextTest')}")
        if nsc_resolved_photometry is not None:
            target_nsc = nsc_resolved_photometry.get("targetControl") or {}
            counterpart_nsc = nsc_resolved_photometry.get("catalogCounterpartEvidence") or {}
            print(f"   NSC DR2 resolved photometry: {nsc_resolved_photometry.get('classification')}")
            print(f"   NSC residual origin: {nsc_resolved_photometry.get('residualModeOrigin')}")
            print(f"   pair separately resolved in NSC: {nsc_resolved_photometry.get('pairSeparatelyResolvedInNSC')}")
            print(f"   target accepted bands: {target_nsc.get('acceptedBands')}")
            print(f"   counterpart accepted bands: {counterpart_nsc.get('acceptedBands')}")
            print(f"   recommended next test: {nsc_resolved_photometry.get('recommendedNextTest')}")
        if noirlab_image_forced_photometry is not None:
            target_noirlab = noirlab_image_forced_photometry.get("targetControl") or {}
            counterpart_noirlab = noirlab_image_forced_photometry.get("catalogCounterpartEvidence") or {}
            print(f"   NOIRLab image forced photometry: {noirlab_image_forced_photometry.get('classification')}")
            print(f"   NOIRLab image residual origin: {noirlab_image_forced_photometry.get('residualModeOrigin')}")
            print(f"   candidate images: {noirlab_image_forced_photometry.get('candidateExposures')}")
            print(f"   successful forced-photometry images: {noirlab_image_forced_photometry.get('successfulForcedPhotometryExposures')}")
            print(f"   target accepted bands: {target_noirlab.get('acceptedBands')}")
            print(f"   counterpart accepted bands: {counterpart_noirlab.get('acceptedBands')}")
            print(f"   recommended next test: {noirlab_image_forced_photometry.get('recommendedNextTest')}")
        if des_dr2_se_local_forced_photometry is not None:
            target_des = des_dr2_se_local_forced_photometry.get("targetControl") or {}
            counterpart_des = des_dr2_se_local_forced_photometry.get("catalogCounterpartEvidence") or {}
            print(f"   DES DR2 source-local forced photometry: {des_dr2_se_local_forced_photometry.get('classification')}")
            print(f"   DES residual origin: {des_dr2_se_local_forced_photometry.get('residualModeOrigin')}")
            print(f"   actual Gaia pair separation: {des_dr2_se_local_forced_photometry.get('pairSeparationArcsec')} arcsec")
            print(f"   candidate images: {des_dr2_se_local_forced_photometry.get('candidateExposures')}")
            print(f"   source successes: {des_dr2_se_local_forced_photometry.get('sourceSuccesses')}")
            print(f"   target accepted bands: {target_des.get('acceptedBands')}")
            print(f"   counterpart accepted bands: {counterpart_des.get('acceptedBands')}")
            print(f"   recommended next test: {des_dr2_se_local_forced_photometry.get('recommendedNextTest')}")
        if atlas_forced_photometry is not None:
            target_atlas = atlas_forced_photometry.get("targetControl") or {}
            counterpart_atlas = atlas_forced_photometry.get("catalogCounterpartEvidence") or {}
            print(f"   ATLAS forced photometry: {atlas_forced_photometry.get('classification')}")
            print(f"   ATLAS residual origin: {atlas_forced_photometry.get('residualModeOrigin')}")
            print(f"   corrected Gaia source separation: {atlas_forced_photometry.get('gaiaPairSeparationArcsec')} arcsec")
            print(f"   target accepted bands: {target_atlas.get('acceptedBands')}")
            print(f"   counterpart accepted bands: {counterpart_atlas.get('acceptedBands')}")
            print(f"   recommended next test: {atlas_forced_photometry.get('recommendedNextTest')}")
        if atlas_forced_photometry_reanalysis is not None:
            target_atlas_re = atlas_forced_photometry_reanalysis.get("targetControl") or {}
            counterpart_atlas_re = atlas_forced_photometry_reanalysis.get("catalogCounterpartEvidence") or {}
            print(f"   ATLAS signed-flux reanalysis: {atlas_forced_photometry_reanalysis.get('classification')}")
            print(f"   ATLAS reanalysis residual origin: {atlas_forced_photometry_reanalysis.get('residualModeOrigin')}")
            print(f"   target accepted bands: {target_atlas_re.get('acceptedBands')}")
            print(f"   counterpart accepted bands: {counterpart_atlas_re.get('acceptedBands')}")
            print(f"   recommended next test: {atlas_forced_photometry_reanalysis.get('recommendedNextTest')}")
        if atlas_time_resolved is not None:
            counterpart_atlas_time = atlas_time_resolved.get("catalogCounterpartEvidence") or {}
            print(f"   ATLAS time-resolved recurrence: {atlas_time_resolved.get('classification')}")
            print(f"   ATLAS time-resolved residual origin: {atlas_time_resolved.get('residualModeOrigin')}")
            print(f"   accepted seasons: {counterpart_atlas_time.get('acceptedSeasons')}")
            print(f"   accepted bands: {counterpart_atlas_time.get('acceptedBands')}")
            print(f"   cross-band-consistent seasons: {counterpart_atlas_time.get('crossBandConsistentSeasons')}")
            print(f"   counterpart supported: {counterpart_atlas_time.get('sourceSupported')}")
            print(f"   recommended next test: {atlas_time_resolved.get('recommendedNextTest')}")
        if atlas_fixed_window is not None:
            counterpart_atlas_fixed = (
                atlas_fixed_window.get(
                    "catalogCounterpartEvidence"
                )
                or {}
            )
            print(
                "   ATLAS fixed-window recurrence: "
                f"{atlas_fixed_window.get('classification')}"
            )
            print(
                "   ATLAS fixed-window residual origin: "
                f"{atlas_fixed_window.get('residualModeOrigin')}"
            )
            print(
                "   accepted windows: "
                f"{counterpart_atlas_fixed.get('acceptedWindows')}"
            )
            print(
                "   accepted bands: "
                f"{counterpart_atlas_fixed.get('acceptedBands')}"
            )
            print(
                "   cross-band-consistent windows: "
                f"{counterpart_atlas_fixed.get('crossBandConsistentWindows')}"
            )
            print(
                "   counterpart supported: "
                f"{counterpart_atlas_fixed.get('sourceSupported')}"
            )
            print(
                "   recommended next test: "
                f"{atlas_fixed_window.get('recommendedNextTest')}"
            )
        if targeted_observation_plan is not None:
            geometry_plan = (
                targeted_observation_plan.get("sourceGeometry") or {}
            )
            cadence_plan = (
                targeted_observation_plan.get("cadence") or {}
            )
            print(
                "   targeted observation plan: "
                f"{targeted_observation_plan.get('status')}"
            )
            print(
                "   Gaia source separation: "
                f"{geometry_plan.get('separationArcsec')} arcsec"
            )
            print(
                "   minimum/preferred baseline: "
                f"{cadence_plan.get('minimumBaselineDays')} / "
                f"{cadence_plan.get('preferredBaselineDays')} days"
            )
            print(
                "   minimum/preferred nights: "
                f"{cadence_plan.get('minimumDistinctNights')} / "
                f"{cadence_plan.get('preferredDistinctNights')}"
            )
            print(
                "   recommended next test: "
                f"{targeted_observation_plan.get('recommendedNextTest')}"
            )
        print(f"   report: {report_path}")

        project_ids = tuple(
            str(stage.provenance.project_ids[0])
            for stage in investigation.stages
            if stage.provenance is not None and stage.provenance.project_ids
        )
        return StageOutcome(
            result=conclusion,
            stop=True,
            final_status=final_status,
            input_hashes={
                "primaryAnalysis": sha256_json(primary_analysis),
                "planner": sha256_json(planner),
            },
            project_ids=project_ids,
            artifacts=(
                _artifact(conclusion_path, "application/json"),
                _artifact(report_path, "text/markdown"),
            ),
        )

    engine.register_handler("openstar.tess.prepare-target", prepare_target)
    engine.register_handler("openstar.tess.primary-project.run", run_primary)
    engine.register_handler("openstar.tess.catalog-identity", identity_stage)
    engine.register_handler("openstar.tess.hypotheses", hypothesis_stage)
    engine.register_handler("openstar.tess.planner", planner_stage)
    engine.register_handler("openstar.tess.followup.prepare-low-frequency", prepare_followup)
    engine.register_handler("openstar.tess.followup.run", run_followup)
    engine.register_handler("openstar.tess.followup.interpret", interpret_stage)
    engine.register_handler("openstar.tess.independent.prepare", prepare_independent)
    engine.register_handler("openstar.tess.independent.run", run_independent)
    engine.register_handler("openstar.tess.independent.interpret", interpret_independent)
    engine.register_handler("openstar.tess.independent.broad.prepare", prepare_broad_independent)
    engine.register_handler("openstar.tess.independent.broad.run", run_broad_independent)
    engine.register_handler("openstar.tess.independent.broad.interpret", interpret_broad_independent)
    engine.register_handler(
        "openstar.tess.independent.harmonic-family.interpret",
        reinterpret_harmonic_family,
    )
    engine.register_handler(
        "openstar.tess.morphology.analyze",
        morphology_stage,
    )
    engine.register_handler(BLIND_TRANSIT_SEARCH_HANDLER_ID, blind_transit_search_stage)
    engine.register_handler(
        EXHAUSTED_RESIDUAL_RUN_HANDLER_ID,
        exhausted_residual_candidate_run_stage,
    )
    engine.register_handler(
        EXHAUSTED_RESIDUAL_INTERPRET_HANDLER_ID,
        exhausted_residual_candidate_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.physical.interpret",
        physical_interpretation_stage,
    )
    engine.register_handler(
        "openstar.tess.binary-confirmation.analyze",
        binary_confirmation_stage,
    )
    engine.register_handler(ECLIPSE_LOCALIZATION_PREPARE_HANDLER_ID,
                            eclipse_event_localization_prepare_stage)
    engine.register_handler(ECLIPSE_LOCALIZATION_HANDLER_ID, eclipse_event_localization_stage)
    engine.register_handler(SOURCE_ATTRIBUTION_REVIEW_HANDLER_ID, source_attribution_review_stage)
    engine.register_handler(EVENT_DEPTH_FREEZE_HANDLER_ID, event_depth_photometry_freeze_stage)
    engine.register_handler(EVENT_DEPTH_AUDIT_HANDLER_ID, event_depth_attenuation_audit_stage)
    engine.register_handler(JOINT_EVENT_PHASE_MODEL_HANDLER_ID, joint_event_phase_model_stage)
    engine.register_handler(EXTERNAL_EVIDENCE_FREEZE_HANDLER_ID, external_evidence_freeze_stage)
    engine.register_handler(EXTERNAL_EVIDENCE_INTERPRET_HANDLER_ID, external_evidence_interpret_stage)
    engine.register_handler(COMPANION_SYNTHESIS_HANDLER_ID, companion_evidence_synthesis_stage)
    engine.register_handler(
        "openstar.tess.source-localization.analyze",
        source_localization_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.prepare",
        multimode_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.run",
        multimode_run_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.interpret",
        multimode_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.summarize",
        multimode_summary_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.prepare",
        time_frequency_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.run",
        time_frequency_run_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.interpret",
        time_frequency_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.summarize",
        time_frequency_summary_stage,
    )
    engine.register_handler(
        V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID,
        v20_8_long_baseline_time_frequency_confirmation_stage,
    )
    engine.register_handler(
        TRANSIENT_MODE_VALIDATION_HANDLER_ID,
        transient_mode_validation_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.prepare",
        nonstationary_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.mode-identification.analyze", mode_identification_stage,
    )
    engine.register_handler(
        LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID,
        long_baseline_frequency_confirmation_stage,
    )
    engine.register_handler(
        "openstar.tess.dynamic-harmonic.analyze", dynamic_harmonic_stage,
    )
    engine.register_handler(
        "openstar.tess.dynamic-harmonic.frequency-refinement",
        dynamic_harmonic_frequency_refinement_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.run",
        nonstationary_run_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.interpret",
        nonstationary_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.summarize",
        nonstationary_summary_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization.prepare",
        residual_mode_localization_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization.run",
        residual_mode_localization_run_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization.interpret",
        residual_mode_localization_interpret_stage,
    )
    engine.register_handler(
        RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID,
        residual_external_evidence_stage,
    )
    engine.register_handler(
        TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID,
        target_residual_astrophysical_mechanism_stage,
    )
    engine.register_handler(
        NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID,
        neighbor_catalog_pixel_response_review_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization-review.prepare",
        residual_mode_localization_review_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization-review.run",
        residual_mode_localization_review_run_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization-review.interpret",
        residual_mode_localization_review_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.multi-source-residual.prepare",
        multisource_residual_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.multi-source-residual.run",
        multisource_residual_run_stage,
    )
    engine.register_handler(
        "openstar.tess.multi-source-residual.interpret",
        multisource_residual_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.intrinsic-nonstationary.analyze",
        intrinsic_nonstationary_stage,
    )
    engine.register_handler(
        "openstar.tess.target-residual-mechanism.analyze",
        target_residual_mechanism_stage,
    )
    engine.register_handler(
        "openstar.tess.target-residual-mechanism-adjudication.analyze",
        target_residual_mechanism_adjudication_stage,
    )
    engine.register_handler(
        "openstar.tess.target-residual-mechanism-predictive-validation.analyze",
        target_residual_mechanism_predictive_validation_stage,
    )
    engine.register_handler("openstar.tess.target-residual-archival-baseline.prepare", archival_baseline_prepare_stage)
    engine.register_handler("openstar.tess.target-residual-archival-baseline.run", archival_baseline_run_stage)
    engine.register_handler("openstar.tess.target-residual-archival-baseline.interpret", archival_baseline_interpret_stage)
    engine.register_handler("openstar.tess.target-residual-pixel-recurrence.prepare", pixel_recurrence_prepare_stage)
    engine.register_handler("openstar.tess.target-residual-pixel-recurrence.run", pixel_recurrence_run_stage)
    engine.register_handler("openstar.tess.target-residual-pixel-recurrence.interpret", pixel_recurrence_interpret_stage)
    engine.register_handler("openstar.tess.target-residual-multisector-source.prepare", multisector_source_prepare_stage)
    engine.register_handler("openstar.tess.target-residual-multisector-source.run", multisector_source_run_stage)
    engine.register_handler("openstar.tess.target-residual-multisector-source.interpret", multisector_source_interpret_stage)
    engine.register_handler(
        "openstar.tess.offset-source-identification.analyze",
        offset_source_identification_stage,
    )
    engine.register_handler(
        "openstar.tess.catalog-counterpart-identification.analyze",
        catalog_counterpart_identification_stage,
    )
    engine.register_handler(
        DEEP_CATALOG_COUNTERPART_HANDLER_ID,
        deep_catalog_counterpart_identification_stage,
    )
    engine.register_handler(
        DEEP_CATALOG_PRF_PREPARE_HANDLER_ID,
        deep_catalog_prf_localization_prepare_stage,
    )
    engine.register_handler(
        DEEP_CATALOG_PRF_RUN_HANDLER_ID,
        deep_catalog_prf_localization_run_stage,
    )
    engine.register_handler(
        DEEP_CATALOG_PRF_INTERPRET_HANDLER_ID,
        deep_catalog_prf_localization_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.catalog-guided-source-localization.prepare",
        catalog_guided_localization_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.catalog-guided-source-localization.run",
        catalog_guided_localization_run_stage,
    )
    engine.register_handler(
        "openstar.tess.catalog-guided-source-localization.interpret",
        catalog_guided_localization_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-phase-difference-imaging.prepare",
        residual_phase_difference_image_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-phase-difference-imaging.run",
        residual_phase_difference_image_run_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-phase-difference-imaging.interpret",
        residual_phase_difference_image_interpret_stage,
    )
    engine.register_handler(
        PERIOD_FAMILY_DIFFERENCE_PREPARE_HANDLER,
        period_family_difference_image_prepare_stage,
    )
    engine.register_handler(
        PERIOD_FAMILY_DIFFERENCE_RUN_HANDLER,
        period_family_difference_image_run_stage,
    )
    engine.register_handler(
        PERIOD_FAMILY_DIFFERENCE_INTERPRET_HANDLER,
        period_family_difference_image_interpret_stage,
    )
    engine.register_handler(
        PERIOD_FAMILY_TIME_DOMAIN_PREPARE_HANDLER,
        period_family_time_domain_prepare_stage,
    )
    engine.register_handler(
        PERIOD_FAMILY_TIME_DOMAIN_RUN_HANDLER,
        period_family_time_domain_run_stage,
    )
    engine.register_handler(
        PERIOD_FAMILY_TIME_DOMAIN_INTERPRET_HANDLER,
        period_family_time_domain_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.source-switching-temporal-model.prepare",
        source_switching_temporal_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.source-switching-temporal-model.run",
        source_switching_temporal_run_stage,
    )
    engine.register_handler(
        "openstar.tess.source-switching-temporal-model.interpret",
        source_switching_temporal_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.time-resolved-residual-phase-localization.prepare",
        time_resolved_residual_phase_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.time-resolved-residual-phase-localization.run",
        time_resolved_residual_phase_run_stage,
    )
    engine.register_handler(
        "openstar.tess.time-resolved-residual-phase-localization.interpret",
        time_resolved_residual_phase_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.time-resolved-frequency-localization.prepare",
        time_resolved_frequency_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.time-resolved-frequency-localization.run",
        time_resolved_frequency_run_stage,
    )
    engine.register_handler(
        "openstar.tess.time-resolved-frequency-localization.interpret",
        time_resolved_frequency_interpret_stage,
    )
    engine.register_handler("openstar.tess.additional-sector-source-localization.prepare",
                            additional_sector_prepare_stage)
    engine.register_handler("openstar.tess.additional-sector-source-localization.run",
                            additional_sector_run_stage)
    engine.register_handler("openstar.tess.additional-sector-source-localization.interpret",
                            additional_sector_interpret_stage)
    engine.register_handler(
        "openstar.tess.offset-source-variability.prepare",
        offset_source_variability_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.offset-source-variability.run",
        offset_source_variability_run_stage,
    )
    engine.register_handler(
        "openstar.tess.offset-source-variability.interpret",
        offset_source_variability_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare",
        current_gaia_counterpart_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.gaia-source-resolved-counterpart-photometry.run",
        current_gaia_counterpart_run_stage,
    )
    engine.register_handler(
        "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
        current_gaia_counterpart_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
        skymapper_counterpart_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.skymapper-resolved-counterpart-photometry.run",
        skymapper_counterpart_run_stage,
    )
    engine.register_handler(
        "openstar.tess.skymapper-resolved-counterpart-photometry.interpret",
        skymapper_counterpart_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.calibrated-prf-deblending.prepare",
        calibrated_prf_deblending_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.calibrated-prf-deblending.run",
        calibrated_prf_deblending_run_stage,
    )
    engine.register_handler(
        "openstar.tess.calibrated-prf-deblending.interpret",
        calibrated_prf_deblending_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.difference-image-localization.prepare",
        difference_image_localization_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.difference-image-localization.run",
        difference_image_localization_run_stage,
    )
    engine.register_handler(
        "openstar.tess.difference-image-localization.interpret",
        difference_image_localization_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.frequency-localized-pixel-response.prepare",
        frequency_localized_pixel_response_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.frequency-localized-pixel-response.run",
        frequency_localized_pixel_response_run_stage,
    )
    engine.register_handler(
        "openstar.tess.frequency-localized-pixel-response.interpret",
        frequency_localized_pixel_response_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.official-spoc-prf-forward-modeling.prepare",
        official_spoc_prf_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.official-spoc-prf-forward-modeling.run",
        official_spoc_prf_run_stage,
    )
    engine.register_handler(
        "openstar.tess.official-spoc-prf-forward-modeling.interpret",
        official_spoc_prf_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.external-high-resolution-variability-validation.prepare",
        external_high_resolution_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.external-high-resolution-variability-validation.run",
        external_high_resolution_run_stage,
    )
    engine.register_handler(
        "openstar.tess.external-high-resolution-variability-validation.interpret",
        external_high_resolution_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.skymapper-resolved-photometry.prepare",
        skymapper_resolved_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.skymapper-resolved-photometry.run",
        skymapper_resolved_run_stage,
    )
    engine.register_handler(
        "openstar.tess.skymapper-resolved-photometry.interpret",
        skymapper_resolved_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.nsc-resolved-photometry.prepare",
        nsc_resolved_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.nsc-resolved-photometry.run",
        nsc_resolved_run_stage,
    )
    engine.register_handler(
        "openstar.tess.nsc-resolved-photometry.interpret",
        nsc_resolved_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.noirlab-image-forced-photometry.prepare",
        noirlab_forced_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.noirlab-image-forced-photometry.run",
        noirlab_forced_run_stage,
    )
    engine.register_handler(
        "openstar.tess.noirlab-image-forced-photometry.interpret",
        noirlab_forced_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.des-dr2-se-local-forced-photometry.prepare",
        des_dr2_se_local_forced_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.des-dr2-se-local-forced-photometry.run",
        des_dr2_se_local_forced_run_stage,
    )
    engine.register_handler(
        "openstar.tess.des-dr2-se-local-forced-photometry.interpret",
        des_dr2_se_local_forced_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-forced-photometry.prepare",
        atlas_forced_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-forced-photometry.collect",
        atlas_forced_collect_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-forced-photometry.run",
        atlas_forced_run_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-forced-photometry.interpret",
        atlas_forced_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-forced-photometry-reanalysis.prepare",
        atlas_reanalysis_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-forced-photometry-reanalysis.run",
        atlas_reanalysis_run_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-forced-photometry-reanalysis.interpret",
        atlas_reanalysis_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-time-resolved.prepare",
        atlas_time_resolved_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-time-resolved.run",
        atlas_time_resolved_run_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-time-resolved.interpret",
        atlas_time_resolved_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-fixed-window.prepare",
        atlas_fixed_window_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-fixed-window.run",
        atlas_fixed_window_run_stage,
    )
    engine.register_handler(
        "openstar.tess.atlas-fixed-window.interpret",
        atlas_fixed_window_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.targeted-observation-planning.generate",
        targeted_observation_plan_stage,
    )
    engine.register_handler(
        "openstar.tess.period-semantics.reinterpret",
        period_semantics_stage,
    )
    engine.register_handler("openstar.tess.finalize", finalize_stage)
    engine.register_handler(
        "openstar.tess.target-residual-astrophysical-interpretation.analyze",
        target_residual_astrophysical_interpretation_stage,
    )
    engine.register_handler("openstar.tess.main-family-time-domain-recurrence.analyze",
        main_family_time_domain_recurrence_stage)
    engine.register_handler("openstar.tess.main-family-frequency-domain-reassessment.analyze",
        main_family_frequency_domain_reassessment_stage)
    # All TESS handlers share this provider-to-workflow adapter.  Localization
    # builders call _download_tpf indirectly, so a centralized boundary also
    # protects new experiments from persisting MAST outages as NON_RETRYABLE.
    for handler_id, handler in tuple(engine.handlers.items()):
        if handler_id.startswith("openstar.tess."):
            engine.handlers[handler_id] = _retry_transient_tess_archive_failures(handler)
    return engine
