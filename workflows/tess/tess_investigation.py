Warning: truncated output (original token count: 136379)
Total output lines: 10191

from __future__ import annotations

import copy

import json
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
from .tess_physical import analyze_physical_interpretation
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
    build_nonstationary_project,
    interpret_nonstationary_project,
    summarize_nonstationary_modeling,
)
from .tess_mode_identification import identify_residual_mode
from .tess_dynamic_harmonic import model_dynamic_harmonics, refine_harmonic_family_frequency
from .tess_residual_localization import (
    build_residual_mode_pixel_project,
    interpret_residual_mode_pixel_project,
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
SOFTWARE_VERSION = "20.36"


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


def time_frequency_continuation(summary: dict[str, Any], *, request_id: str) -> StageRequest:
    """Continue only the explicitly recommended, still-unresolved experiment."""

    run_mode_identification = (
        summary.get("recommendedNextTest") == "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
        and summary.get("physicalMechanismResolved") is False
    )
    run_nonstationary = (
        summary.get("recommendedNextTest")
        == "LONG_BASELINE_NONSTATIONARY_MODE_MODELING"
        and summary.get("physicalMechanismResolved") is False
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "mode-identification" if run_mode_identification else
            ("prepare-nonstationary" if run_nonstationary else "finalize"),
        ),
        handler_id=(
            "openstar.tess.mode-identification.analyze" if run_mode_identification else
            ("openstar.tess.nonstationary.prepare"
            if run_nonstationary
            else "openstar.tess.finalize"
            )
        ),
        parameters={} if (run_nonstationary or run_mode_identification) else {"outputSuffix": "v20.8"},
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
    return StageRequest(
        id=_next_stage_id(request_id, "refine-harmonic-frequency" if refine else
                          ("prepare-time-frequency" if residual else "finalize")),
        handler_id=("openstar.tess.dynamic-harmonic.frequency-refinement" if refine else
                    ("openstar.tess.time-frequency.prepare" if residual else "openstar.tess.finalize")),
        parameters=({} if refine else
                    ({"entryReason": "DYNAMIC_HARMONIC_RESIDUAL"} if residual else
                     {"outputSuffix": "v20.10-dynamic-harmonic"})),
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
    """Route only the persisted, unresolved source-localization review request."""

    run_review = (
        summary.get("recommendedNextTest")
        == "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"
        and summary.get("physicalMechanismResolved") is False
    )
    return StageRequest(
        id=_next_stage_id(
            request_id,
            "prepare-residual-mode-localization-review" if run_review else "finalize",
        ),
        handler_id=(
            "openstar.tess.residual-mode-localization-review.prepare"
            if run_review
            else "openstar.tess.finalize"
        ),
        parameters={} if run_review else {"outputSuffix": "v20.10"},
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
            f"- Established period family: {mode_identification.get('establishedPeriodFamily')}",
            f"- Residual candidate period/frequency: {candidate.get('refinedPeriodDays')} days / {candidate.get('refinedFrequencyCyclesPerDay')} cycles/day",
            f"- Tested harmonic relation: order {relation.get('testedOrder')}, commensurate within measured resolution={relation.get('commensurateWithinResolution')}",
            f"- BIC model comparison: {comparison}",
            f"- Independent-sector support: {mode_identification.get('independentSectorSupport')}",
            f"- Classification: {mode_identification.get('classification')}",
            f"- Independent-mode evidence survived: {mode_identification.get('independentModeEvidenceSurvived')}",
            f"- Physical mechanism resolved: {mode_identification.get('physicalMechanismResolved')}",
            f"- Recommended next test: {mode_identification.get('recommendedNextTest')}",
        ])

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
        analysis = analyze(
            primary,
            identity,
            observation_baseline_days=prepared.get("observationBaselineDays"),
            primary_minimum_frequency=(
                (_load_json(Path(prepared["datasetPath"])).get("frequencySearch") or {}).get(
                    "minimumFrequency"
                )
            ),
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
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        analysis = _latest_result_for_handler(investigation, "openstar.tess.hypotheses")
        if identity is None or analysis is None:
            raise RuntimeError("Planner requires completed identity and hypotheses.")
        planned = plan(analysis, identity)
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
            investigation, "openstar.tess.hyp…86379 tokens truncated… claim_decision["claim"],
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
        if target_residual_astrophysical_interpretation is not None:
            recommended_next_test = target_residual_astrophysical_interpretation.get("recommendedNextTest")
        elif target_residual_multisector_source is not None:
            recommended_next_test = target_residual_multisector_source.get("recommendedNextTest")
        elif target_residual_pixel_recurrence is not None:
            recommended_next_test = target_residual_pixel_recurrence.get("recommendedNextTest")
        elif target_residual_archival_baseline_extension is not None and not later_unrelated_science:
            # This branch is appended after v20.16. Genuinely later unrelated
            # evidence remains above it when introduced in the precedence list.
            recommended_next_test = target_residual_archival_baseline_extension.get("recommendedNextTest")
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
        elif time_frequency_evolution is not None:
            recommended_next_test = time_frequency_evolution.get("recommendedNextTest")
        elif multimode_decomposition is not None:
            recommended_next_test = multimode_decomposition.get("recommendedNextTest")
        elif source_localization is not None:
            recommended_next_test = source_localization.get("recommendedNextTest")
        else:
            recommended_next_test = (physical_interpretation or {}).get("recommendedNextTest")

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
            "physicalInterpretation": physical_interpretation,
            "sourceLocalization": source_localization,
            "multiModeDecomposition": multimode_decomposition,
            "timeFrequencyEvolution": time_frequency_evolution,
            "nonstationaryModeling": nonstationary_modeling,
            "modeIdentification": mode_identification,
            "dynamicHarmonicModeling": dynamic_harmonic_modeling,
            "dynamicHarmonicFrequencyRefinement": dynamic_harmonic_frequency_refinement,
            "residualModeLocalization": residual_mode_localization,
            "residualModeLocalizationReview": residual_mode_localization_review,
            "multiSourceResidualDecomposition": multisource_residual,
            "targetResidualMechanismPredictiveValidation": (
                target_residual_mechanism_predictive_validation
            ),
            "targetResidualAstrophysicalInterpretation": target_residual_astrophysical_interpretation,
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
        if source_localization is not None:
            cross = source_localization.get("crossSector") or {}
            print(f"   source localization: {cross.get('classification')}")
            print(f"   variable signal origin: {cross.get('variableSignalOrigin')}")
        if nonstationary_modeling is not None:
            comparison = nonstationary_modeling.get("modelComparison") or {}
            print(f"   long-baseline residual model: {nonstationary_modeling.get('classification')}")
            print(f"   preferred temporal model: {comparison.get('bestModelID')}")
            print(f"   residual period at reference: {nonstationary_modeling.get('preferredPeriodAtReferenceDays')} days")
            print(f"   fractional frequency drift/day: {nonstationary_modeling.get('fractionalFrequencyDriftPerDay')}")
            if residual_mode_localization is None:
                print(f"   recommended next test: {nonstationary_modeling.get('recommendedNextTest')}")
        elif time_frequency_evolution is not None:
            residual = time_frequency_evolution.get("residualEvolution") or {}
            family = time_frequency_evolution.get("familyEvolution") or {}
            print(f"   time-frequency structure: {time_frequency_evolution.get('classification')}")
            print(f"   residual evolution: {residual.get('classification')}")
            print(f"   established-family evolution: {family.get('classification')}")
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
        if residual_mode_localization is not None:
            residual_cross = residual_mode_localization.get("crossSector") or {}
            print(f"   residual-mode localization: {residual_cross.get('classification')}")
            print(f"   residual-mode origin: {residual_cross.get('residualModeOrigin')}")
            if residual_mode_localization_review is None:
                print(f"   recommended next test: {residual_mode_localization.get('recommendedNextTest')}")
        if residual_mode_localization_review is not None:
            review_cross = residual_mode_localization_review.get("crossTime") or {}
            print(f"   time-resolved residual localization: {review_cross.get('classification')}")
            print(f"   time-resolved residual origin: {review_cross.get('residualModeOrigin')}")
            print(f"   source-switching sectors: {review_cross.get('sourceSwitchingSectors')}")
            if multisource_residual is None:
                print(f"   recommended next test: {residual_mode_localization_review.get('recommendedNextTest')}")
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
    engine.register_handler(
        "openstar.tess.physical.interpret",
        physical_interpretation_stage,
    )
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
        "openstar.tess.nonstationary.prepare",
        nonstationary_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.mode-identification.analyze", mode_identification_stage,
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
    # All TESS handlers share this provider-to-workflow adapter.  Localization
    # builders call _download_tpf indirectly, so a centralized boundary also
    # protects new experiments from persisting MAST outages as NON_RETRYABLE.
    for handler_id, handler in tuple(engine.handlers.items()):
        if handler_id.startswith("openstar.tess."):
            engine.handlers[handler_id] = _retry_transient_tess_archive_failures(handler)
    return engine
