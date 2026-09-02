from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import InvestigationStore, sha256_file, sha256_json
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import (
    SOFTWARE_ID,
    SOFTWARE_VERSION,
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    build_engine,
)
from workflows.tess.tess_long_baseline_frequency_confirmation import (
    HANDLER_ID as LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID,
    build_method_contract as build_long_baseline_frequency_confirmation_contract,
    validate_ambiguous_mode_identification,
    validate_frozen_dataset_lineage,
)
from workflows.tess.tess_v20_8_long_baseline_time_frequency_confirmation import (
    HANDLER_ID as V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID,
    build_dataset_specs as build_v20_8_long_baseline_dataset_specs,
    build_method_contract as build_v20_8_long_baseline_method_contract,
    method_contract_hash as v20_8_confirmation_method_contract_hash,
    validate_frozen_window_lineage as validate_v20_8_frozen_window_lineage,
)
from workflows.tess.tess_transient_mode_validation import (
    HANDLER_ID as TRANSIENT_MODE_VALIDATION_HANDLER_ID,
    build_dataset_specs as build_transient_mode_dataset_specs,
    build_method_contract as build_transient_mode_method_contract,
    validate_frozen_dataset_lineage as validate_transient_mode_frozen_lineage,
)
from workflows.tess.tess_recurrent_residual_long_baseline_confirmation import (
    build_dataset_specs as build_recurrent_residual_dataset_specs,
    build_method_contract as build_recurrent_residual_method_contract,
)
from workflows.tess.tess_mode_identification import (
    MULTIMODE_MODE_EVIDENCE_LINEAGE,
    V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE,
    build_confirmed_coherent_mode_method_contract,
    validate_confirmed_coherent_mode_dataset_lineage,
    validate_v20_8_confirmed_coherent_residual,
)
from workflows.tess.tess_nonstationary import (
    CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE,
    RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE,
    build_confirmed_nonstationary_method_contract,
    build_recurrent_residual_nonstationary_method_contract,
    validate_confirmed_nonstationary_localization_boundary,
    validate_recurrent_residual_nonstationary_boundary,
)
from workflows.tess.tess_residual_external_evidence import (
    HANDLER_ID as RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID,
    validate_target_supported_boundary,
)
from workflows.tess.tess_target_residual_astrophysical_mechanism import (
    HANDLER_ID as TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID,
    validate_mechanism_followup_boundary,
)
from workflows.tess.tess_neighbor_catalog_pixel_response_review import (
    HANDLER_ID as NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID,
    validate_review_boundary as validate_neighbor_catalog_pixel_response_boundary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenStar v20.20 deterministic TESS investigation plugin. "
            "It derives a single-target project from an existing frozen project, "
            "runs distributed compute, resolves catalogs, evaluates hypotheses, "
            "and conditionally launches same-sector and independent multi-sector follow-ups."
        )
    )
    parser.add_argument("--project", required=True, help="Existing frozen OpenStar project manifest.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--dataset-id")
    selector.add_argument("--tic", type=int)
    parser.add_argument("--investigation-id", required=True)
    parser.add_argument("--coordinator", default="http://127.0.0.1:8080")
    parser.add_argument("--store", default="data/investigations")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=None)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the current interrupted RUNNING stage of an existing "
            "investigation instead of creating a new investigation."
        ),
    )
    recovery.add_argument(
        "--continue-contradiction",
        action="store_true",
        help=(
            "Reopen a terminal HUMAN_REVIEW_REQUIRED investigation whose "
            "targeted independent-sector check has already completed, and "
            "append the v20.3 target-independent broad-search stages."
        ),
    )
    recovery.add_argument(
        "--continue-harmonic-family",
        action="store_true",
        help=(
            "Reopen a terminal investigation with completed v20.3 broad "
            "independent-sector compute and append the v20.3.1 stricter "
            "harmonic-family reinterpretation without rerunning compute."
        ),
    )
    recovery.add_argument(
        "--continue-period-semantics",
        action="store_true",
        help=(
            "Append the v20.3.3 period-evidence semantic correction to an "
            "existing harmonic-family investigation without rerunning compute."
        ),
    )
    recovery.add_argument(
        "--continue-morphology",
        action="store_true",
        help=(
            "Append v20.4 local folded-light-curve morphology and physical-cycle "
            "discrimination using already-frozen primary and independent sectors."
        ),
    )
    recovery.add_argument(
        "--continue-physical-interpretation",
        action="store_true",
        help=(
            "Append v20.5 local physical-mechanism discrimination using the already-frozen "
            "sector light curves, resolved morphology, and existing identity metadata."
        ),
    )
    recovery.add_argument(
        "--continue-source-localization",
        action="store_true",
        help=(
            "Append v20.6 TESS pixel-level periodic-source localization. This uses MAST "
            "target-pixel products or TESScut fallback data, but no distributed workers."
        ),
    )
    recovery.add_argument(
        "--continue-multimode",
        action="store_true",
        help=(
            "Append v20.7 distributed iterative residual multi-mode decomposition. "
            "Requires the coordinator and compatible distributed workers."
        ),
    )
    recovery.add_argument(
        "--continue-time-frequency",
        action="store_true",
        help=(
            "Append v20.8 distributed sliding-window residual time-frequency analysis "
            "plus local fixed-frequency amplitude/phase tracking. Requires the coordinator "
            "and compatible distributed workers."
        ),
    )
    recovery.add_argument(
        "--continue-nonstationary",
        action="store_true",
        help=(
            "Append v20.9 distributed long-baseline nonstationary mode modeling. "
            "The workflow creates ordinary generic Lomb-Scargle work units from "
            "deterministically time-warped residual datasets."
        ),
    )
    recovery.add_argument(
        "--continue-long-baseline-frequency-confirmation",
        action="store_true",
        help=(
            "Append local, network-free leave-one-independent-sector-out "
            "confirmation of an exact AMBIGUOUS_HARMONIC_OR_MODE result. "
            "This is separate from --continue-nonstationary."
        ),
    )
    recovery.add_argument(
        "--continue-v20-8-long-baseline-time-frequency-confirmation",
        action="store_true",
        help=(
            "Append local, network-free leave-one-independent-sector-out "
            "confirmation of either the exact terminal unresolved v20.8 "
            "boundary or the exact resolved-cycle v20.8.1 recurrent-residual "
            "boundary. This is separate from "
            "--continue-long-baseline-frequency-confirmation and "
            "--continue-nonstationary."
        ),
    )
    recovery.add_argument(
        "--continue-confirmed-coherent-mode-identification",
        action="store_true",
        help=(
            "Append local, network-free full-sector mode identification "
            "from the exact finalized v20.8.1 "
            "COHERENT_RESIDUAL_FREQUENCY_CONFIRMED boundary."
        ),
    )
    recovery.add_argument(
        "--continue-transient-mode-validation",
        action="store_true",
        help=(
            "Append local, network-free leave-one-detection-sector-out "
            "validation from the exact finalized resolved-cycle v20.8 "
            "TRANSIENT_RESIDUAL_MODE boundary."
        ),
    )
    recovery.add_argument(
        "--continue-confirmed-nonstationary-mode-modeling",
        action="store_true",
        help=("Append v20.9.2 distributed nonstationary modeling from the exact "
              "completed v20.9.1 nonstationary/intermittent confirmation boundary."),
    )
    recovery.add_argument(
        "--continue-recurrent-residual-nonstationary-mode-modeling",
        action="store_true",
        help=(
            "Append v20.9.3 distributed nonstationary modeling from the "
            "exact finalized v20.8.2 recurrent-residual confirmation "
            "boundary using only frozen family-subtracted windows."
        ),
    )
    recovery.add_argument(
        "--continue-residual-mode-localization",
        action="store_true",
        help=(
            "Append v20.10 distributed pixel localization of the v20.9 drifting residual mode. "
            "The workflow prewhitens and time-warps TESS pixel light curves, then exposes each "
            "usable pixel as an ordinary openstar.lomb-scargle.v1 dataset."
        ),
    )
    recovery.add_argument(
        "--continue-residual-external-evidence",
        action="store_true",
        help=(
            "Append v20.10.1 network-free adjudication of the frozen external "
            "variability and binary classifications after target-supported v20.10 "
            "residual-mode localization."
        ),
    )
    recovery.add_argument(
        "--continue-target-residual-astrophysical-mechanism",
        action="store_true",
        help=(
            "Append v20.10.2 network-free mechanism-hypothesis adjudication "
            "from the exact target-associated nonbinary v20.10.1 boundary."
        ),
    )
    recovery.add_argument(
        "--continue-residual-mode-localization-review",
        action="store_true",
        help=(
            "Append v20.11 distributed time-resolved source-localization review of the "
            "v20.9 drifting residual mode after unresolved v20.10 static localization. "
            "Each usable pixel-window remains an ordinary openstar.lomb-scargle.v1 dataset."
        ),
    )
    recovery.add_argument(
        "--continue-neighbor-catalog-pixel-response-review",
        action="store_true",
        help=(
            "Append v20.11.1 catalog-guided review of the already-persisted "
            "v20.11 residual pixel-response maps. This queries a fixed TIC/Gaia "
            "neighborhood but performs no TESS download or distributed work."
        ),
    )
    recovery.add_argument(
        "--continue-multi-source-residual",
        action="store_true",
        help=(
            "Append v20.12 distributed multi-source residual decomposition after v20.11 "
            "finds source switching or blended residual variability. Spatial component "
            "light curves are exposed as ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-offset-source-identification",
        action="store_true",
        help=(
            "Append v20.13 catalog identification of the v20.12 best offset residual component. "
            "This is a network/catalog continuation and does not require coordinator workers."
        ),
    )
    recovery.add_argument(
        "--continue-offset-source-variability",
        action="store_true",
        help=(
            "Append v20.14 distributed variability validation of the v20.13 catalog counterpart. "
            "Catalog-guided target/counterpart residual series are exposed as ordinary "
            "openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-calibrated-prf-deblending",
        action="store_true",
        help=(
            "Append v20.15 distributed sector-calibrated pixel-response deblending after v20.14 "
            "finds only suggestive or unresolved catalog-counterpart support. Separated source "
            "series remain ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-difference-image-localization",
        action="store_true",
        help=(
            "Append v20.16 distributed residual-frequency refinement plus TESS difference-image "
            "source localization after unresolved v20.15 calibrated pixel-response deblending. "
            "Workers execute ordinary openstar.lomb-scargle.v1 frequency work only."
        ),
    )
    recovery.add_argument(
        "--continue-frequency-localized-pixel-response",
        action="store_true",
        help=(
            "Append v20.17 distributed narrow-band per-pixel Lomb-Scargle confirmation after "
            "unresolved v20.16 difference imaging. The TESS workflow combines generic pixel powers "
            "with fixed-frequency phase coherence to localize the residual source."
        ),
    )
    recovery.add_argument(
        "--continue-official-spoc-prf-forward-modeling",
        action="store_true",
        help=(
            "Append v20.18 official SPOC PRF forward modeling after v20.17 cannot securely localize "
            "the residual. The TESS workflow downloads/interpolates the public MAST PRF calibration, "
            "separates target/counterpart series, then exposes them as ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-external-high-resolution-variability-validation",
        action="store_true",
        help=(
            "Append v20.19 source-resolved external variability validation after v20.18 reaches the "
            "TESS spatial-attribution limit. Gaia DR3 epoch photometry is queried independently for the "
            "frozen target/counterpart pair; usable G-band series run as ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-skymapper-resolved-photometry",
        action="store_true",
        help=(
            "Append v20.20 SkyMapper DR4 resolved-epoch-photometry screening after v20.19 finds no Gaia DR3 epoch data. "
            "Only distinct SkyMapper objects and clean, good-seeing PSF detections are admitted; usable single-band series "
            "run as ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-nsc-resolved-photometry",
        action="store_true",
        help=(
            "Append v20.21 NOIRLab Source Catalog DR2 resolved-photometry screening after v20.20 remains unresolved. "
            "The frozen Gaia pair must map to two distinct NSC objects; only same-exposure/filter co-detections that "
            "independently position-match both sources are admitted to ordinary openstar.lomb-scargle.v1 work."
        ),
    )
    recovery.add_argument(
        "--continue-noirlab-image-forced-photometry",
        action="store_true",
        help=(
            "Append v20.22 public NOIRLab image-level two-source forced photometry after v20.21 remains unresolved. "
            "The frozen Gaia positions are fit directly in calibrated single-epoch SIA cutouts; only strict unsaturated, "
            "well-resolved, well-conditioned source series become ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-des-dr2-se-local-forced-photometry",
        action="store_true",
        help=(
            "Append v20.23 DES DR2 single-epoch source-local forced photometry after v20.22 remains unresolved. "
            "The two frozen Gaia sources are fit in independent local cutouts, so saturation of one source does not "
            "automatically veto the other; accepted source-band series run as ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-atlas-forced-photometry",
        action="store_true",
        help=(
            "Append v20.24 ATLAS calibrated target-image forced photometry after v20.23 finds no DES single-epoch coverage. "
            "Credentials are read from OPENSTAR_ATLAS_API_TOKEN or OPENSTAR_ATLAS_USERNAME/OPENSTAR_ATLAS_PASSWORD. "
            "Accepted nightly source-band series run as ordinary openstar.lomb-scargle.v1 datasets."
        ),
    )
    recovery.add_argument(
        "--continue-atlas-forced-photometry-reanalysis",
        action="store_true",
        help=(
            "Append v20.25 reanalysis of the immutable v20.24 ATLAS target-image files. "
            "The inappropriate individual >=3-sigma detection gate is removed; signed quality-valid forced fluxes are "
            "inverse-variance binned nightly before ordinary openstar.lomb-scargle.v1 work."
        ),
    )
    recovery.add_argument(
        "--continue-atlas-time-resolved",
        action="store_true",
        help=(
            "Append v20.26 time-resolved ATLAS counterpart recurrence after the v20.25 global signed-flux analysis remains unresolved. "
            "No new archive query is performed; the immutable counterpart light curve is split into shared c/o observing seasons and "
            "each season/filter is searched independently with the unchanged strict prominence gate."
        ),
    )
    recovery.add_argument(
        "--continue-atlas-fixed-window-recurrence",
        action="store_true",
        help=(
            "Append v20.27 deterministic ATLAS fixed-window counterpart recurrence after v20.26 gap-based splitting collapses into one season. "
            "No new ATLAS query is performed. The immutable counterpart nightly photometry is divided into non-overlapping 180-day bins "
            "anchored to absolute MJD zero; every window/filter keeps the unchanged RELIABLE + prominence>=2.0 acceptance rule."
        ),
    )
    recovery.add_argument(
        "--continue-targeted-observation-planning",
        action="store_true",
        help=(
            "Append v20.28 targeted high-resolution time-series observation planning after v20.27 exhausts the archival recurrence branch. "
            "This stage performs no new archive query and no distributed period search; it freezes the campaign cadence, image-quality, "
            "paired exposure tiers, filters, acceptance criteria, and OpenStar ingest contract before new observations are collected."
        ),
    )
    return parser.parse_args()



def _next_stage_number(investigation) -> int:
    numbers = []
    for stage in investigation.stages:
        try:
            numbers.append(int(str(stage.id).split("-", 1)[0]))
        except (TypeError, ValueError):
            continue
    return max(numbers, default=0) + 1


def _can_continue_contradiction(investigation) -> None:
    if investigation.status != "HUMAN_REVIEW_REQUIRED":
        raise RuntimeError(
            "--continue-contradiction requires a HUMAN_REVIEW_REQUIRED investigation."
        )

    targeted = None
    broad_exists = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.independent.interpret"
            and stage.status == "COMPLETE"
        ):
            targeted = stage.result
        if stage.handler_id.startswith("openstar.tess.independent.broad."):
            broad_exists = True

    if targeted is None:
        raise RuntimeError(
            "Investigation has no completed targeted independent-sector interpretation."
        )
    if broad_exists:
        raise RuntimeError(
            "Investigation already contains v20.3 broad independent-sector stages."
        )

    claim = ((targeted.get("claimDecision") or {}).get("claim"))
    if claim == "INDEPENDENT_PERIOD_ESTIMATE":
        raise RuntimeError(
            "Targeted independent sectors already confirmed the period; contradiction continuation is unnecessary."
        )


def _can_continue_harmonic_family(investigation) -> bool:
    """Validate zero-compute harmonic-family continuation.

    Returns True when the top-level investigation status is an orphaned
    RUNNING snapshot with no RUNNING stage. That can happen in v20.3.1 when
    the continuation set RUNNING before an unnecessary coordinator health
    check failed. The completed immutable stages remain authoritative, so
    this state is safe to continue.
    """
    running_stages = [
        stage for stage in investigation.stages if stage.status == "RUNNING"
    ]
    if running_stages:
        raise RuntimeError(
            "Investigation contains an actual RUNNING stage. Use --resume "
            "for an interrupted stage instead of --continue-harmonic-family."
        )

    if investigation.status not in {
        "COMPLETE",
        "HUMAN_REVIEW_REQUIRED",
        "RUNNING",
    }:
        raise RuntimeError(
            "--continue-harmonic-family requires completed broad-search "
            "evidence from a terminal investigation."
        )

    broad_prepare = False
    broad_run = False
    already_reinterpreted = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.independent.broad.prepare"
            and stage.status == "COMPLETE"
        ):
            broad_prepare = True
        elif (
            stage.handler_id == "openstar.tess.independent.broad.run"
            and stage.status == "COMPLETE"
        ):
            broad_run = True
        elif stage.handler_id == "openstar.tess.independent.harmonic-family.interpret":
            already_reinterpreted = True

    if not broad_prepare or not broad_run:
        raise RuntimeError(
            "Investigation does not contain completed v20.3 broad independent-sector compute."
        )
    if already_reinterpreted:
        raise RuntimeError(
            "Investigation already contains a v20.3.1 harmonic-family reinterpretation."
        )

    return investigation.status == "RUNNING"



def _can_continue_period_semantics(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before period-semantic continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-period-semantics requires a terminal investigation."
        )
    has_harmonic = any(
        stage.handler_id == "openstar.tess.independent.harmonic-family.interpret"
        and stage.status == "COMPLETE"
        for stage in investigation.stages
    )
    already_done = any(
        stage.handler_id == "openstar.tess.period-semantics.reinterpret"
        for stage in investigation.stages
    )
    if not has_harmonic:
        raise RuntimeError(
            "Period-semantic continuation requires completed harmonic-family evidence."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains the v20.3.3 period-semantic correction."
        )


def _can_continue_morphology(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before morphology continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError("--continue-morphology requires a terminal investigation.")
    has_semantics = any(
        stage.handler_id == "openstar.tess.period-semantics.reinterpret"
        and stage.status == "COMPLETE"
        for stage in investigation.stages
    )
    has_independent_prepare = any(
        stage.handler_id == "openstar.tess.independent.prepare"
        and stage.status == "COMPLETE"
        for stage in investigation.stages
    )
    already_done = any(
        stage.handler_id == "openstar.tess.morphology.analyze"
        for stage in investigation.stages
    )
    if not has_semantics:
        raise RuntimeError(
            "Run --continue-period-semantics first so v20.4 starts from the corrected period-evidence model."
        )
    if not has_independent_prepare:
        raise RuntimeError(
            "Morphology continuation requires already-frozen independent TESS sectors."
        )
    if already_done:
        raise RuntimeError("Investigation already contains v20.4 morphology analysis.")



def _can_continue_physical_interpretation(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before physical-interpretation continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-physical-interpretation requires a terminal investigation."
        )
    has_morphology = any(
        stage.handler_id == "openstar.tess.morphology.analyze"
        and stage.status == "COMPLETE"
        and bool((stage.result or {}).get("physicalCycleResolved"))
        for stage in investigation.stages
    )
    has_identity = any(
        stage.handler_id == "openstar.tess.catalog-identity"
        and stage.status == "COMPLETE"
        for stage in investigation.stages
    )
    already_done = any(
        stage.handler_id == "openstar.tess.physical.interpret"
        for stage in investigation.stages
    )
    if not has_morphology:
        raise RuntimeError(
            "Run --continue-morphology first and resolve the physical cycle before v20.5 physical interpretation."
        )
    if not has_identity:
        raise RuntimeError(
            "Physical interpretation requires the completed catalog-identity stage."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.5 physical-mechanism discrimination."
        )

def _can_continue_source_localization(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before source-localization continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-source-localization requires a terminal investigation."
        )
    physical = None
    already_done = False
    for stage in investigation.stages:
        if stage.handler_id == "openstar.tess.physical.interpret" and stage.status == "COMPLETE":
            physical = stage.result
        if stage.handler_id == "openstar.tess.source-localization.analyze":
            already_done = True
    if physical is None:
        raise RuntimeError(
            "Run --continue-physical-interpretation first so v20.6 has a resolved physical period and contamination screen."
        )
    if physical.get("recommendedNextTest") != "PIXEL_LEVEL_SOURCE_LOCALIZATION":
        raise RuntimeError(
            "v20.5 did not recommend PIXEL_LEVEL_SOURCE_LOCALIZATION for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.6 source localization."
        )


def _can_continue_multimode(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before multi-mode continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError("--continue-multimode requires a terminal investigation.")

    localization = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.source-localization.analyze"
            and stage.status == "COMPLETE"
        ):
            localization = stage.result
        if stage.handler_id.startswith("openstar.tess.multimode."):
            already_done = True

    if localization is None:
        raise RuntimeError(
            "Run --continue-source-localization first so v20.7 knows the periodic signal belongs to the TIC target."
        )
    cross = (localization or {}).get("crossSector") or {}
    if cross.get("classification") != "TARGET_SOURCE_SUPPORTED":
        raise RuntimeError(
            "v20.7 requires TARGET_SOURCE_SUPPORTED; residual mode decomposition should not interpret an off-target signal as target physics."
        )
    if localization.get("recommendedNextTest") != "MULTI_MODE_FREQUENCY_DECOMPOSITION":
        raise RuntimeError(
            "v20.6 did not recommend MULTI_MODE_FREQUENCY_DECOMPOSITION for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.7 multi-mode decomposition stages. Use --resume only if one is actually interrupted."
        )


def _can_continue_time_frequency(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before time-frequency continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError("--continue-time-frequency requires a terminal investigation.")

    multimode = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.multimode.summarize"
            and stage.status == "COMPLETE"
        ):
            multimode = stage.result
        if stage.handler_id.startswith("openstar.tess.time-frequency."):
            already_done = True

    if multimode is None:
        raise RuntimeError(
            "Run --continue-multimode first so v20.8 has the residual decomposition result."
        )
    if multimode.get("recommendedNextTest") != "TIME_FREQUENCY_EVOLUTION_ANALYSIS":
        raise RuntimeError(
            "v20.7 did not recommend TIME_FREQUENCY_EVOLUTION_ANALYSIS for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.8 time-frequency evolution stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_nonstationary(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before nonstationary continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError("--continue-nonstationary requires a terminal investigation.")

    time_frequency = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.time-frequency.summarize"
            and stage.status == "COMPLETE"
        ):
            time_frequency = stage.result
        if stage.handler_id.startswith("openstar.tess.nonstationary."):
            already_done = True

    if time_frequency is None:
        raise RuntimeError(
            "Run --continue-time-frequency first so v20.9 has the sliding-window evolution result."
        )
    if time_frequency.get("recommendedNextTest") != "LONG_BASELINE_NONSTATIONARY_MODE_MODELING":
        raise RuntimeError(
            "v20.8 did not recommend LONG_BASELINE_NONSTATIONARY_MODE_MODELING for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.9 nonstationary modeling stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_long_baseline_frequency_confirmation(investigation) -> None:
    """Validate the exact terminal ambiguity before any state mutation."""
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before "
            "long-baseline frequency confirmation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-long-baseline-frequency-confirmation requires a "
            "terminal investigation."
        )
    if any(
        stage.handler_id == LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation already contains long-baseline frequency "
            "confirmation. Use --resume only if it is actually interrupted."
        )

    mode_stage = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.mode-identification.analyze"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if mode_stage is None:
        raise RuntimeError(
            "A completed mode-identification stage is required before "
            "long-baseline frequency confirmation."
        )
    mode_evidence = validate_ambiguous_mode_identification(mode_stage.result)
    contract = build_long_baseline_frequency_confirmation_contract(
        mode_stage.result
    )
    mode_index = investigation.stages.index(mode_stage)
    if any(
        stage.status == "COMPLETE"
        and stage.handler_id != "openstar.tess.finalize"
        for stage in investigation.stages[mode_index + 1:]
    ):
        raise RuntimeError(
            "Later scientific stages already consume the mode-identification "
            "boundary."
        )
    latest = investigation.stages[-1] if investigation.stages else None
    if not (
        latest is not None
        and latest.status == "COMPLETE"
        and latest.handler_id == "openstar.tess.finalize"
        and latest.stop is True
    ):
        raise RuntimeError(
            "Long-baseline frequency confirmation requires a completed "
            "terminal finalization stage."
        )

    required_handlers = (
        "openstar.tess.source-localization.analyze",
        "openstar.tess.multimode.summarize",
        "openstar.tess.time-frequency.summarize",
    )
    if any(not any(
        stage.handler_id == handler
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
        for stage in investigation.stages
    ) for handler in required_handlers):
        raise RuntimeError(
            "Completed source-localization, multi-mode, and time-frequency "
            "evidence are required."
        )
    multimode_stage = next(
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.multimode.summarize"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    )
    if not (
        mode_stage.parameters == {
            "evidenceLineage": MULTIMODE_MODE_EVIDENCE_LINEAGE
        }
        and mode_stage.triggered_by_stage_id == multimode_stage.id
    ):
        raise RuntimeError(
            "Mode identification does not carry the exact recurrent "
            "multi-mode lineage."
        )

    prepared_stage = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    independent_stage = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.independent.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if prepared_stage is None or independent_stage is None:
        raise RuntimeError("Frozen primary/independent preparation is missing.")
    prepared = prepared_stage.result
    support = set(mode_evidence["independentSectors"])
    independent = [
        item for item in independent_stage.result.get("preparedSectors") or []
        if isinstance(item, dict)
        and item.get("sector") is not None
        and item.get("datasetPath")
    ]
    if not support.issubset({int(item["sector"]) for item in independent}):
        raise RuntimeError(
            "Frozen independent datasets do not match mode-sector support."
        )
    try:
        dataset_specs = [{
            "datasetID": prepared["datasetID"],
            "datasetPath": prepared["datasetPath"],
            "ticID": prepared["ticID"],
            "sector": prepared["sector"],
            "role": "PRIMARY",
        }]
        dataset_specs.extend({
            "datasetID": item["datasetID"],
            "datasetPath": item["datasetPath"],
            "ticID": prepared["ticID"],
            "sector": int(item["sector"]),
            "role": "INDEPENDENT",
        } for item in independent)
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Frozen dataset lineage is incomplete.") from None
    validate_frozen_dataset_lineage(
        method_contract=contract,
        dataset_specs=dataset_specs,
    )


def _can_continue_v20_8_long_baseline_time_frequency_confirmation(
    investigation,
) -> None:
    """Validate one exact terminal v20.8 long-baseline boundary."""
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before "
            "v20.8 long-baseline time-frequency confirmation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-v20-8-long-baseline-time-frequency-confirmation "
            "requires a terminal investigation."
        )
    if any(
        stage.handler_id
        == V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation already contains v20.8 long-baseline "
            "time-frequency confirmation."
        )

    summary_stage = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.time-frequency.summarize"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    transient_stage = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == TRANSIENT_MODE_VALIDATION_HANDLER_ID
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if summary_stage is None or latest is None:
        raise RuntimeError(
            "The exact finalized v20.8 long-baseline boundary is required."
        )

    summary_index = investigation.stages.index(summary_stage)
    direct_boundary = (
        transient_stage is None
        and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE"
        and latest.stop is True
        and latest.triggered_by_stage_id == summary_stage.id
        and latest.parameters.get("outputSuffix") == "v20.8"
        and isinstance(latest.result, dict)
        and latest.result.get("timeFrequencyEvolution")
        == summary_stage.result
        and latest.result.get("recommendedNextTest")
        == "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
        and tuple(investigation.stages[summary_index + 1:]) == (latest,)
    )

    prior_terminal = None
    recurrent_boundary = False
    if transient_stage is not None:
        transient_index = investigation.stages.index(transient_stage)
        if transient_index > 0:
            prior_terminal = investigation.stages[transient_index - 1]
        transient_hashes = (
            transient_stage.provenance.input_hashes
            if transient_stage.provenance else {}
        )
        recurrent_boundary = (
            prior_terminal is not None
            and prior_terminal.handler_id == "openstar.tess.finalize"
            and prior_terminal.status == "COMPLETE"
            and prior_terminal.stop is True
            and prior_terminal.triggered_by_stage_id == summary_stage.id
            and prior_terminal.parameters.get("outputSuffix") == "v20.8"
            and isinstance(prior_terminal.result, dict)
            and prior_terminal.result.get("timeFrequencyEvolution")
            == summary_stage.result
            and prior_terminal.result.get("recommendedNextTest")
            == "TRANSIENT_MODE_VALIDATION"
            and transient_stage.triggered_by_stage_id == summary_stage.id
            and transient_hashes.get("timeFrequencySummary")
            == sha256_json(summary_stage.result)
            and transient_hashes.get("methodContract")
            == transient_stage.result.get("methodContractHash")
            and latest.handler_id == "openstar.tess.finalize"
            and latest.status == "COMPLETE"
            and latest.stop is True
            and latest.triggered_by_stage_id == transient_stage.id
            and latest.parameters.get("outputSuffix")
            == "v20.8.1-transient-mode-validation"
            and isinstance(latest.result, dict)
            and latest.result.get("transientModeValidation")
            == transient_stage.result
            and latest.result.get("recommendedNextTest")
            == "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
            and tuple(investigation.stages[summary_index + 1:])
            == (prior_terminal, transient_stage, latest)
        )

    if not (direct_boundary or recurrent_boundary):
        raise RuntimeError(
            "The exact finalized v20.8 long-baseline boundary is required."
        )

    interpretation_stage = next((
        stage for stage in investigation.stages
        if stage.id == summary_stage.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.interpret"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    run_stage = next((
        stage for stage in investigation.stages
        if interpretation_stage is not None
        and stage.id == interpretation_stage.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.run"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    preparation_stage = next((
        stage for stage in investigation.stages
        if run_stage is not None
        and stage.id == run_stage.triggered_by_stage_id
        and stage.handler_id == "openstar.tess.time-frequency.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    prepared_stage = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    morphology_stage = next((
        stage for stage in reversed(investigation.stages[:summary_index])
        if stage.handler_id == "openstar.tess.morphology.analyze"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if any(stage is None for stage in (
        interpretation_stage,
        run_stage,
        preparation_stage,
        prepared_stage,
        morphology_stage,
    )):
        raise RuntimeError(
            "The completed v20.8 prepare/run/interpret/morphology lineage "
            "is incomplete."
        )

    interpretation_hashes = (
        interpretation_stage.provenance.input_hashes
        if interpretation_stage.provenance else {}
    )
    preparation_hashes = (
        preparation_stage.provenance.input_hashes
        if preparation_stage.provenance else {}
    )
    summary_hashes = (
        summary_stage.provenance.input_hashes
        if summary_stage.provenance else {}
    )
    if not (
        interpretation_hashes.get("preparation")
        == sha256_json(preparation_stage.result)
        and interpretation_hashes.get("projectResult")
        == sha256_json(run_stage.result)
        and preparation_hashes.get("morphology")
        == sha256_json(morphology_stage.result)
        and summary_hashes.get("morphology")
        == sha256_json(morphology_stage.result)
        and summary_hashes.get("timeFrequencyInterpretation")
        == sha256_json(interpretation_stage.result)
    ):
        raise RuntimeError(
            "v20.8 provenance does not match the authoritative completed "
            "physical-period and window-analysis lineage."
        )

    if recurrent_boundary:
        contract = build_recurrent_residual_method_contract(
            transient_validation=transient_stage.result,
        )
        dataset_specs = build_recurrent_residual_dataset_specs(
            expected_tic_id=int(prepared_stage.result["ticID"]),
            preparation=preparation_stage.result,
        )
    else:
        contract = build_v20_8_long_baseline_method_contract(
            preparation=preparation_stage.result,
            interpretation=interpretation_stage.result,
            summary=summary_stage.result,
        )
        dataset_specs = build_v20_8_long_baseline_dataset_specs(
            expected_tic_id=int(prepared_stage.result["ticID"]),
            preparation=preparation_stage.result,
        )
    validate_v20_8_frozen_window_lineage(
        method_contract=contract,
        dataset_specs=dataset_specs,
    )

def _can_continue_transient_mode_validation(investigation) -> None:
    """Validate the exact finalized resolved-cycle transient v20.8 boundary."""
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before "
            "transient-mode validation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-transient-mode-validation requires a terminal "
            "investigation."
        )
    if any(
        stage.handler_id == TRANSIENT_MODE_VALIDATION_HANDLER_ID
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation already contains transient-mode validation."
        )

    summary = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.time-frequency.summarize"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    binary = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.binary-confirmation.analyze"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if summary is None or binary is None or latest is None or not (
        latest.handler_id == "openstar.tess.finalize"
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
        raise RuntimeError(
            "The exact finalized resolved-cycle transient v20.8 boundary "
            "is required."
        )
    summary_index = investigation.stages.index(summary)
    if tuple(investigation.stages[summary_index + 1:]) != (latest,):
        raise RuntimeError(
            "Later stages already consume the transient v20.8 boundary."
        )

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
        raise RuntimeError(
            "The completed transient morphology/binary/prepare/run/interpret/"
            "summarize lineage is incomplete."
        )

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
    ):
        raise RuntimeError(
            "Transient v20.8 provenance does not match the authoritative "
            "physical-period and window-analysis lineage."
        )

    # Freeze the deterministic method before the validator opens any window
    # dataset containing family-subtracted flux.
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


def _confirmed_coherent_mode_inputs(investigation):
    confirmation = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id
        == V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    if confirmation is None or not (
        latest is not None
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
        raise RuntimeError(
            "The exact finalized confirmed coherent v20.8.1 boundary is required."
        )
    confirmation_index = investigation.stages.index(confirmation)
    if tuple(investigation.stages[confirmation_index + 1:]) != (latest,):
        raise RuntimeError(
            "Later stages already consume the confirmed coherent v20.8.1 boundary."
        )

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
    morphology = next((
        stage for stage in reversed(
            investigation.stages[:confirmation_index]
        )
        if stage.handler_id == "openstar.tess.morphology.analyze"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    independent = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.independent.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if any(stage is None for stage in (
        summary, interpretation, run, preparation, morphology,
        prepared, independent,
    )):
        raise RuntimeError(
            "Confirmed coherent mode identification requires the completed "
            "v20.8 and frozen full-sector lineage."
        )

    confirmation_hashes = (
        confirmation.provenance.input_hashes
        if confirmation.provenance else {}
    )
    rebuilt_confirmation_contract = build_v20_8_long_baseline_method_contract(
        preparation=preparation.result,
        interpretation=interpretation.result,
        summary=summary.result,
    )
    rebuilt_confirmation_hash = (
        v20_8_confirmation_method_contract_hash(
            rebuilt_confirmation_contract
        )
    )
    if not (
        confirmation.result.get("methodContract")
        == rebuilt_confirmation_contract
        and confirmation.result.get("methodContractHash")
        == rebuilt_confirmation_hash
        and confirmation_hashes.get("methodContract")
        == rebuilt_confirmation_hash
        and confirmation_hashes.get("morphology")
        == sha256_json(morphology.result)
        and confirmation_hashes.get("timeFrequencyPreparation")
        == sha256_json(preparation.result)
        and confirmation_hashes.get("timeFrequencyProjectResult")
        == sha256_json(run.result)
        and confirmation_hashes.get("timeFrequencyInterpretation")
        == sha256_json(interpretation.result)
        and confirmation_hashes.get("timeFrequencySummary")
        == sha256_json(summary.result)
    ):
        raise RuntimeError(
            "The confirmed coherent v20.8.1 provenance has changed."
        )
    evidence = validate_v20_8_confirmed_coherent_residual(
        confirmation.result
    )

    prepared_by_sector = {}
    for item in independent.result.get("preparedSectors") or []:
        if not isinstance(item, dict) or item.get("sector") is None:
            continue
        sector = int(item["sector"])
        if sector in prepared_by_sector:
            raise RuntimeError(
                "Frozen independent-sector preparation is duplicated."
            )
        prepared_by_sector[sector] = item
    support = evidence["independentSectors"]
    if not set(support).issubset(prepared_by_sector):
        raise RuntimeError(
            "Confirmed coherent sectors do not match frozen independent data."
        )
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
    contract = build_confirmed_coherent_mode_method_contract(
        confirmation=confirmation.result,
        dataset_specs=dataset_specs,
    )

    window_specs = build_v20_8_long_baseline_dataset_specs(
        expected_tic_id=int(prepared.result["ticID"]),
        preparation=preparation.result,
    )
    validate_v20_8_frozen_window_lineage(
        method_contract=rebuilt_confirmation_contract,
        dataset_specs=window_specs,
    )
    for spec in window_specs:
        key = (
            "frozenWindowDataset:"
            f"{spec['role']}:{spec['sector']}:{spec['windowIndex']}"
        )
        if confirmation_hashes.get(key) != sha256_file(spec["datasetPath"]):
            raise RuntimeError(
                "A frozen v20.8.1 confirmation window has changed."
            )
    validate_confirmed_coherent_mode_dataset_lineage(
        method_contract=contract,
        dataset_specs=dataset_specs,
    )
    return confirmation, contract, dataset_specs


def _can_continue_confirmed_coherent_mode_identification(
    investigation,
) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before "
            "confirmed coherent mode identification."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-confirmed-coherent-mode-identification requires "
            "a terminal investigation."
        )
    if any(
        stage.handler_id == "openstar.tess.mode-identification.analyze"
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation already contains mode identification."
        )
    _confirmed_coherent_mode_inputs(investigation)


def _can_continue_confirmed_nonstationary_mode_modeling(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError("Investigation contains a RUNNING stage. Use --resume first.")
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError("Confirmed nonstationary modeling requires a terminal investigation.")
    if any(stage.handler_id.startswith("openstar.tess.nonstationary.")
           for stage in investigation.stages):
        raise RuntimeError("Investigation already contains nonstationary modeling stages.")
    confirmation = next((stage for stage in reversed(investigation.stages)
                         if stage.handler_id == LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID
                         and stage.status == "COMPLETE" and isinstance(stage.result, dict)), None)
    latest = investigation.stages[-1] if investigation.stages else None
    if confirmation is None or not (
        latest is not None and latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE" and latest.stop is True
        and latest.triggered_by_stage_id == confirmation.id
        and latest.parameters.get("outputSuffix") == "v20.9.1-long-baseline-frequency-confirmation"
        and (latest.result or {}).get("longBaselineFrequencyConfirmation") == confirmation.result
    ):
        raise RuntimeError("The exact finalized v20.9.1 confirmation boundary is required.")
    index = investigation.stages.index(confirmation)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        raise RuntimeError("Later stages already consume the confirmation boundary.")
    contract = build_confirmed_nonstationary_method_contract(confirmation.result)
    for path in (contract.get("evidenceBoundary") or {}).get("frozenDatasetPaths") or []:
        if not isinstance(path, str) or not Path(path).is_file():
            raise RuntimeError(f"Frozen confirmation dataset is missing: {path}")


def _recurrent_residual_nonstationary_inputs(investigation):
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume first."
        )
    if investigation.status not in {
        "COMPLETE", "HUMAN_REVIEW_REQUIRED"
    }:
        raise RuntimeError(
            "Recurrent-residual nonstationary modeling requires a "
            "terminal investigation."
        )
    if any(
        stage.handler_id.startswith("openstar.tess.nonstationary.")
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation already contains nonstationary modeling stages."
        )

    confirmation = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id
        == V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    transient = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == TRANSIENT_MODE_VALIDATION_HANDLER_ID
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    if (
        confirmation is None
        or transient is None
        or latest is None
        or confirmation.triggered_by_stage_id != transient.id
        or latest.handler_id != "openstar.tess.finalize"
        or latest.status != "COMPLETE"
        or latest.stop is not True
        or latest.triggered_by_stage_id != confirmation.id
        or latest.parameters.get("outputSuffix")
        != "v20.8.2-recurrent-residual-long-baseline-confirmation"
        or not isinstance(latest.result, dict)
        or latest.result.get("longBaselineTimeFrequencyConfirmation")
        != confirmation.result
        or latest.result.get("recommendedNextTest")
        != "LONG_BASELINE_NONSTATIONARY_MODE_MODELING"
    ):
        raise RuntimeError(
            "The exact finalized v20.8.2 recurrent-residual "
            "confirmation boundary is required."
        )
    confirmation_index = investigation.stages.index(confirmation)
    if tuple(investigation.stages[confirmation_index + 1:]) != (latest,):
        raise RuntimeError(
            "Later stages already consume the recurrent-residual "
            "confirmation boundary."
        )

    boundary = validate_recurrent_residual_nonstationary_boundary(
        confirmation.result
    )
    contract = (
        build_recurrent_residual_nonstationary_method_contract(
            confirmation.result
        )
    )
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    preparation = next((
        stage for stage in reversed(investigation.stages)
        if stage.handler_id == "openstar.tess.time-frequency.prepare"
        and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    if prepared is None or preparation is None:
        raise RuntimeError(
            "The frozen v20.8 time-frequency preparation is missing."
        )
    specs = build_v20_8_long_baseline_dataset_specs(
        expected_tic_id=int(prepared.result["ticID"]),
        preparation=preparation.result,
    )

    confirmation_hashes = (
        confirmation.provenance.input_hashes
        if confirmation.provenance else {}
    )
    transient_hashes = (
        transient.provenance.input_hashes
        if transient.provenance else {}
    )
    if not (
        confirmation_hashes.get("methodContract")
        == confirmation.result.get("methodContractHash")
        and confirmation_hashes.get("transientModeValidation")
        == sha256_json(transient.result)
        and transient_hashes.get("methodContract")
        == transient.result.get("methodContractHash")
        and sorted(boundary["frozenWindowDatasetPaths"])
        == sorted(spec["datasetPath"] for spec in specs)
    ):
        raise RuntimeError(
            "The v20.8.2 confirmation provenance no longer matches "
            "its recurrent-residual lineage."
        )

    # The method contract is built before this validation opens any flux.
    validate_v20_8_frozen_window_lineage(
        method_contract=contract,
        dataset_specs=specs,
    )
    for spec in specs:
        key = (
            "frozenWindowDataset:"
            f"{spec['role']}:{spec['sector']}:{spec['windowIndex']}"
        )
        if (
            confirmation_hashes.get(key)
            != sha256_file(spec["datasetPath"])
        ):
            raise RuntimeError(
                "A frozen v20.8.2 residual window has changed."
            )
    return confirmation, contract, specs


def _can_continue_recurrent_residual_nonstationary_mode_modeling(
    investigation,
) -> None:
    _recurrent_residual_nonstationary_inputs(investigation)


def _can_continue_residual_mode_localization(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before residual-mode localization continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-residual-mode-localization requires a terminal investigation."
        )

    nonstationary = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.nonstationary.summarize"
            and stage.status == "COMPLETE"
        ):
            nonstationary = stage.result
        if stage.handler_id.startswith("openstar.tess.residual-mode-localization."):
            already_done = True

    if nonstationary is None:
        raise RuntimeError(
            "Run --continue-nonstationary first so v20.10 has the preferred residual drift model."
        )
    if nonstationary.get("recommendedNextTest") != "RESIDUAL_MODE_PIXEL_LOCALIZATION":
        raise RuntimeError(
            "v20.9 did not recommend RESIDUAL_MODE_PIXEL_LOCALIZATION for this investigation."
        )
    if (nonstationary.get("evidenceLineage")
            == CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE):
        confirmation = next((
            stage.result for stage in reversed(investigation.stages)
            if stage.handler_id == LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID
            and stage.status == "COMPLETE" and isinstance(stage.result, dict)
        ), None)
        localization = next((
            stage.result for stage in reversed(investigation.stages)
            if stage.handler_id == "openstar.tess.source-localization.analyze"
            and stage.status == "COMPLETE" and isinstance(stage.result, dict)
        ), None)
        cycle = (
            localization.get("physicalCycleEvidence")
            if isinstance(localization, dict) else None
        )
        validate_confirmed_nonstationary_localization_boundary(
            nonstationary, confirmation, cycle
        )
        latest = investigation.stages[-1] if investigation.stages else None
        if not (
            latest is not None
            and latest.handler_id == "openstar.tess.finalize"
            and latest.status == "COMPLETE"
            and latest.stop is True
            and latest.parameters.get("outputSuffix")
            == "v20.9.2-confirmed-nonstationary"
            and (latest.result or {}).get("nonstationaryModeling")
            == nonstationary
        ):
            raise RuntimeError(
                "Confirmed residual localization requires the exact finalized "
                "v20.9.2 boundary."
            )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.10 residual-mode localization stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_residual_external_evidence(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before "
            "residual external-evidence continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-residual-external-evidence requires a terminal investigation."
        )
    if any(stage.handler_id == RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID
           for stage in investigation.stages):
        raise RuntimeError(
            "Investigation already contains residual external-evidence analysis."
        )
    def completed(handler_id):
        return next((
            stage for stage in reversed(investigation.stages)
            if stage.handler_id == handler_id and stage.status == "COMPLETE"
            and isinstance(stage.result, dict)
        ), None)
    localization = completed("openstar.tess.residual-mode-localization.interpret")
    nonstationary = completed("openstar.tess.nonstationary.summarize")
    confirmation = completed(LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID)
    source_localization = completed("openstar.tess.source-localization.analyze")
    identity = completed("openstar.tess.catalog-identity")
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
        raise RuntimeError(
            "The completed v20.9.1/v20.9.2/v20.10 evidence lineage is required."
        )
    if not (
        latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE" and latest.stop is True
        and latest.triggered_by_stage_id == localization.id
        and latest.parameters == {"outputSuffix": "v20.10"}
        and (latest.result or {}).get("residualModeLocalization")
        == localization.result
    ):
        raise RuntimeError("The exact finalized v20.10 boundary is required.")
    index = investigation.stages.index(localization)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        raise RuntimeError("Later stages already consume the v20.10 boundary.")
    validate_target_supported_boundary(
        localization=localization.result,
        nonstationary=nonstationary.result,
        confirmation=confirmation.result,
        physical_cycle=source_localization.result.get("physicalCycleEvidence"),
        identity=identity.result,
        expected_tic_id=int(prepared.result["ticID"]),
    )


def _can_continue_target_residual_astrophysical_mechanism(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before "
            "target-residual astrophysical-mechanism continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-target-residual-astrophysical-mechanism requires a "
            "terminal investigation."
        )
    if any(
        stage.handler_id == TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation already contains target-residual astrophysical-"
            "mechanism follow-up."
        )

    def completed(handler_id):
        return next((
            stage for stage in reversed(investigation.stages)
            if stage.handler_id == handler_id and stage.status == "COMPLETE"
            and isinstance(stage.result, dict)
        ), None)

    external = completed(RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID)
    identity = completed("openstar.tess.catalog-identity")
    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target" and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    latest = investigation.stages[-1] if investigation.stages else None
    if any(stage is None for stage in (external, identity, prepared, latest)):
        raise RuntimeError("The completed v20.10.1 evidence lineage is required.")
    if not (
        latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE" and latest.stop is True
        and latest.triggered_by_stage_id == external.id
        and latest.parameters == {
            "outputSuffix": "v20.10.1-residual-external-evidence"
        }
        and (latest.result or {}).get("residualExternalEvidence") == external.result
        and (latest.result or {}).get("recommendedNextTest")
        == "TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_FOLLOWUP"
    ):
        raise RuntimeError("The exact finalized v20.10.1 boundary is required.")
    index = investigation.stages.index(external)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        raise RuntimeError("Later stages already consume the v20.10.1 boundary.")
    hashes = external.provenance.input_hashes if external.provenance else {}
    if hashes.get("catalogIdentity") != sha256_json(identity.result):
        raise RuntimeError("The frozen catalog-identity lineage has changed.")
    validate_mechanism_followup_boundary(
        external_evidence=external.result,
        identity=identity.result,
        expected_tic_id=int(prepared.result["ticID"]),
    )



def _can_continue_residual_mode_localization_review(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before residual-mode localization review."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-residual-mode-localization-review requires a terminal investigation."
        )

    residual_localization = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.residual-mode-localization.interpret"
            and stage.status == "COMPLETE"
        ):
            residual_localization = stage.result
        if stage.handler_id.startswith("openstar.tess.residual-mode-localization-review."):
            already_done = True

    if residual_localization is None:
        raise RuntimeError(
            "Run --continue-residual-mode-localization first so v20.11 has the unresolved static pixel-localization result."
        )
    if residual_localization.get("recommendedNextTest") != "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW":
        raise RuntimeError(
            "v20.10 did not recommend RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.11 residual-mode localization-review stages. "
            "Use --resume only if one is actually interrupted."
        )



def _can_continue_neighbor_catalog_pixel_response_review(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before "
            "neighbor catalog/pixel-response continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-neighbor-catalog-pixel-response-review requires a "
            "terminal investigation."
        )
    if any(
        stage.handler_id == NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation already contains neighbor catalog/pixel-response review."
        )

    def completed(handler_id):
        return next((
            stage for stage in reversed(investigation.stages)
            if stage.handler_id == handler_id and stage.status == "COMPLETE"
            and isinstance(stage.result, dict)
        ), None)

    prepared = next((
        stage for stage in investigation.stages
        if stage.id == "001-prepare-target" and stage.status == "COMPLETE"
        and isinstance(stage.result, dict)
    ), None)
    identity = completed("openstar.tess.catalog-identity")
    mode = completed("openstar.tess.mode-identification.analyze")
    review_preparation = completed(
        "openstar.tess.residual-mode-localization-review.prepare"
    )
    review = completed("openstar.tess.residual-mode-localization-review.interpret")
    latest = investigation.stages[-1] if investigation.stages else None
    if any(stage is None for stage in (
        prepared, identity, mode, review_preparation, review, latest,
    )):
        raise RuntimeError("The completed v20.11 evidence lineage is required.")
    if not (
        latest.handler_id == "openstar.tess.finalize"
        and latest.status == "COMPLETE"
        and latest.stop is True
        and latest.triggered_by_stage_id == review.id
        and latest.parameters == {"outputSuffix": "v20.11"}
        and (latest.result or {}).get("residualModeLocalizationReview")
        == review.result
        and (latest.result or {}).get("recommendedNextTest")
        == "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW"
    ):
        raise RuntimeError("The exact finalized unresolved v20.11 boundary is required.")
    index = investigation.stages.index(review)
    if tuple(investigation.stages[index + 1:]) != (latest,):
        raise RuntimeError("Later stages already consume the v20.11 boundary.")
    hashes = review.provenance.input_hashes if review.provenance else {}
    if hashes.get("preparation") != sha256_json(review_preparation.result):
        raise RuntimeError("The frozen v20.11 preparation lineage has changed.")
    validate_neighbor_catalog_pixel_response_boundary(
        preparation=review_preparation.result,
        localization_review=review.result,
        mode_identification=mode.result,
        identity=identity.result,
        expected_tic_id=int(prepared.result["ticID"]),
    )


def _can_continue_multi_source_residual(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before multi-source residual continuation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-multi-source-residual requires a terminal investigation."
        )

    review = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.residual-mode-localization-review.interpret"
            and stage.status == "COMPLETE"
        ):
            review = stage.result
        if stage.handler_id.startswith("openstar.tess.multi-source-residual."):
            already_done = True

    if review is None:
        raise RuntimeError(
            "Run --continue-residual-mode-localization-review first so v20.12 has the time-resolved source-switching evidence."
        )
    if review.get("recommendedNextTest") != "MULTI_SOURCE_RESIDUAL_DECOMPOSITION":
        raise RuntimeError(
            "v20.11 did not recommend MULTI_SOURCE_RESIDUAL_DECOMPOSITION for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.12 multi-source residual stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_offset_source_identification(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before offset-source identification."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-offset-source-identification requires a terminal investigation."
        )

    multisource = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.multi-source-residual.interpret"
            and stage.status == "COMPLETE"
        ):
            multisource = stage.result
        if stage.handler_id.startswith("openstar.tess.offset-source-identification."):
            already_done = True

    if multisource is None:
        raise RuntimeError(
            "Run --continue-multi-source-residual first so v20.13 has the spatially decomposed offset component."
        )
    if multisource.get("recommendedNextTest") not in {
        "IDENTIFY_OFFSET_RESIDUAL_VARIABLE_SOURCE",
        "NEIGHBOR_SOURCE_IDENTIFICATION_AND_CATALOG_CROSSMATCH",
    }:
        raise RuntimeError(
            "v20.12 did not recommend offset/neighbor source identification for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.13 offset-source identification stages. "
            "Use --resume only if one is actually interrupted."
        )

def _can_continue_offset_source_variability(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before offset-source variability validation."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-offset-source-variability requires a terminal investigation."
        )

    offset_source = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.offset-source-identification.analyze"
            and stage.status == "COMPLETE"
        ):
            offset_source = stage.result
        if stage.handler_id.startswith("openstar.tess.offset-source-variability."):
            already_done = True

    if offset_source is None:
        raise RuntimeError(
            "Run --continue-offset-source-identification first so v20.14 has a catalog counterpart to validate."
        )
    if offset_source.get("recommendedNextTest") not in {
        "OFFSET_SOURCE_VARIABILITY_VALIDATION",
        "OFFSET_SOURCE_VARIABILITY_MATCH_TEST",
    }:
        raise RuntimeError(
            "v20.13 did not recommend direct offset-source variability validation for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.14 offset-source variability stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_calibrated_prf_deblending(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before calibrated PRF deblending."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-calibrated-prf-deblending requires a terminal investigation."
        )

    offset_variability = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.offset-source-variability.interpret"
            and stage.status == "COMPLETE"
        ):
            offset_variability = stage.result
        if stage.handler_id.startswith("openstar.tess.calibrated-prf-deblending."):
            already_done = True

    if offset_variability is None:
        raise RuntimeError(
            "Run --continue-offset-source-variability first so v20.15 has the v20.14 target/counterpart test."
        )
    if offset_variability.get("recommendedNextTest") != "CALIBRATED_PRF_SOURCE_DEBLENDING":
        raise RuntimeError(
            "v20.14 did not recommend CALIBRATED_PRF_SOURCE_DEBLENDING for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.15 calibrated-PRF deblending stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_difference_image_localization(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before difference-image localization."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-difference-image-localization requires a terminal investigation."
        )

    calibrated = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.calibrated-prf-deblending.interpret"
            and stage.status == "COMPLETE"
        ):
            calibrated = stage.result
        if stage.handler_id.startswith("openstar.tess.difference-image-localization."):
            already_done = True

    if calibrated is None:
        raise RuntimeError(
            "Run --continue-calibrated-prf-deblending first so v20.16 has the v20.15 deblend result."
        )
    if calibrated.get("recommendedNextTest") != "DIFFERENCE_IMAGE_SOURCE_LOCALIZATION":
        raise RuntimeError(
            "v20.15 did not recommend DIFFERENCE_IMAGE_SOURCE_LOCALIZATION for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.16 difference-image localization stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_frequency_localized_pixel_response(investigation) -> None:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before frequency-localized pixel response."
        )
    if investigation.status not in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        raise RuntimeError(
            "--continue-frequency-localized-pixel-response requires a terminal investigation."
        )

    difference_image = None
    already_done = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.difference-image-localization.interpret"
            and stage.status == "COMPLETE"
        ):
            difference_image = stage.result
        if stage.handler_id.startswith("openstar.tess.frequency-localized-pixel-response."):
            already_done = True

    if difference_image is None:
        raise RuntimeError(
            "Run --continue-difference-image-localization first so v20.17 has the v20.16 image result."
        )
    if difference_image.get("recommendedNextTest") != "FREQUENCY_LOCALIZED_PIXEL_RESPONSE_CONFIRMATION":
        raise RuntimeError(
            "v20.16 did not recommend FREQUENCY_LOCALIZED_PIXEL_RESPONSE_CONFIRMATION for this investigation."
        )
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.17 frequency-localized pixel-response stages. "
            "Use --resume only if one is actually interrupted."
        )


def _can_continue_official_spoc_prf_forward_modeling(investigation) -> bool:
    """
    Validate v20.18 continuation.

    Returns True only when retrying a previously FAILED v20.18 prepare stage.
    Failed terminal stages are preserved as immutable provenance; the retry is
    appended as a new stage with the next stage id.
    """
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before official SPOC PRF forward modeling."
        )

    frequency_localized = None
    completed_v20_18 = False
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.frequency-localized-pixel-response.interpret"
            and stage.status == "COMPLETE"
        ):
            frequency_localized = stage.result

        if (
            stage.handler_id.startswith(
                "openstar.tess.official-spoc-prf-forward-modeling."
            )
            and stage.status == "COMPLETE"
        ):
            completed_v20_18 = True

    if frequency_localized is None:
        raise RuntimeError(
            "Run --continue-frequency-localized-pixel-response first so v20.18 has the v20.17 localization result."
        )
    if frequency_localized.get("recommendedNextTest") != "OFFICIAL_SPOC_PRF_FORWARD_MODELING":
        raise RuntimeError(
            "v20.17 did not recommend OFFICIAL_SPOC_PRF_FORWARD_MODELING for this investigation."
        )
    if completed_v20_18:
        raise RuntimeError(
            "Investigation already contains completed v20.18 official SPOC PRF forward-modeling stages."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        existing_v20_18 = any(
            stage.handler_id.startswith(
                "openstar.tess.official-spoc-prf-forward-modeling."
            )
            for stage in investigation.stages
        )
        if existing_v20_18:
            raise RuntimeError(
                "Investigation already contains v20.18 official SPOC PRF forward-modeling stages."
            )
        return False

    if investigation.status == "FAILED":
        if not investigation.stages:
            raise RuntimeError(
                "FAILED investigation has no stage history to validate for v20.18 retry."
            )

        failed_stage = investigation.stages[-1]
        if (
            failed_stage.status != "FAILED"
            or failed_stage.handler_id
            != "openstar.tess.official-spoc-prf-forward-modeling.prepare"
        ):
            raise RuntimeError(
                "--continue-official-spoc-prf-forward-modeling can retry a FAILED investigation "
                "only when its most recent stage is the failed v20.18 prepare stage."
            )
        return True

    raise RuntimeError(
        "--continue-official-spoc-prf-forward-modeling requires a terminal investigation "
        "or a FAILED v20.18 prepare stage eligible for retry."
    )


def _can_continue_external_high_resolution_variability_validation(investigation) -> str:
    """
    Validate v20.19 continuation and return a recovery mode.

    Modes:
      NEW
      RETRY_PREPARE
      RETRY_RUN
      RETRY_INTERPRET

    Failed stages remain immutable provenance. Recovery always appends a new
    stage with the next stage id.
    """
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before external high-resolution validation."
        )

    official_spoc = None
    completed_interpretation = None
    existing_v20_19 = []
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.official-spoc-prf-forward-modeling.interpret"
            and stage.status == "COMPLETE"
        ):
            official_spoc = stage.result

        if stage.handler_id.startswith(
            "openstar.tess.external-high-resolution-variability-validation."
        ):
            existing_v20_19.append(stage)
            if (
                stage.handler_id
                == "openstar.tess.external-high-resolution-variability-validation.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if official_spoc is None:
        raise RuntimeError(
            "Run --continue-official-spoc-prf-forward-modeling first so v20.19 has the v20.18 result."
        )
    if official_spoc.get("recommendedNextTest") != "EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION":
        raise RuntimeError(
            "v20.18 did not recommend EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION for this investigation."
        )
    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.19 external high-resolution interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing_v20_19:
            raise RuntimeError(
                "Investigation already contains incomplete v20.19 stages but is terminal; "
                "refusing to append ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-external-high-resolution-variability-validation requires a terminal investigation "
            "or a FAILED v20.19 stage eligible for retry."
        )
    if not investigation.stages:
        raise RuntimeError(
            "FAILED investigation has no stage history to validate for v20.19 retry."
        )

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.19 recovery."
        )

    handler = failed_stage.handler_id
    if handler == "openstar.tess.external-high-resolution-variability-validation.prepare":
        return "RETRY_PREPARE"

    if handler == "openstar.tess.external-high-resolution-variability-validation.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id
                == "openstar.tess.external-high-resolution-variability-validation.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.19 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"

    if handler == "openstar.tess.external-high-resolution-variability-validation.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id
                == "openstar.tess.external-high-resolution-variability-validation.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.19 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-external-high-resolution-variability-validation can recover a FAILED investigation "
        "only when its most recent stage is a v20.19 prepare, run, or interpret stage."
    )



def _can_continue_skymapper_resolved_photometry(investigation) -> str:
    """
    Validate v20.20 continuation and return a recovery mode.

    Modes:
      NEW
      RETRY_PREPARE
      RETRY_RUN
      RETRY_INTERPRET

    Failed stages remain immutable provenance. Recovery appends a new stage.
    """
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before SkyMapper validation."
        )

    external = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.skymapper-resolved-photometry."
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.external-high-resolution-variability-validation.interpret"
            and stage.status == "COMPLETE"
        ):
            external = stage.result
        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id == "openstar.tess.skymapper-resolved-photometry.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if external is None:
        raise RuntimeError(
            "Run --continue-external-high-resolution-variability-validation first so v20.20 has the v20.19 result."
        )
    if external.get("recommendedNextTest") != "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY":
        raise RuntimeError(
            "v20.19 did not leave the investigation at the targeted high-resolution follow-up branch."
        )
    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.20 SkyMapper interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.20 stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-skymapper-resolved-photometry requires a terminal investigation or a FAILED v20.20 stage eligible for retry."
        )
    if not investigation.stages:
        raise RuntimeError("FAILED investigation has no stage history to validate for v20.20 retry.")

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.20 recovery."
        )

    handler = failed_stage.handler_id
    if handler == "openstar.tess.skymapper-resolved-photometry.prepare":
        return "RETRY_PREPARE"
    if handler == "openstar.tess.skymapper-resolved-photometry.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.skymapper-resolved-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.20 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"
    if handler == "openstar.tess.skymapper-resolved-photometry.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.skymapper-resolved-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.20 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-skymapper-resolved-photometry can recover a FAILED investigation only when its most recent stage is a v20.20 prepare, run, or interpret stage."
    )



def _can_continue_nsc_resolved_photometry(investigation) -> str:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before NSC validation."
        )

    skymapper = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.nsc-resolved-photometry."
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.skymapper-resolved-photometry.interpret"
            and stage.status == "COMPLETE"
        ):
            skymapper = stage.result
        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id == "openstar.tess.nsc-resolved-photometry.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if skymapper is None:
        raise RuntimeError(
            "Run --continue-skymapper-resolved-photometry first so v20.21 has the v20.20 result."
        )
    if skymapper.get("recommendedNextTest") != "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY":
        raise RuntimeError(
            "v20.20 did not leave the investigation at the targeted high-resolution follow-up branch."
        )
    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.21 NSC interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.21 stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-nsc-resolved-photometry requires a terminal investigation or a FAILED v20.21 stage eligible for retry."
        )
    if not investigation.stages:
        raise RuntimeError("FAILED investigation has no stage history to validate for v20.21 retry.")

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.21 recovery."
        )

    handler = failed_stage.handler_id
    if handler == "openstar.tess.nsc-resolved-photometry.prepare":
        return "RETRY_PREPARE"
    if handler == "openstar.tess.nsc-resolved-photometry.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.nsc-resolved-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.21 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"
    if handler == "openstar.tess.nsc-resolved-photometry.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.nsc-resolved-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.21 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-nsc-resolved-photometry can recover a FAILED investigation only when its most recent stage is a v20.21 prepare, run, or interpret stage."
    )



def _can_continue_noirlab_image_forced_photometry(investigation) -> str:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before NOIRLab image forced photometry."
        )

    nsc = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.noirlab-image-forced-photometry."
    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.nsc-resolved-photometry.interpret"
            and stage.status == "COMPLETE"
        ):
            nsc = stage.result
        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id == "openstar.tess.noirlab-image-forced-photometry.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if nsc is None:
        raise RuntimeError(
            "Run --continue-nsc-resolved-photometry first so v20.22 has the v20.21 result."
        )
    if nsc.get("recommendedNextTest") != "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY":
        raise RuntimeError(
            "v20.21 did not leave the investigation at the targeted high-resolution follow-up branch."
        )
    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.22 NOIRLab image forced-photometry interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.22 stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-noirlab-image-forced-photometry requires a terminal investigation or a FAILED v20.22 stage eligible for retry."
        )
    if not investigation.stages:
        raise RuntimeError("FAILED investigation has no stage history to validate for v20.22 retry.")

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.22 recovery."
        )

    handler = failed_stage.handler_id
    if handler == "openstar.tess.noirlab-image-forced-photometry.prepare":
        return "RETRY_PREPARE"
    if handler == "openstar.tess.noirlab-image-forced-photometry.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.noirlab-image-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.22 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"
    if handler == "openstar.tess.noirlab-image-forced-photometry.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.noirlab-image-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.22 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-noirlab-image-forced-photometry can recover a FAILED investigation only when its most recent stage is a v20.22 prepare, run, or interpret stage."
    )


def _can_continue_des_dr2_se_local_forced_photometry(investigation) -> str:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before DES DR2 source-local forced photometry."
        )

    noirlab = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.des-dr2-se-local-forced-photometry."

    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.noirlab-image-forced-photometry.interpret"
            and stage.status == "COMPLETE"
        ):
            noirlab = stage.result
        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id == "openstar.tess.des-dr2-se-local-forced-photometry.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if noirlab is None:
        raise RuntimeError(
            "Run --continue-noirlab-image-forced-photometry first so v20.23 has the completed v20.22 result."
        )
    if noirlab.get("recommendedNextTest") != "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY":
        raise RuntimeError(
            "v20.22 did not leave the investigation at the targeted high-resolution time-series branch."
        )
    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.23 DES DR2 source-local interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.23 stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-des-dr2-se-local-forced-photometry requires a terminal investigation or a FAILED v20.23 stage eligible for retry."
        )
    if not investigation.stages:
        raise RuntimeError("FAILED investigation has no stage history to validate for v20.23 retry.")

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.23 recovery."
        )

    handler = failed_stage.handler_id
    if handler == "openstar.tess.des-dr2-se-local-forced-photometry.prepare":
        return "RETRY_PREPARE"

    if handler == "openstar.tess.des-dr2-se-local-forced-photometry.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.des-dr2-se-local-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.23 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"

    if handler == "openstar.tess.des-dr2-se-local-forced-photometry.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.des-dr2-se-local-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.23 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-des-dr2-se-local-forced-photometry can recover a FAILED investigation only when its most recent stage is a v20.23 prepare, run, or interpret stage."
    )


def _atlas_credentials_available() -> bool:
    token = (os.environ.get("OPENSTAR_ATLAS_API_TOKEN") or "").strip()
    username = (os.environ.get("OPENSTAR_ATLAS_USERNAME") or "").strip()
    password = (os.environ.get("OPENSTAR_ATLAS_PASSWORD") or "").strip()
    return bool(token or (username and password))


def _can_continue_atlas_forced_photometry(investigation) -> str:
    # Credential preflight deliberately occurs before the caller changes the
    # investigation status. Missing credentials therefore leave the terminal
    # v20.23 investigation untouched.
    if not _atlas_credentials_available():
        raise RuntimeError(
            "v20.24 requires ATLAS forced-photometry credentials before changing investigation state. "
            "Set OPENSTAR_ATLAS_API_TOKEN, or set both OPENSTAR_ATLAS_USERNAME and OPENSTAR_ATLAS_PASSWORD."
        )

    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before ATLAS forced photometry."
        )

    des = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.atlas-forced-photometry."

    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.des-dr2-se-local-forced-photometry.interpret"
            and stage.status == "COMPLETE"
        ):
            des = stage.result

        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id == "openstar.tess.atlas-forced-photometry.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if des is None:
        raise RuntimeError(
            "Run --continue-des-dr2-se-local-forced-photometry first so v20.24 has the completed v20.23 result."
        )

    if des.get("recommendedNextTest") != "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY":
        raise RuntimeError(
            "v20.23 did not leave the investigation at the targeted high-resolution time-series branch."
        )

    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.24 ATLAS interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.24 stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-atlas-forced-photometry requires a terminal investigation or a FAILED v20.24 stage eligible for retry."
        )

    if not investigation.stages:
        raise RuntimeError(
            "FAILED investigation has no stage history to validate for v20.24 retry."
        )

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.24 recovery."
        )

    handler = failed_stage.handler_id
    if handler == "openstar.tess.atlas-forced-photometry.prepare":
        return "RETRY_PREPARE"

    if handler == "openstar.tess.atlas-forced-photometry.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.24 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"

    if handler == "openstar.tess.atlas-forced-photometry.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.24 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-atlas-forced-photometry can recover a FAILED investigation only when its most recent stage is a v20.24 prepare, run, or interpret stage."
    )


def _can_continue_atlas_forced_photometry_reanalysis(investigation) -> str:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before ATLAS reanalysis."
        )

    atlas = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.atlas-forced-photometry-reanalysis."

    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.atlas-forced-photometry.interpret"
            and stage.status == "COMPLETE"
        ):
            atlas = stage.result

        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id == "openstar.tess.atlas-forced-photometry-reanalysis.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if atlas is None:
        raise RuntimeError(
            "Run --continue-atlas-forced-photometry first so v20.25 has the completed v20.24 result."
        )

    if atlas.get("classification") != "ATLAS_NO_QUALIFYING_FORCED_PHOTOMETRY_TIME_SERIES":
        raise RuntimeError(
            "v20.25 is preregistered only for the completed v20.24 individual-SNR-gate correction branch."
        )

    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.25 ATLAS reanalysis interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.25 stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-atlas-forced-photometry-reanalysis requires a terminal investigation or a FAILED v20.25 stage eligible for retry."
        )

    if not investigation.stages:
        raise RuntimeError(
            "FAILED investigation has no stage history to validate for v20.25 retry."
        )

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.25 recovery."
        )

    handler = failed_stage.handler_id

    if handler == "openstar.tess.atlas-forced-photometry-reanalysis.prepare":
        return "RETRY_PREPARE"

    if handler == "openstar.tess.atlas-forced-photometry-reanalysis.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-forced-photometry-reanalysis.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.25 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"

    if handler == "openstar.tess.atlas-forced-photometry-reanalysis.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-forced-photometry-reanalysis.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.25 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-atlas-forced-photometry-reanalysis can recover a FAILED investigation only when its most recent stage is a v20.25 prepare, run, or interpret stage."
    )


def _can_continue_atlas_time_resolved(investigation) -> str:
    if any(stage.status == "RUNNING" for stage in investigation.stages):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. Use --resume before ATLAS time-resolved recurrence."
        )

    atlas_v20_25 = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.atlas-time-resolved."

    for stage in investigation.stages:
        if (
            stage.handler_id == "openstar.tess.atlas-forced-photometry-reanalysis.interpret"
            and stage.status == "COMPLETE"
        ):
            atlas_v20_25 = stage.result

        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id == "openstar.tess.atlas-time-resolved.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if atlas_v20_25 is None:
        raise RuntimeError(
            "Run --continue-atlas-forced-photometry-reanalysis first so v20.26 has the completed v20.25 result."
        )

    if atlas_v20_25.get("classification") != "ATLAS_REANALYSIS_SOURCE_ATTRIBUTION_UNRESOLVED":
        raise RuntimeError(
            "v20.26 is preregistered only for the completed unresolved v20.25 global ATLAS branch."
        )

    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed v20.26 ATLAS time-resolved interpretation."
        )

    if investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"}:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.26 stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-atlas-time-resolved requires a terminal investigation or a FAILED v20.26 stage eligible for retry."
        )

    if not investigation.stages:
        raise RuntimeError(
            "FAILED investigation has no stage history to validate for v20.26 retry."
        )

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; refusing ambiguous v20.26 recovery."
        )

    handler = failed_stage.handler_id

    if handler == "openstar.tess.atlas-time-resolved.prepare":
        return "RETRY_PREPARE"

    if handler == "openstar.tess.atlas-time-resolved.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-time-resolved.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None or not preparation.get("projectPath"):
            raise RuntimeError(
                "Cannot retry the failed v20.26 run because its completed preparation/projectPath is missing."
            )
        return "RETRY_RUN"

    if handler == "openstar.tess.atlas-time-resolved.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-time-resolved.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            ),
            None,
        )
        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.26 interpretation because its completed preparation is missing."
            )
        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-atlas-time-resolved can recover a FAILED investigation only when its most recent stage is a v20.26 prepare, run, or interpret stage."
    )


def _can_continue_atlas_fixed_window_recurrence(
    investigation,
) -> str:
    if any(
        stage.status == "RUNNING"
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. "
            "Use --resume before ATLAS fixed-window recurrence."
        )

    atlas_v20_26 = None
    completed_interpretation = None
    existing = []
    prefix = "openstar.tess.atlas-fixed-window."

    for stage in investigation.stages:
        if (
            stage.handler_id
            == "openstar.tess.atlas-time-resolved.interpret"
            and stage.status == "COMPLETE"
        ):
            atlas_v20_26 = stage.result

        if stage.handler_id.startswith(prefix):
            existing.append(stage)

            if (
                stage.handler_id
                == "openstar.tess.atlas-fixed-window.interpret"
                and stage.status == "COMPLETE"
            ):
                completed_interpretation = stage.result

    if atlas_v20_26 is None:
        raise RuntimeError(
            "Run --continue-atlas-time-resolved first so "
            "v20.27 has the completed v20.26 result."
        )

    if atlas_v20_26.get("classification") != (
        "ATLAS_TIME_RESOLVED_COUNTERPART_RECURRENCE_NOT_CONFIRMED"
    ):
        raise RuntimeError(
            "v20.27 is preregistered only for the completed "
            "v20.26 single-gap-season branch."
        )

    seasons = atlas_v20_26.get("seasons") or []
    if len(seasons) != 1:
        raise RuntimeError(
            "v20.27 is specifically the correction for v20.26 "
            "producing exactly one cadence-gap season."
        )

    if completed_interpretation is not None:
        raise RuntimeError(
            "Investigation already contains a completed "
            "v20.27 ATLAS fixed-window interpretation."
        )

    if investigation.status in {
        "COMPLETE",
        "HUMAN_REVIEW_REQUIRED",
    }:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.27 "
                "stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-atlas-fixed-window-recurrence requires "
            "a terminal investigation or a FAILED v20.27 stage "
            "eligible for retry."
        )

    if not investigation.stages:
        raise RuntimeError(
            "FAILED investigation has no stage history "
            "to validate for v20.27 retry."
        )

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; "
            "refusing ambiguous v20.27 recovery."
        )

    handler = failed_stage.handler_id

    if handler == "openstar.tess.atlas-fixed-window.prepare":
        return "RETRY_PREPARE"

    if handler == "openstar.tess.atlas-fixed-window.run":
        preparation = next(
            (
                stage.result
                for stage in reversed(
                    investigation.stages
                )
                if (
                    stage.handler_id
                    == "openstar.tess.atlas-fixed-window.prepare"
                    and stage.status == "COMPLETE"
                    and stage.result is not None
                )
            ),
            None,
        )

        if (
            preparation is None
            or not preparation.get("projectPath")
        ):
            raise RuntimeError(
                "Cannot retry the failed v20.27 run because "
                "its completed preparation/projectPath is missing."
            )

        return "RETRY_RUN"

    if handler == "openstar.tess.atlas-fixed-window.interpret":
        preparation = next(
            (
                stage.result
                for stage in reversed(
                    investigation.stages
                )
                if (
                    stage.handler_id
                    == "openstar.tess.atlas-fixed-window.prepare"
                    and stage.status == "COMPLETE"
                    and stage.result is not None
                )
            ),
            None,
        )

        if preparation is None:
            raise RuntimeError(
                "Cannot retry the failed v20.27 interpretation "
                "because its completed preparation is missing."
            )

        return "RETRY_INTERPRET"

    raise RuntimeError(
        "--continue-atlas-fixed-window-recurrence can recover "
        "a FAILED investigation only when its most recent stage "
        "is a v20.27 prepare, run, or interpret stage."
    )


def _can_continue_targeted_observation_planning(
    investigation,
) -> str:
    if any(
        stage.status == "RUNNING"
        for stage in investigation.stages
    ):
        raise RuntimeError(
            "Investigation contains a RUNNING stage. "
            "Use --resume before targeted observation planning."
        )

    atlas_fixed = None
    completed_plan = None
    existing = []
    prefix = "openstar.tess.targeted-observation-planning."

    for stage in investigation.stages:
        if (
            stage.handler_id
            == "openstar.tess.atlas-fixed-window.interpret"
            and stage.status == "COMPLETE"
        ):
            atlas_fixed = stage.result

        if stage.handler_id.startswith(prefix):
            existing.append(stage)
            if (
                stage.handler_id
                == "openstar.tess.targeted-observation-planning.generate"
                and stage.status == "COMPLETE"
            ):
                completed_plan = stage.result

    if atlas_fixed is None:
        raise RuntimeError(
            "Run --continue-atlas-fixed-window-recurrence first so "
            "v20.28 has the completed v20.27 result."
        )

    if atlas_fixed.get("recommendedNextTest") != (
        "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    ):
        raise RuntimeError(
            "v20.27 did not leave the investigation at targeted "
            "high-resolution time-series photometry."
        )

    if completed_plan is not None:
        raise RuntimeError(
            "Investigation already contains a completed "
            "v20.28 targeted observation plan."
        )

    if investigation.status in {
        "COMPLETE",
        "HUMAN_REVIEW_REQUIRED",
    }:
        if existing:
            raise RuntimeError(
                "Investigation already contains incomplete v20.28 "
                "stages but is terminal; refusing ambiguous duplicate history."
            )
        return "NEW"

    if investigation.status != "FAILED":
        raise RuntimeError(
            "--continue-targeted-observation-planning requires a terminal "
            "investigation or a FAILED v20.28 stage eligible for retry."
        )

    if not investigation.stages:
        raise RuntimeError(
            "FAILED investigation has no stage history "
            "to validate for v20.28 retry."
        )

    failed_stage = investigation.stages[-1]
    if failed_stage.status != "FAILED":
        raise RuntimeError(
            "FAILED investigation does not end in a FAILED stage; "
            "refusing ambiguous v20.28 recovery."
        )

    if (
        failed_stage.handler_id
        == "openstar.tess.targeted-observation-planning.generate"
    ):
        return "RETRY_GENERATE"

    raise RuntimeError(
        "--continue-targeted-observation-planning can recover a FAILED "
        "investigation only when its most recent stage is the "
        "v20.28 observation-plan generation stage."
    )


def main():
    args = parse_args()
    store = InvestigationStore(args.store)
    project_path = str(Path(args.project).expanduser().resolve())
    coordinator = OpenStarCoordinatorClient(args.coordinator)

    # Zero-compute harmonic-family reinterpretation is intentionally offline.
    # Every other path may need coordinator access, so verify connectivity
    # before mutating the investigation snapshot.
    health = None
    if not (
        args.continue_harmonic_family
        or args.continue_period_semantics
        or args.continue_morphology
        or args.continue_physical_interpretation
        or args.continue_source_localization
        or args.continue_offset_source_identification
        or args.continue_long_baseline_frequency_confirmation
        or args.continue_v20_8_long_baseline_time_frequency_confirmation
        or args.continue_confirmed_coherent_mode_identification
        or args.continue_transient_mode_validation
        or args.continue_neighbor_catalog_pixel_response_review
    ):
        health = coordinator.health()

    recovered_orphaned_status = False
    retrying_failed_official_spoc_prf_prepare = False
    external_high_resolution_recovery_mode = None
    skymapper_recovery_mode = None
    nsc_recovery_mode = None
    noirlab_forced_recovery_mode = None
    des_dr2_se_local_recovery_mode = None
    atlas_forced_recovery_mode = None
    atlas_reanalysis_recovery_mode = None
    atlas_time_resolved_recovery_mode = None
    atlas_fixed_window_recovery_mode = None
    targeted_observation_plan_recovery_mode = None

    if args.resume:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot resume investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        if investigation.workflow_version != WORKFLOW_VERSION:
            raise RuntimeError(
                "Cannot resume investigation with a different workflow version: "
                f"{investigation.workflow_version}"
            )

        investigation, interrupted_stage = (
            store.restart_current_running_stage(investigation)
        )
        initial_stage = StageRequest(
            id=interrupted_stage.id,
            handler_id=interrupted_stage.handler_id,
            parameters=dict(interrupted_stage.parameters),
            triggered_by_stage_id=(
                interrupted_stage.triggered_by_stage_id
            ),
        )
    elif args.continue_harmonic_family:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        recovered_orphaned_status = _can_continue_harmonic_family(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-reinterpret-harmonic-family",
            handler_id="openstar.tess.independent.harmonic-family.interpret",
            parameters={"continuation": True},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_period_semantics:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_period_semantics(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-period-semantics",
            handler_id="openstar.tess.period-semantics.reinterpret",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_morphology:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_morphology(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-morphology",
            handler_id="openstar.tess.morphology.analyze",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_physical_interpretation:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_physical_interpretation(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-physical-interpretation",
            handler_id="openstar.tess.physical.interpret",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_source_localization:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_source_localization(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-source-localization",
            handler_id="openstar.tess.source-localization.analyze",
            parameters={
                "evidenceLineage":
                "PHYSICAL_INTERPRETATION_PIXEL_LOCALIZATION",
            },
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_multimode:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_multimode(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-multimode-iteration-1",
            handler_id="openstar.tess.multimode.prepare",
            parameters={"iteration": 1},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_time_frequency:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_time_frequency(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-time-frequency",
            handler_id="openstar.tess.time-frequency.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_nonstationary:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_nonstationary(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-nonstationary",
            handler_id="openstar.tess.nonstationary.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_long_baseline_frequency_confirmation:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_long_baseline_frequency_confirmation(investigation)
        last_stage_id = next(
            stage.id for stage in reversed(investigation.stages)
            if stage.handler_id == "openstar.tess.mode-identification.analyze"
            and stage.status == "COMPLETE"
        )
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-long-baseline-frequency-confirmation",
            handler_id=LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID,
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_v20_8_long_baseline_time_frequency_confirmation:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_v20_8_long_baseline_time_frequency_confirmation(
            investigation
        )
        summary_stage = next(
            stage for stage in reversed(investigation.stages)
            if stage.handler_id == "openstar.tess.time-frequency.summarize"
            and stage.status == "COMPLETE"
        )
        transient_stage = next((
            stage for stage in reversed(investigation.stages)
            if stage.handler_id == TRANSIENT_MODE_VALIDATION_HANDLER_ID
            and stage.status == "COMPLETE"
        ), None)
        trigger = transient_stage or summary_stage
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=(
                f"{next_number:03d}-long-baseline-time-frequency-confirmation"
            ),
            handler_id=(
                V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
            ),
            parameters={},
            triggered_by_stage_id=trigger.id,
        )
    elif args.continue_confirmed_coherent_mode_identification:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_confirmed_coherent_mode_identification(investigation)
        confirmation = next(
            stage for stage in reversed(investigation.stages)
            if stage.handler_id
            == V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
            and stage.status == "COMPLETE"
        )
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-mode-identification",
            handler_id="openstar.tess.mode-identification.analyze",
            parameters={
                "evidenceLineage": (
                    V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE
                )
            },
            triggered_by_stage_id=confirmation.id,
        )
    elif args.continue_transient_mode_validation:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_transient_mode_validation(investigation)
        summary = next(
            stage for stage in reversed(investigation.stages)
            if stage.handler_id == "openstar.tess.time-frequency.summarize"
            and stage.status == "COMPLETE"
        )
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-transient-mode-validation",
            handler_id=TRANSIENT_MODE_VALIDATION_HANDLER_ID,
            parameters={},
            triggered_by_stage_id=summary.id,
        )
    elif args.continue_confirmed_nonstationary_mode_modeling:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError("Cannot continue investigation with a different workflow.")
        _can_continue_confirmed_nonstationary_mode_modeling(investigation)
        confirmation = next(stage for stage in reversed(investigation.stages)
                            if stage.handler_id == LONG_BASELINE_FREQUENCY_CONFIRMATION_HANDLER_ID
                            and stage.status == "COMPLETE")
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-confirmed-nonstationary",
            handler_id="openstar.tess.nonstationary.prepare",
            parameters={"evidenceLineage": CONFIRMED_NONSTATIONARY_EVIDENCE_LINEAGE},
            triggered_by_stage_id=confirmation.id,
        )
    elif args.continue_recurrent_residual_nonstationary_mode_modeling:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_recurrent_residual_nonstationary_mode_modeling(
            investigation
        )
        confirmation = next(
            stage for stage in reversed(investigation.stages)
            if stage.handler_id
            == V20_8_LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION_HANDLER_ID
            and stage.status == "COMPLETE"
        )
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=(
                f"{next_number:03d}-prepare-recurrent-residual-"
                "nonstationary"
            ),
            handler_id="openstar.tess.nonstationary.prepare",
            parameters={
                "evidenceLineage": (
                    RECURRENT_RESIDUAL_NONSTATIONARY_EVIDENCE_LINEAGE
                )
            },
            triggered_by_stage_id=confirmation.id,
        )
    elif args.continue_residual_mode_localization:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_residual_mode_localization(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-residual-mode-localization",
            handler_id="openstar.tess.residual-mode-localization.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_residual_external_evidence:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_residual_external_evidence(investigation)
        localization_stage_id = next(
            stage.id for stage in reversed(investigation.stages)
            if stage.handler_id
            == "openstar.tess.residual-mode-localization.interpret"
            and stage.status == "COMPLETE"
        )
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-residual-external-evidence",
            handler_id=RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID,
            parameters={},
            triggered_by_stage_id=localization_stage_id,
        )
    elif args.continue_target_residual_astrophysical_mechanism:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_target_residual_astrophysical_mechanism(investigation)
        external_stage_id = next(
            stage.id for stage in reversed(investigation.stages)
            if stage.handler_id == RESIDUAL_EXTERNAL_EVIDENCE_HANDLER_ID
            and stage.status == "COMPLETE"
        )
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-target-residual-astrophysical-mechanism",
            handler_id=TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_HANDLER_ID,
            parameters={},
            triggered_by_stage_id=external_stage_id,
        )
    elif args.continue_residual_mode_localization_review:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_residual_mode_localization_review(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-residual-mode-localization-review",
            handler_id="openstar.tess.residual-mode-localization-review.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_neighbor_catalog_pixel_response_review:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_neighbor_catalog_pixel_response_review(investigation)
        review_stage_id = next(
            stage.id for stage in reversed(investigation.stages)
            if stage.handler_id
            == "openstar.tess.residual-mode-localization-review.interpret"
            and stage.status == "COMPLETE"
        )
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-neighbor-catalog-pixel-response-review",
            handler_id=NEIGHBOR_CATALOG_PIXEL_RESPONSE_REVIEW_HANDLER_ID,
            parameters={},
            triggered_by_stage_id=review_stage_id,
        )
    elif args.continue_multi_source_residual:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_multi_source_residual(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-multi-source-residual",
            handler_id="openstar.tess.multi-source-residual.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_offset_source_identification:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_offset_source_identification(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-identify-offset-residual-source",
            handler_id="openstar.tess.offset-source-identification.analyze",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_offset_source_variability:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_offset_source_variability(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-offset-source-variability",
            handler_id="openstar.tess.offset-source-variability.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_calibrated_prf_deblending:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_calibrated_prf_deblending(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-calibrated-prf-deblending",
            handler_id="openstar.tess.calibrated-prf-deblending.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_difference_image_localization:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_difference_image_localization(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-difference-image-localization",
            handler_id="openstar.tess.difference-image-localization.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_frequency_localized_pixel_response:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_frequency_localized_pixel_response(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-frequency-localized-pixel-response",
            handler_id="openstar.tess.frequency-localized-pixel-response.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_official_spoc_prf_forward_modeling:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        retrying_failed_official_spoc_prf_prepare = (
            _can_continue_official_spoc_prf_forward_modeling(investigation)
        )
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-official-spoc-prf-forward-modeling",
            handler_id="openstar.tess.official-spoc-prf-forward-modeling.prepare",
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )
    elif args.continue_external_high_resolution_variability_validation:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        external_high_resolution_recovery_mode = (
            _can_continue_external_high_resolution_variability_validation(investigation)
        )
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")

        if external_high_resolution_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-external-high-resolution-variability-validation",
                handler_id="openstar.tess.external-high-resolution-variability-validation.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif external_high_resolution_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id
                == "openstar.tess.external-high-resolution-variability-validation.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-external-high-resolution-variability-validation",
                handler_id="openstar.tess.external-high-resolution-variability-validation.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id
                == "openstar.tess.external-high-resolution-variability-validation.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-external-high-resolution-variability-validation",
                handler_id="openstar.tess.external-high-resolution-variability-validation.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )
    elif args.continue_skymapper_resolved_photometry:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        skymapper_recovery_mode = _can_continue_skymapper_resolved_photometry(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")

        if skymapper_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-skymapper-resolved-photometry",
                handler_id="openstar.tess.skymapper-resolved-photometry.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif skymapper_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.skymapper-resolved-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-skymapper-resolved-photometry",
                handler_id="openstar.tess.skymapper-resolved-photometry.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id == "openstar.tess.skymapper-resolved-photometry.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-skymapper-resolved-photometry",
                handler_id="openstar.tess.skymapper-resolved-photometry.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )
    elif args.continue_nsc_resolved_photometry:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        nsc_recovery_mode = _can_continue_nsc_resolved_photometry(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")

        if nsc_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-nsc-resolved-photometry",
                handler_id="openstar.tess.nsc-resolved-photometry.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif nsc_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.nsc-resolved-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-nsc-resolved-photometry",
                handler_id="openstar.tess.nsc-resolved-photometry.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id == "openstar.tess.nsc-resolved-photometry.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-nsc-resolved-photometry",
                handler_id="openstar.tess.nsc-resolved-photometry.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )
    elif args.continue_noirlab_image_forced_photometry:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        noirlab_forced_recovery_mode = _can_continue_noirlab_image_forced_photometry(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")

        if noirlab_forced_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-noirlab-image-forced-photometry",
                handler_id="openstar.tess.noirlab-image-forced-photometry.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif noirlab_forced_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.noirlab-image-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-noirlab-image-forced-photometry",
                handler_id="openstar.tess.noirlab-image-forced-photometry.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id == "openstar.tess.noirlab-image-forced-photometry.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-noirlab-image-forced-photometry",
                handler_id="openstar.tess.noirlab-image-forced-photometry.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )
    elif args.continue_des_dr2_se_local_forced_photometry:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )

        des_dr2_se_local_recovery_mode = (
            _can_continue_des_dr2_se_local_forced_photometry(investigation)
        )
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")

        if des_dr2_se_local_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-des-dr2-se-local-forced-photometry",
                handler_id="openstar.tess.des-dr2-se-local-forced-photometry.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif des_dr2_se_local_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.des-dr2-se-local-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-des-dr2-se-local-forced-photometry",
                handler_id="openstar.tess.des-dr2-se-local-forced-photometry.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id == "openstar.tess.des-dr2-se-local-forced-photometry.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-des-dr2-se-local-forced-photometry",
                handler_id="openstar.tess.des-dr2-se-local-forced-photometry.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )

    elif args.continue_atlas_forced_photometry:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )

        atlas_forced_recovery_mode = _can_continue_atlas_forced_photometry(
            investigation
        )
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)

        # This mutation occurs only after credential/recovery validation.
        investigation = store.set_status(investigation, "RUNNING")

        if atlas_forced_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-atlas-forced-photometry",
                handler_id="openstar.tess.atlas-forced-photometry.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif atlas_forced_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-forced-photometry.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-atlas-forced-photometry",
                handler_id="openstar.tess.atlas-forced-photometry.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id == "openstar.tess.atlas-forced-photometry.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-atlas-forced-photometry",
                handler_id="openstar.tess.atlas-forced-photometry.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )

    elif args.continue_atlas_forced_photometry_reanalysis:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )

        atlas_reanalysis_recovery_mode = (
            _can_continue_atlas_forced_photometry_reanalysis(investigation)
        )
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")

        if atlas_reanalysis_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-atlas-forced-photometry-reanalysis",
                handler_id="openstar.tess.atlas-forced-photometry-reanalysis.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif atlas_reanalysis_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-forced-photometry-reanalysis.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-atlas-forced-photometry-reanalysis",
                handler_id="openstar.tess.atlas-forced-photometry-reanalysis.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id == "openstar.tess.atlas-forced-photometry-reanalysis.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-atlas-forced-photometry-reanalysis",
                handler_id="openstar.tess.atlas-forced-photometry-reanalysis.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )

    elif args.continue_atlas_time_resolved:
        investigation = store.load(args.investigation_id)

        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )

        atlas_time_resolved_recovery_mode = _can_continue_atlas_time_resolved(
            investigation
        )
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")

        if atlas_time_resolved_recovery_mode in {"NEW", "RETRY_PREPARE"}:
            initial_stage = StageRequest(
                id=f"{next_number:03d}-prepare-atlas-time-resolved",
                handler_id="openstar.tess.atlas-time-resolved.prepare",
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )
        elif atlas_time_resolved_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(investigation.stages)
                if stage.handler_id == "openstar.tess.atlas-time-resolved.prepare"
                and stage.status == "COMPLETE"
                and stage.result is not None
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-run-atlas-time-resolved",
                handler_id="openstar.tess.atlas-time-resolved.run",
                parameters={"projectPath": preparation["projectPath"]},
                triggered_by_stage_id=last_stage_id,
            )
        else:
            distributed_run_expected = any(
                stage.handler_id == "openstar.tess.atlas-time-resolved.run"
                and stage.status == "COMPLETE"
                for stage in investigation.stages
            )
            initial_stage = StageRequest(
                id=f"{next_number:03d}-interpret-atlas-time-resolved",
                handler_id="openstar.tess.atlas-time-resolved.interpret",
                parameters={"distributedRunExpected": distributed_run_expected},
                triggered_by_stage_id=last_stage_id,
            )

    elif args.continue_atlas_fixed_window_recurrence:
        investigation = store.load(
            args.investigation_id
        )

        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )

        atlas_fixed_window_recovery_mode = (
            _can_continue_atlas_fixed_window_recurrence(
                investigation
            )
        )

        last_stage_id = (
            investigation.stages[-1].id
            if investigation.stages
            else None
        )
        next_number = _next_stage_number(
            investigation
        )
        investigation = store.set_status(
            investigation,
            "RUNNING",
        )

        if atlas_fixed_window_recovery_mode in {
            "NEW",
            "RETRY_PREPARE",
        }:
            initial_stage = StageRequest(
                id=(
                    f"{next_number:03d}-"
                    "prepare-atlas-fixed-window"
                ),
                handler_id=(
                    "openstar.tess.atlas-fixed-window.prepare"
                ),
                parameters={},
                triggered_by_stage_id=last_stage_id,
            )

        elif atlas_fixed_window_recovery_mode == "RETRY_RUN":
            preparation = next(
                stage.result
                for stage in reversed(
                    investigation.stages
                )
                if (
                    stage.handler_id
                    == "openstar.tess.atlas-fixed-window.prepare"
                    and stage.status == "COMPLETE"
                    and stage.result is not None
                )
            )

            initial_stage = StageRequest(
                id=(
                    f"{next_number:03d}-"
                    "run-atlas-fixed-window"
                ),
                handler_id=(
                    "openstar.tess.atlas-fixed-window.run"
                ),
                parameters={
                    "projectPath": preparation[
                        "projectPath"
                    ],
                },
                triggered_by_stage_id=last_stage_id,
            )

        else:
            distributed_run_expected = any(
                (
                    stage.handler_id
                    == "openstar.tess.atlas-fixed-window.run"
                    and stage.status == "COMPLETE"
                )
                for stage in investigation.stages
            )

            initial_stage = StageRequest(
                id=(
                    f"{next_number:03d}-"
                    "interpret-atlas-fixed-window"
                ),
                handler_id=(
                    "openstar.tess.atlas-fixed-window.interpret"
                ),
                parameters={
                    "distributedRunExpected": (
                        distributed_run_expected
                    ),
                },
                triggered_by_stage_id=last_stage_id,
            )

    elif args.continue_targeted_observation_planning:
        investigation = store.load(
            args.investigation_id
        )

        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )

        targeted_observation_plan_recovery_mode = (
            _can_continue_targeted_observation_planning(
                investigation
            )
        )

        last_stage_id = (
            investigation.stages[-1].id
            if investigation.stages
            else None
        )
        next_number = _next_stage_number(
            investigation
        )
        investigation = store.set_status(
            investigation,
            "RUNNING",
        )

        initial_stage = StageRequest(
            id=(
                f"{next_number:03d}-"
                "generate-targeted-observation-plan"
            ),
            handler_id=(
                "openstar.tess.targeted-observation-planning.generate"
            ),
            parameters={},
            triggered_by_stage_id=last_stage_id,
        )

    elif args.continue_contradiction:
        investigation = store.load(args.investigation_id)
        if investigation.workflow_id != WORKFLOW_ID:
            raise RuntimeError(
                "Cannot continue investigation with a different workflow: "
                f"{investigation.workflow_id}"
            )
        _can_continue_contradiction(investigation)
        last_stage_id = investigation.stages[-1].id if investigation.stages else None
        next_number = _next_stage_number(investigation)
        investigation = store.set_status(investigation, "RUNNING")
        initial_stage = StageRequest(
            id=f"{next_number:03d}-prepare-broad-independent-search",
            handler_id="openstar.tess.independent.broad.prepare",
            parameters={"continuation": True},
            triggered_by_stage_id=last_stage_id,
        )
    else:
        investigation = store.create(
            args.investigation_id,
            WORKFLOW_ID,
            WORKFLOW_VERSION,
            metadata={
                "sourceProjectPath": project_path,
                "datasetID": args.dataset_id,
                "ticID": args.tic,
                "coordinator": args.coordinator,
            },
        )
        initial_stage = StageRequest(
            id="001-prepare-target",
            handler_id="openstar.tess.prepare-target",
            parameters={
                "projectPath": args.project,
                "datasetID": args.dataset_id,
                "ticID": args.tic,
            },
        )

    print("⭐ OpenStar TESS Investigation")
    print(f"Investigation: {investigation.id}")
    print(f"Workflow: {WORKFLOW_ID}")
    if (
        args.continue_harmonic_family
        or args.continue_period_semantics
        or args.continue_morphology
        or args.continue_physical_interpretation
        or args.continue_source_localization
        or args.continue_long_baseline_frequency_confirmation
        or args.continue_v20_8_long_baseline_time_frequency_confirmation
        or args.continue_confirmed_coherent_mode_identification
        or args.continue_transient_mode_validation
        or args.continue_neighbor_catalog_pixel_response_review
    ):
        print("Coordinator: not required for local/network non-distributed continuation")
    else:
        print(f"Coordinator: {args.coordinator}")
        print(f"Coordinator build: {(health or {}).get('build', 'unknown')}")

    if args.resume:
        print("↩️ Resuming interrupted investigation stage")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
    elif args.continue_harmonic_family:
        print("🧬 Continuing terminal investigation with v20.3.1 harmonic-family reinterpretation")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   prior distributed compute will not be rerun")
        if recovered_orphaned_status:
            print("   recovered orphaned RUNNING snapshot from failed v20.3.1 startup")
    elif args.continue_period_semantics:
        print("🧾 Continuing terminal investigation with v20.3.3 period semantics")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   evidence and distributed compute will not be changed")
    elif args.continue_morphology:
        print("🧬 Continuing terminal investigation with v20.4 morphology analysis")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   reusing frozen sector light curves; no MAST and no distributed compute")
    elif args.continue_physical_interpretation:
        print("🔬 Continuing terminal investigation with v20.5 physical-mechanism discrimination")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   reusing frozen sector light curves and existing identity metadata")
        print("   no MAST, no catalog network refresh, and no distributed compute")
    elif args.continue_source_localization:
        print("🎯 Continuing terminal investigation with v20.6 pixel-level source localization")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   MAST target-pixel/TESScut network access required")
        print("   no coordinator and no distributed workers required")
    elif args.continue_multimode:
        print("🎛️ Continuing terminal investigation with v20.7 distributed residual multi-mode decomposition")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   start Mac and iPhone workers before launching this command if you want both to contribute")
        print("   frozen sector light curves will be prewhitened locally; residual frequency grids run distributed")
    elif args.continue_time_frequency:
        print("🌊 Continuing terminal investigation with v20.8 distributed time-frequency evolution analysis")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   start Mac and iPhone workers before launching this command if you want both to contribute")
        print("   established family is fit locally; sliding-window residual frequency grids run distributed")
    elif args.continue_nonstationary:
        print("🌀 Continuing terminal investigation with v20.9 distributed long-baseline nonstationary modeling")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   worker executes ordinary Lomb-Scargle chunks on workflow-created time-warped residual datasets")
    elif args.continue_long_baseline_frequency_confirmation:
        print(
            "🔭 Continuing terminal investigation with v20.9.1 "
            "long-baseline frequency confirmation"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and distributed workers are not required")
        print("   no archive query or new data download is performed")
        print("   frozen primary and independent-sector light curves are reused")
        print("   each independent sector is held out without frequency or phase leakage")
    elif args.continue_v20_8_long_baseline_time_frequency_confirmation:
        print(
            "🔭 Continuing the terminal unresolved v20.8 boundary with "
            "v20.8.1 long-baseline time-frequency confirmation"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and distributed workers are not required")
        print("   no archive query or new data download is performed")
        print("   only frozen v20.8 residual-window datasets are reused")
        print("   original sector flux is not read")
        print("   each independent sector is held out without frequency or phase leakage")
    elif args.continue_confirmed_coherent_mode_identification:
        print(
            "🎼 Continuing the confirmed coherent v20.8.1 boundary with "
            "v20.8.2 mode identification"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and distributed workers are not required")
        print("   no archive query or new data download is performed")
        print("   frozen primary and confirmed independent sectors are reused")
        print("   the deterministic method contract is frozen before flux is read")
    elif args.continue_transient_mode_validation:
        print(
            "🫧 Continuing the resolved-cycle transient v20.8 boundary with "
            "v20.8.1 transient-mode validation"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and distributed workers are not required")
        print("   no archive query or new data download is performed")
        print("   only frozen v20.8 family-subtracted windows are reused")
        print("   original sector flux is not read")
        print("   held-out detection and control windows cannot select frequency or phase")
    elif args.continue_confirmed_nonstationary_mode_modeling:
        print("🌀 Continuing the confirmed v20.9.1 boundary with v20.9.2 nonstationary modeling")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and a compatible generic worker are required")
        print("   frozen datasets are reused; no archive query or download is performed")
    elif args.continue_recurrent_residual_nonstationary_mode_modeling:
        print(
            "🌀 Continuing the recurrent v20.8.2 boundary with "
            "v20.9.3 nonstationary modeling"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print(
            "   coordinator and a compatible generic worker are required"
        )
        print(
            "   only frozen family-subtracted residual windows are reused"
        )
        print("   original sector flux is not read")
        print("   no archive query or download is performed")
    elif args.continue_residual_mode_localization:
        print("🎯 Continuing terminal investigation with v20.10 distributed residual-mode pixel localization")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   MAST pixel download occurs during preparation; each usable pixel then runs as ordinary Lomb-Scargle work")
    elif args.continue_residual_external_evidence:
        print(
            "📚 Continuing the target-supported v20.10 boundary with v20.10.1 "
            "frozen external variability and binary evidence"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and distributed workers are not required")
        print("   no catalog query, archive query, or download is performed")
        print("   the completed TIC, SIMBAD, VSX, and Gaia evidence is reused")
        print("   discordant off-target residual sectors remain explicit cautions")
    elif args.continue_target_residual_astrophysical_mechanism:
        print(
            "🔬 Continuing the target-associated nonbinary v20.10.1 boundary "
            "with v20.10.2 mechanism-hypothesis adjudication"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and distributed workers are not required")
        print("   no flux values, catalog queries, archive queries, or downloads are used")
        print("   frozen catalog period comparisons and TIC stellar metadata are reused")
        print("   physical mechanism and claim level remain unresolved")
    elif args.continue_residual_mode_localization_review:
        print("🧭 Continuing terminal investigation with v20.11 distributed time-resolved residual-mode source localization review")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   MAST pixel download occurs during preparation; each usable pixel-window runs as ordinary Lomb-Scargle work")
    elif args.continue_neighbor_catalog_pixel_response_review:
        print(
            "🧭 Continuing the unresolved v20.11 boundary with v20.11.1 "
            "neighbor catalog and frozen pixel-response review"
        )
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator and distributed workers are not required")
        print("   fixed-radius TIC and Gaia DR3 catalog access is required")
        print("   no TESS download or flux read is performed")
        print("   persisted v20.11 power maps and sky Jacobians are reused")
    elif args.continue_multi_source_residual:
        print("🧩 Continuing terminal investigation with v20.12 distributed multi-source residual decomposition")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   MAST pixel preparation is domain-specific; decomposed source-component searches run as ordinary Lomb-Scargle work")
    elif args.continue_offset_source_identification:
        print("🔎 Continuing terminal investigation with v20.13 offset residual source identification")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator not required")
        print("   no distributed workers required")
        print("   TIC / Gaia DR3 / SIMBAD / VSX network access required")
    elif args.continue_offset_source_variability:
        print("🧪 Continuing terminal investigation with v20.14 distributed offset-source variability validation")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   TESS pixel preparation is catalog-guided; separated target/counterpart residual series run as ordinary Lomb-Scargle work")
    elif args.continue_calibrated_prf_deblending:
        print("🔬 Continuing terminal investigation with v20.15 distributed sector-calibrated pixel-response deblending")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   each TPF calibrates its source-response shape locally; separated series then run as generic Lomb-Scargle work")
    elif args.continue_difference_image_localization:
        print("🖼️ Continuing terminal investigation with v20.16 distributed difference-image source localization")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   generic Lomb-Scargle refines each sector residual frequency; TESS high-minus-low difference images are localized afterward")
    elif args.continue_frequency_localized_pixel_response:
        print("🎚️ Continuing terminal investigation with v20.17 distributed frequency-localized pixel-response confirmation")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   each usable cached TESS pixel becomes an ordinary narrow-band openstar.lomb-scargle.v1 dataset")
    elif args.continue_official_spoc_prf_forward_modeling:
        if retrying_failed_official_spoc_prf_prepare:
            print("🛰️ Retrying v20.18 official SPOC PRF forward modeling after failed preparation")
            print("   prior FAILED prepare stage is preserved as immutable provenance")
        else:
            print("🛰️ Continuing terminal investigation with v20.18 official SPOC PRF forward modeling")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   MAST official PRF + TPF downloads occur during preparation; separated source series run as generic openstar.lomb-scargle.v1 work")
    elif args.continue_external_high_resolution_variability_validation:
        if external_high_resolution_recovery_mode == "NEW":
            print("🔭 Continuing terminal investigation with v20.19 external high-resolution variability validation")
        else:
            print(
                "🔭 Retrying v20.19 external high-resolution variability validation "
                f"from {external_high_resolution_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required when Gaia epoch datasets are available")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   Gaia DR3 source-resolved epoch photometry is prepared locally; usable G-band series run as generic openstar.lomb-scargle.v1 work")
        print("   the TESS v20.9 drift law is not extrapolated backward into the Gaia observing epoch")
    elif args.continue_skymapper_resolved_photometry:
        if skymapper_recovery_mode == "NEW":
            print("🔎 Continuing terminal investigation with v20.20 SkyMapper DR4 resolved-photometry screen")
        else:
            print(
                "🔎 Retrying v20.20 SkyMapper DR4 resolved-photometry screen "
                f"from {skymapper_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required only when clean SkyMapper source datasets are prepared")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   only distinct SkyMapper objects plus clean, good-seeing per-image PSF measurements are admitted")
        print("   usable single-band series run as ordinary openstar.lomb-scargle.v1 work")
        print("   the TESS v20.9 drift law is not extrapolated into the SkyMapper observing epoch")
    elif args.continue_nsc_resolved_photometry:
        if nsc_recovery_mode == "NEW":
            print("🔎 Continuing terminal investigation with v20.21 NOIRLab Source Catalog DR2 resolved-photometry screen")
        else:
            print(
                "🔎 Retrying v20.21 NOIRLab Source Catalog DR2 resolved-photometry screen "
                f"from {nsc_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required only when qualifying NSC source datasets are prepared")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   the frozen pair must map to two distinct NSC objects")
        print("   only same-exposure/filter co-detections independently position-matched to both Gaia sources are admitted")
        print("   usable single-band series run as ordinary openstar.lomb-scargle.v1 work")
        print("   the TESS drift law is not extrapolated into the NSC observing epochs")
    elif args.continue_noirlab_image_forced_photometry:
        if noirlab_forced_recovery_mode == "NEW":
            print("🖼️ Continuing terminal investigation with v20.22 NOIRLab image-level forced two-source photometry")
        else:
            print(
                "🖼️ Retrying v20.22 NOIRLab image-level forced two-source photometry "
                f"from {noirlab_forced_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required only when qualifying source-band light curves are prepared")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   preparation queries public NSC DR2 SIA cutouts and fits both frozen Gaia positions directly in the image pixels")
        print("   saturation and source-separation quality guards are not relaxed")
        print("   usable source-band series run as ordinary openstar.lomb-scargle.v1 work")
        print("   the TESS drift law is not extrapolated into the NOIRLab observing epochs")
    elif args.continue_des_dr2_se_local_forced_photometry:
        if des_dr2_se_local_recovery_mode == "NEW":
            print("🌌 Continuing terminal investigation with v20.23 DES DR2 single-epoch source-local forced photometry")
        else:
            print(
                "🌌 Retrying v20.23 DES DR2 single-epoch source-local forced photometry "
                f"from {des_dr2_se_local_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required only when qualifying source-band light curves are prepared")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   target and counterpart are measured in independent local DES cutouts")
        print("   saturation of one source does not veto the other unless its local pixels are contaminated")
        print("   usable source-band series run as ordinary openstar.lomb-scargle.v1 work")
        print("   the TESS drift law is not extrapolated into the DES observing epochs")
    elif args.continue_atlas_forced_photometry:
        if atlas_forced_recovery_mode == "NEW":
            print("🌐 Continuing terminal investigation with v20.24 ATLAS source-resolved forced photometry")
        else:
            print(
                "🌐 Retrying v20.24 ATLAS source-resolved forced photometry "
                f"from {atlas_forced_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   ATLAS credentials are read only from OPENSTAR_ATLAS_* environment variables")
        print("   calibrated target-image forced photometry is requested at both frozen Gaia coordinates")
        print("   southern difference-image photometry is not used")
        print("   coordinator required only when qualifying nightly source-band light curves are prepared")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   usable nightly source-band series run as ordinary openstar.lomb-scargle.v1 work")
        print("   the TESS drift law is not extrapolated into the ATLAS observing epochs")
    elif args.continue_atlas_forced_photometry_reanalysis:
        if atlas_reanalysis_recovery_mode == "NEW":
            print("♻️ Continuing terminal investigation with v20.25 ATLAS signed forced-photometry reanalysis")
        else:
            print(
                "♻️ Retrying v20.25 ATLAS signed forced-photometry reanalysis "
                f"from {atlas_reanalysis_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   no new ATLAS query is performed")
        print("   immutable v20.24 raw photometry artifacts are reused")
        print("   individual positive detection SNR is NOT required before nightly binning")
        print("   signed quality-valid forced fluxes are retained")
        print("   coordinator required only when qualifying nightly source-band light curves are prepared")
        print("   usable nightly source-band series run as ordinary openstar.lomb-scargle.v1 work")
    elif args.continue_atlas_time_resolved:
        if atlas_time_resolved_recovery_mode == "NEW":
            print("🕰️ Continuing terminal investigation with v20.26 ATLAS time-resolved counterpart recurrence")
        else:
            print(
                "🕰️ Retrying v20.26 ATLAS time-resolved counterpart recurrence "
                f"from {atlas_time_resolved_recovery_mode}"
            )
            print("   prior FAILED stage is preserved as immutable provenance")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   no new ATLAS query is performed")
        print("   immutable v20.24 counterpart photometry is reused")
        print("   c/o measurements are split into shared independent observing seasons")
        print("   the prominence>=2.0 gate is unchanged")
        print("   coordinator required when qualifying season/filter light curves are prepared")
        print("   season/filter datasets run as ordinary openstar.lomb-scargle.v1 work")
        print("   the TESS drift law is not extrapolated into ATLAS observing epochs")
    elif args.continue_atlas_fixed_window_recurrence:
        if atlas_fixed_window_recovery_mode == "NEW":
            print(
                "🧱 Continuing terminal investigation with "
                "v20.27 ATLAS deterministic fixed-window recurrence"
            )
        else:
            print(
                "🧱 Retrying v20.27 ATLAS deterministic "
                "fixed-window recurrence "
                f"from {atlas_fixed_window_recovery_mode}"
            )
            print(
                "   prior FAILED stage is preserved "
                "as immutable provenance"
            )

        print(f"   stage: {initial_stage.id}")
        print(
            f"   handler: {initial_stage.handler_id}"
        )
        print("   no new ATLAS query is performed")
        print(
            "   immutable v20.24 counterpart photometry is reused"
        )
        print(
            "   fixed non-overlapping 180-day windows are anchored to absolute MJD zero"
        )
        print(
            "   the RELIABLE + prominence>=2.0 acceptance gate is unchanged"
        )
        print(
            "   search-grid boundary hits are rejected"
        )
        print(
            "   coordinator required when qualifying window/filter light curves are prepared"
        )
        print(
            "   each window/filter runs as ordinary openstar.lomb-scargle.v1 work"
        )
        print(
            "   the TESS drift law is not extrapolated into ATLAS observing epochs"
        )
    elif args.continue_targeted_observation_planning:
        if targeted_observation_plan_recovery_mode == "NEW":
            print(
                "🔭 Continuing terminal investigation with "
                "v20.28 targeted observation planning"
            )
        else:
            print(
                "🔭 Retrying v20.28 targeted observation planning "
                f"from {targeted_observation_plan_recovery_mode}"
            )
            print(
                "   prior FAILED stage is preserved "
                "as immutable provenance"
            )

        print(f"   stage: {initial_stage.id}")
        print(
            f"   handler: {initial_stage.handler_id}"
        )
        print("   no new archive query is performed")
        print("   no distributed science work is activated")
        print(
            "   the frozen residual-frequency band and acceptance rules are preserved"
        )
        print(
            "   campaign cadence, source-resolution, paired exposure tiers, filters, and ingest schema are preregistered"
        )
        print(
            "   the resulting artifacts are ready to hand to an observer or telescope scheduler"
        )
    elif args.continue_contradiction:
        print("🔁 Continuing terminal investigation with v20.3 contradiction resolution")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   completed prior stages will not be rerun")

    engine = build_engine(
        store,
        coordinator,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    final = engine.run(
        investigation,
        initial_stage,
        software_id=SOFTWARE_ID,
        software_version=SOFTWARE_VERSION,
        max_stages=100,
    )

    print()
    print("⭐ Investigation terminal")
    print(f"status: {final.status}")
    print(f"stages: {len(final.stages)}")
    print(f"record: {store.path_for(final.id)}")
    terminal_result = final.stages[-1].result if final.stages else None
    conclusion_path = (terminal_result or {}).get("conclusionPath")
    report_path = (terminal_result or {}).get("reportPath")
    print(f"conclusion: {conclusion_path or (store.directory_for(final.id) / 'conclusion.json')}")
    print(f"report: {report_path or (store.directory_for(final.id) / 'report.md')}")
    if terminal_result:
        claim = terminal_result.get("claim") or {}
        print("terminal claim:")
        print(json.dumps(claim, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
