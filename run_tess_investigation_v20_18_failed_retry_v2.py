from __future__ import annotations

import argparse
import json
from pathlib import Path

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import (
    SOFTWARE_ID,
    SOFTWARE_VERSION,
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    build_engine,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenStar v20.18 deterministic TESS investigation plugin. "
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
        "--continue-residual-mode-localization",
        action="store_true",
        help=(
            "Append v20.10 distributed pixel localization of the v20.9 drifting residual mode. "
            "The workflow prewhitens and time-warps TESS pixel light curves, then exposes each "
            "usable pixel as an ordinary openstar.lomb-scargle.v1 dataset."
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
    if already_done:
        raise RuntimeError(
            "Investigation already contains v20.10 residual-mode localization stages. "
            "Use --resume only if one is actually interrupted."
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
    ):
        health = coordinator.health()

    recovered_orphaned_status = False
    retrying_failed_official_spoc_prf_prepare = False

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
    elif args.continue_residual_mode_localization:
        print("🎯 Continuing terminal investigation with v20.10 distributed residual-mode pixel localization")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   MAST pixel download occurs during preparation; each usable pixel then runs as ordinary Lomb-Scargle work")
    elif args.continue_residual_mode_localization_review:
        print("🧭 Continuing terminal investigation with v20.11 distributed time-resolved residual-mode source localization review")
        print(f"   stage: {initial_stage.id}")
        print(f"   handler: {initial_stage.handler_id}")
        print("   coordinator required")
        print("   one compatible generic worker is sufficient; concurrency is not required")
        print("   MAST pixel download occurs during preparation; each usable pixel-window runs as ordinary Lomb-Scargle work")
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
        max_stages=90,
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
