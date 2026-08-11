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
            "Run the OpenStar v20.4 deterministic TESS investigation plugin. "
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

def main():
    args = parse_args()
    store = InvestigationStore(args.store)
    project_path = str(Path(args.project).expanduser().resolve())
    coordinator = OpenStarCoordinatorClient(args.coordinator)

    # Zero-compute harmonic-family reinterpretation is intentionally offline.
    # Every other path may need coordinator access, so verify connectivity
    # before mutating the investigation snapshot.
    health = None
    if not (args.continue_harmonic_family or args.continue_period_semantics or args.continue_morphology):
        health = coordinator.health()

    recovered_orphaned_status = False

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
    if args.continue_harmonic_family or args.continue_period_semantics or args.continue_morphology:
        print("Coordinator: not required for local/zero-distributed-compute continuation")
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
        max_stages=28,
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
