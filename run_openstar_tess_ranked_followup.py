#!/usr/bin/env python3
"""Refresh a sector ranking and schedule its durable deep-follow-up pool."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_dispatch import InvestigationDispatcher
from openstar_external_jobs import ExternalJobMonitor, ExternalJobStore, apply_external_job_wakeups
from openstar_investigation import InvestigationStore
from openstar_scheduler import InvestigationScheduler
from openstar_science_runs import recorded_science_run
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    plan_tess_branches,
    register_tess_workflow_handlers,
    repair_obsolete_terminal_wait,
)
from workflows.tess.tess_ranked_followup import (
    TessDeepAdmissionStore, TessNoveltyScreenStore, TessRankedFollowupTargetSource,
    shallow_primary_for_screen, verified_reusable_primary,
)
from workflows.tess.tess_identity import collect_identity
from workflows.tess.tess_sector_archive import TessSectorInventoryStore
from workflows.tess.tess_sector_ranking import TessSectorRankingStore, aggregate_tess_sector_ranking

SOFTWARE_ID = "openstar.tess-ranked-followup-runner"
SOFTWARE_VERSION = "1"


def validate_state_roots(sector_state_dir: str | Path, deep_state_dir: str | Path) -> tuple[Path, Path]:
    sector_root = Path(sector_state_dir).expanduser().resolve()
    deep_root = Path(deep_state_dir).expanduser().resolve()
    if sector_root == deep_root or sector_root.is_relative_to(deep_root) or deep_root.is_relative_to(sector_root):
        raise RuntimeError("Sector and deep state directories must be separate non-overlapping trees")
    for root in (sector_root, deep_root):
        # A child directory is not an isolation boundary from legacy Blind C
        # state.  Resolve first, then inspect the root and every containing
        # directory so symlinks cannot bypass the guard.
        for container in (root, *root.parents):
            legacy = [
                name
                for name in ("lifecycle.json", "portfolio.json")
                if (container / name).exists()
            ]
            if legacy:
                raise RuntimeError(
                    "Ranked follow-up refuses state nested in legacy state at "
                    f"{container}: {', '.join(legacy)}"
                )
    return sector_root, deep_root


def _claim(investigation) -> str | None:
    for stage in reversed(investigation.stages):
        result = stage.result
        if isinstance(result, dict):
            claim = result.get("claim")
            if isinstance(claim, dict): claim = claim.get("claim")
            if isinstance(claim, str): return claim
    return None


@recorded_science_run("tess-ranked-followup", "deep_state_dir",
                      logical_identity="sector", metadata=("sector", "sector_state_dir"))
def run_tess_ranked_followup(sector: int, sector_state_dir: str | Path,
                             deep_state_dir: str | Path, coordinator_url: str,
                             promote_top: int | None = None, *, promote_novel: int | None = None,
                             known_period_validation_quota: int = 0,
                             max_concurrent_investigations: int | None = None,
                             poll_interval: float = 1.0, timeout: float | None = None,
                             coordinator=None) -> int:
    if sector < 1: raise ValueError("sector must be positive")
    if (promote_top is None) == (promote_novel is None):
        raise ValueError("select exactly one of promote_top or promote_novel")
    if promote_top is not None and promote_top < 1: raise ValueError("promote_top must be positive")
    if promote_novel is not None and promote_novel < 1: raise ValueError("promote_novel must be positive")
    if known_period_validation_quota < 0: raise ValueError("known quota cannot be negative")
    if promote_top is not None and known_period_validation_quota:
        raise ValueError("known quota is only valid with promote_novel")
    sector_root, deep_root = validate_state_roots(sector_state_dir, deep_state_dir)
    inventory = TessSectorInventoryStore(sector_root / f"tess-sector-{sector}-inventory.json").load()
    if inventory.sector != sector: raise RuntimeError("Inventory sector does not match requested sector")
    ranking = aggregate_tess_sector_ranking(inventory, InvestigationStore(sector_root / "investigations"))
    # This is the sole intended write to shallow state.
    TessSectorRankingStore(sector_root / f"tess-sector-{sector}-ranking.json").save(ranking)
    ledger = TessDeepAdmissionStore(deep_root / f"tess-sector-{sector}-deep-admissions.json", sector)
    novelty_stats = {key: 0 for key in ("novelty_screened_this_run", "novel_candidates_found",
        "known_period_matches_screened", "catalog_coverage_incomplete")}
    if promote_top is not None:
        admissions, new, excluded = ledger.admit(ranking, promote_top)
    else:
        shallow_store = InvestigationStore(sector_root / "investigations")
        screen = TessNoveltyScreenStore(
            deep_root / f"tess-sector-{sector}-novelty-screening.json", sector)
        selected, novelty_stats = screen.select(ranking, promote_novel,
            known_period_validation_quota, {item.ticID for item in ledger.load()},
            collect_identity, lambda entry: shallow_primary_for_screen(shallow_store, entry))
        admissions, new, excluded = ledger.admit_selected(ranking, selected)

    store = InvestigationStore(deep_root / "investigations")
    shallow_store = InvestigationStore(sector_root / "investigations")
    # Only absent investigations receive new reuse metadata. Existing deep
    # state is immutable even when an old admission ledger is loaded.
    reusable = {}
    for admission in admissions:
        if not store.path_for(admission.deepInvestigationID).exists():
            verified = verified_reusable_primary(shallow_store, admission)
            if verified is not None:
                reusable[admission.deepInvestigationID] = verified
    # Apply only TESS's narrow durable compatibility migrations before the
    # domain-neutral scheduler classifies already-admitted investigations.
    for admission in admissions:
        path = store.path_for(admission.deepInvestigationID)
        if path.exists():
            repair_obsolete_terminal_wait(store, store.load(admission.deepInvestigationID))
    jobs = ExternalJobStore(deep_root / "external-jobs")
    if jobs.pending():
        from workflows.tess.tess_atlas_forced_photometry import ATLASExternalJobProvider
        ExternalJobMonitor(jobs, {"atlas-forced-photometry": ATLASExternalJobProvider()}).poll_due()
    apply_external_job_wakeups(store, jobs.ready_dependencies())
    coordinator = coordinator or OpenStarCoordinatorClient(coordinator_url)
    workflow = register_tess_workflow_handlers(store, coordinator, poll_interval=poll_interval, timeout=timeout)
    scheduler = InvestigationScheduler(store, InvestigationDispatcher(store, workflow),
        TessRankedFollowupTargetSource(admissions, reusable), {WORKFLOW_ID: plan_tess_branches},
        software_id=SOFTWARE_ID, software_version=SOFTWARE_VERSION,
        max_concurrent_investigations=max_concurrent_investigations)
    result = scheduler.run_until_idle()
    counts = {name: 0 for name in ("COMPLETE", "RUNNABLE", "WAITING_EXTERNAL_DATA", "BLOCKED_PREREQUISITES", "FAILED")}
    for outcome in result.outcomes: counts[outcome.state.value] = counts.get(outcome.state.value, 0) + 1
    print("OpenStar ranked TESS follow-up:")
    print(f"sector={sector}")
    print(f"ranked={ranking.content['eligibleRankedCount']}")
    print(f"admitted={len(admissions)}")
    print(f"new_admissions={len(new)}")
    for key, value in novelty_stats.items(): print(f"{key}={value}")
    print(f"new_novel_admissions={sum(a.admissionBasis == 'NOVEL_PRIORITY' for a in new)}")
    print(f"new_known_validation_admissions={sum(a.admissionBasis == 'KNOWN_PERIOD_VALIDATION' for a in new)}")
    for key in ("complete", "runnable", "waiting_external", "blocked", "failed"):
        state = {"waiting_external": "WAITING_EXTERNAL_DATA", "blocked": "BLOCKED_PREREQUISITES"}.get(key, key.upper())
        print(f"{key}={counts.get(state, 0)}")
    outcomes = {item.investigation.id: item for item in result.outcomes}
    for admission in sorted(admissions, key=lambda item: (item.admittedRankingRank, item.ticID)):
        outcome = outcomes[admission.deepInvestigationID]
        line = f"rank={admission.admittedRankingRank} tic={admission.ticID} state={outcome.state.value}"
        if admission.admissionBasis: line += f" admission_basis={admission.admissionBasis}"
        claim = _claim(outcome.investigation)
        if claim: line += f" claim={claim}"
        print(line)
    for item in excluded: print(f"tic={item['ticID']} state=EXCLUDED reason={item['reason']}")
    return 1 if any(item.error is not None for item in result.outcomes) else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run durable ranked TESS deep follow-up.")
    parser.add_argument("--sector", type=int, required=True)
    parser.add_argument("--sector-state-dir", required=True); parser.add_argument("--deep-state-dir", required=True)
    parser.add_argument("--coordinator-url", required=True)
    promotion = parser.add_mutually_exclusive_group(required=True)
    promotion.add_argument("--promote-top", type=int)
    promotion.add_argument("--promote-novel", type=int)
    parser.add_argument("--known-period-validation-quota", type=int, default=0)
    parser.add_argument("--max-concurrent-investigations", type=int)
    parser.add_argument("--poll-interval", type=float, default=1.0); parser.add_argument("--timeout", type=float)
    args = parser.parse_args(argv)
    for name in ("sector", "promote_top", "promote_novel", "max_concurrent_investigations", "poll_interval", "timeout"):
        value = getattr(args, name)
        if value is not None and value <= 0: parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.known_period_validation_quota < 0:
        parser.error("--known-period-validation-quota cannot be negative")
    if args.promote_top is not None and args.known_period_validation_quota:
        parser.error("--known-period-validation-quota requires --promote-novel")
    return args


def main(argv=None):
    args = parse_args(argv)
    try: return run_tess_ranked_followup(args.sector, args.sector_state_dir, args.deep_state_dir,
        args.coordinator_url, args.promote_top, promote_novel=args.promote_novel,
        known_period_validation_quota=args.known_period_validation_quota,
        max_concurrent_investigations=args.max_concurrent_investigations,
        poll_interval=args.poll_interval, timeout=args.timeout)
    except KeyboardInterrupt: return 130
    except Exception as error:
        print(f"OpenStar ranked TESS follow-up: error={type(error).__name__}: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
