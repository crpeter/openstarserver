#!/usr/bin/env python3
"""Create/resume a durable, shallow broad scan of one TESS sector."""

from __future__ import annotations

import argparse
import builtins
import os
import sys
from pathlib import Path

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStore
from openstar_scheduler import InvestigationScheduler
from openstar_science_runs import ScienceRunRecorder
from openstar_sector_sweep_status import sector_sweep_projection
from workflows.tess.tess_sector_archive import MastTessSectorArchiveProvider, TessSectorInventoryStore
from workflows.tess.tess_sector_scan import (
    WORKFLOW_ID, TessSectorScanTargetSource, plan_tess_sector_scan,
    register_tess_sector_scan_handlers, repair_legacy_archive_transport_failure,
)
from workflows.tess.tess_preprocessing import broad_tess_frequency_search

SOFTWARE_ID = "openstar.tess-sector-sweep-runner"
SOFTWARE_VERSION = "1"


def _install_perf_timing_filter() -> None:
    if os.getenv("OPENSTAR_PERF_TIMING") == "1":
        return

    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if args and str(args[0]).startswith("⏱️"):
            return
        original_print(*args, **kwargs)

    builtins.print = filtered_print


def run_tess_sector_sweep(sector: int, coordinator_url: str, state_dir: str | Path, *,
                          max_concurrent_investigations: int | None = None,
                          max_targets: int | None = None, provider=None,
                          coordinator=None, poll_interval: float = 1.0,
                          timeout: float | None = None,
                          frequencies_per_work_unit: int | None = None) -> int:
    if frequencies_per_work_unit is not None and frequencies_per_work_unit <= 0:
        raise ValueError("frequencies_per_work_unit must be positive")
    scan_profile = broad_tess_frequency_search()
    if frequencies_per_work_unit is not None:
        scan_profile["frequenciesPerWorkUnit"] = frequencies_per_work_unit
    root = Path(state_dir).expanduser().resolve()
    legacy = [
        name for name in ("lifecycle.json", "portfolio.json")
        if (root / name).exists()
    ]
    if legacy:
        raise RuntimeError(
            "TESS sector sweep refuses legacy single-lifecycle state: "
            + ", ".join(legacy)
        )
    root.mkdir(parents=True, exist_ok=True)
    provider = provider or MastTessSectorArchiveProvider()
    inventory = TessSectorInventoryStore(root / f"tess-sector-{sector}-inventory.json").create_or_load(sector, provider)
    store = InvestigationStore(root / "investigations")
    target_source = TessSectorScanTargetSource(inventory, max_targets=max_targets)
    for target in target_source.enumerate_targets():
        repair_legacy_archive_transport_failure(store, target.investigation_id)
    coordinator = coordinator or OpenStarCoordinatorClient(coordinator_url)
    workflow = register_tess_sector_scan_handlers(store, coordinator, provider,
                                                   poll_interval=poll_interval, timeout=timeout,
                                                   scan_profile=scan_profile)
    scheduler = InvestigationScheduler(store, InvestigationDispatcher(store, workflow),
        target_source,
        {WORKFLOW_ID: plan_tess_sector_scan}, software_id=SOFTWARE_ID,
        software_version=SOFTWARE_VERSION,
        max_concurrent_investigations=max_concurrent_investigations)
    result = scheduler.run_until_idle()
    counts: dict[str, int] = {}
    for outcome in result.outcomes: counts[outcome.state.value] = counts.get(outcome.state.value, 0) + 1
    summary = " ".join(f"{key.lower()}={counts[key]}" for key in sorted(counts))
    print(f"OpenStar TESS sector sweep: sector={sector} frequencies-per-work-unit={scan_profile['frequenciesPerWorkUnit']} inventory={len(inventory.entries)} admitted={len(result.outcomes)} {summary}")
    for outcome in result.outcomes:
        if outcome.error is not None:
            print(f"OpenStar TESS sector sweep target failure: investigation={outcome.investigation.id} error={type(outcome.error).__name__}: {outcome.error}", file=sys.stderr)
    return 1 if any(item.error is not None for item in result.outcomes) else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a shallow distributed scan of a TESS sector.")
    parser.add_argument("--sector", type=int, required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--max-concurrent-investigations", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--frequencies-per-work-unit", type=int)
    args = parser.parse_args(argv)
    for name in ("sector", "max_concurrent_investigations", "max_targets",
                 "frequencies_per_work_unit"):
        value = getattr(args, name)
        if value is not None and value < 1: parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.poll_interval <= 0: parser.error("--poll-interval must be positive")
    if args.timeout is not None and args.timeout <= 0: parser.error("--timeout must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.state_dir).expanduser().resolve()
    recorder = ScienceRunRecorder(
        kind="tess-sector-sweep",
        display_name=f"TESS Sector {args.sector} Sweep",
        state_root=root,
        workflow_id=WORKFLOW_ID,
        metadata={"mission": "TESS", "sector": args.sector},
        identity=str(args.sector),
    )
    try:
        code = run_tess_sector_sweep(args.sector, args.coordinator_url, root,
            max_concurrent_investigations=args.max_concurrent_investigations,
            max_targets=args.max_targets, poll_interval=args.poll_interval, timeout=args.timeout,
            frequencies_per_work_unit=args.frequencies_per_work_unit)
        projection = next(
            (item for item in sector_sweep_projection(root) if item["sector"] == args.sector),
            None,
        )
        status = "FAILED" if code else "FINISHED"
        if projection is not None and projection.get("status") == "COMPLETE":
            status = "COMPLETE"
        recorder.finish(
            status=status,
            summary={"exitCode": code, "sectorSweep": projection or {}},
        )
        return code
    except KeyboardInterrupt:
        recorder.interrupt()
        return 130
    except Exception as error:
        recorder.fail(error)
        print(f"OpenStar TESS sector sweep: error={type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    _install_perf_timing_filter()
    raise SystemExit(main())
