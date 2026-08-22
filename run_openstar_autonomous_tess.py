#!/usr/bin/env python3
"""Production entry point for the durable autonomous TESS lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_external_jobs import (
    ExternalJobMonitor,
    ExternalJobStore,
    apply_external_job_wakeups,
)
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStore
from openstar_lifecycle import InvestigationLifecycleLoop, LifecycleResult
from openstar_scheduler import InvestigationScheduler
from openstar_science_runs import ScienceRunRecorder
from openstar_targets import InvestigationTargetPortfolio, NoEligibleTargetError
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    TessInvestigationTargetSource,
    plan_tess_branches,
    register_tess_workflow_handlers,
    repair_obsolete_terminal_wait,
)

SOFTWARE_ID = "openstar.autonomous-tess-runner"
SOFTWARE_VERSION = "1"


def _status(result: LifecycleResult) -> str:
    investigation = result.investigation
    fields = [
        f"disposition={result.disposition}",
        f"investigation={investigation.id}",
        f"status={investigation.status}",
        f"stages={len(investigation.stages)}",
        f"transitions={result.transitions}",
    ]
    selection = investigation.metadata.get("targetSelection")
    if isinstance(selection, dict) and selection.get("selectionID"):
        fields.append(f"selection={selection['selectionID']}")
    if investigation.stages:
        stage = investigation.stages[-1]
        fields.extend(
            (f"latest={stage.id}:{stage.status}", f"handler={stage.handler_id}")
        )
        if stage.provenance is not None:
            fields.append(
                f"software={stage.provenance.software_id}@{stage.provenance.software_version}"
            )
            if stage.provenance.project_ids:
                fields.append(f"projects={','.join(stage.provenance.project_ids)}")
    return "OpenStar lifecycle: " + " ".join(fields)


def run_autonomous_tess(
    project_paths: list[str | Path],
    coordinator_url: str,
    state_dir: str | Path,
    *,
    poll_interval: float = 1.0,
    timeout: float | None = None,
    multi_investigation: bool = False,
    max_concurrent_investigations: int | None = None,
) -> int:
    """Construct and run the existing lifecycle, returning a process exit code."""
    root = Path(state_dir).expanduser().resolve()
    if multi_investigation:
        legacy = [
            name
            for name in ("lifecycle.json", "portfolio.json")
            if (root / name).exists()
        ]
        if legacy:
            raise RuntimeError(
                "Multi-investigation mode refuses legacy lifecycle state: "
                + ", ".join(legacy)
            )
    root.mkdir(parents=True, exist_ok=True)
    store = InvestigationStore(root / "investigations")
    external_jobs = ExternalJobStore(root / "external-jobs")
    if external_jobs.pending():
        from workflows.tess.tess_atlas_forced_photometry import ATLASExternalJobProvider

        ExternalJobMonitor(
            external_jobs,
            {
                "atlas-forced-photometry": ATLASExternalJobProvider(),
            },
        ).poll_due()
    # Reconstruct from durable terminal records even if a process crashed
    # after the last job completed but before its investigation was awakened.
    apply_external_job_wakeups(store, external_jobs.ready_dependencies())
    coordinator = OpenStarCoordinatorClient(coordinator_url)
    workflow = register_tess_workflow_handlers(
        store, coordinator, poll_interval=poll_interval, timeout=timeout
    )
    dispatcher = InvestigationDispatcher(store, workflow)
    source = TessInvestigationTargetSource(project_paths)
    if multi_investigation:
        scheduler = InvestigationScheduler(
            store,
            dispatcher,
            source,
            {WORKFLOW_ID: plan_tess_branches},
            software_id=SOFTWARE_ID,
            software_version=SOFTWARE_VERSION,
            max_concurrent_investigations=max_concurrent_investigations,
        )
        result = scheduler.run_until_idle()
        for outcome in result.outcomes:
            fields = [
                f"investigation={outcome.investigation.id}",
                f"state={outcome.state.value}",
                f"status={outcome.investigation.status}",
                f"stages={len(outcome.investigation.stages)}",
            ]
            if outcome.error is not None:
                fields.append(f"error={type(outcome.error).__name__}: {outcome.error}")
            print("OpenStar scheduler: " + " ".join(fields))
        return 1 if any(outcome.error is not None for outcome in result.outcomes) else 0

    lifecycle_path = root / "lifecycle.json"
    portfolio = InvestigationTargetPortfolio(root / "portfolio.json", store, dispatcher)
    lifecycle = InvestigationLifecycleLoop(
        lifecycle_path,
        store,
        dispatcher,
        portfolio,
        source,
        {WORKFLOW_ID: plan_tess_branches},
        software_id=SOFTWARE_ID,
        software_version=SOFTWARE_VERSION,
    )

    if not lifecycle_path.exists():
        try:
            target, provenance = portfolio.select_initial(source)
        except NoEligibleTargetError:
            print("OpenStar lifecycle: disposition=NO_ELIGIBLE_TARGETS")
            return 0
        target = replace(
            target,
            metadata={**(target.metadata or {}), "targetSelection": provenance},
        )
        lifecycle.start(target)
        current_investigation_id = target.investigation_id
        print(
            "OpenStar lifecycle: disposition=STARTED "
            f"target={target.id} investigation={target.investigation_id}"
        )
    else:
        persisted = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        target = persisted.get("currentTarget", {})
        current_investigation_id = str(target.get("investigation_id", ""))
        print(
            "OpenStar lifecycle: disposition=RESUMING "
            f"target={target.get('id', 'unknown')} "
            f"investigation={target.get('investigation_id', 'unknown')}"
        )

    # Compatibility migration for the one obsolete decision emitted by the
    # previous TESS adapter. The repair predicate is deliberately TESS-specific
    # and leaves all other durable lifecycle actions untouched.
    if current_investigation_id:
        investigation = store.load(current_investigation_id)
        repair_obsolete_terminal_wait(store, investigation)

    while True:
        result = lifecycle.run()
        print(_status(result))
        if result.disposition == "LIFECYCLE_CHECKPOINT":
            continue
        return 2 if result.disposition == "EXPERIMENT_RECOVERY_REQUIRED" else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch or resume autonomous TESS investigations."
    )
    parser.add_argument(
        "--project", action="append", required=True, help="TESS project manifest path."
    )
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--multi-investigation", action="store_true")
    parser.add_argument("--max-concurrent-investigations", type=int)
    args = parser.parse_args(argv)
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (
        args.max_concurrent_investigations is not None
        and args.max_concurrent_investigations <= 0
    ):
        parser.error("--max-concurrent-investigations must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.state_dir).expanduser().resolve()
    recorder = ScienceRunRecorder(
        kind="autonomous-investigation",
        display_name="Autonomous TESS Investigation",
        state_root=root,
        workflow_id=WORKFLOW_ID,
        metadata={
            "mission": "TESS",
            "projects": [str(Path(path).expanduser()) for path in args.project],
            "multiInvestigation": args.multi_investigation,
        },
    )
    try:
        code = run_autonomous_tess(
            args.project,
            args.coordinator_url,
            root,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            multi_investigation=args.multi_investigation,
            max_concurrent_investigations=args.max_concurrent_investigations,
        )
        status = "ATTENTION_REQUIRED" if code == 2 else ("FAILED" if code else "FINISHED")
        recorder.finish(status=status, summary={"exitCode": code})
        return code
    except KeyboardInterrupt:
        recorder.interrupt()
        print("OpenStar lifecycle: disposition=SHUTDOWN", file=sys.stderr)
        return 130
    except Exception as error:
        recorder.fail(error)
        print(f"OpenStar lifecycle: disposition=ERROR error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
