#!/usr/bin/env python3
"""Validate or run the manual untouched-sector time-domain evolution test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openstar_investigation import InvestigationStore
from openstar_state_storage import require_durable_state_path
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_period_family_time_domain_evolution import (
    HANDLER_PREFIX,
    admit_period_family_time_domain_evolution,
    verified_time_domain_evolution_boundary,
)


SOFTWARE_ID = "openstar.tess-manual-period-family-time-domain-evolution"
SOFTWARE_VERSION = "1"


def _existing_completed_result(store, investigation):
    stages = [stage for stage in investigation.stages
              if stage.handler_id.startswith(HANDLER_PREFIX)]
    if not stages:
        return None
    if len(investigation.stages) != 18:
        raise RuntimeError("Completed time-domain continuation is not the exact 18-stage boundary.")
    expected = [
        ("016-prepare-period-family-time-domain-evolution", HANDLER_PREFIX + "prepare"),
        ("017-run-period-family-time-domain-evolution", HANDLER_PREFIX + "run"),
        ("018-interpret-period-family-time-domain-evolution", HANDLER_PREFIX + "interpret"),
    ]
    if len(stages) != 3:
        raise RuntimeError("Time-domain continuation is partial; use explicit stage recovery.")
    for stage, (stage_id, handler_id) in zip(stages, expected):
        if not (stage.id == stage_id and stage.handler_id == handler_id
                and stage.status == "COMPLETE" and stage.result is not None
                and store.verified_terminal_stage_ledger_hash(investigation.id, stage)):
            raise RuntimeError(f"Completed time-domain ledger verification failed at {stage_id}.")
    for stage in investigation.stages:
        if store.verified_terminal_stage_ledger_hash(investigation.id, stage) is None:
            raise RuntimeError(f"Completed investigation ledger verification failed at {stage.id}.")
    return stages[-1].result


def run_manual_period_family_time_domain_evolution(
    *, state_dir: str | Path, investigation_id: str, execute: bool = False,
    allow_temporary_state: bool = False,
) -> int:
    root = require_durable_state_path(
        state_dir,
        allow_temporary_state=allow_temporary_state,
        label="TESS science state directory",
    )
    store = InvestigationStore(root / "investigations")
    investigation = store.load(investigation_id)
    existing = _existing_completed_result(store, investigation)
    if existing is not None:
        print("OpenStar manual period-family time-domain evolution:")
        print(f"investigation={investigation.id}")
        print("verified_stage_ledgers=18")
        print("disposition=VALIDATED_EXISTING_NO_CHANGES")
        print("latest_stage=018-interpret-period-family-time-domain-evolution")
        print(f"classification={existing.get('classification')}")
        print(f"claim={((existing.get('claimDecision') or {}).get('claim'))}")
        print(f"recommended_next_test={existing.get('recommendedNextTest')}")
        return 0

    frozen, hashes = verified_time_domain_evolution_boundary(store, investigation)
    print("OpenStar manual period-family time-domain evolution:")
    print(f"investigation={investigation.id}")
    print(f"tic={frozen['ticID']}")
    print(f"verified_stage_ledgers={len(hashes)}")
    print(f"untouched_sectors={','.join(str(value) for value in frozen['untouchedSectors'])}")
    print(f"campaigns={','.join(frozen['campaigns'])}")
    print("flux_products=SAP,PDCSAP")
    print("observable=gap-aware-ACF-plus-cycle-waveform-similarity")
    print("lomb_scargle=not_run")
    if not execute:
        print("disposition=VALIDATED_NO_CHANGES")
        print("next=rerun_with_--execute_to_append_stages_016_through_018")
        return 0

    admitted = admit_period_family_time_domain_evolution(store, investigation)
    selected = (admitted.metadata.get("controlState") or {}).get("selectedExperiment")
    if not isinstance(selected, dict):
        raise RuntimeError("Manual admission did not produce a selected experiment.")
    from openstar_workflow import StageRequest
    engine = build_engine(store, object(), poll_interval=1.0, timeout=None)
    completed = engine.run(
        admitted,
        StageRequest(**selected),
        software_id=SOFTWARE_ID,
        software_version=SOFTWARE_VERSION,
        max_stages=3,
    )
    interpretation = completed.stages[-1].result or {}
    print("disposition=APPENDED")
    print(f"status={completed.status}")
    print(f"latest_stage={completed.stages[-1].id}")
    print(f"classification={interpretation.get('classification')}")
    print(f"supporting_sectors={','.join(str(value) for value in interpretation.get('supportingSectors') or [])}")
    print(f"claim={((interpretation.get('claimDecision') or {}).get('claim'))}")
    print(f"recommended_next_test={interpretation.get('recommendedNextTest')}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--investigation-id", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Append and run stages 016-018. Without this flag, verify only and make no changes.",
    )
    parser.add_argument("--allow-temporary-state", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_manual_period_family_time_domain_evolution(
            state_dir=args.state_dir,
            investigation_id=args.investigation_id,
            execute=args.execute,
            allow_temporary_state=args.allow_temporary_state,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(
            "OpenStar manual period-family time-domain evolution: "
            f"error={type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
