#!/usr/bin/env python3
"""Validate or run the manual unresolved-period-family localization experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openstar_investigation import InvestigationStore
from openstar_state_storage import require_durable_state_path
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_period_family_difference_image import (
    admit_period_family_difference_imaging,
    verified_period_family_boundary,
)


SOFTWARE_ID = "openstar.tess-manual-period-family-localization"
SOFTWARE_VERSION = "1"


def run_manual_period_family_localization(
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
    frozen, hashes = verified_period_family_boundary(store, investigation)
    sectors = [item["sector"] for item in frozen["independentSectorDetections"]]
    print("OpenStar manual period-family localization:")
    print(f"investigation={investigation.id}")
    print(f"tic={frozen['ticID']}")
    print(f"verified_stage_ledgers={len(hashes)}")
    print(f"independent_sectors={','.join(str(value) for value in sectors)}")
    print("period_search=not_repeated")
    if not execute:
        print("disposition=VALIDATED_NO_CHANGES")
        print("next=rerun_with_--execute_to_append_stages_013_through_015")
        return 0

    admitted = admit_period_family_difference_imaging(store, investigation)
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
        help="Append and run stages 013-015. Without this flag, verify only and make no changes.",
    )
    parser.add_argument("--allow-temporary-state", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_manual_period_family_localization(
            state_dir=args.state_dir,
            investigation_id=args.investigation_id,
            execute=args.execute,
            allow_temporary_state=args.allow_temporary_state,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(
            f"OpenStar manual period-family localization: error={type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
