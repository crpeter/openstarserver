"""Read-only projections of durable TESS sector-sweep state."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable

from openstar_investigation import Investigation, InvestigationStore
from openstar_lifecycle import InvestigationSchedulingState, persisted_scheduling_state
from workflows.tess.tess_sector_scan import WORKFLOW_ID


def _load_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sector_sweep_projection(state_dir: str | Path) -> list[dict[str, Any]]:
    """Project counters from the same durable records consumed by the scheduler."""
    root = Path(state_dir)
    investigations: list[Investigation] = []
    investigations_root = root / "investigations"
    if investigations_root.is_dir():
        for path in investigations_root.glob("*/investigation.json"):
            value = _load_object(path)
            if value is not None and value.get("workflow_id") == WORKFLOW_ID:
                try:
                    investigations.append(InvestigationStore.decode_snapshot(value))
                except (KeyError, TypeError, ValueError):
                    continue
    projections = []
    for inventory_path in root.glob("tess-sector-*-inventory.json"):
        inventory = _load_object(inventory_path)
        if inventory is None:
            continue
        try:
            sector = int(inventory["sector"])
        except (KeyError, TypeError, ValueError):
            continue
        entries = inventory.get("entries")
        inventory_count = len(entries) if isinstance(entries, list) else 0
        admitted_items = [
            item for item in investigations if item.metadata.get("sector") == sector
        ]
        complete = sum(item.status == "COMPLETE" for item in admitted_items)
        remaining = max(0, inventory_count - complete)
        states = [persisted_scheduling_state(item) for item in admitted_items]
        persisted_running = [
            any(stage.status == "RUNNING" for stage in item.stages)
            for item in admitted_items
        ]
        state_counts = {
            state: sum(item == state for item in states)
            for state in InvestigationSchedulingState
        }
        progress = complete / inventory_count if inventory_count else 0.0
        projections.append(
            {
                "sector": sector,
                "status": (
                    "COMPLETE" if inventory_count > 0 and remaining == 0 else "RUNNING"
                ),
                "inventory": inventory_count,
                "admitted": len(admitted_items),
                "complete": complete,
                "remaining": remaining,
                "recoveryRequired": state_counts[
                    InvestigationSchedulingState.RECOVERY_REQUIRED
                ],
                "inFlightOrRecovery": sum(persisted_running),
                "runnable": state_counts[InvestigationSchedulingState.RUNNABLE],
                "waitingExternalData": state_counts[
                    InvestigationSchedulingState.WAITING_EXTERNAL_DATA
                ],
                "blockedPrerequisites": state_counts[
                    InvestigationSchedulingState.BLOCKED_PREREQUISITES
                ],
                "failed": state_counts[InvestigationSchedulingState.FAILED],
                "unclassified": sum(
                    state is None and not running
                    for state, running in zip(states, persisted_running)
                ),
                "progress": progress,
            }
        )
    return sorted(
        projections, key=lambda item: (item["status"] == "COMPLETE", item["sector"])
    )


def sector_sweeps_projection(state_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [
        item for state_dir in state_dirs for item in sector_sweep_projection(state_dir)
    ]
