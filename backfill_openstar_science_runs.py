#!/usr/bin/env python3
"""Backfill pre-catalog OpenStar science runs without modifying science state."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from openstar_science_runs import ScienceRunCatalog, science_run_id
from openstar_sector_sweep_status import sector_sweep_projection
from workflows.tess.tess_sector_scan import WORKFLOW_ID

ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_ROOTS = (Path("/tmp"), ROOT / "data")


def _option(tokens: list[str], name: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        prefix = name + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def active_sector_sweep_processes(commands: Iterable[str]) -> dict[Path, int]:
    """Return state roots proven active by a local sector-sweep process command."""
    active: dict[Path, int] = {}
    for command in commands:
        if "run_openstar_tess_sector_sweep.py" not in command:
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        state_dir = _option(tokens, "--state-dir")
        sector_raw = _option(tokens, "--sector")
        if state_dir is None or sector_raw is None:
            continue
        try:
            sector = int(sector_raw)
        except ValueError:
            continue
        active[Path(state_dir).expanduser().resolve()] = sector
    return active


def local_active_sector_sweeps() -> dict[Path, int]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}
    return active_sector_sweep_processes(result.stdout.splitlines())


def discover_sector_inventory_paths(search_roots: Iterable[str | Path]) -> list[Path]:
    """Find likely science-run roots without walking investigation subtrees."""
    found: set[Path] = set()
    patterns = (
        "tess-sector-*-inventory.json",
        "*/tess-sector-*-inventory.json",
        "*/*/tess-sector-*-inventory.json",
    )
    for raw_root in search_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                for path in root.glob(pattern):
                    if path.is_file():
                        found.add(path.resolve())
            except OSError:
                continue
    return sorted(found)


def _number(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _candidate_score(candidate: dict) -> tuple[int, int, int, int, int, str]:
    """Prefer the active run, then the most substantial durable run for a sector."""
    projection = candidate["projection"]
    return (
        1 if candidate["active"] else 0,
        1 if projection.get("status") == "COMPLETE" else 0,
        _number(projection.get("inventory")),
        _number(projection.get("admitted")),
        _number(projection.get("complete")),
        str(candidate["state_root"]),
    )


def backfill_sector_sweeps(
    catalog: ScienceRunCatalog,
    search_roots: Iterable[str | Path],
    *,
    active: dict[Path, int] | None = None,
) -> list[dict]:
    """Register one canonical pre-catalog sector run per sector.

    Old development/smoke roots are ambiguous because they predate the catalog. Prefer an
    actually active process when one exists; otherwise retain the most substantial persisted
    run. Future instrumented runs are never collapsed because they self-register explicitly.
    """
    active = local_active_sector_sweeps() if active is None else active
    candidates_by_sector: dict[int, list[dict]] = {}
    for inventory_path in discover_sector_inventory_paths(search_roots):
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            sector = int(inventory["sector"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        state_root = inventory_path.parent.resolve()
        projection = next(
            (
                item
                for item in sector_sweep_projection(state_root)
                if item.get("sector") == sector
            ),
            None,
        )
        if projection is None:
            continue
        candidates_by_sector.setdefault(sector, []).append(
            {
                "sector": sector,
                "state_root": state_root,
                "projection": projection,
                "active": active.get(state_root) == sector,
            }
        )

    winners = {
        sector: max(candidates, key=_candidate_score)
        for sector, candidates in candidates_by_sector.items()
    }
    winner_ids = {
        sector: science_run_id(
            "tess-sector-sweep", winner["state_root"], identity=str(sector)
        )
        for sector, winner in winners.items()
    }

    # Clean up only ambiguous pre-catalog entries created by this backfill. Never delete
    # instrumented runs or any authoritative investigation/science state.
    for existing in catalog.list_runs():
        metadata = existing.get("metadata")
        if (
            existing.get("kind") == "tess-sector-sweep"
            and isinstance(metadata, dict)
            and metadata.get("backfilled") is True
        ):
            try:
                sector = int(metadata.get("sector"))
            except (TypeError, ValueError):
                sector = None
            if sector is None or existing.get("id") != winner_ids.get(sector):
                catalog.delete(existing["id"])

    registered = []
    for sector in sorted(winners):
        winner = winners[sector]
        state_root = winner["state_root"]
        projection = winner["projection"]
        active_at_backfill = winner["active"]
        if active_at_backfill:
            status = "DISCOVERED_ACTIVE"
        elif projection.get("status") == "COMPLETE":
            status = "COMPLETE"
        else:
            status = "DISCOVERED_INCOMPLETE"
        registered.append(
            catalog.register(
                winner_ids[sector],
                kind="tess-sector-sweep",
                display_name=f"TESS Sector {sector} Sweep",
                status=status,
                state_root=state_root,
                workflow_id=WORKFLOW_ID,
                metadata={
                    "mission": "TESS",
                    "sector": sector,
                    "backfilled": True,
                    "activeAtBackfill": active_at_backfill,
                },
                summary={"sectorSweep": projection},
            )
        )
    return registered


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Discover and register pre-catalog OpenStar science runs."
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Root to scan for historical science state (repeatable). Defaults to /tmp and data.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    roots = args.search_root or [str(path) for path in DEFAULT_SEARCH_ROOTS]
    registered = backfill_sector_sweeps(ScienceRunCatalog(), roots)
    for run in registered:
        print(
            f"OpenStar science run backfill: {run['displayName']} status={run['status']}"
        )
    print(f"OpenStar science run backfill: registered={len(registered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
