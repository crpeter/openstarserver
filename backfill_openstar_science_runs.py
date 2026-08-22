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


def backfill_sector_sweeps(
    catalog: ScienceRunCatalog,
    search_roots: Iterable[str | Path],
    *,
    active: dict[Path, int] | None = None,
) -> list[dict]:
    active = local_active_sector_sweeps() if active is None else active
    registered = []
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
        if active.get(state_root) == sector:
            status = "RUNNING"
        elif projection.get("status") == "COMPLETE":
            status = "COMPLETE"
        else:
            status = "DISCOVERED_INCOMPLETE"
        run_id = science_run_id(
            "tess-sector-sweep", state_root, identity=str(sector)
        )
        registered.append(
            catalog.register(
                run_id,
                kind="tess-sector-sweep",
                display_name=f"TESS Sector {sector} Sweep",
                status=status,
                state_root=state_root,
                workflow_id=WORKFLOW_ID,
                metadata={"mission": "TESS", "sector": sector, "backfilled": True},
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
