#!/usr/bin/env python3
"""Rank existing TESS sector-sweep evidence without performing any I/O remotely."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openstar_investigation import InvestigationStore
from openstar_state_storage import require_durable_state_path
from workflows.tess.tess_sector_archive import TessSectorInventoryStore
from workflows.tess.tess_sector_ranking import (TessSectorRankingStore,
    aggregate_tess_sector_ranking, write_promotion_manifest)


def _sector_output_path(root: Path, requested: str | Path | None, default_name: str) -> Path:
    """Resolve an output and confine it to the sector-sweep state tree."""
    candidate = Path(requested).expanduser() if requested is not None else Path(default_name)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"TESS sector ranking output escapes sector-sweep state directory: {resolved}"
        )
    return resolved


def run_tess_sector_ranking(sector: int, state_dir: str | Path, output: str | Path | None = None,
                            promote_top: int | None = None, promotion_output: str | Path | None = None,
                            *, allow_temporary_state: bool = False) -> int:
    root = require_durable_state_path(state_dir, allow_temporary_state=allow_temporary_state)
    legacy = [name for name in ("lifecycle.json", "portfolio.json") if (root / name).exists()]
    if legacy:
        raise RuntimeError("TESS sector ranking refuses legacy single-lifecycle state: " + ", ".join(legacy))
    # Resolve and validate every requested destination together, before loading
    # state or constructing a store (which may create its root directory).
    output_path = _sector_output_path(root, output, f"tess-sector-{sector}-ranking.json")
    promotion_path = None
    if promote_top is not None:
        promotion_path = _sector_output_path(
            root, promotion_output, f"tess-sector-{sector}-promoted-top-{promote_top}.json"
        )
    inventory_path = root / f"tess-sector-{sector}-inventory.json"
    inventory = TessSectorInventoryStore(inventory_path).load()
    if inventory.sector != sector: raise RuntimeError("Inventory sector does not match requested sector")
    ranking = aggregate_tess_sector_ranking(inventory, InvestigationStore(root / "investigations"))
    TessSectorRankingStore(output_path).save(ranking)
    if promotion_path is not None:
        write_promotion_manifest(ranking, promote_top, promotion_path)
    c = ranking.content
    print("OpenStar TESS sector ranking:")
    print(f"sector={sector} inventory={c['inventoryCount']} complete={c['completedCount']} eligible={c['eligibleRankedCount']} remaining={c['remainingCount']} ranking_complete={str(c['rankingComplete']).lower()}")
    for entry in c["rankedEntries"][:10]:
        print(f"rank={entry['rank']} tic={entry['ticID']} confidence={entry['periodConfidence']} power={entry['bestPower']}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rank durable shallow TESS sector scans.")
    parser.add_argument("--sector", required=True, type=int); parser.add_argument("--state-dir", required=True)
    parser.add_argument("--allow-temporary-state", action="store_true")
    parser.add_argument("--output"); parser.add_argument("--promote-top", type=int); parser.add_argument("--promotion-output")
    args = parser.parse_args(argv)
    if args.sector < 1: parser.error("--sector must be positive")
    if args.promote_top is not None and args.promote_top < 1: parser.error("--promote-top must be positive")
    if args.promotion_output and args.promote_top is None: parser.error("--promotion-output requires --promote-top")
    return args


def main(argv=None):
    args = parse_args(argv)
    try: return run_tess_sector_ranking(args.sector, args.state_dir, args.output, args.promote_top, args.promotion_output,
        allow_temporary_state=args.allow_temporary_state)
    except Exception as error:
        print(f"OpenStar TESS sector ranking: error={type(error).__name__}: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
