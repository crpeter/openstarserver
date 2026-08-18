#!/usr/bin/env python3
"""Rank existing TESS sector-sweep evidence without performing any I/O remotely."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openstar_investigation import InvestigationStore
from workflows.tess.tess_sector_archive import TessSectorInventoryStore
from workflows.tess.tess_sector_ranking import (TessSectorRankingStore,
    aggregate_tess_sector_ranking, write_promotion_manifest)


def run_tess_sector_ranking(sector: int, state_dir: str | Path, output: str | Path | None = None,
                            promote_top: int | None = None, promotion_output: str | Path | None = None) -> int:
    root = Path(state_dir).expanduser().resolve()
    legacy = [name for name in ("lifecycle.json", "portfolio.json") if (root / name).exists()]
    if legacy:
        raise RuntimeError("TESS sector ranking refuses legacy single-lifecycle state: " + ", ".join(legacy))
    inventory_path = root / f"tess-sector-{sector}-inventory.json"
    inventory = TessSectorInventoryStore(inventory_path).load()
    if inventory.sector != sector: raise RuntimeError("Inventory sector does not match requested sector")
    ranking = aggregate_tess_sector_ranking(inventory, InvestigationStore(root / "investigations"))
    output_path = Path(output) if output else root / f"tess-sector-{sector}-ranking.json"
    TessSectorRankingStore(output_path).save(ranking)
    if promote_top is not None:
        promotion_path = Path(promotion_output) if promotion_output else root / f"tess-sector-{sector}-promoted-top-{promote_top}.json"
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
    parser.add_argument("--output"); parser.add_argument("--promote-top", type=int); parser.add_argument("--promotion-output")
    args = parser.parse_args(argv)
    if args.sector < 1: parser.error("--sector must be positive")
    if args.promote_top is not None and args.promote_top < 1: parser.error("--promote-top must be positive")
    if args.promotion_output and args.promote_top is None: parser.error("--promotion-output requires --promote-top")
    return args


def main(argv=None):
    args = parse_args(argv)
    try: return run_tess_sector_ranking(args.sector, args.state_dir, args.output, args.promote_top, args.promotion_output)
    except Exception as error:
        print(f"OpenStar TESS sector ranking: error={type(error).__name__}: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
