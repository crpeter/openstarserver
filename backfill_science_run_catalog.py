#!/usr/bin/env python3
"""Explicit, bounded, read-only discovery of historical science roots."""

from __future__ import annotations

import argparse
from pathlib import Path

from openstar_science_runs import MODULE_ROOT, backfill_science_runs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", dest="roots",
                        help="Root whose direct children may contain science state (repeatable).")
    parser.add_argument("--catalog")
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not args.roots:
        args.roots = ["/tmp", "/private/tmp", str(MODULE_ROOT / "data")]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    count = backfill_science_runs((Path(root) for root in args.roots), args.catalog,
                                  limit=args.limit)
    print(f"OpenStar science-run backfill: registered={count} inspected-limit={args.limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
