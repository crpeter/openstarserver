#!/usr/bin/env python3
"""Explicitly register an immutable TESS sector reconstruction for observability."""

from __future__ import annotations

import argparse
from pathlib import Path

from openstar_science_runs import register_sector_reconstruction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, help="override the science-run catalog")
    args = parser.parse_args()
    run_id = register_sector_reconstruction(args.manifest, args.catalog)
    print(f"Registered TESS sector reconstruction: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
