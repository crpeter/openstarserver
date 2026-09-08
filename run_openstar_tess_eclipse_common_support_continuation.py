#!/usr/bin/env python3
"""Verify and record an explicitly reviewed v2 eclipse localization, offline."""
import argparse
import json

from openstar_state_storage import require_durable_state_path
from workflows.tess.tess_eclipse_common_support_continuation import run_continuation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--investigation-id", required=True)
    parser.add_argument("--reanalysis-file", required=True)
    parser.add_argument("--reviewed-sha256", required=True,
                        help="Exact SHA-256 of the JSON artifact that was reviewed.")
    parser.add_argument("--execute", action="store_true",
                        help="Append the verified localization and conclusion to the idle investigation.")
    args = parser.parse_args(argv)
    result = run_continuation(
        require_durable_state_path(args.state_dir), args.investigation_id,
        require_durable_state_path(args.reanalysis_file), args.reviewed_sha256,
        execute=args.execute)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
