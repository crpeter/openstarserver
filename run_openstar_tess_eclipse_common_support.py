#!/usr/bin/env python3
"""Reanalyze a verified v1 eclipse-localization conflict on common pixel support."""
import argparse
import json

from openstar_state_storage import require_durable_state_path
from workflows.tess.tess_eclipse_common_support import run_reanalysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--investigation-id", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Reacquire frozen pixels and publish a separate v2 result for review.")
    args = parser.parse_args(argv)
    result = run_reanalysis(
        require_durable_state_path(args.state_dir), args.investigation_id,
        require_durable_state_path(args.output_file), execute=args.execute)
    print("Reanalysis status:", result["status"])
    if args.execute:
        print("Classification:", result["classification"])
        print("Investigation modified:", result["investigationModified"])
        for sector in result["sectorResults"]:
            print(json.dumps({key: sector.get(key) for key in (
                "sector", "originalClassification", "classification", "usable",
                "measuredPixelCentroid", "centroidUncertaintyPixels", "catalogDistances",
                "qualityRejectionReasons")}, indent=2))
        print("Reanalysis:", args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
