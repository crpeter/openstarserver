#!/usr/bin/env python3
"""Audit a completed conflicting eclipse localization without modifying its state."""
import argparse
import json

from openstar_state_storage import require_durable_state_path
from workflows.tess.tess_eclipse_localization_audit import run_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--investigation-id", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--execute", action="store_true", help="Reacquire frozen pixel products and write a separate diagnostic JSON.")
    args = parser.parse_args()
    state = require_durable_state_path(args.state_dir)
    output = require_durable_state_path(args.output_file)
    result = run_audit(state, args.investigation_id, output, execute=args.execute)
    print("Audit status:", result["status"])
    for sector in result.get("sectorResults", []):
        print(json.dumps({k: sector[k] for k in (
            "sector", "originalClassification", "outOfEventMinusCatalogTargetPixels",
            "differenceMinusOutOfEventPixels", "legacyMinusFixedApertureCentroidPixels",
            "fixedApertureSignedFractionalLoss", "empiricalImageShapeFit", "fluxBudget")}, indent=2))
    if args.execute:
        print("Audit:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
