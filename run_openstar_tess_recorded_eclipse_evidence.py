#!/usr/bin/env python3
"""Continue recorded v2 localization through the existing companion-evidence pipeline."""
import argparse
import json

from openstar_state_storage import require_durable_state_path
from workflows.tess.tess_recorded_eclipse_evidence import run_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--investigation-id", required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Run source review, photometry/depth checks, then the existing external-evidence chain.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry the last transient photometry or external-archive acquisition failure.")
    args = parser.parse_args(argv)
    result = run_evidence(require_durable_state_path(args.state_dir), args.investigation_id,
                          execute=args.execute, retry_failed=args.retry_failed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
