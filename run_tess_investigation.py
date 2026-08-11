from __future__ import annotations

import argparse
import json
from pathlib import Path

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import (
    SOFTWARE_ID,
    SOFTWARE_VERSION,
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    build_engine,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenStar v20.1 deterministic TESS investigation plugin. "
            "It derives a single-target project from an existing frozen project, "
            "runs distributed compute, resolves catalogs, evaluates hypotheses, "
            "and conditionally launches one decisive lower-frequency follow-up."
        )
    )
    parser.add_argument("--project", required=True, help="Existing frozen OpenStar project manifest.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--dataset-id")
    selector.add_argument("--tic", type=int)
    parser.add_argument("--investigation-id", required=True)
    parser.add_argument("--coordinator", default="http://127.0.0.1:8080")
    parser.add_argument("--store", default="data/investigations")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    store = InvestigationStore(args.store)
    project_path = str(Path(args.project).expanduser().resolve())
    investigation = store.create(
        args.investigation_id,
        WORKFLOW_ID,
        WORKFLOW_VERSION,
        metadata={
            "sourceProjectPath": project_path,
            "datasetID": args.dataset_id,
            "ticID": args.tic,
            "coordinator": args.coordinator,
        },
    )

    coordinator = OpenStarCoordinatorClient(args.coordinator)
    health = coordinator.health()
    print("⭐ OpenStar TESS Investigation")
    print(f"Investigation: {investigation.id}")
    print(f"Workflow: {WORKFLOW_ID}")
    print(f"Coordinator: {args.coordinator}")
    print(f"Coordinator build: {health.get('build', 'unknown')}")

    engine = build_engine(
        store,
        coordinator,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    final = engine.run(
        investigation,
        StageRequest(
            id="001-prepare-target",
            handler_id="openstar.tess.prepare-target",
            parameters={
                "projectPath": args.project,
                "datasetID": args.dataset_id,
                "ticID": args.tic,
            },
        ),
        software_id=SOFTWARE_ID,
        software_version=SOFTWARE_VERSION,
        max_stages=12,
    )

    print()
    print("⭐ Investigation terminal")
    print(f"status: {final.status}")
    print(f"stages: {len(final.stages)}")
    print(f"record: {store.path_for(final.id)}")
    print(f"conclusion: {store.directory_for(final.id) / 'conclusion.json'}")
    print(f"report: {store.directory_for(final.id) / 'report.md'}")
    if final.stages and final.stages[-1].result:
        claim = final.stages[-1].result.get("claim") or {}
        print("terminal claim:")
        print(json.dumps(claim, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
