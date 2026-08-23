from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import (
    InvestigationStore,
    sha256_file,
)
from openstar_workflow import (
    StageOutcome,
    StageRequest,
    WorkflowEngine,
)
from openstar_science_runs import ScienceRunRecorder


WORKFLOW_ID = "openstar.workflow.project-smoke.v1"
WORKFLOW_VERSION = "20.0"
SOFTWARE_ID = "openstar.workflow-engine"
SOFTWARE_VERSION = "20.0"


def previous_stage_result(investigation) -> dict[str, Any]:
    if len(investigation.stages) < 2:
        raise RuntimeError("No previous completed stage is available.")
    previous = investigation.stages[-2]
    if previous.status != "COMPLETE" or previous.result is None:
        raise RuntimeError("Previous stage is not complete.")
    return previous.result


def build_engine(
    store: InvestigationStore,
    coordinator: OpenStarCoordinatorClient,
    *,
    poll_interval: float,
    timeout: float | None,
) -> WorkflowEngine:
    engine = WorkflowEngine(store)

    def prepare_project(investigation, request):
        project_path = Path(request.parameters["projectPath"]).expanduser().resolve()
        if not project_path.exists():
            raise FileNotFoundError(project_path)

        manifest_hash = sha256_file(project_path)
        return StageOutcome(
            result={
                "projectPath": str(project_path),
                "projectManifestSha256": manifest_hash,
            },
            next_stage=StageRequest(
                id="002-distributed-project",
                handler_id="openstar.project.run",
                parameters={
                    "projectPath": str(project_path),
                    "projectManifestSha256": manifest_hash,
                },
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"projectManifest": manifest_hash},
        )

    def run_project(investigation, request):
        project_path = request.parameters["projectPath"]
        manifest_hash = request.parameters["projectManifestSha256"]

        run = coordinator.run_project(
            project_path,
            poll_interval=poll_interval,
            timeout=timeout,
        )

        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id="003-terminal-check",
                handler_id="generic.project.terminal-check",
                parameters={
                    "expectedProjectID": run.project_id,
                },
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"projectManifest": manifest_hash},
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def terminal_check(investigation, request):
        project_result = previous_stage_result(investigation)
        expected = str(request.parameters["expectedProjectID"])
        actual = str(project_result.get("projectID"))

        total = int(project_result.get("projectTotalWorkUnits") or 0)
        completed = int(project_result.get("projectCompletedWorkUnits") or 0)
        failed = int(project_result.get("projectFailedWorkUnits") or 0)

        passed = (
            actual == expected
            and total > 0
            and completed + failed == total
        )

        return StageOutcome(
            result={
                "passed": passed,
                "projectID": actual,
                "completedWorkUnits": completed,
                "failedWorkUnits": failed,
                "totalWorkUnits": total,
                "rule": "projectID matches and completed+failed == total",
            },
            stop=True,
            final_status="COMPLETE" if passed else "REVIEW_REQUIRED",
            project_ids=(actual,),
        )

    engine.register_handler("local.project.prepare", prepare_project)
    engine.register_handler("openstar.project.run", run_project)
    engine.register_handler("generic.project.terminal-check", terminal_check)
    return engine


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the v20 generic OpenStar investigation smoke workflow. "
            "The workflow activates one ordinary project, waits for it, "
            "records provenance, and applies a deterministic terminal rule."
        )
    )
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--investigation-id",
        required=True,
        help="Stable ID for the investigation output directory.",
    )
    parser.add_argument(
        "--coordinator",
        default="http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--store",
        default="data/investigations",
    )
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional project timeout in seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    store = InvestigationStore(args.store)
    recorder = ScienceRunRecorder("generic-investigation", args.store, metadata={
        "investigationID": args.investigation_id}, logical_identity=args.investigation_id)
    with recorder:
        return _run(args, store)


def _run(args, store: InvestigationStore):
    investigation = store.create(
        args.investigation_id,
        WORKFLOW_ID,
        WORKFLOW_VERSION,
        metadata={
            "projectPath": str(Path(args.project).expanduser().resolve()),
            "coordinator": args.coordinator,
        },
    )

    coordinator = OpenStarCoordinatorClient(args.coordinator)
    health = coordinator.health()
    print("⭐ OpenStar Investigation")
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
            id="001-prepare-project",
            handler_id="local.project.prepare",
            parameters={"projectPath": args.project},
        ),
        software_id=SOFTWARE_ID,
        software_version=SOFTWARE_VERSION,
    )

    print()
    print("🏁 Investigation terminal")
    print(f"status: {final.status}")
    print(f"stages: {len(final.stages)}")
    print(f"record: {store.path_for(final.id)}")

    last = final.stages[-1]
    if last.result is not None:
        print("terminal result:")
        print(json.dumps(last.result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
