from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from openstar_autonomy import ExternalDataDependency, ScientificBranch
from openstar_investigation import Investigation, InvestigationStore
from openstar_targets import InvestigationTarget
from openstar_workflow import StageRequest, WorkflowEngine

# Kept identical to the public identifiers in tess_investigation.  Importing that
# module eagerly would also import optional numerical/astronomy dependencies,
# even when a server is only enumerating targets.
WORKFLOW_ID = "openstar.workflow.tess-investigation.v1"
WORKFLOW_VERSION = "20.2"


def _stable_id(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-")
    if not normalized:
        raise ValueError("TESS target identifiers must not be empty.")
    return normalized


class TessInvestigationTargetSource:
    """Expose datasets in persisted TESS project manifests to the portfolio."""

    id = "openstar.tess-project-targets"
    version = "1"

    def __init__(self, project_paths: Sequence[str | Path]):
        self.project_paths = tuple(
            Path(path).expanduser().resolve() for path in project_paths
        )

    def enumerate_targets(self) -> tuple[InvestigationTarget, ...]:
        targets: list[InvestigationTarget] = []
        for project_path in sorted(self.project_paths, key=str):
            with project_path.open("r", encoding="utf-8") as handle:
                project = json.load(handle)
            project_id = project.get("id")
            datasets = project.get("datasets")
            if not project_id or not isinstance(datasets, list):
                raise ValueError(f"Not an OpenStar project manifest: {project_path}")

            for position, dataset in enumerate(datasets):
                if not isinstance(dataset, dict) or not dataset.get("id"):
                    raise ValueError(f"Invalid dataset entry in {project_path}")
                dataset_id = str(dataset["id"])
                target_id = f"{_stable_id(project_id)}:{_stable_id(dataset_id)}"
                investigation_id = str(
                    dataset.get("investigationID")
                    or f"tess-{_stable_id(project_id)}-{_stable_id(dataset_id)}"
                )
                raw_priority = dataset.get("autonomousPriority", position)
                eligible = dataset.get("autonomousEligible", True) is True
                targets.append(
                    InvestigationTarget(
                        id=target_id,
                        investigation_id=investigation_id,
                        workflow_id=WORKFLOW_ID,
                        workflow_version=WORKFLOW_VERSION,
                        priority=int(raw_priority),
                        eligible=eligible,
                        metadata={
                            "sourceProjectPath": str(project_path),
                            "sourceProjectID": str(project_id),
                            "datasetID": dataset_id,
                            "ticID": dataset.get("ticID"),
                            "targetName": dataset.get("targetName"),
                        },
                    )
                )
        return tuple(targets)


def _latest_complete(investigation: Investigation, handler_id: str):
    return next(
        (
            stage
            for stage in reversed(investigation.stages)
            if stage.handler_id == handler_id and stage.status == "COMPLETE"
        ),
        None,
    )


def plan_tess_branches(
    investigation: Investigation, target: InvestigationTarget
) -> tuple[ScientificBranch, ...]:
    """Translate persisted TESS evidence into domain-neutral branch declarations."""

    targeted = _latest_complete(
        investigation, "openstar.tess.targeted-observation-planning.generate"
    )
    if targeted is not None:
        availability = investigation.metadata.get("externalDataAvailability") or {}
        dependency_id = "targeted-high-resolution-time-series-photometry"
        available = availability.get(dependency_id) is True
        return (
            ScientificBranch(
                id="targeted-observation-analysis",
                experiment=StageRequest(
                    id=f"{len(investigation.stages) + 1:03d}-targeted-observation-analysis",
                    handler_id="openstar.tess.targeted-observation-analysis.run",
                    parameters={"observationPlanStageID": targeted.id},
                    triggered_by_stage_id=targeted.id,
                ),
                external_data=(
                    ExternalDataDependency(
                        dependency_id,
                        available,
                        (
                            None
                            if available
                            else "The preregistered targeted observations have not been acquired and ingested."
                        ),
                    ),
                ),
            ),
        )

    # The existing TESS handlers already make the within-run scientific
    # continuation decision in StageOutcome.next_stage.  Autonomous dispatch
    # records that decision on the immutable stage and asks this adapter to
    # translate it, rather than reimplementing the decision tree here.
    if investigation.stages:
        latest = investigation.stages[-1]
        if latest.status == "COMPLETE" and latest.next_stage is not None:
            raw = latest.next_stage
            return (
                ScientificBranch(
                    id=f"continue-after-{latest.id}",
                    experiment=StageRequest(
                        id=str(raw["id"]),
                        handler_id=str(raw["handler_id"]),
                        parameters=dict(raw.get("parameters") or {}),
                        triggered_by_stage_id=raw.get("triggered_by_stage_id"),
                    ),
                ),
            )

    atlas_fixed = _latest_complete(
        investigation, "openstar.tess.atlas-fixed-window.interpret"
    )
    if (
        atlas_fixed is not None
        and (atlas_fixed.result or {}).get("recommendedNextTest")
        == "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
    ):
        return (
            ScientificBranch(
                id="targeted-observation-planning",
                experiment=StageRequest(
                    id=f"{len(investigation.stages) + 1:03d}-targeted-observation-planning",
                    handler_id="openstar.tess.targeted-observation-planning.generate",
                    parameters={},
                    triggered_by_stage_id=atlas_fixed.id,
                ),
            ),
        )

    if not investigation.stages:
        metadata = investigation.metadata
        return (
            ScientificBranch(
                id="primary-tess-investigation",
                experiment=StageRequest(
                    id="001-prepare-target",
                    handler_id="openstar.tess.prepare-target",
                    parameters={
                        "projectPath": metadata["sourceProjectPath"],
                        "datasetID": metadata["datasetID"],
                        "ticID": metadata.get("ticID"),
                    },
                ),
            ),
        )

    # A nonempty state with no recognized scientific continuation is not proof
    # that the investigation is complete.
    return (
        ScientificBranch(
            id="unresolved-tess-continuation",
            experiment=StageRequest(
                id=f"{len(investigation.stages) + 1:03d}-unresolved-continuation",
                handler_id="openstar.tess.unresolved-continuation",
                parameters={},
            ),
            required_stage_ids=("tess-continuation-decision",),
        ),
    )


def register_tess_workflow_handlers(
    store: InvestigationStore,
    coordinator,
    *,
    poll_interval: float = 1.0,
    timeout: float | None = None,
) -> WorkflowEngine:
    """Register the existing v20.28 TESS handlers for autonomous dispatch."""

    from .tess_investigation import build_engine

    engine = build_engine(
        store,
        coordinator,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    engine.chain_stages = False
    return engine
