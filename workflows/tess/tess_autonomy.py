from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from openstar_autonomy import (
    AutonomousInvestigationEngine,
    ExternalDataDependency,
    ScientificBranch,
)
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


def _continuation_stage_id(stage, label: str) -> str:
    """Reconstruct the workflow continuation ID from its triggering stage."""
    try:
        number = int(str(stage.id).split("-", 1)[0]) + 1
    except (TypeError, ValueError):
        raise ValueError(f"Stage id must begin with an integer prefix: {stage.id}")
    return f"{number:03d}-{label}"


def _has_terminal_tess_evidence(investigation: Investigation) -> bool:
    if not investigation.stages:
        return False
    latest = investigation.stages[-1]
    return latest.status == "COMPLETE" and (
        latest.stop
        # Backward compatibility for terminal TESS evidence persisted before
        # InvestigationStage retained StageOutcome.stop.
        or latest.handler_id == "openstar.tess.finalize"
    )


def _persisted_gaia_continuation(investigation: Investigation):
    """Return an unattempted continuation recorded by a completed Gaia stage."""
    attempted_ids = {stage.id for stage in investigation.stages}
    gaia_handlers = {
        "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare",
        "openstar.tess.gaia-source-resolved-counterpart-photometry.run",
    }
    for stage in reversed(investigation.stages):
        if (
            stage.handler_id in gaia_handlers
            and stage.status == "COMPLETE"
            and stage.next_stage is not None
            and str(stage.next_stage.get("id")) not in attempted_ids
        ):
            return stage, stage.next_stage
    return None


def _request_from_persisted(raw: dict) -> StageRequest:
    return StageRequest(
        id=str(raw["id"]),
        handler_id=str(raw["handler_id"]),
        parameters=dict(raw.get("parameters") or {}),
        triggered_by_stage_id=raw.get("triggered_by_stage_id"),
    )


def repair_obsolete_terminal_wait(
    store: InvestigationStore, investigation: Investigation
) -> Investigation:
    """Repair only known obsolete terminal TESS decisions from older code."""

    control = investigation.metadata.get("controlState")
    if (
        investigation.workflow_id != WORKFLOW_ID
        or not isinstance(control, dict)
    ):
        return investigation

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _has_terminal_tess_evidence(investigation)
    ):
        repaired, _ = AutonomousInvestigationEngine(store).decide(investigation, ())
        return repaired

    gaia_continuation = _persisted_gaia_continuation(investigation)
    if (
        investigation.status == "COMPLETE"
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and gaia_continuation is not None
    ):
        metadata = investigation.metadata or {}
        target = InvestigationTarget(
            id=str(metadata.get("datasetID") or investigation.id),
            investigation_id=investigation.id,
            workflow_id=investigation.workflow_id,
            workflow_version=investigation.workflow_version,
            metadata=dict(metadata),
        )
        repaired, _ = AutonomousInvestigationEngine(store).decide(
            investigation, plan_tess_branches(investigation, target)
        )
        return repaired

    offset_variability = _latest_complete(
        investigation, "openstar.tess.offset-source-variability.interpret"
    )
    gaia_attempted = any(
        stage.handler_id.startswith(
            "openstar.tess.gaia-source-resolved-counterpart-photometry."
        )
        for stage in investigation.stages
    )
    offset_result = (offset_variability.result or {}) if offset_variability else {}
    if not (
        investigation.status == "COMPLETE"
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and offset_variability is not None
        and offset_result.get("recommendedNextTest")
        == "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
        and offset_result.get("physicalMechanismResolved") is False
        and not gaia_attempted
    ):
        return investigation

    metadata = investigation.metadata or {}
    target = InvestigationTarget(
        id=str(metadata.get("datasetID") or investigation.id),
        investigation_id=investigation.id,
        workflow_id=investigation.workflow_id,
        workflow_version=investigation.workflow_version,
        metadata=dict(metadata),
    )
    repaired, _ = AutonomousInvestigationEngine(store).decide(
        investigation, plan_tess_branches(investigation, target)
    )
    return repaired


def plan_tess_branches(
    investigation: Investigation, target: InvestigationTarget
) -> tuple[ScientificBranch, ...]:
    """Translate persisted TESS evidence into domain-neutral branch declarations."""

    prf_interpretation = _latest_complete(
        investigation, "openstar.tess.official-spoc-prf-forward-modeling.interpret"
    )
    catalog_identification = _latest_complete(
        investigation, "openstar.tess.catalog-counterpart-identification.analyze"
    )
    variability_interpretation = _latest_complete(
        investigation, "openstar.tess.offset-source-variability.interpret"
    )
    gaia_interpretation = _latest_complete(
        investigation,
        "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
    )
    if gaia_interpretation is not None:
        return ()

    gaia_continuation = _persisted_gaia_continuation(investigation)
    if gaia_continuation is not None:
        completed, raw = gaia_continuation
        return (
            ScientificBranch(
                id=f"continue-after-{completed.id}",
                experiment=_request_from_persisted(raw),
            ),
        )
    if (
        prf_interpretation is not None
        and catalog_identification is None
        and (prf_interpretation.result or {}).get("recommendedNextTest")
        == "CATALOG_COUNTERPART_IDENTIFICATION"
        and (prf_interpretation.result or {}).get("physicalMechanismResolved") is False
    ):
        return (
            ScientificBranch(
                id=f"continue-catalog-after-{prf_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        prf_interpretation, "catalog-counterpart"
                    ),
                    handler_id="openstar.tess.catalog-counterpart-identification.analyze",
                    parameters={},
                    triggered_by_stage_id=prf_interpretation.id,
                ),
            ),
        )

    catalog_result = (catalog_identification.result or {}) if catalog_identification else {}
    preferred = catalog_result.get("preferredCandidate") or {}
    preferred_ids = preferred.get("catalogIDs") or {}
    justified_preferred = (
        preferred.get("raDeg") is not None
        and preferred.get("decDeg") is not None
        and (preferred_ids.get("ticID") is not None
             or preferred_ids.get("gaiaDR3SourceID") is not None)
    )
    if (
        catalog_identification is not None
        and variability_interpretation is None
        and catalog_result.get("recommendedNextTest")
        == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
        and catalog_result.get("physicalMechanismResolved") is False
        and justified_preferred
        and not any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"}
            for stage in investigation.stages
        )
    ):
        return (
            ScientificBranch(
                id=f"continue-counterpart-variability-after-{catalog_identification.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        catalog_identification, "prepare-offset-source-variability"
                    ),
                    handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={},
                    triggered_by_stage_id=catalog_identification.id,
                ),
            ),
        )

    variability_result = (
        variability_interpretation.result or {}
        if variability_interpretation is not None
        else {}
    )
    gaia_started = any(
        stage.handler_id.startswith(
            "openstar.tess.gaia-source-resolved-counterpart-photometry."
        )
        for stage in investigation.stages
    )
    if (
        variability_interpretation is not None
        and gaia_interpretation is None
        and not gaia_started
        and variability_result.get("recommendedNextTest")
        == "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
        and variability_result.get("physicalMechanismResolved") is False
    ):
        return (
            ScientificBranch(
                id=f"continue-gaia-counterpart-after-{variability_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        variability_interpretation,
                        "prepare-gaia-source-resolved-counterpart-photometry",
                    ),
                    handler_id=(
                        "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare"
                    ),
                    parameters={},
                    triggered_by_stage_id=variability_interpretation.id,
                ),
            ),
        )

    if variability_interpretation is not None:
        return ()

    if _has_terminal_tess_evidence(investigation):
        return ()

    # v20.2 could persist this preparation failure before preparation had a
    # structured non-executable result.  Preserve the immutable failed record,
    # but resume at the scientifically appropriate independent-sector branch.
    latest = investigation.stages[-1] if investigation.stages else None
    if (
        latest is not None
        and latest.status == "FAILED"
        and latest.handler_id == "openstar.tess.followup.prepare-low-frequency"
        and "Follow-up frequency window is invalid" in (latest.error or "")
    ):
        return (
            ScientificBranch(
                id="recover-unavailable-low-frequency-followup",
                experiment=StageRequest(
                    id=f"{len(investigation.stages) + 1:03d}-prepare-independent-sectors",
                    handler_id="openstar.tess.independent.prepare",
                    parameters={},
                    triggered_by_stage_id=latest.id,
                ),
            ),
        )

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
