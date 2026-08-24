from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from openstar_autonomy import (
    AutonomousInvestigationEngine,
    ExternalDataDependency,
    ScientificBranch,
)
from openstar_investigation import Investigation, InvestigationStore, sha256_file
from openstar_targets import InvestigationTarget
from openstar_workflow import StageRequest, WorkflowEngine
from .tess_localization_evidence import frozen_residual_localization_family
from .tess_source_pair_lineage import frozen_source_pair_evidence

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


def _intrinsic_target_boundary(investigation: Investigation):
    """Return only the unconsumed exact v20.12 target-dominant boundary."""
    stage = _latest_complete(investigation, "openstar.tess.multi-source-residual.interpret")
    result = (stage.result or {}) if stage is not None else {}
    attempted = any(item.handler_id == "openstar.tess.intrinsic-nonstationary.analyze"
                    for item in investigation.stages)
    if (stage is not None and not attempted
            and result.get("recommendedNextTest") == "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION"
            and result.get("classification") == "TARGET_RESIDUAL_COMPONENT_DOMINANT"
            and result.get("residualModeOrigin") == "TARGET_DOMINANT"
            and result.get("physicalMechanismResolved") is False):
        return stage
    return None


def _target_residual_mechanism_boundary(investigation: Investigation):
    """Return only an unconsumed v20.13 temporal-mechanism recommendation."""
    stage = _latest_complete(investigation, "openstar.tess.intrinsic-nonstationary.analyze")
    result = (stage.result or {}) if stage is not None else {}
    attempted = any(item.handler_id == "openstar.tess.target-residual-mechanism.analyze"
                    for item in investigation.stages)
    if (stage is not None and not attempted
            and result.get("classification") in {
                "AMPLITUDE_EVOLVING_TARGET_RESIDUAL",
                "TRANSIENT_INTERMITTENT_TARGET_RESIDUAL",
            }
            and result.get("recommendedNextTest")
            == "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
            and result.get("physicalMechanismResolved") is False):
        return stage
    return None


def _old_target_residual_adjudication_boundary(investigation: Investigation):
    """Match only the known stage-028 old-semantics pending-finalizer boundary."""
    stage = next((item for item in investigation.stages
        if item.id == "028-target-residual-mechanism"
        and item.handler_id == "openstar.tess.target-residual-mechanism.analyze"
        and item.status == "COMPLETE" and item.result is not None), None)
    if (stage is None or stage.result.get("adjudicationVersion") is not None
            or not any(Path(reference.path).name == "target-residual-mechanism-v20.14.json"
                       and bool(reference.sha256) for reference in stage.artifacts)):
        return None
    if any(item.handler_id == "openstar.tess.target-residual-mechanism-adjudication.analyze"
           for item in investigation.stages):
        return None
    control = investigation.metadata.get("controlState") or {}
    selected = control.get("selectedExperiment") or {}
    expected_next_stage = {
        "id": "029-finalize",
        "handler_id": "openstar.tess.finalize",
        "parameters": {"outputSuffix": "v20.14-intrinsic"},
        "triggered_by_stage_id": "028-target-residual-mechanism",
    }
    if (investigation.status != "RUNNING"
            or control.get("schedulerAction") != "RUN_EXPERIMENT"
            or not isinstance(stage.next_stage, dict)
            or stage.next_stage != expected_next_stage
            or selected != stage.next_stage):
        return None
    return stage


def _v2015_predictive_validation_boundary(investigation: Investigation):
    """Match only the known stage-029 v20.15 pending-finalizer boundary."""
    stage = next((item for item in investigation.stages
        if item.id == "029-target-residual-mechanism-adjudication"
        and item.handler_id == "openstar.tess.target-residual-mechanism-adjudication.analyze"
        and item.status == "COMPLETE" and item.result is not None), None)
    if stage is None or any(item.handler_id ==
            "openstar.tess.target-residual-mechanism-predictive-validation.analyze"
            for item in investigation.stages):
        return None
    result = stage.result
    if (result.get("classification") != "TARGET_RESIDUAL_MECHANISM_UNRESOLVED"
            or result.get("recommendedNextTest") != "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"
            or result.get("physicalMechanismResolved") is not False
            or result.get("failClosedReasons")
            or not any(Path(ref.path).name == "target-residual-mechanism-adjudication-v20.15.json"
                       and ref.sha256 and Path(ref.path).is_file()
                       and sha256_file(ref.path) == ref.sha256 for ref in stage.artifacts)):
        return None
    expected = {"id": "030-finalize", "handler_id": "openstar.tess.finalize",
        "parameters": {"outputSuffix": "v20.15-intrinsic-corrective-adjudication"},
        "triggered_by_stage_id": stage.id}
    control = investigation.metadata.get("controlState") or {}
    if (investigation.status != "RUNNING" or control.get("schedulerAction") != "RUN_EXPERIMENT"
            or stage.next_stage != expected or control.get("selectedExperiment") != expected):
        return None
    return stage


def _awaiting_nsc_adapter(investigation: Investigation) -> bool:
    """Identify only the incremental SkyMapper-to-NSC implementation boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.skymapper-resolved-counterpart-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest") == "NSC_RESOLVED_COUNTERPART_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.nsc-resolved-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_noirlab_adapter(investigation: Investigation) -> bool:
    """Identify only the incremental NSC-to-image-level implementation boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.nsc-resolved-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest") == "NOIRLAB_IMAGE_LEVEL_FORCED_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.noirlab-image-forced-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_current_des_adapter(investigation: Investigation) -> bool:
    """Keep the intentional NOIRLab-to-current-DES boundary durable."""
    stage = _latest_complete(
        investigation, "openstar.tess.noirlab-image-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest")
        == "DES_DR2_SINGLE_EPOCH_LOCAL_FORCED_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.des-dr2-se-local-forced-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_current_atlas_adapter(investigation: Investigation) -> bool:
    """Identify only the DES-to-credential-aware ATLAS implementation boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.des-dr2-se-local-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest") == "ATLAS_FORCED_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(
            item.handler_id.startswith("openstar.tess.atlas-forced-photometry.")
            for item in investigation.stages
        )
    )


def _awaiting_atlas_signed_reanalysis_adapter(investigation: Investigation) -> bool:
    stage = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("recommendedNextTest")
        == "ATLAS_SIGNED_FORCED_PHOTOMETRY_REANALYSIS"
        and result.get("physicalMechanismResolved") is False
        and not any(item.handler_id.startswith(
            "openstar.tess.atlas-forced-photometry-reanalysis."
        ) for item in investigation.stages)
    )


def _awaiting_post_atlas_targeted_observation_adapter(
    investigation: Investigation,
) -> bool:
    """Recognize only the complete, unambiguous direct v20.24 boundary."""
    stage = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    if not (
        stage is not None
        and result.get("classification") == "ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("residualModeOrigin")
        == "ARCHIVAL_ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("recommendedNextTest")
        == "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
    ):
        return False
    position = investigation.stages.index(stage)
    preceding = investigation.stages[:position]
    later = investigation.stages[position + 1:]
    if not stage.triggered_by_stage_id or not any(
        item.id == stage.triggered_by_stage_id
        and item.status == "COMPLETE"
        and item.handler_id in (
            "openstar.tess.atlas-forced-photometry.run",
            "openstar.tess.atlas-forced-photometry.prepare",
        )
        for item in preceding
    ):
        return False

    search = dict(
        (result.get("distributedValidation") or {}).get("frequencySearch")
        or result.get("frequencySearch")
        or {}
    )
    minimum = search.get(
        "minimumFrequency", search.get("minFrequency", search.get("startFrequency"))
    )
    maximum = search.get(
        "maximumFrequency", search.get("maxFrequency", search.get("endFrequency"))
    )
    try:
        if maximum is None:
            total = int(search.get("totalFrequencies", search.get("frequencyCount")))
            step = float(search.get("frequencyStep", search.get("step")))
            maximum = float(minimum) + max(total - 1, 0) * step
        valid_search = (
            math.isfinite(float(minimum))
            and math.isfinite(float(maximum))
            and 0 < float(minimum) < float(maximum)
        )
    except (TypeError, ValueError):
        valid_search = False
    if not valid_search:
        return False

    if frozen_source_pair_evidence(investigation, stage) is None:
        return False

    return not any(
        item.handler_id.startswith((
            "openstar.tess.atlas-forced-photometry-reanalysis.",
            "openstar.tess.atlas-time-resolved.",
            "openstar.tess.atlas-fixed-window.",
            "openstar.tess.targeted-observation-planning.",
        ))
        for item in (*preceding, *later)
    )


def _is_unresolved_atlas_targeted_observation_boundary(investigation) -> bool:
    """Identify the semantic boundary even when its adapter evidence is invalid."""
    stage = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.interpret"
    )
    result = (stage.result or {}) if stage is not None else {}
    return bool(
        stage is not None
        and result.get("classification") == "ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("residualModeOrigin")
        == "ARCHIVAL_ATLAS_SOURCE_ATTRIBUTION_UNRESOLVED"
        and result.get("recommendedNextTest")
        == "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"
        and result.get("physicalMechanismResolved") is False
        and not any(item.handler_id.startswith(
            "openstar.tess.targeted-observation-planning."
        ) for item in investigation.stages)
    )


def _persisted_archive_continuation(investigation: Investigation):
    """Return a valid unattempted continuation recorded by an archive stage."""
    attempted_ids = {stage.id for stage in investigation.stages}
    archive_handlers = {
        "openstar.tess.gaia-source-resolved-counterpart-photometry.prepare",
        "openstar.tess.gaia-source-resolved-counterpart-photometry.run",
        "openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
        "openstar.tess.skymapper-resolved-counterpart-photometry.run",
        "openstar.tess.nsc-resolved-photometry.prepare",
        "openstar.tess.nsc-resolved-photometry.run",
        "openstar.tess.noirlab-image-forced-photometry.prepare",
        "openstar.tess.noirlab-image-forced-photometry.run",
        "openstar.tess.des-dr2-se-local-forced-photometry.prepare",
        "openstar.tess.des-dr2-se-local-forced-photometry.run",
        "openstar.tess.atlas-forced-photometry.prepare",
        "openstar.tess.atlas-forced-photometry.collect",
        "openstar.tess.atlas-forced-photometry.run",
    }
    for stage in reversed(investigation.stages):
        if stage.handler_id not in archive_handlers or stage.status != "COMPLETE":
            continue
        raw = stage.next_stage
        if not isinstance(raw, dict):
            continue
        continuation_id = raw.get("id")
        handler_id = raw.get("handler_id")
        parameters = raw.get("parameters")
        if (
            not isinstance(continuation_id, str)
            or not continuation_id
            or continuation_id in attempted_ids
            or not isinstance(handler_id, str)
            or not isinstance(parameters, dict)
            or raw.get("triggered_by_stage_id") != stage.id
        ):
            continue

        # Collection is an asynchronous archive boundary.  Resume only the exact
        # immutable continuation that the collector produced, and ensure its
        # parameters agree with that durable collection result.  In particular,
        # never infer a replacement run or repeat archive collection here.
        if stage.handler_id == "openstar.tess.atlas-forced-photometry.collect":
            result = stage.result if isinstance(stage.result, dict) else {}
            project_path = result.get("projectPath")
            if handler_id == "openstar.tess.atlas-forced-photometry.run":
                valid = (
                    isinstance(project_path, str)
                    and bool(project_path)
                    and parameters == {"projectPath": project_path}
                )
            elif handler_id == "openstar.tess.atlas-forced-photometry.interpret":
                valid = not project_path and parameters == {
                    "distributedRunExpected": False
                }
            else:
                valid = False
            if not valid or any(
                item.handler_id in {
                    "openstar.tess.atlas-forced-photometry.run",
                    "openstar.tess.atlas-forced-photometry.interpret",
                }
                for item in investigation.stages
            ):
                continue
        return stage, raw
    return None


def _request_from_persisted(raw: dict) -> StageRequest:
    return StageRequest(
        id=str(raw["id"]),
        handler_id=str(raw["handler_id"]),
        parameters=dict(raw.get("parameters") or {}),
        triggered_by_stage_id=raw.get("triggered_by_stage_id"),
    )


_LEGACY_TRANSIENT_QUERY_TYPES = {
    "Timeout", "ReadTimeout", "ConnectTimeout", "TimeoutError", "ConnectionError",
    "ConnectError", "NewConnectionError", "MaxRetryError", "NameResolutionError",
    "NetworkError",
}


def _transient_required_identity_failure(result: dict) -> bool:
    # Catalog enrichment outages do not invalidate an identity already anchored
    # by TIC.  This also makes historical snapshots (which predate aggregate
    # coverage fields) restart-safe without scheduling repeated VizieR calls.
    if result.get("identityResolved") is True or (result.get("tic") or {}).get("found"):
        return False
    for key in ("tic", "vsx", "gaiaDR3", "gaiaVariability"):
        query = result.get(key) or {}
        if query.get("queryErrorClassification") == "TRANSIENT_INFRASTRUCTURE":
            return True
        # Historical identity snapshots retained the exception type at the
        # beginning of queryError, before structured classification existed.
        error_type = query.get("queryErrorType")
        if not error_type and isinstance(query.get("queryError"), str):
            error_type = query["queryError"].partition(":")[0]
        if error_type in _LEGACY_TRANSIENT_QUERY_TYPES:
            return True
    return False


def _repair_catalog_timeout_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    if investigation.status != "COMPLETE" or control.get("schedulerAction") != "INVESTIGATION_COMPLETE":
        return None
    conclusion = next((stage.result for stage in reversed(investigation.stages)
                       if stage.status == "COMPLETE" and isinstance(stage.result, dict)
                       and (stage.result.get("claim") is not None)), None)
    claim = conclusion.get("claim") if conclusion else None
    if isinstance(claim, dict):
        claim = claim.get("claim")
    planner = _latest_complete(investigation, "openstar.tess.planner")
    if claim != "HUMAN_REVIEW_REQUIRED" or planner is None or (planner.result or {}).get("reason") != "catalog-coverage-incomplete":
        return None

    identities = [stage for stage in investigation.stages
                  if stage.handler_id == "openstar.tess.catalog-identity"
                  and stage.status == "COMPLETE" and isinstance(stage.result, dict)]
    failed_index = next((index for index, stage in enumerate(identities)
                         if _transient_required_identity_failure(stage.result or {})), None)
    if failed_index is None or any(
        not _transient_required_identity_failure(stage.result or {})
        for stage in identities[failed_index + 1:]
    ):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-catalog-identity",
        handler_id="openstar.tess.catalog-identity",
        parameters={},
        triggered_by_stage_id=identities[failed_index].id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TRANSIENT_CATALOG_IDENTITY_COMPATIBILITY_RETRY",
        },
    )


_OFFICIAL_PRF_PREPARE = "openstar.tess.official-spoc-prf-forward-modeling.prepare"
_OFFICIAL_PRF_RUN = "openstar.tess.official-spoc-prf-forward-modeling.run"
_OFFICIAL_PRF_INTERPRET = "openstar.tess.official-spoc-prf-forward-modeling.interpret"


def _legacy_prf_transport_error(error: object) -> bool:
    """Recognize only transport text emitted by the obsolete PRF runner."""
    if not isinstance(error, str):
        return False
    return bool(
        re.fullmatch(
            r"TimeoutError: (?:The read operation timed out|timed out|read timed out)",
            error,
        )
        or re.fullmatch(
            r"URLError: <urlopen error (?:timed out|\[Errno 60\] Operation timed out)>",
            error,
        )
        or (
            error.startswith("URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING]")
            and error.endswith(">")
        )
    )


def _repair_official_prf_transport_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Select one append-only retry for a legacy transport-only PRF result."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
        or control.get("selectedExperiment") not in (None, {})
        or not _has_terminal_tess_evidence(investigation)
    ):
        return None

    stages = investigation.stages
    interpret_index = next((
        index for index in range(len(stages) - 1, -1, -1)
        if stages[index].handler_id == _OFFICIAL_PRF_INTERPRET
        and stages[index].status == "COMPLETE"
    ), None)
    if interpret_index is None:
        return None
    interpretation = stages[interpret_index]
    result = interpretation.result or {}
    if not (
        result.get("classification") == "BLOCKED_EXTERNAL_DATA"
        and result.get("recommendedNextTest")
        == "RETRY_PIXEL_RESPONSE_FUNCTION_DEBLENDING"
        and result.get("physicalMechanismResolved") is False
    ):
        return None

    by_id = {stage.id: stage for stage in stages}
    run = by_id.get(interpretation.triggered_by_stage_id)
    prepare = by_id.get(run.triggered_by_stage_id) if run is not None else None
    if not (
        run is not None and run.handler_id == _OFFICIAL_PRF_RUN
        and run.status == "COMPLETE" and isinstance(run.result, dict)
        and prepare is not None and prepare.handler_id == _OFFICIAL_PRF_PREPARE
        and prepare.status == "COMPLETE"
    ):
        return None
    sector_results = run.result.get("sectorResults")
    errors = run.result.get("errors")
    if sector_results != [] or not isinstance(errors, list) or not errors:
        return None
    if not all(
        isinstance(item, dict) and _legacy_prf_transport_error(item.get("error"))
        for item in errors
    ):
        return None

    # The completed finalizer must be the current terminal evidence and must
    # descend directly from this interpretation. Any later PRF stage means a
    # retry has already been attempted or superseded.
    terminal = stages[-1]
    if not (
        terminal.handler_id == "openstar.tess.finalize"
        and terminal.status == "COMPLETE"
        and terminal.triggered_by_stage_id == interpretation.id
        and all(
            not stage.handler_id.startswith(
                "openstar.tess.official-spoc-prf-forward-modeling."
            )
            for stage in stages[interpret_index + 1:]
        )
    ):
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-run-official-spoc-prf-forward-modeling",
        handler_id=_OFFICIAL_PRF_RUN,
        parameters={},
        triggered_by_stage_id=interpretation.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_OFFICIAL_PRF_TRANSPORT_COMPATIBILITY_RETRY",
        },
    )


def _positive_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _repair_promoted_period_characterization_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Continue pre-v20.4 promoted period families directly to morphology."""
    if (
        investigation.status != "COMPLETE"
        or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"
    ):
        return None

    broad_index = next((
        index for index in range(len(investigation.stages) - 1, -1, -1)
        if investigation.stages[index].status == "COMPLETE"
        and investigation.stages[index].handler_id
        == "openstar.tess.independent.broad.interpret"
    ), None)
    if broad_index is None:
        return None
    broad = investigation.stages[broad_index]
    result = broad.result or {}
    family = result.get("harmonicFamily") or {}
    if not (
        (result.get("claimDecision") or {}).get("claim")
        == "INDEPENDENT_PERIOD_ESTIMATE"
        and result.get("promotionEligible") is True
        and _positive_number(family.get("representativeRawPeriodDays"))
        and _positive_number(family.get("possibleDoubleCycleDays"))
        and family.get("physicalCycleResolved") is not True
    ):
        return None

    later_stages = investigation.stages[broad_index + 1:]
    if any(
        stage.status == "COMPLETE"
        and stage.handler_id != "openstar.tess.finalize"
        for stage in later_stages
    ):
        return None
    terminal = next((
        stage for stage in reversed(later_stages)
        if stage.status == "COMPLETE"
        and stage.handler_id == "openstar.tess.finalize"
    ), None)
    terminal_claim = (((terminal.result or {}).get("claim") or {}).get("claim")
                      if terminal is not None else None)
    if terminal_claim != "INDEPENDENT_PERIOD_ESTIMATE":
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-morphology",
        handler_id="openstar.tess.morphology.analyze",
        parameters={},
        triggered_by_stage_id=broad.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": (
                "TESS_INDEPENDENT_PERIOD_CHARACTERIZATION_"
                "COMPATIBILITY_CONTINUATION"
            ),
        },
    )


def _repair_mode_identification_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append the stage introduced after the stable-residual terminal boundary."""
    if (investigation.status != "COMPLETE"
            or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"):
        return None
    summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    result = (summary.result or {}) if summary is not None else {}
    if not (summary is not None
            and result.get("recommendedNextTest") == "MODE_IDENTIFICATION_OR_PULSATION_MODELING"
            and result.get("physicalMechanismResolved") is False
            and not any(stage.handler_id == "openstar.tess.mode-identification.analyze"
                        for stage in investigation.stages)):
        return None
    # Only repair the old implementation boundary: no completed scientific
    # stage other than finalize may follow the recommendation.
    index = investigation.stages.index(summary)
    if any(stage.status == "COMPLETE" and stage.handler_id != "openstar.tess.finalize"
           for stage in investigation.stages[index + 1:]):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-mode-identification",
        handler_id="openstar.tess.mode-identification.analyze",
        parameters={},
        triggered_by_stage_id=summary.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(continuation),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_MODE_IDENTIFICATION_COMPATIBILITY_CONTINUATION"},
    )


def _repair_dynamic_harmonic_terminal(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append only the stage introduced after the old harmonic boundary."""
    if (investigation.status != "COMPLETE"
            or control.get("schedulerAction") != "INVESTIGATION_COMPLETE"):
        return None
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    result = (mode.result or {}) if mode is not None else {}
    if not (mode is not None
            and result.get("recommendedNextTest") == "DYNAMIC_HARMONIC_MODELING"
            and result.get("physicalMechanismResolved") is False
            and not any(stage.handler_id == "openstar.tess.dynamic-harmonic.analyze"
                        for stage in investigation.stages)):
        return None
    index = investigation.stages.index(mode)
    if any(stage.status == "COMPLETE" and stage.handler_id != "openstar.tess.finalize"
           for stage in investigation.stages[index + 1:]):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-dynamic-harmonic-modeling",
        handler_id="openstar.tess.dynamic-harmonic.analyze",
        parameters={}, triggered_by_stage_id=mode.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(continuation),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_DYNAMIC_HARMONIC_COMPATIBILITY_CONTINUATION"},
    )


def _repair_unresolved_dynamic_localization_review_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Resume only the obsolete v20.11 unresolved-family preparation failure."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.residual-mode-localization-review.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error not in {
                "RuntimeError: v20.11 requires the morphology-resolved physical period.",
                "RuntimeError: v20.11 requires the completed v20.9 nonstationary model.",
            }):
        return None
    scheduler_action = control.get("schedulerAction")
    if scheduler_action not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED"):
        return None
    selected = control.get("selectedExperiment")
    if selected is not None:
        if not isinstance(selected, dict):
            return None
        # The real persisted failures retain the failed request itself.  Do not
        # broaden recovery to an earlier request merely because it is an ancestor.
        selected_stage = next(
            (stage for stage in investigation.stages if stage.id == selected.get("id")),
            None,
        )
        if selected_stage is None or (
            selected.get("handler_id") != selected_stage.handler_id
            or not isinstance(selected.get("parameters"), dict)
            or selected.get("parameters") != selected_stage.parameters
            or selected.get("triggered_by_stage_id")
            != selected_stage.triggered_by_stage_id
        ):
            return None
        if selected_stage.id != failed.id:
            return None
    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    mode_result = (mode.result or {}) if mode else {}
    localization_result = (localization.result or {}) if localization else {}
    family_context = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode_result if mode else None,
    )
    if not (family_context is not None
            and localization
            and localization_result.get("recommendedNextTest")
                 == "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"):
        return None
    orders = list(family_context[1])
    localization_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    subtracted = ((localization_prepare.result or {}).get("subtractedHarmonicOrders")
                  if localization_prepare else None)
    rerun_localization = list(subtracted or [1, 2]) != orders
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=(f"{max(prefixes, default=0) + 1:03d}-" +
            ("prepare-residual-mode-localization" if rerun_localization
             else "prepare-residual-mode-localization-review")),
        handler_id=("openstar.tess.residual-mode-localization.prepare" if rerun_localization
                    else "openstar.tess.residual-mode-localization-review.prepare"),
        parameters={}, triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(continuation),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_UNRESOLVED_DYNAMIC_LOCALIZATION_REVIEW_COMPATIBILITY_RETRY"},
    )


def _repair_archive_timeout_localization_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Retry only a localization spawned by the obsolete stage-022 repair."""
    if investigation.status != "FAILED" or len(investigation.stages) < 2:
        return None
    failed = investigation.stages[-1]
    review_failure = investigation.stages[-2]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.residual-mode-localization.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error != (
                "RuntimeError: v20.10 could not prepare any residual-mode pixel datasets."
            )
            or failed.triggered_by_stage_id != review_failure.id
            or review_failure.status != "FAILED"
            or review_failure.handler_id
               != "openstar.tess.residual-mode-localization-review.prepare"
            or review_failure.failure_classification != "NON_RETRYABLE"
            or review_failure.error not in {
                "RuntimeError: v20.11 requires the morphology-resolved physical period.",
                "RuntimeError: v20.11 requires the completed v20.9 nonstationary model.",
            }):
        return None
    if control.get("schedulerAction") not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED"):
        return None
    selected = control.get("selectedExperiment")
    if not isinstance(selected, dict) or any((
        selected.get("id") != failed.id,
        selected.get("handler_id") != failed.handler_id,
        selected.get("parameters") != failed.parameters,
        selected.get("triggered_by_stage_id") != failed.triggered_by_stage_id,
    )):
        return None

    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    family = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode.result if mode else None,
    )
    if (family is None or localization is None
            or (localization.result or {}).get("recommendedNextTest")
               != "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"
            or review_failure.triggered_by_stage_id != localization.id):
        return None
    prior_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    if (prior_prepare is None
            or list((prior_prepare.result or {}).get("subtractedHarmonicOrders") or [1, 2])
               == list(family[1])):
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    continuation = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-residual-mode-localization",
        handler_id="openstar.tess.residual-mode-localization.prepare",
        parameters={}, triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={
            "branchAssessments": [], "selectedExperiment": asdict(continuation),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_ARCHIVE_TIMEOUT_LOCALIZATION_COMPATIBILITY_RETRY",
        },
    )


def _repair_unresolved_dynamic_multisource_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Retry only obsolete v20.12 frozen-family compatibility gates append-only."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.multi-source-residual.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error not in {
                "RuntimeError: v20.12 requires the morphology-resolved physical period.",
                "RuntimeError: v20.12 requires the completed v20.9 nonstationary model.",
                "RuntimeError: v20.12 requires either a morphology-resolved physical "
                "period or an established unresolved dynamic harmonic family.",
            }):
        return None
    if control.get("schedulerAction") not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED"):
        return None
    selected = control.get("selectedExperiment")
    if selected is not None:
        if (not isinstance(selected, dict)
                or selected.get("id") != failed.id
                or selected.get("handler_id") != failed.handler_id
                or selected.get("parameters") != failed.parameters
                or selected.get("triggered_by_stage_id")
                   != failed.triggered_by_stage_id):
            return None
    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    localization_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    localization_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.run"
    )
    review_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.prepare"
    )
    review_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.run"
    )
    review = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.interpret"
    )
    review_result = (review.result or {}) if review else {}
    cross = review_result.get("crossTime") or {}
    family = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode.result if mode else None,
    )
    valid_review_lineage = bool(
        localization_prepare and localization_run and localization
        and review_prepare and review_run and review
        and localization_run.triggered_by_stage_id == localization_prepare.id
        and localization.triggered_by_stage_id == localization_run.id
        and review_prepare.triggered_by_stage_id == localization.id
        and review_run.triggered_by_stage_id == review_prepare.id
        and review.triggered_by_stage_id == review_run.id
    )
    if not (morphology and (morphology.result or {}).get("physicalCycleResolved") is False
            and family is not None
            and review
            and valid_review_lineage
            and failed.triggered_by_stage_id == review.id
            and cross.get("classification")
                == "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND"
            and cross.get("residualModeOrigin") == "TIME_VARIABLE_OR_BLENDED"
            and review_result.get("recommendedNextTest")
                == "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-multi-source-residual",
        handler_id=failed.handler_id,
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(retry),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_UNRESOLVED_DYNAMIC_MULTISOURCE_COMPATIBILITY_RETRY"},
    )


def _repair_resolved_family_multisource_failure(
    store: InvestigationStore, investigation: Investigation, control: dict
) -> Investigation | None:
    """Append a retry for the obsolete resolved-without-v20.9 v20.12 gate."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if (failed.status != "FAILED"
            or failed.handler_id != "openstar.tess.multi-source-residual.prepare"
            or failed.failure_classification != "NON_RETRYABLE"
            or failed.error != (
                "RuntimeError: v20.12 requires the completed v20.9 nonstationary model."
            )):
        return None
    selected = control.get("selectedExperiment")
    expected_selected = asdict(StageRequest(
        failed.id, failed.handler_id, dict(failed.parameters),
        failed.triggered_by_stage_id,
    ))
    if (control.get("schedulerAction") not in ("RUN_EXPERIMENT", "INVESTIGATION_FAILED")
            or not isinstance(selected, dict)
            or selected != expected_selected):
        return None

    morphology = _latest_complete(investigation, "openstar.tess.morphology.analyze")
    dynamic = _latest_complete(investigation, "openstar.tess.dynamic-harmonic.analyze")
    tf_prepare = _latest_complete(investigation, "openstar.tess.time-frequency.prepare")
    tf_summary = _latest_complete(investigation, "openstar.tess.time-frequency.summarize")
    mode = _latest_complete(investigation, "openstar.tess.mode-identification.analyze")
    nonstationary = _latest_complete(investigation, "openstar.tess.nonstationary.summarize")
    localization_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.prepare"
    )
    localization_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.run"
    )
    localization = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization.interpret"
    )
    review_prepare = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.prepare"
    )
    review_run = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.run"
    )
    review = _latest_complete(
        investigation, "openstar.tess.residual-mode-localization-review.interpret"
    )
    family = frozen_residual_localization_family(
        morphology.result if morphology else None,
        dynamic.result if dynamic else None,
        tf_prepare.result if tf_prepare else None,
        tf_summary.result if tf_summary else None,
        mode.result if mode else None,
    )
    review_result = (review.result or {}) if review else {}
    cross = review_result.get("crossTime") or {}
    valid_lineage = bool(
        localization_prepare and localization_run and localization
        and review_prepare and review_run and review
        and localization_run.triggered_by_stage_id == localization_prepare.id
        and localization.triggered_by_stage_id == localization_run.id
        and review_prepare.triggered_by_stage_id == localization.id
        and review_run.triggered_by_stage_id == review_prepare.id
        and review.triggered_by_stage_id == review_run.id
        and failed.triggered_by_stage_id == review.id
    )
    if not (morphology
            and (morphology.result or {}).get("physicalCycleResolved") is True
            and nonstationary is None
            and family is not None
            and family[3] == "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
            and valid_lineage
            and cross.get("classification")
                == "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND"
            and cross.get("residualModeOrigin") == "TIME_VARIABLE_OR_BLENDED"
            and review_result.get("recommendedNextTest")
                == "MULTI_SOURCE_RESIDUAL_DECOMPOSITION"):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-multi-source-residual",
        handler_id=failed.handler_id,
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation, status="RUNNING",
        control_state={"branchAssessments": [], "selectedExperiment": asdict(retry),
                       "schedulerAction": "RUN_EXPERIMENT",
                       "recovery": "TESS_RESOLVED_FAMILY_MULTISOURCE_COMPATIBILITY_RETRY"},
    )


def _repair_closed_file_independent_prepare(
    store: InvestigationStore, investigation: Investigation
) -> Investigation | None:
    """Retry only the known pre-lock Lightkurve closed-file durable failure."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if not (
        failed.status == "FAILED"
        and failed.handler_id == "openstar.tess.independent.prepare"
        and failed.error == "ValueError: I/O operation on closed file."
        and failed.failure_classification == "NON_RETRYABLE"
    ):
        return None
    required = {
        "openstar.tess.prepare-target",
        "openstar.tess.primary-project.run",
        "openstar.tess.catalog-identity",
        "openstar.tess.planner",
    }
    completed = {stage.handler_id for stage in investigation.stages
                 if stage.status == "COMPLETE"}
    if not required.issubset(completed):
        return None
    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-independent-sectors",
        handler_id="openstar.tess.independent.prepare",
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_LIGHTKURVE_CLOSED_FILE_COMPATIBILITY_RETRY",
        },
    )


def _repair_shifted_stage_lookup_independent_prepare(
    store: InvestigationStore, investigation: Investigation
) -> Investigation | None:
    """Retry the one obsolete numeric catalog-stage lookup failure shape."""
    if investigation.status != "FAILED" or not investigation.stages:
        return None
    failed = investigation.stages[-1]
    if not (
        failed.status == "FAILED"
        and failed.handler_id == "openstar.tess.independent.prepare"
        and failed.failure_classification == "NON_RETRYABLE"
        and failed.error
        == "RuntimeError: Stage is not COMPLETE with a result: 003-catalog-identity"
    ):
        return None

    identity_failures = [
        index for index, stage in enumerate(investigation.stages)
        if stage.handler_id == "openstar.tess.catalog-identity"
        and stage.status == "FAILED"
        and stage.failure_classification == "TRANSIENT_INFRASTRUCTURE"
    ]
    successful_identity = next((
        index for index, stage in enumerate(investigation.stages)
        if stage.handler_id == "openstar.tess.catalog-identity"
        and stage.status == "COMPLETE"
        and any(failed_index < index for failed_index in identity_failures)
    ), None)
    hypotheses = next((
        index for index, stage in enumerate(investigation.stages)
        if successful_identity is not None and index > successful_identity
        and stage.handler_id == "openstar.tess.hypotheses" and stage.status == "COMPLETE"
    ), None)
    planner = next((
        index for index, stage in enumerate(investigation.stages)
        if hypotheses is not None and index > hypotheses
        and stage.handler_id == "openstar.tess.planner" and stage.status == "COMPLETE"
    ), None)
    failed_index = len(investigation.stages) - 1
    if planner is None or any(
        index > failed_index
        and stage.handler_id == "openstar.tess.independent.prepare"
        and stage.status == "COMPLETE"
        for index, stage in enumerate(investigation.stages)
    ):
        return None

    prefixes = [int(stage.id.partition("-")[0]) for stage in investigation.stages
                if stage.id.partition("-")[0].isdigit()]
    retry = StageRequest(
        id=f"{max(prefixes, default=0) + 1:03d}-prepare-independent-sectors",
        handler_id="openstar.tess.independent.prepare",
        parameters=dict(failed.parameters),
        triggered_by_stage_id=failed.id,
    )
    return store.set_control_state(
        investigation,
        status="RUNNING",
        control_state={
            "branchAssessments": [],
            "selectedExperiment": asdict(retry),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_RETRY_SHIFTED_STAGE_LOOKUP_COMPATIBILITY_RETRY",
        },
    )


def repair_obsolete_terminal_wait(
    store: InvestigationStore, investigation: Investigation
) -> Investigation:
    """Repair only known obsolete terminal TESS decisions from older code."""

    control = investigation.metadata.get("controlState")
    if investigation.workflow_id != WORKFLOW_ID or not isinstance(control, dict):
        return investigation

    old_mechanism = _old_target_residual_adjudication_boundary(investigation)
    if old_mechanism is not None:
        request = StageRequest(
            id=_continuation_stage_id(old_mechanism, "target-residual-mechanism-adjudication"),
            handler_id="openstar.tess.target-residual-mechanism-adjudication.analyze",
            parameters={}, triggered_by_stage_id=old_mechanism.id,
        )
        return store.set_control_state(
            investigation, status="RUNNING",
            control_state={"branchAssessments": [], "selectedExperiment": asdict(request),
                           "schedulerAction": "RUN_EXPERIMENT",
                           "recovery": "TESS_V20_14_ROUTE_ADJUDICATION_V20_15"},
        )

    adjudication = _v2015_predictive_validation_boundary(investigation)
    if adjudication is not None:
        request = StageRequest(id="030-target-residual-mechanism-predictive-validation",
            handler_id="openstar.tess.target-residual-mechanism-predictive-validation.analyze",
            parameters={}, triggered_by_stage_id=adjudication.id)
        return store.set_control_state(investigation, status="RUNNING", control_state={
            "branchAssessments": [], "selectedExperiment": asdict(request),
            "schedulerAction": "RUN_EXPERIMENT",
            "recovery": "TESS_V20_15_PREDICTIVE_VALIDATION_V20_16"})

    mechanism = _target_residual_mechanism_boundary(investigation)
    if (mechanism is not None and investigation.status == "COMPLETE"
            and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"):
        request = StageRequest(
            id=_continuation_stage_id(mechanism, "target-residual-mechanism"),
            handler_id="openstar.tess.target-residual-mechanism.analyze",
            parameters={}, triggered_by_stage_id=mechanism.id,
        )
        return store.set_control_state(
            investigation, status="RUNNING",
            control_state={"branchAssessments": [], "selectedExperiment": asdict(request),
                           "schedulerAction": "RUN_EXPERIMENT",
                           "recovery": "TESS_V20_13_TARGET_MECHANISM_CONTINUATION"},
        )

    intrinsic = _intrinsic_target_boundary(investigation)
    if (intrinsic is not None and investigation.status == "COMPLETE"
            and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"):
        request = StageRequest(
            id=_continuation_stage_id(intrinsic, "classify-intrinsic-target-residual"),
            handler_id="openstar.tess.intrinsic-nonstationary.analyze",
            parameters={}, triggered_by_stage_id=intrinsic.id,
        )
        return store.set_control_state(
            investigation, status="RUNNING",
            control_state={"branchAssessments": [], "selectedExperiment": asdict(request),
                           "schedulerAction": "RUN_EXPERIMENT",
                           "recovery": "TESS_V20_12_TARGET_INTRINSIC_CONTINUATION"},
        )

    prf_transport_repair = _repair_official_prf_transport_terminal(
        store, investigation, control
    )
    if prf_transport_repair is not None:
        return prf_transport_repair

    archive_timeout_repair = _repair_archive_timeout_localization_failure(
        store, investigation, control
    )
    if archive_timeout_repair is not None:
        return archive_timeout_repair

    localization_review_repair = _repair_unresolved_dynamic_localization_review_failure(
        store, investigation, control
    )
    if localization_review_repair is not None:
        return localization_review_repair

    multisource_repair = _repair_resolved_family_multisource_failure(
        store, investigation, control
    )
    if multisource_repair is not None:
        return multisource_repair

    multisource_repair = _repair_unresolved_dynamic_multisource_failure(
        store, investigation, control
    )
    if multisource_repair is not None:
        return multisource_repair

    independent_repair = _repair_shifted_stage_lookup_independent_prepare(
        store, investigation
    )
    if independent_repair is not None:
        return independent_repair

    independent_repair = _repair_closed_file_independent_prepare(store, investigation)
    if independent_repair is not None:
        return independent_repair

    catalog_repair = _repair_catalog_timeout_terminal(store, investigation, control)
    if catalog_repair is not None:
        return catalog_repair

    period_repair = _repair_promoted_period_characterization_terminal(
        store, investigation, control
    )
    if period_repair is not None:
        return period_repair

    mode_repair = _repair_mode_identification_terminal(store, investigation, control)
    if mode_repair is not None:
        return mode_repair

    dynamic_repair = _repair_dynamic_harmonic_terminal(store, investigation, control)
    if dynamic_repair is not None:
        return dynamic_repair

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_noirlab_adapter(investigation)
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

    if (
        (
            investigation.status == "BLOCKED"
            and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
            and (
                _awaiting_atlas_signed_reanalysis_adapter(investigation)
                or _awaiting_post_atlas_targeted_observation_adapter(investigation)
            )
        )
        or (
            investigation.status == "COMPLETE"
            and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
            and _awaiting_post_atlas_targeted_observation_adapter(investigation)
        )
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

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_current_atlas_adapter(investigation)
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
        control_state = dict(repaired.metadata.get("controlState") or {})
        if control_state.get("schedulerAction") == "WAIT_FOR_PREREQUISITES":
            branch = plan_tess_branches(repaired, target)[0]
            control_state["selectedExperiment"] = asdict(branch.experiment)
            control_state["missingPrerequisites"] = list(branch.required_stage_ids)
            repaired = store.set_control_state(
                repaired, status="BLOCKED", control_state=control_state
            )
        return repaired

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_nsc_adapter(investigation)
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

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _awaiting_current_des_adapter(investigation)
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

    if (
        investigation.status == "BLOCKED"
        and control.get("schedulerAction") == "WAIT_FOR_PREREQUISITES"
        and _has_terminal_tess_evidence(investigation)
        and not _awaiting_nsc_adapter(investigation)
        and not _awaiting_noirlab_adapter(investigation)
        and not _awaiting_current_atlas_adapter(investigation)
        and not _awaiting_atlas_signed_reanalysis_adapter(investigation)
        and not _awaiting_post_atlas_targeted_observation_adapter(investigation)
        and not _is_unresolved_atlas_targeted_observation_boundary(investigation)
    ):
        repaired, _ = AutonomousInvestigationEngine(store).decide(investigation, ())
        return repaired

    gaia_continuation = _persisted_archive_continuation(investigation)
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
    gaia_interpretation = _latest_complete(
        investigation, "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret"
    )
    skymapper_attempted = any(stage.handler_id.startswith(
        "openstar.tess.skymapper-resolved-counterpart-photometry."
    ) for stage in investigation.stages)
    gaia_result = (gaia_interpretation.result or {}) if gaia_interpretation else {}
    obsolete_gaia_terminal = (
        gaia_interpretation is not None
        and gaia_result.get("recommendedNextTest") == "SKYMAPPER_RESOLVED_COUNTERPART_PHOTOMETRY"
        and gaia_result.get("physicalMechanismResolved") is False
        and not skymapper_attempted
    )
    obsolete_skymapper_terminal = _awaiting_nsc_adapter(investigation)
    if not (
        investigation.status == "COMPLETE"
        and control.get("schedulerAction") == "INVESTIGATION_COMPLETE"
        and (obsolete_gaia_terminal or obsolete_skymapper_terminal or (
            offset_variability is not None
            and offset_result.get("recommendedNextTest")
            == "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
            and offset_result.get("physicalMechanismResolved") is False
            and not gaia_attempted
        ))
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

    mechanism = _target_residual_mechanism_boundary(investigation)
    if mechanism is not None:
        return (ScientificBranch(
            id=f"target-residual-mechanism-after-{mechanism.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(mechanism, "target-residual-mechanism"),
                handler_id="openstar.tess.target-residual-mechanism.analyze",
                parameters={}, triggered_by_stage_id=mechanism.id,
            ),
        ),)

    intrinsic = _intrinsic_target_boundary(investigation)
    if intrinsic is not None:
        return (ScientificBranch(
            id=f"intrinsic-target-residual-after-{intrinsic.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(intrinsic, "classify-intrinsic-target-residual"),
                handler_id="openstar.tess.intrinsic-nonstationary.analyze",
                parameters={}, triggered_by_stage_id=intrinsic.id,
            ),
        ),)

    prf_interpretation = _latest_complete(
        investigation, "openstar.tess.official-spoc-prf-forward-modeling.interpret"
    )
    catalog_identification = _latest_complete(
        investigation, "openstar.tess.catalog-counterpart-identification.analyze"
    )
    catalog_guided_localization = _latest_complete(
        investigation, "openstar.tess.catalog-guided-source-localization.interpret"
    )
    residual_phase_difference_image = _latest_complete(
        investigation, "openstar.tess.residual-phase-difference-imaging.interpret"
    )
    source_switching_temporal = _latest_complete(
        investigation, "openstar.tess.source-switching-temporal-model.interpret"
    )
    time_resolved_localization = _latest_complete(
        investigation, "openstar.tess.time-resolved-residual-phase-localization.interpret"
    )
    time_resolved_frequency = _latest_complete(
        investigation, "openstar.tess.time-resolved-frequency-localization.interpret"
    )
    variability_interpretation = _latest_complete(
        investigation, "openstar.tess.offset-source-variability.interpret"
    )
    gaia_interpretation = _latest_complete(
        investigation,
        "openstar.tess.gaia-source-resolved-counterpart-photometry.interpret",
    )
    nsc_interpretation = _latest_complete(
        investigation, "openstar.tess.nsc-resolved-photometry.interpret"
    )
    atlas_submission = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.prepare"
    )
    atlas_collection = _latest_complete(
        investigation, "openstar.tess.atlas-forced-photometry.collect"
    )
    if (atlas_submission is not None and atlas_collection is None
            and atlas_submission.next_stage is None):
        dependency_id = str((atlas_submission.result or {}).get("externalDependencyID") or "")
        available = (investigation.metadata.get("externalDataAvailability") or {}).get(dependency_id) is True
        return (ScientificBranch(
            id=f"collect-atlas-after-{atlas_submission.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(atlas_submission, "collect-atlas-forced-photometry"),
                handler_id="openstar.tess.atlas-forced-photometry.collect", parameters={},
                triggered_by_stage_id=atlas_submission.id),
            external_data=(ExternalDataDependency(dependency_id, available,
                None if available else "The two persisted ATLAS jobs are still pending."),),
        ),)
    if _awaiting_current_atlas_adapter(investigation):
        from .tess_atlas_forced_photometry import atlas_credentials_available
        des = _latest_complete(
            investigation, "openstar.tess.des-dr2-se-local-forced-photometry.interpret"
        )
        return (
            ScientificBranch(
                id=f"await-current-atlas-adapter-after-{des.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(des, "prepare-atlas-forced-photometry"),
                    handler_id="openstar.tess.atlas-forced-photometry.prepare",
                    parameters={},
                    triggered_by_stage_id=des.id,
                ),
                required_stage_ids=() if atlas_credentials_available() else (
                    "openstar.capability.atlas-forced-photometry-credentials",
                ),
            ),
        )
    if _awaiting_atlas_signed_reanalysis_adapter(investigation):
        atlas = _latest_complete(
            investigation, "openstar.tess.atlas-forced-photometry.interpret"
        )
        return (
            ScientificBranch(
                id=f"await-current-atlas-signed-reanalysis-adapter-after-{atlas.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        atlas, "prepare-atlas-forced-photometry-reanalysis"
                    ),
                    handler_id="openstar.tess.atlas-forced-photometry-reanalysis.prepare",
                    parameters={}, triggered_by_stage_id=atlas.id,
                ),
                required_stage_ids=(
                    "openstar.capability.current-atlas-signed-reanalysis-adapter",
                ),
            ),
        )
    if _awaiting_post_atlas_targeted_observation_adapter(investigation):
        atlas = _latest_complete(
            investigation, "openstar.tess.atlas-forced-photometry.interpret"
        )
        return (
            ScientificBranch(
                id=f"await-current-targeted-observation-adapter-after-{atlas.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        atlas, "generate-targeted-observation-plan"
                    ),
                    handler_id="openstar.tess.targeted-observation-planning.generate",
                    parameters={}, triggered_by_stage_id=atlas.id,
                ),
                required_stage_ids=(),
            ),
        )
    if _awaiting_current_des_adapter(investigation):
        noirlab = _latest_complete(
            investigation, "openstar.tess.noirlab-image-forced-photometry.interpret"
        )
        pair = ((noirlab.result or {}).get("sourcePair") or {})
        adapter_ready = pair.get("version") == "openstar.current-source-pair.v1"
        return (
            ScientificBranch(
                id=f"await-current-des-adapter-after-{noirlab.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        noirlab, "prepare-des-dr2-single-epoch-local-forced-photometry"
                    ),
                    handler_id="openstar.tess.des-dr2-se-local-forced-photometry.prepare",
                    parameters={}, triggered_by_stage_id=noirlab.id,
                ),
                required_stage_ids=() if adapter_ready else (
                    "openstar.capability.current-des-source-pair-adapter",
                ),
            ),
        )
    if _awaiting_noirlab_adapter(investigation):
        return (
            ScientificBranch(
                id=f"continue-noirlab-after-{nsc_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        nsc_interpretation, "prepare-noirlab-image-level-forced-photometry"
                    ),
                    handler_id="openstar.tess.noirlab-image-forced-photometry.prepare",
                    parameters={}, triggered_by_stage_id=nsc_interpretation.id,
                ),
                required_stage_ids=(),
            ),
        )
    archive_continuation = _persisted_archive_continuation(investigation)
    if archive_continuation is not None:
        completed, raw = archive_continuation
        return (
            ScientificBranch(
                id=f"continue-after-{completed.id}",
                experiment=_request_from_persisted(raw),
            ),
        )

    skymapper_interpretation = _latest_complete(
        investigation, "openstar.tess.skymapper-resolved-counterpart-photometry.interpret"
    )
    skymapper_started = any(stage.handler_id.startswith(
        "openstar.tess.skymapper-resolved-counterpart-photometry."
    ) for stage in investigation.stages)
    gaia_result = (gaia_interpretation.result or {}) if gaia_interpretation else {}
    if _awaiting_nsc_adapter(investigation):
        return (
            ScientificBranch(
                id=f"continue-nsc-after-{skymapper_interpretation.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        skymapper_interpretation,
                        "prepare-nsc-resolved-counterpart-photometry",
                    ),
                    handler_id="openstar.tess.nsc-resolved-photometry.prepare",
                    parameters={},
                    triggered_by_stage_id=skymapper_interpretation.id,
                ),
            ),
        )
    if (gaia_interpretation is not None and skymapper_interpretation is None
            and not skymapper_started
            and gaia_result.get("recommendedNextTest") == "SKYMAPPER_RESOLVED_COUNTERPART_PHOTOMETRY"
            and gaia_result.get("physicalMechanismResolved") is False):
        return (ScientificBranch(
            id=f"continue-skymapper-after-{gaia_interpretation.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(
                    gaia_interpretation, "prepare-skymapper-resolved-counterpart-photometry"
                ),
                handler_id="openstar.tess.skymapper-resolved-counterpart-photometry.prepare",
                parameters={}, triggered_by_stage_id=gaia_interpretation.id,
            ),
        ),)
    if gaia_interpretation is not None:
        return ()

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
    # A completed stage 056 is authoritative over the stage-053 continuation.
    # Recreate its direct candidate-validation request after restart, exactly once.
    if time_resolved_frequency is not None:
        result = time_resolved_frequency.result or {}
        candidate = result.get("preferredCandidate") or {}; ids = candidate.get("catalogIDs") or {}
        justified = candidate.get("raDeg") is not None and candidate.get("decDeg") is not None and (
            ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None)
        started = any(stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (not started and justified and result.get("classification") in {
                "STABLE_CANDIDATE_1_LOCALIZATION", "STABLE_CANDIDATE_2_LOCALIZATION"}
                and result.get("recommendedNextTest") ==
                "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"):
            return (ScientificBranch(id=f"continue-counterpart-variability-after-{time_resolved_frequency.id}",
                experiment=StageRequest(id=_continuation_stage_id(time_resolved_frequency,
                    "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare", parameters={},
                    triggered_by_stage_id=time_resolved_frequency.id)),)
        return ()
    if time_resolved_localization is not None:
        localization_result = time_resolved_localization.result or {}
        followup_started = any(
            stage.handler_id.startswith("openstar.tess.time-resolved-frequency-localization.")
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (time_resolved_frequency is None and not followup_started
                and localization_result.get("classification") == "TIME_VARIABLE_LOCALIZATION"
                and localization_result.get("sourceAttributionResolved") is False
                and localization_result.get("physicalMechanismResolved") is False
                and localization_result.get("recommendedNextTest")
                == "TIME_VARIABLE_SOURCE_LOCALIZATION_FOLLOWUP"):
            return (ScientificBranch(
                id=f"continue-frequency-localization-after-{time_resolved_localization.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        time_resolved_localization, "prepare-time-resolved-frequency-localization"),
                    handler_id="openstar.tess.time-resolved-frequency-localization.prepare", parameters={},
                    triggered_by_stage_id=time_resolved_localization.id)),)
        candidate = localization_result.get("preferredCandidate") or {}; ids = candidate.get("catalogIDs") or {}
        validation_started = any(stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (not validation_started and candidate.get("raDeg") is not None
                and candidate.get("decDeg") is not None
                and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None)
                and localization_result.get("classification") in {
                    "STABLE_CANDIDATE_1_LOCALIZATION", "STABLE_CANDIDATE_2_LOCALIZATION"}
                and localization_result.get("sourceAttributionResolved") is True
                and localization_result.get("recommendedNextTest") ==
                "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"):
            return (ScientificBranch(id=f"continue-counterpart-variability-after-{time_resolved_localization.id}",
                experiment=StageRequest(id=_continuation_stage_id(time_resolved_localization,
                    "prepare-offset-source-variability"), handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={}, triggered_by_stage_id=time_resolved_localization.id)),)
        return ()
    # Stage 053 supersedes stage 050 for a resolved temporal candidate.  Check it
    # before the stage-050 terminal guard so restart recovery remains reachable.
    if source_switching_temporal is not None:
        temporal_result = source_switching_temporal.result or {}
        localization_started = any(
            stage.handler_id.startswith("openstar.tess.time-resolved-residual-phase-localization.")
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (time_resolved_localization is None and not localization_started
                and temporal_result.get("classification") == "SECTOR_VARIABLE_MULTI_SOURCE"
                and temporal_result.get("sourceIdentifiable") is True
                and temporal_result.get("sourceAttributionResolved") is False
                and temporal_result.get("physicalMechanismResolved") is False
                and temporal_result.get("recommendedNextTest")
                == "ADDITIONAL_SOURCE_LOCALIZATION_DATA"):
            return (ScientificBranch(
                id=f"continue-time-resolved-localization-after-{source_switching_temporal.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(source_switching_temporal,
                        "prepare-time-resolved-residual-phase-localization"),
                    handler_id="openstar.tess.time-resolved-residual-phase-localization.prepare",
                    parameters={}, triggered_by_stage_id=source_switching_temporal.id)),)
        if localization_started:
            return ()
        candidate = temporal_result.get("preferredCandidate") or {}
        ids = candidate.get("catalogIDs") or {}
        justified = (candidate.get("raDeg") is not None and candidate.get("decDeg") is not None
                     and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None))
        validation_started = any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"} for stage in investigation.stages)
        if (not validation_started and temporal_result.get("classification") in {
                "STATIONARY_CANDIDATE_1_SOURCE", "STATIONARY_CANDIDATE_2_SOURCE"}
                and temporal_result.get("recommendedNextTest")
                == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
                and justified):
            return (ScientificBranch(
                id=f"continue-counterpart-variability-after-{source_switching_temporal.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        source_switching_temporal, "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare", parameters={},
                    triggered_by_stage_id=source_switching_temporal.id),
            ),)
        if validation_started:
            return ()
    # Once independent residual-phase difference imaging has completed, it is
    # authoritative over the older unresolved catalog-guided interpretation.
    # Reproduce the handler's direct continuation after a process restart.
    if residual_phase_difference_image is not None:
        spatial_result = residual_phase_difference_image.result or {}
        candidate = spatial_result.get("preferredCandidate") or {}
        ids = candidate.get("catalogIDs") or {}
        justified = (
            candidate.get("raDeg") is not None
            and candidate.get("decDeg") is not None
            and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None)
        )
        validation_started = any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            and stage.status in {"RUNNING", "COMPLETE"}
            for stage in investigation.stages
        )
        if (
            not validation_started
            and spatial_result.get("classification")
            in {"CANDIDATE_1_SUPPORTED", "CANDIDATE_2_SUPPORTED"}
            and spatial_result.get("sourceAttributionResolved") is True
            and spatial_result.get("recommendedNextTest")
            == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
            and justified
        ):
            return (ScientificBranch(
                id=f"continue-counterpart-variability-after-{residual_phase_difference_image.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        residual_phase_difference_image, "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={}, triggered_by_stage_id=residual_phase_difference_image.id),
            ),)
        # Other spatial outcomes deliberately remain quiescent until their
        # recommended scientific continuation is implemented.
        return ()
    localization_started = any(
        stage.handler_id.startswith("openstar.tess.catalog-guided-source-localization.")
        for stage in investigation.stages
    )
    if (
        catalog_identification is not None
        and catalog_guided_localization is None
        and not localization_started
        and catalog_result.get("recommendedNextTest") == "CATALOG_GUIDED_SOURCE_LOCALIZATION"
        and catalog_result.get("physicalMechanismResolved") is False
        and len(catalog_result.get("plausibleCatalogCandidates") or []) >= 2
    ):
        return (ScientificBranch(
            id=f"continue-catalog-guided-localization-after-{catalog_identification.id}",
            experiment=StageRequest(
                id=_continuation_stage_id(
                    catalog_identification, "prepare-catalog-guided-source-localization"),
                handler_id="openstar.tess.catalog-guided-source-localization.prepare",
                parameters={}, triggered_by_stage_id=catalog_identification.id),
        ),)
    if catalog_guided_localization is not None:
        localization_result = catalog_guided_localization.result or {}
        candidate = localization_result.get("preferredCandidate") or {}
        ids = candidate.get("catalogIDs") or {}
        justified = (candidate.get("raDeg") is not None and candidate.get("decDeg") is not None
                     and (ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None))
        validation_started = any(
            stage.handler_id == "openstar.tess.offset-source-variability.prepare"
            for stage in investigation.stages)
        difference_image_started = any(
            stage.handler_id.startswith("openstar.tess.residual-phase-difference-imaging.")
            for stage in investigation.stages)
        temporal_started = any(
            stage.handler_id.startswith("openstar.tess.source-switching-temporal-model.")
            for stage in investigation.stages)
        if (residual_phase_difference_image is not None
                and source_switching_temporal is None and not temporal_started):
            difference_result = residual_phase_difference_image.result or {}
            if (difference_result.get("classification") == "SOURCE_SWITCHING_BY_SECTOR"
                    and difference_result.get("recommendedNextTest")
                    == "SOURCE_SWITCHING_TEMPORAL_MODEL"):
                return (ScientificBranch(
                    id=f"continue-source-switching-temporal-after-{residual_phase_difference_image.id}",
                    experiment=StageRequest(
                        id=_continuation_stage_id(
                            residual_phase_difference_image,
                            "prepare-source-switching-temporal-model"),
                        handler_id="openstar.tess.source-switching-temporal-model.prepare",
                        parameters={}, triggered_by_stage_id=residual_phase_difference_image.id),
                ),)
        if (residual_phase_difference_image is None and not difference_image_started
                and localization_result.get("recommendedNextTest")
                == "ADDITIONAL_SOURCE_LOCALIZATION_DATA"
                and localization_result.get("classification") == "UNRESOLVED"
                and localization_result.get("sourceAttributionResolved") is False):
            return (ScientificBranch(
                id=f"continue-residual-phase-difference-imaging-after-{catalog_guided_localization.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        catalog_guided_localization,
                        "prepare-residual-phase-difference-imaging"),
                    handler_id="openstar.tess.residual-phase-difference-imaging.prepare",
                    parameters={}, triggered_by_stage_id=catalog_guided_localization.id),
            ),)
        if (not validation_started
                and localization_result.get("sourceAttributionResolved") is True
                and justified
                and localization_result.get("recommendedNextTest")
                == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"):
            return (ScientificBranch(
                id=f"continue-counterpart-variability-after-{catalog_guided_localization.id}",
                experiment=StageRequest(
                    id=_continuation_stage_id(
                        catalog_guided_localization, "prepare-offset-source-variability"),
                    handler_id="openstar.tess.offset-source-variability.prepare",
                    parameters={}, triggered_by_stage_id=catalog_guided_localization.id),
            ),)
        return ()
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
