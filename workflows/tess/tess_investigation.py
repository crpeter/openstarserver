from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import (
    ArtifactReference,
    Investigation,
    InvestigationStore,
    sha256_file,
    sha256_json,
)
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine

from .tess_claims import validate_claim
from .tess_followup import (
    build_low_frequency_followup,
    build_single_target_primary,
)
from .tess_hypotheses import analyze, interpret_followup, plan
from .tess_identity import collect_identity


WORKFLOW_ID = "openstar.workflow.tess-investigation.v1"
WORKFLOW_VERSION = "20.1"
SOFTWARE_ID = "openstar.tess-investigation-plugin"
SOFTWARE_VERSION = "20.1"


def _stage(investigation: Investigation, stage_id: str):
    for stage in investigation.stages:
        if stage.id == stage_id:
            return stage
    raise KeyError(f"Investigation stage not found: {stage_id}")


def _result(investigation: Investigation, stage_id: str) -> dict[str, Any]:
    stage = _stage(investigation, stage_id)
    if stage.status != "COMPLETE" or stage.result is None:
        raise RuntimeError(f"Stage is not COMPLETE with a result: {stage_id}")
    return stage.result


def _artifact(path: Path, media_type: str) -> ArtifactReference:
    return ArtifactReference(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        media_type=media_type,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _target_result(project_status: dict[str, Any]) -> dict[str, Any]:
    datasets = project_status.get("datasets") or []
    if len(datasets) != 1:
        raise RuntimeError(
            "The v20.1 TESS investigation primary/follow-up project must contain exactly one dataset; "
            f"got {len(datasets)}."
        )
    return dict(datasets[0])


def _render_report(conclusion: dict[str, Any]) -> str:
    claim = conclusion["claim"]
    target = conclusion["target"]
    lines = [
        "# OpenStar TESS Investigation",
        "",
        f"- Investigation: `{conclusion['investigationID']}`",
        f"- TIC: `{target['ticID']}`",
        f"- Target: {target.get('targetName') or target.get('datasetID')}",
        f"- Claim level: **{claim['claim']}**",
        f"- Selected period: {conclusion.get('selectedPeriodDays') if conclusion.get('selectedPeriodDays') is not None else '[none]'} days",
        f"- Selected source: {conclusion.get('selectedSource') or '[none]'}",
        "",
        "## Rationale",
        "",
    ]
    for reason in claim.get("rationale") or []:
        lines.append(f"- {reason}")

    analysis = conclusion.get("primaryAnalysis") or {}
    rotation = analysis.get("rotationSanity") or {}
    lines.extend([
        "",
        "## Primary distributed result",
        "",
        f"- Period status: {analysis.get('periodStatus')}",
        f"- Confidence: {analysis.get('periodConfidence')}",
        f"- Observed period: {analysis.get('observedPeriodDays')} days",
        f"- Preferred relation: {analysis.get('preferredPhysicalPeriodRelation')}",
        "",
        "## Catalog / physical checks",
        "",
        f"- Catalog period match: {'yes' if analysis.get('bestCatalogMatch') else 'no'}",
        f"- Rotation sanity: {rotation.get('status')}",
    ])

    if conclusion.get("followup") is not None:
        followup = conclusion["followup"]
        lines.extend([
            "",
            "## Follow-up",
            "",
            f"- Trigger: {conclusion.get('planner', {}).get('reason')}",
            f"- Reliable: {followup.get('followupReliable')}",
            f"- Selected follow-up period: {followup.get('selectedPeriodDays')} days",
        ])

    lines.extend([
        "",
        "## Claim policy",
        "",
        "This report is produced by deterministic OpenStar rules. It never automatically emits `DISCOVERY`.",
        "",
    ])
    return "\n".join(lines)


def build_engine(
    store: InvestigationStore,
    coordinator: OpenStarCoordinatorClient,
    *,
    poll_interval: float,
    timeout: float | None,
) -> WorkflowEngine:
    engine = WorkflowEngine(store)

    def prepare_target(investigation, request):
        source_project = Path(request.parameters["projectPath"]).expanduser().resolve()
        if not source_project.exists():
            raise FileNotFoundError(source_project)

        artifact_root = store.directory_for(investigation.id) / "artifacts"
        prepared = build_single_target_primary(
            source_project_path=source_project,
            output_dir=artifact_root,
            investigation_id=investigation.id,
            dataset_id=request.parameters.get("datasetID"),
            tic_id=request.parameters.get("ticID"),
        )
        source_dataset = Path(prepared["datasetPath"])
        primary_manifest = Path(prepared["projectPath"])

        print("🔒 TESS target frozen for investigation")
        print(f"   target: {prepared.get('targetName')}")
        print(f"   dataset: {prepared['datasetID']}")
        print(f"   TIC: {prepared['ticID']}")
        print(f"   primary project: {prepared['projectID']}")

        return StageOutcome(
            result=prepared,
            next_stage=StageRequest(
                id="002-primary-distributed-search",
                handler_id="openstar.tess.primary-project.run",
                parameters={
                    "projectPath": prepared["projectPath"],
                    "projectID": prepared["projectID"],
                },
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "sourceProjectManifest": sha256_file(source_project),
                "sourceDataset": sha256_file(source_dataset),
            },
            artifacts=(_artifact(primary_manifest, "application/json"),),
        )

    def run_primary(investigation, request):
        print("⚙️ Activating primary distributed period search")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        target = _target_result(run.status)
        print("✅ Primary distributed search complete")
        print(f"   period status: {target.get('periodStatus')}")
        print(f"   preferred period: {target.get('preferredPhysicalPeriodDays')} days")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id="003-catalog-identity",
                handler_id="openstar.tess.catalog-identity",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def identity_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        primary = _target_result(_result(investigation, "002-primary-distributed-search"))
        tic_id = int(prepared["ticID"])

        print("🌐 Resolving TIC / SIMBAD / VSX / Gaia / TESS identity")
        identity = collect_identity(tic_id)
        artifact_path = store.directory_for(investigation.id) / "artifacts" / "identity" / "identity.json"
        _write_json(artifact_path, identity)
        print(f"   identity resolved: {identity.get('identityResolved')}")
        print(f"   catalog query errors: {len(identity.get('queryErrors') or [])}")

        return StageOutcome(
            result=identity,
            next_stage=StageRequest(
                id="004-hypotheses",
                handler_id="openstar.tess.hypotheses",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"primaryTargetResult": sha256_json(primary)},
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def hypothesis_stage(investigation, request):
        primary = _target_result(_result(investigation, "002-primary-distributed-search"))
        identity = _result(investigation, "003-catalog-identity")
        analysis = analyze(primary, identity)
        print("🧠 Deterministic TESS hypotheses evaluated")
        print(f"   reliable primary: {analysis.get('primaryReliable')}")
        print(f"   catalog period match: {bool(analysis.get('bestCatalogMatch'))}")
        print(f"   rotation sanity: {(analysis.get('rotationSanity') or {}).get('status')}")
        return StageOutcome(
            result=analysis,
            next_stage=StageRequest(
                id="005-planner",
                handler_id="openstar.tess.planner",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "primaryTargetResult": sha256_json(primary),
                "identity": sha256_json(identity),
            },
        )

    def planner_stage(investigation, request):
        identity = _result(investigation, "003-catalog-identity")
        analysis = _result(investigation, "004-hypotheses")
        planned = plan(analysis, identity)
        print("🧭 Deterministic planner")
        print(f"   action: {planned['action']}")
        print(f"   reason: {planned['reason']}")

        if planned["action"] == "LOW_FREQUENCY_FOLLOWUP":
            next_stage = StageRequest(
                id="006-prepare-followup",
                handler_id="openstar.tess.followup.prepare-low-frequency",
                parameters={},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id="006-finalize",
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=planned,
            next_stage=next_stage,
            input_hashes={"hypothesisAnalysis": sha256_json(analysis)},
        )

    def prepare_followup(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        follow = build_low_frequency_followup(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_path=prepared["datasetPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        dataset_path = Path(follow["datasetPath"])
        manifest_path = Path(follow["projectPath"])
        print("🔬 Decisive lower-frequency follow-up prepared")
        print(
            "   frequency range: "
            f"{follow['frequencySearch']['minimumFrequency']:.6f} - "
            f"{follow['frequencySearch']['maximumFrequency']:.6f} cycles/day"
        )
        print(
            "   work units: "
            f"{(follow['frequencySearch']['totalFrequencies'] + follow['frequencySearch']['frequenciesPerWorkUnit'] - 1) // follow['frequencySearch']['frequenciesPerWorkUnit']}"
        )
        return StageOutcome(
            result=follow,
            next_stage=StageRequest(
                id="007-run-followup",
                handler_id="openstar.tess.followup.run",
                parameters={"projectPath": follow["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"sourceDataset": sha256_file(prepared["datasetPath"])},
            artifacts=(
                _artifact(dataset_path, "application/json"),
                _artifact(manifest_path, "application/json"),
            ),
        )

    def run_followup(investigation, request):
        print("⚙️ Activating distributed lower-frequency follow-up")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        target = _target_result(run.status)
        print("✅ Follow-up distributed search complete")
        print(f"   period status: {target.get('periodStatus')}")
        print(f"   preferred period: {target.get('preferredPhysicalPeriodDays')} days")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id="008-interpret-followup",
                handler_id="openstar.tess.followup.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def interpret_stage(investigation, request):
        analysis = _result(investigation, "004-hypotheses")
        followup = _result(investigation, "007-run-followup")
        interpreted = interpret_followup(analysis, followup)
        print("🔎 Follow-up interpretation")
        print(f"   claim: {interpreted['claimDecision']['claim']}")
        print(f"   selected period: {interpreted.get('selectedPeriodDays')} days")
        return StageOutcome(
            result=interpreted,
            next_stage=StageRequest(
                id="009-finalize",
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"followupResult": sha256_json(followup)},
        )

    def finalize_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        primary_analysis = _result(investigation, "004-hypotheses")
        planner = _result(investigation, "005-planner")

        followup_interpretation = None
        if any(stage.id == "008-interpret-followup" for stage in investigation.stages):
            followup_interpretation = _result(investigation, "008-interpret-followup")
            claim_decision = followup_interpretation["claimDecision"]
            selected_period = followup_interpretation.get("selectedPeriodDays")
            selected_source = followup_interpretation.get("selectedSource")
        else:
            claim_decision = planner.get("claimDecision")
            selected_period = primary_analysis.get("observedPeriodDays")
            selected_source = "primary-distributed-search"

        if not claim_decision:
            raise RuntimeError("Finalization reached without a claim decision.")
        validate_claim(claim_decision["claim"])

        conclusion = {
            "investigationID": investigation.id,
            "workflowID": WORKFLOW_ID,
            "workflowVersion": WORKFLOW_VERSION,
            "target": {
                "datasetID": prepared["datasetID"],
                "ticID": prepared["ticID"],
                "targetName": prepared.get("targetName"),
                "sector": prepared.get("sector"),
            },
            "claim": claim_decision,
            "selectedPeriodDays": selected_period,
            "selectedSource": selected_source,
            "primaryAnalysis": primary_analysis,
            "planner": planner,
            "followup": followup_interpretation,
            "automaticDiscoveryClaim": False,
        }

        output_dir = store.directory_for(investigation.id)
        conclusion_path = output_dir / "conclusion.json"
        report_path = output_dir / "report.md"
        _write_json(conclusion_path, conclusion)
        report_path.write_text(_render_report(conclusion), encoding="utf-8")

        final_status = (
            "HUMAN_REVIEW_REQUIRED"
            if claim_decision["claim"] == "HUMAN_REVIEW_REQUIRED"
            else "COMPLETE"
        )
        print("🏁 TESS investigation conclusion")
        print(f"   claim: {claim_decision['claim']}")
        print(f"   selected period: {selected_period} days")
        print(f"   report: {report_path}")

        project_ids = tuple(
            str(stage.provenance.project_ids[0])
            for stage in investigation.stages
            if stage.provenance is not None and stage.provenance.project_ids
        )
        return StageOutcome(
            result=conclusion,
            stop=True,
            final_status=final_status,
            input_hashes={
                "primaryAnalysis": sha256_json(primary_analysis),
                "planner": sha256_json(planner),
            },
            project_ids=project_ids,
            artifacts=(
                _artifact(conclusion_path, "application/json"),
                _artifact(report_path, "text/markdown"),
            ),
        )

    engine.register_handler("openstar.tess.prepare-target", prepare_target)
    engine.register_handler("openstar.tess.primary-project.run", run_primary)
    engine.register_handler("openstar.tess.catalog-identity", identity_stage)
    engine.register_handler("openstar.tess.hypotheses", hypothesis_stage)
    engine.register_handler("openstar.tess.planner", planner_stage)
    engine.register_handler("openstar.tess.followup.prepare-low-frequency", prepare_followup)
    engine.register_handler("openstar.tess.followup.run", run_followup)
    engine.register_handler("openstar.tess.followup.interpret", interpret_stage)
    engine.register_handler("openstar.tess.finalize", finalize_stage)
    return engine
