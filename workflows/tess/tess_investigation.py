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
from .tess_hypotheses import (
    analyze,
    interpret_broad_independent_sectors,
    interpret_followup,
    interpret_independent_sectors,
    plan,
    plan_independent_contradiction_resolution,
)
from .tess_identity import collect_identity
from .tess_morphology import analyze_morphology
from .tess_multisector import (
    build_broad_independent_sector_project,
    build_independent_sector_project,
)


WORKFLOW_ID = "openstar.workflow.tess-investigation.v1"
WORKFLOW_VERSION = "20.2"
SOFTWARE_ID = "openstar.tess-investigation-plugin"
SOFTWARE_VERSION = "20.4"


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


def _latest_result_for_handler(
    investigation: Investigation,
    handler_id: str,
) -> dict[str, Any] | None:
    for stage in reversed(investigation.stages):
        if stage.handler_id == handler_id and stage.status == "COMPLETE":
            return stage.result
    return None


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


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_baseline_days(path: str | Path) -> float | None:
    dataset = _load_json(path)
    source = dataset.get("source") or {}
    if source.get("baselineDays") is not None:
        try:
            return float(source["baselineDays"])
        except (TypeError, ValueError):
            pass
    times = dataset.get("times") or []
    if len(times) > 1:
        return float(times[-1]) - float(times[0])
    return None


def _target_result(project_status: dict[str, Any]) -> dict[str, Any]:
    datasets = project_status.get("datasets") or []
    if len(datasets) != 1:
        raise RuntimeError(
            "A primary/same-dataset TESS investigation project must contain exactly one dataset; "
            f"got {len(datasets)}."
        )
    return dict(datasets[0])


def _next_stage_id(current_id: str, label: str) -> str:
    try:
        number = int(str(current_id).split("-", 1)[0]) + 1
    except (TypeError, ValueError):
        raise ValueError(f"Stage id must begin with an integer prefix: {current_id}")
    return f"{number:03d}-{label}"



def _build_period_evidence(
    *,
    claim_decision: dict[str, Any],
    selected_period: float | None,
    selected_source: str | None,
    primary_analysis: dict[str, Any],
    followup_interpretation: dict[str, Any] | None,
    independent_interpretation: dict[str, Any] | None,
    broad_interpretation: dict[str, Any] | None,
    harmonic_family_interpretation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate recurrent photometric evidence from a resolved physical cycle."""
    harmonic_result = harmonic_family_interpretation or broad_interpretation or {}
    family = harmonic_result.get("harmonicFamily") or {}

    recurrent = family.get("representativeRawPeriodDays")
    possible_cycle = family.get("possibleDoubleCycleDays")
    if recurrent is None:
        recurrent = selected_period

    family_interpretation = str(family.get("interpretation") or "")
    physical_cycle_resolved = bool(family.get("physicalCycleResolved"))
    claim = str(claim_decision.get("claim") or "")

    if not possible_cycle and claim in {
        "KNOWN_PERIOD_RECOVERED",
        "KNOWN_PHENOMENON_EXPLAINED",
        "INDEPENDENT_PERIOD_ESTIMATE",
    }:
        physical_cycle_resolved = selected_period is not None

    if family_interpretation == "possible-double-wave-period-family":
        physical_cycle_resolved = False

    physical_period = selected_period if physical_cycle_resolved else None

    return {
        "candidatePeriodDays": selected_period,
        "candidateSource": selected_source,
        "recurrentPhotometricPeriodDays": recurrent,
        "possiblePhysicalCycleDays": possible_cycle,
        "physicalCycleResolved": physical_cycle_resolved,
        "physicalPeriodDays": physical_period,
        "primaryRawPeriodDays": primary_analysis.get("rawCandidatePeriodDays"),
        "primaryPreferredCycleDays": primary_analysis.get("observedPeriodDays"),
        "sameSectorCandidateDays": (followup_interpretation or {}).get("selectedPeriodDays"),
        "independentTargetedCandidateDays": (independent_interpretation or {}).get("selectedPeriodDays"),
        "interpretation": (
            "physical-cycle-resolved"
            if physical_cycle_resolved
            else "photometric-period-family-physical-cycle-unresolved"
        ),
    }

def _render_report(conclusion: dict[str, Any]) -> str:
    claim = conclusion["claim"]
    target = conclusion["target"]
    analysis = conclusion.get("primaryAnalysis") or {}
    rotation = analysis.get("rotationSanity") or {}
    preferred_coverage = analysis.get("preferredCycleCoverage") or {}

    lines = [
        "# OpenStar TESS Investigation",
        "",
        f"- Investigation: `{conclusion['investigationID']}`",
        f"- TIC: `{target['ticID']}`",
        f"- Target: {target.get('targetName') or target.get('datasetID')}",
        f"- Claim level: **{claim['claim']}**",
    ]
    period_evidence = conclusion.get("periodEvidence") or {}
    if period_evidence.get("recurrentPhotometricPeriodDays") is not None:
        lines.append(
            f"- Recurrent photometric periodicity: {period_evidence.get('recurrentPhotometricPeriodDays')} days"
        )
    if period_evidence.get("possiblePhysicalCycleDays") is not None:
        lines.append(
            f"- Possible double-wave / physical cycle: {period_evidence.get('possiblePhysicalCycleDays')} days"
        )
    if period_evidence.get("physicalCycleResolved"):
        lines.append(f"- Physical period: {period_evidence.get('physicalPeriodDays')} days")
    else:
        lines.append("- Physical period: **unresolved**")
    lines.extend([
        f"- Evidence source: {period_evidence.get('candidateSource') or '[none]'}",
        "",
        "## Rationale",
        "",
    ])
    for reason in claim.get("rationale") or []:
        lines.append(f"- {reason}")

    lines.extend([
        "",
        "## Primary distributed result",
        "",
        f"- Period status: {analysis.get('periodStatus')}",
        f"- Confidence: {analysis.get('periodConfidence')}",
        f"- Raw candidate period: {analysis.get('rawCandidatePeriodDays')} days",
        f"- Preferred period: {analysis.get('observedPeriodDays')} days",
        f"- Preferred relation: {analysis.get('preferredPhysicalPeriodRelation')}",
        f"- Observation baseline: {analysis.get('observationBaselineDays')} days",
        f"- Preferred-cycle coverage: {preferred_coverage.get('observedCycles')}",
        "",
        "## Catalog / physical checks",
        "",
        f"- Catalog period match: {'yes' if analysis.get('bestCatalogMatch') else 'no'}",
        f"- Rotation sanity: {rotation.get('status')}",
        f"- Equatorial speed: {rotation.get('equatorialSpeedKmS')} km/s",
        f"- Minimum mass for subcritical rotation: {rotation.get('minimumMassForSubcriticalRotationMsun')} M_sun",
    ])

    if conclusion.get("followup") is not None:
        followup = conclusion["followup"]
        diagnostics = followup.get("diagnostics") or {}
        coverage = diagnostics.get("cycleCoverage") or {}
        lines.extend([
            "",
            "## Same-sector hypothesis follow-up",
            "",
            f"- Trigger: {conclusion.get('planner', {}).get('reason')}",
            f"- Reliable: {followup.get('followupReliable')}",
            f"- Selected period: {followup.get('selectedPeriodDays')} days",
            f"- Boundary hit: {diagnostics.get('boundaryHit')}",
            f"- Observed cycles: {coverage.get('observedCycles')}",
        ])

    independent = conclusion.get("independentVerification")
    if independent is not None:
        lines.extend([
            "",
            "## Independent TESS-sector verification",
            "",
            f"- Eligible sectors: {independent.get('eligibleSectorCount')}",
            f"- Supporting sectors: {independent.get('supportingSectorCount')}",
            f"- Required supporting sectors: {independent.get('requiredSupportingSectorCount')}",
        ])
        for item in independent.get("sectorResults") or []:
            coverage = item.get("cycleCoverage") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: period={item.get('candidatePeriodDays')} d, "
                f"cycles={coverage.get('observedCycles')}, "
                f"support={item.get('supportsTarget')}"
            )

    broad = conclusion.get("independentBroadVerification")
    if broad is not None:
        lines.extend([
            "",
            "## Contradiction-resolution broad independent search",
            "",
            f"- Eligible sectors: {broad.get('eligibleSectorCount')}",
            f"- Required cluster support: {broad.get('requiredClusterSupportCount')}",
            f"- Best cluster: {broad.get('bestCluster')}",
        ])
        for item in broad.get("sectorResults") or []:
            coverage = item.get("cycleCoverage") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: period={item.get('candidatePeriodDays')} d, "
                f"cycles={coverage.get('observedCycles')}, "
                f"prominence={item.get('candidatePeakProminenceRatio')}, "
                f"eligible={item.get('eligibleForClustering')}"
            )

    harmonic = conclusion.get("independentHarmonicFamilyVerification")
    if harmonic is not None:
        family = harmonic.get("harmonicFamily") or {}
        cluster = harmonic.get("bestCluster") or {}
        lines.extend([
            "",
            "## Harmonic-family reinterpretation",
            "",
            f"- Promotion eligible: {harmonic.get('promotionEligible')}",
            f"- Promotion blockers: {harmonic.get('promotionBlockers')}",
            f"- Supporting sectors: {cluster.get('sectors')}",
            f"- Representative raw periodicity: {family.get('representativeRawPeriodDays')} days",
            f"- Possible 2x physical cycle: {family.get('possibleDoubleCycleDays')} days",
            f"- Physical cycle resolved: {family.get('physicalCycleResolved')}",
        ])

    morphology = conclusion.get("morphology")
    if morphology is not None:
        lines.extend([
            "",
            "## Morphology / physical-cycle discrimination",
            "",
            f"- Morphology class: {morphology.get('morphologyClass')}",
            f"- Phenomenology: {morphology.get('phenomenology')}",
            f"- Eligible sectors: {morphology.get('eligibleSectorCount')}",
            f"- Independent eligible sectors: {morphology.get('independentEligibleSectorCount')}",
            f"- Required independent support: {morphology.get('requiredIndependentSupportCount')}",
            f"- Raw-cycle supporters: {morphology.get('rawCycleSupportingSectors')}",
            f"- Double-cycle supporters: {morphology.get('doubleCycleSupportingSectors')}",
            f"- Physical cycle resolved: {morphology.get('physicalCycleResolved')}",
            f"- Resolved physical period: {morphology.get('resolvedPhysicalPeriodDays')} days",
        ])
        for item in morphology.get("sectorResults") or []:
            double_metrics = item.get("doubleWaveMetrics") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: rawEV={(item.get('rawProfile') or {}).get('explainedVariance')}, "
                f"doubleEV={(item.get('doubleProfile') or {}).get('explainedVariance')}, "
                f"doubleGain={item.get('doubleExplainedVarianceImprovement')}, "
                f"halfDiff={double_metrics.get('halfCycleDifferenceRatio')}, "
                f"rawSupport={item.get('supportsRawCycle')}, "
                f"doubleSupport={item.get('supportsDoubleCycle')}"
            )

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
        prepared["observationBaselineDays"] = _dataset_baseline_days(source_dataset)

        print("🔒 TESS target frozen for investigation")
        print(f"   target: {prepared.get('targetName')}")
        print(f"   dataset: {prepared['datasetID']}")
        print(f"   TIC: {prepared['ticID']}")
        print(f"   sector: {prepared.get('sector')}")
        print(f"   baseline: {prepared.get('observationBaselineDays')} days")
        print(f"   primary project: {prepared['projectID']}")

        return StageOutcome(
            result=prepared,
            next_stage=StageRequest(
                id="002-primary-distributed-search",
                handler_id="openstar.tess.primary-project.run",
                parameters={"projectPath": prepared["projectPath"]},
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
        print(f"   raw period: {target.get('candidatePeriodDays')} days")
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
        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "identity"
            / "identity.json"
        )
        _write_json(artifact_path, identity)
        print(f"   identity resolved: {identity.get('identityResolved')}")
        print(f"   catalog query errors: {len(identity.get('queryErrors') or [])}")
        tess = identity.get("tess") or {}
        print(f"   official TESS sectors: {tess.get('officialSectors') or []}")

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
        prepared = _result(investigation, "001-prepare-target")
        primary = _target_result(_result(investigation, "002-primary-distributed-search"))
        identity = _result(investigation, "003-catalog-identity")
        analysis = analyze(
            primary,
            identity,
            observation_baseline_days=prepared.get("observationBaselineDays"),
        )
        print("🧠 Deterministic TESS hypotheses evaluated")
        print(f"   reliable primary: {analysis.get('primaryReliable')}")
        print(f"   catalog period match: {bool(analysis.get('bestCatalogMatch'))}")
        print(f"   rotation sanity: {(analysis.get('rotationSanity') or {}).get('status')}")
        coverage = analysis.get("preferredCycleCoverage") or {}
        print(f"   preferred-cycle coverage: {coverage.get('observedCycles')}")
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
        elif planned["action"] == "INDEPENDENT_SECTOR_FOLLOWUP":
            next_stage = StageRequest(
                id="006-prepare-independent-sectors",
                handler_id="openstar.tess.independent.prepare",
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
        analysis = _result(investigation, "004-hypotheses")
        planner = _result(investigation, "005-planner")
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        follow = build_low_frequency_followup(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_path=prepared["datasetPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            output_dir=artifact_root,
            investigation_id=investigation.id,
            trigger_reason=planner.get("reason"),
            primary_period_days=analysis.get("observedPeriodDays"),
        )
        dataset_path = Path(follow["datasetPath"])
        manifest_path = Path(follow["projectPath"])
        print("🔬 Decisive same-sector frequency follow-up prepared")
        print(f"   mode: {follow.get('followupMode')}")
        print(f"   target period: {follow.get('targetPeriodDays')} days")
        print(
            "   frequency range: "
            f"{follow['frequencySearch']['minimumFrequency']:.6f} - "
            f"{follow['frequencySearch']['maximumFrequency']:.6f} cycles/day"
        )
        print(f"   source baseline: {follow.get('sourceBaselineDays')} days")
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
        print("⚙️ Activating distributed same-sector follow-up")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        target = _target_result(run.status)
        print("✅ Same-sector follow-up complete")
        print(f"   period status: {target.get('periodStatus')}")
        print(f"   candidate period: {target.get('candidatePeriodDays')} days")
        print(f"   reducer preferred period: {target.get('preferredPhysicalPeriodDays')} days")
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
        identity = _result(investigation, "003-catalog-identity")
        followup_spec = _result(investigation, "006-prepare-followup")
        followup = _result(investigation, "007-run-followup")
        interpreted = interpret_followup(
            analysis,
            followup,
            followup_spec=followup_spec,
            identity=identity,
        )
        print("🔎 Same-sector follow-up interpretation")
        print(f"   claim: {interpreted['claimDecision']['claim']}")
        print(f"   selected period: {interpreted.get('selectedPeriodDays')} days")
        coverage = (interpreted.get("diagnostics") or {}).get("cycleCoverage") or {}
        print(f"   observed cycles: {coverage.get('observedCycles')}")

        if (
            interpreted["claimDecision"]["claim"] == "CANDIDATE_PERIOD"
            and interpreted.get("selectedPeriodDays") is not None
        ):
            next_stage = StageRequest(
                id="009-prepare-independent-sectors",
                handler_id="openstar.tess.independent.prepare",
                parameters={},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id="009-finalize",
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=interpreted,
            next_stage=next_stage,
            input_hashes={"followupResult": sha256_json(followup)},
        )

    def prepare_independent(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        identity = _result(investigation, "003-catalog-identity")
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        planner = _result(investigation, "005-planner")
        analysis = _result(investigation, "004-hypotheses")

        if followup_interpretation is not None:
            target_period = followup_interpretation.get("selectedPeriodDays")
        else:
            target_period = analysis.get("observedPeriodDays")

        if target_period is None:
            raise RuntimeError("Independent-sector follow-up has no candidate period to test.")

        official_sectors = ((identity.get("tess") or {}).get("officialSectors") or [])
        artifact_root = store.directory_for(investigation.id) / "artifacts"

        print("🛰 Preparing independent TESS-sector verification")
        print(f"   primary sector: {prepared.get('sector')}")
        print(f"   target period: {target_period} days")
        print(f"   catalog official sectors: {official_sectors}")

        spec = build_independent_sector_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            tic_id=int(prepared["ticID"]),
            primary_sector=prepared.get("sector"),
            target_period_days=float(target_period),
            candidate_sectors=list(official_sectors),
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )

        print(f"   prepared independent sectors: {[item.get('sector') for item in spec.get('preparedSectors') or []]}")
        if spec.get("errors"):
            print(f"   sector preparation errors: {len(spec['errors'])}")

        artifacts: list[ArtifactReference] = []
        for item in spec.get("preparedSectors") or []:
            artifacts.append(_artifact(Path(item["datasetPath"]), "application/json"))
        if spec.get("projectPath"):
            artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))

        if spec.get("available"):
            print(f"   independent work units: {spec.get('totalWorkUnits')}")
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "run-independent-sectors"),
                handler_id="openstar.tess.independent.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            )
        else:
            print("   no independent TESS sectors could be prepared")
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=spec,
            next_stage=next_stage,
            input_hashes={
                "identity": sha256_json(identity),
                "planner": sha256_json(planner),
            },
            artifacts=tuple(artifacts),
        )

    def run_independent(investigation, request):
        print("⚙️ Activating independent TESS-sector verification")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Independent-sector distributed search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-independent-sectors"),
                handler_id="openstar.tess.independent.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def interpret_independent(investigation, request):
        spec = _latest_result_for_handler(investigation, "openstar.tess.independent.prepare")
        run = _latest_result_for_handler(investigation, "openstar.tess.independent.run")
        if spec is None or run is None:
            raise RuntimeError("Independent-sector interpretation is missing its prepare/run stages.")

        target_period = float(spec["targetPeriodDays"])
        interpreted = interpret_independent_sectors(
            target_period_days=target_period,
            project_status=run,
            independent_spec=spec,
        )
        print("🔭 Independent-sector recurrence interpretation")
        print(f"   eligible sectors: {interpreted.get('eligibleSectorCount')}")
        print(f"   supporting sectors: {interpreted.get('supportingSectorCount')}")
        print(f"   required support: {interpreted.get('requiredSupportingSectorCount')}")
        print(f"   claim: {interpreted['claimDecision']['claim']}")
        print(f"   selected period: {interpreted.get('selectedPeriodDays')} days")

        contradiction_plan = plan_independent_contradiction_resolution(
            interpreted
        )
        print("🧭 Independent contradiction planner")
        print(f"   action: {contradiction_plan['action']}")
        print(f"   reason: {contradiction_plan['reason']}")
        print(f"   reliable sectors: {contradiction_plan.get('reliableSectorCount')}")
        print(f"   boundary hits: {contradiction_plan.get('boundaryHitCount')}")

        if contradiction_plan["action"] == "BROAD_INDEPENDENT_SEARCH":
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "prepare-broad-independent-search"),
                handler_id="openstar.tess.independent.broad.prepare",
                parameters={"continuation": False},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        interpreted = dict(interpreted)
        interpreted["contradictionPlan"] = contradiction_plan
        return StageOutcome(
            result=interpreted,
            next_stage=next_stage,
            input_hashes={"independentProjectResult": sha256_json(run)},
        )

    def prepare_broad_independent(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        targeted_spec = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        if targeted_spec is None:
            raise RuntimeError(
                "Broad independent search requires the frozen independent-sector preparation."
            )

        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🌐 Preparing target-independent broad independent-sector search")
        print("   reusing frozen sectors; no MAST download")
        spec = build_broad_independent_sector_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            independent_spec=targeted_spec,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        spec["continuation"] = bool(request.parameters.get("continuation"))
        search = spec.get("frequencySearch") or {}
        print(
            "   broad frequency range: "
            f"{search.get('minimumFrequency'):.6f} - "
            f"{search.get('maximumFrequency'):.6f} cycles/day"
        )
        print(
            "   reused sectors: "
            f"{[item.get('sector') for item in spec.get('preparedSectors') or []]}"
        )
        print(f"   broad work units: {spec.get('totalWorkUnits')}")

        artifacts: list[ArtifactReference] = []
        for item in spec.get("preparedSectors") or []:
            artifacts.append(_artifact(Path(item["datasetPath"]), "application/json"))
        if spec.get("projectPath"):
            artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))

        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-broad-independent-search"),
                handler_id="openstar.tess.independent.broad.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"targetedIndependentSpec": sha256_json(targeted_spec)},
            artifacts=tuple(artifacts),
        )

    def run_broad_independent(investigation, request):
        print("⚙️ Activating target-independent broad sector search")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Broad independent-sector search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-broad-independent-search"),
                handler_id="openstar.tess.independent.broad.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def interpret_broad_independent(investigation, request):
        spec = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.run",
        )
        if spec is None or run is None:
            raise RuntimeError(
                "Broad independent interpretation is missing its prepare/run stages."
            )

        primary_analysis = _result(investigation, "004-hypotheses")
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        interpreted = interpret_broad_independent_sectors(
            project_status=run,
            broad_spec=spec,
            primary_raw_period_days=primary_analysis.get("rawCandidatePeriodDays"),
            primary_preferred_period_days=primary_analysis.get("observedPeriodDays"),
            same_sector_candidate_days=(
                (followup_interpretation or {}).get("selectedPeriodDays")
            ),
        )
        cluster = interpreted.get("bestCluster") or {}
        print("🧩 Independent-sector period clustering")
        print(f"   eligible sectors: {interpreted.get('eligibleSectorCount')}")
        print(f"   best cluster sectors: {cluster.get('sectors') or []}")
        print(f"   cluster median period: {cluster.get('medianPeriodDays')} days")
        print(f"   required support: {interpreted.get('requiredClusterSupportCount')}")
        print(f"   promotion eligible: {interpreted.get('promotionEligible')}")
        blockers = interpreted.get("promotionBlockers") or []
        if blockers:
            print(f"   promotion blockers: {blockers}")
        harmonic_family = interpreted.get("harmonicFamily") or {}
        if harmonic_family:
            print(
                "   recurrent raw family: "
                f"{harmonic_family.get('representativeRawPeriodDays')} days"
            )
            print(
                "   possible 2x cycle: "
                f"{harmonic_family.get('possibleDoubleCycleDays')} days"
            )
        print(f"   claim: {interpreted['claimDecision']['claim']}")

        finalize_parameters = {}
        if request.parameters.get("outputSuffix"):
            finalize_parameters["outputSuffix"] = request.parameters["outputSuffix"]
        elif spec.get("continuation"):
            finalize_parameters["outputSuffix"] = "v20.3.1"

        return StageOutcome(
            result=interpreted,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters=finalize_parameters,
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"broadIndependentProjectResult": sha256_json(run)},
        )

    def reinterpret_harmonic_family(investigation, request):
        spec = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.run",
        )
        if spec is None or run is None:
            raise RuntimeError(
                "Harmonic-family reinterpretation requires completed broad independent prepare/run stages."
            )

        primary_analysis = _result(investigation, "004-hypotheses")
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        interpreted = interpret_broad_independent_sectors(
            project_status=run,
            broad_spec=spec,
            primary_raw_period_days=primary_analysis.get("rawCandidatePeriodDays"),
            primary_preferred_period_days=primary_analysis.get("observedPeriodDays"),
            same_sector_candidate_days=(
                (followup_interpretation or {}).get("selectedPeriodDays")
            ),
        )

        cluster = interpreted.get("bestCluster") or {}
        harmonic_family = interpreted.get("harmonicFamily") or {}
        print("🧬 Reinterpreting independent evidence as a harmonic family")
        print(f"   eligible sectors: {interpreted.get('eligibleSectorCount')}")
        print(f"   best raw cluster sectors: {cluster.get('sectors') or []}")
        print(f"   raw family median: {cluster.get('medianPeriodDays')} days")
        if harmonic_family:
            print(
                "   possible 2x physical cycle: "
                f"{harmonic_family.get('possibleDoubleCycleDays')} days"
            )
        print(f"   promotion eligible: {interpreted.get('promotionEligible')}")
        print(f"   promotion blockers: {interpreted.get('promotionBlockers') or []}")
        print(f"   claim: {interpreted['claimDecision']['claim']}")

        return StageOutcome(
            result=interpreted,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.3.1"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"broadIndependentProjectResult": sha256_json(run)},
        )

    def morphology_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        harmonic = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.harmonic-family.interpret",
        )
        broad = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.interpret",
        )
        family = ((harmonic or broad or {}).get("harmonicFamily") or {})
        raw_period = family.get("representativeRawPeriodDays")
        double_period = family.get("possibleDoubleCycleDays")
        if independent_prepare is None:
            raise RuntimeError(
                "Morphology analysis requires the frozen independent-sector preparation."
            )
        if raw_period is None or double_period is None:
            raise RuntimeError(
                "Morphology analysis requires a recurrent raw period and possible doubled cycle."
            )

        print("🧬 Analyzing folded light-curve morphology across frozen TESS sectors")
        print(f"   recurrent raw family: {raw_period} days")
        print(f"   possible 2x physical cycle: {double_period} days")
        print("   no MAST download; no distributed compute")

        morphology = analyze_morphology(
            primary_dataset_path=prepared["datasetPath"],
            independent_spec=independent_prepare,
            raw_period_days=float(raw_period),
            possible_double_cycle_days=float(double_period),
        )

        for item in morphology.get("sectorResults") or []:
            double_metrics = item.get("doubleWaveMetrics") or {}
            print(
                "   sector "
                f"{item.get('sector')}: "
                f"double gain={item.get('doubleExplainedVarianceImprovement'):.4f}, "
                f"half difference={double_metrics.get('halfCycleDifferenceRatio')}, "
                f"raw={item.get('supportsRawCycle')}, "
                f"double={item.get('supportsDoubleCycle')}"
            )
        print(f"   morphology class: {morphology.get('morphologyClass')}")
        print(f"   phenomenology: {morphology.get('phenomenology')}")
        print(f"   physical cycle resolved: {morphology.get('physicalCycleResolved')}")
        if morphology.get("resolvedPhysicalPeriodDays") is not None:
            print(
                "   resolved physical period: "
                f"{morphology.get('resolvedPhysicalPeriodDays')} days"
            )

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "morphology"
            / "morphology-v20.4.json"
        )
        _write_json(artifact_path, morphology)

        input_hashes = {
            "periodFamily": sha256_json(family),
            "primaryDataset": sha256_file(Path(prepared["datasetPath"])),
        }
        for item in independent_prepare.get("preparedSectors") or []:
            sector = item.get("sector")
            path = item.get("datasetPath")
            if path:
                input_hashes[f"independentSector{sector}"] = sha256_file(Path(path))

        return StageOutcome(
            result=morphology,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.4"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes=input_hashes,
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def period_semantics_stage(investigation, request):
        broad = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.interpret",
        )
        harmonic = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.harmonic-family.interpret",
        )
        if harmonic is None and broad is None:
            raise RuntimeError(
                "Period-semantics reinterpretation requires completed broad/harmonic evidence."
            )
        family = ((harmonic or broad or {}).get("harmonicFamily") or {})
        print("🧾 Rewriting period semantics without changing the evidence")
        print(
            "   recurrent photometric periodicity: "
            f"{family.get('representativeRawPeriodDays')} days"
        )
        print(
            "   possible physical/full cycle: "
            f"{family.get('possibleDoubleCycleDays')} days"
        )
        print("   physical period: unresolved")
        return StageOutcome(
            result={
                "semanticModel": "period-evidence-v1",
                "evidenceChanged": False,
                "physicalCycleResolved": False,
            },
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.3.3"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={"harmonicEvidence": sha256_json(harmonic or broad or {})},
        )

    def finalize_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        primary_analysis = _result(investigation, "004-hypotheses")
        planner = _result(investigation, "005-planner")
        followup_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.followup.interpret",
        )
        independent_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.interpret",
        )
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        harmonic_family_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.harmonic-family.interpret",
        )
        broad_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.interpret",
        )
        broad_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.prepare",
        )
        morphology_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )

        if harmonic_family_interpretation is not None:
            claim_decision = harmonic_family_interpretation["claimDecision"]
            selected_period = harmonic_family_interpretation.get("selectedPeriodDays")
            selected_source = harmonic_family_interpretation.get("selectedSource")
        elif broad_interpretation is not None:
            claim_decision = broad_interpretation["claimDecision"]
            selected_period = broad_interpretation.get("selectedPeriodDays")
            selected_source = broad_interpretation.get("selectedSource")
        elif independent_interpretation is not None:
            claim_decision = independent_interpretation["claimDecision"]
            selected_period = independent_interpretation.get("selectedPeriodDays")
            selected_source = independent_interpretation.get("selectedSource")
        elif followup_interpretation is not None:
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

        if morphology_interpretation is not None and morphology_interpretation.get(
            "physicalCycleResolved"
        ):
            resolved_period = morphology_interpretation.get("resolvedPhysicalPeriodDays")
            morphology_class = morphology_interpretation.get("morphologyClass")
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": [
                    (
                        "Multi-sector folded-light-curve morphology resolves the harmonic "
                        f"interpretation as {morphology_class} at approximately "
                        f"{resolved_period} days."
                    ),
                    (
                        "The claim level is not automatically upgraded by morphology alone; "
                        "independent recurrence promotion remains governed by the existing "
                        "sector-count, cluster-width, prominence, boundary, and coverage rules."
                    ),
                ],
            }

        period_evidence = _build_period_evidence(
            claim_decision=claim_decision,
            selected_period=selected_period,
            selected_source=selected_source,
            primary_analysis=primary_analysis,
            followup_interpretation=followup_interpretation,
            independent_interpretation=independent_interpretation,
            broad_interpretation=broad_interpretation,
            harmonic_family_interpretation=harmonic_family_interpretation,
        )
        if morphology_interpretation is not None:
            period_evidence["morphologyClass"] = morphology_interpretation.get("morphologyClass")
            period_evidence["phenomenology"] = morphology_interpretation.get("phenomenology")
            if morphology_interpretation.get("physicalCycleResolved"):
                period_evidence["physicalCycleResolved"] = True
                period_evidence["physicalPeriodDays"] = morphology_interpretation.get(
                    "resolvedPhysicalPeriodDays"
                )
                period_evidence["interpretation"] = "morphology-resolved-physical-cycle"
                period_evidence["candidateSource"] = "multi-sector-morphology-discrimination"

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
            "periodEvidence": period_evidence,
            "selectedPeriodDays": period_evidence.get("physicalPeriodDays"),
            "selectedSource": (
                period_evidence.get("candidateSource")
                if period_evidence.get("physicalCycleResolved")
                else None
            ),
            "primaryAnalysis": primary_analysis,
            "planner": planner,
            "followup": followup_interpretation,
            "independentPreparation": independent_prepare,
            "independentVerification": independent_interpretation,
            "independentBroadPreparation": broad_prepare,
            "independentBroadVerification": broad_interpretation,
            "independentHarmonicFamilyVerification": harmonic_family_interpretation,
            "morphology": morphology_interpretation,
            "automaticDiscoveryClaim": False,
        }

        output_dir = store.directory_for(investigation.id)
        suffix = str(request.parameters.get("outputSuffix") or "").strip()
        if suffix:
            conclusion_path = output_dir / f"conclusion-{suffix}.json"
            report_path = output_dir / f"report-{suffix}.md"
        else:
            conclusion_path = output_dir / "conclusion.json"
            report_path = output_dir / "report.md"
        conclusion["conclusionPath"] = str(conclusion_path)
        conclusion["reportPath"] = str(report_path)
        _write_json(conclusion_path, conclusion)
        report_path.write_text(_render_report(conclusion), encoding="utf-8")

        final_status = (
            "HUMAN_REVIEW_REQUIRED"
            if claim_decision["claim"] == "HUMAN_REVIEW_REQUIRED"
            else "COMPLETE"
        )
        print("🏁 TESS investigation conclusion")
        print(f"   claim: {claim_decision['claim']}")
        if period_evidence.get("recurrentPhotometricPeriodDays") is not None:
            print(
                "   recurrent photometric periodicity: "
                f"{period_evidence.get('recurrentPhotometricPeriodDays')} days"
            )
        if period_evidence.get("possiblePhysicalCycleDays") is not None:
            print(
                "   possible physical/full cycle: "
                f"{period_evidence.get('possiblePhysicalCycleDays')} days"
            )
        if period_evidence.get("physicalCycleResolved"):
            print(f"   physical period: {period_evidence.get('physicalPeriodDays')} days")
        else:
            print("   physical period: unresolved")
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
    engine.register_handler("openstar.tess.independent.prepare", prepare_independent)
    engine.register_handler("openstar.tess.independent.run", run_independent)
    engine.register_handler("openstar.tess.independent.interpret", interpret_independent)
    engine.register_handler("openstar.tess.independent.broad.prepare", prepare_broad_independent)
    engine.register_handler("openstar.tess.independent.broad.run", run_broad_independent)
    engine.register_handler("openstar.tess.independent.broad.interpret", interpret_broad_independent)
    engine.register_handler(
        "openstar.tess.independent.harmonic-family.interpret",
        reinterpret_harmonic_family,
    )
    engine.register_handler(
        "openstar.tess.morphology.analyze",
        morphology_stage,
    )
    engine.register_handler(
        "openstar.tess.period-semantics.reinterpret",
        period_semantics_stage,
    )
    engine.register_handler("openstar.tess.finalize", finalize_stage)
    return engine
