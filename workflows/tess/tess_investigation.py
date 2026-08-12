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
from .tess_physical import analyze_physical_interpretation
from .tess_localization import localize_periodic_source
from .tess_multimode import (
    MAX_RESIDUAL_ITERATIONS,
    build_residual_search_project,
    interpret_residual_iteration,
    summarize_multimode_decomposition,
)
from .tess_time_frequency import (
    build_time_frequency_project,
    interpret_time_frequency_project,
    summarize_time_frequency_evolution,
)
from .tess_nonstationary import (
    build_nonstationary_project,
    interpret_nonstationary_project,
    summarize_nonstationary_modeling,
)
from .tess_residual_localization import (
    build_residual_mode_pixel_project,
    interpret_residual_mode_pixel_project,
)
from .tess_residual_localization_review import (
    build_residual_mode_localization_review_project,
    interpret_residual_mode_localization_review_project,
)
from .tess_multisource_residual import (
    build_multisource_residual_project,
    interpret_multisource_residual_project,
)
from .tess_offset_source import identify_offset_residual_source
from .tess_offset_variability import (
    build_offset_source_variability_project,
    interpret_offset_source_variability_project,
)
from .tess_prf_deblend import (
    build_calibrated_prf_deblending_project,
    interpret_calibrated_prf_deblending_project,
)
from .tess_difference_image import (
    build_difference_image_project,
    interpret_difference_image_project,
)
from .tess_frequency_localized_pixel import (
    build_frequency_localized_pixel_project,
    interpret_frequency_localized_pixel_project,
)
from .tess_spoc_prf import (
    build_official_spoc_prf_project,
    interpret_official_spoc_prf_project,
)
from .tess_external_highres import (
    build_external_high_resolution_project,
    interpret_external_high_resolution_project,
)
from .tess_multisector import (
    build_broad_independent_sector_project,
    build_independent_sector_project,
)


WORKFLOW_ID = "openstar.workflow.tess-investigation.v1"
WORKFLOW_VERSION = "20.2"
SOFTWARE_ID = "openstar.tess-investigation-plugin"
SOFTWARE_VERSION = "20.19"


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


def _all_results_for_handler(
    investigation: Investigation,
    handler_id: str,
) -> list[dict[str, Any]]:
    return [
        stage.result
        for stage in investigation.stages
        if stage.handler_id == handler_id
        and stage.status == "COMPLETE"
        and stage.result is not None
    ]


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

    physical = conclusion.get("physicalInterpretation")
    if physical is not None:
        fourier = physical.get("crossSectorFourierSummary") or {}
        rotation = physical.get("rotationConstraint") or {}
        contamination = physical.get("contaminationScreen") or {}
        lines.extend([
            "",
            "## Physical-mechanism discrimination",
            "",
            f"- Physical mechanism resolved: {physical.get('physicalMechanismResolved')}",
            f"- Preferred photometric hypothesis: {physical.get('preferredPhotometricHypothesis')}",
            f"- Recommended next test: {physical.get('recommendedNextTest')}",
            f"- Independent Fourier sectors: {fourier.get('independentEligibleSectorCount')}",
            f"- Harmonic-dominant sectors: {fourier.get('independentHarmonicDominantSectors')}",
            f"- Relative harmonic phase concentration: {fourier.get('relativeHarmonicPhaseConcentration')}",
            f"- Harmonic amplitude variation fraction: {fourier.get('firstHarmonicAmplitudeVariationFraction')}",
            f"- Rotation status at resolved physical period: {rotation.get('status')}",
            f"- Equatorial speed: {rotation.get('equatorialSpeedKmS')} km/s",
            f"- Minimum mass for subcritical rotation: {rotation.get('minimumMassForSubcriticalRotationMsun')} M_sun",
            f"- TIC contamination ratio: {contamination.get('ticContaminationRatio')}",
            f"- Existing metadata can exclude TESS aperture contamination: {contamination.get('canExcludeTessApertureContamination')}",
            "",
            "### Mechanism ranking",
            "",
        ])
        for item in physical.get("mechanismRankings") or []:
            lines.append(
                f"- {item.get('hypothesis')}: score={item.get('score')}, "
                f"level={item.get('evidenceLevel')}, reasons={item.get('reasons')}, "
                f"cautions={item.get('cautions')}"
            )
        lines.extend([
            "",
            "### Per-sector Fourier model",
            "",
        ])
        for item in physical.get("sectorResults") or []:
            fit = item.get("fourierFit") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: R2={fit.get('explainedVariance')}, "
                f"A1={fit.get('fundamentalAmplitude')}, "
                f"A2={fit.get('firstHarmonicAmplitude')}, "
                f"A2/A1={fit.get('firstHarmonicToFundamentalAmplitudeRatio')}, "
                f"component={fit.get('dominantFourierComponent')}, "
                f"relativePhase={fit.get('translationInvariantRelativeHarmonicPhaseRad')}"
            )

    localization = conclusion.get("sourceLocalization")
    if localization is not None:
        cross = localization.get("crossSector") or {}
        lines.extend([
            "",
            "## Pixel-level source localization",
            "",
            f"- Classification: {cross.get('classification')}",
            f"- Variable signal origin: {cross.get('variableSignalOrigin')}",
            f"- Independent eligible sectors: {cross.get('independentEligibleSectorCount')}",
            f"- Required independent support: {cross.get('requiredIndependentSupportCount')}",
            f"- Target-supporting sectors: {cross.get('targetSupportingSectors')}",
            f"- Off-target sectors: {cross.get('offTargetSectors')}",
            f"- Ambiguous sectors: {cross.get('ambiguousSectors')}",
            f"- Median sky separation: {cross.get('medianSkySeparationArcsec')} arcsec",
            f"- Off-target sky-offset scatter: {cross.get('offTargetSkyOffsetScatterArcsec')} arcsec",
            f"- Recommended next test: {localization.get('recommendedNextTest')}",
            "- Static aperture contamination cleared: False",
            "",
            "### Per-sector localization",
            "",
        ])
        for item in localization.get("sectorResults") or []:
            source = item.get("source") or {}
            lines.append(
                "- Sector "
                f"{item.get('sector')}: role={item.get('role')}, "
                f"source={source.get('sourceType')}, "
                f"offsetPixels={item.get('offsetPixels')}, "
                f"skySeparationArcsec={item.get('skySeparationArcsec')}, "
                f"classification={item.get('classification')}"
            )

    multimode = conclusion.get("multiModeDecomposition")
    if multimode is not None:
        recurrent = multimode.get("bestRecurrentSecondaryMode") or {}
        lines.extend([
            "",
            "## Residual multi-mode frequency decomposition",
            "",
            f"- Classification: {multimode.get('classification')}",
            f"- Iterations completed: {multimode.get('iterationsCompleted')}",
            f"- Independent sectors with residual modes: {multimode.get('independentSectorsWithAcceptedResidualModes')}",
            f"- Resolved-near-primary sectors: {multimode.get('resolvedNearPrimaryFamilySectors')}",
            f"- Recurrent secondary period: {recurrent.get('medianPeriodDays')} days",
            f"- Recurrent secondary supporting sectors: {recurrent.get('independentSectors')}",
            f"- Physical mechanism resolved: {multimode.get('physicalMechanismResolved')}",
            f"- Recommended next test: {multimode.get('recommendedNextTest')}",
        ])

    time_frequency = conclusion.get("timeFrequencyEvolution")
    if time_frequency is not None:
        residual = time_frequency.get("residualEvolution") or {}
        family = time_frequency.get("familyEvolution") or {}
        best_cluster = residual.get("bestCluster") or {}
        lines.extend([
            "",
            "## Time-frequency evolution",
            "",
            f"- Overall classification: {time_frequency.get('classification')}",
            f"- Sliding windows: {time_frequency.get('windowCount')}",
            f"- Accepted residual-window features: {time_frequency.get('acceptedFeatureCount')}",
            f"- Independent accepted features: {time_frequency.get('acceptedIndependentFeatureCount')}",
            f"- Independent sectors with accepted features: {time_frequency.get('acceptedIndependentSectors')}",
            f"- Residual evolution: {residual.get('classification')}",
            f"- Established-family evolution: {family.get('classification')}",
            f"- Best residual cluster period: {best_cluster.get('medianPeriodDays')} days",
            f"- Best residual cluster independent sectors: {best_cluster.get('independentSectors')}",
            f"- Physical mechanism resolved: {time_frequency.get('physicalMechanismResolved')}",
            f"- Recommended next test: {time_frequency.get('recommendedNextTest')}",
            "",
            "### Sliding-window results",
            "",
        ])
        for item in time_frequency.get("windowResults") or []:
            lines.append(
                "- Sector "
                f"{item.get('sector')} window {item.get('windowIndex')}: "
                f"period={item.get('candidatePeriodDays')} d, "
                f"prominence={item.get('candidatePeakProminenceRatio')}, "
                f"accepted={item.get('acceptedTimeFrequencyFeature')}, "
                f"nearFamily={item.get('nearEstablishedFamily')}"
            )

    nonstationary = conclusion.get("nonstationaryModeling")
    if nonstationary is not None:
        preferred = nonstationary.get("preferredModel") or {}
        comparison = nonstationary.get("modelComparison") or {}
        lines.extend([
            "",
            "## Long-baseline nonstationary mode modeling",
            "",
            f"- Classification: {nonstationary.get('classification')}",
            f"- Generic workload: {(nonstationary.get('distributedModeling') or {}).get('workloadID')}",
            f"- Distributed work units: {(nonstationary.get('distributedModeling') or {}).get('totalWorkUnits')}",
            f"- Preferred model: {comparison.get('bestModelID')}",
            f"- BIC improvement over null: {comparison.get('bicImprovementOverNull')}",
            f"- Preferred period at reference time: {nonstationary.get('preferredPeriodAtReferenceDays')} days",
            f"- Fractional frequency drift/day: {nonstationary.get('fractionalFrequencyDriftPerDay')}",
            f"- Frequency derivative: {nonstationary.get('frequencyDerivativeCyclesPerDaySquared')} cycles/day²",
            f"- Signal sectors: {preferred.get('signalSectors')}",
            f"- Physical mechanism resolved: {nonstationary.get('physicalMechanismResolved')}",
            f"- Recommended next test: {nonstationary.get('recommendedNextTest')}",
            "",
            "### Competing models",
            "",
        ])
        for model in comparison.get("models") or []:
            lines.append(
                f"- {model.get('modelID')}: BIC={model.get('bic')}, "
                f"period={model.get('periodDays')} d, q={model.get('fractionalFrequencyDriftPerDay')}, "
                f"sectors={model.get('signalSectors')}"
            )

    residual_localization = conclusion.get("residualModeLocalization")
    if residual_localization is not None:
        cross = residual_localization.get("crossSector") or {}
        distributed = residual_localization.get("distributedLocalization") or {}
        lines.extend([
            "",
            "## Drifting residual-mode pixel localization",
            "",
            f"- Classification: {cross.get('classification')}",
            f"- Residual-mode origin: {cross.get('residualModeOrigin')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Residual period at reference: {residual_localization.get('residualPeriodAtReferenceDays')} days",
            f"- Fractional frequency drift/day: {residual_localization.get('fractionalFrequencyDriftPerDay')}",
            f"- Independent eligible sectors: {cross.get('independentEligibleSectorCount')}",
            f"- Target-supporting sectors: {cross.get('targetSupportingSectors')}",
            f"- Off-target sectors: {cross.get('offTargetSectors')}",
            f"- Ambiguous sectors: {cross.get('ambiguousSectors')}",
            f"- Median sky separation: {cross.get('medianSkySeparationArcsec')} arcsec",
            f"- Physical mechanism resolved: {residual_localization.get('physicalMechanismResolved')}",
            f"- Recommended next test: {residual_localization.get('recommendedNextTest')}",
            "",
            "### Sector localization",
            "",
        ])
        for item in residual_localization.get("sectorResults") or []:
            lines.append(
                f"- Sector {item.get('sector')}: role={item.get('role')}, "
                f"offsetPixels={item.get('offsetPixels')}, "
                f"skySeparationArcsec={item.get('skySeparationArcsec')}, "
                f"peakPower={item.get('peakPower')}, "
                f"powerContrast={item.get('powerContrast')}, "
                f"classification={item.get('classification')}"
            )


    residual_review = conclusion.get("residualModeLocalizationReview")
    if residual_review is not None:
        cross_time = residual_review.get("crossTime") or {}
        distributed = residual_review.get("distributedLocalizationReview") or {}
        lines.extend([
            "",
            "## Time-resolved residual-mode source localization review",
            "",
            f"- Classification: {cross_time.get('classification')}",
            f"- Residual-mode origin: {cross_time.get('residualModeOrigin')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target-dominant sectors: {cross_time.get('targetDominantSectors')}",
            f"- Off-target-dominant sectors: {cross_time.get('offTargetDominantSectors')}",
            f"- Source-switching sectors: {cross_time.get('sourceSwitchingSectors')}",
            f"- Off-target sky scatter: {cross_time.get('offTargetSkyOffsetScatterArcsec')} arcsec",
            f"- Physical mechanism resolved: {residual_review.get('physicalMechanismResolved')}",
            f"- Recommended next test: {residual_review.get('recommendedNextTest')}",
            "",
            "### Time-window localization",
            "",
        ])
        for item in residual_review.get("windowResults") or []:
            lines.append(
                f"- Sector {item.get('sector')} window {item.get('windowIndex')}: "
                f"offsetPixels={item.get('offsetPixels')}, "
                f"skySeparationArcsec={item.get('skySeparationArcsec')}, "
                f"peakPower={item.get('peakPower')}, "
                f"powerContrast={item.get('powerContrast')}, "
                f"classification={item.get('classification')}"
            )

    multisource = conclusion.get("multiSourceResidualDecomposition")
    if multisource is not None:
        distributed = multisource.get("distributedDecomposition") or {}
        lines.extend([
            "",
            "## Multi-source residual decomposition",
            "",
            f"- Classification: {multisource.get('classification')}",
            f"- Residual-mode origin: {multisource.get('residualModeOrigin')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Target component: {multisource.get('targetComponentID')}",
            f"- Best offset component: {multisource.get('bestOffsetComponentID')}",
            f"- Physical mechanism resolved: {multisource.get('physicalMechanismResolved')}",
            f"- Recommended next test: {multisource.get('recommendedNextTest')}",
            "",
            "### Spatial component evidence",
            "",
        ])
        for item in multisource.get("componentSummaries") or []:
            lines.append(
                f"- {item.get('componentID')}: type={item.get('componentType')}, "
                f"independentSupport={item.get('independentSupportCount')}, "
                f"sectors={item.get('independentSupportingSectors')}, "
                f"combinedPower={item.get('combinedPower')}, "
                f"combinedPeriod={item.get('combinedPeriodDays')} d"
            )

    offset_source = conclusion.get("offsetResidualSourceIdentification")
    if offset_source is not None:
        component = offset_source.get("component") or {}
        sky = component.get("componentSky") or {}
        best = offset_source.get("bestCandidate") or {}
        ids = best.get("catalogIDs") or {}
        vsx = best.get("vsxMatch") or {}
        simbad = best.get("simbadMatch") or {}
        gaia_var = best.get("gaiaVariability") or {}
        lines.extend([
            "",
            "## Offset residual source identification",
            "",
            f"- Classification: {offset_source.get('classification')}",
            f"- Offset component: {component.get('componentID')}",
            f"- Component sky position: RA={sky.get('raDeg')} deg, Dec={sky.get('decDeg')} deg",
            f"- Separation from TIC target: {component.get('targetSeparationArcsec')} arcsec",
            f"- Secure association radius: {(offset_source.get('search') or {}).get('secureAssociationRadiusArcsec')} arcsec",
            f"- Best candidate separation: {best.get('separationArcsec')} arcsec",
            f"- TIC counterpart: {ids.get('ticID')}",
            f"- Gaia DR3 counterpart: {ids.get('gaiaDR3SourceID')}",
            f"- VSX match: {vsx.get('name')}",
            f"- VSX type: {vsx.get('type')}",
            f"- VSX period: {vsx.get('periodDays')} days",
            f"- SIMBAD match: {simbad.get('mainID')}",
            f"- SIMBAD object type: {simbad.get('objectType')}",
            f"- Gaia variability classification: {gaia_var.get('classification')}",
            f"- Known-variable catalog evidence: {offset_source.get('knownVariableCatalogEvidence')}",
            f"- Catalog query errors: {offset_source.get('queryErrors')}",
            f"- Recommended next test: {offset_source.get('recommendedNextTest')}",
            "- Guard: positional catalog association is not treated as direct proof of the residual variability source.",
        ])

    offset_variability = conclusion.get("offsetSourceVariabilityValidation")
    if offset_variability is not None:
        counterpart = offset_variability.get("catalogCounterpart") or {}
        target_control = offset_variability.get("targetControl") or {}
        candidate_evidence = offset_variability.get("catalogCounterpartEvidence") or {}
        distributed = offset_variability.get("distributedValidation") or {}
        lines.extend([
            "",
            "## Offset catalog-counterpart variability validation",
            "",
            f"- Classification: {offset_variability.get('classification')}",
            f"- Residual-mode origin: {offset_variability.get('residualModeOrigin')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Reference residual period: {offset_variability.get('referencePeriodDays')} days",
            f"- Counterpart independent support: {candidate_evidence.get('independentSupportCount')}",
            f"- Counterpart supporting sectors: {candidate_evidence.get('independentSupportingSectors')}",
            f"- Counterpart combined power: {candidate_evidence.get('combinedPower')}",
            f"- Counterpart combined period: {candidate_evidence.get('combinedPeriodDays')} days",
            f"- Target-control independent support: {target_control.get('independentSupportCount')}",
            f"- Target-control combined power: {target_control.get('combinedPower')}",
            f"- Physical mechanism resolved: {offset_variability.get('physicalMechanismResolved')}",
            f"- Recommended next test: {offset_variability.get('recommendedNextTest')}",
            "- Guard: this is catalog-guided Gaussian deblending, not a calibrated TESS PRF solution.",
        ])

    calibrated_prf = conclusion.get("calibratedPrfSourceDeblending")
    if calibrated_prf is not None:
        counterpart = calibrated_prf.get("catalogCounterpart") or {}
        target_control = calibrated_prf.get("targetControl") or {}
        candidate_evidence = calibrated_prf.get("catalogCounterpartEvidence") or {}
        distributed = calibrated_prf.get("distributedValidation") or {}
        lines.extend([
            "",
            "## Sector-calibrated pixel-response source deblending",
            "",
            f"- Classification: {calibrated_prf.get('classification')}",
            f"- Residual-mode origin: {calibrated_prf.get('residualModeOrigin')}",
            f"- Deblend backend: {calibrated_prf.get('deblendBackend')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Reference residual period: {calibrated_prf.get('referencePeriodDays')} days",
            f"- Counterpart independent support: {candidate_evidence.get('independentSupportCount')}",
            f"- Counterpart supporting sectors: {candidate_evidence.get('independentSupportingSectors')}",
            f"- Counterpart combined power: {candidate_evidence.get('combinedPower')}",
            f"- Counterpart combined period: {candidate_evidence.get('combinedPeriodDays')} days",
            f"- Target-control independent support: {target_control.get('independentSupportCount')}",
            f"- Target-control supporting sectors: {target_control.get('independentSupportingSectors')}",
            f"- Target-control combined power: {target_control.get('combinedPower')}",
            f"- Physical mechanism resolved: {calibrated_prf.get('physicalMechanismResolved')}",
            f"- Recommended next test: {calibrated_prf.get('recommendedNextTest')}",
            "",
            "### Sector ePRF calibration diagnostics",
            "",
        ])
        for item in calibrated_prf.get("calibrationDiagnostics") or []:
            lines.append(
                f"- Sector {item.get('sector')}: backend={item.get('backend')}, "
                f"R2={item.get('explainedVariance')}, sigmaX={item.get('sigmaX')}, "
                f"axisRatio={item.get('axisRatio')}, wingFraction={item.get('wingFraction')}, "
                f"shift=({item.get('dx')},{item.get('dy')}), "
                f"templateCorrelation={item.get('targetCounterpartTemplateCorrelation')}, "
                f"condition={item.get('designConditionNumber')}"
            )
        lines.append(
            "- Guard: this is a sector-calibrated empirical TESS pixel-response model; it is not claimed to be identical to the SPOC engineering PRF calibration."
        )

    difference_image = conclusion.get("differenceImageSourceLocalization")
    if difference_image is not None:
        counterpart = difference_image.get("catalogCounterpart") or {}
        distributed = difference_image.get("distributedFrequencyRefinement") or {}
        lines.extend([
            "",
            "## Difference-image residual source localization",
            "",
            f"- Classification: {difference_image.get('classification')}",
            f"- Residual-mode origin: {difference_image.get('residualModeOrigin')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Reference residual period: {difference_image.get('referencePeriodDays')} days",
            f"- Quality sectors: {difference_image.get('qualitySectorCount')}",
            f"- Counterpart-supporting sectors: {difference_image.get('counterpartSupportingSectors')}",
            f"- Target-supporting sectors: {difference_image.get('targetSupportingSectors')}",
            f"- Ambiguous sectors: {difference_image.get('ambiguousSectors')}",
            f"- Median counterpart separation: {difference_image.get('medianCounterpartSeparationArcsec')} arcsec",
            f"- Median target separation: {difference_image.get('medianTargetSeparationArcsec')} arcsec",
            f"- Recommended next test: {difference_image.get('recommendedNextTest')}",
            "",
            "### Sector difference-image evidence",
            "",
        ])
        for item in difference_image.get("sectorResults") or []:
            image = item.get("differenceImage") or {}
            frequency = item.get("frequencyResult") or {}
            lines.append(
                f"- Sector {item.get('sector')}: classification={item.get('classification')}, "
                f"frequency={frequency.get('candidateFrequency')} c/d, "
                f"period={frequency.get('candidatePeriodDays')} d, "
                f"peakSNR={image.get('peakSNR')}, "
                f"targetDistance={item.get('targetDistancePixels')} px, "
                f"counterpartDistance={item.get('counterpartDistancePixels')} px, "
                f"centroidUncertainty={item.get('centroidUncertaintyPixels')} px"
            )
        lines.append(
            "- Guard: difference-image attribution requires recurring independent-sector localization and does not alter the v20.6 target association of the established main periodic family."
        )

    frequency_localized = conclusion.get("frequencyLocalizedPixelResponse")
    if frequency_localized is not None:
        counterpart = frequency_localized.get("catalogCounterpart") or {}
        distributed = frequency_localized.get("distributedPixelSearch") or {}
        lines.extend([
            "",
            "## Frequency-localized pixel-response confirmation",
            "",
            f"- Classification: {frequency_localized.get('classification')}",
            f"- Residual-mode origin: {frequency_localized.get('residualModeOrigin')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Pixel datasets: {distributed.get('datasetCount')}",
            f"- Reference residual period: {frequency_localized.get('referencePeriodDays')} days",
            f"- Quality sectors: {frequency_localized.get('qualitySectorCount')}",
            f"- Counterpart-supporting sectors: {frequency_localized.get('counterpartSupportingSectors')}",
            f"- Target-supporting sectors: {frequency_localized.get('targetSupportingSectors')}",
            f"- Ambiguous sectors: {frequency_localized.get('ambiguousSectors')}",
            f"- Recommended next test: {frequency_localized.get('recommendedNextTest')}",
            "",
            "### Sector frequency-localized evidence",
            "",
        ])
        for item in frequency_localized.get("sectorResults") or []:
            response = item.get("response") or {}
            lines.append(
                f"- Sector {item.get('sector')}: classification={item.get('classification')}, "
                f"targetFrequency={item.get('targetFrequency')} c/d, "
                f"source={item.get('frequencySource')}, peakPower={response.get('peakPower')}, "
                f"powerContrast={response.get('powerContrast')}, phaseConcentration={response.get('phaseConcentration')}, "
                f"clusterWeightFraction={response.get('clusterWeightFraction')}, "
                f"targetDistance={item.get('targetDistancePixels')} px, "
                f"counterpartDistance={item.get('counterpartDistancePixels')} px, "
                f"centroidUncertainty={item.get('centroidUncertaintyPixels')} px"
            )
        lines.append(
            "- Guard: the narrow-band per-pixel searches are conditioned on the existing residual family and are used only for source localization; they are not independent period discoveries."
        )

    official_spoc_prf = conclusion.get("officialSpocPrfForwardModeling")
    if official_spoc_prf is not None:
        counterpart = official_spoc_prf.get("catalogCounterpart") or {}
        distributed = official_spoc_prf.get("distributedValidation") or {}
        candidate_evidence = official_spoc_prf.get("catalogCounterpartEvidence") or {}
        target_control = official_spoc_prf.get("targetControl") or {}
        lines.extend([
            "",
            "## Official SPOC PRF forward modeling",
            "",
            f"- Classification: {official_spoc_prf.get('classification')}",
            f"- Residual-mode origin: {official_spoc_prf.get('residualModeOrigin')}",
            f"- Backend: {official_spoc_prf.get('deblendBackend')}",
            f"- TIC counterpart: {counterpart.get('ticID')}",
            f"- Gaia DR3 counterpart: {counterpart.get('gaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- Counterpart independent support: {candidate_evidence.get('independentSupportCount')}",
            f"- Counterpart supporting sectors: {candidate_evidence.get('independentSupportingSectors')}",
            f"- Counterpart combined power: {candidate_evidence.get('combinedPower')}",
            f"- Counterpart combined period: {candidate_evidence.get('combinedPeriodDays')} days",
            f"- Target-control independent support: {target_control.get('independentSupportCount')}",
            f"- Target-control supporting sectors: {target_control.get('independentSupportingSectors')}",
            f"- Target-control combined power: {target_control.get('combinedPower')}",
            f"- Recommended next test: {official_spoc_prf.get('recommendedNextTest')}",
            "",
            "### Sector official-PRF diagnostics",
            "",
        ])
        for item in official_spoc_prf.get("calibrationDiagnostics") or []:
            lines.append(
                f"- Sector {item.get('sector')}: start_s{item.get('officialPRFStartSector')}, "
                f"camera={item.get('camera')}, ccd={item.get('ccd')}, R2={item.get('explainedVariance')}, "
                f"shift=({item.get('dx')},{item.get('dy')}), "
                f"templateCorrelation={item.get('targetCounterpartTemplateCorrelation')}, "
                f"condition={item.get('designConditionNumber')}"
            )
        lines.append(
            "- Guard: official SPOC PRF source attribution still requires recurring independent-sector residual-frequency evidence; an unresolved result is retained as a TESS spatial-attribution limit."
        )

    external_high_resolution = conclusion.get("externalHighResolutionVariabilityValidation")
    if external_high_resolution is not None:
        distributed = external_high_resolution.get("distributedValidation") or {}
        pair = external_high_resolution.get("sourcePair") or {}
        target = external_high_resolution.get("targetControl") or {}
        counterpart = external_high_resolution.get("catalogCounterpartEvidence") or {}
        lines.extend([
            "",
            "## External high-resolution variability validation",
            "",
            f"- Classification: {external_high_resolution.get('classification')}",
            f"- Residual-mode origin: {external_high_resolution.get('residualModeOrigin')}",
            f"- Archive: {external_high_resolution.get('archive')}",
            f"- Target Gaia DR3 source: {pair.get('targetGaiaDR3SourceID')}",
            f"- Counterpart Gaia DR3 source: {pair.get('counterpartGaiaDR3SourceID')}",
            f"- Generic workload: {distributed.get('workloadID')}",
            f"- Distributed work units: {distributed.get('totalWorkUnits')}",
            f"- TESS drift extrapolated into Gaia epoch: {external_high_resolution.get('driftExtrapolatedToGaiaEpoch')}",
            f"- Target residual-band accepted: {(target or {}).get('acceptedResidualBandVariability')}",
            f"- Target period: {(target or {}).get('candidatePeriodDays')} days",
            f"- Target power: {(target or {}).get('candidatePower')}",
            f"- Counterpart residual-band accepted: {(counterpart or {}).get('acceptedResidualBandVariability')}",
            f"- Counterpart period: {(counterpart or {}).get('candidatePeriodDays')} days",
            f"- Counterpart power: {(counterpart or {}).get('candidatePower')}",
            f"- Recommended next test: {external_high_resolution.get('recommendedNextTest')}",
            f"- Guard: {external_high_resolution.get('interpretationGuard')}",
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

    def physical_interpretation_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        identity = _latest_result_for_handler(
            investigation,
            "openstar.tess.catalog-identity",
        )
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        broad = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.broad.interpret",
        )
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        if identity is None:
            raise RuntimeError("Physical interpretation requires the completed catalog-identity stage.")
        if independent_prepare is None:
            raise RuntimeError("Physical interpretation requires frozen independent-sector light curves.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("Physical interpretation requires a morphology-resolved physical cycle.")

        physical_period = morphology.get("resolvedPhysicalPeriodDays")
        print("🔬 Discriminating physical mechanisms from frozen multi-sector evidence")
        print(f"   resolved physical period: {physical_period} days")
        print("   fitting fundamental + first harmonic in every frozen sector")
        print("   existing identity metadata only; no MAST and no distributed compute")

        interpretation = analyze_physical_interpretation(
            primary_dataset_path=prepared["datasetPath"],
            independent_spec=independent_prepare,
            identity=identity,
            morphology=morphology,
            broad_interpretation=broad,
        )

        fourier = interpretation.get("crossSectorFourierSummary") or {}
        rotation = interpretation.get("rotationConstraint") or {}
        contamination = interpretation.get("contaminationScreen") or {}
        print(f"   independent Fourier sectors: {fourier.get('independentEligibleSectorCount')}")
        print(f"   harmonic-dominant sectors: {fourier.get('independentHarmonicDominantSectors')}")
        print(f"   relative harmonic phase concentration: {fourier.get('relativeHarmonicPhaseConcentration')}")
        print(f"   harmonic amplitude variation: {fourier.get('firstHarmonicAmplitudeVariationFraction')}")
        print(f"   rotation status: {rotation.get('status')}")
        print(f"   equatorial speed: {rotation.get('equatorialSpeedKmS')} km/s")
        print(
            "   minimum mass for subcritical rotation: "
            f"{rotation.get('minimumMassForSubcriticalRotationMsun')} M_sun"
        )
        print(f"   contamination flagged by existing metadata: {contamination.get('flaggedByExistingMetadata')}")
        print(f"   preferred photometric hypothesis: {interpretation.get('preferredPhotometricHypothesis')}")
        for item in interpretation.get("mechanismRankings") or []:
            print(
                "      "
                f"{item.get('hypothesis')}: score={item.get('score')} "
                f"({item.get('evidenceLevel')})"
            )
        print(f"   physical mechanism resolved: {interpretation.get('physicalMechanismResolved')}")
        print(f"   recommended next test: {interpretation.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "physical"
            / "physical-interpretation-v20.5.json"
        )
        _write_json(artifact_path, interpretation)

        input_hashes = {
            "identity": sha256_json(identity),
            "morphology": sha256_json(morphology),
            "primaryDataset": sha256_file(Path(prepared["datasetPath"])),
        }
        if broad is not None:
            input_hashes["broadIndependentInterpretation"] = sha256_json(broad)
        for item in independent_prepare.get("preparedSectors") or []:
            sector = item.get("sector")
            path = item.get("datasetPath")
            if path:
                input_hashes[f"independentSector{sector}"] = sha256_file(Path(path))

        return StageOutcome(
            result=interpretation,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.5"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes=input_hashes,
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def source_localization_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        identity = _latest_result_for_handler(
            investigation,
            "openstar.tess.catalog-identity",
        )
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        physical = _latest_result_for_handler(
            investigation,
            "openstar.tess.physical.interpret",
        )
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        if identity is None:
            raise RuntimeError("Source localization requires the completed catalog-identity stage.")
        if independent_prepare is None:
            raise RuntimeError("Source localization requires frozen independent-sector metadata.")
        if physical is None:
            raise RuntimeError("Source localization requires completed v20.5 physical interpretation.")
        if physical.get("recommendedNextTest") != "PIXEL_LEVEL_SOURCE_LOCALIZATION":
            raise RuntimeError("v20.5 did not recommend pixel-level source localization.")
        physical_period = (physical.get("physicalPeriodDays") or
                           (morphology or {}).get("resolvedPhysicalPeriodDays"))
        if physical_period is None:
            raise RuntimeError("Source localization requires a resolved physical period.")

        print("🎯 Localizing the periodic signal on TESS pixels")
        print(f"   TIC: {prepared.get('ticID')}")
        print(f"   physical period: {physical_period} days")
        print(f"   first-harmonic photometric period: {float(physical_period) / 2.0} days")
        print("   MAST target-pixel products preferred; TESScut fallback when needed")
        print("   no distributed compute")

        artifact_root = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "source-localization"
        )
        localization = localize_periodic_source(
            tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=float(physical_period),
            artifact_root=artifact_root,
        )
        cross = localization.get("crossSector") or {}
        print("🎯 Cross-sector source localization")
        print(f"   classification: {cross.get('classification')}")
        print(f"   variable signal origin: {cross.get('variableSignalOrigin')}")
        print(f"   independent eligible sectors: {cross.get('independentEligibleSectorCount')}")
        print(f"   target-supporting sectors: {cross.get('targetSupportingSectors')}")
        print(f"   off-target sectors: {cross.get('offTargetSectors')}")
        print(f"   ambiguous sectors: {cross.get('ambiguousSectors')}")
        print(f"   median sky separation: {cross.get('medianSkySeparationArcsec')} arcsec")
        print(f"   recommended next test: {localization.get('recommendedNextTest')}")

        main_path = artifact_root / "pixel-localization-v20.6.json"
        artifacts = [_artifact(main_path, "application/json")]
        for item in localization.get("sectorResults") or []:
            sector = item.get("sector")
            path = artifact_root / f"sector-{sector}-localization.json"
            if path.exists():
                artifacts.append(_artifact(path, "application/json"))

        return StageOutcome(
            result=localization,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.6"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "physicalInterpretation": sha256_json(physical),
                "independentPreparation": sha256_json(independent_prepare),
            },
            artifacts=tuple(artifacts),
        )

    def multimode_prepare_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.source-localization.analyze",
        )
        if independent_prepare is None:
            raise RuntimeError("v20.7 requires the frozen independent-sector preparation.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.7 requires a morphology-resolved physical period.")
        if localization is None or (localization.get("crossSector") or {}).get("classification") != "TARGET_SOURCE_SUPPORTED":
            raise RuntimeError("v20.7 requires TARGET_SOURCE_SUPPORTED source localization.")

        iteration = int(request.parameters.get("iteration") or 1)
        prior_iterations = _all_results_for_handler(
            investigation,
            "openstar.tess.multimode.interpret",
        )
        physical_period = float(morphology["resolvedPhysicalPeriodDays"])
        artifact_root = store.directory_for(investigation.id) / "artifacts"

        print(f"🧹 Preparing residual multi-mode search iteration {iteration}/{MAX_RESIDUAL_ITERATIONS}")
        print(f"   resolved physical period: {physical_period} days")
        print("   subtracting physical fundamental + first harmonic")
        if prior_iterations:
            print("   also subtracting accepted residual modes from prior iterations")
        spec = build_residual_search_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            primary_dataset_path=prepared["datasetPath"],
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=physical_period,
            prior_iterations=prior_iterations,
            iteration=iteration,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        search = spec.get("frequencySearch") or {}
        print(
            "   residual search range: "
            f"{search.get('minimumFrequency'):.4f} - {search.get('maximumFrequency'):.4f} cycles/day"
        )
        print(f"   datasets: {len(spec.get('preparedDatasets') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")
        if any(item.get("role") == "combined-residual-multimode" for item in spec.get("preparedDatasets") or []):
            print("   combined long-baseline residual dataset: yes")
        else:
            print("   combined long-baseline residual dataset: unavailable (missing absolute time origins)")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedDatasets") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, f"run-multimode-iteration-{iteration}"),
                handler_id="openstar.tess.multimode.run",
                parameters={
                    "iteration": iteration,
                    "projectPath": spec["projectPath"],
                },
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "morphology": sha256_json(morphology),
                "sourceLocalization": sha256_json(localization),
                "independentPreparation": sha256_json(independent_prepare),
            },
            artifacts=tuple(artifacts),
        )

    def multimode_run_stage(investigation, request):
        iteration = int(request.parameters.get("iteration") or 1)
        print(f"⚙️ Activating distributed residual-mode search iteration {iteration}")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print(f"✅ Residual-mode distributed search iteration {iteration} complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, f"interpret-multimode-iteration-{iteration}"),
                handler_id="openstar.tess.multimode.interpret",
                parameters={"iteration": iteration},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def multimode_interpret_stage(investigation, request):
        iteration = int(request.parameters.get("iteration") or 1)
        preparations = _all_results_for_handler(
            investigation,
            "openstar.tess.multimode.prepare",
        )
        runs = _all_results_for_handler(
            investigation,
            "openstar.tess.multimode.run",
        )
        preparation = next(
            (item for item in reversed(preparations) if int(item.get("iteration") or 0) == iteration),
            None,
        )
        run = runs[-1] if runs else None
        if preparation is None or run is None:
            raise RuntimeError("Residual-mode interpretation is missing its prepare/run stage.")

        interpreted = interpret_residual_iteration(
            project_status=run,
            preparation=preparation,
        )
        print(f"🧩 Residual-mode interpretation iteration {iteration}")
        for item in interpreted.get("datasetResults") or []:
            label = (
                "combined"
                if item.get("role") == "combined-residual-multimode"
                else f"sector {item.get('sector')}"
            )
            print(
                f"   {label}: period={item.get('candidatePeriodDays')} d, "
                f"prominence={item.get('candidatePeakProminenceRatio')}, "
                f"accepted={item.get('acceptedDistinctMode')}, "
                f"near-family={item.get('resolvedNearPrimaryFamily')}"
            )
        print(
            "   independent sectors with accepted new mode: "
            f"{interpreted.get('acceptedIndependentSectorCount')}"
        )
        print(f"   continue decomposition: {interpreted.get('continueRecommended')}")

        if interpreted.get("continueRecommended") and iteration < MAX_RESIDUAL_ITERATIONS:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, f"prepare-multimode-iteration-{iteration + 1}"),
                handler_id="openstar.tess.multimode.prepare",
                parameters={"iteration": iteration + 1},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "summarize-multimode"),
                handler_id="openstar.tess.multimode.summarize",
                parameters={},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=interpreted,
            next_stage=next_stage,
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
        )

    def multimode_summary_stage(investigation, request):
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("Multi-mode summary requires a resolved physical period.")
        iterations = _all_results_for_handler(
            investigation,
            "openstar.tess.multimode.interpret",
        )
        summary = summarize_multimode_decomposition(
            iteration_results=iterations,
            physical_period_days=float(morphology["resolvedPhysicalPeriodDays"]),
        )
        print("🎛️ Multi-mode frequency decomposition")
        print(f"   iterations completed: {summary.get('iterationsCompleted')}")
        print(f"   classification: {summary.get('classification')}")
        recurrent = summary.get("bestRecurrentSecondaryMode") or {}
        if recurrent:
            print(f"   recurrent secondary period: {recurrent.get('medianPeriodDays')} days")
            print(f"   supporting independent sectors: {recurrent.get('independentSectors')}")
        print(
            "   independent sectors with residual modes: "
            f"{summary.get('independentSectorsWithAcceptedResidualModes')}"
        )
        print(f"   physical mechanism resolved: {summary.get('physicalMechanismResolved')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "multimode"
            / "multimode-v20.7.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.7"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "morphology": sha256_json(morphology),
                "iterationResults": sha256_json(iterations),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def time_frequency_prepare_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        multimode = _latest_result_for_handler(
            investigation,
            "openstar.tess.multimode.summarize",
        )
        if independent_prepare is None:
            raise RuntimeError("v20.8 requires the frozen independent-sector preparation.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.8 requires a morphology-resolved physical period.")
        if multimode is None or multimode.get("recommendedNextTest") != "TIME_FREQUENCY_EVOLUTION_ANALYSIS":
            raise RuntimeError("v20.8 requires v20.7 to recommend TIME_FREQUENCY_EVOLUTION_ANALYSIS.")

        physical_period = float(morphology["resolvedPhysicalPeriodDays"])
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🪟 Preparing sliding-window residual time-frequency search")
        print(f"   resolved physical period: {physical_period} days")
        print("   fitting/subtracting the established fundamental + first harmonic locally")
        spec = build_time_frequency_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            primary_dataset_path=prepared["datasetPath"],
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=physical_period,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        search = spec.get("frequencySearch") or {}
        sectors = {}
        for item in spec.get("preparedWindows") or []:
            sectors.setdefault(str(item.get("sectorKey")), 0)
            sectors[str(item.get("sectorKey"))] += 1
        print(
            "   residual search range: "
            f"{search.get('minimumFrequency'):.4f} - {search.get('maximumFrequency'):.4f} cycles/day"
        )
        print(f"   window length: {spec.get('windowLengthDays')} days")
        print(f"   windows by sector: {sectors}")
        print(f"   distributed window datasets: {len(spec.get('preparedWindows') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedWindows") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-time-frequency"),
                handler_id="openstar.tess.time-frequency.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "morphology": sha256_json(morphology),
                "multimode": sha256_json(multimode),
                "independentPreparation": sha256_json(independent_prepare),
            },
            artifacts=tuple(artifacts),
        )

    def time_frequency_run_stage(investigation, request):
        print("⚙️ Activating distributed sliding-window residual searches")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed time-frequency window search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-time-frequency"),
                handler_id="openstar.tess.time-frequency.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def time_frequency_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.time-frequency.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.time-frequency.run",
        )
        if preparation is None or run is None:
            raise RuntimeError("Time-frequency interpretation is missing its prepare/run stage.")
        interpreted = interpret_time_frequency_project(
            project_status=run,
            preparation=preparation,
        )
        print("🧩 Sliding-window residual interpretation")
        for item in interpreted.get("windowResults") or []:
            print(
                f"   sector {item.get('sector')} window {item.get('windowIndex')}: "
                f"period={item.get('candidatePeriodDays')} d, "
                f"prominence={item.get('candidatePeakProminenceRatio')}, "
                f"accepted={item.get('acceptedTimeFrequencyFeature')}, "
                f"near-family={item.get('nearEstablishedFamily')}"
            )
        print(f"   accepted features: {interpreted.get('acceptedFeatureCount')}")
        print(f"   accepted near established family: {interpreted.get('acceptedNearFamilyCount')}")
        return StageOutcome(
            result=interpreted,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "summarize-time-frequency"),
                handler_id="openstar.tess.time-frequency.summarize",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
        )

    def time_frequency_summary_stage(investigation, request):
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        interpreted = _latest_result_for_handler(
            investigation,
            "openstar.tess.time-frequency.interpret",
        )
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("Time-frequency summary requires a resolved physical period.")
        if interpreted is None:
            raise RuntimeError("Time-frequency summary requires the interpreted window search.")
        summary = summarize_time_frequency_evolution(
            interpretation=interpreted,
            physical_period_days=float(morphology["resolvedPhysicalPeriodDays"]),
        )
        residual = summary.get("residualEvolution") or {}
        family = summary.get("familyEvolution") or {}
        print("🌊 Time-frequency evolution analysis")
        print(f"   windows analyzed: {summary.get('windowCount')}")
        print(f"   accepted residual features: {summary.get('acceptedFeatureCount')}")
        print(f"   residual evolution: {residual.get('classification')}")
        print(f"   established-family evolution: {family.get('classification')}")
        print(f"   classification: {summary.get('classification')}")
        best = residual.get("bestCluster") or {}
        if best:
            print(f"   best residual cluster period: {best.get('medianPeriodDays')} days")
            print(f"   cluster independent sectors: {best.get('independentSectors')}")
        print(f"   physical mechanism resolved: {summary.get('physicalMechanismResolved')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "time-frequency"
            / "time-frequency-v20.8.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.8"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "morphology": sha256_json(morphology),
                "timeFrequencyInterpretation": sha256_json(interpreted),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def nonstationary_prepare_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        time_frequency = _latest_result_for_handler(
            investigation,
            "openstar.tess.time-frequency.summarize",
        )
        if independent_prepare is None:
            raise RuntimeError("v20.9 requires the frozen independent-sector preparation.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.9 requires a morphology-resolved physical period.")
        if time_frequency is None or time_frequency.get("recommendedNextTest") != "LONG_BASELINE_NONSTATIONARY_MODE_MODELING":
            raise RuntimeError("v20.9 requires v20.8 to recommend LONG_BASELINE_NONSTATIONARY_MODE_MODELING.")

        physical_period = float(morphology["resolvedPhysicalPeriodDays"])
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🌀 Preparing generic long-baseline nonstationary work")
        print(f"   resolved physical period: {physical_period} days")
        print("   subtracting the established fundamental + first harmonic per sector")
        print("   creating ordinary Lomb-Scargle datasets over a deterministic frequency-drift grid")
        spec = build_nonstationary_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            primary_dataset_path=prepared["datasetPath"],
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=physical_period,
            time_frequency_summary=time_frequency,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        search = spec.get("frequencySearch") or {}
        grid = spec.get("driftGrid") or {}
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   residual center period: {spec.get('residualCenterPeriodDays')} days")
        print(
            "   frequency range: "
            f"{search.get('minimumFrequency'):.6f} - {search.get('maximumFrequency'):.6f} cycles/day"
        )
        print(
            "   fractional drift grid: "
            f"{grid.get('minimumFractionalFrequencyDriftPerDay')} to "
            f"{grid.get('maximumFractionalFrequencyDriftPerDay')} /day "
            f"({grid.get('count')} candidates)"
        )
        print(f"   model groups: {[item.get('groupID') for item in spec.get('groups') or []]}")
        print(f"   distributed datasets: {len(spec.get('preparedDatasets') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedDatasets") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        artifacts.append(_artifact(Path(spec["analysisSeriesPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-nonstationary"),
                handler_id="openstar.tess.nonstationary.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "morphology": sha256_json(morphology),
                "timeFrequency": sha256_json(time_frequency),
                "independentPreparation": sha256_json(independent_prepare),
            },
            artifacts=tuple(artifacts),
        )

    def nonstationary_run_stage(investigation, request):
        print("⚙️ Activating generic long-baseline nonstationary work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed nonstationary drift-grid search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-nonstationary"),
                handler_id="openstar.tess.nonstationary.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def nonstationary_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.run",
        )
        if preparation is None or run is None:
            raise RuntimeError("Nonstationary interpretation is missing its prepare/run stage.")
        interpreted = interpret_nonstationary_project(
            project_status=run,
            preparation=preparation,
        )
        print("🧩 Interpreting distributed frequency-drift grid")
        for group_id, group in (interpreted.get("groups") or {}).items():
            stationary = group.get("stationaryCandidate") or {}
            best = group.get("bestCandidate") or {}
            print(f"   group: {group_id}")
            print(
                "      stationary: "
                f"period={stationary.get('candidatePeriodDays')} d, "
                f"power={stationary.get('candidatePower')}"
            )
            print(
                "      best drift: "
                f"period={best.get('candidatePeriodDays')} d, "
                f"q={best.get('fractionalFrequencyDriftPerDay')}, "
                f"power={best.get('candidatePower')}"
            )
            print(f"      power gain: {group.get('distributedPowerGainOverStationary')}")
        return StageOutcome(
            result=interpreted,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "summarize-nonstationary"),
                handler_id="openstar.tess.nonstationary.summarize",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
        )

    def nonstationary_summary_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.prepare",
        )
        interpreted = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.interpret",
        )
        if preparation is None or interpreted is None:
            raise RuntimeError("Nonstationary summary requires prepare + interpretation stages.")
        summary = summarize_nonstationary_modeling(
            interpretation=interpreted,
            preparation=preparation,
        )
        comparison = summary.get("modelComparison") or {}
        preferred = summary.get("preferredModel") or {}
        print("🌀 Long-baseline nonstationary model comparison")
        print(f"   classification: {summary.get('classification')}")
        print(f"   preferred model: {comparison.get('bestModelID')}")
        print(f"   BIC improvement over null: {comparison.get('bicImprovementOverNull')}")
        print(f"   preferred period at reference: {summary.get('preferredPeriodAtReferenceDays')} days")
        print(f"   fractional frequency drift/day: {summary.get('fractionalFrequencyDriftPerDay')}")
        print(f"   signal sectors: {preferred.get('signalSectors')}")
        print(f"   physical mechanism resolved: {summary.get('physicalMechanismResolved')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "nonstationary"
            / "nonstationary-v20.9.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.9"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "interpretation": sha256_json(interpreted),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def residual_mode_localization_prepare_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        identity = _latest_result_for_handler(
            investigation,
            "openstar.tess.catalog-identity",
        )
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        nonstationary = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.summarize",
        )
        if identity is None:
            raise RuntimeError("v20.10 requires the completed catalog-identity stage.")
        if independent_prepare is None:
            raise RuntimeError("v20.10 requires frozen independent-sector metadata.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.10 requires the morphology-resolved physical period.")
        if nonstationary is None or nonstationary.get("recommendedNextTest") != "RESIDUAL_MODE_PIXEL_LOCALIZATION":
            raise RuntimeError("v20.10 requires v20.9 to recommend RESIDUAL_MODE_PIXEL_LOCALIZATION.")

        physical_period = float(morphology["resolvedPhysicalPeriodDays"])
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🎯 Preparing distributed drifting residual-mode pixel localization")
        print(f"   TIC: {prepared.get('ticID')}")
        print(f"   residual period at reference: {nonstationary.get('preferredPeriodAtReferenceDays')} days")
        print(f"   fractional frequency drift/day: {nonstationary.get('fractionalFrequencyDriftPerDay')}")
        print("   downloading TESS pixel stamps, subtracting the established family per pixel")
        print("   each usable pixel becomes one ordinary generic Lomb-Scargle dataset")
        spec = build_residual_mode_pixel_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=physical_period,
            nonstationary_summary=nonstationary,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   signal sectors: {spec.get('signalSectors')}")
        print(f"   prepared pixel datasets: {len(spec.get('preparedPixels') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedPixels") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-residual-mode-localization"),
                handler_id="openstar.tess.residual-mode-localization.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "independentPreparation": sha256_json(independent_prepare),
                "morphology": sha256_json(morphology),
                "nonstationaryModeling": sha256_json(nonstationary),
            },
            artifacts=tuple(artifacts),
        )

    def residual_mode_localization_run_stage(investigation, request):
        print("⚙️ Activating generic residual-mode pixel work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed residual-mode pixel search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-residual-mode-localization"),
                handler_id="openstar.tess.residual-mode-localization.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def residual_mode_localization_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization.run",
        )
        if preparation is None or run is None:
            raise RuntimeError("Residual-mode localization requires prepare + run stages.")
        localization = interpret_residual_mode_pixel_project(
            project_status=run,
            preparation=preparation,
        )
        cross = localization.get("crossSector") or {}
        print("🎯 Drifting residual-mode source localization")
        for item in localization.get("sectorResults") or []:
            print(
                f"   sector {item.get('sector')}: offset={item.get('offsetPixels')} px, "
                f"peakPower={item.get('peakPower')}, contrast={item.get('powerContrast')}, "
                f"classification={item.get('classification')}"
            )
        print(f"   classification: {cross.get('classification')}")
        print(f"   residual mode origin: {cross.get('residualModeOrigin')}")
        print(f"   independent eligible sectors: {cross.get('independentEligibleSectorCount')}")
        print(f"   target-supporting sectors: {cross.get('targetSupportingSectors')}")
        print(f"   off-target sectors: {cross.get('offTargetSectors')}")
        print(f"   ambiguous sectors: {cross.get('ambiguousSectors')}")
        print(f"   median sky separation: {cross.get('medianSkySeparationArcsec')} arcsec")
        print(f"   recommended next test: {localization.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "residual-mode-localization"
            / "residual-mode-localization-v20.10.json"
        )
        _write_json(artifact_path, localization)
        return StageOutcome(
            result=localization,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.10"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )


    def residual_mode_localization_review_prepare_stage(investigation, request):
        prepared = _result(investigation, "001-prepare-target")
        identity = _latest_result_for_handler(
            investigation,
            "openstar.tess.catalog-identity",
        )
        independent_prepare = _latest_result_for_handler(
            investigation,
            "openstar.tess.independent.prepare",
        )
        morphology = _latest_result_for_handler(
            investigation,
            "openstar.tess.morphology.analyze",
        )
        nonstationary = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.summarize",
        )
        residual_localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization.interpret",
        )
        if identity is None:
            raise RuntimeError("v20.11 requires the completed catalog-identity stage.")
        if independent_prepare is None:
            raise RuntimeError("v20.11 requires frozen independent-sector metadata.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.11 requires the morphology-resolved physical period.")
        if nonstationary is None:
            raise RuntimeError("v20.11 requires the completed v20.9 nonstationary model.")
        if residual_localization is None:
            raise RuntimeError("v20.11 requires the completed v20.10 residual-mode localization.")
        if residual_localization.get("recommendedNextTest") != "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW":
            raise RuntimeError(
                "v20.10 did not recommend RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW."
            )

        physical_period = float(morphology["resolvedPhysicalPeriodDays"])
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🧭 Preparing time-resolved residual-mode source localization review")
        print(f"   TIC: {prepared.get('ticID')}")
        print(f"   residual period at reference: {nonstationary.get('preferredPeriodAtReferenceDays')} days")
        print(f"   fractional frequency drift/day: {nonstationary.get('fractionalFrequencyDriftPerDay')}")
        print("   re-downloading TESS pixel stamps and reusing the established-family/drift model")
        print("   each time-windowed usable pixel becomes one ordinary generic Lomb-Scargle dataset")
        spec = build_residual_mode_localization_review_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=physical_period,
            nonstationary_summary=nonstationary,
            residual_localization_summary=residual_localization,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   time windows: {len(spec.get('windowMetadata') or [])}")
        print(f"   prepared pixel-window datasets: {len(spec.get('preparedPixels') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedPixels") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-residual-mode-localization-review"),
                handler_id="openstar.tess.residual-mode-localization-review.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "independentPreparation": sha256_json(independent_prepare),
                "morphology": sha256_json(morphology),
                "nonstationaryModeling": sha256_json(nonstationary),
                "residualModeLocalization": sha256_json(residual_localization),
            },
            artifacts=tuple(artifacts),
        )

    def residual_mode_localization_review_run_stage(investigation, request):
        print("⚙️ Activating generic time-resolved residual localization work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed residual localization review complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-residual-mode-localization-review"),
                handler_id="openstar.tess.residual-mode-localization-review.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def residual_mode_localization_review_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization-review.prepare",
        )
        run = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization-review.run",
        )
        if preparation is None or run is None:
            raise RuntimeError("Residual-mode localization review requires prepare + run stages.")
        review = interpret_residual_mode_localization_review_project(
            project_status=run,
            preparation=preparation,
        )
        cross = review.get("crossTime") or {}
        print("🧭 Time-resolved residual-mode source localization review")
        for item in review.get("sectorTemporalSummaries") or []:
            print(
                f"   sector {item.get('sector')}: classification={item.get('classification')}, "
                f"targetWindows={item.get('targetWindowCount')}, "
                f"offTargetWindows={item.get('offTargetWindowCount')}, "
                f"qualityWindows={item.get('qualityWindowCount')}"
            )
        print(f"   classification: {cross.get('classification')}")
        print(f"   residual mode origin: {cross.get('residualModeOrigin')}")
        print(f"   target-dominant sectors: {cross.get('targetDominantSectors')}")
        print(f"   off-target-dominant sectors: {cross.get('offTargetDominantSectors')}")
        print(f"   source-switching sectors: {cross.get('sourceSwitchingSectors')}")
        print(f"   off-target sky scatter: {cross.get('offTargetSkyOffsetScatterArcsec')} arcsec")
        print(f"   recommended next test: {review.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "residual-mode-localization-review"
            / "residual-mode-localization-review-v20.11.json"
        )
        _write_json(artifact_path, review)
        return StageOutcome(
            result=review,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.11"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def multisource_residual_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        independent_prepare = _latest_result_for_handler(investigation, "openstar.tess.independent.prepare")
        morphology = _latest_result_for_handler(investigation, "openstar.tess.morphology.analyze")
        nonstationary = _latest_result_for_handler(investigation, "openstar.tess.nonstationary.summarize")
        review = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization-review.interpret",
        )
        if prepared is None:
            raise RuntimeError("v20.12 requires the completed target-preparation stage.")
        if identity is None:
            raise RuntimeError("v20.12 requires the completed catalog-identity stage.")
        if independent_prepare is None:
            raise RuntimeError("v20.12 requires frozen independent-sector metadata.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.12 requires the morphology-resolved physical period.")
        if nonstationary is None:
            raise RuntimeError("v20.12 requires the completed v20.9 nonstationary model.")
        if review is None:
            raise RuntimeError("v20.12 requires the completed v20.11 localization review.")
        if review.get("recommendedNextTest") != "MULTI_SOURCE_RESIDUAL_DECOMPOSITION":
            raise RuntimeError("v20.11 did not recommend MULTI_SOURCE_RESIDUAL_DECOMPOSITION.")

        physical_period = float(morphology["resolvedPhysicalPeriodDays"])
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🧩 Preparing multi-source residual decomposition")
        print(f"   TIC: {prepared.get('ticID')}")
        print(f"   residual period at reference: {nonstationary.get('preferredPeriodAtReferenceDays')} days")
        print(f"   fractional frequency drift/day: {nonstationary.get('fractionalFrequencyDriftPerDay')}")
        print("   inferring target + offset spatial components from v20.11 time-resolved localization")
        print("   spatially decomposed residual component light curves become ordinary generic Lomb-Scargle datasets")
        spec = build_multisource_residual_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=physical_period,
            nonstationary_summary=nonstationary,
            localization_review=review,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   spatial components: {[item.get('componentID') for item in spec.get('spatialComponents') or []]}")
        print(f"   prepared component datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedSeries") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-multi-source-residual"),
                handler_id="openstar.tess.multi-source-residual.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "independentPreparation": sha256_json(independent_prepare),
                "morphology": sha256_json(morphology),
                "nonstationaryModeling": sha256_json(nonstationary),
                "localizationReview": sha256_json(review),
            },
            artifacts=tuple(artifacts),
        )

    def multisource_residual_run_stage(investigation, request):
        print("⚙️ Activating generic multi-source residual component work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed multi-source residual decomposition search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-multi-source-residual"),
                handler_id="openstar.tess.multi-source-residual.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def multisource_residual_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation, "openstar.tess.multi-source-residual.prepare"
        )
        run = _latest_result_for_handler(
            investigation, "openstar.tess.multi-source-residual.run"
        )
        if preparation is None or run is None:
            raise RuntimeError("Multi-source residual decomposition requires prepare + run stages.")
        summary = interpret_multisource_residual_project(
            project_status=run,
            preparation=preparation,
        )
        print("🧩 Multi-source residual decomposition")
        for item in summary.get("componentSummaries") or []:
            print(
                f"   {item.get('componentID')}: type={item.get('componentType')}, "
                f"independentSupport={item.get('independentSupportCount')}, "
                f"combinedPower={item.get('combinedPower')}, "
                f"combinedPeriod={item.get('combinedPeriodDays')}"
            )
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   target component: {summary.get('targetComponentID')}")
        print(f"   best offset component: {summary.get('bestOffsetComponentID')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "multi-source-residual"
            / "multi-source-residual-v20.12.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.12"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def offset_source_identification_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        multisource = _latest_result_for_handler(
            investigation,
            "openstar.tess.multi-source-residual.interpret",
        )
        if prepared is None:
            raise RuntimeError("v20.13 requires the completed target-preparation stage.")
        if identity is None:
            raise RuntimeError("v20.13 requires the completed catalog-identity stage.")
        if multisource is None:
            raise RuntimeError("v20.13 requires the completed v20.12 multi-source residual decomposition.")
        if multisource.get("recommendedNextTest") not in {
            "IDENTIFY_OFFSET_RESIDUAL_VARIABLE_SOURCE",
            "NEIGHBOR_SOURCE_IDENTIFICATION_AND_CATALOG_CROSSMATCH",
        }:
            raise RuntimeError("v20.12 did not recommend offset/neighbor source identification.")

        print("🔎 Identifying the dominant offset residual component in external catalogs")
        print(f"   TIC target: {prepared.get('ticID')}")
        print(f"   offset component: {multisource.get('bestOffsetComponentID')}")
        print("   querying TIC / Gaia DR3 / SIMBAD / VSX")
        print("   no coordinator and no distributed workers required")
        summary = identify_offset_residual_source(
            tic_id=int(prepared["ticID"]),
            identity=identity,
            multisource_summary=multisource,
        )
        component = summary.get("component") or {}
        best = summary.get("bestCandidate") or {}
        ids = best.get("catalogIDs") or {}
        print(f"   component sky: RA={((component.get('componentSky') or {}).get('raDeg'))}, Dec={((component.get('componentSky') or {}).get('decDeg'))}")
        print(f"   component target separation: {component.get('targetSeparationArcsec')} arcsec")
        print(f"   classification: {summary.get('classification')}")
        print(f"   TIC counterpart: {ids.get('ticID')}")
        print(f"   Gaia DR3 counterpart: {ids.get('gaiaDR3SourceID')}")
        print(f"   counterpart separation: {best.get('separationArcsec')} arcsec")
        print(f"   known variable catalog evidence: {summary.get('knownVariableCatalogEvidence')}")
        print(f"   catalog query errors: {len(summary.get('queryErrors') or [])}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "offset-source-identification"
            / "offset-source-identification-v20.13.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.13"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "multiSourceResidual": sha256_json(multisource),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def offset_source_variability_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        independent_prepare = _latest_result_for_handler(investigation, "openstar.tess.independent.prepare")
        morphology = _latest_result_for_handler(investigation, "openstar.tess.morphology.analyze")
        nonstationary = _latest_result_for_handler(investigation, "openstar.tess.nonstationary.summarize")
        independent_prepare = _latest_result_for_handler(investigation, "openstar.tess.independent.prepare")
        multisource = _latest_result_for_handler(investigation, "openstar.tess.multi-source-residual.interpret")
        offset_source = _latest_result_for_handler(
            investigation, "openstar.tess.offset-source-identification.analyze"
        )
        if prepared is None or identity is None or independent_prepare is None:
            raise RuntimeError("v20.14 requires frozen target, identity, and independent-sector preparation.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.14 requires the morphology-resolved physical period.")
        if nonstationary is None or multisource is None or offset_source is None:
            raise RuntimeError("v20.14 requires completed v20.9, v20.12, and v20.13 results.")
        if offset_source.get("recommendedNextTest") not in {
            "OFFSET_SOURCE_VARIABILITY_VALIDATION",
            "OFFSET_SOURCE_VARIABILITY_MATCH_TEST",
        }:
            raise RuntimeError("v20.13 did not recommend direct offset-source variability validation.")

        candidate = offset_source.get("bestCandidate") or {}
        ids = candidate.get("catalogIDs") or {}
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🧪 Preparing direct variability validation of the catalog-matched offset source")
        print(f"   target TIC: {prepared.get('ticID')}")
        print(f"   counterpart TIC: {ids.get('ticID')}")
        print(f"   counterpart Gaia DR3: {ids.get('gaiaDR3SourceID')}")
        print(f"   offset component: {multisource.get('bestOffsetComponentID')}")
        print("   simultaneously deblending target-control and catalog-counterpart residual series per sector")
        print("   established 13.72-day family is removed before distributed residual searches")
        spec = build_offset_source_variability_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            target_tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=float(morphology["resolvedPhysicalPeriodDays"]),
            nonstationary_summary=nonstationary,
            multisource_summary=multisource,
            offset_source_identification=offset_source,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   reference residual period: {spec.get('referencePeriodDays')} days")
        print(f"   prepared source-component datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedSeries") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-offset-source-variability"),
                handler_id="openstar.tess.offset-source-variability.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "morphology": sha256_json(morphology),
                "nonstationaryModeling": sha256_json(nonstationary),
                "multiSourceResidual": sha256_json(multisource),
                "offsetSourceIdentification": sha256_json(offset_source),
            },
            artifacts=tuple(artifacts),
        )

    def offset_source_variability_run_stage(investigation, request):
        print("⚙️ Activating generic catalog-counterpart variability work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed catalog-counterpart variability search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-offset-source-variability"),
                handler_id="openstar.tess.offset-source-variability.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def offset_source_variability_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation, "openstar.tess.offset-source-variability.prepare"
        )
        run = _latest_result_for_handler(
            investigation, "openstar.tess.offset-source-variability.run"
        )
        if preparation is None or run is None:
            raise RuntimeError("Offset-source variability validation requires prepare + run stages.")
        summary = interpret_offset_source_variability_project(
            project_status=run,
            preparation=preparation,
        )
        candidate = summary.get("catalogCounterpartEvidence") or {}
        target_control = summary.get("targetControl") or {}
        ids = summary.get("catalogCounterpart") or {}
        print("🧪 Offset catalog-counterpart variability validation")
        print(f"   counterpart TIC: {ids.get('ticID')}")
        print(f"   counterpart Gaia DR3: {ids.get('gaiaDR3SourceID')}")
        print(
            f"   counterpart: independentSupport={candidate.get('independentSupportCount')}, "
            f"combinedPower={candidate.get('combinedPower')}, "
            f"combinedPeriod={candidate.get('combinedPeriodDays')}"
        )
        print(
            f"   target control: independentSupport={target_control.get('independentSupportCount')}, "
            f"combinedPower={target_control.get('combinedPower')}, "
            f"combinedPeriod={target_control.get('combinedPeriodDays')}"
        )
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "offset-source-variability"
            / "offset-source-variability-v20.14.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.14"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def calibrated_prf_deblending_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        independent_prepare = _latest_result_for_handler(investigation, "openstar.tess.independent.prepare")
        morphology = _latest_result_for_handler(investigation, "openstar.tess.morphology.analyze")
        nonstationary = _latest_result_for_handler(investigation, "openstar.tess.nonstationary.summarize")
        multisource = _latest_result_for_handler(investigation, "openstar.tess.multi-source-residual.interpret")
        offset_source = _latest_result_for_handler(
            investigation, "openstar.tess.offset-source-identification.analyze"
        )
        offset_variability = _latest_result_for_handler(
            investigation, "openstar.tess.offset-source-variability.interpret"
        )
        if prepared is None or identity is None or independent_prepare is None:
            raise RuntimeError("v20.15 requires frozen target, identity, and independent-sector preparation.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.15 requires the morphology-resolved physical period.")
        if nonstationary is None or multisource is None or offset_source is None or offset_variability is None:
            raise RuntimeError("v20.15 requires completed v20.9, v20.12, v20.13, and v20.14 results.")
        if offset_variability.get("recommendedNextTest") != "CALIBRATED_PRF_SOURCE_DEBLENDING":
            raise RuntimeError("v20.14 did not recommend calibrated PRF source deblending.")

        counterpart = offset_source.get("bestCandidate") or {}
        ids = counterpart.get("catalogIDs") or {}
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🔬 Preparing sector-calibrated ePRF source deblending")
        print(f"   target TIC: {prepared.get('ticID')}")
        print(f"   counterpart TIC: {ids.get('ticID')}")
        print(f"   counterpart Gaia DR3: {ids.get('gaiaDR3SourceID')}")
        print(f"   offset component: {multisource.get('bestOffsetComponentID')}")
        print("   calibrating a sector-specific empirical pixel-response shape on each TPF")
        print("   separated target/counterpart residual series then run as ordinary Lomb-Scargle work")
        spec = build_calibrated_prf_deblending_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            target_tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=float(morphology["resolvedPhysicalPeriodDays"]),
            nonstationary_summary=nonstationary,
            multisource_summary=multisource,
            offset_source_identification=offset_source,
            offset_source_variability=offset_variability,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   deblend backend: {spec.get('deblendBackend')}")
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   reference residual period: {spec.get('referencePeriodDays')} days")
        print(f"   prepared source-component datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedSeries") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-calibrated-prf-deblending"),
                handler_id="openstar.tess.calibrated-prf-deblending.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "morphology": sha256_json(morphology),
                "nonstationaryModeling": sha256_json(nonstationary),
                "multiSourceResidual": sha256_json(multisource),
                "offsetSourceIdentification": sha256_json(offset_source),
                "offsetSourceVariability": sha256_json(offset_variability),
            },
            artifacts=tuple(artifacts),
        )

    def calibrated_prf_deblending_run_stage(investigation, request):
        print("⚙️ Activating generic calibrated-ePRF source work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed calibrated-ePRF source search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-calibrated-prf-deblending"),
                handler_id="openstar.tess.calibrated-prf-deblending.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def calibrated_prf_deblending_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation, "openstar.tess.calibrated-prf-deblending.prepare"
        )
        run = _latest_result_for_handler(
            investigation, "openstar.tess.calibrated-prf-deblending.run"
        )
        if preparation is None or run is None:
            raise RuntimeError("Calibrated ePRF source deblending requires prepare + run stages.")
        summary = interpret_calibrated_prf_deblending_project(
            project_status=run,
            preparation=preparation,
        )
        candidate = summary.get("catalogCounterpartEvidence") or {}
        target_control = summary.get("targetControl") or {}
        ids = summary.get("catalogCounterpart") or {}
        print("🔬 Calibrated ePRF source deblending")
        print(f"   backend: {summary.get('deblendBackend')}")
        print(f"   counterpart TIC: {ids.get('ticID')}")
        print(f"   counterpart Gaia DR3: {ids.get('gaiaDR3SourceID')}")
        print(
            f"   counterpart: independentSupport={candidate.get('independentSupportCount')}, "
            f"combinedPower={candidate.get('combinedPower')}, "
            f"combinedPeriod={candidate.get('combinedPeriodDays')}"
        )
        print(
            f"   target control: independentSupport={target_control.get('independentSupportCount')}, "
            f"combinedPower={target_control.get('combinedPower')}, "
            f"combinedPeriod={target_control.get('combinedPeriodDays')}"
        )
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "calibrated-prf-deblending"
            / "calibrated-prf-deblending-v20.15.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.15"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def difference_image_localization_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        independent_prepare = _latest_result_for_handler(investigation, "openstar.tess.independent.prepare")
        morphology = _latest_result_for_handler(investigation, "openstar.tess.morphology.analyze")
        nonstationary = _latest_result_for_handler(investigation, "openstar.tess.nonstationary.summarize")
        multisource = _latest_result_for_handler(investigation, "openstar.tess.multi-source-residual.interpret")
        offset_source = _latest_result_for_handler(
            investigation, "openstar.tess.offset-source-identification.analyze"
        )
        calibrated = _latest_result_for_handler(
            investigation, "openstar.tess.calibrated-prf-deblending.interpret"
        )
        if prepared is None or identity is None or independent_prepare is None:
            raise RuntimeError("v20.16 requires frozen target, identity, and independent-sector preparation.")
        if morphology is None or not morphology.get("physicalCycleResolved"):
            raise RuntimeError("v20.16 requires the morphology-resolved physical period.")
        if nonstationary is None or multisource is None or offset_source is None or calibrated is None:
            raise RuntimeError("v20.16 requires completed v20.9, v20.12, v20.13, and v20.15 results.")
        if calibrated.get("recommendedNextTest") != "DIFFERENCE_IMAGE_SOURCE_LOCALIZATION":
            raise RuntimeError("v20.15 did not recommend difference-image source localization.")

        counterpart = offset_source.get("bestCandidate") or {}
        ids = counterpart.get("catalogIDs") or {}
        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🖼️ Preparing residual-frequency refinement + difference-image source localization")
        print(f"   target TIC: {prepared.get('ticID')}")
        print(f"   counterpart TIC: {ids.get('ticID')}")
        print(f"   counterpart Gaia DR3: {ids.get('gaiaDR3SourceID')}")
        print("   established 13.72-day family is removed from each pixel before difference imaging")
        print("   residual aperture frequencies are refined by ordinary generic Lomb-Scargle work")
        spec = build_difference_image_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            target_tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=float(morphology["resolvedPhysicalPeriodDays"]),
            nonstationary_summary=nonstationary,
            multisource_summary=multisource,
            offset_source_identification=offset_source,
            calibrated_prf_summary=calibrated,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   reference residual period: {spec.get('referencePeriodDays')} days")
        print(f"   prepared frequency datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   cached sector pixel cubes: {len(spec.get('sectorCaches') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedSeries") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-difference-image-localization"),
                handler_id="openstar.tess.difference-image-localization.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "identity": sha256_json(identity),
                "morphology": sha256_json(morphology),
                "nonstationaryModeling": sha256_json(nonstationary),
                "multiSourceResidual": sha256_json(multisource),
                "offsetSourceIdentification": sha256_json(offset_source),
                "calibratedPrfDeblending": sha256_json(calibrated),
            },
            artifacts=tuple(artifacts),
        )

    def difference_image_localization_run_stage(investigation, request):
        print("⚙️ Activating generic residual-frequency refinement work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed difference-image frequency refinement complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-difference-image-localization"),
                handler_id="openstar.tess.difference-image-localization.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def difference_image_localization_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation, "openstar.tess.difference-image-localization.prepare"
        )
        run = _latest_result_for_handler(
            investigation, "openstar.tess.difference-image-localization.run"
        )
        if preparation is None or run is None:
            raise RuntimeError("Difference-image source localization requires prepare + run stages.")
        summary = interpret_difference_image_project(
            project_status=run,
            preparation=preparation,
        )
        counterpart = summary.get("catalogCounterpart") or {}
        print("🖼️ Difference-image residual source localization")
        print(f"   counterpart TIC: {counterpart.get('ticID')}")
        print(f"   counterpart Gaia DR3: {counterpart.get('gaiaDR3SourceID')}")
        for item in summary.get("sectorResults") or []:
            print(
                f"   sector {item.get('sector')}: class={item.get('classification')}, "
                f"target={item.get('targetDistancePixels'):.3f}px, "
                f"counterpart={item.get('counterpartDistancePixels'):.3f}px, "
                f"peakSNR={(item.get('differenceImage') or {}).get('peakSNR'):.2f}"
            )
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   counterpart-supporting sectors: {summary.get('counterpartSupportingSectors')}")
        print(f"   target-supporting sectors: {summary.get('targetSupportingSectors')}")
        print(f"   ambiguous sectors: {summary.get('ambiguousSectors')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "difference-image-localization"
            / "difference-image-localization-v20.16.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.16"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def frequency_localized_pixel_response_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        difference_preparation = _latest_result_for_handler(
            investigation, "openstar.tess.difference-image-localization.prepare"
        )
        difference_summary = _latest_result_for_handler(
            investigation, "openstar.tess.difference-image-localization.interpret"
        )
        if prepared is None or difference_preparation is None or difference_summary is None:
            raise RuntimeError("v20.17 requires frozen target plus completed v20.16 preparation and interpretation.")
        if difference_summary.get("recommendedNextTest") != "FREQUENCY_LOCALIZED_PIXEL_RESPONSE_CONFIRMATION":
            raise RuntimeError("v20.16 did not recommend frequency-localized pixel-response confirmation.")

        artifact_root = store.directory_for(investigation.id) / "artifacts"
        counterpart = difference_summary.get("catalogCounterpart") or {}
        print("🎚️ Preparing distributed frequency-localized pixel-response confirmation")
        print(f"   target TIC: {prepared.get('ticID')}")
        print(f"   counterpart TIC: {counterpart.get('ticID')}")
        print(f"   counterpart Gaia DR3: {counterpart.get('gaiaDR3SourceID')}")
        print("   reusing v20.16 established-family-prewhitened pixel caches; no new MAST download required")
        print("   each usable pixel becomes an ordinary narrow-band openstar.lomb-scargle.v1 dataset")
        spec = build_frequency_localized_pixel_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            difference_image_preparation=difference_preparation,
            difference_image_summary=difference_summary,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   reference residual period: {spec.get('referencePeriodDays')} days")
        print(f"   sectors: {len(spec.get('sectorPreparations') or [])}")
        print(f"   prepared pixel datasets: {len(spec.get('preparedPixels') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedPixels") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-frequency-localized-pixel-response"),
                handler_id="openstar.tess.frequency-localized-pixel-response.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "differenceImagePreparation": sha256_json(difference_preparation),
                "differenceImageLocalization": sha256_json(difference_summary),
            },
            artifacts=tuple(artifacts),
        )

    def frequency_localized_pixel_response_run_stage(investigation, request):
        print("⚙️ Activating generic narrow-band per-pixel work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed frequency-localized pixel search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-frequency-localized-pixel-response"),
                handler_id="openstar.tess.frequency-localized-pixel-response.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def frequency_localized_pixel_response_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation, "openstar.tess.frequency-localized-pixel-response.prepare"
        )
        run = _latest_result_for_handler(
            investigation, "openstar.tess.frequency-localized-pixel-response.run"
        )
        if preparation is None or run is None:
            raise RuntimeError("Frequency-localized pixel-response confirmation requires prepare + run stages.")
        summary = interpret_frequency_localized_pixel_project(
            project_status=run,
            preparation=preparation,
        )
        counterpart = summary.get("catalogCounterpart") or {}
        print("🎚️ Frequency-localized pixel-response confirmation")
        print(f"   counterpart TIC: {counterpart.get('ticID')}")
        print(f"   counterpart Gaia DR3: {counterpart.get('gaiaDR3SourceID')}")
        for item in summary.get("sectorResults") or []:
            response = item.get("response") or {}
            print(
                f"   sector {item.get('sector')}: class={item.get('classification')}, "
                f"frequency={item.get('targetFrequency'):.8f} c/d, "
                f"target={item.get('targetDistancePixels'):.3f}px, "
                f"counterpart={item.get('counterpartDistancePixels'):.3f}px, "
                f"powerContrast={response.get('powerContrast'):.2f}, "
                f"phaseConcentration={response.get('phaseConcentration'):.2f}"
            )
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   counterpart-supporting sectors: {summary.get('counterpartSupportingSectors')}")
        print(f"   target-supporting sectors: {summary.get('targetSupportingSectors')}")
        print(f"   ambiguous sectors: {summary.get('ambiguousSectors')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "frequency-localized-pixel-response"
            / "frequency-localized-pixel-response-v20.17.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.17"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def official_spoc_prf_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        independent_prepare = _latest_result_for_handler(
            investigation, "openstar.tess.independent.prepare"
        )
        nonstationary = _latest_result_for_handler(investigation, "openstar.tess.nonstationary.summarize")
        multisource = _latest_result_for_handler(investigation, "openstar.tess.multi-source-residual.interpret")
        offset_identity = _latest_result_for_handler(investigation, "openstar.tess.offset-source-identification.analyze")
        frequency_localized = _latest_result_for_handler(
            investigation, "openstar.tess.frequency-localized-pixel-response.interpret"
        )
        morphology = _latest_result_for_handler(investigation, "openstar.tess.morphology.analyze")
        if any(
            item is None
            for item in (
                prepared,
                identity,
                independent_prepare,
                nonstationary,
                multisource,
                offset_identity,
                frequency_localized,
                morphology,
            )
        ):
            raise RuntimeError(
                "v20.18 requires the frozen target, independent-sector preparation, "
                "and completed v20.4/v20.9/v20.12/v20.13/v20.17 evidence."
            )
        if frequency_localized.get("recommendedNextTest") != "OFFICIAL_SPOC_PRF_FORWARD_MODELING":
            raise RuntimeError("v20.17 did not recommend official SPOC PRF forward modeling.")
        physical_period = morphology.get("resolvedPhysicalPeriodDays")
        if physical_period is None:
            raise RuntimeError("v20.18 requires the morphology-resolved physical period.")

        artifact_root = store.directory_for(investigation.id) / "artifacts"
        counterpart = offset_identity.get("bestCandidate") or {}
        print("🛰️ Preparing official SPOC PRF forward modeling")
        print(f"   target TIC: {prepared.get('ticID')}")
        print(f"   catalog counterpart: {counterpart.get('catalogIDs')}")
        print("   downloading/interpolating the public SPOC PRF calibration for each sector camera/CCD")
        print("   official-PRF-separated target/counterpart residual series become ordinary openstar.lomb-scargle.v1 datasets")
        spec = build_official_spoc_prf_project(
            source_project_path=prepared["sourceProjectPath"],
            source_dataset_entry=prepared["sourceDatasetEntry"],
            target_tic_id=int(prepared["ticID"]),
            identity=identity,
            primary_sector=prepared.get("sector"),
            independent_spec=independent_prepare,
            physical_period_days=float(physical_period),
            nonstationary_summary=nonstationary,
            multisource_summary=multisource,
            offset_source_identification=offset_identity,
            frequency_localized_summary=frequency_localized,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )
        print(f"   backend: {spec.get('deblendBackend')}")
        print(f"   generic workload: {spec.get('workloadID')}")
        print(f"   reference residual period: {spec.get('referencePeriodDays')} days")
        print(f"   successful sector calibrations: {len(spec.get('calibrationDiagnostics') or [])}")
        print(f"   prepared source-component datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   work units: {spec.get('totalWorkUnits')}")

        artifacts = [
            _artifact(Path(item["datasetPath"]), "application/json")
            for item in spec.get("preparedSeries") or []
            if item.get("datasetPath")
        ]
        artifacts.append(_artifact(Path(spec["projectPath"]), "application/json"))
        return StageOutcome(
            result=spec,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "run-official-spoc-prf-forward-modeling"),
                handler_id="openstar.tess.official-spoc-prf-forward-modeling.run",
                parameters={"projectPath": spec["projectPath"]},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "frequencyLocalizedPixelResponse": sha256_json(frequency_localized),
                "offsetSourceIdentification": sha256_json(offset_identity),
            },
            artifacts=tuple(artifacts),
        )

    def official_spoc_prf_run_stage(investigation, request):
        print("⚙️ Activating generic official-PRF-separated source work units")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed official SPOC PRF source search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-official-spoc-prf-forward-modeling"),
                handler_id="openstar.tess.official-spoc-prf-forward-modeling.interpret",
                parameters={},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def official_spoc_prf_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation, "openstar.tess.official-spoc-prf-forward-modeling.prepare"
        )
        run = _latest_result_for_handler(
            investigation, "openstar.tess.official-spoc-prf-forward-modeling.run"
        )
        if preparation is None or run is None:
            raise RuntimeError("Official SPOC PRF forward modeling requires prepare + run stages.")
        summary = interpret_official_spoc_prf_project(
            project_status=run,
            preparation=preparation,
        )
        counterpart = summary.get("catalogCounterpart") or {}
        candidate = summary.get("catalogCounterpartEvidence") or {}
        target = summary.get("targetControl") or {}
        print("🛰️ Official SPOC PRF forward modeling")
        print(f"   counterpart TIC: {counterpart.get('ticID')}")
        print(f"   counterpart Gaia DR3: {counterpart.get('gaiaDR3SourceID')}")
        print(
            f"   counterpart: independentSupport={candidate.get('independentSupportCount')}, "
            f"combinedPower={candidate.get('combinedPower')}, combinedPeriod={candidate.get('combinedPeriodDays')}"
        )
        print(
            f"   target control: independentSupport={target.get('independentSupportCount')}, "
            f"combinedPower={target.get('combinedPower')}, combinedPeriod={target.get('combinedPeriodDays')}"
        )
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "official-spoc-prf-forward-modeling"
            / "official-spoc-prf-forward-modeling-v20.18.json"
        )
        _write_json(artifact_path, summary)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.18"},
                triggered_by_stage_id=request.id,
            ),
            input_hashes={
                "preparation": sha256_json(preparation),
                "projectResult": sha256_json(run),
            },
            artifacts=(_artifact(artifact_path, "application/json"),),
        )

    def external_high_resolution_prepare_stage(investigation, request):
        prepared = _latest_result_for_handler(investigation, "openstar.tess.prepare-target")
        identity = _latest_result_for_handler(investigation, "openstar.tess.catalog-identity")
        offset_identity = _latest_result_for_handler(
            investigation, "openstar.tess.offset-source-identification.analyze"
        )
        official_spoc = _latest_result_for_handler(
            investigation, "openstar.tess.official-spoc-prf-forward-modeling.interpret"
        )
        if any(item is None for item in (prepared, identity, offset_identity, official_spoc)):
            raise RuntimeError(
                "v20.19 requires the frozen target, catalog identity, v20.13 offset-source identity, "
                "and completed v20.18 official SPOC PRF result."
            )
        if official_spoc.get("recommendedNextTest") != "EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION":
            raise RuntimeError(
                "v20.18 did not recommend EXTERNAL_HIGH_RESOLUTION_VARIABILITY_VALIDATION."
            )

        artifact_root = store.directory_for(investigation.id) / "artifacts"
        print("🔭 Preparing external high-resolution variability validation")
        print(f"   target TIC: {prepared.get('ticID')}")
        print("   archive: Gaia DR3 source-resolved epoch photometry")
        print("   the v20.9 TESS drift law is NOT extrapolated backward into the Gaia epoch")
        print("   any usable Gaia G-band source series becomes ordinary openstar.lomb-scargle.v1 work")

        spec = build_external_high_resolution_project(
            source_project_id=str(prepared["sourceProjectID"]),
            source_dataset_id=str(prepared["datasetID"]),
            identity=identity,
            offset_source_identification=offset_identity,
            official_spoc_prf_summary=official_spoc,
            output_dir=artifact_root,
            investigation_id=investigation.id,
        )

        pair = spec.get("sourcePair") or {}
        print(f"   target Gaia DR3: {pair.get('targetGaiaDR3SourceID')}")
        print(f"   counterpart Gaia DR3: {pair.get('counterpartGaiaDR3SourceID')}")
        print(f"   prepared Gaia datasets: {len(spec.get('preparedSeries') or [])}")
        print(f"   distributed work units: {spec.get('totalWorkUnits')}")

        artifacts: list[ArtifactReference] = []
        for item in spec.get("preparedSeries") or []:
            dataset_path = item.get("datasetPath")
            if dataset_path and Path(dataset_path).exists():
                artifacts.append(_artifact(Path(dataset_path), "application/json"))
            raw_path = item.get("rawEpochPath")
            if raw_path and Path(raw_path).exists():
                artifacts.append(_artifact(Path(raw_path), "text/csv"))

        project_path = spec.get("projectPath")
        if project_path and Path(project_path).exists():
            artifacts.append(_artifact(Path(project_path), "application/json"))

        if spec.get("available") and project_path:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "run-external-high-resolution-variability-validation"),
                handler_id="openstar.tess.external-high-resolution-variability-validation.run",
                parameters={"projectPath": project_path},
                triggered_by_stage_id=request.id,
            )
        else:
            next_stage = StageRequest(
                id=_next_stage_id(request.id, "interpret-external-high-resolution-variability-validation"),
                handler_id="openstar.tess.external-high-resolution-variability-validation.interpret",
                parameters={"distributedRunExpected": False},
                triggered_by_stage_id=request.id,
            )

        return StageOutcome(
            result=spec,
            next_stage=next_stage,
            input_hashes={
                "identity": sha256_json(identity),
                "offsetSourceIdentification": sha256_json(offset_identity),
                "officialSpocPrfForwardModeling": sha256_json(official_spoc),
            },
            artifacts=tuple(artifacts),
        )

    def external_high_resolution_run_stage(investigation, request):
        print("⚙️ Activating generic Gaia DR3 source-resolved Lomb-Scargle work")
        run = coordinator.run_project(
            request.parameters["projectPath"],
            poll_interval=poll_interval,
            timeout=timeout,
        )
        print("✅ Distributed external source-resolved search complete")
        print(f"   datasets: {len(run.status.get('datasets') or [])}")
        return StageOutcome(
            result=run.status,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "interpret-external-high-resolution-variability-validation"),
                handler_id="openstar.tess.external-high-resolution-variability-validation.interpret",
                parameters={"distributedRunExpected": True},
                triggered_by_stage_id=request.id,
            ),
            node_contributions=run.node_contributions,
            project_ids=(run.project_id,),
        )

    def external_high_resolution_interpret_stage(investigation, request):
        preparation = _latest_result_for_handler(
            investigation, "openstar.tess.external-high-resolution-variability-validation.prepare"
        )
        run = _latest_result_for_handler(
            investigation, "openstar.tess.external-high-resolution-variability-validation.run"
        )
        if preparation is None:
            raise RuntimeError("External high-resolution variability validation requires a prepare stage.")
        if bool(request.parameters.get("distributedRunExpected")) and run is None:
            raise RuntimeError(
                "External high-resolution variability validation expected a distributed run result."
            )

        summary = interpret_external_high_resolution_project(
            project_status=run,
            preparation=preparation,
        )
        pair = summary.get("sourcePair") or {}
        target = summary.get("targetControl") or {}
        counterpart = summary.get("catalogCounterpartEvidence") or {}
        print("🔭 External high-resolution variability validation")
        print(f"   archive: {summary.get('archive')}")
        print(f"   target Gaia DR3: {pair.get('targetGaiaDR3SourceID')}")
        print(f"   counterpart Gaia DR3: {pair.get('counterpartGaiaDR3SourceID')}")
        print(
            f"   target: accepted={target.get('acceptedResidualBandVariability')}, "
            f"period={target.get('candidatePeriodDays')}, power={target.get('candidatePower')}"
        )
        print(
            f"   counterpart: accepted={counterpart.get('acceptedResidualBandVariability')}, "
            f"period={counterpart.get('candidatePeriodDays')}, power={counterpart.get('candidatePower')}"
        )
        print(f"   classification: {summary.get('classification')}")
        print(f"   residual mode origin: {summary.get('residualModeOrigin')}")
        print(f"   recommended next test: {summary.get('recommendedNextTest')}")

        artifact_path = (
            store.directory_for(investigation.id)
            / "artifacts"
            / "external-high-resolution-variability"
            / "external-high-resolution-variability-v20.19.json"
        )
        _write_json(artifact_path, summary)
        input_hashes = {"preparation": sha256_json(preparation)}
        if run is not None:
            input_hashes["projectResult"] = sha256_json(run)
        return StageOutcome(
            result=summary,
            next_stage=StageRequest(
                id=_next_stage_id(request.id, "finalize"),
                handler_id="openstar.tess.finalize",
                parameters={"outputSuffix": "v20.19"},
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
        physical_interpretation = _latest_result_for_handler(
            investigation,
            "openstar.tess.physical.interpret",
        )
        source_localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.source-localization.analyze",
        )
        multimode_decomposition = _latest_result_for_handler(
            investigation,
            "openstar.tess.multimode.summarize",
        )
        time_frequency_evolution = _latest_result_for_handler(
            investigation,
            "openstar.tess.time-frequency.summarize",
        )
        nonstationary_modeling = _latest_result_for_handler(
            investigation,
            "openstar.tess.nonstationary.summarize",
        )
        residual_mode_localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization.interpret",
        )
        residual_mode_localization_review = _latest_result_for_handler(
            investigation,
            "openstar.tess.residual-mode-localization-review.interpret",
        )
        multisource_residual = _latest_result_for_handler(
            investigation,
            "openstar.tess.multi-source-residual.interpret",
        )
        offset_source_identification = _latest_result_for_handler(
            investigation,
            "openstar.tess.offset-source-identification.analyze",
        )
        offset_source_variability = _latest_result_for_handler(
            investigation,
            "openstar.tess.offset-source-variability.interpret",
        )
        calibrated_prf_deblending = _latest_result_for_handler(
            investigation,
            "openstar.tess.calibrated-prf-deblending.interpret",
        )
        difference_image_localization = _latest_result_for_handler(
            investigation,
            "openstar.tess.difference-image-localization.interpret",
        )
        frequency_localized_pixel_response = _latest_result_for_handler(
            investigation,
            "openstar.tess.frequency-localized-pixel-response.interpret",
        )
        official_spoc_prf_forward_modeling = _latest_result_for_handler(
            investigation,
            "openstar.tess.official-spoc-prf-forward-modeling.interpret",
        )
        external_high_resolution_validation = _latest_result_for_handler(
            investigation,
            "openstar.tess.external-high-resolution-variability-validation.interpret",
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

        if physical_interpretation is not None:
            preferred = physical_interpretation.get("preferredPhotometricHypothesis")
            next_test = physical_interpretation.get("recommendedNextTest")
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                (
                    "v20.5 physical-mechanism discrimination ranks the frozen multi-sector "
                    f"photometry with preferred photometric hypothesis {preferred or 'UNRESOLVED'}; "
                    "it does not treat morphology alone as a resolved physical mechanism."
                )
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {next_test}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if source_localization is not None:
            cross = source_localization.get("crossSector") or {}
            localization_class = cross.get("classification")
            next_test = source_localization.get("recommendedNextTest")
            existing_rationale = list(claim_decision.get("rationale") or [])
            if localization_class == "TARGET_SOURCE_SUPPORTED":
                existing_rationale.append(
                    "v20.6 pixel-level localization supports the TIC target as the origin of the periodic variability across a strict majority of eligible independent sectors; static aperture-flux contamination is still not claimed absent."
                )
            elif localization_class == "OFF_TARGET_VARIABLE_SOURCE_SUPPORTED":
                existing_rationale.append(
                    "v20.6 pixel-level localization places the periodic variability away from the TIC target across a strict majority of eligible independent sectors with a consistent sky offset."
                )
                claim_decision = {
                    "claim": "HUMAN_REVIEW_REQUIRED",
                    "rationale": existing_rationale,
                }
            else:
                existing_rationale.append(
                    "v20.6 pixel-level localization did not resolve the source association strongly enough to attribute the periodic variability to the TIC target or to an offset source."
                )
            existing_rationale.append(f"Recommended next confirmation step: {next_test}.")
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }
            validate_claim(claim_decision["claim"])

        if multimode_decomposition is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.7 iterative residual prewhitening classifies the post-family frequency structure as "
                f"{multimode_decomposition.get('classification')}; this does not by itself change the claim level or resolve the physical mechanism."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {multimode_decomposition.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if time_frequency_evolution is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            residual = time_frequency_evolution.get("residualEvolution") or {}
            family = time_frequency_evolution.get("familyEvolution") or {}
            existing_rationale.append(
                "v20.8 sliding-window time-frequency analysis classifies the residual evolution as "
                f"{residual.get('classification')} and the established-family evolution as "
                f"{family.get('classification')}; it does not by itself change the claim level or resolve the physical mechanism."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {time_frequency_evolution.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if nonstationary_modeling is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            existing_rationale.append(
                "v20.9 distributed long-baseline model comparison classifies the residual structure as "
                f"{nonstationary_modeling.get('classification')} using generic Lomb-Scargle work units over a deterministic frequency-drift grid; "
                "it does not by itself change the claim level or resolve the physical mechanism."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {nonstationary_modeling.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if residual_mode_localization is not None:
            cross = residual_mode_localization.get("crossSector") or {}
            localization_class = cross.get("classification")
            existing_rationale = list(claim_decision.get("rationale") or [])
            if localization_class == "RESIDUAL_MODE_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.10 distributed pixel localization supports the TIC target as the origin of the v20.9 drifting residual component across a strict majority of eligible independent sectors; this supports intrinsic multi-component variability but does not identify the physical mechanism."
                )
            elif localization_class == "RESIDUAL_MODE_OFF_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.10 distributed pixel localization places the drifting residual component away from the TIC target across a consistent strict majority of eligible independent sectors; this does not undo the v20.6 target localization of the main periodic family."
                )
            else:
                existing_rationale.append(
                    "v20.10 distributed pixel localization does not resolve whether the drifting residual component is target-centered or offset; the main periodic family's v20.6 target association remains unchanged."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {residual_mode_localization.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }


        if residual_mode_localization_review is not None:
            cross_time = residual_mode_localization_review.get("crossTime") or {}
            review_class = cross_time.get("classification")
            existing_rationale = list(claim_decision.get("rationale") or [])
            if review_class == "RESIDUAL_MODE_TARGET_SUPPORTED_TIME_RESOLVED":
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization supports the TIC target as the dominant origin of the drifting residual component across a strict majority of eligible independent sectors."
                )
            elif review_class == "RESIDUAL_MODE_OFF_TARGET_SUPPORTED_TIME_RESOLVED":
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization supports a consistent off-target origin for the drifting residual component; this remains separate from the v20.6 target association of the established main family."
                )
            elif review_class == "RESIDUAL_MODE_SOURCE_SWITCHING_OR_BLEND":
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization finds target-centered and offset residual behavior across time/sectors, supporting source switching or blended nonstationary variability rather than a single static residual centroid."
                )
            else:
                existing_rationale.append(
                    "v20.11 time-resolved distributed pixel localization still does not uniquely assign the drifting residual component to the TIC target or one consistent offset source."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {residual_mode_localization_review.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if multisource_residual is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = multisource_residual.get("classification")
            if classification == "MULTIPLE_RESIDUAL_SOURCES_SUPPORTED":
                existing_rationale.append(
                    "v20.12 spatial decomposition supports both target-centered and offset residual components, explaining the v20.11 source-switching/blend behavior without changing the established main-family target association."
                )
            elif classification == "TARGET_RESIDUAL_COMPONENT_DOMINANT":
                existing_rationale.append(
                    "v20.12 spatial decomposition finds the target-centered residual component dominant after separating the v20.11 blended/source-switching evidence."
                )
            elif classification == "OFF_TARGET_RESIDUAL_COMPONENT_DOMINANT":
                existing_rationale.append(
                    "v20.12 spatial decomposition finds an offset residual component dominant; this remains separate from the v20.6 target localization of the established main periodic family."
                )
            else:
                existing_rationale.append(
                    "v20.12 spatial decomposition does not yet separate the time-variable residual structure strongly enough to assign it to one or multiple sky components."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {multisource_residual.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if offset_source_identification is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = offset_source_identification.get("classification")
            best = offset_source_identification.get("bestCandidate") or {}
            ids = best.get("catalogIDs") or {}
            if classification == "KNOWN_VARIABLE_CATALOG_COUNTERPART_IDENTIFIED":
                existing_rationale.append(
                    "v20.13 catalog crossmatch identifies a spatially favored non-target counterpart to the dominant offset residual component and finds independent catalog evidence that the counterpart is variable; this still requires a direct variability-match test before attributing the TESS residual to that source."
                )
            elif classification == "CATALOG_COUNTERPART_IDENTIFIED":
                existing_rationale.append(
                    "v20.13 catalog crossmatch identifies a spatially favored non-target counterpart to the dominant offset residual component, but catalog position alone does not prove that source produces the TESS residual variability."
                )
            elif classification == "MULTIPLE_PLAUSIBLE_CATALOG_COUNTERPARTS":
                existing_rationale.append(
                    "v20.13 catalog crossmatch finds multiple plausible non-target counterparts within the offset-component localization uncertainty, so the residual source identity remains unresolved."
                )
            elif classification == "CATALOG_QUERY_INCOMPLETE":
                existing_rationale.append(
                    "v20.13 could not complete enough external catalog queries to identify the dominant offset residual component securely."
                )
            else:
                existing_rationale.append(
                    "v20.13 finds no secure catalog counterpart for the dominant offset residual component within the adopted localization uncertainty."
                )
            if ids.get("ticID") is not None or ids.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Best offset counterpart identifiers: TIC={ids.get('ticID')}, GaiaDR3={ids.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {offset_source_identification.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if offset_source_variability is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = offset_source_variability.get("classification")
            counterpart = offset_source_variability.get("catalogCounterpart") or {}
            if classification == "OFFSET_COUNTERPART_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.14 catalog-guided target/counterpart deblending supports the v20.13 non-target catalog counterpart as the dominant carrier of residual variability matching the v20.12 offset component; this does not alter the target association of the established main periodic family."
                )
            elif classification == "TARGET_AND_OFFSET_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.14 finds matching residual variability in both the catalog counterpart and the target-control component, supporting a blended two-source residual interpretation rather than assigning all residual structure to one object."
                )
            elif classification == "OFFSET_COUNTERPART_VARIABILITY_NOT_SUPPORTED":
                existing_rationale.append(
                    "v20.14 does not validate the v20.13 catalog counterpart as the carrier of the offset residual; the target-control component is better supported and calibrated PRF deblending remains necessary."
                )
            elif classification == "OFFSET_COUNTERPART_VARIABILITY_SUGGESTIVE":
                existing_rationale.append(
                    "v20.14 finds suggestive but insufficient independent-sector variability support for the v20.13 catalog counterpart, so the residual source attribution remains provisional."
                )
            else:
                existing_rationale.append(
                    "v20.14 catalog-guided variability validation remains unresolved and does not securely attribute the residual variability to the v20.13 counterpart or the target-control component."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Validated counterpart identifiers under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {offset_source_variability.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if calibrated_prf_deblending is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = calibrated_prf_deblending.get("classification")
            counterpart = calibrated_prf_deblending.get("catalogCounterpart") or {}
            if classification == "PRF_OFFSET_COUNTERPART_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.15 sector-calibrated empirical pixel-response deblending supports the v20.13 non-target catalog counterpart as the dominant carrier of residual variability matching the v20.12 offset component; the established main periodic family's v20.6 target association remains unchanged."
                )
            elif classification == "PRF_TARGET_AND_OFFSET_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.15 sector-calibrated pixel-response deblending supports residual variability in both the target-control and catalog-counterpart components, strengthening the blended two-source interpretation."
                )
            elif classification == "PRF_TARGET_CONTROL_DOMINANT":
                existing_rationale.append(
                    "v20.15 sector-calibrated pixel-response deblending favors the Blind C target-control component over the proposed offset counterpart for the residual family, so the intrinsic target residual must be modeled directly."
                )
            elif classification == "PRF_OFFSET_COUNTERPART_VARIABILITY_SUGGESTIVE":
                existing_rationale.append(
                    "v20.15 improves the spatial model beyond fixed Gaussian templates and still finds only suggestive offset-counterpart support; difference-image localization is required before source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.15 sector-calibrated pixel-response deblending remains unresolved between the target and offset counterpart; it does not change the established main-family source association."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Calibrated deblend counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {calibrated_prf_deblending.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if difference_image_localization is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = difference_image_localization.get("classification")
            counterpart = difference_image_localization.get("catalogCounterpart") or {}
            if classification == "DIFFERENCE_IMAGE_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.16 drift-corrected high-minus-low TESS difference images localize the residual variability to the v20.13 catalog counterpart across at least three independent sectors; this is an image-domain confirmation separate from the v20.14/v20.15 source-amplitude deblends and does not alter the established main-family target association."
                )
            elif classification == "DIFFERENCE_IMAGE_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.16 difference images localize the residual variability back to the Blind C target across at least three independent sectors, arguing that the residual is intrinsic after the established-family subtraction."
                )
            elif classification == "DIFFERENCE_IMAGE_MIXED_OR_BLENDED":
                existing_rationale.append(
                    "v20.16 difference images support both target-centered and counterpart-centered residual behavior in independent sectors, preserving a blended/time-variable two-source interpretation."
                )
            elif classification == "DIFFERENCE_IMAGE_COUNTERPART_SUGGESTIVE":
                existing_rationale.append(
                    "v20.16 difference images favor the catalog counterpart in multiple sectors but do not reach the independent-sector threshold required for secure source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.16 difference-image localization remains unresolved and therefore does not assign the drifting residual securely to either Blind C or the catalog counterpart."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Difference-image counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {difference_image_localization.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if frequency_localized_pixel_response is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = frequency_localized_pixel_response.get("classification")
            counterpart = frequency_localized_pixel_response.get("catalogCounterpart") or {}
            if classification == "FREQUENCY_LOCALIZED_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.17 narrow-band per-pixel Lomb-Scargle plus fixed-frequency phase-coherent response mapping localizes the residual variability to the catalog counterpart across at least three independent sectors; this source attribution remains separate from the v20.6 target association of the established main periodic family."
                )
            elif classification == "FREQUENCY_LOCALIZED_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.17 frequency-localized phase-coherent pixel response maps localize the residual variability back to Blind C across at least three independent sectors, supporting an intrinsic target residual after the established-family subtraction."
                )
            elif classification == "FREQUENCY_LOCALIZED_MIXED_OR_BLENDED":
                existing_rationale.append(
                    "v20.17 frequency-localized pixel response maps support both Blind C and the catalog counterpart in independent sectors, preserving a mixed or blended residual-source interpretation."
                )
            elif classification in {
                "FREQUENCY_LOCALIZED_COUNTERPART_SUGGESTIVE",
                "FREQUENCY_LOCALIZED_TARGET_SUGGESTIVE",
            }:
                existing_rationale.append(
                    "v20.17 frequency-localized pixel response mapping favors one source in multiple sectors but does not reach the independent-sector threshold required for secure residual-source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.17 frequency-localized pixel response confirmation remains unresolved and therefore does not securely assign the residual to Blind C or the catalog counterpart."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Frequency-localized counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {frequency_localized_pixel_response.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if official_spoc_prf_forward_modeling is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = official_spoc_prf_forward_modeling.get("classification")
            counterpart = official_spoc_prf_forward_modeling.get("catalogCounterpart") or {}
            if classification == "SPOC_PRF_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling independently separates the target and catalog counterpart and supports the counterpart residual across the required independent sectors; this attribution applies only to the residual component and does not alter the v20.6 target association of the established main periodic family."
                )
            elif classification == "SPOC_PRF_TARGET_SUPPORTED":
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling supports the residual component on Blind C across the required independent sectors after separating the catalog neighbor, favoring an intrinsic target residual."
                )
            elif classification == "SPOC_PRF_TARGET_AND_COUNTERPART_SUPPORTED":
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling supports residual variability in both Blind C and the catalog counterpart across independent sectors, preserving a blended/multi-source interpretation."
                )
            elif classification in {"SPOC_PRF_COUNTERPART_SUGGESTIVE", "SPOC_PRF_TARGET_SUGGESTIVE"}:
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling favors one source but still does not reach the independent-sector threshold required for secure residual-source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.18 official SPOC PRF forward modeling remains unable to securely assign the residual to Blind C or the catalog counterpart; the TESS pixel-scale source attribution is therefore treated as unresolved rather than forcing a source."
                )
            if counterpart.get("ticID") is not None or counterpart.get("gaiaDR3SourceID") is not None:
                existing_rationale.append(
                    f"Official-PRF counterpart under test: TIC={counterpart.get('ticID')}, GaiaDR3={counterpart.get('gaiaDR3SourceID')}."
                )
            existing_rationale.append(
                f"Recommended next confirmation step: {official_spoc_prf_forward_modeling.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
            }

        if external_high_resolution_validation is not None:
            existing_rationale = list(claim_decision.get("rationale") or [])
            classification = external_high_resolution_validation.get("classification")
            pair = external_high_resolution_validation.get("sourcePair") or {}
            if classification == "GAIA_DR3_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry supports residual-band variability on the catalog counterpart while the target control does not meet the same acceptance guard. This external attribution applies only to the residual component and does not alter the established main periodic family's v20.6 target association."
                )
            elif classification == "GAIA_DR3_TARGET_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry supports residual-band variability on Blind C while the catalog counterpart does not meet the same acceptance guard, favoring an intrinsic target residual."
                )
            elif classification == "GAIA_DR3_TARGET_AND_COUNTERPART_RESIDUAL_BAND_VARIABILITY_SUPPORTED":
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry supports residual-band variability in both Blind C and the catalog counterpart, preserving a multi-source interpretation."
                )
            elif classification == "GAIA_DR3_EPOCH_PHOTOMETRY_UNAVAILABLE":
                existing_rationale.append(
                    "v20.19 could not obtain usable Gaia DR3 epoch photometry for either frozen source, so the external archive provides no source-attribution evidence."
                )
            elif classification == "GAIA_DR3_SOURCE_RESOLVED_COVERAGE_INCOMPLETE":
                existing_rationale.append(
                    "v20.19 has source-resolved Gaia DR3 epoch coverage for only part of the frozen target/counterpart pair, so it does not force a residual-source attribution."
                )
            else:
                existing_rationale.append(
                    "v20.19 Gaia DR3 source-resolved epoch photometry does not securely distinguish the residual-band variability between Blind C and the catalog counterpart."
                )
            existing_rationale.append(
                "The v20.9 TESS frequency-drift model was not extrapolated backward into the Gaia observing epoch; Gaia was searched independently inside the frozen residual-frequency band."
            )
            existing_rationale.append(
                f"External source pair: target GaiaDR3={pair.get('targetGaiaDR3SourceID')}, counterpart GaiaDR3={pair.get('counterpartGaiaDR3SourceID')}."
            )
            existing_rationale.append(
                f"Recommended next confirmation step: {external_high_resolution_validation.get('recommendedNextTest')}."
            )
            claim_decision = {
                "claim": claim_decision["claim"],
                "rationale": existing_rationale,
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
        if source_localization is not None:
            cross = source_localization.get("crossSector") or {}
            period_evidence["sourceAssociation"] = cross.get("variableSignalOrigin")
            period_evidence["sourceAssociationClassification"] = cross.get("classification")
            period_evidence["targetAssociationResolved"] = cross.get("classification") in {
                "TARGET_SOURCE_SUPPORTED",
                "OFF_TARGET_VARIABLE_SOURCE_SUPPORTED",
            }

        if external_high_resolution_validation is not None:
            recommended_next_test = external_high_resolution_validation.get("recommendedNextTest")
        elif official_spoc_prf_forward_modeling is not None:
            recommended_next_test = official_spoc_prf_forward_modeling.get("recommendedNextTest")
        elif frequency_localized_pixel_response is not None:
            recommended_next_test = frequency_localized_pixel_response.get("recommendedNextTest")
        elif difference_image_localization is not None:
            recommended_next_test = difference_image_localization.get("recommendedNextTest")
        elif calibrated_prf_deblending is not None:
            recommended_next_test = calibrated_prf_deblending.get("recommendedNextTest")
        elif offset_source_variability is not None:
            recommended_next_test = offset_source_variability.get("recommendedNextTest")
        elif offset_source_identification is not None:
            recommended_next_test = offset_source_identification.get("recommendedNextTest")
        elif multisource_residual is not None:
            recommended_next_test = multisource_residual.get("recommendedNextTest")
        elif residual_mode_localization_review is not None:
            recommended_next_test = residual_mode_localization_review.get("recommendedNextTest")
        elif residual_mode_localization is not None:
            recommended_next_test = residual_mode_localization.get("recommendedNextTest")
        elif nonstationary_modeling is not None:
            recommended_next_test = nonstationary_modeling.get("recommendedNextTest")
        elif time_frequency_evolution is not None:
            recommended_next_test = time_frequency_evolution.get("recommendedNextTest")
        elif multimode_decomposition is not None:
            recommended_next_test = multimode_decomposition.get("recommendedNextTest")
        elif source_localization is not None:
            recommended_next_test = source_localization.get("recommendedNextTest")
        else:
            recommended_next_test = (physical_interpretation or {}).get("recommendedNextTest")

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
            "physicalInterpretation": physical_interpretation,
            "sourceLocalization": source_localization,
            "multiModeDecomposition": multimode_decomposition,
            "timeFrequencyEvolution": time_frequency_evolution,
            "nonstationaryModeling": nonstationary_modeling,
            "residualModeLocalization": residual_mode_localization,
            "residualModeLocalizationReview": residual_mode_localization_review,
            "multiSourceResidualDecomposition": multisource_residual,
            "offsetResidualSourceIdentification": offset_source_identification,
            "offsetSourceVariabilityValidation": offset_source_variability,
            "calibratedPrfSourceDeblending": calibrated_prf_deblending,
            "differenceImageSourceLocalization": difference_image_localization,
            "frequencyLocalizedPixelResponse": frequency_localized_pixel_response,
            "officialSpocPrfForwardModeling": official_spoc_prf_forward_modeling,
            "externalHighResolutionVariabilityValidation": external_high_resolution_validation,
            "recommendedNextTest": recommended_next_test,
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
        if physical_interpretation is not None:
            print(
                "   preferred photometric hypothesis: "
                f"{physical_interpretation.get('preferredPhotometricHypothesis')}"
            )
            print(
                "   physical mechanism resolved: "
                f"{physical_interpretation.get('physicalMechanismResolved')}"
            )
        if source_localization is not None:
            cross = source_localization.get("crossSector") or {}
            print(f"   source localization: {cross.get('classification')}")
            print(f"   variable signal origin: {cross.get('variableSignalOrigin')}")
        if nonstationary_modeling is not None:
            comparison = nonstationary_modeling.get("modelComparison") or {}
            print(f"   long-baseline residual model: {nonstationary_modeling.get('classification')}")
            print(f"   preferred temporal model: {comparison.get('bestModelID')}")
            print(f"   residual period at reference: {nonstationary_modeling.get('preferredPeriodAtReferenceDays')} days")
            print(f"   fractional frequency drift/day: {nonstationary_modeling.get('fractionalFrequencyDriftPerDay')}")
            if residual_mode_localization is None:
                print(f"   recommended next test: {nonstationary_modeling.get('recommendedNextTest')}")
        elif time_frequency_evolution is not None:
            residual = time_frequency_evolution.get("residualEvolution") or {}
            family = time_frequency_evolution.get("familyEvolution") or {}
            print(f"   time-frequency structure: {time_frequency_evolution.get('classification')}")
            print(f"   residual evolution: {residual.get('classification')}")
            print(f"   established-family evolution: {family.get('classification')}")
            print(f"   recommended next test: {time_frequency_evolution.get('recommendedNextTest')}")
        elif multimode_decomposition is not None:
            print(f"   residual frequency structure: {multimode_decomposition.get('classification')}")
            recurrent = multimode_decomposition.get("bestRecurrentSecondaryMode") or {}
            if recurrent:
                print(f"   recurrent secondary period: {recurrent.get('medianPeriodDays')} days")
            print(f"   recommended next test: {multimode_decomposition.get('recommendedNextTest')}")
        elif source_localization is not None:
            print(f"   recommended next test: {source_localization.get('recommendedNextTest')}")
        elif physical_interpretation is not None:
            print(
                "   recommended next test: "
                f"{physical_interpretation.get('recommendedNextTest')}"
            )
        if residual_mode_localization is not None:
            residual_cross = residual_mode_localization.get("crossSector") or {}
            print(f"   residual-mode localization: {residual_cross.get('classification')}")
            print(f"   residual-mode origin: {residual_cross.get('residualModeOrigin')}")
            if residual_mode_localization_review is None:
                print(f"   recommended next test: {residual_mode_localization.get('recommendedNextTest')}")
        if residual_mode_localization_review is not None:
            review_cross = residual_mode_localization_review.get("crossTime") or {}
            print(f"   time-resolved residual localization: {review_cross.get('classification')}")
            print(f"   time-resolved residual origin: {review_cross.get('residualModeOrigin')}")
            print(f"   source-switching sectors: {review_cross.get('sourceSwitchingSectors')}")
            if multisource_residual is None:
                print(f"   recommended next test: {residual_mode_localization_review.get('recommendedNextTest')}")
        if multisource_residual is not None:
            print(f"   multi-source residual decomposition: {multisource_residual.get('classification')}")
            print(f"   decomposed residual origin: {multisource_residual.get('residualModeOrigin')}")
            print(f"   target residual component: {multisource_residual.get('targetComponentID')}")
            print(f"   best offset residual component: {multisource_residual.get('bestOffsetComponentID')}")
            if offset_source_identification is None:
                print(f"   recommended next test: {multisource_residual.get('recommendedNextTest')}")
        if offset_source_identification is not None:
            best = offset_source_identification.get("bestCandidate") or {}
            ids = best.get("catalogIDs") or {}
            print(f"   offset source identification: {offset_source_identification.get('classification')}")
            print(f"   offset component: {(offset_source_identification.get('component') or {}).get('componentID')}")
            print(f"   matched TIC: {ids.get('ticID')}")
            print(f"   matched Gaia DR3: {ids.get('gaiaDR3SourceID')}")
            print(f"   counterpart separation: {best.get('separationArcsec')} arcsec")
            print(f"   known variable catalog evidence: {offset_source_identification.get('knownVariableCatalogEvidence')}")
            if offset_source_variability is None:
                print(f"   recommended next test: {offset_source_identification.get('recommendedNextTest')}")
        if offset_source_variability is not None:
            counterpart = offset_source_variability.get("catalogCounterpart") or {}
            candidate = offset_source_variability.get("catalogCounterpartEvidence") or {}
            target_control = offset_source_variability.get("targetControl") or {}
            print(f"   offset counterpart variability: {offset_source_variability.get('classification')}")
            print(f"   tested TIC: {counterpart.get('ticID')}")
            print(f"   tested Gaia DR3: {counterpart.get('gaiaDR3SourceID')}")
            print(f"   counterpart independent support: {candidate.get('independentSupportCount')}")
            print(f"   counterpart combined power: {candidate.get('combinedPower')}")
            print(f"   target-control independent support: {target_control.get('independentSupportCount')}")
            print(f"   target-control combined power: {target_control.get('combinedPower')}")
            if calibrated_prf_deblending is None:
                print(f"   recommended next test: {offset_source_variability.get('recommendedNextTest')}")
        if calibrated_prf_deblending is not None:
            counterpart = calibrated_prf_deblending.get("catalogCounterpart") or {}
            candidate = calibrated_prf_deblending.get("catalogCounterpartEvidence") or {}
            target_control = calibrated_prf_deblending.get("targetControl") or {}
            print(f"   calibrated ePRF deblending: {calibrated_prf_deblending.get('classification')}")
            print(f"   calibrated residual origin: {calibrated_prf_deblending.get('residualModeOrigin')}")
            print(f"   deblend backend: {calibrated_prf_deblending.get('deblendBackend')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart independent support: {candidate.get('independentSupportCount')}")
            print(f"   counterpart combined power: {candidate.get('combinedPower')}")
            print(f"   target-control independent support: {target_control.get('independentSupportCount')}")
            print(f"   target-control combined power: {target_control.get('combinedPower')}")
            if difference_image_localization is None:
                print(f"   recommended next test: {calibrated_prf_deblending.get('recommendedNextTest')}")
        if difference_image_localization is not None:
            counterpart = difference_image_localization.get("catalogCounterpart") or {}
            print(f"   difference-image localization: {difference_image_localization.get('classification')}")
            print(f"   difference-image residual origin: {difference_image_localization.get('residualModeOrigin')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart-supporting sectors: {difference_image_localization.get('counterpartSupportingSectors')}")
            print(f"   target-supporting sectors: {difference_image_localization.get('targetSupportingSectors')}")
            print(f"   ambiguous sectors: {difference_image_localization.get('ambiguousSectors')}")
            if frequency_localized_pixel_response is None:
                print(f"   recommended next test: {difference_image_localization.get('recommendedNextTest')}")
        if frequency_localized_pixel_response is not None:
            counterpart = frequency_localized_pixel_response.get("catalogCounterpart") or {}
            print(f"   frequency-localized pixel response: {frequency_localized_pixel_response.get('classification')}")
            print(f"   frequency-localized residual origin: {frequency_localized_pixel_response.get('residualModeOrigin')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart-supporting sectors: {frequency_localized_pixel_response.get('counterpartSupportingSectors')}")
            print(f"   target-supporting sectors: {frequency_localized_pixel_response.get('targetSupportingSectors')}")
            print(f"   ambiguous sectors: {frequency_localized_pixel_response.get('ambiguousSectors')}")
            if official_spoc_prf_forward_modeling is None:
                print(f"   recommended next test: {frequency_localized_pixel_response.get('recommendedNextTest')}")
        if official_spoc_prf_forward_modeling is not None:
            counterpart = official_spoc_prf_forward_modeling.get("catalogCounterpart") or {}
            candidate = official_spoc_prf_forward_modeling.get("catalogCounterpartEvidence") or {}
            target_control = official_spoc_prf_forward_modeling.get("targetControl") or {}
            print(f"   official SPOC PRF forward modeling: {official_spoc_prf_forward_modeling.get('classification')}")
            print(f"   official-PRF residual origin: {official_spoc_prf_forward_modeling.get('residualModeOrigin')}")
            print(f"   counterpart TIC: {counterpart.get('ticID')}")
            print(f"   counterpart independent support: {candidate.get('independentSupportCount')}")
            print(f"   counterpart combined power: {candidate.get('combinedPower')}")
            print(f"   target-control independent support: {target_control.get('independentSupportCount')}")
            print(f"   target-control combined power: {target_control.get('combinedPower')}")
            if external_high_resolution_validation is None:
                print(f"   recommended next test: {official_spoc_prf_forward_modeling.get('recommendedNextTest')}")
        if external_high_resolution_validation is not None:
            pair = external_high_resolution_validation.get("sourcePair") or {}
            target_external = external_high_resolution_validation.get("targetControl") or {}
            counterpart_external = external_high_resolution_validation.get("catalogCounterpartEvidence") or {}
            print(f"   external high-resolution validation: {external_high_resolution_validation.get('classification')}")
            print(f"   external residual origin: {external_high_resolution_validation.get('residualModeOrigin')}")
            print(f"   archive: {external_high_resolution_validation.get('archive')}")
            print(f"   target Gaia DR3: {pair.get('targetGaiaDR3SourceID')}")
            print(f"   counterpart Gaia DR3: {pair.get('counterpartGaiaDR3SourceID')}")
            print(f"   target residual-band accepted: {target_external.get('acceptedResidualBandVariability')}")
            print(f"   counterpart residual-band accepted: {counterpart_external.get('acceptedResidualBandVariability')}")
            print(f"   recommended next test: {external_high_resolution_validation.get('recommendedNextTest')}")
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
        "openstar.tess.physical.interpret",
        physical_interpretation_stage,
    )
    engine.register_handler(
        "openstar.tess.source-localization.analyze",
        source_localization_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.prepare",
        multimode_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.run",
        multimode_run_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.interpret",
        multimode_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.multimode.summarize",
        multimode_summary_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.prepare",
        time_frequency_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.run",
        time_frequency_run_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.interpret",
        time_frequency_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.time-frequency.summarize",
        time_frequency_summary_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.prepare",
        nonstationary_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.run",
        nonstationary_run_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.interpret",
        nonstationary_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.nonstationary.summarize",
        nonstationary_summary_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization.prepare",
        residual_mode_localization_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization.run",
        residual_mode_localization_run_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization.interpret",
        residual_mode_localization_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization-review.prepare",
        residual_mode_localization_review_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization-review.run",
        residual_mode_localization_review_run_stage,
    )
    engine.register_handler(
        "openstar.tess.residual-mode-localization-review.interpret",
        residual_mode_localization_review_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.multi-source-residual.prepare",
        multisource_residual_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.multi-source-residual.run",
        multisource_residual_run_stage,
    )
    engine.register_handler(
        "openstar.tess.multi-source-residual.interpret",
        multisource_residual_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.offset-source-identification.analyze",
        offset_source_identification_stage,
    )
    engine.register_handler(
        "openstar.tess.offset-source-variability.prepare",
        offset_source_variability_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.offset-source-variability.run",
        offset_source_variability_run_stage,
    )
    engine.register_handler(
        "openstar.tess.offset-source-variability.interpret",
        offset_source_variability_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.calibrated-prf-deblending.prepare",
        calibrated_prf_deblending_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.calibrated-prf-deblending.run",
        calibrated_prf_deblending_run_stage,
    )
    engine.register_handler(
        "openstar.tess.calibrated-prf-deblending.interpret",
        calibrated_prf_deblending_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.difference-image-localization.prepare",
        difference_image_localization_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.difference-image-localization.run",
        difference_image_localization_run_stage,
    )
    engine.register_handler(
        "openstar.tess.difference-image-localization.interpret",
        difference_image_localization_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.frequency-localized-pixel-response.prepare",
        frequency_localized_pixel_response_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.frequency-localized-pixel-response.run",
        frequency_localized_pixel_response_run_stage,
    )
    engine.register_handler(
        "openstar.tess.frequency-localized-pixel-response.interpret",
        frequency_localized_pixel_response_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.official-spoc-prf-forward-modeling.prepare",
        official_spoc_prf_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.official-spoc-prf-forward-modeling.run",
        official_spoc_prf_run_stage,
    )
    engine.register_handler(
        "openstar.tess.official-spoc-prf-forward-modeling.interpret",
        official_spoc_prf_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.external-high-resolution-variability-validation.prepare",
        external_high_resolution_prepare_stage,
    )
    engine.register_handler(
        "openstar.tess.external-high-resolution-variability-validation.run",
        external_high_resolution_run_stage,
    )
    engine.register_handler(
        "openstar.tess.external-high-resolution-variability-validation.interpret",
        external_high_resolution_interpret_stage,
    )
    engine.register_handler(
        "openstar.tess.period-semantics.reinterpret",
        period_semantics_stage,
    )
    engine.register_handler("openstar.tess.finalize", finalize_stage)
    return engine
