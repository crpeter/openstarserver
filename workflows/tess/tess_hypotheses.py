from __future__ import annotations

import math
import statistics
from typing import Any

from .tess_claims import ClaimDecision, decision


G = 6.67430e-11
M_SUN_KG = 1.98847e30
R_SUN_M = 6.957e8
SECONDS_PER_DAY = 86400.0

def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _period_match(observed_days: float, published_days: float) -> dict[str, Any]:
    candidates = (
        ("0.5x", observed_days * 0.5),
        ("1x", observed_days),
        ("2x", observed_days * 2.0),
    )
    relation, candidate = min(
        candidates,
        key=lambda item: abs(item[1] - published_days),
    )
    absolute_error = abs(candidate - published_days)
    relative_error = absolute_error / max(abs(published_days), 1e-12)
    return {
        "relation": relation,
        "observedEquivalentDays": candidate,
        "publishedPeriodDays": published_days,
        "absoluteErrorDays": absolute_error,
        "relativeError": relative_error,
        "matches": absolute_error <= max(0.02, 0.03 * published_days),
    }


def catalog_period_evidence(identity: dict[str, Any], observed_period: float) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    vsx = identity.get("vsx") or {}
    for match in vsx.get("matches") or []:
        period = _float(match.get("periodDays"))
        if period is None or period <= 0:
            continue
        comparison = _period_match(observed_period, period)
        comparison.update({
            "source": "AAVSO VSX",
            "name": match.get("name"),
            "classification": match.get("type"),
            "separationArcsec": match.get("separationArcsec"),
        })
        evidence.append(comparison)

    variability = identity.get("gaiaVariability") or {}
    for candidate in variability.get("periodCandidates") or []:
        period = _float(candidate.get("periodDays"))
        if period is None or period <= 0:
            continue
        comparison = _period_match(observed_period, period)
        comparison.update({
            "source": candidate.get("source") or "Gaia DR3",
            "field": candidate.get("field"),
            "classification": (variability.get("classification") or {}).get("class"),
        })
        evidence.append(comparison)

    evidence.sort(key=lambda item: (not item["matches"], item["relativeError"]))
    return evidence


def primary_period_days(primary: dict[str, Any]) -> float | None:
    """Select the period used for catalog interpretation everywhere."""
    period = _float(primary.get("preferredPhysicalPeriodDays"))
    if period is None:
        period = _float(primary.get("candidatePeriodDays"))
    return period if period is not None and period > 0 else None


def cycle_coverage(period_days: float | None, baseline_days: float | None) -> dict[str, Any]:
    period = _float(period_days)
    baseline = _float(baseline_days)
    cycles = None
    if period is not None and period > 0 and baseline is not None and baseline > 0:
        cycles = baseline / period

    if cycles is None:
        status = "unknown"
    elif cycles >= 2.0:
        status = "well-covered"
    elif cycles >= 1.5:
        status = "limited"
    else:
        status = "under-covered"

    return {
        "periodDays": period,
        "baselineDays": baseline,
        "observedCycles": cycles,
        "status": status,
        "minimumCyclesForCandidateSupport": 1.5,
        "minimumCyclesForStrongSupport": 2.0,
    }


def rotational_sanity(identity: dict[str, Any], period_days: float) -> dict[str, Any]:
    metadata = ((identity.get("tic") or {}).get("metadata") or {})
    radius_rsun = _float(metadata.get("radiusRsun"))
    mass_msun = _float(metadata.get("massMsun"))

    result: dict[str, Any] = {
        "evaluated": False,
        "periodDays": period_days,
        "radiusRsun": radius_rsun,
        "massMsun": mass_msun,
        "status": "unknown",
    }

    if radius_rsun is None or radius_rsun <= 0:
        result["reason"] = "TIC radius is required for the deterministic rotation sanity test."
        return result

    radius_m = radius_rsun * R_SUN_M
    period_seconds = period_days * SECONDS_PER_DAY
    equatorial_speed = 2.0 * math.pi * radius_m / period_seconds
    minimum_mass_kg = (equatorial_speed ** 2) * radius_m / G
    minimum_mass_msun = minimum_mass_kg / M_SUN_KG

    result.update({
        "equatorialSpeedKmS": equatorial_speed / 1000.0,
        "minimumMassForSubcriticalRotationMsun": minimum_mass_msun,
    })

    if mass_msun is None or mass_msun <= 0:
        result["reason"] = (
            "TIC mass is unavailable. OpenStar records the radius-only break-up "
            "constraint but does not convert it into a hard rotation rejection."
        )
        return result

    mass_kg = mass_msun * M_SUN_KG
    critical_speed = math.sqrt(G * mass_kg / radius_m)
    ratio = equatorial_speed / critical_speed

    if ratio >= 1.0:
        status = "ruled-out"
    elif ratio >= 0.7:
        status = "strongly-disfavored"
    else:
        status = "not-ruled-out"

    result.update({
        "evaluated": True,
        "criticalSpeedKmS": critical_speed / 1000.0,
        "equatorialToCriticalRatio": ratio,
        "status": status,
        "rule": "rotation is ruled out when v_eq >= sqrt(GM/R); strongly disfavored above 0.7 of critical speed",
    })
    return result

def catalog_coverage_complete(identity: dict[str, Any]) -> bool:
    """Return True only when the period-catalog path was actually queryable.

    Missing optional inventory/SIMBAD data does not by itself block period
    interpretation, but TIC coordinates plus VSX/Gaia query availability are
    required before OpenStar may interpret "no catalog match" as evidence.
    """
    if "catalogCoverageComplete" in identity:
        return identity["catalogCoverageComplete"] is True
    tic = identity.get("tic") or {}
    vsx = identity.get("vsx") or {}
    gaia = identity.get("gaiaDR3") or {}
    gaia_variability = identity.get("gaiaVariability") or {}

    if not tic.get("found"):
        return False
    if vsx.get("queryError"):
        return False
    if gaia.get("queryError"):
        return False
    if gaia_variability.get("queryError"):
        return False
    return True


def analyze(
    primary: dict[str, Any],
    identity: dict[str, Any],
    *,
    observation_baseline_days: float | None = None,
    primary_minimum_frequency: float | None = None,
    primary_maximum_frequency: float | None = None,
    primary_frequency_step: float | None = None,
) -> dict[str, Any]:
    period = primary_period_days(primary)

    status = str(primary.get("periodStatus") or "").upper()
    confidence = str(primary.get("periodConfidence") or "none").lower()
    reliable = status == "RELIABLE" and confidence in {"high", "medium"}

    catalog_evidence = catalog_period_evidence(identity, period) if period else []
    best_catalog_match = next((item for item in catalog_evidence if item["matches"]), None)
    rotation = rotational_sanity(identity, period) if period else {
        "evaluated": False,
        "status": "unknown",
        "reason": "No usable period.",
    }

    harmonic_candidates = primary.get("harmonicCandidates") or []
    preferred_relation = primary.get("preferredPhysicalPeriodRelation") or "1x"
    raw_candidate_period = _float(primary.get("candidatePeriodDays"))
    raw_candidate_frequency = _float(primary.get("candidateFrequency"))
    if (
        raw_candidate_frequency is None
        and raw_candidate_period is not None
        and raw_candidate_period > 0
    ):
        raw_candidate_frequency = 1.0 / raw_candidate_period
    minimum_primary_frequency = _float(primary_minimum_frequency)
    maximum_primary_frequency = _float(primary_maximum_frequency)
    primary_step = _float(primary_frequency_step)
    primary_boundary_hit = False
    primary_boundary = None
    if (
        raw_candidate_frequency is not None
        and minimum_primary_frequency is not None
        and maximum_primary_frequency is not None
        and maximum_primary_frequency > minimum_primary_frequency
    ):
        span = maximum_primary_frequency - minimum_primary_frequency
        guard = max((primary_step or 0.0) * 4.0, span * 0.002, 1e-12)
        if raw_candidate_frequency <= minimum_primary_frequency + guard:
            primary_boundary_hit = True
            primary_boundary = "minimum"
        elif raw_candidate_frequency >= maximum_primary_frequency - guard:
            primary_boundary_hit = True
            primary_boundary = "maximum"
    reliable = reliable and not primary_boundary_hit
    preferred_coverage = cycle_coverage(period, observation_baseline_days)
    raw_coverage = cycle_coverage(raw_candidate_period, observation_baseline_days)

    preferred_frequency = 1.0 / period if period is not None and period > 0 else None
    harmonic_low_frequency_applicable = (
        preferred_relation != "1x"
        and preferred_frequency is not None
        and minimum_primary_frequency is not None
        # The complete targeted window must be novel low-frequency territory;
        # trimming it at the primary boundary would change the scientific test.
        and preferred_frequency * 1.35 < minimum_primary_frequency
        and max(0.005, preferred_frequency * 0.65)
        < preferred_frequency * 1.35
    )

    hypotheses: list[dict[str, Any]] = []
    if best_catalog_match is not None:
        hypotheses.append({
            "id": "catalog-period",
            "status": "supported",
            "evidence": best_catalog_match,
        })
    else:
        hypotheses.append({
            "id": "catalog-period",
            "status": "not-found",
            "evidence": catalog_evidence[:5],
        })

    hypotheses.append({
        "id": "rotation-at-primary-period",
        "status": rotation["status"],
        "evidence": rotation,
    })

    if preferred_relation != "1x":
        hypotheses.append({
            "id": "harmonic-or-multiplicity",
            "status": "supported",
            "evidence": {
                "preferredRelation": preferred_relation,
                "harmonicCandidates": harmonic_candidates,
            },
        })
    else:
        hypotheses.append({
            "id": "harmonic-or-multiplicity",
            "status": "not-preferred",
            "evidence": {
                "preferredRelation": preferred_relation,
                "harmonicCandidates": harmonic_candidates,
            },
        })

    return {
        "primaryReliable": reliable,
        "primaryBoundaryHit": primary_boundary_hit,
        "primaryBoundary": primary_boundary,
        "periodStatus": status,
        "periodConfidence": confidence,
        "observedPeriodDays": period,
        "rawCandidatePeriodDays": raw_candidate_period,
        "observationBaselineDays": _float(observation_baseline_days),
        "preferredCycleCoverage": preferred_coverage,
        "rawCycleCoverage": raw_coverage,
        "bestCatalogMatch": best_catalog_match,
        "catalogPeriodEvidence": catalog_evidence,
        "catalogCoverageComplete": catalog_coverage_complete(identity),
        "rotationSanity": rotation,
        "preferredPhysicalPeriodRelation": preferred_relation,
        "harmonicLowFrequencyFollowup": {
            "applicable": harmonic_low_frequency_applicable,
            "preferredFrequency": preferred_frequency,
            "primaryMinimumFrequency": minimum_primary_frequency,
            "reason": (
                "target-window-is-inside-low-frequency-domain"
                if harmonic_low_frequency_applicable
                else "target-window-is-not-inside-low-frequency-domain"
            ),
        },
        "hypotheses": hypotheses,
    }


def plan(
    analysis: dict[str, Any],
    identity: dict[str, Any],
    investigation_goal: str | None = None,
) -> dict[str, Any]:
    period = _float(analysis.get("observedPeriodDays"))
    best_match = analysis.get("bestCatalogMatch")
    rotation = analysis.get("rotationSanity") or {}
    query_errors = identity.get("queryErrors") or []
    full_characterization = investigation_goal == "FULL_CHARACTERIZATION"

    if analysis.get("primaryBoundaryHit") is True:
        claim = decision(
            "HUMAN_REVIEW_REQUIRED",
            "The primary periodogram winner landed on the search-grid boundary and is not a reliable isolated period.",
            "A full-characterization investigation may use independent frozen sectors to test whether another signal class recurs.",
        )
        return {
            "action": (
                "INDEPENDENT_SECTOR_FOLLOWUP"
                if full_characterization and period is not None
                else "STOP"
            ),
            "claimDecision": claim.as_dict(),
            "reason": "primary-period-search-boundary",
            **(
                {"investigationGoal": investigation_goal}
                if full_characterization
                else {}
            ),
        }

    if best_match is not None:
        claim = decision(
            "KNOWN_PERIOD_RECOVERED",
            "OpenStar recovered a period consistent with an external catalog period or its 0.5x/2x harmonic relation.",
            f"Best catalog source: {best_match.get('source')}",
        )
        return {
            "action": (
                "INDEPENDENT_SECTOR_FOLLOWUP" if full_characterization else "STOP"
            ),
            "claimDecision": claim.as_dict(),
            "reason": "catalog-period-match",
            **(
                {"investigationGoal": investigation_goal}
                if full_characterization else {}
            ),
        }

    if not analysis.get("primaryReliable"):
        claim = decision(
            "CANDIDATE_PERIOD" if period is not None else "HUMAN_REVIEW_REQUIRED",
            "The primary distributed search did not meet the reliable-period threshold.",
        )
        return {
            "action": "STOP",
            "claimDecision": claim.as_dict(),
            "reason": "primary-period-not-reliable",
        }

    if not analysis.get("catalogCoverageComplete"):
        claim = decision(
            "CANDIDATE_PERIOD",
            "The primary period is usable for independent follow-up, but external catalog coverage is incomplete.",
            "No absence or uncataloged-period conclusion may be drawn from unavailable catalogs; their evidence gap remains unresolved.",
        )
        return {
            "action": "INDEPENDENT_SECTOR_FOLLOWUP",
            "claimDecision": claim.as_dict(),
            "reason": "catalog-coverage-incomplete",
            "queryErrors": list(query_errors),
            "unresolvedCatalogDependencies": list(
                identity.get("catalogsTransientlyUnavailable") or []),
        }

    if rotation.get("status") in {"ruled-out", "strongly-disfavored"}:
        return {
            "action": "LOW_FREQUENCY_FOLLOWUP",
            "reason": "primary-period-rotation-physically-disfavored",
            "claimDecision": None,
        }

    if analysis.get("preferredPhysicalPeriodRelation") != "1x":
        preferred_coverage = analysis.get("preferredCycleCoverage") or {}
        observed_cycles = _float(preferred_coverage.get("observedCycles"))
        if observed_cycles is not None and observed_cycles < 1.5:
            claim = decision(
                "CANDIDATE_PERIOD",
                "The coordinator preferred a harmonic/full-cycle interpretation, but the single-sector baseline covers fewer than 1.5 cycles of that preferred period.",
                "OpenStar will seek independent observing-sector recurrence instead of treating the under-covered harmonic preference as established.",
            )
            return {
                "action": "INDEPENDENT_SECTOR_FOLLOWUP",
                "reason": "harmonic-cycle-undercovered-needs-independent-sector",
                "claimDecision": claim.as_dict(),
            }
        if (analysis.get("harmonicLowFrequencyFollowup") or {}).get("applicable"):
            return {
                "action": "LOW_FREQUENCY_FOLLOWUP",
                "reason": "harmonic-cycle-preferred",
                "claimDecision": None,
            }
        claim = decision(
            "CANDIDATE_PERIOD",
            "The reliable primary result prefers a harmonic/full-cycle interpretation, but that cycle is already inside the primary search domain rather than the novel low-frequency domain.",
            "OpenStar will test recurrence in independent observing sectors instead of running an inapplicable same-sector low-frequency extension.",
        )
        return {
            "action": "INDEPENDENT_SECTOR_FOLLOWUP",
            "reason": "harmonic-cycle-needs-independent-sector",
            "claimDecision": claim.as_dict(),
        }

    claim = decision(
        "CANDIDATE_PERIOD",
        "The distributed single-sector period is reliable and no matching external catalog period was found in the queried sources.",
        "Independent observing-sector recurrence is required before OpenStar may upgrade this to an independent period estimate.",
    )
    return {
        "action": "INDEPENDENT_SECTOR_FOLLOWUP",
        "claimDecision": claim.as_dict(),
        "reason": "reliable-uncataloged-period-needs-independent-sector",
    }


def interpret_followup(
    primary_analysis: dict[str, Any],
    followup_result: dict[str, Any],
    followup_spec: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    datasets = followup_result.get("datasets") or []
    followup = datasets[0] if datasets else followup_result

    followup_status = str(followup.get("periodStatus") or "").upper()
    followup_confidence = str(followup.get("periodConfidence") or "none").lower()

    # A follow-up generated to test a harmonic/longer-period hypothesis must
    # interpret the actual Lomb-Scargle winner. Re-applying the coordinator's
    # preferredPhysicalPeriodDays here can recursively double an edge hit.
    candidate_period = _float(followup.get("candidatePeriodDays"))
    candidate_frequency = _float(followup.get("candidateFrequency"))
    if candidate_period is None:
        candidate_period = _float(followup.get("bestPeriodDays"))
    if candidate_frequency is None:
        candidate_frequency = _float(followup.get("bestFrequency"))

    reliable = (
        followup_status == "RELIABLE"
        and followup_confidence in {"high", "medium"}
        and candidate_period is not None
        and candidate_frequency is not None
    )

    search = (followup_spec or {}).get("frequencySearch") or {}
    minimum_frequency = _float(search.get("minimumFrequency"))
    maximum_frequency = _float(search.get("maximumFrequency"))
    frequency_step = _float(search.get("frequencyStep"))

    boundary_hit = False
    boundary = None
    if (
        candidate_frequency is not None
        and minimum_frequency is not None
        and maximum_frequency is not None
        and maximum_frequency > minimum_frequency
    ):
        span = maximum_frequency - minimum_frequency
        guard = max(
            (frequency_step or 0.0) * 4.0,
            span * 0.002,
            1e-12,
        )
        if candidate_frequency <= minimum_frequency + guard:
            boundary_hit = True
            boundary = "minimum"
        elif candidate_frequency >= maximum_frequency - guard:
            boundary_hit = True
            boundary = "maximum"

    target_period = _float((followup_spec or {}).get("targetPeriodDays"))
    target_relative_error = None
    target_consistent = None
    if target_period is not None and candidate_period is not None and target_period > 0:
        target_relative_error = abs(candidate_period - target_period) / target_period
        target_consistent = target_relative_error <= 0.25

    catalog_complete = True
    if identity is not None:
        catalog_complete = catalog_coverage_complete(identity)

    source_baseline_days = _float((followup_spec or {}).get("sourceBaselineDays"))
    coverage = cycle_coverage(candidate_period, source_baseline_days)

    diagnostics = {
        "candidateFrequency": candidate_frequency,
        "candidatePeriodDays": candidate_period,
        "searchMinimumFrequency": minimum_frequency,
        "searchMaximumFrequency": maximum_frequency,
        "boundaryHit": boundary_hit,
        "boundary": boundary,
        "targetPeriodDays": target_period,
        "targetRelativeError": target_relative_error,
        "targetConsistent": target_consistent,
        "catalogCoverageComplete": catalog_complete,
        "cycleCoverage": coverage,
    }

    if not catalog_complete:
        claim = decision(
            "HUMAN_REVIEW_REQUIRED",
            "The follow-up completed, but catalog coverage was incomplete; OpenStar cannot make an independent or uncataloged-period claim from this run.",
        )
        return {
            "claimDecision": claim.as_dict(),
            "selectedPeriodDays": primary_analysis.get("observedPeriodDays"),
            "selectedSource": "primary-unresolved",
            "followupReliable": reliable,
            "followupDataset": followup,
            "diagnostics": diagnostics,
        }

    if boundary_hit:
        claim = decision(
            "HUMAN_REVIEW_REQUIRED",
            "The follow-up periodogram winner landed on the search-grid boundary, so the follow-up did not isolate an interior period peak.",
            "A boundary winner must not be promoted through the 0.5x/2x harmonic heuristic.",
        )
        return {
            "claimDecision": claim.as_dict(),
            "selectedPeriodDays": primary_analysis.get("observedPeriodDays"),
            "selectedSource": "primary-unresolved",
            "followupReliable": False,
            "followupDataset": followup,
            "diagnostics": diagnostics,
        }

    if (
        reliable
        and target_consistent is not False
        and (coverage.get("observedCycles") or 0.0) >= 1.5
    ):
        # This is the same frozen light curve searched over a decisive new grid.
        # It is supporting evidence, not an independent dataset.
        claim: ClaimDecision = decision(
            "CANDIDATE_PERIOD",
            "A deterministic same-dataset follow-up produced an interior reliable peak consistent with the follow-up hypothesis.",
            "Because the follow-up reuses the same frozen light curve, it is not labeled an independent period estimate.",
        )
        return {
            "claimDecision": claim.as_dict(),
            "selectedPeriodDays": candidate_period,
            "selectedSource": "same-dataset-followup",
            "followupReliable": True,
            "followupDataset": followup,
            "diagnostics": diagnostics,
        }

    claim = decision(
        "HUMAN_REVIEW_REQUIRED",
        "The primary interpretation required a decisive lower-frequency follow-up, but that follow-up did not produce a trustworthy, sufficiently cycle-covered interior peak consistent with the tested hypothesis.",
    )
    return {
        "claimDecision": claim.as_dict(),
        "selectedPeriodDays": primary_analysis.get("observedPeriodDays"),
        "selectedSource": "primary-unresolved",
        "followupReliable": False,
        "followupDataset": followup,
        "diagnostics": diagnostics,
    }


def interpret_independent_sectors(
    *,
    target_period_days: float,
    project_status: dict[str, Any],
    independent_spec: dict[str, Any],
) -> dict[str, Any]:
    target_period = _float(target_period_days)
    if target_period is None or target_period <= 0:
        raise ValueError("A positive target period is required for independent-sector interpretation.")

    search = independent_spec.get("frequencySearch") or {}
    minimum_frequency = _float(search.get("minimumFrequency"))
    maximum_frequency = _float(search.get("maximumFrequency"))
    frequency_step = _float(search.get("frequencyStep"))
    prepared = {
        str(item.get("datasetID")): item
        for item in (independent_spec.get("preparedSectors") or [])
    }

    sector_results: list[dict[str, Any]] = []
    eligible_count = 0
    support_count = 0
    resolution_limited_count = 0
    supported_periods: list[float] = []

    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        spec = prepared.get(dataset_id) or {}
        sector = spec.get("sector") or dataset.get("sector")
        baseline = _float(spec.get("baselineDays"))
        candidate_period = _float(dataset.get("candidatePeriodDays"))
        candidate_frequency = _float(dataset.get("candidateFrequency"))
        frequency_interval = dataset.get("candidateFrequencyConfidenceInterval") or {}
        frequency_interval_lower = _float(frequency_interval.get("lower"))
        frequency_interval_upper = _float(frequency_interval.get("upper"))
        status = str(dataset.get("periodStatus") or "").upper()
        confidence = str(dataset.get("periodConfidence") or "none").lower()
        coverage = cycle_coverage(candidate_period, baseline)

        boundary_hit = False
        boundary = None
        if (
            candidate_frequency is not None
            and minimum_frequency is not None
            and maximum_frequency is not None
            and maximum_frequency > minimum_frequency
        ):
            span = maximum_frequency - minimum_frequency
            guard = max((frequency_step or 0.0) * 4.0, span * 0.002, 1e-12)
            if candidate_frequency <= minimum_frequency + guard:
                boundary_hit = True
                boundary = "minimum"
            elif candidate_frequency >= maximum_frequency - guard:
                boundary_hit = True
                boundary = "maximum"

        relative_error = None
        target_frequency = 1.0 / target_period
        frequency_relative_error = None
        if candidate_period is not None:
            relative_error = abs(candidate_period - target_period) / target_period
        if candidate_frequency is not None:
            frequency_relative_error = (
                abs(candidate_frequency - target_frequency) / target_frequency
            )

        rayleigh_resolution = (
            1.0 / baseline
            if baseline is not None and baseline > 0
            else None
        )
        frequency_separation = (
            abs(candidate_frequency - target_frequency)
            if candidate_frequency is not None
            else None
        )
        resolution_limited = (
            frequency_separation is not None
            and rayleigh_resolution is not None
            and frequency_separation < rayleigh_resolution
        )

        # The grid step describes numerical sampling, not peak uncertainty.
        # A targeted search can confirm recurrence only when its result carries
        # a statistically estimated frequency confidence interval.  In its
        # absence, the Rayleigh scale can show that two peaks are resolvably
        # different, but cannot turn an unresolved nearby peak into affirmative
        # evidence.
        has_frequency_interval = (
            frequency_interval_lower is not None
            and frequency_interval_upper is not None
            and frequency_interval_lower <= frequency_interval_upper
        )
        interval_contains_target = (
            has_frequency_interval
            and frequency_interval_lower <= target_frequency <= frequency_interval_upper
        )
        frequency_interval_width = (
            frequency_interval_upper - frequency_interval_lower
            if has_frequency_interval
            else None
        )
        frequency_interval_resolved = (
            frequency_interval_width is not None
            and rayleigh_resolution is not None
            and frequency_interval_width <= rayleigh_resolution
        )

        coverage_ok = (coverage.get("observedCycles") or 0.0) >= 1.5
        reliable = status == "RELIABLE" and confidence in {"high", "medium"}
        eligible = coverage_ok and not boundary_hit
        supports = (
            eligible
            and reliable
            and interval_contains_target
            and frequency_interval_resolved
        )
        inconclusive = (
            eligible
            and reliable
            and (
                (not has_frequency_interval and resolution_limited)
                or (resolution_limited and not interval_contains_target)
                or (
                    has_frequency_interval
                    and interval_contains_target
                    and not frequency_interval_resolved
                )
            )
        )
        if supports:
            recurrence_classification = "SUPPORTING"
        elif inconclusive:
            recurrence_classification = "RESOLUTION_LIMITED"
        else:
            recurrence_classification = "NONSUPPORTING"

        if eligible:
            eligible_count += 1
        if supports and candidate_period is not None:
            support_count += 1
            supported_periods.append(candidate_period)
        if inconclusive:
            resolution_limited_count += 1

        sector_results.append({
            "sector": sector,
            "datasetID": dataset_id,
            "periodStatus": status,
            "periodConfidence": confidence,
            "candidatePeriodDays": candidate_period,
            "candidateFrequency": candidate_frequency,
            "targetRelativeError": relative_error,
            "targetFrequency": target_frequency,
            "targetFrequencyRelativeError": frequency_relative_error,
            "frequencySeparation": frequency_separation,
            "rayleighFrequencyResolution": rayleigh_resolution,
            "resolutionLimited": resolution_limited,
            "candidateFrequencyConfidenceInterval": (
                {
                    "lower": frequency_interval_lower,
                    "upper": frequency_interval_upper,
                    "method": frequency_interval.get("method"),
                    "confidenceLevel": frequency_interval.get("confidenceLevel"),
                }
                if has_frequency_interval
                else None
            ),
            "frequencyIntervalContainsTarget": interval_contains_target,
            "frequencyIntervalWidth": frequency_interval_width,
            "frequencyIntervalResolved": frequency_interval_resolved,
            "harmonicOrAliasAccepted": False,
            "boundaryHit": boundary_hit,
            "boundary": boundary,
            "cycleCoverage": coverage,
            "eligibleForRecurrence": eligible,
            "supportsTarget": supports,
            "recurrenceClassification": recurrence_classification,
        })

    required_support = (eligible_count // 2 + 1) if eligible_count else 0

    if eligible_count > 0 and support_count >= required_support:
        selected = float(statistics.median(supported_periods))
        claim = decision(
            "INDEPENDENT_PERIOD_ESTIMATE",
            "The candidate period recurred in independent TESS observing sector data that were not used to produce the original single-sector estimate.",
            f"Supporting independent sectors: {support_count}/{eligible_count} eligible sectors; required support: {required_support}.",
        )
    elif support_count > 0:
        selected = float(statistics.median(supported_periods))
        claim = decision(
            "CANDIDATE_PERIOD",
            "At least one independent TESS sector supported the candidate, but recurrence was not strong enough across the eligible independent sectors to upgrade the claim.",
            f"Supporting independent sectors: {support_count}/{eligible_count} eligible sectors; required support: {required_support}.",
        )
    elif resolution_limited_count > 0:
        selected = target_period
        claim = decision(
            "HUMAN_REVIEW_REQUIRED",
            "Independent TESS sector verification recovered a nearby reliable peak, but the available baseline could not resolve it from the target frequency and no statistical frequency confidence interval was available.",
            "Resolution-limited sectors are inconclusive and do not count as affirmative recurrence evidence.",
        )
    else:
        selected = target_period
        claim = decision(
            "HUMAN_REVIEW_REQUIRED",
            "Independent TESS sector verification did not recover sufficient recurrence of the candidate period under the deterministic coverage and boundary rules.",
        )

    return {
        "claimDecision": claim.as_dict(),
        "selectedPeriodDays": selected,
        "selectedSource": "independent-tess-sectors" if support_count else "same-dataset-candidate-unconfirmed",
        "targetPeriodDays": target_period,
        "eligibleSectorCount": eligible_count,
        "supportingSectorCount": support_count,
        "resolutionLimitedSectorCount": resolution_limited_count,
        "requiredSupportingSectorCount": required_support,
        "sectorResults": sector_results,
    }



BROAD_CLUSTER_RELATIVE_TOLERANCE = 0.12
BROAD_MAX_PROMOTION_CLUSTER_SPAN_RELATIVE = 0.08
BROAD_MIN_PROMOTION_SECTORS = 3
BROAD_MIN_PROMOTION_PEAK_PROMINENCE = 1.5
BROAD_CANDIDATE_MIN_SECTORS = 2


def broad_independent_next_handler(interpreted: dict[str, Any]) -> str:
    """Select characterization only when persisted broad evidence warrants it."""

    characterization = interpreted.get("variabilityCharacterization") or {}
    if characterization.get("warranted") is True:
        return str(characterization["nextHandler"])
    return "openstar.tess.finalize"


HARMONIC_CONTEXT_TOLERANCE_RELATIVE = 0.15


def plan_independent_contradiction_resolution(
    targeted_interpretation: dict[str, Any],
) -> dict[str, Any]:
    """
    Decide whether a failed targeted independent check has enough usable
    independent evidence to justify one target-independent broad search before
    asking for human review.
    """
    claim = ((targeted_interpretation.get("claimDecision") or {}).get("claim"))
    sector_results = targeted_interpretation.get("sectorResults") or []
    reliable_count = sum(
        1
        for item in sector_results
        if str(item.get("periodStatus") or "").upper() == "RELIABLE"
        and str(item.get("periodConfidence") or "none").lower() in {"high", "medium"}
        and _float(item.get("candidatePeriodDays")) is not None
    )
    boundary_count = sum(1 for item in sector_results if item.get("boundaryHit"))

    if claim == "INDEPENDENT_PERIOD_ESTIMATE":
        return {
            "action": "STOP",
            "reason": "targeted-independent-recurrence-confirmed",
            "reliableSectorCount": reliable_count,
            "boundaryHitCount": boundary_count,
        }

    if reliable_count >= 2:
        return {
            "action": "BROAD_INDEPENDENT_SEARCH",
            "reason": (
                "targeted-candidate-not-recurrent-independent-sectors-contain-"
                "alternate-reliable-structure"
            ),
            "reliableSectorCount": reliable_count,
            "boundaryHitCount": boundary_count,
        }

    return {
        "action": "STOP",
        "reason": "insufficient-independent-evidence-for-broad-contradiction-search",
        "reliableSectorCount": reliable_count,
        "boundaryHitCount": boundary_count,
    }


def _cluster_periods(
    sector_results: list[dict[str, Any]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    eligible = [
        item
        for item in sector_results
        if item.get("eligibleForClustering")
        and _float(item.get("candidatePeriodDays")) is not None
    ]
    if not eligible:
        return []

    candidate_groups: dict[tuple[int, ...], dict[str, Any]] = {}
    for seed in eligible:
        seed_period = float(seed["candidatePeriodDays"])
        members = [
            item
            for item in eligible
            if abs(float(item["candidatePeriodDays"]) - seed_period)
            / max(seed_period, 1e-12)
            <= tolerance
        ]
        if not members:
            continue
        median = float(
            statistics.median(
                float(item["candidatePeriodDays"])
                for item in members
            )
        )
        members = [
            item
            for item in eligible
            if abs(float(item["candidatePeriodDays"]) - median)
            / max(median, 1e-12)
            <= tolerance
        ]
        member_sectors = tuple(
            sorted(
                int(item["sector"])
                for item in members
                if item.get("sector") is not None
            )
        )
        if not member_sectors:
            continue
        periods = [float(item["candidatePeriodDays"]) for item in members]
        median = float(statistics.median(periods))
        mean_abs_relative_deviation = sum(
            abs(period - median) / max(median, 1e-12)
            for period in periods
        ) / len(periods)
        relative_span = (
            (max(periods) - min(periods)) / max(median, 1e-12)
            if len(periods) > 1
            else 0.0
        )
        prominences = [
            _float(item.get("candidatePeakProminenceRatio"))
            for item in members
        ]
        finite_prominences = [value for value in prominences if value is not None]
        minimum_prominence = min(finite_prominences) if finite_prominences else None
        strong_prominence_count = sum(
            1
            for value in prominences
            if value is not None
            and value >= BROAD_MIN_PROMOTION_PEAK_PROMINENCE
        )
        candidate_groups[member_sectors] = {
            "sectors": list(member_sectors),
            "count": len(members),
            "medianPeriodDays": median,
            "minimumPeriodDays": min(periods),
            "maximumPeriodDays": max(periods),
            "meanAbsoluteRelativeDeviation": mean_abs_relative_deviation,
            "relativeSpan": relative_span,
            "minimumPeakProminenceRatio": minimum_prominence,
            "strongPeakProminenceCount": strong_prominence_count,
            "allPeaksMeetPromotionProminence": (
                len(members) > 0
                and strong_prominence_count == len(members)
            ),
        }

    return sorted(
        candidate_groups.values(),
        key=lambda item: (
            -int(item["count"]),
            float(item["relativeSpan"]),
            float(item["meanAbsoluteRelativeDeviation"]),
            float(item["medianPeriodDays"]),
        ),
    )


def _relative_error(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference <= 0:
        return None
    return abs(value - reference) / reference


def _harmonic_family(
    best_cluster: dict[str, Any] | None,
    *,
    primary_raw_period_days: float | None,
    primary_preferred_period_days: float | None,
    same_sector_candidate_days: float | None,
) -> dict[str, Any] | None:
    if not best_cluster or int(best_cluster.get("count") or 0) < BROAD_CANDIDATE_MIN_SECTORS:
        return None

    raw_period = _float(best_cluster.get("medianPeriodDays"))
    if raw_period is None or raw_period <= 0:
        return None
    double_period = raw_period * 2.0

    raw_primary_error = _relative_error(raw_period, _float(primary_raw_period_days))
    double_primary_error = _relative_error(
        double_period,
        _float(primary_preferred_period_days),
    )
    double_followup_error = _relative_error(
        double_period,
        _float(same_sector_candidate_days),
    )

    context_matches = {
        "rawVsPrimaryRaw": (
            raw_primary_error is not None
            and raw_primary_error <= HARMONIC_CONTEXT_TOLERANCE_RELATIVE
        ),
        "doubleVsPrimaryPreferred": (
            double_primary_error is not None
            and double_primary_error <= HARMONIC_CONTEXT_TOLERANCE_RELATIVE
        ),
        "doubleVsSameSectorCandidate": (
            double_followup_error is not None
            and double_followup_error <= HARMONIC_CONTEXT_TOLERANCE_RELATIVE
        ),
    }

    possible_double_wave = any(context_matches.values())
    return {
        "interpretation": (
            "possible-double-wave-period-family"
            if possible_double_wave
            else "recurrent-raw-period-family"
        ),
        "supportingSectors": list(best_cluster.get("sectors") or []),
        "supportingSectorCount": int(best_cluster.get("count") or 0),
        "representativeRawPeriodDays": raw_period,
        "possibleDoubleCycleDays": double_period,
        "rawClusterRelativeSpan": _float(best_cluster.get("relativeSpan")),
        "minimumPeakProminenceRatio": _float(
            best_cluster.get("minimumPeakProminenceRatio")
        ),
        "context": {
            "primaryRawPeriodDays": _float(primary_raw_period_days),
            "primaryPreferredPeriodDays": _float(primary_preferred_period_days),
            "sameSectorCandidateDays": _float(same_sector_candidate_days),
            "rawVsPrimaryRawRelativeError": raw_primary_error,
            "doubleVsPrimaryPreferredRelativeError": double_primary_error,
            "doubleVsSameSectorCandidateRelativeError": double_followup_error,
            "matches": context_matches,
        },
        "physicalCycleResolved": False,
    }


def interpret_broad_independent_sectors(
    *,
    project_status: dict[str, Any],
    broad_spec: dict[str, Any],
    primary_raw_period_days: float | None = None,
    primary_preferred_period_days: float | None = None,
    same_sector_candidate_days: float | None = None,
) -> dict[str, Any]:
    """
    Let independent sectors discover their own recurring raw period without
    using the original Sector-62 candidate to choose the numeric winner.

    v20.3.1 separates two questions:
      1. Is there enough independent evidence for a promoted numeric period?
      2. Is there a recurrent raw/double-wave period family worth retaining as
         candidate evidence even when promotion is not justified?
    """
    search = broad_spec.get("frequencySearch") or {}
    minimum_frequency = _float(search.get("minimumFrequency"))
    maximum_frequency = _float(search.get("maximumFrequency"))
    frequency_step = _float(search.get("frequencyStep"))
    prepared = {
        str(item.get("datasetID")): item
        for item in (broad_spec.get("preparedSectors") or [])
    }

    sector_results: list[dict[str, Any]] = []
    eligible_count = 0

    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        spec = prepared.get(dataset_id) or {}
        sector = spec.get("sector") or dataset.get("sector")
        baseline = _float(spec.get("baselineDays"))
        candidate_period = _float(dataset.get("candidatePeriodDays"))
        candidate_frequency = _float(dataset.get("candidateFrequency"))
        candidate_prominence = _float(
            dataset.get("candidatePeakProminenceRatio")
        )
        status = str(dataset.get("periodStatus") or "").upper()
        confidence = str(dataset.get("periodConfidence") or "none").lower()
        coverage = cycle_coverage(candidate_period, baseline)

        boundary_hit = False
        boundary = None
        if (
            candidate_frequency is not None
            and minimum_frequency is not None
            and maximum_frequency is not None
            and maximum_frequency > minimum_frequency
        ):
            span = maximum_frequency - minimum_frequency
            guard = max((frequency_step or 0.0) * 4.0, span * 0.002, 1e-12)
            if candidate_frequency <= minimum_frequency + guard:
                boundary_hit = True
                boundary = "minimum"
            elif candidate_frequency >= maximum_frequency - guard:
                boundary_hit = True
                boundary = "maximum"

        coverage_ok = (coverage.get("observedCycles") or 0.0) >= 1.5
        reliable = status == "RELIABLE" and confidence in {"high", "medium"}
        eligible = (
            reliable
            and coverage_ok
            and not boundary_hit
            and candidate_period is not None
        )
        if eligible:
            eligible_count += 1

        sector_results.append({
            "sector": sector,
            "datasetID": dataset_id,
            "periodStatus": status,
            "periodConfidence": confidence,
            "candidatePeriodDays": candidate_period,
            "candidateFrequency": candidate_frequency,
            "candidatePeakProminenceRatio": candidate_prominence,
            "boundaryHit": boundary_hit,
            "boundary": boundary,
            "cycleCoverage": coverage,
            "eligibleForClustering": eligible,
        })

    clusters = _cluster_periods(
        sector_results,
        tolerance=BROAD_CLUSTER_RELATIVE_TOLERANCE,
    )
    best_cluster = clusters[0] if clusters else None
    cluster_count = int(best_cluster["count"]) if best_cluster else 0
    strict_majority = (eligible_count // 2 + 1) if eligible_count else 0
    required_promotion_support = (
        max(BROAD_MIN_PROMOTION_SECTORS, strict_majority)
        if eligible_count
        else BROAD_MIN_PROMOTION_SECTORS
    )

    cluster_span = (
        _float(best_cluster.get("relativeSpan"))
        if best_cluster
        else None
    )
    span_ok = (
        cluster_span is not None
        and cluster_span <= BROAD_MAX_PROMOTION_CLUSTER_SPAN_RELATIVE
    )
    prominence_ok = bool(
        best_cluster
        and best_cluster.get("allPeaksMeetPromotionProminence")
    )
    support_ok = (
        eligible_count >= BROAD_MIN_PROMOTION_SECTORS
        and cluster_count >= required_promotion_support
    )
    promotion_eligible = support_ok and span_ok and prominence_ok

    harmonic_family = _harmonic_family(
        best_cluster,
        primary_raw_period_days=primary_raw_period_days,
        primary_preferred_period_days=primary_preferred_period_days,
        same_sector_candidate_days=same_sector_candidate_days,
    )

    promotion_blockers: list[str] = []
    if not support_ok:
        promotion_blockers.append("insufficient-independent-sector-support")
    if best_cluster and not span_ok:
        promotion_blockers.append("cluster-spread-too-wide")
    if best_cluster and not prominence_ok:
        promotion_blockers.append("supporting-peak-prominence-too-weak")

    if promotion_eligible and best_cluster is not None:
        selected = float(best_cluster["medianPeriodDays"])
        claim = decision(
            "INDEPENDENT_PERIOD_ESTIMATE",
            "A target-independent broad search found a tight recurring raw period cluster across at least three independent TESS sectors and a strict majority of eligible sectors.",
            (
                f"Cluster support: {cluster_count}/{eligible_count}; "
                f"required support: {required_promotion_support}; "
                f"relative span: {cluster_span:.4f}; "
                f"minimum peak prominence: {best_cluster.get('minimumPeakProminenceRatio')}."
            ),
        )
        source = "independent-broad-sector-cluster"
    elif cluster_count >= BROAD_CANDIDATE_MIN_SECTORS and best_cluster is not None:
        selected = float(best_cluster["medianPeriodDays"])
        family_reason = ""
        if harmonic_family is not None:
            family_reason = (
                f" The recurrent raw family is centered near "
                f"{harmonic_family['representativeRawPeriodDays']:.6f} days; "
                f"a possible 2x cycle is near "
                f"{harmonic_family['possibleDoubleCycleDays']:.6f} days."
            )
        claim = decision(
            "CANDIDATE_PERIOD",
            "Multiple independent TESS sectors show a recurrent raw periodicity family, but the evidence does not satisfy the stricter independent-period promotion rules." + family_reason,
            "The exact physical cycle remains unresolved; raw and possible double-wave interpretations are retained separately.",
        )
        source = "independent-broad-harmonic-family-candidate"
    else:
        selected = None
        claim = decision(
            "HUMAN_REVIEW_REQUIRED",
            "A target-independent broad search did not reveal a stable recurrent raw period family across multiple eligible independent TESS sectors.",
        )
        source = "independent-broad-no-stable-cluster"

    candidate_characterization_reasons = [
        blocker
        for blocker in promotion_blockers
        if blocker in {
            "insufficient-independent-sector-support",
            "cluster-spread-too-wide",
            "supporting-peak-prominence-too-weak",
        }
    ]
    candidate_characterization_warranted = bool(
        claim.claim == "CANDIDATE_PERIOD"
        and harmonic_family is not None
        and cluster_count >= BROAD_CANDIDATE_MIN_SECTORS
        and candidate_characterization_reasons
    )
    independent_physical_cycle_unresolved = bool(
        claim.claim == "INDEPENDENT_PERIOD_ESTIMATE"
        and promotion_eligible is True
        and harmonic_family is not None
        and (_float(harmonic_family.get("representativeRawPeriodDays")) or 0.0) > 0.0
        and (_float(harmonic_family.get("possibleDoubleCycleDays")) or 0.0) > 0.0
        and harmonic_family.get("physicalCycleResolved") is not True
    )
    characterization_warranted = (
        candidate_characterization_warranted
        or independent_physical_cycle_unresolved
    )
    if independent_physical_cycle_unresolved:
        characterization_state = "INDEPENDENT_PERIOD_PHYSICAL_CYCLE_UNRESOLVED"
        characterization_reasons = [
            "independent-period-estimate-needs-physical-cycle-characterization"
        ]
    elif candidate_characterization_warranted:
        characterization_state = (
            "RECURRENT_BUT_UNRESOLVED_CROSS_SECTOR_VARIABILITY"
        )
        characterization_reasons = candidate_characterization_reasons
    else:
        characterization_state = "NOT_WARRANTED_BY_BROAD_CLUSTERING"
        characterization_reasons = candidate_characterization_reasons

    return {
        "claimDecision": claim.as_dict(),
        "selectedPeriodDays": selected,
        "selectedSource": source,
        "eligibleSectorCount": eligible_count,
        "strictMajorityCount": strict_majority,
        "requiredClusterSupportCount": required_promotion_support,
        "minimumPromotionSectorCount": BROAD_MIN_PROMOTION_SECTORS,
        "clusterToleranceRelative": BROAD_CLUSTER_RELATIVE_TOLERANCE,
        "maximumPromotionClusterSpanRelative": (
            BROAD_MAX_PROMOTION_CLUSTER_SPAN_RELATIVE
        ),
        "minimumPromotionPeakProminenceRatio": (
            BROAD_MIN_PROMOTION_PEAK_PROMINENCE
        ),
        "promotionEligible": promotion_eligible,
        "promotionBlockers": promotion_blockers,
        "variabilityCharacterization": {
            "state": characterization_state,
            "warranted": characterization_warranted,
            "reasons": characterization_reasons,
            "nextHandler": (
                "openstar.tess.morphology.analyze"
                if characterization_warranted
                else None
            ),
            "scientificQuestion": (
                "Is the recurrent family a stable periodic source, evolving or "
                "nonstationary variability, contamination or blending, or a mixture?"
            ),
        },
        "bestCluster": best_cluster,
        "clusters": clusters,
        "harmonicFamily": harmonic_family,
        "sectorResults": sector_results,
    }
