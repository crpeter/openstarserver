"""Fail-closed adapters for authoritative resolved photometric cycles."""
from __future__ import annotations

import math
from typing import Any

from .tess_dynamic_harmonic import (
    MIN_ALIAS_SUPPORTING_HELD_OUT_SECTORS,
    NESTED_ALIAS_EVIDENCE_LINEAGES,
    NESTED_ALIAS_METHOD,
    NESTED_ALIAS_RESOLVED_CLASSIFICATION,
)
from .tess_mode_identification import MIN_BIC_IMPROVEMENT


CONTRACT_VERSION = "openstar.tess-authoritative-resolved-cycle.v1"
MORPHOLOGY_SOURCE = "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
NESTED_ALIAS_SOURCE = "NESTED_ODD_HARMONIC_PREDICTIVE_RESOLUTION"
CORROBORATED_SOURCE = "MORPHOLOGY_AND_NESTED_PREDICTION_CONSISTENT"


def _positive_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _morphology_cycle(morphology: dict[str, Any]) -> dict[str, Any] | None:
    period = _positive_finite(morphology.get("resolvedPhysicalPeriodDays"))
    if morphology.get("physicalCycleResolved") is not True or period is None:
        return None
    return {
        "contractVersion": CONTRACT_VERSION,
        "periodDays": period,
        "sourceKind": MORPHOLOGY_SOURCE,
        "sourceClassification": morphology.get("morphologyClass"),
        "physicalCycleResolved": True,
        "physicalMechanismResolved": False,
    }


def _nested_alias_cycle(dynamic: dict[str, Any]) -> dict[str, Any] | None:
    alias = dynamic.get("periodAliasResolution") or {}
    resolved_period = _positive_finite(dynamic.get("resolvedPhysicalPeriodDays"))
    reference_period = _positive_finite(dynamic.get("referenceFamilyPeriodDays"))
    possible_double = _positive_finite(dynamic.get("possibleDoubleCycleDays"))
    selected_period = _positive_finite(alias.get("selectedPeriodDays"))
    raw_period = _positive_finite(dynamic.get("rawFamilyPeriodDays"))
    aggregate = _positive_finite(
        alias.get("aggregateIndependentDeltaBicFullMinusEvenOnly"))
    threshold = _positive_finite(alias.get("conservativeThreshold"))
    minimum_support = alias.get("minimumSupportingIndependentHeldOutSectors")
    primary_sector = alias.get("primarySector")
    supporters = alias.get(
        "oddHarmonicSupportingIndependentHeldOutSectors") or []
    comparisons = alias.get("comparisons") or []
    try:
        minimum_support = int(minimum_support)
        primary_sector = int(primary_sector)
        supporter_ids = [int(value) for value in supporters]
    except (TypeError, ValueError):
        return None
    supporter_comparisons = {}
    try:
        for item in comparisons:
            if (
                isinstance(item, dict)
                and item.get("sector") is not None
                and item.get("role") == "INDEPENDENT"
                and item.get("oddHarmonicStructureSupported") is True
            ):
                supporter_comparisons[int(item["sector"])] = item
    except (TypeError, ValueError):
        return None
    exact_periods = (
        resolved_period,
        reference_period,
        possible_double,
        selected_period,
    )
    valid = (
        dynamic.get("evidenceLineage") in NESTED_ALIAS_EVIDENCE_LINEAGES
        and dynamic.get("classification")
        == NESTED_ALIAS_RESOLVED_CLASSIFICATION
        and dynamic.get("physicalCycleResolved") is True
        and dynamic.get("physicalMechanismResolved") is False
        and dynamic.get("referencePeriodRole")
        == "PREDICTIVELY_RESOLVED_PHOTOMETRIC_CYCLE"
        and alias.get("method") == NESTED_ALIAS_METHOD
        and alias.get("criterion") == "BIC"
        and alias.get("physicalCycleResolved") is True
        and alias.get("selectedPeriodRelation") == "DOUBLE_CYCLE"
        and alias.get("equalHalfEvenHarmonicOrders") == [2, 4, 6, 8]
        and alias.get("discriminatingOddHarmonicOrders") == [1, 3, 5, 7]
        and alias.get("fullDoubleCycleHarmonicOrders") == list(range(1, 9))
        and alias.get("maximumAbsoluteFrequencyMatched") is True
        and all(value is not None for value in exact_periods)
        and all(math.isclose(value, resolved_period, rel_tol=1e-9, abs_tol=1e-12)
                for value in exact_periods[1:])
        and raw_period is not None
        and math.isclose(resolved_period, 2.0 * raw_period,
                         rel_tol=1e-9, abs_tol=1e-12)
        and threshold is not None
        and math.isclose(threshold, MIN_BIC_IMPROVEMENT,
                         rel_tol=0.0, abs_tol=1e-12)
        and aggregate is not None
        and aggregate >= threshold
        and minimum_support == MIN_ALIAS_SUPPORTING_HELD_OUT_SECTORS
        and len(supporter_ids) >= minimum_support
        and len(set(supporter_ids)) == len(supporter_ids)
        and primary_sector not in supporter_ids
        and all(sector in supporter_comparisons for sector in supporter_ids)
    )
    if not valid:
        return None
    return {
        "contractVersion": CONTRACT_VERSION,
        "periodDays": resolved_period,
        "rawFamilyPeriodDays": raw_period,
        "possibleDoubleCycleDays": possible_double,
        "referenceFamilyPeriodDays": reference_period,
        "selectedPeriodDays": selected_period,
        "sourceKind": NESTED_ALIAS_SOURCE,
        "sourceClassification": dynamic.get("classification"),
        "sourceEvidenceLineage": dynamic.get("evidenceLineage"),
        "physicalCycleResolved": True,
        "physicalMechanismResolved": False,
        "criterion": "BIC",
        "conservativeThreshold": threshold,
        "aggregateIndependentDeltaBicFullMinusEvenOnly": aggregate,
        "supportingIndependentSectors": sorted(supporter_ids),
        "minimumSupportingIndependentSectors": minimum_support,
        "primarySectorExcludedFromSupport": True,
        "maximumAbsoluteFrequencyMatched": True,
    }


def authoritative_resolved_cycle(
    *,
    morphology: dict[str, Any] | None,
    dynamic_harmonic: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return one consistent resolved-cycle contract or fail closed."""
    morphology_cycle = _morphology_cycle(morphology or {})
    nested_cycle = _nested_alias_cycle(dynamic_harmonic or {})
    if morphology_cycle is not None and nested_cycle is not None:
        if not math.isclose(
            morphology_cycle["periodDays"], nested_cycle["periodDays"],
            rel_tol=1e-9, abs_tol=1e-12,
        ):
            return None
        return {
            **nested_cycle,
            "sourceKind": CORROBORATED_SOURCE,
            "morphologySourceClassification": morphology_cycle.get(
                "sourceClassification"),
        }
    return nested_cycle or morphology_cycle


def validated_cycle_period(contract: dict[str, Any] | None) -> float | None:
    """Validate the distilled contract again at each consuming stage."""
    cycle = contract or {}
    period = _positive_finite(cycle.get("periodDays"))
    source = cycle.get("sourceKind")
    if not (
        cycle.get("contractVersion") == CONTRACT_VERSION
        and cycle.get("physicalCycleResolved") is True
        and cycle.get("physicalMechanismResolved") is False
        and period is not None
        and source in {MORPHOLOGY_SOURCE, NESTED_ALIAS_SOURCE,
                       CORROBORATED_SOURCE}
    ):
        return None
    if source == MORPHOLOGY_SOURCE:
        return period
    threshold = _positive_finite(cycle.get("conservativeThreshold"))
    aggregate = _positive_finite(
        cycle.get("aggregateIndependentDeltaBicFullMinusEvenOnly"))
    raw_period = _positive_finite(cycle.get("rawFamilyPeriodDays"))
    possible_double = _positive_finite(cycle.get("possibleDoubleCycleDays"))
    reference_period = _positive_finite(cycle.get("referenceFamilyPeriodDays"))
    selected_period = _positive_finite(cycle.get("selectedPeriodDays"))
    supporters = cycle.get("supportingIndependentSectors") or []
    try:
        minimum = int(cycle.get("minimumSupportingIndependentSectors"))
        supporter_ids = [int(value) for value in supporters]
    except (TypeError, ValueError):
        return None
    if not (
        cycle.get("criterion") == "BIC"
        and threshold is not None
        and math.isclose(threshold, MIN_BIC_IMPROVEMENT,
                         rel_tol=0.0, abs_tol=1e-12)
        and aggregate is not None
        and aggregate >= threshold
        and raw_period is not None
        and math.isclose(period, 2.0 * raw_period,
                         rel_tol=1e-9, abs_tol=1e-12)
        and all(
            value is not None
            and math.isclose(value, period, rel_tol=1e-9, abs_tol=1e-12)
            for value in (possible_double, reference_period, selected_period)
        )
        and minimum == MIN_ALIAS_SUPPORTING_HELD_OUT_SECTORS
        and len(supporter_ids) >= minimum
        and len(set(supporter_ids)) == len(supporter_ids)
        and cycle.get("primarySectorExcludedFromSupport") is True
        and cycle.get("maximumAbsoluteFrequencyMatched") is True
    ):
        return None
    return period
