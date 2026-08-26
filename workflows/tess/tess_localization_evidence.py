from __future__ import annotations

import math
from typing import Any


def frozen_residual_localization_family(
    morphology: dict[str, Any] | None,
    dynamic: dict[str, Any] | None,
    time_frequency_prepare: dict[str, Any] | None,
    time_frequency: dict[str, Any] | None,
    mode: dict[str, Any] | None,
) -> tuple[float, tuple[int, ...], dict[str, Any], str] | None:
    """Adapt complete, mutually consistent persisted evidence for localization.

    The returned model is an interface adapter for the stable residual evidence;
    it does not assert a morphology resolution or a physical interpretation.
    """
    if not all((morphology, time_frequency_prepare, time_frequency, mode)):
        return None
    residual = (time_frequency or {}).get("residualEvolution") or {}
    stable = (residual.get("classification") == "STABLE_RESIDUAL_MODE"
              or (time_frequency or {}).get("classification") == "STABLE_RESIDUAL_MODE")
    candidate = (mode or {}).get("modeCandidate") or {}
    family = (mode or {}).get("establishedPeriodFamily") or {}
    try:
        family_period = float(family["referencePeriodDays"])
        frequency = float(candidate["frequencyCyclesPerDay"])
        candidate_period = float(candidate["periodDays"])
        orders = tuple(int(value) for value in family["modeledHarmonicOrders"])
        sectors = tuple(int(value) for value in candidate["supportingSectors"])
        time_reference = float(time_frequency_prepare["absoluteTimeReferenceDays"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (stable and (mode or {}).get("independentModeEvidenceSurvived") is True
            and (mode or {}).get("physicalMechanismResolved") is False
            and math.isfinite(family_period) and family_period > 0
            and math.isfinite(frequency) and frequency > 0
            and math.isfinite(candidate_period) and candidate_period > 0
            and math.isfinite(time_reference)
            and orders and sectors and all(value > 0 for value in orders)
            and len(set(orders)) == len(orders)
            and math.isclose(candidate_period, 1.0 / frequency, rel_tol=1e-6)):
        return None

    path = "MODE_IDENTIFICATION_ESTABLISHED_PERIOD_FAMILY"
    dynamic_period = (dynamic or {}).get("referenceFamilyPeriodDays")
    dynamic_orders = (dynamic or {}).get("supportedHarmonicOrders") or ()
    if dynamic_period is not None and dynamic_orders:
        try:
            adapted_dynamic_period = float(dynamic_period)
            adapted_dynamic_orders = tuple(int(value) for value in dynamic_orders)
        except (TypeError, ValueError):
            return None
        if (
            not math.isfinite(adapted_dynamic_period)
            or adapted_dynamic_period <= 0
            or not adapted_dynamic_orders
            or any(value <= 0 for value in adapted_dynamic_orders)
            or len(set(adapted_dynamic_orders)) != len(adapted_dynamic_orders)
            or not math.isclose(
                adapted_dynamic_period,
                family_period,
                rel_tol=1e-9,
            )
        ):
            return None

        if adapted_dynamic_orders != orders:
            relation = (mode or {}).get("harmonicRelation") or {}
            try:
                tested_order = int(relation["testedOrder"])
            except (KeyError, TypeError, ValueError):
                return None

            expected_mode_orders = (1, 2, tested_order)

            if not (
                (mode or {}).get("classification")
                == "INDEPENDENT_STABLE_MODE"
                and (mode or {}).get(
                    "independentModeEvidenceSurvived"
                ) is True
                and relation.get(
                    "commensurateWithinResolution"
                ) is False
                and tested_order >= 3
                and orders == expected_mode_orders
                and 1 in adapted_dynamic_orders
                and 2 in adapted_dynamic_orders
            ):
                return None

        family_period, orders = (
            adapted_dynamic_period,
            adapted_dynamic_orders,
        )
        path = "DYNAMIC_HARMONIC_ESTABLISHED_PERIOD_FAMILY"

    resolved = (morphology or {}).get("physicalCycleResolved") is True
    if resolved:
        try:
            physical_period = float((morphology or {})["resolvedPhysicalPeriodDays"])
        except (KeyError, TypeError, ValueError):
            return None
        if (not math.isfinite(physical_period) or physical_period <= 0
                or not math.isclose(
                    physical_period, family_period, rel_tol=1e-9, abs_tol=1e-12
                )):
            return None
        reference_kind = "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
    else:
        physical_period = family_period
        reference_kind = "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE"

    model = {
        "preferredFrequencyAtReference": frequency,
        "preferredPeriodAtReferenceDays": candidate_period,
        "fractionalFrequencyDriftPerDay": 0.0,
        "timeReferenceDays": time_reference,
        "preferredModel": {"signalSectors": list(sectors)},
        "recommendedNextTest": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
        "evidenceSource": {"path": path},
    }
    return physical_period, orders, model, reference_kind
