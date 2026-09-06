from __future__ import annotations

import math
from typing import Any

from .tess_mode_identification import (
    CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID,
    CONFIRMED_COHERENT_MODE_RESULT_VERSION,
    V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE,
    confirmed_coherent_mode_method_contract_hash,
)


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


def frozen_confirmed_mode_localization_preparation_family(
    morphology: dict[str, Any] | None,
    mode: dict[str, Any] | None,
    preparation: dict[str, Any] | None,
) -> tuple[float, tuple[int, ...], dict[str, Any], str] | None:
    """Reuse the exact model already consumed by a completed v20.10 fit."""
    if not all((morphology, mode, preparation)):
        return None
    method_contract = (mode or {}).get("methodContract") or {}
    boundary = method_contract.get("evidenceBoundary") or {}
    comparison = method_contract.get("modelComparison") or {}
    candidate = (mode or {}).get("modeCandidate") or {}
    period_reference = (preparation or {}).get("periodReference") or {}
    try:
        physical_period = float(preparation["physicalPeriodDays"])
        morphology_period = float(morphology["resolvedPhysicalPeriodDays"])
        established_period = float(boundary["establishedPeriodDays"])
        family_period = float(
            mode["establishedPeriodFamily"]["referencePeriodDays"]
        )
        frequency = float(preparation["residualFrequencyAtReference"])
        residual_period = float(
            preparation["residualPeriodAtReferenceDays"]
        )
        candidate_frequency = float(candidate["frequencyCyclesPerDay"])
        candidate_period = float(candidate["periodDays"])
        drift = float(preparation["fractionalFrequencyDriftPerDay"])
        time_reference = float(preparation["timeReferenceDays"])
        orders = tuple(
            int(value) for value in preparation["subtractedHarmonicOrders"]
        )
        sectors = tuple(int(value) for value in preparation["signalSectors"])
        candidate_sectors = tuple(
            int(value) for value in candidate["supportingSectors"]
        )
        period_reference_period = float(period_reference["periodDays"])
        total_work_units = int(preparation["totalWorkUnits"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None

    positive = (
        physical_period,
        morphology_period,
        established_period,
        family_period,
        frequency,
        residual_period,
        candidate_frequency,
        candidate_period,
        period_reference_period,
    )
    exact = (
        morphology.get("physicalCycleResolved") is True
        and mode.get("version") == CONFIRMED_COHERENT_MODE_RESULT_VERSION
        and mode.get("evidenceLineage")
        == V20_8_CONFIRMED_COHERENT_MODE_EVIDENCE_LINEAGE
        and mode.get("classification") == "INDEPENDENT_STABLE_MODE"
        and mode.get("independentModeEvidenceSurvived") is True
        and mode.get("physicalMechanismResolved") is False
        and mode.get("pulsationMechanismResolved") is False
        and mode.get("claimLevelChanged") is False
        and mode.get("automaticDiscoveryClaim") is False
        and mode.get("recommendedNextTest")
        == "RESIDUAL_MODE_PIXEL_LOCALIZATION"
        and mode.get("methodContractID")
        == CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID
        and method_contract.get("methodContractID")
        == CONFIRMED_COHERENT_MODE_METHOD_CONTRACT_ID
        and mode.get("methodContractHash")
        == confirmed_coherent_mode_method_contract_hash(method_contract)
        and comparison.get("familyHarmonicOrders") == [1, 2]
        and preparation.get("available") is True
        and preparation.get("physicalMechanismResolved") is False
        and preparation.get("workloadID") == "openstar.lomb-scargle.v1"
        and isinstance(preparation.get("projectPath"), str)
        and bool(preparation.get("projectPath"))
        and isinstance(preparation.get("preparedPixels"), list)
        and bool(preparation.get("preparedPixels"))
        and total_work_units > 0
        and period_reference.get("kind")
        == "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
        and period_reference.get("physicalCycleResolved") is True
        and all(math.isfinite(value) and value > 0.0 for value in positive)
        and math.isfinite(drift)
        and math.isfinite(time_reference)
        and orders == (1, 2)
        and sectors
        and len(set(sectors)) == len(sectors)
        and sectors == candidate_sectors
        and math.isclose(
            physical_period, morphology_period, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            physical_period, established_period, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            physical_period, family_period, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(
            physical_period,
            period_reference_period,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        and math.isclose(
            frequency, candidate_frequency, rel_tol=1e-12, abs_tol=1e-15
        )
        and math.isclose(
            residual_period,
            candidate_period,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(
            residual_period,
            1.0 / frequency,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    )
    if not exact:
        return None

    model = {
        "preferredFrequencyAtReference": frequency,
        "preferredPeriodAtReferenceDays": residual_period,
        "fractionalFrequencyDriftPerDay": drift,
        "timeReferenceDays": time_reference,
        "preferredModel": {"signalSectors": list(sectors)},
        "recommendedNextTest": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
        "evidenceLineage": mode["evidenceLineage"],
        "methodContractID": mode["methodContractID"],
        "methodContractHash": mode["methodContractHash"],
        "evidenceSource": {
            "path": "CONFIRMED_MODE_COMPLETED_LOCALIZATION_PREPARATION"
        },
    }
    return physical_period, orders, model, period_reference["kind"]


def frozen_confirmed_mode_prf_preparation_family(
    morphology: dict[str, Any] | None,
    mode: dict[str, Any] | None,
    localization_preparation: dict[str, Any] | None,
    multisource_preparation: dict[str, Any] | None,
    prf_preparation: dict[str, Any] | None,
) -> tuple[float, tuple[int, ...], dict[str, Any], str] | None:
    """Reuse the confirmed-mode family only after exact v20.12/PRF transport."""
    family = frozen_confirmed_mode_localization_preparation_family(
        morphology, mode, localization_preparation,
    )
    if family is None or not multisource_preparation or not prf_preparation:
        return None
    physical_period, orders, model, reference_kind = family
    family_provenance = multisource_preparation.get("familyModelProvenance") or {}
    family_source = family_provenance.get("sourceEvidence") or {}
    residual_provenance = multisource_preparation.get("residualModelProvenance") or {}
    residual_source = residual_provenance.get("sourceEvidence") or {}
    adapter = "frozen_confirmed_mode_localization_preparation_family"
    try:
        multisource_period = float(multisource_preparation["referenceFamilyPeriodDays"])
        multisource_frequency = float(multisource_preparation["referenceFrequency"])
        multisource_drift = float(
            multisource_preparation["fractionalFrequencyDriftPerDay"]
        )
        multisource_time_reference = float(multisource_preparation["timeReferenceDays"])
        multisource_orders = tuple(
            int(value)
            for value in multisource_preparation["subtractedHarmonicOrders"]
        )
        multisource_sectors = tuple(
            int(value) for value in residual_provenance["signalSectors"]
        )
        prf_period = float(prf_preparation["referenceFamilyPeriodDays"])
        prf_frequency = float(prf_preparation["residualReferenceFrequency"])
        prf_drift = float(prf_preparation["fractionalFrequencyDriftPerDay"])
        prf_time_reference = float(prf_preparation["residualTimeReferenceDays"])
        prf_orders = tuple(
            int(value) for value in prf_preparation["subtractedHarmonicOrders"]
        )
        prf_sectors = tuple(int(value) for value in prf_preparation["sectors"])
    except (KeyError, TypeError, ValueError):
        return None
    model_sectors = tuple(model["preferredModel"]["signalSectors"])
    exact = (
        multisource_preparation.get("available") is True
        and multisource_preparation.get("workloadID")
        == "openstar.lomb-scargle.v1"
        and multisource_preparation.get("physicalCycleResolved") is True
        and family_provenance.get("physicalCycleResolved") is True
        and family_source.get("adapter") == adapter
        and family_source.get("referenceKind") == reference_kind
        and residual_source.get("adapter") == adapter
        and prf_preparation.get("version")
        == "openstar.tess-prf-deblending.v1"
        and prf_preparation.get("modelSource")
        == "official-public-SPOC-TESS-PRF-FITS"
        and prf_preparation.get("physicalCycleResolved") is True
        and prf_preparation.get("familyModelProvenance") == family_provenance
        and prf_preparation.get("residualModelProvenance")
        == residual_provenance
        and multisource_orders == orders == prf_orders
        and multisource_sectors == model_sectors
        and prf_sectors == tuple(sorted(model_sectors))
        and all(math.isfinite(value) for value in (
            multisource_period, multisource_frequency, multisource_drift,
            multisource_time_reference, prf_period, prf_frequency, prf_drift,
            prf_time_reference,
        ))
        and math.isclose(
            multisource_period, physical_period, rel_tol=1e-9, abs_tol=1e-12
        )
        and math.isclose(prf_period, physical_period, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(
            multisource_frequency,
            model["preferredFrequencyAtReference"],
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        and math.isclose(
            prf_frequency, multisource_frequency, rel_tol=1e-12, abs_tol=1e-15
        )
        and math.isclose(prf_drift, multisource_drift, abs_tol=1e-15)
        and math.isclose(
            prf_time_reference, multisource_time_reference, abs_tol=1e-12
        )
    )
    return family if exact else None
