"""Resolved-cycle recurrent-residual long-baseline confirmation.

This continuation consumes only the immutable v20.8 family-subtracted window
datasets and the completed v20.8.1 transient-mode validation.  It keeps the
resolved physical cycle fixed while testing an exact higher harmonic, one
training-selected coherent residual frequency, and an offset-only null with
leave-one-independent-sector-out prediction.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Iterable

from .tess_transient_mode_validation import (
    METHOD_CONTRACT_ID as TRANSIENT_METHOD_CONTRACT_ID,
    RESULT_VERSION as TRANSIENT_RESULT_VERSION,
    classify_transient_validation,
    method_contract_hash as transient_method_contract_hash,
)
from .tess_v20_8_long_baseline_time_frequency_confirmation import (
    CLASSIFICATIONS,
    FREQUENCY_GRID_POINT_COUNT,
    MIN_ACCEPTED_WINDOW_FRACTION_FOR_NONSTATIONARY,
    MIN_BIC_IMPROVEMENT,
    MIN_INDEPENDENT_SECTORS,
    analyze_long_baseline_time_frequency_confirmation,
    build_dataset_specs as build_v20_8_dataset_specs,
    method_contract_hash,
)


HANDLER_ID = (
    "openstar.tess.v20-8-long-baseline-time-frequency-confirmation.analyze"
)
METHOD_CONTRACT_ID = (
    "openstar.tess.recurrent-residual-long-baseline-confirmation."
    "leave-one-independent-sector-out.v1"
)
RESULT_VERSION = (
    "openstar.tess-recurrent-residual-long-baseline-confirmation.v1"
)

RECURRENT_CLASSIFICATION = "RESIDUAL_STRUCTURE_RECURRENT_ACROSS_BASELINE"
RECOMMENDED_NEXT_TEST = "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
_STRUCTURED_SUPPORT = {
    "TRANSIENT_FREQUENCY",
    "HARMONIC",
    "STRUCTURED_UNRESOLVED",
}


def _finite_positive(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{label} must be finite and positive.") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive.")
    return parsed


def _positive_frequencies(values: Iterable[Any]) -> list[float]:
    frequencies = {
        _finite_positive(value, "Recurrent-residual frequency")
        for value in values
        if value is not None
    }
    if not frequencies:
        raise RuntimeError("Recurrent-residual frequency anchors are empty.")
    return sorted(frequencies)


def validate_recurrent_residual_boundary(
    transient_validation: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact conservative v20.8.1 recurrent terminal result."""
    if not isinstance(transient_validation, dict):
        raise RuntimeError("Transient-mode validation evidence must be an object.")

    contract = transient_validation.get("methodContract")
    folds = transient_validation.get("perDetectionSectorEvidence")
    controls = transient_validation.get("perControlWindowEvidence")
    aggregate = transient_validation.get("aggregateDecision")
    if not (
        isinstance(contract, dict)
        and isinstance(folds, list)
        and isinstance(controls, list)
        and isinstance(aggregate, dict)
        and all(isinstance(item, dict) for item in [*folds, *controls])
    ):
        raise RuntimeError("Transient-mode validation evidence is malformed.")

    decision = classify_transient_validation(folds, controls)
    if not all(
        transient_validation.get(key) == decision[key]
        for key in (
            "classification",
            "recommendedNextTest",
            "aggregateDecision",
        )
    ):
        raise RuntimeError(
            "Transient-mode recurrent decision does not recompute exactly."
        )

    boundary = contract.get("evidenceBoundary") or {}
    models = contract.get("models") or {}
    data_policy = contract.get("dataPolicy") or {}
    detection_keys = boundary.get("transientDetectionWindowKeys") or []
    detection_sectors = sorted(
        int(value)
        for value in boundary.get("transientDetectionSectors") or []
    )
    independent_sectors = sorted(
        int(value)
        for value in boundary.get("acceptedIndependentSectors") or []
    )
    frozen_paths = [
        str(value)
        for value in boundary.get("frozenWindowDatasetPaths") or []
    ]
    if not (
        all(isinstance(item, dict) for item in detection_keys)
        and len(frozen_paths) == len(set(frozen_paths))
        and frozen_paths
    ):
        raise RuntimeError("Frozen recurrent-residual lineage is malformed.")

    try:
        detection_key_set = {
            (int(item["sector"]), int(item["windowIndex"]))
            for item in detection_keys
        }
        fold_sectors = sorted(int(item["heldOutSector"]) for item in folds)
        recurrent_sectors = sorted(
            int(value)
            for value in aggregate.get("recurrentStructuredSectors") or []
        )
        structured_control_count = int(
            aggregate["structuredControlWindowCount"]
        )
        sufficient_control_count = int(
            aggregate["sufficientIndependentControlWindowCount"]
        )
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(
            "Transient-mode recurrent sector lineage is invalid."
        ) from None

    independent_controls = [
        item for item in controls if item.get("role") == "INDEPENDENT_WINDOW"
    ]
    sufficient_controls = [
        item for item in independent_controls
        if item.get("support") != "INSUFFICIENT"
    ]
    structured_controls = [
        item for item in sufficient_controls
        if item.get("support") in _STRUCTURED_SUPPORT
    ]
    structured_sectors = sorted({
        *detection_sectors,
        *(int(item["sector"]) for item in structured_controls),
    })
    independent_window_count = (
        len(detection_key_set) + len(independent_controls)
    )
    structured_window_count = (
        len(detection_key_set) + len(structured_controls)
    )

    exact_harmonic = _finite_positive(
        transient_validation.get("exactHarmonicFrequencyCyclesPerDay"),
        "Exact recurrent-residual harmonic frequency",
    )
    persisted_transient = _finite_positive(
        boundary.get("persistedTransientFrequencyCyclesPerDay"),
        "Persisted transient frequency",
    )
    learned_all = _finite_positive(
        transient_validation.get(
            "allDetectionWindowLearnedFrequencyCyclesPerDay"
        ),
        "All-detection-window learned frequency",
    )
    anchors = _positive_frequencies([
        exact_harmonic,
        persisted_transient,
        learned_all,
        *(
            item.get("learnedTransientFrequencyCyclesPerDay")
            for item in folds
        ),
    ])

    exact = (
        transient_validation.get("version") == TRANSIENT_RESULT_VERSION
        and transient_validation.get("methodContractID")
        == TRANSIENT_METHOD_CONTRACT_ID
        and transient_validation.get("methodContractHash")
        == transient_method_contract_hash(contract)
        and transient_validation.get("classification")
        == RECURRENT_CLASSIFICATION
        and transient_validation.get("recommendedNextTest")
        == RECOMMENDED_NEXT_TEST
        and transient_validation.get("leaveOneTransientDetectionSectorOut")
        is True
        and transient_validation.get("controlWindowsUsedForSelection") is False
        and transient_validation.get("physicalMechanismResolved") is False
        and transient_validation.get("claimLevelChanged") is False
        and transient_validation.get("automaticDiscoveryClaim") is False
        and contract.get("methodContractID") == TRANSIENT_METHOD_CONTRACT_ID
        and contract.get("resultVersion") == TRANSIENT_RESULT_VERSION
        and contract.get("execution") == "PYTHON_SERVER"
        and contract.get("networkAccess") is False
        and data_policy.get("reuseFrozenV20_8ResidualWindowDatasets") is True
        and data_policy.get("downloadNewData") is False
        and data_policy.get("readOriginalSectorFlux") is False
        and boundary.get("classification") == "TRANSIENT_RESIDUAL_MODE"
        and boundary.get("residualEvolutionClassification")
        == "TRANSIENT_RESIDUAL_MODE"
        and boundary.get("recommendedNextTest") == "TRANSIENT_MODE_VALIDATION"
        and (boundary.get("periodReference") or {}).get("physicalCycleResolved")
        is True
        and models.get("H") == "EXACT_FAMILY_LOCKED_HIGHER_HARMONIC"
        and models.get("T")
        == "ONE_TRAINING_SELECTED_TRANSIENT_RESIDUAL_FREQUENCY"
        and models.get("N") == "RESIDUAL_FREE_OFFSET_ONLY_NULL"
        and math.isclose(
            exact_harmonic,
            _finite_positive(
                models.get("exactHarmonicFrequencyCyclesPerDay"),
                "Contract exact harmonic frequency",
            ),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        and len(detection_key_set) == len(detection_keys)
        and len(detection_sectors) >= 2
        and fold_sectors == detection_sectors
        and len(set(detection_sectors)) == len(detection_sectors)
        and len(independent_sectors) >= 3
        and set(detection_sectors).issubset(independent_sectors)
        and recurrent_sectors == structured_sectors
        and len(recurrent_sectors) >= 3
        and structured_control_count == len(structured_controls)
        and sufficient_control_count == len(sufficient_controls)
        and structured_window_count >= 3
        and independent_window_count >= structured_window_count
        and all(
            int(item["sector"]) in independent_sectors
            for item in independent_controls
        )
    )
    if not exact:
        raise RuntimeError(
            "Resolved-cycle long-baseline confirmation requires the exact "
            "v20.8.1 recurrent transient-mode validation boundary."
        )

    return {
        "periodReference": deepcopy(boundary["periodReference"]),
        "familyFrequencyCyclesPerDay": _finite_positive(
            boundary.get("familyFrequencyCyclesPerDay"),
            "Resolved family frequency",
        ),
        "persistedTransientFrequencyCyclesPerDay": persisted_transient,
        "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
        "frequencyAnchorsCyclesPerDay": anchors,
        "acceptedIndependentSectors": independent_sectors,
        "acceptedIndependentWindowCount": structured_window_count,
        "independentWindowCount": independent_window_count,
        "frozenWindowDatasetPaths": frozen_paths,
    }


def build_method_contract(
    *,
    transient_validation: dict[str, Any],
) -> dict[str, Any]:
    """Freeze all choices before any recurrent window flux is opened."""
    evidence = validate_recurrent_residual_boundary(transient_validation)
    return {
        "methodContractID": METHOD_CONTRACT_ID,
        "resultVersion": RESULT_VERSION,
        "execution": "PYTHON_SERVER",
        "networkAccess": False,
        "dataPolicy": {
            "reuseFrozenV20_8ResidualWindowDatasets": True,
            "downloadNewData": False,
            "readOriginalSectorFlux": False,
            "constructAndHashContractBeforeReadingWindowFlux": True,
        },
        "evidenceBoundary": {
            "classification": RECURRENT_CLASSIFICATION,
            "recommendedNextTest": RECOMMENDED_NEXT_TEST,
            "physicalMechanismResolved": False,
            "periodReference": evidence["periodReference"],
            "familyReferenceFrequencyCyclesPerDay": evidence[
                "familyFrequencyCyclesPerDay"
            ],
            "persistedResidualFrequencyCyclesPerDay": evidence[
                "persistedTransientFrequencyCyclesPerDay"
            ],
            "acceptedIndependentSectors": evidence[
                "acceptedIndependentSectors"
            ],
            "acceptedIndependentWindowCount": evidence[
                "acceptedIndependentWindowCount"
            ],
            "independentWindowCount": evidence["independentWindowCount"],
            "sourceTransientMethodContractID": (
                transient_validation["methodContractID"]
            ),
            "sourceTransientMethodContractHash": (
                transient_validation["methodContractHash"]
            ),
            "frozenWindowDatasetPaths": evidence[
                "frozenWindowDatasetPaths"
            ],
        },
        "crossValidation": {
            "scheme": "LEAVE_ONE_INDEPENDENT_SECTOR_OUT",
            "trainingIncludesFrozenPrimaryWindows": True,
            "frequencySelectionUsesTrainingSectorsOnly": True,
            "phaseLearningUsesTrainingSectorsOnly": True,
            "heldOutFrequencySelection": False,
            "heldOutPhaseSelection": False,
            "heldOutNuisanceParameters": [
                "offsetPerWindow",
                "signedAmplitudePerWindow",
            ],
        },
        "models": {
            "H": "EXACT_FAMILY_LOCKED_HIGHER_HARMONIC",
            "S": "ONE_TRAINING_SELECTED_COHERENT_RESIDUAL_FREQUENCY",
            "N": "RESIDUAL_FREE_OFFSET_ONLY_NULL",
            "exactHarmonicFrequencyCyclesPerDay": evidence[
                "exactHarmonicFrequencyCyclesPerDay"
            ],
        },
        "frequencyGrid": {
            "spacing": "LINEAR_IN_FREQUENCY",
            "pointCount": FREQUENCY_GRID_POINT_COUNT,
            "inclusiveMinimumCyclesPerDay": evidence[
                "frequencyAnchorsCyclesPerDay"
            ][0],
            "inclusiveMaximumCyclesPerDay": evidence[
                "frequencyAnchorsCyclesPerDay"
            ][-1],
            "mandatoryAnchorFrequenciesCyclesPerDay": evidence[
                "frequencyAnchorsCyclesPerDay"
            ],
            "selectionCriterion": "MINIMUM_TRAINING_BIC",
            "tieBreak": "LOWEST_FREQUENCY",
        },
        "decisionRules": {
            "criterion": "BIC",
            "minimumBICImprovement": MIN_BIC_IMPROVEMENT,
            "thresholdIsInclusive": True,
            "minimumIndependentSectors": MIN_INDEPENDENT_SECTORS,
            "frequencyStabilityTolerance": (
                "ONE_LONG_BASELINE_RAYLEIGH_RESOLUTION"
            ),
            "minimumAcceptedWindowFractionForNonstationary": (
                MIN_ACCEPTED_WINDOW_FRACTION_FOR_NONSTATIONARY
            ),
            "classifications": list(CLASSIFICATIONS),
        },
    }


def analyze_recurrent_residual_long_baseline_confirmation(
    *,
    method_contract: dict[str, Any],
    dataset_specs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Run the preregistered resolved-cycle recurrent-residual analysis."""
    return analyze_long_baseline_time_frequency_confirmation(
        method_contract=method_contract,
        dataset_specs=dataset_specs,
        expected_method_contract_id=METHOD_CONTRACT_ID,
        result_version=RESULT_VERSION,
    )


def build_dataset_specs(
    *, expected_tic_id: int, preparation: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    return build_v20_8_dataset_specs(
        expected_tic_id=expected_tic_id,
        preparation=preparation,
    )
