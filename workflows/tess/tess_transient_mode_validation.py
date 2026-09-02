"""Predictive validation of a transient v20.8 residual-frequency cluster.

The analysis consumes only the persisted, family-subtracted sliding-window
datasets from the completed v20.8 time-frequency stage.  Its deterministic
method contract is constructed and hashed before any frozen flux is read.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from statistics import median
from typing import Any, Iterable, Sequence

from .tess_long_baseline_frequency_confirmation import (
    _predictive_fit,
    _training_fit,
)
from .tess_v20_8_long_baseline_time_frequency_confirmation import (
    build_dataset_specs as _build_v20_8_dataset_specs,
    validate_frozen_window_lineage,
)


HANDLER_ID = "openstar.tess.transient-mode-validation.analyze"
RESULT_VERSION = "openstar.tess-transient-mode-validation.v1"
METHOD_CONTRACT_ID = (
    "openstar.tess.transient-mode-validation."
    "leave-one-detection-sector-out.v1"
)

MIN_BIC_IMPROVEMENT = 10.0
MIN_DETECTION_SECTORS = 2
MIN_INDEPENDENT_SECTORS_WITH_CONTROLS = 3
MIN_RECURRENT_SECTORS = 3
MIN_RECURRENT_CONTROL_WINDOWS = 3
FREQUENCY_GRID_POINT_COUNT = 101

TRANSIENT_INDEPENDENT = "TRANSIENT_INDEPENDENT_FREQUENCY_SUPPORTED"
TRANSIENT_HARMONIC = "TRANSIENT_HARMONIC_STRUCTURE_SUPPORTED"
RECURRENT = "RESIDUAL_STRUCTURE_RECURRENT_ACROSS_BASELINE"
INCONCLUSIVE = "TRANSIENT_MODE_VALIDATION_INCONCLUSIVE"

CLASSIFICATIONS = (
    TRANSIENT_INDEPENDENT,
    TRANSIENT_HARMONIC,
    RECURRENT,
    INCONCLUSIVE,
)

RECOMMENDED_NEXT_TESTS = {
    TRANSIENT_INDEPENDENT: "TRANSIENT_RESIDUAL_PIXEL_LOCALIZATION",
    TRANSIENT_HARMONIC: "DYNAMIC_HARMONIC_MODELING",
    RECURRENT: "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION",
    INCONCLUSIVE: "HUMAN_SCIENTIFIC_REVIEW",
}


def _finite_positive(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{label} must be finite and positive.") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive.")
    return parsed


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def method_contract_hash(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_key(item: dict[str, Any]) -> tuple[int, int]:
    try:
        sector = int(item["sector"])
        window_index = int(item["windowIndex"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Transient-window sector lineage is invalid.") from None
    if sector < 1 or window_index < 1:
        raise RuntimeError("Transient-window indices are invalid.")
    return sector, window_index


def validate_transient_boundary(
    *,
    morphology: dict[str, Any],
    binary_confirmation: dict[str, Any],
    preparation: dict[str, Any],
    interpretation: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact resolved-cycle transient v20.8 boundary."""
    if not all(isinstance(value, dict) for value in (
        morphology,
        binary_confirmation,
        preparation,
        interpretation,
        summary,
    )):
        raise RuntimeError("Transient-mode evidence must contain objects.")

    period_reference = summary.get("periodReference") or {}
    residual = summary.get("residualEvolution") or {}
    best_cluster = residual.get("bestCluster") or {}
    binary_independent = binary_confirmation.get("independentEvidence") or {}
    binary_ephemeris = binary_confirmation.get("linearEphemeris") or {}
    prepared_windows = preparation.get("preparedWindows") or []
    interpreted_windows = interpretation.get("windowResults") or []
    summary_windows = summary.get("windowResults") or []
    members = best_cluster.get("members") or []
    if not all(isinstance(value, list) for value in (
        prepared_windows,
        interpreted_windows,
        summary_windows,
        members,
    )):
        raise RuntimeError("Transient-mode window evidence is malformed.")

    physical_period = _finite_positive(
        period_reference.get("periodDays"),
        "Transient-mode physical period",
    )
    morphology_period = _finite_positive(
        morphology.get("resolvedPhysicalPeriodDays"),
        "Morphology-resolved physical period",
    )
    preparation_period = _finite_positive(
        preparation.get("physicalPeriodDays"),
        "Time-frequency preparation period",
    )
    summary_period = _finite_positive(
        summary.get("physicalPeriodDays"),
        "Time-frequency summary period",
    )
    family_frequency = _finite_positive(
        preparation.get("physicalFrequency"),
        "Time-frequency family frequency",
    )
    interpretation_frequency = _finite_positive(
        interpretation.get("physicalFrequency"),
        "Time-frequency interpretation frequency",
    )
    summary_frequency = _finite_positive(
        summary.get("physicalFrequency"),
        "Time-frequency summary frequency",
    )

    try:
        harmonic_orders = tuple(
            int(value) for value in preparation.get("subtractedHarmonicOrders")
        )
    except (TypeError, ValueError):
        raise RuntimeError("Transient-mode harmonic lineage is invalid.") from None
    if harmonic_orders != (1, 2):
        raise RuntimeError(
            "Transient-mode validation requires the frozen fundamental and "
            "first-harmonic subtraction."
        )

    prepared_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    prepared_by_id: dict[str, dict[str, Any]] = {}
    independent_sectors = set()
    for item in prepared_windows:
        if not isinstance(item, dict) or not item.get("datasetID") or not item.get("datasetPath"):
            raise RuntimeError("Prepared transient-window lineage is incomplete.")
        key = _window_key(item)
        dataset_id = str(item["datasetID"])
        if key in prepared_by_key or dataset_id in prepared_by_id:
            raise RuntimeError("Prepared transient-window lineage is duplicated.")
        role = str(item.get("role") or "")
        if role not in {
            "primary-time-frequency-window",
            "independent-time-frequency-window",
        }:
            raise RuntimeError("Prepared transient-window role is invalid.")
        prepared_by_key[key] = item
        prepared_by_id[dataset_id] = item
        if role == "independent-time-frequency-window":
            independent_sectors.add(key[0])

    interpreted_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    interpreted_by_id: dict[str, dict[str, Any]] = {}
    for item in interpreted_windows:
        if not isinstance(item, dict) or not item.get("datasetID"):
            raise RuntimeError("Interpreted transient-window lineage is incomplete.")
        key = _window_key(item)
        dataset_id = str(item["datasetID"])
        prepared = prepared_by_id.get(dataset_id)
        if prepared is None or key in interpreted_by_key or dataset_id in interpreted_by_id:
            raise RuntimeError(
                "Interpreted transient windows do not match preparation."
            )
        if not (
            key == _window_key(prepared)
            and item.get("role") == prepared.get("role")
        ):
            raise RuntimeError("Transient-window target/sector lineage changed.")
        interpreted_by_key[key] = item
        interpreted_by_id[dataset_id] = item

    if set(interpreted_by_id) != set(prepared_by_id):
        raise RuntimeError("Prepared/interpreted transient-window sets differ.")
    if summary_windows != interpreted_windows:
        raise RuntimeError(
            "Transient-mode summary no longer matches its interpretation."
        )

    member_keys = set()
    member_frequencies = []
    member_sectors = set()
    for member in members:
        if not isinstance(member, dict):
            raise RuntimeError("Transient-cluster membership is malformed.")
        key = _window_key(member)
        interpreted = interpreted_by_key.get(key)
        if key in member_keys or interpreted is None:
            raise RuntimeError("Transient-cluster membership is inconsistent.")
        frequency = _finite_positive(
            member.get("frequency"),
            "Transient-cluster member frequency",
        )
        if not (
            interpreted.get("role") == "independent-time-frequency-window"
            and interpreted.get("acceptedTimeFrequencyFeature") is True
            and interpreted.get("nearEstablishedFamily") is False
            and _close(
                frequency,
                _finite_positive(
                    interpreted.get("candidateFrequency"),
                    "Interpreted transient frequency",
                ),
            )
        ):
            raise RuntimeError(
                "Transient-cluster membership does not match accepted evidence."
            )
        member_keys.add(key)
        member_frequencies.append(frequency)
        member_sectors.add(key[0])
    if not member_frequencies:
        raise RuntimeError("Transient-cluster membership is empty.")

    cluster_frequency = _finite_positive(
        best_cluster.get("medianFrequency"),
        "Transient-cluster frequency",
    )
    cluster_period = _finite_positive(
        best_cluster.get("medianPeriodDays"),
        "Transient-cluster period",
    )
    expected_cluster_sectors = sorted(member_sectors)
    persisted_cluster_sectors = sorted(
        int(value) for value in best_cluster.get("independentSectors") or []
    )
    persisted_summary_sectors = sorted(
        int(value) for value in summary.get("acceptedIndependentSectors") or []
    )
    accepted_summary_sectors = sorted({
        int(item["sector"])
        for item in summary_windows
        if item.get("role") == "independent-time-frequency-window"
        and item.get("acceptedTimeFrequencyFeature") is True
    })
    try:
        cluster_window_count = int(best_cluster.get("windowCount"))
        cluster_sector_count = int(best_cluster.get("independentSectorCount"))
        window_count = int(summary.get("windowCount"))
        accepted_count = int(summary.get("acceptedFeatureCount"))
        accepted_independent_count = int(
            summary.get("acceptedIndependentFeatureCount")
        )
    except (TypeError, ValueError):
        raise RuntimeError("Transient-mode persisted counts are invalid.") from None
    actual_accepted = [
        item for item in summary_windows
        if item.get("acceptedTimeFrequencyFeature") is True
    ]
    actual_accepted_independent = [
        item for item in actual_accepted
        if item.get("role") == "independent-time-frequency-window"
    ]

    exact = (
        morphology.get("physicalCycleResolved") is True
        and morphology.get("morphologyClass")
        == "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED"
        and binary_independent.get("classification")
        == "ECLIPSE_LIKE_EVENT_UNRESOLVED"
        and binary_ephemeris.get("coherent") is False
        and preparation.get("available") is True
        and summary.get("classification") == "TRANSIENT_RESIDUAL_MODE"
        and residual.get("classification") == "TRANSIENT_RESIDUAL_MODE"
        and summary.get("recommendedNextTest") == "TRANSIENT_MODE_VALIDATION"
        and summary.get("physicalMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and period_reference.get("kind")
        == "MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD"
        and period_reference.get("physicalCycleResolved") is True
        and _close(physical_period, morphology_period)
        and _close(physical_period, preparation_period)
        and _close(physical_period, summary_period)
        and _close(family_frequency, 1.0 / physical_period)
        and _close(interpretation_frequency, family_frequency)
        and _close(summary_frequency, family_frequency)
        and _close(cluster_period, 1.0 / cluster_frequency)
        and _close(cluster_frequency, float(median(member_frequencies)))
        and len(member_keys) >= MIN_DETECTION_SECTORS
        and len(member_sectors) == MIN_DETECTION_SECTORS
        and cluster_window_count == len(member_keys)
        and cluster_sector_count == len(member_sectors)
        and persisted_cluster_sectors == expected_cluster_sectors
        and len(independent_sectors)
        >= MIN_INDEPENDENT_SECTORS_WITH_CONTROLS
        and expected_cluster_sectors
        and set(expected_cluster_sectors).issubset(independent_sectors)
        and persisted_summary_sectors == accepted_summary_sectors
        and window_count == len(summary_windows) == len(prepared_windows)
        and accepted_count == len(actual_accepted)
        and accepted_independent_count == len(actual_accepted_independent)
    )
    if not exact:
        raise RuntimeError(
            "Transient-mode validation requires the exact terminal resolved-cycle "
            "v20.8 TRANSIENT_RESIDUAL_MODE boundary."
        )

    tested_order = max(3, int(round(cluster_frequency / family_frequency)))
    exact_harmonic_frequency = tested_order * family_frequency
    return {
        "physicalPeriodDays": physical_period,
        "familyFrequencyCyclesPerDay": family_frequency,
        "subtractedHarmonicOrders": list(harmonic_orders),
        "persistedTransientFrequencyCyclesPerDay": cluster_frequency,
        "persistedTransientPeriodDays": cluster_period,
        "transientMemberFrequenciesCyclesPerDay": member_frequencies,
        "transientDetectionWindowKeys": [
            {"sector": sector, "windowIndex": window_index}
            for sector, window_index in sorted(member_keys)
        ],
        "transientDetectionSectors": expected_cluster_sectors,
        "allIndependentSectors": sorted(independent_sectors),
        "testedHarmonicOrder": tested_order,
        "exactHarmonicFrequencyCyclesPerDay": exact_harmonic_frequency,
        "preparedWindows": deepcopy(prepared_windows),
    }


def build_method_contract(
    *,
    morphology: dict[str, Any],
    binary_confirmation: dict[str, Any],
    preparation: dict[str, Any],
    interpretation: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Freeze every model and decision choice before reading window flux."""
    evidence = validate_transient_boundary(
        morphology=morphology,
        binary_confirmation=binary_confirmation,
        preparation=preparation,
        interpretation=interpretation,
        summary=summary,
    )
    anchors = sorted({
        *evidence["transientMemberFrequenciesCyclesPerDay"],
        evidence["persistedTransientFrequencyCyclesPerDay"],
        evidence["exactHarmonicFrequencyCyclesPerDay"],
    })
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
            "classification": summary["classification"],
            "residualEvolutionClassification": summary[
                "residualEvolution"
            ]["classification"],
            "recommendedNextTest": summary["recommendedNextTest"],
            "physicalMechanismResolved": False,
            "periodReference": deepcopy(summary["periodReference"]),
            "familyFrequencyCyclesPerDay": evidence[
                "familyFrequencyCyclesPerDay"
            ],
            "persistedTransientFrequencyCyclesPerDay": evidence[
                "persistedTransientFrequencyCyclesPerDay"
            ],
            "transientDetectionWindowKeys": evidence[
                "transientDetectionWindowKeys"
            ],
            "transientDetectionSectors": evidence[
                "transientDetectionSectors"
            ],
            "acceptedIndependentSectors": evidence["allIndependentSectors"],
            "frozenWindowDatasetPaths": [
                item["datasetPath"] for item in evidence["preparedWindows"]
            ],
        },
        "crossValidation": {
            "scheme": "LEAVE_ONE_TRANSIENT_DETECTION_SECTOR_OUT",
            "frequencySelectionUsesTrainingDetectionWindowsOnly": True,
            "phaseLearningUsesTrainingDetectionWindowsOnly": True,
            "heldOutFrequencySelection": False,
            "heldOutPhaseSelection": False,
            "controlWindowsUsedForSelection": False,
            "heldOutNuisanceParameters": ["offset", "signedAmplitude"],
        },
        "models": {
            "H": "EXACT_FAMILY_LOCKED_HIGHER_HARMONIC",
            "T": "ONE_TRAINING_SELECTED_TRANSIENT_RESIDUAL_FREQUENCY",
            "N": "RESIDUAL_FREE_OFFSET_ONLY_NULL",
            "testedHarmonicOrder": evidence["testedHarmonicOrder"],
            "exactHarmonicFrequencyCyclesPerDay": evidence[
                "exactHarmonicFrequencyCyclesPerDay"
            ],
        },
        "frequencyGrid": {
            "spacing": "LINEAR_IN_FREQUENCY",
            "pointCount": FREQUENCY_GRID_POINT_COUNT,
            "inclusiveMinimumCyclesPerDay": anchors[0],
            "inclusiveMaximumCyclesPerDay": anchors[-1],
            "mandatoryAnchorFrequenciesCyclesPerDay": anchors,
            "selectionCriterion": "MINIMUM_TRAINING_BIC",
            "tieBreak": "LOWEST_FREQUENCY",
        },
        "decisionRules": {
            "criterion": "BIC",
            "minimumBICImprovement": MIN_BIC_IMPROVEMENT,
            "thresholdIsInclusive": True,
            "minimumTransientDetectionSectors": MIN_DETECTION_SECTORS,
            "minimumIndependentSectorsWithControls": (
                MIN_INDEPENDENT_SECTORS_WITH_CONTROLS
            ),
            "minimumRecurrentSectors": MIN_RECURRENT_SECTORS,
            "minimumRecurrentControlWindows": MIN_RECURRENT_CONTROL_WINDOWS,
            "frequencySeparationTolerance": (
                "ONE_ALL_WINDOW_LONG_BASELINE_RAYLEIGH_RESOLUTION"
            ),
            "positiveInterpretationRequiresAggregatePredictiveSupport": True,
            "classifications": list(CLASSIFICATIONS),
        },
    }


def build_dataset_specs(
    *, expected_tic_id: int, preparation: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    return _build_v20_8_dataset_specs(
        expected_tic_id=expected_tic_id,
        preparation=preparation,
    )


def validate_frozen_dataset_lineage(
    *,
    method_contract: dict[str, Any],
    dataset_specs: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate all frozen windows after contract preregistration."""
    return validate_frozen_window_lineage(
        method_contract=method_contract,
        dataset_specs=dataset_specs,
    )


def _frequency_grid(contract: dict[str, Any]) -> tuple[float, ...]:
    spec = contract["frequencyGrid"]
    lower = float(spec["inclusiveMinimumCyclesPerDay"])
    upper = float(spec["inclusiveMaximumCyclesPerDay"])
    count = int(spec["pointCount"])
    values = {lower}
    if not _close(lower, upper):
        values.update(
            lower + (upper - lower) * index / (count - 1)
            for index in range(count)
        )
    values.update(
        float(value)
        for value in spec["mandatoryAnchorFrequenciesCyclesPerDay"]
    )
    return tuple(sorted(value for value in values if value > 0.0))


def _aggregate_prediction(
    *,
    datasets: Sequence[dict[str, Any]],
    frequency: float | None,
    phase: float | None,
    trained_frequency_parameters: int = 0,
) -> dict[str, Any]:
    fits = []
    total_bic = 0.0
    total_rss = 0.0
    total_samples = 0
    for dataset in sorted(
        datasets,
        key=lambda item: (
            int(item["sector"]),
            int(item["windowIndex"]),
            item["datasetID"],
        ),
    ):
        fit = _predictive_fit(
            dataset,
            () if frequency is None else (frequency,),
            () if phase is None else (phase,),
            trained_frequency_parameters=trained_frequency_parameters,
        )
        total_bic += float(fit["bic"])
        total_rss += float(fit["rss"])
        total_samples += int(fit["sampleCount"])
        fits.append({
            "datasetID": dataset["datasetID"],
            "sector": int(dataset["sector"]),
            "windowIndex": int(dataset["windowIndex"]),
            **fit,
        })
    return {
        "bic": total_bic,
        "rss": total_rss,
        "sampleCount": total_samples,
        "windowFits": fits,
    }


def _support(
    *,
    harmonic_bic: float,
    transient_bic: float,
    null_bic: float,
    learned_frequency: float,
    exact_harmonic_frequency: float,
    frequency_resolution: float,
) -> tuple[str, list[str]]:
    harmonic_over_null = null_bic - harmonic_bic
    transient_over_null = null_bic - transient_bic
    transient_over_harmonic = harmonic_bic - transient_bic
    separated = (
        abs(learned_frequency - exact_harmonic_frequency)
        > frequency_resolution
    )
    if (
        transient_over_null >= MIN_BIC_IMPROVEMENT
        and transient_over_harmonic >= MIN_BIC_IMPROVEMENT
        and separated
    ):
        return "TRANSIENT_FREQUENCY", []
    if (
        harmonic_over_null >= MIN_BIC_IMPROVEMENT
        and -transient_over_harmonic >= MIN_BIC_IMPROVEMENT
    ):
        return "HARMONIC", []
    if max(harmonic_over_null, transient_over_null) >= MIN_BIC_IMPROVEMENT:
        reasons = [
            "PREDICTIVE_STRUCTURE_DOES_NOT_SEPARATE_TRANSIENT_AND_HARMONIC_MODELS"
        ]
        if not separated:
            reasons.append(
                "TRANSIENT_AND_HARMONIC_FREQUENCIES_UNRESOLVED_AT_LONG_BASELINE"
            )
        return "STRUCTURED_UNRESOLVED", reasons
    return "NEITHER", [
        "NO_FIXED_PHASE_RESIDUAL_MODEL_IMPROVES_ON_NULL_BY_THRESHOLD"
    ]


def classify_transient_validation(
    detection_folds: Sequence[dict[str, Any]],
    control_windows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply conservative aggregate rules to predictive evidence."""
    sufficient = [
        fold for fold in detection_folds
        if fold.get("support") != "INSUFFICIENT"
        and isinstance(fold.get("predictiveBIC"), dict)
    ]
    bics = {
        model: sum(float(fold["predictiveBIC"][model]) for fold in sufficient)
        for model in ("H", "T", "N")
    }
    transient_folds = [
        fold for fold in sufficient
        if fold.get("support") == "TRANSIENT_FREQUENCY"
    ]
    harmonic_folds = [
        fold for fold in sufficient if fold.get("support") == "HARMONIC"
    ]
    transient_aggregate = (
        bics["N"] - bics["T"] >= MIN_BIC_IMPROVEMENT
        and bics["H"] - bics["T"] >= MIN_BIC_IMPROVEMENT
    )
    harmonic_aggregate = (
        bics["N"] - bics["H"] >= MIN_BIC_IMPROVEMENT
        and bics["T"] - bics["H"] >= MIN_BIC_IMPROVEMENT
    )
    control_sufficient = [
        item for item in control_windows
        if item.get("role") == "INDEPENDENT_WINDOW"
        and item.get("support") != "INSUFFICIENT"
    ]
    structured_controls = [
        item for item in control_sufficient
        if item.get("support") in {
            "TRANSIENT_FREQUENCY",
            "HARMONIC",
            "STRUCTURED_UNRESOLVED",
        }
    ]
    detection_structured_sectors = {
        int(fold["heldOutSector"])
        for fold in sufficient
        if fold.get("support") in {
            "TRANSIENT_FREQUENCY",
            "HARMONIC",
            "STRUCTURED_UNRESOLVED",
        }
    }
    control_structured_sectors = {
        int(item["sector"]) for item in structured_controls
    }
    recurrent_sectors = sorted(
        detection_structured_sectors | control_structured_sectors
    )
    recurrent = (
        len(recurrent_sectors) >= MIN_RECURRENT_SECTORS
        and len(structured_controls) >= MIN_RECURRENT_CONTROL_WINDOWS
    )

    if (
        len(harmonic_folds) >= MIN_DETECTION_SECTORS
        and harmonic_aggregate
    ):
        classification = TRANSIENT_HARMONIC
    elif recurrent:
        classification = RECURRENT
    elif (
        len(transient_folds) >= MIN_DETECTION_SECTORS
        and transient_aggregate
    ):
        classification = TRANSIENT_INDEPENDENT
    else:
        classification = INCONCLUSIVE

    return {
        "classification": classification,
        "recommendedNextTest": RECOMMENDED_NEXT_TESTS[classification],
        "aggregateDecision": {
            "predictiveBIC": bics,
            "bicImprovementTransientOverNull": bics["N"] - bics["T"],
            "bicImprovementTransientOverHarmonic": bics["H"] - bics["T"],
            "bicImprovementHarmonicOverNull": bics["N"] - bics["H"],
            "sufficientDetectionFoldCount": len(sufficient),
            "transientSupportingDetectionSectorCount": len(transient_folds),
            "harmonicSupportingDetectionSectorCount": len(harmonic_folds),
            "structuredControlWindowCount": len(structured_controls),
            "sufficientIndependentControlWindowCount": len(control_sufficient),
            "recurrentStructuredSectors": recurrent_sectors,
        },
    }


def analyze_transient_mode_validation(
    *,
    method_contract: dict[str, Any],
    dataset_specs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Predict detection windows without using held-out or control flux."""
    if method_contract.get("methodContractID") != METHOD_CONTRACT_ID:
        raise RuntimeError("Unsupported transient-mode method contract.")
    contract_hash = method_contract_hash(method_contract)
    datasets = validate_frozen_dataset_lineage(
        method_contract=method_contract,
        dataset_specs=dataset_specs,
    )
    all_times = [time for dataset in datasets for time in dataset["times"]]
    baseline = max(all_times) - min(all_times)
    if baseline <= 0.0:
        raise RuntimeError("Frozen transient windows have no positive baseline.")
    resolution = 1.0 / baseline

    boundary = method_contract["evidenceBoundary"]
    detection_keys = {
        (int(item["sector"]), int(item["windowIndex"]))
        for item in boundary["transientDetectionWindowKeys"]
    }
    detection = [
        item for item in datasets
        if (int(item["sector"]), int(item["windowIndex"])) in detection_keys
    ]
    controls = [
        item for item in datasets
        if (int(item["sector"]), int(item["windowIndex"])) not in detection_keys
    ]
    detection_sectors = sorted({int(item["sector"]) for item in detection})
    if (
        len(detection) != len(detection_keys)
        or detection_sectors != sorted(
            int(value) for value in boundary["transientDetectionSectors"]
        )
        or len(detection_sectors) < MIN_DETECTION_SECTORS
        or not controls
    ):
        raise RuntimeError("Frozen transient detection/control lineage changed.")

    grid = _frequency_grid(method_contract)
    exact_harmonic = float(
        method_contract["models"]["exactHarmonicFrequencyCyclesPerDay"]
    )
    folds = []
    for held_out_sector in detection_sectors:
        held_out = [
            item for item in detection
            if int(item["sector"]) == held_out_sector
        ]
        training = [
            item for item in detection
            if int(item["sector"]) != held_out_sector
        ]
        try:
            if not training or not held_out:
                raise RuntimeError(
                    "Transient fold lacks training or held-out detection windows."
                )
            candidates = []
            for frequency in grid:
                fit = _training_fit(
                    training,
                    (frequency,),
                    selected_frequency_parameters=1,
                )
                candidates.append((fit["bic"], frequency, fit))
            _, learned_frequency, transient_training = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            harmonic_training = _training_fit(training, (exact_harmonic,))
            harmonic_prediction = _aggregate_prediction(
                datasets=held_out,
                frequency=exact_harmonic,
                phase=harmonic_training["learnedPhasesRadians"][0],
            )
            transient_prediction = _aggregate_prediction(
                datasets=held_out,
                frequency=learned_frequency,
                phase=transient_training["learnedPhasesRadians"][0],
                trained_frequency_parameters=1,
            )
            null_prediction = _aggregate_prediction(
                datasets=held_out,
                frequency=None,
                phase=None,
            )
            predictive_bic = {
                "H": harmonic_prediction["bic"],
                "T": transient_prediction["bic"],
                "N": null_prediction["bic"],
            }
            support, reasons = _support(
                harmonic_bic=predictive_bic["H"],
                transient_bic=predictive_bic["T"],
                null_bic=predictive_bic["N"],
                learned_frequency=learned_frequency,
                exact_harmonic_frequency=exact_harmonic,
                frequency_resolution=resolution,
            )
            folds.append({
                "trainingDetectionSectors": sorted({
                    int(item["sector"]) for item in training
                }),
                "trainingDetectionWindowKeys": [
                    {
                        "sector": int(item["sector"]),
                        "windowIndex": int(item["windowIndex"]),
                    }
                    for item in sorted(
                        training,
                        key=lambda value: (
                            int(value["sector"]),
                            int(value["windowIndex"]),
                        ),
                    )
                ],
                "heldOutSector": held_out_sector,
                "heldOutWindowIndices": sorted(
                    int(item["windowIndex"]) for item in held_out
                ),
                "learnedTransientFrequencyCyclesPerDay": learned_frequency,
                "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
                "frequencySeparationCyclesPerDay": abs(
                    learned_frequency - exact_harmonic
                ),
                "longBaselineFrequencyResolutionCyclesPerDay": resolution,
                "trainingBIC": {
                    "H": harmonic_training["bic"],
                    "T": transient_training["bic"],
                },
                "predictiveBIC": predictive_bic,
                "predictiveBICDeltas": {
                    "harmonicOverNull": predictive_bic["N"] - predictive_bic["H"],
                    "transientOverNull": predictive_bic["N"] - predictive_bic["T"],
                    "transientOverHarmonic": predictive_bic["H"] - predictive_bic["T"],
                },
                "support": support,
                "heldOutWindowFits": {
                    "H": harmonic_prediction["windowFits"],
                    "T": transient_prediction["windowFits"],
                    "N": null_prediction["windowFits"],
                },
                "failureOrInsufficiencyReasons": reasons,
            })
        except (RuntimeError, OverflowError, ValueError) as error:
            folds.append({
                "trainingDetectionSectors": sorted({
                    int(item["sector"]) for item in training
                }),
                "heldOutSector": held_out_sector,
                "heldOutWindowIndices": sorted(
                    int(item["windowIndex"]) for item in held_out
                ),
                "learnedTransientFrequencyCyclesPerDay": None,
                "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
                "frequencySeparationCyclesPerDay": None,
                "longBaselineFrequencyResolutionCyclesPerDay": resolution,
                "trainingBIC": None,
                "predictiveBIC": None,
                "predictiveBICDeltas": None,
                "support": "INSUFFICIENT",
                "failureOrInsufficiencyReasons": [
                    f"{type(error).__name__}: {error}"
                ],
            })

    control_results = []
    try:
        final_candidates = []
        for frequency in grid:
            fit = _training_fit(
                detection,
                (frequency,),
                selected_frequency_parameters=1,
            )
            final_candidates.append((fit["bic"], frequency, fit))
        _, final_frequency, final_transient_training = min(
            final_candidates, key=lambda item: (item[0], item[1])
        )
        final_harmonic_training = _training_fit(detection, (exact_harmonic,))
        for control in sorted(
            controls,
            key=lambda item: (
                item["role"] != "PRIMARY_WINDOW",
                int(item["sector"]),
                int(item["windowIndex"]),
                item["datasetID"],
            ),
        ):
            try:
                harmonic_fit = _predictive_fit(
                    control,
                    (exact_harmonic,),
                    (final_harmonic_training["learnedPhasesRadians"][0],),
                )
                transient_fit = _predictive_fit(
                    control,
                    (final_frequency,),
                    (final_transient_training["learnedPhasesRadians"][0],),
                    trained_frequency_parameters=1,
                )
                null_fit = _predictive_fit(control, (), ())
                support, reasons = _support(
                    harmonic_bic=harmonic_fit["bic"],
                    transient_bic=transient_fit["bic"],
                    null_bic=null_fit["bic"],
                    learned_frequency=final_frequency,
                    exact_harmonic_frequency=exact_harmonic,
                    frequency_resolution=resolution,
                )
                control_results.append({
                    "datasetID": control["datasetID"],
                    "role": control["role"],
                    "sector": int(control["sector"]),
                    "windowIndex": int(control["windowIndex"]),
                    "learnedTransientFrequencyCyclesPerDay": final_frequency,
                    "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
                    "predictiveBIC": {
                        "H": harmonic_fit["bic"],
                        "T": transient_fit["bic"],
                        "N": null_fit["bic"],
                    },
                    "predictiveBICDeltas": {
                        "harmonicOverNull": null_fit["bic"] - harmonic_fit["bic"],
                        "transientOverNull": null_fit["bic"] - transient_fit["bic"],
                        "transientOverHarmonic": harmonic_fit["bic"] - transient_fit["bic"],
                    },
                    "support": support,
                    "failureOrInsufficiencyReasons": reasons,
                })
            except (RuntimeError, OverflowError, ValueError) as error:
                control_results.append({
                    "datasetID": control["datasetID"],
                    "role": control["role"],
                    "sector": int(control["sector"]),
                    "windowIndex": int(control["windowIndex"]),
                    "learnedTransientFrequencyCyclesPerDay": final_frequency,
                    "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
                    "predictiveBIC": None,
                    "predictiveBICDeltas": None,
                    "support": "INSUFFICIENT",
                    "failureOrInsufficiencyReasons": [
                        f"{type(error).__name__}: {error}"
                    ],
                })
    except (RuntimeError, OverflowError, ValueError) as error:
        final_frequency = None
        for control in controls:
            control_results.append({
                "datasetID": control["datasetID"],
                "role": control["role"],
                "sector": int(control["sector"]),
                "windowIndex": int(control["windowIndex"]),
                "learnedTransientFrequencyCyclesPerDay": None,
                "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
                "predictiveBIC": None,
                "predictiveBICDeltas": None,
                "support": "INSUFFICIENT",
                "failureOrInsufficiencyReasons": [
                    f"{type(error).__name__}: {error}"
                ],
            })

    decision = classify_transient_validation(folds, control_results)
    return {
        "version": RESULT_VERSION,
        "methodContractID": METHOD_CONTRACT_ID,
        "methodContractHash": contract_hash,
        "methodContract": deepcopy(method_contract),
        "leaveOneTransientDetectionSectorOut": True,
        "controlWindowsUsedForSelection": False,
        "perDetectionSectorEvidence": folds,
        "perControlWindowEvidence": control_results,
        "allDetectionWindowLearnedFrequencyCyclesPerDay": final_frequency,
        "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
        "longBaselineDays": baseline,
        "longBaselineFrequencyResolutionCyclesPerDay": resolution,
        **decision,
        "failureOrInsufficiencyReasons": [
            reason
            for item in [*folds, *control_results]
            for reason in item["failureOrInsufficiencyReasons"]
        ],
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "automaticDiscoveryClaim": False,
        "dataReuse": {
            "frozenWindowDatasetPaths": [
                dataset["datasetPath"] for dataset in datasets
            ],
            "downloadPerformed": False,
            "originalSectorFluxRead": False,
        },
    }
