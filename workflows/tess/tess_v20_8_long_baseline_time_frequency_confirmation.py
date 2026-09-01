"""Long-baseline confirmation of unresolved v20.8 residual variability.

This continuation is intentionally separate from the mode-identification
harmonic-versus-mode confirmation and from v20.9 nonstationary modeling.  It
reuses only the frozen, already family-subtracted v20.8 window datasets.  The
method contract is built and hashed before any window flux is read.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from .tess_long_baseline_frequency_confirmation import (
    _load_frozen_dataset,
    _predictive_fit,
    _training_fit,
)


HANDLER_ID = (
    "openstar.tess.long-baseline-time-frequency-confirmation.analyze"
)
RESULT_VERSION = (
    "openstar.tess-long-baseline-time-frequency-confirmation.v1"
)
METHOD_CONTRACT_ID = (
    "openstar.tess.long-baseline-time-frequency-confirmation."
    "leave-one-independent-sector-out.v1"
)

MIN_BIC_IMPROVEMENT = 10.0
MIN_INDEPENDENT_SECTORS = 3
FREQUENCY_GRID_POINT_COUNT = 101
MIN_ACCEPTED_WINDOW_FRACTION_FOR_NONSTATIONARY = 2.0 / 3.0

COHERENT = "COHERENT_RESIDUAL_FREQUENCY_CONFIRMED"
HARMONIC = "HARMONIC_LOCKED_RESIDUAL_CONFIRMED"
NONSTATIONARY = "NONSTATIONARY_RESIDUAL_STRUCTURE_CONFIRMED"
INTERMITTENT = "INTERMITTENT_RESIDUAL_STRUCTURE_CONFIRMED"
INCONCLUSIVE = "LONG_BASELINE_TIME_FREQUENCY_INCONCLUSIVE"

CLASSIFICATIONS = (
    COHERENT,
    HARMONIC,
    NONSTATIONARY,
    INTERMITTENT,
    INCONCLUSIVE,
)

RECOMMENDED_NEXT_TESTS = {
    COHERENT: "MODE_IDENTIFICATION_OR_PULSATION_MODELING",
    HARMONIC: "DYNAMIC_HARMONIC_MODELING",
    NONSTATIONARY: "LONG_BASELINE_NONSTATIONARY_MODE_MODELING",
    INTERMITTENT: "TRANSIENT_MODE_VALIDATION",
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


def validate_v20_8_boundary(
    *,
    preparation: dict[str, Any],
    interpretation: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact terminal v20.8 nonstationary evidence boundary."""
    if not all(isinstance(value, dict) for value in (
        preparation, interpretation, summary
    )):
        raise RuntimeError("v20.8 confirmation evidence must contain objects.")

    residual = summary.get("residualEvolution") or {}
    period_reference = summary.get("periodReference") or {}
    prepared_windows = preparation.get("preparedWindows") or []
    interpreted_windows = interpretation.get("windowResults") or []
    summary_windows = summary.get("windowResults") or []
    if not all(isinstance(value, list) for value in (
        prepared_windows, interpreted_windows, summary_windows
    )):
        raise RuntimeError("v20.8 window evidence is malformed.")

    period = _finite_positive(
        period_reference.get("periodDays"), "v20.8 family reference period"
    )
    preparation_period = _finite_positive(
        preparation.get("physicalPeriodDays"), "v20.8 preparation period"
    )
    summary_period = _finite_positive(
        summary.get("physicalPeriodDays"), "v20.8 summary period"
    )
    frequency = _finite_positive(
        preparation.get("physicalFrequency"), "v20.8 family frequency"
    )
    interpretation_frequency = _finite_positive(
        interpretation.get("physicalFrequency"),
        "v20.8 interpretation frequency",
    )
    summary_frequency = _finite_positive(
        summary.get("physicalFrequency"), "v20.8 summary frequency"
    )
    try:
        harmonic_orders = tuple(
            int(value) for value in preparation.get("subtractedHarmonicOrders")
        )
    except (TypeError, ValueError):
        raise RuntimeError("v20.8 harmonic-order lineage is invalid.") from None
    if (
        not harmonic_orders
        or any(order < 1 for order in harmonic_orders)
        or len(set(harmonic_orders)) != len(harmonic_orders)
    ):
        raise RuntimeError("v20.8 harmonic-order lineage is invalid.")

    metadata_by_id = {}
    for item in prepared_windows:
        if not isinstance(item, dict) or not item.get("datasetID"):
            raise RuntimeError("v20.8 prepared-window lineage is incomplete.")
        dataset_id = str(item["datasetID"])
        if dataset_id in metadata_by_id or not item.get("datasetPath"):
            raise RuntimeError("v20.8 prepared-window lineage is duplicated.")
        try:
            sector = int(item["sector"])
            window_index = int(item["windowIndex"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                "v20.8 prepared-window sector lineage is invalid."
            ) from None
        if sector < 1 or window_index < 1:
            raise RuntimeError("v20.8 prepared-window indices are invalid.")
        role = str(item.get("role") or "")
        if role not in {
            "primary-time-frequency-window",
            "independent-time-frequency-window",
        }:
            raise RuntimeError("v20.8 prepared-window role is invalid.")
        metadata_by_id[dataset_id] = item

    interpreted_by_id = {}
    accepted_independent = []
    independent_windows = []
    accepted_frequencies = []
    for item in interpreted_windows:
        if not isinstance(item, dict) or not item.get("datasetID"):
            raise RuntimeError("v20.8 interpreted-window lineage is incomplete.")
        dataset_id = str(item["datasetID"])
        metadata = metadata_by_id.get(dataset_id)
        if metadata is None or dataset_id in interpreted_by_id:
            raise RuntimeError("v20.8 interpreted windows do not match preparation.")
        try:
            sector = int(item["sector"])
            window_index = int(item["windowIndex"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("v20.8 interpreted-window lineage is invalid.") from None
        if not (
            sector == int(metadata["sector"])
            and window_index == int(metadata["windowIndex"])
            and item.get("role") == metadata.get("role")
        ):
            raise RuntimeError("v20.8 window target/sector lineage has changed.")
        interpreted_by_id[dataset_id] = item
        if item.get("role") == "independent-time-frequency-window":
            independent_windows.append(item)
        if item.get("acceptedTimeFrequencyFeature") is True:
            candidate_frequency = _finite_positive(
                item.get("candidateFrequency"),
                "Accepted v20.8 residual frequency",
            )
            candidate_period = _finite_positive(
                item.get("candidatePeriodDays"),
                "Accepted v20.8 residual period",
            )
            if not _close(candidate_period, 1.0 / candidate_frequency):
                raise RuntimeError(
                    "Accepted v20.8 residual period/frequency is inconsistent."
                )
            accepted_frequencies.append(candidate_frequency)
            if item.get("role") == "independent-time-frequency-window":
                accepted_independent.append(item)

    if set(interpreted_by_id) != set(metadata_by_id):
        raise RuntimeError("v20.8 prepared/interpreted window sets differ.")
    if summary_windows != interpreted_windows:
        raise RuntimeError("v20.8 summary no longer matches its interpretation.")

    accepted_sectors = sorted({
        int(item["sector"]) for item in accepted_independent
    })
    summary_sectors = sorted(
        int(value) for value in summary.get("acceptedIndependentSectors") or []
    )
    try:
        accepted_count = int(summary.get("acceptedFeatureCount"))
        accepted_independent_count = int(
            summary.get("acceptedIndependentFeatureCount")
        )
        window_count = int(summary.get("windowCount"))
    except (TypeError, ValueError):
        raise RuntimeError("v20.8 accepted-window counts are invalid.") from None

    exact = (
        preparation.get("available") is True
        and summary.get("classification") == "NONSTATIONARY_VARIABILITY"
        and residual.get("classification")
        == "NONSTATIONARY_RESIDUAL_VARIABILITY"
        and summary.get("recommendedNextTest")
        == "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
        and summary.get("physicalMechanismResolved") is False
        and summary.get("claimLevelChanged") is False
        and period_reference.get("kind")
        == "UNRESOLVED_FAMILY_ANALYSIS_REFERENCE"
        and period_reference.get("physicalCycleResolved") is False
        and _close(period, preparation_period)
        and _close(period, summary_period)
        and _close(frequency, 1.0 / period)
        and _close(interpretation_frequency, frequency)
        and _close(summary_frequency, frequency)
        and tuple(harmonic_orders) == (1, 2)
        and window_count == len(summary_windows) == len(prepared_windows)
        and accepted_count == len(accepted_frequencies)
        and accepted_independent_count == len(accepted_independent)
        and summary_sectors == accepted_sectors
        and len(accepted_sectors) >= MIN_INDEPENDENT_SECTORS
        and len(independent_windows) >= len(accepted_independent)
    )
    if not exact:
        raise RuntimeError(
            "Long-baseline time-frequency confirmation requires the exact "
            "terminal unresolved v20.8 NONSTATIONARY_VARIABILITY boundary."
        )

    residual_frequency = float(median(accepted_frequencies))
    tested_order = max(3, int(round(residual_frequency / frequency)))
    exact_harmonic_frequency = tested_order * frequency
    return {
        "familyReferencePeriodDays": period,
        "familyReferenceFrequencyCyclesPerDay": frequency,
        "subtractedHarmonicOrders": list(harmonic_orders),
        "testedHarmonicOrder": tested_order,
        "exactHarmonicFrequencyCyclesPerDay": exact_harmonic_frequency,
        "persistedResidualFrequencyCyclesPerDay": residual_frequency,
        "acceptedFrequenciesCyclesPerDay": accepted_frequencies,
        "acceptedIndependentSectors": accepted_sectors,
        "acceptedIndependentWindowCount": len(accepted_independent),
        "independentWindowCount": len(independent_windows),
        "preparedWindows": deepcopy(prepared_windows),
    }


def build_method_contract(
    *,
    preparation: dict[str, Any],
    interpretation: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Freeze all model and decision choices before reading window flux."""
    evidence = validate_v20_8_boundary(
        preparation=preparation,
        interpretation=interpretation,
        summary=summary,
    )
    anchors = sorted({
        *evidence["acceptedFrequenciesCyclesPerDay"],
        evidence["persistedResidualFrequencyCyclesPerDay"],
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
            "residualEvolutionClassification": (
                summary["residualEvolution"]["classification"]
            ),
            "recommendedNextTest": summary["recommendedNextTest"],
            "physicalMechanismResolved": False,
            "periodReference": deepcopy(summary["periodReference"]),
            "familyReferenceFrequencyCyclesPerDay": evidence[
                "familyReferenceFrequencyCyclesPerDay"
            ],
            "subtractedHarmonicOrders": evidence[
                "subtractedHarmonicOrders"
            ],
            "persistedResidualFrequencyCyclesPerDay": evidence[
                "persistedResidualFrequencyCyclesPerDay"
            ],
            "acceptedIndependentSectors": evidence[
                "acceptedIndependentSectors"
            ],
            "acceptedIndependentWindowCount": evidence[
                "acceptedIndependentWindowCount"
            ],
            "independentWindowCount": evidence["independentWindowCount"],
            "frozenWindowDatasetPaths": [
                item["datasetPath"] for item in evidence["preparedWindows"]
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


def _frequency_grid(contract: dict[str, Any]) -> tuple[float, ...]:
    spec = contract["frequencyGrid"]
    lower = float(spec["inclusiveMinimumCyclesPerDay"])
    upper = float(spec["inclusiveMaximumCyclesPerDay"])
    count = int(spec["pointCount"])
    if _close(lower, upper):
        values = {lower}
    else:
        values = {
            lower + (upper - lower) * index / (count - 1)
            for index in range(count)
        }
    values.update(float(value) for value in
                  spec["mandatoryAnchorFrequenciesCyclesPerDay"])
    return tuple(sorted(value for value in values if value > 0.0))


def _dataset_specs(
    *, expected_tic_id: int, preparation: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    specs = []
    for item in preparation.get("preparedWindows") or []:
        role = (
            "PRIMARY_WINDOW"
            if item.get("role") == "primary-time-frequency-window"
            else "INDEPENDENT_WINDOW"
        )
        specs.append({
            "datasetID": str(item["datasetID"]),
            "datasetPath": str(item["datasetPath"]),
            "ticID": int(expected_tic_id),
            "sector": int(item["sector"]),
            "windowIndex": int(item["windowIndex"]),
            "role": role,
        })
    specs.sort(key=lambda item: (
        item["role"] != "PRIMARY_WINDOW",
        item["sector"],
        item["windowIndex"],
        item["datasetID"],
    ))
    return tuple(specs)


def validate_frozen_window_lineage(
    *,
    method_contract: dict[str, Any],
    dataset_specs: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Read and validate frozen windows only after contract preregistration."""
    specs = tuple(deepcopy(spec) for spec in dataset_specs)
    contract_paths = sorted(
        str(value) for value in method_contract["evidenceBoundary"][
            "frozenWindowDatasetPaths"
        ]
    )
    spec_paths = sorted(str(spec["datasetPath"]) for spec in specs)
    if spec_paths != contract_paths:
        raise RuntimeError(
            "Frozen v20.8 window specifications do not match the method contract."
        )
    datasets = []
    for spec in specs:
        dataset = _load_frozen_dataset(spec)
        path = Path(str(spec["datasetPath"])).expanduser().resolve()
        try:
            with path.open("r", encoding="utf-8") as handle:
                frozen = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Frozen v20.8 window is unreadable: {path}: {error}"
            ) from error
        source = frozen.get("source") or {}
        science = frozen.get("science") or {}
        try:
            source_window_index = int(source["timeFrequencyWindowIndex"])
            science_window_index = int(science["windowIndex"])
            window_start = float(source["windowStartDatasetDays"])
            window_center = float(source["windowCenterDatasetDays"])
            absolute_center = float(source["absoluteWindowCenterDays"])
            source_origin = float(source["originalTimeOriginDays"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                f"Frozen v20.8 window time lineage is incomplete: {path}"
            ) from None
        expected_role = (
            "primary-time-frequency-window"
            if spec["role"] == "PRIMARY_WINDOW"
            else "independent-time-frequency-window"
        )
        if not (
            source_window_index == int(spec["windowIndex"])
            and science_window_index == int(spec["windowIndex"])
            and science.get("purpose")
            == "sliding-window-time-frequency-evolution"
            and science.get("role") == expected_role
            and math.isfinite(window_start)
            and math.isfinite(window_center)
            and math.isfinite(absolute_center)
            and math.isfinite(source_origin)
            and math.isclose(
                source_origin + window_center,
                absolute_center,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                f"Frozen v20.8 window time/role lineage mismatch: {path}"
            )
        # The generic loader restores the sector-level origin.  v20.8 window
        # samples were additionally rebased to their own window start, so add
        # that persisted offset to recover the absolute long-baseline phase.
        dataset["times"] = [
            float(value) + window_start for value in dataset["times"]
        ]
        dataset["windowIndex"] = int(spec["windowIndex"])
        dataset["absoluteWindowCenterDays"] = absolute_center
        datasets.append(dataset)
    independent = [
        item for item in datasets if item["role"] == "INDEPENDENT_WINDOW"
    ]
    independent_sectors = {int(item["sector"]) for item in independent}
    expected_sectors = {
        int(value) for value in method_contract["evidenceBoundary"][
            "acceptedIndependentSectors"
        ]
    }
    keys = {
        (item["role"], int(item["sector"]), int(item["windowIndex"]))
        for item in datasets
    }
    if (
        not any(item["role"] == "PRIMARY_WINDOW" for item in datasets)
        or len(independent_sectors) < MIN_INDEPENDENT_SECTORS
        or not expected_sectors.issubset(independent_sectors)
        or len(keys) != len(datasets)
    ):
        raise RuntimeError("Frozen v20.8 window lineage is inconsistent.")
    return tuple(datasets)


def _aggregate_predictive(
    *,
    held_out: Sequence[dict[str, Any]],
    frequency: float | None,
    phase: float | None,
    selected_frequency_parameters: int = 0,
) -> dict[str, Any]:
    windows = []
    total_bic = 0.0
    total_rss = 0.0
    total_samples = 0
    for dataset in sorted(
        held_out, key=lambda item: (item["windowIndex"], item["datasetID"])
    ):
        frequencies = () if frequency is None else (frequency,)
        phases = () if phase is None else (phase,)
        fit = _predictive_fit(
            dataset,
            frequencies,
            phases,
            trained_frequency_parameters=selected_frequency_parameters,
        )
        total_bic += float(fit["bic"])
        total_rss += float(fit["rss"])
        total_samples += int(fit["sampleCount"])
        windows.append({
            "datasetID": dataset["datasetID"],
            "windowIndex": dataset["windowIndex"],
            **fit,
        })
    return {
        "bic": total_bic,
        "rss": total_rss,
        "sampleCount": total_samples,
        "windowFits": windows,
    }


def _fold_support(
    *,
    harmonic_bic: float,
    coherent_bic: float,
    null_bic: float,
    learned_frequency: float,
    exact_harmonic_frequency: float,
    frequency_resolution: float,
) -> tuple[str, list[str]]:
    harmonic_over_null = null_bic - harmonic_bic
    coherent_over_null = null_bic - coherent_bic
    coherent_over_harmonic = harmonic_bic - coherent_bic
    separated = (
        abs(learned_frequency - exact_harmonic_frequency)
        > frequency_resolution
    )
    if (
        coherent_over_null >= MIN_BIC_IMPROVEMENT
        and coherent_over_harmonic >= 0.0
        and separated
    ):
        return "COHERENT", []
    if (
        harmonic_over_null >= MIN_BIC_IMPROVEMENT
        and coherent_over_harmonic <= 0.0
        and not separated
    ):
        return "HARMONIC", []
    if max(harmonic_over_null, coherent_over_null) >= MIN_BIC_IMPROVEMENT:
        return "STRUCTURED_UNRESOLVED", [
            "PREDICTIVE_STRUCTURE_DOES_NOT_SEPARATE_COHERENT_AND_HARMONIC_MODELS"
        ]
    return "NEITHER", [
        "NO_FIXED_PHASE_RESIDUAL_MODEL_IMPROVES_ON_NULL_BY_THRESHOLD"
    ]


def classify_confirmation(
    fold_results: Sequence[dict[str, Any]],
    *,
    long_baseline_frequency_resolution: float,
    accepted_independent_window_count: int,
    independent_window_count: int,
) -> dict[str, Any]:
    """Apply conservative aggregate rules without consulting held-out choices."""
    resolution = _finite_positive(
        long_baseline_frequency_resolution,
        "Long-baseline frequency resolution",
    )
    sufficient = [
        fold for fold in fold_results if fold.get("support") != "INSUFFICIENT"
    ]
    bics = {
        model: sum(float(fold["predictiveBIC"][model]) for fold in sufficient)
        for model in ("H", "S", "N")
    }
    coherent_count = sum(
        fold.get("support") == "COHERENT" for fold in sufficient
    )
    harmonic_count = sum(
        fold.get("support") == "HARMONIC" for fold in sufficient
    )
    structured_count = sum(
        fold.get("support") in {
            "COHERENT", "HARMONIC", "STRUCTURED_UNRESOLVED"
        }
        for fold in sufficient
    )
    frequencies = [
        float(fold["learnedCoherentFrequencyCyclesPerDay"])
        for fold in sufficient
        if fold.get("learnedCoherentFrequencyCyclesPerDay") is not None
    ]
    frequency_range = max(frequencies) - min(frequencies) if frequencies else None
    stable = bool(
        frequencies
        and len(frequencies) == len(sufficient)
        and frequency_range is not None
        and frequency_range <= resolution
    )
    window_fraction = (
        accepted_independent_window_count / independent_window_count
        if independent_window_count > 0 else 0.0
    )
    enough = len(sufficient) >= MIN_INDEPENDENT_SECTORS
    coherent_aggregate = (
        bics["N"] - bics["S"] >= MIN_BIC_IMPROVEMENT
        and bics["H"] - bics["S"] >= MIN_BIC_IMPROVEMENT
    )
    harmonic_aggregate = (
        bics["N"] - bics["H"] >= MIN_BIC_IMPROVEMENT
        and bics["S"] - bics["H"] >= MIN_BIC_IMPROVEMENT
    )
    persisted_structure = (
        accepted_independent_window_count >= MIN_INDEPENDENT_SECTORS
    )

    if (
        enough
        and coherent_count >= MIN_INDEPENDENT_SECTORS
        and coherent_aggregate
        and stable
    ):
        classification = COHERENT
    elif (
        enough
        and harmonic_count >= MIN_INDEPENDENT_SECTORS
        and harmonic_aggregate
    ):
        classification = HARMONIC
    elif (
        enough
        and persisted_structure
        and window_fraction >= MIN_ACCEPTED_WINDOW_FRACTION_FOR_NONSTATIONARY
        and (
            not stable
            or coherent_count < MIN_INDEPENDENT_SECTORS
            or structured_count < MIN_INDEPENDENT_SECTORS
        )
    ):
        classification = NONSTATIONARY
    elif enough and persisted_structure and window_fraction > 0.0:
        classification = INTERMITTENT
    else:
        classification = INCONCLUSIVE

    aggregate = {
        "predictiveBIC": bics,
        "bicImprovementHarmonicOverNull": bics["N"] - bics["H"],
        "bicImprovementCoherentOverNull": bics["N"] - bics["S"],
        "bicImprovementCoherentOverHarmonic": bics["H"] - bics["S"],
        "sufficientHeldOutSectorCount": len(sufficient),
        "coherentSupportingSectorCount": coherent_count,
        "harmonicSupportingSectorCount": harmonic_count,
        "structuredUnresolvedSectorCount": sum(
            fold.get("support") == "STRUCTURED_UNRESOLVED"
            for fold in sufficient
        ),
        "acceptedIndependentWindowCount": accepted_independent_window_count,
        "independentWindowCount": independent_window_count,
        "acceptedIndependentWindowFraction": window_fraction,
    }
    return {
        "classification": classification,
        "recommendedNextTest": RECOMMENDED_NEXT_TESTS[classification],
        "aggregateDecision": aggregate,
        "frequencyStability": {
            "learnedFrequenciesCyclesPerDay": frequencies,
            "medianFrequencyCyclesPerDay": (
                float(median(frequencies)) if frequencies else None
            ),
            "rangeCyclesPerDay": frequency_range,
            "maximumAllowedRangeCyclesPerDay": resolution,
            "stableWithinLongBaselineResolution": stable,
        },
    }


def analyze_long_baseline_time_frequency_confirmation(
    *,
    method_contract: dict[str, Any],
    dataset_specs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Run leave-one-independent-sector-out fixed-phase prediction."""
    if method_contract.get("methodContractID") != METHOD_CONTRACT_ID:
        raise RuntimeError("Unsupported v20.8 confirmation method contract.")
    contract_hash = method_contract_hash(method_contract)
    datasets = validate_frozen_window_lineage(
        method_contract=method_contract,
        dataset_specs=dataset_specs,
    )
    all_times = [time for dataset in datasets for time in dataset["times"]]
    baseline = max(all_times) - min(all_times)
    if baseline <= 0.0:
        raise RuntimeError("Frozen v20.8 windows have no positive long baseline.")
    resolution = 1.0 / baseline
    primary = [item for item in datasets if item["role"] == "PRIMARY_WINDOW"]
    independent = [
        item for item in datasets if item["role"] == "INDEPENDENT_WINDOW"
    ]
    sectors = sorted({int(item["sector"]) for item in independent})
    exact_harmonic = float(
        method_contract["models"]["exactHarmonicFrequencyCyclesPerDay"]
    )
    grid = _frequency_grid(method_contract)
    folds = []
    for held_out_sector in sectors:
        held_out = [
            item for item in independent
            if int(item["sector"]) == held_out_sector
        ]
        training = [
            *primary,
            *(item for item in independent
              if int(item["sector"]) != held_out_sector),
        ]
        try:
            candidates = []
            for frequency in grid:
                fit = _training_fit(
                    training,
                    (frequency,),
                    selected_frequency_parameters=1,
                )
                candidates.append((fit["bic"], frequency, fit))
            _, learned_frequency, training_coherent = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            training_harmonic = _training_fit(training, (exact_harmonic,))
            harmonic_prediction = _aggregate_predictive(
                held_out=held_out,
                frequency=exact_harmonic,
                phase=training_harmonic["learnedPhasesRadians"][0],
            )
            coherent_prediction = _aggregate_predictive(
                held_out=held_out,
                frequency=learned_frequency,
                phase=training_coherent["learnedPhasesRadians"][0],
                selected_frequency_parameters=1,
            )
            null_prediction = _aggregate_predictive(
                held_out=held_out,
                frequency=None,
                phase=None,
            )
            predictive_bic = {
                "H": harmonic_prediction["bic"],
                "S": coherent_prediction["bic"],
                "N": null_prediction["bic"],
            }
            support, reasons = _fold_support(
                harmonic_bic=predictive_bic["H"],
                coherent_bic=predictive_bic["S"],
                null_bic=predictive_bic["N"],
                learned_frequency=learned_frequency,
                exact_harmonic_frequency=exact_harmonic,
                frequency_resolution=resolution,
            )
            folds.append({
                "trainingSectors": sorted({
                    int(item["sector"]) for item in training
                }),
                "heldOutSector": held_out_sector,
                "heldOutWindowIndices": sorted(
                    int(item["windowIndex"]) for item in held_out
                ),
                "learnedCoherentFrequencyCyclesPerDay": learned_frequency,
                "exactHarmonicFrequencyCyclesPerDay": exact_harmonic,
                "frequencySeparationCyclesPerDay": abs(
                    learned_frequency - exact_harmonic
                ),
                "longBaselineFrequencyResolutionCyclesPerDay": resolution,
                "trainingBIC": {
                    "H": training_harmonic["bic"],
                    "S": training_coherent["bic"],
                },
                "predictiveBIC": predictive_bic,
                "predictiveBICDeltas": {
                    "harmonicOverNull": (
                        predictive_bic["N"] - predictive_bic["H"]
                    ),
                    "coherentOverNull": (
                        predictive_bic["N"] - predictive_bic["S"]
                    ),
                    "coherentOverHarmonic": (
                        predictive_bic["H"] - predictive_bic["S"]
                    ),
                },
                "support": support,
                "heldOutWindowFits": {
                    "H": harmonic_prediction["windowFits"],
                    "S": coherent_prediction["windowFits"],
                    "N": null_prediction["windowFits"],
                },
                "failureOrInsufficiencyReasons": reasons,
            })
        except (RuntimeError, OverflowError, ValueError) as error:
            folds.append({
                "trainingSectors": sorted({
                    int(item["sector"]) for item in training
                }),
                "heldOutSector": held_out_sector,
                "heldOutWindowIndices": sorted(
                    int(item["windowIndex"]) for item in held_out
                ),
                "learnedCoherentFrequencyCyclesPerDay": None,
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

    boundary = method_contract["evidenceBoundary"]
    decision = classify_confirmation(
        folds,
        long_baseline_frequency_resolution=resolution,
        accepted_independent_window_count=int(
            boundary["acceptedIndependentWindowCount"]
        ),
        independent_window_count=int(boundary["independentWindowCount"]),
    )
    return {
        "version": RESULT_VERSION,
        "methodContractID": METHOD_CONTRACT_ID,
        "methodContractHash": contract_hash,
        "methodContract": deepcopy(method_contract),
        "leaveOneIndependentSectorOut": True,
        "perSectorEvidence": folds,
        "longBaselineDays": baseline,
        "longBaselineFrequencyResolutionCyclesPerDay": resolution,
        **decision,
        "failureOrInsufficiencyReasons": [
            reason
            for fold in folds
            for reason in fold["failureOrInsufficiencyReasons"]
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


def build_dataset_specs(
    *, expected_tic_id: int, preparation: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Expose deterministic frozen-window specifications to workflow gates."""
    return _dataset_specs(
        expected_tic_id=expected_tic_id,
        preparation=preparation,
    )
