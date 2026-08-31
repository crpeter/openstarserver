"""Predictive long-baseline confirmation of an ambiguous TESS residual.

The analysis in this module is server-owned and network-free.  Its method
contract is constructed and hashed before any frozen flux values are read.
Each independent sector is then held out in turn.  Frequencies and phases are
learned from the remaining frozen sectors; only an offset and signed component
amplitudes are fitted to the held-out sector.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


HANDLER_ID = "openstar.tess.long-baseline-frequency-confirmation.analyze"
METHOD_CONTRACT_ID = (
    "openstar.tess.long-baseline-frequency-confirmation.leave-one-sector-out.v1"
)
METHOD_CONTRACT_VERSION = 1
MIN_BIC_IMPROVEMENT = 10.0
MIN_SUPPORTING_INDEPENDENT_SECTORS = 3
FREQUENCY_GRID_POINT_COUNT = 101

CLASSIFICATIONS = (
    "INDEPENDENT_STABLE_MODE_CONFIRMED",
    "HARMONIC_LOCKED_ACROSS_BASELINE",
    "NONSTATIONARY_OR_INTERMITTENT_STRUCTURE",
    "LONG_BASELINE_CONFIRMATION_INCONCLUSIVE",
)

RECOMMENDED_NEXT_TESTS = {
    "INDEPENDENT_STABLE_MODE_CONFIRMED": "RESIDUAL_MODE_PIXEL_LOCALIZATION",
    "HARMONIC_LOCKED_ACROSS_BASELINE": "BINARY_ROTATION_EXTERNAL_EVIDENCE",
    "NONSTATIONARY_OR_INTERMITTENT_STRUCTURE": (
        "LONG_BASELINE_NONSTATIONARY_MODE_MODELING"
    ),
    "LONG_BASELINE_CONFIRMATION_INCONCLUSIVE": None,
}


def _finite_positive(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{label} must be finite and positive.") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise RuntimeError(f"{label} must be finite and positive.")
    return parsed


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def method_contract_hash(contract: dict[str, Any]) -> str:
    """Return the repository-compatible canonical SHA-256 JSON hash."""
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_ambiguous_mode_identification(
    mode_identification: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the complete persisted ambiguity is self-consistent."""
    if not isinstance(mode_identification, dict):
        raise RuntimeError("Mode-identification evidence must be an object.")
    if not (
        mode_identification.get("classification")
        == "AMBIGUOUS_HARMONIC_OR_MODE"
        and mode_identification.get("recommendedNextTest")
        == "LONG_BASELINE_TIME_FREQUENCY_CONFIRMATION"
        and mode_identification.get("physicalMechanismResolved") is False
        and mode_identification.get("independentModeEvidenceSurvived") is False
        and mode_identification.get("modeCandidate") is None
    ):
        raise RuntimeError(
            "Long-baseline confirmation requires the exact unresolved "
            "AMBIGUOUS_HARMONIC_OR_MODE recommendation."
        )

    family = mode_identification.get("establishedPeriodFamily") or {}
    candidate = mode_identification.get("residualCandidate") or {}
    relation = mode_identification.get("harmonicRelation") or {}
    comparison = mode_identification.get("modelComparison") or {}
    support = mode_identification.get("independentSectorSupport") or {}
    data_reuse = mode_identification.get("dataReuse") or {}

    period = _finite_positive(
        family.get("referencePeriodDays"), "Established reference period"
    )
    family_frequency = _finite_positive(
        family.get("referenceFrequencyCyclesPerDay"),
        "Established reference frequency",
    )
    if not _close(family_frequency, 1.0 / period):
        raise RuntimeError("Established period/frequency lineage is inconsistent.")

    try:
        orders = tuple(int(value) for value in family.get("modeledHarmonicOrders"))
        tested_order = int(relation.get("testedOrder"))
    except (TypeError, ValueError):
        raise RuntimeError("Persisted harmonic orders are invalid.") from None
    if (
        not orders
        or any(order < 1 for order in orders)
        or len(set(orders)) != len(orders)
        or tested_order < 3
        or tested_order not in orders
    ):
        raise RuntimeError("Persisted harmonic-order lineage is inconsistent.")
    established_orders = tuple(order for order in orders if order != tested_order)
    if not established_orders:
        raise RuntimeError("The established family cannot be empty.")

    measured_period = _finite_positive(
        candidate.get("measuredPeriodDays"), "Measured residual period"
    )
    measured_frequency = _finite_positive(
        candidate.get("measuredFrequencyCyclesPerDay"),
        "Measured residual frequency",
    )
    refined_period = _finite_positive(
        candidate.get("refinedPeriodDays"), "Refined residual period"
    )
    refined_frequency = _finite_positive(
        candidate.get("refinedFrequencyCyclesPerDay"),
        "Refined residual frequency",
    )
    harmonic_frequency = _finite_positive(
        relation.get("harmonicFrequencyCyclesPerDay"),
        "Tested harmonic frequency",
    )
    try:
        separation = float(relation.get("absoluteFrequencySeparation"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "Persisted frequency separation must be finite and non-negative."
        ) from None
    if not math.isfinite(separation) or separation < 0:
        raise RuntimeError(
            "Persisted frequency separation must be finite and non-negative."
        )
    resolution = _finite_positive(
        relation.get("frequencyResolutionCyclesPerDay"),
        "Persisted frequency resolution",
    )
    baseline = _finite_positive(
        relation.get("baselineDays"), "Persisted frequency baseline"
    )
    if not (
        _close(measured_period, 1.0 / measured_frequency)
        and _close(refined_period, 1.0 / refined_frequency)
        and _close(harmonic_frequency, tested_order * family_frequency)
        and _close(separation, abs(measured_frequency - harmonic_frequency))
        and _close(resolution, 1.0 / baseline)
        and relation.get("commensurateWithinResolution")
        is (separation <= resolution)
    ):
        raise RuntimeError("Persisted residual/harmonic evidence is inconsistent.")

    models = comparison.get("models") or {}
    try:
        family_bic = float((models.get("establishedFamily") or {})["bic"])
        harmonic_bic = float((models.get("extendedHigherHarmonics") or {})["bic"])
        independent_bic = float(
            (models.get("familyPlusIndependentFreeFrequency") or {})["bic"]
        )
        threshold = float(comparison.get("conservativeThreshold"))
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("Persisted mode model comparison is incomplete.") from None
    if not all(math.isfinite(value) for value in (
        family_bic, harmonic_bic, independent_bic, threshold
    )):
        raise RuntimeError("Persisted mode BIC values must be finite.")
    try:
        extended_over_family = float(
            comparison.get("bicImprovementExtendedOverFamily")
        )
        independent_over_family = float(
            comparison.get("bicImprovementIndependentOverFamily")
        )
        independent_over_extended = float(
            comparison.get("bicImprovementIndependentOverExtended")
        )
    except (TypeError, ValueError):
        raise RuntimeError("Persisted mode BIC lineage is incomplete.") from None
    if not (
        comparison.get("criterion") == "BIC"
        and _close(threshold, MIN_BIC_IMPROVEMENT)
        and _close(extended_over_family, family_bic - harmonic_bic)
        and _close(independent_over_family, family_bic - independent_bic)
        and _close(independent_over_extended, harmonic_bic - independent_bic)
    ):
        raise RuntimeError("Persisted mode BIC lineage is inconsistent.")

    commensurate = bool(relation.get("commensurateWithinResolution"))
    independent_survives = (
        not commensurate
        and family_bic - independent_bic >= MIN_BIC_IMPROVEMENT
        and harmonic_bic - independent_bic >= MIN_BIC_IMPROVEMENT
    )
    harmonic_wins = (
        commensurate
        and family_bic - harmonic_bic >= MIN_BIC_IMPROVEMENT
        and harmonic_bic - independent_bic <= MIN_BIC_IMPROVEMENT
    )
    no_compelling_residual = max(
        family_bic - harmonic_bic,
        family_bic - independent_bic,
    ) < MIN_BIC_IMPROVEMENT
    if independent_survives or harmonic_wins or no_compelling_residual:
        raise RuntimeError(
            "Persisted BIC evidence does not produce the declared ambiguity."
        )

    try:
        sectors = tuple(sorted(int(value) for value in support.get("sectors")))
        count = int(support.get("count"))
        required = int(support.get("requiredCount"))
    except (TypeError, ValueError):
        raise RuntimeError("Independent-sector support is invalid.") from None
    if not (
        len(set(sectors)) == len(sectors)
        and count == len(sectors)
        and required == MIN_SUPPORTING_INDEPENDENT_SECTORS
        and support.get("sufficient") is (count >= required)
        and count >= MIN_SUPPORTING_INDEPENDENT_SECTORS
    ):
        raise RuntimeError(
            "At least three internally consistent independent sectors are required."
        )

    paths = data_reuse.get("frozenDatasetPaths")
    if not (
        isinstance(paths, list)
        and len(paths) >= count + 1
        and all(isinstance(path, str) and path for path in paths)
        and len(set(paths)) == len(paths)
        and data_reuse.get("downloadPerformed") is False
    ):
        raise RuntimeError("Frozen mode-identification dataset lineage is invalid.")

    return {
        "establishedPeriodDays": period,
        "establishedFamilyFrequencyCyclesPerDay": family_frequency,
        "establishedFamilyHarmonicOrders": list(established_orders),
        "persistedModeledHarmonicOrders": list(orders),
        "measuredResidualFrequencyCyclesPerDay": measured_frequency,
        "refinedResidualFrequencyCyclesPerDay": refined_frequency,
        "testedHarmonicOrder": tested_order,
        "testedHarmonicFrequencyCyclesPerDay": harmonic_frequency,
        "independentSectors": list(sectors),
        "frozenDatasetPaths": list(paths),
    }


def build_method_contract(
    mode_identification: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the deterministic analysis choices without reading photometry."""
    evidence = validate_ambiguous_mode_identification(mode_identification)
    anchors = sorted({
        evidence["measuredResidualFrequencyCyclesPerDay"],
        evidence["refinedResidualFrequencyCyclesPerDay"],
        evidence["testedHarmonicFrequencyCyclesPerDay"],
    })
    return {
        "methodContractID": METHOD_CONTRACT_ID,
        "methodContractVersion": METHOD_CONTRACT_VERSION,
        "execution": "PYTHON_SERVER",
        "networkAccess": False,
        "dataPolicy": {
            "reuseFrozenPrimaryAndIndependentSectorDatasets": True,
            "downloadNewData": False,
            "constructAndHashContractBeforeReadingFlux": True,
        },
        "evidenceBoundary": {
            "classification": mode_identification["classification"],
            "physicalMechanismResolved": False,
            "independentModeEvidenceSurvived": False,
            "modeCandidate": None,
            "recommendedNextTest": mode_identification["recommendedNextTest"],
            "establishedPeriodFamily": deepcopy(
                mode_identification["establishedPeriodFamily"]
            ),
            "residualCandidate": deepcopy(mode_identification["residualCandidate"]),
            "harmonicRelation": deepcopy(mode_identification["harmonicRelation"]),
            "modelComparison": deepcopy(mode_identification["modelComparison"]),
            "independentSectorSupport": deepcopy(
                mode_identification["independentSectorSupport"]
            ),
            "frozenDatasetPaths": list(evidence["frozenDatasetPaths"]),
        },
        "crossValidation": {
            "scheme": "LEAVE_ONE_INDEPENDENT_SECTOR_OUT",
            "trainingIncludesFrozenPrimarySector": True,
            "frequencySelectionUsesTrainingSectorsOnly": True,
            "phaseLearningUsesTrainingSectorsOnly": True,
            "heldOutFittedNuisanceParameters": [
                "offset",
                "signedAmplitudePerFixedComponent",
            ],
            "heldOutFrequencySelection": False,
            "heldOutPhaseSelection": False,
        },
        "models": {
            "A": "ESTABLISHED_FAMILY_PLUS_EXACT_TESTED_HARMONIC",
            "B": "ESTABLISHED_FAMILY_PLUS_TRAINING_SELECTED_COHERENT_FREQUENCY",
            "C": "ESTABLISHED_FAMILY_ONLY_NULL",
            "establishedFamilyHarmonicOrders": list(
                evidence["establishedFamilyHarmonicOrders"]
            ),
            "exactTestedHarmonicFrequencyCyclesPerDay": evidence[
                "testedHarmonicFrequencyCyclesPerDay"
            ],
            "predictiveBICTrainedFrequencyParameterCounts": {
                "A": 0,
                "B": 1,
                "C": 0,
            },
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
            "minimumSupportingIndependentSectors": (
                MIN_SUPPORTING_INDEPENDENT_SECTORS
            ),
            "positiveInterpretationRequiresAggregatePredictiveSupport": True,
            "frequencyStabilityTolerance": (
                "ONE_LONG_BASELINE_RAYLEIGH_RESOLUTION"
            ),
            "classifications": list(CLASSIFICATIONS),
        },
    }


def _load_frozen_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(spec["datasetPath"])).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as handle:
            dataset = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Frozen dataset is unreadable: {path}: {error}") from error
    if not isinstance(dataset, dict):
        raise RuntimeError(f"Frozen dataset is not an object: {path}")

    source = dataset.get("source") or {}
    metadata = dataset.get("metadata") or {}
    if not isinstance(source, dict) or not isinstance(metadata, dict):
        raise RuntimeError(
            f"Frozen dataset target/sector lineage is incomplete: {path}"
        )
    expected_id = str(spec["datasetID"])
    expected_tic = int(spec["ticID"])
    expected_sector = int(spec["sector"])
    try:
        observed_id = str(dataset["id"])
        tic_values = [
            int(container["ticID"])
            for container in (source, metadata)
            if container.get("ticID") is not None
        ]
        sector_values = [
            int(container["sector"])
            for container in (source, metadata)
            if container.get("sector") is not None
        ]
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(
            f"Frozen dataset target/sector lineage is incomplete: {path}"
        ) from None
    if not tic_values or not sector_values:
        raise RuntimeError(
            f"Frozen dataset target/sector lineage is incomplete: {path}"
        )
    if len(set(tic_values)) != 1 or len(set(sector_values)) != 1:
        raise RuntimeError(
            f"Frozen dataset target/sector lineage mismatch: {path}"
        )
    observed_tic = tic_values[0]
    observed_sector = sector_values[0]
    if (
        observed_id != expected_id
        or observed_tic != expected_tic
        or observed_sector != expected_sector
    ):
        raise RuntimeError(
            f"Frozen dataset target/sector lineage mismatch: {path}"
        )

    raw_times = dataset.get("times")
    raw_flux = dataset.get("flux")
    if not isinstance(raw_times, list) or not isinstance(raw_flux, list):
        raise RuntimeError(f"Frozen dataset times/flux are missing: {path}")
    if len(raw_times) != len(raw_flux) or len(raw_times) < 16:
        raise RuntimeError(f"Frozen dataset has invalid sample counts: {path}")
    origins = []
    try:
        for container in (source, metadata):
            origin = container.get("originalTimeOriginDays")
            if origin is None:
                origin = container.get("timeOriginDays")
            if origin is not None:
                origins.append(float(origin))
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Frozen dataset has an invalid time origin: {path}"
        ) from None
    if origins and any(
        not math.isclose(value, origins[0], rel_tol=1e-12, abs_tol=1e-12)
        for value in origins[1:]
    ):
        raise RuntimeError(
            f"Frozen dataset target/sector lineage mismatch: {path}"
        )
    time_offset = origins[0] if origins else 0.0
    if not math.isfinite(time_offset):
        raise RuntimeError(f"Frozen dataset has an invalid time origin: {path}")
    try:
        times = [float(value) + time_offset for value in raw_times]
        flux = [float(value) for value in raw_flux]
    except (TypeError, ValueError):
        raise RuntimeError(f"Frozen dataset contains non-numeric samples: {path}") from None
    if not (
        all(math.isfinite(value) for value in times)
        and all(math.isfinite(value) for value in flux)
    ):
        raise RuntimeError(f"Frozen dataset contains non-finite samples: {path}")
    if max(times) <= min(times):
        raise RuntimeError(f"Frozen dataset has no positive time baseline: {path}")
    return {
        "datasetID": expected_id,
        "datasetPath": str(path),
        "ticID": expected_tic,
        "sector": expected_sector,
        "role": str(spec["role"]),
        "times": times,
        "flux": flux,
        "sampleCount": len(times),
    }


def validate_frozen_dataset_lineage(
    *,
    method_contract: dict[str, Any],
    dataset_specs: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate frozen files after the caller has preregistered the contract."""
    specs = tuple(deepcopy(spec) for spec in dataset_specs)
    contract_paths = tuple(
        str(Path(path).expanduser().resolve())
        for path in method_contract["evidenceBoundary"]["frozenDatasetPaths"]
    )
    spec_paths = tuple(
        str(Path(str(spec.get("datasetPath"))).expanduser().resolve())
        for spec in specs
    )
    if spec_paths != contract_paths:
        raise RuntimeError(
            "Frozen dataset specifications do not match the method contract."
        )
    datasets = tuple(_load_frozen_dataset(spec) for spec in specs)
    primary = [dataset for dataset in datasets if dataset["role"] == "PRIMARY"]
    independent = [
        dataset for dataset in datasets if dataset["role"] == "INDEPENDENT"
    ]
    expected_sectors = method_contract["evidenceBoundary"][
        "independentSectorSupport"
    ]["sectors"]
    independent_sectors = {dataset["sector"] for dataset in independent}
    if (
        len(primary) != 1
        or len(independent) < MIN_SUPPORTING_INDEPENDENT_SECTORS
        or not {int(value) for value in expected_sectors}.issubset(
            independent_sectors
        )
        or len({dataset["sector"] for dataset in datasets}) != len(datasets)
    ):
        raise RuntimeError("Frozen primary/independent sector lineage is inconsistent.")
    return datasets


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(len(vector)):
        pivot = max(
            range(column, len(vector)),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise RuntimeError("Long-baseline predictive design is singular.")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(len(vector)):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * other
                for value, other in zip(augmented[row], augmented[column])
            ]
    return [row[-1] for row in augmented]


def _least_squares(rows: Sequence[Sequence[float]], values: Sequence[float]):
    size = len(rows[0])
    normal = [
        [sum(row[i] * row[j] for row in rows) for j in range(size)]
        for i in range(size)
    ]
    rhs = [sum(row[i] * value for row, value in zip(rows, values))
           for i in range(size)]
    coefficients = _solve(normal, rhs)
    rss = sum(
        (value - sum(coefficient * basis for coefficient, basis
                     in zip(coefficients, row))) ** 2
        for row, value in zip(rows, values)
    )
    return coefficients, rss


def _training_fit(
    datasets: Sequence[dict[str, Any]],
    frequencies: Sequence[float],
    *,
    selected_frequency_parameters: int = 0,
) -> dict[str, Any]:
    """Fit shared phases/frequencies using training sectors only."""
    rows: list[list[float]] = []
    values: list[float] = []
    for dataset_index, dataset in enumerate(datasets):
        for time, flux in zip(dataset["times"], dataset["flux"]):
            row = [1.0 if index == dataset_index else 0.0
                   for index in range(len(datasets))]
            for frequency in frequencies:
                angle = 2.0 * math.pi * frequency * time
                row.extend((math.sin(angle), math.cos(angle)))
            rows.append(row)
            values.append(flux)
    coefficients, rss = _least_squares(rows, values)
    phases = []
    start = len(datasets)
    for index in range(len(frequencies)):
        sine = coefficients[start + 2 * index]
        cosine = coefficients[start + 2 * index + 1]
        phases.append(math.atan2(cosine, sine))
    samples = len(values)
    parameters = (
        len(datasets) + 2 * len(frequencies) + selected_frequency_parameters
    )
    bic = samples * math.log(max(rss, 1e-300) / samples) + parameters * math.log(samples)
    return {
        "bic": float(bic),
        "rss": float(rss),
        "sampleCount": samples,
        "parameterCount": parameters,
        "learnedPhasesRadians": phases,
    }


def _predictive_fit(
    held_out: dict[str, Any],
    frequencies: Sequence[float],
    phases: Sequence[float],
    *,
    trained_frequency_parameters: int = 0,
) -> dict[str, Any]:
    rows = []
    for time in held_out["times"]:
        row = [1.0]
        for frequency, phase in zip(frequencies, phases):
            row.append(math.sin(2.0 * math.pi * frequency * time + phase))
        rows.append(row)
    coefficients, rss = _least_squares(rows, held_out["flux"])
    samples = len(rows)
    parameters = len(coefficients) + trained_frequency_parameters
    bic = samples * math.log(max(rss, 1e-300) / samples) + parameters * math.log(samples)
    return {
        "bic": float(bic),
        "rss": float(rss),
        "sampleCount": samples,
        "parameterCount": parameters,
        "fittedOffset": coefficients[0],
        "fittedSignedAmplitudes": coefficients[1:],
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
    values.update(float(value) for value in spec[
        "mandatoryAnchorFrequenciesCyclesPerDay"
    ])
    return tuple(sorted(value for value in values if value > 0))


def _sector_support(
    *,
    harmonic_bic: float,
    independent_bic: float,
    null_bic: float,
    learned_frequency: float,
    exact_harmonic_frequency: float,
    frequency_resolution: float,
) -> tuple[str, list[str]]:
    threshold = MIN_BIC_IMPROVEMENT
    harmonic_over_null = null_bic - harmonic_bic
    independent_over_null = null_bic - independent_bic
    independent_over_harmonic = harmonic_bic - independent_bic
    separated = (
        abs(learned_frequency - exact_harmonic_frequency)
        > frequency_resolution
    )
    if (
        independent_over_null >= threshold
        and separated
        and independent_over_harmonic >= 0.0
    ):
        return "INDEPENDENT_MODE", []
    if (
        harmonic_over_null >= threshold
        and not separated
        and independent_over_harmonic <= 0.0
    ):
        return "HARMONIC", []
    reasons = []
    if max(harmonic_over_null, independent_over_null) < threshold:
        reasons.append("NEITHER_RESIDUAL_MODEL_IMPROVES_ON_NULL_BY_THRESHOLD")
    else:
        reasons.append("RESIDUAL_MODELS_NOT_SEPARATED_BY_THRESHOLD")
    return "NEITHER", reasons


def classify_long_baseline_confirmation(
    fold_results: Sequence[dict[str, Any]],
    *,
    long_baseline_frequency_resolution: float,
) -> dict[str, Any]:
    """Apply the preregistered conservative aggregate decision rules."""
    resolution = _finite_positive(
        long_baseline_frequency_resolution, "Long-baseline frequency resolution"
    )
    sufficient = [fold for fold in fold_results
                  if fold.get("support") != "INSUFFICIENT"]
    harmonic_count = sum(fold.get("support") == "HARMONIC" for fold in sufficient)
    independent_count = sum(
        fold.get("support") == "INDEPENDENT_MODE" for fold in sufficient
    )
    frequencies = [
        float(fold["learnedIndependentFrequencyCyclesPerDay"])
        for fold in sufficient
        if fold.get("learnedIndependentFrequencyCyclesPerDay") is not None
    ]
    frequency_range = max(frequencies) - min(frequencies) if frequencies else None
    frequency_median = median(frequencies) if frequencies else None
    stable = bool(
        frequencies
        and len(frequencies) == len(sufficient)
        and frequency_range is not None
        and frequency_range <= resolution
    )

    aggregate = {
        "predictiveBIC": {
            model: sum(float(fold["predictiveBIC"][model]) for fold in sufficient)
            for model in ("A", "B", "C")
        }
    }
    bics = aggregate["predictiveBIC"]
    aggregate["bicImprovementHarmonicOverNull"] = bics["C"] - bics["A"]
    aggregate["bicImprovementIndependentOverNull"] = bics["C"] - bics["B"]
    aggregate["bicImprovementIndependentOverHarmonic"] = bics["A"] - bics["B"]
    aggregate["sufficientHeldOutSectorCount"] = len(sufficient)
    aggregate["harmonicSupportingSectorCount"] = harmonic_count
    aggregate["independentSupportingSectorCount"] = independent_count

    enough_folds = len(sufficient) >= MIN_SUPPORTING_INDEPENDENT_SECTORS
    independent_aggregate = (
        aggregate["bicImprovementIndependentOverNull"] >= MIN_BIC_IMPROVEMENT
        and aggregate["bicImprovementIndependentOverHarmonic"]
        >= MIN_BIC_IMPROVEMENT
    )
    harmonic_aggregate = (
        aggregate["bicImprovementHarmonicOverNull"] >= MIN_BIC_IMPROVEMENT
        and -aggregate["bicImprovementIndependentOverHarmonic"]
        >= MIN_BIC_IMPROVEMENT
    )
    if (
        enough_folds
        and independent_count >= MIN_SUPPORTING_INDEPENDENT_SECTORS
        and independent_aggregate
        and stable
    ):
        classification = "INDEPENDENT_STABLE_MODE_CONFIRMED"
    elif (
        enough_folds
        and harmonic_count >= MIN_SUPPORTING_INDEPENDENT_SECTORS
        and harmonic_aggregate
    ):
        classification = "HARMONIC_LOCKED_ACROSS_BASELINE"
    else:
        structured = (
            enough_folds
            and (
                aggregate["bicImprovementHarmonicOverNull"]
                >= MIN_BIC_IMPROVEMENT
                or aggregate["bicImprovementIndependentOverNull"]
                >= MIN_BIC_IMPROVEMENT
                or harmonic_count + independent_count > 0
            )
        )
        nonstationary = structured and (
            not stable
            or (
                harmonic_count > 0
                and independent_count > 0
            )
            or max(harmonic_count, independent_count)
            < MIN_SUPPORTING_INDEPENDENT_SECTORS
        )
        classification = (
            "NONSTATIONARY_OR_INTERMITTENT_STRUCTURE"
            if nonstationary
            else "LONG_BASELINE_CONFIRMATION_INCONCLUSIVE"
        )

    return {
        "classification": classification,
        "recommendedNextTest": RECOMMENDED_NEXT_TESTS[classification],
        "aggregateDecision": aggregate,
        "frequencyStability": {
            "learnedFrequenciesCyclesPerDay": frequencies,
            "medianFrequencyCyclesPerDay": frequency_median,
            "rangeCyclesPerDay": frequency_range,
            "maximumAllowedRangeCyclesPerDay": resolution,
            "stableWithinLongBaselineResolution": stable,
        },
    }


def analyze_long_baseline_frequency_confirmation(
    *,
    method_contract: dict[str, Any],
    dataset_specs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Run deterministic leave-one-independent-sector-out prediction."""
    expected_hash = method_contract_hash(method_contract)
    if method_contract.get("methodContractID") != METHOD_CONTRACT_ID:
        raise RuntimeError("Unsupported long-baseline method contract.")
    specs = tuple(deepcopy(spec) for spec in dataset_specs)
    datasets = validate_frozen_dataset_lineage(
        method_contract=method_contract,
        dataset_specs=specs,
    )
    primary = [dataset for dataset in datasets if dataset["role"] == "PRIMARY"]
    independent = [
        dataset for dataset in datasets if dataset["role"] == "INDEPENDENT"
    ]

    all_times = [time for dataset in datasets for time in dataset["times"]]
    baseline = max(all_times) - min(all_times)
    if baseline <= 0:
        raise RuntimeError("Frozen datasets have no positive long baseline.")
    resolution = 1.0 / baseline
    family = method_contract["evidenceBoundary"]["establishedPeriodFamily"]
    family_frequency = float(family["referenceFrequencyCyclesPerDay"])
    tested_frequency = float(
        method_contract["models"]["exactTestedHarmonicFrequencyCyclesPerDay"]
    )
    family_frequencies = tuple(
        family_frequency * int(order)
        for order in method_contract["models"][
            "establishedFamilyHarmonicOrders"
        ]
    )
    grid = _frequency_grid(method_contract)

    folds = []
    for held_out in sorted(independent, key=lambda item: item["sector"]):
        training = [primary[0], *(
            dataset for dataset in independent
            if dataset["sector"] != held_out["sector"]
        )]
        try:
            candidates = []
            for frequency in grid:
                fit = _training_fit(
                    training,
                    (*family_frequencies, frequency),
                    selected_frequency_parameters=1,
                )
                candidates.append((fit["bic"], frequency, fit))
            _, learned_frequency, training_b = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            training_a = _training_fit(
                training, (*family_frequencies, tested_frequency)
            )
            training_c = _training_fit(training, family_frequencies)
            predictive_a = _predictive_fit(
                held_out,
                (*family_frequencies, tested_frequency),
                training_a["learnedPhasesRadians"],
            )
            predictive_b = _predictive_fit(
                held_out,
                (*family_frequencies, learned_frequency),
                training_b["learnedPhasesRadians"],
                trained_frequency_parameters=1,
            )
            predictive_c = _predictive_fit(
                held_out,
                family_frequencies,
                training_c["learnedPhasesRadians"],
            )
            bics = {
                "A": predictive_a["bic"],
                "B": predictive_b["bic"],
                "C": predictive_c["bic"],
            }
            support, reasons = _sector_support(
                harmonic_bic=bics["A"],
                independent_bic=bics["B"],
                null_bic=bics["C"],
                learned_frequency=learned_frequency,
                exact_harmonic_frequency=tested_frequency,
                frequency_resolution=resolution,
            )
            folds.append({
                "trainingSectors": [dataset["sector"] for dataset in training],
                "heldOutSector": held_out["sector"],
                "learnedIndependentFrequencyCyclesPerDay": learned_frequency,
                "exactHarmonicFrequencyCyclesPerDay": tested_frequency,
                "frequencySeparationCyclesPerDay": abs(
                    learned_frequency - tested_frequency
                ),
                "longBaselineFrequencyResolutionCyclesPerDay": resolution,
                "trainingBIC": {
                    "A": training_a["bic"],
                    "B": training_b["bic"],
                    "C": training_c["bic"],
                },
                "predictiveBIC": bics,
                "predictiveBICDeltas": {
                    "harmonicOverNull": bics["C"] - bics["A"],
                    "independentOverNull": bics["C"] - bics["B"],
                    "independentOverHarmonic": bics["A"] - bics["B"],
                },
                "support": support,
                "failureOrInsufficiencyReasons": reasons,
                "heldOutFittedNuisanceParameters": {
                    "A": predictive_a,
                    "B": predictive_b,
                    "C": predictive_c,
                },
            })
        except (RuntimeError, OverflowError, ValueError) as error:
            folds.append({
                "trainingSectors": [dataset["sector"] for dataset in training],
                "heldOutSector": held_out["sector"],
                "learnedIndependentFrequencyCyclesPerDay": None,
                "exactHarmonicFrequencyCyclesPerDay": tested_frequency,
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

    decision = classify_long_baseline_confirmation(
        folds,
        long_baseline_frequency_resolution=resolution,
    )
    return {
        "methodContractID": METHOD_CONTRACT_ID,
        "methodContractHash": expected_hash,
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
            "frozenDatasetPaths": [
                dataset["datasetPath"] for dataset in datasets
            ],
            "downloadPerformed": False,
        },
    }
