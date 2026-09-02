"""Build a bounded blind morphology-grid project from verified preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from openstar_workloads.plugins.morphology_grid import (
    COMPONENT_TEMPLATE_FAMILY_ID,
    DATASET_SCHEMA_ID,
    EXECUTION_CONTRACT_ID,
    EXECUTION_CONTRACT_VERSION,
    INDEPENDENT_PULSES,
    MAX_SAFE_INTEGER,
    MODEL_CLASS_IDS,
    MORPHOLOGY_FAMILY_ID,
    ORDERED_NEGATIVE_POSITIVE_DOUBLET,
    PAYLOAD_SCHEMA_ID,
    PLUGIN as MORPHOLOGY_GRID_PLUGIN,
    POSITIVE_PULSE_ONLY,
    RESULT_SCHEMA_ID,
    WORKLOAD_ID,
)
from workflows.microlensing.coarse_grid import (
    CoarseGridBuildError,
    _assert_identity_free,
    _atomic_write_bytes,
    _canonical_compact_json_bytes,
    _read_regular_file,
    _stable_json_bytes,
)
from workflows.microlensing.prepare_anomaly_morphology import (
    ARTIFACT_MANIFEST_SCHEMA_ID,
    ARTIFACT_MANIFEST_VERSION,
    CONTRACT_RELATIVE_PATH as PREPARATION_CONTRACT_RELATIVE_PATH,
    DATASET_DIRECTORY as PREPARATION_DATASET_DIRECTORY,
    MANIFEST_RELATIVE_PATH as PREPARATION_MANIFEST_RELATIVE_PATH,
    MODEL_CLASS_IDS as PREPARATION_MODEL_CLASS_IDS,
    MORPHOLOGY_CONTRACT_ID,
    MORPHOLOGY_CONTRACT_VERSION,
    MORPHOLOGY_DATASET_SCHEMA_ID,
    MORPHOLOGY_DATASET_VERSION,
    MORPHOLOGY_PREPARATION_SCHEMA_ID,
    MORPHOLOGY_PREPARATION_VERSION,
    NEXT_TEST as PREPARATION_NEXT_TEST,
    PREPARATION_RELATIVE_PATH,
    AnomalyMorphologyPreparationError,
    _regular_directory as _preparation_regular_directory,
    _reject_symlink_components as _preparation_reject_symlink_components,
)


COARSE_GRID_CONTRACT_ID = (
    "openstar.microlensing-anomaly-morphology-coarse-grid.v1"
)
COARSE_GRID_CONTRACT_VERSION = "1.0"
COARSE_GRID_ALGORITHM_ID = (
    "openstar.microlensing-morphology-global-axis-stride.v1"
)
COARSE_GRID_ALGORITHM_VERSION = "1.0"
BUILD_MANIFEST_SCHEMA_ID = (
    "openstar.microlensing-anomaly-morphology-coarse-grid-build.v1"
)
BUILD_MANIFEST_VERSION = "1.0"

DEFAULT_MAXIMUM_CANDIDATES_PER_SEARCH = 8192
MAXIMUM_ALLOWED_CANDIDATES_PER_SEARCH = 1_000_000
CANDIDATES_PER_WORK_UNIT = 64

PROJECT_RELATIVE_PATH = "project.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"
CONTRACT_RELATIVE_PATH = "coarse-grid-contract.json"
DATASET_DIRECTORY = "datasets"

POSITIVE_DATASET_RELATIVE_PATH = "datasets/positive-pulse-only.json"
ORDERED_DATASET_RELATIVE_PATH = (
    "datasets/ordered-negative-positive-doublet.json"
)
INDEPENDENT_DATASET_RELATIVE_PATHS = (
    "datasets/independent-pulses-series-001.json",
    "datasets/independent-pulses-series-002.json",
)

_SAFE_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_GENERIC_SERIES_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "archivefilename",
        "archiveurl",
        "catalogidentifier",
        "citation",
        "eventidentity",
        "eventname",
        "observatory",
        "observatoryid",
        "sealedmetadata",
        "skycoordinates",
        "sourcefilename",
        "starid",
        "uid",
    }
)

_CONTRACT_FIELDS = frozenset(
    {
        "admittedGenericSeriesIDs",
        "axisRules",
        "benchmarkKind",
        "candidateIndexMapping",
        "comparisonMetrics",
        "contractHashRule",
        "contractID",
        "contractVersion",
        "crossSeriesRequirements",
        "deterministicExecution",
        "decisionRules",
        "effectiveWidthBounds",
        "familyIdentities",
        "finiteValueRules",
        "identityIsolationStatement",
        "independentAggregation",
        "interpretationLimits",
        "invalidCandidateBehavior",
        "modelClassOrder",
        "modelClasses",
        "parameterAxes",
        "preparedCoordinateBounds",
        "separationRules",
    }
)
_PREPARATION_FIELDS = frozenset(
    {
        "admittedGenericSeriesIDs",
        "confirmedComponentProvenance",
        "discoveryClaim",
        "modelClassIDs",
        "morphologyContractID",
        "morphologyContractSHA256",
        "morphologyContractVersion",
        "parentHashes",
        "parentIDs",
        "planetaryInterpretationResolved",
        "preparedCoordinateBounds",
        "preparedDatasets",
        "recommendedNextTest",
        "resultSchemaID",
        "resultVersion",
        "sampleCounts",
        "widthInterpretationResolved",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "artifactManifestSchemaID",
        "artifactManifestVersion",
        "identityIsolationStatement",
        "modelScopeStatement",
        "morphologyContractFileSHA256",
        "morphologyContractID",
        "morphologyContractSHA256",
        "morphologyPreparationFileSHA256",
        "orderedDatasetFiles",
        "orderedGenericSeriesIDs",
        "outputSHA256s",
        "parentHashes",
        "parentIDs",
        "totalSampleCount",
    }
)
_DATASET_FIELDS = frozenset(
    {
        "coordinates",
        "genericSeriesID",
        "inclusionReasons",
        "inverseVariances",
        "morphologyContractID",
        "morphologyContractSHA256",
        "morphologyContractVersion",
        "morphologyDatasetSchemaID",
        "morphologyDatasetVersion",
        "positiveWeightSupport",
        "preparedCoordinateBounds",
        "residualValues",
        "sampleCount",
        "sourceResidualSeriesSHA256",
        "sourceSampleIndices",
    }
)
_PREPARED_DATASET_RECORD_FIELDS = frozenset(
    {
        "genericSeriesID",
        "outputFile",
        "outputSHA256",
        "sampleCount",
        "sourceResidualSeriesSHA256",
    }
)

_EXPECTED_ROOT_NAMES = frozenset(
    {
        PREPARATION_RELATIVE_PATH,
        PREPARATION_CONTRACT_RELATIVE_PATH,
        PREPARATION_MANIFEST_RELATIVE_PATH,
        PREPARATION_DATASET_DIRECTORY,
    }
)
_EXPECTED_PREPARATION_DATASET_PATHS = (
    "datasets/morphology-series-001.json",
    "datasets/morphology-series-002.json",
)


class AnomalyMorphologyCoarseGridBuildError(RuntimeError):
    """The bounded blind morphology project cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class _Axis:
    name: str
    count: int
    start: float | None
    step: float | None
    values: tuple[float, ...] | None

    def coarse(self, stride: int) -> Mapping[str, Any]:
        if type(stride) is not int or stride < 1:
            raise _fail("coarse stride must be a positive integer")
        coarse_count = (self.count - 1) // stride + 1
        if self.values is not None:
            retained = list(self.values[::stride])
            if len(retained) != coarse_count or not retained:
                raise _fail(f"{self.name} explicit stride is inconsistent")
            return {"values": retained}
        if self.start is None or self.step is None:
            raise _fail(f"{self.name} linear axis is malformed")
        coarse_step = self.step * stride
        endpoint = self.start + (coarse_count - 1) * coarse_step
        if not math.isfinite(coarse_step) or not math.isfinite(endpoint):
            raise _fail(f"{self.name} coarse axis is non-finite")
        return {
            "count": coarse_count,
            "start": self.start,
            "step": coarse_step,
        }


@dataclass(frozen=True, slots=True)
class _VerifiedPreparation:
    contract: Mapping[str, Any]
    preparation: Mapping[str, Any]
    manifest: Mapping[str, Any]
    series: tuple[Mapping[str, Any], ...]
    axes: Mapping[str, _Axis]
    full_candidate_counts: Mapping[str, int]
    independent_per_series_candidate_count: int
    contract_canonical_sha256: str
    contract_file_sha256: str
    preparation_file_sha256: str
    manifest_file_sha256: str
    series_file_sha256s: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _Search:
    dataset_id: str
    output_file: str
    model_class_id: str
    generic_series_ids: tuple[str, ...]
    full_candidate_count: int
    stride: int
    morphology_grid: Mapping[str, Any]
    coarse_candidate_count: int


def _fail(message: str) -> AnomalyMorphologyCoarseGridBuildError:
    return AnomalyMorphologyCoarseGridBuildError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(f"{field_name} must be a lowercase SHA-256")
    return value


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field_name} must be a nonempty string")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _fail(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise _fail(f"{field_name} must be finite")
    return number


def _exact_count(value: Any, field_name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        qualifier = "positive " if positive else "nonnegative "
        raise _fail(f"{field_name} must be a {qualifier}integer")
    if value > MAX_SAFE_INTEGER:
        raise _fail(f"{field_name} exceeds the safe integer range")
    return value


def _safe_product(values: Sequence[int], field_name: str) -> int:
    product = 1
    for value in values:
        factor = _exact_count(value, field_name)
        if factor and product > MAX_SAFE_INTEGER // factor:
            raise _fail(f"{field_name} exceeds the safe integer range")
        product *= factor
    return product


def _safe_sum(values: Sequence[int], field_name: str) -> int:
    total = 0
    for value in values:
        item = _exact_count(value, field_name)
        if item > MAX_SAFE_INTEGER - total:
            raise _fail(f"{field_name} exceeds the safe integer range")
        total += item
    return total


def _safe_work_unit_count(candidate_count: int) -> int:
    candidate_count = _exact_count(
        candidate_count,
        "candidate count",
        positive=True,
    )
    if candidate_count > MAX_SAFE_INTEGER - (CANDIDATES_PER_WORK_UNIT - 1):
        raise _fail("work-unit count exceeds the safe integer range")
    return (
        candidate_count + CANDIDATES_PER_WORK_UNIT - 1
    ) // CANDIDATES_PER_WORK_UNIT


def _reject_symlink_components(path: Path, description: str) -> None:
    try:
        _preparation_reject_symlink_components(path, description)
    except AnomalyMorphologyPreparationError as error:
        raise _fail(str(error)) from error


def _regular_directory(path: Path, description: str) -> Path:
    try:
        return _preparation_regular_directory(path, description)
    except AnomalyMorphologyPreparationError as error:
        raise _fail(str(error)) from error


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return _read_regular_file(path, description)
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error


def _json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(f"JSON contains duplicate key {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _fail(f"JSON contains non-finite constant {value}")


def _read_json(path: Path, description: str) -> tuple[Mapping[str, Any], bytes]:
    payload = _read_bytes(path, description)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(document, Mapping):
        raise _fail(f"{description} must contain a JSON object")
    try:
        if _stable_json_bytes(document) != payload:
            raise _fail(f"{description} is not canonical stable JSON")
    except (TypeError, ValueError, OverflowError) as error:
        raise _fail(f"{description} is not canonical finite JSON") from error
    return dict(document), payload


def _safe_relative_path(root: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _fail(f"{field_name} is malformed or unsafe")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise _fail(f"{field_name} is malformed or unsafe")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise _fail(f"{field_name} traverses a symlink or nondirectory")
    if candidate.is_symlink() or not candidate.is_file():
        raise _fail(f"{field_name} is not a regular non-symlink file")
    try:
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise _fail(f"{field_name} escapes the morphology root")
    except OSError as error:
        raise _fail(f"{field_name} cannot be resolved safely") from error
    return candidate


def _assert_identity_isolated(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail("blind artifact keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            if normalized in _FORBIDDEN_IDENTITY_KEYS:
                raise _fail(f"blind artifact contains forbidden identity field {key}")
            _assert_identity_isolated(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_identity_isolated(item)


def _require_exact_fields(
    value: Any,
    fields: frozenset[str],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _fail(f"{field_name} does not match the supported field set")
    return value


def _verify_parent_mapping(
    value: Any,
    field_name: str,
    *,
    hashes: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise _fail(f"{field_name} must be a nonempty mapping")
    result: dict[str, str] = {}
    for key in sorted(value):
        name = _nonempty_string(key, f"{field_name} key")
        item = (
            _sha256_string(value[key], f"{field_name}.{name}")
            if hashes
            else _nonempty_string(value[key], f"{field_name}.{name}")
        )
        result[name] = item
    return result


def _linear_axis(value: Any, name: str, *, exponentiated: bool) -> _Axis:
    expected = {"count", "start", "step"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise _fail(f"{name} must be a strict linear axis")
    count = _exact_count(value["count"], f"{name}.count", positive=True)
    start = _finite_number(value["start"], f"{name}.start")
    step = _finite_number(value["step"], f"{name}.step")
    if step <= 0.0:
        raise _fail(f"{name}.step must be positive")
    endpoint = start + (count - 1) * step
    if not math.isfinite(endpoint):
        raise _fail(f"{name} endpoint is non-finite")
    if exponentiated:
        try:
            expanded = (math.exp(start), math.exp(endpoint))
        except OverflowError as error:
            raise _fail(f"{name} exponentiated endpoint is invalid") from error
        if any(not math.isfinite(item) or item <= 0.0 for item in expanded):
            raise _fail(f"{name} exponentiated endpoint is invalid")
    return _Axis(name=name, count=count, start=start, step=step, values=None)


def _explicit_axis(value: Any, name: str) -> _Axis:
    expected = {"count", "ordering", "values"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise _fail(f"{name} must be a strict explicit axis")
    if value["ordering"] != "strictly ascending explicit values":
        raise _fail(f"{name} ordering is unsupported")
    raw = value["values"]
    if not isinstance(raw, list) or not raw:
        raise _fail(f"{name}.values must be nonempty")
    values = tuple(
        _finite_number(item, f"{name}.values[{index}]")
        for index, item in enumerate(raw)
    )
    count = _exact_count(value["count"], f"{name}.count", positive=True)
    if count != len(values) or any(
        right <= left for left, right in zip(values, values[1:])
    ):
        raise _fail(f"{name} values or count are inconsistent")
    try:
        expanded = tuple(math.exp(item) for item in values)
    except OverflowError as error:
        raise _fail(f"{name} exponentiated value is invalid") from error
    if any(not math.isfinite(item) or item <= 0.0 for item in expanded):
        raise _fail(f"{name} exponentiated value is invalid")
    return _Axis(name=name, count=count, start=None, step=None, values=values)


def _verify_execution_semantics(contract: Mapping[str, Any]) -> None:
    execution = contract.get("deterministicExecution")
    if not isinstance(execution, Mapping):
        raise _fail("morphology deterministic execution contract is missing")
    required = {
        "amplitudeConstraints",
        "arithmetic",
        "candidateWinnerOrdering",
        "comparisonTolerances",
        "componentBasis",
        "constrainedLinearFit",
        "decisionThresholdComparison",
        "designMatrices",
        "linearSolve",
        "modelWinnerOrdering",
        "normalEquations",
        "objective",
        "weightRules",
    }
    if set(execution) != required:
        raise _fail("morphology deterministic execution field set is unsupported")
    arithmetic = execution.get("arithmetic")
    if not isinstance(arithmetic, Mapping) or (
        arithmetic.get("format") != "IEEE-754 binary64"
        or arithmetic.get("roundingMode") != "roundTiesToEven"
    ):
        raise _fail("morphology arithmetic contract is unsupported")
    basis = execution.get("componentBasis")
    expected_equation = (
        "scale=exp(logScale); shape=exp(logShape); "
        "z=(coordinate-center)/scale; uSquared=shape*shape+z*z; "
        "basis=(uSquared+2)/(sqrt(uSquared)*sqrt(uSquared+4))"
    )
    if not isinstance(basis, Mapping) or basis.get("equation") != expected_equation:
        raise _fail("morphology component basis is unsupported")
    tolerances = execution.get("comparisonTolerances")
    if not isinstance(tolerances, Mapping) or (
        tolerances.get("objectiveRelativeTolerance") != 1.0e-9
        or tolerances.get("rankRelativeTolerance") != 1.0e-12
        or tolerances.get("constraintTolerance") != 0.0
        or tolerances.get("timingComparisonTolerance") != 0.0
    ):
        raise _fail("morphology numerical tolerances are unsupported")
    expected_order = [
        "finite WRSS ascending within relative tolerance",
        "finite BIC ascending within relative tolerance",
        "finite AICc ascending within relative tolerance; null after finite",
        "global candidate index ascending",
    ]
    if execution.get("candidateWinnerOrdering") != expected_order:
        raise _fail("morphology candidate ordering is unsupported")
    weights = execution.get("weightRules")
    if not isinstance(weights, Mapping) or set(weights) != {
        "negative",
        "positive",
        "zero",
    }:
        raise _fail("morphology weight rules are unsupported")


def _verify_contract(
    contract: Mapping[str, Any],
) -> tuple[tuple[str, str], Mapping[str, _Axis], Mapping[str, int], int]:
    _require_exact_fields(contract, _CONTRACT_FIELDS, "morphology contract")
    if (
        contract.get("contractID") != MORPHOLOGY_CONTRACT_ID
        or contract.get("contractVersion") != MORPHOLOGY_CONTRACT_VERSION
        or contract.get("benchmarkKind") != "known-event-recovery"
    ):
        raise _fail("morphology contract identity or benchmark kind is unsupported")
    if contract.get("modelClassOrder") != list(MODEL_CLASS_IDS) or tuple(
        PREPARATION_MODEL_CLASS_IDS
    ) != tuple(MODEL_CLASS_IDS):
        raise _fail("morphology model-class order is unsupported")
    admitted = contract.get("admittedGenericSeriesIDs")
    if not isinstance(admitted, list) or len(admitted) != 2:
        raise _fail("morphology preparation must admit exactly two series")
    admitted_ids = tuple(
        _nonempty_string(item, f"admittedGenericSeriesIDs[{index}]")
        for index, item in enumerate(admitted)
    )
    if (
        len(set(admitted_ids)) != 2
        or admitted_ids != tuple(sorted(admitted_ids))
        or any(_SAFE_GENERIC_SERIES_ID.fullmatch(item) is None for item in admitted_ids)
    ):
        raise _fail("admitted generic series order or identifiers are invalid")

    families = contract.get("familyIdentities")
    if not isinstance(families, Mapping) or dict(families) != {
        "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
        "componentTemplateScope": (
            "Identifies only one unit symmetric radial component, not any "
            "compound morphology model class."
        ),
        "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
        "morphologyFamilyScope": (
            "Identifies the three compound residual-morphology classes and "
            "their deterministic execution contract."
        ),
    }:
        raise _fail("morphology family identities are unsupported")
    if contract.get("identityIsolationStatement") != (
        "Only generic series identifiers and verified identity-free "
        "numerical residual evidence are admitted."
    ):
        raise _fail("morphology identity-isolation statement is invalid")
    _verify_execution_semantics(contract)

    axes_value = contract.get("parameterAxes")
    if not isinstance(axes_value, Mapping) or set(axes_value) != {
        "CENTER",
        "LOG_SCALE",
        "LOG_SHAPE",
        "SEPARATION",
    }:
        raise _fail("morphology parameter axes are malformed")
    axes: dict[str, _Axis] = {
        "CENTER": _linear_axis(
            axes_value["CENTER"],
            "CENTER",
            exponentiated=False,
        ),
        "LOG_SCALE": _linear_axis(
            axes_value["LOG_SCALE"],
            "LOG_SCALE",
            exponentiated=True,
        ),
        "LOG_SHAPE": _explicit_axis(
            axes_value["LOG_SHAPE"],
            "LOG_SHAPE",
        ),
        "SEPARATION": _linear_axis(
            axes_value["SEPARATION"],
            "SEPARATION",
            exponentiated=False,
        ),
    }
    if axes["CENTER"].count < 2:
        raise _fail("morphology center axis cannot form an independent pair")

    center_count = axes["CENTER"].count
    log_scale_count = axes["LOG_SCALE"].count
    log_shape_count = axes["LOG_SHAPE"].count
    separation_count = axes["SEPARATION"].count
    center_pair_count = _safe_product(
        [center_count, center_count - 1],
        "full independent center-pair numerator",
    ) // 2
    positive_count = _safe_product(
        [center_count, log_scale_count, log_shape_count],
        "full positive candidate count",
    )
    ordered_count = _safe_product(
        [
            center_count,
            separation_count,
            log_scale_count,
            log_shape_count,
            log_scale_count,
            log_shape_count,
        ],
        "full ordered candidate count",
    )
    independent_per_series = _safe_product(
        [
            center_pair_count,
            log_scale_count,
            log_shape_count,
            log_scale_count,
            log_shape_count,
        ],
        "full independent per-series candidate count",
    )
    independent_total = _safe_product(
        [independent_per_series, len(admitted_ids)],
        "full independent total candidate count",
    )
    expected_counts = {
        POSITIVE_PULSE_ONLY: positive_count,
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: ordered_count,
        INDEPENDENT_PULSES: independent_total,
    }
    mapping = contract.get("candidateIndexMapping")
    if not isinstance(mapping, Mapping) or set(mapping) != {
        "axisOrderingByModelClass",
        "axisSourceByModelClass",
        "candidateCounts",
        "globalCandidateCount",
        "globalCandidateOffsets",
        "independentPerSeriesMapping",
        "linearizationRule",
        "maximumSafeInteger",
    }:
        raise _fail("morphology candidate-index mapping is malformed")
    expected_axis_ordering = {
        POSITIVE_PULSE_ONLY: ["CENTER", "LOG_SCALE", "LOG_SHAPE"],
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: [
            "NEGATIVE_CENTER",
            "SEPARATION",
            "NEGATIVE_LOG_SCALE",
            "NEGATIVE_LOG_SHAPE",
            "POSITIVE_LOG_SCALE",
            "POSITIVE_LOG_SHAPE",
        ],
        INDEPENDENT_PULSES: [
            "NEGATIVE_CENTER_PAIR_POSITIVE_CENTER",
            "NEGATIVE_LOG_SCALE",
            "NEGATIVE_LOG_SHAPE",
            "POSITIVE_LOG_SCALE",
            "POSITIVE_LOG_SHAPE",
        ],
    }
    expected_axis_sources = {
        POSITIVE_PULSE_ONLY: {
            "CENTER": "CENTER",
            "LOG_SCALE": "LOG_SCALE",
            "LOG_SHAPE": "LOG_SHAPE",
        },
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
            "NEGATIVE_CENTER": "CENTER",
            "NEGATIVE_LOG_SCALE": "LOG_SCALE",
            "NEGATIVE_LOG_SHAPE": "LOG_SHAPE",
            "POSITIVE_LOG_SCALE": "LOG_SCALE",
            "POSITIVE_LOG_SHAPE": "LOG_SHAPE",
            "SEPARATION": "SEPARATION",
        },
        INDEPENDENT_PULSES: {
            "NEGATIVE_CENTER_PAIR_POSITIVE_CENTER": "CENTER",
            "NEGATIVE_LOG_SCALE": "LOG_SCALE",
            "NEGATIVE_LOG_SHAPE": "LOG_SHAPE",
            "POSITIVE_LOG_SCALE": "LOG_SCALE",
            "POSITIVE_LOG_SHAPE": "LOG_SHAPE",
        },
    }
    if (
        mapping.get("axisOrderingByModelClass") != expected_axis_ordering
        or mapping.get("axisSourceByModelClass") != expected_axis_sources
        or mapping.get("linearizationRule")
        != (
            "Shared model classes use the declared class order and rightmost-"
            "fastest mixed radix. INDEPENDENT_PULSES concatenates canonical "
            "per-series searches; it never forms a product across series."
        )
    ):
        raise _fail("morphology candidate-axis mapping is inconsistent")
    if mapping.get("candidateCounts") != expected_counts:
        raise _fail("morphology full candidate counts are inconsistent")
    expected_offsets = {
        POSITIVE_PULSE_ONLY: 0,
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: positive_count,
        INDEPENDENT_PULSES: _safe_sum(
            [positive_count, ordered_count],
            "full independent offset",
        ),
    }
    if mapping.get("globalCandidateOffsets") != expected_offsets:
        raise _fail("morphology global candidate offsets are inconsistent")
    expected_global = _safe_sum(
        [positive_count, ordered_count, independent_total],
        "full global candidate count",
    )
    if (
        mapping.get("globalCandidateCount") != expected_global
        or mapping.get("maximumSafeInteger") != MAX_SAFE_INTEGER
    ):
        raise _fail("morphology global candidate count is inconsistent")

    layout = mapping.get("independentPerSeriesMapping")
    if not isinstance(layout, Mapping) or set(layout) != {
        "centerPairIndexFormula",
        "independentSeriesSearches",
        "localMixedRadixFormula",
        "orderedCenterPairCount",
        "orderedCenterPairRule",
        "perSeriesCandidateCount",
        "totalCandidateCount",
    } or (
        layout.get("orderedCenterPairCount") != center_pair_count
        or layout.get("perSeriesCandidateCount") != independent_per_series
        or layout.get("totalCandidateCount") != independent_total
    ):
        raise _fail("independent per-series candidate mapping is inconsistent")
    if (
        layout.get("centerPairIndexFormula")
        != (
            "pairIndex = negativeCenterIndex * "
            "(2 * centerCount - negativeCenterIndex - 1) // 2 + "
            "(positiveCenterIndex - negativeCenterIndex - 1)"
        )
        or layout.get("localMixedRadixFormula")
        != (
            "localIndex = ((((pairIndex * logScaleCount + "
            "negativeLogScaleIndex) * logShapeCount + "
            "negativeLogShapeIndex) * logScaleCount + "
            "positiveLogScaleIndex) * logShapeCount + "
            "positiveLogShapeIndex)"
        )
        or layout.get("orderedCenterPairRule")
        != (
            "Enumerate negativeCenterIndex ascending, then positiveCenterIndex "
            "ascending, retaining exactly 0 <= negativeCenterIndex < "
            "positiveCenterIndex < centerCount."
        )
    ):
        raise _fail("independent mixed-radix mapping is unsupported")
    searches = layout.get("independentSeriesSearches")
    if not isinstance(searches, list) or len(searches) != len(admitted_ids):
        raise _fail("independent per-series searches are malformed")
    independent_start = expected_offsets[INDEPENDENT_PULSES]
    for ordinal, (series_id, search) in enumerate(zip(admitted_ids, searches)):
        if not isinstance(search, Mapping):
            raise _fail("independent per-series search is malformed")
        start = _safe_sum(
            [independent_start, ordinal * independent_per_series],
            "independent per-series start",
        )
        end = _safe_sum(
            [start, independent_per_series],
            "independent per-series end",
        )
        if dict(search) != {
            "candidateCount": independent_per_series,
            "canonicalSeriesIndex": ordinal,
            "genericSeriesID": series_id,
            "globalEndExclusive": end,
            "globalStartIndex": start,
        }:
            raise _fail("independent per-series search mapping is inconsistent")

    metrics = contract.get("comparisonMetrics")
    if not isinstance(metrics, Mapping):
        raise _fail("morphology comparison metrics are malformed")
    parameter_counts = metrics.get("parameterCounts")
    expected_parameter_counts = {
        POSITIVE_PULSE_ONLY: {"linear": 4, "nonlinear": 3, "total": 7},
        ORDERED_NEGATIVE_POSITIVE_DOUBLET: {
            "linear": 6,
            "nonlinear": 6,
            "total": 12,
        },
        INDEPENDENT_PULSES: {"linear": 6, "nonlinear": 12, "total": 18},
    }
    if parameter_counts != expected_parameter_counts:
        raise _fail("morphology parameter-count mapping is inconsistent")
    return admitted_ids, axes, expected_counts, independent_per_series


def _verify_series(
    document: Mapping[str, Any],
    *,
    expected_id: str,
    expected_contract_sha256: str,
    expected_bounds: Mapping[str, float],
) -> Mapping[str, Any]:
    _require_exact_fields(document, _DATASET_FIELDS, f"{expected_id} dataset")
    if document.get("genericSeriesID") != expected_id:
        raise _fail("prepared dataset generic series ID is inconsistent")
    if (
        document.get("morphologyDatasetSchemaID")
        != MORPHOLOGY_DATASET_SCHEMA_ID
        or document.get("morphologyDatasetVersion")
        != MORPHOLOGY_DATASET_VERSION
        or document.get("morphologyContractID") != MORPHOLOGY_CONTRACT_ID
        or document.get("morphologyContractVersion")
        != MORPHOLOGY_CONTRACT_VERSION
        or document.get("morphologyContractSHA256")
        != expected_contract_sha256
    ):
        raise _fail("prepared dataset contract identity is inconsistent")
    sample_count = _exact_count(
        document.get("sampleCount"),
        f"{expected_id} sample count",
        positive=True,
    )
    arrays: list[list[Any]] = []
    for field_name in ("coordinates", "residualValues", "inverseVariances"):
        value = document.get(field_name)
        if not isinstance(value, list) or len(value) != sample_count:
            raise _fail(f"{expected_id} {field_name} length is inconsistent")
        arrays.append(value)
    coordinates, residuals, weights = arrays
    numeric_coordinates = [
        _finite_number(value, f"{expected_id} coordinates[{index}]")
        for index, value in enumerate(coordinates)
    ]
    for index, value in enumerate(residuals):
        _finite_number(value, f"{expected_id} residualValues[{index}]")
    numeric_weights = [
        _finite_number(value, f"{expected_id} inverseVariances[{index}]")
        for index, value in enumerate(weights)
    ]
    if any(
        right <= left
        for left, right in zip(numeric_coordinates, numeric_coordinates[1:])
    ):
        raise _fail(f"{expected_id} coordinates are not strictly increasing")
    if any(weight < 0.0 for weight in numeric_weights):
        raise _fail(f"{expected_id} inverse variances contain a negative weight")

    indices = document.get("sourceSampleIndices")
    if not isinstance(indices, list) or len(indices) != sample_count:
        raise _fail(f"{expected_id} source sample indices are malformed")
    parsed_indices = [
        _exact_count(item, f"{expected_id} sourceSampleIndices[{index}]")
        for index, item in enumerate(indices)
    ]
    if any(right <= left for left, right in zip(parsed_indices, parsed_indices[1:])):
        raise _fail(f"{expected_id} source sample indices are not increasing")
    reasons = document.get("inclusionReasons")
    if not isinstance(reasons, list) or len(reasons) != sample_count or any(
        not isinstance(item, list)
        or not item
        or any(not isinstance(reason, str) or not reason for reason in item)
        for item in reasons
    ):
        raise _fail(f"{expected_id} inclusion reasons are malformed")
    _sha256_string(
        document.get("sourceResidualSeriesSHA256"),
        f"{expected_id} source residual series SHA-256",
    )
    bounds = document.get("preparedCoordinateBounds")
    if not isinstance(bounds, Mapping) or set(bounds) != {"minimum", "maximum"}:
        raise _fail(f"{expected_id} prepared coordinate bounds are malformed")
    minimum = _finite_number(bounds["minimum"], f"{expected_id} minimum")
    maximum = _finite_number(bounds["maximum"], f"{expected_id} maximum")
    if dict(bounds) != dict(expected_bounds) or minimum >= maximum or any(
        coordinate < minimum or coordinate > maximum
        for coordinate in numeric_coordinates
    ):
        raise _fail(f"{expected_id} prepared coordinate bounds are inconsistent")
    support = document.get("positiveWeightSupport")
    expected_support_fields = {
        "confirmedPositiveWithinTwoEffectiveWidths",
        "leftBaseline",
        "precedingNegativeWithinTwoEffectiveWidths",
        "rightBaseline",
    }
    if not isinstance(support, Mapping) or set(support) != expected_support_fields:
        raise _fail(f"{expected_id} positive-weight support is malformed")
    positive_weight_count = sum(weight > 0.0 for weight in numeric_weights)
    for key, value in support.items():
        count = _exact_count(value, f"{expected_id} positiveWeightSupport.{key}")
        if count > positive_weight_count:
            raise _fail(f"{expected_id} positive-weight support is inconsistent")
    return document


def _verify_preparation(root: Path) -> _VerifiedPreparation:
    root = _regular_directory(root, "morphology preparation root")
    try:
        if {entry.name for entry in root.iterdir()} != _EXPECTED_ROOT_NAMES:
            raise _fail(
                "morphology preparation artifact set is incomplete or unexpected"
            )
    except OSError as error:
        raise _fail("morphology preparation root is unreadable") from error
    dataset_root = _regular_directory(
        root / PREPARATION_DATASET_DIRECTORY,
        "morphology preparation dataset directory",
    )
    try:
        if {entry.name for entry in dataset_root.iterdir()} != {
            "morphology-series-001.json",
            "morphology-series-002.json",
        }:
            raise _fail(
                "morphology preparation dataset set is incomplete or unexpected"
            )
    except OSError as error:
        raise _fail("morphology preparation dataset directory is unreadable") from error

    contract, contract_bytes = _read_json(
        root / PREPARATION_CONTRACT_RELATIVE_PATH,
        "morphology contract",
    )
    preparation, preparation_bytes = _read_json(
        root / PREPARATION_RELATIVE_PATH,
        "morphology preparation result",
    )
    manifest, manifest_bytes = _read_json(
        root / PREPARATION_MANIFEST_RELATIVE_PATH,
        "morphology artifact manifest",
    )
    _assert_identity_isolated((contract, preparation, manifest))
    admitted_ids, axes, full_counts, independent_per_series = _verify_contract(
        contract
    )
    contract_bounds = contract.get("preparedCoordinateBounds")
    if not isinstance(contract_bounds, Mapping) or set(contract_bounds) != {
        "maximum",
        "minimum",
    }:
        raise _fail("morphology contract coordinate bounds are malformed")
    contract_minimum = _finite_number(
        contract_bounds["minimum"],
        "morphology contract coordinate minimum",
    )
    contract_maximum = _finite_number(
        contract_bounds["maximum"],
        "morphology contract coordinate maximum",
    )
    if contract_minimum >= contract_maximum:
        raise _fail("morphology contract coordinate bounds are invalid")
    contract_canonical_sha256 = _sha256_bytes(
        _canonical_compact_json_bytes(contract)
    )
    contract_file_sha256 = _sha256_bytes(contract_bytes)
    preparation_file_sha256 = _sha256_bytes(preparation_bytes)
    manifest_file_sha256 = _sha256_bytes(manifest_bytes)

    _require_exact_fields(preparation, _PREPARATION_FIELDS, "preparation result")
    if (
        preparation.get("resultSchemaID") != MORPHOLOGY_PREPARATION_SCHEMA_ID
        or preparation.get("resultVersion") != MORPHOLOGY_PREPARATION_VERSION
        or preparation.get("morphologyContractID") != MORPHOLOGY_CONTRACT_ID
        or preparation.get("morphologyContractVersion")
        != MORPHOLOGY_CONTRACT_VERSION
        or preparation.get("morphologyContractSHA256")
        != contract_canonical_sha256
        or preparation.get("recommendedNextTest") != PREPARATION_NEXT_TEST
        or preparation.get("modelClassIDs") != list(MODEL_CLASS_IDS)
        or preparation.get("admittedGenericSeriesIDs") != list(admitted_ids)
        or preparation.get("planetaryInterpretationResolved") is not False
        or preparation.get("discoveryClaim") is not False
        or preparation.get("widthInterpretationResolved") is not False
    ):
        raise _fail("morphology preparation result is unsupported or inconsistent")
    parent_hashes = _verify_parent_mapping(
        preparation.get("parentHashes"),
        "preparation parentHashes",
        hashes=True,
    )
    parent_ids = _verify_parent_mapping(
        preparation.get("parentIDs"),
        "preparation parentIDs",
        hashes=False,
    )
    preparation_bounds = preparation.get("preparedCoordinateBounds")
    if not isinstance(preparation_bounds, Mapping) or set(preparation_bounds) != {
        "anomalyCoreMaximum",
        "anomalyCoreMinimum",
        "maximum",
        "minimum",
    }:
        raise _fail("morphology preparation coordinate bounds are malformed")
    core_minimum = _finite_number(
        preparation_bounds["anomalyCoreMinimum"],
        "morphology anomaly-core minimum",
    )
    core_maximum = _finite_number(
        preparation_bounds["anomalyCoreMaximum"],
        "morphology anomaly-core maximum",
    )
    if (
        preparation_bounds["minimum"] != contract_minimum
        or preparation_bounds["maximum"] != contract_maximum
        or not contract_minimum < core_minimum < core_maximum < contract_maximum
    ):
        raise _fail("morphology preparation coordinate bounds are inconsistent")

    _require_exact_fields(manifest, _MANIFEST_FIELDS, "artifact manifest")
    if (
        manifest.get("artifactManifestSchemaID") != ARTIFACT_MANIFEST_SCHEMA_ID
        or manifest.get("artifactManifestVersion") != ARTIFACT_MANIFEST_VERSION
        or manifest.get("morphologyContractID") != MORPHOLOGY_CONTRACT_ID
        or manifest.get("morphologyContractSHA256")
        != contract_canonical_sha256
        or manifest.get("morphologyContractFileSHA256")
        != contract_file_sha256
        or manifest.get("morphologyPreparationFileSHA256")
        != preparation_file_sha256
        or manifest.get("orderedGenericSeriesIDs") != list(admitted_ids)
        or manifest.get("parentHashes") != parent_hashes
        or manifest.get("parentIDs") != parent_ids
        or manifest.get("identityIsolationStatement") != (
            "Artifacts contain generic identifiers and identity-free numerical "
            "evidence only."
        )
    ):
        raise _fail("morphology artifact manifest is inconsistent")
    ordered_files = manifest.get("orderedDatasetFiles")
    if ordered_files != list(_EXPECTED_PREPARATION_DATASET_PATHS):
        raise _fail("morphology dataset path order is unsupported")
    output_hashes = manifest.get("outputSHA256s")
    if not isinstance(output_hashes, Mapping) or set(output_hashes) != set(
        _EXPECTED_PREPARATION_DATASET_PATHS
    ):
        raise _fail("morphology dataset hash mapping is malformed")
    records = preparation.get("preparedDatasets")
    if not isinstance(records, list) or len(records) != len(admitted_ids):
        raise _fail("morphology prepared-dataset records are malformed")

    series_documents: list[Mapping[str, Any]] = []
    series_hashes: dict[str, str] = {}
    sample_counts: dict[str, int] = {}
    for ordinal, (series_id, relative_path, record) in enumerate(
        zip(admitted_ids, _EXPECTED_PREPARATION_DATASET_PATHS, records)
    ):
        _require_exact_fields(
            record,
            _PREPARED_DATASET_RECORD_FIELDS,
            f"prepared dataset record {ordinal}",
        )
        candidate = _safe_relative_path(
            root,
            relative_path,
            f"prepared dataset path {ordinal}",
        )
        document, payload = _read_json(candidate, f"prepared dataset {ordinal}")
        file_sha256 = _sha256_bytes(payload)
        if (
            record.get("genericSeriesID") != series_id
            or record.get("outputFile") != relative_path
            or record.get("outputSHA256") != file_sha256
            or output_hashes.get(relative_path) != file_sha256
        ):
            raise _fail("prepared dataset record or hash is inconsistent")
        verified = _verify_series(
            document,
            expected_id=series_id,
            expected_contract_sha256=contract_canonical_sha256,
            expected_bounds=contract_bounds,
        )
        sample_count = _exact_count(
            record.get("sampleCount"),
            f"{series_id} recorded sample count",
            positive=True,
        )
        source_sha256 = _sha256_string(
            record.get("sourceResidualSeriesSHA256"),
            f"{series_id} recorded source SHA-256",
        )
        if (
            sample_count != verified["sampleCount"]
            or source_sha256 != verified["sourceResidualSeriesSHA256"]
        ):
            raise _fail("prepared dataset source provenance is inconsistent")
        sample_counts[series_id] = sample_count
        series_hashes[relative_path] = file_sha256
        series_documents.append(verified)

    sample_count_record = preparation.get("sampleCounts")
    total_sample_count = _safe_sum(
        list(sample_counts.values()),
        "prepared total sample count",
    )
    if (
        not isinstance(sample_count_record, Mapping)
        or set(sample_count_record) != {"perSeries", "total"}
        or sample_count_record.get("perSeries") != sample_counts
        or sample_count_record.get("total") != total_sample_count
        or manifest.get("totalSampleCount") != total_sample_count
    ):
        raise _fail("prepared sample-count accounting is inconsistent")
    return _VerifiedPreparation(
        contract=dict(contract),
        preparation=dict(preparation),
        manifest=dict(manifest),
        series=tuple(dict(item) for item in series_documents),
        axes=dict(axes),
        full_candidate_counts=dict(full_counts),
        independent_per_series_candidate_count=independent_per_series,
        contract_canonical_sha256=contract_canonical_sha256,
        contract_file_sha256=contract_file_sha256,
        preparation_file_sha256=preparation_file_sha256,
        manifest_file_sha256=manifest_file_sha256,
        series_file_sha256s=dict(series_hashes),
    )


def _coarse_axis_count(axis: Mapping[str, Any]) -> int:
    if set(axis) == {"count", "start", "step"}:
        return _exact_count(axis["count"], "coarse axis count", positive=True)
    if set(axis) == {"values"} and isinstance(axis["values"], list):
        return _exact_count(
            len(axis["values"]),
            "coarse explicit axis count",
            positive=True,
        )
    raise _fail("coarse axis representation is malformed")


def _grid_for_stride(
    model_class_id: str,
    axes: Mapping[str, _Axis],
    stride: int,
) -> tuple[Mapping[str, Any], int] | None:
    center = axes["CENTER"].coarse(stride)
    log_scale = axes["LOG_SCALE"].coarse(stride)
    log_shape = axes["LOG_SHAPE"].coarse(stride)
    center_count = _coarse_axis_count(center)
    log_scale_count = _coarse_axis_count(log_scale)
    log_shape_count = _coarse_axis_count(log_shape)
    if model_class_id == POSITIVE_PULSE_ONLY:
        grid = {
            "centerAxis": center,
            "logScaleAxis": log_scale,
            "logShapeAxis": log_shape,
        }
        count = _safe_product(
            [center_count, log_scale_count, log_shape_count],
            "coarse positive candidate count",
        )
        return grid, count
    if model_class_id == ORDERED_NEGATIVE_POSITIVE_DOUBLET:
        separation = axes["SEPARATION"].coarse(stride)
        separation_count = _coarse_axis_count(separation)
        grid = {
            "negativeCenterAxis": center,
            "separationAxis": separation,
            "negativeLogScaleAxis": log_scale,
            "negativeLogShapeAxis": log_shape,
            "positiveLogScaleAxis": dict(log_scale),
            "positiveLogShapeAxis": dict(log_shape),
        }
        count = _safe_product(
            [
                center_count,
                separation_count,
                log_scale_count,
                log_shape_count,
                log_scale_count,
                log_shape_count,
            ],
            "coarse ordered candidate count",
        )
        return grid, count
    if model_class_id == INDEPENDENT_PULSES:
        if center_count < 2:
            return None
        center_pair_count = _safe_product(
            [center_count, center_count - 1],
            "coarse independent center-pair numerator",
        ) // 2
        grid = {
            "centerAxis": center,
            "negativeLogScaleAxis": log_scale,
            "negativeLogShapeAxis": log_shape,
            "positiveLogScaleAxis": dict(log_scale),
            "positiveLogShapeAxis": dict(log_shape),
        }
        count = _safe_product(
            [
                center_pair_count,
                log_scale_count,
                log_shape_count,
                log_scale_count,
                log_shape_count,
            ],
            "coarse independent candidate count",
        )
        return grid, count
    raise _fail("unsupported morphology model class")


def _select_stride(
    model_class_id: str,
    axes: Mapping[str, _Axis],
    maximum_candidates: int,
) -> tuple[int, Mapping[str, Any], int]:
    maximum_candidates = _exact_count(
        maximum_candidates,
        "maximum candidates per search",
        positive=True,
    )
    maximum_stride = max(axis.count for axis in axes.values())
    if model_class_id == INDEPENDENT_PULSES:
        maximum_stride = axes["CENTER"].count - 1
    for stride in range(1, maximum_stride + 1):
        derived = _grid_for_stride(model_class_id, axes, stride)
        if derived is None:
            continue
        grid, count = derived
        if count <= maximum_candidates:
            return stride, grid, count
    raise _fail(
        f"no admissible {model_class_id} stride satisfies the candidate limit"
    )


def _searches(
    project_id: str,
    verified: _VerifiedPreparation,
    maximum_candidates: int,
) -> tuple[_Search, ...]:
    admitted_ids = tuple(verified.preparation["admittedGenericSeriesIDs"])
    definitions = (
        (
            f"{project_id}.positive-pulse-only",
            POSITIVE_DATASET_RELATIVE_PATH,
            POSITIVE_PULSE_ONLY,
            admitted_ids,
            verified.full_candidate_counts[POSITIVE_PULSE_ONLY],
        ),
        (
            f"{project_id}.ordered-negative-positive-doublet",
            ORDERED_DATASET_RELATIVE_PATH,
            ORDERED_NEGATIVE_POSITIVE_DOUBLET,
            admitted_ids,
            verified.full_candidate_counts[ORDERED_NEGATIVE_POSITIVE_DOUBLET],
        ),
        *(
            (
                f"{project_id}.independent-pulses.{series_id}",
                INDEPENDENT_DATASET_RELATIVE_PATHS[ordinal],
                INDEPENDENT_PULSES,
                (series_id,),
                verified.independent_per_series_candidate_count,
            )
            for ordinal, series_id in enumerate(admitted_ids)
        ),
    )
    searches: list[_Search] = []
    for dataset_id, output_file, model_class_id, series_ids, full_count in definitions:
        stride, grid, coarse_count = _select_stride(
            model_class_id,
            verified.axes,
            maximum_candidates,
        )
        searches.append(
            _Search(
                dataset_id=dataset_id,
                output_file=output_file,
                model_class_id=model_class_id,
                generic_series_ids=tuple(series_ids),
                full_candidate_count=full_count,
                stride=stride,
                morphology_grid=grid,
                coarse_candidate_count=coarse_count,
            )
        )
    return tuple(searches)


def _contract(
    verified: _VerifiedPreparation,
    searches: Sequence[_Search],
    maximum_candidates: int,
) -> dict[str, Any]:
    return {
        "algorithmID": COARSE_GRID_ALGORITHM_ID,
        "algorithmVersion": COARSE_GRID_ALGORITHM_VERSION,
        "candidateLimitRule": (
            "For each ordered model search, start stride at one and increase by "
            "one. Apply that single stride to every source axis and accept the "
            "first admissible grid whose candidate count does not exceed the "
            "frozen per-search limit."
        ),
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "contractHashRule": (
            "SHA-256 of UTF-8 JSON with sorted keys, no insignificant "
            "whitespace, non-ASCII preserved, and nonfinite numbers forbidden."
        ),
        "contractID": COARSE_GRID_CONTRACT_ID,
        "contractVersion": COARSE_GRID_CONTRACT_VERSION,
        "executionCompatibility": {
            "sourceMorphologyContractID": MORPHOLOGY_CONTRACT_ID,
            "sourceMorphologyContractVersion": MORPHOLOGY_CONTRACT_VERSION,
            "verificationRule": (
                "The supported preparation contract's deterministicExecution "
                "semantics are verified before binding datasets to the workload "
                "execution contract."
            ),
            "workloadExecutionContractID": EXECUTION_CONTRACT_ID,
            "workloadExecutionContractVersion": EXECUTION_CONTRACT_VERSION,
        },
        "explicitAxisRule": (
            "Retain source values at indices 0,stride,2*stride,... without "
            "appending an off-stride endpoint."
        ),
        "identityIsolationStatement": (
            "Only generic series identifiers, verified identity-free numerical "
            "samples, and immutable artifact hashes are published."
        ),
        "independentSearchRule": (
            "Each admitted series is one independent workload dataset. The "
            "center axis is strided before strict ordered center-pair counting; "
            "no cross-series candidate product is formed."
        ),
        "linearAxisRule": (
            "Preserve start, multiply step by stride, and set count to "
            "floor((sourceCount-1)/stride)+1 without appending an irregular endpoint."
        ),
        "maximumCandidatesPerSearch": maximum_candidates,
        "modelClassOrder": list(MODEL_CLASS_IDS),
        "modelScopeStatement": (
            "This is blind generic morphology model comparison preparation. It "
            "does not evaluate a candidate, interpret a model, or make a "
            "discovery claim."
        ),
        "noCandidateEvaluationStatement": (
            "The builder performs no basis evaluation, nuisance fit, objective "
            "calculation, model ranking, classification, or interpretation."
        ),
        "orderedSearches": [
            {
                "coarseCandidateCount": search.coarse_candidate_count,
                "datasetID": search.dataset_id,
                "fullCandidateCount": search.full_candidate_count,
                "genericSeriesIDs": list(search.generic_series_ids),
                "modelClassID": search.model_class_id,
                "outputFile": search.output_file,
                "publishedMorphologyGrid": dict(search.morphology_grid),
                "selectedStride": search.stride,
            }
            for search in searches
        ],
        "sourceMorphologyContractCanonicalSHA256": (
            verified.contract_canonical_sha256
        ),
        "sourceMorphologyContractFileSHA256": verified.contract_file_sha256,
        "workloadIdentities": {
            "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
            "datasetSchemaID": DATASET_SCHEMA_ID,
            "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
            "payloadSchemaID": PAYLOAD_SCHEMA_ID,
            "resultSchemaID": RESULT_SCHEMA_ID,
            "workloadID": WORKLOAD_ID,
        },
    }


def _dataset(
    *,
    search: _Search,
    series_by_id: Mapping[str, Mapping[str, Any]],
    verified: _VerifiedPreparation,
    coarse_contract_sha256: str,
) -> dict[str, Any]:
    series = []
    source_indices: dict[str, list[int]] = {}
    source_hashes: dict[str, str] = {}
    for series_id in search.generic_series_ids:
        source = series_by_id[series_id]
        series.append(
            {
                "genericSeriesID": series_id,
                "coordinates": list(source["coordinates"]),
                "values": list(source["residualValues"]),
                "inverseVariances": list(source["inverseVariances"]),
            }
        )
        source_indices[series_id] = list(source["sourceSampleIndices"])
        source_hashes[series_id] = source["sourceResidualSeriesSHA256"]
    dataset = {
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "coarseGridContractID": COARSE_GRID_CONTRACT_ID,
        "coarseGridContractSHA256": coarse_contract_sha256,
        "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "executionContractID": EXECUTION_CONTRACT_ID,
        "executionContractVersion": EXECUTION_CONTRACT_VERSION,
        "id": search.dataset_id,
        "modelClassID": search.model_class_id,
        "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
        "morphologyGrid": dict(search.morphology_grid),
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "series": series,
        "sourceGenericSeriesIDs": list(search.generic_series_ids),
        "sourceMorphologyContractSHA256": verified.contract_canonical_sha256,
        "sourceMorphologyPreparationSHA256": verified.preparation_file_sha256,
        "sourceResidualSeriesSHA256s": source_hashes,
        "sourceSampleIndicesBySeries": source_indices,
        "workloadID": WORKLOAD_ID,
    }
    try:
        MORPHOLOGY_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise _fail(
            f"constructed morphology-grid dataset is invalid: {error}"
        ) from error
    return dataset


def _project(project_id: str, searches: Sequence[_Search]) -> dict[str, Any]:
    return {
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "datasets": [
            {"id": search.dataset_id, "path": search.output_file}
            for search in searches
        ],
        "id": project_id,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "workloadID": WORKLOAD_ID,
    }


def _build_anomaly_morphology_coarse_grid_impl(
    morphology_root: str | Path,
    *,
    project_id: str,
    output_root: str | Path,
    maximum_candidates_per_search: int,
) -> dict[str, Any]:
    if (
        not isinstance(project_id, str)
        or _SAFE_PROJECT_ID.fullmatch(project_id) is None
    ):
        raise _fail("project ID is malformed or unsafe")
    maximum_candidates = _exact_count(
        maximum_candidates_per_search,
        "maximum candidates per search",
        positive=True,
    )
    if maximum_candidates > MAXIMUM_ALLOWED_CANDIDATES_PER_SEARCH:
        raise _fail("maximum candidates per search is unreasonably large")

    morphology = Path(morphology_root).expanduser().absolute()
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    _reject_symlink_components(morphology, "morphology preparation root")
    _reject_symlink_components(output.parent, "output root")
    verified = _verify_preparation(morphology)
    searches = _searches(project_id, verified, maximum_candidates)
    if len(searches) != 4 or tuple(
        search.model_class_id for search in searches
    ) != (
        POSITIVE_PULSE_ONLY,
        ORDERED_NEGATIVE_POSITIVE_DOUBLET,
        INDEPENDENT_PULSES,
        INDEPENDENT_PULSES,
    ):
        raise _fail("coarse morphology search structure is inconsistent")

    contract = _contract(verified, searches, maximum_candidates)
    contract_sha256 = _sha256_bytes(_canonical_compact_json_bytes(contract))
    contract_bytes = _stable_json_bytes(contract)
    series_by_id = {
        source["genericSeriesID"]: source for source in verified.series
    }
    dataset_documents: list[tuple[_Search, Mapping[str, Any], bytes]] = []
    dataset_records: list[dict[str, Any]] = []
    work_unit_counts: list[int] = []
    evaluation_counts: list[int] = []
    for search in searches:
        dataset = _dataset(
            search=search,
            series_by_id=series_by_id,
            verified=verified,
            coarse_contract_sha256=contract_sha256,
        )
        dataset_bytes = _stable_json_bytes(dataset)
        sample_count = _safe_sum(
            [len(item["coordinates"]) for item in dataset["series"]],
            "dataset sample count",
        )
        work_unit_count = _safe_work_unit_count(search.coarse_candidate_count)
        evaluation_count = _safe_product(
            [sample_count, search.coarse_candidate_count],
            "sample-candidate evaluation count",
        )
        work_unit_counts.append(work_unit_count)
        evaluation_counts.append(evaluation_count)
        dataset_documents.append((search, dataset, dataset_bytes))
        dataset_records.append(
            {
                "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
                "coarseCandidateCount": search.coarse_candidate_count,
                "datasetID": search.dataset_id,
                "expectedSampleCandidateEvaluationCount": evaluation_count,
                "expectedWorkUnitCount": work_unit_count,
                "fullCandidateCount": search.full_candidate_count,
                "genericSeriesIDs": list(search.generic_series_ids),
                "modelClassID": search.model_class_id,
                "outputFile": search.output_file,
                "outputSHA256": _sha256_bytes(dataset_bytes),
                "sampleCount": sample_count,
                "selectedStride": search.stride,
            }
        )

    project = _project(project_id, searches)
    project_bytes = _stable_json_bytes(project)
    total_work_units = _safe_sum(
        work_unit_counts,
        "total expected work-unit count",
    )
    total_evaluations = _safe_sum(
        evaluation_counts,
        "total sample-candidate evaluation count",
    )
    build_manifest = {
        "algorithmID": COARSE_GRID_ALGORITHM_ID,
        "algorithmVersion": COARSE_GRID_ALGORITHM_VERSION,
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID,
        "buildManifestVersion": BUILD_MANIFEST_VERSION,
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "datasets": dataset_records,
        "identityIsolationStatement": contract["identityIsolationStatement"],
        "inputArtifactManifestFileSHA256": verified.manifest_file_sha256,
        "inputMorphologyContractCanonicalSHA256": (
            verified.contract_canonical_sha256
        ),
        "inputMorphologyContractFileSHA256": verified.contract_file_sha256,
        "inputMorphologyContractID": MORPHOLOGY_CONTRACT_ID,
        "inputMorphologyContractVersion": MORPHOLOGY_CONTRACT_VERSION,
        "inputPreparationID": MORPHOLOGY_PREPARATION_SCHEMA_ID,
        "inputPreparationFileSHA256": verified.preparation_file_sha256,
        "inputPreparationResultSchemaID": MORPHOLOGY_PREPARATION_SCHEMA_ID,
        "inputPreparationResultVersion": MORPHOLOGY_PREPARATION_VERSION,
        "inputPreparationSHA256": verified.preparation_file_sha256,
        "maximumCandidatesPerSearch": maximum_candidates,
        "modelScopeStatement": contract["modelScopeStatement"],
        "noCandidateEvaluationStatement": contract[
            "noCandidateEvaluationStatement"
        ],
        "orderedDatasetIDs": [record["datasetID"] for record in dataset_records],
        "outputHashes": {
            "coarseGridContract": _sha256_bytes(contract_bytes),
            "datasets": {
                record["outputFile"]: record["outputSHA256"]
                for record in dataset_records
            },
            "project": _sha256_bytes(project_bytes),
        },
        "parentHashes": verified.preparation["parentHashes"],
        "parentIDs": verified.preparation["parentIDs"],
        "projectID": project_id,
        "relativeArtifactPaths": {
            "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
            "coarseGridContract": CONTRACT_RELATIVE_PATH,
            "datasets": [record["outputFile"] for record in dataset_records],
            "project": PROJECT_RELATIVE_PATH,
        },
        "sourceDatasetFileSHA256s": dict(verified.series_file_sha256s),
        "totalCoarseCandidateCount": _safe_sum(
            [search.coarse_candidate_count for search in searches],
            "total coarse candidate count",
        ),
        "totalExpectedSampleCandidateEvaluationCount": total_evaluations,
        "totalExpectedWorkUnitCount": total_work_units,
        "workloadIdentities": contract["workloadIdentities"],
    }
    build_manifest_bytes = _stable_json_bytes(build_manifest)
    try:
        _assert_identity_free(
            (
                contract_bytes,
                project_bytes,
                build_manifest_bytes,
                *(item[2] for item in dataset_documents),
            )
        )
    except CoarseGridBuildError as error:
        raise _fail(str(error)) from error
    _assert_identity_isolated(
        (
            contract,
            project,
            build_manifest,
            *(item[1] for item in dataset_documents),
        )
    )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _fail("output parent cannot be created") from error
    _reject_symlink_components(output.parent, "output root")
    if output.exists() or output.is_symlink():
        raise _fail("output root already exists")
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    except OSError as error:
        raise _fail("atomic output staging cannot be created") from error
    try:
        _atomic_write_bytes(staging / CONTRACT_RELATIVE_PATH, contract_bytes)
        for search, _, dataset_bytes in dataset_documents:
            _atomic_write_bytes(staging / search.output_file, dataset_bytes)
        _atomic_write_bytes(staging / PROJECT_RELATIVE_PATH, project_bytes)
        _atomic_write_bytes(
            staging / BUILD_MANIFEST_RELATIVE_PATH,
            build_manifest_bytes,
        )
        if output.exists() or output.is_symlink():
            raise _fail("output root already exists")
        staging.rename(output)
    except Exception as error:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if isinstance(error, AnomalyMorphologyCoarseGridBuildError):
            raise
        raise _fail("atomic coarse morphology publication failed") from error
    return {
        "buildManifest": build_manifest,
        "contract": contract,
        "datasets": [item[1] for item in dataset_documents],
        "project": project,
    }


def build_anomaly_morphology_coarse_grid(
    morphology_root: str | Path,
    *,
    project_id: str,
    output_root: str | Path,
    maximum_candidates_per_search: int = (
        DEFAULT_MAXIMUM_CANDIDATES_PER_SEARCH
    ),
) -> dict[str, Any]:
    """Publish a bounded blind project behind one stable public error type."""

    try:
        return _build_anomaly_morphology_coarse_grid_impl(
            morphology_root,
            project_id=project_id,
            output_root=output_root,
            maximum_candidates_per_search=maximum_candidates_per_search,
        )
    except AnomalyMorphologyCoarseGridBuildError:
        raise
    except (
        AnomalyMorphologyPreparationError,
        CoarseGridBuildError,
        KeyError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise _fail(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bounded blind morphology-grid project from a verified "
            "anomaly-morphology preparation."
        )
    )
    parser.add_argument("--morphology-root", required=True, type=Path)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--maximum-candidates-per-search",
        default=DEFAULT_MAXIMUM_CANDIDATES_PER_SEARCH,
        type=int,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_anomaly_morphology_coarse_grid(
        arguments.morphology_root,
        project_id=arguments.project_id,
        output_root=arguments.output_root,
        maximum_candidates_per_search=(
            arguments.maximum_candidates_per_search
        ),
    )
    manifest = result["buildManifest"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind anomaly-morphology coarse project ready")
    print(f"project ID: {manifest['projectID']}")
    print(f"searches: {len(manifest['datasets'])}")
    print(f"expected work units: {manifest['totalExpectedWorkUnitCount']}")
    print(f"project: {output / PROJECT_RELATIVE_PATH}")
    print(f"build manifest: {output / BUILD_MANIFEST_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
