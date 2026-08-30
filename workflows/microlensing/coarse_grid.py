"""Build a deterministic CurveGrid project from verified blind series."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from openstar_workloads.plugins.curve_grid import (
    DATASET_SCHEMA_ID,
    FAMILY_ID,
    MAX_SAFE_INTEGER,
    PAYLOAD_SCHEMA_ID,
    PLUGIN as CURVE_GRID_PLUGIN,
    RESULT_SCHEMA_ID,
    WORKLOAD_ID,
)
from workflows.microlensing.prepare import (
    BLIND_MANIFEST_SCHEMA_ID,
    PREPARATION_CONTRACT_ID,
    PREPARATION_CONTRACT_SHA256,
    SERIES_SCHEMA_ID,
)


COARSE_GRID_CONTRACT_ID = "openstar.microlensing-coarse-grid.v1"
BUILD_MANIFEST_SCHEMA_ID = "openstar.microlensing-coarse-grid-build.v1"

CENTER_AXIS = {"start": 2245.5, "step": 0.05, "count": 61}
LOG_SCALE_AXIS = {
    "start": math.log(0.25),
    "step": math.log(2.0),
    "count": 9,
}
LOG_SHAPE_AXIS = {
    "start": math.log(0.01),
    "step": math.log(10.0) / 4.0,
    "count": 9,
}
REPRESENTED_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
CANDIDATES_PER_WORK_UNIT = 64
TOTAL_CANDIDATE_COUNT = 61 * 9 * 9
EXPECTED_WORK_UNIT_COUNT = (
    TOTAL_CANDIDATE_COUNT + CANDIDATES_PER_WORK_UNIT - 1
) // CANDIDATES_PER_WORK_UNIT

CONTRACT_RELATIVE_PATH = "coarse-search-contract.json"
DATASET_RELATIVE_PATH = "datasets/primary-series.json"
PROJECT_RELATIVE_PATH = "project.json"
BUILD_MANIFEST_RELATIVE_PATH = "build-manifest.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PREPARATION_MANIFEST_FIELDS = frozenset(
    (
        "benchmarkKind",
        "blindTargetID",
        "orderedSeriesIDs",
        "preparationContractID",
        "preparationContractSHA256",
        "preparationManifestSchemaID",
        "series",
        "totalSampleCount",
        "totalSeriesCount",
    )
)
_SERIES_RECORD_FIELDS = frozenset(
    (
        "coordinateRange",
        "observableRepresentation",
        "sampleCount",
        "seriesFile",
        "seriesID",
        "sha256",
    )
)
_SERIES_FIELDS = frozenset(
    (
        "blindTargetID",
        "coordinates",
        "inverseVariances",
        "seriesID",
        "seriesSchemaID",
        "values",
    )
)
_COORDINATE_RANGE_FIELDS = frozenset(("minimum", "maximum"))
_OUTPUT_IDENTITY_TOKENS = (
    "0302608",
    "OGLE",
    "724L",
    "Hirao",
    "UID_",
    "exoplanetarchive.ipac.caltech.edu",
)


class CoarseGridBuildError(RuntimeError):
    """The blind coarse-grid project cannot be built safely."""


@dataclass(frozen=True, slots=True)
class _VerifiedSeries:
    series_id: str
    sample_count: int
    sha256: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _VerifiedPreparation:
    blind_target_id: str
    manifest_sha256: str
    ordered_series: tuple[_VerifiedSeries, ...]


def _canonical_compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


COARSE_GRID_CONTRACT: dict[str, Any] = {
    "axisProvenanceStatement": (
        "Axes were frozen from blind generic-series characterization without "
        "consulting sealed identity or expected published parameters."
    ),
    "benchmarkKind": "known-event-recovery",
    "candidateCount": TOTAL_CANDIDATE_COUNT,
    "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
    "contractHashRule": (
        "SHA-256 of UTF-8 JSON with sorted keys, no insignificant whitespace, "
        "non-ASCII preserved, and nonfinite numbers forbidden."
    ),
    "contractID": COARSE_GRID_CONTRACT_ID,
    "curveGrid": {
        "centerAxis": {
            **CENTER_AXIS,
            "endpoint": 2248.5,
        },
        "familyID": FAMILY_ID,
        "logScaleAxis": {
            **LOG_SCALE_AXIS,
            "endpoint": math.log(64.0),
            "representedPositiveScales": list(REPRESENTED_SCALES),
        },
        "logShapeAxis": {
            **LOG_SHAPE_AXIS,
            "endpoint": 0.0,
            "endpointPositiveShape": 1.0,
        },
    },
    "modelScopeStatement": (
        "This phase fits only the smooth symmetric radial-amplification "
        "family and does not recover or classify a planetary anomaly."
    ),
    "primarySeriesSelectionRule": {
        "maximum": "sampleCount",
        "selectedSeriesCount": 1,
        "tieBreak": "earliest position in orderedSeriesIDs",
    },
    "schemaTuple": {
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "workloadID": WORKLOAD_ID,
    },
}

COARSE_GRID_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_compact_json_bytes(COARSE_GRID_CONTRACT)
).hexdigest()


def _json_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoarseGridBuildError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise CoarseGridBuildError(f"nonfinite JSON number is forbidden: {value}")


def _decode_json(payload: bytes, description: str) -> Mapping[str, Any]:
    try:
        decoded = payload.decode("utf-8-sig")
        value = json.loads(
            decoded,
            object_pairs_hook=_json_object_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except CoarseGridBuildError:
        raise
    except (UnicodeError, ValueError) as error:
        raise CoarseGridBuildError(f"{description} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise CoarseGridBuildError(f"{description} must be a JSON object")
    return value


def _read_regular_file(path: Path, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CoarseGridBuildError(
            f"{description} is not a regular non-symlink file"
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise CoarseGridBuildError(f"{description} is unreadable") from error


def _safe_blind_file(blind_root: Path, relative_name: str) -> Path:
    if (
        not isinstance(relative_name, str)
        or not relative_name
        or "\\" in relative_name
    ):
        raise CoarseGridBuildError("series path is malformed or unsafe")
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise CoarseGridBuildError("series path is malformed or unsafe")

    current = blind_root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise CoarseGridBuildError(
                "series path traverses a symlink or nondirectory"
            )
    candidate = blind_root.joinpath(*relative.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise CoarseGridBuildError(
            "series input is not a regular non-symlink file"
        )
    try:
        if not candidate.resolve().is_relative_to(blind_root.resolve()):
            raise CoarseGridBuildError("series path escapes the blind root")
    except OSError as error:
        raise CoarseGridBuildError(
            "series path cannot be resolved safely"
        ) from error
    return candidate


def _positive_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise CoarseGridBuildError(f"{field_name} must be a positive integer")
    if value > MAX_SAFE_INTEGER:
        raise CoarseGridBuildError(f"{field_name} exceeds the safe integer range")
    return value


def _safe_product(left: int, right: int, field_name: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise CoarseGridBuildError(f"{field_name} has invalid factors")
    if left and right > MAX_SAFE_INTEGER // left:
        raise CoarseGridBuildError(f"{field_name} exceeds the safe integer range")
    return left * right


def _safe_sum(values: Sequence[int], field_name: str) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise CoarseGridBuildError(f"{field_name} has invalid terms")
        if value > MAX_SAFE_INTEGER - total:
            raise CoarseGridBuildError(
                f"{field_name} exceeds the safe integer range"
            )
        total += value
    return total


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoarseGridBuildError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CoarseGridBuildError(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise CoarseGridBuildError(f"{field_name} must be finite")
    return number


def _required_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoarseGridBuildError(f"{field_name} must be a nonempty string")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CoarseGridBuildError(f"{field_name} must be a lowercase SHA-256")
    return value


def _validate_series_payload(
    payload: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    blind_target_id: str,
) -> None:
    if set(payload) != _SERIES_FIELDS:
        raise CoarseGridBuildError("series file does not match its schema fields")
    if payload.get("seriesSchemaID") != SERIES_SCHEMA_ID:
        raise CoarseGridBuildError("series schema ID is invalid")
    if payload.get("blindTargetID") != blind_target_id:
        raise CoarseGridBuildError("series blind target ID does not match")
    if payload.get("seriesID") != record["seriesID"]:
        raise CoarseGridBuildError("series ID does not match its manifest record")

    arrays: list[list[Any]] = []
    for field_name in ("coordinates", "values", "inverseVariances"):
        values = payload.get(field_name)
        if not isinstance(values, list):
            raise CoarseGridBuildError(f"series {field_name} must be an array")
        arrays.append(values)
    coordinates, values, inverse_variances = arrays
    if len(coordinates) < 3:
        raise CoarseGridBuildError("series must contain at least three samples")
    if not (len(coordinates) == len(values) == len(inverse_variances)):
        raise CoarseGridBuildError("series arrays must have equal length")
    if len(coordinates) != record["sampleCount"]:
        raise CoarseGridBuildError("series sample count does not match manifest")

    numeric_coordinates = tuple(
        _finite_number(value, f"coordinates[{index}]")
        for index, value in enumerate(coordinates)
    )
    tuple(
        _finite_number(value, f"values[{index}]")
        for index, value in enumerate(values)
    )
    numeric_weights = tuple(
        _finite_number(value, f"inverseVariances[{index}]")
        for index, value in enumerate(inverse_variances)
    )
    if any(weight <= 0.0 for weight in numeric_weights):
        raise CoarseGridBuildError("series inverse variances must be positive")

    minimum = min(numeric_coordinates)
    maximum = max(numeric_coordinates)
    if (
        minimum != record["coordinateMinimum"]
        or maximum != record["coordinateMaximum"]
    ):
        raise CoarseGridBuildError("series coordinate range does not match manifest")


def _verify_blind_preparation(prepared_root: Path) -> _VerifiedPreparation:
    blind_root = prepared_root / "blind"
    if blind_root.is_symlink() or not blind_root.is_dir():
        raise CoarseGridBuildError("blind root is not a regular directory")
    manifest_path = blind_root / "preparation-manifest.json"
    manifest_bytes = _read_regular_file(manifest_path, "preparation manifest")
    manifest = _decode_json(manifest_bytes, "preparation manifest")
    if set(manifest) != _PREPARATION_MANIFEST_FIELDS:
        raise CoarseGridBuildError("preparation manifest field set is invalid")
    if manifest.get("preparationManifestSchemaID") != BLIND_MANIFEST_SCHEMA_ID:
        raise CoarseGridBuildError("preparation manifest schema ID is invalid")
    if manifest.get("preparationContractID") != PREPARATION_CONTRACT_ID:
        raise CoarseGridBuildError("preparation contract ID does not match")
    if manifest.get("preparationContractSHA256") != PREPARATION_CONTRACT_SHA256:
        raise CoarseGridBuildError("preparation contract SHA-256 does not match")
    if manifest.get("benchmarkKind") != "known-event-recovery":
        raise CoarseGridBuildError("benchmark kind is invalid")
    blind_target_id = _required_nonempty_string(
        manifest.get("blindTargetID"), "blindTargetID"
    )

    ordered_ids = manifest.get("orderedSeriesIDs")
    records = manifest.get("series")
    if not isinstance(ordered_ids, list) or not ordered_ids:
        raise CoarseGridBuildError("orderedSeriesIDs must be a nonempty array")
    if not isinstance(records, list) or not records:
        raise CoarseGridBuildError("series must be a nonempty array")
    total_series_count = _positive_integer(
        manifest.get("totalSeriesCount"), "totalSeriesCount"
    )
    total_sample_count = _positive_integer(
        manifest.get("totalSampleCount"), "totalSampleCount"
    )
    if (
        total_series_count != len(ordered_ids)
        or total_series_count != len(records)
    ):
        raise CoarseGridBuildError("preparation manifest series counts disagree")

    normalized_order: list[str] = []
    for index, value in enumerate(ordered_ids):
        series_id = _required_nonempty_string(value, f"orderedSeriesIDs[{index}]")
        if series_id in normalized_order:
            raise CoarseGridBuildError("orderedSeriesIDs contains duplicates")
        normalized_order.append(series_id)

    records_by_id: dict[str, Mapping[str, Any]] = {}
    paths: set[str] = set()
    normalized_records: list[Mapping[str, Any]] = []
    for index, value in enumerate(records):
        if not isinstance(value, Mapping) or set(value) != _SERIES_RECORD_FIELDS:
            raise CoarseGridBuildError(f"series record {index} is malformed")
        series_id = _required_nonempty_string(value.get("seriesID"), "seriesID")
        if series_id in records_by_id:
            raise CoarseGridBuildError("series records contain duplicate IDs")
        relative_path = _required_nonempty_string(
            value.get("seriesFile"), "seriesFile"
        )
        if relative_path in paths:
            raise CoarseGridBuildError("series records contain duplicate paths")
        paths.add(relative_path)
        sample_count = _positive_integer(value.get("sampleCount"), "sampleCount")
        if sample_count < 3:
            raise CoarseGridBuildError("series record has fewer than three samples")
        if value.get("observableRepresentation") != "relative-linear-flux":
            raise CoarseGridBuildError("series observable representation is invalid")
        coordinate_range = value.get("coordinateRange")
        if (
            not isinstance(coordinate_range, Mapping)
            or set(coordinate_range) != _COORDINATE_RANGE_FIELDS
        ):
            raise CoarseGridBuildError("series coordinate range is malformed")
        minimum = _finite_number(
            coordinate_range.get("minimum"), "coordinateRange.minimum"
        )
        maximum = _finite_number(
            coordinate_range.get("maximum"), "coordinateRange.maximum"
        )
        if minimum > maximum:
            raise CoarseGridBuildError("series coordinate range is reversed")
        digest = _sha256(value.get("sha256"), "series sha256")
        normalized = {
            **dict(value),
            "coordinateMinimum": minimum,
            "coordinateMaximum": maximum,
            "sampleCount": sample_count,
            "sha256": digest,
        }
        records_by_id[series_id] = normalized
        normalized_records.append(normalized)

    if set(normalized_order) != set(records_by_id):
        raise CoarseGridBuildError(
            "orderedSeriesIDs has missing or unexpected series IDs"
        )

    verified_by_id: dict[str, _VerifiedSeries] = {}
    for record in normalized_records:
        series_path = _safe_blind_file(blind_root, record["seriesFile"])
        series_bytes = _read_regular_file(series_path, "blind series")
        actual_sha256 = hashlib.sha256(series_bytes).hexdigest()
        if actual_sha256 != record["sha256"]:
            raise CoarseGridBuildError("series SHA-256 does not match manifest")
        payload = _decode_json(series_bytes, "blind series")
        _validate_series_payload(
            payload,
            record=record,
            blind_target_id=blind_target_id,
        )
        verified_by_id[record["seriesID"]] = _VerifiedSeries(
            series_id=record["seriesID"],
            sample_count=record["sampleCount"],
            sha256=actual_sha256,
            payload=payload,
        )

    ordered_series = tuple(
        verified_by_id[series_id] for series_id in normalized_order
    )
    calculated_total = _safe_sum(
        [series.sample_count for series in ordered_series],
        "total sample count",
    )
    if total_sample_count != calculated_total:
        raise CoarseGridBuildError("totalSampleCount does not match series records")
    return _VerifiedPreparation(
        blind_target_id=blind_target_id,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        ordered_series=ordered_series,
    )


def _select_primary_series(
    ordered_series: Sequence[_VerifiedSeries],
) -> _VerifiedSeries:
    if not ordered_series:
        raise CoarseGridBuildError("no verified series are available")
    return max(
        enumerate(ordered_series),
        key=lambda item: (item[1].sample_count, -item[0]),
    )[1]


def _curve_grid() -> dict[str, Any]:
    return {
        "candidatesPerWorkUnit": CANDIDATES_PER_WORK_UNIT,
        "centerAxis": dict(CENTER_AXIS),
        "familyID": FAMILY_ID,
        "logScaleAxis": dict(LOG_SCALE_AXIS),
        "logShapeAxis": dict(LOG_SHAPE_AXIS),
    }


def _dataset(
    project_id: str,
    preparation: _VerifiedPreparation,
    selected: _VerifiedSeries,
) -> dict[str, Any]:
    payload = selected.payload
    dataset = {
        "blindTargetID": preparation.blind_target_id,
        "coarseSearchContractID": COARSE_GRID_CONTRACT_ID,
        "coarseSearchContractSHA256": COARSE_GRID_CONTRACT_SHA256,
        "coordinates": list(payload["coordinates"]),
        "curveGrid": _curve_grid(),
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "id": f"{project_id}.primary-series",
        "inverseVariances": list(payload["inverseVariances"]),
        "preparationContractID": PREPARATION_CONTRACT_ID,
        "preparationContractSHA256": PREPARATION_CONTRACT_SHA256,
        "sourceGenericSeriesID": selected.series_id,
        "values": list(payload["values"]),
    }
    try:
        CURVE_GRID_PLUGIN.validate_dataset(dataset)
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise CoarseGridBuildError(
            f"constructed CurveGrid dataset is invalid: {error}"
        ) from error
    return dataset


def _project(project_id: str, dataset_id: str) -> dict[str, Any]:
    return {
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "datasets": [
            {
                "id": dataset_id,
                "path": DATASET_RELATIVE_PATH,
            }
        ],
        "id": project_id,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "workloadID": WORKLOAD_ID,
    }


def _assert_identity_free(documents: Sequence[bytes]) -> None:
    serialized = b"\n".join(documents).decode("utf-8").casefold()
    for token in _OUTPUT_IDENTITY_TOKENS:
        if token.casefold() in serialized:
            raise CoarseGridBuildError(
                "coarse-grid output would contain source identity or provenance"
            )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_coarse_grid_project(
    prepared_root: str | Path,
    *,
    project_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build one deterministic, directly activatable CurveGrid project."""

    if (
        not isinstance(project_id, str)
        or _SAFE_PROJECT_ID.fullmatch(project_id) is None
    ):
        raise CoarseGridBuildError("project ID is malformed or unsafe")
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise CoarseGridBuildError("output root already exists")
    prepared = Path(prepared_root).expanduser().absolute()
    if not prepared.is_dir():
        raise CoarseGridBuildError(
            "prepared root does not exist or is not a directory"
        )

    preparation = _verify_blind_preparation(prepared)
    selected = _select_primary_series(preparation.ordered_series)
    dataset = _dataset(project_id, preparation, selected)
    project = _project(project_id, dataset["id"])

    sample_candidate_evaluations = _safe_product(
        selected.sample_count,
        TOTAL_CANDIDATE_COUNT,
        "sample-candidate evaluation count",
    )
    contract_bytes = _stable_json_bytes(COARSE_GRID_CONTRACT)
    dataset_bytes = _stable_json_bytes(dataset)
    project_bytes = _stable_json_bytes(project)
    build_manifest = {
        "blindTargetID": preparation.blind_target_id,
        "buildManifestSchemaID": BUILD_MANIFEST_SCHEMA_ID,
        "coarseSearchContractID": COARSE_GRID_CONTRACT_ID,
        "coarseSearchContractSHA256": COARSE_GRID_CONTRACT_SHA256,
        "expectedSampleCandidateEvaluationCount": sample_candidate_evaluations,
        "expectedWorkUnitCount": EXPECTED_WORK_UNIT_COUNT,
        "inputSeriesSHA256": selected.sha256,
        "outputDatasetSHA256": hashlib.sha256(dataset_bytes).hexdigest(),
        "preparationManifestSHA256": preparation.manifest_sha256,
        "projectID": project_id,
        "relativeArtifactPaths": {
            "buildManifest": BUILD_MANIFEST_RELATIVE_PATH,
            "coarseSearchContract": CONTRACT_RELATIVE_PATH,
            "dataset": DATASET_RELATIVE_PATH,
            "project": PROJECT_RELATIVE_PATH,
        },
        "selectedSampleCount": selected.sample_count,
        "selectedSeriesID": selected.series_id,
        "totalCandidateCount": TOTAL_CANDIDATE_COUNT,
    }
    build_manifest_bytes = _stable_json_bytes(build_manifest)
    _assert_identity_free(
        (contract_bytes, dataset_bytes, project_bytes, build_manifest_bytes)
    )

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise CoarseGridBuildError("output root already exists") from error
    _atomic_write_bytes(output / CONTRACT_RELATIVE_PATH, contract_bytes)
    _atomic_write_bytes(output / DATASET_RELATIVE_PATH, dataset_bytes)
    _atomic_write_bytes(output / PROJECT_RELATIVE_PATH, project_bytes)
    _atomic_write_bytes(
        output / BUILD_MANIFEST_RELATIVE_PATH,
        build_manifest_bytes,
    )
    return {
        "buildManifest": build_manifest,
        "dataset": dataset,
        "project": project,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bounded generic CurveGrid project from verified blind "
            "microlensing preparation state."
        )
    )
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_coarse_grid_project(
        arguments.prepared_root,
        project_id=arguments.project_id,
        output_root=arguments.output_root,
    )
    manifest = result["buildManifest"]
    output = arguments.output_root.expanduser().absolute()
    print("Blind coarse-grid project ready")
    print(f"project ID: {manifest['projectID']}")
    print(f"selected generic series: {manifest['selectedSeriesID']}")
    print(f"selected samples: {manifest['selectedSampleCount']}")
    print(f"grid candidates: {manifest['totalCandidateCount']}")
    print(f"expected work units: {manifest['expectedWorkUnitCount']}")
    print(f"project: {output / PROJECT_RELATIVE_PATH}")
    print(f"dataset: {output / DATASET_RELATIVE_PATH}")
    print(f"build manifest: {output / BUILD_MANIFEST_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
