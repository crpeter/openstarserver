"""Prepare verified archive tables as identity-isolated weighted time series."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from workflows.microlensing.acquire import (
    ARCHIVE_HOST,
    INVENTORY_RELATIVE_PATH,
    INVENTORY_SCHEMA_ID,
    MANIFEST_RELATIVE_PATH,
    SOURCE_SCRIPT_URL,
    ArchiveIntegrityError,
    _load_existing_manifest,
    _record_matches_file,
    _safe_state_path,
    _validate_inventory_provenance,
    extract_uid,
    sha256_file,
)


PREPARATION_CONTRACT_ID = "openstar.microlensing-preparation.v1"
SERIES_SCHEMA_ID = "openstar.generic-weighted-time-series.v1"
BLIND_MANIFEST_SCHEMA_ID = "openstar.microlensing-blind-preparation.v1"
IDENTITY_SEAL_SCHEMA_ID = "openstar.microlensing-identity-seal.v1"

SUPPORTED_TIME_COLUMNS = ("HJD", "JD")
FLUX_PAIR = ("RELATIVE_FLUX", "FLUX_UNCERTAINTY")
MAGNITUDE_PAIR = ("RELATIVE_MAGNITUDE", "MAGNITUDE_UNCERTAINTY")
MINIMUM_SAMPLE_COUNT = 3

PREPARATION_CONTRACT: dict[str, Any] = {
    "contractID": PREPARATION_CONTRACT_ID,
    "contractHashRule": (
        "SHA-256 of UTF-8 JSON with sorted keys, no insignificant whitespace, "
        "and non-ASCII characters preserved"
    ),
    "supportedTimeColumns": list(SUPPORTED_TIME_COLUMNS),
    "fixedWidthParsingRule": (
        "derive column spans from consecutive pipe positions in the "
        "column-name header and slice every data row with those spans"
    ),
    "validSampleRule": (
        "time, value, and uncertainty are finite and uncertainty is "
        "strictly positive"
    ),
    "observablePairs": {
        "flux": list(FLUX_PAIR),
        "magnitude": list(MAGNITUDE_PAIR),
    },
    "observableSelectionRule": (
        "select the pair with more valid samples and prefer flux on an "
        "exact tie; never mix pairs within one series"
    ),
    "fluxScalingRule": (
        "divide values and uncertainties by the median absolute nonzero "
        "selected flux value, falling back to the median positive "
        "uncertainty"
    ),
    "magnitudeConversionRule": (
        "use the median selected magnitude m_ref; emit "
        "10**(-0.4*(m-m_ref)) with uncertainty "
        "(ln(10)/2.5)*f*sigma_m"
    ),
    "coordinateOriginRule": (
        "subtract floor(minimum selected absolute time across all series)"
    ),
    "orderingRule": (
        "sort source files by archive-relative filename, stable-sort samples "
        "by time, and preserve repeated times as separate measurements"
    ),
    "minimumSampleCount": MINIMUM_SAMPLE_COUNT,
    "identityIsolationRule": (
        "source identity and absolute time remain under sealed/; blind/ "
        "contains only generic series identities and shifted coordinates"
    ),
}


class PreparationError(RuntimeError):
    """Preparation cannot continue without weakening its contract."""


@dataclass(frozen=True, slots=True)
class FixedWidthTable:
    """A structurally parsed table whose cells retain fixed-width positions."""

    columns: tuple[str, ...]
    header_rows: tuple[tuple[str, ...], ...]
    metadata_lines: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def total_data_rows(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class SelectedObservable:
    """One table's selected and normalized linear observable."""

    observable_kind: str
    absolute_coordinates: tuple[float, ...]
    values: tuple[float, ...]
    uncertainties: tuple[float, ...]
    total_data_rows: int
    selected_rows: int
    dropped_rows: int
    normalization_kind: str
    normalization_value: float


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    relative_filename: str
    canonical_source_url: str
    byte_size: int
    sha256: str
    metadata_lines: tuple[str, ...]
    metadata: Mapping[str, tuple[str, ...]]
    selected: SelectedObservable


@dataclass(frozen=True, slots=True)
class _IdentityIsolationChecks:
    raw_substrings: tuple[str, ...]
    exact_json_string_values: tuple[str, ...]


_METADATA_ASSIGNMENT = re.compile(
    r"^\\\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$"
)
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


PREPARATION_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_compact_json_bytes(PREPARATION_CONTRACT)
).hexdigest()


def _parse_header_cells(line: str, line_number: int) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise PreparationError(
            f"line {line_number}: malformed IPAC header delimiters"
        )
    return tuple(cell.strip() for cell in stripped[1:-1].split("|"))


def parse_fixed_width_ipac(path: str | Path) -> FixedWidthTable:
    """Parse IPAC data cells by header-derived fixed-width column spans."""

    table_path = Path(path)
    if table_path.is_symlink() or not table_path.is_file():
        raise PreparationError("source table is not a regular non-symlink file")
    try:
        lines = table_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise PreparationError(f"source table is unreadable: {error}") from error

    metadata_lines: list[str] = []
    header_rows: list[tuple[str, ...]] = []
    rows: list[tuple[str, ...]] = []
    column_spans: tuple[tuple[int, int], ...] | None = None
    final_pipe_position: int | None = None
    data_started = False

    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("\\"):
            metadata_lines.append(line)
            continue
        if stripped.startswith("|"):
            if data_started:
                raise PreparationError(
                    f"line {line_number}: header row appears after data"
                )
            cells = _parse_header_cells(line, line_number)
            if not header_rows:
                pipe_positions = tuple(
                    index for index, character in enumerate(line) if character == "|"
                )
                if len(pipe_positions) < 2:
                    raise PreparationError(
                        f"line {line_number}: malformed IPAC header delimiters"
                    )
                column_spans = tuple(
                    (left + 1, right)
                    for left, right in zip(pipe_positions, pipe_positions[1:])
                )
                final_pipe_position = pipe_positions[-1]
            header_rows.append(cells)
            continue

        data_started = True
        if column_spans is None or final_pipe_position is None:
            raise PreparationError(
                f"line {line_number}: data appears before the IPAC header"
            )
        if "|" in line:
            raise PreparationError(
                f"line {line_number}: unexpected pipe delimiter in data row"
            )
        if line[final_pipe_position:].strip():
            raise PreparationError(
                f"line {line_number}: data extends beyond declared column widths"
            )
        rows.append(tuple(line[start:end].strip() for start, end in column_spans))

    if not header_rows:
        raise PreparationError("missing IPAC header")
    columns = header_rows[0]
    if any(not column for column in columns):
        raise PreparationError("empty column name")
    if len(set(columns)) != len(columns):
        raise PreparationError("duplicate column names")
    for header_index, header_row in enumerate(header_rows[1:], 2):
        if len(header_row) != len(columns):
            raise PreparationError(
                f"header row {header_index} has {len(header_row)} fields; "
                f"expected {len(columns)}"
            )
    if column_spans is None or len(column_spans) != len(columns):
        raise PreparationError("column header has inconsistent pipe delimiters")

    return FixedWidthTable(
        columns=columns,
        header_rows=tuple(header_rows),
        metadata_lines=tuple(metadata_lines),
        rows=tuple(rows),
    )


def _metadata_mapping(lines: Sequence[str]) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = {}
    for line in lines:
        match = _METADATA_ASSIGNMENT.fullmatch(line.strip())
        if match is None:
            continue
        key = match.group(1).upper()
        value = match.group(2).strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        values.setdefault(key, []).append(value)
    return {key: tuple(items) for key, items in sorted(values.items())}


def _finite_number(cell: str) -> float | None:
    if not cell:
        return None
    try:
        value = float(cell)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _valid_samples(
    table: FixedWidthTable,
    *,
    time_column: str,
    pair: tuple[str, str],
) -> list[tuple[float, float, float]]:
    if pair[0] not in table.columns or pair[1] not in table.columns:
        return []
    indices = {name: table.columns.index(name) for name in (time_column, *pair)}
    samples: list[tuple[float, float, float]] = []
    for row in table.rows:
        time = _finite_number(row[indices[time_column]])
        value = _finite_number(row[indices[pair[0]]])
        uncertainty = _finite_number(row[indices[pair[1]]])
        if (
            time is not None
            and value is not None
            and uncertainty is not None
            and uncertainty > 0.0
        ):
            samples.append((time, value, uncertainty))
    return samples


def _require_finite_emission(
    coordinates: Sequence[float],
    values: Sequence[float],
    uncertainties: Sequence[float],
) -> None:
    if not all(math.isfinite(value) for value in coordinates):
        raise PreparationError("prepared coordinates are not finite")
    if not all(math.isfinite(value) for value in values):
        raise PreparationError("prepared values are not finite")
    if not all(
        math.isfinite(value) and value > 0.0 for value in uncertainties
    ):
        raise PreparationError("prepared uncertainties are not finite and positive")
    for uncertainty in uncertainties:
        variance = uncertainty * uncertainty
        inverse_variance = 1.0 / variance if variance != 0.0 else math.inf
        if not math.isfinite(inverse_variance) or inverse_variance <= 0.0:
            raise PreparationError(
                "prepared inverse variances are not finite and positive"
            )


def select_and_normalize_observable(table: FixedWidthTable) -> SelectedObservable:
    """Select one supported pair and normalize it without row-wise mixing."""

    time_columns = [
        column for column in SUPPORTED_TIME_COLUMNS if column in table.columns
    ]
    if len(time_columns) != 1:
        raise PreparationError(
            "table must contain exactly one supported time column "
            f"from {SUPPORTED_TIME_COLUMNS!r}"
        )
    time_column = time_columns[0]
    flux_samples = _valid_samples(table, time_column=time_column, pair=FLUX_PAIR)
    magnitude_samples = _valid_samples(
        table,
        time_column=time_column,
        pair=MAGNITUDE_PAIR,
    )

    if len(flux_samples) >= len(magnitude_samples):
        observable_kind = FLUX_PAIR[0]
        selected = flux_samples
    else:
        observable_kind = MAGNITUDE_PAIR[0]
        selected = magnitude_samples
    if len(selected) < MINIMUM_SAMPLE_COUNT:
        raise PreparationError(
            f"selected observable has fewer than {MINIMUM_SAMPLE_COUNT} valid samples"
        )

    selected = sorted(selected, key=lambda sample: sample[0])
    coordinates = tuple(sample[0] for sample in selected)
    raw_values = tuple(sample[1] for sample in selected)
    raw_uncertainties = tuple(sample[2] for sample in selected)

    if observable_kind == FLUX_PAIR[0]:
        absolute_nonzero = [abs(value) for value in raw_values if value != 0.0]
        if absolute_nonzero:
            normalization_value = float(statistics.median(absolute_nonzero))
        else:
            positive_uncertainties = [
                value for value in raw_uncertainties if value > 0.0
            ]
            if not positive_uncertainties:
                raise PreparationError("flux series has no finite positive scale")
            normalization_value = float(
                statistics.median(positive_uncertainties)
            )
        if not math.isfinite(normalization_value) or normalization_value <= 0.0:
            raise PreparationError("flux series has no finite positive scale")
        values = tuple(value / normalization_value for value in raw_values)
        uncertainties = tuple(
            value / normalization_value for value in raw_uncertainties
        )
        normalization_kind = "fluxScale"
    else:
        normalization_value = float(statistics.median(raw_values))
        if not math.isfinite(normalization_value):
            raise PreparationError("magnitude reference is not finite")
        converted_values: list[float] = []
        converted_uncertainties: list[float] = []
        for magnitude, magnitude_uncertainty in zip(
            raw_values, raw_uncertainties
        ):
            try:
                value = 10.0 ** (-0.4 * (magnitude - normalization_value))
            except OverflowError as error:
                raise PreparationError(
                    "magnitude conversion produced a nonfinite value"
                ) from error
            uncertainty = (
                (math.log(10.0) / 2.5)
                * value
                * magnitude_uncertainty
            )
            converted_values.append(value)
            converted_uncertainties.append(uncertainty)
        values = tuple(converted_values)
        uncertainties = tuple(converted_uncertainties)
        normalization_kind = "referenceMagnitude"

    _require_finite_emission(coordinates, values, uncertainties)
    return SelectedObservable(
        observable_kind=observable_kind,
        absolute_coordinates=coordinates,
        values=values,
        uncertainties=uncertainties,
        total_data_rows=table.total_data_rows,
        selected_rows=len(selected),
        dropped_rows=table.total_data_rows - len(selected),
        normalization_kind=normalization_kind,
        normalization_value=normalization_value,
    )


def _load_json_mapping(path: Path, description: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{description} is not a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{description} is unreadable") from error
    if not isinstance(value, Mapping):
        raise PreparationError(f"{description} is not a JSON object")
    return value


def _verify_complete_inventory(
    root: Path,
    manifest: Mapping[str, Any],
    record_names: set[str],
    manifest_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    inventory_path = _safe_state_path(root, INVENTORY_RELATIVE_PATH)
    inventory = _load_json_mapping(inventory_path, "archive inventory")
    expected_count = manifest["expectedFileCount"]
    complete = bool(
        inventory.get("inventorySchemaID") == INVENTORY_SCHEMA_ID
        and inventory.get("manifestComplete") is True
        and inventory.get("manifestSHA256") == manifest_sha256
        and type(inventory.get("expectedFileCount")) is int
        and inventory.get("expectedFileCount") == expected_count
        and type(inventory.get("totalFiles")) is int
        and inventory.get("totalFiles") == len(record_names)
        and type(inventory.get("parsedFileCount")) is int
        and inventory.get("parsedFileCount") == len(record_names)
        and type(inventory.get("parseFailureCount")) is int
        and inventory.get("parseFailureCount") == 0
        and inventory.get("parseFailures") == []
        and isinstance(inventory.get("files"), list)
        and len(inventory["files"]) == len(record_names)
        and isinstance(inventory.get("filesGroupedByUID"), Mapping)
    )
    if not complete:
        raise PreparationError(
            "archive inventory must be complete, current, and have zero parse failures"
        )
    filenames: set[str] = set()
    expected_groups: dict[str, list[str]] = {}
    for item in inventory["files"]:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("filename"), str
        ):
            raise PreparationError("archive inventory has a malformed file record")
        filename = item["filename"]
        if filename in filenames:
            raise PreparationError("archive inventory has duplicate file records")
        filenames.add(filename)
        filename_uid = extract_uid(filename)
        if filename_uid is None or item.get("uid") != filename_uid:
            raise PreparationError("archive inventory has invalid UID provenance")
        expected_groups.setdefault(filename_uid, []).append(filename)
    if filenames != record_names:
        raise PreparationError(
            "archive inventory does not contain the exact manifest file set"
        )
    expected_groups = {
        uid: sorted(names) for uid, names in sorted(expected_groups.items())
    }
    if inventory["filesGroupedByUID"] != expected_groups:
        raise PreparationError("archive inventory UID grouping is inconsistent")
    return inventory, sha256_file(inventory_path)


def _selected_filenames(
    inventory: Mapping[str, Any], uid: str
) -> tuple[str, ...]:
    grouped = inventory["filesGroupedByUID"]
    selected = grouped.get(uid)
    if not isinstance(selected, list) or not selected:
        raise PreparationError(f"archive inventory has no files for UID {uid}")
    filenames: list[str] = []
    seen: set[str] = set()
    for filename in selected:
        if not isinstance(filename, str) or extract_uid(filename) != uid:
            raise PreparationError("archive inventory has an invalid UID grouping")
        if filename in seen:
            raise PreparationError("archive inventory UID grouping has duplicates")
        seen.add(filename)
        filenames.append(filename)
    return tuple(sorted(filenames))


def _consistent_star_id(
    prepared_sources: Sequence[_PreparedSource],
) -> str:
    star_ids: list[str] = []
    for source in prepared_sources:
        values = source.metadata.get("STAR_ID", ())
        if not values or any(not value for value in values):
            raise PreparationError(
                f"STAR_ID metadata is missing from {source.relative_filename}"
            )
        if len(set(values)) != 1:
            raise PreparationError(
                f"STAR_ID metadata is inconsistent within {source.relative_filename}"
            )
        star_ids.append(values[0])
    if len(set(star_ids)) != 1:
        raise PreparationError("STAR_ID metadata is inconsistent across selected files")
    return star_ids[0]


def _relevant_metadata(
    metadata: Mapping[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    relevant = {}
    for key, values in metadata.items():
        if key == "STAR_ID" or "REFERENCE" in key or "BIBCODE" in key:
            relevant[key] = list(values)
    return relevant


def _source_identity_checks(
    uid: str,
    star_id: str,
    prepared_sources: Sequence[_PreparedSource],
) -> _IdentityIsolationChecks:
    raw_substrings = {
        uid,
        "OGLE",
        star_id,
        SOURCE_SCRIPT_URL,
        ARCHIVE_HOST,
    }
    raw_substrings.update(
        word
        for word in re.findall(r"[A-Za-z0-9]+", star_id)
        if len(word) >= 4 and sum(character.isalpha() for character in word) >= 2
    )
    raw_substrings.update(
        {
            "reference",
            "bibcode",
            "observatory",
            "telescope",
            "instrument",
            "filter",
        }
    )
    exact_json_string_values = {"RA", "DEC"}
    for source in prepared_sources:
        raw_substrings.add(source.relative_filename)
        raw_substrings.add(Path(source.relative_filename).name)
        raw_substrings.add(source.canonical_source_url)
        for values in source.metadata.values():
            exact_json_string_values.update(value for value in values if value)
    return _IdentityIsolationChecks(
        raw_substrings=tuple(
            sorted(
                (token for token in raw_substrings if token),
                key=str.casefold,
            )
        ),
        exact_json_string_values=tuple(
            sorted(exact_json_string_values, key=str.casefold)
        ),
    )


def _assert_identity_isolated(
    documents: Sequence[bytes], checks: _IdentityIsolationChecks
) -> None:
    serialized = b"\n".join(documents).decode("utf-8").casefold()
    for token in checks.raw_substrings:
        if token.casefold() in serialized:
            raise PreparationError(
                "blind output would contain source identity or provenance"
            )
    for value in checks.exact_json_string_values:
        json_literal = json.dumps(value, ensure_ascii=False).casefold()
        if json_literal in serialized:
            raise PreparationError(
                "blind output would contain source identity or provenance"
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


def prepare_archive(
    archive_root: str | Path,
    *,
    uid: str,
    blind_target_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Prepare one verified UID and write sealed plus blind JSON outputs."""

    if not isinstance(uid, str) or re.fullmatch(r"[0-9]+", uid) is None:
        raise PreparationError("UID must be a nonempty decimal string")
    if not isinstance(blind_target_id, str) or not blind_target_id.strip():
        raise PreparationError("blind target ID must be a nonempty string")

    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise PreparationError("output root already exists")

    root = Path(archive_root).expanduser().resolve()
    if not root.is_dir():
        raise PreparationError("archive root does not exist or is not a directory")
    manifest_path = root / MANIFEST_RELATIVE_PATH
    try:
        manifest = _load_existing_manifest(manifest_path, refresh=False)
        if manifest is None:
            raise PreparationError("archive manifest does not exist")
        if manifest.get("archiveComplete") is not True:
            raise PreparationError("archive manifest must be complete")
        records = _validate_inventory_provenance(root, manifest)
    except ArchiveIntegrityError as error:
        raise PreparationError(f"archive provenance verification failed: {error}") from error

    manifest_sha256 = sha256_file(manifest_path)
    inventory, inventory_sha256 = _verify_complete_inventory(
        root,
        manifest,
        set(records),
        manifest_sha256,
    )
    selected_filenames = _selected_filenames(inventory, uid)

    inventory_filenames = {
        item["filename"] for item in inventory["files"] if isinstance(item, Mapping)
    }
    prepared_sources: list[_PreparedSource] = []
    for relative_filename in selected_filenames:
        record = records.get(relative_filename)
        if record is None or relative_filename not in inventory_filenames:
            raise PreparationError(
                f"selected source is absent from verified archive state: {relative_filename}"
            )
        source_path = _safe_state_path(root, relative_filename)
        if not _record_matches_file(record, source_path):
            raise PreparationError(
                f"selected source size or SHA-256 mismatch: {relative_filename}"
            )
        table = parse_fixed_width_ipac(source_path)
        metadata = _metadata_mapping(table.metadata_lines)
        prepared_sources.append(
            _PreparedSource(
                relative_filename=relative_filename,
                canonical_source_url=record["canonicalSourceURL"],
                byte_size=record["byteSize"],
                sha256=record["sha256"],
                metadata_lines=table.metadata_lines,
                metadata=metadata,
                selected=select_and_normalize_observable(table),
            )
        )

    if not prepared_sources:
        raise PreparationError("at least one selected source file is required")
    star_id = _consistent_star_id(prepared_sources)
    common_origin = math.floor(
        min(
            coordinate
            for source in prepared_sources
            for coordinate in source.selected.absolute_coordinates
        )
    )

    series_payloads: list[tuple[str, bytes]] = []
    blind_series_records: list[dict[str, Any]] = []
    sealed_sources: list[dict[str, Any]] = []
    series_source_mapping: list[dict[str, str]] = []

    for index, source in enumerate(prepared_sources, 1):
        series_id = f"series-{index:03d}"
        filename = f"{series_id}.json"
        shifted_coordinates = tuple(
            value - common_origin
            for value in source.selected.absolute_coordinates
        )
        _require_finite_emission(
            shifted_coordinates,
            source.selected.values,
            source.selected.uncertainties,
        )
        inverse_variances = tuple(
            1.0 / (uncertainty * uncertainty)
            for uncertainty in source.selected.uncertainties
        )
        series_payload = {
            "blindTargetID": blind_target_id,
            "coordinates": list(shifted_coordinates),
            "inverseVariances": list(inverse_variances),
            "seriesID": series_id,
            "seriesSchemaID": SERIES_SCHEMA_ID,
            "values": list(source.selected.values),
        }
        series_bytes = _stable_json_bytes(series_payload)
        series_sha256 = hashlib.sha256(series_bytes).hexdigest()
        series_payloads.append((filename, series_bytes))
        blind_series_records.append(
            {
                "coordinateRange": {
                    "maximum": max(shifted_coordinates),
                    "minimum": min(shifted_coordinates),
                },
                "observableRepresentation": "relative-linear-flux",
                "sampleCount": source.selected.selected_rows,
                "seriesFile": f"series/{filename}",
                "seriesID": series_id,
                "sha256": series_sha256,
            }
        )
        normalization = {
            "kind": source.selected.normalization_kind,
            (
                "scale"
                if source.selected.normalization_kind == "fluxScale"
                else "referenceMagnitude"
            ): source.selected.normalization_value,
        }
        sealed_sources.append(
            {
                "byteSize": source.byte_size,
                "canonicalSourceURL": source.canonical_source_url,
                "droppedRowCount": source.selected.dropped_rows,
                "genericSeriesID": series_id,
                "metadataLines": list(source.metadata_lines),
                "normalization": normalization,
                "originalRelevantMetadata": _relevant_metadata(source.metadata),
                "selectedObservableKind": source.selected.observable_kind,
                "selectedRowCount": source.selected.selected_rows,
                "sha256": source.sha256,
                "sourceFilename": source.relative_filename,
                "totalDataRowCount": source.selected.total_data_rows,
            }
        )
        series_source_mapping.append(
            {
                "genericSeriesID": series_id,
                "sourceFilename": source.relative_filename,
            }
        )

    blind_manifest = {
        "benchmarkKind": "known-event-recovery",
        "blindTargetID": blind_target_id,
        "orderedSeriesIDs": [record["seriesID"] for record in blind_series_records],
        "preparationContractID": PREPARATION_CONTRACT_ID,
        "preparationContractSHA256": PREPARATION_CONTRACT_SHA256,
        "preparationManifestSchemaID": BLIND_MANIFEST_SCHEMA_ID,
        "series": blind_series_records,
        "totalSampleCount": sum(
            record["sampleCount"] for record in blind_series_records
        ),
        "totalSeriesCount": len(blind_series_records),
    }
    identity_seal = {
        "absoluteCommonTimeOrigin": common_origin,
        "archiveInventorySHA256": inventory_sha256,
        "archiveManifestSHA256": manifest_sha256,
        "benchmarkKind": "known-event-recovery",
        "blindTargetID": blind_target_id,
        "identitySealSchemaID": IDENTITY_SEAL_SCHEMA_ID,
        "preparationContract": PREPARATION_CONTRACT,
        "preparationContractID": PREPARATION_CONTRACT_ID,
        "preparationContractSHA256": PREPARATION_CONTRACT_SHA256,
        "seriesSourceMapping": series_source_mapping,
        "sources": sealed_sources,
        "starID": star_id,
        "uid": uid,
    }

    blind_manifest_bytes = _stable_json_bytes(blind_manifest)
    identity_seal_bytes = _stable_json_bytes(identity_seal)
    blind_documents = [blind_manifest_bytes] + [
        payload for _, payload in series_payloads
    ]
    _assert_identity_isolated(
        blind_documents,
        _source_identity_checks(uid, star_id, prepared_sources),
    )

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PreparationError("output root already exists") from error
    _atomic_write_bytes(output / "sealed" / "identity-seal.json", identity_seal_bytes)
    _atomic_write_bytes(
        output / "blind" / "preparation-manifest.json", blind_manifest_bytes
    )
    for filename, payload in series_payloads:
        _atomic_write_bytes(output / "blind" / "series" / filename, payload)
    return {
        "blindManifest": blind_manifest,
        "identitySeal": identity_seal,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one verified archive UID as identity-isolated generic "
            "weighted time series for a known-event recovery benchmark."
        )
    )
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--blind-target-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    prepare_archive(
        arguments.archive_root,
        uid=arguments.uid,
        blind_target_id=arguments.blind_target_id,
        output_root=arguments.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
