"""Acquire and structurally inventory the contributed MICROLENSING bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


ARCHIVE_HOST = "exoplanetarchive.ipac.caltech.edu"
SOURCE_SCRIPT_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/"
    "bulk_data_download/wget_MICROLENSING.bat"
)
SOURCE_SCRIPT_RELATIVE_PATH = "source/wget_MICROLENSING.bat"
MANIFEST_RELATIVE_PATH = "archive-manifest.json"
INVENTORY_RELATIVE_PATH = "archive-inventory.json"
MANIFEST_SCHEMA_ID = "openstar.microlensing-archive-manifest.v1"
INVENTORY_SCHEMA_ID = "openstar.microlensing-archive-inventory.v1"
HTTP_TIMEOUT_SECONDS = 120

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.tbl$")
_UID_FILENAME = re.compile(r"^UID_([0-9]+)_PLC_[0-9]+\.tbl$")


class ArchiveAcquisitionError(RuntimeError):
    """The archive could not be acquired without weakening its contract."""


class WgetScriptError(ArchiveAcquisitionError):
    """The official source script did not match the accepted structure."""


class ArchiveIntegrityError(ArchiveAcquisitionError):
    """Existing state conflicts with recorded or newly retrieved bytes."""


class TableStructureError(ValueError):
    """An archive table could not be structurally inventoried."""


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    canonical_source_url: str
    local_relative_filename: str


@dataclass(frozen=True, slots=True)
class TableStructure:
    columns: tuple[str, ...]
    header_rows: tuple[tuple[str, ...], ...]
    metadata_lines: tuple[str, ...]
    row_count: int
    schema_signature: str


FetchToPath = Callable[[str, Path], None]


def _canonical_nasa_url(
    value: str,
    *,
    normalize_http: bool,
) -> str:
    if not isinstance(value, str) or not value:
        raise WgetScriptError("source URL must be a nonempty string")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise WgetScriptError(f"malformed source URL: {value!r}") from error

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise WgetScriptError(f"unexpected URL scheme: {scheme or '<none>'}")
    if not normalize_http and scheme != "https":
        raise WgetScriptError("archive retrieval requires HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise WgetScriptError("source URLs must not contain credentials")
    if (parsed.hostname or "").lower() != ARCHIVE_HOST:
        raise WgetScriptError("source URL host is not the NASA archive")
    expected_ports = {None, 80} if scheme == "http" else {None, 443}
    if port not in expected_ports:
        raise WgetScriptError("source URL uses an unexpected port")
    if parsed.query or parsed.fragment:
        raise WgetScriptError("source URLs must not contain query or fragment data")
    if not parsed.path.startswith("/"):
        raise WgetScriptError("source URL path must be absolute")
    if re.search(r"%(?![0-9A-Fa-f]{2})", parsed.path):
        raise WgetScriptError("source URL path has malformed percent encoding")

    decoded_path = parsed.path
    for _ in range(4):
        expanded_path = urllib.parse.unquote(decoded_path)
        if expanded_path == decoded_path:
            break
        decoded_path = expanded_path
    else:
        raise WgetScriptError("source URL path has excessive encoding")
    if any(character.isspace() or ord(character) < 32 for character in decoded_path):
        raise WgetScriptError("source URL path contains unsafe characters")
    if "\\" in decoded_path or any(
        part in {"", ".", ".."}
        for part in decoded_path.split("/")[1:]
    ):
        raise WgetScriptError("source URL path is unsafe")

    return urllib.parse.urlunsplit(
        ("https", ARCHIVE_HOST, parsed.path, "", "")
    )


def _safe_output_filename(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_FILENAME.fullmatch(value):
        raise WgetScriptError(f"unsafe output filename: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise WgetScriptError(f"unsafe output filename: {value!r}")
    return value


def _parse_wget_line(line: str, line_number: int) -> ArchiveEntry:
    try:
        tokens = shlex.split(line, comments=False, posix=True)
    except ValueError as error:
        raise WgetScriptError(
            f"line {line_number}: malformed shell quoting"
        ) from error
    if not tokens or tokens[0] != "wget":
        raise WgetScriptError(f"line {line_number}: expected a wget entry")

    output_name = None
    positional = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-O", "--output-document"}:
            if output_name is not None or index + 1 >= len(tokens):
                raise WgetScriptError(
                    f"line {line_number}: malformed output option"
                )
            output_name = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--output-document="):
            if output_name is not None:
                raise WgetScriptError(
                    f"line {line_number}: duplicate output option"
                )
            output_name = token.split("=", 1)[1]
            index += 1
            continue
        if token in {"-a", "--append-output"}:
            if index + 1 >= len(tokens):
                raise WgetScriptError(
                    f"line {line_number}: malformed append-output option"
                )
            index += 2
            continue
        if token.startswith("--append-output="):
            index += 1
            continue
        if token.startswith("-"):
            raise WgetScriptError(
                f"line {line_number}: unsupported wget option {token!r}"
            )
        positional.append(token)
        index += 1

    if output_name is None:
        raise WgetScriptError(f"line {line_number}: missing output filename")
    if len(positional) != 1:
        raise WgetScriptError(
            f"line {line_number}: expected exactly one source URL"
        )

    filename = _safe_output_filename(output_name)
    canonical_url = _canonical_nasa_url(
        positional[0],
        normalize_http=True,
    )
    return ArchiveEntry(
        canonical_source_url=canonical_url,
        local_relative_filename=f"data/{filename}",
    )


def parse_wget_script(script: bytes | str) -> tuple[ArchiveEntry, ...]:
    """Parse, validate, normalize, deduplicate, and sort wget entries."""

    if isinstance(script, bytes):
        try:
            text = script.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise WgetScriptError("source script is not valid UTF-8") from error
    elif isinstance(script, str):
        text = script
    else:
        raise TypeError("script must be bytes or text")

    entries_by_name: dict[str, ArchiveEntry] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entry = _parse_wget_line(stripped, line_number)
        prior = entries_by_name.get(entry.local_relative_filename)
        if prior is not None:
            if prior.canonical_source_url != entry.canonical_source_url:
                raise WgetScriptError(
                    "conflicting URLs for output "
                    f"{entry.local_relative_filename!r}"
                )
            continue
        entries_by_name[entry.local_relative_filename] = entry

    if not entries_by_name:
        raise WgetScriptError("source script contains no wget entries")
    return tuple(
        sorted(
            entries_by_name.values(),
            key=lambda item: (
                item.local_relative_filename,
                item.canonical_source_url,
            ),
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ArchiveIntegrityError(f"archive path is not a regular file: {path}")
    return path.stat().st_size, sha256_file(path)


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_stable_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_state_path(root: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ArchiveIntegrityError(
            f"unsafe state-relative path: {relative_name!r}"
        )
    candidate = root.joinpath(*relative.parts)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if not candidate.parent.resolve().is_relative_to(root.resolve()):
        raise ArchiveIntegrityError(
            f"state-relative path escapes output root: {relative_name!r}"
        )
    return candidate


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _publish_temporary(temporary: Path, destination: Path) -> None:
    if temporary.is_symlink() or not temporary.is_file():
        raise ArchiveAcquisitionError("fetcher did not produce a regular file")
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


class _NasaOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request,
        file_pointer,
        code,
        message,
        headers,
        new_url,
    ):
        _canonical_nasa_url(new_url, normalize_http=False)
        return super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )


def fetch_to_path(url: str, destination: Path) -> None:
    """Stream one NASA archive URL to a caller-provided temporary path."""

    canonical_url = _canonical_nasa_url(url, normalize_http=False)
    request = urllib.request.Request(
        canonical_url,
        headers={"User-Agent": "OpenStar-Microlensing-Archive/1"},
    )
    opener = urllib.request.build_opener(_NasaOnlyRedirectHandler())
    with opener.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        final_url = response.geturl()
        _canonical_nasa_url(final_url, normalize_http=False)
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())


def _timestamp(value: datetime | None) -> str:
    selected = value or datetime.now(timezone.utc)
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("retrieval time must be timezone-aware")
    return (
        selected.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _load_existing_manifest(path: Path, refresh: bool) -> Mapping[str, Any] | None:
    if not _path_present(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise ArchiveIntegrityError("archive manifest is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if refresh:
            return None
        raise ArchiveIntegrityError(
            "archive manifest is unreadable; use --refresh to replace state"
        ) from error
    structurally_valid = bool(
        isinstance(value, Mapping)
        and value.get("manifestSchemaID") == MANIFEST_SCHEMA_ID
        and isinstance(value.get("sourceScript"), Mapping)
        and isinstance(value.get("files"), list)
        and type(value.get("expectedFileCount")) is int
        and value["expectedFileCount"] >= 0
        and type(value.get("archiveComplete")) is bool
        and len(value["files"]) <= value["expectedFileCount"]
        and value["archiveComplete"]
        == (len(value["files"]) == value["expectedFileCount"])
    )
    if not structurally_valid:
        if refresh:
            return None
        raise ArchiveIntegrityError(
            "archive manifest is malformed; use --refresh to replace state"
        )
    source_record = value["sourceScript"]
    source_record_valid = bool(
        source_record.get("canonicalSourceURL") == SOURCE_SCRIPT_URL
        and source_record.get("localRelativeFilename")
        == SOURCE_SCRIPT_RELATIVE_PATH
        and type(source_record.get("byteSize")) is int
        and source_record["byteSize"] >= 0
        and isinstance(source_record.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", source_record["sha256"])
    )
    if not source_record_valid:
        if refresh:
            return None
        raise ArchiveIntegrityError(
            "archive manifest has invalid source provenance; "
            "use --refresh to replace state"
        )
    try:
        _recorded_timestamp(source_record, "source script record")
        _existing_file_records(value, refresh=refresh)
    except ArchiveIntegrityError:
        if refresh:
            return None
        raise
    return value


def _existing_file_records(
    manifest: Mapping[str, Any] | None,
    *,
    refresh: bool,
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    if manifest is None:
        return records
    for item in manifest["files"]:
        if not isinstance(item, Mapping):
            if refresh:
                return {}
            raise ArchiveIntegrityError("archive manifest has a malformed file record")
        relative_name = item.get("localRelativeFilename")
        if not isinstance(relative_name, str) or relative_name in records:
            if refresh:
                return {}
            raise ArchiveIntegrityError(
                "archive manifest has invalid or duplicate file records"
            )
        relative_path = PurePosixPath(relative_name)
        valid_record = bool(
            len(relative_path.parts) == 2
            and relative_path.parts[0] == "data"
            and _SAFE_FILENAME.fullmatch(relative_path.parts[1])
            and type(item.get("byteSize")) is int
            and item["byteSize"] >= 0
            and isinstance(item.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            and item.get("sourceScriptURL") == SOURCE_SCRIPT_URL
            and isinstance(item.get("sourceScriptSHA256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sourceScriptSHA256"])
        )
        if not valid_record:
            if refresh:
                return {}
            raise ArchiveIntegrityError(
                f"archive manifest has an invalid record for {relative_name}"
            )
        try:
            _recorded_timestamp(item, f"file record for {relative_name}")
        except ArchiveIntegrityError:
            if refresh:
                return {}
            raise
        records[relative_name] = item
    return records


def _record_matches_file(record: Mapping[str, Any], path: Path) -> bool:
    try:
        expected_size = record["byteSize"]
        expected_sha256 = record["sha256"]
        if type(expected_size) is not int or expected_size < 0:
            return False
        if (
            not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        ):
            return False
        actual_size, actual_sha256 = _file_identity(path)
    except (KeyError, OSError, ArchiveIntegrityError):
        return False
    return actual_size == expected_size and actual_sha256 == expected_sha256


def _recorded_timestamp(record: Mapping[str, Any], description: str) -> str:
    value = record.get("retrievedAtUTC")
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        value,
    ):
        raise ArchiveIntegrityError(
            f"{description} has an invalid UTC retrieval timestamp"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ArchiveIntegrityError(
            f"{description} has an invalid UTC retrieval timestamp"
        ) from error
    return value


def _source_record(
    *,
    retrieved_at: str,
    byte_size: int,
    sha256: str,
) -> dict[str, Any]:
    return {
        "byteSize": byte_size,
        "canonicalSourceURL": SOURCE_SCRIPT_URL,
        "localRelativeFilename": SOURCE_SCRIPT_RELATIVE_PATH,
        "retrievedAtUTC": retrieved_at,
        "sha256": sha256,
    }


def _archive_record(
    entry: ArchiveEntry,
    *,
    retrieved_at: str,
    byte_size: int,
    sha256: str,
    source_script_sha256: str,
) -> dict[str, Any]:
    return {
        "byteSize": byte_size,
        "canonicalSourceURL": entry.canonical_source_url,
        "localRelativeFilename": entry.local_relative_filename,
        "retrievedAtUTC": retrieved_at,
        "sha256": sha256,
        "sourceScriptSHA256": source_script_sha256,
        "sourceScriptURL": SOURCE_SCRIPT_URL,
    }


def _manifest(
    source_record: Mapping[str, Any],
    file_records: Sequence[Mapping[str, Any]],
    *,
    expected_file_count: int,
) -> dict[str, Any]:
    records = sorted(
        (dict(item) for item in file_records),
        key=lambda item: (
            item["localRelativeFilename"],
            item["canonicalSourceURL"],
        ),
    )
    return {
        "archiveComplete": len(records) == expected_file_count,
        "expectedFileCount": expected_file_count,
        "files": records,
        "manifestSchemaID": MANIFEST_SCHEMA_ID,
        "sourceScript": dict(source_record),
    }


def _fetch_temporary(fetcher: FetchToPath, url: str, destination: Path) -> Path:
    temporary = _temporary_path(destination)
    try:
        fetcher(url, temporary)
        if temporary.is_symlink() or not temporary.is_file():
            raise ArchiveAcquisitionError(
                f"fetcher did not produce a regular file for {url}"
            )
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def acquire_archive(
    output_root: str | Path,
    *,
    fetcher: FetchToPath = fetch_to_path,
    refresh: bool = False,
    retrieved_at: datetime | None = None,
    write_inventory: bool = False,
) -> dict[str, Any]:
    """Acquire the current script and every listed file into external state."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = _safe_state_path(root, MANIFEST_RELATIVE_PATH)
    existing_manifest = _load_existing_manifest(manifest_path, refresh)
    existing_records = _existing_file_records(
        existing_manifest,
        refresh=refresh,
    )

    source_path = _safe_state_path(root, SOURCE_SCRIPT_RELATIVE_PATH)
    source_temporary = _fetch_temporary(
        fetcher,
        SOURCE_SCRIPT_URL,
        source_path,
    )
    try:
        script_bytes = source_temporary.read_bytes()
        entries = parse_wget_script(script_bytes)
        fetched_source_size, fetched_source_sha256 = _file_identity(
            source_temporary
        )

        existing_source = (
            existing_manifest.get("sourceScript")
            if existing_manifest is not None
            else None
        )
        if isinstance(existing_source, Mapping):
            try:
                existing_source_timestamp = _recorded_timestamp(
                    existing_source,
                    "source script record",
                )
            except ArchiveIntegrityError:
                if not refresh:
                    raise
                existing_source = None
                existing_source_timestamp = None
        else:
            existing_source_timestamp = None
        source_matches_record = bool(
            isinstance(existing_source, Mapping)
            and existing_source.get("canonicalSourceURL") == SOURCE_SCRIPT_URL
            and existing_source.get("localRelativeFilename")
            == SOURCE_SCRIPT_RELATIVE_PATH
            and type(existing_source.get("byteSize")) is int
            and existing_source.get("byteSize") == fetched_source_size
            and isinstance(existing_source.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", existing_source["sha256"])
            and existing_source.get("sha256") == fetched_source_sha256
        )
        if (
            existing_manifest is not None
            and not source_matches_record
            and not refresh
        ):
            raise ArchiveIntegrityError(
                "source script changed or has invalid provenance; "
                "use --refresh to replace it"
            )
        source_reused = bool(
            source_matches_record
            and _path_present(source_path)
            and _record_matches_file(existing_source, source_path)
        )
        if _path_present(source_path) and not source_reused and not refresh:
            raise ArchiveIntegrityError(
                "source script changed or is corrupt; use --refresh to replace it"
            )
        if source_reused:
            source_temporary.unlink(missing_ok=True)
            source_retrieved_at = existing_source_timestamp
        else:
            _publish_temporary(source_temporary, source_path)
            source_retrieved_at = _timestamp(retrieved_at)
    finally:
        source_temporary.unlink(missing_ok=True)

    source_record = _source_record(
        retrieved_at=source_retrieved_at,
        byte_size=fetched_source_size,
        sha256=fetched_source_sha256,
    )

    reusable_records: dict[str, dict[str, Any]] = {}
    entries_by_name = {
        entry.local_relative_filename: entry for entry in entries
    }
    if existing_manifest is not None and not refresh:
        unexpected_records = set(existing_records) - set(entries_by_name)
        if (
            existing_manifest["expectedFileCount"] != len(entries)
            or unexpected_records
        ):
            raise ArchiveIntegrityError(
                "archive manifest does not match the preserved source script; "
                "use --refresh to replace state"
            )
        for relative_name, existing_record in existing_records.items():
            entry = entries_by_name[relative_name]
            if (
                existing_record.get("canonicalSourceURL")
                != entry.canonical_source_url
                or existing_record.get("sourceScriptSHA256")
                != fetched_source_sha256
            ):
                raise ArchiveIntegrityError(
                    f"file provenance changed for {relative_name}; "
                    "use --refresh to replace state"
                )
    for relative_name, entry in entries_by_name.items():
        destination = _safe_state_path(root, relative_name)
        existing_record = existing_records.get(relative_name)
        same_url = bool(
            isinstance(existing_record, Mapping)
            and existing_record.get("canonicalSourceURL")
            == entry.canonical_source_url
        )
        reusable = bool(
            same_url
            and _path_present(destination)
            and _record_matches_file(existing_record, destination)
        )
        if _path_present(destination) and not reusable and not refresh:
            raise ArchiveIntegrityError(
                f"{relative_name} changed, is corrupt, or lacks trusted "
                "manifest provenance; use --refresh to replace it"
            )
        if reusable:
            reusable_records[relative_name] = _archive_record(
                entry,
                retrieved_at=_recorded_timestamp(
                    existing_record,
                    f"file record for {relative_name}",
                ),
                byte_size=existing_record["byteSize"],
                sha256=existing_record["sha256"],
                source_script_sha256=fetched_source_sha256,
            )

    completed_records = dict(reusable_records)
    _atomic_write_json(
        manifest_path,
        _manifest(
            source_record,
            completed_records.values(),
            expected_file_count=len(entries),
        ),
    )

    for entry in entries:
        relative_name = entry.local_relative_filename
        if relative_name in completed_records:
            continue
        destination = _safe_state_path(root, relative_name)
        temporary = _fetch_temporary(
            fetcher,
            entry.canonical_source_url,
            destination,
        )
        try:
            byte_size, digest = _file_identity(temporary)
            if _path_present(destination) and not refresh:
                raise ArchiveIntegrityError(
                    f"{relative_name} appeared during acquisition; "
                    "use --refresh to replace it"
                )
            _publish_temporary(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        completed_records[relative_name] = _archive_record(
            entry,
            retrieved_at=_timestamp(retrieved_at),
            byte_size=byte_size,
            sha256=digest,
            source_script_sha256=fetched_source_sha256,
        )
        _atomic_write_json(
            manifest_path,
            _manifest(
                source_record,
                completed_records.values(),
                expected_file_count=len(entries),
            ),
        )

    result = _manifest(
        source_record,
        completed_records.values(),
        expected_file_count=len(entries),
    )
    if write_inventory:
        inventory_archive(root)
    return result


def extract_uid(filename: str) -> str | None:
    match = _UID_FILENAME.fullmatch(Path(filename).name)
    return match.group(1) if match is not None else None


def _parse_ipac_header_row(line: str, line_number: int) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise TableStructureError(
            f"line {line_number}: malformed IPAC header row"
        )
    cells = tuple(cell.strip() for cell in stripped[1:-1].split("|"))
    if not cells or any(not cell for cell in cells):
        raise TableStructureError(
            f"line {line_number}: empty IPAC header field"
        )
    return cells


def inspect_ipac_table(path: str | Path) -> TableStructure:
    """Read table structure without assigning meaning to any field."""

    table_path = Path(path)
    try:
        lines = table_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        raise TableStructureError(f"table is unreadable: {error}") from error

    metadata_lines = []
    header_rows = []
    row_count = 0
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
                raise TableStructureError(
                    f"line {line_number}: header row appears after data"
                )
            header_rows.append(_parse_ipac_header_row(line, line_number))
            continue
        data_started = True
        row_count += 1

    if not header_rows:
        raise TableStructureError("missing IPAC header")
    columns = header_rows[0]
    if len(set(columns)) != len(columns):
        raise TableStructureError("duplicate column names")
    for index, header_row in enumerate(header_rows[1:], 2):
        if len(header_row) != len(columns):
            raise TableStructureError(
                f"header row {index} has {len(header_row)} fields; "
                f"expected {len(columns)}"
            )

    signature_value = {
        "columns": list(columns),
        "headerRows": [list(row) for row in header_rows[1:]],
    }
    signature = hashlib.sha256(
        json.dumps(
            signature_value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return TableStructure(
        columns=columns,
        header_rows=tuple(header_rows),
        metadata_lines=tuple(metadata_lines),
        row_count=row_count,
        schema_signature=signature,
    )


def _concise_reason(error: Exception) -> str:
    text = " ".join(str(error).split()) or error.__class__.__name__
    return text[:200]


def _validate_inventory_provenance(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    source_path = _safe_state_path(root, SOURCE_SCRIPT_RELATIVE_PATH)
    if source_path.is_symlink() or not source_path.is_file():
        raise ArchiveIntegrityError(
            "preserved source script is not a regular non-symlink file"
        )
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise ArchiveIntegrityError(
            "preserved source script is unreadable"
        ) from error

    source_record = manifest["sourceScript"]
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if (
        len(source_bytes) != source_record["byteSize"]
        or source_sha256 != source_record["sha256"]
    ):
        raise ArchiveIntegrityError(
            "preserved source script size or SHA-256 does not match manifest"
        )
    try:
        entries = parse_wget_script(source_bytes)
    except WgetScriptError as error:
        raise ArchiveIntegrityError(
            "preserved source script cannot be parsed safely"
        ) from error

    if manifest["expectedFileCount"] != len(entries):
        raise ArchiveIntegrityError(
            "expectedFileCount does not match preserved source script"
        )
    entries_by_name = {
        entry.local_relative_filename: entry for entry in entries
    }
    records_by_name = _existing_file_records(manifest, refresh=False)
    for relative_name in sorted(records_by_name):
        record = records_by_name[relative_name]
        canonical_url = record.get("canonicalSourceURL")
        try:
            validated_url = _canonical_nasa_url(
                canonical_url,
                normalize_http=False,
            )
        except WgetScriptError as error:
            raise ArchiveIntegrityError(
                f"invalid NASA HTTPS provenance for {relative_name}"
            ) from error
        if validated_url != canonical_url:
            raise ArchiveIntegrityError(
                f"noncanonical source URL for {relative_name}"
            )

        entry = entries_by_name.get(relative_name)
        if entry is None:
            raise ArchiveIntegrityError(
                f"unexpected manifest file record: {relative_name}"
            )
        if canonical_url != entry.canonical_source_url:
            raise ArchiveIntegrityError(
                f"source URL does not match preserved script for {relative_name}"
            )
        if record.get("sourceScriptURL") != SOURCE_SCRIPT_URL:
            raise ArchiveIntegrityError(
                f"source script URL does not match for {relative_name}"
            )
        if record.get("sourceScriptSHA256") != source_record["sha256"]:
            raise ArchiveIntegrityError(
                f"source script SHA-256 does not match for {relative_name}"
            )

    record_names = set(records_by_name)
    entry_names = set(entries_by_name)
    if not record_names.issubset(entry_names):
        raise ArchiveIntegrityError(
            "manifest contains records outside the preserved source script"
        )
    if manifest["archiveComplete"]:
        if record_names != entry_names:
            raise ArchiveIntegrityError(
                "complete manifest does not contain the exact source entry set"
            )
    elif record_names == entry_names:
        raise ArchiveIntegrityError(
            "partial manifest incorrectly contains the complete source entry set"
        )
    return records_by_name


def inventory_archive(output_root: str | Path) -> dict[str, Any]:
    """Write a deterministic structural inventory of manifest-listed tables."""

    root = Path(output_root).expanduser().resolve()
    manifest_path = _safe_state_path(root, MANIFEST_RELATIVE_PATH)
    manifest = _load_existing_manifest(manifest_path, refresh=False)
    if manifest is None:
        raise ArchiveIntegrityError("archive manifest does not exist")
    records_by_name = _validate_inventory_provenance(root, manifest)

    files = []
    failures = []
    grouped_by_uid: dict[str, list[str]] = {}
    schema_counts: dict[str, int] = {}
    records = [records_by_name[name] for name in sorted(records_by_name)]
    for record in records:
        relative_name = (
            record.get("localRelativeFilename")
            if isinstance(record, Mapping)
            else None
        )
        if not isinstance(relative_name, str):
            failures.append(
                {"filename": "<manifest-entry>", "reason": "invalid file record"}
            )
            continue
        uid = extract_uid(relative_name)
        if uid is not None:
            grouped_by_uid.setdefault(uid, []).append(relative_name)
        try:
            path = _safe_state_path(root, relative_name)
            if not _record_matches_file(record, path):
                raise ArchiveIntegrityError(
                    "size or SHA-256 does not match archive manifest"
                )
            if uid is None:
                raise TableStructureError("filename does not contain a UID")
            structure = inspect_ipac_table(path)
            item = {
                "columns": list(structure.columns),
                "filename": relative_name,
                "headerRows": [list(row) for row in structure.header_rows],
                "metadataLines": list(structure.metadata_lines),
                "rowCount": structure.row_count,
                "schemaSignature": structure.schema_signature,
                "uid": uid,
            }
            files.append(item)
            schema_counts[structure.schema_signature] = (
                schema_counts.get(structure.schema_signature, 0) + 1
            )
        except (ArchiveIntegrityError, OSError, TableStructureError) as error:
            failures.append(
                {"filename": relative_name, "reason": _concise_reason(error)}
            )

    inventory = {
        "manifestComplete": bool(
            manifest.get("archiveComplete") is True
            and manifest.get("expectedFileCount") == len(records)
        ),
        "countsBySchemaSignature": {
            key: schema_counts[key] for key in sorted(schema_counts)
        },
        "expectedFileCount": manifest.get("expectedFileCount"),
        "files": sorted(files, key=lambda item: item["filename"]),
        "filesGroupedByUID": {
            uid: sorted(grouped_by_uid[uid]) for uid in sorted(grouped_by_uid)
        },
        "inventorySchemaID": INVENTORY_SCHEMA_ID,
        "manifestSHA256": sha256_file(manifest_path),
        "parseFailureCount": len(failures),
        "parseFailures": sorted(
            failures,
            key=lambda item: (item["filename"], item["reason"]),
        ),
        "parsedFileCount": len(files),
        "totalFiles": len(records),
    }
    inventory_path = _safe_state_path(root, INVENTORY_RELATIVE_PATH)
    _atomic_write_json(inventory_path, inventory)
    return inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire or structurally inventory the NASA Exoplanet Archive "
            "contributed MICROLENSING bundle."
        )
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="External state directory for source, data, and JSON records.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Permit atomic replacement of changed, corrupt, or untracked files.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--inventory",
        action="store_true",
        help="Write archive-inventory.json after acquisition.",
    )
    mode.add_argument(
        "--inventory-only",
        action="store_true",
        help="Inventory existing manifest-listed files without network access.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(arguments.output_root)
    if arguments.inventory_only:
        inventory_archive(root)
        print(root.expanduser().resolve() / INVENTORY_RELATIVE_PATH)
        return 0

    acquire_archive(
        root,
        refresh=arguments.refresh,
        write_inventory=arguments.inventory,
    )
    print(root.expanduser().resolve() / MANIFEST_RELATIVE_PATH)
    if arguments.inventory:
        print(root.expanduser().resolve() / INVENTORY_RELATIVE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
