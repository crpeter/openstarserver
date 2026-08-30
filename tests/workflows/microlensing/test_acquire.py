import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from workflows.microlensing.acquire import (
    ArchiveIntegrityError,
    INVENTORY_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    SOURCE_SCRIPT_RELATIVE_PATH,
    SOURCE_SCRIPT_URL,
    TableStructureError,
    WgetScriptError,
    acquire_archive,
    extract_uid,
    inspect_ipac_table,
    inventory_archive,
    parse_wget_script,
)


FIXTURES = Path(__file__).parent / "fixtures"
FIXED_TIME = datetime(2026, 8, 30, 16, 0, tzinfo=timezone.utc)
REAL_FIVE_COLUMN_TABLE = """\
| JD | RELATIVE_MAGNITUDE | MAGNITUDE_UNCERTAINTY | RELATIVE_FLUX | FLUX_UNCERTAINTY |
| real | real | real | real | real |
| days | mag | mag | | |
1.0 2.0 0.1 3.0 0.2
"""


class FakeFetcher:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    def __call__(self, url, destination):
        self.calls.append(url)
        destination.write_bytes(self.payloads[url])


def fixture_bytes(name):
    return (FIXTURES / name).read_bytes()


def fixture_payloads():
    script = fixture_bytes("wget_mini.bat")
    entries = parse_wget_script(script)
    by_name = {
        "UID_0300030_PLC_001.tbl": fixture_bytes(
            "UID_0300030_PLC_001.tbl"
        ),
        "UID_0300030_PLC_003.tbl": fixture_bytes(
            "UID_0300030_PLC_003.tbl"
        ),
        "UID_0300031_PLC_002.tbl": fixture_bytes(
            "UID_0300031_PLC_002.tbl"
        ),
    }
    return {
        SOURCE_SCRIPT_URL: script,
        **{
            entry.canonical_source_url: by_name[
                Path(entry.local_relative_filename).name
            ]
            for entry in entries
        },
    }


def read_manifest(root):
    return json.loads(
        (root / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def write_manifest(root, manifest):
    (root / MANIFEST_RELATIVE_PATH).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def inspect_table_text(text):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "table.tbl"
        path.write_text(text, encoding="utf-8")
        return inspect_ipac_table(path)


class WgetScriptParsingTests(unittest.TestCase):
    def test_parsing_is_deterministic_sorted_and_normalizes_https(self):
        script = fixture_bytes("wget_mini.bat")
        first = parse_wget_script(script)
        second = parse_wget_script(script)

        self.assertEqual(first, second)
        self.assertEqual(
            [
                "data/UID_0300030_PLC_001.tbl",
                "data/UID_0300030_PLC_003.tbl",
                "data/UID_0300031_PLC_002.tbl",
            ],
            [entry.local_relative_filename for entry in first],
        )
        self.assertTrue(
            all(
                entry.canonical_source_url.startswith(
                    "https://exoplanetarchive.ipac.caltech.edu/"
                )
                for entry in first
            )
        )
        self.assertTrue(
            all(":80/" not in entry.canonical_source_url for entry in first)
        )

    def test_foreign_hosts_unsafe_names_and_malformed_entries_are_rejected(self):
        cases = (
            "wget -O safe.tbl https://example.com/data/safe.tbl",
            (
                "wget -O ../unsafe.tbl "
                "https://exoplanetarchive.ipac.caltech.edu/data/unsafe.tbl"
            ),
            (
                "wget -O safe.tbl ftp://exoplanetarchive.ipac.caltech.edu/"
                "data/safe.tbl"
            ),
            (
                "wget -O safe.tbl https://exoplanetarchive.ipac.caltech.edu/"
                "data/%252e%252e/safe.tbl"
            ),
            (
                "wget -O safe.tbl https://exoplanetarchive.ipac.caltech.edu/"
                "data/%ZZ/safe.tbl"
            ),
            (
                "wget -O safe.tbl https://exoplanetarchive.ipac.caltech.edu/"
                "data/%00/safe.tbl"
            ),
            (
                "wget https://exoplanetarchive.ipac.caltech.edu/data/safe.tbl"
            ),
            "echo not-a-wget-entry",
        )
        for script in cases:
            with self.subTest(script=script), self.assertRaises(WgetScriptError):
                parse_wget_script(script)

    def test_duplicate_equal_entries_deduplicate_and_conflicts_fail(self):
        url_one = (
            "http://exoplanetarchive.ipac.caltech.edu:80/data/one/same.tbl"
        )
        url_two = (
            "http://exoplanetarchive.ipac.caltech.edu:80/data/two/same.tbl"
        )
        equal = f"wget -O same.tbl {url_one}\nwget -O same.tbl {url_one}\n"
        self.assertEqual(1, len(parse_wget_script(equal)))

        conflicting = (
            f"wget -O same.tbl {url_one}\nwget -O same.tbl {url_two}\n"
        )
        with self.assertRaisesRegex(WgetScriptError, "conflicting URLs"):
            parse_wget_script(conflicting)


class IpacTableStructureTests(unittest.TestCase):
    def test_blank_optional_unit_cells_are_accepted_and_preserved(self):
        structure = inspect_table_text(
            "| coordinate | value | uncertainty |\n"
            "| real | real | real |\n"
            "| days | | |\n"
            "1.0 2.0 0.1\n"
        )

        self.assertEqual(
            ("days", "", ""),
            structure.header_rows[2],
        )

    def test_real_five_column_structural_shape_is_preserved(self):
        structure = inspect_table_text(REAL_FIVE_COLUMN_TABLE)

        self.assertEqual(
            (
                "JD",
                "RELATIVE_MAGNITUDE",
                "MAGNITUDE_UNCERTAINTY",
                "RELATIVE_FLUX",
                "FLUX_UNCERTAINTY",
            ),
            structure.columns,
        )
        self.assertEqual(
            ("days", "mag", "mag", "", ""),
            structure.header_rows[2],
        )
        self.assertEqual(1, structure.row_count)

    def test_empty_column_name_remains_rejected(self):
        with self.assertRaisesRegex(TableStructureError, "empty column name"):
            inspect_table_text(
                "| coordinate | | uncertainty |\n"
                "| real | real | real |\n"
            )

    def test_duplicate_column_names_remain_rejected(self):
        with self.assertRaisesRegex(TableStructureError, "duplicate column"):
            inspect_table_text(
                "| coordinate | value | value |\n"
                "| real | real | real |\n"
            )

    def test_subsequent_header_row_with_wrong_width_remains_rejected(self):
        with self.assertRaisesRegex(TableStructureError, "expected 3"):
            inspect_table_text(
                "| coordinate | value | uncertainty |\n"
                "| real | real |\n"
            )

    def test_malformed_header_delimiters_remain_rejected(self):
        with self.assertRaisesRegex(TableStructureError, "malformed"):
            inspect_table_text(
                "| coordinate | value\n"
                "1.0 2.0\n"
            )

    def test_schema_signature_deterministically_includes_empty_cells(self):
        first = inspect_table_text(REAL_FIVE_COLUMN_TABLE)
        second = inspect_table_text(REAL_FIVE_COLUMN_TABLE)
        filled = inspect_table_text(
            REAL_FIVE_COLUMN_TABLE.replace(
                "| days | mag | mag | | |",
                "| days | mag | mag | flux | flux |",
            )
        )
        signature_value = {
            "columns": list(first.columns),
            "headerRows": [
                list(row) for row in first.header_rows[1:]
            ],
        }
        expected = hashlib.sha256(
            json.dumps(
                signature_value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        self.assertEqual(first.schema_signature, second.schema_signature)
        self.assertEqual(expected, first.schema_signature)
        self.assertNotEqual(first.schema_signature, filled.schema_signature)
        self.assertEqual(
            ["days", "mag", "mag", "", ""],
            signature_value["headerRows"][1],
        )


class ArchiveAcquisitionTests(unittest.TestCase):
    def test_manifest_records_hashes_sizes_provenance_and_preserved_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = fixture_payloads()
            manifest = acquire_archive(
                root,
                fetcher=FakeFetcher(payloads),
                retrieved_at=FIXED_TIME,
            )

            self.assertTrue(manifest["archiveComplete"])
            self.assertEqual(3, manifest["expectedFileCount"])
            self.assertEqual(3, len(manifest["files"]))
            self.assertEqual(
                fixture_bytes("wget_mini.bat"),
                (root / SOURCE_SCRIPT_RELATIVE_PATH).read_bytes(),
            )
            script_sha = hashlib.sha256(
                fixture_bytes("wget_mini.bat")
            ).hexdigest()
            source_record = manifest["sourceScript"]
            self.assertEqual(SOURCE_SCRIPT_URL, source_record["canonicalSourceURL"])
            self.assertEqual(
                SOURCE_SCRIPT_RELATIVE_PATH,
                source_record["localRelativeFilename"],
            )
            self.assertEqual(
                len(fixture_bytes("wget_mini.bat")),
                source_record["byteSize"],
            )
            self.assertEqual("2026-08-30T16:00:00Z", source_record["retrievedAtUTC"])
            self.assertEqual(script_sha, source_record["sha256"])
            for record in manifest["files"]:
                path = root / record["localRelativeFilename"]
                content = path.read_bytes()
                self.assertEqual(len(content), record["byteSize"])
                self.assertEqual(
                    hashlib.sha256(content).hexdigest(),
                    record["sha256"],
                )
                self.assertEqual(SOURCE_SCRIPT_URL, record["sourceScriptURL"])
                self.assertEqual(script_sha, record["sourceScriptSHA256"])
                self.assertEqual("2026-08-30T16:00:00Z", record["retrievedAtUTC"])

            on_disk = json.loads(
                (root / MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest, on_disk)
            self.assertEqual(
                sorted(
                    record["localRelativeFilename"]
                    for record in manifest["files"]
                ),
                [
                    record["localRelativeFilename"]
                    for record in manifest["files"]
                ],
            )

    def test_safe_rerun_reuses_data_and_keeps_manifest_bytes_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = fixture_payloads()
            first_fetcher = FakeFetcher(payloads)
            acquire_archive(
                root,
                fetcher=first_fetcher,
                retrieved_at=FIXED_TIME,
            )
            before = (root / MANIFEST_RELATIVE_PATH).read_bytes()

            second_fetcher = FakeFetcher(payloads)
            acquire_archive(
                root,
                fetcher=second_fetcher,
                retrieved_at=datetime(
                    2026,
                    8,
                    31,
                    16,
                    0,
                    tzinfo=timezone.utc,
                ),
            )
            after = (root / MANIFEST_RELATIVE_PATH).read_bytes()

            self.assertEqual([SOURCE_SCRIPT_URL], second_fetcher.calls)
            self.assertEqual(before, after)
            self.assertFalse(any(root.rglob("*.download")))
            self.assertFalse(any(root.rglob("*.tmp")))

    def test_corruption_requires_refresh_before_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = fixture_payloads()
            manifest = acquire_archive(
                root,
                fetcher=FakeFetcher(payloads),
                retrieved_at=FIXED_TIME,
            )
            record = manifest["files"][0]
            corrupt_path = root / record["localRelativeFilename"]
            corrupt_path.write_bytes(b"corrupt")

            refusing_fetcher = FakeFetcher(payloads)
            with self.assertRaisesRegex(ArchiveIntegrityError, "--refresh"):
                acquire_archive(
                    root,
                    fetcher=refusing_fetcher,
                    retrieved_at=FIXED_TIME,
                )
            self.assertEqual(b"corrupt", corrupt_path.read_bytes())

            refreshing_fetcher = FakeFetcher(payloads)
            refreshed = acquire_archive(
                root,
                fetcher=refreshing_fetcher,
                refresh=True,
                retrieved_at=FIXED_TIME,
            )
            refreshed_record = next(
                item
                for item in refreshed["files"]
                if item["localRelativeFilename"]
                == record["localRelativeFilename"]
            )
            self.assertEqual(
                refreshed_record["sha256"],
                hashlib.sha256(corrupt_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                payloads[record["canonicalSourceURL"]],
                corrupt_path.read_bytes(),
            )
            self.assertEqual(
                [SOURCE_SCRIPT_URL, record["canonicalSourceURL"]],
                refreshing_fetcher.calls,
            )

    def test_broken_symlink_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            unsafe_path = root / "data" / "UID_0300030_PLC_001.tbl"
            unsafe_path.symlink_to(root / "missing-target.tbl")

            with self.assertRaisesRegex(ArchiveIntegrityError, "--refresh"):
                acquire_archive(
                    root,
                    fetcher=FakeFetcher(fixture_payloads()),
                    retrieved_at=FIXED_TIME,
                )
            self.assertTrue(unsafe_path.is_symlink())

    def test_conflicting_url_for_existing_name_requires_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = fixture_payloads()
            acquire_archive(
                root,
                fetcher=FakeFetcher(payloads),
                retrieved_at=FIXED_TIME,
            )

            changed_payloads = copy.deepcopy(payloads)
            changed_script = fixture_bytes("wget_mini.bat").replace(
                b"/data/mini/UID_0300030_PLC_001.tbl",
                b"/data/revised/UID_0300030_PLC_001.tbl",
            )
            changed_payloads[SOURCE_SCRIPT_URL] = changed_script
            with self.assertRaisesRegex(ArchiveIntegrityError, "source script"):
                acquire_archive(
                    root,
                    fetcher=FakeFetcher(changed_payloads),
                    retrieved_at=FIXED_TIME,
                )


class ArchiveInventoryTests(unittest.TestCase):
    def acquire_fixture_archive(self, root):
        acquire_archive(
            root,
            fetcher=FakeFetcher(fixture_payloads()),
            retrieved_at=FIXED_TIME,
        )

    def test_uid_extraction_is_exact(self):
        self.assertEqual("0300030", extract_uid("UID_0300030_PLC_001.tbl"))
        self.assertEqual(
            "0300030",
            extract_uid("data/UID_0300030_PLC_999.tbl"),
        )
        self.assertIsNone(extract_uid("UID_0300030_OTHER_001.tbl"))

    def test_inventory_preserves_multiple_schemas_and_reports_malformed_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquire_archive(
                root,
                fetcher=FakeFetcher(fixture_payloads()),
                retrieved_at=FIXED_TIME,
            )
            inventory = inventory_archive(root)

            self.assertTrue(inventory["manifestComplete"])
            self.assertEqual(3, inventory["expectedFileCount"])
            self.assertEqual(3, inventory["totalFiles"])
            self.assertEqual(2, inventory["parsedFileCount"])
            self.assertEqual(1, inventory["parseFailureCount"])
            self.assertEqual(2, len(inventory["countsBySchemaSignature"]))
            self.assertEqual(
                {1},
                set(inventory["countsBySchemaSignature"].values()),
            )
            self.assertEqual(
                ["time", "value"],
                next(
                    item["columns"]
                    for item in inventory["files"]
                    if item["uid"] == "0300030"
                ),
            )
            self.assertEqual(
                ["epoch", "flux", "uncertainty"],
                next(
                    item["columns"]
                    for item in inventory["files"]
                    if item["uid"] == "0300031"
                ),
            )
            self.assertEqual(
                ["day", "arbitrary", ""],
                next(
                    item["headerRows"][2]
                    for item in inventory["files"]
                    if item["uid"] == "0300031"
                ),
            )
            self.assertEqual(
                {"0300030": 2, "0300031": 1},
                {
                    item["uid"]: item["rowCount"]
                    for item in inventory["files"]
                },
            )
            self.assertEqual(
                [
                    "data/UID_0300030_PLC_001.tbl",
                    "data/UID_0300030_PLC_003.tbl",
                ],
                inventory["filesGroupedByUID"]["0300030"],
            )
            self.assertEqual(
                ["data/UID_0300031_PLC_002.tbl"],
                inventory["filesGroupedByUID"]["0300031"],
            )
            self.assertEqual(
                "data/UID_0300030_PLC_003.tbl",
                inventory["parseFailures"][0]["filename"],
            )
            self.assertIn(
                "missing IPAC header",
                inventory["parseFailures"][0]["reason"],
            )
            self.assertEqual(
                ["\\catalog = 'mini-primary'"],
                next(
                    item["metadataLines"]
                    for item in inventory["files"]
                    if item["uid"] == "0300030"
                ),
            )

    def test_inventory_json_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acquire_archive(
                root,
                fetcher=FakeFetcher(fixture_payloads()),
                retrieved_at=FIXED_TIME,
            )
            first = inventory_archive(root)
            first_bytes = (root / INVENTORY_RELATIVE_PATH).read_bytes()
            second = inventory_archive(root)
            second_bytes = (root / INVENTORY_RELATIVE_PATH).read_bytes()

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertTrue(first_bytes.endswith(b"\n"))

    def test_inventory_only_rejects_missing_preserved_source_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.acquire_fixture_archive(root)
            (root / SOURCE_SCRIPT_RELATIVE_PATH).unlink()

            with patch(
                "workflows.microlensing.acquire.inspect_ipac_table"
            ) as inspect_table:
                with self.assertRaisesRegex(
                    ArchiveIntegrityError,
                    "preserved source script",
                ):
                    inventory_archive(root)
                inspect_table.assert_not_called()

    def test_inventory_only_rejects_corrupt_preserved_source_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.acquire_fixture_archive(root)
            source_path = root / SOURCE_SCRIPT_RELATIVE_PATH
            source_bytes = source_path.read_bytes()
            source_path.write_bytes(b"X" + source_bytes[1:])

            with self.assertRaisesRegex(
                ArchiveIntegrityError,
                "size or SHA-256",
            ):
                inventory_archive(root)

    def test_inventory_only_rejects_tampered_file_source_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.acquire_fixture_archive(root)
            manifest = read_manifest(root)
            manifest["files"][0]["canonicalSourceURL"] = (
                "https://exoplanetarchive.ipac.caltech.edu/"
                "data/tampered/UID_0300030_PLC_001.tbl"
            )
            write_manifest(root, manifest)

            with self.assertRaisesRegex(
                ArchiveIntegrityError,
                "does not match preserved script",
            ):
                inventory_archive(root)

    def test_inventory_only_rejects_mismatched_source_script_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.acquire_fixture_archive(root)
            manifest = read_manifest(root)
            manifest["files"][0]["sourceScriptSHA256"] = "0" * 64
            write_manifest(root, manifest)

            with self.assertRaisesRegex(
                ArchiveIntegrityError,
                "source script SHA-256",
            ):
                inventory_archive(root)

    def test_inventory_only_rejects_unexpected_manifest_file_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.acquire_fixture_archive(root)
            manifest = read_manifest(root)
            unexpected = copy.deepcopy(manifest["files"][0])
            unexpected["localRelativeFilename"] = (
                "data/UID_9999999_PLC_999.tbl"
            )
            unexpected["canonicalSourceURL"] = (
                "https://exoplanetarchive.ipac.caltech.edu/"
                "data/mini/UID_9999999_PLC_999.tbl"
            )
            manifest["files"][0] = unexpected
            write_manifest(root, manifest)

            with self.assertRaisesRegex(
                ArchiveIntegrityError,
                "unexpected manifest file record",
            ):
                inventory_archive(root)

    def test_inventory_only_accepts_valid_partial_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.acquire_fixture_archive(root)
            manifest = read_manifest(root)
            omitted_name = "data/UID_0300030_PLC_003.tbl"
            manifest["files"] = [
                record
                for record in manifest["files"]
                if record["localRelativeFilename"] != omitted_name
            ]
            manifest["archiveComplete"] = False
            write_manifest(root, manifest)
            (root / omitted_name).unlink()

            inventory = inventory_archive(root)

            self.assertFalse(inventory["manifestComplete"])
            self.assertEqual(3, inventory["expectedFileCount"])
            self.assertEqual(2, inventory["totalFiles"])
            self.assertEqual(2, inventory["parsedFileCount"])
            self.assertEqual(0, inventory["parseFailureCount"])


if __name__ == "__main__":
    unittest.main()
