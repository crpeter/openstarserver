import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from workflows.microlensing.acquire import (
    INVENTORY_RELATIVE_PATH,
    INVENTORY_SCHEMA_ID,
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_ID,
    SOURCE_SCRIPT_RELATIVE_PATH,
    SOURCE_SCRIPT_URL,
    extract_uid,
    inspect_ipac_table,
)
from workflows.microlensing.prepare import (
    PREPARATION_CONTRACT_ID,
    PREPARATION_CONTRACT_SHA256,
    SERIES_SCHEMA_ID,
    PreparationError,
    parse_fixed_width_ipac,
    prepare_archive,
    select_and_normalize_observable,
)


UID = "0302608"
STAR_ID = "OGLE 2012-BLG-724L"
BLIND_TARGET_ID = "openstar.microlensing-recovery-a.v1"
TIMESTAMP = "2026-08-30T16:00:00Z"
COLUMNS = (
    "HJD",
    "RELATIVE_MAGNITUDE",
    "MAGNITUDE_UNCERTAINTY",
    "RELATIVE_FLUX",
    "FLUX_UNCERTAINTY",
)
WIDTHS = (16, 24, 25, 20, 20)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def header_row(values, widths=WIDTHS):
    return "|" + "|".join(
        f"{value:<{width}}" for value, width in zip(values, widths)
    ) + "|"


def data_row(values, widths=WIDTHS):
    return " " + " ".join(
        f"{str(value):>{width}}" if value != "" else " " * width
        for value, width in zip(values, widths)
    )


def table_text(
    rows,
    *,
    star_id=STAR_ID,
    columns=COLUMNS,
    widths=WIDTHS,
    extra_metadata=(),
):
    metadata = [
        f'\\STAR_ID = "{star_id}"',
        '\\REFERENCE = "Distinctive Source Reference"',
        '\\BIBCODE = "2012FixtureBibcode"',
        '\\RA = "17:57:42.1"',
        '\\DEC = "-29:12:34"',
        '\\OBSERVATORY = "Fixture Observatory"',
        '\\TELESCOPE = "Fixture Telescope"',
        '\\INSTRUMENT = "Fixture Instrument"',
        '\\FILTER = "Fixture Filter"',
        *extra_metadata,
    ]
    structural = [
        header_row(columns, widths),
        header_row(("real",) * len(columns), widths),
        header_row(("days", "mag", "mag", "", ""), widths),
    ]
    return "\n".join(
        [*metadata, *structural, *(data_row(row, widths) for row in rows), ""]
    )


def write_table(path, text):
    path.write_text(text, encoding="utf-8")


def parse_table(text):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "table.tbl"
    write_table(path, text)
    table = parse_fixed_width_ipac(path)
    return temporary, table


def build_archive(root, table_payloads):
    source_path = root / SOURCE_SCRIPT_RELATIVE_PATH
    source_path.parent.mkdir(parents=True)
    data_root = root / "data"
    data_root.mkdir()

    script_lines = []
    records = []
    for filename in sorted(table_payloads):
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/data/"
            f"MICROLENSING/{filename}"
        )
        script_lines.append(f"wget -O {filename} {url}")
    script_bytes = ("\n".join(script_lines) + "\n").encode("utf-8")
    source_path.write_bytes(script_bytes)
    source_sha256 = sha256_bytes(script_bytes)

    for filename in sorted(table_payloads):
        payload = table_payloads[filename].encode("utf-8")
        relative_filename = f"data/{filename}"
        (root / relative_filename).write_bytes(payload)
        records.append(
            {
                "byteSize": len(payload),
                "canonicalSourceURL": (
                    "https://exoplanetarchive.ipac.caltech.edu/data/"
                    f"MICROLENSING/{filename}"
                ),
                "localRelativeFilename": relative_filename,
                "retrievedAtUTC": TIMESTAMP,
                "sha256": sha256_bytes(payload),
                "sourceScriptSHA256": source_sha256,
                "sourceScriptURL": SOURCE_SCRIPT_URL,
            }
        )

    manifest = {
        "archiveComplete": True,
        "expectedFileCount": len(records),
        "files": records,
        "manifestSchemaID": MANIFEST_SCHEMA_ID,
        "sourceScript": {
            "byteSize": len(script_bytes),
            "canonicalSourceURL": SOURCE_SCRIPT_URL,
            "localRelativeFilename": SOURCE_SCRIPT_RELATIVE_PATH,
            "retrievedAtUTC": TIMESTAMP,
            "sha256": source_sha256,
        },
    }
    manifest_bytes = stable_json_bytes(manifest)
    (root / MANIFEST_RELATIVE_PATH).write_bytes(manifest_bytes)

    files = []
    grouped = {}
    signature_counts = {}
    for record in records:
        relative_filename = record["localRelativeFilename"]
        structure = inspect_ipac_table(root / relative_filename)
        uid = extract_uid(relative_filename)
        files.append(
            {
                "columns": list(structure.columns),
                "filename": relative_filename,
                "headerRows": [list(row) for row in structure.header_rows],
                "metadataLines": list(structure.metadata_lines),
                "rowCount": structure.row_count,
                "schemaSignature": structure.schema_signature,
                "uid": uid,
            }
        )
        grouped.setdefault(uid, []).append(relative_filename)
        signature_counts[structure.schema_signature] = (
            signature_counts.get(structure.schema_signature, 0) + 1
        )
    inventory = {
        "countsBySchemaSignature": dict(sorted(signature_counts.items())),
        "expectedFileCount": len(records),
        "files": files,
        "filesGroupedByUID": {
            key: sorted(value) for key, value in sorted(grouped.items())
        },
        "inventorySchemaID": INVENTORY_SCHEMA_ID,
        "manifestComplete": True,
        "manifestSHA256": sha256_bytes(manifest_bytes),
        "parseFailureCount": 0,
        "parseFailures": [],
        "parsedFileCount": len(records),
        "totalFiles": len(records),
    }
    (root / INVENTORY_RELATIVE_PATH).write_bytes(stable_json_bytes(inventory))
    return manifest, inventory


def flux_rows(start=2456000.25):
    return [
        (start + 2.0, "", "", 4.0, 0.4),
        (start, "", "", 2.0, 0.2),
        (start + 1.0, "", "", 3.0, 0.3),
    ]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def serialized_tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
    }


class FixedWidthParsingTests(unittest.TestCase):
    def test_populated_magnitude_fields_retain_positions(self):
        temporary, table = parse_table(
            table_text([(2456000.1, 18.25, 0.03, "", "")])
        )
        self.addCleanup(temporary.cleanup)

        self.assertEqual("18.25", table.rows[0][1])
        self.assertEqual("0.03", table.rows[0][2])
        self.assertEqual("", table.rows[0][3])
        self.assertEqual("", table.rows[0][4])

    def test_blank_magnitude_fields_do_not_shift_populated_flux_fields(self):
        temporary, table = parse_table(
            table_text([(2456000.1, "", "", 1.25, 0.04)])
        )
        self.addCleanup(temporary.cleanup)

        self.assertEqual("", table.rows[0][1])
        self.assertEqual("", table.rows[0][2])
        self.assertEqual("1.25", table.rows[0][3])
        self.assertEqual("0.04", table.rows[0][4])

    def test_internal_blanks_are_preserved_as_cells(self):
        temporary, table = parse_table(
            table_text([(2456000.1, "", 0.02, 1.5, "")])
        )
        self.addCleanup(temporary.cleanup)

        self.assertEqual(
            ("2456000.1", "", "0.02", "1.5", ""),
            table.rows[0],
        )


class ObservablePreparationTests(unittest.TestCase):
    def test_observable_with_more_valid_samples_is_selected(self):
        rows = [
            (4.0, 18.4, 0.1, 1.0, 0.1),
            (1.0, 18.1, 0.1, 2.0, 0.1),
            (3.0, 18.3, 0.1, 3.0, 0.1),
            (2.0, 18.2, 0.1, "", ""),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        self.assertEqual("RELATIVE_MAGNITUDE", selected.observable_kind)
        self.assertEqual(4, selected.selected_rows)
        self.assertEqual(0, selected.dropped_rows)

    def test_exact_valid_sample_tie_prefers_flux(self):
        rows = [
            (1.0, 18.1, 0.1, 1.0, 0.1),
            (2.0, 18.2, 0.1, 2.0, 0.1),
            (3.0, 18.3, 0.1, 3.0, 0.1),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        self.assertEqual("RELATIVE_FLUX", selected.observable_kind)

    def test_invalid_and_nonpositive_uncertainties_are_not_valid_samples(self):
        rows = [
            (1.0, 18.1, 0.1, 1.0, 0.1),
            (2.0, 18.2, 0.1, 2.0, 0.0),
            (3.0, 18.3, 0.1, 3.0, -0.1),
            (4.0, 18.4, 0.1, 4.0, "nan"),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        self.assertEqual("RELATIVE_MAGNITUDE", selected.observable_kind)
        self.assertEqual(4, selected.selected_rows)

    def test_fewer_than_three_valid_selected_samples_are_rejected(self):
        rows = [
            (1.0, "", "", 1.0, 0.1),
            (2.0, "", "", 2.0, 0.0),
            (3.0, "", "", 3.0, -0.1),
            (4.0, "", "", 4.0, 0.4),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(PreparationError, "fewer than 3"):
            select_and_normalize_observable(table)

    def test_stable_sort_retains_repeated_times(self):
        rows = [
            (2.0, "", "", 4.0, 0.4),
            (1.0, "", "", 2.0, 0.2),
            (1.0, "", "", 3.0, 0.3),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        self.assertEqual((1.0, 1.0, 2.0), selected.absolute_coordinates)
        self.assertEqual((2.0 / 3.0, 1.0, 4.0 / 3.0), selected.values)

    def test_flux_uses_median_absolute_nonzero_scale(self):
        rows = [
            (1.0, "", "", -2.0, 0.2),
            (2.0, "", "", 4.0, 0.4),
            (3.0, "", "", 0.0, 0.3),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        self.assertEqual("fluxScale", selected.normalization_kind)
        self.assertEqual(3.0, selected.normalization_value)
        self.assertEqual((-2.0 / 3.0, 4.0 / 3.0, 0.0), selected.values)
        for actual, expected in zip(
            selected.uncertainties,
            (0.2 / 3.0, 0.4 / 3.0, 0.1),
        ):
            self.assertAlmostEqual(expected, actual, places=15)

    def test_zero_flux_falls_back_to_median_positive_uncertainty(self):
        rows = [
            (1.0, "", "", 0.0, 0.2),
            (2.0, "", "", 0.0, 0.4),
            (3.0, "", "", 0.0, 0.6),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        self.assertEqual(0.4, selected.normalization_value)
        self.assertEqual((0.0, 0.0, 0.0), selected.values)
        for actual, expected in zip(
            selected.uncertainties,
            (0.5, 1.0, 1.5),
        ):
            self.assertAlmostEqual(expected, actual, places=15)

    def test_jd_is_an_explicit_supported_time_column(self):
        columns = ("JD", *COLUMNS[1:])
        temporary, table = parse_table(
            table_text(flux_rows(100.25), columns=columns)
        )
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        self.assertEqual((100.25, 101.25, 102.25), selected.absolute_coordinates)

    def test_magnitude_conversion_and_propagated_uncertainty(self):
        rows = [
            (1.0, 19.0, 0.1, "", ""),
            (2.0, 20.0, 0.2, "", ""),
            (3.0, 21.0, 0.3, "", ""),
        ]
        temporary, table = parse_table(table_text(rows))
        self.addCleanup(temporary.cleanup)

        selected = select_and_normalize_observable(table)

        expected_values = (
            10.0 ** 0.4,
            1.0,
            10.0 ** -0.4,
        )
        expected_uncertainties = tuple(
            (math.log(10.0) / 2.5) * value * uncertainty
            for value, uncertainty in zip(expected_values, (0.1, 0.2, 0.3))
        )
        self.assertEqual("referenceMagnitude", selected.normalization_kind)
        self.assertEqual(20.0, selected.normalization_value)
        for actual, expected in zip(selected.values, expected_values):
            self.assertAlmostEqual(expected, actual, places=15)
        for actual, expected in zip(
            selected.uncertainties, expected_uncertainties
        ):
            self.assertAlmostEqual(expected, actual, places=15)


class ArchivePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "archive"
        self.archive.mkdir()

    def build_two_source_archive(self, second_star_id=STAR_ID):
        build_archive(
            self.archive,
            {
                f"UID_{UID}_PLC_002.tbl": table_text(
                    flux_rows(2456002.25), star_id=second_star_id
                ),
                f"UID_{UID}_PLC_001.tbl": table_text(flux_rows(2456000.25)),
            },
        )

    def test_common_origin_generic_ids_seal_and_blind_outputs(self):
        self.build_two_source_archive()
        output = self.root / "prepared"

        result = prepare_archive(
            self.archive,
            uid=UID,
            blind_target_id=BLIND_TARGET_ID,
            output_root=output,
        )

        seal = read_json(output / "sealed" / "identity-seal.json")
        blind = read_json(output / "blind" / "preparation-manifest.json")
        first_series = read_json(output / "blind" / "series" / "series-001.json")
        self.assertEqual(2456000, seal["absoluteCommonTimeOrigin"])
        self.assertEqual(UID, seal["uid"])
        self.assertEqual(STAR_ID, seal["starID"])
        self.assertEqual("known-event-recovery", seal["benchmarkKind"])
        self.assertEqual(PREPARATION_CONTRACT_ID, seal["preparationContractID"])
        self.assertEqual(
            PREPARATION_CONTRACT_SHA256,
            seal["preparationContractSHA256"],
        )
        self.assertEqual(
            PREPARATION_CONTRACT_SHA256,
            blind["preparationContractSHA256"],
        )
        self.assertEqual(["series-001", "series-002"], blind["orderedSeriesIDs"])
        self.assertEqual(6, blind["totalSampleCount"])
        self.assertEqual(SERIES_SCHEMA_ID, first_series["seriesSchemaID"])
        self.assertEqual([0.25, 1.25, 2.25], first_series["coordinates"])
        self.assertEqual(result["identitySeal"], seal)
        self.assertIn("REFERENCE", seal["sources"][0]["originalRelevantMetadata"])
        self.assertIn("BIBCODE", seal["sources"][0]["originalRelevantMetadata"])
        self.assertEqual(
            f"data/UID_{UID}_PLC_001.tbl",
            seal["seriesSourceMapping"][0]["sourceFilename"],
        )
        self.assertTrue(seal["archiveManifestSHA256"])
        self.assertTrue(seal["archiveInventorySHA256"])
        self.assertTrue(
            seal["sources"][0]["canonicalSourceURL"].startswith(
                "https://exoplanetarchive.ipac.caltech.edu/"
            )
        )
        self.assertGreater(seal["sources"][0]["byteSize"], 0)
        self.assertEqual(64, len(seal["sources"][0]["sha256"]))
        self.assertEqual("fluxScale", seal["sources"][0]["normalization"]["kind"])
        self.assertEqual(3.0, seal["sources"][0]["normalization"]["scale"])
        self.assertEqual(
            "RELATIVE_FLUX",
            seal["sources"][0]["selectedObservableKind"],
        )

    def test_outputs_and_recorded_series_hashes_are_deterministic(self):
        self.build_two_source_archive()
        first_output = self.root / "prepared-one"
        second_output = self.root / "prepared-two"

        prepare_archive(
            self.archive,
            uid=UID,
            blind_target_id=BLIND_TARGET_ID,
            output_root=first_output,
        )
        prepare_archive(
            self.archive,
            uid=UID,
            blind_target_id=BLIND_TARGET_ID,
            output_root=second_output,
        )

        self.assertEqual(serialized_tree(first_output), serialized_tree(second_output))
        blind = read_json(first_output / "blind" / "preparation-manifest.json")
        for record in blind["series"]:
            series_path = first_output / "blind" / record["seriesFile"]
            self.assertEqual(sha256_bytes(series_path.read_bytes()), record["sha256"])

    def test_selected_source_integrity_failure_is_rejected(self):
        self.build_two_source_archive()
        selected = self.archive / "data" / f"UID_{UID}_PLC_001.tbl"
        selected.write_bytes(selected.read_bytes() + b"corruption")

        with self.assertRaisesRegex(PreparationError, "size or SHA-256 mismatch"):
            prepare_archive(
                self.archive,
                uid=UID,
                blind_target_id=BLIND_TARGET_ID,
                output_root=self.root / "prepared",
            )

    def test_incomplete_inventory_is_rejected(self):
        self.build_two_source_archive()
        inventory_path = self.archive / INVENTORY_RELATIVE_PATH
        inventory = read_json(inventory_path)
        inventory["manifestComplete"] = False
        inventory_path.write_bytes(stable_json_bytes(inventory))

        with self.assertRaisesRegex(PreparationError, "must be complete"):
            prepare_archive(
                self.archive,
                uid=UID,
                blind_target_id=BLIND_TARGET_ID,
                output_root=self.root / "prepared",
            )

    def test_inconsistent_star_id_is_rejected(self):
        self.build_two_source_archive(second_star_id="DIFFERENT SOURCE ID")

        with self.assertRaisesRegex(PreparationError, "inconsistent across"):
            prepare_archive(
                self.archive,
                uid=UID,
                blind_target_id=BLIND_TARGET_ID,
                output_root=self.root / "prepared",
            )

    def test_existing_output_root_is_rejected_without_changes(self):
        self.build_two_source_archive()
        output = self.root / "prepared"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(PreparationError, "already exists"):
            prepare_archive(
                self.archive,
                uid=UID,
                blind_target_id=BLIND_TARGET_ID,
                output_root=output,
            )
        self.assertEqual("keep", marker.read_text(encoding="utf-8"))

    def test_blind_directory_contains_no_source_identity_tokens(self):
        self.build_two_source_archive()
        output = self.root / "prepared"
        prepare_archive(
            self.archive,
            uid=UID,
            blind_target_id=BLIND_TARGET_ID,
            output_root=output,
        )

        serialized = b"\n".join(
            path.read_bytes() for path in sorted((output / "blind").rglob("*.json"))
        ).decode("utf-8").casefold()
        forbidden = (
            UID,
            "OGLE",
            STAR_ID,
            "OGLE-2012-BLG-0724L",
            "17:57:42.1",
            "-29:12:34",
            "Distinctive Source Reference",
            "2012FixtureBibcode",
            f"UID_{UID}_PLC_001.tbl",
            f"UID_{UID}_PLC_002.tbl",
            "https://exoplanetarchive.ipac.caltech.edu/",
            "2456000.25",
            "Fixture Observatory",
            "Fixture Telescope",
            "Fixture Instrument",
            "Fixture Filter",
            '"ra"',
            '"dec"',
            "reference",
            "bibcode",
            "observatory",
            "telescope",
            "instrument",
            "filter",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token.casefold(), serialized)


if __name__ == "__main__":
    unittest.main()
