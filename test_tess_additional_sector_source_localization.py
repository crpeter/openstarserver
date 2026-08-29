from pathlib import Path
import tempfile
import unittest
from unittest import mock
from workflows.tess.tess_additional_sector_source_localization import (
    interpret_additional_sector_source_localization,
    prepare_additional_sector_source_localization,
    run_additional_sector_source_localization,
    unused_official_sectors,
)


def bridge(drift=0.0):
    return {
        "ticID": 277940827,
        "referenceFamilyPeriodDays": 10.30084080080649,
        "residualReferenceFrequency": 1 / 2.2071724078510457,
        "residualTimeReferenceDays": 2500.0,
        "fractionalFrequencyDriftPerDay": drift,
        "subtractedHarmonicOrders": [1, 2, 3, 4],
        "catalogCandidates": [
            {"id": "a", "raDeg": 1, "decDeg": 2},
            {"id": "b", "raDeg": 3, "decDeg": 4},
        ],
        "targetSky": {"raDeg": 5, "decDeg": 6},
        "sectors": [94, 95, 102, 103],
    }


def authorization():
    return {
        "classification": "UNRESOLVED",
        "recommendedNextTest": "ADDITIONAL_INDEPENDENT_SOURCE_LOCALIZATION_DATA",
        "sourceAttributionResolved": False,
        "physicalMechanismResolved": False,
    }


def identity():
    return {"tess": {"officialSectors": [103, 1, 28, 94, 67, 95, 27, 68, 102]}}


def sector(source=None, multiple=False, quality=True):
    sources = [source] if source else []
    if multiple:
        sources = ["target", "candidate-1"]
    model = "m"
    return {
        "sector": 1,
        "calibrationResolved": quality,
        "scientificallyValid": quality,
        "fullDataComparison": {
            "completeModelFullRank": quality,
            "bestModelIdentifiable": quality,
            "bestModel": model,
            "bestModelSourceIDs": sources,
        },
        "temporalPredictiveValidation": {
            "predictiveSupport": quality,
            "predictiveModel": model,
            "sourceVectorTemporalCompatibility": {"compatible": quality},
        },
    }


class AdditionalSectorSourceLocalizationTests(unittest.TestCase):
    def test_real_unused_sectors_are_deterministic_and_exclude_used(self):
        self.assertEqual(
            unused_official_sectors(identity(), bridge()), [1, 27, 28, 67, 68]
        )
        self.assertEqual(
            unused_official_sectors(identity(), bridge()),
            unused_official_sectors(identity(), bridge()),
        )
        self.assertFalse(
            set(unused_official_sectors(identity(), bridge())) & {94, 95, 102, 103}
        )

    def test_preparation_preserves_frozen_evidence_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = prepare_additional_sector_source_localization(
                interpretation=authorization(),
                localization_bridge=bridge(),
                identity=identity(),
                output_dir=output_dir,
                investigation_id="real",
            )
            for key in (
                "referenceFamilyPeriodDays",
                "residualReferenceFrequency",
                "residualTimeReferenceDays",
                "subtractedHarmonicOrders",
                "catalogCandidates",
                "targetSky",
                "ticID",
            ):
                self.assertEqual(result[key], bridge()[key])
            with self.assertRaisesRegex(RuntimeError, "incomplete or unsafe"):
                prepare_additional_sector_source_localization(
                    interpretation=authorization(),
                    localization_bridge=bridge(0.01),
                    identity=identity(),
                    output_dir=output_dir,
                    investigation_id="real",
                )

    def test_conservative_cross_sector_rules(self):
        prep = bridge()
        one = interpret_additional_sector_source_localization(
            prep,
            {
                "sectorResults": [
                    sector("candidate-1"),
                    sector(),
                    sector(),
                    sector(),
                    sector(),
                ]
            },
        )
        self.assertEqual(one["classification"], "UNRESOLVED")
        self.assertEqual(one["multipleSourceSupportCount"], 0)
        self.assertEqual(
            [row["sourceSupportClassification"] for row in one["sectorResults"]],
            [
                "UNIQUE_SOURCE_SUPPORTED",
                "UNRESOLVED",
                "UNRESOLVED",
                "UNRESOLVED",
                "UNRESOLVED",
            ],
        )
        candidate = interpret_additional_sector_source_localization(
            prep,
            {
                "sectorResults": [sector("candidate-1") for _ in range(3)]
                + [sector(), sector()]
            },
        )
        self.assertEqual(candidate["classification"], "CATALOG_SOURCE_SUPPORTED")
        self.assertEqual(candidate["preferredCandidate"], prep["catalogCandidates"][0])
        target = interpret_additional_sector_source_localization(
            prep,
            {
                "sectorResults": [sector("target") for _ in range(3)]
                + [sector(), sector()]
            },
        )
        self.assertEqual(target["classification"], "TARGET_SUPPORTED")
        switching = interpret_additional_sector_source_localization(
            prep,
            {
                "sectorResults": [
                    sector("target"),
                    sector("target"),
                    sector("candidate-1"),
                    sector("candidate-1"),
                    sector("target"),
                ]
            },
        )
        self.assertEqual(switching["classification"], "SOURCE_SWITCHING_OR_BLEND")
        blend = interpret_additional_sector_source_localization(
            prep,
            {
                "sectorResults": [sector("target") for _ in range(3)]
                + [sector(multiple=True), sector(multiple=True)]
            },
        )
        self.assertEqual(blend["classification"], "SOURCE_SWITCHING_OR_BLEND")

    def test_invalid_sector_cannot_support_source(self):
        result = interpret_additional_sector_source_localization(
            bridge(),
            {"sectorResults": [sector("candidate-1", quality=False) for _ in range(5)]},
        )
        self.assertEqual(result["classification"], "UNRESOLVED")
        self.assertTrue(
            all(
                row["sourceSupportClassification"] == "SCIENTIFICALLY_INVALID"
                for row in result["sectorResults"]
            )
        )

    def test_run_propagates_contract_errors_but_records_pixel_unavailability(self):
        preparation = {**bridge(), "sectors": [1], "artifactRoot": "/tmp"}
        with mock.patch(
            "workflows.tess.tess_additional_sector_source_localization._production_sector_inputs",
            side_effect=ValueError("bad evidence contract"),
        ), self.assertRaises(ValueError):
            run_additional_sector_source_localization(preparation)
        with mock.patch(
            "workflows.tess.tess_additional_sector_source_localization._production_sector_inputs",
            side_effect=RuntimeError(
                "No official TPF or TESScut coverage available for Sector 1."
            ),
        ):
            result = run_additional_sector_source_localization(preparation)
        self.assertEqual(result["sectorResults"][0]["availability"], "UNAVAILABLE")
        with mock.patch(
            "workflows.tess.tess_additional_sector_source_localization._production_sector_inputs",
            side_effect=OSError("generic filesystem failure"),
        ), self.assertRaisesRegex(OSError, "generic filesystem failure"):
            run_additional_sector_source_localization(preparation)


if __name__ == "__main__":
    unittest.main()
