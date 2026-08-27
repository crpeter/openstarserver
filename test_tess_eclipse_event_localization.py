import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from workflows.tess.tess_eclipse_event_localization import (
    authoritative_binary_gate,
    localize_eclipse_events,
)
from openstar_investigation import ArtifactReference, InvestigationStage, InvestigationStore, sha256_file, sha256_json
from openstar_workflow import RetryableExecutionError, StageRequest
from workflows.tess.tess_sector_archive import TessArchiveTransientError


class EclipseEventLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.binary = {
            "resultVersion": "2.0", "recommendedNextTest": "ECLIPSE_EVENT_SOURCE_LOCALIZATION",
            "catalogAnswerKeyUsed": False,
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "independentEvidence": {"classification": "REPLICATED_ECLIPSE_LIKE_EVENT_SUPPORTED",
                                    "supportingIndependentSectorCount": 3},
            "linearEphemeris": {"coherent": True, "primaryTimingConsistent": True,
                                "refinedPeriodDays": 2.0, "referenceEpoch": 0.5,
                                "cycleAssignments": [
                                    {"sector": sector, "eventEpoch": 0.5, "cycleNumber": 0}
                                    for sector in (1, 2, 3, 4)]},
            "sectorResults": [
                {"sector": sector, "role": "PRIMARY" if sector == 1 else "INDEPENDENT",
                 "usable": True, "durationDays": 0.2, "eventEpoch": 0.5}
                for sector in (1, 2, 3, 4)
            ],
        }
        self.identity = {"tic": {"metadata": {"raDeg": 1.0, "decDeg": 2.0}}}

    def sector(self, sector, source=(3.0, 3.0), weak=False, candidates=None,
               sky_offset=None, unstable_event=False):
        rng = np.random.default_rng(sector)
        times = np.linspace(0, 12, 1200)
        yy, xx = np.mgrid[:7, :7]
        psf = np.exp(-((xx-source[0])**2 + (yy-source[1])**2) / 1.2)
        phase = abs((times - 0.5 + 1.0) % 2.0 - 1.0)
        depth = 0.001 if weak else 4.0
        cube = rng.normal(0, 0.05, (len(times), 7, 7)) + 20 * psf
        cube[phase <= 0.1] -= depth * psf
        if unstable_event:
            shifted = np.exp(-((xx-5.0)**2 + (yy-3.0)**2) / 1.2)
            event = (phase <= 0.1) & (np.rint((times - 0.5) / 2.0) == 3)
            cube[event] += depth * psf
            cube[event] -= depth * shifted
        result = {"sector": sector, "times": times, "fluxCube": cube,
                "targetPixel": {"x": 3.0, "y": 3.0},
                "catalogHypotheses": candidates or [
                    {"id": "TARGET", "isTarget": True, "pixel": {"x": 3.0, "y": 3.0}}],
                "pixelScaleArcsec": 21.0}
        if sky_offset is not None:
            result.update({"skyOffsetEastArcsec": sky_offset[0],
                           "skyOffsetNorthArcsec": sky_offset[1],
                           "centroidSky": {"raDeg": 1.0 + sky_offset[0] / 3600.0,
                                           "decDeg": 2.0 + sky_offset[1] / 3600.0}})
        return result

    def localize(self, inputs):
        return localize_eclipse_events(binary_confirmation=self.binary, identity=self.identity,
                                       tic_id=42, sector_inputs=inputs)

    def test_target_centered_eclipses_resolve_to_target(self):
        result = self.localize([self.sector(i) for i in (1, 2, 3, 4)])
        self.assertEqual("TARGET_CONSISTENT_ECLIPSE_SOURCE", result["classification"])

    def test_off_target_eclipses_resolve_to_frozen_candidate(self):
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 5, "y": 3}}]
        result = self.localize([self.sector(i, (5, 3), candidates=candidates) for i in (1, 2, 3, 4)])
        self.assertEqual("OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE", result["classification"])
        self.assertEqual("NEIGHBOR", result["attributedCatalogHypothesis"])

    def test_conflicting_independent_sectors_are_unresolved(self):
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 5, "y": 3}}]
        inputs = [self.sector(1, candidates=candidates), self.sector(2, candidates=candidates),
                  self.sector(3, candidates=candidates), self.sector(4, (5, 3), candidates=candidates)]
        self.assertEqual("CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND", self.localize(inputs)["classification"])

    def test_primary_cannot_rescue_two_independent_sectors(self):
        binary = dict(self.binary)
        binary["sectorResults"] = self.binary["sectorResults"][:3]
        result = localize_eclipse_events(binary_confirmation=binary, identity=self.identity, tic_id=42,
                                         sector_inputs=[self.sector(i) for i in (1, 2, 3)])
        self.assertFalse(result["sourceAttributionResolved"])

    def test_weak_images_remain_unresolved(self):
        result = self.localize([self.sector(i, weak=True) for i in (1, 2, 3, 4)])
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertTrue(result["sectorResults"])
        for sector in result["sectorResults"]:
            self.assertLess(sector["differenceImagePeakSNR"], 4.0)
            self.assertIn("WEAK_DIFFERENCE_IMAGE_SNR", sector["qualityRejectionReasons"])

    def test_overlapping_catalog_positions_fail_closed(self):
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "BLEND", "isTarget": False, "pixel": {"x": 3.1, "y": 3.1}}]
        result = self.localize([self.sector(i, candidates=candidates) for i in (1, 2, 3, 4)])
        self.assertFalse(result["sourceAttributionResolved"])

    def test_nearby_candidate_is_ambiguous_at_centroid_uncertainty(self):
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "NEARBY", "isTarget": False, "pixel": {"x": 3.6, "y": 3}}]
        result = self.localize([self.sector(i, source=(3.25, 3.0), candidates=candidates)
                                for i in (1, 2, 3, 4)])
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertTrue(all(item["classification"] == "AMBIGUOUS" for item in result["sectorResults"]))
        for item in result["sectorResults"]:
            distances = sorted(value["distancePixels"] for value in item["catalogDistances"])
            self.assertLess(distances[1] - distances[0], item["requiredCatalogMarginPixels"])

    def test_unrelated_off_catalog_detector_centroids_do_not_resolve(self):
        inputs = [self.sector(1, (5, 3), sky_offset=(20, 0)),
                  self.sector(2, (5, 3), sky_offset=(20, 0)),
                  self.sector(3, (5, 3), sky_offset=(-80, 40)),
                  self.sector(4, (5, 3), sky_offset=(90, -50))]
        self.assertFalse(self.localize(inputs)["sourceAttributionResolved"])

    def test_missing_wcs_cannot_resolve_off_catalog(self):
        result = self.localize([self.sector(i, (5, 3)) for i in (1, 2, 3, 4)])
        self.assertFalse(result["sourceAttributionResolved"])

    def test_three_sky_consistent_off_catalog_sectors_resolve(self):
        inputs = [self.sector(1, (5, 3), sky_offset=(22, -4)),
                  self.sector(2, (5, 3), sky_offset=(21, -3)),
                  self.sector(3, (5, 3), sky_offset=(23, -5)),
                  self.sector(4, (5, 3), sky_offset=(20, -4))]
        result = self.localize(inputs)
        self.assertEqual("CONSISTENTLY_OFF_TARGET_ECLIPSE_SOURCE", result["classification"])
        self.assertTrue(result["sourceAttributionResolved"])

    def test_event_jackknife_instability_blocks_attribution(self):
        result = self.localize([self.sector(i, unstable_event=True) for i in (1, 2, 3, 4)])
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertTrue(any("EVENT_JACKKNIFE_UNSTABLE" in item["qualityRejectionReasons"]
                            for item in result["sectorResults"]))

    def test_programmer_errors_propagate(self):
        malformed = self.sector(1)
        del malformed["fluxCube"]
        with self.assertRaises(KeyError):
            self.localize([malformed] + [self.sector(i) for i in (2, 3, 4)])

    def test_pixel_data_cannot_change_frozen_clock_or_duration(self):
        result = self.localize([self.sector(i) for i in (1, 2, 3, 4)])
        self.assertEqual(2.0, result["frozenEphemeris"]["refinedPeriodDays"])
        self.assertTrue(all(item["frozenMask"]["durationDays"] == 0.2
                            and not item["frozenMask"]["phaseOrDurationSearched"]
                            for item in result["sectorResults"]))

    def test_exact_gate_rejects_answer_key_or_wrong_recommendation(self):
        self.assertTrue(authoritative_binary_gate(self.binary))
        for key, value in (("catalogAnswerKeyUsed", True), ("recommendedNextTest", "OTHER")):
            changed = dict(self.binary); changed[key] = value
            self.assertFalse(authoritative_binary_gate(changed))

    def test_archive_transient_stage_preserves_frozen_catalog_provenance(self):
        from workflows.tess.tess_investigation import build_engine
        from workflows.tess.tess_eclipse_event_localization import HANDLER_ID, PREPARE_HANDLER_ID
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            store = InvestigationStore(directory)
            investigation = store.create("eclipse-retry", "test", "1")
            catalog = {"catalogHypotheses": [{"sourceID": "TIC-42", "isTarget": True,
                                               "raDeg": 1.0, "decDeg": 2.0}]}
            catalog_sha = sha256_json(catalog)
            artifact_path = Path(directory) / "frozen-catalog.json"
            artifact_path.write_text("{}\n", encoding="utf-8")
            artifact = ArtifactReference(str(artifact_path.resolve()), sha256_file(artifact_path), "application/json")
            stages = (
                InvestigationStage("001-prepare-target", "openstar.tess.prepare-target", "COMPLETE", None, {},
                                   result={"ticID": 42}),
                InvestigationStage("003-catalog-identity", "openstar.tess.catalog-identity", "COMPLETE", None, {},
                                   result=self.identity),
                InvestigationStage("006-independent", "openstar.tess.independent.prepare", "COMPLETE", None, {},
                                   result={"preparedSectors": []}),
                InvestigationStage("020-binary", "openstar.tess.binary-confirmation.analyze", "COMPLETE", None, {},
                                   result=self.binary),
                InvestigationStage("021-freeze", PREPARE_HANDLER_ID, "COMPLETE", "020-binary", {},
                                   result={"frozenCatalog": catalog, "frozenCatalogSHA256": catalog_sha},
                                   artifacts=(artifact,)),
            )
            investigation = type(investigation)(**{**investigation.__dict__, "stages": stages})
            store.save(investigation)
            engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            request = StageRequest("022-localize", HANDLER_ID, {}, "021-freeze")
            with mock.patch("workflows.tess.tess_investigation.localize_eclipse_events",
                            side_effect=TessArchiveTransientError("archive outage")), \
                    mock.patch("workflows.tess.tess_investigation.freeze_catalog_hypotheses") as freeze, \
                    self.assertRaises(RetryableExecutionError):
                engine.run_stage(investigation, request, software_id="test", software_version="1")
            freeze.assert_not_called()
            failed = store.load(investigation.id).stages[-1]
            self.assertEqual("TRANSIENT_INFRASTRUCTURE", failed.failure_classification)
            self.assertEqual(catalog_sha, failed.provenance.input_hashes["frozenCatalog"])
            self.assertEqual(sha256_json(self.binary), failed.provenance.input_hashes["binaryConfirmation"])
            self.assertEqual((artifact,), failed.artifacts)


if __name__ == "__main__":
    unittest.main()
