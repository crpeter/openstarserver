import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import (
    ArtifactReference, InvestigationStage, InvestigationStore, StageProvenance,
    sha256_file, sha256_json,
)
from workflows.tess import tess_eclipse_common_support as common
from workflows.tess import tess_eclipse_event_localization as original
import test_tess_eclipse_event_localization as fixtures


class CommonSupportImageTests(unittest.TestCase):
    @staticmethod
    def cube_with_noisy_core(wing_x=2, event_x=5, scene_x=5):
        """Balanced noise changes SNR spatially without changing either mean image."""
        yy, xx = np.mgrid[:11, :11]
        event = 100 * np.exp(-((xx - event_x) ** 2 + (yy - 5) ** 2) / 2)
        scene = 10000 * np.exp(-((xx - scene_x) ** 2 + (yy - 5) ** 2) / 2) + 20 * event
        desired_snr = np.full((11, 11), 20.0)
        desired_snr[5, wing_x] = 40.0
        noise_amplitude = event / desired_snr * np.sqrt(99 / 2)
        balanced = np.tile([-1.0, 1.0], 50)[:, None, None]
        high = scene + balanced * noise_amplitude
        low = scene - event + balanced * noise_amplitude
        return np.concatenate([high, low]), np.ones((11, 11), bool), np.arange(100), np.arange(100, 200)

    def test_snr_wing_cannot_exclude_core_in_either_direction(self):
        for wing_x in (2, 8):
            with self.subTest(wing_x=wing_x):
                args = self.cube_with_noisy_core(wing_x)
                legacy = original._centroid_from_frames(*args)
                revised = common.common_support_image(*args)
                self.assertEqual((legacy["peakX"], legacy["peakY"]), (wing_x, 5))
                # The old radius is 2.5 pixels: the central maximum is excluded.
                self.assertGreater(abs(legacy["centroidX"] - 5), 1.0)
                self.assertAlmostEqual(revised["centroidX"], 5.0, places=8)
                self.assertAlmostEqual(revised["centroidY"], 5.0, places=8)
                self.assertTrue(revised["commonSupportMask"][5][5])
                self.assertEqual(revised["supportPixelCount"], 121)
                self.assertFalse(revised["peakUsedForSpatialSelection"])
                self.assertFalse(revised["snrUsedForCentroidWeights"])

    def test_genuine_off_target_loss_is_not_recentered_to_scene(self):
        revised = common.common_support_image(*self.cube_with_noisy_core(wing_x=4, event_x=7, scene_x=4))
        self.assertGreater(revised["centroidX"], 6.9)
        self.assertLess(revised["outOfEventCentroid"]["x"], 5.0)
        self.assertGreater(revised["differenceMinusOutOfEventPixels"]["distancePixels"], 2.0)
        self.assertFalse(revised["astrometricCorrectionApplied"])

    def test_invalid_pixels_never_enter_either_image_centroid(self):
        cube, valid, high, low = self.cube_with_noisy_core()
        valid[0, 0] = False
        expected = common.common_support_image(cube, valid, high, low)
        cube[:, 0, 0] = 1e20
        actual = common.common_support_image(cube, valid, high, low)
        self.assertEqual(expected, actual)
        self.assertEqual(actual["outOfEventImage"][0][0], 0)

    def test_negative_event_loss_is_unavailable(self):
        cube, valid, high, low = self.cube_with_noisy_core()
        with self.assertRaisesRegex(original.EclipseLocalizationDataUnavailable, "No positive common-support event loss"):
            common.common_support_image(cube, valid, low, high)

    def test_malformed_valid_pixel_data_is_not_silently_ignored(self):
        cube, valid, high, low = self.cube_with_noisy_core()
        cube[0, 5, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            common.common_support_image(cube, valid, high, low)


class CommonSupportSectorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EclipseEventLocalizationTests()
        self.fixture.setUp()
        self.candidates = [
            {"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
            {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 5, "y": 3}},
        ]

    def measure(self, sector, **kwargs):
        item = self.fixture.sector(sector, candidates=self.candidates, **kwargs)
        frozen = original._frozen_sector(self.fixture.binary, sector)
        return original.measure_eclipse_sector(item, frozen, self.fixture.binary["linearEphemeris"],
                                                centroid_method=common.CENTROID_METHOD)

    def summarize(self, sectors):
        return original.summarize_eclipse_results(
            results=sectors, rejections=[], binary_confirmation=self.fixture.binary,
            identity=self.fixture.identity, frozen_catalog=None, result_version=common.RESULT_VERSION)

    def test_target_events_pass_the_existing_replication_rule(self):
        result = self.summarize([self.measure(i) for i in (1, 2, 3, 4)])
        self.assertEqual(result["classification"], "TARGET_CONSISTENT_ECLIPSE_SOURCE")
        self.assertEqual(result["resultVersion"], common.RESULT_VERSION)
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertFalse(result["companionNatureResolved"])

    def test_genuine_neighbor_events_keep_neighbor_attribution(self):
        result = self.summarize([self.measure(i, source=(5, 3)) for i in (1, 2, 3, 4)])
        self.assertEqual(result["classification"], "OFF_TARGET_CATALOG_CANDIDATE_ECLIPSE_SOURCE")
        self.assertEqual(result["attributedCatalogHypothesis"], "NEIGHBOR")

    def test_one_conflicting_sector_cannot_be_outvoted(self):
        sectors = [self.measure(i) for i in (1, 2, 3)] + [self.measure(4, source=(5, 3))]
        # Three supporting INDEPENDENT sectors plus one usable conflict.
        extra = copy.deepcopy(sectors[1]); extra["sector"] = 5
        result = self.summarize(sectors + [extra])
        self.assertEqual(result["classification"], "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND")
        self.assertFalse(result["sourceAttributionResolved"])

    def test_primary_does_not_count_as_third_independent_sector(self):
        result = self.summarize([self.measure(i) for i in (1, 2, 3)])
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertEqual(result["usableIndependentSectorCount"], 2)

    def test_unstable_event_positions_fail_the_existing_uncertainty_gate(self):
        item = self.fixture.sector(1, candidates=self.candidates)
        yy, xx = np.mgrid[:7, :7]
        target = np.exp(-((xx - 3) ** 2 + (yy - 3) ** 2) / 1.2)
        neighbor = np.exp(-((xx - 5) ** 2 + (yy - 3) ** 2) / 1.2)
        inside, _, cycles = original._event_bins(item["times"], 2.0, 0.5, 0.2)
        # One event strongly favors a different source. Omitting it must alter
        # the full-support centroid, rather than inheriting v1 uncertainty.
        item["fluxCube"][inside & (cycles == 3)] += 4 * target - 80 * neighbor
        sector = original.measure_eclipse_sector(
            item, original._frozen_sector(self.fixture.binary, 1), self.fixture.binary["linearEphemeris"],
            centroid_method=common.CENTROID_METHOD)
        self.assertGreater(sector["centroidUncertaintyPixels"], original.MAX_JACKKNIFE_UNCERTAINTY_PIXELS)
        self.assertIn("EVENT_JACKKNIFE_UNSTABLE", sector["qualityRejectionReasons"])
        self.assertFalse(sector["usable"])

    def test_jackknife_uses_the_new_estimator_on_identical_support(self):
        with mock.patch.object(common, "common_support_image", wraps=common.common_support_image) as image:
            sector = self.measure(1)
        self.assertGreater(len(image.call_args_list), 3)
        first_support = image.call_args_list[0].args[1]
        for call in image.call_args_list[1:]:
            np.testing.assert_array_equal(call.args[1], first_support)
        self.assertEqual(len(sector["eventJackknifeCentroids"]), len(image.call_args_list) - 1)
        self.assertGreaterEqual(sector["centroidUncertaintyPixels"], 0.12)
        self.assertEqual(sector["frozenMask"]["durationDays"], 0.2)
        self.assertEqual(sector["frozenMask"]["periodDays"], 2.0)
        self.assertFalse(sector["frozenMask"]["phaseOrDurationSearched"])

    def test_default_path_keeps_v1_and_never_calls_new_method(self):
        with mock.patch.object(common, "common_support_image", side_effect=AssertionError("opt-in only")):
            result = self.fixture.localize([self.fixture.sector(i) for i in (1, 2, 3, 4)])
        self.assertEqual(result["resultVersion"], original.RESULT_VERSION)
        self.assertNotIn("methodID", result["sectorResults"][0]["differenceImage"])

    def test_unknown_method_rejected_before_pixel_processing(self):
        with self.assertRaisesRegex(ValueError, "Unknown eclipse centroid method"):
            original.measure_eclipse_sector({}, {}, {}, centroid_method="typo")


class CommonSupportReanalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state, self.output = self.root / "state", self.root / "result-v2.json"
        self.store = InvestigationStore(self.state / "investigations")
        fixture = fixtures.EclipseEventLocalizationTests(); fixture.setUp()
        self.binary, self.identity = fixture.binary, fixture.identity
        self.catalog = {"catalogHypotheses": [
            {"sourceID": "TARGET", "ticID": 42, "isTarget": True},
            {"sourceID": "NEIGHBOR", "ticID": 43, "isTarget": False}]}
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 5, "y": 3}}]
        self.inputs = {i: fixture.sector(i, (5, 3) if i == 4 else (3, 3), candidates=candidates)
                       for i in (1, 2, 3, 4)}
        self.original_result = original.localize_eclipse_events(
            binary_confirmation=self.binary, identity=self.identity, tic_id=42,
            sector_inputs=list(self.inputs.values()), frozen_catalog=self.catalog)
        # Match eclipse_event_source_localization_stage's production envelope.
        self.original_result["frozenCatalogSHA256"] = sha256_json(self.catalog)
        self.original_result["pixelEvidenceSHA256BySector"] = {
            str(s["sector"]): s["pixelInputSHA256"] for s in self.original_result["sectorResults"]}
        self.investigation = self.store.create("common-support-fixture", "openstar.tess", "1")
        for handler, result in (
            ("openstar.tess.catalog-identity", self.identity),
            ("openstar.tess.binary-confirmation.analyze", self.binary),
            (original.PREPARE_HANDLER_ID, {"frozenCatalog": self.catalog,
                                          "frozenCatalogSHA256": sha256_json(self.catalog)}),
            (original.HANDLER_ID, self.original_result),
            ("openstar.tess.finalize", {"claim": {"claim": "CANDIDATE_PERIOD"}}),
        ):
            self.append(handler, result)
        self.investigation = replace(self.investigation, status="COMPLETE")
        self.store.save(self.investigation)

    def append(self, handler, result):
        stages = self.investigation.stages
        stage = InvestigationStage(
            id=f"{len(stages) + 1:03d}-stage", handler_id=handler, status="RUNNING",
            triggered_by_stage_id=stages[-1].id if stages else None, parameters={})
        self.investigation = self.store.append_running_stage(self.investigation, stage)
        artifact = self.store.directory_for(self.investigation.id) / (stage.id + "-result.json")
        artifact.write_text(json.dumps(result))
        terminal = replace(stage, status="COMPLETE", result=result, stop=handler == "openstar.tess.finalize",
                           artifacts=(ArtifactReference(str(artifact), sha256_file(artifact)),),
                           provenance=StageProvenance("test", "1", result_hash=sha256_json(result)))
        self.investigation = self.store.complete_current_stage(self.investigation, terminal)

    def rewrite_localization(self, result):
        """Deliberately forge self-consistent ledgers to reach semantic gates."""
        stage = self.investigation.stages[-2]
        artifact = Path(stage.artifacts[0].path)
        artifact.write_text(json.dumps(result))
        stage = replace(stage, result=result,
                        artifacts=(ArtifactReference(str(artifact), sha256_file(artifact)),),
                        provenance=replace(stage.provenance, result_hash=sha256_json(result)))
        self.store.stage_path_for(self.investigation.id, stage.id).write_text(json.dumps(asdict(stage)))
        self.investigation = replace(self.investigation, stages=self.investigation.stages[:-2] +
                                     (stage, self.investigation.stages[-1]))
        self.store.save(self.investigation)

    def run_reanalysis(self, execute=False):
        return common.run_reanalysis(self.state, self.investigation.id, self.output, execute=execute)

    def acquisition(self, inputs=None):
        inputs = self.inputs if inputs is None else inputs
        return mock.patch.object(original, "_production_input",
                                 side_effect=lambda tic, identity, frozen, catalog: inputs[frozen["sector"]])

    def test_validation_does_not_acquire_or_publish(self):
        with mock.patch.object(original, "_production_input", side_effect=AssertionError("no acquisition")):
            self.assertEqual(self.run_reanalysis()["status"], "VALIDATED_NO_CHANGES")
        self.assertFalse(self.output.exists())

    def test_public_reanalysis_preserves_bytes_and_real_conflict(self):
        before = {p: p.read_bytes() for p in self.state.rglob("*") if p.is_file()}
        with self.acquisition() as acquire:
            result = self.run_reanalysis(True)
        self.assertEqual(acquire.call_count, 4)
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(result["resultVersion"], common.RESULT_VERSION)
        self.assertEqual(result["classification"], "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND")
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertFalse(result["investigationModified"])
        self.assertFalse(result["claimLevelChanged"])
        self.assertEqual([s["sector"] for s in result["sectorResults"]], [1, 2, 3, 4])
        self.assertEqual(result["parentHashes"]["originalResultSHA256"], sha256_json(self.original_result))
        self.assertEqual(result["methodContractSHA256"], sha256_json(result["methodContract"]))
        self.assertEqual(before, {p: p.read_bytes() for p in self.state.rglob("*") if p.is_file()})

    def test_wrong_parent_or_clock_or_coverage_rejected_with_consistent_ledgers(self):
        for change, message in (("parent", "ancestry"), ("clock", "event definition"),
                                ("coverage", "frozen localization sectors"),
                                ("metadata", "summary")):
            with self.subTest(change=change):
                bad = copy.deepcopy(self.original_result)
                if change == "parent": bad["binaryConfirmationSHA256"] = "a" * 64
                elif change == "clock": bad["sectorResults"][0]["frozenMask"]["durationDays"] *= 2
                elif change == "coverage": bad["sectorResults"].pop()
                else: bad["pixelEvidenceSHA256BySector"]["1"] = "a" * 64
                self.rewrite_localization(bad)
                with mock.patch.object(original, "_production_input", side_effect=AssertionError("no acquisition")):
                    with self.assertRaisesRegex(ValueError, message):
                        self.run_reanalysis(True)
                self.assertFalse(self.output.exists())
        self.rewrite_localization(self.original_result)

    def test_changed_hash_or_pixels_refused_before_publication(self):
        for change in ("hash", "pixels"):
            with self.subTest(change=change):
                inputs = copy.deepcopy(self.inputs)
                if change == "hash": inputs[4]["pixelInputSHA256"] = "a" * 64
                else: inputs[4]["fluxCube"][:, 3, 3] *= 2
                with self.acquisition(inputs), self.assertRaisesRegex(ValueError, "pixel input differs|did not reproduce"):
                    self.run_reanalysis(True)
                self.assertFalse(self.output.exists())

    def test_ledger_or_artifact_corruption_refused_before_acquisition(self):
        stage = self.investigation.stages[0]
        for path in (self.store.stage_path_for(self.investigation.id, stage.id), Path(stage.artifacts[0].path)):
            saved = path.read_bytes()
            try:
                path.write_text("{}")
                with mock.patch.object(original, "_production_input", side_effect=AssertionError("no acquisition")):
                    with self.assertRaisesRegex(ValueError, "ledger|artifact"):
                        self.run_reanalysis(True)
                self.assertFalse(self.output.exists())
            finally:
                path.write_bytes(saved)

    def test_existing_or_in_state_output_refused(self):
        self.output.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_reanalysis(True)
        self.assertEqual(self.output.read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "outside"):
            common.run_reanalysis(self.state, self.investigation.id, self.state / "bad.json")

    def test_disappeared_source_fails_before_acquisition(self):
        with mock.patch.object(common, "_source_hashes", side_effect=FileNotFoundError("source disappeared")), \
                mock.patch.object(original, "_production_input", side_effect=AssertionError("no acquisition")):
            with self.assertRaises(FileNotFoundError):
                self.run_reanalysis(True)
        self.assertFalse(self.output.exists())

    def test_source_change_during_execution_refuses_output(self):
        with self.acquisition(), mock.patch.object(common, "_source_hashes", side_effect=[{"method": "a"}, {"method": "b"}]):
            with self.assertRaisesRegex(ValueError, "Method source changed"):
                self.run_reanalysis(True)
        self.assertFalse(self.output.exists())

    def test_state_change_during_execution_refuses_output(self):
        calls = 0

        def acquire(tic, identity, frozen, catalog):
            nonlocal calls
            calls += 1
            if calls == 4:
                changed = replace(self.investigation, metadata={"changedDuringRun": True})
                self.store.save(changed)
            return self.inputs[frozen["sector"]]

        with mock.patch.object(original, "_production_input", side_effect=acquire):
            with self.assertRaisesRegex(ValueError, "Investigation changed"):
                self.run_reanalysis(True)
        self.assertFalse(self.output.exists())

    def test_acquisition_error_cannot_silently_drop_a_conflicting_sector(self):
        def acquire(tic, identity, frozen, catalog):
            if frozen["sector"] == 4:
                raise RuntimeError("archive unavailable")
            return self.inputs[frozen["sector"]]

        with mock.patch.object(original, "_production_input", side_effect=acquire):
            with self.assertRaisesRegex(RuntimeError, "archive unavailable"):
                self.run_reanalysis(True)
        self.assertFalse(self.output.exists())

    def test_missing_wcs_clears_old_sky_position(self):
        for item in self.inputs.values():
            item.update({"centroidSky": {"raDeg": 99., "decDeg": 99.},
                         "skyOffsetEastArcsec": 99., "skyOffsetNorthArcsec": 99.})
        with self.acquisition():
            result = self.run_reanalysis(True)
        for sector in result["sectorResults"]:
            self.assertIsNone(sector["centroidSky"])
            self.assertIsNone(sector["skyOffsetEastArcsec"])
            self.assertIsNone(sector["skyOffsetNorthArcsec"])


if __name__ == "__main__":
    unittest.main()
