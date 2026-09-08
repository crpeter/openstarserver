import copy
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import InvestigationStore, sha256_file, sha256_json
from workflows.tess import tess_eclipse_common_support as common
from workflows.tess import tess_eclipse_common_support_continuation as continuation
from workflows.tess import tess_eclipse_event_localization as original
import test_tess_eclipse_common_support as fixtures


class CommonSupportContinuationTests(unittest.TestCase):
    def setUp(self):
        # Use real producers and immutable ledgers. The fourth sector contains a
        # target-centered event whose high-SNR wing misleads only the v1 method.
        self.fixture = fixtures.CommonSupportReanalysisTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        cube, _, _, _ = fixtures.CommonSupportImageTests.cube_with_noisy_core()
        times, frames = [], []
        for cycle in range(6):
            epoch = 0.5 + 2 * cycle
            times.extend(np.linspace(epoch - 0.08, epoch + 0.08, 100))
            frames.extend(cube[100:])
            times.extend(np.linspace(epoch + 0.18, epoch + 0.32, 100))
            frames.extend(cube[:100])
        f.inputs[4] = {
            "sector": 4, "times": np.asarray(times), "fluxCube": np.asarray(frames),
            "targetPixel": {"x": 5.0, "y": 5.0}, "pixelScaleArcsec": 21.0,
            "catalogHypotheses": [
                {"id": "TARGET", "isTarget": True, "pixel": {"x": 5, "y": 5}},
                {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 3, "y": 5}}],
        }
        parent = original.localize_eclipse_events(
            binary_confirmation=f.binary, identity=f.identity, tic_id=42,
            sector_inputs=list(f.inputs.values()), frozen_catalog=f.catalog)
        parent["frozenCatalogSHA256"] = sha256_json(f.catalog)
        parent["pixelEvidenceSHA256BySector"] = {
            str(s["sector"]): s["pixelInputSHA256"] for s in parent["sectorResults"]}
        self.assertEqual(parent["classification"], "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND")
        f.rewrite_localization(parent)
        f.original_result = parent
        with f.acquisition():
            self.result = common.run_reanalysis(f.state, f.investigation.id, f.output, execute=True)
        self.assertEqual(self.result["classification"], "TARGET_CONSISTENT_ECLIPSE_SOURCE")
        self.digest = sha256_file(f.output)
        self.before = self.snapshot()
        # All calls under test are offline, even the execution path.
        self.offline = mock.patch.object(original, "_production_input", side_effect=AssertionError("No acquisition"))
        self.offline.start()
        self.addCleanup(self.offline.stop)

    def snapshot(self):
        return {str(p): p.read_bytes() for p in self.fixture.state.rglob("*") if p.is_file()}

    def run_continuation(self, execute=False, digest=None):
        f = self.fixture
        return continuation.run_continuation(f.state, f.investigation.id, f.output,
                                            self.digest if digest is None else digest, execute=execute)

    def write_altered(self, result):
        self.fixture.output.write_text(json.dumps(result))
        return sha256_file(self.fixture.output)

    def test_validation_is_read_only(self):
        result = self.run_continuation()
        self.assertEqual(result["status"], "VALIDATED_NO_CHANGES")
        self.assertFalse(result["investigationModified"])
        self.assertEqual(self.before, self.snapshot())
        self.assertFalse(Path(result["reportPath"]).exists())

    def test_records_two_versioned_stages_preserves_history_and_claim(self):
        f = self.fixture
        result = self.run_continuation(True)
        self.assertEqual(result["status"], "RECORDED")
        current = f.store.load(f.investigation.id)
        self.assertEqual(current.stages[:-2], f.investigation.stages)
        self.assertEqual(current.metadata, f.investigation.metadata)
        self.assertEqual(current.status, "COMPLETE")
        for path, data in self.before.items():
            if path != str(f.store.path_for(f.investigation.id)):
                self.assertEqual(Path(path).read_bytes(), data)
        accepted, final = current.stages[-2:]
        self.assertEqual(accepted.handler_id, continuation.IMPORT_HANDLER)
        self.assertEqual(accepted.result, self.result)
        self.assertEqual(accepted.triggered_by_stage_id, f.investigation.stages[-1].id)
        self.assertEqual(final.handler_id, continuation.FINAL_HANDLER)
        self.assertEqual(final.triggered_by_stage_id, accepted.id)
        self.assertEqual(accepted.next_stage["id"], final.id)
        self.assertTrue(final.stop)
        self.assertIsNone(final.next_stage)
        self.assertEqual(final.result["claim"], f.investigation.stages[-1].result["claim"])
        self.assertFalse(final.result["claimLevelChanged"])
        self.assertFalse(final.result["companionNatureResolved"])
        self.assertEqual(final.result["recommendedNextTest"], "SOURCE_ATTRIBUTION_REVIEW")
        for stage in (accepted, final):
            self.assertTrue(f.store.verified_terminal_stage_ledger_hash(current.id, stage))
            self.assertEqual(stage.provenance.result_hash, sha256_json(stage.result))
            for artifact in stage.artifacts:
                self.assertEqual(sha256_file(artifact.path), artifact.sha256)
        saved = next(a for a in accepted.artifacts if a.path.endswith("reviewed-reanalysis-v2.json"))
        self.assertEqual(Path(saved.path).read_bytes(), f.output.read_bytes())
        parent = next(a for a in accepted.artifacts if a.path.endswith("parent-investigation.json"))
        self.assertEqual(Path(parent.path).read_bytes(), self.before[str(f.store.path_for(current.id))])
        self.assertEqual(json.loads(Path(result["conclusionPath"]).read_text()), final.result)
        report = Path(result["reportPath"]).read_text()
        self.assertIn("CANDIDATE_PERIOD", report)
        self.assertIn(self.digest, report)
        self.assertIn("| 4 | INDEPENDENT | NEIGHBOR | TARGET |", report)

    def test_repeat_is_idempotent_and_verifies_recorded_artifacts(self):
        result = self.run_continuation(True)
        after = self.snapshot()
        self.assertEqual(self.run_continuation(True)["status"], "ALREADY_RECORDED")
        self.assertEqual(after, self.snapshot())
        report = Path(result["reportPath"])
        report.write_text("altered")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            self.run_continuation(True)

    def test_wrong_reviewed_digest_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, "Reviewed file SHA-256"):
            self.run_continuation(True, "0" * 64)
        self.assertEqual(self.before, self.snapshot())

    def test_rejects_duplicate_keys_and_nonfinite_json(self):
        original_bytes = self.fixture.output.read_bytes()
        for payload in (b'{"status":1,"status":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.subTest(payload=payload):
                self.fixture.output.write_bytes(payload)
                with self.assertRaises(ValueError):
                    self.run_continuation(True, sha256_file(self.fixture.output))
                self.assertEqual(self.before, self.snapshot())
        self.fixture.output.write_bytes(original_bytes)

    def test_semantic_tampering_rejected_even_with_matching_file_digest(self):
        changes = {
            "parent": lambda r: r["parentHashes"].update(investigationSHA256="a" * 64),
            "contract": lambda r: r["methodContract"].update(sourceMatchMaxPixels=9),
            "sources": lambda r: r["methodSourceSHA256"].pop(next(iter(r["methodSourceSHA256"]))),
            "claim": lambda r: r.update(companionNatureResolved=True),
            "summary": lambda r: r.update(usableIndependentSectorCount=999),
            "coverage": lambda r: r["sectorResults"].pop(),
            "pixelHash": lambda r: r["sectorResults"][0].update(pixelInputSHA256="b" * 64),
            "clock": lambda r: r["sectorResults"][0]["frozenMask"].update(periodDays=8),
            "image": lambda r: r["sectorResults"][0]["differenceImage"]["differenceImage"][0].__setitem__(0, 999),
            "centroid": lambda r: r["sectorResults"][0]["measuredPixelCentroid"].update(x=99),
            "uncertainty": lambda r: r["sectorResults"][0].update(centroidUncertaintyPixels=0.01),
            "jackknife": lambda r: r["sectorResults"][0]["eventJackknifeCentroids"].pop(),
            "conflict": lambda r: r["sectorResults"][-1].update(classification="CATALOG_CANDIDATE_CONSISTENT"),
            "discard": lambda r: r["sectorResults"][-1].update(usable=False),
            "margin": lambda r: r["sectorResults"][-1]["catalogDistances"][1].update(distancePixels=0.1),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                bad = copy.deepcopy(self.result)
                change(bad)
                digest = self.write_altered(bad)
                with self.assertRaises(ValueError):
                    self.run_continuation(True, digest)
                self.assertEqual(self.before, self.snapshot())

    def test_source_paths_can_relocate_but_digests_cannot_change(self):
        moved = copy.deepcopy(self.result)
        moved["methodSourceSHA256"] = {
            "/another/checkout/" + str(Path(p).relative_to(continuation.REPOSITORY_ROOT)): h
            for p, h in moved["methodSourceSHA256"].items()}
        digest = self.write_altered(moved)
        self.assertEqual(self.run_continuation(digest=digest)["status"], "VALIDATED_NO_CHANGES")
        moved["methodSourceSHA256"][next(iter(moved["methodSourceSHA256"]))] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Method source hashes"):
            self.run_continuation(True, self.write_altered(moved))

    def test_parent_ledger_tampering_refuses_writes(self):
        f = self.fixture
        ledger = f.store.stage_path_for(f.investigation.id, f.investigation.stages[0].id)
        ledger.write_text("{}")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "ledger"):
            self.run_continuation(True)
        self.assertEqual(before, self.snapshot())

    def test_existing_output_is_never_overwritten(self):
        package = self.fixture.store.directory_for(self.fixture.investigation.id) / "artifacts" / continuation.PACKAGE
        package.mkdir(parents=True)
        marker = package / "keep.txt"
        marker.write_text("preserve")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.run_continuation(True)
        self.assertEqual(before, self.snapshot())

    def test_snapshot_commit_failure_rolls_back_new_files_only(self):
        with mock.patch.object(InvestigationStore, "save", side_effect=OSError("injected save failure")):
            with self.assertRaisesRegex(OSError, "injected save failure"):
                self.run_continuation(True)
        self.assertEqual(self.before, self.snapshot())
        self.assertEqual(self.run_continuation(True)["status"], "RECORDED")

    def test_exception_after_snapshot_commit_preserves_published_evidence(self):
        real_save = InvestigationStore.save

        def saved_then_failed(store, investigation):
            real_save(store, investigation)
            raise OSError("injected post-commit failure")

        with mock.patch.object(InvestigationStore, "save", saved_then_failed):
            with self.assertRaisesRegex(OSError, "post-commit failure"):
                self.run_continuation(True)
        self.assertEqual(self.run_continuation(True)["status"], "ALREADY_RECORDED")

    def test_detected_state_change_refuses_publication(self):
        real_boundary = common.verified_boundary
        calls = 0

        def checked(*args):
            nonlocal calls
            calls += 1
            result = real_boundary(*args)
            if calls == 2:
                result = {**result, "investigationSHA256": "a" * 64}
            return result

        with mock.patch.object(common, "verified_boundary", side_effect=checked):
            with self.assertRaisesRegex(ValueError, "changed before publication"):
                self.run_continuation(True)
        self.assertEqual(self.before, self.snapshot())

    def test_input_change_after_validation_refuses_publication(self):
        real_validate = continuation.validate_reanalysis

        def changed(*args):
            result = real_validate(*args)
            self.fixture.output.write_text("{}")
            return result

        with mock.patch.object(continuation, "validate_reanalysis", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "file changed before publication"):
                self.run_continuation(True)
        self.assertEqual(self.before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
