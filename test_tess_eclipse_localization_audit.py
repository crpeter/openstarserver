import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from openstar_investigation import (
    ArtifactReference, InvestigationStage, InvestigationStore, StageProvenance,
    sha256_file, sha256_json,
)
from workflows.tess import tess_eclipse_event_localization as localization
from workflows.tess import tess_eclipse_localization_audit as audit
import test_tess_eclipse_event_localization as fixtures


class EclipseLocalizationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / "state"
        self.output = self.root / "audit.json"
        self.store = InvestigationStore(self.state / "investigations")
        fixture = fixtures.EclipseEventLocalizationTests()
        fixture.setUp()
        self.binary, self.identity = fixture.binary, fixture.identity
        self.catalog = {
            "catalogHypotheses": [
                {"sourceID": "TARGET", "ticID": 42, "isTarget": True},
                {"sourceID": "NEIGHBOR", "ticID": 43, "isTarget": False},
            ],
            "catalogQueries": {"tic": {"sources": [{"ticID": 42, "tmag": 10.0}, {"ticID": 43, "tmag": 20.0}]}},
        }
        candidates = [{"id": "TARGET", "isTarget": True, "pixel": {"x": 3, "y": 3}},
                      {"id": "NEIGHBOR", "isTarget": False, "pixel": {"x": 5, "y": 3}}]
        self.inputs = {i: fixture.sector(i, (5, 3) if i == 4 else (3, 3), candidates=candidates)
                       for i in (1, 2, 3, 4)}
        self.result = localization.localize_eclipse_events(
            binary_confirmation=self.binary, identity=self.identity, tic_id=42,
            sector_inputs=list(self.inputs.values()), frozen_catalog=self.catalog)
        self.investigation = self.store.create("audit-fixture", "openstar.tess", "1")
        for handler, result in (
            ("openstar.tess.catalog-identity", self.identity),
            ("openstar.tess.binary-confirmation.analyze", self.binary),
            (localization.PREPARE_HANDLER_ID, {"frozenCatalog": self.catalog, "frozenCatalogSHA256": sha256_json(self.catalog)}),
            (localization.HANDLER_ID, self.result),
            ("openstar.tess.finalize", {"claim": {"claim": "CANDIDATE_PERIOD"}}),
        ):
            self.append(handler, result)
        self.investigation = replace(self.investigation, status="COMPLETE")
        self.store.save(self.investigation)

    def append(self, handler, result):
        stages = self.investigation.stages
        stage = InvestigationStage(
            id=f"{len(stages) + 1:03d}-stage", handler_id=handler, status="RUNNING",
            triggered_by_stage_id=stages[-1].id if stages else None, parameters={},
        )
        self.investigation = self.store.append_running_stage(self.investigation, stage)
        artifact = self.store.directory_for(self.investigation.id) / (stage.id + "-result.json")
        artifact.write_text(json.dumps(result))
        terminal = replace(stage, status="COMPLETE", result=result, stop=handler == "openstar.tess.finalize",
                           artifacts=(ArtifactReference(str(artifact), sha256_file(artifact)),),
                           provenance=StageProvenance("audit-test", "1", result_hash=sha256_json(result)))
        self.investigation = self.store.complete_current_stage(self.investigation, terminal)

    def run_audit(self, **kwargs):
        return audit.run_audit(self.state, self.investigation.id, self.output, **kwargs)

    def measure(self, item, previous=None):
        frozen = self.binary["sectorResults"][0]
        if previous is None:
            previous = localization.measure_eclipse_sector(item, frozen, self.binary["linearEphemeris"])
            previous["pixelInputSHA256"] = item["pixelInputSHA256"]
        return audit.measure_audit(item, frozen, self.binary["linearEphemeris"], previous, self.catalog)

    def test_validation_has_no_acquisition_or_output(self):
        with mock.patch.object(localization, "_production_input", side_effect=AssertionError("acquisition")):
            result = self.run_audit()
        self.assertEqual(result["status"], "VALIDATED_NO_CHANGES")
        self.assertFalse(self.output.exists())

    def test_public_execution_preserves_state_and_original_conflict(self):
        before = {p: p.read_bytes() for p in self.state.rglob("*") if p.is_file()}
        with mock.patch.object(localization, "_production_input", side_effect=lambda tic, identity, frozen, catalog: self.inputs[frozen["sector"]]) as acquire:
            result = self.run_audit(execute=True)
        self.assertEqual(acquire.call_count, 4)
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertFalse(result["sourceAttributionResolved"])
        self.assertFalse(result["claimLevelChanged"])
        self.assertEqual(result["originalClassification"], "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND")
        self.assertEqual(before, {p: p.read_bytes() for p in self.state.rglob("*") if p.is_file()})

    def test_common_image_translation_cancels_in_differential_centroid(self):
        fixture = fixtures.EclipseEventLocalizationTests()
        fixture.setUp()
        item = fixture.sector(1, source=(3.6, 3.0))
        item["pixelInputSHA256"] = "a" * 64
        measured = self.measure(item)
        self.assertGreater(measured["outOfEventMinusCatalogTargetPixels"]["distancePixels"], 0.4)
        self.assertLess(measured["differenceMinusOutOfEventPixels"]["distancePixels"], 0.1)

    def test_off_target_event_does_not_cancel_with_scene_reference(self):
        item = copy.deepcopy(self.inputs[1])
        yy, xx = np.mgrid[:7, :7]
        neighbor = np.exp(-((xx - 5.0) ** 2 + (yy - 3.0) ** 2) / 1.2)
        target = np.exp(-((xx - 3.0) ** 2 + (yy - 3.0) ** 2) / 1.2)
        inside, _, _ = localization._event_bins(item["times"], 2.0, 0.5, 0.2)
        item["fluxCube"] += 8 * neighbor
        item["fluxCube"][inside] += 4 * target - 4 * neighbor
        measured = self.measure(item)
        self.assertGreater(measured["differenceMinusOutOfEventPixels"]["distancePixels"], 0.5)

    def test_flux_budget_is_conditional_and_never_excludes_candidate(self):
        result = audit.flux_budget(self.catalog, 0.01)["sources"][1]
        self.assertAlmostEqual(result["fluxRatioToTarget"], 0.0001)
        self.assertGreater(result["measuredApertureLossOverConditionalMaximum"], 100)
        self.assertFalse(result["candidateExcluded"])
        catalog = copy.deepcopy(self.catalog)
        catalog["catalogQueries"]["tic"]["sources"][1]["tmag"] = None
        result = audit.flux_budget(catalog, 0.01)["sources"][1]
        self.assertIsNone(result["equalThroughputMaximumPairDepth"])

    def test_changed_pixels_or_event_definition_refused_before_publication(self):
        for change in ("hash", "flux"):
            with self.subTest(change=change):
                items = copy.deepcopy(self.inputs)
                if change == "hash":
                    items[4]["pixelInputSHA256"] = "b" * 64
                else:
                    items[4]["fluxCube"][:, 3, 3] *= 2
                with mock.patch.object(localization, "_production_input", side_effect=lambda tic, identity, frozen, catalog: items[frozen["sector"]]):
                    with self.assertRaises(ValueError):
                        self.run_audit(execute=True)
                self.assertFalse(self.output.exists())
        previous = copy.deepcopy(self.result["sectorResults"][0])
        frozen = copy.deepcopy(self.binary["sectorResults"][0])
        frozen["eventEpoch"] += 0.15
        with self.assertRaises(ValueError):
            audit.measure_audit(self.inputs[1], frozen, self.binary["linearEphemeris"], previous, self.catalog)

    def test_unverified_ledger_or_artifact_refused_without_acquisition(self):
        paths = [self.store.stage_path_for(self.investigation.id, self.investigation.stages[0].id),
                 Path(self.investigation.stages[0].artifacts[0].path)]
        for path in paths:
            with self.subTest(path=path):
                original_bytes = path.read_bytes()
                path.write_text("{}")
                try:
                    with mock.patch.object(localization, "_production_input", side_effect=AssertionError("acquisition")):
                        with self.assertRaises(ValueError):
                            self.run_audit(execute=True)
                    self.assertFalse(self.output.exists())
                finally:
                    path.write_bytes(original_bytes)

    def test_existing_output_and_output_inside_state_refused(self):
        self.output.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_audit(execute=True)
        self.assertEqual(self.output.read_text(), "keep")
        with self.assertRaises(ValueError):
            audit.run_audit(self.state, self.investigation.id, self.state / "audit.json")


if __name__ == "__main__":
    unittest.main()
