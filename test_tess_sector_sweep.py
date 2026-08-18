import json
import tempfile
import threading
import time
import unittest
import math
import struct
from dataclasses import replace
from pathlib import Path

from openstar_coordinator_client import ProjectRunResult
from workflows.tess.tess_preprocessing import prepare_tess_samples
from workflows.tess.tess_sector_archive import (
    MastTessSectorArchiveProvider, TessArchiveProduct, TessSectorInventoryStore,
)
from workflows.tess.tess_sector_scan import TessSectorScanTargetSource
from run_openstar_tess_sector_sweep import run_tess_sector_sweep


def product(tic, *, sector=7, author="SPOC", cadence=120, filename=None, rights="PUBLIC"):
    return TessArchiveProduct(sector, tic, f"TIC {tic}" if tic else "unknown",
        observation_id=f"obs-{tic}", mast_observation_id=f"mast-{tic}",
        data_uri=f"mast:TESS/{tic}", product_uri=f"mast:TESS/product/{tic}",
        product_filename=filename or f"tic-{tic}-{author}-{cadence}-lc.fits",
        author=author, cadence_seconds=cadence, data_rights=rights,
        source_fields={"proposal_id": "official"})


class FakeProvider:
    id, version = "fake-archive", "3"
    def __init__(self, products): self.products, self.calls, self.downloads = products, [], []
    def inventory_sector(self, sector): self.calls.append(sector); return self.products
    def download_light_curve(self, selected, destination):
        self.downloads.append(selected.tic_id); destination.mkdir(parents=True, exist_ok=True)
        path = destination / selected.product_filename; path.write_bytes(f"FITS {selected.tic_id}".encode()); return path


class Prepared:
    coordinates = (0.0, 1.0, 2.0); values = (-1.0, 0.0, 1.0)
    source_sample_count = finite_sample_count = sample_count = 3
    time_origin_days = 1400.0; baseline_days = 2.0


class FakeCoordinator:
    def __init__(self, fail_tic=None): self.calls = []; self.active = self.maximum_active = 0; self.lock = threading.Lock(); self.fail_tic = fail_tic
    def run_project(self, path, **kwargs):
        manifest = json.loads(Path(path).read_text()); tic = manifest["datasets"][0]["ticID"]
        with self.lock: self.active += 1; self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(.02)
            self.calls.append(tic)
            if tic == self.fail_tic: raise RuntimeError("bad coordinator target")
            dataset = {"bestFrequency": .5, "bestPeriodDays": 2.0, "bestPower": .8,
                       "periodStatus": "RELIABLE", "periodConfidence": "high",
                       "candidateFoldCoherence": .7, "coverageComplete": True}
            return ProjectRunResult(f"project-{tic}", {"projectID": f"project-{tic}", "status": "COMPLETE",
                 "datasets": [dataset], "nodeContributions": {"mac": 2, "iphone": 1}})
        finally:
            with self.lock: self.active -= 1


class InventoryTests(unittest.TestCase):
    def test_selection_inventory_resume_and_stable_ids(self):
        products = [product(3, cadence=600), product(2, author="TESS-SPOC", cadence=200),
                    product(3, cadence=120), product(2, author="SPOC", cadence=1800),
                    product(4, author="TESS-SPOC", cadence=600), product(5, author="OTHER"),
                    product(None), product(6, rights="PROPRIETARY"), product(7, sector=8)]
        provider = FakeProvider(products)
        with tempfile.TemporaryDirectory() as tmp:
            store = TessSectorInventoryStore(Path(tmp) / "inventory.json")
            inventory = store.create_or_load(7, provider)
            self.assertEqual([7], provider.calls)
            self.assertEqual([2, 3, 4], [x.product.tic_id for x in inventory.entries])
            self.assertEqual(1800, inventory.entries[0].product.cadence_seconds) # SPOC beats TESS-SPOC
            self.assertEqual(120, inventory.entries[1].product.cadence_seconds) # preferred SPOC
            reasons = {x.reason for x in inventory.skipped}
            self.assertTrue({"DUPLICATE_LOWER_PRIORITY_PRODUCT", "UNSUPPORTED_AUTHOR",
                             "NO_PARSEABLE_TIC_ID", "NONPUBLIC_PRODUCT", "INVALID_SECTOR"} <= reasons)
            raw = json.loads(store.path.read_text()); self.assertEqual("mast-2", raw["entries"][0]["product"]["mast_observation_id"])
            again = store.create_or_load(7, provider); self.assertEqual([7], provider.calls)
            self.assertEqual(TessSectorScanTargetSource(inventory).enumerate_targets(),
                             TessSectorScanTargetSource(again).enumerate_targets())
            self.assertEqual("tess-sector-scan-7-tic-2", TessSectorScanTargetSource(again).enumerate_targets()[0].investigation_id)

    def test_incompatible_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TessSectorInventoryStore(Path(tmp) / "inventory.json")
            store.create_or_load(7, FakeProvider([product(1)]))
            with self.assertRaises(RuntimeError): store.create_or_load(8, FakeProvider([]))
            other = FakeProvider([]); other.id = "different"
            with self.assertRaises(RuntimeError): store.create_or_load(7, other)

    def test_tess_spoc_shortest_fallback(self):
        provider = FakeProvider([product(1, author="TESS-SPOC", cadence=1800), product(1, author="TESS-SPOC", cadence=600)])
        with tempfile.TemporaryDirectory() as tmp:
            inv = TessSectorInventoryStore(Path(tmp)/"i.json").create_or_load(7, provider)
            self.assertEqual(600, inv.entries[0].product.cadence_seconds)


class PreprocessingTests(unittest.TestCase):
    def test_filter_sort_normalize_shift_quantize_and_downsample(self):
        result = prepare_tess_samples([1402, math.nan, 1400, 1403, 1401], [4, 5, 1, 7, 2], max_samples=3)
        self.assertEqual(5, result.source_sample_count); self.assertEqual(4, result.finite_sample_count)
        self.assertEqual((0.0, 1.0, 3.0), result.coordinates)
        mean = sum(result.values) / len(result.values)
        deviation = math.sqrt(sum((x - mean) ** 2 for x in result.values) / len(result.values))
        self.assertAlmostEqual(0, mean, places=6)
        self.assertAlmostEqual(1, deviation, places=6)
        self.assertEqual(1400, result.time_origin_days); self.assertEqual(3, result.baseline_days)
        self.assertTrue(all(struct.unpack("!f", struct.pack("!f", x))[0] == x for x in result.coordinates + result.values))

    def test_invalid_curves(self):
        with self.assertRaisesRegex(RuntimeError, "no finite"): prepare_tess_samples([math.nan], [1])
        with self.assertRaisesRegex(RuntimeError, "standard deviation"): prepare_tess_samples([1, 2], [3, 3])


class SweepTests(unittest.TestCase):
    def test_concurrent_shallow_evidence_resume_and_runtime_limit(self):
        provider = FakeProvider([product(3), product(1), product(2)])
        coordinator = FakeCoordinator()
        with tempfile.TemporaryDirectory() as tmp:
            # Patch the default preprocessing boundary only at the injected handler factory call.
            from workflows.tess import tess_sector_scan
            original = tess_sector_scan.read_and_prepare_tess_light_curve
            tess_sector_scan.read_and_prepare_tess_light_curve = lambda path: Prepared()
            try:
                code = run_tess_sector_sweep(7, "unused", tmp, max_targets=2,
                    max_concurrent_investigations=2, provider=provider, coordinator=coordinator)
                self.assertEqual(0, code); self.assertGreaterEqual(coordinator.maximum_active, 2)
                self.assertEqual([1, 2], sorted(coordinator.calls))
                inventory = json.loads((Path(tmp)/"tess-sector-7-inventory.json").read_text())
                self.assertEqual(3, len(inventory["entries"])) # limiter did not mutate inventory
                investigations = sorted((Path(tmp)/"investigations").glob("*/investigation.json"))
                self.assertEqual(2, len(investigations))
                for path in investigations:
                    state = json.loads(path.read_text()); self.assertEqual("COMPLETE", state["status"])
                    self.assertEqual(["openstar.tess-sector-scan.materialize-light-curve",
                                      "openstar.tess-sector-scan.broad-distributed-scan",
                                      "openstar.tess-sector-scan.persist-scan-evidence"], [x["handler_id"] for x in state["stages"]])
                    evidence = state["stages"][-1]["result"]
                    self.assertEqual(True, evidence["coverageComplete"]); self.assertEqual({"iphone": 1, "mac": 2}, evidence["nodeContributions"])
                    dataset = json.loads(Path(evidence["datasetArtifact"]).read_text())
                    self.assertEqual(dataset["coordinates"], dataset["times"]); self.assertEqual(dataset["values"], dataset["flux"])
                calls = list(coordinator.calls)
                run_tess_sector_sweep(7, "unused", tmp, max_targets=2, provider=provider, coordinator=coordinator)
                self.assertEqual(calls, coordinator.calls); self.assertEqual([7], provider.calls)
            finally: tess_sector_scan.read_and_prepare_tess_light_curve = original

    def test_one_failure_isolated_and_default_admits_all(self):
        provider = FakeProvider([product(1), product(2), product(3)]); coordinator = FakeCoordinator(fail_tic=2)
        with tempfile.TemporaryDirectory() as tmp:
            from workflows.tess import tess_sector_scan
            original = tess_sector_scan.read_and_prepare_tess_light_curve; tess_sector_scan.read_and_prepare_tess_light_curve = lambda path: Prepared()
            try:
                self.assertEqual(1, run_tess_sector_sweep(7, "unused", tmp, provider=provider, coordinator=coordinator,
                                                         max_concurrent_investigations=3))
                states = [json.loads(p.read_text())["status"] for p in (Path(tmp)/"investigations").glob("*/investigation.json")]
                self.assertEqual(3, len(states)); self.assertEqual(2, states.count("COMPLETE")); self.assertEqual(1, states.count("FAILED"))
            finally: tess_sector_scan.read_and_prepare_tess_light_curve = original


if __name__ == "__main__": unittest.main()
