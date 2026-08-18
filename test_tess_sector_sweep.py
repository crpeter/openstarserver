import json
import tempfile
import threading
import time
import unittest
import math
import struct
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from openstar_coordinator_client import ProjectRunResult
from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import InvestigationStore
from openstar_scheduler import InvestigationScheduler
from workflows.tess.tess_preprocessing import prepare_tess_samples
from workflows.tess.tess_sector_archive import (
    MastTessSectorArchiveProvider, TessArchiveProduct, TessSectorInventoryStore,
)
from workflows.tess.tess_sector_scan import (
    WORKFLOW_ID, TessSectorScanTargetSource, plan_tess_sector_scan,
    register_tess_sector_scan_handlers,
)
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
    def __init__(self, fail_tic=None, expected_concurrent=None):
        self.calls = []; self.active = self.maximum_active = 0; self.lock = threading.Lock(); self.fail_tic = fail_tic
        self.barrier = threading.Barrier(expected_concurrent) if expected_concurrent else None
    def run_project(self, path, **kwargs):
        manifest = json.loads(Path(path).read_text()); tic = manifest["datasets"][0]["ticID"]
        with self.lock: self.active += 1; self.maximum_active = max(self.maximum_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
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
    class _Row(dict):
        @property
        def colnames(self):
            return list(self)

    class _Scalar:
        def __init__(self, value): self.value = value
        def item(self): return self.value

    def _mock_mast_download(self, result, *, create_file):
        observations = MagicMock()

        def download_file(uri, *, local_path, cache):
            if create_file:
                Path(local_path).write_bytes(b"downloaded TESS FITS")
            return result

        observations.download_file.side_effect = download_file
        mast = types.ModuleType("astroquery.mast")
        mast.Observations = observations
        astroquery = types.ModuleType("astroquery")
        astroquery.mast = mast
        return observations, patch.dict(
            sys.modules,
            {"astroquery": astroquery, "astroquery.mast": mast},
        )

    def test_mast_tuple_complete_download_uses_exact_uri_and_path(self):
        selected = product(42, filename="selected-light-curve.fits")
        observations, mocked_modules = self._mock_mast_download(
            ("COMPLETE", None, None), create_file=True
        )
        with tempfile.TemporaryDirectory() as tmp, mocked_modules:
            destination = Path(tmp) / "source"
            expected = destination / "selected-light-curve.fits"
            actual = MastTessSectorArchiveProvider().download_light_curve(
                selected, destination
            )
            self.assertEqual(expected, actual)
            self.assertTrue(actual.exists())
            observations.download_file.assert_called_once_with(
                selected.product_uri, local_path=str(expected), cache=True
            )

    def test_mast_noncomplete_download_raises_even_if_file_exists(self):
        observations, mocked_modules = self._mock_mast_download(
            ("ERROR", "failed", None), create_file=True
        )
        with tempfile.TemporaryDirectory() as tmp, mocked_modules:
            with self.assertRaisesRegex(RuntimeError, "MAST download failed"):
                MastTessSectorArchiveProvider().download_light_curve(
                    product(42), Path(tmp)
                )
            self.assertEqual(1, observations.download_file.call_count)

    def test_mast_complete_download_without_file_raises(self):
        observations, mocked_modules = self._mock_mast_download(
            ("COMPLETE", None, None), create_file=False
        )
        with tempfile.TemporaryDirectory() as tmp, mocked_modules:
            with self.assertRaisesRegex(RuntimeError, "MAST download failed"):
                MastTessSectorArchiveProvider().download_light_curve(
                    product(42), Path(tmp)
                )
            self.assertEqual(1, observations.download_file.call_count)

    def test_mast_row_fields_replaces_nonfinite_values(self):
        row = self._Row(nan=float("nan"), positive=float("inf"),
                        negative=float("-inf"), finite=12.5,
                        scalar=self._Scalar(float("nan")))

        fields = MastTessSectorArchiveProvider._row_fields(row)

        self.assertEqual({"nan": None, "positive": None, "negative": None,
                          "finite": 12.5, "scalar": None}, fields)

    def test_mast_row_fields_recursively_replaces_nested_nonfinite_values(self):
        row = self._Row(metadata={"values": [1, float("nan"),
                                              (self._Scalar(float("inf")), -2.5)]})

        fields = MastTessSectorArchiveProvider._row_fields(row)

        self.assertEqual({"values": [1, None, (None, -2.5)]}, fields["metadata"])

    def test_mast_nonfinite_exptime_is_missing_and_inventory_is_strict_json(self):
        observations = [self._Row(obsid=101, obs_id="TIC 42", target_name="TIC 42",
                                  provenance_name="SPOC", t_exptime=float("nan"),
                                  missing=float("-inf"))]
        products = [self._Row(parent_obsid=101, productFilename="tic-42-lc.fits",
                              dataURI="mast:TESS/product/42", dataRights="PUBLIC",
                              nested=[float("inf"), {"missing": self._Scalar(float("nan"))}])]
        mast_observations = MagicMock()
        mast_observations.query_criteria.return_value = observations
        mast_observations.get_product_list.return_value = products
        mast = types.ModuleType("astroquery.mast"); mast.Observations = mast_observations
        astroquery = types.ModuleType("astroquery"); astroquery.mast = mast

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"astroquery": astroquery, "astroquery.mast": mast}
        ):
            provider = MastTessSectorArchiveProvider()
            discovered = provider.inventory_sector(7)
            self.assertIsNone(discovered[0].cadence_seconds)
            store = TessSectorInventoryStore(Path(tmp) / "inventory.json")
            store.create_or_load(7, provider)
            raw = json.loads(store.path.read_text(encoding="utf-8"))

        source = raw["entries"][0]["product"]["source_fields"]
        self.assertIsNone(source["t_exptime"])
        self.assertEqual([None, {"missing": None}], source["nested"])

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

    def test_missing_cadence_sorts_after_finite_cadence_deterministically(self):
        products = [product(1, cadence=None, filename="a-missing-lc.fits"),
                    product(1, cadence=600, filename="finite-lc.fits"),
                    product(2, cadence=None, filename="z-missing-lc.fits"),
                    product(2, cadence=None, filename="a-missing-lc.fits")]
        with tempfile.TemporaryDirectory() as tmp:
            inv = TessSectorInventoryStore(Path(tmp)/"i.json").create_or_load(
                7, FakeProvider(products))

        self.assertEqual("finite-lc.fits", inv.entries[0].product.product_filename)
        self.assertEqual("a-missing-lc.fits", inv.entries[1].product.product_filename)


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
    def _assert_legacy_state_rejected(self, legacy_name):
        provider, coordinator = FakeProvider([product(1)]), FakeCoordinator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path = root / legacy_name
            original = b"legacy-state\x00must-remain-unchanged\n"
            legacy_path.write_bytes(original)

            with self.assertRaisesRegex(
                RuntimeError, "refuses legacy single-lifecycle state"
            ):
                run_tess_sector_sweep(
                    7, "unused", root, provider=provider, coordinator=coordinator
                )

            self.assertEqual(original, legacy_path.read_bytes())
            self.assertEqual([], provider.calls)
            self.assertEqual([], provider.downloads)
            self.assertEqual([], coordinator.calls)
            self.assertFalse(any(root.glob("tess-sector-*-inventory.json")))
            self.assertFalse((root / "investigations").exists())
            self.assertEqual([legacy_name], [path.name for path in root.iterdir()])

    def test_lifecycle_state_directory_is_rejected_before_writes(self):
        self._assert_legacy_state_rejected("lifecycle.json")

    def test_portfolio_state_directory_is_rejected_before_writes(self):
        self._assert_legacy_state_rejected("portfolio.json")

    def test_fresh_sector_sweep_state_directory_still_runs(self):
        provider, coordinator = FakeProvider([product(1)]), FakeCoordinator()
        with tempfile.TemporaryDirectory() as tmp:
            from workflows.tess import tess_sector_scan
            original = tess_sector_scan.read_and_prepare_tess_light_curve
            tess_sector_scan.read_and_prepare_tess_light_curve = lambda path: Prepared()
            try:
                self.assertEqual(
                    0,
                    run_tess_sector_sweep(
                        7, "unused", tmp, provider=provider, coordinator=coordinator
                    ),
                )
            finally:
                tess_sector_scan.read_and_prepare_tess_light_curve = original
            self.assertEqual([7], provider.calls)
            self.assertTrue((Path(tmp) / "tess-sector-7-inventory.json").exists())
            self.assertEqual([1], provider.downloads)
            self.assertEqual([1], coordinator.calls)

    def _partial_scheduler(self, root, provider, coordinator, *, chained):
        inventory = TessSectorInventoryStore(root / "inventory.json").create_or_load(7, provider)
        store = InvestigationStore(root / "investigations")
        workflow = register_tess_sector_scan_handlers(
            store, coordinator, provider, preprocessing=lambda path: Prepared()
        )
        workflow.chain_stages = chained
        scheduler = InvestigationScheduler(
            store, InvestigationDispatcher(store, workflow),
            TessSectorScanTargetSource(inventory),
            {WORKFLOW_ID: plan_tess_sector_scan}, software_id="test-sector-sweep",
            software_version="1", max_concurrent_investigations=1,
        )
        return store, scheduler

    def _assert_resumed_terminal_scan(self, investigation, provider, coordinator):
        self.assertEqual("COMPLETE", investigation.status)
        self.assertEqual(
            ["001-materialize-light-curve", "002-broad-distributed-scan",
             "003-persist-scan-evidence"],
            [stage.id for stage in investigation.stages],
        )
        self.assertTrue(all(stage.status == "COMPLETE" for stage in investigation.stages))
        self.assertEqual(1, provider.downloads.count(1))
        self.assertEqual(1, coordinator.calls.count(1))
        scan = investigation.stages[1]
        self.assertEqual(("project-1",), scan.provenance.project_ids)
        self.assertEqual({"iphone": 1, "mac": 2}, scan.provenance.node_contributions)
        evidence = investigation.stages[2]
        self.assertTrue(evidence.stop)
        self.assertEqual(["project-1"], evidence.result["computeProjectIDs"])
        self.assertEqual({"iphone": 1, "mac": 2}, evidence.result["nodeContributions"])
        self.assertTrue(Path(evidence.artifacts[0].path).exists())

    def test_restart_after_materialization_resumes_persisted_002(self):
        provider, coordinator = FakeProvider([product(1)]), FakeCoordinator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, before_crash = self._partial_scheduler(
                root, provider, coordinator, chained=False
            )
            first = before_crash.run_round().outcomes[0].investigation
            self.assertEqual("RUNNING", first.status)
            self.assertEqual(["001-materialize-light-curve"], [s.id for s in first.stages])
            self.assertEqual("002-broad-distributed-scan", first.stages[0].next_stage["id"])
            self.assertEqual([1], provider.downloads)
            self.assertEqual([], coordinator.calls)

            store, restarted = self._partial_scheduler(
                root, provider, coordinator, chained=True
            )
            restarted.run_until_idle()
            self._assert_resumed_terminal_scan(
                store.load("tess-sector-scan-7-tic-1"), provider, coordinator
            )

    def test_restart_after_compute_resumes_persisted_003(self):
        provider, coordinator = FakeProvider([product(1)]), FakeCoordinator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store, before_crash = self._partial_scheduler(
                root, provider, coordinator, chained=False
            )
            before_crash.run_round()
            second = before_crash.run_round().outcomes[0].investigation
            self.assertEqual("RUNNING", second.status)
            self.assertEqual(
                ["001-materialize-light-curve", "002-broad-distributed-scan"],
                [stage.id for stage in second.stages],
            )
            self.assertEqual("003-persist-scan-evidence", second.stages[1].next_stage["id"])
            self.assertEqual([1], provider.downloads)
            self.assertEqual([1], coordinator.calls)

            store, restarted = self._partial_scheduler(
                root, provider, coordinator, chained=True
            )
            restarted.run_until_idle()
            self._assert_resumed_terminal_scan(
                store.load("tess-sector-scan-7-tic-1"), provider, coordinator
            )

    def test_concurrent_shallow_evidence_resume_and_runtime_limit(self):
        provider = FakeProvider([product(3), product(1), product(2)])
        coordinator = FakeCoordinator(expected_concurrent=2)
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
