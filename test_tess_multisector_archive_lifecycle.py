import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.table import MaskedColumn, Table

from openstar_investigation import InvestigationStage, InvestigationStore
from workflows.tess.tess_autonomy import WORKFLOW_ID, repair_obsolete_terminal_wait
from workflows.tess.tess_multisector import (
    TessArchiveInfrastructureError,
    TessSectorUnavailableError,
    _MAST_LIGHTKURVE_LOCK,
    _exptime_seconds,
    _sector_from_search_row,
    _select_product_from_search,
    build_independent_sector_project,
)


class _SearchResult:
    def __init__(self, table, selected_indices=None):
        self.table = table
        self.selected_indices = selected_indices

    def __getitem__(self, item):
        indices = list(range(len(self.table)))[item]
        if isinstance(indices, int):
            indices = [indices]
        return _SearchResult(self.table[item], indices)


class TessMaskedArchiveMetadataTests(unittest.TestCase):
    def test_masked_sequence_number_uses_mission_fallback(self):
        table = Table()
        table["sequence_number"] = MaskedColumn([1], mask=[True])
        table["mission"] = ["TESS Sector 0105"]
        self.assertEqual(105, _sector_from_search_row(table, 0))

    def test_masked_sequence_number_without_fallback_is_unavailable(self):
        table = Table()
        table["sequence_number"] = MaskedColumn([1], mask=[True])
        table["mission"] = MaskedColumn(["TESS Sector 1"], mask=[True])
        self.assertIsNone(_sector_from_search_row(table, 0))

    def test_masked_exptime_is_none_not_nan(self):
        value = MaskedColumn([120.0], mask=[True], unit=u.s)[0]
        self.assertTrue(np.ma.is_masked(value))
        self.assertIsNone(_exptime_seconds(value))

    def test_masked_irrelevant_rows_do_not_hide_deterministic_valid_product(self):
        table = Table()
        table["sequence_number"] = MaskedColumn(
            [1, 105, 105, 105, 105], mask=[True, False, False, False, False]
        )
        table["mission"] = MaskedColumn(
            ["irrelevant", "TESS Sector 105", "TESS Sector 105",
             "TESS Sector 105", "TESS Sector 105"],
            mask=[True, False, False, False, False],
        )
        table["author"] = ["QLP", "TESS-SPOC", "SPOC", "SPOC", "SPOC"]
        table["exptime"] = MaskedColumn(
            [600.0, 20.0, 120.0, 20.0, 20.0],
            mask=[True, False, True, False, False], unit=u.s,
        )

        selected, author, cadence = _select_product_from_search(
            _SearchResult(table), 105
        )

        # SPOC priority wins over TESS-SPOC, a finite cadence wins over the
        # masked cadence, and row index 3 wins the equal-cadence tie.
        self.assertEqual("SPOC", author)
        self.assertEqual(20.0, cadence)
        self.assertEqual([3], selected.selected_indices)


class TessArchiveLifecycleTests(unittest.TestCase):
    def _build(self, root, investigation_id, candidate_sectors=None):
        project = root / "source.json"
        if not project.exists():
            project.write_text(json.dumps({"id": "source", "name": "source", "workloadID": "ls",
                                           "datasets": []}))
        return build_independent_sector_project(
            source_project_path=project,
            source_dataset_entry={"id": "target", "targetName": "target"},
            tic_id=1, primary_sector=1, target_period_days=2.0,
            candidate_sectors=candidate_sectors or [2], output_dir=root / investigation_id,
            investigation_id=investigation_id,
        )

    def test_concurrent_archive_objects_are_serialized_until_materialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = 0
            maximum = 0
            guard = threading.Lock()

            def search(_):
                nonlocal active, maximum
                with guard:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.03)
                return object()

            def prepare(_):
                nonlocal active
                time.sleep(0.03)
                with guard:
                    active -= 1
                values = np.arange(40, dtype=np.float32)
                return values, values, {"originalSamples": 40, "distributedSamples": 40,
                    "originalTimeOriginDays": 0.0, "sourceFluxMean": 0.0,
                    "sourceFluxStddev": 1.0, "baselineDays": 39.0}

            with patch("workflows.tess.tess_multisector._search_lightcurves", side_effect=search), \
                 patch("workflows.tess.tess_multisector._select_product_from_search",
                       return_value=(object(), "SPOC", 120.0)), \
                 patch("workflows.tess.tess_multisector._download_selected_sector",
                       return_value=(object(), {"author": "SPOC", "cadenceSeconds": 120.0})), \
                 patch("workflows.tess.tess_multisector._prepare_samples", side_effect=prepare):
                threads = [threading.Thread(target=self._build, args=(root, f"i{x}")) for x in range(2)]
                for thread in threads: thread.start()
                for thread in threads: thread.join()
            self.assertEqual(1, maximum)

    def test_archive_lock_does_not_serialize_coordinator_compute(self):
        ran = threading.Event()
        with _MAST_LIGHTKURVE_LOCK:
            thread = threading.Thread(target=ran.set)
            thread.start()
            self.assertTrue(ran.wait(0.2))
            thread.join()

    def test_closed_file_during_materialization_has_structured_transient_context(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch("workflows.tess.tess_multisector._search_lightcurves", return_value=object()), \
             patch("workflows.tess.tess_multisector._select_product_from_search",
                   return_value=(object(), "SPOC", 120.0)), \
             patch("workflows.tess.tess_multisector._download_selected_sector",
                   return_value=(object(), {})), \
             patch("workflows.tess.tess_multisector._prepare_samples",
                   side_effect=ValueError("I/O operation on closed file.")):
            with self.assertRaises(TessArchiveInfrastructureError) as caught:
                self._build(Path(temporary), "failed")
            self.assertEqual("archive-materialization", caught.exception.diagnostics["errors"][0]["operation"])

    def test_typed_unavailable_sector_is_recorded_while_valid_sector_proceeds(self):
        values = np.arange(40, dtype=np.float32)
        prep = {
            "originalSamples": 40, "distributedSamples": 40,
            "originalTimeOriginDays": 0.0, "sourceFluxMean": 0.0,
            "sourceFluxStddev": 1.0, "baselineDays": 39.0,
        }

        def select(_search, sector):
            if sector == 3:
                raise TessSectorUnavailableError("no usable Sector 3 product")
            return object(), "SPOC", 120.0

        with tempfile.TemporaryDirectory() as temporary, \
             patch("workflows.tess.tess_multisector._search_lightcurves", return_value=object()), \
             patch("workflows.tess.tess_multisector._select_product_from_search",
                   side_effect=select), \
             patch("workflows.tess.tess_multisector._download_selected_sector",
                   return_value=(object(), {})), \
             patch("workflows.tess.tess_multisector._prepare_samples",
                   return_value=(values, values, prep)):
            result = self._build(Path(temporary), "partial", [2, 3])

        self.assertTrue(result["available"])
        self.assertEqual([2], [item["sector"] for item in result["preparedSectors"]])
        self.assertEqual(3, result["errors"][0]["sector"])
        self.assertIn("TessSectorUnavailableError", result["errors"][0]["error"])

    def test_plain_runtime_error_fails_preparation(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch("workflows.tess.tess_multisector._search_lightcurves", return_value=object()), \
             patch("workflows.tess.tess_multisector._select_product_from_search",
                   side_effect=RuntimeError("broken internal invariant")):
            with self.assertRaisesRegex(RuntimeError, "broken internal invariant"):
                self._build(Path(temporary), "failed")


class TessClosedFileCompatibilityTests(unittest.TestCase):
    def _failed(self, root, *, error="ValueError: I/O operation on closed file.", handler="openstar.tess.independent.prepare"):
        store = InvestigationStore(root)
        investigation = store.create("target", WORKFLOW_ID, "20.2", metadata={"controlState": {
            "schedulerAction": "INVESTIGATION_FAILED"}})
        stages = tuple(
            InvestigationStage(f"{index:03d}-{name}", handler_id, "COMPLETE", None, {})
            for index, (name, handler_id) in enumerate((
                ("prepare-target", "openstar.tess.prepare-target"),
                ("primary-distributed-search", "openstar.tess.primary-project.run"),
                ("catalog-identity", "openstar.tess.catalog-identity"),
                ("hypotheses", "openstar.tess.hypotheses"),
                ("planner", "openstar.tess.planner")), 1)
        ) + (InvestigationStage("006-prepare-independent-sectors", handler, "FAILED", "005-planner", {},
                                error=error, failure_classification="NON_RETRYABLE"),)
        investigation = replace(investigation, status="FAILED", stages=stages)
        store.save(investigation)
        return store, investigation

    def test_repair_preserves_failed_attempt_is_idempotent_and_skips_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._failed(temporary)
            repaired = repair_obsolete_terminal_wait(store, investigation)
            self.assertEqual("RUNNING", repaired.status)
            self.assertEqual(6, len(repaired.stages))
            retry = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("007-prepare-independent-sectors", retry["id"])
            self.assertEqual("openstar.tess.independent.prepare", retry["handler_id"])
            self.assertEqual("006-prepare-independent-sectors", retry["triggered_by_stage_id"])
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))
            self.assertEqual(1, sum(s.handler_id == "openstar.tess.primary-project.run" for s in repaired.stages))

    def test_unrelated_failures_are_not_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation = self._failed(temporary, error="ValueError: unrelated")
            self.assertEqual(investigation, repair_obsolete_terminal_wait(store, investigation))
            store, investigation = self._failed(Path(temporary) / "other", handler="other.handler")
            self.assertEqual(investigation, repair_obsolete_terminal_wait(store, investigation))


if __name__ == "__main__":
    unittest.main()
