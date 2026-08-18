import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openstar_investigation import InvestigationStage, InvestigationStore
from workflows.tess.tess_autonomy import WORKFLOW_ID, repair_obsolete_terminal_wait
from workflows.tess.tess_multisector import (
    TessArchiveInfrastructureError,
    _MAST_LIGHTKURVE_LOCK,
    build_independent_sector_project,
)


class TessArchiveLifecycleTests(unittest.TestCase):
    def _build(self, root, investigation_id):
        project = root / "source.json"
        if not project.exists():
            project.write_text(json.dumps({"id": "source", "name": "source", "workloadID": "ls",
                                           "datasets": []}))
        return build_independent_sector_project(
            source_project_path=project,
            source_dataset_entry={"id": "target", "targetName": "target"},
            tic_id=1, primary_sector=1, target_period_days=2.0,
            candidate_sectors=[2], output_dir=root / investigation_id,
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
