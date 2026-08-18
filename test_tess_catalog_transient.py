import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from openstar_investigation import InvestigationStage, InvestigationStore
from workflows.tess.tess_autonomy import WORKFLOW_ID, repair_obsolete_terminal_wait
from workflows.tess.tess_identity import (
    TRANSIENT_INFRASTRUCTURE,
    classify_query_exception,
    collect_identity,
    transient_required_catalog_failures,
)


class CatalogTransientTests(unittest.TestCase):
    def test_transport_and_retryable_http_failures_are_transient(self):
        ReadTimeout = type("ReadTimeout", (TimeoutError,), {"__module__": "urllib3.exceptions"})
        TransportConnectionError = type(
            "ConnectionError", (Exception,), {"__module__": "httpx"})
        self.assertEqual(TRANSIENT_INFRASTRUCTURE,
                         classify_query_exception(ReadTimeout("late")))
        self.assertEqual(TRANSIENT_INFRASTRUCTURE,
                         classify_query_exception(TransportConnectionError("offline")))
        for status in (408, 425, 429, 500, 503, 599):
            error = RuntimeError("service")
            error.response = Mock(status_code=status)
            self.assertEqual(TRANSIENT_INFRASTRUCTURE,
                             classify_query_exception(error), status)
        error = RuntimeError("not found")
        error.response = Mock(status_code=404)
        self.assertIsNone(classify_query_exception(error))

    @patch("workflows.tess.tess_identity._query_tess_products", return_value={"found": False})
    @patch("workflows.tess.tess_identity._query_gaia_variability")
    @patch("workflows.tess.tess_identity._query_gaia_main")
    @patch("workflows.tess.tess_identity._query_vsx", return_value={"found": False, "matches": []})
    @patch("workflows.tess.tess_identity._query_simbad", return_value={"found": False})
    @patch("workflows.tess.tess_identity._coordinate", return_value=object())
    @patch("workflows.tess.tess_identity._query_tic", return_value={"found": True})
    def test_valid_zero_results_are_not_transient(self, *_mocks):
        # With no Gaia source, variability is not queried; both zero-match
        # required catalogs remain successful queries rather than outages.
        with patch("workflows.tess.tess_identity._query_gaia_main",
                   return_value={"found": False, "sources": []}):
            identity = collect_identity(1)
        self.assertEqual([], transient_required_catalog_failures(identity))

    def test_simbad_only_transient_is_optional(self):
        identity = {
            "tic": {"found": True}, "vsx": {"found": False, "matches": []},
            "gaiaDR3": {"found": False, "sources": []}, "gaiaVariability": {},
            "simbad": {"queryErrorClassification": TRANSIENT_INFRASTRUCTURE},
        }
        self.assertEqual([], transient_required_catalog_failures(identity))

class CatalogRepairTests(unittest.TestCase):
    def _terminal(self, root, *, reason="catalog-coverage-incomplete", transient=True):
        store = InvestigationStore(root)
        inv = store.create("old", WORKFLOW_ID, "20.2")
        identity = {"tic": {"found": True}, "vsx": {}, "gaiaVariability": {},
                    "gaiaDR3": ({"queryError": "ReadTimeout: late"} if transient
                                 else {"queryError": "valid incomplete state"})}
        evidence = (
            ("001-prepare-target", "openstar.tess.prepare-target", {}),
            ("002-primary-distributed-search", "openstar.tess.primary-project.run", {}),
            ("003-catalog-identity", "openstar.tess.catalog-identity", identity),
            ("004-hypotheses", "openstar.tess.hypotheses", {}),
            ("005-planner", "openstar.tess.planner", {"reason": reason}),
            ("006-finalize", "openstar.tess.finalize", {"claim": "HUMAN_REVIEW_REQUIRED"}),
        )
        for sid, handler, result in evidence:
            running = InvestigationStage(sid, handler, "RUNNING", None, {})
            inv = store.append_running_stage(inv, running)
            terminal = store.build_terminal_stage(
                stage_id=sid, handler_id=handler, status="COMPLETE", triggered_by_stage_id=None,
                parameters={}, result=result, error=None, software_id="old", software_version="1",
                started_at=running.started_at, stop=(sid == "006-finalize"))
            inv = store.complete_current_stage(inv, terminal)
        inv = store.set_control_state(inv, status="COMPLETE", control_state={
            "branchAssessments": [], "selectedExperiment": None,
            "schedulerAction": "INVESTIGATION_COMPLETE"})
        return store, inv

    def test_terminal_timeout_repair_is_narrow_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store, inv = self._terminal(directory)
            stage_bytes = {path: path.read_bytes() for path in
                           (Path(directory) / "old/stages").glob("*.json")}
            repaired = repair_obsolete_terminal_wait(store, inv)
            self.assertEqual("RUNNING", repaired.status)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("openstar.tess.catalog-identity", selected["handler_id"])
            self.assertEqual("007-catalog-identity", selected["id"])
            self.assertEqual(stage_bytes, {path: path.read_bytes() for path in stage_bytes})
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))

    def test_nontransient_and_unrelated_terminal_states_are_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            store, inv = self._terminal(Path(directory) / "nontransient", transient=False)
            self.assertEqual(inv, repair_obsolete_terminal_wait(store, inv))
            store, inv = self._terminal(Path(directory) / "unrelated", reason="other")
            self.assertEqual(inv, repair_obsolete_terminal_wait(store, inv))


if __name__ == "__main__":
    unittest.main()
