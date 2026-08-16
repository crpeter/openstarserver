import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTarget
from openstar_workflow import StageOutcome, StageRequest, WorkflowEngine
from workflows.tess.tess_catalog_counterpart import identify_catalog_counterparts
from workflows.tess.tess_offset_source import _coordinate_separation_arcsec

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")

from workflows.tess.tess_investigation import prf_catalog_counterpart_continuation
from workflows.tess.tess_autonomy import plan_tess_branches


class CatalogCounterpartTest(unittest.TestCase):
    def setUp(self):
        self.preparation = {
            "targetSky": {"raDeg": 100.0, "decDeg": -30.0},
            "offset": {"componentID": "offset-synthetic",
                       "initialGeometry": {"eastArcsec": 30.0, "northArcsec": 0.0,
                                           "supportingSectors": [10, 11]}},
        }
        self.prf = {"classification": "PRF_SOURCE_SWITCHING",
                    "recommendedNextTest": "CATALOG_COUNTERPART_IDENTIFICATION",
                    "physicalMechanismResolved": False}

    def test_coordinate_separation_does_not_import_astropy(self):
        real_import = __import__

        def without_astropy(name, *args, **kwargs):
            if name == "astropy" or name.startswith("astropy."):
                raise ModuleNotFoundError("Astropy deliberately unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=without_astropy):
            separation = _coordinate_separation_arcsec(10.0, 0.0, 10.0 + 1.0 / 3600.0, 0.0)
        self.assertAlmostEqual(1.0, separation, places=8)

    @staticmethod
    def _tic(_coordinate, _target):
        return {"found": True, "sources": [
            {"catalog": "TIC", "ticID": 1, "isTargetTIC": True,
             "gaiaSourceID": 101, "raDeg": 100.0, "decDeg": -30.0,
             "separationArcsec": 30.0, "tmag": 8.0},
            {"catalog": "TIC", "ticID": 2, "isTargetTIC": False,
             "gaiaSourceID": 202, "raDeg": 100.00955, "decDeg": -30.0,
             "separationArcsec": 0.22, "tmag": 13.1},
            {"catalog": "TIC", "ticID": 3, "isTargetTIC": False,
             "gaiaSourceID": 303, "raDeg": 100.013, "decDeg": -30.0,
             "separationArcsec": 11.0, "tmag": 15.0},
        ]}

    @staticmethod
    def _gaia(_coordinate):
        return {"found": True, "sources": [
            {"catalog": "GaiaDR3", "gaiaSourceID": 101, "raDeg": 100.0,
             "decDeg": -30.0, "separationArcsec": 30.0, "gMag": 8.2},
            {"catalog": "GaiaDR3", "gaiaSourceID": 202, "raDeg": 100.00955,
             "decDeg": -30.0, "separationArcsec": 0.22, "gMag": 13.4},
            {"catalog": "GaiaDR3", "gaiaSourceID": 303, "raDeg": 100.013,
             "decDeg": -30.0, "separationArcsec": 11.0, "gMag": 15.3},
        ]}

    def test_frozen_catalog_ranking_is_conservative_and_replayable(self):
        result = identify_catalog_counterparts(
            tic_id=1, preparation=self.preparation, prf_summary=self.prf,
            query_tic=self._tic, query_gaia=self._gaia)
        self.assertEqual("PLAUSIBLE_NEARBY_CATALOG_COUNTERPARTS", result["classification"])
        preferred = result["preferredCandidate"]
        self.assertEqual(2, preferred["catalogIDs"]["ticID"])
        self.assertEqual(202, preferred["catalogIDs"]["gaiaDR3SourceID"])
        self.assertFalse(preferred["isTarget"])
        self.assertFalse(preferred["variabilityConfirmed"])
        self.assertEqual([10, 11], preferred["motivatingSectors"])
        self.assertGreater(preferred["targetSeparationArcsec"], 20.0)

        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create("catalog-replay", "test", "1")
            engine = WorkflowEngine(store)
            engine.register_handler("catalog", lambda _i, _r: StageOutcome(result, stop=True))
            investigation = engine.run(investigation, StageRequest("001", "catalog", {}),
                                       software_id="test", software_version="1")
            replay = store.load(investigation.id)
            self.assertEqual(result, replay.stages[-1].result)
            self.assertEqual(202, replay.stages[-1].result["preferredCandidate"]
                             ["catalogIDs"]["gaiaDR3SourceID"])

    def test_catalog_failure_is_structured_external_unavailability(self):
        failed = lambda *_args: {"found": False, "sources": [],
                                 "queryError": "frozen service outage"}
        result = identify_catalog_counterparts(
            tic_id=1, preparation=self.preparation, prf_summary=self.prf,
            query_tic=failed, query_gaia=failed)
        self.assertEqual("EXTERNAL_CATALOG_DATA_UNAVAILABLE", result["classification"])
        self.assertEqual(2, len(result["queryErrors"]))
        self.assertEqual("BLOCKED_EXTERNAL_DATA", result["externalDataState"])
        self.assertEqual("RETRY_CATALOG_COUNTERPART_IDENTIFICATION",
                         result["recommendedNextTest"])

    def test_exact_prf_continuation_guard(self):
        request = prf_catalog_counterpart_continuation(self.prf, request_id="020")
        self.assertEqual("openstar.tess.catalog-counterpart-identification.analyze",
                         request.handler_id)
        resolved = dict(self.prf, physicalMechanismResolved=True)
        self.assertEqual("openstar.tess.finalize",
                         prf_catalog_counterpart_continuation(
                             resolved, request_id="020").handler_id)

    def test_old_finalize_continues_catalog_once_then_remains_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create(
                "historical-finalize", "openstar.workflow.tess-investigation.v1", "20.2")
            prf = InvestigationStage(
                "035-prf-interpret", "openstar.tess.official-spoc-prf-forward-modeling.interpret",
                "COMPLETE", None, {}, result=self.prf)
            old_finalize = InvestigationStage(
                "036-finalize", "openstar.tess.finalize", "COMPLETE", None, {}, result={})
            investigation = replace(investigation, status="COMPLETE",
                                    stages=(prf, old_finalize))
            target = InvestigationTarget(
                "synthetic", investigation.id, investigation.workflow_id,
                investigation.workflow_version)

            branches = plan_tess_branches(investigation, target)
            self.assertEqual(1, len(branches))
            self.assertEqual("openstar.tess.catalog-counterpart-identification.analyze",
                             branches[0].experiment.handler_id)
            self.assertEqual(prf.id, branches[0].experiment.triggered_by_stage_id)

            catalog = InvestigationStage(
                "037-catalog-counterpart",
                "openstar.tess.catalog-counterpart-identification.analyze",
                "COMPLETE", prf.id, {}, result={"classification": "NO_USABLE_CATALOG_CANDIDATES"})
            new_finalize = InvestigationStage(
                "038-finalize", "openstar.tess.finalize", "COMPLETE", catalog.id, {}, result={})
            restarted = replace(investigation, stages=investigation.stages +
                                (catalog, new_finalize))
            self.assertEqual((), plan_tess_branches(restarted, target))


if __name__ == "__main__":
    unittest.main()
