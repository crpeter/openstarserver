import json
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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
    HAS_NUMPY = hasattr(numpy, "arange")
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")
    HAS_NUMPY = False
    _installed_numpy_stub = True
else:
    _installed_numpy_stub = False

from workflows.tess.tess_offset_variability import (
    MIN_COMPONENT_SAMPLES,
    _catalog_candidate,
    build_offset_source_variability_project,
    interpret_offset_source_variability_project,
)
from workflows.tess.tess_investigation import (
    build_engine,
    catalog_counterpart_variability_continuation,
    prf_catalog_counterpart_continuation,
)
from workflows.tess.tess_autonomy import plan_tess_branches

if _installed_numpy_stub:
    sys.modules.pop("numpy", None)


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

    @staticmethod
    def _new_catalog_result():
        return {
            "classification": "PLAUSIBLE_NEARBY_CATALOG_COUNTERPARTS",
            "preferredCandidate": {
                "raDeg": 100.01, "decDeg": -30.01,
                "motivatingComponentID": "offset-2",
                "catalogIDs": {"ticID": 736900598,
                               "gaiaDR3SourceID": 5284296077579591040},
            },
            "catalogCandidates": [],
            "physicalMechanismResolved": False,
            "recommendedNextTest":
                "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        }

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

    def test_catalog_continuation_requires_justified_preferred_candidate(self):
        request = catalog_counterpart_variability_continuation(
            self._new_catalog_result(), request_id="021")
        self.assertEqual("openstar.tess.offset-source-variability.prepare",
                         request.handler_id)
        ambiguous = dict(self._new_catalog_result(), preferredCandidate=None,
                         classification="AMBIGUOUS_MULTIPLE_CATALOG_COUNTERPARTS")
        localization = catalog_counterpart_variability_continuation(
            dict(ambiguous, plausibleCatalogCandidates=[
                {"raDeg": 1.0, "decDeg": 2.0},
                {"raDeg": 1.1, "decDeg": 2.1},
            ], recommendedNextTest="CATALOG_GUIDED_SOURCE_LOCALIZATION"),
            request_id="044")
        self.assertEqual(
            "openstar.tess.catalog-guided-source-localization.prepare",
            localization.handler_id,
        )
        self.assertEqual("045-prepare-catalog-guided-source-localization", localization.id)
        self.assertEqual(
            self._new_catalog_result()["preferredCandidate"],
            _catalog_candidate(self._new_catalog_result()),
        )

    @unittest.skipUnless(HAS_NUMPY, "NumPy is required for variability preparation")
    def test_new_catalog_contract_prepares_generic_target_and_counterpart_work(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_count = MIN_COMPONENT_SAMPLES + 8
            source_project = root / "source.json"
            source_project.write_text(json.dumps({
                "id": "blind", "name": "Blind", "workloadID": "openstar.lomb-scargle.v1"
            }), encoding="utf-8")
            fake_tpf = types.SimpleNamespace(
                time=types.SimpleNamespace(value=np.arange(sample_count, dtype=float)),
                flux=types.SimpleNamespace(
                    value=np.ones((sample_count, 2, 2), dtype=float)),
            )
            series = {
                "target-control": np.linspace(-1, 1, sample_count),
                "catalog-counterpart": np.linspace(1, -1, sample_count),
            }
            with mock.patch.multiple(
                "workflows.tess.tess_offset_variability",
                _skycoord=mock.DEFAULT, _download_tpf=mock.DEFAULT,
                _background_subtract_cube=mock.DEFAULT, _prewhiten_cube_raw=mock.DEFAULT,
                _catalog_guided_series=mock.DEFAULT,
            ) as patched:
                patched["_skycoord"].side_effect = lambda ra, dec: (ra, dec)
                patched["_download_tpf"].return_value = (
                    fake_tpf, {"sourceType": "TPF", "author": "SPOC", "cadenceSeconds": 120}
                )
                patched["_background_subtract_cube"].return_value = (
                    np.ones((sample_count, 2, 2)), {})
                patched["_prewhiten_cube_raw"].return_value = (
                    np.ones((sample_count, 2, 2)), np.ones((2, 2), dtype=bool))
                patched["_catalog_guided_series"].return_value = (series, {})
                spec = build_offset_source_variability_project(
                    source_project_path=source_project,
                    source_dataset_entry={"id": "blind-c", "targetName": "Blind C"},
                    target_tic_id=1,
                    identity={"tic": {"metadata": {"raDeg": 100.0, "decDeg": -30.0}}},
                    primary_sector=None,
                    independent_spec={"preparedSectors": [{"sector": 1}, {"sector": 2}]},
                    physical_period_days=13.72,
                    nonstationary_summary={
                        "preferredFrequencyAtReference": 0.3,
                        "fractionalFrequencyDriftPerDay": 0.001,
                        "timeReferenceDays": 1000.0,
                        "preferredModel": {"signalSectors": [1, 2]},
                    },
                    multisource_summary={
                        "bestOffsetComponentID": "offset-2",
                        "componentSummaries": [{"componentID": "offset-2",
                                                "combinedFrequency": 0.3}],
                    },
                    offset_source_identification=self._new_catalog_result(),
                    output_dir=root, investigation_id="test",
                )
                historical_prewhiten = patched["_prewhiten_cube_raw"].call_args_list[0]
                unresolved_spec = build_offset_source_variability_project(
                    source_project_path=source_project,
                    source_dataset_entry={"id": "blind-c", "targetName": "Blind C"},
                    target_tic_id=1,
                    identity={"tic": {"metadata": {"raDeg": 100.0, "decDeg": -30.0}}},
                    primary_sector=None, independent_spec={"preparedSectors": []},
                    multisource_summary={"bestOffsetComponentID": "offset-2",
                        "componentSummaries": [{"componentID": "offset-2"}]},
                    offset_source_identification=self._new_catalog_result(),
                    output_dir=root, investigation_id="unresolved",
                    reference_family_period_days=10.30084080080649,
                    harmonic_orders=[1, 2, 3, 4], physical_cycle_resolved=False,
                    residual_reference_frequency=0.3,
                    residual_time_reference_days=1000.0,
                    fractional_frequency_drift_per_day=0.001,
                    frozen_sectors=[1, 2],
                    family_residual_provenance={"bridge": "stage-045"},
                )

            self.assertEqual("openstar.lomb-scargle.v1", spec["workloadID"])
            self.assertEqual((1, 2), historical_prewhiten.kwargs["harmonic_orders"])
            self.assertTrue(spec["physicalCycleResolved"])
            self.assertEqual(10.30084080080649,
                             unresolved_spec["referenceFamilyPeriodDays"])
            self.assertEqual([1, 2, 3, 4], unresolved_spec["subtractedHarmonicOrders"])
            self.assertFalse(unresolved_spec["physicalCycleResolved"])
            self.assertEqual([1, 2, 3, 4],
                             list(patched["_prewhiten_cube_raw"].call_args.kwargs["harmonic_orders"]))
            self.assertEqual(736900598, spec["catalogCounterpart"]["ticID"])
            self.assertEqual(5284296077579591040,
                             spec["catalogCounterpart"]["gaiaDR3SourceID"])
            self.assertEqual(100.01, spec["catalogCounterpart"]["raDeg"])
            roles = {item["componentID"] for item in spec["preparedSeries"]}
            self.assertEqual({"target-control", "catalog-counterpart"}, roles)
            per_sector = [item for item in spec["preparedSeries"] if not item["combined"]]
            self.assertEqual({1, 2}, {item["sector"] for item in per_sector})
            self.assertEqual(
                {(component, sector)
                 for component in ("target-control", "catalog-counterpart")
                 for sector in (1, 2)},
                {(item["componentID"], item["sector"]) for item in per_sector},
            )
            combined = [item for item in spec["preparedSeries"] if item["combined"]]
            self.assertEqual(
                {"target-control", "catalog-counterpart"},
                {item["componentID"] for item in combined},
            )

    def test_new_contract_interpretation_persists_component_evidence(self):
        prepared_series = []
        datasets = []
        for component in ("catalog-counterpart", "target-control"):
            for sector in (1, 2, 3):
                dataset_id = f"{component}-{sector}"
                prepared_series.append({
                    "datasetID": dataset_id, "componentID": component,
                    "sector": sector, "role": "independent", "combined": False,
                    "baselineDays": 30.0,
                })
                reliable = component == "catalog-counterpart"
                datasets.append({
                    "datasetID": dataset_id,
                    "candidateFrequency": 0.3,
                    "candidatePeriodDays": 1 / 0.3,
                    "candidatePower": 0.8 if reliable else 0.02,
                    "candidatePeakProminenceRatio": 2.5,
                    "periodStatus": "RELIABLE", "periodConfidence": "high",
                })
            combined_id = f"{component}-combined"
            prepared_series.append({
                "datasetID": combined_id, "componentID": component,
                "sector": None, "role": "combined", "combined": True,
                "baselineDays": 90.0,
            })
            datasets.append({
                "datasetID": combined_id, "candidateFrequency": 0.3,
                "candidatePeriodDays": 1 / 0.3,
                "candidatePower": 0.9 if component == "catalog-counterpart" else 0.02,
                "candidatePeakProminenceRatio": 3.0,
                "periodStatus": "RELIABLE", "periodConfidence": "high",
            })
        preparation = {
            "workloadID": "openstar.lomb-scargle.v1",
            "catalogCounterpart": self._new_catalog_result()["preferredCandidate"]["catalogIDs"],
            "referenceFrequency": 0.3, "referencePeriodDays": 1 / 0.3,
            "fractionalFrequencyDriftPerDay": 0.001,
            "frequencySearch": {"minimumFrequency": 0.24, "maximumFrequency": 0.36,
                                "frequencyStep": 0.0001},
            "preparedSeries": prepared_series,
        }
        result = interpret_offset_source_variability_project(
            project_status={"datasets": datasets}, preparation=preparation)
        self.assertEqual("OFFSET_COUNTERPART_VARIABILITY_SUPPORTED",
                         result["classification"])
        self.assertTrue(result["variabilityConfirmed"])
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertEqual(3, result["catalogCounterpartEvidence"]
                         ["independentSupportCount"])
        self.assertEqual(0, result["targetControl"]["independentSupportCount"])
        self.assertEqual(3, len(result["counterpartPerSectorResults"]))
        self.assertEqual(3, len(result["targetControlPerSectorResults"]))
        self.assertFalse(result["provenance"]["independentTelescopeEvidence"])

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
            self.assertEqual("036-catalog-counterpart", branches[0].experiment.id)
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

    def test_restart_after_catalog_finalize_appends_variability_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create(
                "catalog-restart", "openstar.workflow.tess-investigation.v1", "20.2")
            catalog = InvestigationStage(
                "037-catalog-counterpart",
                "openstar.tess.catalog-counterpart-identification.analyze",
                "COMPLETE", None, {}, result=self._new_catalog_result())
            old_finalize = InvestigationStage(
                "038-finalize", "openstar.tess.finalize", "COMPLETE",
                catalog.id, {}, result={}, stop=True)
            investigation = replace(investigation, status="COMPLETE",
                                    stages=(catalog, old_finalize))
            target = InvestigationTarget(
                "synthetic", investigation.id, investigation.workflow_id,
                investigation.workflow_version)

            branches = plan_tess_branches(investigation, target)
            self.assertEqual(1, len(branches))
            self.assertEqual("openstar.tess.offset-source-variability.prepare",
                             branches[0].experiment.handler_id)
            self.assertEqual("038-prepare-offset-source-variability",
                             branches[0].experiment.id)
            self.assertEqual(catalog.id, branches[0].experiment.triggered_by_stage_id)

            completed = InvestigationStage(
                "041-interpret-offset-source-variability",
                "openstar.tess.offset-source-variability.interpret",
                "COMPLETE", None, {}, result={"classification": "UNRESOLVED"})
            after_validation = replace(
                investigation, stages=investigation.stages + (completed,))
            self.assertEqual((), plan_tess_branches(after_validation, target))

    def test_finalize_reports_persisted_catalog_evidence_after_prf(self):
        catalog = {
            "classification": "PLAUSIBLE_NEARBY_CATALOG_COUNTERPARTS",
            "preferredCandidate": {
                "catalogIDs": {
                    "ticID": 736900598,
                    "gaiaDR3SourceID": 5284296077579591040,
                },
                "rankingEvidence": {
                    "residualPositionSeparationArcsec": 2.4418044547184197,
                    "targetSeparationArcsec": 32.4663996776444,
                },
            },
            "variabilityConfirmed": False,
            "recommendedNextTest": (
                "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
            ),
        }
        variability = {
            "classification": "OFFSET_COUNTERPART_VARIABILITY_UNRESOLVED",
            "catalogCounterpart": {"ticID": 736900598,
                                   "gaiaDR3SourceID": 5284296077579591040},
            "catalogCounterpartEvidence": {"independentSupportCount": 1},
            "targetControl": {"independentSupportCount": 0},
            "variabilityConfirmed": False,
            "physicalMechanismResolved": False,
            "recommendedNextTest": "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY",
        }
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create("catalog-finalize", "test", "1")
            stages = (
                InvestigationStage(
                    "001-prepare-target", "openstar.tess.prepare-target", "COMPLETE",
                    None, {}, result={"datasetID": "blind-c", "ticID": 1},
                ),
                InvestigationStage(
                    "004-hypotheses", "openstar.tess.hypotheses", "COMPLETE",
                    None, {}, result={"observedPeriodDays": 1.0},
                ),
                InvestigationStage(
                    "005-planner", "openstar.tess.planner", "COMPLETE", None, {},
                    result={"claimDecision": {"claim": "CANDIDATE_PERIOD",
                                               "rationale": ["frozen test"]}},
                ),
                InvestigationStage(
                    "035-prf-interpret",
                    "openstar.tess.official-spoc-prf-forward-modeling.interpret",
                    "COMPLETE", None, {}, result=self.prf,
                ),
                InvestigationStage(
                    "036-catalog-counterpart",
                    "openstar.tess.catalog-counterpart-identification.analyze",
                    "COMPLETE", "035-prf-interpret", {}, result=catalog,
                ),
                InvestigationStage(
                    "039-interpret-offset-source-variability",
                    "openstar.tess.offset-source-variability.interpret",
                    "COMPLETE", "038-run-offset-source-variability", {},
                    result=variability,
                ),
            )
            investigation = replace(investigation, stages=stages)
            store.save(investigation)
            engine = build_engine(
                store, coordinator=types.SimpleNamespace(), poll_interval=0.0, timeout=None
            )
            engine.chain_stages = False
            output = io.StringIO()
            with redirect_stdout(output):
                completed, _ = engine.run_stage(
                    investigation,
                    StageRequest("037-finalize", "openstar.tess.finalize", {}),
                    software_id="test", software_version="1",
                )

            conclusion = completed.stages[-1].result
            report = Path(conclusion["reportPath"]).read_text(encoding="utf-8")
            self.assertEqual(catalog, conclusion["catalogCounterpartIdentification"])
            self.assertEqual(variability["recommendedNextTest"],
                             conclusion["recommendedNextTest"])
            self.assertEqual(variability,
                             conclusion["offsetSourceVariabilityValidation"])
            self.assertIn("Variability confirmed: False", report)
            for expected in (
                "Catalog counterpart classification: PLAUSIBLE_NEARBY_CATALOG_COUNTERPARTS",
                "Counterpart TIC: 736900598",
                "Counterpart Gaia DR3: 5284296077579591040",
                "Residual-position separation: 2.4418044547184197 arcsec",
                "Target-to-counterpart separation: 32.4663996776444 arcsec",
                "Variability confirmed: False",
                "Recommended next test: INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
            ):
                self.assertIn(expected, report)
                self.assertIn(expected.lower(), output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
