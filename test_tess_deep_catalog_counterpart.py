import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTarget
from workflows.tess.tess_autonomy import (
    plan_tess_branches,
    repair_obsolete_terminal_wait,
)
from workflows.tess.tess_deep_catalog_counterpart import (
    HANDLER_ID,
    identify_deep_catalog_counterparts,
    validate_catalog_boundary,
)
from workflows.tess.tess_investigation import deep_catalog_counterpart_continuation


class DeepCatalogCounterpartTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "version": "openstar.tess-catalog-counterpart-identification.v1",
            "classification": "NO_USABLE_CATALOG_CANDIDATES",
            "counterpartIdentified": False,
            "preferredCandidate": None,
            "plausibleCatalogCandidates": [],
            "physicalMechanismResolved": False,
            "claimLevelChanged": False,
            "recommendedNextTest": "DEEPER_CATALOG_OR_HIGH_RESOLUTION_IMAGING",
            "searchPosition": {
                "componentID": "offset-1",
                "raDeg": 100.0,
                "decDeg": -30.0,
                "targetRaDeg": 99.98,
                "targetDecDeg": -30.0,
                "supportingSectors": [2, 68, 69],
            },
        }

    @staticmethod
    def _deep_result():
        return {
            "version": "openstar.tess-deep-catalog-counterpart-identification.v1",
            "classification": "AMBIGUOUS_DEEP_CATALOG_COUNTERPARTS",
            "counterpartIdentified": False,
            "preferredCandidate": None,
            "plausibleCatalogCandidates": [{"catalogIDs": {"nscDR2ObjectID": "1"}}],
            "variabilityConfirmed": False,
            "physicalMechanismResolved": False,
            "claimLevelChanged": False,
            "recommendedNextTest": "HIGH_RESOLUTION_RESIDUAL_SOURCE_LOCALIZATION",
        }

    @staticmethod
    def _skymapper(_position):
        return [{
            "object_id": "42", "raj2000": "100.00010", "dej2000": "-30.00000",
            "flags": "0", "nimaflags": "0", "flags_psf": "0", "ngood": "25",
            "g_ngood": "8", "r_ngood": "7", "i_ngood": "6", "z_ngood": "4",
            "gaia_dr3_id1": "", "gaia_dr3_dist1": "",
        }]

    @staticmethod
    def _nsc(_position):
        return [{
            "id": "nsc-7", "ra": "100.00011", "dec": "-30.00001",
            "ndet": "31", "class_star": "0.96", "flags": "0",
            "gmag": "20.1", "rmag": "19.8", "imag": "19.7", "zmag": "19.6",
        }]

    def test_cross_catalog_source_is_merged_and_ranked(self):
        result = identify_deep_catalog_counterparts(
            catalog_summary=self.catalog,
            query_skymapper=self._skymapper,
            query_nsc=self._nsc,
        )
        self.assertEqual("DEEP_CATALOG_COUNTERPART_IDENTIFIED", result["classification"])
        self.assertEqual(1, len(result["plausibleCatalogCandidates"]))
        preferred = result["preferredCandidate"]
        self.assertEqual(42, preferred["catalogIDs"]["skyMapperDR4ObjectID"])
        self.assertEqual("nsc-7", preferred["catalogIDs"]["nscDR2ObjectID"])
        self.assertEqual(2, preferred["rankingEvidence"]["catalogCount"])
        self.assertFalse(result["variabilityConfirmed"])
        self.assertEqual(
            "DEEP_CATALOG_GUIDED_SOURCE_LOCALIZATION", result["recommendedNextTest"])

    def test_empty_successful_queries_route_to_dedicated_imaging(self):
        result = identify_deep_catalog_counterparts(
            catalog_summary=self.catalog,
            query_skymapper=lambda _position: [],
            query_nsc=lambda _position: [],
        )
        self.assertEqual("NO_DEEP_CATALOG_COUNTERPART", result["classification"])
        self.assertIsNone(result["preferredCandidate"])
        self.assertEqual("DEDICATED_HIGH_RESOLUTION_IMAGING",
                         result["recommendedNextTest"])
        self.assertEqual("AVAILABLE", result["externalDataState"])

    def test_any_catalog_failure_does_not_become_a_no_source_result(self):
        def unavailable(_position):
            raise TimeoutError("frozen outage")

        result = identify_deep_catalog_counterparts(
            catalog_summary=self.catalog,
            query_skymapper=unavailable,
            query_nsc=lambda _position: [],
        )
        self.assertEqual("EXTERNAL_DEEP_CATALOG_DATA_UNAVAILABLE",
                         result["classification"])
        self.assertEqual("BLOCKED_EXTERNAL_DATA", result["externalDataState"])
        self.assertEqual("RETRY_DEEP_CATALOG_COUNTERPART_IDENTIFICATION",
                         result["recommendedNextTest"])

    def test_boundary_is_fail_closed(self):
        validate_catalog_boundary(self.catalog)
        for key, value in (
            ("classification", "TARGET_CONSISTENT_ONLY"),
            ("counterpartIdentified", True),
            ("physicalMechanismResolved", True),
        ):
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    validate_catalog_boundary(dict(self.catalog, **{key: value}))

    def test_result_always_finalizes_before_any_downstream_science(self):
        request = deep_catalog_counterpart_continuation({}, request_id="020-deep")
        self.assertEqual("openstar.tess.finalize", request.handler_id)
        self.assertEqual("021-finalize", request.id)
        self.assertEqual({"outputSuffix": "deep-catalog-counterpart"}, request.parameters)

    def test_exact_historical_finalize_is_repaired_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create(
                "deep-catalog-repair", "openstar.workflow.tess-investigation.v1", "20.2")
            catalog = InvestigationStage(
                "020-catalog-counterpart",
                "openstar.tess.catalog-counterpart-identification.analyze",
                "COMPLETE", "019-prf", {}, result=self.catalog,
            )
            conclusion = {
                "catalogCounterpartIdentification": self.catalog,
                "recommendedNextTest": "DEEPER_CATALOG_OR_HIGH_RESOLUTION_IMAGING",
            }
            final = InvestigationStage(
                "021-finalize", "openstar.tess.finalize", "COMPLETE",
                catalog.id, {"outputSuffix": "catalog-counterpart"},
                result=conclusion, stop=True,
            )
            metadata = dict(investigation.metadata)
            metadata["controlState"] = {
                "schedulerAction": "INVESTIGATION_COMPLETE",
                "selectedExperiment": None,
            }
            investigation = replace(
                investigation, status="COMPLETE", stages=(catalog, final), metadata=metadata)

            repaired = repair_obsolete_terminal_wait(store, investigation)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("RUNNING", repaired.status)
            self.assertEqual(HANDLER_ID, selected["handler_id"])
            self.assertEqual("022-deep-catalog-counterpart", selected["id"])
            self.assertEqual(catalog.id, selected["triggered_by_stage_id"])
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))

    def test_autonomy_plans_exact_boundary_and_stops_after_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            investigation = InvestigationStore(Path(directory)).create(
                "deep-catalog-plan", "openstar.workflow.tess-investigation.v1", "20.2")
            catalog = InvestigationStage(
                "020-catalog-counterpart",
                "openstar.tess.catalog-counterpart-identification.analyze",
                "COMPLETE", None, {}, result=self.catalog,
            )
            final = InvestigationStage(
                "021-finalize", "openstar.tess.finalize", "COMPLETE", catalog.id,
                {"outputSuffix": "catalog-counterpart"}, result={}, stop=True,
            )
            investigation = replace(
                investigation, status="COMPLETE", stages=(catalog, final))
            target = InvestigationTarget(
                "synthetic", investigation.id, investigation.workflow_id,
                investigation.workflow_version)
            branches = plan_tess_branches(investigation, target)
            self.assertEqual(1, len(branches))
            self.assertEqual(HANDLER_ID, branches[0].experiment.handler_id)

            completed = InvestigationStage(
                "022-deep-catalog-counterpart", HANDLER_ID, "COMPLETE", catalog.id, {},
                result=self._deep_result(),
                next_stage={
                    "id": "023-finalize",
                    "handler_id": "openstar.tess.finalize",
                    "parameters": {"outputSuffix": "deep-catalog-counterpart"},
                    "triggered_by_stage_id": "022-deep-catalog-counterpart",
                },
            )
            attempted = replace(
                investigation, stages=investigation.stages + (completed,))
            finalize = plan_tess_branches(attempted, target)
            self.assertEqual(1, len(finalize))
            self.assertEqual("openstar.tess.finalize",
                             finalize[0].experiment.handler_id)

            final = InvestigationStage(
                "023-finalize", "openstar.tess.finalize", "COMPLETE",
                completed.id, {"outputSuffix": "deep-catalog-counterpart"},
                result={}, stop=True,
            )
            finalized = replace(attempted, stages=attempted.stages + (final,))
            self.assertEqual((), plan_tess_branches(finalized, target))

    def test_completed_deep_stage_repairs_finalize_handoff_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create(
                "deep-finalize-repair", "openstar.workflow.tess-investigation.v1", "20.2")
            deep = InvestigationStage(
                "022-deep-catalog-counterpart", HANDLER_ID, "COMPLETE", "020-catalog", {},
                result=self._deep_result(),
                next_stage={
                    "id": "023-finalize",
                    "handler_id": "openstar.tess.finalize",
                    "parameters": {"outputSuffix": "deep-catalog-counterpart"},
                    "triggered_by_stage_id": "022-deep-catalog-counterpart",
                },
            )
            metadata = dict(investigation.metadata)
            metadata["controlState"] = {
                "schedulerAction": "INVESTIGATION_COMPLETE",
                "selectedExperiment": None,
            }
            investigation = replace(
                investigation, status="COMPLETE", stages=(deep,), metadata=metadata)

            repaired = repair_obsolete_terminal_wait(store, investigation)
            selected = repaired.metadata["controlState"]["selectedExperiment"]
            self.assertEqual("RUNNING", repaired.status)
            self.assertEqual("openstar.tess.finalize", selected["handler_id"])
            self.assertEqual("023-finalize", selected["id"])
            self.assertEqual(
                "TESS_DEEP_CATALOG_FINALIZE_HANDOFF",
                repaired.metadata["controlState"]["recovery"],
            )
            self.assertEqual(repaired, repair_obsolete_terminal_wait(store, repaired))


if __name__ == "__main__":
    unittest.main()
