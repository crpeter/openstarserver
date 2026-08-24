import unittest
from unittest import mock
import copy, json, tempfile
from types import SimpleNamespace
from openstar_workflow import StageRequest
from dataclasses import replace
from pathlib import Path
from openstar_investigation import ArtifactReference, InvestigationStage, StageProvenance, sha256_file, sha256_json
from workflows.tess.tess_autonomy import repair_obsolete_terminal_wait
import test_tess_target_residual_pixel_recurrence as v2018_tests

try:
    import numpy as np
    from workflows.tess.tess_catalog_guided_localization import (
        COMPONENT_IDS, MODEL_COMPONENTS, compare_source_hypotheses,
        generate_source_hypotheses)
    NUMPY_AVAILABLE = True
except ModuleNotFoundError:
    np = None; NUMPY_AVAILABLE = False
from workflows.tess.tess_target_residual_multisector_source import (
    MAX_COMPETING_SOURCES, V2018_SECTOR_IDS, derive_additional_sectors,
    derive_competing_sources, interpret_multisector, run_multisector_source_localization,
)


def evidence(sector):
    return {"sector": sector, "candidateFrequency": 0.19 + sector / 10000,
            "originalTimeOriginDays": 1000.0 + sector,
            "supportsHistoricalResidualFamily": True,
            "recurrenceClassification": "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY"}


def sector(sector_id, sources=(), *, available=True, identifiable=True):
    model = "model-" + "-".join(sources)
    return {"sector": sector_id, "availability": "AVAILABLE" if available else "UNAVAILABLE",
        "fullDataComparison": {"bestModel": model, "bestModelSourceIDs": list(sources),
            "bestModelIdentifiable": identifiable, "completeModelFullRank": identifiable,
            "conditionallyIdentifiableSources": list(sources) if identifiable else []},
        "temporalPredictiveValidation": {"predictiveModel": model, "predictiveSupport": True}}


class MultisectorSourceTests(unittest.TestCase):
    STALE = {"branchAssessments": [],
        "recovery": "TESS_V20_17_PIXEL_RECURRENCE_LOCALIZATION_V20_18",
        "schedulerAction": "RUN_EXPERIMENT", "selectedExperiment": {
            "handler_id": "openstar.tess.target-residual-pixel-recurrence.prepare",
            "id": "036-target-residual-pixel-recurrence-prepare", "parameters": {},
            "triggered_by_stage_id": "035-finalize"}}

    def _production_preparation(self, directory):
        return {"artifactRoot": directory, "ticID": 1, "targetSky": {"raDeg": 10., "decDeg": 20.},
            "targetSourceID": "target", "catalogHypotheses": [
                {"sourceID": "target", "raDeg": 10., "decDeg": 20.}],
            "additionalSectorEvidence": [{"sector": 100, "candidateFrequency": .123456789,
                "originalTimeOriginDays": 1000.}], "excludedV2018SectorIDs": list(V2018_SECTOR_IDS),
            "establishedPhysicalFamilyFrequency": .2}

    def _boundary(self, directory, control=None):
        fixture = v2018_tests.TargetResidualPixelRecurrenceTests(methodName="test_unique_target_and_catalog_localizations")
        store, inv = fixture._boundary(directory)
        def artifact(name, value):
            path = Path(directory) / name; path.write_text(json.dumps(value) + "\n")
            return ArtifactReference(str(path), sha256_file(path), "application/json")
        stages = list(inv.stages); v17 = copy.deepcopy(stages[-2].result)
        v17["sectorEvidence"].append(evidence(100))
        v17_path = Path(stages[-2].artifacts[0].path); v17_path.write_text(json.dumps(v17) + "\n")
        stages[-2] = replace(stages[-2], result=v17,
            artifacts=(replace(stages[-2].artifacts[0], sha256=sha256_file(v17_path)),))
        v17_final = {**stages[-1].result, "targetResidualArchivalBaselineExtension": v17}
        final_path = Path(stages[-1].artifacts[0].path); final_path.write_text(json.dumps(v17_final) + "\n")
        stages[-1] = replace(stages[-1], result=v17_final,
            artifacts=(replace(stages[-1].artifacts[0], sha256=sha256_file(final_path)),))
        inv = replace(inv, stages=tuple(stages))
        prep = {"ticID": 1, "targetSourceID": "TIC-1", "targetSky": {"raDeg": 10., "decDeg": 20.},
            "catalogHypotheses": [{"sourceID": "TIC-1", "raDeg": 10., "decDeg": 20.}],
            "selectedSectorEvidence": [{"sector": value} for value in V2018_SECTOR_IDS],
            "frozenEstablishedPhysicalFrequency": .1}
        run = {"sectorResults": [{"sector": 2, "classification": "AMBIGUOUS_OR_BLENDED",
            "distancesPixels": {"TIC-1": .1}}]}
        science = {"classification": "PIXEL_RECURRENCE_LOCALIZATION_UNRESOLVED",
            "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA", "sourceAttributionResolved": False,
            "physicalMechanismResolved": False, "crossSectorPhaseUsed": False,
            "historicalResidualDriftExtrapolated": False}
        s36 = InvestigationStage("036-target-residual-pixel-recurrence-prepare",
            "openstar.tess.target-residual-pixel-recurrence.prepare", "COMPLETE", "035-finalize", {}, result=prep,
            artifacts=(artifact("target-residual-pixel-recurrence-prepare-v20.18.json", prep),),
            provenance=StageProvenance("test", "1", {"v20.17": sha256_json(inv.stages[-2].result)}))
        s37 = InvestigationStage("037-target-residual-pixel-recurrence-run",
            "openstar.tess.target-residual-pixel-recurrence.run", "COMPLETE", s36.id, {}, result=run,
            artifacts=(artifact("target-residual-pixel-recurrence-run-v20.18.json", run),))
        s38 = InvestigationStage("038-target-residual-pixel-recurrence-interpret",
            "openstar.tess.target-residual-pixel-recurrence.interpret", "COMPLETE", s37.id, {}, result=science,
            artifacts=(artifact("target-residual-pixel-recurrence-v20.18.json", science),),
            provenance=StageProvenance("test", "1", {"preparation": sha256_json(prep), "run": sha256_json(run)}))
        conclusion = {"targetResidualPixelRecurrenceValidation": science,
                      "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA"}
        s39 = InvestigationStage("039-finalize", "openstar.tess.finalize", "COMPLETE", s38.id,
            {"outputSuffix": "v20.18-target-residual-pixel-recurrence-validation"}, result=conclusion,
            artifacts=(artifact("conclusion-v20.18-target-residual-pixel-recurrence-validation.json", conclusion),), stop=True)
        inv = replace(inv, stages=inv.stages + (s36, s37, s38, s39))
        return store, store.set_control_state(inv, status="COMPLETE", control_state=copy.deepcopy(control or {
            "branchAssessments": [], "selectedExperiment": None, "schedulerAction": "INVESTIGATION_COMPLETE"}))

    def test_terminal_and_exact_real_stale_controls_admit_idempotently(self):
        for control in (None, self.STALE):
            with tempfile.TemporaryDirectory() as directory:
                store, inv = self._boundary(directory, control); before = inv.stages
                admitted = repair_obsolete_terminal_wait(store, inv)
                self.assertEqual("040-target-residual-multisector-source-prepare",
                    admitted.metadata["controlState"]["selectedExperiment"]["id"])
                self.assertIs(before, admitted.stages)
                self.assertEqual(admitted, repair_obsolete_terminal_wait(store, admitted))

    def test_stale_control_near_misses_refuse(self):
        for key in ("recovery", "schedulerAction"):
            with tempfile.TemporaryDirectory() as directory:
                control = copy.deepcopy(self.STALE); control[key] = "WRONG"
                store, inv = self._boundary(directory, control)
                self.assertEqual(inv, repair_obsolete_terminal_wait(store, inv))

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for full public-handler lifecycle")
    def test_public_040_through_043_lifecycle_acquires_without_sector_input_injection(self):
        from workflows.tess.tess_investigation import build_engine
        with tempfile.TemporaryDirectory() as directory:
            store, inv = self._boundary(directory); historical = copy.deepcopy(inv.stages)
            inv = repair_obsolete_terminal_wait(store, inv)
            engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None)
            engine.chain_stages = False
            request = StageRequest(**inv.metadata["controlState"]["selectedExperiment"])
            class WCS:
                def world_to_pixel(self, coordinate): return 1., 1.
            tpf = SimpleNamespace(time=SimpleNamespace(value=np.linspace(0., 27., 120)),
                flux=SimpleNamespace(value=np.ones((120, 3, 3))), wcs=WCS())
            analyzed = {"sector": 100, "scientificallyValid": False,
                "fullDataComparison": {"bestModel": None, "bestModelIdentifiable": False,
                    "completeModelFullRank": False},
                "temporalPredictiveValidation": {"predictiveModel": None, "predictiveSupport": False}}
            with mock.patch("workflows.tess.tess_residual_localization._download_tpf",
                    return_value=(tpf, {"sourceType": "OFFICIAL_TPF", "author": "SPOC"})), mock.patch(
                    "workflows.tess.tess_spoc_prf._tpf_detector_geometry", return_value=(1, 1, 0., 0.)), mock.patch(
                    "workflows.tess.tess_spoc_prf._list_official_prf_grid", return_value=[]), mock.patch(
                    "workflows.tess.tess_spoc_prf._official_prf_at_detector_position",
                    return_value=(np.ones((3, 3)), {}, ["official-prf.fits"])), mock.patch(
                    "workflows.tess.tess_catalog_guided_localization.analyze_generalized_catalog_guided_sector",
                    return_value=analyzed):
                for _ in range(4):
                    inv, request = engine.run_stage(inv, request, software_id="test", software_version="20.36")
            self.assertEqual(historical, inv.stages[:len(historical)])
            self.assertEqual(["040-target-residual-multisector-source-prepare",
                "041-target-residual-multisector-source-run",
                "042-target-residual-multisector-source-interpret", "043-finalize"],
                [stage.id for stage in inv.stages[len(historical):]])
            stage042, final = inv.stages[-2:]
            self.assertEqual(stage042.result,
                final.result["targetResidualMultisectorSourceLocalization"])
            self.assertEqual(stage042.result["recommendedNextTest"],
                             final.result["recommendedNextTest"])
            self.assertIsNone(request); self.assertFalse(any(stage.id.startswith("044-") for stage in inv.stages))

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for public-handler retry test")
    def test_transient_tpf_failure_is_retryable_infrastructure(self):
        from workflows.tess.tess_investigation import build_engine
        from workflows.tess.tess_sector_archive import TessArchiveTransientError
        from openstar_workflow import RetryableExecutionError
        with tempfile.TemporaryDirectory() as directory:
            store, inv = self._boundary(directory); inv = repair_obsolete_terminal_wait(store, inv)
            engine = build_engine(store, SimpleNamespace(), poll_interval=0, timeout=None); engine.chain_stages = False
            request = StageRequest(**inv.metadata["controlState"]["selectedExperiment"])
            inv, request = engine.run_stage(inv, request, software_id="test", software_version="20.36")
            with mock.patch("workflows.tess.tess_residual_localization._download_tpf",
                            side_effect=TessArchiveTransientError("temporary outage")), self.assertRaises(RetryableExecutionError):
                engine.run_stage(inv, request, software_id="test", software_version="20.36")
            failed = store.load(inv.id).stages[-1]
            self.assertEqual("TRANSIENT_INFRASTRUCTURE", failed.failure_classification)

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for generalized model tests")
    def test_hypotheses_are_all_nonempty_subsets_in_deterministic_order(self):
        self.assertEqual(255, len(generate_source_hypotheses([f"s{i}" for i in range(8)])))
        self.assertEqual(list(generate_source_hypotheses(("a", "b", "c"))), [
            "SOURCE_SUBSET_a", "SOURCE_SUBSET_b", "SOURCE_SUBSET_c",
            "SOURCE_SUBSET_a__b", "SOURCE_SUBSET_a__c", "SOURCE_SUBSET_b__c",
            "SOURCE_SUBSET_a__b__c"])

    def test_exact_old_sectors_excluded_and_remaining_support_used(self):
        all_rows = [evidence(value) for value in (*V2018_SECTOR_IDS, 4, 5, 6, 7)]
        selected = derive_additional_sectors(all_rows)
        self.assertEqual([4, 5, 6, 7], [row["sector"] for row in selected])
        self.assertFalse(set(V2018_SECTOR_IDS) & {row["sector"] for row in selected})

    def test_real_shape_selects_eleven_without_truncation(self):
        rows = [evidence(value) for value in (*V2018_SECTOR_IDS, *range(100, 111))]
        self.assertEqual(11, len(derive_additional_sectors(rows)))

    def test_competitors_preserve_frozen_order_and_bound(self):
        frozen = [{"sourceID": f"s{i}", "raDeg": i, "decDeg": -i} for i in range(8)]
        selected = derive_competing_sources(frozen, [{"classification": "AMBIGUOUS_OR_BLENDED",
            "distancesPixels": {row["sourceID"]: 0.5 for row in frozen}}])
        self.assertEqual(frozen, selected)
        too_many = frozen + [{"sourceID": "s8", "raDeg": 8, "decDeg": -8}]
        with self.assertRaises(RuntimeError):
            derive_competing_sources(too_many, [{"classification": "AMBIGUOUS_OR_BLENDED",
                "distancesPixels": {row["sourceID"]: 0.5 for row in too_many}}])
        self.assertEqual(8, MAX_COMPETING_SOURCES)

    def test_unresolved_old_sector_cannot_contribute_competitors(self):
        with self.assertRaises(RuntimeError):
            derive_competing_sources([{"sourceID": "bad"}], [{"classification": "UNRESOLVED",
                "distancesPixels": {"bad": 0.0}}])

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for generalized model tests")
    def test_identifiable_omitted_competitor_blocks_unique_winner(self):
        def weighted(**kwargs):
            ids = kwargs["component_ids"]
            return {"bic": 0.0 if ids == ["A"] else 10.0, "fullRank": True,
                "parameterEstimates": [0.0] * (2 * len(ids)),
                "sourceEstimates": [{"componentID": source, "individuallyIdentifiable": True,
                    "sinA": 1.0, "cosB": 0.0, "covariance": [[1, 0], [0, 1]]} for source in ids]}
        with mock.patch("workflows.tess.tess_catalog_guided_localization._weighted_hypothesis",
                        side_effect=weighted):
            result = compare_source_hypotheses(np.zeros((2, 2)), np.repeat(np.eye(2)[None], 2, 0),
                                               np.eye(2), ("A", "B"))
        self.assertEqual(["A"], result["bestModelSourceIDs"])
        self.assertFalse(result["bestModelIdentifiable"])
        self.assertEqual({"A", "B"}, set(result["completeModelSourceIdentifiability"]))

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for generalized model tests")
    def test_legacy_three_source_serialized_comparison_is_unchanged(self):
        self.assertEqual(("target", "candidate-1", "candidate-2"), COMPONENT_IDS)
        self.assertEqual(["TARGET_ONLY", "CANDIDATE_1_ONLY", "CANDIDATE_2_ONLY",
            "TARGET_PLUS_CANDIDATE_1", "TARGET_PLUS_CANDIDATE_2",
            "CANDIDATE_1_PLUS_CANDIDATE_2", "TARGET_PLUS_BOTH"], list(MODEL_COMPONENTS))

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for production acquisition tests")
    def test_exact_no_coverage_is_unavailable(self):
        def absent(**kwargs):
            raise RuntimeError("No official TPF or TESScut coverage available for Sector 100.")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "workflows.tess.tess_residual_localization._download_tpf", side_effect=absent):
            result = run_multisector_source_localization(self._production_preparation(directory))
        self.assertEqual("UNAVAILABLE", result["sectorResults"][0]["availability"])

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy required for production acquisition tests")
    def test_production_preserves_mask_frequency_and_harmonic_prewhitening(self):
        class WCS:
            def world_to_pixel(self, coordinate): return 1., 1.
        masked = np.ma.array(np.ones((120, 3, 3)), mask=False); masked.mask[4, 1, 1] = True
        tpf = SimpleNamespace(time=SimpleNamespace(value=np.linspace(1000., 1027., 120)),
            flux=SimpleNamespace(value=masked), wcs=WCS())
        seen = {}
        def background(cube):
            seen["maskedNaN"] = bool(np.isnan(cube[4, 1, 1])); return cube, {"method": "mock"}
        def prewhiten(**kwargs):
            seen["orders"] = kwargs["harmonic_orders"]
            return kwargs["cube"], np.ones((3, 3), bool)
        def analyze(**kwargs):
            seen["frequency"] = kwargs["candidate_frequency"]
            return {"sector": 100, "scientificallyValid": False,
                "fullDataComparison": {"bestModel": None, "bestModelIdentifiable": False,
                    "completeModelFullRank": False},
                "temporalPredictiveValidation": {"predictiveModel": None, "predictiveSupport": False}}
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "workflows.tess.tess_residual_localization._download_tpf",
                return_value=(tpf, {"sourceType": "OFFICIAL_TPF", "author": "SPOC"})), mock.patch(
                "workflows.tess.tess_residual_localization._background_subtract_cube", side_effect=background), mock.patch(
                "workflows.tess.tess_multisource_residual._prewhiten_cube_raw", side_effect=prewhiten), mock.patch(
                "workflows.tess.tess_spoc_prf._tpf_detector_geometry", return_value=(1, 1, 0., 0.)), mock.patch(
                "workflows.tess.tess_spoc_prf._list_official_prf_grid", return_value=[]), mock.patch(
                "workflows.tess.tess_spoc_prf._official_prf_at_detector_position",
                return_value=(np.ones((3, 3)), {}, ["official-prf.fits"])), mock.patch(
                "workflows.tess.tess_catalog_guided_localization.analyze_generalized_catalog_guided_sector",
                side_effect=analyze):
            run_multisector_source_localization(self._production_preparation(directory))
        self.assertTrue(seen["maskedNaN"])
        self.assertEqual((1, 2), seen["orders"])
        self.assertEqual(.123456789, seen["frequency"])

    def test_target_catalog_blend_switching_unavailable_and_unresolved(self):
        target = interpret_multisector([sector(i, ("target",)) for i in range(3)], "target")
        self.assertEqual("TARGET_SUPPORTED", target["classification"])
        catalog = interpret_multisector([sector(i, ("other",)) for i in range(3)], "target")
        self.assertEqual("CATALOG_SOURCE_SUPPORTED", catalog["classification"])
        blend = interpret_multisector([sector(i, ("target", "other")) for i in range(2)], "target")
        self.assertEqual("SOURCE_SWITCHING_OR_BLEND", blend["classification"])
        self.assertFalse(blend["sourceAttributionResolved"])
        self.assertTrue(blend["sourceSwitchingOrBlendDetected"])
        target_with_blend = interpret_multisector(
            [sector(i, ("target",)) for i in range(3)]
            + [sector(10 + i, ("target", "other")) for i in range(2)], "target")
        self.assertEqual("SOURCE_SWITCHING_OR_BLEND", target_with_blend["classification"])
        self.assertFalse(target_with_blend["sourceAttributionResolved"])
        self.assertTrue(target_with_blend["sourceSwitchingOrBlendDetected"])
        self.assertEqual("SOURCE_SWITCHING_TEMPORAL_MODEL",
                         target_with_blend["recommendedNextTest"])
        catalog_with_blend = interpret_multisector(
            [sector(i, ("other",)) for i in range(3)]
            + [sector(10 + i, ("target", "other")) for i in range(2)], "target")
        self.assertEqual("SOURCE_SWITCHING_OR_BLEND", catalog_with_blend["classification"])
        self.assertFalse(catalog_with_blend["sourceAttributionResolved"])
        self.assertTrue(catalog_with_blend["sourceSwitchingOrBlendDetected"])
        switching = interpret_multisector([sector(1, ("target",)), sector(2, ("target",)),
            sector(3, ("other",)), sector(4, ("other",))], "target")
        self.assertEqual("SOURCE_SWITCHING_OR_BLEND", switching["classification"])
        unresolved = interpret_multisector([sector(1, ("target",)), sector(2, available=False),
            sector(3, ("target",), identifiable=False)], "target")
        self.assertEqual("UNRESOLVED", unresolved["classification"])
        self.assertEqual([2], unresolved["unavailableSectors"])
        self.assertEqual("TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY",
                         unresolved["recommendedNextTest"])

    def test_rank_deficient_complete_model_is_scientifically_invalid_not_negative(self):
        invalid = sector(7, ("target",), identifiable=False)
        invalid["scientificallyValid"] = False
        result = interpret_multisector([invalid, sector(8, available=False)], "target")
        self.assertEqual("UNRESOLVED", result["classification"])
        self.assertEqual(0, result["validSectorCount"])
        self.assertEqual([7], result["scientificallyInvalidSectors"])
        self.assertEqual([8], result["unavailableSectors"])


if __name__ == "__main__":
    unittest.main()
