from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_investigation import ArtifactReference, InvestigationStage, InvestigationStore, sha256_file
from openstar_targets import InvestigationTarget
from workflows.tess.tess_autonomy import WORKFLOW_ID, plan_tess_branches, repair_obsolete_terminal_wait
from workflows.tess.tess_intrinsic_nonstationary import classify_target_component


class TessIntrinsicNonstationaryTests(unittest.TestCase):
    def boundary(self):
        return {"classification": "TARGET_RESIDUAL_COMPONENT_DOMINANT",
                "residualModeOrigin": "TARGET_DOMINANT", "physicalMechanismResolved": False,
                "recommendedNextTest": "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION",
                "targetComponentID": "target",
                "componentSummaries": [{"componentID": "target", "componentType": "TARGET",
                                        "independentSupportCount": 4}]}

    def preparation(self, root: Path, *, epochs=(100.0, 300.0, 600.0),
                    frequencies=None, common=True, sector_ids=None, q=0.0,
                    time_reference=0.0):
        frequencies = frequencies or (1.2,) * len(epochs)
        sector_ids = sector_ids or tuple(range(1, len(epochs) + 1))
        series, artifacts = [], []
        for sector, epoch, frequency in zip(sector_ids, epochs, frequencies):
            absolute_times = [epoch + index / 20 for index in range(160)]
            relative_times = [value - time_reference for value in absolute_times]
            common_times = [value + 0.5 * q * value * value for value in relative_times]
            local_times = [value - min(common_times) for value in common_times]
            values = [math.sin(2 * math.pi * frequency * value) for value in absolute_times]
            coefficient = root / f"target-{sector}-coefficients.json"
            dataset = root / f"target-{sector}.json"
            payload = {"times": local_times, "coefficients": values, "componentID": "target"}
            if common:
                payload["commonWarpedTimes"] = common_times
                payload["absoluteTimes"] = absolute_times
                payload["timeReferenceDays"] = time_reference
                payload["fractionalFrequencyDriftPerDay"] = q
            coefficient.write_text(json.dumps(payload))
            dataset.write_text(json.dumps({"science": {"componentID": "target"},
                                           "source": {"timeReferenceDays": 1500.0}}))
            series.append({"datasetID": f"target-{sector}", "componentID": "target",
                           "componentType": "TARGET", "sector": sector, "combined": False,
                           "coefficientSeriesPath": str(coefficient), "datasetPath": str(dataset)})
            for path in (coefficient, dataset):
                artifacts.append(ArtifactReference(str(path.resolve()), sha256_file(path), "application/json"))
        return {"referenceFrequency": 1.2, "preparedSeries": series}, tuple(artifacts)

    def classify(self, preparation, artifacts):
        return classify_target_component(preparation=preparation, decomposition=self.boundary(),
                                         authoritative_artifacts=artifacts,
                                         preparation_link_verified=True)

    def investigation(self, root: Path, *, complete=False):
        store = InvestigationStore(root)
        investigation = store.create("tess-real-shaped", WORKFLOW_ID, "20.2")
        stage = InvestigationStage("026-interpret-multi-source-residual",
            "openstar.tess.multi-source-residual.interpret", "COMPLETE", "025-run", {},
            result=self.boundary())
        investigation = replace(investigation, stages=(stage,))
        store.save(investigation)
        if complete:
            investigation = store.set_control_state(investigation, status="COMPLETE",
                control_state={"schedulerAction": "INVESTIGATION_COMPLETE"})
        return store, investigation

    def test_exact_boundary_selects_new_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            _, investigation = self.investigation(Path(directory))
            target = InvestigationTarget("target", investigation.id, WORKFLOW_ID, "20.2")
            request = plan_tess_branches(investigation, target)[0].experiment
        self.assertEqual("openstar.tess.intrinsic-nonstationary.analyze", request.handler_id)

    def test_complete_boundary_reopens_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            store, investigation = self.investigation(Path(directory), complete=True)
            old_stages = investigation.stages
            repaired = repair_obsolete_terminal_wait(store, investigation)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(old_stages, repaired.stages)

    def test_unrelated_routes_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            _, investigation = self.investigation(Path(directory))
            target = InvestigationTarget("target", investigation.id, WORKFLOW_ID, "20.2")
            for classification, origin, recommendation in (
                ("OFFSET_RESIDUAL_COMPONENT_DOMINANT", "OFFSET_DOMINANT", "IDENTIFY_OFFSET_RESIDUAL_VARIABLE_SOURCE"),
                ("MIXED_RESIDUAL_COMPONENTS", "MIXED", "NEIGHBOR_SOURCE_IDENTIFICATION_AND_CATALOG_CROSSMATCH"),
                ("UNRESOLVED", "UNKNOWN", "PIXEL_RESPONSE_FUNCTION_DEBLENDING")):
                result = self.boundary(); result.update(classification=classification,
                    residualModeOrigin=origin, recommendedNextTest=recommendation)
                changed = replace(investigation, stages=(replace(investigation.stages[0], result=result),))
                branches = plan_tess_branches(changed, target)
                self.assertFalse(branches and branches[0].experiment.handler_id ==
                                 "openstar.tess.intrinsic-nonstationary.analyze")

    def test_complete_v2013_mechanism_boundary_reopens_append_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create("v2013-mechanism", WORKFLOW_ID, "20.2")
            result = {"classification": "AMPLITUDE_EVOLVING_TARGET_RESIDUAL",
                      "physicalMechanismResolved": False,
                      "recommendedNextTest": "TARGET_RESIDUAL_PHYSICAL_MECHANISM_FOLLOWUP"}
            stage = InvestigationStage("027-classify-intrinsic-target-residual",
                "openstar.tess.intrinsic-nonstationary.analyze", "COMPLETE", "026-interpret", {},
                result=result)
            investigation = replace(investigation, stages=(stage,))
            store.save(investigation)
            investigation = store.set_control_state(investigation, status="COMPLETE",
                control_state={"schedulerAction": "INVESTIGATION_COMPLETE"})
            immutable_stages = investigation.stages
            repaired = repair_obsolete_terminal_wait(store, investigation)
            repeated = repair_obsolete_terminal_wait(store, repaired)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual("openstar.tess.target-residual-mechanism.analyze",
                         repaired.metadata["controlState"]["selectedExperiment"]["handler_id"])
        self.assertEqual(immutable_stages, repaired.stages)
        self.assertEqual(repaired, repeated)

    def test_unrelated_complete_v2013_does_not_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create("v2013-unrelated", WORKFLOW_ID, "20.2")
            stage = InvestigationStage("027-classify-intrinsic-target-residual",
                "openstar.tess.intrinsic-nonstationary.analyze", "COMPLETE", "026-interpret", {},
                result={"classification": "STATIONARY_FREQUENCY_COMPATIBLE_TARGET_RESIDUAL",
                        "physicalMechanismResolved": False,
                        "recommendedNextTest": "OTHER"})
            investigation = replace(investigation, stages=(stage,))
            store.save(investigation)
            investigation = store.set_control_state(investigation, status="COMPLETE",
                control_state={"schedulerAction": "INVESTIGATION_COMPLETE"})
            repaired = repair_obsolete_terminal_wait(store, investigation)
        self.assertEqual(investigation, repaired)

    def test_modified_coefficient_fails_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation, artifacts = self.preparation(Path(directory))
            Path(preparation["preparedSeries"][0]["coefficientSeriesPath"]).write_text("{}")
            result = self.classify(preparation, artifacts)
        self.assertEqual("INSUFFICIENT_TARGET_COMPONENT_TEMPORAL_EVIDENCE", result["classification"])
        self.assertTrue(any("failed frozen hash" in reason for reason in result["failClosedReasons"]))

    def test_modified_dataset_fails_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation, artifacts = self.preparation(Path(directory))
            Path(preparation["preparedSeries"][0]["datasetPath"]).write_text("{}")
            result = self.classify(preparation, artifacts)
        self.assertEqual("INSUFFICIENT_TARGET_COMPONENT_TEMPORAL_EVIDENCE", result["classification"])

    def test_missing_authoritative_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation, _ = self.preparation(Path(directory))
            result = self.classify(preparation, ())
        self.assertTrue(result["failClosedReasons"])

    def test_real_v2012_local_time_reset_cannot_claim_drift_or_coherence(self):
        with tempfile.TemporaryDirectory() as directory:
            preparation, artifacts = self.preparation(Path(directory), common=False,
                frequencies=(1.17, 1.20, 1.23))
            result = self.classify(preparation, artifacts)
        self.assertFalse(result["modelSelectionDiagnostics"]["commonTimingAvailable"])
        self.assertNotIn("COHERENT", result["classification"])
        self.assertNotIn("DRIFTING", result["classification"])
        self.assertTrue(result["timingLimitations"])

    def test_valid_common_time_stationary_signal_measures_phase_coherence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.classify(*self.preparation(Path(directory)))
        self.assertEqual("STATIONARY_PHASE_COHERENT_TARGET_RESIDUAL_MODE", result["classification"])
        self.assertLess(result["modelSelectionDiagnostics"]["crossSectorPhaseCircularStdRadians"], .35)

    def test_drift_uses_elapsed_observation_time(self):
        epochs = (100.0, 300.0, 600.0, 1000.0)
        frequencies = tuple(1.14 + .00012 * epoch for epoch in epochs)
        with tempfile.TemporaryDirectory() as directory:
            result = self.classify(*self.preparation(Path(directory), epochs=epochs,
                                                     frequencies=frequencies))
        self.assertEqual("SMOOTHLY_FREQUENCY_DRIFTING_TARGET_RESIDUAL_MODE", result["classification"])
        self.assertGreater(result["modelSelectionDiagnostics"]["frequencyEpochLinearCorrelation"], .8)

    def test_sector_id_shuffle_does_not_change_temporal_result(self):
        epochs = (100.0, 300.0, 600.0, 1000.0)
        frequencies = tuple(1.14 + .00012 * epoch for epoch in epochs)
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first = self.classify(*self.preparation(Path(left), epochs=epochs, frequencies=frequencies,
                                                    sector_ids=(1, 2, 3, 4)))
            second = self.classify(*self.preparation(Path(right), epochs=epochs, frequencies=frequencies,
                                                     sector_ids=(87, 1, 61, 27)))
        self.assertEqual(first["classification"], second["classification"])

    def test_epoch_change_not_sector_ordinal_controls_drift(self):
        frequencies = (1.15, 1.17, 1.20, 1.24)
        with tempfile.TemporaryDirectory() as directory:
            result = self.classify(*self.preparation(Path(directory),
                epochs=(100.0, 250.0, 500.0, 900.0), frequencies=frequencies,
                sector_ids=(10, 9, 8, 7)))
        self.assertGreater(result["modelSelectionDiagnostics"]["frequencyEpochLinearCorrelation"], .8)

    def test_nonzero_frozen_warp_cannot_erase_absolute_time_drift(self):
        epochs = (100.0, 300.0, 600.0, 1000.0)
        frequencies = tuple(1.14 + .00012 * epoch for epoch in epochs)
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            unwarped = self.classify(*self.preparation(
                Path(left), epochs=epochs, frequencies=frequencies, q=0.0,
                sector_ids=(1, 2, 3, 4)))
            warped = self.classify(*self.preparation(
                Path(right), epochs=epochs, frequencies=frequencies, q=0.0002,
                sector_ids=(61, 1, 87, 27)))
        self.assertEqual("SMOOTHLY_FREQUENCY_DRIFTING_TARGET_RESIDUAL_MODE",
                         unwarped["classification"])
        self.assertEqual(unwarped["classification"], warped["classification"])
        self.assertTrue(all(item["frequencyCoordinate"] == "ORIGINAL_ABSOLUTE_TIME"
                            for item in warped["temporalModelEvidence"]))

    def test_upstream_maximum_drift_range_expands_physical_search(self):
        # v20.9 permits a 30% edge drift.  The last physical frequency is
        # outside a fixed +20% search and must nevertheless be recovered.
        epochs = (0.0, 333.0, 667.0, 1000.0)
        q = 0.0003
        frequencies = tuple(1.2 * (1.0 + q * epoch) for epoch in epochs)
        with tempfile.TemporaryDirectory() as directory:
            result = self.classify(*self.preparation(
                Path(directory), epochs=epochs, frequencies=frequencies, q=q))
        last = max(result["temporalModelEvidence"],
                   key=lambda item: item["observationEpochAbsoluteDays"])
        self.assertGreater(last["frequency"], 1.2 * 1.2)
        self.assertGreater(last["maximumFrequency"], frequencies[-1])
        self.assertFalse(last["winnerAtSearchBoundary"])

    def test_zero_q_is_invariant_to_global_absolute_time_translation(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first = self.classify(*self.preparation(
                Path(left), epochs=(100.0, 300.0, 600.0), q=0.0))
            shifted = self.classify(*self.preparation(
                Path(right), epochs=(10100.0, 10300.0, 10600.0), q=0.0))
        self.assertEqual(first["classification"], shifted["classification"])
        self.assertEqual(
            [round(item["frequency"], 9) for item in first["temporalModelEvidence"]],
            [round(item["frequency"], 9) for item in shifted["temporalModelEvidence"]],
        )

    def test_observable_is_only_target_component(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.classify(*self.preparation(Path(directory)))
        self.assertEqual("v20.12 spatially-decomposed target coefficient series", result["observable"])
        self.assertTrue(all(item["authoritativeSha256"] == item["verifiedCurrentSha256"]
                            for item in result["inputProvenance"]["preparationArtifacts"]))


if __name__ == "__main__":
    unittest.main()
