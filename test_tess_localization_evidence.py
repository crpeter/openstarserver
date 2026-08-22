import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_localization_evidence import (
    frozen_residual_localization_family,
)


class FrozenResidualLocalizationFamilyTests(unittest.TestCase):
    def _evidence(self, *, morphology_period=10.510316195053623):
        morphology = {
            "physicalCycleResolved": True,
            "resolvedPhysicalPeriodDays": morphology_period,
        }
        time_frequency_prepare = {"absoluteTimeReferenceDays": 2500.0}
        time_frequency = {
            "residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"}
        }
        mode = {
            "classification": "INDEPENDENT_STABLE_MODE",
            "independentModeEvidenceSurvived": True,
            "physicalMechanismResolved": False,
            "establishedPeriodFamily": {
                "referencePeriodDays": 10.510316195053623,
                "modeledHarmonicOrders": [1, 2, 3],
            },
            "modeCandidate": {
                "frequencyCyclesPerDay": 0.27101611598985065,
                "periodDays": 1.0 / 0.27101611598985065,
                "supportingSectors": [28, 68, 92, 95],
            },
        }
        return morphology, time_frequency_prepare, time_frequency, mode

    def test_real_resolved_family_adapts_and_preserves_morphology_period(self):
        morphology, tf_prepare, tf_summary, mode = self._evidence()

        adapted = frozen_residual_localization_family(
            morphology, None, tf_prepare, tf_summary, mode
        )

        self.assertIsNotNone(adapted)
        self.assertEqual(10.510316195053623, adapted[0])
        self.assertEqual((1, 2, 3), adapted[1])
        self.assertEqual("MORPHOLOGY_RESOLVED_PHYSICAL_PERIOD", adapted[3])

    def test_resolved_morphology_and_mode_family_mismatch_fails_closed(self):
        morphology, tf_prepare, tf_summary, mode = self._evidence(
            morphology_period=11.510316195053623
        )

        self.assertIsNone(frozen_residual_localization_family(
            morphology, None, tf_prepare, tf_summary, mode
        ))

    def test_actual_review_prepare_rejects_mismatched_resolved_family(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = InvestigationStore(Path(temporary.name) / "investigations")
        investigation = store.create("inconsistent-review", "workflow", "1")
        morphology, tf_prepare, tf_summary, mode = self._evidence(
            morphology_period=11.510316195053623
        )
        stages = (
            InvestigationStage("001-prepare-target", "openstar.tess.prepare-target",
                               "COMPLETE", None, {}, result={}),
            InvestigationStage("002-identity", "openstar.tess.catalog-identity",
                               "COMPLETE", "001-prepare-target", {}, result={}),
            InvestigationStage("003-independent", "openstar.tess.independent.prepare",
                               "COMPLETE", "002-identity", {}, result={}),
            InvestigationStage("010-morphology", "openstar.tess.morphology.analyze",
                               "COMPLETE", "003-independent", {}, result=morphology),
            InvestigationStage("012-time-frequency-prepare",
                               "openstar.tess.time-frequency.prepare", "COMPLETE",
                               "010-morphology", {}, result=tf_prepare),
            InvestigationStage("013-time-frequency-summary",
                               "openstar.tess.time-frequency.summarize", "COMPLETE",
                               "012-time-frequency-prepare", {}, result=tf_summary),
            InvestigationStage("018-mode-identification",
                               "openstar.tess.mode-identification.analyze", "COMPLETE",
                               "013-time-frequency-summary", {}, result=mode),
            InvestigationStage("021-interpret-residual-mode-localization",
                               "openstar.tess.residual-mode-localization.interpret",
                               "COMPLETE", "018-mode-identification", {}, result={
                                   "recommendedNextTest":
                                   "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW"
                               }),
        )
        investigation = replace(investigation, status="RUNNING", stages=stages)
        store.save(investigation)
        request = StageRequest(
            "022-prepare-residual-mode-localization-review",
            "openstar.tess.residual-mode-localization-review.prepare", {},
            "021-interpret-residual-mode-localization",
        )
        engine = build_engine(store, mock.Mock(), poll_interval=0.0, timeout=1.0)

        with mock.patch(
            "workflows.tess.tess_investigation."
            "build_residual_mode_localization_review_project"
        ) as build_project:
            with self.assertRaisesRegex(RuntimeError, "completed v20.9 nonstationary"):
                engine.run_stage(
                    investigation, request, software_id="test", software_version="current"
                )
        build_project.assert_not_called()


if __name__ == "__main__":
    unittest.main()
