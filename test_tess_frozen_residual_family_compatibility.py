import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_investigation import InvestigationStage, InvestigationStore
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    repair_obsolete_terminal_wait,
)
from workflows.tess.tess_localization_evidence import (
    frozen_residual_localization_family,
)


FAMILY_PERIOD = 10.30084080080649
RESIDUAL_FREQUENCY = 0.45306836767392505
RESIDUAL_PERIOD = 2.2071724078510457


def _evidence():
    morphology = {
        "physicalCycleResolved": False,
        "resolvedPhysicalPeriodDays": None,
    }
    dynamic = {
        "referenceFamilyPeriodDays": FAMILY_PERIOD,
        "supportedHarmonicOrders": [1, 2, 3, 4],
    }
    time_frequency_prepare = {
        "absoluteTimeReferenceDays": 3857.082833057051,
    }
    time_frequency = {
        "classification": "STABLE_RESIDUAL_MODE",
        "residualEvolution": {"classification": "STABLE_RESIDUAL_MODE"},
    }
    mode = {
        "classification": "INDEPENDENT_STABLE_MODE",
        "independentModeEvidenceSurvived": True,
        "physicalMechanismResolved": False,
        "establishedPeriodFamily": {
            "referencePeriodDays": FAMILY_PERIOD,
            "modeledHarmonicOrders": [1, 2, 5],
        },
        "modeCandidate": {
            "frequencyCyclesPerDay": RESIDUAL_FREQUENCY,
            "periodDays": RESIDUAL_PERIOD,
            "supportingSectors": [94, 95, 102, 103],
        },
        "harmonicRelation": {
            "testedOrder": 5,
            "harmonicFrequencyCyclesPerDay": 0.4853972696683684,
            "absoluteFrequencySeparation": 0.03208952050303532,
            "frequencyResolutionCyclesPerDay": 0.00023938149140801442,
            "baselineDays": 4177.4324076523835,
            "commensurateWithinResolution": False,
        },
    }
    return morphology, dynamic, time_frequency_prepare, time_frequency, mode


class FrozenResidualFamilyCompatibilityTests(unittest.TestCase):
    def test_noncommensurate_competing_harmonic_uses_dynamic_supported_orders(self):
        morphology, dynamic, tf_prepare, tf_summary, mode = _evidence()

        adapted = frozen_residual_localization_family(
            morphology, dynamic, tf_prepare, tf_summary, mode
        )

        self.assertIsNotNone(adapted)
        self.assertEqual(FAMILY_PERIOD, adapted[0])
        self.assertEqual((1, 2, 3, 4), adapted[1])
        self.assertEqual("UNRESOLVED_FAMILY_ANALYSIS_REFERENCE", adapted[3])

    def test_unexplained_harmonic_order_mismatch_still_fails_closed(self):
        morphology, dynamic, tf_prepare, tf_summary, mode = _evidence()
        mode["establishedPeriodFamily"]["modeledHarmonicOrders"] = [1, 2, 6]

        self.assertIsNone(
            frozen_residual_localization_family(
                morphology, dynamic, tf_prepare, tf_summary, mode
            )
        )

    def test_historical_v2011_failure_selects_append_only_stage_028(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = InvestigationStore(Path(temporary.name) / "investigations")
        investigation = store.create(
            "tess-discovery-sector-1-tic-277940827",
            WORKFLOW_ID,
            WORKFLOW_VERSION,
        )

        morphology, dynamic, tf_prepare, tf_summary, mode = _evidence()
        stages = (
            InvestigationStage(
                "012-characterize-variability",
                "openstar.tess.morphology.analyze",
                "COMPLETE",
                None,
                {},
                result=morphology,
            ),
            InvestigationStage(
                "018-dynamic-harmonic-modeling",
                "openstar.tess.dynamic-harmonic.analyze",
                "COMPLETE",
                "012-characterize-variability",
                {},
                result=dynamic,
            ),
            InvestigationStage(
                "019-prepare-time-frequency",
                "openstar.tess.time-frequency.prepare",
                "COMPLETE",
                "018-dynamic-harmonic-modeling",
                {},
                result=tf_prepare,
            ),
            InvestigationStage(
                "022-summarize-time-frequency",
                "openstar.tess.time-frequency.summarize",
                "COMPLETE",
                "019-prepare-time-frequency",
                {},
                result=tf_summary,
            ),
            InvestigationStage(
                "023-mode-identification",
                "openstar.tess.mode-identification.analyze",
                "COMPLETE",
                "022-summarize-time-frequency",
                {},
                result=mode,
            ),
            InvestigationStage(
                "024-prepare-residual-mode-localization",
                "openstar.tess.residual-mode-localization.prepare",
                "COMPLETE",
                "023-mode-identification",
                {},
                result={"subtractedHarmonicOrders": [1, 2, 3, 4]},
            ),
            InvestigationStage(
                "026-interpret-residual-mode-localization",
                "openstar.tess.residual-mode-localization.interpret",
                "COMPLETE",
                "024-prepare-residual-mode-localization",
                {},
                result={
                    "recommendedNextTest": "RESIDUAL_MODE_SOURCE_LOCALIZATION_REVIEW",
                    "physicalMechanismResolved": False,
                },
            ),
            InvestigationStage(
                "027-prepare-residual-mode-localization-review",
                "openstar.tess.residual-mode-localization-review.prepare",
                "FAILED",
                "026-interpret-residual-mode-localization",
                {},
                error=(
                    "RuntimeError: v20.11 requires a complete frozen "
                    "residual-mode family."
                ),
                failure_classification="NON_RETRYABLE",
            ),
        )
        failed = stages[-1]
        selected = {
            "id": failed.id,
            "handler_id": failed.handler_id,
            "parameters": {},
            "triggered_by_stage_id": failed.triggered_by_stage_id,
        }
        investigation = replace(
            investigation,
            status="FAILED",
            stages=stages,
            metadata={
                "controlState": {
                    "branchAssessments": [],
                    "selectedExperiment": selected,
                    "schedulerAction": "INVESTIGATION_FAILED",
                }
            },
        )
        store.save(investigation)

        repaired = repair_obsolete_terminal_wait(store, store.load(investigation.id))
        control = repaired.metadata["controlState"]
        retry = control["selectedExperiment"]

        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(stages, repaired.stages)
        self.assertEqual(
            "TESS_UNRESOLVED_DYNAMIC_LOCALIZATION_REVIEW_COMPATIBILITY_RETRY",
            control["recovery"],
        )
        self.assertEqual(
            "028-prepare-residual-mode-localization-review",
            retry["id"],
        )
        self.assertEqual(
            "openstar.tess.residual-mode-localization-review.prepare",
            retry["handler_id"],
        )
        self.assertEqual(failed.id, retry["triggered_by_stage_id"])


if __name__ == "__main__":
    unittest.main()
