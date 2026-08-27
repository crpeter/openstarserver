import tempfile
import unittest
from pathlib import Path

from openstar_coordinator_client import OpenStarCoordinatorClient
from openstar_investigation import Investigation, InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from workflows.tess.tess_investigation import (
    SOFTWARE_ID, SOFTWARE_VERSION, build_engine,
)


HANDLER = "openstar.tess.main-family-frequency-domain-reassessment.analyze"


class LedgerAcceptingStore(InvestigationStore):
    def verified_terminal_stage_ledger_hash(self, investigation_id, stage):
        if stage.status not in {"COMPLETE", "FAILED"}:
            return None
        return f"verified-{stage.id}"


def stage(stage_id, handler, *, status="COMPLETE", trigger=None, result=None, stop=False):
    return InvestigationStage(
        id=stage_id, handler_id=handler, status=status,
        triggered_by_stage_id=trigger, parameters={},
        result=result, stop=stop,
    )


class Stage035RuntimeBoundaryTests(unittest.TestCase):
    def test_workflow_engine_running_stage_does_not_invalidate_stage034_boundary(self):
        prior = {
            "classification": "FREQUENCY_FAMILY_NOT_TIME_DOMAIN_REPLICATED",
            "recommendedNextTest": "MAIN_FAMILY_FREQUENCY_DOMAIN_REASSESSMENT",
            "combinedEvidence": {
                "rawFamilyRecurrenceSectorIDs": [],
                "possibleDoubleRecurrenceSectorIDs": [],
            },
            "frozenDatasetProvenance": [],
        }
        science = stage(
            "031-target-residual-astrophysical-interpretation",
            "openstar.tess.target-residual-astrophysical-interpretation.analyze",
        )
        recurrence = stage(
            "033-main-family-time-domain-recurrence",
            "openstar.tess.main-family-time-domain-recurrence.analyze",
            trigger="032-finalize", result=prior,
        )
        final = stage(
            "034-finalize", "openstar.tess.finalize",
            trigger=recurrence.id, stop=True,
        )
        investigation = Investigation(
            id="stage035-runtime-boundary",
            workflow_id="openstar.workflow.tess-investigation.v1",
            workflow_version="20.2", status="RUNNING",
            created_at="2026-08-27T00:00:00+00:00",
            updated_at="2026-08-27T00:00:00+00:00",
            metadata={}, stages=(science, recurrence, final),
        )
        request = StageRequest(
            "035-main-family-frequency-domain-reassessment", HANDLER, {}, final.id
        )
        with tempfile.TemporaryDirectory() as temp:
            store = LedgerAcceptingStore(Path(temp) / "investigations")
            engine = build_engine(
                store, OpenStarCoordinatorClient(), poll_interval=0.01, timeout=None
            )
            with self.assertRaisesRegex(
                RuntimeError, "frequency reassessment requires stage033 frozen sectors"
            ):
                engine.run_stage(
                    investigation, request,
                    software_id=SOFTWARE_ID, software_version=SOFTWARE_VERSION,
                )
            persisted = store.load(investigation.id)
            self.assertEqual(
                "RuntimeError: frequency reassessment requires stage033 frozen sectors",
                persisted.stages[-1].error,
            )
            self.assertNotIn(
                "exact stage034 boundary verification failed",
                persisted.stages[-1].error or "",
            )


if __name__ == "__main__":
    unittest.main()
