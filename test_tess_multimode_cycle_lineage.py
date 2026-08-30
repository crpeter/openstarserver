import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_workflow import StageRequest
from test_tess_resolved_cycle import nested_result
from workflows.tess.tess_autonomy import WORKFLOW_ID, WORKFLOW_VERSION
from workflows.tess.tess_investigation import build_engine
from workflows.tess.tess_resolved_cycle import authoritative_resolved_cycle


class TessMultimodeCycleLineageTests(unittest.TestCase):
    def _investigation(self, root, *, nested=True, tamper=None):
        store = InvestigationStore(Path(root) / "investigations")
        investigation = store.create(
            "multimode-cycle-lineage", WORKFLOW_ID, WORKFLOW_VERSION)
        if nested:
            morphology = {
                "physicalCycleResolved": False,
                "resolvedPhysicalPeriodDays": None,
                "morphologyClass": "UNRESOLVED_DOUBLE_WAVE_ALIAS",
            }
            cycle = authoritative_resolved_cycle(
                morphology=None, dynamic_harmonic=nested_result())
        else:
            morphology = {
                "physicalCycleResolved": True,
                "resolvedPhysicalPeriodDays": 13.0,
                "morphologyClass": "DOUBLE_WAVE_PHYSICAL_CYCLE_SUPPORTED",
            }
            cycle = authoritative_resolved_cycle(morphology=morphology)

        physical = {
            "version": "openstar.tess-physical-interpretation.v2",
            "physicalPeriodDays": 13.0,
            "photometricFirstHarmonicPeriodDays": 6.5,
            "physicalCycleEvidence": copy.deepcopy(cycle),
            "physicalMechanismResolved": False,
        }
        localization = {
            "version": "openstar.tess-pixel-localization.v1",
            "physicalPeriodDays": 13.0,
            "photometricFirstHarmonicPeriodDays": 6.5,
            "physicalCycleEvidence": copy.deepcopy(cycle),
            "crossSector": {
                "classification": "TARGET_SOURCE_SUPPORTED",
                "variableSignalOrigin": "TARGET_CONSISTENT",
                "recommendedNextTest":
                "MULTI_MODE_FREQUENCY_DECOMPOSITION",
            },
            "recommendedNextTest": "MULTI_MODE_FREQUENCY_DECOMPOSITION",
        }
        if tamper is not None:
            tamper(localization)
        stages = (
            InvestigationStage(
                "001-prepare-target", "openstar.tess.prepare-target",
                "COMPLETE", None, {}, result={
                    "datasetPath": str(Path(root) / "primary.json"),
                    "sourceProjectPath": str(Path(root) / "source.json"),
                    "sourceDatasetEntry": {
                        "id": "primary",
                        "targetName": "Synthetic target",
                    },
                    "sector": 1,
                }),
            InvestigationStage(
                "006-prepare-independent", "openstar.tess.independent.prepare",
                "COMPLETE", "001-prepare-target", {},
                result={"preparedSectors": []}),
            InvestigationStage(
                "010-morphology", "openstar.tess.morphology.analyze",
                "COMPLETE", "006-prepare-independent", {}, result=morphology),
            InvestigationStage(
                "019-physical-interpretation", "openstar.tess.physical.interpret",
                "COMPLETE", "010-morphology", {}, result=physical),
            InvestigationStage(
                "021-source-localization",
                "openstar.tess.source-localization.analyze", "COMPLETE",
                "019-physical-interpretation", {}, result=localization),
        )
        investigation = replace(investigation, stages=stages)
        store.save(investigation)
        return store, investigation, cycle

    @staticmethod
    def _prepared_spec(root):
        project = Path(root) / "prepared-multimode-project.json"
        project.write_text("{}\n", encoding="utf-8")
        return {
            "available": True,
            "iteration": 1,
            "projectPath": str(project),
            "frequencySearch": {
                "minimumFrequency": 0.01,
                "maximumFrequency": 12.0,
            },
            "preparedDatasets": [],
            "totalWorkUnits": 1,
        }

    def test_prepare_accepts_nested_and_legacy_morphology_cycle_sources(self):
        for nested in (True, False):
            with self.subTest(nested=nested), \
                    tempfile.TemporaryDirectory() as temporary:
                store, investigation, cycle = self._investigation(
                    temporary, nested=nested)
                engine = build_engine(
                    store, coordinator=mock.Mock(),
                    poll_interval=0.0, timeout=None)
                engine.chain_stages = False
                spec = self._prepared_spec(temporary)
                with mock.patch(
                    "workflows.tess.tess_investigation."
                    "build_residual_search_project",
                    return_value=spec,
                ) as build:
                    completed, next_request = engine.run_stage(
                        investigation,
                        StageRequest(
                            "023-prepare-multimode-iteration-1",
                            "openstar.tess.multimode.prepare",
                            {"iteration": 1},
                            "021-source-localization",
                        ),
                        software_id="integration",
                        software_version="20.7",
                    )

                self.assertEqual(
                    13.0, build.call_args.kwargs["physical_period_days"])
                self.assertEqual(
                    "openstar.tess.multimode.run", next_request.handler_id)
                hashes = completed.stages[-1].provenance.input_hashes
                self.assertIn("resolvedCycle", hashes)
                self.assertIn("physicalInterpretation", hashes)
                self.assertIn("sourceLocalization", hashes)
                self.assertEqual(cycle, completed.stages[-2].result[
                    "physicalCycleEvidence"])

    def test_summary_uses_nested_cycle_when_morphology_is_unresolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation, _ = self._investigation(
                temporary, nested=True)
            investigation = replace(
                investigation,
                stages=investigation.stages + (InvestigationStage(
                    "025-interpret-multimode-iteration-1",
                    "openstar.tess.multimode.interpret", "COMPLETE",
                    "024-run-multimode-iteration-1", {},
                    result={"iteration": 1, "datasetResults": []}),),
            )
            store.save(investigation)
            engine = build_engine(
                store, coordinator=mock.Mock(),
                poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            summary = {
                "iterationsCompleted": 1,
                "classification": "NO_RECURRENT_SECONDARY_MODE",
                "independentSectorsWithAcceptedResidualModes": [],
                "physicalMechanismResolved": False,
                "recommendedNextTest": "TIME_FREQUENCY_EVOLUTION_ANALYSIS",
            }
            with mock.patch(
                "workflows.tess.tess_investigation."
                "summarize_multimode_decomposition",
                return_value=summary,
            ) as summarize:
                completed, next_request = engine.run_stage(
                    investigation,
                    StageRequest(
                        "026-summarize-multimode",
                        "openstar.tess.multimode.summarize", {},
                        "025-interpret-multimode-iteration-1",
                    ),
                    software_id="integration",
                    software_version="20.7",
                )

            self.assertEqual(
                13.0, summarize.call_args.kwargs["physical_period_days"])
            self.assertEqual(
                "openstar.tess.finalize", next_request.handler_id)
            hashes = completed.stages[-1].provenance.input_hashes
            self.assertIn("resolvedCycle", hashes)
            self.assertIn("physicalInterpretation", hashes)
            self.assertIn("sourceLocalization", hashes)

    def test_prepare_rejects_tampered_localized_cycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, investigation, _ = self._investigation(
                temporary,
                tamper=lambda localization: localization.update(
                    physicalPeriodDays=12.0),
            )
            engine = build_engine(
                store, coordinator=mock.Mock(),
                poll_interval=0.0, timeout=None)
            engine.chain_stages = False
            with self.assertRaisesRegex(
                RuntimeError, "exact authoritative physical-cycle evidence"):
                engine.run_stage(
                    investigation,
                    StageRequest(
                        "023-prepare-multimode-iteration-1",
                        "openstar.tess.multimode.prepare",
                        {"iteration": 1},
                        "021-source-localization",
                    ),
                    software_id="integration",
                    software_version="20.7",
                )


if __name__ == "__main__":
    unittest.main()
