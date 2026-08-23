from __future__ import annotations

import json
import math
import tempfile
import sys
import types
import unittest
from pathlib import Path

try:
    import numpy as _numpy
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")
    _installed_numpy_stub = True
else:
    _installed_numpy_stub = not hasattr(_numpy, "ndarray")

from workflows.tess.tess_intrinsic_nonstationary import classify_target_component
from workflows.tess.tess_investigation import multisource_residual_continuation
from workflows.tess.tess_autonomy import WORKFLOW_ID, repair_obsolete_terminal_wait
from openstar_investigation import InvestigationStage, InvestigationStore
from dataclasses import replace

if _installed_numpy_stub:
    del sys.modules["numpy"]


class TessIntrinsicNonstationaryTests(unittest.TestCase):
    def boundary(self):
        # Real-shaped TIC 350519062 boundary; deliberately excludes observed powers.
        return {"classification": "TARGET_RESIDUAL_COMPONENT_DOMINANT",
                "residualModeOrigin": "TARGET_DOMINANT", "physicalMechanismResolved": False,
                "recommendedNextTest": "INTRINSIC_NONSTATIONARY_VARIABILITY_CLASSIFICATION",
                "targetComponentID": "target",
                "componentSummaries": [{"componentID": "target", "componentType": "TARGET",
                                        "independentSupportCount": 4}]}

    def preparation(self, root: Path):
        series = []
        for sector in (1, 27):
            times = [index / 20 for index in range(160)]
            values = [math.sin(2 * math.pi * 1.2 * value) for value in times]
            coefficients = root / f"target-{sector}-coefficients.json"
            dataset = root / f"target-{sector}.json"
            coefficients.write_text(json.dumps({"times": times, "coefficients": values,
                                                "componentID": "target"}))
            dataset.write_text(json.dumps({"science": {"componentID": "target"},
                                           "source": {"timeReferenceDays": 1500.0}}))
            series.append({"componentID": "target", "componentType": "TARGET", "sector": sector,
                           "combined": False, "coefficientSeriesPath": str(coefficients),
                           "datasetPath": str(dataset)})
        return {"referenceFrequency": 1.2, "preparedSeries": series}

    def test_exact_boundary_selects_new_stage(self):
        request = multisource_residual_continuation(self.boundary(), request_id="026-interpret")
        self.assertEqual("openstar.tess.intrinsic-nonstationary.analyze", request.handler_id)

    def test_other_routes_unchanged(self):
        prf = multisource_residual_continuation(
            {"recommendedNextTest": "PIXEL_RESPONSE_FUNCTION_DEBLENDING",
             "physicalMechanismResolved": False}, request_id="026-interpret")
        self.assertEqual("openstar.tess.official-spoc-prf-forward-modeling.prepare", prf.handler_id)
        for classification in ("OFFSET_RESIDUAL_COMPONENT_DOMINANT", "MIXED_RESIDUAL_COMPONENTS"):
            boundary = self.boundary(); boundary["classification"] = classification
            self.assertEqual("openstar.tess.finalize",
                             multisource_residual_continuation(boundary, request_id="026-x").handler_id)

    def test_uses_target_coefficients_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = classify_target_component(preparation=self.preparation(Path(directory)),
                                               decomposition=self.boundary())
        self.assertEqual([1, 27], result["sectorsUsed"])
        self.assertEqual("v20.12 spatially-decomposed target coefficient series", result["observable"])
        self.assertEqual(4, len(result["inputProvenance"]["preparationArtifacts"]))
        self.assertFalse(result["physicalMechanismResolved"])

    def test_missing_provenance_fails_closed(self):
        result = classify_target_component(preparation={"referenceFrequency": 1.2,
                                                        "preparedSeries": []},
                                           decomposition=self.boundary())
        self.assertEqual("INSUFFICIENT_TARGET_COMPONENT_TEMPORAL_EVIDENCE", result["classification"])
        self.assertTrue(result["failClosedReasons"])

    def test_boundary_is_not_broadened(self):
        for key, value in (("residualModeOrigin", "TIME_VARIABLE_OR_BLENDED"),
                           ("physicalMechanismResolved", True)):
            boundary = self.boundary(); boundary[key] = value
            self.assertEqual("openstar.tess.finalize",
                             multisource_residual_continuation(boundary, request_id="026-x").handler_id)

    def test_complete_boundary_reopens_without_replacing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory))
            investigation = store.create("tess-real-shaped", WORKFLOW_ID, "20.2")
            stage = InvestigationStage("026-interpret-multi-source-residual",
                "openstar.tess.multi-source-residual.interpret", "COMPLETE", "025-run", {},
                result=self.boundary())
            investigation = replace(investigation, stages=(stage,))
            store.save(investigation)
            investigation = store.set_control_state(investigation, status="COMPLETE",
                control_state={"schedulerAction": "INVESTIGATION_COMPLETE"})
            old_stages = investigation.stages
            repaired = repair_obsolete_terminal_wait(store, investigation)
        self.assertEqual("RUNNING", repaired.status)
        self.assertEqual(old_stages, repaired.stages)
        self.assertEqual("openstar.tess.intrinsic-nonstationary.analyze",
                         repaired.metadata["controlState"]["selectedExperiment"]["handler_id"])


if __name__ == "__main__":
    unittest.main()
