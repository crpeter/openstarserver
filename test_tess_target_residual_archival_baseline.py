import math
import unittest
from types import SimpleNamespace
from workflows.tess.tess_target_residual_archival_baseline import (
    FREQUENCIES_PER_WORK_UNIT, HARMONIC_ORDERS, TOTAL_FREQUENCIES,
    adjudicate_sector, adjudicate_target, frozen_search_grid,
    prewhiten_established_family, previously_consumed_tess_sectors,
)

class TessTargetResidualArchivalBaselineTests(unittest.TestCase):
    def test_explicit_consumption_excludes_primary_and_known_materializations(self):
        stages=[SimpleNamespace(id="000-prepare",handler_id="openstar.tess.prepare-target",status="COMPLETE",result={"sector":1,"unrelatedNumber":99}),
            SimpleNamespace(id="010-independent",handler_id="openstar.tess.independent.prepare",status="COMPLETE",result={"preparedSectors":[{"sector":4},{"sector":7}]}),
            SimpleNamespace(id="noise",handler_id="unknown",status="COMPLETE",result={"sector":88})]
        consumed=previously_consumed_tess_sectors(stages)
        self.assertEqual([1,4,7],list(consumed)); self.assertNotIn(99,consumed); self.assertNotIn(88,consumed)
        self.assertEqual("000-prepare",consumed[1][0]["stageID"])

    def test_frozen_grid_is_exact_and_targeted(self):
        grid=frozen_search_grid(.1)
        self.assertAlmostEqual(.08,grid["minimumFrequency"]); self.assertAlmostEqual(.12,grid["maximumFrequency"])
        self.assertEqual(TOTAL_FREQUENCIES,8192); self.assertEqual(FREQUENCIES_PER_WORK_UNIT,2048)

    def _candidate(self, sector=1, frequency=.1, origin=0, **updates):
        value={"sector":sector,"candidateFrequency":frequency,"candidateFrequencyConfidenceInterval":{"lower":frequency-.002,"upper":frequency+.002},
            "periodStatus":"RELIABLE","periodConfidence":"high","baselineDays":27,"cycleCoverage":2.7,"boundaryHit":False,"originalTimeOriginDays":origin}
        value.update(updates); return value

    def test_recurrence_requires_all_preregistered_rules(self):
        supported=adjudicate_sector(self._candidate(),(.09,.11)); self.assertTrue(supported["supportsHistoricalResidualFamily"])
        for changed in ({"boundaryHit":True},{"periodStatus":"UNRELIABLE"},{"cycleCoverage":1.49},{"candidateFrequencyConfidenceInterval":None},{"candidateFrequency":.12}):
            self.assertFalse(adjudicate_sector(self._candidate(**changed),(.09,.11))["supportsHistoricalResidualFamily"])
        limited=adjudicate_sector(self._candidate(candidateFrequencyConfidenceInterval={"lower":.05,"upper":.15}),(.09,.11))
        self.assertEqual("RESOLUTION_LIMITED",limited["recurrenceClassification"])

    def test_target_classifications_and_invariants(self):
        rows=[adjudicate_sector(self._candidate(i,.1,(i-1)*400),(.09,.11)) for i in range(1,4)]
        result=adjudicate_target(rows); self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_RECURRENCE_SUPPORTED",result["classification"])
        for key in ("physicalMechanismResolved","sourceAttributionResolved","crossSectorPhaseUsed","historicalResidualDriftExtrapolated"): self.assertFalse(result[key])
        self.assertEqual("ARCHIVAL_TARGET_RESIDUAL_BASELINE_INSUFFICIENT",adjudicate_target(rows[:2])["classification"])
        with self.assertRaises(ValueError): adjudicate_target(rows+[rows[0]])

if __name__ == "__main__": unittest.main()
