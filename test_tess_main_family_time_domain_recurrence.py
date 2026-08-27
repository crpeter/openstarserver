import math
import unittest

import pytest
np = pytest.importorskip("numpy", reason="numpy scientific runtime is not installed")

from workflows.tess.tess_main_family_time_domain_recurrence import (
    METHOD, ROTATION_MULTICYCLE, UNRESOLVED, analyze_sector,
    analyze_time_domain_recurrence, combine_sector_results,
)
from workflows.tess.tess_investigation import astrophysical_interpretation_continuation


class MainFamilyTimeDomainRecurrenceTests(unittest.TestCase):
    P = 3.600708338567666
    FAMILY = {"available": True, "representativeRawPeriodDays": 7.546257528330875,
              "possibleDoubleCycleDays": 15.09251505666175,
              "physicalCycleResolved": False, "supportingSectorIDs": [94, 95]}

    def sector(self, sector, alternating=False, gap=False):
        time = np.arange(0, 40, .04)
        cycle = np.floor(time/self.P).astype(int)
        flux = np.sin(2*math.pi*time/self.P)
        if alternating:
            flux += .7*((-1.0)**cycle)*np.cos(4*math.pi*time/self.P)
        if gap:
            keep = ~(((time > 8)&(time < 11)) | ((time > 24)&(time < 27)))
            time, flux = time[keep], flux[keep]
        return {"sectorID":sector,"time":time.tolist(),"flux":flux.tolist()}

    def test_gap_aware_acf_is_deterministic_and_recovers_rotation(self):
        a = analyze_sector(**{"time":self.sector(1, gap=True)["time"],
            "flux":self.sector(1, gap=True)["flux"]}, sector_id=1,
            rotation_period_days=self.P, possible_double_days=self.FAMILY["possibleDoubleCycleDays"])
        b = analyze_sector(self.sector(1,gap=True)["time"],self.sector(1,gap=True)["flux"],
            sector_id=1,rotation_period_days=self.P,possible_double_days=self.FAMILY["possibleDoubleCycleDays"])
        self.assertEqual(a,b)
        self.assertAlmostEqual(self.P,a["rotationRecurrencePeak"]["lagDays"],delta=.2)
        self.assertIn("perResamplePeakLocationsDays",a["rotationJackknife"])

    def test_alternating_morphology_preserves_pairs(self):
        result=analyze_time_domain_recurrence([self.sector(94,True),self.sector(95,True)],
            rotation_period_days=self.P,rotation_classification="ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED",
            main_photometric_family=self.FAMILY)
        pairs=result["cycleRecurrenceSectorResults"][0]["cyclePairMeasurements"]
        self.assertGreater(len(pairs["2"]),3)
        self.assertGreater(np.median([x["correlation"] for x in pairs["2"]]),
                           np.median([x["correlation"] for x in pairs["1"]]))
        self.assertIn(result["classification"],(ROTATION_MULTICYCLE,UNRESOLVED))
        self.assertFalse(result["physicalCycleResolved"])

    def test_replication_required(self):
        synthetic={"sectorID":94,"rotationRecurrencePeak":{"lagDays":3.6,"localPeakWidthDays":.2,
            "uncertaintyEstimate":{"intervalDays":[3.55,3.65]}},
            "acfPeaks":[{"lagDays":7.2,"peakCorrelation":.8,"localPeakWidthDays":.2}],
            "cycleSeparationSummaries":{"1":{"pairCount":3,"medianCorrelation":0},
                "2":{"pairCount":3,"medianCorrelation":.9},"4":{"pairCount":0,"medianCorrelation":None}}}
        combined=combine_sector_results([synthetic],rotation_period_days=self.P,
            family_period_days=7.2,possible_double_days=14.4)
        self.assertEqual(UNRESOLVED,combined["classification"])

    def test_short_sector_fails_closed_for_double(self):
        s=self.sector(94); keep=np.asarray(s["time"])<18
        result=analyze_sector(np.asarray(s["time"])[keep],np.asarray(s["flux"])[keep],
            sector_id=94,rotation_period_days=self.P,possible_double_days=15.09)
        self.assertFalse(result["canConstrainPossibleDouble"])

    def test_historical_values_are_dynamic_and_future_route_precedes_finalizer(self):
        summary={"targetResidualMechanismResolved":True,
            "classification":"ROTATIONAL_ACTIVE_REGION_MODULATION_SUPPORTED",
            "mainPhotometricFamily":dict(self.FAMILY)}
        request=astrophysical_interpretation_continuation(summary,request_id="031-target-residual-astrophysical-interpretation")
        self.assertEqual("032-main-family-time-domain-recurrence",request.id)
        self.assertNotEqual("openstar.tess.finalize",request.handler_id)
        self.assertEqual([1.0,18.0],METHOD["lagSearchDays"])


if __name__ == "__main__": unittest.main()
