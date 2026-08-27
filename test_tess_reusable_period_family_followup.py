import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from openstar_investigation import InvestigationStage, InvestigationStore
from openstar_targets import InvestigationTarget
from workflows.tess.period_family_followup import select_untouched_sectors
from workflows.tess.tess_autonomy import plan_tess_branches
from workflows.tess.external_long_baseline import (
    ASASSNSkyPatrolProvider, MalformedProviderData, ProviderUnavailable,
    analyze_seasonal_coherence, run_external_experiment,
)


class Transport:
    def __init__(self, rows, available=True): self.rows=rows; self.available=available
    def coverage(self, target): return {"available": self.available, "product":"asas-sn-lightcurve-v1"}
    def acquire(self, target, request): return json.dumps(self.rows).encode()


def rows(evolving=False):
    out=[]; period=4.55
    for season in range(4):
        for i in range(35):
            t=season*365.25+i*0.71
            phase=(0.35*season if evolving else 0)
            out.append({"time":t,"flux":math.sin(2*math.pi*t/period+phase),
                        "uncertainty":0.15,"band":"g","quality":"GOOD"})
    return out


class ReusableFollowupTests(unittest.TestCase):
    def test_sector_selection_is_deterministic_and_excludes_consumed(self):
        catalog=[{"sector":s,"epoch":e,"product":"SPOC_LIGHTCURVE","cadenceSeconds":120,"available":True}
                 for s,e in [(20,"B"),(3,"A"),(40,"C"),(4,"A")]]
        first=select_untouched_sectors(catalog,[3])
        self.assertEqual(first["selectedSectors"],[4,20,40])
        self.assertEqual(first,select_untouched_sectors(reversed(catalog),[3]))
        self.assertFalse(first["fluxInspectedDuringSelection"])
        self.assertEqual(first["rejectedSectors"][0]["reason"],"already-consumed-by-time-domain-observable")
        self.assertEqual(select_untouched_sectors(catalog,[3,40])["status"],"INSUFFICIENT_EPOCH_COVERAGE")

    def test_three_semantic_autonomy_routes_and_quiescence(self):
        target=InvestigationTarget("x","inv","openstar.workflow.tess-investigation.v1","20.2",{})
        routes={
          "PERIOD_FAMILY_DIFFERENCE_IMAGE_LOCALIZATION":"period-family-difference-imaging.prepare",
          "UNTOUCHED_SECTOR_TIME_DOMAIN_EVOLUTION":"period-family-time-domain-evolution.prepare",
          "ADDITIONAL_LONG_BASELINE_TIME_DOMAIN_DATA":"external-long-baseline.analyze"}
        for n,(trigger,suffix) in enumerate(routes.items()):
            result={"recommendedNextTest":trigger,"periodFamilyResolved":False,
                    "claimDecision":{"claim":"HUMAN_REVIEW_REQUIRED"}}
            stage=InvestigationStage(f"00{n+1}-e","science","COMPLETE",None,{},result=result,stop=True)
            from openstar_investigation import Investigation
            inv=Investigation(f"different-{n}",target.workflow_id,"20.2","RUNNING","now","now",{},(stage,))
            branch=plan_tess_branches(inv,target)[0]
            self.assertTrue(branch.experiment.handler_id.endswith(suffix))
            quiet=replace(inv,status="QUIESCENT_AWAITING_DATA")
            # Upgrade does not choose any of the new routes.
            self.assertFalse(any(b.experiment.handler_id.endswith(suffix) for b in plan_tess_branches(quiet,target)))

    def test_asassn_success_unavailable_malformed_and_contamination(self):
        with tempfile.TemporaryDirectory() as d:
            provider=ASASSNSkyPatrolProvider(Transport(rows()))
            result=run_external_experiment(target={"raDeg":1,"decDeg":2}, family_window=[4.50,4.60],
                neighbors=[],providers=[provider],artifact_root=Path(d))
            self.assertEqual(result["classification"],"EXTERNAL_STABLE_CLOCK_SUPPORTED")
            self.assertFalse(result["lombScarglePerformed"])
            self.assertTrue(Path(result["rawResponsePath"]).exists())
        with self.assertRaises(ProviderUnavailable): ASASSNSkyPatrolProvider().coverage({})
        with self.assertRaises(MalformedProviderData): ASASSNSkyPatrolProvider(Transport([])).parse(b"bad")
        with tempfile.TemporaryDirectory() as d:
            result=run_external_experiment(target={},family_window=[4.5,4.6],
                neighbors=[{"separationArcsec":2,"fluxFraction":.5}],providers=[provider],artifact_root=Path(d))
            self.assertEqual(result["classification"],"EXTERNAL_CONTAMINATION_AMBIGUOUS")

    def test_stable_evolving_and_insufficient_science(self):
        stable=analyze_seasonal_coherence(rows(),[4.5,4.6])
        evolving=analyze_seasonal_coherence(rows(True),[4.5,4.6])
        insufficient=analyze_seasonal_coherence(rows()[:20],[4.5,4.6])
        self.assertEqual(stable["classification"],"EXTERNAL_STABLE_CLOCK_SUPPORTED")
        self.assertIn(evolving["classification"],{"EXTERNAL_EVOLVING_RECURRENCE_SUPPORTED","EXTERNAL_RECURRENCE_NOT_REPLICATED"})
        self.assertEqual(insufficient["classification"],"EXTERNAL_DATA_INSUFFICIENT")
        self.assertGreater(stable["periodUncertaintyDays"],0)

if __name__ == '__main__': unittest.main()
