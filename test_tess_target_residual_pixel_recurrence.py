import unittest

try:
    from workflows.tess.tess_target_residual_pixel_recurrence import (
        classify_centroid, frozen_catalog_hypotheses, interpret_sectors,
    )
except ModuleNotFoundError as error:
    classify_centroid = None
    IMPORT_ERROR = error


@unittest.skipIf(classify_centroid is None, "optional numerical dependencies unavailable")
class TargetResidualPixelRecurrenceTests(unittest.TestCase):
    def test_unique_target_and_catalog_localizations(self):
        hypotheses=[{"sourceID":"target","x":1.0,"y":1.0},
                    {"sourceID":"catalog","x":4.0,"y":4.0}]
        self.assertEqual("target",classify_centroid((1.0,1.0),hypotheses,.1,100)["preferredSource"])
        self.assertEqual("catalog",classify_centroid((4.0,4.0),hypotheses,.1,100)["preferredSource"])

    def test_ambiguity_uncertainty_and_snr_gates(self):
        hypotheses=[{"sourceID":"a","x":1.0,"y":1.0},{"sourceID":"b","x":1.2,"y":1.0}]
        self.assertIsNone(classify_centroid((1.1,1),hypotheses,.01,100)["preferredSource"])
        self.assertIsNone(classify_centroid((1,1),hypotheses,1.0,100)["preferredSource"])
        self.assertIsNone(classify_centroid((1,1),hypotheses,.01,0)["preferredSource"])

    def test_cross_sector_resolution_threshold_and_switching(self):
        rows=[{"sector":n,"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":"target"} for n in range(3)]
        result=interpret_sectors(rows,"target")
        self.assertTrue(result["sourceAttributionResolved"])
        self.assertFalse(result["crossSectorPhaseUsed"])
        self.assertFalse(result["historicalResidualDriftExtrapolated"])
        self.assertFalse(interpret_sectors(rows[:2],"target")["sourceAttributionResolved"])
        switched=rows[:2]+[{"sector":3+n,"classification":"UNIQUE_SOURCE_SUPPORTED","preferredSource":"other"} for n in range(2)]
        self.assertEqual("PIXEL_RECURRENCE_SOURCE_SWITCHING_OR_BLEND",interpret_sectors(switched,"target")["classification"])

    def test_catalog_hypotheses_are_frozen_without_post_centroid_selection(self):
        identity={"catalogSources":[{"ticID":2,"gaiaSourceID":3,"raDeg":2.0,"decDeg":3.0}]}
        frozen=frozen_catalog_hypotheses(identity,tic_id=1,ra_deg=1.0,dec_deg=1.0)
        self.assertEqual(["TIC-1","TIC-2"],[x["sourceID"] for x in frozen])


if __name__ == "__main__": unittest.main()
