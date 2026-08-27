import copy
import math
import unittest
from unittest import mock

from openstar_investigation import sha256_json
from workflows.tess.tess_event_depth_accuracy import (
    AUDIT_HANDLER_ID, FREEZE_HANDLER_ID, _harmonic_on_original_scale, _measure,
    acquire_full_precision_photometry, audit_depth_attenuation, freeze_photometry,
    validate_freeze,
)


class Column:
    def __init__(self, values): self.values = values
    def __getitem__(self, index): return self.values[index]


class Table:
    def __init__(self, rows):
        self.rows = rows; self.colnames = list(rows[0]) if rows else []
    def __len__(self): return len(self.rows)
    def __getitem__(self, name): return Column([row.get(name) for row in self.rows])


class Values:
    def __init__(self, values, unit=None): self.value = values; self.unit = unit


class Curve:
    def __init__(self, sector, time, flux):
        self.sector = sector; self.time = Values(time); self.flux = Values(flux, "electron / s")
        self.meta = {"SECTOR": sector, "FLUX_ORIGIN": "PDCSAP_FLUX"}


class Search:
    def __init__(self, entries):
        self.entries = entries; self.table = Table([entry[0] for entry in entries])
    def __getitem__(self, value): return Search(self.entries[value])
    def download(self, quality_bitmask):
        assert quality_bitmask == "default"
        return self.entries[0][1]

class Lock:
    def __enter__(self): return self
    def __exit__(self, *args): pass


class TessEventDepthAccuracyTests(unittest.TestCase):
    def samples(self, sector, *, cadence=.01, depth=.01, duration=.12, phase=.003,
                baseline=1000., noise=.0001, integrate=False):
        times = [sector*20+i*cadence for i in range(int(12/cadence))]
        def instantaneous(t):
            primary = abs((t+1) % 2-1) <= duration/2
            opposite = abs((t-1+1) % 2-1) <= duration/2
            return baseline*(1+phase*math.sin(math.pi*t)-depth*primary-depth*.25*opposite)
        flux = []
        for i, time in enumerate(times):
            if integrate:
                value = sum(instantaneous(time+cadence*(j/20-.5)) for j in range(21))/21
            else: value = instantaneous(time)
            flux.append(value+baseline*noise*math.sin(i*1.618))
        return times, flux

    def product(self, sector, **kwargs):
        times, flux = self.samples(sector, **kwargs)
        return {"sector": sector, "time": times, "flux": flux, "cadenceSeconds": kwargs.get("cadence", .01)*86400,
                "author": "SPOC", "productIdentity": {"dataURI": f"mast:{sector}"},
                "sourceProductProvenance": {"selectionRule": "official-policy"},
                "fluxColumn": "PDCSAP_FLUX", "fluxUnits": "electron / s",
                "qualityMaskPolicy": "Lightkurve quality_bitmask='default'"}

    def binary(self, sectors=(1, 2, 3), coherent=True):
        return {"linearEphemeris": {"coherent": coherent, "referenceEpoch": 20.,
                  "refinedPeriodDays": 2., "timingSectors": list(sectors)},
                "sectorResults": [{"usable": True, "dutyCycle": .06, "sector": x} for x in sectors],
                "catalogAnswerKeyUsed": False}

    def chronology(self):
        return {"verifiedFromCompletedStages": True, "externalEvidenceStageAlreadyCompleted": False,
                "sourceAttributionReviewStageID": "review", "completedStageHandlerIDs": ["review"]}

    def frozen(self, products=None, binary=None):
        binary = binary or self.binary(); digest = sha256_json(binary)
        return freeze_photometry(products or [self.product(x) for x in (1, 2, 3)],
            binary["linearEphemeris"]["timingSectors"], binary_confirmation_sha256=digest,
            chronology_proof=self.chronology())

    def fake_search(self):
        entries = []
        for sector in (1, 2, 3):
            time, flux = self.samples(sector)
            row = {"sequence_number": sector, "author": "SPOC", "exptime": 864.,
                   "obs_id": f"obs-{sector}", "productFilename": f"sector-{sector}-lc.fits",
                   "dataURI": f"mast:{sector}", "mission": f"TESS Sector {sector}"}
            entries.append((row, Curve(sector, time, flux)))
        return Search(entries)

    def test_production_acquisition_selects_downloads_and_freezes_full_cadence(self):
        binary = self.binary()
        search = mock.Mock(return_value=self.fake_search())
        selector = lambda catalog, sector: (catalog[sector-1:sector], "SPOC", 864.)
        downloader = lambda selected, **kwargs: (selected.download(quality_bitmask="default"),
            {"author": kwargs["author"], "cadenceSeconds": kwargs["cadence_seconds"], "sector": kwargs["sector"]})
        frozen = acquire_full_precision_photometry(tic_id=123, timing_sectors=[1, 2, 3],
            binary_confirmation_sha256=sha256_json(binary), chronology_proof=self.chronology(), search=search,
            selector=selector, downloader=downloader, archive_lock=Lock())
        search.assert_called_once_with(123)
        self.assertEqual([1200]*3, [x["sampleCount"] for x in frozen["sectors"]])
        self.assertEqual(["SPOC"]*3, [x["author"] for x in frozen["sectors"]])
        self.assertTrue(frozen["frozenBeforeExternalKnownObjectQuery"])

    def test_freeze_rejects_bad_chronology_duplicates_and_nonpositive_baseline(self):
        binary = self.binary(); digest = sha256_json(binary); products = [self.product(x) for x in (1,2,3)]
        bad = self.chronology(); bad["externalEvidenceStageAlreadyCompleted"] = True
        with self.assertRaises(ValueError): freeze_photometry(products, [1,2,3], binary_confirmation_sha256=digest, chronology_proof=bad)
        with self.assertRaises(ValueError): freeze_photometry(products[:2]+[products[1]], [1,2,3], binary_confirmation_sha256=digest, chronology_proof=self.chronology())
        products[0]["flux"] = [-x for x in products[0]["flux"]]
        with self.assertRaises(ValueError): freeze_photometry(products, [1,2,3], binary_confirmation_sha256=digest, chronology_proof=self.chronology())

    def test_mutation_of_every_hash_link_and_sector_binding_is_detected(self):
        binary = self.binary(); digest = sha256_json(binary); frozen = self.frozen(binary=binary)
        validate_freeze(frozen, binary, digest)
        mutations = []
        for key in ("timeBTJDFloat64", "originalFluxFloat64", "relativeFluxFloat64"):
            value = copy.deepcopy(frozen); value["sectors"][0][key][0] += .1; mutations.append(value)
        value = copy.deepcopy(frozen); value["sectors"][0]["author"] = "TESS-SPOC"; mutations.append(value)
        value = copy.deepcopy(frozen); value["freezeSHA256"] = "0"*64; mutations.append(value)
        value = copy.deepcopy(frozen); value["sectors"][0]["sector"] = 9; mutations.append(value)
        for value in mutations:
            with self.assertRaises(ValueError): validate_freeze(value, binary, digest)
        with self.assertRaises(ValueError): validate_freeze(frozen, binary, "1"*64)

    def test_per_event_fractional_math_uncertainty_and_roundtrip(self):
        direct = _measure(list(range(20)), [1000.]*8+[990.]*4+[1000.]*8,
                          9.5, 4., [9.5])
        self.assertAlmostEqual(.01, direct["depthFractionalFlux"])
        binary = self.binary(); result = audit_depth_attenuation(self.frozen(binary=binary), binary,
            binary_confirmation_sha256=sha256_json(binary), downsampling_cap=10000)
        self.assertEqual("COMPLETE", result["status"])
        events = [e for s in result["sectorResults"] for e in s["eventResults"]]
        self.assertTrue(any(e["eventType"] == "PRIMARY" for e in events))
        self.assertTrue(any(e["eventType"] == "OPPOSITE_CONJUNCTION" for e in events))
        primary = [e for e in events if e["eventType"] == "PRIMARY" and e["fullPrecisionLocalBaseline"].get("usable")]
        self.assertGreater(len(primary), 3)
        self.assertTrue(all(e["fullPrecisionLocalBaseline"]["depthUncertaintyFractionalFlux"] >= 0 for e in primary))
        self.assertTrue(all(abs(e["attenuationFractions"]["standardizationFloat32"] or 0) < 1e-5 for e in primary))
        self.assertIn("robustUncertainty", result["crossSectorRobustSummary"]["downsampling"])

    def test_downsampling_and_real_duration_grid_attenuation(self):
        binary = self.binary(); frozen = self.frozen([self.product(x, cadence=.002, duration=.074) for x in (1,2,3)], binary)
        result = audit_depth_attenuation(frozen, binary, binary_confirmation_sha256=sha256_json(binary), downsampling_cap=300)
        events = [e for s in result["sectorResults"] for e in s["eventResults"] if e["eventType"] == "PRIMARY"]
        self.assertTrue(any(abs(e["attenuationFractions"]["downsampling"] or 0) > .01 for e in events))
        self.assertTrue(any(e["durationGridSelection"]["selectedDurationDays"] != e["establishedDurationDays"] for e in events))
        self.assertTrue(any(abs(e["attenuationFractions"]["discreteBoxDuration"] or 0) > .01 for e in events))

    def test_protected_vs_unprotected_harmonic_attenuation(self):
        times, raw = self.samples(1, phase=.01); scale = sorted(raw)[len(raw)//2]; flux = [x/scale for x in raw]
        centers = [20+2*i for i in range(7)] + [21+2*i for i in range(6)]
        masks = [any(abs(x-c) <= .12 for c in centers) for x in times]
        protected = _harmonic_on_original_scale(times, flux, 2., masks)
        unprotected = _harmonic_on_original_scale(times, flux, 2., [False]*len(times))
        p = _measure(times, protected, 20., .12, centers); u = _measure(times, unprotected, 20., .12, centers)
        self.assertGreater(p["depthFractionalFlux"], u["depthFractionalFlux"])

    def test_exposure_integration_changes_apparent_event_depth(self):
        binary = self.binary()
        instantaneous = audit_depth_attenuation(self.frozen([self.product(x, cadence=.04) for x in (1,2,3)], binary), binary,
            binary_confirmation_sha256=sha256_json(binary))
        integrated = audit_depth_attenuation(self.frozen([self.product(x, cadence=.04, integrate=True) for x in (1,2,3)], binary), binary,
            binary_confirmation_sha256=sha256_json(binary))
        def median_depth(result):
            return sorted(e["fullPrecisionLocalBaseline"]["depthFractionalFlux"] for s in result["sectorResults"] for e in s["eventResults"] if e["eventType"] == "PRIMARY" and e["fullPrecisionLocalBaseline"].get("usable"))[3]
        self.assertLess(median_depth(integrated), median_depth(instantaneous))

    def test_sector_baselines_noise_and_fail_closed_timing(self):
        binary = self.binary(); products = [self.product(1, baseline=800, noise=.0002), self.product(2, baseline=1500, noise=.0008), self.product(3, baseline=3000, noise=.001)]
        result = audit_depth_attenuation(self.frozen(products, binary), binary, binary_confirmation_sha256=sha256_json(binary))
        self.assertEqual(3, len(result["sectorResults"]))
        incoherent = self.binary(coherent=False)
        unresolved = audit_depth_attenuation({}, incoherent, binary_confirmation_sha256=sha256_json(incoherent))
        self.assertEqual("UNRESOLVED", unresolved["status"])
        insufficient = self.binary((1,)); frozen = self.frozen([self.product(1)], insufficient)
        self.assertEqual("UNRESOLVED", audit_depth_attenuation(frozen, insufficient, binary_confirmation_sha256=sha256_json(insufficient))["status"])

    def test_blind_contract_and_registered_identifiers(self):
        self.assertEqual("openstar.tess.event-depth-photometry.freeze", FREEZE_HANDLER_ID)
        self.assertEqual("openstar.tess.event-depth-attenuation.audit", AUDIT_HANDLER_ID)
        result = self.frozen()
        self.assertFalse(result["externalCatalogInformationUsed"]); self.assertFalse(result["catalogAnswerKeyUsed"])


if __name__ == "__main__": unittest.main()
