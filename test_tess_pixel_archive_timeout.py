from __future__ import annotations

import socket
import sys
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

# The timeout boundary itself does not perform array work.  Keep this transport
# regression runnable in the repository's dependency-light unit-test lane.
try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")

from workflows.tess.tess_localization import _download_tpf
from workflows.tess.tess_sector_archive import (
    TESS_ARCHIVE_TIMEOUT_SECONDS,
    TessArchiveTransientError,
    _archive_timeout_seconds,
    configure_tess_archive_timeout,
)


class _Table:
    colnames = ()

    def __init__(self, count=1):
        self.count = count

    def __len__(self):
        return self.count


class _Search:
    def __init__(self, *, download=None, count=1):
        self.table = _Table(count)
        self._download = download

    def __getitem__(self, _item):
        return self

    def download(self, **_kwargs):
        if isinstance(self._download, BaseException):
            raise self._download
        return self._download


class TessPixelArchiveTimeoutTests(unittest.TestCase):
    def test_timeout_config_uses_dependency_specific_types(self):
        class _StrictIntegerConfig:
            def __init__(self):
                self._timeout = None

            @property
            def timeout(self):
                return self._timeout

            @timeout.setter
            def timeout(self, value):
                if type(value) is not int:
                    raise TypeError(f"{value!r} is not an integer timeout")
                self._timeout = value

        mast_conf = _StrictIntegerConfig()
        observations = types.SimpleNamespace(TIMEOUT=None)
        tesscut = types.SimpleNamespace(TIMEOUT=None)
        mast = types.ModuleType("astroquery.mast")
        mast.conf, mast.Observations, mast.Tesscut = mast_conf, observations, tesscut
        astroquery = types.ModuleType("astroquery")
        astroquery.mast = mast
        data_conf = types.SimpleNamespace(remote_timeout=None)
        data = types.ModuleType("astropy.utils.data")
        data.conf = data_conf
        utils = types.ModuleType("astropy.utils")
        utils.data = data
        astropy = types.ModuleType("astropy")
        astropy.utils = utils
        with mock.patch.dict(sys.modules, {
            "astroquery": astroquery, "astroquery.mast": mast,
            "astropy": astropy, "astropy.utils": utils,
            "astropy.utils.data": data,
        }):
            configure_tess_archive_timeout()

        self.assertIs(type(TESS_ARCHIVE_TIMEOUT_SECONDS), int)
        self.assertEqual(60, mast_conf.timeout)
        self.assertIs(type(observations.TIMEOUT), int)
        self.assertIs(type(tesscut.TIMEOUT), int)
        self.assertIs(type(data_conf.remote_timeout), float)

    def test_timeout_value_rejects_float_syntax_and_nonpositive_values(self):
        for value in ("60.0", "0", "-1", "nan", ""):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "must be a positive integer"
            ):
                _archive_timeout_seconds(value)

    def _modules(self, lightkurve):
        class _Degree:
            def __rmul__(self, value):
                return value

        coordinates = types.ModuleType("astropy.coordinates")
        coordinates.SkyCoord = lambda *args, **kwargs: object()
        units = types.ModuleType("astropy.units")
        units.deg = _Degree()
        astropy = types.ModuleType("astropy")
        astropy.coordinates = coordinates
        astropy.units = units
        return mock.patch.dict(sys.modules, {
            "lightkurve": lightkurve,
            "astropy": astropy,
            "astropy.coordinates": coordinates,
            "astropy.units": units,
        })

    def _call(self, lightkurve, selected=None):
        with self._modules(lightkurve), \
             mock.patch("workflows.tess.tess_localization.configure_tess_archive_timeout"), \
             mock.patch("workflows.tess.tess_localization._select_official_tpf", return_value=selected):
            return _download_tpf(tic_id=1, sector=1, ra_deg=1.0, dec_deg=2.0)

    def test_official_search_timeout_is_bounded_and_transient(self):
        def timeout(*args, **kwargs):
            raise socket.timeout("read deadline")
        lk = types.SimpleNamespace(search_targetpixelfile=timeout)
        started = time.monotonic()
        with self.assertRaises(TessArchiveTransientError):
            self._call(lk)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_official_download_timeout_is_transient(self):
        lk = types.SimpleNamespace(search_targetpixelfile=lambda *a, **k: object())
        with self.assertRaises(TessArchiveTransientError):
            self._call(lk, (_Search(download=socket.timeout("read deadline")), "SPOC", 120.0))

    def test_tesscut_search_timeout_is_transient(self):
        lk = types.SimpleNamespace(
            search_targetpixelfile=lambda *a, **k: object(),
            search_tesscut=lambda *a, **k: (_ for _ in ()).throw(socket.timeout("deadline")),
        )
        with self.assertRaises(TessArchiveTransientError):
            self._call(lk)

    def test_tesscut_download_timeout_is_transient(self):
        lk = types.SimpleNamespace(
            search_targetpixelfile=lambda *a, **k: object(),
            search_tesscut=lambda *a, **k: _Search(download=socket.timeout("deadline")),
        )
        with self.assertRaises(TessArchiveTransientError):
            self._call(lk)

    def test_permanent_download_error_is_not_mislabeled(self):
        lk = types.SimpleNamespace(search_targetpixelfile=lambda *a, **k: object())
        with self.assertRaisesRegex(ValueError, "bad product"):
            self._call(lk, (_Search(download=ValueError("bad product")), "SPOC", 120.0))

    def test_successful_official_and_tesscut_paths_are_preserved(self):
        official = object()
        lk = types.SimpleNamespace(search_targetpixelfile=lambda *a, **k: object())
        result, provenance = self._call(lk, (_Search(download=official), "SPOC", 120.0))
        self.assertIs(official, result)
        self.assertEqual("OFFICIAL_TPF", provenance["sourceType"])

        cutout = object()
        lk = types.SimpleNamespace(
            search_targetpixelfile=lambda *a, **k: object(),
            search_tesscut=lambda *a, **k: _Search(download=cutout),
        )
        result, provenance = self._call(lk)
        self.assertIs(cutout, result)
        self.assertEqual("TESSCUT_FFI", provenance["sourceType"])

    def test_two_concurrent_timeouts_return_both_threads(self):
        lk = types.SimpleNamespace(
            search_targetpixelfile=lambda *a, **k: (_ for _ in ()).throw(socket.timeout("deadline"))
        )
        def run():
            with self.assertRaises(TessArchiveTransientError):
                self._call(lk)
            return True
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run) for _ in range(2)]
            self.assertEqual([True, True], [item.result(timeout=1) for item in futures])


if __name__ == "__main__":
    unittest.main()
