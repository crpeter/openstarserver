from __future__ import annotations

import socket
import sys
import types
import urllib.error
import unittest
from unittest import mock

# Transport and early-error runner tests do not execute numerical work. Keep
# them in the dependency-light lane used by the archive timeout tests.
try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")

from workflows.tess import tess_spoc_prf
from workflows.tess.tess_prf_refinement import run_prf_deblending
from workflows.tess.tess_sector_archive import (
    TESS_ARCHIVE_TIMEOUT_SECONDS,
    TessArchiveTransientError,
)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"payload"


class TessSpocPrfTransportTests(unittest.TestCase):
    def test_http_get_uses_shared_timeout(self):
        with mock.patch.object(
            tess_spoc_prf.urllib.request, "urlopen", return_value=_Response()
        ) as urlopen:
            self.assertEqual(b"payload", tess_spoc_prf._http_get("https://example.test/prf"))
        self.assertEqual(TESS_ARCHIVE_TIMEOUT_SECONDS, urlopen.call_args.kwargs["timeout"])

    def test_typed_transient_transport_errors_are_wrapped(self):
        errors = (
            urllib.error.URLError(socket.timeout("timed out")),
            socket.timeout("read operation timed out"),
            urllib.error.HTTPError("https://example.test", 503, "busy", {}, None),
        )
        for error in errors:
            with self.subTest(error=error), mock.patch.object(
                tess_spoc_prf.urllib.request, "urlopen", side_effect=error
            ), self.assertRaises(TessArchiveTransientError):
                tess_spoc_prf._http_get("https://example.test/prf")

    def test_permanent_http_error_is_not_wrapped(self):
        error = urllib.error.HTTPError("https://example.test", 404, "missing", {}, None)
        with mock.patch.object(
            tess_spoc_prf.urllib.request, "urlopen", side_effect=error
        ), self.assertRaises(urllib.error.HTTPError):
            tess_spoc_prf._http_get("https://example.test/prf")

    def test_prf_runner_preserves_scientific_errors_per_sector(self):
        preparation = {"artifactRoot": "/tmp", "sectors": [2], "ticID": 1,
                       "targetSky": {"raDeg": 1.0, "decDeg": 2.0}}
        with mock.patch(
            "workflows.tess.tess_prf_refinement._download_tpf",
            side_effect=RuntimeError("malformed FITS"),
        ):
            run = run_prf_deblending(preparation)
        self.assertEqual([], run["sectorResults"])
        self.assertEqual("RuntimeError: malformed FITS", run["errors"][0]["error"])

    def test_prf_runner_reraises_archive_transient(self):
        preparation = {"artifactRoot": "/tmp", "sectors": [2], "ticID": 1,
                       "targetSky": {"raDeg": 1.0, "decDeg": 2.0}}
        with mock.patch(
            "workflows.tess.tess_prf_refinement._download_tpf",
            side_effect=TessArchiveTransientError("temporary"),
        ), self.assertRaises(TessArchiveTransientError):
            run_prf_deblending(preparation)


if __name__ == "__main__":
    unittest.main()
