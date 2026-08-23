"""Shared isolation for tests that execute recorded science runners."""

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from openstar_science_runs import CATALOG_ENV


@contextmanager
def isolated_science_run_catalog():
    """Route default catalog access to a disposable catalog and restore the env."""
    with tempfile.TemporaryDirectory(prefix="openstar-test-science-runs-") as temporary:
        path = Path(temporary) / "science-runs.sqlite3"
        with patch.dict(os.environ, {CATALOG_ENV: str(path)}):
            yield path


class IsolatedScienceRunTestCase(unittest.TestCase):
    """TestCase base for tests that may call ``@recorded_science_run`` code."""

    def run(self, result=None):
        with isolated_science_run_catalog() as path:
            self.science_run_catalog_path = path
            return super().run(result)
