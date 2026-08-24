from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from openstar_path_relocation import HistoricalPathResolver


class HistoricalPathResolverTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".relocation-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_exact_descendant_suffix_unmapped_and_original_unchanged(self):
        destination = self.root / "durable"
        destination.mkdir()
        old = Path("/historical/missing/openstar")
        original = str(old / "a" / "b.json")
        resolver = HistoricalPathResolver({old: destination})
        self.assertEqual(destination.resolve(), resolver.resolve(old))
        self.assertEqual((destination / "a/b.json").resolve(), resolver.resolve(original))
        self.assertEqual(original, str(old / "a/b.json"))
        self.assertEqual(Path("/unmapped/file").resolve(), resolver.resolve("/unmapped/file"))

    def test_longest_prefix_and_multiple_roots(self):
        first, nested, shallow = self.root / "first", self.root / "nested", self.root / "shallow"
        for path in (first, nested, shallow): path.mkdir()
        resolver = HistoricalPathResolver([
            ("/history/deep", first), ("/history/deep/specific", nested),
            ("/history/shallow", shallow)])
        self.assertEqual(nested / "x", resolver.resolve("/history/deep/specific/x"))
        self.assertEqual(shallow / "x", resolver.resolve("/history/shallow/x"))

    def test_duplicate_missing_unsafe_and_overlapping_rejected(self):
        destination = self.root / "durable"; destination.mkdir()
        with self.assertRaises(ValueError):
            HistoricalPathResolver([("/old", destination), ("/old/../old", destination)])
        with self.assertRaises(ValueError): HistoricalPathResolver({"/old": self.root / "missing"})
        with self.assertRaises(RuntimeError): HistoricalPathResolver({"/old": Path("/tmp")})
        with self.assertRaises(ValueError): HistoricalPathResolver({destination: destination / "child"})

    def test_no_fallback_no_writes_and_escape_rejected(self):
        destination = self.root / "durable"; destination.mkdir()
        resolver = HistoricalPathResolver({"/historical/root": destination})
        before = tuple(self.root.rglob("*"))
        self.assertEqual(destination / "missing", resolver.resolve("/historical/root/missing"))
        self.assertEqual(before, tuple(self.root.rglob("*")))
        with self.assertRaises(ValueError): resolver.resolve("/historical/root/../escape")


if __name__ == "__main__":
    unittest.main()
