import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openstar_state_storage import (
    is_temporary_state_path, require_durable_state_path, temporary_state_roots,
)
from run_openstar_autonomous_tess import run_autonomous_tess
from run_openstar_tess_ranked_followup import run_tess_ranked_followup
from run_openstar_tess_sector_ranking import run_tess_sector_ranking
from run_openstar_tess_sector_sweep import run_tess_sector_sweep


class StateStorageSafetyTests(unittest.TestCase):
    def test_accepts_durable_and_tmp_named_sibling(self):
        durable = Path.home() / "OpenStarScience" / "state"
        sibling = Path.home() / "tmp-results" / "state"
        self.assertEqual(durable.resolve(), require_durable_state_path(durable))
        self.assertFalse(is_temporary_state_path(sibling))

    def test_rejects_standard_temporary_roots_and_children(self):
        candidates = [
            Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"), Path("/dev/shm"),
            Path(tempfile.gettempdir()),
        ]
        for root in candidates:
            if root.exists():
                with self.subTest(root=root):
                    with self.assertRaisesRegex(RuntimeError, "Refusing durable OpenStar"):
                        require_durable_state_path(root)
                    with self.assertRaises(RuntimeError):
                        require_durable_state_path(root / "openstar-new-state")

    def test_nonexistent_environment_temporary_roots_are_rejected_without_creation(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as durable_parent:
            for name in ("TMPDIR", "TMP", "TEMP"):
                root = Path(durable_parent) / f"not-yet-created-{name.lower()}"
                child = root / "openstar-sector"
                self.assertFalse(root.exists())
                with self.subTest(name=name), patch.dict(
                    os.environ, {name: str(root)}, clear=False
                ):
                    with self.assertRaisesRegex(RuntimeError, "Refusing durable OpenStar"):
                        require_durable_state_path(child)
                self.assertFalse(root.exists())
                self.assertFalse(child.exists())

    def test_environment_temporary_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("TMPDIR", "TMP", "TEMP"):
                with self.subTest(name=name), patch.dict(os.environ, {name: temporary}, clear=False):
                    root = Path(temporary).resolve()
                    self.assertIn(root, temporary_state_roots())
                    with self.assertRaises(RuntimeError):
                        require_durable_state_path(root / "state")

    def test_explicit_override_accepts_temporary_path(self):
        path = Path(tempfile.gettempdir()) / "disposable-openstar-state"
        self.assertEqual(path.resolve(), require_durable_state_path(
            path, allow_temporary_state=True))

    def test_symlink_into_temporary_root_is_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            with tempfile.TemporaryDirectory(dir=Path.home()) as durable_parent:
                alias = Path(durable_parent) / "alias"
                alias.symlink_to(temporary, target_is_directory=True)
                target = alias / "not-created"
                with self.assertRaises(RuntimeError):
                    require_durable_state_path(target)
                self.assertFalse((temporary / "not-created").exists())

    def test_production_runners_reject_before_creating_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases = (
                ("sector sweep", base / "sweep", lambda path: run_tess_sector_sweep(
                    1, "unused", path)),
                ("sector ranking", base / "ranking", lambda path: run_tess_sector_ranking(
                    1, path)),
                ("autonomous TESS", base / "autonomous", lambda path: run_autonomous_tess(
                    [], "unused", path)),
            )
            with patch.dict(os.environ, {
                "OPENSTAR_SCIENCE_RUN_CATALOG": str(base / "catalog.sqlite3")
            }):
                for label, target, invoke in cases:
                    with self.subTest(label=label), self.assertRaises(RuntimeError):
                        invoke(target)
                    self.assertFalse(target.exists())

    def test_ranked_followup_rejects_either_temporary_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            durable_shallow = Path.home() / "OpenStarScience" / "shallow-not-created"
            durable_deep = Path.home() / "OpenStarScience" / "deep-not-created"
            with patch.dict(os.environ, {
                "OPENSTAR_SCIENCE_RUN_CATALOG": str(base / "catalog.sqlite3")
            }):
                for shallow, deep in (
                    (base / "unsafe-shallow", durable_deep),
                    (durable_shallow, base / "unsafe-deep"),
                ):
                    with self.subTest(shallow=shallow, deep=deep), self.assertRaises(RuntimeError):
                        run_tess_ranked_followup(1, shallow, deep, "unused", 1)
                    self.assertFalse(base.joinpath("unsafe-shallow").exists())
                    self.assertFalse(base.joinpath("unsafe-deep").exists())


if __name__ == "__main__":
    unittest.main()
