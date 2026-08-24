"""Safety policy for caller-selected durable OpenStar state roots."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def temporary_state_roots() -> tuple[Path, ...]:
    """Return distinct OS and environment-configured temporary roots."""
    standard_candidates: list[str | Path] = [
        "/tmp", "/private/tmp", "/var/tmp", "/dev/shm", tempfile.gettempdir(),
    ]
    environment_candidates = [
        value for name in ("TMPDIR", "TMP", "TEMP")
        if (value := os.environ.get(name))
    ]
    roots: list[Path] = []
    for candidate in standard_candidates:
        root = _resolved(candidate)
        if root.exists() and root not in roots:
            roots.append(root)
    # Environment-configured roots are policy declarations, not observations
    # about the current filesystem. Keep them even when they do not yet exist,
    # so a caller cannot create one through the durable-state guard.
    for candidate in environment_candidates:
        root = _resolved(candidate)
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def is_temporary_state_path(path: str | Path) -> bool:
    """Return whether *path* resolves to a temporary root or its descendant."""
    resolved = _resolved(path)
    return any(resolved == root or resolved.is_relative_to(root)
               for root in temporary_state_roots())


def require_durable_state_path(
    path: str | Path,
    *,
    allow_temporary_state: bool = False,
    label: str = "state directory",
) -> Path:
    """Resolve a durable state path, rejecting temporary storage by default."""
    resolved = _resolved(path)
    if not allow_temporary_state and is_temporary_state_path(resolved):
        raise RuntimeError(
            "Refusing durable OpenStar science state in temporary storage: "
            f"{resolved} ({label}). Use a durable filesystem location or explicitly "
            "opt in with --allow-temporary-state for disposable smoke/test work."
        )
    return resolved
