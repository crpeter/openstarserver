"""Explicit, read-only relocation of paths recorded in immutable history."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from openstar_state_storage import require_durable_state_path


class HistoricalPathResolver:
    """Resolve persisted paths through an immutable set of exact root mappings."""

    def __init__(self, mappings: Mapping[str | Path, str | Path] | Iterable[tuple[str | Path, str | Path]] = ()):
        entries = list(mappings.items() if isinstance(mappings, Mapping) else mappings)
        normalized: list[tuple[Path, Path]] = []
        sources: set[Path] = set()
        for raw_source, raw_destination in entries:
            source = Path(raw_source).expanduser().resolve()
            destination = Path(raw_destination).expanduser().resolve()
            if source in sources:
                raise ValueError(f"duplicate historical source root: {source}")
            sources.add(source)
            if not destination.exists() or not destination.is_dir():
                raise ValueError(f"historical relocation destination must be an existing directory: {destination}")
            require_durable_state_path(destination, label="historical relocation destination")
            normalized.append((source, destination))
        for source, destination in normalized:
            if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
                raise ValueError(f"historical source and destination trees overlap: {source} -> {destination}")
        self._mappings = tuple(sorted(normalized, key=lambda item: len(item[0].parts), reverse=True))

    @property
    def mappings(self) -> tuple[tuple[Path, Path], ...]:
        return self._mappings

    def resolve(self, persisted_path: str | Path) -> Path:
        """Return the mapped read path without probing, creating, or changing files."""
        raw = Path(persisted_path).expanduser()
        candidate = raw.resolve()
        for source, destination in self._mappings:
            # A lexical descendant containing ``..`` must not turn into an
            # unmapped read outside the configured destination.
            try:
                lexical_suffix = raw.relative_to(source)
            except ValueError:
                lexical_suffix = None
            if lexical_suffix is not None and candidate != source and not candidate.is_relative_to(source):
                raise ValueError(f"historical path escapes configured source root: {persisted_path}")
            if candidate == source or candidate.is_relative_to(source):
                suffix = candidate.relative_to(source)
                resolved = (destination / suffix).resolve()
                if resolved != destination and not resolved.is_relative_to(destination):
                    raise ValueError(f"historical path escapes relocation destination: {persisted_path}")
                return resolved
        return candidate


NO_HISTORICAL_PATH_RELOCATION = HistoricalPathResolver()
