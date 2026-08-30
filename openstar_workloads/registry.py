"""Immutable, deterministic, fail-closed workload registry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

from .conformance import validate_plugin
from .contract import WorkloadPlugin


class WorkloadRegistry:
    """Validated lookup table for trusted workload implementations."""

    def __init__(self, plugins: Iterable[WorkloadPlugin] = ()) -> None:
        validated = []
        for plugin in plugins:
            validate_plugin(plugin)
            validated.append(plugin)

        entries = {}
        for plugin in sorted(
            validated,
            key=lambda item: item.definition.workload_id,
        ):
            workload_id = plugin.definition.workload_id
            if workload_id in entries:
                raise RuntimeError(
                    f"Duplicate workload ID: {workload_id}"
                )
            entries[workload_id] = plugin

        self._entries: Mapping[str, WorkloadPlugin] = MappingProxyType(
            entries
        )

    @property
    def workload_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def require(self, workload_id: str) -> WorkloadPlugin:
        try:
            return self._entries[str(workload_id)]
        except KeyError as error:
            raise RuntimeError(
                f"Unknown workload ID: {workload_id}"
            ) from error

    def __iter__(self) -> Iterator[WorkloadPlugin]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)
