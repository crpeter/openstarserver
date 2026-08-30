"""Fail-closed registry for trusted workload plugins."""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

from .contract import WorkloadPlugin
from .conformance import validate_plugin


class WorkloadRegistry:
    def __init__(self, plugins: Iterable[WorkloadPlugin] = ()) -> None:
        registered = {}
        for plugin in plugins:
            validate_plugin(plugin)
            workload_id = plugin.definition.workload_id
            if workload_id in registered:
                raise RuntimeError(f"Duplicate workload registration: {workload_id}")
            registered[workload_id] = plugin
        self._plugins: Mapping[str, WorkloadPlugin] = MappingProxyType(registered)

    def require(self, workload_id: str) -> WorkloadPlugin:
        try:
            return self._plugins[str(workload_id)]
        except KeyError as error:
            raise RuntimeError(f"Unknown workload ID: {workload_id}") from error

    def __iter__(self) -> Iterator[WorkloadPlugin]:
        return iter(self._plugins.values())

    def __len__(self) -> int:
        return len(self._plugins)
