"""Immutable fail-closed workload registry."""

from types import MappingProxyType

from .conformance import validate_plugin


class WorkloadRegistry:
    def __init__(self, plugins=()):
        entries = {}
        for plugin in plugins:
            validate_plugin(plugin)
            workload_id = plugin.definition.workload_id
            if workload_id in entries:
                raise RuntimeError(f"Duplicate workload ID: {workload_id}")
            entries[workload_id] = plugin
        self._entries = MappingProxyType(entries)

    def require(self, workload_id):
        try:
            return self._entries[str(workload_id)]
        except KeyError as error:
            raise RuntimeError(f"Unknown workload ID: {workload_id}") from error

    def __iter__(self):
        return iter(self._entries.values())
