"""Structural validation shared by discovery and package tests."""

from __future__ import annotations

from .contract import WorkloadDefinition

_METHODS = (
    "validate_dataset", "build_work_payloads", "canonicalize_result",
    "validate_result", "reduce_dataset", "contribution_metrics",
)


def validate_plugin(plugin: object) -> None:
    definition = getattr(plugin, "definition", None)
    if not isinstance(definition, WorkloadDefinition):
        raise RuntimeError("Workload plugin has no valid WorkloadDefinition")
    for method in _METHODS:
        if not callable(getattr(plugin, method, None)):
            raise RuntimeError(
                f"Malformed workload plugin {definition.workload_id}: missing {method}"
            )
