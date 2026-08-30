"""Fail-fast structural checks for workload definitions."""

from .contract import WorkloadDefinition

REQUIRED_METHODS = (
    "validate_dataset", "build_work_payloads", "canonicalize_result",
    "validate_result", "reduce_dataset", "contribution_metrics",
)


def validate_plugin(plugin) -> None:
    definition = getattr(plugin, "definition", None)
    if not isinstance(definition, WorkloadDefinition):
        raise RuntimeError("Invalid workload plugin definition")
    if not isinstance(getattr(plugin, "uses_legacy_coordinator_diagnostics", None), bool):
        raise RuntimeError(f"Invalid diagnostics mode: {definition.workload_id}")
    for method in REQUIRED_METHODS:
        if not callable(getattr(plugin, method, None)):
            raise RuntimeError(f"Invalid workload plugin {definition.workload_id}: missing {method}")
