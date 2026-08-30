"""Fail-fast structural validation for workload plugins."""

from __future__ import annotations

from .contract import WorkloadDefinition


REQUIRED_METHODS = (
    "validate_dataset",
    "build_work_payloads",
    "legacy_work_unit_fields",
    "canonicalize_result",
    "validate_result",
    "reduce_dataset",
    "contribution_metrics",
)

REQUIRED_BOOLEAN_FLAGS = (
    "uses_legacy_coordinator_diagnostics",
    "uses_legacy_science_metadata_validation",
)


def validate_plugin(plugin: object) -> None:
    """Reject an incomplete or ambiguous plugin before coordinator startup."""

    definition = getattr(plugin, "definition", None)
    if not isinstance(definition, WorkloadDefinition):
        raise RuntimeError("Workload plugin has no valid WorkloadDefinition")

    for flag_name in REQUIRED_BOOLEAN_FLAGS:
        if type(getattr(plugin, flag_name, None)) is not bool:
            raise RuntimeError(
                f"Invalid {flag_name} for workload "
                f"{definition.workload_id}"
            )

    for method_name in REQUIRED_METHODS:
        if not callable(getattr(plugin, method_name, None)):
            raise RuntimeError(
                f"Invalid workload plugin {definition.workload_id}: "
                f"missing {method_name}"
            )
