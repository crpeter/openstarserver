"""Stable contracts implemented by server-owned workload packages."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class WorkloadDefinition:
    """Immutable wire identities for one version of a workload."""

    workload_id: str
    dataset_schema_id: str
    payload_schema_id: str
    result_schema_id: str
    allows_legacy_schemaless_workers: bool = False

    def __post_init__(self) -> None:
        for name in (
            "workload_id", "dataset_schema_id", "payload_schema_id", "result_schema_id"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty canonical string")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    message: str
    details: Mapping[str, Any]

    @classmethod
    def accept(cls, message: str = "Result is valid", **details: Any):
        return cls(True, message, MappingProxyType(dict(details)))

    @classmethod
    def reject(cls, message: str, **details: Any):
        return cls(False, message, MappingProxyType(dict(details)))


@runtime_checkable
class WorkloadPlugin(Protocol):
    """The complete scientific-computation boundary owned by a workload lane.

    Implementations must be deterministic and must not trust result-provided
    accounting dimensions. Methods receive ordinary mappings so packages stay
    independent of the coordinator implementation.
    """

    definition: WorkloadDefinition

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None: ...

    def build_work_payloads(
        self, dataset: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]: ...

    def canonicalize_result(
        self, work_unit: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def validate_result(
        self, work_unit: Mapping[str, Any], result: Mapping[str, Any]
    ) -> ValidationResult: ...

    def reduce_dataset(
        self,
        dataset: Mapping[str, Any],
        work_units: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        *,
        terminal: bool,
    ) -> Mapping[str, Any]: ...

    def contribution_metrics(
        self, work_unit: Mapping[str, Any], dataset: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
