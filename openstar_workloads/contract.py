"""Stable contracts implemented by trusted server-owned workload packages."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable


def _immutable_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class WorkloadDefinition:
    """Immutable wire identities and compatibility mode for one workload."""

    workload_id: str
    dataset_schema_id: str
    payload_schema_id: str
    result_schema_id: str
    allows_legacy_schemaless_workers: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "workload_id",
            "dataset_schema_id",
            "payload_schema_id",
            "result_schema_id",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a non-empty canonical string"
                )
        if type(self.allows_legacy_schemaless_workers) is not bool:
            raise TypeError(
                "allows_legacy_schemaless_workers must be a bool"
            )


@dataclass(frozen=True, slots=True)
class ResultValidation:
    """The trusted validation decision for one completed worker result."""

    accepted: bool
    message: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be a bool")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")
        object.__setattr__(
            self,
            "details",
            _immutable_mapping(self.details, "details"),
        )


@dataclass(frozen=True, slots=True)
class DatasetReduction:
    """Workload-owned reduced payload and fields merged into dataset status."""

    payload: Mapping[str, Any]
    status_fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload",
            _immutable_mapping(self.payload, "payload"),
        )
        object.__setattr__(
            self,
            "status_fields",
            _immutable_mapping(self.status_fields, "status_fields"),
        )


@runtime_checkable
class WorkloadPlugin(Protocol):
    """Complete workload boundary consumed by coordinator core.

    Implementations must be pure, stateless, reentrant, deterministic, and
    free of I/O. Accounting values must come only from server-owned work units
    and datasets, never from a worker result.
    """

    definition: WorkloadDefinition
    uses_legacy_coordinator_diagnostics: bool
    uses_legacy_science_metadata_validation: bool

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None: ...

    def build_work_payloads(
        self,
        dataset: Mapping[str, Any],
    ) -> Iterable[Mapping[str, Any]]: ...

    def legacy_work_unit_fields(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def canonicalize_result(
        self,
        work_unit: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def validate_result(
        self,
        work_unit: Mapping[str, Any],
        result: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> ResultValidation: ...

    def reduce_dataset(
        self,
        dataset: Mapping[str, Any],
        work_units: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any] | None],
        terminal: bool,
    ) -> DatasetReduction: ...

    def contribution_metrics(
        self,
        work_unit: Mapping[str, Any],
        dataset: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...
