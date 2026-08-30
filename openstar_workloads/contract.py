"""Contract between coordinator core and trusted workload packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class WorkloadDefinition:
    workload_id: str
    dataset_schema_id: str
    payload_schema_id: str
    result_schema_id: str
    allows_legacy_schemaless_workers: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.workload_id,
            self.dataset_schema_id,
            self.payload_schema_id,
            self.result_schema_id,
        )
        if any(not isinstance(value, str) or not value or value.strip() != value
               for value in identities):
            raise ValueError("Workload and schema identities must be canonical strings")


@dataclass(frozen=True, slots=True)
class ResultValidation:
    accepted: bool
    message: str
    details: Mapping[str, Any]


class WorkloadPlugin(Protocol):
    definition: WorkloadDefinition
    uses_legacy_coordinator_diagnostics: bool

    def validate_dataset(self, dataset: Mapping[str, Any]) -> None: ...
    def build_work_payloads(self, dataset: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...
    def canonicalize_result(self, work_unit: Mapping[str, Any], result: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def validate_result(self, work_unit: Mapping[str, Any], result: Mapping[str, Any]) -> ResultValidation: ...
    def reduce_dataset(self, dataset: Mapping[str, Any], work_units: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]], *, terminal: bool) -> Mapping[str, Any]: ...
    def contribution_metrics(self, work_unit: Mapping[str, Any], dataset: Mapping[str, Any]) -> Mapping[str, Any]: ...
