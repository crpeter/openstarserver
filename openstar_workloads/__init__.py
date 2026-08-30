"""Trusted server-owned workload plugin infrastructure."""

from .contract import (
    DatasetReduction,
    ResultValidation,
    WorkloadDefinition,
    WorkloadPlugin,
)
from .discovery import discover_workloads
from .registry import WorkloadRegistry

__all__ = [
    "DatasetReduction",
    "ResultValidation",
    "WorkloadDefinition",
    "WorkloadPlugin",
    "WorkloadRegistry",
    "discover_workloads",
]
