"""Server-owned workload plugin infrastructure."""

from .discovery import discover_workloads
from .registry import WorkloadRegistry

__all__ = ["WorkloadRegistry", "discover_workloads"]
