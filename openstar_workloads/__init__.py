"""Trusted, server-side workload plugin contracts and discovery."""

from .discovery import discover_workloads
from .registry import WorkloadRegistry

__all__ = ["WorkloadRegistry", "discover_workloads"]
