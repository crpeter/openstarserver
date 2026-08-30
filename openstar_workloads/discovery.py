"""Deterministic discovery limited to the in-repository plugin package."""

from __future__ import annotations

import importlib
import pkgutil

from .registry import WorkloadRegistry

TRUSTED_PACKAGE = "openstar_workloads.plugins"


def discover_workloads() -> WorkloadRegistry:
    package = importlib.import_module(TRUSTED_PACKAGE)
    names = sorted(
        item.name for item in pkgutil.iter_modules(package.__path__)
        if not item.name.startswith("_")
    )
    plugins = []
    for name in names:
        module = importlib.import_module(f"{TRUSTED_PACKAGE}.{name}")
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            raise RuntimeError(f"Malformed workload module {module.__name__}: missing PLUGIN")
        plugins.append(plugin)
    return WorkloadRegistry(plugins)
