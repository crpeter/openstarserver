"""Deterministic discovery from one trusted in-repository namespace."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Iterable

from .contract import WorkloadPlugin
from .registry import WorkloadRegistry


TRUSTED_PLUGIN_PACKAGE = "openstar_workloads.plugins"


def _module_plugins(module: ModuleType) -> Iterable[WorkloadPlugin]:
    try:
        exported = module.PLUGIN
    except AttributeError as error:
        raise RuntimeError(
            f"Trusted workload package lacks PLUGIN: {module.__name__}"
        ) from error

    if isinstance(exported, (tuple, list)):
        if not exported:
            raise RuntimeError(
                f"Trusted workload package has empty PLUGIN: "
                f"{module.__name__}"
            )
        yield from exported
        return

    yield exported


def discover_workloads() -> WorkloadRegistry:
    """Import only direct, sorted children of the trusted plugin package."""

    package = importlib.import_module(TRUSTED_PLUGIN_PACKAGE)
    child_names = sorted(
        item.name
        for item in pkgutil.iter_modules(package.__path__)
        if not item.name.startswith("_")
    )

    plugins = []
    for child_name in child_names:
        module = importlib.import_module(
            f"{TRUSTED_PLUGIN_PACKAGE}.{child_name}"
        )
        plugins.extend(_module_plugins(module))

    return WorkloadRegistry(plugins)
