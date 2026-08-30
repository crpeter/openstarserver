"""Deterministic discovery from one trusted in-repository namespace."""

import importlib
import pkgutil

from .registry import WorkloadRegistry

TRUSTED_PLUGIN_PACKAGE = "openstar_workloads.plugins"


def discover_workloads():
    package = importlib.import_module(TRUSTED_PLUGIN_PACKAGE)
    module_names = sorted(
        module.name for module in pkgutil.iter_modules(package.__path__)
        if not module.name.startswith("_")
    )
    plugins = []
    for module_name in module_names:
        module = importlib.import_module(f"{TRUSTED_PLUGIN_PACKAGE}.{module_name}")
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None:
            raise RuntimeError(f"Trusted workload module lacks PLUGIN: {module.__name__}")
        if isinstance(plugin, (tuple, list)):
            plugins.extend(plugin)
        else:
            plugins.append(plugin)
    return WorkloadRegistry(plugins)
