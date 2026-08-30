"""Compatibility identity for the existing box-period search lane."""

from openstar_workloads.contract import WorkloadDefinition
from .period_search import PeriodSearchCompatibilityPlugin


class BoxPeriodCompatibilityPlugin(PeriodSearchCompatibilityPlugin):
    definition = WorkloadDefinition(
        workload_id="openstar.tess-period-search.v1",
        dataset_schema_id="openstar.dataset.period-search.v1",
        payload_schema_id="openstar.payload.frequency-shard.v1",
        result_schema_id="openstar.result.period-search-shard.v1",
        allows_legacy_schemaless_workers=True,
    )


PLUGIN = BoxPeriodCompatibilityPlugin()
