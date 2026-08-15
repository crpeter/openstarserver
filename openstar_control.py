from __future__ import annotations

from typing import Any

from openstar_investigation import Investigation, InvestigationStore, utc_now_iso


def update_investigation_metadata(
    store: InvestigationStore,
    investigation: Investigation,
    values: dict[str, Any],
    *,
    status: str | None = None,
) -> Investigation:
    """Persist scheduler metadata without extending the core store API."""
    metadata = dict(investigation.metadata)
    metadata.update(values)
    updated = Investigation(
        id=investigation.id,
        workflow_id=investigation.workflow_id,
        workflow_version=investigation.workflow_version,
        status=status if status is not None else investigation.status,
        created_at=investigation.created_at,
        updated_at=utc_now_iso(),
        metadata=metadata,
        stages=investigation.stages,
    )
    store.save(updated)
    return updated
