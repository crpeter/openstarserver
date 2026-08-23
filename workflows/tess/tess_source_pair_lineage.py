"""Narrow validation for the persisted current-source-pair archive lineage."""

from __future__ import annotations

import math


INTERPRETATION_SOURCE_PAIR_HANDLERS = (
    "openstar.tess.noirlab-image-forced-photometry.interpret",
    "openstar.tess.des-dr2-se-local-forced-photometry.interpret",
)
TRANSPORT_SOURCE_PAIR_HANDLERS = (
    "openstar.tess.atlas-forced-photometry.prepare",
    "openstar.tess.atlas-forced-photometry.collect",
)


def valid_current_source_pair(pair: object) -> bool:
    if not isinstance(pair, dict) or pair.get("version") != "openstar.current-source-pair.v1":
        return False
    target = pair.get("target")
    counterpart = pair.get("counterpart")
    if not isinstance(target, dict) or not isinstance(counterpart, dict):
        return False
    identities = (target.get("gaiaDR3SourceID"), counterpart.get("gaiaDR3SourceID"))
    if identities[0] is None or identities[1] is None or identities[0] == identities[1]:
        return False
    try:
        return all(math.isfinite(float(source[key])) for source in (target, counterpart)
                   for key in ("raDeg", "decDeg"))
    except (KeyError, TypeError, ValueError):
        return False


def frozen_source_pair_evidence(investigation, atlas_interpretation_stage):
    """Return one allowlisted, persisted summary carrying the exact ATLAS pair.

    Interpretation summaries are preferred as scientific provenance.  ATLAS
    preparation/collection summaries are accepted only when no matching current
    archive interpretation is present.
    """
    atlas_pair = (atlas_interpretation_stage.result or {}).get("sourcePair")
    if not valid_current_source_pair(atlas_pair):
        return None
    position = investigation.stages.index(atlas_interpretation_stage)
    preceding = investigation.stages[:position]
    for handlers in (INTERPRETATION_SOURCE_PAIR_HANDLERS,
                     TRANSPORT_SOURCE_PAIR_HANDLERS):
        for stage in reversed(preceding):
            result = stage.result if isinstance(stage.result, dict) else None
            if (stage.status == "COMPLETE" and stage.handler_id in handlers
                    and result is not None and result.get("sourcePair") == atlas_pair
                    and valid_current_source_pair(result.get("sourcePair"))):
                return result
    return None
