"""Append-only v20.15 reinterpretation of frozen v20.14 model evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from openstar_investigation import sha256_file, sha256_json
from .tess_target_residual_mechanism import (
    ADJUDICATION_VERSION,
    DECISIVE_DELTA_BIC,
    adjudicate_sector_model_evidence,
)


def adjudicate_frozen_target_residual_mechanism(*, v2014_result: dict[str, Any],
        authoritative_v2014_artifacts: Iterable[Any]) -> dict[str, Any]:
    """Verify and reinterpret v20.14; this function performs no fitting or I/O beyond its artifact."""
    verified_sha = None
    frozen = None
    for reference in authoritative_v2014_artifacts:
        path = Path(reference.path if hasattr(reference, "path") else reference.get("path", ""))
        expected = str(reference.sha256 if hasattr(reference, "sha256") else reference.get("sha256", ""))
        if path.name != "target-residual-mechanism-v20.14.json" or not path.is_file():
            continue
        if not expected or sha256_file(path) != expected:
            raise RuntimeError("v20.15 frozen v20.14 artifact SHA verification failed.")
        with path.open(encoding="utf-8") as handle:
            frozen = json.load(handle)
        verified_sha = expected
        break
    if frozen is None or verified_sha is None:
        raise RuntimeError("v20.15 requires the frozen v20.14 result artifact.")
    if frozen != v2014_result:
        raise RuntimeError("v20.15 frozen v20.14 artifact and persisted result differ.")

    adjudication = adjudicate_sector_model_evidence(
        frozen.get("sectorModelEvidence") or [],
        fail_closed_reasons=frozen.get("failClosedReasons") or [],
    )
    return {
        "classification": adjudication["classification"],
        "physicalMechanismResolved": False,
        "recommendedNextTest": adjudication["recommendedNextTest"],
        "sectorModelEvidence": adjudication["sectorModelEvidence"],
        "replicatedMechanisms": adjudication["replicatedMechanisms"],
        "replicatedMechanismSupportingSectorIDs":
            adjudication["replicatedMechanismSupportingSectorIDs"],
        "failClosedReasons": adjudication["failClosedReasons"],
        "crossSectorPhaseUsed": False,
        "correctionType": "CORRECTIVE_ROUTE_INDEPENDENT_ADJUDICATION_OF_FROZEN_V20.14_MODEL_EVIDENCE",
        "newModelFittingPerformed": False,
        "distributedWorkPerformed": False,
        "inputProvenance": {
            "frozenV20.14ResultHash": sha256_json(v2014_result),
            "frozenV20.14ArtifactSHA256": verified_sha,
        },
        "adjudicationRules": {
            "version": ADJUDICATION_VERSION,
            "decisiveDeltaBIC": DECISIVE_DELTA_BIC,
            "allEligibleModelsCompared": True,
            "episodicMorphologyGateRequired": True,
        },
    }
