"""Generic distributed candidate generation for exhausted transit residuals.

All astronomy-specific preparation and interpretation remains on the server.
Workers receive two ordinary ``openstar.lomb-scargle.v1`` datasets and return
generic numerical period candidates only.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .tess_blind_transit_search import (
    _frequency_bounds,
    _load,
    prepare_exhausted_residual_sectors,
)


WORKLOAD_ID = "openstar.lomb-scargle.v1"
PREPARATION_VERSION = (
    "openstar.tess-exhausted-residual-candidate-generation-preparation.v1"
)
INTERPRETATION_VERSION = (
    "openstar.tess-exhausted-residual-candidate-generation.v1"
)
TOTAL_FREQUENCIES = 262_144
FREQUENCIES_PER_WORK_UNIT = 4_096
RESIDUAL_METHODS = (
    "CUMULATIVE_TRANSIT_WINDOW_MASKING",
    "BOX_MODEL_SUBTRACTION",
)


def _safe(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return normalized or "investigation"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def distributed_candidate_generation_warranted(
    blind_transit_result: dict[str, Any],
) -> bool:
    availability = blind_transit_result.get(
        "independentSectorAvailability"
    ) or {}
    iterative = blind_transit_result.get("iterativeSearch") or {}
    census = blind_transit_result.get(
        "exhaustedSectorResidualFamilyCensus"
    ) or {}
    methods = {
        item.get("residualSearchMethod")
        for item in census.get("methods") or []
    }
    return bool(
        blind_transit_result.get("classification")
        == "REPLICATED_BLIND_TRANSIT_LIKE_CANDIDATE"
        and blind_transit_result.get("recommendedNextTest")
        == "GENERIC_DISTRIBUTED_RESIDUAL_TRANSIT_CANDIDATE_GENERATION"
        and availability.get("allCandidateSectorsPrepared") is True
        and iterative.get("terminationReason")
        == "NEXT_RESIDUAL_SIGNAL_UNRESOLVED"
        and set(RESIDUAL_METHODS).issubset(methods)
        and 0 < len(blind_transit_result.get("candidateSignals") or []) < 4
    )


def _pooled_residual_dataset(
    *, dataset_id: str, target_name: str,
    sectors: list[dict[str, Any]], residual_method: str,
    minimum_frequency: float, maximum_frequency: float,
) -> dict[str, Any]:
    import numpy as np

    if not sectors:
        raise ValueError("distributed residual dataset has no sectors")
    absolute_times = np.concatenate([item["times"] for item in sectors])
    values = np.concatenate([
        item["residual"] / item["sigma"] for item in sectors
    ])
    finite = np.isfinite(absolute_times) & np.isfinite(values)
    absolute_times = absolute_times[finite]
    values = values[finite]
    if absolute_times.size < 200:
        raise ValueError("distributed residual dataset has too few samples")
    order = np.argsort(absolute_times)
    absolute_times = absolute_times[order]
    values = values[order]
    origin = float(absolute_times[0])
    relative = (absolute_times - origin).astype(np.float32)
    standardized = (values - np.median(values)).astype(np.float32)
    frequency_step = (
        maximum_frequency - minimum_frequency
    ) / (TOTAL_FREQUENCIES - 1)
    return {
        "id": dataset_id,
        "targetName": target_name,
        "mission": "generic-distributed-time-series",
        "times": relative.astype(float).tolist(),
        "flux": standardized.astype(float).tolist(),
        "frequencySearch": {
            "minimumFrequency": minimum_frequency,
            "maximumFrequency": maximum_frequency,
            "frequencyStep": frequency_step,
            "totalFrequencies": TOTAL_FREQUENCIES,
            "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
        },
        "reference": {},
        "source": {
            "timeOriginDays": origin,
            "samplePrecision": "Float32",
            "residualSearchMethod": residual_method,
            "sectorCount": len(sectors),
        },
        "science": {
            "role": "generic-candidate-generation",
            "selectionAuthority": "SERVER_SIDE_UNCHANGED_TRANSIT_GATES_ONLY",
            "catalogAnswerKeyUsed": False,
        },
    }


def build_exhausted_residual_candidate_project(
    *, source_project_path: str | Path,
    primary_dataset_path: str | Path,
    independent_spec: dict[str, Any],
    blind_transit_result: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    if not distributed_candidate_generation_warranted(blind_transit_result):
        raise ValueError(
            "generic distributed residual candidate generation is not warranted"
        )
    source_project_path = Path(source_project_path).expanduser().resolve()
    primary_dataset_path = Path(primary_dataset_path).expanduser().resolve()
    with source_project_path.open(encoding="utf-8") as handle:
        source_project = json.load(handle)
    primary_dataset = _load(primary_dataset_path)
    minimum, maximum = _frequency_bounds(primary_dataset)
    accepted_signals = list(blind_transit_result["candidateSignals"])
    artifact_dir = (
        Path(output_dir).expanduser().resolve()
        / "exhausted-residual-candidate-generation"
    )

    prepared = []
    entries = []
    for method in RESIDUAL_METHODS:
        method_slug = method.lower().replace("_", "-")
        dataset_id = (
            f"{_safe(investigation_id)}-exhausted-residual-{method_slug}-v1"
        )
        sectors = prepare_exhausted_residual_sectors(
            primary_dataset_path=primary_dataset_path,
            independent_spec=independent_spec,
            accepted_signals=accepted_signals,
            residual_search_method=method,
        )
        dataset = _pooled_residual_dataset(
            dataset_id=dataset_id,
            target_name=f"Exhausted residual candidate generation — {method}",
            sectors=sectors,
            residual_method=method,
            minimum_frequency=minimum,
            maximum_frequency=maximum,
        )
        dataset_path = artifact_dir / f"{_safe(dataset_id)}.json"
        _write_json(dataset_path, dataset)
        entry = {
            "id": dataset_id,
            "path": str(dataset_path.resolve()),
            "targetName": dataset["targetName"],
            "role": "generic-candidate-generation",
            "residualSearchMethod": method,
        }
        entries.append(entry)
        prepared.append({
            "datasetID": dataset_id,
            "datasetPath": str(dataset_path.resolve()),
            "residualSearchMethod": method,
            "sampleCount": len(dataset["times"]),
            "timeOriginDays": dataset["source"]["timeOriginDays"],
            "frequencySearch": dataset["frequencySearch"],
        })

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "exhausted-residual-candidates-v1"
    )
    manifest = {
        "id": project_id,
        "name": "Generic exhausted residual candidate generation",
        "workloadID": WORKLOAD_ID,
        "datasets": entries,
        "investigation": {
            "purpose": "generic-exhausted-residual-candidate-generation",
            "sourceProjectID": source_project["id"],
            "specializedWorkerLogic": False,
        },
    }
    project_path = artifact_dir / "project.json"
    _write_json(project_path, manifest)
    return {
        "version": PREPARATION_VERSION,
        "executable": True,
        "projectID": project_id,
        "projectPath": str(project_path.resolve()),
        "workloadID": WORKLOAD_ID,
        "workerSemantics": "GENERIC_LOMB_SCARGLE",
        "specializedTessWorkerLogic": False,
        "normalTopTwelveSelectionPathChanged": False,
        "scienceThresholdsChanged": False,
        "primaryDatasetPath": str(primary_dataset_path),
        "independentSpec": independent_spec,
        "preparedDatasets": prepared,
        "totalWorkUnits": len(prepared) * math.ceil(
            TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT
        ),
        "catalogAnswerKeyUsed": False,
    }


def interpret_exhausted_residual_candidate_project(
    *, preparation: dict[str, Any], project_status: dict[str, Any],
) -> dict[str, Any]:
    if preparation.get("version") != PREPARATION_VERSION:
        raise ValueError("unexpected distributed residual preparation version")
    if preparation.get("workloadID") != WORKLOAD_ID:
        raise ValueError("distributed residual preparation is not generic Lomb-Scargle")
    if str(project_status.get("projectID")) != str(preparation.get("projectID")):
        raise ValueError("distributed residual project identity changed")
    if project_status.get("workloadID") != WORKLOAD_ID:
        raise ValueError("distributed residual run used an unexpected workload")
    statuses = {
        str(item.get("id")): item
        for item in project_status.get("datasets") or []
    }
    expected = {
        str(item["datasetID"]): item
        for item in preparation.get("preparedDatasets") or []
    }
    if set(statuses) != set(expected):
        raise ValueError("distributed residual run dataset set changed")

    methods = []
    candidate_map = {}
    for dataset_id, prepared in expected.items():
        status = statuses[dataset_id]
        if (
            status.get("coverageComplete") is not True
            or int(status.get("failedWorkUnits") or 0) != 0
        ):
            raise RuntimeError(
                f"distributed residual dataset did not complete: {dataset_id}"
            )
        candidates = []
        for item in status.get("independentCandidates") or []:
            try:
                frequency = float(item["frequency"])
                power = float(item["power"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                math.isfinite(frequency)
                and frequency > 0.0
                and math.isfinite(power)
            ):
                candidates.append({
                    "frequency": frequency,
                    "periodDays": 1.0 / frequency,
                    "power": power,
                })
        method = prepared["residualSearchMethod"]
        candidate_map[method] = candidates
        methods.append({
            "residualSearchMethod": method,
            "datasetID": dataset_id,
            "periodStatus": status.get("periodStatus"),
            "coverageComplete": True,
            "candidateCount": len(candidates),
            "candidates": candidates,
        })
    return {
        "version": INTERPRETATION_VERSION,
        "classification": "GENERIC_DISTRIBUTED_RESIDUAL_CANDIDATES_COMPLETE",
        "workloadID": WORKLOAD_ID,
        "workerSemantics": "GENERIC_LOMB_SCARGLE",
        "specializedTessWorkerLogic": False,
        "methods": methods,
        "candidateMap": candidate_map,
        "candidateSelectionPerformedByWorkers": False,
        "claimDecisionPerformedByWorkers": False,
        "catalogAnswerKeyUsed": False,
    }
