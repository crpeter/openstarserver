from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _safe(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value or "investigation"


def resolve_dataset_path(project_path: str | Path, entry: dict[str, Any]) -> Path:
    project_path = Path(project_path).expanduser().resolve()
    raw = Path(str(entry["path"]))
    if raw.is_absolute():
        return raw
    candidates = [
        Path.cwd() / raw,
        project_path.parent / raw,
        project_path.parent.parent / raw,
        project_path.parent.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def select_dataset_entry(
    project: dict[str, Any],
    *,
    dataset_id: str | None = None,
    tic_id: int | None = None,
) -> dict[str, Any]:
    matches = []
    for entry in project.get("datasets") or []:
        if dataset_id is not None and str(entry.get("id")) == str(dataset_id):
            matches.append(entry)
            continue
        if tic_id is not None and str(entry.get("ticID")) == str(tic_id):
            matches.append(entry)

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one dataset match; "
            f"found {len(matches)} for dataset_id={dataset_id!r}, tic_id={tic_id!r}."
        )
    return copy.deepcopy(matches[0])


def build_single_target_primary(
    *,
    source_project_path: str | Path,
    output_dir: str | Path,
    investigation_id: str,
    dataset_id: str | None = None,
    tic_id: int | None = None,
) -> dict[str, Any]:
    source_project_path = Path(source_project_path).expanduser().resolve()
    project = _load_json(source_project_path)
    entry = select_dataset_entry(project, dataset_id=dataset_id, tic_id=tic_id)
    dataset_path = resolve_dataset_path(source_project_path, entry)
    dataset = _load_json(dataset_path)

    selected_tic = entry.get("ticID")
    if selected_tic is None:
        selected_tic = (dataset.get("source") or {}).get("ticID")
    if selected_tic is None:
        raise ValueError("Selected TESS dataset has no TIC ID.")

    project_id = f"{project['id']}.investigation.{_safe(investigation_id)}.primary"
    manifest = {
        "id": project_id,
        "name": f"{project.get('name', project['id'])} — investigation target",
        "workloadID": project["workloadID"],
        "datasets": [entry],
        "investigation": {
            "sourceProjectID": project["id"],
            "sourceProjectPath": str(source_project_path),
            "sourceDatasetID": entry["id"],
            "purpose": "single-target-primary",
        },
    }

    output_dir = Path(output_dir)
    manifest_path = output_dir / "primary" / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    return {
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "sourceProjectID": project["id"],
        "sourceProjectPath": str(source_project_path),
        "datasetID": str(entry["id"]),
        "datasetPath": str(dataset_path),
        "ticID": int(selected_tic),
        "targetName": entry.get("targetName") or dataset.get("targetName"),
        "sector": entry.get("sector") or (dataset.get("source") or {}).get("sector"),
        "sourceDatasetEntry": entry,
    }


def build_low_frequency_followup(
    *,
    source_project_path: str | Path,
    source_dataset_path: str | Path,
    source_dataset_entry: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
) -> dict[str, Any]:
    source_project_path = Path(source_project_path).expanduser().resolve()
    source_dataset_path = Path(source_dataset_path).expanduser().resolve()
    project = _load_json(source_project_path)
    dataset = _load_json(source_dataset_path)

    original_search = dataset.get("frequencySearch") or {}
    original_min = float(original_search.get("minimumFrequency") or 0.1)
    maximum_frequency = max(0.03, original_min * 1.25)
    minimum_frequency = max(0.01, maximum_frequency / 10.0)
    total_frequencies = 262144
    per_work_unit = int(original_search.get("frequenciesPerWorkUnit") or 4096)
    per_work_unit = max(4096, min(per_work_unit, total_frequencies))
    frequency_step = (maximum_frequency - minimum_frequency) / (total_frequencies - 1)

    follow_dataset = copy.deepcopy(dataset)
    follow_dataset_id = f"{dataset['id']}-low-frequency-v1"
    follow_dataset["id"] = follow_dataset_id
    follow_dataset["targetName"] = f"{dataset.get('targetName', dataset['id'])} low-frequency follow-up"
    follow_dataset["frequencySearch"] = {
        "minimumFrequency": minimum_frequency,
        "maximumFrequency": maximum_frequency,
        "frequencyStep": frequency_step,
        "totalFrequencies": total_frequencies,
        "frequenciesPerWorkUnit": per_work_unit,
    }
    # Reference values were calculated for the old grid. Reference data is
    # optional in production and must never be carried onto a different grid.
    follow_dataset["reference"] = {}
    science = dict(follow_dataset.get("science") or {})
    science["role"] = "follow-up"
    science["followupReason"] = "lower-frequency-extension"
    follow_dataset["science"] = science

    output_dir = Path(output_dir)
    dataset_path = output_dir / "followup" / f"{_safe(follow_dataset_id)}.json"
    _write_json(dataset_path, follow_dataset)

    project_id = f"{project['id']}.investigation.{_safe(investigation_id)}.low-frequency-v1"
    entry = copy.deepcopy(source_dataset_entry)
    entry["id"] = follow_dataset_id
    entry["path"] = str(dataset_path.resolve())
    entry["targetName"] = follow_dataset["targetName"]
    entry["role"] = "follow-up"

    manifest = {
        "id": project_id,
        "name": f"{project.get('name', project['id'])} — lower-frequency follow-up",
        "workloadID": project["workloadID"],
        "datasets": [entry],
        "investigation": {
            "sourceProjectID": project["id"],
            "sourceDatasetID": source_dataset_entry["id"],
            "purpose": "lower-frequency-extension",
        },
    }
    manifest_path = output_dir / "followup" / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)

    return {
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "datasetID": follow_dataset_id,
        "datasetPath": str(dataset_path.resolve()),
        "frequencySearch": follow_dataset["frequencySearch"],
    }
