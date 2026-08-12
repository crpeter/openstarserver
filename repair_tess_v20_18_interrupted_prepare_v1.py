from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_STAGE_ID = "072-prepare-official-spoc-prf-forward-modeling"
EXPECTED_HANDLER_ID = "openstar.tess.official-spoc-prf-forward-modeling.prepare"
EXPECTED_PREVIOUS_HANDLER_ID = "openstar.tess.finalize"
RESTORED_INVESTIGATION_STATUS = "COMPLETE"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Investigation record is not a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def existing_key(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in mapping:
            return key
    return None


def stage_id(stage: dict[str, Any]) -> str:
    value = first_present(stage, "id", "stageID", "stageId")
    return "" if value is None else str(value)


def stage_handler_id(stage: dict[str, Any]) -> str:
    value = first_present(stage, "handler_id", "handlerID", "handlerId")
    return "" if value is None else str(value)


def stage_status(stage: dict[str, Any]) -> str:
    value = first_present(stage, "status")
    return "" if value is None else str(value).upper()


def stage_result(stage: dict[str, Any]) -> Any:
    return first_present(stage, "result")


def stage_provenance(stage: dict[str, Any]) -> Any:
    return first_present(stage, "provenance")


def investigation_status(record: dict[str, Any]) -> str:
    value = first_present(record, "status")
    return "" if value is None else str(value).upper()


def investigation_id(record: dict[str, Any]) -> str:
    value = first_present(record, "id", "investigationID", "investigationId")
    return "" if value is None else str(value)


def candidate_record_paths(root: Path, investigation: str) -> list[Path]:
    return [
        root / investigation / "investigation.json",
        root / f"{investigation}.json",
    ]


def locate_record(root: Path, investigation: str) -> Path:
    candidates = candidate_record_paths(root, investigation)
    existing = [path for path in candidates if path.is_file()]

    if len(existing) == 1:
        return existing[0]

    if len(existing) > 1:
        joined = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            "Multiple investigation records matched; refusing to guess:\n"
            f"{joined}"
        )

    joined = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Investigation record not found. Checked:\n"
        f"{joined}"
    )


def validate_repair_target(
    record: dict[str, Any],
    expected_investigation_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    actual_id = investigation_id(record)
    if actual_id and actual_id != expected_investigation_id:
        raise RuntimeError(
            "Investigation ID mismatch: "
            f"record={actual_id!r}, requested={expected_investigation_id!r}."
        )

    status = investigation_status(record)
    if status != "RUNNING":
        raise RuntimeError(
            "Repair is only valid for the interrupted RUNNING v20.18 state; "
            f"current investigation status is {status or '[missing]'}."
        )

    stages_raw = record.get("stages")
    if not isinstance(stages_raw, list) or len(stages_raw) < 2:
        raise RuntimeError("Investigation does not contain enough stages to repair safely.")

    if not all(isinstance(item, dict) for item in stages_raw):
        raise RuntimeError("Investigation stages are not all JSON objects.")

    stages: list[dict[str, Any]] = stages_raw
    interrupted = stages[-1]
    previous = stages[-2]

    if stage_id(interrupted) != EXPECTED_STAGE_ID:
        raise RuntimeError(
            "Last stage is not the expected interrupted v20.18 prepare stage: "
            f"{stage_id(interrupted) or '[missing]'}."
        )

    if stage_handler_id(interrupted) != EXPECTED_HANDLER_ID:
        raise RuntimeError(
            "Last stage handler is not the expected v20.18 prepare handler: "
            f"{stage_handler_id(interrupted) or '[missing]'}."
        )

    if stage_status(interrupted) not in {"RUNNING", "PENDING"}:
        raise RuntimeError(
            "Expected the v20.18 prepare stage to be RUNNING/PENDING, got "
            f"{stage_status(interrupted) or '[missing]'}."
        )

    if stage_result(interrupted) is not None:
        raise RuntimeError(
            "Interrupted v20.18 prepare stage already has a result; refusing to remove it."
        )

    if stage_provenance(interrupted) is not None:
        raise RuntimeError(
            "Interrupted v20.18 prepare stage already has provenance; refusing to remove it."
        )

    if stage_status(previous) != "COMPLETE":
        raise RuntimeError(
            "Stage before interrupted v20.18 prepare is not COMPLETE; refusing repair."
        )

    if stage_handler_id(previous) != EXPECTED_PREVIOUS_HANDLER_ID:
        raise RuntimeError(
            "Stage before interrupted v20.18 prepare is not the v20.17 finalize stage; "
            f"handler={stage_handler_id(previous) or '[missing]'}."
        )

    later_official_stages = [
        item
        for item in stages[:-1]
        if stage_handler_id(item).startswith(
            "openstar.tess.official-spoc-prf-forward-modeling."
        )
    ]
    if later_official_stages:
        raise RuntimeError(
            "Official SPOC PRF stages already exist before the interrupted prepare stage; "
            "refusing to rewrite history."
        )

    return stages, interrupted, previous


def repair_record(path: Path, expected_investigation_id: str) -> Path:
    record = load_json(path)
    stages, interrupted, previous = validate_repair_target(
        record,
        expected_investigation_id,
    )

    backup_path = path.with_name(
        f"{path.name}.before-v20-18-repair-{timestamp_for_filename()}.bak"
    )
    shutil.copy2(path, backup_path)

    repaired = dict(record)
    repaired["stages"] = stages[:-1]
    repaired["status"] = RESTORED_INVESTIGATION_STATUS

    updated_key = existing_key(repaired, "updated_at", "updatedAt")
    if updated_key is not None:
        repaired[updated_key] = utc_now_iso()

    write_json_atomic(path, repaired)

    verify = load_json(path)
    verify_stages = verify.get("stages")
    if not isinstance(verify_stages, list) or len(verify_stages) != len(stages) - 1:
        raise RuntimeError(
            "Post-write verification failed: unexpected stage count. "
            f"Original backup remains at {backup_path}."
        )

    if investigation_status(verify) != RESTORED_INVESTIGATION_STATUS:
        raise RuntimeError(
            "Post-write verification failed: investigation status was not restored. "
            f"Original backup remains at {backup_path}."
        )

    if not verify_stages or stage_handler_id(verify_stages[-1]) != EXPECTED_PREVIOUS_HANDLER_ID:
        raise RuntimeError(
            "Post-write verification failed: v20.17 finalize is not the final stage. "
            f"Original backup remains at {backup_path}."
        )

    print("⭐ OpenStar interrupted-stage repair")
    print(f"   investigation: {expected_investigation_id}")
    print(f"   record: {path}")
    print(f"   removed stage: {stage_id(interrupted)}")
    print(f"   removed handler: {stage_handler_id(interrupted)}")
    print(f"   restored terminal stage: {stage_id(previous)}")
    print(f"   restored status: {RESTORED_INVESTIGATION_STATUS}")
    print(f"   backup: {backup_path}")
    print("✅ Ready to rerun --continue-official-spoc-prf-forward-modeling")

    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair the specific OpenStar v20.18 state left behind when "
            "072-prepare-official-spoc-prf-forward-modeling crashed before "
            "producing a result."
        )
    )
    parser.add_argument(
        "--investigation-id",
        required=True,
        help="Investigation ID, for example tess-v20-2-blind-c.",
    )
    parser.add_argument(
        "--root",
        default="data/investigations",
        help="Investigation storage root (default: data/investigations).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    record_path = locate_record(root, args.investigation_id)
    repair_record(record_path, args.investigation_id)


if __name__ == "__main__":
    main()
