#!/usr/bin/env python3
"""Prepare one catalog-blind TIC from official TESS light-curve products."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import lightkurve as lk
import numpy as np

from prepare_tess import (
    FALLBACK_AUTHOR,
    FREQUENCIES_PER_WORK_UNIT,
    MAXIMUM_FREQUENCY,
    MINIMUM_FREQUENCY,
    PREFERRED_AUTHOR,
    PREFERRED_EXPTIME_SECONDS,
    TOTAL_FREQUENCIES,
    WORKLOAD_ID,
    calculate_astropy_reference,
    expected_work_unit_count,
    frequency_step,
    prepare_light_curve,
)


PREPARER_ID = "openstar.tess-known-target-blind-benchmark-preparer"
PREPARER_SCHEMA_VERSION = 1
OWNED_FILES = frozenset(("dataset.json", "project.json"))


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
    if not result:
        raise ValueError("identifier must contain a letter or number")
    return result


def _search(tic: int, sector: int, *, author: str, exptime: int | None):
    options = {"mission": "TESS", "sector": sector, "author": author}
    if exptime is not None:
        options["exptime"] = exptime
    return lk.search_lightcurve(f"TIC {tic}", **options)


def _exptime(result, index: int) -> float:
    value = result.exptime[index]
    return float(getattr(value, "value", value))


def select_product(tic: int, sector: int):
    """Select SPOC 120 s, otherwise the shortest deterministic TESS-SPOC row."""
    preferred = _search(
        tic, sector, author=PREFERRED_AUTHOR, exptime=PREFERRED_EXPTIME_SECONDS
    )
    if len(preferred):
        return preferred[0:1], PREFERRED_AUTHOR, _exptime(preferred, 0)

    fallback = _search(tic, sector, author=FALLBACK_AUTHOR, exptime=None)
    if not len(fallback):
        raise RuntimeError(
            f"No official SPOC or TESS-SPOC light curve for TIC {tic}, Sector {sector}."
        )
    exposures = np.asarray([_exptime(fallback, i) for i in range(len(fallback))])
    finite = np.flatnonzero(np.isfinite(exposures))
    if not len(finite):
        raise RuntimeError(
            f"TESS-SPOC products for TIC {tic}, Sector {sector} have no finite cadence."
        )
    index = int(finite[np.argmin(exposures[finite])])
    return fallback[index : index + 1], FALLBACK_AUTHOR, _exptime(fallback, index)


def _array_sha256(values: np.ndarray) -> str:
    frozen = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(frozen.tobytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _owned_manifest(project_id: str) -> dict:
    return {
        "preparerID": PREPARER_ID,
        "schemaVersion": PREPARER_SCHEMA_VERSION,
        "ownedFiles": sorted(OWNED_FILES),
        "projectID": project_id,
    }


def _validate_owned_output(output: Path, project_id: str) -> None:
    """Fail closed unless output is exactly a directory created by this preparer."""
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("Safe overwrite requires a non-symlink output directory.")
    entries = list(output.iterdir())
    if {entry.name for entry in entries} != OWNED_FILES:
        raise RuntimeError("Safe overwrite refuses missing or unexpected output entries.")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise RuntimeError("Safe overwrite requires exact regular owned files; symlinks are refused.")
    try:
        manifest = json.loads((output / "project.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Safe overwrite refuses a malformed project.json.") from error
    if not isinstance(manifest, dict) or manifest.get("preparer") != _owned_manifest(project_id):
        raise RuntimeError("Safe overwrite ownership marker, schema, or project ID mismatch.")
    if manifest.get("id") != project_id:
        raise RuntimeError("Safe overwrite project ID mismatch.")


def _remove_owned_directory(path: Path) -> None:
    """Remove an already-validated owned directory without recursive deletion."""
    for name in OWNED_FILES:
        (path / name).unlink()
    path.rmdir()


def _validate_staged_output(stage: Path, output: Path, project_id: str) -> None:
    _validate_owned_output(stage, project_id)
    manifest = json.loads((stage / "project.json").read_text(encoding="utf-8"))
    dataset = json.loads((stage / "dataset.json").read_text(encoding="utf-8"))
    entry = manifest["datasets"][0]
    if entry.get("path") != str((output / "dataset.json").resolve()):
        raise RuntimeError("Staged manifest has an invalid final dataset path.")
    if entry.get("ticID") != (dataset.get("source") or {}).get("ticID"):
        raise RuntimeError("Staged manifest and dataset TIC IDs disagree.")
    if hashlib.sha256((stage / "dataset.json").read_bytes()).hexdigest() != entry.get(
        "datasetSHA256"
    ):
        raise RuntimeError("Staged dataset hash mismatch.")


def prepare_benchmark(
    *, tic: int, primary_sector: int, blind_label: str, project_id: str,
    output_dir: str | Path, overwrite: bool = False,
) -> Path:
    requested_output = Path(output_dir).expanduser()
    if requested_output.is_symlink():
        raise RuntimeError("Safe overwrite requires a non-symlink output directory.")
    output = requested_output.resolve()
    manifest_path = output / "project.json"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output project already exists: {output}")
        _validate_owned_output(output, project_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    backup = output.with_name(f".{output.name}.replace-backup")
    if backup.exists() or backup.is_symlink():
        stage.rmdir()
        raise RuntimeError(f"Safe overwrite backup path already exists: {backup}")

    try:
        selected, author, cadence = select_product(tic, primary_sector)
        light_curve = selected.download(quality_bitmask="default")
        if light_curve is None:
            raise RuntimeError("The selected official TESS light curve could not be downloaded.")
        actual_sector = getattr(light_curve, "sector", None)
        if actual_sector is None:
            actual_sector = getattr(light_curve, "meta", {}).get("SECTOR", primary_sector)
        if int(actual_sector) != primary_sector:
            raise RuntimeError(f"Downloaded Sector {actual_sector}; expected {primary_sector}.")

        original_count = len(light_curve)
        finite_count = int(np.count_nonzero(
            np.isfinite(np.asarray(light_curve.time.value, dtype=np.float64))
            & np.isfinite(np.asarray(light_curve.flux.value, dtype=np.float64))
        ))
        times, flux, origin = prepare_light_curve(light_curve)
        reference = calculate_astropy_reference(times, flux)
        dataset_id = f"tess-tic-{tic}-sector-{primary_sector}"
        dataset = {
        "id": dataset_id,
        "targetName": blind_label,
        "mission": "TESS",
        "source": {
            "archive": "MAST", "author": author, "ticID": tic,
            "sector": primary_sector, "cadenceSeconds": cadence,
            "qualityBitmask": "default", "originalTimeOriginDays": origin,
            "originalSampleCount": original_count, "finiteSampleCount": finite_count,
            "distributedSampleCount": len(times),
        },
        "science": {"role": "blind", "catalogAnswerKeyUsed": False},
        "timeUnit": "days", "timeReference": "relative-to-first-distributed-sample",
        "numericRepresentation": "Float32", "fluxUnit": "normalized",
        "times": [float(x) for x in times], "flux": [float(x) for x in flux],
        "hashes": {"timesFloat32SHA256": _array_sha256(times),
                   "fluxFloat32SHA256": _array_sha256(flux)},
        "frequencySearch": {
            "minimumFrequency": MINIMUM_FREQUENCY, "maximumFrequency": MAXIMUM_FREQUENCY,
            "frequencyStep": frequency_step(), "totalFrequencies": TOTAL_FREQUENCIES,
            "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
        },
        "reference": reference,
        }
        staged_dataset_path = stage / "dataset.json"
        _write_json(staged_dataset_path, dataset)
        manifest = {
            "id": project_id, "name": blind_label, "workloadID": WORKLOAD_ID,
            "preparer": _owned_manifest(project_id),
            "datasets": [{
            "id": dataset_id, "path": str((output / "dataset.json").resolve()),
            "targetName": blind_label,
            "ticID": tic, "sector": primary_sector, "author": author,
            "cadenceSeconds": cadence, "role": "blind",
            "datasetSHA256": hashlib.sha256(staged_dataset_path.read_bytes()).hexdigest(),
            "workUnitCount": expected_work_unit_count(),
            }],
        }
        _write_json(stage / "project.json", manifest)
        _validate_staged_output(stage, output, project_id)

        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup.exists() and not output.exists():
                backup.rename(output)
            raise
        if backup.exists():
            _remove_owned_directory(backup)
    except BaseException:
        if stage.exists():
            entries = list(stage.iterdir())
            if all(entry.name in OWNED_FILES and entry.is_file() and not entry.is_symlink()
                   for entry in entries):
                for entry in entries:
                    entry.unlink()
                stage.rmdir()
        raise
    return manifest_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tic", required=True, type=int)
    parser.add_argument("--primary-sector", required=True, type=int)
    parser.add_argument("--blind-label", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace only a marked project with the same project id.")
    args = parser.parse_args(argv)
    if args.tic <= 0:
        parser.error("--tic must be positive")
    if args.primary_sector <= 0:
        parser.error("--primary-sector must be positive")
    if not args.blind_label.strip():
        parser.error("--blind-label must not be empty")
    try:
        _slug(args.project_id)
    except ValueError as error:
        parser.error(f"invalid --project-id: {error}")
    if args.project_id != _slug(args.project_id):
        parser.error("--project-id must already be canonical lowercase [a-z0-9._-]")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    path = prepare_benchmark(
        tic=args.tic, primary_sector=args.primary_sector, blind_label=args.blind_label,
        project_id=args.project_id, output_dir=args.output_dir, overwrite=args.overwrite,
    )
    print(f"Prepared blind TESS project: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
