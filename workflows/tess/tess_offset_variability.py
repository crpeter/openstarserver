from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .tess_multisource_residual import (
    GAUSSIAN_RADIUS_PIXELS,
    GAUSSIAN_SIGMA_PIXELS,
    MIN_COMPONENT_SAMPLES,
    _gaussian_template,
    _prewhiten_cube_raw,
)
from .tess_residual_localization import (
    GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
    LOMB_SCARGLE_WORKLOAD_ALIASES,
    MAX_CADENCES,
    _background_subtract_cube,
    _download_tpf,
    _float,
    _int,
    _load_json,
    _safe,
    _sector_candidates,
    _time_warp,
    _uniform_indices,
    _write_json,
)

TOTAL_FREQUENCIES = 8_192
FREQUENCIES_PER_WORK_UNIT = 2_048
FREQUENCY_HALF_WIDTH_FRACTION = 0.20
MIN_PEAK_PROMINENCE = 1.5
MIN_OBSERVED_CYCLES = 2.0
MIN_CANDIDATE_POWER = 0.08
MIN_INDEPENDENT_SUPPORT = 3
REFERENCE_FREQUENCY_TOLERANCE_FRACTION = 0.12
DOMINANCE_POWER_RATIO = 1.25
MAX_NUISANCE_SOURCES = 2
MIN_TEMPLATE_SEPARATION_PIXELS = 0.55


def _skycoord(ra_deg: float, dec_deg: float):
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    return SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")


def _frequency_search(reference_frequency: float) -> dict[str, Any]:
    minimum = float(reference_frequency) * (1.0 - FREQUENCY_HALF_WIDTH_FRACTION)
    maximum = float(reference_frequency) * (1.0 + FREQUENCY_HALF_WIDTH_FRACTION)
    if minimum <= 0 or maximum <= minimum:
        raise RuntimeError("Invalid v20.14 catalog-counterpart frequency search.")
    return {
        "minimumFrequency": minimum,
        "maximumFrequency": maximum,
        "frequencyStep": (maximum - minimum) / (TOTAL_FREQUENCIES - 1),
        "totalFrequencies": TOTAL_FREQUENCIES,
        "frequenciesPerWorkUnit": FREQUENCIES_PER_WORK_UNIT,
    }


def _best_offset_summary(multisource_summary: dict[str, Any]) -> dict[str, Any]:
    component_id = str(multisource_summary.get("bestOffsetComponentID") or "").strip()
    if not component_id:
        raise RuntimeError("v20.14 requires v20.12 to select a best offset component.")
    summary = next(
        (
            item
            for item in multisource_summary.get("componentSummaries") or []
            if str(item.get("componentID")) == component_id
        ),
        None,
    )
    if summary is None:
        raise RuntimeError(f"v20.14 cannot find the v20.12 summary for {component_id}.")
    return summary


def _catalog_candidate(catalog_identification: dict[str, Any]) -> dict[str, Any]:
    historical_contract = catalog_identification.get("classification") in {
        "CATALOG_COUNTERPART_IDENTIFIED",
        "KNOWN_VARIABLE_CATALOG_COUNTERPART_IDENTIFIED",
    }
    authoritative_contract = (
        catalog_identification.get("recommendedNextTest")
        == "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION"
        and catalog_identification.get("physicalMechanismResolved") is False
    )
    if not historical_contract and not authoritative_contract:
        raise RuntimeError("v20.14 requires a justified preferred catalog counterpart.")
    candidate = (
        catalog_identification.get("preferredCandidate")
        if authoritative_contract
        else catalog_identification.get("bestCandidate")
    ) or {}
    ra = _float(candidate.get("raDeg"))
    dec = _float(candidate.get("decDeg"))
    ids = candidate.get("catalogIDs") or {}
    if ra is None or dec is None:
        raise RuntimeError("v20.14 requires RA/Dec for the preferred catalog counterpart.")
    if _int(ids.get("ticID")) is None and _int(ids.get("gaiaDR3SourceID")) is None:
        raise RuntimeError("v20.14 requires a TIC or Gaia identifier for the preferred counterpart.")
    return candidate


def _candidate_label(candidate: dict[str, Any]) -> str:
    ids = candidate.get("catalogIDs") or {}
    tic_id = _int(ids.get("ticID"))
    gaia_id = _int(ids.get("gaiaDR3SourceID"))
    if tic_id is not None:
        return f"TIC {tic_id}"
    if gaia_id is not None:
        return f"Gaia DR3 {gaia_id}"
    return "offset counterpart"


def _nuisance_catalog_sources(
    *,
    offset_source_identification: dict[str, Any],
    best_candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    best_ids = best_candidate.get("catalogIDs") or {}
    best_tic = _int(best_ids.get("ticID"))
    best_gaia = _int(best_ids.get("gaiaDR3SourceID"))
    ranked: list[tuple[float, float, dict[str, Any]]] = []
    for candidate in offset_source_identification.get("catalogCandidates") or []:
        ids = candidate.get("catalogIDs") or {}
        tic_id = _int(ids.get("ticID"))
        gaia_id = _int(ids.get("gaiaDR3SourceID"))
        if (best_tic is not None and tic_id == best_tic) or (
            best_gaia is not None and gaia_id == best_gaia
        ):
            continue
        ra = _float(candidate.get("raDeg"))
        dec = _float(candidate.get("decDeg"))
        if ra is None or dec is None:
            continue
        tic = candidate.get("tic") or {}
        gaia = candidate.get("gaiaDR3") or {}
        mag = _float(tic.get("tmag"))
        if mag is None:
            mag = _float(gaia.get("gMag"))
        if mag is None:
            mag = 99.0
        separation = _float(candidate.get("separationArcsec"))
        if separation is None:
            separation = 999.0
        ranked.append((float(mag), float(separation), candidate))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [dict(item[2]) for item in ranked[:MAX_NUISANCE_SOURCES]]


def _catalog_guided_series(
    *,
    residual_cube: np.ndarray,
    valid_pixels: np.ndarray,
    tpf: Any,
    target_sky: Any,
    candidate_sky: Any,
    nuisance_sources: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows, cols = valid_pixels.shape
    source_defs: list[tuple[str, Any]] = [
        ("target-control", target_sky),
        ("catalog-counterpart", candidate_sky),
    ]
    for index, source in enumerate(nuisance_sources, start=1):
        ra = _float(source.get("raDeg"))
        dec = _float(source.get("decDeg"))
        if ra is None or dec is None:
            continue
        source_defs.append((f"nuisance-{index}", _skycoord(ra, dec)))

    templates: list[np.ndarray] = []
    used: list[dict[str, Any]] = []
    centers: list[tuple[float, float]] = []
    for component_id, coordinate in source_defs:
        try:
            x, y = tpf.wcs.world_to_pixel(coordinate)
            x = float(x)
            y = float(y)
        except Exception:
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if x < -GAUSSIAN_RADIUS_PIXELS or x > cols - 1 + GAUSSIAN_RADIUS_PIXELS:
            continue
        if y < -GAUSSIAN_RADIUS_PIXELS or y > rows - 1 + GAUSSIAN_RADIUS_PIXELS:
            continue
        if component_id.startswith("nuisance-") and any(
            math.hypot(x - px, y - py) < MIN_TEMPLATE_SEPARATION_PIXELS
            for px, py in centers
        ):
            continue
        template = _gaussian_template(
            rows=rows,
            cols=cols,
            x=x,
            y=y,
            valid_pixels=valid_pixels,
        )
        if float(np.linalg.norm(template)) <= 1e-12:
            continue
        templates.append(template)
        centers.append((x, y))
        used.append({"componentID": component_id, "pixelCenter": {"x": x, "y": y}})

    required_ids = {item["componentID"] for item in used}
    if not {"target-control", "catalog-counterpart"}.issubset(required_ids):
        raise RuntimeError("Target and catalog-counterpart templates are not both usable in this TPF.")

    background = np.asarray(valid_pixels, dtype=np.float64).reshape(-1)
    bg_norm = float(np.linalg.norm(background))
    if bg_norm > 0:
        templates.append(background / bg_norm)

    spatial = np.column_stack(templates)
    condition_number = float(np.linalg.cond(spatial))
    pinv = np.linalg.pinv(spatial)
    flat = residual_cube.reshape(len(residual_cube), -1).astype(np.float64)
    coefficients = (pinv @ flat.T).T

    series: dict[str, np.ndarray] = {}
    for index, source in enumerate(used):
        values = np.asarray(coefficients[:, index], dtype=np.float64)
        values -= float(np.mean(values))
        std = float(np.std(values))
        if not math.isfinite(std) or std <= 1e-12:
            continue
        series[source["componentID"]] = values / std

    if "target-control" not in series or "catalog-counterpart" not in series:
        raise RuntimeError("Catalog-guided spatial decomposition did not produce both validation series.")
    return series, {
        "templates": used,
        "designConditionNumber": condition_number,
        "gaussianSigmaPixels": GAUSSIAN_SIGMA_PIXELS,
        "gaussianRadiusPixels": GAUSSIAN_RADIUS_PIXELS,
    }


def build_offset_source_variability_project(
    *,
    source_project_path: str | Path,
    source_dataset_entry: dict[str, Any],
    target_tic_id: int,
    identity: dict[str, Any],
    primary_sector: int | None,
    independent_spec: dict[str, Any],
    multisource_summary: dict[str, Any],
    offset_source_identification: dict[str, Any],
    output_dir: str | Path,
    investigation_id: str,
    physical_period_days: float | None = None,
    nonstationary_summary: dict[str, Any] | None = None,
    reference_family_period_days: float | None = None,
    harmonic_orders: tuple[int, ...] | list[int] | None = None,
    physical_cycle_resolved: bool | None = None,
    residual_reference_frequency: float | None = None,
    residual_time_reference_days: float | None = None,
    fractional_frequency_drift_per_day: float | None = None,
    frozen_sectors: list[int] | None = None,
    family_residual_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_project = _load_json(source_project_path)
    source_workload_id = str(source_project.get("workloadID") or "")
    if source_workload_id and source_workload_id not in LOMB_SCARGLE_WORKLOAD_ALIASES:
        raise RuntimeError(
            "v20.14 requires a Lomb-Scargle-compatible source project; "
            f"found workloadID={source_workload_id}."
        )
    if offset_source_identification.get("recommendedNextTest") not in {
        "OFFSET_SOURCE_VARIABILITY_VALIDATION",
        "OFFSET_SOURCE_VARIABILITY_MATCH_TEST",
        "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
    }:
        raise RuntimeError("v20.14 requires v20.13 to recommend offset-source variability validation.")

    nonstationary_summary = nonstationary_summary or {}
    best_offset = _best_offset_summary(multisource_summary)
    reference_frequency = _float(residual_reference_frequency)
    if reference_frequency is None or reference_frequency <= 0:
        reference_frequency = _float(best_offset.get("combinedFrequency"))
    if reference_frequency is None or reference_frequency <= 0:
        reference_frequency = _float(nonstationary_summary.get("preferredFrequencyAtReference"))
    q = _float(fractional_frequency_drift_per_day)
    if q is None:
        q = _float(nonstationary_summary.get("fractionalFrequencyDriftPerDay"))
    time_reference = _float(residual_time_reference_days)
    if time_reference is None:
        time_reference = _float(nonstationary_summary.get("timeReferenceDays"))
    if reference_frequency is None or reference_frequency <= 0 or q is None or time_reference is None:
        raise RuntimeError("v20.14 requires the completed v20.9 drift model and v20.12 offset frequency.")

    target_meta = ((identity.get("tic") or {}).get("metadata") or {})
    target_ra = _float(target_meta.get("raDeg"))
    target_dec = _float(target_meta.get("decDeg"))
    if target_ra is None or target_dec is None:
        raise RuntimeError("v20.14 requires TIC RA/Dec from the identity stage.")

    candidate = _catalog_candidate(offset_source_identification)
    candidate_ra = float(candidate["raDeg"])
    candidate_dec = float(candidate["decDeg"])
    candidate_ids = candidate.get("catalogIDs") or {}
    nuisance_sources = _nuisance_catalog_sources(
        offset_source_identification=offset_source_identification,
        best_candidate=candidate,
    )

    signal_sectors = [
        int(value)
        for value in ((nonstationary_summary.get("preferredModel") or {}).get("signalSectors") or [])
        if _int(value) is not None
    ]
    sectors = ([(int(value), "frozen-family-residual-bridge") for value in frozen_sectors]
               if frozen_sectors else _sector_candidates(
                   primary_sector=primary_sector,
                   independent_spec=independent_spec,
                   signal_sectors=signal_sectors))
    if not sectors:
        raise RuntimeError("v20.14 found no frozen sectors to validate the catalog counterpart.")

    family_period = _float(reference_family_period_days)
    if family_period is None:
        family_period = _float(physical_period_days)
    if family_period is None or family_period <= 0:
        raise RuntimeError("v20.14 requires a persisted family prewhitening period.")
    orders = tuple(int(value) for value in (harmonic_orders or (1, 2)))
    if not orders or any(value <= 0 for value in orders):
        raise RuntimeError("v20.14 requires positive persisted harmonic orders.")
    physical_frequency = 1.0 / family_period
    cycle_resolved = (bool(physical_cycle_resolved) if physical_cycle_resolved is not None
                      else reference_family_period_days is None and physical_period_days is not None)
    search = _frequency_search(float(reference_frequency))
    target_sky = _skycoord(float(target_ra), float(target_dec))
    candidate_sky = _skycoord(candidate_ra, candidate_dec)
    root = Path(output_dir) / "offset-source-variability"
    root.mkdir(parents=True, exist_ok=True)
    source_base_id = str(source_dataset_entry.get("id") or f"tic-{target_tic_id}")

    dataset_entries: list[dict[str, Any]] = []
    prepared_series: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    decomposition_diagnostics: list[dict[str, Any]] = []
    combined: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "target-control": [],
        "catalog-counterpart": [],
    }

    for sector_index, (sector, role) in enumerate(sectors, start=1):
        print(
            f"   Sector {sector} ({sector_index}/{len(sectors)}): catalog-guided target/counterpart deblend",
            flush=True,
        )
        try:
            tpf, source = _download_tpf(
                tic_id=int(target_tic_id),
                sector=int(sector),
                ra_deg=float(target_ra),
                dec_deg=float(target_dec),
            )
            absolute_times = np.asarray(tpf.time.value, dtype=np.float64)
            flux = getattr(tpf.flux, "value", tpf.flux)
            if np.ma.isMaskedArray(flux):
                flux = np.ma.filled(flux, np.nan)
            cube = np.asarray(flux, dtype=np.float64)
            finite_time = np.isfinite(absolute_times)
            finite_frame = np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
            keep = finite_time & finite_frame
            absolute_times = absolute_times[keep]
            cube = cube[keep]
            if len(absolute_times) < MIN_COMPONENT_SAMPLES:
                raise RuntimeError(f"Only {len(absolute_times)} usable cadences.")

            indices = _uniform_indices(len(absolute_times), MAX_CADENCES)
            absolute_times = absolute_times[indices]
            cube = cube[indices]
            corrected, _ = _background_subtract_cube(cube)
            residual_cube, valid_pixels = _prewhiten_cube_raw(
                absolute_times=absolute_times,
                cube=corrected,
                physical_frequency=physical_frequency,
                harmonic_orders=orders,
            )
            component_series, diagnostic = _catalog_guided_series(
                residual_cube=residual_cube,
                valid_pixels=valid_pixels,
                tpf=tpf,
                target_sky=target_sky,
                candidate_sky=candidate_sky,
                nuisance_sources=nuisance_sources,
            )
            diagnostic.update(
                {
                    "sector": int(sector),
                    "role": role,
                    "sourceType": source.get("sourceType"),
                    "author": source.get("author"),
                    "cadenceSeconds": source.get("cadenceSeconds"),
                }
            )
            decomposition_diagnostics.append(diagnostic)

            relative_times = absolute_times - float(time_reference)
            warped = _time_warp(relative_times, float(q))
            local_times = warped - float(np.min(warped))
            for component_id in ("target-control", "catalog-counterpart"):
                values = component_series[component_id]
                dataset_id = f"{source_base_id}-offset-validation-{component_id}-sector-{sector}-v1"
                target_name = (
                    f"{source_dataset_entry.get('targetName') or source_base_id} "
                    f"offset validation {component_id} sector {sector}"
                )
                output_path = root / f"{_safe(dataset_id)}.json"
                dataset = {
                    "id": dataset_id,
                    "targetName": target_name,
                    "times": np.asarray(local_times, dtype=np.float32).tolist(),
                    "flux": np.asarray(values, dtype=np.float32).tolist(),
                    "frequencySearch": search,
                    "reference": {},
                    "science": {
                        "role": "offset-source-variability-validation",
                        "componentID": component_id,
                        "sector": int(sector),
                        "sectorRole": role,
                        "referenceFrequency": float(reference_frequency),
                        "fractionalFrequencyDriftPerDay": float(q),
                        "referenceFamilyPeriodDays": float(family_period),
                        "subtractedHarmonicOrders": list(orders),
                        "physicalCycleResolved": cycle_resolved,
                        "catalogCounterpart": {
                            "ticID": _int(candidate_ids.get("ticID")),
                            "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
                            "raDeg": candidate_ra,
                            "decDeg": candidate_dec,
                        },
                    },
                    "source": {
                        "mission": "TESS",
                        "sector": int(sector),
                        "distributedSamples": int(len(local_times)),
                        "baselineDays": float(np.max(absolute_times) - np.min(absolute_times)),
                        "timeReferenceDays": float(time_reference),
                        "sourceType": source.get("sourceType"),
                        "author": source.get("author"),
                        "cadenceSeconds": source.get("cadenceSeconds"),
                    },
                }
                _write_json(output_path, dataset)
                dataset_entries.append(
                    {"id": dataset_id, "path": str(output_path.resolve()), "targetName": target_name}
                )
                prepared_series.append(
                    {
                        "datasetID": dataset_id,
                        "datasetPath": str(output_path.resolve()),
                        "componentID": component_id,
                        "sector": int(sector),
                        "role": role,
                        "combined": False,
                        "baselineDays": float(np.max(absolute_times) - np.min(absolute_times)),
                    }
                )
                combined[component_id].append(
                    (np.asarray(warped, dtype=np.float64), np.asarray(values, dtype=np.float64))
                )
            print(
                f"      extracted target-control + {_candidate_label(candidate)} residual series",
                flush=True,
            )
        except Exception as exc:
            errors.append({"sector": int(sector), "error": f"{type(exc).__name__}: {exc}"})
            print(f"      unavailable: {type(exc).__name__}: {exc}", flush=True)

    for component_id in ("target-control", "catalog-counterpart"):
        pieces = combined.get(component_id) or []
        if len(pieces) < 2:
            continue
        all_times = np.concatenate([item[0] for item in pieces])
        all_flux = np.concatenate([item[1] for item in pieces])
        order = np.argsort(all_times)
        all_times = all_times[order]
        all_flux = all_flux[order]
        local_times = all_times - float(np.min(all_times))
        dataset_id = f"{source_base_id}-offset-validation-{component_id}-combined-v1"
        target_name = (
            f"{source_dataset_entry.get('targetName') or source_base_id} "
            f"offset validation {component_id} combined"
        )
        output_path = root / f"{_safe(dataset_id)}.json"
        dataset = {
            "id": dataset_id,
            "targetName": target_name,
            "times": np.asarray(local_times, dtype=np.float32).tolist(),
            "flux": np.asarray(all_flux, dtype=np.float32).tolist(),
            "frequencySearch": search,
            "reference": {},
            "science": {
                "role": "offset-source-variability-validation-combined",
                "componentID": component_id,
                "referenceFrequency": float(reference_frequency),
                "fractionalFrequencyDriftPerDay": float(q),
                "referenceFamilyPeriodDays": float(family_period),
                "subtractedHarmonicOrders": list(orders),
                "physicalCycleResolved": cycle_resolved,
                "catalogCounterpart": {
                    "ticID": _int(candidate_ids.get("ticID")),
                    "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
                    "raDeg": candidate_ra,
                    "decDeg": candidate_dec,
                },
            },
            "source": {
                "mission": "TESS",
                "distributedSamples": int(len(local_times)),
                "baselineDays": float(np.max(all_times) - np.min(all_times)),
                "timeReferenceDays": float(time_reference),
                "combinedSectors": True,
            },
        }
        _write_json(output_path, dataset)
        dataset_entries.append(
            {"id": dataset_id, "path": str(output_path.resolve()), "targetName": target_name}
        )
        prepared_series.append(
            {
                "datasetID": dataset_id,
                "datasetPath": str(output_path.resolve()),
                "componentID": component_id,
                "sector": None,
                "role": "combined",
                "combined": True,
                "baselineDays": float(np.max(all_times) - np.min(all_times)),
            }
        )

    if not dataset_entries:
        raise RuntimeError("v20.14 could not prepare any catalog-counterpart validation datasets.")

    project_id = (
        f"{source_project['id']}.investigation.{_safe(investigation_id)}."
        "offset-source-variability-validation-v1"
    )
    manifest = {
        "id": project_id,
        "name": f"{source_project.get('name', source_project['id'])} — offset source variability validation",
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "datasets": dataset_entries,
        "investigation": {
            "sourceProjectID": source_project["id"],
            "sourceDatasetID": source_dataset_entry.get("id"),
            "purpose": "offset-source-variability-validation",
            "workerSemantics": (
                "Each dataset is a catalog-guided spatially deblended, established-family-prewhitened, "
                "v20.9 drift-corrected source-component light curve. Workers execute ordinary Lomb-Scargle only."
            ),
            "referenceFrequency": float(reference_frequency),
            "fractionalFrequencyDriftPerDay": float(q),
            "referenceFamilyPeriodDays": float(family_period),
            "subtractedHarmonicOrders": list(orders),
            "physicalCycleResolved": cycle_resolved,
            "catalogCounterpart": {
                "ticID": _int(candidate_ids.get("ticID")),
                "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
                "raDeg": candidate_ra,
                "decDeg": candidate_dec,
            },
        },
    }
    manifest_path = root / f"{_safe(project_id)}.json"
    _write_json(manifest_path, manifest)
    work_units_per_dataset = math.ceil(TOTAL_FREQUENCIES / FREQUENCIES_PER_WORK_UNIT)
    return {
        "available": True,
        "version": "openstar.tess-offset-source-variability-validation-preparation.v1",
        "projectID": project_id,
        "projectPath": str(manifest_path.resolve()),
        "workloadID": GENERIC_LOMB_SCARGLE_WORKLOAD_ID,
        "workerSemantics": "generic-lomb-scargle-on-catalog-guided-deblended-drift-corrected-source-series",
        "targetTIC": int(target_tic_id),
        "catalogCounterpart": {
            "ticID": _int(candidate_ids.get("ticID")),
            "gaiaDR3SourceID": _int(candidate_ids.get("gaiaDR3SourceID")),
            "raDeg": candidate_ra,
            "decDeg": candidate_dec,
            "catalogSeparationArcsec": _float(candidate.get("separationArcsec")),
        },
        "catalogCounterpartIdentification": {
            "classification": offset_source_identification.get("classification"),
            "motivatingComponentID": candidate.get("motivatingComponentID"),
        },
        "bestOffsetComponentID": multisource_summary.get("bestOffsetComponentID"),
        "referenceFrequency": float(reference_frequency),
        "referencePeriodDays": float(1.0 / reference_frequency),
        "referenceFamilyPeriodDays": float(family_period),
        "subtractedHarmonicOrders": list(orders),
        "physicalCycleResolved": cycle_resolved,
        "familyResidualModelProvenance": dict(family_residual_provenance or {}),
        "frozenSectors": [int(sector) for sector, _ in sectors],
        "fractionalFrequencyDriftPerDay": float(q),
        "timeReferenceDays": float(time_reference),
        "frequencySearch": search,
        "preparedSeries": prepared_series,
        "decompositionDiagnostics": decomposition_diagnostics,
        "nuisanceCatalogSources": nuisance_sources,
        "errors": errors,
        "workUnitsPerDataset": work_units_per_dataset,
        "totalWorkUnits": int(len(dataset_entries) * work_units_per_dataset),
        "interpretationGuard": (
            "The catalog-guided spatial fit uses deterministic Gaussian source templates rather than a calibrated "
            "TESS PRF. It is a direct variability-association test, not precision deblended photometry."
        ),
    }


def _boundary_hit(frequency: float | None, search: dict[str, Any]) -> bool:
    if frequency is None:
        return True
    minimum = _float(search.get("minimumFrequency"))
    maximum = _float(search.get("maximumFrequency"))
    step = _float(search.get("frequencyStep"))
    if minimum is None or maximum is None:
        return True
    margin = max((step or 0.0) * 2.0, (maximum - minimum) * 0.002)
    return frequency <= minimum + margin or frequency >= maximum - margin


def _result_record(dataset: dict[str, Any], meta: dict[str, Any], preparation: dict[str, Any]) -> dict[str, Any]:
    frequency = _float(dataset.get("candidateFrequency"))
    period = _float(dataset.get("candidatePeriodDays"))
    power = _float(dataset.get("candidatePower"))
    prominence = _float(dataset.get("candidatePeakProminenceRatio"))
    status = str(dataset.get("periodStatus") or "").upper()
    confidence = str(dataset.get("periodConfidence") or "none").lower()
    baseline = _float(meta.get("baselineDays")) or 0.0
    observed_cycles = (baseline / period) if period and period > 0 else 0.0
    reference_frequency = float(preparation["referenceFrequency"])
    relative_difference = (
        abs(float(frequency) - reference_frequency) / reference_frequency
        if frequency is not None
        else None
    )
    rayleigh = (1.0 / baseline) if baseline > 0 else None
    reference_consistent = bool(
        frequency is not None
        and (
            (relative_difference is not None and relative_difference <= REFERENCE_FREQUENCY_TOLERANCE_FRACTION)
            or (rayleigh is not None and abs(float(frequency) - reference_frequency) <= rayleigh)
        )
    )
    accepted = bool(
        status == "RELIABLE"
        and confidence in {"high", "medium"}
        and power is not None
        and power >= MIN_CANDIDATE_POWER
        and prominence is not None
        and prominence >= MIN_PEAK_PROMINENCE
        and observed_cycles >= MIN_OBSERVED_CYCLES
        and not _boundary_hit(frequency, preparation.get("frequencySearch") or {})
        and reference_consistent
    )
    return {
        **meta,
        "candidateFrequency": frequency,
        "candidatePeriodDays": period,
        "candidatePower": power,
        "candidatePeakProminenceRatio": prominence,
        "periodStatus": status,
        "periodConfidence": confidence,
        "observedCycles": observed_cycles,
        "relativeFrequencyDifferenceFromReference": relative_difference,
        "referenceConsistent": reference_consistent,
        "acceptedResidualMatch": accepted,
    }


def _component_summary(component_id: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    component = [item for item in results if item.get("componentID") == component_id]
    sectors = [item for item in component if not item.get("combined")]
    independent = [item for item in sectors if item.get("role") == "independent"]
    accepted_independent = [item for item in independent if item.get("acceptedResidualMatch")]
    accepted_all = [item for item in sectors if item.get("acceptedResidualMatch")]
    combined = next((item for item in component if item.get("combined")), None)
    powers = [
        float(item["candidatePower"])
        for item in accepted_all
        if _float(item.get("candidatePower")) is not None
    ]
    return {
        "componentID": component_id,
        "independentSupportCount": len(accepted_independent),
        "independentSupportingSectors": sorted(int(item["sector"]) for item in accepted_independent),
        "allSupportingSectors": sorted(
            int(item["sector"])
            for item in accepted_all
            if item.get("sector") is not None
        ),
        "medianAcceptedSectorPower": statistics.median(powers) if powers else None,
        "combinedAccepted": bool(combined and combined.get("acceptedResidualMatch")),
        "combinedPower": combined.get("candidatePower") if combined else None,
        "combinedPeriodDays": combined.get("candidatePeriodDays") if combined else None,
        "combinedFrequency": combined.get("candidateFrequency") if combined else None,
        "combinedProminence": combined.get("candidatePeakProminenceRatio") if combined else None,
    }


def interpret_offset_source_variability_project(
    *,
    project_status: dict[str, Any],
    preparation: dict[str, Any],
) -> dict[str, Any]:
    prepared = {
        str(item.get("datasetID")): item
        for item in preparation.get("preparedSeries") or []
    }
    results: list[dict[str, Any]] = []
    for dataset in project_status.get("datasets") or []:
        dataset_id = str(dataset.get("datasetID") or dataset.get("id") or "")
        meta = prepared.get(dataset_id)
        if meta is None:
            continue
        results.append(_result_record(dataset, meta, preparation))

    target = _component_summary("target-control", results)
    counterpart = _component_summary("catalog-counterpart", results)
    target_sector_results = [
        item for item in results
        if item.get("componentID") == "target-control" and not item.get("combined")
    ]
    counterpart_sector_results = [
        item for item in results
        if item.get("componentID") == "catalog-counterpart" and not item.get("combined")
    ]
    target_support = int(target.get("independentSupportCount") or 0)
    candidate_support = int(counterpart.get("independentSupportCount") or 0)
    target_power = float(target.get("combinedPower") or 0.0)
    candidate_power = float(counterpart.get("combinedPower") or 0.0)
    target_present = target_support >= MIN_INDEPENDENT_SUPPORT and bool(target.get("combinedAccepted"))
    candidate_present = candidate_support >= MIN_INDEPENDENT_SUPPORT and bool(counterpart.get("combinedAccepted"))
    candidate_dominant = bool(
        candidate_present
        and (
            not target_present
            or target_power <= 0
            or candidate_power >= target_power * DOMINANCE_POWER_RATIO
        )
    )

    if candidate_dominant:
        classification = "OFFSET_COUNTERPART_VARIABILITY_SUPPORTED"
        origin = "CATALOG_COUNTERPART_SUPPORTED"
        next_test = "TARGET_RESIDUAL_REANALYSIS_AFTER_OFFSET_REMOVAL"
    elif candidate_present and target_present:
        classification = "TARGET_AND_OFFSET_VARIABILITY_SUPPORTED"
        origin = "BLENDED_TARGET_AND_COUNTERPART"
        next_test = "JOINT_TARGET_OFFSET_VARIABILITY_MODEL"
    elif target_present and not candidate_present:
        classification = "OFFSET_COUNTERPART_VARIABILITY_NOT_SUPPORTED"
        origin = "TARGET_CONTROL_DOMINANT"
        next_test = "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
    elif candidate_support >= 2:
        classification = "OFFSET_COUNTERPART_VARIABILITY_SUGGESTIVE"
        origin = "CATALOG_COUNTERPART_SUGGESTIVE"
        next_test = "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"
    else:
        classification = "OFFSET_COUNTERPART_VARIABILITY_UNRESOLVED"
        origin = "UNRESOLVED"
        next_test = "INDEPENDENT_SOURCE_RESOLVED_COUNTERPART_PHOTOMETRY"

    return {
        "version": "openstar.tess-offset-source-variability-validation.v1",
        "distributedValidation": {
            "workloadID": preparation.get("workloadID"),
            "workerSemantics": preparation.get("workerSemantics"),
            "totalWorkUnits": preparation.get("totalWorkUnits"),
            "frequencySearch": preparation.get("frequencySearch"),
        },
        "catalogCounterpart": preparation.get("catalogCounterpart"),
        "bestOffsetComponentID": preparation.get("bestOffsetComponentID"),
        "referenceFrequency": preparation.get("referenceFrequency"),
        "referencePeriodDays": preparation.get("referencePeriodDays"),
        "fractionalFrequencyDriftPerDay": preparation.get("fractionalFrequencyDriftPerDay"),
        "componentResults": results,
        "counterpartPerSectorResults": counterpart_sector_results,
        "targetControlPerSectorResults": target_sector_results,
        "targetControl": target,
        "catalogCounterpartEvidence": counterpart,
        "classification": classification,
        "residualModeOrigin": origin,
        "offsetCounterpartVariabilitySupported": classification == "OFFSET_COUNTERPART_VARIABILITY_SUPPORTED",
        "variabilityConfirmed": candidate_dominant,
        "physicalMechanismResolved": False,
        "claimLevelChanged": False,
        "recommendedNextTest": next_test,
        "preparationErrors": preparation.get("errors") or [],
        "decompositionDiagnostics": preparation.get("decompositionDiagnostics") or [],
        "provenance": {
            "instrument": "TESS",
            "method": "catalog-guided Gaussian-template spatial decomposition",
            "independentTelescopeEvidence": False,
            "establishedFamilyPrewhitened": True,
            "nonstationaryDriftCorrected": True,
        },
        "interpretationGuard": (
            "v20.14 validates only whether the v20.13 catalog counterpart carries residual variability matching "
            "the v20.12 offset component after established-family subtraction and v20.9 drift correction. "
            "The catalog-guided Gaussian spatial fit is not a calibrated TESS PRF solution, and this result "
            "does not alter v20.6's target association for the established 13.72-day family."
        ),
    }
