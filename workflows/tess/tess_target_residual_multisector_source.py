"""v20.19 catalog-constrained localization in unused recurrence sectors.

Only frozen v20.17/v20.18 evidence enters preparation.  Pixel acquisition and
the scientific fit are coordinator responsibilities; generic workers remain
unaware of TESS or catalog sources.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Iterable

from openstar_path_relocation import HistoricalPathResolver, NO_HISTORICAL_PATH_RELOCATION
from openstar_investigation import sha256_json
from .tess_difference_image_constants import SOURCE_MATCH_MAX_PIXELS
from .tess_target_residual_archival_baseline import verified_json_result
from .tess_target_residual_pixel_recurrence import verify_v2017_lineage

PREFIX = "openstar.tess.target-residual-multisector-source."
MAX_COMPETING_SOURCES = 8
MAX_ADDITIONAL_SOURCE_LOCALIZATION_SECTORS = 12
V2018_SECTOR_IDS = (2, 65, 33, 13, 27, 39)


def verify_v2018_lineage(stages: Iterable[Any], *, resolver: HistoricalPathResolver | None = None
                         ) -> dict[str, Any]:
    """Verify the complete immutable v20.17 -> v20.18 terminal boundary."""
    rows = list(stages)
    resolver = resolver or NO_HISTORICAL_PATH_RELOCATION
    v17 = verify_v2017_lineage(rows, resolver=resolver)
    expected = (
        ("036-target-residual-pixel-recurrence-prepare", PREFIX.replace("multisector-source", "pixel-recurrence") + "prepare"),
        ("037-target-residual-pixel-recurrence-run", PREFIX.replace("multisector-source", "pixel-recurrence") + "run"),
        ("038-target-residual-pixel-recurrence-interpret", PREFIX.replace("multisector-source", "pixel-recurrence") + "interpret"),
        ("039-finalize", "openstar.tess.finalize"))
    found = []
    for stage_id, handler in expected:
        matches = [stage for stage in rows if stage.id == stage_id]
        if len(matches) != 1 or matches[0].handler_id != handler or matches[0].status != "COMPLETE":
            raise RuntimeError(f"invalid v20.18 stage {stage_id}")
        found.append(matches[0])
    if found[-1] is not rows[-1] or found[0].triggered_by_stage_id != v17["finalizer"].id:
        raise RuntimeError("v20.18 is not the final, contiguous persisted boundary")
    if any(found[index].triggered_by_stage_id != found[index - 1].id for index in range(1, 4)):
        raise RuntimeError("invalid v20.18 trigger chain")
    prepare, run, science, finalizer = found
    verified_json_result(prepare, "target-residual-pixel-recurrence-prepare-v20.18.json", resolver=resolver)
    verified_json_result(run, "target-residual-pixel-recurrence-run-v20.18.json", resolver=resolver)
    verified_json_result(science, "target-residual-pixel-recurrence-v20.18.json", resolver=resolver)
    verified_json_result(finalizer, "conclusion-v20.18-target-residual-pixel-recurrence-validation.json", resolver=resolver)
    hashes = science.provenance.input_hashes if science.provenance else {}
    if hashes.get("preparation") != sha256_json(prepare.result) or hashes.get("run") != sha256_json(run.result):
        raise RuntimeError("v20.18 interpretation input binding is damaged")
    selected_ids = tuple(int(row["sector"]) for row in prepare.result.get("selectedSectorEvidence") or [])
    if selected_ids != V2018_SECTOR_IDS:
        raise RuntimeError("v20.18 selected-sector boundary is not the frozen six-sector experiment")
    prepare_hashes = prepare.provenance.input_hashes if prepare.provenance else {}
    if prepare_hashes.get("v20.17") != sha256_json(v17["science"].result):
        raise RuntimeError("v20.18 preparation is not bound to v20.17 science")
    required = {"classification": "PIXEL_RECURRENCE_LOCALIZATION_UNRESOLVED",
        "recommendedNextTest": "ADDITIONAL_SOURCE_LOCALIZATION_DATA",
        "sourceAttributionResolved": False, "physicalMechanismResolved": False,
        "crossSectorPhaseUsed": False, "historicalResidualDriftExtrapolated": False}
    if any(science.result.get(key) != value for key, value in required.items()):
        raise RuntimeError("altered v20.18 science boundary")
    if (finalizer.parameters != {"outputSuffix": "v20.18-target-residual-pixel-recurrence-validation"}
            or finalizer.result.get("targetResidualPixelRecurrenceValidation") != science.result
            or finalizer.result.get("recommendedNextTest") != "ADDITIONAL_SOURCE_LOCALIZATION_DATA"):
        raise RuntimeError("altered v20.18 finalizer")
    return {"v20.17": v17, "prepare": prepare, "run": run,
            "science": science, "finalizer": finalizer}


def derive_competing_sources(catalog_hypotheses: list[dict[str, Any]],
                             sector_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select frozen hypotheses within the match radius in any quality sector."""
    qualifying = set()
    for sector in sector_results:
        if sector.get("classification") not in {"UNIQUE_SOURCE_SUPPORTED", "AMBIGUOUS_OR_BLENDED"}:
            continue
        distances = sector.get("distancesPixels") or {}
        for source_id, distance in distances.items():
            if isinstance(distance, (int, float)) and math.isfinite(float(distance)) \
                    and float(distance) <= SOURCE_MATCH_MAX_PIXELS:
                qualifying.add(str(source_id))
    selected = [copy.deepcopy(row) for row in catalog_hypotheses
                if str(row.get("sourceID")) in qualifying]
    if len(selected) > MAX_COMPETING_SOURCES:
        raise RuntimeError("more than eight frozen catalog competitors; refusing truncation")
    if not selected:
        raise RuntimeError("no frozen catalog competitors satisfy the source-match rule")
    return selected


def eligible_additional_sectors(sector_evidence: list[dict[str, Any]],
                                excluded: Iterable[int] = V2018_SECTOR_IDS) -> list[dict[str, Any]]:
    excluded_ids = {int(value) for value in excluded}
    eligible = []
    for row in sector_evidence:
        frequency = row.get("candidateFrequency")
        if (row.get("supportsHistoricalResidualFamily") is True
                and row.get("recurrenceClassification") == "SUPPORTING_HISTORICAL_RESIDUAL_FAMILY"
                and int(row["sector"]) not in excluded_ids
                and isinstance(frequency, (int, float)) and math.isfinite(float(frequency))
                and float(frequency) > 0):
            eligible.append(copy.deepcopy(row))
    eligible.sort(key=lambda row: int(row["sector"]))
    for row in eligible:
        row["selectionReason"] = "unused-v20.17-supporting-sector-with-frozen-positive-frequency"
    return eligible


def derive_additional_sectors(sector_evidence: list[dict[str, Any]],
                              excluded: Iterable[int] = V2018_SECTOR_IDS) -> list[dict[str, Any]]:
    eligible = eligible_additional_sectors(sector_evidence, excluded)
    if len(eligible) <= MAX_ADDITIONAL_SOURCE_LOCALIZATION_SECTORS: return eligible
    # Preregistered deterministic coverage: strongest recurrence evidence, then sector ID.
    ranked = sorted(eligible, key=lambda row: (-float(row.get("recurrenceScore", 0.0)), int(row["sector"])))
    chosen = ranked[:MAX_ADDITIONAL_SOURCE_LOCALIZATION_SECTORS]
    for row in chosen:
        row["selectionReason"] = "bounded-by-descending-frozen-recurrence-score-then-sector-id"
    return chosen


def classify_sector_model(model_evidence: dict[str, Any]) -> dict[str, Any]:
    if model_evidence.get("availability") == "UNAVAILABLE":
        return {"classification": "UNAVAILABLE", "supportedSources": []}
    if model_evidence.get("scientificallyValid") is False:
        return {"classification": "SCIENTIFICALLY_INVALID", "supportedSources": []}
    full = model_evidence.get("fullDataComparison") or {}
    predictive = model_evidence.get("temporalPredictiveValidation") or {}
    sources = list(full.get("bestModelSourceIDs") or [])
    conditional = set(full.get("conditionallyIdentifiableSources") or [])
    supported = [source for source in sources if source in conditional]
    valid = bool(full.get("bestModelIdentifiable") and full.get("completeModelFullRank")
                 and predictive.get("predictiveSupport")
                 and predictive.get("predictiveModel") == full.get("bestModel"))
    classification = ("UNRESOLVED" if not valid else
        "UNIQUE_SOURCE_SUPPORTED" if len(supported) == 1 and len(sources) == 1 else
        "MULTIPLE_SOURCES_SUPPORTED" if len(supported) >= 2 else "UNRESOLVED")
    return {"classification": classification, "supportedSources": supported}


def interpret_multisector(sectors: list[dict[str, Any]], target_source_id: str) -> dict[str, Any]:
    rows = []
    for sector in sectors:
        classified = classify_sector_model(sector)
        rows.append({**sector, **classified})
    unique = [row for row in rows if row["classification"] == "UNIQUE_SOURCE_SUPPORTED"]
    counts: dict[str, list[int]] = {}
    for row in unique:
        counts.setdefault(row["supportedSources"][0], []).append(int(row["sector"]))
    repeated = [source for source, sector_ids in counts.items() if len(sector_ids) >= 2]
    multis = [row for row in rows if row["classification"] == "MULTIPLE_SOURCES_SUPPORTED"]
    blends: dict[tuple[str, ...], int] = {}
    for row in multis:
        key = tuple(row["supportedSources"]); blends[key] = blends.get(key, 0) + 1
    winner = max(counts, key=lambda source: len(counts[source])) if counts else None
    resolved = bool(winner and len(counts[winner]) >= 3
                    and len(counts[winner]) > len(unique) / 2 and len(repeated) < 2)
    switching = len(repeated) >= 2 or any(value >= 2 for value in blends.values())
    decision = ("TARGET_SUPPORTED" if resolved and winner == target_source_id else
                "CATALOG_SOURCE_SUPPORTED" if resolved else
                "SOURCE_SWITCHING_OR_BLEND" if switching else "UNRESOLVED")
    recommendation = {"TARGET_SUPPORTED": "ARCHIVAL_RECURRENCE_INFORMED_TARGET_MECHANISM_MODELING",
        "CATALOG_SOURCE_SUPPORTED": "INDEPENDENT_COUNTERPART_PHOTOMETRIC_VARIABILITY_VALIDATION",
        "SOURCE_SWITCHING_OR_BLEND": "SOURCE_SWITCHING_TEMPORAL_MODEL",
        "UNRESOLVED": "TARGETED_HIGH_RESOLUTION_TIME_SERIES_PHOTOMETRY"}[decision]
    return {"classification": decision,
        "validSectorCount": sum(row["classification"] not in {"UNAVAILABLE", "SCIENTIFICALLY_INVALID"} for row in rows),
        "uniquelyResolvedSectorCount": len(unique),
        "targetSupportingSectors": counts.get(target_source_id, []),
        "supportByCatalogSource": {source: ids for source, ids in counts.items() if source != target_source_id},
        "multiSourceSupportingSectors": [int(row["sector"]) for row in multis],
        "unresolvedSectors": [int(row["sector"]) for row in rows if row["classification"] == "UNRESOLVED"],
        "unavailableSectors": [int(row["sector"]) for row in rows if row["classification"] == "UNAVAILABLE"],
        "scientificallyInvalidSectors": [int(row["sector"]) for row in rows if row["classification"] == "SCIENTIFICALLY_INVALID"],
        "sourceSupportTable": counts, "sectorResults": rows, "preferredSource": winner if resolved else None,
        "sourceAttributionResolved": resolved, "sourceSwitchingOrBlendDetected": switching,
        "physicalMechanismResolved": False,
        "crossSectorPhaseUsed": False, "historicalResidualDriftExtrapolated": False,
        "recommendedNextTest": recommendation}


def run_multisector_source_localization(preparation: dict[str, Any], *, sector_inputs=None
                                        ) -> dict[str, Any]:
    """Acquire official pixel/PRF data and execute the generalized comparison.

    Tests may provide already acquired inputs, but the public no-argument path is
    the complete production archive path.  No period search is performed.
    """
    import numpy as np
    from pathlib import Path
    from .tess_catalog_guided_localization import analyze_generalized_catalog_guided_sector
    from .tess_target_residual_pixel_recurrence import acquire_selected_sector, tpf_flux_cube, NoPixelCoverageError
    from .tess_residual_localization import _download_tpf, _background_subtract_cube, MAX_CADENCES, _uniform_indices
    from .tess_multisource_residual import _prewhiten_cube_raw
    from .tess_offset_variability import _skycoord
    from .tess_prf_deblend import _background_columns
    from .tess_spoc_prf import (_tpf_detector_geometry, _list_official_prf_grid,
        _official_prf_at_detector_position, _render_prf_template)
    frozen = {int(row["sector"]): row for row in preparation["additionalSectorEvidence"]}
    source_ids = [row["sourceID"] for row in preparation["catalogHypotheses"]]
    if sector_inputs is None:
        sector_inputs = []
        cache_root = Path(preparation["artifactRoot"]) / "official-prf-cache"
        coordinates = [_skycoord(row["raDeg"], row["decDeg"])
                       for row in preparation["catalogHypotheses"]]
        for sector, clock in frozen.items():
            try:
                tpf, source = acquire_selected_sector(_download_tpf,
                    tic_id=preparation["ticID"], sector=sector,
                    ra_deg=preparation["targetSky"]["raDeg"], dec_deg=preparation["targetSky"]["decDeg"])
            except NoPixelCoverageError as error:
                sector_inputs.append({"sector": sector, "availability": "UNAVAILABLE",
                    "acquisitionProvenance": {"condition": str(error)}})
                continue
            times = np.asarray(tpf.time.value, float); cube = tpf_flux_cube(tpf)
            keep = np.isfinite(times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
            times, cube = times[keep], cube[keep]
            indices = _uniform_indices(len(times), MAX_CADENCES); times, cube = times[indices], cube[indices]
            if len(times) < 100: raise RuntimeError(f"Sector {sector} has only {len(times)} usable cadences.")
            corrected, background = _background_subtract_cube(cube)
            residual, valid = _prewhiten_cube_raw(absolute_times=times, cube=corrected,
                physical_frequency=preparation["establishedPhysicalFamilyFrequency"], harmonic_orders=(1, 2))
            rows, cols = valid.shape; valid_flat = valid.reshape(-1)
            centers = []
            for source_id, coordinate in zip(source_ids, coordinates):
                x, y = tpf.wcs.world_to_pixel(coordinate)
                if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                    raise RuntimeError(f"{source_id} has no finite WCS position in sector {sector}")
                centers.append({"componentID": source_id, "x": float(x), "y": float(y)})
            camera, ccd, tpf_col, tpf_row = _tpf_detector_geometry(tpf)
            grid = _list_official_prf_grid(sector=sector, camera=camera, ccd=ccd); prfs = []
            for center in centers:
                image, header, files = _official_prf_at_detector_position(sector=sector, camera=camera,
                    ccd=ccd, detector_row=tpf_row + center["y"], detector_col=tpf_col + center["x"],
                    archive_cache=cache_root / f"sector-{sector:04d}", grid_entries=grid)
                prfs.append({**center, "image": image, "header": header, "modelFiles": files})
            def render(dx, dy, source_models=prfs, mask=valid, selection=valid_flat):
                return np.column_stack([_render_prf_template(image=model["image"], header=model["header"],
                    source_x=model["x"] + dx, source_y=model["y"] + dy, rows=mask.shape[0],
                    cols=mask.shape[1], valid_pixels=mask) for model in source_models])[selection]
            sector_inputs.append({"sector": sector, "times": times, "prewhitened": residual,
                "valid": valid, "calibrationImage": np.nanmedian(corrected, axis=0).reshape(-1)[valid_flat],
                "backgroundColumns": [column[valid_flat] for column in _background_columns(rows, cols, valid)],
                "renderTemplates": render, "availability": "AVAILABLE",
                "acquisitionProvenance": {"tpf": source, "backgroundSubtraction": background,
                    "componentPixelCenters": centers, "officialPRFModels": [x["modelFiles"] for x in prfs],
                    "finiteCadenceCount": len(times), "subtractedHarmonicOrders": [1, 2]}})
    results = []
    for item in sector_inputs:
        sector = int(item["sector"])
        if sector not in frozen or sector in preparation["excludedV2018SectorIDs"]:
            raise RuntimeError("sector input was not preregistered for v20.19")
        if item.get("availability") == "UNAVAILABLE":
            results.append({"sector": sector, "availability": "UNAVAILABLE",
                            "acquisitionProvenance": item.get("acquisitionProvenance")})
            continue
        result = analyze_generalized_catalog_guided_sector(sector=sector, times=item["times"],
            prewhitened=item["prewhitened"], valid=item["valid"], calibration_image=item["calibrationImage"],
            background_columns=item["backgroundColumns"], render_templates=item["renderTemplates"],
            candidate_frequency=frozen[sector]["candidateFrequency"],
            original_time_origin=frozen[sector]["originalTimeOriginDays"],
            physical_frequency=preparation["establishedPhysicalFamilyFrequency"], component_ids=source_ids,
            block_count=int(item.get("blockCount", 4)))
        result["acquisitionProvenance"] = copy.deepcopy(item.get("acquisitionProvenance")); results.append(result)
    return {"version": "openstar.tess-target-residual-multisector-source-run.v1",
            "sectorResults": results, "crossSectorPhaseUsed": False,
            "historicalResidualDriftExtrapolated": False}
