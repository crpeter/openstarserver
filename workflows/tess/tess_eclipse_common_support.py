"""Opt-in eclipse localization on common, SNR-independent spatial support.

This produces a separate versioned reanalysis of a verified v1 conflict. It never
updates an investigation, corrects astrometry from a blended scene, or changes
the frozen event definition or existing cross-sector acceptance policy.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

from openstar_investigation import InvestigationStore, sha256_file, sha256_json
from . import tess_eclipse_event_localization as original


RESULT_VERSION = "openstar.tess-eclipse-event-source-localization.v2"
METHOD_ID = "openstar.tess-eclipse-common-support.v1"
CENTROID_METHOD = "common-support-v2"


def _positive_centroid(image, support):
    weights = np.where(support, np.maximum(image, 0.0), 0.0)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise original.EclipseLocalizationDataUnavailable("No positive common-support image flux")
    yy, xx = np.indices(image.shape)
    return {"x": float((weights * xx).sum() / total),
            "y": float((weights * yy).sum() / total)}


def common_support_image(cube, valid_pixels, high, low):
    """Measure both images over every valid pixel, without peak-based selection.

The validity mask is determined from the full frozen cadence cube before event
splitting. The identical mask is reused in every leave-one-event-out measurement.
SNR remains a detection diagnostic and never selects or weights centroid pixels.
"""
    cube = np.asarray(cube, dtype=np.float64)
    support = np.asarray(valid_pixels, dtype=bool)
    if cube.ndim != 3 or support.shape != cube.shape[1:]:
        raise ValueError("Common-support mask must match the pixel cube")
    if len(high) < 2 or len(low) < 2:
        raise original.EclipseLocalizationDataUnavailable("Insufficient common-support phase bins")
    if not np.any(support):
        raise original.EclipseLocalizationDataUnavailable("No valid common-support pixels")
    out_frames, in_frames = cube[high], cube[low]
    if not (np.isfinite(out_frames[:, support]).all()
            and np.isfinite(in_frames[:, support]).all()):
        raise ValueError("Common-support pixels must be finite after frozen filling")
    out_image = np.mean(out_frames, axis=0)
    in_image = np.mean(in_frames, axis=0)
    difference = out_image - in_image
    variance = (np.var(out_frames, axis=0, ddof=1) / len(high)
                + np.var(in_frames, axis=0, ddof=1) / len(low))
    noise = np.sqrt(np.maximum(variance, 0.0))
    snr = np.divide(difference, noise, out=np.zeros_like(difference),
                    where=np.isfinite(noise) & (noise > 1e-12))
    snr = np.where(support & np.isfinite(snr), snr, 0.0)
    py, px = np.unravel_index(np.argmax(np.where(support, snr, -np.inf)), support.shape)
    if snr[py, px] <= 0 or float(difference[support].sum()) <= 0:
        raise original.EclipseLocalizationDataUnavailable("No positive common-support event loss")
    out_flux = float(out_image[support].sum())
    if out_flux <= 0:
        raise original.EclipseLocalizationDataUnavailable("No positive out-of-event reference flux")
    out_centroid = _positive_centroid(out_image, support)
    diff_centroid = _positive_centroid(difference, support)
    dx = diff_centroid["x"] - out_centroid["x"]
    dy = diff_centroid["y"] - out_centroid["y"]
    return {
        "methodID": METHOD_ID,
        "centroidX": diff_centroid["x"], "centroidY": diff_centroid["y"],
        "peakX": int(px), "peakY": int(py), "peakSNR": float(snr[py, px]),
        "peakUsedForSpatialSelection": False, "snrUsedForCentroidWeights": False,
        "supportPixelCount": int(support.sum()), "commonSupportMask": support.tolist(),
        "differenceImage": np.asarray(np.where(support, difference, 0), dtype=np.float32).tolist(),
        "snrImage": np.asarray(snr, dtype=np.float32).tolist(),
        "outOfEventImage": np.where(support, out_image, 0).tolist(),
        "outOfEventCentroid": out_centroid,
        "differenceMinusOutOfEventPixels": {"x": dx, "y": dy, "distancePixels": math.hypot(dx, dy)},
        "signedFractionalLoss": float(difference[support].sum()) / out_flux,
        "astrometricCorrectionApplied": False,
    }


def _is_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def verified_boundary(state_dir, investigation_id):
    """Authenticate the original finalized conflict before any pixel acquisition."""
    root = Path(state_dir).expanduser().resolve() / "investigations"
    if not root.is_dir():
        raise ValueError("Investigation directory does not exist")
    store = InvestigationStore(root)
    investigation = store.load(investigation_id)
    if investigation.status != "COMPLETE" or not investigation.stages:
        raise ValueError("Reanalysis requires a completed investigation")
    hashes = {}
    for stage in investigation.stages:
        digest = store.verified_terminal_stage_ledger_hash(investigation.id, stage)
        if not digest:
            raise ValueError(f"Unverified terminal ledger: {stage.id}")
        hashes[stage.id] = digest
        if stage.result is not None and (
            stage.provenance is None or stage.provenance.result_hash != sha256_json(stage.result)
        ):
            raise ValueError(f"Stage result hash mismatch: {stage.id}")
        for artifact in stage.artifacts:
            if not _is_hash(artifact.sha256) or sha256_file(artifact.path) != artifact.sha256:
                raise ValueError(f"Stage artifact hash mismatch: {stage.id}")

    def latest(handler):
        matches = [s for s in investigation.stages if s.handler_id == handler and s.status == "COMPLETE"]
        if not matches:
            raise ValueError(f"Missing completed handler: {handler}")
        return matches[-1]

    localization = latest(original.HANDLER_ID)
    binary = latest("openstar.tess.binary-confirmation.analyze")
    identity = latest("openstar.tess.catalog-identity")
    catalog = latest(original.PREPARE_HANDLER_ID)
    final = investigation.stages[-1]
    result = localization.result or {}
    order = {stage.id: index for index, stage in enumerate(investigation.stages)}
    if (final.handler_id != "openstar.tess.finalize" or not final.stop
            or final.next_stage is not None
            or final.triggered_by_stage_id != localization.id
            or localization.triggered_by_stage_id != catalog.id
            or result.get("resultVersion") != original.RESULT_VERSION
            or result.get("classification") != "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND"
            or result.get("sourceAttributionResolved") is not False
            or result.get("recommendedNextTest") != "ADDITIONAL_SPATIAL_EVIDENCE"
            or not order[identity.id] < order[binary.id] < order[catalog.id] < order[localization.id] < order[final.id]
            or not original.authoritative_binary_gate(binary.result or {})):
        raise ValueError("Not the finalized v1 conflicting eclipse-localization boundary")
    if (result.get("binaryConfirmationSHA256") != sha256_json(binary.result)
            or result.get("identitySHA256") != sha256_json(identity.result)
            or sha256_json(result.get("frozenCatalog")) != (catalog.result or {}).get("frozenCatalogSHA256")
            or result.get("frozenCatalog") != (catalog.result or {}).get("frozenCatalog")):
        raise ValueError("Frozen localization ancestry mismatch")
    frozen_catalog = result["frozenCatalog"]
    targets = [s for s in frozen_catalog["catalogHypotheses"] if s.get("isTarget") is True]
    if len(targets) != 1 or not targets[0].get("ticID"):
        raise ValueError("Frozen catalog must identify exactly one target TIC")
    sectors = result.get("sectorResults") or []
    rejections = result.get("sectorRejections") or []
    ids = [s["sector"] for s in sectors + rejections]
    expected = [s["sector"] for s in binary.result["sectorResults"] if s.get("usable") is True]
    if (not sectors or len(ids) != len(set(ids)) or len(expected) != len(set(expected))
            or set(ids) != set(expected)):
        raise ValueError("Missing or duplicate frozen localization sectors")
    for sector in sectors:
        if not _is_hash(sector.get("pixelInputSHA256")):
            raise ValueError("Localization sector lacks an exact pixel input hash")
        frozen = original._frozen_sector(binary.result, sector["sector"])
        _verify_mask(sector, frozen, binary.result["linearEphemeris"])
    reproduced = original.summarize_eclipse_results(
        results=sectors, rejections=rejections, binary_confirmation=binary.result,
        identity=identity.result, frozen_catalog=frozen_catalog)
    # The production stage adds these fields after the numerical function.
    reproduced["frozenCatalogSHA256"] = sha256_json(frozen_catalog)
    reproduced["pixelEvidenceSHA256BySector"] = {
        str(s["sector"]): s["pixelInputSHA256"] for s in sectors}
    if reproduced != result:
        raise ValueError("Original localization summary does not match its frozen sector evidence")
    return {
        "investigationID": investigation.id, "ticID": targets[0]["ticID"],
        "localization": result, "binary": binary.result, "identity": identity.result,
        "stageLedgerSHA256": hashes,
        "investigationSHA256": sha256_file(store.path_for(investigation.id)),
    }


def _verify_mask(previous, frozen, ephemeris):
    mask = previous.get("frozenMask") or {}
    assignments = [a for a in ephemeris.get("cycleAssignments", []) if a.get("sector") == frozen["sector"]]
    if (len(assignments) != 1 or mask.get("cycleAssignment") != assignments[0]
            or mask.get("periodDays") != ephemeris["refinedPeriodDays"]
            or mask.get("referenceEpoch") != ephemeris["referenceEpoch"]
            or mask.get("sectorEventEpoch") != frozen["eventEpoch"]
            or mask.get("durationDays") != frozen["durationDays"]
            or mask.get("phaseOrDurationSearched") is not False
            or mask.get("oppositeConjunctionExcluded") is not True
            or previous["role"] != frozen["role"]):
        raise ValueError("Frozen event definition mismatch")


def _verify_pixels(item, previous, frozen, ephemeris):
    if item.get("pixelInputSHA256") != previous["pixelInputSHA256"]:
        raise ValueError(f"Sector {frozen['sector']}: pixel input differs from frozen localization")
    reproduced = original.measure_eclipse_sector(item, frozen, ephemeris)
    # Reproduce the diagnostic images AND the original selection, not just a
    # caller-provided hash. No old result is relabeled with the new method ID.
    if (reproduced["differenceImage"] != previous["differenceImage"]
            or reproduced["frozenMask"] != previous["frozenMask"]):
        raise ValueError(f"Sector {frozen['sector']}: original localization did not reproduce exactly")


def _source_hashes():
    from . import tess_difference_image_constants
    from .tess_target_residual_pixel_recurrence import acquire_selected_sector
    sources = {Path(__file__), Path(original.__file__)}
    sources.add(Path(tess_difference_image_constants.__file__))
    sources.update(Path(inspect.getfile(function)) for function in (
        original._background_subtract_cube, original._filled_cube, original._uniform_indices,
        original._centroid_from_frames, original._download_tpf,
        InvestigationStore.verified_terminal_stage_ledger_hash, sha256_json, acquire_selected_sector))
    return {str(path.resolve()): sha256_file(path) for path in sorted(sources)}


def _method_contract():
    return {
        "methodID": METHOD_ID, "resultVersion": RESULT_VERSION,
        "support": "All pixels valid under the frozen full-cube filling rule; identical for both images and jackknives",
        "weights": "Positive image flux only; no SNR weighting or peak-centered pixel selection",
        "sourcePosition": "Difference-image centroid in original detector coordinates",
        "referencePosition": "Out-of-event scene centroid, diagnostic only; never subtracted for catalog matching",
        "uncertainty": "Leave-one-event-out centroid jackknife, including corresponding controls; existing 0.12 pixel floor",
        "minImagePeakSNR": original.MIN_IMAGE_PEAK_SNR,
        "maxCadences": original.MAX_CADENCES, "minValidCadences": original.MIN_VALID_CADENCES,
        "minBinCadences": original.MIN_BIN_CADENCES, "minEvents": original.MIN_EVENTS,
        "sourceMatchMaxPixels": original.SOURCE_MATCH_MAX_PIXELS,
        "sourceMarginFloorPixels": original.SOURCE_MARGIN_FLOOR_PIXELS,
        "maxJackknifeUncertaintyPixels": original.MAX_JACKKNIFE_UNCERTAINTY_PIXELS,
        "catalogOverlapPixels": original.CATALOG_OVERLAP_PIXELS,
        "offTargetMinPixels": original.OFF_TARGET_MIN_PIXELS,
        "maxOffCatalogScatterArcsec": original.MAX_OFF_CATALOG_SCATTER_ARCSEC,
        "minIndependentSectors": original.MIN_INDEPENDENT_SECTORS,
        "primaryCanSatisfyReplication": False,
        "conflictingUsableSectorsCanBeOutvoted": False,
    }


def run_reanalysis(state_dir, investigation_id, output_file, *, execute=False):
    output = Path(output_file).expanduser().absolute()
    state = Path(state_dir).expanduser().resolve()

    def check_output():
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Reanalysis output already exists: {output}")
        if output.resolve().is_relative_to(state):
            raise ValueError("Reanalysis output must be outside the investigation state directory")

    check_output()
    boundary = verified_boundary(state, investigation_id)
    if not execute:
        return {"status": "VALIDATED_NO_CHANGES", "resultVersion": RESULT_VERSION,
                "sectors": [s["sector"] for s in boundary["localization"]["sectorResults"]]}
    sources = _source_hashes()  # Fail before acquisition if source files vanished.
    contract = _method_contract()
    old = boundary["localization"]
    catalog = old["frozenCatalog"]
    ephemeris = boundary["binary"]["linearEphemeris"]
    measured, rejections = [], copy.deepcopy(old.get("sectorRejections") or [])
    for previous in old["sectorResults"]:
        sector = previous["sector"]
        frozen = original._frozen_sector(boundary["binary"], sector)
        print(f"Reanalyzing frozen eclipse pixels: Sector {sector}", flush=True)
        item = original._production_input(boundary["ticID"], boundary["identity"], frozen, catalog)
        _verify_pixels(item, previous, frozen, ephemeris)
        try:
            result = original.measure_eclipse_sector(item, frozen, ephemeris, centroid_method=CENTROID_METHOD)
        except original.EclipseLocalizationDataUnavailable as error:
            # Retain the sector explicitly; losing quality is never silent.
            measured.append({"sector": sector, "role": frozen["role"], "usable": False,
                             "classification": "NO_QUALITY_LOCALIZATION", "matchedCatalogHypothesis": None,
                             "qualityRejectionReasons": [str(error)],
                             "pixelInputSHA256": previous["pixelInputSHA256"],
                             "frozenMask": previous["frozenMask"],
                             "originalClassification": previous["classification"]})
            continue
        wcs, target = item.get("_wcs"), item.get("_targetCoordinate")
        if wcs is not None and target is not None:
            centroid = result["measuredPixelCentroid"]
            sky = wcs.pixel_to_world(centroid["x"], centroid["y"])
            east, north, _ = original._world_offsets_arcsec(target, sky)
            result.update({"centroidSky": {"raDeg": float(sky.ra.deg), "decDeg": float(sky.dec.deg)},
                           "skyOffsetEastArcsec": east, "skyOffsetNorthArcsec": north})
        else:
            # Never reuse a sky position calculated for the previous centroid.
            result.update({"centroidSky": None, "skyOffsetEastArcsec": None, "skyOffsetNorthArcsec": None})
        result.update({"originalClassification": previous["classification"],
                       "originalMatchedCatalogHypothesis": previous.get("matchedCatalogHypothesis"),
                       "originalPixelCentroid": previous["measuredPixelCentroid"],
                       "originalCentroidUncertaintyPixels": previous["centroidUncertaintyPixels"]})
        measured.append(result)
    result = original.summarize_eclipse_results(
        results=measured, rejections=rejections, binary_confirmation=boundary["binary"],
        identity=boundary["identity"], frozen_catalog=catalog, result_version=RESULT_VERSION)
    result.update({
        "status": "REANALYSIS_REQUIRES_REVIEW", "investigationID": investigation_id,
        "methodContract": contract, "methodContractSHA256": sha256_json(contract),
        "parentHashes": {"investigationSHA256": boundary["investigationSHA256"],
                         "stageLedgers": boundary["stageLedgerSHA256"], "originalResultSHA256": sha256_json(old)},
        "originalResultVersion": old["resultVersion"], "originalClassification": old["classification"],
        "investigationModified": False, "claimLevelChanged": False, "discoveryClaim": False,
        "catalogQueriesRepeated": False, "methodSourceSHA256": sources,
        "frozenCatalogSHA256": sha256_json(catalog),
        "pixelEvidenceSHA256BySector": {str(s["sector"]): s["pixelInputSHA256"] for s in measured},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "limitations": ["Positive-flux moments are not calibrated PRF astrometry.",
                        "Crowding, saturation, finite stamps and background errors can bias both centroids.",
                        "The scene reference is diagnostic and cannot establish target attribution by itself.",
                        "No catalog magnitude-based source exclusion or investigation claim promotion is performed."],
    })
    if verified_boundary(state, investigation_id) != boundary:
        raise ValueError("Investigation changed during reanalysis; output refused")
    if _source_hashes() != sources:
        raise ValueError("Method source changed during reanalysis; output refused")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    check_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".eclipse-common-support-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        Path(temporary).unlink()
    return result
