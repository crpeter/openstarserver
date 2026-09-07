"""Diagnostic comparison of frozen eclipse images; never changes attribution.

This is a separate audit, not an alternative source-acceptance rule. Pixel inputs
must reproduce the original localization hash. Catalog flux budgets assume equal
aperture throughput and are not exclusions or measurements of companion nature.
"""
from __future__ import annotations

import json
import math
import os
import inspect
import platform
import tempfile
from pathlib import Path

import numpy as np

from openstar_investigation import InvestigationStore, sha256_file, sha256_json
from . import tess_eclipse_event_localization as original


VERSION = "openstar.tess-eclipse-localization-audit.v1"
APERTURE_RADIUS_PIXELS = 3.0


def verified_boundary(state_dir, investigation_id):
    root = Path(state_dir).resolve() / "investigations"
    if not root.is_dir():
        raise ValueError("Investigation directory does not exist.")
    store = InvestigationStore(root)
    investigation = store.load(investigation_id)
    if investigation.status != "COMPLETE" or not investigation.stages:
        raise ValueError("Audit requires a completed investigation.")
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
            if sha256_file(artifact.path) != artifact.sha256:
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
            or final.triggered_by_stage_id != localization.id
            or localization.triggered_by_stage_id != catalog.id
            or result.get("resultVersion") != original.RESULT_VERSION
            or result.get("classification") != "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND"
            or result.get("sourceAttributionResolved") is not False
            or result.get("recommendedNextTest") != "ADDITIONAL_SPATIAL_EVIDENCE"
            or not order[identity.id] < order[binary.id] < order[catalog.id] < order[localization.id] < order[final.id]
            or not original.authoritative_binary_gate(binary.result or {})):
        raise ValueError("Not the finalized conflicting eclipse-localization boundary.")
    if (result.get("binaryConfirmationSHA256") != sha256_json(binary.result)
            or result.get("identitySHA256") != sha256_json(identity.result)
            or sha256_json(result.get("frozenCatalog")) != (catalog.result or {}).get("frozenCatalogSHA256")
            or result.get("frozenCatalog") != (catalog.result or {}).get("frozenCatalog")):
        raise ValueError("Frozen localization ancestry mismatch.")
    frozen = result["frozenCatalog"]
    targets = [s for s in frozen["catalogHypotheses"] if s.get("isTarget") is True]
    if len(targets) != 1 or not targets[0].get("ticID"):
        raise ValueError("Frozen catalog must identify exactly one target TIC.")
    sectors = result.get("sectorResults") or []
    ids = [s["sector"] for s in sectors]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate localization sectors.")
    for sector in sectors:
        digest = sector.get("pixelInputSHA256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Localization sector lacks an exact pixel input hash.")
        original._frozen_sector(binary.result, sector["sector"])
    return {
        "investigationID": investigation.id, "ticID": targets[0]["ticID"],
        "localization": result, "binary": binary.result, "identity": identity.result,
        "stageLedgerSHA256": hashes,
        "investigationSHA256": sha256_file(store.path_for(investigation.id)),
    }


def _moment(image, mask):
    weights = np.where(mask, np.maximum(image, 0), 0)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return None
    yy, xx = np.indices(image.shape)
    return {"x": float((weights * xx).sum() / total), "y": float((weights * yy).sum() / total)}


def _offset(first, second):
    if first is None or second is None:
        return None
    dx, dy = first["x"] - second["x"], first["y"] - second["y"]
    return {"x": dx, "y": dy, "distancePixels": math.hypot(dx, dy)}


def flux_budget(catalog, fractional_loss):
    hypotheses = catalog["catalogHypotheses"]
    rows = ((catalog.get("catalogQueries") or {}).get("tic") or {}).get("sources") or []

    def magnitude(tic):
        matches = [r for r in rows if r.get("ticID") == tic]
        return original._finite(matches[0].get("tmag")) if len(matches) == 1 else None

    target = next(s for s in hypotheses if s.get("isTarget") is True)
    target_mag = magnitude(target.get("ticID"))
    results = []
    for source in hypotheses:
        mag = magnitude(source.get("ticID"))
        ratio = None
        if mag is not None and target_mag is not None:
            exponent = -0.4 * (mag - target_mag)
            if -300 < exponent < 300:
                ratio = 10 ** exponent
        # Equal throughput, target + candidate only, total candidate disappearance.
        cap = ratio / (1 + ratio) if ratio is not None and not source.get("isTarget") else None
        results.append({
            "sourceID": source["sourceID"], "tmag": mag, "fluxRatioToTarget": ratio,
            "equalThroughputMaximumPairDepth": cap,
            "measuredApertureLossOverConditionalMaximum": (
                fractional_loss / cap if cap and fractional_loss is not None else None),
            "candidateExcluded": False,
        })
    return {"assumption": "Equal target/candidate aperture throughput; reliable catalog magnitudes; no other light.",
            "limitation": "Aperture throughput, crowding and catalog errors are not calibrated. No candidate is excluded.",
            "sources": results}


def measure_audit(item, frozen, ephemeris, previous, catalog):
    if item.get("pixelInputSHA256") != previous["pixelInputSHA256"]:
        raise ValueError(f"Sector {frozen['sector']}: reacquired pixel input differs from frozen localization.")
    mask = previous.get("frozenMask") or {}
    if (mask.get("periodDays") != ephemeris["refinedPeriodDays"]
            or mask.get("sectorEventEpoch") != frozen["eventEpoch"]
            or mask.get("durationDays") != frozen["durationDays"]
            or previous["sector"] != frozen["sector"] or previous["role"] != frozen["role"]):
        raise ValueError("Frozen event definition mismatch.")
    # Repeat the exact quality selection, cadence cap and background processing.
    times = np.asarray(item["times"], dtype=float)
    cube = np.asarray(item["fluxCube"], dtype=float)
    if cube.ndim != 3 or len(times) != len(cube):
        raise ValueError("Pixel cube does not match cadence times.")
    keep = np.isfinite(times) & np.any(np.isfinite(cube.reshape(len(cube), -1)), axis=1)
    if item.get("qualityMask") is not None:
        keep &= np.asarray(item["qualityMask"], dtype=bool)
    indices = np.flatnonzero(keep)[original._uniform_indices(int(keep.sum()), original.MAX_CADENCES)]
    times, cube = times[indices], cube[indices]
    if len(times) < original.MIN_VALID_CADENCES:
        raise ValueError("Insufficient audit cadences.")
    corrected, background = original._background_subtract_cube(cube)
    corrected, valid = original._filled_cube(corrected)
    inside, control, cycles = original._event_bins(
        times, float(ephemeris["refinedPeriodDays"]), float(frozen["eventEpoch"]), float(frozen["durationDays"]))
    if min(inside.sum(), control.sum()) < original.MIN_BIN_CADENCES or len(set(cycles[inside])) < original.MIN_EVENTS:
        raise ValueError("Insufficient frozen event/control coverage.")
    out_image = np.mean(corrected[control], axis=0)
    in_image = np.mean(corrected[inside], axis=0)
    difference = out_image - in_image
    # This identity ensures the audit has not silently changed the original masks.
    saved_image = np.asarray(previous["differenceImage"]["differenceImage"], dtype=np.float32)
    reproduced = np.asarray(np.where(valid, difference, 0), dtype=np.float32)
    if saved_image.shape != reproduced.shape or not np.array_equal(saved_image, reproduced):
        raise ValueError("Frozen difference image did not reproduce exactly.")
    target = item["targetPixel"]
    yy, xx = np.indices(difference.shape)
    aperture = valid & (np.hypot(xx - target["x"], yy - target["y"]) <= APERTURE_RADIUS_PIXELS)
    if aperture.sum() < 6:
        raise ValueError("Insufficient valid fixed-aperture pixels.")
    out_centroid, diff_centroid = _moment(out_image, aperture), _moment(difference, aperture)
    out_flux = float(out_image[aperture].sum())
    loss = float(difference[aperture].sum())
    fraction = loss / out_flux if out_flux > 0 else None
    # Empirical image-shape comparison, not a PRF fit or an astrometric correction.
    template = np.where(valid, out_image, 0)
    gy, gx = np.gradient(template)
    design = np.column_stack([template[aperture], gx[aperture], gy[aperture], np.ones(aperture.sum())])
    scales = np.linalg.norm(design, axis=0)
    scaled = design / np.where(scales > 0, scales, 1)
    beta, _, rank, _ = np.linalg.lstsq(scaled, difference[aperture], rcond=None)
    residual = difference[aperture] - scaled @ beta
    total = float(np.sum((difference[aperture] - difference[aperture].mean()) ** 2))
    shape = {"model": "out-of-event image + x/y image gradients + constant",
             "rank": int(rank), "explainedVariance": 1 - float(residual @ residual) / total if total > 0 and rank == 4 else None,
             "astrometricCorrectionApplied": False}
    edge = np.zeros(valid.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    positive_total = float(np.maximum(difference[valid], 0).sum())
    return {
        "sector": frozen["sector"], "role": frozen["role"],
        "originalClassification": previous["classification"],
        "originalMatchedSource": previous.get("matchedCatalogHypothesis"),
        "pixelInputSHA256": previous["pixelInputSHA256"], "frozenMask": previous["frozenMask"],
        "outOfEventCentroid": out_centroid, "positiveDifferenceCentroid": diff_centroid,
        "outOfEventMinusCatalogTargetPixels": _offset(out_centroid, target),
        "differenceMinusOutOfEventPixels": _offset(diff_centroid, out_centroid),
        "legacyMinusFixedApertureCentroidPixels": _offset(previous["measuredPixelCentroid"], diff_centroid),
        "fixedApertureRadiusPixels": APERTURE_RADIUS_PIXELS,
        "fixedApertureSignedFractionalLoss": fraction,
        "positiveDifferenceEdgeFraction": float(np.maximum(difference[valid & edge], 0).sum()) / positive_total if positive_total else None,
        "empiricalImageShapeFit": shape, "fluxBudget": flux_budget(catalog, fraction),
        "backgroundCorrection": background,
        "images": {"outOfEvent": np.where(valid, out_image, 0).tolist(),
                   "inEvent": np.where(valid, in_image, 0).tolist(), "difference": reproduced.tolist(),
                   "aperture": aperture.tolist(), "validPixels": valid.tolist()},
    }


def run_audit(state_dir, investigation_id, output_file, *, execute=False):
    output = Path(output_file).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Audit output already exists: {output}")
    if output.resolve().is_relative_to(Path(state_dir).resolve()):
        raise ValueError("Audit output must be outside the investigation state directory.")
    boundary = verified_boundary(state_dir, investigation_id)
    if not execute:
        return {"status": "VALIDATED_NO_CHANGES", "sectors": [s["sector"] for s in boundary["localization"]["sectorResults"]]}
    result = {"version": VERSION, "status": "DIAGNOSTIC_ONLY", "investigationID": investigation_id,
              "parentHashes": {"investigationSHA256": boundary["investigationSHA256"],
                               "stageLedgers": boundary["stageLedgerSHA256"]},
              "sourceAttributionResolved": False, "claimLevelChanged": False,
              "companionNatureResolved": False, "catalogQueriesRepeated": False,
              "periodOrDurationSearched": False, "originalClassification": boundary["localization"]["classification"],
              "limitations": ["Positive-flux moments can be biased by crowding and clipping.",
                              "Out-of-event centroid is a scene reference, not a calibrated target position.",
                              "No PRF, saturation correction, source exclusion or claim promotion is performed."],
              "sectorResults": []}
    catalog = boundary["localization"]["frozenCatalog"]
    for previous in boundary["localization"]["sectorResults"]:
        frozen = original._frozen_sector(boundary["binary"], previous["sector"])
        print(f"Auditing frozen eclipse pixels: Sector {previous['sector']}", flush=True)
        item = original._production_input(boundary["ticID"], boundary["identity"], frozen, catalog)
        result["sectorResults"].append(measure_audit(item, frozen, boundary["binary"]["linearEphemeris"], previous, catalog))
    if verified_boundary(state_dir, investigation_id) != boundary:
        raise ValueError("Investigation changed during audit; output refused.")
    sources = {Path(__file__), Path(original.__file__)}
    sources.update(Path(inspect.getfile(function)) for function in (
        original._background_subtract_cube, original._filled_cube, original._uniform_indices,
        original._centroid_from_frames, original._download_tpf))
    result["methodSourceSHA256"] = {path.name: sha256_file(path) for path in sorted(sources)}
    result["runtime"] = {"python": platform.python_version(), "numpy": np.__version__}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".eclipse-audit-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.link(temporary, output)  # Atomic publication; refuses an existing output.
    finally:
        Path(temporary).unlink()
    return result
