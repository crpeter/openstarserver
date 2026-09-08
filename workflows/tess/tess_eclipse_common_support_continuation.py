"""Offline, explicit adoption of a reviewed common-support localization result.

The reviewed file digest is required. Saved images and diagnostics can be checked
offline, but they do not contain the cadence cube or WCS needed to independently
rerun the jackknives or catalog projection. No such rerun is claimed here.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from openstar_investigation import (
    ArtifactReference, InvestigationStore, canonical_json_bytes, sha256_file,
    sha256_json, utc_now_iso,
)
from . import tess_eclipse_common_support as common
from . import tess_eclipse_event_localization as original


VERSION = "openstar.tess-eclipse-common-support-continuation.v1"
IMPORT_HANDLER = "openstar.tess.eclipse-common-support.accept"
FINAL_HANDLER = "openstar.tess.eclipse-common-support.finalize"
PACKAGE = "eclipse-common-support-continuation-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _equal(actual, expected, label):
    _require(canonical_json_bytes(actual) == canonical_json_bytes(expected),
             f"{label} mismatch")


def _near(actual, expected, label, *, tolerance=2e-6):
    _require(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=tolerance),
             f"{label} mismatch")


def _strict_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    result = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    # Also reject overflow such as 1e999, which parse_constant does not receive.
    json.dumps(result, allow_nan=False)
    _require(isinstance(result, dict), "Reanalysis must be a JSON object")
    return result


def _source_inventory(recorded):
    """Relocate source paths by repository-relative name, never by basename."""
    expected = {str(Path(path).relative_to(REPOSITORY_ROOT)): digest
                for path, digest in common._source_hashes().items()}
    _require(isinstance(recorded, dict) and len(recorded) == len(expected),
             "Method source inventory is incomplete")
    found = {}
    for path, digest in recorded.items():
        matches = [rel for rel in expected if path.endswith("/" + rel)]
        _require(len(matches) == 1 and matches[0] not in found,
                 "Method source path is ambiguous or duplicated")
        found[matches[0]] = digest
    _equal(found, expected, "Method source hashes")
    return expected


def _verify_sector(sector, previous, catalog):
    """Check saved evidence without trusting its summary or changing its gates."""
    for key in ("sector", "role", "pixelInputSHA256", "frozenMask",
                "inEventCadenceCount", "controlCadenceCount", "eclipseEventCount",
                "backgroundCorrection", "acquisitionProvenance", "catalogQueryProvenance"):
        _equal(sector.get(key), previous.get(key), f"Sector {previous['sector']} {key}")
    for key, old_key in (("originalClassification", "classification"),
                         ("originalMatchedCatalogHypothesis", "matchedCatalogHypothesis"),
                         ("originalPixelCentroid", "measuredPixelCentroid"),
                         ("originalCentroidUncertaintyPixels", "centroidUncertaintyPixels")):
        _equal(sector.get(key), previous.get(old_key), key)
    _require(sector.get("usable") is True and sector.get("qualityRejectionReasons") == [],
             "Continuation requires every originally measured sector to remain usable")
    image = sector["differenceImage"]
    _equal(image["methodID"], common.METHOD_ID, "Image method")
    for key in ("peakUsedForSpatialSelection", "snrUsedForCentroidWeights", "astrometricCorrectionApplied"):
        _require(image.get(key) is False, f"Unsupported image policy: {key}")
    # The original stored float32 difference and SNR maps are unchanged by v2.
    for key in ("differenceImage", "snrImage"):
        _equal(image[key], previous["differenceImage"][key], f"Frozen {key}")
    support = np.asarray(image["commonSupportMask"])
    difference = np.asarray(image["differenceImage"], dtype=float)
    out = np.asarray(image["outOfEventImage"], dtype=float)
    snr = np.asarray(image["snrImage"], dtype=float)
    _require(support.dtype == np.dtype(bool) and support.ndim == 2
             and support.shape == difference.shape == out.shape == snr.shape
             and bool(support.any()) and np.isfinite(difference).all()
             and np.isfinite(out).all() and np.isfinite(snr).all(), "Invalid image support or data")
    _equal(image["supportPixelCount"], int(support.sum()), "Support pixel count")
    _require(not np.any(difference[~support]) and not np.any(out[~support])
             and not np.any(snr[~support]), "Image contains flux outside support")
    measured = common._positive_centroid(difference, support)
    reference = common._positive_centroid(out, support)
    for axis, name in (("x", "centroidX"), ("y", "centroidY")):
        _near(image[name], measured[axis], "Saved image centroid")
        _equal(sector["measuredPixelCentroid"][axis], image[name], "Measured centroid")
        _near(image["outOfEventCentroid"][axis], reference[axis], "Scene centroid")
        _near(image["differenceMinusOutOfEventPixels"][axis],
              measured[axis] - reference[axis], "Paired image offset")
    _near(image["differenceMinusOutOfEventPixels"]["distancePixels"],
          math.hypot(measured["x"] - reference["x"], measured["y"] - reference["y"]),
          "Paired image separation")
    _require(difference[support].sum() > 0 and out[support].sum() > 0, "Nonpositive event loss")
    _near(image["signedFractionalLoss"], float(difference[support].sum() / out[support].sum()),
          "Signed fractional loss", tolerance=1e-8)
    _near(image["peakSNR"], float(snr[support].max()), "Image peak SNR")
    _equal(sector["differenceImagePeakSNR"], image["peakSNR"], "Sector peak SNR")
    _require(image["peakSNR"] >= original.MIN_IMAGE_PEAK_SNR, "Weak difference image")
    # This verifies the recorded jackknife aggregate, not a rerun on raw cadences.
    jackknife = sector["eventJackknifeCentroids"]
    cycles = [item["omittedCycle"] for item in jackknife]
    _require(all(type(cycle) is int for cycle in cycles)
             and len(set(cycles)) == len(cycles) == sector["eclipseEventCount"]
             and len(cycles) >= original.MIN_EVENTS, "Missing or duplicate event jackknives")
    for key in ("inEventCadenceCount", "controlCadenceCount"):
        _require(type(sector[key]) is int and sector[key] >= original.MIN_BIN_CADENCES,
                 "Insufficient event/control cadences")
    xs = [item["centroidX"] for item in jackknife]
    ys = [item["centroidY"] for item in jackknife]
    _require(all(type(v) in (int, float) and math.isfinite(v) for v in xs + ys),
             "Invalid event jackknife coordinates")
    cx, cy, n = statistics.mean(xs), statistics.mean(ys), len(xs)
    uncertainty = max(0.12, math.sqrt((n - 1) / n * sum(
        (x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys))))
    _near(sector["centroidUncertaintyPixels"], uncertainty, "Jackknife uncertainty", tolerance=1e-10)
    _require(uncertainty <= original.MAX_JACKKNIFE_UNCERTAINTY_PIXELS, "Unstable event jackknife")
    distances = sector["catalogDistances"]
    identities = [{"id": item["id"], "isTarget": item["isTarget"]} for item in distances]
    expected = [{"id": item["sourceID"], "isTarget": item["isTarget"]}
                for item in catalog["catalogHypotheses"]]
    _equal(identities, expected, "Catalog distance identities")
    _require(all(type(item["distancePixels"]) in (int, float)
                 and math.isfinite(item["distancePixels"]) and item["distancePixels"] >= 0
                 for item in distances), "Invalid catalog distances")
    ordered = sorted(distances, key=lambda item: item["distancePixels"])
    margin = max(original.SOURCE_MARGIN_FLOOR_PIXELS, 2 * uncertainty)
    _near(sector["requiredCatalogMarginPixels"], margin, "Catalog margin", tolerance=1e-10)
    _require(ordered and ordered[0]["isTarget"] is True
             and ordered[0]["distancePixels"] <= original.SOURCE_MATCH_MAX_PIXELS
             and (len(ordered) == 1 or ordered[1]["distancePixels"] - ordered[0]["distancePixels"] >= margin),
             "Sector does not uniquely favor the target under the existing policy")
    _equal(sector["classification"], "TARGET_CONSISTENT", "Sector classification")
    _equal(sector["matchedCatalogHypothesis"], ordered[0]["id"], "Matched source")


def validate_reanalysis(state_dir, investigation_id, reanalysis_file, reviewed_sha256):
    """Read and verify only; the digest must identify the artifact actually reviewed."""
    _require(common._is_hash(reviewed_sha256), "Reviewed SHA-256 must be lowercase hexadecimal")
    payload = Path(reanalysis_file).expanduser().read_bytes()
    _equal(hashlib.sha256(payload).hexdigest(), reviewed_sha256, "Reviewed file SHA-256")
    result = _strict_json(payload)
    boundary = common.verified_boundary(state_dir, investigation_id)
    old = boundary["localization"]
    _equal(result["investigationID"], investigation_id, "Investigation ID")
    _equal(result["resultVersion"], common.RESULT_VERSION, "Reanalysis version")
    _equal(result["status"], "REANALYSIS_REQUIRES_REVIEW", "Reanalysis status")
    _equal(result["originalResultVersion"], old["resultVersion"], "Original result version")
    _equal(result["originalClassification"], old["classification"], "Original classification")
    _equal(result["parentHashes"], {
        "investigationSHA256": boundary["investigationSHA256"],
        "stageLedgers": boundary["stageLedgerSHA256"],
        "originalResultSHA256": sha256_json(old)}, "Reanalysis ancestry")
    _equal(result["methodContract"], common._method_contract(), "Method contract")
    _equal(result["methodContractSHA256"], sha256_json(result["methodContract"]), "Method contract hash")
    _source_inventory(result["methodSourceSHA256"])
    for key in ("investigationModified", "claimLevelChanged", "discoveryClaim", "catalogQueriesRepeated",
                "catalogAnswerKeyUsed", "pixelDataChangedFrozenEventDefinition",
                "physicalMechanismResolved", "companionNatureResolved"):
        _require(result.get(key) is False, f"Unsupported reanalysis claim: {key}")
    _equal(result["frozenCatalogSHA256"], sha256_json(old["frozenCatalog"]), "Frozen catalog hash")
    _equal(result["sectorRejections"], old["sectorRejections"], "Original rejected sectors")
    _equal([s["sector"] for s in result["sectorResults"]],
           [s["sector"] for s in old["sectorResults"]], "Measured sector accounting")
    for sector, previous in zip(result["sectorResults"], old["sectorResults"]):
        _verify_sector(sector, previous, old["frozenCatalog"])
    _equal(result["pixelEvidenceSHA256BySector"], old["pixelEvidenceSHA256BySector"], "Pixel ancestry")
    summary = original.summarize_eclipse_results(
        results=result["sectorResults"], rejections=result["sectorRejections"],
        binary_confirmation=boundary["binary"], identity=boundary["identity"],
        frozen_catalog=old["frozenCatalog"], result_version=common.RESULT_VERSION)
    for key, value in summary.items():
        _equal(result.get(key), value, f"Recomputed summary {key}")
    _require(summary["classification"] == "TARGET_CONSISTENT_ECLIPSE_SOURCE"
             and summary["sourceAttributionResolved"] is True,
             "Continuation requires resolved target-consistent v2 evidence")
    return result, boundary, payload


def _report(conclusion):
    result = conclusion["eclipseCommonSupportLocalization"]
    rows = ["# OpenStar TESS localization continuation", "",
            f"Investigation: `{conclusion['investigationID']}`", "",
            f"Claim level: **{conclusion['claim']['claim']}** (unchanged)", "",
            f"Source attribution: **{result['classification']}**", "",
            f"Attributed catalog source: `{result['attributedCatalogHypothesis']}`", "",
            f"Frozen physical period: {result['frozenEphemeris']['refinedPeriodDays']} days", "",
            "| Sector | Role | Previous attribution | Revised attribution | Target distance (pixels) | Uncertainty (pixels) |",
            "| --- | --- | --- | --- | ---: | ---: |"]
    for sector in result["sectorResults"]:
        distance = next(d["distancePixels"] for d in sector["catalogDistances"] if d["isTarget"])
        rows.append(f"| {sector['sector']} | {sector['role']} | {sector['originalMatchedCatalogHypothesis']} | "
                    f"{sector['matchedCatalogHypothesis']} | {distance:.6f} | {sector['centroidUncertaintyPixels']:.6f} |")
    rows.extend(["", f"Usable independent sectors: {result['usableIndependentSectorCount']}; "
                 f"required: {result['requiredIndependentSectorCount']}. The primary does not count.", "",
                 "The original v1 conflicting localization, terminal ledgers, reports and claim decision are preserved.",
                 "This continuation records the reviewed v2 evidence with its own identity and provenance.", "",
                 f"Reviewed artifact SHA-256: `{conclusion['reviewedReanalysisSHA256']}`", "",
                 "Saved image moments, jackknife aggregates and policy decisions were checked offline. "
                 "Raw cadence jackknives, WCS projection and catalog-overlap geometry were not rerun; "
                 "these remain bound to the explicitly reviewed artifact and its producer provenance.", "",
                 "The companion's nature and physical mechanism remain unresolved. Unresolved blends "
                 "and image systematics are not excluded by this attribution.", "",
                 "Recommended next test: `SOURCE_ATTRIBUTION_REVIEW`. No subsequent science stage was scheduled.", ""])
    return "\n".join(rows).encode("utf-8")


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


@contextmanager
def _publication_lock(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _already_recorded(store, investigation, reviewed_sha256):
    if not investigation.stages or investigation.stages[-1].handler_id != FINAL_HANDLER:
        return None
    final = investigation.stages[-1]
    result = final.result or {}
    _require(len(investigation.stages) >= 2
             and investigation.stages[-2].handler_id == IMPORT_HANDLER
             and final.triggered_by_stage_id == investigation.stages[-2].id
             and result.get("resultVersion") == VERSION,
             "Recorded continuation identity or causality mismatch")
    _equal(result.get("reviewedReanalysisSHA256"), reviewed_sha256, "Already recorded reanalysis")
    _require(investigation.status == "COMPLETE" and final.stop and final.next_stage is None,
             "Recorded continuation is not terminal")
    for stage in investigation.stages:
        _require(store.verified_terminal_stage_ledger_hash(investigation.id, stage),
                 "Recorded terminal ledger mismatch")
        if stage.result is not None:
            _require(stage.provenance is not None and stage.provenance.result_hash == sha256_json(stage.result),
                     "Recorded stage result mismatch")
        for artifact in stage.artifacts:
            _equal(sha256_file(artifact.path), artifact.sha256, "Recorded artifact hash")
    return {"status": "ALREADY_RECORDED", "investigationModified": False,
            "reportPath": result["reportPath"], "conclusionPath": result["conclusionPath"]}


def run_continuation(state_dir, investigation_id, reanalysis_file, reviewed_sha256, *, execute=False):
    """Append immutable stages and replace only the current snapshot, last.

Directory flock serializes this CLI's writers. The investigation must be idle;
legacy writers do not participate in this lock. Snapshot checks refuse detected
concurrent changes. Publication failures before snapshot commit roll back only
this call's new files. Existing history is never edited.
"""
    state = Path(state_dir).expanduser().resolve()
    root = state / "investigations"
    _require(root.is_dir(), "Investigation directory does not exist")
    store = InvestigationStore(root)
    snapshot = store.path_for(investigation_id).read_bytes()
    investigation = store.decode_snapshot(_strict_json(snapshot))
    reanalysis_file = Path(reanalysis_file).expanduser().resolve()
    _require(common._is_hash(reviewed_sha256), "Invalid reviewed SHA-256")
    _equal(sha256_file(reanalysis_file), reviewed_sha256, "Reviewed file SHA-256")
    recorded = _already_recorded(store, investigation, reviewed_sha256)
    if recorded is not None:
        return recorded
    result, boundary, payload = validate_reanalysis(state, investigation_id, reanalysis_file, reviewed_sha256)
    _equal(boundary["investigationSHA256"], hashlib.sha256(snapshot).hexdigest(),
           "Investigation changed during validation")
    directory = store.directory_for(investigation_id)
    package = directory / "artifacts" / PACKAGE
    stage_ids = [f"{len(investigation.stages) + offset:03d}-{suffix}"
                 for offset, suffix in ((1, "accept-eclipse-common-support-v2"),
                                        (2, "finalize-eclipse-common-support-v2"))]
    destinations = [store.stage_path_for(investigation_id, stage_id) for stage_id in stage_ids]
    _require(not package.exists() and not package.is_symlink()
             and all(not p.exists() and not p.is_symlink() for p in destinations),
             "Continuation output already exists; incomplete publication requires inspection")
    previous_final = investigation.stages[-1]
    claim = copy.deepcopy((previous_final.result or {}).get("claim"))
    _require(isinstance(claim, dict) and claim.get("claim") == "CANDIDATE_PERIOD",
             "Continuation requires the unchanged CANDIDATE_PERIOD claim")
    source_hash = sha256_file(__file__)
    preview = {"status": "VALIDATED_NO_CHANGES", "investigationModified": False,
               "classification": result["classification"], "newStageIDs": stage_ids,
               "reportPath": str(package / "report-eclipse-common-support-v2.md"),
               "conclusionPath": str(package / "conclusion-eclipse-common-support-v2.json")}
    if not execute:
        return preview
    with _publication_lock(directory):
        _equal(common.verified_boundary(state, investigation_id), boundary, "Investigation changed before publication")
        _equal(sha256_file(reanalysis_file), reviewed_sha256, "Reviewed file changed before publication")
        _equal(sha256_file(__file__), source_hash, "Continuation source changed")
        _source_inventory(result["methodSourceSHA256"])
        parent_bytes = store.path_for(investigation_id).read_bytes()
        _equal(hashlib.sha256(parent_bytes).hexdigest(), boundary["investigationSHA256"], "Parent snapshot")
        conclusion = {
            "resultVersion": VERSION, "investigationID": investigation_id,
            "claim": claim, "claimLevelChanged": False, "automaticDiscoveryClaim": False,
            "physicalMechanismResolved": False, "companionNatureResolved": False,
            "reviewedReanalysisSHA256": reviewed_sha256,
            "previousConclusionSHA256": sha256_json(previous_final.result),
            "parentHashes": copy.deepcopy(result["parentHashes"]),
            "periodEvidence": copy.deepcopy((previous_final.result or {}).get("periodEvidence")),
            "eclipseCommonSupportLocalization": result,
            "sourceAttributionResolved": True, "classification": result["classification"],
            "recommendedNextTest": "SOURCE_ATTRIBUTION_REVIEW",
            "reportPath": preview["reportPath"], "conclusionPath": preview["conclusionPath"],
            "continuationSourceSHA256": source_hash,
        }
        files = {"reviewed-reanalysis-v2.json": payload, "parent-investigation.json": parent_bytes,
                 "report-eclipse-common-support-v2.md": _report(conclusion),
                 "conclusion-eclipse-common-support-v2.json": _json_bytes(conclusion)}
        refs = {name: ArtifactReference(str(package / name), hashlib.sha256(data).hexdigest(),
                                       "text/markdown" if name.endswith(".md") else "application/json")
                for name, data in files.items()}
        params = {"reviewedReanalysisSHA256": reviewed_sha256, "continuationVersion": VERSION}
        inputs = {"reviewedReanalysis": reviewed_sha256,
                  "parentInvestigation": boundary["investigationSHA256"],
                  "originalLocalization": sha256_json(boundary["localization"]),
                  "continuationSource": source_hash}
        accepted = store.build_terminal_stage(
            stage_id=stage_ids[0], handler_id=IMPORT_HANDLER, status="COMPLETE",
            triggered_by_stage_id=previous_final.id, parameters=params, result=result, error=None,
            software_id=VERSION, software_version="1", input_hashes=inputs,
            artifacts=(refs["reviewed-reanalysis-v2.json"], refs["parent-investigation.json"]),
            next_stage={"id": stage_ids[1], "handler_id": FINAL_HANDLER, "parameters": {},
                        "triggered_by_stage_id": stage_ids[0]})
        final = store.build_terminal_stage(
            stage_id=stage_ids[1], handler_id=FINAL_HANDLER, status="COMPLETE",
            triggered_by_stage_id=accepted.id, parameters={}, result=conclusion, error=None,
            software_id=VERSION, software_version="1", input_hashes={"acceptedLocalization": sha256_json(result)},
            artifacts=(refs["report-eclipse-common-support-v2.md"], refs["conclusion-eclipse-common-support-v2.json"]),
            stop=True)
        updated = replace(investigation, stages=investigation.stages + (accepted, final),
                          updated_at=utc_now_iso())
        temporary = Path(tempfile.mkdtemp(prefix=".eclipse-continuation-", dir=directory))
        published, ledgers = False, []
        try:
            for name, data in files.items():
                with (temporary / name).open("wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            for index, stage in enumerate((accepted, final)):
                (temporary / f"stage-{index}.json").write_bytes(_json_bytes(asdict(stage)))
            _equal(common.verified_boundary(state, investigation_id), boundary, "Investigation changed during publication")
            _equal(sha256_file(reanalysis_file), reviewed_sha256, "Reviewed file changed during publication")
            _equal(sha256_file(__file__), source_hash, "Continuation source changed")
            _source_inventory(result["methodSourceSHA256"])
            package.parent.mkdir(parents=True, exist_ok=True)
            _require(not package.exists() and not package.is_symlink(), "Continuation output appeared")
            os.rename(temporary, package)
            published = True
            for index, path in enumerate(destinations):
                os.link(package / f"stage-{index}.json", path)
                ledgers.append(path)
            _equal(sha256_file(store.path_for(investigation_id)), boundary["investigationSHA256"],
                   "Investigation changed before snapshot commit")
            store.save(updated)
        except BaseException:
            # Never delete published evidence if the new snapshot became visible.
            if store.load(investigation_id).stages == investigation.stages:
                for path in ledgers:
                    path.unlink()
                if published:
                    shutil.rmtree(package)
            raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {**preview, "status": "RECORDED", "investigationModified": True}
