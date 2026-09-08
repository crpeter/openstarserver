"""Verified bridge from the recorded v2 localization to existing companion stages."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

from openstar_investigation import InvestigationStore, sha256_file, sha256_json
from openstar_workflow import StageRequest
from . import tess_eclipse_common_support as common
from . import tess_eclipse_common_support_continuation as recorded
from . import tess_eclipse_event_localization as original
from .tess_external_companion_evidence import (
    REVIEW_HANDLER_ID, FREEZE_HANDLER_ID, INTERPRET_HANDLER_ID, review_source_attribution,
)


SOFTWARE_ID = "openstar.tess-recorded-eclipse-evidence"
SOFTWARE_VERSION = "1"
DEPTH_FREEZE = "openstar.tess.event-depth-photometry.freeze"
DEPTH_AUDIT = "openstar.tess.event-depth-attenuation.audit"
JOINT_MODEL = "openstar.tess.joint-event-phase-model.fit"
SYNTHESIS = "openstar.tess.final-companion-evidence-synthesis"
FINALIZE = "openstar.tess.finalize"
ALLOWED_HANDLERS = {REVIEW_HANDLER_ID, DEPTH_FREEZE, DEPTH_AUDIT, JOINT_MODEL,
                    FREEZE_HANDLER_ID, INTERPRET_HANDLER_ID, SYNTHESIS, FINALIZE}
_require, _equal = recorded._require, recorded._equal


def verified_recorded_localization(store, investigation):
    """Return v2 plus ledger provenance, or None when no v2 continuation exists.

Verify the immutable parent snapshot against the live prefix; never relabel v2
as v1, reopen the original boundary, or read a catalog answer to repair evidence.
The saved-cadence/WCS verification limits of PR #191 still apply.
"""
    accepted = [s for s in investigation.stages if s.handler_id == recorded.IMPORT_HANDLER]
    finalized = [s for s in investigation.stages if s.handler_id == recorded.FINAL_HANDLER]
    if not accepted and not finalized:
        return None, {}
    _require(len(accepted) == len(finalized) == 1, "Recorded v2 stages are missing or duplicated")
    accepted, final = accepted[0], finalized[0]
    index = investigation.stages.index(accepted)
    _require(index > 0 and investigation.stages[index + 1:index + 2] == (final,)
             and accepted.status == final.status == "COMPLETE"
             and accepted.triggered_by_stage_id == investigation.stages[index - 1].id
             and final.triggered_by_stage_id == accepted.id
             and final.stop and final.next_stage is None, "Recorded v2 stage causality mismatch")
    _equal(accepted.next_stage, {"id": final.id, "handler_id": recorded.FINAL_HANDLER,
                                "parameters": {}, "triggered_by_stage_id": accepted.id}, "Recorded continuation link")
    ledgers = {}
    for position, stage in enumerate(investigation.stages):
        if stage.status == "RUNNING" and position == len(investigation.stages) - 1:
            continue
        digest = store.verified_terminal_stage_ledger_hash(investigation.id, stage)
        _require(digest, f"Unverified terminal ledger: {stage.id}")
        ledgers[stage.id] = digest
        if stage.result is not None:
            _require(stage.provenance is not None and stage.provenance.result_hash == sha256_json(stage.result),
                     f"Stage result hash mismatch: {stage.id}")
        for artifact in stage.artifacts:
            _equal(sha256_file(artifact.path), artifact.sha256, "Recorded artifact hash")

    def artifact_bytes(name):
        matches = [a for a in accepted.artifacts if Path(a.path).name == name]
        _require(len(matches) == 1, f"Missing unique recorded artifact: {name}")
        payload = Path(matches[0].path).read_bytes()
        _equal(hashlib.sha256(payload).hexdigest(), matches[0].sha256, "Recorded artifact changed while reading")
        return payload, matches[0].sha256

    parent_bytes, parent_hash = artifact_bytes("parent-investigation.json")
    result_bytes, result_hash = artifact_bytes("reviewed-reanalysis-v2.json")
    parent = InvestigationStore.decode_snapshot(recorded._strict_json(parent_bytes))
    result = recorded._strict_json(result_bytes)
    _require(parent.id == investigation.id and parent.workflow_id == investigation.workflow_id
             and parent.workflow_version == investigation.workflow_version and parent.status == "COMPLETE",
             "Parent investigation identity mismatch")
    _equal([asdict(s) for s in parent.stages],
           [asdict(s) for s in investigation.stages[:index]], "Preserved investigation prefix")
    _equal(result, accepted.result, "Accepted reviewed localization")
    _equal(result["investigationID"], investigation.id, "Localization investigation ID")
    _equal(result["resultVersion"], common.RESULT_VERSION, "Recorded localization version")
    _equal(result["status"], "REANALYSIS_REQUIRES_REVIEW", "Recorded localization status")
    _equal(result["methodContract"], common._method_contract(), "Recorded method contract")
    _equal(result["methodContractSHA256"], sha256_json(result["methodContract"]), "Recorded method contract hash")
    recorded._source_inventory(result["methodSourceSHA256"])
    conclusion = final.result or {}
    _equal(conclusion.get("resultVersion"), recorded.VERSION, "Recorded conclusion version")
    _equal(conclusion.get("eclipseCommonSupportLocalization"), result, "Conclusion localization")
    _equal(conclusion.get("reviewedReanalysisSHA256"), result_hash, "Reviewed artifact digest")
    _equal(accepted.parameters.get("reviewedReanalysisSHA256"), result_hash, "Recorded approval parameters")
    _equal(accepted.provenance.input_hashes.get("reviewedReanalysis"), result_hash, "Recorded approval provenance")
    _equal(conclusion.get("continuationSourceSHA256"), sha256_file(recorded.__file__), "Recording source digest")
    _equal(conclusion.get("claim"), parent.stages[-1].result.get("claim"), "Preserved claim")
    _require(conclusion.get("claimLevelChanged") is False
             and conclusion.get("automaticDiscoveryClaim") is False
             and conclusion.get("physicalMechanismResolved") is False
             and conclusion.get("companionNatureResolved") is False
             and conclusion.get("sourceAttributionResolved") is True
             and conclusion.get("recommendedNextTest") == "SOURCE_ATTRIBUTION_REVIEW",
             "Invalid recorded conclusion claims")

    def latest(handler):
        matches = [s for s in parent.stages if s.handler_id == handler and s.status == "COMPLETE"]
        _require(matches, f"Missing parent handler: {handler}")
        return matches[-1]

    old_stage = latest(original.HANDLER_ID)
    old = old_stage.result
    binary = latest("openstar.tess.binary-confirmation.analyze").result
    identity = latest("openstar.tess.catalog-identity").result
    _require(parent.stages[-1].handler_id == FINALIZE and parent.stages[-1].stop
             and parent.stages[-1].triggered_by_stage_id == old_stage.id
             and old["classification"] == "CROSS_SECTOR_SOURCE_DISAGREEMENT_OR_BLEND"
             and old["resultVersion"] == original.RESULT_VERSION
             and original.authoritative_binary_gate(binary), "Invalid original conflicting boundary")
    _equal(result["parentHashes"], {
        "investigationSHA256": parent_hash,
        "stageLedgers": {s.id: ledgers[s.id] for s in parent.stages},
        "originalResultSHA256": sha256_json(old)}, "Recorded v2 ancestry")
    _equal(conclusion.get("parentHashes"), result["parentHashes"], "Conclusion ancestry")
    _equal(result["frozenCatalogSHA256"], sha256_json(old["frozenCatalog"]), "Frozen catalog digest")
    _equal(result["sectorRejections"], old["sectorRejections"], "Preserved rejected sectors")
    _equal([s["sector"] for s in result["sectorResults"]],
           [s["sector"] for s in old["sectorResults"]], "Recorded sector accounting")
    for sector, previous in zip(result["sectorResults"], old["sectorResults"]):
        recorded._verify_sector(sector, previous, old["frozenCatalog"])
    _equal(result["pixelEvidenceSHA256BySector"], old["pixelEvidenceSHA256BySector"], "Frozen pixel ancestry")
    summary = original.summarize_eclipse_results(
        results=result["sectorResults"], rejections=result["sectorRejections"],
        binary_confirmation=binary, identity=identity, frozen_catalog=old["frozenCatalog"],
        result_version=common.RESULT_VERSION)
    for key, value in summary.items():
        _equal(result.get(key), value, f"Recorded summary {key}")
    _require(result["classification"] == "TARGET_CONSISTENT_ECLIPSE_SOURCE"
             and result["sourceAttributionResolved"] is True, "Recorded v2 source is unresolved")
    review_source_attribution(result, allow_common_support_v2=True)
    return result, {"recordedLocalizationLedger": ledgers[accepted.id],
                    "recordedConclusionLedger": ledgers[final.id],
                    "reviewedLocalizationFile": result_hash}


def _verify_tail(boundary, tail):
    previous = boundary
    for stage in tail:
        _require(stage.triggered_by_stage_id == previous.id, "Evidence stage causality mismatch")
        if previous.handler_id == recorded.FINAL_HANDLER:
            _require(stage.handler_id == REVIEW_HANDLER_ID and stage.parameters == {},
                     "Evidence must begin with source-attribution review")
        elif previous.status == "FAILED":
            _require(previous.failure_classification == "TRANSIENT_INFRASTRUCTURE"
                     and previous.handler_id in {DEPTH_FREEZE, FREEZE_HANDLER_ID}
                     and stage.handler_id == previous.handler_id and stage.parameters == previous.parameters,
                     "Invalid evidence retry lineage")
        else:
            _equal(previous.next_stage, {"id": stage.id, "handler_id": stage.handler_id,
                                        "parameters": stage.parameters, "triggered_by_stage_id": previous.id},
                   "Persisted evidence stage link")
        if stage.status == "COMPLETE":
            value = stage.result or {}
            if stage.handler_id == FINALIZE:
                _require(stage.stop and stage.next_stage is None, "Finalizer must stop")
            else:
                if stage.handler_id == REVIEW_HANDLER_ID:
                    expected = DEPTH_FREEZE if value.get("recommendedNextTest") == "EXTERNAL_COMPANION_EVIDENCE_FREEZE" else FINALIZE
                elif stage.handler_id == DEPTH_FREEZE:
                    expected = DEPTH_AUDIT
                elif stage.handler_id == DEPTH_AUDIT:
                    from .tess_joint_event_phase_model import model_required
                    expected = JOINT_MODEL if model_required(value) else FREEZE_HANDLER_ID
                elif stage.handler_id == JOINT_MODEL:
                    expected = FREEZE_HANDLER_ID
                elif stage.handler_id == FREEZE_HANDLER_ID:
                    expected = INTERPRET_HANDLER_ID
                elif stage.handler_id == INTERPRET_HANDLER_ID:
                    expected = SYNTHESIS if (value.get("externalCompanionEvidenceResolved") is True
                                             and value.get("recommendedNextTest") == "FINAL_COMPANION_EVIDENCE_SYNTHESIS") else FINALIZE
                else:
                    expected = FINALIZE
                _require(not stage.stop and (stage.next_stage or {}).get("handler_id") == expected,
                         "Evidence stage skipped the prescribed chronology")
        previous = stage


def _next_request(store, investigation, *, retry_failed=False):
    result, lineage = verified_recorded_localization(store, investigation)
    _require(result is not None, "A recorded v2 localization is required")
    boundary_index = next(i for i, s in enumerate(investigation.stages)
                          if s.handler_id == recorded.FINAL_HANDLER)
    _require(not any(s.handler_id in ALLOWED_HANDLERS - {FINALIZE}
                     for s in investigation.stages[:boundary_index]),
             "Earlier companion evidence would violate the fresh evidence chronology")
    tail = investigation.stages[boundary_index + 1:]
    _require(all(s.handler_id in ALLOWED_HANDLERS for s in tail), "Unrelated stage after recorded v2 boundary")
    _verify_tail(investigation.stages[boundary_index], tail)
    numbers = [int(s.id.split("-", 1)[0]) for s in investigation.stages]
    next_id = max(numbers) + 1
    last = investigation.stages[-1]
    if not tail:
        _require(investigation.status == "COMPLETE" and not retry_failed, "Expected completed v2 boundary")
        for name in ("external-companion-evidence", "event-depth-accuracy",
                     "joint-event-phase-model", "companion-evidence-synthesis"):
            directory = store.directory_for(investigation.id) / "artifacts" / name
            _require(not directory.exists(), f"Existing companion evidence requires inspection: {directory}")
        return StageRequest(f"{next_id:03d}-source-attribution-review", REVIEW_HANDLER_ID,
                            {}, last.id), result, lineage
    if last.status == "FAILED":
        _require(retry_failed and investigation.status == "FAILED"
                 and last.failure_classification == "TRANSIENT_INFRASTRUCTURE"
                 and last.handler_id in {DEPTH_FREEZE, FREEZE_HANDLER_ID},
                 "Only a transient acquisition failure can be retried with --retry-failed")
        return StageRequest(f"{next_id:03d}-retry-{last.id.split('-', 1)[1]}", last.handler_id,
                            dict(last.parameters), last.id), result, lineage
    _require(last.status == "COMPLETE" and not retry_failed, "Investigation has an interrupted or nonterminal stage")
    if last.stop:
        _require(last.handler_id == FINALIZE and investigation.status in {"COMPLETE", "HUMAN_REVIEW_REQUIRED"},
                 "Unexpected terminal evidence stage")
        return None, result, lineage
    _require(last.next_stage is not None and last.next_stage.get("handler_id") in ALLOWED_HANDLERS,
             "Missing authorized evidence continuation")
    request = StageRequest(**last.next_stage)
    _require(request.triggered_by_stage_id == last.id and request.id not in {s.id for s in investigation.stages},
             "Invalid persisted evidence continuation")
    return request, result, lineage


def run_evidence(state_dir, investigation_id, *, execute=False, retry_failed=False):
    root = Path(state_dir).expanduser().resolve() / "investigations"
    _require(root.is_dir(), "Investigation directory does not exist")
    store = InvestigationStore(root)
    investigation = store.load(investigation_id)
    _equal(investigation.id, investigation_id, "Requested investigation ID")
    request, localization, lineage = _next_request(store, investigation, retry_failed=retry_failed)
    snapshot_hash = sha256_file(store.path_for(investigation_id))
    preview = {"status": "VALIDATED_NO_CHANGES" if request else "ALREADY_COMPLETE",
               "sourceReview": review_source_attribution(localization, allow_common_support_v2=True),
               "nextStage": asdict(request) if request else None, "investigationModified": False}
    if not execute or request is None:
        return preview
    from .tess_investigation import build_engine
    with recorded._publication_lock(store.directory_for(investigation_id)):
        _equal(sha256_file(store.path_for(investigation_id)), snapshot_hash, "Investigation changed before execution")
        current = store.load(investigation_id)
        checked, _, checked_lineage = _next_request(store, current, retry_failed=retry_failed)
        _equal(asdict(checked), asdict(request), "Evidence request changed")
        _equal(checked_lineage, lineage, "Recorded lineage changed")
        engine = build_engine(store, None, poll_interval=0, timeout=None)
        current = store.set_status(current, "RUNNING")
        final = engine.run(current, request, software_id=SOFTWARE_ID,
                           software_version=SOFTWARE_VERSION, max_stages=16)
    conclusion = final.stages[-1].result or {}
    return {"status": final.status, "investigationModified": True, "stageCount": len(final.stages),
            "claim": conclusion.get("claim"), "recommendedNextTest": conclusion.get("recommendedNextTest"),
            "reportPath": conclusion.get("reportPath"), "conclusionPath": conclusion.get("conclusionPath")}
