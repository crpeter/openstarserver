import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from openstar_investigation import (
    ArtifactReference,
    Investigation,
    InvestigationStage,
    StageProvenance,
    sha256_file,
    sha256_json,
)
from run_tess_investigation import (
    _can_continue_neighbor_catalog_pixel_response_review,
)
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    _repair_neighbor_catalog_pixel_response_review_terminal,
)
from workflows.tess.tess_neighbor_catalog_pixel_response_review import (
    HANDLER_ID,
    INCOMPLETE,
    INCONCLUSIVE,
    MULTI_SOURCE,
    NEIGHBOR_SUPPORTED,
    TARGET_SUPPORTED,
    analyze_neighbor_catalog_pixel_response_review,
    build_method_contract,
    method_contract_hash,
    validate_review_boundary,
)


TARGET_RA = 75.0
TARGET_DEC = -68.0
SECTORS = (2, 3, 97, 98)


def _power_map(source):
    values = [[0.1 for _ in range(5)] for _ in range(5)]
    if source == "TARGET":
        values[2][1] = 10.0
    elif source == "NEIGHBOR":
        values[2][3] = 10.0
    elif source == "BOTH":
        values[2][1] = 10.0
        values[2][3] = 10.0
    return values


def _evidence(source_by_sector=None):
    source_by_sector = source_by_sector or {sector: "TARGET" for sector in SECTORS}
    identity = {
        "ticID": 123,
        "identityResolved": True,
        "tic": {
            "found": True,
            "aliases": {"GAIA_field": 999},
        },
        "gaiaDR3": {"nearest": {"sourceID": 999}},
    }
    mode = {
        "classification": "INDEPENDENT_STABLE_MODE",
        "independentModeEvidenceSurvived": True,
        "physicalMechanismResolved": False,
        "modeCandidate": {
            "frequencyCyclesPerDay": 0.4,
            "periodDays": 2.5,
            "supportingSectors": list(SECTORS),
        },
        "independentSectorSupport": {
            "count": 4,
            "requiredCount": 3,
            "sectors": list(SECTORS),
            "sufficient": True,
        },
    }
    metadata = []
    windows = []
    for sector in SECTORS:
        key = f"sector-{sector}-window-1"
        target_pixel = {"x": 1.0, "y": 2.0}
        metadata.append({
            "windowKey": key,
            "sector": sector,
            "role": "independent",
            "windowIndex": 1,
            "shape": [5, 5],
            "targetPixel": target_pixel,
            "skyJacobian": {
                "xToEastArcsec": 20.0,
                "xToNorthArcsec": 0.0,
                "yToEastArcsec": 0.0,
                "yToNorthArcsec": 20.0,
            },
        })
        windows.append({
            "windowKey": key,
            "sector": sector,
            "role": "independent",
            "windowIndex": 1,
            "shape": [5, 5],
            "targetPixel": target_pixel,
            "classification": "TARGET_CONSISTENT",
            "localizationQualityPass": True,
            "candidatePowerMap": _power_map(source_by_sector[sector]),
        })
    preparation = {
        "available": True,
        "ticID": 123,
        "targetSky": {"raDeg": TARGET_RA, "decDeg": TARGET_DEC},
        "residualFrequencyAtReference": 0.4,
        "signalSectors": list(SECTORS),
        "windowMetadata": metadata,
    }
    review = {
        "version": "openstar.tess-residual-mode-source-localization-review.v1",
        "ticID": 123,
        "targetSky": {"raDeg": TARGET_RA, "decDeg": TARGET_DEC},
        "residualFrequencyAtReference": 0.4,
        "signalSectors": list(SECTORS),
        "windowResults": windows,
        "crossTime": {
            "classification": (
                "RESIDUAL_MODE_TIME_RESOLVED_LOCALIZATION_UNRESOLVED"
            ),
            "residualModeOrigin": "UNRESOLVED",
            "independentEligibleSectorCount": 4,
            "requiredIndependentSupportCount": 3,
            "recommendedNextTest": (
                "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW"
            ),
        },
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW",
    }
    return preparation, review, mode, identity


def _snapshot(*, error=False, close=False):
    separation = 18.0 if close else 40.0
    delta_ra = separation / (3600.0 * math.cos(math.radians(TARGET_DEC)))
    tic = {
        "sources": [
            {
                "catalog": "TIC",
                "ticID": 123,
                "isTargetTIC": True,
                "gaiaSourceID": 999,
                "raDeg": TARGET_RA,
                "decDeg": TARGET_DEC,
                "separationArcsec": 0.0,
            },
            {
                "catalog": "TIC",
                "ticID": 124,
                "isTargetTIC": False,
                "gaiaSourceID": 1000,
                "raDeg": TARGET_RA + delta_ra,
                "decDeg": TARGET_DEC,
                "separationArcsec": separation,
            },
        ]
    }
    gaia = {
        "sources": [
            {
                "catalog": "GaiaDR3",
                "gaiaSourceID": 999,
                "raDeg": TARGET_RA,
                "decDeg": TARGET_DEC,
                "separationArcsec": 0.0,
            },
            {
                "catalog": "GaiaDR3",
                "gaiaSourceID": 1000,
                "raDeg": TARGET_RA + delta_ra,
                "decDeg": TARGET_DEC,
                "separationArcsec": separation,
            },
        ]
    }
    if error:
        gaia["queryError"] = "TimeoutError: test"
    return {"tic": tic, "gaiaDR3": gaia}


def _analyze(source_by_sector=None, *, snapshot=None):
    preparation, review, mode, identity = _evidence(source_by_sector)
    return analyze_neighbor_catalog_pixel_response_review(
        preparation=preparation,
        localization_review=review,
        mode_identification=mode,
        identity=identity,
        expected_tic_id=123,
        catalog_provider=lambda *_: snapshot or _snapshot(),
    )


class NeighborCatalogPixelResponseReviewTests(unittest.TestCase):
    def test_method_contract_hash_is_deterministic(self):
        _, review, mode, _ = _evidence()
        first = build_method_contract(
            localization_review=review, mode_identification=mode
        )
        second = build_method_contract(
            localization_review=copy.deepcopy(review),
            mode_identification=copy.deepcopy(mode),
        )
        self.assertEqual(first, second)
        self.assertEqual(method_contract_hash(first), method_contract_hash(second))

    def test_method_contract_is_frozen_before_catalog_provider_access(self):
        preparation, review, mode, identity = _evidence()
        frozen = False
        original = build_method_contract

        def build(**kwargs):
            nonlocal frozen
            result = original(**kwargs)
            frozen = True
            return result

        def provider(*_):
            self.assertTrue(frozen)
            return _snapshot()

        with patch(
            "workflows.tess.tess_neighbor_catalog_pixel_response_review."
            "build_method_contract",
            side_effect=build,
        ):
            analyze_neighbor_catalog_pixel_response_review(
                preparation=preparation,
                localization_review=review,
                mode_identification=mode,
                identity=identity,
                expected_tic_id=123,
                catalog_provider=provider,
            )

    def test_target_neighbor_multi_source_and_inconclusive_classifications(self):
        target = _analyze()
        neighbor = _analyze({sector: "NEIGHBOR" for sector in SECTORS})
        mixed = _analyze({2: "TARGET", 3: "TARGET", 97: "NEIGHBOR", 98: "NEIGHBOR"})
        inconclusive = _analyze({sector: "BOTH" for sector in SECTORS})
        self.assertEqual(TARGET_SUPPORTED, target["classification"])
        self.assertEqual(NEIGHBOR_SUPPORTED, neighbor["classification"])
        self.assertEqual(MULTI_SOURCE, mixed["classification"])
        self.assertEqual(INCONCLUSIVE, inconclusive["classification"])
        self.assertEqual(SECTORS, tuple(target["aggregateDecision"]["targetSupportingSectors"]))
        self.assertEqual(
            SECTORS,
            tuple(neighbor["aggregateDecision"]["bestNeighborSupportingSectors"]),
        )

    def test_query_failure_and_close_neighbor_fail_closed(self):
        incomplete = _analyze(snapshot=_snapshot(error=True))
        missing_query = _analyze(snapshot={"tic": _snapshot()["tic"]})
        close = _analyze(snapshot=_snapshot(close=True))
        self.assertEqual(INCOMPLETE, incomplete["classification"])
        self.assertEqual(INCOMPLETE, missing_query["classification"])
        self.assertEqual(INCONCLUSIVE, close["classification"])
        self.assertTrue(any(
            row["classification"] == "UNRESOLVED_TARGET_NEIGHBOR_BLEND"
            for row in close["windowEvidence"]
        ))

    def test_boundary_rejects_altered_evidence_and_corrupted_maps(self):
        preparation, review, mode, identity = _evidence()
        for changed_preparation, changed_review, changed_mode in (
            (preparation, {**review, "recommendedNextTest": "OTHER"}, mode),
            (preparation, review, {**mode, "physicalMechanismResolved": True}),
            (
                {**preparation, "residualFrequencyAtReference": 0.41},
                review,
                mode,
            ),
        ):
            with self.assertRaises(RuntimeError):
                validate_review_boundary(
                    preparation=changed_preparation,
                    localization_review=changed_review,
                    mode_identification=changed_mode,
                    identity=identity,
                    expected_tic_id=123,
                )
        corrupted = copy.deepcopy(review)
        corrupted["windowResults"][0]["candidatePowerMap"][0][0] = float("nan")
        with self.assertRaises(RuntimeError):
            validate_review_boundary(
                preparation=preparation,
                localization_review=corrupted,
                mode_identification=mode,
                identity=identity,
                expected_tic_id=123,
            )

    def test_results_never_upgrade_claim_or_resolve_mechanism(self):
        for result in (
            _analyze(),
            _analyze({sector: "NEIGHBOR" for sector in SECTORS}),
            _analyze({sector: "BOTH" for sector in SECTORS}),
        ):
            self.assertFalse(result["claimLevelChanged"])
            self.assertFalse(result["physicalMechanismResolved"])

    def test_manual_validation_and_append_only_automatic_repair(self):
        preparation, review, mode, identity = _evidence()
        prepared_result = {"ticID": 123}
        final_result = {
            "residualModeLocalizationReview": review,
            "recommendedNextTest": "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review_path = root / "residual-mode-localization-review-v20.11.json"
            conclusion_path = root / "conclusion-v20.11.json"
            review_path.write_text(json.dumps(review), encoding="utf-8")
            conclusion_path.write_text(json.dumps(final_result), encoding="utf-8")
            stages = (
                InvestigationStage(
                    "001-prepare-target", "openstar.tess.prepare-target",
                    "COMPLETE", None, {}, result=prepared_result,
                ),
                InvestigationStage(
                    "003-catalog-identity", "openstar.tess.catalog-identity",
                    "COMPLETE", "001-prepare-target", {}, result=identity,
                ),
                InvestigationStage(
                    "017-mode-identification",
                    "openstar.tess.mode-identification.analyze",
                    "COMPLETE", "003-catalog-identity", {}, result=mode,
                ),
                InvestigationStage(
                    "021-prepare-residual-mode-localization-review",
                    "openstar.tess.residual-mode-localization-review.prepare",
                    "COMPLETE", "017-mode-identification", {}, result=preparation,
                    provenance=StageProvenance(
                        "test", "1",
                        input_hashes={"modeIdentification": sha256_json(mode)},
                    ),
                ),
                InvestigationStage(
                    "023-interpret-residual-mode-localization-review",
                    "openstar.tess.residual-mode-localization-review.interpret",
                    "COMPLETE", "021-prepare-residual-mode-localization-review",
                    {}, result=review,
                    provenance=StageProvenance(
                        "test", "1",
                        input_hashes={"preparation": sha256_json(preparation)},
                    ),
                    artifacts=(ArtifactReference(
                        str(review_path), sha256_file(review_path), "application/json"
                    ),),
                ),
                InvestigationStage(
                    "024-finalize", "openstar.tess.finalize", "COMPLETE",
                    "023-interpret-residual-mode-localization-review",
                    {"outputSuffix": "v20.11"}, result=final_result, stop=True,
                    artifacts=(ArtifactReference(
                        str(conclusion_path), sha256_file(conclusion_path),
                        "application/json",
                    ),),
                ),
            )
            investigation = Investigation(
                "test", WORKFLOW_ID, WORKFLOW_VERSION, "COMPLETE", "", "",
                {}, stages,
            )
            _can_continue_neighbor_catalog_pixel_response_review(investigation)

            class ProbeStore:
                def verified_terminal_stage_ledger_hash(self, *args):
                    return True

                def set_control_state(self, value, *, status, control_state):
                    return replace(
                        value, status=status,
                        metadata={**value.metadata, "controlState": control_state},
                    )

            control = {
                "branchAssessments": [],
                "selectedExperiment": None,
                "schedulerAction": "INVESTIGATION_COMPLETE",
            }
            repaired = _repair_neighbor_catalog_pixel_response_review_terminal(
                ProbeStore(), investigation, control
            )
            repeated = _repair_neighbor_catalog_pixel_response_review_terminal(
                ProbeStore(), repaired, repaired.metadata["controlState"]
            )
        self.assertEqual(stages, repaired.stages)
        self.assertEqual("RUNNING", repaired.status)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(HANDLER_ID, selected["handler_id"])
        self.assertEqual(
            "023-interpret-residual-mode-localization-review",
            selected["triggered_by_stage_id"],
        )
        self.assertIsNone(repeated)

    def test_running_nonterminal_wrong_recommendation_and_existing_stage_rejected(self):
        preparation, review, mode, identity = _evidence()
        final = {
            "residualModeLocalizationReview": review,
            "recommendedNextTest": "NEIGHBOR_CATALOG_AND_PIXEL_RESPONSE_REVIEW",
        }
        stages = (
            InvestigationStage(
                "001-prepare-target", "openstar.tess.prepare-target",
                "COMPLETE", None, {}, result={"ticID": 123},
            ),
            InvestigationStage(
                "003-catalog-identity", "openstar.tess.catalog-identity",
                "COMPLETE", "001-prepare-target", {}, result=identity,
            ),
            InvestigationStage(
                "017-mode-identification", "openstar.tess.mode-identification.analyze",
                "COMPLETE", "003-catalog-identity", {}, result=mode,
            ),
            InvestigationStage(
                "021-review-prepare",
                "openstar.tess.residual-mode-localization-review.prepare",
                "COMPLETE", "017-mode-identification", {}, result=preparation,
            ),
            InvestigationStage(
                "023-review", "openstar.tess.residual-mode-localization-review.interpret",
                "COMPLETE", "021-review-prepare", {}, result=review,
                provenance=StageProvenance(
                    "test", "1", input_hashes={"preparation": sha256_json(preparation)}
                ),
            ),
            InvestigationStage(
                "024-finalize", "openstar.tess.finalize", "COMPLETE", "023-review",
                {"outputSuffix": "v20.11"}, result=final, stop=True,
            ),
        )
        base = Investigation(
            "test", WORKFLOW_ID, WORKFLOW_VERSION, "COMPLETE", "", "", {}, stages
        )
        for changed in (
            replace(base, status="RUNNING"),
            replace(base, status="FAILED"),
            replace(
                base,
                stages=stages[:-1] + (
                    replace(stages[-1], result={**final, "recommendedNextTest": "OTHER"}),
                ),
            ),
            replace(
                base,
                stages=stages + (InvestigationStage(
                    "025-review", HANDLER_ID, "FAILED", "023-review", {}
                ),),
            ),
        ):
            with self.assertRaises(RuntimeError):
                _can_continue_neighbor_catalog_pixel_response_review(changed)


if __name__ == "__main__":
    unittest.main()
