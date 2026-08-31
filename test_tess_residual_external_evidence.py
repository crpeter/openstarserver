import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from openstar_investigation import (
    ArtifactReference,
    Investigation,
    InvestigationStage,
    StageProvenance,
    sha256_file,
    sha256_json,
)
from run_tess_investigation import _can_continue_residual_external_evidence
from test_tess_confirmed_residual_mode_localization import _boundary
from test_tess_long_baseline_frequency_confirmation import INDEPENDENT_SECTORS
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    _repair_target_supported_residual_external_evidence_terminal,
)
from workflows.tess.tess_investigation import (
    residual_mode_localization_continuation,
)
from workflows.tess.tess_residual_external_evidence import (
    HANDLER_ID,
    analyze_residual_external_evidence,
    build_method_contract,
    method_contract_hash,
    validate_target_supported_boundary,
)


def _identity(*, vsx_type=None, gaia_class=None, catalog_error=False):
    vsx = {
        "found": vsx_type is not None,
        "matches": ([{
            "name": "TEST-VSX",
            "type": vsx_type,
            "periodDays": 10.0,
            "separationArcsec": 0.2,
        }] if vsx_type is not None else []),
        "queryProvenance": {"catalog": "B/vsx/vsx"},
    }
    if catalog_error:
        vsx["queryError"] = "TimeoutError: frozen failure"
    return {
        "ticID": 123,
        "identityResolved": True,
        "catalogCoverageComplete": not catalog_error,
        "tic": {
            "found": True,
            "aliases": {"GAIA_field": 987},
        },
        "simbad": {"found": False, "queryProvenance": {"objectID": "TIC 123"}},
        "vsx": vsx,
        "gaiaDR3": {
            "found": True,
            "nearest": {"sourceID": 987, "separationArcsec": 0.1},
        },
        "gaiaVariability": {
            "found": gaia_class is not None,
            "classification": ({"class": gaia_class} if gaia_class else None),
            "periodCandidates": ([{"periodDays": 3.288577149}]
                                 if gaia_class else []),
            "queryProvenance": {"gaiaSourceID": 987},
        },
    }


def _evidence(root: Path):
    paths = [root / f"sector-{sector}.json"
             for sector in (1, *INDEPENDENT_SECTORS)]
    confirmation, nonstationary, cycle = _boundary(paths)
    nonstationary["preferredModel"] = {
        "signalSectors": [1, 2, 4, 97, 98]
    }
    localization = {
        "version": "openstar.tess-residual-mode-pixel-localization.v1",
        "ticID": 123,
        "physicalPeriodDays": 10.0,
        "residualFrequencyAtReference": 0.304082882,
        "residualPeriodAtReferenceDays": 3.288577149,
        "fractionalFrequencyDriftPerDay": 0.0,
        "signalSectors": [1, 2, 4, 97, 98],
        "sectorResults": [
            {"sector": 1, "role": "primary", "classification": "TARGET_CONSISTENT"},
            {"sector": 2, "role": "independent", "classification": "TARGET_CONSISTENT"},
            {"sector": 4, "role": "independent", "classification": "TARGET_CONSISTENT"},
            {"sector": 97, "role": "independent", "classification": "OFF_TARGET"},
            {"sector": 98, "role": "independent", "classification": "TARGET_CONSISTENT"},
        ],
        "crossSector": {
            "classification": "RESIDUAL_MODE_TARGET_SUPPORTED",
            "residualModeOrigin": "TARGET_CONSISTENT",
            "independentEligibleSectorCount": 4,
            "requiredIndependentSupportCount": 3,
            "targetSupportingSectors": [2, 4, 98],
            "offTargetSectors": [97],
            "ambiguousSectors": [],
            "recommendedNextTest": (
                "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
            ),
        },
        "claimLevelChanged": False,
        "physicalMechanismResolved": False,
        "recommendedNextTest": (
            "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
        ),
    }
    return confirmation, nonstationary, cycle, localization


class ResidualExternalEvidenceTests(unittest.TestCase):
    def _analyze(self, identity):
        with tempfile.TemporaryDirectory() as temporary:
            confirmation, nonstationary, cycle, localization = _evidence(
                Path(temporary)
            )
            return analyze_residual_external_evidence(
                localization=localization,
                nonstationary=nonstationary,
                confirmation=confirmation,
                physical_cycle=cycle,
                identity=identity,
                expected_tic_id=123,
            )

    def test_method_contract_hash_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, nonstationary, _, localization = _evidence(Path(temporary))
            first = build_method_contract(
                localization=localization, nonstationary=nonstationary
            )
            second = build_method_contract(
                localization=copy.deepcopy(localization),
                nonstationary=copy.deepcopy(nonstationary),
            )
        self.assertEqual(first, second)
        self.assertEqual(method_contract_hash(first), method_contract_hash(second))

    def test_binary_classification_preserves_claim_and_spatial_caution(self):
        result = self._analyze(_identity(vsx_type="EW"))
        self.assertEqual(
            "TARGET_ASSOCIATED_BINARY_EVIDENCE_PRESENT",
            result["classification"],
        )
        self.assertEqual("SPECTROSCOPIC_BINARY_CONFIRMATION",
                         result["recommendedNextTest"])
        self.assertFalse(result["physicalMechanismResolved"])
        self.assertFalse(result["claimLevelChanged"])
        self.assertEqual([97], result["spatialEvidence"]["offTargetSectors"])
        self.assertEqual(1, len(result["spatialEvidence"]["cautions"]))

    def test_nonbinary_conflicting_and_inconclusive_classifications(self):
        nonbinary = self._analyze(_identity(gaia_class="ROT"))
        conflict = self._analyze(_identity(vsx_type="EW", gaia_class="ROT"))
        inconclusive = self._analyze(_identity(catalog_error=True))
        self.assertEqual(
            "TARGET_ASSOCIATED_NONBINARY_VARIABILITY_EVIDENCE_PRESENT",
            nonbinary["classification"],
        )
        self.assertEqual("CONFLICTING_TARGET_EXTERNAL_CLASSIFICATIONS",
                         conflict["classification"])
        self.assertEqual(
            "EXTERNAL_VARIABILITY_AND_BINARY_EVIDENCE_INCONCLUSIVE",
            inconclusive["classification"],
        )
        self.assertIn(
            "VSX_QUERY_WAS_NOT_AVAILABLE_IN_FROZEN_IDENTITY",
            inconclusive["insufficiencyReasons"],
        )

    def test_boundary_rejects_altered_localization_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            confirmation, nonstationary, cycle, localization = _evidence(
                Path(temporary)
            )
            identity = _identity()
            for mutator in (
                lambda value: value.update({"recommendedNextTest": "OTHER"}),
                lambda value: value.update({"physicalMechanismResolved": True}),
                lambda value: value["crossSector"].update(
                    {"targetSupportingSectors": [2, 4]}
                ),
            ):
                changed = copy.deepcopy(localization)
                mutator(changed)
                with self.assertRaises(RuntimeError):
                    validate_target_supported_boundary(
                        localization=changed,
                        nonstationary=nonstationary,
                        confirmation=confirmation,
                        physical_cycle=cycle,
                        identity=identity,
                        expected_tic_id=123,
                    )
            wrong_identity = copy.deepcopy(identity)
            wrong_identity["ticID"] = 124
            with self.assertRaises(RuntimeError):
                validate_target_supported_boundary(
                    localization=localization,
                    nonstationary=nonstationary,
                    confirmation=confirmation,
                    physical_cycle=cycle,
                    identity=wrong_identity,
                    expected_tic_id=123,
                )

    def test_workflow_continuation_routes_only_exact_positive_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, localization = _evidence(Path(temporary))
        request = residual_mode_localization_continuation(
            localization, request_id="040-interpret-residual-mode-localization"
        )
        self.assertEqual(HANDLER_ID, request.handler_id)
        self.assertEqual("040-interpret-residual-mode-localization",
                         request.triggered_by_stage_id)
        changed = copy.deepcopy(localization)
        changed["crossSector"]["classification"] = "OTHER"
        self.assertEqual(
            "openstar.tess.finalize",
            residual_mode_localization_continuation(
                changed, request_id="040-localization"
            ).handler_id,
        )

    def test_manual_validation_and_append_only_automatic_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            confirmation, nonstationary, cycle, localization = _evidence(root)
            identity = _identity()
            prepared_result = {"ticID": 123}
            final_result = {
                "residualModeLocalization": localization,
                "recommendedNextTest": (
                    "EXTERNAL_VARIABILITY_CLASSIFICATION_AND_BINARY_EVIDENCE"
                ),
            }
            localization_path = root / "residual-mode-localization-v20.10.json"
            conclusion_path = root / "conclusion-v20.10.json"
            localization_path.write_text(json.dumps(localization), encoding="utf-8")
            conclusion_path.write_text(json.dumps(final_result), encoding="utf-8")
            provenance = StageProvenance(
                "test", "1",
                input_hashes={"nonstationaryModeling": sha256_json(nonstationary)},
            )
            stages = (
                InvestigationStage("001-prepare-target", "openstar.tess.prepare-target",
                                   "COMPLETE", None, {}, result=prepared_result),
                InvestigationStage("003-catalog-identity", "openstar.tess.catalog-identity",
                                   "COMPLETE", "001-prepare-target", {}, result=identity),
                InvestigationStage("006-source-localization",
                                   "openstar.tess.source-localization.analyze",
                                   "COMPLETE", "003-catalog-identity", {},
                                   result={"physicalCycleEvidence": cycle}),
                InvestigationStage("036-confirmation",
                                   "openstar.tess.long-baseline-frequency-confirmation.analyze",
                                   "COMPLETE", "006-source-localization", {},
                                   result=confirmation),
                InvestigationStage("041-nonstationary",
                                   "openstar.tess.nonstationary.summarize",
                                   "COMPLETE", "036-confirmation", {},
                                   result=nonstationary),
                InvestigationStage(
                    "045-interpret-residual-mode-localization",
                    "openstar.tess.residual-mode-localization.interpret",
                    "COMPLETE", "041-nonstationary", {}, result=localization,
                    provenance=provenance,
                    artifacts=(ArtifactReference(
                        str(localization_path), sha256_file(localization_path),
                        "application/json",
                    ),),
                ),
                InvestigationStage(
                    "046-finalize", "openstar.tess.finalize", "COMPLETE",
                    "045-interpret-residual-mode-localization",
                    {"outputSuffix": "v20.10"}, result=final_result, stop=True,
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
            _can_continue_residual_external_evidence(investigation)

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
            repaired = _repair_target_supported_residual_external_evidence_terminal(
                ProbeStore(), investigation, control
            )
            repeated = _repair_target_supported_residual_external_evidence_terminal(
                ProbeStore(), repaired, repaired.metadata["controlState"]
            )
        self.assertEqual(stages, repaired.stages)
        self.assertEqual("RUNNING", repaired.status)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(HANDLER_ID, selected["handler_id"])
        self.assertEqual("045-interpret-residual-mode-localization",
                         selected["triggered_by_stage_id"])
        self.assertIsNone(repeated)


if __name__ == "__main__":
    unittest.main()
