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
from run_tess_investigation import (
    _can_continue_target_residual_astrophysical_mechanism,
)
from test_tess_residual_external_evidence import _evidence, _identity
from workflows.tess.tess_autonomy import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    _repair_target_residual_astrophysical_mechanism_terminal,
)
from workflows.tess.tess_residual_external_evidence import (
    analyze_residual_external_evidence,
)
from workflows.tess.tess_target_residual_astrophysical_mechanism import (
    ESTABLISHED_FAMILY,
    HANDLER_ID,
    INCONCLUSIVE,
    PULSATION_SUPPORTED,
    ROTATION_SUPPORTED,
    analyze_target_residual_astrophysical_mechanism,
    build_method_contract,
    method_contract_hash,
    validate_mechanism_followup_boundary,
)


def _external(root: Path, identity):
    confirmation, nonstationary, cycle, localization = _evidence(root)
    return analyze_residual_external_evidence(
        localization=localization,
        nonstationary=nonstationary,
        confirmation=confirmation,
        physical_cycle=cycle,
        identity=identity,
        expected_tic_id=123,
    )


class TargetResidualAstrophysicalMechanismTests(unittest.TestCase):
    def _analyze(self, identity):
        with tempfile.TemporaryDirectory() as temporary:
            external = _external(Path(temporary), identity)
            return analyze_target_residual_astrophysical_mechanism(
                external_evidence=external,
                identity=identity,
                expected_tic_id=123,
            )

    def test_method_contract_hash_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            external = _external(Path(temporary), _identity(gaia_class="ROT"))
            first = build_method_contract(external_evidence=external)
            second = build_method_contract(
                external_evidence=copy.deepcopy(external)
            )
        self.assertEqual(first, second)
        self.assertEqual(method_contract_hash(first), method_contract_hash(second))

    def test_rotation_and_pulsation_hypotheses_remain_unresolved(self):
        rotation = self._analyze(_identity(gaia_class="ROT"))
        pulsation = self._analyze(_identity(gaia_class="DSCT"))
        self.assertEqual(ROTATION_SUPPORTED, rotation["classification"])
        self.assertEqual(
            "SPECTROSCOPIC_ROTATION_CONSTRAINT",
            rotation["recommendedNextTest"],
        )
        self.assertEqual(PULSATION_SUPPORTED, pulsation["classification"])
        self.assertEqual(
            "ASTEROSPECTROSCOPIC_MODE_CLASSIFICATION",
            pulsation["recommendedNextTest"],
        )
        for result in (rotation, pulsation):
            self.assertFalse(result["physicalMechanismResolved"])
            self.assertFalse(result["claimLevelChanged"])
            self.assertEqual([97], result["spatialEvidence"]["offTargetSectors"])
            self.assertEqual(1, len(result["spatialEvidence"]["cautions"]))

    def test_established_family_and_inconclusive_classifications(self):
        established = self._analyze(_identity(vsx_type="ROT"))
        inconclusive = self._analyze(_identity(gaia_class="MISC"))
        self.assertEqual(ESTABLISHED_FAMILY, established["classification"])
        self.assertEqual(INCONCLUSIVE, inconclusive["classification"])
        self.assertIn(
            "NO_MECHANISM_SPECIFIC_PERSISTED_PERIOD_MATCH",
            inconclusive["insufficiencyReasons"],
        )

    def test_conflicting_residual_families_are_inconclusive(self):
        identity = _identity(vsx_type="DSCT", gaia_class="ROT")
        identity["vsx"]["matches"][0]["periodDays"] = 3.288577149
        result = self._analyze(identity)
        self.assertEqual(INCONCLUSIVE, result["classification"])
        self.assertIn(
            "CONFLICTING_RESIDUAL_MECHANISM_FAMILIES",
            result["insufficiencyReasons"],
        )

    def test_boundary_rejects_altered_or_wrong_upstream_evidence(self):
        identity = _identity(gaia_class="ROT")
        with tempfile.TemporaryDirectory() as temporary:
            external = _external(Path(temporary), identity)
        for mutator in (
            lambda value: value.update({"recommendedNextTest": "OTHER"}),
            lambda value: value.update({"physicalMechanismResolved": True}),
            lambda value: value["spatialEvidence"].update(
                {"targetSupportingSectors": [2, 4]}
            ),
            lambda value: value["targetAssociatedNonbinaryVariabilityEvidence"][0]
            .update({"classificationFamily": "BINARY_LIKE"}),
            lambda value: value["methodContract"].update({"altered": True}),
        ):
            changed = copy.deepcopy(external)
            mutator(changed)
            with self.assertRaises(RuntimeError):
                validate_mechanism_followup_boundary(
                    external_evidence=changed,
                    identity=identity,
                    expected_tic_id=123,
                )
        with self.assertRaises(RuntimeError):
            validate_mechanism_followup_boundary(
                external_evidence=external,
                identity={**identity, "ticID": 124},
                expected_tic_id=123,
            )

    def test_manual_validation_and_append_only_automatic_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = _identity(gaia_class="ROT")
            external = _external(root, identity)
            prepared_result = {"ticID": 123}
            final_result = {
                "residualExternalEvidence": external,
                "recommendedNextTest": (
                    "TARGET_RESIDUAL_ASTROPHYSICAL_MECHANISM_FOLLOWUP"
                ),
            }
            external_path = root / "residual-external-evidence-v20.10.1.json"
            conclusion_path = (
                root / "conclusion-v20.10.1-residual-external-evidence.json"
            )
            external_path.write_text(json.dumps(external), encoding="utf-8")
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
                    "047-residual-external-evidence",
                    "openstar.tess.residual-external-evidence.analyze",
                    "COMPLETE", "045-interpret-residual-mode-localization", {},
                    result=external,
                    provenance=StageProvenance(
                        "test", "1",
                        input_hashes={"catalogIdentity": sha256_json(identity)},
                    ),
                    artifacts=(ArtifactReference(
                        str(external_path), sha256_file(external_path),
                        "application/json",
                    ),),
                ),
                InvestigationStage(
                    "048-finalize", "openstar.tess.finalize", "COMPLETE",
                    "047-residual-external-evidence",
                    {"outputSuffix": "v20.10.1-residual-external-evidence"},
                    result=final_result, stop=True,
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
            _can_continue_target_residual_astrophysical_mechanism(investigation)

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
            repaired = _repair_target_residual_astrophysical_mechanism_terminal(
                ProbeStore(), investigation, control
            )
            repeated = _repair_target_residual_astrophysical_mechanism_terminal(
                ProbeStore(), repaired, repaired.metadata["controlState"]
            )
        self.assertEqual(stages, repaired.stages)
        self.assertEqual("RUNNING", repaired.status)
        selected = repaired.metadata["controlState"]["selectedExperiment"]
        self.assertEqual(HANDLER_ID, selected["handler_id"])
        self.assertEqual(
            "047-residual-external-evidence", selected["triggered_by_stage_id"]
        )
        self.assertIsNone(repeated)


if __name__ == "__main__":
    unittest.main()
