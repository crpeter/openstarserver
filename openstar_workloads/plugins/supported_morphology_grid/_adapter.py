"""Private, read-only numerical bridge; never accepts or translates v1 results."""

from typing import Any, Mapping

from openstar_workloads.plugins import morphology_grid as _v1

WORKLOAD_ID = "openstar.supported-morphology-grid.v1"
DATASET_SCHEMA_ID = "openstar.dataset.supported-morphology-grid.v1"
PAYLOAD_SCHEMA_ID = "openstar.payload.supported-morphology-grid-shard.v1"
RESULT_SCHEMA_ID = "openstar.result.supported-morphology-grid-shard.v1"
EXECUTION_CONTRACT_ID = "openstar.supported-morphology-grid-execution.v1"
EXECUTION_CONTRACT_VERSION = "1.0"
VALIDATOR_ID = "openstar.supported-morphology-grid.local-double.v1"
SUPPORT_POLICY_ID = "openstar.morphology-support.two-effective-widths.v1"
MORPHOLOGY_FAMILY_ID = _v1.MORPHOLOGY_FAMILY_ID
COMPONENT_TEMPLATE_FAMILY_ID = _v1.COMPONENT_TEMPLATE_FAMILY_ID
POSITIVE_PULSE_ONLY = _v1.POSITIVE_PULSE_ONLY
ORDERED_NEGATIVE_POSITIVE_DOUBLET = _v1.ORDERED_NEGATIVE_POSITIVE_DOUBLET
INDEPENDENT_PULSES = _v1.INDEPENDENT_PULSES
MODEL_CLASS_IDS = _v1.MODEL_CLASS_IDS
MAX_SAFE_INTEGER = _v1.MAX_SAFE_INTEGER
RESULT_RELATIVE_TOLERANCE = _v1.RESULT_RELATIVE_TOLERANCE

# These pure helpers retain the published v1 arithmetic and nested result shape.
evaluate_candidate = _v1._evaluate_candidate
candidate_precedes = _v1._candidate_precedes
candidate_payload = _v1._candidate_payload
strict_candidate_payload = _v1._strict_candidate_payload
candidate_payload_matches = _v1._candidate_payload_matches
positive_integer = _v1._positive_integer
nonnegative_integer = _v1._nonnegative_integer
safe_product = _v1._safe_product
candidate_index = _v1.candidate_index
candidate_indices = _v1.candidate_indices
independent_center_pair_index = _v1.independent_center_pair_index
independent_center_pair_indices = _v1.independent_center_pair_indices
flattenable_result_fields = _v1._FLATTENABLE_RESULT_FIELDS


def check_identities(document: Mapping[str, Any], *, required: bool = False) -> None:
    """Validate public routing IDs before making any internal numerical view."""
    identities = {
        "workloadID": WORKLOAD_ID,
        "datasetSchemaID": DATASET_SCHEMA_ID,
        "payloadSchemaID": PAYLOAD_SCHEMA_ID,
        "resultSchemaID": RESULT_SCHEMA_ID,
        "executionContractID": EXECUTION_CONTRACT_ID,
        "executionContractVersion": EXECUTION_CONTRACT_VERSION,
        "supportPolicyID": SUPPORT_POLICY_ID,
        "morphologyFamilyID": MORPHOLOGY_FAMILY_ID,
        "componentTemplateFamilyID": COMPONENT_TEMPLATE_FAMILY_ID,
    }
    required_fields = {
        "datasetSchemaID", "executionContractID", "executionContractVersion",
        "supportPolicyID", "morphologyFamilyID", "componentTemplateFamilyID",
    } if required else set()
    for key, expected in identities.items():
        if key in document or key in required_fields:
            if not isinstance(document.get(key), str) or document[key] != expected:
                raise ValueError(f"{key} is invalid")


def numerical_view(dataset: Mapping[str, Any]):
    if not isinstance(dataset, Mapping):
        raise RuntimeError("supported-morphology-grid dataset must be a mapping")
    try:
        check_identities(dataset, required=True)
    except ValueError as error:
        raise RuntimeError(f"supported-morphology-grid dataset: {error}") from error
    # Only a transient numerical input is adapted. No work/result envelope or
    # persisted result is routed through the old plugin.
    view = dict(dataset)
    view.update({
        "datasetSchemaID": _v1.DATASET_SCHEMA_ID,
        "executionContractID": _v1.EXECUTION_CONTRACT_ID,
    })
    return view, _v1._validated_dataset(view)
