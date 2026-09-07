# Supported morphology-grid v1: frozen Apple/server wire contract

This independently discovered workload selects the canonical numerically valid,
observationally supported candidate in a fixed generic numerical grid. It has
no target-specific logic, scientific interpretation, I/O, or coordinator changes.
Existing MorphologyGrid datasets and results keep their original identities.
The private `_adapter.py` validates new public identities before constructing a
transient numerical view for the existing v1 evaluator. It never translates a
persisted v1 result into a supported-workload result.

## Identities

| Field | Frozen value |
| --- | --- |
| workloadID | `openstar.supported-morphology-grid.v1` |
| datasetSchemaID | `openstar.dataset.supported-morphology-grid.v1` |
| payloadSchemaID | `openstar.payload.supported-morphology-grid-shard.v1` |
| resultSchemaID | `openstar.result.supported-morphology-grid-shard.v1` |
| executionContractID | `openstar.supported-morphology-grid-execution.v1` |
| executionContractVersion | `1.0` |
| Apple validatorID | `openstar.supported-morphology-grid.local-double.v1` |
| supportPolicyID | `openstar.morphology-support.two-effective-widths.v1` |
| morphologyFamilyID | `openstar.microlensing-residual-morphology.v1` |
| componentTemplateFamilyID | `openstar.curve-family.symmetric-radial-amplification.v1` |

Workers must advertise the new workload/schema tuple. The Apple validator ID
is a worker capability identity, not an additional work/result payload field.

## Dataset and exact payloads

The dataset retains the [v1 numerical structure](../morphology_grid/README.md).
Required top-level fields: `id`, `datasetSchemaID`, `morphologyFamilyID`,
`componentTemplateFamilyID`, `modelClassID`, `series`, `morphologyGrid`,
`candidatesPerWorkUnit`, `executionContractID`, `executionContractVersion`,
and the additional required string `supportPolicyID`. Missing, unknown, or
incorrect policy IDs fail closed. Opaque top-level metadata remains extensible;
recognized optional routing identities, when supplied, must identify this new
workload. Nested series, grids, axes, parameters, and fits remain strict.

Each series has exactly `genericSeriesID`, `coordinates`, `values`, and
`inverseVariances`. Samples retain their original order and strictly increasing
coordinates; series use canonical generic-series-ID order. Nonnegative weights,
positive-weight rank requirements, finite arithmetic, and safe integer bounds
are unchanged. No normalization or sample filtering is performed.

Exact work payload fields (no flattening or extras):

```text
morphologyFamilyID, modelClassID, supportPolicyID, gridStartIndex, gridCount
```

Exact completed result payload fields (no flattening or extras):

```text
morphologyFamilyID, modelClassID, supportPolicyID, gridStartIndex, gridCount,
bestCandidate, evaluatedCandidateCount, invalidCandidateCount,
supportRejectedCandidateCount
```

`bestCandidate` is null or the exact v1 nested object:

```text
gridIndex, parameters, seriesFits, positiveWeightSampleCount,
weightedResidualSumSquares, nominalParameterCount,
bayesianInformationCriterion, correctedAkaikeInformationCriterion,
correctedAkaikeInformationCriterionDefined
```

Parameter fields are exactly:

- `POSITIVE_PULSE_ONLY`: `center`, `logScale`, `logShape`.
- `ORDERED_NEGATIVE_POSITIVE_DOUBLET`: `negativeCenter`, `separation`,
  `negativeLogScale`, `negativeLogShape`, `positiveLogScale`, `positiveLogShape`.
- `INDEPENDENT_PULSES`: `negativeCenter`, `positiveCenter`, `negativeLogScale`,
  `negativeLogShape`, `positiveLogScale`, `positiveLogShape`.

Each positive-only series fit has exactly `genericSeriesID`,
`positiveWeightSampleCount`, `offset`, `positiveAmplitude`,
`positiveAmplitudeSign`, and `weightedResidualSumSquares`. Both doublet models
add `negativeAmplitude` and `negativeAmplitudeSign`. Sign labels are exactly
`negative`, `zero`, and `positive`. The ordered positive center is derived,
never added to the strict ordered parameter payload.

## Numerical evaluation and eligibility

Preserve all v1 numerical behavior: sample accumulation order, separate basis
operations, weighted normal equations, partial-pivot elimination, active-set
ordering and non-strict sign constraints, parameter counts, WRSS/BIC/AICc,
strict independent center-pair ordering and rightmost-fastest mixed-radix
indices. The original evaluator and comparison helpers are reused directly.
Rank tolerance remains `1e-12`; result/objective comparison tolerance remains
`1e-9 * max(1, abs(left), abs(right))`. Winner ordering remains WRSS, BIC,
finite AICc, defined AICc before undefined, then smaller global grid index.
No additional amplitude sign gate is introduced.

For every index in ascending shard order:

1. Evaluate with the existing v1 numerical rules. If invalid, increment only
   `invalidCandidateCount`.
2. For a numerically valid candidate, compute each component's support geometry
   in this exact operation order:

   ```text
   scale = exp(logScale)
   shape = exp(logShape)
   effectiveWidth = scale * shape
   radius = 2.0 * effectiveWidth
   ```

   Never reassociate to `exp(logScale + logShape)`. Center must be finite; scale,
   shape, effective width and radius must be finite and strictly positive.
   For an ordered doublet, `positiveCenter = negativeCenter + separation`.
3. Every component must have at least one observation with `weight > 0` and
   `abs(coordinate - center) <= radius` in **every applicable series**.
   Independent searches apply only to their single series. The inclusive rule
   applies even to an exact-zero fitted amplitude. Invalid support geometry or
   absent support increments only `supportRejectedCandidateCount`.
4. Admit supported candidates to the existing deterministic winner comparison.

`evaluatedCandidateCount == gridCount` always. Eligible count is evaluated minus
both mutually exclusive rejection counts. A fully processed shard with zero
eligible candidates successfully completes with `bestCandidate: null`; it is
not a computation failure and must not be retried on that basis.

Server validation independently evaluates every original shard index and
reconstructs both rejection counts and the canonical eligible winner. It rejects
fabricated counts, unsupported/noncanonical winners, wrong indices, malformed
identities and nested fields, and inconsistent null winners. Canonicalization
preserves submitted identities and never relabels an old result.

## Reduction

Reduction accepts only verified results for the exact ascending, contiguous,
nonduplicated shard sequence, including a partial final shard. It exposes:

- `workloadStatus` and `supportedMorphologyGridStatus`, each
  `SUPPORTED_MORPHOLOGY_GRID_COMPLETE` or `SUPPORTED_MORPHOLOGY_GRID_INCOMPLETE`;
- `coverageComplete`, `totalCandidateCount`, `completedCandidateCount`,
  `totalInvalidCandidateCount`, `totalSupportRejectedCandidateCount`,
  `totalEligibleCandidateCount`, and `supportPolicyID`;
- all original `best*` summary fields and `payload.bestCandidate`.

For incomplete coverage, rejection and eligible totals describe accepted,
completed candidates only. Complete coverage with zero eligible candidates is
COMPLETE, with every winner field null. This establishes a result only for the
searched grid, without a discovery or planetary interpretation claim.

## Small portable conformance examples

The following complete single-candidate dataset uses nonconstant data and
requires no generated/downloaded fixture. Shard `(gridStartIndex=0, gridCount=1)`
has evaluated/invalid/support-rejected counts `(1, 0, 0)` and winner index `0`:

```json
{
  "id": "support-boundary",
  "datasetSchemaID": "openstar.dataset.supported-morphology-grid.v1",
  "morphologyFamilyID": "openstar.microlensing-residual-morphology.v1",
  "componentTemplateFamilyID": "openstar.curve-family.symmetric-radial-amplification.v1",
  "executionContractID": "openstar.supported-morphology-grid-execution.v1",
  "executionContractVersion": "1.0",
  "supportPolicyID": "openstar.morphology-support.two-effective-widths.v1",
  "modelClassID": "POSITIVE_PULSE_ONLY",
  "series": [{
    "genericSeriesID": "series-001",
    "coordinates": [2.0, 3.0, 4.0],
    "values": [2.0, 1.0, 0.5],
    "inverseVariances": [1.0, 1.0, 1.0]
  }],
  "morphologyGrid": {
    "centerAxis": {"start": 0.0, "step": 1.0, "count": 1},
    "logScaleAxis": {"start": 0.0, "step": 1.0, "count": 1},
    "logShapeAxis": {"values": [0.0]}
  },
  "candidatesPerWorkUnit": 1
}
```

| Change from the complete example | Evaluated | Invalid | Support rejected | Winner |
| --- | ---: | ---: | ---: | --- |
| None: first sample is exactly at radius 2 | 1 | 0 | 0 | Index 0 |
| First coordinate becomes `2.0000000000000004` | 1 | 0 | 1 | null |
| First inverse variance becomes `0.0` | 1 | 0 | 1 | null |

The last two examples fully reduce to COMPLETE with zero eligible candidates.
When comparing fitted floating-point fields, use v1 tolerances rather than an
exact-zero WRSS assertion.

Additional compact vectors, covered by the focused tests:

| Case | Inputs and expected result |
| --- | --- |
| Mixed-radix mapping | Counts `(3,4)`, indices `(1,2)` map to index `6` and invert exactly. |
| Independent pair order | Three centers give pairs `(0,1)`, `(0,2)`, `(1,2)` at pair indices `0,1,2`. |
| Ordered support | Centers `negative=-1`, `separation=2` derive `positive=1`. With both scales `0.25`, both shapes `1`, observations at `-1,+1` support both components. Zero weight at `+1` in any applicable series rejects the candidate when no other sample is within `0.5`. |
| Mixed rejection categories | Positive centers `[0,1,2]`, log scale `0`, log shapes `[-400, ln(0.25)]`, samples `[-3,-2,-1,1,2,3]`, all weights `1`: six candidates, two numerical-invalid, two support-rejected, two eligible. Use the nonconstant values below. |
| Unsupported numerical winner | Positive centers `[0,1]`, log scale `ln(0.25)`, log shape `0`, the same six samples and weights, values `100*basis(x,0,ln(0.25),0)+0.001*x`: original numerical winner `0`, supported winner `1`. |

For the mixed case, use those same six nonconstant values from the last row;
`basis` is the exact operation sequence in the linked v1 contract. The tiny
shape at an observed center underflows its squared value, causing numerical
invalidity before support is considered. At unobserved center zero it remains
numerically valid but lacks support. Expected counts are independent of fitted
amplitude being zero.

Local focused tests (not executed during implementation):

```bash
python -m unittest discover -s tests/workloads/supported_morphology_grid -p 'test_*.py' -v
python -m unittest tests.workflows.microlensing.test_build_supported_anomaly_morphology_grid -v
```
