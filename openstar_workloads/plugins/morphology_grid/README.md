# Morphology-grid workload v1

`openstar.morphology-grid.v1` evaluates deterministic grids of generic signed
component templates against one or more generic weighted numerical series. It
contains no target identity, archive knowledge, scientific interpretation,
workflow paths, classification, or discovery logic.

## Published identities

- Workload: `openstar.morphology-grid.v1`
- Dataset: `openstar.dataset.morphology-grid.v1`
- Work payload: `openstar.payload.morphology-grid-shard.v1`
- Result: `openstar.result.morphology-grid-shard.v1`
- Morphology family: `openstar.microlensing-residual-morphology.v1`
- Component template: `openstar.curve-family.symmetric-radial-amplification.v1`
- Execution contract: `openstar.morphology-grid-execution.v1`, version `1.0`

Workers must advertise the complete workload/schema tuple. The existing
`openstar.curve-grid.v1` workload remains a separate capability.

## Models and datasets

Every dataset supplies canonically ordered generic series, a strict
model-specific `morphologyGrid`, a positive shard size, and the exact execution
contract identity. Series preserve complete equal-length coordinate, value,
and nonnegative inverse-variance arrays. Coordinates are strictly increasing.
Required top-level fields are enforced while unrelated opaque top-level
metadata is ignored; series, grids, axes, work payloads, and results are strict.
Zero-weight rows remain present but contribute to neither fitting, `N`, nor
WRSS; negative weights are invalid.

`POSITIVE_PULSE_ONLY` shares one center, log scale, and log shape across all
series and fits an unconstrained offset plus a nonnegative amplitude per
series. `ORDERED_NEGATIVE_POSITIVE_DOUBLET` shares a negative center, positive
separation, and both components' log scales and log shapes while fitting an
offset, nonpositive negative amplitude, and nonnegative positive amplitude per
series. `INDEPENDENT_PULSES` accepts exactly one series and searches a strict
negative-center/positive-center pair plus both components' log scales and log
shapes. Cross-series independent aggregation is deliberately outside this
workload.

The strict model-grid fields and rightmost-fastest axis orders are:

- `POSITIVE_PULSE_ONLY`: `centerAxis`, `logScaleAxis`, `logShapeAxis`.
- `ORDERED_NEGATIVE_POSITIVE_DOUBLET`: `negativeCenterAxis`,
  `separationAxis`, `negativeLogScaleAxis`, `negativeLogShapeAxis`,
  `positiveLogScaleAxis`, `positiveLogShapeAxis`.
- `INDEPENDENT_PULSES`: the strict ordered-pair dimension derived from
  `centerAxis`, followed by `negativeLogScaleAxis`, `negativeLogShapeAxis`,
  `positiveLogScaleAxis`, and `positiveLogShapeAxis`.

Linear axes contain exactly `start`, positive `step`, and positive `count`.
Log-shape axes may instead contain strictly increasing explicit `values`.
Candidate indices are rightmost-fastest. Independent center pairs are
enumerated by negative index and then positive index while retaining exactly
`negativeIndex < positiveIndex`; equal and reversed pairs do not exist.
Pure public forward and inverse helpers cover both strict center-pair indexing
and generic mixed-radix candidate indexing for cross-language golden vectors.

## Deterministic evaluation

The component basis follows the published symmetric radial operation order:

```text
scale = exp(logScale)
shape = exp(logShape)
difference = coordinate - center
z = difference / scale
shapeSquared = shape * shape
zSquared = z * z
uSquared = shapeSquared + zSquared
u = sqrt(uSquared)
numerator = uSquared + 2.0
rooted = sqrt(uSquared + 4.0)
denominator = u * rooted
basis = numerator / denominator
```

Per-series nuisance parameters use source-order weighted normal equations,
fixed active-set ordering, and partial-pivot Gaussian elimination. The rank
tolerance is `1e-12`; result recomputation and deterministic objective ties use
`1e-9 * max(1, abs(left), abs(right))`. Candidates compare by WRSS, BIC,
finite AICc, defined AICc before undefined AICc, and finally the smaller global
index. All counts and products remain within `(1 << 53) - 1`.

Positive fits use design columns `intercept, positiveComponent`. Doublet fits
use `intercept, negativeComponent, positiveComponent`. Active states are
`FREE`, `ZERO` for the positive model and `FREE_FREE`, `ZERO_FREE`,
`FREE_ZERO`, `ZERO_ZERO` for both doublets. Exact zero satisfies either
non-strict amplitude constraint. Gram terms, right-hand sides, predictions,
and WRSS are accumulated in source sample order; series WRSS values are added
in canonical generic-series order. Nonfinite basis, solve, fit, prediction, or
objective values invalidate the candidate.

Shards are ascending, contiguous, non-overlapping, and include a deterministic
partial final shard. Strict work payloads contain only the morphology family,
model class, grid start, and grid count. For every submitted shard, the server
evaluates every index in the server-owned range. It reconstructs the evaluated
count, exact invalid count, canonical winner, and complete winner payload. A
valid but nonwinning worker candidate, forged null winner, forged invalid
count, or incomplete evaluation count is rejected. Reduction counts coverage
only from results that pass this full-shard recomputation and requires ordered,
exact, nonduplicated shard coverage.

Completed result payloads contain exactly the morphology family, model class,
shard range, `bestCandidate`, evaluated count, and invalid count. A best
candidate contains exactly:

- `gridIndex`
- strict model-specific `parameters`
- canonical `seriesFits`
- `positiveWeightSampleCount`
- `weightedResidualSumSquares`
- `nominalParameterCount`
- `bayesianInformationCriterion`
- nullable `correctedAkaikeInformationCriterion`
- `correctedAkaikeInformationCriterionDefined`

Positive-only series fits contain exactly the generic series ID, positive-weight
sample count, offset, positive amplitude, its sign label, and series WRSS.
Doublet fits additionally contain the negative amplitude and its sign label.
The exact sign labels are `negative`, `zero`, and `positive`; exact zero remains
feasible for either non-strict sign constraint and is labeled `zero`.

With total positive-weight sample count `N`, nominal count `k`, and total WRSS,
the metrics are:

```text
BIC = WRSS + k * ln(N)
AICc = WRSS + 2*k + 2*k*(k+1)/(N-k-1)
```

Nominal counts are `2*S + 3` for `POSITIVE_PULSE_ONLY`, `3*S + 6` for
`ORDERED_NEGATIVE_POSITIVE_DOUBLET`, and `9` for the single-series
`INDEPENDENT_PULSES`; zero amplitudes never reduce `k`. When `N <= k + 1`,
AICc is exactly null and its defined flag is false. Otherwise it is finite and
the flag is true.

A shard with no valid candidate uses a null best candidate if and only if the
server recomputes every candidate as invalid. Reduction publishes the exact
total invalid count and the winning N, WRSS, k, BIC, nullable AICc, AICc-defined
state, parameters, and complete ordered series fits.
