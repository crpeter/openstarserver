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
`1e-9 * max(1, abs(left), abs(right))`. Candidate ties select the smaller
global index. All counts and products remain within `(1 << 53) - 1`.

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
model class, grid start, and grid count. The server recomputes the reported
winner, reduces accepted shard winners, and derives accounting solely from the
validated dataset and work payload.

Completed result payloads contain exactly the morphology family, model class,
shard range, `bestCandidate`, evaluated count, and invalid count. A best
candidate contains exactly its global index, strict model-specific parameters,
canonical per-series nuisance fits, and total WRSS. A shard with no valid
candidate uses a null best candidate, marks every candidate invalid, and is
accepted only after the server recomputes that entire exceptional shard.
