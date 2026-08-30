# Curve-grid workload v1

`openstar.curve-grid.v1` searches a deterministic Cartesian grid for the
smallest weighted residual sum of squares produced by a versioned curve
family. The server package validates datasets, creates contiguous shards,
checks each reported shard winner by recomputing only that candidate, reduces
accepted shard winners, and derives contribution metrics from server-owned
inputs.

## Published identities

- Workload: `openstar.curve-grid.v1`
- Dataset: `openstar.dataset.curve-grid.v1`
- Work payload: `openstar.payload.curve-grid-shard.v1`
- Result: `openstar.result.curve-grid-shard.v1`
- Curve family: `openstar.curve-family.symmetric-radial-amplification.v1`

Workers must advertise the complete workload and schema tuple. Schema-less
capabilities are not compatible with this workload.

## Dataset

The dataset contains equal-length `coordinates`, `values`, and
`inverseVariances` arrays with at least three finite samples. Every inverse
variance is positive. `curveGrid` contains the published family ID, three
linear axes named `centerAxis`, `logScaleAxis`, and `logShapeAxis`, and a
positive `candidatesPerWorkUnit` integer. Each axis has finite `start` and
`step` numbers plus a positive integer `count`.

Axis counts and candidate counts are JSON-safe integers. Grid-size and
sample-candidate products must also be JSON-safe. Boolean values are never
accepted as numbers or integers.

The flattened index is:

```text
((centerIndex * logScaleCount) + scaleIndex) * logShapeCount + shapeIndex
```

Shape varies fastest. Shards cover that index space in ascending contiguous
ranges and contain exactly `familyID`, `gridStartIndex`, and `gridCount`.

## Operator and fit

For each coordinate, a candidate derives positive `scale` and `shape` by
exponentiating its log-axis values, calculates the published symmetric radial
amplification basis in Float64, and fits `offset + amplitude * basis` with
weighted two-parameter linear least squares. Nonfinite calculations and
numerically singular fits are invalid candidates.

Workers evaluate candidates in ascending global grid-index order. The lowest
weighted residual sum of squares wins; an exact tie selects the smaller global
grid index.

## Result and reduction

A completed result keeps all workload-owned fields inside its strict payload.
The payload identifies the assigned shard and its best grid candidate, fitted
parameters, objective, evaluated count, and invalid count. The coordinator
envelope retains identity, status, and schema ownership.

Validation checks shard identity, exact grid parameters, count invariants,
finite values, and a Float64 recomputation of only the reported winner. The
accepted result must agree within `1e-9 * max(1, abs(expected))` for offset,
amplitude, and objective.

Reduction chooses the minimum `(objective, gridIndex)` pair. It reports
`CURVE_GRID_COMPLETE` only after all expected shards have accepted results and
the server-owned shards exactly cover the grid. Every other state reports
`CURVE_GRID_INCOMPLETE`.

Contribution metrics contain workload ID, family ID, sample count, candidate
count, and their server-derived product.
