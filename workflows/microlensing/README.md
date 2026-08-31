# Microlensing archive acquisition

This package prepares the NASA Exoplanet Archive's contributed MICROLENSING
light-curve bundle for later server-owned work.

Official sources:

- [Bulk data download](https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/)
- [MICROLENSING wget script](https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/wget_MICROLENSING.bat)
- [Microlensing documentation](https://exoplanetarchive.ipac.caltech.edu/docs/microlensing.html)

## Acquire

```bash
python -m workflows.microlensing.acquire \
  --output-root /path/to/state/microlensing
```

The command preserves the official wget script, parses its entries, upgrades
listed data URLs to HTTPS, restricts downloads to the NASA Exoplanet Archive
host, and publishes every completed file atomically. It writes:

- `source/wget_MICROLENSING.bat`
- `data/*.tbl`
- `archive-manifest.json`

The manifest records source URLs, relative paths, UTC retrieval timestamps,
byte sizes, SHA-256 hashes, and the exact source-script hash used for every
file.

An ordinary rerun re-fetches the source script to detect upstream changes and
reuses archive files only when their recorded size and SHA-256 still match.
Changed, corrupt, or untracked destination files stop the run. Pass
`--refresh` to permit their atomic replacement.

## Inventory

Inventory immediately after acquisition:

```bash
python -m workflows.microlensing.acquire \
  --output-root /path/to/state/microlensing \
  --inventory
```

Inventory previously acquired files without network access:

```bash
python -m workflows.microlensing.acquire \
  --output-root /path/to/state/microlensing \
  --inventory-only
```

This writes `archive-inventory.json`. Inventory is structural: it records
UIDs, exact column names, IPAC header rows, metadata lines, row counts, schema
signatures, schema counts, and concise parse failures. It does not assume that
all contributed tables share one schema and does not assign meanings to
unknown fields. Inventory-only mode first verifies the preserved source
script's size and SHA-256, reparses its entries, and validates every manifest
record against that exact script version before reading any table.

Archive files are external state. Do not place the output root inside git or
commit downloaded data. This phase performs no model fitting, anomaly
analysis, scientific classification, or CurveGrid project creation.

## Prepare the known-event recovery pilot

After acquisition and a complete, zero-failure inventory, prepare the selected
pilot UID as identity-isolated generic weighted time series:

```bash
python -m workflows.microlensing.prepare \
  --archive-root /path/to/nasa-contributed-v1 \
  --uid 0302608 \
  --blind-target-id openstar.microlensing-recovery-a.v1 \
  --output-root /path/to/microlensing-recovery-a-prepared
```

UID `0302608` is the published known event OGLE-2012-BLG-0724L (archive
`STAR_ID` value `OGLE 2012-BLG-724L`). This is a known-event recovery
benchmark, not a blind discovery.

Preparation re-verifies archive and inventory provenance, verifies every
selected source file, parses data rows from IPAC fixed-width column spans,
selects one supported observable per source, normalizes it into linear generic
values, and applies one shared shifted time origin. The versioned preparation
contract and its SHA-256 freeze these rules. The exact supported time-column
set is `HJD` and `JD`; the supported observable pairs are relative flux with
flux uncertainty and relative magnitude with magnitude uncertainty.

The output root must not already exist. Source identities, metadata, hashes,
normalization constants, and the absolute time origin are written only to
`sealed/identity-seal.json`. Generic series and their preparation manifest are
written under `blind/`; those files contain neither original filenames nor
archive identity and provenance. Keep the sealed output under server control.

The preparation phase does not create a CurveGrid project, fit a model,
inspect expected anomaly parameters, classify a result, or make a scientific
claim.

## Build the bounded blind coarse-grid project

After blind preparation, build the first directly activatable CurveGrid
project without reading the sealed identity state:

```bash
python -m workflows.microlensing.coarse_grid \
  --prepared-root /path/to/microlensing-recovery-a-prepared \
  --project-id openstar.microlensing-recovery-a.coarse-grid.v1 \
  --output-root /path/to/microlensing-recovery-a-coarse-grid
```

The frozen grid contains exactly 4,941 candidates: 61 center values, nine
log-scale values, and nine log-shape values. With 64 candidates per work unit,
the project will produce 78 work units when later activated. This builder only
validates and writes the project; it does not activate the coordinator or
evaluate any candidate.

The initial project deliberately uses only the generic series with the largest
sample count, breaking ties by its position in `orderedSeriesIDs`. The current
CurveGrid result contract retains only each shard winner, so it cannot yet sum
every candidate objective correctly across multiple independent series.

Identity remains confined to `sealed/`, the server-side microlensing workflow
owns blind preparation and project construction, and workers receive only the
generic CurveGrid dataset and workload contract. The bounded stage fits only
the smooth single-lens-like symmetric radial-amplification curve. Completing
it is not yet recovery or classification of a planetary anomaly and is not a
discovery claim.

## Build the verified blind refinement project

After the coarse project has completed through the generic project-smoke
investigation, derive a narrower refinement grid from its persisted winner:

```bash
python -m workflows.microlensing.refine_grid \
  --prepared-root /path/to/microlensing-recovery-a-prepared \
  --coarse-project-root /path/to/microlensing-recovery-a-coarse-grid \
  --coarse-investigation-record \
    /path/to/investigations/coarse-run/investigation.json \
  --project-id openstar.microlensing-recovery-a.refinement-grid.v1 \
  --output-root /path/to/microlensing-recovery-a-refinement-grid
```

The builder re-verifies the blind preparation, every coarse project artifact,
the completed smoke-investigation record, and the exact immutable JSON ledger
for every completed stage. It binds the new contract to the verified coarse
run-stage ledger hash. It never reads `sealed/` or trusts terminal console
output.

The refinement axes are derived mechanically from the accepted coarse winner.
The center range spans one coarse center step on either side with one-tenth
steps and 21 points. The log-scale and log-shape ranges each span the adjacent
coarse cells with one-eighth steps and 17 points. This produces exactly 6,069
candidates and 95 expected work units at 64 candidates per work unit.

The provenance chain is therefore blind preparation → coarse contract and
dataset → immutable coarse investigation ledgers → refinement contract and
dataset. Identity remains sealed, while generic workers see only the ordinary
CurveGrid payloads. This remains smooth-event modeling with the symmetric
radial-amplification family; it is not yet planetary-anomaly recovery,
classification, or a discovery claim.

## Recenter a verified boundary refinement

When the completed first refinement has an accepted winner on at least one
axis boundary, build a second grid centered mechanically on that winner:

```bash
python -m workflows.microlensing.recenter_grid \
  --prepared-root /path/to/microlensing-recovery-a-prepared \
  --coarse-project-root /path/to/microlensing-recovery-a-coarse-grid \
  --coarse-investigation-record \
    /path/to/investigations/coarse-run/investigation.json \
  --refinement-project-root \
    /path/to/microlensing-recovery-a-refinement-grid \
  --refinement-investigation-record \
    /path/to/investigations/refinement-run/investigation.json \
  --project-id openstar.microlensing-recovery-a.recentered-grid.v1 \
  --output-root /path/to/microlensing-recovery-a-recentered-grid
```

The builder verifies the blind preparation and selected generic series; every
coarse contract, dataset, project, build artifact, and immutable investigation
ledger; every first-refinement artifact and recorded hash; and the completed
three-stage first-refinement project-smoke investigation. It requires exact
project and schema identities, stage causality, parameter and result hashes,
node-contribution accounting, 95 completed and zero failed work units, and
complete coverage of all 6,069 first-refinement candidates. The accepted
winner must map exactly from its flattened grid index, agree with the dataset
status, and lie on at least one first-refinement axis boundary. An interior
winner makes recentering unjustified and is rejected.

Every new axis retains its corresponding first-refinement step. The center
axis has 21 points beginning ten center steps below the accepted center. The
log-scale and log-shape axes each have 17 points beginning eight corresponding
steps below their accepted values. The verified winner is therefore at new
indices 10, 8, and 8. The resulting grid again contains exactly 6,069
candidates and produces 95 work units at 64 candidates per work unit.

The recentered contract and build manifest bind the preparation-manifest hash,
all coarse artifact and investigation hashes, all first-refinement artifact
and investigation hashes, every immutable first-refinement stage-ledger hash,
the accepted run-stage ledger, the full accepted winner, and the frozen
recentering derivation. The builder does not read `sealed/`, trust terminal
output, or consult original identity or published event parameters.

This is still known-event recovery and smooth-event convergence. It does not
recover or classify a planetary anomaly and makes no discovery claim.

## Build the verified blind second recenter

After the first recentered grid has completed through a separate generic
project-smoke investigation, build another ordinary CurveGrid project from
the complete immutable chain:

```bash
python -m workflows.microlensing.second_recenter_grid \
  --prepared-root /path/to/microlensing-recovery-a-prepared \
  --coarse-project-root /path/to/microlensing-recovery-a-coarse-grid \
  --coarse-investigation-record \
    /path/to/investigations/coarse-run/investigation.json \
  --refinement-project-root \
    /path/to/microlensing-recovery-a-refinement-grid \
  --refinement-investigation-record \
    /path/to/investigations/refinement-run/investigation.json \
  --first-recenter-project-root \
    /path/to/microlensing-recovery-a-recentered-grid \
  --first-recenter-investigation-record \
    /path/to/investigations/first-recenter-run/investigation.json \
  --project-id openstar.microlensing-recovery-a.second-recentered-grid.v1 \
  --output-root /path/to/microlensing-recovery-a-second-recentered-grid
```

The required parents are the verified blind preparation, coarse project and
completed investigation, first-refinement project and completed
investigation, and first-recenter project and completed investigation. The
builder reconstructs each parent artifact from its verified ancestry, checks
every immutable stage ledger, and requires the first-recenter project-smoke
result to have zero failures, all 95 work units complete, and complete
coverage of all 6,069 candidates. Execution of the produced project remains a
separate project-smoke step; the builder never activates a coordinator or
executes work.

For each axis, the second recenter retains the parent count and step and uses
the exact rule `newStart = winner - ((count - 1) / 2) * step`. Thus the
accepted first-recenter winner becomes index 10 on the 21-point center axis
and index 8 on each 17-point logarithmic axis, whether that parent winner was
interior or on a boundary. The grid remains 6,069 candidates at 64 candidates
per work unit, for 95 expected work units.

The output root is published transactionally and contains:

- `second-recentered-search-contract.json`
- `datasets/primary-series.json`
- `project.json`
- `build-manifest.json`

The contract and manifest bind every parent project and investigation ID,
parent artifact and ledger hashes, the accepted first-recenter winner, parent
and derived axes, sample and work accounting, and output hashes. The builder
does not read `sealed/` or consult source filenames, event names, catalog
identifiers, publications, sky coordinates, or published physical
parameters. This remains a blind known-event benchmark and smooth-event
convergence phase, not planetary-anomaly recovery, classification, or a
discovery claim.

## Prepare residuals after verified smooth-model convergence

After the second recentered grid has completed through its own generic
project-smoke investigation, freeze the converged nonlinear geometry and
prepare identity-free residuals for every generic series:

```bash
python -m workflows.microlensing.prepare_residuals \
  --prepared-root /path/to/microlensing-recovery-a-prepared \
  --coarse-project-root /path/to/microlensing-recovery-a-coarse-grid \
  --coarse-investigation-record \
    /path/to/investigations/coarse-run/investigation.json \
  --refinement-project-root \
    /path/to/microlensing-recovery-a-refinement-grid \
  --refinement-investigation-record \
    /path/to/investigations/refinement-run/investigation.json \
  --first-recenter-project-root \
    /path/to/microlensing-recovery-a-recentered-grid \
  --first-recenter-investigation-record \
    /path/to/investigations/first-recenter-run/investigation.json \
  --second-recenter-project-root \
    /path/to/microlensing-recovery-a-second-recentered-grid \
  --second-recenter-investigation-record \
    /path/to/investigations/second-recenter-run/investigation.json \
  --output-root /path/to/microlensing-recovery-a-residuals
```

The required immutable ancestry is blind preparation → coarse project and
completed investigation → first-refinement project and completed
investigation → first-recenter project and completed investigation →
second-recenter project and completed investigation. The builder reconstructs
the deterministic second-recenter project expected from the verified
first-recenter winner instead of trusting its manifest, and re-verifies all
project artifacts, hashes, stage ledgers, stage order and causality, project
identities, zero-failure work accounting, all 95 completed work units, and
coverage of all 6,069 candidates.

Residual publication requires exact convergence. The verified second-recenter
winner must be interior on the center, log-scale, and log-shape axes, and its
`bestCenter`, `bestLogScale`, `bestLogShape`, and
`bestWeightedResidualSumSquares` must exactly equal the verified accepted
first-recenter values. A boundary winner or any change in geometry or objective
is rejected.

The shared frozen geometry is `center = bestCenter`,
`scale = exp(bestLogScale)`, and `shape = exp(bestLogShape)`. For every
prepared generic series in canonical `orderedSeriesIDs` order, the builder
evaluates the exact
`openstar.curve-family.symmetric-radial-amplification.v1` basis and fits only
an unconstrained offset and amplitude by deterministic weighted linear least
squares. It preserves coordinates and inverse variances, writes model values
and `observed - model` residuals, and records fit diagnostics and provenance
hashes independently for every series; it does not select only the strongest
series.

The previously nonexistent output root is published atomically with this
layout:

- `residual-preparation-contract.json`
- `residual-manifest.json`
- `series/residual-series-001.json`
- one sequential residual-series file for every prepared generic series

The builder never reads `sealed/`, archive sources, source filenames, event
names, catalog identifiers, publications, sky coordinates, or published event
parameters. This phase only prepares residuals after deterministic convergence
of the smooth symmetric model. Searching those residuals for an anomaly is a
later, separate phase; residual preparation does not detect, classify, or
claim a planetary anomaly.

## Build the blind localized residual grid

After residual preparation, build one ordinary multi-dataset CurveGrid
project for generic localized residual modeling:

```bash
python -m workflows.microlensing.residual_grid \
  --prepared-root /path/to/microlensing-recovery-a-prepared \
  --coarse-project-root /path/to/microlensing-recovery-a-coarse-grid \
  --coarse-investigation-record \
    /path/to/investigations/coarse-run/investigation.json \
  --refinement-project-root \
    /path/to/microlensing-recovery-a-refinement-grid \
  --refinement-investigation-record \
    /path/to/investigations/refinement-run/investigation.json \
  --first-recenter-project-root \
    /path/to/microlensing-recovery-a-recentered-grid \
  --first-recenter-investigation-record \
    /path/to/investigations/first-recenter-run/investigation.json \
  --second-recenter-project-root \
    /path/to/microlensing-recovery-a-second-recentered-grid \
  --second-recenter-investigation-record \
    /path/to/investigations/second-recenter-run/investigation.json \
  --residual-root /path/to/microlensing-recovery-a-residuals \
  --project-id openstar.microlensing-recovery-a.residual-grid.v1 \
  --output-root /path/to/microlensing-recovery-a-residual-grid
```

The builder reconstructs the complete immutable ancestry from blind
preparation through both recenter investigations, then regenerates residual
preparation in a private temporary root and requires the supplied residual
contract, manifest, and every ordered residual-series artifact to match that
deterministic reconstruction byte for byte. This re-verifies sample counts,
input and output hashes, frozen geometry, nuisance fits, model and residual
values, weights, per-series WRSS, the manifest total, and all parent artifact
and stage-ledger provenance. The same convergence gate applies: the verified
second-recenter winner must be interior on all three axes, and its center,
log scale, log shape, and WRSS must exactly equal the verified first-recenter
winner.

The search geometry is derived without inspecting residual values. With
`coreWidth = exp(frozenLogScale) * exp(frozenLogShape)`, every admitted series
uses the same `openstar.curve-family.symmetric-radial-amplification.v1` grid:

- center: 129 values beginning at `eventCenter - 4 * coreWidth`, with step
  `coreWidth / 16`; the event center is index 64 and the final value is
  `eventCenter + 4 * coreWidth`
- log scale: 17 values beginning at `frozenLogScale - log(16)`, with step
  `log(16) / 16`
- log shape: one value at `frozenLogShape`, with a positive finite schema step
  that cannot change that sole value
- 64 candidates per work unit, giving 2,193 candidates and 35 expected work
  units per admitted dataset

Admission also cannot inspect residual values. For every generic series, it
counts positive-weight samples in the inclusive interval
`eventCenter ± 4 * coreWidth` and admits the complete series only when that
count is at least eight. Every admission and exclusion is recorded. An
admitted dataset retains all coordinates, signed residual values, and inverse
variances; it is not cropped to the admission window or to a residual peak.
Totals are derived from the actual admissions rather than from expected
series IDs or an expected dataset count.

The previously nonexistent output root is published atomically with this
layout:

- `residual-search-contract.json`
- `datasets/residual-series-001.json` and one sequential file per admitted
  series
- `project.json`
- `build-manifest.json`

Project execution is a separate operation. Run the generic project-smoke
workflow only when the coordinator and workers are intentionally available:

```bash
python run_investigation.py \
  --project /path/to/microlensing-recovery-a-residual-grid/project.json \
  --investigation-id generic-residual-grid-investigation \
  --store /path/to/investigations
```

The builder never reads `sealed/`, archive sources, source filenames, event
names, catalog identifiers, publications, identity coordinates, or published
physical parameters. Its window and admissions use only verified frozen
geometry, coordinates, weights, and fixed coverage rules. This is generic
residual localization, not planetary classification or a discovery claim;
anomaly interpretation is a later, separate phase.

## Validate localized residual structure across series

After the residual-grid project has completed through the generic
project-smoke workflow, validate every accepted localized component against
all other admitted generic series:

```bash
python -m workflows.microlensing.validate_residual_grid \
  --residual-root /path/to/microlensing-recovery-a-residuals \
  --residual-grid-root /path/to/microlensing-recovery-a-residual-grid \
  --residual-grid-investigation-record \
    /path/to/investigations/generic-residual-grid-investigation/investigation.json \
  --output-root /path/to/microlensing-recovery-a-cross-validation
```

The analyzer treats all paths as untrusted and reconstructs the residual
preparation artifacts, residual-grid contract, build manifest, project, and
every admitted dataset. It then verifies the complete three-stage
`openstar.workflow.project-smoke.v1` investigation, immutable stage ledgers,
project identity and hash, zero-failure work accounting, complete candidate
coverage for each admitted dataset, agreement between aggregate and nested
winners, and exact winner-to-grid mapping. For the three-series real project,
this requires all 105 work units to be complete. Each accepted winner is also
re-evaluated with the canonical CurveGrid
`openstar.curve-family.symmetric-radial-amplification.v1` implementation and
must agree within the workload's established deterministic floating-point
tolerance.

Validation order is fixed by the admitted generic series order. For each
discovery series, the analyzer freezes the accepted center, log scale, log
shape, and amplitude sign. On every other admitted series, it evaluates that
same geometry and fits only an offset and unconstrained signed amplitude by
deterministic weighted least squares. It records the offset-only null WRSS,
frozen-template WRSS, their difference, the fitted amplitude and sign, and
positive-weight support within one and two frozen effective widths. A
discovery series is never allowed to validate itself, and every ordered
discovery-to-validation pair is retained, including insufficient-support and
fit-failure results.

The decision thresholds are predeclared and are not CLI options:

- the discovery must have complete grid coverage and `deltaWRSS >= 30.0`
- a held-out series must have at least one positive-weight sample within two
  frozen effective widths, the same nonzero amplitude sign, and
  `deltaWRSS >= 9.0`
- a searched-axis boundary is reported and limits width interpretation, but
  does not automatically reject the discovery
- a component is `CROSS_SERIES_CONFIRMED` only when the discovery gate passes
  and at least one distinct held-out series passes; otherwise it is
  `NOT_CROSS_SERIES_CONFIRMED`
- the overall result is `REPRODUCIBLE_LOCALIZED_RESIDUAL_STRUCTURE` when at
  least one component is confirmed, and
  `NO_REPRODUCIBLE_LOCALIZED_RESIDUAL_STRUCTURE` otherwise

The previously nonexistent output root is published atomically with:

- `residual-cross-validation-contract.json`
- `residual-cross-validation.json`

The result always leaves `planetaryInterpretationResolved` and
`discoveryClaim` false. Confirmed reproducible structure recommends
`BLIND_MICROLENSING_ANOMALY_MORPHOLOGY_MODELING`; a negative result recommends
`RESIDUAL_SYSTEMATICS_AND_ERROR_MODEL_REVIEW`. The former is a later blind
morphology test and the latter is a review of residual systematics and the
error model. Neither next test is a planetary classification or discovery
claim. The analyzer never reads sealed identity, archive sources, source
filenames, event names, catalog identifiers, publications, sky coordinates,
or published physical parameters.

## Prepare blind anomaly-morphology evidence

After cross-series validation reports exactly one
`CROSS_SERIES_CONFIRMED` positive component and recommends
`BLIND_MICROLENSING_ANOMALY_MORPHOLOGY_MODELING`, prepare the identity-free
evidence and the predeclared contract for the later distributed morphology
grid:

```bash
python -m workflows.microlensing.prepare_anomaly_morphology \
  --residual-root /path/to/microlensing-recovery-a-residuals \
  --residual-grid-root /path/to/microlensing-recovery-a-residual-grid \
  --residual-grid-investigation-record \
    /path/to/investigations/generic-residual-grid-investigation/investigation.json \
  --cross-validation-root \
    /path/to/microlensing-recovery-a-cross-validation \
  --output-root /path/to/microlensing-recovery-a-morphology-preparation
```

The preparer does not trust summary fields. It reconstructs the residual
preparation, residual-grid project and datasets, completed three-stage
project-smoke investigation, immutable ledgers, multi-dataset counter scopes,
complete per-dataset and project coverage, and accepted winners. It then
recomputes the complete cross-validation result from those verified parents
and requires the supplied contract and result to match exactly, including
their IDs and hashes. The required result is
`REPRODUCIBLE_LOCALIZED_RESIDUAL_STRUCTURE` with exactly one confirmed
component, at least one distinct passing held-out series, unresolved planetary
interpretation, no discovery claim, and the morphology-modeling next test.

Series selection is generic and deterministic: include the confirmed positive
discovery series and every distinct held-out series that passed its frozen
validation gate. The closest earlier, discovery-gated negative winner is the
preceding component anchor. No series ID or coordinate is hardcoded. The
shared coordinate window brackets two effective widths around both anchors,
then expands until every selected series contributes at least three
positive-weight baseline samples on each side. Every selected series must
also have positive-weight support within two effective widths of both
components. Complete source indices, coordinates, signed residuals, inverse
variances, inclusion reasons, and source hashes are preserved.

The contract predeclares three generic residual model classes without
evaluating them:

- `POSITIVE_PULSE_ONLY` fits one positive component with shared nonlinear
  geometry and per-series offset and amplitude.
- `ORDERED_NEGATIVE_POSITIVE_DOUBLET` requires a negative component followed
  by a positive component, shared nonlinear timing and shape, and per-series
  offsets and signed component amplitudes.
- `INDEPENDENT_PULSES` allows each series independent ordered negative and
  positive centers as the less-constrained comparison while retaining shared
  width and shape axes.

Axes, rightmost-fastest mixed-radix candidate ordering, global candidate
offsets, invalid-candidate behavior, finite-value rules, parameter counts,
WRSS, BIC, and AICc are fixed in the contract. Widths are strictly positive.
The width grid extends at least sixteen-fold below the earlier minimum-width
boundary and is capped at one quarter of the prepared coordinate span.
Separation is strictly positive and bounded by the prepared center axis.

The ordered doublet can be preferred over the positive pulse only with global
delta WRSS at least 30, per-series delta WRSS at least 9, delta BIC at least
10, and all cross-series timing and sign requirements. Independent pulses can
reject the ordered doublet only with delta WRSS at least 18, delta BIC at
least 10, valid signs, and center dispersion beyond the predeclared timing
tolerance. Any winning width boundary remains unresolved and cannot be
reported as a measured duration.

The previously nonexistent output root is published atomically with:

- `anomaly-morphology-preparation.json`
- `morphology-contract.json`
- `datasets/morphology-series-001.json` and one sequential file per selected
  generic series
- `artifact-manifest.json`

This step only prepares blind numerical evidence and a later model contract.
It never reads sealed identity, raw archives, source filenames, event names,
catalog identifiers, citations, observatory identity, sky coordinates, or
published physical parameters. It does not run the morphology grid, implement
a binary-lens model, resolve a planetary interpretation, or make a discovery
claim. The next separate step is
`DISTRIBUTED_BLIND_MICROLENSING_ANOMALY_MORPHOLOGY_GRID`.

