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
- `INDEPENDENT_PULSES` gives every admitted series its own independently
  indexed six-axis search: negative center, positive center, and separate
  log-scale and log-shape values for both components. Negative center must
  strictly precede positive center. Per-series search ranges are concatenated
  in canonical series order; no Cartesian product is formed between series,
  so candidate growth is linear in admitted-series count.

Axes, rightmost-fastest mixed-radix candidate ordering, safe-integer class and
per-series offsets, invalid-candidate behavior, finite-value rules, nominal
parameter counts, WRSS, BIC, and AICc are fixed in the contract. Independent
per-series winners are aggregated in canonical series order into global WRSS,
AICc, BIC, timing dispersion, and the predeclared comparison decision. Widths
are strictly positive.
The width grid extends at least sixteen-fold below the earlier minimum-width
boundary and is capped at one quarter of the prepared coordinate span.
Separation is strictly positive and bounded by the prepared center axis.

The contract distinguishes the canonical single-component template family
`openstar.curve-family.symmetric-radial-amplification.v1` from the compound
morphology family `openstar.microlensing-residual-morphology.v1`. It freezes
the exact symmetric-radial basis equation and operation order, IEEE-754
binary64 arithmetic, design-matrix column and sample ordering, zero- and
negative-weight handling, amplitude sign constraints, active-set enumeration,
partial-pivot Gaussian elimination and rank threshold, source-order WRSS
accumulation, nominal parameter counts even for zero amplitudes, undefined
AICc behavior, relative objective tolerance, exact tie ordering, invalid
results, and independent-series aggregation. These rules are intended to make
later Python and Swift implementations produce the same deterministic result;
this preparer does not execute a fit.

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

## Build the bounded blind morphology coarse grid

Convert a verified anomaly-morphology preparation into an ordinary
`openstar.morphology-grid.v1` project without evaluating the approximately
65-million-candidate full grid:

```bash
python -m workflows.microlensing.build_anomaly_morphology_coarse_grid \
  --morphology-root \
    /path/to/microlensing-recovery-a-morphology-preparation \
  --project-id generic-morphology-coarse \
  --output-root /path/to/microlensing-recovery-a-morphology-coarse
```

The input root must be the complete, immutable output of
`prepare_anomaly_morphology.py`: its preparation result, morphology contract,
artifact manifest, and both ordered generic-series files. The builder verifies
their schemas, canonical bytes, hashes, series ordering, source sample arrays,
full candidate mapping, family identities, deterministic execution semantics,
parent provenance, identity-isolation statement, and recommended next test.
Symlinks, path traversal, unexpected files, nonfinite values, negative weights,
and an existing output root are rejected.

The per-search candidate ceiling defaults to 8,192 and can be lowered with
`--maximum-candidates-per-search`. For each search independently, one integer
stride starts at one and increases by one until the first admissible grid is at
or below the frozen limit. A linear axis retains its start, multiplies its step
by the stride, and uses `floor((sourceCount - 1) / stride) + 1` values. An
explicit axis retains source indices `0, stride, 2*stride, ...`. No irregular
endpoint is appended. Independent-pulse searches stride their center axis
before counting strict negative-center-before-positive-center pairs and must
retain at least two centers. Thus every coarse value maps to an original full
axis index; no sample value, residual magnitude, candidate fit, or identity is
consulted when choosing a stride.

The project always contains four searches in this order:

1. one shared `POSITIVE_PULSE_ONLY` dataset containing both generic series;
2. one shared `ORDERED_NEGATIVE_POSITIVE_DOUBLET` dataset containing both;
3. one single-series `INDEPENDENT_PULSES` dataset for the first series; and
4. one single-series `INDEPENDENT_PULSES` dataset for the second series.

The independent searches are separate workload datasets, never a cross-series
Cartesian product. Every dataset preserves complete source coordinates, signed
residual values, inverse variances, generic series order, and identity-free
hash provenance. It uses 64 candidates per work unit, with candidate,
work-unit, and sample-candidate evaluation totals derived from the emitted
axes.

The previously nonexistent output root is published atomically with:

- `coarse-grid-contract.json`
- `datasets/positive-pulse-only.json`
- `datasets/ordered-negative-positive-doublet.json`
- `datasets/independent-pulses-series-001.json`
- `datasets/independent-pulses-series-002.json`
- `project.json`
- `build-manifest.json`

The builder does not evaluate a basis, fit nuisance parameters, rank a
candidate, classify morphology, or interpret the result. This bounded project
remains a blind generic morphology comparison and makes no discovery claim.
Only after a completed distributed run should a separate verifier perform
fitting-result validation and interpretation. Run that separate investigation
through the ordinary project runner:

```bash
python run_investigation.py \
  --project /path/to/microlensing-recovery-a-morphology-coarse/project.json \
  --investigation-id generic-morphology-coarse-investigation \
  --coordinator http://127.0.0.1:8080 \
  --store /path/to/investigations
```

## Verify the completed bounded morphology coarse investigation

After the four-search investigation above is complete, build a deterministic
report in a **new** output directory. This command reads the saved artifacts
and reproduces only their four accepted winners; it does not run a grid search,
contact the coordinator, choose a replacement winner, or execute a follow-up.

```bash
python -m workflows.microlensing.validate_anomaly_morphology_coarse_grid \
  --morphology-root /path/to/microlensing-recovery-a-morphology \
  --coarse-project-root /path/to/microlensing-recovery-a-morphology-coarse \
  --coarse-investigation-record /path/to/investigations/generic-morphology-coarse-investigation/investigation.json \
  --output-root /path/to/microlensing-recovery-a-morphology-coarse-validation
```

The verifier checks preparation and coarse contracts, source arrays and hashes,
recursive typed ancestry (including `coarse.contractID` metadata), exact producer
reconstruction, investigation identity and terminal state, immutable stage
ledgers, and all four datasets' coverage and contribution counts. Project totals
use `projectCompletedWorkUnits` and `projectTotalWorkUnits`; unprefixed run
counters must agree with the final dataset. Every accepted grid index must map
to its recorded geometry, reproduce under the existing numerical tolerances,
and agree with the duplicated winner summary fields.

Publication creates two versioned JSON files:

- `anomaly-morphology-coarse-validation.json`: numerical results, support,
  boundaries, comparison gates, classification, and follow-up recommendation.
- `artifact-manifest.json`: result hash, input artifact and immutable ledger
  hashes, and inherited typed ancestry. Inputs are preserved; existing output
  directories, symlinks, and outputs nested inside input roots are rejected.

Read the actual report fields with:

```bash
python - /path/to/microlensing-recovery-a-morphology-coarse-validation <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = json.loads((root / "anomaly-morphology-coarse-validation.json").read_text())
print("classification:", result["overallClassification"])
print("model preference:", result["modelPreference"])
print("unsupported searches:", result["unsupportedSearchCount"])
print("project work units:", result["projectCompletedWorkUnits"], "/", result["projectTotalWorkUnits"])
print("planetary interpretation resolved:", result["planetaryInterpretationResolved"])
print("discovery claim:", result["discoveryClaim"])
for search in result["searches"]:
    print("\n", search["datasetID"], "accepted index:", search["acceptedWinner"]["gridIndex"])
    print("searched boundaries:", search["searchedBoundaryAxes"], "fixed axes:", search["fixedAxes"])
    for component in search["components"]:
        print(component["component"], "center:", component["center"], "effective width:", component["effectiveWidth"])
        for support in component["seriesSupport"]:
            print(support["genericSeriesID"],
                  "within two widths:", support["positiveWeightSamplesWithinTwoEffectiveWidths"],
                  "nearest distance / width:", support["nearestPositiveWeightDistanceInEffectiveWidths"],
                  "supported:", support["supportRequirementMet"])
print("\nIndependent aggregate:")
print(json.dumps(result["independentAggregate"], indent=2))
print("\nModel comparisons:")
print(json.dumps(result["modelComparisons"], indent=2))
print("\nRecommended next test:", result["recommendedNextTest"])
print(result["recommendation"])
PY
```

Support uses `effectiveWidth = exp(logScale) * exp(logShape)` in that order.
Only positive-weight observations count, and the two-width threshold is
inclusive. For an ordered doublet, the positive center is derived as
`negativeCenter + separation`. Each component records support separately for
every applicable series. An axis with one value is `FIXED`, never a searched
boundary. Searched width boundaries remain unresolved and do not establish a
measured duration.

`independentAggregate` adds the accepted per-series WRSS and positive-weight N
in canonical series order, retains nominal k even for zero amplitudes, then
computes global BIC and AICc once. It never sums per-series information criteria;
AICc is null with `correctedAkaikeInformationCriterionDefined = false` when
`N <= k + 1`. `modelComparisons` retains global and per-series improvements,
signs, exact frozen threshold gates, and support gates. Center dispersions and
the timing-consistency gate are in `independentAggregate`.

Insufficient support is a valid completed report. If any accepted search has an
unsupported component, `overallClassification` is
`UNRESOLVED_OBSERVATIONAL_SUPPORT`, `modelPreference` is null, and
`modelPreferenceResolved` is false. When all four accepted searches have this
problem, `unsupportedSearchCount` is 4. Numerical comparisons remain available.
The recommendation is `SUPPORT_AWARE_BLIND_ANOMALY_MORPHOLOGY_SEARCH`, which
requires a separately predeclared follow-up accounting for positive-weight
observations in every applicable series. This verifier neither creates nor
executes it, and it does not alter workload validity rules.

The saved winners do **not** establish that no supported candidate exists in
the grid; no unverified runner-up is promoted. Even when support and comparison
gates pass, these remain generic morphology results with
`planetaryInterpretationResolved = false` and `discoveryClaim = false`.

The repository owner can run the focused suite locally:

```bash
python -m unittest tests.workflows.microlensing.test_validate_anomaly_morphology_coarse_grid
```

## Build the supported morphology grid on the unchanged coarse domain

The separately identified `openstar.supported-morphology-grid.v1` workload
selects the best numerically valid candidate with a positive-weight observation
within two effective widths of every component in every applicable series.
Its [frozen wire contract and portable examples](../../openstar_workloads/plugins/supported_morphology_grid/README.md)
are shared with the separately developed Apple implementation.

Use the real PR171 preparation, PR177 bounded coarse project, completed v3
investigation record, and PR186 validation report **and** artifact manifest:

```bash
python -m workflows.microlensing.build_supported_anomaly_morphology_grid \
  --morphology-root /path/to/microlensing-recovery-a-morphology-preparation \
  --coarse-project-root /path/to/microlensing-recovery-a-morphology-coarse \
  --coarse-investigation-record /path/to/investigations/generic-morphology-coarse-v3/investigation.json \
  --coarse-validation-root /path/to/microlensing-recovery-a-morphology-coarse-validation \
  --project-id generic-supported-morphology \
  --output-root /path/to/microlensing-recovery-a-supported-morphology
```

Replace the paths with existing verified artifacts and choose a new output
directory. The builder reuses the preparation, typed ancestry, coarse artifact,
investigation ledger, exact coverage and accepted-winner verifiers. It
reconstructs the entire PR186 report and manifest, including hashes, support and
metric fields; a classification alone is insufficient. Only the four saved
winners are reproduced during source verification. No grid search, coordinator,
resampling, normalization, refinement, widening or winner preselection occurs.

All four original datasets retain their numerical arrays, sample-index
provenance, axes, strict center-pair and mixed-radix indexing,
`candidatesPerWorkUnit`, candidate counts and work-unit counts. Thus the real
441-work-unit source remains a 441-work-unit project; the builder copies verified
counts and does not hard-code 441 for miniature or other valid sources.

Only project/dataset IDs, workload/schema/execution IDs and the required
`supportPolicyID` change, with explicit provenance added for preparation,
coarse artifacts, investigation/ledgers and validation artifacts. Old results
are not republished under new identities. `project.json`, four dataset files,
and a versioned `build-manifest.json` publish atomically. The manifest records
output file SHA-256 values and `buildManifestSHA256`, defined over its canonical
compact JSON body with that one digest field omitted. Existing output roots,
symlinks and output paths inside input directories are rejected; inputs remain
unchanged.

A processed shard with no eligible candidates succeeds with a null winner.
Complete grid coverage with no eligible candidates is COMPLETE with null
winner summaries. Both rejection counts remain visible, and the result only
establishes support and ranking for this searched grid.

Run these focused tests locally:

```bash
python -m unittest discover -s tests/workloads/supported_morphology_grid -p 'test_*.py' -v
python -m unittest tests.workflows.microlensing.test_build_supported_anomaly_morphology_grid -v
```

Tests, builds, downloads, servers, coordinators and workloads were not run as
part of this implementation.

## Validate the completed supported morphology investigation

The offline validator verifies the original PR171 preparation, PR177 coarse
project and completed investigation, PR186 validation report, and the complete
PR187 supported project. It reconstructs the supported project, all four
datasets and the hashed build manifest against the verified coarse domain,
including numerical arrays, axes, indexing, shard sizes and provenance. It then
checks the actual supported investigation identities, canonical stage order,
immutable ledgers, terminal result, dataset mapping and complete accounting.
It uses project-prefixed counters for project coverage and checks every dataset
and the sums; unprefixed run counters describe the final dataset.

Run the focused test locally from the repository root:

```bash
python -m unittest tests.workflows.microlensing.test_validate_supported_anomaly_morphology_grid -v
```

Complete command for the existing local artifacts (the output must be new):

```bash
python -m workflows.microlensing.validate_supported_anomaly_morphology_grid \
  --morphology-root /Users/petercody/Documents/OpenStarScience/microlensing/recovery-a-anomaly-morphology-pr171-v1 \
  --coarse-project-root /Users/petercody/Documents/OpenStarScience/microlensing/recovery-a-anomaly-morphology-coarse-pr177-v1 \
  --coarse-investigation-record /Users/petercody/Documents/OpenStarScience/microlensing/recovery-a-anomaly-morphology-coarse-run-v3/microlensing-recovery-a-anomaly-morphology-coarse-v3/investigation.json \
  --coarse-validation-root /Users/petercody/Documents/OpenStarScience/microlensing/recovery-a-anomaly-morphology-validation-pr186-v1 \
  --supported-project-root /Users/petercody/Documents/OpenStarScience/microlensing/recovery-a-supported-morphology-pr187-v1 \
  --supported-investigation-record /Users/petercody/Documents/OpenStarScience/microlensing/recovery-a-supported-morphology-run-v1/microlensing-recovery-a-supported-morphology-v1/investigation.json \
  --output-root /Users/petercody/Documents/OpenStarScience/microlensing/recovery-a-supported-morphology-validation-v1
```

Three files publish atomically:

- `supported-anomaly-morphology-validation.json`: versioned report, input hashes,
  provenance, all four searches, candidate accounting, support counts and nearest
  distances, amplitude signs, axis indices/boundaries, numerical ranking,
  independent joint statistics, every interpretation gate and its inputs and
  thresholds, classification, limitations and deterministic next test.
- `supported-anomaly-morphology-validation.md`: short readable summary.
- `artifact-manifest.json`: versioned manifest with input and output hashes.

Each non-null saved supported winner is reproduced numerically through the
supported workload's identity-checking adapter. Its exact index/parameter mapping,
fits and information criteria must agree under the published tolerances, and
its support is independently checked using separate exponentials, inclusive
2-effective-width boundaries and positive weights. Zero fitted amplitudes still
require geometric support. Unsupported accepted winners fail validation.
Complete searches with zero eligible candidates and null winners remain valid
completed searches with unresolved interpretation.

The report reuses the frozen PR186 interpretation rules. Strictly negative
components cannot be replaced by zero amplitudes, and combined improvement
cannot hide a series that fails the per-series improvement gate. Independent
joint BIC/AICc are computed once from summed WRSS, summed nominal parameter
counts and combined positive-weight sample count, not by summing separate
BIC/AICc values. Numerical ranking remains separate from gated preference.

This is an accepted-winner and accounting-consistency audit. It reproduces only
the four original saved winners plus each non-null supported winner. It does not
enumerate the grid, recompute shards, independently reproduce every rejection
count, or independently prove global optimality. Eligibility is not measured
duration, convergence, cross-series replication or planetary evidence. Searched
boundary warnings remain; fixed axes are not searched boundaries. The report
always retains `planetaryInterpretationResolved: false` and
`discoveryClaim: false`.

Canonical JSON, exact artifact sets, hashes, stage ledgers and path/symlink
protections fail closed. Existing output directories are rejected and inputs
remain unchanged. Tests and real-artifact execution were not run during
implementation; the Mac artifacts were unavailable in the implementation
environment. The reported 441/441 work units and 28,078 / 0 / 23,251 / 4,827
candidate counts are reference observations, not implementation constants.
