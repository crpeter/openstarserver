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
