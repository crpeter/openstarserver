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

This phase does not create a CurveGrid project, fit a model, inspect expected
anomaly parameters, classify a result, or make a scientific claim.
