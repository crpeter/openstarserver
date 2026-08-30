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
