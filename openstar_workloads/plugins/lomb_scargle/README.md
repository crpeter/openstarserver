# Lomb-Scargle compatibility lane

This lane implements `openstar.lomb-scargle.v1` and the historical
`openstar.tess-period-search.v1` alias. Both use the existing frequency-grid,
payload, canonical result, structural validation, and legacy flattened fields.

- Dataset schema: `openstar.dataset.lomb-scargle.v1`
- Payload schema: `openstar.payload.lomb-scargle-shard.v1`
- Result schema: `openstar.result.lomb-scargle-shard.v1`

For strict ledger compatibility, only the canonical Lomb ID emits the existing
sample/frequency accounting dimensions. The historical alias retains its
existing workload-ID-only accounting record.
