# Periodic-box compatibility lane

This lane preserves the server's existing frequency-window periodic-box
workload. Each configured frequency window becomes exactly one generic worker
payload. Accepted results must prove that frequency, phase, duration, indexes,
and sample counts agree with the server-owned payload and dataset.

- Workload: `openstar.box-period-search.v1`
- Dataset schema: `openstar.dataset.box-period-search.v1`
- Payload schema: `openstar.payload.box-period-shard.v1`
- Result schema: `openstar.result.box-period-shard.v1`

Reduction retains every accepted window winner in deterministic window/index
order and publishes the existing box-search status surface without promoting a
box score into a Lomb-Scargle period claim.
