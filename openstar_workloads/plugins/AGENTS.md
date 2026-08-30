# Workload lane rules

Every module/package must export exactly one `PLUGIN` conforming to
`WorkloadPlugin`. Schema IDs are versioned wire contracts: changing semantics
requires a new identity. Keep lane-specific tests below `tests/platform/`.

Workers remain generic compute executors. Never put scientific orchestration or
interpretation into worker payload execution requirements.
