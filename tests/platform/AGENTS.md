# Workload Platform Test Rules

This directory owns tests for the shared workload-plugin contract, registry,
discovery, schema routing, and coordinator integration boundaries.

- Keep workload-specific algorithms and parity fixtures under
  `tests/workloads/<workload>/`.
- Do not add target, mission, astronomy, or worker-platform assumptions here.
- Exercise plugins through the public contract and coordinator interfaces.
- Unknown, malformed, partially identified, and duplicate workloads must fail
  closed.
- Preserve schema-less compatibility only where the trusted workload definition
  explicitly allows it.
- Do not move these tests into the repository-root test modules.
