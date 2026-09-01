# Morphology-grid lane ownership

This directory exclusively owns `openstar.morphology-grid.v1`.

- Changes must remain in this package and `tests/workloads/morphology_grid/`.
- Preserve the published workload, dataset, payload, result, morphology-family,
  component-template-family, and execution-contract IDs.
- Keep the worker contract generic and free of target identity, interpretation,
  workflow policy, and discovery claims.
- Hooks must remain pure, stateless, deterministic, reentrant, and free of
  network and filesystem I/O.
- Do not add legacy flattened work-unit or result fields.
- Shared workload-platform code and sibling workload lanes are out of scope.
