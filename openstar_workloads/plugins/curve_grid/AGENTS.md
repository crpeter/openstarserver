# Curve-grid lane ownership

This directory exclusively owns `openstar.curve-grid.v1`.

- Changes must remain in this package and `tests/workloads/curve_grid/`.
- Preserve the published workload, dataset, payload, result, and family IDs.
- Keep the worker contract domain-neutral and free of interpretation.
- Hooks must remain pure, stateless, deterministic, reentrant, and free of
  network and filesystem I/O.
- Do not add legacy flattened work-unit or result fields.
- Shared workload-platform code and sibling workload lanes are out of scope.
