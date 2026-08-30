# Workload platform ownership

This package is the server-owned boundary between coordinator mechanics and
versioned computational workloads.

- Shared contracts, discovery, conformance, and registry code are foundation
  ownership. Change them in a dedicated platform PR.
- Each built-in workload lives in one direct subpackage below `plugins/`.
- Workload-lane changes must stay inside that lane and its matching tests.
- Discovery must remain deterministic and restricted to trusted repository
  code under `openstar_workloads.plugins`.
- Plugin hooks must be pure, stateless, reentrant, deterministic, and fast.
  They must not perform network or filesystem I/O.
- Workload plugins own schema identities, dataset validation, deterministic
  sharding, result canonicalization and validation, reduction, and trusted
  accounting dimensions.
- Do not put scientific interpretation or workload-specific branches back in
  coordinator core.
