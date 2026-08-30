# Workload lane test ownership

Each direct child directory is the exclusive test lane for the matching
package under `openstar_workloads/plugins/`.

- A workload lane may edit only its own test directory and matching plugin
  package.
- Do not edit sibling workload tests, `tests/platform/`, or repository-root
  tests from a workload-lane change.
- Keep shared contract, discovery, registry, schema-routing, and coordinator
  boundary tests in `tests/platform/` under foundation ownership.
- Breaking wire-contract changes require a new versioned workload and a new
  matching test lane.
