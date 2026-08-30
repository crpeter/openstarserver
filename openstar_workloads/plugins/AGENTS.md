# Built-in workload lanes

Each direct child package is an independently owned workload lane and exports
`PLUGIN` from its `__init__.py`.

- A lane may add private helper modules beneath its own package; discovery does
  not inspect those nested modules.
- A lane must not edit another lane, shared platform files, coordinator core,
  or another lane's tests.
- Preserve published wire IDs and compatibility behavior within the existing
  workload version. Breaking changes require a new versioned workload ID.
- Workers remain generic: science metadata and interpretation stay server-side.
