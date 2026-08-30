# Workload package ownership

Each workload lane owns its immutable schema identities, dataset validation,
deterministic sharding, result validation/canonicalization, reduction, and
trusted contribution metrics. Add a lane as a self-contained module or package
under `plugins/`; do not add scientific conditionals to coordinator core.

Plugins are trusted server code. Discovery must remain deterministic and must
not load entry points, arbitrary filesystem paths, or worker-provided modules.
