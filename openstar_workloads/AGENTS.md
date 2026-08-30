# Workload package ownership

Each workload package owns its schemas, dataset validation, deterministic shard
construction, result validation/canonicalization, reduction, and contribution
metrics. Add lane-specific behavior below `plugins/`, never to coordinator core.

Discovery must remain restricted to trusted repository code.
