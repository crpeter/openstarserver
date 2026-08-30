# Box-Period Workload Test Rules

This directory owns the production box-period search wire contract and
scientific-result validation parity tests.

- Preserve `boxPeriodSearch.frequencyWindows`; do not replace it with a generic
  period grid.
- Preserve all payload fields, sample gates, validation checks, deterministic
  candidate ordering, legacy status fields, and contribution metrics.
- Keep box-period tests independent from Lomb-Scargle internals.
- Do not add tests for other workloads or edit shared/root test modules.
