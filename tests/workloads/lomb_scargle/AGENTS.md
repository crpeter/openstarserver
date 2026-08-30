# Lomb-Scargle Workload Test Rules

This directory owns tests for the canonical Lomb-Scargle workload and its
historical TESS workload alias.

- Preserve the existing frequency-grid payload and legacy flattened fields.
- Preserve legacy coordinator diagnostics and science-metadata validation.
- Keep canonical Lomb contribution dimensions compatible with the existing
  ledger; the historical alias must not silently gain new accounting metrics.
- Do not add tests for other workloads or edit shared/root test modules.
