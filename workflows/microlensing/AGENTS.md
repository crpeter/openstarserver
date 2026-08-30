# Microlensing workflow ownership

This package owns server-side microlensing archive preparation and future
microlensing interpretation.

- Keep archive acquisition, provenance, structural inventory, and later
  microlensing-specific reasoning in this package.
- Do not put microlensing concepts into generic workload plugins, workers,
  coordinator core, schedulers, or shared registries.
- Preserve official source artifacts and fail closed on provenance or
  integrity conflicts.
- Archive data belongs in caller-selected external state and must never be
  added to the repository.
- Tests belong only in `tests/workflows/microlensing/`.
