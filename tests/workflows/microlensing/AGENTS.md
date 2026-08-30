# Microlensing workflow test ownership

Tests here cover only `workflows/microlensing/`.

- Use local miniature source and table fixtures only.
- Never contact the network.
- Exercise acquisition through an injected fake fetcher.
- Tests may construct deterministic external project and dataset artifacts for
  microlensing workflow builders.
- Do not activate a coordinator, execute workloads, or interpret results.
- Do not use downloaded archive fixtures for project-builder tests.
- Do not edit shared or existing tests for this lane.
