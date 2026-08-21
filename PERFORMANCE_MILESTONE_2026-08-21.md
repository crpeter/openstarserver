# OpenStar Sector 1 Performance Milestone — August 21, 2026

## Summary

On August 21, 2026, OpenStar removed a major coordinator-side bottleneck in the TESS Sector 1 distributed period-search pipeline.

Observed sustained throughput increased from roughly **296 to 557 Sector 1 targets/hour** while preserving the existing scientific frequency-uncertainty method and worker accounting.

Compared with the earlier ~207 targets/hour baseline, the optimized system reached approximately **2.7× the original throughput**.

## Workload

- Sector 1 targets: 15,889
- Frequencies per target: 4,194,304
- Frequencies per atomic work unit: 4,096
- Atomic work units per target: 1,024
- Typical samples per frequency: ~18,000
- Worker workload: `openstar.lomb-scargle.v1`

Workers perform generic numeric compute. Science planning, interpretation, provenance, archive access, and investigation control remain server responsibilities.

## Performance

| Stage | Approx. throughput |
|---|---:|
| Earlier baseline | ~207 targets/hour |
| Tuned distributed system | ~296 targets/hour |
| After coordinator optimization | ~557 targets/hour |

Representative post-fix performance:

- ~11.69 billion sample-frequency evaluations/second
- 27+ billion evaluations/device-second aggregate Metal efficiency
- ~88% faster than the previous tuned system
- ~169% faster than the earlier baseline

## Bottleneck

Instrumentation showed that workers were being starved whenever a target completed.

The dominant delay was the server-side frequency-confidence interval calculation:

- before: ~3.7 seconds per target
- after: ~0.04–0.05 seconds per target

The implementation was changed from repeated Python loops over ~18,000 samples to reusable NumPy arrays and vectorized calculations.

The scientific profile-likelihood method, search behavior, and profile evaluation count were preserved.

## Result

After the optimization:

- terminal science finalization fell to roughly 0.06–0.09 seconds
- new worker claims generally resumed within roughly 0.2–0.4 seconds
- sustained fleet throughput approximately doubled

The improvement came from keeping existing hardware busy, not from reducing the scientific workload.

## Why it matters

OpenStar is testing whether meaningful scientific computing can be performed at scale using heterogeneous consumer hardware that already exists.

This milestone showed that orchestration and server-side analysis can be as important as raw compute capacity: the same worker fleet produced dramatically more useful science after removing a software bottleneck.

Performance claims will remain tied to reproducible workload definitions, device accounting, wall-clock measurements, persisted evidence, and scientific-equivalence tests.