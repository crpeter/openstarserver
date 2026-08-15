# OpenStar Server Agent Rules

## Mission

OpenStar is an open-source distributed science system.

The server coordinates scientific investigations and distributes generic computational work to workers such as Macs and iPhones.

The long-term goal is for OpenStar to autonomously produce scientifically defensible results while preserving reproducibility, provenance, and clear separation between scientific reasoning and distributed computation.

---

## Core Architecture

### Server responsibilities

The server may:

- manage projects and datasets
- manage investigations
- decide which scientific workflow step should run next
- determine whether a scientific branch is executable
- generate generic work units
- schedule compatible workers
- collect distributed results
- validate and aggregate results
- interpret scientific results
- track provenance
- determine when an investigation is complete, blocked, or awaiting new data
- advance autonomous investigation to another target

### Worker responsibilities

Workers are generic compute workers.

Workers must NOT:

- understand TESS
- understand astronomy
- understand stars, planets, eclipses, light curves, or astrophysical hypotheses
- decide what scientific experiment should run
- select the next investigation step
- interpret scientific results
- contain target-specific logic

Workers receive generic work units, execute supported computational workloads, and return results.

A Mac worker must not become a special OpenStar orchestrator.

An iPhone worker must follow the same conceptual worker model as any other worker.

---

## Architectural Boundary

Maintain this separation:

```text
Scientific workflow
        ↓
Investigation engine
        ↓
Coordinator / scheduler
        ↓
Generic work units
        ↓
Generic workers
        ↓
Computed results
        ↓
Scientific workflow interpretation

