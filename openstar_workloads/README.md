# OpenStar workload plugins

This package lets the coordinator schedule multiple versioned compute
workloads without teaching coordinator core the meaning of their payloads.

Every direct package in `plugins/` exports `PLUGIN`, either one plugin or a
fixed tuple of plugins. Discovery imports only that trusted in-repository
namespace, in sorted order. The registry validates every plugin, rejects
duplicate workload IDs, and fails closed for unknown IDs.

A plugin owns:

- its workload, dataset, payload, and result schema identities;
- deterministic, single-pass work-payload construction;
- the small set of flattened fields required by legacy workers;
- result canonicalization and validation;
- dataset reduction and status fields; and
- accounting dimensions derived only from server-owned work and dataset data.

Each lane publishes its workload and schema identities in its own README and
tests. The discovered registry is the authoritative installed-ID inventory;
adding a lane must not require editing this shared foundation document.

Plugin hooks are deliberately ordinary mapping-in/value-out calls. Keep them
pure, stateless, reentrant, deterministic, and free of I/O so coordinator
locking and retries cannot introduce hidden behavior.
