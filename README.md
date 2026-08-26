# OpenStar Server

```bash
source .venv/bin/activate
```

## Fleet dashboard sidecar

The optional dashboard is a separate process. It only performs `GET` requests against the
coordinator's existing `/v1` APIs, so the coordinator and workers operate normally when the
dashboard is stopped or absent. For sector-level progress, the sidecar may also read explicitly
configured persisted TESS sweep roots directly; those reads never start, restart, or modify a
sweep.

```bash
# Terminal 1: existing coordinator
python coordinator.py --idle --host 127.0.0.1 --port 8080

# Terminal 2: dashboard. Repeat --sector-sweep-state-dir for additional roots.
python openstar_dashboard.py --coordinator http://127.0.0.1:8080 \
  --host 127.0.0.1 --port 8081 \
  --sector-sweep-state-dir /path/to/existing-sector-sweep-state
```

Open <http://127.0.0.1:8081/dashboard/>.

Science runners also register their durable state roots in the best-effort
SQLite catalog at `data/science-runs.sqlite3`.  The dashboard uses that catalog
to discover sector sweeps and investigation history without repeated
`--sector-sweep-state-dir` arguments.  Set `OPENSTAR_SCIENCE_RUN_CATALOG` for
both runners and the dashboard (or pass `--science-run-catalog` to the
dashboard) to keep the catalog elsewhere.  Catalog failures affect visibility
only and never determine or repair science execution.

### Durable science-state locations

Production science state must be placed on a durable filesystem. Do **not** use
OS cleanup locations such as `/tmp/...` or `/private/tmp/...`. Select an
appropriate durable location explicitly, for example:

```bash
python run_openstar_tess_sector_sweep.py --sector 4 \
  --coordinator-url http://127.0.0.1:8080 \
  --state-dir ~/Documents/OpenStarScience/tess-sector-4
```

The production TESS runners reject temporary state roots by default, including
temporary paths reached through aliases or symlinks. `--allow-temporary-state`
is an explicit escape hatch intended **only** for disposable smoke/test runs;
it must not be used for authoritative or production science state. Durable
paths remain user-selected—this example is guidance, not a required application
directory.

### Immutable historical path relocation

An investigation copied from an old or unsafe location can explicitly redirect
historical reads with repeatable `--relocate-historical-root OLD=NEW` options.
This runtime facility does not migrate files or rewrite science history:
persisted paths, provenance, and expected artifact hashes remain authoritative,
and bytes at each mapped durable destination must pass the original SHA-256
checks. Relocation is explicit and performs no filesystem search or fallback to
the old root. It is intended for recovery or movement of authoritative science
state between durable locations; new science should start in durable storage
and normally needs no relocation mapping.

Historical roots can be indexed explicitly without changing their science
files. The bounded scan examines only each configured root and its direct
children; by default it checks `/tmp`, `/private/tmp`, and repository `data`:

```bash
python backfill_science_run_catalog.py
python backfill_science_run_catalog.py --root /science/archive --limit 100
```

Opening a catalog created by the earlier #72 observability implementation
migrates that SQLite catalog in place while preserving its records; no science
root is touched. The dashboard also reads the contribution ledger in SQLite
read-only mode and treats a newer accepted-work timestamp as worker-presence
evidence. Use `--contribution-ledger` to select that optional ledger.

New workers may optionally send generic operational telemetry directly to the sidecar. This
in-memory heartbeat is not required for contribution and never enters scheduling, accounting,
or scientific result state:

```http
POST /api/telemetry/heartbeat
Content-Type: application/json

{"nodeID":"worker-id","telemetry":{"deviceName":"Lab Mac","thermalState":"nominal","batteryLevel":0.82,"powerState":"charging","lowPowerMode":false,"osVersion":"15.0","appVersion":"1.0","computeState":"computing"}}
```

## Targeted observation campaign

Two Sector 1 deep investigations have reached a genuine external-observation boundary. OpenStar
has exhausted the intended archive/computational continuation for these residual-source questions
and generated pre-registered `openstar.tess-targeted-observation-plan.v1` plans. Do not replace or
retune those plans after looking at new data.

### Targets

| Investigation | TIC | Target Gaia DR3 | Counterpart Gaia DR3 | Gaia separation | Frozen residual frequency band |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tess-discovery-sector-1-tic-24705603` | 24705603 | 4697673673172278528 | 4698424605253172352 | 143.604 arcsec | 0.223447247-0.232567543 cycles/day |
| `tess-discovery-sector-1-tic-350862394` | 350862394 | 5498909546046688256 | 5498909782268097280 | 70.627 arcsec | 0.351962961-0.527944441 cycles/day |

The established main TESS periodic family remains target-associated. This campaign tests only
which frozen Gaia source produces the separate drifting residual component: target, counterpart,
both, or neither.

### Pre-registered observing contract

Minimum campaign requirements per target:

- baseline: 45 days minimum; 90 days preferred
- distinct nights: 24 minimum; 45 preferred
- visits: at least 2 per observed night, preferably separated by 1-3 hours
- filters: at least 2 standard/well-characterized passbands; prefer `r/i`, acceptable alternate `g/r`
- image quality: prefer <=3 arcsec FWHM; hard source-attribution limit <=5 arcsec FWHM
- pixel scale: prefer <=1 arcsec/pixel; hard maximum <=2 arcsec/pixel
- each visit must contain paired short-tier and deep-tier imaging
- short tier: target S/N >=100, unsaturated, peak <=70% of documented detector linear/saturation level
- deep tier: counterpart S/N >=20, preferred >=30; target saturation is allowed only if its saturation structure cannot contaminate the counterpart measurement
- vary observation time across the campaign when practical rather than sampling at the same sidereal/local time every night

Preferred deliverables are calibrated/reduced FITS images for every accepted and rejected
exposure plus the generated OpenStar CSV ingest schema. Preserve timing, filter, exposure,
observatory, airmass/FWHM when available, saturation/contamination flags, and rejected
measurements. BJD_TDB is preferred; UTC mid-exposure plus observatory coordinates is sufficient
for OpenStar to compute barycentric time.

### Frozen analysis / stop rule

Do not expand the frequency band, move cadence windows, lower the prominence threshold, or change
filter acceptance after seeing the new periodograms. A supported source requires a reliable,
non-boundary signal with independent-peak prominence >=2, recurrence in at least two filters,
and recurrence in the pre-defined time-resolved campaign windows. If neither source satisfies the
contract, leave the source unresolved and reassess the model instead of tuning until a source wins.

### Execution options

The targets are southern objects (approximately Dec -67 and -57), so a southern observatory or
worldwide robotic network is required. Candidate paths researched for the first campaign are:

1. **AAVSOnet** - preferred low-cost proposal route for a sustained multi-night photometric campaign.
2. **Slooh** - use a small pilot first to verify that its presets can satisfy the paired short/deep dynamic-range requirements and return usable calibrated FITS.
3. **iTelescope / Siding Spring** - paid fallback when exact exposure and filter control is required.
4. **Las Cumbres Observatory** - proposal/DDT route when professional robotic-network time is available.

Before committing a full paid campaign, obtain pilot FITS for at least one target and measure target
peak counts/linearity, target S/N, counterpart S/N, FWHM, and contamination. Use those measurements
to choose the exposure durations without changing the already-frozen cadence/filter/analysis rules.

Generated plan artifacts live under each investigation's
`artifacts/targeted-observation-plan/` directory in its durable deep-state root.

## Unattended autonomous TESS portfolios

`run_openstar_autonomous_tess.py --multi-investigation` retains its one-shot
semantics: it drains all work that is currently runnable and then exits. Add
`--daemon` to keep the multi-investigation supervisor alive. The daemon polls
only due durable external jobs, applies durable dependency wakeups, drains the
scheduler to idle, atomically writes `autonomy-heartbeat.json` beneath the state
directory, sleeps for `--daemon-interval-seconds` (60 seconds by default), and
repeats until SIGINT or SIGTERM.

FAILED and RECOVERY_REQUIRED investigations are reported and quarantined while
healthy targets continue. BLOCKED_PREREQUISITES and WAITING_EXTERNAL_DATA are
reported without being repeatedly dispatched. RECOVERY_REQUIRED remains a
fail-closed condition needing a human decision or a narrow workflow-specific
recovery adapter with an explicit replay-safety contract. The heartbeat is
observability only and is never used to reconstruct scientific state.
