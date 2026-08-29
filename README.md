# OpenStar Server

## Development setup and tests

OpenStar supports Python 3.11 and newer. From the repository root, create a clean
virtual environment and install the runtime dependencies plus the development test
tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Run a focused test module or the complete buffered `unittest` suite:

```bash
python -m unittest test_tess_additional_sector_source_localization
python -m unittest discover -b
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

## Manual unresolved-period-family localization

`run_openstar_tess_period_family_localization.py` is an explicit, target-selected
continuation for the narrow 12-stage boundary where targeted independent sectors
all recover reliable but resolution-limited nearby peaks and the subsequent broad
search finds no promotable family. It is not part of autonomous repair or branch
planning.

The default command is read-only. It verifies all 12 immutable stage ledgers and
prints the frozen sectors and disposition without downloading TESS data or changing
the investigation:

```bash
python run_openstar_tess_period_family_localization.py \
  --state-dir ~/Documents/OpenStarScience/autonomous-sector-1-scale-1/state \
  --investigation-id tess-tess-sector-scan-1-tic-238919539-tess-sector-1-tic-238919539
```

After reviewing that output, add `--execute` to append stages 013-015. The server
uses each sector's already-persisted frequency only as a phase reference, downloads
official TESS target pixels (with the existing TESScut fallback), and measures
high-minus-low phase-image centroids with jackknife uncertainty. It performs no new
Lomb-Scargle search and sends no TESS-specific work to compute workers. The result
always retains `HUMAN_REVIEW_REQUIRED`, `physicalCycleResolved=false`, and
`physicalMechanismResolved=false`; the localization outcome only determines the
next conservative experiment.

### TIC 238919539 untouched-sector time-domain continuation

After stages 013-015 localize the period family to the TIC target, validate the
next manual boundary without changing the investigation:

```bash
python run_openstar_tess_period_family_time_domain_evolution.py \
  --state-dir ~/Documents/OpenStarScience/autonomous-sector-1-scale-1/state \
  --investigation-id tess-tess-sector-scan-1-tic-238919539-tess-sector-1-tic-238919539
```

The preregistered experiment freezes untouched Sectors 5, 6, 7, 8, 11, 61,
65, 66, 67, 68, 69, 87, and 88. It requires official SPOC 120-second products,
freezes both SAP and PDCSAP flux, and compares gap-aware autocorrelation with
cycle-to-cycle and first-half/second-half waveform similarity. It does not run
Lomb-Scargle and does not send TESS-specific work to generic compute workers.

After reviewing the validation output, add `--execute` to append stages 016-018.
SAP/PDCSAP disagreement fails closed. A replicated time-domain family remains a
detection-level result: the exact physical cycle and mechanism remain unresolved,
the claim remains `HUMAN_REVIEW_REQUIRED`, and the next test is selected from the
persisted stability/evolution outcome.

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

## Autonomous period-family follow-up

OpenStar admits period-family follow-up from verified persisted semantics, never a
TIC identifier or stage count. An unresolved/source-ambiguous family may enter
phase difference-image localization; target-supported unresolved families may
enter untouched-sector time-domain evolution; replicated but clock-unresolved
families may enter external long-baseline photometry. Historical COMPLETE or
QUIESCENT investigations are not reopened by deployment. Their existing manual
admission commands remain the explicit compatibility path.

Sector selection starts with persisted official sector/product/cadence/epoch
identity, rejects already consumed sectors, and freezes a deterministic
one-per-epoch selection before flux access. Small phase imaging and time-domain
model comparisons run coordinator-local. Only heavy frequency searches are sent
to generic workers as `openstar.lomb-scargle.v1`; workers contain no TESS or
survey logic.

Exact duplicate product metadata is deduplicated canonically. Conflicting
author, mission, cadence, or observation-epoch metadata for the same eligible
sector fails closed as `CONFLICTING_PRODUCT_METADATA`; catalog input order can
never decide the frozen epoch assignment. Selection never reads flux.

External surveys are tried in preregistered priority order. The first provider
passing coverage, season, band, quality, and persisted catalog-neighbor blending
gates is frozen with its raw response and SHA-256 provenance. ASAS-SN Sky Patrol
is the first adapter, but production fails closed as data unavailable unless its
public transport is configured; tests are network-free. The external observable
is blocked seasonal phase prediction inside the already frozen period-family
window, not another Lomb–Scargle detection.

**Detection is not source attribution, stable-clock resolution, a physical
cycle, or a physical mechanism.** Results retain `HUMAN_REVIEW_REQUIRED` unless
separate evidence justifies more.

Read-only compatibility validation (use a copied fixture/state tree, **never the
live science state**) is:

```bash
python -m pytest -q test_historical_stage_ledger.py \
  test_tess_period_family_difference_image.py \
  test_tess_period_family_time_domain_evolution.py
```

### ASAS-SN configuration and immutable external stages

The optional official Sky Patrol distribution is named `skypatrol` and exposes
the `pyasassn` import package. Its public `SkyPatrolClient()` constructor does
not use OpenStar credentials. If the optional client is absent or has the wrong
version, the run persists `PROVIDER_CONFIGURATION_UNAVAILABLE`; it does not
reinterpret an operational condition as scientific insufficiency.

Install the coordinator-only, pinned interface with
`python -m pip install -r requirements-optional-asassn.txt`. OpenStar supports
`skypatrol==0.6.21` for this adapter; it is not a worker dependency.

External follow-up is an append-only prepare/run/interpret sequence. Preparation
freezes the provider priority, family window, target identity, authoritative
catalog-neighbor evidence, and the complete versioned analysis method. The
method contract includes coverage gates, season definition, period grid,
held-out-season procedure, predictive and null thresholds, stability and alias
rules, uncertainty floors, and band agreement. Its hash is verified before a
provider is constructed or any flux is analyzed. Run separately freezes and ledgers
the provider coverage response, canonical raw response, cleaned measurements,
quality-filter counts, and acquisition metadata. Interpret binds both stage
results and preserves
physical-cycle and physical-mechanism resolution as false. Recovery reuses
byte-identical artifacts and rejects a mismatched frozen hash.

Provider configuration, transient transport failure, operational unavailability,
and malformed provider data remain operational outcomes. Only a successfully
parsed provider response that fails the frozen measurement, baseline, season,
phase-coverage, or band gates is `EXTERNAL_DATA_INSUFFICIENT`.

Period-family admission performs semantic validation in addition to SHA-256
verification. A matching hash proves immutability, while the shared validator
separately requires supported versions, finite positive periods, an increasing
acceptance window containing the frozen family, a supported observable, and
unique positive integer sectors.
