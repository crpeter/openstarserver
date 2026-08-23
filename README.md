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
