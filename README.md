# OpenStar Server

```bash
source .venv/bin/activate
```

## Fleet dashboard sidecar

The optional dashboard is a separate process. It only performs `GET` requests against the
coordinator's existing `/v1` APIs, so the coordinator and workers operate normally when the
dashboard is stopped or absent.

Science runners register themselves in the durable operational science-run catalog at
`data/science-runs.sqlite3`. The dashboard reads that catalog automatically, so it can show
current and historical science without being given TESS sectors, state directories, or other
science-specific launch arguments. Catalog metadata never replaces or rewrites authoritative
investigation/science history.

For science state created before the catalog existed, run the one-time backfill. It discovers
likely sector-sweep roots under `/tmp` and `data`; if a matching sector-sweep process is actually
running locally, that run is recorded as running. Other incomplete historical state is kept
conservatively as discovered/incomplete.

```bash
python backfill_openstar_science_runs.py
```

Normal dashboard launch:

```bash
# Terminal 1: existing coordinator
python coordinator.py --idle --host 127.0.0.1 --port 8080

# Terminal 2: dashboard
python openstar_dashboard.py --coordinator http://127.0.0.1:8080 \
  --host 127.0.0.1 --port 8081
```

Open <http://127.0.0.1:8081/dashboard/>.

New workers may optionally send generic operational telemetry directly to the sidecar. This
in-memory heartbeat is not required for contribution and never enters scheduling, accounting,
or scientific result state:

```http
POST /api/telemetry/heartbeat
Content-Type: application/json

{"nodeID":"worker-id","telemetry":{"deviceName":"Lab Mac","thermalState":"nominal","batteryLevel":0.82,"powerState":"charging","lowPowerMode":false,"osVersion":"15.0","appVersion":"1.0","computeState":"computing"}}
```
