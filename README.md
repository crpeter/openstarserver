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

New workers may optionally send generic operational telemetry directly to the sidecar. This
in-memory heartbeat is not required for contribution and never enters scheduling, accounting,
or scientific result state:

```http
POST /api/telemetry/heartbeat
Content-Type: application/json

{"nodeID":"worker-id","telemetry":{"deviceName":"Lab Mac","thermalState":"nominal","batteryLevel":0.82,"powerState":"charging","lowPowerMode":false,"osVersion":"15.0","appVersion":"1.0","computeState":"computing"}}
```
