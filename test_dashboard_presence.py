import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from coordinator import _registered_nodes_with_live_activity
from dashboard import build_snapshot
from openstar_dashboard import ContributionActivityReader, _overlay_latest_activity


class _FakeState:
    def __init__(self, nodes):
        self.lock = threading.RLock()
        self.nodes = nodes


class _FakeRuntime:
    def __init__(self, durable, states):
        self.lock = threading.RLock()
        self._durable = durable
        self._states = states

    def registered_nodes(self):
        return [dict(node) for node in self._durable]


class CoordinatorNodePresenceTests(unittest.TestCase):
    def test_live_scheduler_sighting_overrides_stale_durable_last_seen(self):
        runtime = _FakeRuntime(
            [{"nodeID": "phone-1", "lastSeenAt": 10.0, "hardwareIdentifier": "Phone"}],
            {"project": _FakeState({"phone-1": {"id": "phone-1", "lastSeenAt": 95.0}})},
        )

        nodes = _registered_nodes_with_live_activity(runtime)

        self.assertEqual(1, len(nodes))
        self.assertEqual(95.0, nodes[0]["lastSeenAt"])
        self.assertEqual("Phone", nodes[0]["hardwareIdentifier"])


class ContributionPresenceTests(unittest.TestCase):
    def test_reader_uses_latest_accepted_contribution_per_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "contributions.sqlite3")
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE TABLE contributions (node_id TEXT NOT NULL, accepted_at REAL NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO contributions(node_id, accepted_at) VALUES(?, ?)",
                    [("phone-1", 20.0), ("phone-1", 80.0), ("mac-1", 60.0)],
                )
                connection.commit()
            finally:
                connection.close()

            latest = ContributionActivityReader(path).latest_seen()

        self.assertEqual({"phone-1": 80.0, "mac-1": 60.0}, latest)

    def test_accepted_work_makes_recent_worker_connected(self):
        nodes = _overlay_latest_activity(
            [{"nodeID": "phone-1", "lastSeenAt": 10.0}],
            {"phone-1": 95.0},
        )
        snapshot = build_snapshot(
            nodes,
            {"allTime": {"nodes": []}, "currentSession": {}},
            [],
            {},
            now=100.0,
        )

        worker = snapshot["workers"][0]
        self.assertEqual("connected", worker["connectionState"])
        self.assertEqual(95.0, worker["lastSeenAt"])

    def test_frontend_shows_connection_status_and_precise_freshness(self):
        source = Path("dashboard/app.js").read_text(encoding="utf-8")
        self.assertIn('text: "ONLINE"', source)
        self.assertIn('text: "RECENTLY DISCONNECTED"', source)
        self.assertIn('`${Math.floor(seconds)}s ago`', source)
        self.assertIn("setInterval(renderWorkers, 1000)", source)
        self.assertNotIn("worker.computeState.toUpperCase()", source)


if __name__ == "__main__":
    unittest.main()
