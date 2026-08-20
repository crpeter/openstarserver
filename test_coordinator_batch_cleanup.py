import unittest
from unittest.mock import patch

from openstar_coordinator_client import (
    CoordinatorUnavailableError,
    OpenStarCoordinatorClient,
)


class CoordinatorBatchCleanupTests(unittest.TestCase):
    def test_partial_batch_transport_failure_does_not_remove_completed_project(self):
        client = OpenStarCoordinatorClient()
        complete_a = {
            "status": "COMPLETE",
            "projectID": "a",
            "nodeContributions": {"node": 2},
        }
        running_b = {"status": "RUNNING", "projectID": "b"}

        with patch.object(
            client,
            "activate_project",
            side_effect=[{"projectID": "a"}, {"projectID": "b"}],
        ), patch.object(
            client,
            "project_status",
            side_effect=[
                complete_a,
                running_b,
                CoordinatorUnavailableError("offline"),
            ],
        ), patch.object(
            client, "remove_project"
        ) as remove, patch(
            "openstar_coordinator_client.time.sleep"
        ), self.assertRaises(CoordinatorUnavailableError):
            client.run_projects(["one.json", "two.json"], poll_interval=0)

        remove.assert_not_called()

    def test_successful_batch_captures_all_results_before_cleanup(self):
        client = OpenStarCoordinatorClient()
        events = []
        statuses = [
            {
                "status": "COMPLETE",
                "projectID": "a",
                "nodeContributions": {"node": 2},
            },
            {"status": "RUNNING", "projectID": "b"},
            {
                "status": "COMPLETE",
                "projectID": "b",
                "nodeContributions": {"node": 3},
            },
        ]

        def status(project_id):
            events.append(("status", project_id))
            return statuses.pop(0)

        def remove(project_id):
            events.append(("remove", project_id))

        with patch.object(
            client,
            "activate_project",
            side_effect=[{"projectID": "a"}, {"projectID": "b"}],
        ), patch.object(
            client, "project_status", side_effect=status
        ), patch.object(
            client, "remove_project", side_effect=remove
        ), patch(
            "openstar_coordinator_client.time.sleep"
        ):
            result = client.run_projects(["one.json", "two.json"], poll_interval=0)

        self.assertEqual(
            (("status", "a"), ("status", "b"), ("status", "b")),
            tuple(events[:3]),
        )
        self.assertEqual((("remove", "a"), ("remove", "b")), tuple(events[3:]))
        self.assertEqual(("a", "b"), result.project_ids)
        self.assertEqual({"node": 5}, result.node_contributions)


if __name__ == "__main__":
    unittest.main()
