import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from openstar_external_jobs import (
    ExternalJob, ExternalJobMonitor, ExternalJobStore, PollResult, stable_job_id,
)


class Provider:
    def __init__(self, results): self.results, self.calls = iter(results), []
    def poll(self, job): self.calls.append(job.remoteTaskURL); return next(self.results)


class ExternalJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.store = ExternalJobStore(self.root / "external-jobs")

    def tearDown(self): self.temp.cleanup()

    def job(self, role="target-control"):
        return ExternalJob.create(provider="atlas", investigation_id="inv",
            trigger_stage_id="052-prepare", dependency_id="atlas:inv:052", role=role)

    def test_atomic_round_trip_and_stable_distinct_ids(self):
        first = replace(self.job(), remoteTaskURL="https://task/1")
        second = replace(self.job("catalog-counterpart"), remoteTaskURL="https://task/2")
        self.store.save(first); self.store.save(second)
        self.assertEqual(first, ExternalJobStore(self.store.root).load(first.id))
        self.assertNotEqual(first.id, second.id)
        self.assertFalse(list(self.store.root.glob("*.tmp")))

    def test_record_schema_cannot_persist_credentials(self):
        self.store.save(replace(self.job(), remoteTaskURL="https://task"))
        raw = json.loads(self.store.path_for(self.job().id).read_text())
        self.assertEqual("openstar.external-job.v1", raw["version"])
        self.assertFalse({"token", "username", "password"} & set(raw))

    def test_complete_is_not_replaceable(self):
        complete = replace(self.job(), state="COMPLETE", remoteTaskURL="task",
                           remoteResultURL="result")
        self.store.save(complete)
        with self.assertRaises(ValueError):
            self.store.save(replace(complete, state="SUBMITTED"))

    def test_monitor_polls_once_and_wakes_only_complete_group(self):
        now = datetime.now(timezone.utc)
        a = replace(self.job(), remoteTaskURL="a", nextCheckAt=now.isoformat())
        b = replace(self.job("catalog-counterpart"), remoteTaskURL="b",
                    nextCheckAt=now.isoformat())
        self.store.save(a); self.store.save(b)
        provider = Provider([PollResult("COMPLETE", "ra"), PollResult("QUEUED")])
        monitor = ExternalJobMonitor(self.store, {"atlas": provider})
        self.assertEqual((), monitor.poll_due(now=now)); self.assertEqual(2, len(provider.calls))

    def test_both_complete_wake_dependency_once(self):
        for job in (self.job(), self.job("catalog-counterpart")):
            self.store.save(replace(job, remoteTaskURL=job.role))
        provider = Provider([PollResult("COMPLETE", "r1"), PollResult("COMPLETE", "r2")])
        monitor = ExternalJobMonitor(self.store, {"atlas": provider})
        self.assertEqual((("inv", "atlas:inv:052"),), monitor.poll_due())
        self.assertEqual((), monitor.poll_due())

    def test_poll_failure_retains_authoritative_task(self):
        job = replace(self.job(), remoteTaskURL="authoritative")
        self.store.save(job)
        class Failure:
            def poll(self, unused): raise RuntimeError("HTTP 429")
        self.assertEqual((), ExternalJobMonitor(self.store, {"atlas": Failure()}).poll_due())
        saved = self.store.load(job.id)
        self.assertEqual("authoritative", saved.remoteTaskURL)
        self.assertEqual("SUBMITTED", saved.state)
        self.assertIn("429", saved.lastOperationalError)


if __name__ == "__main__": unittest.main()
