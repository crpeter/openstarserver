"""Existing box-period search workload adapter."""

import math

from openstar_workloads.contract import ResultValidation, WorkloadDefinition


class BoxPeriodPlugin:
    uses_legacy_coordinator_diagnostics = False
    definition = WorkloadDefinition(
        "openstar.box-period-search.v1", "openstar.dataset.box-period-search.v1",
        "openstar.payload.box-period-shard.v1", "openstar.result.box-period-shard.v1",
        True,
    )

    def _grid(self, dataset):
        search = dataset["periodSearch"]
        minimum = float(search["minimumPeriodDays"])
        step = float(search["periodStepDays"])
        total = int(search["totalPeriods"])
        size = int(search["periodsPerWorkUnit"])
        if not math.isfinite(minimum) or not math.isfinite(step) or minimum <= 0 or step <= 0 or total <= 0 or size <= 0:
            raise RuntimeError("Invalid box-period grid")
        return minimum, step, total, size

    def validate_dataset(self, dataset):
        self._grid(dataset)

    def build_work_payloads(self, dataset):
        minimum, step, total, size = self._grid(dataset)
        return [{"periodStartIndex": start, "startPeriodDays": minimum + start * step,
                 "periodStepDays": step, "periodCount": min(size, total - start)}
                for start in range(0, total, size)]

    def canonicalize_result(self, work_unit, result):
        result = dict(result)
        payload = result.get("payload")
        if isinstance(payload, dict):
            for key in ("bestPeriodDays", "bestScore"):
                if result.get(key) is None and payload.get(key) is not None:
                    result[key] = payload[key]
        return result

    def validate_result(self, work_unit, result):
        if result.get("status") != "completed":
            return ResultValidation(False, "Work unit did not complete.", {})
        try:
            period, score = float(result["bestPeriodDays"]), float(result["bestScore"])
        except (KeyError, TypeError, ValueError):
            return ResultValidation(False, "Best box period/score must be numeric.", {})
        start, step, count = (float(work_unit["startPeriodDays"]),
                              float(work_unit["periodStepDays"]), int(work_unit["periodCount"]))
        tolerance = max(step * 2, 1e-9)
        if not math.isfinite(period) or not math.isfinite(score):
            return ResultValidation(False, "Best box period/score must be finite.", {})
        if period < start - tolerance or period > start + max(count - 1, 0) * step + tolerance:
            return ResultValidation(False, "Best period is outside work-unit range.", {})
        return ResultValidation(True, "Box-period result is valid.", {"periodDays": period, "score": score})

    def reduce_dataset(self, dataset, work_units, results, *, terminal):
        best = max(results, key=lambda result: float(result["bestScore"]), default=None)
        return {"status": "COMPLETE" if terminal else "SEARCHING",
                "bestPeriodDays": best.get("bestPeriodDays") if best else None,
                "bestScore": best.get("bestScore") if best else None}

    def contribution_metrics(self, work_unit, dataset):
        samples = len(dataset.get("times", []))
        count = int(work_unit["payload"]["periodCount"])
        return {"workloadID": self.definition.workload_id, "sampleCount": samples,
                "periodCount": count, "samplePeriodEvaluations": samples * count}


PLUGIN = BoxPeriodPlugin()
