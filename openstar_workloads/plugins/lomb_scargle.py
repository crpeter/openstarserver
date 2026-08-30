"""Compatibility adapters for existing Lomb-Scargle identities."""

import math

from openstar_workloads.contract import ResultValidation, WorkloadDefinition


def _first(mapping, *keys):
    return next((mapping[key] for key in keys if mapping.get(key) is not None), None)


class LombScarglePlugin:
    uses_legacy_coordinator_diagnostics = True

    def __init__(self, workload_id):
        self.definition = WorkloadDefinition(
            workload_id, "openstar.dataset.lomb-scargle.v1",
            "openstar.payload.lomb-scargle-shard.v1",
            "openstar.result.lomb-scargle-shard.v1", True,
        )

    def validate_dataset(self, dataset):
        self._grid(dataset)

    def _grid(self, dataset):
        search = dataset["frequencySearch"]
        minimum = float(_first(search, "minimumFrequency", "minFrequency", "startFrequency"))
        total = int(_first(search, "totalFrequencies", "frequencyCount"))
        size = int(_first(search, "frequenciesPerWorkUnit", "workUnitFrequencyCount", "chunkSize"))
        step = _first(search, "frequencyStep", "step")
        if step is None:
            maximum = float(_first(search, "maximumFrequency", "maxFrequency", "endFrequency"))
            step = 0 if total <= 1 else (maximum - minimum) / total
        step = float(step)
        if not math.isfinite(minimum) or not math.isfinite(step) or step <= 0 or total <= 0 or size <= 0:
            raise RuntimeError("Invalid Lomb-Scargle frequency grid")
        return minimum, step, total, size

    def build_work_payloads(self, dataset):
        minimum, step, total, size = self._grid(dataset)
        return [{"frequencyStartIndex": start, "startFrequency": minimum + start * step,
                 "frequencyStep": step, "frequencyCount": min(size, total - start)}
                for start in range(0, total, size)]

    def canonicalize_result(self, work_unit, result):
        result = dict(result)
        payload = result.get("payload")
        if isinstance(payload, dict):
            for key in ("bestFrequency", "bestPeriodDays", "bestPower"):
                if result.get(key) is None and payload.get(key) is not None:
                    result[key] = payload[key]
        return result

    def validate_result(self, work_unit, result):
        if result.get("status") != "completed":
            return ResultValidation(False, "Work unit did not complete.", {})
        try:
            frequency = float(result["bestFrequency"])
            power = float(result["bestPower"])
        except (KeyError, TypeError, ValueError):
            return ResultValidation(False, "Best frequency/power must be numeric.", {})
        start = float(work_unit["startFrequency"])
        step = float(work_unit["frequencyStep"])
        count = int(work_unit["frequencyCount"])
        tolerance = max(abs(step) * 2, 1e-7)
        end = start + max(count - 1, 0) * step
        if not math.isfinite(frequency) or not math.isfinite(power):
            return ResultValidation(False, "Best frequency/power must be finite.", {})
        if frequency < start - tolerance or frequency > end + tolerance:
            return ResultValidation(False, "Best frequency is outside work-unit range.", {})
        return ResultValidation(True, "Metal result is structurally valid.",
                                {"deviceFrequency": frequency, "devicePower": power})

    def reduce_dataset(self, dataset, work_units, results, *, terminal):
        best = max(results, key=lambda result: float(result["bestPower"]), default=None)
        return {"best": best, "terminal": terminal}

    def contribution_metrics(self, work_unit, dataset):
        samples = len(dataset.get("times", []))
        count = int(work_unit["payload"]["frequencyCount"])
        return {"workloadID": self.definition.workload_id, "sampleCount": samples,
                "frequencyCount": count, "sampleFrequencyEvaluations": samples * count}


# openstar.tess-period-search.v1 is the historical Lomb alias, not box search.
PLUGIN = (
    LombScarglePlugin("openstar.lomb-scargle.v1"),
    LombScarglePlugin("openstar.tess-period-search.v1"),
)
