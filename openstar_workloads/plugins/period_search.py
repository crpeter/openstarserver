"""Compatibility adapters for the two existing period-search identities."""

from __future__ import annotations

import math
from typing import Any, Mapping

from openstar_workloads.contract import ValidationResult, WorkloadDefinition


def _first(mapping, *keys):
    return next((mapping[k] for k in keys if mapping.get(k) is not None), None)


class PeriodSearchCompatibilityPlugin:
    definition = WorkloadDefinition(
        workload_id="openstar.lomb-scargle.v1",
        dataset_schema_id="openstar.dataset.period-search.v1",
        payload_schema_id="openstar.payload.frequency-shard.v1",
        result_schema_id="openstar.result.period-search-shard.v1",
        allows_legacy_schemaless_workers=True,
    )
    uses_legacy_period_diagnostics = True

    def validate_dataset(self, dataset):
        search = dataset.get("frequencySearch")
        if not isinstance(search, Mapping):
            raise RuntimeError("Dataset is missing frequencySearch")
        minimum = float(_first(search, "minimumFrequency", "minFrequency", "startFrequency"))
        count = int(_first(search, "totalFrequencies", "frequencyCount"))
        chunk = int(_first(search, "frequenciesPerWorkUnit", "workUnitFrequencyCount", "chunkSize"))
        step = _first(search, "frequencyStep", "step")
        if step is None:
            maximum = float(_first(search, "maximumFrequency", "maxFrequency", "endFrequency"))
            step = 0.0 if count <= 1 else (maximum - minimum) / count
        if not math.isfinite(minimum) or not math.isfinite(float(step)) or float(step) <= 0:
            raise RuntimeError("Frequency grid must be finite with a positive step")
        if count <= 0 or chunk <= 0:
            raise RuntimeError("Frequency and shard counts must be positive")

    def build_work_payloads(self, dataset):
        search = dataset["frequencySearch"]
        minimum = float(_first(search, "minimumFrequency", "minFrequency", "startFrequency"))
        total = int(_first(search, "totalFrequencies", "frequencyCount"))
        size = int(_first(search, "frequenciesPerWorkUnit", "workUnitFrequencyCount", "chunkSize"))
        step = _first(search, "frequencyStep", "step")
        if step is None:
            maximum = float(_first(search, "maximumFrequency", "maxFrequency", "endFrequency"))
            step = 0.0 if total <= 1 else (maximum - minimum) / total
        step = float(step)
        return [
            {"frequencyStartIndex": start, "startFrequency": minimum + start * step,
             "frequencyStep": step, "frequencyCount": min(size, total - start)}
            for start in range(0, total, size)
        ]

    def canonicalize_result(self, work_unit, result):
        canonical = dict(result)
        payload = canonical.get("payload")
        if isinstance(payload, Mapping):
            for key in ("bestFrequency", "bestPeriodDays", "bestPower"):
                if canonical.get(key) is None and payload.get(key) is not None:
                    canonical[key] = payload[key]
        return canonical

    def validate_result(self, work_unit, result):
        if result.get("status") != "completed":
            return ValidationResult.reject("Work unit did not complete.")
        try:
            frequency, power = float(result["bestFrequency"]), float(result["bestPower"])
        except (KeyError, TypeError, ValueError):
            return ValidationResult.reject("Best frequency/power must be numeric.")
        if not math.isfinite(frequency) or not math.isfinite(power):
            return ValidationResult.reject("Best frequency/power must be finite.")
        start, step, count = (float(work_unit["startFrequency"]),
                              float(work_unit["frequencyStep"]), int(work_unit["frequencyCount"]))
        if not start - max(abs(step) * 2, 1e-7) <= frequency <= start + max(count - 1, 0) * step + max(abs(step) * 2, 1e-7):
            return ValidationResult.reject("Best frequency is outside work-unit range.")
        return ValidationResult.accept("Metal result is structurally valid.", deviceFrequency=frequency, devicePower=power)

    def reduce_dataset(self, dataset, work_units, results, *, terminal):
        best = max((r for r in results if r.get("bestPower") is not None),
                   key=lambda r: float(r["bestPower"]), default=None)
        return {"best": best, "terminal": terminal}

    def contribution_metrics(self, work_unit, dataset):
        samples = dataset.get("times")
        sample_count = len(samples) if isinstance(samples, list) else 0
        count = int(work_unit.get("payload", {}).get("frequencyCount", work_unit.get("frequencyCount", 0)))
        return {"workloadID": self.definition.workload_id, "sampleCount": sample_count,
                "frequencyCount": count, "sampleFrequencyEvaluations": sample_count * count}


PLUGIN = PeriodSearchCompatibilityPlugin()
