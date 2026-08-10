import argparse
import json
import math
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


LEASE_SECONDS = 120.0
RETRY_COOLDOWN_SECONDS = 1.0
BUILD = "multi-target-v4"
DEFAULT_PROJECT_PATH = "data/projects/openstar.tess-multi-target-v1.json"


def normalize_id(value):
    return None if value is None else str(value).strip().lower()


def first_value(mapping, *keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


class CoordinatorState:
    def __init__(self, project_path):
        self.lock = threading.RLock()

        self.project_path = Path(project_path)
        self.project = self._load_json(self.project_path)

        self.project_id = self.project["id"]
        self.project_name = self.project.get("name", self.project_id)
        self.workload_id = self.project["workloadID"]

        self.datasets = {}
        self.dataset_manifest_entries = {}

        self.work_units = {}
        self.work_ids_by_dataset = {}

        self.pending = deque()
        self.assigned = {}
        self.completed = {}
        self.retry_after = {}

        self.nodes = {}

        self.reported_completed_datasets = set()
        self.reported_project_complete = False

        self._load_datasets()
        self._build_work_units()

    @staticmethod
    def _load_json(path):
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _resolve_dataset_path(self, entry):
        raw_path = Path(entry["path"])

        if raw_path.is_absolute():
            return raw_path

        candidates = [
            Path.cwd() / raw_path,
            self.project_path.parent / raw_path,
            self.project_path.parent.parent / raw_path,
            self.project_path.parent.parent.parent / raw_path,
            ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return candidates[0].resolve()

    def _load_datasets(self):
        for entry in self.project.get("datasets", []):
            dataset_id = entry["id"]
            dataset = self._load_json(self._resolve_dataset_path(entry))

            self.datasets[dataset_id] = dataset
            self.dataset_manifest_entries[dataset_id] = entry

    @staticmethod
    def _frequency_search(dataset):
        search = dataset["frequencySearch"]

        minimum_frequency = float(
            first_value(
                search,
                "minimumFrequency",
                "minFrequency",
                "startFrequency",
            )
        )

        total_frequencies = int(
            first_value(
                search,
                "totalFrequencies",
                "frequencyCount",
            )
        )

        frequencies_per_work_unit = int(
            first_value(
                search,
                "frequenciesPerWorkUnit",
                "workUnitFrequencyCount",
                "chunkSize",
            )
        )

        frequency_step = first_value(
            search,
            "frequencyStep",
            "step",
        )

        if frequency_step is None:
            maximum_frequency = float(
                first_value(
                    search,
                    "maximumFrequency",
                    "maxFrequency",
                    "endFrequency",
                )
            )

            frequency_step = (
                0.0
                if total_frequencies <= 1
                else (
                             maximum_frequency - minimum_frequency
                     ) / total_frequencies
            )

        return {
            "minimumFrequency": minimum_frequency,
            "frequencyStep": float(frequency_step),
            "totalFrequencies": total_frequencies,
            "frequenciesPerWorkUnit": frequencies_per_work_unit,
        }

    def _build_work_units(self):
        for dataset_id, dataset in self.datasets.items():
            search = self._frequency_search(dataset)

            minimum_frequency = search["minimumFrequency"]
            frequency_step = search["frequencyStep"]
            total_frequencies = search["totalFrequencies"]
            frequencies_per_work_unit = search[
                "frequenciesPerWorkUnit"
            ]

            for start_index in range(
                    0,
                    total_frequencies,
                    frequencies_per_work_unit,
            ):
                frequency_count = min(
                    frequencies_per_work_unit,
                    total_frequencies - start_index,
                    )

                work_id = str(uuid.uuid4())
                normalized_work_id = normalize_id(work_id)

                work_unit = {
                    "id": work_id,
                    "projectID": self.project_id,
                    "workloadID": self.workload_id,
                    "datasetID": dataset_id,
                    "frequencyStartIndex": start_index,
                    "startFrequency": (
                            minimum_frequency
                            + start_index * frequency_step
                    ),
                    "frequencyStep": frequency_step,
                    "frequencyCount": frequency_count,
                }

                self.work_units[
                    normalized_work_id
                ] = work_unit

                self.work_ids_by_dataset.setdefault(
                    dataset_id,
                    [],
                ).append(
                    normalized_work_id
                )

                self.pending.append(
                    normalized_work_id
                )

    def register_node(self, payload):
        node_id = str(payload["nodeID"])
        capabilities = payload.get("capabilities", {})

        with self.lock:
            self.nodes[normalize_id(node_id)] = {
                "id": node_id,
                "capabilities": capabilities,
                "registeredAt": time.time(),
                "lastSeenAt": time.time(),
            }

        print()
        print("⭐ Node registered")
        print(f"   id: {node_id}")
        print(
            f"   platform: "
            f"{capabilities.get('platform', 'unknown')}"
        )
        print(
            "   hardware: "
            f"{capabilities.get('hardwareIdentifier', 'unknown')}"
        )
        print(
            f"   gpu: "
            f"{capabilities.get('gpuName', 'unknown')}"
        )

    def _requeue_expired_locked(self):
        now = time.time()

        expired = [
            work_id
            for work_id, assignment
            in self.assigned.items()
            if assignment["leaseExpiresAt"] <= now
        ]

        for work_id in expired:
            self.assigned.pop(
                work_id,
                None,
            )

            if work_id not in self.completed:
                self.pending.append(
                    work_id
                )

    def claim_work(self, node_id):
        node_key = normalize_id(node_id)

        with self.lock:
            self._requeue_expired_locked()

            if node_key in self.nodes:
                self.nodes[node_key][
                    "lastSeenAt"
                ] = time.time()

            now = time.time()

            deferred = []
            work_unit = None

            while self.pending:
                work_id = self.pending.popleft()

                if work_id in self.completed:
                    continue

                if (
                        self.retry_after.get(
                            work_id,
                            0.0,
                        )
                        > now
                ):
                    deferred.append(
                        work_id
                    )
                    continue

                candidate = self.work_units.get(
                    work_id
                )

                if candidate is None:
                    continue

                self.assigned[work_id] = {
                    "nodeID": str(node_id),
                    "assignedAt": now,
                    "leaseExpiresAt": (
                            now + LEASE_SECONDS
                    ),
                }

                work_unit = dict(
                    candidate
                )

                break

            for work_id in deferred:
                self.pending.append(
                    work_id
                )

        if work_unit is not None:
            print()
            print("📦 Science work assigned")
            print(
                f"   work: "
                f"{work_unit['id']}"
            )
            print(
                f"   node: "
                f"{node_id}"
            )
            print(
                f"   dataset: "
                f"{work_unit['datasetID']}"
            )
            print(
                "   frequency start: "
                f"{work_unit['startFrequency']:.6f}"
            )
            print(
                f"   frequencies: "
                f"{work_unit['frequencyCount']}"
            )

        return work_unit

    @staticmethod
    def _chunk_references(dataset):
        reference = dataset.get(
            "reference",
            {},
        )

        for key in (
                "chunks",
                "chunkReferences",
                "workUnits",
                "workUnitReferences",
        ):
            value = reference.get(key)

            if isinstance(
                    value,
                    list,
            ):
                return value

        return []

    def _chunk_reference(
            self,
            dataset_id,
            start_index,
    ):
        dataset = self.datasets[
            dataset_id
        ]

        for chunk in self._chunk_references(
                dataset
        ):
            chunk_start = first_value(
                chunk,
                "frequencyStartIndex",
                "startIndex",
                "index",
            )

            if (
                    chunk_start is not None
                    and int(chunk_start)
                    == int(start_index)
            ):
                return chunk

        return None

    def verification_tolerance(
            self,
            dataset_id,
            frequency_step,
    ):
        times = self.datasets[
            dataset_id
        ]["times"]

        if len(times) < 2:
            return (
                    frequency_step * 4.0
            )

        baseline_days = (
                float(times[-1])
                - float(times[0])
        )

        if baseline_days <= 0:
            return (
                    frequency_step * 4.0
            )

        peak_width = (
                1.0 / baseline_days
        )

        return max(
            frequency_step * 4.0,
            peak_width * 0.005,
            )

    def _verify_result(
            self,
            work_unit,
            result,
    ):
        if (
                result.get("status")
                != "completed"
        ):
            return (
                False,
                "Work unit did not complete.",
            )

        if (
                result.get("bestFrequency")
                is None
        ):
            return (
                False,
                "Missing best frequency.",
            )

        if (
                result.get("bestPower")
                is None
        ):
            return (
                False,
                "Missing best power.",
            )

        best_frequency = float(
            result["bestFrequency"]
        )

        best_power = float(
            result["bestPower"]
        )

        if not math.isfinite(
                best_frequency
        ):
            return (
                False,
                "Best frequency is not finite.",
            )

        if not math.isfinite(
                best_power
        ):
            return (
                False,
                "Best power is not finite.",
            )

        start_frequency = float(
            work_unit["startFrequency"]
        )

        frequency_step = float(
            work_unit["frequencyStep"]
        )

        frequency_count = int(
            work_unit["frequencyCount"]
        )

        end_frequency = (
                start_frequency
                + max(
            frequency_count - 1,
            0,
            )
                * frequency_step
        )

        grid_tolerance = max(
            abs(frequency_step) * 2.0,
            1e-7,
            )

        if (
                best_frequency
                < start_frequency
                - grid_tolerance
                or best_frequency
                > end_frequency
                + grid_tolerance
        ):
            return (
                False,
                "Best frequency is outside work-unit range.",
            )

        reference = self._chunk_reference(
            work_unit["datasetID"],
            work_unit[
                "frequencyStartIndex"
            ],
        )

        if reference is None:
            return (
                True,
                "Result accepted.",
            )

        reference_frequency = first_value(
            reference,
            "bestFrequency",
            "frequency",
        )

        if reference_frequency is None:
            return (
                True,
                "Result accepted.",
            )

        tolerance = (
            self.verification_tolerance(
                work_unit["datasetID"],
                frequency_step,
            )
        )

        if (
                abs(
                    best_frequency
                    - float(
                        reference_frequency
                    )
                )
                > tolerance
        ):
            return (
                False,
                "Frequency verification failed.",
            )

        return (
            True,
            "Result accepted.",
        )

    def submit_result(
            self,
            route_work_id,
            result,
    ):
        work_id = normalize_id(
            route_work_id
        )

        with self.lock:
            work_unit = self.work_units.get(
                work_id
            )

            if work_unit is None:
                return (
                    False,
                    "Unknown work unit.",
                    404,
                )

            if work_id in self.completed:
                return (
                    True,
                    "Result already accepted.",
                    200,
                )

            accepted, message = (
                self._verify_result(
                    work_unit,
                    result,
                )
            )

            if not accepted:
                self.assigned.pop(
                    work_id,
                    None,
                )

                self.retry_after[
                    work_id
                ] = (
                        time.time()
                        + RETRY_COOLDOWN_SECONDS
                )

                self.pending.append(
                    work_id
                )

                return (
                    False,
                    message,
                    400,
                )

            stored_result = dict(
                result
            )

            stored_result[
                "datasetID"
            ] = work_unit[
                "datasetID"
            ]

            self.completed[
                work_id
            ] = stored_result

            self.assigned.pop(
                work_id,
                None,
            )

            self.retry_after.pop(
                work_id,
                None,
            )

            completed_count = len(
                self.completed
            )

            total_count = len(
                self.work_units
            )

        print()
        print("✅ Science result accepted")
        print(
            f"   work: "
            f"{work_unit['id']}"
        )
        print(
            f"   node: "
            f"{result.get('nodeID', 'unknown')}"
        )
        print(
            f"   dataset: "
            f"{work_unit['datasetID']}"
        )
        print(
            "   frequency: "
            f"{float(result['bestFrequency']):.8f}"
        )

        if (
                result.get(
                    "bestPeriodDays"
                )
                is not None
        ):
            print(
                "   period: "
                f"{float(result['bestPeriodDays']):.8f} days"
            )

        print(
            "   power: "
            f"{float(result['bestPower']):.8f}"
        )

        if (
                result.get("duration")
                is not None
        ):
            print(
                "   duration: "
                f"{float(result['duration']):.4f}s"
            )

        print(
            f"   project: "
            f"{completed_count}/{total_count}"
        )

        self._report_completions()

        return (
            True,
            "Result accepted.",
            200,
        )

    def _dataset_counts_locked(
            self,
            dataset_id,
    ):
        work_ids = (
            self.work_ids_by_dataset.get(
                dataset_id,
                [],
            )
        )

        work_id_set = set(
            work_ids
        )

        pending_count = sum(
            1
            for work_id
            in self.pending
            if work_id
            in work_id_set
        )

        assigned_count = sum(
            1
            for work_id
            in self.assigned
            if work_id
            in work_id_set
        )

        completed_count = sum(
            1
            for work_id
            in self.completed
            if work_id
            in work_id_set
        )

        return (
            pending_count,
            assigned_count,
            completed_count,
            len(work_ids),
        )

    def _dataset_best_locked(
            self,
            dataset_id,
    ):
        best = None

        for work_id in (
                self.work_ids_by_dataset.get(
                    dataset_id,
                    [],
                )
        ):
            result = self.completed.get(
                work_id
            )

            if (
                    result is None
                    or result.get(
                "bestPower"
            )
                    is None
            ):
                continue

            if (
                    best is None
                    or float(
                result["bestPower"]
            )
                    > float(
                best["bestPower"]
            )
            ):
                best = result

        return best

    def dataset_status(
            self,
            dataset_id,
    ):
        with self.lock:
            (
                pending,
                assigned,
                completed,
                total,
            ) = (
                self._dataset_counts_locked(
                    dataset_id
                )
            )

            best = (
                self._dataset_best_locked(
                    dataset_id
                )
            )

            dataset = self.datasets[
                dataset_id
            ]

            manifest = (
                self.dataset_manifest_entries.get(
                    dataset_id,
                    {},
                )
            )

            metadata = dataset.get(
                "metadata",
                {},
            )

            work_ids = (
                self.work_ids_by_dataset.get(
                    dataset_id,
                    [],
                )
            )

            retry_count = sum(
                1
                for work_id
                in work_ids
                if (
                        work_id
                        in self.retry_after
                        and work_id
                        not in self.completed
                )
            )

            return {
                "id": dataset_id,
                "targetName": dataset.get(
                    "targetName",
                    dataset_id,
                ),
                "mission": dataset.get(
                    "mission",
                    "TESS",
                ),
                "ticID": manifest.get(
                    "ticID",
                    metadata.get("ticID"),
                ),
                "sector": manifest.get(
                    "sector",
                    metadata.get("sector"),
                ),
                "pendingWorkUnits": pending,
                "assignedWorkUnits": assigned,
                "completedWorkUnits": completed,
                "retryCount": retry_count,
                "totalWorkUnits": total,
                "progress": (
                    completed / total
                    if total
                    else 1.0
                ),
                "bestFrequency": (
                    best.get(
                        "bestFrequency"
                    )
                    if best
                    else None
                ),
                "bestPeriodDays": (
                    best.get(
                        "bestPeriodDays"
                    )
                    if best
                    else None
                ),
                "bestPower": (
                    best.get(
                        "bestPower"
                    )
                    if best
                    else None
                ),
            }

    def _current_dataset_id_locked(
            self
    ):
        dataset_ids = list(
            self.datasets.keys()
        )

        if not dataset_ids:
            return None

        for dataset_id in dataset_ids:
            (
                _,
                _,
                completed,
                total,
            ) = (
                self._dataset_counts_locked(
                    dataset_id
                )
            )

            if completed < total:
                return dataset_id

        return dataset_ids[-1]

    def project_status(self):
        with self.lock:
            self._requeue_expired_locked()

            project_pending = len(
                self.pending
            )

            project_assigned = len(
                self.assigned
            )

            project_completed = len(
                self.completed
            )

            project_total = len(
                self.work_units
            )

            dataset_statuses = [
                self.dataset_status(
                    dataset_id
                )
                for dataset_id
                in self.datasets
            ]

            current_dataset_id = (
                self._current_dataset_id_locked()
            )

            if (
                    current_dataset_id
                    is None
            ):
                current = {
                    "id": "",
                    "targetName": "",
                    "mission": "",
                    "pendingWorkUnits": 0,
                    "assignedWorkUnits": 0,
                    "completedWorkUnits": 0,
                    "retryCount": 0,
                    "totalWorkUnits": 0,
                    "progress": 1.0,
                    "bestFrequency": None,
                    "bestPeriodDays": None,
                    "bestPower": None,
                }

                sample_count = 0

            else:
                current = next(
                    status
                    for status
                    in dataset_statuses
                    if status["id"]
                    == current_dataset_id
                )

                sample_count = len(
                    self.datasets[
                        current_dataset_id
                    ].get(
                        "times",
                        [],
                    )
                )

            return {
                "projectID": self.project_id,
                "workloadID": self.workload_id,

                "datasetID": current["id"],
                "targetName": current[
                    "targetName"
                ],
                "mission": current[
                    "mission"
                ],

                "sampleCount": sample_count,
                "samples": sample_count,

                "pending": current[
                    "pendingWorkUnits"
                ],
                "assigned": current[
                    "assignedWorkUnits"
                ],
                "completed": current[
                    "completedWorkUnits"
                ],
                "total": current[
                    "totalWorkUnits"
                ],

                "pendingWorkUnits": current[
                    "pendingWorkUnits"
                ],
                "assignedWorkUnits": current[
                    "assignedWorkUnits"
                ],
                "completedWorkUnits": current[
                    "completedWorkUnits"
                ],
                "retryCount": current[
                    "retryCount"
                ],
                "totalWorkUnits": current[
                    "totalWorkUnits"
                ],

                "progress": current[
                    "progress"
                ],

                "activeNodes": len(
                    self.nodes
                ),

                "bestFrequency": current[
                    "bestFrequency"
                ],
                "bestPeriodDays": current[
                    "bestPeriodDays"
                ],
                "bestPower": current[
                    "bestPower"
                ],

                "projectPendingWorkUnits": (
                    project_pending
                ),
                "projectAssignedWorkUnits": (
                    project_assigned
                ),
                "projectCompletedWorkUnits": (
                    project_completed
                ),
                "projectRetryCount": sum(
                    1
                    for work_id
                    in self.retry_after
                    if work_id
                    not in self.completed
                ),
                "projectTotalWorkUnits": (
                    project_total
                ),
                "projectProgress": (
                    project_completed
                    / project_total
                    if project_total
                    else 1.0
                ),

                "datasets": (
                    dataset_statuses
                ),
            }

    @staticmethod
    def _global_reference(
            dataset
    ):
        reference = dataset.get(
            "reference",
            {},
        )

        return (
            first_value(
                reference,
                "bestFrequency",
                "frequency",
            ),
            first_value(
                reference,
                "bestPeriodDays",
                "periodDays",
                "period",
            ),
            first_value(
                reference,
                "bestPower",
                "power",
            ),
        )

    def _report_dataset_complete(
            self,
            dataset_id,
    ):
        with self.lock:
            if (
                    dataset_id
                    in self.reported_completed_datasets
            ):
                return

            (
                _,
                _,
                completed,
                total,
            ) = (
                self._dataset_counts_locked(
                    dataset_id
                )
            )

            if (
                    total == 0
                    or completed != total
            ):
                return

            best = (
                self._dataset_best_locked(
                    dataset_id
                )
            )

            dataset = self.datasets[
                dataset_id
            ]

            self.reported_completed_datasets.add(
                dataset_id
            )

        print()
        print("🌟 Dataset complete")
        print(
            f"   target: "
            f"{dataset.get('targetName', dataset_id)}"
        )
        print(
            f"   dataset: "
            f"{dataset_id}"
        )

        if best is None:
            return

        best_frequency = float(
            best["bestFrequency"]
        )

        best_period = best.get(
            "bestPeriodDays"
        )

        best_power = float(
            best["bestPower"]
        )

        print(
            "   OpenStar frequency: "
            f"{best_frequency:.8f} cycles/day"
        )

        if best_period is not None:
            print(
                "   OpenStar period: "
                f"{float(best_period):.8f} days"
            )

        print(
            "   OpenStar power: "
            f"{best_power:.8f}"
        )

        (
            reference_frequency,
            reference_period,
            _,
        ) = self._global_reference(
            dataset
        )

        if (
                reference_frequency
                is not None
        ):
            reference_frequency = float(
                reference_frequency
            )

            print(
                "   Astropy frequency: "
                f"{reference_frequency:.8f} cycles/day"
            )

            print(
                "   frequency error: "
                f"{abs(best_frequency - reference_frequency):.8f}"
            )

        if (
                reference_period
                is not None
                and best_period
                is not None
        ):
            reference_period = float(
                reference_period
            )

            print(
                "   Astropy period: "
                f"{reference_period:.8f} days"
            )

            print(
                "   period error: "
                f"{abs(float(best_period) - reference_period):.8f} days"
            )

    def _report_project_complete(
            self
    ):
        with self.lock:
            if (
                    self.reported_project_complete
            ):
                return

            if (
                    len(self.completed)
                    != len(self.work_units)
            ):
                return

            self.reported_project_complete = True

            results = [
                (
                    dataset_id,
                    self.datasets[
                        dataset_id
                    ].get(
                        "targetName",
                        dataset_id,
                    ),
                    self._dataset_best_locked(
                        dataset_id
                    ),
                )
                for dataset_id
                in self.datasets
            ]

        print()
        print("🏁 Project complete")
        print(
            f"   project: "
            f"{self.project_id}"
        )

        for (
                dataset_id,
                target_name,
                best,
        ) in results:
            print()
            print(
                f"   target: "
                f"{target_name}"
            )
            print(
                f"   dataset: "
                f"{dataset_id}"
            )

            if best is not None:
                print(
                    "   best frequency: "
                    f"{float(best['bestFrequency']):.8f}"
                )

                if (
                        best.get(
                            "bestPeriodDays"
                        )
                        is not None
                ):
                    print(
                        "   best period: "
                        f"{float(best['bestPeriodDays']):.8f} days"
                    )

                print(
                    "   best power: "
                    f"{float(best['bestPower']):.8f}"
                )

    def _report_completions(
            self
    ):
        for dataset_id in self.datasets:
            self._report_dataset_complete(
                dataset_id
            )

        self._report_project_complete()

    def print_startup_summary(
            self,
            port,
    ):
        print()
        print("⭐ OpenStar Coordinator")
        print(f"Build: {BUILD}")
        print(
            f"File: "
            f"{Path(__file__).resolve()}"
        )
        print(
            f"Listening on port "
            f"{port}"
        )

        print()
        print(
            f"Project: "
            f"{self.project_id}"
        )
        print(
            f"Workload: "
            f"{self.workload_id}"
        )
        print(
            f"Datasets: "
            f"{len(self.datasets)}"
        )
        print(
            f"Work units: "
            f"{len(self.work_units)}"
        )

        for (
                dataset_id,
                dataset,
        ) in self.datasets.items():
            (
                reference_frequency,
                reference_period,
                _,
            ) = self._global_reference(
                dataset
            )

            print()
            print(
                f"Target: "
                f"{dataset.get('targetName', dataset_id)}"
            )
            print(
                f"Dataset: "
                f"{dataset_id}"
            )
            print(
                f"Mission: "
                f"{dataset.get('mission', 'TESS')}"
            )
            print(
                f"Samples: "
                f"{len(dataset.get('times', []))}"
            )

            if (
                    reference_period
                    is not None
            ):
                print(
                    "Astropy reference period: "
                    f"{float(reference_period):.8f} days"
                )

            if (
                    reference_frequency
                    is not None
            ):
                print(
                    "Astropy reference frequency: "
                    f"{float(reference_frequency):.8f} cycles/day"
                )


STATE = None


class RequestHandler(
    BaseHTTPRequestHandler
):
    server_version = (
        "OpenStarCoordinator/1.0"
    )

    def log_message(
            self,
            format,
            *args,
    ):
        return

    def _read_json(self):
        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        if content_length <= 0:
            return {}

        body = self.rfile.read(
            content_length
        )

        if not body:
            return {}

        return json.loads(
            body.decode("utf-8")
        )

    def _send_json(
            self,
            status_code,
            payload,
    ):
        body = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        self.send_response(
            status_code
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    def _send_no_content(self):
        self.send_response(
            204
        )

        self.send_header(
            "Content-Length",
            "0",
        )

        self.end_headers()

    def _send_error_json(
            self,
            status_code,
            message,
    ):
        self._send_json(
            status_code,
            {
                "accepted": False,
                "message": message,
            },
        )

    def do_GET(self):
        global STATE

        path = urlparse(
            self.path
        ).path

        print(
            f"🌐 GET {path}"
        )

        if (
                path
                == "/v1/projects/current/status"
        ):
            status = (
                STATE.project_status()
            )

            print(
                "   status targetName: "
                f"{status.get('targetName')}"
            )

            print(
                "   status datasetID: "
                f"{status.get('datasetID')}"
            )

            print(
                "   status retryCount: "
                f"{status.get('retryCount')}"
            )

            self._send_json(
                200,
                status,
            )

            return

        dataset_prefix = (
            "/v1/datasets/"
        )

        if path.startswith(
                dataset_prefix
        ):
            dataset_id = unquote(
                path[
                    len(
                        dataset_prefix
                    ):
                ]
            )

            dataset = (
                STATE.datasets.get(
                    dataset_id
                )
            )

            if dataset is None:
                self._send_error_json(
                    404,
                    "Unknown dataset.",
                )
                return

            print(
                "   dataset targetName: "
                f"{dataset.get('targetName')}"
            )

            self._send_json(
                200,
                dataset,
            )

            return

        self._send_error_json(
            404,
            "Not found.",
        )

    def do_POST(self):
        global STATE

        path = urlparse(
            self.path
        ).path

        print(
            f"🌐 POST {path}"
        )

        try:
            payload = (
                self._read_json()
            )

        except (
                json.JSONDecodeError,
                UnicodeDecodeError,
        ):
            self._send_error_json(
                400,
                "Invalid JSON.",
            )
            return

        if (
                path
                == "/v1/nodes/register"
        ):
            try:
                STATE.register_node(
                    payload
                )

            except KeyError:
                self._send_error_json(
                    400,
                    "Missing node ID.",
                )
                return

            self._send_json(
                200,
                {
                    "accepted": True,
                    "message": (
                        "Node registered."
                    ),
                },
            )

            return

        if (
                path
                == "/v1/work/claim"
        ):
            node_id = first_value(
                payload,
                "nodeID",
                "nodeId",
                "id",
            )

            if node_id is None:
                self._send_error_json(
                    400,
                    "Missing node ID.",
                )
                return

            work_unit = (
                STATE.claim_work(
                    node_id
                )
            )

            if work_unit is None:
                self._send_no_content()
                return

            self._send_json(
                200,
                work_unit,
            )

            return

        result_prefix = (
            "/v1/work/"
        )

        result_suffix = (
            "/result"
        )

        if (
                path.startswith(
                    result_prefix
                )
                and path.endswith(
            result_suffix
        )
        ):
            work_id = path[
                len(
                    result_prefix
                ):
                -len(
                    result_suffix
                )
            ].strip("/")

            if not work_id:
                self._send_error_json(
                    400,
                    "Missing work unit ID.",
                )
                return

            (
                accepted,
                message,
                status_code,
            ) = (
                STATE.submit_result(
                    unquote(
                        work_id
                    ),
                    payload,
                )
            )

            self._send_json(
                status_code,
                {
                    "accepted": accepted,
                    "message": message,
                },
            )

            return

        self._send_error_json(
            404,
            "Not found.",
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "OpenStar coordinator"
        )
    )

    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT_PATH,
        help=(
            "Path to generated "
            "project manifest."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help=(
            "Coordinator HTTP port."
        ),
    )

    return parser.parse_args()


def main():
    global STATE

    args = parse_args()

    STATE = CoordinatorState(
        args.project
    )

    STATE.print_startup_summary(
        args.port
    )

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            args.port,
        ),
        RequestHandler,
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print()
        print(
            "Stopping OpenStar Coordinator."
        )

    finally:
        server.server_close()


if __name__ == "__main__":
    main()