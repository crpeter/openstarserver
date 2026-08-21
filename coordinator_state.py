import json
import math
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from frequency_uncertainty import estimate_frequency_interval


LEASE_SECONDS = 120.0
RETRY_COOLDOWN_SECONDS = 1.0
EXECUTION_FAILURE_NODE_AVOID_SECONDS = 30.0
NODE_EXECUTION_FAILURE_STREAK_LIMIT = 3
NODE_EXECUTION_COOLDOWN_SECONDS = 60.0
SCIENTIFIC_FAILURE_NODE_AVOID_SECONDS = 30.0
CROSS_NODE_FREQUENCY_TOLERANCE_STEPS = 2.0
CROSS_NODE_POWER_RELATIVE_TOLERANCE = 0.005

# Chunk verification is reduction-first:
#
#   1. If OpenStar and Astropy select the same peak frequency within the
#      dataset-derived primary tolerance, accept the result. Peak power is
#      diagnostic on this path.
#
#   2. Otherwise, if OpenStar reports effectively the same maximum power for
#      the assigned chunk, accept it even when the argmax frequency differs.
#      In weak/flat chunks, Float32 Metal and Float64 Astropy can choose
#      different bins while agreeing on the chunk's maximum contribution to
#      the global reduction.
#
#   3. As a final fallback, a nearby peak inside the narrow ambiguity band may
#      use a looser power tolerance.
#
#   4. Everything else is rejected.
EQUIVALENT_POWER_RELATIVE_TOLERANCE = 0.005
AMBIGUOUS_POWER_RELATIVE_TOLERANCE = 0.03
POWER_ABSOLUTE_TOLERANCE = 1.0e-5

# A deterministic scientific mismatch must never spin forever.
MAX_VERIFICATION_FAILURES_PER_WORK_UNIT = 3

# Dataset-level interpretation. These do NOT change chunk verification.
#
# The purpose is to stop a mathematically largest but scientifically weak
# Lomb-Scargle peak from automatically becoming an authoritative period.
PERIOD_RELIABLE_MIN_POWER = 0.01
PERIOD_RELIABLE_MIN_FOLD_COHERENCE = 0.02
PERIOD_LOW_CONFIDENCE_POWER = 0.02
PERIOD_LOW_CONFIDENCE_FOLD_COHERENCE = 0.05

# A doubled/halved period is only promoted above the raw LS period when its
# folded light curve is materially more coherent.
HARMONIC_PREFERENCE_MIN_COHERENCE_GAIN = 0.15
PERIOD_FOLD_BINS = 100
PERIOD_MIN_POINTS_PER_BIN = 3
PERIOD_INDEPENDENT_CANDIDATE_COUNT = 5


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

        self.project_path = Path(project_path).resolve()
        self.project = self._load_json(self.project_path)

        self.project_id = self.project["id"]
        self.project_name = self.project.get("name", self.project_id)
        self.workload_id = self.project["workloadID"]

        self.datasets = {}
        self.dataset_manifest_entries = {}

        self.work_units = {}
        self.work_ids_by_dataset = {}
        self.chunk_references_by_dataset = {}

        self.pending = deque()
        self.assigned = {}
        self.completed = {}
        # Original, normalized client submissions are retained separately from
        # server-enriched results so transport retries can be compared safely.
        self.accepted_result_submissions = {}
        self.failed = {}
        self.retry_after = {}
        self.retry_counts = {}
        self.verification_failure_counts = {}
        self.reference_mismatch_counts = {}
        self.execution_failure_counts = {}
        self.execution_failure_history = {}
        self.environment_unavailable_counts = {}
        self.environment_unavailable_history = {}
        self.transport_unavailable_counts = {}
        self.transport_unavailable_history = {}
        self.execution_avoid_until = {}
        self.node_execution_failure_streaks = {}
        self.node_execution_cooldown_until = {}
        self.scientific_rejections = {}

        self.nodes = {}

        self.reported_completed_datasets = set()
        self.reported_project_complete = False
        # Operational-only hooks/timestamps.  They are intentionally absent
        # from project status and all persisted scientific artifacts.
        self.terminal_observer = None
        self.terminal_monotonic = None
        self._terminal_observer_notified = False

        # Dataset interpretation is calculated only after a dataset becomes
        # terminal, then cached. Status polling therefore stays cheap.
        self.dataset_diagnostic_cache = {}

        self._load_datasets()
        self._index_chunk_references()
        self._build_work_units()

    @staticmethod
    def _load_json(path):
        with Path(path).open("r", encoding="utf-8") as file:
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
        dataset_entries = self.project.get("datasets", [])
        if not dataset_entries:
            raise RuntimeError("Project manifest contains no datasets.")

        for entry in dataset_entries:
            dataset_id = str(entry["id"])

            if dataset_id in self.datasets:
                raise RuntimeError(f"Duplicate dataset id: {dataset_id}")

            dataset_path = self._resolve_dataset_path(entry)
            if not dataset_path.exists():
                raise RuntimeError(
                    f"Dataset file does not exist for {dataset_id}: {dataset_path}"
                )

            dataset = self._load_json(dataset_path)

            embedded_id = dataset.get("id")
            if embedded_id is not None and str(embedded_id) != dataset_id:
                raise RuntimeError(
                    f"Dataset id mismatch: manifest={dataset_id}, file={embedded_id}"
                )

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

        frequency_step = first_value(search, "frequencyStep", "step")

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
                else (maximum_frequency - minimum_frequency) / total_frequencies
            )

        frequency_step = float(frequency_step)

        if not math.isfinite(minimum_frequency):
            raise RuntimeError("minimumFrequency must be finite.")
        if not math.isfinite(frequency_step) or frequency_step <= 0:
            raise RuntimeError("frequencyStep must be finite and > 0.")
        if total_frequencies <= 0:
            raise RuntimeError("totalFrequencies must be > 0.")
        if frequencies_per_work_unit <= 0:
            raise RuntimeError("frequenciesPerWorkUnit must be > 0.")

        return {
            "minimumFrequency": minimum_frequency,
            "frequencyStep": frequency_step,
            "totalFrequencies": total_frequencies,
            "frequenciesPerWorkUnit": frequencies_per_work_unit,
        }

    @staticmethod
    def _chunk_references(dataset):
        reference = dataset.get("reference", {})

        for key in (
            "chunks",
            "chunkReferences",
            "workUnits",
            "workUnitReferences",
        ):
            value = reference.get(key)
            if isinstance(value, list):
                return value

        return []

    def _index_chunk_references(self):
        self.chunk_references_by_dataset = {}

        for dataset_id, dataset in self.datasets.items():
            indexed = {}

            for chunk in self._chunk_references(dataset):
                start_index = first_value(
                    chunk,
                    "frequencyStartIndex",
                    "startIndex",
                    "index",
                )

                if start_index is None:
                    continue

                start_index = int(start_index)

                if start_index in indexed:
                    raise RuntimeError(
                        f"{dataset_id}: duplicate Astropy chunk reference at "
                        f"frequencyStartIndex={start_index}"
                    )

                indexed[start_index] = chunk

            self.chunk_references_by_dataset[dataset_id] = indexed

    def _build_work_units(self):
        for dataset_id, dataset in self.datasets.items():
            search = self._frequency_search(dataset)

            minimum_frequency = search["minimumFrequency"]
            frequency_step = search["frequencyStep"]
            total_frequencies = search["totalFrequencies"]
            frequencies_per_work_unit = search["frequenciesPerWorkUnit"]

            self.work_ids_by_dataset[dataset_id] = []

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

                start_frequency = (
                    minimum_frequency + start_index * frequency_step
                )

                # Generic workload envelope. New worker implementations read
                # workload-specific parameters from payload. The flattened
                # frequency fields remain temporarily for v18 client/server
                # compatibility and can be removed after migration.
                workload_payload = {
                    "frequencyStartIndex": start_index,
                    "startFrequency": start_frequency,
                    "frequencyStep": frequency_step,
                    "frequencyCount": frequency_count,
                }

                work_unit = {
                    "id": work_id,
                    "projectID": self.project_id,
                    "workloadID": self.workload_id,
                    "datasetID": dataset_id,
                    "payload": workload_payload,
                    "frequencyStartIndex": start_index,
                    "startFrequency": start_frequency,
                    "frequencyStep": frequency_step,
                    "frequencyCount": frequency_count,
                }

                self.work_units[normalized_work_id] = work_unit
                self.work_ids_by_dataset[dataset_id].append(normalized_work_id)
                self.pending.append(normalized_work_id)
                self.retry_counts[normalized_work_id] = 0
                self.verification_failure_counts[normalized_work_id] = 0
                self.reference_mismatch_counts[normalized_work_id] = 0

    @staticmethod
    def _global_reference(dataset):
        reference = dataset.get("reference", {})

        return (
            first_value(reference, "bestFrequency", "frequency"),
            first_value(
                reference,
                "bestPeriodDays",
                "periodDays",
                "period",
            ),
            first_value(reference, "bestPower", "power"),
        )

    def _validate_reference_data(self):
        """
        Astropy/reference data is optional.

        Production/discovery projects may contain no reference data at all.
        When reference data is present, validate only the supplied values so
        they can be used as diagnostics. Missing references never block a run.
        """
        errors = []

        for dataset_id, dataset in self.datasets.items():
            search = self._frequency_search(dataset)

            global_frequency, global_period, global_power = self._global_reference(
                dataset
            )

            for field_name, value in (
                ("global Astropy frequency reference", global_frequency),
                ("global Astropy period reference", global_period),
                ("global Astropy power reference", global_power),
            ):
                if value is None:
                    continue

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    errors.append(f"{dataset_id}: {field_name} is not numeric")
                    continue

                if not math.isfinite(numeric_value):
                    errors.append(f"{dataset_id}: {field_name} is not finite")

            expected_starts = set(
                range(
                    0,
                    search["totalFrequencies"],
                    search["frequenciesPerWorkUnit"],
                )
            )

            indexed = self.chunk_references_by_dataset.get(dataset_id, {})
            extra_starts = sorted(set(indexed.keys()) - expected_starts)

            if extra_starts:
                errors.append(
                    f"{dataset_id}: unexpected Astropy chunk reference start "
                    "indexes: "
                    + ", ".join(str(value) for value in extra_starts[:10])
                )

            for start_index, chunk in indexed.items():
                if start_index not in expected_starts:
                    continue

                expected_count = min(
                    search["frequenciesPerWorkUnit"],
                    search["totalFrequencies"] - start_index,
                )

                chunk_count = first_value(
                    chunk,
                    "frequencyCount",
                    "count",
                    default=expected_count,
                )

                try:
                    chunk_count = int(chunk_count)
                except (TypeError, ValueError):
                    errors.append(
                        f"{dataset_id}: chunk {start_index} frequencyCount "
                        "is not an integer"
                    )
                else:
                    if chunk_count != expected_count:
                        errors.append(
                            f"{dataset_id}: chunk {start_index} frequencyCount "
                            f"is {chunk_count}, expected {expected_count}"
                        )

                for field_name, keys in (
                    ("bestFrequency", ("bestFrequency", "frequency")),
                    ("bestPeriodDays", ("bestPeriodDays", "periodDays", "period")),
                    ("bestPower", ("bestPower", "power")),
                ):
                    value = first_value(chunk, *keys)

                    if value is None:
                        errors.append(
                            f"{dataset_id}: chunk {start_index} missing "
                            f"{field_name}"
                        )
                        continue

                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        errors.append(
                            f"{dataset_id}: chunk {start_index} {field_name} "
                            "is not numeric"
                        )
                        continue

                    if not math.isfinite(numeric_value):
                        errors.append(
                            f"{dataset_id}: chunk {start_index} {field_name} "
                            "is not finite"
                        )

        if errors:
            raise RuntimeError(
                "Optional Astropy reference data is malformed:\n - "
                + "\n - ".join(errors)
            )

    def _validate_science_metadata(self):
        errors = []

        for dataset_id, dataset in self.datasets.items():
            science = dataset.get("science", {})
            role = science.get("role")

            if role not in (None, "known", "control", "blind", "discovery"):
                errors.append(
                    f"{dataset_id}: science.role must be known, control, blind, "
                    "discovery, or omitted"
                )
                continue

            if role == "known":
                if not science.get("classification"):
                    errors.append(
                        f"{dataset_id}: known target is missing classification"
                    )

                published_period = science.get("publishedPeriodDays")
                if published_period is None:
                    errors.append(
                        f"{dataset_id}: known target is missing publishedPeriodDays"
                    )
                else:
                    try:
                        published_period = float(published_period)
                    except (TypeError, ValueError):
                        errors.append(
                            f"{dataset_id}: publishedPeriodDays is not numeric"
                        )
                    else:
                        if not math.isfinite(published_period) or published_period <= 0:
                            errors.append(
                                f"{dataset_id}: publishedPeriodDays must be finite and > 0"
                            )

            if role == "blind":
                forbidden = (
                    "classification",
                    "publishedPeriodDays",
                    "publishedFrequency",
                    "answerKeySource",
                )

                for key in forbidden:
                    if science.get(key) is not None:
                        errors.append(
                            f"{dataset_id}: blind target contains forbidden field {key}"
                        )

        if errors:
            raise RuntimeError(
                "Scientific validation metadata failed:\n - "
                + "\n - ".join(errors)
            )

    def validate_startup(self):
        self._validate_reference_data()
        self._validate_science_metadata()

    def register_node(self, payload):
        node_id = str(payload["nodeID"])
        capabilities = payload.get("capabilities", {})

        now = time.time()

        with self.lock:
            self.nodes[normalize_id(node_id)] = {
                "id": node_id,
                "capabilities": capabilities,
                "registeredAt": now,
                "lastSeenAt": now,
            }

        print()
        print("⭐ Node registered")
        print(f"   id: {node_id}")
        print(f"   platform: {capabilities.get('platform', 'unknown')}")
        print(
            "   hardware: "
            f"{capabilities.get('hardwareIdentifier', 'unknown')}"
        )
        print(f"   gpu: {capabilities.get('gpuName', 'unknown')}")

        advertised = capabilities.get("workloads", [])
        if isinstance(advertised, list) and advertised:
            workload_ids = []
            for item in advertised:
                if isinstance(item, str):
                    workload_ids.append(item)
                elif isinstance(item, dict) and item.get("workloadID"):
                    workload_ids.append(str(item["workloadID"]))

            if workload_ids:
                print("   workloads: " + ", ".join(workload_ids))

    def _mark_node_seen_locked(self, node_id):
        node_key = normalize_id(node_id)
        if node_key in self.nodes:
            self.nodes[node_key]["lastSeenAt"] = time.time()

    def _node_supported_workload_ids_locked(self, node_id):
        """Return advertised workload ids, or None for a legacy unrestricted node."""
        node = self.nodes.get(normalize_id(node_id))
        if node is None:
            return set()

        capabilities = node.get("capabilities", {})
        advertised = capabilities.get("workloads")

        # Backward compatibility: nodes registered before workload capability
        # advertisement are allowed to claim work exactly as before.
        if not isinstance(advertised, list) or not advertised:
            return None

        workload_ids = set()

        for item in advertised:
            if isinstance(item, str):
                workload_id = item
            elif isinstance(item, dict):
                workload_id = item.get("workloadID")
            else:
                workload_id = None

            if workload_id is not None:
                workload_ids.add(str(workload_id))

        return workload_ids

    def _node_supports_workload_locked(self, node_id, workload_id):
        supported = self._node_supported_workload_ids_locked(node_id)

        if supported is None:
            return True

        return str(workload_id) in supported

    def _node_execution_cooldown_remaining_locked(self, node_id, now=None):
        node_key = normalize_id(node_id)
        if now is None:
            now = time.time()

        cooldown_until = self.node_execution_cooldown_until.get(
            node_key,
            0.0,
        )

        if cooldown_until <= now:
            self.node_execution_cooldown_until.pop(node_key, None)
            return 0.0

        return cooldown_until - now

    def _record_node_execution_failure_locked(self, node_id, now=None):
        node_key = normalize_id(node_id)
        if now is None:
            now = time.time()

        streak = self.node_execution_failure_streaks.get(node_key, 0) + 1
        self.node_execution_failure_streaks[node_key] = streak

        cooldown_applied = False

        if streak >= NODE_EXECUTION_FAILURE_STREAK_LIMIT:
            self.node_execution_cooldown_until[node_key] = (
                now + NODE_EXECUTION_COOLDOWN_SECONDS
            )
            cooldown_applied = True

        return streak, cooldown_applied

    def _record_node_execution_success_locked(self, node_id):
        node_key = normalize_id(node_id)
        self.node_execution_failure_streaks.pop(node_key, None)
        self.node_execution_cooldown_until.pop(node_key, None)

    def _requeue_expired_locked(self):
        now = time.time()

        expired = [
            work_id
            for work_id, assignment in self.assigned.items()
            if assignment["leaseExpiresAt"] <= now
        ]

        for work_id in expired:
            self.assigned.pop(work_id, None)

            if work_id not in self.completed and work_id not in self.failed:
                self.retry_counts[work_id] = self.retry_counts.get(work_id, 0) + 1
                self.pending.append(work_id)

    def _dataset_counts_locked(self, dataset_id):
        work_ids = self.work_ids_by_dataset.get(dataset_id, [])
        work_id_set = set(work_ids)

        pending_count = sum(
            1 for work_id in self.pending if work_id in work_id_set
        )
        assigned_count = sum(
            1 for work_id in self.assigned if work_id in work_id_set
        )
        completed_count = sum(
            1 for work_id in self.completed if work_id in work_id_set
        )

        return (
            pending_count,
            assigned_count,
            completed_count,
            len(work_ids),
        )

    def _dataset_retry_count_locked(self, dataset_id):
        return sum(
            self.retry_counts.get(work_id, 0)
            for work_id in self.work_ids_by_dataset.get(dataset_id, [])
        )

    def _dataset_verification_failure_count_locked(self, dataset_id):
        return sum(
            self.verification_failure_counts.get(work_id, 0)
            for work_id in self.work_ids_by_dataset.get(dataset_id, [])
        )

    def _dataset_reference_mismatch_count_locked(self, dataset_id):
        return sum(
            self.reference_mismatch_counts.get(work_id, 0)
            for work_id in self.work_ids_by_dataset.get(dataset_id, [])
        )

    def _dataset_execution_failure_count_locked(self, dataset_id):
        return sum(
            self.execution_failure_counts.get(work_id, 0)
            for work_id in self.work_ids_by_dataset.get(dataset_id, [])
        )

    def _dataset_execution_failure_kinds_locked(self, dataset_id):
        counts = {}

        for work_id in self.work_ids_by_dataset.get(dataset_id, []):
            for record in self.execution_failure_history.get(work_id, []):
                failure_kind = str(
                    record.get("failureKind") or "unknown"
                )
                counts[failure_kind] = counts.get(failure_kind, 0) + 1

        return counts

    def _dataset_environment_unavailable_count_locked(self, dataset_id):
        return sum(
            self.environment_unavailable_counts.get(work_id, 0)
            for work_id in self.work_ids_by_dataset.get(dataset_id, [])
        )

    def _dataset_transport_unavailable_count_locked(self, dataset_id):
        return sum(
            self.transport_unavailable_counts.get(work_id, 0)
            for work_id in self.work_ids_by_dataset.get(dataset_id, [])
        )

    def _dataset_failed_count_locked(self, dataset_id):
        work_ids = set(self.work_ids_by_dataset.get(dataset_id, []))
        return sum(1 for work_id in self.failed if work_id in work_ids)

    def _current_dataset_id_locked(self):
        dataset_ids = list(self.datasets.keys())

        if not dataset_ids:
            return None

        for dataset_id in dataset_ids:
            _, _, completed, total = self._dataset_counts_locked(dataset_id)
            failed = self._dataset_failed_count_locked(dataset_id)

            # Hard-failed work is terminal for scheduling, but remains failed.
            if completed + failed < total:
                return dataset_id

        return dataset_ids[-1]

    def claim_work(self, node_id):
        work_units = self.claim_work_batch(node_id, 1)
        return work_units[0] if work_units else None

    def claim_work_batch(self, node_id, max_work_units):
        """Lease up to ``max_work_units`` mutually compatible atomic units.

        Compatibility is deliberately limited to generic routing identifiers;
        interpretation of the workload payload remains the worker plugin's job.
        Every selected unit receives its own normal lease entry.
        """
        if isinstance(max_work_units, bool) or not isinstance(max_work_units, int):
            raise ValueError("max_work_units must be a positive integer.")
        if max_work_units < 1:
            raise ValueError("max_work_units must be a positive integer.")

        node_key = normalize_id(node_id)

        with self.lock:
            self._requeue_expired_locked()
            self._mark_node_seen_locked(node_id)

            if node_key not in self.nodes:
                return None

            now = time.time()

            # A node that repeatedly fails to execute work is temporarily
            # quarantined from ALL work, not just the individual chunks it
            # already failed. This prevents a broken/flaky device from
            # burning through hundreds of unique work units while healthy
            # nodes are available. After cooldown, one new claim acts as a
            # probe; a successful completed execution clears the streak.
            if self._node_execution_cooldown_remaining_locked(
                node_id,
                now,
            ) > 0.0:
                return None

            current_dataset_id = self._current_dataset_id_locked()
            if current_dataset_id is None:
                return None

            deferred = []
            work_units = []
            compatibility_key = None

            pending_length = len(self.pending)

            for _ in range(pending_length):
                work_id = self.pending.popleft()

                if work_id in self.completed or work_id in self.failed:
                    continue

                candidate = self.work_units.get(work_id)
                if candidate is None:
                    continue

                if not self._node_supports_workload_locked(
                    node_id,
                    candidate.get("workloadID"),
                ):
                    deferred.append(work_id)
                    continue

                if candidate["datasetID"] != current_dataset_id:
                    deferred.append(work_id)
                    continue

                if self.retry_after.get(work_id, 0.0) > now:
                    deferred.append(work_id)
                    continue

                # Retry avoidance applies to both operational execution
                # failures and scientific verification failures. If this node
                # recently failed this exact work unit, defer it so another
                # registered node gets a chance. Avoidance expires
                # automatically so a single-node project cannot deadlock.
                avoid_by_node = self.execution_avoid_until.get(work_id)
                if avoid_by_node:
                    expired_nodes = [
                        avoided_node
                        for avoided_node, avoid_until in avoid_by_node.items()
                        if avoid_until <= now
                    ]
                    for avoided_node in expired_nodes:
                        avoid_by_node.pop(avoided_node, None)

                    if not avoid_by_node:
                        self.execution_avoid_until.pop(work_id, None)
                    elif avoid_by_node.get(node_key, 0.0) > now:
                        deferred.append(work_id)
                        continue

                candidate_key = (
                    candidate.get("projectID"),
                    candidate.get("workloadID"),
                    candidate.get("datasetID"),
                )
                if compatibility_key is not None and candidate_key != compatibility_key:
                    deferred.append(work_id)
                    continue

                self.assigned[work_id] = {
                    "nodeID": str(node_id),
                    "assignedAt": now,
                    "leaseExpiresAt": now + LEASE_SECONDS,
                }

                work_units.append(dict(candidate))
                if compatibility_key is None:
                    compatibility_key = candidate_key
                if len(work_units) >= max_work_units:
                    break

            for work_id in deferred:
                self.pending.append(work_id)

        return work_units

    def _chunk_reference(self, dataset_id, start_index):
        return self.chunk_references_by_dataset.get(dataset_id, {}).get(
            int(start_index)
        )

    def verification_tolerance(self, dataset_id, frequency_step):
        times = self.datasets[dataset_id].get("times", [])

        if len(times) < 2:
            return abs(frequency_step) * 4.0

        baseline_days = float(times[-1]) - float(times[0])

        if baseline_days <= 0:
            return abs(frequency_step) * 4.0

        peak_width = 1.0 / baseline_days

        return max(
            abs(frequency_step) * 4.0,
            peak_width * 0.005,
        )

    def ambiguity_frequency_tolerance(self, dataset_id, frequency_step):
        times = self.datasets[dataset_id].get("times", [])
        primary_tolerance = self.verification_tolerance(
            dataset_id,
            frequency_step,
        )

        if len(times) < 2:
            return max(
                primary_tolerance,
                abs(frequency_step) * 8.0,
            )

        baseline_days = float(times[-1]) - float(times[0])

        if baseline_days <= 0:
            return max(
                primary_tolerance,
                abs(frequency_step) * 8.0,
            )

        peak_width = 1.0 / baseline_days

        return max(
            primary_tolerance,
            abs(frequency_step) * 8.0,
            peak_width * 0.02,
        )

    def _validate_metal_result(self, work_unit, result):
        """
        Validate the result itself, independent of any external reference.

        These checks are production execution checks: completed status,
        numeric finite values, and a winning frequency inside the assigned
        frequency range.
        """
        details = {
            "deviceFrequency": None,
            "devicePower": None,
        }

        if result.get("status") != "completed":
            return False, "Work unit did not complete.", details

        if result.get("bestFrequency") is None:
            return False, "Missing best frequency.", details

        if result.get("bestPower") is None:
            return False, "Missing best power.", details

        try:
            best_frequency = float(result["bestFrequency"])
            best_power = float(result["bestPower"])
        except (TypeError, ValueError):
            return False, "Best frequency/power must be numeric.", details

        details["deviceFrequency"] = best_frequency
        details["devicePower"] = best_power

        if not math.isfinite(best_frequency):
            return False, "Best frequency is not finite.", details

        if not math.isfinite(best_power):
            return False, "Best power is not finite.", details

        start_frequency = float(work_unit["startFrequency"])
        frequency_step = float(work_unit["frequencyStep"])
        frequency_count = int(work_unit["frequencyCount"])

        end_frequency = (
            start_frequency
            + max(frequency_count - 1, 0) * frequency_step
        )

        grid_tolerance = max(abs(frequency_step) * 2.0, 1e-7)

        if (
            best_frequency < start_frequency - grid_tolerance
            or best_frequency > end_frequency + grid_tolerance
        ):
            return False, "Best frequency is outside work-unit range.", details

        return True, "Metal result is structurally valid.", details

    def _reference_comparison(self, work_unit, result):
        """
        Optional diagnostic comparison against a frozen Astropy chunk result.

        This function never decides whether the Metal result is accepted.
        """
        reference = self._chunk_reference(
            work_unit["datasetID"],
            work_unit["frequencyStartIndex"],
        )

        if reference is None:
            return {
                "status": "not-available",
                "matched": None,
                "method": None,
                "message": "No Astropy chunk reference supplied.",
                "details": {},
            }

        matched, message, details = self._verify_result(
            work_unit,
            result,
        )

        return {
            "status": "match" if matched else "mismatch",
            "matched": matched,
            "method": details.get("method"),
            "message": message,
            "details": dict(details),
        }

    def _verify_result(self, work_unit, result):
        details = {
            "method": None,
            "deviceFrequency": None,
            "astropyFrequency": None,
            "frequencyError": None,
            "frequencyTolerance": None,
            "ambiguityFrequencyTolerance": None,
            "devicePower": None,
            "astropyPower": None,
            "powerError": None,
            "powerTolerance": None,
        }

        if result.get("status") != "completed":
            return False, "Work unit did not complete.", details

        if result.get("bestFrequency") is None:
            return False, "Missing best frequency.", details

        if result.get("bestPower") is None:
            return False, "Missing best power.", details

        try:
            best_frequency = float(result["bestFrequency"])
            best_power = float(result["bestPower"])
        except (TypeError, ValueError):
            return False, "Best frequency/power must be numeric.", details

        details["deviceFrequency"] = best_frequency
        details["devicePower"] = best_power

        if not math.isfinite(best_frequency):
            return False, "Best frequency is not finite.", details

        if not math.isfinite(best_power):
            return False, "Best power is not finite.", details

        start_frequency = float(work_unit["startFrequency"])
        frequency_step = float(work_unit["frequencyStep"])
        frequency_count = int(work_unit["frequencyCount"])

        end_frequency = (
            start_frequency
            + max(frequency_count - 1, 0) * frequency_step
        )

        grid_tolerance = max(abs(frequency_step) * 2.0, 1e-7)

        if (
            best_frequency < start_frequency - grid_tolerance
            or best_frequency > end_frequency + grid_tolerance
        ):
            return False, "Best frequency is outside work-unit range.", details

        reference = self._chunk_reference(
            work_unit["datasetID"],
            work_unit["frequencyStartIndex"],
        )

        if reference is None:
            return (
                False,
                "Matching Astropy chunk reference was not found.",
                details,
            )

        reference_frequency = first_value(
            reference,
            "bestFrequency",
            "frequency",
        )
        reference_power = first_value(
            reference,
            "bestPower",
            "power",
        )

        if reference_frequency is None:
            return (
                False,
                "Astropy chunk reference is missing best frequency.",
                details,
            )

        if reference_power is None:
            return (
                False,
                "Astropy chunk reference is missing best power.",
                details,
            )

        try:
            reference_frequency = float(reference_frequency)
            reference_power = float(reference_power)
        except (TypeError, ValueError):
            return (
                False,
                "Astropy chunk reference frequency/power is not numeric.",
                details,
            )

        if not math.isfinite(reference_frequency):
            return False, "Astropy best frequency is not finite.", details

        if not math.isfinite(reference_power):
            return False, "Astropy best power is not finite.", details

        frequency_tolerance = self.verification_tolerance(
            work_unit["datasetID"],
            frequency_step,
        )
        ambiguity_frequency_tolerance = self.ambiguity_frequency_tolerance(
            work_unit["datasetID"],
            frequency_step,
        )

        frequency_error = abs(best_frequency - reference_frequency)
        power_error = abs(best_power - reference_power)
        equivalent_power_tolerance = max(
            POWER_ABSOLUTE_TOLERANCE,
            abs(reference_power)
            * EQUIVALENT_POWER_RELATIVE_TOLERANCE,
        )
        ambiguity_power_tolerance = max(
            POWER_ABSOLUTE_TOLERANCE,
            abs(reference_power)
            * AMBIGUOUS_POWER_RELATIVE_TOLERANCE,
        )

        details.update(
            {
                "astropyFrequency": reference_frequency,
                "frequencyError": frequency_error,
                "frequencyTolerance": frequency_tolerance,
                "ambiguityFrequencyTolerance": (
                    ambiguity_frequency_tolerance
                ),
                "astropyPower": reference_power,
                "powerError": power_error,
                "powerTolerance": equivalent_power_tolerance,
                "ambiguityPowerTolerance": ambiguity_power_tolerance,
            }
        )

        # Primary path: if Metal and Astropy identify the same scientific peak,
        # frequency agreement is authoritative. Power remains diagnostic.
        if frequency_error <= frequency_tolerance:
            details["method"] = "frequency"
            return (
                True,
                "Result accepted by frequency agreement; peak power recorded "
                "diagnostically.",
                details,
            )

        # Secondary path: for a distributed max reduction, an effectively
        # identical chunk maximum is also a valid result. Weak/flat chunks can
        # have unstable argmax locations even when the maximum power itself is
        # numerically equivalent. This path is intentionally strict on power.
        if power_error <= equivalent_power_tolerance:
            details["method"] = "equivalent-power"
            return (
                True,
                "Result accepted by equivalent chunk maximum power; argmax "
                "frequency differs in an ambiguous/flat chunk.",
                details,
            )

        # Tertiary path: a nearby local maximum may use a looser power check,
        # but only inside the narrow dataset-derived ambiguity band.
        if (
            frequency_error <= ambiguity_frequency_tolerance
            and power_error <= ambiguity_power_tolerance
        ):
            details["method"] = "nearby-peak"
            details["powerTolerance"] = ambiguity_power_tolerance
            return (
                True,
                "Result accepted as a nearby ambiguous peak with similar "
                "power.",
                details,
            )

        return (
            False,
            "Astropy frequency/power verification failed: "
            f"deviceFrequency={best_frequency:.8f}, "
            f"astropyFrequency={reference_frequency:.8f}, "
            f"frequencyError={frequency_error:.8f}, "
            f"primaryTolerance={frequency_tolerance:.8f}, "
            f"ambiguityTolerance={ambiguity_frequency_tolerance:.8f}, "
            f"devicePower={best_power:.8f}, "
            f"astropyPower={reference_power:.8f}, "
            f"powerError={power_error:.8f}, "
            f"equivalentPowerTolerance={equivalent_power_tolerance:.8f}, "
            f"nearbyPowerTolerance={ambiguity_power_tolerance:.8f}.",
            details,
        )

    def _try_cross_node_consensus_locked(
        self,
        work_id,
        work_unit,
        assignment,
        result,
        verification,
    ):
        # Cross-node consensus is only a fallback for a result that already
        # failed strict Astropy verification but still lies inside Astropy's
        # narrow ambiguity-frequency band. It never broadens that band.
        current_frequency_error = verification.get("frequencyError")
        ambiguity_tolerance = verification.get("ambiguityFrequencyTolerance")

        if (
            current_frequency_error is None
            or ambiguity_tolerance is None
            or float(current_frequency_error) > float(ambiguity_tolerance)
        ):
            return False, None, None

        try:
            current_frequency = float(result["bestFrequency"])
            current_power = float(result["bestPower"])
        except (KeyError, TypeError, ValueError):
            return False, None, None

        current_node_key = normalize_id(assignment.get("nodeID"))
        frequency_step = abs(float(work_unit["frequencyStep"]))
        device_frequency_tolerance = max(
            frequency_step * CROSS_NODE_FREQUENCY_TOLERANCE_STEPS,
            1.0e-7,
        )

        for prior in self.scientific_rejections.get(work_id, []):
            prior_node_key = normalize_id(prior.get("nodeID"))

            # Consensus must come from a genuinely different registered node.
            if prior_node_key is None or prior_node_key == current_node_key:
                continue

            prior_verification = prior.get("verification", {})
            prior_frequency_error = prior_verification.get("frequencyError")
            prior_ambiguity_tolerance = prior_verification.get(
                "ambiguityFrequencyTolerance"
            )

            if (
                prior_frequency_error is None
                or prior_ambiguity_tolerance is None
                or float(prior_frequency_error)
                > float(prior_ambiguity_tolerance)
            ):
                continue

            try:
                prior_frequency = float(prior["bestFrequency"])
                prior_power = float(prior["bestPower"])
            except (KeyError, TypeError, ValueError):
                continue

            device_frequency_error = abs(
                current_frequency - prior_frequency
            )
            device_power_error = abs(current_power - prior_power)
            device_power_tolerance = max(
                POWER_ABSOLUTE_TOLERANCE,
                max(abs(current_power), abs(prior_power))
                * CROSS_NODE_POWER_RELATIVE_TOLERANCE,
            )

            if (
                device_frequency_error <= device_frequency_tolerance
                and device_power_error <= device_power_tolerance
            ):
                consensus = dict(verification)
                consensus.update(
                    {
                        "method": "cross-node-consensus",
                        "consensusNodeID": prior.get("nodeID"),
                        "consensusFrequency": prior_frequency,
                        "consensusPower": prior_power,
                        "crossNodeFrequencyError": device_frequency_error,
                        "crossNodeFrequencyTolerance": (
                            device_frequency_tolerance
                        ),
                        "crossNodePowerError": device_power_error,
                        "crossNodePowerTolerance": device_power_tolerance,
                    }
                )

                return (
                    True,
                    "Result accepted by cross-node consensus: a different "
                    "node independently reproduced the same ambiguous "
                    "frequency/power result inside the existing Astropy "
                    "ambiguity-frequency band.",
                    consensus,
                )

        return False, None, None

    def _record_scientific_rejection_locked(
        self,
        work_id,
        assignment,
        result,
        verification,
    ):
        rejections = self.scientific_rejections.setdefault(work_id, [])
        node_key = normalize_id(assignment.get("nodeID"))

        # Keep only the latest rejected scientific result per node.
        rejections[:] = [
            item
            for item in rejections
            if normalize_id(item.get("nodeID")) != node_key
        ]
        rejections.append(
            {
                "nodeID": assignment.get("nodeID"),
                "bestFrequency": result.get("bestFrequency"),
                "bestPower": result.get("bestPower"),
                "verification": dict(verification),
                "recordedAt": time.time(),
            }
        )

    @staticmethod
    def _print_optional_float(label, value, digits=8):
        if value is None:
            return
        print(f"   {label}: {float(value):.{digits}f}")

    def _print_rejection(
        self,
        *,
        work_unit,
        assignment,
        message,
        details,
        failure_count,
        hard_failed,
    ):
        print()
        print("❌ Science result rejected")
        print(f"   work: {work_unit['id']}")
        print(f"   node: {assignment.get('nodeID', 'unknown')}")
        print(f"   dataset: {work_unit['datasetID']}")
        print(
            "   frequency start index: "
            f"{work_unit['frequencyStartIndex']}"
        )
        self._print_optional_float(
            "device frequency",
            details.get("deviceFrequency"),
        )
        self._print_optional_float(
            "Astropy frequency",
            details.get("astropyFrequency"),
        )
        self._print_optional_float(
            "frequency error",
            details.get("frequencyError"),
        )
        self._print_optional_float(
            "frequency tolerance",
            details.get("frequencyTolerance"),
        )
        self._print_optional_float(
            "ambiguity frequency tolerance",
            details.get("ambiguityFrequencyTolerance"),
        )
        self._print_optional_float(
            "device power",
            details.get("devicePower"),
        )
        self._print_optional_float(
            "Astropy power",
            details.get("astropyPower"),
        )
        self._print_optional_float(
            "power error",
            details.get("powerError"),
        )
        self._print_optional_float(
            "power tolerance",
            details.get("powerTolerance"),
        )
        print(f"   reason: {message}")
        print(
            "   verification failures: "
            f"{failure_count}/"
            f"{MAX_VERIFICATION_FAILURES_PER_WORK_UNIT}"
        )

        if hard_failed:
            print("   action: HARD FAILED - not requeued")
        else:
            print(
                "   action: requeued; failing node temporarily avoided for "
                "this work unit"
            )

    def _print_environment_unavailable(
        self,
        *,
        work_unit,
        assignment,
        result,
        interruption_count,
    ):
        client_reason = first_value(
            result,
            "errorMessage",
            "error",
            "message",
            "failureReason",
            "reason",
        )

        print()
        print("⏸️ Work returned - node environment unavailable")
        print(f"   work: {work_unit['id']}")
        print(f"   node: {assignment.get('nodeID', 'unknown')}")
        print(f"   workload: {work_unit.get('workloadID', 'unknown')}")
        print(f"   dataset: {work_unit.get('datasetID', 'none')}")
        print("   failure kind: environment-unavailable")
        if client_reason is not None:
            print(f"   client reason: {client_reason}")
        print(
            "   environment interruptions for work unit: "
            f"{interruption_count}"
        )
        print(
            "   action: requeued with no retry penalty, no execution-failure "
            "count, and no node cooldown penalty"
        )

    def _print_transport_unavailable(
        self,
        *,
        work_unit,
        assignment,
        result,
        interruption_count,
    ):
        client_reason = first_value(
            result,
            "errorMessage",
            "error",
            "message",
            "failureReason",
            "reason",
        )

        print()
        print("📡 Work returned - coordinator transport unavailable")
        print(f"   work: {work_unit['id']}")
        print(f"   node: {assignment.get('nodeID', 'unknown')}")
        print(f"   workload: {work_unit.get('workloadID', 'unknown')}")
        print(f"   dataset: {work_unit.get('datasetID', 'none')}")
        print("   failure kind: transport-unavailable")
        if client_reason is not None:
            print(f"   client reason: {client_reason}")
        print(
            "   transport interruptions for work unit: "
            f"{interruption_count}"
        )
        print(
            "   action: requeued with no retry penalty, no execution-failure "
            "count, and no node cooldown penalty"
        )

    def _print_execution_failure(
        self,
        *,
        work_unit,
        assignment,
        result,
        execution_failure_count,
        node_execution_failure_streak,
        node_cooldown_applied,
    ):
        result_status = str(result.get("status", "unknown"))
        failure_kind = str(
            first_value(
                result,
                "failureKind",
                default="unknown",
            )
        )
        client_reason = first_value(
            result,
            "errorMessage",
            "error",
            "message",
            "failureReason",
            "reason",
        )

        print()
        print("⚠️ Work execution failed")
        print(f"   work: {work_unit['id']}")
        print(f"   node: {assignment.get('nodeID', 'unknown')}")
        print(f"   workload: {work_unit.get('workloadID', 'unknown')}")
        print(f"   dataset: {work_unit.get('datasetID', 'none')}")
        print(f"   failure kind: {failure_kind}")
        print(f"   status: {result_status}")
        if client_reason is not None:
            print(f"   client reason: {client_reason}")
        print(
            "   execution failures for work unit: "
            f"{execution_failure_count}"
        )
        print(
            "   consecutive execution failures for node: "
            f"{node_execution_failure_streak}"
        )

        if node_cooldown_applied:
            print(
                "   node action: all work paused for "
                f"{NODE_EXECUTION_COOLDOWN_SECONDS:.0f}s; "
                "one probe claim allowed after cooldown"
            )
        else:
            print(
                "   action: requeued; failing node temporarily avoided for "
                "this work unit"
            )

    def submit_result(self, route_work_id, result):
        work_id = normalize_id(route_work_id)
        result = dict(result)

        # Generic result envelope. Current Lomb-Scargle reduction still reads
        # the historical flattened fields, so mirror them from payload during
        # the transition. OpenStar Core does not require other workloads to
        # use these keys.
        result_payload = result.get("payload")
        if isinstance(result_payload, dict):
            for key in (
                "bestFrequency",
                "bestPeriodDays",
                "bestPower",
            ):
                if result.get(key) is None and result_payload.get(key) is not None:
                    result[key] = result_payload[key]

        with self.lock:
            work_unit = self.work_units.get(work_id)

            if work_unit is None:
                return False, "Unknown work unit.", 404

            if work_id in self.completed:
                if self.accepted_result_submissions.get(work_id) == result:
                    return True, "Identical result already accepted.", 200
                return (
                    False,
                    "A different result was already accepted for this work unit.",
                    409,
                )

            if work_id in self.failed:
                return (
                    False,
                    "Work unit is hard-failed and terminal.",
                    200,
                )

            assignment = self.assigned.get(work_id)

            if assignment is None:
                return False, "Work unit is not currently assigned.", 409

            # A client that explicitly reports status != completed did not
            # produce a scientific result. Failure provenance determines
            # whether this is a real execution failure or only a temporary
            # environment availability interruption.
            if result.get("status") != "completed":
                self.assigned.pop(work_id, None)

                now = time.time()
                failure_kind = str(
                    first_value(
                        result,
                        "failureKind",
                        default="unknown",
                    )
                )
                client_reason = first_value(
                    result,
                    "errorMessage",
                    "error",
                    "message",
                    "failureReason",
                    "reason",
                )

                if failure_kind == "environment-unavailable":
                    interruption_count = (
                        self.environment_unavailable_counts.get(work_id, 0) + 1
                    )
                    self.environment_unavailable_counts[work_id] = (
                        interruption_count
                    )
                    self.environment_unavailable_history.setdefault(
                        work_id,
                        [],
                    ).append({
                        "workID": work_unit["id"],
                        "projectID": work_unit.get("projectID"),
                        "workloadID": work_unit.get("workloadID"),
                        "datasetID": work_unit.get("datasetID"),
                        "nodeID": assignment["nodeID"],
                        "failureKind": failure_kind,
                        "errorMessage": client_reason,
                        "interruptionCount": interruption_count,
                        "returnedAt": now,
                    })

                    # Immediate lease return. This deliberately does NOT touch
                    # retry_counts, execution_failure_counts, node failure
                    # streaks, execution avoidance, or cooldown state.
                    self.retry_after.pop(work_id, None)
                    self.pending.appendleft(work_id)
                    self._mark_node_seen_locked(assignment["nodeID"])

                    environment_unavailable_args = {
                        "work_unit": dict(work_unit),
                        "assignment": dict(assignment),
                        "result": dict(result),
                        "interruption_count": interruption_count,
                    }
                    transport_unavailable_args = None
                    execution_failure_args = None
                    execution_failure_message = (
                        "Work requeued because node environment is temporarily "
                        "unavailable."
                    )
                    execution_failed = True
                elif failure_kind == "transport-unavailable":
                    interruption_count = (
                        self.transport_unavailable_counts.get(work_id, 0) + 1
                    )
                    self.transport_unavailable_counts[work_id] = (
                        interruption_count
                    )
                    self.transport_unavailable_history.setdefault(
                        work_id,
                        [],
                    ).append({
                        "workID": work_unit["id"],
                        "projectID": work_unit.get("projectID"),
                        "workloadID": work_unit.get("workloadID"),
                        "datasetID": work_unit.get("datasetID"),
                        "nodeID": assignment["nodeID"],
                        "failureKind": failure_kind,
                        "errorMessage": client_reason,
                        "interruptionCount": interruption_count,
                        "returnedAt": now,
                    })

                    # Transport loss is not evidence of bad compute. Return the
                    # lease immediately without touching retry/failure/cooldown
                    # accounting.
                    self.retry_after.pop(work_id, None)
                    self.pending.appendleft(work_id)
                    self._mark_node_seen_locked(assignment["nodeID"])

                    transport_unavailable_args = {
                        "work_unit": dict(work_unit),
                        "assignment": dict(assignment),
                        "result": dict(result),
                        "interruption_count": interruption_count,
                    }
                    environment_unavailable_args = None
                    execution_failure_args = None
                    execution_failure_message = (
                        "Work requeued because coordinator transport is "
                        "temporarily unavailable."
                    )
                    execution_failed = True
                else:
                    self.retry_counts[work_id] = (
                        self.retry_counts.get(work_id, 0) + 1
                    )
                    execution_failure_count = (
                        self.execution_failure_counts.get(work_id, 0) + 1
                    )
                    self.execution_failure_counts[work_id] = (
                        execution_failure_count
                    )

                    failure_record = {
                        "workID": work_unit["id"],
                        "projectID": work_unit.get("projectID"),
                        "workloadID": work_unit.get("workloadID"),
                        "datasetID": work_unit.get("datasetID"),
                        "nodeID": assignment["nodeID"],
                        "failureKind": failure_kind,
                        "errorMessage": client_reason,
                        "workUnitFailureCount": execution_failure_count,
                        "failedAt": now,
                    }
                    self.execution_failure_history.setdefault(
                        work_id,
                        [],
                    ).append(failure_record)

                    failing_node_key = normalize_id(assignment["nodeID"])
                    avoid_by_node = self.execution_avoid_until.setdefault(
                        work_id,
                        {},
                    )
                    avoid_by_node[failing_node_key] = (
                        now + EXECUTION_FAILURE_NODE_AVOID_SECONDS
                    )

                    (
                        node_execution_failure_streak,
                        node_cooldown_applied,
                    ) = self._record_node_execution_failure_locked(
                        assignment["nodeID"],
                        now,
                    )

                    self.retry_after[work_id] = now + RETRY_COOLDOWN_SECONDS
                    self.pending.append(work_id)
                    self._mark_node_seen_locked(assignment["nodeID"])

                    execution_failure_args = {
                        "work_unit": dict(work_unit),
                        "assignment": dict(assignment),
                        "result": dict(result),
                        "execution_failure_count": execution_failure_count,
                        "node_execution_failure_streak": (
                            node_execution_failure_streak
                        ),
                        "node_cooldown_applied": node_cooldown_applied,
                    }
                    environment_unavailable_args = None
                    transport_unavailable_args = None
                    execution_failure_message = "Work unit did not complete."
                    execution_failed = True
            else:
                execution_failed = False
                execution_failure_args = None
                environment_unavailable_args = None
                transport_unavailable_args = None
                self._record_node_execution_success_locked(
                    assignment["nodeID"]
                )

            if execution_failed:
                accepted = False
                message = execution_failure_message
                verification = None
                reference_comparison = None
            else:
                accepted, message, validation_details = (
                    self._validate_metal_result(
                        work_unit,
                        result,
                    )
                )

                reference_comparison = None

                if accepted:
                    reference_comparison = self._reference_comparison(
                        work_unit,
                        result,
                    )

                    mismatch = (
                        reference_comparison.get("status")
                        == "mismatch"
                    )
                    self.reference_mismatch_counts[work_id] = (
                        1 if mismatch else 0
                    )

                    comparison_details = reference_comparison.get(
                        "details",
                        {},
                    )

                    verification = {
                        **comparison_details,
                        "method": "metal-result",
                        "referenceComparisonStatus": (
                            reference_comparison.get("status")
                        ),
                        "referenceComparisonMethod": (
                            reference_comparison.get("method")
                        ),
                        "referenceComparisonMessage": (
                            reference_comparison.get("message")
                        ),
                    }

                    if mismatch:
                        message = (
                            "Metal result accepted. Astropy comparison "
                            "disagreed, recorded diagnostically only."
                        )
                    elif (
                        reference_comparison.get("status")
                        == "match"
                    ):
                        message = (
                            "Metal result accepted. Astropy comparison "
                            "matched."
                        )
                    else:
                        message = (
                            "Metal result accepted. No Astropy reference "
                            "was supplied."
                        )
                else:
                    verification = {
                        "method": "metal-result-invalid",
                        **validation_details,
                    }

            if execution_failed:
                rejection_args = None
            elif not accepted:
                self._record_scientific_rejection_locked(
                    work_id,
                    assignment,
                    result,
                    verification,
                )
                self.assigned.pop(work_id, None)

                self.retry_counts[work_id] = (
                    self.retry_counts.get(work_id, 0) + 1
                )
                failure_count = (
                    self.verification_failure_counts.get(work_id, 0) + 1
                )
                self.verification_failure_counts[work_id] = failure_count

                hard_failed = (
                    failure_count
                    >= MAX_VERIFICATION_FAILURES_PER_WORK_UNIT
                )

                if hard_failed:
                    self.retry_after.pop(work_id, None)
                    self.execution_avoid_until.pop(work_id, None)
                    self.scientific_rejections.pop(work_id, None)
                    self.failed[work_id] = {
                        "datasetID": work_unit["datasetID"],
                        "nodeID": assignment["nodeID"],
                        "reason": message,
                        "verificationFailures": failure_count,
                        "verification": dict(verification),
                        "failedAt": time.time(),
                    }
                    self.dataset_diagnostic_cache.pop(
                        work_unit["datasetID"],
                        None,
                    )
                    self._capture_terminal_edge_locked()
                else:
                    now = time.time()

                    # The Metal result itself was malformed/out-of-range.
                    # Requeue it and temporarily avoid the same node for this
                    # exact work unit so another execution can try it.
                    failing_node_key = normalize_id(assignment["nodeID"])
                    avoid_by_node = self.execution_avoid_until.setdefault(
                        work_id,
                        {},
                    )
                    avoid_by_node[failing_node_key] = (
                        now + SCIENTIFIC_FAILURE_NODE_AVOID_SECONDS
                    )

                    self.retry_after[work_id] = (
                        now + RETRY_COOLDOWN_SECONDS
                    )
                    self.pending.append(work_id)

                self._mark_node_seen_locked(assignment["nodeID"])

                rejection_args = {
                    "work_unit": dict(work_unit),
                    "assignment": dict(assignment),
                    "message": message,
                    "details": dict(verification),
                    "failure_count": failure_count,
                    "hard_failed": hard_failed,
                }

            else:
                stored_result = dict(result)
                stored_result["datasetID"] = work_unit["datasetID"]
                stored_result["nodeID"] = assignment["nodeID"]
                stored_result["verification"] = dict(verification)

                if stored_result.get("bestPeriodDays") is None:
                    best_frequency = float(stored_result["bestFrequency"])
                    stored_result["bestPeriodDays"] = (
                        1.0 / best_frequency
                        if best_frequency != 0
                        else None
                    )

                self.completed[work_id] = stored_result
                self.accepted_result_submissions[work_id] = dict(result)
                self.dataset_diagnostic_cache.pop(
                    work_unit["datasetID"],
                    None,
                )
                self.assigned.pop(work_id, None)
                self.retry_after.pop(work_id, None)
                self.execution_avoid_until.pop(work_id, None)
                self.scientific_rejections.pop(work_id, None)
                self._mark_node_seen_locked(assignment["nodeID"])

                completed_count = len(self.completed)
                total_count = len(self.work_units)
                self._capture_terminal_edge_locked()

        if execution_failed:
            if environment_unavailable_args is not None:
                self._print_environment_unavailable(
                    **environment_unavailable_args
                )
            elif transport_unavailable_args is not None:
                self._print_transport_unavailable(
                    **transport_unavailable_args
                )
            elif execution_failure_args is not None:
                self._print_execution_failure(**execution_failure_args)

            # The client reported that execution itself did not complete.
            # Returning 200 acknowledges the report so the networking layer
            # does not immediately repost the same failed payload. The work
            # unit has already been requeued for another claim.
            return False, message, 200

        self._notify_terminal_observer()

        if not accepted:
            self._print_rejection(**rejection_args)

            # Terminal scientific failures must advance scheduling while
            # remaining explicit failures.
            if rejection_args.get("hard_failed"):
                self._report_completions()

            # This is an application-level scientific rejection, not a broken
            # HTTP request. Returning 200 prevents networking layers from
            # immediately POSTing the identical deterministic result again.
            return False, message, 200

        self._report_completions()

        return True, message, 200

    def _capture_terminal_edge_locked(self):
        """Capture the true terminal edge before synchronous reporting starts."""
        if self.terminal_monotonic is not None:
            return
        if len(self.completed) + len(self.failed) != len(self.work_units):
            return
        terminal_at = time.monotonic()
        self.terminal_monotonic = terminal_at

    def _notify_terminal_observer(self):
        """Notify outside the state lock to preserve scheduler lock ordering."""
        with self.lock:
            if self._terminal_observer_notified or self.terminal_monotonic is None:
                return
            self._terminal_observer_notified = True
            observer = self.terminal_observer
            terminal_at = self.terminal_monotonic
        if observer is not None:
            try:
                observer(str(self.project_id), terminal_at)
            except Exception:
                # Operational diagnostics must never affect accepted results.
                pass

    def _dataset_best_locked(self, dataset_id):
        best = None

        for work_id in self.work_ids_by_dataset.get(dataset_id, []):
            result = self.completed.get(work_id)

            if result is None or result.get("bestPower") is None:
                continue

            if best is None or float(result["bestPower"]) > float(best["bestPower"]):
                best = result

        return best

    @staticmethod
    def _finite_float_list(values):
        result = []

        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(value):
                result.append(value)

        return result

    @staticmethod
    def _fold_metrics(times, flux, period_days):
        try:
            period_days = float(period_days)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(period_days) or period_days <= 0:
            return None

        sample_count = min(len(times), len(flux))

        if sample_count < 2:
            return None

        bin_sums = [0.0] * PERIOD_FOLD_BINS
        bin_counts = [0] * PERIOD_FOLD_BINS
        usable = []

        for index in range(sample_count):
            try:
                time_value = float(times[index])
                flux_value = float(flux[index])
            except (TypeError, ValueError):
                continue

            if not math.isfinite(time_value) or not math.isfinite(flux_value):
                continue

            phase = (time_value / period_days) % 1.0
            bin_index = int(phase * PERIOD_FOLD_BINS)

            if bin_index >= PERIOD_FOLD_BINS:
                bin_index = PERIOD_FOLD_BINS - 1

            bin_sums[bin_index] += flux_value
            bin_counts[bin_index] += 1
            usable.append((bin_index, flux_value))

        if len(usable) < 2:
            return None

        overall_mean = sum(value for _, value in usable) / len(usable)
        total_ss = sum(
            (value - overall_mean) ** 2
            for _, value in usable
        )

        if total_ss <= 0:
            return None

        bin_means = [None] * PERIOD_FOLD_BINS

        for bin_index in range(PERIOD_FOLD_BINS):
            count = bin_counts[bin_index]

            if count >= PERIOD_MIN_POINTS_PER_BIN:
                bin_means[bin_index] = (
                    bin_sums[bin_index] / count
                )

        residual_ss = 0.0
        residual_count = 0

        for bin_index, value in usable:
            bin_mean = bin_means[bin_index]

            if bin_mean is None:
                continue

            residual_ss += (value - bin_mean) ** 2
            residual_count += 1

        if residual_count <= 0:
            return None

        within_bin_rms = math.sqrt(
            residual_ss / residual_count
        )

        total_rms = math.sqrt(
            total_ss / len(usable)
        )

        coherence = 1.0 - (
            residual_ss / total_ss
        )

        coherence = max(
            0.0,
            min(
                1.0,
                coherence,
            ),
        )

        return {
            "periodDays": period_days,
            "withinBinRMS": within_bin_rms,
            "totalRMS": total_rms,
            "coherence": coherence,
            "usedSamples": residual_count,
            "occupiedBins": sum(
                1
                for count in bin_counts
                if count >= PERIOD_MIN_POINTS_PER_BIN
            ),
        }

    def _independent_candidates_locked(self, dataset_id):
        dataset = self.datasets[dataset_id]
        times = self._finite_float_list(
            dataset.get("times", [])
        )

        baseline = None

        if len(times) >= 2:
            baseline = max(times) - min(times)

        rayleigh = (
            1.0 / baseline
            if baseline is not None and baseline > 0
            else None
        )

        candidates = []

        for work_id in self.work_ids_by_dataset.get(dataset_id, []):
            result = self.completed.get(work_id)

            if (
                result is None
                or result.get("bestPower") is None
                or result.get("bestFrequency") is None
            ):
                continue

            try:
                frequency = float(result["bestFrequency"])
                power = float(result["bestPower"])
            except (TypeError, ValueError):
                continue

            if (
                not math.isfinite(frequency)
                or not math.isfinite(power)
                or frequency <= 0
            ):
                continue

            candidates.append(
                {
                    "frequency": frequency,
                    "periodDays": 1.0 / frequency,
                    "power": power,
                    "workID": work_id,
                }
            )

        candidates.sort(
            key=lambda item: item["power"],
            reverse=True,
        )

        selected = []

        for candidate in candidates:
            if rayleigh is not None and any(
                abs(
                    candidate["frequency"]
                    - existing["frequency"]
                ) < rayleigh
                for existing in selected
            ):
                continue

            selected.append(candidate)

            if len(selected) >= PERIOD_INDEPENDENT_CANDIDATE_COUNT:
                break

        return selected

    def _distributed_chunk_mode_coverage_locked(self, dataset_id):
        """Return every raw chunk winner before display candidate reduction."""
        chunks = []
        work_ids = self.work_ids_by_dataset.get(dataset_id, [])
        for work_id in work_ids:
            work_unit = self.work_units[work_id]
            result = self.completed.get(work_id)
            if result is None or result.get("bestFrequency") is None:
                continue
            start = float(work_unit["startFrequency"])
            step = float(work_unit["frequencyStep"])
            count = int(work_unit["frequencyCount"])
            chunks.append({
                "frequency": float(result["bestFrequency"]),
                "power": float(result["bestPower"]),
                "startFrequency": start,
                "endFrequency": start + max(count - 1, 0) * step,
                "workID": work_id,
            })
        dataset = self.datasets[dataset_id]
        has_measurement_uncertainties = any(
            dataset.get(key) is not None
            for key in (
                "measurementUncertainties",
                "fluxUncertainties",
                "valueUncertainties",
            )
        )
        return {
            "chunks": chunks,
            "complete": len(chunks) == len(work_ids),
            # Existing Lomb-Scargle workers do not declare that their chunk
            # winner used absolute heteroscedastic uncertainties. Refuse the
            # global-mode guarantee in that case rather than assuming it.
            "objectiveMatches": not has_measurement_uncertainties,
            "selectedPower": max(
                (chunk["power"] for chunk in chunks),
                default=None,
            ),
        }

    def _dataset_result_diagnostics_locked(self, dataset_id):
        pending, assigned, completed, total = self._dataset_counts_locked(
            dataset_id
        )
        failed = self._dataset_failed_count_locked(dataset_id)
        terminal = (
            total > 0
            and completed + failed == total
        )

        best = self._dataset_best_locked(dataset_id)

        if not terminal:
            return {
                "periodStatus": "SEARCHING",
                "periodConfidence": None,
                "coverageComplete": False,
                "candidate": (
                    {
                        "frequency": float(best["bestFrequency"]),
                        "periodDays": float(best["bestPeriodDays"]),
                        "power": float(best["bestPower"]),
                    }
                    if best is not None
                    else None
                ),
                "authoritative": None,
                "harmonicCandidates": [],
                "preferredPhysicalPeriodDays": None,
                "independentCandidates": [],
            }

        cached = self.dataset_diagnostic_cache.get(dataset_id)

        if cached is not None:
            return cached

        dataset = self.datasets[dataset_id]
        times = dataset.get("times", [])
        flux = dataset.get("flux", [])

        independent_candidates = self._independent_candidates_locked(
            dataset_id
        )

        if best is None:
            diagnostic = {
                "periodStatus": (
                    "INCOMPLETE_COVERAGE"
                    if failed
                    else "NO_RESULT"
                ),
                "periodConfidence": "none",
                "coverageComplete": failed == 0,
                "candidate": None,
                "authoritative": None,
                "harmonicCandidates": [],
                "preferredPhysicalPeriodDays": None,
                "independentCandidates": independent_candidates,
            }
            self.dataset_diagnostic_cache[dataset_id] = diagnostic
            return diagnostic

        best_frequency = float(best["bestFrequency"])
        best_period = float(best["bestPeriodDays"])
        best_power = float(best["bestPower"])

        primary_fold = self._fold_metrics(
            times,
            flux,
            best_period,
        )
        doubled_fold = self._fold_metrics(
            times,
            flux,
            best_period * 2.0,
        )
        half_fold = self._fold_metrics(
            times,
            flux,
            best_period * 0.5,
        )

        primary_coherence = (
            primary_fold["coherence"]
            if primary_fold is not None
            else 0.0
        )

        second_power = (
            independent_candidates[1]["power"]
            if len(independent_candidates) >= 2
            else None
        )

        prominence_ratio = (
            best_power / second_power
            if second_power is not None and second_power > 0
            else None
        )

        candidate = {
            "frequency": best_frequency,
            "periodDays": best_period,
            "power": best_power,
            "foldCoherence": primary_coherence,
            "foldWithinBinRMS": (
                primary_fold["withinBinRMS"]
                if primary_fold is not None
                else None
            ),
            "independentPeakProminenceRatio": prominence_ratio,
        }

        frequency_interval, frequency_interval_diagnostics = (
            estimate_frequency_interval(
                dataset,
                best_frequency,
                (),
                self._distributed_chunk_mode_coverage_locked(dataset_id),
            )
        )
        candidate["frequencyConfidenceInterval"] = frequency_interval
        candidate["frequencyUncertaintyDiagnostics"] = (
            frequency_interval_diagnostics
        )

        harmonic_candidates = []

        for relation, fold in (
            ("0.5x", half_fold),
            ("1x", primary_fold),
            ("2x", doubled_fold),
        ):
            if fold is None:
                continue

            harmonic_candidates.append(
                {
                    "relation": relation,
                    "periodDays": fold["periodDays"],
                    "foldCoherence": fold["coherence"],
                    "foldWithinBinRMS": fold["withinBinRMS"],
                }
            )

        preferred_physical_period = best_period
        preferred_relation = "1x"
        preferred_coherence = primary_coherence

        for relation, fold in (
            ("0.5x", half_fold),
            ("2x", doubled_fold),
        ):
            if fold is None:
                continue

            coherence_gain = (
                fold["coherence"]
                - primary_coherence
            )

            if (
                coherence_gain
                >= HARMONIC_PREFERENCE_MIN_COHERENCE_GAIN
                and fold["coherence"] > preferred_coherence
            ):
                preferred_physical_period = fold["periodDays"]
                preferred_relation = relation
                preferred_coherence = fold["coherence"]

        if failed:
            period_status = "INCOMPLETE_COVERAGE"
            period_confidence = "none"
            authoritative = None
        elif (
            best_power < PERIOD_RELIABLE_MIN_POWER
            and primary_coherence < PERIOD_RELIABLE_MIN_FOLD_COHERENCE
        ):
            period_status = "NO_RELIABLE_PERIOD"
            period_confidence = "none"
            authoritative = None
        else:
            if (
                best_power < PERIOD_LOW_CONFIDENCE_POWER
                or primary_coherence < PERIOD_LOW_CONFIDENCE_FOLD_COHERENCE
            ):
                period_status = "LOW_CONFIDENCE"
                period_confidence = "low"
            elif (
                best_power >= 0.10
                and primary_coherence >= 0.10
            ):
                period_status = "RELIABLE"
                period_confidence = "high"
            else:
                period_status = "RELIABLE"
                period_confidence = "medium"

            authoritative = {
                "frequency": best_frequency,
                "periodDays": best_period,
                "power": best_power,
            }

        diagnostic = {
            "periodStatus": period_status,
            "periodConfidence": period_confidence,
            "coverageComplete": failed == 0,
            "candidate": candidate,
            "authoritative": authoritative,
            "harmonicCandidates": harmonic_candidates,
            "preferredPhysicalPeriodDays": (
                preferred_physical_period
                if authoritative is not None
                else None
            ),
            "preferredPhysicalPeriodRelation": (
                preferred_relation
                if authoritative is not None
                else None
            ),
            "preferredPhysicalPeriodCoherence": (
                preferred_coherence
                if authoritative is not None
                else None
            ),
            "independentCandidates": independent_candidates,
        }

        self.dataset_diagnostic_cache[dataset_id] = diagnostic
        return diagnostic

    def _dataset_node_contributions_locked(self, dataset_id):
        contributions = {}

        for work_id in self.work_ids_by_dataset.get(dataset_id, []):
            result = self.completed.get(work_id)
            if result is None:
                continue

            node_id = result.get("nodeID")
            if node_id is None:
                continue

            node_id = str(node_id)
            contributions[node_id] = contributions.get(node_id, 0) + 1

        return dict(sorted(contributions.items()))

    def _dataset_contributions_locked(self, dataset_id):
        iphone = 0
        mac = 0
        other = 0

        for work_id in self.work_ids_by_dataset.get(dataset_id, []):
            result = self.completed.get(work_id)

            if result is None:
                continue

            node = self.nodes.get(normalize_id(result.get("nodeID")), {})
            capabilities = node.get("capabilities", {})

            platform = str(capabilities.get("platform", "")).strip().lower()
            hardware = str(
                capabilities.get("hardwareIdentifier", "")
            ).strip().lower()

            if platform in ("ios", "iphoneos") or hardware.startswith("iphone"):
                iphone += 1
            elif platform in ("macos", "mac", "maccatalyst") or hardware.startswith(
                "mac"
            ):
                mac += 1
            else:
                other += 1

        return {
            "iPhone": iphone,
            "Mac": mac,
            "other": other,
        }

    def dataset_status(self, dataset_id):
        with self.lock:
            pending, assigned, completed, total = self._dataset_counts_locked(
                dataset_id
            )
            best = self._dataset_best_locked(dataset_id)
            diagnostics = self._dataset_result_diagnostics_locked(
                dataset_id
            )

            dataset = self.datasets[dataset_id]
            manifest = self.dataset_manifest_entries.get(dataset_id, {})
            metadata = dataset.get("metadata", {})
            science = dataset.get("science", {})
            contributions = self._dataset_contributions_locked(dataset_id)
            node_contributions = (
                self._dataset_node_contributions_locked(dataset_id)
            )
            retry_count = self._dataset_retry_count_locked(dataset_id)
            verification_failure_count = (
                self._dataset_verification_failure_count_locked(dataset_id)
            )
            execution_failure_count = (
                self._dataset_execution_failure_count_locked(dataset_id)
            )
            execution_failure_kinds = (
                self._dataset_execution_failure_kinds_locked(dataset_id)
            )
            environment_unavailable_count = (
                self._dataset_environment_unavailable_count_locked(dataset_id)
            )
            transport_unavailable_count = (
                self._dataset_transport_unavailable_count_locked(dataset_id)
            )
            reference_mismatch_count = (
                self._dataset_reference_mismatch_count_locked(dataset_id)
            )
            failed_count = self._dataset_failed_count_locked(dataset_id)

            authoritative = diagnostics.get("authoritative")
            candidate = diagnostics.get("candidate")

            # During an active search, preserve the old best-so-far fields so
            # existing clients continue to show live progress. Once terminal,
            # best* means authoritative. Incomplete/weak datasets therefore
            # return null instead of publishing a false final period.
            if diagnostics["periodStatus"] == "SEARCHING":
                legacy_best_frequency = (
                    best.get("bestFrequency")
                    if best
                    else None
                )
                legacy_best_period = (
                    best.get("bestPeriodDays")
                    if best
                    else None
                )
                legacy_best_power = (
                    best.get("bestPower")
                    if best
                    else None
                )
            else:
                legacy_best_frequency = (
                    authoritative.get("frequency")
                    if authoritative
                    else None
                )
                legacy_best_period = (
                    authoritative.get("periodDays")
                    if authoritative
                    else None
                )
                legacy_best_power = (
                    authoritative.get("power")
                    if authoritative
                    else None
                )

            return {
                "id": dataset_id,
                "targetName": dataset.get("targetName", dataset_id),
                "mission": dataset.get("mission", "TESS"),
                "ticID": manifest.get("ticID", metadata.get("ticID")),
                "sector": manifest.get("sector", metadata.get("sector")),
                "role": science.get("role"),
                "classification": science.get("classification"),
                "publishedPeriodDays": science.get("publishedPeriodDays"),
                "pendingWorkUnits": pending,
                "assignedWorkUnits": assigned,
                "completedWorkUnits": completed,
                "totalWorkUnits": total,
                "progress": (
                    (completed + failed_count) / total
                    if total
                    else 1.0
                ),
                "retryCount": retry_count,
                "verificationFailureCount": verification_failure_count,
                "executionFailureCount": execution_failure_count,
                "executionFailureKinds": execution_failure_kinds,
                "environmentUnavailableCount": environment_unavailable_count,
                "transportUnavailableCount": transport_unavailable_count,
                "referenceMismatchCount": reference_mismatch_count,
                "failedWorkUnits": failed_count,
                "periodStatus": diagnostics["periodStatus"],
                "periodConfidence": diagnostics["periodConfidence"],
                "coverageComplete": diagnostics["coverageComplete"],
                "bestFrequency": legacy_best_frequency,
                "bestPeriodDays": legacy_best_period,
                "bestPower": legacy_best_power,
                "candidateFrequency": (
                    candidate.get("frequency")
                    if candidate
                    else None
                ),
                "candidatePeriodDays": (
                    candidate.get("periodDays")
                    if candidate
                    else None
                ),
                "candidatePower": (
                    candidate.get("power")
                    if candidate
                    else None
                ),
                "candidateFoldCoherence": (
                    candidate.get("foldCoherence")
                    if candidate
                    else None
                ),
                "candidatePeakProminenceRatio": (
                    candidate.get(
                        "independentPeakProminenceRatio"
                    )
                    if candidate
                    else None
                ),
                "candidateFrequencyConfidenceInterval": (
                    candidate.get("frequencyConfidenceInterval")
                    if candidate
                    else None
                ),
                "candidateFrequencyUncertaintyDiagnostics": (
                    candidate.get("frequencyUncertaintyDiagnostics")
                    if candidate
                    else None
                ),
                "preferredPhysicalPeriodDays": diagnostics.get(
                    "preferredPhysicalPeriodDays"
                ),
                "preferredPhysicalPeriodRelation": diagnostics.get(
                    "preferredPhysicalPeriodRelation"
                ),
                "harmonicCandidates": diagnostics.get(
                    "harmonicCandidates",
                    [],
                ),
                "independentCandidates": diagnostics.get(
                    "independentCandidates",
                    [],
                ),
                "iPhoneContribution": contributions["iPhone"],
                "macContribution": contributions["Mac"],
                "otherContribution": contributions["other"],
                "nodeContributions": node_contributions,
            }

    def project_status(self):
        with self.lock:
            self._requeue_expired_locked()

            project_pending = len(self.pending)
            project_assigned = len(self.assigned)
            project_completed = len(self.completed)
            project_failed = len(self.failed)
            project_execution_failures = sum(
                self.execution_failure_counts.values()
            )
            project_environment_unavailable = sum(
                self.environment_unavailable_counts.values()
            )
            project_transport_unavailable = sum(
                self.transport_unavailable_counts.values()
            )
            project_total = len(self.work_units)
            project_node_contributions = {}
            for result in self.completed.values():
                node_id = result.get("nodeID")
                if node_id is None:
                    continue
                node_id = str(node_id)
                project_node_contributions[node_id] = (
                    project_node_contributions.get(node_id, 0) + 1
                )
            project_node_contributions = dict(
                sorted(project_node_contributions.items())
            )

            dataset_statuses = [
                self.dataset_status(dataset_id)
                for dataset_id in self.datasets
            ]

            current_dataset_id = self._current_dataset_id_locked()

            if current_dataset_id is None:
                current = {
                    "id": "",
                    "targetName": "",
                    "mission": "",
                    "pendingWorkUnits": 0,
                    "assignedWorkUnits": 0,
                    "completedWorkUnits": 0,
                    "totalWorkUnits": 0,
                    "progress": 1.0,
                    "retryCount": 0,
                    "verificationFailureCount": 0,
                    "executionFailureCount": 0,
                    "executionFailureKinds": {},
                    "environmentUnavailableCount": 0,
                    "transportUnavailableCount": 0,
                    "failedWorkUnits": 0,
                    "bestFrequency": None,
                    "bestPeriodDays": None,
                    "bestPower": None,
                    "periodStatus": "NO_DATASET",
                    "periodConfidence": None,
                    "coverageComplete": False,
                    "candidateFrequency": None,
                    "candidatePeriodDays": None,
                    "candidatePower": None,
                    "preferredPhysicalPeriodDays": None,
                    "preferredPhysicalPeriodRelation": None,
                    "harmonicCandidates": [],
                }
                sample_count = 0
            else:
                current = next(
                    status
                    for status in dataset_statuses
                    if status["id"] == current_dataset_id
                )
                sample_count = len(
                    self.datasets[current_dataset_id].get("times", [])
                )

            return {
                "projectID": self.project_id,
                "workloadID": self.workload_id,

                # Legacy/current-dataset fields used by the existing clients.
                "datasetID": current["id"],
                "targetName": current["targetName"],
                "mission": current["mission"],
                "sampleCount": sample_count,
                "samples": sample_count,
                "pending": current["pendingWorkUnits"],
                "assigned": current["assignedWorkUnits"],
                "completed": current["completedWorkUnits"],
                "total": current["totalWorkUnits"],
                "pendingWorkUnits": current["pendingWorkUnits"],
                "assignedWorkUnits": current["assignedWorkUnits"],
                "completedWorkUnits": current["completedWorkUnits"],
                "totalWorkUnits": current["totalWorkUnits"],
                "progress": current["progress"],
                "retryCount": current["retryCount"],
                "verificationFailureCount": current[
                    "verificationFailureCount"
                ],
                "executionFailureCount": current["executionFailureCount"],
                "executionFailureKinds": current.get(
                    "executionFailureKinds",
                    {},
                ),
                "environmentUnavailableCount": current.get(
                    "environmentUnavailableCount",
                    0,
                ),
                "transportUnavailableCount": current.get(
                    "transportUnavailableCount",
                    0,
                ),
                "failedWorkUnits": current["failedWorkUnits"],
                "activeNodes": len(self.nodes),
                "bestFrequency": current["bestFrequency"],
                "bestPeriodDays": current["bestPeriodDays"],
                "bestPower": current["bestPower"],
                "periodStatus": current.get("periodStatus"),
                "periodConfidence": current.get("periodConfidence"),
                "coverageComplete": current.get("coverageComplete"),
                "candidateFrequency": current.get("candidateFrequency"),
                "candidatePeriodDays": current.get("candidatePeriodDays"),
                "candidatePower": current.get("candidatePower"),
                "preferredPhysicalPeriodDays": current.get(
                    "preferredPhysicalPeriodDays"
                ),
                "preferredPhysicalPeriodRelation": current.get(
                    "preferredPhysicalPeriodRelation"
                ),
                "harmonicCandidates": current.get("harmonicCandidates", []),

                # Project-wide fields for newer clients.
                "projectPendingWorkUnits": project_pending,
                "projectAssignedWorkUnits": project_assigned,
                "projectCompletedWorkUnits": project_completed,
                "projectFailedWorkUnits": project_failed,
                "projectExecutionFailureCount": project_execution_failures,
                "projectEnvironmentUnavailableCount": (
                    project_environment_unavailable
                ),
                "projectTransportUnavailableCount": (
                    project_transport_unavailable
                ),
                "projectTotalWorkUnits": project_total,
                "projectProgress": (
                    (project_completed + project_failed) / project_total
                    if project_total
                    else 1.0
                ),
                "nodeContributions": project_node_contributions,
                "datasets": dataset_statuses,
            }

    def _print_dataset_result_summary(self, dataset_id, indent="   "):
        with self.lock:
            dataset = self.datasets[dataset_id]
            science = dataset.get("science", {})
            diagnostics = self._dataset_result_diagnostics_locked(
                dataset_id
            )
            contributions = self._dataset_contributions_locked(dataset_id)
            reference_mismatch_count = (
                self._dataset_reference_mismatch_count_locked(dataset_id)
            )
            execution_failure_count = (
                self._dataset_execution_failure_count_locked(dataset_id)
            )
            execution_failure_kinds = (
                self._dataset_execution_failure_kinds_locked(dataset_id)
            )
            environment_unavailable_count = (
                self._dataset_environment_unavailable_count_locked(dataset_id)
            )
            transport_unavailable_count = (
                self._dataset_transport_unavailable_count_locked(dataset_id)
            )
            reference_frequency, reference_period, reference_power = (
                self._global_reference(dataset)
            )

        print(f"{indent}target: {dataset.get('targetName', dataset_id)}")
        print(f"{indent}dataset: {dataset_id}")
        print(f"{indent}role: {science.get('role', 'unspecified')}")

        role = science.get("role")

        if role == "known":
            print(f"{indent}classification: {science.get('classification')}")
            print(
                f"{indent}published period: "
                f"{float(science['publishedPeriodDays']):.8f} days"
            )
        elif role == "blind":
            print(f"{indent}classification: [BLINDED]")
            print(f"{indent}published period: [BLINDED]")

        period_status = diagnostics["periodStatus"]
        period_confidence = diagnostics.get("periodConfidence")
        candidate = diagnostics.get("candidate")
        authoritative = diagnostics.get("authoritative")

        print(f"{indent}OpenStar period status: {period_status}")

        if execution_failure_count:
            print(
                f"{indent}execution failures: "
                f"{execution_failure_count}"
            )
            for failure_kind, count in sorted(
                execution_failure_kinds.items()
            ):
                print(
                    f"{indent}   {failure_kind}: {count}"
                )

        if environment_unavailable_count:
            print(
                f"{indent}environment-unavailable interruptions: "
                f"{environment_unavailable_count}"
            )

        if transport_unavailable_count:
            print(
                f"{indent}transport-unavailable interruptions: "
                f"{transport_unavailable_count}"
            )

        reference_chunk_count = len(
            self.chunk_references_by_dataset.get(dataset_id, {})
        )
        if reference_chunk_count:
            print(
                f"{indent}Astropy chunk comparison mismatches: "
                f"{reference_mismatch_count}/{reference_chunk_count}"
            )

        if period_confidence is not None:
            print(
                f"{indent}OpenStar period confidence: "
                f"{period_confidence}"
            )

        if not diagnostics.get("coverageComplete", False):
            print(
                f"{indent}science coverage: INCOMPLETE "
                "(one or more frequency chunks hard-failed)"
            )

        if candidate is None:
            print(f"{indent}OpenStar candidate: none")
            return

        if authoritative is None:
            print(
                f"{indent}candidate frequency: "
                f"{float(candidate['frequency']):.8f} cycles/day"
            )
            print(
                f"{indent}candidate period: "
                f"{float(candidate['periodDays']):.8f} days"
            )
            print(
                f"{indent}candidate power: "
                f"{float(candidate['power']):.8f}"
            )
        else:
            print(
                f"{indent}OpenStar frequency: "
                f"{float(authoritative['frequency']):.8f} cycles/day"
            )
            print(
                f"{indent}OpenStar period: "
                f"{float(authoritative['periodDays']):.8f} days"
            )
            print(
                f"{indent}OpenStar power: "
                f"{float(authoritative['power']):.8f}"
            )

        if candidate.get("foldCoherence") is not None:
            print(
                f"{indent}fold coherence: "
                f"{float(candidate['foldCoherence']):.6f}"
            )

        if candidate.get("independentPeakProminenceRatio") is not None:
            print(
                f"{indent}independent-peak prominence: "
                f"{float(candidate['independentPeakProminenceRatio']):.3f}x"
            )

        harmonic_candidates = diagnostics.get(
            "harmonicCandidates",
            [],
        )

        if harmonic_candidates:
            print(f"{indent}fold candidates:")

            for harmonic in harmonic_candidates:
                print(
                    f"{indent}   {harmonic['relation']}: "
                    f"{float(harmonic['periodDays']):.8f} days | "
                    f"coherence "
                    f"{float(harmonic['foldCoherence']):.6f}"
                )

        preferred_period = diagnostics.get(
            "preferredPhysicalPeriodDays"
        )
        preferred_relation = diagnostics.get(
            "preferredPhysicalPeriodRelation"
        )

        if (
            authoritative is not None
            and preferred_period is not None
            and preferred_relation is not None
            and preferred_relation != "1x"
        ):
            print(
                f"{indent}preferred full-cycle candidate: "
                f"{float(preferred_period):.8f} days "
                f"({preferred_relation})"
            )

        if reference_frequency is not None:
            reference_frequency = float(reference_frequency)
            print(
                f"{indent}Astropy frequency: "
                f"{reference_frequency:.8f} cycles/day"
            )

            if authoritative is not None:
                print(
                    f"{indent}frequency error: "
                    f"{abs(float(authoritative['frequency']) - reference_frequency):.8f}"
                )

        if reference_period is not None:
            reference_period = float(reference_period)
            print(
                f"{indent}Astropy period: "
                f"{reference_period:.8f} days"
            )

            if authoritative is not None:
                print(
                    f"{indent}period error: "
                    f"{abs(float(authoritative['periodDays']) - reference_period):.8f} days"
                )

        if reference_power is not None:
            print(
                f"{indent}Astropy power: "
                f"{float(reference_power):.8f}"
            )

        if (
            role == "known"
            and authoritative is not None
            and science.get("publishedPeriodDays") is not None
        ):
            published_period = float(
                science["publishedPeriodDays"]
            )
            published_error = abs(
                float(authoritative["periodDays"])
                - published_period
            )
            print(
                f"{indent}OpenStar vs published error: "
                f"{published_error:.8f} days"
            )

        print(
            f"{indent}iPhone contribution: "
            f"{contributions['iPhone']} work units"
        )
        print(
            f"{indent}Mac contribution: "
            f"{contributions['Mac']} work units"
        )

        if contributions["other"]:
            print(
                f"{indent}other contribution: "
                f"{contributions['other']} work units"
            )

    def _report_dataset_complete(self, dataset_id):
        with self.lock:
            if dataset_id in self.reported_completed_datasets:
                return

            _, _, completed, total = self._dataset_counts_locked(dataset_id)
            failed = self._dataset_failed_count_locked(dataset_id)

            if total == 0 or completed + failed != total:
                return

            self.reported_completed_datasets.add(dataset_id)

        print()

        if failed:
            print("⚠️ Dataset finished with hard failures")
            print(f"   accepted work units: {completed}/{total}")
            print(f"   hard-failed work units: {failed}/{total}")
        else:
            print("🌟 Dataset complete")

        self._print_dataset_result_summary(dataset_id)

    def _report_project_complete(self):
        with self.lock:
            if self.reported_project_complete:
                return

            completed = len(self.completed)
            failed = len(self.failed)
            total = len(self.work_units)

            if completed + failed != total:
                return

            self.reported_project_complete = True
            dataset_ids = list(self.datasets.keys())

        print()
        print("🏁 Project complete")
        print(f"   project: {self.project_id}")

        if failed:
            print("   status: COMPLETE WITH HARD FAILURES")
            print(f"   accepted work units: {completed}/{total}")
            print(f"   hard-failed work units: {failed}/{total}")
        else:
            print("   status: COMPLETE")
            print(f"   accepted work units: {completed}/{total}")

        for dataset_id in dataset_ids:
            print()
            self._print_dataset_result_summary(dataset_id)

    def _report_completions(self):
        report_started = time.monotonic()
        was_reported = self.reported_project_complete
        diagnostics = 0.0
        for dataset_id in self.datasets:
            diagnostic_started = time.monotonic()
            self._report_dataset_complete(dataset_id)
            diagnostics += time.monotonic() - diagnostic_started

        self._report_project_complete()
        if not was_reported and self.reported_project_complete:
            total = time.monotonic() - report_started
            try:
                print(
                    "⏱️ Project terminal finalization: "
                    f"project={self.project_id} diagnostics={diagnostics:.3f}s "
                    f"total={total:.3f}s"
                )
            except Exception:
                pass

    def print_startup_summary(self, port, host="0.0.0.0"):
        self.validate_startup()

        print()
        print("⭐ OpenStar Coordinator")
        print("Build: multi-target-v20.0-workflow-control")
        print(f"File: {Path(__file__).resolve()}")
        print(f"Listening on {host}:{port}")
        print()
        print(f"Project: {self.project_id}")
        print(f"Workload: {self.workload_id}")
        print(f"Datasets: {len(self.datasets)}")
        print(f"Work units: {len(self.work_units)}")
        print(
            "Scheduling: work units use workloadID + payload; registered node "
            "workload capabilities are matched before assignment"
        )
        print(
            "Failure provenance: failed results record failureKind + "
            "errorMessage; dataset summaries report failures by kind"
        )
        print(
            "Environment availability: environment-unavailable work is "
            "requeued without retry, execution-failure, or cooldown penalty"
        )
        print(
            "Transport availability: transport-unavailable work is requeued "
            "without retry, execution-failure, or cooldown penalty"
        )
        print(
            "Execution authority: valid Metal chunk results are accepted "
            "without Astropy voting"
        )
        print(
            "Reference validation: Astropy chunk references are optional "
            "diagnostics only; mismatches do not retry, replace, or discard "
            "Metal results"
        )
        print(
            "Execution failure handling: requeue without scientific failure; "
            f"avoid same node for {EXECUTION_FAILURE_NODE_AVOID_SECONDS:.0f}s"
        )
        print(
            "Node execution cooldown: after "
            f"{NODE_EXECUTION_FAILURE_STREAK_LIMIT} consecutive execution "
            f"failures, pause all work to that node for "
            f"{NODE_EXECUTION_COOLDOWN_SECONDS:.0f}s; successful execution "
            "resets the streak"
        )
        print(
            "Invalid-result handling: malformed/out-of-range Metal results "
            f"retry up to {MAX_VERIFICATION_FAILURES_PER_WORK_UNIT} times"
        )
        print(
            "Final period handling: incomplete coverage and weak signals do "
            "not publish an authoritative period"
        )
        print(
            "Harmonic handling: 0.5x / 1x / 2x fold coherence is reported; "
            "a materially better fold may be preferred as the physical cycle"
        )

        for dataset_id, dataset in self.datasets.items():
            reference_frequency, reference_period, reference_power = (
                self._global_reference(dataset)
            )
            science = dataset.get("science", {})
            chunk_count = len(
                self.chunk_references_by_dataset.get(dataset_id, {})
            )
            expected_chunk_count = len(
                self.work_ids_by_dataset.get(dataset_id, [])
            )

            print()
            print(f"Target: {dataset.get('targetName', dataset_id)}")
            print(f"Dataset: {dataset_id}")
            print(f"Mission: {dataset.get('mission', 'TESS')}")
            print(f"Samples: {len(dataset.get('times', []))}")
            print(f"Role: {science.get('role', 'unspecified')}")
            print(
                "Astropy diagnostic chunk references: "
                f"{chunk_count}/{expected_chunk_count}"
            )

            if reference_frequency is not None:
                print(
                    "Astropy reference frequency: "
                    f"{float(reference_frequency):.8f} cycles/day"
                )

            if reference_period is not None:
                print(
                    "Astropy reference period: "
                    f"{float(reference_period):.8f} days"
                )

            if reference_power is not None:
                print(
                    "Astropy reference power: "
                    f"{float(reference_power):.8f}"
                )
