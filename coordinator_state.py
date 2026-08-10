import json
import math
import threading
import time
import uuid
from collections import deque
from pathlib import Path


LEASE_SECONDS = 120.0
RETRY_COOLDOWN_SECONDS = 1.0
EXECUTION_FAILURE_NODE_AVOID_SECONDS = 30.0

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
        self.failed = {}
        self.retry_after = {}
        self.retry_counts = {}
        self.verification_failure_counts = {}
        self.execution_failure_counts = {}
        self.execution_avoid_until = {}

        self.nodes = {}

        self.reported_completed_datasets = set()
        self.reported_project_complete = False

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

                work_unit = {
                    "id": work_id,
                    "projectID": self.project_id,
                    "workloadID": self.workload_id,
                    "datasetID": dataset_id,
                    "frequencyStartIndex": start_index,
                    "startFrequency": minimum_frequency + start_index * frequency_step,
                    "frequencyStep": frequency_step,
                    "frequencyCount": frequency_count,
                }

                self.work_units[normalized_work_id] = work_unit
                self.work_ids_by_dataset[dataset_id].append(normalized_work_id)
                self.pending.append(normalized_work_id)
                self.retry_counts[normalized_work_id] = 0
                self.verification_failure_counts[normalized_work_id] = 0

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
                    errors.append(f"{dataset_id}: {field_name} not found")
                    continue

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    errors.append(f"{dataset_id}: {field_name} is not numeric")
                    continue

                if not math.isfinite(numeric_value):
                    errors.append(f"{dataset_id}: {field_name} is not finite")

            expected_starts = list(
                range(
                    0,
                    search["totalFrequencies"],
                    search["frequenciesPerWorkUnit"],
                )
            )

            indexed = self.chunk_references_by_dataset.get(dataset_id, {})

            if len(indexed) != len(expected_starts):
                errors.append(
                    f"{dataset_id}: Astropy work-unit references incomplete "
                    f"({len(indexed)}/{len(expected_starts)})"
                )

            expected_start_set = set(expected_starts)
            extra_starts = sorted(set(indexed.keys()) - expected_start_set)

            if extra_starts:
                errors.append(
                    f"{dataset_id}: unexpected Astropy chunk reference start indexes: "
                    + ", ".join(str(value) for value in extra_starts[:10])
                )

            for start_index in expected_starts:
                chunk = indexed.get(start_index)

                if chunk is None:
                    errors.append(
                        f"{dataset_id}: missing Astropy chunk reference at "
                        f"frequencyStartIndex={start_index}"
                    )
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

                if int(chunk_count) != expected_count:
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
                            f"{dataset_id}: chunk {start_index} missing {field_name}"
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
                "Astropy reference validation failed. OpenStar will not run "
                "unverified science work:\n - " + "\n - ".join(errors)
            )

    def _validate_science_metadata(self):
        errors = []

        for dataset_id, dataset in self.datasets.items():
            science = dataset.get("science", {})
            role = science.get("role")

            if role not in ("known", "control", "blind"):
                errors.append(
                    f"{dataset_id}: science.role must be known, control, or blind"
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

    def _mark_node_seen_locked(self, node_id):
        node_key = normalize_id(node_id)
        if node_key in self.nodes:
            self.nodes[node_key]["lastSeenAt"] = time.time()

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

    def _dataset_execution_failure_count_locked(self, dataset_id):
        return sum(
            self.execution_failure_counts.get(work_id, 0)
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
            if completed < total:
                return dataset_id

        return dataset_ids[-1]

    def claim_work(self, node_id):
        node_key = normalize_id(node_id)

        with self.lock:
            self._requeue_expired_locked()
            self._mark_node_seen_locked(node_id)

            if node_key not in self.nodes:
                return None

            current_dataset_id = self._current_dataset_id_locked()
            if current_dataset_id is None:
                return None

            now = time.time()
            deferred = []
            work_unit = None

            pending_length = len(self.pending)

            for _ in range(pending_length):
                work_id = self.pending.popleft()

                if work_id in self.completed or work_id in self.failed:
                    continue

                candidate = self.work_units.get(work_id)
                if candidate is None:
                    continue

                if candidate["datasetID"] != current_dataset_id:
                    deferred.append(work_id)
                    continue

                if self.retry_after.get(work_id, 0.0) > now:
                    deferred.append(work_id)
                    continue

                # Execution failures are operational, not scientific. If this
                # node recently failed to execute this exact work unit, defer
                # it so another registered node gets a chance. The avoidance
                # expires automatically so a single-node project cannot
                # deadlock forever.
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

                self.assigned[work_id] = {
                    "nodeID": str(node_id),
                    "assignedAt": now,
                    "leaseExpiresAt": now + LEASE_SECONDS,
                }

                work_unit = dict(candidate)
                break

            for work_id in deferred:
                self.pending.append(work_id)

        if work_unit is not None:
            print()
            print("📦 Science work assigned")
            print(f"   work: {work_unit['id']}")
            print(f"   node: {node_id}")
            print(f"   dataset: {work_unit['datasetID']}")
            print(f"   frequency start: {work_unit['startFrequency']:.6f}")
            print(f"   frequencies: {work_unit['frequencyCount']}")

        return work_unit

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
            print("   action: requeued")

    def _print_execution_failure(
            self,
            *,
            work_unit,
            assignment,
            result,
            execution_failure_count,
    ):
        result_status = str(result.get("status", "unknown"))
        client_reason = first_value(
            result,
            "error",
            "message",
            "failureReason",
            "reason",
        )

        print()
        print("⚠️ Work execution failed")
        print(f"   work: {work_unit['id']}")
        print(f"   node: {assignment.get('nodeID', 'unknown')}")
        print(f"   dataset: {work_unit['datasetID']}")
        print(
            "   frequency start index: "
            f"{work_unit['frequencyStartIndex']}"
        )
        print(f"   status: {result_status}")
        if client_reason is not None:
            print(f"   client reason: {client_reason}")
        print(
            "   execution failures for work unit: "
            f"{execution_failure_count}"
        )
        print(
            "   action: requeued; failing node temporarily avoided for this "
            "work unit"
        )

    def submit_result(self, route_work_id, result):
        work_id = normalize_id(route_work_id)

        with self.lock:
            work_unit = self.work_units.get(work_id)

            if work_unit is None:
                return False, "Unknown work unit.", 404

            if work_id in self.completed:
                return True, "Result already accepted.", 200

            if work_id in self.failed:
                return (
                    False,
                    "Work unit is hard-failed and requires coordinator restart "
                    "after investigation.",
                    200,
                )

            assignment = self.assigned.get(work_id)

            if assignment is None:
                return False, "Work unit is not currently assigned.", 409

            # A client that explicitly reports status != completed did not
            # produce a scientific result. Treat this as an operational
            # execution failure, not an Astropy verification failure.
            #
            # Requeue it, do not increment verification_failure_counts, and
            # temporarily prevent the same node from reclaiming this exact
            # work unit so another device can try it.
            if result.get("status") != "completed":
                self.assigned.pop(work_id, None)

                now = time.time()
                self.retry_counts[work_id] = (
                        self.retry_counts.get(work_id, 0) + 1
                )
                execution_failure_count = (
                        self.execution_failure_counts.get(work_id, 0) + 1
                )
                self.execution_failure_counts[work_id] = (
                    execution_failure_count
                )

                failing_node_key = normalize_id(assignment["nodeID"])
                avoid_by_node = self.execution_avoid_until.setdefault(
                    work_id,
                    {},
                )
                avoid_by_node[failing_node_key] = (
                        now + EXECUTION_FAILURE_NODE_AVOID_SECONDS
                )

                self.retry_after[work_id] = now + RETRY_COOLDOWN_SECONDS
                self.pending.append(work_id)
                self._mark_node_seen_locked(assignment["nodeID"])

                execution_failure_args = {
                    "work_unit": dict(work_unit),
                    "assignment": dict(assignment),
                    "result": dict(result),
                    "execution_failure_count": execution_failure_count,
                }
                execution_failure_message = "Work unit did not complete."
                execution_failed = True
            else:
                execution_failed = False

            if execution_failed:
                accepted = False
                message = execution_failure_message
                verification = None
            else:
                accepted, message, verification = self._verify_result(
                    work_unit,
                    result,
                )

            if execution_failed:
                rejection_args = None
            elif not accepted:
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
                    self.failed[work_id] = {
                        "datasetID": work_unit["datasetID"],
                        "nodeID": assignment["nodeID"],
                        "reason": message,
                        "verificationFailures": failure_count,
                        "verification": dict(verification),
                        "failedAt": time.time(),
                    }
                else:
                    self.retry_after[work_id] = (
                            time.time() + RETRY_COOLDOWN_SECONDS
                    )
                    self.pending.append(work_id)

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
                self.assigned.pop(work_id, None)
                self.retry_after.pop(work_id, None)
                self.execution_avoid_until.pop(work_id, None)
                self._mark_node_seen_locked(assignment["nodeID"])

                completed_count = len(self.completed)
                total_count = len(self.work_units)

        if execution_failed:
            self._print_execution_failure(**execution_failure_args)

            # The client reported that execution itself did not complete.
            # Returning 200 acknowledges the report so the networking layer
            # does not immediately repost the same failed payload. The work
            # unit has already been requeued for another claim.
            return False, message, 200

        if not accepted:
            self._print_rejection(**rejection_args)

            # This is an application-level scientific rejection, not a broken
            # HTTP request. Returning 200 prevents networking layers from
            # immediately POSTing the identical deterministic result again.
            return False, message, 200

        print()
        print("✅ Science result accepted")
        print(f"   work: {work_unit['id']}")
        print(f"   node: {stored_result.get('nodeID', 'unknown')}")
        print(f"   dataset: {work_unit['datasetID']}")
        print(f"   frequency: {float(stored_result['bestFrequency']):.8f}")

        if stored_result.get("bestPeriodDays") is not None:
            print(
                f"   period: {float(stored_result['bestPeriodDays']):.8f} days"
            )

        print(f"   power: {float(stored_result['bestPower']):.8f}")
        print(
            "   verification: "
            f"{stored_result['verification'].get('method')}"
        )

        if stored_result.get("duration") is not None:
            print(f"   duration: {float(stored_result['duration']):.4f}s")

        print(f"   project: {completed_count}/{total_count}")

        self._report_completions()

        return True, message, 200

    def _dataset_best_locked(self, dataset_id):
        best = None

        for work_id in self.work_ids_by_dataset.get(dataset_id, []):
            result = self.completed.get(work_id)

            if result is None or result.get("bestPower") is None:
                continue

            if best is None or float(result["bestPower"]) > float(best["bestPower"]):
                best = result

        return best

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

            dataset = self.datasets[dataset_id]
            manifest = self.dataset_manifest_entries.get(dataset_id, {})
            metadata = dataset.get("metadata", {})
            science = dataset.get("science", {})
            contributions = self._dataset_contributions_locked(dataset_id)
            retry_count = self._dataset_retry_count_locked(dataset_id)
            verification_failure_count = (
                self._dataset_verification_failure_count_locked(dataset_id)
            )
            execution_failure_count = (
                self._dataset_execution_failure_count_locked(dataset_id)
            )
            failed_count = self._dataset_failed_count_locked(dataset_id)

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
                "progress": completed / total if total else 1.0,
                "retryCount": retry_count,
                "verificationFailureCount": verification_failure_count,
                "executionFailureCount": execution_failure_count,
                "failedWorkUnits": failed_count,
                "bestFrequency": best.get("bestFrequency") if best else None,
                "bestPeriodDays": best.get("bestPeriodDays") if best else None,
                "bestPower": best.get("bestPower") if best else None,
                "iPhoneContribution": contributions["iPhone"],
                "macContribution": contributions["Mac"],
                "otherContribution": contributions["other"],
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
            project_total = len(self.work_units)

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
                    "failedWorkUnits": 0,
                    "bestFrequency": None,
                    "bestPeriodDays": None,
                    "bestPower": None,
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
                "failedWorkUnits": current["failedWorkUnits"],
                "activeNodes": len(self.nodes),
                "bestFrequency": current["bestFrequency"],
                "bestPeriodDays": current["bestPeriodDays"],
                "bestPower": current["bestPower"],

                # Project-wide fields for newer clients.
                "projectPendingWorkUnits": project_pending,
                "projectAssignedWorkUnits": project_assigned,
                "projectCompletedWorkUnits": project_completed,
                "projectFailedWorkUnits": project_failed,
                "projectExecutionFailureCount": project_execution_failures,
                "projectTotalWorkUnits": project_total,
                "projectProgress": (
                    project_completed / project_total
                    if project_total
                    else 1.0
                ),
                "datasets": dataset_statuses,
            }

    def _print_dataset_result_summary(self, dataset_id, indent="   "):
        with self.lock:
            dataset = self.datasets[dataset_id]
            science = dataset.get("science", {})
            best = self._dataset_best_locked(dataset_id)
            contributions = self._dataset_contributions_locked(dataset_id)
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

        if best is None:
            print(f"{indent}OpenStar result: none")
            return

        best_frequency = float(best["bestFrequency"])
        best_period = best.get("bestPeriodDays")
        best_power = float(best["bestPower"])

        print(
            f"{indent}OpenStar frequency: "
            f"{best_frequency:.8f} cycles/day"
        )

        if best_period is not None:
            print(
                f"{indent}OpenStar period: "
                f"{float(best_period):.8f} days"
            )

        print(f"{indent}OpenStar power: {best_power:.8f}")

        if reference_frequency is not None:
            reference_frequency = float(reference_frequency)
            print(
                f"{indent}Astropy frequency: "
                f"{reference_frequency:.8f} cycles/day"
            )
            print(
                f"{indent}frequency error: "
                f"{abs(best_frequency - reference_frequency):.8f}"
            )

        if reference_period is not None and best_period is not None:
            reference_period = float(reference_period)
            print(
                f"{indent}Astropy period: "
                f"{reference_period:.8f} days"
            )
            print(
                f"{indent}period error: "
                f"{abs(float(best_period) - reference_period):.8f} days"
            )

        if reference_power is not None:
            print(f"{indent}Astropy power: {float(reference_power):.8f}")

        if (
                role == "known"
                and best_period is not None
                and science.get("publishedPeriodDays") is not None
        ):
            published_period = float(science["publishedPeriodDays"])
            published_error = abs(float(best_period) - published_period)
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

            if total == 0 or completed != total:
                return

            self.reported_completed_datasets.add(dataset_id)

        print()
        print("🌟 Dataset complete")
        self._print_dataset_result_summary(dataset_id)

    def _report_project_complete(self):
        with self.lock:
            if self.reported_project_complete:
                return

            if len(self.completed) != len(self.work_units):
                return

            self.reported_project_complete = True
            dataset_ids = list(self.datasets.keys())

        print()
        print("🏁 Project complete")
        print(f"   project: {self.project_id}")

        for dataset_id in dataset_ids:
            print()
            self._print_dataset_result_summary(dataset_id)

    def _report_completions(self):
        for dataset_id in self.datasets:
            self._report_dataset_complete(dataset_id)

        self._report_project_complete()

    def print_startup_summary(self, port, host="0.0.0.0"):
        self.validate_startup()

        print()
        print("⭐ OpenStar Coordinator")
        print("Build: multi-target-v11-execution-requeue-verification")
        print(f"File: {Path(__file__).resolve()}")
        print(f"Listening on {host}:{port}")
        print()
        print(f"Project: {self.project_id}")
        print(f"Workload: {self.workload_id}")
        print(f"Datasets: {len(self.datasets)}")
        print(f"Work units: {len(self.work_units)}")
        print(
            "Verification: frequency OR equivalent chunk-max power, with nearby-peak fallback"
        )
        print(
            "Primary match: frequency agreement; secondary match: strict "
            "equivalent chunk power"
        )
        print(
            "Equivalent-power tolerance: "
            f"max({POWER_ABSOLUTE_TOLERANCE:.8f}, "
            "Astropy power * "
            f"{EQUIVALENT_POWER_RELATIVE_TOLERANCE:.4f})"
        )
        print(
            "Nearby-peak power tolerance: "
            f"max({POWER_ABSOLUTE_TOLERANCE:.8f}, "
            "Astropy power * "
            f"{AMBIGUOUS_POWER_RELATIVE_TOLERANCE:.4f})"
        )
        print(
            "Max verification failures/work unit: "
            f"{MAX_VERIFICATION_FAILURES_PER_WORK_UNIT}"
        )
        print(
            "Execution failure handling: requeue without scientific failure; "
            f"avoid same node for {EXECUTION_FAILURE_NODE_AVOID_SECONDS:.0f}s"
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
                "Astropy work-unit references: "
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
