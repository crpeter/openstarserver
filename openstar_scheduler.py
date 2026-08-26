from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import Investigation, InvestigationStore, utc_now_iso
from openstar_lifecycle import (
    InvestigationLifecycleDriver,
    InvestigationPreparation,
    InvestigationSchedulingState,
)
from openstar_targets import (
    BranchPlanner,
    InvestigationTarget,
    InvestigationTargetSource,
)


@dataclass(frozen=True)
class InvestigationScheduleOutcome:
    target: InvestigationTarget
    investigation: Investigation
    state: InvestigationSchedulingState
    error: BaseException | None = None


@dataclass(frozen=True)
class SchedulingRoundResult:
    outcomes: tuple[InvestigationScheduleOutcome, ...]
    dispatched_investigation_ids: tuple[str, ...]
    raised_investigation_ids: tuple[str, ...] = ()

    @property
    def immediately_runnable(self) -> bool:
        return any(
            item.state == InvestigationSchedulingState.RUNNABLE
            for item in self.outcomes
        )


class InvestigationScheduler:
    """Admit and concurrently dispatch independent durable investigations."""

    def __init__(
        self,
        store: InvestigationStore,
        dispatcher: InvestigationDispatcher,
        target_source: InvestigationTargetSource,
        planners: dict[str, BranchPlanner],
        *,
        software_id: str,
        software_version: str,
        max_concurrent_investigations: int | None = None,
        concurrency_limit: int | None = None,
    ):
        if concurrency_limit is not None:
            if max_concurrent_investigations is not None:
                raise ValueError("Specify only one scheduler concurrency limit.")
            max_concurrent_investigations = concurrency_limit
        if (
            max_concurrent_investigations is not None
            and max_concurrent_investigations < 1
        ):
            raise ValueError("max_concurrent_investigations must be positive.")
        self.store = store
        self.target_source = target_source
        self.max_concurrent_investigations = max_concurrent_investigations
        self.driver = InvestigationLifecycleDriver(
            store,
            dispatcher,
            planners,
            software_id=software_id,
            software_version=software_version,
        )
        # Operational, process-lifetime quarantine only.  This is deliberately
        # not persisted as scientific state and is rebuilt by evaluation after
        # a process restart.
        self._preparation_quarantine: dict[str, InvestigationScheduleOutcome] = {}

    def admit_targets(self) -> tuple[InvestigationTarget, ...]:
        targets = tuple(self.target_source.enumerate_targets())
        target_ids: set[str] = set()
        investigation_ids: set[str] = set()
        for target in targets:
            if target.id in target_ids:
                raise ValueError(
                    f"Target source returned duplicate target ID: {target.id}"
                )
            if target.investigation_id in investigation_ids:
                raise ValueError(
                    "Targets map to duplicate investigation ID: "
                    f"{target.investigation_id}"
                )
            target_ids.add(target.id)
            investigation_ids.add(target.investigation_id)

        eligible = tuple(
            sorted(
                (target for target in targets if target.eligible),
                key=lambda target: (target.priority, target.id),
            )
        )
        return eligible

    def _investigation_after_preparation_error(
        self, target: InvestigationTarget
    ) -> Investigation:
        """Load authoritative state, or return an unsaved operational projection."""
        if self.store.path_for(target.investigation_id).exists():
            try:
                return self.store.load(target.investigation_id)
            except Exception:
                pass
        now = utc_now_iso()
        return Investigation(
            id=target.investigation_id,
            workflow_id=target.workflow_id,
            workflow_version=target.workflow_version,
            status="UNAVAILABLE",
            created_at=now,
            updated_at=now,
            metadata={"operationalProjection": True},
        )

    def _quarantine_preparation_error(
        self, target: InvestigationTarget, error: Exception
    ) -> InvestigationScheduleOutcome:
        outcome = InvestigationScheduleOutcome(
            target,
            self._investigation_after_preparation_error(target),
            InvestigationSchedulingState.FAILED,
            error,
        )
        self._preparation_quarantine[target.investigation_id] = outcome
        return outcome

    @staticmethod
    def _outcome(
        target: InvestigationTarget,
        preparation: InvestigationPreparation,
        error: BaseException | None = None,
    ) -> InvestigationScheduleOutcome:
        return InvestigationScheduleOutcome(
            target, preparation.investigation, preparation.state, error
        )

    def run_round(
        self, deferred_investigation_ids: set[str] | None = None
    ) -> SchedulingRoundResult:
        deferred = deferred_investigation_ids or set()
        targets = self.admit_targets()
        prepared: list[tuple[InvestigationTarget, InvestigationPreparation]] = []
        preparation_outcomes: dict[str, InvestigationScheduleOutcome] = {}
        for target in targets:
            quarantined = self._preparation_quarantine.get(target.investigation_id)
            if quarantined is not None:
                preparation_outcomes[target.id] = quarantined
                continue
            try:
                prepared.append((target, self.driver.prepare(target)))
            except Exception as error:
                preparation_outcomes[target.id] = self._quarantine_preparation_error(
                    target, error
                )
        prepared_by_id = {target.id: item for target, item in prepared}
        runnable = [
            (target, item)
            for target, item in prepared
            if item.state == InvestigationSchedulingState.RUNNABLE
            and target.investigation_id not in deferred
        ]
        if not runnable:
            return SchedulingRoundResult(
                tuple(
                    preparation_outcomes.get(target.id)
                    or self._outcome(target, prepared_by_id[target.id])
                    for target in targets
                ),
                (),
            )

        workers = self.max_concurrent_investigations or len(runnable)
        results: dict[str, InvestigationScheduleOutcome] = {}
        raised: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.driver.dispatch_runnable, target): target
                for target, _ in runnable
            }
            for future in as_completed(futures):
                target = futures[future]
                try:
                    results[target.id] = self._outcome(target, future.result())
                except BaseException as error:
                    try:
                        prepared_after_error = self.driver.prepare(target)
                    except Exception as preparation_error:
                        results[target.id] = self._quarantine_preparation_error(
                            target, preparation_error
                        )
                    else:
                        raised.append(target.investigation_id)
                        results[target.id] = self._outcome(
                            target, prepared_after_error, error
                        )

        outcomes = []
        for target in targets:
            if target.id in preparation_outcomes:
                outcomes.append(preparation_outcomes[target.id])
                continue
            item = prepared_by_id[target.id]
            outcomes.append(results.get(target.id, self._outcome(target, item)))
        return SchedulingRoundResult(
            tuple(outcomes),
            tuple(target.investigation_id for target, _ in runnable),
            tuple(raised),
        )

    def run_until_idle(self) -> SchedulingRoundResult:
        deferred: set[str] = set()
        deferred_outcomes: dict[str, InvestigationScheduleOutcome] = {}
        dispatched: list[str] = []
        while True:
            result = self.run_round(deferred)
            dispatched.extend(result.dispatched_investigation_ids)
            deferred.update(result.raised_investigation_ids)
            for outcome in result.outcomes:
                if outcome.error is not None:
                    deferred_outcomes[outcome.investigation.id] = outcome
            if not result.dispatched_investigation_ids:
                outcomes = tuple(
                    deferred_outcomes.get(outcome.investigation.id, outcome)
                    for outcome in result.outcomes
                )
                return SchedulingRoundResult(
                    outcomes,
                    tuple(dict.fromkeys(dispatched)),
                    tuple(deferred),
                )

    def run(self) -> SchedulingRoundResult:
        """Run scheduling rounds until no investigation is immediately runnable."""
        return self.run_until_idle()
