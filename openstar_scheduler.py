from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from openstar_dispatch import InvestigationDispatcher
from openstar_investigation import Investigation, InvestigationStore
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
        for target in eligible:
            self.driver.attach(target)
        return eligible

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
        prepared = [(target, self.driver.prepare(target)) for target in targets]
        runnable = [
            (target, item)
            for target, item in prepared
            if item.state == InvestigationSchedulingState.RUNNABLE
            and target.investigation_id not in deferred
        ]
        if not runnable:
            return SchedulingRoundResult(
                tuple(self._outcome(target, item) for target, item in prepared), ()
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
                    prepared_after_error = self.driver.prepare(target)
                    raised.append(target.investigation_id)
                    results[target.id] = self._outcome(
                        target, prepared_after_error, error
                    )

        outcomes = []
        for target, item in prepared:
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
