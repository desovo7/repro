"""Reference probability model for RePro v8; no environment or trainer code.

The contract is binary success reward {0, 1}, gamma=1, one row per actual
environment decision, and a finite action budget. This is not a drop-in adapter
for an environment that emits success=10 or invalid-action penalties.
"""

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Callable, Mapping


@dataclass(frozen=True)
class BudgetBuckets:
    max_budget: int
    cuts: tuple[float, ...] = (0.2, 0.5)

    def __post_init__(self):
        if not isinstance(self.max_budget, int) or self.max_budget <= 0:
            raise ValueError("max_budget must be a positive integer")
        cuts = tuple(self.cuts)
        if any(not 0 < cut < 1 for cut in cuts) or any(
            a >= b for a, b in zip(cuts, cuts[1:])
        ):
            raise ValueError("cuts must be strictly increasing within (0, 1)")
        object.__setattr__(self, "cuts", cuts)

    def index(self, budget: int) -> int:
        if not isinstance(budget, int) or not 0 <= budget <= self.max_budget:
            raise ValueError("Budget must be an integer within [0, max_budget]")
        if budget == 0:
            return 0
        for i, cut in enumerate(self.cuts, start=1):
            if budget <= self.max_budget * cut:
                return i
        return len(self.cuts) + 1


@dataclass(frozen=True)
class SupportPolicy:
    min_trajectories: int = 5
    min_donor_tasks: int = 2

    def __post_init__(self):
        if self.min_trajectories < 1 or self.min_donor_tasks < 1:
            raise ValueError("Support thresholds must be positive")


@dataclass(frozen=True)
class ValueEstimate:
    value: float | None
    supported: bool
    reason: str = ""

    def __post_init__(self):
        if not self.supported:
            # Never retain a numeric placeholder, including NaN, for unknown.
            object.__setattr__(self, "value", None)
        elif self.value is None or not isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("A supported probability must be finite in [0, 1]")

    @classmethod
    def known(cls, value: float, reason: str = "") -> "ValueEstimate":
        return cls(float(value), True, reason)

    @classmethod
    def unknown(cls, reason: str) -> "ValueEstimate":
        return cls(None, False, reason)


@dataclass
class _MCRow:
    labels: dict[tuple[str, str], int] = field(default_factory=dict)


class ContinuationTable:
    """First-entry MC per (family, frontier, positive budget bucket, trajectory).

    The caller presents entries chronologically, including re-entry to F and
    entry into a lower budget bucket while staying in F. Trajectory IDs must be
    unique within a task across policy iterations. Artificially interrupted
    episodes have no terminal label and are skipped. Create a fresh table from
    the completed rolling window; do not mutate it during a policy update.
    """

    def __init__(self, buckets: BudgetBuckets,
                 support: SupportPolicy = SupportPolicy()):
        self.buckets = buckets
        self.support = support
        self.rows: dict[tuple[str, str, int], _MCRow] = {}
        self._episode_labels: dict[tuple[str, str, str], int] = {}

    def add_entry(self, *, family: str, frontier: str, budget: int,
                  success: int | bool, task_id: str, trajectory_id: str,
                  episode_complete: bool = True) -> bool:
        bucket = self.buckets.index(budget)
        if success not in (0, 1):
            raise ValueError("MC labels must be binary terminal success")
        if not episode_complete or bucket == 0:
            return False
        episode_key = (family, task_id, trajectory_id)
        label = int(success)
        if episode_key in self._episode_labels and self._episode_labels[episode_key] != label:
            raise ValueError("One trajectory cannot have conflicting final labels")
        self._episode_labels[episode_key] = label
        row = self.rows.setdefault((family, frontier, bucket), _MCRow())
        independent_id = (task_id, trajectory_id)
        if independent_id in row.labels:
            return False
        row.labels[independent_id] = label
        return True

    def query(self, family: str, frontier: str, budget: int) -> ValueEstimate:
        bucket = self.buckets.index(budget)
        if bucket == 0:
            return ValueEstimate.known(0, "budget exhausted")
        row = self.rows.get((family, frontier, bucket))
        n = len(row.labels) if row else 0
        if n < self.support.min_trajectories:
            return ValueEstimate.unknown(f"U has {n} independent trajectories")
        return ValueEstimate.known(sum(row.labels.values()) / n)


@dataclass(frozen=True)
class Outcome:
    kind: str
    key: str = ""

    def __post_init__(self):
        if self.kind not in {"D", "F", "SUCCESS", "FAILURE", "OTHER"}:
            raise ValueError("Unknown recovery outcome kind")
        if self.kind in {"D", "F"} and not self.key:
            raise ValueError("D/F outcomes require a mapped semantic state key")
        if self.kind not in {"D", "F"} and self.key:
            raise ValueError("Terminal/OTHER outcomes do not carry a state key")


SUCCESS = Outcome("SUCCESS")
FAILURE = Outcome("FAILURE")
# OTHER is a nonterminal successor whose abstraction or exit cannot be mapped.
# Immediate success/failure is determined from the environment BEFORE mapping.
OTHER = Outcome("OTHER")


@dataclass
class _TransitionRow:
    counts: Counter = field(default_factory=Counter)
    trajectories: set[tuple[str, str]] = field(default_factory=set)
    tasks: set[str] = field(default_factory=set)

    def add(self, outcome: Outcome, task_id: str, trajectory_id: str):
        self.counts[outcome] += 1
        self.trajectories.add((task_id, trajectory_id))
        self.tasks.add(task_id)


@dataclass(frozen=True)
class KernelRow:
    probabilities: Mapping[Outcome, float]
    supported: bool
    reason: str = ""
    local_trajectories: int = 0
    donor_trajectories: int = 0
    donor_tasks: int = 0

    def __post_init__(self):
        probs = dict(self.probabilities)
        if any(not isinstance(k, Outcome) or not isfinite(p) or p < 0
               for k, p in probs.items()):
            raise ValueError("Kernel probabilities must be nonnegative and finite")
        if probs and abs(sum(probs.values()) - 1) > 1e-10:
            raise ValueError("A nonempty kernel must preserve total probability 1")
        if self.supported and not probs:
            raise ValueError("An empty kernel cannot be supported")
        object.__setattr__(self, "probabilities", MappingProxyType(probs))


class RecoveryCounts:
    """Every real step counts, but support counts independent trajectories.

    `state` must be a BEFORE-only canonical key including compatibility colors.
    Choose donor families by source eligibility, never by observed next outcome.
    A donor outcome mapper returns None for an unmappable outcome; this retains
    its probability as OTHER rather than conditioning on successful mappings.

    If supplied, the mapper applies to EVERY outcome, including SUCCESS and
    FAILURE. A source success may map to a target F exit with remaining work.
    Without a mapper, identity mapping assumes the environment adapter has
    already validated target-task terminal equivalence. Returning OTHER must
    mean a target nonterminal outcome with known immediate reward zero, not an
    unknown target reward or termination condition.
    """

    def __init__(self):
        self.rows: dict[tuple[str, str, int], _TransitionRow] = {}

    def add(self, *, family: str, state: str, bucket: int, outcome: Outcome,
            task_id: str, trajectory_id: str):
        if bucket < 1:
            raise ValueError("Executed actions require a positive-budget bucket")
        self.rows.setdefault((family, state, bucket), _TransitionRow()).add(
            outcome, task_id, trajectory_id
        )

    def kernel(
        self, *, family: str, state: str, bucket: int,
        support: SupportPolicy = SupportPolicy(), kappa: float = 8,
        donor_families: set[str] | None = None,
        map_donor_outcome: Callable[[str, Outcome], Outcome | None] | None = None,
    ) -> KernelRow:
        if not isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        local = self.rows.get((family, state, bucket), _TransitionRow())
        donors = Counter()
        donor_trajectories = set()
        donor_tasks = set()
        for (source_family, source_state, source_bucket), row in self.rows.items():
            if source_family == family or source_state != state or source_bucket != bucket:
                continue
            if donor_families is not None and source_family not in donor_families:
                continue
            donor_trajectories.update((source_family, task, traj)
                                      for task, traj in row.trajectories)
            donor_tasks.update((source_family, task) for task in row.tasks)
            for outcome, count in row.counts.items():
                mapped = outcome
                if map_donor_outcome is not None:
                    mapped = map_donor_outcome(source_family, outcome)
                if mapped is not None and not isinstance(mapped, Outcome):
                    raise TypeError("An outcome mapper must return Outcome or None")
                donors[mapped if mapped is not None else OTHER] += count

        n_local = sum(local.counts.values())
        n_donor = sum(donors.values())
        local_ok = len(local.trajectories) >= support.min_trajectories
        donor_ok = (len(donor_trajectories) >= support.min_trajectories
                    and len(donor_tasks) >= support.min_donor_tasks)
        # Conservative reference rule: a donor pool must independently meet
        # both thresholds; sparse local counts may then shrink toward it.
        use_donor = donor_ok and n_donor > 0 and kappa > 0
        mass = Counter({o: float(n) for o, n in local.counts.items()})
        denominator = float(n_local)
        if use_donor:
            for outcome, count in donors.items():
                mass[outcome] += kappa * count / n_donor
            denominator += kappa
        probabilities = {o: n / denominator for o, n in mass.items()} if denominator else {}
        supported = bool(probabilities) and (local_ok or use_donor)
        return KernelRow(
            probabilities, supported,
            "" if supported else "insufficient independent local/donor support",
            len(local.trajectories), len(donor_trajectories), len(donor_tasks),
        )


@dataclass(frozen=True)
class RecoveryValues:
    family: str
    buckets: BudgetBuckets
    values: Mapping[tuple[str, int], ValueEstimate]

    def query(self, state: str, budget: int) -> ValueEstimate:
        self.buckets.index(budget)
        if budget == 0:
            return ValueEstimate.known(0, "budget exhausted")
        return self.values.get((state, budget),
                               ValueEstimate.unknown("unmodeled recovery state"))


def solve_recovery_values(
    *, family: str, continuation: ContinuationTable,
    kernels: Mapping[tuple[str, str, int], KernelRow],
    states: set[str] | None = None,
) -> RecoveryValues:
    """Finite-budget D-only DP with a fixed MC boundary on return to F.

    No timeout outcome is added. An OTHER successor is unknown while time
    remains, but its continuation is known zero if this action exhausts budget.
    All positive-probability successors must be supported. This deliberately
    conservative rule can yield low coverage; inspect support before training.
    """
    buckets = continuation.buckets
    modeled = set(states or ())
    for (row_family, state, _), row in kernels.items():
        if row_family != family:
            continue
        modeled.add(state)
        modeled.update(o.key for o in row.probabilities if o.kind == "D")
    values = {(state, 0): ValueEstimate.known(0) for state in modeled}
    for budget in range(1, buckets.max_budget + 1):
        bucket = buckets.index(budget)
        for state in sorted(modeled):
            row = kernels.get((family, state, bucket))
            if row is None or not row.supported:
                values[state, budget] = ValueEstimate.unknown("unsupported transition row")
                continue
            estimate = 0.0
            missing = None
            for outcome, probability in row.probabilities.items():
                if probability == 0:
                    continue
                if outcome.kind == "SUCCESS":
                    successor = ValueEstimate.known(1)
                elif outcome.kind == "FAILURE" or budget == 1:
                    successor = ValueEstimate.known(0)
                elif outcome.kind == "F":
                    successor = continuation.query(family, outcome.key, budget - 1)
                elif outcome.kind == "D":
                    successor = values.get((outcome.key, budget - 1),
                                           ValueEstimate.unknown("unmodeled successor"))
                else:
                    successor = ValueEstimate.unknown("OTHER successor has time remaining")
                if not successor.supported:
                    missing = f"{outcome.kind}:{outcome.key}: {successor.reason}"
                    break
                estimate += probability * successor.value
            if missing is not None:
                values[state, budget] = ValueEstimate.unknown(missing)
            else:
                # Clip floating-point roundoff only, not model probabilities.
                values[state, budget] = ValueEstimate.known(min(1.0, max(0.0, estimate)))
    return RecoveryValues(family, buckets, MappingProxyType(values))
