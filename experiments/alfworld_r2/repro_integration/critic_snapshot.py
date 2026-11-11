"""Build immutable, prior-iteration RePro critics from the validated window."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from types import MappingProxyType

from repro_core.core import (BudgetBuckets, ContinuationTable, KernelRow,
                             SupportPolicy, ValueEstimate, solve_recovery_values)

from .outcome_mapping import map_transition_outcome
from .trace_schema import content_hash, episode_complete, json_copy


METHODS = {"gigpo", "local_mc", "repro_target_only", "repro_transfer"}


def _state_key(state):
    if state["kind"] == "F":
        return "F", state["frontier"]
    if state["kind"] == "D":
        return "D", state["source_key"]
    return state["kind"], ""


@dataclass(frozen=True)
class CriticSnapshot:
    iteration_id: int
    method: str
    run_id: str
    config_hash: str
    snapshot_id: str
    max_budget: int
    _estimates: object
    _metadata_json: str
    input_action_uids: frozenset
    input_trajectory_uids: frozenset

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    def query(self, family, state, budget):
        if type(budget) is not int or not 0 <= budget <= self.max_budget:
            raise ValueError("Value query has invalid budget")
        if state.get("terminal", False) or state["kind"] == "terminal" or budget == 0:
            return ValueEstimate.known(0, "terminal/budget boundary")
        if self.method == "gigpo":
            return ValueEstimate.unknown("RePro disabled for baseline")
        kind, key = _state_key(state)
        if kind == "unsupported":
            return ValueEstimate.unknown(state.get("reason", "unsupported state"))
        return self._estimates.get((family, kind, key, budget),
                                   ValueEstimate.unknown("state has no supported history"))


def build_snapshot(window, iteration_id, *, method="repro_transfer", max_budget=None,
                   min_trajectories=5, min_donor_tasks=2, kappa=8.0,
                   target_families=("clean", "heat", "cool")):
    """Freeze statistics from completed iterations [i-M, i-1] only.

    Scientific transition fields are retained until target mapping. The starter count table
    alone cannot express a donor's unknown target *immediate reward*, so this
    adapter constructs KernelRow objects only after checking every event. The
    core probability DP and MC/GAE definitions are reused without alteration.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    max_budget = window.max_budget if max_budget is None else max_budget
    if max_budget != window.max_budget:
        raise ValueError("Window and critic budget contracts disagree")
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("Invalid donor pseudocount")
    buckets = BudgetBuckets(max_budget)
    support = SupportPolicy(min_trajectories, min_donor_tasks)
    traces = window.completed_for(iteration_id)
    families = sorted(set(target_families) | {t["task_family"] for t in traces})
    continuation = ContinuationTable(buckets, support)
    mc_all = ContinuationTable(buckets, support)
    rows = defaultdict(list)
    states, frontiers, all_mc_keys = set(), set(), set()
    input_actions, input_trajectories = set(), set()
    for trace in traces:
        if trace["iteration_id"] >= iteration_id or trace["split"] != "train":
            raise ValueError("Current/future iteration or nontraining split leaked into critic")
        family = trace["task_family"]
        tid = json.dumps([trace["run_id"], trace["trajectory_uid"]])
        input_trajectories.add(trace["trajectory_uid"])
        chronological = [(s["before"], s["budget_before"]) for s in trace["steps"]]
        chronological.append((trace["steps"][-1]["after"], trace["steps"][-1]["budget_after"]))
        for state, budget in chronological:
            kind, key = _state_key(state)
            if kind in {"F", "D"}:
                all_mc_keys.add((kind, key))
                mc_all.add_entry(family=family, frontier=json.dumps([kind, key]), budget=budget,
                                 success=trace["success"], task_id=trace["task_uid"],
                                 trajectory_id=tid, episode_complete=episode_complete(trace))
            if kind == "F":
                frontiers.add(key)
                continuation.add_entry(family=family, frontier=key, budget=budget,
                                       success=trace["success"], task_id=trace["task_uid"],
                                       trajectory_id=tid, episode_complete=episode_complete(trace))
        for step in trace["steps"]:
            input_actions.add(step["action_uid"])
            if step["before"]["kind"] == "D":
                key = step["before"]["source_key"]
                states.add(key)
                rows[(family, key, buckets.index(step["budget_before"]))].append((trace, step))

    estimates, kernels, diagnostics = {}, {}, []
    for family in families:
        for kind, key in all_mc_keys:
            if method == "local_mc" or kind == "F":
                for budget in range(1, max_budget + 1):
                    value = (mc_all.query(family, json.dumps([kind, key]), budget)
                             if method == "local_mc" else continuation.query(family, key, budget))
                    estimates[family, kind, key, budget] = value
        if method not in {"repro_target_only", "repro_transfer"}:
            continue
        for key in sorted(states):
            for bucket in range(1, len(buckets.cuts) + 2):
                local = rows.get((family, key, bucket), [])
                donor = [entry for (source, source_key, source_bucket), records in rows.items()
                         if source != family and source_key == key and source_bucket == bucket
                         for entry in records] if method == "repro_transfer" else []
                local_traj = {(t["run_id"], t["trajectory_uid"]) for t, _ in local}
                donor_traj = {(t["task_family"], t["run_id"], t["trajectory_uid"]) for t, _ in donor}
                donor_tasks = {(t["task_family"], t["task_uid"]) for t, _ in donor}
                use_donor = (kappa > 0 and len(donor_traj) >= min_trajectories
                             and len(donor_tasks) >= min_donor_tasks)
                enough = len(local_traj) >= min_trajectories or use_donor
                local_mass, donor_mass, failures = Counter(), Counter(), []
                for pool, records in ((local_mass, local), (donor_mass, donor if use_donor else [])):
                    for trace, step in records:
                        mapped = map_transition_outcome(trace, step, target_family=family)
                        if not mapped.supported:
                            failures.append(dict(action_uid=step["action_uid"], reason=mapped.reason))
                        else:
                            pool[mapped.outcome] += 1
                denominator = len(local) + (kappa if use_donor else 0)
                if failures:
                    # Unknown immediate reward/terminal means no probability model
                    # can be evaluated, even for a single remaining action.
                    probabilities = {}
                    reason = "target outcome mapping unknown: " + failures[0]["reason"]
                else:
                    mass = Counter({o: float(n) for o, n in local_mass.items()})
                    if use_donor:
                        for outcome, count in donor_mass.items():
                            mass[outcome] += kappa * count / len(donor)
                    probabilities = {o: n / denominator for o, n in mass.items()} if denominator else {}
                    reason = "" if enough and probabilities else "insufficient independent local/donor support"
                row = KernelRow(probabilities, bool(enough and probabilities and not failures), reason,
                                len(local_traj), len(donor_traj), len(donor_tasks))
                kernels[family, key, bucket] = row
                if local or donor:
                    diagnostics.append(dict(family=family, source_key=key, bucket=bucket,
                                            local_transitions=len(local), donor_transitions=len(donor),
                                            local_trajectories=len(local_traj),
                                            donor_trajectories=len(donor_traj), donor_tasks=len(donor_tasks),
                                            donor_used=use_donor, supported=row.supported, reason=row.reason,
                                            unknown_target_outcomes=failures,
                                            probabilities=[dict(kind=o.kind, key=o.key, probability=p)
                                                           for o, p in row.probabilities.items()]))
        values = solve_recovery_values(family=family, continuation=continuation,
                                       kernels=kernels, states=states)
        for (key, budget), value in values.values.items():
            estimates[family, "D", key, budget] = value

    settings = dict(method=method, max_budget=max_budget, min_trajectories=min_trajectories,
                    min_donor_tasks=min_donor_tasks, kappa=kappa, target_families=families)
    metadata = dict(run_id=window.run_id, config_hash=window.config_hash,
                    iteration_id=iteration_id, settings=settings,
                    input_iterations=sorted({t["iteration_id"] for t in traces}),
                    input_policy_versions=sorted({t["policy_version"] for t in traces}),
                    input_trajectory_uids=sorted(input_trajectories),
                    input_action_uids=sorted(input_actions), trace_hash=content_hash(traces),
                    window_storage=window.provenance_for(iteration_id),
                    kernel_rows=diagnostics,
                    states=len(states), frontiers=len(frontiers),
                    supported_queries=sum(v.supported for v in estimates.values()),
                    positive_queries=sum(v.supported and v.value > 0 for v in estimates.values()))
    snapshot_id = content_hash(metadata)
    metadata["snapshot_id"] = snapshot_id
    return CriticSnapshot(iteration_id, method, window.run_id, window.config_hash,
                          snapshot_id, max_budget, MappingProxyType(estimates),
                          json.dumps(metadata, sort_keys=True, allow_nan=False),
                          frozenset(input_actions), frozenset(input_trajectories))
