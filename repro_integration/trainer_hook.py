"""RePro action credit before batch duplication, then token-aligned injection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from types import MappingProxyType

from repro_core.advantages import supported_gae

from .trace_schema import episode_complete, validate_trajectory


@dataclass(frozen=True)
class CreditResult:
    aux_by_action_uid: object
    diagnostics: dict


def compute_action_credits(snapshot, trajectories, *, expected_group_size=8,
                           lambda_=0.5, enabled=True, beta=1.0):
    """Return AF-gated *unscaled* credit; beta is applied once at injection.

    beta=0 bypasses all value arithmetic and returns zeros. The parameter is
    accepted here for baseline equivalence checks, never used as a multiplier.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if type(expected_group_size) is not int or expected_group_size < 1:
        raise ValueError("Expected group size must be positive")
    unique = {}
    for raw in trajectories:
        trace = validate_trajectory(raw, max_budget=snapshot.max_budget)
        if trace["run_id"] != snapshot.run_id or trace["split"] != "train":
            raise ValueError("Auxiliary actor training requires this run's training traces")
        if trace["iteration_id"] != snapshot.iteration_id:
            raise ValueError("Snapshot is for a different actor iteration")
        uid = trace["trajectory_uid"]
        if uid in snapshot.input_trajectory_uids:
            raise ValueError("Current trajectory leaked into its critic snapshot")
        if uid in unique and unique[uid] != trace:
            raise ValueError("Conflicting duplicate current trajectory")
        unique[uid] = trace
    groups = defaultdict(list)
    output, owners = {}, {}
    for trace in unique.values():
        groups[trace["group_uid"]].append(trace)
        for step in trace["steps"]:
            uid = step["action_uid"]
            if uid in owners or uid in snapshot.input_action_uids:
                raise ValueError("Action identity collision or snapshot leakage")
            owners[uid] = (trace["trajectory_uid"], step["step_id"])
            output[uid] = 0.0
    counts = Counter()
    reasons = Counter()
    relation_counts = Counter()
    for group, traces in groups.items():
        identities = {(t["run_id"], t["iteration_id"], t["policy_version"], t["task_uid"])
                      for t in traces}
        if len(identities) != 1:
            raise ValueError("group_uid merges distinct tasks, iterations or policies")
        complete = len(traces) == expected_group_size and all(episode_complete(t) for t in traces)
        counts["groups"] += 1
        counts["complete_groups"] += int(complete)
        if not complete:
            counts["excluded_incomplete_groups"] += 1
            continue
        is_af = not any(t["success"] for t in traces)
        counts["af_groups"] += int(is_af)
        if not is_af or not enabled or beta == 0 or snapshot.method == "gigpo":
            continue
        for trace in traces:
            steps = trace["steps"]
            states = [s["before"] for s in steps] + [steps[-1]["after"]]
            budgets = [s["budget_before"] for s in steps] + [steps[-1]["budget_after"]]
            values = [snapshot.query(trace["task_family"], state, budget)
                      for state, budget in zip(states, budgets)]
            for value in values:
                if not value.supported:
                    reasons[value.reason] += 1
            auxiliary = supported_gae([s["reward_rel"] for s in steps], values,
                                       budgets=budgets, terminated=[s["terminated"] for s in steps],
                                       lambda_=lambda_)
            for step, value in zip(steps, auxiliary):
                output[step["action_uid"]] = value
                relation = step["before"]["kind"] + "->" + step["after"]["kind"]
                relation_counts[relation + ":total"] += 1
                relation_counts[relation + ":nonzero"] += int(value != 0)
                counts["nonzero_actions"] += int(value != 0)
    diagnostics = dict(counts)
    diagnostics.update(actions=len(output), trajectories=len(unique),
                       snapshot_id=snapshot.snapshot_id,
                       auxiliary_scaled=False, beta_for_injection=beta,
                       unknown_reasons=dict(reasons), transition_credit_counts=dict(relation_counts))
    return CreditResult(MappingProxyType(output), diagnostics)


def inject_auxiliary(batch, credits, *, enabled=True, beta=1.0):
    """Mutate DataProto advantages once, after base advantages are computed.

    One action identity per decision row survives copy/reorder. Only actual
    actor-generated tokens receive auxiliary credit; padding is left exactly as
    it was in base. Calling this twice on the same batch is an error.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not enabled or beta == 0:
        return dict(applied=False, nonzero_tokens=0)
    import torch

    meta = batch.meta_info
    if meta.get("repro_credit_applied"):
        raise ValueError("RePro credit already applied to this batch")
    auxiliary = credits.aux_by_action_uid if isinstance(credits, CreditResult) else credits
    base, mask = batch.batch["advantages"], batch.batch["response_mask"]
    if base.ndim != 2 or mask.shape != base.shape:
        raise ValueError("Decision/token advantages and response mask must have equal 2D shape")
    if not torch.isfinite(base).all() or not torch.isfinite(mask).all():
        raise ValueError("Nonfinite base advantage or mask")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("response_mask must be binary")
    uids = list(batch.non_tensor_batch["repro_action_uid"])
    if len(uids) != base.shape[0]:
        raise ValueError("Action identities do not align with decision rows")
    values = []
    for index, uid in enumerate(uids):
        if not bool(mask[index].any()):
            values.append(0.0)
            continue
        if uid not in auxiliary:
            raise ValueError(f"Missing auxiliary identity for actor row: {uid}")
        value = auxiliary[uid]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Applied auxiliary credit must be finite")
        values.append(value)
    delta = torch.as_tensor(values, device=base.device, dtype=base.dtype).detach()[:, None]
    delta = beta * delta * mask.to(dtype=base.dtype)
    combined = base + delta
    if not torch.isfinite(combined).all():
        raise ValueError("Nonfinite combined advantages")
    batch.batch["advantages"] = combined.detach()
    meta["repro_credit_applied"] = True
    if isinstance(credits, CreditResult):
        meta["repro_snapshot_id"] = credits.diagnostics.get("snapshot_id")
    return dict(applied=True, nonzero_tokens=int(torch.count_nonzero(delta).item()),
                nonzero_rows=sum(v != 0 for v in values),
                max_abs_delta=float(delta.abs().max().item()) if delta.numel() else 0.0)
