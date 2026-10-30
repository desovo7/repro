"""Online mechanism evidence on unique decisions, separate from PPO copies."""
from collections import Counter, defaultdict
import math

from .critic_snapshot import build_snapshot
from .trainer_hook import compute_action_credits


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, min=None, median=None, max=None, nonzero=0)
    return dict(n=len(values), mean=sum(values) / len(values), min=values[0],
                median=values[len(values)//2], max=values[-1],
                nonzero=sum(abs(v) > 1e-12 for v in values))


def bucket(budget):
    return 0 if budget <= 10 else 1 if budget <= 25 else 2


def online_diagnostics(window, iteration, trajectories, *, kappa=8.):
    target = build_snapshot(window, iteration, method="repro_target_only", kappa=kappa)
    transfer = build_snapshot(window, iteration, method="repro_transfer", kappa=kappa)
    target_credits = compute_action_credits(target, trajectories)
    transfer_credits = compute_action_credits(transfer, trajectories)
    counts, unknown = Counter(), Counter()
    motif_trajectories = defaultdict(set)
    recovery_times, pair_values, credit_differences, common_credit_differences = [], [], [], []
    d_credit_differences, cross_bucket_td = [], []
    parser_counts = Counter()
    supported_by_kind = Counter()
    positive_by_kind = Counter()
    for trace in trajectories:
        active_conflict = None
        previous_action = None
        for step in trace["steps"]:
            before, after = step["before"], step["after"]
            counts["decisions"] += 1
            counts["generated_tokens"] += int(sum(step.get("response_mask", [])))
            counts["prompt_tokens"] += int(step.get("prompt_tokens", 0))
            counts["eos_terminated_decisions"] += int(step.get("eos_terminated", False))
            counts["response_limit_hit"] += int(step.get("response_limit_hit", False))
            counts["consecutive_repeated_actions"] += int(previous_action == step["action"])
            previous_action = step["action"]
            for flag in ("format_valid", "executable", "unexpected_thinking"):
                parser_counts[flag] += int(step.get("parser", {}).get(flag, False))
            if before["kind"] == "D":
                motif = before["motif"]
                signature = (motif, before.get("witness"))
                counts["motif_steps/" + motif] += 1
                motif_trajectories[motif].add(trace["trajectory_uid"])
                if active_conflict is not None and active_conflict[1:] != signature:
                    counts["nonrecovery_exit/conflict_changed"] += 1
                    active_conflict = None
                if active_conflict is None:
                    active_conflict = (step["step_id"], *signature)
                    counts["conflict_entries/" + motif] += 1
            if (active_conflict is not None and after["kind"] == "D"
                    and (after["motif"], after.get("witness")) != active_conflict[1:]):
                counts["nonrecovery_exit/conflict_changed"] += 1
                active_conflict = None
            if active_conflict is not None and after["kind"] != "D":
                if before.get("witness") != after.get("witness"):
                    counts["conflict_closed_with_witness_switch"] += 1
                if after["kind"] == "F":
                    counts["observed_recoveries"] += 1
                    recovery_times.append(step["step_id"] - active_conflict[0] + 1)
                else:
                    counts["nonrecovery_exit/" + after["kind"]] += 1
                active_conflict = None
            tv = [target.query(trace["task_family"], state, budget_) for state, budget_ in
                  ((before, step["budget_before"]), (after, step["budget_after"]))]
            rv = [transfer.query(trace["task_family"], state, budget_) for state, budget_ in
                  ((before, step["budget_before"]), (after, step["budget_after"]))]
            counts["queries/" + before["kind"]] += 1
            supported_by_kind[before["kind"]] += int(rv[0].supported)
            positive_by_kind[before["kind"]] += int(rv[0].supported and rv[0].value > 0)
            if not rv[0].supported:
                unknown[rv[0].reason] += 1
            if rv[0].supported and tv[0].supported:
                pair_values.append(rv[0].value - tv[0].value)
            common = all(v.supported for v in rv + tv)
            donor_new_support = all(v.supported for v in rv) and not all(v.supported for v in tv)
            counts["donor_new_supported_actions"] += int(donor_new_support)
            uid = step["action_uid"]
            delta = transfer_credits.aux_by_action_uid[uid] - target_credits.aux_by_action_uid[uid]
            credit_differences.append(delta)
            if common:
                common_credit_differences.append(delta)
            if before["kind"] == "D" or after["kind"] == "D":
                d_credit_differences.append(delta)
            before_key = before.get("source_key") if before["kind"] == "D" else before.get("frontier")
            after_key = after.get("source_key") if after["kind"] == "D" else after.get("frontier")
            if (before["kind"] == after["kind"] and before_key == after_key
                    and bucket(step["budget_before"]) != bucket(step["budget_after"]) and all(v.supported for v in rv)):
                cross_bucket_td.append(step["reward_rel"] + rv[1].value - rv[0].value)
        if active_conflict is not None:
            counts["unrecovered_at_" + trace["end_reason"]] += 1
    kernel_rows = transfer.metadata["kernel_rows"]
    other_mass = [sum(p["probability"] for p in row["probabilities"] if p["kind"] == "OTHER")
                  for row in kernel_rows if row["supported"]]
    return dict(counts=dict(counts), parser_counts=dict(parser_counts),
        motif_independent_trajectories={k: len(v) for k, v in motif_trajectories.items()},
        supported_by_kind=dict(supported_by_kind), positive_by_kind=dict(positive_by_kind),
        unknown_reasons=dict(unknown), recovery_time_conditional=distribution(recovery_times),
        paired_supported_value_transfer_minus_target=distribution(pair_values),
        af_credit_transfer_minus_target=distribution(credit_differences),
        common_supported_af_credit_difference=distribution(common_credit_differences),
        d_related_af_credit_difference=distribution(d_credit_differences),
        unchanged_state_cross_bucket_td=distribution(cross_bucket_td),
        supported_kernel_other_mass=distribution(other_mass),
        target_snapshot_id=target.snapshot_id, transfer_snapshot_id=transfer.snapshot_id,
        kernel_rows=kernel_rows,
        interpretation="Value differences use only jointly supported queries; auxiliary differences retain explicit support gating. Recovery time is conditional on observed F exits; unrecovered and unknown exits are counted separately.")
