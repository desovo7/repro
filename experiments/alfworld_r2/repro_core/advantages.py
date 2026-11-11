"""Action-level credit and explicit masks; tensor/trainer integration is external."""

from math import isfinite
from typing import Hashable, Sequence

from .core import ValueEstimate


def supported_gae(
    rewards: Sequence[float], values: Sequence[ValueEstimate], *,
    budgets: Sequence[int], terminated: Sequence[bool] | None = None,
    lambda_: float = 0.5,
) -> list[float]:
    """One complete episode, binary reward, gamma=1, no normalization.

    `budgets` and `values` have T+1 entries. Budget decreases exactly once for
    each environment decision, including a counted invalid decision. A genuine
    terminal or budget-zero successor is zero even if its annotation is unknown.
    Missing support resets the auxiliary GAE recursion before reading its value.
    """
    length = len(rewards)
    if len(values) != length + 1 or len(budgets) != length + 1:
        raise ValueError("values and budgets must have T+1 entries")
    terminal = list(terminated) if terminated is not None else [False] * length
    if len(terminal) != length or any(terminal[:-1]):
        raise ValueError("Pass one episode; no action may follow a genuine terminal")
    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be within [0, 1]")
    if any(not isfinite(r) or r not in (0, 1) for r in rewards):
        raise ValueError("Adapt rewards to binary success {0, 1} before calling")
    if any(not isinstance(b, int) or b < 0 for b in budgets):
        raise ValueError("budgets must be nonnegative integers")
    for before, after in zip(budgets, budgets[1:]):
        if before <= 0 or after != before - 1:
            raise ValueError("Every executed decision consumes exactly one budget unit")
    if length and not terminal[-1] and budgets[-1] > 0:
        raise ValueError("A complete episode must end at a real terminal or budget zero; "
                         "collector-truncated chunks are not supported")
    if any(reward == 1 and not terminal[t] for t, reward in enumerate(rewards)):
        raise ValueError("A success reward must coincide with a genuine terminal")
    result = [0.0] * length
    carry = 0.0
    for t in range(length - 1, -1, -1):
        current = values[t]
        nxt = (ValueEstimate.known(0) if terminal[t] or budgets[t + 1] == 0
               else values[t + 1])
        if not current.supported or not nxt.supported:
            carry = 0.0
        else:
            delta = rewards[t] + nxt.value - current.value
            carry = delta + lambda_ * carry
        result[t] = carry
    return result


def af_group_mask(group_ids: Sequence[Hashable],
                  successes: Sequence[int | bool], *,
                  completed: Sequence[bool] | None = None,
                  expected_group_size: int | None = None) -> list[bool]:
    """Per-trajectory mask. A group ID identifies one sampling group, not a task
    merged across iterations. Partial rewards and invalid-action penalties are
    not substitutes for environment success labels. Pass completion metadata
    to reject interrupted groups and expected size to reject missing members.
    If omitted, the caller must have validated those conditions upstream.
    """
    if len(group_ids) != len(successes):
        raise ValueError("Group IDs and success labels must have equal lengths")
    if any(s not in (0, 1) for s in successes):
        raise ValueError("Success labels must be binary")
    if completed is not None and len(completed) != len(group_ids):
        raise ValueError("Completion flags must match trajectories")
    if expected_group_size is not None and expected_group_size < 1:
        raise ValueError("Expected group size must be positive")
    any_success = {}
    counts = {}
    complete_groups = {}
    for i, (group, success) in enumerate(zip(group_ids, successes)):
        any_success[group] = any_success.get(group, False) or bool(success)
        counts[group] = counts.get(group, 0) + 1
        complete_groups[group] = complete_groups.get(group, True) and (
            completed is None or bool(completed[i])
        )
    return [not any_success[group] and complete_groups[group]
            and (expected_group_size is None or counts[group] == expected_group_size)
            for group in group_ids]


def combine_advantages(base: Sequence[float], auxiliary: Sequence[float], *,
                       is_af: bool, beta: float = 1.0) -> list[float]:
    if len(base) != len(auxiliary):
        raise ValueError("Action advantage lengths must agree")
    if not isfinite(beta) or beta < 0 or any(not isfinite(v) for v in base):
        raise ValueError("beta and base advantages must be finite; beta >= 0")
    if not is_af or beta == 0:
        # Do not perform 0 * NaN for an unused auxiliary branch.
        return list(base)
    if any(not isfinite(v) for v in auxiliary):
        raise ValueError("Applied auxiliary advantages must be finite")
    return [b + beta * a for b, a in zip(base, auxiliary)]


def action_to_token_advantages(
    action_advantages: Sequence[float], token_action_ids: Sequence[int],
    actor_token_mask: Sequence[bool],
) -> list[float]:
    """Broadcast each decision's advantage only to its actor-generated tokens.

    The rollout adapter owns token spans, action parsing, and observation masks.
    It must include exactly the tokens used by the base trainer for that actor
    decision. This function does not choose loss reduction or divide by span
    length; preserve the base trainer's reduction for all comparisons.
    """
    if len(token_action_ids) != len(actor_token_mask):
        raise ValueError("Token IDs and actor mask lengths must agree")
    if any(not isfinite(v) for v in action_advantages):
        raise ValueError("Action advantages must be finite")
    result = []
    for action_id, active in zip(token_action_ids, actor_token_mask):
        if not active:
            result.append(0.0)
        elif not isinstance(action_id, int) or not 0 <= action_id < len(action_advantages):
            raise ValueError("Every trainable actor token needs a valid decision ID")
        else:
            result.append(float(action_advantages[action_id]))
    return result
