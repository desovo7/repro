"""Map donor outcomes only when target reward and termination are established.

Source eligibility is decided before this function. A failed mapping poisons
the complete candidate kernel row; callers must never discard its event.
"""

from dataclasses import dataclass

from repro_core.core import Outcome, SUCCESS, FAILURE, OTHER


@dataclass(frozen=True)
class MappedOutcome:
    outcome: Outcome | None
    supported: bool
    reason: str = ""


def _physical_outcome(state):
    if state["kind"] == "D":
        return Outcome("D", state["source_key"])
    if state["kind"] == "F":
        return Outcome("F", state["frontier"])
    return OTHER


def map_transition_outcome(trace, step, *, target_family):
    """Known nonterminal recovery semantics may transfer; terminal labels may not.

    The adapter may provide ``target_outcomes[family]`` with explicit validated
    ``kind/key/reward_rel/terminated`` derived from target predicates. Otherwise
    it must certify ``after.target_reward_zero_nonterminal`` for cross-family
    nonterminal transitions. Neither flag is inferred from eventual recovery.
    """
    if trace["task_family"] == target_family:
        if step["reward_rel"] == 1:
            return MappedOutcome(SUCCESS, True)
        if step["terminated"]:
            return MappedOutcome(FAILURE, True)
        return MappedOutcome(_physical_outcome(step["after"]), True)

    explicit = step.get("target_outcomes", {}).get(target_family)
    if explicit is not None:
        if not isinstance(explicit, dict) or explicit.get("validated") is not True:
            return MappedOutcome(None, False, "target outcome lacks validated predicate mapping")
        reward, terminated = explicit.get("reward_rel"), explicit.get("terminated")
        if reward not in (0, 1) or type(terminated) is not bool:
            return MappedOutcome(None, False, "unknown target reward or termination")
        try:
            outcome = Outcome(explicit["kind"], explicit.get("key", ""))
        except (KeyError, ValueError):
            return MappedOutcome(None, False, "invalid mapped outcome")
        if (outcome.kind == "SUCCESS") != (reward == 1 and terminated):
            return MappedOutcome(None, False, "mapped success predicates disagree")
        if outcome.kind == "FAILURE" and not (reward == 0 and terminated):
            return MappedOutcome(None, False, "mapped failure predicates disagree")
        if outcome.kind in {"D", "F", "OTHER"} and (reward != 0 or terminated):
            return MappedOutcome(None, False, "nonterminal mapping has uncertain immediate reward")
        return MappedOutcome(outcome, True)

    after = step["after"]
    if (not step["terminated"] and step["reward_rel"] == 0
            and after.get("target_reward_zero_nonterminal") is True):
        return MappedOutcome(_physical_outcome(after), True)
    return MappedOutcome(None, False, "cross-family target reward/termination not established")
