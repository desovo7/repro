"""Validated JSON traces at the boundary between the environment and RePro.

The actor need not receive any of these annotations. Extra JSON fields are
preserved so raw observations, facts and provenance remain auditable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from repro_core.graphs import ColoredGraph, canonicalize


END_REASONS = {"success", "failure", "horizon", "collector_truncation"}


def json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def content_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _nonempty(record: Mapping, field: str) -> None:
    if not isinstance(record.get(field), str) or not record[field]:
        raise ValueError(f"{field} must be a nonempty string")


def validate_state(state: Mapping) -> dict:
    if not isinstance(state, Mapping):
        raise TypeError("State must be a mapping")
    result = json_copy(dict(state))
    kind = result.get("kind")
    if kind not in {"F", "D", "unsupported", "terminal"}:
        raise ValueError("Unknown state kind")
    if type(result.get("terminal", kind == "terminal")) is not bool:
        raise ValueError("State terminal must be a boolean")
    if result.get("terminal", False) != (kind == "terminal"):
        raise ValueError("terminal flag and state kind disagree")
    result.setdefault("terminal", kind == "terminal")
    if kind == "F":
        _nonempty(result, "frontier")
    if kind == "D":
        _nonempty(result, "source_key")
        _nonempty(result, "motif")
        graph = result.get("graph")
        if not isinstance(graph, dict) or not graph.get("nodes"):
            raise ValueError("D state requires a nonempty before graph")
        canonical = canonicalize(ColoredGraph(graph["nodes"], tuple(graph["edges"]))).key
        if result["source_key"] != canonical:
            raise ValueError("D source_key does not equal its before graph canonical key")
    if kind == "unsupported" and not result.get("reason"):
        raise ValueError("Unsupported state requires a reason")
    return result


def trajectory_identity(trace: Mapping) -> tuple[str, str]:
    return trace["run_id"], trace["trajectory_uid"]


def action_identity(trace: Mapping, step: Mapping) -> tuple[str, str, int]:
    return trace["run_id"], trace["trajectory_uid"], step["step_id"]


def episode_complete(trace: Mapping) -> bool:
    return trace["end_reason"] in {"success", "failure", "horizon"}


def validate_trajectory(trace: Mapping, *, max_budget: int = 50) -> dict:
    """Return an independent JSON copy or reject inconsistent physical time."""
    if not isinstance(trace, Mapping):
        raise TypeError("Trajectory must be a mapping")
    result = json_copy(dict(trace))
    for name in ("run_id", "policy_version", "task_uid", "task_family", "group_uid",
                 "trajectory_uid", "split", "gamefile"):
        _nonempty(result, name)
    if type(result.get("iteration_id")) is not int or result["iteration_id"] < 0:
        raise ValueError("iteration_id must be a nonnegative integer")
    if type(result.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if type(result.get("success")) is not bool:
        raise ValueError("success must be the environment's boolean success label")
    if result.get("end_reason") not in END_REASONS:
        raise ValueError("Unknown end_reason")
    if result["success"] != (result["end_reason"] == "success"):
        raise ValueError("Success and end_reason disagree")
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Trajectory must have at least one actual decision")
    uids = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("step_id") != index:
            raise ValueError("step_id must be consecutive actual decision indices from zero")
        _nonempty(step, "action_uid")
        if step["action_uid"] in uids:
            raise ValueError("Repeated action_uid within trajectory")
        uids.add(step["action_uid"])
        for name in ("action", "observation_before", "observation_after"):
            if not isinstance(step.get(name), str):
                raise ValueError(f"{name} must be a string")
        before, after = step.get("budget_before"), step.get("budget_after")
        if type(before) is not int or type(after) is not int or not 0 < before <= max_budget:
            raise ValueError("Invalid real action budget")
        if after != before - 1:
            raise ValueError("Every actual decision consumes exactly one budget unit")
        if step.get("reward_rel") not in (0, 1) or type(step.get("reward_rel")) is bool:
            raise ValueError("reward_rel must be binary numeric success reward")
        if type(step.get("terminated")) is not bool:
            raise ValueError("terminated must be boolean, excluding horizon truncation")
        step["before"] = validate_state(step.get("before"))
        step["after"] = validate_state(step.get("after"))
        if step["before"]["terminal"]:
            raise ValueError("No action may occur after a terminal state")
        if index:
            previous = steps[index - 1]
            if previous["terminated"] or previous["budget_after"] != before:
                raise ValueError("Broken episode chronology")
            if previous["after"] != step["before"]:
                raise ValueError("Adjacent physical state annotations must agree")
        if step["reward_rel"] == 1 and not step["terminated"]:
            raise ValueError("Success reward must end the episode")
        if step["terminated"] != step["after"]["terminal"]:
            raise ValueError("Physical terminal annotation disagrees with step termination")
        if index < len(steps) - 1 and step["terminated"]:
            raise ValueError("No decision may follow success/failure")
    final = steps[-1]
    if result["end_reason"] in {"success", "failure"} and not final["terminated"]:
        raise ValueError("A real success/failure must terminate")
    if result["end_reason"] == "horizon" and (final["budget_after"] != 0 or final["terminated"]):
        raise ValueError("Horizon is budget zero, not a synthetic failure transition")
    if result["end_reason"] == "collector_truncation" and (final["terminated"] or final["budget_after"] == 0):
        raise ValueError("Collector truncation must preserve a nonterminal positive budget")
    if sum(step["reward_rel"] for step in steps) != int(result["success"]):
        raise ValueError("Binary success events disagree with episode success")
    return result
