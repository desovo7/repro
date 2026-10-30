"""Deterministic, causal ALFWorld state tracking for RePro v8.3.

Only task metadata, environment observations and observed successful actions
establish state. Model explanations and unconfirmed attempted actions do not.
This module imports no learning, environment, or GPU libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import NamedTuple


FAMILY_ALIASES = {
    "simple": "pick_and_place_simple", "clean": "pick_clean_then_place_in_recep",
    "heat": "pick_heat_then_place_in_recep", "cool": "pick_cool_then_place_in_recep",
    "look_at": "look_at_obj_in_light", "pick_two": "pick_two_obj_and_place",
}
STAGES = {"pick_and_place_simple": 2, "pick_clean_then_place_in_recep": 3,
          "pick_heat_then_place_in_recep": 3, "pick_cool_then_place_in_recep": 3,
          "look_at_obj_in_light": 3, "pick_two_obj_and_place": 4}
TOOLS = {"pick_clean_then_place_in_recep": "sinkbasin",
         "pick_heat_then_place_in_recep": "microwave",
         "pick_cool_then_place_in_recep": "fridge",
         "look_at_obj_in_light": "desklamp"}
PROCESS = {"pick_clean_then_place_in_recep": "clean",
           "pick_heat_then_place_in_recep": "heat",
           "pick_cool_then_place_in_recep": "cool"}
_INSTANCE = r"[a-z][a-z0-9_]* \d+"
_ARRIVE = re.compile(r"\byou arrive at (?:the )?(" + _INSTANCE + r")\.")
_TAKE_ACTION = re.compile(r"take (" + _INSTANCE + r") from (" + _INSTANCE + r")")
_TAKE_FEEDBACK = re.compile(r"\byou pick up the (" + _INSTANCE + r") from the (" + _INSTANCE + r")\.")
_PUT_ACTION = re.compile(r"(?:move|put) (" + _INSTANCE + r") (?:to|in/on|in|on) (" + _INSTANCE + r")")
_PUT_FEEDBACK = re.compile(r"\byou (?:move|put) the (" + _INSTANCE + r") (?:to|in/on|in|on) the (" + _INSTANCE + r")\.")
_PROCESS_ACTION = re.compile(r"(clean|heat|cool) (" + _INSTANCE + r") (?:with|using) (" + _INSTANCE + r")")
_PROCESS_FEEDBACK = re.compile(r"\byou (clean|heat|cool) the (" + _INSTANCE + r") using the (" + _INSTANCE + r")\.")
_CONTENTS = re.compile(r"\b(?:on|in) (?:the )?(" + _INSTANCE + r"), you see ([^.]+)\.")
_CONTENTS_IT = re.compile(r"\bin it, you see ([^.]+)\.")
_OPEN_HOST = re.compile(r"\b(?:you open the|the) (" + _INSTANCE + r")(?:\.| is open\.)")
_OBJECTS = re.compile(r"\b(?:a|an) (" + _INSTANCE + r")\b")
_LAMP_ACTION = re.compile(r"use (desklamp \d+)")
_LAMP_FEEDBACK = re.compile(r"\byou turn (on|off) the (desklamp \d+)\.")


@lru_cache(maxsize=32768)
def normalize_observation(observation: str) -> str:
    """Whitespace and case normalization; preserve all object instance IDs."""
    return " ".join(observation.split()).lower()


norm_observation = normalize_observation


@lru_cache(maxsize=16384)
def _observation_details(text):
    """Parse repeated environment text once; this cache contains no state."""
    arrival = _ARRIVE.search(text)
    loc = arrival[1] if arrival else None
    middle = "you are in the middle of a room" in text
    contents = tuple((m[1], tuple(_OBJECTS.findall(m[2]))) for m in _CONTENTS.finditer(text))
    it = _CONTENTS_IT.search(text)
    if it:
        hosts = list(_OPEN_HOST.finditer(text))
        # "In it" may also describe an examined object (e.g. a mug).
        # Resolve only an arrival/open host explicitly reported in this same
        # observation; the player's location is not a reliable antecedent.
        it = (hosts[-1][1] if hosts else loc, tuple(_OBJECTS.findall(it[1])))
    return loc, middle, contents, it


@lru_cache(maxsize=8192)
def _action_details(action):
    """Exact action parsing, independent of any trajectory state."""
    if action.startswith('take '):
        m = _TAKE_ACTION.fullmatch(action)
        return ('take', m.groups()) if m else (None, ())
    if action.startswith(('move ', 'put ')):
        m = _PUT_ACTION.fullmatch(action)
        return ('place', m.groups()) if m else (None, ())
    if action.startswith(('clean ', 'heat ', 'cool ')):
        m = _PROCESS_ACTION.fullmatch(action)
        return ('process', m.groups()) if m else (None, ())
    if action.startswith('use desklamp '):
        m = _LAMP_ACTION.fullmatch(action)
        return ('lamp', m.groups()) if m else (None, ())
    return None, ()


@lru_cache(maxsize=16384)
def _action_feedback(kind, text):
    pattern = {'take': _TAKE_FEEDBACK, 'place': _PUT_FEEDBACK,
               'process': _PROCESS_FEEDBACK, 'lamp': _LAMP_FEEDBACK}.get(kind)
    match = pattern.search(text) if pattern else None
    return match.groups() if match else ()


def object_class(instance: str | None) -> str | None:
    return instance.rsplit(" ", 1)[0] if instance else None


def _class(value):
    if value is None or str(value).lower() in {"none", "", "-1"}:
        return None
    return str(value).strip().lower()


@dataclass(frozen=True, slots=True)
class Task:
    family: str
    target_type: str
    goal_recep_type: str | None = None
    tool_recep_type: str | None = None

    def __post_init__(self):
        family = FAMILY_ALIASES.get(self.family, self.family)
        if family not in STAGES:
            raise ValueError(f"Unsupported task family: {self.family}")
        target, goal = _class(self.target_type), _class(self.goal_recep_type)
        if not target or not re.fullmatch(r"[a-z][a-z0-9_]*", target):
            raise ValueError("Task requires an exact canonical target object class")
        if family == "look_at_obj_in_light":
            goal = None
        elif not goal:
            raise ValueError("Placement tasks require a destination class")
        tool = _class(self.tool_recep_type) or TOOLS.get(family)
        if tool != TOOLS.get(family):
            raise ValueError("Tool class differs from the v8.3 task contract")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "target_type", target)
        object.__setattr__(self, "goal_recep_type", goal)
        object.__setattr__(self, "tool_recep_type", tool)

    @property
    def K(self) -> int:
        return STAGES[self.family]

    @property
    def terminal_stage(self) -> int:
        return self.K

    @property
    def k(self) -> int:
        return self.K

    @classmethod
    def from_gamefile(cls, path):
        """Read goal metadata when present, otherwise parse a portable game ID.

        The fallback handles saved logs with inaccessible remote absolute paths.
        It never reads expert plans or initializes an ALFWorld environment.
        """
        path = Path(path)
        metadata = path if path.name == "traj_data.json" else path.parent / "traj_data.json"
        if metadata.is_file():
            data = json.loads(metadata.read_text())
            params = data["pddl_params"]
            if params.get("object_sliced", False):
                raise ValueError("Sliced-object tasks are outside the v8.3 six-family specification")
            family = data["task_type"]
            tool = params.get("toggle_target") if family == "look_at_obj_in_light" else None
            return cls(family, params["object_target"], params.get("parent_target"), tool)
        for part in reversed(path.parts):
            fields = part.split("-")
            if len(fields) >= 5 and fields[0] in STAGES:
                family, target, _, destination = fields[:4]
                return cls(family, target, None if family == "look_at_obj_in_light" else destination,
                           destination if family == "look_at_obj_in_light" else None)
        raise ValueError(f"Cannot establish task metadata from game file or ID: {path}")


class Snapshot(NamedTuple):
    p: int
    nav_bit: int
    loc: str | None
    held_obj: str | None
    delivered_objs: frozenset[str]
    processed_objs: frozenset[str]
    unprocessed_in_goal: frozenset[str]
    seen_target_locs: dict[str, str]
    seen_lamp_locs: frozenset[str]
    lit_lamp_locs: frozenset[str]
    visited_receps_subgoal: frozenset[str]
    known_candidates: frozenset[str]
    won: bool
    observation: str | None
    prelit_bonus: float = 0.1

    @property
    def b_nav(self):
        return self.nav_bit

    @property
    def held_obj_t(self):
        return self.held_obj


class Transition(NamedTuple):
    before: Snapshot
    after: Snapshot
    no_op: bool
    no_op_reasons: tuple[str, ...]
    first_lamp_activation: bool
    feedback_missing: bool


def is_search(task: Task, p: int) -> bool:
    return (p == 0 or (task.family == "pick_two_obj_and_place" and p == 2)
            or (task.family == "look_at_obj_in_light" and p == 1))


def prior_potential(task: Task, snapshot: Snapshot) -> float:
    value = -0.15 if snapshot.p == -1 else snapshot.p / task.K
    if task.family == "look_at_obj_in_light" and snapshot.p in {0, 1} and snapshot.lit_lamp_locs:
        value += snapshot.prelit_bonus
    return value


class TrajectoryStateTracker:
    def __init__(self, task: Task, prelit_bonus=0.1):
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if not 0 <= prelit_bonus <= 1:
            raise ValueError("prelit_bonus must be within [0, 1]")
        self.task, self.prelit_bonus = task, float(prelit_bonus)
        self.state = None

    def reset(self, observation: str, won=False) -> Snapshot:
        self.loc = self.held_obj = None
        self.delivered_objs = self.processed_objs = self.unprocessed_in_goal = frozenset()
        self.seen_target_locs, self.lamp_hosts = {}, {}
        self.seen_lamp_locs = self.lit_lamp_locs = frozenset()
        self.visited_receps_subgoal = frozenset()
        self._observe(normalize_observation(observation))
        p = self._progress(bool(won))
        if self.loc is not None and is_search(self.task, p):
            self.visited_receps_subgoal = frozenset((self.loc,))
        self.state = self._snapshot(p, 0 if is_search(self.task, p) else 1,
                                    frozenset(), bool(won), observation)
        return self.state

    def _progress(self, won):
        task = self.task
        if won:
            return task.K
        if task.family == "pick_two_obj_and_place":
            if len(self.delivered_objs) >= 2:
                return task.K
        elif task.family != "look_at_obj_in_light" and self.delivered_objs:
            return task.K
        holding_target = object_class(self.held_obj) == task.target_type
        if (self.held_obj is not None and not holding_target) or self.unprocessed_in_goal:
            return -1
        if task.family == "pick_two_obj_and_place":
            return 2 * len(self.delivered_objs) + int(holding_target and self.held_obj not in self.delivered_objs)
        if not holding_target:
            return 0
        if task.family in PROCESS:
            return 2 if self.held_obj in self.processed_objs else 1
        if task.family == "look_at_obj_in_light":
            return 2 if self.loc in self.seen_lamp_locs else 1
        return 1

    def _observe(self, text):
        location, middle, contents, it = _observation_details(text)
        if location:
            self.loc = location
        elif middle:
            self.loc = None
        for host, objects in contents:
            self._contents(host, objects)
        # Native ALFWorld uses "In it" for opened containers, including in
        # arrival observations. Resolve it only to a host in this observation.
        if it:
            host = it[0]
            if host:
                self._contents(host, it[1])

    def _contents(self, host, objects):
        visible_targets = {obj for obj in objects if object_class(obj) == self.task.target_type}
        # A complete contents listing disproves stale remembered locations.
        for obj in tuple(self.seen_target_locs):
            if self.seen_target_locs[obj] == host and obj not in visible_targets:
                self.seen_target_locs.pop(obj)
        for obj in visible_targets:
            if obj != self.held_obj:
                self.seen_target_locs[obj] = host
        for obj in objects:
            if object_class(obj) == "desklamp":
                if host not in self.seen_lamp_locs:
                    self.seen_lamp_locs = self.seen_lamp_locs | {host}
                self.lamp_hosts[obj] = host

    def _snapshot(self, p, nav, candidates, won, observation):
        return Snapshot(p, nav, self.loc, self.held_obj,
            self.delivered_objs, self.processed_objs,
            self.unprocessed_in_goal, dict(self.seen_target_locs),
            self.seen_lamp_locs, self.lit_lamp_locs,
            self.visited_receps_subgoal, candidates, won,
            observation, self.prelit_bonus)

    def _candidates(self, before, p):
        if self.task.family == "look_at_obj_in_light" and p == 1:
            return before.seen_lamp_locs
        if p == 0 or (self.task.family == "pick_two_obj_and_place" and p == 2):
            return frozenset(recep for obj, recep in before.seen_target_locs.items()
                             if obj not in self.delivered_objs and obj != self.held_obj)
        return frozenset()

    def step(self, action: str, next_observation: str | None, won=False, format_valid=True) -> Transition:
        if self.state is None:
            raise RuntimeError("reset must precede step")
        before = self.state
        if before.won:
            raise RuntimeError("Cannot step a completed successful trajectory")
        action = normalize_observation(action)
        text = normalize_observation(next_observation) if next_observation is not None else None
        reasons = []
        if not format_valid:
            reasons.append("invalid_format")
        if text == "nothing happens.":
            reasons.append("environment_rejection")
        if text is not None and before.observation is not None and text == normalize_observation(before.observation):
            reasons.append("text_invariance")
        if action.startswith("examine ") or action in {"look", "inventory"}:
            reasons.append("redundant_inspection")
        kind, action_args = _action_details(action)
        feedback = _action_feedback(kind, text) if kind and text is not None else ()
        if (kind == 'lamp' and before.loc in before.lit_lamp_locs
                and not (feedback and feedback[0] == "off")):
            reasons.append("lamp_already_on")
        first_lamp = False
        if text is not None:
            # Apply a state-changing response only when the executed action
            # names the same exact object/receptacle as the environment.
            if kind == 'take' and action_args == feedback:
                obj = feedback[0]
                self.held_obj = obj
                if obj in self.delivered_objs:
                    self.delivered_objs = self.delivered_objs - {obj}
                if obj in self.unprocessed_in_goal:
                    self.unprocessed_in_goal = self.unprocessed_in_goal - {obj}
                self.seen_target_locs.pop(obj, None)
            elif kind == 'place' and action_args == feedback and before.held_obj == feedback[0]:
                obj, recep = feedback
                self.held_obj = None
                if obj in self.delivered_objs:
                    self.delivered_objs = self.delivered_objs - {obj}
                if obj in self.unprocessed_in_goal:
                    self.unprocessed_in_goal = self.unprocessed_in_goal - {obj}
                if object_class(obj) == self.task.target_type:
                    self.seen_target_locs[obj] = recep
                    if object_class(recep) == self.task.goal_recep_type:
                        if self.task.family not in PROCESS or obj in self.processed_objs:
                            self.delivered_objs = self.delivered_objs | {obj}
                        else:
                            self.unprocessed_in_goal = self.unprocessed_in_goal | {obj}
            elif (kind == 'process' and action_args == feedback
                  and before.held_obj == feedback[1]
                  and feedback[0] == PROCESS.get(self.task.family)
                  and object_class(feedback[2]) == self.task.tool_recep_type):
                if feedback[1] not in self.processed_objs:
                    self.processed_objs = self.processed_objs | {feedback[1]}
            if kind == 'lamp' and feedback and action_args[0] == feedback[1]:
                host = before.loc or self.lamp_hosts.get(action_args[0])
                if host:
                    if host not in self.seen_lamp_locs:
                        self.seen_lamp_locs = self.seen_lamp_locs | {host}
                    self.lamp_hosts[action_args[0]] = host
                    if feedback[0] == "on":
                        first_lamp = host not in self.lit_lamp_locs
                        if first_lamp:
                            self.lit_lamp_locs = self.lit_lamp_locs | {host}
                    else:
                        if host in self.lit_lamp_locs:
                            self.lit_lamp_locs = self.lit_lamp_locs - {host}
            self._observe(text)
        # Terminal success has priority in progress-state ordering only.
        # No-op conditions are independent and must not be cleared by a reward
        # flag: that would make the real winning-step audit pass by construction.
        no_op = bool(reasons)
        p = self._progress(bool(won))
        candidates = self._candidates(before, p)
        if not is_search(self.task, p):
            nav = 1
        elif no_op or text is None or self.loc is None:
            nav = 0
        elif candidates:
            nav = int(self.loc in candidates)
        else:
            previous_visits = before.visited_receps_subgoal if before.p == p else frozenset()
            nav = int(self.loc not in previous_visits)
        if p != before.p:
            self.visited_receps_subgoal = frozenset()
        if is_search(self.task, p) and self.loc is not None and self.loc not in self.visited_receps_subgoal:
            self.visited_receps_subgoal = self.visited_receps_subgoal | {self.loc}
        self.state = self._snapshot(p, nav, candidates, bool(won), next_observation)
        return Transition(before, self.state, no_op, tuple(reasons) if no_op else (),
                          first_lamp, next_observation is None)
