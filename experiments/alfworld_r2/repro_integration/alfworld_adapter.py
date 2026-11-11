"""Critic-only ALFWorld state annotation with persistent physical identities.

The input is the instrumented TextWorld facts interface, not an expert plan.
Only actor observations, admissible commands and own executed moves establish
known locations. Oracle locations may describe a graph but cannot mark known.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re

from repro_core.graphs import ColoredGraph, canonicalize, canonical_transition
from .frontier import frontier_key

FAMILIES = {"pick_clean_then_place_in_recep": "clean",
            "pick_heat_then_place_in_recep": "heat",
            "pick_cool_then_place_in_recep": "cool"}
PROPERTIES = {"clean": ("isclean", "cleanable"),
              "heat": ("ishot", "heatable"), "cool": ("iscool", "coolable")}


def task_uid(gamefile, split=None):
    """Stable across machines while preserving split/task/trial identity."""
    parts = os.path.normpath(gamefile).split(os.sep)
    positions = [i for i, part in enumerate(parts) if part in ("train", "valid_seen", "valid_unseen")]
    relative = "/".join(parts[positions[-1]:]) if positions else os.path.normpath(gamefile)
    return hashlib.sha256(relative.encode()).hexdigest()[:24]


def _type(value):
    value = value.strip().lower()
    return value[:-4] if value.endswith("type") else value


class Facts:
    def __init__(self, info):
        raw = info.get("extra.repro")
        if not raw or raw.get("schema") != "alfworld_facts_v1":
            raise ValueError("Missing verified ALFWorld critic facts interface")
        self.raw = raw
        self.names = {str(k): str(v).lower() for k, v in raw["entity_names"].items()}
        reverse = {v: k for k, v in self.names.items()}
        self.rows = {}
        for fact in raw["facts"]:
            args = tuple(reverse.get(str(a["name"]).strip().lower(), str(a["name"]).strip()) for a in fact["arguments"])
            self.rows.setdefault(fact["name"].lower(), set()).add(args)
        self.object_types = {o: _type(t) for o, t in self.rows.get("objecttype", ())}
        self.receptacle_types = {r: _type(t) for r, t in self.rows.get("receptacletype", ())}
        self.location_sets = {}
        for obj, rec in sorted(self.rows.get("inreceptacle", ())):
            self.location_sets.setdefault(obj, set()).add(rec)
        # Some real ALFWorld worlds contain nested/duplicate memberships. Keep
        # all evidence and reject an ambiguous selected object rather than
        # choosing a location using nondeterministic set iteration.
        self.locations = {obj: sorted(recs)[0] for obj, recs in self.location_sets.items()}
        self.held = {args[1] for args in self.rows.get("holds", ()) if len(args) == 2}
        self.agent_locations = {args[1] for args in self.rows.get("atlocation", ()) if len(args) == 2}
        self.rec_locations = {r: loc for r, loc in self.rows.get("receptacleatlocation", ())}

    def unary(self, name, obj):
        return (obj,) in self.rows.get(name, set())

    def at(self, rec):
        return bool(rec and self.rec_locations.get(rec) in self.agent_locations)

    def accessible(self, rec):
        return bool(rec and (not self.unary("openable", rec) or self.unary("opened", rec)))

    def transformed(self, obj, family):
        return self.unary(PROPERTIES[family][0], obj)

    def can_contain(self, rec, obj):
        rt, ot = self.receptacle_types.get(rec), self.object_types.get(obj)
        return any(_type(a) == rt and _type(b) == ot for a, b in self.rows.get("cancontain", ()))


class AlfworldAnnotator:
    def __init__(self, gamefile, max_steps=50):
        self.gamefile = os.path.realpath(gamefile)
        self.max_steps = max_steps
        self.known_locations = {}
        self.misplaced = {}
        self.witness = None
        self.goal_receptacle = None
        self.step_id = 0
        self.state = None

    def reset(self, observation, info):
        self.known_locations.clear()
        self.misplaced.clear()
        self.witness = self.goal_receptacle = None
        self.step_id = 0
        self.facts = Facts(info)
        actual = os.path.realpath(self.facts.raw["gamefile"])
        if actual != self.gamefile:
            raise ValueError(f"Explicit task reset mismatch: {actual} != {self.gamefile}")
        self.task = self.facts.raw["task"]
        self.family = FAMILIES.get(self.task.get("task_type"), "unsupported")
        self.observation = observation
        self._observe(observation, info, self.facts)
        self.state = self._state(self.facts)
        return copy.deepcopy(self.state)

    def _observe(self, observation, info, facts):
        # A take command contains both visible object and its accessible location.
        commands = info.get("admissible_commands", ())
        reverse = {name: entity for entity, name in facts.names.items()}
        for command in commands:
            match = re.fullmatch(r"take (.+) from (.+)", command.lower())
            if match and match[1] in reverse and match[2] in reverse:
                self.known_locations[reverse[match[1]]] = reverse[match[2]]
        # Observation may expose an object before the next take command is offered.
        text = observation.lower()
        for obj, rec in sorted(facts.rows.get("inreceptacle", ())):
            name = facts.names.get(obj, "")
            if (name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text)
                    and facts.at(rec) and facts.accessible(rec)):
                self.known_locations[obj] = rec
        for obj in facts.held:
            self.known_locations[obj] = "@hand"

    def _select_witness(self, facts):
        params = self.task.get("pddl_params", {})
        object_type, rec_type = _type(params.get("object_target", "")), _type(params.get("parent_target", ""))
        if not object_type or not rec_type or params.get("object_sliced", False):
            return None, None, "unsupported_goal_parameters"
        capability = PROPERTIES[self.family][1]
        objects = sorted(o for o, t in facts.object_types.items() if t == object_type and facts.unary(capability, o))
        receptacles = sorted(r for r, t in facts.receptacle_types.items() if t == rec_type)
        if not objects or not receptacles:
            return None, None, "missing_goal_entity_or_capability"
        # Current holding and achieved property dominate; tie-breaking is causal.
        ranked = sorted(objects, key=lambda o: (o not in facts.held, not facts.transformed(o, self.family),
                                                o != self.witness, o not in self.known_locations, o))
        chosen = ranked[0]
        if chosen != self.witness:
            reason = "initial" if self.witness is None else (
                "compatible_object_held" if chosen in facts.held else "achieved_required_property")
        else:
            reason = "unchanged"
        existing = facts.locations.get(chosen)
        goal = existing if existing in receptacles else self.goal_receptacle if self.goal_receptacle in receptacles else receptacles[0]
        return chosen, goal, reason

    def _state(self, facts, *, terminal=False, success=False):
        if terminal:
            physical = self._state(facts)
            self.misplaced.clear()
            return dict(physical, kind="terminal", terminal=True, success=bool(success), active_conflicts=[])
        if self.family not in PROPERTIES:
            return {"kind": "unsupported", "terminal": False, "reason": "task_family_outside_mvp"}
        old_witness = self.witness
        witness, goal, reason = self._select_witness(facts)
        if witness is None:
            return {"kind": "unsupported", "terminal": False, "reason": reason}
        if len(facts.location_sets.get(witness, ())) > 1:
            return {"kind": "unsupported", "terminal": False, "reason": "multiple_selected_object_locations",
                    "witness": witness, "binding_locations": sorted(facts.location_sets[witness])}
        self.witness, self.goal_receptacle = witness, goal
        if old_witness is not None and old_witness != witness:
            self.misplaced.pop(old_witness, None)
        location = facts.locations.get(witness)
        known = self.known_locations.get(witness) == ("@hand" if witness in facts.held else location)
        transformed = facts.transformed(witness, self.family)
        wrong = sorted(facts.held - {witness})
        marker = self.misplaced.get(witness)
        misplaced = bool(marker and location == marker["receptacle"] and witness not in facts.held
                         and facts.receptacle_types.get(location) != facts.receptacle_types[goal])
        common = {"terminal": False, "witness": witness, "witness_previous": old_witness,
                  "witness_switch_reason": reason, "transform_satisfied": transformed,
                  "known_locations": dict(sorted(self.known_locations.items())),
                  "active_conflicts": (["WRONG_HELD"] if wrong else []) + (["TARGET_MISPLACED"] if misplaced else [])}
        # Switch event belongs to this state once, guaranteeing adjacent states agree.
        if len(wrong) > 1 or (wrong and misplaced):
            return dict(common, kind="unsupported", reason="multiple_dependent_conflicts")
        frontier = frontier_key(held=witness in facts.held, transformed=transformed,
                                location_known=known, at_object=facts.at(location),
                                object_open=facts.accessible(location), at_goal=facts.at(goal),
                                goal_open=facts.accessible(goal))
        try:
            graph = self._graph(facts, witness, goal, wrong[0] if wrong else None,
                                location, known, transformed, misplaced, frontier)
        except ValueError as error:
            return dict(common, kind="unsupported", reason=str(error))
        common.update(graph={"nodes": dict(graph.nodes), "edges": [list(e) for e in graph.edges]}, frontier=frontier)
        if not wrong and not misplaced:
            return dict(common, kind="F")
        try:
            key = canonicalize(graph).key
        except ValueError as error:
            return dict(common, kind="unsupported", reason=str(error))
        return dict(common, kind="D", motif="WRONG_HELD" if wrong else "TARGET_MISPLACED", source_key=key)

    def _graph(self, facts, obj, goal, wrong, location, known, transformed, misplaced, frontier):
        colors, edges = {}, []
        def node(entity, color):
            colors.setdefault(entity, set()).add(color)
        def edge(a, relation, b):
            edges.append((a, relation, b))
        node("@true", "TRUE_GOAL")
        node("@pseudo", "PSEUDO_GOAL")
        node("@hand", "HAND_RESOURCE")
        node("@frontier", "FRONTIER:" + frontier)
        node(obj, "GOAL_OBJECT|required_property=" + str(int(transformed)) + "|location_known=" + str(int(known)))
        node(goal, "GOAL_RECEPTACLE")
        edge("@true", "REQUIRED_OBJECT", obj)
        edge("@true", "GOAL_SLOT", obj)
        edge(obj, "REQUIRED_IN", goal)
        edge("@true", "SHARED_FRONTIER", "@frontier")
        edge("@pseudo", "SHARED_FRONTIER", "@frontier")
        edge("@pseudo", "REPAIR_REJOINS", "@frontier")
        if obj in facts.held:
            edge(obj, "ACTUAL_HELD_BY", "@hand")
        elif location:
            node(location, "OBJECT_LOCATION")
            edge(obj, "ACTUAL_IN", location)
        if wrong:
            # Type compatibility matters; names and task families do not.
            compatible = facts.object_types.get(wrong) == facts.object_types[obj]
            node(wrong, "INTERFERING_OBJECT|compatible=" + str(int(compatible)))
            edge(wrong, "ACTUAL_HELD_BY", "@hand")
            edge("@pseudo", "SUBSTITUTES_SLOT", wrong)
            edge(obj, "REQUIRES_RESOURCE", "@hand")
            for rec in sorted(r for r in facts.rec_locations if facts.at(r)):
                node(rec, "CURRENT_RECEPTACLE")
                edge("@pseudo", "REPAIR_PUT_TARGET", rec)
                node(rec, "CAN_RECEIVE_INTERFERER=" + str(int(facts.can_contain(rec, wrong))))
        if misplaced:
            edge("@pseudo", "SUBSTITUTES_SLOT", location)
            edge(location, "CONFLICTS_WITH", goal)
        receptacles = ({goal, location} | {r for r in colors if r in facts.rec_locations}) - {None}
        for rec in receptacles:
            node(rec, "OPEN=" + str(int(facts.accessible(rec))))
            node(rec, "AT_CURRENT_LOCATION=" + str(int(facts.at(rec))))
            node(rec, "OPENABLE=" + str(int(facts.unary("openable", rec))))
            node(rec, "CAN_RECEIVE_GOAL_OBJECT=" + str(int(facts.can_contain(rec, obj))))
        if len(colors) > 12:
            raise ValueError("local_relation_graph_exceeds_12_nodes")
        return ColoredGraph({n: "&".join(sorted(c)) for n, c in colors.items()}, tuple(set(edges)))

    def step(self, action, observation, info, done=False, original_reward=0):
        if self.state is None or self.state["terminal"] or self.step_id >= self.max_steps:
            raise ValueError("Cannot annotate a decision outside an active episode")
        before = copy.deepcopy(self.state)
        old_facts = self.facts
        facts = Facts(info)
        if os.path.realpath(facts.raw["gamefile"]) != self.gamefile:
            raise ValueError("Game changed without explicit reset")
        selected = self.witness
        # Only an executed placement of the selected physical object starts an event.
        if action.startswith(("put ", "move ")) and selected in old_facts.held and selected in facts.locations:
            rec = facts.locations[selected]
            if facts.receptacle_types.get(rec) != _type(self.task["pddl_params"]["parent_target"]):
                self.misplaced[selected] = {"receptacle": rec, "activated_step": self.step_id}
            else:
                self.misplaced.pop(selected, None)
            self.known_locations[selected] = rec
        for obj in facts.held:
            self.misplaced.pop(obj, None)
        self._observe(observation, info, facts)
        success = bool(info.get("won", False))
        lost = bool(info.get("lost", False))
        # Gym time-limit done is not an environment failure.
        terminated = success or lost
        if done and not terminated and self.step_id + 1 < self.max_steps:
            raise ValueError("Unexpected early ALFWorld done without won/lost predicate")
        after = self._state(facts, terminal=terminated, success=success)
        record = {"step_id": self.step_id, "action": action,
                  "observation_before": self.observation, "observation_after": observation,
                  "budget_before": self.max_steps-self.step_id, "budget_after": self.max_steps-self.step_id-1,
                  "before": before, "after": after, "reward_rel": int(success),
                  "terminated": terminated, "environment_done": bool(done),
                  "original_environment_reward": float(original_reward),
                  "critic_before": copy.deepcopy(old_facts.raw), "critic_after": copy.deepcopy(facts.raw),
                  "admissible_commands_after": list(info.get("admissible_commands", ())),
                  "misplaced_events": copy.deepcopy(self.misplaced)}
        if "graph" in before and "graph" in after:
            bg, ag = before["graph"], after["graph"]
            try:
                record["joint_transition_key"] = canonical_transition(
                    ColoredGraph(bg["nodes"], tuple(bg["edges"])),
                    ColoredGraph(ag["nodes"], tuple(ag["edges"]))).joint_key
            except ValueError as error:
                record["joint_mapping_unsupported"] = str(error)
        # Certify only immediate relation-preserving recovery events. A changed
        # goal property/witness or a task-specific transform makes mapping unknown.
        same_witness = after.get("witness") == before.get("witness")
        same_property = (not terminated and after.get("transform_satisfied") == before.get("transform_satisfied"))
        safe_action = not action.startswith(("clean ", "heat ", "cool ", "slice ", "use "))
        recovery = before["kind"] == "D" and same_witness and safe_action
        if recovery and ((same_property and not terminated) or (success and before.get("transform_satisfied"))):
            kind = "SUCCESS" if success else {"D": "D", "F": "F"}.get(after["kind"], "OTHER")
            key = after.get("source_key", "") if kind == "D" else after.get("frontier", "") if kind == "F" else ""
            record["target_outcomes"] = {family: {"validated": True, "kind": kind, "key": key,
                        "reward_rel": int(success), "terminated": terminated}
                        for family in PROPERTIES}
            record["target_mapping_proof"] = "same_physical_witness_and_required_property; relation_only_action; role_goal_predicates"
        self.facts, self.state, self.observation = facts, after, observation
        self.step_id += 1
        return record
