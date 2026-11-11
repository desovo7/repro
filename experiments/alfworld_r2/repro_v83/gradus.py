"""GRADUS: Group-Relative Advantage from Discovered Unweighted Stages.

Auxiliary credit for all-failed (AF) task groups, built from learned orders
turned into a potential with prescribed equal spacing (no estimated value
magnitudes), and only from text the agent itself sees:

  The text interface (feedback patterns and the five state fields) is
  hand-specified environment knowledge; what is learned is which states are
  frequent on successful paths, their order, and the receptacle priorities.

  C2  Precedence ranks over instruction-grounded predicate states.  States are
      sets of predicates ("an object of slot k has been sighted", "holding slot
      k", "held object processed by verb v", "slot-k object placed in slot-j
      receptacle", "slot-j device turned on here").  Slots are the instruction
      tokens that name entity classes seen in observations; no task-family
      tables and no task metadata files are read.  Ranks come from the mean
      normalised first-visit position of each state on successful donor
      trajectories of the same instruction skeleton.
  C3  Within-stage search order: receptacle classes ranked by how often the
      instruction's object classes are sighted in them (environment text);
      equal rates share one rank.  Search potential = rank-weighted share of
      the room's receptacles whose contents have been revealed; bounded below
      one C2 rank step.
  Off-path: a state seen at least ``off_path_support`` times and never on the
      raw (not loop-erased) path of any successful donor.
  C5  Credit is centred among actions that leave the same abstract state in
      the same task group (fallback: same decision bucket).
  C6  Undiscounted potential differences only (a step's credit is its own
      increment; loops sum to zero); reaching the step limit is not terminal;
      a step the environment reports as having no effect does not inherit
      later credit.

Donors: successful and failed trajectories of mixed groups in the current
iteration, plus every trajectory of the last ``window`` committed iterations.
AF trajectories never contribute to their own iteration's tables.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import groupby
import json
import math
import re
import time

from .credit import budget_bucket, center_within_bucket, group_membership
from .tracker import _observation_details, normalize_observation

_INST = r"[a-z][a-z0-9_]* \d+"
_TASK = re.compile(r"your task is to: (.*)$")
_ROOM = re.compile(r"looking quickly around you, you see (.*?)\. your task is to")
_OBJECTS = re.compile(r"\b(?:a|an) (" + _INST + r")\b")
_ANY_INSTANCE = re.compile(r"\b([a-z][a-z0-9_]*) \d+\b")
_TAKE = re.compile(r"\byou pick up the (" + _INST + r") from the (" + _INST + r")\.")
_PUT = re.compile(r"\byou (?:move|put) the (" + _INST + r") (?:to|in/on|in|on) the (" + _INST + r")\.")
_PROCESS = re.compile(r"\byou ([a-z]+) the (" + _INST + r") using the (" + _INST + r")\.")
_TOGGLE = re.compile(r"\byou turn (on|off) the (" + _INST + r")\.")
_WORD = re.compile(r"[a-z]+")


def entity_class(instance):
    return instance.rsplit(" ", 1)[0]


@dataclass(frozen=True)
class GradusConfig:
    window: int = 4                 # committed iterations kept as donors
    min_success_support: int = 3    # successful donors needed to rank a state
    landmark_fraction: float = 0.5  # share of successful donors that must visit a ranked state
    off_path_support: int = 10      # donors needed to call a never-successful state off-path
    min_reveals: int = 5            # reveals needed to rank a receptacle class
    search_epsilon: float = 0.5     # C3 range as a fraction of one C2 rank step
    unranked_search_weight: float = 0.5
    # delta = difference_gamma * Phi(s') - Phi(s).  1.0 (undiscounted) makes a
    # step's credit its own potential increment: standing still costs nothing,
    # loops sum to exactly zero, and the credit no longer depends on the absolute
    # potential level.  0.98 reproduces the first deployed version.
    difference_gamma: float = 1.0

    def __post_init__(self):
        if self.window < 0 or self.min_success_support < 1 or self.off_path_support < 1 or self.min_reveals < 1:
            raise ValueError('Invalid GRADUS support thresholds')
        if not 0 < self.landmark_fraction <= 1:
            raise ValueError('Landmark fraction must be in (0,1]')
        if not 0 <= self.search_epsilon < 1:
            raise ValueError('C3 must stay strictly below one C2 rank step')
        if not 0 <= self.unranked_search_weight <= 1:
            raise ValueError('Unranked search weight must be in [0,1]')
        if not 0 < self.difference_gamma <= 1:
            raise ValueError('Difference discount must be in (0,1]')


# ---------------------------------------------------------------------------
# Text-only abstraction
# ---------------------------------------------------------------------------

class _WorldState:
    """Causal state built from executed actions and environment feedback only."""

    def __init__(self, room):
        self.room = frozenset(room)
        self.loc = None
        self.held = None
        self.known = {}          # object instance -> host receptacle instance
        self.placed = {}         # object instance -> receptacle (agent-confirmed)
        self.processed = defaultdict(set)
        self.lamp_loc = {}       # turned-on device instance -> location
        self.revealed = set()

    def observe(self, text):
        loc, middle, contents, it = _observation_details(text)
        if loc:
            self.loc = loc
        elif middle:
            self.loc = None
        listings = list(contents)
        if it and it[0]:
            listings.append(it)
        new = []
        for host, objects in listings:
            if host not in self.revealed:
                self.revealed.add(host)
                if host in self.room:
                    new.append(host)
            listed = set(objects)
            for obj in [o for o, h in self.known.items() if h == host and o not in listed]:
                self.known.pop(obj)
            for obj in listed:
                if obj != self.held:
                    self.known[obj] = host
        return new

    def act(self, action, text):
        m = _TAKE.search(text)
        if m and m[1] in action:
            obj = m[1]
            self.held = obj
            self.known.pop(obj, None)
            self.placed.pop(obj, None)
        m = _PUT.search(text)
        if m and self.held == m[1] and m[1] in action:
            obj, recep = m[1], m[2]
            self.held = None
            self.placed[obj] = recep
            self.known[obj] = recep
        m = _PROCESS.search(text)
        if m and self.held == m[2] and m[2] in action:
            self.processed[m[2]].add(m[1])
        m = _TOGGLE.search(text)
        if m and m[2] in action:
            if m[1] == 'on':
                self.lamp_loc[m[2]] = self.loc
            else:
                self.lamp_loc.pop(m[2], None)
        return self.observe(text)

    def key(self, slots):
        held_slot = slots.get(entity_class(self.held)) if self.held else None
        holding = held_slot if held_slot is not None else ('x' if self.held else None)
        verbs = sorted(self.processed.get(self.held, ())) if held_slot is not None else []
        placed = Counter()
        for obj, recep in self.placed.items():
            k, j = slots.get(entity_class(obj)), slots.get(entity_class(recep))
            if k is not None and j is not None:
                placed[(k, j, tuple(sorted(self.processed.get(obj, ()))))] += 1
        sighted = sorted({slots[entity_class(o)] for o in self.known
                          if entity_class(o) in slots and o != self.held and o not in self.placed
                          and slots[entity_class(o)] != held_slot})
        on_here = sorted({slots[entity_class(d)] for d, where in self.lamp_loc.items()
                          if entity_class(d) in slots and where is not None and where == self.loc})
        return json.dumps([sighted, holding, verbs, sorted([list(k) + [n] for k, n in placed.items()]), on_here],
                          separators=(',', ':'))


@dataclass
class GradusTrace:
    task_uid: str
    group_uid: str
    trajectory_uid: str
    success: bool
    complete: bool
    steps: list
    instruction: str
    room: tuple
    states: list = field(default_factory=list)       # abstract keys for s_0..s_T
    noops: list = field(default_factory=list)        # environment reported no effect, per action
    reveals: list = field(default_factory=list)      # newly revealed room receptacles at s_0..s_T
    sightings: list = field(default_factory=list)    # (receptacle instance, object classes listed) per reveal
    classes_seen: set = field(default_factory=set)
    object_classes: set = field(default_factory=set)
    skeleton: str = ''
    slots: dict = field(default_factory=dict)


def _first_observation(raw):
    return normalize_observation(raw['steps'][0]['observation_before'])


def scan_trace(raw):
    """One pass over a raw trajectory, independent of any learned table."""
    first = _first_observation(raw)
    task = _TASK.search(first)
    if not task:
        raise ValueError('Initial observation lacks an instruction')
    room_match = _ROOM.search(first)
    room = tuple(_OBJECTS.findall(room_match[1])) if room_match else ()
    steps = raw['steps']
    success = bool(raw.get('success', False))
    if bool(steps[-1].get('won', False)) != success:
        raise ValueError('Final success and step won disagree')
    complete = success or len(steps) == 50 or raw.get('end_reason') == 'failure'
    trace = GradusTrace(task_uid=str(raw['task_uid']), group_uid=str(raw['group_uid']),
                        trajectory_uid=str(raw['trajectory_uid']), success=success, complete=complete,
                        steps=steps, instruction=task[1].strip(), room=room)
    texts = [first] + [normalize_observation(s['observation_after']) for s in steps]
    for text in texts:
        trace.classes_seen.update(_ANY_INSTANCE.findall(text))
        for host, objects in _observation_details(text)[2]:
            trace.object_classes.update(entity_class(o) for o in objects)
        it = _observation_details(text)[3]
        if it:
            trace.object_classes.update(entity_class(o) for o in it[1])
    return trace


def assign_slots(trace, vocabulary):
    """Instruction tokens naming known entity classes become ordered slots."""
    tokens = _WORD.findall(trace.instruction.lower())
    slots, skeleton = {}, []
    for token in tokens:
        match = None
        for candidate in (token, token[:-1] if token.endswith('s') else None,
                          token[:-2] if token.endswith('es') else None):
            if candidate and candidate in vocabulary:
                match = candidate
                break
        if match is None:
            skeleton.append(token)
        else:
            if match not in slots:
                slots[match] = len(slots)
            skeleton.append(f'<{slots[match]}>')
    trace.slots = slots
    trace.skeleton = ' '.join(skeleton)


def abstract_trace(trace):
    world = _WorldState(trace.room)
    first = _first_observation({'steps': trace.steps})
    new = world.observe(first)
    trace.states = [world.key(trace.slots)]
    trace.reveals = [new]
    trace.sightings = [(h, tuple(sorted({entity_class(o) for o, where in world.known.items() if where == h})))
                       for h in new]
    previous = first
    for step in trace.steps:
        after = normalize_observation(step['observation_after'])
        # Text-level no-op: the environment itself reports no effect.
        trace.noops.append(bool(not step.get('format_valid', True) or after == 'nothing happens.' or after == previous))
        previous = after
        new = world.act(normalize_observation(step['action']), after)
        trace.states.append(world.key(trace.slots))
        trace.reveals.append(new)
        trace.sightings.extend((h, tuple(sorted({entity_class(o) for o, where in world.known.items() if where == h})))
                               for h in new)
    return trace


# ---------------------------------------------------------------------------
# Donor statistics (mergeable, JSON-serialisable)
# ---------------------------------------------------------------------------

def empty_stats():
    return dict(order={}, occurs={}, successes={}, visited={}, reveals={}, sightings={})


def loop_erased(states):
    """Erase every cycle that returns to an earlier state (detours on a success path)."""
    path, index = [], {}
    for key in states:
        if key in index:
            cut = index[key] + 1
            for dropped in path[cut:]:
                index.pop(dropped, None)
            del path[cut:]
        else:
            index[key] = len(path)
            path.append(key)
    return path


def add_trace_stats(stats, trace):
    """Order statistics from successes, occurrence and sighting counts from all."""
    distinct = list(dict.fromkeys(trace.states[:-1] if trace.success else trace.states))
    occurs = stats['occurs'].setdefault(trace.skeleton, {})
    for key in distinct:
        occurs[key] = occurs.get(key, 0) + 1
    if trace.success and distinct:
        stats['successes'][trace.skeleton] = stats['successes'].get(trace.skeleton, 0) + 1
        # Raw path, before loop erasure: a state a success passed through is not
        # off-path even when the loop erasure below drops it.
        visited = stats['visited'].setdefault(trace.skeleton, {})
        for key in set(trace.states):
            visited[key] = visited.get(key, 0) + 1
        order = stats['order'].setdefault(trace.skeleton, {})
        path = loop_erased(trace.states[:-1])
        n = len(path)
        for i, key in enumerate(path):
            total, count = order.get(key, (0.0, 0))
            order[key] = (total + i / n, count + 1)
    for host, listed in trace.sightings:
        r = entity_class(host)
        stats['reveals'][r] = stats['reveals'].get(r, 0) + 1
        for c in listed:
            row = stats['sightings'].setdefault(c, {})
            row[r] = row.get(r, 0) + 1


def merge_stats(parts):
    merged = empty_stats()
    for part in parts:
        for skel, table in part['order'].items():
            dest = merged['order'].setdefault(skel, {})
            for key, (total, count) in table.items():
                t0, c0 = dest.get(key, (0.0, 0))
                dest[key] = (t0 + total, c0 + count)
        for skel, table in part['occurs'].items():
            dest = merged['occurs'].setdefault(skel, {})
            for key, count in table.items():
                dest[key] = dest.get(key, 0) + count
        for skel, count in part.get('successes', {}).items():
            merged['successes'][skel] = merged['successes'].get(skel, 0) + count
        for skel, table in part['visited'].items():
            dest = merged['visited'].setdefault(skel, {})
            for key, count in table.items():
                dest[key] = dest.get(key, 0) + count
        for r, count in part['reveals'].items():
            merged['reveals'][r] = merged['reveals'].get(r, 0) + count
        for c, row in part['sightings'].items():
            dest = merged['sightings'].setdefault(c, {})
            for r, count in row.items():
                dest[r] = dest.get(r, 0) + count
    return merged


# ---------------------------------------------------------------------------
# Learned orders
# ---------------------------------------------------------------------------

@dataclass
class OrderTables:
    ranks: dict          # skeleton -> {state key: rank (1..R)}
    sizes: dict          # skeleton -> R
    off_path: dict       # skeleton -> set of state keys
    search: dict         # object class -> {receptacle class: weight in (0,1]}

    def diagnostics(self, top=8):
        return dict(
            skeletons=len(self.ranks),
            ranked_states=sum(len(v) for v in self.ranks.values()),
            off_path_states=sum(len(v) for v in self.off_path.values()),
            search_object_classes=len(self.search),
            orders={s: [k for k, _ in sorted(r.items(), key=lambda kv: kv[1])][:top]
                    for s, r in sorted(self.ranks.items())},
        )


def build_tables(stats, config):
    ranks, sizes, off_path = {}, {}, {}
    for skel, table in stats['order'].items():
        # Landmarks: states most successful donors pass through.  Detours that
        # only some successes take are left unranked (credit holds through them).
        need = max(config.min_success_support,
                   math.ceil(config.landmark_fraction * stats['successes'].get(skel, 0)))
        ranked = [(total / count, key) for key, (total, count) in table.items() if count >= need]
        ranked.sort()
        if ranked:
            ranks[skel] = {key: i + 1 for i, (_, key) in enumerate(ranked)}
            sizes[skel] = len(ranked)
    for skel, occurs in stats['occurs'].items():
        successful = stats['visited'].get(skel, {})
        off = {key for key, n in occurs.items() if n >= config.off_path_support and key not in successful}
        if off:
            off_path[skel] = off
    search = {}
    eligible = {r for r, n in stats['reveals'].items() if n >= config.min_reveals}
    for c, row in stats['sightings'].items():
        if eligible and row:
            search[c] = _search_weights({r: row.get(r, 0) / stats['reveals'][r] for r in eligible})
    return OrderTables(ranks, sizes, off_path, search)


def _search_weights(rates):
    """weight = 1 - rank/n by descending rate; equal rates share their mean rank."""
    weights, ahead = {}, 0
    for _, tied in groupby(sorted(rates, key=lambda r: -rates[r]), key=rates.get):
        tied = list(tied)
        weights.update(dict.fromkeys(tied, 1.0 - (ahead + (len(tied) - 1) / 2) / len(rates)))
        ahead += len(tied)
    return weights


def potentials(trace, tables, config):
    """Lexicographic ordinal potential for s_0..s_T; C2 rank first, C3 second."""
    ranks = tables.ranks.get(trace.skeleton, {})
    size = tables.sizes.get(trace.skeleton, 0)
    off = tables.off_path.get(trace.skeleton, set())
    objects = [c for c in trace.slots if c in tables.search]
    weights = {}
    for r in trace.room:
        rc = entity_class(r)
        w = max((tables.search[c].get(rc, config.unranked_search_weight) for c in objects),
                default=config.unranked_search_weight)
        weights[r] = w
    norm = sum(weights.values())
    revealed, cover = 0.0, []
    for new in trace.reveals:
        revealed += sum(weights.get(r, 0.0) for r in new)
        cover.append(revealed / norm if norm > 0 else 0.0)
    values, component, unknown = [], None, 0
    for key, c3 in zip(trace.states, cover):
        if key in ranks:
            component = ranks[key]
        elif key in off:
            component = 0
        else:
            unknown += 1
            if component is None:
                component = 0
        values.append((component + config.search_epsilon * c3) / (size + 1))
    if trace.success:
        values[-1] = 1.0 + config.search_epsilon / (size + 1)
    return values, unknown


# ---------------------------------------------------------------------------
# Runtime object
# ---------------------------------------------------------------------------

class Gradus:
    def __init__(self, config: GradusConfig, credit_config):
        self.config = config
        self.credit_config = credit_config
        self.history = deque(maxlen=config.window if config.window > 0 else None)
        self.vocabulary = set()
        self.last_committed_iteration = None
        self.tables = None

    # -- per iteration ------------------------------------------------------
    def adapt(self, raw_traces):
        traces = [scan_trace(raw) for raw in raw_traces]
        for t in traces:
            self.vocabulary.update(t.classes_seen)
        vocabulary = set(self.vocabulary)
        for t in traces:
            assign_slots(t, vocabulary)
            abstract_trace(t)
        return traces

    def credit(self, traces, *, group_size):
        started = time.perf_counter()
        groups, af = group_membership(traces, group_size)
        current = empty_stats()
        for t in traces:
            if not af[t.group_uid]:
                add_trace_stats(current, t)
        self.tables = build_tables(merge_stats(list(self.history) + [current]), self.config)
        cfg = self.credit_config
        raw, deltas, centered = {}, {}, {}
        anchors = defaultdict(list)
        unknown_states = af_states = 0
        for t in traces:
            for step in t.steps:
                uid = step['action_uid']
                if uid in raw:
                    raise ValueError('Duplicate original action identity')
                raw[uid] = deltas[uid] = centered[uid] = 0.0
            if not af[t.group_uid]:
                continue
            phi, unknown = potentials(t, self.tables, self.config)
            unknown_states += unknown
            af_states += len(phi)
            d = [self.config.difference_gamma * phi[i + 1] - phi[i] for i in range(len(t.steps))]
            tail = 0.0
            aux = [0.0] * len(d)
            for i in reversed(range(len(d))):
                # A no-op cuts its own outgoing GAE edge: it cannot inherit later progress.
                decay = 0.0 if t.noops[i] else cfg.lambda_aux
                tail = d[i] + cfg.gamma * decay * tail
                aux[i] = tail
            for i, step in enumerate(t.steps):
                uid = step['action_uid']
                raw[uid], deltas[uid] = aux[i], d[i]
                anchors[(t.group_uid, t.states[i])].append((uid, step['step_id']))
        buckets = defaultdict(list)
        anchor_sizes = []
        for (group, state), members in anchors.items():
            anchor_sizes.append(len(members))
            if len(members) >= 2:
                buckets[('anchor', group, state)] = [uid for uid, _step in members]
            else:
                uid, step_id = members[0]
                buckets[('bucket', group, budget_bucket(step_id))].append(uid)
        zero_variance = 0
        for members in buckets.values():
            values = center_within_bucket([raw[uid] for uid in members])
            if not any(values):
                zero_variance += 1
            for uid, v in zip(members, values):
                centered[uid] = v
        af_uids = [s['action_uid'] for t in traces if af[t.group_uid] for s in t.steps]
        diagnostics = dict(
            groups=len(groups), af_groups=sum(af.values()), actions=len(raw), af_actions=len(af_uids),
            nonzero_raw_aux=sum(raw[u] != 0 for u in af_uids),
            nonzero_centered_actions=sum(centered[u] != 0 for u in af_uids),
            centered_actions_positive=sum(centered[u] > 0 for u in af_uids),
            centered_actions_negative=sum(centered[u] < 0 for u in af_uids),
            anchor_buckets=sum(1 for k in buckets if k[0] == 'anchor'),
            fallback_buckets=sum(1 for k in buckets if k[0] == 'bucket'),
            singleton_anchor_ratio=(sum(1 for n in anchor_sizes if n == 1) / len(anchor_sizes)) if anchor_sizes else 0.0,
            zero_variance_buckets=zero_variance,
            unknown_state_ratio=(unknown_states / af_states) if af_states else 0.0,
            donor_iterations=len(self.history),
            vocabulary=len(self.vocabulary),
            seconds=time.perf_counter() - started,
            **{k: v for k, v in self.tables.diagnostics().items() if k != 'orders'})
        return dict(centered=centered, raw=raw, deltas=deltas, af=af, diagnostics=diagnostics)

    def commit(self, iteration, traces):
        stats = empty_stats()
        for t in traces:
            add_trace_stats(stats, t)
        self.history.append(_jsonable(stats))
        self.last_committed_iteration = int(iteration)

    # -- checkpointing --------------------------------------------------------
    def state_dict(self):
        return dict(history=list(self.history), vocabulary=sorted(self.vocabulary),
                    last_committed_iteration=self.last_committed_iteration,
                    config=self.config.__dict__)

    def load_state_dict(self, state):
        if state['config'] != self.config.__dict__:
            raise ValueError('GRADUS checkpoint configuration mismatch')
        self.history = deque((_jsonable(_from_json(s)) for s in state['history']),
                             maxlen=self.config.window if self.config.window > 0 else None)
        self.vocabulary = set(state['vocabulary'])
        self.last_committed_iteration = state['last_committed_iteration']


def _jsonable(stats):
    return dict(order={s: {k: [t, c] for k, (t, c) in tab.items()} for s, tab in stats['order'].items()},
                occurs=stats['occurs'], successes=stats['successes'], visited=stats['visited'],
                reveals=stats['reveals'], sightings=stats['sightings'])


def _from_json(stats):
    return dict(order={s: {k: (float(t), int(c)) for k, (t, c) in tab.items()} for s, tab in stats['order'].items()},
                occurs={s: {k: int(n) for k, n in tab.items()} for s, tab in stats['occurs'].items()},
                successes={s: int(n) for s, n in stats.get('successes', {}).items()},
                visited={s: {k: int(n) for k, n in tab.items()} for s, tab in stats['visited'].items()},
                reveals={r: int(n) for r, n in stats['reveals'].items()},
                sightings={c: {r: int(n) for r, n in row.items()} for c, row in stats['sightings'].items()})
