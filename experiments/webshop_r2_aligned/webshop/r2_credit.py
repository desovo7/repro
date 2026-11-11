"""Pure statistics and credit of the deployed ALFWorld R2 (gradus-20261106a).

Source core SHA256: e1b3f5b5752577a54ef1d562768745aa36e82a229e6ed2183275d9acf8c62f6d
The environment supplies instruction-grounded states, candidate entities and
reveal events. No task reward/HTML/parser or optimizer lives in this module.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
import math


def entity_class(instance):
    return instance.rsplit(' ', 1)[0]


@dataclass
class Trace:
    task_uid: str
    group_uid: str
    trajectory_uid: str
    success: bool
    complete: bool
    steps: list
    instruction: str
    room: tuple
    states: list = field(default_factory=list)
    noops: list = field(default_factory=list)
    reveals: list = field(default_factory=list)
    sightings: list = field(default_factory=list)
    skeleton: str = ''
    slots: dict = field(default_factory=dict)


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


def group_membership(traces, expected_group_size=8):
    groups=defaultdict(list)
    seen=set()
    for trace in traces:
        if trace.trajectory_uid in seen:
            raise ValueError('Duplicate original trajectory before batch adjustment')
        seen.add(trace.trajectory_uid)
        groups[trace.group_uid].append(trace)
    flags={}
    for uid,group in groups.items():
        if len(group)!=expected_group_size:
            raise ValueError(f'Incomplete original task group {uid}: {len(group)}')
        if len({t.task_uid for t in group})!=1:
            raise ValueError('Task group contains more than one task')
        for trace in group:
            if not trace.success and not trace.complete:
                raise ValueError('A truncated unsuccessful rollout cannot be classified all-failed')
        flags[uid]=not any(t.success for t in group)
    return groups,flags


def center_within_bucket(values):
    """Specification T2-2: action-weighted centering with a variance floor."""
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite auxiliary credit')
    if len(values)<2:return [0.0]*len(values)
    mean=sum(values)/len(values)
    centered=[v-mean for v in values]
    if math.fsum(v*v for v in centered)/len(centered)<=1e-12:
        return [0.0]*len(values)
    return centered


def budget_bucket(step_id, horizon):
    """Relative form of ALF's [1,10], [11,25], [26,50] buckets."""
    if not 0 <= step_id < horizon:
        raise ValueError('Decision index outside horizon')
    return 0 if step_id < math.ceil(.2 * horizon) else 1 if step_id < math.ceil(.5 * horizon) else 2


def credit(traces, history, config, credit_config, group_size, horizon, table_transform=None, centered_transform=None):
    groups, af = group_membership(traces, group_size)
    current = empty_stats()
    for t in traces:
        if not af[t.group_uid]:
            add_trace_stats(current, t)
    tables = build_tables(merge_stats(list(history) + [current]), config)
    if table_transform is not None:
        tables = table_transform(tables)
    cfg = credit_config
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
        phi, unknown = potentials(t, tables, config)
        unknown_states += unknown
        af_states += len(phi)
        d = [config.difference_gamma * phi[i + 1] - phi[i] for i in range(len(t.steps))]
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
            buckets[('bucket', group, budget_bucket(step_id, horizon))].append(uid)
    zero_variance = 0
    for members in buckets.values():
        values = center_within_bucket([raw[uid] for uid in members])
        if centered_transform is not None:
            values = centered_transform(values, members)
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
        donor_iterations=len(history),
        **{k: v for k, v in tables.diagnostics().items() if k != 'orders'})
    return dict(centered=centered, raw=raw, deltas=deltas, af=af, diagnostics=diagnostics, tables=tables)

