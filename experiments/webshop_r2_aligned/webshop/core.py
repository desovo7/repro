"""Transactional WebShop runtime for the deployed ALFWorld R2 credit method."""
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import re

SCHEMA = 'gradus-webshop-r2-aligned-v1'
JVM = '-Xmx128m -Xms16m -XX:+UseSerialGC'
from .adapter import NAV, norm, parts, parse_trace, visible_product, price_relation
from .r2_credit import empty_stats, add_trace_stats, merge_stats, build_tables, credit


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    horizon: int = 15
    group_size: int = 8
    window: int = 4
    min_success_support: int = 3
    landmark_fraction: float = 0.5
    off_path_support: int = 10
    min_reveals: int = 5
    unranked_search_weight: float = 0.5
    difference_gamma: float = 1.0
    search_epsilon: float = 0.5
    gamma: float = 0.98
    lambda_aux: float = 0.5
    beta: float = 0.5
    epsilon: float = 1e-6
    max_abs_injection: float = 5.0
    control: str = 'none'  # none | permute | coverage_only | rank_shuffle
    control_seed: int = 0

    def __post_init__(self):
        if self.horizon != 15 or self.group_size != 8 or self.window < 1:
            raise ValueError('WebShop requires 15 actions, groups of 8 and bounded donor history')
        if self.min_success_support < 1 or not 0 < self.landmark_fraction <= 1:
            raise ValueError('Invalid donor support')
        if not 0 <= self.search_epsilon < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.lambda_aux <= 1:
            raise ValueError('Invalid potential/propagation parameters')
        if not all(math.isfinite(x) and x > 0 for x in (self.beta, self.epsilon, self.max_abs_injection)):
            raise ValueError('Invalid injection parameters')
        if self.off_path_support < 1 or self.min_reveals < 1 or not 0 <= self.unranked_search_weight <= 1 or self.difference_gamma != 1.0:
            raise ValueError('Invalid deployed R2 support or potential parameters')
        if self.control not in {'none', 'permute', 'coverage_only', 'rank_shuffle'}:
            raise ValueError('Unknown control')


def shuffle_ranks(ranks, seed, iteration):
    """Reassign rank labels, retaining supported keys and the rank multiset.

    An identity draw is replaced by one rotation when at least two levels
    exist. Ties can still leave some labels unchanged; report the actual count.
    """
    keys = sorted(ranks)
    values = [ranks[k] for k in keys]
    if len(set(values)) < 2:
        return dict(ranks), 0
    shuffled = list(values)
    random.Random(stable_hash(['rank_shuffle', seed, iteration])).shuffle(shuffled)
    if shuffled == values:
        shuffled = values[1:] + values[:1]
    changed = sum(a != b for a, b in zip(values, shuffled))
    return dict(zip(keys, shuffled)), changed


class CreditEngine:
    def __init__(self, config=Config()):
        self.config = config
        self.history = deque(maxlen=config.window)
        self.last_iteration = 0
        self.pending = None
        self.table_snapshot = None

    def ranks(self):
        stats = merge_stats(self.history)
        tables = build_tables(stats, self.config)
        return tables.ranks, sum(tables.sizes.values()), sum(stats['successes'].values())

    def prepare(self, raws, iteration):
        if self.pending is not None or iteration != self.last_iteration + 1:
            raise ValueError('Credit iteration is pending, repeated or skipped')
        if not raws or any(t.get('split') != 'train' or t.get('iteration_id') != iteration for t in raws):
            raise ValueError('Only this iteration training rollouts can enter credit')
        cfg = self.config
        parsed = [parse_trace(raw, cfg) for raw in raws]
        traces = [p['trace'] for p in parsed]
        changed = [0]

        def transform(tables):
            if cfg.control == 'coverage_only':
                # Keep the original scale and learned search component.
                tables.ranks = {}
                tables.off_path = {}
            elif cfg.control == 'rank_shuffle':
                for skel, ranks in tables.ranks.items():
                    tables.ranks[skel], n = shuffle_ranks(ranks, cfg.control_seed, iteration)
                    changed[0] += n
            return tables

        def centered_control(values, members):
            if cfg.control != 'permute' or len(values) < 2:
                return values
            order = list(range(len(values)))
            random.Random(stable_hash([cfg.control_seed, iteration, members])).shuffle(order)
            return [values[i] for i in order]

        result = credit(traces, self.history, cfg, cfg, cfg.group_size, cfg.horizon,
                        table_transform=transform, centered_transform=centered_control)
        tables = result.pop('tables')
        self.table_snapshot = dict(ranks=tables.ranks, sizes=tables.sizes,
                                  off_path={k: sorted(v) for k, v in tables.off_path.items()}, search=tables.search)
        result['action_groups'] = {s['action_uid']: t.group_uid for t in traces for s in t.steps}
        if not cfg.enabled:
            for field in ('raw', 'deltas', 'centered'):
                result[field] = dict.fromkeys(result[field], 0.0)
        current = empty_stats()
        for t in traces:
            if not result['af'][t.group_uid]:
                add_trace_stats(current, t)
        donors = merge_stats(list(self.history) + [current])
        supported = total = 0
        for t in traces:
            if result['af'][t.group_uid]:
                total += len(t.states)
                supported += sum(s in tables.ranks.get(t.skeleton, {}) for s in t.states)
        adapter = Counter()
        for p in parsed:
            adapter.update(p['audit'])
        result['diagnostics'].update(
            donor_successes=sum(donors['successes'].values()), current_mixed_successes=sum(current['successes'].values()),
            history_iterations=len(self.history), rank_levels=sum(tables.sizes.values()),
            rank_labels_changed=changed[0], rank_shuffle_active=int(changed[0] > 0),
            rank_coverage_fraction=supported / total if total else 0.,
            unknown_state_fraction=result['diagnostics']['unknown_state_ratio'],
            nonzero_centered_actions=sum(v != 0 for v in result['centered'].values()),
            **{'adapter_' + k: v for k, v in adapter.items()})
        committed = empty_stats()
        for t in traces:
            add_trace_stats(committed, t)
        self.pending = iteration, json.loads(json.dumps(committed, allow_nan=False))
        return result

    def commit(self, iteration):
        if self.pending is None or self.pending[0] != iteration:
            raise ValueError('Cannot commit unmatched actor update')
        self.history.append(self.pending[1])
        self.last_iteration = iteration
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise ValueError('Cannot save pending credit')
        return dict(schema=SCHEMA, config=asdict(self.config), history=list(self.history), last_iteration=self.last_iteration)

    def load_state_dict(self, state):
        if state['schema'] != SCHEMA or state['config'] != asdict(self.config):
            raise ValueError('Method checkpoint contract mismatch')
        if len(state['history']) > self.config.window or state['last_iteration'] < len(state['history']):
            raise ValueError('Invalid donor history')
        self.history = deque(state['history'], maxlen=self.config.window)
        self.last_iteration = state['last_iteration']
        self.pending = None


def token_injections(result, uids, token_counts, config):
    if len(uids) != len(token_counts) or not all(math.isfinite(w) and w >= 0 for w in token_counts):
        raise ValueError('Invalid actor rows or token weights')
    vals = [result['centered'][u] for u in uids]
    selected = [(v, w) for u, v, w in zip(uids, vals, token_counts) if result['af'][result['action_groups'][u]]]
    total = sum(w for _, w in selected)
    mean = sum(v*w for v, w in selected) / total if total else 0.0
    sigma = math.sqrt(sum(w*(v-mean)**2 for v, w in selected) / total) if total else 0.0
    scale = config.beta / (sigma + config.epsilon) if config.enabled else 0.0
    raw = [v*scale for v in vals]
    delta = [max(-config.max_abs_injection, min(config.max_abs_injection, v)) for v in raw]
    af_mask = [result['af'][result['action_groups'][u]] for u in uids]
    af_mean = sum(w*d for w, d, active in zip(token_counts, delta, af_mask) if active) / total if total else 0.0
    af_squared = sum(w*d*d for w, d, active in zip(token_counts, delta, af_mask) if active)
    clipped_tokens = sum(w for w, v, active in zip(token_counts, raw, af_mask)
                         if active and abs(v) > config.max_abs_injection)
    return delta, dict(token_std_before_scale=sigma, scale=scale,
                       clipped_rows=sum(abs(v) > config.max_abs_injection for v in raw),
                       af_response_tokens=total,
                       af_nonzero_tokens=sum(w for w, d, active in zip(token_counts, delta, af_mask) if active and d != 0),
                       af_clipped_tokens=clipped_tokens,
                       af_clipped_token_fraction=clipped_tokens / total if total else 0.0,
                       af_weighted_mean=af_mean,
                       af_weighted_rms=math.sqrt(af_squared / total) if total else 0.0,
                       af_weighted_std=math.sqrt(sum(w*(d-af_mean)**2 for w, d, active in zip(token_counts, delta, af_mask)
                                                     if active) / total) if total else 0.0,
                       weighted_rms=math.sqrt(sum(w*d*d for w, d in zip(token_counts, delta)) / sum(token_counts)) if sum(token_counts) else 0.0)
