"""GRADUS-WebShop v1: CPU-only causal abstraction and historical ordinal credit.

Only instruction, pre-terminal observations, executed actions, and visible
clickables enter abstraction. Binary success labels are used for donor selection
and AF gating, never partial scores or the terminal page's hidden target dump.
"""
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import re

SCHEMA = 'gradus-webshop-v1'
JVM = '-Xmx128m -Xms16m -XX:+UseSerialGC'
NAV = {'back to search', '< prev', 'next >', 'description', 'features', 'reviews', 'attributes', 'buy now', 'search'}
STOP = set('a an the i me my want need looking for please find buy with and or of to in on is it that this should be has have price lower than dollars dollar under less'.split())


def norm(text):
    return ' '.join(str(text).casefold().split())


def parts(text):
    return [norm(x).strip("'\"") for x in str(text).split(' [SEP] ')]


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
            raise ValueError('v1 requires 15 actions, groups of 8 and bounded donor history')
        if self.min_success_support < 1 or not 0 < self.landmark_fraction <= 1:
            raise ValueError('Invalid donor support')
        if not 0 <= self.search_epsilon < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.lambda_aux <= 1:
            raise ValueError('Invalid potential/propagation parameters')
        if not all(math.isfinite(x) and x > 0 for x in (self.beta, self.epsilon, self.max_abs_injection)):
            raise ValueError('Invalid injection parameters')
        if self.control not in {'none', 'permute', 'coverage_only', 'rank_shuffle'}:
            raise ValueError('Unknown control')


def page_type(observation, previous='unknown', action=''):
    p = parts(observation)
    if 'buy now' in p:
        return 'item'
    if any(re.fullmatch(r'page \d+ \(total results: [\d,]+\)', x) for x in p):
        return 'results'
    if 'search' in p:
        return 'home'
    if action in {'click[description]', 'click[features]', 'click[reviews]', 'click[attributes]'}:
        return 'detail'
    return previous if previous == 'detail' else 'unknown'


def parse_trace(raw, config):
    steps = raw['steps']
    if not 1 <= len(steps) <= config.horizon:
        raise ValueError('Invalid trajectory length')
    if any(s['step_id'] != i for i, s in enumerate(steps)):
        raise ValueError('Noncontiguous original decisions')
    if any(s['environment_done'] for s in steps[:-1]):
        raise ValueError('A trajectory includes actions after termination')
    if bool(steps[-1]['won']) != bool(raw['success']):
        raise ValueError('Success disagrees with final active decision')
    if not (steps[-1]['environment_done'] or len(steps) == config.horizon):
        raise ValueError('Early collector truncation is not an all-failed episode')
    instruction = norm(raw['instruction'])
    words = set(re.findall(r'[a-z]+', instruction)) - STOP
    price_bound = re.search(r'price lower than (\d+(?:\.\d+)?) dollars', instruction)
    limit = float(price_bound[1]) if price_bound else None
    current_page = page_type(steps[0]['observation_before'])
    if current_page != 'home':
        raise ValueError('WebShop rollout must start at its reset search page')
    product = None
    title = ''
    price = None
    option_attempts = set()
    visited = set()
    states, coverage, noops = [], [0.0], []

    def key():
        # No product ID, goal ID, score, action index or trajectory length.
        title_words = set(re.findall(r'[a-z]+', title))
        overlap = len(words & title_words) / len(words) if words and title else 0.0
        overlap_bin = min(3, int(4 * overlap))
        # Upstream get_reward uses <= despite instruction wording "lower than".
        price_relation = 'unknown' if limit is None or price is None else ('within' if price <= limit else 'over')
        return json.dumps([current_page, overlap_bin, price_relation, min(2, len(option_attempts))], separators=(',', ':'))

    states.append(key())
    previous = steps[0]['observation_before']
    for step in steps:
        if step['observation_before'] != previous:
            raise ValueError('Discontinuous observation chain')
        # Match the environment's exact lowercased argument; collapsing spaces
        # would turn some invalid clicks into apparently accepted clicks.
        action = str(step['action']).lower()
        available = step['available_before']
        clickables = {str(x).lower() for x in available['clickables']}
        parsed = re.match(r'(.+)\[(.+)\]', action)  # exact upstream parse_action grammar
        click = (None, parsed[2]) if parsed and parsed[1] == 'click' else None
        search = (None, parsed[2]) if parsed and parsed[1] == 'search' else None
        action = f'{parsed[1]}[{parsed[2]}]' if parsed else action
        # Execution validity is distinct from response format validity.
        noop = not bool(search or (click and click[1] in clickables and click[1] != 'search'))
        noops.append(noop)
        if step['environment_done']:
            # Terminal text contains score and even hidden target fields in
            # upstream simple mode. Do not parse ANY of it, on success or failure.
            states.append(states[-1])
            coverage.append(coverage[-1])
            previous = step['observation_after']
            continue
        after_page = page_type(step['observation_after'], current_page, action)
        if search or (click and click[1] == 'back to search') or after_page in {'home', 'results'}:
            product, title, price = None, '', None
            option_attempts.clear()
        if current_page == 'results' and after_page == 'item' and click and click[1] in clickables:
            product = click[1]
            option_attempts.clear()
            visited.add(product)
        elif current_page == 'item' and click and not noop and click[1] not in NAV:
            # This is an accepted visible option attempt, not a claim that its
            # value is correct. Repeated clicks do not increase the count.
            option_attempts.add(click[1])
        current_page = after_page
        if current_page == 'item':
            fields = parts(step['observation_after'])
            for i, text in enumerate(fields):
                match = re.fullmatch(r'price:\s*\$?([\d,]+(?:\.\d+)?)', text)
                if match:
                    price = float(match[1].replace(',', ''))
                    title = fields[i - 1] if i else ''
                    break
        states.append(key())
        coverage.append(min(len(visited), config.horizon) / config.horizon)
        previous = step['observation_after']
    return dict(raw=raw, states=states, coverage=coverage, noops=noops)


def loop_erased(states):
    path = []
    for state in states:
        if state in path:
            path = path[:path.index(state) + 1]
        else:
            path.append(state)
    return path


def summarize(traces):
    order = {}
    successes = 0
    for t in traces:
        if not t['raw']['success']:
            continue
        successes += 1
        path = loop_erased(t['states'][:-1])
        for i, state in enumerate(path):
            total, count = order.get(state, (0.0, 0))
            order[state] = [total + i / len(path), count + 1]
    return dict(successes=successes, order=order)


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

    def ranks(self):
        count, sums = 0, {}
        for record in self.history:
            count += record['successes']
            for key, (total, support) in record['order'].items():
                old_total, old_support = sums.get(key, (0.0, 0))
                sums[key] = (old_total + total, old_support + support)
        threshold = max(self.config.min_success_support, math.ceil(self.config.landmark_fraction * count))
        values = {k: total / support for k, (total, support) in sums.items() if support >= threshold}
        # Equal mean positions are tied, never ordered lexicographically.
        levels = {v: i + 1 for i, v in enumerate(sorted(set(values.values())))}
        return {k: levels[v] for k, v in values.items()}, len(levels), count

    def prepare(self, raws, iteration):
        if self.pending is not None or iteration != self.last_iteration + 1:
            raise ValueError('Credit iteration is pending, repeated or skipped')
        if any(t.get('split') != 'train' or t.get('iteration_id') != iteration for t in raws):
            raise ValueError('Only this iteration training rollouts can enter credit')
        cfg = self.config
        groups = defaultdict(list)
        trajectory_ids, action_ids = set(), set()
        for raw in raws:
            if raw['trajectory_uid'] in trajectory_ids:
                raise ValueError('Duplicate original trajectory')
            trajectory_ids.add(raw['trajectory_uid'])
            groups[raw['group_uid']].append(raw)
            for step in raw['steps']:
                if step['action_uid'] in action_ids:
                    raise ValueError('Duplicate original action')
                action_ids.add(step['action_uid'])
        if not groups:
            raise ValueError('Empty collection')
        for group in groups.values():
            if len(group) != cfg.group_size or len({t['task_uid'] for t in group}) != 1:
                raise ValueError('Incomplete or inconsistent task group')
        traces = [parse_trace(raw, cfg) for raw in raws]
        af = {g: not any(t['success'] for t in group) for g, group in groups.items()}
        ranks, levels, donors = self.ranks()
        use_ranks = ranks if cfg.control != 'coverage_only' else {}
        changed = 0
        if cfg.control == 'rank_shuffle':
            use_ranks, changed = shuffle_ranks(ranks, cfg.control_seed, iteration)
        raw_credit = dict.fromkeys(action_ids, 0.0)
        centered = dict.fromkeys(action_ids, 0.0)
        action_groups = {}
        anchors = defaultdict(list)
        unknown = supported = total = 0
        for t in traces:
            raw = t['raw']
            steps = raw['steps']
            action_groups.update({s['action_uid']: raw['group_uid'] for s in steps})
            if not af[raw['group_uid']] or not cfg.enabled:
                continue
            component, phi = 0, []
            for state, cover in zip(t['states'], t['coverage']):
                total += 1
                supported += state in ranks
                if state in use_ranks:
                    component = use_ranks[state]
                else:
                    unknown += 1
                phi.append((component + cfg.search_epsilon * cover) / (levels + 1))
            tail = 0.0
            for i in reversed(range(len(steps))):
                delta = phi[i + 1] - phi[i]
                if t['noops'][i]:
                    delta = 0.0
                tail = delta + (0.0 if t['noops'][i] else cfg.gamma * cfg.lambda_aux * tail)
                uid = steps[i]['action_uid']
                raw_credit[uid] = tail
                anchors[(raw['group_uid'], t['states'][i])].append((uid, i))
        buckets = defaultdict(list)
        for (group, state), members in anchors.items():
            if len(members) >= 2:
                buckets[('state', group, state)] = [u for u, _ in members]
            else:
                uid, i = members[0]
                buckets[('time', group, i // 5)].append(uid)
        for key, uids in buckets.items():
            uids.sort()
            mean = math.fsum(raw_credit[u] for u in uids) / len(uids)
            values = [raw_credit[u] - mean for u in uids]
            if math.fsum(x*x for x in values) / len(values) <= 1e-12:
                values = [0.0] * len(uids)
            if cfg.control == 'permute' and len(uids) > 1:
                rng = random.Random(stable_hash([cfg.control_seed, iteration, key]))
                order = rng.sample(range(len(uids)), len(uids))
                values = [values[order[(order.index(i) + 1) % len(order)]] for i in range(len(uids))]
            centered.update(zip(uids, values))
        result = dict(centered=centered, raw=raw_credit, af=af, action_groups=action_groups,
                      diagnostics=dict(groups=len(groups), af_groups=sum(af.values()), donor_successes=donors,
                                       ranked_states=len(ranks), rank_levels=levels,
                                       rank_labels_changed=changed,
                                       rank_shuffle_active=int(changed > 0),
                                       rank_coverage_fraction=supported / total if total else 0.0,
                                       unknown_state_fraction=unknown / total if total else 0.0,
                                       nonzero_centered_actions=sum(v != 0 for v in centered.values()),
                                       history_iterations=len(self.history)))
        self.pending = (iteration, summarize(traces))
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
    scale = config.beta / (sigma + config.epsilon) if config.enabled and sigma > 0 else 0.0
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
